from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from nilearn import datasets, image, plotting
from pilot_integrity import verify_preparation_inputs, require_exact_ids, file_hash


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs" / "pilot_diverse"
OUTPUT_DIR = INPUT_DIR / "prepared"
VOLUME_DIR = OUTPUT_DIR / "volumes"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [OUTPUT_DIR, VOLUME_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 1. Recupera le mappe che hanno superato i controlli precedenti.
manifest = pd.read_csv(INPUT_DIR / "manifest.csv", dtype=str)
accepted_list = json.loads((INPUT_DIR / "basic_pass_ids.json").read_text())
selection_sha256 = verify_preparation_inputs(
    INPUT_DIR, ROOT.parent / "selections" / "pilot_diverse",
    manifest["image_id"].astype(int).tolist(), accepted_list,
)
accepted_ids = set(accepted_list)

manifest["image_id"] = manifest["image_id"].astype(int)
manifest = manifest[
    manifest["image_id"].isin(accepted_ids)
].sort_values("image_id").reset_index(drop=True)

if manifest.empty:
    raise RuntimeError("Nessuna immagine disponibile.")
if manifest["image_id"].duplicated().any():
    raise RuntimeError("ID duplicati nel manifest.")
if set(manifest["image_id"]) != accepted_ids:
    raise RuntimeError("Il manifest non contiene tutti gli ID accettati.")

# Esclusioni specifiche di questa analisi; gli originali restano disponibili.
exclusions_path = ROOT.parent / "selections" / "pilot_diverse" / "exclusions.json"
exclusion_reasons = json.loads(exclusions_path.read_text(encoding="utf-8"))
excluded_ids = {int(value) for value in exclusion_reasons}

excluded = manifest[
    manifest["image_id"].isin(excluded_ids)
].copy()
excluded["exclusion_reason"] = excluded["image_id"].map(
    lambda value: exclusion_reasons[str(value)]
)

manifest = manifest[
    ~manifest["image_id"].isin(excluded_ids)
].reset_index(drop=True)

if manifest.empty:
    raise RuntimeError("Tutte le immagini sono state escluse.")

# Salva la nuova preparazione separatamente dalla precedente.
OUTPUT_DIR = INPUT_DIR / "prepared_fullcoverage"
VOLUME_DIR = OUTPUT_DIR / "volumes"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [OUTPUT_DIR, VOLUME_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

excluded.to_csv(OUTPUT_DIR / "excluded_images.csv", index=False)

# 2. Definisci il riferimento anatomico e la griglia a 3 mm.
template = datasets.load_mni152_template(resolution=3)
brain_img = datasets.load_mni152_brain_mask(resolution=3)

brain_img = image.resample_to_img(
    brain_img,
    template,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)
brain_mask = brain_img.get_fdata() > 0

nib.save(template, str(OUTPUT_DIR / "reference_mni_3mm.nii.gz"))
nib.save(brain_img, str(OUTPUT_DIR / "reference_brain_mask.nii.gz"))

# Coordinate fisiche dei voxel cerebrali del riferimento.
brain_indices = np.column_stack(np.where(brain_mask))
brain_world = nib.affines.apply_affine(template.affine, brain_indices)

common_mask = brain_mask.copy()
coverage_count = np.zeros(template.shape, dtype=np.uint16)
resampled_paths = []
qc_records = []

# 3. Ricampiona ciascuna immagine e misura il campo di vista.
for position, row in manifest.iterrows():
    image_id = int(row["image_id"])
    source = nib.load(row["file"])
    source_values = source.get_fdata()

    if source_values.ndim != 3:
        raise ValueError(f"{image_id}: il volume non è 3D.")
    if not np.isfinite(source_values).all():
        raise ValueError(f"{image_id}: contiene valori non finiti.")

    # Verifica geometrica: quali voxel del riferimento cadono
    # all'interno della griglia del volume originale?
    source_coords = nib.affines.apply_affine(
        np.linalg.inv(source.affine), brain_world
    )
    inside = np.all(
        (source_coords >= 0)
        & (source_coords <= np.asarray(source.shape) - 1),
        axis=1,
    )

    geometric_mask = np.zeros(template.shape, dtype=bool)
    geometric_mask[brain_mask] = inside

    resampled = image.resample_to_img(
        source,
        template,
        interpolation="linear",
        fill_value=0,
        force_resample=True,
        copy_header=True,
    )

    values = resampled.get_fdata(dtype=np.float32)
    usable = geometric_mask & np.isfinite(values)

    common_mask &= usable
    coverage_count += usable.astype(np.uint16)

    # All'esterno del dominio utilizzabile salviamo NaN,
    # evitando di confonderlo con uno zero originale.
    values[~usable] = np.nan
    saved_img = image.new_img_like(template, values)

    output_path = VOLUME_DIR / f"image_{image_id}_mni3mm.nii.gz"
    nib.save(saved_img, str(output_path))
    resampled_paths.append(output_path)

    valid_values = values[usable]
    coverage = usable.sum() / brain_mask.sum()
    zero_fraction = (
        np.mean(valid_values == 0) if valid_values.size else np.nan
    )

    qc_records.append({
        "image_id": image_id,
        "original_shape": str(source.shape),
        "original_voxel_sizes": str(source.header.get_zooms()[:3]),
        "original_spatial_unit": source.header.get_xyzt_units()[0],
        "geometric_coverage_fraction": float(coverage),
        "zero_fraction_in_covered_brain": float(zero_fraction),
        "resampled_file": str(output_path.resolve()),
    })

    # Solo per il disegno: sostituisci i NaN con zero.
    # Il file salvato e la futura matrice non vengono modificati.
    plot_values = np.where(usable, values, 0)
    plot_img = image.new_img_like(template, plot_values)

    display = plotting.plot_stat_map(
        plot_img,
        bg_img=template,
        display_mode="z",
        cut_coords=[-30, -15, 0, 15, 30, 45, 60],
        threshold=3,
        cmap="RdBu_r",
        symmetric_cbar=True,
        title=f"ID {image_id} - controllo MNI",
    )
    display.savefig(str(FIGURE_DIR / f"alignment_{image_id}.png"))
    display.close()

    print(
        f"[{position + 1}/{len(manifest)}] ID {image_id}: "
        f"copertura geometrica {coverage:.1%}, "
        f"zeri {zero_fraction:.1%}",
        flush=True,
    )

# 4. Salva il controllo di copertura e la maschera comune.
pd.DataFrame(qc_records).to_csv(
    OUTPUT_DIR / "coverage_qc.csv", index=False
)

common_img = image.new_img_like(
    template, common_mask.astype(np.uint8)
)
nib.save(common_img, str(OUTPUT_DIR / "common_mask.nii.gz"))

coverage_img = image.new_img_like(
    template, coverage_count.astype(np.float32) / len(manifest)
)
nib.save(coverage_img, str(OUTPUT_DIR / "geometric_coverage_fraction.nii.gz"))

if not common_mask.any():
    raise RuntimeError(
        "Intersezione vuota: esaminare coverage_qc.csv prima di proseguire."
    )

display = plotting.plot_roi(
    common_img,
    bg_img=template,
    display_mode="z",
    cut_coords=[-30, -15, 0, 15, 30, 45, 60],
    title="Maschera comune provvisoria",
)
display.savefig(str(FIGURE_DIR / "common_mask.png"))
display.close()

# 5. Costruisci X: righe = immagini, colonne = voxel comuni.
X = np.stack([
    nib.load(str(path)).get_fdata(dtype=np.float32)[common_mask]
    for path in resampled_paths
])

if not np.isfinite(X).all():
    raise RuntimeError("La matrice contiene valori non finiti.")

ids = manifest["image_id"].to_numpy(dtype=int)

# Ordine dei voxel: lo stesso usato da values[common_mask].
np.save(OUTPUT_DIR / "voxel_indices.npy",
        np.column_stack(np.where(common_mask)))
np.save(OUTPUT_DIR / "image_ids.npy", ids)

# 6. Allinea le annotazioni allo stesso ordine delle immagini.
annotations = pd.read_csv(INPUT_DIR / "image_term_weights.csv")
if annotations["image_id"].duplicated().any():
    raise RuntimeError("ID duplicati nella matrice delle annotazioni.")

require_exact_ids(
    pd.read_csv(INPUT_DIR / "manifest.csv")["image_id"].astype(int).tolist(),
    annotations["image_id"].astype(int).tolist(),
    "ID annotazioni",
)
annotations = annotations.set_index("image_id").loc[ids]
Y = annotations.to_numpy(dtype=float)

if not np.isfinite(Y).all():
    raise RuntimeError("Annotazioni mancanti o non finite.")

annotations.to_csv(OUTPUT_DIR / "aligned_image_term_weights.csv")
np.save(OUTPUT_DIR / "Y_terms.npy", Y)
np.save(OUTPUT_DIR / "X_voxels_raw.npy", X)

summary = {
    "nilearn_version": nilearn.__version__,
    "selection_sha256": selection_sha256,
    "input_metadata_sha256": file_hash(INPUT_DIR / "metadata.json"),
    "exclusions_sha256": file_hash(exclusions_path),
    "excluded_ids": excluded["image_id"].astype(int).tolist(),
    "image_ids": ids.tolist(),
    "reference_shape": list(template.shape),
    "reference_affine": template.affine.tolist(),
    "reference_brain_voxels": int(brain_mask.sum()),
    "common_voxels": int(common_mask.sum()),
    "common_fraction": float(common_mask.sum() / brain_mask.sum()),
    "X_shape": list(X.shape),
    "Y_shape": list(Y.shape),
    "constant_voxel_columns": int(np.count_nonzero(np.ptp(X, axis=0) == 0)),
    "normalization": "none",
    "mask_status": "provisional_pending_visual_review",
}
(OUTPUT_DIR / "preparation_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\n--- RIEPILOGO PREPARAZIONE ---")
print("Immagini:", len(ids))
print("Griglia comune:", template.shape)
print("Voxel della maschera cerebrale:", brain_mask.sum())
print("Voxel comuni:", common_mask.sum())
print("Frazione comune:", f"{summary['common_fraction']:.1%}")
print("Matrice immagini-voxel:", X.shape)
print("Matrice immagini-termini:", Y.shape)
print("Colonne voxel costanti:", summary["constant_voxel_columns"])
print("Normalizzazione: nessuna")
print("Risultati:", OUTPUT_DIR)

