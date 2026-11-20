from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import nibabel as nib
import nilearn
import numpy as np
from nilearn.datasets import fetch_neurovault, fetch_neurovault_ids


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "pilot"
OUTPUT_DIR = ROOT / "outputs" / "pilot_diverse"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FILE = OUTPUT_DIR / "selected_ids.json"
N_IMAGES = 40

options = dict(
    data_dir=str(DATA_DIR),
    fetch_neurosynth_words=True,
    vectorize_words=True,
    resample=False,
    timeout=60,
    verbose=3,
)

# Dopo il primo download, riutilizza gli stessi ID.
if SELECTION_FILE.exists():
    image_ids = json.loads(SELECTION_FILE.read_text())
    dataset = fetch_neurovault_ids(image_ids=image_ids, **options)
else:
    dataset = fetch_neurovault(
        max_images=N_IMAGES,
        modality="fMRI-BOLD",
        map_type="Z map",
        **options,
    )

if not dataset.images:
    raise RuntimeError("Nessuna immagine scaricata.")

image_ids = [int(meta["id"]) for meta in dataset.images_meta]

if SELECTION_FILE.exists():
    missing = set(json.loads(SELECTION_FILE.read_text())) - set(image_ids)
    if missing:
        raise RuntimeError(f"Immagini richieste ma mancanti: {sorted(missing)}")
else:
    SELECTION_FILE.write_text(json.dumps(image_ids, indent=2))

if (
    dataset.get("vocabulary") is None
    or dataset.get("word_frequencies") is None
):
    raise RuntimeError(
        "Annotazioni non disponibili. Controllare i messaggi di download."
    )

terms = np.asarray(dataset.vocabulary)
weights = dataset.word_frequencies
if hasattr(weights, "toarray"):
    weights = weights.toarray()
weights = np.asarray(weights, dtype=float)

if weights.shape != (len(image_ids), len(terms)):
    raise RuntimeError(f"Dimensioni matrice inattese: {weights.shape}")

records = []

for index, (filename, meta) in enumerate(
    zip(dataset.images, dataset.images_meta)
):
    record = {
        "image_id": int(meta["id"]),
        "collection_id": meta.get("collection_id"),
        "name": meta.get("name"),
        "task": meta.get("task"),
        "paradigm": meta.get("cognitive_paradigm_cogatlas"),
        "analysis_level": meta.get("analysis_level"),
        "map_type": meta.get("map_type"),
        "modality": meta.get("modality"),
        "template": meta.get("target_template_image"),
        "brain_coverage_reported": meta.get("brain_coverage"),
        "perc_bad_voxels_reported": meta.get("perc_bad_voxels"),
        "file": str(Path(filename).resolve()),
    }

    reasons = []

    if meta.get("is_thresholded") is not False:
        reasons.append("soglia non confermata come assente")
    if meta.get("not_mni") is not False:
        reasons.append("spazio MNI non confermato")
    if meta.get("map_type") != "Z map":
        reasons.append("tipo diverso da Z map")
    if meta.get("modality") != "fMRI-BOLD":
        reasons.append("modalità diversa da fMRI-BOLD")

    try:
        img = nib.load(filename)
        values = img.get_fdata()
        finite = values[np.isfinite(values)]

        record.update({
            "shape": str(img.shape),
            "voxel_sizes": str(img.header.get_zooms()[:3]),
            "spatial_unit": img.header.get_xyzt_units()[0],
            "nonfinite_voxels": int(values.size - finite.size),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        })

        if values.ndim != 3:
            reasons.append("volume non 3D")
        if finite.size != values.size:
            reasons.append("voxel non finiti")
        if finite.size == 0 or np.ptp(finite) == 0:
            reasons.append("mappa vuota o costante")
        if (
            not np.isfinite(img.affine).all()
            or np.linalg.matrix_rank(img.affine[:3, :3]) < 3
        ):
            reasons.append("affine non valida")

    except Exception as exc:
        reasons.append(f"lettura fallita: {exc}")

    row = weights[index]
    annotation_ok = bool(np.isfinite(row).all() and np.any(row != 0))
    record["annotation_nonzero_terms"] = int(
        np.count_nonzero(np.isfinite(row) & (row != 0))
    )
    if not annotation_ok:
        reasons.append("annotazioni assenti o non utilizzabili")

    record["basic_checks_passed"] = not reasons
    record["issues"] = "; ".join(reasons)
    records.append(record)

# Registro di tutte le mappe, comprese quelle con problemi.
columns = list(dict.fromkeys(key for row in records for key in row))
with (OUTPUT_DIR / "manifest.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)

# Matrice completa: ogni riga conserva il proprio ID.
with (OUTPUT_DIR / "image_term_weights.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", *terms.tolist()])
    for image_id, row in zip(image_ids, weights):
        writer.writerow([image_id, *row.tolist()])

accepted_ids = [
    row["image_id"] for row in records if row["basic_checks_passed"]
]
(OUTPUT_DIR / "basic_pass_ids.json").write_text(
    json.dumps(accepted_ids, indent=2)
)

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "nilearn_version": nilearn.__version__,
            "requested_count": N_IMAGES,
            "images_meta": dataset.images_meta,
            "collections_meta": dataset.collections_meta,
        },
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)

print("\n--- RIEPILOGO ---")
print("Mappe scaricate:", len(records))
print("Mappe che superano i controlli di base:", len(accepted_ids))
print("Termini Neurosynth:", len(terms))
print("Matrice immagini-termini:", weights.shape)

print("\nMappe per collezione:")
print(Counter(str(row["collection_id"]) for row in records))

print("\nMappe per compito/paradigma:")
print(Counter(
    row["task"] or row["paradigm"] or "non specificato"
    for row in records
))

print("\nProblemi:")
for row in records:
    if row["issues"]:
        print(row["image_id"], ":", row["issues"])

print("\nRisultati:", OUTPUT_DIR)