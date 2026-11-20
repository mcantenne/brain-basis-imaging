from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from neuromaps import datasets, transforms
from nilearn import plotting


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "surface"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_ID = 32015
DENSITY = "10k"

filename = (
    ROOT / "data" / "neurovault" / "collection_1952"
    / f"image_{IMAGE_ID}.nii.gz"
)

if not filename.exists():
    raise FileNotFoundError(filename)

# 1. Leggi la mappa volumetrica originale.
img = nib.load(filename)
volume = img.get_fdata()

if volume.ndim != 3:
    raise ValueError("Serve una mappa volumetrica 3D.")

# 2. Aggiungi un margine senza dati e aggiorna l'affine.
# I voxel originali mantengono le stesse coordinate fisiche.
padding = 10
padded_volume = np.pad(
    volume,
    pad_width=padding,
    mode="constant",
    constant_values=np.nan,
)

padded_affine = img.affine.copy()
padded_affine[:3, 3] = nib.affines.apply_affine(
    img.affine, [-padding, -padding, -padding]
)

padded_img = nib.Nifti1Image(padded_volume, padded_affine)

# 3. Proietta tramite le corrispondenze MNI -> fsaverage.
# Non applichiamo soglie ai dati.
left_map, right_map = transforms.mni152_to_fsaverage(
    padded_img,
    fsavg_density=DENSITY,
    method="linear",
)

# 4. Carica superfici e maschere corticali corrispondenti.
atlas = datasets.fetch_fsaverage(
    density=DENSITY,
    data_dir=str(ROOT / "data" / "neuromaps"),
)

for index, (hemi, surface_map) in enumerate(
    [("left", left_map), ("right", right_map)]
):
    values = np.asarray(surface_map.agg_data()).copy()

    # La maschera esclude la parete mediale non corticale.
    cortex = (
        np.asarray(nib.load(atlas["medial"][index]).agg_data()) > 0
    )
    valid = cortex & np.isfinite(values)

    print(f"\nEmisfero: {hemi}")
    print("Vertici totali:", values.size)
    print("Vertici corticali:", cortex.sum())
    print("Vertici corticali con valori finiti:", valid.sum())
    print("Vertici corticali senza dati:", (cortex & ~valid).sum())

    if not valid.any():
        raise RuntimeError(f"Nessun dato corticale valido: {hemi}")

    print("Min/max corticali:", values[valid].min(), values[valid].max())

    # Conserva l'output grezzo della trasformazione.
    nib.save(
        surface_map,
        str(OUTPUT_DIR / f"map_{IMAGE_ID}_{hemi}.func.gii"),
    )

    # Versione per l'analisi: parete mediale e dati mancanti = NaN.
    values[~valid] = np.nan
    np.save(OUTPUT_DIR / f"map_{IMAGE_ID}_{hemi}.npy", values)
    np.save(OUTPUT_DIR / f"valid_{IMAGE_ID}_{hemi}.npy", valid)

    # La superficie gonfiata serve solo per vedere meglio i solchi.
    for view in ["lateral", "medial"]:
        figure = plotting.plot_surf_stat_map(
            surf_mesh=str(atlas["inflated"][index]),
            stat_map=values,
            hemi=hemi,
            view=view,
            engine="matplotlib",
            threshold=3,
            cmap="RdBu_r",
            symmetric_cbar=True,
            vmax=13,
            colorbar=True,
            title=f"NeuroVault {IMAGE_ID} - {hemi}, {view}",
        )
        figure.savefig(
            OUTPUT_DIR / f"map_{IMAGE_ID}_{hemi}_{view}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)

print("\nRisultati salvati in:", OUTPUT_DIR)