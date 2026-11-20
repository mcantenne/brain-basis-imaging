from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # Salva figure anche senza desktop grafico.

import nibabel as nib
import numpy as np
from nilearn import datasets, plotting


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Due immagini usate nell'esempio ufficiale di Nilearn.
IMAGE_IDS = [32015, 32016]

dataset = datasets.fetch_neurovault_ids(
    image_ids=IMAGE_IDS,
    data_dir=str(DATA_DIR),
    fetch_neurosynth_words=False,
    resample=False,
    timeout=60,
)

received_ids = {int(meta["id"]) for meta in dataset.images_meta}
missing_ids = set(IMAGE_IDS) - received_ids
if missing_ids:
    raise RuntimeError(f"Immagini non scaricate: {sorted(missing_ids)}")

for filename, metadata in zip(dataset.images, dataset.images_meta):
    image_id = metadata["id"]
    img = nib.load(filename)
    values = img.get_fdata()
    finite_values = values[np.isfinite(values)]

    print(f"\n--- NeuroVault ID {image_id} ---")
    print("Nome:", metadata.get("name"))
    print("Tipo mappa:", metadata.get("map_type"))
    print("Soglia applicata:", metadata.get("is_thresholded"))
    print("Non in MNI:", metadata.get("not_mni"))
    print("File:", filename)
    print("Dimensioni:", img.shape)
    print("Dimensioni voxel:", img.header.get_zooms())
    print("Unità:", img.header.get_xyzt_units())
    print("Affine:\n", img.affine)
    print("Valori non finiti:", values.size - finite_values.size)

    if values.ndim != 3 or finite_values.size == 0:
        raise ValueError(f"L'immagine {image_id} non è un volume 3D valido.")

    print("Min/max:", finite_values.min(), finite_values.max())

    metadata_path = OUTPUT_DIR / f"map_{image_id}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    display = plotting.plot_stat_map(
        img,
        bg_img=None,
        display_mode="ortho",
        threshold=None,
        title=f"NeuroVault {image_id}",
    )
    display.savefig(str(OUTPUT_DIR / f"map_{image_id}.png"))
    display.close()

print("\nTest completato. Risultati in:", OUTPUT_DIR)