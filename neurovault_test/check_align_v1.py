from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from nilearn import datasets, plotting

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Riferimento anatomico per un controllo preliminare.
template = datasets.load_mni152_template(resolution=2)

for image_id in [32015, 32016]:
    filename = (
        ROOT / "data" / "neurovault" / "collection_1952"
        / f"image_{image_id}.nii.gz"
    )

    if not filename.exists():
        raise FileNotFoundError(filename)

    display = plotting.plot_stat_map(
        str(filename),
        bg_img=template,
        display_mode="z",
        cut_coords=[-30, -15, 0, 15, 30, 45, 60],
        threshold=3,
        cmap="RdBu_r",
        symmetric_cbar=True,
        title=f"NeuroVault {image_id} su riferimento MNI",
    )

    output = OUTPUT_DIR / f"alignment_{image_id}.png"
    display.savefig(str(output))
    display.close()
    print("Salvato:", output)