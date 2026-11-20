from pathlib import Path
import csv
import json
import warnings

import nilearn
import numpy as np
from nilearn.datasets import fetch_neurovault_ids


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "neurosynth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_IDS = [32015, 32016]

print("Nilearn:", nilearn.__version__)

# Conserva anche gli avvisi: il download delle immagini potrebbe
# riuscire mentre quello delle annotazioni fallisce.
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always")

    try:
        dataset = fetch_neurovault_ids(
            image_ids=IMAGE_IDS,
            data_dir=str(ROOT / "data" / "neurosynth_probe"),
            fetch_neurosynth_words=True,
            vectorize_words=True,
            resample=False,
            timeout=60,
            verbose=3,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        (OUTPUT_DIR / "error.txt").write_text(
            message, encoding="utf-8"
        )
        raise

warning_messages = [str(item.message) for item in recorded]
(OUTPUT_DIR / "warnings.json").write_text(
    json.dumps(warning_messages, indent=2),
    encoding="utf-8",
)

for message in warning_messages:
    print("AVVISO:", message)

print("\nCampi restituiti:", list(dataset.keys()))

if (
    dataset.get("vocabulary") is None
    or dataset.get("word_frequencies") is None
):
    raise RuntimeError(
        "Il servizio non ha restituito vocabolario e pesi. "
        "Controllare gli avvisi e il log del download."
    )

terms = np.asarray(dataset.vocabulary)
matrix = dataset.word_frequencies

# Gestisce sia matrici sparse sia array NumPy.
if hasattr(matrix, "toarray"):
    matrix = matrix.toarray()
matrix = np.asarray(matrix, dtype=float)

if matrix.shape != (len(dataset.images_meta), len(terms)):
    raise RuntimeError(f"Dimensioni inattese: {matrix.shape}")

received_ids = [int(meta["id"]) for meta in dataset.images_meta]
missing_ids = set(IMAGE_IDS) - set(received_ids)
if missing_ids:
    raise RuntimeError(f"Immagini mancanti: {sorted(missing_ids)}")

print("ID nell'ordine delle righe:", received_ids)
print("Numero termini:", len(terms))
print("Dimensioni matrice:", matrix.shape)

# Salva la matrice completa con etichette esplicite.
with (OUTPUT_DIR / "image_term_weights.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", *terms.tolist()])
    for image_id, row in zip(received_ids, matrix):
        writer.writerow([image_id, *row.tolist()])

# Salva metadati e versione del software.
(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(
        {
            "nilearn_version": nilearn.__version__,
            "requested_ids": IMAGE_IDS,
            "images_meta": dataset.images_meta,
        },
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)

for image_id, row in zip(received_ids, matrix):
    usable = np.isfinite(row) & (row != 0)
    indices = np.flatnonzero(usable)
    indices = indices[np.argsort(row[indices])[::-1]]

    print(f"\n--- Immagine {image_id} ---")
    print("Pesi finiti non nulli:", len(indices))
    print("Pesi non finiti:", np.count_nonzero(~np.isfinite(row)))

    if not len(indices):
        print("Nessuna annotazione utilizzabile: non equivale "
              "ad assenza di associazioni cognitive.")
        continue

    print("Primi 15 termini per peso decrescente:")
    for index in indices[:15]:
        print(f"  {terms[index]:30s} {row[index]: .6f}")

print("\nRisultati salvati in:", OUTPUT_DIR)