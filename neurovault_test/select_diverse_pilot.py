from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urljoin
import json
import random

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "pilot_diverse"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES = 40
MAX_PER_COLLECTION = 5
MAX_PER_CONTRAST = 2
SEED = 42
MAX_PAGES = 100

SELECTION_FILE = OUTPUT_DIR / "selected_ids.json"

if SELECTION_FILE.exists():
    raise SystemExit(
        "La selezione esiste già. Usa build_diverse_pilot.py "
        "per scaricare gli stessi ID."
    )

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )),
)


def collection_id(meta):
    value = meta.get("collection_id")
    if value is None:
        value = str(meta.get("collection", "")).rstrip("/").split("/")[-1]
    return int(value)


def contrast_key(meta):
    # Se il contrasto non è specificato, raggruppa conservativamente
    # usando il compito/paradigma. Non usare il nome del soggetto.
    for field in [
        "contrast_definition",
        "task",
        "cognitive_contrast_cogatlas",
        "cognitive_paradigm_cogatlas",
    ]:
        value = meta.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.lower().split())
    return "non_specificato"


def eligible(meta):
    return (
        int(meta["id"]) != 56803
        and meta.get("map_type") == "Z map"
        and meta.get("modality") == "fMRI-BOLD"
        and meta.get("is_thresholded") is False
        and meta.get("not_mni") is False
        and meta.get("is_valid") is True
        and meta.get("image_type") != "atlas"
    )


def choose_sample(candidates):
    rng = random.Random(SEED)
    groups = defaultdict(list)

    # Ordine iniziale stabile.
    for meta in sorted(candidates, key=lambda item: int(item["id"])):
        groups[collection_id(meta)].append(meta)

    collections = sorted(groups)
    rng.shuffle(collections)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    per_collection = Counter()
    per_contrast = Counter()

    # Un'immagine per collezione a ogni giro, finché possibile.
    while len(selected) < N_IMAGES:
        added = False

        for cid in collections:
            if per_collection[cid] >= MAX_PER_COLLECTION:
                continue

            for index, meta in enumerate(groups[cid]):
                key = (cid, contrast_key(meta))
                if per_contrast[key] >= MAX_PER_CONTRAST:
                    continue

                selected.append(groups[cid].pop(index))
                per_collection[cid] += 1
                per_contrast[key] += 1
                added = True
                break

            if len(selected) == N_IMAGES:
                break

        if not added:
            break

    return selected


# Esamina un insieme limitato di metadati, senza scaricare immagini.
url = "https://neurovault.org/api/images/?limit=100"
candidates = {}
visited = set()

for page_number in range(1, MAX_PAGES + 1):
    if not url or url in visited:
        break
    visited.add(url)

    response = session.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        batch = payload
        next_url = None
    else:
        batch = payload["results"]
        next_url = payload.get("next")

    for meta in batch:
        if eligible(meta):
            try:
                collection_id(meta)
            except (TypeError, ValueError):
                continue
            candidates[int(meta["id"])] = meta

    pool = list(candidates.values())
    n_collections = len({collection_id(meta) for meta in pool})
    selected = choose_sample(pool)

    print(
        f"Pagina {page_number}: {len(pool)} candidati, "
        f"{n_collections} collezioni, "
        f"{len(selected)} selezionabili",
        flush=True,
    )

    # Cerca più collezioni del minimo necessario.
    if len(selected) == N_IMAGES and n_collections >= 20:
        break

    url = urljoin(url, next_url) if next_url else None

pool = list(candidates.values())
selected = choose_sample(pool)

# Salva il gruppo esaminato anche se non basta a completare il campione.
(OUTPUT_DIR / "candidate_metadata.json").write_text(
    json.dumps(pool, indent=2), encoding="utf-8"
)

if len(selected) < N_IMAGES:
    raise SystemExit(
        f"Trovate solo {len(selected)} mappe compatibili con i limiti. "
        "Candidati salvati; nessuna selezione definitiva creata. "
        "Non vengono allentati automaticamente i criteri."
    )

ids = [int(meta["id"]) for meta in selected]

(OUTPUT_DIR / "selection_metadata.json").write_text(
    json.dumps(
        {
            "seed": SEED,
            "max_per_collection": MAX_PER_COLLECTION,
            "max_per_contrast": MAX_PER_CONTRAST,
            "candidate_count": len(pool),
            "selected_images": selected,
        },
        indent=2,
    ),
    encoding="utf-8",
)
SELECTION_FILE.write_text(json.dumps(ids, indent=2), encoding="utf-8")

print("\n--- SELEZIONE ---")
print("Mappe:", len(ids))
print("Per collezione:")
print(Counter(collection_id(meta) for meta in selected))
print("\nPer collezione e contrasto:")
print(Counter(
    (collection_id(meta), contrast_key(meta)) for meta in selected
))
print("\nID salvati in:", SELECTION_FILE)