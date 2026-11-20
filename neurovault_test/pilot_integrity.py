"""Controlli di provenienza del pilota, senza dipendenze scientifiche."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json


def unique_ids(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label}: serve un elenco non vuoto di ID.")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{label}: gli ID devono essere interi positivi.")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: ID duplicati.")
    return values


def require_exact_ids(expected, actual, label):
    unique_ids(expected, "ID richiesti")
    unique_ids(actual, label)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"{label}: mancanti={missing}; aggiuntivi={extra}.")


def collection_id(meta):
    value = meta.get("collection_id")
    if value is None:
        value = str(meta.get("collection", "")).rstrip("/").split("/")[-1]
    return int(value)


def contrast_key(meta):
    for field in ("contrast_definition", "task", "cognitive_contrast_cogatlas",
                  "cognitive_paradigm_cogatlas"):
        value = meta.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.lower().split())
    return "non_specificato"


def selection_hash(ids):
    unique_ids(ids, "Selezione")
    return sha256(json.dumps(sorted(ids)).encode("utf-8")).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_selection(ids, snapshot):
    unique_ids(ids, "Selezione")
    maps = snapshot["selected_images"]
    require_exact_ids(ids, [int(meta["id"]) for meta in maps], "Metadati selezione")
    # I limiti non possono essere allentati cambiando soltanto lo snapshot.
    collection_limit = snapshot.get("max_per_collection", 5)
    contrast_limit = snapshot.get("max_per_contrast", 2)
    if type(collection_limit) is not int or not 1 <= collection_limit <= 5:
        raise ValueError("Limite per collezione non valido (massimo 5).")
    if type(contrast_limit) is not int or not 1 <= contrast_limit <= 2:
        raise ValueError("Limite per contrasto non valido (massimo 2).")
    counts = Counter(collection_id(meta) for meta in maps)
    contrasts = Counter((collection_id(meta), contrast_key(meta)) for meta in maps)
    if max(counts.values()) > collection_limit:
        raise ValueError("La selezione supera il limite per collezione.")
    if max(contrasts.values()) > contrast_limit:
        raise ValueError("La selezione supera il limite per contrasto.")
    return snapshot


def load_selection(directory):
    directory = Path(directory)
    ids_path = directory / "selected_ids.json"
    snapshot_path = directory / "selection_metadata.json"
    if not ids_path.exists() or not snapshot_path.exists():
        raise FileNotFoundError(
            "Selezione obbligatoria: eseguire select_diverse_pilot.py, oppure "
            "select_diverse_pilot.py --import-legacy. Nessuna ricerca generica avviata."
        )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    validate_selection(ids, snapshot)
    return ids, snapshot


def import_legacy_selection(source, destination):
    """Recupera lo snapshot originale; non si fida degli ID legacy riscritti."""
    source, destination = Path(source), Path(destination)
    snapshot = json.loads((source / "selection_metadata.json").read_text(encoding="utf-8"))
    ids = [int(meta["id"]) for meta in snapshot["selected_images"]]
    validate_selection(ids, snapshot)
    if any((destination / name).exists() for name in
           ("selected_ids.json", "selection_metadata.json")):
        raise FileExistsError("La selezione di destinazione esiste già; non sovrascritta.")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "selection_metadata.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    (destination / "selected_ids.json").write_text(
        json.dumps(ids, indent=2), encoding="utf-8"
    )
    return ids


def verify_preparation_inputs(input_dir, selection_dir, manifest_ids, accepted_ids):
    """Fallisce prima del ricampionamento se i file appartengono a run diversi."""
    input_dir = Path(input_dir)
    requested, _ = load_selection(selection_dir)
    require_exact_ids(requested, manifest_ids, "Manifest")
    unique_ids(accepted_ids, "ID accettati")
    if not set(accepted_ids).issubset(requested):
        raise ValueError("Gli ID accettati non appartengono alla selezione.")
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("selection_sha256") != selection_hash(requested):
        raise ValueError("Provenienza assente o diversa: rieseguire il builder corretto.")
    if metadata.get("selection_metadata_sha256") != file_hash(
        Path(selection_dir) / "selection_metadata.json"
    ):
        raise ValueError("Lo snapshot della selezione è cambiato dopo il download.")
    require_exact_ids(requested, metadata["requested_ids"], "Richiesta registrata")
    require_exact_ids(requested, [int(m["id"]) for m in metadata["images_meta"]],
                      "Metadati del download")
    for name in ("manifest.csv", "image_term_weights.csv", "basic_pass_ids.json"):
        if metadata.get("artifact_sha256", {}).get(name) != file_hash(input_dir / name):
            raise ValueError(f"{name}: file cambiato o provenienza non verificabile.")
    return selection_hash(requested)
