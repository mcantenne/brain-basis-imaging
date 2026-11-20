"""Regressioni offline: nessun download o output nel progetto reale."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import contextlib
import csv
import io
import json
import runpy
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "neurovault_test"))
from pilot_integrity import (
    file_hash, import_legacy_selection, load_selection, require_exact_ids,
    selection_hash, validate_selection, verify_preparation_inputs,
)


def snapshot(ids=(1, 2)):
    return {
        "max_per_collection": 5,
        "max_per_contrast": 2,
        "selected_images": [
            {"id": value, "collection_id": 10, "task": f"task-{value}"}
            for value in ids
        ],
    }


def write_selection(directory, ids=(1, 2)):
    directory.mkdir(parents=True)
    (directory / "selected_ids.json").write_text(json.dumps(list(ids)))
    (directory / "selection_metadata.json").write_text(json.dumps(snapshot(ids)))


class IntegrityTests(unittest.TestCase):
    def test_exact_ids_allow_different_order(self):
        require_exact_ids([1, 2], [2, 1], "download")

    def test_exact_ids_reject_extra_missing_and_duplicates(self):
        for actual in ([1, 2, 3], [1], [1, 1, 2]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                require_exact_ids([1, 2], actual, "download")

    def test_selection_rejects_invalid_ids(self):
        for ids in ([], [True], [1.2], ["1"], [-1], [1, 1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                selection_hash(ids)

    def test_fingerprint_depends_on_set_not_order(self):
        self.assertEqual(selection_hash([1, 2]), selection_hash([2, 1]))
        self.assertNotEqual(selection_hash([1, 2]), selection_hash([1, 3]))

    def test_missing_selection_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_selection(directory)

    def test_snapshot_must_match_selected_ids(self):
        with self.assertRaises(ValueError):
            validate_selection([1, 3], snapshot())

    def test_rejects_concentrated_collection(self):
        with self.assertRaisesRegex(ValueError, "collezione"):
            validate_selection(list(range(1, 7)), snapshot(range(1, 7)))

    def test_rejects_concentrated_contrast(self):
        data = snapshot([1, 2, 3])
        for meta in data["selected_images"]:
            meta["task"] = "same task"
        with self.assertRaisesRegex(ValueError, "contrasto"):
            validate_selection([1, 2, 3], data)

    def test_legacy_import_uses_snapshot_not_rewritten_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy"
            destination = Path(directory) / "selection"
            write_selection(source)
            (source / "selected_ids.json").write_text("[99]")
            self.assertEqual(import_legacy_selection(source, destination), [1, 2])
            self.assertEqual(load_selection(destination)[0], [1, 2])
            with self.assertRaises(FileExistsError):
                import_legacy_selection(source, destination)

    def test_preparation_checks_artifact_and_snapshot_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection"
            output = Path(directory) / "output"
            write_selection(selection)
            output.mkdir()
            for name in ("manifest.csv", "image_term_weights.csv", "basic_pass_ids.json"):
                (output / name).write_text("original")
            metadata = {
                "selection_sha256": selection_hash([1, 2]),
                "selection_metadata_sha256": file_hash(selection / "selection_metadata.json"),
                "requested_ids": [1, 2],
                "images_meta": snapshot()["selected_images"],
                "artifact_sha256": {name.name: file_hash(name) for name in output.iterdir()},
            }
            (output / "metadata.json").write_text(json.dumps(metadata))
            verify_preparation_inputs(output, selection, [1, 2], [2])
            with self.assertRaises(ValueError):
                verify_preparation_inputs(output, selection, [1, 99], [1])
            (output / "manifest.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "manifest.csv"):
                verify_preparation_inputs(output, selection, [1, 2], [1, 2])
            (output / "manifest.csv").write_text("original")
            (selection / "selection_metadata.json").write_text(
                json.dumps(snapshot(), indent=2)
            )
            with self.assertRaisesRegex(ValueError, "snapshot"):
                verify_preparation_inputs(output, selection, [1, 2], [1, 2])


class BuilderTests(unittest.TestCase):
    def run_builder(self, directory, returned_ids, with_selection=True):
        root = Path(directory)
        scripts = root / "neurovault_test"
        scripts.mkdir()
        script = scripts / "build_diverse_pilot_v1.py"
        shutil.copyfile(ROOT / "neurovault_test" / script.name, script)
        selection = root / "selections" / "pilot_diverse"
        if with_selection:
            write_selection(selection)

        class Bunch(dict):
            __getattr__ = dict.__getitem__

        metas = []
        for image_id in returned_ids:
            meta = snapshot([image_id])["selected_images"][0]
            meta.update(is_thresholded=False, not_mni=False,
                        map_type="Z map", modality="fMRI-BOLD")
            metas.append(meta)
        dataset = Bunch(
            images=[f"image_{value}.nii.gz" for value in returned_ids],
            images_meta=metas, collections_meta=[], vocabulary=["term"],
            word_frequencies=np.asarray(returned_ids, dtype=float).reshape(-1, 1),
        )
        fake_datasets = ModuleType("nilearn.datasets")
        from unittest.mock import Mock
        fetch = Mock(return_value=dataset)
        fake_datasets.fetch_neurovault_ids = fetch
        fake_nilearn = ModuleType("nilearn")
        fake_nilearn.__version__ = "offline-test"
        fake_nilearn.datasets = fake_datasets
        fake_nib = ModuleType("nibabel")
        fake_nib.load = lambda path: SimpleNamespace(
            get_fdata=lambda: np.arange(8, dtype=float).reshape(2, 2, 2),
            shape=(2, 2, 2), affine=np.eye(4),
            header=SimpleNamespace(get_zooms=lambda: (1, 1, 1),
                                   get_xyzt_units=lambda: ("mm", "unknown")),
        )
        modules = {"nilearn": fake_nilearn, "nilearn.datasets": fake_datasets,
                   "nibabel": fake_nib}
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script), run_name="__main__")
        return scripts / "outputs" / "pilot_diverse", fetch

    def test_download_order_preserves_annotation_alignment_and_request(self):
        with tempfile.TemporaryDirectory() as directory:
            output, fetch = self.run_builder(directory, [2, 1])
            with (output / "image_term_weights.csv").open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows, [["image_id", "term"], ["2", "2.0"], ["1", "1.0"]])
            self.assertEqual(fetch.call_args.kwargs["image_ids"], [1, 2])
            self.assertIn(selection_hash([1, 2]), fetch.call_args.kwargs["data_dir"])
            self.assertEqual(json.loads((Path(directory) / "selections/pilot_diverse/selected_ids.json").read_text()), [1, 2])
            verify_preparation_inputs(output, Path(directory) / "selections/pilot_diverse", [2, 1], [2, 1])

    def test_rejects_wrong_download_before_writing_manifest(self):
        for ids in ([1, 2, 3], [1], [1, 1, 2]):
            with self.subTest(ids=ids), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.run_builder(directory, ids)
                self.assertFalse((Path(directory) / "neurovault_test/outputs/pilot_diverse/manifest.csv").exists())

    def test_no_selection_never_falls_back_to_generic_search(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self.run_builder(directory, [1, 2], with_selection=False)


if __name__ == "__main__":
    unittest.main()
