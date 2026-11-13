from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo/kit"))

import build_kit  # noqa: E402
import verify_kit  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_fixture(root: Path) -> tuple[Path, Path]:
    model_id = "LLV-HD-Wide-2x-v0.3.0_2304x1296"
    assets = root / "assets"
    model = assets / model_id
    model.mkdir(parents=True)
    model_bytes = b"synthetic model identity"
    (model / "model.mil").write_text("program(1.3)\n")
    (model / "weights.bin").write_bytes(b"\x00\x00" * 8)
    source_graph = {"schema_version": 1, "nodes": [], "outputs": ["out_yuv", "state_yuv"]}
    (model / "source_graph.json").write_text(json.dumps(source_graph, indent=2) + "\n")
    report = {"passed": True, "metrics": {"out_yuv": {"passed": True, "max_abs": 0.0}}}
    (model / "export_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    files = {
        name: digest((model / name).read_bytes())
        for name in ("model.mil", "weights.bin", "source_graph.json", "export_validation.json")
    }
    manifest = {
        "model_id": model_id,
        "source_model_sha256": digest(model_bytes),
        "node_count": 1,
        "coefficient_group_count": 1,
        "inputs": [
            {"name": "cur_frame", "port": "cur", "shape": [1, 6, 648, 1152]},
            {"name": "his_frame", "port": "his", "shape": [1, 6, 648, 1152]},
            {"name": "dnr_zoom_map", "port": "dnr", "shape": [1, 1, 324, 576]},
            {"name": "fusion_zoom_map", "port": "fusion", "shape": [1, 1, 324, 576]},
        ],
        "outputs": [
            {"name": "out_yuv", "port": "out", "shape": [1, 6, 648, 1152]},
            {"name": "state_yuv", "port": "state", "shape": [1, 6, 648, 1152]},
        ],
        "files": files,
        "known_failed_evidence": [],
        "target_family": 2,
        "target_arch": "h13",
    }
    (model / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    catalog = root / "catalog"
    (catalog / "Tunings").mkdir(parents=True)
    tuning = catalog / "Tunings/Test-LLVTuningParameters.plist"
    tuning.write_bytes(b"synthetic tuning")
    catalog_doc = {
        "version": 2,
        "platform": "V63",
        "firmware_build": "test",
        "default_model_id": model_id,
        "scope": "synthetic fixture",
        "models": [{
            "id": model_id,
            "name": "LLV-HD-Wide-2x-v0.3.0",
            "title": "test",
            "family": "Wide",
            "zoom": "2x",
            "model_version": "v0.3.0",
            "width": 2304,
            "height": 1296,
            "bytes": len(model_bytes),
            "sha256": digest(model_bytes),
            "original_file": "Originals/model.hwx",
            "task_count": 1,
            "ne_groups": 1,
            "lut_groups": 0,
            "output_names": ["out_yuv", "state_yuv"],
            "preset_ids": ["Test-LLVTuningParameters"],
            "runtime_directory": None,
            "status": "source_only",
            "status_detail": "fixture",
            "runtime_manifest_sha256": None,
        }],
        "presets": [{
            "id": "Test-LLVTuningParameters",
            "name": "Test",
            "model_name": "LLV-HD-Wide-2x-v0.3.0",
            "file": "Tunings/Test-LLVTuningParameters.plist",
            "input_modulation": {"LUT": [0.0, 1.0]},
            "output_modulation": {"LUT": [0.0, 1.0]},
        }],
        "files": {"Tunings/Test-LLVTuningParameters.plist": {
            "sha256": digest(tuning.read_bytes()), "bytes": tuning.stat().st_size
        }},
    }
    (catalog / "catalog.json").write_text(json.dumps(catalog_doc, indent=2) + "\n")
    return assets, catalog


def args(assets: Path, catalog: Path, output: Path, *, no_zip: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        assets=assets, catalog=catalog, output=output,
        license=ROOT / "third_party/ANEForge/LICENSE",
        allow_partial=False, no_zip=no_zip,
    )


class KitBuilderTests(unittest.TestCase):
    def test_build_and_verify_directory_and_zip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-kit-test-") as tmp:
            root = Path(tmp)
            assets, catalog = make_fixture(root)
            kit = root / "local.llvnrkit"
            build_kit.build(args(assets, catalog, kit))
            result = verify_kit.verify(kit)
            self.assertTrue(result["passed"])
            self.assertEqual(result["models"], 1)
            self.assertTrue((kit / "Catalog/Tunings/Test-LLVTuningParameters.plist").is_file())
            self.assertFalse((kit / "Catalog/Tunings/Other.plist").exists())

            zipped = root / "zipped.llvnrkit"
            build_kit.build(args(assets, catalog, zipped, no_zip=False))
            archive = zipped.with_name(zipped.name + ".zip")
            self.assertTrue(archive.is_file())
            with tempfile.TemporaryDirectory(prefix="llvnr-unzip-test-") as extracted_tmp:
                extracted = Path(extracted_tmp)
                verify_kit.extract_archive(archive, extracted)
                verified = verify_kit.verify(verify_kit.locate_kit_root(extracted))
                self.assertTrue(verified["passed"])

    def test_target_fields_must_be_consistent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-kit-test-") as tmp:
            root = Path(tmp)
            assets, catalog = make_fixture(root)
            model = assets / "LLV-HD-Wide-2x-v0.3.0_2304x1296"
            manifest_path = model / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("target_arch")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            with self.assertRaises(ValueError):
                build_kit.build(args(assets, catalog, root / "local.llvnrkit"))

    def test_tamper_and_raw_artifact_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-kit-test-") as tmp:
            root = Path(tmp)
            assets, catalog = make_fixture(root)
            kit = root / "local.llvnrkit"
            build_kit.build(args(assets, catalog, kit))
            (kit / "Models/LLV-HD-Wide-2x-v0.3.0_2304x1296/model.mil").write_text("tampered")
            with self.assertRaises(ValueError):
                verify_kit.verify(kit)

    def test_zip_path_traversal_symlink_and_duplicate_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-zip-test-") as tmp:
            root = Path(tmp)
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as zf:
                zf.writestr("../evil", b"x")
            with self.assertRaises(ValueError):
                verify_kit.extract_archive(traversal, root / "out1")

            backslash = root / "backslash.zip"
            with zipfile.ZipFile(backslash, "w") as zf:
                zf.writestr("kit\\evil", b"x")
            with self.assertRaises(ValueError):
                verify_kit.extract_archive(backslash, root / "out1b")

            symlink = root / "symlink.zip"
            info = zipfile.ZipInfo("kit/link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(symlink, "w") as zf:
                zf.writestr(info, "target")
            with self.assertRaises(ValueError):
                verify_kit.extract_archive(symlink, root / "out2")

            duplicate = root / "duplicate.zip"
            with zipfile.ZipFile(duplicate, "w") as zf:
                zf.writestr("kit/kit.json", b"{}")
                zf.writestr("kit/kit.json", b"{}")
            with self.assertRaises(ValueError):
                verify_kit.extract_archive(duplicate, root / "out3")


if __name__ == "__main__":
    unittest.main()
