from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo/kit"))

import export_runtime_assets as exporter  # noqa: E402


class RuntimeExporterTests(unittest.TestCase):
    def test_structural_export_emits_mil_weights_and_normalized_graph(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-export-test-") as tmp:
            root = Path(tmp)
            model_id = "LLV-HD-Wide-2x-v0.3.0_2304x1296"
            source = root / "sources" / model_id
            source.mkdir(parents=True)
            weights = np.ones((1, 1, 1, 1), dtype=np.float16)
            bias = np.zeros((1,), dtype=np.float16)
            np.save(source / "weights.npy", weights)
            np.save(source / "bias.npy", bias)
            plan = {
                "schema_version": 1,
                "model_sha256": "a" * 64,
                "coefficients": {
                    "K0": {
                        "weights": "weights.npy",
                        "bias": "bias.npy",
                        "weights_sha256_fp16": hashlib.sha256(weights.tobytes()).hexdigest(),
                        "bias_sha256_fp16": hashlib.sha256(bias.tobytes()).hexdigest(),
                    }
                },
                "nodes": [
                    {"name": "his_frame", "op": "input", "inputs": [], "shape": [1, 6, 648, 1152]},
                    {"name": "cur_frame", "op": "input", "inputs": [], "shape": [1, 6, 648, 1152]},
                    {"name": "dnr_zoom_map", "op": "input", "inputs": [], "shape": [1, 1, 324, 576]},
                    {"name": "fusion_zoom_map", "op": "input", "inputs": [], "shape": [1, 1, 324, 576]},
                    {"name": "out_yuv", "op": "affine", "inputs": ["cur_frame"], "alpha": 1.0, "beta": 0.0},
                    {"name": "state_yuv", "op": "affine", "inputs": ["his_frame"], "alpha": 1.0, "beta": 0.0},
                    {"name": "dnr_out", "op": "affine", "inputs": ["dnr_zoom_map"], "alpha": 1.0, "beta": 0.0},
                    {"name": "fusion_out", "op": "affine", "inputs": ["fusion_zoom_map"], "alpha": 1.0, "beta": 0.0},
                ],
                "outputs": ["out_yuv", "state_yuv", "dnr_out", "fusion_out"],
            }
            (source / "source_graph.json").write_text(json.dumps(plan, indent=2) + "\n")
            assets = root / "assets"
            assets.mkdir()
            result = exporter.export_one(source, assets, source, "none", 1, None)
            self.assertIsNone(result["validation"])
            destination = assets / model_id
            for name in ("model.mil", "weights.bin", "source_graph.json", "manifest.json"):
                self.assertTrue((destination / name).is_file(), name)
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual({port["name"] for port in manifest["inputs"]}, exporter.EXPECTED_INPUTS)
            self.assertEqual({port["name"] for port in manifest["ane_inputs"]}, exporter.EXPECTED_INPUTS)
            normalized = json.loads((destination / "source_graph.json").read_text())
            self.assertTrue(normalized["coefficients"]["K0"]["weights"].startswith("coefficients/"))

    def test_absolute_coefficient_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-export-test-") as tmp:
            root = Path(tmp)
            source = root / "LLV-HD-Wide-2x-v0.3.0_2304x1296"
            source.mkdir()
            plan = {
                "coefficients": {"K0": {"weights": "/tmp/weights.npy", "bias": None}},
                "nodes": [], "outputs": [],
            }
            with self.assertRaises(ValueError):
                exporter.normalize_plan(plan, source, root, root / "out")


if __name__ == "__main__":
    unittest.main()
