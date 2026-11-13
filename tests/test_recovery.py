from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo/kit"))
sys.path.insert(0, str(ROOT / "research/work/llvnr_port/graph"))
sys.path.insert(0, str(ROOT / "research/work"))

import recover_portable_sources as recovery  # noqa: E402


class FakeGroup:
    def __init__(self, raw: bytes, starts: tuple[int, ...] = (0,)):
        self.raw = raw
        self.starts = starts


def task(cin: int, cout: int, *, geometry: dict | None = None,
         input_hw: tuple[int, int] = (576, 324), output_hw: tuple[int, int] = (576, 324)) -> dict:
    return {
        "explicit_dimensions_whc": {
            "input": [input_hw[0], input_hw[1], cin],
            "output": [output_hw[0], output_hw[1], cout],
        },
        "geometry": geometry or {"kh": 1, "kw": 1, "output_x_factor": 1, "output_y_factor": 1},
    }


class RecoveryTests(unittest.TestCase):
    def test_special_nearest_replication_is_reconstructed(self) -> None:
        result = recovery.recover_special(
            FakeGroup(bytes.fromhex("0000003c") + b"\0" * 124),
            task(1, 1, geometry={"kh": 1, "kw": 1, "output_x_factor": 2, "output_y_factor": 2}),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["primitive"], "nearest_replication")
        np.testing.assert_array_equal(result["weights"], np.ones((1, 1, 2, 2), dtype=np.float16))

    def test_special_padding_identity_is_reconstructed(self) -> None:
        result = recovery.recover_special(
            FakeGroup(bytes.fromhex("003c0000") + b"\0" * 124),
            task(1, 1, input_hw=(576, 324), output_hw=(578, 326)),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["primitive"], "padding_identity")
        np.testing.assert_array_equal(result["weights"], np.ones((1, 1, 1, 1), dtype=np.float16))

    def test_special_haar_is_reconstructed(self) -> None:
        result = recovery.recover_special(
            FakeGroup(bytes.fromhex("380000c3") + b"\0" * 380),
            task(24, 6, geometry={"kh": 2, "kw": 1, "output_x_factor": 2, "output_y_factor": 2}),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["primitive"], "conv_transpose_haar")
        self.assertEqual(result["weights"].shape, (24, 6, 2, 2))
        np.testing.assert_array_equal(result["weights"][0, 0], np.full((2, 2), 0.5, dtype=np.float16))
        np.testing.assert_array_equal(result["weights"][6, 0], np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float16))

    def test_dependencies_include_required_baselines(self) -> None:
        self.assertIn(
            "LLV-HD-Wide-2x-v0.3.0_2304x1296",
            recovery.dependencies({"LLV-HD-Tele-4x-v0.3.0_2304x1296"}),
        )
        required = recovery.dependencies({"LLV-HD-Tele-4x-v0.3.0_2464x1388"})
        self.assertIn("LLV-HD-Wide-2x-v0.3.0_2304x1296", required)
        self.assertIn("LLV-HD-Tele-4x-v0.3.0_2304x1296", required)

    def test_recovery_profiles_are_sanitized_structural_metadata(self) -> None:
        path = ROOT / "research/work/llvnr_port/recovery_profiles.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            profile = json.load(stream)
        self.assertGreater(len(profile["transpose_layouts"]), 0)
        self.assertGreater(len(profile["sparse_transpose_layouts"]), 0)
        self.assertGreater(len(profile["padding_layouts"]), 0)
        text = json.dumps(profile, sort_keys=True)
        self.assertNotIn("/Users/", text)
        self.assertNotIn("model.hwx", text)
        self.assertNotIn("model_sha256", text)
        self.assertNotIn("weights.npy", text)

    def test_discover_models_uses_bundle_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-recovery-test-") as tmp:
            root = Path(tmp)
            model = root / "Wide.bundle/H19.bundle/main/main_ane/model.hwx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"synthetic")
            # The test bundle is intentionally not a valid LLVNR model name.
            self.assertEqual(recovery.discover_models(root), {})
            valid = root / "LLV-HD-Wide-2x-v0.3.0_2304x1296.bundle/H19.bundle/main/main_ane/model.hwx"
            valid.parent.mkdir(parents=True)
            valid.write_bytes(b"synthetic")
            found = recovery.discover_models(root)
            self.assertEqual(found["LLV-HD-Wide-2x-v0.3.0_2304x1296"], valid)


if __name__ == "__main__":
    unittest.main()
