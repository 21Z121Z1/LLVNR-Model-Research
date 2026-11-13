#!/usr/bin/env python3
"""Model-free smoke tests for the public research snapshot."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/work/llvnr_port/graph"))
sys.path.insert(0, str(ROOT / "research/work"))
sys.path.insert(0, str(ROOT / "demo/kit"))

from portable_ops import conv2d, fp16, pooling  # noqa: E402
from h19_dense_codec import decode as dense_decode  # noqa: E402
from h19_dense_codec import encode as dense_encode  # noqa: E402
from extract_ipsw_assets import extract_pattern  # noqa: E402


class PortableOpsTests(unittest.TestCase):
    def test_fp16_bits(self) -> None:
        self.assertEqual(fp16(0x3C00), 1.0)

    def test_conv2d_single_channel(self) -> None:
        image = np.ones((1, 3, 3), dtype=np.float32)
        weights = np.ones((1, 1, 2, 2), dtype=np.float32)
        result = conv2d(image, weights, np.array([0.0]), (1, 2, 2), (1, 1), (0, 0))
        np.testing.assert_allclose(result, np.full((1, 2, 2), 4.0), rtol=0, atol=1e-6)

    def test_avg_pool(self) -> None:
        image = np.ones((1, 4, 4), dtype=np.float32)
        geometry = {"kh": 2, "kw": 2, "stride_y": 2, "stride_x": 2, "pad_top": 0, "pad_left": 0}
        result = pooling(image, (1, 2, 2), geometry, mode=0)
        np.testing.assert_allclose(result, np.ones((1, 2, 2)), rtol=0, atol=1e-6)

    def test_dense_coefficient_roundtrip(self) -> None:
        weights = np.arange(8, dtype=np.float16).reshape(4, 2, 1, 1)
        bias = np.arange(4, dtype=np.float16)
        raw = dense_encode(weights, bias, 2)
        decoded_w, decoded_b = dense_decode(raw, weights.shape, 2, True)
        np.testing.assert_array_equal(decoded_w.view(np.uint16), weights.view(np.uint16))
        np.testing.assert_array_equal(decoded_b.view(np.uint16), bias.view(np.uint16))


class IpswExtractorTests(unittest.TestCase):
    def test_pattern_is_limited_to_lowlightvideo(self) -> None:
        pattern = extract_pattern("V63")
        self.assertIn("ISPKit\\.framework/V63/LowLightVideo", pattern)
        self.assertIn("model\\.hwx", pattern)
        self.assertIn("LLVTuningParameters\\.plist", pattern)
        self.assertNotIn("CameraSetup", pattern)


if __name__ == "__main__":
    unittest.main()
