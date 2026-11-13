from __future__ import annotations

import json
import plistlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo/kit"))

import build_catalog  # noqa: E402


def write_tuning(path: Path, model_name: str) -> None:
    document = {
        "LowLightVideoParameters": {
            "ModelInfo": {"Name": model_name},
            "NetworkInputsModulation": {"DNRZoomFactorScaleLUT": [0.0, 1.0, 100.0, 2.0]},
            "NetworkOutputsModulation": {"HistoryBlendingFactorLUT": [0.0, 0.5]},
        }
    }
    with path.open("wb") as stream:
        plistlib.dump(document, stream)


class CatalogBuilderTests(unittest.TestCase):
    def make_extraction(self, root: Path) -> None:
        lowlight = root / "24C5037i__iPhone19,2_3_7/System/Library/PrivateFrameworks/ISPKit.framework/V63/LowLightVideo"
        lowlight.mkdir(parents=True)
        for model_id, model_name in (
            ("LLV-HD-Wide-2x-v0.3.0_2304x1296", "LLV-HD-Wide-2x-v0.3.0"),
            ("LLV-HD-Tele-4x-v0.3.0_2464x1388", "LLV-HD-Tele-4x-v0.3.0"),
        ):
            model = lowlight / f"{model_id}.bundle/H19.bundle/main/main_ane/model.hwx"
            model.parent.mkdir(parents=True)
            model.write_bytes(("synthetic:" + model_id).encode())
            write_tuning(lowlight / f"{model_name}-LLVTuningParameters.plist", model_name)

    def test_catalog_maps_tuning_by_model_info_and_filters_models(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-catalog-test-") as tmp:
            extraction = Path(tmp) / "extracted"
            self.make_extraction(extraction)
            output = Path(tmp) / "catalog"
            result = build_catalog.build_catalog(
                extraction, None, output, allow_unported=True
            )
            self.assertEqual(result["models"], 2)
            self.assertEqual(result["presets"], 2)
            catalog = json.loads((output / "catalog.json").read_text())
            self.assertEqual(catalog["version"], 2)
            self.assertEqual(catalog["platform"], "V63")
            self.assertEqual(catalog["firmware_build"], "24C5037i")
            self.assertEqual({entry["status"] for entry in catalog["models"]}, {"source_only"})
            self.assertEqual({preset["model_name"] for preset in catalog["presets"]}, {
                "LLV-HD-Wide-2x-v0.3.0", "LLV-HD-Tele-4x-v0.3.0"
            })
            text = (output / "catalog.json").read_text()
            self.assertNotIn(str(extraction), text)
            self.assertNotIn("/Users/", text)

            filtered = Path(tmp) / "filtered"
            build_catalog.build_catalog(
                extraction, None, filtered, allow_unported=True,
                selected_models=["LLV-HD-Tele-4x-v0.3.0_2464x1388"],
            )
            one = json.loads((filtered / "catalog.json").read_text())
            self.assertEqual([entry["id"] for entry in one["models"]], ["LLV-HD-Tele-4x-v0.3.0_2464x1388"])
            self.assertEqual(len(one["presets"]), 1)

    def test_missing_model_mapping_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-catalog-test-") as tmp:
            extraction = Path(tmp) / "extracted"
            lowlight = extraction / "V63/LowLightVideo"
            lowlight.mkdir(parents=True)
            model = lowlight / "LLV-HD-Wide-1x-v0.3.0_2304x1296.bundle/H19.bundle/main/main_ane/model.hwx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            write_tuning(lowlight / "Other-LLVTuningParameters.plist", "LLV-HD-Tele-8x-v0.3.0")
            with self.assertRaises(ValueError):
                build_catalog.build_catalog(extraction, None, Path(tmp) / "catalog", allow_unported=True)


if __name__ == "__main__":
    unittest.main()
