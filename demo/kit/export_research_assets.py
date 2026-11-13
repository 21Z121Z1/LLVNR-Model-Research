#!/usr/bin/env python3
"""Export portable LLVNR runtime assets with a local research checkout.

This bridge deliberately stops at the boundary of the user's private research
tree. It does not contain firmware parsing, decryption, or extraction code, and
it never downloads Apple firmware.

The wrapped ``export_catalog_assets.py`` must already exist in ``--research-root``
and must have been produced/authorized by the user.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# The twelve V63 rear-camera configurations used by LLVNR Demo. Keeping the
# set explicit prevents an accidental "export everything" when a research tree
# contains additional experiments or incomplete model directories.
DEFAULT_MODEL_IDS = (
    "LLV-HD-SuperWide-0.5x-v0.2.2_2304x1296",
    "LLV-HD-SuperWide-0.5x-v0.2.2_2464x1388",
    "LLV-HD-SuperWide-1x-v0.3.0_2304x1296",
    "LLV-HD-SuperWide-1x-v0.3.0_2464x1388",
    "LLV-HD-Tele-4x-v0.3.0_2304x1296",
    "LLV-HD-Tele-4x-v0.3.0_2464x1388",
    "LLV-HD-Tele-8x-v0.3.0_2304x1296",
    "LLV-HD-Tele-8x-v0.3.0_2464x1388",
    "LLV-HD-Wide-1x-v0.3.0_2304x1296",
    "LLV-HD-Wide-1x-v0.3.0_2464x1388",
    "LLV-HD-Wide-2x-v0.3.0_2304x1296",
    "LLV-HD-Wide-2x-v0.3.0_2464x1388",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
        help="LLVNR research root containing export_catalog_assets.py",
    )
    parser.add_argument(
        "--assets-out",
        type=Path,
        required=True,
        help="new output directory for portable runtime assets",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_IDS),
        metavar="MODEL_ID",
        help="one or more model IDs (defaults to the twelve LLVNR Demo models)",
    )
    args = parser.parse_args()

    root = args.research_root.resolve()
    exporter = root / "export_catalog_assets.py"
    if not exporter.is_file():
        print(f"error: exporter not found: {exporter}", file=sys.stderr)
        return 1

    assets_out = args.assets_out.resolve()
    if assets_out.exists():
        print(f"error: output already exists: {assets_out}", file=sys.stderr)
        return 1

    duplicates = sorted({model for model in args.models if args.models.count(model) > 1})
    if duplicates:
        print("error: duplicate model IDs: " + ", ".join(duplicates), file=sys.stderr)
        return 1

    assets_out.mkdir(parents=True)
    command = [sys.executable, str(exporter), *args.models, "--out", str(assets_out)]
    try:
        subprocess.run(command, cwd=str(root), check=True)
    except subprocess.CalledProcessError as exc:
        print(f"error: research exporter failed with status {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1

    missing = [model for model in args.models if not (assets_out / model / "manifest.json").is_file()]
    if missing:
        print("error: exporter did not produce manifests for: " + ", ".join(missing), file=sys.stderr)
        return 1

    report = {
        "research_root": str(root),
        "assets_out": str(assets_out),
        "models": args.models,
        "model_count": len(args.models),
        "note": (
            "Portable runtime assets were produced locally by the research "
            "exporter. This report contains no model weights."
        ),
    }
    (assets_out / "EXPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
