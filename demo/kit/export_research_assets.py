#!/usr/bin/env python3
"""Compatibility wrapper for the public LLVNR runtime asset exporter.

New code should call ``export_runtime_assets.py`` directly. This wrapper keeps
the historical ``--research-root`` bridge for users who already have a private
research exporter, while also accepting a portable source-graph tree through
``--sources`` so a clean checkout can use the public conversion path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[1] if (KIT.parents[1] / "research").is_dir() else KIT.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sources", type=Path, help="public portable source-graph root")
    source.add_argument("--research-root", type=Path, help="legacy private research root containing export_catalog_assets.py")
    parser.add_argument("--coefficient-root", type=Path, help="coefficient root for --sources (default: --sources)")
    parser.add_argument("--assets-out", type=Path, required=True, help="new output directory for portable runtime assets")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_IDS), metavar="MODEL_ID")
    parser.add_argument("--validation", choices=("ane", "cpu", "none"), default="ane")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--precision-dylib", type=Path)
    args = parser.parse_args()

    assets_out = args.assets_out.resolve()
    if assets_out.exists() and any(assets_out.iterdir()):
        print(f"error: output must be new or empty: {assets_out}", file=sys.stderr)
        return 1
    assets_out.mkdir(parents=True, exist_ok=True)
    if args.sources:
        command = [sys.executable, str(KIT / "export_runtime_assets.py"),
                   "--sources", str(args.sources.resolve()), "--output", str(assets_out),
                   "--validation", args.validation, "--seed", str(args.seed),
                   "--models", *args.models]
        if args.coefficient_root:
            command += ["--coefficient-root", str(args.coefficient_root.resolve())]
        if args.precision_dylib:
            command += ["--precision-dylib", str(args.precision_dylib.resolve())]
        result = subprocess.run(command, cwd=str(ROOT))
        return result.returncode

    # Legacy private bridge. It intentionally remains opt-in and is not needed
    # by the public conversion path.
    root = args.research_root.resolve()
    exporter = root / "export_catalog_assets.py"
    if not exporter.is_file():
        print(f"error: legacy exporter not found: {exporter}", file=sys.stderr)
        return 1
    duplicates = sorted({model for model in args.models if args.models.count(model) > 1})
    if duplicates:
        print("error: duplicate model IDs: " + ", ".join(duplicates), file=sys.stderr)
        return 1
    command = [sys.executable, str(exporter), *args.models, "--out", str(assets_out)]
    result = subprocess.run(command, cwd=str(root))
    if result.returncode:
        return result.returncode
    missing = [model for model in args.models if not (assets_out / model / "manifest.json").is_file()]
    if missing:
        print("error: exporter did not produce manifests for: " + ", ".join(missing), file=sys.stderr)
        return 1
    report = {"assets_out": str(assets_out), "models": args.models, "legacy_research_bridge": True}
    (assets_out / "EXPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
