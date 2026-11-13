#!/usr/bin/env python3
"""Run the local LLVNR extraction -> catalog -> kit -> verify pipeline.

All model-derived output stays in the caller's private directories. The script
never downloads firmware and never publishes anything. A clean checkout can run
the pipeline when the caller supplies either:

* ``--ota`` / ``--extracted`` only: recover portable source graphs and
  coefficient arrays from the local model HWX files, then export them;
* ``--sources``: use already generated portable source graphs and arrays; or
* ``--assets``: use an already generated portable runtime asset root.

Recovery is implemented by ``recover_portable_sources.py`` and uses only local
HWX files plus sanitized structural metadata. The exporter then uses the
vendored ANEForge subset. The pipeline fails closed when a model graph cannot be
recovered or validated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[1] if (KIT.parents[1] / "research").is_dir() else KIT.parent


def run(command: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--ota", type=Path, help="local .ipsw/.aea/.ota file to extract")
    source.add_argument("--extracted", type=Path, help="already extracted private directory")
    parser.add_argument("--sources", type=Path, help="portable source graph root; omit to recover from --extracted")
    parser.add_argument("--assets", type=Path, help="already generated portable asset root")
    parser.add_argument("--recovery-profiles", type=Path, help="optional recovery_profiles.json.gz override")
    parser.add_argument("--recovery-validation", choices=("none", "cpu"), default="none",
                        help="optional whole-frame CPU comparison during source recovery")
    parser.add_argument("--coefficient-root", type=Path, help="root for relative coefficient paths (default: --sources)")
    parser.add_argument("--work", type=Path, required=True, help="new/empty private work directory")
    parser.add_argument("--output", type=Path, required=True, help="new .llvnrkit output directory")
    parser.add_argument("--models", nargs="+", help="optional model IDs to include")
    parser.add_argument("--variant", default="V63", choices=("V63", "V64", "V64s"), help="ISPKit variant")
    parser.add_argument("--firmware-build", help="firmware build label override")
    parser.add_argument("--camera-setup", type=Path, help="optional local CameraSetup.plist")
    parser.add_argument("--validation", choices=("ane", "cpu", "none"), default="ane", help="exporter validation mode")
    parser.add_argument("--seed", type=int, default=20261112, help="synthetic validation seed")
    parser.add_argument("--license", type=Path, default=ROOT / "third_party" / "ANEForge" / "LICENSE")
    parser.add_argument("--no-zip", action="store_true", help="do not create the sibling kit ZIP")
    parser.add_argument("--allow-partial-extraction", action="store_true", help="allow incomplete OTA extraction")
    args = parser.parse_args()

    work = args.work.resolve()
    output = args.output.resolve()
    if work.exists() and any(work.iterdir()):
        parser.error(f"work directory must be new or empty: {work}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be new or empty: {output}")
    work.mkdir(parents=True, exist_ok=True)

    extracted = args.extracted.resolve() if args.extracted else work / "extracted"
    if args.ota:
        command = [sys.executable, str(KIT / "extract_ipsw_assets.py"), str(args.ota.resolve()),
                   "--output", str(extracted), "--variant", args.variant]
        if args.allow_partial_extraction:
            command.append("--allow-partial")
        run(command)
    elif not args.extracted:
        parser.error("provide --extracted or --ota")

    if args.assets:
        assets = args.assets.resolve()
    else:
        sources = args.sources.resolve() if args.sources else work / "sources"
        if not args.sources:
            command = [sys.executable, str(KIT / "recover_portable_sources.py"),
                       "--extracted", str(extracted), "--output", str(sources),
                       "--validate", args.recovery_validation]
            if args.allow_partial_extraction:
                command.append("--allow-partial")
            if args.models:
                command += ["--models", *args.models]
            if args.recovery_profiles:
                command += ["--profiles", str(args.recovery_profiles.resolve())]
            run(command)
        assets = work / "assets"
        command = [sys.executable, str(KIT / "export_runtime_assets.py"),
                   "--sources", str(sources), "--output", str(assets),
                   "--validation", args.validation, "--seed", str(args.seed)]
        if args.coefficient_root:
            command += ["--coefficient-root", str(args.coefficient_root.resolve())]
        if args.models:
            command += ["--models", *args.models]
        run(command)

    catalog = work / "catalog"
    command = [sys.executable, str(KIT / "build_catalog.py"),
               "--extracted", str(extracted), "--assets", str(assets), "--output", str(catalog),
               "--variant", args.variant]
    if args.firmware_build:
        command += ["--firmware-build", args.firmware_build]
    if args.camera_setup:
        command += ["--camera-setup", str(args.camera_setup.resolve())]
    if args.models:
        command += ["--models", *args.models]
    run(command)

    command = [sys.executable, str(KIT / "build_kit.py"),
               "--assets", str(assets), "--catalog", str(catalog), "--output", str(output),
               "--license", str(args.license.resolve())]
    if args.no_zip:
        command.append("--no-zip")
    run(command)
    run([sys.executable, str(KIT / "verify_kit.py"), str(output)])

    summary = {
        "kit": str(output),
        "zip": None if args.no_zip else str(output.with_name(output.name + ".zip")),
        "work": str(work),
        "models": args.models or "all catalog models with assets",
        "validation": args.validation,
        "recovery_validation": args.recovery_validation,
        "sources": "provided" if args.sources else "recovered from local extraction",
        "note": "Private model-derived output; do not publish without confirming rights.",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
