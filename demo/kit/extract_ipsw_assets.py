#!/usr/bin/env python3
"""Extract LLVNR model and tuning files from a local IPSW/OTA.

This is a thin, auditable wrapper around the external `ipsw` CLI. It does not
contain firmware keys, download model files, or implement decryption itself.
The caller supplies a local IPSW/OTA that they are authorized to process.

The extraction pattern is deliberately limited to ISPKit LowLightVideo:
  * H19.bundle/main/main_ane/model.hwx
  * *LLVTuningParameters.plist

Use the output only in a private workspace. Model and tuning files are not
covered by this repository's Apache-2.0 license.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_VARIANT = "V63"
VARIANTS = ("V63", "V64", "V64s")
BUILD_PREFIX = "LLV-HD-"
MODEL_RE = re.compile(r"^LLV-HD-.+\.bundle$")
TUNING_RE = re.compile(r".*LLVTuningParameters\.plist$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_regex(value: str) -> str:
    if value == "all":
        return "(?:V63|V64|V64s)"
    if value not in VARIANTS:
        raise argparse.ArgumentTypeError("variant must be V63, V64, V64s, or all")
    return re.escape(value)


def is_ota(path: Path) -> bool:
    return path.suffix.lower() in {".aea", ".ota", ".zip"}


def extract_pattern(variant: str) -> str:
    # The two alternatives are intentionally narrow. In particular, this does
    # not extract arbitrary IPSKit binaries, CameraSetup.plist, or other camera
    # tuning.
    lowlight = (
        rf"^System/Library/PrivateFrameworks/ISPKit\.framework/"
        rf"{variant_regex(variant)}/LowLightVideo/"
    )
    return lowlight + (
        r"(?:LLV-HD-[^/]+\.bundle/H19\.bundle/main/main_ane/model\.hwx"
        r"|.*LLVTuningParameters\.plist)$"
    )


def ipsw_command(tool: str, source: Path, output: Path, pattern: str, variant: str,
                 range_regex: str | None) -> list[str]:
    if is_ota(source):
        command = [tool, "ota", "extract", str(source), "--pattern", pattern,
                   "--confirm", "--output", str(output)]
        # `--confirm` is required for non-interactive OTA payloadv2 search.
    else:
        command = [tool, "extract", "--files", "--pattern", pattern,
                   "--output", str(output)]
    if range_regex:
        command.extend(["--range", range_regex])
    return command


def walk_assets(root: Path, variants: set[str]) -> tuple[list[dict], list[dict]]:
    models: list[dict] = []
    tunings: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        selected_variant = next((p for p in parts if p in variants), None)
        if selected_variant is None:
            continue
        if path.name == "model.hwx" and path.parent.name == "main_ane":
            model = next((p for p in parts if MODEL_RE.fullmatch(p)), None)
            if model is None:
                continue
            models.append({
                "variant": selected_variant,
                "model_id": model.removesuffix(".bundle"),
                "path": str(rel),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        elif TUNING_RE.fullmatch(path.name):
            tunings.append({
                "variant": selected_variant,
                "name": path.stem,
                "path": str(rel),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return models, tunings


def tool_version(tool: str) -> str | None:
    try:
        result = subprocess.run([tool, "version"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="local .ipsw, .aea, .ota, or OTA zip")
    parser.add_argument("--output", type=Path, required=True,
                        help="new private extraction directory")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=(*VARIANTS, "all"), help="ISPKit variant (default: V63)")
    parser.add_argument("--range", dest="range_regex", default=None,
                        help="optional ipsw payload range regex, e.g. 'payload\\.(033|040)'")
    parser.add_argument("--allow-partial", action="store_true",
                        help="allow missing model/tuning files; useful while discovering a new OTA")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    parser.add_argument("--ipsw", default=os.environ.get("IPSW", "ipsw"),
                        help="path to the ipsw executable (default: $IPSW or ipsw)")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source is not a file: {source}")
    tool = shutil.which(args.ipsw) if os.path.sep not in args.ipsw else args.ipsw
    if not tool:
        parser.error(f"ipsw executable not found: {args.ipsw}")

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pattern = extract_pattern(args.variant)
    command = ipsw_command(tool, source, output, pattern, args.variant, args.range_regex)
    print(json.dumps({"command": [part if part != str(source) else "<local-input>" for part in command],
                      "pattern": pattern}, indent=2))
    if args.dry_run:
        return 0

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"error: ipsw exited with status {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1

    variants = set(VARIANTS) if args.variant == "all" else {args.variant}
    models, tunings = walk_assets(output, variants)
    expected_models = 36 if args.variant == "all" else 12
    expected_tunings = 18 if args.variant == "all" else 6
    complete = len(models) == expected_models and len(tunings) == expected_tunings
    if not args.allow_partial and not complete:
        print(
            f"error: expected {expected_models} models and {expected_tunings} tunings, "
            f"found {len(models)} and {len(tunings)}; use --allow-partial only for discovery",
            file=sys.stderr,
        )
        return 2

    report = {
        "schema_version": 1,
        "source_kind": "OTA" if is_ota(source) else "IPSW",
        "source_bytes": source.stat().st_size,
        "variant": args.variant,
        "pattern": pattern,
        "ipsw_version": tool_version(tool),
        "models": models,
        "tunings": tunings,
        "counts": {"models": len(models), "tunings": len(tunings)},
        "complete_expected_set": complete,
        "note": (
            "Local private extraction only. Model.hwx and LLVTuningParameters.plist "
            "are not redistributed by LLVNR-Model-Research; the caller is responsible "
            "for the rights and handling of the supplied firmware and these files."
        ),
    }
    report_path = output / "EXTRACT-MANIFEST.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(report_path), **report["counts"],
                      "complete_expected_set": complete}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
