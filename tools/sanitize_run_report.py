#!/usr/bin/env python3
"""Create a shareable copy of a local LLVNR run report.

The raw report is intentionally diagnostic: it can contain a hostname, local
paths, frame timing records, and model-file names. This tool removes those
fields before a user sends a report to someone else. It does not upload data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SENSITIVE_KEY_PARTS = ("host", "user", "home", "directory", "path", "cwd", "volume", "username")
LOCAL_PATH = re.compile(r"(?:/" + "Users" + r"/|/" + "Volumes" + r"/|/" + "private" + r"/|~/)[^\s\"']*")
FILE_URL = re.compile(r"file://[^\s\"']*")


def sanitize_value(value: Any, markers: set[str]) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in SENSITIVE_KEY_PARTS):
                continue
            result[str(key)] = sanitize_value(item, markers)
        return result
    if isinstance(value, list):
        return [sanitize_value(item, markers) for item in value]
    if isinstance(value, str):
        text = FILE_URL.sub("<redacted-local-path>", value)
        text = LOCAL_PATH.sub("<redacted-local-path>", text)
        for marker in markers:
            if marker:
                text = text.replace(marker, "<redacted-host>")
        return text
    return value


def sanitize_line(line: str, markers: set[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return sanitize_value(value, markers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="directory containing summary.json and frames.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="new/empty output directory")
    parser.add_argument("--include-frames", action="store_true", help="include sanitized per-frame JSONL (raw frame rows are omitted by default)")
    parser.add_argument("--host", action="append", default=[], help="additional hostname/local identity to redact")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    summary_path = run_dir / "summary.json"
    frames_path = run_dir / "frames.jsonl"
    if not summary_path.is_file():
        print(f"error: summary.json not found: {summary_path}", file=sys.stderr)
        return 1
    if output.exists() and any(output.iterdir()):
        print(f"error: output directory must be new or empty: {output}", file=sys.stderr)
        return 1
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        print(f"error: invalid summary.json: {exc}", file=sys.stderr)
        return 1
    markers = {str(value) for value in args.host if value}
    for key in ("host", "hostname", "runDirectory", "run_directory"):
        value = summary.get(key) if isinstance(summary, dict) else None
        if isinstance(value, str):
            markers.add(value)
            markers.add(Path(value).name)
    sanitized = sanitize_value(summary, markers)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    frame_count = 0
    if args.include_frames and frames_path.is_file():
        with frames_path.open() as source, (output / "frames.redacted.jsonl").open("w") as destination:
            for line in source:
                row = sanitize_line(line, markers)
                if row is None:
                    continue
                # Timing rows can reveal source media cadence and are not needed
                # for a minimal bug report. Keep only explicit errors/recovery
                # events unless the caller deliberately requests all rows.
                if row.get("event") not in {"run_started", "run_finished", "recoverable_error", "recovered"}:
                    continue
                destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                frame_count += 1
    manifest = {
        "source": "LLVNR Demo local run",
        "frames_included": bool(args.include_frames and frames_path.is_file()),
        "frames_retained": frame_count,
        "raw_frames_omitted_by_default": True,
        "redactions": ["hostname", "local paths", "user/home/volume/directory/path fields"],
        "note": "Sanitized locally; no network upload was performed.",
    }
    (output / "SANITIZATION.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
