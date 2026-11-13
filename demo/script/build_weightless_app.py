#!/usr/bin/env python3
"""Copy a built app and remove all model/firmware-derived resources.

The resulting app keeps the runtime and user interface, but expects a local
model kit in Application Support. It is not a replacement for legal review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import socket
import subprocess
from pathlib import Path




def assert_no_local_identity(root: Path) -> None:
    """Reject build paths/usernames accidentally retained in a release App."""
    home = Path.home()
    full_name = pwd.getpwuid(os.getuid()).pw_gecos.split(",", 1)[0]
    markers = {
        str(home).encode(),
        home.name.encode(),
        full_name.encode(),
        socket.gethostname().encode(),
        socket.gethostname().split(".", 1)[0].encode(),
    }
    markers = {marker for marker in markers if len(marker) >= 3}
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        relative = str(path.relative_to(root))
        if any(marker in data for marker in markers) or re.search(rb"/Users/[^/\x00\s]+", data):
            findings.append(relative)
    if findings:
        raise SystemExit("local identity/build path found in App: " + ", ".join(findings[:5]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=Path("dist/LLVNR Demo.app"))
    parser.add_argument("--output", type=Path, default=Path("dist/LLVNR Demo Weightless.app"))
    parser.add_argument("--no-sign", action="store_true")
    args = parser.parse_args()
    source = args.app.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"missing built app: {source}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    shutil.copytree(source, output)
    resource_roots = [
        output / "Contents/Resources/LLVNRDemo_LLVNRDemo.bundle/Contents/Resources/Resources",
        output / "Contents/Resources/Resources",
    ]
    removed = []
    for root in resource_roots:
        for name in ("Models", "Catalog"):
            path = root / name
            if path.exists():
                shutil.rmtree(path)
                removed.append(str(path.relative_to(output)))
    forbidden = []
    forbidden_names = {
        "weights.bin", "model.mil", "manifest.json", "catalog.json",
        "CameraSetup.plist",
    }
    forbidden_suffixes = {".npy", ".hwx", ".net", ".weights", ".shape", ".espressif", ".mil"}
    for path in output.rglob("*"):
        relative = str(path.relative_to(output))
        if path.is_symlink():
            forbidden.append(relative + " (symlink)")
            continue
        if path.is_file():
            if path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes:
                forbidden.append(relative)
            if path.name.endswith("LLVTuningParameters.plist"):
                forbidden.append(relative)
        elif path.is_dir() and path.name in {"Models", "Catalog"} and "Contents/Resources" in relative:
            forbidden.append(relative + "/")
    if forbidden:
        raise SystemExit("weightless check failed: " + ", ".join(forbidden[:5]))
    assert_no_local_identity(output)
    if not args.no_sign:
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(output)], check=True)
        subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(output)], check=True)
    info = {
        "source_app": str(source),
        "output_app": str(output),
        "removed_resources": removed,
        "forbidden_files": forbidden,
        "local_identity_scan": "passed",
        "signed": not args.no_sign,
        "note": "Model kit must be installed locally before starting inference.",
    }
    (output.parent / "WEIGHTLESS-VERIFICATION.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
