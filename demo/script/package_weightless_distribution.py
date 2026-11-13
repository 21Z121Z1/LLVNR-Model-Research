#!/usr/bin/env python3
"""Package the model-free App and the local kit builder for distribution.

The packager intentionally accepts only an already verified weightless App. It
never copies a generated `.llvnrkit`, model file, tuning, or CameraSetup.plist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import os
import pwd
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path


KIT_FILES = (
    "README.md",
    "extract_ipsw_assets.py",
    "build_kit.py",
    "verify_kit.py",
    "export_research_assets.py",
    "schema/kit.schema.json",
)
FORBIDDEN_NAMES = {
    "weights.bin",
    "model.mil",
    "manifest.json",
    "catalog.json",
    "CameraSetup.plist",
}
FORBIDDEN_SUFFIXES = {".hwx", ".net", ".weights", ".shape", ".espressif", ".npy"}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()




def assert_no_local_identity(root: Path) -> None:
    """Reject usernames, build roots, or hostnames retained by the compiler."""
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
        relative = path.relative_to(root).as_posix()
        if any(marker in data for marker in markers) or re.search(rb"/Users/[^/\x00\s]+", data):
            findings.append(relative)
    if findings:
        raise RuntimeError("local identity/build path found in package: " + ", ".join(findings[:5]))


def inventory(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"unexpected symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def assert_no_model_assets(root: Path) -> None:
    bad: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            bad.append(f"symlink:{relative}")
            continue
        if path.is_file():
            if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
                bad.append(relative)
            if path.name.endswith("LLVTuningParameters.plist"):
                bad.append(relative)
            if ".llvnrkit" in path.name.lower():
                bad.append(relative)
        if path.is_dir() and path.name in {"Models", "Catalog"} and "Contents/Resources" in relative:
            bad.append(relative + "/")
    if bad:
        preview = ", ".join(bad[:8])
        raise RuntimeError(f"model/firmware-derived files found in weightless package: {preview}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new package directory; ZIP is placed next to it")
    parser.add_argument("--app", type=Path, default=Path("dist/LLVNR Demo Weightless.app"))
    parser.add_argument("--readme", type=Path, default=Path("WEIGHTLESS_DISTRIBUTION.md"))
    parser.add_argument("--kit-dir", type=Path, default=Path("kit"))
    parser.add_argument(
        "--license",
        type=Path,
        default=Path("Sources/ANERuntime/LICENSE-ANEForge"),
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    app = (project / args.app).resolve() if not args.app.is_absolute() else args.app.resolve()
    output = args.output.resolve()
    archive = output.with_name(output.name + ".zip")
    checksum_file = output.with_name(output.name + ".sha256")
    readme = (project / args.readme).resolve() if not args.readme.is_absolute() else args.readme.resolve()
    kit_dir = (project / args.kit_dir).resolve() if not args.kit_dir.is_absolute() else args.kit_dir.resolve()
    license_file = (project / args.license).resolve() if not args.license.is_absolute() else args.license.resolve()

    if any(path.exists() for path in (output, archive, checksum_file)):
        raise RuntimeError("output already exists; choose a new package path")
    if not app.is_dir():
        raise RuntimeError(f"weightless App not found: {app}")
    if not readme.is_file() or not license_file.is_file():
        raise RuntimeError("README or runtime license is missing")
    for relative in KIT_FILES:
        if not (kit_dir / relative).is_file():
            raise RuntimeError(f"kit source file missing: {relative}")

    assert_no_model_assets(app)
    source_inventory = inventory(app)
    if not source_inventory:
        raise RuntimeError("weightless App is empty")
    run("/usr/bin/codesign", "--verify", "--deep", "--strict", str(app))

    executable = app / "Contents/MacOS/LLVNRDemo"
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    dependencies = run("/usr/bin/otool", "-L", str(executable)).stdout
    for line in dependencies.splitlines()[1:]:
        dependency = line.strip().split(" (", 1)[0]
        if not dependency.startswith(("/System/Library/", "/usr/lib/")):
            raise RuntimeError(f"dependency is not self-contained: {dependency}")

    output.mkdir(parents=True)
    staged_app = output / app.name
    run("/usr/bin/ditto", str(app), str(staged_app))
    if inventory(staged_app) != source_inventory:
        raise RuntimeError("staged App differs from source App")

    shutil.copy2(readme, output / "README.md")
    shutil.copy2(license_file, output / "LICENSE-ANEForge.txt")
    staged_kit = output / "kit"
    for relative in KIT_FILES:
        destination = staged_kit / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kit_dir / relative, destination)

    (output / "KIT-DISTRIBUTION-NOTICE.md").write_text(
        "# Distribution notice\n\n"
        "This package contains no LLVNR model, weight, tuning, CameraSetup.plist, "
        "or generated .llvnrkit. Users must produce a local kit from material they "
        "are authorized to use. The converter source does not grant rights to Apple "
        "firmware, model weights, tuning data, or derived model files.\n\n"
        "The kit directory contains tooling only. Do not replace it with a generated "
        "model kit when publishing this package.\n"
    )

    assert_no_model_assets(output)
    assert_no_local_identity(output)
    gatekeeper = run(
        "/usr/sbin/spctl", "--assess", "--type", "execute", "--verbose=2", str(staged_app), check=False
    )
    report = {
        "bundle_identifier": info["CFBundleIdentifier"],
        "version": info["CFBundleShortVersionString"],
        "build": info["CFBundleVersion"],
        "declared_minimum_macos": info["LSMinimumSystemVersion"],
        "architecture": run("/usr/bin/lipo", "-archs", str(executable)).stdout.strip(),
        "signing": "ad-hoc; not Developer ID signed or notarized",
        "gatekeeper_assessment_exit_code": gatekeeper.returncode,
        "gatekeeper_output": gatekeeper.stderr.strip().replace(str(staged_app), staged_app.name),
        "model_assets_included": False,
        "generated_kit_included": False,
        "local_identity_scan": "passed",
        "kit_tools": list(KIT_FILES),
        "source_app_files_sha256": source_inventory,
        "kit_tools_sha256": {
            relative: sha256(staged_kit / relative) for relative in KIT_FILES
        },
        "source_app_unchanged": inventory(app) == source_inventory,
        "runtime_retest": "Model runtime was not exercised during packaging; local kit installation remains user-performed.",
    }
    (output / "PACKAGE-VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    packaged_inventory = inventory(output)

    run(
        "/usr/bin/ditto",
        "-c",
        "-k",
        "--sequesterRsrc",
        "--keepParent",
        str(output),
        str(archive),
    )
    with tempfile.TemporaryDirectory(prefix="llvnr-weightless-package-check-") as temporary:
        run("/usr/bin/ditto", "-x", "-k", str(archive), temporary)
        restored = Path(temporary) / output.name
        if inventory(restored) != packaged_inventory:
            raise RuntimeError("archive roundtrip differs from staged package")
        assert_no_model_assets(restored)
        assert_no_local_identity(restored)
        run("/usr/bin/codesign", "--verify", "--deep", "--strict", str(restored / app.name))

    checksum_file.write_text(f"{sha256(archive)}  {archive.name}\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256_file": str(checksum_file),
                "version": report["version"],
                "build": report["build"],
                "model_assets_included": False,
                "checks": "passed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
