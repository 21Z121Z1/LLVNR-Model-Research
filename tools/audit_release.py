#!/usr/bin/env python3
"""Fail closed on personal data, firmware-derived assets, and release debris.

The audit intentionally reports only paths/rule names. It never prints the
matched username, hostname, email, certificate field, or other value.
"""
from __future__ import annotations

import argparse
import os
import pwd
import re
import socket
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".hwx", ".net", ".weights", ".shape", ".espressif", ".dat", ".fp16",
    ".npy", ".npz", ".bgra", ".llvnrkit", ".p12", ".mobileprovision",
    ".cer", ".crt", ".key", ".dylib",
}
FORBIDDEN_NAMES = {"weights.bin", "model.mil", "manifest.json", "catalog.json",
                   "CameraSetup.plist", ".DS_Store"}
IMAGE_VIDEO_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff",
                        ".gif", ".webp", ".mov", ".mp4", ".m4v", ".avi"}
PERSONAL_PATH = re.compile(rb"/" + rb"Users" + rb"/[A-Za-z0-9._-][^/\x00\s\"']*")
VOLUME_PATH = re.compile(rb"/" + rb"Volumes" + rb"/[^\x00\s\"']+")
IGNORED_DIRS = {".git", ".build", "__pycache__", ".swift-module-cache", "DerivedData", "dist", "Validation"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".swiftmodule", ".swiftdoc", ".dia", ".pcm", ".o", ".d"}
EMAIL = re.compile(rb"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
SECRET = re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")
TOKEN = re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})\b")
# CI hosts use generic account names such as "runner". Treating those as
# personal identity would flag ordinary prose (for example, "experiment
# runner") without improving privacy detection.
GENERIC_IDENTITIES = {"runner", "user", "root", "admin", "builder", "build", "github"}


def identities() -> set[bytes]:
    home = Path.home()
    values = {str(home)}
    if home.name.lower() not in GENERIC_IDENTITIES:
        values.add(home.name)
    try:
        gecos = pwd.getpwuid(os.getuid()).pw_gecos.split(",", 1)[0]
        if gecos and gecos.lower() not in GENERIC_IDENTITIES:
            values.add(gecos)
    except (KeyError, OSError):
        pass
    host = socket.gethostname()
    if host.lower() not in GENERIC_IDENTITIES:
        values.add(host)
    host_short = host.split(".", 1)[0]
    if host_short.lower() not in GENERIC_IDENTITIES:
        values.add(host_short)
    return {value.encode() for value in values if len(value) >= 3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    findings: list[str] = []
    markers = identities()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_symlink():
            findings.append(f"symlink: {rel}")
            continue
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in IGNORED_SUFFIXES:
            continue
        if path.name in FORBIDDEN_NAMES:
            findings.append(f"forbidden-name: {rel}")
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden-suffix: {rel}")
        if suffix in IMAGE_VIDEO_SUFFIXES:
            findings.append(f"personal-media-or-image: {rel}")
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"large-file: {rel}")
        if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024:
            data = path.read_bytes()
            if any(marker in data for marker in markers):
                findings.append(f"local-identity: {rel}")
            if PERSONAL_PATH.search(data):
                findings.append(f"absolute-user-path: {rel}")
            if VOLUME_PATH.search(data):
                findings.append(f"absolute-volume-path: {rel}")
            if EMAIL.search(data):
                findings.append(f"email-like-string: {rel}")
            if SECRET.search(data):
                findings.append(f"private-key: {rel}")
            if TOKEN.search(data):
                findings.append(f"token-like-string: {rel}")

    if findings:
        print("RELEASE AUDIT FAILED")
        for item in sorted(set(findings)):
            print(item)
        return 1
    print("RELEASE AUDIT PASSED")
    print(f"root={root.name} files={sum(1 for p in root.rglob('*') if p.is_file())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
