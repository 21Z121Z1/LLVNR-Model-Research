#!/usr/bin/env python3
"""Verify a local ``.llvnrkit`` directory or ZIP without loading a model.

The verifier is intentionally independent from the App. It rejects path
traversal, symlinks, unlisted files, raw firmware artifacts, model/catalog
identity mismatches, and any content whose SHA-256 no longer matches the kit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


RAW_SUFFIXES = {".hwx", ".net", ".weights", ".shape", ".espressif"}
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_MEMBER_PATH = 4096


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"unsafe path: {value!r}")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        raise ValueError(f"unsafe path: {value}")
    return Path(*posix.parts)


def inventory(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed: {relative}")
        if path.is_file():
            result[relative.as_posix()] = {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
    return result


def verify(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError("kit must be a directory")
    kit_path = root / "kit.json"
    if not kit_path.is_file() or kit_path.is_symlink():
        raise ValueError("kit.json is missing or not a regular file")
    kit = read_json(kit_path)

    if kit.get("schema_version") != 1:
        raise ValueError("unsupported kit schema")
    model_ids = kit.get("model_ids")
    if not isinstance(model_ids, list) or not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("kit model_ids are empty or duplicated")
    files = kit.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("kit files table is missing")

    actual_files = inventory(root)
    actual_files.pop("kit.json", None)
    if set(actual_files) != set(files):
        missing = sorted(set(files) - set(actual_files))
        extra = sorted(set(actual_files) - set(files))
        raise ValueError(f"kit file inventory mismatch; missing={missing[:3]} extra={extra[:3]}")

    for relative, expected in files.items():
        path = safe_relative(relative)
        full_path = root / path
        if not full_path.is_file() or full_path.is_symlink():
            raise ValueError(f"missing regular file: {relative}")
        record = actual_files[relative]
        if record["bytes"] != expected.get("bytes"):
            raise ValueError(f"size mismatch: {relative}")
        if record["sha256"] != expected.get("sha256"):
            raise ValueError(f"hash mismatch: {relative}")

    raw = [path for path in actual_files if Path(path).suffix.lower() in RAW_SUFFIXES]
    if raw:
        raise ValueError("raw firmware artifacts found: " + ", ".join(raw[:3]))

    expected_total = sum(record["bytes"] for record in actual_files.values())
    if kit.get("total_bytes") != expected_total:
        raise ValueError("kit total_bytes mismatch")
    identity_material = json.dumps(
        {name: record["sha256"] for name, record in sorted(actual_files.items())},
        sort_keys=True,
    ).encode()
    expected_kit_id = "llvnr-" + hashlib.sha256(identity_material).hexdigest()[:16]
    if kit.get("kit_id") != expected_kit_id:
        raise ValueError("kit_id mismatch")

    catalog_path = root / "Catalog" / "catalog.json"
    if not catalog_path.is_file() or digest(catalog_path) != kit.get("catalog_sha256"):
        raise ValueError("catalog hash mismatch")
    catalog = read_json(catalog_path)
    catalog_models = catalog.get("models")
    if not isinstance(catalog_models, list):
        raise ValueError("catalog models are missing")
    catalog_ids = [entry.get("id") for entry in catalog_models]
    if len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError("catalog model IDs are duplicated")
    catalog_by_id = {entry.get("id"): entry for entry in catalog_models}
    if set(model_ids) != set(catalog_by_id):
        raise ValueError("kit model IDs and catalog model IDs differ")

    catalog_files = catalog.get("files")
    if not isinstance(catalog_files, dict):
        raise ValueError("catalog file inventory is missing")
    for relative, expected in catalog_files.items():
        safe_relative(relative)
        kit_relative = f"Catalog/{relative}"
        if kit_relative not in files:
            raise ValueError(f"catalog file is not in kit inventory: {relative}")
        actual = files[kit_relative]
        if actual.get("sha256") != expected.get("sha256") or actual.get("bytes") != expected.get("bytes"):
            raise ValueError(f"catalog file record mismatch: {relative}")

    model_records = kit.get("models")
    if not isinstance(model_records, dict) or set(model_records) != set(model_ids):
        raise ValueError("kit model records are missing or inconsistent")
    if catalog.get("default_model_id") not in model_ids:
        raise ValueError("catalog default model is not present in the kit")

    for model_id in model_ids:
        try:
            model_key = safe_relative(model_id)
        except ValueError as exc:
            raise ValueError(f"invalid model ID: {model_id!r}") from exc
        if len(model_key.parts) != 1:
            raise ValueError(f"model ID must be one path component: {model_id!r}")

        record = model_records[model_id]
        manifest_path = root / "Models" / model_id / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"manifest missing: {model_id}")
        if digest(manifest_path) != record.get("manifest_sha256"):
            raise ValueError(f"manifest hash mismatch: {model_id}")
        manifest = read_json(manifest_path)
        if manifest.get("model_id") != model_id:
            raise ValueError(f"manifest model_id mismatch: {model_id}")
        target_family = manifest.get("target_family")
        target_arch = manifest.get("target_arch")
        if (target_family is None) != (target_arch is None):
            raise ValueError(f"target fields are incomplete: {model_id}")
        if target_family is not None:
            if isinstance(target_family, bool) or not isinstance(target_family, int) or target_family < 2:
                raise ValueError(f"target family is invalid: {model_id}")
            if not isinstance(target_arch, str) or not target_arch:
                raise ValueError(f"target architecture is invalid: {model_id}")
        if manifest.get("source_model_sha256") != record.get("source_model_sha256"):
            raise ValueError(f"source model mismatch: {model_id}")
        if manifest.get("source_model_sha256") != catalog_by_id[model_id].get("sha256"):
            raise ValueError(f"catalog source model mismatch: {model_id}")

        declared = manifest.get("files")
        if not isinstance(declared, dict) or not declared:
            raise ValueError(f"runtime file list missing: {model_id}")
        model_root = root / "Models" / model_id
        actual_model_files = inventory(model_root)
        actual_model_files.pop("manifest.json", None)
        if set(actual_model_files) != set(declared):
            raise ValueError(f"runtime file inventory mismatch: {model_id}")
        for relative, expected_hash in declared.items():
            safe_relative(relative)
            if actual_model_files[relative]["sha256"] != expected_hash:
                raise ValueError(f"runtime file mismatch: {model_id}/{relative}")

    return {
        "passed": True,
        "kit_id": kit.get("kit_id"),
        "models": len(model_ids),
        "files": len(files),
        "bytes": expected_total,
    }


def extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("ZIP contains too many members")
        total = 0
        seen: set[str] = set()
        for member in members:
            name = member.filename
            if len(name) > MAX_MEMBER_PATH:
                raise ValueError("ZIP member path is too long")
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if member.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member is not allowed: {name}")
            if member.file_size < 0 or member.file_size > MAX_ARCHIVE_BYTES:
                raise ValueError(f"ZIP member is too large: {name}")
            total += member.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("ZIP expands beyond the verifier size limit")
            safe_relative(name.rstrip("/"))
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink in ZIP is not allowed: {name}")
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / safe_relative(member.filename.rstrip("/"))
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def locate_kit_root(extracted: Path) -> Path:
    if (extracted / "kit.json").is_file():
        return extracted
    children = [path for path in extracted.iterdir() if not path.name.startswith(".")]
    candidates = [path for path in children if path.is_dir() and (path / "kit.json").is_file()]
    if len(candidates) != 1 or len(children) != 1:
        raise ValueError("ZIP must contain exactly one kit directory and no sibling files")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kit", type=Path)
    args = parser.parse_args()
    try:
        if args.kit.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory(prefix="llvnr-kit-verify-") as tmp:
                extracted = Path(tmp)
                extract_archive(args.kit, extracted)
                result = verify(locate_kit_root(extracted))
        else:
            result = verify(args.kit)
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
