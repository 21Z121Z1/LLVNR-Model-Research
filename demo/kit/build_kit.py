#!/usr/bin/env python3
"""Build a self-contained LLVNR model kit from locally produced runtime assets.

The builder never downloads firmware and never embeds model weights in this
script. It copies only runtime files that the user produced locally, validates
their manifests and hashes, and writes a portable `.llvnrkit` directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
REQUIRED_MODEL_FILES = {"model.mil", "weights.bin", "source_graph.json"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or not path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*path.parts)


def files_under(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in a kit: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
    return result


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ValueError(f"missing file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_manifest(model_id: str, manifest: dict, report: dict) -> None:
    if not manifest.get("known_failed_evidence") == []:
        raise ValueError(f"{model_id}: manifest contains known failures")
    if report.get("passed") is not True:
        raise ValueError(f"{model_id}: export validation did not pass")
    if not isinstance(report.get("metrics"), dict):
        raise ValueError(f"{model_id}: export validation has no CPU/ANE metrics")
    expected = {"cur_frame", "his_frame", "dnr_zoom_map", "fusion_zoom_map"}
    if {p.get("name") for p in manifest.get("inputs", [])} != expected:
        raise ValueError(f"{model_id}: unexpected input ports")
    outputs = {p.get("name") for p in manifest.get("outputs", [])}
    if not {"out_yuv", "state_yuv"}.issubset(outputs):
        raise ValueError(f"{model_id}: missing required output ports")
    current = next((p for p in manifest.get("inputs", []) if p.get("name") == "cur_frame"), None)
    if not current or len(current.get("shape", [])) != 4:
        raise ValueError(f"{model_id}: cur_frame shape is missing")
    padded_width, padded_height = current["shape"][3] * 2, current["shape"][2] * 2
    if (padded_width, padded_height) not in ((2304, 1296), (2464, 1388)):
        raise ValueError(f"{model_id}: unsupported padded canvas")
    target_family = manifest.get("target_family")
    target_arch = manifest.get("target_arch")
    if (target_family is None) != (target_arch is None):
        raise ValueError(f"{model_id}: target family and architecture must be specified together")
    if target_family is not None:
        if isinstance(target_family, bool) or not isinstance(target_family, int) or target_family < 2:
            raise ValueError(f"{model_id}: invalid target family")
        if not isinstance(target_arch, str) or not target_arch:
            raise ValueError(f"{model_id}: invalid target architecture")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not REQUIRED_MODEL_FILES.issubset(declared):
        raise ValueError(f"{model_id}: incomplete runtime file list")


def copy_model(asset_root: Path, model_id: str, catalog_entry: dict, destination: Path) -> dict:
    source = asset_root / model_id
    manifest_path = source / "manifest.json"
    report_path = source / "export_validation.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError(f"{model_id}: missing manifest or export validation")
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    validate_manifest(model_id, manifest, report)
    if manifest.get("source_model_sha256") != catalog_entry.get("sha256"):
        raise ValueError(f"{model_id}: catalog/model source identity mismatch")
    current = next(p for p in manifest["inputs"] if p["name"] == "cur_frame")
    padded_width, padded_height = current["shape"][3] * 2, current["shape"][2] * 2
    if (padded_width, padded_height) != (catalog_entry.get("width"), catalog_entry.get("height")):
        raise ValueError(f"{model_id}: catalog/model canvas mismatch")

    model_destination = destination / "Models" / model_id
    # The research asset tree also contains sparse coefficient .npy files. The
    # app runtime uses model.mil + weights.bin only; exclude those intermediates
    # from the portable kit.
    runtime_paths = ["model.mil", "weights.bin", "source_graph.json", "export_validation.json"]
    for directory in ("precision", "evidence"):
        root = source / directory
        if root.is_dir():
            runtime_paths.extend(str(p.relative_to(source)) for p in root.rglob("*") if p.is_file())
    for relative in runtime_paths:
        rel = safe_relative(relative)
        if rel.suffix.lower() in {".hwx", ".espressif", ".net", ".weights", ".shape"}:
            raise ValueError(f"{model_id}: raw firmware artifact is not allowed: {relative}")
        source_file = source / rel
        if not source_file.is_file():
            raise ValueError(f"{model_id}: missing runtime file: {relative}")
        copy_file(source_file, model_destination / rel)
    copy_file(manifest_path, model_destination / "manifest.json")
    # Normalize the portable research manifest into the app's runtime schema.
    for key in ("inputs", "outputs", "ane_inputs", "ane_outputs"):
        for port in manifest.get(key, []):
            port["bytes"] = 2 * int(__import__("math").prod(port["shape"]))
    origin_x = ((padded_width - 1920) // 4) * 2
    origin_y = 112 if padded_height == 1296 else ((padded_height - 1080) // 4) * 2
    manifest.update(
        version=2,
        width=1920,
        height=1080,
        padded_width=padded_width,
        padded_height=padded_height,
        origin_x=origin_x,
        origin_y=origin_y,
        fps=30,
        source_graph_sha256=digest(model_destination / "source_graph.json"),
    )
    # Recompute the runtime manifest list from the copied tree. This catches
    # accidental omissions and makes the manifest safe for an installer.
    copied_files = files_under(model_destination)
    manifest["files"] = {
        name: record["sha256"]
        for name, record in copied_files.items()
        if name != "manifest.json"
    }
    (model_destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "manifest_sha256": digest(model_destination / "manifest.json"),
        "source_model_sha256": manifest["source_model_sha256"],
        "files": dict(manifest["files"]),
    }


def copy_catalog(source: Path, destination: Path, selected_models: set[str]) -> dict:
    catalog_path = source / "catalog.json"
    if not catalog_path.is_file():
        raise ValueError(f"catalog.json not found under {source}")
    catalog = read_json(catalog_path)
    originals = catalog.get("models", [])
    if not isinstance(originals, list):
        raise ValueError("catalog.models must be an array")
    ids = [entry.get("id") for entry in originals]
    if len(ids) != len(set(ids)) or not selected_models.issubset(set(ids)):
        raise ValueError("catalog model IDs are missing or duplicated")

    selected_entries = [entry for entry in originals if entry.get("id") in selected_models]
    if len(selected_entries) != len(selected_models):
        raise ValueError("catalog selection is incomplete")
    selected_preset_ids = {
        preset_id
        for entry in selected_entries
        for preset_id in entry.get("preset_ids", [])
    }
    presets = catalog.get("presets", [])
    if not isinstance(presets, list):
        raise ValueError("catalog.presets must be an array")
    selected_presets = [preset for preset in presets if preset.get("id") in selected_preset_ids]
    found_preset_ids = {preset.get("id") for preset in selected_presets}
    if found_preset_ids != selected_preset_ids:
        raise ValueError("catalog preset selection is incomplete")

    destination_catalog = destination / "Catalog"
    destination_catalog.mkdir(parents=True, exist_ok=True)

    catalog_files = catalog.get("files", {})
    if not isinstance(catalog_files, dict):
        raise ValueError("catalog.files must be an object")
    wanted_files = {preset["file"] for preset in selected_presets if isinstance(preset.get("file"), str)}
    copied_files: dict[str, dict] = {}
    for relative in sorted(wanted_files):
        if relative not in catalog_files:
            raise ValueError(f"catalog file record missing: {relative}")
        record = catalog_files[relative]
        rel = safe_relative(relative)
        source_file = source / rel
        if not source_file.is_file() or source_file.is_symlink():
            raise ValueError(f"catalog file missing: {relative}")
        if source_file.stat().st_size != record.get("bytes") or digest(source_file) != record.get("sha256"):
            raise ValueError(f"catalog file hash mismatch: {relative}")
        copy_file(source_file, destination_catalog / rel)
        copied_files[relative] = record

    camera = source / "CameraSetup.plist"
    if camera.is_file() and not camera.is_symlink():
        relative = "CameraSetup.plist"
        if relative not in catalog_files:
            copy_file(camera, destination_catalog / relative)
            copied_files[relative] = {"sha256": digest(camera), "bytes": camera.stat().st_size}

    for entry in selected_entries:
        model_id = entry["id"]
        entry["runtime_directory"] = f"Models/{model_id}"
        entry["status"] = "ported_local"
        entry["status_detail"] = "完整模型已通过本机数值验证"
    catalog["models"] = selected_entries
    catalog["presets"] = selected_presets
    catalog["files"] = copied_files

    if catalog.get("default_model_id") not in selected_models:
        catalog["default_model_id"] = selected_entries[0]["id"]
    if len(selected_entries) != len(originals):
        catalog["scope"] = (
            f"Local LLVNR subset: {len(selected_entries)} of {len(originals)} catalog models; "
            "source data remains in the private research tree"
        )
    (destination_catalog / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    )
    return catalog


def build(args: argparse.Namespace) -> dict:
    assets = args.assets.resolve()
    catalog_source = args.catalog.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if not assets.is_dir() or not catalog_source.is_dir():
        raise ValueError("assets and catalog must be directories")

    source_catalog = read_json(catalog_source / "catalog.json")
    entries = source_catalog.get("models", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("catalog has no models")
    selected = {entry["id"] for entry in entries}
    missing = sorted(model_id for model_id in selected if not (assets / model_id).is_dir())
    if missing and not args.allow_partial:
        raise ValueError("missing model assets: " + ", ".join(missing))
    if missing:
        selected -= set(missing)
    if not selected:
        raise ValueError("no complete models selected")

    output.mkdir(parents=True)
    catalog = copy_catalog(catalog_source, output, selected)
    model_records: dict[str, dict] = {}
    for entry in catalog["models"]:
        model_id = entry["id"]
        if model_id not in selected:
            continue
        model_records[model_id] = copy_model(assets, model_id, entry, output)
        entry["runtime_manifest_sha256"] = model_records[model_id]["manifest_sha256"]
    # Rewrite once more after adding manifest hashes.
    (output / "Catalog" / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")

    if args.license and args.license.is_file():
        copy_file(args.license.resolve(), output / "LICENSE-ANEForge.txt")
    (output / "NOTICE.md").write_text(
        "# LLVNR Kit Notice\n\n"
        "This kit was produced locally from user-supplied assets. It is not a\n"
        "license to distribute Apple firmware, models, weights, tuning data, or\n"
        "other third-party material. The kit must only be used with material the\n"
        "creator is authorized to use. The runtime license, if supplied, applies\n"
        "only to the runtime code and not to model assets.\n")
    (output / "README.md").write_text(
        "# LLVNR Model Kit\n\n"
        "This is a self-contained local model kit for LLVNR Demo. It contains\n"
        "model runtime files supplied by its creator; no firmware download is\n"
        "performed by this kit.\n\n"
        "Install by selecting this directory in LLVNR Demo: Model -> Import Model Kit.\n"
        "Do not redistribute this directory unless all included assets are covered\n"
        "by an appropriate license.\n")

    all_files = files_under(output)
    kit_id = "llvnr-" + hashlib.sha256(
        json.dumps({k: v["sha256"] for k, v in sorted(all_files.items())}, sort_keys=True).encode()
    ).hexdigest()[:16]
    kit = {
        "schema_version": SCHEMA_VERSION,
        "kit_id": kit_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_ids": sorted(model_records),
        "catalog_sha256": digest(output / "Catalog" / "catalog.json"),
        "total_bytes": sum(record["bytes"] for record in all_files.values()),
        "files": all_files,
        "models": model_records,
        "notice": "User-generated model kit; assets may be subject to third-party rights.",
    }
    (output / "kit.json").write_text(json.dumps(kit, indent=2) + "\n")

    if not args.no_zip:
        archive = output.with_name(output.name + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    zf.write(path, Path(output.name) / path.relative_to(output))
        print(json.dumps({
            "kit": str(output),
            "zip": str(archive),
            "models": len(model_records),
            "bytes": kit["total_bytes"],
        }, indent=2))
    else:
        print(json.dumps({"kit": str(output), "models": len(model_records), "bytes": kit["total_bytes"]}, indent=2))
    return kit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True,
                        help="root containing one model directory per catalog entry")
    parser.add_argument("--catalog", type=Path, required=True,
                        help="directory containing catalog.json and optional Catalog assets")
    parser.add_argument("--output", type=Path, required=True,
                        help="new .llvnrkit directory")
    parser.add_argument("--license", type=Path, default=None,
                        help="optional runtime license file to include")
    parser.add_argument("--allow-partial", action="store_true",
                        help="build a kit containing only models present in --assets")
    parser.add_argument("--no-zip", action="store_true", help="do not create the sibling ZIP")
    args = parser.parse_args()
    try:
        build(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
