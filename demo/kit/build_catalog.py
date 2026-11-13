#!/usr/bin/env python3
"""Build a version-2 LLVNR Demo catalog from a private local extraction.

The input is the directory produced by ``extract_ipsw_assets.py``. This tool
does not download or decrypt firmware. It reads the user's local ``model.hwx``
and ``LLVTuningParameters.plist`` files, copies only the tuning files into the
catalog directory, and optionally binds a portable asset tree to the catalog.

The output is private model-derived material. Do not publish the generated
``catalog.json``, tuning plist files, or a kit unless you have the necessary
rights to the underlying Apple material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import plistlib
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


MODEL_RE = re.compile(
    r"^LLV-HD-(?P<lens>SuperWide|Wide|Tele)-(?P<zoom>\d+(?:\.\d+)?x)-"
    r"v(?P<version>\d+(?:\.\d+)*)_(?P<width>\d+)x(?P<height>\d+)$"
)
TUNING_SUFFIX = "-LLVTuningParameters"
FAMILY_LABEL = {"SuperWide": "超广角", "Wide": "广角", "Tele": "长焦"}
SUPPORTED_CANVASES = {(2304, 1296), (2464, 1388)}
DEFAULT_MODEL_ID = "LLV-HD-Wide-2x-v0.3.0_2304x1296"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe model or file component: {value!r}")
    return value


def model_identity(model_id: str) -> dict[str, Any]:
    match = MODEL_RE.fullmatch(model_id)
    if not match:
        raise ValueError(f"unsupported LLVNR model ID: {model_id!r}")
    values = match.groupdict()
    width, height = int(values["width"]), int(values["height"])
    if (width, height) not in SUPPORTED_CANVASES:
        raise ValueError(f"unsupported model canvas in {model_id!r}: {width}x{height}")
    name = model_id.rsplit("_", 1)[0]
    return {
        "id": model_id,
        "name": name,
        "family": values["lens"],
        "zoom": values["zoom"],
        "model_version": "v" + values["version"],
        "width": width,
        "height": height,
        "title": f"{FAMILY_LABEL[values['lens']]} {values['zoom']} · {width}x{height}",
    }


def finite_table(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 2 or len(value) % 2:
        raise ValueError(f"{field} must be an interleaved x/y list")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field} contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field} contains a non-finite value")
        result.append(number)
    return result


def modulation(value: Any, field: str) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise ValueError(f"missing {field} dictionary")
    result = {str(key): finite_table(table, f"{field}.{key}") for key, table in value.items()}
    if not result:
        raise ValueError(f"{field} is empty")
    return result


def read_tuning(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            document = plistlib.load(stream)
    except Exception as exc:
        raise ValueError(f"invalid plist: {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("LowLightVideoParameters"), dict):
        raise ValueError(f"missing LowLightVideoParameters: {path}")
    lowlight = document["LowLightVideoParameters"]
    model_info = lowlight.get("ModelInfo")
    if not isinstance(model_info, dict) or not isinstance(model_info.get("Name"), str):
        raise ValueError(f"missing ModelInfo.Name: {path}")
    model_name = model_info["Name"]
    if not model_name.startswith("LLV-HD-"):
        raise ValueError(f"invalid ModelInfo.Name in {path}: {model_name!r}")
    tuning_id = path.name.removesuffix(".plist")
    if not tuning_id.endswith(TUNING_SUFFIX):
        raise ValueError(f"unexpected tuning filename: {path.name}")
    return {
        "id": tuning_id,
        "name": tuning_id.removesuffix(TUNING_SUFFIX),
        "model_name": model_name,
        "input_modulation": modulation(lowlight.get("NetworkInputsModulation"), "NetworkInputsModulation"),
        "output_modulation": modulation(lowlight.get("NetworkOutputsModulation"), "NetworkOutputsModulation"),
        "source": path,
    }


def scan_models(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("model.hwx")):
        if path.is_symlink() or not path.is_file() or path.parent.name != "main_ane":
            continue
        bundle = next((part for part in path.parts if part.endswith(".bundle") and part.startswith("LLV-HD-")), None)
        if bundle is None:
            continue
        model_id = bundle.removesuffix(".bundle")
        model_identity(model_id)
        if model_id in found:
            raise ValueError(f"duplicate model ID: {model_id}")
        found[model_id] = path
    return found


def scan_tunings(root: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob(f"*{TUNING_SUFFIX}.plist")):
        if path.is_symlink() or not path.is_file():
            continue
        tuning = read_tuning(path)
        if tuning["id"] in found:
            raise ValueError(f"duplicate tuning ID: {tuning['id']}")
        found[tuning["id"]] = tuning
    return found


def extract_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "EXTRACT-MANIFEST.json"
    return read_json(path) if path.is_file() else None


def firmware_build_from_paths(paths: Iterable[Path], root: Path) -> str | None:
    for path in paths:
        for part in path.relative_to(root).parts:
            if "__" in part and part.split("__", 1)[0]:
                return part.split("__", 1)[0]
    return None


def variant_from_manifest(manifest: dict[str, Any] | None, root: Path, model_paths: Iterable[Path] = ()) -> str | None:
    if manifest and isinstance(manifest.get("variant"), str) and manifest["variant"] != "all":
        return manifest["variant"]
    for part in root.parts:
        if part in {"V63", "V64", "V64s"}:
            return part
    for path in model_paths:
        for part in path.parts:
            if part in {"V63", "V64", "V64s"}:
                return part
    return None


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def asset_metadata(assets: Path | None, model_id: str, source_hash: str) -> dict[str, Any]:
    if assets is None:
        return {"runnable": False, "manifest": None, "report": None}
    folder = assets / safe_component(model_id)
    manifest_path = folder / "manifest.json"
    report_path = folder / "export_validation.json"
    if not manifest_path.is_file():
        return {"runnable": False, "manifest": None, "report": None}
    manifest = read_json(manifest_path)
    if manifest.get("model_id") != model_id:
        raise ValueError(f"asset manifest model_id mismatch: {model_id}")
    if manifest.get("source_model_sha256") != source_hash:
        raise ValueError(f"asset source model hash mismatch: {model_id}")
    report = read_json(report_path) if report_path.is_file() else None
    runnable = report is not None and report.get("passed") is True
    return {
        "runnable": runnable,
        "manifest": manifest,
        "report": report,
        "manifest_sha256": sha256(manifest_path) if runnable else None,
    }


def output_port_names(manifest: dict[str, Any] | None, source_graph: Path | None) -> list[str]:
    if manifest and isinstance(manifest.get("outputs"), list):
        names = [port.get("name") for port in manifest["outputs"] if isinstance(port, dict)]
        if names and all(isinstance(name, str) for name in names):
            return sorted(set(names))
    if source_graph and source_graph.is_file():
        plan = read_json(source_graph)
        outputs = plan.get("outputs")
        if isinstance(outputs, list) and all(isinstance(name, str) for name in outputs):
            return sorted(set(outputs))
    return []


def build_catalog(
    extracted: Path,
    assets: Path | None,
    output: Path,
    *,
    variant: str | None = None,
    firmware_build: str | None = None,
    camera_setup: Path | None = None,
    allow_partial: bool = False,
    allow_unported: bool = False,
    selected_models: list[str] | None = None,
) -> dict[str, Any]:
    extracted = extracted.resolve()
    assets = assets.resolve() if assets else None
    output = output.resolve()
    if not extracted.is_dir():
        raise ValueError(f"extraction directory not found: {extracted}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output}")

    manifest = extract_manifest(extracted)
    models = scan_models(extracted)
    if selected_models:
        missing = sorted(set(selected_models) - set(models))
        if missing:
            raise ValueError("requested model IDs were not found: " + ", ".join(missing))
        models = {model_id: models[model_id] for model_id in selected_models}
    selected_names = {model_identity(model_id)["name"] for model_id in models}
    tunings = {key: value for key, value in scan_tunings(extracted).items() if value["model_name"] in selected_names}
    if not models:
        raise ValueError("no LLV-HD model.hwx files found")
    if not tunings:
        raise ValueError("no LLVTuningParameters.plist files found")

    detected_variant = variant or variant_from_manifest(manifest, extracted, models.values())
    if detected_variant is None:
        raise ValueError("cannot determine ISPKit variant; pass --variant")
    if detected_variant not in {"V63", "V64", "V64s"}:
        raise ValueError(f"unsupported variant: {detected_variant}")

    presets_by_model: dict[str, list[dict[str, Any]]] = {}
    for tuning in tunings.values():
        presets_by_model.setdefault(tuning["model_name"], []).append(tuning)

    output.mkdir(parents=True, exist_ok=True)
    catalog_dir = output
    tunings_dir = catalog_dir / "Tunings"
    tunings_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}

    preset_records: list[dict[str, Any]] = []
    for tuning in sorted(tunings.values(), key=lambda row: row["id"]):
        destination = tunings_dir / f"{tuning['id']}.plist"
        shutil.copy2(tuning["source"], destination)
        relative = str(destination.relative_to(catalog_dir))
        files[relative] = file_record(destination)
        preset_records.append({
            "id": tuning["id"],
            "name": tuning["name"],
            "model_name": tuning["model_name"],
            "file": relative,
            "input_modulation": tuning["input_modulation"],
            "output_modulation": tuning["output_modulation"],
        })

    if camera_setup is not None:
        camera_setup = camera_setup.resolve()
        if not camera_setup.is_file():
            raise ValueError(f"CameraSetup.plist not found: {camera_setup}")
        destination = catalog_dir / "CameraSetup.plist"
        shutil.copy2(camera_setup, destination)
        files["CameraSetup.plist"] = file_record(destination)

    entries: list[dict[str, Any]] = []
    for model_id, model_path in sorted(models.items()):
        identity = model_identity(model_id)
        model_tunings = sorted(presets_by_model.get(identity["name"], []), key=lambda row: row["id"])
        if not model_tunings:
            if allow_partial:
                continue
            raise ValueError(f"no tuning plist maps to model {identity['name']}")
        metadata = asset_metadata(assets, model_id, sha256(model_path))
        if assets is not None and not metadata["runnable"] and not allow_unported:
            raise ValueError(f"model has no passing exported runtime assets: {model_id}")
        source_graph = assets / model_id / "source_graph.json" if assets else None
        manifest_row = metadata["manifest"]
        task_count = int(manifest_row.get("node_count", 0)) if manifest_row else 0
        coefficient_groups = int(manifest_row.get("coefficient_group_count", 0)) if manifest_row else 0
        lut_groups = 0
        if source_graph and source_graph.is_file():
            plan = read_json(source_graph)
            lut_groups = sum(1 for node in plan.get("nodes", []) if isinstance(node, dict) and node.get("activation_lut_hex"))
        runnable = bool(metadata["runnable"])
        entries.append({
            **identity,
            "bytes": model_path.stat().st_size,
            "sha256": sha256(model_path),
            "original_file": f"Originals/{model_id}.bundle/H19.bundle/main/main_ane/model.hwx",
            "task_count": task_count,
            "ne_groups": coefficient_groups,
            "lut_groups": lut_groups,
            "output_names": output_port_names(manifest_row, source_graph),
            "preset_ids": [tuning["id"] for tuning in model_tunings],
            "runtime_directory": f"Models/{model_id}" if runnable else None,
            "status": "ported_local" if runnable else "source_only",
            "status_detail": "本地运行资产已通过导出校验" if runnable else "仅有提取物；尚未生成通过校验的运行资产",
            "runtime_manifest_sha256": metadata.get("manifest_sha256"),
        })

    if not entries:
        raise ValueError("catalog would contain no models")
    ids = [entry["id"] for entry in entries]
    runnable_ids = [entry["id"] for entry in entries if entry["status"] == "ported_local"]
    default_id = DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in runnable_ids else (runnable_ids[0] if runnable_ids else ids[0])
    detected_build = firmware_build or firmware_build_from_paths(models.values(), extracted) or "unknown"
    catalog = {
        "version": 2,
        "platform": detected_variant,
        "firmware_build": detected_build,
        "default_model_id": default_id,
        "scope": f"{len(entries)} {detected_variant} rear-camera LLVNR models from a local extraction; no front-camera LLVNR model",
        "models": entries,
        "presets": preset_records,
        "files": dict(sorted(files.items())),
    }
    (catalog_dir / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    return {
        "catalog": str(catalog_dir / "catalog.json"),
        "models": len(entries),
        "runnable_models": len(runnable_ids),
        "presets": len(preset_records),
        "files": len(files),
        "platform": detected_variant,
        "firmware_build": detected_build,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extracted", type=Path, required=True, help="private extraction directory from extract_ipsw_assets.py")
    parser.add_argument("--output", type=Path, required=True, help="new catalog directory")
    parser.add_argument("--assets", type=Path, help="optional portable runtime asset root")
    parser.add_argument("--variant", choices=("V63", "V64", "V64s"), help="ISPKit variant override")
    parser.add_argument("--firmware-build", help="firmware build label override")
    parser.add_argument("--camera-setup", type=Path, help="optional local CameraSetup.plist to copy")
    parser.add_argument("--allow-partial", action="store_true", help="drop models with no tuning mapping")
    parser.add_argument("--allow-unported", action="store_true", help="allow catalog entries without passing runtime assets")
    parser.add_argument("--models", nargs="+", help="optional model IDs to include")
    args = parser.parse_args()
    try:
        result = build_catalog(
            args.extracted,
            args.assets,
            args.output,
            variant=args.variant,
            firmware_build=args.firmware_build,
            camera_setup=args.camera_setup,
            allow_partial=args.allow_partial,
            allow_unported=args.allow_unported,
            selected_models=args.models,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
