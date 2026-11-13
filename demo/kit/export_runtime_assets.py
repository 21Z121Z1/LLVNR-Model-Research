#!/usr/bin/env python3
"""Export portable LLVNR runtime assets from user-provided portable source graphs.

This is the public, reproducible bridge between a user's local graph/coefficient
research output and a ``.llvnrkit``. It never reads firmware, never downloads
anything, and never writes outside ``--output``. The source graph format is the
same JSON format used by the research scripts; coefficient arrays must be
provided by the user and are copied only into the private output tree.

The exporter emits the files consumed by the App:

* ``model.mil`` and ``weights.bin`` (via the vendored ANEForge compiler subset);
* ``source_graph.json`` with normalized relative coefficient paths;
* ``manifest.json`` and, unless ``--validation none`` is used, a real
  ``export_validation.json`` comparing CPU and ANE execution on synthetic input.

The generated tree is private model-derived material and must not be published
without confirming the rights to the underlying Apple material.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "research").is_dir():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parents[1] / "research").is_dir():
    ROOT = SCRIPT_DIR.parents[1]
else:
    ROOT = SCRIPT_DIR.parent
THIRD_PARTY = ROOT / "third_party" / "ANEForge"
if not THIRD_PARTY.is_dir():
    THIRD_PARTY = SCRIPT_DIR / "third_party" / "ANEForge"
GRAPH_DIR = ROOT / "research" / "work" / "llvnr_port" / "graph"
if not GRAPH_DIR.is_dir():
    GRAPH_DIR = SCRIPT_DIR / "runtime"
sys.path[:0] = [str(THIRD_PARTY), str(GRAPH_DIR)]

from aneforge._compile import _EMIT, _Emitter, _emit_program_dir, _topo_multi  # noqa: E402
from portable_graph import build_ane, checked_inputs, evaluate_cpu, load_coefficients  # noqa: E402

EXPECTED_INPUTS = {"cur_frame", "his_frame", "dnr_zoom_map", "fusion_zoom_map"}
EXPECTED_OUTPUTS = {"out_yuv", "state_yuv"}
DEFAULT_SEED = 20261112


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


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def resolve_coefficient(source_dir: Path, coefficient_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    if Path(value).is_absolute():
        raise ValueError(f"coefficient paths must be relative: {value}")
    candidates = [
        source_dir / value,
        coefficient_root / value,
        coefficient_root / source_dir.name / value,
    ]
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise FileNotFoundError(
        f"coefficient not found: {value}; tried {', '.join(str(path) for path in candidates)}"
    )


def normalize_plan(plan: dict[str, Any], source_dir: Path, coefficient_root: Path,
                   destination: Path) -> dict[str, Any]:
    result = copy.deepcopy(plan)
    coefficients = result.get("coefficients")
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError(f"{source_dir}: source graph has no coefficients")
    normalized: dict[str, dict[str, Any]] = {}
    for group, record in coefficients.items():
        if not isinstance(group, str) or not group:
            raise ValueError(f"{source_dir}: invalid coefficient group")
        if not isinstance(record, dict):
            raise ValueError(f"{source_dir}: coefficient record must be an object: {group}")
        weights_source = resolve_coefficient(source_dir, coefficient_root, record.get("weights"))
        bias_source = resolve_coefficient(source_dir, coefficient_root, record.get("bias"))
        if weights_source is None:
            raise ValueError(f"{source_dir}: missing weights path for {group}")
        weight_folder = destination / "coefficients" / group
        weight_folder.mkdir(parents=True, exist_ok=True)
        weights_destination = weight_folder / "weights.npy"
        shutil.copy2(weights_source, weights_destination)
        record = dict(record)
        record["weights"] = str(weights_destination.relative_to(destination))
        if bias_source is not None:
            bias_destination = weight_folder / "bias.npy"
            shutil.copy2(bias_source, bias_destination)
            record["bias"] = str(bias_destination.relative_to(destination))
        else:
            record["bias"] = None
        normalized[group] = record
    result["coefficients"] = normalized
    result.setdefault("schema_version", 1)
    if not isinstance(result.get("outputs"), list) or not result["outputs"]:
        raise ValueError(f"{source_dir}: source graph has no outputs")
    input_names = {node.get("name") for node in result.get("nodes", []) if node.get("op") == "input"}
    if input_names != EXPECTED_INPUTS:
        raise ValueError(f"{source_dir}: expected inputs {sorted(EXPECTED_INPUTS)}, got {sorted(input_names)}")
    if not EXPECTED_OUTPUTS.issubset(set(result["outputs"])):
        raise ValueError(f"{source_dir}: missing outputs {sorted(EXPECTED_OUTPUTS - set(result['outputs']))}")
    return result


def synthetic_inputs(plan: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for node in plan.get("nodes", []):
        if node.get("op") != "input":
            continue
        name, shape = node["name"], tuple(int(value) for value in node["shape"])
        if name in {"dnr_zoom_map", "fusion_zoom_map"}:
            result[name] = np.full(shape, 0.5, dtype=np.float16)
        else:
            image = rng.normal(0.2, 0.015, shape).astype(np.float32)
            if len(shape) >= 2 and shape[1] >= 4:
                image[:, 4:] -= 0.2
            result[name] = image.astype(np.float16)
    return checked_inputs(plan, result)


def port_record(name: str, port: str, shape: list[int]) -> dict[str, Any]:
    count = math.prod(shape)
    return {"name": name, "port": port, "shape": shape, "bytes": count * 2}


def build_program(plan: dict[str, Any], coefficients: dict[str, Any], destination: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution_plan = plan
    if plan.get("execution_profile") == "metal_state_ane_denoiser":
        from metal_prefix import export_prefix
        _, execution_plan = export_prefix(plan, coefficients, destination / "precision")
    logical_graph = build_ane(plan, coefficients)
    execution_graph = build_ane(execution_plan, coefficients)
    missing_inputs = [node["name"] for node in execution_plan.get("nodes", []) if node.get("op") == "input" and node["name"] not in execution_graph]
    if missing_inputs:
        raise ValueError("execution graph contains unused inputs: " + ", ".join(missing_inputs))
    outputs = [execution_graph[name] for name in execution_plan["outputs"]]
    order = _topo_multi(*outputs)
    for index, tensor in enumerate(order):
        tensor._name = f"t{index}"
    emitter = _Emitter(int8=False)
    for tensor in order:
        if tensor.op != "input":
            _EMIT[tensor.op](emitter, tensor, tensor._name, [source._name for source in tensor.srcs])
    inputs = sorted((tensor for tensor in order if tensor.op == "input"), key=lambda tensor: tensor.attrs.get("idx", 0))
    _emit_program_dir(emitter, inputs, ", ".join(tensor._name for tensor in outputs), None, destination)
    return logical_graph, execution_graph, execution_plan


def manifest_from_plan(plan: dict[str, Any], logical_graph: dict[str, Any],
                       execution_graph: dict[str, Any], execution_plan: dict[str, Any],
                       destination: Path, model_id: str, source_graph_sha256: str) -> dict[str, Any]:
    if not isinstance(plan.get("model_sha256"), str) or len(plan["model_sha256"]) != 64:
        raise ValueError(f"{model_id}: source graph has no model_sha256")
    current = next(node for node in plan["nodes"] if node.get("name") == "cur_frame")
    shape = [int(value) for value in current["shape"]]
    if len(shape) != 4:
        raise ValueError(f"{model_id}: cur_frame must be four-dimensional")
    padded_width, padded_height = shape[3] * 2, shape[2] * 2
    if (padded_width, padded_height) not in {(2304, 1296), (2464, 1388)}:
        raise ValueError(f"{model_id}: unsupported canvas {padded_width}x{padded_height}")

    external_inputs = [node for node in plan["nodes"] if node.get("op") == "input"]
    # The execution graph may add Metal-prefix ports. Its tensor names are stable
    # after build_program renamed them, so derive ports from graph tensors.
    input_records = []
    for node in external_inputs:
        tensor = logical_graph[node["name"]]
        input_records.append(port_record(node["name"], node["name"], list(tensor.shape)))
    output_records = []
    for name in plan["outputs"]:
        tensor = logical_graph[name]
        output_records.append(port_record(name, name, list(tensor.shape)))
    ane_inputs = [port_record(node["name"], execution_graph[node["name"]]._name, list(execution_graph[node["name"]].shape))
                  for node in execution_plan["nodes"] if node.get("op") == "input"]
    ane_outputs = [port_record(name, execution_graph[name]._name, list(execution_graph[name].shape))
                   for name in execution_plan["outputs"]]

    files = {str(path.relative_to(destination)): sha256(path) for path in sorted(destination.rglob("*"))
             if path.is_file() and path.name not in {"manifest.json", "export_validation.json"}}
    manifest = {
        "model_id": model_id,
        "source_model_sha256": plan["model_sha256"],
        "source_graph_sha256": source_graph_sha256,
        "node_count": len(plan.get("nodes", [])),
        "coefficient_group_count": len(plan.get("coefficients", {})),
        "execution_profile": plan.get("execution_profile", "ane"),
        "inputs": input_records,
        "outputs": output_records,
        "ane_inputs": ane_inputs,
        "ane_outputs": ane_outputs,
        "files": files,
        "evidence": [],
        "scope": "recovered graph executed locally; original H19 hardware parity is separate",
        "known_failed_evidence": [],
        **host_target(),
        "padded_width": padded_width,
        "padded_height": padded_height,
        "width": 1920,
        "height": 1080,
        "origin_x": ((padded_width - 1920) // 4) * 2,
        "origin_y": 112 if padded_height == 1296 else ((padded_height - 1080) // 4) * 2,
        "fps": 30,
        "version": 2,
    }
    return manifest


def host_target() -> dict[str, Any]:
    """Return a known host ANE compiler target, or an empty map if unknown."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return {}
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        match = re.search(r"Apple M(\d+)", brand)
        generation = int(match.group(1)) if match else 0
        family_by_generation = {1: 2, 2: 3, 3: 4, 4: 5, 5: 5}
        arch_by_family = {2: "h13", 3: "h14", 4: "h15", 5: "h16s"}
        family = family_by_generation.get(generation)
        if family is None:
            return {}
        return {"target_family": family, "target_arch": arch_by_family[family]}
    except Exception:
        # The ANE validation path will fail closed below. A structural CPU/none
        # export can still be inspected without a target.
        return {}


def output_summary(outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "finite": bool(np.isfinite(value).all()),
            "sha256_fp16": hashlib.sha256(value.astype(np.float16).tobytes()).hexdigest(),
        }
        for name, value in outputs.items()
    }


def ensure_precision_dylib(destination: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"precision dylib not found: {explicit}")
        return explicit.resolve()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Metal precision validation requires Apple Silicon macOS; pass --validation cpu or none")
    output = destination / "libllvnr_precision.dylib"
    source = ROOT / "common" / "PrecisionRuntime.mm"
    if not source.is_file():
        source = SCRIPT_DIR / "common" / "PrecisionRuntime.mm"
    if not source.is_file():
        raise FileNotFoundError(f"precision source not found: {source}")
    command = ["xcrun", "clang++", "-O2", "-dynamiclib", "-fPIC", "-fobjc-arc",
               "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to build precision dylib: {result.stderr.strip()}")
    return output


def validate_assets(plan: dict[str, Any], coefficients: dict[str, Any], destination: Path,
                    manifest: dict[str, Any], validation: str, seed: int,
                    precision_dylib: Path | None) -> dict[str, Any]:
    if validation == "none":
        return {}
    inputs = synthetic_inputs(plan, seed)
    cpu_values = evaluate_cpu(plan, coefficients, inputs)
    cpu = {
        "model_id": manifest["model_id"],
        "backend": "cpu",
        "source_model_sha256": manifest["source_model_sha256"],
        "outputs": output_summary({name: cpu_values[name] for name in plan["outputs"]}),
    }
    if validation == "cpu":
        report = {
            "model_id": manifest["model_id"],
            "cpu": cpu,
            "metrics": {},
            "passed": False,
            "scope": "CPU-only structural validation; ANE execution was intentionally skipped",
        }
        (destination / "export_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("ANE validation requires Apple Silicon macOS; pass --validation cpu or none")
    if not host_target():
        raise RuntimeError("cannot identify a supported Apple M-series ANE target; pass --validation cpu or none")
    explicit = os.environ.get("ANEFORGE_DYLIB")
    if explicit and Path(explicit).is_file():
        dylib = Path(explicit)
    else:
        cache = Path(os.environ.get("LLVNR_ANE_DYLIB_CACHE", str(Path.home() / "Library/Caches/LLVNRModelResearch/ANEForge")))
        dylib = cache / "libane_e5rt_dispatch.dylib"
        if not dylib.is_file():
            from aneforge.build import build_dylib
            cache.mkdir(parents=True, exist_ok=True)
            dylib = build_dylib(dylib)
        os.environ["ANEFORGE_DYLIB"] = str(dylib)
    from aneforge._runtime import E5RT
    feeds = inputs
    precision = None
    if plan.get("execution_profile") == "metal_state_ane_denoiser":
        from precision_runtime import MetalPrefix
        library = ensure_precision_dylib(destination, precision_dylib)
        precision = MetalPrefix(destination / "precision", library)
        feeds = {**inputs, **precision.run(inputs)}
    try:
        in_ports = {port["port"]: tuple(port["shape"]) for port in manifest["ane_inputs"]}
        out_ports = {port["port"]: tuple(port["shape"]) for port in manifest["ane_outputs"]}
        with E5RT.compile(destination / "model.mil", cache_dir=destination / "cache",
                          inputs=in_ports, outputs=out_ports) as program:
            actual = program.eval({port["port"]: feeds[port["name"]] for port in manifest["ane_inputs"]})
            ane_values = {
                port["name"]: np.array(actual[port["port"]], dtype=np.float16, copy=True)
                for port in manifest["ane_outputs"]
            }
    finally:
        if precision is not None:
            precision.close()
    external_ane_values = {name: ane_values.get(name, feeds.get(name)) for name in plan["outputs"]}
    if any(value is None for value in external_ane_values.values()):
        raise RuntimeError("ANE result is missing an external output")
    ane = {
        "model_id": manifest["model_id"],
        "backend": "ane",
        "source_model_sha256": manifest["source_model_sha256"],
        "outputs": output_summary(external_ane_values),
    }
    metrics: dict[str, Any] = {}
    for name in plan["outputs"]:
        expected, observed = cpu_values[name].astype(np.float32), external_ane_values[name].astype(np.float32)
        if expected.shape != observed.shape:
            metrics[name] = {"shape_equal": False, "passed": False}
            continue
        error = np.abs(expected - observed)
        maximum, mean = float(error.max()), float(error.mean())
        metrics[name] = {
            "max_abs": maximum,
            "mean_abs": mean,
            "finite": bool(np.isfinite(observed).all()),
            "passed": bool(np.isfinite(observed).all() and maximum <= 0.02 and mean <= 0.001),
        }
    report = {
        "model_id": manifest["model_id"],
        "cpu": cpu,
        "ane": ane,
        "metrics": metrics,
        "passed": all(row.get("passed") is True for row in metrics.values()),
        "scope": "exported-file CPU and direct MIL ANE execution on synthetic input; not original H19 hardware or temporal acceptance",
    }
    (destination / "export_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_one(source_dir: Path, output_root: Path, coefficient_root: Path, validation: str,
               seed: int, precision_dylib: Path | None) -> dict[str, Any]:
    source_graph = source_dir / "source_graph.json"
    if not source_graph.is_file():
        raise ValueError(f"missing source_graph.json: {source_dir}")
    model_id = source_dir.name
    if model_id.startswith("LLV-HD-") is False:
        raise ValueError(f"model directory name must start with LLV-HD-: {model_id}")
    destination = output_root / model_id
    destination.mkdir(parents=True, exist_ok=False)
    plan = normalize_plan(read_json(source_graph), source_dir, coefficient_root, destination)
    source_graph_sha256 = hashlib.sha256((json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    (destination / "source_graph.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    coefficients = load_coefficients(plan, destination)
    logical_graph, execution_graph, execution_plan = build_program(plan, coefficients, destination)
    manifest = manifest_from_plan(plan, logical_graph, execution_graph, execution_plan, destination, model_id, source_graph_sha256)
    # Recompute file hashes after the program and precision assets exist.
    manifest["files"] = {
        str(path.relative_to(destination)): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "export_validation.json"}
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = validate_assets(plan, coefficients, destination, manifest, validation, seed, precision_dylib)
    return {
        "model_id": model_id,
        "source_graph": str(source_graph),
        "manifest": str(destination / "manifest.json"),
        "validation": report.get("passed", None) if report else None,
    }


def discover_sources(root: Path, models: list[str] | None) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"portable source root is not a directory: {root}")
    if models:
        result = [root / model for model in models]
    else:
        result = sorted(path for path in root.iterdir() if path.is_dir() and (path / "source_graph.json").is_file())
    if not result:
        raise ValueError(f"no model source graphs found under {root}")
    for path in result:
        if not (path / "source_graph.json").is_file():
            raise ValueError(f"model source has no source_graph.json: {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, required=True, help="root containing MODEL_ID/source_graph.json")
    parser.add_argument("--coefficient-root", type=Path, help="root used to resolve relative coefficient paths (default: --sources)")
    parser.add_argument("--output", type=Path, required=True, help="new private portable asset root")
    parser.add_argument("--models", nargs="+", help="optional model IDs to export")
    parser.add_argument("--validation", choices=("ane", "cpu", "none"), default="ane",
                        help="ane = require local ANE CPU/ANE comparison; cpu = structural CPU report only; none = no report")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic synthetic input seed")
    parser.add_argument("--precision-dylib", type=Path, help="optional locally built libllvnr_precision.dylib")
    args = parser.parse_args()
    try:
        sources = args.sources.resolve()
        coefficient_root = (args.coefficient_root or sources).resolve()
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"output directory must be new or empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        results = [
            export_one(source, output, coefficient_root, args.validation, args.seed, args.precision_dylib)
            for source in discover_sources(sources, args.models)
        ]
        summary = {"assets": str(output), "models": len(results), "validation": args.validation, "results": results}
        (output / "EXPORT.json").write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
