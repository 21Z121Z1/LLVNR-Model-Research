#!/usr/bin/env python3
"""Recover private portable source graphs from a local LLVNR extraction.

The public recovery path is intentionally self-contained: it reads the user's
local ``model.hwx`` files and a repository-contained, sanitized set of compiler
layout metadata.  No Apple model weights, tuning files, probe HWX files, or
private research paths are shipped in the repository.

The output is private model-derived material.  It contains coefficient arrays
and source graphs and must not be published as part of the code repository.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np

KIT = pathlib.Path(__file__).resolve().parent
if (KIT / "research").is_dir():
    ROOT = KIT
elif (KIT.parents[1] / "research").is_dir():
    ROOT = KIT.parents[1]
else:
    ROOT = KIT.parent
RESEARCH = ROOT / "research" / "work"
PORT = RESEARCH / "llvnr_port"
GRAPH = PORT / "graph"
for path in (RESEARCH, PORT, GRAPH):
    sys.path.insert(0, str(path))

from h19_deconv_probe import skeleton  # noqa: E402
from h19_weightlib import kernel_groups, kernel_section, parse_hwx  # noqa: E402
from llvnr_ir import export  # noqa: E402
from llvnr_taskflow import address  # noqa: E402
import recover_catalog as catalog_recovery  # noqa: E402
import sparse_transpose_codec  # noqa: E402
from recovered_reference import RecoveredReference, load_profiles  # noqa: E402
import portable_graph  # noqa: E402
import wide1_graph  # noqa: E402
import superwide05_graph  # noqa: E402
import tele4_graph  # noqa: E402
import tele8_graph  # noqa: E402
import large_canvas_graph  # noqa: E402

MODEL_RE = re.compile(
    r"^LLV-HD-(?P<lens>SuperWide|Wide|Tele)-(?P<zoom>\d+(?:\.\d+)?x)-"
    r"v(?P<version>\d+(?:\.\d+)*)_(?P<width>\d+)x(?P<height>\d+)$"
)
WIDE2_SMALL = "LLV-HD-Wide-2x-v0.3.0_2304x1296"
TELE4_SMALL = "LLV-HD-Tele-4x-v0.3.0_2304x1296"
WIDE1_MODELS = set(wide1_graph.MODELS)
SUPERWIDE05_MODELS = set(superwide05_graph.MODELS)
TELE8_MODELS = set(tele8_graph.MODELS)
LARGE_MODELS = set(large_canvas_graph.MODELS)
TELE4_MODELS = {
    "LLV-HD-Tele-4x-v0.3.0_2304x1296",
    "LLV-HD-Tele-4x-v0.3.0_2464x1388",
}
ALL_MODELS = WIDE1_MODELS | SUPERWIDE05_MODELS | TELE8_MODELS | LARGE_MODELS | TELE4_MODELS | {
    "LLV-HD-Wide-2x-v0.3.0_2304x1296",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def discover_models(extracted: pathlib.Path) -> dict[str, pathlib.Path]:
    found: dict[str, pathlib.Path] = {}
    for path in sorted(extracted.rglob("model.hwx")):
        if len(path.parents) < 4:
            continue
        bundle = path.parents[3].name
        if not bundle.endswith(".bundle"):
            continue
        model_id = bundle.removesuffix(".bundle")
        if not MODEL_RE.fullmatch(model_id):
            continue
        if model_id in found:
            raise ValueError(f"duplicate model bundle: {model_id}")
        found[model_id] = path
    return found


def dependencies(model_ids: set[str]) -> set[str]:
    required = set(model_ids)
    if any(model in LARGE_MODELS or model in TELE4_MODELS for model in required):
        required.add(WIDE2_SMALL)
    if any(model in LARGE_MODELS and model.startswith("LLV-HD-Tele-4x-") for model in required):
        required.add(TELE4_SMALL)
    return required


def matching_tasks(ir: dict[str, Any], group) -> list[dict[str, Any]]:
    offset = min(group.starts)
    return [
        task for task in ir["tasks"]
        if any(packet["register_word"] == 0x1544 and packet["bar"] == ir["kernel_bar"]
               and packet["offset"] == offset for packet in task["address_writes"])
    ]


def recover_transpose(group, task, profiles: dict[str, Any]):
    dims = task["explicit_dimensions_whc"]
    cin, cout = dims["input"][2] or 16, dims["output"][2] or 16
    geometry = task["geometry"] or {}
    if max(geometry.get("output_x_factor", 1), geometry.get("output_y_factor", 1)) != 2:
        return None
    active = 1 << ((task["written_registers"].get("000f", 0) >> 19) & 7)
    for layout in profiles.get("transpose_layouts", []):
        if layout.get("shape", [])[:2] != [cin, cout]:
            continue
        if active not in (layout.get("controlled_active_ne") or []):
            continue
        size = layout.get("region_bytes")
        locations = layout.get("weight_byte_locations")
        if not size or not locations:
            continue
        native = group.raw[:size]
        if len(native) != size:
            continue
        offsets = [offset for row in locations for offset in row]
        template = skeleton(native, offsets)
        if sha256(template) != layout.get("skeleton_sha256"):
            continue
        if any(len({native[offset:offset + 2] for offset in row}) != 1 for row in locations):
            continue
        bits = np.array([int.from_bytes(native[row[0]:row[0] + 2], "little")
                         for row in locations], dtype=np.uint16)
        weights = bits.view(np.float16).reshape(layout["shape"])
        if not np.isfinite(weights).all():
            continue
        rebuilt = bytearray(template)
        for positions, value in zip(locations, bits):
            for offset in positions:
                rebuilt[offset:offset + 2] = int(value).to_bytes(2, "little")
        if bytes(rebuilt) != native:
            raise ValueError("transpose native round-trip failed")
        return {
            "weights": weights,
            "bias": None,
            "primitive": "conv_transpose_then_lut" if layout.get("embedded_activation") else "conv_transpose",
            "region": native,
            "source": "embedded:transpose_layouts",
            "pad_for_probe": layout.get("pad", [0, 0, 0, 0]),
            "native_origin_top_left": layout.get("native_origin_top_left"),
            "native_origin_by_geometry_padding": layout.get("native_origin_by_geometry_padding"),
        }
    return None


def recover_sparse_transpose(group, task, raw: bytes, profiles: dict[str, Any]):
    for record in profiles.get("sparse_transpose_layouts", []):
        if record.get("shape", [])[:2] != [
            task["explicit_dimensions_whc"]["input"][2] or 16,
            task["explicit_dimensions_whc"]["output"][2] or 16,
        ]:
            continue
        if record.get("geometry") != task.get("geometry"):
            continue
        try:
            result = sparse_transpose_codec.recover(
                group, task, raw, record["shape"], record["origin"])
            result["source"] = "embedded:sparse_transpose_layouts"
            return result
        except (ValueError, IndexError):
            continue
    return None


def recover_special(group, task):
    """Recover compiler-emitted identity/replication/Haar coefficients.

    These groups are structural filters rather than learned tensors.  The
    patterns are identified from task dimensions and are re-encoded from
    constants, so no model-specific coefficient array is embedded in the
    repository.
    """
    dims = task["explicit_dimensions_whc"]
    cin, cout = dims["input"][2] or 16, dims["output"][2] or 16
    geometry = task["geometry"] or {}
    xf, yf = geometry.get("output_x_factor", 1), geometry.get("output_y_factor", 1)
    if (cin == cout == 1 and (xf, yf) == (2, 2) and len(group.raw) >= 128
            and group.raw[:4] == bytes.fromhex("0000003c")):
        return {
            "weights": np.ones((1, 1, 2, 2), dtype=np.float16),
            "bias": None,
            "primitive": "nearest_replication",
            "region": group.raw[:128],
            "source": "structural_special:nearest_replication",
        }
    if (cin == cout == 1 and (xf, yf) == (1, 1) and len(group.raw) >= 128
            and group.raw[:4] == bytes.fromhex("003c0000")):
        input_hw = tuple(dims["input"][:2])
        output_hw = tuple(dims["output"][:2])
        if output_hw[0] >= input_hw[0] and output_hw[1] >= input_hw[1] and output_hw != input_hw:
            return {
                "weights": np.ones((1, 1, 1, 1), dtype=np.float16),
                "bias": None,
                "primitive": "padding_identity",
                "region": group.raw[:128],
                "source": "structural_special:padding_identity",
            }
    if ((cin, cout) == (24, 6) and (xf, yf) == (2, 2)
            and geometry.get("kh") == 2 and geometry.get("kw") == 1
            and len(group.raw) >= 384 and group.raw[:4] == bytes.fromhex("380000c3")):
        weights = np.zeros((24, 6, 2, 2), dtype=np.float16)
        patterns = (
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float16),
            np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float16),
            np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float16),
            np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float16),
        )
        for co in range(24):
            weights[co, co % 6] = patterns[co // 6]
        return {
            "weights": weights,
            "bias": None,
            "primitive": "conv_transpose_haar",
            "region": group.raw[:384],
            "source": "structural_special:haar",
        }
    return None


def recover_padding(group, task, raw: bytes, profiles: dict[str, Any]):
    channels = task["explicit_dimensions_whc"]["input"][2] or 16
    origin = min(group.starts)
    for probe in profiles.get("padding_layouts", []):
        if probe.get("input_shape", [0, 0])[1] != channels:
            continue
        if probe.get("geometry") != task.get("geometry"):
            continue
        payload_bytes = probe.get("payload_bytes")
        if not payload_bytes or sha256(raw[origin:origin + payload_bytes]) != probe.get("payload_sha256"):
            continue
        return {
            "weights": np.eye(channels, dtype=np.float16)[:, :, None, None],
            "bias": None,
            "primitive": "padding_identity_then_lut" if probe.get("activation_bounds") else "padding_identity",
            "region": raw[origin:origin + payload_bytes],
            "source": "embedded:padding_layouts",
            "padding": probe["pad"],
            "padding_mode": probe.get("padding_mode", "edge"),
            "native_padding_reference_top_left": probe.get("native_padding_reference_top_left"),
            "activation_bounds": probe.get("activation_bounds"),
        }
    return None


def recover_one(model_path: pathlib.Path, output: pathlib.Path, profiles: dict[str, Any]) -> dict[str, Any]:
    model_id = model_path.parents[3].name.removesuffix(".bundle")
    ir = export(model_path)
    data, sections, symbols = parse_hwx(model_path)
    groups = kernel_groups(data, sections, symbols)
    _, raw = kernel_section(data, sections)
    folder = output / model_id
    folder.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for group in groups:
        if group.kind != "ne":
            continue
        tasks = matching_tasks(ir, group)
        row: dict[str, Any] = {
            "group": group.stem,
            "task_references": [task["index"] for task in tasks],
            "status": "unresolved",
        }
        recovered = None
        if tasks:
            recovered = (
                recover_special(group, tasks[0])
                or recover_padding(group, tasks[0], raw, profiles)
                or recover_transpose(group, tasks[0], profiles)
                or recover_sparse_transpose(group, tasks[0], raw, profiles)
                or catalog_recovery.recover_expansion_2x2(group, tasks[0], raw)
            )
        if recovered is not None:
            if row["status"] == "unresolved":
                row["status"] = "controlled_round_trip"
            optional = {
                "source": recovered.get("source"),
                "primitive": recovered.get("primitive"),
                "shape": list(recovered["weights"].shape),
                "region_bytes": len(recovered.get("region", b"")),
                "payload_sha256": sha256(recovered.get("region", b"")),
                "pad_for_probe": recovered.get("pad_for_probe"),
                "native_origin_top_left": recovered.get("native_origin_top_left"),
                "native_origin_by_geometry_padding": recovered.get("native_origin_by_geometry_padding"),
                "padding": recovered.get("padding"),
                "padding_mode": recovered.get("padding_mode"),
                "native_padding_reference_top_left": recovered.get("native_padding_reference_top_left"),
                "activation_bounds": recovered.get("activation_bounds"),
            }
            row.update({key: value for key, value in optional.items() if value is not None})
        elif tasks:
            candidates, reason = catalog_recovery.recover_group(group, tasks[0], raw, groups)
            row["candidate_count"] = len(candidates)
            if len(candidates) == 1:
                recovered = candidates[0]
                row.update({
                    "status": "layout_round_trip_candidate",
                    "primitive": recovered["primitive"],
                    "shape": list(recovered["weights"].shape),
                    "groups": recovered["groups"],
                    "packing": list(recovered["packing"]),
                    "mode": recovered["mode"],
                    "region_bytes": recovered["bytes"],
                    "physical_block_addresses": recovered["block_starts"],
                    "payload_sha256": sha256(raw[min(group.starts):min(group.starts) + recovered["bytes"]]),
                })
            else:
                row["reason"] = reason
        else:
            row["reason"] = "no explicit coefficient reference"
        if recovered is not None:
            dest = folder / "coefficients" / group.stem
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / "weights.npy", recovered["weights"])
            if recovered["bias"] is not None:
                np.save(dest / "bias.npy", recovered["bias"])
            else:
                (dest / "bias.npy").unlink(missing_ok=True)
            row["weight_sha256"] = sha256(recovered["weights"].tobytes())
            row["bias_sha256"] = sha256(recovered["bias"].tobytes()) if recovered["bias"] is not None else None
        rows.append(row)
    summary = dict(collections.Counter(row["status"] for row in rows))
    report = {
        "model_id": model_id,
        "model_sha256": ir["model_sha256"],
        "kernel_bar": ir["kernel_bar"],
        "task_count": ir["task_count"],
        "ports": ir["ports"],
        "groups": rows,
        "summary": summary,
        "runtime_ready": False,
        "scope": "local coefficient recovery only; graph and runtime acceptance are recorded separately",
    }
    (folder / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(model_id, summary, flush=True)
    return report


def plan_for(model_id: str, reference: RecoveredReference,
             plans: dict[str, dict[str, Any]], references: dict[str, RecoveredReference]) -> dict[str, Any]:
    if model_id == WIDE2_SMALL:
        return portable_graph.source_plan(reference)
    if model_id in WIDE1_MODELS:
        return wide1_graph.source_plan(reference)
    if model_id in SUPERWIDE05_MODELS:
        return superwide05_graph.source_plan(reference)
    if model_id in TELE8_MODELS:
        return tele8_graph.source_plan(reference)
    if model_id == TELE4_SMALL:
        if WIDE2_SMALL not in plans:
            raise ValueError("Tele4 source recovery requires the Wide2 2304 baseline")
        return tele4_graph.source_plan(reference, plans[WIDE2_SMALL])
    if model_id in LARGE_MODELS:
        if model_id.startswith("LLV-HD-Tele-4x-"):
            base_id = TELE4_SMALL
        else:
            base_id = WIDE2_SMALL
        if base_id not in plans or base_id not in references:
            raise ValueError(f"{model_id} requires baseline {base_id}")
        return large_canvas_graph.source_plan(reference, plans[base_id], references[base_id])
    raise ValueError(f"unsupported model: {model_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extracted", type=Path, required=True, help="private extraction directory")
    parser.add_argument("--output", type=Path, required=True, help="new private portable-source directory")
    parser.add_argument("--models", nargs="+", help="optional model IDs (dependencies are added automatically)")
    parser.add_argument("--profiles", type=Path, help="optional recovery_profiles.json.gz override")
    parser.add_argument("--validate", choices=("none", "cpu"), default="none",
                        help="optional whole-frame CPU comparison against the native task schedule")
    parser.add_argument("--allow-partial", action="store_true", help="allow a requested model set to be incomplete")
    args = parser.parse_args()

    extracted = args.extracted.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not extracted.is_dir():
        parser.error(f"extraction directory not found: {extracted}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    available = discover_models(extracted)
    requested = set(args.models or available)
    unknown = requested - set(available)
    if unknown:
        parser.error(f"requested model not found in extraction: {', '.join(sorted(unknown))}")
    if not requested:
        parser.error("no LLVNR model.hwx files found")
    required = dependencies(requested)
    missing = required - set(available)
    if missing:
        parser.error(f"required model/dependency missing: {', '.join(sorted(missing))}")
    if not args.allow_partial and args.models is None and not ALL_MODELS.issubset(available):
        parser.error(f"extraction does not contain the full V63 model set; missing: {', '.join(sorted(ALL_MODELS - set(available)))}")

    profiles = load_profiles(args.profiles)
    reports: dict[str, dict[str, Any]] = {}
    for model_id in sorted(required):
        reports[model_id] = recover_one(available[model_id], output, profiles)

    references = {
        model_id: RecoveredReference(available[model_id], output / model_id, model_id=model_id, profiles=profiles)
        for model_id in sorted(required)
    }
    plans: dict[str, dict[str, Any]] = {}
    order = [WIDE2_SMALL, TELE4_SMALL, *sorted(required - {WIDE2_SMALL, TELE4_SMALL})]
    for model_id in order:
        if model_id not in required:
            continue
        plan = plan_for(model_id, references[model_id], plans, references)
        if plan.get("model_id") != model_id:
            raise ValueError(f"graph model identity mismatch: {model_id}")
        if not plan.get("coefficients") or not plan.get("nodes"):
            raise ValueError(f"incomplete graph for {model_id}")
        folder = output / model_id
        (folder / "source_graph.json").write_text(json.dumps(plan, indent=2) + "\n")
        plans[model_id] = plan
        reports[model_id]["graph_nodes"] = len(plan["nodes"])
        reports[model_id]["graph_coefficient_groups"] = len(plan["coefficients"])
        if args.validate == "cpu":
            from native_reference import synthetic_inputs
            from portable_graph import compare, evaluate_cpu, load_coefficients
            inputs = synthetic_inputs(20261112, references[model_id].canvas_wh)
            values = evaluate_cpu(plan, load_coefficients(plan, folder), inputs)
            native = references[model_id].run(inputs)
            names = [name for name in ("out_yuv", "state_yuv") if name in plan.get("outputs", [])]
            checks = {name: compare(values[name], native[name]) for name in names}
            passed = all(row.get("exact_fraction") == 1 for row in checks.values())
            reports[model_id]["schedule_comparison"] = {"passed": passed, "comparisons": checks,
                                                         "scope": "whole-frame CPU versus native task CPU"}
            (folder / "schedule_comparison.json").write_text(
                json.dumps(reports[model_id]["schedule_comparison"], indent=2) + "\n")
            if not passed:
                raise ValueError(f"CPU schedule comparison failed: {model_id}")
        (folder / "recovery.json").write_text(json.dumps(reports[model_id], indent=2) + "\n")

    summary = {
        "schema_version": 1,
        "extracted": extracted.name,
        "models": {
            model_id: {
                "model_sha256": reports[model_id]["model_sha256"],
                "graph_nodes": reports[model_id].get("graph_nodes"),
                "graph_coefficient_groups": reports[model_id].get("graph_coefficient_groups"),
                "recovery": reports[model_id]["summary"],
            }
            for model_id in sorted(required)
        },
        "note": "Private model-derived output. Do not publish coefficients or source graphs.",
    }
    (output / "RECOVERY-SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"models": len(required), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
