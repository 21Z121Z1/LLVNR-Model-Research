#!/usr/bin/env python3
"""2464 graph variants, bound to recovered tensors and their own native schedules."""
import argparse
import copy
import hashlib
import json
import pathlib

import numpy as np

from native_reference import synthetic_inputs
from portable_graph import evaluate_cpu, load_coefficients, compare, execute_h13
from recovered_reference import RecoveredReference

PORT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ("LLV-HD-Wide-2x-v0.3.0_2464x1388", "LLV-HD-Tele-4x-v0.3.0_2464x1388")


def source_plan(reference, baseline, baseline_reference):
    model_id = reference.model_id
    base_bytes = json.dumps(baseline, indent=2, sort_keys=True).encode()
    expected = baseline_reference.coefficients
    rebound, files, bindings = {}, {}, []
    for old_group, old in expected.items():
        matches = [k for k, v in reference.coefficients.items()
                   if np.array_equal(v["weights"], old["weights"])
                   and ((v["bias"] is None and old["bias"] is None) or np.array_equal(v["bias"], old["bias"]))]
        if not matches:
            raise ValueError(f"canvas variant has different or unrecovered coefficients: {old_group}")
        group = old_group if old_group in matches else sorted(matches)[0]
        rebound[old_group] = group
        bindings.append({"base_group": old_group, "native_group": group, "decoded_coefficients_equal": True})
    nodes = []
    align = {"motion_coarse_up": "cur_mid", "motion_mid_up": "cur_low"}
    for original in baseline["nodes"]:
        node = copy.deepcopy(original)
        node.pop("native_task", None)
        if "group" in node:
            node["group"] = rebound[node["group"]]
        if node["op"] == "input":
            node["shape"] = [1, *reference.port_views[node["name"]]["shape_chw"]]
        name = node["name"]
        if name in ("encoder_1", "encoder_2"):
            node["output_rounding"] = "floor_stride"
        if name in ("decoder_2_up", "decoder_1_up"):
            node["output_padding"] = [1, 0]
        if name in align:
            node["name"] = name + "_raw"
            nodes.append(node)
            nodes.append({"name": name, "op": "match_spatial", "inputs": [name + "_raw", align[name]],
                          "alignment": "replicate trailing row or crop trailing staging row; validated against own native schedule"})
        else:
            nodes.append(node)
    files = reference.coefficient_records({node["group"] for node in nodes if "group" in node})
    return {"schema_version": 1, "model_id": model_id, "model_sha256": reference.ir["model_sha256"],
            "execution_profile": baseline.get("execution_profile","ane"),
            "coefficients": files, "nodes": nodes, "outputs": baseline["outputs"], "coefficient_bindings": bindings,
            "base_graph_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "validation_status": "candidate odd-height canvas adaptation; per-model schedule and H13 checks required"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True, help="local model.hwx")
    parser.add_argument("--recovery", type=pathlib.Path, required=True, help="private recovery directory")
    parser.add_argument("--baseline-model", type=pathlib.Path, required=True, help="Wide2 2304 model.hwx")
    parser.add_argument("--baseline-recovery", type=pathlib.Path, required=True, help="Wide2 2304 recovery directory")
    parser.add_argument("--baseline", type=pathlib.Path, required=True, help="Wide2 2304 template graph")
    parser.add_argument("--output", type=pathlib.Path, help="private graph output directory")
    parser.add_argument("--h13", action="store_true")
    args = parser.parse_args()
    reference = RecoveredReference(args.model, args.recovery)
    baseline_reference = RecoveredReference(args.baseline_model, args.baseline_recovery)
    baseline = json.loads(args.baseline.read_text())
    plan = source_plan(reference, baseline, baseline_reference)
    folder = args.output or args.recovery
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source_graph.json").write_text(json.dumps(plan, indent=2) + "\n")
    inputs = synthetic_inputs(20261112, reference.canvas_wh)
    values = evaluate_cpu(plan, load_coefficients(plan, folder), inputs)
    native = reference.run(inputs)
    checks = {name: compare(values[name], native[name]) for name in ("out_yuv", "state_yuv")}
    report = {"model_id": reference.model_id, "model_sha256": plan["model_sha256"], "nodes": len(plan["nodes"]),
              "native_tasks_executed": len(reference.trace), "comparisons": checks,
              "passed": all(row["exact_fraction"] == 1 for row in checks.values()),
              "scope": "whole-frame CPU versus own original task CPU; not original H19 hardware parity"}
    (folder / "schedule_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        return 1
    if args.h13:
        return 0 if execute_h13(plan, load_coefficients(plan, folder), inputs, values, folder / "h13")["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
