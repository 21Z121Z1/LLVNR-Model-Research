#!/usr/bin/env python3
"""Tele 4x 2304 graph, with its own temporal and noise modulation branches."""
import argparse
import copy
import hashlib
import json
import pathlib
import time

import numpy as np

from native_reference import synthetic_inputs
from portable_graph import evaluate_cpu, load_coefficients, compare, execute_h13
from portable_ops import fp16, fp32
from recovered_reference import RecoveredReference

PORT = pathlib.Path(__file__).resolve().parents[1]
MODEL = "LLV-HD-Tele-4x-v0.3.0_2304x1296"


def source_plan(reference, baseline):
    original = {n["name"]: n for n in baseline["nodes"]}
    nodes = []

    def add(name, op, inputs, **attrs):
        nodes.append({"name": name, "op": op, "inputs": list(inputs), **attrs})
        return name

    def copy_node(name, task=None):
        node = copy.deepcopy(original[name])
        if task is not None:
            row = reference.ir["tasks"][task]
            address = next(a for a in row["address_writes"] if a["register_word"] == 0x1544)
            group = reference.groups_by_offset[address["offset"]].stem
            node["group"] = group
            if node["op"] == "conv2d":
                cfg = row["written_registers"].get("1242", 0)
                node["activation"] = {0: "identity", 1: "relu", 2: "sigmoid"}[cfg >> 16 & 3]
                node["edge"] = bool(cfg & 0x40000)
                node["native_task"] = task
                if node["activation"] == "sigmoid":
                    node["activation_lut_hex"] = reference.activation_tables[group].tobytes().hex()
        nodes.append(node)
        if name.startswith("fusion_weight_"):
            regs = reference.ir["tasks"][10]["written_registers"]
            node.update(alpha=fp32(regs["114a"]), beta=fp32(regs["114b"]))

    def unary(name, x, task):
        row = reference.ir["tasks"][task]
        r = row["written_registers"]
        address = next(a for a in row["address_writes"] if a["register_word"] == 0x158E)
        group = reference.groups_by_offset[address["offset"]].stem
        add(name, "affine_sigmoid", [x], bias=fp16(r.get("1244", 0)), scale=fp16(r.get("1245", 0x3c00)),
            output_scale=1.0, activation_lut_hex=reference.activation_tables[group].tobytes().hex(), native_task=task)

    # Both variants use the same verified multiscale motion feature structure;
    # every learned layer is bound to Tele's own coefficient group/task.
    mapping = {"his_low": 0, "cur_low": 3, "motion_coarse_1": 12, "motion_coarse_2": 13,
               "motion_coarse": 14, "motion_mid_1": 19, "motion_mid_2": 20, "motion_mid": 21,
               "motion_fine_1": 28, "motion_fine_2": 29, "motion_fine": 30}
    for node in baseline["nodes"]:
        if node["name"] == "temporal_sigmoid":
            break
        copy_node(node["name"], mapping.get(node["name"]))
    copy_node("fusion_negative_epsilon")
    add("temporal_weight", "add", ["motion_fine", "fusion_negative_epsilon"], native_task=50)
    add("temporal_smoothed", "pool", ["temporal_weight"], kernel=5, stride=1, mode=0, edge=True, native_task=52)
    add("temporal_guarded", "maximum", ["temporal_smoothed", "highlight_guard"])
    add("state_weight_band0", "clip", ["temporal_guarded"], minimum=0.0999755859375, maximum=1.0, native_task=53)
    add("temporal_full", "nearest2", ["state_weight_band0"])
    for name in ("history_weight", "history_part", "current_part", "state_yuv"):
        copy_node(name)
    copy_node("analysis", 104)
    add("analysis_c0", "channel", ["analysis"], index=4)
    add("analysis_c1", "channel", ["analysis"], index=5)
    add("c0_negative20", "affine", ["analysis_c0"], alpha=-20.0, beta=0.0, native_task=105)
    unary("c0_negative_gate", "c0_negative20", 106)
    add("c1_positive20", "affine", ["analysis_c1"], alpha=20.0, beta=0.0, native_task=107)
    unary("c1_negative_gate", "c1_positive20", 108)
    add("noise_product_a", "mul", ["c0_negative_gate", "c1_negative_gate"], native_task=109)
    r = reference.ir["tasks"][110]["written_registers"]
    add("c0_linear", "clipped_affine", ["analysis_c0"], alpha=fp32(r["1144"]), beta=fp32(r["1141"]),
        minimum=0.0, maximum=1.0, native_task=110)
    unary("c1_positive_gate", "c1_positive20", 111)
    add("noise_product_b", "mul", ["c0_linear", "c1_positive_gate"], native_task=112)
    add("noise_sum_a", "add", ["noise_product_b", "noise_product_a"], native_task=113)
    unary("c0_positive_gate", "c0_negative20", 114)
    unary("c1_gate", "analysis_c1", 115)
    add("noise_product_c", "mul", ["c0_positive_gate", "c1_gate"], native_task=116)
    add("noise_sum_b", "add", ["noise_sum_a", "noise_product_c"])
    add("noise_gate", "clip", ["noise_sum_b"], minimum=0.0, maximum=1.0, native_task=117)
    add("noise_inverse", "affine", ["noise_gate"], alpha=-1.0, beta=1.0, native_task=118)
    r = reference.ir["tasks"][119]["written_registers"]
    add("dnr_noise_part", "mul_scaled", ["dnr_zoom_map", "noise_gate"], scale=fp32(r["1144"]), native_task=119)
    add("dnr_inverse_part", "mul", ["dnr_zoom_map", "noise_inverse"], native_task=120)
    r = reference.ir["tasks"][121]["written_registers"]
    add("dnr_feature", "add_affine", ["dnr_noise_part", "dnr_inverse_part"], alpha=fp32(r["1142"]), beta=fp32(r["114b"]), native_task=121)
    # Encoder/decoder topology shares the wavelet architecture, with Tele's
    # independently recovered learned tensors, including transpose phases.
    mapping = {"encoder_1": 122, "encoder_2": 123, "bottleneck_1": 124, "bottleneck_2": 125,
               "bottleneck_3": 126, "decoder_2_up": 127, "decoder_2_merge": 128, "decoder_2_1": 129,
               "decoder_2_2": 130, "decoder_1_up": 132, "decoder_1_merge": 133, "decoder_1": 134,
               "pre_synthesis": 204, "synthesis": 205, "output_refine_1": 206, "output_refine_2": 207, "out_yuv": 368}
    tail = False
    for node in baseline["nodes"]:
        tail |= node["name"] == "encoder_input"
        if tail:
            copy_node(node["name"], mapping.get(node["name"]))
    used = {n["group"] for n in nodes if "group" in n}
    files = reference.coefficient_records(used)
    return {"schema_version": 1, "model_id": reference.model_id,
            "model_sha256": reference.ir["model_sha256"], "coefficients": files,
            "nodes": nodes, "outputs": ["out_yuv", "state_yuv", "state_weight_band0"],
            "validation_status": "candidate Tele graph; independent schedule/H13 checks required"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="local model.hwx")
    parser.add_argument("--recovery", type=Path, required=True, help="private recovery directory")
    parser.add_argument("--baseline", type=Path, required=True, help="portable Wide2 2304 template graph")
    parser.add_argument("--output", type=Path, help="private graph output directory")
    parser.add_argument("--h13", action="store_true")
    args = parser.parse_args()
    folder = args.output or args.recovery
    folder.mkdir(parents=True, exist_ok=True)
    reference = RecoveredReference(args.model, args.recovery)
    baseline = json.loads(args.baseline.read_text())
    plan = source_plan(reference, baseline)
    (folder / "source_graph.json").write_text(json.dumps(plan, indent=2) + "\n")
    inputs = synthetic_inputs(20261112)
    started = time.perf_counter()
    values = evaluate_cpu(plan, load_coefficients(plan, folder), inputs)
    direct = {14: "motion_coarse", 21: "motion_mid", 50: "temporal_weight", 52: "temporal_smoothed", 53: "state_weight_band0",
              104: "analysis", 105: "c0_negative20", 106: "c0_negative_gate", 107: "c1_positive20", 108: "c1_negative_gate",
              109: "noise_product_a", 110: "c0_linear", 111: "c1_positive_gate", 112: "noise_product_b", 113: "noise_sum_a",
              114: "c0_positive_gate", 115: "c1_gate", 116: "noise_product_c", 117: "noise_gate", 118: "noise_inverse",
              120: "dnr_inverse_part", 121: "dnr_feature", 123: "encoder_2", 124: "bottleneck_1", 125: "bottleneck_2",
              126: "bottleneck_3", 127: "decoder_2_up", 128: "decoder_2_merge", 129: "decoder_2_1", 130: "decoder_2_2"}
    native = reference.run(inputs, capture_tasks=direct)
    comparisons = {name: compare(values[name], native[name]) for name in ("out_yuv", "state_yuv")}
    report = {"nodes": len(plan["nodes"]), "coefficient_groups": len(plan["coefficients"]), "seconds": time.perf_counter() - started,
              "comparisons": comparisons, "passed": all(v["exact_fraction"] == 1.0 for v in comparisons.values()),
              "scope": "whole-frame CPU versus native task CPU; H19 original hardware permission denied",
              "intermediates": {name: compare(values[name], reference.captures[task][None]) for task, name in direct.items()}}
    (folder / "schedule_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        return 1
    if args.h13:
        h13 = execute_h13(plan, load_coefficients(plan, folder), inputs, values, folder / "h13")
        print(json.dumps(h13, indent=2))
        return 0 if h13["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
