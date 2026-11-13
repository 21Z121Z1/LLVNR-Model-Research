#!/usr/bin/env python3
"""Whole-frame Wide1 / SuperWide1 graph from their own original task schedules."""
import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from native_reference import synthetic_inputs
from portable_graph import evaluate_cpu, load_coefficients, compare, execute_h13
from portable_ops import fp32
from recovered_reference import RecoveredReference
from llvnr_taskflow import address

PORT = pathlib.Path(__file__).resolve().parents[1]
MODELS = tuple(f"LLV-HD-{lens}-1x-v0.3.0_{size}" for lens in ("Wide", "SuperWide")
               for size in ("2304x1296", "2464x1388"))


def source_plan(reference, task_map_override=None):
    nodes = []
    large = reference.canvas_wh == (2464,1388)
    task_map = ({435:531,436:532,437:533,438:534,439:535,443:536,444:537,445:543,446:544,447:545,
                 449:547,450:549,451:550,533:706,534:708,535:709,536:710,537:711} if large else {})
    if task_map_override is not None:
        task_map = task_map_override

    def add(name, op, inputs, **attrs):
        nodes.append({"name": name, "op": op, "inputs": list(inputs), **attrs})
        return name

    def conv(name, source, task_index, stride=1, pad=1, transpose=False):
        task_index = task_map.get(task_index,task_index)
        task = reference.ir["tasks"][task_index]
        group = reference.groups_by_offset[address(task, 0x1544)["offset"]].stem
        record = reference.coefficients[group]
        cfg = task["written_registers"].get("1242", 0)
        activation = {0: "identity", 1: "relu", 2: reference.luts.get(group, {}).get("operation", "unknown")}[cfg >> 16 & 3]
        attrs = {"group": group, "stride": stride, "pad": pad, "edge": bool(cfg & 0x40000),
                 "activation": activation, "groups": record["evidence"].get("groups", 1), "native_task": task_index}
        if activation == "clip":
            attrs["activation_bounds"] = reference.luts[group]["bounds"]
        elif activation == "sigmoid":
            attrs["activation_lut_hex"] = reference.activation_tables[group].tobytes().hex()
        if transpose:
            attrs["output_padding"] = 1
        if large and name in ("encoder_2","encoder_3"):
            attrs["output_rounding"] = "floor_stride"
        if large and name in ("decoder_2_up","decoder_1_up"):
            raw = add(name+"_raw", "conv_transpose", [source], **attrs)
            add(name+"_aligned", "match_spatial", [raw, "encoder_2" if name=="decoder_2_up" else "encoder_1"])
            return add(name, "clip", [name+"_aligned"], minimum=0.0, maximum=6.0)
        return add(name, "conv_transpose" if transpose else "conv2d", [source], **attrs)

    for name in ("cur_frame", "his_frame"):
        add(name, "input", [], shape=[1, *reference.port_views[name]["shape_chw"]])
    for name in ("dnr_zoom_map", "fusion_zoom_map"):
        add(name, "input", [], shape=[1, *reference.port_views[name]["shape_chw"]])
    scale = fp32(reference.ir["tasks"][2]["written_registers"]["1142"])
    add("fusion_scaled", "affine", ["fusion_zoom_map"], alpha=scale, beta=0.0)
    add("fusion_full", "nearest2", ["fusion_scaled"])
    add("motion_input", "concat", ["cur_frame", "his_frame", "fusion_full"])
    conv("motion_average", "motion_input", 4, stride=2)
    conv("motion_1", "motion_average", 5)
    conv("motion_2", "motion_1", 6)
    conv("motion_probability", "motion_2", 7)
    add("temporal_weight", "clip", ["motion_probability"], minimum=0.0999755859375, maximum=1.0, native_task=8)
    add("temporal_full", "nearest2", ["temporal_weight"])
    add("history_weight", "affine", ["temporal_full"], alpha=-1.0, beta=1.0)
    add("current_part", "mul", ["cur_frame", "temporal_full"])
    add("history_part", "mul", ["his_frame", "history_weight"])
    add("state_yuv", "add", ["current_part", "history_part"])
    add("dnr_scaled", "affine", ["dnr_zoom_map"], alpha=scale, beta=0.0)
    add("dnr_full", "nearest2", ["dnr_scaled"])
    add("denoiser_input", "concat", ["state_yuv", "dnr_full"])
    conv("encoder_1", "denoiser_input", 435, stride=2)
    conv("encoder_2", "encoder_1", 436, stride=2)
    conv("encoder_3", "encoder_2", 437, stride=2)
    conv("bottleneck_1", "encoder_3", 438)
    conv("bottleneck_2", "bottleneck_1", 439)
    conv("bottleneck_3", "bottleneck_2", 443)
    conv("decoder_2_up", "bottleneck_3", 444, stride=2, transpose=True)
    add("decoder_2_input", "concat", ["encoder_2", "decoder_2_up"])
    conv("decoder_2_merge", "decoder_2_input", 445, pad=0)
    conv("decoder_2_1", "decoder_2_merge", 446)
    conv("decoder_2_2", "decoder_2_1", 447)
    conv("decoder_1_up", "decoder_2_2", 449, stride=2, transpose=True)
    add("decoder_1_input", "concat", ["encoder_1", "decoder_1_up"])
    conv("decoder_1_merge", "decoder_1_input", 450, pad=0)
    conv("decoder_1_1", "decoder_1_merge", 451)
    conv("decoder_1_2", "decoder_1_1", 533)
    conv("decoder_0_up", "decoder_1_2", 534, stride=2, transpose=True)
    add("decoder_0_input", "concat", ["denoiser_input", "decoder_0_up"])
    conv("decoder_0_merge", "decoder_0_input", 535, pad=0)
    conv("decoder_0_1", "decoder_0_merge", 536)
    conv("out_yuv", "decoder_0_1", 537)

    aliases = []
    pairs = (((4,16),(6,18),(537,539),(537,541),(547,552),(547,702),(708,713),(708,1138),(710,715)) if large else
             ((4, 16), (6, 18), (438, 441), (439, 442), (449, 453), (449, 529),
              (534, 540), (534, 546), (534, 1086), (536, 1088)))
    for task_a, task_b in (() if task_map_override is not None else pairs):
        keys = [reference.groups_by_offset[address(reference.ir["tasks"][i], 0x1544)["offset"]].stem for i in (task_a, task_b)]
        a, b = [reference.coefficients[k] for k in keys]
        if not np.array_equal(a["weights"], b["weights"]) or not ((a["bias"] is None and b["bias"] is None) or np.array_equal(a["bias"], b["bias"])):
            raise ValueError(f"native phase coefficient mismatch: {task_a}/{task_b}")
        aliases.append({"tasks": [task_a, task_b], "groups": keys, "coefficients_equal": True})
    used = {n["group"] for n in nodes if "group" in n}
    files = reference.coefficient_records(used)
    model_id = reference.model_id
    return {"schema_version": 1, "model_id": model_id, "model_sha256": reference.ir["model_sha256"],
            "nodes": nodes, "outputs": ["out_yuv", "state_yuv"], "coefficients": files, "coefficient_aliases": aliases,
            "validation_status": "candidate: full native schedule and H13 checks required"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="local model.hwx")
    parser.add_argument("--recovery", type=Path, required=True, help="private recovery directory")
    parser.add_argument("--output", type=Path, help="private graph output directory")
    parser.add_argument("--h13", action="store_true")
    args = parser.parse_args()
    reference = RecoveredReference(args.model, args.recovery)
    plan = source_plan(reference)
    folder = args.output or args.recovery
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source_graph.json").write_text(json.dumps(plan, indent=2) + "\n")
    inputs, started = synthetic_inputs(20261112, reference.canvas_wh), time.perf_counter()
    values = evaluate_cpu(plan, load_coefficients(plan, folder), inputs)
    names = ("encoder_1","encoder_2","bottleneck_3","decoder_2_up","decoder_2_merge","decoder_2_1","decoder_2_2")
    direct = {n["native_task"]: n["name"] for n in plan["nodes"] if n["name"] in names and "native_task" in n}
    native = reference.run(inputs, capture_tasks=direct)
    comparisons = {name: compare(values[name], native[name]) for name in plan["outputs"]}
    report = {"model_id": reference.model_id, "model_sha256": plan["model_sha256"], "nodes": len(plan["nodes"]),
              "seconds": time.perf_counter() - started, "comparisons": comparisons,
              "intermediates": {name: compare(values[name], reference.captures[t][None]) for t, name in direct.items()},
              "passed": all(v["exact_fraction"] == 1 for v in comparisons.values()),
              "scope": "whole-frame CPU versus original task CPU; not H19 hardware golden"}
    (folder / "schedule_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        return 1
    if args.h13:
        return 0 if execute_h13(plan, load_coefficients(plan, folder), inputs, values, folder / "h13")["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
