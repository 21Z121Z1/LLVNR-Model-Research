#!/usr/bin/env python3
"""Tele 8x graph, recovered from its multiscale motion and two-output schedule."""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from llvnr_taskflow import address
from native_reference import synthetic_inputs
from portable_graph import compare, evaluate_cpu, execute_h13, load_coefficients
from recovered_reference import RecoveredReference

PORT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ("LLV-HD-Tele-8x-v0.3.0_2304x1296", "LLV-HD-Tele-8x-v0.3.0_2464x1388")


def source_plan(reference):
    model_id = reference.model_id
    large = reference.canvas_wh == (2464, 1388)
    indices = ({"his": (3,4,5), "cur": (6,7,8), "motion_mid": (19,20,21), "motion_fine": (31,32,33),
                "analysis": 94, "sqrt": 96, "encoder": (98,99), "bottleneck": (100,101,102),
                "decoder2": (103,111,112,114), "decoder1": (115,117,118)} if large else
               {"his": (0,1,2), "cur": (3,4,5), "motion_mid": (18,19,20), "motion_fine": (29,30,31),
                "analysis": 105, "sqrt": 107, "encoder": (109,110), "bottleneck": (111,112,113),
                "decoder2": (114,116,117,118), "decoder1": (119,120,121)})
    nodes = []

    def add(name, op, inputs, **attrs):
        nodes.append({"name": name, "op": op, "inputs": list(inputs), **attrs})
        return name

    def conv(name, source, index, stride=1, pad=1, transpose=False, floor=False):
        task = reference.ir["tasks"][index]
        group = reference.groups_by_offset[address(task, 0x1544)["offset"]].stem
        record = reference.coefficients[group]
        cfg = task["written_registers"].get("1242", 0)
        activation = {0: "identity", 1: "relu", 2: reference.luts.get(group, {}).get("operation", "unknown")}[cfg >> 16 & 3]
        attrs = {"group": group, "stride": stride, "pad": pad, "edge": bool(cfg & 0x40000),
                 "activation": activation, "native_task": index}
        if activation == "sigmoid":
            attrs["activation_lut_hex"] = reference.activation_tables[group].tobytes().hex()
        if activation == "clip":
            attrs["activation_bounds"] = reference.luts[group]["bounds"]
        if floor:
            attrs["output_rounding"] = "floor_stride"
        return add(name, "conv_transpose" if transpose else "conv2d", [source], **attrs)

    for name in ("cur_frame", "his_frame", "dnr_zoom_map", "fusion_zoom_map"):
        add(name, "input", [], shape=[1, *reference.port_views[name]["shape_chw"]])
    conv("his_low", "his_frame", indices["his"][0], stride=2, pad=0)
    conv("cur_low", "cur_frame", indices["cur"][0], stride=2, pad=0)
    for prefix in ("his", "cur"):
        _, task_a, task_b = indices[prefix]
        conv(prefix + "_mid", prefix + "_low", task_a, stride=2, pad=0)
        conv(prefix + "_coarse", prefix + "_mid", task_b, stride=2, pad=0)
    add("fusion_mid", "pool", ["fusion_zoom_map"], kernel=2, stride=2, mode=0, edge=False)
    add("fusion_coarse", "pool", ["fusion_mid"], kernel=2, stride=2, mode=0, edge=False)
    for suffix, scale, tasks, map_name in (("coarse", "coarse", (11, 12, 13), "fusion_coarse"),
                                          ("mid", "mid", indices["motion_mid"], "fusion_mid"),
                                          ("fine", "low", indices["motion_fine"], "fusion_zoom_map")):
        add("difference_" + suffix, "absdiff", ["cur_" + scale, "his_" + scale])
        inputs = ["cur_" + scale, "his_" + scale, "difference_" + suffix]
        if suffix != "coarse":
            previous = "coarse" if suffix == "mid" else "mid"
            add("motion_" + previous + "_up_raw", "nearest2", ["motion_" + previous])
            add("motion_" + previous + "_up", "match_spatial", ["motion_" + previous + "_up_raw", "cur_" + scale])
            inputs.append("motion_" + previous + "_up")
        inputs.append(map_name)
        add("motion_" + suffix + "_input", "concat", inputs)
        conv("motion_" + suffix + "_1", "motion_" + suffix + "_input", tasks[0])
        conv("motion_" + suffix + "_2", "motion_" + suffix + "_1", tasks[1])
        conv("motion_" + suffix, "motion_" + suffix + "_2", tasks[2])
    add("fusion_negative_epsilon", "clip", ["fusion_zoom_map"], minimum=-1.0013580322265625e-5, maximum=0.0)
    add("temporal_raw", "add", ["motion_fine", "fusion_negative_epsilon"])
    add("temporal_padded", "replicate_pad", ["temporal_raw"], padding=[3, 3, 3, 3])
    add("temporal_dilated", "pool", ["temporal_padded"], kernel=3, stride=1, mode=2, edge=False)
    add("temporal_smoothed", "pool", ["temporal_dilated"], kernel=5, stride=1, mode=0, edge=False)
    add("temporal_weight", "clip", ["temporal_smoothed"], minimum=0.0999755859375, maximum=1.0)
    add("temporal_full", "nearest2", ["temporal_weight"])
    add("history_weight", "affine", ["temporal_full"], alpha=-1.0, beta=1.0)
    add("history_part", "mul", ["his_frame", "history_weight"])
    add("current_part", "mul", ["cur_frame", "temporal_full"])
    add("state_yuv", "add", ["history_part", "current_part"])
    conv("analysis", "state_yuv", indices["analysis"], stride=2, pad=0)
    task = reference.ir["tasks"][indices["sqrt"]]
    group = reference.groups_by_offset[address(task, 0x158e)["offset"]].stem
    add("noise_weight", "sqrt", ["temporal_weight"], activation_lut_hex=reference.activation_tables[group].tobytes().hex())
    add("dnr_feature", "mul", ["dnr_zoom_map", "noise_weight"])
    add("encoder_input", "concat", ["analysis", "dnr_feature"])
    conv("encoder_1", "encoder_input", indices["encoder"][0], stride=2, floor=True)
    conv("encoder_2", "encoder_1", indices["encoder"][1], stride=2, floor=True)
    conv("bottleneck_1", "encoder_2", indices["bottleneck"][0])
    conv("bottleneck_2", "bottleneck_1", indices["bottleneck"][1])
    conv("bottleneck_3", "bottleneck_2", indices["bottleneck"][2])
    conv("decoder_2_up_raw", "bottleneck_3", indices["decoder2"][0], stride=2, pad=0, transpose=True)
    add("decoder_2_up", "match_spatial", ["decoder_2_up_raw", "encoder_1"])
    add("decoder_2_input", "concat", ["encoder_1", "decoder_2_up"])
    conv("decoder_2_merge", "decoder_2_input", indices["decoder2"][1], pad=0)
    conv("decoder_2_1", "decoder_2_merge", indices["decoder2"][2])
    conv("decoder_2_2", "decoder_2_1", indices["decoder2"][3])
    conv("decoder_1_up_raw", "decoder_2_2", indices["decoder1"][0], stride=2, pad=0, transpose=True)
    add("decoder_1_up", "match_spatial", ["decoder_1_up_raw", "analysis"])
    add("decoder_1_input", "concat", ["analysis", "decoder_1_up"])
    conv("decoder_1_merge", "decoder_1_input", indices["decoder1"][1], pad=0)
    conv("decoder_1", "decoder_1_merge", indices["decoder1"][2])
    conv("pre_synthesis", "decoder_1", 270)
    conv("synthesis_coefficients", "pre_synthesis", 271, pad=0)
    conv("synthesis", "synthesis_coefficients", 272, stride=2, pad=0, transpose=True)
    add("output_refine_input", "concat", ["state_yuv", "synthesis"])
    conv("output_refine", "output_refine_input", 273)
    conv("out_yuv", "output_refine", 274)
    files = reference.coefficient_records({n["group"] for n in nodes if "group" in n})
    return {"schema_version": 1, "model_id": model_id, "model_sha256": reference.ir["model_sha256"],
            "nodes": nodes, "outputs": ["out_yuv", "state_yuv"], "coefficients": files,
            "validation_status": "requires whole native schedule and ANE acceptance"}


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
    inputs = synthetic_inputs(20261112, reference.canvas_wh)
    coefficients = load_coefficients(plan, folder)
    values = evaluate_cpu(plan, coefficients, inputs)
    names = ("motion_coarse", "motion_mid", "analysis", "encoder_1", "encoder_2", "bottleneck_1", "bottleneck_2")
    direct = {n["native_task"]: n["name"] for n in plan["nodes"] if n["name"] in names}
    native = reference.run(inputs, capture_tasks=direct)
    metrics = {k: compare(values[k], native[k]) for k in plan["outputs"]}
    report = {"model_id": reference.model_id, "native_tasks_executed": len(reference.trace), "comparisons": metrics,
              "intermediates": {n: compare(values[n], reference.captures[t][None]) for t, n in direct.items()},
              "passed": all(v["exact_fraction"] == 1 for v in metrics.values())}
    (folder / "schedule_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        return 1
    return 0 if not args.h13 or execute_h13(plan, coefficients, inputs, values, folder / "h13")["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
