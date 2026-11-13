#!/usr/bin/env python3
"""Reproduce full-schedule, portable CPU and H13 checks on controlled inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import time

import numpy as np

from native_reference import NativeReference, PORT, sigmoid_lut, synthetic_inputs
from portable_graph import H13Model, compare, evaluate_cpu, load_coefficients, source_plan


TASK_BINDINGS = {
    14: "motion_coarse", 21: "motion_mid", 49: "temporal_sigmoid", 50: "temporal_gain",
    51: "temporal_weight_raw", 53: "temporal_weight", 55: "temporal_smoothed",
    56: "temporal_smooth_gate", 60: "temporal_mix", 61: "temporal_band0",
    288: "state_weight_band0", 289: "encoder_2", 290: "bottleneck_1",
    291: "bottleneck_2", 292: "bottleneck_3", 293: "decoder_2_up",
    294: "decoder_2_merge", 295: "decoder_2_1", 296: "decoder_2_2", 534: "out_yuv",
}


def controlled_cases(canvas_wh=(2304, 1296)):
    width, height = (v // 2 for v in canvas_wh)
    map_width = width // 2
    yield "correlated_noise", synthetic_inputs(20261112, canvas_wh)
    zero = {key: np.zeros_like(value) for key, value in synthetic_inputs(0, canvas_wh).items()}
    yield "zero", zero
    flat = {key: value.copy() for key, value in zero.items()}
    flat["his_frame"][:, :4] = 0.5
    flat["cur_frame"][:, :4] = 0.55
    flat["dnr_zoom_map"][:] = 1
    flat["fusion_zoom_map"][:] = 1
    yield "flat_step", flat
    rng = np.random.default_rng(20261113)
    random = {key: value.copy() for key, value in zero.items()}
    random["his_frame"][:] = rng.uniform(0, 1, random["his_frame"].shape)
    random["his_frame"][:, 4:] -= np.float16(0.5)
    random["cur_frame"][:] = random["his_frame"] + rng.normal(0, 0.025, random["his_frame"].shape)
    for name in ("dnr_zoom_map", "fusion_zoom_map"):
        random[name][:] = rng.uniform(0, 1, random[name].shape)
    yield "independent_map_and_noise", random
    ramp = {key: value.copy() for key, value in zero.items()}
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    for channel in range(6):
        offset = 0.0 if channel < 4 else -0.5
        ramp["his_frame"][0, channel] = xx / (width - 1) + offset
        ramp["cur_frame"][0, channel] = yy / (height - 1) + offset
    for name in ("dnr_zoom_map", "fusion_zoom_map"):
        ramp[name][0, 0] = np.linspace(0, 1, map_width, dtype=np.float16)[None]
    ramp["fusion_zoom_map"][:, :, :16, :16] = -0.001
    yield "spatial_ramps_and_map_clip", ramp
    impulse = {key: value.copy() for key, value in zero.items()}
    rows = sorted({r for r in (0, 1, 31, 32, 63, 64, 127, 128, 255, 256, 323, 324, 647, height - 1) if r < height})
    columns = sorted({c for c in (0, 1, 63, 64, 127, 128, 575, 576, 1151, width - 1) if c < width})
    for row in rows:
        for col in columns:
            impulse["cur_frame"][0, :4, row, col] = 1
            impulse["his_frame"][0, :4, row, col] = 0.125
            impulse["cur_frame"][0, 4:, row, col] = -0.25
    impulse["dnr_zoom_map"][:] = 0.5
    impulse["fusion_zoom_map"][:] = 0.5
    yield "boundary_and_strip_impulses", impulse


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def numeric_check(actual, expected):
    row = compare(actual, expected)
    row["finite"] = bool(np.isfinite(actual).all())
    row["passed"] = row["shape_equal"] and row["finite"] and row["max_abs"] <= 0.02 and row["mean_abs"] <= 0.001
    return row


def check_luts(reference, folder, use_h13):
    folder.mkdir(parents=True, exist_ok=True)
    points = np.arange(65536, dtype=np.uint16).view(np.float16)
    points = points[np.isfinite(points) & (points >= -12) & (points <= 12)]
    unique_points = len(points)
    points = np.resize(points, 49152).reshape(1, 1, 24, 2048)
    results = []
    for prefix, scale in (("K2C1AE", 1.0), ("KB4E00", 0.39990234375)):
        name = next(key for key in reference.activation_tables if key.startswith(prefix))
        table = reference.activation_tables[name]
        expected = sigmoid_lut(points.astype(np.float32), table).astype(np.float16)
        row = {"group": name, "table_sha256": sha(table), "unique_fp16_points": unique_points,
               "range": [-12, 12], "interpolation": "33 samples, step 0.5, two linear tails, saturation"}
        if use_h13:
            import aneforge as af
            tensor = af.input(points.shape).sigmoid()
            if scale != 1:
                tensor = tensor * scale
            net = af.compile(tensor, build_dir=folder / prefix)
            try:
                actual = np.asarray(net(points), dtype=np.float16).copy()
            finally:
                net.release()
            row["h13"] = compare(actual, expected)
            row["h13"]["passed"] = row["h13"]["max_abs"] <= 0.00048828125 and row["h13"]["mean_abs"] <= 0.00002
            np.savez_compressed(folder / (prefix + ".npz"), x=points, actual=actual, expected=expected)
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h13", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, default=PORT / "graph/portable/acceptance")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = NativeReference()
    plan = source_plan(reference)
    coefficients = load_coefficients(plan)
    (args.out / "source_graph.json").write_text(json.dumps(plan, indent=2) + "\n")
    model = H13Model(plan, coefficients, args.out / "h13") if args.h13 else None
    report = {"model_sha256": plan["model_sha256"], "nodes": len(plan["nodes"]),
              "source_graph_sha256": hashlib.sha256((json.dumps(plan,indent=2)+"\n").encode()).hexdigest(),
              "execution_profile": plan.get("execution_profile","ane"),
              "environment": {"platform": platform.platform(), "numpy": np.__version__},
              "numeric_gate": {"max_abs": 0.02, "mean_abs": 0.001, "finite": True},
              "cases": [], "hardware_reference": "H13 local, H19 execution unavailable",
              "synthetic_intermediate_operands": 0, "replacement_weights": 0}
    path = args.out / "validation.json"
    try:
        for name, inputs in controlled_cases():
            started = time.perf_counter()
            values = evaluate_cpu(plan, coefficients, inputs)
            cpu_seconds = time.perf_counter() - started
            native = reference.run(inputs, capture_tasks=TASK_BINDINGS)
            metrics = {key: compare(values[key], reference.captures[task][None]) for task, key in TASK_BINDINGS.items()}
            metrics["state_yuv"] = compare(values["state_yuv"], native["state_yuv"])
            row = {"case": name, "input_sha256": {key: sha(value) for key, value in inputs.items()},
                   "native_tasks_executed": len(reference.trace), "portable_cpu_seconds": cpu_seconds,
                   "schedule_comparisons": metrics,
                   "schedule_passed": all(metric["shape_equal"] and metric["exact_fraction"] == 1 for metric in metrics.values())}
            if model:
                started = time.perf_counter()
                outputs = model.run(inputs)
                row["h13_seconds"] = time.perf_counter() - started
                row["h13"] = {key: numeric_check(outputs[key], values[key]) for key in plan["outputs"]}
                row["h13_passed"] = all(metric["passed"] for metric in row["h13"].values())
            report["cases"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(name, "schedule", row["schedule_passed"], "h13", row.get("h13_passed"), flush=True)
        if model:
            base = synthetic_inputs(20261112)
            first = model.run(base)
            second = model.run(base)
            changed = {key: value.copy() for key, value in base.items()}
            changed["cur_frame"][:, :4, 128:256, 256:512] += np.float16(0.125)
            perturbed = model.run(changed)
            report["same_frame_causality"] = {key: compare(perturbed[key], first[key]) for key in plan["outputs"]}
            report["deterministic_reexecution"] = all(np.array_equal(first[key], second[key]) for key in plan["outputs"])
            report["same_frame_causality_passed"] = all(report["same_frame_causality"][key]["max_abs"] > 0
                                                         for key in ("out_yuv", "state_yuv"))
            report["compile_seconds"] = model.compile_seconds
        report["activation_luts"] = check_luts(reference, args.out / "luts", args.h13)
    finally:
        if model:
            model.close()
    report["passed"] = all(row["schedule_passed"] and row.get("h13_passed", True) for row in report["cases"])
    if model:
        report["passed"] &= report["deterministic_reexecution"] and report["same_frame_causality_passed"]
        report["passed"] &= all(row["h13"]["passed"] for row in report["activation_luts"])
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
