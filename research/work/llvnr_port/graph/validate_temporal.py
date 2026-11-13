#!/usr/bin/env python3
"""Check same-frame response, recurrent replay and explicit scene reset behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from portable_graph import H13Model, PORT, compare, load_coefficients
from run_portable import numeric_check, run_sequence, sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PORT / "graph/portable/temporal_acceptance")
    parser.add_argument("--source", type=Path, default=PORT / "graph/portable/source_graph.json")
    args = parser.parse_args()
    plan = json.loads(args.source.read_text())
    coefficients = load_coefficients(plan)
    with np.load(args.inputs, allow_pickle=False) as archive:
        frames = sequence(archive["cur"],next(n["shape"] for n in plan["nodes"] if n["name"]=="cur_frame"))
        maps = {key: archive[key].copy() for key in ("dnr_zoom_map", "fusion_zoom_map")}
    if len(frames) < 8:
        raise ValueError("temporal regression uses eight frames")
    frames = frames[:8]
    args.out.mkdir(parents=True, exist_ok=True)
    model = H13Model(plan, coefficients, args.out / "h13")
    try:
        base, trace = run_sequence(plan, coefficients, frames, None, maps, model, reset_indices=(0, 4))
        repeated, _ = run_sequence(plan, coefficients, frames, None, maps, model, reset_indices=(0, 4))
        changed = frames.copy()
        changed[2, :, :4, 128:256, 256:512] += np.float16(0.125)
        perturbed, _ = run_sequence(plan, coefficients, changed, None, maps, model, reset_indices=(0, 4))
        cpu, _ = run_sequence(plan, coefficients, frames, None, maps, reset_indices=(0, 4))
        continuous, _ = run_sequence(plan, coefficients, frames, None, maps, model, reset_indices=(0,))
        continuous_cpu, _ = run_sequence(plan, coefficients, frames, None, maps, reset_indices=(0,))
    finally:
        model.close()
    metrics = {}
    for key in plan["outputs"]:
        metrics[key] = {
            "identical_replay": bool(np.array_equal(base[key], repeated[key])),
            "earlier_frames_unchanged": bool(np.array_equal(base[key][:2], perturbed[key][:2])),
            "same_frame_perturbation": compare(perturbed[key][2], base[key][2]),
            "after_reset_identical": bool(np.array_equal(base[key][4:], perturbed[key][4:])),
            "independently_recurrent_cpu": compare(base[key], cpu[key]),
        }
        metrics[key]["independently_recurrent_cpu"].update(numeric_check(base[key], cpu[key]))
    history_links = [{"frame": row["frame"], "passed": row["history_sha256"] ==
                      trace[row["frame"] - 1]["outputs"]["state_yuv"]["sha256_fp16"]}
                     for row in trace if not row["reset"]]
    passed = all(row["identical_replay"] and row["earlier_frames_unchanged"] and row["after_reset_identical"]
                 for row in metrics.values())
    passed &= all(row["passed"] for row in history_links)
    passed &= all(row["independently_recurrent_cpu"]["passed"] for row in metrics.values())
    passed &= all(metrics[key]["same_frame_perturbation"]["max_abs"] > 0 for key in ("out_yuv", "state_yuv"))
    continuous_metrics = {key: numeric_check(continuous[key], continuous_cpu[key]) for key in plan["outputs"]}
    continuous_passed = all(row["passed"] for row in continuous_metrics.values())
    report = {"passed": passed and continuous_passed, "reset_and_causality_passed": passed,
              "source_graph_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "execution_profile": plan.get("execution_profile","ane"),
              "continuous_recurrence_passed": continuous_passed, "continuous_recurrence": continuous_metrics,
              "frames": 8, "reset_indices": [0, 4], "perturbed_frame": 2,
              "numeric_gate": {"max_abs": 0.02, "mean_abs": 0.001, "finite": True},
              "input_file_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              "history_links": history_links, "outputs": metrics,
              "host_policy": "previous state_yuv, first-frame initialization at reset",
              "scope": "implemented host-state behavior, native ISP host feedback remains unverified"}
    (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
