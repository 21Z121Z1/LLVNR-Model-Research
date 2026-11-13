#!/usr/bin/env python3
"""Run the recovered full graph on a supplied frame sequence, with explicit host state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from portable_graph import H13Model, PORT, evaluate_cpu, load_coefficients


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sequence(array, frame_shape=(1, 6, 648, 1152)):
    array = np.asarray(array, dtype=np.float16)
    if array.ndim == 4:
        array = array[:, None]
    if array.ndim != 5 or array.shape[1:] != tuple(frame_shape) or not len(array):
        raise ValueError(f"frame sequence must have shape [T,{','.join(map(str,frame_shape))}]")
    if not np.isfinite(array).all():
        raise ValueError("frame sequence must be finite")
    return array


def numeric_check(actual, expected):
    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    finite = bool(np.isfinite(actual).all())
    maximum, mean = float(error.max()), float(error.mean())
    return {"max_abs": maximum, "mean_abs": mean, "finite": finite,
            "passed": finite and maximum <= 0.02 and mean <= 0.001}


def run_sequence(plan, coefficients, frames, supplied_history, maps, model=None,
                 history_mode="state_yuv", reset_indices=(), initial_state="first_frame", check_cpu=False):
    reset_indices = set(reset_indices) | {0}
    previous, records = None, []
    accumulated = {key: [] for key in plan["outputs"]}
    for index, current in enumerate(frames):
        reset = index in reset_indices
        if history_mode == "supplied":
            if supplied_history is None or len(supplied_history) != len(frames):
                raise ValueError("supplied history requires one history tensor per frame")
            history = supplied_history[index]
        elif reset:
            history = current.copy() if initial_state == "first_frame" else np.zeros_like(current)
        else:
            history = previous
        feeds = {"cur_frame": current, "his_frame": history}
        for key, array in maps.items():
            feeds[key] = array[index] if array.ndim == 5 else array
        started = time.perf_counter()
        if model is None:
            values = evaluate_cpu(plan, coefficients, feeds)
            outputs = {key: values[key] for key in plan["outputs"]}
        else:
            outputs = model.run(feeds)
        seconds = time.perf_counter() - started
        if not all(np.isfinite(value).all() for value in outputs.values()):
            raise RuntimeError(f"non-finite frame {index}")
        row = {"frame": index, "reset": reset, "history_mode": history_mode,
               "history_sha256": digest(history), "current_sha256": digest(current),
               "host_call_seconds": seconds,
               "outputs": {key: {"sha256_fp16": digest(value), "min": float(value.min()), "max": float(value.max()),
                                 "mean": float(value.astype(np.float32).mean())} for key, value in outputs.items()}}
        if check_cpu and model is not None:
            expected = evaluate_cpu(plan, coefficients, feeds)
            row["cpu_same_input"] = {key: numeric_check(value, expected[key]) for key, value in outputs.items()}
            row["cpu_passed"] = all(metric["passed"] for metric in row["cpu_same_input"].values())
        for key, value in outputs.items():
            accumulated[key].append(value[0].copy())
        if history_mode in ("state_yuv", "out_yuv"):
            previous = outputs[history_mode].copy()
        else:
            previous = current.copy()
        records.append(row)
        print(f"frame {index} reset={reset} seconds={seconds:.6f} cpu={row.get('cpu_passed')}", flush=True)
    return {key: np.stack(values) for key, values in accumulated.items()}, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PORT / "graph/portable/source_graph.json")
    parser.add_argument("--coefficient-root", type=Path, default=PORT)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "h13"), default="cpu")
    parser.add_argument("--history", choices=("state_yuv", "out_yuv", "previous_input", "supplied"), default="state_yuv")
    parser.add_argument("--initial-state", choices=("first_frame", "zero"), default="first_frame")
    parser.add_argument("--reset-indices", default="0")
    parser.add_argument("--zoom-fill", type=float, default=0.5)
    parser.add_argument("--check-cpu", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reset_indices = [int(item) for item in args.reset_indices.split(",") if item]
    plan = json.loads(args.source.read_text())
    coefficients = load_coefficients(plan, args.coefficient_root)
    with np.load(args.inputs, allow_pickle=False) as archive:
        frames = sequence(archive["cur"] if "cur" in archive else archive["cur_frame"])
        supplied = sequence(archive["his"]) if "his" in archive else None
        maps = {key: np.asarray(archive[key], dtype=np.float16) if key in archive else
                np.full((1, 1, 324, 576), args.zoom_fill, dtype=np.float16)
                for key in ("dnr_zoom_map", "fusion_zoom_map")}
        metadata = json.loads(archive["metadata_json"].item()) if "metadata_json" in archive else {}
    if any(index < 0 or index >= len(frames) for index in reset_indices):
        raise ValueError("reset index outside the sequence")
    if not np.isfinite(args.zoom_fill):
        raise ValueError("zoom fill must be finite")
    args.out.mkdir(parents=True, exist_ok=True)
    model = H13Model(plan, coefficients, args.out / "h13") if args.backend == "h13" else None
    try:
        outputs, records = run_sequence(plan, coefficients, frames, supplied, maps, model=model,
                                        history_mode=args.history, reset_indices=reset_indices,
                                        initial_state=args.initial_state, check_cpu=args.check_cpu)
    finally:
        if model:
            model.close()
    np.savez_compressed(args.out / "outputs.npz", **outputs)
    report = {"source_graph_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "firmware_model_sha256": plan["model_sha256"], "nodes": len(plan["nodes"]),
              "input_file_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              "input_array_sha256": digest(frames), "input_metadata": metadata,
              "backend": args.backend, "compile_seconds": model.compile_seconds if model else None,
              "history_mode": args.history, "initial_state": args.initial_state,
              "reset_indices": sorted(set(reset_indices) | {0}),
              "host_feedback_semantics": "experimental policy, ISP host behavior unverified",
              "state_weight_band0_feedback": "recorded output, host map feedback rule unresolved",
              "camera_preprocessing_equivalence": "unverified", "frames": records,
              "cpu_same_input_checked": bool(model is not None and args.check_cpu),
              "cpu_same_input_passed": all(row["cpu_passed"] for row in records) if model is not None and args.check_cpu else None,
              "synthetic_intermediate_operands": 0, "replacement_weights": 0,
              "outputs": {key: {"shape": list(value.shape), "sha256_fp16": digest(value),
                                "finite": bool(np.isfinite(value).all())} for key, value in outputs.items()}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frames": len(records), "cpu_same_input_passed": report["cpu_same_input_passed"],
                      "outputs": report["outputs"]}, indent=2), flush=True)
    return 1 if report["cpu_same_input_passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
