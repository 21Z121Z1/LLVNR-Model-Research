#!/usr/bin/env python3
"""Execute recovered LLVNR task arithmetic with explicit native storage views.

This reference uses recovered coefficients throughout. It records its remaining
arithmetic assumptions and has no synthetic replacement weights or operands.
Synthetic external inputs are available for controlled bring-up. H19 hardware
numeric parity and ISP preprocessing remain separate acceptance requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

WORK = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
from h19_weightlib import kernel_groups, parse_hwx
from llvnr_ir import DEFAULT_MODEL, export
from llvnr_taskflow import BAR_BYTES, INPUT_BARS, L2_BYTES, OUTPUT_BARS, address, dimensions, indices, storage_view

PORT = WORK / "llvnr_port"


from portable_ops import conv2d, conv_transpose, fp16, fp32, pe, pooling, sigmoid, sigmoid_lut, sqrt_lut


class NativeReference:
    def __init__(self, model=DEFAULT_MODEL):
        self.ir = export(model)
        data, sections, symbols = parse_hwx(model)
        groups = kernel_groups(data, sections, symbols)
        self.groups_by_offset = {min(group.starts): group for group in groups}
        self.activation_tables = {group.stem: np.frombuffer(group.raw[:128], dtype="<f2").copy()
                                  for group in groups if group.kind == "actlut"}
        self.coefficients = {}
        for folder in ("dense_weights", "special_weights", "deconv_weights"):
            manifest = json.loads((PORT / folder / "manifest.json").read_text())
            if manifest["model_sha256"] != self.ir["model_sha256"]:
                raise ValueError(f"model mismatch in {folder}")
            for row in manifest["groups"]:
                group = row["group"]
                path = PORT / folder / group
                weights = np.load(path / "weights.npy").astype(np.float32)
                bias = np.load(path / "bias.npy").astype(np.float32) if (path / "bias.npy").exists() else None
                primitive = row.get("primitive", "conv_transpose" if folder == "deconv_weights" else "conv2d")
                self.coefficients[group] = {"weights": weights, "bias": bias, "primitive": primitive, "evidence": row}
                if primitive == "conv2d_then_sigmoid":
                    native = next(item for item in groups if item.stem == group)
                    self.activation_tables[group] = np.frombuffer(native.raw[:128], dtype="<f2").copy()
        special = json.loads((PORT / "special_weights/manifest.json").read_text())
        self.luts = {row["group"]: row for row in special["activation_luts"]}
        self.storage = {}
        self.trace = []
        self.captures = {}

    def read(self, view):
        if (view["space"],view["base"]) in getattr(self,"goc_vectors",set()):
            view = {**view,"channel_stride":2,"row_stride":0,"x_stride":0}
        result = self.storage[view["space"]][indices(view)].astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError(f"uninitialized or non-finite read from {view}")
        if "retained_l2_input" in view:
            self.storage["l2"][indices(view["retained_l2_input"])] = result.astype(np.float16)
        return result

    def neuron(self, task, registers, x, target):
        cfg = registers.get(0x1242, 0)
        addr = address(task, 0x1544)
        record = None
        if addr:
            group = self.groups_by_offset[addr["offset"]]
            record = self.coefficients[group.stem]
        if record:
            primitive = record["primitive"]
            evidence = record["evidence"]
            weights, bias = record["weights"], record["bias"]
            if primitive in ("padding_identity", "padding_identity_then_lut"):
                if "padding" in evidence:
                    top, bottom, left, right = evidence["padding"]
                    value = np.pad(x, ((0, 0), (top, bottom), (left, right)), mode=evidence["padding_mode"])
                    reference_origin = evidence.get("native_padding_reference_top_left")
                    start_y = start_x = 0
                    if reference_origin is not None:
                        start_y = reference_origin[0] - task["geometry"]["pad_top"]
                        start_x = reference_origin[1] - task["geometry"]["pad_left"]
                        if min(start_y, start_x) < 0:
                            raise ValueError("unprobed padding origin before controlled source")
                    value = value[:, start_y:start_y + target[1], start_x:start_x + target[2]]
                else:
                    delta_h = target[1] - x.shape[1]
                    delta_w = target[2] - x.shape[2]
                    if min(delta_h, delta_w) < 0:
                        raise ValueError("padding identity target is smaller than its source")
                    # Most compiler padding layouts are symmetric. For an odd
                    # remainder, the task geometry identifies the leading edge;
                    # assign the single extra row/column there instead of
                    # silently dropping it with floor division.
                    if delta_h % 2:
                        top = min(task["geometry"].get("pad_top", 0), delta_h)
                        bottom = delta_h - top
                    else:
                        top = bottom = delta_h // 2
                    if delta_w % 2:
                        left = min(task["geometry"].get("pad_left", 0), delta_w)
                        right = delta_w - left
                    else:
                        left = right = delta_w // 2
                    value = np.pad(x, ((0, 0), (top, bottom), (left, right)),
                                   mode="edge" if cfg & 0x40000 else "constant")
            elif primitive in ("conv_transpose", "conv_transpose_then_lut", "conv_transpose_haar", "nearest_replication"):
                if primitive in ("conv_transpose", "conv_transpose_then_lut"):
                    geometry = task["geometry"]
                    # A valid interior 4x4/stride-2 window begins two output
                    # pixels into the local source expansion. Common padding
                    # moves that origin back one output pixel at a boundary.
                    top_left = (weights.shape[2] - 2 - geometry["pad_top"], weights.shape[3] - 2 - geometry["pad_left"])
                    if evidence.get("native_origin_top_left") is not None:
                        top_left = tuple(evidence["native_origin_top_left"])
                    if evidence.get("native_origin_by_geometry_padding"):
                        key = f'{geometry["pad_top"]},{geometry["pad_left"]}'
                        top_left = tuple(evidence["native_origin_by_geometry_padding"][key])
                else:
                    top_left = (1 if group.stem.startswith(("K5B83DE", "KBD2DD")) else 0, 0)
                value = conv_transpose(x, weights, target, (2, 2), top_left)
                if bias is not None:
                    value += bias[:, None, None]
            else:
                geometry = task["geometry"] or {"stride_y": 1, "stride_x": 1, "pad_top": 0, "pad_left": 0}
                groups = evidence.get("groups", 1)
                if groups == 1:
                    value = conv2d(x, weights, bias, target, (geometry["stride_y"], geometry["stride_x"]),
                                   (geometry["pad_top"], geometry["pad_left"]), bool(cfg & 0x40000))
                else:
                    ci, co = x.shape[0] // groups, target[0] // groups
                    value = np.concatenate([conv2d(x[g*ci:(g+1)*ci], weights[g*co:(g+1)*co],
                        bias[g*co:(g+1)*co] if bias is not None else None, (co, *target[1:]),
                        (geometry["stride_y"], geometry["stride_x"]), (geometry["pad_top"], geometry["pad_left"]),
                        bool(cfg & 0x40000)) for g in range(groups)])
        else:
            if cfg & 7 not in (0, 4):
                raise ValueError(f"unimplemented coefficient-free NE mode {cfg & 7}")
            value = (x + fp16(registers.get(0x1244, 0))) * fp16(registers.get(0x1245, 0x3C00))
        nonlinear = cfg >> 16 & 3
        if nonlinear == 1:
            value = np.maximum(value, 0)
        elif nonlinear == 2:
            lut_address = address(task, 0x158E)
            if lut_address:
                lut = self.luts[self.groups_by_offset[lut_address["offset"]].stem]
                if lut["operation"] == "clip":
                    value = np.clip(value, *lut["bounds"])
                elif lut["operation"] in ("sigmoid", "scaled_sigmoid"):
                    value = sigmoid_lut(value, self.activation_tables[lut["group"]])
                elif lut["operation"] == "sqrt":
                    value = sqrt_lut(value, self.activation_tables[self.groups_by_offset[lut_address["offset"]].stem])
                else:
                    raise ValueError(f"unimplemented LUT {lut}")
            elif record and record["primitive"] in ("conv2d_then_sigmoid", "conv2d_then_lut", "conv_transpose_then_lut", "padding_identity_then_lut"):
                lut = self.luts.get(group.stem)
                value = np.clip(value, *lut["bounds"]) if lut and lut["operation"] == "clip" else sigmoid_lut(value, self.activation_tables[group.stem])
            else:
                raise ValueError("NE lookup activation without identified table")
        elif nonlinear:
            raise ValueError(f"unimplemented NE nonlinearity {nonlinear}")
        return value, record["primitive"] if record else "ne_affine_or_bypass"

    def run(self, inputs, stop_task=None, capture_tasks=()):
        bar_bytes = getattr(self, "bar_bytes", BAR_BYTES)
        input_bars = getattr(self, "input_bars", INPUT_BARS)
        output_bars = getattr(self, "output_bars", OUTPUT_BARS)
        self.storage = {str(bar): np.full(size // 2, np.nan, dtype=np.float16) for bar, size in bar_bytes.items()}
        self.storage["l2"] = np.full(L2_BYTES // 2, np.nan, dtype=np.float16)
        for bar, raw in getattr(self,"static_bars",{}).items():
            self.storage[str(bar)] = np.frombuffer(raw,dtype='<f2').copy()
        for bar, name in input_bars.items():
            array = np.asarray(inputs[name], dtype=np.float16)
            view = getattr(self, "port_views", {}).get(name)
            if not np.isfinite(array).all() or (view is None and array.size != self.storage[str(bar)].size):
                raise ValueError(f"external input contract mismatch for {name}")
            if view:
                if tuple(array.shape) != (1, *view["shape_chw"]):
                    raise ValueError(f"external input shape mismatch for {name}")
                self.storage[str(bar)].fill(0)
                self.storage[str(bar)][indices(view)] = array[0]
            else:
                self.storage[str(bar)][:] = array.reshape(-1)
        self.trace = []
        self.captures = {}
        self.goc_vectors = set()
        capture_tasks = set(capture_tasks)
        for task in self.ir["tasks"]:
            if stop_task is not None and task["index"] > stop_task:
                break
            started = time.perf_counter()
            registers = {int(key, 16): value for key, value in task["written_registers"].items()}
            source_shape, target = dimensions(registers), dimensions(registers, True)
            geometry = task["geometry"]
            if geometry:
                target = (target[0], target[1] * max(1, geometry["output_y_factor"]), target[2] * max(1, geometry["output_x_factor"]))
            if task["task_type_code"] == 5:
                target = (target[0],1,1)
            if task["index"] == 2 and task["task_type_code"] == 1:
                canvas_w, canvas_h = getattr(self, "canvas_wh", (2304, 1296))
                source_shape = (1, canvas_h // 4, canvas_w // 4)
            source = storage_view(task, registers, source_shape, "source1")
            destination = storage_view(task, registers, target, "output")
            if task["task_type_code"] == 4:
                destination = {**destination,"channel_stride":2}
            try:
                x = self.read(source)
                kind = task["task_type_code"]
                if kind == 0:
                    result, primitive = self.neuron(task, registers, x, target)
                elif kind == 1:
                    result = pooling(x, target, geometry, registers.get(0x1140, 0) & 3, 2 if task["index"] == 2 else 0)
                    result = pe(result, None, registers, kind)
                    primitive = "pool_then_pe"
                elif kind in (2, 3):
                    other = self.read(storage_view(task, registers, source_shape, "source2")) if kind == 3 else None
                    result = pe(x, other, registers, kind)
                    primitive = "pe_binary" if kind == 3 else "pe_unary"
                elif kind == 4:
                    # Controlled goc_vector_{coded,scale} and goc_onehot prove
                    # source2 consists of C offsets followed by C multipliers.
                    a = address(task,0x134c)
                    if a is None or a["bar"] != 4 or source_shape[1:] != (1,1):
                        raise ValueError("unprobed GOC source contract")
                    channels = source_shape[0]
                    const = self.storage["4"][a["offset"]//2:a["offset"]//2+channels*2].astype(np.float32)
                    if const.shape != (channels*2,):
                        raise ValueError("truncated GOC affine constants")
                    result = pe(x*const[channels:,None,None]+const[:channels,None,None],None,registers,kind)
                    primitive = "goc_channel_affine"
                elif kind == 5:
                    result = pe(x.sum(axis=(1,2),keepdims=True,dtype=np.float32),None,registers,kind)
                    primitive = "spatial_sum_then_pe"
                else:
                    raise ValueError(f"unimplemented task type {kind}")
                if result.shape != target:
                    raise ValueError(f"result shape {result.shape} != {target}")
                rounded = result.astype(np.float16)
                if not np.isfinite(rounded).all():
                    raise ValueError("non-finite task output")
                self.storage[destination["space"]][indices(destination)] = rounded
                if kind == 4:
                    self.goc_vectors.add((destination["space"],destination["base"]))
            except Exception as error:
                self.trace.append({"task": task["index"], "status": "failed", "error": str(error)})
                raise RuntimeError(f"task {task['index']}: {error}") from error
            self.trace.append({"task": task["index"], "status": "ok", "primitive": primitive,
                               "shape_chw": list(target), "seconds": time.perf_counter() - started,
                               "min": float(rounded.min()), "max": float(rounded.max()),
                               "mean": float(rounded.astype(np.float32).mean()),
                               "sha256_fp16": hashlib.sha256(rounded.tobytes()).hexdigest()})
            if task["index"] in capture_tasks:
                self.captures[task["index"]] = rounded.copy()
            if task["index"] % 25 == 0:
                print(f"task {task['index']:03d} {primitive} {target} {self.trace[-1]['seconds']:.3f}s", flush=True)
        shapes = {36: (1, 6, 648, 1152), 40: (1, 1, 324, 576), 44: (1, 6, 648, 1152)}
        if hasattr(self, "port_views"):
            return {name: self.storage[str(bar)][indices(self.port_views[name])][None].copy() for bar, name in output_bars.items()}
        return {name: self.storage[str(bar)].reshape(shapes[bar]).copy() for bar, name in output_bars.items()}


def synthetic_inputs(seed, canvas_wh=(2304, 1296)):
    rng = np.random.default_rng(seed)
    width, height = canvas_wh
    image = rng.normal(0.2, 0.015, (1, 6, height // 2, width // 2)).astype(np.float32)
    image[:, 4:] -= 0.2
    current = image + rng.normal(0, 0.005, image.shape)
    return {"his_frame": image.astype(np.float16), "cur_frame": current.astype(np.float16),
            "dnr_zoom_map": np.full((1, 1, height // 4, width // 4), 0.5, dtype=np.float16),
            "fusion_zoom_map": np.full((1, 1, height // 4, width // 4), 0.5, dtype=np.float16)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=PORT / "graph/native_reference_run")
    parser.add_argument("--stop-task", type=int)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    net = NativeReference()
    inputs = synthetic_inputs(args.seed)
    started = time.perf_counter()
    error = None
    try:
        outputs = net.run(inputs, args.stop_task)
    except RuntimeError as failure:
        error, outputs = str(failure), {}
    elapsed = time.perf_counter() - started
    report = {"model_sha256": net.ir["model_sha256"], "backend": "NumPy FP32 arithmetic with FP16 task buffers",
              "input": {"kind": "controlled synthetic external inputs", "seed": args.seed,
                        "sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()}},
              "seconds": elapsed, "tasks_executed": sum(row["status"] == "ok" for row in net.trace),
              "error": error, "trace": net.trace,
              "remaining_validation": ["H19 hardware numeric parity", "hardware LUT interpolation rounding",
                                       "PE fused arithmetic rounding order", "transpose trailing staging cells",
                                       "task 2 virtual DMA padding", "ISP preprocessing and host temporal feedback"]}
    if outputs:
        np.savez_compressed(args.out / "outputs.npz", **outputs)
        report["outputs"] = {key: {"shape": list(value.shape), "finite": bool(np.isfinite(value).all()),
                                   "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for key, value in outputs.items()}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("trace", "input")}, indent=2), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
