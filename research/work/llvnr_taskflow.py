#!/usr/bin/env python3
"""Trace explicitly addressed tensor storage in the 535-task LLVNR program.

Each FP16 cell stores its producer ID. This validates storage coverage and
recovers candidate dependencies, independently of unimplemented arithmetic.
Missing register words use per-task defaults, never the preceding task's words.
All uninitialized reads and unsupported layouts are retained in the report.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from llvnr_ir import DEFAULT_MODEL, ROOT, export

INPUT_BARS = {20: "cur_frame", 24: "dnr_zoom_map", 28: "fusion_zoom_map", 32: "his_frame"}
OUTPUT_BARS = {36: "out_yuv", 40: "state_weight_band0", 44: "state_yuv"}
BAR_BYTES = {12: 45093888, 20: 8957952, 24: 373248, 28: 373248,
             32: 8957952, 36: 8957952, 40: 373248, 44: 8957952}
L2_BYTES = 4194304  # The native HWX records --memcache-size=4194304.


def dimensions(registers, output=False):
    base = 5 if output else 1
    # Controlled compiler tasks omit dimensions equal to the reset value 16.
    return tuple(registers.get(base + index, 16) for index in (2, 1, 0))


def offsets(shape, base, channel_stride, row_stride, interleave=1, row_wrap=None, x_stride=None):
    channels, height, width = shape
    c = np.arange(channels, dtype=np.int64)[:, None, None]
    y = np.arange(height, dtype=np.int64)[None, :, None]
    if row_wrap:
        y = (y + row_wrap["start"]) % row_wrap["rows"]
    x = np.arange(width, dtype=np.int64)[None, None, :]
    return base + (c // interleave) * channel_stride + y * row_stride + x * (2 * interleave if x_stride is None else x_stride) + (c % interleave) * 2


def address(task, register):
    return next((row for row in task["address_writes"] if row["register_word"] == register), None)


def storage_view(task, registers, shape, role):
    if role == "output":
        cfg, base, channel, row = (0x1052, 0x1053, 0x1054, 0x1055)
        dma_address, dma_channel, dma_row, dma_format, dma_crop = (0x1444, 0x1447, 0x1446, 0x1450, 0x1457)
    elif role == "source2":
        cfg, base, channel, row = (0x1042, 0x1049, 0x104A, 0x104B)
        dma_address, dma_channel, dma_row, dma_format, dma_crop = (0x134C, 0x134F, 0x134E, 0x135D, 0x1367)
    else:
        cfg, base, channel, row = (0x1041, 0x1044, 0x1045, 0x1046)
        dma_address, dma_channel, dma_row, dma_format, dma_crop = (0x1346, 0x1349, 0x1348, 0x135C, 0x1363)
    mode = registers.get(cfg, 0) & 3
    if mode == 0:
        view = {"space": "l2", "base": registers.get(base, 0), "interleave": 1,
                "channel_stride": registers.get(channel, 0), "row_stride": registers.get(row, 0)}
        shift, wrap_register = {"source1": (0, 0x105A), "source2": (4, 0x105B), "output": (8, 0x105D)}[role]
        wrap_mode = registers.get(0x1059, 0) >> shift & 7
        if wrap_mode:
            if wrap_mode != 2:
                raise ValueError(f"{role}: unsupported L2 wrap dimension {wrap_mode}")
            packed = registers.get(wrap_register, 0)
            wrap_index, start = packed & 65535, packed >> 16
            if not wrap_index + start:
                raise ValueError(f"{role}: empty L2 row ring")
            view["row_wrap"] = {"index": wrap_index, "start": start, "rows": wrap_index + start}
    elif mode in (1, 2):
        addr = address(task, dma_address)
        if addr is None:
            raise ValueError(f"{role}: DMA mode without explicit BAR address")
        view = {"space": str(addr["bar"]), "base": addr["offset"],
                "interleave": registers.get(dma_format, 0x3211) >> 4 & 15,
                "channel_stride": registers.get(dma_channel, 0), "row_stride": registers.get(dma_row, 0)}
        if view["interleave"] == 0:
            raise ValueError(f"{role}: unsupported zero DMA interleave")
        crop = registers.get(dma_crop, 0)
        view["crop_x"], view["crop_y"] = crop >> 16, crop & 65535
        view["base"] += view["crop_y"] * view["row_stride"] + view["crop_x"] * 2 * view["interleave"]
        if mode == 1:
            view["retained_l2_input"] = {"space": "l2", "base": registers.get(base, 0), "interleave": 1,
                                         "channel_stride": registers.get(channel, 0), "row_stride": registers.get(row, 0),
                                         "shape_chw": list(shape)}
    else:
        raise ValueError(f"{role}: unsupported L2 access mode {mode}")
    view["shape_chw"] = list(shape)
    if role != "output":
        broadcast = registers.get(0x12,0) >> (4 if role=="source2" else 0) & 15
        if broadcast & 1:
            view["x_stride"] = 0
        if broadcast & 2:
            view["row_stride"] = 0
    return view


def indices(view):
    result = offsets(view["shape_chw"], view["base"], view["channel_stride"],
                     view["row_stride"], view["interleave"], view.get("row_wrap"), view.get("x_stride"))
    if np.any(result & 1):
        raise ValueError("unaligned FP16 cell address")
    return result // 2


def run(ir):
    storage = {str(bar): np.zeros(size // 2, dtype=np.int16) for bar, size in BAR_BYTES.items()}
    storage["l2"] = np.zeros(L2_BYTES // 2, dtype=np.int16)
    for bar in INPUT_BARS:
        storage[str(bar)].fill(-bar)
    tasks = []
    for task in ir["tasks"]:
        registers = {int(key, 16): value for key, value in task["written_registers"].items()}
        input_shape, output_shape = dimensions(registers), dimensions(registers, True)
        geometry = task["geometry"]
        if geometry:
            output_shape = (output_shape[0], output_shape[1] * max(geometry["output_y_factor"], 1),
                            output_shape[2] * max(geometry["output_x_factor"], 1))
        row = {"task": task["index"], "task_type": task["task_type_code"], "input_shape_chw": input_shape,
               "output_shape_chw": output_shape, "sources": [], "issues": []}
        # The first mask pool's DMA source remains 576x324 while Common records
        # its virtual 2-pixel border. Keep this exception explicit for review.
        if task["index"] == 2:
            input_shape = (1, 324, 576)
            row["virtual_input_padding"] = {"pixels": 2, "validation": "pending exact DMA-padding decode"}
        roles = ["source1"]
        if task["task_type_code"] == 3:
            roles.append("source2")
        parents = set()
        for role in roles:
            try:
                view = storage_view(task, registers, input_shape, role)
                memory = storage[view["space"]]
                idx = indices(view)
                valid = (idx >= 0) & (idx < memory.size)
                values = np.zeros(idx.shape, dtype=np.int16)
                values[valid] = memory[idx[valid]]
                if "retained_l2_input" in view:
                    storage["l2"][indices(view["retained_l2_input"])] = values
                labels, counts = np.unique(values, return_counts=True)
                producers = {str(int(label) - 1) if label > 0 else INPUT_BARS.get(-int(label), "uninitialized"):
                             int(count) for label, count in zip(labels, counts)}
                parents.update(int(label) - 1 for label in labels if label > 0)
                view.update({"role": role, "producer_cells": producers,
                             "out_of_bounds_cells": int(np.count_nonzero(~valid)),
                             "uninitialized_cells": int(np.count_nonzero(values == 0))})
                if not valid.all() or np.any(values == 0):
                    row["issues"].append(f"{role}: storage coverage incomplete")
                row["sources"].append(view)
            except (ValueError, KeyError) as error:
                row["issues"].append(str(error))
        try:
            view = storage_view(task, registers, output_shape, "output")
            idx = indices(view)
            memory = storage[view["space"]]
            valid = (idx >= 0) & (idx < memory.size)
            view["out_of_bounds_cells"] = int(np.count_nonzero(~valid))
            if not valid.all():
                row["issues"].append("output: storage bounds exceeded")
            memory[idx[valid]] = task["index"] + 1
            row["output"] = view
        except (ValueError, KeyError) as error:
            row["issues"].append(str(error))
        row["candidate_predecessor_tasks"] = sorted(parents)
        row["local_storage_coverage_passed"] = not row["issues"]
        row["transitive_storage_coverage_passed"] = not row["issues"] and all(
            tasks[parent]["transitive_storage_coverage_passed"] for parent in parents)
        tasks.append(row)
    outputs = {}
    for bar, name in OUTPUT_BARS.items():
        values, counts = np.unique(storage[str(bar)], return_counts=True)
        outputs[name] = {"initialized_cells": int(np.count_nonzero(storage[str(bar)])),
                         "total_cells": int(storage[str(bar)].size),
                         "producer_cells": {str(int(label) - 1): int(count)
                                            for label, count in zip(values, counts) if label > 0}}
    return {"model_sha256": ir["model_sha256"], "task_count": len(tasks),
            "local_storage_coverage_passed": sum(row["local_storage_coverage_passed"] for row in tasks),
            "transitive_storage_coverage_passed": sum(row["transitive_storage_coverage_passed"] for row in tasks),
            "interpretation": "candidate storage dependencies, arithmetic and circular-buffer semantics require independent validation",
            "explicit_assumptions": ["missing common dimensions use compiler reset value 16",
                                     "cached L2 tensors use linear FP16 columns and explicit row/channel strides",
                                     "DMA interleave is format bits 4..7",
                                     "L2 capacity is the native compiler's 4 MiB memcache setting",
                                     "L2 height rings use explicit wrap config, wrap index and start offset",
                                     "DMA source mode 1 retains the input at its explicit L2 layout for later reads",
                                     "task 2 virtual source padding is recorded separately"],
            "outputs": outputs, "tasks": tasks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "llvnr_port/artifacts/taskflow.json")
    args = parser.parse_args()
    report = run(export(args.model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))
    print("first issues", [(row["task"], row["issues"]) for row in report["tasks"] if row["issues"]][:20])


if __name__ == "__main__":
    main()
