#!/usr/bin/env python3
"""FP16 dense H19 coefficient packing validated by compiler permutations.

A physical coefficient block interleaves P output channels for each input
coefficient. P biases precede that payload when per-channel bias is present.
Each block is padded to 64 bytes. Sparse, palettized and embedded-LUT modes
need their own decoders and are rejected by this aligned dense codec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from h19_statewalk import parse_macho, walk_tasks, words_at
from h19_weightlib import kernel_section, parse_hwx

ROOT = pathlib.Path(__file__).resolve().parent


def dimensions(shape, packing, has_bias):
    cout, cin, kh, kw = shape
    if min(shape) < 1 or packing < 1 or cout % packing:
        raise ValueError("dense mode requires positive dimensions and complete output-channel blocks")
    coefficients = cin * kh * kw
    used = packing * (coefficients + int(has_bias)) * 2
    return cout // packing, coefficients, used, (used + 63) & ~63


def block_layout(shape, packing, has_bias):
    cout, cin, kh, kw = shape
    if isinstance(packing, int):
        dimensions(shape, packing, has_bias)
        counts = [packing] * (cout // packing)
    else:
        counts = list(packing)
        if min(shape) < 1 or not counts or min(counts) < 1 or sum(counts) != cout:
            raise ValueError("variable packing must account for every output channel")
    offset, first_output = 0, 0
    result = []
    for count in counts:
        used = count * (cin * kh * kw + int(has_bias)) * 2
        stride = (used + 63) & ~63
        result.append((offset, first_output, count, used, stride))
        offset += stride
        first_output += count
    return result


def decode(raw, shape, packing, has_bias=True):
    layout = block_layout(shape, packing, has_bias)
    expected = sum(row[4] for row in layout)
    if len(raw) != expected:
        raise ValueError(f"dense region has {len(raw)} bytes, expected {expected}")
    weights = np.empty(shape, dtype=np.float16)
    bias = np.empty(shape[0], dtype=np.float16) if has_bias else None
    coefficients = int(np.prod(shape[1:]))
    for block, (offset, start, count, used, stride) in enumerate(layout):
        lane = raw[offset:offset + stride]
        if any(lane[used:]):
            raise ValueError(f"block {block} has nonzero padding, incompatible mode or shape")
        values = np.frombuffer(lane[:used], dtype="<f2")
        end = start + count
        if has_bias:
            bias[start:end] = values[:count]
        weights[start:end] = values[count * int(has_bias):].reshape(coefficients, count).T.reshape((count,) + tuple(shape[1:]))
    if not np.isfinite(weights).all() or (bias is not None and not np.isfinite(bias).all()):
        raise ValueError("dense decode produced non-finite coefficients")
    return weights, bias


def encode(weights, bias, packing):
    has_bias = bias is not None
    layout = block_layout(weights.shape, packing, has_bias)
    coefficients = int(np.prod(weights.shape[1:]))
    result = bytearray(sum(row[4] for row in layout))
    for offset, start, count, used, stride in layout:
        end = start + count
        payload = weights[start:end].reshape(count, coefficients).T.astype("<f2").tobytes()
        if has_bias:
            payload = np.asarray(bias[start:end], dtype="<f2").tobytes() + payload
        result[offset:offset + used] = payload
    return bytes(result)


def regression(probe_root):
    rows = []
    for manifest in sorted(probe_root.glob("*/layout.json")):
        evidence = json.loads(manifest.read_text())
        shape = tuple(evidence["shape"])
        cout = shape[0]
        data, sections, symbols = parse_hwx(manifest.parent / "out/model.hwx")
        section, raw = kernel_section(data, sections)
        source_weights = np.arange(0x2001, 0x2001 + int(np.prod(shape)), dtype=np.uint16).reshape(shape).view(np.float16)
        source_bias = np.arange(0x7000, 0x7000 + cout, dtype=np.uint16).view(np.float16)
        _, _, _, _, state_sections, _ = parse_macho(manifest.parent / "out/model.hwx")
        tasks = walk_tasks(words_at(data, state_sections, ("__TEXT", "__text")))
        for task in tasks:
            if task.get("bad") or 0x1544 not in task["regs"]:
                continue
            base = task["regs"][0x1544]
            halfwords = np.frombuffer(raw[base:], dtype="<u2")
            has_bias = cout != 1
            packing = []
            first_output, offset = 0, 0
            if has_bias:
                offsets = [int(np.flatnonzero(halfwords == value)[0]) * 2
                           for value in source_bias.view(np.uint16)]
                while first_output < cout:
                    if offsets[first_output] != offset:
                        raise ValueError("compiler bias header is outside the inferred block")
                    count = 1
                    while first_output + count < cout and offsets[first_output + count] == offset + 2 * count:
                        count += 1
                    packing.append(count)
                    offset += ((int(np.prod(shape[1:])) + 1) * count * 2 + 63) & ~63
                    first_output += count
            else:
                packing = [1]
            size = sum(row[4] for row in block_layout(shape, packing, has_bias))
            region = raw[base:base + size]
            weights, bias = decode(region, shape, packing, has_bias)
            bias_source = "coefficient block"
            if not has_bias:
                bias_word = task["regs"].get(0x1244)
                if bias_word is None:
                    raise ValueError("single-channel bias register is missing")
                bias = np.array([bias_word & 0xFFFF], dtype=np.uint16).view(np.float16)
                bias_source = "NE bias register 0x1244"
            passed = (np.array_equal(weights.view(np.uint16), source_weights.view(np.uint16))
                      and np.array_equal(bias.view(np.uint16), source_bias.view(np.uint16))
                      and encode(weights, bias if has_bias else None, packing) == region)
            rows.append({
                "case": manifest.parent.name, "stream": task["stream_index"],
                "shape": shape, "packing": packing, "region": [base, base + size],
                "bias_source": bias_source, "passed": passed,
                "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            })
    if not rows:
        raise ValueError("no dense compiler probes found")
    return {"cases": rows, "passed": all(row["passed"] for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=pathlib.Path, default=ROOT / "h19_dense_probes")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "llvnr_port/artifacts/dense_regression.json")
    args = parser.parse_args()
    report = regression(args.probes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["cases"]:
        print(row["case"], "stream", row["stream"], "P", sorted(set(row["packing"])), "passed", row["passed"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
