#!/usr/bin/env python3
"""Probed H19 sparse stride-2 2x2/3x3 kernels with P=1/2 output packing.

The two mask bits per tap select the two output slots. P=1 uses even bits.
For P=2 both slots participate, with payload ordered input/tap/output. Biases
come first, followed by the count field, a reserved zero byte and interleaved
mask pairs plus their payloads. The length field includes the complete mask
byte count and FP16 payload bytes. Odd 3x3 stride-2 kernels split input parity
before applying the 2x2 mask format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from h19_sparse_unpack import load_expected_blobfile
from h19_weightlib import kernel_section, parse_hwx

ROOT = pathlib.Path(__file__).resolve().parent


def mask_tiles(cin, kh, kw):
    return [(ci, ky, kx) for ci in range(cin)
            for ky in range(0, kh, 2) for kx in range(0, kw, 2)]


def tap_position(base_y, base_x, tap, kh, kw):
    # Odd stride-2 kernels are split by input parity before compression.
    # 3x3 therefore has four 2x2 sub-kernels, including padded tap slots.
    if (kh, kw) == (3, 3):
        return base_y // 2 + (tap // 2) * 2, base_x // 2 + (tap % 2) * 2
    return base_y + tap // 2, base_x + tap % 2


def region_size(raw, packing):
    cursor = 0
    for count in packing:
        header = cursor + count * 2
        if header + 3 > len(raw):
            raise ValueError("truncated sparse region header")
        code = int.from_bytes(raw[header:header + 2], "little")
        cursor += (count * 2 + 3 + code + 63) & ~63
    if cursor > len(raw):
        raise ValueError("sparse region exceeds its coefficient section")
    return cursor


def decode_block(raw, packing, cin=6, kh=2, kw=2):
    if (kh, kw) not in ((2, 2), (3, 3)):
        raise ValueError("this codec covers the probed stride-2 2x2 and 3x3 modes")
    if packing not in (1, 2) or not raw or len(raw) % 64:
        raise ValueError("validated sparse blocks have P=1/2 and 64-byte alignment")
    tiles = mask_tiles(cin, kh, kw)
    mask_bytes = (len(tiles) + 1) & ~1
    bias = np.frombuffer(raw[:packing * 2], dtype="<f2").copy()
    code = int.from_bytes(raw[packing * 2:packing * 2 + 2], "little")
    if code < mask_bytes or (code - mask_bytes) % 2 or raw[packing * 2 + 2] != 0:
        raise ValueError("unrecognized sparse header")
    expected_count = (code - mask_bytes) // 2
    if expected_count > packing * cin * kh * kw:
        raise ValueError("sparse count exceeds the tensor capacity")
    weights = np.zeros((packing, cin, kh, kw), dtype=np.float16)
    cursor, count = packing * 2 + 3, 0
    for first_tile in range(0, mask_bytes, 2):
        if count == expected_count:
            break
        if cursor + 2 > len(raw):
            raise ValueError("truncated sparse mask")
        masks = raw[cursor:cursor + 2]
        cursor += 2
        if packing == 1 and any(mask & 0xAA for mask in masks):
            raise ValueError("single-output block selected an inactive output slot")
        selected = []
        for relative_tile, mask in enumerate(masks):
            index = first_tile + relative_tile
            if index >= len(tiles):
                if mask:
                    raise ValueError("sparse mask selected a padded tile")
                continue
            ci, base_y, base_x = tiles[index]
            for tap in range(4):
                for output in range(packing):
                    if mask & (1 << (2 * tap + output)):
                        ky, kx = tap_position(base_y, base_x, tap, kh, kw)
                        if ky >= kh or kx >= kw:
                            raise ValueError("sparse mask selected a padded kernel tap")
                        selected.append((ci, ky, kx, output))
        for ci, ky, kx, output in selected:
            if cursor + 2 > len(raw) or count >= expected_count:
                raise ValueError("sparse payload exceeds its declared count")
            weights[output, ci, ky, kx] = np.frombuffer(raw[cursor:cursor + 2], dtype="<f2")[0]
            cursor += 2
            count += 1
    if count != expected_count or any(raw[cursor:]):
        raise ValueError("sparse payload count or trailing padding is inconsistent")
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("non-finite sparse coefficients")
    return weights, bias


def encode_block(weights, bias):
    packing, cin, kh, kw = weights.shape
    if packing not in (1, 2) or (kh, kw) not in ((2, 2), (3, 3)):
        raise ValueError("unsupported sparse block shape")
    tiles = mask_tiles(cin, kh, kw)
    mask_bytes = (len(tiles) + 1) & ~1
    nonzero_count = int(np.count_nonzero(weights))
    raw = bytearray(np.asarray(bias, dtype="<f2").tobytes())
    raw.extend((mask_bytes + 2 * nonzero_count).to_bytes(2, "little"))
    raw.append(0)
    count = 0
    for first_tile in range(0, mask_bytes, 2):
        if count == nonzero_count:
            break
        masks, values = [], []
        for relative_tile in range(2):
            mask = 0
            index = first_tile + relative_tile
            if index >= len(tiles):
                masks.append(0)
                continue
            ci, base_y, base_x = tiles[index]
            for tap in range(4):
                ky, kx = tap_position(base_y, base_x, tap, kh, kw)
                if ky >= kh or kx >= kw:
                    continue
                for output in range(packing):
                    value = weights[output, ci, ky, kx]
                    if value != 0:
                        mask |= 1 << (2 * tap + output)
                        values.append((ci, ky, kx, output, value))
            masks.append(mask)
        raw.extend(masks)
        raw.extend(np.asarray([row[4] for row in values], dtype="<f2").tobytes())
        count += len(values)
    size = (packing * 2 + 3 + mask_bytes + 2 * nonzero_count + 63) & ~63
    return bytes(raw).ljust(size, b"\0")


def decode(raw, packing, input_shape=(6, 2, 2)):
    if len(raw) != region_size(raw, packing):
        raise ValueError("sparse region length disagrees with its block count")
    weights, biases = [], []
    cursor = 0
    for count in packing:
        code = int.from_bytes(raw[cursor + count * 2:cursor + count * 2 + 2], "little")
        size = (count * 2 + 3 + code + 63) & ~63
        weight, bias = decode_block(raw[cursor:cursor + size], count, *input_shape)
        weights.append(weight)
        biases.append(bias)
        cursor += size
    return np.concatenate(weights), np.concatenate(biases)


def regression():
    rows = []
    for manifest in sorted((ROOT / "h19_sparse_probes").glob("*/manifest.json")):
        item = json.loads(manifest.read_text())
        shape = tuple(item["shape"])
        default_stride = 2 if shape[2:] == (2, 2) else 1
        if shape not in ((6, 6, 2, 2), (24, 6, 2, 2), (32, 25, 3, 3), (32, 32, 3, 3)) or item.get("stride", default_stride) != 2:
            continue
        expected_w, expected_b = load_expected_blobfile(manifest.parent / "weights.bin", item["blob_offsets"], shape)
        data, sections, symbols = parse_hwx(manifest.parent / "out/model.hwx")
        _, raw = kernel_section(data, sections)
        if shape[0] == 6:
            packing = [1] * 6
        elif shape[0] == 24:
            packing = [2] * 8 + [1] * 8
        else:
            packing = [2] * 16
        region = raw[:region_size(raw, packing)]
        weights, bias = decode(region, packing, shape[1:])
        packed = b"".join(encode_block(weights[sum(packing[:i]):sum(packing[:i + 1])],
                                      bias[sum(packing[:i]):sum(packing[:i + 1])])
                          for i in range(len(packing)))
        passed = (np.array_equal(weights.view(np.uint16), expected_w.view(np.uint16))
                  and np.array_equal(bias.view(np.uint16), expected_b.view(np.uint16))
                  and packed == region)
        rows.append({"case": manifest.parent.name, "shape": shape, "passed": passed,
                     "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest()})
    if not rows:
        raise ValueError("no controlled sparse probes found")
    return {"cases": rows, "passed": all(row["passed"] for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "llvnr_port/artifacts/sparse_regression.json")
    args = parser.parse_args()
    report = regression()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["cases"]:
        print(row["case"], "passed", row["passed"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
