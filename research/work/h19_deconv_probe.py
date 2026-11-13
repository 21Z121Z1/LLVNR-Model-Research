#!/usr/bin/env python3
"""Recover H19 4x4 transpose-convolution permutations with compiler oracles.

Two coordinate-coded tensors establish each byte permutation. An independent
signed tensor validates it. A native region is accepted only when every
non-coefficient byte matches the controlled format and re-encoding is exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

from h19_sparse_probe import FLAGS
from h19_weightlib import kernel_groups, kernel_section, parse_hwx
from llvnr_ir import DEFAULT_MODEL

ROOT = pathlib.Path(__file__).resolve().parent
CASES = ((32, 1, 0x9400, ("KB358A6",)),
         (24, 1, 0x7000, ("KD5A651",)),
         (24, 0, 0x6C00, ("K8197D5", "K32C40F")))


def source_bits(cout, seed):
    count = 32 * cout * 16
    values = np.arange(0x2001, 0x2001 + count, dtype=np.uint16)
    if seed:
        np.random.default_rng(20261112 + seed).shuffle(values)
    if seed == 2:
        values[::2] |= 0x8000
    return values.reshape(32, cout, 4, 4)


def compile_case(cout, top, seed, folder):
    sys.path.insert(0, str(ROOT / "ANEForge"))
    import aneforge as af
    from aneforge._compile import _lower_fused_to_dir
    folder.mkdir(parents=True, exist_ok=True)
    weights = source_bits(cout, seed).view(np.float16)
    graph = af.conv_transpose(af.input((1, 32, 81, 144)), weights, stride=2, pad=1)
    _lower_fused_to_dir(graph, folder)
    mil = (folder / "model.mil").read_text().replace("[1, 1, 1, 1]", f"[{top}, {2 - top}, 1, 1]")
    (folder / "model.mil").write_text(mil)
    command = [str(ROOT / "anecc"), str(folder), str(folder / "out"), "h19", *FLAGS]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (folder / "compiler.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"compiler exited {result.returncode}, see {folder / 'compiler.log'}")
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")


def skeleton(raw, offsets):
    result = bytearray(raw)
    for offset in offsets:
        result[offset:offset + 2] = b"\0\0"
    return bytes(result)


def learn(cout, top, region_bytes, folder):
    regions = []
    for seed in range(3):
        data, sections, symbols = parse_hwx(folder / f"coded{seed}/out/model.hwx")
        _, raw = kernel_section(data, sections)
        regions.append(raw[:region_bytes])
    first_offsets = {}
    for offset in range(len(regions[0]) - 1):
        bits = int.from_bytes(regions[0][offset:offset + 2], "little")
        first_offsets.setdefault(bits, []).append(offset)
    first, second, heldout = [source_bits(cout, seed).reshape(-1) for seed in range(3)]
    offsets = []
    for index, bits in enumerate(first):
        candidates = [offset for offset in first_offsets.get(int(bits), [])
                      if int.from_bytes(regions[1][offset:offset + 2], "little") == int(second[index])]
        if len(candidates) != 1:
            raise ValueError(f"coefficient {index} has {len(candidates)} candidate byte locations")
        offsets.append(candidates[0])
    occupied = [byte for offset in offsets for byte in (offset, offset + 1)]
    if len(set(occupied)) != len(occupied):
        raise ValueError("coefficient byte locations overlap")
    template = skeleton(regions[0], offsets)
    if any(skeleton(region, offsets) != template for region in regions[1:]):
        raise ValueError("controlled format changed between source tensors")
    decoded = np.array([int.from_bytes(regions[2][offset:offset + 2], "little") for offset in offsets], dtype=np.uint16)
    if not np.array_equal(decoded, heldout):
        raise ValueError("independent signed-tensor validation failed")
    report = {
        "shape": [32, cout, 4, 4], "stride": 2, "pad": [top, 2 - top, 1, 1],
        "region_bytes": region_bytes, "weight_byte_offsets": offsets,
        "skeleton_sha256": hashlib.sha256(template).hexdigest(),
        "heldout_bitwise_equal": True,
    }
    (folder / "layout.json").write_text(json.dumps(report, indent=1) + "\n")
    return report, template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-probes", action="store_true")
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--probes", type=pathlib.Path, default=ROOT / "h19_deconv_probes")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "llvnr_port/deconv_weights")
    args = parser.parse_args()
    data, sections, symbols = parse_hwx(args.model)
    groups = kernel_groups(data, sections, symbols)
    _, native = kernel_section(data, sections)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, recovered = [], {}
    for cout, top, size, prefixes in CASES:
        folder = args.probes / f"32x{cout}_top{top}"
        if args.build_probes:
            for seed in range(3):
                compile_case(cout, top, seed, folder / f"coded{seed}")
        proof, template = learn(cout, top, size, folder)
        offsets = proof["weight_byte_offsets"]
        for prefix in prefixes:
            group = next(group for group in groups if group.stem.startswith(prefix))
            start = min(group.starts)
            region = native[start:start + size]
            if skeleton(region, offsets) != template:
                raise ValueError(f"{prefix}: native masks, headers or padding disagree with the controlled format")
            bits = np.array([int.from_bytes(region[offset:offset + 2], "little") for offset in offsets], dtype=np.uint16)
            weights = bits.view(np.float16).reshape(32, cout, 4, 4)
            if not np.isfinite(weights).all():
                raise ValueError(f"{prefix}: non-finite recovered coefficients")
            rebuilt = bytearray(template)
            for offset, value in zip(offsets, bits):
                rebuilt[offset:offset + 2] = int(value).to_bytes(2, "little")
            if bytes(rebuilt) != region:
                raise ValueError(f"{prefix}: native byte round-trip failed")
            output = args.out / group.stem
            output.mkdir(exist_ok=True)
            np.save(output / "weights.npy", weights)
            recovered[prefix] = bits
            row = {"group": group.stem, "shape": list(weights.shape), "stride": 2,
                   "pad_for_probe": proof["pad"], "region": [start, start + size],
                   "coefficient_count": weights.size, "native_skeleton_equal": True,
                   "heldout_probe_equal": True, "native_round_trip_equal": True,
                   "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
                   "operator_status": "standard conv_transpose weights recovered, native tile origin pending"}
            rows.append(row)
            print(prefix, weights.shape, "format_and_round_trip", True, flush=True)
    equivalent = all(np.array_equal(recovered["KD5A651"], recovered[prefix]) for prefix in ("K8197D5", "K32C40F"))
    report = {"model_sha256": hashlib.sha256(data).hexdigest(), "groups": rows,
              "three_24_channel_phase_variants_have_identical_source_weights": equivalent}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("24-channel source weights equal across three native phase variants", equivalent)
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
