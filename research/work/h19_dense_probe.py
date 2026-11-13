#!/usr/bin/env python3
"""Recover packed dense-kernel permutations from unique FP16 bit patterns.

Each source coefficient and bias has a unique finite bit pattern. The report
records its byte location, so an unexpected transform or dropped coefficient
fails the experiment instead of silently choosing a layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ANEForge"))
import aneforge as af
from aneforge._compile import _lower_fused_to_dir
from h19_sparse_probe import FLAGS
from h19_weightlib import kernel_groups, kernel_section, parse_hwx


def probe(shape, output_root):
    cout, cin, kh, kw = shape
    count = int(np.prod(shape))
    if count > 0x4000:
        raise ValueError("probe exceeds the unique finite coefficient range")
    bits = np.arange(0x2001, 0x2001 + count, dtype=np.uint16).reshape(shape)
    bias_bits = np.arange(0x7000, 0x7000 + cout, dtype=np.uint16)
    weights, bias = bits.view(np.float16), bias_bits.view(np.float16)
    source = output_root / ("dense_" + "x".join(map(str, shape)))
    source.mkdir(parents=True, exist_ok=True)
    graph = af.conv(af.input((1, cin, 81, 144)), weights, pad=kh // 2, bias=bias)
    _lower_fused_to_dir(graph, source)
    command = [str(ROOT / "anecc"), str(source), str(source / "out"), "h19", *FLAGS]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (source / "compiler.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"compiler exited {result.returncode}, see {source / 'compiler.log'}")
    data, sections, symbols = parse_hwx(source / "out/model.hwx")
    groups = [group for group in kernel_groups(data, sections, symbols) if group.kind == "ne"]
    if not groups:
        raise ValueError("compiled probe has no kernel group")
    group = groups[0]
    section, raw = kernel_section(data, sections)
    by_value = {}
    for offset in range(0, len(raw) - 1, 2):
        value = int.from_bytes(raw[offset:offset + 2], "little")
        by_value.setdefault(value, []).append(offset)
    weight_locations = []
    bias_locations = []
    duplicates = []
    missing = []
    for kind, expected, locations in (("weight", bits, weight_locations), ("bias", bias_bits, bias_locations)):
        for flat_index, value in enumerate(expected.flat):
            # This probe establishes the aligned dense mode. The separately
            # validated sparse format uses unaligned payloads.
            offsets = by_value.get(int(value), [])
            if not offsets:
                missing.append([kind, flat_index])
            locations.append(offsets[:1])
    extent = max((offset + 2 for locations in (weight_locations, bias_locations)
                  for row in locations for offset in row), default=0)
    first_region_end = (extent + 63) & ~63
    for kind, expected in (("weight", bits), ("bias", bias_bits)):
        for flat_index, value in enumerate(expected.flat):
            offsets = [offset for offset in by_value.get(int(value), []) if offset < first_region_end]
            if len(offsets) > 1:
                duplicates.append([kind, flat_index, offsets])
    external_bias_equal = False
    if cout == 1 and missing == [["bias", 0]]:
        from h19_statewalk import parse_macho, walk_tasks, words_at
        _, _, _, _, state_sections, _ = parse_macho(source / "out/model.hwx")
        task = walk_tasks(words_at(data, state_sections, ("__TEXT", "__text")))[0]
        external_bias_equal = task["regs"].get(0x1244) == int(bias_bits[0])
    passed = (not missing or external_bias_equal) and not duplicates
    report = {
        "shape": list(shape),
        "command": command,
        "group": group.stem,
        "first_region_bytes": first_region_end,
        "all_kernel_bytes": len(raw),
        "kernel_groups": len(groups),
        "symbol_indices": list(group.indices),
        "symbol_offsets": [symbol.value - sections[("__KERN_0", "__kern_0")].address for symbol in group.symbols],
        "weight_byte_locations": weight_locations,
        "bias_byte_locations": bias_locations,
        "missing": missing,
        "duplicates": duplicates,
        "external_ne_bias_equal": external_bias_equal,
        "passed": passed,
        "source_weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "source_bias_sha256": hashlib.sha256(bias.tobytes()).hexdigest(),
        "hwx_sha256": hashlib.sha256(data).hexdigest(),
    }
    (source / "layout.json").write_text(json.dumps(report, indent=1) + "\n")
    print("shape", shape, "bytes", first_region_end, "missing", len(missing), "duplicates", len(duplicates), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shapes", nargs="+", help="CoutxCinxKhxKw")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "h19_dense_probes")
    args = parser.parse_args()
    passed = True
    for encoded in args.shapes:
        shape = tuple(map(int, encoded.split("x")))
        if len(shape) != 4:
            parser.error("each shape needs four dimensions")
        passed = probe(shape, args.out) and passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
