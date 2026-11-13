#!/usr/bin/env python3
"""Decode H19 6-lane sparse 2x2 convolution kernels.

Format established by controlled H19 compiler probes:

* each physical lane starts with one fp16 bias;
* bytes 2..3 encode ``6 + 2 * nonzero_count``;
* byte 4 is zero in the observed layout;
* the sparse mask begins at byte 5 and is interleaved by input-channel pairs;
* each pair contributes two mask bytes and then immediately its fp16 nonzeros;
* one mask byte describes the 2x2 taps of one input channel, with logical tap
  ``k`` stored at bit ``2*k`` (so four taps contribute 0x55);
* within each pair, fp16 nonzero payload follows row-major selected-tap order.

This is intentionally narrow and fail-closed.  It supports the 6-output,
6-input, 2x2 stride-2 kernel used by LLVNR task 0.  Broader lane packing will
be added only after separate probes establish it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from h19_weightlib import KernelGroup, kernel_groups, parse_hwx

def decode_lane(raw: bytes, shape: tuple[int, int, int, int]) -> dict:
    cout, cin, kh, kw = shape
    if (cout, cin, kh, kw) != (6, 6, 2, 2):
        raise ValueError(f"this decoder currently supports only (6,6,2,2), got {shape}")
    if len(raw) < 7:
        raise ValueError("lane is shorter than its sparse header")
    bias = float(np.frombuffer(raw[0:2], dtype="<f2")[0])
    encoded_count = int.from_bytes(raw[2:4], "little")
    if encoded_count < 6 or (encoded_count - 6) % 2:
        raise ValueError(f"invalid sparse count field 0x{encoded_count:04x}")
    nonzero_count = (encoded_count - 6) // 2
    dense = np.zeros(shape, dtype=np.float16)
    mask_bytes = np.zeros(6, dtype=np.uint8)
    payload_segments: list[list[int]] = []
    cursor = 5
    decoded_count = 0
    selected: list[list[int]] = []
    for pair in range(3):
        if decoded_count == nonzero_count:
            break
        if cursor + 2 > len(raw):
            raise ValueError("truncated input-channel mask pair")
        pair_mask = raw[cursor:cursor + 2]
        cursor += 2
        if any(byte & 0xAA for byte in pair_mask):
            raise ValueError("mask has bits outside even logical tap slots")
        pair_count = sum(byte.bit_count() for byte in pair_mask)
        if decoded_count + pair_count > nonzero_count:
            raise ValueError("mask pair exceeds the header nonzero count")
        payload_start = cursor
        payload_end = payload_start + 2 * pair_count
        if payload_end > len(raw):
            raise ValueError("truncated fp16 sparse payload")
        values = np.frombuffer(raw[payload_start:payload_end], dtype="<f2").astype(np.float16)
        cursor = payload_end
        value_index = 0
        mask_bytes[pair * 2:pair * 2 + 2] = np.frombuffer(pair_mask, dtype=np.uint8)
        for input_channel in range(pair * 2, pair * 2 + 2):
            byte = int(mask_bytes[input_channel])
            for tap in range(kh * kw):
                if byte & (1 << (2 * tap)):
                    khw = np.unravel_index(tap, (kh, kw))
                    dense[0, input_channel, khw[0], khw[1]] = values[value_index]
                    selected.append([input_channel, int(khw[0]), int(khw[1])])
                    value_index += 1
        payload_segments.append([payload_start, payload_end, pair_count])
        decoded_count += pair_count
    if decoded_count != nonzero_count:
        raise ValueError(f"decoded {decoded_count} values, expected {nonzero_count}")
    return {
        "bias": bias,
        "encoded_count": encoded_count,
        "nonzero_count": nonzero_count,
        "mask_bytes_read": cursor - 2 * nonzero_count - 5,
        "mask_bytes": mask_bytes.tolist(),
        "payload_segments": payload_segments,
        "selected": selected,
        "dense": dense,
    }


def decode_task0(
    hwx: pathlib.Path,
    group_stem: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    data, sections, symbols = parse_hwx(hwx)
    groups = kernel_groups(data, sections, symbols)
    if group_stem is not None:
        matches = [item for item in groups if item.stem == group_stem.upper()]
        if len(matches) != 1:
            raise ValueError(f"expected one group {group_stem}, got {len(matches)}")
        group: KernelGroup = matches[0]
    else:
        candidates = [item for item in groups if len(item.starts) == 6]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one six-lane kernel group, got {len(candidates)}; "
                "pass --group-stem to disambiguate"
            )
        group = candidates[0]
    if len(group.starts) != 6:
        raise ValueError(f"expected six physical lanes, got {len(group.starts)}")
    weights = np.zeros((6, 6, 2, 2), dtype=np.float16)
    biases = np.zeros(6, dtype=np.float16)
    evidence = []
    origin = group.starts[0]
    for lane_index, start in enumerate(group.starts):
        end = group.starts[lane_index + 1] if lane_index + 1 < len(group.starts) else origin + group.span
        raw = group.raw[start - origin:end - origin]
        item = decode_lane(raw, (6, 6, 2, 2))
        weights[lane_index] = item.pop("dense")[0]
        biases[lane_index] = np.float16(item["bias"])
        item["lane"] = lane_index
        evidence.append(item)
    return weights, biases, evidence


def load_expected_blobfile(path: pathlib.Path, offsets: dict[str, int], shape=(6, 6, 2, 2)):
    data = path.read_bytes()
    payloads: dict[str, bytes] = {}
    for name, offset in offsets.items():
        import struct
        magic, code, length, data_offset = struct.unpack_from("<IIQQ", data, offset)
        if magic != 0xDEADBEEF:
            raise ValueError(f"{name}: bad blob descriptor magic at {offset}")
        payloads[name] = data[data_offset:data_offset + length]
    bits = np.unpackbits(np.frombuffer(payloads["mask"], dtype=np.uint8), bitorder="little")[:int(np.prod(shape))]
    nz = np.frombuffer(payloads["nz"], dtype="<f2")
    dense = np.zeros(int(np.prod(shape)), dtype=np.float16)
    dense[bits.astype(bool)] = nz
    return dense.reshape(shape), np.frombuffer(payloads["bias"], dtype="<f2")


def run_regression(probe_root: pathlib.Path) -> int:
    cases = [
        probe_root / "onehot_batch0",
        probe_root / "onehot_batch1",
        probe_root / "onehot_batch2",
        probe_root / "onehot_batch3",
    ]
    ok = True
    for source in cases:
        manifest = json.loads((source / "manifest.json").read_text())
        got_w, got_b, evidence = decode_task0(source / "out" / "model.hwx")
        want_w, want_b = load_expected_blobfile(
            source / "weights.bin", manifest["blob_offsets"], tuple(manifest["shape"])
        )
        weight_equal = np.array_equal(got_w, want_w)
        bias_equal = np.array_equal(got_b, want_b)
        ok &= weight_equal and bias_equal
        print(
            f"{source.name}: weight_equal={weight_equal} bias_equal={bias_equal} "
            f"lanes={len(evidence)}"
        )
        for item in evidence:
            print(
                "  lane={lane} mask_bytes_read={mask_bytes_read} nz={nonzero_count} "
                "mask={mask_bytes} payload={payload_segments}".format(**item)
            )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hwx", nargs="?", type=pathlib.Path)
    parser.add_argument("--group-stem")
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()
    if args.regression:
        return run_regression(pathlib.Path(__file__).resolve().parent / "h19_sparse_probes")
    if args.hwx is None:
        parser.error("provide an HWX path or --regression")
    weights, biases, evidence = decode_task0(args.hwx, args.group_stem)
    print(json.dumps({"biases": biases.astype(float).tolist(), "lanes": evidence}, indent=2))
    print("weight_nonzero", int(np.count_nonzero(weights)), "sha256", sys.modules["hashlib"].sha256(weights.tobytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
