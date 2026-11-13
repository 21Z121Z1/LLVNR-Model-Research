#!/usr/bin/env python3
"""Build and inspect controlled H19 sparse-weight probes.

This tool is deliberately serial: it invokes one ANECCompile process at a
time.  It generates BLOBFILE payloads from explicit masks/nonzero values and
then extracts the resulting HWX kernel lanes.  No shipped model bytes are
copied into probe artifacts.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from h19_weightlib import kernel_groups, parse_hwx

ROOT = pathlib.Path(__file__).resolve().parent
ANECC = ROOT / "anecc"
PROBE_ROOT = ROOT / "h19_sparse_probes"
SHAPE = (6, 6, 2, 2)
FLAGS = (
    "ScanWeightsForCompression=true",
    "KernelRewind=true",
    "SpatialSplitMode=GenericDAG",
    "Optimize=ReductionPerf",
    "EnableCircularBufferInSpatialSplit=true",
    "DisableBondedNetworks=true",
    "EnableNonbondedNetworks=true",
    "UseExtendedMachoFormat=false",
    "EnableMulticastBruteforceSearch=true",
)


def _descriptor(payload: bytes, payload_type: int = 1) -> bytes:
    return struct.pack("<IIQQ", 0xDEADBEEF, payload_type, len(payload), 0) + b"\0" * 40


def build_blobfile(mask: np.ndarray, nonzero: np.ndarray, bias: np.ndarray) -> tuple[bytes, dict[str, int]]:
    """Return a BLOBFILE and its stable descriptor offsets."""
    if mask.ndim != 4 or mask.dtype != np.uint8:
        raise ValueError("mask must be a four-dimensional uint8 tensor")
    packed_mask = np.packbits(mask.reshape(-1).astype(bool), bitorder="little").tobytes()
    nz = np.ascontiguousarray(nonzero, dtype="<f2").tobytes()
    bb = np.ascontiguousarray(bias, dtype="<f2").tobytes()
    items = [(packed_mask, 1), (nz, 1), (bb, 1)]
    offsets: dict[str, int] = {}
    parts = [struct.pack("<II", len(items), 2) + b"\0" * 56]
    cursor = 64
    for name, (payload, code) in zip(("mask", "nz", "bias"), items):
        offsets[name] = cursor
        data_offset = cursor + 64
        parts.append(struct.pack("<IIQQ", 0xDEADBEEF, code, len(payload), data_offset) + b"\0" * 40)
        parts.append(payload)
        cursor = data_offset + len(payload)
    return b"".join(parts), offsets


def mil_source(offsets: dict[str, int], nz_count: int, shape=SHAPE, stride=None) -> str:
    cout, cin, kh, kw = shape
    if kh != kw or kh not in (1, 2, 3):
        raise ValueError("controlled sparse probes support square 1x1, 2x2 or 3x3 kernels")
    stride = stride if stride is not None else (2 if kh == 2 else 1)
    pad = 0 if kh == 2 else kh // 2
    height = (648 + 2 * pad - kh) // stride + 1
    width = (1152 + 2 * pad - kw) // stride + 1
    encoded_shape = ",".join(map(str, shape))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3600.16.1"}}, {{"coremlc-version", "3600.22.1"}}}})]
{{
 func main<ios18>(tensor<fp16, [1, {cin}, 648, 1152]> x) {{
  tensor<uint1,[{encoded_shape}]> mask=const()[name=string("mask"),val=tensor<uint1,[{encoded_shape}]>(BLOBFILE(path=string("@model_path/weights.bin"),offset=uint64({offsets['mask']})))];
  tensor<fp16,[{nz_count}]> nz=const()[name=string("nz"),val=tensor<fp16,[{nz_count}]>(BLOBFILE(path=string("@model_path/weights.bin"),offset=uint64({offsets['nz']})))];
  tensor<fp16,[{encoded_shape}]> w=constexpr_sparse_to_dense(mask=mask,nonzero_data=nz)[name=string("w")];
  string pt=const()[name=string("pt"),val=string("custom")]; tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([{stride},{stride}])]; tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([{pad},{pad},{pad},{pad}])]; tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])]; int32 gp=const()[name=string("gp"),val=int32(1)];
  tensor<fp16,[{cout}]> bb=const()[name=string("bb"),val=tensor<fp16,[{cout}]>(BLOBFILE(path=string("@model_path/weights.bin"),offset=uint64({offsets['bias']})))];
  tensor<fp16,[1,{cout},{height},{width}]> y=conv(bias=bb,dilations=dl,groups=gp,pad=pd,pad_type=pt,strides=st,weight=w,x=x)[name=string("y")];
 }} -> (y);
}}
'''


def compile_probe(name: str, mask: np.ndarray, values: np.ndarray, bias: np.ndarray, stride=None) -> pathlib.Path:
    source = PROBE_ROOT / name
    outdir = source / "out"
    source.mkdir(parents=True, exist_ok=True)
    blob, offsets = build_blobfile(mask, values, bias)
    (source / "weights.bin").write_bytes(blob)
    (source / "model.mil").write_text(mil_source(offsets, int(values.size), mask.shape, stride))
    manifest = {
        "name": name,
        "shape": mask.shape,
        "mask_flat": mask.reshape(-1).astype(int).tolist(),
        "nonzero": np.asarray(values, dtype=np.float16).astype(float).tolist(),
        "bias": np.asarray(bias, dtype=np.float16).astype(float).tolist(),
        "blob_offsets": offsets,
        "stride": stride if stride is not None else (2 if mask.shape[2] == 2 else 1),
    }
    (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cmd = [str(ANECC), str(source), str(outdir), "h19", *FLAGS]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return source


def extract_lanes(hwx: pathlib.Path) -> list[dict]:
    data, sections, symbols = parse_hwx(hwx)
    groups = kernel_groups(data, sections, symbols)
    if not groups:
        raise ValueError("expected a kernel group")
    group = groups[0]
    origin = group.starts[0]
    lanes = []
    for i, start in enumerate(group.starts):
        end = group.starts[i + 1] if i + 1 < len(group.starts) else origin + group.span
        raw = group.raw[start - origin:end - origin]
        lanes.append({
            "index": i,
            "offset": start,
            "size": len(raw),
            "head": raw[:32].hex(" "),
            "bias_half": float(np.frombuffer(raw[:2], dtype="<f2")[0]) if len(raw) >= 2 else None,
            "u16_02": int.from_bytes(raw[2:4], "little") if len(raw) >= 4 else None,
            "u8_04": raw[4] if len(raw) >= 5 else None,
            "mask_word": int.from_bytes(raw[5:7], "little") if len(raw) >= 7 else None,
        })
    return lanes


def onehot_batch(batch: int) -> None:
    """Six outputs cover six consecutive row-major mask positions."""
    mask = np.zeros(SHAPE, dtype=np.uint8)
    selected: list[int] = []
    for out in range(6):
        linear = batch * 6 + out
        linear %= int(np.prod(SHAPE[1:]))
        mask[out].reshape(-1)[linear] = 1
        selected.append(linear)
    nz_count = int(mask.sum())
    values = np.array([(i + 1) * 0.03125 for i in range(nz_count)], dtype=np.float16)
    bias = np.array([0.125 * i - 0.25 for i in range(6)], dtype=np.float16)
    name = f"onehot_batch{batch}"
    source = compile_probe(name, mask, values, bias)
    lanes = extract_lanes(source / "out" / "model.hwx")
    print(json.dumps({"selected": selected, "lanes": lanes}, indent=2))


def random_probe(seed: int, nonzero_per_output: int, shape=SHAPE, stride=None) -> None:
    """Compile a deterministic random sparse mask and report decoded lanes."""
    rng = random.Random(seed)
    mask = np.zeros(shape, dtype=np.uint8)
    if not 1 <= nonzero_per_output <= int(np.prod(shape[1:])):
        raise ValueError("nonzero count is outside the per-output coefficient range")
    for out in range(shape[0]):
        selected = rng.sample(range(int(np.prod(shape[1:]))), nonzero_per_output)
        mask[out].reshape(-1)[selected] = 1
    nz_count = int(mask.sum())
    # Unique, exactly representable fp16 values exercise payload order without
    # introducing negative zero or rounding ambiguity.
    values = np.array(
        [((i % 31) + 1) * (0.00390625 if i % 2 else -0.0078125) for i in range(nz_count)],
        dtype=np.float16,
    )
    if shape[2:] == (3, 3):
        if nz_count > 0x4000:
            raise ValueError("coordinate-coded probe exceeds the finite unique-value range")
        values = np.arange(0x2001, 0x2001 + nz_count, dtype=np.uint16).view(np.float16)
    bias = np.array([(i - 2.5) * 0.0625 for i in range(shape[0])], dtype=np.float16)
    name = f"random_seed{seed}_nz{nonzero_per_output}"
    if shape != SHAPE:
        name += "_" + "x".join(map(str, shape))
    if stride is not None:
        name += f"_s{stride}"
    source = compile_probe(name, mask, values, bias, stride)
    lanes = extract_lanes(source / "out" / "model.hwx")
    print(json.dumps({
        "name": name,
        "nonzero_per_output": nonzero_per_output,
        "mask_flat": mask.reshape(-1).astype(int).tolist(),
        "values": values.astype(float).tolist(),
        "bias": bias.astype(float).tolist(),
        "lanes": lanes,
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onehot-batch", type=int, choices=range(4))
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--nonzero-per-output", type=int, default=4)
    parser.add_argument("--shape", default="6x6x2x2", help="CoutxCinxKhxKw")
    parser.add_argument("--stride", type=int, choices=(1, 2))
    args = parser.parse_args()
    PROBE_ROOT.mkdir(exist_ok=True)
    if args.onehot_batch is not None:
        onehot_batch(args.onehot_batch)
    elif args.random_seed is not None:
        shape = tuple(map(int, args.shape.split("x")))
        if len(shape) != 4 or min(shape) < 1:
            parser.error("--shape requires four positive dimensions")
        random_probe(args.random_seed, args.nonzero_per_output, shape, args.stride)
    else:
        parser.error("provide --onehot-batch or --random-seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
