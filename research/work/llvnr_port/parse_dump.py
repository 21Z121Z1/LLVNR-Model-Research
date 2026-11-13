#!/usr/bin/env python3
"""Parse the textual hwx_dump output for the H19 LLVNR model into JSON.

The dump is produced by work/coreml_to_ane_hwx/hwx_dump/hwx_parsing.py
(public HWX format research) and contains one detailed block per ANE task.
This parser is intentionally read-only and fail-soft: unknown lines are kept
in raw_lines so that a later pass can extend the schema without losing
evidence.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

TASK_RE = re.compile(r"^\s*\[ANE Task (\d+) @ (0x[0-9a-fA-F]+)\] \(Size: (0x[0-9a-fA-F]+) bytes\)", re.M)
FIELD_RE = re.compile(r"^\s{8,}([A-Za-z][A-Za-z0-9 _/\[\]\(\)\.\-]*?)\s*:\s?(.*)$")
REG_RE = re.compile(r"^\s{10,}(0x[0-9a-fA-F]+):\s*(0x[0-9a-fA-F]+)\s*$")
BLOCK_RE = re.compile(r"^\s{8,}\[(0x[0-9a-fA-F]+)\]\s+(.+?):\s*$")

DIM_RE = re.compile(r"W=(\d+)\s+H=(\d+)\s+C=(\d+)\s+D=(\d+)\s+Type=(\w+)(?:\s+\(Src2Type=(\w+)\))?")
CONV_RE = re.compile(r"K=(\d+)x(\d+)\s+S=(\d+)x(\d+)\s+P\(left/top\)=(0x[0-9a-fA-F]+)\s+O=(\d+)x(\d+)")
TASKTYPE_RE = re.compile(r"TaskType=(\d+)\s+\((.*?)\)(.*)")
PE_RE = re.compile(r"Pool=(\d+)\s+\((.*?)\)\s+Op=(\d+)\s+\((.*?)\)\s+LutEn=(\d+)\s+Cond=(\d+)\s+\((.*?)\)")
PE_DECOMP_RE = re.compile(r"Pool=(\d+)\s+\((.*?)\)\s+Op=(\d+)\s+\((.*?)\)\s+LutEn=(\d+)\s+Cond=(\d+)\s+\((.*?)\)\s+RedIdx=(\d+)\s+RedKeep=(\d+)\s+NLMode=(\d+)\s+Src1=(\d+)\s+Src2=(\d+)")
STRIDES_RE = re.compile(r"Row=(0x[0-9a-fA-F]+)\s+Chan=(0x[0-9a-fA-F]+)\s+Depth=(0x[0-9a-fA-F]+)\s+Group=(0x[0-9a-fA-F]+)")
DST_STRIDES_RE = re.compile(r"Row=(0x[0-9a-fA-F]+)\s+Plane=(0x[0-9a-fA-F]+)\s+Depth=(0x[0-9a-fA-F]+)\s+Group=(0x[0-9a-fA-F]+)")


def parse_floats(text: str):
    out = []
    for m in re.finditer(r"0x([0-9a-fA-F]{8})\s+\(([-+0-9.eE]+)\)", text):
        out.append({"hex": "0x" + m.group(1), "value": float(m.group(2))})
    return out


def parse_task(block: str, index: int, addr: str, size: str):
    t = {
        "index": index,
        "addr": addr,
        "size_bytes": int(size, 16),
        "fields": {},
        "blocks": {},
        "registers": {},
        "raw_lines": [],
    }
    current_block = "header"
    for line in block.splitlines():
        m = TASK_RE.match(line)
        if m:
            continue
        bm = BLOCK_RE.match(line)
        if bm:
            current_block = "0x" + bm.group(1)[2:].lower()
            t["blocks"].setdefault(current_block, {"name": bm.group(2), "registers": {}})
            continue
        rm = REG_RE.match(line)
        if rm and current_block != "header":
            t["blocks"][current_block]["registers"][rm.group(1).lower()] = rm.group(2).lower()
            t["registers"][rm.group(1).lower()] = rm.group(2).lower()
            continue
        fm = FIELD_RE.match(line)
        if fm:
            key = re.sub(r"\s+", " ", fm.group(1).strip())
            val = fm.group(2).strip()
            t["fields"].setdefault(key, val)
            continue
        if line.strip():
            t["raw_lines"].append(line.rstrip())

    f = t["fields"]

    def dim(name):
        m = DIM_RE.search(f.get(name, ""))
        if not m:
            return None
        return {
            "W": int(m.group(1)), "H": int(m.group(2)), "C": int(m.group(3)), "D": int(m.group(4)),
            "type": m.group(5), "src2_type": m.group(6),
        }

    t["in_dim"] = dim("InDim")
    t["out_dim"] = dim("OutDim")

    m = CONV_RE.search(f.get("ConvCfg", ""))
    if m:
        t["conv"] = {
            "kh": int(m.group(1)), "kw": int(m.group(2)),
            "sh": int(m.group(3)), "sw": int(m.group(4)),
            "pad_left_top": m.group(5), "oh": int(m.group(6)), "ow": int(m.group(7)),
        }
    else:
        t["conv"] = None

    m = TASKTYPE_RE.search(f.get("MacCfg", ""))
    if m:
        t["task_type"] = {
            "id": int(m.group(1)),
            "name": m.group(2),
            "detail": m.group(3).strip(),
        }
        rest = m.group(3)
        for k in ("ActiveNE", "SmSrc", "ReluType", "OutTrans", "FillLowerNE"):
            mm = re.search(k + r"=(\d+)", rest)
            if mm:
                t["task_type"][k] = int(mm.group(1))

    m = PE_DECOMP_RE.search(f.get("PE Config", ""))
    if m:
        t["pe"] = {
            "pool_id": int(m.group(1)), "pool": m.group(2),
            "op_id": int(m.group(3)), "op": m.group(4),
            "lut": int(m.group(5)), "cond_id": int(m.group(6)), "cond": m.group(7),
            "red_idx": int(m.group(8)), "red_keep": int(m.group(9)),
            "nl_mode": int(m.group(10)), "src1": int(m.group(11)), "src2": int(m.group(12)),
        }
    else:
        m = PE_RE.search(f.get("PE Config", ""))
        if m:
            t["pe"] = {
                "pool_id": int(m.group(1)), "pool": m.group(2),
                "op_id": int(m.group(3)), "op": m.group(4),
                "lut": int(m.group(5)), "cond_id": int(m.group(6)), "cond": m.group(7),
            }
        else:
            t["pe"] = None

    t["pe_bias"] = parse_floats(f.get("PE Bias", ""))
    t["pe_prescale"] = parse_floats(f.get("PE PreScale", ""))
    t["pe_postscale"] = parse_floats(f.get("PE PostScale", ""))

    m = STRIDES_RE.search(f.get("Src1Strides", ""))
    if m:
        t["src1_strides"] = {"row": m.group(1), "chan": m.group(2), "depth": m.group(3), "group": m.group(4)}
    m = DST_STRIDES_RE.search(f.get("DstStrides", ""))
    if m:
        t["dst_strides"] = {"row": m.group(1), "plane": m.group(2), "depth": m.group(3), "group": m.group(4)}
    t["src1_base"] = f.get("Src1Base")
    t["master_cfg"] = f.get("MasterCfg")
    m = re.search(r"Sparse=(\d)", f.get("MasterCfg", ""))
    if m:
        t["kernel_sparse"] = int(m.group(1))
    m = re.search(r"SparseEn=(\d)", f.get("KernelCfg", ""))
    if m:
        t["ne_sparse"] = int(m.group(1))
    t["kernel_cfg"] = f.get("KernelCfg")
    t["matrix_bias"] = f.get("MatrixBias")
    t["nic"] = f.get("NID")
    t["patch"] = f.get("PatchCfg")
    t["l2_result_base"] = f.get("L2_ResultBase")
    t["l2_src1_base"] = f.get("L2_Src1Base")
    t["l2_src1_strides"] = f.get("L2_Src1Strides")
    t["l2_result_cfg"] = f.get("L2_ResultCfg")
    t["src1_cfg"] = f.get("Src1Cfg")
    t["src2_cfg"] = f.get("Src2Cfg")
    t["coeff_bases"] = {k: v for k, v in f.items() if k.startswith("CoeffBase")}
    for kind in ("Src1", "Src2", "Dst"):
        key = kind + "DMAConfig"
        if key in f:
            t[kind.lower() + "_dma"] = f[key]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    args = ap.parse_args()
    text = pathlib.Path(args.dump).read_text(errors="replace")
    matches = list(TASK_RE.finditer(text))
    tasks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        tasks.append(parse_task(text[start:end], int(m.group(1)), m.group(2), m.group(3)))
    out = {
        "source": str(pathlib.Path(args.dump).resolve()),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"parsed {len(tasks)} tasks -> {args.out}")
    from collections import Counter
    print("task types:", Counter((t.get("task_type") or {}).get("id") for t in tasks))
    print("with pe:", sum(1 for t in tasks if t["pe"]))
    print("with conv:", sum(1 for t in tasks if t["conv"]))
    print("with kernel_sparse:", sum(1 for t in tasks if t.get("kernel_sparse") == 1))


if __name__ == "__main__":
    main()
