#!/usr/bin/env python3
"""Folded expansion channels, established using controlled H19 compiler graphs."""
import hashlib
import json
import pathlib
import sys

import numpy as np

PORT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(PORT.parent))
from h19_sparse_stream import decode,encode


def fold(weights, bias, origin, kernel_wh):
    cin,cout,kh,kw = weights.shape
    native_kw,native_kh = kernel_wh
    physical = np.zeros((cout*4,cin*native_kh*native_kw,1,1),np.float16)
    for y in range(kh):
        for x in range(kw):
            py,px = (y-origin[0])%2,(x-origin[1])%2
            ty,tx = -((y-origin[0])//2),-((x-origin[1])//2)
            if not 0<=ty<native_kh or not 0<=tx<native_kw:
                raise ValueError("transpose coefficient outside controlled folded kernel")
            for co in range(cout):
                physical[co*4+py*2+px,np.arange(cin)*native_kh*native_kw+ty*native_kw+tx,0,0] = weights[:,co,y,x]
    physical_bias = np.repeat(bias,4) if bias is not None else None
    return physical,physical_bias


def unfold(physical, physical_bias, logical_shape, origin, kernel_wh):
    cin,cout,kh,kw = logical_shape
    native_kw,native_kh = kernel_wh
    weights = np.zeros(logical_shape,np.float16)
    for y in range(kh):
        for x in range(kw):
            py,px = (y-origin[0])%2,(x-origin[1])%2
            ty,tx = -((y-origin[0])//2),-((x-origin[1])//2)
            for co in range(cout):
                weights[:,co,y,x] = physical[co*4+py*2+px,np.arange(cin)*native_kh*native_kw+ty*native_kw+tx,0,0]
    bias = None
    if physical_bias is not None:
        phases = physical_bias.reshape(cout,4)
        if not np.all(phases == phases[:,:1]):
            raise ValueError("phase-dependent transpose bias")
        bias = phases[:,0].copy()
    rebuilt,rebuilt_bias = fold(weights,bias,origin,kernel_wh)
    if not np.array_equal(rebuilt,physical) or not (physical_bias is None and rebuilt_bias is None or np.array_equal(physical_bias,rebuilt_bias)):
        raise ValueError("unexplained folded coefficients")
    return weights,bias


def recover(group, task, raw, logical_shape, origin):
    cin,cout,kh,kw = logical_shape
    r,g = task["written_registers"],task["geometry"]
    ne = min(cout,1<<((r.get("000f",0)>>19)&7))
    counts = [cout//ne+(i<cout%ne) for i in range(ne)]
    cfg = r.get("1242",0)
    lut_bytes = 128 if cfg>>16&3==2 else 0
    has_bias = bool(cfg&0x10)
    if len(group.starts)<ne or not r.get("1240",0)&0x100:
        raise ValueError("folded sparse lane count or compression mismatch")
    weights,biases,tables,segments = [],[],[],[]
    for start,count in zip(group.starts[:ne],counts):
        offset = start
        if lut_bytes:
            tables.append(raw[offset:offset+lut_bytes])
            offset += lut_bytes
        native_shape = (cin*g["kh"]*g["kw"],1,1)
        physical,pbias,size = decode(raw[offset:],count*4,native_shape,has_bias)
        if encode(physical,pbias)!=raw[offset:offset+size]:
            raise ValueError("folded stream native re-encoding mismatch")
        w,b = unfold(physical,pbias,(cin,count,kh,kw),origin,(g["kw"],g["kh"]))
        weights.append(w)
        if b is not None:biases.append(b)
        segments.append([start,offset+size])
    if len(set(tables))>1:
        raise ValueError("different activation tables across transpose lanes")
    return {"weights":np.concatenate(weights,axis=1),"bias":np.concatenate(biases) if biases else None,
            "tables":tables[:1],"segments":segments,"region":raw[min(group.starts):max(end for _,end in segments)],
            "primitive":"conv_transpose_then_lut" if lut_bytes else "conv_transpose",
            "native_origin_top_left":list(origin),"pad_for_probe":[0]*4}


def registry():
    path = PORT / "format_probes/sparse_stream_validation.json"
    if not path.exists():return []
    report = json.loads(path.read_text())
    if not report["passed"]:raise ValueError("folded sparse proof failed")
    records = []
    for row in report["cases"]:
        source = PORT / row["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row["model_sha256"]:
            raise ValueError("folded sparse compiler probe identity mismatch")
        records.append(row)
    return records


def recover_validated(group,task,raw,records):
    if not task["geometry"] or not task["written_registers"].get("1240",0)&0x100:
        return None
    g = task["geometry"]
    cin,cout = (task["explicit_dimensions_whc"][k][2] or 16 for k in ("input","output"))
    if (g["output_x_factor"],g["output_y_factor"])!=(2,2):return None
    for row in records:
        if row["shape"][:2]!=[cin,cout] or row["geometry"]!=g:
            continue
        try:
            result = recover(group,task,raw,row["shape"],row["origin"])
            result["source"] = "llvnr_port/format_probes/sparse_stream_validation.json"
            return result
        except (ValueError,IndexError):continue
    return None
