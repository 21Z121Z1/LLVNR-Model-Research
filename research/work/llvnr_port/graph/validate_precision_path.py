#!/usr/bin/env python3
"""Verify explicit Metal state + ANE denoiser against the unmodified RNE reference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from metal_prefix import export_prefix
from precision_runtime import MetalPrefix
from portable_graph import PORT,H13Model,evaluate_cpu,load_coefficients,compare
from run_portable import sequence,numeric_check


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=PORT/"graph/portable/source_graph.json")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.source.read_text());coefficients=load_coefficients(plan)
    prefix,body=export_prefix(plan,coefficients,args.out)
    metal=MetalPrefix(args.out)
    ane=H13Model(body,coefficients,args.out/"ane_body")
    with np.load(args.input,allow_pickle=False) as z:
        frame_shape=next(n["shape"] for n in plan["nodes"] if n["name"]=="cur_frame")
        frames=sequence(z["cur"],frame_shape)[:8];maps={n:z[n].copy() for n in ("dnr_zoom_map","fusion_zoom_map")}
    cpu_history=actual_history=None
    report={"scope":"Metal FP32 state operators with explicit RNE FP16 stores, ANE deep denoiser; unchanged original CPU source reference",
            "input_file_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),"source_graph_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
            "numeric_gate":{"max_abs":.02,"mean_abs":.001,"finite":True},"frames":[]}
    def save():(args.out/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        for i,current in enumerate(frames):
            if i==0:cpu_history=current.copy();actual_history=current.copy()
            per_maps={k:v[i] if v.ndim==5 else v for k,v in maps.items()}
            inputs={"cur_frame":current,"his_frame":actual_history,**per_maps}
            expected=evaluate_cpu(plan,coefficients,{**inputs,"his_frame":cpu_history})
            start=time.perf_counter();actual=metal.run(inputs,capture=True);metal_seconds=time.perf_counter()-start
            ane_inputs={n["name"]:actual.get(n["name"],inputs.get(n["name"])) for n in body["nodes"] if n["op"]=="input"}
            start=time.perf_counter();actual.update(ane.run(ane_inputs));ane_seconds=time.perf_counter()-start
            metrics={k:numeric_check(actual[k],expected[k]) for k in plan["outputs"]}
            prefix_checks={n["name"]:compare(actual[n["name"]],expected[n["name"]]) for n in prefix["nodes"]}
            report["frames"].append({"frame":i,"outputs":metrics,"prefix":prefix_checks,"metal_seconds":metal_seconds,"ane_seconds":ane_seconds});save()
            print(i,{k:v["max_abs"] for k,v in metrics.items()},"metal",metal_seconds,"ane",ane_seconds,flush=True)
            cpu_history=expected["state_yuv"].copy();actual_history=actual["state_yuv"].copy()
        report["passed"]=all(m["passed"] for r in report["frames"] for m in r["outputs"].values());save()
    finally:ane.close();metal.close()
    return 0 if report.get("passed") else 1


if __name__=="__main__":raise SystemExit(main())
