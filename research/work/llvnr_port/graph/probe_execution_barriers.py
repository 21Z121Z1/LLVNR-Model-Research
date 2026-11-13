#!/usr/bin/env python3
"""Test explicit FP16 storage boundaries without changing recovered arithmetic."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from portable_graph import PORT, H13Model, build_ane, evaluate_cpu, load_coefficients
from run_portable import sequence, numeric_check


def partition(plan, coefficients, barriers):
    graph = build_ane(plan, coefficients)
    nodes = {n["name"]: n for n in plan["nodes"]}
    available = {n["name"] for n in plan["nodes"] if n["op"] == "input"}
    stages = []
    for outputs in [*barriers, plan["outputs"]]:
        outputs = [name for name in outputs if name not in available]
        if not outputs:
            continue
        required = set()
        def visit(name):
            if name in required:
                return
            required.add(name)
            if name not in available:
                for source in nodes[name]["inputs"]:
                    visit(source)
        for name in outputs:
            visit(name)
        stage_nodes = []
        for original in plan["nodes"]:
            name = original["name"]
            if name in required:
                stage_nodes.append({"name": name, "op": "input", "inputs": [], "shape": list(graph[name].shape)}
                                   if name in available else copy.deepcopy(original))
        stages.append({**plan, "nodes": stage_nodes, "outputs": outputs})
        available.update(outputs)
    return stages


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=PORT/"graph/portable/source_graph.json")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--barriers",default='[["temporal_band0"],["history_weight"],["history_part","current_part"],["state_yuv"]]')
    args=parser.parse_args()
    plan=json.loads(args.source.read_text()); coefficients=load_coefficients(plan)
    barriers=json.loads(args.barriers)
    args.out.mkdir(parents=True,exist_ok=True)
    stages=partition(plan,coefficients,barriers)
    models=[]
    report={"input_file_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "source_graph_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),"barriers":barriers,"frames":[]}
    def save(): (args.out/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        for i,stage in enumerate(stages):
            print("compile",i,len(stage["nodes"]),stage["outputs"],flush=True)
            models.append(H13Model(stage,coefficients,args.out/f"stage{i}"))
        with np.load(args.input,allow_pickle=False) as archive:
            frames=sequence(archive["cur"])[:8]
            maps={n:archive[n].copy() for n in ("dnr_zoom_map","fusion_zoom_map")}
        cpu_history=ane_history=None
        for i,current in enumerate(frames):
            if i==0: cpu_history=current.copy();ane_history=current.copy()
            per_maps={k:v[i] if v.ndim==5 else v for k,v in maps.items()}
            expected=evaluate_cpu(plan,coefficients,{"cur_frame":current,"his_frame":cpu_history,**per_maps})
            actual={"cur_frame":current,"his_frame":ane_history,**per_maps}
            start=time.perf_counter()
            for stage,model in zip(stages,models):
                actual.update(model.run({n["name"]:actual[n["name"]] for n in stage["nodes"] if n["op"]=="input"}))
            seconds=time.perf_counter()-start
            metrics={k:numeric_check(actual[k],expected[k]) for k in plan["outputs"]}
            report["frames"].append({"frame":i,"outputs":metrics,"seconds":seconds})
            save();print(i,{k:v["max_abs"] for k,v in metrics.items()},seconds,flush=True)
            cpu_history=expected["state_yuv"].copy();ane_history=actual["state_yuv"].copy()
        report["passed"]=all(m["passed"] for r in report["frames"] for m in r["outputs"].values());save()
    finally:
        for model in models: model.close()
    return 0 if report.get("passed") else 1


if __name__=="__main__":raise SystemExit(main())
