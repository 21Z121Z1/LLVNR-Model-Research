#!/usr/bin/env python3
"""SuperWide 0.5x's native chroma statistics, map selection and channel gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from native_reference import synthetic_inputs
from portable_graph import evaluate_cpu,load_coefficients,compare
from portable_ops import fp32
from recovered_reference import RecoveredReference
from wide1_graph import source_plan as backbone
from llvnr_taskflow import address

PORT = Path(__file__).resolve().parents[1]
MODELS = tuple("LLV-HD-SuperWide-0.5x-v0.2.2_"+s for s in ("2304x1296","2464x1388"))


def source_plan(reference):
    large = reference.canvas_wh == (2464,1388)
    task_map = ({435:542,436:543,437:544,438:545,439:546,443:547,444:548,445:554,446:555,447:562,
                 449:573,450:577,451:578,533:783,534:785,535:786,536:787,537:788} if large else
                {435:446,436:447,437:448,438:449,439:450,443:454,444:455,445:456,446:457,447:462,
                 449:474,450:476,451:477,533:609,534:610,535:611,536:612,537:613})
    plan = backbone(reference,task_map)
    start,gate,goc_index = (529,563,572) if large else (433,463,472)
    extra = []

    def add(name,op,inputs,**attrs):
        extra.append({"name":name,"op":op,"inputs":list(inputs),**attrs})
        return name

    for offset,channel in ((0,5),(2,4)):
        task = reference.ir["tasks"][start+offset]
        add(f"chroma_{channel}","channel",["state_yuv"],index=channel)
        add(f"chroma_mean_{channel}","spatial_mean",[f"chroma_{channel}"],
            scale=fp32(task["written_registers"]["1145"]),native_task=start+offset)
        add(f"chroma_abs_{channel}","abs",[f"chroma_mean_{channel}"])
    add("chroma_abs_max","maximum",["chroma_abs_4","chroma_abs_5"])
    r = reference.ir["tasks"][start+4]["written_registers"]
    add("chroma_threshold","affine",["chroma_abs_max"],alpha=fp32(r["114a"]),beta=fp32(r["114b"]))
    add("chroma_map_condition","affine_compare",["chroma_threshold"],
        registers=reference.ir["tasks"][start+5]["written_registers"])
    add("chroma_map_inverse","affine",["chroma_map_condition"],alpha=-1.0,beta=1.0)
    task=reference.ir["tasks"][start+7]
    group=reference.groups_by_offset[address(task,0x158e)["offset"]].stem
    low,high=reference.luts[group]["bounds"]
    add("dnr_chroma_clipped","clip",["dnr_zoom_map"],minimum=low,maximum=high)
    add("dnr_chroma_true","mul",["chroma_map_condition","dnr_chroma_clipped"])
    add("dnr_chroma_false","mul",["chroma_map_inverse","dnr_zoom_map"])

    nodes=[]
    for n in plan["nodes"]:
        if n["name"]=="dnr_scaled":
            nodes.extend(extra)
            n={**n,"op":"add_affine","inputs":["dnr_chroma_true","dnr_chroma_false"]}
        nodes.append(n)
        if n["name"]=="decoder_1_up":
            for offset,ch in ((0,5),(4,4)):
                nodes.append({"name":f"chroma_gate_{ch}","op":"affine_compare","inputs":[f"chroma_mean_{ch}"],
                              "registers":reference.ir["tasks"][gate+offset]["written_registers"]})
            nodes.append({"name":"chroma_gate_product","op":"mul_scaled","inputs":["chroma_gate_4","chroma_gate_5"],
                          "scale":fp32(reference.ir["tasks"][goc_index-1]["written_registers"]["1144"])})
            t=reference.ir["tasks"][goc_index]
            a=address(t,0x134c)
            raw=reference.static_bars[a["bar"]][a["offset"]:a["offset"]+128]
            const=np.frombuffer(raw,dtype='<f2').astype(np.float32)
            if len(const)!=64:raise ValueError("incomplete native GOC constants")
            # Controlled goc_onehot has the exact same affine table and PE
            # post-op as this task. Constants remain model-owned.
            r=t["written_registers"]
            beta=fp32(r.get("114b",0))
            nodes.append({"name":"chroma_channel_gain","op":"broadcast_affine","inputs":["chroma_gate_product"],
                          "channel_scale":const[32:].tolist(),"channel_bias":(const[:32]+beta).tolist(),
                          "native_task":goc_index,"native_constants_sha256":hashlib.sha256(raw).hexdigest()})
            nodes.append({"name":"decoder_1_chroma_gated","op":"mul","inputs":["decoder_1_up","chroma_channel_gain"]})
        if n["name"]=="decoder_1_input":
            n["inputs"]=["encoder_1","decoder_1_chroma_gated"]
    plan["nodes"]=nodes
    plan["validation_status"]="requires chroma branch and full original schedule acceptance"
    return plan


def chroma_cases(canvas_wh):
    for name,c4,c5 in (("chroma_gain_active",0.02,-0.02),("chroma_gain_first_false",0.02,0.0),
                       ("chroma_gain_second_false",0.0,-0.02),("chroma_map_active",0.01,0.01)):
        inputs=synthetic_inputs(20261115,canvas_wh)
        for key in ("cur_frame","his_frame"):
            inputs[key][:,4]=c4
            inputs[key][:,5]=c5
        yield name,inputs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True,help="local model.hwx")
    parser.add_argument("--recovery",type=Path,required=True,help="private recovery directory")
    parser.add_argument("--output",type=Path,help="private graph output directory")
    args=parser.parse_args()
    reference=RecoveredReference(args.model,args.recovery)
    plan=source_plan(reference)
    folder=args.output or args.recovery
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"source_graph.json").write_text(json.dumps(plan,indent=2)+"\n")
    results=[]
    for name,inputs in [("correlated_noise",synthetic_inputs(20261112,reference.canvas_wh)),*chroma_cases(reference.canvas_wh)]:
        values=evaluate_cpu(plan,load_coefficients(plan,folder),inputs)
        original=reference.run(inputs)
        metrics={n:compare(values[n],original[n]) for n in plan["outputs"]}
        results.append({"case":name,"comparisons":metrics,"passed":all(r["exact_fraction"]==1 for r in metrics.values()),
                        "gain":values["chroma_channel_gain"].reshape(-1).astype(float).tolist()})
        print(name,results[-1]["passed"],metrics,flush=True)
    report={"model_id":reference.model_id,"cases":results,"passed":all(r["passed"] for r in results),
            "scope":"full native schedule CPU versus portable CPU; original H19 outputs unavailable"}
    (folder/"schedule_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0 if report["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
