#!/usr/bin/env python3
"""Validate target PE tie handling independently of LLVNR outputs or fixtures."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from portable_ops import round_fp16

PORT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PORT.parent/"ANEForge"))
import aneforge as af
from aneforge._compile import compile_multi


def main():
    folder=PORT/"graph/precision_probes/target_rounding"
    folder.mkdir(parents=True,exist_ok=True)
    shape=(1,1,32,1024)
    x,y=af.input(shape),af.input(shape)
    tensors={"one_minus_x":(x*-1).adds(1),"x_minus_one":x.adds(-1),"add":x+y,"mul":x*y}
    model=compile_multi(list(tensors.values()),build_dir=folder)
    rows=[]
    try:
        sweep=np.arange(65536,dtype=np.uint16).view(np.float16)
        sweep=sweep[np.isfinite(sweep)&(sweep>=0)&(sweep<=1)]
        for seed in (20261112,20261115,20261117):
            rng=np.random.default_rng(seed)
            a=(np.resize(sweep,np.prod(shape)).reshape(shape) if seed==20261112 else
               (rng.uniform(.03125,2,shape)*rng.choice([-1,1],shape)).astype(np.float16))
            b=(rng.uniform(.03125,2,shape)*rng.choice([-1,1],shape)).astype(np.float16)
            outputs=model(a,b)
            exact={"one_minus_x":1-a.astype(np.float32),"x_minus_one":a.astype(np.float32)-1,
                   "add":a.astype(np.float32)+b.astype(np.float32),"mul":a.astype(np.float32)*b.astype(np.float32)}
            for name,(_,port) in zip(tensors,model.output_ports):
                actual=np.asarray(outputs[port],dtype=np.float16)
                expected=round_fp16(exact[name],"ane_rta")
                normal=(np.abs(exact[name])>=np.finfo(np.float16).tiny)|(exact[name]==0)
                difference=np.abs(actual.astype(np.float32)-expected.astype(np.float32))
                row={"seed":seed,"operation":name,"input_sha256":hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
                     "normal_samples":int(normal.sum()),"rne_exact_fraction":float((actual==round_fp16(exact[name]))[normal].mean()),
                     "rta_exact_fraction":float((actual==expected)[normal].mean()),"max_abs":float(difference[normal].max()),
                     "passed":bool((actual==expected)[normal].all())}
                rows.append(row)
        report={"passed":all(r["passed"] for r in rows),"profile":"nearest with midpoint ties away from zero",
                "scope":"controlled target PE normal FP16 arithmetic; original H19 rounding remains unmeasured",
                "checks":rows}
        (folder/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:model.release()
    return 0 if report["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
