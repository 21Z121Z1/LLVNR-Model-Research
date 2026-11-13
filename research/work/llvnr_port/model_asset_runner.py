#!/usr/bin/env python3
"""Run an exported LLVNR model without its source firmware or recovery tree."""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from portable_graph import checked_inputs, evaluate_cpu, load_coefficients


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def synthetic_inputs(plan, seed):
    rng = np.random.default_rng(seed)
    shape = next(n["shape"] for n in plan["nodes"] if n["name"] == "cur_frame")
    image = rng.normal(0.2, 0.015, shape).astype(np.float32)
    image[:, 4:] -= 0.2
    current = image + rng.normal(0, 0.005, image.shape)
    result = {"his_frame": image.astype(np.float16), "cur_frame": current.astype(np.float16)}
    for node in plan["nodes"]:
        if node["op"] == "input" and node["name"] not in result:
            result[node["name"]] = np.full(node["shape"], 0.5, np.float16)
    return result


def run(folder, backend, inputs):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha((folder / name).read_bytes()) != digest:
            raise ValueError(f"asset checksum mismatch: {name}")
    plan = json.loads((folder / "source_graph.json").read_text())
    inputs = checked_inputs(plan, inputs)
    started = time.perf_counter()
    if backend == "cpu":
        values = evaluate_cpu(plan, load_coefficients(plan, folder), inputs)
        outputs = {name: values[name] for name in plan["outputs"]}
    else:
        from aneforge._runtime import E5RT
        ane_inputs=manifest.get("ane_inputs",manifest["inputs"])
        ane_outputs=manifest.get("ane_outputs",manifest["outputs"])
        in_ports = {p["port"]: tuple(p["shape"]) for p in ane_inputs}
        out_ports = {p["port"]: tuple(p["shape"]) for p in ane_outputs}
        cache = ROOT / "cache" / folder.name
        precision=None
        try:
            feeds=inputs
            if manifest.get("execution_profile")=="metal_state_ane_denoiser":
                from precision_runtime import MetalPrefix
                precision=MetalPrefix(folder/"precision",ROOT/"runtime/libllvnr_precision.dylib")
                feeds={**inputs,**precision.run(inputs)}
            with E5RT.compile(folder / "model.mil", cache_dir=cache, inputs=in_ports, outputs=out_ports) as program:
                actual = program.eval({p["port"]: feeds[p["name"]] for p in ane_inputs})
                result={p["name"]:np.array(actual[p["port"]],dtype=np.float16,copy=True) for p in ane_outputs}
                outputs={p["name"]:result.get(p["name"],feeds.get(p["name"])) for p in manifest["outputs"]}
        finally:
            if precision:precision.close()
    return outputs, {"model_id": manifest["model_id"], "backend": backend, "seconds": time.perf_counter() - started,
                     "source_model_sha256": manifest["source_model_sha256"],
                     "outputs": {n: {"shape": list(v.shape), "finite": bool(np.isfinite(v).all()), "sha256_fp16": sha(v.tobytes())}
                                 for n, v in outputs.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="model directory name under this kit, or an absolute directory")
    parser.add_argument("--backend", choices=("cpu", "ane"), default="ane")
    parser.add_argument("--input", type=pathlib.Path, help="NPZ containing the four named FP16 inputs")
    parser.add_argument("--seed", type=int, default=20261112, help="controlled synthetic input seed when --input is omitted")
    parser.add_argument("--output", type=pathlib.Path, help="save named FP16 outputs as NPZ")
    args = parser.parse_args()
    folder = pathlib.Path(args.model)
    if not folder.is_absolute():
        folder = ROOT / folder
    plan = json.loads((folder / "source_graph.json").read_text())
    inputs = dict(np.load(args.input)) if args.input else synthetic_inputs(plan, args.seed)
    outputs, report = run(folder, args.backend, inputs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **outputs)
    print(json.dumps(report, indent=2))
    return 0 if all(v["finite"] for v in report["outputs"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
