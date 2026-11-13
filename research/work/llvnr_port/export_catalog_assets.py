#!/usr/bin/env python3
"""Export standalone local model research assets, independent of original HWX."""
import argparse
import copy
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np

PORT = pathlib.Path(__file__).resolve().parent
WORK = PORT.parent
sys.path[:0] = [str(PORT / "graph"), str(WORK / "ANEForge")]
from portable_graph import build_ane, load_coefficients
from metal_prefix import export_prefix
from aneforge._compile import _topo_multi, _Emitter, _EMIT, _emit_program_dir
from aneforge._runtime import _find_dylib


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare_runtime(root):
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("portable_graph.py", "portable_ops.py", "precision_runtime.py"):
        shutil.copy2(PORT / "graph" / name, runtime / name)
    shutil.copy2(PORT/"common/libllvnr_precision.dylib", runtime/"libllvnr_precision.dylib")
    shutil.copy2(PORT / "model_asset_runner.py", root / "run_model.py")
    package = runtime / "aneforge"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text('"""Local LLVNR kit: ANEForge E5RT runtime subset (MIT)."""\n')
    for name in ("_runtime.py", "_circuit.py"):
        shutil.copy2(WORK / "ANEForge/aneforge" / name, package / name)
    library = package / "_lib"
    library.mkdir(exist_ok=True)
    shutil.copy2(_find_dylib(), library / "libane_e5rt_dispatch.dylib")
    shutil.copy2(WORK / "ANEForge/LICENSE", root / "ANEForge-LICENSE.txt")
    (root / "README.md").write_text('''# LLVNR Mac 模型研究资产

每个目录包含该型号自己的 FP16 系数、整图 JSON、通用 MIL 和独立验证证据。
这里没有原始 H19 HWX，也不依赖 OTA 或恢复脚本。运行器只需要 Python 3、NumPy；
ANE 后端另使用随目录保存的 ANEForge E5RT 运行时。当前硬件证据来自 M1 Pro / macOS 27.3。

```sh
python3 run_model.py MODEL_ID --backend cpu
python3 run_model.py MODEL_ID --backend ane --output result.npz
python3 run_model.py MODEL_ID --backend ane --input inputs.npz --output result.npz
```

`inputs.npz` 使用四个名称：`cur_frame`、`his_frame`、`dnr_zoom_map`、`fusion_zoom_map`。
形状见各目录的 manifest.json；历史状态由调用者把上一帧的 state_yuv 送回 his_frame。
未提供输入时使用可复现的合成输入。此运行器验证模型，不提供相机或视频前后处理。

这是本地移植研究资产，未做应用签名、公证或跨机器系统兼容性认证。
逐套验收状态、已知失败与原始 H19 硬件对照缺口见各目录 evidence 和总状态表。
''')


def export_model(model_id, root):
    if model_id == "LLV-HD-Wide-2x-v0.3.0_2304x1296":
        source = PORT / "graph/portable/source_graph.json"
        evidence = [PORT / "graph/portable/acceptance/validation.json", PORT / "graph/portable/temporal_acceptance/validation.json"]
    else:
        source = PORT / "catalog_ports" / model_id / "portable/source_graph.json"
        evidence = [PORT / "catalog_ports" / model_id / "acceptance/validation.json"]
    plan = json.loads(source.read_text())
    coefficients = load_coefficients(plan)
    precision = plan.get("execution_profile") == "metal_state_ane_denoiser"
    if model_id == "LLV-HD-Wide-2x-v0.3.0_2464x1388":
        evidence += [source.parent.parent/"temporal_acceptance/validation.json"]
    fixed_report = evidence[0] if precision else source.parent / "h13/validation.json"
    if not fixed_report.exists():
        fixed_report = PORT / "catalog_ports" / model_id / "acceptance/validation.json"
    if not fixed_report.exists() or not json.loads(fixed_report.read_text()).get("passed"):
        raise ValueError(f"no passing fixed-input ANE evidence for {model_id}")
    folder = root / model_id
    folder.mkdir(parents=True, exist_ok=True)
    portable_plan = copy.deepcopy(plan)
    for group, record in portable_plan["coefficients"].items():
        dest = folder / "coefficients" / group
        dest.mkdir(parents=True, exist_ok=True)
        for key in ("weights", "bias"):
            if record[key]:
                shutil.copy2(PORT / record[key], dest / (key + ".npy"))
                record[key] = str((dest / (key + ".npy")).relative_to(folder))
    logical_graph = build_ane(plan, coefficients)
    execution_plan = export_prefix(plan,coefficients,folder/"precision")[1] if precision else plan
    graph = build_ane(execution_plan, coefficients)
    for node in plan["nodes"]:
        if node["op"] == "concat":
            if len({logical_graph[n].shape[2:] for n in node["inputs"]}) != 1:
                raise ValueError(f"invalid exported spatial contract: {node['name']}")
        elif node["op"] in ("add", "mul", "maximum", "absdiff"):
            shapes = [logical_graph[n].shape for n in node["inputs"]]
            if tuple(np.broadcast_shapes(*shapes)) != tuple(logical_graph[node["name"]].shape):
                raise ValueError(f"invalid exported broadcast contract: {node['name']}")
    outputs = [graph[n] for n in execution_plan["outputs"]]
    order = _topo_multi(*outputs)
    for index, tensor in enumerate(order):
        tensor._name = f"t{index}"
    emitter = _Emitter(int8=False)
    for tensor in order:
        if tensor.op != "input":
            _EMIT[tensor.op](emitter, tensor, tensor._name, [s._name for s in tensor.srcs])
    inputs = sorted((t for t in order if t.op == "input"), key=lambda t: t.attrs.get("idx", 0))
    _emit_program_dir(emitter, inputs, ", ".join(t._name for t in outputs), None, folder)
    (folder / "source_graph.json").write_text(json.dumps(portable_plan, indent=2) + "\n")
    evidence += [fixed_report, source.parent / "schedule_comparison.json"]
    evidence_folder = folder / "evidence"
    evidence_folder.mkdir(exist_ok=True)
    validation = []
    for index, path in enumerate(evidence):
        if path.exists():
            dest = evidence_folder / f"{index}_{path.parent.name}_{path.name}"
            shutil.copy2(path, dest)
            validation.append({"source": str(path.relative_to(PORT)), "file": str(dest.relative_to(folder)),
                               "passed": json.loads(path.read_text()).get("passed")})
    def port(name):
        return {"name": name, "port": graph[name]._name, "shape": list(graph[name].shape)}
    ane_inputs=[port(n["name"]) for n in execution_plan["nodes"] if n["op"]=="input"]
    ane_outputs=[port(n) for n in execution_plan["outputs"]]
    def external_port(name):
        return {"name":name,"port":name,"shape":list(logical_graph[name].shape)}
    files = {str(p.relative_to(folder)): sha(p.read_bytes()) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and p.name != "export_validation.json"}
    manifest = {"model_id": model_id, "source_model_sha256": plan["model_sha256"],
                "source_graph_sha256_before_export": sha(source.read_bytes()),
                "node_count": len(plan["nodes"]), "coefficient_group_count": len(coefficients),
                "execution_profile":plan.get("execution_profile","ane"),
                "inputs": [external_port(n["name"]) if precision else port(n["name"]) for n in plan["nodes"] if n["op"] == "input"],
                "outputs": [external_port(n) if precision else port(n) for n in plan["outputs"]],
                "ane_inputs":ane_inputs,"ane_outputs":ane_outputs,"files": files, "evidence": validation,
                "scope": "recovered graph on local Mac ANE; original H19 permission denied; native ISP pipeline is separate",
                "known_failed_evidence": [r["file"] for r in validation if r["passed"] is False]}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(model_id, "nodes", len(plan["nodes"]), "weights", len(coefficients), "known failures", manifest["known_failed_evidence"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_ids", nargs="+")
    parser.add_argument("--out", type=pathlib.Path, default=PORT / "model_assets")
    args = parser.parse_args()
    prepare_runtime(args.out)
    for model_id in args.model_ids:
        export_model(model_id, args.out)
