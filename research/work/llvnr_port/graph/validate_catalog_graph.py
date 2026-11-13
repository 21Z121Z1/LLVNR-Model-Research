#!/usr/bin/env python3
"""Per-model whole-frame CPU, native schedule and local ANE acceptance."""
import argparse
import hashlib
import json
import pathlib
import platform

import numpy as np

from catalog_reference import CatalogReference
from wide1_graph import MODELS as SMALL_MODELS, source_plan as small_source_plan
from large_canvas_graph import MODELS as LARGE_MODELS, source_plan as large_source_plan
from tele8_graph import MODELS as TELE8_MODELS, source_plan as tele8_source_plan
from superwide05_graph import MODELS as SUPERWIDE05_MODELS, source_plan as superwide05_source_plan, chroma_cases
from portable_graph import H13Model, evaluate_cpu, load_coefficients, compare
from validate_portable import controlled_cases, numeric_check, sha

PORT = pathlib.Path(__file__).resolve().parents[1]
MODELS = SMALL_MODELS + LARGE_MODELS + TELE8_MODELS + SUPERWIDE05_MODELS


def validate(model_id):
    folder = PORT / "catalog_ports" / model_id / "acceptance"
    folder.mkdir(parents=True, exist_ok=True)
    reference = CatalogReference(model_id)
    builder = (superwide05_source_plan if model_id in SUPERWIDE05_MODELS else
               tele8_source_plan if model_id in TELE8_MODELS else (large_source_plan if model_id in LARGE_MODELS else small_source_plan))
    plan = builder(reference)
    portable = folder.parent / "portable"
    portable.mkdir(exist_ok=True)
    plan_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    (portable / "source_graph.json").write_bytes(plan_bytes)
    coefficients = load_coefficients(plan)
    report = {"model_id": model_id, "model_sha256": plan["model_sha256"],
              "execution_profile": plan.get("execution_profile","ane"),
              "source_graph_sha256": hashlib.sha256(plan_bytes).hexdigest(),
              "cases": [], "recurrence": [], "passed": False,
              "numeric_gate": {"max_abs": 0.02, "mean_abs": 0.001, "finite": True},
              "host": {"system": platform.platform(), "machine": platform.machine()},
              "scope": "full original task CPU versus independent whole-frame CPU and local Mac ANE; original H19 hardware golden unavailable",
              "host_policy": "explicit state_yuv feedback with independent CPU/ANE histories; ISP alignment and host tuning remain separate"}

    def save():
        (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    model = H13Model(plan, coefficients, folder / "h13")
    try:
        cases = [*controlled_cases(reference.canvas_wh)]
        if model_id in SUPERWIDE05_MODELS:
            cases += list(chroma_cases(reference.canvas_wh))
        for name, inputs in cases:
            values = evaluate_cpu(plan, coefficients, inputs)
            native = reference.run(inputs)
            outputs = model.run(inputs)
            schedule = {key: compare(values[key], native[key]) for key in plan["outputs"]}
            ane = {key: numeric_check(outputs[key], values[key]) for key in plan["outputs"]}
            row = {"case": name, "inputs_sha256": {k: sha(v) for k, v in inputs.items()},
                   "native_tasks_executed": len(reference.trace), "schedule": schedule, "h13": ane,
                   "passed": all(v["exact_fraction"] == 1 for v in schedule.values()) and all(v["passed"] for v in ane.values())}
            report["cases"].append(row)
            save()
            print(model_id, name, row["passed"], flush=True)
        cpu_history = h13_history = None
        rng = np.random.default_rng(20261114)
        for index in range(8):
            width, height = reference.canvas_wh
            current = rng.normal(0.2, 0.02, (1, 6, height // 2, width // 2)).astype(np.float16)
            current[:, 4:] -= np.float16(0.2)
            reset = index in (0, 4)
            if reset:
                cpu_history, h13_history = current.copy(), current.copy()
            maps = {"dnr_zoom_map": np.full((1, 1, height // 4, width // 4), 0.5, np.float16),
                    "fusion_zoom_map": np.full((1, 1, height // 4, width // 4), 0.5, np.float16)}
            cpu_inputs = {"cur_frame": current, "his_frame": cpu_history, **maps}
            h13_inputs = {"cur_frame": current, "his_frame": h13_history, **maps}
            cpu = evaluate_cpu(plan, coefficients, cpu_inputs)
            actual = model.run(h13_inputs)
            metrics = {k: numeric_check(actual[k], cpu[k]) for k in plan["outputs"]}
            row = {"frame": index, "reset": reset, "cpu_inputs_sha256": {k: sha(v) for k, v in cpu_inputs.items()},
                   "h13_inputs_sha256": {k: sha(v) for k, v in h13_inputs.items()}, "h13": metrics,
                   "passed": all(v["passed"] for v in metrics.values())}
            report["recurrence"].append(row)
            save()
            cpu_history, h13_history = cpu["state_yuv"].copy(), actual["state_yuv"].copy()
            print(model_id, "recurrence", index, row["passed"], flush=True)
        report["expected_case_count"] = len(cases)
        report["passed"] = len(report["cases"]) == len(cases) and len(report["recurrence"]) == 8 and all(r["passed"] for r in report["cases"] + report["recurrence"])
        report["compile_seconds"] = model.compile_seconds
        save()
        return report["passed"]
    except Exception as error:
        report["error"] = str(error)
        save()
        raise
    finally:
        model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_ids", nargs="+", choices=MODELS)
    args = parser.parse_args()
    results = [validate(model_id) for model_id in args.model_ids]
    raise SystemExit(0 if all(results) else 1)
