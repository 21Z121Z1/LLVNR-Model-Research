#!/usr/bin/env python3
"""Bring up each recovered model against its own native task schedule.

Ports and external row strides come from each model's own storage contract.
No catalog entry becomes runnable based on this CPU bring-up alone.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from native_reference import NativeReference, synthetic_inputs
from llvnr_taskflow import BAR_BYTES, INPUT_BARS, OUTPUT_BARS

PORT = pathlib.Path(__file__).resolve().parents[1]
WORK = PORT.parent
sys.path.insert(0, str(PORT))
from recover_catalog import baseline_registry, recover_group
from h19_weightlib import kernel_groups, kernel_section, parse_hwx
from llvnr_ir import export


class CatalogReference(NativeReference):
    def __init__(self, model_id):
        model = next(p for p in (WORK / "ota_extract/llv_catalog").rglob("model.hwx")
                     if p.parents[3].stem == model_id)
        self.ir = export(model)
        canvas_w, canvas_h = map(int, model_id.rsplit("_", 1)[1].split("x"))
        self.canvas_wh = (canvas_w, canvas_h)
        data, sections, symbols = parse_hwx(model)
        self.static_bars = {}
        if (section := sections.get(("__TEXT","__const"))) is not None:
            self.static_bars[4] = data[section.offset:section.offset+section.size]
        self.bar_bytes = {12: sections[("__DATA", "__bss")].size}
        self.input_bars, self.output_bars, self.port_views = {}, {}, {}
        for index, port in enumerate(sorted(self.ir["ports"], key=lambda p: p["vm_address"])):
            bar, size = 20 + 4 * index, port["section_bytes"][0]
            name = port["name"].removesuffix("_tensor")
            self.bar_bytes[bar] = size
            (self.output_bars if port["name"].endswith("_tensor") else self.input_bars)[bar] = name
            channels, height, width = (1, canvas_h // 4, canvas_w // 4) if "map" in name or "weight" in name else (6, canvas_h // 2, canvas_w // 2)
            row_bytes, remainder = divmod(size, channels * height)
            if remainder or row_bytes < width * 2 or row_bytes % 2:
                raise ValueError(f"incompatible port storage for {name}")
            self.port_views[name] = {"space": str(bar), "base": 0, "interleave": 1,
                                     "channel_stride": row_bytes * height, "row_stride": row_bytes,
                                     "shape_chw": [channels, height, width]}
        groups = kernel_groups(data, sections, symbols)
        self.groups_by_offset = {min(g.starts): g for g in groups}
        folder = PORT / "catalog_ports" / model_id
        manifest = json.loads((folder / "recovery.json").read_text())
        if manifest["model_sha256"] != self.ir["model_sha256"]:
            raise ValueError("coefficient source model identity mismatch")
        self.coefficients, self.activation_tables, self.luts = {}, {}, {}
        known = baseline_registry()
        self.remaining_validation = ["controlled new coefficient formats", "H19 hardware numerics unavailable: entitlement denial",
                                     "whole-frame graph versus native schedule", "H13 execution and continuous history"]
        for row in manifest["groups"]:
            stem = row["group"]
            if row["status"] == "unresolved":
                raise ValueError(f"unrecovered coefficients {stem}")
            path = folder / stem
            weights = np.load(path / "weights.npy")
            bias_path = path / "bias.npy"
            bias = np.load(bias_path) if bias_path.exists() else None
            if hashlib.sha256(weights.tobytes()).hexdigest() != row["weight_sha256"]:
                raise ValueError(f"coefficient identity mismatch: {stem}")
            bias_hash = hashlib.sha256(bias.tobytes()).hexdigest() if bias is not None else None
            if bias_hash != row["bias_sha256"]:
                raise ValueError(f"bias identity mismatch: {stem}")
            weights = weights.astype(np.float32)
            bias = bias.astype(np.float32) if bias is not None else None
            primitive = row["primitive"]
            evidence = dict(row)
            if row["status"] == "reused_exact_native_payload":
                evidence.update(known[stem]["evidence"])
            self.coefficients[stem] = {"weights": weights, "bias": bias, "primitive": primitive, "evidence": evidence}
        sigmoid_table = known[next(k for k in known if k.startswith("K2C1AE"))]["region"][:128]
        known_luts = {row["sha256"]: row for row in json.loads((PORT / "special_weights/manifest.json").read_text())["activation_luts"]
                      if row.get("controlled_compiler_equal")}
        sqrt_probe = PORT / "format_probes/sqrt/inspect.json/validation.json"
        if sqrt_probe.exists():
            probe = json.loads(sqrt_probe.read_text())
            if hashlib.sha256(pathlib.Path(probe["source"]).read_bytes()).hexdigest() != probe["hwx_sha256"]:
                raise ValueError("controlled sqrt probe identity mismatch")
            known_luts[probe["lut_sha256"]] = {"operation": "sqrt", "controlled_compiler_equal": True,
                                              "source": str(sqrt_probe.relative_to(PORT))}

        def classify(group, raw):
            table = np.frombuffer(raw, dtype="<f2").copy()
            self.activation_tables[group] = table
            if (digest := hashlib.sha256(raw).hexdigest()) in known_luts:
                self.luts[group] = {**known_luts[digest], "group": group}
            elif raw == sigmoid_table:
                self.luts[group] = {"group": group, "operation": "sigmoid", "scale": 1.0}
            elif ((table[0] < 0 and table[1] == 0 and not np.any(table[2:4]))
                  or (table[0] == table[2] and table[1] == table[3]
                      and np.array_equal(table[37:41], [1, 0, 1, 0]))):
                self.luts[group] = {"group": group, "operation": "clip", "bounds": table[:2].astype(float).tolist()}
            else:
                raise ValueError(f"new activation table requires controlled validation: {group}")

        for group in groups:
            if group.kind == "actlut":
                classify(group.stem, group.raw[:128])
            elif group.stem in self.coefficients and self.coefficients[group.stem]["primitive"] in ("conv2d_then_sigmoid", "conv2d_then_lut", "conv_transpose_then_lut", "padding_identity_then_lut"):
                classify(group.stem, group.raw[:128])
        self.storage, self.trace, self.captures = {}, [], {}


def baseline_decoder_regression():
    """New descriptor-driven layouts must match independently recovered tensors."""
    model = next((WORK / "ota_extract/llv_model_2304").rglob("model.hwx"))
    ir = export(model)
    data, sections, symbols = parse_hwx(model)
    groups = kernel_groups(data, sections, symbols)
    _, raw = kernel_section(data, sections)
    known = baseline_registry()
    rows = []
    for group in groups:
        if group.stem not in known:
            continue
        tasks = [t for t in ir["tasks"] if any(a["register_word"] == 0x1544 and a["offset"] == min(group.starts)
                                             for a in t["address_writes"])]
        candidates, _ = recover_group(group, tasks[0], raw, groups)
        if len(candidates) != 1:
            continue
        got, expected = candidates[0], known[group.stem]
        passed = np.array_equal(got["weights"], expected["weights"])
        passed &= ((got["bias"] is None and expected["bias"] is None) or np.array_equal(got["bias"], expected["bias"]))
        rows.append({"group": group.stem, "passed": bool(passed), "shape": list(got["weights"].shape)})
    if len(rows) < 20 or not all(row["passed"] for row in rows):
        raise ValueError(f"generic decoder disagrees with baseline: {rows}")
    return {"cases": rows, "passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", nargs="?", default="LLV-HD-Tele-4x-v0.3.0_2304x1296")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    folder = PORT / "catalog_ports" / args.model_id / "native_cpu"
    folder.mkdir(parents=True, exist_ok=True)
    regression = baseline_decoder_regression()
    (folder / "decoder_regression.json").write_text(json.dumps(regression, indent=2) + "\n")
    reference = CatalogReference(args.model_id)
    inputs = synthetic_inputs(args.seed, reference.canvas_wh)
    started, failure, outputs = time.perf_counter(), None, {}
    try:
        outputs = reference.run(inputs)
    except RuntimeError as error:
        failure = str(error)
    report = {"model_id": args.model_id, "model_sha256": reference.ir["model_sha256"],
              "tasks_executed": sum(r["status"] == "ok" for r in reference.trace), "task_count": reference.ir["task_count"],
              "seconds": time.perf_counter() - started, "error": failure, "trace": reference.trace,
              "decoder_regression_passed": regression["passed"],
              "remaining_validation": reference.remaining_validation,
              "scope": "native-schedule CPU candidate, not H19 hardware golden or accepted Mac app port"}
    if outputs:
        np.savez_compressed(folder / "outputs.npz", **outputs)
        report["outputs"] = {name: {"finite": bool(np.isfinite(value).all()), "shape": list(value.shape),
                                            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                             for name, value in outputs.items()}
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
