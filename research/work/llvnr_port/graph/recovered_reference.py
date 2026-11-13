#!/usr/bin/env python3
"""Build a task reference from a local extraction and recovered coefficients.

This module is deliberately independent of the old private ``dense_weights`` /
``special_weights`` research trees.  The caller supplies:

* a ``model.hwx`` from its own local extraction; and
* the private recovery directory produced by ``recover_portable_sources.py``.

Only the model-derived arrays and task metadata are read locally.  The
repository carries no model weights or probe HWX files.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

GRAPH_DIR = pathlib.Path(__file__).resolve().parent
PORT = GRAPH_DIR.parent
WORK = PORT.parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(GRAPH_DIR))

from native_reference import NativeReference  # noqa: E402
from h19_weightlib import kernel_groups, parse_hwx  # noqa: E402
from llvnr_ir import export  # noqa: E402

PROFILE_PATH = PORT / "recovery_profiles.json.gz"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_profiles(path: pathlib.Path | None = None) -> dict[str, Any]:
    path = path or PROFILE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"recovery profiles not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"invalid recovery profile: {path}")
    return value


class RecoveredReference(NativeReference):
    """A ``NativeReference`` backed by a user's extracted model and arrays."""

    def __init__(self, model_path: pathlib.Path, recovery_dir: pathlib.Path,
                 model_id: str | None = None, profiles: dict[str, Any] | None = None):
        self.model_path = pathlib.Path(model_path).resolve()
        self.recovery_dir = pathlib.Path(recovery_dir).resolve()
        self.model_id = model_id or self.model_path.parents[3].stem
        self.ir = export(self.model_path)
        canvas_text = self.model_id.rsplit("_", 1)[1]
        self.canvas_wh = tuple(map(int, canvas_text.split("x")))
        if len(self.canvas_wh) != 2:
            raise ValueError(f"invalid canvas in model id: {self.model_id}")

        data, sections, symbols = parse_hwx(self.model_path)
        self.static_bars: dict[int, bytes] = {}
        if (section := sections.get(("__TEXT", "__const"))) is not None:
            self.static_bars[4] = data[section.offset:section.offset + section.size]
        self.bar_bytes = {12: sections[("__DATA", "__bss")].size}
        self.input_bars: dict[int, str] = {}
        self.output_bars: dict[int, str] = {}
        self.port_views: dict[str, dict[str, Any]] = {}
        for index, port in enumerate(sorted(self.ir["ports"], key=lambda p: p["vm_address"])):
            bar, size = 20 + 4 * index, port["section_bytes"][0]
            name = port["name"].removesuffix("_tensor")
            self.bar_bytes[bar] = size
            (self.output_bars if port["name"].endswith("_tensor") else self.input_bars)[bar] = name
            channels, height, width = ((1, self.canvas_wh[1] // 4, self.canvas_wh[0] // 4)
                                       if "map" in name or "weight" in name
                                       else (6, self.canvas_wh[1] // 2, self.canvas_wh[0] // 2))
            row_bytes, remainder = divmod(size, channels * height)
            if remainder or row_bytes < width * 2 or row_bytes % 2:
                raise ValueError(f"incompatible port storage for {name}")
            self.port_views[name] = {
                "space": str(bar), "base": 0, "interleave": 1,
                "channel_stride": row_bytes * height, "row_stride": row_bytes,
                "shape_chw": [channels, height, width],
            }

        groups = kernel_groups(data, sections, symbols)
        self.groups_by_offset = {min(group.starts): group for group in groups}
        manifest_path = self.recovery_dir / "recovery.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"recovery manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("model_sha256") != self.ir["model_sha256"]:
            raise ValueError(f"coefficient source model identity mismatch: {self.model_id}")

        self.coefficients: dict[str, dict[str, Any]] = {}
        self.recovery_rows = {row["group"]: row for row in manifest.get("groups", [])}
        for group, row in self.recovery_rows.items():
            if row.get("status") == "unresolved":
                continue
            folder = self.recovery_dir / "coefficients" / group
            weights_path, bias_path = folder / "weights.npy", folder / "bias.npy"
            if not weights_path.is_file():
                raise FileNotFoundError(f"recovered weights missing: {weights_path}")
            weights = np.load(weights_path)
            bias = np.load(bias_path) if bias_path.is_file() else None
            if sha256(weights.tobytes()) != row.get("weight_sha256"):
                raise ValueError(f"coefficient identity mismatch: {group}")
            bias_hash = sha256(bias.tobytes()) if bias is not None else None
            if bias_hash != row.get("bias_sha256"):
                raise ValueError(f"bias identity mismatch: {group}")
            self.coefficients[group] = {
                "weights": weights.astype(np.float32),
                "bias": bias.astype(np.float32) if bias is not None else None,
                "primitive": row.get("primitive", "conv2d"),
                "evidence": dict(row),
            }

        self.activation_tables: dict[str, np.ndarray] = {}
        self.luts: dict[str, dict[str, Any]] = {}
        profile = profiles or load_profiles()
        self.lut_profiles = profile.get("activation_luts", {})
        for group in groups:
            if group.kind == "actlut":
                self.classify_lut(group.stem, group.raw[:128])
            elif (group.stem in self.coefficients and
                  self.coefficients[group.stem]["primitive"] in
                  ("conv2d_then_sigmoid", "conv2d_then_lut", "conv_transpose_then_lut",
                   "padding_identity_then_lut")):
                self.classify_lut(group.stem, group.raw[:128])
        self.remaining_validation = [
            "whole-frame graph versus native task CPU",
            "H13 execution and continuous history",
            "H19 hardware numerics unavailable on this host",
            "ISP preprocessing and temporal feedback are outside the graph",
        ]
        self.storage: dict[str, np.ndarray] = {}
        self.trace: list[dict[str, Any]] = []
        self.captures: dict[int, np.ndarray] = {}

    def classify_lut(self, group: str, raw: bytes) -> None:
        table = np.frombuffer(raw, dtype="<f2").copy()
        digest = sha256(raw)
        self.activation_tables[group] = table
        known = self.lut_profiles.get(digest)
        if known:
            row = dict(known)
            row["group"] = group
            if row.get("operation") == "clip":
                row["bounds"] = row.get("bounds") or table[:2].astype(float).tolist()
            self.luts[group] = row
            return
        # The common 128-byte compiler tables have stable structural shapes.
        if (table[0] < 0 and table[1] == 0 and not np.any(table[2:4])) or (
                table[0] == table[2] and table[1] == table[3]
                and np.array_equal(table[37:41], [1, 0, 1, 0])):
            self.luts[group] = {"group": group, "operation": "clip",
                                "bounds": table[:2].astype(float).tolist()}
            return
        raise ValueError(f"new activation table requires controlled validation: {group}")

    def coefficient_records(self, groups: set[str] | list[str]) -> dict[str, dict[str, Any]]:
        """Return portable relative records for the selected coefficient groups."""
        result: dict[str, dict[str, Any]] = {}
        for group in sorted(set(groups)):
            if group not in self.coefficients:
                raise ValueError(f"coefficient group not recovered for {self.model_id}: {group}")
            record = self.coefficients[group]
            weights, bias = record["weights"], record["bias"]
            result[group] = {
                "weights": f"coefficients/{group}/weights.npy",
                "bias": f"coefficients/{group}/bias.npy" if bias is not None else None,
                "weights_sha256_fp16": sha256(weights.astype(np.float16).tobytes()),
                "bias_sha256_fp16": sha256(bias.astype(np.float16).tobytes()) if bias is not None else None,
            }
        return result
