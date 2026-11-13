#!/usr/bin/env python3
"""Recover independently bounded coefficient candidates for all 12 models.

Reused coefficients require exact native payload equality. New dense/sparse
candidates require descriptor shapes, physical block addresses and byte-exact
re-encoding. This is coefficient evidence, never model inference acceptance.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

import numpy as np

PORT = pathlib.Path(__file__).resolve().parent
WORK = PORT.parent
sys.path.insert(0, str(WORK))
from h19_dense_codec import block_layout, decode as dense_decode, encode as dense_encode
from h19_sparse_codec import decode as sparse_decode, encode_block, region_size
from h19_sparse_extended import decode as extended_decode, encode as extended_encode
from h19_sparse_stream import decode as stream_decode, encode as stream_encode
from sparse_transpose_codec import registry as sparse_transpose_registry, recover_validated as recover_sparse_transpose
from h19_weightlib import kernel_groups, kernel_section, parse_hwx
from llvnr_ir import export
from h19_deconv_probe import skeleton
from llvnr_taskflow import address, dimensions, storage_view


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def baseline_registry():
    source = next((WORK / "ota_extract/llv_model_2304").rglob("model.hwx"))
    data, sections, symbols = parse_hwx(source)
    groups = {g.stem: g for g in kernel_groups(data, sections, symbols)}
    _, raw = kernel_section(data, sections)
    result = {}
    for directory in ("dense_weights", "special_weights", "deconv_weights"):
        manifest = json.loads((PORT / directory / "manifest.json").read_text())
        assert manifest["model_sha256"] == sha(data)
        for row in manifest["groups"]:
            stem = row["group"]
            folder = PORT / directory / stem
            weights = np.load(folder / "weights.npy")
            bias_path = folder / "bias.npy"
            bias = np.load(bias_path) if bias_path.exists() else None
            if "region" in row:
                start, end = row["region"]
                region = raw[start:end]
            elif "coefficient_region_bytes" in row:
                region = groups[stem].raw[:row["coefficient_region_bytes"]]
            elif directory == "special_weights" and row["primitive"] == "conv2d":
                shape = weights.shape
                packing = [1] * 6 if shape[0] == 6 else ([2] * 8 + [1] * 8 if shape[0] == 24 else [2] * 16)
                region = groups[stem].raw[:region_size(groups[stem].raw, packing)]
            elif row["primitive"] == "conv2d_then_sigmoid":
                region = groups[stem].raw[:448]
            elif row["primitive"] == "conv_transpose_haar":
                region = groups[stem].raw[:384]
            else:
                raise ValueError(f"unbounded baseline evidence: {stem}")
            result[stem] = {"region": region, "weights": weights, "bias": bias,
                            "primitive": row.get("primitive", "conv_transpose" if directory == "deconv_weights" else "conv2d"),
                            "source": str(folder.relative_to(PORT)), "evidence": row}
    return result


def packing_candidates(cout):
    candidates = [(p,) * (cout // p) for p in range(1, cout + 1) if cout % p == 0]
    # Compiler lanes can finish with smaller blocks (e.g. 24 outputs across
    # 16 physical lanes). These observed patterns are also fully re-encoded.
    for large in range(2, 9):
        for count in range(1, 17):
            rest = cout - large * count
            if 0 <= rest <= 16:
                candidates.append((large,) * count + (1,) * rest)
    return sorted(set(candidates), key=lambda p: (len(p), p))


def transpose_registry():
    result = []

    def bind_lanes(layout, ir, offsets):
        layout["controlled_active_ne"] = sorted({1 << ((task["written_registers"].get("000f",0) >> 19) & 7)
            for task in ir["tasks"] if address(task,0x1544) and address(task,0x1544)["offset"] in offsets})
    for path in sorted((WORK / "h19_deconv_probes").glob("*/layout.json")):
        layout = json.loads(path.read_text())
        source = path.parent / "coded0/out/model.hwx"
        data, sections, symbols = parse_hwx(source)
        _, raw = kernel_section(data, sections)
        template = skeleton(raw[:layout["region_bytes"]], layout["weight_byte_offsets"])
        if sha(template) != layout["skeleton_sha256"] or not layout["heldout_bitwise_equal"]:
            raise ValueError("invalid controlled transpose evidence")
        layout["weight_byte_locations"] = [[o] for o in layout["weight_byte_offsets"]]
        ir = export(source)
        matching = {min(g.starts) for g in kernel_groups(data, sections, symbols)
                    if g.raw[:layout["region_bytes"]] == raw[:layout["region_bytes"]]}
        origins = {}
        for task in ir["tasks"]:
            addr = address(task, 0x1544)
            if addr is None or addr["offset"] not in matching:
                continue
            regs = {int(k, 16): v for k, v in task["written_registers"].items()}
            src = storage_view(task, regs, dimensions(regs), "source1")
            output = storage_view(task, regs, dimensions(regs, True), "output")
            local_y = local_x = 0
            if output["space"] == "16":
                copy = ir["tasks"][task["index"] + 1]
                copy_regs = {int(k, 16): v for k, v in copy["written_registers"].items()}
                copy_source = storage_view(copy, copy_regs, dimensions(copy_regs), "source1")
                local_y, local_bytes = divmod(copy_source["base"] - output["base"], output["row_stride"])
                local_x = local_bytes // 2
                output = storage_view(copy, copy_regs, dimensions(copy_regs, True), "output")
            origin = [output.get("crop_y", 0) - 2 * src.get("crop_y", 0) + layout["pad"][0] - local_y,
                      output.get("crop_x", 0) - 2 * src.get("crop_x", 0) + layout["pad"][2] - local_x]
            key = f'{task["geometry"]["pad_top"]},{task["geometry"]["pad_left"]}'
            if key in origins and origins[key] != origin:
                raise ValueError("inconsistent controlled 4x4 transpose origin")
            origins[key] = origin
        layout["native_origin_by_geometry_padding"] = origins
        bind_lanes(layout, ir, matching)
        result.append((path, layout, template))
    for path in sorted((PORT / "format_probes").glob("transpose*/layout_*.json")):
        layout = json.loads(path.read_text())
        if layout["missing_coefficients"] or not layout["heldout_bitwise_equal"]:
            continue
        data, sections, symbols = parse_hwx(PORT / layout["source"])
        group = [g for g in kernel_groups(data, sections, symbols) if g.kind == "ne"][layout["group_index"]]
        offsets = [o for row in layout["weight_byte_locations"] for o in row]
        template = skeleton(group.raw, offsets)
        if sha(template) != layout["skeleton_sha256"]:
            raise ValueError("invalid catalog permutation template")
        bind_lanes(layout, export(PORT / layout["source"]), {min(group.starts)})
        layout["pad"] = [0, 0, 0, 0]
        layout["embedded_activation"] = "_plain" not in layout["case"] and layout["case"] not in ("transpose32", "transpose32to24_k2")
        result.append((path, layout, template))
    for path in sorted((PORT / "pipeline_probes").glob("transpose*/layout_*.json")):
        layout = json.loads(path.read_text())
        data, sections, symbols = parse_hwx(PORT / layout["source"])
        group = next(g for g in kernel_groups(data, sections, symbols) if g.stem == layout["source_group"])
        offsets = [o for row in layout["weight_byte_locations"] for o in row]
        template = skeleton(group.raw[:layout["region_bytes"]], offsets)
        if sha(template) != layout["skeleton_sha256"] or not layout["heldout_bitwise_equal"]:
            raise ValueError("invalid controlled pipeline transpose evidence")
        bind_lanes(layout, export(PORT / layout["source"]), {min(group.starts)})
        result.append((path, layout, template))
    return result


def recover_transpose(group, task, registry):
    dims = task["explicit_dimensions_whc"]
    cin, cout = dims["input"][2] or 16, dims["output"][2] or 16
    geo = task["geometry"] or {}
    if max(geo.get("output_x_factor", 1), geo.get("output_y_factor", 1)) != 2:
        return None
    for path, layout, template in registry:
        if layout["shape"][:2] != [cin, cout]:
            continue
        if (1 << ((task["written_registers"].get("000f",0) >> 19) & 7)) not in layout["controlled_active_ne"]:
            continue
        size = layout["region_bytes"]
        native = group.raw[:size]
        locations = layout["weight_byte_locations"]
        offsets = [o for row in locations for o in row]
        if len(native) != size or skeleton(native, offsets) != template:
            continue
        if any(len({native[o:o + 2] for o in row}) != 1 for row in locations):
            continue
        bits = np.array([int.from_bytes(native[row[0]:row[0] + 2], "little") for row in locations], dtype=np.uint16)
        weights = bits.view(np.float16).reshape(layout["shape"])
        if not np.isfinite(weights).all():
            continue
        rebuilt = bytearray(template)
        for positions, value in zip(locations, bits):
            for offset in positions:
                rebuilt[offset:offset + 2] = int(value).to_bytes(2, "little")
        if bytes(rebuilt) != native:
            raise ValueError("transpose native round-trip failed")
        return {"weights": weights, "bias": None, "primitive": "conv_transpose_then_lut" if layout.get("embedded_activation") else "conv_transpose", "region": native,
                "source": str(path.relative_to(WORK)), "pad_for_probe": layout["pad"],
                "native_origin_top_left": layout.get("native_origin_top_left"),
                "native_origin_by_geometry_padding": layout.get("native_origin_by_geometry_padding")}
    return None


def recover_expansion_2x2(group, task, raw):
    """Inverse of the independently probed 2x2 phase/output packing."""
    regs, geo = task["written_registers"], task["geometry"] or {}
    if (geo.get("kh"), geo.get("kw"), geo.get("output_y_factor"), geo.get("output_x_factor")) != (1, 1, 2, 2):
        return None
    if regs.get("1240", 0) & 4:
        return None
    dims = task["explicit_dimensions_whc"]
    cin, cout = dims["input"][2] or 16, dims["output"][2] or 16
    cores = min(cout, 1 << (regs.get("000f", 0) >> 19 & 7))
    counts = [cout // cores + (i < cout % cores) for i in range(cores)]
    cfg = regs.get("1242", 0)
    if cfg >> 16 & 3:
        return None
    has_bias = bool(cfg & 0x10)
    weights, biases = [], []
    origin, offset = min(group.starts), 0
    try:
        for count in counts:
            if origin + offset not in group.starts:
                raise ValueError("expansion lane is not symbol-addressed")
            physical = count * 4
            block = raw[origin + offset:]
            if regs.get("1240", 0) & 0x100:
                w, b, size = extended_decode(block, physical, (cin, 1, 1), has_bias)
                encoded = extended_encode(w, b)
            else:
                size = sum(b[4] for b in block_layout((physical, cin, 1, 1), physical, has_bias))
                w, b = dense_decode(block[:size], (physical, cin, 1, 1), physical, has_bias)
                encoded = dense_encode(w, b, physical)
            if encoded != block[:size]:
                raise ValueError("expansion native round-trip mismatch")
            weights.append(w.reshape(count, 2, 2, cin).transpose(3, 0, 1, 2))
            if b is not None:
                b = b.reshape(count, 4)
                if not np.all(b == b[:, :1]):
                    raise ValueError("phase-dependent expansion bias")
                biases.append(b[:, 0])
            offset += size
        return {"weights": np.concatenate(weights, axis=1).copy(),
                "bias": np.concatenate(biases) if biases else None, "primitive": "conv_transpose",
                "region": raw[origin:origin + offset], "pad_for_probe": [0, 0, 0, 0],
                "source": "llvnr_port/format_probes/transpose32/layout_0.json"}
    except (ValueError, IndexError):
        return None


def padding_registry():
    result = []
    for path in sorted((PORT / "format_probes").glob("pad_*/report.json")):
        report = json.loads(path.read_text())
        source = path.parent / "out/model.hwx"
        data, sections, symbols = parse_hwx(source)
        if sha(data) != report["model_sha256"]:
            raise ValueError("padding probe model identity mismatch")
        groups = [g for g in kernel_groups(data, sections, symbols) if g.kind == "ne"]
        if len(groups) != 1:
            raise ValueError("padding probe requires one coefficient group")
        result.append((path, report, groups[0].raw))
    return result


def recover_padding(group, task, registry):
    channels = task["explicit_dimensions_whc"]["input"][2] or 16
    for path, probe, payload in registry:
        if probe["input_shape"][1] != channels or task["geometry"] != probe["tasks"][0]["geometry"]:
            continue
        if not group.raw.startswith(payload):
            continue
        return {"weights": np.eye(channels, dtype=np.float16)[:, :, None, None], "bias": None,
                "primitive": "padding_identity_then_lut" if probe.get("activation_bounds") else "padding_identity",
                "region": payload, "source": str(path.relative_to(PORT)), "padding": probe["pad"],
                "native_padding_reference_top_left": [probe["tasks"][0]["geometry"]["pad_top"],
                                                       probe["tasks"][0]["geometry"]["pad_left"]],
                "activation_bounds": probe.get("activation_bounds"), "padding_mode": "edge"}
    return None


def recover_group_linear(group, task, raw, all_groups):
    regs = task["written_registers"]
    dims = task["explicit_dimensions_whc"]
    cin, cout = dims["input"][2] or 16, dims["output"][2] or 16
    geo = task["geometry"] or {"kh": 1, "kw": 1, "output_x_factor": 1, "output_y_factor": 1}
    if max(geo["output_x_factor"], geo["output_y_factor"]) > 1:
        return [], "unrecovered expansion/transposed kernel"
    depthwise = regs.get("000c", 0) & 0x1c000 == 0x14000
    if depthwise and cout != cin:
        return [], "unprobed depthwise channel multiplier"
    group_count = cin if depthwise else 1
    shape = (cout, cin // group_count, geo["kh"], geo["kw"])
    cfg = regs.get("1242", 0)
    bias = bool(cfg & 0x10)
    lut_bytes = 128 if cfg >> 16 & 3 == 2 else 0
    # `0x1240` bit 8 distinguishes compressed stride-2 coefficients in the
    # controlled compiler probes. Trying both remains bounded by exact masks,
    # explicit block starts and byte reconstruction, without inventing zeros.
    modes = ("sparse",) if regs.get("1240", 0) & 0x100 else ("dense",)
    origin = min(group.starts)
    next_group = min((s for g in all_groups for s in g.starts if s > max(group.starts)), default=len(raw))
    # Symbol groups can share complete physical blocks. A following group's
    # first symbol is not an upper bound on this group's last shared lane.
    span = raw[origin:next_group]
    accepted = []
    active_ne = 1 << (regs.get("000f", 0) >> 19 & 7)
    lanes = min(cout, active_ne)
    native_packing = tuple(cout // lanes + (i < cout % lanes) for i in range(lanes))
    for packing in (native_packing,):
        for mode in modes:
            if mode == "sparse" and max(packing) > 16:
                continue
            try:
                offset, weights, biases, tables, block_starts = 0, [], [], [], []
                for count in packing:
                    block_starts.append(origin + offset)
                    if origin + offset not in group.starts:
                        raise ValueError("missing physical block address")
                    if lut_bytes:
                        table = span[offset:offset + lut_bytes]
                        if len(table) != lut_bytes:
                            raise ValueError("truncated embedded activation")
                        tables.append(table)
                        offset += lut_bytes
                    block_shape = (count, *shape[1:])
                    if mode == "dense":
                        size = sum(b[4] for b in block_layout(block_shape, count, bias))
                        payload = span[offset:offset + size]
                        w, b = dense_decode(payload, block_shape, count, bias)
                        encoded = dense_encode(w, b, count)
                    else:
                        try:
                            w, b, size = extended_decode(span[offset:], count, shape[1:], bias)
                            payload = span[offset:offset + size]
                            encoded = extended_encode(w, b)
                            if encoded != payload:
                                raise ValueError("extended native payload mismatch")
                        except (ValueError, IndexError):
                            w, b, size = stream_decode(span[offset:], count, shape[1:], bias)
                            payload = span[offset:offset + size]
                            encoded = stream_encode(w, b)
                    if encoded != payload:
                        raise ValueError("native payload re-encoding mismatch")
                    offset += size
                    weights.append(w)
                    if b is not None:
                        biases.append(b)
                if len(set(tables)) > 1:
                    raise ValueError("different activation tables across channel blocks")
                if {s for s in group.starts if s < origin + offset} != set(block_starts):
                    raise ValueError("layout skipped an explicitly addressed physical block")
                if max(group.starts) > origin + offset:
                    raise ValueError("layout ended before the final physical lane address")
                w = np.concatenate(weights)
                if not bias and cout == 1:
                    b = np.array([regs.get("1244", 0) & 0xffff], dtype=np.uint16).view(np.float16)
                else:
                    b = np.concatenate(biases) if biases else None
                accepted.append({"weights": w, "bias": b, "shape": shape, "packing": packing,
                                 "groups": group_count,
                                 "bytes": offset, "mode": mode, "tables": tables[:1],
                                 "block_starts": block_starts,
                                 "primitive": "conv2d_then_lut" if tables else "conv2d"})
            except (ValueError, IndexError):
                continue
    # Equivalent packings can coincide when all padding is zero. Retain all
    # metadata but reject multiple different source tensors.
    unique = {}
    for item in accepted:
        key = (sha(item["weights"].tobytes()), sha(item["bias"].tobytes()) if item["bias"] is not None else None,
               item["bytes"], tuple(item["tables"]))
        unique.setdefault(key, item)
    return list(unique.values()), "no unique byte-exact layout" if len(unique) != 1 else None


def recover_group_symbol_candidate(group, task, raw, all_groups):
    """Decode logical NE lanes, retaining symbol aliases and inactive lanes.

    Unique byte offsets are not output-channel order: identical NE lanes can
    share an address, while inactive lanes can point at a standalone LUT.
    """
    regs = task["written_registers"]
    dims = task["explicit_dimensions_whc"]
    cin, cout = dims["input"][2] or 16, dims["output"][2] or 16
    geo = task["geometry"] or {"kh": 1, "kw": 1, "output_x_factor": 1, "output_y_factor": 1}
    if max(geo["output_x_factor"], geo["output_y_factor"]) > 1:
        return [], "unrecovered expansion/transposed kernel"
    depthwise = regs.get("000c", 0) & 0x1c000 == 0x14000
    if depthwise and cout != cin:
        return [], "unprobed depthwise channel multiplier"
    group_count = cin if depthwise else 1
    shape = (cout, cin // group_count, geo["kh"], geo["kw"])
    cfg = regs.get("1242", 0)
    bias = bool(cfg & 0x10)
    lut_bytes = 128 if cfg >> 16 & 3 == 2 else 0
    mode = "sparse" if regs.get("1240", 0) & 0x100 else "dense"
    lanes = min(cout, 16, max(group.indices) + 1)
    packing = [cout // lanes + (i < cout % lanes) for i in range(lanes)]
    by_lane = dict(zip(group.indices, group.symbols))
    first_symbol = min(s.value for s in group.symbols)
    origin = min(group.starts)
    weights, biases, tables, segments, addresses = [], [], [], [], []
    try:
        for lane, count in enumerate(packing):
            offset = origin + by_lane[lane].value - first_symbol
            start = offset
            addresses.append(start)
            if lut_bytes:
                tables.append(raw[offset:offset + lut_bytes])
                offset += lut_bytes
            if mode == "dense":
                size = sum(b[4] for b in block_layout((count, *shape[1:]), count, bias))
                w, b = dense_decode(raw[offset:offset + size], (count, *shape[1:]), count, bias)
                encoded = dense_encode(w, b, count)
            else:
                w, b, size = extended_decode(raw[offset:], count, shape[1:], bias)
                encoded = extended_encode(w, b)
            if encoded != raw[offset:offset + size]:
                raise ValueError("symbol-addressed native round-trip mismatch")
            segments.append([start, offset + size])
            weights.append(w)
            if b is not None:
                biases.append(b)
        if len(set(tables)) > 1:
            raise ValueError("different embedded tables across active lanes")
        w = np.concatenate(weights)
        if not bias and cout == 1:
            b = np.array([regs.get("1244", 0) & 0xffff], dtype=np.uint16).view(np.float16)
        else:
            b = np.concatenate(biases) if biases else None
        return [{"weights": w, "bias": b, "shape": shape, "packing": packing, "groups": group_count,
                 "bytes": max(end for _, end in segments) - origin, "mode": mode, "tables": tables[:1],
                 "block_starts": addresses, "segments": segments,
                 "primitive": "conv2d_then_lut" if tables else "conv2d"}], None
    except (ValueError, KeyError, IndexError) as error:
        return [], str(error)


recover_group = recover_group_linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=PORT / "catalog_ports")
    args = parser.parse_args()
    registry = baseline_registry()
    transposes = transpose_registry()
    sparse_transposes = sparse_transpose_registry()
    paddings = padding_registry()
    models = []
    for path in sorted((WORK / "ota_extract/llv_catalog").rglob("model.hwx")):
        ir = export(path)
        data, sections, symbols = parse_hwx(path)
        groups = kernel_groups(data, sections, symbols)
        _, raw = kernel_section(data, sections)
        model_id = path.parents[3].stem
        folder = args.out / model_id
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for group in groups:
            if group.kind != "ne":
                continue
            tasks = [t for t in ir["tasks"] if any(a["register_word"] == 0x1544 and a["bar"] == ir["kernel_bar"]
                      and a["offset"] == min(group.starts) for a in t["address_writes"])]
            row = {"group": group.stem, "task_references": [t["index"] for t in tasks], "status": "unresolved"}
            recovered = None
            known = registry.get(group.stem)
            if known and raw[min(group.starts):min(group.starts) + len(known["region"])] == known["region"]:
                recovered = known
                row.update(status="reused_exact_native_payload", source=known["source"], region_bytes=len(known["region"]),
                           payload_sha256=sha(known["region"]), primitive=known["primitive"])
            elif tasks and (padding := recover_padding(group, tasks[0], paddings)):
                recovered = padding
                row.update(status="controlled_padding_round_trip", source=padding["source"], primitive=padding["primitive"],
                           region_bytes=len(padding["region"]), payload_sha256=sha(padding["region"]),
                           padding=padding["padding"], padding_mode=padding["padding_mode"],
                           native_padding_reference_top_left=padding["native_padding_reference_top_left"],
                           activation_bounds=padding["activation_bounds"])
            elif tasks and (transposed := (recover_transpose(group, tasks[0], transposes)
                                           or recover_sparse_transpose(group, tasks[0], raw, sparse_transposes)
                                           or recover_expansion_2x2(group, tasks[0], raw))):
                recovered = transposed
                row.update(status="controlled_transpose_round_trip", source=transposed["source"],
                           primitive=transposed["primitive"], region_bytes=len(transposed["region"]),
                           payload_sha256=sha(transposed["region"]), pad_for_probe=transposed["pad_for_probe"],
                           native_origin_top_left=transposed.get("native_origin_top_left"),
                           native_origin_by_geometry_padding=transposed.get("native_origin_by_geometry_padding"),
                           operator_equivalence="native tile origin and graph placement pending")
                if transposed.get("tables"):
                    row["activation_lut_hex"] = transposed["tables"][0].hex()
                if transposed.get("segments"):
                    row["physical_segments"] = transposed["segments"]
            elif tasks:
                candidates, reason = recover_group(group, tasks[0], raw, groups)
                row["candidate_count"] = len(candidates)
                if len(candidates) == 1:
                    recovered = candidates[0]
                    row.update(status="layout_round_trip_candidate", shape=list(recovered["shape"]),
                               groups=recovered["groups"],
                               packing=list(recovered["packing"]), mode=recovered["mode"], primitive=recovered["primitive"],
                               region_bytes=recovered["bytes"], physical_block_addresses=recovered["block_starts"],
                               payload_sha256=sha(raw[min(group.starts):min(group.starts) + recovered["bytes"]]),
                               operator_equivalence="pending controlled compiler and graph arithmetic validation")
                    if recovered["tables"]:
                        row["activation_lut_hex"] = recovered["tables"][0].hex()
                else:
                    row["reason"] = reason
            else:
                row["reason"] = "no explicit coefficient reference"
            if recovered:
                dest = folder / group.stem
                dest.mkdir(exist_ok=True)
                np.save(dest / "weights.npy", recovered["weights"])
                if recovered["bias"] is not None:
                    np.save(dest / "bias.npy", recovered["bias"])
                else:
                    (dest / "bias.npy").unlink(missing_ok=True)
                row["weight_sha256"] = sha(recovered["weights"].tobytes())
                row["bias_sha256"] = sha(recovered["bias"].tobytes()) if recovered["bias"] is not None else None
            rows.append(row)
        summary = dict(collections.Counter(row["status"] for row in rows))
        report = {"model_id": model_id, "model_sha256": ir["model_sha256"], "kernel_bar": ir["kernel_bar"],
                  "task_count": ir["task_count"], "ports": ir["ports"], "groups": rows, "summary": summary,
                  "runtime_ready": False, "scope": "coefficient recovery only; original H19 inference permission denied on phone"}
        (folder / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
        models.append({k: report[k] for k in ("model_id", "model_sha256", "kernel_bar", "task_count", "summary", "runtime_ready")})
        print(model_id, summary, flush=True)
    (args.out / "recovery_summary.json").write_text(json.dumps({"models": models}, indent=2) + "\n")


if __name__ == "__main__":
    main()
