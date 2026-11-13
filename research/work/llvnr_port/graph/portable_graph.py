#!/usr/bin/env python3
"""Whole-frame LLVNR source graph recovered from coefficients and task storage.

The JSON plan is portable operator source. Native task scheduling and the
whole-frame schedule can be compared without synthetic intermediate operands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

from portable_ops import conv2d, conv_transpose, fp16, fp32, pe, pooling, round_fp16, sigmoid_lut, sqrt_lut

WORK = pathlib.Path(__file__).resolve().parents[2]
PORT = WORK / "llvnr_port"


def source_plan(reference):
    nodes = []
    weights = reference.coefficients

    def add(name, op, inputs, **attrs):
        nodes.append({"name": name, "op": op, "inputs": list(inputs), **attrs})
        return name

    def conv(name, x, prefix, task, stride=1, pad=1):
        group = next(group for group in weights if group.startswith(prefix))
        cfg = reference.ir["tasks"][task]["written_registers"].get("1242", 0)
        activation = {0: "identity", 1: "relu", 2: "sigmoid"}[cfg >> 16 & 3]
        attrs = {"group": group, "stride": stride, "pad": pad, "edge": bool(cfg & 0x40000),
                 "activation": activation, "native_task": task}
        if activation == "sigmoid":
            attrs["activation_lut_hex"] = reference.activation_tables[group].tobytes().hex()
        return add(name, "conv2d", [x], **attrs)

    def deconv(name, x, prefix, pad):
        group = next(group for group in weights if group.startswith(prefix))
        return add(name, "conv_transpose", [x], group=group, stride=2, pad=pad)

    def pool(name, x, kernel=2, stride=2, mode=0, edge=False):
        return add(name, "pool", [x], kernel=kernel, stride=stride, mode=mode, edge=edge)

    def affine(name, x, alpha, beta):
        return add(name, "affine", [x], alpha=alpha, beta=beta)

    def unary_ne(name, x, task):
        r = reference.ir["tasks"][task]["written_registers"]
        lut_address = next(a for a in reference.ir["tasks"][task]["address_writes"] if a["register_word"] == 0x158E)
        lut_group = reference.groups_by_offset[lut_address["offset"]].stem
        lut = reference.luts[lut_group]
        return add(name, "affine_sigmoid", [x], bias=fp16(r.get("1244", 0)),
                   scale=fp16(r.get("1245", 0x3C00)), output_scale=lut["scale"], native_task=task,
                   activation_lut_hex=reference.activation_tables[lut_group].tobytes().hex())

    def binary(name, op, x, y):
        return add(name, op, [x, y])

    def channel(name, x, index):
        return add(name, "channel", [x], index=index)

    for name in ("his_frame", "cur_frame"):
        add(name, "input", [], shape=[1, 6, 648, 1152])
    for name in ("dnr_zoom_map", "fusion_zoom_map"):
        add(name, "input", [], shape=[1, 1, 324, 576])

    conv("his_low", "his_frame", "K3B459", 0, stride=2, pad=0)
    conv("cur_low", "cur_frame", "K3B459", 3, stride=2, pad=0)
    channel("his_low_y0", "his_low", 0)
    add("highlight", "clipped_affine", ["his_low_y0"], alpha=100.0, beta=-783.0, minimum=0.0, maximum=1.0)
    add("highlight_guard", "max_pool_same", ["highlight"], kernel=5)
    for prefix in ("his", "cur"):
        pool(prefix + "_mid", prefix + "_low")
        pool(prefix + "_coarse", prefix + "_mid")
    pool("fusion_mid", "fusion_zoom_map")
    for suffix in ("coarse", "mid", "fine"):
        if suffix == "coarse":
            pool("fusion_weight_coarse", "fusion_mid")
            nodes[-1].update(alpha=-0.239990234375, beta=1.1396484375)
        else:
            map_name = {"mid": "fusion_mid", "fine": "fusion_zoom_map"}[suffix]
            affine("fusion_weight_" + suffix, map_name, -0.239990234375, 1.1396484375)
    for suffix, current, history in (("coarse", "cur_coarse", "his_coarse"),
                                     ("mid", "cur_mid", "his_mid"), ("fine", "cur_low", "his_low")):
        binary("difference_" + suffix, "absdiff", current, history)
        binary("weighted_difference_" + suffix, "mul", "difference_" + suffix, "fusion_weight_" + suffix)
    add("motion_coarse_input", "concat", ["cur_coarse", "his_coarse", "weighted_difference_coarse"])
    conv("motion_coarse_1", "motion_coarse_input", "KE66FAB", 12)
    conv("motion_coarse_2", "motion_coarse_1", "KA66672", 13)
    conv("motion_coarse", "motion_coarse_2", "K2C1AE", 14)
    add("motion_coarse_up", "nearest2", ["motion_coarse"])
    add("motion_mid_input", "concat", ["cur_mid", "his_mid", "weighted_difference_mid", "motion_coarse_up"])
    conv("motion_mid_1", "motion_mid_input", "K45DFFF", 19)
    conv("motion_mid_2", "motion_mid_1", "KE66DB5", 20)
    conv("motion_mid", "motion_mid_2", "KA8702", 21)
    add("motion_mid_up", "nearest2", ["motion_mid"])
    add("motion_fine_input", "concat", ["cur_low", "his_low", "weighted_difference_fine", "motion_mid_up"])
    conv("motion_fine_1", "motion_fine_input", "KD3756", 28)
    conv("motion_fine_2", "motion_fine_1", "KCE150", 29)
    conv("motion_fine", "motion_fine_2", "K0DBDE", 30)
    unary_ne("temporal_sigmoid", "motion_fine", 49)
    affine("temporal_gain", "temporal_sigmoid", 1.0, fp32(reference.ir["tasks"][50]["written_registers"]["1141"]))
    binary("temporal_weight_raw", "mul", "motion_fine", "temporal_gain")
    add("fusion_negative_epsilon", "clip", ["fusion_zoom_map"], minimum=-1.0013580322265625e-5, maximum=0.0)
    binary("temporal_weight", "add", "temporal_weight_raw", "fusion_negative_epsilon")
    pool("temporal_smoothed", "temporal_weight", 3, 1, edge=True)
    unary_ne("temporal_smooth_gate", "temporal_smoothed", 56)
    binary("temporal_smooth_part", "mul", "temporal_smooth_gate", "temporal_smoothed")
    affine("temporal_raw_gate", "temporal_smooth_gate", -1.0, 1.0)
    binary("temporal_raw_part", "mul", "temporal_raw_gate", "temporal_weight")
    binary("temporal_mix", "add", "temporal_smooth_part", "temporal_raw_part")
    binary("temporal_guarded", "maximum", "temporal_mix", "highlight_guard")
    add("temporal_band0", "clip", ["temporal_guarded"], minimum=0.0999755859375, maximum=1.0)
    add("temporal_full", "nearest2", ["temporal_band0"])
    affine("history_weight", "temporal_full", -1.0, 1.0)
    binary("history_part", "mul", "his_frame", "history_weight")
    binary("current_part", "mul", "cur_frame", "temporal_full")
    binary("state_yuv", "add", "history_part", "current_part")

    conv("analysis", "state_yuv", "K730718", 112, stride=2, pad=0)
    for name, index in (("analysis_lh_y0", 6), ("analysis_hl_y0", 12),
                        ("analysis_c0", 4), ("analysis_c1", 5)):
        channel(name, "analysis", index)
    add("abs_lh_y0", "abs", ["analysis_lh_y0"])
    add("abs_hl_y0", "abs", ["analysis_hl_y0"])
    binary("detail_sum", "add_affine", "abs_lh_y0", "abs_hl_y0")
    nodes[-1].update(alpha=-100.0, beta=10.0)
    for name, src, task in (("c1_upper", "analysis_c1", 117), ("c1_lower", "analysis_c1", 118),
                            ("c0_upper", "analysis_c0", 120), ("c0_lower", "analysis_c0", 121)):
        unary_ne(name, src, task)
    binary("c1_window", "mul", "c1_upper", "c1_lower")
    binary("c0_window", "mul", "c0_upper", "c0_lower")
    binary("chroma_window", "mul", "c0_window", "c1_window")
    unary_ne("detail_gate", "detail_sum", 124)
    binary("detail_window", "mul", "chroma_window", "detail_gate")
    for name, src, task in (("c0_falloff", "analysis_c0", 126), ("c1_falloff", "analysis_c1", 128)):
        r = reference.ir["tasks"][task]["written_registers"]
        add(name, "abs_piecewise", [src], registers={key: value for key, value in r.items() if key.startswith("114")})
    binary("chroma_falloff", "mul", "c0_falloff", "c1_falloff")
    binary("noise_gate_raw", "maximum", "detail_window", "chroma_falloff")
    pool("noise_gate", "noise_gate_raw", 5, 1, edge=True)
    affine("noise_inverse", "noise_gate", -1.0, 1.0)
    binary("state_weight_base", "mul", "noise_inverse", "temporal_band0")
    affine("state_weight_modulation", "temporal_band0", 5.0, -4.0)
    binary("state_weight_correction", "mul", "noise_gate", "state_weight_modulation")
    binary("state_weight_band0", "add", "state_weight_correction", "state_weight_base")
    affine("dnr_feature", "dnr_zoom_map", 1.412109375, -0.2353515625)
    add("encoder_input", "concat", ["analysis", "dnr_feature"])
    conv("encoder_1", "encoder_input", "K3483", 132, stride=2)
    conv("encoder_2", "encoder_1", "KCFE7", 289, stride=2)
    conv("bottleneck_1", "encoder_2", "KFA1693", 290)
    conv("bottleneck_2", "bottleneck_1", "K7C31B2", 291)
    conv("bottleneck_3", "bottleneck_2", "K1A3985", 292)
    deconv("decoder_2_up", "bottleneck_3", "KB358", 1)
    add("decoder_2_input", "concat", ["encoder_1", "decoder_2_up"])
    conv("decoder_2_merge", "decoder_2_input", "K7F28", 294, pad=0)
    conv("decoder_2_1", "decoder_2_merge", "K6F687", 295)
    conv("decoder_2_2", "decoder_2_1", "KB6F8", 296)
    deconv("decoder_1_up", "decoder_2_2", "KD5A651", 1)
    add("decoder_1_input", "concat", ["analysis", "decoder_1_up"])
    conv("decoder_1_merge", "decoder_1_input", "K927A35", 299, pad=0)
    conv("decoder_1", "decoder_1_merge", "KF74DEC", 300)
    conv("pre_synthesis", "decoder_1", "K314425", 370)
    deconv("synthesis", "pre_synthesis", "KE075D", 0)
    add("output_refine_input", "concat", ["state_yuv", "synthesis"])
    conv("output_refine_1", "output_refine_input", "KD6494", 372, pad=0)
    conv("output_refine_2", "output_refine_1", "K94C099", 373)
    conv("out_yuv", "output_refine_2", "K76C518", 534)
    files = {}
    used = {node["group"] for node in nodes if "group" in node}
    for group in sorted(used):
        weight_path = next((PORT / folder / group / "weights.npy" for folder in
                            ("dense_weights", "special_weights", "deconv_weights")
                            if (PORT / folder / group / "weights.npy").is_file()))
        bias_path = weight_path.with_name("bias.npy")
        files[group] = {"weights": str(weight_path.relative_to(PORT)),
                        "bias": str(bias_path.relative_to(PORT)) if bias_path.exists() else None,
                        "weights_sha256_fp16": hashlib.sha256(np.load(weight_path).tobytes()).hexdigest(),
                        "bias_sha256_fp16": hashlib.sha256(np.load(bias_path).tobytes()).hexdigest() if bias_path.exists() else None}
    return {"schema_version": 1, "model_sha256": reference.ir["model_sha256"], "coefficients": files,
            "execution_profile": "metal_state_ane_denoiser",
            "nodes": nodes, "outputs": ["out_yuv", "state_yuv", "state_weight_band0"],
            "validation_status": "candidate whole-frame graph, compare with native task schedule and H13 before acceptance"}


def load_coefficients(plan, source_root=PORT):
    result = {}
    for group, row in plan["coefficients"].items():
        weights = np.load(source_root / row["weights"])
        bias = np.load(source_root / row["bias"]) if row["bias"] else None
        if hashlib.sha256(weights.tobytes()).hexdigest() != row["weights_sha256_fp16"]:
            raise ValueError(f"source weight hash mismatch for {group}")
        if bias is not None and hashlib.sha256(bias.tobytes()).hexdigest() != row["bias_sha256_fp16"]:
            raise ValueError(f"source bias hash mismatch for {group}")
        result[group] = {"weights": weights.astype(np.float32), "bias": bias.astype(np.float32) if bias is not None else None}
    return result


def checked_inputs(plan, inputs):
    expected = {node["name"]: tuple(node["shape"]) for node in plan["nodes"] if node["op"] == "input"}
    if set(inputs) != set(expected):
        raise ValueError(f"expected external inputs {sorted(expected)}, got {sorted(inputs)}")
    result = {}
    for name, shape in expected.items():
        value = np.asarray(inputs[name], dtype=np.float16)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid input {name}: expected finite FP16 {shape}, got {value.shape}")
        result[name] = value
    return result


def evaluate_cpu(plan, coefficients, inputs, rounding="rne"):
    inputs = checked_inputs(plan, inputs)
    values = {}
    for node in plan["nodes"]:
        name, op = node["name"], node["op"]
        args = [values[key].astype(np.float32) for key in node["inputs"]]
        if op == "input":
            value = np.asarray(inputs[name], dtype=np.float32)
        elif op in ("conv2d", "conv_transpose"):
            entry = coefficients[node["group"]]
            weights, bias = entry["weights"], entry["bias"]
            x, stride, pad = args[0][0], node["stride"], node["pad"]
            if op == "conv2d":
                target = (weights.shape[0], (x.shape[1] + 2 * pad - weights.shape[2]) // stride + 1,
                          (x.shape[2] + 2 * pad - weights.shape[3]) // stride + 1)
                if node.get("output_rounding") == "floor_stride":
                    if stride != 2 or pad != 1 or weights.shape[2:] != (3, 3):
                        raise ValueError("floor-stride contract is only established for 3x3/pad1/stride2")
                    target = (target[0], x.shape[1] // stride, x.shape[2] // stride)
                groups = node.get("groups", 1)
                ci, co = x.shape[0] // groups, target[0] // groups
                value = np.concatenate([conv2d(x[g*ci:(g+1)*ci], weights[g*co:(g+1)*co],
                    bias[g*co:(g+1)*co] if bias is not None else None, (co, *target[1:]),
                    (stride, stride), (pad, pad), node["edge"]) for g in range(groups)])
            else:
                extra = node.get("output_padding", 0)
                ey, ex = (extra, extra) if isinstance(extra, int) else extra
                target = (weights.shape[1], (x.shape[1] - 1) * stride + weights.shape[2] - 2 * pad + ey,
                          (x.shape[2] - 1) * stride + weights.shape[3] - 2 * pad + ex)
                value = conv_transpose(x, weights, target, (stride, stride), (pad, pad))
                if bias is not None:
                    value += bias[:, None, None]
            if node.get("activation") == "relu":
                value = np.maximum(value, 0)
            elif node.get("activation") == "sigmoid":
                table = np.frombuffer(bytes.fromhex(node["activation_lut_hex"]), dtype="<f2")
                value = sigmoid_lut(value, table)
            elif node.get("activation") == "clip":
                value = np.clip(value, *node["activation_bounds"])
            value = value[None]
        elif op in ("pool", "max_pool_same"):
            x = args[0][0]
            kernel = node["kernel"]
            stride = node.get("stride", 1)
            border = kernel // 2 if node.get("edge") or op == "max_pool_same" else 0
            if border:
                x = np.pad(x, ((0, 0), (border, border), (border, border)),
                           mode="edge" if node.get("edge") else "constant")
            target = (x.shape[0], (x.shape[1] - kernel) // stride + 1, (x.shape[2] - kernel) // stride + 1)
            value = pooling(x, target, {"kh": kernel, "kw": kernel, "stride_y": stride, "stride_x": stride,
                                       "pad_top": 0, "pad_left": 0}, node.get("mode", 2))[None]
            if "alpha" in node:
                value = value * node["alpha"] + node["beta"]
        elif op == "concat":
            value = np.concatenate(args, axis=1)
        elif op == "match_spatial":
            height, width = args[1].shape[2:]
            value = np.pad(args[0], ((0, 0), (0, 0), (0, max(0, height - args[0].shape[2])),
                                                    (0, max(0, width - args[0].shape[3]))), mode="edge")
            value = value[:, :, :height, :width]
        elif op == "replicate_pad":
            top, bottom, left, right = node["padding"]
            value = np.pad(args[0], ((0, 0), (0, 0), (top, bottom), (left, right)), mode="edge")
        elif op == "channel":
            value = args[0][:, node["index"]:node["index"] + 1]
        elif op == "spatial_mean":
            value = args[0].sum(axis=(2,3),keepdims=True,dtype=np.float32)*node["scale"]
        elif op == "affine_compare":
            r = {int(k,16):v for k,v in node["registers"].items()}
            value = pe(args[0],None,r,2)
        elif op == "broadcast_affine":
            value = args[0]*np.asarray(node["channel_scale"],np.float32).reshape(1,-1,1,1)+np.asarray(node["channel_bias"],np.float32).reshape(1,-1,1,1)
        elif op == "nearest2":
            value = np.repeat(np.repeat(args[0], 2, axis=2), 2, axis=3)
        elif op == "affine":
            value = args[0] * node["alpha"] + node["beta"]
        elif op == "clipped_affine":
            value = np.clip(args[0] * node["alpha"] + node["beta"], node["minimum"], node["maximum"])
        elif op == "affine_sigmoid":
            table = np.frombuffer(bytes.fromhex(node["activation_lut_hex"]), dtype="<f2")
            value = sigmoid_lut((args[0] + node["bias"]) * node["scale"], table)
        elif op == "sqrt":
            table = np.frombuffer(bytes.fromhex(node["activation_lut_hex"]), dtype="<f2")
            value = sqrt_lut(args[0], table)
        elif op == "abs_piecewise":
            registers = {int(key, 16): value for key, value in node["registers"].items()}
            value = pe(args[0], None, registers, 2)
        elif op == "clip":
            value = np.clip(args[0], node["minimum"], node["maximum"])
        elif op == "abs":
            value = np.abs(args[0])
        elif op == "absdiff":
            value = np.abs(args[0] - args[1])
        elif op == "add_affine":
            value = (args[0] + args[1]) * node["alpha"] + node["beta"]
        elif op == "add":
            value = args[0] + args[1]
        elif op == "mul":
            value = args[0] * args[1]
        elif op == "mul_scaled":
            value = (args[0] * node["scale"]) * args[1]
        elif op == "maximum":
            value = np.maximum(args[0], args[1])
        else:
            raise ValueError(f"unimplemented portable operator {op}")
        rounded = round_fp16(value, rounding)
        if not np.isfinite(rounded).all():
            raise ValueError(f"non-finite portable node {name}")
        values[name] = rounded
    return values


def build_ane(plan, coefficients):
    sys.path.insert(0, str(WORK / "ANEForge"))
    import aneforge as af
    from aneforge._compile import op
    from aneforge.graph import Tensor

    @op("llvnr_conv_native_bias")
    def emit_native_bias(emitter, tensor, name, sources):
        from aneforge._compile import _emit_conv_params
        a=tensor.attrs
        weight=emitter.weight(f"{name}_w",a["weight"],allow_int8=False)
        bias=emitter.weight(f"{name}_bias",a["bias"],allow_int8=False)
        _emit_conv_params(emitter,name,a["pad"],a["stride"],1,a["groups"])
        emitter.line(f'{emitter.ty(tensor.shape)} {name} = conv(bias={bias}, dilations={name}_dl, groups={name}_g, pad={name}_pd, pad_type={name}_pt, strides={name}_st, weight={weight}, x={sources[0]})[name=string("{name}")];')

    @op("llvnr_replicate_pad")
    def emit_replicate_pad(emitter, tensor, name, sources):
        top, bottom, left, right = tensor.attrs["padding"]
        padding = [0, 0, 0, 0, top, bottom, left, right]
        emitter.line(f'tensor<int32, [8]> {name}_padding = const()[name=string("{name}_padding"), val=tensor<int32, [8]>({padding})];')
        emitter.line(f'{emitter.ty(tensor.shape)} {name} = pad(x={sources[0]}, pad={name}_padding, mode=string("replicate"), constant_val=fp16(0.0))[name=string("{name}")];')

    @op("llvnr_transpose_extra")
    def emit_transpose_extra(emitter, tensor, name, sources):
        attrs = tensor.attrs
        w = emitter.weight(f"{name}_w", attrs["weight"], allow_int8=False)
        p, extra = attrs["pad"], attrs["output_padding"]
        ey, ex = (extra, extra) if isinstance(extra, int) else extra
        padding = [p, p - ey, p, p - ex]
        if min(padding) < 0:
            raise ValueError("unsupported transpose output padding")
        stride = attrs["stride"]
        emitter.line(f'tensor<int32, [4]> {name}_pd = const()[name=string("{name}_pd"), val=tensor<int32, [4]>({padding})];')
        emitter.line(f'tensor<int32, [2]> {name}_st = const()[name=string("{name}_st"), val=tensor<int32, [2]>([{stride}, {stride}])];')
        emitter.line(f'{emitter.ty(tensor.shape)} {name} = conv_transpose(dilations=tensor<int32, [2]>([1, 1]), groups=int32(1), pad={name}_pd, pad_type=string("custom"), strides={name}_st, weight={w}, x={sources[0]})[name=string("{name}")];')
        if attrs.get("bias") is not None:
            b = emitter.weight(f"{name}_bw", attrs["bias"].reshape(1, tensor.shape[1], 1, 1), allow_int8=False)
            emitter.line(f'{emitter.ty(tensor.shape)} {name}_b = add(x={name}, y={b})[name=string("{name}_b")];')
            tensor._name = f"{name}_b"

    def padded(tensor, border):
        n, c, h, w = tensor.shape
        top, bottom, left, right = (border,) * 4 if isinstance(border, int) else border
        return Tensor((n, c, h + top + bottom, w + left + right), "llvnr_replicate_pad", [tensor],
                      {"padding": [top, bottom, left, right]})

    values = {}
    for node in plan["nodes"]:
        name, kind = node["name"], node["op"]
        args = [values[key] for key in node["inputs"]]
        if kind == "input":
            value = af.input(tuple(node["shape"]))
        elif kind in ("conv2d", "conv_transpose"):
            entry = coefficients[node["group"]]
            weights = entry["weights"].astype(np.float16)
            bias = entry["bias"].astype(np.float16) if entry["bias"] is not None else None
            x, pad = args[0], node["pad"]
            if kind == "conv_transpose":
                value = af.conv_transpose(x, weights, stride=node["stride"], pad=pad, bias=bias)
                if node.get("output_padding"):
                    n, c, h, w = value.shape
                    extra = node["output_padding"]
                    ey, ex = (extra, extra) if isinstance(extra, int) else extra
                    value = Tensor((n, c, h + ey, w + ex), "llvnr_transpose_extra", [x],
                                   {"weight": weights, "bias": bias, "stride": node["stride"], "pad": pad, "output_padding": extra})
            else:
                if node["edge"] and pad:
                    x, pad = padded(x, pad), 0
                value = af.conv(x, weights, stride=node["stride"], pad=pad, bias=bias, groups=node.get("groups", 1))
                if plan.get("ane_bias_mode") == "fused" and bias is not None:
                    value = Tensor(value.shape,"llvnr_conv_native_bias",[x],
                                   {"weight":weights,"bias":bias,"stride":node["stride"],"pad":pad,"groups":node.get("groups",1)})
                if node.get("output_rounding") == "floor_stride":
                    height, width = args[0].shape[2] // node["stride"], args[0].shape[3] // node["stride"]
                    value = af.crop(value, 0, value.shape[2] - height, 0, value.shape[3] - width)
            if node.get("activation") == "relu":
                value = value.relu()
            elif node.get("activation") == "sigmoid":
                value = value.sigmoid()
            elif node.get("activation") == "clip":
                value = value.clip(*node["activation_bounds"])
        elif kind == "pool":
            x = padded(args[0], node["kernel"] // 2) if node["edge"] else args[0]
            value = (x.max_pool(node["kernel"], node["stride"]) if node.get("mode", 0) == 2
                     else x.avg_pool(node["kernel"], node["stride"], exclude_pad=True))
            if "alpha" in node:
                value = value.linear_activation(node["alpha"], node["beta"])
        elif kind == "max_pool_same":
            value = args[0].max_pool(node["kernel"], 1, node["kernel"] // 2)
        elif kind == "concat":
            value = af.concat(args, axis=1)
        elif kind == "match_spatial":
            height, width = args[1].shape[2:]
            value = args[0]
            if value.shape[2] < height or value.shape[3] < width:
                value = padded(value, (0, max(0, height - value.shape[2]), 0, max(0, width - value.shape[3])))
            if value.shape[2] > height or value.shape[3] > width:
                value = af.crop(value, 0, value.shape[2] - height, 0, value.shape[3] - width)
        elif kind == "replicate_pad":
            value = padded(args[0], node["padding"])
        elif kind == "channel":
            value = af.split(args[0], args[0].shape[1], axis=1)[node["index"]]
        elif kind == "spatial_mean":
            value = args[0].mean((2,3))
        elif kind == "affine_compare":
            from aneforge.graph import _const
            r = {int(k,16):v for k,v in node["registers"].items()}
            if fp32(r.get(0x1142,0x3f800000)) <= 0 or r.get(0x1144,0x3f800000)!=0x3f800000:
                raise ValueError("unprobed affine comparison scale")
            threshold = -fp32(r.get(0x1141,0))
            if float(np.float16(threshold)) != threshold:
                raise ValueError("comparison boundary is not FP16-exact")
            rhs = _const(np.full(args[0].shape,threshold,np.float16))
            operation = {1:"less",2:"greater",3:"not_equal",4:"equal",5:"less_equal",6:"greater_equal"}[(r[0x1140]>>6)&7]
            condition = getattr(args[0],operation)(rhs)
            value = af.select(condition,_const(np.ones(condition.shape,np.float16)),_const(np.zeros(condition.shape,np.float16)))
        elif kind == "broadcast_affine":
            from aneforge.graph import _const
            scale = _const(np.asarray(node["channel_scale"],np.float16).reshape(1,-1,1,1))
            bias = _const(np.asarray(node["channel_bias"],np.float16).reshape(1,-1,1,1))
            value = args[0]*scale+bias
        elif kind == "nearest2":
            value = args[0].upsample(2)
        elif kind == "affine":
            value = args[0].linear_activation(node["alpha"], node["beta"])
        elif kind == "clipped_affine":
            # ReLU at an exactly representable anchor preserves this clamped
            # function and prevents NE bias quantization of -783/100.
            threshold = -node["beta"] / node["alpha"]
            anchor = np.float16(threshold)
            if anchor > threshold:
                anchor = np.nextafter(anchor, np.float16(-np.inf))
            anchor = float(anchor)
            value = args[0].linear_activation(1.0, -anchor).relu()
            value = value.linear_activation(node["alpha"], node["beta"] + anchor * node["alpha"])
            value = value.clip(node["minimum"], node["maximum"])
        elif kind == "affine_sigmoid":
            value = args[0].linear_activation(node["scale"], node["bias"] * node["scale"]).sigmoid() * node["output_scale"]
        elif kind == "sqrt":
            value = args[0].sqrt()
        elif kind == "abs_piecewise":
            r = {int(key, 16): value for key, value in node["registers"].items()}
            value = args[0].abs().adds(fp32(r[0x1141])) * fp32(r[0x1142])
            value = value.clip(0.0, 1.0).linear_activation(-1.0, 1.0)
        elif kind == "clip":
            value = args[0].clip(node["minimum"], node["maximum"])
        elif kind == "abs":
            value = args[0].abs()
        elif kind == "absdiff":
            value = (args[0] - args[1]).abs()
        elif kind == "add_affine":
            value = (args[0] + args[1]).linear_activation(node["alpha"], node["beta"])
        elif kind == "add":
            value = args[0] + args[1]
        elif kind == "mul":
            value = args[0] * args[1]
        elif kind == "mul_scaled":
            value = args[0].linear_activation(node["scale"], 0.0) * args[1]
        elif kind == "maximum":
            value = af.maximum(args[0], args[1])
        else:
            raise ValueError(f"unimplemented ANE operator {kind}")
        values[name] = value
    return values


class H13Model:
    """Compile one whole-frame source graph and reuse it across video frames."""

    def __init__(self, plan, coefficients, folder):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("H13 execution requires the Apple Silicon Mac")
        self.plan = plan
        self.precision = None
        self.body = None
        if plan.get("execution_profile") == "metal_state_ane_denoiser":
            from metal_prefix import export_prefix
            from precision_runtime import MetalPrefix
            started=time.perf_counter()
            _,body_plan=export_prefix(plan,coefficients,pathlib.Path(folder)/"precision")
            body_plan.pop("execution_profile",None)
            self.precision=MetalPrefix(pathlib.Path(folder)/"precision")
            try:self.body=H13Model(body_plan,coefficients,pathlib.Path(folder)/"denoiser")
            except Exception:
                self.precision.close()
                raise
            self.compile_seconds=time.perf_counter()-started
            return
        self.graph = build_ane(plan, coefficients)
        from aneforge._compile import compile_multi
        started = time.perf_counter()
        self.net = compile_multi([self.graph[key] for key in plan["outputs"]], build_dir=folder)
        self.compile_seconds = time.perf_counter() - started

    def run(self, inputs):
        inputs = checked_inputs(self.plan, inputs)
        if self.precision is not None:
            intermediate=self.precision.run(inputs)
            feeds={**inputs,**intermediate}
            outputs=self.body.run({n["name"]:feeds[n["name"]] for n in self.body.plan["nodes"] if n["op"]=="input"})
            return {name:outputs.get(name,intermediate.get(name)) for name in self.plan["outputs"]}
        feeds = {self.graph[key]._name: value for key, value in inputs.items()}
        result = self.net(*[feeds[name] for _, name in self.net.input_ports])
        return {key: np.asarray(result[port], dtype=np.float16).copy()
                for key, (_, port) in zip(self.plan["outputs"], self.net.output_ports)}

    def close(self):
        if self.precision is not None:
            self.body.close();self.precision.close()
            return
        self.net.release()


def execute_h13(plan, coefficients, inputs, expected, folder):
    model = H13Model(plan, coefficients, folder)
    calls, metrics, result_arrays = [], {}, {}
    try:
        for iteration in range(3):
            started = time.perf_counter()
            result = model.run(inputs)
            calls.append(time.perf_counter() - started)
        for key in plan["outputs"]:
            actual = result[key]
            result_arrays[key] = actual
            metrics[key] = compare(actual, expected[key])
            metrics[key]["finite"] = bool(np.isfinite(actual).all())
            metrics[key]["passed"] = metrics[key]["finite"] and metrics[key]["max_abs"] <= 0.02 and metrics[key]["mean_abs"] <= 0.001
    finally:
        model.close()
    np.savez_compressed(folder / "outputs.npz", **result_arrays)
    report = {"compile_seconds": model.compile_seconds, "host_calls_seconds": calls, "outputs": metrics,
              "passed": all(row["passed"] for row in metrics.values()),
              "numeric_gate": {"max_abs": 0.02, "mean_abs": 0.001, "finite": True},
              "scope": "whole source graph H13 versus CPU, native H19 hardware parity pending"}
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(actual, expected):
    if actual.shape != expected.shape:
        return {"shape_equal": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {"shape_equal": True, "max_abs": float(error.max()), "mean_abs": float(error.mean()),
            "exact_fraction": float(np.mean(actual.view(np.uint16) == expected.view(np.uint16))),
            "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}


def main():
    from native_reference import NativeReference, synthetic_inputs

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=PORT / "graph/portable")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--h13", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = NativeReference()
    plan = source_plan(reference)
    (args.out / "source_graph.json").write_text(json.dumps(plan, indent=2) + "\n")
    inputs = synthetic_inputs(args.seed)
    started = time.perf_counter()
    coefficients = load_coefficients(plan)
    values = evaluate_cpu(plan, coefficients, inputs)
    portable_seconds = time.perf_counter() - started
    capture = [14, 21, 49, 50, 51, 53, 55, 56, 60, 61, 288, 289, 290, 291, 292, 293, 294, 295, 296, 534]
    native = reference.run(inputs, capture_tasks=capture)
    direct = {14: "motion_coarse", 21: "motion_mid", 49: "temporal_sigmoid", 50: "temporal_gain",
              51: "temporal_weight_raw", 53: "temporal_weight", 55: "temporal_smoothed", 56: "temporal_smooth_gate",
              60: "temporal_mix", 61: "temporal_band0", 288: "state_weight_band0", 289: "encoder_2",
              290: "bottleneck_1", 291: "bottleneck_2", 292: "bottleneck_3", 293: "decoder_2_up",
              294: "decoder_2_merge", 295: "decoder_2_1", 296: "decoder_2_2", 534: "out_yuv"}
    metrics = {name: compare(values[name], reference.captures[task][None]) for task, name in direct.items()}
    metrics["state_yuv"] = compare(values["state_yuv"], native["state_yuv"])
    report = {"nodes": len(plan["nodes"]), "portable_cpu_seconds": portable_seconds,
              "native_tasks_executed": len(reference.trace), "comparisons": metrics,
              "scope": "whole-frame versus native task schedule, shared coefficient arithmetic, H19 hardware parity pending"}
    (args.out / "schedule_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.out / "outputs.npz", **{key: values[key] for key in plan["outputs"]})
    print(json.dumps(report, indent=2), flush=True)
    if args.h13:
        result = execute_h13(plan, load_coefficients(plan), inputs, values, args.out / "h13")
        print(json.dumps({"h13": result}, indent=2), flush=True)
        return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
