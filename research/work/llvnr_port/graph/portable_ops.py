#!/usr/bin/env python3
"""Recovered LLVNR arithmetic without hardware, compiler or firmware dependencies."""
from __future__ import annotations

import struct
import numpy as np


def fp32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fp16(bits):
    return np.array([bits & 65535], dtype=np.uint16).view(np.float16)[0].item()


def round_fp16(value, mode="rne"):
    """Explicit reference rounding; ANE PE uses nearest with ties away from zero.

    Keep IEEE nearest-even available for source-schedule equivalence. The target
    profile is separately validated by controlled arithmetic, not inferred from
    a recurrent model's output.
    """
    value = np.asarray(value, dtype=np.float32)
    rounded = value.astype(np.float16)
    if mode == "rne":
        return rounded
    if mode != "ane_rta":
        raise ValueError(f"unknown FP16 rounding profile {mode}")
    direction = np.copysign(np.float16(np.inf), value).astype(np.float16)
    away = np.nextafter(rounded, direction)
    nearest = rounded.astype(np.float32)
    alternative = away.astype(np.float32)
    ties = (np.abs(value-nearest) == np.abs(value-alternative)) & (np.abs(value) > np.abs(nearest))
    return np.where(ties, away, rounded).astype(np.float16)


def sigmoid(x):
    result = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def sigmoid_lut(x, table):
    """Interpolate the recovered 33-point ANE sigmoid coefficient table.

    The four leading FP16 values give saturation limits and values. Samples
    4..36 cover [-8, 8] at 0.5 intervals. Values 37..40 describe the two linear
    tails. Hardware interpolation rounding is checked separately on H13.
    """
    table = np.asarray(table, dtype=np.float32)
    if table.shape != (64,):
        raise ValueError("expected one 128-byte activation coefficient table")
    position = (np.clip(x, -8.0, 8.0) + 8.0) * 2.0
    index = np.minimum(position.astype(np.int32), 31)
    value = table[4 + index] + (position - index) * (table[5 + index] - table[4 + index])
    value = np.where(x < -8.0, x * table[37] + table[38], value)
    value = np.where(x > 8.0, x * table[39] + table[40], value)
    value = np.where(x < table[0], table[2], value)
    return np.where(x > table[1], table[3], value).astype(np.float32)


def sqrt_lut(x, table):
    """Exponent-normalized sqrt table, identified by a controlled MIL probe."""
    x = np.asarray(x, dtype=np.float32)
    if np.any(x < 0):
        raise ValueError("sqrt activation requires nonnegative input")
    table = np.asarray(table, dtype=np.float32)
    mantissa, exponent = np.frexp(x)
    normalized = np.ldexp(mantissa, exponent & 1)
    position = normalized * 16.0
    index = np.minimum(position.astype(np.int32), 31)
    value = table[4 + index] + (position - index) * (table[5 + index] - table[4 + index])
    return np.where(x == 0, 0, np.ldexp(value, exponent // 2)).astype(np.float32)


def conv2d(x, weights, bias, target, stride, pad_top_left, edge=False):
    cout, cin, kh, kw = weights.shape
    if x.shape[0] != cin or target[0] != cout:
        raise ValueError(f"conv channels {x.shape} / {weights.shape} / {target}")
    sy, sx = stride
    pt, pl = pad_top_left
    oh, ow = target[1:]
    pb = max(0, (oh - 1) * sy + kh - x.shape[1] - pt)
    pr = max(0, (ow - 1) * sx + kw - x.shape[2] - pl)
    padded = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), mode="edge" if edge else "constant")
    result = np.zeros(target, dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            taps = weights[:, :, ky, kx]
            if not np.any(taps):
                continue
            window = padded[:, ky:ky + oh * sy:sy, kx:kx + ow * sx:sx]
            product = taps @ window.reshape(cin, oh * ow)
            result += product.reshape(target)
    if bias is not None:
        result += bias[:, None, None]
    return result


def conv_transpose(x, weights, target, stride, pad_top_left):
    cin, cout, kh, kw = weights.shape
    if cin != x.shape[0] or cout != target[0]:
        raise ValueError(f"transpose channels {x.shape} / {weights.shape} / {target}")
    sy, sx = stride
    _, height, width = x.shape
    full = np.zeros((cout, (height - 1) * sy + kh, (width - 1) * sx + kw), dtype=np.float32)
    flat = x.reshape(cin, height * width)
    for ky in range(kh):
        for kx in range(kw):
            taps = weights[:, :, ky, kx]
            if not np.any(taps):
                continue
            product = (taps.T @ flat).reshape(cout, height, width)
            full[:, ky:ky + height * sy:sy, kx:kx + width * sx:sx] += product
    pt, pl = pad_top_left
    result = np.zeros(target, dtype=np.float32)
    cropped = full[:, pt:pt + target[1], pl:pl + target[2]]
    # The NE expansion can expose trailing staging cells that the consumer's
    # crop discards. Keep their fill explicit until hardware parity is checked.
    result[:, :cropped.shape[1], :cropped.shape[2]] = cropped
    return result


def pooling(x, target, geometry, mode, virtual_pad=0):
    kh, kw = geometry["kh"], geometry["kw"]
    sy, sx = geometry["stride_y"], geometry["stride_x"]
    pt, pl = geometry["pad_top"] + virtual_pad, geometry["pad_left"] + virtual_pad
    oh, ow = target[1:]
    pb = max(0, (oh - 1) * sy + kh - x.shape[1] - pt)
    pr = max(0, (ow - 1) * sx + kw - x.shape[2] - pl)
    fill = -np.inf if mode == 2 else 0.0
    padded = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), constant_values=fill)
    if mode == 2:
        result = np.full(target, -np.inf, dtype=np.float32)
    else:
        result = np.zeros(target, dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            window = padded[:, ky:ky + oh * sy:sy, kx:kx + ow * sx:sx]
            result = np.maximum(result, window) if mode == 2 else result + window
    if mode in (0, 1):
        if mode == 0 and (pt or pl or pb or pr):
            valid = np.pad(np.ones_like(x[:1]), ((0, 0), (pt, pb), (pl, pr)))
            count = np.zeros((1, oh, ow), dtype=np.float32)
            for ky in range(kh):
                for kx in range(kw):
                    count += valid[:, ky:ky + oh * sy:sy, kx:kx + ow * sx:sx]
            result /= count
        else:
            result /= kh * kw
    elif mode != 2:
        raise ValueError(f"unimplemented pool mode {mode}")
    return result


def pe(x, other, registers, task_type):
    cfg = registers.get(0x1140, 0)
    x = x * fp32(registers.get(0x1144, 0x3F800000))
    if task_type == 3:
        if cfg & 0xFFFFF == 0xC01C0:
            value = np.abs(x - other)
        else:
            operation = cfg >> 2 & 7
            if operation == 0:
                value = x + other
            elif operation == 1:
                value = x * other
            elif operation == 2:
                value = np.maximum(x, other)
            else:
                raise ValueError(f"unimplemented PE binary configuration {cfg:#x}")
    else:
        value = np.abs(x) if cfg & 0x1C0 == 0x1C0 else x
    value = (value + fp32(registers.get(0x1141, 0))) * fp32(registers.get(0x1142, 0x3F800000))
    condition = cfg >> 6 & 7
    if condition in (1,2,3,4,5,6):
        # Canonical numerical booleans. The native integer/float buffer casts
        # preserve these 0/1 values; condition inputs include the affine stage.
        value = {1: np.less, 2: np.greater, 3: np.not_equal, 4: np.equal,
                 5: np.less_equal, 6: np.greater_equal}[condition](value,0).astype(np.float32)
    if cfg & 0x20:
        xmin = fp32(registers.get(0x1146, 0xFF800000))
        xmax = fp32(registers.get(0x1147, 0x7F800000))
        ymin = fp32(registers.get(0x1148, 0xFF800000))
        ymax = fp32(registers.get(0x1149, 0x7F800000))
        slope = fp32(registers.get(0x114A, 0x3F800000))
        offset = fp32(registers.get(0x114B, 0))
        if registers.get(0x114A, 0x3F800000) != registers.get(0x114C, 0x3F800000) or registers.get(0x114B, 0) != registers.get(0x114D, 0):
            raise ValueError("unimplemented asymmetric PE linear branches")
        value = np.where(value < xmin, ymin, np.where(value > xmax, ymax, value * slope + offset))
    return value * fp32(registers.get(0x1145, 0x3F800000))
