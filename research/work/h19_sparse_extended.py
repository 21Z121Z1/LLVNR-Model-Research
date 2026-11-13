#!/usr/bin/env python3
"""Count-first H19 sparse lanes, including three output channels per lane.

The mask has a power-of-two number of output slots, with at least two slots.
Coefficient-free bias headers are omitted. Controlled compiler fixtures must
establish each shape/mode before recovered candidates are accepted.
"""
import numpy as np
from h19_sparse_codec import mask_tiles, tap_position


def tiles_for_shape(cin, kh, kw):
    return [(ci, 0, 0) for ci in range(0, cin, 4)] if (kh, kw) == (1, 1) else mask_tiles(cin, kh, kw)


def position(ci, by, bx, tap, kh, kw, parity):
    if (kh, kw) == (1, 1):
        return ci + tap, 0, 0
    ky, kx = tap_position(by, bx, tap, kh, kw) if parity else (by + tap // 2, bx + tap % 2)
    return ci, ky, kx


def decode(raw, packing, shape, has_bias, parity=True):
    cin, kh, kw = shape
    if packing < 1 or packing > 16 or kh not in (1, 2, 3) or kw != kh:
        raise ValueError("unprobed sparse shape")
    slots = max(2, 1 << (packing - 1).bit_length())
    mask_width = 4 * slots // 8
    tiles_per_mask = max(1, 2 // mask_width)
    tiles = tiles_for_shape(cin, kh, kw)
    mask_bytes = ((len(tiles) + tiles_per_mask - 1) // tiles_per_mask) * tiles_per_mask * mask_width
    prefix = 2 * packing if has_bias else 0
    if len(raw) < prefix + 3:
        raise ValueError("truncated sparse header")
    code = int.from_bytes(raw[prefix:prefix + 2], "little")
    used = prefix + 3 + code
    size = (used + 63) & ~63
    if len(raw) < size or code < mask_bytes or (code - mask_bytes) % 2 or raw[prefix + 2]:
        raise ValueError("incompatible sparse header")
    expected = (code - mask_bytes) // 2
    if expected > packing * cin * kh * kw:
        raise ValueError("sparse count exceeds source shape")
    weights = np.zeros((packing, cin, kh, kw), dtype=np.float16)
    bias = np.frombuffer(raw[:prefix], dtype="<f2").copy() if has_bias else None
    cursor, count = prefix + 3, 0
    for first in range(0, len(tiles), tiles_per_mask):
        if count == expected:
            break
        masks = raw[cursor:cursor + tiles_per_mask * mask_width]
        cursor += tiles_per_mask * mask_width
        for relative in range(tiles_per_mask):
            mask = int.from_bytes(masks[relative * mask_width:(relative + 1) * mask_width], "little")
            for tap in range(4):
                for output in range(slots):
                    if not mask & (1 << (tap * slots + output)):
                        continue
                    if output >= packing or first + relative >= len(tiles):
                        raise ValueError("sparse mask selected inactive lane")
                    ci, by, bx = tiles[first + relative]
                    ci, ky, kx = position(ci, by, bx, tap, kh, kw, parity)
                    if ci >= cin or ky >= kh or kx >= kw or count >= expected or cursor + 2 > used:
                        raise ValueError("sparse mask selected padded coefficient")
                    weights[output, ci, ky, kx] = np.frombuffer(raw[cursor:cursor + 2], dtype="<f2")[0]
                    cursor += 2
                    count += 1
    if count != expected or any(raw[cursor:size]) or not np.isfinite(weights).all():
        raise ValueError("sparse count, padding or coefficients invalid")
    if bias is not None and not np.isfinite(bias).all():
        raise ValueError("non-finite sparse bias")
    return weights, bias, size


def encode(weights, bias, parity=True):
    packing, cin, kh, kw = weights.shape
    slots = max(2, 1 << (packing - 1).bit_length())
    mask_width = 4 * slots // 8
    tiles_per_mask = max(1, 2 // mask_width)
    tiles = tiles_for_shape(cin, kh, kw)
    mask_bytes = ((len(tiles) + tiles_per_mask - 1) // tiles_per_mask) * tiles_per_mask * mask_width
    raw = bytearray(np.asarray(bias, dtype="<f2").tobytes() if bias is not None else b"")
    raw.extend(int(mask_bytes + 2 * np.count_nonzero(weights)).to_bytes(2, "little"))
    raw.append(0)
    consumed = 0
    for first in range(0, len(tiles), tiles_per_mask):
        if consumed == np.count_nonzero(weights):
            break
        masks, payload = bytearray(), bytearray()
        for relative in range(tiles_per_mask):
            mask = 0
            if first + relative < len(tiles):
                ci_base, by, bx = tiles[first + relative]
                for tap in range(4):
                    ci, ky, kx = position(ci_base, by, bx, tap, kh, kw, parity)
                    if ci >= cin or ky >= kh or kx >= kw:
                        continue
                    for output in range(packing):
                        value = weights[output, ci, ky, kx]
                        if value != 0:
                            mask |= 1 << (tap * slots + output)
                            payload.extend(np.asarray(value, dtype="<f2").tobytes())
                            consumed += 1
            masks.extend(mask.to_bytes(mask_width, "little"))
        raw.extend(masks)
        raw.extend(payload)
    return bytes(raw).ljust((len(raw) + 63) & ~63, b"\0")
