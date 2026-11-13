#!/usr/bin/env python3
"""Read-only helpers for H19 HWX kernel metadata.

This module deliberately separates *metadata* from *semantic decoding*.  A
`K<hash>_ne_<n>` symbol group describes one packed kernel region; interpreting
its bytes as a dense fp16 tensor is only valid after the task's kernel mode
(including compression/sparsity) has been decoded.

No model bytes are written or published.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np

MAGIC = 0xBEEFFACE
KERNEL_SYMBOL_RE = re.compile(r"^K([0-9A-Fa-f]+)_(ne|actlut)(?:_(\d+))?$")


@dataclass(frozen=True)
class Section:
    segment: str
    name: str
    address: int
    size: int
    offset: int
    reloff: int
    nreloc: int


@dataclass(frozen=True)
class Symbol:
    name: str
    type: int
    section: int
    desc: int
    value: int


@dataclass(frozen=True)
class KernelGroup:
    stem: str
    kind: str
    symbols: tuple[Symbol, ...]
    indices: tuple[int, ...]
    starts: tuple[int, ...]
    span: int
    raw: bytes


def _cstr(data: bytes, offset: int) -> str:
    end = data.find(b"\0", offset)
    if end < 0:
        raise ValueError(f"unterminated string at 0x{offset:x}")
    return data[offset:end].decode("latin1")


def parse_hwx(path: str | pathlib.Path) -> tuple[bytes, dict[tuple[str, str], Section], list[Symbol]]:
    data = pathlib.Path(path).read_bytes()
    if len(data) < 32:
        raise ValueError("truncated Mach-O header")
    magic, cpu, subtype, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack_from("<8I", data)
    if magic != MAGIC:
        raise ValueError(f"bad HWX magic 0x{magic:08x}")
    if 32 + sizeofcmds > len(data):
        raise ValueError("load-command table exceeds file")

    sections: dict[tuple[str, str], Section] = {}
    symbols: list[Symbol] = []
    cursor = 32
    for command_index in range(ncmds):
        if cursor + 8 > len(data):
            raise ValueError(f"load command {command_index} is truncated")
        command, size = struct.unpack_from("<II", data, cursor)
        if size < 8 or cursor + size > len(data):
            raise ValueError(f"load command {command_index} has invalid size 0x{size:x}")
        if command == 0x19:  # LC_SEGMENT_64
            if size < 72:
                raise ValueError("segment command is truncated")
            seg = _cstr(data, cursor + 8)
            address, vm_size, fileoff, filesize, maxprot, initprot, nsects, segflags = struct.unpack_from(
                "<4Q4I", data, cursor + 24
            )
            section_cursor = cursor + 72
            if nsects * 80 > size - 72:
                raise ValueError(f"segment {seg} section table is truncated")
            for _ in range(nsects):
                sect = _cstr(data, section_cursor)
                sect_addr, sect_size, sect_off = struct.unpack_from("<QQI", data, section_cursor + 32)
                reloff, nreloc = struct.unpack_from("<2I", data, section_cursor + 56)
                sections[(seg, sect)] = Section(
                    seg, sect, sect_addr, sect_size, sect_off, reloff, nreloc
                )
                section_cursor += 80
        elif command == 0x2:  # LC_SYMTAB
            if size < 24:
                raise ValueError("symtab command is truncated")
            symoff, nsyms, stroff, strsize = struct.unpack_from("<4I", data, cursor + 8)
            if symoff + nsyms * 16 > len(data) or stroff + strsize > len(data):
                raise ValueError("symtab/stro strings exceed file")
            for index in range(nsyms):
                strx, ntype, nsect, ndesc, nvalue = struct.unpack_from(
                    "<IBBHQ", data, symoff + index * 16
                )
                if strx >= strsize:
                    raise ValueError(f"symbol {index} has invalid string index")
                name = _cstr(data, stroff + strx)
                symbols.append(Symbol(name, ntype, nsect, ndesc, nvalue))
        cursor += size
    return data, sections, symbols


def kernel_section(data: bytes, sections: dict[tuple[str, str], Section]) -> tuple[Section, bytes]:
    section = sections.get(("__KERN_0", "__kern_0"))
    if section is None:
        raise ValueError("HWX has no __KERN_0,__kern_0")
    end = section.offset + section.size
    if end > len(data):
        raise ValueError("kernel section exceeds file")
    return section, data[section.offset:end]


def kernel_groups(
    data: bytes,
    sections: dict[tuple[str, str], Section],
    symbols: Iterable[Symbol],
) -> list[KernelGroup]:
    section, raw = kernel_section(data, sections)
    grouped: dict[tuple[str, str], list[tuple[int, Symbol]]] = {}
    for symbol in symbols:
        match = KERNEL_SYMBOL_RE.match(symbol.name)
        if not match:
            continue
        if symbol.type != 0x0F:
            continue
        if not (section.address <= symbol.value < section.address + section.size):
            continue
        stem = f"K{match.group(1).upper()}"
        kind = match.group(2)
        if match.group(2) == "ne" and match.group(3) is None:
            continue
        index = int(match.group(3) or 0)
        grouped.setdefault((stem, kind), []).append((index, symbol))

    # The first unique symbol address marks the beginning of each region. A
    # hash can have duplicate addresses (several logical lanes share a physical
    # lane), so deduplicate addresses, not suffixes.
    all_starts = sorted({
        symbol.value - section.address
        for values in grouped.values()
        for _, symbol in values
    })
    next_start = {start: (all_starts[i + 1] if i + 1 < len(all_starts) else section.size)
                  for i, start in enumerate(all_starts)}
    result: list[KernelGroup] = []
    for (stem, kind), values in grouped.items():
        values.sort(key=lambda item: item[0])
        starts = tuple(sorted({symbol.value - section.address for _, symbol in values}))
        begin = starts[0]
        own = set(starts)
        end = next((start for start in all_starts if start > max(starts) and start not in own), section.size)
        result.append(KernelGroup(
            stem=stem,
            kind=kind,
            symbols=tuple(symbol for _, symbol in values),
            indices=tuple(index for index, _ in values),
            starts=starts,
            span=end - begin,
            raw=raw[begin:end],
        ))
    result.sort(key=lambda group: min(group.starts))
    return result


def raw_half_stats(raw: bytes) -> dict[str, object]:
    """Naive interpretation used only to detect non-plain-fp16 kernel modes."""
    if len(raw) % 2:
        return {"aligned": False, "halfwords": len(raw) // 2}
    values = np.frombuffer(raw, dtype="<f2")
    finite = np.isfinite(values)
    abs_values = np.abs(values[finite].astype(np.float32))
    return {
        "aligned": True,
        "halfwords": int(values.size),
        "nan": int(np.isnan(values).sum()),
        "inf": int(np.isinf(values).sum()),
        "nonzero": int(np.count_nonzero(values)),
        "max_finite_abs": float(abs_values.max()) if abs_values.size else 0.0,
        "large_finite_ge_64": int((abs_values >= 64.0).sum()),
    }


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
