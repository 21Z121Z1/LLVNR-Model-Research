#!/usr/bin/env python3
"""Export the complete H19 task stream without discarding unresolved fields.

Hardware descriptors, candidate arithmetic, and verified portable operations
are separate fields. A retained register value alone does not establish that
the corresponding DMA stream or math unit participates in the current task.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import struct

from h19_statewalk import parse_macho, walk_tasks, words_at
from h19_weightlib import kernel_groups, parse_hwx

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = next((ROOT / "ota_extract/llv_model_2304").rglob("model.hwx"), None)


def mapped_regions(data):
    """Keep duplicate segment/section names and recover port allocation sizes."""
    regions, allocations = [], {}
    cursor = 32
    for _ in range(struct.unpack_from("<I", data, 16)[0]):
        command, size = struct.unpack_from("<II", data, cursor)
        if command == 0x19:
            address, vm_size = struct.unpack_from("<QQ", data, cursor + 24)
            section_count = struct.unpack_from("<I", data, cursor + 64)[0]
            section_sizes = [struct.unpack_from("<Q", data, cursor + 72 + i * 80 + 40)[0]
                             for i in range(section_count)]
            allocations[address] = {"vm_bytes": vm_size, "section_bytes": section_sizes}
        if command == 0x40:
            address = struct.unpack_from("<Q", data, cursor + 16)[0]
            end = data.index(b"\0", cursor + 24, cursor + size)
            regions.append({"name": data[cursor + 24:end].decode(), "vm_address": address})
        cursor += size
    for region in regions:
        region.update(allocations.get(region["vm_address"], {}))
    return regions


def float_bits(value):
    number = struct.unpack("<f", struct.pack("<I", value))[0]
    # JSON has no standard representation for NaN or infinity.
    if number == float("inf"):
        return "inf"
    if number == float("-inf"):
        return "-inf"
    if number != number:
        return "nan"
    return number


def export(model):
    data, sections, symbols = parse_hwx(model)
    _, cpu, subtype, ncmds, state_sections, ports = parse_macho(model)
    tasks = walk_tasks(words_at(data, state_sections, ("__TEXT", "__text")))
    if any(task.get("bad") for task in tasks):
        raise ValueError("malformed or unsupported task stream")
    groups = kernel_groups(data, sections, symbols) if ("__KERN_0", "__kern_0") in sections else []
    kernel_bars = {packet["bar"] for task in tasks for packet in task["address_packets"]
                   if packet["reg"] in (0x1544, 0x158E)}
    if len(kernel_bars) > 1 or (groups and not kernel_bars):
        raise ValueError(f"ambiguous kernel BAR bindings: {sorted(kernel_bars)}")
    kernel_bar = next(iter(kernel_bars), None)
    by_address = collections.defaultdict(list)
    kernel_section = sections.get(("__KERN_0", "__kern_0"))
    for symbol in symbols:
        if kernel_section is not None and symbol.type == 0x0F and symbol.section and symbol.name.startswith("K"):
            by_address[symbol.value - kernel_section.address].append(symbol.name)
    rows = []
    for task in tasks:
        written, effective = task["regs"], task["effective"]
        address_rows = []
        for packet in task["address_packets"]:
            low, *high = packet["values"]
            offset = low | ((high[0] if high else 0) << 32)
            address_rows.append({
                "register_word": packet["reg"], "bar": packet["bar"],
                "offset": offset, "packet_word_offset": packet["word_index"],
                "kernel_symbols_at_offset": by_address.get(offset, []) if packet["bar"] == kernel_bar else [],
            })
        conv = written.get(0xA)
        geometry = None if conv is None else {
            "kw": conv & 63, "kh": conv >> 6 & 63,
            "stride_x": conv >> 13 & 3, "stride_y": conv >> 15 & 3,
            "pad_left": conv >> 17 & 31, "pad_top": conv >> 22 & 31,
            "output_x_factor": conv >> 28 & 3, "output_y_factor": conv >> 30 & 3,
        }
        rows.append({
            "index": task["index"], "stream": task["stream_index"], "task_id": task["task_id"],
            "text_byte_offset": task["pos"] * 4, "bytes": task["size"] * 4,
            "header_words": task["header"],
            "task_type_code": written.get(0xF, 0) >> 4 & 15,
            "explicit_dimensions_whc": {
                "input": [written.get(addr) for addr in (1, 2, 3)],
                "output": [written.get(addr) for addr in (5, 6, 7)],
            },
            "geometry": geometry,
            "pe_explicit": {f"{addr:04x}": {"bits": value, "fp32": float_bits(value)}
                            for addr, value in written.items() if 0x1140 <= addr < 0x1150},
            "written_registers": {f"{addr:04x}": value for addr, value in written.items()},
            "carried_registers": {f"{addr:04x}": value for addr, value in effective.items()},
            "address_writes": address_rows,
            "packets": [{"register_word": addr, "header": header, "values": values}
                        for addr, header, values in task["packets"]],
            "portable_status": "unresolved",
        })
    return {
        "schema_version": 1,
        "model": str(model.resolve()), "model_sha256": hashlib.sha256(data).hexdigest(),
        "cpu_type": cpu, "cpu_subtype": subtype,
        "kernel_bar": kernel_bar,
        "task_count": len(tasks), "streams": len({task["stream_index"] for task in tasks}),
        "task_type_counts": dict(collections.Counter(row["task_type_code"] for row in rows)),
        "ports": mapped_regions(data),
        "kernel_group_counts": dict(collections.Counter(group.kind for group in groups)),
        "interpretation": {
            "missing_dimension": "unresolved, no inferred zero or inherited channel count",
            "carried_registers": "diagnostic register history, unit participation requires independent resolution",
            "kernel_regions": "symbol addresses may alias adjacent groups, group spans are diagnostic bounds",
        },
        "kernel_groups": [{
            "stem": group.stem, "kind": group.kind,
            "symbol_indices": list(group.indices),
            "symbol_offsets": [symbol.value - kernel_section.address for symbol in group.symbols],
            "unique_offsets": list(group.starts), "diagnostic_span_bytes": group.span,
            "diagnostic_span_sha256": hashlib.sha256(group.raw).hexdigest(),
        } for group in groups],
        "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "llvnr_port/artifacts/hardware_ir.json")
    args = parser.parse_args()
    if args.model is None:
        parser.error("provide the extracted model.hwx")
    report = export(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("model_sha256", "task_count", "streams", "task_type_counts", "kernel_group_counts")}, indent=2))


if __name__ == "__main__":
    main()
