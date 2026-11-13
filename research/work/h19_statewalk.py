#!/usr/bin/env python3
"""Stateful v24 TD walker for the extracted H19 LLVNR model.

This parser follows the documented TD format:
  * 4-word stream header, then TDs.
  * each TD has a 9-word task header;
  * packets are continuous writes (bit 31 = 0) or masked writes (bit 31 = 1);
  * address packets are continuous writes with bit 29 set;
  * explicit register writes are retained independently from a diagnostic
    register history. An omitted word can encode the unit's default, so that
    history alone must not determine a task's arithmetic or tensor addresses.

The output is deliberately diagnostic: no model bytes are written.
"""
import argparse
import collections
import struct


def cstr(data, offset):
    end = data.find(b"\0", offset)
    return data[offset:end].decode("latin1")


def parse_macho(path):
    b = open(path, "rb").read()
    magic, cpu, subtype, ftype, ncmds, sizeofcmds, flags, _ = struct.unpack_from("<8I", b, 0)
    if magic != 0xBEEFFACE:
        raise SystemExit(f"bad magic {magic:#x}")
    o = 32
    sections = {}
    ports = []
    for index in range(ncmds):
        cmd, size = struct.unpack_from("<II", b, o)
        if cmd == 0x19:
            seg = cstr(b, o + 8)
            vmaddr, vmsize, fileoff, filesize, maxp, initp, nsects, fl = struct.unpack_from("<4Q4I", b, o + 24)
            so = o + 72
            for _ in range(nsects):
                sect = cstr(b, so)
                addr, ssize, soff = struct.unpack_from("<QQI", b, so + 32)
                sections[(seg, sect)] = (addr, ssize, soff)
                so += 80
        elif cmd == 0x40:
            # Newer HWX variants use LC_0x40 port records.  Older H19
            # products keep the same information in LC_THREAD strings.
            vm = struct.unpack_from("<Q", b, o + 16)[0]
            ports.append((vm, cstr(b, o + 24)))
        o += size
    return b, cpu, subtype, ncmds, sections, ports


def words_at(b, sections, key):
    addr, size, off = sections[key]
    return list(struct.unpack_from(f"<{size // 4}I", b, off))


def walk_tasks(w):
    """Return task dictionaries from a v24 __text stream."""
    pos = 0
    regs = {}
    addresses = {}
    tasks = []
    stream_index = -1
    while pos < len(w):
        while pos < len(w) and w[pos] == 0:
            pos += 1
        if pos >= len(w):
            break
        if w[pos:pos + 4] == [1, 0, 0, 0]:
            # Compiler probes can contain several page-aligned entry streams.
            # Register inheritance applies within one stream only.
            stream_index += 1
            regs = {}
            addresses = {}
            pos += 4
            continue
        h0 = w[pos]
        n = (h0 >> 16) & 0x7FF
        if not n or n > 0x7FF or pos + n > len(w):
            # Keep a bounded diagnostic instead of guessing past the stream.
            tasks.append({"header": h0, "size": n, "pos": pos, "bad": True})
            break
        td = w[pos:pos + n]
        task = {
            "index": len(tasks),
            "stream_index": max(stream_index, 0),
            "task_id": h0 & 0xFFFF,
            "pos": pos,
            "size": n,
            "header": td[:9],
            "packets": [],
            "address_packets": [],
            "regs": {},
            "first_regs": {},
            "effective": {},
            "bad": False,
        }
        i = 9
        while i < len(td):
            h = td[i]
            a = h & 0x7FFF
            f = (h >> 15) & 0xFFFF
            if h >> 31:
                addrs = [a] + [a + 1 + k for k in range(16) if (f >> k) & 1]
            else:
                count = (f & 0x3F) + 1
                addrs = [a + k for k in range(count)]
            vals = td[i + 1:i + 1 + len(addrs)]
            if len(vals) != len(addrs):
                task["bad"] = True
                break
            if not (h >> 31) and ((h >> 29) & 1):
                address = {
                    "word_index": i,
                    "reg": a,
                    "bar": (h >> 21) & 0xFF,
                    "values": vals,
                    "header": h,
                }
                task["address_packets"].append(address)
                addresses[a] = dict(address)
            task["packets"].append((a, h, vals))
            for aa, vv in zip(addrs, vals):
                regs[aa] = vv
                task["regs"][aa] = vv
                task["first_regs"].setdefault(aa, vv)
            i += 1 + len(addrs)
        task["effective"] = dict(regs)
        task["effective_addresses"] = {
            address: dict(value) for address, value in addresses.items()
        }
        tasks.append(task)
        pos += n
    return tasks


def effective_addr(task, base):
    r = task["effective"]
    lo = r.get(base)
    hi = r.get(base + 1)
    if lo is None:
        return None
    return (hi << 32 | lo) if hi is not None else lo


def fmt_addr(x):
    return "None" if x is None else f"0x{x:x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--text-only", action="store_true")
    args = ap.parse_args()
    b, cpu, subtype, ncmds, sections, ports = parse_macho(args.model)
    if ("__TEXT", "__text") not in sections:
        raise SystemExit("no __TEXT,__text")
    w = words_at(b, sections, ("__TEXT", "__text"))
    tasks = walk_tasks(w)
    print(f"model={args.model}")
    print(f"cpu={cpu} subtype={subtype} ncmds={ncmds} text_words={len(w)} tasks={len(tasks)}")
    for i, (vm, name) in enumerate(ports):
        print(f"port[{i}] vm=0x{vm:x} name={name}")
    if args.text_only:
        return
    print("idx typ size W H C oW oH oC src1 src2 dst kbase flags ext_bars")
    for t in tasks:
        if t.get("bad"):
            print(f"BAD pos=0x{t['pos']:x} h=0x{t['header']:x} n={t['size']}")
            continue
        r = t["effective"]
        ext = []
        for a in t["address_packets"]:
            if a["bar"] >= 20 and a["bar"] <= 44:
                ext.append(f'{a["reg"]:04x}:bar{a["bar"]}:{a["values"][0]:x}')
        print(
            f'{t["index"]:3d} {(r.get(0xf,0)>>4)&15:3d} {t["size"]*4:4x} '
            f'{r.get(1,0):4d} {r.get(2,0):4d} {r.get(3,0):3d} '
            f'{r.get(5,0):4d} {r.get(6,0):4d} {r.get(7,0):3d} '
            f'{fmt_addr(effective_addr(t,0x1346)):>10} '
            f'{fmt_addr(effective_addr(t,0x134c)):>10} '
            f'{fmt_addr(effective_addr(t,0x1444)):>10} '
            f'{fmt_addr(effective_addr(t,0x1544)):>10} '
            f'{r.get(0xf,0):08x} ' + ",".join(ext)
        )


if __name__ == "__main__":
    main()
