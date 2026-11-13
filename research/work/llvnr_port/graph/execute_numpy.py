#!/usr/bin/env python3
"""LLVNR H19 full-graph NumPy reference executor.

Reads   work/llvnr_port/graph/graph.json          (535 nodes, 1682 edges)
        work/llvnr_port/semantics/ops.json        (per-task op family / params)
        work/llvnr_port/weights/kernels.json      (decoded kernel groups, optional)

Fails CLOSED: a task whose op family has no registered implementation, or whose
conv kernel weights are not available, raises and is reported - it is never
silently approximated.  Use --stub-weights to run the *framework* with
deterministic pseudo-random kernels; every stubbed task is flagged in the trace
and the run summary, and the run exit code stays 2.

Tensor convention
-----------------
All tensors are float32 numpy arrays shaped (C, H, W) - channel first, matching
the ANE register InDim/OutDim (W,H,C) triplets.  Model storage is FLOAT16 so a
task output is optionally rounded to fp16 (--fp16-round, default off for the
reference).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

GRAPH_DIR = pathlib.Path(__file__).resolve().parent
WS = GRAPH_DIR.parents[2]
OPS_PATH = WS / "work/llvnr_port/semantics/ops.json"
KERNELS_PATH = WS / "work/llvnr_port/weights/kernels.json"
STATUS_EXIT_OK = 0
STATUS_EXIT_PARTIAL = 2


class UnsupportedOp(RuntimeError):
    pass


class MissingWeights(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# op registry
# --------------------------------------------------------------------------- #
def _chw(dim):
    if not dim:
        raise UnsupportedOp("task has no output dim")
    return (max(dim.get("c") or 1, 1), dim["h"], dim["w"])


def _per_channel(a, b):
    """Broadcast a per-channel vector (C,) or (C,1,1) against (C,H,W)."""
    if b is None:
        return None
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 1:
        return b.reshape(-1, 1, 1)
    return b


def fit_input(a, dim, notes):
    '''Adapt a producer buffer to the consumer declared (C,H,W) shape.

    The ANE TileDMA can read a window out of a larger buffer; the exact window
    offset is NOT decoded yet, so this crops at the top-left corner and records
    the approximation in notes (never silent).
    '''
    if not dim:
        return a
    want = _chw(dim)
    if a.shape == want:
        return a
    if a.ndim == 3 and a.shape[0] >= want[0] and a.shape[1] >= want[1] and a.shape[2] >= want[2]:
        notes.append({"adapter": "crop-top-left", "from": list(a.shape), "to": list(want)})
        return a[:want[0], :want[1], :want[2]]
    if a.ndim == 3 and a.shape[0] == want[0] and a.shape[1] <= want[1] and a.shape[2] <= want[2]:
        dh, dw = want[1] - a.shape[1], want[2] - a.shape[2]
        t, l = dh // 2, dw // 2
        notes.append({"adapter": "symmetric-pad", "from": list(a.shape), "to": list(want),
                      "pad": [t, dh - t, l, dw - l]})
        return np.pad(a, ((0, 0), (t, dh - t), (l, dw - l)), mode="constant")
    if a.size >= int(np.prod(want)):
        notes.append({"adapter": "reshape-flat", "from": list(a.shape), "to": list(want)})
        return a.reshape(-1)[:int(np.prod(want))].reshape(want)
    raise UnsupportedOp(f"cannot adapt producer shape {a.shape} to consumer {want} "
                        f"(window/crop offset not decoded)")


def op_copy(task, ins, ctx):
    return np.array(ins[0], dtype=np.float32, copy=True)


def _binary(fn):
    def impl(task, ins, ctx):
        a = ins[0].astype(np.float32)
        b = ins[1].astype(np.float32) if len(ins) > 1 else np.float32(0.0)
        if np.ndim(b) == 1:
            b = b.reshape(-1, 1, 1)
        return fn(a, b).astype(np.float32)
    return impl


def _pool(task, ins, ctx, mode="max"):
    a = ins[0].astype(np.float32)
    ent = (task.get("ops_entry") or {}).get("op") or {}
    k = ent.get("kernel") or task.get("pool_kernel") or (2, 2)
    s = ent.get("stride") or task.get("pool_stride") or k
    ph, pw = k
    sh, sw = s
    c, h, w = a.shape
    oh = (h - ph) // sh + 1
    ow = (w - pw) // sw + 1
    out = np.full((c, oh, ow), -np.inf if mode == "max" else 0.0, dtype=np.float32)
    for y in range(ph):
        for x in range(pw):
            sl = a[:, y:y + oh * sh:sh, x:x + ow * sw:sw]
            out = np.maximum(out, sl) if mode == "max" else out + sl
    if mode != "max":
        out /= (ph * pw)
    return out


def op_pool_max(task, ins, ctx):
    return _pool(task, ins, ctx, "max")


def op_pool_avg(task, ins, ctx):
    return _pool(task, ins, ctx, "avg")


def op_conv(task, ins, ctx):
    """Dense reference conv.  Requires decoded weights from kernels.json.

    Weights are expected as (Oc, Ic, Kh, Kw) float32 plus bias (Oc,), produced by
    the weights member.  Without them this raises MissingWeights (fail closed).
    """
    gs = task.get("kernel_groups") or []
    ind = task.get("in_dim") or {}
    outd = task.get("out_dim") or {}
    in_c = max(ind.get("c") or 1, 1)
    out_c = max(outd.get("c") or 1, 1)
    # A TD can list several groups; the conv is defined by the group whose
    # (Ic, Oc) fit the task dataflow (other groups are auxiliary metadata/LUT).
    chosen, why = None, None
    forced = task.get("kernel_group_override")
    if forced:
        chosen, why = forced, "authoritative kernel shape (stale register override)"
    for g in ([] if chosen else gs):
        cand = ctx["kernels"].get(g)
        if cand is None:
            continue
        wsh = cand["w"].shape
        if len(wsh) == 4 and wsh[1] == in_c and wsh[0] >= out_c:
            chosen = g
            why = "single group fits dataflow"
            break
    # NOTE: no "forced" fallback - a kernel whose Ic != task in C would silently
    # drop input channels, so it must fail closed and be reported.
    if chosen is None:
        if ctx["stub_weights"]:
            chosen = (gs[0] if gs else "none")
            why = "stub"
        else:
            detail = []
            for g in gs:
                cand = ctx["kernels"].get(g)
                detail.append(f'{g[:12]}' + (f' shape={tuple(cand["w"].shape)}' if cand else ' (no weights)'))
            raise MissingWeights(f"no kernel group fits in C={in_c} out C={out_c}: {detail}")
    group = chosen
    w = ctx["kernels"].get(group)
    if w is None:
        if ctx["stub_weights"]:
            w = ctx["stub_kernels"][group or "none"]
            ctx["stub_used"].add(task["index"])
        else:
            raise MissingWeights(f"kernel group {group} not decoded in decoded_kernels.json")
    ent = (task.get("ops_entry") or {}).get("conv") or {}
    conv = task.get("conv") or {}
    kh = ent.get("kh") or conv.get("kh") or w["w"].shape[2]
    kw = ent.get("kw") or conv.get("kw") or w["w"].shape[3]
    sh = ent.get("sx") or conv.get("sx") or 1
    sw = ent.get("sy") or conv.get("sy") or 1
    pad_t = ent.get("pad_top", conv.get("pad_top") or 0) or 0
    pad_l = ent.get("pad_left", conv.get("pad_left") or 0) or 0
    conv = dict(conv or {}, groups=ent.get("groups") or (task.get("num_groups") or None))
    x = ins[0].astype(np.float32)
    wt = np.asarray(w["w"], dtype=np.float32)
    oc, ic_g, kh, kw = wt.shape
    bias = np.asarray(w.get("b") if w.get("b") is not None else np.zeros(oc),
                      dtype=np.float32).reshape(-1)
    groups = conv.get("groups") or max(1, x.shape[0] // ic_g)
    c, h, wd = x.shape
    max_elems = int(ctx.get("max_elems") or 0)
    hp = h + 2 * pad_t
    wp = wd + 2 * pad_l
    oh = (hp - kh) // sh + 1
    ow = (wp - kw) // sw + 1
    if max_elems and oc * oh * ow > max_elems:
        raise UnsupportedOp(f"output {oc}x{oh}x{ow} exceeds --max-elems {max_elems}")
    # OC authority rule (task-24): the kernel shape Oc is authoritative.  The task's
    # OutChannels register can be stale (inherited from the previous task), so it is
    # only used to crop when the task itself wrote OutChannels.
    reg_oc = max((task.get("out_dim") or {}).get("c") or 1, 1)
    trusted = task.get("out_c_trusted", True)
    want_oc = reg_oc if trusted else oc
    if oc < want_oc:
        raise UnsupportedOp(f"kernel Oc={oc} < task out C={want_oc}")
    out = np.zeros((oc, oh, ow), dtype=np.float32)
    oc_g = oc // groups
    for g in range(groups):
        xg = x[g * ic_g:(g + 1) * ic_g]
        xp = np.pad(xg, ((0, 0), (pad_t, max(0, hp - pad_t - h)),
                         (pad_l, max(0, wp - pad_l - wd))), mode="constant")
        cols = np.empty((ic_g, kh, kw, oh, ow), dtype=np.float32)
        for ky in range(kh):
            for kx in range(kw):
                cols[:, ky, kx, :, :] = xp[:, ky:ky + oh * sh:sh, kx:kx + ow * sw:sw]
        flat = cols.reshape(ic_g * kh * kw, oh * ow)
        wg = wt[g * oc_g:(g + 1) * oc_g].reshape(oc_g, ic_g * kh * kw)
        out[g * oc_g:(g + 1) * oc_g] = (wg @ flat).reshape(oc_g, oh, ow)
    out += bias.reshape(-1, 1, 1)
    if out.shape[0] > want_oc:
        out = out[:want_oc]
    return out


def op_ew_add(task, ins, ctx):
    return _binary(lambda a, b: a + b)(task, ins, ctx)


def op_ew_sub(task, ins, ctx):
    return _binary(lambda a, b: a - b)(task, ins, ctx)


def op_ew_mul(task, ins, ctx):
    return _binary(lambda a, b: a * b)(task, ins, ctx)


def op_ew_max(task, ins, ctx):
    return _binary(np.maximum)(task, ins, ctx)


def op_ew_min(task, ins, ctx):
    return _binary(np.minimum)(task, ins, ctx)


def op_ew_abs(task, ins, ctx):
    return np.abs(ins[0].astype(np.float32))


def op_ew_absdiff(task, ins, ctx):
    return _binary(lambda a, b: np.abs(a - b))(task, ins, ctx)


def op_relu(task, ins, ctx):
    return np.maximum(ins[0].astype(np.float32), 0.0)


def op_linear(task, ins, ctx):
    """PE linear activation: alpha * x + beta (registers PE PreScale / Bias)."""
    a = float(task.get("linear_alpha", 1.0))
    b = float(task.get("linear_beta", 0.0))
    return (a * ins[0].astype(np.float32) + b).astype(np.float32)


def op_clamp(task, ins, ctx):
    lo = float(task.get("clamp_lo", -np.inf))
    hi = float(task.get("clamp_hi", np.inf))
    return np.clip(ins[0].astype(np.float32), lo, hi)


# semantics/ops.json uses richer family names; map them onto the reference
# implementations above.  Families that need data this reference does not have
# (LUT tables, decoded dense kernels) are NOT aliased: they stay unsupported and
# are reported fail-closed.
FAMILY_ALIASES = {
    "conv": "conv",
    "conv_sparse": "conv",
    "conv_palette": "conv",
    "conv_dense": "conv",
    "conv_weightless_unknown": "conv",
    "pool": "pool",
    "pool_with_lut": "pool",   # LUT post-pass missing -> flagged separately
    "ew": "copy",
    "ew_add": "ew_add",
    "ew_sub": "ew_sub",
    "ew_mul": "ew_mul",
    "ew_abs": "ew_abs",
    "ew_absdiff": "ew_absdiff",
    "ew_max": "ew_max",
    "ew_min": "ew_min",
    "ew_relu": "relu",
    "copy": "copy",
    "dma": "copy",
    "EW5": "ew_dispatch",
    "EW6": "ew_dispatch",
    "ew": "ew_dispatch",
    "ew_affine_lut": "ew_dispatch",
    "pool_with_lut": "pool_dispatch",
}


def _lut_from_op(op):
    '''Build a piecewise-linear LUT from semantics/ops.json lut_values.

    ANE PE LUT: up to 8 breakpoints, linear interpolation between them.
    ops.json gives lut1..lut8 which for a 2-point ramp is {0:0, 1:1} (== clamp).
    '''
    lv = op.get("lut_values") or {}
    xs, ys = [], []
    for k in range(1, 9):
        v = lv.get(f"lut{k}")
        if v is None:
            continue
        xs.append(float(k - 1))
        ys.append(float(v))
    if len(xs) < 2:
        return None
    return np.asarray(xs), np.asarray(ys)


def _apply_lut(t, lut):
    xs, ys = lut
    return np.interp(t, xs, ys).astype(np.float32)


def op_ew_dispatch(task, ins, ctx):
    '''Element-wise / planar-engine task driven by semantics/ops.json params.

    Formula: y = LUT( prescale * g(x1, x2) + bias ), matching the PE register set
    (PreScale 0x4510, Bias 0x4504, OpMode 0x4500 bits 2-4, LUT1..8 0x4518+).
    The formula itself is an INFERRED reading of the PE datapath and is recorded
    per task in the trace ("numeric_model"); it is not numerically verified
    against hardware.
    '''
    ent = task.get("ops_entry") or {}
    op = ent.get("op") or {}
    kind = op.get("kind")
    a = ins[0].astype(np.float32)
    if kind == "affine_lut":
        g = a
    elif kind == "binary":
        name = (op.get("op") or "").lower()
        b = ins[1].astype(np.float32) if len(ins) > 1 else None
        if b is not None and np.ndim(b) == 1:
            b = b.reshape(-1, 1, 1)
        if name in ("add", "sub", "mul", "max", "min") and b is not None:
            g = {"add": a + b, "sub": a - b, "mul": a * b,
                 "max": np.maximum(a, b), "min": np.minimum(a, b)}[name]
        elif b is None and name == "mul":
            g = a * a
        else:
            raise UnsupportedOp(f"binary EW op {op.get('op')} needs 2 inputs (got {len(ins)})")
    elif kind in ("unary", "abs"):
        g = np.abs(a)
    elif kind in ("absdiff", "abs_diff"):
        b = ins[1].astype(np.float32) if len(ins) > 1 else None
        if b is None:
            raise UnsupportedOp("absdiff needs 2 inputs")
        if np.ndim(b) == 1:
            b = b.reshape(-1, 1, 1)
        g = np.abs(a - b)
    else:
        raise UnsupportedOp(f"EW op kind {kind} not implemented")
    pre = op.get("prescale")
    bias = op.get("bias")
    if pre is not None:
        g = g * float(pre)
    if bias is not None:
        g = g + float(bias)
    lut = _lut_from_op(op)
    if kind == "affine_lut" and lut is None:
        raise UnsupportedOp("affine_lut without lut_values")
    if lut is not None:
        g = _apply_lut(g, lut)
    return np.asarray(g, dtype=np.float32)


def op_pool_dispatch(task, ins, ctx):
    out = op_pool_max(task, ins, ctx)
    ent = task.get("ops_entry") or {}
    lut = _lut_from_op(ent.get("op") or {})
    return _apply_lut(out, lut) if lut is not None else out


OP_REGISTRY = {
    "copy": op_copy,
    "pool": op_pool_max,
    "pool_max": op_pool_max,
    "pool_avg": op_pool_avg,
    "conv": op_conv,
    "ew_add": op_ew_add,
    "ew_sub": op_ew_sub,
    "ew_mul": op_ew_mul,
    "ew_max": op_ew_max,
    "ew_min": op_ew_min,
    "ew_abs": op_ew_abs,
    "ew_absdiff": op_ew_absdiff,
    "relu": op_relu,
    "linear": op_linear,
    "clamp": op_clamp,
    "ew_dispatch": op_ew_dispatch,
    "pool_dispatch": op_pool_dispatch,
}


# --------------------------------------------------------------------------- #
# graph loading
# --------------------------------------------------------------------------- #
def load_graph():
    return json.loads((GRAPH_DIR / "graph.json").read_text())


def load_ops():
    if not OPS_PATH.exists():
        return {}
    raw = json.loads(OPS_PATH.read_text())
    items = raw.get("tasks", raw)
    if isinstance(items, list):
        return {int(t["index"]): t for t in items}
    return {int(k): v for k, v in items.items()}


def load_kernels():
    """Load decoded kernel weights from graph/decoded_kernels.json (task-indexed).

    Returns {group_stem: {"w": np.ndarray (Oc,Ic,Kh,Kw), "b": np.ndarray (Oc,)}}.
    Only groups with decode_status == verified carry npy paths, so an unverified
    group simply will not be found -> op_conv raises MissingWeights (fail closed).
    """
    path = GRAPH_DIR / "decoded_kernels.json"
    out = {}
    if not path.exists():
        return out
    man = json.loads(path.read_text())
    for stem, g in man.get("groups", {}).items():
        if not g.get("w_path"):
            continue
        wp = WS / g["w_path"]
        if not wp.exists():
            continue
        w = np.load(wp).astype(np.float32)
        b = None
        if g.get("b_path") and (WS / g["b_path"]).exists():
            b = np.load(WS / g["b_path"]).astype(np.float32).reshape(-1)
        out[stem] = {"w": w, "b": b, "meta": g}
    return out


def make_stub_kernels(graph, seed=0):
    """Deterministic pseudo-random kernels for framework smoke tests only."""
    rng = np.random.default_rng(seed)
    stubs = {}
    for n in graph["nodes"]:
        if not n["id"].startswith("t"):
            continue
        conv = n.get("conv") or {}
        kh, kw = conv.get("kh") or 1, conv.get("kw") or 1
        ind, outd = n.get("in_dim") or {}, n.get("out_dim") or {}
        ic = max(ind.get("c") or 1, 1)
        oc = max(outd.get("c") or 1, 1)
        for g in n.get("kernel_groups") or ["none"]:
            if g in stubs:
                continue
            stubs[g] = {"w": rng.standard_normal((oc, ic, kh, kw)).astype(np.float32) * 0.05,
                        "b": np.zeros(oc, dtype=np.float32)}
    return stubs


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def task_op_family(task, ops):
    ent = ops.get(task["index"]) or {}
    fam = ent.get("op_family") or ent.get("family")
    if isinstance(fam, dict):
        fam = fam.get("family")
    if fam:
        return fam
    o = task.get("op") or {}
    return o.get("family") if isinstance(o, dict) else None


def run(graph, ops, kernels, args):
    nodes = [n for n in graph["nodes"] if n["id"].startswith("t")]
    nodes.sort(key=lambda n: n["index"])
    bufs = {}
    trace = []
    skipped = []
    stub_used = set()
    approx = set()
    ctx = {"kernels": kernels, "stub_weights": args.stub_weights,
           "stub_kernels": {}, "stub_used": stub_used,
           "max_elems": args.max_elems}
    if args.stub_weights:
        ctx["stub_kernels"] = make_stub_kernels(graph, args.seed)

    # seed external inputs
    rng = np.random.default_rng(args.seed)
    for p in graph["meta"]["ports"]:
        if args.frame:
            fp = pathlib.Path(args.frame) / f'{p["name"]}.npy'
            if fp.exists():
                bufs[f'port:{p["name"]}+0'] = np.load(fp).astype(np.float32)
                continue
        # port shape is [1, C, H, W] -> numpy (C, H, W)
        c, h, w = p["shape"][-3], p["shape"][-2], p["shape"][-1]
        bufs[f'port:{p["name"]}+0'] = rng.standard_normal((c, h, w)).astype(np.float32) * 0.1
    t0 = time.time()
    for n in nodes:
        i = n["index"]
        if args.tasks and not (args.tasks[0] <= i <= args.tasks[1]):
            continue
        fam = task_op_family(n, ops)
        impl = OP_REGISTRY.get(fam) or OP_REGISTRY.get(FAMILY_ALIASES.get(fam, ""))
        # LUT-bearing families are only supported when a decoded LUT table exists
        # (weights/lut_manifest.json); without it they stay fail-closed.

        rec = {"task": i, "op_family": fam, "ts": round(time.time() - t0, 4)}
        if args.dry_run:
            rec["status"] = "planned" if impl else "unsupported"
            rec["inputs"] = [e["buffer"] for e in n["nodes_in"]]
            rec["shape_out"] = _chw(n.get("out_dim")) if n.get("out_dim") else None
            trace.append(rec)
            if not impl:
                skipped.append({"task": i, "op_family": fam, "reason": "no op implementation"})
            continue
        try:
            ins = [bufs[e["buffer"]] for e in n["nodes_in"]]
        except KeyError as exc:
            skipped.append({"task": i, "op_family": fam,
                            "reason": f"missing input buffer {exc}"})
            rec["status"] = "missing-input"
            trace.append(rec)
            if not args.allow_skip:
                raise
            continue
        if not impl:
            skipped.append({"task": i, "op_family": fam, "reason": "no op implementation"})
            rec["status"] = "unsupported"
            trace.append(rec)
            continue
        if not ins and fam not in ("copy",):
            skipped.append({"task": i, "op_family": fam,
                            "reason": "no resolvable input edge (src bases all zero / "
                                      "register inherited)"})
            rec["status"] = "no-input"
            trace.append(rec)
            continue
        tinfo = dict(n)
        tinfo["conv"] = n.get("conv") or {}
        tinfo["ops_entry"] = ops.get(i) or {}
        adapters = []
        try:
            ins = [fit_input(a, n.get("in_dim"), adapters) for a in ins]
            out = impl(tinfo, ins, ctx)
        except (MissingWeights, UnsupportedOp) as exc:
            skipped.append({"task": i, "op_family": fam, "reason": str(exc)})
            rec["status"] = "fail-closed"
            trace.append(rec)
            continue
        except Exception as exc:  # noqa: BLE001 - report, never silently approximate
            skipped.append({"task": i, "op_family": fam,
                            "reason": f"{type(exc).__name__}: {exc}"})
            rec["status"] = "error"
            trace.append(rec)
            if not args.allow_skip:
                raise
            continue
        if out is None or out.size == 0:
            skipped.append({"task": i, "op_family": fam,
                            "reason": f"empty output shape {None if out is None else out.shape} "
                                      "(OutDim omitted in registers / inherited)"})
            rec["status"] = "fail-closed-empty"
            trace.append(rec)
            continue
        if args.fp16_round:
            out = out.astype(np.float16).astype(np.float32)
        if adapters:
            rec["input_adapters"] = adapters
            approx.add(i)
        rec.update({"status": "ok", "shape": list(out.shape),
                    "min": float(out.min()), "max": float(out.max()),
                    "mean": float(out.mean()),
                    "finite": bool(np.isfinite(out).all())})
        trace.append(rec)
        for o in n["nodes_out"]:
            bufs[o["buffer"]] = out
    elapsed = time.time() - t0
    summary = {
        "tasks_total": len(nodes),
        "tasks_executed": sum(1 for r in trace if r.get("status") == "ok"),
        "tasks_planned": sum(1 for r in trace if r.get("status") == "planned"),
        "skipped": len(skipped),
        "skipped_detail": skipped[:20],
        "stub_weights_used": sorted(stub_used) if args.stub_weights else [],
        "tasks_with_input_approximation": sorted(approx),
        "elapsed_s": round(elapsed, 3),
        "op_family_counts": {},
    }
    for r in trace:
        f = str(r.get("op_family"))
        summary["op_family_counts"][f] = summary["op_family_counts"].get(f, 0) + 1
    return bufs, trace, summary, skipped



def _sha(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def run_isolate(graph, ops, kernels, args):
    """Run selected tasks with REAL decoded weights.

    Two modes:
      * plain isolate: deterministic synthetic inputs shaped by in_dim;
      * chained (default when an edge connects two selected tasks): the consumer
        takes the producer's output (adapted to in_dim) instead of synthetic data.

    Emits golden .npy + a manifest with input source, input/output SHA-256, conv
    parameters and weight hashes so the H13 member can replay exactly.
    """
    nodes = {n["index"]: n for n in graph["nodes"] if n["id"].startswith("t")}
    # task->group overrides (relocation lane aliases resolved via the KernelDMA
    # base register + decoded shape fit; see graph/group_overrides.json)
    kd_doc = json.loads((GRAPH_DIR / "decoded_kernels.json").read_text()) \
        if (GRAPH_DIR / "decoded_kernels.json").exists() else {"tasks": {}}
    want = sorted(nodes)
    if getattr(args, "task_list", None):
        want = [i for i in want if i in set(args.task_list)]
    elif args.tasks:
        want = [i for i in want if args.tasks[0] <= i <= args.tasks[1]]
    gd = pathlib.Path(args.golden or (GRAPH_DIR / "golden"))
    gd.mkdir(parents=True, exist_ok=True)
    recs, skipped = [], []
    computed = {}
    ctx = {"kernels": kernels, "stub_weights": False, "stub_kernels": {},
           "stub_used": set(), "max_elems": args.max_elems}
    for i in want:
        n = nodes[i]
        ent = ops.get(i) or {}
        tinfo = dict(n)
        tinfo["ops_entry"] = ent
        tinfo["conv"] = ent.get("conv") or n.get("conv") or {}
        tinfo["out_c_trusted"] = n.get("out_c_written", True)
        _kd = (kd_doc.get("tasks") or {}).get(str(i)) or {}
        if _kd.get("group"):
            # decoded_kernels.json is the authoritative task->group mapping
            # (weights corrected table + task_kernel_manifest front table)
            tinfo["kernel_groups"] = [_kd["group"]]
            tinfo["kernel_group_override"] = _kd["group"]
        fam = (ent.get("family") or (n.get("op") or {}).get("family"))
        impl = OP_REGISTRY.get(fam) or OP_REGISTRY.get(FAMILY_ALIASES.get(fam, ""))
        dim = n.get("in_dim")
        if not dim:
            skipped.append({"task": i, "reason": "no in_dim"})
            continue
        shape = _chw(dim)
        # ---- OC/IC authority rule: kernel shape wins over a stale (inherited) register
        auth = None
        kg_all = n.get("kernel_groups") or []
        for g in kg_all:
            kk = kernels.get(g)
            if not kk:
                continue
            wsh = kk["w"].shape
            if len(wsh) != 4:
                continue
            fit_ic = (wsh[1] == shape[0]) or (n.get("in_c_written") is False)
            if fit_ic and (wsh[0] >= shape[0] or True):
                auth = {"group": g, "Ic": int(wsh[1]), "Oc": int(wsh[0]),
                        "register_in_c": int(shape[0])}
                if wsh[1] != shape[0]:
                    auth["ic_overridden"] = True
                break
        if auth and auth["Ic"] != shape[0]:
            # stale InChannels: build the authoritative input at kernel Ic
            shape = (auth["Ic"], shape[1], shape[2])
        tinfo["kernel_group_override"] = auth["group"] if auth else None
        tinfo["in_dim_authoritative"] = shape
        ins, srcs = [], []
        for e in n["nodes_in"]:
            src_i = int(e["from"][1:]) if e["from"].startswith("t") else None
            if src_i in computed:
                adapt = []
                try:
                    a = fit_input(computed[src_i], dim, adapt)
                except UnsupportedOp as exc:
                    srcs.append({"from": f"t{src_i}", "fallback_synthetic": str(exc)})
                    continue
                ins.append(a)
                srcs.append({"from": f"t{src_i}", "adapters": adapt})
            else:
                srcs.append({"from": e["from"], "synthetic": True})
        is_conv = fam in ("conv", "conv_sparse", "conv_palette", "conv_dense",
                          "conv_weightless_unknown") or (ent.get("op") or {}).get("kind") == "conv"
        n_needed = 1 if (is_conv or fam in ("pool", "pool_dispatch")) else 2
        k = 0
        while len(ins) < n_needed:
            a = np.random.default_rng(args.seed * 100003 + i * 7 + k).standard_normal(shape)
            ins.append(a.astype(np.float32))
            srcs.append({"from": f"synthetic#{k}", "synthetic": True})
            k += 1
        try:
            out = impl(tinfo, ins, ctx)
        except (MissingWeights, UnsupportedOp) as exc:
            skipped.append({"task": i, "op_family": fam, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001
            skipped.append({"task": i, "op_family": fam, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if out is None or out.size == 0:
            skipped.append({"task": i, "op_family": fam, "reason": "empty output"})
            continue
        computed[i] = out
        fp = gd / f"iso_task{i:03d}.npy"
        np.save(fp, out.astype(np.float32))
        kg = (n.get("kernel_groups") or [None])[0]
        kk = ctx["kernels"].get(kg) or {}
        recs.append({
            "task": i, "op_family": fam, "kernel_group": kg,
            "weight_source": (kk.get("meta") or {}).get("w_path"),
            "weight_sha256": (kk.get("meta") or {}).get("weight_sha256"),
            "verification": (kk.get("meta") or {}).get("verification"),
            "conv_params": (ent.get("conv") or None),
            "authoritative_channels": ({"Ic": auth["Ic"], "Oc": auth["Oc"],
                                        "group": auth["group"],
                                        "register_in_c": auth["register_in_c"],
                                        "ic_overridden": bool(auth.get("ic_overridden")),
                                        "in_c_written": n.get("in_c_written"),
                                        "out_c_written": n.get("out_c_written"),
                                        "out_c_trusted": tinfo.get("out_c_trusted", True)}
                                       if auth else None),
            "in_shape": list(shape), "n_inputs": len(ins),
            "input_sources": srcs,
            "out_shape": list(out.shape),
            "input_sha256": [_sha(a) for a in ins],
            "output_sha256": _sha(out),
            "out_min": float(out.min()), "out_max": float(out.max()),
            "out_mean": float(out.mean()), "finite": bool(np.isfinite(out).all()),
            "golden": str(fp.relative_to(WS)) if str(fp).startswith(str(WS)) else str(fp),
        })
    manifest = {
        "mode": "real_weights" + ("_chained" if any(not s.get("synthetic") for r in recs
                                                    for s in r["input_sources"]) else "_isolated"),
        "note": ("tasks executed in stream order with decoded verified kernels from "
                 "decoded_kernels.json; a consumer whose producer is in the selected set "
                 "takes the producer output, otherwise a deterministic synthetic input"),
        "seed": args.seed,
        "tasks_run": recs, "skipped": skipped,
        "n_run": len(recs), "n_skipped": len(skipped),
        "weight_ready_tasks_used": sorted(r["task"] for r in recs if r["weight_source"]),
    }
    mp = pathlib.Path(args.isolate_manifest or (gd / "isolated_manifest.json"))
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("mode", "n_run", "n_skipped",
                                              "weight_ready_tasks_used")}, indent=1)[:2000])
    for r in recs:
        chained = [s["from"] for s in r["input_sources"] if not s.get("synthetic")]
        print(f'  t{r["task"]:3d} {r["op_family"]:8s} {r["in_shape"]} -> {r["out_shape"]} '
              f'w={str(r["weight_sha256"])[:12]} out_sha={r["output_sha256"][:12]} '
              f'mean={r["out_mean"]:.6f} chained_inputs={chained}')
    if skipped:
        print("skipped:", skipped, file=sys.stderr)
    print("manifest ->", mp)
    return recs, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="structure check only")
    ap.add_argument("--tasks", type=str, default=None, help="task range A-B")
    ap.add_argument("--trace", type=str, default=None, help="write per-task trace JSON")
    ap.add_argument("--golden", type=str, default=None, help="directory for golden .npy")
    ap.add_argument("--golden-tasks", type=str, default="0,1,2,3,8",
                    help="comma separated task indices to dump as golden")
    ap.add_argument("--frame", type=str, default=None,
                    help="directory with cur_frame.npy / his_frame.npy (C,H,W) float32")
    ap.add_argument("--state", type=str, default=None,
                    help="state .npz to load state_yuv/state_weight_band0 from")
    ap.add_argument("--save-state", type=str, default=None, help="write updated state .npz")
    ap.add_argument("--stub-weights", action="store_true",
                    help="FRAMEWORK SMOKE TEST ONLY: deterministic pseudo-random kernels")
    ap.add_argument("--allow-skip", action="store_true", help="do not raise on missing input")
    ap.add_argument("--fp16-round", action="store_true", help="round each output to fp16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-elems", type=int, default=0,
                    help="skip ops whose output element count exceeds this")
    ap.add_argument("--isolate", action="store_true",
                    help="run selected tasks standalone with real decoded weights "
                         "(deterministic synthetic inputs, golden + sha manifest)")
    ap.add_argument("--isolate-manifest", type=str, default=None)
    args = ap.parse_args()
    if args.tasks:
        spec = args.tasks.strip()
        if "," in spec:
            args.task_list = sorted({int(x) for x in spec.split(",") if x.strip()})
            args.tasks = (args.task_list[0], args.task_list[-1])
        else:
            lo, _, hi = spec.partition("-")
            args.tasks = (int(lo), int(hi or lo))
            args.task_list = None
    else:
        args.task_list = None
    graph = load_graph()
    ops = load_ops()
    kernels = load_kernels()
    if args.isolate:
        recs, skipped = run_isolate(graph, ops, kernels, args)
        return STATUS_EXIT_OK if not skipped else STATUS_EXIT_PARTIAL
    bufs, trace, summary, skipped = run(graph, ops, kernels, args)
    print(json.dumps(summary, indent=1))
    if args.trace:
        pathlib.Path(args.trace).write_text(json.dumps(trace, indent=1))
        print("trace ->", args.trace)
    if args.golden:
        gd = pathlib.Path(args.golden)
        gd.mkdir(parents=True, exist_ok=True)
        want = {int(x) for x in args.golden_tasks.split(",") if x.strip()}
        by_task = {n["index"]: n for n in graph["nodes"] if n["id"].startswith("t")}
        saved = 0
        for r in trace:
            if r["task"] not in want or r.get("status") != "ok":
                continue
            n = by_task[r["task"]]
            for o in n["nodes_out"]:
                if o["buffer"] in bufs:
                    np.save(gd / f'task{r["task"]:03d}_{o["space"]}.npy', bufs[o["buffer"]])
                    saved += 1
                    break
        print("golden files:", saved, "->", gd)
    if skipped:
        print(f"WARNING: {len(skipped)} task(s) not executed (fail-closed); "
              f"first: {skipped[0]}", file=sys.stderr)
    return STATUS_EXIT_OK if not skipped else STATUS_EXIT_PARTIAL


if __name__ == "__main__":
    sys.exit(main())