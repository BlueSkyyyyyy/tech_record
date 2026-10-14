"""P7-61: the **same-CTA `dw` handoff** of the K2-shaped CUDA B2 -- the last structural item
named by §49.7 and left open by §63 (`prod` proved the six segments coexist in one kernel, but
DKGB still consumed a *global* `dw` that the GEMM core itself had just written).  `-DB2M_HDW`
keeps the core's completed `dw` (=`accw`) in `swall` and runs the `dkgb = A @ (-dw)` mma
**post-loop** off that mirror: the segment never stages `dw` from global and the core never
stores it -- the whole `dw` round-trip disappears.

Correctness is bit-identical *by construction*: the handed-off value is exactly the `accw` the
global path stored and read back (the harness always feeds the segments with `core.dw`).  Part A
verifies this against the P7-60 `prod` build for every output, and Part A2 makes the handoff
self-evident -- feeding `-DB2M_HDW` a *different* `dw` input (zeros) must not change a single
output bit.

    Part A   correctness: prod_hdw vs prod -- every output bit-identical (incl non-div T=250).
    Part A2  handoff proof: prod_hdw(dw=0) vs prod_hdw(dw=core) -- bit-identical (dw unused).
    Part B   resources: regs / spill / smem of prod vs prod_hdw (+ TMA variants).
    Part C   timing: one prod_hdw launch vs one prod launch (both single-launch), CUPTI `dev`
            口径 -- the saving is the `dw` global write + re-read staging.
    Part D   wall: event-pair wall (`REPS` calls per pair, averaged) as a second timing口径, so
             the (notable) ncu-vs-CUPTI disagreement on this kernel can be cross-checked.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_hdw.py \
        > .../harness/check_bwd_b2_hdw.out.txt 2>&1
    B2H_MODE=a | b | c
"""
import ctypes
import os
import re
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs   # noqa: E402
from check_bwd_ws_reuse import cdiv                                  # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                        # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

SEG = {
    "core":     [],
    "prod":     ["-DB2M_PROD=1"],
    "prod_hdw": ["-DB2M_PROD=1", "-DB2M_HDW=1"],
    "prod_tma": ["-DB2M_PROD=1", "-DB2M_TMA=1"],
    "prod_hdw_tma": ["-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_TMA=1"],
    # isolated DKGB segment (in-loop, global dw) vs handoff (post-loop, in-kernel dw) -- isolates
    # the in-loop->post-loop move + staging removal from the six-segment prod kernel.
    "dkgb":     ["-DB2M_DKGB=1"],
    "dkgb_hdw": ["-DB2M_DKGB=1", "-DB2M_HDW=1"],
}


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build_v(tag, extra):
    so = f"/tmp/b2_hdw_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"] + extra + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return so, r.stderr


def build(tag, extra):
    return build_v(tag, extra)[0]


def load(so):
    return ctypes.CDLL(so)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def run_core(lib, inp):
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    rc = lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                          ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                          B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2mma_launch rc={rc}")
    return dict(dw=dw, dgk=dgk)


def run_prod(lib, inp, dw):
    """Unified entry: `prod` / `prod_hdw` / TMA variants share `b2prod_launch`'s signature."""
    lib.b2prod_launch.restype = ctypes.c_int
    lib.b2prod_launch.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] +
                                  [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 +
                                  [ctypes.c_void_p])
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dvb = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dqg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dkg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    db_out = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    rc = lib.b2prod_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dw),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk),
                           ptr(inp["A"]), ptr(inp["beta"]), ptr(dv2), ptr(dvb),
                           ptr(dkgb), ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                           ptr(db_out), ptr(dApre), ptr(inp["A"]), ptr(dA),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2prod_launch rc={rc}")
    return dict(dw=dw_out, dgk=dgk, dkgb=dkgb, dv2=dv2, dvb=dvb, db=db_out,
                dApre=dApre, dA=dA, dqg=dqg, dkg=dkg, dg=dg)


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print(f"    {name:32s} max_abs={d:.3e}  [{flag}]")
    return d


# ------------------------------------------------------------------------- parts
def part_a(cases):
    print("=" * 78)
    print("Part A: prod_hdw (same-CTA dw handoff) vs prod (global dw round-trip)")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in SEG.items()}
    worst = 0.0
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        dw = run_core(libs["core"], inp)["dw"]
        p = run_prod(libs["prod"], inp, dw)
        h = run_prod(libs["prod_hdw"], inp, dw)
        ht = run_prod(libs["prod_hdw_tma"], inp, dw)
        for f in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
            worst = max(worst, bits(f"prod_hdw.{f} vs prod", h[f], p[f]))
        worst = max(worst, bits("prod_hdw_tma.dv2 vs prod_hdw", ht["dv2"], h["dv2"]))
        worst = max(worst, bits("prod_hdw_tma.dA vs prod_hdw", ht["dA"], h["dA"]))
        worst = max(worst, bits("prod_hdw_tma.dqg vs prod_hdw", ht["dqg"], h["dqg"]))
    print(f"\n  worst over all tables: max_abs={worst:.3e}  "
          f"[{'BIT-IDENTICAL' if worst == 0.0 else 'MISMATCH'}]")


def part_a2(cases):
    print("=" * 78)
    print("Part A2: handoff proof -- prod_hdw input `dw` is genuinely unused")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in SEG.items()}
    worst = 0.0
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        dw = run_core(libs["core"], inp)["dw"]
        dw0 = torch.zeros_like(dw)
        a = run_prod(libs["prod_hdw"], inp, dw)
        b = run_prod(libs["prod_hdw"], inp, dw0)
        for f in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
            worst = max(worst, bits(f"prod_hdw(dw=0).{f} vs (dw=core)", b[f], a[f]))
    print(f"\n  worst: max_abs={worst:.3e}  [{'dw IGNORED (handoff)' if worst == 0.0 else 'dw USED!'}]")


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def part_b():
    print("=" * 78)
    print("Part B: resources (registers / spill / smem)")
    print("=" * 78)
    print(f"  {'build':16s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for tag in ("prod", "prod_hdw", "prod_tma", "prod_hdw_tma"):
        so, txt = build_v(tag, SEG[tag])
        r, sp = ptxas_info(txt)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:16s} {r:6d} {sp:7d} {smem:9d}")


def part_c(cases):
    print("=" * 78)
    print("Part C: timing -- one prod_hdw launch vs one prod launch (both single-launch)")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in SEG.items()}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = run_core(libs["core"], inp)["dw"]
        torch.cuda.synchronize()

        dev_hdw, ph = profile_dev(lambda: run_prod(libs["prod_hdw"], inp, dw))
        dev_prod, pp = profile_dev(lambda: run_prod(libs["prod"], inp, dw))
        print(f"  {name:20s} prod_hdw={dev_hdw:8.1f}us   prod={dev_prod:8.1f}us   "
              f"hdw/prod={dev_hdw/dev_prod:.3f}x")


def run_dkgb(lib, inp, dw):
    lib.b2dkgb_launch.restype = ctypes.c_int
    lib.b2dkgb_launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    rc = lib.b2dkgb_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(dw), ptr(inp["A"]),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dkgb),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2dkgb_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw_out, dgk=dgk, dkgb=dkgb)


def part_e(cases):
    print("=" * 78)
    print("Part E: isolated DKGB -- in-loop (global dw) vs post-loop handoff (in-kernel dw)")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in SEG.items()}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = run_core(libs["core"], inp)["dw"]
        torch.cuda.synchronize()
        a = run_dkgb(libs["dkgb"], inp, dw)["dkgb"]
        b = run_dkgb(libs["dkgb_hdw"], inp, dw)["dkgb"]
        d = bits("dkgb_hdw.dkgb vs dkgb", b, a)
        dev_a, _ = profile_dev(lambda: run_dkgb(libs["dkgb"], inp, dw))
        dev_b, _ = profile_dev(lambda: run_dkgb(libs["dkgb_hdw"], inp, dw))
        print(f"    {name:20s} dkgb={dev_a:8.1f}us  dkgb_hdw={dev_b:8.1f}us  "
              f"hdw/base={dev_b/dev_a:.3f}x")


def wall_dev(fn, reps=30, rounds=3):
    """Event-pair wall: one pair wraps `reps` calls, 3 rounds, take the median (P7-1 convention)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        xs.append(s.elapsed_time(e) / reps * 1000.0)   # us
    xs.sort()
    return xs[len(xs) // 2]


def part_d(cases):
    print("=" * 78)
    print("Part D: wall -- event-pair wall of prod_hdw vs prod (second timing口径)")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in SEG.items()}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = run_core(libs["core"], inp)["dw"]
        torch.cuda.synchronize()
        w_hdw = wall_dev(lambda: run_prod(libs["prod_hdw"], inp, dw))
        w_prod = wall_dev(lambda: run_prod(libs["prod"], inp, dw))
        print(f"  {name:20s} prod_hdw={w_hdw:8.1f}us   prod={w_prod:8.1f}us   "
              f"hdw/prod={w_hdw/w_prod:.3f}x")


def main():
    mode = os.environ.get("B2H_MODE", "a")
    if mode == "a":
        part_a(CASES)
    elif mode == "a2":
        part_a2(CASES)
    elif mode == "b":
        part_b()
    elif mode == "c":
        part_c(CASES)
    elif mode == "d":
        part_d(CASES)
    elif mode == "e":
        part_e(CASES)


if __name__ == "__main__":
    main()
