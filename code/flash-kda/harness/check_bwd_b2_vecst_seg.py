"""P7-59: vectorize the **non-GEMM B2 segment stores** in the K2-shaped CUDA B2 core.

P7-48 (§51) established, on the B2 GEMM core, that the default stores were 25 scalar
32-bit `STG.E` that each cover only half a 32-B sector (`l1tex st` sectors = 1.98x the
data sectors), and that synthesizing 64-bit `float2` stores (`-DB2M_VECST`) restores the
1.0x ratio bit-identically and is free-to-positive (0.87-0.99x, small shapes 10%+).

Every subsequent non-GEMM segment (P7-52 output branch `dv2/dvb`, P7-53 `dkgb`, P7-56
`db`, P7-57 gate chain `dqg/dkg/dg`) was built on top of that core but its own stores
were left **scalar** -- each round's "边界" line reads "本段 store 未接 `-DB2M_VECST`".
This script closes that gap: a `st2()` helper (`b2_mma_probe.cu`) emits one `STG.E.64`
under `B2M_VECST` (two 32-bit otherwise) and is now used by the output branch
(`dv2/dvb`), `dkgb`, and the gate chain (`dqg/dkg/dg`, plus `dg`).  The `[BT,BT]` `dA`
stores are left scalar (small, one-thread-per-entry layout).

  Part A  correctness: base vs vecst per segment -> **bit-identical** (`max_abs=0`) on
          every output, all shapes (incl. non-divisible T=250); gate build also
          cross-checked against FLA (tf32 floor, unchanged from P7-57).
  Part B  timing: CUPTI `dev` base vs vecst per segment, and the gate segment delta.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_vecst_seg.py \
        > .../harness/check_bwd_b2_vecst_seg.out.txt 2>&1
    B2V_MODE=a | b
"""
import ctypes
import math
import os
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

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs, fla_wy_dqkg  # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                                       # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                                    # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
SCALE = 1.0 / math.sqrt(128)

# (tag, macro flags, launcher) -- each built twice (base / +VECST)
SEGS = [
    ("out",  ["-DB2M_OUT=1"],                                "b2out_launch"),
    ("dkgb", ["-DB2M_DKGB=1"],                               "b2dkgb_launch"),
    ("db",   ["-DB2M_OUT=1", "-DB2M_DKGB=1", "-DB2M_DB=1"],  "b2db_launch"),
    ("gate", ["-DB2M_DKGB=1", "-DB2M_GATE=1"],               "b2gate_launch"),
]


def build(tag, extra, vecst):
    so = f"/tmp/b2_vecst_{tag}_{'v' if vecst else 'b'}.so"
    flags = list(extra) + (["-DB2M_VECST=1"] if vecst else [])
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + flags + ["-lcuda", os.path.join(HARNESS, "b2_mma_probe.cu"),
           "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag} vecst={vecst}")
    return ctypes.CDLL(so)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def empty(shape, dev):
    return torch.empty(*shape, device=dev, dtype=torch.float32)


def alloc_out(tag, inp):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    o = {}
    if tag in ("out", "db"):
        o["dv2"] = empty((B, T, H, V), dev)
        o["dvb"] = empty((B, T, H, V), dev)
    if tag in ("dkgb", "db", "gate"):
        o["dkgb"] = empty((B, T, H, K), dev)
    if tag in ("out", "dkgb", "db", "gate"):
        o["dq"] = empty((B, T, H, K), dev)
        o["dk"] = empty((B, T, H, K), dev)
        o["dw"] = empty((B, T, H, K), dev)
        o["dgk"] = empty((B, NT, H, K), dev)
    if tag == "db":
        o["db"] = empty((B, T, H), dev)
    if tag == "gate":
        o["dqg"] = empty((B, T, H, K), dev)
        o["dkg"] = empty((B, T, H, K), dev)
        o["dg"] = empty((B, T, H, K), dev)
    return o


def run(tag, lib, inp):
    """Dispatch to the segment's dedicated launcher; returns its output dict."""
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    o = alloc_out(tag, inp)
    P = ctypes.c_void_p
    C = ctypes.c_int
    if tag == "out":
        lib.b2out_launch.restype = C
        lib.b2out_launch.argtypes = [P] * 14 + [C] * 4 + [P]
        rc = lib.b2out_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                              ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["A"]), ptr(inp["beta"]),
                              ptr(o["dq"]), ptr(o["dk"]), ptr(o["dw"]), ptr(o["dgk"]),
                              ptr(o["dv2"]), ptr(o["dvb"]), B, T, H, NT, stream)
    elif tag == "dkgb":
        lib.b2dkgb_launch.restype = C
        lib.b2dkgb_launch.argtypes = [P] * 13 + [C] * 4 + [P]
        rc = lib.b2dkgb_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                               ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["dw"]), ptr(inp["A"]),
                               ptr(o["dq"]), ptr(o["dk"]), ptr(o["dw"]), ptr(o["dgk"]),
                               ptr(o["dkgb"]), B, T, H, NT, stream)
    elif tag == "db":
        lib.b2db_launch.restype = C
        lib.b2db_launch.argtypes = [P] * 19 + [C] * 4 + [P]
        rc = lib.b2db_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                             ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["dw"]),
                             ptr(inp["k"]), ptr(inp["g"]), ptr(inp["A"]), ptr(inp["beta"]),
                             ptr(o["dq"]), ptr(o["dk"]), ptr(o["dw"]), ptr(o["dgk"]),
                             ptr(o["dkgb"]), ptr(o["dv2"]), ptr(o["dvb"]), ptr(o["db"]),
                             B, T, H, NT, stream)
    elif tag == "gate":
        lib.b2gate_launch.restype = C
        lib.b2gate_launch.argtypes = [P] * 20 + [ctypes.c_float] + [C] * 4 + [P]
        rc = lib.b2gate_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                               ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["dw"]),
                               ptr(inp["A"]), ptr(inp["beta"]), ptr(inp["q"]),
                               ptr(inp["k"]), ptr(inp["g"]),
                               ptr(o["dq"]), ptr(o["dk"]), ptr(o["dw"]), ptr(o["dgk"]),
                               ptr(o["dkgb"]), ptr(o["dqg"]), ptr(o["dkg"]), ptr(o["dg"]),
                               SCALE, B, T, H, NT, stream)
    else:
        raise ValueError(tag)
    if rc != 0:
        raise RuntimeError(f"{tag} rc={rc}")
    return o


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def run_core_dw(lib_core, inp):
    """`dkgb`/`db` consume the GEMM core's `dw = dv @ h`; obtain it from the plain core build
    so both the base and vecst variants receive the *same* operand."""
    lib_core.b2mma_launch.restype = ctypes.c_int
    lib_core.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = empty((B, T, H, K), dev)
    dk = empty((B, T, H, K), dev)
    dw = empty((B, T, H, K), dev)
    dgk = empty((B, NT, H, K), dev)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib_core.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                               ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw),
                               ptr(dgk), B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"core rc={rc}")
    return dw


# outputs compared per segment
OUTS = {
    "out": ["dq", "dk", "dw", "dv2", "dvb"],
    "dkgb": ["dq", "dk", "dw", "dkgb"],
    "db": ["dv2", "dvb", "dkgb", "db"],
    "gate": ["dqg", "dkg", "dg"],
}


def part_a(libs):
    print("=" * 96)
    print("Part A: base vs vecst per segment -- expect bit-identical (max_abs=0), all shapes")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        inp["dw"] = run_core_dw(libs[("core", False)], inp)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for tag, _, _ in SEGS:
            ob = run(tag, libs[(tag, False)], inp)
            ov = run(tag, libs[(tag, True)], inp)
            for fld in OUTS[tag]:
                _, ma = stats(f"  {tag}.{fld} base vs vecst", ob[fld], ov[fld])
                worst[f"{tag}.{fld}"] = max(worst.get(f"{tag}.{fld}", 0.0), ma)
        # gate vs FLA (semantic sanity; expect the P7-57 tf32 floor ~1e-3)
        if "gate" in OUTS:
            og = run("gate", libs[("gate", True)], inp)
            fla = fla_wy_dqkg(inp, SCALE, BT)
            for fld, key in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg")):
                _, mr = stats(f"  gate.{fld} vecst vs FLA", og[fld], fla[key])
                worst[f"gatefla.{fld}"] = max(worst.get(f"gatefla.{fld}", 0.0), mr)
    print("\nworst max_abs over all shapes (base vs vecst; expect 0):")
    for tag, _, _ in SEGS:
        for fld in OUTS[tag]:
            print(f"  {tag}.{fld:5s} {worst.get(f'{tag}.{fld}', 0.0):.3e}")
    print("\nworst max_rel (gate vecst vs FLA; expect tf32 floor ~1e-3):")
    for fld in ("dqg", "dkg", "dg"):
        print(f"  {fld:5s} {worst.get(f'gatefla.{fld}', 0.0):.3e}")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: CUPTI dev (us) base vs vecst per segment, plus gate-segment delta")
    print("=" * 96)
    for tag, _, _ in SEGS:
        print(f"\n# segment: {tag}")
        print(f"  {'case':18s} | {'base':>9s} {'vecst':>9s} | {'vecst/base':>10s}")
        for (name, B, T, H, _) in CASES:
            inp = make(B, T, H)
            inp["dw"] = run_core_dw(libs[("core", False)], inp)
            run(tag, libs[(tag, False)], inp)
            run(tag, libs[(tag, True)], inp)
            torch.cuda.synchronize()
            db, _ = profile_dev(lambda: run(tag, libs[(tag, False)], inp), reps=40)
            dv, _ = profile_dev(lambda: run(tag, libs[(tag, True)], inp), reps=40)
            print(f"  {name:18s} | {db:9.1f} {dv:9.1f} | {dv/db:9.3f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    libs[("core", False)] = build("core", [], False)
    for tag, extra, _ in SEGS:
        libs[(tag, False)] = build(tag, extra, False)
        libs[(tag, True)] = build(tag, extra, True)
    mode = os.environ.get("B2V_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
