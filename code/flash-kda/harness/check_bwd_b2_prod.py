"""P7-60: the **single production launch** of the K2-shaped CUDA B2 -- all six non-GEMM segments
(OUT + DKGB + DA + SAND + DB + GATE) plus the GEMM core in **one** kernel invocation, instead of
the segment-per-launcher builds the earlier rounds (P7-52/53/54/55/56/57) used for isolated
verification.  `-DB2M_PROD` forces the six flags and adds `b2prod_launch`.

The segment bodies are untouched; the only shared-code changes are (a) allowing the flag
combination and (b) declaring the shared `beta` operand once.  So every output must be
**bit-identical** to the corresponding isolated build run on the same operands -- that is the
whole correctness claim of this round (merging changed nothing), and every isolated output was
already anchored to torch/FLA in its own round.

  Part A  correctness: for each case, prod's `dv2/dvb/dkgb/db/dApre/dA/dqg/dkg/dg/dw/dgk` vs the
          isolated `db` build (dv2/dvb/dkgb/db), `sand` build (dApre/dA), `gate` build
          (dqg/dkg/dg) and `core` build (dw/dgk) -- bit-identity -- and the TMA variant
          (`-DB2M_TMA`) vs the cp.async prod build (bit-identity across stage layouts).
  Part B  resources: registers/spills/smem of prod vs each segment build.
  Part C  timing: one prod launch vs the sum of the three launches the old design needed to emit
          all outputs (`db` + `sand` + `gate`, each carrying its own copy of the GEMM core).
  Part D  torch sanity: prod's gate triple vs a torch transcription of the gate chain.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_prod.py \
        > .../harness/check_bwd_b2_prod.out.txt 2>&1
    B2P_MODE=a | b | c | d
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

# isolated reference builds
SEG = {
    "core": [],
    "db":   ["-DB2M_OUT=1", "-DB2M_DKGB=1", "-DB2M_DB=1"],
    "sand": ["-DB2M_DA=1", "-DB2M_SAND=1"],
    "gate": ["-DB2M_DKGB=1", "-DB2M_GATE=1"],
    "prod": ["-DB2M_PROD=1"],
    "prod_tma": ["-DB2M_PROD=1", "-DB2M_TMA=1"],
}


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build_v(tag, extra):
    """Compile and return (path, ptxas-text)."""
    so = f"/tmp/b2_prod_{tag}.so"
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


# --------------------------------------------------------------------- launchers
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


def run_db(lib, inp, dw):
    lib.b2db_launch.restype = ctypes.c_int
    lib.b2db_launch.argtypes = [ctypes.c_void_p] * 19 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dvb = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db_out = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    rc = lib.b2db_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                         ptr(inp["do"]), ptr(inp["dv"]), ptr(dw),
                         ptr(inp["k"]), ptr(inp["g"]), ptr(inp["A"]), ptr(inp["beta"]),
                         ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dkgb),
                         ptr(dv2), ptr(dvb), ptr(db_out),
                         B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2db_launch rc={rc}")
    return dict(dw=dw_out, dgk=dgk, dkgb=dkgb, dv2=dv2, dvb=dvb, db=db_out)


def run_sand(lib, inp):
    lib.b2sand_launch.restype = ctypes.c_int
    lib.b2sand_launch.argtypes = [ctypes.c_void_p] * 16 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    rc = lib.b2sand_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(inp["A"]), ptr(inp["beta"]),
                           ptr(dq), ptr(dk), ptr(dw), ptr(dgk), ptr(dApre), ptr(dA),
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2sand_launch rc={rc}")
    return dict(dApre=dApre, dA=dA)


def run_gate(lib, inp, dw):
    lib.b2gate_launch.restype = ctypes.c_int
    lib.b2gate_launch.argtypes = ([ctypes.c_void_p] * 20 + [ctypes.c_float] +
                                  [ctypes.c_int] * 4 + [ctypes.c_void_p])
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dqg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dkg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    rc = lib.b2gate_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(dw), ptr(inp["A"]),
                           ptr(inp["beta"]), ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dkgb),
                           ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                           B, T, H, NT, stream())
    if rc != 0:
        raise RuntimeError(f"b2gate_launch rc={rc}")
    return dict(dqg=dqg, dkg=dkg, dg=dg, dw=dw_out, dgk=dgk, dkgb=dkgb)


def run_prod(lib, inp, dw):
    lib.b2prod_launch.restype = ctypes.c_int
    lib.b2prod_launch.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] +
                                  [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 +
                                  [ctypes.c_void_p])
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)   # scratch (gate suppresses raw)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)   # scratch
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


# ------------------------------------------------------------------------- parts
def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print(f"    {name:28s} max_abs={d:.3e}  [{flag}]")
    return d


def run_isolated(inp):
    libs = {t: load(build(t, e)) for t, e in SEG.items()}
    core = run_core(libs["core"], inp)
    dw = core["dw"]
    db = run_db(libs["db"], inp, dw)
    sand = run_sand(libs["sand"], inp)
    gate = run_gate(libs["gate"], inp, dw)
    prod = run_prod(libs["prod"], inp, dw)
    prod_tma = run_prod(libs["prod_tma"], inp, dw)
    return core, db, sand, gate, prod, prod_tma


def part_a(cases):
    print("=" * 78)
    print("Part A: prod (one launch) vs isolated segment builds -- bit-identity")
    print("=" * 78)
    worst = 0.0
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        core, db, sand, gate, prod, prod_tma = run_isolated(inp)
        worst = max(worst, bits("prod.dv2   vs db", prod["dv2"], db["dv2"]))
        worst = max(worst, bits("prod.dvb   vs db", prod["dvb"], db["dvb"]))
        worst = max(worst, bits("prod.dkgb  vs db", prod["dkgb"], db["dkgb"]))
        worst = max(worst, bits("prod.db    vs db", prod["db"], db["db"]))
        worst = max(worst, bits("prod.dApre vs sand", prod["dApre"], sand["dApre"]))
        worst = max(worst, bits("prod.dA    vs sand", prod["dA"], sand["dA"]))
        worst = max(worst, bits("prod.dqg   vs gate", prod["dqg"], gate["dqg"]))
        worst = max(worst, bits("prod.dkg   vs gate", prod["dkg"], gate["dkg"]))
        worst = max(worst, bits("prod.dg    vs gate", prod["dg"], gate["dg"]))
        worst = max(worst, bits("prod.dw    vs core", prod["dw"], core["dw"]))
        worst = max(worst, bits("prod.dgk   vs core", prod["dgk"], core["dgk"]))
        # TMA layout must reproduce the cp.async prod bit-for-bit
        worst = max(worst, bits("prod_tma.dv2  vs prod", prod_tma["dv2"], prod["dv2"]))
        worst = max(worst, bits("prod_tma.dqg  vs prod", prod_tma["dqg"], prod["dqg"]))
        worst = max(worst, bits("prod_tma.dA   vs prod", prod_tma["dA"], prod["dA"]))
    print(f"\n  worst over all tables: max_abs={worst:.3e}  "
          f"[{'BIT-IDENTICAL' if worst == 0.0 else 'MISMATCH'}]")


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def part_b():
    print("=" * 78)
    print("Part B: resources (registers / spill / smem)")
    print("=" * 78)
    print(f"  {'build':12s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for tag in ("core", "db", "sand", "gate", "prod", "prod_tma"):
        so, txt = build_v(tag, SEG[tag])
        r, sp = ptxas_info(txt)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:12s} {r:6d} {sp:7d} {smem:9d}")


def part_c(cases):
    print("=" * 78)
    print("Part C: timing -- one prod launch vs the 3 launches the old design needed")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in SEG.items()}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = run_core(libs["core"], inp)["dw"]
        torch.cuda.synchronize()

        def f_prod():
            run_prod(libs["prod"], inp, dw)

        def f_old():
            run_db(libs["db"], inp, dw)
            run_sand(libs["sand"], inp)
            run_gate(libs["gate"], inp, dw)

        dev_prod, pr = profile_dev(f_prod)
        dev_old, po = profile_dev(f_old)
        n_prod = sum(ev.count for ev in pr.key_averages() if "b2mma_kernel" in ev.key)
        n_old = sum(ev.count for ev in po.key_averages() if "b2mma_kernel" in ev.key)
        print(f"  {name:20s} prod={dev_prod:8.1f}us ({n_prod} b2 kernels)   "
              f"db+sand+gate={dev_old:8.1f}us ({n_old} b2 kernels)   "
              f"prod/old={dev_prod/dev_old:.3f}x")


def part_d(cases):
    print("=" * 78)
    print("Part D: torch sanity -- prod's gate triple vs the literal gate chain on the same raw")
    print("=" * 78)
    from check_bwd_b2_gate import ref_gate
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        libs = {t: load(build(t, e)) for t, e in SEG.items()}
        core = run_core(libs["core"], inp)
        prod = run_prod(libs["prod"], inp, core["dw"])
        # raw dq/dk/dgk/dkgb from the isolated core/gate builds (same operands)
        libc = libs["core"]
        Br, Tr, Hr, Kr = inp["q"].shape
        NT = cdiv(T, BT)
        dev = inp["q"].device
        dqr = torch.empty(Br, Tr, Hr, Kr, device=dev, dtype=torch.float32)
        dkr = torch.empty(Br, Tr, Hr, Kr, device=dev, dtype=torch.float32)
        dwr = torch.empty(Br, Tr, Hr, Kr, device=dev, dtype=torch.float32)
        dgkr = torch.empty(Br, NT, Hr, Kr, device=dev, dtype=torch.float32)
        libc.b2mma_launch.restype = ctypes.c_int
        libc.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        libc.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                          ptr(inp["do"]), ptr(inp["dv"]), ptr(dqr), ptr(dkr), ptr(dwr), ptr(dgkr),
                          Br, Tr, Hr, NT, stream())
        ref = ref_gate(inp, dqr, dkr, dgkr, prod["dkgb"])
        bits("prod.dqg  vs torch", prod["dqg"], ref["dqg"])
        bits("prod.dkg  vs torch", prod["dkg"], ref["dkg"])
        bits("prod.dg   vs torch", prod["dg"], ref["dg"])
        # dv2 = beta*(A@dv): torch on the same operands (pad to a whole number of chunks)
        A = inp["A"].float()
        beta = inp["beta"].float()
        dv = inp["dv"].float()
        Apad = torch.zeros(B, NT * BT, H, BT, device=dev, dtype=torch.float32)
        Apad[:, :T] = A
        dvpad = torch.zeros(B, NT * BT, H, dv.shape[-1], device=dev, dtype=torch.float32)
        dvpad[:, :T] = dv
        A4 = Apad.view(B, NT, BT, H, BT)                    # [b,c,o_t,h,o_A]
        dvv = dvpad.view(B, NT, BT, H, -1)
        dvb_ref = torch.einsum("bcthp,bcthv->bchpv", A4, dvv)    # contract o_t -> [b,c,h,o_A,v]
        dvb_ref = dvb_ref.permute(0, 1, 3, 2, 4).reshape(
            B, NT * BT, H, dv.shape[-1])[:, :T]
        dv2_ref = dvb_ref * beta.unsqueeze(-1)
        bits("prod.dv2  vs torch", prod["dv2"], dv2_ref)


def main():
    mode = os.environ.get("B2P_MODE", "a")
    if mode == "a":
        part_a(CASES)
    elif mode == "b":
        part_b()
    elif mode == "c":
        part_c(CASES)
    elif mode == "d":
        part_d(CASES)


if __name__ == "__main__":
    main()
