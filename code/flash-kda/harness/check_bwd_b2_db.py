"""P7-56: the *fifth non-GEMM segment* of B2 in CUDA -- the per-token write-gate gradient

    db[o] = sum_v dvb[o,v] * v[o,v]  +  sum_k dkgb[o,k] * kg[o,k]

FLA `chunk_bwd.py` / `check_bwd_dqkg_kernel.py:180-210`: `b_db = sum_V(b_dvb*b_v)` then
`b_db += sum_k(b_dkgb*b_kg)`, with `dvb = A@dv` (P7-52 output branch) and `dkgb = A@(-dw)`
(P7-53 finalize term).  Both operands are computed in the same kernel, each owned by a per-warp
V/k column slice, so `db` is exactly a **cross-warp reduction** of two warp-private column sums.
It is a *complete* B2 output (never touched by the gate chain), so it is verifiable standalone.

  Part A  correctness: CUDA `db` vs a torch reduction over the *CUDA* `dvb`/`dkgb` (same
          operands, isolates the reduction), bit-identity across non-TMA/TMA32/TMA64, and
          cross-check vs FLA's `db` (floor set by the tf32 `dvb`/`dkgb`).
  Part B  timing: db kernel vs the same build with the db segment skipped (`B2M_NODB`), and
          vs the full Triton B2.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_db.py \
        > .../harness/check_bwd_b2_db.out.txt 2>&1
    B2DB_MODE=a | b
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# CUDA dvb/dkgb are tf32; the `db` reduction itself is fp32 scalar.  Keep torch high precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs,   # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton,
                                   fla_wy_dqkg)
from check_bwd_ws_reuse import stats, cdiv                            # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                          # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
# `db*` = OUT+DKGB+DB builds; `*_nodb` = same build with the db segment skipped (timing baseline).
VARIANTS = [
    ("core",        []),                                   # bare GEMM core -> supplies `dw`
    ("db",          ["-DB2M_OUT=1", "-DB2M_DKGB=1", "-DB2M_DB=1"]),
    ("db_nodb",     ["-DB2M_OUT=1", "-DB2M_DKGB=1", "-DB2M_DB=1", "-DB2M_NODB=1"]),
    ("db_tma",      ["-DB2M_OUT=1", "-DB2M_DKGB=1", "-DB2M_DB=1", "-DB2M_TMA=1"]),
    ("db_tma_nodb", ["-DB2M_OUT=1", "-DB2M_DKGB=1", "-DB2M_DB=1", "-DB2M_TMA=1", "-DB2M_NODB=1"]),
    ("db_tma64",    ["-DB2M_OUT=1", "-DB2M_DKGB=1", "-DB2M_DB=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]),
]
DB_TAGS = ("db", "db_tma", "db_tma64")
TIMING_TAGS = ("db_nodb", "db", "db_tma_nodb", "db_tma", "db_tma64")


def build(tag, extra):
    so = f"/tmp/b2_db_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", os.path.join(HARNESS, "b2_mma_probe.cu"),
           "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    return ctypes.CDLL(so)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


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
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                          ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                          B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2mma_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk)


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
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2db_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                         ptr(inp["do"]), ptr(inp["dv"]), ptr(dw),
                         ptr(inp["k"]), ptr(inp["g"]), ptr(inp["A"]), ptr(inp["beta"]),
                         ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dkgb),
                         ptr(dv2), ptr(dvb), ptr(db_out),
                         B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2db_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw_out, dgk=dgk, dkgb=dkgb, dv2=dv2, dvb=dvb, db=db_out)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def ref_db(inp, dvb, dkgb):
    """torch(fp32) reduction on the *given* CUDA `dvb`/`dkgb` (isolates the reduction)."""
    v = inp["v"].float()
    kg = inp["k"].float() * torch.exp2(inp["g"].float())
    return (dvb.float() * v).sum(-1) + (dkgb.float() * kg).sum(-1)


def ref_operands(inp, dw, BT):
    """torch(fp32) transcription of `dvb`, `dkgb`, `db` from the operands (per-chunk, fp32)."""
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device

    def pad(x):
        o = torch.zeros((B, NT * BT,) + x.shape[2:], device=dev, dtype=torch.float32)
        o[:, :T] = x.float()
        return o

    A4 = pad(inp["A"]).view(B, NT, BT, H, BT)                     # [b,c,o_t,h,o_A]
    b_A = A4.permute(0, 1, 3, 4, 2)                               # [b,c,h,o_A,o_t]
    v4 = pad(inp["v"]).view(B, NT, BT, H, V).permute(0, 1, 3, 2, 4)
    dv4 = pad(inp["dv"]).view(B, NT, BT, H, V).permute(0, 1, 3, 2, 4)
    k4 = pad(inp["k"]).view(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)
    g4 = pad(inp["g"]).view(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)
    dw4 = pad(dw).view(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)    # [b,c,h,o_t,k]
    kg = k4 * torch.exp2(g4)
    dvb = torch.einsum("bchtu,bchuv->bchtv", b_A, dv4)            # [b,c,h,o_A,v]
    dkgb = torch.einsum("bchtu,bchuk->bchtk", b_A, -dw4)          # [b,c,h,o_A,k]
    db = (dvb * v4).sum(-1) + (dkgb * kg).sum(-1)                 # [b,c,h,o_A]
    # back to [B,T,H,*]
    dvb = dvb.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, V)[:, :T].contiguous()
    dkgb = dkgb.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, K)[:, :T].contiguous()
    db = db.permute(0, 1, 3, 2).reshape(B, NT * BT, H)[:, :T].contiguous()
    return dict(dvb=dvb, dkgb=dkgb, db=db)


def part_a(libs, tk):
    print("=" * 96)
    print("Part A: correctness -- CUDA db = sum_v dvb*v + sum_k dkgb*kg, bit-id, vs FLA")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]
        out = {tag: run_db(libs[tag], inp, dw) for tag in DB_TAGS}
        ref = ref_db(inp, out["db"]["dvb"], out["db"]["dkgb"])
        rop = ref_operands(inp, dw, BT)
        fla = fla_wy_dqkg(inp, 1.0, BT)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        _, mr = stats("  db vs torch(same dvb/dkgb)", out["db"]["db"], ref)
        worst["torch"] = max(worst.get("torch", 0.0), mr)
        _, mr = stats("  db vs torch(from operands)", out["db"]["db"], rop["db"])
        worst["torch_ops"] = max(worst.get("torch_ops", 0.0), mr)
        _, mr = stats("  dvb vs torch(from operands)", out["db"]["dvb"], rop["dvb"])
        worst["dvb_torch"] = max(worst.get("dvb_torch", 0.0), mr)
        _, mr = stats("  dkgb vs torch(from operands)", out["db"]["dkgb"], rop["dkgb"])
        worst["dkgb_torch"] = max(worst.get("dkgb_torch", 0.0), mr)
        _, mr = stats("  db vs FLA db", out["db"]["db"], fla["db"])
        worst["vs_fla"] = max(worst.get("vs_fla", 0.0), mr)
        # sanity: the combined build still reproduces FLA's dv2 (P7-52)
        _, mrdvb = stats("  dv2 vs FLA", out["db"]["dv2"], fla["dv"])
        worst["dv2_fla"] = max(worst.get("dv2_fla", 0.0), mrdvb)
        for tag in ("db_tma", "db_tma64"):
            _, mrid = stats(f"  db vs {tag}", out["db"]["db"], out[tag]["db"])
            worst[f"vs_{tag}"] = max(worst.get(f"vs_{tag}", 0.0), mrid)
    print("\nworst max_rel over all shapes:")
    for kk in ("torch", "torch_ops", "dvb_torch", "dkgb_torch", "vs_fla", "dv2_fla",
               "vs_db_tma", "vs_db_tma64"):
        print(f"  {kk:14s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- db segment delta (vs B2M_NODB baseline) and vs full Triton B2")
    print("=" * 96)
    print(f"  {'case':18s} | {'db_nodb':>8s} {'db':>8s} | {'tma_nodb':>9s} {'db_tma':>8s} "
          f"{'db_tma64':>9s} | {'fullB2':>8s} | {'db/db_nodb':>11s} {'tma/tma_nodb':>13s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]
        for tag in TIMING_TAGS:
            run_db(libs[tag], inp, dw)
        torch.cuda.synchronize()
        dev = {}
        for tag in TIMING_TAGS:
            dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_db(lib, inp, dw), reps=40)
        dev["fullB2"], _ = profile_dev(
            lambda: wy_dqkg_triton(tk, inp, 1.0, BT, num_warps=4), reps=40)
        print(f"  {name:18s} | {dev['db_nodb']:8.1f} {dev['db']:8.1f} | "
              f"{dev['db_tma_nodb']:9.1f} {dev['db_tma']:8.1f} {dev['db_tma64']:9.1f} | "
              f"{dev['fullB2']:8.1f} | {dev['db']/dev['db_nodb']:10.2f}x "
              f"{dev['db_tma']/dev['db_tma_nodb']:12.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    tk = build_dqkg()
    mode = os.environ.get("B2DB_MODE", "")
    if mode in ("", "a"):
        part_a(libs, tk)
    if mode in ("", "b"):
        part_b(libs, tk)
    print("\nDONE")


if __name__ == "__main__":
    main()
