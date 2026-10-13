"""P7-55: the *fourth non-GEMM segment* of B2 in CUDA -- the final `dA` sandwich,
the remaining `dA` post-op §55.6/§56.6/§57.6 left open:

    dAm[o_t,o_u] = (o_t>o_u & valid) ? dApre[o_t,o_u] * beta[o_u] : 0
    dA           = where(o_t>o_u & valid, -(b_A @ (dAm @ b_A)), 0)

with `b_A[p,j] = A[o_t=j, o_A=p]` (the FLA/Triton `b_A` orientation,
`check_bwd_dqkg_kernel.py:240`) and `dApre` P7-54's pre-sandwich `dv@v^T + dw@kg^T`.
This is the *complete* B2 `dA` output.  It is a pure [BT,BT] fp32 op on data already in
registers/smem (no re-read of `dApre`), so:

  Part A  correctness: CUDA `dA` vs a torch sandwich on the *same* CUDA `dApre`
          (bit-level: both are fp32 scalar), bit-identity across non-TMA/TMA32/TMA64,
          and cross-check vs FLA's final `dA` (floor set by the `dw` tf32 difference, §57.2).
  Part B  timing: sand kernel (GEMM core + dApre + sandwich) vs the dA kernel (delta),
          and vs the full Triton B2.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_sand.py \
        > .../harness/check_bwd_b2_sand.out.txt 2>&1
    B2S_MODE=a | b
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# CUDA segment is fp32 scalar for the sandwich (no tf32); true-fp32 torch reference.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
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
# `da*` = P7-54 baselines (dApre only); `sand*` = this round (+ final dA).
VARIANTS = [
    ("da",         ["-DB2M_DA=1"]),
    ("da_tma",     ["-DB2M_DA=1", "-DB2M_TMA=1"]),
    ("sand",       ["-DB2M_DA=1", "-DB2M_SAND=1"]),
    ("sand_tma",   ["-DB2M_DA=1", "-DB2M_SAND=1", "-DB2M_TMA=1"]),
    ("sand_tma64", ["-DB2M_DA=1", "-DB2M_SAND=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]),
]
SAND_TAGS = ("sand", "sand_tma", "sand_tma64")
DA_TAGS = ("da", "da_tma")


def build(tag, extra):
    so = f"/tmp/b2_sand_{tag}.so"
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


def run_da(lib, inp):
    lib.b2da_launch.restype = ctypes.c_int
    lib.b2da_launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2da_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                         ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                         ptr(dq), ptr(dk), ptr(dw), ptr(dgk), ptr(dApre),
                         B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2da_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk, dApre=dApre)


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
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2sand_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                           ptr(inp["A"]), ptr(inp["beta"]),
                           ptr(dq), ptr(dk), ptr(dw), ptr(dgk), ptr(dApre), ptr(dA),
                           B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2sand_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk, dApre=dApre, dA=dA)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def ref_sand(inp, dApre, BT):
    """torch(fp32) reference: the literal Triton op sequence on the *given* `dApre`."""
    B, T, H = inp["q"].shape[:3]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    A = inp["A"].float()          # [B,T,H,BT]
    beta = inp["beta"].float()    # [B,T,H]
    dap = dApre.float()           # [B,T,H,BT]

    def padT(x):
        o = torch.zeros((B, NT * BT,) + x.shape[2:], device=dev, dtype=torch.float32)
        o[:, :T] = x
        return o

    A4 = padT(A).view(B, NT, BT, H, BT)             # [b,c,o_t,h,o_A]
    b_A = A4.permute(0, 1, 3, 4, 2)                 # [b,c,h,o_A,o_t]
    dAp4 = padT(dap).view(B, NT, BT, H, BT).permute(0, 1, 3, 2, 4)   # [b,c,h,o_t,o_u]
    b4 = padT(beta.unsqueeze(-1)).view(B, NT, BT, H).permute(0, 1, 3, 2)  # [b,c,h,o_t]

    mt = (torch.arange(NT * BT, device=dev) < T).view(NT, BT)
    o = torch.arange(BT, device=dev)
    m2 = (o[:, None] > o[None, :]) & mt[:, :, None] & mt[:, None, :]   # [NT,BT,BT]

    b4u = b4.unsqueeze(3)                           # [b,c,h,1,o_t] -> broadcasts on o_u
    dAm = torch.where(m2[None, :, None, :, :], dAp4 * b4u, torch.zeros(1, device=dev))
    X1 = torch.matmul(dAm, b_A)                     # [b,c,h,o_t,o_t]
    X2 = torch.matmul(b_A, X1)                      # [b,c,h,o_A,o_t]
    dA = torch.where(m2[None, :, None, :, :], -X2, torch.zeros(1, device=dev))
    dA = dA.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    return dict(dA=dA)


def part_a(libs, tk):
    print("=" * 96)
    print("Part A: correctness -- CUDA dA = -A^T diag(beta) A^T (dApre) vs torch, bit-id, vs FLA")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        out = {tag: run_sand(libs[tag], inp) for tag in SAND_TAGS}
        dApre = out["sand"]["dApre"]                 # CUDA dApre (same for all stage layouts)
        ref = ref_sand(inp, dApre, BT)
        fla = fla_wy_dqkg(inp, 1.0, BT)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for tag in SAND_TAGS:
            _, mr = stats(f"  {tag} dA vs torch(same dApre)", out[tag]["dA"], ref["dA"])
            worst[f"{tag}_torch"] = max(worst.get(f"{tag}_torch", 0.0), mr)
        _, mr = stats("  sand dA vs FLA final dA", out["sand"]["dA"], fla["dA"])
        worst["vs_fla"] = max(worst.get("vs_fla", 0.0), mr)
        for tag in ("sand_tma", "sand_tma64"):
            _, mrid = stats(f"  sand dA vs {tag}", out["sand"]["dA"], out[tag]["dA"])
            worst[f"sand_vs_{tag}"] = max(worst.get(f"sand_vs_{tag}", 0.0), mrid)
            _, mridp = stats(f"  sand dApre vs {tag}", out["sand"]["dApre"], out[tag]["dApre"])
            worst[f"dapre_vs_{tag}"] = max(worst.get(f"dapre_vs_{tag}", 0.0), mridp)
    print("\nworst max_rel over all shapes:")
    for kk in ("sand_torch", "sand_tma_torch", "sand_tma64_torch", "vs_fla",
               "sand_vs_sand_tma", "sand_vs_sand_tma64",
               "dapre_vs_sand_tma", "dapre_vs_sand_tma64"):
        print(f"  {kk:24s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- sand kernel vs dA kernel (delta) and vs full Triton B2")
    print("=" * 96)
    print(f"  {'case':18s} | {'da':>7s} {'da_tma':>7s} | {'sand':>7s} {'sand_tma':>8s} "
          f"{'sand_tma64':>10s} | {'fullB2':>7s} | {'sand/da':>8s} {'sand_tma/da_tma':>15s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        for tag in libs:
            (run_da if tag in DA_TAGS else run_sand)(libs[tag], inp)
        torch.cuda.synchronize()
        dev = {}
        for tag in libs:
            fn = run_da if tag in DA_TAGS else run_sand
            dev[tag], _ = profile_dev(lambda lib=libs[tag], f=fn: f(lib, inp), reps=40)
        fn2 = lambda: wy_dqkg_triton(tk, inp, 1.0, BT, num_warps=4)
        dev["fullB2"], _ = profile_dev(fn2, reps=40)
        print(f"  {name:18s} | {dev['da']:7.1f} {dev['da_tma']:7.1f} | {dev['sand']:7.1f} "
              f"{dev['sand_tma']:8.1f} {dev['sand_tma64']:10.1f} | {dev['fullB2']:7.1f} | "
              f"{dev['sand']/dev['da']:7.2f}x {dev['sand_tma']/dev['da_tma']:14.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    tk = build_dqkg()
    mode = os.environ.get("B2S_MODE", "")
    if mode in ("", "a"):
        part_a(libs, tk)
    if mode in ("", "b"):
        part_b(libs, tk)
    print("\nDONE")


if __name__ == "__main__":
    main()
