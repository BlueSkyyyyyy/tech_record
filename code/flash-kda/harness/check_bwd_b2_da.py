"""P7-54: the *third non-GEMM segment* of B2 in CUDA -- the two GEMM contributions to `dA`
*before* the final `-A^T diag(beta) A^T` sandwich:

    dApre[o_t,o_u] = sum_v dv[o_t,v]*v[o_u,v]     (base;  FLA `b_dA += tl.dot(b_dv, v^T)`)
                   + sum_k dw[o_t,k]*kg[o_u,k]     (FLA `b_dA += tl.dot(b_dw, kg^T)`)

with `kg = k*exp2(g)` and `dw` the GEMM core's `dw = dv @ h` output (FLA `chunk_bwd.py:264/280-281`,
`check_bwd_dqkg_kernel.py:168/180-181`).  Both contractions are *intra-chunk* (no cross-chunk
coupling), so this segment is verifiable standalone like P7-52's `dvb` / P7-53's `dkgb`.

  Part A  correctness: CUDA `dApre` vs torch(fp32) using the *same* `dw` operand (from the CUDA
          GEMM core, hence also tf32), and bit-identity across non-TMA/TMA32/TMA64.
  Part B  timing: dA kernel (GEMM core + segment) vs the GEMM-only core (delta) and vs the full
          Triton B2.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_da.py \
        > .../harness/check_bwd_b2_da.out.txt 2>&1
    B2DA_MODE=a | b
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# CUDA segment is tf32 (matches `tl.dot`); true-fp32 torch reference.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs,   # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton)
from check_bwd_ws_reuse import stats, cdiv                            # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                          # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
VARIANTS = [
    ("core", []),
    ("core_tma", ["-DB2M_TMA=1"]),
    ("da", ["-DB2M_DA=1"]),
    ("da_tma", ["-DB2M_DA=1", "-DB2M_TMA=1"]),
    ("da_tma64", ["-DB2M_DA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]),
]


def build(tag, extra):
    so = f"/tmp/b2_da_{tag}.so"
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


def run_da(lib, inp):
    lib.b2da_launch.restype = ctypes.c_int
    lib.b2da_launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2da_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                         ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                         ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dApre),
                         B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2da_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw_out, dgk=dgk, dApre=dApre)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def ref_dapre(inp, dw, BT):
    """torch(fp32) reference for exactly the `dApre` segment, using the *given* `dw` operand.

    base[o_t,o_u] = sum_v dv[o_t,v]*v[o_u,v]  (FLA line 264);
    dApre        -= sum_k dw[o_t,k]*kg[o_u,k],  kg=k*exp2(g)  (FLA negates dw, line 280-281).

    NOTE (P7-55 correction): FLA does `b_dw = -b_dw; b_dA += b_dw @ kg^T`, i.e. the `dw@kg^T`
    term enters `dApre` *negated*.  P7-54 originally added it; the standalone P7-54 check did
    not catch it (both sides used `+`).  Only the full `dA` sandwich vs FLA exposes it.
    """
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dv, vv, k, g = inp["dv"].float(), inp["v"].float(), inp["k"].float(), inp["g"].float()
    dwd = dw.float()
    kg = k * torch.exp2(g)

    def pad4(x):   # [B,T,H,W] -> [B,NT,H,BT,W]
        o = torch.zeros((B, NT * BT, H) + x.shape[3:], device=dev, dtype=torch.float32)
        o[:, :T] = x
        return o.view((B, NT, BT, H) + x.shape[3:]).permute(0, 1, 3, 2, 4).contiguous()

    dv4, v4 = pad4(dv), pad4(vv)          # [B,NT,H,BT,V]
    dw4, kg4 = pad4(dwd), pad4(kg)        # [B,NT,H,BT,K]
    base = torch.matmul(dv4, v4.transpose(-1, -2))     # [B,NT,H,BT(o_t),BT(o_u)]
    dwkg = torch.matmul(dw4, kg4.transpose(-1, -2))    # [B,NT,H,BT(o_t),BT(o_u)]
    dapre = (base - dwkg).permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    return dict(dApre=dapre)


def part_a(libs, tk):
    print("=" * 96)
    print("Part A: correctness -- CUDA dApre = dv@v^T + dw@kg^T vs torch(fp32), dw from CUDA core")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]          # tf32 dw operand (same for all)
        ref = ref_dapre(inp, dw, BT)
        out = {tag: run_da(libs[tag], inp)
               for tag in ("da", "da_tma", "da_tma64")}
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for tag in ("da", "da_tma", "da_tma64"):
            _, mr = stats(f"  {tag} dApre vs torch", out[tag]["dApre"], ref["dApre"])
            worst[f"{tag}_dapre"] = max(worst.get(f"{tag}_dapre", 0.0), mr)
        _, mrid = stats("  da vs da_tma64 ", out["da"]["dApre"], out["da_tma64"]["dApre"])
        _, mrit = stats("  da vs da_tma   ", out["da"]["dApre"], out["da_tma"]["dApre"])
        worst["da_vs_tma64"] = max(worst.get("da_vs_tma64", 0.0), mrid)
        worst["da_vs_tma"] = max(worst.get("da_vs_tma", 0.0), mrit)
    print("\nworst max_rel over all shapes:")
    for kk in ("da_dapre", "da_tma_dapre", "da_tma64_dapre",
               "da_vs_tma", "da_vs_tma64"):
        print(f"  {kk:20s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- dA kernel vs GEMM-only core and vs full Triton B2")
    print("=" * 96)
    iscore = lambda tag: tag.startswith("core")
    print(f"  {'case':18s} | {'core':>7s} {'core_tma':>8s} | {'da':>7s} {'da_tma':>7s} "
          f"{'da_tma64':>8s} | {'fullB2':>7s} | {'da/core':>8s} {'da_tma/ctma':>12s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        for tag in libs:
            if iscore(tag):
                run_core(libs[tag], inp)
            else:
                run_da(libs[tag], inp)
        torch.cuda.synchronize()
        dev = {}
        for tag in libs:
            if iscore(tag):
                dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_core(lib, inp), reps=40)
            else:
                dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_da(lib, inp), reps=40)
        fn2 = lambda: wy_dqkg_triton(tk, inp, 1.0, BT, num_warps=4)
        dev["fullB2"], _ = profile_dev(fn2, reps=40)
        print(f"  {name:18s} | {dev['core']:7.1f} {dev['core_tma']:8.1f} | {dev['da']:7.1f} "
              f"{dev['da_tma']:7.1f} {dev['da_tma64']:8.1f} | {dev['fullB2']:7.1f} | "
              f"{dev['da']/dev['core']:7.2f}x {dev['da_tma']/dev['core_tma']:11.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    tk = build_dqkg()
    mode = os.environ.get("B2DA_MODE", "")
    if mode in ("", "a"):
        part_a(libs, tk)
    if mode in ("", "b"):
        part_b(libs, tk)
    print("\nDONE")


if __name__ == "__main__":
    main()
