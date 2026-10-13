"""P7-52: the *first non-GEMM segment* of B2 in CUDA -- the `i_k==0` **output branch**.

§49.7 (`docs/07`) named the next real engineering step for the K2-shaped CUDA B2:

    "把本核作为 GEMM 段基座，向其中补非 GEMM 段（同 CTA 内交接 `dA`/`dv2`/`db`）".

This round lands the first, cleanest piece: the output branch

    dvb[o_A, v] = sum_{o_t} Akk[o_t, o_A] * dv[o_t, v]        (== A @ dv)
    dv2[o_A, v] = beta[o_A] * dvb[o_A, v]

`dv2` is a *complete* B2 output (the torch golden / FLA never modify it after the
`i_k==0` branch -- `check_bwd_dqkg_kernel.py:294/297`), so it is verifiable standalone;
`dvb` is emitted too so the `A @ dv` GEMM is checked directly.

Design note (why this segment is cheap to add to the streaming K2 shape): unlike the
three V-reduction GEMMs of the core, here the *contraction* is over chunk rows `o_t` and
the *output row* is also a chunk row `o_A`.  So `Akk` (`[B,T,H,BT]`, contiguous in `o_A`)
feeds the mma A-operand with a plain index remap -- **no smem transpose** -- and the
already-staged `dv` tile (`[BT, BVK]`) is the B-operand.  The operand roles were verified
three ways below.

  Part A  correctness: CUDA `dv2`/`dvb` vs torch(fp32) and vs the full Triton B2 (`dv2`).
  Part B  timing: output-branch kernel (GEMM + branch) vs the GEMM-only core (delta) and
          vs the full Triton B2; across `BVK`, TMA/non-TMA, vecst.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_out.py \
        > .../harness/check_bwd_b2_out.out.txt 2>&1
    B2O_MODE=a | b
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

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs,   # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton)
from check_bwd_ws_reuse import stats, cdiv                             # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev               # noqa: E402

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
    ("out", ["-DB2M_OUT=1"]),
    ("out_tma", ["-DB2M_OUT=1", "-DB2M_TMA=1"]),
    ("out_tma64", ["-DB2M_OUT=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]),
    ("out_tma64v", ["-DB2M_OUT=1", "-DB2M_TMA=1", "-DB2M_BVK=64", "-DB2M_VECST=1"]),
]


def build(tag, extra):
    so = f"/tmp/b2_out_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", os.path.join(HARNESS, "b2_mma_probe.cu"),
           "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib


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


def run_out(lib, inp):
    lib.b2out_launch.restype = ctypes.c_int
    lib.b2out_launch.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    dvb = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2out_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                          ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["A"]), ptr(inp["beta"]),
                          ptr(dq), ptr(dk), ptr(dw), ptr(dgk), ptr(dv2), ptr(dvb),
                          B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2out_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk, dv2=dv2, dvb=dvb)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def ref_out(inp, BT):
    """torch(fp32) reference for exactly the output branch."""
    A, dv, beta = inp["A"].float(), inp["dv"].float(), inp["beta"].float()
    B, T, H, BT_ = A.shape
    V = dv.shape[-1]
    NT = cdiv(T, BT)
    dev = A.device

    def pad3(x):  # [B,T,H,*] -> [B,NT,BT,H,*]
        shape = (B, NT * BT, H) + x.shape[3:]
        o = torch.zeros(shape, device=dev, dtype=torch.float32)
        o[:, :T] = x
        return o.view((B, NT, BT, H) + x.shape[3:])

    a4 = pad3(A)          # [B,NT,BT(o_t),H,BT(o_A)]
    dv4 = pad3(dv)        # [B,NT,BT(o_t),H,V]
    b4 = pad3(beta)       # [B,NT,BT(o_t),H]
    dvb = torch.einsum("bctha,bcthv->bcahv", a4, dv4)      # [B,NT,BT(o_A),H,V]
    # b4 is indexed by o_t but the output row is o_A; both are the chunk token axis, so
    # multiplying index-wise is exactly `beta[o_A] * dvb[o_A, v]`.
    dv2 = b4.view(B, NT, BT, H, 1) * dvb
    def unpad(x):  # [B,NT,*,H,*] -> [B,T,H,*]
        x = x.reshape(B, NT * BT, H, x.shape[-1])[:, :T]
        return x.contiguous()
    return dict(dv2=unpad(dv2), dvb=unpad(dvb))


def part_a(libs, tk):
    print("=" * 96)
    print("Part A: correctness -- CUDA output branch dv2/dvb vs torch(fp32) / vs full Triton B2")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        ref = ref_out(inp, BT)
        tri = wy_dqkg_triton(tk, inp, 1.0, BT, num_warps=4)   # tri['dv'] == dv2
        out = {tag: run_out(libs[tag], inp) for tag in ("out", "out_tma", "out_tma64")}
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for tag in ("out", "out_tma", "out_tma64"):
            _, mr2 = stats(f"  {tag} dv2  vs torch", out[tag]["dv2"], ref["dv2"])
            _, mrb = stats(f"  {tag} dvb  vs torch", out[tag]["dvb"], ref["dvb"])
            worst[f"{tag}_dv2"] = max(worst.get(f"{tag}_dv2", 0.0), mr2)
            worst[f"{tag}_dvb"] = max(worst.get(f"{tag}_dvb", 0.0), mrb)
        _, mrt = stats("  trit  dv2  vs torch", tri["dv"], ref["dv2"])
        _, mrct = stats("  out   dv2  vs trit ", out["out"]["dv2"], tri["dv"])
        _, mrid = stats("  out   dv2  vs out_tma64", out["out"]["dv2"], out["out_tma64"]["dv2"])
        worst["trit_dv2"] = max(worst.get("trit_dv2", 0.0), mrt)
        worst["out_vs_trit"] = max(worst.get("out_vs_trit", 0.0), mrct)
        worst["out_vs_tma64"] = max(worst.get("out_vs_tma64", 0.0), mrid)
    print("\nworst max_rel over all shapes:")
    for kk in ("out_dv2", "out_tma_dv2", "out_tma64_dv2", "out_dvb",
               "trit_dv2", "out_vs_trit", "out_vs_tma64"):
        print(f"  {kk:16s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- output-branch kernel vs GEMM-only core and vs full Triton B2")
    print("=" * 96)
    iscore = lambda tag: tag.startswith("core")
    print(f"  {'case':18s} | {'core':>7s} {'core_tma':>8s} | {'out':>7s} {'out_tma':>7s} "
          f"{'out_tma64':>10s} | {'fullB2':>7s} | {'out/core':>8s} {'out_tma/ctma':>12s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        for tag in libs:
            (run_core if iscore(tag) else run_out)(libs[tag], inp)
        torch.cuda.synchronize()
        dev = {}
        for tag in libs:
            fn = (lambda lib=libs[tag]: run_core(lib, inp)) if iscore(tag) \
                else (lambda lib=libs[tag]: run_out(lib, inp))
            dev[tag], _ = profile_dev(fn, reps=40)
        fn2 = lambda: wy_dqkg_triton(tk, inp, 1.0, BT, num_warps=4)
        dev["fullB2"], _ = profile_dev(fn2, reps=40)
        print(f"  {name:18s} | {dev['core']:7.1f} {dev['core_tma']:8.1f} | {dev['out']:7.1f} "
              f"{dev['out_tma']:7.1f} {dev['out_tma64']:10.1f} | {dev['fullB2']:7.1f} | "
              f"{dev['out']/dev['core']:7.2f}x {dev['out_tma']/dev['core_tma']:11.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        so = build(tag, extra)
        lib = load(so)
        if tag == "core":
            lib.b2mma_smem_bytes.restype = ctypes.c_int
        libs[tag] = lib
    tk = build_dqkg()
    mode = os.environ.get("B2O_MODE", "")
    if mode in ("", "a"):
        part_a(libs, tk)
    if mode in ("", "b"):
        part_b(libs, tk)
    print("\nDONE")


if __name__ == "__main__":
    main()
