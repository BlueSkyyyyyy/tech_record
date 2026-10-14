"""P7-53: the *second non-GEMM segment* of B2 in CUDA -- the `i_k==0` finalize term

    dkgb[o_A, k] = sum_{o_t} A[o_A, o_t] * (-dw[o_t, k])        (== A @ (-dw))

P7-52 (§55) landed the *output branch* `dvb = A @ dv` (`docs/07` §55).  §55.6 named the rest
of the non-GEMM work: "`dA`/`db` 的后处理项（`dw@kg^T`、`A^T diag A^T`、`dkgb`）与门控链".
This round lands the next chunk-row contraction `dkgb = A @ (-dw)` (FLA / torch line
`b_dkgb = einsum("bhtu,bhuk->bhtk", b_A, b_dw)`, check_bwd_dqkg_kernel.py:182; `b_dw = -b_dw`
at :180).  It is structurally the *same* as `A @ dv` (contraction over chunk rows `o_t`,
output row `o_A`), so `Akk` feeds the mma A-operand by plain index remap (no transpose) and the
freshly-staged `dw` tile (`[BT, BVK]`, k-tiled) is the B-operand -- negated on the fly.

`dkgb` is an intermediate (feeds `db`, `dg`, `dk`, `dA`), so like P7-52's `dvb` it is exposed
as its own output to check the GEMM standalone.

  Part A  correctness: CUDA `dkgb` vs torch(fp32) using the *same* `dw` operand (obtained from
          the CUDA GEMM core, so also tf32), and bit-identity across non-TMA/TMA32/TMA64.
  Part B  timing: dkgb kernel (GEMM core + segment) vs the GEMM-only core (delta) and vs the
          full Triton B2.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_dkgb.py \
        > .../harness/check_bwd_b2_dkgb.out.txt 2>&1
    B2D_MODE=a | b
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
from check_bwd_ws_reuse import stats, cdiv                             # noqa: E402
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
    ("dkgb", ["-DB2M_DKGB=1"]),
    ("dkgb_tma", ["-DB2M_DKGB=1", "-DB2M_TMA=1"]),
    ("dkgb_tma64", ["-DB2M_DKGB=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]),
]


def build(tag, extra):
    so = f"/tmp/b2_dkgb_{tag}.so"
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
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2dkgb_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(dw), ptr(inp["A"]),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dkgb),
                           B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2dkgb_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw_out, dgk=dgk, dkgb=dkgb)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def ref_dkgb(inp, dw, BT):
    """torch(fp32) reference for exactly the dkgb segment, using the *given* dw operand.

    `b_A[o_A,o_t] = A4[b,c,o_t,h,o_A]` (matches check_bwd_dqkg_kernel.py:149-150); the
    reference then negates dw (:180) and contracts `b_dkgb = A @ b_dw` (:182).
    """
    A, dwd = inp["A"].float(), dw.float()
    B, T, H, BT_ = A.shape
    K = dwd.shape[-1]
    NT = cdiv(T, BT)
    dev = A.device

    def pad3(x):  # [B,T,H,*] -> [B,NT,BT,H,*]
        shape = (B, NT * BT, H) + x.shape[3:]
        o = torch.zeros(shape, device=dev, dtype=torch.float32)
        o[:, :T] = x
        return o.view((B, NT, BT, H) + x.shape[3:])

    a4 = pad3(A)          # [B,NT,BT(o_t),H,BT(o_A)]
    dw4 = pad3(dwd)       # [B,NT,BT(o_t),H,K]
    # dkgb[o_A,k] = sum_{o_t} A4[o_t,o_A] * (-dw4[o_t,k])
    dkgb = torch.einsum("bctha,bcthk->bcahk", a4, -dw4)    # [B,NT,BT(o_A),H,K]

    def unpad(x):  # [B,NT,BT,H,*] -> [B,T,H,*]  (o_A axis -> token)
        x = x.reshape(B, NT * BT, H, x.shape[-1])[:, :T]
        return x.contiguous()
    return dict(dkgb=unpad(dkgb))


def part_a(libs, tk):
    print("=" * 96)
    print("Part A: correctness -- CUDA dkgb = A@(-dw) vs torch(fp32), dw operand from CUDA core")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]          # tf32 dw operand (same for all)
        ref = ref_dkgb(inp, dw, BT)
        out = {tag: run_dkgb(libs[tag], inp, dw)
               for tag in ("dkgb", "dkgb_tma", "dkgb_tma64")}
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for tag in ("dkgb", "dkgb_tma", "dkgb_tma64"):
            _, mr = stats(f"  {tag} dkgb  vs torch", out[tag]["dkgb"], ref["dkgb"])
            worst[f"{tag}_dkgb"] = max(worst.get(f"{tag}_dkgb", 0.0), mr)
        _, mrid = stats("  dkgb  vs dkgb_tma64", out["dkgb"]["dkgb"], out["dkgb_tma64"]["dkgb"])
        _, mrit = stats("  dkgb  vs dkgb_tma ", out["dkgb"]["dkgb"], out["dkgb_tma"]["dkgb"])
        worst["dkgb_vs_tma64"] = max(worst.get("dkgb_vs_tma64", 0.0), mrid)
        worst["dkgb_vs_tma"] = max(worst.get("dkgb_vs_tma", 0.0), mrit)
    print("\nworst max_rel over all shapes:")
    for kk in ("dkgb_dkgb", "dkgb_tma_dkgb", "dkgb_tma64_dkgb",
               "dkgb_vs_tma", "dkgb_vs_tma64"):
        print(f"  {kk:20s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- dkgb kernel vs GEMM-only core and vs full Triton B2")
    print("=" * 96)
    iscore = lambda tag: tag.startswith("core")
    print(f"  {'case':18s} | {'core':>7s} {'core_tma':>8s} | {'dkgb':>7s} {'dkgb_tma':>8s} "
          f"{'dkgb_tma64':>10s} | {'fullB2':>7s} | {'dkgb/core':>9s} {'dkgb_tma/ctma':>13s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]
        for tag in libs:
            if iscore(tag):
                run_core(libs[tag], inp)
            else:
                run_dkgb(libs[tag], inp, dw)
        torch.cuda.synchronize()
        dev = {}
        for tag in libs:
            if iscore(tag):
                dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_core(lib, inp), reps=40)
            else:
                dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_dkgb(lib, inp, dw), reps=40)
        fn2 = lambda: wy_dqkg_triton(tk, inp, 1.0, BT, num_warps=4)
        dev["fullB2"], _ = profile_dev(fn2, reps=40)
        print(f"  {name:18s} | {dev['core']:7.1f} {dev['core_tma']:8.1f} | {dev['dkgb']:7.1f} "
              f"{dev['dkgb_tma']:8.1f} {dev['dkgb_tma64']:10.1f} | {dev['fullB2']:7.1f} | "
              f"{dev['dkgb']/dev['core']:8.2f}x {dev['dkgb_tma']/dev['core_tma']:12.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    tk = build_dqkg()
    mode = os.environ.get("B2D_MODE", "")
    if mode in ("", "a"):
        part_a(libs, tk)
    if mode in ("", "b"):
        part_b(libs, tk)
    print("\nDONE")


if __name__ == "__main__":
    main()
