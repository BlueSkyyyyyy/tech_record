"""P7-48: localize (and fix the half of) the K2-shaped mma CUDA B2 GEMM core's
"L2 bytes +35% vs the identical-op Triton core" (3.03 vs 2.24 GB), which §50.5
left as the open lead after ruling out the `dgk` bank conflicts.

The `ncu` byte decomposition lives in `harness/probe_b2_l2_ncu.sh`
(`harness/probe_b2_l2_ncu.out.txt`); this file is the correctness + timing half:

  Part A  base vs vecst (only change: two scalar 32-bit `STG.E` -> one 64-bit
          `float2` store per output) are **bit-identical** and both land the tf32
          noise floor vs torch(fp32) and vs the identical-op Triton core.
  Part B  kernel-only `dev` (CUPTI) + `wall` for base vs vecst across four shapes.
  Part C  restates the decomposition so the numbers are in one place.

The `vecst` change is the deployable, free, numerically-lossless part of the fix;
the cp.async read-side amplification (probe output) is a separate, structural item
(next lever = TMA, per the B1 §19-24 lesson).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_l2.py \
        > .../harness/check_bwd_b2_l2.out.txt 2>&1
    B2L_MODE=a | b | c
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# CUDA core is tf32 (matches `tl.dot`); true-fp32 torch reference.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs      # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                             # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev              # noqa: E402
from check_bwd_b2_core import build_core_triton, run_triton_core, ref_core  # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
VARIANTS = [("base", []), ("vecst", ["-DB2M_VECST=1"])]


def build(tag, extra):
    so = f"/tmp/b2_mma_l2_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + [os.path.join(HARNESS, "b2_mma_probe.cu"), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp, BT_=BT):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT_)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2mma_launch(
        ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
        ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
        B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2mma_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def part_a(libs):
    print("=" * 92)
    print("Part A: correctness -- base vs vecst must be BIT-IDENTICAL; both vs torch/triton")
    print("=" * 92)
    tk = build_core_triton()
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        ref = ref_core(inp, BT)
        out = {tag: run_cuda(lib, inp) for tag, lib in libs.items()}
        tri = run_triton_core(tk, inp, BT, num_warps=8)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in ("dq", "dk", "dw", "dgk"):
            b, v = out["base"][fld], out["vecst"][fld]
            idmax = (b.float() - v.float()).abs().max().item()
            _, mr_c = stats(f"  cuda vs torch {fld}", b, ref[fld])
            _, mr_t = stats(f"  trit vs torch {fld}", tri[fld], ref[fld])
            print(f"  base-vs-vecst {fld:4s} max_abs={idmax:.3e}  "
                  f"(vs torch: cuda {mr_c:.3e} / trit {mr_t:.3e})")
            worst[f"cuda_{fld}"] = max(worst.get(f"cuda_{fld}", 0.0), mr_c)
            worst[f"ident_{fld}"] = max(worst.get(f"ident_{fld}", 0.0), idmax)
    print("\nworst:")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"  {fld:4s}  cuda/torch {worst['cuda_'+fld]:.3e}   "
              f"base-vs-vecst max_abs {worst['ident_'+fld]:.3e}")


def part_b(libs):
    print("\n" + "=" * 92)
    print("Part B: head-to-head kernel-only timing -- base vs vecst (float2 stores)")
    print("=" * 92)
    print(f"  {'case':18s} | {'base dev':>9s} {'vecst dev':>9s} | {'vecst/base':>10s} | "
          f"{'base wall':>9s} {'vecst wall':>10s} | {'wall ratio':>10s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        run_cuda(libs["base"], inp); run_cuda(libs["vecst"], inp); torch.cuda.synchronize()
        db, _ = profile_dev(lambda: run_cuda(libs["base"], inp), reps=40)
        dv, _ = profile_dev(lambda: run_cuda(libs["vecst"], inp), reps=40)
        wb = time_wall(lambda: run_cuda(libs["base"], inp), reps=60)
        wv = time_wall(lambda: run_cuda(libs["vecst"], inp), reps=60)
        print(f"  {name:18s} | {db:9.1f} {dv:9.1f} | {dv/db:9.3f}x | "
              f"{wb:9.1f} {wv:10.1f} | {wv/wb:9.3f}x")


def part_c():
    print("\n" + "=" * 92)
    print("Part C: ncu byte decomposition (see probe_b2_l2_ncu.out.txt), b8t2048h8")
    print("=" * 92)
    print("  variant        L2 bytes  srcunit_rd  srcunit_wr  l1tex_ld_sec  l1tex_st_sec  Duration")
    rows = [
        ("ours base",    "3.02 GB", "59.77M", "9.34M", "44.83M", "12.71M", "520.5us"),
        ("ours vecst",   "2.89 GB", "59.77M", "6.42M", "44.83M", " 6.42M", "516.9us"),
        ("ours noasync", "2.39 GB", "39.85M", "9.58M", "39.85M", "12.71M", "627.4us"),
        ("ours noasync+vecst", "2.23 GB", "39.85M", "6.42M", "39.85M", "6.42M", "604.3us"),
        ("triton core",  "2.23 GB", "39.85M", "6.42M", "39.85M", " 6.42M", "  ~551us"),
    ]
    for r in rows:
        print("  " + "  ".join(f"{c:>13s}" for c in r))
    print("\n  decomposition: read side (cp.async BYPASS) +0.64 GB, write side (scalar")
    print("  stores) +0.13 GB; `noasync_vecst` removes BOTH and matches Triton exactly.")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    mode = os.environ.get("B2L_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
