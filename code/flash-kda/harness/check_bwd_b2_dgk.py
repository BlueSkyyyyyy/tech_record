"""P7-47: locate & kill the B2 K2-mma core's 8.81M smem bank conflicts (§49.4.3 / §49.6).

§49.4.3 attributed the 8.81M `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` to
"direct smem reads of the mma fragments (tf32 has no ldmatrix)".  This harness tests that
attribution with a controlled ablation on `b2_mma_probe.cu`:

  * `B2M_NODGK`  -- skip the `dgk` column reduction; if the conflicts vanish here they were
                    the dgk loop, not the mma fragment reads.
  * `B2M_DGKCF`  -- conflict-free dgk reduction (8 kcols / instruction, banks 4*c+s cover
                    0..31 exactly for BVP=36); correctness + timing + ncu.

The other three ops (`dq/dk/dw`) are untouched in every variant, so their bitwise equality
with the baseline is a strong invariant that the ablation is controlled.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_dgk.py \
        > .../harness/check_bwd_b2_dgk.out.txt 2>&1
    B2DGK_MODE=a | b
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

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_b2_mma import make, CASES, BT, D                       # noqa: E402
from check_bwd_b2_core import build_core_triton, run_triton_core, ref_core  # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                            # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev             # noqa: E402

VARIANTS = {
    "base":  [],
    "nodgk": ["-DB2M_NODGK=1"],
    "dgkcf": ["-DB2M_DGKCF=1"],
}


def build(tag):
    src = os.path.join(HARNESS, "b2_mma_probe.cu")
    so = f"/tmp/b2_dgk_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + VARIANTS[tag] + [src, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp, BT_):
    B, T, H, K = inp["q"].shape
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


def part_a(libs):
    print("=" * 92)
    print("Part A: controlled ablation correctness (dq/dk/dw must be bitwise vs base)")
    print("=" * 92)
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        ref = ref_core(inp, BT)
        base = run_cuda(libs["base"], inp, BT)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for tag in ("nodgk", "dgkcf"):
            out = run_cuda(libs[tag], inp, BT)
            for fld in ("dq", "dk", "dw"):
                d = (out[fld] - base[fld]).abs().max().item()
                print(f"  {tag:6s} {fld} vs base max_abs={d:.3e}"
                      + ("  (BITWISE)" if d == 0 else "  <-- NOT bitwise"))
            if tag == "nodgk":
                print(f"  nodgk  dgk  max_abs vs base = {(out['dgk']-base['dgk']).abs().max().item():.3e}"
                      f"  (expect ~base magnitude: {base['dgk'].abs().mean().item():.4g})")
            else:
                rel = (out["dgk"] - base["dgk"]).abs() / (base["dgk"].abs() + 1e-30)
                print(f"  dgkcf  dgk  vs base  max_abs={(out['dgk']-base['dgk']).abs().max().item():.3e}"
                      f"  max_rel={rel.max().item():.3e}")
                _, mrr = stats(f"  dgkcf  dgk  vs torch", out["dgk"], ref["dgk"])
                _, mrb = stats(f"  base   dgk  vs torch", base["dgk"], ref["dgk"])


def part_b(libs):
    print("\n" + "=" * 92)
    print("Part B: kernel-only dev timing -- base / nodgk / dgkcf")
    print("=" * 92)
    print(f"  {'case':18s} | {'base dev':>9s} | {'nodgk dev':>9s} {'x':>6s} | "
          f"{'dgkcf dev':>9s} {'x':>6s} | {'trit nw8':>9s}")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        res = {}
        for tag in ("base", "nodgk", "dgkcf"):
            lib = libs[tag]
            run_cuda(lib, inp, BT); torch.cuda.synchronize()
            res[tag], _ = profile_dev(lambda: run_cuda(lib, inp, BT), reps=30)
        tk = build_core_triton()
        run_triton_core(tk, inp, BT, num_warps=8); torch.cuda.synchronize()
        dtr, _ = profile_dev(lambda: run_triton_core(tk, inp, BT, num_warps=8), reps=30)
        print(f"  {name:18s} | {res['base']:9.1f} | {res['nodgk']:9.1f} "
              f"{res['base']/res['nodgk']:6.3f} | {res['dgkcf']:9.1f} "
              f"{res['base']/res['dgkcf']:6.3f} | {dtr:9.1f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {tag: load(build(tag)) for tag in VARIANTS}
    mode = os.environ.get("B2DGK_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
