"""P7-32: does a TMA `dh` store recover the ~33% §34.1 attributed to `lg_throttle`?

§34.1 measured that *removing* B1's per-thread scalar `dh` global store makes B1
0.28-0.67x as fast (33-72%), and `ncu` localized the cost to `lg_throttle` (the
LSU/store-queue backs up) plus a ~3x L2 write amplification -- **not** DRAM bandwidth.
§34.4 then flagged the obvious follow-up (**待验证**): if the cost is store-*issue*
efficiency, then moving the store to the TMA engine (off the LSU path entirely) should
recover most of that 33% *without* any B1->B2 fusion.

This script implements that: a `-DB1T_TMADH` variant of `b1_tma_probe.cu` that stages
each warp's `[KD,BW]` slice into an unpadded smem tile (double buffered) and stores it
with one `cp.async.bulk.tensor.5d.global.shared::cta.bulk_group` per warp per chunk
(SWIZZLE_NONE, inner dim 32 B; no cross-warp barrier).  Build three variants:

  base   -DB1T_ETMA                      (scalar `dh` store; §24 best)
  nodh   -DB1T_ETMA -DB1T_NODH           (no `dh` at all; the §34.1 upper bound)
  tma    -DB1T_ETMA -DB1T_TMADH          (TMA `dh` store; this round)

Part A  correctness: tma vs base `dh/dv2/dh0` must be bit-identical; vs FLA `dhu` sanity.
Part B  CUPTI us/call for all three; `tma/base` shows how much of the removed-store gain
        is recovered, `nodh/base` is the §34.1 bound.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_tma_store.py \
        > .../harness/check_bwd_b1_tma_store.out.txt 2>&1
"""
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dh_roundtrip import (                                       # noqa: E402
    build, load, run_b1,
)
from check_bwd_dhu_dtype import (                                          # noqa: E402
    cuda_kernel_us, stats, make_inputs, as_dtype, fla_dhu, cdiv,
)

D = 128
B1_BASE_SO = "/tmp/b1t_base.so"      # ETMA
B1_NODH_SO = "/tmp/b1t_nodh.so"      # ETMA + B1T_NODH
B1_TMA_SO = "/tmp/b1t_tma.so"        # ETMA + B1T_TMADH


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build("b1_tma_probe.cu", B1_BASE_SO, extra=["-lcuda", "-DB1T_ETMA"])
    build("b1_tma_probe.cu", B1_NODH_SO, extra=["-lcuda", "-DB1T_ETMA", "-DB1T_NODH"])
    build("b1_tma_probe.cu", B1_TMA_SO, extra=["-lcuda", "-DB1T_ETMA", "-DB1T_TMADH"])
    lib_base = load(B1_BASE_SO)
    lib_nodh = load(B1_NODH_SO)
    lib_tma = load(B1_TMA_SO)
    scale = D ** -0.5

    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8), (1, 250, 3)]

    # ------------------------------------------------------------------ Part A
    print("\n" + "=" * 100)
    print("Part A: correctness — TMA `dh` store vs scalar store must be BIT-IDENTICAL")
    print("   tma vs base: dh / dv2 / dh0 max_abs;  vs FLA `dhu`: dh max_abs (sanity)")
    print("=" * 100)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'dh':>10s} {'dv2':>10s} {'dh0':>10s} "
          f"| {'dh vs FLA':>12s}")
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            for NS in (1, 2):
                dh_b, dh0_b, dv2_b = run_b1(lib_base, F["fi"], do, F["dv"], scale, BT, NS)
                dh_t, dh0_t, dv2_t = run_b1(lib_tma, F["fi"], do, F["dv"], scale, BT, NS)
                d_dh = (dh_t.float() - dh_b.float()).abs().max().item()
                d_dv2 = (dv2_t.float() - dv2_b.float()).abs().max().item()
                d_dh0 = (dh0_t.float() - dh0_b.float()).abs().max().item()
                d_fla = (dh_t.float() - F["dh"].float()).abs().max().item()
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {d_dh:10.2e} {d_dv2:10.2e} "
                      f"{d_dh0:10.2e} | {d_fla:12.2e}")
                del dh_b, dh0_b, dv2_b, dh_t, dh0_t, dv2_t

    # ------------------------------------------------------------------ Part B
    print("\n" + "=" * 100)
    print("Part B: timing — CUPTI us/call (idle card).  nodh/base = §34.1 bound;"
          " tma/base = recovered")
    print("=" * 100)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base µs':>9s} {'nodh µs':>9s} "
          f"{'tma µs':>9s} | {'nodh/base':>9s} {'tma/base':>9s} | {'tma vs nodh':>11s}")
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            for NS in (1, 2):
                us_b = cuda_kernel_us(
                    lambda: run_b1(lib_base, F["fi"], do, F["dv"], scale, BT, NS),
                    "b1_tma_kernel")
                us_n = cuda_kernel_us(
                    lambda: run_b1(lib_nodh, F["fi"], do, F["dv"], scale, BT, NS),
                    "b1_tma_kernel")
                us_t = cuda_kernel_us(
                    lambda: run_b1(lib_tma, F["fi"], do, F["dv"], scale, BT, NS),
                    "b1_tma_kernel")
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {us_b:9.1f} {us_n:9.1f} "
                      f"{us_t:9.1f} | {us_n/us_b:9.3f} {us_t/us_b:9.3f} | "
                      f"{(us_b-us_t)/(us_b-us_n):11.3f}")

    # ------------------------------------------------------------------ Part C
    print("\n" + "=" * 100)
    print("Part C: summary — fraction of the removed-store gain recovered by TMA")
    print("=" * 100)
    print("  recovered = (base - tma) / (base - nodh);  1.0 == full recovery of the §34.1 bound")
    print("\nDONE")


if __name__ == "__main__":
    main()
