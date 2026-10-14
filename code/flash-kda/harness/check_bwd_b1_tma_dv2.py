"""P7-33: generalize the B1 TMA store from `dh` (P7-32) to `dv2` too.

§35.5 left one **待验证**: the B1's per-chunk `dv2` output (`[BT,BW]` per warp, written
token-by-token with an explicit `tr<T` mask) is *structurally* the same as `dh` and could
also go through the TMA engine, but P7-32 only changed `dh` (to match §34.1's ablation).
This script closes that gap: a `-DB1T_TMADV` variant of `b1_tma_probe.cu` stages each
warp's `[BT,BW]` slice into an unpadded smem tile (double buffered, sharing the `dh`
bulk-group commit) and stores it with one `cp.async.bulk.tensor.4d` (tensor map dims
{VD,T,H,B}, box {BW,BT,1,1}, SWIZZLE_NONE).  A non-divisible `T` needs no explicit mask:
TMA clips the OOB token rows against the `T` tensor bound.

Variants (all also `-DB1T_ETMA`):
  base    scalar `dh` + scalar `dv2`      (§24 best)
  tma     TMA `dh`  + scalar `dv2`        (P7-32)
  tma2    TMA `dh`  + TMA `dv2`           (this round)
  nodh    no `dh` (upper bound for the `dh` store)
  nodv2   no `dv2` (upper bound for the `dv2` store)
  nodhv2  neither

Part A  correctness: tma2 vs base must be BIT-IDENTICAL (`dh`/`dv2`/`dh0`); vs FLA sanity.
Part B  CUPTI us/call for all six; recovered = (base-variant)/(base-ablation).

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_tma_dv2.py \
        > .../harness/check_bwd_b1_tma_dv2.out.txt 2>&1
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

from check_bwd_dh_roundtrip import build, load, run_b1                          # noqa: E402
from check_bwd_dhu_dtype import cuda_kernel_us, make_inputs, as_dtype, fla_dhu  # noqa: E402

D = 128

VARIANTS = {
    "base":   ["-DB1T_ETMA"],
    "tma":    ["-DB1T_ETMA", "-DB1T_TMADH"],
    "tma2":   ["-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV"],
    "nodh":   ["-DB1T_ETMA", "-DB1T_NODH"],
    "nodv2":  ["-DB1T_ETMA", "-DB1T_NODV2"],
    "nodhv2": ["-DB1T_ETMA", "-DB1T_NODH", "-DB1T_NODV2"],
}


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for name, extra in VARIANTS.items():
        so = f"/tmp/b1t_{name}.so"
        build("b1_tma_probe.cu", so, extra=["-lcuda"] + extra)
        libs[name] = load(so)
    scale = D ** -0.5

    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8), (1, 250, 3)]

    # ------------------------------------------------------------------ Part A
    print("\n" + "=" * 100)
    print("Part A: correctness — tma2 (TMA dh+dv2) vs base (scalar both) MUST BE BIT-IDENTICAL")
    print("   tma2 vs base: dh / dv2 / dh0 max_abs;   vs FLA `dhu` sanity (dh, dv2)")
    print("=" * 100)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'dh':>10s} {'dv2':>10s} {'dh0':>10s} "
          f"| {'dh vs FLA':>11s} {'dv2 vs FLA':>11s}")
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            for NS in (1, 2):
                dh_b, dh0_b, dv2_b = run_b1(libs["base"], F["fi"], do, F["dv"], scale, BT, NS)
                dh_t, dh0_t, dv2_t = run_b1(libs["tma2"], F["fi"], do, F["dv"], scale, BT, NS)
                d_dh = (dh_t.float() - dh_b.float()).abs().max().item()
                d_dv2 = (dv2_t.float() - dv2_b.float()).abs().max().item()
                d_dh0 = (dh0_t.float() - dh0_b.float()).abs().max().item()
                dfla_dh = (dh_t.float() - F["dh"].float()).abs().max().item()
                dfla_dv2 = (dv2_t.float() - F["dv2"].float()).abs().max().item()
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {d_dh:10.2e} {d_dv2:10.2e} "
                      f"{d_dh0:10.2e} | {dfla_dh:11.2e} {dfla_dv2:11.2e}")
                del dh_b, dh0_b, dv2_b, dh_t, dh0_t, dv2_t

    # ------------------------------------------------------------------ Part B
    print("\n" + "=" * 100)
    print("Part B: timing — CUPTI us/call (idle card).  nodh/nodv2/nodhv2 = upper bounds")
    print("=" * 100)
    hdr = (f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base':>8s} {'tma(dh)':>8s} "
           f"{'tma2':>8s} {'nodv2':>8s} {'nodh':>8s} {'nodhv2':>8s} | "
           f"{'ndh/b':>6s} {'ndv2/b':>6s} {'tma/b':>6s} {'tma2/b':>6s}")
    print(hdr)
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            for NS in (1, 2):
                us = {name: cuda_kernel_us(
                          lambda L=lib: run_b1(L, F["fi"], do, F["dv"], scale, BT, NS),
                          "b1_tma_kernel")
                      for name, lib in libs.items()}
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {us['base']:8.1f} "
                      f"{us['tma']:8.1f} {us['tma2']:8.1f} {us['nodv2']:8.1f} "
                      f"{us['nodh']:8.1f} {us['nodhv2']:8.1f} | "
                      f"{us['nodh']/us['base']:6.3f} {us['nodv2']/us['base']:6.3f} "
                      f"{us['tma']/us['base']:6.3f} {us['tma2']/us['base']:6.3f}")

    # ------------------------------------------------------------------ Part C
    print("\n" + "=" * 100
          )
    print("Part C: recovery fractions")
    print("   recovered_dv2 = (base - tma2) / (base - nodv2)  ... vs only removing dv2?")
    print("   better: marginal dv2 cost = tma - tma2 ; total both = base - nodhv2")
    print("=" * 100)
    print("  (see per-row tma vs tma2 columns above; tma2==tma => dv2 TMA is free/neutral)")
    print("\nDONE")


if __name__ == "__main__":
    main()
