"""P7-41: replace the hand-written B1 TMA pipeline's `full`/`empty` mbarrier protocol
with the mature CUTLASS `PipelineTmaAsync` — the literal "要么换 CUTLASS PipelineTmaAsync"
next step left by §24.5 (and echoed by §23.6/§36).

Motivation.  §23–§41 pushed the K2-shaped CUDA B1 to `xFLA` ~1.7–3.8× with a *hand-written*
per-stage `mbarrier` handshake.  §24.5 isolated the residual to "每 warp 发射效率"
(`wait`/`short_sb`) and named two options: build a mature pipeline wrapper, or accept the
Triton version.  §36 reiterated the CUTLASS option as "唯一剩下的更大工程".  This round
does the smallest honest version of that: swap *only* the pipeline management
(`cutlass::PipelineTmaAsync<NSTAGE>`) while keeping everything else byte-for-byte the same
(raw-PTX `cp.async.bulk.tensor` copies, `Swizzle<3,4,3>` operands, `mma.sync` compute, TMA
`dh`/`dv2` stores, `fence.proxy.async`), so any delta is attributable to the pipeline itself.

Variants (all `-DB1T_ETMA`):
  base  scalar `dh`/`dv2`, hand mbarrier                            (§24 best)
  hnd   TMA `dh`/`dv2` + EPROXY + SPROXY, hand mbarrier             (= §40 `tma2_full`)
  cut   scalar `dh`/`dv2`, CUTLASS `PipelineTmaAsync`               (this round)
  cut2  TMA `dh`/`dv2` + EPROXY + SPROXY, CUTLASS `PipelineTmaAsync`(this round, best)

Part A  correctness: `cut` vs `base` and `cut2` vs `hnd` must be BIT-IDENTICAL
        (`dh`/`dv2`/`dh0`, BT=32/64, non-divisible T); `cut2` vs matched-bf16 FLA `dhu` 1 ulp.
Part B  CUPTI us/call for all four + direct bf16 FLA `dhu`; `x` columns = ours/FLA.
Part C  run-to-run determinism of the CUTLASS build (BT 16/32/64 × NS 1/2/3), since §38/§39
        showed the hand pipeline itself was racy at NS>=2 before the fence fix.

Assumes CUTLASS headers at ~/github/cutlass/include.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_cutlass.py \
        > .../harness/check_bwd_b1_cutlass.out.txt 2>&1
"""
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dh_roundtrip import build, load, run_b1                      # noqa: E402
from check_bwd_dhu_dtype import cuda_kernel_us, make_inputs, as_dtype, fla_dhu  # noqa: E402
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu     # noqa: E402

D = 128
CUTLASS_INC = "-I~/github/cutlass/include"

STORES = ["-DB1T_TMADH", "-DB1T_TMADV", "-DB1T_EPROXY", "-DB1T_SPROXY"]

VARIANTS = {
    "base": ["-DB1T_ETMA"],
    "hnd":  ["-DB1T_ETMA"] + STORES,
    "cut":  ["-DB1T_ETMA", "-DB1T_CUTLASS", CUTLASS_INC],
    "cut2": ["-DB1T_ETMA", "-DB1T_CUTLASS", CUTLASS_INC] + STORES,
}

SHAPES = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8), (1, 250, 3)]
BTS = (32, 64)
NSS = (1, 2)


def fla_dhu_fn(F, h0, do, scale, BT):
    fi = F["fi"]
    return lambda: chunk_gated_delta_rule_bwd_dhu(
        q=fi["qg"], k=fi["kg"], w=fi["w"], gk=fi["g_cum"],
        h0=h0, dht=None, do=do, dv=F["dv"], scale=scale, chunk_size=BT)


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for name, extra in VARIANTS.items():
        so = f"/tmp/b1c_{name}.so"
        build("b1_tma_probe.cu", so, extra=["-lcuda"] + extra)
        libs[name] = load(so)
    scale = D ** -0.5

    # ---------------------------------------------------------------- Part A
    print("\n" + "=" * 108)
    print("Part A: correctness — CUTLASS pipeline must be BIT-IDENTICAL to the hand pipeline")
    print("   cut vs base (scalar both), cut2 vs hnd (TMA store both), and cut2 vs FLA bf16 `dhu`")
    print("=" * 108)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'cut-base dh':>11s} {'dv2':>10s} {'dh0':>10s} "
          f"| {'cut2-hnd dh':>11s} {'dv2':>10s} {'dh0':>10s} | {'cut2 vs FLA':>11s} {'dv2':>10s}")
    for (B, T, H) in SHAPES:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in BTS:
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            for NS in NSS:
                rb = run_b1(libs["base"], F["fi"], do, F["dv"], scale, BT, NS)
                rc = run_b1(libs["cut"], F["fi"], do, F["dv"], scale, BT, NS)
                rh = run_b1(libs["hnd"], F["fi"], do, F["dv"], scale, BT, NS)
                r2 = run_b1(libs["cut2"], F["fi"], do, F["dv"], scale, BT, NS)
                d_cb = [(rc[j].float() - rb[j].float()).abs().max().item() for j in range(3)]
                d_2h = [(r2[j].float() - rh[j].float()).abs().max().item() for j in range(3)]
                d_fla_dh = (r2[0].float() - F["dh"].float()).abs().max().item()
                d_fla_dv2 = (r2[2].float() - F["dv2"].float()).abs().max().item()
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | "
                      f"{d_cb[0]:11.2e} {d_cb[2]:10.2e} {d_cb[1]:10.2e} | "
                      f"{d_2h[0]:11.2e} {d_2h[2]:10.2e} {d_2h[1]:10.2e} | "
                      f"{d_fla_dh:11.2e} {d_fla_dv2:10.2e}")
                del rb, rc, rh, r2

    # ---------------------------------------------------------------- Part B
    print("\n" + "=" * 108)
    print("Part B: timing — CUPTI us/call (idle card).  FLA = direct bf16 `dhu` (matched dtype)")
    print("   xFLA = ours / FLA_bf16   (>1 = slower than FLA);   xHND = ours / hand pipeline")
    print("=" * 108)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base':>7s} {'hnd':>7s} {'cut':>7s} "
          f"{'cut2':>7s} {'FLAbf16':>8s} | {'xHND':>6s} {'xFLA':>6s}")
    times = {}
    fla_t = {}
    for (B, T, H) in SHAPES:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in BTS:
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            us_fla = cuda_kernel_us(fla_dhu_fn(F, h0, do, scale, BT), "bwd_kernel_dhu")
            fla_t[(B, T, H, BT)] = us_fla
            for NS in NSS:
                us = {name: cuda_kernel_us(
                          lambda L=lib: run_b1(L, F["fi"], do, F["dv"], scale, BT, NS),
                          "b1_tma_kernel")
                      for name, lib in libs.items()}
                times[(B, T, H, BT, NS)] = us
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {us['base']:7.1f} {us['hnd']:7.1f} "
                      f"{us['cut']:7.1f} {us['cut2']:7.1f} {us_fla:8.1f} | "
                      f"{us['cut2']/us['hnd']:6.3f} {us['cut2']/us_fla:6.2f}")

    # ---------------------------------------------------------------- Part C
    print("\n" + "=" * 108)
    print("Part C: determinism — run-to-run of each build, and cross build.  §38 found the")
    print("   UNFENCED hand pipeline (`base`) racy at NS>=2; CUTLASS's own parity bookkeeping")
    print("   should make `cut2` reproducible.  (all compared on `dh`, max_abs)")
    print("=" * 108)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base/base':>10s} {'hnd/hnd':>9s} "
          f"{'cut2/cut2':>10s} | {'cut2-hnd':>10s} {'cut2-base':>10s} | verdict")
    for (B, T, H) in [(2, 1024, 8), (8, 2048, 8)]:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for BT in (16, 32, 64):
            for NS in (1, 2, 3):
                if BT == 64 and NS == 3:
                    continue
                b0 = run_b1(libs["base"], F["fi"], do, F["dv"], scale, BT, NS)
                b1 = run_b1(libs["base"], F["fi"], do, F["dv"], scale, BT, NS)
                h0_ = run_b1(libs["hnd"], F["fi"], do, F["dv"], scale, BT, NS)
                h1_ = run_b1(libs["hnd"], F["fi"], do, F["dv"], scale, BT, NS)
                c0 = run_b1(libs["cut2"], F["fi"], do, F["dv"], scale, BT, NS)
                c1 = run_b1(libs["cut2"], F["fi"], do, F["dv"], scale, BT, NS)
                m = lambda a, b: (a[0].float() - b[0].float()).abs().max().item()
                d_bb, d_hh, d_cc = m(b0, b1), m(h0_, h1_), m(c0, c1)
                d_ch, d_cb = m(c0, h0_), m(c0, b0)
                verdict = "ok" if (d_bb == 0 and d_hh == 0 and d_cc == 0 and d_ch == 0) \
                    else "RACY/DIFF"
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {d_bb:10.2e} {d_hh:9.2e} "
                      f"{d_cc:10.2e} | {d_ch:10.2e} {d_cb:10.2e} | {verdict}")
                del b0, b1, h0_, h1_, c0, c1

    print("\nDONE")


if __name__ == "__main__":
    main()
