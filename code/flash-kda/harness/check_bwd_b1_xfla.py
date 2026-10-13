"""P7-35: close the loop on the K2-shaped CUDA B1 — its xFLA vs **matched-bf16** FLA `dhu`.

Motivation.  §22–§36 built the target-card-shaped CUDA B1 incrementally and each round
reported an `xFLA` against a *different* FLA baseline:

  §22 (P7-19, register-resident + ldmatrix/mma.sync)    xFLA 2.10–3.75  (BT=64)
  §23 (P7-20, + TMA input + warp-spec)                  xFLA 1.73–3.80
  §24 (P7-21, + E^tot via 3D TMA, `-DB1T_ETMA`)         xFLA best 1.63
  §35/§36 (P7-32/33, + TMA `dh`/`dv2` store)            only reported *relative* gains

The store rounds (P7-32/33) never re-expressed their gain as an `xFLA`, so the cumulative
"best CUDA prototype" (= `-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV`, called `tma2` here) has
**no** published comparison against FLA.  §37 (P7-34) also fixed the *dtype* convention:
the honest denominator is FLA `dhu` on **bf16** operands (fp32 FLA is 4.0–5.5× slower via
2× bytes + tf32 `tl.dot`).  This round measures `tma2` vs matched-bf16 FLA `dhu` directly
and splits the remaining gap into **(i) BT granularity** (deployment is BT=16, FLA=64) and
**(ii) implementation**.

Variants (all `-DB1T_ETMA`, so `E^tot` is already in the pipeline):
  base   scalar `dh` + scalar `dv2`                     (§24 best)
  tma    TMA `dh`  + scalar `dv2`                       (P7-32)
  tma2   TMA `dh`  + TMA `dv2`                           (P7-33 = current best)

Part A  correctness: `tma2` vs `base` bit-identical (`dh`/`dv2`/`dh0`) at BT=32/64
        (FLA forward only supports chunk 32/64); `tma2` vs FLA bf16 `dhu` sanity (1 ulp).
Part B  CUPTI us/call for base/tma/tma2 **and** direct bf16 FLA `dhu`, per shape×BT×NS,
        with `xFLA = ours/FLA` for all three.
Part C  the matched-granularity summary (BT=64) for the best variant `tma2`.
Part D  pipeline determinism scan (BT 16/32/64 × NS 1/2/3): exposes a **race at
        (BT=16, NS=2)** in the hand-written mbarrier pipeline (base run-to-run differs),
        while NS=1/3 and BT=32/64 are bit-stable.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_xfla.py \
        > .../harness/check_bwd_b1_xfla.out.txt 2>&1
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

VARIANTS = {
    "base": ["-DB1T_ETMA"],
    "tma":  ["-DB1T_ETMA", "-DB1T_TMADH"],
    "tma2": ["-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV"],
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
        so = f"/tmp/b1x_{name}.so"
        build("b1_tma_probe.cu", so, extra=["-lcuda"] + extra)
        libs[name] = load(so)
    scale = D ** -0.5

    # ---------------------------------------------------------------- Part A
    print("\n" + "=" * 104)
    print("Part A: correctness — tma2 (TMA dh+dv2) vs base (scalar both) MUST BE BIT-IDENTICAL")
    print("   BT=32/64 with FLA operands;  vs FLA bf16 `dhu` sanity")
    print("=" * 104)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'dh':>10s} {'dv2':>10s} {'dh0':>10s} "
          f"| {'dh vs FLA':>11s} {'dv2 vs FLA':>11s}")
    for (B, T, H) in SHAPES:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in BTS:
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            for NS in NSS:
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

    # ---------------------------------------------------------------- Part B
    print("\n" + "=" * 104)
    print("Part B: timing — CUPTI us/call (idle card).  FLA = direct bf16 `dhu` (matched dtype)")
    print("   xFLA = ours / FLA_bf16   (>1 = slower than FLA)")
    print("=" * 104)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base':>7s} {'tma':>7s} {'tma2':>7s} "
          f"{'FLAbf16':>8s} | {'xbase':>6s} {'xtma':>6s} {'xtma2':>6s}")
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
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {us['base']:7.1f} {us['tma']:7.1f} "
                      f"{us['tma2']:7.1f} {us_fla:8.1f} | {us['base']/us_fla:6.2f} "
                      f"{us['tma']/us_fla:6.2f} {us['tma2']/us_fla:6.2f}")

    # ---------------------------------------------------------------- Part C
    print("\n" + "=" * 104)
    print("Part C: summary — BT=64 (matched granularity => pure implementation gap)")
    print("=" * 104)
    print("   matched BT=64:  ours(tma2) / FLA_bf16(BT=64)")
    print(f"  {'shape':16s} {'NS':>3s} | {'ours tma2':>10s} {'FLA bf16':>9s} {'xFLA':>6s}")
    for (B, T, H) in SHAPES:
        for NS in NSS:
            us = times.get((B, T, H, 64, NS))
            if us is None:
                continue
            us_fla = fla_t[(B, T, H, 64)]
            print(f"  B{B} T{T} H{H:<6} 64 {NS:3d} | {us['tma2']:10.1f} {us_fla:9.1f} "
                  f"{us['tma2']/us_fla:6.2f}")

    print("\n   (per-BT xFLA of the best variant `tma2`, NS=1)")
    print(f"  {'shape':16s} {'BT':>3s} | {'tma2 us':>8s} {'FLAbf16':>8s} {'xFLA':>6s} "
          f"| note")
    for (B, T, H) in SHAPES:
        for BT in BTS:
            us_fla = fla_t[(B, T, H, BT)]
            us = times[(B, T, H, BT, 1)]["tma2"]
            note = "matched to FLA" if BT == 64 else ""
            print(f"  B{B} T{T} H{H:<6} {BT:3d} | {us:8.1f} {us_fla:8.1f} {us/us_fla:6.2f} "
                  f"| {note}")

    # ---------------------------------------------------------------- Part D
    # Determinism scan: the K2-shaped pipeline must be reproducible run-to-run.  Adding
    # BT=16 coverage surfaced a *race* in this probe at NSTAGE=2 (two consecutive `base`
    # runs differ), while NSTAGE=1/3 are stable; BT=32/64 NS=2 were bit-identical in
    # §35/§36.  We scan `base` (no TMA store) so the race cannot be blamed on the store
    # path, and `tma2`, and only trust tma2-vs-base identity where base is deterministic.
    print("\n" + "=" * 104)
    print("Part D: pipeline determinism (two consecutive runs of the same kernel must agree)")
    print("   base/base = run base twice; tma2/base = TMA store vs scalar (must be 0)")
    print("=" * 104)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base/base dh':>13s} {'tma2/base dh':>13s} "
          f"{'dh0':>10s} | verdict")
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
                    continue  # not instantiated by b1t_launch
                r0 = run_b1(libs["base"], F["fi"], do, F["dv"], scale, BT, NS)
                r1 = run_b1(libs["base"], F["fi"], do, F["dv"], scale, BT, NS)
                o2 = run_b1(libs["tma2"], F["fi"], do, F["dv"], scale, BT, NS)
                d_bb = (r0[0].float() - r1[0].float()).abs().max().item()
                d_tb = (o2[0].float() - r0[0].float()).abs().max().item()
                d_tb0 = (o2[1].float() - r0[1].float()).abs().max().item()
                verdict = "ok" if (d_bb == 0 and d_tb == 0 and d_tb0 == 0) else "RACY"
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {d_bb:13.2e} {d_tb:13.2e} "
                      f"{d_tb0:10.2e} | {verdict}")
                del r0, r1, o2

    print("\nDONE")


if __name__ == "__main__":
    main()
