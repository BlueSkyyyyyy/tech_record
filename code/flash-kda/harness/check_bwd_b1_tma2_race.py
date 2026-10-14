"""P7-37: close the §39.5 待验证 -- does the *store* side of the strongest TMA B1
(`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV`, i.e. `tma2`) also need a `fence.proxy.async`?

§36/§39 left this open: §38's race was on the **input** side (consumer generic `ld` of the
TMA-refilled `st->e` vs producer async-proxy refill = WAR), fixed by `-DB1T_EPROXY` (§39).
But `tma2` adds a second generic<->async pair in the **store** direction: the compute warp
writes the per-warp `dh`/`dv2` tile into smem via the generic proxy, then `cp.async.bulk.tensor`
(async proxy) reads it back out (RAW), and the tile is double-buffered (reuse is a WAR handled
only by `cp.async.bulk.wait_group.read`).  §39.5 explicitly flagged this as **待验证** (it was
only scanned on the scalar-store `-DB1T_ETMA` baseline).

This script adds the missing store-side fence (`-DB1T_SPROXY`, after the smem writes and before
the `cp.async.bulk.tensor...bulk_group` store) and does the same controlled determinism scan on
the *deployed-best* variant:

  Part A  determinism scan (NR runs of the SAME kernel, `dh` compared run-to-run), large grid
          (the §38/§39 failing regime), over {BT}x{NS} and the build matrix
            scalar      (-DB1T_ETMA)                              racy (input side)
            scalar_p    (-DB1T_ETMA -DB1T_EPROXY)                 clean (input fixed, no store TMA)
            tma2        (-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV)      ? (§38 deployed-best, no fence)
            tma2_s      (+ -DB1T_SPROXY)                          ? (store fence only)
            tma2_in     (+ -DB1T_EPROXY)                          ? (input fixed, store RAW open)
            tma2_full   (+ -DB1T_EPROXY -DB1T_SPROXY)             clean (both fixed)
          => tells whether the store-side RAW/WAR is observable in this probe at all, and
             whether the deployed `tma2` variant was silently relying on luck like §38.

  Part B  the fence changes nothing numerically: `tma2_full` vs `scalar_p` must be
          **bit-identical** on dh/dv2/dh0 (the fence only orders, no arithmetic), and vs a
          direct bf16 FLA `dhu` still 1 ulp.  Includes a non-divisible `T=250`.

  Part C  cost of the store-side fence (CUPTI us/call, idle card): `tma2_full/tma2`.

  Part D  after the fix, `tma2_full` NS=2/3 vs NS=1 are bit-identical (multi-stage now safe).

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_tma2_race.py \
        > .../harness/check_bwd_b1_tma2_race.out.txt 2>&1
"""
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dh_roundtrip import build, load, run_b1                        # noqa: E402
from check_bwd_dhu_dtype import cuda_kernel_us, make_inputs, as_dtype, fla_dhu  # noqa: E402

D = 128

# name -> extra nvcc flags.  `ETMA` is the common base; `TMADH`/`TMADV` turn on the TMA
# stores; `EPROXY` is the §39 input-side fence; `SPROXY` is the new P7-37 store-side fence.
VARIANTS = {
    "scalar":    ["-DB1T_ETMA"],
    "scalar_p":  ["-DB1T_ETMA", "-DB1T_EPROXY"],
    "tma2":      ["-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV"],
    "tma2_s":    ["-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV", "-DB1T_SPROXY"],
    "tma2_in":   ["-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV", "-DB1T_EPROXY"],
    "tma2_full": ["-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV", "-DB1T_EPROXY", "-DB1T_SPROXY"],
}

SHAPE = (8, 2048, 8)
CONFIGS = [(16, 2), (16, 3), (32, 3), (32, 2), (64, 2)]
NR = 40


def make_case(B, T, H):
    q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
    q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                 for x in (q, k, v, g_log2, beta, h0))
    torch.manual_seed(99)
    do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
    return q, k, v, g_log2, beta, h0, do


def part_a(libs):
    print("\n" + "=" * 108)
    print("Part A: determinism scan on the full `tma2` build matrix -- NR runs of the SAME")
    print(f"   kernel, `dh` compared run-to-run;  shape B{SHAPE[0]} T{SHAPE[1]} H{SHAPE[2]}, NR={NR}")
    print("   nbad>0 == NOT run-to-run reproducible (intermittent race)")
    print("=" * 108)
    B, T, H = SHAPE
    q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
    scale = D ** -0.5
    F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
    names = list(libs)
    hdr = f"  {'BT':>3s} {'NS':>3s} |" + "".join(f" {n:>10s}" for n in names)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (BT, NS) in CONFIGS:
        row = f"  {BT:3d} {NS:3d} |"
        for name in names:
            lib = libs[name]
            ref = None
            nbad = 0
            nrun = 0
            for _ in range(NR):
                try:
                    dh, dh0, dv2 = run_b1(lib, F["fi"], do, F["dv"], scale, BT, NS)
                except RuntimeError:
                    row += f" {'OOM':>10s}"
                    break
                nrun += 1
                if ref is None:
                    ref = dh.clone()
                d = (dh.float() - ref.float()).abs().max().item()
                if d > 0:
                    nbad += 1
                del dh, dh0, dv2
            else:
                row += f" {nbad:4d}/{nrun:<4d}".rjust(11)
        print(row)
    print("  (cell = #runs whose `dh` differs from run 0 / NR.  `scalar` is the known-racy")
    print("   control; `scalar_p`/`tma2_full` should be 0.  If `tma2_in` is also 0 everywhere")
    print("   then the store-side RAW/WAR is not observable in this probe -- the fence is")
    print("   still added as the documented requirement, gratis (Part C).)")


def part_b(libs, cases):
    print("\n" + "=" * 108)
    print("Part B: the store-side fence changes nothing numerically (fence = ordering only)")
    print("   `tma2_full` vs `scalar_p`: dh/dv2/dh0 must be bit-identical;  vs bf16 FLA: 1 ulp")
    print("=" * 108)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'dh max_abs':>11s} {'dv2 max_abs':>12s} "
          f"{'dh0 max_abs':>12s} | {'vsFLA dh':>9s} {'vsFLA dv2':>10s}")
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for BT in (32, 64):
            for NS in (1, 2):
                dh_f, dh0_f, dv2_f = run_b1(libs["tma2_full"], F["fi"], do, F["dv"], scale, BT, NS)
                dh_s, dh0_s, dv2_s = run_b1(libs["scalar_p"], F["fi"], do, F["dv"], scale, BT, NS)
                d_dh = (dh_f.float() - dh_s.float()).abs().max().item()
                d_dv2 = (dv2_f.float() - dv2_s.float()).abs().max().item()
                d_dh0 = (dh0_f - dh0_s).abs().max().item()
                # vs direct bf16 FLA dhu (bf16 stores -> 1 ulp).  FLA ref is chunk 64, so
                # `dh` (which carries the chunk axis) is only comparable at BT==64.
                d_fla_dv2 = (dv2_f.float() - F["dv2"].float()).abs().max().item()
                if BT == 64:
                    d_fla_dh = (dh_f.float() - F["dh"].float()).abs().max().item()
                    s_fla_dh = f"{d_fla_dh:9.2e}"
                else:
                    s_fla_dh = f"{'--':>9s}"
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {d_dh:11.2e} {d_dv2:12.2e} "
                      f"{d_dh0:12.2e} | {s_fla_dh} {d_fla_dv2:10.2e}")
                del dh_f, dh0_f, dv2_f, dh_s, dh0_s, dv2_s


def part_c(libs, cases):
    print("\n" + "=" * 108)
    print("Part C: cost of the store-side fence vs the unfixed deployed `tma2` (CUPTI us/call)")
    print("=" * 108)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'tma2':>9s} {'tma2_full':>10s} "
          f"{'scalar_p':>9s} | {'full/tma2':>10s} {'full/scalar_p':>13s}")
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for (BT, NS) in [(16, 1), (16, 2), (32, 1), (32, 3), (64, 1), (64, 2)]:
            try:
                us = {name: cuda_kernel_us(
                          lambda L=lib: run_b1(L, F["fi"], do, F["dv"], scale, BT, NS),
                          "b1_tma_kernel")
                      for name, lib in libs.items() if name in ("tma2", "tma2_full", "scalar_p")}
            except RuntimeError:
                print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} |  (OOM)")
                continue
            print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {us['tma2']:9.1f} "
                  f"{us['tma2_full']:10.1f} {us['scalar_p']:9.1f} | "
                  f"{us['tma2_full']/us['tma2']:10.3f} {us['scalar_p']/us['tma2_full']:13.3f}")


def part_d(lib, cases):
    print("\n" + "=" * 108)
    print("Part D: after the fix, multi-stage (NS=2/3) is bit-identical to NS=1")
    print("   `tma2_full` NS=2/3 vs `tma2_full` NS=1 (max_abs of dh;  -- = OOM)")
    print("=" * 108)
    print(f"  {'shape':16s} {'BT':>3s} | {'NS=2 vs NS=1':>14s} {'NS=3 vs NS=1':>14s}")
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for BT in (16, 32, 64):
            d1, _, _ = run_b1(lib, F["fi"], do, F["dv"], scale, BT, 1)
            outs = {}
            for NS in (2, 3):
                try:
                    p, _, _ = run_b1(lib, F["fi"], do, F["dv"], scale, BT, NS)
                except RuntimeError:
                    outs[NS] = None
                    continue
                outs[NS] = (p.float() - d1.float()).abs().max().item()
                del p
            s2 = f"{outs[2]:14.3e}" if outs[2] is not None else f"{'--':>14s}"
            s3 = f"{outs[3]:14.3e}" if outs[3] is not None else f"{'--':>14s}"
            print(f"  B{B} T{T} H{H:<6} {BT:3d} | {s2} {s3}")
            del d1


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for name, extra in VARIANTS.items():
        so = f"/tmp/b1tma2race_{name}.so"
        build("b1_tma_probe.cu", so, extra=["-lcuda"] + extra)
        libs[name] = load(so)

    part_a(libs)
    part_b(libs, [(2, 1024, 8), (8, 2048, 8), (1, 512, 4), (1, 250, 3)])
    part_c(libs, [(2, 1024, 8), (8, 2048, 8)])
    part_d(libs["tma2_full"], [(2, 1024, 8), (8, 2048, 8)])
    print("\nDONE")


if __name__ == "__main__":
    main()
