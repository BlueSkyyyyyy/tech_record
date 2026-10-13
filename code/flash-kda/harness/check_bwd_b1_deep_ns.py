"""P7-38: close the §40.6 boundary of P7-37.

§40.5 closed the `tma2` *store* side by adding `-DB1T_SPROXY`, but §40.6 flagged the one
remaining boundary as **待验证**: "更深 pipeline / 不同 store tile 形状改变写->store 时序时，
输出侧可观测性可能变化".  §40 measured the output-side RAW/WAR as unobservable at the
deployment shape (BT 16-64, NSTAGE 1-3, output double-buffer depth 2).  This script stresses
exactly those two knobs:

  * **deeper pipeline**  -- the probe now instantiates `NSTAGE` 4/5/6 (BT=16) and 4 (BT=32),
    so the producer's TMA refill window over `st->e` (the input-side WAR of §39) is widened.
  * **different store-buffer depth** -- new `-DB1T_SDEPTH=N` parametrizes the `dho`/`duo`
    output double buffer (`i % SDEPTH`, freed by `cp.async.bulk.wait_group.read SDEPTH-1`),
    which changes the generic-write -> async-`cp.async.bulk.tensor`-read timing of the tile
    (the output-side RAW of §40).

Parts:
  A  determinism scan (NR runs of the SAME kernel, `dh` compared run-to-run), large grid
     `B8T2048H8` (the §38/§39 failing regime), over {BT}x{NS}x{SDEPTH} and the build matrix:
       scalar      (-DB1T_ETMA)                     known-racy control (input side, §39)
       tma2*       (-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV)                    no fence at all
       tma2_full*  (+ -DB1T_EPROXY -DB1T_SPROXY)                            deployment
     nbad>0 == NOT run-to-run reproducible.  If the unfenced `tma2` stays 0/N even at the
     deepest NS/SDEPTH with the widest reuse window, the boundary is closed: the store-side
     fence is a programming-model requirement (kept in deployment) but not observable here.
  B  numerics: the deployment config at deep NS/SDEPTH is still bit-identical to the
     scalar-store path (`tma2_full` vs `scalar_p`) and 1 ulp vs a direct bf16 FLA `dhu`.
  C  cost: does a deeper pipeline / deeper store buffer help?  CUPTI us/call (idle card).
  D  after the fences, deep NS/SDEPTH vs the NS=1/SDEPTH=2 baseline is bit-identical.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_deep_ns.py \
        > .../harness/check_bwd_b1_deep_ns.out.txt 2>&1
"""
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dh_roundtrip import build, load, run_b1                         # noqa: E402
from check_bwd_dhu_dtype import cuda_kernel_us, make_inputs, as_dtype, fla_dhu  # noqa: E402

D = 128

_TMA2 = ["-DB1T_ETMA", "-DB1T_TMADH", "-DB1T_TMADV"]


def _variants():
    v = {"scalar": ["-DB1T_ETMA"], "scalar_p": ["-DB1T_ETMA", "-DB1T_EPROXY"]}
    for sd in (1, 2, 3, 4):
        v[f"tma2_sd{sd}"] = _TMA2 + [f"-DB1T_SDEPTH={sd}"]
        v[f"tma2_full_sd{sd}"] = _TMA2 + ["-DB1T_EPROXY", "-DB1T_SPROXY", f"-DB1T_SDEPTH={sd}"]
    return v


VARIANTS = _variants()

# {BT: [(NS,...)]} -- deepest the probe now instantiates.
GRID = {16: [1, 2, 3, 4, 5, 6], 32: [1, 2, 3, 4]}
SHAPE = (8, 2048, 8)
NR = 80
DEPTHS = (1, 2, 3, 4)


def make_case(B, T, H):
    q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
    q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                 for x in (q, k, v, g_log2, beta, h0))
    torch.manual_seed(99)
    do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
    return q, k, v, g_log2, beta, h0, do


def scan_cell(lib, F, do, scale, BT, NS):
    """Return (nbad, nrun) or None if the launch fails (smem too big / unsupported NS)."""
    ref = None
    nbad = nrun = 0
    for _ in range(NR):
        try:
            dh, dh0, dv2 = run_b1(lib, F["fi"], do, F["dv"], scale, BT, NS)
        except RuntimeError:
            return None
        nrun += 1
        if ref is None:
            ref = dh.clone()
        if (dh.float() - ref.float()).abs().max().item() > 0:
            nbad += 1
        del dh, dh0, dv2
    return nbad, nrun


def part_a(libs, F, do, scale):
    print("\n" + "=" * 100)
    print("Part A: determinism scan -- NR runs of the SAME kernel, `dh` compared run-to-run")
    print(f"   shape B{SHAPE[0]} T{SHAPE[1]} H{SHAPE[2]}, NR={NR};  nbad>0 == NOT reproducible")
    print("   `scalar` (input-side race, §39) is the control: it must show nbad>0 at NS>=2.")
    print("   Cell = nbad/NR.  N/A = launch rejected (smem too big or NS unsupported).  fenced variants are `_full`.")
    print("=" * 100)
    for sd in DEPTHS:
        print(f"\n  ---- store-buffer depth SDEPTH={sd} ----")
        names = [n for n in libs if n.endswith(f"sd{sd}")]
        if sd == 2:                       # scalar control is SDEPTH-independent -> once
            names = ["scalar", "scalar_p"] + names
        hdr = f"    {'BT':>3s} {'NS':>3s} |" + "".join(f" {n:>14s}" for n in names)
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for BT, nss in GRID.items():
            for NS in nss:
                row = f"    {BT:3d} {NS:3d} |"
                for name in names:
                    r = scan_cell(libs[name], F, do, scale, BT, NS)
                    cell = "N/A" if r is None else f"{r[0]}/{r[1]}"
                    row += f" {cell:>14s}"
                print(row)
        del names


def part_b(libs, cases):
    print("\n" + "=" * 100)
    print("Part B: deep NS/SDEPTH is numerically identical (fence & depth = ordering only)")
    print("   `tma2_full_sd3` NS=3 vs `scalar_p`: dh/dv2/dh0 bit-identical; vs bf16 FLA 1 ulp")
    print("=" * 100)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'dh max_abs':>11s} {'dv2 max_abs':>12s} "
          f"{'dh0 max_abs':>12s} | {'vsFLA dh':>9s} {'vsFLA dv2':>10s}")
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for BT, NS in [(16, 3), (64, 2)]:
            # BT=64's bigger stage can't fit SDEPTH=3 alongside NS=2 -> use SDEPTH=2 there.
            deep = libs["tma2_full_sd3"] if BT < 64 else libs["tma2_full_sd2"]
            dh_f, dh0_f, dv2_f = run_b1(deep, F["fi"], do, F["dv"], scale, BT, NS)
            dh_s, dh0_s, dv2_s = run_b1(libs["scalar_p"], F["fi"], do, F["dv"], scale, BT, NS)
            d_dh = (dh_f.float() - dh_s.float()).abs().max().item()
            d_dv2 = (dv2_f.float() - dv2_s.float()).abs().max().item()
            d_dh0 = (dh0_f - dh0_s).abs().max().item()
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
    print("\n" + "=" * 100)
    print("Part C: does a deeper pipeline / store buffer help?  CUPTI us/call (idle card)")
    print("   deployment `tma2_full`;  rows = SDEPTH, cols = NS (BT=16 left, BT=32 right)")
    print("=" * 100)
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        print(f"\n  shape B{B} T{T} H{H}")
        for sd in (1, 2, 3):
            lib = libs[f"tma2_full_sd{sd}"]
            cells = []
            for BT, nss in ((16, [1, 2, 3, 4, 5, 6]), (32, [1, 2, 3, 4])):
                for NS in nss:
                    try:
                        us = cuda_kernel_us(
                            lambda L=lib: run_b1(L, F["fi"], do, F["dv"], scale, BT, NS),
                            "b1_tma_kernel")
                        cells.append(f"B{BT} NS{NS}={us:6.1f}")
                    except RuntimeError:
                        cells.append(f"B{BT} NS{NS}=  N/A ")
            print(f"    SDEPTH={sd}: " + "  ".join(cells[:6]))
            print(f"              " + "  ".join(cells[6:]))
            del lib


def part_d(lib, cases):
    print("\n" + "=" * 100)
    print("Part D: after the fences, deep NS/SDEPTH vs the NS=1/SDEPTH=2 baseline is bit-identical")
    print("   `tma2_full` (max depth) NS=6/SDEPTH=3 vs NS=1 (same SDEPTH=3; max_abs of dh;  -- = launch failed)")
    print("=" * 100)
    print(f"  {'shape':16s} {'BT':>3s} | {'deep vs base':>14s}")
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for BT in (16, 32):
            d1, _, _ = run_b1(lib, F["fi"], do, F["dv"], scale, BT, 1)
            try:
                p, _, _ = run_b1(lib, F["fi"], do, F["dv"], scale, BT, 6 if BT == 16 else 4)
                d = (p.float() - d1.float()).abs().max().item()
                s = f"{d:14.3e}"
                del p
            except RuntimeError:
                s = f"{'--':>14s}"
            print(f"  B{B} T{T} H{H:<6} {BT:3d} | {s}")
            del d1


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for name, extra in VARIANTS.items():
        so = f"/tmp/b1deep_{name}.so"
        build("b1_tma_probe.cu", so, extra=["-lcuda"] + extra)
        libs[name] = load(so)

    B, T, H = SHAPE
    q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
    scale = D ** -0.5
    F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
    part_a(libs, F, do, scale)
    del q, k, v, g_log2, beta, h0, do, F

    part_b(libs, [(2, 1024, 8), (8, 2048, 8), (1, 512, 4), (1, 250, 3)])
    part_c(libs, [(2, 1024, 8), (8, 2048, 8)])
    part_d(libs["tma2_full_sd3"], [(2, 1024, 8), (8, 2048, 8)])
    print("\nDONE")


if __name__ == "__main__":
    main()
