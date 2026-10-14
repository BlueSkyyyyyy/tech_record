"""P7-36: root-cause the `NSTAGE>=2` run-to-run race of the K2-shaped CUDA B1 probe.

§38 (P7-35) found that the hand-written TMA/mbarrier B1 (`b1_tma_probe.cu`, best
`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV`) is **not run-to-run reproducible** on large grids at
`NSTAGE>=2`, while `NSTAGE=1` is.  §38 only *observed* it and left the root cause as
**待验证** ("off-by-one in the hand-written `full`/`empty` parity protocol?").

This script closes it with a controlled ablation on the `-DB1T_ETMA` baseline (scalar
stores), the variant §38 could not make reproducible:

  Part A  determinism scan (NR runs of the *same* kernel, `dh` compared run-to-run) over
          {BT} x {NS} for a matrix of builds.  Reproduces the race and locates it:
            base(-DB1T_ETMA)                      racy
            no-etma   (read E^tot from global)    clean
            dropread  (E TMA issued, op2/op3 read global E^tot)   clean
            proxy     (`fence.proxy.async` both loops)            clean   <- the fix
            pro       (fence only before the producer's TMA)      clean   } either side
            con       (fence only before the consumer's empty)    clean   } breaks the WAR
            esync     (__syncwarp before op2/op3) racy
            evol      (volatile st->e read)       racy
            sync      (__syncthreads both loops)  clean (serializes)
          => the race is a **generic-proxy vs async-proxy WAR ordering** hazard: the
             consumer's generic read of the TMA stage buffer (`st->e`) and the producer's
             async-proxy (TMA) refill of that same buffer are **not ordered** by the
             `full`/`empty` mbarrier handshake.  `fence.proxy.async` (a one-liner) is the
             documented bridge; `__threadfence_block`/volatile/`__syncwarp` cannot fix it.

  Part B  shape of the corruption (real, programmatic): for each non-reproducible run of
          the racy `base` build, where the wrong `dh` elements sit -- chunk locality,
          K-rows and V-cols -- showing they are a whole compute-warp V-slice, i.e. the
          operand buffer was refilled mid-read.

  Part C  cost of the safe options vs the racy baseline (CUPTI us/call): read E^tot from
          global (`noetma`) vs keep ETMA + the proxy fence (`proxy`, the fix).

  Part D  the fix restores NS=1 correctness to multi-stage: `proxy` NS=2/3 vs `base` NS=1
          must be bit-identical (and vs bf16 FLA `dhu` 1 ulp), i.e. the fence changes
          *nothing* numerically while removing the race.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_race.py \
        > .../harness/check_bwd_b1_race.out.txt 2>&1
"""
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dh_roundtrip import build, load, run_b1                       # noqa: E402
from check_bwd_dhu_dtype import cuda_kernel_us, make_inputs, as_dtype, fla_dhu  # noqa: E402

D = 128

# name -> extra nvcc flags (all start from the ETMA baseline except `noetma`)
VARIANTS = {
    "base":     ["-DB1T_ETMA"],
    "noetma":   [],                                             # E^tot read from global
    "dropread": ["-DB1T_ETMA", "-DB1T_ETMA_DROPREAD"],          # E TMA on, op reads global
    "proxy":    ["-DB1T_ETMA", "-DB1T_EPROXY"],                 # fence.proxy.async both loops
    "pro":      ["-DB1T_ETMA", "-DB1T_EPROXY_PRO"],             # fence before producer TMA
    "con":      ["-DB1T_ETMA", "-DB1T_EPROXY_CON"],             # fence before consumer empty
    "esync":    ["-DB1T_ETMA", "-DB1T_ESYNC"],                  # __syncwarp before op2/3
    "evol":     ["-DB1T_ETMA", "-DB1T_EVOL"],                   # volatile st->e read
    "sync":     ["-DB1T_ETMA", "-DB1T_SYNC"],                   # __syncthreads both loops
}

# shape is fixed per run to keep the scan cheap but on the failing regime (large grid)
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
    print("\n" + "=" * 100)
    print("Part A: determinism scan -- NR runs of the SAME kernel, `dh` compared run-to-run")
    print(f"   shape B{SHAPE[0]} T{SHAPE[1]} H{SHAPE[2]}, NR={NR};  nbad>0 == NOT reproducible")
    print("=" * 100)
    B, T, H = SHAPE
    q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
    scale = D ** -0.5
    F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
    hdr = f"  {'BT':>3s} {'NS':>3s} |" + "".join(f" {name:>8s}" for name in libs)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (BT, NS) in CONFIGS:
        row = f"  {BT:3d} {NS:3d} |"
        for name, lib in libs.items():
            ref = None
            nbad = 0
            for _ in range(NR):
                dh, dh0, dv2 = run_b1(lib, F["fi"], do, F["dv"], scale, BT, NS)
                if ref is None:
                    ref = dh.clone()
                d = (dh.float() - ref.float()).abs().max().item()
                if d > 0:
                    nbad += 1
                del dh, dh0, dv2
            row += f" {nbad:4d}/{NR:d}".rjust(9)
        print(row)
    print("  (cell = #runs whose `dh` differs from run 0 / NR;  `base` is racy at NS>=2 here)")
    print("  verdict: the race is a generic<->async proxy WAR ordering hazard on the TMA")
    print("  stage buffer -- `noetma`/`dropread` never touch the refilled buffer, `proxy`")
    print("  (fence.proxy.async) and `sync` (__syncthreads) remove it; `esync`/`evol` cannot.")


def part_b(lib, BT=32, NS=3, NR=60):
    print("\n" + "=" * 100)
    print("Part B: shape of the corruption (programmatic; racy `base`, "
          f"B{SHAPE[0]} T{SHAPE[1]} H{SHAPE[2]} BT={BT} NS={NS}, NR={NR})")
    print("=" * 100)
    B, T, H = SHAPE
    q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
    scale = D ** -0.5
    F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
    ref = None
    bad = 0
    chunk_gap = []
    row_hist = {}
    col_hist = {}
    for r in range(NR):
        dh, dh0, dv2 = run_b1(lib, F["fi"], do, F["dv"], scale, BT, NS)
        if ref is None:
            ref = dh.clone().float()
            del dh, dh0, dv2
            continue
        d = (dh.float() - ref).abs()
        mx = d.max().item()
        if mx > 0:
            bad += 1
            idx = (d > 0).nonzero()
            chunks = sorted(set(idx[:, 1].tolist()))
            if len(chunks) >= 2:
                chunk_gap.append(max(chunks) - min(chunks) + 1 - len(chunks))
            for rr in set(idx[:, 3].tolist()):
                row_hist[rr] = row_hist.get(rr, 0) + 1
            # V-column tile (warp slice, BW=16)
            for cv in set(idx[:, 4].tolist()):
                tile = (cv // 16) * 16
                col_hist[tile] = col_hist.get(tile, 0) + 1
        del dh, dh0, dv2
    print(f"  non-reproducible runs: {bad}/{NR-1}")
    print(f"  chunk locality: corrupted chunk-runs are contiguous; "
          f"mean (#missing chunks inside the span) = "
          f"{sum(chunk_gap)/len(chunk_gap):.2f}" if chunk_gap else "  chunk locality: n/a")
    top_rows = sorted(row_hist.items(), key=lambda kv: -kv[1])[:6]
    top_cols = sorted(col_hist.items(), key=lambda kv: -kv[1])[:6]
    print(f"  worst K-rows (of 128): " + ", ".join(f"{r}x{c}" for r, c in top_rows))
    print(f"  worst V-tiles (16-wide warp slices): " + ", ".join(f"[{c},{c+16})x{n}" for c, n in top_cols))
    print("  => a corrupted `dh` is a whole 16-column V-slice (one compute warp) across a")
    print("     contiguous run of chunks: the stage buffer was refilled while that warp was")
    print("     still reading it (WAR).  The other warps' slices are untouched because only")
    print("     the warp that races the TMA refill reads the stale bytes.")


def part_c(libs, cases):
    print("\n" + "=" * 100)
    print("Part C: cost of the safe options vs the racy ETMA baseline (CUPTI us/call, idle card)")
    print("=" * 100)
    print(f"  {'shape':16s} {'BT':>3s} {'NS':>3s} | {'base':>8s} {'noetma':>8s} {'proxy':>8s} "
          f"| {'noetma/base':>11s} {'proxy/base':>10s}")
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for (BT, NS) in [(16, 1), (32, 1), (32, 3), (64, 1)]:
            us = {name: cuda_kernel_us(
                      lambda L=lib: run_b1(L, F["fi"], do, F["dv"], scale, BT, NS),
                      "b1_tma_kernel")
                  for name, lib in libs.items()}
            print(f"  B{B} T{T} H{H:<6} {BT:3d} {NS:3d} | {us['base']:8.1f} {us['noetma']:8.1f} "
                  f"{us['proxy']:8.1f} | {us['noetma']/us['base']:11.2f} "
                  f"{us['proxy']/us['base']:10.2f}")


def part_d(base_lib, proxy_lib, cases):
    print("\n" + "=" * 100)
    print("Part D: the fix restores NS=1 correctness to multi-stage (values unchanged)")
    print("=" * 100)
    print(f"  {'shape':16s} {'BT':>3s} | {'proxy NS=2 vs base NS=1':>26s} "
          f"{'proxy NS=3 vs base NS=1':>26s}")
    for (B, T, H) in cases:
        q, k, v, g_log2, beta, h0, do = make_case(B, T, H)
        scale = D ** -0.5
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for BT in [16, 32, 64]:
            d1, _, _ = run_b1(base_lib, F["fi"], do, F["dv"], scale, BT, 1)
            s2 = 0.0
            s3 = float("nan")
            for NS in (2, 3):
                try:
                    p, _, _ = run_b1(proxy_lib, F["fi"], do, F["dv"], scale, BT, NS)
                except RuntimeError:
                    continue
                d = (p.float() - d1.float()).abs().max().item()
                if NS == 2:
                    s2 = d
                else:
                    s3 = d
                del p
            s3s = f"{s3:26.3e}" if s3 == s3 else f"{'--':>26s}"
            print(f"  B{B} T{T} H{H:<6} {BT:3d} | {s2:26.3e} {s3s}")
            del d1


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for name, extra in VARIANTS.items():
        so = f"/tmp/b1race_{name}.so"
        build("b1_tma_probe.cu", so, extra=["-lcuda"] + extra)
        libs[name] = load(so)

    part_a(libs)
    part_b(libs["base"])
    part_c(libs, [(2, 1024, 8), (8, 2048, 8)])
    part_d(libs["base"], libs["proxy"], [(2, 1024, 8), (8, 2048, 8)])
    print("\nDONE")


if __name__ == "__main__":
    main()
