"""P7-31: driver so ``ncu`` can profile the two halves of the B1->B2 `dh` round-trip.

  --side b1   K2-shaped TMA B1 (`b1_tma_probe.cu`, prebuilt .so) with/without B1T_NODH
  --side b2   Triton B2 `wy_dqkg` (`build_kernel(nodh=...)`)

nvcc hangs under ncu, so prebuild the B1 .so outside ncu and pass --no-build:
    nvcc ... -DB1T_ETMA              b1_tma_probe.cu -o /tmp/b1_tma_e.so  -lcuda
    nvcc ... -DB1T_ETMA -DB1T_NODH   b1_tma_probe.cu -o /tmp/b1_tma_en.so -lcuda
"""
import argparse
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch                                                              # noqa: E402

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_dh_roundtrip import (                                       # noqa: E402
    build, load, run_b1, B1_BASE_SO, B1_NODH_SO,
)
from check_bwd_b1_tma_store import B1_TMA_SO                               # noqa: E402
from check_bwd_dhu_dtype import make_inputs, as_dtype, fla_dhu, cdiv      # noqa: E402


def synth_b1(B, T, H):
    q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
    q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                 for x in (q, k, v, g_log2, beta, h0))
    torch.manual_seed(99)
    do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
    scale = D ** -0.5
    F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
    return F, do, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["b1", "b2"], required=True)
    ap.add_argument("--nodh", action="store_true")
    ap.add_argument("--tma", action="store_true", help="B1: TMA `dh` store (B1T_TMADH)")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=64)
    ap.add_argument("--ns", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    B, T, H = args.B, args.T, args.H

    if args.side == "b1":
        so = B1_TMA_SO if args.tma else (B1_NODH_SO if args.nodh else B1_BASE_SO)
        if not args.no_build:
            extra = ["-lcuda", "-DB1T_ETMA"]
            if args.tma:
                extra.append("-DB1T_TMADH")
            elif args.nodh:
                extra.append("-DB1T_NODH")
            build("b1_tma_probe.cu", so, extra=extra)
        F, do, scale = synth_b1(B, T, H)
        lib = load(so)
        call = lambda: run_b1(lib, F["fi"], do, F["dv"], scale, args.bt, args.ns)
    else:
        import check_bwd_dqkg_kernel as B2
        k = B2.build_kernel(nodh=args.nodh)
        KV = 128
        inp = B2.make_inputs(B, T, H, KV, KV, 64, "cuda", dtype=torch.bfloat16, seed=1)
        inp = {kk: (vv if vv.is_contiguous() else vv.contiguous()) for kk, vv in inp.items()}
        sc = D ** -0.5
        call = lambda: B2.wy_dqkg_triton(k, inp, sc, 64)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE side={args.side} nodh={args.nodh} tma={args.tma} B={B} T={T} H={H} "
          f"bt={args.bt} ns={args.ns}")


if __name__ == "__main__":
    main()
