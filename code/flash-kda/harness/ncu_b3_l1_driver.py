"""P7-29: minimal `ncu` driver for the B3 `intra` L1-locality variants.

Launches exactly one B3 variant so `ncu -k regex:<name>` targets it unambiguously:
  base  -> _intra_kernel           (P7-23 elementwise baseline)
  mma   -> _intra_kernel_mma       (P7-24 MMA-diagonal)
  loop  -> _intra_kernel_loop      (P7-29: k-blocks looped + software-pipelined)

Usage (inside kernel_lab):
    python ncu_b3_l1_driver.py --which loop --B 8 --T 2048 --H 8 --bt 16 --nw 4 --bk 64 --ns 2
    ncu -k regex:_intra_kernel_loop -c 1 --launch-skip 3 --set full \\
        python ncu_b3_l1_driver.py --which loop ...
"""
import argparse
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton          # noqa: E402
from check_bwd_b3_mma import build_intra_mma, intra_triton_mma                          # noqa: E402
from ncu_b3_mma_driver import synth                                                      # noqa: E402
from check_bwd_b3_l1 import build_intra_loop, intra_triton_loop                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["base", "mma", "loop"], required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    args = ap.parse_args()

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    inp = synth(args.B, args.T, args.H, args.bt, "cuda", dt)
    if args.which == "base":
        k = build_intra()
        call = lambda: intra_triton(k, inp, args.bt, args.bt, num_warps=args.nw)
    elif args.which == "mma":
        k = build_intra_mma()
        call = lambda: intra_triton_mma(k, inp, args.bt, args.bt, num_warps=args.nw)
    else:
        k = build_intra_loop()
        call = lambda: intra_triton_loop(k, inp, args.bt, args.bt, bk=args.bk,
                                         num_warps=args.nw, num_stages=args.ns)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={args.B} T={args.T} H={args.H} bt={args.bt} "
          f"nw={args.nw} bk={args.bk} ns={args.ns} dtype={args.dtype}")


if __name__ == "__main__":
    main()
