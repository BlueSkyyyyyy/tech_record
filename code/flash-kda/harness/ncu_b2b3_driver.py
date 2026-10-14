"""P7-23: minimal driver so ``ncu`` can profile the two *per-chunk* backward GEMM kernels
B2 (`wy_dqkg`) and B3 (`intra`) inside the assembled chain.

P7-13/P7-16/P7-19 profiled B1 (the cross-chunk state adjoint) only; the P7-22 chain split
shows B2+B3 = ~66% of the backward device time at deployment granularity BT=16, so their
hardware bound is now the open question.  This driver synthesizes workspace-shaped operands
(values irrelevant, only shape/dtype/tiling) and launches exactly one of the two kernels,
so ``ncu -k regex:<name>`` targets it unambiguously.

Usage (inside kernel_lab):
    python ncu_b2b3_driver.py --which b2 --B 8 --T 2048 --H 8 --bt 16 --nw 4
    ncu -k regex:_wy_dqkg_kernel -c 1 --launch-skip 3 --set full \\
        python ncu_b2b3_driver.py --which b2 --B 8 --T 2048 --H 8 --bt 16
"""
import argparse
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

D = 128
HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton  # noqa: E402


def synth(B, T, H, bt, dev, dt):
    scale = D ** -0.5
    NT = (T + bt - 1) // bt
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1).to(dt)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1).to(dt)
    v = torch.randn(B, T, H, D, device=dev).to(dt)
    v_new = torch.randn(B, T, H, D, device=dev).to(dt)
    g = (-torch.rand(B, T, H, D, device=dev) * 0.5).to(dt)
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev)).to(dt)
    A = torch.randn(B, T, H, bt, device=dev).to(dt)
    h = torch.randn(B, NT, H, D, D, device=dev).to(dt)
    do = torch.randn(B, T, H, D, device=dev).to(dt)
    dh = torch.randn(B, NT, H, D, D, device=dev).to(dt)
    dv = torch.randn(B, T, H, D, device=dev).to(dt)
    dAqk = torch.randn(B, T, H, bt, device=dev).to(dt)
    dAkk = torch.randn(B, T, H, bt, device=dev).to(dt)
    inp2 = dict(q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h, do=do, dh=dh, dv=dv)
    inp3 = dict(q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk)
    return scale, inp2, inp3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["b2", "b3"], required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--ns", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    args = ap.parse_args()

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    dev = "cuda"
    scale, inp2, inp3 = synth(args.B, args.T, args.H, args.bt, dev, dt)

    if args.which == "b2":
        k = build_dqkg()
        call = lambda: wy_dqkg_triton(k, inp2, scale, args.bt, num_warps=args.nw,
                                      num_stages=args.ns)
    else:
        k = build_intra()
        call = lambda: intra_triton(k, inp3, args.bt, args.bt, num_warps=args.nw,
                                    num_stages=args.ns)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={args.B} T={args.T} H={args.H} bt={args.bt} "
          f"nw={args.nw} ns={args.ns} dtype={args.dtype}")


if __name__ == "__main__":
    main()
