"""P7-24: minimal `ncu` driver for the B3 `intra` MMA-diagonal rewrite.

Launches exactly one B3 variant (elementwise baseline `_intra_kernel` or the MMA-diagonal
`_intra_kernel_mma`) on synthesized workspace-shaped operands so `ncu -k regex:<name>`
targets it unambiguously -- the P7-23 B2/B3 driver pattern.

Usage (inside kernel_lab):
    python ncu_b3_mma_driver.py --which base --B 8 --T 2048 --H 8 --bt 16 --nw 4
    ncu -k regex:_intra_kernel_mma -c 1 --launch-skip 3 --set full \\
        python ncu_b3_mma_driver.py --which mma --B 8 --T 2048 --H 8 --bt 16
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


def synth(B, T, H, bt, dev, dt):
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1).to(dt)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1).to(dt)
    g = (-torch.rand(B, T, H, D, device=dev) * 0.5).to(dt)
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev)).to(dt)
    dAqk = torch.randn(B, T, H, bt, device=dev).to(dt)
    dAkk = torch.randn(B, T, H, bt, device=dev).to(dt)
    return dict(q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["base", "mma"], required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    args = ap.parse_args()

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    inp = synth(args.B, args.T, args.H, args.bt, "cuda", dt)
    if args.which == "base":
        k = build_intra()
        call = lambda: intra_triton(k, inp, args.bt, args.bt, num_warps=args.nw)
    else:
        k = build_intra_mma()
        call = lambda: intra_triton_mma(k, inp, args.bt, args.bt, num_warps=args.nw)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={args.B} T={args.T} H={args.H} bt={args.bt} "
          f"nw={args.nw} dtype={args.dtype}")


if __name__ == "__main__":
    main()
