"""P7-45: minimal driver so ``ncu`` can profile each ablation variant of the fused
B2/B3 kernel (see check_bwd_b2b3_seg.py) to confirm the byte traffic is preserved and
locate each segment's bound.  fp32 operands to match the deployed chain.

Usage (inside kernel_lab):
    python ncu_b2b3_seg_driver.py --variant no_post --B 8 --T 2048 --H 8
    ncu -k regex:_seg_b2b3_kernel -c 1 --launch-skip 3 --set full python ...
"""
import argparse
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_b2b3_seg import build_seg_kernel, seg_triton, VARIANTS   # noqa: E402


def synth(B, T, H, bt, dev, kdim=128, vdim=128):
    scale = kdim ** -0.5
    NT = (T + bt - 1) // bt
    q = torch.nn.functional.normalize(torch.randn(B, T, H, kdim, device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, kdim, device=dev), dim=-1)
    v = torch.randn(B, T, H, vdim, device=dev)
    v_new = torch.randn(B, T, H, vdim, device=dev)
    g = -torch.rand(B, T, H, kdim, device=dev) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev))
    A = torch.randn(B, T, H, bt, device=dev)
    h = torch.randn(B, NT, H, kdim, vdim, device=dev)
    do = torch.randn(B, T, H, vdim, device=dev)
    dh = torch.randn(B, NT, H, kdim, vdim, device=dev)
    dv = torch.randn(B, T, H, vdim, device=dev)
    dAqk = torch.randn(B, T, H, bt, device=dev)
    inp2 = dict(q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h, do=do, dh=dh, dv=dv)
    return scale, inp2, dAqk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--nw", type=int, default=8)
    ap.add_argument("--bv", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    args = ap.parse_args()

    scale, inp2, dAqk = synth(args.B, args.T, args.H, args.bt, "cuda", args.K, args.V)
    k = build_seg_kernel()

    def call():
        seg_triton(k, inp2, dAqk, scale, args.bt, args.variant, BV=args.bv, num_warps=args.nw)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print("DONE")


if __name__ == "__main__":
    main()
