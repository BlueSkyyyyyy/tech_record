"""P7-25: minimal driver so ``ncu`` can profile the *late-load* B2 (`wy_dqkg`) rewrite
against the baseline, in isolation.

Synthesizes workspace-shaped operands (values irrelevant) and launches exactly one B2
variant, so `ncu -k regex:_wy_dqkg_kernel` targets it unambiguously.

Usage (inside kernel_lab):
    python ncu_b2_struct_driver.py --which late --B 8 --T 2048 --H 8 --bt 16 --nw 4
    ncu -k regex:_wy_dqkg_kernel -c 1 --launch-skip 3 --set full \
        python ncu_b2_struct_driver.py --which late --B 8 --T 2048 --H 8 --bt 16
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

from check_bwd_dqkg_kernel import build_kernel as build_base, wy_dqkg_triton as run_base  # noqa: E402
from check_bwd_b2_struct import build_dqkg_late, wy_dqkg_late_triton as run_late         # noqa: E402


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
    return scale, dict(q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h,
                       do=do, dh=dh, dv=dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["base", "late"], required=True)
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
    scale, inp = synth(args.B, args.T, args.H, args.bt, dev, dt)

    if args.which == "base":
        k = build_base()
        call = lambda: run_base(k, inp, scale, args.bt, num_warps=args.nw, num_stages=args.ns)
    else:
        k = build_dqkg_late()
        call = lambda: run_late(k, inp, scale, args.bt, num_warps=args.nw, num_stages=args.ns)

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
