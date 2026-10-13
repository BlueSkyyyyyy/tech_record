"""P7-30: minimal driver so ``ncu`` can profile the *fused* B2+B3 kernel alongside the
unfused B2 (`wy_dqkg`) and B3 (`intra`) at deployment granularity, to check the traffic
claim (fusion removes the `dq/dk/dg` + `dA` round trip).

Operands are fp32 to match the deployed chain (P7-26 §29.1: only the B1 operands are
bf16; B2/B3 run fp32).

Usage (inside kernel_lab):
    python ncu_b2b3_fused_driver.py --which fused --B 8 --T 2048 --H 8 --bt 16 --nw 8
    ncu -k regex:_fused_b2b3_kernel -c 1 --launch-skip 3 --set full \\
        python ncu_b2b3_fused_driver.py --which fused --B 8 --T 2048 --H 8
"""
import argparse
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_b3_mma import build_intra_mma, intra_triton_mma                 # noqa: E402
from check_bwd_b2b3_fused import build_fused_b2b3, fused_b2b3_triton           # noqa: E402


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
    inp2 = dict(q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h, do=do, dh=dh, dv=dv)
    return scale, inp2, dAqk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["b2", "b3", "fused"], required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--nw", type=int, default=8)
    ap.add_argument("--bv", type=int, default=128)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    args = ap.parse_args()

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    dev = "cuda"
    scale, inp2, dAqk = synth(args.B, args.T, args.H, args.bt, dev, dt)

    if args.which == "b2":
        k = build_dqkg()
        call = lambda: wy_dqkg_triton(k, inp2, scale, args.bt, num_warps=args.nw)
    elif args.which == "b3":
        b2 = wy_dqkg_triton(build_dqkg(), inp2, scale, args.bt, num_warps=4)
        inp3 = dict(q=inp2["q"], k=inp2["k"], g=inp2["g"], beta=inp2["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        k = build_intra_mma()
        call = lambda: intra_triton_mma(k, inp3, args.bt, args.bt, num_warps=args.nw)
    else:
        k = build_fused_b2b3()
        call = lambda: fused_b2b3_triton(k, inp2, dAqk, scale, args.bt, BV=args.bv,
                                         num_warps=args.nw)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print("DONE")


if __name__ == "__main__":
    main()
