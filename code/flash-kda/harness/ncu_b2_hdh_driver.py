"""P7-26: a minimal driver so ``ncu`` can profile B2 ``wy_dqkg`` (late-load, P7-25) under
the *deployed dtype mix* -- something the earlier B2 profiles (P7-23 §26.4, P7-25 §28.6)
never did (they fed all-bf16 operands).

Deployed mix, read straight off the assembled chain (`check_bwd_b2_hdh.py`):
    q, k, v, do, dv, h, dh  : fp32
    v_new (U), g, A, beta   : bf16

Two orthogonal flags let us attribute the bytes:
  --hdh {fp32,bf16}   h/dh operand dtype (the §26.6 item-(1) lever).
  --bk  N             K-tile; BK=K removes the `i_k` loop, hence the `K/BK`-fold re-read
                      of the V-side operands v_new/do/dv/v (§26.6 item (2)).

Usage (inside kernel_lab, idle card):
    CUDA_VISIBLE_DEVICES=1 python ncu_b2_hdh_driver.py --hdh fp32 --bk 64
    ncu -k regex:_wy_dqkg_kernel_late -c 1 --launch-skip 3 \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
        python ncu_b2_hdh_driver.py --hdh fp32 --bk 64
"""
import argparse
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_ws_reuse import cdiv                                            # noqa: E402
from check_bwd_b2_struct import build_dqkg_late                                # noqa: E402


def synth(B, T, H, bt, dev, hdh_dtype, bk):
    """Deployed dtype mix: ALL B2 operands are fp32 in the assembled chain (P6-8/P7-25);
    only h/dh are toggled here.  (The chain keeps `fi["v_new"/"g_cum"/"beta"/"Akk"]` fp32;
    `_prep_b` binarizes only qg/kg/w, which B2 does not read.)"""
    scale = D ** -0.5
    import torch.nn.functional as F
    NT = cdiv(T, bt)
    bf = lambda x: x.to(torch.bfloat16).float()          # bf16-valued, stored fp32
    q = bf(F.normalize(torch.randn(B, T, H, D, device=dev), dim=-1))
    k = bf(F.normalize(torch.randn(B, T, H, D, device=dev), dim=-1))
    v = bf(torch.randn(B, T, H, D, device=dev))
    v_new = bf(torch.randn(B, T, H, D, device=dev))
    g = -torch.rand(B, T, H, D, device=dev) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev))
    A = torch.randn(B, T, H, bt, device=dev) * 0.1
    h = bf(torch.randn(B, NT, H, D, D, device=dev) * 0.5).to(hdh_dtype)
    do = bf(torch.randn(B, T, H, D, device=dev))
    dh = bf(torch.randn(B, NT, H, D, D, device=dev) * 0.5).to(hdh_dtype)
    dv = bf(torch.randn(B, T, H, D, device=dev))
    return scale, dict(q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h,
                       do=do, dh=dh, dv=dv), bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--hdh", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--bv", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    args = ap.parse_args()

    dt = torch.float32 if args.hdh == "fp32" else torch.bfloat16
    dev = "cuda"
    scale, inp, _ = synth(args.B, args.T, args.H, args.bt, dev, dt, args.bk)
    kern = build_dqkg_late()
    NT = cdiv(args.T, args.bt)

    def call():
        kern[(NT, args.B * args.H)](
            inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
            inp["do"], inp["dh"], inp["dv"],
            torch.empty(args.B, args.T, args.H, D, device=dev),
            torch.empty(args.B, args.T, args.H, D, device=dev),
            torch.empty(args.B, args.T, args.H, D, device=dev),
            torch.empty(args.B, args.T, args.H, device=dev),
            torch.empty(args.B, args.T, args.H, D, device=dev),
            torch.empty(args.B, args.T, args.H, args.bt, device=dev),
            args.T, NT, scale, H=args.H, K=D, V=D, BT=args.bt,
            BK=args.bk, BV=args.bv, num_warps=args.nw, num_stages=1)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE hdh={args.hdh} bk={args.bk} bv={args.bv} nw={args.nw} "
          f"B={args.B} T={args.T} H={args.H} bt={args.bt}")


if __name__ == "__main__":
    main()
