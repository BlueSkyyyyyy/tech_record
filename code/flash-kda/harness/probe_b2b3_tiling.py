"""Diagnostic 2: B2/B3 *tiling* levers (BK, BV for B2; BC for B3) at BT=16.

Run: CUDA_VISIBLE_DEVICES=1 python .../probe_b2b3_tiling.py
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_dqkg_kernel import build_kernel as build_dqkg  # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra  # noqa: E402
from check_bwd_b1_bf16_e2e import prep, profile_dev  # noqa: E402
from check_bwd_ws_reuse import D, CHUNK, cdiv  # noqa: E402
from check_bwd_dqkg_kernel import wy_dqkg_triton  # noqa: E402
from check_bwd_intra_kernel import intra_triton  # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton  # noqa: E402


def b2_launch(k, inp, scale, BT, BK, BV, nw=4):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    k[(NT, B * H)](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
        inp["do"], inp["dh"], inp["dv"], dq, dk, dv2, db, dg, dA,
        T, NT, scale, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, num_warps=nw, num_stages=1)
    return dict(dq=dq, dk=dk, dv=dv2, db=db, dg=dg, dA=dA)


def b3_launch(k, inp, BT, BC, nw=4):
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    NC = cdiv(BT, BC)
    BK = min(32, triton.next_power_of_2(K))
    NK = cdiv(K, BK)
    dev = inp["q"].device
    dq = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dg = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    k[(NK * NC, NT, B * H)](
        inp["q"], inp["k"], inp["g"], inp["beta"], inp["dAqk"], inp["dAkk"], dq, dk, db, dg,
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, num_warps=nw, num_stages=1)
    return dict(dq=dq, dk=dk, db=db, dg=dg)


def main():
    scale = 1.0 / math.sqrt(D)
    dqkg_k = build_dqkg()
    intra_k = build_intra()
    dav_k = build_dav_kernel()
    for (name, B, T, H, seed, gs) in [
        ("T=512 H=4", 1, 512, 4, 0, 1.0),
        ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
    ]:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, -5.0, seed, gs)
        _, dv2 = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh = torch.randn(B, cdiv(T, CHUNK), H, D, D, device="cuda")
        inp2 = dict(q=qn, k=kn, v=inp["v"].float(), v_new=fi["v_new"], g=fi["g_cum"],
                    beta=fi["beta"], A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
        print(f"\n# B2 {name}")
        for (BK, BV) in [(64, 64), (128, 64), (64, 128), (128, 128), (32, 128), (128, 32)]:
            for nw in (4, 8):
                try:
                    fn = lambda: b2_launch(dqkg_k, inp2, scale, CHUNK, BK, BV, nw)
                    fn(); torch.cuda.synchronize()
                    d, _ = profile_dev(fn, reps=15)
                    print(f"  BK={BK:3d} BV={BV:3d} nw={nw}: {d:7.1f} us")
                except Exception as e:
                    print(f"  BK={BK:3d} BV={BV:3d} nw={nw}: ERR {type(e).__name__}")
        dAqk, _ = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"], dAqk=dAqk,
                    dAkk=torch.randn_like(dAqk))
        print(f"# B3 {name}")
        for BC in (8, 16, 32):
            for nw in (2, 4):
                try:
                    fn = lambda: b3_launch(intra_k, inp3, CHUNK, BC, nw)
                    fn(); torch.cuda.synchronize()
                    d, _ = profile_dev(fn, reps=15)
                    print(f"  BC={BC:3d} nw={nw}: {d:7.1f} us")
                except Exception as e:
                    print(f"  BC={BC:3d} nw={nw}: ERR {type(e).__name__}")
    print("\nDONE")


if __name__ == "__main__":
    main()
