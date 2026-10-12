"""Diagnostic 3: B2 is DRAM-bound and its dominant traffic is h/dh (fp32 in the chain).
Does feeding B2 bf16 h/dh (halving those bytes) move it?  Quick check.

Run: CUDA_VISIBLE_DEVICES=1 python .../probe_b2b3_hdh_bf16.py
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_b1_bf16_e2e import prep, profile_dev  # noqa: E402
from check_bwd_ws_reuse import D, CHUNK, cdiv, stats  # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton  # noqa: E402


def main():
    scale = 1.0 / math.sqrt(D)
    k = build_dqkg()
    dav_k = build_dav_kernel()
    for (name, B, T, H, seed, gs) in [("T=512 H=4", 1, 512, 4, 0, 1.0),
                                      ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0)]:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, -5.0, seed, gs)
        _, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh = torch.randn(B, cdiv(T, CHUNK), H, D, D, device="cuda") * 0.5
        base = dict(q=qn, k=kn, v=inp["v"].float(), v_new=fi["v_new"], g=fi["g_cum"],
                    beta=fi["beta"], A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv)
        bf = dict(base)
        bf["h"] = base["h"].to(torch.bfloat16)
        bf["dh"] = dh.to(torch.bfloat16)
        print(f"\n# {name}  h dtype={base['h'].dtype}  dh dtype={base['dh'].dtype}")
        for label, d in [("fp32 h/dh", base), ("bf16 h/dh", bf)]:
            fn = lambda: wy_dqkg_triton(k, d, scale, CHUNK, num_warps=4)
            fn(); torch.cuda.synchronize()
            t, _ = profile_dev(fn, reps=15)
            print(f"  {label:10s}: {t:7.1f} us")
        r0 = wy_dqkg_triton(k, base, scale, CHUNK, num_warps=4)
        r1 = wy_dqkg_triton(k, bf, scale, CHUNK, num_warps=4)
        print("  bf16-vs-fp32 operand delta:")
        for fld in ["dq", "dk", "dv", "db", "dg"]:
            stats(f"    {fld}", r1[fld], r0[fld])
    print("\nDONE")


if __name__ == "__main__":
    main()
