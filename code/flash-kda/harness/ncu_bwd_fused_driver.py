"""P7-27: minimal driver so ``ncu`` can profile the fused B1+B2 kernel (`_fused_b12_kernel`)
and the unfused pair (`_dhu_kernel` + `_wy_dqkg_kernel`) on identical operands, for DRAM-byte
attribution of the fusion.

Usage (inside kernel_lab):
    python ncu_bwd_fused_driver.py --which fused --B 8 --T 2048 --H 8
    ncu -k regex:"_fused_b12_kernel" -c 1 --launch-skip 2 \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,... \
        python ncu_bwd_fused_driver.py --which fused --B 8 --T 2048 --H 8
"""
import argparse
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
import torch  # noqa: E402

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)
from check_bwd_b1_bf16_e2e import prep, build_dhu_bf16, dhu_bf16_triton      # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                       # noqa: E402
from check_bwd_fused import build_fused, fused_b12_triton                    # noqa: E402
from check_bwd_ws_reuse import D, CHUNK                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["fused", "pair"], required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    args = ap.parse_args()

    scale = 1.0 / math.sqrt(D)
    inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(args.B, args.T, args.H, scale, -5.0,
                                                      args.seed, 1.0)
    fi2 = dict(fi)
    fi2["v_raw"] = inp["v"].float()
    dav_k = build_dav_kernel()
    _, dvin = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)

    if args.which == "fused":
        k = build_fused()

        def call():
            fused_b12_triton(k, fi2, qn, kn, do, dvin, scale, CHUNK, dht=dfin)
    else:
        dhu_k = build_dhu_bf16()
        dqkg_k = build_dqkg()
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        v_f = inp["v"].float()

        def call():
            dh, dh0, dv2 = dhu_bf16_triton(dhu_k, fi_b, do, dvin, scale, CHUNK, dht=dfin)
            inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"],
                        beta=fi["beta"], A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
            wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={args.B} T={args.T} H={args.H}")


if __name__ == "__main__":
    main()
