"""ncu driver for the K2 occupancy experiment: run one FlashKDA K2 variant.

Loads the requested variant (see bench_k2_occ.VARIANTS) and issues warmup+prof
forward calls so ``ncu -k regex:flash_kda_fwd_recurrence`` can profile K2.

Usage (inside kernel_lab):
    ncu -k regex:flash_kda_fwd_recurrence -c 1 --launch-skip 3 \
        --section Occupancy --section SpeedOfLight \
        python ncu_driver_occ.py --variant cond_in2_out2 --B 8 --T 2048 --H 64
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_k2_occ as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    a = ap.parse_args()

    mod = B.load_variant(a.variant)
    tensors = B.make_inputs(a.B, a.T, a.H)
    fn, out = B.make_fn(mod, tensors)
    for _ in range(a.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(a.prof):
        fn()
    torch.cuda.synchronize()
    print(f"DONE variant={a.variant} B={a.B} T={a.T} H={a.H}")


if __name__ == "__main__":
    main()
