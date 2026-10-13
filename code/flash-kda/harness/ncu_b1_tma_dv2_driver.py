"""P7-33: ncu driver for the B1 `dv2` TMA-store variants (docs/07 §36).

nvcc hangs under `ncu`, so prebuild the .so outside ncu and pass `--no-build`.  Variant
name matches `check_bwd_b1_tma_dv2.VARIANTS`; the corresponding .so is
``/tmp/b1t_<variant>.so`` (built by ``check_bwd_b1_tma_dv2.py`` or the probe script).

    python ncu_b1_tma_dv2_driver.py --variant tma2 --B 8 --T 2048 --H 8 --bt 64 --ns 1 --no-build
"""
import argparse
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch                                                               # noqa: E402

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_dh_roundtrip import build, load, run_b1                     # noqa: E402
from check_bwd_b1_tma_dv2 import VARIANTS                                  # noqa: E402
from check_bwd_dhu_dtype import make_inputs, as_dtype, fla_dhu             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=64)
    ap.add_argument("--ns", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    so = f"/tmp/b1t_{args.variant}.so"
    if not args.no_build:
        build("b1_tma_probe.cu", so, extra=["-lcuda"] + VARIANTS[args.variant])
    lib = load(so)

    B, T, H = args.B, args.T, args.H
    q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
    q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                 for x in (q, k, v, g_log2, beta, h0))
    torch.manual_seed(99)
    do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
    scale = D ** -0.5
    F = fla_dhu(q, k, v, g_log2, beta, h0, scale, args.bt, do)

    call = lambda: run_b1(lib, F["fi"], do, F["dv"], scale, args.bt, args.ns)
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE variant={args.variant} B={B} T={T} H={H} bt={args.bt} ns={args.ns}")


if __name__ == "__main__":
    main()
