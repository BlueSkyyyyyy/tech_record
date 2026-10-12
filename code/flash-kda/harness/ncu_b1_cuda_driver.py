"""Driver for `ncu` on the P7-15 hand-written CUDA B1 kernel.

Runs exactly one kernel launch (B=1, T=512, H=4, BT=64, NSTAGE=1) so `ncu -k regex:b1_dhu_cuda
-c 1` profiles a single, deterministic launch.

    CUDA_VISIBLE_DEVICES=1 ncu -k regex:b1_dhu_cuda -c 1 --launch-skip 2 \
        --section SpeedOfLight --section WarpStateStats --section Occupancy \
        --section SchedulerStats python .../harness/ncu_b1_cuda_driver.py
"""
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

import torch                                                              # noqa: E402
from check_bwd_b1_cuda import build, load, run_cuda, D                     # noqa: E402
from check_bwd_dhu_dtype import make_inputs, as_dtype, fla_dhu             # noqa: E402


def main():
    build()
    lib = load()
    scale = D ** -0.5
    B, T, H = 1, 512, 4
    q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
    q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                 for x in (q, k, v, g_log2, beta, h0))
    torch.manual_seed(99)
    do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
    F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
    for _ in range(6):
        run_cuda(lib, F["fi"], do, F["dv"], scale, 64, 1)
    torch.cuda.synchronize()
    print("driver done")


if __name__ == "__main__":
    main()
