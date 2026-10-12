"""Diagnostic: per-kernel split of the assembled backward chain's device time.

Not a deliverable -- a one-off probe to decide the next increment (which kernel of
`dAv -> B1 -> B2 -> B3 -> reverse-cumsum` dominates at deployment granularity BT=16,
now that P7-22 pipelined B1).  Reuses the exact chain from `check_bwd_b1_pipe_e2e`.

Run: CUDA_VISIBLE_DEVICES=1 python .../probe_bwd_chain_split.py
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import D, CHUNK  # noqa: E402
from check_bwd_b1_pipe_e2e import make_chain_fn_pipe, TCASES  # noqa: E402
from check_bwd_b1_bf16_e2e import prep, profile_dev, time_wall  # noqa: E402


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name()} torch={torch.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    for (name, B, T, H, seed, gs) in TCASES:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        ch = make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=3)
        ch()
        torch.cuda.synchronize()
        d, prof = profile_dev(ch, reps=15)
        split = {}
        for ev in prof.key_averages():
            key = ev.key
            split[key] = split.get(key, 0.0) + ev.self_device_time_total / 15
        print(f"\n# {name}  chain dev={d:.1f} us")
        for k, v in sorted(split.items(), key=lambda x: -x[1]):
            print(f"    {k[:70]:70s} {v:8.2f} us")
    print("\nDONE")


if __name__ == "__main__":
    main()
