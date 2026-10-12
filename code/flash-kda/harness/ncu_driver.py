"""P7-2: minimal driver so ``ncu`` can attach to a known number of FlashKDA
forward kernel launches.

Each ``flash_kda.fwd`` call issues three kernels in order:
  1. the C++ workspace init (elementwise),
  2. ``flash_kda_fwd_prepare``   (K1),
  3. ``flash_kda_fwd_recurrence``(K2).

We do ``WARMUP`` calls first, then ``PROF`` calls that ncu actually profiles,
so that ``ncu --launch-skip`` / ``--launch-count`` can target individual kernels
by name (``-k regex:flash_kda_fwd_recurrence``).

Usage (inside kernel_lab):
    python ncu_driver.py --B 1 --T 1024 --H 4
    ncu -k regex:flash_kda_fwd_recurrence -c 1 --set full \\
        python ncu_driver.py --B 1 --T 1024 --H 4
"""

import argparse
import math

import torch

D = 128
SCALE = 1.0 / math.sqrt(D)
LOWER_BOUND = -5.0


def make_inputs(B, T, H, device="cuda", seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16) * 2.0
    beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=device)
    dt_bias = torch.randn(H, D, device=device)
    h0 = torch.randn(B, H, D, D, device=device)
    return q, k, v, g, beta, A_log, dt_bias, h0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import flash_kda

    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(args.B, args.T, args.H, seed=args.seed)
    out = torch.empty(args.B, args.T, args.H, D, device="cuda", dtype=torch.bfloat16)
    fin = torch.empty(args.B, args.H, D, D, device="cuda", dtype=torch.bfloat16)
    h0_fk = h0.bfloat16().transpose(-1, -2).contiguous()

    def call():
        flash_kda.fwd(q, k, v, g, beta, SCALE, out,
                      A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                      initial_state=h0_fk, final_state=fin)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE B={args.B} T={args.T} H={args.H} (warmup={args.warmup} prof={args.prof})")


if __name__ == "__main__":
    main()
