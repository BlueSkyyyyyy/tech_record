"""P7-16: minimal driver so ``ncu`` can profile the *hand-written CUDA* B1
(`harness/b1_cuda_probe.cu`, P7-15 / docs/07 §18).

§18 concluded, qualitatively, that the bare-wmma CUDA B1 is **not** load-bound (its
`NSTAGE` prefetch is worth only ~2-6%), unlike the Triton prototype of §17 where the
loop pipeline closed the whole gap.  §18.4 left the *quantitative* limiter open
("瓶颈在占用率/发射/同步，不在等 global" is a hypothesis).

This driver lets ``ncu`` pin that down.  It isolates a single kernel launch of either:

  --which ours   our hand-written CUDA `b1_dhu_cuda` (this repo), fed synthetic
                 workspace-shaped operands at a chosen ``--bt`` and depth ``--ns``.
  --which fla    FLA's own `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` (bf16,
                 BT=64), the reference the CUDA prototype loses 4.3-9.6x to.

The ``ours`` path synthesises operands directly (never calls FLA/Triton/TileLang): that
setup launches dozens of kernels and, under ``ncu``, dominated wall time (repeated
``nvidia-smi`` subprocesses from the backend dispatcher).

Usage (inside kernel_lab):
    python ncu_b1_cuda_driver.py --which ours --B 2 --T 1024 --H 8 --bt 64 --ns 1
    ncu -k regex:b1_dhu_cuda -c 1 --launch-skip 3 --set full \\
        python ncu_b1_cuda_driver.py --which ours --B 2 --T 1024 --H 8 --bt 64 --ns 1

Environment:
    B1_CFLAGS   extra nvcc flags forwarded to the probe build (e.g. "-DB1_SYNC").
"""

import argparse
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch                                                              # noqa: E402

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_b1_cuda import build, load, run_cuda                        # noqa: E402


def synth(B, T, H, device="cuda"):
    """Workspace-shaped B1 operands (values irrelevant for profiling)."""
    torch.manual_seed(0)
    qg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    kg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    w = (torch.randn(B, T, H, D, device=device) * 0.1).bfloat16()
    g_cum = (-torch.rand(B, T, H, D, device=device) * 0.5).float()
    do = torch.randn(B, T, H, D, device=device).bfloat16()
    dv = torch.randn(B, T, H, D, device=device).bfloat16()
    return dict(qg=qg, kg=kg, w=w, g_cum=g_cum), do, dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ours", "fla"], default="ours")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=64)
    ap.add_argument("--ns", type=int, default=1, help="cp.async depth NSTAGE ('ours')")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-build", action="store_true",
                    help="reuse an existing /tmp/b1_cuda.so (avoids running nvcc under ncu)")
    args = ap.parse_args()

    scale = D ** -0.5
    B, T, H = args.B, args.T, args.H

    if args.which == "ours":
        fi, do, dv = synth(B, T, H)
        if not args.no_build:
            build()
        lib = load()

        def call():
            run_cuda(lib, fi, do, dv, scale, args.bt, args.ns)
    else:
        from check_bwd_dhu_dtype import make_inputs, as_dtype, fla_dhu
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        qq, kk, ww, gk, dv = F["fi"]["qg"], F["fi"]["kg"], F["fi"]["w"], F["fi"]["g_cum"], F["dv"]

        def call():
            chunk_gated_delta_rule_bwd_dhu(
                q=qq, k=kk, w=ww, gk=gk, h0=h0, dht=None, do=do, dv=dv,
                scale=scale, chunk_size=64)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={B} T={T} H={H} bt={args.bt} ns={args.ns} "
          f"(warmup={args.warmup} prof={args.prof})")


if __name__ == "__main__":
    main()
