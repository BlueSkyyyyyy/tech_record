"""P7-13: minimal driver so ``ncu`` can profile the KDA backward state-adjoint kernel.

The backward chain (docs/06, docs/07 §10–§15) was only ever *timed* with CUPTI; its
hardware bound was never measured (P7-2 §5 only profiled the forward K1/K2).  This
driver isolates the single most expensive backward kernel — B1, the cross-chunk
state-adjoint (FLA name ``chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64``) — in the
two implementations we compare:

  --which ours    our Triton ``_dhu_kernel_bf16`` (P7-8/P7-9 bf16-mma clone),
                  fed workspace-shaped operands at a chosen ``--bt`` (16 = FlashKDA
                  CHUNK, 64 = FLA granularity).
  --which fla     FLA's own ``chunk_gated_delta_rule_bwd_dhu`` (bf16, its native BT=64).

We warm up (Triton JIT + FLA autotune) then run ``--prof`` calls that ncu profiles,
so ``--launch-skip`` / ``-k regex:<name>`` can target one kernel unambiguously.

Usage (inside kernel_lab):
    python ncu_bwd_driver.py --which ours --B 2 --T 1024 --H 8 --bt 16
    ncu -k regex:_dhu_kernel -c 1 --launch-skip 3 --set full \\
        python ncu_bwd_driver.py --which ours --B 2 --T 1024 --H 8 --bt 16
"""

import argparse
import os

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
import sys  # noqa: E402

sys.path.insert(0, HARNESS)

import check_bwd_dhu_kernel as K  # noqa: E402
from check_bwd_dhu_dtype import build_kernel_bf16, dhu_triton_bf16  # noqa: E402


def synth_fi(B, T, H, BT, device, dtype):
    """Workspace-shaped B1 operands (values irrelevant for profiling, only shape/dtype)."""
    torch.manual_seed(0)
    qg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=device), dim=-1).to(dtype)
    kg = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=device), dim=-1).to(dtype)
    w = (torch.randn(B, T, H, D, device=device) * 0.1).to(dtype)
    g_cum = (-torch.rand(B, T, H, D, device=device) * 0.5).float()
    do = torch.randn(B, T, H, D, device=device).to(dtype)
    dv = torch.randn(B, T, H, D, device=device).to(dtype)
    fi = dict(qg=qg, kg=kg, w=w, g_cum=g_cum)
    return fi, do, dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ours", "fla"], default="ours")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16, help="chunk size for the 'ours' kernel")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    scale = D ** -0.5
    dev = "cuda"

    if args.which == "ours":
        k = build_kernel_bf16() if dt == torch.bfloat16 else K.build_kernel()
        fi, do, dv = synth_fi(args.B, args.T, args.H, args.bt, dev, dt)

        def call():
            if dt == torch.bfloat16:
                dhu_triton_bf16(k, fi, do, dv, scale, args.bt)
            else:
                K.dhu_triton(k, fi, do, dv, None, scale, args.bt)
    else:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
        q, kk, v, g_log2, beta, h0 = K.make_inputs(args.B, args.T, args.H, D, D, 64, dev, dt,
                                                   seed=args.seed)
        torch.manual_seed(123)
        do = torch.randn(args.B, args.T, args.H, D, device=dev, dtype=dt)
        F = K.fla_dhu(q, kk, v, g_log2, beta, h0, scale, 64, do)
        fi = F["fi"]

        def call():
            chunk_gated_delta_rule_bwd_dhu(
                q=fi["qg"], k=fi["kg"], w=fi["w"], gk=fi["g_cum"], h0=h0, dht=None,
                do=do, dv=F["dv"], scale=scale, chunk_size=64)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={args.B} T={args.T} H={args.H} bt={args.bt} "
          f"dtype={args.dtype} (warmup={args.warmup} prof={args.prof})")


if __name__ == "__main__":
    main()
