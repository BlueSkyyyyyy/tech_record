"""uniEP MoE backward benchmark (bf16).

uniEP ships only forward kernels (`dispatch_cutedsl` / `group_gemm` / `combine_cutedsl`
plus the two mega-kernels); there is **no** `autograd.Function` anywhere in the package,
so a layer backward has to be *composed* from the same group-GEMM primitive:

    FC1 fwd    Y    = A   @ W1^T      group_gemm, N=2I, K=H
    FC1 dgrad  dA   = dY1 @ W1        group_gemm, N=H,  K=2I   (same per-expert M split)
    FC1 wgrad  dW1e = dY1e^T @ Ae     group_gemm, N=2I, M=H, K=rows_e
    FC1 reduce dW1  = all_reduce_EP(dW1)   rows come from all ranks -> every rank holds a partial

    FC2 fwd    OUT  = A2   @ W2^T     group_gemm, N=H, K=I
    FC2 dgrad  dA2  = dOUT @ W2       group_gemm, N=I, K=H
    FC2 wgrad  dW2e = dOUTe^T @ A2e   group_gemm, N=H, M=I, K=rows_e   (+ all_reduce)

    SwiGLU bwd dg = dA2*u*silu'(g),  du = dA2*silu(g)

Details that shape the numbers:

  * `group_gemm` wants `B` k-major `(N, K, E)` with a **single K for all experts**. dgrad
    therefore uses the *transposed* weight (pre-materialised once: weights are static in
    training), and wgrad pads K to `max(rows_e)` (zero-filled), i.e. the usual "gather-K"
    trick. Both transposed operands are built outside the timed region and their cost is
    reported separately as `uniep_transpose_prep`.
  * the weight-grad all-reduce is part of a real training step, so it is inside
    `*_wgrad` / `*_fwd_bwd`, and also measured on its own (`uniep_wgrad_allreduce`).
  * uniEP has no fused SwiGLU backward either: `uniep_swiglu_bwd` is 3 torch kernels.

Correctness: the composed chain is compared against torch autograd on the same per-expert
row splits (`check_chain`), including dA / dW1 / dW2.

Launch:
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=8 bench_uniep_bwd.py \
        --tokens 1024 128 --hidden 1024 3072 --csv moe_ep_uniep_bwd.csv
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe_bench_common import (  # noqa: E402
    BF16, device_name, fmt_shape, gather_rows, make_routing, measure_robust, probe_quiet,
    row_from_stats, write_csv,
)

from flash_comm.ep.ep_kernels import EPCommLayoutDesc  # noqa: E402
from flash_comm.ep_overlap import EPOverlapKernels  # noqa: E402
from flash_comm.utils.test_utils import init_dist  # noqa: E402


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1024, 128], help="tokens per rank")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 3072], help="H = F")
    p.add_argument("--intermediate", type=int, nargs="+", default=None, help="I; default = H")
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--num_experts", type=int, default=32)
    p.add_argument("--local_world_size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--no-check", dest="check", action="store_false", default=True,
                   help="skip the torch-reference check")
    p.add_argument("--csv", type=str, default="moe_ep_uniep_bwd.csv")
    return p.parse_args()


def k_major(x):
    """(E, N, K) row-major -> (N, K, E) k-major view required by group_gemm."""
    return x.permute(1, 2, 0)


def kmaj_of_transposed(x):
    """(E, N, K) -> (K, N, E) k-major (i.e. the transposed weight, k-major)."""
    return x.transpose(1, 2).contiguous().permute(1, 2, 0)


def rel_err(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def check_chain(kernels, A, dOUT, W1, W2, B_fc1_d, B_fc2_d, B1w_km, w1_counts,
                splits, m_al, I, tag):
    """Full layer backward: torch autograd vs the same chain composed from uniEP kernels.

    Reference: OUT = (silu(A @ W1_e^T) * ...) @ W2_e^T, loss = <OUT, dOUT>, autograd grads.
    Composed : dA2 = group_gemm(dOUT, W2^T)  ->  SwiGLU bwd (torch)  ->  dA = group_gemm(dy1, W1^T)
               dW2 = group_gemm(dOUT^T, A2^T),  dW1 = group_gemm(dy1^T, A^T)   (local, no all-reduce)
    """
    epr = W1.shape[0]
    ref_A = A.detach().clone().requires_grad_(True)
    ref_W1 = W1.detach().clone().requires_grad_(True)
    ref_W2 = W2.detach().clone().requires_grad_(True)
    off = 0
    outs = []
    for e, m in enumerate(splits):
        g, u = (ref_A[off:off + m] @ ref_W1[e].t()).split(I, dim=-1)
        outs.append((F.silu(g) * u) @ ref_W2[e].t())
        off += m
    (torch.cat(outs, dim=0).float() * dOUT.float()).sum().backward()

    with torch.no_grad():
        # recompute the forward activations with the kernels
        y1 = kernels.group_gemm(A, B1w_km.new_empty(0) if False else k_major_from(W1), m_al) \
            if False else None
        y1 = kernels.group_gemm(A, W1.permute(1, 2, 0), m_al)
        gg, uu = y1.split(I, dim=-1)
        a2 = (F.silu(gg) * uu).contiguous()

        dA2 = kernels.group_gemm(dOUT, B_fc2_d, m_al)

        # FC2 wgrad: dOUT^T @ A2  (operands materialised here, timing is measured elsewhere)
        A2w = torch.zeros(epr * A.shape[1], max(splits), dtype=A.dtype, device=A.device)
        B2w = torch.zeros(epr, I, max(splits), dtype=A.dtype, device=A.device)
        off = 0
        for e, m in enumerate(splits):
            A2w[e * A.shape[1]:(e + 1) * A.shape[1], :m].copy_(dOUT[off:off + m].t())
            B2w[e, :, :m].copy_(a2[off:off + m].t())
            off += m
        w2_counts = torch.full((epr,), A.shape[1], dtype=torch.int32, device=A.device)
        dW2 = kernels.group_gemm(A2w, k_major(B2w), w2_counts)

        # SwiGLU bwd (torch) -> dy1, then FC1 dgrad / wgrad
        sig = torch.sigmoid(gg)
        dy1 = torch.cat([dA2 * uu * sig * (1 + gg * (1 - sig)), dA2 * F.silu(gg)], dim=-1)
        dA = kernels.group_gemm(dy1, B_fc1_d, m_al)
        A1w = torch.zeros(epr * 2 * I, max(splits), dtype=A.dtype, device=A.device)
        off = 0
        for e, m in enumerate(splits):
            A1w[e * 2 * I:(e + 1) * 2 * I, :m].copy_(dy1[off:off + m].t())
            off += m
        dW1 = kernels.group_gemm(A1w, B1w_km, w1_counts)

    r = (rel_err(dW1, ref_W1.grad.view(epr, -1)),
         rel_err(dW2, ref_W2.grad.view(epr, -1)),
         rel_err(dA, ref_A.grad))
    if dist.get_rank() == 0:
        print(f"  [check {tag}] full-chain rel err: dW1={r[0]:.2e} dW2={r[1]:.2e} dA={r[2]:.2e} "
              f"| |dW1|={dW1.float().abs().max():.3g} |dW2|={dW2.float().abs().max():.3g} "
              f"|ref dW2|={ref_W2.grad.float().abs().max():.3g}", flush=True)
    assert max(r) < 5e-2, f"backward mismatch: {r}"
    del ref_A, ref_W1, ref_W2, y1, a2, dy1, A1w, A2w, B2w
    torch.cuda.empty_cache()


def k_major_from(x):
    return x.permute(1, 2, 0)


def main():
    args = parse_args()
    if args.intermediate is None:
        args.intermediate = list(args.hidden)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ["NCCL_LSA_TEAM_SIZE"] = str(args.local_world_size)

    ep_group = init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = device_name()
    epr = args.num_experts // world
    alignment = EPOverlapKernels.cluster_tile_m
    csv_rows = []

    def sync():
        torch.cuda.synchronize()
        ep_group.barrier()

    try:
        for H, I in zip(args.hidden, args.intermediate):
            for M in args.tokens:
                if rank == 0:
                    print(f"\n=== uniEP bwd: tokens/rank={M} H={H} I={I} topk={args.topk} "
                          f"E={args.num_experts} EP={world} ===", flush=True)

                kernels = EPOverlapKernels(
                    max_m=M, hidden=H, topk=args.topk, num_experts=args.num_experts,
                    local_world_size=args.local_world_size, ep_group=ep_group,
                    num_worst_tokens=-1, expert_alignment=alignment,
                )
                torch.manual_seed(1234 + rank)
                topk_idx, _ = make_routing(M, args.num_experts, args.topk, seed=1234 + rank)
                x = torch.randn(M, H, dtype=BF16, device="cuda")
                W1 = torch.randn(epr, 2 * I, H, dtype=BF16, device="cuda")
                W2 = torch.randn(epr, H, I, dtype=BF16, device="cuda")
                B_fc1, B_fc1_d = k_major(W1), kmaj_of_transposed(W1)
                B_fc2, B_fc2_d = k_major(W2), kmaj_of_transposed(W2)

                # dispatched activations (setup, not timed)
                disp, layout = kernels.dispatch_cutedsl(x, topk_idx, EPCommLayoutDesc())
                m_splits = layout.recv_expert_counts
                # recv_expert_counts is the *unaligned* per-expert count; the dispatched rows are
                # padded to `alignment` per expert and the padding is zero-filled (verified), so
                # both the reference and the wgrad gather-K can simply work on aligned blocks.
                splits = [((int(v) + alignment - 1) // alignment) * alignment
                          for v in m_splits.tolist()]
                m_al = torch.as_tensor(splits, dtype=m_splits.dtype, device=m_splits.device)
                assert sum(splits) == disp.shape[0], (sum(splits), disp.shape)
                A = disp
                n_rows = A.shape[0]
                kpad = max(splits)  # already a multiple of `alignment`
                offs = [sum(splits[:i]) for i in range(len(splits))]

                dY1 = torch.randn(n_rows, 2 * I, dtype=BF16, device="cuda")
                dOUT = torch.randn(n_rows, H, dtype=BF16, device="cuda")

                # wgrad operands: A' = dY^T (M=2I / H), B' = A^T (N=H / I), K padded to kpad
                A1w = torch.zeros(epr * 2 * I, kpad, dtype=BF16, device="cuda")
                B1w = torch.zeros(epr, H, kpad, dtype=BF16, device="cuda")
                A2w = torch.zeros(epr * H, kpad, dtype=BF16, device="cuda")
                B2w = torch.zeros(epr, I, kpad, dtype=BF16, device="cuda")

                Y1 = kernels.group_gemm(A, B_fc1, m_splits)
                g, u = Y1.split(I, dim=-1)
                A2 = (F.silu(g) * u).contiguous()

                def build_transposed():
                    """FC1 wgrad: dY1^T @ A ; FC2 wgrad: dOUT^T @ A2 (K padded to kpad)."""
                    for e, m in enumerate(splits):
                        o = offs[e]
                        A1w[e * 2 * I:(e + 1) * 2 * I, :m].copy_(dY1[o:o + m].t())
                        B1w[e, :, :m].copy_(A[o:o + m].t())
                        A2w[e * H:(e + 1) * H, :m].copy_(dOUT[o:o + m].t())
                        B2w[e, :, :m].copy_(A2[o:o + m].t())

                with torch.no_grad():
                    build_transposed()
                B1w_km, B2w_km = k_major(B1w), k_major(B2w)
                w1_counts = torch.full((epr,), 2 * I, dtype=torch.int32, device="cuda")
                w2_counts = torch.full((epr,), H, dtype=torch.int32, device="cuda")
                OUT = kernels.group_gemm(A2, B_fc2, m_splits)
                dA1 = torch.empty(n_rows, H, dtype=BF16, device="cuda")
                dA2_buf = torch.empty(n_rows, I, dtype=BF16, device="cuda")
                dW1_buf = torch.empty(epr * 2 * I, H, dtype=BF16, device="cuda")
                dW2_buf = torch.empty(epr * H, I, dtype=BF16, device="cuda")
                sync()
                probe_quiet(ep_group)

                shape = fmt_shape([f"T{M}/rank", f"H{H}", f"I{I}", f"topk{args.topk}",
                                   f"E{args.num_experts}", f"EP{world}"])

                def add(op, stats, note=""):
                    csv_rows.append(row_from_stats(
                        stats, framework="uniEP", op=op, dtype="torch.bfloat16", shape=shape,
                        device=dev, tokens_per_rank=M, hidden=H, intermediate=I,
                        topk=args.topk, num_experts=args.num_experts, ep=world, note=note))
                    if rank == 0:
                        print(f"  {op:28s} kernel={stats['kernel_us']:9.2f} us  "
                              f"wall={stats['wall_med_us']:9.2f}/{stats['wall_p90_us']:9.2f} us  {note}",
                              flush=True)

                # ---------------- backward pieces ----------------
                def fc1_dgrad():
                    return kernels.group_gemm(dY1, B_fc1_d, m_splits, output=dA1)

                def fc1_wgrad_local():
                    return kernels.group_gemm(A1w, B1w_km, w1_counts, output=dW1_buf)

                def fc1_wgrad():
                    w = fc1_wgrad_local()
                    dist.all_reduce(w.view(epr, -1), group=ep_group)
                    return w

                def fc2_dgrad():
                    return kernels.group_gemm(dOUT, B_fc2_d, m_splits, output=dA2_buf)

                def fc2_wgrad_local():
                    return kernels.group_gemm(A2w, B2w_km, w2_counts, output=dW2_buf)

                def fc2_wgrad():
                    w = fc2_wgrad_local()
                    dist.all_reduce(w.view(epr, -1), group=ep_group)
                    return w

                def wgrad_allreduce():
                    dist.all_reduce(dW1_buf.view(epr, -1), group=ep_group)
                    dist.all_reduce(dW2_buf.view(epr, -1), group=ep_group)

                def swiglu_bwd():
                    # naive torch backward: sigmoid, silu, 2 muls per branch, then a cat
                    sig = torch.sigmoid(g)
                    dg = dA2_buf * u * sig * (1 + g * (1 - sig))
                    du = dA2_buf * F.silu(g)
                    return torch.cat([dg, du], dim=-1)

                def transpose_prep():
                    build_transposed()
                    return A1w

                def fc1_fwd_bwd():
                    y = kernels.group_gemm(A, B_fc1, m_splits, output=Y1)
                    da = kernels.group_gemm(dY1, B_fc1_d, m_splits, output=dA1)
                    w = kernels.group_gemm(A1w, B1w_km, w1_counts, output=dW1_buf)
                    dist.all_reduce(w.view(epr, -1), group=ep_group)
                    return da, w

                def fc2_fwd_bwd():
                    o = kernels.group_gemm(A2, B_fc2, m_splits, output=OUT)
                    da = kernels.group_gemm(dOUT, B_fc2_d, m_splits, output=dA2_buf)
                    w = kernels.group_gemm(A2w, B2w_km, w2_counts, output=dW2_buf)
                    dist.all_reduce(w.view(epr, -1), group=ep_group)
                    return o, da, w

                def layer_fwd_bwd():
                    y1 = kernels.group_gemm(A, B_fc1, m_splits, output=Y1)
                    gg, uu = y1.split(I, dim=-1)
                    a2 = (F.silu(gg) * uu).contiguous()
                    out = kernels.group_gemm(a2, B_fc2, m_splits, output=OUT)
                    da2 = kernels.group_gemm(dOUT, B_fc2_d, m_splits, output=dA2_buf)
                    w2 = kernels.group_gemm(A2w, B2w_km, w2_counts, output=dW2_buf)
                    dist.all_reduce(w2.view(epr, -1), group=ep_group)
                    sig = torch.sigmoid(gg)
                    dg = da2 * uu * sig * (1 + gg * (1 - sig))
                    du = da2 * F.silu(gg)
                    dy1 = torch.cat([dg, du], dim=-1)
                    da = kernels.group_gemm(dy1, B_fc1_d, m_splits, output=dA1)
                    w1 = kernels.group_gemm(A1w, B1w_km, w1_counts, output=dW1_buf)
                    dist.all_reduce(w1.view(epr, -1), group=ep_group)
                    return out, da, w1, w2

                def build_wgrad_chain(a2, dy1):
                    """per-step operand layout conversion (a2^T for FC2 wgrad, dy1^T for FC1 wgrad)"""
                    for e, m in enumerate(splits):
                        o = offs[e]
                        A1w[e * 2 * I:(e + 1) * 2 * I, :m].copy_(dy1[o:o + m].t())
                        A2w[e * H:(e + 1) * H, :m].copy_(dOUT[o:o + m].t())
                        B2w[e, :, :m].copy_(a2[o:o + m].t())

                def layer_fwd_bwd_full():
                    y1 = kernels.group_gemm(A, B_fc1, m_splits, output=Y1)
                    gg, uu = y1.split(I, dim=-1)
                    a2 = (F.silu(gg) * uu).contiguous()
                    out = kernels.group_gemm(a2, B_fc2, m_splits, output=OUT)
                    da2 = kernels.group_gemm(dOUT, B_fc2_d, m_splits, output=dA2_buf)
                    sig = torch.sigmoid(gg)
                    dg = da2 * uu * sig * (1 + gg * (1 - sig))
                    du = da2 * F.silu(gg)
                    dy1 = torch.cat([dg, du], dim=-1)
                    build_wgrad_chain(a2, dy1)
                    w2 = kernels.group_gemm(A2w, B2w_km, w2_counts, output=dW2_buf)
                    dist.all_reduce(w2.view(epr, -1), group=ep_group)
                    da = kernels.group_gemm(dy1, B_fc1_d, m_splits, output=dA1)
                    w1 = kernels.group_gemm(A1w, B1w_km, w1_counts, output=dW1_buf)
                    dist.all_reduce(w1.view(epr, -1), group=ep_group)
                    return out, da, w1, w2

                add("uniep_fc1_dgrad", measure_robust(fc1_dgrad, args.warmup, args.iters, barrier=sync),
                    f"dY1@W1: N=H={H}, K=2I={2 * I}, rows={n_rows}")
                add("uniep_fc1_wgrad", measure_robust(fc1_wgrad, args.warmup, args.iters, barrier=sync),
                    f"dY1^T@A (K padded {kpad}) + all_reduce {epr * 2 * I * H * 2 / 1e6:.0f}MB")
                add("uniep_fc2_dgrad", measure_robust(fc2_dgrad, args.warmup, args.iters, barrier=sync),
                    f"dOUT@W2: N=I={I}, K=H={H}, rows={n_rows}")
                add("uniep_fc2_wgrad", measure_robust(fc2_wgrad, args.warmup, args.iters, barrier=sync),
                    f"dOUT^T@A2 (K padded {kpad}) + all_reduce {epr * H * I * 2 / 1e6:.0f}MB")
                def fc1_allreduce():
                    return dist.all_reduce(dW1_buf.view(epr, -1), group=ep_group)

                def fc2_allreduce():
                    return dist.all_reduce(dW2_buf.view(epr, -1), group=ep_group)

                add("uniep_fc1_wgrad_gemm", measure_robust(fc1_wgrad_local, args.warmup, args.iters,
                                                           barrier=sync),
                    f"仅 dY1^T@A: M=2I={2 * I}, N=H={H}, K={kpad}, E={epr}")
                add("uniep_fc2_wgrad_gemm", measure_robust(fc2_wgrad_local, args.warmup, args.iters,
                                                           barrier=sync),
                    f"仅 dOUT^T@A2: M=H={H}, N=I={I}, K={kpad}, E={epr}")
                add("uniep_fc1_allreduce", measure_robust(fc1_allreduce, args.warmup, args.iters,
                                                          barrier=sync),
                    f"dW1 EP all_reduce {epr * 2 * I * H * 2 / 1e6:.0f}MB")
                add("uniep_fc2_allreduce", measure_robust(fc2_allreduce, args.warmup, args.iters,
                                                          barrier=sync),
                    f"dW2 EP all_reduce {epr * H * I * 2 / 1e6:.0f}MB")
                add("uniep_wgrad_allreduce", measure_robust(wgrad_allreduce, args.warmup, args.iters,
                                                           barrier=sync),
                    f"2 次 EP all_reduce: {epr * (2 * I * H + H * I) * 2 / 1e6:.0f}MB")
                add("uniep_swiglu_bwd", measure_robust(swiglu_bwd, args.warmup, args.iters, barrier=sync),
                    "silu(g)*u 反向: 3 个 torch kernel (dg, du, cat)")
                add("uniep_transpose_prep", measure_robust(transpose_prep, args.warmup, args.iters,
                                                          barrier=sync),
                    f"4 个转置拷贝 (dY1^T, A^T ×2, dOUT^T) -> K padded {kpad}")
                add("uniep_fc1_fwd_bwd", measure_robust(fc1_fwd_bwd, args.warmup, args.iters, barrier=sync),
                    "FC1 fwd + dgrad + wgrad(+all_reduce)")
                add("uniep_fc2_fwd_bwd", measure_robust(fc2_fwd_bwd, args.warmup, args.iters, barrier=sync),
                    "FC2 fwd + dgrad + wgrad(+all_reduce)")
                add("uniep_layer_fwd_bwd", measure_robust(layer_fwd_bwd, args.warmup, args.iters,
                                                         barrier=sync),
                    "FC1 fwd+SwiGLU+FC2 fwd + FC2 bwd+SwiGLU bwd+FC1 bwd (2×EP all_reduce)")

                add("uniep_layer_fwd_bwd_full", measure_robust(layer_fwd_bwd_full, args.warmup,
                                                              args.iters, barrier=sync),
                    "同上 + 每步 dy1^T / a2^T 转置准备（真实训练必须付这部分）")

                if args.check:
                    check_chain(kernels, A, dOUT, W1, W2, B_fc1_d, B_fc2_d, B1w_km,
                                w1_counts, splits, m_al, I, f"T{M}/H{H}")

                del disp, layout, dY1, dOUT, A1w, B1w, A2w, B2w, B1w_km, B2w_km
                del Y1, A2, OUT, dA1, dA2_buf, dW1_buf, dW2_buf
                torch.cuda.empty_cache()
                kernels.ep_group_barrier()
    finally:
        merged = gather_rows(csv_rows, ep_group)
        if rank == 0 and merged:
            write_csv(args.csv, merged)
            print(f"\nwrote {args.csv} ({len(merged)} rows)", flush=True)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
