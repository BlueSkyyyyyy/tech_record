"""uniEP (FlashComm EPKernels / EPOverlapKernels) MoE benchmark.

Covers the two uniEP mega-kernels and their unfused components, for the
production MoE geometry:

    tokens/rank = 1024 (or 128), topk = 16, EP = 8, experts = 32, H = I = 1024 / 3072

Ops measured (bf16):
    layout                  compute_stable_local_token_within_expert_offset + compute_dispatch_layout
    dispatch                CuTeDSL pull dispatch (layout + stage + barrier + kernel)
    group_gemm_fc1          standalone M-contiguous group-GEMM, N = 2*I
    swiglu                  silu(gate)*up on the FC1 output
    group_gemm_fc2          standalone group-GEMM, N = H
    combine                 CuTeDSL push combine (push + NVL barrier + topk reduce)
    dispatch_group_gemm     MegaKernel 1 (fused dispatch + FC1)
    group_gemm_combine      MegaKernel 2 (fused FC2 + combine + reduce)
    e2e_fused               MK1 + swiglu + MK2  (production overlap path)
    e2e_sequential          dispatch + FC1 + swiglu + FC2 + combine

Launch:
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=8 bench_uniep_ep.py \
        --tokens 1024 128 --hidden 1024 3072 --csv moe_ep_uniep.csv
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import (  # noqa: E402
    BF16, device_name, fmt_shape, gather_rows, local_m_splits, make_routing, measure, measure_robust, probe_quiet,
    row_from_stats,
    write_csv,
)

from flash_comm.ep.ep_kernels import EPCommLayoutDesc  # noqa: E402
from flash_comm.ep_overlap import EPOverlapKernels  # noqa: E402
from flash_comm.utils.test_utils import init_dist  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1024, 128], help="tokens per rank")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 3072], help="H = F")
    p.add_argument("--intermediate", type=int, nargs="+", default=None, help="I; default = H (=F)")
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--num_experts", type=int, default=32)
    p.add_argument("--local_world_size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--static_capacity", action="store_true",
                   help="size recv buffers to M*topk up front (num_worst_tokens) instead of polling "
                        "the receive count on the CPU")
    p.add_argument("--tile_m", type=int, default=128)
    p.add_argument("--tile_n", type=int, default=256)
    p.add_argument("--csv", type=str, default="moe_ep_uniep.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.intermediate is None:
        args.intermediate = list(args.hidden)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ["NCCL_LSA_TEAM_SIZE"] = str(args.local_world_size)

    ep_group = init_dist()
    rank = dist.get_rank()
    global TILE_M, TILE_N
    TILE_M, TILE_N = args.tile_m, args.tile_n
    world = dist.get_world_size()
    dev = device_name()
    epr = args.num_experts // world
    alignment = EPOverlapKernels.cluster_tile_m
    rows = []

    try:
        for H, I in zip(args.hidden, args.intermediate):
            for M in args.tokens:
                if rank == 0:
                    print(f"\n=== uniEP: tokens/rank={M} H={H} I={I} topk={args.topk} "
                          f"E={args.num_experts} EP={world} ===", flush=True)

                worst = int(M * args.topk * 1.2) + epr * alignment
                kernels = EPOverlapKernels(
                    max_m=M, hidden=H, topk=args.topk, num_experts=args.num_experts,
                    local_world_size=args.local_world_size, ep_group=ep_group,
                    num_worst_tokens=(worst if args.static_capacity else -1),
                    expert_alignment=alignment,
                )

                torch.manual_seed(1234 + rank)
                topk_idx, topk_w = make_routing(M, args.num_experts, args.topk, seed=1234 + rank)
                x = torch.randn(M, H, dtype=BF16, device="cuda")
                # (E_local, N, K) row-major -> k-major view (N, K, E_local) required by the kernels
                B_fc1 = torch.randn(epr, 2 * I, H, dtype=BF16, device="cuda").permute(1, 2, 0)
                B_fc2 = torch.randn(epr, H, I, dtype=BF16, device="cuda").permute(1, 2, 0)

                def tag(op):
                    return op

                shape = fmt_shape([f"T{M}/rank", f"H{H}", f"I{I}", f"topk{args.topk}",
                                   f"E{args.num_experts}", f"EP{world}"])
                probe_quiet(ep_group)

                def add(op, stats, note=""):
                    rows.append(row_from_stats(
                        stats, framework="uniEP", op=op, dtype="torch.bfloat16", shape=shape,
                        device=dev, tokens_per_rank=M, hidden=H, intermediate=I,
                        topk=args.topk, num_experts=args.num_experts, ep=world, note=note))
                    if rank == 0:
                        print(f"  {op:28s} kernel={stats['kernel_us']:9.2f} us  "
                              f"wall={stats['wall_med_us']:9.2f}/{stats['wall_p90_us']:9.2f} us  {note}",
                              flush=True)

                # ---------------- unfused components ----------------
                layout = EPCommLayoutDesc()
                disp, layout = kernels.dispatch_cutedsl(x, topk_idx, layout)
                m_splits = layout.recv_expert_counts
                A_fc1 = kernels.group_gemm(disp, B_fc1, m_splits)
                gate, up = A_fc1.split(I, dim=-1)
                A2 = (F.silu(gate) * up).contiguous()
                C = kernels.group_gemm(A2, B_fc2, m_splits)
                dist.barrier()
                kernels.ep_group_barrier()

                # split list only for reporting
                splits = local_m_splits(topk_idx, args.num_experts, world, ep_group)
                aligned = int(layout.recv_aligned_token_count_cpu.max().item()) \
                    if layout.recv_aligned_token_count_cpu is not None else -1
                recv_rows = int(disp.shape[0])
                note_layout = (f"recv_rows={recv_rows} (unaligned sum={sum(splits)}, align={alignment}) "
                               f"local_m={splits}")

                def run_layout():
                    tw, ec = kernels.compute_token_within_expert_offset_and_expert_counts(topk_idx)
                    desc = EPCommLayoutDesc(token_within_expert_offset=tw, expert_counts=ec)
                    kernels._ensure_dispatch_layout(desc, topk_idx)
                    return desc

                st = measure_robust(run_layout, args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_layout", st, "layout only (histogram + cumsum + allgather + layout kernel)")

                layout = EPCommLayoutDesc()
                st = measure_robust(lambda: kernels.dispatch_cutedsl(x, topk_idx, layout),
                             args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_dispatch", st, note_layout)

                st = measure_robust(lambda: kernels.group_gemm(disp, B_fc1, m_splits),
                             args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_group_gemm_fc1", st, f"N=2I={2 * I}, K={H}, rows={disp.shape[0]}")

                st = measure_robust(lambda: (F.silu(A_fc1.split(I, dim=-1)[0]) * A_fc1.split(I, dim=-1)[1]).contiguous(),
                             args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_swiglu", st, f"rows={A_fc1.shape[0]}, I={I}")

                st = measure_robust(lambda: kernels.group_gemm(A2, B_fc2, m_splits),
                             args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_group_gemm_fc2", st, f"N=H={H}, K=I={I}, rows={A2.shape[0]}")

                st = measure_robust(lambda: kernels.combine_cutedsl(C, layout_desc=layout, push_mode="tile", tile_m=TILE_M, tile_n=TILE_N),
                             args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_combine", st, "push_mode=tile (push + NVL barrier + topk reduce)")

                # ---------------- MegaKernel 1 ----------------
                holder = {}
                st = measure_robust(lambda: holder.update(dict(zip(("A", "w", "layout"),
                                                            kernels.dispatch_group_gemm(
                                                                x, topk_idx, B_fc1, None)))),
                             args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_dispatch_group_gemm", st, "MegaKernel 1: dispatch + FC1 overlap")

                # ---------------- MegaKernel 2 ----------------
                mk1_layout = holder["layout"]
                A2_mk1 = holder["A"]
                g2, u2 = A2_mk1.split(I, dim=-1)
                A2_mk1 = (F.silu(g2) * u2).contiguous()
                st = measure_robust(lambda: kernels.group_gemm_combine(A2_mk1, B_fc2, mk1_layout),
                             args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_group_gemm_combine", st, "MegaKernel 2: FC2 + combine + reduce")

                # ---------------- end-to-end ----------------
                def e2e_fused():
                    a, _w, desc = kernels.dispatch_group_gemm(x, topk_idx, B_fc1, None)
                    g, u = a.split(I, dim=-1)
                    a2 = (F.silu(g) * u).contiguous()
                    return kernels.group_gemm_combine(a2, B_fc2, desc)

                def e2e_seq():
                    d, desc = kernels.dispatch_cutedsl(x, topk_idx, layout)
                    a = kernels.group_gemm(d, B_fc1, desc.recv_expert_counts)
                    g, u = a.split(I, dim=-1)
                    a2 = (F.silu(g) * u).contiguous()
                    c = kernels.group_gemm(a2, B_fc2, desc.recv_expert_counts)
                    return kernels.combine_cutedsl(c, layout_desc=desc, push_mode="tile", tile_m=TILE_M, tile_n=TILE_N)

                st = measure_robust(e2e_fused, args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_e2e_fused", st, "MK1 + swiglu + MK2 (2 launches)")
                st = measure_robust(e2e_seq, args.warmup, args.iters, barrier=ep_group.barrier)
                add("uniep_e2e_sequential", st, "dispatch+FC1+swiglu+FC2+combine (5 launches)")

                del kernels, disp, A_fc1, A2, C, holder, B_fc1, B_fc2, x, topk_idx, topk_w
                torch.cuda.empty_cache()
                dist.barrier()
    finally:
        merged = gather_rows(rows, ep_group) if rows else []
        if rank == 0 and merged:
            write_csv(args.csv, merged)
            print(f"\nwrote {len(merged)} rows -> {args.csv}")
        try:
            kernels.finalize()
        except Exception:
            pass
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
