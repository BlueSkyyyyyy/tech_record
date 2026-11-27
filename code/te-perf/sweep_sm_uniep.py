"""uniEP mega-kernel SM-budget sweep (dispatch_group_gemm / group_gemm_combine).

`gemm_num_sm` / `comm_num_sm` decide how the persistent kernels split the 132 SMs
between comm- and comp-workers. `None` means "use the whole device", which is what
the upstream test harness defaults to; this sweep checks whether a split turns the
fused kernels from slower-than-unfused into faster.

Run:
    NCCL_NVLS_ENABLE=0 FLASH_COMM_BUFFER_BACKEND=cuda_ipc NCCL_LSA_TEAM_SIZE=8 \
    torchrun --nproc_per_node=8 sweep_sm_uniep.py --tokens 1024 --hidden 1024
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import BF16, local_m_splits, make_routing, measure, row_from_stats, write_csv  # noqa: E402

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLASH_COMM_BUFFER_BACKEND", "cuda_ipc")

from flash_comm.ep.ep_kernels import EPCommLayoutDesc  # noqa: E402
from flash_comm.ep_overlap import EPOverlapKernels  # noqa: E402
from flash_comm.utils.test_utils import init_dist  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--intermediate", type=int, default=0)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--num_experts", type=int, default=32)
    p.add_argument("--local_world_size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--csv", type=str, default="moe_ep_uniep_sm_sweep.csv")
    args = p.parse_args()
    I = args.intermediate or args.hidden

    ep_group = init_dist()
    rank, world = dist.get_rank(), dist.get_world_size()
    M, H = args.tokens, args.hidden
    ep = EPOverlapKernels(max_m=M, hidden=H, topk=args.topk, num_experts=args.num_experts,
                          local_world_size=args.local_world_size, ep_group=ep_group,
                          expert_alignment=EPOverlapKernels.cluster_tile_m)

    torch.manual_seed(1234 + rank)
    idx, _w = make_routing(M, args.num_experts, args.topk, seed=1234 + rank)
    x = torch.randn(M, H, dtype=BF16, device="cuda")
    B1 = (torch.randn(args.num_experts // world, 2 * I, H, dtype=BF16, device="cuda") * 0.02).permute(1, 2, 0)
    B2 = (torch.randn(args.num_experts // world, H, I, dtype=BF16, device="cuda") * 0.02).permute(1, 2, 0)

    sync = lambda: dist.barrier(group=ep_group)
    rows = []
    shape = f"(T{M}/rank, H{H}, I{I}, topk{args.topk}, E{args.num_experts}, EP{world})"

    # unfused reference
    d, layout = ep.dispatch_cutedsl(x, idx, EPCommLayoutDesc())
    a1 = ep.group_gemm(d, B1, layout.recv_expert_counts)
    g, u = a1.split(I, dim=-1)
    a2 = (F.silu(g) * u).contiguous()
    c2 = ep.group_gemm(a2, B2, layout.recv_expert_counts)
    sync()
    for name, fn in (("dispatch", lambda: ep.dispatch_cutedsl(x, idx, layout)),
                     ("fc1", lambda: ep.group_gemm(d, B1, layout.recv_expert_counts)),
                     ("fc2", lambda: ep.group_gemm(a2, B2, layout.recv_expert_counts)),
                     ("combine", lambda: ep.combine_cutedsl(
                         c2, layout_desc=layout, push_mode="tile", tile_m=128, tile_n=256))):
        st = measure(fn, args.warmup, args.iters, barrier=sync)
        rows.append(row_from_stats(st, framework="uniEP", op=f"unfused_{name}", dtype="bfloat16",
                                   shape=shape, device="H100", tokens_per_rank=M, hidden=H,
                                   intermediate=I, topk=args.topk, num_experts=args.num_experts,
                                   ep=world, note=""))
        if rank == 0:
            print(f"  unfused_{name:9s} kernel={st['kernel_us']:9.2f} us", flush=True)

    combos = [(None, None), (96, None), (64, 32), (132, 132), (32, 96)]
    for gemm_sm, comm_sm in combos:
        try:
            st = measure(lambda: ep.dispatch_group_gemm(x, idx, B1, None, gemm_num_sm=gemm_sm,
                                                        comm_num_sm=comm_sm),
                         args.warmup, args.iters, barrier=sync)
            if rank == 0:
                print(f"  MK1 gemm_sm={gemm_sm} comm_sm={comm_sm}: kernel={st['kernel_us']:9.2f} us",
                      flush=True)
            rows.append(row_from_stats(st, framework="uniEP", op="mk1_dispatch_group_gemm",
                                       dtype="bfloat16", shape=shape, device="H100",
                                       tokens_per_rank=M, hidden=H, intermediate=I, topk=args.topk,
                                       num_experts=args.num_experts, ep=world,
                                       note=f"gemm_sm={gemm_sm} comm_sm={comm_sm}"))
        except Exception as e:
            if rank == 0:
                print(f"  MK1 gemm_sm={gemm_sm} comm_sm={comm_sm} failed: {type(e).__name__}: {str(e)[:120]}")
        try:
            st = measure(lambda: ep.group_gemm_combine(a2, B2, layout, gemm_num_sm=gemm_sm),
                         args.warmup, args.iters, barrier=sync)
            if rank == 0:
                print(f"  MK2 gemm_sm={gemm_sm} comm_sm={comm_sm}: kernel={st['kernel_us']:9.2f} us",
                      flush=True)
            rows.append(row_from_stats(st, framework="uniEP", op="mk2_group_gemm_combine",
                                       dtype="bfloat16", shape=shape, device="H100",
                                       tokens_per_rank=M, hidden=H, intermediate=I, topk=args.topk,
                                       num_experts=args.num_experts, ep=world,
                                       note=f"gemm_sm={gemm_sm}"))
        except Exception as e:
            if rank == 0:
                print(f"  MK2 gemm_sm={gemm_sm} failed: {type(e).__name__}: {str(e)[:120]}")

    if rank == 0:
        write_csv(args.csv, rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")
    try:
        ep.finalize()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
