"""DeepEP (V2.5 EPBuffer) dispatch/combine benchmark.

Same workload as bench_uniep_ep.py:
    tokens/rank = 1024 (or 128), topk = 16, EP = 8, experts = 32, H = 1024 / 3072

Ops measured (bf16):
    deepep_dispatch            plain dispatch (expert_alignment=1, do_cpu_sync=True)
    deepep_dispatch_expanded   expanded layout for grouped GEMM (align=128, expand + zero padding)
    deepep_combine             combine on the plain handle (topk_weights reduction)
    deepep_combine_expanded    "reduced" combine on the expanded handle (training path)
    deepep_e2e                 dispatch + combine only (no expert GEMM)

DeepEP needs NCCL >= 2.32.3 with the Gin device API; the container has it at
/opt/nccl232, so every run must preload it:

    LD_PRELOAD=/opt/nccl232/nvidia/nccl/lib/libnccl.so.2 \
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=8 bench_deepep_ep.py \
        --tokens 1024 128 --hidden 1024 3072 --csv moe_ep_deepep.csv
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import (  # noqa: E402
    BF16, device_name, fmt_shape, gather_rows, local_m_splits, make_routing, measure, measure_robust, probe_quiet,
    row_from_stats,
    write_csv,
)

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")  # no NVSwitch/Fabric Manager here
import deep_ep  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1024, 128])
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 3072])
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--num_experts", type=int, default=32)
    p.add_argument("--local_world_size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--csv", type=str, default="moe_ep_deepep.csv")
    return p.parse_args()


def init_dist(local_world_size):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world, rank=rank,
                            device_id=torch.device(f"cuda:{local_rank}"))
    group = dist.new_group(list(range(world)))
    torch.cuda.synchronize()
    return rank, world, group


def main():
    args = parse_args()
    rank, world, group = init_dist(args.local_world_size)
    dev = device_name()
    rows = []
    buffer = None

    def sync():
        dist.barrier(group=group)

    try:
        for H in args.hidden:
            for M in args.tokens:
                if rank == 0:
                    print(f"\n=== DeepEP: tokens/rank={M} H={H} topk={args.topk} "
                          f"E={args.num_experts} EP={world} ===", flush=True)

                if buffer is not None:
                    buffer.destroy()
                    buffer = None
                buffer = deep_ep.EPBuffer(
                    group,
                    num_max_tokens_per_rank=M,
                    hidden=H,
                    num_topk=args.topk,
                    prefer_overlap_with_compute=False,  # same as DeepEP's own test harness
                    explicitly_destroy=True,
                )
                num_sms = buffer.get_theoretical_num_sms(args.num_experts, args.topk)
                if rank == 0:
                    print(f"  buffer: num_bytes={buffer.num_bytes / 2**20:.0f} MB "
                          f"num_allocated_qps={buffer.num_allocated_qps} num_sms={num_sms}", flush=True)

                torch.manual_seed(1234 + rank)
                idx_i32, topk_w = make_routing(M, args.num_experts, args.topk, seed=1234 + rank)
                topk_idx = idx_i32.to(deep_ep.topk_idx_t)
                x = torch.randn(M, H, dtype=BF16, device="cuda")

                shape = fmt_shape([f"T{M}/rank", f"H{H}", f"I{H}", f"topk{args.topk}",
                                   f"E{args.num_experts}", f"EP{world}"])
                probe_quiet(group)

                def add(op, stats, note=""):
                    rows.append(row_from_stats(
                        stats, framework="DeepEP", op=op, dtype="torch.bfloat16", shape=shape,
                        device=dev, tokens_per_rank=M, hidden=H, intermediate=H,
                        topk=args.topk, num_experts=args.num_experts, ep=world, note=note))
                    if rank == 0:
                        print(f"  {op:28s} kernel={stats['kernel_us']:9.2f} us  "
                              f"wall={stats['wall_med_us']:9.2f}/{stats['wall_p90_us']:9.2f} us  {note}",
                              flush=True)

                plain_args = dict(
                    x=x, topk_idx=topk_idx, topk_weights=topk_w,
                    num_experts=args.num_experts, num_max_tokens_per_rank=M,
                    expert_alignment=1, num_sms=num_sms, do_cpu_sync=True,
                )
                recv_x, recv_idx, recv_w, handle, _evt = buffer.dispatch(**plain_args)
                sync()

                expanded_args = dict(plain_args)
                expanded_args.update(expert_alignment=128, do_expand=True,
                                     use_tma_aligned_col_major_sf=True)
                exp_x, _exp_idx, exp_w, exp_handle, _e2 = buffer.dispatch(**expanded_args)
                sync()

                recv_rows = int(recv_x.shape[0])
                exp_rows = int(exp_x.shape[0])
                splits = local_m_splits(idx_i32, args.num_experts, world, group)
                note_rows = f"recv_rows={recv_rows}"

                st = measure_robust(lambda: buffer.dispatch(**plain_args),
                             args.warmup, args.iters, barrier=sync)
                add("deepep_dispatch", st, note_rows)

                st = measure_robust(lambda: buffer.dispatch(**expanded_args),
                             args.warmup, args.iters, barrier=sync)
                add("deepep_dispatch_expanded", st,
                    f"expanded_rows={exp_rows} align=128 local_m={splits}")

                # NOTE: the combine input must NOT be the dispatch output tensor --
                # that aliases DeepEP's internal receive buffer and makes the reduce
                # kernel ~20x slower (measured 4.1ms vs 0.18ms at M=1024/H=3072).
                c_plain = recv_x.detach().clone()
                c_exp = exp_x.detach().clone()
                sync()

                combine_args = dict(x=c_plain, topk_weights=recv_w, handle=handle, num_sms=num_sms)
                st = measure_robust(lambda: buffer.combine(**combine_args),
                             args.warmup, args.iters, barrier=sync)
                add("deepep_combine", st, f"in_rows={recv_rows}, topk_weights reduction")

                exp_combine_args = dict(x=c_exp, handle=exp_handle, num_sms=num_sms)
                st = measure_robust(lambda: buffer.combine(**exp_combine_args),
                             args.warmup, args.iters, barrier=sync)
                add("deepep_combine_expanded", st,
                    f"in_rows={exp_rows}, expand-mode reduce (training path)")

                # async convention (what DeepEP's own test harness uses): the kernel is
                # timed without the host-side recv-count sync. do_cpu_sync=True costs
                # 3-7ms per call at H=3072 in this environment, which is a host stall,
                # not a kernel cost, so both conventions are reported.
                async_plain = dict(plain_args)
                async_plain.update(async_with_compute_stream=True, defer_epilogue=True,
                                   do_cpu_sync=False)
                async_expanded = dict(expanded_args)
                async_expanded.update(async_with_compute_stream=True, defer_epilogue=True,
                                      do_cpu_sync=False)

                def disp_async(kw):
                    ev = buffer.dispatch(**kw)
                    ev.current_stream_wait()
                    return ev

                st = measure_robust(lambda: disp_async(async_plain), args.warmup, args.iters, barrier=sync)
                add("deepep_dispatch_async", st, f"recv_rows={recv_rows} async+defer, no host sync")

                st = measure_robust(lambda: disp_async(async_expanded), args.warmup, args.iters, barrier=sync)
                add("deepep_dispatch_expanded_async", st,
                    f"expanded_rows={exp_rows} async+defer, no host sync")

                def e2e():
                    r_x, _r_i, r_w, h, _e = buffer.dispatch(**plain_args)
                    return buffer.combine(x=r_x.detach().clone(), topk_weights=r_w, handle=h,
                                          num_sms=num_sms)

                st = measure_robust(e2e, args.warmup, args.iters, barrier=sync)
                add("deepep_e2e", st, "dispatch + combine (comm only, no expert GEMM)")

                def e2e_expanded():
                    e_x, _i, _w2, h, _e = buffer.dispatch(**expanded_args)
                    return buffer.combine(x=e_x.detach().clone(), handle=h, num_sms=num_sms)

                st = measure_robust(e2e_expanded, args.warmup, args.iters, barrier=sync)
                add("deepep_e2e_expanded", st,
                    "expanded dispatch + combine (matches uniEP e2e comm semantics)")

                del recv_x, exp_x, c_plain, c_exp, handle, exp_handle, x, topk_idx, topk_w
                torch.cuda.empty_cache()
                sync()
    finally:
        merged = gather_rows(rows, group) if rows else []
        if rank == 0 and merged:
            write_csv(args.csv, merged)
            print(f"\nwrote {len(merged)} rows -> {args.csv}")
        try:
            if buffer is not None:
                buffer.destroy()
        except Exception:
            pass
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
