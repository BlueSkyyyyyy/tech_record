"""TransformerEngine grouped GEMM (GroupedLinear) benchmark — the "group-gemm" baseline.

Workload identical to the EP benchmarks: routing of `tokens/rank` tokens over
`num_experts` experts with topk, EP = world size, so the local grouped GEMM sees
the same per-expert row counts as uniEP / DeepEP do after the all-to-all.

Ops measured (bf16, forward plus the fwd+bwd pair):
    te_grouped_gemm_fc1        (rows, K=H) @ (E_local, 2I, H)      -> (rows, 2I)
    te_grouped_gemm_fc2        (rows, K=I) @ (E_local, H, I)      -> (rows, H)
    te_grouped_gemm_fc1_fwd_bwd
    te_grouped_gemm_fc2_fwd_bwd
    te_grouped_gemm_layer      FC1 + swiglu + FC2 (expert GEMM part of one MoE layer)

Launch:
    torchrun --nproc_per_node=8 bench_te_group_gemm.py \
        --tokens 1024 128 --hidden 1024 3072 --csv moe_ep_tegmm.csv
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

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")  # no NVSwitch/Fabric Manager in this container
import transformer_engine.pytorch as te  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1024, 128])
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 3072])
    p.add_argument("--intermediate", type=int, nargs="+", default=None)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--num_experts", type=int, default=32)
    p.add_argument("--local_world_size", type=int, default=8)
    p.add_argument("--align", type=int, default=1,
                   help="round each expert's row count up to this multiple (128 = grouped-GEMM alignment)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--csv", type=str, default="moe_ep_tegmm.csv")
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


def tflops(rows, n, k, us):
    return 2.0 * rows * n * k / (us * 1e-6) / 1e12


def main():
    args = parse_args()
    if args.intermediate is None:
        args.intermediate = list(args.hidden)
    rank, world, group = init_dist(args.local_world_size)
    dev = device_name()
    epr = args.num_experts // world
    rows_out = []

    def sync():
        dist.barrier(group=group)

    try:
        for H, I in zip(args.hidden, args.intermediate):
            for M in args.tokens:
                torch.manual_seed(1234 + rank)
                idx_i32, _w = make_routing(M, args.num_experts, args.topk, seed=1234 + rank)
                splits = local_m_splits(idx_i32, args.num_experts, world, group)
                if args.align > 1:
                    splits = [((m + args.align - 1) // args.align) * args.align for m in splits]
                total_rows = int(sum(splits))
                if rank == 0:
                    print(f"\n=== TE grouped GEMM: tokens/rank={M} H={H} I={I} E_local={epr} "
                          f"rows={total_rows} m_splits={splits} ===", flush=True)

                x = torch.randn(total_rows, H, dtype=BF16, device="cuda")
                fc1 = te.GroupedLinear(num_gemms=epr, in_features=H, out_features=2 * I,
                                       bias=False, params_dtype=torch.bfloat16, device="cuda")
                fc2 = te.GroupedLinear(num_gemms=epr, in_features=I, out_features=H,
                                       bias=False, params_dtype=torch.bfloat16, device="cuda")
                for mod in (fc1, fc2):
                    for p in mod.parameters():
                        p.data.normal_(0, 0.02)

                shape = fmt_shape([f"T{M}/rank", f"H{H}", f"I{I}", f"topk{args.topk}",
                                   f"E{args.num_experts}", f"EP{world}"])
                row_meta = dict(device=dev, tokens_per_rank=M, hidden=H, intermediate=I,
                                topk=args.topk, num_experts=args.num_experts, ep=world)
                probe_quiet(group)

                def add(op, stats, note="", flops=None):
                    rec = row_from_stats(stats, framework="TE group-gemm", op=op,
                                         dtype="torch.bfloat16", shape=shape, note=note, **row_meta)
                    rows_out.append(rec)
                    if rank == 0:
                        tf = f"  {flops(stats['kernel_us']):7.1f} TFLOPS" if flops else ""
                        print(f"  {op:32s} kernel={stats['kernel_us']:9.2f} us  "
                              f"wall={stats['wall_med_us']:9.2f}/{stats['wall_p90_us']:9.2f} us{tf}  {note}",
                              flush=True)

                # ---------------- FC1 ----------------
                y1 = fc1(x, splits)
                sync()
                fl1 = lambda us: tflops(total_rows, 2 * I, H, us)
                st = measure_robust(lambda: fc1(x, splits), args.warmup, args.iters, barrier=sync)
                add("te_grouped_gemm_fc1", st, f"N=2I={2 * I}, K={H}, rows={total_rows}", flops=fl1)

                y2_in = (F.silu(y1[:, :I]) * y1[:, I:]).contiguous()
                fl2 = lambda us: tflops(total_rows, H, I, us)
                st = measure_robust(lambda: fc2(y2_in, splits), args.warmup, args.iters, barrier=sync)
                add("te_grouped_gemm_fc2", st, f"N=H={H}, K=I={I}, rows={total_rows}", flops=fl2)

                # ---------------- fwd + bwd ----------------
                def fc1_step():
                    out = fc1(x, splits)
                    out.backward(torch.ones_like(out), retain_graph=False)
                    fc1.zero_grad(set_to_none=True)
                    return out

                st_fwd, st_step = None, None
                try:
                    st_step = measure_robust(fc1_step, max(2, args.warmup // 2), args.iters, barrier=sync)
                    add("te_grouped_gemm_fc1_fwd_bwd", st_step,
                        f"fwd+bwd (grad wrt input+weight), N=2I={2 * I}", flops=None)
                except Exception as e:  # pragma: no cover
                    if rank == 0:
                        print(f"  fc1 fwd_bwd failed: {type(e).__name__}: {e}")

                def fc2_step():
                    # fresh leaf so repeated backward() in the timing loop is legal
                    inp = y2_in.detach().clone().requires_grad_(True)
                    out = fc2(inp, splits)
                    out.backward(torch.ones_like(out), retain_graph=False)
                    fc2.zero_grad(set_to_none=True)
                    return out

                try:
                    st_step = measure_robust(fc2_step, max(2, args.warmup // 2), args.iters, barrier=sync)
                    add("te_grouped_gemm_fc2_fwd_bwd", st_step,
                        f"fwd+bwd, N=H={H}, K=I={I}", flops=None)
                except Exception as e:  # pragma: no cover
                    if rank == 0:
                        print(f"  fc2 fwd_bwd failed: {type(e).__name__}: {e}")

                # ---------------- FC1 + swiglu + FC2 ----------------
                def layer():
                    a = fc1(x, splits)
                    g, u = a.split(I, dim=-1)
                    b = (F.silu(g) * u).contiguous()
                    return fc2(b, splits)

                st = measure_robust(layer, args.warmup, args.iters, barrier=sync)
                add("te_grouped_gemm_layer", st, "FC1 + swiglu + FC2 (forward)")

                def layer_step():
                    a = fc1(x, splits)
                    g, u = a.split(I, dim=-1)
                    b = (F.silu(g) * u).contiguous()
                    out = fc2(b, splits)
                    out.backward(torch.ones_like(out))
                    fc1.zero_grad(set_to_none=True)
                    fc2.zero_grad(set_to_none=True)
                    return out

                try:
                    st = measure_robust(layer_step, max(2, args.warmup // 2), args.iters, barrier=sync)
                    add("te_grouped_gemm_layer_fwd_bwd", st,
                        "FC1 + swiglu + FC2 fwd+bwd (autograd; weight grads 本地, 不含跨卡 reduce)")
                except Exception as e:  # pragma: no cover
                    if rank == 0:
                        print(f"  layer fwd_bwd failed: {type(e).__name__}: {e}")

                del x, fc1, fc2, y1, y2_in
                torch.cuda.empty_cache()
                sync()
    finally:
        merged = gather_rows(rows_out, group) if rows_out else []
        if rank == 0 and merged:
            write_csv(args.csv, merged)
            print(f"\nwrote {len(merged)} rows -> {args.csv}")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
