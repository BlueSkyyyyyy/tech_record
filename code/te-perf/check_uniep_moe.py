"""Correctness gate for the uniEP MoE numbers: fused vs unfused vs a pure-torch MoE.

The CUDA (flash_comm.ep) kernels only support topk in {6, 8}, so at topk=16 all
uniEP paths exercised here are the CuTeDSL ones (flash_comm.ep_overlap). This
script checks that those paths actually compute a MoE layer, i.e. that the
latencies reported by bench_uniep_ep.py are not from a fast-but-wrong kernel:

    y[t] = sum_j  w[t, j] * E_{idx[t,j]}(x[t])

Reference is a per-rank gather + matmul over the *global* token set (all-gathered
routing), reduced across ranks — no dependence on flash_comm maths.

Run:
    NCCL_NVLS_ENABLE=0 FLASH_COMM_BUFFER_BACKEND=cuda_ipc NCCL_LSA_TEAM_SIZE=8 \
    torchrun --nproc_per_node=8 check_uniep_moe.py --tokens 1024 --hidden 1024
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import BF16, make_routing  # noqa: E402

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLASH_COMM_BUFFER_BACKEND", "cuda_ipc")

from flash_comm.ep.ep_kernels import EPCommLayoutDesc  # noqa: E402
from flash_comm.ep_overlap import EPOverlapKernels  # noqa: E402
from flash_comm.utils.test_utils import init_dist  # noqa: E402


def torch_moe_ref(x_all, idx_all, w_all, W1, W2, first_expert, I):
    """Contributions of this rank's experts to every token, accumulated in fp32."""
    T = x_all.shape[0]
    epr = W1.shape[0]
    contrib = torch.zeros(T, W2.shape[1], dtype=torch.float32, device=x_all.device)
    for le in range(epr):
        e = first_expert + le
        hit = idx_all == e                       # [T, topk]
        if not hit.any():
            continue
        tok = hit.any(dim=1).nonzero(as_tuple=True)[0]
        sel = hit[tok]                           # [n, topk]
        xin = x_all[tok].float()
        y1 = xin @ W1[le].float().t()            # [n, 2I]
        act = (F.silu(y1[:, :I]) * y1[:, I:])
        out = act @ W2[le].float().t()           # [n, H]
        # each token selects an expert at most once, so a plain masked sum is exact
        contrib.index_add_(0, tok, out * sel.float().sum(dim=1, keepdim=True))
    return contrib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--intermediate", type=int, default=0)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--num_experts", type=int, default=32)
    p.add_argument("--local_world_size", type=int, default=8)
    args = p.parse_args()
    I = args.intermediate or args.hidden

    ep_group = init_dist()
    rank, world = dist.get_rank(), dist.get_world_size()
    epr = args.num_experts // world
    M, H = args.tokens, args.hidden

    ep = EPOverlapKernels(max_m=M, hidden=H, topk=args.topk, num_experts=args.num_experts,
                          local_world_size=args.local_world_size, ep_group=ep_group,
                          expert_alignment=EPOverlapKernels.cluster_tile_m)

    torch.manual_seed(1234 + rank)
    idx, w = make_routing(M, args.num_experts, args.topk, seed=1234 + rank)
    x = torch.randn(M, H, dtype=BF16, device="cuda")
    W1 = torch.randn(epr, 2 * I, H, dtype=BF16, device="cuda") * 0.02
    W2 = torch.randn(epr, H, I, dtype=BF16, device="cuda") * 0.02
    B1 = W1.permute(1, 2, 0)   # k-major view of (E, N, K); must NOT be made contiguous
    B2 = W2.permute(1, 2, 0)

    # ---- uniEP unfused pipeline ----
    d, layout = ep.dispatch_cutedsl(x, idx, EPCommLayoutDesc())
    a1 = ep.group_gemm(d, B1, layout.recv_expert_counts)
    g, u = a1.split(I, dim=-1)
    a2 = (F.silu(g) * u).contiguous()
    c2 = ep.group_gemm(a2, B2, layout.recv_expert_counts)
    y_seq = ep.combine_cutedsl(c2, layout_desc=layout, push_mode="tile", tile_m=128, tile_n=256,
                               ).float()

    # ---- uniEP fused pipeline ----
    f1, _w, layout_f = ep.dispatch_group_gemm(x, idx, B1, None)
    gf, uf = f1.split(I, dim=-1)
    a2f = (F.silu(gf) * uf).contiguous()
    y_fused, _ = ep.group_gemm_combine(a2f, B2, layout_f)
    y_fused = torch.empty_like(y_seq) if y_fused is None else y_fused.float()

    # ---- reference ----
    x_all = torch.empty(world * M, H, dtype=BF16, device="cuda")
    idx_all = torch.empty(world * M, args.topk, dtype=idx.dtype, device="cuda")
    w_all = torch.empty(world * M, args.topk, dtype=w.dtype, device="cuda")
    dist.all_gather_into_tensor(x_all, x, group=ep_group)
    dist.all_gather_into_tensor(idx_all, idx, group=ep_group)
    dist.all_gather_into_tensor(w_all, w, group=ep_group)
    y_ref = torch_moe_ref(x_all, idx_all, w_all, W1, W2, rank * epr, I)
    dist.all_reduce(y_ref, group=ep_group)  # sum the per-rank expert contributions

    my_rows = slice(rank * M, (rank + 1) * M)
    ref = y_ref[my_rows]
    dist.barrier(group=ep_group)


    def report(tag, got):
        diff = (got - ref).abs()
        denom = ref.abs().max().clamp_min(1e-6)
        print(f"[rank {rank}] {tag:22s} max|diff|={diff.max().item():.3e} "
              f"rel={diff.max().item() / denom.item():.3e} "
              f"mean|ref|={ref.abs().mean().item():.4f}", flush=True)

    print(f"[rank {rank}] shape: x={tuple(x.shape)} dispatch_rows={int(d.shape[0])} "
          f"mk1_rows={int(f1.shape[0])}", flush=True)
    report("e2e_sequential", y_seq)
    report("e2e_fused", y_fused)
    try:
        ep.finalize()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
