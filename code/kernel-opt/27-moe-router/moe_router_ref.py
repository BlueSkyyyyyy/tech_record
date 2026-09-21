#!/usr/bin/env python3
"""MoE router 的 PyTorch 基线：给 self-written kernel 做「同口径」对标。

- gate GEMM: torch.matmul(X, Wg.T)  (bf16, 调 cuBLAS)
- router top-k: torch.topk(choice, 6)  + gather + renorm
- permutation: 用 index_select / index_copy 做 permute / unpermute
- unpermute(weighted): scatter_add 或 index_add

运行（在 kernel_lab 容器里）：
  python3 moe_router_ref.py [M]
"""
import sys
import time

import torch

M = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
H, E, K = 7168, 384, 6
SCALE = 2.5


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    dev = "cuda"
    torch.manual_seed(0)
    x = (0.05 * torch.randn(M, H, device=dev)).bfloat16()
    wg = (0.05 * torch.randn(E, H, device=dev)).bfloat16()
    bias = 0.01 * torch.randn(E, device=dev, dtype=torch.float32)

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"M={M} H={H} E={E} topk={K}")

    # ---- gate GEMM ----
    def gate():
        return torch.matmul(x, wg.t())
    logits = gate()
    t_gate = bench(gate)
    flops = 2 * M * E * H
    print(f"[torch gate ] {t_gate:8.4f} ms  {flops/t_gate/1e9:8.2f} TFLOPS")

    # ---- router top-k ----
    def router():
        s = torch.nn.functional.softplus(logits.float()).sqrt()
        choice = s + bias
        ids = torch.topk(choice, k=K, dim=-1, sorted=True).indices
        w = s.gather(1, ids)
        w = w / w.sum(-1, keepdim=True) * SCALE
        return w, ids
    w_t, id_t = router()
    t_rtr = bench(router)
    print(f"[torch rtr  ] {t_rtr:8.4f} ms  (gate+softplus+topk+gather+renorm)")

    # ---- permutation (permute) ----
    P = M * K
    # 用 argsort 来个「按 expert 归拢」的置换
    order = torch.argsort(id_t.reshape(-1), stable=True)
    src_token = torch.arange(M, device=dev).repeat_interleave(K)[order]
    def permute():
        return x.index_select(0, src_token)
    px = permute()
    t_perm = bench(permute)
    # index_select 对每个出现都读一次 x，故真实流量 = 2*P*H*2
    perm_bytes = 2 * P * H * 2
    print(f"[torch perm ] {t_perm:8.4f} ms  {perm_bytes/t_perm/1e6:8.1f} GB/s  "
          f"({100*perm_bytes/t_perm/1e6/3352.3:5.1f}% HBM, moves {perm_bytes/1e9:.2f} GB)")

    # ---- unpermute ----
    # 把 permuted 结果按 token 加权求和：scatter_add 到 token 维度
    weights_flat = w_t.reshape(-1)[order]
    def unpermute():
        out = torch.zeros(M, H, device=dev, dtype=torch.float32)
        out.index_add_(0, src_token, px.float() * weights_flat[:, None])
        return out.bfloat16()
    t_un = bench(unpermute)
    un_bytes = P * H * 2 + M * H * 2
    print(f"[torch unpm ] {t_un:8.4f} ms  {un_bytes/t_un/1e6:8.1f} GB/s  "
          f"({100*un_bytes/t_un/1e6/3352.3:5.1f}% HBM)")

    print(f"[torch e2e  ] gate {t_gate:.4f} + router {t_rtr:.4f} + perm {t_perm:.4f} "
          f"+ unperm {t_un:.4f} = {t_gate+t_rtr+t_perm+t_un:.4f} ms")


if __name__ == "__main__":
    main()
