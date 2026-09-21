#!/usr/bin/env python3
"""25 MoE grouped GEMM 的 cuBLAS 参照：per-expert loop（torch._scaled_mm）。

三种口径（都在同一 MoE 分布上）：
  1. eager loop        —— 384 次 python 级 _scaled_mm（含 host 派发开销）
  2. cuda-graph loop   —— 把 384 次 launch 录成一张 CUDA graph 回放（去掉 host 开销）
  3. masked(batched)   —— decode：A=(G,max_m,K) 全部按 max_m 计算（cuBLAS batched 语义，无早退）

shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
  hidden=7168(K), moe_intermediate_size=3072(N), n_routed_experts=384, num_experts_per_tok=6
"""
import argparse
import random

import torch

torch.manual_seed(0)


def make_dist(G, expected, align):
    actual, aligned, off = [], [], []
    o = 0
    for _ in range(G):
        a = max(1, int(expected * (0.7 + 0.6 * random.random())))
        al = (a + align - 1) // align * align
        actual.append(a)
        aligned.append(al)
        off.append(o)
        o += al
    return actual, aligned, off, o


def bench(fn, warm=5, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--G", type=int, default=384)
    ap.add_argument("--N", type=int, default=3072)
    ap.add_argument("--K", type=int, default=7168)
    ap.add_argument("--max_m", type=int, default=128)
    args = ap.parse_args()

    dev = "cuda"
    G, N, K, max_m = args.G, args.N, args.K, args.max_m
    sa = torch.tensor(0.8, device=dev)
    sb = torch.tensor(0.9, device=dev)
    print(f"device={torch.cuda.get_device_name(0)}  G={G} N={N} K={K}")

    # ---------------- prefill contiguous ----------------
    expected = args.tokens * 6 // G
    actual, aligned, off, M_total = make_dist(G, expected, 128)
    print(f"\n[prefill] tokens={args.tokens} expected_m/expert={expected} "
          f"M_total={M_total} sum_actual={sum(actual)} "
          f"waste={100*(M_total-sum(actual))/M_total:.1f}%")

    A = (torch.randn(M_total, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    B = (torch.randn(G, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    Bt = [B[g].t() for g in range(G)]          # (K,N) column-major
    D = torch.empty(M_total, N, device=dev, dtype=torch.float32)

    def loop_eager():
        for g in range(G):
            torch._scaled_mm(A[off[g]:off[g] + aligned[g]], Bt[g], scale_a=sa, scale_b=sb,
                             out_dtype=torch.float32, out=D[off[g]:off[g] + aligned[g]])

    t_eager = bench(loop_eager)
    flops_aligned = 2.0 * M_total * N * K
    print(f"  eager loop      {t_eager:8.4f} ms  {flops_aligned/t_eager/1e9:8.2f} TFLOPS")

    # CUDA graph：录 384 次 launch，去掉 host 派发
    try:
        loop_eager()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for g in range(G):
                torch._scaled_mm(A[off[g]:off[g] + aligned[g]], Bt[g], scale_a=sa, scale_b=sb,
                                 out_dtype=torch.float32, out=D[off[g]:off[g] + aligned[g]])
        t_graph = bench(gr.replay)
        print(f"  graph loop      {t_graph:8.4f} ms  {flops_aligned/t_graph/1e9:8.2f} TFLOPS")
    except Exception as ex:  # noqa: BLE001
        print(f"  graph loop      (capture failed: {type(ex).__name__}: {ex})")
        t_graph = None

    # ---------------- decode masked (batched, padded) ----------------
    Am = (torch.randn(G, max_m, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    Dm = torch.empty(G, max_m, N, device=dev, dtype=torch.float32)
    sum_actual_dec = G * 32  # 仅用于展示有用 FLOPs

    def masked_eager():
        for g in range(G):
            torch._scaled_mm(Am[g], Bt[g], scale_a=sa, scale_b=sb, out_dtype=torch.float32,
                             out=Dm[g])

    t_dec = bench(masked_eager)
    flops_dec_pad = 2.0 * G * max_m * N * K
    print(f"\n[decode] max_m={max_m} (batched/padded, 无早退)")
    print(f"  eager loop      {t_dec:8.4f} ms  {flops_dec_pad/t_dec/1e9:8.2f} TFLOPS(padded)  "
          f"B-read {G*N*K/t_dec/1e6:7.1f} GB/s")
    try:
        masked_eager()
        torch.cuda.synchronize()
        gr2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr2):
            for g in range(G):
                torch._scaled_mm(Am[g], Bt[g], scale_a=sa, scale_b=sb, out_dtype=torch.float32,
                                 out=Dm[g])
        t_dec_g = bench(gr2.replay)
        print(f"  graph loop      {t_dec_g:8.4f} ms  {flops_dec_pad/t_dec_g/1e9:8.2f} TFLOPS(padded)  "
              f"B-read {G*N*K/t_dec_g/1e6:7.1f} GB/s")
    except Exception as ex:  # noqa: BLE001
        print(f"  graph loop      (capture failed: {type(ex).__name__}: {ex})")


if __name__ == "__main__":
    main()
