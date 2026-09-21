#!/usr/bin/env python3
"""同 shape 的 cuBLAS bf16 稠密 GEMM 参考（给 30 篇的差距百分比用）。

K1 = up+gate 合并：M=Pp, N=2I=6144, K=H=7168
K2 = down        ：M=Pp, N=H=7168, K=I=3072
用 M=Pp=49152（M_tokens=8192, topk=6, balanced）与 98304（M=16384）。
"""
import torch

def bench(fn, warm=10, it=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it

dev = 'cuda'
print('device:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__)

cases = [('K1 up+gate', 49152, 6144, 7168), ('K1 up+gate', 98304, 6144, 7168),
         ('K2 down', 49152, 7168, 3072), ('K2 down', 98304, 7168, 3072)]
for name, M, N, K in cases:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    ms = bench(lambda: torch.matmul(a, b.t()))
    fl = 2.0 * M * N * K
    print(f'{name:12s} M={M:6d} N={N:5d} K={K:5d}  {ms:8.4f} ms  {fl/ms/1e9:8.2f} TFLOPS  ({fl/ms/1e9/989*100:5.1f}% of bf16 peak)')
