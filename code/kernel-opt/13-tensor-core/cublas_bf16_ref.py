"""cuBLAS (via torch.matmul) 的 BF16 GEMM 参考性能，与手写 TC kernel 同口径。

在 kernel_lab 容器里跑：
    docker exec kernel_lab python3 /ssd/.../13-tensor-core/cublas_bf16_ref.py
"""
import torch

M = N = K = 2048
dev = "cuda"
a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)

for _ in range(10):
    c = a @ b
torch.cuda.synchronize()

iters = 50
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    c = a @ b
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / iters
flops = 2.0 * M * N * K
tflops = flops / (ms / 1e3) / 1e12

print("torch", torch.__version__, "device", torch.cuda.get_device_name(0),
      "dtype", c.dtype)
print("cublas bf16 matmul")
print("M=N=K=%d  %.4f ms  %.2f TFLOPS  (%.1f%% of bf16 TC peak)"
      % (M, ms, tflops, 100.0 * tflops / 989.0))
