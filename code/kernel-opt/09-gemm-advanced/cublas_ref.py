"""cuBLAS (via torch.matmul) 的 fp32 SGEMM 参考性能。

在 kernel_lab 容器里跑：
    docker exec kernel_lab python3 /ssd/home/.../cublas_ref.py
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False  # 纯 FP32，和我们的 kernel 同口径
torch.backends.cudnn.allow_tf32 = False

M = N = K = 2048
dev = "cuda"
a = torch.randn(M, K, device=dev)
b = torch.randn(K, N, device=dev)

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

print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
print("cublas matmul (tf32=%s)" % torch.backends.cuda.matmul.allow_tf32)
print("M=N=K=%d  %.4f ms  %.2f TFLOPS  (%.1f%% of fp32 peak)"
      % (M, ms, tflops, 100.0 * tflops / 66.9))
