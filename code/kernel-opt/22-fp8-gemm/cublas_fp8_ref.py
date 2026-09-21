import sys, time, torch

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
K = int(sys.argv[3]) if len(sys.argv) > 3 else 7168

torch.manual_seed(0)
dev = "cuda"
# e4m3 per-tensor reference via cuBLAS(Lt) backend of torch._scaled_mm
a = (torch.rand(M, K, device=dev) * 2 - 1).to(torch.float8_e4m3fn)
b = (torch.rand(N, K, device=dev) * 2 - 1).to(torch.float8_e4m3fn)
sa = torch.tensor(1.0, device=dev)
sb = torch.tensor(1.0, device=dev)

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

fl = 2.0 * M * N * K

# fp8 per-tensor
try:
    c = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    ms = bench(lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
    print(f"cublas fp8 per-tensor {M}x{N}x{K}: {ms:.4f} ms  {fl/ms/1e9:.2f} TFLOPS  ({100*fl/ms/1e9/1978:.1f}% of 1978)")
except Exception as e:
    print("fp8 per-tensor failed:", str(e).splitlines()[0])

# fp8 rowwise (per-token activation scale / per-row weight scale)
try:
    sa_r = (torch.rand(M, 1, device=dev) * 0.01 + 0.001)
    sb_r = (torch.rand(1, N, device=dev) * 0.01 + 0.001)
    c = torch._scaled_mm(a, b.t(), scale_a=sa_r, scale_b=sb_r, out_dtype=torch.bfloat16)
    ms = bench(lambda: torch._scaled_mm(a, b.t(), scale_a=sa_r, scale_b=sb_r, out_dtype=torch.bfloat16))
    print(f"cublas fp8 per-row   {M}x{N}x{K}: {ms:.4f} ms  {fl/ms/1e9:.2f} TFLOPS  ({100*fl/ms/1e9/1978:.1f}% of 1978)")
except Exception as e:
    print("fp8 per-row failed:", str(e).splitlines()[0])

# bf16 reference
ab = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
bb = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
ms = bench(lambda: ab @ bb.t())
print(f"cublas bf16          {M}x{N}x{K}: {ms:.4f} ms  {fl/ms/1e9:.2f} TFLOPS  ({100*fl/ms/1e9/989:.1f}% of 989)")
