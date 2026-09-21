import sys, time, torch

# MoE gate GEMM: logits[M, E] = X[M, H] @ Wg[E, H]^T
# DeepSeek-V4-Pro: H=hidden=7168, E=n_routed_experts=384
H = int(sys.argv[3]) if len(sys.argv) > 3 else 7168
E = int(sys.argv[4]) if len(sys.argv) > 4 else 384
Ms = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "16384,32768").split(",")]
_dummy = sys.argv[2] if len(sys.argv) > 2 else ""

torch.manual_seed(0)
dev = "cuda"


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}  H={H} E={E}")
for M in Ms:
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    w = torch.randn(E, H, device=dev, dtype=torch.bfloat16)
    fl = 2.0 * M * E * H
    ms = bench(lambda: x @ w.t())
    print(f"cublas bf16 gate M={M:6d} N={E} K={H}: {ms:.4f} ms  {fl/ms/1e9:8.2f} TFLOPS  "
          f"({100*fl/ms/1e9/989:.1f}% of 989)")
