"""55 FP8 fused CE —— PyTorch / cuBLAS FP8 GEMM 参考（同 shape 同数据分步）。

对比：
  1) cuBLAS bf16 gemm  + F.cross_entropy
  2) cuBLAS FP8 gemm（torch._scaled_mm，per-tensor e4m3）+ F.cross_entropy
  3) 精度：fp8 logits vs bf16 logits；loss 差异
运行（容器内）：python3 torch_fp8_ref.py [M]
"""
import sys, time
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
H, V = 7168, 129280          # DeepSeek-V4-Pro: hidden=7168, vocab=129280
x = (torch.rand(M, H, dtype=torch.bfloat16, device=dev) - 0.5) * 0.05
W = (torch.rand(V, H, dtype=torch.bfloat16, device=dev) - 0.5) * 0.05
tgt = torch.randint(0, V, (M,), device=dev)
flops = 2.0 * M * V * H


def bench(fn, it=20, wu=5):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it * 1e3


# ---- 1) cuBLAS bf16 + CE ----
lg_bf = torch.empty(M, V, dtype=torch.bfloat16, device=dev)
t_g = bench(lambda: torch.mm(x, W.t(), out=lg_bf))
t_ce = bench(lambda: F.cross_entropy(lg_bf, tgt))
loss_bf = F.cross_entropy(lg_bf, tgt).item()
print(f"[bf16 ] gemm {t_g:8.3f} ms {flops/t_g/1e9:7.1f} TFLOPS | CE {t_ce:6.3f} ms | total {t_g+t_ce:8.3f} ms | loss {loss_bf:.6f}")

# ---- 2) FP8 per-tensor e4m3（torch._scaled_mm / cuBLAS）----
def quant(t):
    amax = t.abs().max().float()
    scale = amax / 448.0
    q = (t.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale

xq, sa = quant(x)
Wq, sw = quant(W)
lg_f8 = torch.empty(M, V, dtype=torch.bfloat16, device=dev)
try:
    def f8mm():
        return torch._scaled_mm(xq, Wq.t(), sa, sw, out_dtype=torch.bfloat16)
    lg_f8 = f8mm()
    torch.cuda.synchronize()
    t_g8 = bench(f8mm)
    loss_f8 = F.cross_entropy(lg_f8, tgt)
    t_ce8 = bench(lambda: F.cross_entropy(lg_f8, tgt))
    t8 = bench(lambda: F.cross_entropy(f8mm(), tgt))
    # 精度
    a = lg_bf.float(); b = lg_f8.float()
    rel = ((b - a).pow(2).sum() / a.pow(2).sum()).sqrt().item()
    mx = (b - a).abs().max().item()
    print(f"[fp8  ] gemm {t_g8:8.3f} ms {flops/t_g8/1e9:7.1f} TFLOPS | CE {t_ce8:6.3f} ms | total {t8:8.3f} ms | loss {loss_f8.item():.6f}")
    print(f"[acc  ] fp8 vs bf16 logits rel-RMS {rel:.3e} maxabs {mx:.3e} | loss diff {abs(loss_f8.item()-loss_bf):.3e}")
    print(f"[scales] sa={sa.item():.6g} sw={sw.item():.6g}")
except Exception as e:
    print("[fp8  ] torch._scaled_mm failed:", repr(e))
