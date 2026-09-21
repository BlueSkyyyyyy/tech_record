"""54 融合 cross-entropy —— PyTorch eager 基线（真实用户路径）。
   logits = x @ W^T ; loss = cross_entropy(logits, target)
编译/运行（容器内）：python3 torch_eager.py
"""
import torch, time, torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
M, H, V = 8192, 7168, 129280          # DeepSeek-V4-Pro: hidden=7168, vocab=129280
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


# 1) cuBLAS GEMM（物化 bf16 logits）
logits = torch.empty(M, V, dtype=torch.bfloat16, device=dev)
g = lambda: torch.mm(x, W.t(), out=logits)
print("cublas bf16 gemm      %8.3f ms  %7.1f TFLOPS" % (bench(g), flops / bench(g) / 1e9))

# 2) 物化 logits 后 cross_entropy（读 bf16 logits 一遍）
print("F.cross_entropy(bf16) %8.3f ms" % bench(lambda: F.cross_entropy(logits, tgt)))

# 3) fp32 logits（eager log_softmax 会把 logits 升 fp32 物化）
def eager_ce_fp32():
    lg = logits.float()
    return F.cross_entropy(lg, tgt)
print("CE with .float()      %8.3f ms" % bench(eager_ce_fp32))

# 4) 端到端 eager：lm_head + cross_entropy
def e2e():
    lg = x @ W.t()
    return F.cross_entropy(lg, tgt)
print("e2e x@W.t + CE        %8.3f ms" % bench(e2e))
print("peak mem @e2e         %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("loss(reference)       %.6f" % e2e().item())
