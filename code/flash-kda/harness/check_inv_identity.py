#!/usr/bin/env python
"""P2-1 自测：验证 K1 块内求逆 ``inv_fwd_subst_fused_1warp`` 的代数。

对应 ``csrc/smxx/utils.cuh:213-369``：
  X = I + L = [[A,0],[C,B]]   (A,B: 8x8 单位下三角, C = L[8:16,0:8])
  X^-1 = [[A^-1,0],[-B^-1 C A^-1, B^-1]]
实现：
  1. 8x8 对角块用 fp32 forward substitution 求逆（对应 utils.cuh:229-252）；
  2. 组装 P=[A^-1 0;0 B^-1]、M=[0 0;C 0]，两次 bf16 HMMA 得
     o = (-(P@M)) @ P，INV = P + o（对应 utils.cuh:330-355）。

对比 golden：``torch.linalg.inv(I+L)``（fp64）。分两档报告：
  - fp64 纯公式：验证 Schur/前代代数严格正确；
  - bf16 量化（HMMA 输入与中间量按 kernel 一样 cast bf16）：给出 max abs。
"""
import torch


def fwd_subst_inv8(Lblk: torch.Tensor) -> torch.Tensor:
    """单位下三角 8x8 的逆（逐行 forward substitution），fp32/fp64。"""
    X = torch.eye(8, dtype=Lblk.dtype, device=Lblk.device) + Lblk
    Xinv = torch.zeros_like(X)
    for i in range(8):
        Xinv[i, i] = 1.0
        for p in range(i):
            acc = torch.zeros((), dtype=Lblk.dtype, device=Lblk.device)
            for q in range(p, i):
                acc = acc + Lblk[i, q] * Xinv[q, p]
            Xinv[i, p] = -acc
    return Xinv


def block_inv_schur(L: torch.Tensor, quant_bf16: bool) -> torch.Tensor:
    A = L[0:8, 0:8]
    B = L[8:16, 8:16]
    C = L[8:16, 0:8]
    Ainv = fwd_subst_inv8(A)
    Binv = fwd_subst_inv8(B)

    P = torch.zeros(16, 16, dtype=L.dtype, device=L.device)
    P[0:8, 0:8] = Ainv
    P[8:16, 8:16] = Binv
    M = torch.zeros(16, 16, dtype=L.dtype, device=L.device)
    M[8:16, 0:8] = C

    if quant_bf16:
        P = P.to(torch.bfloat16).to(L.dtype)
        M = M.to(torch.bfloat16).to(L.dtype)
    dc = P @ M
    negdc = -dc
    if quant_bf16:
        negdc = negdc.to(torch.bfloat16).to(L.dtype)
    o = negdc @ P
    if quant_bf16:
        o = o.to(torch.bfloat16).to(L.dtype)
    return P + o


def run(dtype, quant_bf16, seed=0):
    torch.manual_seed(seed)
    Lt = torch.randn(16, 16, dtype=dtype)
    L = torch.tril(Lt, diagonal=-1)
    L = L * 0.3
    X = torch.eye(16, dtype=dtype) + L
    P_ref = torch.linalg.inv(X)
    P_kernel = block_inv_schur(L, quant_bf16)
    diff = (P_kernel - P_ref).abs()
    return diff.max().item(), P_ref.abs().max().item(), P_kernel


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    for seed in range(3):
        m64, s64, _ = run(torch.float64, quant_bf16=False, seed=seed)
        m32, s32, _ = run(torch.float32, quant_bf16=False, seed=seed)
        mb, sb, _ = run(torch.float32, quant_bf16=True, seed=seed)
        print(
            f"seed={seed}  |P|max={s64:.3e}  "
            f"max|Δ| fp64(纯公式)={m64:.3e}  "
            f"fp32(纯公式)={m32:.3e}  "
            f"fp32+bf16量化(HMMA 输入/中间量)={mb:.3e}"
        )

    # 边界：近共线 key 时 |L| 接近 1（旧 Neumann 级数会失稳的场景）
    torch.manual_seed(123)
    Lt = torch.randn(16, 16, dtype=torch.float32)
    L = torch.tril(Lt, diagonal=-1) * 0.9
    L = L / L.abs().sum(dim=1, keepdim=True).clamp_min(1e-6) * 0.9
    X = torch.eye(16) + L
    P_ref = torch.linalg.inv(X.double())
    P_k = block_inv_schur(L.double(), False)
    P_kb = block_inv_schur(L, True).double()
    print(
        f"\n近单位 L (|L|~大): max|Δ| fp64={ (P_k-P_ref).abs().max().item():.3e}  "
        f"bf16={ (P_kb-P_ref).abs().max().item():.3e}"
    )
