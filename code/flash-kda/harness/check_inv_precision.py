#!/usr/bin/env python
"""P2-2 自测：K1 块内求逆的两个「精度取舍」量化。

对 ``csrc/smxx/utils.cuh:213-369`` 的求逆实现，比较不同精度路径：
  A. 纯公式 fp32 forward substitution + fp32 Schur 合并（无量化上界）
  B. kernel 实际路径：**seed L 保持 fp32**，8x8 前代 fp32，仅 HMMA 输入 P/M 量化 bf16
  C. 若把 seed 也量化 bf16（反面：说明「seed 不量化」值多少）
  D. 旧实现的 fp16 Neumann 级数 (I-L)(I+L^2)(I+L^4)(I+L^8)
     （对 nilpotent L 代数上精确；高相关 key 时中间量 L^k 放大，fp16 灾难相消）

golden = ``torch.linalg.inv(I+L)``（fp64）。
"""
import torch


def fwd_subst_inv8(Lblk):
    Xinv = torch.zeros(8, 8, dtype=Lblk.dtype, device=Lblk.device)
    for i in range(8):
        Xinv[i, i] = 1.0
        for p in range(i):
            acc = torch.zeros((), dtype=Lblk.dtype, device=Lblk.device)
            for q in range(p, i):
                acc = acc + Lblk[i, q] * Xinv[q, p]
            Xinv[i, p] = -acc
    return Xinv


def schur_inv(L, quant_seed=False, quant_merge=True):
    Lw = L.to(torch.float32)
    if quant_seed:
        Lw = Lw.to(torch.bfloat16).to(torch.float32)
    A, B, C = Lw[0:8, 0:8], Lw[8:16, 8:16], Lw[8:16, 0:8]
    Ainv, Binv = fwd_subst_inv8(A), fwd_subst_inv8(B)
    P = torch.zeros(16, 16)
    P[0:8, 0:8] = Ainv
    P[8:16, 8:16] = Binv
    M = torch.zeros(16, 16)
    M[8:16, 0:8] = C
    if quant_merge:
        P = P.to(torch.bfloat16).to(torch.float32)
        M = M.to(torch.bfloat16).to(torch.float32)
    dc = P @ M
    negdc = (-dc).to(torch.bfloat16).to(torch.float32) if quant_merge else -dc
    o = negdc @ P
    if quant_merge:
        o = o.to(torch.bfloat16).to(torch.float32)
    return P + o


def neumann_inv(L, dtype):
    I = torch.eye(16, dtype=dtype)
    Ld = L.to(dtype)
    L2 = Ld @ Ld
    L4 = L2 @ L2
    L8 = L4 @ L4
    return (I - Ld) @ (I + L2) @ (I + L4) @ (I + L8)


def make_L(seed, scale, near_collinear=False):
    g = torch.Generator().manual_seed(seed)
    Lt = torch.randn(16, 16, generator=g)
    L = torch.tril(Lt, diagonal=-1) * scale
    if near_collinear:
        # 行和压到 ~scale，使 |L| 大（近共线 key）
        L = L / L.abs().sum(dim=1, keepdim=True).clamp_min(1e-6) * scale
    return L


def report(tag, L, P_ref):
    Pa = schur_inv(L, quant_seed=False, quant_merge=False)
    Pb = schur_inv(L, quant_seed=False, quant_merge=True)
    Pc = schur_inv(L, quant_seed=True, quant_merge=True)
    Pd = neumann_inv(L, torch.float16).to(torch.float64)
    Pe = neumann_inv(L, torch.bfloat16).to(torch.float64)
    d = lambda P: (P.double() - P_ref).abs().max().item()
    print(
        f"{tag:28s} |X^-1|max={P_ref.abs().max().item():8.3f}  "
        f"A(fp32)={d(Pa):.2e}  B(kernel,bf16 merge)={d(Pb):.2e}  "
        f"C(bf16 seed)={d(Pc):.2e}  D(neumann fp16)={d(Pd):.2e}  D(neumann bf16)={d(Pe):.2e}"
    )


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    idx = torch.arange(16)
    eye = torch.eye(16, dtype=torch.float64)
    for seed in range(3):
        L = make_L(seed, 0.3)
        report(f"seed={seed} rand |L|~0.3", L, torch.linalg.inv(eye + L.double()))
    print()
    for seed in range(3):
        L = make_L(seed, 0.9, near_collinear=True)
        report(f"seed={seed} 近共线 |L|~0.9", L, torch.linalg.inv(eye + L.double()))
    print()
    # 极端：所有 key 完全共线 → L 的全部严格下三角元素相同 = c。
    # L^k 的元素 = C(i-j-1,k-1) * c^k，最大 C(14,7)*c^7；c=0.9 时 ~1.7e3。
    for c in (0.5, 0.7, 0.9):
        L = torch.tril(torch.ones(16, 16, dtype=torch.float32), -1) * c
        report(f"完全共线 c={c}", L, torch.linalg.inv(eye + L.double()))
