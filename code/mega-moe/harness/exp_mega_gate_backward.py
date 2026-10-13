#!/usr/bin/env python3
"""P117-1: Mega Gate 路由（sqrt-softplus 打分 + 归一化权重）反向的手写推导 vs torch autograd。

DeepGEMM 的 Mega Gate（sm100_bf16_mega_gate.cuh / epilogue/sm100_mega_gate.cuh）只有前向。
本实验把它的**打分与权重语义**用可微 torch 重写，用 autograd 作为独立参考，对拍手写反向：

  前向 (每 token):
    z = x @ Wg                    # [T,E] 打分 logits
    s = sqrt(softplus(z))         # 打分函数 (epilogue:19-24)
    rank = s + bias (可选)         # 排名 (impls:397-398)
    idx = topk(rank, K)           # 硬选择: 索引当常数
    S = sum(s[idx]) + 1e-20
    w = s[idx] / S * scaling      # (epilogue:210-213)

  反向 (hand, 给定 dL/dw = dy):
    ds_i = scaling * (dy_i * S - sum_j dy_j s_j) / S^2      # 归一化 Jacobian
    dz   = ds_i * d sqrt(softplus(z))/dz  (仅选中专家, 其余 0)
    dWg  = x^T dz ;  dx = dz Wg^T

autograd 参考：idx 保持 detached（straight-through），于是 autograd 只经数值路径回传，
与手写推导的语义一致。
"""
import numpy as np
import torch

assert torch.cuda.is_available()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_printoptions(precision=6)


def sqrt_softplus(z):
    # 与实现一致的分段：z>20 用 z，否则 log1p(exp(z))，再开方
    softplus = torch.where(z > 20.0, z, torch.log1p(torch.exp(z)))
    return torch.sqrt(softplus)


def sqrt_softplus_grad(z):
    # 对「实现的分段函数」求导
    e = torch.exp(z)
    sp = torch.log1p(e)
    s = torch.sqrt(sp)
    sig = e / (1.0 + e)
    low = 0.5 * sig / s
    high = 0.5 / torch.sqrt(z)
    return torch.where(z > 20.0, high, low)


def run(T, H, E, K, scaling, use_bias, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(T, H, generator=g, device=dev, dtype=torch.float64)
    Wg = torch.randn(H, E, generator=g, device=dev, dtype=torch.float64) * 0.5
    dy = torch.randn(T, K, generator=g, device=dev, dtype=torch.float64)
    bias = (torch.randn(E, generator=g, device=dev, dtype=torch.float64) * 0.4 *
            torch.where(torch.arange(E, device=dev) < 16, 4.0, 1.0)) if use_bias else None

    # ---- 前向 + 选择（idx 由 topk 给出；手写与 autograd 共用同一 idx）----
    z = x @ Wg
    s = sqrt_softplus(z)
    rank = s + bias if bias is not None else s
    idx = torch.topk(rank, K, dim=1, largest=True, sorted=False).indices   # [T,K] 常数
    s_sel = torch.gather(s, 1, idx)
    S = s_sel.sum(1, keepdim=True) + 1e-20
    w = s_sel / S * scaling

    # ---- 手写反向 ----
    c = (dy * s_sel).sum(1, keepdim=True)                 # Σ_j dy_j s_j
    ds = scaling * (dy * S - c) / (S * S)                 # [T,K]
    z_sel = torch.gather(z, 1, idx)
    dz_sel = ds * sqrt_softplus_grad(z_sel)
    dz_hand = torch.zeros_like(z)
    dz_hand.scatter_(1, idx, dz_sel)
    dWg_hand = x.t() @ dz_hand
    dx_hand = dz_hand @ Wg.t()

    # ---- autograd 参考（可微前向，idx detached）----
    xa = x.clone().requires_grad_(True)
    Wa = Wg.clone().requires_grad_(True)
    za = xa @ Wa
    za.retain_grad()
    sa = sqrt_softplus(za)
    s_sel_a = torch.gather(sa, 1, idx)                    # idx 不含梯度 -> straight-through
    Sa = s_sel_a.sum(1, keepdim=True) + 1e-20
    wa = s_sel_a / Sa * scaling
    loss = (wa * dy).sum()
    loss.backward()
    dz_auto = za.grad

    def rel(a, b):
        return (a - b).norm().item() / (b.norm().item() + 1e-300)

    return dict(dz=rel(dz_hand, dz_auto),
                dWg=rel(dWg_hand, Wa.grad), dx=rel(dx_hand, xa.grad))


def main():
    print("Mega Gate 路由反向：手写推导 vs torch autograd (float64, CUDA)")
    print("(DeepGEMM 的 Mega Gate 为 SM100 专属且只有前向；本实验验证其打分/权重语义的梯度)\n")
    cfgs = [
        (64, 32, 64, 4, 1.5, False),
        (64, 32, 64, 4, 1.5, True),
        (256, 64, 128, 6, 1.5, False),
        (256, 64, 128, 6, 2.0, True),
        (512, 128, 256, 8, 1.5, True),
        (2048, 128, 256, 6, 1.0, False),
    ]
    for (T, H, E, K, sc, ub) in cfgs:
        r = run(T, H, E, K, sc, ub, seed=T * 7 + E)
        print(f"-- T={T:5d} H={H:4d} E={E:4d} K={K} scaling={sc} bias={int(ub)}")
        print(f"     dWg relerr = {r['dWg']:.4e}   dx relerr = {r['dx']:.4e}   dz relerr = {r['dz']:.4e}")

    # 额外：验证 dL/ds 的归一化 Jacobian 单步（外部 dy），与中心差分对拍
    print("\n-- 归一化 Jacobian 中心差分抽查 (T=8,E=32,K=5) --")
    T, E, K, sc = 8, 32, 5, 1.5
    g = torch.Generator(device=dev).manual_seed(999)
    s_sel = torch.rand(T, K, generator=g, device=dev, dtype=torch.float64) + 0.2
    dy = torch.randn(T, K, generator=g, device=dev, dtype=torch.float64)

    def loss_of(sv):
        S = sv.sum(1, keepdim=True) + 1e-20
        return ((sv / S * sc) * dy).sum(1)   # 每 token 的损失 [T]，FD 逐 token 对拍

    S = s_sel.sum(1, keepdim=True) + 1e-20
    c = (dy * s_sel).sum(1, keepdim=True)
    ds_hand = sc * (dy * S - c) / (S * S)
    h = 1e-7
    ds_fd = torch.zeros_like(s_sel)
    for i in range(K):
        sp = s_sel.clone(); sp[:, i] += h
        sm = s_sel.clone(); sm[:, i] -= h
        ds_fd[:, i] = (loss_of(sp) - loss_of(sm)) / (2 * h)
    print(f"   解析 vs 中心差分 max_abs_err = {(ds_hand - ds_fd).abs().max().item():.4e}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
