"""Flash Attention 反向变体参考实现（纯 PyTorch，教学用）。

对照博客《Flash Attention 精读（七）》：
  §1.2  Δ 恒等式      Δ_i = rowsum(dO ∘ O)（即使有 dropout 也成立）
  §1.4  GQA 反向      dK/dV 沿共享的 Q head 求和
  §1.5  dropout 反向  1/keep 折进 dP，mask 用乘法（与 FA2 把 mask 藏进符号位等价）
  §1.6  softcap 反向  dS0 = dS_capped ∘ (1 − (s_capped / c)²)

所有变体都用 autograd 对拍。运行: python bwd_variants_ref.py（CPU 即可）
"""

import math

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def fwd_ref(q, k, v, causal=False, softcap=None, keep_mask=None):
    """朴素前向（物化 N×N），返回 O 与 LSE（LSE 是 softcap 之后、dropout 之前的 logsumexp）。"""
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    s0 = (q @ k.T) * scale
    s = softcap * torch.tanh(s0 / softcap) if softcap is not None else s0
    if causal:
        # 掩码必须加在 softcap 之后：tanh(-inf) = -1 是有限值，会破坏掩码语义
        s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    p = torch.softmax(s, dim=-1)
    lse = torch.logsumexp(s, dim=-1)
    p_out = p if keep_mask is None else p * keep_mask / keep_mask.float().mean()
    return p_out @ v, lse, s


def bwd_ref(q, k, v, o, lse, do, causal=False, softcap=None, keep_mask=None):
    """第 7 篇 §1 的反向：Δ 恒等式 + softmax 雅可比 + softcap/dropout 链式。

    返回 (dq, dk, dv)，与 fwd_ref 的输入一一对应。
    """
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    delta = (do * o).sum(dim=-1)                      # §1.2 Δ_i = dO_i · O_i
    s0 = (q @ k.T) * scale
    s = softcap * torch.tanh(s0 / softcap) if softcap is not None else s0
    causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    if causal:
        s = s.masked_fill(causal_mask, float("-inf"))
    p = torch.exp(s - lse[:, None])                   # §1.3 LSE 一步恢复 P
    if causal:
        p = p.masked_fill(causal_mask, 0.0)

    dp = do @ v.T                                     # dP = dO Vᵀ
    if keep_mask is not None:
        dp = dp * keep_mask / keep_mask.float().mean()  # §1.5 1/keep 折进 dP
    ds_capped = p * (dp - delta[:, None])             # §1.2 dS = P ∘ (dP − Δ)
    if softcap is not None:
        ds0 = ds_capped * (1.0 - (s / softcap) ** 2)  # §1.6 tanh 导数
        ds0 = torch.nan_to_num(ds0, nan=0.0)          # 掩码位 s=-inf 会产出 nan，置 0
    else:
        ds0 = ds_capped
    if causal:
        ds0 = ds0.masked_fill(causal_mask, 0.0)

    dq = (ds0 @ k) * scale
    dk = (ds0.T @ q) * scale
    p_out = p if keep_mask is None else p * keep_mask / keep_mask.float().mean()
    dv = p_out.T @ do
    return dq, dk, dv


def gqa_fwd_bwd(q, k, v, do, causal=False):
    """GQA：q 有 Hq 个 head，k/v 有 Hkv 个 head，g = Hq/Hkv。

    返回 (o, lse, dq, dk, dv)。dk/dv 对每个 kv head 累加其 g 个 q head 的梯度。
    """
    hq, n, d = q.shape
    hkv = k.shape[0]
    g = hq // hkv
    o = torch.zeros_like(q)
    lse = torch.zeros(hq, n)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for h in range(hq):
        kv = h // g  # 该 q head 共享的 kv head
        oh, lh, _ = fwd_ref(q[h], k[kv], v[kv], causal)
        o[h], lse[h] = oh, lh
        dqh, dkh, dvh = bwd_ref(q[h], k[kv], v[kv], oh, lh, do[h], causal)
        dq[h] = dqh
        dk[kv] += dkh  # §1.4 跨 q head 求和（生产实现里是 atomicAdd / TMA reduce）
        dv[kv] += dvh
    return o, lse, dq, dk, dv


def main():
    n, d = 64, 16

    # ---- 1) Δ 恒等式 + softcap / dropout 变体 vs autograd ----
    q = torch.randn(n, d, requires_grad=True)
    k = torch.randn(n, d, requires_grad=True)
    v = torch.randn(n, d, requires_grad=True)
    keep = (torch.rand(n, n) > 0.3).double()  # 固定 dropout mask（keep 概率 0.7）

    for causal in (False, True):
        for softcap in (None, 2.0):
            for keep_mask in (None, keep):
                o, lse, s = fwd_ref(q, k, v, causal, softcap, keep_mask)
                do = torch.randn_like(o)
                o.backward(do, retain_graph=True)
                dq, dk, dv = bwd_ref(
                    q.detach(), k.detach(), v.detach(), o.detach(), lse, do, causal, softcap, keep_mask
                )
                for name, a, b in (("dq", dq, q.grad), ("dk", dk, k.grad), ("dv", dv, v.grad)):
                    assert torch.allclose(a, b, atol=1e-9), f"{name} mismatch causal={causal} softcap={softcap} drop={keep_mask is not None}"
                q.grad = k.grad = v.grad = None
        print(f"[Δ/softcap/dropout] causal={causal}: pass")

    # ---- 2) GQA：dk/dv 沿 q head 求和 vs autograd ----
    hq, hkv = 8, 2
    qg = torch.randn(hq, n, d, requires_grad=True)
    kg = torch.randn(hkv, n, d, requires_grad=True)
    vg = torch.randn(hkv, n, d, requires_grad=True)
    g = hq // hkv

    # autograd 参考：每个 q head 独立 attention 后求和
    o = torch.zeros_like(qg)
    for h in range(hq):
        oh, _, _ = fwd_ref(qg[h], kg[h // g], vg[h // g], False)
        o[h] = oh
    do = torch.randn_like(o)
    o.backward(do)

    dqc, dkc, dvc = gqa_fwd_bwd(qg.detach(), kg.detach(), vg.detach(), do)[2:]
    for name, a, b in (("dq", dqc, qg.grad), ("dk", dkc, kg.grad), ("dv", dvc, vg.grad)):
        assert torch.allclose(a, b, atol=1e-9), f"GQA {name} mismatch"
    print(f"[GQA] Hq={hq} Hkv={hkv}: pass")

    print("all backward variants pass")


if __name__ == "__main__":
    main()
