"""Flash Attention 系列配套参考实现（纯 PyTorch，教学用）。

对照博客《Flash Attention 精读》系列（content/posts/flash-attention-0*/）：
  1. naive_attention    —— 朴素实现，物化 N×N 矩阵（第 1 篇 §1）
  2. online_softmax     —— 流式 softmax 的递推验证（第 1 篇 §3）
  3. flash_attn_fwd     —— 分块前向 + LSE（第 2 篇）
  4. flash_attn_bwd     —— recompute 反向：Δ=rowsum(dO∘O)、dS=P∘(dP−Δ)（第 4 篇 §1）

运行: python ref_impl.py   （CPU 即可）
"""

import math

import torch

torch.manual_seed(0)


def naive_attention(q, k, v, causal=False):
    """第 1 篇 §1：三步朴素实现，S/P 物化在内存里。"""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q @ k.transpose(-2, -1)) * scale
    if causal:
        n = s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v


def online_softmax(x, block=16):
    """第 1 篇 §3：流式 softmax，逐块更新 (m, l)，与 torch.softmax 对拍。

    x: [N] 一行 logits。返回 softmax(x)。
    维护未归一化分子 acc：acc = α·acc + exp(chunk − m_new)，最后除 l。
    """
    m = torch.tensor(float("-inf"))
    l = torch.tensor(0.0)
    acc = torch.zeros_like(x)
    for i in range(0, x.shape[0], block):
        chunk = x[i : i + block]
        m_new = torch.maximum(m, chunk.max())
        alpha = torch.exp(m - m_new)  # 旧坐标系 → 新坐标系的迁移因子
        p = torch.exp(chunk - m_new)  # 未归一化分子
        acc = acc * alpha
        acc[i : i + chunk.shape[0]] = p  # α 乘不动当前块（它已在最新坐标系）
        l = l * alpha + p.sum()
        m = m_new
    return acc / l


def flash_attn_fwd(q, k, v, block_m=32, block_n=32, causal=False):
    """第 2 篇：分块前向。返回 (O, LSE)，LSE = m + log(ℓ)（自然对数域）。

    q, k, v: [N, d]。
    """
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    o = torch.zeros(n, d)
    lse = torch.zeros(n)
    rows_all = torch.arange(n)
    for i0 in range(0, n, block_m):  # 外层 Q（并行维）
        i1 = min(i0 + block_m, n)
        q_blk = q[i0:i1]
        rows = rows_all[i0:i1, None]
        m = torch.full((i1 - i0,), float("-inf"))
        l = torch.zeros(i1 - i0)
        acc = torch.zeros(i1 - i0, d)
        # 内层 KV（循环维）。causal 时块级跳过：只需到对角带为止（第 2 篇 §4）
        j_hi = i1 if causal else n
        for j0 in range(0, j_hi, block_n):
            j1 = min(j0 + block_n, n)
            s = (q_blk @ k[j0:j1].T) * scale
            if causal:
                cols = rows_all[j0:j1][None, :]
                s = s.masked_fill(cols > rows, -1.0e6)  # 哨兵 -1e6 而非 -inf（第 2 篇 §3.1）
            m_new = torch.maximum(m, s.max(dim=1).values)
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new[:, None])
            l = l * alpha + p.sum(dim=1)
            acc = acc * alpha[:, None] + p @ v[j0:j1]
            m = m_new
        o[i0:i1] = acc / l[:, None]  # 归一化推迟到循环外（第 2 篇 §5）
        lse[i0:i1] = m + torch.log(l)
    return o, lse


def flash_attn_bwd(q, k, v, o, lse, do, block=32, causal=False):
    """第 4 篇：recompute 反向。返回 (dq, dk, dv)。

    dV = Pᵀ dO; dS = P ∘ (dP − Δ); dQ = σ dS K; dK = σ dSᵀ Q; Δ_i = dO_i·O_i。
    P = exp(S − LSE) 一步恢复，不重跑 online softmax（第 4 篇 §1.5）。
    """
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    delta = (do * o).sum(dim=1)  # preprocess kernel 的全部工作（Δ 恒等式）
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    rows_all = torch.arange(n)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        q_blk, do_blk = q[i0:i1], do[i0:i1]
        delta_blk, lse_blk = delta[i0:i1], lse[i0:i1]
        rows = rows_all[i0:i1, None]
        j_hi = i1 if causal else n

        # —— dK/dV：沿 Q 块累加（生产实现里是"外层 KV、内层 Q"，教学版循环方向相反，数学等价）
        for j0 in range(0, j_hi, block):
            j1 = min(j0 + block, n)
            s = (q_blk @ k[j0:j1].T) * scale
            p = torch.exp(s - lse_blk[:, None])
            if causal:
                p = p.masked_fill(rows_all[j0:j1][None, :] > rows, 0.0)
            dv[j0:j1] += p.T @ do_blk
            dp = do_blk @ v[j0:j1].T
            ds = p * (dp - delta_blk[:, None])  # 核心公式：dS = P ∘ (dP − Δ)
            dk[j0:j1] += ds.T @ q_blk * scale

        # —— dQ：沿 KV 块累加
        for j0 in range(0, j_hi, block):
            j1 = min(j0 + block, n)
            s = (q_blk @ k[j0:j1].T) * scale
            p = torch.exp(s - lse_blk[:, None])
            if causal:
                p = p.masked_fill(rows_all[j0:j1][None, :] > rows, 0.0)
            dp = do_blk @ v[j0:j1].T
            ds = p * (dp - delta_blk[:, None])
            dq[i0:i1] += ds @ k[j0:j1] * scale
    return dq, dk, dv


def main():
    n, d = 128, 32
    q, k, v = (torch.randn(n, d) for _ in range(3))

    for causal in (False, True):
        # 1) online softmax 递推 vs 标准 softmax
        s = (q @ k.T) / math.sqrt(d)
        if causal:
            s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        assert torch.allclose(torch.softmax(s[0], dim=-1), online_softmax(s[0]), atol=1e-6)

        # 2) flash 前向 vs 朴素 attention
        o_ref = naive_attention(q, k, v, causal)
        o, lse = flash_attn_fwd(q, k, v, causal=causal)
        assert torch.allclose(o, o_ref, atol=1e-5), "fwd mismatch"

        # 3) flash 反向 vs autograd
        qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = naive_attention(qg, kg, vg, causal)
        do = torch.randn_like(out)
        out.backward(do)
        dq, dk, dv = flash_attn_bwd(q, k, v, o, lse, do, causal=causal)
        for name, a, b in (("dq", dq, qg.grad), ("dk", dk, kg.grad), ("dv", dv, vg.grad)):
            assert torch.allclose(a, b, atol=1e-4), f"bwd {name} mismatch"

        print(f"causal={causal}: online softmax / fwd / bwd all pass")


if __name__ == "__main__":
    main()
