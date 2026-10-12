#!/usr/bin/env python3
"""P4-1: MoE 反向数学的手工推导 vs PyTorch autograd 对拍。

验证的是「数学」——linear1/linear2 的 dgrad/wgrad、SwiGLU 反向、router/topk
反向（含 topk 权重归一化的 softmax Jacobian）——而不是 GEMM kernel 的数值（那是 P4-4）。
用 float64 把误差压到浮点极限，从而证明推导正确。

前向用纯 torch 算子实现（可微），autograd 给参考梯度；手写反向从零推导并对比。
另附有限差分抽查。纯 CPU 可跑，无需 GPU。

同时打印 wgrad 的 grouped 形状（k_grouped：K=token 轴分组、M/N 固定）与 dgrad 的
grouped 形状（m_grouped：M=token 轴分组、N/K 固定），对应 DeepGEMM
`k_grouped_fp8_gemm_tn_contiguous` / `m_grouped_fp8_fp4_gemm_nt_contiguous`。
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DTYPE = torch.float64
T, H, I, E, K = 11, 16, 8, 5, 2  # tokens, hidden, intermediate, experts, topk


def silu(z):
    return z * torch.sigmoid(z)


def silu_grad(z):
    s = torch.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def forward(x, Wg, W1, W2):
    logits = x @ Wg                            # [T, E]
    p = F.softmax(logits, dim=-1)
    topk_w, topk_i = torch.topk(p, K, dim=-1)  # [T, K]
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)
    G = torch.einsum('th,eho->teo', x, W1)     # [T, E, 2I]
    Gg, Gu = G[..., :I], G[..., I:]
    A = silu(Gg) * Gu                          # [T, E, I]
    Y = torch.einsum('tei,eih->teh', A, W2)    # [T, E, H]
    sel = torch.zeros_like(p).scatter(1, topk_i, 1.0)
    w = torch.zeros_like(p).scatter(1, topk_i, topk_w)  # [T,E] dense topk weights
    z = torch.einsum('te,teh->th', w, Y)
    return z, (logits, p, topk_w, topk_i, sel, w, Gg, Gu, A, Y)


def manual_backward(dz, saved, x, Wg, W1, W2):
    logits, p, topk_w, topk_i, sel, w, Gg, Gu, A, Y = saved

    # combine backward:  dY_(t,j) = w_tj * dz_t
    dY = w[..., None] * dz[:, None, :]         # [T,E,H]

    # linear2: Y = A @ W2
    dW2 = torch.einsum('tei,teh->eih', A, dY)  # wgrad (K=T grouped)
    dA = torch.einsum('teh,eih->tei', dY, W2)  # dgrad

    # SwiGLU backward
    dGu = dA * silu(Gg)
    dGg = dA * Gu * silu_grad(Gg)
    dG = torch.cat([dGg, dGu], dim=-1)         # [T,E,2I]

    # linear1: G = x @ W1
    dW1 = torch.einsum('th,teo->eho', x, dG)   # wgrad (K=T grouped)
    dx_ffn = torch.einsum('teo,eho->th', dG, W1)  # dgrad

    # router / topk backward
    q = torch.einsum('teh,th->te', Y, dz)      # dL/d(topk_w_tj)
    P = (p * sel).sum(-1, keepdim=True)        # P_orig = sum of *softmax* p over selected
    Pmat = torch.zeros_like(p).scatter(1, topk_i, P.expand(T, K))
    qS = q * sel
    A_sum = (qS * p).sum(-1, keepdim=True).expand(T, E)
    dp = torch.where(sel > 0, (q * Pmat - A_sum) / (Pmat ** 2), torch.zeros_like(p))
    dlogits = p * (dp - (p * dp).sum(-1, keepdim=True))
    dWg = x.t() @ dlogits
    dx_router = dlogits @ Wg.t()

    return dx_ffn + dx_router, dWg, dW1, dW2


def rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-30)


def main():
    x, Wg, W1, W2 = (torch.randn(s, dtype=DTYPE) for s in
                     [(T, H), (H, E), (E, H, 2 * I), (E, I, H)])
    x = x.clone().requires_grad_(True)
    Wg = Wg.clone().requires_grad_(True)
    W1 = W1.clone().requires_grad_(True)
    W2 = W2.clone().requires_grad_(True)

    z, saved = forward(x, Wg, W1, W2)
    dz = torch.randn_like(z)
    ref = torch.autograd.grad(z, (x, Wg, W1, W2), dz)
    ana = manual_backward(dz, saved, x.detach(), Wg.detach(), W1.detach(), W2.detach())

    print('== manual backward vs autograd (float64) ==')
    print(f'{"grad":6s} {"shape":16s} {"relerr":>12s} {"max_abs":>12s}')
    for name, a, b in zip(['dx', 'dWg', 'dW1', 'dW2'], ana, ref):
        print(f'{name:6s} {str(tuple(a.shape)):16s} {rel(a, b):12.3e} {(a-b).abs().max().item():12.3e}')

    # Finite-difference spot check on the analytic grads (independent of autograd).
    def loss(xx, wg, w1, w2):
        return (forward(xx, wg, w1, w2)[0] * dz).sum()

    eps = 1e-6
    print('== central finite-difference spot check (3 entries each) ==')
    for name, t, gt in [('dx', x, ana[0]), ('dWg', Wg, ana[1]), ('dW1', W1, ana[2]), ('dW2', W2, ana[3])]:
        base = [x.detach().clone(), Wg.detach().clone(), W1.detach().clone(), W2.detach().clone()]
        pos = {'dx': 0, 'dWg': 1, 'dW1': 2, 'dW2': 3}[name]
        errs = []
        for flat in range(min(3, t.numel())):
            i = torch.unravel_index(torch.tensor(flat), t.shape)
            pp = [b.clone() for b in base]; pp[pos][i] += eps
            pm = [b.clone() for b in base]; pm[pos][i] -= eps
            num = (loss(*pp) - loss(*pm)).item() / (2 * eps)
            errs.append(abs(num - gt[i].item()))
        print(f'{name:6s} max_abs_fd_err={max(errs):.3e}')

    print('== grouped GEMM view (per expert) ==')
    _, topk_i = torch.topk(saved[1], K, -1)
    counts = torch.bincount(topk_i.flatten(), minlength=E).tolist()
    print(f'experts={E} tokens={T} topk={K} per-expert #tokens={counts} (sum={sum(counts)})')

    # Reproduce wgrad the k_grouped way: concatenate each expert's (token) rows into
    # one [sum_k, dim] matrix, then per-group D[e] = A_e^T @ B_e. Token rows of a
    # group are laid out contiguously and padded to k_alignment (gap rows are zero).
    sd = saved
    dY = sd[5][..., None] * dz[:, None, :]
    # rebuild dG from saved intermediates (same as manual_backward, kept local)
    logits, p, topk_w, topk_i, sel, w, Gg, Gu, A, Y = sd
    dA = torch.einsum('teh,eih->tei', dY, W2.detach())
    dG = torch.cat([dA * Gu * silu_grad(Gg), dA * silu(Gg)], dim=-1)

    k_align = 128

    def k_grouped_wgrad(X, G):
        # X: [T, m] or [T, E, m]; G: [T, n] or [T, E, n]; groups are token sets, K = token axis
        def slice_e(t, e):
            return t[:, e, :] if t.dim() == 3 else t

        sum_k = 0
        for c in counts:
            sum_k += ((c + k_align - 1) // k_align) * k_align
        m_dim = X.shape[-1]
        n_dim = G.shape[-1]
        Amat = torch.zeros((sum_k, m_dim), dtype=DTYPE)
        Bmat = torch.zeros((sum_k, n_dim), dtype=DTYPE)
        off = 0
        for e in range(E):
            rows = torch.nonzero(sel[:, e] > 0).flatten()
            Amat[off:off + len(rows)] = slice_e(X, e)[rows]
            Bmat[off:off + len(rows)] = slice_e(G, e)[rows]
            off += ((len(rows) + k_align - 1) // k_align) * k_align
        out = torch.empty((E, m_dim, n_dim), dtype=DTYPE)
        off = 0
        for e in range(E):
            n = ((counts[e] + k_align - 1) // k_align) * k_align
            out[e] = Amat[off:off + n].t() @ Bmat[off:off + n]
            off += n
        return out, sum_k

    dW1g, sk1 = k_grouped_wgrad(x.detach(), dG)
    dW2g, sk2 = k_grouped_wgrad(A.detach(), dY)
    print(f'wgrad linear1 k_grouped: dW1[e]=X_e^T@dG_e  K={sk1} M={H} N={2*I} '
          f'relerr_vs_autograd={rel(dW1g, ref[2]):.3e}')
    print(f'wgrad linear2 k_grouped: dW2[e]=A_e^T@dY_e  K={sk2} M={I} N={H}   '
          f'relerr_vs_autograd={rel(dW2g, ref[3]):.3e}')
    print(f'dgrad linear1 m_grouped: dX_e=dG_e@W1_e^T  M=per-expert tokens N={H} K={2*I}')
    print(f'dgrad linear2 m_grouped: dA_e=dY_e@W2_e^T  M=per-expert tokens N={I} K={H}')
    print(f'k_alignment={k_align}: linear1 sum_k={sk1} (real tokens={sum(counts)}), '
          f'linear2 sum_k={sk2} (real tokens={sum(counts)})')


if __name__ == '__main__':
    main()
