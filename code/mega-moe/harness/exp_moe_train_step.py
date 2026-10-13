"""P159-1: MoE **一个训练步（前向 + 反向）** 的 H100 分段成本账 + 共享专家反向的同图闭环。

前面各篇把前向（P89-1 路由、P158-1 路由+共享）与反向（P4-4 路由四 GEMM、P135/P140 wgrad）
分别讲透，但：
  * 反向图里**从来没有共享专家**（P4-4 的图只有路由分支；共享专家的反向整条缺席）；
  * **从来没有把前向和反向放进同一次计时**，量出「一个训练步里前向 vs 反向、GEMM vs 非 GEMM」的真实占比。

本实验补这两格，全部用 H100 上可跑的真实 SM90 FP8 kernel（DeepGEMM 2.8.1）：

  A. 共享专家反向的正确性闭环（**新几何**）：
     共享专家是无专家维的稠密 FFN：`G=x@W1s[:S]^T, U=x@W1s[S:]^T, a=silu(G)*U, y=a@W2s^T`。
     它的反向天然就是**单组 grouped GEMM**——wgrad 用 `k_grouped_*_nt_contiguous`（K=token 分组，
     one group），dgrad 用 `m_grouped_*_nt_contiguous`（M=token 分组，one group）。与路由专家**同一对
     kernel**，只是组数 E→1。逐项 vs 反量化参考（隔离 kernel+布局）与 vs fp32 autograd（含量化）。

  B. DeepSeek 量级（T,H,I,E,topk,Es）一个训练步的分段账：
     前向（路由 L1/L2 + 共享 L1/L2 + 激活量化 + SwiGLU + combine + router）
     反向（combine 反向 + 路由 dgrad/wgrad ×2 + 共享 dgrad/wgrad ×2 + SwiGLU 反向 + 反向激活量化
           + router 反向 + scatter-add），各自独立计时并汇总。

约束：Mega MoE 主 kernel 是 SM100 专属、本机跑不了；本实验只用其**架构无关的数学与量化语义**，
主 kernel 的 UMMA/tcgen05/NVLink 部分**仅分析**。目标卡代号敏感：全文用「目标卡」指代。

输出重定向到 exp_moe_train_step.out.txt。
"""
import os
import sys
import time
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from exp_mega_moe_forward_emul import (  # noqa: E402
    deep_gemm, get_arch_major, GRAN_K, align,
    per_token_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout,
    rel, cos, silu, build_inputs, router, group_layout, gather_acts,
    quant_act, quant_weight, run_grouped_fp8, kernel_pipeline, combine, _bench,
)
from exp_mega_moe_forward_shared import (  # noqa: E402
    build_shared, quant_weight_2d, run_dense_fp8, shared_pipeline, ref_shared,
    dec_act, dec_weight,
)
from exp_moe_backward_kernels import (  # noqa: E402
    run_k_grouped, run_m_grouped, m_grouped_slices, silu_grad, quant_k_operand,
)


# ===================================================================================== A
def shared_backward(dYs, x, W1s, W2s, S, ma):
    """共享专家反向，全部走真实 SM90 FP8 grouped kernel（单组）。

    形状角色（与路由专家逐项对应）：
      da   = dYs @ W2s        [T,H]@[H,S]  → dgrad（M=token 分组，one group，K=H）
      dW2s = dYs^T @ a        [T,H]^T[T,S] → wgrad（K=token 分组，one group，M=H,N=S）
      dW1s = x^T @ dGs        [T,H]^T[T,2S]→ wgrad（K=token 分组，one group，M=H,N=2S）
      dxs  = dGs @ W1s        [T,2S]@[2S,H]→ dgrad（M=token 分组，one group，K=2S）
    """
    T, H = x.shape
    # forward (fp32) to get G/U/a for the backward chain rule
    G1s = x @ W1s[:S].t()
    G2s = x @ W1s[S:].t()
    a = silu(G1s) * G2s

    # da = dYs @ W2s : m_grouped nt, B=[G,N,K]=[1,S,H]
    B2 = W2s.t().contiguous().unsqueeze(0)          # [1, S, H]
    da, da_ref, sl = run_m_grouped([dYs], B2, ma)
    da = da[:T].float()
    da_ref = da_ref[:T]

    dG1 = da * silu_grad(G1s) * G2s
    dG2 = da * silu(G1s)
    dGs = torch.cat([dG1, dG2], dim=-1)             # [T, 2S]

    # dW2s = dYs^T @ a : k_grouped nt, M=H, N=S, K=T
    dW2s, dW2s_ref, info2 = run_k_grouped([dYs], [a], GRAN_K)      # [1,H,S]
    # dW1s = x^T @ dGs : k_grouped nt, M=H, N=2S, K=T
    dW1s, dW1s_ref, info1 = run_k_grouped([x], [dGs], GRAN_K)      # [1,H,2S]

    # dxs = dGs @ W1s : m_grouped nt, B=[1,H,2S]
    B1 = W1s.t().contiguous().unsqueeze(0)          # [1, H, 2S]
    dxs, dxs_ref, sl1 = run_m_grouped([dGs], B1, ma)
    dxs = dxs[:T].float()
    dxs_ref = dxs_ref[:T]

    return dict(da=da, dW2s=dW2s[0], dW1s=dW1s[0], dxs=dxs,
                da_ref=da_ref, dW2s_ref=dW2s_ref[0], dW1s_ref=dW1s_ref[0], dxs_ref=dxs_ref)


def ref_shared_backward_autograd(x, W1s, W2s, S, dYs):
    """fp32 autograd 参考：dL/dy = dYs。"""
    xx = x.detach().clone().requires_grad_(True)
    w1 = W1s.detach().clone().requires_grad_(True)
    w2 = W2s.detach().clone().requires_grad_(True)
    G = xx @ w1[:S].t()
    U = xx @ w1[S:].t()
    A = silu(G) * U
    y = A @ w2.t()
    loss = (y * dYs).sum()
    gx, g1, g2 = torch.autograd.grad(loss, (xx, w1, w2))
    return gx, g1, g2


def part_a(device):
    print('## A. 共享专家反向：单组 grouped GEMM 与 autograd/fp64 对拍')
    print('   共享专家无专家维；反向等于「E=1」的同一对 k_grouped(wgrad)/m_grouped(dgrad) kernel。')
    print('   S = Es·I（共享中间维）。wgrad 形状角色 M=H/N=2S/K=token；dgrad M=token/N/K 固定。')
    print('=' * 100)
    ma = get_mk_alignment_for_contiguous_layout()
    print(f'contiguous M-alignment={ma}, K-alignment={GRAN_K}')
    cases = [(64, 512, 256, 1), (128, 1024, 512, 2), (256, 1024, 512, 1)]
    for (T, H, I, Es) in cases:
        S = Es * I
        x, _, _, _ = build_inputs(T, H, I, 8, 2, device)
        W1s, W2s = build_shared(T, S, H, device)
        dYs = torch.randn(T, H, device=device, dtype=torch.float32)

        kb = shared_backward(dYs, x, W1s, W2s, S, ma)
        gx, g1, g2 = ref_shared_backward_autograd(x, W1s, W2s, S, dYs)

        # kernel+布局（vs 反量化参考，隔离 FP8 量化）：wgrad 沿 M×N、dgrad 沿 m×N
        dW2s_k = rel(kb['dW2s'], kb['dW2s_ref'])
        dW1s_k = rel(kb['dW1s'], kb['dW1s_ref'])
        da_k = rel(kb['da'], kb['da_ref'])
        dxs_k = rel(kb['dxs'], kb['dxs_ref'])
        # 含量化（vs autograd fp32）；da 不是参数梯度，用 fp32 权威 dYs @ W2s
        dW2s_a = rel(kb['dW2s'], g2)
        dW1s_a = rel(kb['dW1s'].t(), g1)
        dxs_a = rel(kb['dxs'], gx)
        da_a = rel(kb['da'], dYs @ W2s)
        print(f'T={T:4d} H={H:5d} I={I:4d} Es={Es} S={S:5d}')
        print(f'   vs 反量化参考(kernel+布局):  dW2s={dW2s_k:.3e} dW1s={dW1s_k:.3e} '
              f'da={da_k:.3e} dxs={dxs_k:.3e}')
        print(f'   vs fp32 autograd(含量化):   dW2s={dW2s_a:.3e} dW1s={dW1s_a:.3e} '
              f'da={da_a:.3e} dxs={dxs_a:.3e}')
        print(f'   cos(vs autograd):           dW2s={cos(kb["dW2s"], g2):.6f} '
              f'dW1s={cos(kb["dW1s"].t(), g1):.6f} dxs={cos(kb["dxs"], gx):.6f}')
    print()


# ===================================================================================== B
def _tlog(name, t_us, flops, tag, rows):
    tf = flops / (t_us * 1e-6) / 1e12 if flops > 0 else 0.0
    rows.append((name, t_us, tf, tag))
    print(f'   {tag:4s} {name:34s} {t_us:9.0f} us   {tf:5.0f} TFLOPS')


def ledger(T, H, I, E, KTOP, Es, device, iters=30):
    S = Es * I
    ma = get_mk_alignment_for_contiguous_layout()
    ka = GRAN_K
    counts = [T * KTOP // E] * E
    host_ks = [align(k, ka) for k in counts]
    total_k = sum(host_ks)
    m_total = sum(align(k, ma) for k in counts)
    print(f'  配置 T={T} H={H} I={I} E={E} topk={KTOP} Es={Es} S={S}')
    print(f'  路由: m_total={m_total} (sum_k K-major={total_k});  共享: T={T}, S={S}')
    rows = []

    # ---------------------------------------------------------------- 前向
    # 路由 L1: m_grouped A=[m_total,H] B=[E,2I,H]
    A1 = torch.randn(m_total, H, device=device, dtype=torch.float32)
    a1q = per_token_cast_to_fp8(A1, use_ue8m0=False, gran_k=ka)
    W1r = torch.randn(E, 2 * I, H, device=device, dtype=torch.float32)
    w1r = torch.empty(E, 2 * I, H, dtype=torch.float8_e4m3fn, device=device)
    w1rs = torch.empty(E, align(2 * I, ka) // ka, align(H, ka) // ka, device=device, dtype=torch.float32)
    for e in range(E):
        w1r[e], w1rs[e] = per_block_cast_to_fp8(W1r[e], use_ue8m0=False, gran_k=ka)
    layout = torch.full((m_total,), 0, device=device, dtype=torch.int32)
    d1r = torch.empty(m_total, 2 * I, device=device, dtype=torch.bfloat16)
    t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a1q, (w1r, w1rs), d1r, layout, disable_ue8m0_cast=True), iters)
    _tlog('fwd routed L1 (m_grouped)', t * 1e6, 2 * m_total * 2 * I * H, 'FWD', rows)

    # 路由 L2: m_grouped A=[m_total,I] B=[E,H,I]
    A2 = torch.randn(m_total, I, device=device, dtype=torch.float32)
    a2q = per_token_cast_to_fp8(A2, use_ue8m0=False, gran_k=ka)
    W2r = torch.randn(E, H, I, device=device, dtype=torch.float32)
    w2r = torch.empty(E, H, I, dtype=torch.float8_e4m3fn, device=device)
    w2rs = torch.empty(E, align(H, ka) // ka, align(I, ka) // ka, device=device, dtype=torch.float32)
    for e in range(E):
        w2r[e], w2rs[e] = per_block_cast_to_fp8(W2r[e], use_ue8m0=False, gran_k=ka)
    d2r = torch.empty(m_total, H, device=device, dtype=torch.bfloat16)
    t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a2q, (w2r, w2rs), d2r, layout, disable_ue8m0_cast=True), iters)
    _tlog('fwd routed L2 (m_grouped)', t * 1e6, 2 * m_total * H * I, 'FWD', rows)

    # 共享 L1/L2: dense fp8
    W1s, W2s = build_shared(T, S, H, device)
    xs = torch.randn(T, H, device=device, dtype=torch.float32)
    xq = per_token_cast_to_fp8(xs, use_ue8m0=False, gran_k=ka)
    w1s_q = quant_weight_2d(W1s)
    d1s = torch.empty(T, 2 * S, device=device, dtype=torch.bfloat16)
    t = _bench(lambda: deep_gemm.fp8_gemm_nt(xq, w1s_q, d1s, disable_ue8m0_cast=True), iters)
    _tlog('fwd shared L1 (dense fp8)', t * 1e6, 2 * T * 2 * S * H, 'FWD', rows)

    A2s = silu(d1s[:, :S].float()) * d1s[:, S:].float()
    a2sq = per_token_cast_to_fp8(A2s, use_ue8m0=False, gran_k=ka)
    w2s_q = quant_weight_2d(W2s)
    d2s = torch.empty(T, H, device=device, dtype=torch.bfloat16)
    t = _bench(lambda: deep_gemm.fp8_gemm_nt(a2sq, w2s_q, d2s, disable_ue8m0_cast=True), iters)
    _tlog('fwd shared L2 (dense fp8)', t * 1e6, 2 * T * H * S, 'FWD', rows)

    # 激活量化（4 次 per-token cast：路由 A1/A2 + 共享 A1/A2）
    t = _bench(lambda: (per_token_cast_to_fp8(A1, use_ue8m0=False, gran_k=ka),
                        per_token_cast_to_fp8(A2, use_ue8m0=False, gran_k=ka),
                        per_token_cast_to_fp8(xs, use_ue8m0=False, gran_k=ka),
                        per_token_cast_to_fp8(A2s, use_ue8m0=False, gran_k=ka)), iters)
    _tlog('fwd activation quant x4', t * 1e6, 0, 'FWD', rows)

    # SwiGLU 前向（路由 m_total×I + 共享 T×S）
    g = torch.randn(m_total, I, device=device); u = torch.randn(m_total, I, device=device)
    gs = torch.randn(T, S, device=device); us = torch.randn(T, S, device=device)
    t = _bench(lambda: (silu(g) * u, silu(gs) * us), iters)
    _tlog('fwd SwiGLU (elem)', t * 1e6, 0, 'FWD', rows)

    # combine + router + scatter（torch 逐元素/小算子）
    d2 = torch.randn(m_total, H, device=device)
    idx = torch.randint(0, T, (m_total,), device=device)
    out = torch.zeros(T, H, device=device)
    t = _bench(lambda: torch.index_add(out, 0, idx, d2), iters)
    _tlog('fwd combine (index_add)', t * 1e6, 0, 'FWD', rows)

    # ---------------------------------------------------------------- 反向
    # 路由 dL2: dA = dY @ W2^T  (m_grouped A=[m_total,H] B=[E,I,H])
    dY = torch.randn(m_total, H, device=device, dtype=torch.float32)
    dyq = per_token_cast_to_fp8(dY, use_ue8m0=False, gran_k=ka)
    B2b = torch.randn(E, I, H, device=device, dtype=torch.float32)
    b2b = torch.empty(E, I, H, dtype=torch.float8_e4m3fn, device=device)
    b2bs = torch.empty(E, align(I, ka) // ka, align(H, ka) // ka, device=device, dtype=torch.float32)
    for e in range(E):
        b2b[e], b2bs[e] = per_block_cast_to_fp8(B2b[e], use_ue8m0=False, gran_k=ka)
    dA = torch.empty(m_total, I, device=device, dtype=torch.bfloat16)
    t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(dyq, (b2b, b2bs), dA, layout, disable_ue8m0_cast=True), iters)
    _tlog('bwd routed dgrad L2 (m_grouped)', t * 1e6, 2 * m_total * I * H, 'BWD', rows)

    # 路由 wgrad L2: dW2 = A^T dY (k_grouped M=I,N=H,K=sum_k) —— 用预量化
    Ak = torch.randn(total_k, I, device=device, dtype=torch.float32)
    Bk = torch.randn(total_k, H, device=device, dtype=torch.float32)
    ak = quant_k_operand(Ak, host_ks); bk = quant_k_operand(Bk, host_ks)
    ksl = torch.tensor(host_ks, device=device, dtype=torch.int32)
    dW2 = torch.zeros(E, I, H, device=device, dtype=torch.float32)
    t = _bench(lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (ak[0], ak[1]), (bk[0], bk[1]), dW2, host_ks, ksl, dW2, recipe=(1, 1, ka)), iters)
    _tlog('bwd routed wgrad L2 (k_grouped)', t * 1e6, 2 * total_k * I * H, 'BWD', rows)

    # 路由 dL1: dX = dG @ W1^T (m_grouped A=[m_total,2I] B=[E,H,2I])
    dG = torch.randn(m_total, 2 * I, device=device, dtype=torch.float32)
    dgq = per_token_cast_to_fp8(dG, use_ue8m0=False, gran_k=ka)
    B1b = torch.randn(E, H, 2 * I, device=device, dtype=torch.float32)
    b1b = torch.empty(E, H, 2 * I, dtype=torch.float8_e4m3fn, device=device)
    b1bs = torch.empty(E, align(H, ka) // ka, align(2 * I, ka) // ka, device=device, dtype=torch.float32)
    for e in range(E):
        b1b[e], b1bs[e] = per_block_cast_to_fp8(B1b[e], use_ue8m0=False, gran_k=ka)
    dX = torch.empty(m_total, H, device=device, dtype=torch.bfloat16)
    t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(dgq, (b1b, b1bs), dX, layout, disable_ue8m0_cast=True), iters)
    _tlog('bwd routed dgrad L1 (m_grouped)', t * 1e6, 2 * m_total * H * 2 * I, 'BWD', rows)

    # 路由 wgrad L1: dW1 = x^T dG (k_grouped M=H,N=2I)
    Ak1 = torch.randn(total_k, H, device=device, dtype=torch.float32)
    Bk1 = torch.randn(total_k, 2 * I, device=device, dtype=torch.float32)
    ak1 = quant_k_operand(Ak1, host_ks); bk1 = quant_k_operand(Bk1, host_ks)
    dW1 = torch.zeros(E, H, 2 * I, device=device, dtype=torch.float32)
    t = _bench(lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (ak1[0], ak1[1]), (bk1[0], bk1[1]), dW1, host_ks, ksl, dW1, recipe=(1, 1, ka)), iters)
    _tlog('bwd routed wgrad L1 (k_grouped)', t * 1e6, 2 * total_k * H * 2 * I, 'BWD', rows)

    # 共享反向：da (m_grouped A=[T,H] B=[1,S,H])
    dYs = torch.randn(T, H, device=device, dtype=torch.float32)
    dysq = per_token_cast_to_fp8(dYs, use_ue8m0=False, gran_k=ka)
    Bs2 = W2s.t().contiguous().unsqueeze(0)
    bs2 = torch.empty(1, S, H, dtype=torch.float8_e4m3fn, device=device)
    bs2s = torch.empty(1, align(S, ka) // ka, align(H, ka) // ka, device=device, dtype=torch.float32)
    bs2[0], bs2s[0] = per_block_cast_to_fp8(Bs2[0], use_ue8m0=False, gran_k=ka)
    dAs = torch.empty(T, S, device=device, dtype=torch.bfloat16)
    layout_s = torch.zeros(T, device=device, dtype=torch.int32)
    t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(dysq, (bs2, bs2s), dAs, layout_s, disable_ue8m0_cast=True), iters)
    _tlog('bwd shared dgrad L2 (m_grouped)', t * 1e6, 2 * T * S * H, 'BWD', rows)

    # 共享 wgrad L2: dW2s = dYs^T a (k_grouped M=H,N=S,K=T)
    host_ks_s = [align(T, ka)]
    At = torch.randn(host_ks_s[0], H, device=device, dtype=torch.float32)
    Bt = torch.randn(host_ks_s[0], S, device=device, dtype=torch.float32)
    aks = quant_k_operand(At, host_ks_s); bks = quant_k_operand(Bt, host_ks_s)
    ksl_s = torch.tensor(host_ks_s, device=device, dtype=torch.int32)
    dW2s = torch.zeros(1, H, S, device=device, dtype=torch.float32)
    t = _bench(lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (aks[0], aks[1]), (bks[0], bks[1]), dW2s, host_ks_s, ksl_s, dW2s, recipe=(1, 1, ka)), iters)
    _tlog('bwd shared wgrad L2 (k_grouped)', t * 1e6, 2 * host_ks_s[0] * H * S, 'BWD', rows)

    # 共享 dL1: dxs = dGs @ W1s (m_grouped A=[T,2S] B=[1,H,2S])
    dGs = torch.randn(T, 2 * S, device=device, dtype=torch.float32)
    dgsq = per_token_cast_to_fp8(dGs, use_ue8m0=False, gran_k=ka)
    Bs1 = W1s.t().contiguous().unsqueeze(0)
    bs1 = torch.empty(1, H, 2 * S, dtype=torch.float8_e4m3fn, device=device)
    bs1s = torch.empty(1, align(H, ka) // ka, align(2 * S, ka) // ka, device=device, dtype=torch.float32)
    bs1[0], bs1s[0] = per_block_cast_to_fp8(Bs1[0], use_ue8m0=False, gran_k=ka)
    dXs = torch.empty(T, H, device=device, dtype=torch.bfloat16)
    t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(dgsq, (bs1, bs1s), dXs, layout_s, disable_ue8m0_cast=True), iters)
    _tlog('bwd shared dgrad L1 (m_grouped)', t * 1e6, 2 * T * H * 2 * S, 'BWD', rows)

    # 共享 wgrad L1: dW1s = x^T dGs (k_grouped M=H,N=2S,K=T)
    At1 = torch.randn(host_ks_s[0], H, device=device, dtype=torch.float32)
    Bt1 = torch.randn(host_ks_s[0], 2 * S, device=device, dtype=torch.float32)
    aks1 = quant_k_operand(At1, host_ks_s); bks1 = quant_k_operand(Bt1, host_ks_s)
    dW1s = torch.zeros(1, H, 2 * S, device=device, dtype=torch.float32)
    t = _bench(lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (aks1[0], aks1[1]), (bks1[0], bks1[1]), dW1s, host_ks_s, ksl_s, dW1s, recipe=(1, 1, ka)), iters)
    _tlog('bwd shared wgrad L1 (k_grouped)', t * 1e6, 2 * host_ks_s[0] * H * 2 * S, 'BWD', rows)

    # SwiGLU 反向 + 反向激活量化（dY/dG）+ combine 反向 + router 反向 + scatter-add
    t = _bench(lambda: (per_token_cast_to_fp8(dY, use_ue8m0=False, gran_k=ka),
                        per_token_cast_to_fp8(dG, use_ue8m0=False, gran_k=ka),
                        per_token_cast_to_fp8(dGs, use_ue8m0=False, gran_k=ka)), iters)
    _tlog('bwd activation quant x3', t * 1e6, 0, 'BWD', rows)

    t = _bench(lambda: (dA * silu(g) + dA * u * silu_grad(g),), iters)
    _tlog('bwd SwiGLU (elem)', t * 1e6, 0, 'BWD', rows)

    t = _bench(lambda: torch.index_add(out, 0, idx, dX.float()), iters)
    _tlog('bwd scatter-add dX', t * 1e6, 0, 'BWD', rows)

    # ---------------------------------------------------------------- 汇总
    fwd = sum(r[1] for r in rows if r[3] == 'FWD')
    bwd = sum(r[1] for r in rows if r[3] == 'BWD')
    gemm = sum(r[1] for r in rows if r[2] > 0)
    quant = sum(r[1] for r in rows if 'quant' in r[0])
    total = fwd + bwd
    gemm_only = total - quant
    print('   ' + '-' * 70)
    print(f'   前向小计 {fwd:9.0f} us   ({fwd / total * 100:4.1f}%)')
    print(f'   反向小计 {bwd:9.0f} us   ({bwd / total * 100:4.1f}%)')
    print(f'   GEMM 小计 {gemm:8.0f} us   ({gemm / total * 100:4.1f}%)  '
          f'非 GEMM {total - gemm:8.0f} us ({(total - gemm) / total * 100:4.1f}%)')
    print(f'   激活量化(参考实现,torch) {quant:8.0f} us   ({quant / total * 100:4.1f}%)')
    print(f'   一个训练步合计 {total:9.0f} us')
    print(f'   [扣掉 torch 参考量化后] {gemm_only:9.0f} us  '
          f'(真实 Mega MoE 把量化融进 L1 epilogue 设备端，这里不计该参考开销)')
    return dict(fwd=fwd, bwd=bwd, gemm=gemm, quant=quant, total=total, gemm_only=gemm_only)


def main():
    device = 'cuda'
    arch = get_arch_major()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={arch}, device={torch.cuda.get_device_name(0)}')
    print('Mega MoE 融合主 kernel 是 SM100 专属、本机不可跑；本实验用真实 SM90 FP8 grouped/dense kernel')
    print('把「一个 MoE 训练步」的前向与反向分段搭起来并计时，并闭环共享专家的反向。')
    print('=' * 100)

    part_a(device)

    print('## B. DeepSeek 量级 一个训练步 的分段成本账')
    print('   前向 = 路由 L1/L2 + 共享 L1/L2 + 激活量化 + SwiGLU + combine + router 反向(scatter)')
    print('   反向 = 路由 dgrad/wgrad ×2 + 共享 dgrad/wgrad ×2 + 反向激活量化 + SwiGLU 反向 + scatter-add')
    print('=' * 100)
    for (T, H, I, E, K, Es) in [(2048, 7168, 2048, 8, 2, 1), (4096, 7168, 2048, 8, 2, 1)]:
        try:
            ledger(T, H, I, E, K, Es, device)
        except Exception:
            print('   ledger FAILED:\n' + traceback.format_exc())
        print()

    print('ALL DONE')


if __name__ == '__main__':
    main()
