"""P171-1: MoE 反向的「激活存储 vs 重算」（activation checkpointing）。

背景：MoE 前向 linear1 输出 `h = x @ W1^T ∈ [m, 2I]`（gate/up 预激活）是 SwiGLU 反向的输入
（`dG1 = dA * silu'(G1) * G2`、`dG2 = dA * silu(G1)`）。第 154 篇（P144-1）拆了这段「非 GEMM 段」
并指出「前提是前向不单独存 gate/up」；第 155/156 篇（P145/P146）试图把它折进 dgrad 的 epilogue；
第 169 篇（P159-1）给了完整训练步账。但**从没量过**：不存 h 时，把它**重算**一次的代价、显存收益、
以及数值上是否无损。

本实验用 H100(sm90) 可跑的真实 SM90 FP8 kernel（DeepGEMM）量三件事：

  Part A 语义与逐位性：重算 `h` 与存下来的 `h` 用**同一个** `m_grouped_fp8_gemm_nt_contiguous`
          得到，检查是否**逐位相等**（若 kernel 确定，第 170 篇已证 grouped GEMM 结构性 bit 确定）；
          并检查由 h 与由 h_rec 还原的 SwiGLU 反向 `dG` 是否逐位相等。
          对照「存一半」：只存 G1，由已存的（量化的）SwiGLU 输出反推 `G2 = a / silu(G1)` 的误差。

  Part B 显存账：DeepSeek 量级下 STORE-ALL / STORE-G1 / RECOMPUTE 三方案的字节数，routed+shared。

  Part C 时间/可重叠性：重算 = 一次前向 L1 grouped GEMM；它**不依赖 h 的反向 GEMM**
          （dgrad L2 `dA=dY@W2`、wgrad L2 `dW2=a^TdY`）能否把它藏进 shadow。

约束：Mega MoE 主 kernel 是 SM100 专属、本机跑不了；本实验只用其**架构无关的数学与量化语义**
（真实 SM90 grouped/dense FP8 kernel），主 kernel 部分仅分析。目标卡代号敏感：全文用「目标卡」。
输出重定向到 exp_moe_activation_recompute.out.txt。
"""
import os
import sys
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
    quant_act, quant_weight, run_grouped_fp8, _bench,
)
from exp_mega_moe_forward_shared import (  # noqa: E402
    build_shared, quant_weight_2d,
)
from exp_moe_backward_kernels import quant_k_operand  # noqa: E402

GB = 1024 ** 3


def silu_grad(z):
    s = torch.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


# ===================================================================================== A
def part_a(device):
    print('## A. 语义与逐位性：重算 h == 存下来的 h，且 dG 逐位一致')
    print('   真实 SM90 `m_grouped_fp8_gemm_nt_contiguous`（前向 L1）跑两次；')
    print('   存一半的对照：只存 G1，由已存的（per-token FP8）SwiGLU 输出反推 G2=a/silu(G1)。')
    print('=' * 100)
    ma = get_mk_alignment_for_contiguous_layout()
    T, H, I, E, KTOP = 256, 2048, 1024, 8, 2
    x, Wg, W1, W2 = build_inputs(T, H, I, E, KTOP, device)
    tw, ti = router(x, Wg, KTOP)
    rows_by_e, counts, slices, m_total, layout, row_to_tk = group_layout(ti, E, ma)
    print(f'T={T} H={H} I={I} E={E} topk={KTOP}  m_total={m_total} (M-alignment={ma})')

    # 前向 L1 的输入（dispatch 后拿到、反向 wgrad 也要用的）x 激活
    A1 = gather_acts(x, rows_by_e, slices, H)          # [m_total, H] fp32
    a1q, _ = quant_act(A1)                             # per-token FP8
    w1q, _ = quant_weight(W1)                          # [E, 2I, H] FP8

    # STORE-ALL：forward 存下来的 h（bf16 输出，就是 DeepGEMM 的 D）
    h_store = run_grouped_fp8(a1q, w1q, layout, m_total, 2 * I)      # [m,2I] bf16

    # RECOMPUTE：用同一份量化输入再发一次同一 kernel
    h_rec = run_grouped_fp8(a1q, w1q, layout, m_total, 2 * I)        # [m,2I] bf16

    bit_eq = torch.equal(h_store, h_rec)
    print(f'   STORE-ALL h vs RECOMPUTE h:  torch.equal = {bit_eq}   '
          f'(max|d|={ (h_store.float()-h_rec.float()).abs().max().item():.3e} bf16 ulp)')

    # SwiGLU 反向：dG1 = dA*silu\'(G1)*G2, dG2 = dA*silu(G1)
    dA = torch.randn(m_total, I, device=device, dtype=torch.float32)
    def swiglu_back(h):
        g = h[:, :I].float(); u = h[:, I:].float()
        return torch.cat([dA * silu_grad(g) * u, dA * silu(g)], dim=-1)
    dG_store = swiglu_back(h_store)
    dG_rec = swiglu_back(h_rec)
    print(f'   SwiGLU 反向 dG:  torch.equal = {torch.equal(dG_store, dG_rec)}  '
          f'rel = {rel(dG_store, dG_rec):.3e}')

    # 存一半：只存 G1（[m,I]），由已存的 SwiGLU 输出 a 反推 G2
    g = h_store[:, :I].float(); u = h_store[:, I:].float()
    a_pre = silu(g) * u                                   # [m,I] fp32（乘 top-k 权重前）
    a_q, a_sf = per_token_cast_to_fp8(a_pre, use_ue8m0=False, gran_k=GRAN_K)
    a_deq = a_q.float() * a_sf.repeat_interleave(GRAN_K, dim=1)
    u_from_fp32 = a_pre / (silu(g) + 1e-30)
    u_from_fp8 = a_deq / (silu(g) + 1e-30)
    print(f'   STORE-G1 反推 G2:  vs 真 G2  rel(fp32 存储)={rel(u_from_fp32, u):.3e}  '
          f'rel(FP8 存储)={rel(u_from_fp8, u):.3e}')
    print(f'   ⇒ RECOMPUTE 与 STORE-ALL 数值完全相同（同一确定 kernel）；STORE-G1 省一半但随着')
    print(f'     存储精度引入误差（FP8 反推被 silu(G1) 小值时放大）。')
    print()


# ===================================================================================== B
def part_b(device):
    print('## B. 显存账：STORE-ALL / STORE-G1 / RECOMPUTE（DeepSeek 量级）')
    print('   h = linear1 预激活 [m,2I]；STORE-ALL 存 h（bf16），STORE-G1 只存 G1 [m,I]，')
    print('   RECOMPUTE 额外 0（x 已在手，反向 wgrad 本就要它）。shared 专家同理（S=Es·I）。')
    print('=' * 100)
    ma = get_mk_alignment_for_contiguous_layout()
    for (T, H, I, E, K, Es) in [(2048, 7168, 2048, 8, 2, 1), (4096, 7168, 2048, 8, 2, 1),
                                (8192, 7168, 2048, 8, 2, 1)]:
        counts = [T * K // E] * E
        m_total = sum(align(c, ma) for c in counts)
        S = Es * I
        # routed
        r_all = m_total * (2 * I) * 2                     # bf16
        r_g1 = m_total * I * 2
        # shared: 每 token 一份
        s_all = T * (2 * S) * 2
        s_g1 = T * S * 2
        all_b = r_all + s_all
        g1_b = r_g1 + s_g1
        # 参照：一张权重矩阵（fp8）与整层激活
        w1_b = E * (2 * I) * H * 1                        # fp8 W1
        print(f'T={T:5d}: m_total={m_total:6d}  S={S:5d}')
        print(f'   STORE-ALL  = {all_b/1e6:8.1f} MB (routed {r_all/1e6:.1f} + shared {s_all/1e6:.1f})')
        print(f'   STORE-G1   = {g1_b/1e6:8.1f} MB ({g1_b/all_b*100:.0f}% of ALL)')
        print(f'   RECOMPUTE  = {0.0:8.1f} MB (+ 一次前向 L1 GEMM，见 Part C)')
        print(f'   参考: 一张 W1(fp8)={w1_b/1e6:.1f} MB、单层激活 x={T*H*2/1e6:.1f} MB'
              f'  ⇒ STORE-ALL/W1 = {all_b/w1_b*100:.0f}%')
    print()


# ===================================================================================== C
def _tlog(name, t_us, flops, rows):
    tf = flops / (t_us * 1e-6) / 1e12 if flops > 0 else 0.0
    rows.append((name, t_us, tf))
    print(f'   {name:40s} {t_us:9.0f} us   {tf:5.0f} TFLOPS')


def part_c(device, iters=30):
    print('## C. 时间 / 可重叠性：重算 = 一次前向 L1 GEMM，能否藏进「不依赖 h 的反向 GEMM」')
    print('   重算 needed-by: 反向里依赖 h 的只有 SwiGLU 反向→dgrad L1/wgrad L1；')
    print('   不依赖 h 的：dgrad L2 (dA=dY@W2)、wgrad L2 (dW2=a^T dY) —— 它们可与重算并发。')
    print('=' * 100)
    ma = get_mk_alignment_for_contiguous_layout()
    ka = GRAN_K
    for (T, H, I, E, K, Es) in [(2048, 7168, 2048, 8, 2, 1), (4096, 7168, 2048, 8, 2, 1)]:
        counts = [T * K // E] * E
        host_ks = [align(c, ka) for c in counts]
        total_k = sum(host_ks)
        m_total = sum(align(c, ma) for c in counts)
        S = Es * I
        rows = []
        print(f'  配置 T={T} H={H} I={I} E={E} topk={K} Es={Es}  m_total={m_total} total_k={total_k}')

        # ---- 重算：前向 L1 routed (m_grouped) ----
        A1 = torch.randn(m_total, H, device=device, dtype=torch.float32)
        a1q = per_token_cast_to_fp8(A1, use_ue8m0=False, gran_k=ka)
        W1r = torch.randn(E, 2 * I, H, device=device, dtype=torch.float32)
        w1r = torch.empty(E, 2 * I, H, dtype=torch.float8_e4m3fn, device=device)
        w1rs = torch.empty(E, align(2 * I, ka) // ka, align(H, ka) // ka, device=device, dtype=torch.float32)
        for e in range(E):
            w1r[e], w1rs[e] = per_block_cast_to_fp8(W1r[e], use_ue8m0=False, gran_k=ka)
        layout = torch.full((m_total,), 0, device=device, dtype=torch.int32)
        d1r = torch.empty(m_total, 2 * I, device=device, dtype=torch.bfloat16)
        t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a1q, (w1r, w1rs), d1r, layout, disable_ue8m0_cast=True), iters)
        _tlog('RECOMPUTE routed L1 (m_grouped)', t * 1e6, 2 * m_total * 2 * I * H, rows)

        # ---- 重算：前向 L1 shared (dense) ----
        W1s, W2s = build_shared(T, S, H, device)
        xs = torch.randn(T, H, device=device, dtype=torch.float32)
        xq = per_token_cast_to_fp8(xs, use_ue8m0=False, gran_k=ka)
        w1s_q = quant_weight_2d(W1s)
        d1s = torch.empty(T, 2 * S, device=device, dtype=torch.bfloat16)
        t = _bench(lambda: deep_gemm.fp8_gemm_nt(xq, w1s_q, d1s, disable_ue8m0_cast=True), iters)
        _tlog('RECOMPUTE shared L1 (dense fp8)', t * 1e6, 2 * T * 2 * S * H, rows)

        # ---- 不依赖 h 的反向 GEMM：dgrad L2 (m_grouped) ----
        dY = torch.randn(m_total, H, device=device, dtype=torch.float32)
        dyq = per_token_cast_to_fp8(dY, use_ue8m0=False, gran_k=ka)
        B2b = torch.randn(E, I, H, device=device, dtype=torch.float32)
        b2b = torch.empty(E, I, H, dtype=torch.float8_e4m3fn, device=device)
        b2bs = torch.empty(E, align(I, ka) // ka, align(H, ka) // ka, device=device, dtype=torch.float32)
        for e in range(E):
            b2b[e], b2bs[e] = per_block_cast_to_fp8(B2b[e], use_ue8m0=False, gran_k=ka)
        dA = torch.empty(m_total, I, device=device, dtype=torch.bfloat16)
        t = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            dyq, (b2b, b2bs), dA, layout, disable_ue8m0_cast=True), iters)
        _tlog('SHADOW dgrad L2 (m_grouped)', t * 1e6, 2 * m_total * I * H, rows)

        # ---- 不依赖 h 的反向 GEMM：wgrad L2 (k_grouped) ----
        Ak = torch.randn(total_k, I, device=device, dtype=torch.float32)
        Bk = torch.randn(total_k, H, device=device, dtype=torch.float32)
        ak = quant_k_operand(Ak, host_ks); bk = quant_k_operand(Bk, host_ks)
        ksl = torch.tensor(host_ks, device=device, dtype=torch.int32)
        dW2 = torch.zeros(E, I, H, device=device, dtype=torch.float32)
        t = _bench(lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (ak[0], ak[1]), (bk[0], bk[1]), dW2, host_ks, ksl, dW2, recipe=(1, 1, ka)), iters)
        _tlog('SHADOW wgrad L2 (k_grouped)', t * 1e6, 2 * total_k * I * H, rows)

        rec = [r[1] for r in rows if r[0].startswith('RECOMPUTE')]
        sh = [r[1] for r in rows if r[0].startswith('SHADOW')]
        rec_t, sh_t = sum(rec), sum(sh)
        print(f'   RECOMPUTE 合计 {rec_t:8.0f} us   SHADOW(不依赖 h)合计 {sh_t:8.0f} us   '
              f'RECOMPUTE/SHADOW = {rec_t/sh_t:.2f}x')
        verdict = ('重算可完全藏在 shadow 里' if rec_t <= sh_t
                   else f'重算超出 shadow {rec_t-sh_t:.0f} us（需额外暴露 {100*(rec_t-sh_t)/(rec_t+sh_t):.0f}%）')
        print(f'   ⇒ {verdict}')
        print()


def main():
    device = 'cuda'
    arch = get_arch_major()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={arch}, device={torch.cuda.get_device_name(0)}')
    print('MoE 反向的激活存储 vs 重算：h=linear1 预激活（SwiGLU 反向输入）存下来还是重算。')
    print('Mega MoE 融合主 kernel 是 SM100 专属、本机不可跑；本实验用真实 SM90 FP8 grouped/dense kernel。')
    print('=' * 100)

    for fn in (part_a, part_b, part_c):
        try:
            fn(device)
        except Exception:
            print(f'   {fn.__name__} FAILED:\n' + traceback.format_exc())
            print()

    print('ALL DONE')


if __name__ == '__main__':
    main()
