"""P158-1: 在 H100 上用 SM90 grouped FP8 kernel + 稠密 FP8 kernel 把 **Mega MoE 的前向
（路由专家 + 共享专家 + combine）** 端到端搭起来，闭环第 99 篇（P89-1）只做了路由分支的缺口。

背景：第 99 篇用 SM90 `m_grouped_fp8_gemm_nt_contiguous` 复刻了「dispatch → L1 → SwiGLU → L2 →
combine」这条**路由专家**流水线，但把 Mega MoE 里另一条与路由并行、且贯穿全链路的通路——**共享专家**
（`y = Σ_{i∈topk} w_i·Expert_i(x) + SharedExpert(x)`）——留在了第 44 篇的「独立/bf16」实验里。

本实验把两条通路**在同一个 H100 前向里一起跑**，共享专家用**稠密 FP8** `fp8_gemm_nt`
（共享权重是 2D 张量、没有专家维），并核对三件事的真实语义：
  1. 共享专家的**两次 SwiGLU**：L1 输出 `[T, S·2]`（沿 N 变宽）、`silu*up`**不乘 top-k 权重**
     （`kernel:1068` stored_cached_weight 初值 1.0，仅 `not is_shared()` 才重载 `:1095-1100`），
     L2 的收缩维 K = S（沿 K 变宽）。
  2. 共享结果经 combine buffer 的 **`kNumTopk` 号槽** 被 **fp32 相加一次、不带权重**
     （`kernel:1433`、`:1449-1450` 的 `slot_idx = kNumTopk`）。
  3. 「共享项只加一次」的**反例**（加两次 / 重复 topk 次）误差会爆掉。

数值口径：端到端 fp8-kernel vs fp64；分层（shared/routed 各自 vs 反量化参考）；共享分支的
边际误差与吞吐（dense FP8 GEMM 的 TFLOPS + 共享 MLP 占整层 FLOPs 的比例）。

输出重定向到 exp_mega_moe_forward_shared.out.txt。
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 复用第 99 篇（P89-1）实验的搭台函数：数据、路由、grouped 布局、量化、grouped kernel、combine、计时。
from exp_mega_moe_forward_emul import (  # noqa: E402
    deep_gemm, get_arch_major, GRAN_K, align,
    per_token_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout,
    rel, cos, silu, build_inputs, router, group_layout, gather_acts,
    quant_act, quant_weight, run_grouped_fp8, kernel_pipeline, combine, _bench,
)


def build_shared(T, S, H, device, dtype=torch.float32, seed=1):
    g = torch.Generator(device='cpu').manual_seed(seed)
    W1s = torch.randn(2 * S, H, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    W2s = torch.randn(H, S, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    return W1s, W2s


def quant_weight_2d(W2d):
    """2D 权重 -> (fp8 [N,K], 2D per-128 块 SF)。"""
    q = quant_weight(W2d[None])[0]
    return q[0][0], q[1][0]


def run_dense_fp8(A_quant, B_quant, out_rows):
    """稠密（无分组）FP8 nt：A per-token 1D SF + B per-128 块 2D SF。返回 bf16 输出。"""
    a_f8, a_sf = A_quant
    b_f8, b_sf = B_quant
    d = torch.empty(A_quant[0].shape[0], B_quant[0].shape[0],
                    device=a_f8.device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((a_f8, a_sf), (b_f8, b_sf), d, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    return d


def shared_pipeline(x, W1s, W2s, S):
    """共享专家的 FP8 前向（dense）：x[T,H] → L1[T,2S] → SwiGLU(无权重)[T,S] → L2[T,H]。
    返回 (A1s, l1s, A2s, ys)。"""
    A1s = x.float()
    a1q = quant_act(A1s)[0]
    w1q = quant_weight_2d(W1s)
    l1s = run_dense_fp8(a1q, w1q, A1s.shape[0])           # [T, 2S] bf16
    g = l1s[:, :S].float()
    u = l1s[:, S:].float()
    A2s = silu(g) * u                                     # 不乘 top-k 权重（kernel:1068）
    a2q = quant_act(A2s)[0]
    w2q = quant_weight_2d(W2s)
    ys = run_dense_fp8(a2q, w2q, A2s.shape[0])            # [T, H] bf16
    return A1s, l1s, A2s, ys


def ref_shared(x, W1s, W2s, S, dtype):
    g = x.to(dtype) @ W1s[:S, :].to(dtype).T
    u = x.to(dtype) @ W1s[S:, :].to(dtype).T
    a = silu(g) * u
    return a @ W2s.to(dtype).T


def dec_act(A):
    a_f8, a_sf = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN_K)
    return a_f8.float() * a_sf.repeat_interleave(GRAN_K, dim=1)


def dec_weight(W2):
    wf, wsf = per_block_cast_to_fp8(W2, use_ue8m0=False, gran_k=GRAN_K)
    return wf.float() * wsf.repeat_interleave(GRAN_K, 0).repeat_interleave(GRAN_K, 1)


def run_case(T, H, I, E, KTOP, Es, device, do_perf=False):
    S = Es * I
    ma = get_mk_alignment_for_contiguous_layout()
    x, Wg, W1, W2 = build_inputs(T, H, I, E, KTOP, device)
    W1s, W2s = build_shared(T, S, H, device)
    tw, ti = router(x, Wg, KTOP)
    rows_by_e, counts, slices, m_total, layout, row_to_tk = group_layout(ti, E, ma)

    # ---- fp64 权威：路由（权重在 L1 输出处乘）+ 共享（不加权、只加一次）
    out_r64 = torch.zeros(T, H, device=device, dtype=torch.float64)
    for e, (s, _) in enumerate(slices):
        if not rows_by_e[e]:
            continue
        idx = torch.tensor([t for t, k in rows_by_e[e]], device=device)
        wr = torch.tensor([k for t, k in rows_by_e[e]], device=device)
        xe = x.double()[idx]
        w = tw.double()[idx, wr][:, None]
        y = (silu(xe @ W1[e, :I, :].double().T) * (xe @ W1[e, I:, :].double().T) * w) @ W2[e].double().T
        out_r64.index_add_(0, idx, y)
    sh64 = ref_shared(x, W1s, W2s, S, torch.float64)
    out64 = out_r64 + sh64
    sh32 = ref_shared(x, W1s, W2s, S, torch.float32)

    # ---- 路由 FP8 kernel 分支（第 99 篇）
    A1, d1, A2, d2 = kernel_pipeline(x, W1, W2, tw, ti, rows_by_e, slices, m_total, layout, H, I, 2 * I, H)
    out_route = combine(d2, rows_by_e, slices, T, H, row_to_tk)      # fp32

    # ---- 共享 FP8 kernel 分支
    A1s, l1s, A2s, ys = shared_pipeline(x, W1s, W2s, S)

    # ---- combine：共享项 kNumTopk 槽、fp32 相加一次、无权重
    out_total = out_route + ys.float()

    # ---- 分层：路由 vs 反量化参考 / 共享 vs 反量化参考
    a1q, deqA1 = quant_act(A1)
    w1q, deqW1 = quant_weight(W1)
    a2q, deqA2 = quant_act(A2)
    w2q, deqW2 = quant_weight(W2)
    refL1 = torch.zeros(m_total, 2 * I, device=device)
    refL2 = torch.zeros(m_total, H, device=device)
    for e, (s, _) in enumerate(slices):
        if not rows_by_e[e]:
            continue
        seg = slice(s, s + len(rows_by_e[e]))
        refL1[seg] = deqA1[seg] @ deqW1[e].T
        refL2[seg] = deqA2[seg] @ deqW2[e].T
    valid = torch.tensor([r is not None for r in row_to_tk], device=device)
    kerL1 = rel(d1.float()[valid], refL1[valid])
    kerL2 = rel(d2.float()[valid], refL2[valid])

    # 共享分支 vs 反量化参考（隔离 kernel+布局）
    dW1s = dec_weight(W1s)
    dW2s = dec_weight(W2s)
    refSh = dec_act(A1s) @ dW1s.T
    refSh_l1 = rel(l1s.float(), refSh)
    refSh2 = dec_act(A2s) @ dW2s.T
    refSh_l2 = rel(ys.float(), refSh2)

    # ---- 端到端
    e2e_total = rel(out_total, out64)
    e2e_route_only = rel(out_route, out_r64)
    e2e_shared_only = rel(ys.float(), sh64)

    # ---- 反例：共享项加两次 / 加 topk 次
    e2e_shared_twice = rel(out_route + 2.0 * ys.float(), out64)
    e2e_shared_topk = rel(out_route + KTOP * ys.float(), out64)

    fl_route = 2 * m_total * (2 * I) * H + 2 * m_total * H * I
    fl_shared = 2 * T * (2 * S) * H + 2 * T * H * S
    res = dict(T=T, H=H, I=I, E=E, KTOP=KTOP, Es=Es, S=S, m_total=m_total,
               e2e_total=e2e_total, cos_total=cos(out_total, out64),
               e2e_route_only=e2e_route_only, e2e_shared_only=e2e_shared_only,
               kerL1=kerL1, kerL2=kerL2, refSh_l1=refSh_l1, refSh_l2=refSh_l2,
               e2e_shared_twice=e2e_shared_twice, e2e_shared_topk=e2e_shared_topk,
               fl_route=fl_route, fl_shared=fl_shared,
               sh_frac=fl_shared / (fl_route + fl_shared))
    if do_perf:
        a1q = quant_act(A1s)[0]
        w1q = quant_weight_2d(W1s)
        d1p = torch.empty(T, 2 * S, device=device, dtype=torch.bfloat16)
        t_l1 = _bench(lambda: deep_gemm.fp8_gemm_nt((a1q[0], a1q[1]), (w1q[0], w1q[1]), d1p, disable_ue8m0_cast=True))
        a2q = quant_act(A2s)[0]
        w2q = quant_weight_2d(W2s)
        d2p = torch.empty(T, H, device=device, dtype=torch.bfloat16)
        t_l2 = _bench(lambda: deep_gemm.fp8_gemm_nt((a2q[0], a2q[1]), (w2q[0], w2q[1]), d2p, disable_ue8m0_cast=True))
        # 量化开销（共享激活的 per-token 量化）
        t_q = _bench(lambda: per_token_cast_to_fp8(A1s, use_ue8m0=False, gran_k=GRAN_K))
        res['sh_l1_us'] = t_l1 * 1e6
        res['sh_l2_us'] = t_l2 * 1e6
        res['sh_l1_tflops'] = 2 * T * (2 * S) * H / t_l1 / 1e12
        res['sh_l2_tflops'] = 2 * T * H * S / t_l2 / 1e12
        res['sh_quant_us'] = t_q * 1e6
        # 同一次运行里也测路由 grouped kernel（与共享稠密对照）
        A1q = quant_act(A1)[0]
        W1q = quant_weight(W1)[0]
        d1r = torch.empty(m_total, 2 * I, device=device, dtype=torch.bfloat16)
        t_r1 = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            A1q, W1q, d1r, layout, disable_ue8m0_cast=True))
        A2q = quant_act(A2)[0]
        W2q = quant_weight(W2)[0]
        d2r = torch.empty(m_total, H, device=device, dtype=torch.bfloat16)
        t_r2 = _bench(lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            A2q, W2q, d2r, layout, disable_ue8m0_cast=True))
        res['r_l1_us'] = t_r1 * 1e6
        res['r_l2_us'] = t_r2 * 1e6
        res['r_l1_tflops'] = 2 * m_total * (2 * I) * H / t_r1 / 1e12
        res['r_l2_tflops'] = 2 * m_total * H * I / t_r2 / 1e12
    return res


def main():
    device = 'cuda'
    arch = get_arch_major()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={arch}, device={torch.cuda.get_device_name(0)}')
    print('SM100 fuse 的 Mega MoE 主 kernel 本机不可跑；本实验用 SM90 grouped FP8（路由）'
          '+ 稠密 FP8（共享）逐段复刻「路由 + 共享 + combine」整条前向。')
    print('=' * 105)

    print('## A. 路由 + 共享 端到端误差与分解（小配置）')
    print('   便宜起见共享专家中间维 S = Es·I；kernel:1068 共享 L1 SwiGLU 不乘权重、'
          'kernel:1433/1449 共享槽 = kNumTopk、只加一次。')
    smalls = [
        (64, 512, 256, 8, 2, 1),
        (256, 1024, 512, 8, 2, 2),
        (512, 1024, 512, 16, 4, 1),
    ]
    for (T, H, I, E, K, Es) in smalls:
        r = run_case(T, H, I, E, K, Es, device)
        print(f'T={r["T"]:4d} H={r["H"]:5d} I={r["I"]:4d} E={r["E"]:2d} topk={r["KTOP"]} Es={r["Es"]} '
              f'S={r["S"]:5d} | m_total={r["m_total"]:6d} 共享FLOP占比={r["sh_frac"]*100:4.1f}%')
        print(f'   路由+共享 vs fp64: rel={r["e2e_total"]:.4e} cos={r["cos_total"]:.6f}')
        print(f'   分层(各自 vs fp64):  路由={r["e2e_route_only"]:.4e}  共享={r["e2e_shared_only"]:.4e}')
        print(f'   分层(kernel+布局 vs 反量化):  路由 L1/L2={r["kerL1"]:.4e}/{r["kerL2"]:.4e}  '
              f'共享 L1/L2={r["refSh_l1"]:.4e}/{r["refSh_l2"]:.4e}')
        print(f'   反例(共享项多加):  加两次={r["e2e_shared_twice"]:.4e}  加topk次={r["e2e_shared_topk"]:.4e}')

    print()
    print('## B. DeepSeek 量级（T=2048,H=7168,I=2048,E=8,topk=2,Es=1）+ 共享分支吞吐')
    r = run_case(2048, 7168, 2048, 8, 2, 1, device, do_perf=True)
    print(f'   m_total={r["m_total"]}  S={r["S"]}  共享FLOP占比={r["sh_frac"]*100:.1f}%')
    print(f'   路由+共享 vs fp64: rel={r["e2e_total"]:.4e} cos={r["cos_total"]:.6f}')
    print(f'   分层: 路由={r["e2e_route_only"]:.4e} 共享={r["e2e_shared_only"]:.4e}')
    print(f'   kernel+布局:  路由 L1/L2={r["kerL1"]:.4e}/{r["kerL2"]:.4e}  '
          f'共享 L1/L2={r["refSh_l1"]:.4e}/{r["refSh_l2"]:.4e}')
    print(f'   共享分支吞吐: L1 T={r["T"]} N={2*r["S"]} K={r["H"]} t={r["sh_l1_us"]:.0f}us '
          f'{r["sh_l1_tflops"]:.0f} TFLOPS | L2 N={r["H"]} K={r["S"]} t={r["sh_l2_us"]:.0f}us '
          f'{r["sh_l2_tflops"]:.0f} TFLOPS | 激活量化(参考实现) t={r["sh_quant_us"]:.0f}us')
    print(f'   路由分支吞吐(对照): L1 m={r["m_total"]} N={2*r["I"]} K={r["H"]} t={r["r_l1_us"]:.0f}us '
          f'{r["r_l1_tflops"]:.0f} TFLOPS | L2 N={r["H"]} K={r["I"]} t={r["r_l2_us"]:.0f}us '
          f'{r["r_l2_tflops"]:.0f} TFLOPS')
    print(f'   FLOPs: 路由={r["fl_route"]/1e9:.2f}G  共享={r["fl_shared"]/1e9:.2f}G')

    print()
    print('ALL DONE')


if __name__ == '__main__':
    main()
