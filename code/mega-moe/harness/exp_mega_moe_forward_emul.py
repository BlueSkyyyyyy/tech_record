"""P89-1: 在 H100 上用 SM90 grouped GEMM kernel 把 **Mega MoE 的前向** 端到端搭起来，
与 fp64 / bf16 / fp32 参考对拍，并做「误差来源分解」。

背景：Mega MoE（`sm100_fp8_fp4_mega_moe.cuh`）把 dispatch → linear1 → SwiGLU → linear2 → combine
融合进一个 SM100 mega-kernel（本机 H100 跑不了）。但它每一段的**数学与量化语义**架构无关：
  - L1：`[m_e,H] @ W1_e^T [H,2I]`（门 + up）      —— 激活 per-token FP8(1D) + 权重 per-128 块 FP8(2D)
  - SwiGLU（`kernel:1066-1135`）：gate/up 先 **round bf16**，`silu(gate)*up`，再 **乘 top-k 权重**
    （权重在 L1 epilogue 就乘进激活，`kernel:1135`），随后 per-token UE8M0 量化成 FP8（`kernel:1160-1192`）
  - L2：`[m_e,I] @ W2_e^T [I,H]`               —— 同样 per-token/块 FP8
  - combine（`kernel:1412-1519`）：**不带权重**地对 top-k 贡献求和（权重已进 L1）

因此可以用 H100 上真实可跑的 SM90 `m_grouped_fp8_gemm_nt_contiguous`（1D2D）与
`m_grouped_bf16_gemm_nt_contiguous` 逐段复刻这条流水线，量出：
  1. 端到端误差：fp8-kernel vs fp64 / bf16-torch vs fp64；
  2. **分层误差**：L1、L2 的 kernel+布局误差（vs 反量化操作数上重算）与量化误差；
  3. **消融**：只量化激活 / 只量化权重 / 两者，看谁主导；
  4. 吞吐：两个前向 grouped GEMM 的真实 TFLOPS。

输出重定向到 exp_mega_moe_forward_emul.out.txt。
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import get_arch_major


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              '~/github/DeepGEMM'):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


DEEPGEMM_DIR = _find_deepgemm_dir()
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout,
)

GRAN_K = 128


def rel(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def silu(z):
    return z * torch.sigmoid(z)


# --------------------------------------------------------------------------- data
def build_inputs(T, H, I, E, KTOP, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(T, H, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    Wg = torch.randn(H, E, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    W1 = torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    W2 = torch.randn(E, H, I, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    return x, Wg, W1, W2


def router(x, Wg, KTOP):
    p = torch.softmax(x.float() @ Wg.float(), dim=-1)
    tw, ti = torch.topk(p, KTOP, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)
    return tw, ti


def group_layout(ti, E, ma, m_total_hint=None):
    """按 (t,k)->expert 构造 m-grouped contiguous 布局：返回每段起址、段长、总行数、行->(t,k) 映射。"""
    T, KTOP = ti.shape
    rows_by_e = [[] for _ in range(E)]
    for t in range(T):
        for k in range(KTOP):
            rows_by_e[ti[t, k].item()].append((t, k))
    counts = [len(r) for r in rows_by_e]
    slices, off = [], 0
    for c in counts:
        hm = align(c, ma)
        slices.append((off, off + hm))
        off += hm
    m_total = off
    layout = torch.full((m_total,), -1, device=ti.device, dtype=torch.int32)
    row_to_tk = [None] * m_total
    for e, (s, _) in enumerate(slices):
        for pos, (t, k) in enumerate(rows_by_e[e]):
            layout[s + pos] = e
            row_to_tk[s + pos] = (t, k)
    return rows_by_e, counts, slices, m_total, layout, row_to_tk


def gather_acts(x, rows_by_e, slices, K):
    """把 x 的 token 行按专家段 gather 成 [m_total, K]（fp32 原值，padding 行=0）。"""
    m_total = sum(e - s for s, e in slices)
    A = torch.zeros(m_total, K, device=x.device, dtype=torch.float32)
    for e, (s, _) in enumerate(slices):
        for pos, (t, k) in enumerate(rows_by_e[e]):
            A[s + pos] = x[t].float()
    return A


# --------------------------------------------------------------------------- quant
def quant_act(A, use_ue8m0=False):
    a_f8, a_sf = per_token_cast_to_fp8(A, use_ue8m0=use_ue8m0, gran_k=GRAN_K)
    deq = a_f8.float() * a_sf.repeat_interleave(GRAN_K, dim=1)
    return (a_f8, a_sf), deq


def quant_weight(Wg3, use_ue8m0=False):
    """Wg3: [E,N,K] fp32 -> fp8 data + per-128 块 SF，并给出反量化。"""
    E, N, K = Wg3.shape
    wf = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=Wg3.device)
    wsf = torch.empty(E, align(N, GRAN_K) // GRAN_K, align(K, GRAN_K) // GRAN_K,
                      device=Wg3.device, dtype=torch.float32)
    deq = torch.empty_like(Wg3)
    for e in range(E):
        wf[e], wsf[e] = per_block_cast_to_fp8(Wg3[e], use_ue8m0=use_ue8m0, gran_k=GRAN_K)
        deq[e] = wf[e].float() * wsf[e].repeat_interleave(GRAN_K, 0).repeat_interleave(GRAN_K, 1)
    return (wf, wsf), deq


def run_grouped_fp8(A_quant, B_quant, layout, m_total, N):
    a_f8, a_sf = A_quant
    b_f8, b_sf = B_quant
    d = torch.empty(m_total, N, device=a_f8.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_f8, a_sf), (b_f8, b_sf), d, layout, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    return d


def run_grouped_bf16(A, B, layout, N):
    m_total = A.shape[0]
    ab = A.to(torch.bfloat16)
    bb = B.to(torch.bfloat16)
    d = torch.empty(m_total, N, device=A.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(ab, bb, d, layout)
    torch.cuda.synchronize()
    return d


# --------------------------------------------------------------------------- reference
def ref_pipeline(x, W1, W2, rows_by_e, slices, tw, ti, I, H, dtype):
    """逐专家参考流水线（可 fp64/fp32）。权重在 L1 输出处乘（= Mega MoE 语义）。"""
    T, KTOP = ti.shape
    out = torch.zeros(T, H, device=x.device, dtype=dtype)
    for e, (s, _) in enumerate(slices):
        if not rows_by_e[e]:
            continue
        idx = torch.tensor([t for t, k in rows_by_e[e]], device=x.device)
        wrow = torch.tensor([k for t, k in rows_by_e[e]], device=x.device)
        xe = x[idx].to(dtype)
        w = tw[idx, wrow].to(dtype)[:, None]
        g = xe @ W1[e, :I, :].to(dtype).T
        u = xe @ W1[e, I:, :].to(dtype).T
        a = (silu(g) * u) * w
        y = a @ W2[e].to(dtype).T
        out.index_add_(0, idx, y)
    return out


# --------------------------------------------------------------------------- pipeline (kernel)
def kernel_pipeline(x, W1, W2, tw, ti, rows_by_e, slices, m_total, layout, H, I, N1, N2):
    """FP8 kernel 流水线，返回 L1 输出、L1 量化输入、L2 输出（bf16）。"""
    A1 = gather_acts(x, rows_by_e, slices, H)
    (a1q), _ = quant_act(A1)
    w1q, _ = quant_weight(W1)
    d1 = run_grouped_fp8(a1q, w1q, layout, m_total, N1)          # [m, 2I] bf16

    # SwiGLU + top-k 权重（kernel:1066-1135，bf16 round + 乘权重）
    T, KTOP = ti.shape
    g = d1[:, :I].float()
    u = d1[:, I:].float()
    act = silu(g) * u
    # 把每行的 top-k 权重乘进去
    wrow = torch.zeros(m_total, device=x.device, dtype=torch.float32)
    for e, (s, _) in enumerate(slices):
        for pos, (t, k) in enumerate(rows_by_e[e]):
            wrow[s + pos] = tw[t, k]
    act = act * wrow[:, None]
    A2 = act                                              # [m, I] fp32
    a2q, _ = quant_act(A2)
    w2q, _ = quant_weight(W2)
    d2 = run_grouped_fp8(a2q, w2q, layout, m_total, N2)   # [m, H] bf16
    return A1, d1, A2, d2


def combine(d2, rows_by_e, slices, T, H, row_to_tk):
    out = torch.zeros(T, H, device=d2.device, dtype=torch.float32)
    for e, (s, _) in enumerate(slices):
        for pos, (t, k) in enumerate(rows_by_e[e]):
            out[t] += d2[s + pos].float()
    return out


def _bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# --------------------------------------------------------------------------- one case
def run_case(T, H, I, E, KTOP, device, do_perf=False):
    ma = get_mk_alignment_for_contiguous_layout()
    x, Wg, W1, W2 = build_inputs(T, H, I, E, KTOP, device)
    tw, ti = router(x, Wg, KTOP)
    rows_by_e, counts, slices, m_total, layout, row_to_tk = group_layout(ti, E, ma)

    # references
    out64 = ref_pipeline(x.double(), W1.double(), W2.double(), rows_by_e, slices, tw, ti, I, H, torch.float64)
    out32 = ref_pipeline(x, W1, W2, rows_by_e, slices, tw, ti, I, H, torch.float32)

    # bf16 torch pipeline
    xb = x.to(torch.bfloat16)
    out_bf = ref_pipeline(xb, W1.to(torch.bfloat16), W2.to(torch.bfloat16),
                          rows_by_e, slices, tw.to(torch.bfloat16), ti, I, H, torch.bfloat16)

    # fp8 kernel pipeline
    A1, d1, A2, d2 = kernel_pipeline(x, W1, W2, tw, ti, rows_by_e, slices, m_total, layout, H, I, 2 * I, H)
    out_k = combine(d2, rows_by_e, slices, T, H, row_to_tk)

    # dequant-isolated L1/L2: 在反量化操作数上重算（隔离 kernel+布局）
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

    # kernel out vs dequant-reference pipeline (combine on refL2)
    refL2_comb = combine(refL2.to(torch.bfloat16), rows_by_e, slices, T, H, row_to_tk)
    deq_e2e = rel(out_k, refL2_comb)

    # ablations on the *dequant* reference (isolate quant of act vs weight)
    def pipeline_deq(q_act, q_weight):
        r1 = torch.zeros(m_total, 2 * I, device=device)
        for e, (s, _) in enumerate(slices):
            if not rows_by_e[e]:
                continue
            seg = slice(s, s + len(rows_by_e[e]))
            r1[seg] = (deqA1 if q_act else A1)[seg] @ (deqW1 if q_weight else W1)[e].T
        g = r1[:, :I]; u = r1[:, I:]
        act = silu(g) * u * wrow_expand(rows_by_e, slices, tw, m_total, device)
        r2 = torch.zeros(m_total, H, device=device)
        A2v = act
        a2deq = (a2q[0].float() * a2q[1].repeat_interleave(GRAN_K, 1)) if q_act else A2v
        for e, (s, _) in enumerate(slices):
            if not rows_by_e[e]:
                continue
            seg = slice(s, s + len(rows_by_e[e]))
            r2[seg] = a2deq[seg] @ (deqW2 if q_weight else W2)[e].T
        return combine(r2.to(torch.bfloat16), rows_by_e, slices, T, H, row_to_tk)

    def wrow_expand(rows_by_e, slices, tw, m_total, device):
        w = torch.zeros(m_total, 1, device=device)
        for e, (s, _) in enumerate(slices):
            for pos, (t, k) in enumerate(rows_by_e[e]):
                w[s + pos] = tw[t, k]
        return w

    abl_act = rel(pipeline_deq(True, False), out64)
    abl_w = rel(pipeline_deq(False, True), out64)
    abl_both = rel(pipeline_deq(True, True), out64)
    abl_none = rel(pipeline_deq(False, False), out64)

    res = dict(
        T=T, H=H, I=I, E=E, KTOP=KTOP, m_total=m_total, counts=counts, ma=ma,
        e2e_kernel=rel(out_k, out64), cos_kernel=cos(out_k, out64),
        e2e_bf16=rel(out_bf.float(), out64), e2e_fp32=rel(out32, out64),
        deq_e2e=deq_e2e, kerL1=kerL1, kerL2=kerL2,
        abl_none=abl_none, abl_act=abl_act, abl_w=abl_w, abl_both=abl_both,
    )
    if do_perf:
        # 注意：perf 只用预分配输出 + 直接调 kernel（`run_grouped_fp8` 每调用都 empty+synchronize，
        # 会把 54 TFLOPS 的“含分配/同步”数字误当吞吐；干净测量见下）。
        A1q = quant_act(A1)[0]
        W1q = quant_weight(W1)[0]
        d1p = torch.empty(m_total, 2 * I, device=device, dtype=torch.bfloat16)
        f1 = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            A1q, W1q, d1p, layout, disable_ue8m0_cast=True)
        t1 = _bench(f1)
        A2q = quant_act(A2)[0]
        W2q = quant_weight(W2)[0]
        d2p = torch.empty(m_total, H, device=device, dtype=torch.bfloat16)
        f2 = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            A2q, W2q, d2p, layout, disable_ue8m0_cast=True)
        t2 = _bench(f2)
        fl1 = 2 * m_total * (2 * I) * H
        fl2 = 2 * m_total * H * I
        res['t_l1_us'] = t1 * 1e6
        res['t_l2_us'] = t2 * 1e6
        res['tflops_l1'] = fl1 / t1 / 1e12
        res['tflops_l2'] = fl2 / t2 / 1e12
    return res


def main():
    device = 'cuda'
    arch = get_arch_major()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={arch}, device={torch.cuda.get_device_name(0)}')
    print('SM100 fuse 的 Mega MoE 主 kernel 本机不可跑；本实验用 SM90 grouped FP8/BF16 kernel '
          '逐段复刻其前向数学与量化语义。')
    print('=' * 100)

    # --- 正确性 / 误差分解：小配置 ---
    print('## A. 端到端误差与分解（小配置）')
    smalls = [(64, 512, 256, 8, 2), (256, 1024, 512, 8, 2), (512, 1024, 512, 16, 4)]
    for (T, H, I, E, K) in smalls:
        r = run_case(T, H, I, E, K, device)
        print(f'T={r["T"]:4d} H={r["H"]:5d} I={r["I"]:4d} E={r["E"]:2d} topk={r["KTOP"]} '
              f'| m_total={r["m_total"]:6d} (align {r["ma"]}) counts={r["counts"]}')
        print(f'   端到端 vs fp64:  fp8-kernel rel={r["e2e_kernel"]:.4e} cos={r["cos_kernel"]:.6f} | '
              f'bf16-torch rel={r["e2e_bf16"]:.4e} | fp32-torch rel={r["e2e_fp32"]:.4e}')
        print(f'   分层:  kernel+布局 L1 rel={r["kerL1"]:.4e}  L2 rel={r["kerL2"]:.4e}  '
              f'端到端 vs 反量化参考 rel={r["deq_e2e"]:.4e}')
        print(f'   消融(反量化参考 vs fp64): none={r["abl_none"]:.4e}  仅激活={r["abl_act"]:.4e}  '
              f'仅权重={r["abl_w"]:.4e}  两者={r["abl_both"]:.4e}')

    # --- DeepSeek 量级：误差 + 吞吐 ---
    print()
    print('## B. DeepSeek 量级（T=2048,H=7168,I=2048,E=8,topk=2, m-grouped，1D2D FP8）')
    r = run_case(2048, 7168, 2048, 8, 2, device, do_perf=True)
    print(f'   m_total={r["m_total"]} counts={r["counts"]}')
    print(f'   端到端 vs fp64: fp8-kernel rel={r["e2e_kernel"]:.4e} cos={r["cos_kernel"]:.6f} | '
          f'bf16-torch rel={r["e2e_bf16"]:.4e}')
    print(f'   分层: kernel+布局 L1 rel={r["kerL1"]:.4e} L2 rel={r["kerL2"]:.4e} | '
          f'端到端 vs 反量化参考 rel={r["deq_e2e"]:.4e}')
    print(f'   消融: none={r["abl_none"]:.4e} 仅激活={r["abl_act"]:.4e} 仅权重={r["abl_w"]:.4e} '
          f'两者={r["abl_both"]:.4e}')
    print(f'   吞吐: L1 m={r["m_total"]} N={2*r["I"]} K={r["H"]} t={r["t_l1_us"]:.0f}us '
          f'{r["tflops_l1"]:.0f} TFLOPS | L2 N={r["H"]} K={r["I"]} t={r["t_l2_us"]:.0f}us '
          f'{r["tflops_l2"]:.0f} TFLOPS')

    print()
    print('ALL DONE')


if __name__ == '__main__':
    main()
