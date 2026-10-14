"""P4-4: 把 SM90 grouped GEMM kernel 接到真实 MoE 的**完整反向**，与 PyTorch autograd 对拍。

P4-1 推导了反向数学（float64 对拍，证明公式对），P4-2/P4-3 精读并核对了 wgrad（`k_grouped`）与
dgrad（`m_grouped`）的 API 语义与布局等价。本文（P4-4）把这些 kernel **真的接进一个 MoE 的反向图**：

  前向（torch，fp32，可微）：router/topk → 逐专家 linear1 → SwiGLU → linear2 → combine。
  参考梯度：对 (x, Wg, W1, W2) 调 `torch.autograd.grad`（fp32）。

  反向（kernel，SM90 FP8）：
    - linear2 wgrad  dW2[e] = A_e^T dY_e          → `k_grouped_fp8_gemm_nt_contiguous`（K=token 分组）
    - linear2 dgrad  dA_e   = dY_e @ W2_e^T       → `m_grouped_fp8_gemm_nt_contiguous`（M=token 分组）
    - linear1 wgrad  dW1[e] = x_e^T dG_e          → `k_grouped_*_nt_contiguous`
    - linear1 dgrad  dX_e   = dG_e @ W1_e^T       → `m_grouped_*_nt_contiguous`
  （SwiGLU 反向、combine 反向、router 反向是逐元素/小 dense 算子，保持 torch 实现；只有四个大 GEMM
   换成 DeepGEMM kernel。）

对拍三层：
  1. `vs deq`：与「把喂进 kernel 的 FP8 反量化后在 fp32 重算的 GEMM」比较 —— 隔离 **kernel+布局** 误差；
  2. `vs ref`：与 fp32 autograd 参考比较 —— 含 **FP8 量化** 的总误差；
  3. 端到端：四个 kernel 输出拼回 `dW1/dW2/dX`，与 autograd 的 `dW1/dW2/dx_ffn` 比较。

约束：SM100（1D1D + tcgen05 + packed UE8M0）为**仅分析**；本机 H100 走 SM90 的 1D2D/nt。
输出重定向到 exp_moe_backward_kernels.out.txt。
"""
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import get_arch_major


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              os.path.expanduser('~/github/DeepGEMM')):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


DEEPGEMM_DIR = _find_deepgemm_dir()
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from generators import MajorTypeAB  # noqa: E402
from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout, set_mk_alignment_for_contiguous_layout,
)

GRAN_K = 128


def rel(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def silu(z):
    return z * torch.sigmoid(z)


def silu_grad(z):
    s = torch.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


# --------------------------------------------------------------------------- builders
def k_grouped_slices(t_list, ka):
    """每个专家在 contiguous K 轴物理布局中的 (start, end, host_k)。"""
    out, off = [], 0
    for t in t_list:
        hk = align(t, ka)
        out.append((off, off + hk, hk))
        off += hk
    return out, off


def quant_k_operand(X, host_ks):
    """X: [total_k, mn] fp32，各组位于累计 host_ks 处（gap 行为 0）。
    返回 kernel 需要的 (data 1D fp8, sf [mn, total_k/gran]) 及反量化 [total_k, mn]。"""
    total_k, mn = X.shape
    data = torch.zeros(total_k * mn, dtype=torch.float8_e4m3fn, device=X.device)
    deq = torch.empty_like(X)
    sfs = []
    off = 0
    for hk in host_ks:
        xg = X[off:off + hk].contiguous()
        xf8, sf = per_channel_cast_to_fp8(xg, use_ue8m0=False, gran_k=GRAN_K)
        data[off * mn:(off + hk) * mn] = xf8.T.reshape(-1)
        deq[off:off + hk] = xf8.float() * sf.repeat_interleave(GRAN_K, dim=0)
        sfs.append(sf)
        off += hk
    sf_all = torch.cat(sfs).T if sfs else X.new_zeros((mn, 0))
    return data, sf_all, deq


def run_k_grouped(per_A, per_B, ka, want_deq=True):
    """wgrad：输入两组 per-expert [k_e, ·]（K 轴），返回 D[e] [M,N] 与 deq 参考。

    per_A[e]: [t_e, M]（M 固定）、per_B[e]: [t_e, N]（N 固定），收缩维是 token(K)。
    """
    ks = [a.shape[0] for a in per_A]
    host_ks = [align(k, ka) for k in ks]
    total_k = sum(host_ks)
    M = per_A[0].shape[1]
    N = per_B[0].shape[1]
    A = torch.zeros(total_k, M, device=per_A[0].device, dtype=torch.float32)
    B = torch.zeros(total_k, N, device=per_B[0].device, dtype=torch.float32)
    off = 0
    for e, k in enumerate(ks):
        A[off:off + k] = per_A[e]
        B[off:off + k] = per_B[e]
        off += host_ks[e]
    a = quant_k_operand(A, host_ks)
    b = quant_k_operand(B, host_ks)
    layout = torch.tensor(host_ks, device=A.device, dtype=torch.int32)
    d = torch.zeros(len(ks), M, N, device=A.device, dtype=torch.float32)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (a[0], a[1]), (b[0], b[1]), d, host_ks, layout, d, recipe=(1, 1, GRAN_K))
    torch.cuda.synchronize()
    ref = torch.empty_like(d)
    if want_deq:
        off = 0
        for e, k in enumerate(ks):
            hk = host_ks[e]
            ref[e] = a[2][off:off + hk].T @ b[2][off:off + hk]
            off += hk
    return d, ref, (host_ks, total_k)


def m_grouped_slices(t_list, ma):
    out, off = [], 0
    for t in t_list:
        hm = align(t, ma)
        out.append((off, off + hm))
        off += hm
    return out, off


def run_m_grouped(per_A_rows, B_group, ma, want_deq=True):
    """dgrad：per_A_rows[e] [t_e, K]（M 轴分组），B_group[e] [N, K]，返回 D[e] [t_e, N] 与 deq 参考。"""
    ts = [a.shape[0] for a in per_A_rows]
    K = per_A_rows[0].shape[1]
    N = B_group.shape[1]
    slices, m_total = m_grouped_slices(ts, ma)
    A = torch.zeros(m_total, K, device=per_A_rows[0].device, dtype=torch.float32)
    layout = torch.full((m_total,), -1, device=A.device, dtype=torch.int32)
    for e, (s, epos) in enumerate(slices):
        A[s:s + ts[e]] = per_A_rows[e]
        layout[s:s + ts[e]] = e
    a_f8, a_sf = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN_K)
    deq_a = a_f8.float() * a_sf.repeat_interleave(GRAN_K, dim=1)

    b_f8 = torch.empty(B_group.shape, dtype=torch.float8_e4m3fn, device=B_group.device)
    b_sf = torch.empty((B_group.shape[0], align(N, GRAN_K) // GRAN_K, align(K, GRAN_K) // GRAN_K),
                       device=B_group.device, dtype=torch.float32)
    deq_b = torch.empty_like(B_group)
    for e in range(B_group.shape[0]):
        b_f8[e], b_sf[e] = per_block_cast_to_fp8(B_group[e], use_ue8m0=False, gran_k=GRAN_K)
        deq_b[e] = b_f8[e].float() * b_sf[e].repeat_interleave(GRAN_K, 0).repeat_interleave(GRAN_K, 1)

    d = torch.empty(m_total, N, device=A.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_f8, a_sf), (b_f8, b_sf), d, layout, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    ref = torch.empty_like(d)
    if want_deq:
        for e, (s, _) in enumerate(slices):
            ref[s:s + ts[e]] = deq_a[s:s + ts[e]] @ deq_b[e].T
    return d, ref, slices


# --------------------------------------------------------------------------- pipeline
def build_case(T, H, I, E, KTOP, device):
    torch.manual_seed(0)
    x, Wg, W1, W2 = (torch.randn(s, device=device, dtype=torch.float32) for s in
                     [(T, H), (H, E), (E, H, 2 * I), (E, I, H)])
    x.requires_grad_(True)
    Wg.requires_grad_(True)
    W1.requires_grad_(True)
    W2.requires_grad_(True)

    logits = x @ Wg
    p = F.softmax(logits, dim=-1)
    tw, ti = torch.topk(p, KTOP, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)
    sel = torch.zeros_like(p).scatter_(1, ti, 1.0)
    w = torch.zeros_like(p).scatter_(1, ti, tw)

    G = torch.einsum('th,eho->teo', x, W1)
    Gg, Gu = G[..., :I], G[..., I:]
    A = silu(Gg) * Gu
    Y = torch.einsum('tei,eih->teh', A, W2)
    z = torch.einsum('te,teh->th', w, Y)

    dz = torch.randn_like(z)
    ref = torch.autograd.grad(z, (x, Wg, W1, W2), dz)
    dx_ref, dWg_ref, dW1_ref, dW2_ref = ref
    return dict(x=x, Wg=Wg, W1=W1, W2=W2, dz=dz, p=p, sel=sel, w=w, ti=ti, Gg=Gg, Gu=Gu, A=A, Y=Y,
                dx_ref=dx_ref, dWg_ref=dWg_ref, dW1_ref=dW1_ref, dW2_ref=dW2_ref)


def _bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def perf(T=2048, H=7168, I=2048, E=8, KTOP=2):
    """DeepSeek 量级单 rank 反向：只测四个 grouped GEMM kernel 本体。"""
    device = 'cuda'
    ka = GRAN_K
    counts = [T * KTOP // E] * E

    # --- wgrad linear2: dW2[e]=A_e^T dY_e  (M=I, N=H, K=token) ---
    host_ks = [align(k, ka) for k in counts]
    total_k = sum(host_ks)
    A = torch.randn(total_k, I, device=device, dtype=torch.float32)
    B = torch.randn(total_k, H, device=device, dtype=torch.float32)
    a = quant_k_operand(A, host_ks)
    b = quant_k_operand(B, host_ks)
    layout = torch.tensor(host_ks, device=device, dtype=torch.int32)
    d = torch.zeros(E, I, H, device=device, dtype=torch.float32)
    f = lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (a[0], a[1]), (b[0], b[1]), d, host_ks, layout, d, recipe=(1, 1, ka))
    t = _bench(f)
    fl = 2 * I * H * total_k
    print(f'  [1] linear2 wgrad k_grouped  M={I} N={H} sum_k={total_k} t={t * 1e6:6.0f} us '
          f'{fl / t / 1e12:4.0f} TFLOPS')
    r = [fl / t / 1e12]

    # --- wgrad linear1: dW1[e]=x_e^T dG_e  (M=H, N=2I, K=token) ---
    A = torch.randn(total_k, H, device=device, dtype=torch.float32)
    B = torch.randn(total_k, 2 * I, device=device, dtype=torch.float32)
    a = quant_k_operand(A, host_ks)
    b = quant_k_operand(B, host_ks)
    d = torch.zeros(E, H, 2 * I, device=device, dtype=torch.float32)
    f = lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (a[0], a[1]), (b[0], b[1]), d, host_ks, layout, d, recipe=(1, 1, ka))
    t = _bench(f)
    fl = 2 * H * 2 * I * total_k
    print(f'  [4] linear1 wgrad k_grouped  M={H} N={2 * I} sum_k={total_k} t={t * 1e6:6.0f} us '
          f'{fl / t / 1e12:4.0f} TFLOPS')
    r.append(fl / t / 1e12)

    # --- dgrad (M = token grouped) ---
    ma = get_mk_alignment_for_contiguous_layout()
    m_total = sum(align(k, ma) for k in counts)
    slices, m_total = m_grouped_slices(counts, ma)
    layout_m = torch.full((m_total,), -1, device=device, dtype=torch.int32)
    for e, (s, _) in enumerate(slices):
        layout_m[s:s + counts[e]] = e

    def prep_m(N, K):
        Am = torch.randn(m_total, K, device=device, dtype=torch.float32)
        af, asf = per_token_cast_to_fp8(Am, use_ue8m0=False, gran_k=GRAN_K)
        Bg = torch.randn(E, N, K, device=device, dtype=torch.float32)
        bf = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=device)
        bsf = torch.empty(E, align(N, ka) // ka, align(K, ka) // ka, device=device, dtype=torch.float32)
        for e in range(E):
            bf[e], bsf[e] = per_block_cast_to_fp8(Bg[e], use_ue8m0=False, gran_k=GRAN_K)
        dm = torch.empty(m_total, N, device=device, dtype=torch.bfloat16)
        return (af, asf), (bf, bsf), dm

    # dA: M=token, N=I, K=H
    a, b, dm = prep_m(I, H)
    f = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a, b, dm, layout_m, disable_ue8m0_cast=True)
    t = _bench(f)
    fl = 2 * (T * KTOP) * I * H
    print(f'  [2] linear2 dgrad m_grouped  m={T * KTOP} N={I} K={H} t={t * 1e6:6.0f} us '
          f'{fl / t / 1e12:4.0f} TFLOPS')
    r.append(fl / t / 1e12)

    # dX: M=token, N=H, K=2I
    a, b, dm = prep_m(H, 2 * I)
    f = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a, b, dm, layout_m, disable_ue8m0_cast=True)
    t = _bench(f)
    fl = 2 * (T * KTOP) * H * 2 * I
    print(f'  [3] linear1 dgrad m_grouped  m={T * KTOP} N={H} K={2 * I} t={t * 1e6:6.0f} us '
          f'{fl / t / 1e12:4.0f} TFLOPS')
    r.append(fl / t / 1e12)
    print(f'  4 kernels total ~{sum(r):.0f} TFLOPS (each ~{sum(r) / 4:.0f} avg)')


def main():
    device = 'cuda'
    arch = get_arch_major()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={arch}, '
          f'device={torch.cuda.get_device_name(0)}')
    print('=' * 96)

    M_ALIGN = get_mk_alignment_for_contiguous_layout()
    K_ALIGN = GRAN_K
    print(f'contiguous M-alignment={M_ALIGN}, K-alignment={K_ALIGN}')

    T, H, I, E, KTOP = 64, 512, 256, 8, 2
    c = build_case(T, H, I, E, KTOP, device)
    x, Wg, W1, W2, dz = c['x'], c['Wg'], c['W1'], c['W2'], c['dz']
    sel, w, Gg, Gu, A, Y = c['sel'], c['w'], c['Gg'], c['Gu'], c['A'], c['Y']

    # ---- reference backward (fp32, torch) ----
    dY = w[..., None] * dz[:, None, :]                      # [T,E,H]
    dA = torch.einsum('teh,eih->tei', dY, W2.detach())
    dGu = dA * silu(Gg.detach())
    dGg = dA * Gu.detach() * silu_grad(Gg.detach())
    dG = torch.cat([dGg, dGu], dim=-1)                      # [T,E,2I]

    rows = [torch.nonzero(sel[:, e] > 0).flatten() for e in range(E)]
    counts = [r.numel() for r in rows]
    print(f'MoE: T={T} H={H} I={I} E={E} topk={KTOP} | per-expert #tokens={counts} '
          f'(sum={sum(counts)}, avg={sum(counts) / E:.1f})')
    print('=' * 96)

    # ---- kernel 1: linear2 wgrad  dW2[e]=A_e^T dY_e (K=token) ----
    print('[1] linear2 wgrad  k_grouped : dW2[e] = A_e^T dY_e   (M=I, N=H, K=token)')
    per_A = [A[r, e, :] for e, r in enumerate(rows)]
    per_B = [dY[r, e, :] for e, r in enumerate(rows)]
    dW2_k, dW2_deq, info = run_k_grouped(per_A, per_B, K_ALIGN)
    print(f'    host_ks={info[0]} sum_k={info[1]}   vs deq relerr={rel(dW2_k, dW2_deq):.3e} '
          f'cos={cos(dW2_k, dW2_deq):.6f}')
    print(f'    vs autograd dW2 relerr={rel(dW2_k, c["dW2_ref"]):.3e} '
          f'cos={cos(dW2_k, c["dW2_ref"]):.6f}')

    # ---- kernel 2: linear2 dgrad  dA_e=dY_e @ W2_e^T (M=token) ----
    print('[2] linear2 dgrad  m_grouped : dA_e   = dY_e @ W2_e^T (M=token, N=I, K=H)')
    per_rows = [dY[r, e, :] for e, r in enumerate(rows)]
    dA_k, dA_deq, sl2 = run_m_grouped(per_rows, W2.detach(), M_ALIGN)
    dA_k_valid = torch.cat([dA_k[s:s + counts[e]] for e, (s, _) in enumerate(sl2)])
    dA_deq_valid = torch.cat([dA_deq[s:s + counts[e]] for e, (s, _) in enumerate(sl2)])
    print(f'    m_total={dA_k.shape[0]}   vs deq relerr={rel(dA_k_valid, dA_deq_valid):.3e} '
          f'cos={cos(dA_k_valid, dA_deq_valid):.6f}')
    dA_ana = torch.cat([dA[r, e, :] for e, r in enumerate(rows)])
    print(f'    vs fp32 analysis dA_e relerr={rel(dA_k_valid, dA_ana):.3e} '
          f'cos={cos(dA_k_valid, dA_ana):.6f}')

    # kernel dA is used downstream (SwiGLU backward) to build dG_kernel, exactly like the
    # torch path. Scatter the per-expert kernel dA back into dense [T,E,I] (padding rows=0).
    dA_dense = torch.zeros(T, E, I, device=device)
    for e, (s, _) in enumerate(sl2):
        dA_dense[rows[e], e, :] = dA_k[s:s + counts[e]].float()

    # ---- kernel 3: linear1 dgrad  dX_e=dG_e @ W1_e^T (M=token) ----
    print('[3] linear1 dgrad  m_grouped : dX_e   = dG_e @ W1_e^T (M=token, N=H, K=2I)')
    dGu_k = dA_dense * silu(Gg.detach())
    dGg_k = dA_dense * Gu.detach() * silu_grad(Gg.detach())
    dG_k = torch.cat([dGg_k, dGu_k], dim=-1)                # [T,E,2I]
    per_rows1 = [dG_k[r, e, :] for e, r in enumerate(rows)]
    dX_k, dX_deq, sl3 = run_m_grouped(per_rows1, W1.detach(), M_ALIGN)
    dX_k_valid = torch.cat([dX_k[s:s + counts[e]] for e, (s, _) in enumerate(sl3)])
    dX_deq_valid = torch.cat([dX_deq[s:s + counts[e]] for e, (s, _) in enumerate(sl3)])
    print(f'    m_total={dX_k.shape[0]}   vs deq relerr={rel(dX_k_valid, dX_deq_valid):.3e} '
          f'cos={cos(dX_k_valid, dX_deq_valid):.6f}')
    dX_ana = torch.cat([torch.einsum('to,ho->th', dG[r, e, :], W1.detach()[e])
                        for e, r in enumerate(rows)])
    print(f'    vs fp32 analysis dX_e relerr={rel(dX_k_valid, dX_ana):.3e} '
          f'cos={cos(dX_k_valid, dX_ana):.6f}')

    # ---- kernel 4: linear1 wgrad  dW1[e]=x_e^T dG_e (K=token) ----
    print('[4] linear1 wgrad  k_grouped : dW1[e] = x_e^T dG_e   (M=H, N=2I, K=token)')
    per_A1 = [x.detach()[r] for r in rows]
    per_B1 = [dG_k[r, e, :] for e, r in enumerate(rows)]
    dW1_k, dW1_deq, info1 = run_k_grouped(per_A1, per_B1, K_ALIGN)
    print(f'    host_ks={info1[0]} sum_k={info1[1]}   vs deq relerr={rel(dW1_k, dW1_deq):.3e} '
          f'cos={cos(dW1_k, dW1_deq):.6f}')
    print(f'    vs autograd dW1 relerr={rel(dW1_k, c["dW1_ref"]):.3e} '
          f'cos={cos(dW1_k, c["dW1_ref"]):.6f}')

    # ---- end to end: dW2, dW1, and dx_ffn = scatter-add of dX_e ----
    print('=' * 96)
    print('end-to-end (四个 kernel 拼回 dW1/dW2/dX_ffn)')
    dx_ffn_k = torch.zeros(T, H, device=device)
    for e, (s, _) in enumerate(sl3):
        dx_ffn_k[rows[e]] += dX_k[s:s + counts[e]].float()
    print(f'  dW2 kernel vs autograd : relerr={rel(dW2_k, c["dW2_ref"]):.3e} '
          f'cos={cos(dW2_k, c["dW2_ref"]):.6f}')
    print(f'  dW1 kernel vs autograd : relerr={rel(dW1_k, c["dW1_ref"]):.3e} '
          f'cos={cos(dW1_k, c["dW1_ref"]):.6f}')
    # dx has FFN part (kernels) + router part (torch). Compare the FFN part to autograd's
    # dx minus the analytically-known router part.
    dx_router = _router_dx(c['p'], w, sel, c['ti'], Y.detach(), dz, Wg.detach())
    dx_ffn_ref = c['dx_ref'] - dx_router
    print(f'  dX_ffn kernel vs autograd: relerr={rel(dx_ffn_k, dx_ffn_ref):.3e} '
          f'cos={cos(dx_ffn_k, dx_ffn_ref):.6f}')
    print(f'  full dx: (kernel dx_ffn + torch router) vs autograd: '
          f'relerr={rel(dx_ffn_k + dx_router, c["dx_ref"]):.3e} '
          f'cos={cos(dx_ffn_k + dx_router, c["dx_ref"]):.6f}')

    print('=' * 96)
    print('吞吐（DeepSeek 量级单 rank 反向，仅四个 grouped GEMM kernel 本体）')
    try:
        perf()
    except Exception:
        print('  perf FAILED:\n' + traceback.format_exc())


def _router_dx(p, w, sel, ti, Y, dz, Wg):
    """router/topk 反向对 x 的贡献（与 P4-1 同一推导，fp32）。"""
    T = p.shape[0]
    K = ti.shape[1]
    q = torch.einsum('teh,th->te', Y, dz)
    P = (p * sel).sum(-1, keepdim=True)
    Pmat = torch.zeros_like(p).scatter(1, ti, P.expand(T, K))
    qS = q * sel
    A_sum = (qS * p).sum(-1, keepdim=True).expand_as(p)
    dp = torch.where(sel > 0, (q * Pmat - A_sum) / (Pmat ** 2), torch.zeros_like(p))
    dlogits = p * (dp - (p * dp).sum(-1, keepdim=True))
    return dlogits @ Wg.t()


if __name__ == '__main__':
    main()
