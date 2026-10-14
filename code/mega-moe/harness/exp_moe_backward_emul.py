"""P101-1: 在 H100 上把 **MoE 的反向** 逐段搭起来并做「误差来源分解」——P89-1（前向复刻）的对偶。

第 14 篇（P4-4）回答了「四个 grouped GEMM kernel 接进反向图**是否正确**」（`vs deq` ~2.6e-4）。
但**反向误差从哪来、是否保真梯度方向**从没被量化过。本篇补上：

  A. **真实 kernel 锚点**：用 SM90 `k_grouped_fp8_gemm_nt_contiguous`（wgrad）与
     `m_grouped_fp8_gemm_nt_contiguous`（dgrad）跑一遍反向，和 fp64 精确梯度对拍，确认「量化 +
     kernel」总误差量级，并复现 P4-4 的 kernel 误差（`vs deq`）。
  B. **量化来源分解（架构无关）**：把四个 GEMM 的每个操作数按真实语义量化
     （激活 per-token 1D、权重 per-128 块 2D、上游梯度 dY per-token），在 **去量化后的操作数**上
     做精确 GEMM。于是可以把反向误差**可加地**拆成：
        * 前向量化（L1/L2 的操作数）
        * 反向激活量化（dG / dY / a / x）
        * 反向权重量化（W1 / W2）
        * 上游梯度 dY 自身的量化
     并逐张量给出 relerr 与 cosine（梯度方向）。
  C. **与前向对照**：P89 量到前向端到端 ~6.3e-2；本篇回答反向是多少、差在哪。

参考梯度用 fp64（手写反向 + `torch.autograd.grad` 交叉验证；P4-1 已证公式）。所有数字来自
`harness/exp_moe_backward_emul.{py,sh,out.txt}`（H100，DeepGEMM 2.8.1）。

约束：SM100（1D1D + tcgen05 + packed UE8M0）为**仅分析**；本机 H100 走 SM90 的 1D2D/nt。
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
              os.path.expanduser('~/github/DeepGEMM')):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


DEEPGEMM_DIR = _find_deepgemm_dir()
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_block_cast_to_fp8,
    per_channel_cast_to_fp8, get_mk_alignment_for_contiguous_layout,
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


# --------------------------------------------------------------------------- quantization
def q_act(A, mode, dtype=torch.float32):
    """激活类操作数（行 per-token 1D scale）。mode ∈ {'none','token'}。返回反量化后的值。"""
    if mode == 'none' or A.numel() == 0:
        return A.to(dtype)
    x8, sf = per_token_cast_to_fp8(A.float(), use_ue8m0=False, gran_k=GRAN_K)
    return (x8.float() * sf.repeat_interleave(GRAN_K, dim=1)).to(dtype)


def q_weight(W, mode, dtype=torch.float32):
    """权重类操作数（per-128 块 2D scale）。mode ∈ {'none','block'}。返回反量化后的值。"""
    if mode == 'none':
        return W.to(dtype)
    x8, sf = per_block_cast_to_fp8(W.float(), use_ue8m0=False, gran_k=GRAN_K)
    deq = x8.float() * sf.repeat_interleave(GRAN_K, dim=0).repeat_interleave(GRAN_K, dim=1)
    return deq.to(dtype)


# --------------------------------------------------------------------------- model
def build_case(T, H, I, E, KTOP, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(T, H, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    Wg = torch.randn(H, E, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    W1 = torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    W2 = torch.randn(E, H, I, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    dY = torch.randn(T, H, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    p = torch.softmax(x.float() @ Wg.float(), dim=-1)
    tw, ti = torch.topk(p, KTOP, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(dtype)
    return x, W1, W2, dY, tw, ti


def ffn_forward(x, W1, W2, tw, ti, I, E, fwd_act='none', fwd_w='none', dtype=torch.float32):
    """逐专家前向；返回 y 与每专家的前向中间量 acts。权重 tw 作为常量（detach 后的）。"""
    T, H = x.shape
    y = x.new_zeros(T, H)
    acts = {}
    for e in range(E):
        mask = (ti == e)
        if not mask.any():
            acts[e] = None
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        xe = x[t_idx]
        W1e_q = q_weight(W1[e], fwd_w, dtype)
        ge = q_act(xe, fwd_act, dtype) @ W1e_q[:I].T
        ue = q_act(xe, fwd_act, dtype) @ W1e_q[I:].T
        w = tw[t_idx, k_idx].to(dtype).unsqueeze(1)
        ae = silu(ge) * ue * w
        W2e_q = q_weight(W2[e], fwd_w, dtype)
        ze = q_act(ae, fwd_act, dtype) @ W2e_q.T
        y = y.index_add(0, t_idx, ze)
        acts[e] = dict(t=t_idx, k=k_idx, x=xe, g=ge, u=ue, w=w, a=ae, z=ze)
    return y, acts


def ffn_backward(dY, acts, W1, W2, I, E, dtype=torch.float32,
                 qdy='none', qbwd_act='none', qbwd_w='none'):
    """在（可选量化后的）操作数上做反向；返回 dX, dW1, dW2。

    权重约定：W1=[E,2I,H]、W2=[E,H,I]（于是 g=x@W1[:I]^T，z=a@W2^T）。
    四个 GEMM 的角色（见第 14 篇表）：
      L2 dgrad  dA  = dY @ W2      (A=激活 dY, B=权重 W2)
      L1 dgrad  dX  = dG @ W1      (A=激活 dG, B=权重 W1)
      L2 wgrad  dW2 = dY^T a       (A/B 都激活，结果 [H,I])
      L1 wgrad  dW1 = dG^T x       (A/B 都激活，结果 [2I,H])
    SwiGLU 反向用前向的 g,u,w；combine 反向只是 dY 直传（无权重）。
    """
    T = dY.shape[0]
    H = W2.shape[1]
    dX = torch.zeros(T, H, device=dY.device, dtype=dtype)
    dW1 = torch.zeros_like(W1, dtype=dtype)
    dW2 = torch.zeros_like(W2, dtype=dtype)
    for e in range(E):
        a = acts[e]
        if a is None:
            continue
        t_idx, k_idx = a['t'], a['k']
        dYe = dY[t_idx].to(dtype)
        W2e_q = q_weight(W2[e], qbwd_w, dtype)
        W1e_q = q_weight(W1[e], qbwd_w, dtype)
        dYe_q = q_act(dYe, qdy, dtype)
        # L2 dgrad: dA = dZ @ W2   (W2=[H,I])
        dA = dYe_q @ W2e_q
        # L2 wgrad: dW2 = dY^T a   (W2=[H,I]；a 的量化走 qbwd_act)
        a_q = q_act(a['a'].to(dtype), qbwd_act, dtype)
        dW2[e] = dYe_q.T @ a_q
        # SwiGLU 反向（用前向中间量；前向已按 qfwd_* 量化）
        g, u, w = a['g'].to(dtype), a['u'].to(dtype), a['w'].to(dtype)
        dg = dA * (w * silu_grad(g) * u)
        du = dA * (w * silu(g))
        dG = torch.cat([dg, du], dim=1)
        # L1 dgrad: dX = dG @ W1   (W1=[2I,H])
        dG_q = q_act(dG, qbwd_act, dtype)
        dXe = dG_q @ W1e_q
        dX = dX.index_add(0, t_idx, dXe)
        # L1 wgrad: dW1 = dG^T x   (W1=[2I,H])
        x_q = q_act(a['x'].to(dtype), qbwd_act, dtype)
        dW1[e] = dG_q.T @ x_q
    return dX, dW1, dW2


# --------------------------------------------------------------------------- real kernel path
def quant_k_operand(X, host_ks):
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


def run_k_grouped(per_A, per_B, ka):
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
    deq = torch.empty_like(d)
    off = 0
    for e in range(len(ks)):
        hk = host_ks[e]
        deq[e] = a[2][off:off + hk].T @ b[2][off:off + hk]
        off += hk
    return d, deq


def run_m_grouped(per_A_rows, B_group, ma):
    ts = [a.shape[0] for a in per_A_rows]
    K = per_A_rows[0].shape[1]
    N = B_group.shape[1]
    slices, m_total = [], 0
    for t in ts:
        slices.append((m_total, m_total + align(t, ma)))
        m_total += align(t, ma)
    A = torch.zeros(m_total, K, device=per_A_rows[0].device, dtype=torch.float32)
    layout = torch.full((m_total,), -1, device=A.device, dtype=torch.int32)
    for e, (s, _) in enumerate(slices):
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
    deq = torch.empty(m_total, N, device=A.device, dtype=torch.float32)
    for e, (s, _) in enumerate(slices):
        deq[s:s + ts[e]] = deq_a[s:s + ts[e]] @ deq_b[e].T
    return d, deq, slices


def kernel_backward(x, W1, W2, dY, tw, ti, I, E, device):
    """用真实 SM90 grouped kernel 跑四个 GEMM；返回 (dX, dW1, dW2) 与对应 deq 参考。"""
    ma = get_mk_alignment_for_contiguous_layout()
    ka = GRAN_K
    T, KTOP = ti.shape
    # 前向（torch，生成 kernel 输入所需的 a/g/u/x）
    y, acts = ffn_forward(x, W1, W2, tw, ti, I, E)
    # 收集每专家的量
    xs, As, Gs, dGs, dYs = [], [], [], [], []
    for e in range(E):
        a = acts[e]
        if a is None:
            xs.append(x.new_zeros(0, x.shape[1])); As.append(x.new_zeros(0, I))
            Gs.append(x.new_zeros(0, 2 * I)); dGs.append(x.new_zeros(0, 2 * I))
            dYs.append(x.new_zeros(0, dY.shape[1]))
            continue
        t_idx = a['t']
        xs.append(a['x']); As.append(a['a']); Gs.append(torch.cat([a['g'], a['u']], 1))
        dYe = dY[t_idx]
        dYs.append(dYe)
        W2e = W2[e]
        dA = dYe @ W2e
        g, u, w = a['g'], a['u'], a['w']
        dg = dA * (w * silu_grad(g) * u)
        du = dA * (w * silu(g))
        dGs.append(torch.cat([dg, du], 1))

    # --- L2 wgrad: dW2[e] = dY^T a  (W2=[H,I], K=token 分组) ---
    dW2, dW2_deq = run_k_grouped(dYs, As, ka)
    # --- L1 wgrad: dW1[e] = dG^T x  (W1=[2I,H]) ---
    dW1, dW1_deq = run_k_grouped(dGs, xs, ka)
    # --- L2 dgrad: dA = dY @ W2 (M=token 分组；run_m_grouped 算 A@B^T，故传 W2^T=[I,H]) ---
    dA_k, dA_deq, slices = run_m_grouped(dYs, W2.transpose(1, 2).contiguous(), ma)
    # --- L1 dgrad: dX = dG @ W1 (传 W1^T=[H,2I]) ---
    dX_k, dX_deq, _ = run_m_grouped(dGs, W1.transpose(1, 2).contiguous(), ma)
    # dX_k 已是逐专家 dgrad 结果；按 token scatter-add 回 [T,H]
    dX = torch.zeros(T, W1.shape[2], device=device, dtype=torch.float32)
    dX_deq_full = torch.zeros(T, W1.shape[2], device=device, dtype=torch.float32)
    for e in range(E):
        a = acts[e]
        if a is None:
            continue
        s = slices[e][0]
        n = a['t'].shape[0]
        dX.index_add_(0, a['t'], dX_k[s:s + n].float())
        dX_deq_full.index_add_(0, a['t'], dX_deq[s:s + n])
    return (dX, dW1, dW2), (dX_deq_full, dW1_deq, dW2_deq)


# --------------------------------------------------------------------------- one case (decomposition)
def run_case(T, H, I, E, KTOP, device, do_kernel=False, do_perf=False):
    dtype64 = torch.float64
    x, W1, W2, dY, tw, ti = build_case(T, H, I, E, KTOP, device)

    # 精确 fp64 参考：手写反向（前向/反向都不量化）
    x64, W164, W264, dY64 = x.double(), W1.double(), W2.double(), dY.double()
    _, acts64 = ffn_forward(x64, W164, W264, tw.double(), ti, I, E, dtype=dtype64)
    dXr, dW1r, dW2r = ffn_backward(dY64, acts64, W164, W264, I, E, dtype=dtype64)
    # autograd 交叉验证（同一 y*(dY)）
    xg = x64.clone().requires_grad_(True); W1g = W164.clone().requires_grad_(True)
    W2g = W264.clone().requires_grad_(True)
    y64, _ = ffn_forward(xg, W1g, W2g, tw.double(), ti, I, E, dtype=dtype64)
    ag = torch.autograd.grad((y64 * dY64).sum(), (xg, W1g, W2g))
    ag_err = dict(dX=rel(ag[0], dXr), dW1=rel(ag[1], dW1r), dW2=rel(ag[2], dW2r))

    def run(qf_a, qf_w, qd, qb_a, qb_w):
        _, acts = ffn_forward(x, W1, W2, tw, ti, I, E,
                              fwd_act=qf_a, fwd_w=qf_w, dtype=torch.float32)
        dX, dW1, dW2 = ffn_backward(dY, acts, W1, W2, I, E, dtype=torch.float32,
                                    qdy=qd, qbwd_act=qb_a, qbwd_w=qb_w)
        return dict(dX=rel(dX, dXr), dW1=rel(dW1, dW1r), dW2=rel(dW2, dW2r),
                    cX=cos(dX, dXr), cW1=cos(dW1, dW1r), cW2=cos(dW2, dW2r))

    cases = {
        'none':            run('none', 'none', 'none', 'none', 'none'),
        'fwd_only':        run('token', 'block', 'none', 'none', 'none'),
        '+bwd_act':        run('token', 'block', 'none', 'token', 'none'),
        '+bwd_w':          run('token', 'block', 'none', 'none', 'block'),
        '+dy':             run('token', 'block', 'token', 'none', 'none'),
        'full':            run('token', 'block', 'token', 'token', 'block'),
    }

    res = dict(T=T, H=H, I=I, E=E, KTOP=KTOP, ag_err=ag_err, cases=cases)
    if do_kernel:
        (dk, deqk) = kernel_backward(x, W1, W2, dY, tw, ti, I, E, device)
        res['kernel'] = dict(
            vs_ref=dict(dX=rel(dk[0], dXr), dW1=rel(dk[1], dW1r), dW2=rel(dk[2], dW2r),
                        cX=cos(dk[0], dXr), cW1=cos(dk[1], dW1r), cW2=cos(dk[2], dW2r)),
            vs_deq=dict(dX=rel(dk[0], deqk[0]), dW1=rel(dk[1], deqk[1]), dW2=rel(dk[2], deqk[2])),
        )
    return res


def _bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    device = 'cuda'
    arch = get_arch_major()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={arch}, device={torch.cuda.get_device_name(0)}')
    print('Mega MoE 反向：SM100 主 kernel 不可跑；本实验用真实参考量化器 + SM90 grouped kernel 复刻其反向语义。')
    print('=' * 100)

    print('## 0. 手写 fp64 反向 vs autograd（公式自检）')
    r = run_case(64, 512, 256, 8, 2, device)
    ae = r['ag_err']
    print(f'   autograd vs 手写 fp64: dX rel={ae["dX"]:.2e}  dW1 rel={ae["dW1"]:.2e}  dW2 rel={ae["dW2"]:.2e}')

    print()
    print('## A. 真实 SM90 kernel 反向锚点（配置 64/512/256/8/2）')
    r = run_case(64, 512, 256, 8, 2, device, do_kernel=True)
    k = r['kernel']
    print(f'   kernel vs fp64 : dX rel={k["vs_ref"]["dX"]:.4e} cos={k["vs_ref"]["cX"]:.6f} | '
          f'dW1 rel={k["vs_ref"]["dW1"]:.4e} cos={k["vs_ref"]["cW1"]:.6f} | '
          f'dW2 rel={k["vs_ref"]["dW2"]:.4e} cos={k["vs_ref"]["cW2"]:.6f}')
    print(f'   kernel vs deq  : dX rel={k["vs_deq"]["dX"]:.4e} | dW1 rel={k["vs_deq"]["dW1"]:.4e} | '
          f'dW2 rel={k["vs_deq"]["dW2"]:.4e}  (kernel+布局误差，应 ~1e-4..1e-3)')

    print()
    print('## B. 量化来源分解（架构无关；每行叠加一个反向来源，per-tensor relerr / cos）')
    smalls = [(64, 512, 256, 8, 2), (256, 1024, 512, 8, 2), (512, 1024, 512, 16, 4)]
    for (T, H, I, E, K) in smalls:
        r = run_case(T, H, I, E, K, device)
        print(f'-- T={T} H={H} I={I} E={E} topk={K}')
        for name, c in r['cases'].items():
            print(f'   {name:9s} dX rel={c["dX"]:.4e} cos={c["cX"]:.6f} | '
                  f'dW1 rel={c["dW1"]:.4e} cos={c["cW1"]:.6f} | '
                  f'dW2 rel={c["dW2"]:.4e} cos={c["cW2"]:.6f}')

    print()
    print('## C. DeepSeek 量级（T=2048,H=7168,I=2048,E=8,topk=2）：分解 + 真实 kernel 吞吐')
    T, H, I, E, K = 2048, 7168, 2048, 8, 2
    r = run_case(T, H, I, E, K, device)
    for name, c in r['cases'].items():
        print(f'   {name:9s} dX rel={c["dX"]:.4e} cos={c["cX"]:.6f} | '
              f'dW1 rel={c["dW1"]:.4e} cos={c["cW1"]:.6f} | '
              f'dW2 rel={c["dW2"]:.4e} cos={c["cW2"]:.6f}')

    print()
    print('ALL DONE')


if __name__ == '__main__':
    main()
