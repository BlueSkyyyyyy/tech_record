"""P151-1 实验：8×H100 上的【多卡 EP MoE + 真实 SM90 FP8 grouped kernel】端到端误差与预算。

补的缺口（承接第 141 篇 docs/141-ep-e2e-moe.md「下一轮可做」）：
  * 第 141 篇（P131-1）把 EP 接进完整 MoE 图、量了四组梯度，但计算腿是**纯 torch matmul**——
    没有真实 FP8 kernel，也没有 FP8 量化误差；
  * 第 14 篇（P4-4）把真实 SM90 FP8 grouped kernel 接进了完整 MoE 反向，但只是**单卡、无通信**；
  * 第 99 篇前向复刻把通信折算成单卡 gather/scatter（零通信）。
  「EP 通信 + 真实 FP8 grouped kernel 量化」从没在一条端到端 MoE 图里叠加过。

本篇把第 141 篇的 8×H100 EP 图（NCCL `all_to_all_single` 实现 expand dispatch/combine）的
L1/L2 前向、以及 dgrad/wgrad 反向**全部换成真实 DeepGEMM SM90 FP8 grouped kernel**，
与单卡稠密 fp64 参考对拍，量：
  (1) 前向输出误差三层分解（kernel+布局+通信 / 纯 FP8 量化 / 总计）；
  (2) 反向 dW1/dW2/dX（固定 routing，不训 router）误差；
  (3) 每个 grouped GEMM 的「vs deq」隔离（kernel+布局）误差。

可跑边界：纯 PyTorch + NCCL a2a + 真实 DeepGEMM SM90 kernel（sm90 有实现），H100 真机可跑；
Mega MoE 融合 kernel 本体（SM100）仅分析。

源码坐标（`~/github/DeepGEMM@057ca59`、`~/github/DeepEP@93eb6eb`，行号核对过）：
  * 前向/dgrad（M 轴分组）`csrc/apis/gemm.hpp:181-250`，`m_grouped_fp8_gemm_nt_contiguous`
  * wgrad（K 轴分组）`csrc/apis/gemm.hpp:373-427`，`k_grouped_fp8_gemm_nt_contiguous`
  * SM90 输出形态：1D1D 必 fp32 且累积 `sm90_fp8_gemm_1d1d.hpp:84`；1D2D 必 bf16 直写 `sm90_fp8_gemm_1d2d.hpp:81`
  * dispatch/combine 参考 `deep_ep/utils/refs.py:10-123/177-243`、`README.md:204-220/242-258`
  * 本专题：docs/141（EP e2e 纯 torch）、docs/14（单卡 FP8 反向）、docs/104（反向通信量）

运行：bash scripts/lab.sh exec "EPFP8_WORLD=8 python /abs/path/harness/exp_ep_fp8_e2e_moe.py"
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import (align, per_token_cast_to_fp8, per_channel_cast_to_fp8,
                             per_block_cast_to_fp8, get_mk_alignment_for_contiguous_layout)

WORLD = int(os.environ.get('EPFP8_WORLD', '8'))
GRAN_K = 128
M_ALIGN = get_mk_alignment_for_contiguous_layout()


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def silu(z):
    return z * torch.sigmoid(z)


def silu_grad(z):
    s = torch.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


# ----------------------------------------------------------------------------------
# grouped GEMM 封装（自足，复制自第 14 篇 harness 的量化/布局口径）
# ----------------------------------------------------------------------------------
def quant_k_operand(X, host_ks):
    """X [total_k, mn]：各组位于累计 host_ks（gap 行 0）。返回 (data 1D fp8, sf [mn,tot/gran], deq)。"""
    total_k, mn = X.shape
    data = torch.zeros(total_k * mn, dtype=torch.float8_e4m3fn, device=X.device)
    deq = torch.empty_like(X)
    sfs, off = [], 0
    for hk in host_ks:
        xg = X[off:off + hk].contiguous()
        xf8, sf = per_channel_cast_to_fp8(xg, use_ue8m0=False, gran_k=GRAN_K)
        data[off * mn:(off + hk) * mn] = xf8.T.reshape(-1)
        deq[off:off + hk] = xf8.float() * sf.repeat_interleave(GRAN_K, dim=0)
        sfs.append(sf)
        off += hk
    sf_all = torch.cat(sfs).T if sfs else X.new_zeros((mn, 0))
    return data, sf_all, deq


def run_k_grouped(per_A, per_B):
    """wgrad：per_A[e] [t_e, M]、per_B[e] [t_e, N]，收缩维 K=token。返回 D[e][M,N]、deq 参考。"""
    ks = [a.shape[0] for a in per_A]
    host_ks = [align(k, GRAN_K) for k in ks]
    total_k = sum(host_ks)
    M, N = per_A[0].shape[1], per_B[0].shape[1]
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
    ref, off = torch.empty_like(d), 0
    for e, k in enumerate(ks):
        hk = host_ks[e]
        ref[e] = a[2][off:off + hk].T @ b[2][off:off + hk]
        off += hk
    return d, ref


def _m_slices(ts):
    out, off = [], 0
    for t in ts:
        out.append((off, off + align(t, M_ALIGN)))
        off += align(t, M_ALIGN)
    return out, off


def run_m_grouped(per_rows, B_group):
    """前向/dgrad：per_rows[e] [t_e, K]（M 轴分组），B_group[e] [N, K]。返回 D[tot,N] bf16、deq 参考。"""
    ts = [a.shape[0] for a in per_rows]
    K, N = per_rows[0].shape[1], B_group.shape[1]
    slices, m_total = _m_slices(ts)
    A = torch.zeros(m_total, K, device=per_rows[0].device, dtype=torch.float32)
    layout = torch.full((m_total,), -1, device=A.device, dtype=torch.int32)
    for e, (s, _) in enumerate(slices):
        A[s:s + ts[e]] = per_rows[e]
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
    for e, (s, _) in enumerate(slices):
        ref[s:s + ts[e]] = deq_a[s:s + ts[e]] @ deq_b[e].T
    return d, ref, slices


# ----------------------------------------------------------------------------------
# 单卡稠密 fp64 参考（同 routing；可分别用原始 / FP8 反量化操作数）
# ----------------------------------------------------------------------------------
def _deq_token(x):
    f8, sf = per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=GRAN_K)
    return f8.float() * sf.repeat_interleave(GRAN_K, dim=1)


def _deq_block(w):
    f8, sf = per_block_cast_to_fp8(w, use_ue8m0=False, gran_k=GRAN_K)
    return f8.float() * sf.repeat_interleave(GRAN_K, 0).repeat_interleave(GRAN_K, 1)


def dense_forward(x, W1, W2, idx, w, topk, inter, round_l1=False):
    T = x.size(0)
    out = torch.zeros(T, x.size(1), dtype=torch.float64, device=x.device)
    for k in range(topk):
        e_all = idx[:, k]
        for ex in e_all.unique().tolist():
            m = e_all == ex
            g = x[m].double() @ W1[ex].double()
            if round_l1:
                g = g.float().bfloat16().double()
            gate, up = g[:, :inter], g[:, inter:]
            y = (silu(gate) * up) @ W2[ex].double()
            out[m] += w[m, k].double()[:, None] * y
    return out


def dense_backward_ref(x, W1, W2, idx, w, topk, inter, dout):
    x = x.detach().clone().double().requires_grad_(True)
    W1 = W1.detach().clone().double().requires_grad_(True)
    W2 = W2.detach().clone().double().requires_grad_(True)
    out = dense_forward(x, W1, W2, idx, w, topk, inter)
    g = torch.autograd.grad((out * dout.double()).sum(), (x, W1, W2))
    return g[0], g[1], g[2]


# ----------------------------------------------------------------------------------
# EP 分布式通信：expand dispatch / combine（docs/141 同口径，NCCL all_to_all_single）
# ----------------------------------------------------------------------------------
def _gather_send(x, idx, w, rank, T, topk, per_rank, world):
    """按目的 rank 收集发送行。返回 (sx,se,sw,slocal), send_counts；slocal=本地源 token 号（发送序）。"""
    H = x.size(1)
    dest = idx // per_rank
    gsrc = (rank * T) + torch.arange(T, device=x.device)[:, None].expand(T, topk)
    sx, se, sw, sl = [], [], [], []
    send_counts = torch.zeros(world, dtype=torch.long, device=x.device)
    for d in range(world):
        m = dest == d
        cnt = int(m.sum().item())
        send_counts[d] = cnt
        if cnt == 0:
            continue
        sx.append(x[:, None, :].expand(T, topk, H)[m])
        se.append(idx[m] - d * per_rank)
        sw.append(w[m])
        sl.append(gsrc[m] - rank * T)
    cat = lambda L, shape, dt: (torch.cat(L, 0) if L else torch.empty((0,) + shape[1:], dtype=dt, device=x.device))
    return (cat(sx, (0, H), x.dtype), cat(se, (0,), torch.long),
            cat(sw, (0,), w.dtype), cat(sl, (0,), torch.long)), send_counts


def _a2a(t, rc, sc):
    rc, sc = rc.tolist(), sc.tolist()
    out = t.new_empty((int(sum(rc)),) + tuple(t.shape[1:]))
    dist.all_to_all_single(out, t.contiguous(), output_split_sizes=rc, input_split_sizes=sc)
    return out


def ep_forward_fp8(x, W1_local, W2_local, idx, w, rank, world, topk):
    T, H = x.shape
    per_rank = W1_local.size(0)
    inter = W1_local.size(2) // 2

    (sx, se, sw, sl), send_counts = _gather_send(x, idx, w, rank, T, topk, per_rank, world)
    recv_counts = torch.empty(world, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)
    rx = _a2a(sx, recv_counts, send_counts)
    re = _a2a(se, recv_counts, send_counts)
    rw = _a2a(sw, recv_counts, send_counts)
    rs = _a2a(sl, recv_counts, send_counts)          # 源 rank 的本地 token 号

    order = torch.argsort(re, stable=True)
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=x.device)
    xe, we = rx[order], rw[order]
    re_s = re[order]
    counts = torch.bincount(re_s, minlength=per_rank)
    rows = [torch.nonzero(re_s == e).flatten() for e in range(per_rank)]

    # 前向 D = A @ W^T：B_group 需为 [N,K]（W1_local[e]=[H,2I] → [2I,H]；W2_local[e]=[I,H] → [H,I]）
    W1t = W1_local.transpose(1, 2).contiguous()
    W2t = W2_local.transpose(1, 2).contiguous()
    d1, d1_deq, sl1 = run_m_grouped([xe.float()[rows[e]] for e in range(per_rank)], W1t)
    d1v = torch.cat([d1[s:s + counts[e]] for e, (s, _) in enumerate(sl1)])
    gate, up = d1v[:, :inter].float(), d1v[:, inter:].float()
    act = silu(gate) * up
    d2, d2_deq, sl2 = run_m_grouped([act[rows[e]] for e in range(per_rank)], W2t)
    d2v = torch.cat([d2[s:s + counts[e]] for e, (s, _) in enumerate(sl2)])

    y = (d2v.float() * we[:, None])[inv]
    ry = _a2a(y, send_counts, recv_counts)
    rs_back = _a2a(rs, send_counts, recv_counts)
    out_global = torch.zeros(world * T, H, dtype=torch.float32, device=x.device)
    out_global.index_add_(0, rs_back, ry)
    ctx = dict(xe=xe, we=we, re=re, order=order, inv=inv, counts=counts, rows=rows,
               gate=gate, up=up, act=act, send_counts=send_counts, recv_counts=recv_counts, rs=rs)
    return out_global[rank * T:(rank + 1) * T], ctx


def ep_backward_fp8(dout_local, W1_local, W2_local, idx, w, rank, world, topk, ctx):
    T, H = dout_local.shape
    per_rank = W1_local.size(0)
    send_counts, recv_counts = ctx['send_counts'], ctx['recv_counts']

    # combine 的伴随 = 带权重的 dispatch（docs/141：两个原语互为转置）
    (sdy, sse, ssw, _), sc2 = _gather_send(dout_local, idx, w, rank, T, topk, per_rank, world)
    rc2 = torch.empty(world, dtype=torch.long, device=dout_local.device)
    dist.all_to_all_single(rc2, sc2)
    rdY = _a2a(sdy, rc2, sc2)
    reg = _a2a(sse, rc2, sc2)
    rwg = _a2a(ssw, rc2, sc2)
    order = torch.argsort(reg, stable=True)
    dY = (rdY * rwg[:, None])[order].float()

    rows, counts = ctx['rows'], ctx['counts']
    act, xe, gate, up = ctx['act'], ctx['xe'].float(), ctx['gate'], ctx['up']

    dW2, dW2_deq = run_k_grouped([act[rows[e]] for e in range(per_rank)],
                                 [dY[rows[e]] for e in range(per_rank)])
    dact, dact_deq, sl2 = run_m_grouped([dY[rows[e]] for e in range(per_rank)], W2_local)
    dactv = torch.cat([dact[s:s + counts[e]] for e, (s, _) in enumerate(sl2)]).float()

    dgate = dactv * silu_grad(gate) * up
    dup = dactv * silu(gate)
    dG = torch.cat([dgate, dup], dim=-1)

    dW1, dW1_deq = run_k_grouped([xe[rows[e]] for e in range(per_rank)],
                                 [dG[rows[e]] for e in range(per_rank)])
    dxg, dxg_deq, sl1 = run_m_grouped([dG[rows[e]] for e in range(per_rank)], W1_local)
    dxv = torch.cat([dxg[s:s + counts[e]] for e, (s, _) in enumerate(sl1)]).float()

    # dispatch 的伴随 = combine(dx) 收回源 token（用 ctx['rs'] 的本地源号）
    rdx = _a2a(dxv[ctx['inv']], send_counts, recv_counts)
    rs_back = _a2a(ctx['rs'], send_counts, recv_counts)
    dx_local = torch.zeros(T, H, dtype=torch.float32, device=dout_local.device)
    dx_local.index_add_(0, rs_back, rdx)
    return dict(dW1=dW1, dW2=dW2, dx=dx_local, md=dict(dW1_deq=rel(dW1, dW1_deq), dW2_deq=rel(dW2, dW2_deq),
            dact_deq=rel(dactv, torch.cat([dact_deq[s:s + counts[e]] for e, (s, _) in enumerate(sl2)]).float()),
            dx_deq=rel(dxv, torch.cat([dxg_deq[s:s + counts[e]] for e, (s, _) in enumerate(sl1)]).float())))


# ----------------------------------------------------------------------------------
# 进程主体
# ----------------------------------------------------------------------------------
def _init(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=__import__('datetime').timedelta(seconds=300))
    return torch.device('cuda', rank)


def build_case(T, H, I, E, topk):
    g = torch.Generator(device='cpu').manual_seed(1234)
    W1 = torch.randn(E, H, 2 * I, generator=g, dtype=torch.float32) * (H ** -0.5)
    W2 = torch.randn(E, I, H, generator=g, dtype=torch.float32) * (I ** -0.5)
    x = torch.randn(T, H, generator=g, dtype=torch.float32)
    lg = torch.randn(T, E, generator=g, dtype=torch.float32)
    tw, idx = torch.topk(torch.softmax(lg, dim=-1), topk, dim=-1)
    return x, W1, W2, idx, tw / tw.sum(-1, keepdim=True)


def _gmax(t):
    v = t.detach().float().reshape(-1)[:1].clone()
    dist.all_reduce(v, op=dist.ReduceOp.MAX)
    return float(v)


def run_case(rank, world, tag, T, H, I, E, topk):
    per_rank = E // world
    Tg = world * T                                   # 全局 token 数（每 rank T 个）
    xg, W1, W2, idxg, wg = build_case(Tg, H, I, E, topk)
    xg, W1, W2, idxg, wg = xg.cuda(), W1.cuda(), W2.cuda(), idxg.cuda(), wg.cuda()
    sl = slice(rank * T, (rank + 1) * T)
    x, idx, w = xg[sl], idxg[sl], wg[sl]
    W1_local = W1[rank * per_rank:(rank + 1) * per_rank].contiguous()
    W2_local = W2[rank * per_rank:(rank + 1) * per_rank].contiguous()

    out_ep, ctx = ep_forward_fp8(x, W1_local, W2_local, idx, w, rank, world, topk)
    out_ep_d = out_ep.detach().float()

    x_deq = _deq_token(xg)
    W1_deq = torch.stack([_deq_block(W1[e]) for e in range(E)])
    W2_deq = torch.stack([_deq_block(W2[e]) for e in range(E)])
    out_ref = dense_forward(xg, W1, W2, idxg, wg, topk, I).float()
    out_sim = dense_forward(x_deq, W1_deq, W2_deq, idxg, wg, topk, I, round_l1=True).float()

    # 反向：全局 dz（确定性），每 rank 取本地切片
    gz = torch.Generator(device='cpu').manual_seed(777)
    dz_global = torch.randn(Tg, H, generator=gz, dtype=torch.float32).cuda()
    gp = ep_backward_fp8(dz_global[sl], W1_local, W2_local, idx, w, rank, world, topk, ctx)

    dx_ref, dW1_ref, dW2_ref = dense_backward_ref(xg, W1, W2, idxg, wg, topk, I, dz_global)
    dx_ref_deq, dW1_ref_deq, dW2_ref_deq = dense_backward_ref(x_deq, W1_deq, W2_deq, idxg, wg, topk, I, dz_global)

    dW1_all = torch.empty(E, H, 2 * I, dtype=torch.float32, device=xg.device)
    dist.all_gather(list(dW1_all.view(world, -1)), gp['dW1'].reshape(-1))
    dW2_all = torch.empty(E, I, H, dtype=torch.float32, device=xg.device)
    dist.all_gather(list(dW2_all.view(world, -1)), gp['dW2'].reshape(-1))

    md = gp['md']
    if rank == 0:
        print(f'--- [{tag}] world={world} T_local={T} T_global={Tg} H={H} I={I} E={E} topk={topk} ---')
        print(f'  前向  ep vs fp64    rel={rel(out_ep_d, out_ref[sl]):.3e}  cos={cos(out_ep_d, out_ref[sl]):.6f}')
        print(f'        ep vs fp8模拟 rel={rel(out_ep_d, out_sim[sl]):.3e}  cos={cos(out_ep_d, out_sim[sl]):.6f}   '
              f'(隔离 kernel+布局+通信)')
        print(f'        fp8模拟 vs fp64 rel={rel(out_sim, out_ref):.3e}   (纯 FP8 量化噪声)')
        print(f'  反向  dW1 ep vs fp64    rel={rel(dW1_all, dW1_ref):.3e}  cos={cos(dW1_all, dW1_ref):.6f}')
        print(f'        dW2 ep vs fp64    rel={rel(dW2_all, dW2_ref):.3e}  cos={cos(dW2_all, dW2_ref):.6f}')
        print(f'        dX  ep vs fp64    rel={rel(gp["dx"], dx_ref[sl]):.3e}  cos={cos(gp["dx"], dx_ref[sl]):.6f}')
        print(f'        dW1 ep vs fp8模拟 rel={rel(dW1_all, dW1_ref_deq):.3e}  '
              f'dW2 ep vs fp8模拟 rel={rel(dW2_all, dW2_ref_deq):.3e}')
    m = torch.tensor([md['dW1_deq'], md['dW2_deq'], md['dact_deq'], md['dx_deq']], device=xg.device)
    dist.all_reduce(m, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f'  grouped vs deq (全局最大): dW1={m[0]:.3e} dW2={m[1]:.3e} dact={m[2]:.3e} dx={m[3]:.3e}')
    return rel(out_ep_d, out_ref[sl])


def _bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3            # us


def _time_m_grouped(A, B):
    """A [m_total,K] fp32（M 轴分组）、B [E,N,K] fp32 → 预量化后只计时 kernel。返回 (us, D)。"""
    E, N, K = B.shape
    m_total = A.shape[0]
    layout = torch.full((m_total,), -1, dtype=torch.int32, device=A.device)
    off = 0
    for e in range(E):
        t = _COUNTS[e]
        layout[off:off + t] = e
        off += align(t, M_ALIGN)
    a_f8, a_sf = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN_K)
    b_f8 = torch.empty(E, N, align(K, GRAN_K), dtype=torch.float8_e4m3fn, device=A.device)
    b_sf = torch.empty(E, align(N, GRAN_K) // GRAN_K, align(K, GRAN_K) // GRAN_K,
                       device=A.device, dtype=torch.float32)
    for e in range(E):
        b_f8[e, :, :K], b_sf[e] = per_block_cast_to_fp8(B[e], use_ue8m0=False, gran_k=GRAN_K)
    d = torch.empty(m_total, N, dtype=torch.bfloat16, device=A.device)

    def f():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_f8, a_sf), (b_f8, b_sf), d, layout, disable_ue8m0_cast=True)
    return _bench(f, iters=30), d


def _time_k_grouped(A, B):
    """A/B [total_k, mn] fp32（K 轴分组，按 _COUNTS）→ 预量化后只计时 kernel。返回 (us, D)。"""
    E = len(_COUNTS)
    host_ks = [align(k, GRAN_K) for k in _COUNTS]
    total_k, M, N = A.shape[0], A.shape[1], B.shape[1]
    a = quant_k_operand(A, host_ks)
    b = quant_k_operand(B, host_ks)
    layout = torch.tensor(host_ks, device=A.device, dtype=torch.int32)
    d = torch.zeros(E, M, N, device=A.device, dtype=torch.float32)

    def f():
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (a[0], a[1]), (b[0], b[1]), d, host_ks, layout, d, recipe=(1, 1, GRAN_K))
    return _bench(f, iters=30), d


_COUNTS = []


def measure_budget(rank, world):
    """DeepSeek 量级（T_local=2048,H=7168,I=2048,E=64,topk=2）的通信 + FP8 kernel 预算。

    只计时 **grouped GEMM kernel 本体**（操作数预量化，排除 Python 侧 torch 量化开销）与
    a2a 通信；量化开销本身由本实验的 torch 参考量化器近似，不是 DeepGEMM kernel 的成本。
    """
    global _COUNTS
    T, H, I, E, topk = 2048, 7168, 2048, 64, 2
    per_rank = E // world
    g = torch.Generator(device='cpu').manual_seed(7)
    W1l = (torch.randn(per_rank, H, 2 * I, generator=g) * (H ** -0.5)).cuda()
    W2l = (torch.randn(per_rank, I, H, generator=g) * (I ** -0.5)).cuda()
    x = torch.randn(T, H, generator=g).cuda()
    lg = torch.randn(T, E, generator=g)
    tw, idx = torch.topk(torch.softmax(lg, dim=-1), topk, dim=-1)
    w = (tw / tw.sum(-1, keepdim=True)).cuda()
    idx = idx.cuda()

    _, send_counts = _gather_send(x, idx, w, rank, T, topk, per_rank, world)
    (sx, se, sw, sl), send_counts = _gather_send(x, idx, w, rank, T, topk, per_rank, world)
    recv_counts = torch.empty(world, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)
    nrecv = int(recv_counts.sum().item())
    # 本 rank 各本地专家**收到**的行数（用真实 recv 的专家号统计）
    rx0 = _a2a(sx, recv_counts, send_counts)
    re0 = _a2a(se, recv_counts, send_counts)
    counts = torch.bincount(re0, minlength=per_rank)
    _COUNTS = [int(c) for c in counts.tolist()]
    m_pad = sum(align(c, M_ALIGN) for c in _COUNTS)

    # --- 通信 ---
    t_disp = _bench(lambda: _a2a(sx, recv_counts, send_counts))
    rx = _a2a(sx, recv_counts, send_counts)
    t_comb = _bench(lambda: _a2a(rx, send_counts, recv_counts))

    # --- 6 个 grouped GEMM（操作数预量化，只计 kernel）---
    t_l1, _ = _time_m_grouped(torch.randn(m_pad, H, device=x.device), W1l.transpose(1, 2).contiguous())
    t_l2, _ = _time_m_grouped(torch.randn(m_pad, I, device=x.device), W2l.transpose(1, 2).contiguous())
    # dgrad L2: A=dY[m_pad,H], B=W2 [E,I,H]; dgrad L1: A=dG[m_pad,2I], B=W1 [E,H,2I]
    t_d2, _ = _time_m_grouped(torch.randn(m_pad, H, device=x.device), W2l)
    t_d1, _ = _time_m_grouped(torch.randn(m_pad, 2 * I, device=x.device), W1l)
    # wgrad: A/B K 轴分组（total_k = Σ align(counts) = m_pad）
    t_w2, _ = _time_k_grouped(torch.randn(m_pad, I, device=x.device), torch.randn(m_pad, H, device=x.device))
    t_w1, _ = _time_k_grouped(torch.randn(m_pad, H, device=x.device), torch.randn(m_pad, 2 * I, device=x.device))

    labels = ['fwd-L1', 'fwd-L2', 'dgrad-L2', 'dgrad-L1', 'wgrad-L2', 'wgrad-L1']
    vals = [t_l1, t_l2, t_d2, t_d1, t_w2, t_w1]
    flops = [2 * m_pad * H * 2 * I, 2 * m_pad * I * H, 2 * m_pad * H * I, 2 * m_pad * 2 * I * H,
             2 * m_pad * I * H, 2 * m_pad * H * 2 * I]
    allv = torch.tensor(vals, device=x.device)
    dist.all_reduce(allv, op=dist.ReduceOp.MAX)

    if rank == 0:
        allv = allv.tolist()
        print('-' * 92)
        print(f'预算（DeepSeek 量级 T_local={T}, topk={topk}, H={H}, I={I}, E={E}；每 rank {per_rank} 专家）')
        print(f'  每 rank 收到 {nrecv} 行（平均 {nrecv / per_rank:.0f}/本地专家，padding 后 m={m_pad}）')
        print(f'  通信  dispatch a2a {nrecv * H * 4 / 1e6:6.1f} MB(fp32)  {t_disp:7.1f} us      '
              f'combine a2a {nrecv * H * 4 / 1e6:6.1f} MB  {t_comb:7.1f} us')
        for lb, v, fl in zip(labels, allv, flops):
            print(f'  GEMM  {lb:9s} {v:7.1f} us   {fl / (v * 1e-6) / 1e12:6.0f} TFLOPS')
        tf = allv[0] + allv[1]; tb = allv[2] + allv[3] + allv[4] + allv[5]
        print(f'  前向 kernel 合计 {tf:7.1f} us + 通信 {t_disp + t_comb:7.1f} us = {tf + t_disp + t_comb:7.1f} us')
        print(f'  反向 kernel 合计 {tb:7.1f} us + 通信 {t_disp + t_comb:7.1f} us = {tb + t_disp + t_comb:7.1f} us')
        print(f'  （真机 dispatch 为 FP8/1B、combine 为 BF16/2B ⇒ 通信字节 ≈ '
              f'{nrecv * H / 1e6:.1f} + {nrecv * H * 2 / 1e6:.1f} MB/rank；本实验按 fp32 是保守上界）')


def measure_a2a_anchor(rank, world):
    """平衡负载 all_to_all 带宽锚（与 docs/141 第 1 篇同口径，用于对照争用）。"""
    H = 7168
    if rank == 0:
        print('-' * 92)
        print('  a2a 锚（平衡负载，单向 bf16；对照本机 8 卡争用）')
    for T in (512, 2048, 8192):
        inp = torch.ones(world, T, H, dtype=torch.bfloat16, device='cuda')
        out = torch.empty_like(inp)
        t = _bench(lambda: dist.all_to_all_single(out, inp), iters=30)
        b = world * T * H * 2
        if rank == 0:
            print(f'    {T:6d} tok/rank  发 {b / 1e6:7.1f} MB/rank  {t:8.1f} us  {b / 1e9 / (t / 1e6):7.1f} GB/s')


def worker(rank, world):
    _init(rank, world)
    if rank == 0:
        print(f'torch {torch.__version__}  world={world}  nccl={torch.cuda.nccl.version()}  arch_major={get_arch_major()}')
        print('EP + 真实 SM90 FP8 grouped kernel 端到端 MoE（前向/反向）—— H100 真机')
        print('-' * 92)
    run_case(rank, world, 'correctness', T=128, H=512, I=1024, E=64, topk=4)
    torch.cuda.synchronize()
    dist.barrier()
    run_case(rank, world, 'correctness-2', T=64, H=256, I=512, E=32, topk=2)
    torch.cuda.synchronize()
    dist.barrier()
    measure_budget(rank, world)
    dist.barrier()
    measure_a2a_anchor(rank, world)
    dist.barrier()
    if rank == 0:
        print('done')
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(22000 + (os.getpid() % 10000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
