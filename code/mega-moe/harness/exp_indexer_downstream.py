"""P162-1：indexer top-k 选择翻转对【下游稀疏注意力输出】的端到端影响。

第 171 篇（P161-1）量了 indexer 的 top-k 选择在 FP8/MXFP8/MXFP4 量化下的稳定性
（top-2048 recall 0.966/0.965/0.851），但结尾留了口子④：「选择翻转对下游稀疏注意力
输出的端到端影响未量」。本篇把它补上。

模型（自洽 + 失配两档）：
  - indexer 打分 I[t,j] = Σ_h w[t,h]·relu(<q[t,h,:], kv[j]>)（与第 171 篇同，真实 SM90 kernel 锚定）；
  - 选择 S_t = Top-k(I_t,:)（token 级）或块级 max 聚合后 Top-k；
  - DSA 稀疏注意力：把注意力分布限制在支持集上（softmax 只在支持集内归一化）再对 V 加权。
  - 自洽档（下界）：注意力打分 = indexer 打分本身（选择与注意力完全对齐），
    差异只来自「换了支持集」；失配档：注意力打分 a = ρ·I + √(1-ρ²)·σ_row(I)·noise，
    ρ 越小、indexer 与注意力越脱节，翻转越有后果。

全部纯 torch + 真实 SM90 kernel（fp8 档），H100(sm90) 真机可跑。
"""
import argparse
import torch
import deep_gemm
from deep_gemm.utils import (per_token_cast_to_fp8, per_token_cast_to_fp4,
                             cast_back_from_fp8, cast_back_from_fp4,
                             per_custom_dims_cast_to_fp8)


def mqa_logits(q, kv, w, chunk=8192):
    """logits[t,n] = sum_h w[t,h] * relu(<q[t,h,:], kv[n,:]>)，输入均为 fp32。"""
    T, H, D = q.shape
    C = kv.shape[0]
    out = torch.empty((T, C), dtype=torch.float32, device=q.device)
    for n0 in range(0, C, chunk):
        n1 = min(n0 + chunk, C)
        s = torch.einsum('thd,nd->thn', q, kv[n0:n1])
        out[:, n0:n1] = (s.relu() * w.unsqueeze(-1)).sum(dim=1)
    return out


def quant_q_kv(q, kv, fmt):
    T, H, D = q.shape
    if fmt == 'fp8':
        q_hat = q.to(torch.float8_e4m3fn).float()
        kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
        kv_hat = (kv_fp8.float() * kv_sf.unsqueeze(1))
    elif fmt == 'mxfp8':
        qf, qs = per_token_cast_to_fp8(q.reshape(-1, D), use_ue8m0=True, gran_k=32)
        q_hat = cast_back_from_fp8(qf, qs, gran_k=32).view(T, H, D)
        kf, ks = per_token_cast_to_fp8(kv, use_ue8m0=True, gran_k=32)
        kv_hat = cast_back_from_fp8(kf, ks, gran_k=32)
    elif fmt == 'mxfp4':
        qf, qs = per_token_cast_to_fp4(q.reshape(-1, D), use_ue8m0=True, gran_k=32)
        q_hat = cast_back_from_fp4(qf, qs, gran_k=32).view(T, H, D)
        kf, ks = per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=32)
        kv_hat = cast_back_from_fp4(kf, ks, gran_k=32)
    else:
        raise ValueError(fmt)
    return q_hat, kv_hat


def topk_indices(logits, k):
    return logits.topk(k, dim=1).indices


def recall_jaccard(a, b):
    T, k = a.shape
    inter = torch.zeros(T, dtype=torch.float32, device=a.device)
    al = a.tolist()
    bl = b.tolist()
    for i in range(T):
        inter[i] = len(set(al[i]) & set(bl[i]))
    return (inter / k).mean().item(), inter


def block_scores(logits, block):
    T, C = logits.shape
    return logits.view(T, C // block, block).amax(dim=2)


def block_to_token_support(bidx, block, C):
    """[T,kb] 块号 → [T, kb*block] token 支持集（块内全部 token）。"""
    off = torch.arange(block, device=bidx.device)
    return (bidx.unsqueeze(-1) * block + off).reshape(bidx.shape[0], -1)


def attn_output(logits, support, V, tau=1.0, chunk=128):
    """把 softmax 限制在 support（[T,K] token 索引）上，对 V[支持] 加权。
    logits: [T,C] 注意力打分；返回 [T,dv]。"""
    T = logits.shape[0]
    dv = V.shape[1]
    out = torch.empty(T, dv, dtype=torch.float32, device=logits.device)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        idx = support[t0:t1]                       # [c,K]
        lg = torch.gather(logits[t0:t1], 1, idx)   # [c,K]
        wt = torch.softmax(lg / tau, dim=1)
        vs = V[idx].float()                        # [c,K,dv]
        out[t0:t1] = torch.einsum('ck,ckd->cd', wt, vs)
    return out


def dense_output(logits, V, tau=1.0, chunk=64):
    """全上下文 softmax 注意力输出。"""
    T = logits.shape[0]
    out = torch.empty(T, V.shape[1], dtype=torch.float32, device=logits.device)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        wt = torch.softmax(logits[t0:t1] / tau, dim=1)
        out[t0:t1] = wt @ V.float()
    return out


def rel_l2(x, y):
    return ((x - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-30)).mean().item()


def cos(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=1).mean().item()


def mask_from_support(idx, C):
    m = torch.zeros(idx.shape[0], C, dtype=torch.bool, device=idx.device)
    m.scatter_(1, idx, True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=1536)
    ap.add_argument('--kv', type=int, default=32768)
    ap.add_argument('--heads', type=int, default=32)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--dv', type=int, default=64)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--block', type=int, default=16)
    ap.add_argument('--tau', type=float, default=1.0)
    args = ap.parse_args()
    taus = [1.0, 4.0, 8.0, 16.0]

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, H, D, K, dv = args.seq, args.kv, args.heads, args.dim, args.k, args.dv
    print(f'# indexer selection → downstream sparse-attention output | '
          f'T={T} C={C} H={H} D={D} topk={K} block={args.block} tau={args.tau}')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(C, D, device=dev, dtype=torch.bfloat16)
    w = torch.randn(T, H, device=dev, dtype=torch.float32)
    V = torch.randn(C, dv, device=dev, dtype=torch.bfloat16)

    ks = torch.zeros(T, dtype=torch.int, device=dev)
    ke = torch.full((T,), C, dtype=torch.int, device=dev)

    # ---------------- Part A: 锚定 + 三档支持集 ----------------
    print('\n== Part A: kernel 锚定 + 三档 token 级支持集 ==')
    q_in = (q.to(torch.float8_e4m3fn), None)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
    logits_kernel = deep_gemm.fp8_fp4_mqa_logits(
        q=q_in, kv=(kv_fp8, kv_sf), weights=w,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=C)
    logits_teacher = mqa_logits(q.float(), kv.float(), w)
    q_hat8, kv_hat8 = quant_q_kv(q, kv, 'fp8')
    logits_ref8 = mqa_logits(q_hat8, kv_hat8, w)
    rel_anchor = (logits_kernel.float() - logits_ref8).abs().max().item() / logits_ref8.abs().max().item()
    print(f'  kernel vs torch-fp8 ref: max-abs-rel = {rel_anchor:.3e}')

    teacher_idx = topk_indices(logits_teacher, K)
    variants = {'kernel-fp8': topk_indices(logits_kernel.float(), K)}
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        qh, kvh = quant_q_kv(q, kv, fmt)
        lg = mqa_logits(qh, kvh, w)
        variants[fmt] = topk_indices(lg, K)
        rec, _ = recall_jaccard(teacher_idx, variants[fmt])
        print(f'  [{fmt:10s}] top-{K} recall = {rec:.4f} (flipped {K - rec * K:.0f})')

    m_teacher = mask_from_support(teacher_idx, C)

    # ---------------- Part B: 自洽档，扫注意力温度 tau ----------------
    # logits 尺度 ~18（32 头 relu 点积求和）会让 softmax 极尖锐、top-2048 几乎含全部质量；
    # 真实注意力的尖锐度取决于打分尺度，故用 tau 扫描「注意力分布有多平」这一下游变量。
    print('\n== Part B: 自洽档（注意力打分 = indexer 打分）——换支持集的下游输出误差 vs tau ==')
    rand_idx = torch.rand_like(logits_teacher).topk(K, dim=1).indices
    for tau in taus:
        p_full = torch.softmax(logits_teacher / tau, dim=1)
        teacher_mass = (p_full * m_teacher).sum(dim=1)
        O_dense = dense_output(logits_teacher, V, tau)
        O_teacher = attn_output(logits_teacher, teacher_idx, V, tau)
        O_rand = attn_output(logits_teacher, rand_idx, V, tau)
        base = rel_l2(O_teacher, O_dense)
        rand_e = rel_l2(O_rand, O_teacher)
        print(f'  tau={tau:5.1f}: teacher 质量={teacher_mass.mean():.6f} | 稀疏基线 O(t) vs dense relL2={base:.3e} '
              f'| 随机支持 relL2={rand_e:.3e}')
        for name in ('kernel-fp8', 'fp8', 'mxfp8', 'mxfp4'):
            idx = variants[name]
            m_q = mask_from_support(idx, C)
            dropped = (p_full * (m_teacher & ~m_q)).sum(dim=1)
            added = (p_full * (m_q & ~m_teacher)).sum(dim=1)
            O_q = attn_output(logits_teacher, idx, V, tau)
            print(f'      [{name:10s}] vs O(t) relL2={rel_l2(O_q, O_teacher):.3e} cos={cos(O_q, O_teacher):.6f}'
                  f' | vs dense={rel_l2(O_q, O_dense):.3e} | 丢={dropped.mean()/teacher_mass.mean():.3e}'
                  f' 补={added.mean()/teacher_mass.mean():.3e}')

    # ---------------- Part C: 失配档（indexer 与注意力相关性 ρ 旋钮，tau=8）----------------
    print('\n== Part C: 失配档（tau=8）——注意力打分 a = ρ·I + √(1-ρ²)·σ_row(I)·noise ==')
    tau_c = 8.0
    I = logits_teacher
    Irow_std = I.std(dim=1, keepdim=True)
    noise = torch.randn_like(I)
    for rho in (1.0, 0.7, 0.4, 0.0):
        a = rho * I + (1.0 - rho ** 2) ** 0.5 * Irow_std * noise
        O_t_a = attn_output(a, teacher_idx, V, tau_c)
        O_dense_a = dense_output(a, V, tau_c)
        line = [f'  rho={rho:.1f}: 稀疏基线(O(t) vs dense)={rel_l2(O_t_a, O_dense_a):.3e}']
        for fmt in ('fp8', 'mxfp8', 'mxfp4'):
            O_q_a = attn_output(a, variants[fmt], V, tau_c)
            line.append(f'{fmt}(vs O(t))={rel_l2(O_q_a, O_t_a):.3e}/cos={cos(O_q_a, O_t_a):.6f}')
        print('  '.join(line))
    print('  （ρ=1 即自洽档；ρ 越小 indexer 与注意力越脱节，同样翻转率的后果越大）')

    # ---------------- Part D: 块级（compressed indexer）下游，tau=8 ----------------
    print('\n== Part D: 块级（compressed indexer, max 聚合, tau=8）下游输出误差 ==')
    tau_d = 8.0
    B = args.block
    kb = K // B
    bs_teacher = block_scores(logits_teacher, B)
    bidx_teacher = topk_indices(bs_teacher, kb)
    sup_bt = block_to_token_support(bidx_teacher, B, C)
    O_bt = attn_output(logits_teacher, sup_bt, V, tau_d)
    O_teacher_d = attn_output(logits_teacher, teacher_idx, V, tau_d)
    print(f'  block={B}, top-{kb} blocks → {sup_bt.shape[1]} tokens; '
          f'块级 vs token 级 O(teacher): relL2={rel_l2(O_bt, O_teacher_d):.3e} cos={cos(O_bt, O_teacher_d):.6f}')
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        qh, kvh = quant_q_kv(q, kv, fmt)
        bs = block_scores(mqa_logits(qh, kvh, w), B)
        bidx = topk_indices(bs, kb)
        rec, _ = recall_jaccard(bidx_teacher, bidx)
        sup_b = block_to_token_support(bidx, B, C)
        O_b = attn_output(logits_teacher, sup_b, V, tau_d)
        print(f'  [{fmt:10s}] 块 recall={rec:.4f} | 块级下游 vs O(块级 teacher): '
              f'relL2={rel_l2(O_b, O_bt):.3e} cos={cos(O_b, O_bt):.7f} | vs O(token teacher)={rel_l2(O_b, O_teacher_d):.3e}')

    # ---------------- Part E: 上下文长度缩放（下游，tau=8）----------------
    print('\n== Part E: 下游输出误差 vs 上下文长度（top-k=2048 固定, tau=8）==')
    tau_e = 8.0
    for Cx in (8192, 32768, 131072):
        tcx = T if Cx <= 32768 else max(256, T // 6)
        qx = q[:tcx]
        if Cx <= C:
            kvx = kv[:Cx].contiguous()
        else:
            kvx = torch.randn(Cx, D, device=dev, dtype=torch.bfloat16)
        Vx = torch.randn(Cx, dv, device=dev, dtype=torch.bfloat16)
        ltx = mqa_logits(qx.float(), kvx.float(), w[:tcx])
        itx = topk_indices(ltx, K)
        otx = attn_output(ltx, itx, Vx, tau_e)
        odx = dense_output(ltx, Vx, tau_e)
        line = [f'  C={Cx:6d} (top-{K}/{Cx}={K/Cx:.3f}, 稀疏基线={rel_l2(otx, odx):.3e}):']
        for fmt in ('fp8', 'mxfp4'):
            qh, kvh = quant_q_kv(qx, kvx, fmt)
            lqx = mqa_logits(qh, kvh, w[:tcx])
            iqx = topk_indices(lqx, K)
            oqx = attn_output(ltx, iqx, Vx, tau_e)
            line.append(f'{fmt} relL2={rel_l2(oqx, otx):.3e}/cos={cos(oqx, otx):.6f}')
        print('  '.join(line))
    print('  （每档都用 teacher 打分做注意力权重、只换支持集，隔离「选择」这一环）')


if __name__ == '__main__':
    main()
