"""P161-1：lightning indexer 的 top-k KV（块）选择在量化下的稳定性。

前向打分算子 `fp8_fp4_mqa_logits` 的 Q/KV 在生产里是 FP4(MXFP4)/FP8(MXFP8)（SM100），
本机 H100(sm90) 只能跑它最弱的一档 —— 直接 e4m3 的 Q + 逐 token fp32 scale 的 KV。
但「量化把 logits 擾動了多少、进而改变 top-k 选择多少」这件事与 kernel 架构无关，
可以在 H100 上用真实 SM90 kernel 锚定 + 纯 torch 复刻 FP8/MXFP8/MXFP4 三种量化档来做。

本脚本做四件事：
  A. 锚定：真实 `deep_gemm.fp8_fp4_mqa_logits` vs 同量化档的 torch 参考（应逐位/极近）。
  B. token 级 top-k 选择稳定性：FP8 / MXFP8 / MXFP4 vs FP32 teacher 的 recall / Jaccard
     / 边界 margin，并量各档的 logit 噪声 σ。
  C. 块级（compressed indexer）top-k 选择稳定性：把 token 分数压成块分数（max）再 top-k。
  D. 选择翻转的“梯度/概率质量”代价：teacher 选中集里被丢掉的概率质量、错补进来的质量。

全部纯 torch + 真实 SM90 kernel，H100(sm90) 真机可跑。
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
        s = torch.einsum('thd,nd->thn', q, kv[n0:n1])          # [T,H,c]
        out[:, n0:n1] = (s.relu() * w.unsqueeze(-1)).sum(dim=1)
    return out


def quant_q_kv(q, kv, fmt):
    """返回 (q_hat, kv_hat) —— 把 q/kv 量化再反量化到 fp32，用于 torch 复刻。"""
    T, H, D = q.shape
    if fmt == 'fp8':                 # SM90 生产档：Q 直接 e4m3（无 scale）、KV 逐 token fp32 scale
        q_hat = q.to(torch.float8_e4m3fn).float()
        kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)   # [C,D],[C]
        kv_hat = (kv_fp8.float() * kv_sf.unsqueeze(1))                # 逐 token 广播
    elif fmt == 'mxfp8':             # SM100 MXFP8：逐 32 元素 e4m3 + UE8M0
        qf, qs = per_token_cast_to_fp8(q.reshape(-1, D), use_ue8m0=True, gran_k=32)
        q_hat = cast_back_from_fp8(qf, qs, gran_k=32).view(T, H, D)
        kf, ks = per_token_cast_to_fp8(kv, use_ue8m0=True, gran_k=32)
        kv_hat = cast_back_from_fp8(kf, ks, gran_k=32)
    elif fmt == 'mxfp4':             # SM100 MXFP4：逐 32 元素 E2M1 + UE8M0
        qf, qs = per_token_cast_to_fp4(q.reshape(-1, D), use_ue8m0=True, gran_k=32)
        q_hat = cast_back_from_fp4(qf, qs, gran_k=32).view(T, H, D)
        kf, ks = per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=32)
        kv_hat = cast_back_from_fp4(kf, ks, gran_k=32)
    else:
        raise ValueError(fmt)
    return q_hat, kv_hat


def topk_indices(logits, k):
    """每行取 top-k 的列索引（unsorted）。返回 [T,k] int64。"""
    return logits.topk(k, dim=1).indices


def recall_jaccard(a, b):
    """a,b: [T,k] 索引集。返回集合平均 recall（|A∩B|/k）与 Jaccard，以及每行的交集计数 [T]。"""
    T, k = a.shape
    inter = torch.zeros(T, dtype=torch.float32, device=a.device)
    al = a.tolist()
    bl = b.tolist()
    for i in range(T):
        inter[i] = len(set(al[i]) & set(bl[i]))
    recall = inter / k
    jac = inter / (2 * k - inter)
    return recall.mean().item(), jac.mean().item(), inter


def recall_of(inter, k, mask):
    return (inter[mask] / k).mean().item()


def block_scores(logits, block):
    """把 token 分数压成块分数：块内 max。返回 [T, C/block]。"""
    T, C = logits.shape
    return logits.view(T, C // block, block).amax(dim=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=1536)
    ap.add_argument('--kv', type=int, default=32768)
    ap.add_argument('--heads', type=int, default=32)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--block', type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, H, D, K = args.seq, args.kv, args.heads, args.dim, args.k
    print(f'# indexer selection stability | T={T} C={C} H={H} D={D} topk={K} block={args.block}')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    # 输入：q/kv ~ N(0,1)（bf16 存；teacher 用其 fp32）。weights 用 bf16（SM100）——这里统一 fp32 参考。
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(C, D, device=dev, dtype=torch.bfloat16)
    w = torch.randn(T, H, device=dev, dtype=torch.float32)

    # 单序列、全范围：ks=0, ke=C
    ks = torch.zeros(T, dtype=torch.int, device=dev)
    ke = torch.full((T,), C, dtype=torch.int, device=dev)

    # ---------------- Part A: kernel 锚定 ----------------
    print('\n== Part A: 真实 SM90 kernel vs torch 参考（fp8 档）==')
    q_in = (q.to(torch.float8_e4m3fn), None)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
    logits_kernel = deep_gemm.fp8_fp4_mqa_logits(
        q=q_in, kv=(kv_fp8, kv_sf), weights=w,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=C)
    print(f'  kernel logits: shape={tuple(logits_kernel.shape)} dtype={logits_kernel.dtype}')
    q_hat8, kv_hat8 = quant_q_kv(q, kv, 'fp8')
    logits_ref8 = mqa_logits(q_hat8, kv_hat8, w)
    rel = (logits_kernel.float() - logits_ref8).abs().max().item() / logits_ref8.abs().max().item()
    print(f'  kernel vs torch-fp8 ref: max-abs-rel = {rel:.3e}')

    # ---------------- teacher + 各量化档 logits ----------------
    print('\n== Part B: token 级 top-k 选择稳定性 ==')
    logits_teacher = mqa_logits(q.float(), kv.float(), w)
    teacher_idx = topk_indices(logits_teacher, K)

    variants = {}
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        qh, kvh = quant_q_kv(q, kv, fmt)
        lg = mqa_logits(qh, kvh, w)
        noise = (lg - logits_teacher)
        sigma = noise.std().item()                                  # 全局噪声
        centered = noise - noise.mean(dim=1, keepdim=True)          # 逐行去均值 → 只影响排序的分量
        sigma_r = centered.std().item()
        max_r = centered.abs().max().item()
        idx = topk_indices(lg, K)
        rec, jac, inter = recall_jaccard(teacher_idx, idx)
        print(f'  [{fmt:6s}] noise(sigma={sigma:.4f}, 排序相关 sigma_r={sigma_r:.4f} max={max_r:.3f}) | '
              f'top-{K} recall={rec:.4f} jaccard={jac:.4f} flipped={K - int(inter.mean().item())}/{K}')
        variants[fmt] = (idx, sigma_r, rec, inter)

    # 对照：给 teacher 加同等 σ_r 的 iid 噪声（下界），看 recall 理论掉到多少
    noise_sigma = variants['fp8'][1]
    lg_iid = logits_teacher + torch.randn_like(logits_teacher) * noise_sigma
    idx_iid = topk_indices(lg_iid, K)
    rec_iid, jac_iid, _ = recall_jaccard(teacher_idx, idx_iid)
    # 随机对照（上界不相关）
    idx_rand = torch.rand_like(logits_teacher).topk(K, dim=1).indices
    rec_rand, _, _ = recall_jaccard(teacher_idx, idx_rand)
    print(f'  对照: iid 噪声(σ={noise_sigma:.3f}) recall={rec_iid:.4f} | 随机 recall={rec_rand:.4f}')
    # 逐行噪声与 logit 的相关（判断是否乘性/相关噪声）
    noise_fp8 = mqa_logits(*quant_q_kv(q, kv, 'fp8'), w) - logits_teacher
    corr = torch.corrcoef(torch.stack([noise_fp8[0] / 1, logits_teacher[0]]))[0, 1].item()
    print(f'  fp8 噪声 vs logit 逐行相关（row0）= {corr:.3f}')

    # 边界 margin 分析：teacher 第 K 名与第 K+1 名之差
    sorted_t, _ = torch.sort(logits_teacher, dim=1, descending=True)
    margin = (sorted_t[:, K - 1] - sorted_t[:, K])        # [T]
    print(f'  teacher boundary margin: mean={margin.mean():.4f} median={margin.median():.4f} '
          f'p10={margin.quantile(0.1):.4f} p90={margin.quantile(0.9):.4f}')
    # 正确的诊断量不是 margin，而是「阈值 ±σ 内的分数个数」（翻转候选池）
    thr = sorted_t[:, K - 1]
    for fmt in ('fp8', 'mxfp4'):
        s = variants[fmt][1]
        pool = ((logits_teacher >= (thr - s)[:, None]) & (logits_teacher <= (thr + s)[:, None])).sum(1).float()
        print(f'  [{fmt:6s}] 阈值±σ 内候选池={pool.mean():.1f}/行, 实际翻转={K - int(variants[fmt][3].mean()):.1f}/行')
    # 用某档的 σ_r 作尺度：margin 与 recall 的逐行相关（理论：margin 不是正确的诊断量）
    ref_sigma = variants['fp8'][1]
    ratio = margin / ref_sigma
    print(f'  teacher margin / fp8 σ_r: median={ratio.median():.3f} '
          f'p10={ratio.quantile(0.1):.3f} p90={ratio.quantile(0.9):.3f}（远小于 1，却高 recall）')
    # 逐行 recall 与 margin / 候选池 的相关系数
    for fmt in ('fp8', 'mxfp4'):
        s = variants[fmt][1]
        row_rec = variants[fmt][3] / K
        pool = ((logits_teacher >= (thr - s)[:, None]) & (logits_teacher <= (thr + s)[:, None])).sum(1).float()
        c_margin = torch.corrcoef(torch.stack([row_rec, margin]))[0, 1].item()
        c_pool = torch.corrcoef(torch.stack([row_rec, pool]))[0, 1].item()
        print(f'  [{fmt:6s}] corr(row-recall, margin)={c_margin:+.3f}  corr(row-recall, 候选池)={c_pool:+.3f}')

    # ---------------- Part C: 块级选择 ----------------
    print('\n== Part C: 块级（compressed indexer）选择稳定性 ==')
    B = args.block
    nb = C // B
    kb = max(1, K // B)
    bs_teacher = block_scores(logits_teacher, B)
    bidx_teacher = topk_indices(bs_teacher, kb)
    print(f'  block={B}, num_blocks={nb}, top-{kb} blocks')
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        qh, kvh = quant_q_kv(q, kv, fmt)
        lg = mqa_logits(qh, kvh, w)
        bs = block_scores(lg, B)
        bidx = topk_indices(bs, kb)
        rec, jac, inter = recall_jaccard(bidx_teacher, bidx)
        print(f'  [{fmt:6s}] block top-{kb} recall={rec:.4f} jaccard={jac:.4f} '
              f'flipped={kb - int(inter.mean().item())}/{kb}')

    # ---------------- Part E: 上下文长度缩放 ----------------
    print('\n== Part E: 选择稳定性 vs 上下文长度（k=2048 固定）==')
    for Cx in (8192, 32768, 131072):
        if Cx > C:
            kvc = torch.randn(Cx, D, device=dev, dtype=torch.bfloat16)
        else:
            kvc = kv[:Cx].contiguous()
        lt = mqa_logits(q.float(), kvc.float(), w)
        it = topk_indices(lt, K)
        line = [f'  C={Cx:6d} (top-{K}/{Cx}={K/Cx:.3f}):']
        for fmt in ('fp8', 'mxfp8', 'mxfp4'):
            qh, kvh = quant_q_kv(q, kvc, fmt)
            lq = mqa_logits(qh, kvh, w)
            rec, jac, _ = recall_jaccard(it, topk_indices(lq, K))
            line.append(f'{fmt}={rec:.3f}')
        print('  '.join(line))

    # ---------------- Part D: 翻转的概率质量代价 ----------------
    print('\n== Part D: 选择翻转的概率质量代价 ==')
    # teacher 分布的“真”目标：用一份与 indexer 独立的注意力分数当 teacher 分布 p
    attn = torch.randn(T, C, device=dev) + 2.0 * (torch.arange(C, device=dev)[None, :] < C // 2)
    p = torch.softmax(attn, dim=1)
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        idx = variants[fmt][0]
        mask_t = torch.zeros(T, C, dtype=torch.bool, device=dev)
        mask_t.scatter_(1, teacher_idx, True)
        mask_q = torch.zeros(T, C, dtype=torch.bool, device=dev)
        mask_q.scatter_(1, idx, True)
        dropped = (p * (mask_t & ~mask_q)).sum(dim=1)
        added = (p * (mask_q & ~mask_t)).sum(dim=1)
        teacher_mass = (p * mask_t).sum(dim=1)
        print(f'  [{fmt:6s}] teacher top-k 质量={teacher_mass.mean():.4f} '
              f'丢掉={dropped.mean():.2e} 错补={added.mean():.2e} '
              f'相对丢失={(dropped / teacher_mass).mean():.3e}')


if __name__ == '__main__':
    main()
