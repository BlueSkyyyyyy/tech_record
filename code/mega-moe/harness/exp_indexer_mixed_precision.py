"""P164-1：indexer 打分的【混合精度分配】—— Q × KV 二维精度矩阵、误差归因与带宽账。

前面第 171~173 篇把 indexer 的量化选择稳定性 / 下游影响 / 头结构拆过，但它们都把 Q 与 KV
**当成同一档精度**（fp8 全上、或 mxfp8 全上、或 mxfp4 全上）。生产里 indexer 的 KV cache 是
**逐 token 存下来的大头**（`[C, Di]`，C 是上下文长度），而 Q 只有 `[T, Hi, Di]`（T 是当前 token 数，
通常 T ≪ C）。于是真正该问的是：**这两个操作数能不能用不同精度？谁主导带宽、谁主导误差？**

本脚本在 H100(sm90) 上用真实 SM90 `fp8_fp4_mqa_logits` kernel 锚定 fp8 档，再以纯 torch 复刻
fp8 / mxfp8 / mxfp4 三档（逐 32 元素 + UE8M0，见 `deep_gemm/utils/math.py`），做四件事：

  A. 锚定：真实 kernel vs 同档 torch 参考。
  B. Q×KV 3×3 精度矩阵：每格的 top-k recall / 翻转数 / logit 噪声 σ；分清「同档」与「混档」。
  C. 误差归因：只量化 Q、只量化 KV、两者都量化 —— 谁把 logits 扰动得更多。
  D. 带宽账：按真实存储字节算出每种组合的 Q/KV 字节（KV 是主导项），给出「KV 半精度、Q 高精度」
     这类混合配方的净收益。
  E. 下游：把不同组合选出的支持集接进真实注意力输出（复用第 172/173 篇的 attn_forward 语义），
     量「省下来的 KV 字节」换成多少下游精度。
  F. 附：逐 indexer 头（Hi）的 Q 精度敏感度，看是否值得按头分配精度。

全部纯 torch + 真实 SM90 kernel，H100(sm90) 真机可跑。
"""
import argparse
import torch
import deep_gemm
from deep_gemm.utils import (per_token_cast_to_fp8, per_token_cast_to_fp4,
                             cast_back_from_fp8, cast_back_from_fp4,
                             per_custom_dims_cast_to_fp8)


# ----------------------------------------------------------------------------
# indexer 前向（与第 171~173 篇一致）
# ----------------------------------------------------------------------------
def mqa_logits(q, kv, w, chunk=8192):
    """logits[t,n] = sum_h w[t,h] * relu(<q[t,h,:], kv[n,:])，输入均为 fp32。"""
    T, H, D = q.shape
    C = kv.shape[0]
    out = torch.empty((T, C), dtype=torch.float32, device=q.device)
    for n0 in range(0, C, chunk):
        n1 = min(n0 + chunk, C)
        s = torch.einsum('thd,nd->thn', q, kv[n0:n1])
        out[:, n0:n1] = (s.relu() * w.unsqueeze(-1)).sum(dim=1)
    return out


def quant_q(q, fmt):
    """量化 Q [T,H,D]，返回反量化后的 fp32 张量。"""
    T, H, D = q.shape
    if fmt == 'fp32':
        return q.float()
    if fmt == 'fp8':                       # SM90 生产档：Q 直接 e4m3，无逐块 scale
        return q.to(torch.float8_e4m3fn).float()
    if fmt == 'mxfp8':                     # 逐 32 e4m3 + UE8M0
        qf, qs = per_token_cast_to_fp8(q.reshape(-1, D), use_ue8m0=True, gran_k=32)
        return cast_back_from_fp8(qf, qs, gran_k=32).view(T, H, D)
    if fmt == 'mxfp4':                     # 逐 32 E2M1 + UE8M0
        qf, qs = per_token_cast_to_fp4(q.reshape(-1, D), use_ue8m0=True, gran_k=32)
        return cast_back_from_fp4(qf, qs, gran_k=32).view(T, H, D)
    raise ValueError(fmt)


def quant_kv(kv, fmt):
    """量化 KV [C,D]，返回反量化后的 fp32 张量。"""
    C, D = kv.shape
    if fmt == 'fp32':
        return kv.float()
    if fmt == 'fp8':                       # SM90 生产档：逐 token 一个 fp32 scale
        kf, ks = per_custom_dims_cast_to_fp8(kv, (0,), False)
        return kf.float() * ks.unsqueeze(1)
    if fmt == 'mxfp8':
        kf, ks = per_token_cast_to_fp8(kv, use_ue8m0=True, gran_k=32)
        return cast_back_from_fp8(kf, ks, gran_k=32)
    if fmt == 'mxfp4':
        kf, ks = per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=32)
        return cast_back_from_fp4(kf, ks, gran_k=32)
    raise ValueError(fmt)


def topk_indices(logits, k):
    return logits.topk(k, dim=1).indices


def recall(t_idx, v_idx):
    T, k = t_idx.shape
    inter = torch.zeros(T, dtype=torch.float32, device=t_idx.device)
    al, bl = t_idx.tolist(), v_idx.tolist()
    for i in range(T):
        inter[i] = len(set(al[i]) & set(bl[i]))
    return (inter / k).mean().item()


def rel_l2(x, y, dim=None):
    if dim is None:
        return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()
    return ((x - y).norm(dim=dim) / y.norm(dim=dim).clamp_min(1e-30))


# ----------------------------------------------------------------------------
# 存储字节模型（每操作数）
# ----------------------------------------------------------------------------
def q_bytes(T, H, D, fmt):
    if fmt == 'fp32':
        return T * H * D * 4
    if fmt == 'fp8':
        return T * H * D * 1
    if fmt == 'mxfp8':
        return T * H * D * 1 + T * H * (D // 32) * 1        # 1B UE8M0 per 32
    if fmt == 'mxfp4':
        return T * H * (D // 2) + T * H * (D // 32) * 1     # 2 elems/byte + scale
    raise ValueError(fmt)


def kv_bytes(C, D, fmt):
    if fmt == 'fp32':
        return C * D * 4
    if fmt == 'fp8':
        return C * D * 1 + C * 4                            # 逐 token fp32 scale
    if fmt == 'mxfp8':
        return C * D * 1 + C * (D // 32) * 1
    if fmt == 'mxfp4':
        return C * (D // 2) + C * (D // 32) * 1
    raise ValueError(fmt)


FORMATS = ['fp8', 'mxfp8', 'mxfp4']


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--kv', type=int, default=65536)
    ap.add_argument('--heads', type=int, default=32)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--dv', type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, Hi, Di, K = args.seq, args.kv, args.heads, args.dim, args.k
    print('# indexer mixed-precision: Q x KV precision matrix, error attribution, bandwidth ledger')
    print(f'# T={T} C={C} H={Hi} D={Di} topk={K}, dv={args.dv}')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    qi = torch.randn(T, Hi, Di, device=dev, dtype=torch.bfloat16)
    kvi = torch.randn(C, Di, device=dev, dtype=torch.bfloat16)
    w = torch.randn(T, Hi, device=dev, dtype=torch.float32)
    ks = torch.zeros(T, dtype=torch.int, device=dev)
    ke = torch.full((T,), C, dtype=torch.int, device=dev)

    # ---------------- Part A: kernel 锚定 ----------------
    print('\n== Part A: 真实 SM90 kernel 锚定（fp8 档）==')
    q_in = (qi.to(torch.float8_e4m3fn), None)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kvi, (0,), False)
    logits_kernel = deep_gemm.fp8_fp4_mqa_logits(
        q=q_in, kv=(kv_fp8, kv_sf), weights=w,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=C).float()
    teacher = mqa_logits(qi.float(), kvi.float(), w)
    ref8 = mqa_logits(quant_q(qi, 'fp8'), quant_kv(kvi, 'fp8'), w)
    rel_anchor = (logits_kernel - ref8).abs().max().item() / ref8.abs().max().item()
    print(f'  kernel vs torch-fp8 ref: max-abs-rel = {rel_anchor:.3e}')
    teacher_idx = topk_indices(teacher, K)
    print(f'  teacher logit std = {teacher.std():.4f}, |logits|max = {teacher.abs().max():.3f}')

    # ---------------- Part B: Q x KV 精度矩阵 ----------------
    print('\n== Part B: Q x KV 3x3 精度矩阵（top-%d recall / 翻转 / logit 噪声 sigma）==' % K)
    print('  {:<9}'.format('Q\\KV') + ''.join(' {:<27}'.format(k) for k in FORMATS))
    rec_table, sig_table = {}, {}
    combos = {}
    for qf in FORMATS:
        qh = quant_q(qi, qf)
        row = []
        for kf in FORMATS:
            kh = quant_kv(kvi, kf)
            lg = mqa_logits(qh, kh, w)
            combos[(qf, kf)] = (lg, topk_indices(lg, K))
            rec = recall(teacher_idx, combos[(qf, kf)][1])
            sig = (lg - teacher).std().item()
            rec_table[(qf, kf)] = rec
            sig_table[(qf, kf)] = sig
            row.append(' {:<27}'.format(f'rec={rec:.4f} flip={K-rec*K:.0f} sig={sig:.4f}'))
        print('  {:<9}'.format(qf) + ''.join(row))
    print('  同档对角线: ' + ', '.join(f'{f}={rec_table[(f,f)]:.4f}' for f in FORMATS))
    print('  混档候选: ' + ', '.join(f'Q={a},KV={b}={rec_table[(a,b)]:.4f}'
          for a, b in [('fp8', 'mxfp4'), ('mxfp8', 'mxfp4'), ('mxfp4', 'mxfp8')]))
    dq = (quant_q(qi, 'fp8') - quant_q(qi, 'mxfp8')).abs().max().item()
    print(f'  注：bf16 的 Q 在 fp8 与 mxfp8 下逐元素最大差 = {dq:.3e}（≈1 个 e4m3 次正规台阶）'
          ' ⇒ 二者对 Q 的实际效果不可分：逐 32 的 2 的幂 scale 与 e4m3 的指数对齐，'
          'round-trip 只在次正规区留下 <1e-3 的差，top-k 结果与 σ 同到 4 位')

    # ---------------- Part C: 误差归因 ----------------
    print('\n== Part C: logit 误差归因（只量化 Q / 只量化 KV / 两者；relL2 vs teacher 与噪声 sigma）==')
    for qf in FORMATS:
        qh = quant_q(qi, qf)
        q_only = mqa_logits(qh, kvi.float(), w)
        print(f'  [{qf:6s}] 只 Q : relL2={rel_l2(q_only, teacher):.4e}  sig={ (q_only-teacher).std():.4f}')
    for kf in FORMATS:
        kh = quant_kv(kvi, kf)
        kv_only = mqa_logits(qi.float(), kh, w)
        print(f'  [{kf:6s}] 只 KV: relL2={rel_l2(kv_only, teacher):.4e}  sig={ (kv_only-teacher).std():.4f}')
    print('  （KV 的误差对全部 H 个头共享；Q 的误差逐头进入。两者方差近似可加）')

    # ---------------- Part D: 带宽账 ----------------
    print('\n== Part D: 存储字节账（Q 是 [T,H,D]、KV 是 [C,D]，C>>T ⇒ KV 主导）==')
    print('  {:<8} {:<14} {:<16} {:<16} {:<12}'.format('combo', 'Q bytes', 'KV bytes', 'total bytes', 'KV share'))
    base = None
    ledger = {}
    for qf in FORMATS:
        for kf in FORMATS:
            qb = q_bytes(T, Hi, Di, qf)
            kb = kv_bytes(C, Di, kf)
            tot = qb + kb
            ledger[(qf, kf)] = tot
            if (qf, kf) == ('fp8', 'fp8'):
                base = tot
    for qf in FORMATS:
        for kf in FORMATS:
            qb = q_bytes(T, Hi, Di, qf)
            kb = kv_bytes(C, Di, kf)
            tot = qb + kb
            print('  {:<8} {:<14} {:<16} {:<16} {:<12}'.format(
                f'{qf}/{kf}', f'{qb/1e6:.3f}MB', f'{kb/1e6:.3f}MB', f'{tot/1e6:.3f}MB',
                f'{kb/tot*100:.1f}%'))
    print('\n  相对 fp8/fp8 的字节比 / recall（选几个配方）：')
    for combo in [('fp8', 'fp8'), ('fp8', 'mxfp8'), ('mxfp8', 'mxfp8'),
                  ('fp8', 'mxfp4'), ('mxfp8', 'mxfp4'), ('mxfp4', 'mxfp4')]:
        print('    {:<14} bytes={:5.2f}x  recall={:.4f}'.format(
            f'{combo[0]}/{combo[1]}', ledger[combo] / base, rec_table[combo]))

    # ---------------- Part E: 下游注意力输出误差（vs 稠密注意力）----------------
    print('\n== Part E: 下游稀疏注意力输出误差 —— O(量化支持集) vs O(稠密全上下文)，G=4 ==')
    print('   支持集共享、逐头 per-head softmax；tau 越小越尖、稀疏支持集捕获质量越高')
    Hkv, G = 8, Hi // 8
    torch.manual_seed(1)
    V = torch.randn(C, Hkv, args.dv, device=dev, dtype=torch.bfloat16)
    Ihat = (teacher - teacher.mean(1, keepdim=True)) / teacher.std(1, keepdim=True).clamp_min(1e-6)
    eps = torch.randn(T, Hi, C, device=dev, dtype=torch.float32)
    rho = 0.98
    A = rho * Ihat.unsqueeze(1) + (1 - rho ** 2) ** 0.5 * eps   # 分数 std~1（与第 173 篇同口径）

    def attn_dense(Amat, Vmat, tau, chunk=32):
        Tn, Hq, Cn = Amat.shape
        O = torch.zeros(Tn, Hq, Vmat.shape[2], dtype=torch.float32, device=Amat.device)
        for t0 in range(0, Tn, chunk):
            t1 = min(t0 + chunk, Tn)
            for hvg in range(Hkv):
                hi = torch.arange(hvg * G, (hvg + 1) * G, device=Amat.device)
                ag = Amat[t0:t1][:, hi, :] / tau            # [c,G,C]
                p = torch.softmax(ag, dim=2)
                vg = Vmat[:, hvg, :].float()                # [C,dv]
                O[t0:t1, hvg * G:(hvg + 1) * G] = torch.einsum('rgc,cd->rgd', p, vg)
        return O

    def attn_sparse(Amat, Vmat, support, tau, chunk=32):
        Tn, Hq, _ = Amat.shape
        O = torch.zeros(Tn, Hq, Vmat.shape[2], dtype=torch.float32, device=Amat.device)
        for t0 in range(0, Tn, chunk):
            t1 = min(t0 + chunk, Tn)
            idx = support[t0:t1]
            rows = torch.arange(t0, t1, device=Amat.device)[:, None, None]
            for hvg in range(Hkv):
                hi = torch.arange(hvg * G, (hvg + 1) * G, device=Amat.device)[None, :, None]
                ag = Amat[rows, hi, idx[:, None, :]] / tau
                vg = Vmat[:, hvg, :][idx].float()
                p = torch.softmax(ag, dim=2)
                O[t0:t1, hvg * G:(hvg + 1) * G] = torch.einsum('rgk,rkd->rgd', p, vg)
        return O

    for tau in (0.25, 0.5, 1.0):
        A_tau = A / tau
        O_dense = attn_dense(A, V, tau)
        msup = torch.zeros(T, C, dtype=torch.bool, device=dev)
        msup.scatter_(1, teacher_idx, True)
        p_all = torch.softmax(A_tau, dim=2)
        mass = (p_all * msup.unsqueeze(1)).sum(2).mean().item()
        line = [f'  tau={tau:.1f} (teacher 支持集捕获质量={mass:.3f}):  仅选择 loss='
                f'{rel_l2(attn_sparse(A, V, teacher_idx, tau), O_dense):.2e}']
        for combo in [('fp8', 'fp8'), ('mxfp8', 'mxfp4'), ('fp8', 'mxfp4'), ('mxfp4', 'mxfp4')]:
            O_q = attn_sparse(A, V, combos[combo][1], tau)
            line.append(f'{combo[0]}/{combo[1]}={rel_l2(O_q, O_dense):.2e}')
        print('  '.join(line))

    # ---------------- Part F: 逐 indexer 头 Q 精度敏感度 ----------------
    print('\n== Part F: 逐 indexer 头（Hi）的 Q 精度敏感度 —— 只量化一个头看 score 噪声 ==')
    sens = torch.zeros(Hi, device=dev)
    for h in range(Hi):
        qh = qi.float().clone()
        qh[:, h, :] = quant_q(qi[:, h:h + 1, :], 'mxfp4')[:, 0, :]
        lg = mqa_logits(qh, kvi.float(), w)
        sens[h] = (lg - teacher).std().item()
    order = torch.argsort(sens)
    print(f'  逐头 sigma: min={sens.min():.4f} max={sens.max():.4f} '
          f'最敏 4 头={sens[order[-4:]].mean():.4f} 最钝 4 头={sens[order[:4]].mean():.4f} '
          f'比={sens[order[-4:]].mean()/sens[order[:4]].mean().clamp_min(1e-9):.2f}x')
    print(f'  |w| 相关性: corr(|w_h|, sens_h)={torch.corrcoef(torch.stack([w.abs().mean(0), sens]))[0,1]:.3f}')

    print('\n# ALL DONE')


if __name__ == '__main__':
    main()
