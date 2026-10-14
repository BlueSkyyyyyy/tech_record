"""P167-1：indexer 打分的【粗筛-精排】两级精度 —— 2x 带宽省下来的排序精度能不能补回来。

第 171 篇量到 indexer 的 top-k 选择在量化下会翻（FP8 recall 0.966 / MXFP4 0.851），
第 174 篇把 Q 与 KV 拆开，发现 KV 是带宽主导项（占字节 ~89%）、KV 换 MXFP4 让总字节降到 0.55x，
但 recall 掉到 0.85。这两篇都把打分当成**单级、均匀量化**。本篇问一个工程问题：

    生产 indexer 打分是「扫全上下文 C 个 KV、取 top-k」。与其把这一次扫描均匀压到 MXFP4
    省一半字节（但丢排序精度），能不能**拆成两级**——先用便宜的精度扫完 C 拿到短名单 M，
    再只用高精度重算这 M 个？（即「少读多精」而不是「全读低精」。）

本脚本在 H100(sm90) 上用真实 SM90 `fp8_fp4_mqa_logits` 锚定 fp8 档，再以纯 torch 复刻
fp8 / mxfp8 / mxfp4（逐 32/16 元素 + UE8M0，`deep_gemm/utils/math.py`）做：

  A. 锚定：真实 kernel vs 同档 torch 参考。
  B. 单级基线：均匀 KV 精度（fp8 / mxfp8 / mxfp4 / mxfp4_g16）的 recall@k 与字节。
  C. 两级级联：coarse（全 C，便宜格式）-> top-M 短名单 -> refine（只 M，fp8 或 fp32）。
     量最终 recall@k、coarse 的 recall@M（级联天花板）、以及总读取字节
       bytes = C * bpe(coarse) + M * bpe(refine)。
  D. Pareto：把各方案放在 (bytes/bytes_fp8, recall) 平面上，找膝点；等字节对照。
  E. 下游稀疏注意力输出误差：best 级联 vs 均匀 MXFP4。
  F. 逐 stage 字节/精度分解与「净收益」标量。

全部纯 torch + 真实 SM90 kernel，H100(sm90) 真机可跑。
"""
import argparse
import torch
import deep_gemm
from deep_gemm.utils import (per_token_cast_to_fp8, per_token_cast_to_fp4,
                             cast_back_from_fp8, cast_back_from_fp4,
                             per_custom_dims_cast_to_fp8)


# ----------------------------------------------------------------------------
# indexer 前向（与第 171~174 篇一致）
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
    if fmt == 'mxfp4_g16':                 # 更细的块（每 16 元素一个 UE8M0）
        kf, ks = per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=16)
        return cast_back_from_fp4(kf, ks, gran_k=16)
    if fmt == 'mxfp8_g16':
        kf, ks = per_token_cast_to_fp8(kv, use_ue8m0=True, gran_k=16)
        return cast_back_from_fp8(kf, ks, gran_k=16)
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
# 存储字节模型：每个 KV 元素读多少字节（含 scale 开销）
#   fp8 档：1B/元素 + 逐 token 4B fp32 scale（摊到 D 个元素）
#   mxfp8 档：1B/元素 + 每 32 元素 1B UE8M0
#   mxfp4 档：0.5B/元素 + 每 gran 元素 1B UE8M0
# ----------------------------------------------------------------------------
def kv_bpe(fmt, D):
    if fmt == 'fp8':
        return 1.0 + 4.0 / D
    if fmt == 'mxfp8':
        return 1.0 + 1.0 / 32
    if fmt == 'mxfp8_g16':
        return 1.0 + 1.0 / 16
    if fmt == 'mxfp4':
        return 0.5 + 1.0 / 32
    if fmt == 'mxfp4_g16':
        return 0.5 + 1.0 / 16
    if fmt == 'fp32':
        return 4.0
    raise ValueError(fmt)


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
    print('# indexer coarse-refine two-stage precision: sorting precision vs bandwidth net benefit')
    print(f'# T={T} C={C} H={Hi} D={Di} topk={K}, dv={args.dv}')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    qi = torch.randn(T, Hi, Di, device=dev, dtype=torch.bfloat16)
    kvi = torch.randn(C, Di, device=dev, dtype=torch.bfloat16)
    w = torch.randn(T, Hi, device=dev, dtype=torch.float32)
    ks = torch.zeros(T, dtype=torch.int, device=dev)
    ke = torch.full((T,), C, dtype=torch.int, device=dev)

    bpe_fp8 = kv_bpe('fp8', Di)
    total_fp8 = C * Di * bpe_fp8

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

    # ---------------- Part B: 单级均匀基线 ----------------
    print('\n== Part B: 单级均匀 KV 精度（recall@%d / 翻转 / 字节）==' % K)
    print('  {:<12} {:<10} {:<10} {:<12} {:<10}'.format('KV fmt', 'recall', 'flip', 'bytes/fp8', 'GB'))
    uniform = {}
    for kfmt in ['fp8', 'mxfp8', 'mxfp8_g16', 'mxfp4', 'mxfp4_g16']:
        kh = quant_kv(kvi, kfmt)
        lg = mqa_logits(quant_q(qi, 'fp8'), kh, w)
        idx = topk_indices(lg, K)
        rec = recall(teacher_idx, idx)
        bpe = kv_bpe(kfmt, Di)
        by = C * Di * bpe
        uniform[kfmt] = (rec, by / total_fp8, lg, idx)
        print('  {:<12} {:<10.4f} {:<10.0f} {:<12.4f} {:<10.3f}'.format(
            kfmt, rec, K - rec * K, by / total_fp8, by / 1e9))

    # ---------------- Part C: 两级级联 ----------------
    print('\n== Part C: 两级级联 —— coarse(全 C, 便宜) -> top-M 短名单 -> refine(仅 M, 高精) ==')
    print('   bytes = C*bpe(coarse) + M*bpe(refine)；天花=coarse recall@M')
    qh = quant_q(qi, 'fp8')
    # 预算好各精度下的全量 logits，refine 直接取短名单位置（模拟重算 M 个）
    kv_fine_cache = {}
    for fine in ['fp8', 'mxfp8', 'mxfp4']:
        kv_fine_cache[fine] = mqa_logits(qh, quant_kv(kvi, fine), w)
    # 天花板也可用 teacher 分数（理想 refine，忽略 M 内部量化）
    Ms = [K, 2 * K, 4 * K, 8 * K]
    print('  {:<16} {:<6} {:<9} {:<11} {:<10} {:<10}'.format(
        'coarse', 'M', 'ceil@M', 'recall@k', 'bytes/fp8', 'GB'))
    cascade = {}
    for coarse in ['mxfp4', 'mxfp4_g16', 'mxfp8']:
        coarse_lg = mqa_logits(qh, quant_kv(kvi, coarse), w)
        coarse_idx_M = coarse_lg.topk(max(Ms), dim=1).indices        # [T, maxM]
        # coarse 的 recall@M（级联天花板）按每个 M 前缀算
        for M in Ms:
            shortM = coarse_idx_M[:, :M]
            ceil_rec = recall(teacher_idx, shortM)
            for fine in ['fp8']:
                Lfine = kv_fine_cache[fine]                          # [T, C]
                row = torch.arange(T, device=dev)[:, None]
                ref_scores = Lfine[row, shortM]                      # 重算短名单
                # 只保留短名单，其余 -inf
                casc = torch.full((T, C), float('-inf'), device=dev)
                casc[row, shortM] = ref_scores
                idx = topk_indices(casc, K)
                rec = recall(teacher_idx, idx)
                by = C * Di * kv_bpe(coarse, Di) + M * Di * kv_bpe(fine, Di)
                cascade[(coarse, M, fine)] = (rec, by / total_fp8, idx)
                print('  {:<16} {:<6} {:<9.4f} {:<11.4f} {:<10.4f} {:<10.3f}'.format(
                    coarse, M, ceil_rec, rec, by / total_fp8, by / 1e9))

    # 理想 refine（用 teacher 分数重排短名单）看「coarse 短名单」本身的上界
    print('\n  -- 理想 refine（短名单内用 fp32 teacher 分数重排）--')
    for coarse in ['mxfp4', 'mxfp8']:
        coarse_lg = mqa_logits(qh, quant_kv(kvi, coarse), w)
        coarse_idx_M = coarse_lg.topk(max(Ms), dim=1).indices
        for M in [2 * K, 4 * K]:
            shortM = coarse_idx_M[:, :M]
            row = torch.arange(T, device=dev)[:, None]
            casc = torch.full((T, C), float('-inf'), device=dev)
            casc[row, shortM] = teacher[row, shortM]
            idx = topk_indices(casc, K)
            rec = recall(teacher_idx, idx)
            print('  coarse={:<8} M={:<6} ideal-refine recall@k = {:.4f}'.format(coarse, M, rec))

    # ---------------- Part D: Pareto / 等字节对照 ----------------
    print('\n== Part D: Pareto (bytes/fp8, recall) 与等字节对照 ==')
    pts = []
    for kf, (rec, br, _, _) in uniform.items():
        pts.append((f'uniform/{kf}', br, rec))
    for key, (rec, br, _) in cascade.items():
        pts.append((f'cascade/{key[0]}->{key[2]} M={key[1]}', br, rec))
    pts.sort(key=lambda x: x[1])
    print('  {:<34} {:<12} {:<10}'.format('scheme', 'bytes/fp8', 'recall'))
    for name, br, rec in pts:
        print('  {:<34} {:<12.4f} {:<10.4f}'.format(name, br, rec))
    # 等字节：找均匀 MXFP4（0.53x）之上、recall 更高的级联
    mx_rec, mx_br = uniform['mxfp4'][0], uniform['mxfp4'][1]
    print(f'\n  均匀 MXFP4 锚: bytes={mx_br:.4f}x recall={mx_rec:.4f}')
    best = None
    for key, (rec, br, _) in cascade.items():
        if br <= mx_br + 0.10 and rec > mx_rec:
            gain = (rec - mx_rec) / (br - mx_br) if br > mx_br else float('inf')
            if best is None or rec > best[0]:
                best = (rec, br, key, gain)
    if best:
        rec, br, key, gain = best
        print(f'  最佳近字节级联: coarse={key[0]} M={key[1]} fine={key[2]} '
              f'bytes={br:.4f}x recall={rec:.4f} (+{rec-mx_rec:.4f} recall, +{br-mx_br:.4f}x bytes)')
    # 与均匀 fp8 比：级联能否在更少字节达到 fp8 级 recall
    u8_rec, u8_br = uniform['fp8'][0], uniform['fp8'][1]
    print(f'  均匀 FP8 锚: bytes={u8_br:.4f}x recall={u8_rec:.4f}')
    for key, (rec, br, _) in cascade.items():
        if rec >= u8_rec - 0.01:
            print(f'  >> 级联 {key[0]} M={key[1]} 达 FP8 级 recall: bytes={br:.4f}x '
                  f'(= {br/u8_br:.3f}x of FP8), recall={rec:.4f}')

    # ---------------- Part E: 下游稀疏注意力输出误差 ----------------
    print('\n== Part E: 下游稀疏注意力输出误差 —— best 级联 vs 均匀 MXFP4，G=4 ==')
    Hkv, G = 8, Hi // 8
    torch.manual_seed(1)
    V = torch.randn(C, Hkv, args.dv, device=dev, dtype=torch.bfloat16)
    Ihat = (teacher - teacher.mean(1, keepdim=True)) / teacher.std(1, keepdim=True).clamp_min(1e-6)
    eps = torch.randn(T, Hi, C, device=dev, dtype=torch.float32)
    rho = 0.98
    A = rho * Ihat.unsqueeze(1) + (1 - rho ** 2) ** 0.5 * eps

    def attn_dense(Amat, Vmat, tau, chunk=32):
        Tn, Hq, Cn = Amat.shape
        O = torch.zeros(Tn, Hq, Vmat.shape[2], dtype=torch.float32, device=Amat.device)
        for t0 in range(0, Tn, chunk):
            t1 = min(t0 + chunk, Tn)
            for hvg in range(Hkv):
                hi = torch.arange(hvg * G, (hvg + 1) * G, device=Amat.device)
                ag = Amat[t0:t1][:, hi, :] / tau
                p = torch.softmax(ag, dim=2)
                vg = Vmat[:, hvg, :].float()
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
        O_dense = attn_dense(A, V, tau)
        msup = torch.zeros(T, C, dtype=torch.bool, device=dev)
        msup.scatter_(1, teacher_idx, True)
        p_all = torch.softmax(A / tau, dim=2)
        mass = (p_all * msup.unsqueeze(1)).sum(2).mean().item()
        cand = {
            'uniform/fp8': uniform['fp8'][3],
            'uniform/mxfp4': uniform['mxfp4'][3],
        }
        for key in [('mxfp4', 2 * K, 'fp8'), ('mxfp4', 4 * K, 'fp8'), ('mxfp8', 2 * K, 'fp8')]:
            if key in cascade:
                cand[f'casc/{key[0]} M={key[1]}'] = cascade[key][2]
        line = [f'  tau={tau:.2f} (teacher 捕获质量={mass:.3f}):']
        for nm, sup in cand.items():
            line.append(f'{nm}={rel_l2(attn_sparse(A, V, sup, tau), O_dense):.2e}')
        print('  '.join(line))

    # ---------------- Part F: 净收益标量 ----------------
    print('\n== Part F: 净收益（recall 增益 / 字节代价），以均匀 MXFP4 为参考 ==')
    print('  {:<30} {:<10} {:<10} {:<18}'.format('scheme', 'bytes/fp8', 'recall', 'dRecall/dBytes'))
    for key in [('mxfp4', 2 * K, 'fp8'), ('mxfp4', 4 * K, 'fp8'), ('mxfp8', 2 * K, 'fp8')]:
        if key in cascade:
            rec, br, _ = cascade[key]
            db = br - mx_br
            dr = rec - mx_rec
            slope = (dr / db) if db > 1e-9 else float('inf')
            print('  {:<30} {:<10.4f} {:<10.4f} {:<18.3f}'.format(
                f'{key[0]} M={key[1]}', br, rec, slope))
    print('  （dRecall/dBytes：相对均匀 MXFP4，每多 1.0x 字节换回多少 recall 绝对量）')

    print('\n# ALL DONE')


if __name__ == '__main__':
    main()
