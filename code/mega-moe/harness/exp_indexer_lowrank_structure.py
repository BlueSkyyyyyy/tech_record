"""P166-1：把 indexer↔主注意力接进【真实、低秩】的 q/k 头结构——逐头 ρ_h 不均下，
第 175 篇（P165-1）「按值范数 ν 而不是熵加权」的结论还成不成立？头共享的税有多大？

第 173 篇（P163-1）用「ρ 旋钮 + 逐头 iid 噪声」造主注意力头结构，量到敏感度在逐头尖锐度里；
第 175 篇（P165-1）沿用同一套参数化，证明蒸馏目标该按下游重要度 `ν_h`（≈值范数²）加权、
而不是按熵。两篇都把**头间当作 iid**（`docs/175` §边界 3 自认：「头间可能有结构性相关
（低秩共享 q/k）」未建模）。真实 DSA 的头恰好**不独立**：GQA 让同组的 query 头共享同一份
key/value，预训练也把各头推向前几个主方向——**头间是有低秩共同成分的**。

本篇把「参数化建模」换成**真实 q/k 的低秩结构**，回答两件事：
  ① **逐头对齐 `ρ_h` 本来就不均**（indexer 是另训小模型，对某些头贴得紧、对另一些松）——
     不是第 173 篇那个「所有头同一个 ρ」。用真实 q/k 算出逐头 `ρ_h` 谱。
  ② **头间相关（低秩共享）会改变逐头权重的价值**——头越像、支持集对谁服务越无所谓，
     第 175 篇「按 ν 加权」的收益应随头间相关升高而**收缩**，而「按熵加权是错的」这条应保持。

真实 SM90 `fp8_fp4_mqa_logits` 锚定 fp8 档；MXFP8/MXFP4 用官方 torch 复刻（SM100 专属，仅分析）。
全部纯 torch + 真实 SM90 kernel，H100(sm90) 真机可跑。
"""
import argparse
import torch
import deep_gemm
from deep_gemm.utils import (per_custom_dims_cast_to_fp8, per_token_cast_to_fp8,
                             per_token_cast_to_fp4, cast_back_from_fp8, cast_back_from_fp4)


# ---------------------------------------------------------------------------
# indexer 打分（与第 171~175 篇一致，真实 SM90 kernel 锚定 fp8 档）
# ---------------------------------------------------------------------------
def mqa_logits(q, kv, w, chunk=8192):
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
    al, bl = a.tolist(), b.tolist()
    inter = sum(len(set(al[i]) & set(bl[i])) for i in range(T))
    return inter / (T * k)


def standardize(x, dim=-1):
    mu = x.mean(dim, keepdim=True)
    sd = x.std(dim, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# 真实主注意力头结构：低秩共享 q/k + 逐头偏移
# ---------------------------------------------------------------------------
def build_real_qk(T, C, Hq, Hkv, D, rank, dev, seed=2024):
    """各头 key/value 共享一个 rank 维子空间（预训练主方向），再叠逐头偏移。

    Kheads[g] = Kbase + alpha_g * offset；Kbase = B @ U（rank 维），B∈R^{C×rank}、U 正交。
    Qheads[h] 同理。alpha_h 逐头不同 ⇒ 有的头几乎落在共享子空间（与 indexer 贴得紧）、有的偏离。
    """
    torch.manual_seed(seed)
    U = torch.linalg.qr(torch.randn(D, rank, device=dev)).Q.t()          # [rank, D] 正交
    Kbase = (torch.randn(C, rank, device=dev) @ U)                       # [C, D]
    Qbase = (torch.randn(T, rank, device=dev) @ U)                       # [T, D]
    # 逐头偏移尺度：logspace 谱，让「贴得紧 / 偏得远」的头都有
    alpha = torch.logspace(-1.5, 0.5, Hq, device=dev)
    # 逐 KV 组的共享程度（GQA：同组头共享 key，故组内更相关）——
    # 每个 query 头用自己的偏移，key 按组偏移（GQA 语义）
    Kheads = Kbase.unsqueeze(0) + alpha[:Hkv].view(Hkv, 1, 1) * torch.randn(Hkv, C, D, device=dev)
    Qheads = Qbase.unsqueeze(0) + alpha.view(Hq, 1, 1) * torch.randn(Hq, T, D, device=dev)
    return Qheads, Kheads, alpha


def head_scores(Qheads, Kheads, Hkv, D):
    """a_h[t,j] = <Qheads[h,t], Kheads[h//G,j]> / sqrt(D)，返回 [Hq,T,C]（标准量）。"""
    Hq = Qheads.shape[0]
    G = Hq // Hkv
    out = torch.empty(Hq, Qheads.shape[1], Kheads.shape[1], device=Qheads.device)
    for h in range(Hq):
        a = Qheads[h] @ Kheads[h // G].t() / (D ** 0.5)
        out[h] = standardize(a, dim=1)
    return out


def head_corr_stats(A):
    """A=[Hq,T,C] 标准化头打分 -> 头间相关系数矩阵与其「有效头数」H_eff（参与率）。"""
    Hq = A.shape[0]
    flat = A.reshape(Hq, -1)
    flat = flat - flat.mean(1, keepdim=True)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-30)
    Cm = flat @ flat.t()                                                 # [Hq,Hq]
    off = Cm - torch.eye(Hq, device=A.device) * Cm.diag()
    mean_off = off.sum() / (Hq * (Hq - 1))
    ev = torch.linalg.eigvalsh(Cm).clamp_min(0)
    H_eff = (ev.sum() ** 2) / (ev ** 2).sum()
    return mean_off.item(), H_eff.item(), Cm


# ---------------------------------------------------------------------------
# 下游注意力（真实头结构）：per-head softmax + GQA 共享支持集
# ---------------------------------------------------------------------------
def full_attention(P, V, Hkv, chunk=2048):
    T, Hq, C = P.shape
    G = Hq // Hkv
    O = torch.zeros(T, Hq, V.shape[2], dtype=torch.float32, device=P.device)
    for c0 in range(0, C, chunk):
        c1 = min(c0 + chunk, C)
        for g in range(Hkv):
            Pg = P[:, g * G:(g + 1) * G, c0:c1]
            O[:, g * G:(g + 1) * G] += torch.einsum('tgc,cd->tgd', Pg, V[c0:c1, g, :].float())
    return O


def sparse_attention(A, V, support, Hkv, chunk=64):
    T, Hq, _ = A.shape
    G = Hq // Hkv
    O = torch.zeros(T, Hq, V.shape[2], dtype=torch.float32, device=A.device)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        idx = support[t0:t1]
        rows = torch.arange(t0, t1, device=A.device)[:, None, None]
        for g in range(Hkv):
            hi = torch.arange(g * G, (g + 1) * G, device=A.device)[None, :, None]
            ag = A[rows, hi, idx[:, None, :]]
            p = torch.softmax(ag, dim=2)
            O[t0:t1, g * G:(g + 1) * G] = torch.einsum('cgk,ckd->cgd', p, V[idx, g, :].float())
    return O


def per_head_err(O_s, O_f, dim=2):
    return ((O_s - O_f).norm(dim=dim) / O_f.norm(dim=dim).clamp_min(1e-30)).mean(0)


def weighted_output_err(O_s, O_f, nu):
    num = ((O_s - O_f) ** 2).sum(dim=2).mean(0)
    den = (O_f ** 2).sum(dim=2).mean(0).clamp_min(1e-30)
    return ((nu * num).sum() / (nu * den).sum()).sqrt().item()


def make_target(P, lam):
    return torch.einsum('thc,h->tc', P, lam) / lam.sum()


def oracle_attention(A, V, K, Hkv):
    T, Hq, _ = A.shape
    G = Hq // Hkv
    O = torch.zeros(T, Hq, V.shape[2], dtype=torch.float32, device=A.device)
    for h in range(Hq):
        S_h = topk_indices(A[:, h, :], K)
        a_h = torch.gather(A[:, h, :], 1, S_h)
        p_h = torch.softmax(a_h, dim=1)
        vg = V[S_h, h // G, :].float()
        O[:, h, :] = torch.einsum('tk,tkd->td', p_h, vg)
    return O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--kv', type=int, default=16384)
    ap.add_argument('--heads', type=int, default=32)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--dv', type=int, default=64)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--idim', type=int, default=64)
    ap.add_argument('--iheads', type=int, default=32)
    ap.add_argument('--gqa', type=int, default=4)
    ap.add_argument('--rank', type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, Hq, D, dv, K = args.seq, args.kv, args.heads, args.dim, args.dv, args.k
    Hi, Di = args.iheads, args.idim
    Hkv = Hq // args.gqa
    print('# indexer / main-attention head structure under REAL low-rank shared q/k')
    print(f'# T={T} C={C} H_q={Hq} H_kv={Hkv} G={args.gqa} D={D} dv={dv} topk={K} rank={args.rank} / indexer(H={Hi},d={Di})')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    # ---------------- Part A: 真实 SM90 fp8 kernel 锚定 ----------------
    print('\n== Part A: 真实 SM90 fp8_fp4_mqa_logits 锚定（indexer 打分算子）==')
    qi = torch.randn(T, Hi, Di, device=dev, dtype=torch.bfloat16)
    kvi = torch.randn(C, Di, device=dev, dtype=torch.bfloat16)
    w = torch.randn(T, Hi, device=dev, dtype=torch.float32)
    ks = torch.zeros(T, dtype=torch.int, device=dev)
    ke = torch.full((T,), C, dtype=torch.int, device=dev)
    q_in = (qi.to(torch.float8_e4m3fn), None)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kvi, (0,), False)
    logits_kernel = deep_gemm.fp8_fp4_mqa_logits(
        q=q_in, kv=(kv_fp8, kv_sf), weights=w,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=C)
    q_hat8, kv_hat8 = quant_q_kv(qi, kvi, 'fp8')
    logits_ref8 = mqa_logits(q_hat8, kv_hat8, w)
    print(f'  kernel vs torch-fp8 ref: max-abs-rel = '
          f'{(logits_kernel.float() - logits_ref8).abs().max().item() / logits_ref8.abs().max().item():.3e}')

    # ---------------- Part B: 真实低秩共享 q/k —— 头间相关是可控的 ----------------
    print('\n== Part B: 真实低秩共享 q/k —— 逐头相关矩阵与有效头数 H_eff ==')
    Qheads, Kheads, alpha = build_real_qk(T, C, Hq, Hkv, D, args.rank, dev)
    A_real = head_scores(Qheads, Kheads, Hkv, D)          # [Hq,T,C]
    mean_off, H_eff, Cm = head_corr_stats(A_real)
    print(f'  rank={args.rank}: 头间平均相关 = {mean_off:+.3f}  H_eff = {H_eff:.2f} / H_q={Hq}'
          f'  （H_eff = 相关系数矩阵特征值的参与率；H_eff≪H_q ⇒ 头高度冗余）')
    print(f'  逐头偏移 alpha_h: min={alpha.min():.3f} max={alpha.max():.3f}'
          f'  （贴共享子空间的头 vs 偏离的头）')

    # ---------------- Part C: 逐头对齐 rho_h 谱（indexer 是「另训小模型」）----------------
    print('\n== Part C: 逐头对齐 rho_h = corr(主注意力头打分, indexer 打分) —— 本来就不均 ==')
    # indexer 打分：从共享子空间读出的「另训」单头打分器 + 自身噪声
    shared = (Qheads.mean(0) @ Kheads.mean(0).t()) / (D ** 0.5)          # [T,C] 共享分量
    torch.manual_seed(321)
    I_raw = standardize(shared) + 0.6 * standardize(torch.randn(T, C, device=dev))
    Ihat = standardize(I_raw)
    rho_h = torch.stack([torch.corrcoef(torch.stack([A_real[h].reshape(-1), Ihat.reshape(-1)]))[0, 1]
                         for h in range(Hq)])
    order = torch.argsort(rho_h, descending=True)
    print(f'  rho_h: min={rho_h.min():+.3f} max={rho_h.max():+.3f} std={rho_h.std():.3f} '
          f'mean={rho_h.mean():+.3f}  （不是单一 rho：是一个谱）')
    print(f'  最贴 4 头 rho = {[round(float(rho_h[i]), 3) for i in order[:4]]}'
          f'   最松 4 头 rho = {[round(float(rho_h[i]), 3) for i in order[-4:]]}')
    print(f'  corr(alpha_h, rho_h) = {torch.corrcoef(torch.stack([alpha, rho_h]))[0,1]:+.3f}'
          f'  （偏移越大、与 indexer 越不贴 ⇒ rho 越低）')

    # ---------------- 头结构 → 下游注意力 ----------------
    tau_h = torch.logspace(-1.0, 1.0, Hq, device=dev)     # 逐头温度：越平熵越大（与 nu 解耦）
    A = A_real.permute(1, 0, 2).contiguous() / tau_h.view(1, -1, 1)   # [T,Hq,C]
    torch.manual_seed(7)
    V = torch.randn(C, Hkv, dv, device=dev, dtype=torch.float32)
    vscale = torch.rand(Hkv, device=dev) * 3.5 + 0.3
    V = V * vscale.view(1, Hkv, 1)
    nu = vscale.repeat_interleave(args.gqa)
    P = torch.softmax(A, dim=2)
    ent_h = (-(P * (P + 1e-12).log()).sum(2)).mean(0)
    print('\n== 头统计 ==')
    print(f'  熵 ent_h: min={ent_h.min():.3f} max={ent_h.max():.3f}'
          f'   值范数 nu_h: min={nu.min():.3f} max={nu.max():.3f}')
    print(f'  corr(ent, nu) = {torch.corrcoef(torch.stack([ent_h, nu]))[0,1]:+.3f}')

    # ---------------- Part D: 真实头结构上重跑 lambda 方案 ----------------
    print('\n== Part D: 真实低秩头结构上，各逐头权重 lambda → 共享支持集 Top-k(p_lambda) → 下游误差 ==')
    lam_uniform = torch.ones(Hq, device=dev)
    lam_entropy = ent_h / ent_h.mean()
    lam_inv_ent = (1.0 / ent_h) / (1.0 / ent_h).mean()
    lam_value = nu / nu.mean()
    lam_sharp = torch.softmax(-ent_h * 4, 0) * Hq
    schemes = {'uniform    ': lam_uniform, 'entropy(+) ': lam_entropy,
               'inv-entropy': lam_inv_ent, 'value-norm ': lam_value, 'sharp-only ': lam_sharp}
    O_full = full_attention(P, V, Hkv)
    err_h_cache = {}
    for name, lam in schemes.items():
        S = topk_indices(make_target(P, lam), K)
        O_s = sparse_attention(A, V, S, Hkv)
        err_h_cache[name] = per_head_err(O_s, O_full)
        print(f'  λ={name}  δ_equal={weighted_output_err(O_s, O_full, torch.ones(Hq, device=dev)):.4e}'
              f'  δ_value={weighted_output_err(O_s, O_full, nu):.4e}')
    O_or = oracle_attention(A, V, K, Hkv)
    print(f'  oracle（逐头各自 top-k，头共享下界） δ_equal='
          f'{weighted_output_err(O_or, O_full, torch.ones(Hq, device=dev)):.4e}'
          f'  δ_value={weighted_output_err(O_or, O_full, nu):.4e}')

    # ---------------- Part E: 头间相关 vs 逐头权重收益 / 头共享税 ----------------
    print('\n== Part E: 头间相关（共享子空间 rank）扫描 —— 相关越强，逐头权重越无所谓、头共享税越小 ==')
    print('  （每行：rank、头间平均相关、H_eff、按 ν 的收益、按熵的惩罚、头共享税 oracle vs shared）')
    for rank in (1, 2, 4, 8, 16, 32):
        Qr, Kr, ar = build_real_qk(T, C, Hq, Hkv, D, rank, dev)
        Ar = head_scores(Qr, Kr, Hkv, D)
        mo, he, _ = head_corr_stats(Ar)
        tau_r = torch.logspace(-1.0, 1.0, Hq, device=dev)
        Arr = Ar.permute(1, 0, 2).contiguous() / tau_r.view(1, -1, 1)   # [T,Hq,C]
        Pr = torch.softmax(Arr, dim=2)
        ent_r = (-(Pr * (Pr + 1e-12).log()).sum(2)).mean(0)
        Of = full_attention(Pr, V, Hkv)
        d = {}
        for name, lam in (('uniform    ', torch.ones(Hq, device=dev)),
                          ('entropy(+) ', ent_r / ent_r.mean()),
                          ('value-norm ', nu / nu.mean())):
            S = topk_indices(make_target(Pr, lam), K)
            d[name] = weighted_output_err(sparse_attention(Arr, V, S, Hkv), Of, nu)
        Oor = oracle_attention(Arr, V, K, Hkv)
        tax = (weighted_output_err(sparse_attention(Arr, V, topk_indices(make_target(Pr, torch.ones(Hq, device=dev)), K), Hkv),
                                   Of, nu) - weighted_output_err(Oor, Of, nu)) / weighted_output_err(Oor, Of, nu)
        pen = (d['entropy(+) '] - d['uniform    ']) / d['uniform    ']
        gain = (d['uniform    '] - d['value-norm ']) / d['uniform    ']
        print(f'  rank={rank:2d}: 头间相关={mo:+.3f} H_eff={he:5.2f}  ν收益={gain:+.3%}  熵罚={pen:+.3%}'
              f'  头共享税={tax:+.2%}')

    # ---------------- Part F: 熵仍是诊断量、ν 仍是（弱）权重 ----------------
    print('\n== Part F: 熵是诊断量、ν 是权重——在真实头结构上复核第 175 篇 ==')
    eh_u = err_h_cache['uniform    ']
    print(f'  corr(ent_h, 均匀支持下的逐头误差 err_h) = {torch.corrcoef(torch.stack([ent_h, eh_u]))[0,1]:+.3f}')
    print(f'  corr(ν_h,  均匀支持下的逐头误差 err_h) = {torch.corrcoef(torch.stack([nu, eh_u]))[0,1]:+.3f}')
    print(f'  corr(rho_h, 均匀支持下的逐头误差 err_h) = {torch.corrcoef(torch.stack([rho_h, eh_u]))[0,1]:+.3f}'
          f'  （新变量：与 indexer 越贴的头，误差越小）')

    # ---------------- Part G: lambda = nu^p 扫描 ----------------
    print('\n== Part G: lambda = ν^p 扫描（真实头结构；p=0 均匀、p=1 值范数）==')
    for p in (0.0, 0.5, 1.0, 2.0, 3.0):
        lam_p = (nu ** p) / (nu ** p).mean() if p > 0 else lam_uniform
        S = topk_indices(make_target(P, lam_p), K)
        print(f'  p={p:.1f}: δ_value={weighted_output_err(sparse_attention(A, V, S, Hkv), O_full, nu):.4e}')


if __name__ == '__main__':
    main()
