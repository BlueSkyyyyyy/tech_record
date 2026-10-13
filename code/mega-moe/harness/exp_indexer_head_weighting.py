"""P165-1：indexer 蒸馏 loss 的【逐头权重】该按什么来——熵、值范数，还是均匀？

第 173 篇（P163-1）量到 DSA 主注意力的**逐头敏感度差异极大**（最平 1/4 头输出误差是最尖
1/4 头的 246×，熵-误差 corr +0.82），并在 §4 提出「KL 蒸馏的权重/诊断应落在**平坦头**上」。
第 174 篇（P164-1）则发现 indexer **自身**的逐头权重 `w_h` 已经编码了重要性、逐头分精度无用。
两条线指向同一个没被回答的问题：**indexer 是被「主注意力分布的 KL」蒸馏出来的单头打分器，
它一次选出的支持集对全部 H_q 个头共享——那么构造蒸馏目标时，各主注意力头该按什么权重进
这个目标？** 按熵（把平坦头顶上去，第 173 篇的直觉）？按值范数？还是干脆均匀？

本篇把它做成能核对的数：**逐头权重 λ_h 决定 indexer 学到的目标分布 `p_λ = Σ_h λ_h P_h`
（KL 不动点），支持集 = `Top-k(p_λ)`；再拿真实下游输出误差（逐头 softmax / GQA 共享）去评判
哪个 λ 好。** 结论是一个**修正**：选择该按 **下游输出重要度 ν_h（≈值范数²）** 加权，而**不是
按熵**；当各头下游等权时**均匀就是最优**，按熵加权反而**变差**。熵是「误差出现在哪」的诊断量，
不是「该选什么」的权重。

真实 SM90 `fp8_fp4_mqa_logits` 锚定 fp8 档；MXFP8/MXFP4 用官方 torch 复刻（SM100 专属，仅分析）。
全部纯 torch + 真实 SM90 kernel，H100(sm90) 真机可跑。
"""
import argparse
import torch
import deep_gemm
from deep_gemm.utils import (per_token_cast_to_fp8, per_token_cast_to_fp4,
                             cast_back_from_fp8, cast_back_from_fp4,
                             per_custom_dims_cast_to_fp8)


# ---------------------------------------------------------------------------
# indexer 打分（与第 171~174 篇一致，真实 SM90 kernel 锚定 fp8 档）
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


# ---------------------------------------------------------------------------
# 真实头结构主注意力：per-head softmax + GQA/MQA 共享支持集
# ---------------------------------------------------------------------------
def build_scores(Ihat, eps, rho, gain, D):
    """a = (1/sqrt(D)) * [sqrt(D) * (rho*Ihat + sqrt(1-rho^2)*g_h*eps)]，逐头标准量 ~ N(0,1)。"""
    sig = (rho * Ihat.unsqueeze(1)
           + (1.0 - rho ** 2) ** 0.5 * gain.view(1, -1, 1) * eps)
    return sig


def full_attention(P, V, Hkv, chunk=2048):
    """O_full[t,h,:] = sum_s P[t,h,s] V[s,h//G,:]。P[T,Hq,C]；V[C,Hkv,dv]。"""
    T, Hq, C = P.shape
    G = Hq // Hkv
    dv = V.shape[2]
    O = torch.zeros(T, Hq, dv, dtype=torch.float32, device=P.device)
    for c0 in range(0, C, chunk):
        c1 = min(c0 + chunk, C)
        for g in range(Hkv):
            Pg = P[:, g * G:(g + 1) * G, c0:c1]              # [T,G,c]
            O[:, g * G:(g + 1) * G, :] += torch.einsum('tgc,cd->tgd', Pg, V[c0:c1, g, :].float())
    return O


def sparse_attention(A, V, support, Hkv, chunk=64):
    """O_S[t,h,:] = sum_{s in S_t} softmax_S(a[t,h,:]) V[s,h//G,:]。"""
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
    """逐头相对 L2（对 token 求平均）：返回 [Hq]。"""
    return ((O_s - O_f).norm(dim=dim) / O_f.norm(dim=dim).clamp_min(1e-30)).mean(0)


def rel_l2(x, y):
    return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()


def weighted_output_err(O_s, O_f, nu):
    """按逐头重要度 nu 加权的总输出误差（sqrt(Σ nu||Δ||² / Σ nu||O||²)）。"""
    num = ((O_s - O_f) ** 2).sum(dim=2).mean(0)              # [Hq]
    den = (O_f ** 2).sum(dim=2).mean(0).clamp_min(1e-30)
    return ((nu * num).sum() / (nu * den).sum()).sqrt().item()


# ---------------------------------------------------------------------------
def make_target(P, lam):
    """p_lambda[t,s] = Σ_h λ_h P_h[t,s] / Σ_h λ_h。P[T,Hq,C]、lam[Hq]。"""
    return torch.einsum('thc,h->tc', P, lam) / lam.sum()


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
    ap.add_argument('--gqa', type=int, default=4)            # G = Hq/Hkv
    ap.add_argument('--rho', type=float, default=0.9)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, Hq, D, dv, K = args.seq, args.kv, args.heads, args.dim, args.dv, args.k
    Hi, Di = args.iheads, args.idim
    Hkv = Hq // args.gqa
    print('# indexer distillation per-head loss weighting: which heads should the shared target up-weight?')
    print(f'# T={T} C={C} H_q={Hq} H_kv={Hkv} G={args.gqa} D={D} dv={dv} topk={K} / indexer(H={Hi},d={Di})')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    qi = torch.randn(T, Hi, Di, device=dev, dtype=torch.bfloat16)
    kvi = torch.randn(C, Di, device=dev, dtype=torch.bfloat16)
    w = torch.randn(T, Hi, device=dev, dtype=torch.float32)
    ks = torch.zeros(T, dtype=torch.int, device=dev)
    ke = torch.full((T,), C, dtype=torch.int, device=dev)

    # ---------------- Part A: kernel 锚定 + 支持集 recall ----------------
    print('\n== Part A: 真实 SM90 fp8_fp4_mqa_logits 锚定 + 三档支持集 ==')
    q_in = (qi.to(torch.float8_e4m3fn), None)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kvi, (0,), False)
    logits_kernel = deep_gemm.fp8_fp4_mqa_logits(
        q=q_in, kv=(kv_fp8, kv_sf), weights=w,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=C)
    logits_teacher = mqa_logits(qi.float(), kvi.float(), w)
    q_hat8, kv_hat8 = quant_q_kv(qi, kvi, 'fp8')
    logits_ref8 = mqa_logits(q_hat8, kv_hat8, w)
    print(f'  kernel vs torch-fp8 ref: max-abs-rel = '
          f'{(logits_kernel.float() - logits_ref8).abs().max().item() / logits_ref8.abs().max().item():.3e}')
    teacher_idx = topk_indices(logits_teacher, K)
    variants = {}
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        qh, kvh = quant_q_kv(qi, kvi, fmt)
        variants[fmt] = topk_indices(mqa_logits(qh, kvh, w), K)
        print(f'  [{fmt:6s}] top-{K} recall = {recall_jaccard(teacher_idx, variants[fmt]):.4f}')

    # ---------------- 主注意力头结构（per-head 温度 + GQA）----------------
    Ihat = (logits_teacher - logits_teacher.mean(1, keepdim=True)) / logits_teacher.std(1, keepdim=True).clamp_min(1e-6)
    torch.manual_seed(123)
    eps = torch.randn(T, Hq, C, device=dev, dtype=torch.float32)
    tau_h = torch.logspace(-1.0, 1.0, Hq, device=dev)        # 逐头温度：越大越平（熵越大）
    A = build_scores(Ihat, eps, args.rho, torch.ones(Hq, device=dev), D) / tau_h.view(1, -1, 1)

    # 逐头值范数：让下游重要度 ν_h 与熵【解耦】（值大小随机，与尖锐度无关）
    torch.manual_seed(7)
    V = torch.randn(C, Hkv, dv, device=dev, dtype=torch.float32)
    vscale = torch.rand(Hkv, device=dev) * 3.5 + 0.3        # 逐 KV 组值尺度随机（宽谱，让 ν 差异明显）
    V = V * vscale.view(1, Hkv, 1)
    nu = vscale.repeat_interleave(args.gqa)                 # 逐 query 头重要度 ν_h ≈ ||V||²

    P = torch.softmax(A, dim=2)                             # [T,Hq,C]

    ent_h = (-(P * (P + 1e-12).log()).sum(2)).mean(0)       # [Hq]
    print('\n== Part B: 逐头统计（熵 = 尖锐度；ν = 下游输出重要度）==')
    print(f'  熵 ent_h:  min={ent_h.min():.3f} max={ent_h.max():.3f} mean={ent_h.mean():.3f}')
    print(f'  值范数 ν_h: min={nu.min():.3f} max={nu.max():.3f} mean={nu.mean():.3f}')
    print(f'  corr(ent, ν) = {torch.corrcoef(torch.stack([ent_h, nu]))[0, 1]:+.3f}'
          f'  （≈0 ⇒ 熵与下游重要度解耦，可分开检验）')

    # ---------------- Part C: 各逐头权重方案 → 支持集 → 下游误差 ----------------
    print('\n== Part C: 逐头权重方案 λ → 共享支持集 Top-k(p_λ) → 下游输出误差 ==')
    print('  （δ_equal = 各头等权输出误差；δ_value = 按 ν 加权输出误差；越低越好）')
    lam_uniform = torch.ones(Hq, device=dev)
    lam_entropy = ent_h / ent_h.mean()
    lam_inv_ent = (1.0 / ent_h) / (1.0 / ent_h).mean()
    lam_value = nu / nu.mean()
    lam_sharp = torch.softmax(-ent_h * 4, 0) * Hq
    schemes = {
        'uniform    ': lam_uniform,
        'entropy(+) ': lam_entropy,
        'inv-entropy': lam_inv_ent,
        'value-norm ': lam_value,
        'sharp-only ': lam_sharp,
    }
    O_full = full_attention(P, V, Hkv)
    err_h_cache = {}
    for name, lam in schemes.items():
        p_lam = make_target(P, lam)
        S = topk_indices(p_lam, K)
        O_s = sparse_attention(A, V, S, Hkv)
        eh = per_head_err(O_s, O_full)
        err_h_cache[name] = eh
        print(f'  λ={name}  δ_equal={weighted_output_err(O_s, O_full, torch.ones(Hq, device=dev)):.4e}'
              f'  δ_value={weighted_output_err(O_s, O_full, nu):.4e}'
              f'  （recall vs uniform = {recall_jaccard(S, topk_indices(make_target(P, lam_uniform), K)):.4f}）')

    # 逐头 oracle：每头各自 top-k（不可实现，头共享的下界）
    print('\n== Part C2: 逐头 oracle（每头各自 top-k，头共享的理论下界）==')
    O_or = torch.zeros_like(O_full)
    rows = torch.arange(T, device=dev)[:, None]
    for h in range(Hq):
        S_h = topk_indices(A[:, h, :], K)
        a_h = torch.gather(A[:, h, :], 1, S_h)                # [T,K]
        p_h = torch.softmax(a_h, dim=1)
        vg = V[S_h, h // (Hq // Hkv), :].float()             # [T,K,dv]
        O_or[:, h, :] = torch.einsum('tk,tkd->td', p_h, vg)
    print(f'  oracle      δ_equal={weighted_output_err(O_or, O_full, torch.ones(Hq, device=dev)):.4e}'
          f'  δ_value={weighted_output_err(O_or, O_full, nu):.4e}')

    # ---------------- Part C3: ρ 扫描——头越不一致，逐头权重越重要 ----------------
    print('\n== Part C3: ρ 扫描——ρ 越小头之间越不一致，逐头权重的影响越大 ==')
    print('  （报 δ_value；gain = (δ_entropy − δ_value)/δ_value，正 = 按熵加权更差）')
    for rho in (1.0, 0.9, 0.7, 0.5):
        A_r = build_scores(Ihat, eps, rho, torch.ones(Hq, device=dev), D) / tau_h.view(1, -1, 1)
        P_r = torch.softmax(A_r, dim=2)
        O_f = full_attention(P_r, V, Hkv)
        dv = {}
        for name, lam in (('uniform    ', lam_uniform), ('entropy(+) ', lam_entropy),
                          ('value-norm ', lam_value)):
            S = topk_indices(make_target(P_r, lam), K)
            dv[name] = weighted_output_err(sparse_attention(A_r, V, S, Hkv), O_f, nu)
        pen = (dv['entropy(+) '] - dv['uniform    ']) / dv['uniform    ']
        gain = (dv['uniform    '] - dv['value-norm ']) / dv['uniform    ']
        print(f'  rho={rho:.1f}: uniform={dv["uniform    "]:.4e} entropy={dv["entropy(+) "]:.4e} '
              f'value={dv["value-norm "]:.4e}  熵罚={pen:+.3%}  值范数收益={gain:+.3%}')

    # ---------------- Part C4: λ = ν^p 扫描——值范数方向、但可过度聚焦 ----------------
    print('\n== Part C4: λ = ν^p 扫描（p=0 均匀、p=1 值范数）；δ_value 随聚焦强度单调下降至饱和 ==')
    for p in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
        lam_p = (nu ** p) / (nu ** p).mean() if p > 0 else lam_uniform
        S = topk_indices(make_target(P, lam_p), K)
        print(f'  p={p:.1f}: δ_value={weighted_output_err(sparse_attention(A, V, S, Hkv), O_full, nu):.4e}'
              f'  δ_equal={weighted_output_err(sparse_attention(A, V, S, Hkv), O_full, torch.ones(Hq, device=dev)):.4e}')

    # ---------------- Part D: 最优 λ 的经验反推 ----------------
    print('\n== Part D: 由实测逐头误差反推「最优权重」并与熵/ν 对照 ==')
    eh_u = err_h_cache['uniform    ']
    print(f'  corr(ent_h, 均匀支持下的逐头误差 err_h) = {torch.corrcoef(torch.stack([ent_h, eh_u]))[0, 1]:+.3f}'
          f'  （复现第 173 篇：熵是「误差出现在哪」的诊断量）')
    print(f'  corr(ν_h,  均匀支持下的逐头误差 err_h) = {torch.corrcoef(torch.stack([nu, eh_u]))[0, 1]:+.3f}')

    # ---------------- Part E: KL 不动点自检（λ 确实决定 indexer 学到的目标）----------------
    print('\n== Part E: KL 不动点自检 —— 训练 indexer 打分 I 去匹配 p_λ（证明 λ 决定学到什么）==')
    # 稠密 KL：min_I KL(p_λ || softmax(I))；最优 softmax(I)=p_λ ⇒ 排序（=top-k）与 p_λ 一致。
    # 真实训练用「支持集内稀疏 KL + 每步重选 top-k」，其不动点的排序与此相同。
    lam = lam_value
    p_lam = make_target(P, lam)
    S_val = topk_indices(p_lam, K)
    I = torch.randn(T, C, device=dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([I], lr=0.5)
    rec0 = recall_jaccard(topk_indices(I.detach(), K), S_val)
    for it in range(400):
        opt.zero_grad()
        loss = -(p_lam * torch.log_softmax(I, dim=1)).sum(1).mean()
        loss.backward()
        opt.step()
    I_final = I.detach()
    S_final = topk_indices(I_final, K)
    qf = torch.softmax(I_final, dim=1)
    print(f'  收敛后 KL={loss.item():.3e}  |  支持集 recall: 训练前={rec0:.4f} → 训练后 vs Top-k(p_λ)='
          f'{recall_jaccard(S_final, S_val):.4f}')
    print(f'  ||softmax(I) − p_λ||₁ (mean/token) = {(qf - p_lam).abs().sum(1).mean().item():.3e}'
          f'  （→0 坐实：逐头权重 λ 唯一决定 indexer 学到的目标排序）')

    # ---------------- Part F: 量化档下 λ 的影响是否被淹没 ----------------
    print('\n== Part F: 量化档下逐头权重的影响 vs 量化噪声 ==')
    for fmt in ('fp8', 'mxfp4'):
        qh, kvh = quant_q_kv(qi, kvi, fmt)
        I_q = mqa_logits(qh, kvh, w)
        # 用同一套头结构、但 indexer 分数换量化档（Ihat 重算）
        Ih_q = (I_q - I_q.mean(1, keepdim=True)) / I_q.std(1, keepdim=True).clamp_min(1e-6)
        A_q = build_scores(Ih_q, eps, args.rho, torch.ones(Hq, device=dev), D) / tau_h.view(1, -1, 1)
        P_q = torch.softmax(A_q, dim=2)
        for name in ('uniform    ', 'entropy(+) ', 'value-norm '):
            S = topk_indices(make_target(P_q, schemes[name]), K)
            O_s = sparse_attention(A_q, V, S, Hkv)
            print(f'  [{fmt:6s}] λ={name} δ_value={weighted_output_err(O_s, O_full, nu):.4e}')

    # ---------------- Part G: 同头一致（所有头等重要度）时均匀就是最优 ----------------
    print('\n== Part G: 若各头下游等权（ν≡const），均匀应最优、按熵加权应变差 ==')
    nu_eq = torch.ones(Hq, device=dev)
    for name in ('uniform    ', 'entropy(+) ', 'inv-entropy', 'value-norm '):
        p_lam = make_target(P, schemes[name])
        S = topk_indices(p_lam, K)
        O_s = sparse_attention(A, V, S, Hkv)
        print(f'  λ={name} δ_equal={weighted_output_err(O_s, O_full, nu_eq):.4e}')


if __name__ == '__main__':
    main()
