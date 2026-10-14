"""P168-1：indexer 的「可训练容量」——把 KL 蒸馏真的跑起来，indexer 需要多少秩？

第 175 篇（P165-1）用**自由打分** `I` 做稠密 KL 训练，证明不动点 `softmax_S(I)=p_λ`
（`p_λ=Σ_h λ_h P_h/Σ_h λ_h`）——即 `λ` 唯一决定 indexer 学到的排序；但那是**解析推导 + 一次性拟合**
（`docs/175`：支持集 recall 从随机 0.125 收敛到 1.0000）。第 166 篇（P166-1）用**真实低秩 q/k**
量到主注意力头高度冗余（`H_eff=1.97/32`）、逐头对齐 `ρ_h` 是一个谱，并留口子：
**「用真实训练（非参数化）的 indexer 复核」**。

本篇把这层补上：**真的用梯度下降训一个 indexer**（DSA 训练里 indexer 正是由 KL 蒸馏另训的小模型），
追问三件事：
  ① **优化不是瓶颈**：自由 logits `I∈R^{T×C}` 用 SGD 训到 KL=0、recall=1.0，坐实第 175 篇不动点
     （不只是解析，是真能收敛）。
  ② **表示才是瓶颈**：把 indexer 限制成**低秩** `I = U Vᵀ`（rank R），它最多能保住多少 top-k——
     给出 `recall(R)` 曲线与「达到 FP8 级 recall 所需的最小秩 `R*`」。
  ③ **`R*` 由主注意力的内在秩决定**：扫共享子空间 rank（头相关/冗余度），`R*` 应与
     `log p_λ` 的谱秩和 `H_eff` 同向。

真实 SM90 `fp8_fp4_mqa_logits` 锚定 fp8 档；训练/评估全纯 torch。H100(sm90) 真机可跑。
"""
import argparse
import torch
import torch.nn.functional as F
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8


# ---------------------------------------------------------------------------
# indexer 打分算子（真实 SM90 kernel 锚定，与第 171~177 篇同款）
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


def standardize(x, dim=-1):
    mu = x.mean(dim, keepdim=True)
    sd = x.std(dim, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


def topk_idx(x, k):
    return x.topk(k, dim=1).indices


def recall(a, b):
    T, k = a.shape
    al, bl = a.tolist(), b.tolist()
    inter = sum(len(set(al[i]) & set(bl[i])) for i in range(T))
    return inter / (T * k)


# ---------------------------------------------------------------------------
# 真实主注意力头结构：低秩共享 q/k + 逐头偏移（第 166 篇同款）
# ---------------------------------------------------------------------------
def build_real_qk(T, C, Hq, Hkv, D, rank, dev, seed=2024):
    torch.manual_seed(seed)
    U = torch.linalg.qr(torch.randn(D, rank, device=dev)).Q.t()   # [rank, D]
    Kbase = (torch.randn(C, rank, device=dev) @ U)                # [C, D]
    Qbase = (torch.randn(T, rank, device=dev) @ U)                # [T, D]
    alpha = torch.logspace(-1.5, 0.5, Hq, device=dev)
    Kheads = Kbase.unsqueeze(0) + alpha[:Hkv].view(Hkv, 1, 1) * torch.randn(Hkv, C, D, device=dev)
    Qheads = Qbase.unsqueeze(0) + alpha.view(Hq, 1, 1) * torch.randn(Hq, T, D, device=dev)
    return Qheads, Kheads, alpha


def head_scores(Qheads, Kheads, Hkv, D):
    Hq = Qheads.shape[0]
    G = Hq // Hkv
    out = torch.empty(Hq, Qheads.shape[1], Kheads.shape[1], device=Qheads.device)
    for h in range(Hq):
        out[h] = standardize(Qheads[h] @ Kheads[h // G].t() / (D ** 0.5), dim=1)
    return out


def head_corr_stats(A):
    Hq = A.shape[0]
    flat = A.reshape(Hq, -1)
    flat = flat - flat.mean(1, keepdim=True)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-30)
    Cm = flat @ flat.t()
    off = Cm - torch.eye(Hq, device=A.device) * Cm.diag()
    mean_off = off.sum() / (Hq * (Hq - 1))
    ev = torch.linalg.eigvalsh(Cm).clamp_min(0)
    H_eff = (ev.sum() ** 2) / (ev ** 2).sum()
    return mean_off.item(), H_eff.item()


def head_scores_with_tau(A_real, tau_h):
    return A_real.permute(1, 0, 2).contiguous() / tau_h.view(1, -1, 1)   # [T,Hq,C]


def mixture_target(P, lam):
    """p_λ = Σ_h λ_h P_h / Σ_h λ_h（逐 token 在 C 上归一化）。P=[T,Hq,C]。"""
    return torch.einsum('thc,h->tc', P, lam) / lam.sum()


# ---------------------------------------------------------------------------
# 训练：自由 logits 或低秩 indexer，最小化 KL(p_λ || softmax(I)) = CE
# ---------------------------------------------------------------------------
def excess_kl(target, I):
    """KL(p_λ || softmax(I)) = CE(p_λ, softmax(I)) - H(p_λ)。"""
    H = -(target * (target + 1e-12).log()).sum(1).mean()
    CE = -(target * F.log_softmax(I, dim=1)).sum(1).mean()
    return (CE - H).item(), H.item()


def train_indexer(target, rank=0, steps=400, lr=0.05, seed=0, eval_every=100,
                  recall_k=2048, verbose=False):
    T, C = target.shape
    dev = target.device
    torch.manual_seed(seed)
    full = rank <= 0
    if full:
        U = torch.nn.Parameter(torch.zeros(T, C, device=dev))
        params = [U]

        def get_I():
            return U
    else:
        U = torch.nn.Parameter(torch.randn(T, rank, device=dev) * 0.05)
        V = torch.nn.Parameter(torch.randn(C, rank, device=dev) * 0.05)
        params = [U, V]

        def get_I():
            return U @ V.t()
    opt = torch.optim.Adam(params, lr=lr)
    tgt_idx = topk_idx(target, recall_k)
    hist = []
    for step in range(1, steps + 1):
        I = get_I()
        loss = -(target * F.log_softmax(I, dim=1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                I_e = get_I().detach()
                kl, _ = excess_kl(target, I_e)
                r = recall(topk_idx(I_e, recall_k), tgt_idx)
            hist.append((step, kl, r))
            if verbose:
                print(f'    step {step:4d}: KL={kl:.3e}  recall@k={r:.4f}')
    I = get_I().detach()
    return I, hist


def svd_recall(target, rank, recall_k):
    """Eckart-Young 秩-R 近似 log p_λ 的 recall（与训练对照）。"""
    X = target
    Xc = X - X.mean(0, keepdim=True)
    U, S, V = torch.svd_lowrank(Xc, q=rank)
    Xr = (U * S) @ V.t()
    return recall(topk_idx(Xr, recall_k), topk_idx(target, recall_k))


def spectrum_energy_rank(target, frac=0.99):
    Xc = target - target.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    cum = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return int((cum < frac).sum().item()) + 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--kv', type=int, default=16384)
    ap.add_argument('--heads', type=int, default=32)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--idim', type=int, default=64)
    ap.add_argument('--iheads', type=int, default=32)
    ap.add_argument('--gqa', type=int, default=4)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, Hq, D, dv, K = args.seq, args.kv, args.heads, args.dim, 64, args.k
    Hi, Di = args.iheads, args.idim
    Hkv = Hq // args.gqa
    print('# indexer TRAINED by KL distillation — how much rank does it need?')
    print(f'# T={T} C={C} H_q={Hq} H_kv={Hkv} G={args.gqa} D={D} topk={K} shared_rank={args.rank}'
          f' / indexer(H={Hi},d={Di})')
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
    q_hat = qi.to(torch.float8_e4m3fn).float()
    kv_hat = kv_fp8.float() * kv_sf.unsqueeze(1)
    logits_ref = mqa_logits(q_hat, kv_hat, w)
    print(f'  kernel vs torch-fp8 ref: max-abs-rel = '
          f'{(logits_kernel.float() - logits_ref).abs().max().item() / logits_ref.abs().max().item():.3e}')

    # ---------------- Part B: 真实低秩头结构 + 蒸馏目标 p_λ ----------------
    print('\n== Part B: 真实低秩共享 q/k —— 头冗余、log p_λ 的谱秩与蒸馏目标 ==')
    Qheads, Kheads, alpha = build_real_qk(T, C, Hq, Hkv, D, args.rank, dev)
    A_real = head_scores(Qheads, Kheads, Hkv, D)
    mean_off, H_eff = head_corr_stats(A_real)
    tau_h = torch.logspace(-1.0, 1.0, Hq, device=dev)
    A = head_scores_with_tau(A_real, tau_h)
    P = torch.softmax(A, dim=2)
    lam_uniform = torch.ones(Hq, device=dev)
    target = mixture_target(P, lam_uniform)
    ent_h = (-(P * (P + 1e-12).log()).sum(2)).mean(0)
    arch = spectrum_energy_rank(target, 0.99)
    H_tgt = (-(target * (target + 1e-12).log()).sum(1)).mean().item()
    print(f'  头间平均相关={mean_off:+.3f}  H_eff={H_eff:.2f}/{Hq}'
          f'  （头冗余 ⇒ 需要 indexer 分解的方向少）')
    print(f'  log p_λ（= 蒸馏目标）的 99% 能量秩 = {arch} / {T}  （内在秩；即便共享子空间只 rank={args.rank}，'
          f'softmax 混合也让目标不再是低秩）')
    print(f'  目标熵 H(p_λ) = {H_tgt:.4f}（CE 的地板，须从「KL」里扣掉）')
    print(f'  目标支持集 top-{K}: 逐头熵 ent_h min={ent_h.min():.3f} max={ent_h.max():.3f}')

    # ---------------- Part C: 自由 logits 训练 —— 优化不是瓶颈 ----------------
    print('\n== Part C: 自由 logits I∈R^{T×C} 训练（第 175 篇不动点的 SGD 复核）==')
    I_full, hist = train_indexer(target, rank=0, steps=args.steps, lr=args.lr,
                                 eval_every=max(1, args.steps // 4), recall_k=K, verbose=True)
    print(f'  → 自由 logits 训到 KL={hist[-1][1]:.2e}、recall@k = {hist[-1][2]:.4f}'
          f'（随机基线 = k/C = {K/C:.4f}）⇒ 优化能到不动点，瓶颈不在优化')

    # ---------------- Part D: 低秩 indexer —— 表示才是瓶颈 ----------------
    print('\n== Part D: 低秩 indexer I=U Vᵀ（rank R）—— recall(R) 与最小充足秩 R* ==')
    print(f'  {"R":>4} {"KLafter":>10} {"recall@k":>9} {"svd_recall":>10}')
    R_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    r_star = {}
    svd_best = {}
    # 先备好下游张量（Part G 用同一套），此处只为 recall 曲线
    for R in R_list:
        Ir, h = train_indexer(target, rank=R, steps=args.steps, lr=args.lr,
                              eval_every=args.steps, recall_k=K)
        svr = svd_recall(target, R, K)
        svd_best[R] = svr
        kl = h[-1][1]
        rr = h[-1][2]
        for thr in (0.95, 0.96, 0.99):
            if thr not in r_star and rr >= thr:
                r_star[thr] = R
        print(f'  {R:>4} {kl:>10.3e} {rr:>9.4f} {svr:>10.4f}')
    print(f'  最小充足秩：R*(recall>=0.95)={r_star.get(0.95)}  R*(>=0.96,=FP8级)={r_star.get(0.96)}'
          f'  R*(>=0.99)={r_star.get(0.99)}   内在秩(99%能量)={arch}')

    # ---------------- Part E: R* 由内在秩/头冗余决定 ----------------
    print('\n== Part E: 扫共享子空间 rank（头相关度）—— R* 与内在秩同向 ==')
    print(f'  {"srank":>5} {"corr":>7} {"H_eff":>7} {"intr_rank":>9} {"R*(0.95)":>8} {"R*(0.99)":>9} {"rec@R=16":>9}')
    for srank in (1, 2, 4, 8, 16, 32):
        Qr, Kr, _ = build_real_qk(T, C, Hq, Hkv, D, srank, dev)
        Ar = head_scores(Qr, Kr, Hkv, D)
        mo, he = head_corr_stats(Ar)
        Arr = head_scores_with_tau(Ar, tau_h)
        Pr = torch.softmax(Arr, dim=2)
        tg = mixture_target(Pr, lam_uniform)
        ir = spectrum_energy_rank(tg, 0.99)
        rs95 = rs99 = None
        rec16 = None
        for R in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            _, h = train_indexer(tg, rank=R, steps=max(150, args.steps // 2), lr=args.lr,
                                 eval_every=max(150, args.steps // 2), recall_k=K)
            if R == 16:
                rec16 = h[-1][2]
            if rs95 is None and h[-1][2] >= 0.95:
                rs95 = R
            if rs99 is None and h[-1][2] >= 0.99:
                rs99 = R
        print(f'  {srank:>5} {mo:>+7.3f} {he:>7.2f} {ir:>9d} {str(rs95):>8} {str(rs99):>9} {rec16:>9.4f}')

    # ---------------- Part F: 蒸馏目标的逐头权重 λ 在训练后还有多少影响 ----------------
    print('\n== Part F: 蒸馏目标 p_λ 的逐头权重 λ —— 训练后对 recall 的影响 ==')
    nu_v = torch.rand(Hkv, device=dev) * 3.5 + 0.3
    nu = nu_v.repeat_interleave(args.gqa)
    schemes = {'uniform': lam_uniform,
               'entropy': ent_h / ent_h.mean(),
               'value   ': nu / nu.mean()}
    for name, lam in schemes.items():
        tg = mixture_target(P, lam)
        _, hf = train_indexer(tg, rank=0, steps=max(200, args.steps // 2), lr=args.lr,
                              eval_every=max(200, args.steps // 2), recall_k=K)
        _, hl = train_indexer(tg, rank=8, steps=max(200, args.steps // 2), lr=args.lr,
                              eval_every=max(200, args.steps // 2), recall_k=K)
        print(f'  λ={name}: 自由 recall={hf[-1][2]:.4f}  低秩(R=8) recall={hl[-1][2]:.4f}')

    # ---------------- Part G: 训练后 indexer 下游误差 vs 逐头 oracle ----------------
    print('\n== Part G: 训练后 indexer 支持集的下游误差（per-head softmax + GQA 共享）==')
    V = torch.randn(C, Hkv, dv, device=dev, dtype=torch.float32)
    vscale = torch.rand(Hkv, device=dev) * 3.5 + 0.3
    V = V * vscale.view(1, Hkv, 1)
    G = args.gqa

    def full_attn(Pp, chunk=2048):
        O = torch.zeros(T, Hq, dv, device=dev)
        for c0 in range(0, C, chunk):
            c1 = min(c0 + chunk, C)
            for g in range(Hkv):
                Pg = Pp[:, g * G:(g + 1) * G, c0:c1]
                O[:, g * G:(g + 1) * G] += torch.einsum('tgc,cd->tgd', Pg, V[c0:c1, g, :].float())
        return O

    def shared_attn(support):
        O = torch.zeros(T, Hq, dv, device=dev)
        rows = torch.arange(T, device=dev)[:, None, None]
        for g in range(Hkv):
            hi = torch.arange(g * G, (g + 1) * G, device=dev)[None, :, None]
            ag = A[rows, hi, support[:, None, :]]
            p = torch.softmax(ag, dim=2)
            O[:, g * G:(g + 1) * G] = torch.einsum('cgk,ckd->cgd', p, V[support, g, :].float())
        return O

    def per_head_oracle():
        O = torch.zeros(T, Hq, dv, device=dev)
        for h in range(Hq):
            S = topk_idx(A[:, h, :], K)
            p = torch.softmax(torch.gather(A[:, h, :], 1, S), dim=1)
            O[:, h, :] = torch.einsum('tk,tkd->td', p, V[S, h // G, :].float())
        return O

    O_full = full_attn(P)
    O_or = per_head_oracle()
    lam_val = nu / nu.mean()
    tg_val = mixture_target(P, lam_val)
    for R in (0, 2, 8, 32, 64):
        I, h = train_indexer(tg_val, rank=R, steps=args.steps, lr=args.lr,
                             eval_every=args.steps, recall_k=K)
        S = topk_idx(I, K)
        rel = ((shared_attn(S) - O_full).norm() / O_full.norm()).item()
        print(f'  indexer rank={R:>2}: recall={h[-1][2]:.4f}  输出 relL2={rel:.4e}')
    rel_or = ((O_or - O_full).norm() / O_full.norm()).item()
    print(f'  逐头 oracle（头共享下界）: 输出 relL2={rel_or:.4e}')


if __name__ == '__main__':
    main()
