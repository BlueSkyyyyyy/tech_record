"""P169-1：真实 indexer 架构的可训练容量 —— ReLU 让打分不再是低秩，秩的故事变了

第 168 篇（P168-1）把 indexer 的 KL 蒸馏真的跑起来，量到「撑 top-k recall 要 R≈64–128」，
但它训的是**自由 logits `I∈R^{T×C}`** 与**低秩 `I=U Vᵀ`**——这两种都**不是生产 indexer 的架构**。
生产 indexer 是一个 MQA 打分算子（README `:88-110`）：

    out_ij = Σ_h relu( q[i,h,:] · kv_j ) · weights[i,h]        （`q = x W_q`、`kv = x W_k`）

关键差别在 **ReLU**：`Σ_h (q_h·k_h) w_h` 是一个**双线性型**（等价于秩 ≤ `H_i·d_i` 的
`U Vᵀ`），但 `Σ_h relu(q_h·k_h) w_h` **逐元素非线性**，`relu` 作用在低秩矩阵上会把秩打满。
于是「indexer 需要多少秩」这个问题本身在真实架构下**提法就不对**——真正的资源是
**内维 `H_i·d_i`（q/k 的投影秩预算）**，不是输出 logits 的秩。

本篇用梯度下降真训四种打分器（同一蒸馏目标、同一优化器），在**参数/内维预算**上对拍：
  A) 自由 logits（上界）
  B) 低秩 `I=U Vᵀ`（秩 R，第 168 篇的模型类）
  C) 线性多头求和 `Σ_h (q_h·k_h) w_h`（= 结构化 `U Vᵀ`，秩 ≤ `H_i·d_i`）
  D) 真实 MQA indexer `Σ_h relu(q_h·k_h) w_h`（README 算子）
问：① 同内维预算下 ReLU 能多拿多少 recall？② 真实架构要多大内维才够？
③ 训好的 logits 秩有多高（ReLU 是否真把秩打满）？④ 下游是否跟上？

真实 SM90 `fp8_fp4_mqa_logits` 锚定 fp8 档；训练/评估全纯 torch。H100(sm90) 真机可跑。
"""
import argparse
import torch
import torch.nn.functional as F
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8


# ---------------------------------------------------------------------------
# indexer 打分算子（真实 SM90 kernel 锚定，与第 171~178 篇同款）
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
# 真实主注意力头结构：低秩共享 q/k + 逐头偏移（第 166/168 篇同款）
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
    return torch.einsum('thc,h->tc', P, lam) / lam.sum()


def excess_kl(target, I):
    H = -(target * (target + 1e-12).log()).sum(1).mean()
    CE = -(target * F.log_softmax(I, dim=1)).sum(1).mean()
    return (CE - H).item(), H.item()


def spec_rank(x, frac=0.99):
    """99% 能量秩（对 row-mean-centered 矩阵做 svdvals）。"""
    Xc = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    cum = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return int((cum < frac).sum().item()) + 1


# ---------------------------------------------------------------------------
# 四种打分器。全部直接可训练，经同一 KL/CE 目标用 Adam 训练。
# ---------------------------------------------------------------------------
class FreeLogits(torch.nn.Module):
    """A) 自由 logits I∈R^{T×C}（上界，无结构约束）。"""
    def __init__(self, T, C, dev):
        super().__init__()
        self.U = torch.nn.Parameter(torch.zeros(T, C, device=dev))

    def forward(self):
        return self.U

    def num_params(self):
        return self.U.numel()


class LowRank(torch.nn.Module):
    """B) 低秩 I = U Vᵀ（第 168 篇的模型类；输出 logits 秩 ≤ R）。"""
    def __init__(self, T, C, R, dev):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(T, R, device=dev) * 0.05)
        self.V = torch.nn.Parameter(torch.randn(C, R, device=dev) * 0.05)

    def forward(self):
        return self.U @ self.V.t()

    def num_params(self):
        return self.U.numel() + self.V.numel()


class MultiHead(torch.nn.Module):
    """C/D) 真实 MQA 结构：Σ_h f(q_h·k_h) w_h，f = relu（真实）或 identity（线性对照）。

    内维预算 = Hi*Di。线性版（无 ReLU）等价于**秩 ≤ Hi*Di 的 U Vᵀ**；
    ReLU 版逐元素非线性，输出秩**不受 Hi*Di 限制**。这是本篇的主轴。
    """
    def __init__(self, T, C, Hi, Di, dev, use_relu=True, use_weights=True, scale=True):
        super().__init__()
        self.Hi, self.Di = Hi, Di
        self.use_relu = use_relu
        self.use_weights = use_weights
        self.scale = scale
        self.q = torch.nn.Parameter(torch.randn(T, Hi, Di, device=dev) * (Di ** -0.5))
        self.k = torch.nn.Parameter(torch.randn(C, Di, device=dev) * 0.5)
        self.w = torch.nn.Parameter(torch.ones(T, Hi, device=dev))

    def forward(self):
        s = torch.einsum('thd,nd->thn', self.q, self.k)
        if self.scale:
            s = s / (self.Di ** 0.5)
        if self.use_relu:
            s = s.relu()
        if self.use_weights:
            return torch.einsum('thn,th->tn', s, self.w)
        return s.sum(dim=1)

    def num_params(self):
        return self.q.numel() + self.k.numel() + (self.w.numel() if self.use_weights else 0)


def train(model, target, steps=400, lr=0.05, recall_k=2048, eval_every=None):
    tgt_idx = topk_idx(target, recall_k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    if eval_every is None:
        eval_every = max(1, steps // 4)
    for step in range(1, steps + 1):
        I = model()
        loss = -(target * F.log_softmax(I, dim=1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                Ie = model().detach()
                kl, _ = excess_kl(target, Ie)
                r = recall(topk_idx(Ie, recall_k), tgt_idx)
            hist.append((step, kl, r))
    I = model().detach()
    return I, hist


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
    ap.add_argument('--ilr', type=float, default=0.03)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, Hq, D, dv, K = args.seq, args.kv, args.heads, args.dim, 64, args.k
    Hi, Di = args.iheads, args.idim
    Hkv = Hq // args.gqa

    print('# indexer 真实架构（MQA + ReLU）的可训练容量：同内维预算下 ReLU 值多少？')
    print(f'# T={T} C={C} H_q={Hq} H_kv={Hkv} G={args.gqa} D={D} topk={K} 主注意力共享 rank={args.rank}'
          f' / 默认 indexer(H={Hi},d={Di})')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    # ---------------- Part A: 真实 SM90 fp8 kernel 锚定 ----------------
    print('\n== Part A: 真实 SM90 fp8_fp4_mqa_logits 锚定（README :88-110 的算子）==')
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

    # ---------------- Part B: 蒸馏目标 p_λ + 头结构 ----------------
    print('\n== Part B: 真实低秩共享 q/k —— 蒸馏目标 p_λ 与头结构 ==')
    Qheads, Kheads, alpha = build_real_qk(T, C, Hq, Hkv, D, args.rank, dev)
    A_real = head_scores(Qheads, Kheads, Hkv, D)
    mean_off, H_eff = head_corr_stats(A_real)
    tau_h = torch.logspace(-1.0, 1.0, Hq, device=dev)
    A = head_scores_with_tau(A_real, tau_h)
    P = torch.softmax(A, dim=2)
    lam = torch.ones(Hq, device=dev)
    target = mixture_target(P, lam)
    arch = spec_rank(target, 0.99)
    H_tgt = (-(target * (target + 1e-12).log()).sum(1)).mean().item()
    print(f'  头间平均相关={mean_off:+.3f}  H_eff={H_eff:.2f}/{Hq}')
    print(f'  蒸馏目标 p_λ 的 99% 能量秩 = {arch} / {T}  （第 168 篇同一目标）')
    print(f'  目标熵 H(p_λ) = {H_tgt:.4f}  随机基线 recall = {K/C:.4f}')

    # ---------------- Part C: 自由 logits 上界 ----------------
    print('\n== Part C: 自由 logits I∈R^{T×C}（上界，第 168 篇 Part C 同款）==')
    m = FreeLogits(T, C, dev)
    I_full, hf = train(m, target, steps=args.steps, lr=args.lr, recall_k=K, eval_every=max(1, args.steps // 4))
    print(f'  自由 logits: KL={hf[-1][1]:.2e}  recall@k={hf[-1][2]:.4f}  参数={m.num_params()/1e6:.2f}M'
          f'  训后 logits 能量秩={spec_rank(I_full, 0.99)}')

    # ---------------- Part D: 主轴 —— 同内维预算，线性 vs ReLU ----------------
    print('\n== Part D: 同内维预算 P=Hi·Di 下，线性多头求和 vs 真实 ReLU indexer ==')
    print('   （Hi=1 固定、Di=P，故线性版输出秩 ≤ P；ReLU 版不受此限）')
    print(f'  {"P":>5} {"params":>9} | {"linear rec":>11} {"linear KL":>10} {"rank_lin":>8}'
          f' | {"relu rec":>9} {"relu KL":>9} {"rank_relu":>9}')
    for Pp in (8, 16, 32, 64, 128, 256):
        lin = MultiHead(T, C, 1, Pp, dev, use_relu=False, use_weights=True)
        ilin, hl = train(lin, target, steps=args.steps, lr=args.ilr, recall_k=K, eval_every=args.steps)
        rk_lin = spec_rank(ilin, 0.99)
        rel = MultiHead(T, C, 1, Pp, dev, use_relu=True, use_weights=True)
        irel, hr = train(rel, target, steps=args.steps, lr=args.ilr, recall_k=K, eval_every=args.steps)
        rk_rel = spec_rank(irel, 0.99)
        print(f'  {Pp:>5} {lin.num_params()/1e6:>8.2f}M | {hl[-1][2]:>11.4f} {hl[-1][1]:>10.3e} {rk_lin:>8d}'
              f' | {hr[-1][2]:>9.4f} {hr[-1][1]:>9.3e} {rk_rel:>9d}')

    # ---------------- Part E: 低秩 U Vᵀ 对照（参数匹配） ----------------
    print('\n== Part E: 低秩 I=U Vᵀ（秩 R）—— 与真实 indexer 在同参数量上对拍 ==')
    print(f'  {"R":>4} {"params":>9} {"recall":>8} {"KL":>10}')
    for R in (8, 16, 32, 64, 95, 128, 256):
        mm = LowRank(T, C, R, dev)
        Im, hm = train(mm, target, steps=args.steps, lr=args.lr, recall_k=K, eval_every=args.steps)
        print(f'  {R:>4} {mm.num_params()/1e6:>8.2f}M {hm[-1][2]:>8.4f} {hm[-1][1]:>10.3e}')

    # ---------------- Part F: 真实 indexer 的 Hi/Di 形状 + 权重开关 ----------------
    print('\n== Part F: 真实 indexer 的形状 (Hi,Di) 与逐头权重 w 的作用 ==')
    print(f'  {"Hi":>3} {"Di":>4} {"P=Hi·Di":>8} {"relu rec":>9} {"no-w rec":>9} {"no-relu rec":>11}')
    for Hi_, Di_ in ((1, 64), (4, 64), (8, 64), (16, 64), (32, 64), (32, 8), (32, 16), (32, 32), (8, 128)):
        base = MultiHead(T, C, Hi_, Di_, dev, use_relu=True, use_weights=True)
        _, hb = train(base, target, steps=args.steps, lr=args.ilr, recall_k=K, eval_every=args.steps)
        nw = MultiHead(T, C, Hi_, Di_, dev, use_relu=True, use_weights=False)
        _, hn = train(nw, target, steps=args.steps, lr=args.ilr, recall_k=K, eval_every=args.steps)
        nl = MultiHead(T, C, Hi_, Di_, dev, use_relu=False, use_weights=True)
        _, hl = train(nl, target, steps=args.steps, lr=args.ilr, recall_k=K, eval_every=args.steps)
        print(f'  {Hi_:>3} {Di_:>4} {Hi_*Di_:>8} {hb[-1][2]:>9.4f} {hn[-1][2]:>9.4f} {hl[-1][2]:>11.4f}')

    # ---------------- Part G: 训好的真实 indexer 下游误差 ----------------
    print('\n== Part G: 训好的真实 indexer 支持集的下游误差（per-head softmax + GQA 共享）==')
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

    O_full = full_attn(P)
    print(f'  {"config":>22} {"recall":>8} {"relL2":>10}')
    for Hi_, Di_ in ((1, 16), (1, 32), (8, 64), (32, 64)):
        mm = MultiHead(T, C, Hi_, Di_, dev, use_relu=True, use_weights=True)
        Im, hm = train(mm, target, steps=args.steps, lr=args.ilr, recall_k=K, eval_every=args.steps)
        S = topk_idx(Im, K)
        rel = ((shared_attn(S) - O_full).norm() / O_full.norm()).item()
        print(f'  {"indexer Hi=%d,Di=%d" % (Hi_, Di_):>22} {hm[-1][2]:>8.4f} {rel:>10.4e}')


if __name__ == '__main__':
    main()
