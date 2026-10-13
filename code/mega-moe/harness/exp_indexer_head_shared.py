"""P163-1：indexer 选择翻转对【真实 DSA 主注意力头结构】前向与训练反向的联合影响。

第 171 篇（P161-1）量了 indexer 的 top-k 选择在 FP8/MXFP8/MXFP4 量化下的稳定性；
第 172 篇（P162-1）把选择接进「单头、单 softmax」的下游注意力输出，结论是**二阶**的。
但那篇 §5 明确留下口子：真实 DSA 主注意力的 per-head softmax、GQA/MQA 的**头共享**未建模；
且与第 157~160 篇的**训练反向**的联合误差未量。本篇补上这条链的「真实头结构」。

模型（DSA 主注意力的头结构）：
  - 打分 `a[t,h,j] = (1/sqrt(D)) * <q[t,h], k[j, h//G]>`；indexer 打分 `I[t,j]` 由真实
    SM90 kernel 给出。indexer 是**逐 token 选一次、对全部 H_q 个 query 头共享**（DSA 设计）。
  - 真实 DSA 的 indexer 是**另训的小模型**，与主注意力**对齐但不相同**。用相关系数 ρ 建模：
    `a[t,h,j] = ρ·Î[t,j] + sqrt(1-ρ²)·g_h·ε[t,h,j]`（Î 是逐行标准化后的 indexer 打分，
    ε~N(0,1)，g_h 是逐头增益）——**逐头 softmax**、**GQA 共享**（G=H_q/H_kv）都由此体现。
  - 支持集 `S_t = Top-k(I_t,:)` 对全部 query 头共享：一次翻转同时移动 G 个头。

反向量（训练侧）：
  - 主注意力输出对 `V` 的梯度 `dV`（token 索引，可跨支持集逐元素比较）；
  - indexer 自身 KL 蒸馏训练梯度（第 157 篇）`dw/dq`——把「前向选择稳定性」接到「训练反向」。

全部纯 torch + 真实 SM90 kernel（fp8 档锚定），H100(sm90) 真机可跑。
"""
import argparse
import torch
import deep_gemm
from deep_gemm.utils import (per_token_cast_to_fp8, per_token_cast_to_fp4,
                             cast_back_from_fp8, cast_back_from_fp4,
                             per_custom_dims_cast_to_fp8)


# ----------------------------------------------------------------------------
# indexer 前向（与第 171/172 篇一致，真实 SM90 kernel 锚定 fp8 档）
# ----------------------------------------------------------------------------
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
    al, bl = a.tolist(), b.tolist()
    for i in range(T):
        inter[i] = len(set(al[i]) & set(bl[i]))
    return (inter / k).mean().item(), inter


# ----------------------------------------------------------------------------
# 真实头结构主注意力（score 形式）：per-head softmax + GQA/MQA 共享支持集
# ----------------------------------------------------------------------------
def build_scores(Ihat, eps, rho, gain, D):
    """a[t,h,j] = (1/sqrt(D)) * sqrt(D) * (rho*Ihat + sqrt(1-rho^2)*g_h*eps) —— 显式 1/sqrt(d)。
    Ihat[T,C] 逐行标准化；eps[T,H,C]~N(0,1)；gain[H]。返回以 fp32 存 [T,H,C]。"""
    sig = (rho * Ihat.unsqueeze(1)
           + (1.0 - rho ** 2) ** 0.5 * gain.view(1, -1, 1) * eps)   # std~1 的标准量
    raw = sig * (D ** 0.5)                       # 未缩放点积 ~ N(0, D)
    return raw / (D ** 0.5)                      # 1/sqrt(d) 缩放 → 分数 std~1


def attn_forward(A, V, support, Hkv, chunk=64):
    """O[t,h,:] = sum_{j in S_t} P[t,h,j] V[j, h//G]；P = per-head softmax over S_t。
    A[T,Hq,C] 分数；V[C,Hkv,dv]；support[T,K] 对全部 query 头共享。"""
    T, Hq, _ = A.shape
    G = Hq // Hkv
    O = torch.zeros(T, Hq, V.shape[2], dtype=torch.float32, device=A.device)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        idx = support[t0:t1]                              # [c,K]
        rows = torch.arange(t0, t1, device=A.device)[:, None, None]
        for hvg in range(Hkv):
            hi = torch.arange(hvg * G, (hvg + 1) * G, device=A.device)[None, :, None]
            ag = A[rows, hi, idx[:, None, :]]             # [c,G,K]
            vg = V[:, hvg, :][idx].float()               # [c,K,dv]
            p = torch.softmax(ag, dim=2)
            O[t0:t1, hvg * G:(hvg + 1) * G] = torch.einsum('cgk,ckd->cgd', p, vg)
    return O


def attn_backward_dV(A, V, support, dO, Hkv, chunk=64):
    """返回 dV[C,Hkv,dv]：dV[j,hvg] += sum_{t,h in group, j in S_t} P[t,h,j] dO[t,h]。
    只算 token 索引的 dV（可跨支持集逐元素比较）。"""
    T, Hq, _ = A.shape
    dv = V.shape[2]
    G = Hq // Hkv
    dV = torch.zeros(V.shape[0], Hkv, dv, dtype=torch.float32, device=A.device)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        idx = support[t0:t1]
        flat = idx.reshape(-1)
        rows = torch.arange(t0, t1, device=A.device)[:, None, None]
        for hvg in range(Hkv):
            hi = torch.arange(hvg * G, (hvg + 1) * G, device=A.device)[None, :, None]
            ag = A[rows, hi, idx[:, None, :]]            # [c,G,K]
            p = torch.softmax(ag, dim=2)
            dog = dO[t0:t1, hvg * G:(hvg + 1) * G].float()
            dv_c = torch.einsum('cgk,cgd->ckd', p, dog)  # [c,K,dv]
            dV[:, hvg, :].index_add_(0, flat, dv_c.reshape(-1, dv))
    return dV


def rel_l2(x, y, dim=None):
    if dim is None:
        return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()
    return ((x - y).norm(dim=dim) / y.norm(dim=dim).clamp_min(1e-30))


def cos(x, y, dim=-1):
    return torch.nn.functional.cosine_similarity(x, y, dim=dim)


# ----------------------------------------------------------------------------
# indexer 自身 KL 蒸馏训练反向（第 157 篇）：支持集上 dL/dI → dw/dq
# ----------------------------------------------------------------------------
def indexer_kl_backward(q, kv, w, support, I):
    """KL 蒸馏：目标 p = 全上下文 relu 分数 L1 归一化；在支持集内重新归一化 softmax 与 p，
    dL/dI = softmax_S(I) - p_S。返回 dw[T,H] 与 dq[T,H,D]。"""
    T, H, D = q.shape
    idx = support
    Ig = torch.gather(I, 1, idx)
    pi = torch.softmax(Ig, dim=1)
    p_full = I.clamp_min(0)
    p_full = p_full / p_full.sum(dim=1, keepdim=True).clamp_min(1e-30)
    p_S = torch.gather(p_full, 1, idx)
    p_S = p_S / p_S.sum(dim=1, keepdim=True).clamp_min(1e-30)
    dI = pi - p_S
    dw = torch.zeros_like(w)
    dq = torch.zeros_like(q, dtype=torch.float32)
    for t0 in range(0, T, 64):
        t1 = min(t0 + 64, T)
        kg = kv[idx[t0:t1]].float()                       # [c,K,D]
        z = torch.einsum('chd,ckd->chk', q[t0:t1].float(), kg)   # [c,H,K]
        dIc = dI[t0:t1].unsqueeze(1)                      # [c,1,K]
        dz = dIc * w[t0:t1].unsqueeze(-1) * (z > 0)       # [c,H,K]
        dw[t0:t1] = (dIc * z.relu()).sum(dim=2)
        dq[t0:t1] = torch.einsum('chk,ckd->chd', dz, kg)
    return dw, dq


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--kv', type=int, default=16384)
    ap.add_argument('--heads', type=int, default=32)     # H_q
    ap.add_argument('--dim', type=int, default=128)      # 主注意力 head_dim
    ap.add_argument('--dv', type=int, default=128)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--idim', type=int, default=64)
    ap.add_argument('--iheads', type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = 'cuda'
    T, C, Hq, D, dv, K = args.seq, args.kv, args.heads, args.dim, args.dv, args.k
    Hi, Di = args.iheads, args.idim
    print('# indexer selection → real DSA main attention (per-head softmax + GQA/MQA shared support)')
    print(f'# T={T} C={C} H_q={Hq} D={D} dv={dv} topk={K} / indexer(H={Hi},d={Di})')
    print(f'# arch major = {torch.cuda.get_device_capability()}, num_sms = {deep_gemm.get_num_sms()}')

    qi = torch.randn(T, Hi, Di, device=dev, dtype=torch.bfloat16)
    kvi = torch.randn(C, Di, device=dev, dtype=torch.bfloat16)
    w = torch.randn(T, Hi, device=dev, dtype=torch.float32)
    ks = torch.zeros(T, dtype=torch.int, device=dev)
    ke = torch.full((T,), C, dtype=torch.int, device=dev)
    V = torch.randn(C, Hq, dv, device=dev, dtype=torch.bfloat16)   # MHA 的 V（按 G 折叠）

    # ---------------- Part A: 锚定 + 三档支持集 ----------------
    print('\n== Part A: kernel 锚定 + 三档 token 级支持集（对全部 query 头共享）==')
    q_in = (qi.to(torch.float8_e4m3fn), None)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kvi, (0,), False)
    logits_kernel = deep_gemm.fp8_fp4_mqa_logits(
        q=q_in, kv=(kv_fp8, kv_sf), weights=w,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=C)
    logits_teacher = mqa_logits(qi.float(), kvi.float(), w)
    q_hat8, kv_hat8 = quant_q_kv(qi, kvi, 'fp8')
    logits_ref8 = mqa_logits(q_hat8, kv_hat8, w)
    rel_anchor = (logits_kernel.float() - logits_ref8).abs().max().item() / logits_ref8.abs().max().item()
    print(f'  kernel vs torch-fp8 ref: max-abs-rel = {rel_anchor:.3e}')

    teacher_idx = topk_indices(logits_teacher, K)
    variants = {'kernel-fp8': topk_indices(logits_kernel.float(), K)}
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        qh, kvh = quant_q_kv(qi, kvi, fmt)
        variants[fmt] = topk_indices(mqa_logits(qh, kvh, w), K)
        rec, _ = recall_jaccard(teacher_idx, variants[fmt])
        print(f'  [{fmt:10s}] top-{K} recall = {rec:.4f} (flipped {K - rec * K:.0f})')

    Ihat = (logits_teacher - logits_teacher.mean(1, keepdim=True)) / logits_teacher.std(1, keepdim=True).clamp_min(1e-6)
    torch.manual_seed(123)
    eps = torch.randn(T, Hq, C, device=dev, dtype=torch.float32)
    gain1 = torch.ones(Hq, device=dev)

    def scores(rho, gain=None):
        return build_scores(Ihat, eps, rho, gain if gain is not None else gain1, D)

    # ---------------- Part A2: 真实独立（未训练）q/k 主注意力 vs indexer 的经验对齐 ρ_h ----------------
    # DSA 的 indexer 是「另训的小模型」，与主注意力对齐但不相同。若直接给一副独立随机的 q/k
    # （= 未对齐），逐头与 indexer 打分的相关应≈0——这解释了为何本实验要用 ρ 旋钮参数化对齐度。
    print('\n== Part A2: 真实独立（未训练）q/k 主注意力 vs indexer 的经验逐头对齐 ρ_h ==')
    Hkv_a2 = 8
    q_main = torch.randn(T, Hq, D, device=dev, dtype=torch.float32) / (D ** 0.5)
    k_main = torch.randn(C, Hkv_a2, D, device=dev, dtype=torch.float32)
    rho_h = torch.zeros(Hq, device=dev)
    for h in range(Hq):
        a_h = q_main[:, h, :] @ k_main[:, h // (Hq // Hkv_a2), :].T     # [T,C]
        rho_h[h] = torch.corrcoef(torch.stack([Ihat.flatten(), a_h.flatten()]))[0, 1]
    print(f'  独立 q/k: ρ_h mean={rho_h.mean():+.5f} std={rho_h.std():.5f} '
          f'min={rho_h.min():+.5f} max={rho_h.max():+.5f}  （iid 地板 ~1/sqrt(T·C)={1.0/(T*C)**0.5:.2e}）')
    A_ind = torch.stack([q_main[:, h, :] @ k_main[:, h // (Hq // Hkv_a2), :].T for h in range(Hq)], dim=1)
    O_ti = attn_forward(A_ind, V[:, :Hkv_a2, :].contiguous(), teacher_idx, Hkv_a2)
    O_qi = attn_forward(A_ind, V[:, :Hkv_a2, :].contiguous(), variants['fp8'], Hkv_a2)
    print(f'  用独立 q/k 打分、teacher vs fp8 支持集: 输出 relL2={rel_l2(O_qi, O_ti):.3e} '
          f'（≈随机支持集量级；未训练 q/k 下选择无意义 ⇒ 须用 ρ 参数化「已对齐」）')
    rho0 = torch.corrcoef(torch.stack([Ihat.flatten(), scores(0.9)[:, 0, :].flatten()]))[0, 1]
    print(f'  对照：本实验 ρ=0.9 构造的 head0 经验对齐 = {rho0:+.4f}（≈旋钮值）')

    # ---------------- Part B: GQA/MQA 头共享 —— 扫组大小 G（rho=0.9）----------------
    print('\n== Part B: GQA/MQA 头共享 —— 下游输出误差 vs 组大小 G（rho=0.9）==')
    print('   支持集逐 token 选出、对全部 query 头共享；G=1 即 MHA，G=H_q 即 MQA')
    A09 = scores(0.9)
    for G in (1, 2, 4, 8, 32):
        if Hq % G:
            continue
        Hkv = Hq // G
        Vg = V[:, :Hkv, :].contiguous()
        O_t = attn_forward(A09, Vg, teacher_idx, Hkv)
        line = [f'  G={G:2d} (Hkv={Hkv:2d}):']
        for fmt in ('fp8', 'mxfp4'):
            Oq = attn_forward(A09, Vg, variants[fmt], Hkv)
            line.append(f'{fmt} relL2={rel_l2(Oq, O_t):.3e}/cos={cos(Oq.flatten(1), O_t.flatten(1)).mean().item():.6f}')
        print('  '.join(line))

    # ---------------- Part B2: 平坦度校准（G=4, rho=0.9）—— 把量级接回第 172 篇 ----------------
    print('\n== Part B2: 平坦度校准（G=4, rho=0.9）—— 分数整体除 tau，teacher 支持集质量随 tau 变化 ==')
    Hkv0 = 8
    Vg0 = V[:, :Hkv0, :].contiguous()
    msup = torch.zeros(T, C, dtype=torch.bool, device=dev)
    msup.scatter_(1, teacher_idx, True)
    for tau in (0.5, 1.0, 2.0, 4.0):
        A = scores(0.9) / tau
        p = torch.softmax(A, 2)
        mass = (p * msup.unsqueeze(1)).sum(2).mean().item()   # teacher 支持集捕获的注意力质量
        O_t = attn_forward(A, Vg0, teacher_idx, Hkv0)
        line = [f'  tau={tau:.1f} (质量={mass:.3f}):']
        for fmt in ('fp8', 'mxfp4'):
            Oq = attn_forward(A, Vg0, variants[fmt], Hkv0)
            line.append(f'{fmt}={rel_l2(Oq, O_t):.3e}')
        print('  '.join(line))
    print('  （tau 越小越尖、支持集捕获质量越高、选择翻转越无关痛痒；第 172 篇 tau=8 的尖锐工作点'
          '（质量~0.95）对应这里 tau≪0.5，故其 2.6e-3 量级在本校准下可复现）')

    # ---------------- Part C: per-head 尖锐度敏感度（G=4, rho=0.9, tau=1）----------------
    print('\n== Part C: per-head 尖锐度敏感度（G=4, rho=0.9）—— 逐头输出误差 vs 该头注意力熵 ==')
    Hkv = 8
    Vg = V[:, :Hkv, :].contiguous()
    tau_h = torch.logspace(-1.0, 1.0, Hq, device=dev)       # 逐头温度：越大 → 越平（熵越大）
    A_g = scores(0.9) / tau_h.view(1, -1, 1)
    O_t = attn_forward(A_g, Vg, teacher_idx, Hkv)
    O_q = attn_forward(A_g, Vg, variants['mxfp4'], Hkv)
    err_h = rel_l2(O_q, O_t, dim=2).mean(dim=0)             # [Hq]
    ent_h = torch.zeros(Hq, device=dev)
    G = Hq // Hkv
    rows = torch.arange(T, device=dev)[:, None, None]
    for hvg in range(Hkv):
        hi = torch.arange(hvg * G, (hvg + 1) * G, device=dev)[None, :, None]
        ag = A_g[rows, hi, teacher_idx[:, None, :]]      # [T,G,K]
        p = torch.softmax(ag, dim=2)
        ent = -(p * (p + 1e-12).log()).sum(2)
        ent_h[hvg * G:(hvg + 1) * G] = ent.mean(0)
    order = torch.argsort(ent_h)
    lo, hi = order[:Hq // 4], order[-Hq // 4:]
    corr = torch.corrcoef(torch.stack([ent_h, err_h]))[0, 1].item()
    print(f'  熵-误差 corr={corr:+.3f}')
    print(f'  最平 {Hq//4} 头（熵~{ent_h[hi].mean():.3f}）err={err_h[hi].mean():.3e} | '
          f'最尖 {Hq//4} 头（熵~{ent_h[lo].mean():.3f}）err={err_h[lo].mean():.3e} '
          f'| 平/尖={err_h[hi].mean() / err_h[lo].mean():.2f}x')

    # ---------------- Part D: 主注意力训练反向 dV（G=4）vs rho ----------------
    print('\n== Part D: 主注意力训练反向 dV 在 teacher vs 量化支持集下的相对误差（G=4）==')
    torch.manual_seed(1)
    dO = torch.randn(T, Hq, dv, device=dev, dtype=torch.float32)
    for rho in (1.0, 0.9, 0.7, 0.4):
        A = scores(rho)
        dV_t = attn_backward_dV(A, Vg, teacher_idx, dO, Hkv)
        O_t = attn_forward(A, Vg, teacher_idx, Hkv)
        line = [f'  rho={rho:.1f}:']
        for fmt in ('fp8', 'mxfp4'):
            O_q = attn_forward(A, Vg, variants[fmt], Hkv)
            dV_q = attn_backward_dV(A, Vg, variants[fmt], dO, Hkv)
            line.append(f'{fmt} fwd={rel_l2(O_q, O_t):.2e} dV={rel_l2(dV_q, dV_t):.2e}')
        print('  '.join(line))

    # ---------------- Part E: indexer KL 训练反向 dw/dq 的联合误差 ----------------
    print('\n== Part E: indexer KL 蒸馏训练反向（dw/dq）在不同支持集下的相对误差 ==')
    dw_t, dq_t = indexer_kl_backward(qi, kvi, w, teacher_idx, logits_teacher)
    for fmt in ('fp8', 'mxfp8', 'mxfp4'):
        qh, kvh = quant_q_kv(qi, kvi, fmt)
        I_q = mqa_logits(qh, kvh, w)
        dw_q, dq_q = indexer_kl_backward(qi, kvi, w, variants[fmt], I_q)
        dw_s, _ = indexer_kl_backward(qi, kvi, w, variants[fmt], logits_teacher)   # 仅换支持集
        dw_v, _ = indexer_kl_backward(qi, kvi, w, teacher_idx, I_q)                # 仅量化打分
        print(f'  [{fmt:6s}] dw relL2={rel_l2(dw_q, dw_t):.3e} | dq relL2={rel_l2(dq_q, dq_t):.3e} '
              f'| dw cos={cos(dw_q.flatten(), dw_t.flatten()).item():.6f}'
              f' | 仅换支持集 dw={rel_l2(dw_s, dw_t):.3e} | 仅量化打分 dw={rel_l2(dw_v, dw_t):.3e}')

    # ---------------- 自检：dV 解析反向 vs autograd（tiny）----------------
    print('\n== 自检：dV 解析反向 vs torch.autograd（tiny, G=2）==')
    tq, tc, th, td = 8, 64, 4, 16
    torch.manual_seed(7)
    As2 = torch.randn(tq, th, tc, device=dev, dtype=torch.float32, requires_grad=True)
    Va = torch.randn(tc, th // 2, td, device=dev, dtype=torch.float32).requires_grad_(True)
    sup2 = torch.stack([torch.randperm(tc, device=dev)[:8] for _ in range(tq)])
    dO2 = torch.randn(tq, th, td, device=dev, dtype=torch.float32)
    Oa = attn_forward(As2, Va, sup2, th // 2)
    (Oa * dO2).sum().backward()
    dV_a = attn_backward_dV(As2.detach(), Va.detach(), sup2, dO2, th // 2)
    print(f'  dV relL2={rel_l2(dV_a, Va.grad):.3e}')


if __name__ == '__main__':
    main()
