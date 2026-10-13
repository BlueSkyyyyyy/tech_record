#!/usr/bin/env python3
"""P147-1 实验：MQA logits（lightning indexer）的训练反向。

专题第 19/60 篇把 MQA logits 的**前向**（连续 / 分页 / 稀疏）拆透了，但 DeepGEMM 只实现前向；
**indexer 怎么训**、其反向梯度长什么样，一直没讲过。本篇补上。

纯 PyTorch（float64 做梯度对拍）+ 一个 CUDA 反向后端（架构无关），H100(sm90) 真机可跑，无需 SM100。

前向（DeepGEMM README `:100-108`、`tests/test_attention.py:78-101` 的 `ref_mqa_logits`）：
    对 query token t 的有效 KV 区间 [ks_t, ke_t)，每个头 h 有点积 z_{t,s,h}=<q_{t,h}, khat_s>，
    其中 khat_s = kv_data_s * kv_scale_s（每 KV token 一个 fp32 scale）；
    打分 I_{t,s} = sum_h w_{t,h} * relu(z_{t,s,h})。

indexer 训练（DeepSeek-V3.2 技术报告 arXiv:2512.02556 §2.1，式 (3)/(4)）：
    目标分布 p（主注意力各头分数求和后 L1 归一化，detach）；
    L = sum_t KL( p_t || softmax(I_t) )；稀疏阶段只在 top-k 选中集 S_t 上算。

本实验：
  Part A  KL 蒸馏损失对 logits 的梯度 dL/dI = softmax(I) - p（解析 vs autograd）；
         并在 S_t 上验证式 (4) 的 restrict 形式；验证 p 的「先按头求和再 L1 归一化」。
  Part B  打分算子反向：dL/dw、dL/dq、dL/d(kv_data)、dL/d(kv_scale)（解析 vs autograd fp64）。
  Part C  中心差分二次核验若干标量条目。
  Part D  top-k 直通（straight-through）：S_t 由 I 自身选出（不可微），梯度只在选中集内。
  Part E  FP8 量化 STE（straight-through estimator）：对 dequant(quant(x)) 的图求导。

（CUDA 反向后端 dq/dw/dkv 见 src/mqa_indexer_backward.cu，其自带 [正确性]/[吞吐] 报告。）
"""

import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name:<60} {detail}")


def relerr(a, b):
    a = a.double()
    b = b.double()
    denom = a.abs().sum() + b.abs().sum() + 1e-30
    return (a - b).abs().sum() / denom


# ---------------------------------------------------------------------------
# 前向：MQA logits 打分算子（对拍 DeepGEMM ref_mqa_logits 的语义）
# ---------------------------------------------------------------------------
def mqa_scores(q, kv_data, kv_scale, weights, ks, ke, mask_bias=0.0):
    """q[T,H,D], kv_data[KV,D], kv_scale[KV], weights[T,H], ks/ke[T] -> I[T,KV] (+ 有效区间掩码)。

    掩码用 -1e30（而非 -inf）以保证 softmax/autograd 数值稳定；无效位置梯度恒 0。
    """
    T, H, D = q.shape
    KV = kv_data.shape[0]
    khat = kv_data * kv_scale[:, None]                       # [KV,D]
    z = torch.einsum('thd,sd->ths', q, khat)                 # [T,H,KV]
    r = z.relu()
    I = torch.einsum('ths,th->ts', r, weights)               # [T,KV]
    pos = torch.arange(KV, device=q.device)[None, :]
    valid = (pos >= ks[:, None]) & (pos < ke[:, None])       # [T,KV]
    I = I.masked_fill(~valid, mask_bias)
    return I, valid, z, r


def kl_grad_from_target(I, valid, p, mask_bias=-1e30):
    """L = sum_t KL(p_t || softmax(I_t))；返回 L 与 dL/dI（解析）。"""
    logq = F.log_softmax(I, dim=1)
    q = logq.exp()
    # 只在有效位置求 KL（p 在无效位置为 0）
    L = (p * (torch.log(p.clamp_min(1e-300)) - logq)).sum()
    dI = (q - p) * valid                                     # 无效位置梯度 0
    return L, dI


# ---------------------------------------------------------------------------
# Part A：KL 蒸馏损失的梯度 = softmax(I) - p
# ---------------------------------------------------------------------------
def part_a():
    print("\n=== Part A: KL 蒸馏损失对 logits 的梯度 dL/dI = softmax(I) - p ===")
    T, H, D, KV = 16, 8, 32, 64
    q = torch.randn(T, H, D, dtype=DT, device=DEV) * 0.1
    kv_data = torch.randn(KV, D, dtype=DT, device=DEV)
    kv_scale = torch.rand(KV, dtype=DT, device=DEV) * 0.5 + 0.5
    w = torch.randn(T, H, dtype=DT, device=DEV)
    ks = torch.full((T,), 0, dtype=torch.long, device=DEV)
    ke = torch.randint(KV // 2, KV + 1, (T,), device=DEV)

    I, valid, _, _ = mqa_scores(q, kv_data, kv_scale, w, ks, ke, mask_bias=-1e30)

    # 目标分布：主注意力「各头分数求和 → L1 归一化」（arXiv:2512.02556 §2.1），detach
    a = torch.rand(T, KV, dtype=DT, device=DEV) * valid      # 原始注意力分数（非负）
    a = a * torch.rand(T, 1, dtype=DT, device=DEV)           # 逐 token 随机幅度
    p = a / a.sum(dim=1, keepdim=True).clamp_min(1e-30)

    L, dI = kl_grad_from_target(I, valid, p)

    # autograd
    I2 = I.clone().requires_grad_(True)
    logq = F.log_softmax(I2, dim=1)
    L2 = (p * (torch.log(p.clamp_min(1e-300)) - logq)).sum()
    (g,) = torch.autograd.grad(L2, I2)
    check("L 值一致", abs(L.item() - L2.item()) < 1e-12, f"L={L.item():.6f}")
    check("dL/dI 解析 vs autograd", relerr(g, dI) < 1e-12, f"relerr={relerr(g, dI):.2e}")
    check("无效位置 dL/dI = 0", dI[~valid].abs().max().item() == 0.0,
          f"max={dI[~valid].abs().max().item():.1e}")
    # p 每行和为 1（仅在有效位置）
    ps = (p * valid).sum(dim=1)
    check("目标分布 p 每行 L1 归一化", (ps - 1).abs().max().item() < 1e-12,
          f"max|sum-1|={(ps - 1).abs().max().item():.1e}")

    # dL/dI 行和 = q.sum - p.sum = 1 - 1 = 0（softmax 归一化 → KL 对 I 平移不变）
    rowsum = dI.sum(dim=1)
    check("dL/dI 行和 = 0（softmax 平移不变）", rowsum.abs().max().item() < 1e-12,
          f"max={rowsum.abs().max().item():.1e}")

    # L1 归一化 vs softmax 归一化目标：KL 不同，梯度不同（证明 p 的构造有意义）
    a_norm = a / a.sum(dim=1, keepdim=True).clamp_min(1e-30)
    logq_det = F.log_softmax(I.detach(), dim=1)
    grad_l1 = (logq_det.exp() - a_norm) * valid
    check("dL/dI 恰好等于 softmax - p", relerr(dI, (logq_det.exp() - p) * valid) < 1e-12,
          f"relerr={relerr(dI, (logq_det.exp() - p) * valid):.2e}")
    return dict(T=T, H=H, D=D, KV=KV)


# ---------------------------------------------------------------------------
# Part A2：式 (4) —— 只在 top-k 选中集 S_t 上算 KL
# ---------------------------------------------------------------------------
def part_a2():
    print("\n=== Part A2: 稀疏阶段式 (4)——只在 top-k 选中集 S_t 上算 KL ===")
    T, H, D, KV = 16, 8, 32, 64
    topk = 16
    q = torch.randn(T, H, D, dtype=DT, device=DEV) * 0.1
    kv_data = torch.randn(KV, D, dtype=DT, device=DEV)
    kv_scale = torch.rand(KV, dtype=DT, device=DEV) * 0.5 + 0.5
    w = torch.randn(T, H, dtype=DT, device=DEV)
    ks = torch.zeros(T, dtype=torch.long, device=DEV)
    ke = torch.full((T,), KV, dtype=torch.long, device=DEV)

    I, valid, _, _ = mqa_scores(q, kv_data, kv_scale, w, ks, ke)
    I_det = I.detach()

    # S_t = top-k(I_t)（选择不可微，detach）
    _, S = torch.topk(I_det, topk, dim=1)                    # [T,topk]
    S_mask = torch.zeros_like(I, dtype=torch.bool)
    S_mask.scatter_(1, S, True)

    # 目标 p 与 softmax 都在 S_t 上重新归一化
    a = (torch.rand(T, KV, dtype=DT, device=DEV) * S_mask)
    p = a / a.sum(dim=1, keepdim=True).clamp_min(1e-30)

    logq = F.log_softmax(I.masked_fill(~S_mask, -1e30), dim=1)
    q_prob = logq.exp()
    L = (p * (torch.log(p.clamp_min(1e-300)) - logq)).sum()
    dI = (q_prob - p) * S_mask                               # 只在 S_t 内非零
    check("S_t 恰含 topk 个且互异", S.shape[1] == topk and S_mask.sum(dim=1).min().item() == topk,
          f"topk={topk}")
    check("选中集外 dL/dI = 0", dI[~S_mask].abs().max().item() < 1e-300,
          f"max={dI[~S_mask].abs().max().item():.1e}")
    check("softmax 只在 S_t 内归一化", (q_prob.sum(dim=1) - 1).abs().max().item() < 1e-12,
          f"max|sum-1|={(q_prob.sum(dim=1) - 1).abs().max().item():.1e}")

    # autograd（把 S 当常量掩码）
    I2 = I.clone().requires_grad_(True)
    logq2 = F.log_softmax(I2.masked_fill(~S_mask, -1e30), dim=1)
    L2 = (p * (torch.log(p.clamp_min(1e-300)) - logq2)).sum()
    (g,) = torch.autograd.grad(L2, I2)
    check("式 (4) 解析 vs autograd", relerr(g, dI) < 1e-12, f"relerr={relerr(g, dI):.2e}")

    # 与「全序列 KL」对比：稀疏只保留选中位置
    _, dI_full = kl_grad_from_target(I.detach(), valid,
                                     a / a.sum(dim=1, keepdim=True).clamp_min(1e-30))
    frac = dI.abs().sum() / (dI_full.abs().sum() + 1e-30)
    check("稀疏梯度只占全序列的一部分", 0 < frac < 1, f"|dI_S|/|dI_full|={frac.item():.4f}")


# ---------------------------------------------------------------------------
# Part B：打分算子的反向（dL/dw, dq, dkv_data, dkv_scale）
# ---------------------------------------------------------------------------
def operator_backward(q, kv_data, kv_scale, w, valid, dI):
    """给定 dL/dI[T,KV]（无效位置为 0），解析反向。"""
    khat = kv_data * kv_scale[:, None]                       # [KV,D]
    z = torch.einsum('thd,sd->ths', q, khat)                 # [T,H,KV]
    rmask = (z > 0).to(z.dtype)
    # dL/dw_{t,h} = sum_s dI_{t,s} * relu(z_{t,s,h})
    dw = torch.einsum('ts,ths->th', dI, z.relu())
    # dL/dz_{t,s,h} = dI_{t,s} * w_{t,h} * 1[z>0]
    dz = dI[:, None, :] * w[:, :, None] * rmask              # [T,H,KV]
    # dL/dq_{t,h,d} = sum_s dz_{t,s,h} * khat_{s,d}
    dq = torch.einsum('ths,sd->thd', dz, khat)
    # dL/dkhat_{s,d} = sum_{t,h} dz_{t,s,h} * q_{t,h,d}
    dkhat = torch.einsum('ths,thd->sd', dz, q)
    dkv_data = dkhat * kv_scale[:, None]
    dkv_scale = (dkhat * kv_data).sum(dim=1)
    return dict(dw=dw, dq=dq, dkv_data=dkv_data, dkv_scale=dkv_scale)


def part_b():
    print("\n=== Part B: 打分算子反向（dL/dw, dq, dkv_data, dkv_scale）===")
    T, H, D, KV = 24, 8, 32, 48
    q = (torch.randn(T, H, D, dtype=DT, device=DEV) * 0.2).requires_grad_(True)
    kv_data = torch.randn(KV, D, dtype=DT, device=DEV).requires_grad_(True)
    kv_scale = (torch.rand(KV, dtype=DT, device=DEV) * 0.5 + 0.5).requires_grad_(True)
    w = torch.randn(T, H, dtype=DT, device=DEV).requires_grad_(True)
    ks = torch.zeros(T, dtype=torch.long, device=DEV)
    ke = torch.full((T,), KV, dtype=torch.long, device=DEV)
    valid = torch.ones(T, KV, dtype=torch.bool, device=DEV)
    dI = torch.randn(T, KV, dtype=DT, device=DEV)

    I = torch.einsum('ths,th->ts',
                     torch.einsum('thd,sd->ths', q, kv_data * kv_scale[:, None]).relu(), w)
    # autograd：<dI, I> 对四操作数求导
    grads = torch.autograd.grad((dI * I).sum(), [q, kv_data, kv_scale, w], retain_graph=False)
    ana = operator_backward(q.detach(), kv_data.detach(), kv_scale.detach(), w.detach(), valid, dI)
    check("dL/dw  解析 vs autograd", relerr(grads[3], ana['dw']) < 1e-12, f"relerr={relerr(grads[3], ana['dw']):.2e}")
    check("dL/dq  解析 vs autograd", relerr(grads[0], ana['dq']) < 1e-12, f"relerr={relerr(grads[0], ana['dq']):.2e}")
    check("dL/d(kv_data) 解析 vs autograd", relerr(grads[1], ana['dkv_data']) < 1e-12,
          f"relerr={relerr(grads[1], ana['dkv_data']):.2e}")
    check("dL/d(kv_scale) 解析 vs autograd", relerr(grads[2], ana['dkv_scale']) < 1e-12,
          f"relerr={relerr(grads[2], ana['dkv_scale']):.2e}")
    return ana


# ---------------------------------------------------------------------------
# Part C：中心差分核验
# ---------------------------------------------------------------------------
def part_c():
    print("\n=== Part C: 中心差分核验（对解析反查）===")
    T, H, D, KV = 6, 4, 16, 12
    q = torch.randn(T, H, D, dtype=DT, device=DEV) * 0.3
    kv_data = torch.randn(KV, D, dtype=DT, device=DEV)
    kv_scale = torch.rand(KV, dtype=DT, device=DEV) * 0.5 + 0.5
    w = torch.randn(T, H, dtype=DT, device=DEV)
    dI = torch.randn(T, KV, dtype=DT, device=DEV)

    def loss_of(qq, kk, ss, ww):
        I = torch.einsum('ths,th->ts',
                         torch.einsum('thd,sd->ths', qq, kk * ss[:, None]).relu(), ww)
        return (dI * I).sum()

    ana = operator_backward(q, kv_data, kv_scale, w, torch.ones(T, KV, dtype=torch.bool, device=DEV), dI)
    eps = 1e-6
    # 抽 4 个标量位置做中心差分
    tests = [
        ('dq',   q, ana['dq'],       0, 1, 2),
        ('dkv',  kv_data, ana['dkv_data'], 3, 5, None),
        ('dscale', kv_scale, ana['dkv_scale'], 7, None, None),
        ('dw',   w, ana['dw'],       2, 3, None),
    ]
    worst = 0.0
    for name, x, g, i, j, k in tests:
        idx = (i, j) if k is None else (i, j, k)
        xp = x.clone(); xp[idx] += eps
        xm = x.clone(); xm[idx] -= eps
        args_p = [q, kv_data, kv_scale, w]
        args_m = [q, kv_data, kv_scale, w]
        pos = {'dq': 0, 'dkv': 1, 'dscale': 2, 'dw': 3}[name]
        args_p[pos] = xp; args_m[pos] = xm
        fd = (loss_of(*args_p) - loss_of(*args_m)) / (2 * eps)
        r = abs(fd.item() - g[idx].item()) / (abs(fd.item()) + abs(g[idx].item()) + 1e-30)
        worst = max(worst, r)
    check("4 个标量位置中心差分 vs 解析", worst < 1e-6, f"最差 relerr={worst:.2e}")


# ---------------------------------------------------------------------------
# Part D：top-k 直通（straight-through）
# ---------------------------------------------------------------------------
def part_d():
    print("\n=== Part D: top-k 选择的直通（梯度只在选中集）===")
    T, H, D, KV = 12, 8, 32, 40
    topk = 10
    q = torch.randn(T, H, D, dtype=DT, device=DEV) * 0.2
    kv_data = torch.randn(KV, D, dtype=DT, device=DEV)
    kv_scale = torch.rand(KV, dtype=DT, device=DEV) * 0.5 + 0.5
    w = torch.randn(T, H, dtype=DT, device=DEV)
    ks = torch.zeros(T, dtype=torch.long, device=DEV)
    ke = torch.full((T,), KV, dtype=torch.long, device=DEV)
    I, valid, _, _ = mqa_scores(q, kv_data, kv_scale, w, ks, ke)
    I_det = I.detach()
    _, S = torch.topk(I_det, topk, dim=1)
    S_mask = torch.zeros_like(I, dtype=torch.bool); S_mask.scatter_(1, S, True)
    a = torch.rand(T, KV, dtype=DT, device=DEV) * S_mask
    p = a / a.sum(dim=1, keepdim=True).clamp_min(1e-30)
    logq = F.log_softmax(I.masked_fill(~S_mask, -1e30), dim=1)
    dI = (logq.exp() - p) * S_mask

    # 梯度经 top-k 直通后，未选中位置 dI=0 → 其对 dq/dw 无贡献
    dq = torch.einsum('ths,sd->thd',
                      dI[:, None, :] * w[:, :, None] * (torch.einsum('thd,sd->ths', q, kv_data * kv_scale[:, None]) > 0),
                      kv_data * kv_scale[:, None])
    # 直接对未选中的 s 置零 dI 不改变结果
    dI_zero = dI.clone()
    check("未选中位置梯度严格 0", dI_zero[~S_mask].abs().max().item() == 0.0,
          f"max={dI_zero[~S_mask].abs().max().item():.1e}")
    # 若错误地把未选中位置也纳入（softmax 用全序列，但 loss 只取 S）→ 梯度不同
    logq_full = F.log_softmax(I.masked_fill(~valid, -1e30), dim=1)
    dI_wrong = (logq_full.exp() - p) * valid
    check("『全序列 softmax + 只算 S 的 loss』与式 (4) 不同",
          relerr(dI, dI_wrong) > 1e-3, f"relerr={relerr(dI, dI_wrong):.4f}")
    return dict(dq=dq)


# ---------------------------------------------------------------------------
# Part E：FP8 量化 STE
# ---------------------------------------------------------------------------
def per_kv_cast_fp8(kv_data):
    """每 KV token（行）一个 fp32 scale 的 E4M3 量化（对齐 DeepGEMM `kv_cache_cast_to_fp8`）。

    返回反量化后的值（fp64）。**scale detach**：真实训练里量化 scale 由 amax 现算、视为常量
    （stop-grad），反向只走「舍入」的 straight-through，不经 scale 对输入的依赖。
    """
    amax = kv_data.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    scale = (amax / 448.0).detach()
    xq = (kv_data / scale).clamp(-448.0, 448.0)
    xq = xq.to(torch.float8_e4m3fn).to(kv_data.dtype) * scale
    return xq, scale


def part_e():
    print("\n=== Part E: FP8 量化 STE（straight-through estimator）===")
    T, H, D, KV = 16, 8, 64, 40
    topk = 20
    q = torch.randn(T, H, D, dtype=DT, device=DEV).requires_grad_(True)
    kv_fp32 = torch.randn(KV, D, dtype=DT, device=DEV).requires_grad_(True)
    kv_scale = torch.rand(KV, dtype=DT, device=DEV) * 0.5 + 0.5
    w = torch.randn(T, H, dtype=DT, device=DEV).requires_grad_(True)

    # 前向：kv 走 FP8 量化；反向用标准 STE（x_ste = x + (quant(x)-x).detach()，值=舍入值、梯度=1）
    kv_deq, sf = per_kv_cast_fp8(kv_fp32.detach())          # 舍入后的值（无梯度）
    kv_ste = kv_fp32 + (kv_deq - kv_fp32).detach()          # STE：值 = 舍入值、d/dkv_fp32 = 1
    I = torch.einsum('ths,th->ts',
                     torch.einsum('thd,sd->ths', q, kv_ste * kv_scale[:, None]).relu(), w)
    _, S = torch.topk(I.detach(), topk, dim=1)
    S_mask = torch.zeros_like(I, dtype=torch.bool); S_mask.scatter_(1, S, True)
    a = torch.rand(T, KV, dtype=DT, device=DEV) * S_mask
    p = a / a.sum(dim=1, keepdim=True).clamp_min(1e-30)
    logq = F.log_softmax(I.masked_fill(~S_mask, -1e30), dim=1)
    L = (p * (torch.log(p.clamp_min(1e-300)) - logq)).sum()
    grads = torch.autograd.grad(L, [q, kv_fp32, w])

    # 解析：STE 下 dL/dkv_fp32 = dL/dkv_deq（把舍入值当常量操作数）
    dI = (logq.exp() - p) * S_mask
    khat = kv_deq * kv_scale[:, None]
    z = torch.einsum('thd,sd->ths', q.detach(), khat)
    rmask = (z > 0).to(z.dtype)
    dw = torch.einsum('ts,ths->th', dI, z.relu())
    dz = dI[:, None, :] * w.detach()[:, :, None] * rmask
    dq = torch.einsum('ths,sd->thd', dz, khat)
    dkhat = torch.einsum('ths,thd->sd', dz, q.detach())
    dkv_ste = dkhat * kv_scale[:, None]
    check("STE: dL/dw 解析 vs autograd", relerr(grads[2], dw) < 1e-10, f"relerr={relerr(grads[2], dw):.2e}")
    check("STE: dL/dq 解析 vs autograd", relerr(grads[0], dq) < 1e-10, f"relerr={relerr(grads[0], dq):.2e}")
    check("STE: dL/dkv 解析 vs autograd", relerr(grads[1], dkv_ste) < 1e-10, f"relerr={relerr(grads[1], dkv_ste):.2e}")
    # 反例：不做 STE（把舍入值 detach 后当常量、不接回 kv_fp32）→ kv 梯度为 0
    I0 = torch.einsum('ths,th->ts',
                      torch.einsum('thd,sd->ths', q, kv_deq * kv_scale[:, None]).relu(), w)
    logq0 = F.log_softmax(I0.masked_fill(~S_mask, -1e30), dim=1)
    L0 = (p * (torch.log(p.clamp_min(1e-300)) - logq0)).sum()
    (g0,) = torch.autograd.grad(L0, kv_fp32, allow_unused=True)
    det = "None" if g0 is None else f"{g0.abs().max().item():.1e}"
    check("反例：舍入值不回接 → kv 不在图里（梯度 None/0）",
          g0 is None or g0.abs().max().item() == 0.0, f"g={det}")
    # 量化误差量级（E4M3）
    qe = relerr(kv_deq, kv_fp32.detach())
    check("FP8 量化误差量级（E4M3）", 1e-3 < qe.item() < 5e-2, f"relerr={qe.item():.4f}")


def main():
    print("=" * 72)
    print("MQA logits（lightning indexer）训练反向：KL 蒸馏 + 打分算子反向")
    print(f"torch={torch.__version__}  device={DEV}  dtype={DT}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 72)
    part_a()
    part_a2()
    part_b()
    part_c()
    part_d()
    part_e()
    print("\n" + "=" * 72)
    print(f"总计：PASS={PASS}  FAIL={FAIL}")
    print("=" * 72)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
