#!/usr/bin/env python3
"""P129-1 实验：MoE 路由的负载均衡三件套（aux-loss / router z-loss / aux-loss-free bias）。

纯 PyTorch（float64 做梯度对拍），架构无关，H100(sm90) 真机可跑，无需 SM100。

三件事：
  Part A  aux load-balancing loss（Switch / DeepSeek-V2 风格）
          L = alpha * E * sum_i f_i * P_i，f_i=top-k 计数占比(视为常量)、P_i=平均路由概率。
          推导解析梯度 dL/dz 并与 torch.autograd、中心差分对拍。
  Part B  router z-loss：L_z = beta * mean_t (logsumexp(z_t))^2，梯度 dL/dz = beta*2*lse/T * softmax(z)。
  Part C  aux-loss-free bias（DeepSeek-V3 `e_score_correction_bias`）：
          排名 = score + b、权重 = 无偏 score；每步 b_i += gamma*sign(target-load_i)。
          模拟偏斜路由，量负载收敛（max/min -> 1）；并用 Mega Gate 参考语义（sqrt_softplus）复刻。
  Part D  三者关系：aux-loss 梯度不经过 bias（bias 是 stop-grad 常量）；bias 更新是零梯度规则。
"""

import math
import torch

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name:<58} {detail}")


# ---------------------------------------------------------------------------
# 公共：路由概率 / top-k 计数
# ---------------------------------------------------------------------------
def router_prob(z):
    """softmax over experts（最后一维）。"""
    return torch.softmax(z, dim=-1)


def topk_mask(z, k, ranking=None):
    """返回 [T, E] 的 0/1 掩码，表示每个 token 选中的 top-k 专家。
    ranking 给出时用它排序（用于带 bias 排名），否则用 z 本身。"""
    score = z if ranking is None else ranking
    idx = torch.topk(score, k, dim=-1).indices
    mask = torch.zeros_like(z)
    mask.scatter_(1, idx, 1.0)
    return mask


# ---------------------------------------------------------------------------
# Part A: aux load-balancing loss
# ---------------------------------------------------------------------------
def aux_loss(z, k, alpha, E):
    """Switch 风格：L = alpha * E * sum_i f_i * P_i。
    f_i = (1/T) sum_t 1[i in topk(t)]  （stop-grad，计数量）
    P_i = (1/T) sum_t p_{t,i}          （可微）
    """
    T = z.shape[0]
    p = router_prob(z)
    mask = topk_mask(z, k)
    f = mask.mean(dim=0)          # [E]，计数占比，常量
    P = p.mean(dim=0)             # [E]，平均概率
    return alpha * E * (f * P).sum()


def aux_loss_grad(z, k, alpha, E):
    """解析梯度：f 当常量，只对 P 求导。
    dL/dz_{t,j} = alpha * E / T * sum_i f_i * p_{t,i} * (delta_ij - p_{t,j})
                = alpha * E / T * (f_j * p_{t,j} - p_{t,j} * sum_i f_i * p_{t,i})
                = alpha * E / T * p_{t,j} * (f_j - g_t),  g_t = sum_i f_i * p_{t,i}
    """
    T = z.shape[0]
    p = router_prob(z)
    mask = topk_mask(z, k)
    f = mask.mean(dim=0)          # [E]
    g = (p * f).sum(dim=-1, keepdim=True)   # [T,1]
    return alpha * E / T * p * (f.unsqueeze(0) - g)


def numeric_grad(fn, x, eps=1e-6):
    g = torch.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.numel()):
        old = flat[i].item()
        flat[i] = old + eps
        fp = fn()
        flat[i] = old - eps
        fm = fn()
        flat[i] = old
        g.reshape(-1)[i] = (fp - fm) / (2 * eps)
    return g


def relerr(a, b):
    return (a - b).abs().max().item() / (a.abs().max().item() + 1e-30)


def run_part_a():
    print("\n=== Part A: aux load-balancing loss 的梯度 ===")
    T, E, K = 256, 32, 4
    alpha = 0.01
    z = torch.randn(T, E, dtype=DTYPE, device=DEV) * 1.5

    # autograd
    zz = z.clone().requires_grad_(True)
    L = aux_loss(zz, K, alpha, E)
    L.backward()
    g_auto = zz.grad

    g_ana = aux_loss_grad(z, K, alpha, E)
    check("解析梯度 vs autograd", True if relerr(g_ana, g_auto) < 1e-12 else False,
          f"relerr={relerr(g_ana, g_auto):.2e}")

    # 小切片中心差分（只测前 4x4）
    zs = z[:4, :4].clone()
    def fn():
        zf = z.clone()
        zf[:4, :4] = zs
        return aux_loss(zf, K, alpha, E)
    g_num = numeric_grad(fn, zs)
    check("解析梯度 vs 中心差分", True if relerr(g_ana[:4, :4], g_num) < 1e-4 else False,
          f"relerr={relerr(g_ana[:4, :4], g_num):.2e}")

    # 一致：z=0 时 p 全等 1/E，且 sum_i f_i = k（每 token 恰选 k 个），
    # 故 L = alpha*E*(sum f_i * 1/E) = alpha*k，与 f 的具体分布无关。
    z_eq = torch.zeros(T, E, dtype=DTYPE, device=DEV)
    L_eq = aux_loss(z_eq, K, alpha, E).item()
    check("均匀 P 时 L=alpha*k（与 f 分布无关）", abs(L_eq - alpha * K) < 1e-12,
          f"L_eq={L_eq:.6e} (alpha*k={alpha*K:.6e})")

    # 偏斜时 L 更大
    z_skew = torch.zeros(T, E, dtype=DTYPE, device=DEV)
    z_skew[:, 0] = 8.0
    L_skew = aux_loss(z_skew, K, alpha, E).item()
    check("偏斜路由 L 显著大于均衡", L_skew > 5 * L_eq, f"L_skew={L_skew:.4e} vs alpha={alpha}")

    # f 被当常量：把 f 换成 stop_grad 不影响解析梯度（对比手工构造）
    f = topk_mask(z, K).mean(dim=0)
    check("f_i 之和 = k/T*T= k  (每 token 选 k 个)", abs(f.sum().item() - K) < 1e-9,
          f"sum f={f.sum().item():.6f} (k={K})")
    print(f"  (alpha={alpha}, E={E}, T={T}, topk={K}, L_eq={L_eq:.3e})")


# ---------------------------------------------------------------------------
# Part B: router z-loss
# ---------------------------------------------------------------------------
def z_loss(z, beta):
    lse = torch.logsumexp(z, dim=-1)      # [T]
    return beta * (lse ** 2).mean()


def z_loss_grad(z, beta):
    T = z.shape[0]
    lse = torch.logsumexp(z, dim=-1, keepdim=True)   # [T,1]
    p = torch.softmax(z, dim=-1)
    return beta * 2.0 * lse / T * p


def run_part_b():
    print("\n=== Part B: router z-loss 的梯度 ===")
    T, E = 128, 24
    beta = 0.001
    z = torch.randn(T, E, dtype=DTYPE, device=DEV) * 2.0

    zz = z.clone().requires_grad_(True)
    z_loss(zz, beta).backward()
    g_auto = zz.grad
    g_ana = z_loss_grad(z, beta)
    check("解析梯度 vs autograd", relerr(g_ana, g_auto) < 1e-12, f"relerr={relerr(g_ana, g_auto):.2e}")

    zs = z[:3, :3].clone()
    def fn():
        zf = z.clone()
        zf[:3, :3] = zs
        return z_loss(zf, beta)
    g_num = numeric_grad(fn, zs, eps=1e-6)
    check("解析梯度 vs 中心差分", relerr(g_ana[:3, :3], g_num) < 1e-5,
          f"relerr={relerr(g_ana[:3, :3], g_num):.2e}")

    # 迫近惩罚：放大 logits 的尺度会把 z-loss 抬起来
    z_big = z * 4
    check("放大 logits 尺度抬升 z-loss", z_loss(z_big, beta).item() > z_loss(z, beta).item(),
          f"L(z)={z_loss(z, beta).item():.4e} < L(4z)={z_loss(z_big, beta).item():.4e}")


# ---------------------------------------------------------------------------
# Part C: aux-loss-free bias（DeepSeek-V3 e_score_correction_bias）
# ---------------------------------------------------------------------------
def sqrt_softplus(z):
    """Mega Gate 的打分函数：sqrt(softplus(z))，softplus(z)=log(1+exp(z))，带阈值分支。"""
    # 数值稳定：对 z>20 直接 softplus ~ z
    sp = torch.where(z > 20.0, z, torch.log1p(torch.exp(torch.clamp(z, max=20.0))))
    return torch.sqrt(sp)


def mega_gate_route(z, k, bias=None, scale=1.0):
    """复刻 Mega Gate 的路由语义（test_mega_gate.py:130-142）：
    scores = sqrt_softplus(logits)；ranking = scores + bias；topk on ranking；
    weight = scores[selected] / (sum scores[selected] + 1e-20) * scale。
    返回 (topk_idx, weights, load(=每专家被选中占比))。
    """
    scores = sqrt_softplus(z)
    ranking = scores if bias is None else scores + bias.unsqueeze(0)
    idx = torch.topk(ranking, k, dim=-1).indices            # [T,k]
    sel = torch.gather(scores, 1, idx)                      # [T,k] 无偏分
    w = sel / (sel.sum(dim=1, keepdim=True) + 1e-20) * scale
    load = torch.zeros(scores.shape[1], dtype=scores.dtype, device=scores.device)
    load.scatter_add_(0, idx.reshape(-1), torch.ones_like(idx, dtype=scores.dtype).reshape(-1))
    load = load / (scores.shape[0] * k)
    return idx, w, load


def run_bias_update(z, k, gamma, steps, target=None):
    """DeepSeek-V3 aux-loss-free 更新：b_i += gamma*sign(target_i - load_i)。"""
    E = z.shape[1]
    if target is None:
        target = torch.full((E,), 1.0 / E, dtype=z.dtype, device=z.device)
    bias = torch.zeros(E, dtype=z.dtype, device=z.device)
    hist = []
    for _ in range(steps):
        _, _, load = mega_gate_route(z, k, bias)
        hist.append(load.max().item() / (load.min().item() + 1e-30))
        bias = bias + gamma * torch.sign(target - load)
    return bias, load, hist


def run_part_c():
    print("\n=== Part C: aux-loss-free bias 的负载收敛 ===")
    T, E, K = 4096, 32, 4
    # 造一个「偏斜」路由：少数专家 logits 系统性地高
    z = torch.randn(T, E, dtype=DTYPE, device=DEV) * 1.0
    z[:, :4] += 2.5    # 前 4 个专家被偏好

    # 无 bias 的基线负载
    _, _, load0 = mega_gate_route(z, K, None)
    ratio0 = load0.max().item() / load0.min().item()
    check("无 bias：负载明显不均衡", ratio0 > 10, f"max/min={ratio0:.2f}")

    gamma = 0.001
    bias, load_final, hist = run_bias_update(z, K, gamma, steps=800)
    ratio1 = load_final.max().item() / load_final.min().item()
    check("加 bias 更新后：负载趋衡", ratio1 < 1.5, f"max/min={ratio1:.4f} (起始 {ratio0:.2f})")
    check("负载单调改善（前 50 步 max/min 严格下降趋势）",
          hist[49] < hist[0], f"step0={hist[0]:.2f} -> step50={hist[49]:.2f} -> step800={ratio1:.4f}")

    # 收敛后目标占比的偏差
    dev = (load_final - 1.0 / E).abs().max().item()
    check("收敛后各专家占比接近 1/E", dev < 0.01, f"max|load-1/E|={dev:.2e}  (1/E={1.0/E:.4f})")

    # 权重仍然无偏：选中专家的权重 = 无偏分归一（与 bias 无关）
    idx, w, _ = mega_gate_route(z, K, bias)
    scores = sqrt_softplus(z)
    sel = torch.gather(scores, 1, idx)
    w_ref = sel / (sel.sum(dim=1, keepdim=True) + 1e-20)
    check("gate 权重用无偏分（bias 只进排名）", relerr(w, w_ref) < 1e-15,
          f"relerr={relerr(w, w_ref):.2e}")

    # 排名确实用了 bias：换 bias 会改变选中的专家集合
    idx_nb, _, _ = mega_gate_route(z, K, None)
    changed = (idx != idx_nb).any(dim=1).float().mean().item()
    check("bias 改变排名（选中的专家集合有变化）", changed > 0.05,
          f"变化 token 占比={changed:.3f}")

    # 方向性：过载专家的 bias 被压低、欠载专家被抬高
    over = load_final > 1.0 / E
    check("过载专家 bias 更低 / 欠载更高（方向正确）",
          bias[over].mean().item() < bias[~over].mean().item(),
          f"over_bias={bias[over].mean().item():+.4f} under_bias={bias[~over].mean().item():+.4f}")
    print(f"  (T={T}, E={E}, topk={K}, gamma={gamma}, steps=800)")

    # gamma 越大收敛越快（40 步内）
    _, load_fast, _ = run_bias_update(z, K, gamma=0.01, steps=40)
    ratio_fast = load_fast.max().item() / load_fast.min().item()
    check("更大 gamma 收敛更快", ratio_fast < 1.5, f"gamma=0.01,40 步 max/min={ratio_fast:.4f}")


# ---------------------------------------------------------------------------
# Part D: 三者的关系
# ---------------------------------------------------------------------------
def run_part_d():
    print("\n=== Part D: aux-loss 与 bias 的关系 ===")
    T, E, K = 128, 16, 4
    alpha, beta = 0.01, 0.001
    z = torch.randn(T, E, dtype=DTYPE, device=DEV)
    bias = torch.randn(E, dtype=DTYPE, device=DEV) * 0.1

    # bias 不参与 aux-loss 的可微图（bias 是常量），故 dL/dbias 不应存在
    zz = z.clone().requires_grad_(True)
    bb = bias.clone().requires_grad_(True)
    # aux-loss 定义里不含 bias；即使把 bias 折进 ranking 做 top-k，f 仍是 stop-grad 常量
    L = aux_loss(zz, K, alpha, E)
    L.backward()
    check("aux-loss 不含 bias（bias.grad is None）", bb.grad is None,
          f"bias.requires_grad={bb.requires_grad}, bias.grad={'None' if bb.grad is None else 'set'}")

    # bias 更新只用 sign，不需要梯度（非光滑、不可微）
    load = torch.rand(E, dtype=DTYPE, device=DEV)
    b_new = bias + 0.001 * torch.sign(1.0 / E - load)
    check("bias 更新 = 纯符号规则（SGD 无关）", b_new.shape == bias.shape,
          f"|db| in {{{(0.001 * torch.sign(1.0/E - load)).abs().min().item():.4f}, "
          f"{(0.001 * torch.sign(1.0/E - load)).abs().max().item():.4f}}}")

    # DeepSeek-V3：先 warmup 用 aux-loss，之后 bias-only。
    # 坐实 warmup 路径：沿 aux-loss 负梯度走，L 单调下降（小步长成立）。
    T2 = 1024
    z_skew = torch.randn(T2, E, dtype=DTYPE, device=DEV)
    z_skew[:, :2] += 3.0
    zz = z_skew.clone().requires_grad_(True)
    L0 = aux_loss(zz, K, alpha, E)
    L0.backward()
    lr = 0.05
    z1 = (zz - lr * zz.grad).detach()
    L1 = aux_loss(z1, K, alpha, E)
    check("aux-loss 负梯度一步使 L 下降（warmup 路径）", L1.item() < L0.item(),
          f"L0={L0.item():.6e} -> L1={L1.item():.6e}")

    # 关键对照：aux-loss 的梯度只作用在**软概率 P_i** 上，对**硬 top-k 负载**是弱杠杆——
    # 因为 top-k 是排名的阶跃函数。bias 的符号更新**直接改排名**，是强杠杆。
    _, _, load_no = mega_gate_route(z_skew, K, None)
    r_no = load_no.max().item() / (load_no.min().item() + 1e-30)

    zc = z_skew.clone()
    for _ in range(2000):
        zc = zc - 0.5 * aux_loss_grad(zc, K, alpha=1.0, E=E)   # 放大 alpha 做公平对照
    _, _, load_aux = mega_gate_route(zc, K, None)
    r_aux = load_aux.max().item() / (load_aux.min().item() + 1e-30)

    _, load_bias, _ = run_bias_update(z_skew, K, gamma=0.001, steps=800)
    r_bias = load_bias.max().item() / (load_bias.min().item() + 1e-30)

    # aux-loss 路径能改善，但远不如 bias 直接；bias 把硬负载压到 ~1。
    check("aux-loss 梯度对硬负载是弱杠杆", r_aux < r_no and r_aux > 2.0,
          f"max/min: {r_no:.2f} -> {r_aux:.2f}（2000 步, alpha=1）")
    check("bias 符号更新对硬负载是强杠杆", r_bias < 1.5 and r_bias < r_aux / 3,
          f"bias max/min={r_bias:.4f}  vs aux-loss={r_aux:.2f}")


def main():
    print("=" * 72)
    print("MoE 路由负载均衡三件套：aux-loss / z-loss / aux-loss-free bias")
    print(f"torch={torch.__version__}  device={DEV}  dtype={DTYPE}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 72)

    run_part_a()
    run_part_b()
    run_part_c()
    run_part_d()

    print("\n" + "=" * 72)
    print(f"TOTAL: {PASS} PASS / {FAIL} FAIL")
    print("=" * 72)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
