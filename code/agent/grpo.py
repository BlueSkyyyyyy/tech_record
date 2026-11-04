#!/usr/bin/env python3
"""GRPO vs GSPO：token 级比率和序列级比率，差在哪。

对应专题第 10 篇《Agentic RL：环境、reward 与 GSPO》。

用法：
    python3 grpo.py
    python3 grpo.py --eps 0.2
    python3 grpo.py --test

只依赖标准库。它不是训练器，而是把 PPO/GRPO/GSPO 的目标函数拆开：

    ratio_t   = exp(logp_new(t) - logp_old(t))          # token 级
    ratio_seq = exp(Σ_t logp_new(t) - Σ_t logp_old(t))  # 序列级（GSPO）
    A_i       = (r_i - mean(r)) / std(r)                # 组内相对优势
    L         = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

GRPO 在 token 上做 ratio 和 clip；GSPO 在整条序列上做。序列级的好处是：
一条 1000 token 的轨迹只有一个比率，不会被某个"高比率 token"主导，
MoE 训练也更稳。代价是粒度变粗，credit assignment 更依赖 advantage。
"""

from __future__ import annotations

import argparse
import math


def group_advantage(rewards: list[float]) -> list[float]:
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std == 0:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def token_ratios(old_logps: list[float], new_logps: list[float]) -> list[float]:
    return [math.exp(n - o) for o, n in zip(old_logps, new_logps)]


def sequence_ratio(old_logps: list[float], new_logps: list[float]) -> float:
    return math.exp(sum(new_logps) - sum(old_logps))


def clipped(ratio: float, adv: float, eps: float) -> float:
    return min(ratio * adv, max(1 - eps, min(1 + eps, ratio)) * adv)


def grpo_loss(old: list[float], new: list[float], adv: float, eps: float) -> float:
    rs = token_ratios(old, new)
    return -sum(clipped(r, adv, eps) for r in rs) / len(rs)


def gspo_loss(old: list[float], new: list[float], adv: float, eps: float) -> float:
    return -clipped(sequence_ratio(old, new), adv, eps)


def demo(eps: float) -> None:
    rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    adv = group_advantage(rewards)
    print("rewards:", rewards)
    print("advantages:", [round(a, 2) for a in adv])

    old = [-0.10, -0.20, -0.30, -0.40]
    new = [0.20, -0.26, -0.36, -0.46]
    rs = token_ratios(old, new)
    seq = sequence_ratio(old, new)
    print(f"\ntoken ratios: {[round(r, 3) for r in rs]}")
    print(f"sequence ratio: {seq:.3f}")
    a = adv[0]
    print(f"\nA=+{a:.2f} (成功样本)")
    print(f"  GRPO loss={grpo_loss(old, new, a, eps):.4f}  GSPO loss={gspo_loss(old, new, a, eps):.4f}")
    print("  单看 token 1 的比率是 %.3f，超过 1+ε=%.1f；整条序列的比率却只有 %.3f。" % (rs[0], 1 + eps, seq))
    print("  GRPO 会把这个 token 裁掉；GSPO 只看序列比率，仍落在 trust region 内。")


def test() -> None:
    adv = group_advantage([1, 0, 0, 0])
    assert abs(sum(adv)) < 1e-9
    assert adv[0] > 0 and adv[1] < 0
    old = [math.log(0.5), math.log(0.5)]
    new = [math.log(0.5), math.log(0.5)]
    assert token_ratios(old, new) == [1.0, 1.0]
    assert abs(sequence_ratio(old, new) - 1.0) < 1e-12
    # 序列比率是 token 比率的乘积
    old2 = [-0.2, -0.3]
    new2 = [-0.1, -0.2]
    assert abs(sequence_ratio(old2, new2) - token_ratios(old2, new2)[0] * token_ratios(old2, new2)[1]) < 1e-12
    # clip 的边界
    assert clipped(2.0, 1.0, 0.2) == 1.2
    assert clipped(0.1, -1.0, 0.2) == -0.8
    # 零方差组不产生梯度
    assert group_advantage([0.5, 0.5]) == [0.0, 0.0]
    print("grpo: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    test() if args.test else demo(args.eps)


if __name__ == "__main__":
    main()
