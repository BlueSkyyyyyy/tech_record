#!/usr/bin/env python3
"""pass@k 与 pass^k：同一个 75%，在 k=10 时讲出相反的故事。

对应专题第 7 篇《评测：SWE-bench、Terminal-Bench 与 Harness 混淆》。

用法：
    python3 passk.py
    python3 passk.py --p 0.75 --trials 10
    python3 passk.py --test

只依赖标准库。两个指标回答的是不同问题：

    pass@k  = k 次里至少成功 1 次的概率        → "多给几次机会能不能做出来"
    pass^k  = k 次里每一次都成功的概率          → "每次都能不能做对"

前者随 k 上升，后者随 k 下降。对客服、运维、自动改代码这类"每次都要对"的
agent，应该看 pass^k；对生成候选、搜索这类"一次成功就够"的任务，才看 pass@k。

脚本还复现了 2026 年一篇论文里最简单的评测仪器错误：一部分 episode 因为
没有触发 grader 而被记成失败。它不会让模型变差，但会让排名变化。
"""

from __future__ import annotations

import argparse
import math
import random


def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏估计：1 - C(n-c, k) / C(n, k)。n=总试验数，c=成功数。"""
    if k > n:
        raise ValueError("k must be <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(p: float, k: int) -> float:
    """假设每次独立、成功率 p，则 k 次全对的概率是 p^k。"""
    return p**k


def wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = success / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def table(p: float, trials: int) -> None:
    print(f"per-trial p={p:.0%}, trials={trials}")
    print(f"{'k':>3}{'pass@k':>10}{'pass^k':>10}")
    for k in range(1, trials + 1):
        print(f"{k:>3}{1 - (1 - p) ** k:>10.1%}{pass_hat_k(p, k):>10.1%}")
    print("\n注意 k=1 时两者相同；k=3 时 pass^3 = 0.75^3 ≈ 42%；k=10 时 pass@10≈100%，pass^10≈6%。")


def grader_bug_demo(seed: int = 0) -> None:
    """一半 episode 没触发 grader，被记成失败：pooled 分数被拉低。"""
    rng = random.Random(seed)
    n_tasks = 100
    true_p = 0.75
    graded = []
    for _ in range(n_tasks):
        success = rng.random() < true_p
        triggered = rng.random() < 0.5  # 只有一半会走到 grading
        graded.append(success if triggered else False)
    observed = sum(graded) / n_tasks
    lo, hi = wilson_ci(sum(graded), n_tasks)
    print(f"\ngrader bug: true={true_p:.0%}  observed={observed:.0%}  95% CI=[{lo:.0%},{hi:.0%}]")
    print("修复方式不是重跑模型，而是把未评分的 episode 按 benchmark 自己的规则补评。")


def test() -> None:
    # pass@k 的边界
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 0, 1) == 0.0
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-12
    assert pass_at_k(10, 5, 10) == 1.0
    # 小样本无偏估计：n=5,c=1,k=2 → 1 - C(4,2)/C(5,2) = 0.4
    assert abs(pass_at_k(5, 1, 2) - 0.4) < 1e-12
    # pass^k
    assert abs(pass_hat_k(0.75, 3) - 0.421875) < 1e-12
    assert pass_hat_k(0.75, 10) < 0.1
    lo, hi = wilson_ci(75, 100)
    assert lo < 0.75 < hi
    assert 0 <= lo <= hi <= 1
    print("passk: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.75)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        test()
        return
    table(args.p, args.trials)
    grader_bug_demo()


if __name__ == "__main__":
    main()
