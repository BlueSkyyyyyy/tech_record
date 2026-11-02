#!/usr/bin/env python3
"""VLA 评测的统计：10 次试验能说明什么（stdlib）.

对应文章：VLA 精读（十一）评测与失败模式。

VLA 论文里最常见的一句话是“成功率 80%”。但 8/10 和 80/100 是两回事：
前者的 95% 置信区间大约是 [49%, 94%]，后者是 [71%, 87%]。
这个脚本给三张表：

1. 不同试验次数下，成功率估计的 Wilson 95% 置信区间；
2. 要区分两个策略，每组需要多少次试验（80% power）；
3. 多任务平均时，任务间方差如何淹没试验次数带来的精度。

运行：

    python eval_ci.py            # 三张表
    python eval_ci.py --test     # 自测
"""
from __future__ import annotations

import argparse
import math

Z95 = 1.959963984540054
Z80 = 0.8416212335729143


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """二项比例的 Wilson score interval。"""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def n_per_group(p1: float, p2: float, alpha_z: float = Z95, power_z: float = Z80) -> int:
    """两比例检验的每组样本量（正态近似）。"""
    pbar = (p1 + p2) / 2
    delta = abs(p1 - p2)
    if delta == 0:
        return 10**9
    n = 2 * (alpha_z + power_z) ** 2 * pbar * (1 - pbar) / (delta**2)
    return math.ceil(n)


def report() -> None:
    print("== 1. 成功率的 95% Wilson 区间 ==")
    print(f"{'k/n':>8}{'点估计':>8}{'下限':>8}{'上限':>8}{'宽度(pp)':>10}")
    for k, n in ((8, 10), (16, 20), (40, 50), (80, 100), (400, 500), (0, 10), (10, 10)):
        lo, hi = wilson(k, n)
        print(f"{k}/{n:<6}{k/n:>8.1%}{lo:>8.1%}{hi:>8.1%}{(hi-lo)*100:>10.1f}")

    print("\n== 2. 区分两个策略需要多少试验（每组，80% power）==")
    print(f"{'p1':>6}{'p2':>6}{'差值(pp)':>10}{'每组 n':>9}")
    for p1, p2 in ((0.6, 0.8), (0.7, 0.8), (0.5, 0.7), (0.8, 0.9)):
        print(f"{p1:>6.0%}{p2:>6.0%}{(p2-p1)*100:>10.0f}{n_per_group(p1, p2):>9}")

    print("\n== 3. 两个具体对比：8/10 vs 6/10、80/100 vs 60/100 ==")
    for a, b, n in (((8, 10), (6, 10), 10), ((80, 100), (60, 100), 100)):
        la, ha = wilson(*a)
        lb, hb = wilson(*b)
        overlap = min(ha, hb) > max(la, lb)
        print(f"{a[0]}/{n} = {a[0]/n:.0%} [{la:.0%}, {ha:.0%}]  vs  "
              f"{b[0]}/{n} = {b[0]/n:.0%} [{lb:.0%}, {hb:.0%}]  "
              f"→ {'区间重叠，不能区分' if overlap else '区间不重叠，可以区分'}")

    print("\n== 4. 多任务平均：任务间方差才是大头 ==")
    # 10 个任务，每个任务真实成功率从 0.2 到 0.95 不等；每个任务跑 n 次
    true_rates = [0.20, 0.35, 0.45, 0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
    mean_true = sum(true_rates) / len(true_rates)
    var_task = sum((r - mean_true) ** 2 for r in true_rates) / (len(true_rates) - 1)
    print(f"真实任务均值 = {mean_true:.1%}，任务间 std = {math.sqrt(var_task):.1%}")
    print(f"{'每任务 n':>9}{'总试验':>8}{'平均值的标准误':>16}{'95% 半宽':>10}")
    for n in (10, 20, 50, 100):
        # 总均值 = 任务均值 + 抽样噪声；方差 = var_task/10 + mean(p(1-p))/ (10*n)
        var_samp = sum(r * (1 - r) for r in true_rates) / len(true_rates) / n
        se = math.sqrt(var_task / len(true_rates) + var_samp / len(true_rates))
        print(f"{n:>9}{n*10:>8}{se:>16.1%}{1.96*se:>10.1%}")
    print("→ 把每任务试验数从 10 加到 100，半宽只从约 17pp 降到约 15pp；")
    print("  想真正区分策略，先增加任务数/场景多样性，而不是死磕单任务试验次数。")


def _self_test() -> None:
    # 经典 Wilson 例子：8/10 -> [0.490, 0.943]
    lo, hi = wilson(8, 10)
    assert 0.48 < lo < 0.50 and 0.93 < hi < 0.95, (lo, hi)
    # 0/10 的下限是 0，上限约 0.28
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.31, (lo, hi)
    # 10/10 对称
    lo, hi = wilson(10, 10)
    assert 0.69 < lo < 0.75 and hi > 0.999, (lo, hi)
    # 区间宽度随 n 单调下降
    widths = [wilson(int(0.8 * n), n)[1] - wilson(int(0.8 * n), n)[0] for n in (10, 50, 500)]
    assert widths[0] > widths[1] > widths[2], widths
    # 60% vs 80% 大约每组 80 出头
    n = n_per_group(0.6, 0.8)
    assert 70 <= n <= 100, n
    print("[test] eval_ci OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        report()
