#!/usr/bin/env python3
"""RTC 的引导权重：新旧 chunk 在重叠区怎么“接骨”（numpy）.

对应文章：VLA 精读（八）推理系统 /（九）LeRobot 代码走读。

LeRobot 的 `RTCProcessor.get_prefix_weights`（`policies/rtc/modeling_rtc.py:255`）
构造一条权重曲线 w[i]：

- i < inference_delay：w=1，强制新 chunk 的前缀跟旧 chunk 已执行的部分一致；
- inference_delay <= i < execution_horizon：从 1 平滑降到 0；
- i >= execution_horizon：w=0，完全听新 chunk 的。

四种 schedule：ZEROS / ONES / LINEAR / EXP。EXP 是 LeRobot 的默认（论文里也是）。

这个脚本不做真正的去噪引导（那需要模型和梯度），而是把“引导”退化成
对两个 chunk 做加权插值，单独看**拼接处的跳变**有多大。真正的 RTC 在每一步
去噪时把误差的梯度加回速度场，效果类似但会随去噪步数累积。

运行：

    python rtc_toy.py           # 权重曲线 + 拼接跳变
    python rtc_toy.py --test    # 自测
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def prefix_weights(start: int, end: int, total: int, schedule: str = "exp") -> np.ndarray:
    """复刻 LeRobot 的 get_prefix_weights。"""
    start = min(start, end)
    if schedule == "zeros":
        w = np.zeros(total)
        w[:start] = 1.0
        return w
    if schedule == "ones":
        w = np.ones(total)
        w[end:] = 0.0
        return w

    # linear ramp over [start, end)
    n = max(total - max(total - end, 0) - start, 0)
    if end <= start or n <= 0:
        lin = np.array([])
    else:
        lin = np.linspace(1, 0, n + 2)[1:-1]
    if schedule == "exp":
        lin = lin * np.expm1(lin) / (math.e - 1)
    w = np.zeros(total)
    w[:start] = 1.0
    w[start : start + len(lin)] = lin
    w[end:] = 0.0
    return w


def chunks(total: int = 20, delay: int = 4, shift: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """旧 chunk 是斜率 0.02 的直线；新 chunk 整体上移 shift（换了计划）。"""
    i = np.arange(total)
    prev = 0.02 * i
    new = 0.02 * i + shift
    return prev, new


def splice(prev: np.ndarray, new: np.ndarray, w: np.ndarray) -> np.ndarray:
    """在重叠区按 w 把新 chunk 拉向旧 chunk。"""
    return w * prev + (1 - w) * new


def jumps(x: np.ndarray, start: int, prev: np.ndarray) -> tuple[float, float]:
    """返回 (拼接处跳变, 相邻步最大跳变)。

    拼接处比较的是"上一拍真正执行过的动作"prev[start-1] 和"新 chunk 的第一个执行点"
    x[start]，而不是新 chunk 内部的相邻两拍。
    """
    at_switch = abs(x[start] - prev[start - 1])
    step = float(np.abs(np.diff(x)).max())
    return float(at_switch), step


def report() -> None:
    total, delay, end = 20, 4, 10
    prev, new = chunks(total, delay)
    print("== 权重曲线（start=4, end=10, total=20）==")
    print(f"{'schedule':<8}" + "".join(f"{i:>6}" for i in range(12)))
    for sched in ("zeros", "ones", "linear", "exp"):
        w = prefix_weights(delay, end, total, sched)
        print(f"{sched:<8}" + "".join(f"{w[i]:>6.2f}" for i in range(12)))

    print("\n== 拼接处的跳变（动作单位）==")
    print(f"{'方式':<12}{'拼接跳变':>10}{'chunk内最大步跳变':>18}{'前10步偏离新计划':>18}")
    raw_jump, raw_step = jumps(new, delay, prev)
    print(f"{'不引导':<12}{raw_jump:>10.4f}{raw_step:>12.4f}{0.0:>18.4f}")
    for sched in ("zeros", "ones", "linear", "exp"):
        w = prefix_weights(delay, end, total, sched)
        x = splice(prev, new, w)
        j, st = jumps(x, delay, prev)
        dev = float(np.abs(x[:end] - new[:end]).mean())
        print(f"{sched:<12}{j:>10.4f}{st:>12.4f}{dev:>18.4f}")

    print("\n== 读法 ==")
    print("- zeros：只固定已执行段（i<4），切换点直接吃下全部差异 → 跳变最大。")
    print("- ones：整段 [0,10) 都跟旧计划，把 0.3 的差异推迟到 horizon 末尾。")
    print("- linear：把差异摊到 6 步里，最大步跳变最小（0.063）。")
    print("- exp：凸函数，中段比 linear 更低，更早放开旧 chunk、更快转向新计划，")
    print("  代价是最大步跳变比 linear 大（0.117）。")
    print("- 真实 RTC 还会把这个误差的梯度乘上 guidance_weight 加回速度场，")
    print("  且 guidance_weight 随时间步变化（接近 t=0 时会被 max_guidance_weight 截断）。")


def _self_test() -> None:
    w = prefix_weights(4, 10, 20, "linear")
    assert np.all(w[:4] == 1.0)
    assert np.all(w[10:] == 0.0)
    assert np.all(np.diff(w[4:10]) <= 0), w
    # 首个 ramp 点是 6/7，不是 1：linspace(1,0,n+2)[1:-1] 的端点被切掉了
    assert abs(w[4] - 6 / 7) < 1e-9 and w[9] < 0.2

    we = prefix_weights(4, 10, 20, "exp")
    assert np.all(we[:4] == 1.0) and np.all(we[10:] == 0.0)
    # exp 是凸的，中段比 linear 更低（更早放开旧 chunk）
    assert we[6] < w[6], (we[6], w[6])

    prev, new = chunks()
    raw, _ = jumps(new, 4, prev)
    blended = splice(prev, new, prefix_weights(4, 10, 20, "exp"))
    got, _ = jumps(blended, 4, prev)
    assert got < raw * 0.5, (got, raw)
    print("[test] rtc_toy OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        report()
