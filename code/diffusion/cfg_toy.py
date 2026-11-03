#!/usr/bin/env python3
"""Classifier-free guidance 的锐化与坍缩（NumPy，解析 score）.

对应文章：扩散模型精读（四）引导。

场景：一维条件生成。类别 c=1 的条件分布是两个峰的混合：

    0.6 * N(1, 0.15^2) + 0.4 * N(2, 0.15^2)

无条件分布是 4 个峰（c=0 的 -2/-1，c=1 的 1/2），权重 0.25/0.25/0.10/0.40。
把 x≈1 设成「条件上常见、无条件上罕见」，CFG 就会把它放大、把 x≈2 压掉。
用 CFG 的 score 形式采样：

    score_s = score_u + s * (score_c - score_u)
            = s * score_c + (1-s) * score_u

s=1 是纯条件采样；s>1 放大条件项。观察三件事：

1. 少数峰（x≈2）的比例随 s 怎么掉——这就是「多样性换保真度」；
2. 均值是否越过 1.5（过饱和的开始）；
3. 标准差怎么缩——guidance 不只是选峰，它把每个峰也压窄了。

注意记号：CFG 论文里的 w 是**增量**，score 形式是 (1+w)score_c - w score_u；
diffusers 的 `guidance_scale` 是 s=1+w。s=7.5 对应论文的 w=6.5。

运行：

    python cfg_toy.py          # 约 2 秒
    python cfg_toy.py --test   # 自测
"""
from __future__ import annotations

import argparse
import math

import numpy as np

rng = np.random.default_rng(0)

T = 1000
B1, B2 = 1e-4, 0.02
SIGMA = 0.15
MU = np.array([-2.0, -1.0, 1.0, 2.0])
# 无条件：4 峰等权；条件 c=1：后两峰 0.7/0.3
W_UNCOND = np.array([0.25, 0.25, 0.10, 0.40])
W_COND = np.array([0.0, 0.0, 0.60, 0.40])


def ab_c(t: float) -> float:
    return math.exp(-(T * B1 * t + 0.5 * T * (B2 - B1) * t * t))


def mix_score(x: np.ndarray, ab: float, w: np.ndarray) -> np.ndarray:
    """一维混合高斯的 score，x shape (n,)。"""
    s2 = 1.0 - ab + ab * SIGMA**2
    m = math.sqrt(ab) * MU
    diff = x[:, None] - m[None, :]
    logits = -0.5 * diff**2 / s2 + np.log(np.maximum(w, 1e-300))[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    post = np.exp(logits)
    post /= post.sum(axis=1, keepdims=True)
    mean = post @ m
    return (mean - x) / s2


def guided_score(x: np.ndarray, ab: float, s: float) -> np.ndarray:
    su = mix_score(x, ab, W_UNCOND)
    sc = mix_score(x, ab, W_COND)
    return su + s * (sc - su)


def sample(s: float, n: int = 20000, steps: int = 50) -> np.ndarray:
    x = rng.standard_normal(n)
    ts = np.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        t, tn = ts[k], ts[k + 1]
        ab, abn = ab_c(t), ab_c(tn)
        score = guided_score(x, ab, s)
        eps = -math.sqrt(max(1e-12, 1 - ab)) * score
        x0 = (x - math.sqrt(max(1e-12, 1 - ab)) * eps) / math.sqrt(ab)
        eps_hat = (x - math.sqrt(ab) * x0) / math.sqrt(max(1e-12, 1 - ab))
        x = math.sqrt(abn) * x0 + math.sqrt(max(0.0, 1 - abn)) * eps_hat
    return x


def stats(x: np.ndarray) -> tuple:
    minority = float(np.mean(x > 1.5))       # x≈2 的峰
    wrong = float(np.mean(x < 0.0))          # 跑到 c=0 的区域
    return minority, float(x.mean()), float(x.std()), wrong


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print(f"{'guidance_scale s':>16} {'(论文 w)':>10} {'少数峰%':>9} {'mean':>8} "
          f"{'std':>8} {'错类%':>7}")
    results = {}
    for s in [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]:
        x = sample(s)
        minority, mean, std, wrong = stats(x)
        results[s] = (minority, mean, std, wrong)
        print(f"{s:>16.1f} {s-1:>10.1f} {minority*100:>8.1f}% {mean:>8.3f} "
              f"{std:>8.3f} {wrong*100:>6.1f}%")

    print("\n说明：条件分布真实比例是 60/40（少数峰 40%），真实 mean=1.40、std≈0.49。")
    print("s=1 是纯条件采样；s 越大，少数峰越少、std 越小、mean 越靠近 1。")

    if args.test:
        assert results[1.0][0] > 0.3, "s=1 时少数峰应接近 40%"
        assert results[7.5][0] < results[1.0][0] / 3, "s=7.5 时少数峰应大幅减少"
        assert results[15.0][2] < results[1.0][2] * 0.8, "强引导应压窄分布"
        assert results[15.0][3] < 0.01, "不应跑到错误类别"
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
