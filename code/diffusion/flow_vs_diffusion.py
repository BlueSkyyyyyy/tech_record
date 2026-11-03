#!/usr/bin/env python3
"""Flow Matching vs 扩散：路径有多直，步数就有多省（NumPy）.

对应文章：扩散模型精读（七）Flow Matching 与 Rectified Flow。

同一个 8 高斯目标，两条路径：

- **扩散（VP/DDIM）**：x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps；
- **Flow Matching**：x_t = (1-t) x0 + t x1，x1~N(0,I)，目标速度 v=x1-x0。

两者的边缘速度/score 都有解析式（高斯混合的后验加权），所以可以只比较
「路径几何 + 离散化误差」，不掺模型误差。

输出：

1. K=1/2/4/10/20 步的 sliced Wasserstein；
2. 轨迹的「弯曲度」= 路径长度 / 首尾直线距离（K=20）；
3. 一步 Euler 的落点（会落在 0 附近，说明为什么需要蒸馏）。

运行：

    python flow_vs_diffusion.py          # 约 2 秒
    python flow_vs_diffusion.py --test   # 自测
"""
from __future__ import annotations

import argparse
import math

import numpy as np

rng = np.random.default_rng(0)

N_MODES = 8
RADIUS = 2.0
MODE_STD = 0.08
T = 1000
B1, B2 = 1e-4, 0.02
MU = np.stack([
    [RADIUS * math.cos(2 * math.pi * i / N_MODES),
     RADIUS * math.sin(2 * math.pi * i / N_MODES)]
    for i in range(N_MODES)
])
W = np.full(N_MODES, 1.0 / N_MODES)


def ab_c(t: float) -> float:
    return math.exp(-(T * B1 * t + 0.5 * T * (B2 - B1) * t * t))


def sample_data(n: int) -> np.ndarray:
    idx = rng.integers(0, N_MODES, n)
    return MU[idx] + MODE_STD * rng.standard_normal((n, 2))


def responsibilities(x: np.ndarray, means: np.ndarray, var: float) -> np.ndarray:
    diff = x[:, None, :] - means[None, :, :]
    logits = -0.5 * (diff**2).sum(-1) / var + np.log(W)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def fm_velocity(x: np.ndarray, t: float) -> np.ndarray:
    """线性插值路径的边缘速度场（高斯混合解析式）。"""
    m = (1 - t) * MU                      # (8,2)
    var = (1 - t) ** 2 * MODE_STD**2 + t**2
    w = responsibilities(x, m, var)       # (n,8)
    diff = x[:, None, :] - m[None, :, :]  # (n,8,2)
    # v_i = [(t - (1-t)sigma^2) * diff / var] - mu
    vi = ((t - (1 - t) * MODE_STD**2) / var) * diff - MU[None, :, :]
    return np.einsum("nk,nkd->nd", w, vi)


def fm_euler(x: np.ndarray, steps: int) -> np.ndarray:
    # 约定 t=0 是数据、t=1 是噪声，所以从 1 积到 0（dt 为负）
    ts = np.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        dt = ts[k + 1] - ts[k]
        x = x + dt * fm_velocity(x, ts[k])
    return x


def score_diffusion(x: np.ndarray, ab: float) -> np.ndarray:
    s2 = 1.0 - ab + ab * MODE_STD**2
    m = math.sqrt(ab) * MU
    w = responsibilities(x, m, s2)
    mean = w @ m
    return (mean - x) / s2


def ddim(x: np.ndarray, steps: int) -> np.ndarray:
    ts = np.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        t, tn = ts[k], ts[k + 1]
        ab, abn = ab_c(t), ab_c(tn)
        score = score_diffusion(x, ab)
        eps = -math.sqrt(max(1e-12, 1 - ab)) * score
        x0 = (x - math.sqrt(max(1e-12, 1 - ab)) * eps) / math.sqrt(ab)
        eps_hat = (x - math.sqrt(ab) * x0) / math.sqrt(max(1e-12, 1 - ab))
        x = math.sqrt(abn) * x0 + math.sqrt(max(0.0, 1 - abn)) * eps_hat
    return x


def sliced_w2(a: np.ndarray, b: np.ndarray, n_proj: int = 32) -> float:
    r = np.random.default_rng(1)
    dirs = r.standard_normal((n_proj, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    total = 0.0
    for d in dirs:
        pa, pb = np.sort(a @ d), np.sort(b @ d)
        q = (np.arange(len(pa)) + 0.5) / len(pa)
        total += np.mean((pa - np.quantile(pb, q)) ** 2)
    return float(math.sqrt(total / n_proj))


def path_metrics(sampler, steps: int = 20) -> tuple:
    """返回 (路径长度/直线距离, 终点)。轨迹在给定时序上重算。"""
    x = rng.standard_normal((256, 2))
    x0 = x.copy()
    length = 0.0
    if sampler is fm_euler:
        ts = np.linspace(1.0, 0.0, steps + 1)
        for k in range(steps):
            dt = ts[k + 1] - ts[k]
            v = fm_velocity(x, ts[k])
            x = x + dt * v
            length += np.linalg.norm(dt * v, axis=1).sum()
    else:
        ts = np.linspace(1.0, 0.0, steps + 1)
        for k in range(steps):
            t, tn = ts[k], ts[k + 1]
            ab, abn = ab_c(t), ab_c(tn)
            score = score_diffusion(x, ab)
            eps = -math.sqrt(max(1e-12, 1 - ab)) * score
            xn = (x - math.sqrt(max(1e-12, 1 - ab)) * eps) / math.sqrt(ab)
            eps_hat = (x - math.sqrt(ab) * xn) / math.sqrt(max(1e-12, 1 - ab))
            xn = math.sqrt(abn) * xn + math.sqrt(max(0.0, 1 - abn)) * eps_hat
            length += np.linalg.norm(xn - x, axis=1).sum()
            x = xn
    straight = np.linalg.norm(x - x0, axis=1).sum()
    return length / straight, x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    n = 4096
    real = sample_data(8192)
    x_init = rng.standard_normal((n, 2))

    print(f"{'步骤 K':>7} {'FM Euler':>12} {'DDIM':>12}")
    results = {}
    for k in [1, 2, 4, 10, 20]:
        fm = sliced_w2(fm_euler(x_init.copy(), k), real)
        dd = sliced_w2(ddim(x_init.copy(), k), real)
        results[k] = (fm, dd)
        print(f"{k:>7} {fm:>12.4f} {dd:>12.4f}")

    r_fm, end_fm = path_metrics(fm_euler, 20)
    r_dd, end_dd = path_metrics(ddim, 20)
    print(f"\n路径长度 / 首尾直线距离（K=20）: FM={r_fm:.3f}, DDIM={r_dd:.3f}")
    print(f"一步 Euler 的终点均值: {fm_euler(x_init.copy(), 1).mean(0).round(3)}")

    if args.test:
        assert results[1][0] > results[20][0] * 5, "一步 FM 应显著差于 20 步"
        assert results[10][0] < 0.2, "10 步 FM 应接近真实分布"
        assert r_fm < 1.6 and r_dd < 3.0, "轨迹弯曲度应在合理范围"
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
