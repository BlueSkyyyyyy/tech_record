#!/usr/bin/env python3
"""采样器对比：DDIM / Euler / Heun 在同一个解析 score 场上（NumPy）.

对应文章：扩散模型精读（二）分数与 SDE、（三）快速采样。

这里不训练网络，直接用 8 高斯混合的**解析 score**，把「模型误差」和
「离散化误差」拆开：剩下的差距纯粹来自用多少步、用几阶求解器。

目标分布：8 个高斯，均值在半径 2 的圆上，std=0.08。
前向：VP 过程 x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps，beta 线性 1e-4..0.02，T=1000。

三个求解器：

1. **DDIM (eta=0)**：代数重参数化，一步从 x_t 跳到 x_{t'}，不做泰勒展开；
2. **Euler**：probability-flow ODE 的一阶积分；
3. **Heun**：二阶预测-校正，每步 2 次 score。

指标：与 8192 个真实样本的 sliced Wasserstein（32 个随机方向），
以及均值/标准差/8 峰覆盖。

运行：

    python sampler_zoo.py          # 约 2 秒
    python sampler_zoo.py --test   # 自测
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


def beta_c(t: float) -> float:
    """连续化后的 beta(t)，t in [0,1]；乘 T 是为了和离散 1000 步对齐。"""
    return T * (B1 + (B2 - B1) * t)


def ab_c(t: float) -> float:
    """alpha_bar(t) = exp(-int_0^t beta(s) ds)，与 1000 步离散值一致。"""
    return math.exp(-(T * B1 * t + 0.5 * T * (B2 - B1) * t * t))


def sample_data(n: int) -> np.ndarray:
    idx = rng.integers(0, N_MODES, n)
    return MU[idx] + MODE_STD * rng.standard_normal((n, 2))


def score(x: np.ndarray, ab: float) -> np.ndarray:
    """混合高斯的解析 score：sum_i w_i N(x; sqrt(ab) mu_i, s^2 I) 的梯度。"""
    s2 = 1.0 - ab + ab * MODE_STD**2
    m = math.sqrt(ab) * MU  # (8,2)
    diff = x[:, None, :] - m[None, :, :]  # (n,8,2)
    logits = -0.5 * (diff**2).sum(-1) / s2
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    mean = np.einsum("nk,kd->nd", w, m)
    return (mean - x) / s2


def eps_from_score(x: np.ndarray, ab: float) -> np.ndarray:
    # VP: score = -eps / sqrt(1-ab)
    return -math.sqrt(max(1e-12, 1.0 - ab)) * score(x, ab)


def x0_from_eps(x: np.ndarray, eps: np.ndarray, ab: float) -> np.ndarray:
    return (x - math.sqrt(max(1e-12, 1.0 - ab)) * eps) / math.sqrt(ab)


def ddim(x: np.ndarray, steps: int) -> np.ndarray:
    ts = np.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        t, tn = ts[k], ts[k + 1]
        ab, abn = ab_c(t), ab_c(tn)
        eps = eps_from_score(x, ab)
        x0 = x0_from_eps(x, eps, ab)
        eps_hat = (x - math.sqrt(ab) * x0) / math.sqrt(max(1e-12, 1 - ab))
        x = math.sqrt(abn) * x0 + math.sqrt(max(0.0, 1 - abn)) * eps_hat
    return x


def pf_rhs(x: np.ndarray, t: float) -> np.ndarray:
    # probability-flow ODE: dx/dt = -0.5 beta(t) (x + score)
    return -0.5 * beta_c(t) * (x + score(x, ab_c(t)))


def euler(x: np.ndarray, steps: int) -> np.ndarray:
    ts = np.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        dt = ts[k + 1] - ts[k]
        x = x + dt * pf_rhs(x, ts[k])
    return x


def heun(x: np.ndarray, steps: int) -> np.ndarray:
    ts = np.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        dt = ts[k + 1] - ts[k]
        k1 = pf_rhs(x, ts[k])
        k2 = pf_rhs(x + dt * k1, ts[k + 1])
        x = x + 0.5 * dt * (k1 + k2)
    return x


def sliced_w2(a: np.ndarray, b: np.ndarray, n_proj: int = 32) -> float:
    """sliced Wasserstein-2 的平方根近似（越小越好）。"""
    r = np.random.default_rng(1)
    dirs = r.standard_normal((n_proj, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    total = 0.0
    for d in dirs:
        pa = np.sort(a @ d)
        pb = np.sort(b @ d)
        # 两组样本数不同：把 b 插值到 a 的分位点上，不能直接截断（会取到下半段）
        q = (np.arange(len(pa)) + 0.5) / len(pa)
        total += np.mean((pa - np.quantile(pb, q)) ** 2)
    return float(math.sqrt(total / n_proj))


def mode_counts(x: np.ndarray) -> list:
    idx = np.argmin(((x[:, None, :] - MU[None, :, :]) ** 2).sum(-1), axis=1)
    return np.bincount(idx, minlength=N_MODES).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    n = 4096
    real = sample_data(8192)
    x_init = rng.standard_normal((n, 2))

    print(f"真实分布: mean={real.mean(0).round(3)}, std={real.std(0).round(3)}")
    print(f"\n{'solver':>6} {'steps':>6} {'slicedW2':>10} {'mean_r':>8} {'std':>7} {'covered':>8}")
    results = {}
    for name, fn in [("DDIM", ddim), ("Euler", euler), ("Heun", heun)]:
        for steps in [5, 10, 20, 50, 100]:
            x = fn(x_init.copy(), steps)
            sw = sliced_w2(x, real)
            r = float(np.linalg.norm(x, axis=1).mean())
            cov = sum(1 for c in mode_counts(x) if c > n * 0.01)
            results[(name, steps)] = (sw, r, cov)
            print(f"{name:>6} {steps:>6} {sw:>10.4f} {r:>8.3f} "
                  f"{x.std():>7.3f} {cov:>8}/8")

    print("\n参考：直接把初始噪声当样本（不采样）的 slicedW2 = "
          f"{sliced_w2(x_init, real):.4f}")

    if args.test:
        # 步数越多不应更差（允许 5% 抖动）
        for name in ["DDIM", "Euler", "Heun"]:
            vals = [results[(name, s)][0] for s in [5, 10, 20, 50, 100]]
            assert vals[-1] <= vals[0] * 1.05, f"{name} 100 步不应差于 5 步"
        assert results[("DDIM", 20)][0] < 0.5, "DDIM 20 步应接近真实分布"
        assert results[("Euler", 5)][0] > results[("DDIM", 20)][0], "5 步 Euler 应明显更差"
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
