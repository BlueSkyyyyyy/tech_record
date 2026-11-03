#!/usr/bin/env python3
"""噪声日程：线性 vs cosine，信号与噪声到底谁先没（纯标准库 + NumPy）.

对应文章：扩散模型精读（一）DDPM。

DDPM 的线性日程 beta: 1e-4 -> 0.02, T=1000。它有一个常被忽略的后果：
信号系数 sqrt(alpha_bar) 在最后 100 步里几乎掉到 0，而噪声系数早就饱和到 1。
换句话说，**大部分去噪步在和一个几乎纯噪声的输入搏斗**，真正决定细节的
只有末尾一小段。这解释了为什么 1000 步可以砍到 50 步、20 步而质量不怎么掉。

本脚本算三张表：

1. 线性日程在 t=0,100,...,1000 的 alpha_bar、SNR(dB)、sqrt(alpha_bar)；
2. cosine 日程的同一张表（Nichol & Dhariwal 2021）；
3. 两个日程的「有效步数」：SNR 跨过 0 dB 的步、alpha_bar < 1e-3 的步。

运行：

    python schedules.py          # 打印表格
    python schedules.py --test   # 自测（结构断言）
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def linear_beta(T: int = 1000, b1: float = 1e-4, b2: float = 0.02) -> np.ndarray:
    """DDPM 的线性日程（Ho et al. 2020, Sec. 4）。"""
    return np.linspace(b1, b2, T, dtype=np.float64)


def cosine_beta(T: int = 1000, s: float = 0.008) -> np.ndarray:
    """cosine 日程（Nichol & Dhariwal 2021, Eq. 17）。

    alpha_bar_t = f(t)/f(0), f(t)=cos(((t/T)+s)/(1+s) * pi/2)^2。
    先算 alpha_bar，再反推 beta，保证两端行为可控。
    """
    steps = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos(((steps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
    ab = f / f[0]
    beta = np.clip(1.0 - ab[1:] / ab[:-1], 1e-8, 0.999)
    return beta


def alpha_bar_from_beta(beta: np.ndarray) -> np.ndarray:
    """alpha_bar_t = prod_{s<=t}(1-beta_s)，返回长度 T（下标 0 对应 t=1）。"""
    alpha = 1.0 - beta
    return np.cumprod(alpha)


def snr_db(ab: float) -> float:
    """信噪比 10*log10(alpha_bar/(1-alpha_bar))；ab=0/1 时返回 -inf/+inf。"""
    if ab <= 0.0:
        return float("-inf")
    if ab >= 1.0:
        return float("inf")
    return 10.0 * math.log10(ab / (1.0 - ab))


def summarize(beta: np.ndarray, name: str, T: int = 1000) -> dict:
    ab = alpha_bar_from_beta(beta)
    snrs = np.array([snr_db(x) for x in ab])
    # 有效步：SNR >= 0 dB（信号不弱于噪声）的步数；以及 alpha_bar < 1e-3 的步数
    keep = int(np.sum(snrs >= 0.0))
    dead = int(np.sum(ab < 1e-3))
    print(f"\n== {name} ==")
    print(f"{'t':>5} {'alpha_bar':>12} {'sqrt(ab)':>10} {'sqrt(1-ab)':>11} {'SNR(dB)':>10}")
    for t in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000]:
        if t == 0:
            print(f"{0:>5} {1.0:>12.6f} {1.0:>10.6f} {0.0:>11.6f} {'inf':>10}")
            continue
        i = t - 1
        print(f"{t:>5} {ab[i]:>12.6e} {math.sqrt(ab[i]):>10.6f} "
              f"{math.sqrt(max(0.0, 1.0 - ab[i])):>11.6f} {snrs[i]:>10.2f}")
    print(f"SNR >= 0 dB 的步数: {keep}/{T}  |  alpha_bar < 1e-3 的步数: {dead}/{T}")
    print(f"alpha_bar_T = {ab[-1]:.3e}")
    return dict(ab=ab, snr=snrs, keep=keep, dead=dead)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="只跑结构断言")
    args = parser.parse_args()

    lin = summarize(linear_beta(), "linear beta 1e-4 -> 0.02, T=1000")
    cos = summarize(cosine_beta(), "cosine (Nichol & Dhariwal 2021), T=1000")

    # 一张对照表：同样 T=1000，末尾还剩多少信号
    print("\n== 末尾信号对照 ==")
    print(f"{'t':>5} {'linear ab':>12} {'cosine ab':>12}")
    for t in [500, 800, 900, 950, 1000]:
        print(f"{t:>5} {lin['ab'][t-1]:>12.3e} {cos['ab'][t-1]:>12.3e}")

    # 线性日程：alpha_bar 从 1 到 ~0 的「半衰期」
    half = int(np.argmax(lin["ab"] < 0.5)) + 1
    print(f"\n线性日程 alpha_bar 首次 < 0.5 的步: t={half}")
    print(f"线性日程 SNR 首次 < 0 dB 的步: t={int(np.argmax(lin['snr'] < 0)) + 1}")

    if args.test:
        assert lin["ab"].shape == (1000,)
        assert np.all(np.diff(lin["ab"]) < 0), "alpha_bar 必须严格递减"
        assert np.all(np.diff(cos["ab"]) < 0)
        assert lin["ab"][-1] < 1e-3, "DDPM 线性日程末尾应几乎无信号"
        assert lin["keep"] < 700, "线性日程有效去噪步应远小于 1000"
        assert cos["ab"][-1] < 1e-4, "cosine 日程末尾也应接近 0"
        # cosine 的 SNR 在中间段更平缓：0 dB 附近步数更多
        assert cos["keep"] > lin["keep"], "cosine 应保留更多有效步"
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
