#!/usr/bin/env python3
"""FID 的有限样本偏差：为什么 5 万张是门槛（NumPy）.

对应文章：扩散模型精读（十二）评测与 2026 地图。

FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})

它比较的是两个高斯拟合，而不是样本本身。即使生成分布和真实分布**完全相同**，
用有限样本估出来的 FID 也大于 0。偏差的主项不是 d/n，而是 d^2/n（协方差估计
的噪声），更准确地说按**有效秩** r 走：bias ~ r^2/n。
各向同性 d=64 时实测系数约 0.54；Inception 特征高度相关，r 远小于 2048，
所以 5 万张的「同分布 FID」不会炸到几十，但换数据集/换特征就不可比。

本脚本：

1. d=64 的高斯，两个独立样本，n=100/500/2000/10000，重复 20 次，看偏差；
2. 把理论偏差 d(1/n1+1/n2) 和实测对照；
3. 固定 n=5000，给 fake 加不同强度的噪声，看 FID 单调性；
4. 外推到 d=2048（Inception 特征维度）。

运行：

    python fid_bias.py          # 约 3 秒
    python fid_bias.py --test   # 自测
"""
from __future__ import annotations

import argparse

import numpy as np

rng = np.random.default_rng(0)


def frechet_distance(mu1, s1, mu2, s2) -> float:
    """Fr\'echet distance between N(mu1,s1) and N(mu2,s2)。"""
    diff = mu1 - mu2
    # Tr((s1^{1/2} s2 s1^{1/2})^{1/2})，用对称 PSD 的特征值算
    w, v = np.linalg.eigh(s1)
    w = np.clip(w, 0, None)
    s1_half = (v * np.sqrt(w)) @ v.T
    m = s1_half @ s2 @ s1_half
    w2 = np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)
    tr_sqrt = float(np.sqrt(w2).sum())
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)


def fit(x: np.ndarray) -> tuple:
    return x.mean(0), np.cov(x, rowvar=False)


def same_dist_fid(d: int, n: int, trials: int = 20) -> float:
    vals = []
    for _ in range(trials):
        a = rng.standard_normal((n, d))
        b = rng.standard_normal((n, d))
        vals.append(frechet_distance(*fit(a), *fit(b)))
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    d = 64
    print(f"== 同分布的两个样本，d={d}，FID 的期望不是 0 ==")
    print(f"{'n':>8} {'实测 FID':>12} {'0.54*d^2/n':>12} {'实测/理论':>10}")
    res = {}
    coefs = []
    for n in [100, 500, 2000, 10000]:
        fid = same_dist_fid(d, n)
        theory = 0.54 * d**2 / n
        res[n] = fid
        coefs.append(fid / (d**2 / n))
        print(f"{n:>8} {fid:>12.3f} {theory:>12.3f} {fid/theory:>10.2f}")
    coef = float(np.mean(coefs))
    print(f"\n实测系数 c = FID / (d^2/n) = {coef:.3f}（各向同性高斯，d={d}）")
    print("外推 n=50000：")
    for r in [64, 256, 2048]:
        print(f"  有效秩 r={r:>4}: 同分布 FID ≈ {coef * r**2 / 50000:.3f}")

    print("\n== 固定 n=5000，fake 加噪后的 FID ==")
    n = 5000
    real = rng.standard_normal((n, d))
    mr, sr = fit(real)
    print(f"{'噪声强度':>10} {'FID':>10}")
    for sigma in [0.0, 0.05, 0.1, 0.3, 0.6]:
        fake = rng.standard_normal((n, d)) + sigma * rng.standard_normal((n, d))
        print(f"{sigma:>10.2f} {frechet_distance(mr, sr, *fit(fake)):>10.3f}")

    if args.test:
        assert res[100] > res[2000] > 0, "样本越少，同分布 FID 越大"
        assert res[100] > 1.0, "n=100 时偏差应明显大于 0"
        assert 0.3 < coef < 1.0, "实测偏差系数应接近 0.54（各向同性）"
        # 加噪必须让 FID 单调变差
        vals = []
        for sigma in [0.0, 0.3, 0.6]:
            fake = rng.standard_normal((n, d)) + sigma * rng.standard_normal((n, d))
            vals.append(frechet_distance(mr, sr, *fit(fake)))
        assert vals[0] < vals[1] < vals[2]
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
