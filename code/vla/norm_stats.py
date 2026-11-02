#!/usr/bin/env python3
"""动作归一化：mean/std 还是 q01/q99？跨本体 padding 会把统计量带偏（numpy）.

对应文章：VLA 精读（七）数据与训练配方。

两件事：

1. OpenVLA / π0 用的是 **quantile 归一化**：把 [q01, q99] 映射到 [-1, 1]，
   超出部分在推理时被自然截断。mean/std 对离群点敏感——夹爪偶尔的大值
   会把 99% 的正常动作压进 [-0.5, 0.5]，等效分辨率掉一半。
2. 跨本体拼接时，7 维单臂要 pad 到 32 维。如果对 padded 的全 32 维算统计量，
   零填充会把均值/方差带偏；必须按 mask 只在有效维度上统计。

运行：

    python norm_stats.py           # 对比表
    python norm_stats.py --test    # 自测
"""
from __future__ import annotations

import argparse

import numpy as np


def make_actions(n: int = 20000, seed: int = 0) -> np.ndarray:
    """7 维动作：前 6 维是平滑关节角，第 7 维夹爪带离群点。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, size=(n, 7))
    # 夹爪：大多数在 [0,1]，0.5% 的记录出现 20 的尖峰（传感器/标定错误）
    # 比例刻意小于 1%，这样 q99 仍然落在正常范围内，而 mean/std 会被拉爆
    x[:, 6] = rng.uniform(0, 1, size=n)
    spike = rng.random(n) < 0.005
    x[spike, 6] = 20.0
    return x


def mean_std_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    return (x - mu) / sd, mu, sd


def quantile_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q01, q99 = np.quantile(x, 0.01, axis=0), np.quantile(x, 0.99, axis=0)
    return (x - q01) / (q99 - q01 + 1e-6) * 2 - 1, q01, q99


def coverage(z: np.ndarray) -> dict[str, float]:
    return {
        "in_1": float((np.abs(z) <= 1).mean()),
        "in_05": float((np.abs(z) <= 0.5).mean()),
        "std": float(z.std()),
        "max": float(np.abs(z).max()),
    }


def pad_report() -> None:
    """7 维单臂 pad 到 32 维，比较“全维统计”和“按 mask 统计”。"""
    rng = np.random.default_rng(1)
    arm = rng.normal(0.5, 0.2, size=(5000, 7))
    padded = np.zeros((5000, 32))
    padded[:, :7] = arm

    mu_all, sd_all = padded.mean(0), padded.std(0)
    mu_masked = np.array([arm[:, d].mean() for d in range(7)])
    sd_masked = np.array([arm[:, d].std() for d in range(7)])

    print("\n== padding 对统计量的影响（前 7 维）==")
    print(f"{'dim':>4}{'全维均值':>10}{'mask均值':>10}{'全维std':>10}{'mask std':>10}")
    for d in range(7):
        print(f"{d:>4}{mu_all[d]:>10.3f}{mu_masked[d]:>10.3f}{sd_all[d]:>10.3f}{sd_masked[d]:>10.3f}")
    print(f"第 7~31 维：全维统计 mean={mu_all[7:].mean():.2f} std={sd_all[7:].mean():.2f}，"
          f"这些维度根本没有数据。")


def report() -> None:
    x = make_actions()
    z_ms, mu, sd = mean_std_norm(x)
    z_q, q01, q99 = quantile_norm(x)

    print("== 1. 两种归一化的覆盖 ==")
    print(f"{'方法':<14}{'|z|<=1':>9}{'|z|<=0.5':>10}{'z.std':>9}{'|z|max':>9}")
    for name, z in (("mean/std", z_ms), ("q01/q99", z_q)):
        c = coverage(z)
        print(f"{name:<14}{c['in_1']:>8.1%}{c['in_05']:>10.1%}{c['std']:>9.3f}{c['max']:>9.2f}")

    print("\n== 2. 夹爪维度（含 0.5% 的 20.0 尖峰）==")
    print(f"mean/std: mean={mu[6]:.3f} std={sd[6]:.3f}")
    print(f"q01/q99 : q01={q01[6]:.3f} q99={q99[6]:.3f}")
    normal = x[:, 6] < 5
    z_ms_normal = np.abs(z_ms[normal, 6])
    z_q_normal = np.abs(z_q[normal, 6])
    print(f"正常样本被压到 |z|<=0.5 的比例：mean/std={np.mean(z_ms_normal <= 0.5):.1%}，"
          f"q01/q99={np.mean(z_q_normal <= 0.5):.1%}")
    print("→ mean/std 下，正常夹爪值几乎全挤在 0 附近，等效分辨率大幅下降。")

    pad_report()


def _self_test() -> None:
    x = make_actions(5000, seed=3)
    z_q, q01, q99 = quantile_norm(x)
    # 按定义，q01 映射到 -1、q99 映射到 +1
    assert np.allclose((q01 - q01) / (q99 - q01 + 1e-6) * 2 - 1, -1)
    assert np.allclose((q99 - q01) / (q99 - q01 + 1e-6) * 2 - 1, 1)
    # 98% 的数据落在 [-1,1] 内（1% 被截断在两端）
    inside = np.mean(np.abs(z_q) <= 1.0)
    assert 0.97 <= inside <= 0.99, inside

    z_ms, _, _ = mean_std_norm(x)
    # 均值归一化后，正常夹爪值更集中
    normal = x[:, 6] < 5
    assert np.mean(np.abs(z_ms[normal, 6]) <= 0.5) > np.mean(np.abs(z_q[normal, 6]) <= 0.5)

    # padding 不应改变前 7 维的按 mask 统计
    arm = x[:, :7]
    padded = np.zeros((len(arm), 32))
    padded[:, :7] = arm
    assert np.allclose(padded.mean(0)[:7], arm.mean(0))
    # 全维均值会被零填充拉低
    assert padded.mean(0)[7:].mean() == 0.0
    print("[test] norm_stats OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        report()
