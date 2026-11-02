#!/usr/bin/env python3
"""动作离散化：256 个 bin 到底损失了什么（CPU only, numpy）.

对应文章：VLA 精读（一）动作表示。

这个脚本只回答三个问题：

1. OpenVLA 的 256 个 bin 真的等于 256 个档位吗？
   —— 不是。`np.linspace(-1, 1, 256)` 给出 256 条边界、255 个中心，
      有效档位是 255；而 π0.5 把 state 塞进 prompt 时用的是
      `np.linspace(-1, 1, 257)[:-1]`，那是 256 个档位。两处不一致。
2. 单个动作的量化误差有多大？对一段 50 步的 chunk，误差怎么累积？
3. 用 DCT 压缩整段 chunk，能换到多少 token 节省？
   —— 这是 FAST 的思路的“玩具版”：只做 DCT + 均匀量化，
      不做 BPE，所以这里报的是“需要多少个系数”，不是 FAST 的 token 数。

运行：

    python action_binning.py            # 打印扫参表
    python action_binning.py --test     # 自测（断言）
"""
from __future__ import annotations

import argparse

import numpy as np


# ---------------------------------------------------------------------------
# 1. 均匀分箱（复刻 OpenVLA / RT-2 的 ActionTokenizer）
# ---------------------------------------------------------------------------
def uniform_bins(n_edges: int = 256, lo: float = -1.0, hi: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """返回 (边界, 中心)。

    注意 OpenVLA 的命名：`n_bins=256` 传给 `np.linspace` 的是**边界数**，
    所以 center 只有 255 个。
    """
    edges = np.linspace(lo, hi, n_edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def encode(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """OpenVLA 的做法：np.digitize 得到 0..len(edges)，再夹到中心索引范围。"""
    idx = np.digitize(np.clip(x, edges[0], edges[-1]), edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def decode(idx: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return centers[idx]


# ---------------------------------------------------------------------------
# 2. DCT 压缩（FAST 的玩具版：DCT-II + 均匀量化）
# ---------------------------------------------------------------------------
def dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 矩阵，C @ x 即 DCT。"""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c = c * np.sqrt(2.0 / n)
    c[0] /= np.sqrt(2.0)
    return c


def dct_quantize(chunk: np.ndarray, keep_ratio: float = 1.0, n_levels: int = 256) -> np.ndarray:
    """对 [T, D] 的 chunk 逐维做 DCT，量化后反变换。

    keep_ratio：保留低频系数的比例（1.0 = 全保留）。
    量化步长按每个维度系数的 max-abs 自适应——真实 FAST 是对归一化后的
    数据做固定 scale，这里为了看清“能量集中”这件事做了简化。
    """
    t, d = chunk.shape
    c = dct_matrix(t)
    coeff = c @ chunk  # [T, D]
    k = max(1, int(round(t * keep_ratio)))
    out = np.zeros_like(coeff)
    out[:k] = coeff[:k]
    # 均匀量化：每个维度单独定 scale
    scale = np.abs(out).max(axis=0, keepdims=True) + 1e-12
    step = 2.0 * scale / (n_levels - 1)
    q = np.round(out / step) * step
    return c.T @ q


def trajectory(n_steps: int = 50, n_dim: int = 7, seed: int = 0) -> np.ndarray:
    """一段平滑的关节轨迹，范围落在 [-1, 1]。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_steps)[:, None]
    amp = rng.uniform(0.2, 0.8, size=(1, n_dim))
    freq = rng.uniform(0.5, 1.5, size=(1, n_dim))
    phase = rng.uniform(0, 2 * np.pi, size=(1, n_dim))
    return amp * np.sin(2 * np.pi * freq * t + phase)


def report() -> None:
    edges, centers = uniform_bins()
    print("== 1. 分箱口径 ==")
    print(f"边界数={len(edges)} 中心数={len(centers)} 步长={edges[1]-edges[0]:.6f}")
    print(f"理论最大误差={0.5*(edges[1]-edges[0]):.6f} (半格)")

    x = np.linspace(-1, 1, 10001)
    err = np.abs(decode(encode(x, edges), centers) - x)
    print(f"实测最大误差={err.max():.6f}  RMS={np.sqrt((err**2).mean()):.6f}")

    print("\n== 2. 小动作幅度的相对误差 ==")
    print(f"{'动作范围':>10} {'绝对误差(max)':>14} {'相对范围':>10}")
    for r in (1.0, 0.5, 0.1, 0.02):
        xx = np.linspace(-r, r, 20001)
        e = np.abs(decode(encode(xx, edges), centers) - xx).max()
        print(f"±{r:<9.2f} {e:>14.6f} {e / (2 * r):>9.2%}")

    print("\n== 3. chunk 的 DCT 压缩（50 步 × 7 维 = 350 个数）==")
    chunk = trajectory()
    print(f"{'每维保留':>8} {'总系数':>8} {'RMSE':>10} {'最大误差':>10} {'压缩比':>8}")
    for ratio in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
        rec = dct_quantize(chunk, keep_ratio=ratio)
        rmse = np.sqrt(((rec - chunk) ** 2).mean())
        mx = np.abs(rec - chunk).max()
        k = max(1, int(round(chunk.shape[0] * ratio)))
        total = k * chunk.shape[1]
        print(f"{k:>8} {total:>8} {rmse:>10.5f} {mx:>10.5f} {350/total:>7.2f}x")

    print("\n== 4. 误差累积：delta 动作 vs 绝对动作 ==")
    # 假设每步都有独立同分布的量化误差，delta 动作会积分
    step_err = 1.0 / 255.0
    print(f"单步量化误差(上界) = {step_err:.6f}")
    for n in (10, 25, 50, 100):
        worst = n * step_err
        rms = np.sqrt(n) * step_err / np.sqrt(3)
        print(f"步数={n:>3}  最坏累积={worst:.4f}  RMS 累积={rms:.4f}")


def _self_test() -> None:
    edges, centers = uniform_bins()
    assert len(edges) == 256 and len(centers) == 255, "OpenVLA 的 256 bins 实际是 255 个中心"

    # 端点：-1 落在第 0 格，+1 被 clip 到最后一格
    assert encode(np.array([-1.0]), edges)[0] == 0
    assert encode(np.array([1.0]), edges)[0] == len(centers) - 1
    # 量化误差不超过半格
    x = np.linspace(-1, 1, 50001)
    err = np.abs(decode(encode(x, edges), centers) - x)
    assert err.max() <= 0.5 * (edges[1] - edges[0]) + 1e-12

    # DCT 全保留时只剩系数量化误差：约为动作范围的 1%
    chunk = trajectory(n_steps=32, n_dim=3, seed=1)
    rec = dct_quantize(chunk, keep_ratio=1.0)
    rel_rmse = np.sqrt(((rec - chunk) ** 2).mean()) / np.abs(chunk).max()
    assert rel_rmse < 0.02, rel_rmse
    # 正交性
    c = dct_matrix(16)
    assert np.allclose(c @ c.T, np.eye(16), atol=1e-10)

    # π0.5 的 state 分箱是 256 个档位（边界 257 去掉最后一个）
    state_edges = np.linspace(-1, 1, 257)[:-1]
    state_idx = np.digitize(np.array([-1.0, 1.0]), state_edges) - 1
    assert state_idx.tolist() == [0, 255], state_idx
    print("[test] action_binning OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        report()
