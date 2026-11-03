#!/usr/bin/env python3
"""少步生成：一步回归、配对蒸馏、轨迹速度场（PyTorch CPU）.

对应文章：扩散模型精读（八）少步生成。

在 8 高斯上比较四种「学生」：

1. **独立配对的一步生成器**：训练数据 (noise, x0) 随机配对，MSE 最优解是
   E[x0|noise]=0，样本会塌成一个点——「拿 MSE 直接蒸一个生成器」的经典失败。
2. **轨迹配对的一步生成器**：用教师 50 步采出的 (noise, x0) 配对，学生学到的
   是教师 ODE 的确定性映射，能保留峰，但一步到位没有修正机会。
3. **速度场蒸馏 + 1/2/4/10 步 Euler**：学教师的边缘速度场，步数越少误差越大；
   1 步落在 E[x0|noise] 上，8 个峰会被平均掉一部分。
4. **教师 50 步**：上界。

指标：sliced Wasserstein、平均半径、8 峰覆盖。

运行：

    python distill_toy.py            # 约 10 秒
    python distill_toy.py --test     # 少训一点
"""
from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
torch.manual_seed(0)
rng = np.random.default_rng(0)

N_MODES = 8
RADIUS, MODE_STD = 2.0, 0.08
MU = np.stack([[RADIUS * math.cos(2 * math.pi * i / N_MODES),
                RADIUS * math.sin(2 * math.pi * i / N_MODES)] for i in range(N_MODES)])
W = np.full(N_MODES, 1.0 / N_MODES)


def sample_data(n: int) -> np.ndarray:
    idx = rng.integers(0, N_MODES, n)
    return MU[idx] + MODE_STD * rng.standard_normal((n, 2))


def teacher_velocity(x: np.ndarray, t) -> np.ndarray:
    """线性 FM 的边缘速度场；t 可以是标量或长度 n 的数组。t=0 数据，t=1 噪声。"""
    t = np.asarray(t, dtype=np.float64)
    if t.ndim == 0:
        t = np.full(len(x), float(t))
    m = (1 - t)[:, None, None] * MU[None, :, :]
    var = (1 - t) ** 2 * MODE_STD**2 + t**2
    diff = x[:, None, :] - m
    logits = -0.5 * (diff**2).sum(-1) / var[:, None] + np.log(W)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    coef = ((t - (1 - t) * MODE_STD**2) / var)[:, None, None]
    vi = coef * diff - MU[None, :, :]
    return np.einsum("nk,nkd->nd", w, vi)


def teacher_sample(noise: np.ndarray, steps: int = 50) -> np.ndarray:
    x = noise.copy()
    ts = np.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        dt = ts[k + 1] - ts[k]
        x = x + dt * teacher_velocity(x, float(ts[k]))
    return x


class MLP(nn.Module):
    def __init__(self, hidden: int = 128, time_emb: bool = False):
        super().__init__()
        self.time_emb = time_emb
        self.net = nn.Sequential(
            nn.Linear(2 + (32 if time_emb else 0), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if self.time_emb:
            freqs = torch.exp(-math.log(10000.0) * torch.arange(16) / 16)
            args = t[:, None] * freqs[None, :] * 2 * math.pi
            x = torch.cat([x, torch.sin(args), torch.cos(args)], dim=1)
        return self.net(x)


def train(model: MLP, xs: torch.Tensor, ys: torch.Tensor, ts: torch.Tensor | None,
          steps: int, lr: float = 2e-3) -> MLP:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(xs)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,))
        pred = model(xs[idx], ts[idx] if ts is not None else None)
        loss = ((pred - ys[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def sample_velocity(model: MLP, noise: torch.Tensor, steps: int) -> np.ndarray:
    x = noise.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for k in range(steps):
        dt = float(ts[k + 1] - ts[k])
        t = torch.full((len(x),), float(ts[k]))
        x = x + dt * model(x, t)
    return x.numpy()


def sliced_w2(a: np.ndarray, b: np.ndarray, n_proj: int = 32) -> float:
    r = np.random.default_rng(1)
    dirs = r.standard_normal((n_proj, 2)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    total = 0.0
    for d in dirs:
        pa, pb = np.sort(a @ d), np.sort(b @ d)
        q = (np.arange(len(pa)) + 0.5) / len(pa)
        total += np.mean((pa - np.quantile(pb, q)) ** 2)
    return float(math.sqrt(total / n_proj))


def mode_stats(x: np.ndarray) -> tuple:
    idx = np.argmin(((x[:, None, :] - MU[None, :, :]) ** 2).sum(-1), axis=1)
    counts = np.bincount(idx, minlength=N_MODES)
    covered = sum(1 for c in counts if c > len(x) * 0.01)
    return covered, float(np.linalg.norm(x, axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    steps = 600 if args.test else 1500

    n_pairs, n_eval = 20000, 4096
    noise = rng.standard_normal((n_pairs, 2)).astype(np.float32)
    data = sample_data(n_pairs).astype(np.float32)
    paired = teacher_sample(noise.astype(np.float64), 50).astype(np.float32)
    real = sample_data(8192)
    eval_noise = torch.randn(n_eval, 2)

    t50 = teacher_sample(eval_noise.numpy().astype(np.float64), 50)
    print(f"教师 50 步:      W2={sliced_w2(t50, real):.4f} "
          f"半径={mode_stats(t50)[1]:.3f} 覆盖={mode_stats(t50)[0]}/8")

    g_ind = train(MLP(), torch.from_numpy(noise), torch.from_numpy(data), None, steps)
    with torch.no_grad():
        out_ind = g_ind(eval_noise).numpy()
    print(f"独立配对 1 步:   W2={sliced_w2(out_ind, real):.4f} "
          f"半径={mode_stats(out_ind)[1]:.3f} 覆盖={mode_stats(out_ind)[0]}/8")

    g_pair = train(MLP(), torch.from_numpy(noise), torch.from_numpy(paired), None, steps)
    with torch.no_grad():
        out_pair = g_pair(eval_noise).numpy()
    print(f"轨迹配对 1 步:   W2={sliced_w2(out_pair, real):.4f} "
          f"半径={mode_stats(out_pair)[1]:.3f} 覆盖={mode_stats(out_pair)[0]}/8")

    t_train = rng.uniform(0, 1, n_pairs).astype(np.float32)
    xt = (1 - t_train[:, None]) * paired + t_train[:, None] * noise
    vt = teacher_velocity(xt.astype(np.float64), t_train).astype(np.float32)
    v_model = train(MLP(time_emb=True), torch.from_numpy(xt), torch.from_numpy(vt),
                    torch.from_numpy(t_train), steps)
    for k in [1, 2, 4, 10]:
        out = sample_velocity(v_model, eval_noise, k)
        print(f"速度蒸馏 {k:2d} 步:  W2={sliced_w2(out, real):.4f} "
              f"半径={mode_stats(out)[1]:.3f} 覆盖={mode_stats(out)[0]}/8")

    if args.test:
        assert mode_stats(out_ind)[1] < 0.5, "独立配对的一步生成器应塌到中心"
        assert mode_stats(out_pair)[1] > 1.0, "轨迹配对的一步生成器应保留峰的半径"
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
