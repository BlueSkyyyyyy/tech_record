#!/usr/bin/env python3
"""在 2D 八高斯上训一个最小 DDPM：1000 步去噪到底在做什么（PyTorch CPU）.

对应文章：扩散模型精读（一）DDPM。

数据是 8 个高斯组成的环。模型是一个 3 层 MLP，输入 (x_t, t)，输出 epsilon。
训练目标就是 DDPM 的 L_simple：

    L = E_{t,x0,eps} || eps - eps_theta(sqrt(ab_t) x0 + sqrt(1-ab_t) eps, t) ||^2

训练完做三件事：

1. 用 1000 步 DDPM 采样，看 8 个峰是否都覆盖到；
2. 用 100 步、20 步采样（直接把 t 均匀跳采），看质量怎么掉；
3. 算「epsilon 误差 -> x0 误差」的放大系数 1/sqrt(SNR)：
   t=900 时约 60 倍，所以高噪声段预测 x0 再 clip 是个坏主意。

运行：

    python ddpm_toy.py            # 训练 + 采样，约 10-20 秒
    python ddpm_toy.py --test     # 少训一点，只做结构断言
"""
from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
torch.manual_seed(0)

T = 1000
B1, B2 = 1e-4, 0.02
N_MODES = 8
RADIUS = 2.0
MODE_STD = 0.08


def schedule() -> tuple[torch.Tensor, torch.Tensor]:
    beta = torch.linspace(B1, B2, T)
    alpha = 1.0 - beta
    ab = torch.cumprod(alpha, dim=0)
    return beta, ab


BETA, AB = schedule()


def sample_data(n: int) -> torch.Tensor:
    """8 个高斯，均值在半径 2 的圆上，标准差 0.08。"""
    idx = torch.randint(0, N_MODES, (n,))
    theta = 2 * math.pi * idx / N_MODES
    mean = torch.stack([RADIUS * torch.cos(theta), RADIUS * torch.sin(theta)], dim=1)
    return mean + MODE_STD * torch.randn(n, 2)


class TimeEmb(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0,1]；频率取几何级数，和 Transformer 的正弦位置编码同构
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None, :] * 2 * math.pi
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class EpsNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.temb = TimeEmb(64)
        self.net = nn.Sequential(
            nn.Linear(2 + 64, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.temb(t)], dim=1))


def train(steps: int = 3000, batch: int = 256, lr: float = 2e-3) -> tuple[EpsNet, list]:
    model = EpsNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for i in range(steps):
        x0 = sample_data(batch)
        t_idx = torch.randint(0, T, (batch,))
        ab = AB[t_idx][:, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        t = (t_idx.float() + 1) / T
        pred = model(xt, t)
        loss = ((pred - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (i + 1) % 500 == 0:
            print(f"  step {i+1:5d}  loss {sum(losses[-200:])/200:.4f}")
    return model, losses


@torch.no_grad()
def sample(model: EpsNet, n: int = 2000, steps: int = T) -> torch.Tensor:
    """DDPM 采样；steps<T 时按等间隔取子序列（最朴素的 respacing）。"""
    x = torch.randn(n, 2)
    seq = torch.linspace(0, T - 1, steps).long()
    for k in range(len(seq) - 1, -1, -1):
        i = int(seq[k])
        t = torch.full((n,), (i + 1) / T)
        eps = model(x, t)
        ab, ab_prev = AB[i], (AB[seq[k - 1]] if k > 0 else torch.tensor(1.0))
        beta = BETA[i]
        alpha = 1 - beta
        mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
        if k > 0:
            # 用真实后验方差 beta_tilde，而不是 beta（两者在 t 小时差别不大）
            var = (1 - ab_prev) / (1 - ab) * beta
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x


def mode_stats(samples: torch.Tensor) -> tuple[int, float, list]:
    """把样本分到最近的峰，返回 (覆盖峰数, 平均半径, 每峰计数)。"""
    counts = [0] * N_MODES
    for s in samples:
        theta = math.atan2(float(s[1]), float(s[0])) % (2 * math.pi)
        counts[int(round(theta / (2 * math.pi) * N_MODES)) % N_MODES] += 1
    radii = samples.norm(dim=1)
    covered = sum(1 for c in counts if c > len(samples) * 0.01)
    return covered, float(radii.mean()), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    steps = 600 if args.test else args.steps

    print(f"训练 {steps} 步（8 高斯，2D）...")
    model, losses = train(steps=steps)
    assert losses[-1] < losses[0], "loss 应该下降"

    print("\n== epsilon 误差 -> x0 误差的放大系数 1/sqrt(SNR) ==")
    for t in [100, 300, 500, 700, 900, 1000]:
        ab = float(AB[t - 1])
        print(f"  t={t:4d}  sqrt(ab)={math.sqrt(ab):.4f}  "
              f"x0 误差放大={math.sqrt((1-ab)/ab):10.2f}x")

    print("\n== 采样步数 vs 8 峰覆盖 ==")
    results = {}
    for k in [1000, 100, 20]:
        s = sample(model, n=2000, steps=k)
        cov, r, counts = mode_stats(s)
        results[k] = (cov, r)
        print(f"  {k:4d} 步: 覆盖 {cov}/8 峰, 平均半径 {r:.3f}, 每峰计数 {counts}")
        if k == 1000:
            print(f"        样本示例: {s[:4].tolist()}")

    if args.test:
        assert results[1000][0] >= 6, "1000 步应覆盖大部分峰"
        assert results[1000][1] > 1.5, "样本半径应接近数据半径 2"
        # 20 步仍然应该落在环附近（不会飞掉），但覆盖可能变差
        assert results[20][1] > 1.0, "20 步采样不应发散"
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
