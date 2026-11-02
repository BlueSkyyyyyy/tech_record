#!/usr/bin/env python3
"""Flow matching vs L2 回归：动作多峰时谁会“平均掉”（CPU, PyTorch）.

对应文章：VLA 精读（三）动作头。

场景刻意做得很小：一维动作 x，给定观测 obs，真实动作有两个可能的位置
（-1.5 或 +1.5）。当 obs 接近 0 时，后验是双峰的——机器人要么往左、要么往右。
这正是 VLA 里“同一个画面，人类演示有不同做法”的最小模型。

三个对照组：

1. **L2 回归**：直接 predict action = f(obs)，最优点是把两峰平均 → 输出 0，
   一个物理上不存在的动作。
2. **Flow matching**（π0 的约定）：学速度场 v(x_t, t)，从噪声积分到动作，
   每次采样会选一个峰。
3. **单步去噪**（t=1 直接回归 x1）：等价于在噪声上做 L2，仍然会平均。

π0 的约定（openpi `pi0.py:199`）：

    x_t = t * noise + (1 - t) * action
    u_t = noise - action
    loss = MSE(v_theta(x_t, t), u_t)

注意 t=1 是噪声、t=0 是动作，和论文里的写法相反（代码注释自己也承认了）。

运行：

    python flow_matching_toy.py          # 训练 + 评估，约 10 秒
    python flow_matching_toy.py --test   # 快速自测
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

# 小 MLP 在多核机器上会因线程调度变慢，限制线程数
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
torch.manual_seed(0)

LEFT, RIGHT = -1.5, 1.5


# ---------------------------------------------------------------------------
# 数据：obs 是 mode 的带噪观测；obs≈0 时后验双峰
# ---------------------------------------------------------------------------
def sample_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    mode = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    x = mode * 1.5 + 0.05 * torch.randn(n)
    y = 0.05 * torch.randn(n)
    obs = mode * 0.8 + 0.6 * torch.randn(n)
    return torch.stack([x, y], dim=-1), obs[:, None]


class VelocityNet(nn.Module):
    """v(x_t, t, obs) -> 2 维速度。"""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t, obs], dim=-1))


class RegressionNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def train_flow(steps: int = 2500, batch: int = 512) -> VelocityNet:
    model = VelocityNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(steps):
        action, obs = sample_data(batch)
        noise = torch.randn_like(action)
        # π0 的 Beta(1.5, 1)：偏向 t=1（噪声端）
        t = torch.distributions.Beta(1.5, 1.0).sample((batch, 1)) * 0.999 + 0.001
        x_t = t * noise + (1 - t) * action
        u_t = noise - action
        pred = model(x_t, t, obs)
        loss = ((pred - u_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  flow step {step:5d} loss {loss.item():.4f}")
    return model


def train_regression(steps: int = 2000, batch: int = 512) -> RegressionNet:
    model = RegressionNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(steps):
        action, obs = sample_data(batch)
        loss = ((model(obs) - action) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  reg  step {step:5d} loss {loss.item():.4f}")
    return model


@torch.no_grad()
def sample_flow(model: VelocityNet, obs: torch.Tensor, steps: int = 10) -> torch.Tensor:
    x = torch.randn(obs.shape[0], 2)
    dt = -1.0 / steps
    t = torch.ones(obs.shape[0], 1)
    for _ in range(steps):
        x = x + dt * model(x, t, obs)
        t = t + dt
    return x


def stats(x: torch.Tensor) -> dict[str, float]:
    ax = x[:, 0]
    near_left = (ax - LEFT).abs() < 0.4
    near_right = (ax - RIGHT).abs() < 0.4
    in_gap = ax.abs() < 0.5
    return {
        "left": near_left.float().mean().item(),
        "right": near_right.float().mean().item(),
        "gap": in_gap.float().mean().item(),
        "abs_mean": ax.abs().mean().item(),
    }


def hist(ax: torch.Tensor, bins: int = 15, lo: float = -2.2, hi: float = 2.2) -> str:
    edges = torch.linspace(lo, hi, bins + 1)
    counts = torch.histc(ax.clamp(lo, hi), bins=bins, min=lo, max=hi)
    counts = (counts / counts.max() * 20).round().int()
    lines = []
    for i in range(bins - 1, -1, -1):
        lines.append(f"{edges[i]:+5.2f} | {'#' * int(counts[i])}")
    return "\n".join(lines)


def demo() -> None:
    print("训练 flow matching ...")
    fm = train_flow()
    print("训练 L2 回归 ...")
    reg = train_regression()

    # 取一批 obs 落在歧义区（|obs|<0.2）的样本
    _, obs_all = sample_data(4000)
    obs = obs_all[obs_all.abs().squeeze() < 0.2][:400]
    print(f"\n歧义区样本数: {len(obs)}")

    fm_x = sample_flow(fm, obs)
    reg_x = reg(obs)
    # 只走一步 Euler（t=1 -> 0）：检验“多步积分”本身值多少钱
    noise = torch.randn(len(obs), 2)
    one_step = noise - fm(noise, torch.ones(len(obs), 1), obs)

    print("\n== 指标（歧义区，理想情况应一半左一半右、几乎不落在中间）==")
    print(f"{'方法':<14}{'左峰':>8}{'右峰':>8}{'中间':>8}{'|x|均值':>10}")
    for name, x in (("flow matching", fm_x), ("L2 回归", reg_x), ("flow 单步", one_step)):
        s = stats(x)
        print(f"{name:<14}{s['left']:>7.1%}{s['right']:>8.1%}{s['gap']:>8.1%}{s['abs_mean']:>10.3f}")

    print("\n== flow matching 采样的 x 分布 ==")
    print(hist(fm_x[:, 0]))
    print("\n== L2 回归输出的 x 分布 ==")
    print(hist(reg_x[:, 0]))


def _self_test() -> None:
    fm = train_flow(steps=800, batch=256)
    reg = train_regression(steps=600, batch=256)

    _, obs_all = sample_data(2000)
    obs = obs_all[obs_all.abs().squeeze() < 0.2][:300]
    fm_x = sample_flow(fm, obs)
    reg_x = reg(obs)

    s_fm, s_reg = stats(fm_x), stats(reg_x)
    # flow matching 应该两个峰都覆盖到
    assert s_fm["left"] > 0.2 and s_fm["right"] > 0.2, s_fm
    # 回归会塌到中间
    assert s_reg["gap"] > 0.5, s_reg
    # flow matching 的 |x| 均值应明显大于回归
    assert s_fm["abs_mean"] > s_reg["abs_mean"], (s_fm, s_reg)
    print("[test] flow_matching_toy OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        demo()
