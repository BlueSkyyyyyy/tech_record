#!/usr/bin/env python3
"""多智能体的 token 账本：并行不是免费的，隔离才是它的卖点。

对应专题第 9 篇《多智能体：context 隔离、handoff 与 token 倍率》。

用法：
    python3 multiagent_cost.py
    python3 multiagent_cost.py --workers 4 --worker-in 30000 --worker-out 1500
    python3 multiagent_cost.py --test

只依赖标准库。它只算账，不预测质量：

    单 agent：lead 自己读所有材料，context 线性增长，最后可能放不下。
    多 agent：lead 只拿计划 + 每个 worker 的 1~2k token 摘要；
              worker 各自有完整 context，读多少都不污染 lead。

代价是 token：N 个 worker 的输入都要付钱，而且 lead 还要付协调 token。
Anthropic 的实测是 agent 约为 chat 的 4 倍、多 agent 约为 15 倍；
这里把倍数拆成可调的几项，方便算自己的场景。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Price:
    input_per_m: float = 3.0
    output_per_m: float = 15.0


def cost(in_tokens: int, out_tokens: int, p: Price) -> float:
    return in_tokens / 1e6 * p.input_per_m + out_tokens / 1e6 * p.output_per_m


def single_agent(task_tokens: int, steps: int, p: Price) -> dict:
    """一个 agent 从头读到尾：每步都带上前面积累的 context。"""
    total_in = 0
    ctx = 2_000  # system + tools
    for _ in range(steps):
        ctx += task_tokens // steps
        total_in += ctx
    total_out = steps * 800
    return {
        "input": total_in,
        "output": total_out,
        "cost": cost(total_in, total_out, p),
        "peak_context": ctx,
    }


def multi_agent(workers: int, worker_in: int, worker_out: int, lead_in: int, lead_out: int, p: Price) -> dict:
    """lead 规划 + N 个 worker 并行；每个 worker 的完整轨迹不回传，只回摘要。"""
    worker_cost = workers * cost(worker_in, worker_out, p)
    lead_cost = cost(lead_in, lead_out, p)
    total_in = workers * worker_in + lead_in
    total_out = workers * worker_out + lead_out
    return {
        "input": total_in,
        "output": total_out,
        "cost": worker_cost + lead_cost,
        "peak_context": max(worker_in, lead_in),
    }


def report(args) -> None:
    p = Price(args.input_price, args.output_price)
    s = single_agent(args.task_tokens, args.steps, p)
    m = multi_agent(args.workers, args.worker_in, args.worker_out, args.lead_in, args.lead_out, p)
    chat_in, chat_out = 10_000, 1_000
    chat_cost = cost(chat_in, chat_out, p)
    print(f"price: ${p.input_per_m}/M in, ${p.output_per_m}/M out\n")
    print(f"{'':<16}{'single':>14}{'multi':>14}{'chat':>14}")
    print(f"{'input tok':<16}{s['input']:>14,}{m['input']:>14,}{chat_in:>14,}")
    print(f"{'output tok':<16}{s['output']:>14,}{m['output']:>14,}{chat_out:>14,}")
    print(f"{'cost $':<16}{s['cost']:>14,.2f}{m['cost']:>14,.2f}{chat_cost:>14,.2f}")
    print(f"{'peak ctx':<16}{s['peak_context']:>14,}{m['peak_context']:>14,}{chat_in:>14,}")
    print(f"\nmulti/chat token 倍率: {(m['input'] + m['output']) / (chat_in + chat_out):.2f}×")
    print(f"multi/single token 倍率: {(m['input'] + m['output']) / (s['input'] + s['output']):.2f}×")
    if m["peak_context"] < s["peak_context"]:
        print("lead 的峰值 context 更低——这是用 token 换 context 隔离。")


def test() -> None:
    p = Price(3.0, 15.0)
    s = single_agent(100_000, 10, p)
    m = multi_agent(4, 30_000, 1_500, 20_000, 4_000, p)
    assert s["input"] > 0 and m["input"] > 0
    assert m["peak_context"] == 30_000
    assert s["peak_context"] > m["peak_context"]
    assert abs(cost(1_000_000, 0, p) - 3.0) < 1e-9
    assert abs(cost(0, 1_000_000, p) - 15.0) < 1e-9
    # worker 数量线性增加输入 token
    a = multi_agent(2, 10_000, 1_000, 10_000, 1_000, p)
    b = multi_agent(4, 10_000, 1_000, 10_000, 1_000, p)
    assert b["input"] - a["input"] == 20_000
    print("multiagent_cost: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worker-in", type=int, default=30_000)
    ap.add_argument("--worker-out", type=int, default=1_500)
    ap.add_argument("--lead-in", type=int, default=20_000)
    ap.add_argument("--lead-out", type=int, default=4_000)
    ap.add_argument("--task-tokens", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--input-price", type=float, default=3.0)
    ap.add_argument("--output-price", type=float, default=15.0)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    test() if args.test else report(args)


if __name__ == "__main__":
    main()
