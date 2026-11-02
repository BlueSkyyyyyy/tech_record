#!/usr/bin/env python3
"""DP 梯度 bucket 的通算重叠模拟（CPU only）。

模型
----
- 反向按 bucket 逆序就绪：第 i 个 bucket 在 (i+1)*c 时刻算完，
  其中 c 是单桶反向计算时间。
- 每个 bucket 的梯度通信时间 t = volume/BW + launch。
- K 条通信通道（K 个 NCCL communicator / 多流），每条一次传一个桶；
  桶分配给最早空闲的通道。
- 暴露时间 = 最后一个桶的通信结束时间 - N*c。

这个模型解释了 Megatron `overlap_grad_reduce` 的核心 trade-off：
桶太大 -> 最后一个桶的 t 直接暴露；桶太小 -> launch 次数多、带宽打不满。
"""
from __future__ import annotations

import argparse


def simulate(n_buckets: int, c: float, t: float, channels: int = 1,
             launch: float = 0.0) -> tuple[list[list[tuple[float, float, int]]], float, float]:
    """返回 (每条通道的区间, makespan, exposed)。"""
    if n_buckets < 1 or channels < 1:
        raise ValueError("n_buckets and channels must be >= 1")
    t_eff = t + launch
    free = [0.0] * channels
    timeline: list[list[tuple[float, float, int]]] = [[] for _ in range(channels)]
    for i in range(n_buckets):
        ready = (i + 1) * c
        ch = min(range(channels), key=lambda k: free[k])
        start = max(ready, free[ch])
        end = start + t_eff
        timeline[ch].append((start, end, i))
        free[ch] = end
    makespan = max(free)
    exposed = makespan - n_buckets * c
    return timeline, makespan, exposed


def exposed_formula(n_buckets: int, c: float, t: float) -> float:
    """单通道闭式：t + (N-1) * max(0, t-c)。"""
    return t + (n_buckets - 1) * max(0.0, t - c)


def sweep(total_compute: float, total_comm: float, launch: float,
          channels: int = 1, max_buckets: int = 64) -> list[dict]:
    rows = []
    for n in [1, 2, 4, 8, 16, 32, 64]:
        if n > max_buckets:
            continue
        c = total_compute / n
        t = total_comm / n
        _, mk, exp = simulate(n, c, t, channels, launch)
        rows.append({"n": n, "c": c, "t": t + launch, "makespan": mk,
                     "exposed": exp, "overhead": mk - total_compute})
    return rows


def print_timeline(timeline, makespan: float, scale: int = 40) -> None:
    print(f"\ncomm timeline (1 char ≈ {makespan/scale:.2f} time units)")
    for ch, intervals in enumerate(timeline):
        row = ["."] * (scale + 1)
        for start, end, idx in intervals:
            a = min(scale, int(start / makespan * scale))
            b = min(scale, max(a + 1, int(end / makespan * scale)))
            for x in range(a, b):
                row[x] = str(idx % 10)
        print(f"ch{ch}: " + "".join(row))


def _self_test() -> None:
    # 单通道闭式
    for n, c, t in [(1, 1.0, 0.5), (4, 1.0, 0.5), (4, 1.0, 1.5), (16, 0.5, 0.5)]:
        _, _, exp = simulate(n, c, t, 1, 0)
        assert abs(exp - exposed_formula(n, c, t)) < 1e-9, (n, c, t, exp)

    # 多通道不会更差
    _, _, e1 = simulate(16, 1.0, 0.6, 1, 0)
    _, _, e4 = simulate(16, 1.0, 0.6, 4, 0)
    assert e4 <= e1 + 1e-9

    # launch 开销 -> 存在内部最优桶数
    rows = sweep(total_compute=100, total_comm=60, launch=2.0, channels=1)
    best = min(rows, key=lambda r: r["makespan"])
    assert 1 < best["n"] < 64, best

    # 无 launch 且通信快于计算时，桶越细尾延迟越小（暴露趋近单桶 t）
    rows = sweep(total_compute=100, total_comm=60, launch=0.0, channels=1)
    assert rows[-1]["makespan"] <= rows[0]["makespan"] + 1e-9

    print("bucket_sim: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--compute", type=float, default=1.0, help="per-bucket compute")
    ap.add_argument("--comm", type=float, default=0.6, help="per-bucket comm")
    ap.add_argument("--launch", type=float, default=0.0)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--sweep", action="store_true",
                    help="fix total compute/comm volume and sweep bucket count")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        _self_test()
        return

    if args.sweep:
        print(f"total_compute=100 total_comm=60 launch={args.launch} channels={args.channels}")
        print(f"{'N':>4} {'c':>7} {'t':>7} {'makespan':>10} {'exposed':>9} {'overhead':>9}")
        for r in sweep(100, 60, args.launch, args.channels):
            print(f"{r['n']:>4} {r['c']:>7.2f} {r['t']:>7.2f} "
                  f"{r['makespan']:>10.2f} {r['exposed']:>9.2f} {r['overhead']:>9.2f}")
        return

    timeline, mk, exp = simulate(args.buckets, args.compute, args.comm,
                                 args.channels, args.launch)
    print(f"N={args.buckets} c={args.compute} t={args.comm} launch={args.launch} "
          f"channels={args.channels}")
    print(f"makespan={mk:.2f} compute_total={args.buckets*args.compute:.2f} "
          f"exposed={exp:.2f} "
          f"overlap={1 - exp/(args.buckets*(args.comm+args.launch)):.1%}")
    print_timeline(timeline, mk)


if __name__ == "__main__":
    main()
