#!/usr/bin/env python3
"""通算并行的统一性能账本（CPU only, no dependencies）.

这个模块把系列里反复用到的几个"账"放在一处：

- 流水线气泡：1F1B / Zero Bubble 的解析式（按 ZB 论文 Table 2 的定义）
- DP 梯度 bucket 的暴露时间：N 个 bucket、单桶计算 c、通信 t
- TP 分块 all-gather + GEMM 的暴露时间：n 个 chunk
- 通算并行的总账：T = compute + exposed + bubble

所有公式都在 `_self_test()` 里用可手算的小例子对过。运行：

    python overlap_model.py          # demo 表
    python overlap_model.py --test   # 自测
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Ledger:
    """一次迭代/一层的通算账本。单位任意，只要一致。"""

    compute: float          # 纯计算时间
    comm: float             # 通信总时间（含被藏住的部分）
    exposed: float = 0.0    # 关键路径上仍暴露的通信
    bubble: float = 0.0     # 因依赖/流水线空转浪费的时间

    @property
    def total(self) -> float:
        return self.compute + self.exposed + self.bubble

    @property
    def overlap_efficiency(self) -> float:
        """通信被藏住的比例；comm=0 时定义为 1。"""
        if self.comm <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.exposed / self.comm))

    @property
    def bubble_rate(self) -> float:
        return self.bubble / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"compute={self.compute:.3f} comm={self.comm:.3f} "
            f"exposed={self.exposed:.3f} bubble={self.bubble:.3f} "
            f"total={self.total:.3f} overlap={self.overlap_efficiency:.1%} "
            f"bubble_rate={self.bubble_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# 1. 流水线：解析式
# ---------------------------------------------------------------------------
def pipeline_1f1b(p: int, m: int, f: float, b: float, w: float = 0.0) -> Ledger:
    """1F1B 的气泡账。

    采用 Zero Bubble 论文 Table 2 的定义：
        bubble = (p - 1) * (T_F + T_B + T_W)
        ideal  = m * (T_F + T_B + T_W)
    这里把 "通信" 记为 0：PP 的点对点通信默认被算进 stage 时间里。
    """
    if p < 1 or m < 1:
        raise ValueError("p and m must be >= 1")
    per = f + b + w
    bubble = (p - 1) * per
    return Ledger(compute=m * per, comm=0.0, bubble=bubble)


def pipeline_zbh2(p: int, m: int, f: float, b: float, w: float) -> Ledger:
    """ZB-H2 的气泡账：bubble = (p-1)(T_F + T_B - 2 T_W)。

    w 足够大时气泡为 0；注意这是**单步**公式，跨 step 的稳态还要看
    `schedule_sim.py --iters`。负值截断为 0。
    """
    per = f + b + w
    bubble = max(0.0, (p - 1) * (f + b - 2 * w))
    return Ledger(compute=m * per, comm=0.0, bubble=bubble)


def pipeline_dualpipe(p: int, m: int, f: float, b: float, w: float,
                      fb: float) -> Ledger:
    """DualPipe 论文 README 的气泡式：bubble = (p/2 - 1)(F&B + B - 3W)。

    p 必须是偶数；F&B 表示一对前向/反向重叠后的等效时间。
    """
    if p % 2:
        raise ValueError("DualPipe requires an even number of PP stages")
    per = f + b + w
    bubble = max(0.0, (p / 2 - 1) * (fb + b - 3 * w))
    return Ledger(compute=m * per, comm=0.0, bubble=bubble)


# ---------------------------------------------------------------------------
# 2. DP：bucket 化梯度通信
# ---------------------------------------------------------------------------
def dp_exposed(n_buckets: int, c: float, t: float) -> float:
    """N 个 bucket 依次就绪（每 c 时间一个），通信每桶 t，单通道 FIFO。

    第 i 个 bucket 在 (i+1)*c 就绪；通信完成时间递推：
        finish_i = max((i+1)c, finish_{i-1}) + t
    暴露时间 = finish_N - N*c（最后一个 bucket 的计算结束之后仍在通信）。

    闭式：exposed = t + (N-1) * max(0, t - c)
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    return t + (n_buckets - 1) * max(0.0, t - c)


def dp_ledger(n_buckets: int, c: float, t: float, launch: float = 0.0) -> Ledger:
    """把 launch 开销算进每桶通信时间，返回完整账本。"""
    t_eff = t + launch
    return Ledger(compute=n_buckets * c, comm=n_buckets * t_eff,
                  exposed=dp_exposed(n_buckets, c, t_eff))


# ---------------------------------------------------------------------------
# 3. TP：分块 all-gather + GEMM
# ---------------------------------------------------------------------------
def tp_exposed(n_chunks: int, c: float, g: float) -> float:
    """n 个 chunk 的 two-stage 流水（先通信 g，再计算 c）。

    总时间 = max(n*c, n*g) + min(c, g)；暴露 = 总时间 - n*c。
    n=1 时总时间 = c+g，暴露 = g（没有重叠机会）。
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    total = max(n_chunks * c, n_chunks * g) + min(c, g)
    return total - n_chunks * c


def tp_ledger(n_chunks: int, c: float, g: float) -> Ledger:
    return Ledger(compute=n_chunks * c, comm=n_chunks * g,
                  exposed=tp_exposed(n_chunks, c, g))


# ---------------------------------------------------------------------------
# 4. MoE：dispatch / expert / combine 三段
# ---------------------------------------------------------------------------
def _two_machine_makespan(tasks, comm_tasks: set) -> float:
    """两台机器（compute / comm）的贪心 list scheduler。

    tasks: list of (name, dur, deps[name...])
    comm_tasks: 哪些任务占通信机器；其余占计算机器。
    优先级：通信任务优先（D 早发、B 早收），同机器 FIFO。
    """
    done: dict[str, float] = {}
    busy = {"comm": None, "compute": None}
    t = 0.0
    pending = {name: (dur, deps) for name, dur, deps in tasks}
    while pending or busy["comm"] or busy["compute"]:
        for mach in ("comm", "compute"):
            if busy[mach] is not None:
                continue
            ready = [
                (name, dur)
                for name, (dur, deps) in pending.items()
                if (mach == "comm") == (name in comm_tasks)
                and all(dep in done for dep in deps)
            ]
            if ready:
                # 通信任务里 D 先于 B；计算任务只有一个候选
                name, dur = sorted(ready, key=lambda x: (x[0][0] != "D", x[0]))[0]
                busy[mach] = (name, dur, t + dur)
        if not busy["comm"] and not busy["compute"]:
            # 推进到最早可满足的依赖完成时间
            candidates = [done[dep] for _, (_, deps) in pending.items() for dep in deps if dep in done]
            if not candidates:
                raise RuntimeError("deadlock in two-machine schedule")
            t = min(candidates)
            continue
        t = min(v[2] for v in busy.values() if v is not None)
        for mach in ("comm", "compute"):
            if busy[mach] is not None and abs(busy[mach][2] - t) < 1e-12:
                name, _, _ = busy[mach]
                done[name] = t
                pending.pop(name, None)
                busy[mach] = None
    return t


def moe_exposed(n_chunks: int, d: float, c: float, b: float) -> float:
    """三段流水的粗模型：每个 chunk 依次 dispatch(d) -> compute(c) -> combine(b)。

    dispatch 和 combine 共享一条通信通道，compute 独占计算通道；
    D/C/B 之间只有同 chunk 的依赖。用两台机器的 list scheduler 求 makespan，
    返回 makespan - n*c（纯专家计算时间）。
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    tasks = []
    for i in range(n_chunks):
        tasks.append((f"D{i}", d, ()))
        tasks.append((f"C{i}", c, (f"D{i}",)))
        tasks.append((f"B{i}", b, (f"C{i}",)))
    makespan = _two_machine_makespan(tasks, {f"D{i}" for i in range(n_chunks)} | {f"B{i}" for i in range(n_chunks)})
    return makespan - n_chunks * c


def moe_ledger(n_chunks: int, d: float, c: float, b: float) -> Ledger:
    return Ledger(compute=n_chunks * c, comm=n_chunks * (d + b),
                  exposed=moe_exposed(n_chunks, d, c, b))


# ---------------------------------------------------------------------------
# demo / self-test
# ---------------------------------------------------------------------------
def _demo() -> None:
    print("== 流水线气泡（F=B=W=1）==")
    print(f"{'p':>3} {'m':>4} {'1F1B':>8} {'ZB-H2':>8} {'1F1B气泡率':>12} {'ZB气泡率':>10}")
    for p, m in [(4, 16), (8, 24), (8, 64), (16, 48), (16, 128)]:
        a = pipeline_1f1b(p, m, 1, 1, 1)
        z = pipeline_zbh2(p, m, 1, 1, 1)
        print(f"{p:>3} {m:>4} {a.total:>8.1f} {z.total:>8.1f} "
              f"{a.bubble_rate:>11.1%} {z.bubble_rate:>9.1%}")

    print("\n== DP bucket：N 个桶，c=1.0，t=0.6（通信比计算快）==")
    print(f"{'N':>4} {'exposed':>9} {'overlap':>9}")
    for n in [1, 2, 4, 8, 16, 32]:
        e = dp_exposed(n, 1.0, 0.6)
        print(f"{n:>4} {e:>9.2f} {1 - e / (n * 0.6):>8.1%}")

    print("\n== TP 分块：c=1.0，g=1.5（通信比计算慢）==")
    print(f"{'chunks':>6} {'total':>8} {'exposed':>9}")
    for n in [1, 2, 4, 8, 16]:
        led = tp_ledger(n, 1.0, 1.5)
        print(f"{n:>6} {led.total:>8.2f} {led.exposed:>9.2f}")


def _self_test() -> None:
    # 1F1B: p=8,m=24,F=B=W=1 -> bubble=21, total=93, rate=22.58%
    led = pipeline_1f1b(8, 24, 1, 1, 1)
    assert led.bubble == 21
    assert abs(led.total - 93) < 1e-9
    assert abs(led.bubble_rate - 21 / 93) < 1e-9

    # ZB-H2: F=B=W=1 -> 气泡 0
    assert pipeline_zbh2(8, 24, 1, 1, 1).bubble == 0
    # ZB-H2 的公式在 W 很小时仍有气泡
    assert pipeline_zbh2(8, 24, 1, 1, 0.1).bubble == (8 - 1) * (1 + 1 - 0.2)

    # DP: N=1 时暴露 t；t<=c 时暴露恒为 t；t>c 时线性增长
    assert dp_exposed(1, 1.0, 0.6) == 0.6
    assert dp_exposed(8, 1.0, 0.6) == 0.6
    assert abs(dp_exposed(8, 1.0, 1.5) - (1.5 + 7 * 0.5)) < 1e-9

    # TP: n=1 无重叠；n 很大时暴露趋近于 min(c,g)*... 且 <= n*max(0,g-c)+min
    assert tp_exposed(1, 1.0, 1.5) == 1.5
    assert tp_exposed(10, 1.0, 1.5) == (10 * 1.5 + 1.0) - 10 * 1.0
    assert tp_exposed(10, 1.0, 0.5) == (10 * 1.0 + 0.5) - 10 * 1.0

    # MoE: 单 chunk 暴露 = d+b；多 chunk 应严格小于串行
    assert moe_exposed(1, 1.0, 2.0, 1.0) == 2.0
    assert moe_exposed(8, 1.0, 2.0, 1.0) < 8 * 2.0

    # 账本恒等式与边界
    led = tp_ledger(4, 1.0, 0.5)
    assert abs(led.total - (led.compute + led.exposed + led.bubble)) < 1e-12
    assert 0 <= led.overlap_efficiency <= 1
    assert Ledger(1, 0).overlap_efficiency == 1.0

    print("overlap_model: all tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="run self-test")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        _demo()
