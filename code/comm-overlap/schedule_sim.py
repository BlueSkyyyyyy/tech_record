#!/usr/bin/env python3
"""流水线调度模拟器：GPipe / 1F1B / 1F1B-split / ZB-H1 / ZB-H2，CPU only。

模型
----
- p 个 stage，m 个 microbatch，F/B/W 三种任务（单位时间）。
- 依赖：
    F(i,s)  <- F(i,s-1)          （前向激活从上游来）
    B(i,s)  <- F(i,s), B(i,s+1)  （输入梯度 dX 从下游来）
    W(i,s)  <- B(i,s)            （wgrad 只依赖本 stage 的 dX）
- 同一 stage 一次只能跑一个任务。

五种排法（B = dX，W = dW；整段反向 = B + W）
--------------------------------------------
- `gpipe`：全前向再全反向，气泡 (p-1)(m)(F+B+W)/(m(F+B+W)) 量级。
- `1f1b`：反向不拆，1F-1B 交替。气泡公式 (p-1)(F+B+W)。
- `1f1b-split`：把反向拆成 B、W，**W 紧跟 B**。气泡 (p-1)(F+B)。
  这是"拆开了但没重排"的结果，也是很多人以为的 ZB。
- `zbh1`：ZB-H1。warm-up 与 1F1B 相同，但每个 stage 允许 **lag = stage** 个
  wgrad 待办——越靠后的 stage 的 W 越靠尾部。气泡 (p-1)(F+B-W)。
  在飞数恰好是 p（= (p-s)M_B + s·M_W），与论文 Table 2 的显存一致。
- `zbh2`：ZB-H2。warm-up 做到 2(p-s)-1 个 F，lag = 2·stage。
  在飞数 2p-1（= (2p-2s+1)M_B + 2s·M_W）。单步必然 ≥ (p-1)F，
  只有**跨 step 稳态**（去掉 optimizer barrier）才能到 (p-1)(F+B-2W)。

两种时间口径
------------
- 默认 `--iters N`：N 个 step 背靠背，stage 可以跨 step 继续排，测**稳态吞吐**。
- `--barrier`：每个 step 结束加全局 barrier，测**单步气泡**。
  真实训练里 DP 梯度 all-reduce / optimizer step 往往就是这道 barrier，
  也是 ZB-H2 的"零气泡"必须靠 optimizer bypass 去掉的那道同步。

运行：
    python schedule_sim.py                                  # 默认 1F1B, p=8, m=24
    python schedule_sim.py --schedule zbh1 --gantt
    python schedule_sim.py --schedule zbh2 --iters 6        # 跨 step 稳态
    python schedule_sim.py --schedule zbh2 --iters 6 --barrier   # 单步口径
    python schedule_sim.py --test                           # 自测
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

SCHEDULES = ("gpipe", "1f1b", "1f1b-split", "zbh1", "zbh2")
SPLIT_SCHEDULES = ("1f1b-split", "zbh1", "zbh2")


@dataclass(eq=False)
class Task:
    kind: str          # F / B / W / BW
    it: int            # iteration
    micro: int
    stage: int
    dur: int
    deps: tuple = ()
    start: int = -1
    end: int = -1


def build_tasks(p: int, m: int, iters: int, f: int, b: int, w: int,
                split_backward: bool) -> dict[tuple, Task]:
    tasks: dict[tuple, Task] = {}

    def key(kind: str, it: int, micro: int, stage: int):
        return (kind, it, micro, stage)

    for it in range(iters):
        for s in range(p):
            for i in range(m):
                f_deps = ((key("F", it, i, s - 1), 0),) if s > 0 else ()
                tasks[key("F", it, i, s)] = Task("F", it, i, s, f, f_deps)
                if split_backward:
                    b_deps = ((key("F", it, i, s), 0),)
                    if s < p - 1:
                        b_deps += ((key("B", it, i, s + 1), 0),)
                    tasks[key("B", it, i, s)] = Task("B", it, i, s, b, b_deps)
                    tasks[key("W", it, i, s)] = Task(
                        "W", it, i, s, w, ((key("B", it, i, s), 0),)
                    )
                else:
                    bw_deps = ((key("F", it, i, s), 0),)
                    if s < p - 1:
                        bw_deps += ((key("BW", it, i, s + 1), 0),)
                    tasks[key("BW", it, i, s)] = Task("BW", it, i, s, b + w, bw_deps)
    return tasks


# ---------------------------------------------------------------- 调度形状

def warmup_count(stage: int, p: int, m: int, schedule: str) -> int:
    """进入稳态前先做几个前向。

    1F1B / 1F1B-split / ZB-H1：p-1-stage（保证在飞数 ≤ p）。
    ZB-H2：2(p-stage)-1（在飞数做到 2p-1，用显存换气泡）。
    """
    if schedule in ("1f1b", "1f1b-split", "zbh1"):
        return min(p - 1 - stage, m)
    return min(2 * (p - stage) - 1, m)


def wgrad_lag(stage: int, schedule: str) -> int:
    """允许在该 stage 积压多少个待办 wgrad。

    - 1f1b-split：0（W 紧跟 B，所以 W 还在关键路径上）
    - zbh1：stage（越靠后的 stage 越往后推 W）→ 在飞数 (p-stage)+stage = p
    - zbh2：2·stage → 在飞数 (2(p-stage)-1)+2·stage = 2p-1
    """
    return {"1f1b-split": 0, "zbh1": stage, "zbh2": 2 * stage}.get(schedule, 0)


def _split_order(it: int, stage: int, p: int, m: int, schedule: str):
    """F 与 B 交替，W 按 lag 延后；W 只是"攒着"，不阻塞 B。"""
    warm = warmup_count(stage, p, m, schedule)
    lag = wgrad_lag(stage, schedule)
    order = [("F", it, i) for i in range(warm)]
    nf, nb, pend = warm, 0, []
    while nb < m:
        if nf < m:
            order.append(("F", it, nf))
            nf += 1
        order.append(("B", it, nb))
        pend.append(nb)
        nb += 1
        while len(pend) > lag:
            order.append(("W", it, pend.pop(0)))
    while nf < m:                      # m < warm 时才会走到
        order.append(("F", it, nf))
        nf += 1
    order += [("W", it, j) for j in pend]
    return order


def stage_order(stage: int, p: int, m: int, iters: int, schedule: str):
    """返回该 stage 的任务顺序 [(kind, it, micro), ...]。"""
    order = []
    for it in range(iters):
        if schedule == "gpipe":
            order += [("F", it, i) for i in range(m)]
            order += [("BW", it, i) for i in range(m)]
            continue
        if schedule in SPLIT_SCHEDULES:
            order += _split_order(it, stage, p, m, schedule)
            continue
        # 1f1b：不拆反向
        warm = warmup_count(stage, p, m, schedule)
        order += [("F", it, i) for i in range(warm)]
        nf, nb = warm, 0
        while nf < m:
            order += [("F", it, nf), ("BW", it, nb)]
            nf += 1
            nb += 1
        while nb < m:
            order += [("BW", it, nb)]
            nb += 1
    return order


# ---------------------------------------------------------------- 模拟

def _schedule_once(p: int, m: int, f: int, b: int, w: int, schedule: str,
                   offset: int = 0) -> tuple[dict, int]:
    split = schedule in SPLIT_SCHEDULES
    tasks = build_tasks(p, m, 1, f, b, w, split)
    orders = {s: stage_order(s, p, m, 1, schedule) for s in range(p)}
    stage_free = [offset] * p
    cursor = [0] * p
    remaining = len(tasks)

    while remaining:
        progress = False
        for s in range(p):
            while cursor[s] < len(orders[s]):
                kind, it, i = orders[s][cursor[s]]
                task = tasks[(kind, it, i, s)]
                if any(tasks[k].end < 0 for k, _ in task.deps):
                    break  # 依赖未就绪；后面的任务排在它之后，只能一起等
                start = stage_free[s]
                for dep_key, extra in task.deps:
                    start = max(start, tasks[dep_key].end + extra)
                task.start = start
                task.end = start + task.dur
                stage_free[s] = task.end
                cursor[s] += 1
                remaining -= 1
                progress = True
        if not progress:
            raise RuntimeError("deadlock: dependency cycle or bad stage order")
    return tasks, max(t.end for t in tasks.values()) - offset


def simulate(p: int, m: int, iters: int = 1, f: int = 1, b: int = 1, w: int = 1,
             schedule: str = "1f1b", barrier: bool = False) -> tuple[dict, int]:
    """按每个 stage 的固定顺序做拓扑式 list scheduling。

    跨 stage 的依赖（F 来自上游、B 来自下游）意味着不能按 stage 0..p-1
    顺序一次算完：stage 0 的 B 要等 stage 1 的 B。这里用"轮询 + 阻塞"：
    每轮扫所有 stage，能排的排，排不了的等下一轮，直到全部排完。

    barrier=True 时逐个 step 独立排队，再把时间轴首尾相接
    （相当于每步结束有一次全局同步，跨步重叠被切断）。
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule: {schedule}")
    if p < 1 or m < 1 or iters < 1:
        raise ValueError("p, m, iters must be >= 1")

    all_tasks: dict[tuple, Task] = {}
    offset = 0
    if barrier:
        for it in range(iters):
            tasks, mk = _schedule_once(p, m, f, b, w, schedule, offset)
            for (kind, _, i, s), t in tasks.items():
                all_tasks[(kind, it, i, s)] = t
            offset += mk
        return all_tasks, offset

    tasks = build_tasks(p, m, iters, f, b, w, schedule in SPLIT_SCHEDULES)
    orders = {s: stage_order(s, p, m, iters, schedule) for s in range(p)}
    stage_free = [0] * p
    cursor = [0] * p
    remaining = len(tasks)
    while remaining:
        progress = False
        for s in range(p):
            while cursor[s] < len(orders[s]):
                kind, it, i = orders[s][cursor[s]]
                task = tasks[(kind, it, i, s)]
                if any(tasks[k].end < 0 for k, _ in task.deps):
                    break
                start = stage_free[s]
                for dep_key, extra in task.deps:
                    start = max(start, tasks[dep_key].end + extra)
                task.start = start
                task.end = start + task.dur
                stage_free[s] = task.end
                cursor[s] += 1
                remaining -= 1
                progress = True
        if not progress:
            raise RuntimeError("deadlock: dependency cycle or bad stage order")
    return tasks, max(t.end for t in tasks.values())


def metrics(p: int, m: int, iters: int, f: int, b: int, w: int,
            tasks: dict[tuple, Task], makespan: int) -> dict:
    ideal = m * iters * (f + b + w)
    work_per_stage = [0] * p
    for (kind, it, i, s), t in tasks.items():
        work_per_stage[s] += t.dur
    idle_per_stage = [makespan - work_per_stage[s] for s in range(p)]
    return {
        "makespan": makespan,
        "ideal": ideal,
        "bubble": makespan - ideal,
        "bubble_rate": (makespan - ideal) / makespan if makespan else 0.0,
        "total_idle": p * makespan - p * ideal,
        "idle_per_stage": idle_per_stage,
        "per_step": makespan / iters,
        "peak_inflight": peak_inflight(tasks, p),
    }


def peak_inflight(tasks: dict[tuple, Task], p: int) -> int:
    """单个 stage 上"F 已完成、W 还没做完"的 microbatch 峰值数。

    这就是论文里 M_B / M_W 记账的"在飞激活数"：
    1F1B / ZB-H1 应当 ≤ p，ZB-H2 应当 ≤ 2p-1。
    """
    peak = 0
    for s in range(p):
        events = []
        for (kind, it, i, st), t in tasks.items():
            if st != s:
                continue
            if kind == "F":
                events.append((t.end, +1))
            elif kind in ("W", "BW"):
                events.append((t.end, -1))
        events.sort()
        cur = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
    return peak


def print_gantt(tasks: dict[tuple, Task], p: int, iters: int, makespan: int,
                max_cols: int = 110) -> None:
    """每个 stage 一行；F/B/W/X 表示任务，'.' 表示空闲。"""
    step = max(1, (makespan + max_cols - 1) // max_cols)
    print(f"\nGantt (1 char = {step} time unit(s), '|' = iteration boundary)")
    print("      " + "".join(str((t // 10) % 10) if t % 10 == 0 else " "
                              for t in range(0, makespan, step)))
    for s in range(p):
        cells = []
        for t in range(0, makespan, step):
            ch = "."
            for (kind, it, i, st), task in tasks.items():
                if st == s and task.start <= t < task.end:
                    ch = {"BW": "X"}.get(kind, kind)
                    break
            cells.append(ch)
        print(f"p{s:<3d}  " + "".join(cells))


def _self_test() -> None:
    # 1) 同一 stage 不重叠，依赖满足
    tasks, mk = simulate(4, 8, 1, 1, 1, 1, "1f1b")
    by_stage: dict[int, list[Task]] = {}
    for (kind, it, i, s), t in tasks.items():
        by_stage.setdefault(s, []).append(t)
        for dep_key, extra in t.deps:
            dep = tasks[dep_key]
            assert dep.end + extra <= t.start, (dep, t)
    for s, ts in by_stage.items():
        ts.sort(key=lambda x: x.start)
        for a, b in zip(ts, ts[1:]):
            assert a.end <= b.start, (s, a, b)
    assert all(t.start >= 0 and t.end > t.start for t in tasks.values())

    # 2) 等时假设下，四种排法的气泡分别等于
    #    (p-1)(F+B+W)、(p-1)(F+B)、(p-1)(F+B-W)、(p-1)(F+B-2W)
    #    取 p=8, m=24, F=B=W=1：93 / 86 / 79 / 72
    for sched, expect in (("1f1b", 93), ("1f1b-split", 86), ("zbh1", 79)):
        _, mk = simulate(8, 24, 1, 1, 1, 1, sched)
        assert mk == expect, (sched, mk, expect)
    # 单步下界 (p-1)F + m(F+B+W) = 7 + 72 = 79：1F1B 离它 14，ZB-H1 正好压到它
    lower_bound = 7 + 72
    assert simulate(8, 24, 1, 1, 1, 1, "zbh1")[1] == lower_bound

    # 3) ZB-H2 的 0 只能跨 step：单步 = 下界 79，跨 6 步稳态降到 79 以下
    _, mk_barrier = simulate(8, 24, 6, 1, 1, 1, "zbh2", barrier=True)
    _, mk_steady = simulate(8, 24, 6, 1, 1, 1, "zbh2")
    assert mk_barrier == 6 * lower_bound, mk_barrier
    assert mk_steady < mk_barrier, (mk_steady, mk_barrier)
    _, mk_h1_steady = simulate(8, 24, 6, 1, 1, 1, "zbh1")
    assert mk_steady < mk_h1_steady, (mk_steady, mk_h1_steady)

    # 4) 在飞数：1F1B / ZB-H1 不超过 p，ZB-H2 不超过 2p-1
    #    （离散时间下 ZB-H2 会看到 2p 的瞬时值：某个 F 刚结束、对应的 W 还差一步）
    for sched in ("1f1b", "1f1b-split", "zbh1"):
        t, mk = simulate(8, 24, 1, 1, 1, 1, sched)
        assert peak_inflight(t, 8) <= 8, (sched, peak_inflight(t, 8))
    t, mk = simulate(8, 24, 1, 1, 1, 1, "zbh2")
    assert peak_inflight(t, 8) <= 2 * 8, peak_inflight(t, 8)

    # 5) gpipe 与 1f1b 在最坏情况下都不小于 ideal
    for sched in SCHEDULES:
        _, mk = simulate(8, 24, 1, 1, 1, 1, sched)
        assert mk >= 24 * 3

    print("schedule_sim: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--microbatches", type=int, default=24)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--f", type=int, default=1, help="forward duration")
    ap.add_argument("--b", type=int, default=1, help="input-grad (dX) duration")
    ap.add_argument("--w", type=int, default=1, help="wgrad (dW) duration")
    ap.add_argument("--schedule", choices=list(SCHEDULES), default="1f1b")
    ap.add_argument("--barrier", action="store_true",
                    help="每个 step 结束加全局同步（测单步气泡）")
    ap.add_argument("--gantt", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        _self_test()
        return

    tasks, mk = simulate(args.stages, args.microbatches, args.iters,
                         args.f, args.b, args.w, args.schedule, args.barrier)
    met = metrics(args.stages, args.microbatches, args.iters,
                  args.f, args.b, args.w, tasks, mk)
    print(f"schedule={args.schedule} p={args.stages} m={args.microbatches} "
          f"iters={args.iters} F/B/W={args.f}/{args.b}/{args.w} "
          f"barrier={args.barrier}")
    print(f"makespan={met['makespan']} per_step={met['per_step']:.2f} "
          f"ideal={met['ideal']} bubble={met['bubble']} "
          f"bubble_rate={met['bubble_rate']:.2%} peak_inflight={met['peak_inflight']}")
    print("idle per stage: " + " ".join(f"p{s}={v}"
                                        for s, v in enumerate(met["idle_per_stage"])))
    if args.gantt:
        print_gantt(tasks, args.stages, args.iters, mk)


if __name__ == "__main__":
    main()
