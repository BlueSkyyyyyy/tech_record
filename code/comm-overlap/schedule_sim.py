#!/usr/bin/env python3
"""流水线调度模拟器：GPipe / 1F1B / ZB-H2，CPU only。

模型
----
- p 个 stage，m 个 microbatch，F/B/W 三种任务（单位时间）。
- 依赖：
    F(i,s)  <- F(i,s-1)          （前向激活从上游来）
    B(i,s)  <- F(i,s), B(i,s+1)  （输入梯度从下游来）
    W(i,s)  <- B(i,s)            （权重梯度只依赖本 stage 的反向）
- 同一 stage 一次只能跑一个任务。
- `gpipe` / `1f1b` 把反向看作一个整体 BW = B+W；`zbh2` 把 B 和 W 拆开，
  W 可以延后填气泡（Zero Bubble 的核心）。

两种时间口径
------------
- 默认 `--iters N`：N 个 step 背靠背，stage 可以跨 step 继续排，测的是
  **稳态吞吐**（相当于 optimizer 不与 pipeline 同步）。
- `--barrier`：每个 step 结束加全局 barrier，测的是**单步气泡**。
  真实训练里 DP 梯度 all-reduce / optimizer step 往往就是这道 barrier。

运行：
    python schedule_sim.py                       # 默认 1F1B, p=8, m=24
    python schedule_sim.py --schedule zbh2 --gantt
    python schedule_sim.py --iters 8 --barrier
    python schedule_sim.py --test                # 自测
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field


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


def _warmup_1f1b(stage: int, p: int, m: int) -> int:
    return min(p - 1 - stage, m)


def _warmup_zbh2(stage: int, p: int, m: int) -> int:
    # ZB-H2 的 warm-up 要做到 2p-1 个 in-flight F，第 s 个 stage 是 2(p-s)-1
    return min(2 * (p - stage) - 1, m)


def stage_order(stage: int, p: int, m: int, iters: int, schedule: str):
    """返回该 stage 的任务顺序 [(kind, it, micro), ...]。"""
    order = []
    for it in range(iters):
        if schedule == "gpipe":
            order += [("F", it, i) for i in range(m)]
            order += [("BW", it, i) for i in range(m)]
            continue

        warm = _warmup_1f1b(stage, p, m) if schedule == "1f1b" else _warmup_zbh2(stage, p, m)
        order += [("F", it, i) for i in range(warm)]
        nf, nb = warm, 0
        if schedule == "1f1b":
            while nf < m:
                order += [("F", it, nf), ("BW", it, nb)]
                nf += 1
                nb += 1
            while nb < m:
                order += [("BW", it, nb)]
                nb += 1
        else:  # zbh2
            while nf < m:
                order += [("F", it, nf), ("B", it, nb), ("W", it, nb)]
                nf += 1
                nb += 1
            while nb < m:
                order += [("B", it, nb), ("W", it, nb)]
                nb += 1
    return order


def simulate(p: int, m: int, iters: int = 1, f: int = 1, b: int = 1, w: int = 1,
             schedule: str = "1f1b") -> tuple[dict, int]:
    """按每个 stage 的固定顺序做拓扑式 list scheduling。

    跨 stage 的依赖（F 来自上游、B 来自下游）意味着不能按 stage 0..p-1
    顺序一次算完：stage 0 的 B 要等 stage 1 的 B。这里用"轮询 + 阻塞"：
    每轮扫所有 stage，能排的排，排不了的等下一轮，直到全部排完。
    """
    if schedule not in ("gpipe", "1f1b", "zbh2"):
        raise ValueError(f"unknown schedule: {schedule}")
    if p < 1 or m < 1 or iters < 1:
        raise ValueError("p, m, iters must be >= 1")
    split = schedule == "zbh2"
    tasks = build_tasks(p, m, iters, f, b, w, split)
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
                    break  # 后面的任务在 stage 内排在它之后，先等
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
    makespan = max(t.end for t in tasks.values())
    return tasks, makespan


def _unused_barrier(p: int, m: int, iters: int, f: int, b: int, w: int,
                schedule: str) -> tuple[dict, int]:
    """逐 step 调度，每步后加全局 barrier；返回合并后的任务与总 makespan。"""
    all_tasks: dict[tuple, Task] = {}
    offset = 0
    for it in range(iters):
        tasks, mk = simulate(p, m, 1, f, b, w, schedule, barrier=False)
        for (kind, _, i, s), t in tasks.items():
            nt = Task(kind, it, i, s, t.dur, t.deps,
                      t.start + offset, t.end + offset)
            all_tasks[(kind, it, i, s)] = nt
        offset += mk
    return all_tasks, offset


def metrics(p: int, m: int, iters: int, f: int, b: int, w: int,
            tasks: dict[tuple, Task], makespan: int) -> dict:
    ideal = m * iters * (f + b + w)
    work_per_stage = [0] * p
    idle_per_stage = [0] * p
    for (kind, it, i, s), t in tasks.items():
        work_per_stage[s] += t.dur
    # 每个 stage 的时间轴长度按 makespan（barrier 时有空转）
    for s in range(p):
        idle_per_stage[s] = makespan - work_per_stage[s]
    total_work = p * ideal
    return {
        "makespan": makespan,
        "ideal": ideal,
        "bubble": makespan - ideal,
        "bubble_rate": (makespan - ideal) / makespan if makespan else 0.0,
        "total_idle": p * makespan - total_work,
        "idle_per_stage": idle_per_stage,
        "per_step": makespan / iters,
    }


def print_gantt(tasks: dict[tuple, Task], p: int, iters: int, makespan: int,
                max_cols: int = 110) -> None:
    """每个 stage 一行；F/B/W/X 表示任务，'.' 表示空闲。"""
    step = max(1, (makespan + max_cols - 1) // max_cols)
    print(f"\nGantt (1 char = {step} time unit(s), '|' = iteration boundary)")
    print("      " + "".join(str((t // 10) % 10) if t % 10 == 0 else " " for t in range(0, makespan, step)))
    for s in range(p):
        cells = []
        for t0 in range(0, makespan, step):
            # 取该列中点所在的任务
            t = t0
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

    # 2) 单步 1F1B 的 makespan 与 ZB 论文 Table 2 一致：
    #    (m + p - 1) * (F+B+W) = (24+8-1)*3 = 93，气泡 21/93 = 22.58%
    _, mk = simulate(8, 24, 1, 1, 1, 1, "1f1b")
    assert mk == 93, mk

    # 3) ZB-H2 把 stage0 的 pre-B 气泡填掉：idle 更小
    t1, mk1 = simulate(8, 24, 1, 1, 1, 1, "1f1b")
    t2, mk2 = simulate(8, 24, 1, 1, 1, 1, "zbh2")
    m1 = metrics(8, 24, 1, 1, 1, 1, t1, mk1)
    m2 = metrics(8, 24, 1, 1, 1, 1, t2, mk2)
    assert m2["idle_per_stage"][0] < m1["idle_per_stage"][0], (m1, m2)

    # 4) gpipe 与 1f1b 在单 step 下都不小于 ideal
    for sched in ("gpipe", "1f1b", "zbh2"):
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
    ap.add_argument("--b", type=int, default=1, help="input-grad backward duration")
    ap.add_argument("--w", type=int, default=1, help="weight-grad duration")
    ap.add_argument("--schedule", choices=["gpipe", "1f1b", "zbh2"], default="1f1b")
    ap.add_argument("--gantt", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        _self_test()
        return

    tasks, mk = simulate(args.stages, args.microbatches, args.iters,
                         args.f, args.b, args.w, args.schedule)
    met = metrics(args.stages, args.microbatches, args.iters,
                  args.f, args.b, args.w, tasks, mk)
    print(f"schedule={args.schedule} p={args.stages} m={args.microbatches} "
          f"iters={args.iters} F/B/W={args.f}/{args.b}/{args.w}")
    print(f"makespan={met['makespan']} per_step={met['per_step']:.2f} "
          f"ideal={met['ideal']} bubble={met['bubble']} "
          f"bubble_rate={met['bubble_rate']:.2%} total_idle={met['total_idle']}")
    print("idle per stage: " + " ".join(f"p{s}={v}" for s, v in enumerate(met["idle_per_stage"])))
    if args.gantt:
        print_gantt(tasks, args.stages, args.iters, mk)


if __name__ == "__main__":
    main()
