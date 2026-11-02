#!/usr/bin/env python3
"""动作 chunk 的延迟账：什么时候异步推理真的能藏住延迟（stdlib）.

对应文章：VLA 精读（八）推理系统。

把机器人控制想成一条 50 Hz 的时钟（dt=20 ms）。策略一次吐 C 步动作（chunk），
推理耗时 L。三个结论都能手算：

1. **同步**：执行完 C 步再推理，每 C 步 stall L。刷新率 = C / (C*dt + L)。
2. **异步（队列式）**：机器人执行当前 chunk 的同时，后台算下一块；算完就排队，
   当前块执行完立刻接上。稳态下每个 chunk 仍然覆盖 C*dt 秒，所以**周期 = max(C*dt, L)**：
   - **C*dt >= L**：不断档，stall = 0（这是"藏住延迟"的门槛）；
   - **C*dt >= 2L**：GPU 占用率不超过 50%，有抖动余量
     （这是"有余量"的门槛）。介于两者之间时，GPU 占用率超过 50%、仍然不断档。
   如果想更早切换（用反应性换 horizon），可以执行 C*dt-L 就切；这需要 C*dt >= 2L，
   且有效 horizon 缩短到 C - L/dt。
3. **异步但不做时间对齐**：新 chunk 一到就从 index 0 开始执行，等于把
   L 毫秒前的动作当成“现在”，轨迹会出现一次向后的跳变，幅度约为 v*L。

第 3 条正是 Real-Time Chunking（RTC）要修的东西：用旧 chunk 未执行完的尾巴
去引导新 chunk 的开头（见 rtc_toy.py）。

运行：

    python chunk_latency.py           # 表格 + 跳变演示
    python chunk_latency.py --test    # 自测
"""
from __future__ import annotations

import argparse
import math


def sync_metrics(chunk: int, latency_s: float, dt: float) -> dict[str, float]:
    """执行完再推理。"""
    exec_s = chunk * dt
    cycle = exec_s + latency_s
    return {
        "cycle_s": cycle,
        "chunks_per_s": 1.0 / cycle,
        "stall_frac": latency_s / cycle,
        "stall_per_chunk_s": latency_s,
        "hidden": 0.0,
    }


def async_metrics(chunk: int, latency_s: float, dt: float) -> dict[str, float]:
    """异步 + 绝对时间对齐。"""
    exec_s = chunk * dt
    # 队列式异步：每个 chunk 覆盖 exec_s 秒，推理在后台并行；只有 L > exec_s 才断档
    cycle = max(exec_s, latency_s)
    stall = max(0.0, latency_s - exec_s)
    return {
        "cycle_s": cycle,
        "chunks_per_s": 1.0 / cycle if cycle > 0 else float("inf"),
        "stall_frac": stall / cycle if cycle > 0 else 0.0,
        "stall_per_chunk_s": stall,
        "hidden": 1.0 if stall == 0.0 else 0.0,
        "slack": 1.0 if exec_s >= 2 * latency_s else 0.0,
    }


def reference(t: float) -> float:
    """一条平滑轨迹，峰值速度约 0.5*2π*0.4 ≈ 1.26 /s。"""
    return 0.5 * math.sin(2 * math.pi * 0.4 * t)


def stale_jump(chunk: int, latency_s: float, dt: float, n_chunks: int = 6) -> dict[str, float]:
    """比较两种 chunk 拼接方式的最大跳变。

    - naive：新 chunk 到达后从 index 0 执行（把 L 秒前的动作当现在）
    - aligned：从 index round(L/dt) 执行（按绝对时间对齐）

    两者都假设策略完美预测未来，所以误差只来自时间对齐。
    """
    naive_jump = 0.0
    aligned_jump = 0.0
    t = 0.0
    for k in range(n_chunks):
        # 第 k 个 chunk 在 t 时刻观测，覆盖 [t, t+C*dt]
        # naive：在 t+L 时从 index 0 开始，动作对应 t+dt
        a_naive = reference(t + dt)
        # 上一拍实际执行到 t+L（若上一 chunk 覆盖得到）
        prev = reference(t + latency_s)
        naive_jump = max(naive_jump, abs(a_naive - prev))
        # aligned：从 index L/dt 开始，动作对应 t+L+dt，与当前时间只差 dt
        a_aligned = reference(t + latency_s + dt)
        aligned_jump = max(aligned_jump, abs(a_aligned - prev))
        t += chunk * dt - latency_s
    return {"naive_jump": naive_jump, "aligned_jump": aligned_jump}


def report() -> None:
    dt = 0.02
    print("== 1. 同步 vs 异步（dt=20ms, 50Hz）==")
    print(f"{'C':>4}{'L(ms)':>7}{'同步刷新Hz':>11}{'同步stall':>10}{'异步刷新Hz':>11}{'异步stall':>10}{'藏住?':>7}{'有余量?':>8}")
    for c in (10, 25, 50):
        for l_ms in (50, 100, 200, 400, 800):
            l = l_ms / 1000
            s = sync_metrics(c, l, dt)
            a = async_metrics(c, l, dt)
            print(
                f"{c:>4}{l_ms:>7}{s['chunks_per_s']:>11.2f}{s['stall_frac']:>9.0%}"
                f"{a['chunks_per_s']:>11.2f}{a['stall_frac']:>9.0%}"
                f"{'yes' if a['hidden'] else 'no':>7}{'yes' if a['slack'] else 'no':>8}"
            )

    print("\n== 2. 两个门槛 ==")
    print(f"{'C':>4}{'chunk(ms)':>10}{'不断档 L<=':>12}{'有余量 L<=':>12}")
    for c in (10, 25, 50):
        print(f"{c:>4}{c*dt*1000:>10.0f}{c*dt*1000:>12.0f}{c*dt/2*1000:>12.0f}")

    print("\n== 3. 不做时间对齐会跳多少（C=50, dt=20ms）==")
    print(f"{'L(ms)':>7}{'naive跳变':>12}{'aligned跳变':>13}")
    for l_ms in (50, 100, 200, 400):
        r = stale_jump(50, l_ms / 1000, dt)
        print(f"{l_ms:>7}{r['naive_jump']:>12.4f}{r['aligned_jump']:>13.4f}")
    print("\n跳变单位是动作值（这里假设动作范围 ±0.5）。")

    print("\n== 4. 结论 ==")
    print("- 同步模式的刷新率被 L 直接卡死；C 越大越接近 1/dt，但反应越慢。")
    print("- 异步不断档的条件是 C*dt >= L；GPU 有余量的条件是 C*dt >= 2L。")
    print("- 介于两者之间时，延迟藏住了，但 GPU 占用率超过 50%，没有抖动余量。")
    print("- 对齐方式比‘异步’本身更重要：不对齐时，L 秒的陈旧动作会让轨迹倒退。")


def _self_test() -> None:
    dt = 0.02
    s = sync_metrics(50, 0.2, dt)
    assert abs(s["cycle_s"] - 1.2) < 1e-9
    assert abs(s["stall_frac"] - 0.2 / 1.2) < 1e-9
    assert abs(s["chunks_per_s"] - 1 / 1.2) < 1e-9

    # L=500ms, C*dt=1000ms: 不断档，周期仍是 1s，且 2L=1000ms 刚好有 50% 余量
    a = async_metrics(50, 0.5, dt)
    assert a["hidden"] == 1.0 and a["stall_per_chunk_s"] == 0.0
    assert a["slack"] == 1.0 and abs(a["cycle_s"] - 1.0) < 1e-9
    # L=700ms: 仍不断档，但 2L=1400ms 没有余量，GPU 占用率 >50%
    a = async_metrics(50, 0.7, dt)
    assert a["hidden"] == 1.0 and a["slack"] == 0.0 and abs(a["cycle_s"] - 1.0) < 1e-9
    # L=1200ms > C*dt=1000ms: 每周期 stall 200ms，周期被拉长到 1.2s
    a = async_metrics(50, 1.2, dt)
    assert a["hidden"] == 0.0 and abs(a["stall_per_chunk_s"] - 0.2) < 1e-9
    assert abs(a["cycle_s"] - 1.2) < 1e-9 and abs(a["chunks_per_s"] - 1 / 1.2) < 1e-9

    r = stale_jump(50, 0.2, dt)
    assert r["naive_jump"] > 5 * r["aligned_jump"] + 1e-9, r
    # 对齐后只剩一拍的时间差
    assert r["aligned_jump"] < 0.05, r
    print("[test] chunk_latency OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        report()
