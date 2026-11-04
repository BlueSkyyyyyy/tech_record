#!/usr/bin/env python3
"""两级 overlap 的时间线账本：串行 vs event_loop_overlap vs TBO。

对应专题第 5 篇（两级 overlap）。模型只保留三条时间轴（采样单独占一条）：

    CPU    调度一步：取请求、组 batch、填 seq_lens、提交 H2D      cpu_ms
    GPU    一步 forward（权重带宽受限）                           gpu_ms
    采样   等上一步的 logits 落地、填 grammar mask、采一个 token    sample_ms

三种模式：

    serial   CPU → GPU → 采样，严格串行；一步 = cpu + gpu + sample
    overlap  SGLang 的 event_loop_overlap（managers/scheduler.py:2143）：
             发 forward N 的同时调度第 N+1 批，第 N-1 步的结果延后一格处理
             （result_queue 深度恒 ≤ 2）。一步 ≈ max(cpu + sample, gpu)
    tbo      在 overlap 之上把一条 batch 切成两个 micro-batch，通信与算力交错
             （enable_two_batch_overlap，默认 False）。通信 M 有一半被另一半的
             micro-batch 算力盖住，代价是多一次 kernel 启动 launch_ms

时间线是**算出来的**：每个任务带依赖（GPU 等 CPU、采样等 GPU、CPU 的下一轮
等上一步结果），按资源串起来后取总墙钟。没有 GPU，毫秒数只是量级假设，不是实测。

用法：
    python3 overlap_sim.py                       # 三种模式的账本 + 前 4 步时间线
    python3 overlap_sim.py --comm-frac 0         # 关掉通信，看 TBO 还剩什么
    python3 overlap_sim.py --cpu-ms 3 --gpu-ms 45   # 671B / EP16 的量级
    python3 overlap_sim.py --test
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

LANES = ("CPU", "GPU", "SMP")   # SMP = 采样（sampler），用 ASCII 保证时间线对齐


# ---------------------------------------------------------------------------
# 时间线的基本单位
# ---------------------------------------------------------------------------
@dataclass
class Seg:
    """一段时间线上的一段：忙（C/G/m/S）或空等（W/c/s）。"""

    lane: str
    start: float
    end: float
    mark: str


@dataclass
class Timeline:
    mode: str
    steps: int
    total_ms: float
    gpu_ms: float                     # 每步 GPU 占用的墙钟（含通信与 launch）
    cpu_busy: float
    gpu_busy: float
    smp_busy: float
    segs: list[Seg] = field(default_factory=list)

    @property
    def step_ms(self) -> float:
        return self.total_ms / self.steps

    @property
    def cpu_idle(self) -> float:
        return self.total_ms - self.cpu_busy

    @property
    def gpu_idle(self) -> float:
        return self.total_ms - self.gpu_busy

    @property
    def smp_idle(self) -> float:
        return self.total_ms - self.smp_busy


def _waits(busy: list[Seg], end: float, mark: str, tail: bool = False) -> list[Seg]:
    """把某条轴上的空隙填成「等」，空隙的来源由调用方指定。

    tail=False 不补最后一段之后的部分（比如串行里最后一步采样时 GPU 只是闲着，
    不是「等 CPU」）。
    """
    out: list[Seg] = []
    t = 0.0
    for s in sorted(busy, key=lambda x: x.start):
        if s.start > t + 1e-12:
            out.append(Seg(s.lane, t, s.start, mark))
        t = max(t, s.end)
    if tail and end > t + 1e-12:
        out.append(Seg(busy[0].lane, t, end, mark))
    return out


# ---------------------------------------------------------------------------
# 三种模式的依赖图
# ---------------------------------------------------------------------------
def build(
    mode: str,
    steps: int,
    cpu: float,
    gpu: float,
    smp: float,
    comm_frac: float,
    launch: float,
) -> Timeline:
    """按依赖关系排出时间线。

    依赖只有四条：
        gpu_i   等 cpu_i（同一个 batch 得先组出来），且等 gpu_{i-1}（单流）
        smp_i   等 gpu_i（要 logits）
        cpu_i+1 等 cpu_i（CPU 只有一个），且等 smp_{i-1}（overlap 下先处理上一步结果）
        serial 里 CPU 还额外等 smp_i，所以退化成严格串行
    """
    if steps <= 0:
        raise ValueError("steps 必须为正")
    if not 0.0 <= comm_frac < 1.0:
        raise ValueError("comm_frac 必须在 [0, 1) 内")
    if min(cpu, gpu, smp) < 0 or launch < 0:
        raise ValueError("时间参数不能为负")

    comm = gpu * comm_frac
    compute = gpu - comm
    busy: list[Seg] = []

    if mode == "serial":
        step = cpu + gpu + smp
        total = steps * step
        for i in range(steps):
            t = i * step
            busy += [
                Seg("CPU", t, t + cpu, "C"),
                Seg("GPU", t + cpu, t + cpu + gpu, "G"),
                Seg("SMP", t + cpu + gpu, t + step, "S"),
            ]
        # 串行：CPU 等 GPU，GPU 等 CPU，采样等 GPU
        segs = _waits([s for s in busy if s.lane == "CPU"], total, "W", tail=True)
        segs += _waits([s for s in busy if s.lane == "GPU"], total, "c")
        segs += _waits([s for s in busy if s.lane == "SMP"], total, "s", tail=True)
        return Timeline("serial", steps, total, gpu, steps * cpu, steps * gpu,
                        steps * smp, segs + busy)

    if mode == "overlap":
        g = gpu
    elif mode == "tbo":
        # 通信一半被另一个 micro-batch 的算力盖住，剩下一半在尾巴上；
        # 多切一刀，多一次 kernel 启动。
        g = compute + comm / 2 + launch
    else:
        raise ValueError(f"未知模式: {mode}")

    total = cpu + steps * g + smp
    cpu_busy: list[Seg] = [Seg("CPU", 0.0, cpu, "C")]
    gpu_busy: list[Seg] = []
    smp_busy: list[Seg] = []

    for i in range(steps):
        s = cpu + i * g
        if mode == "overlap":
            gpu_busy.append(Seg("GPU", s, s + g, "G"))
        else:
            c2, m2 = compute / 2, comm / 2
            # 先画通信、再画算力：重叠的格子由算力覆盖，剩下的 m 就是没藏住的部分
            gpu_busy += [
                Seg("GPU", s + c2, s + c2 + m2, "m"),
                Seg("GPU", s + compute + launch, s + g, "m"),
                Seg("GPU", s, s + c2, "G"),
                Seg("GPU", s + c2 + launch, s + compute + launch, "G"),
            ]
        smp_busy.append(Seg("SMP", s + g, s + g + smp, "S"))
        if i + 1 < steps:
            # 下一轮 CPU：先跑自己的调度，但第 i-1 步的结果还没落地时采不了样
            wait_to = smp_busy[i - 1].end if i >= 1 else 0.0
            start = max(cpu_busy[-1].end, wait_to)
            cpu_busy.append(Seg("CPU", start, start + cpu, "C"))

    segs = _waits(cpu_busy, total, "W", tail=True)
    segs.append(Seg("GPU", 0.0, cpu, "c"))      # 第一步的 GPU 等 CPU
    segs += _waits(smp_busy, total, "s", tail=True)
    return Timeline(mode, steps, total, g, steps * cpu, steps * g, steps * smp,
                    segs + gpu_busy + smp_busy + cpu_busy)


# ---------------------------------------------------------------------------
# ASCII 时间线
# ---------------------------------------------------------------------------
def _ruler(cols: int) -> str:
    line = [" "] * cols
    for c in range(0, cols, 10):
        for k, ch in enumerate(str(c)):
            if c + k < cols:
                line[c + k] = ch
    return "".join(line)


def raster(segs: list[Seg], horizon: float, width: float = 1.0) -> dict[str, str]:
    """一格 = width 毫秒；取「占这一格面积最大」的段，这样 0.4 ms 的采样不会凭空消失。

    面积相同的格子由后画的段赢：调用方把「等」放在前面、忙放在后面。
    """
    cols = max(1, int(math.ceil(horizon / width - 1e-9)))
    rows = {lane: ["."] * cols for lane in LANES}
    best = {lane: [0.0] * cols for lane in LANES}
    for s in segs:
        if s.lane not in rows:
            continue
        a = max(0, int(s.start // width))
        b = min(cols, int(math.ceil(min(s.end, horizon) / width)))
        for c in range(a, b):
            lo, hi = c * width, min((c + 1) * width, horizon)
            cov = max(0.0, min(s.end, hi) - max(s.start, lo))
            # 采样只有 0.4 ms，按面积算经常输给隔壁的 G；它一出现就画出来，
            # 否则这张图会让人误以为「采样不占时间」。
            if cov > 0 and (s.mark == "S" or cov >= best[s.lane][c] - 1e-9):
                best[s.lane][c] = cov
                rows[s.lane][c] = s.mark
    return {k: "".join(v) for k, v in rows.items()}


def timeline_text(tl: Timeline, n: int, launch: float) -> list[str]:
    """从完整时间线里截前 n 步画 ASCII（节拍与账本一致，不重新热身）。"""
    horizon = min(tl.total_ms, n * tl.step_ms)
    rows = raster(tl.segs, horizon)
    out = [f"  {tl.mode} —— 一步 {tl.step_ms:.3f} ms，前 {n} 步共 {horizon:.1f} ms"]
    out.append(f"    {'t/ms':<5}{_ruler(len(rows['CPU']))}")
    for lane in LANES:
        out.append(f"    {lane:<5}{rows[lane]}")
    if tl.mode == "tbo":
        out.append(
            f"    （G = 算力、m = 通信；1 格 1 ms，{launch} ms 的 launch 比一格还窄，图上被取整吸收）"
        )
    return out


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------
def report(args):
    params = dict(steps=args.steps, cpu=args.cpu_ms, gpu=args.gpu_ms,
                  smp=args.sample_ms, comm_frac=args.comm_frac, launch=args.launch_ms)
    tls = [build(m, **params) for m in ("serial", "overlap", "tbo")]
    base = tls[0]
    comm = args.gpu_ms * args.comm_frac
    print(f"一步的组成：CPU {args.cpu_ms} ms + GPU {args.gpu_ms} ms + 采样 {args.sample_ms} ms；"
          f"GPU 里通信 {comm:.2f} ms（{args.comm_frac:.0%}）\n")

    hdr = (f"{'模式':<9}{'每步 ms':>9}{'总 ms':>11}{'吞吐比':>8}"
           f"{'CPU 忙':>8}{'GPU 忙':>8}{'采样忙':>7}{'CPU 空等':>9}")
    print(hdr)
    print("-" * (len(hdr) + 4))
    for tl in tls:
        print(f"{tl.mode:<9}{tl.step_ms:>9.3f}{tl.total_ms:>11.1f}"
              f"{base.total_ms / tl.total_ms:>8.2f}"
              f"{tl.cpu_busy:>8.0f}{tl.gpu_busy:>8.0f}{tl.smp_busy:>7.0f}{tl.cpu_idle:>9.0f}")

    print("\n前 4 步的时间线（1 格 = 1 ms；采样只有 0.4 ms，跨格时相邻两格都会标出来）")
    print("图例：C CPU 忙 · W CPU 等 GPU 结果 · G GPU 算 · m GPU 通信（没被算力盖住的部分）"
          " · S 采样忙 · s 采样等 GPU · c GPU 等 CPU · . 空闲")
    o, t = tls[1], tls[2]
    for tl in tls:
        print()
        print("\n".join(timeline_text(tl, 4, args.launch_ms)))
        if tl.mode == "serial":
            print(f"      谁在等谁：CPU 空等 {tl.cpu_idle:.0f} ms（每一步都等 forward + 采样）"
                  f"；GPU 空等 {tl.gpu_idle:.0f} ms（等 CPU 组 batch）；"
                  f"采样空等 {tl.smp_idle:.0f} ms")
        elif tl.mode == "overlap":
            print(f"      谁在等谁：CPU 空等 {tl.cpu_idle:.0f} ms（等第 N-1 步结果落地才能采样）"
                  f"；GPU 只在首尾空等 {tl.gpu_idle:.1f} ms；"
                  f"采样空等 {tl.smp_idle:.0f} ms（没有结果可采）")
        else:
            print(f"      谁在等谁：通信 {comm:.2f} ms/步，被另一半的算力盖住 {comm / 2:.2f} ms，"
                  f"净赚 {comm / 2 - args.launch_ms:.2f} ms（= M/2 - launch）；"
                  f"GPU 每步 {o.gpu_ms:.2f} → {tl.gpu_ms:.2f} ms")

    cpu_plus_smp = args.cpu_ms + args.sample_ms
    print(
        "\n读法：\n"
        "  · serial 的一步是 cpu + gpu + sample，三者谁也不能和谁重叠；overlap 只把\n"
        f"    cpu + sample（{cpu_plus_smp:.1f} ms）塞进 gpu 的 {args.gpu_ms:g} ms 窗口，"
        f"所以一步是 {o.step_ms:.2f} ms——省下的\n"
        "    恰好是「CPU/GPU 时间比」，模型越小、batch 越小越划算，大 batch 高端几乎不值钱。\n"
        "  · overlap 图里 CPU 行大段 W：它不是没事干，而是在等第 N-1 步的结果落地才能采样；\n"
        "    这正是 result_queue 与 last_batch 存在的理由（结果延后一格）。\n"
        "  · TBO 不动 CPU，只在 GPU 内部把通信劈成两半：第二个 micro-batch 的 GEMM 去盖\n"
        f"    第一个的 a2a。省的是 M/2 - launch：通信占比 {args.comm_frac:.0%} 时净赚 "
        f"{comm / 2 - args.launch_ms:.2f} ms/步，"
        f"整步 {o.step_ms:.2f} → {t.step_ms:.2f} ms（快 {(o.total_ms / t.total_ms - 1):.1%}）。\n"
        "  · 把 --comm-frac 调到 0，TBO 就没有可藏的通信，只剩 "
        f"{args.launch_ms:g} ms 的额外启动开销，\n"
        "    此时它比 overlap 更慢——重叠本身不创造时间，它只是把等待挪到别处。"
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _test():
    p = dict(steps=1000, cpu=1.2, gpu=8.0, smp=0.4, launch=0.3)
    s = build("serial", comm_frac=0.18, **p)
    o = build("overlap", comm_frac=0.18, **p)
    t = build("tbo", comm_frac=0.18, **p)

    # 1) 串行 > overlap > tbo
    assert s.total_ms > o.total_ms > t.total_ms, (s.total_ms, o.total_ms, t.total_ms)

    # 2) overlap 的吞吐比落在 1.1–1.4（cpu+sample 藏进 gpu 窗口）
    ratio = s.total_ms / o.total_ms
    assert 1.1 <= ratio <= 1.4, ratio
    assert abs(o.step_ms - 8.0) < 0.01, o.step_ms

    # 3) tbo 在 overlap 上再快 3–12%
    gain = o.total_ms / t.total_ms - 1.0
    assert 0.03 <= gain <= 0.12, gain

    # 4) comm_frac=0：没有通信可藏，TBO 只剩启动开销，不能更快
    o0 = build("overlap", steps=1000, cpu=1.2, gpu=8.0, smp=0.4, comm_frac=0.0, launch=0.3)
    t0 = build("tbo", steps=1000, cpu=1.2, gpu=8.0, smp=0.4, comm_frac=0.0, launch=0.3)
    assert t0.total_ms >= o0.total_ms, (t0.total_ms, o0.total_ms)
    assert abs((t0.total_ms - o0.total_ms) - 0.3 * 1000) < 1e-6, t0.total_ms - o0.total_ms

    # 5) 通信占比越高，TBO 省得越多（在可藏范围内）
    t10 = build("tbo", steps=1000, cpu=1.2, gpu=8.0, smp=0.4, comm_frac=0.10, launch=0.3)
    t30 = build("tbo", steps=1000, cpu=1.2, gpu=8.0, smp=0.4, comm_frac=0.30, launch=0.3)
    assert t30.step_ms < t.step_ms < t10.step_ms, (t30.step_ms, t.step_ms, t10.step_ms)

    # 6) 确定性：同样参数两次，时间线与分段逐项相等
    o2 = build("overlap", comm_frac=0.18, **p)
    assert o2.total_ms == o.total_ms
    assert [(x.lane, x.start, x.end, x.mark) for x in o2.segs] == \
           [(x.lane, x.start, x.end, x.mark) for x in o.segs]

    print("overlap_sim: all tests passed")
    print(f"  serial {s.total_ms:.1f} ms · overlap {o.total_ms:.1f} ms（{ratio:.2f}x）"
          f" · tbo {t.total_ms:.1f} ms（{s.total_ms / t.total_ms:.2f}x，比 overlap 快 {gain:.1%}）")
    print(f"  comm_frac=0：overlap {o0.total_ms:.1f} ms，tbo {t0.total_ms:.1f} ms"
          f"（多出的 {t0.total_ms - o0.total_ms:.0f} ms = 1000 步 × {0.3} ms launch）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--cpu-ms", type=float, default=1.2, help="每步 CPU 调度时间")
    ap.add_argument("--gpu-ms", type=float, default=8.0, help="一步 forward 的 GPU 时间")
    ap.add_argument("--sample-ms", type=float, default=0.4, help="采样时间")
    ap.add_argument("--comm-frac", type=float, default=0.18, help="GPU 时间里通信占比")
    ap.add_argument("--launch-ms", type=float, default=0.3, help="TBO 多出的一次 kernel 启动")
    args = ap.parse_args()
    if args.test:
        _test()
        return
    report(args)


if __name__ == "__main__":
    main()
