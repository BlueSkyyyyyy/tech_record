#!/usr/bin/env python3
"""SM 分区模型：通信该拿多少 SM？（CPU only）

把一层/一个 iteration 看成两份工作：
- 计算工作量 W_c（单位：SM·us），只用分到的 S_c 个 SM；
- 通信工作量 W_m（单位：SM·us），只用分到的 S_m 个 SM，且带宽在 s0 个 SM 后饱和。

在"通信与计算能完全并发"的理想化假设下：
    T_compute = W_c / S_c
    T_comm    = W_m / min(S_m, s0)
    T         = max(T_compute, T_comm)
真实系统还要加 wave 量化和启动开销；这里用 `wave_unit` 做一个粗粒度惩罚。

运行：
    python sm_split_model.py
    python sm_split_model.py --test
"""
from __future__ import annotations

import argparse
import math


def times(total_sms: int, w_compute: float, w_comm: float, s_comm: int,
          sm_sat: int, wave_unit: float = 0.0) -> tuple[float, float, float]:
    """返回 (compute_time, comm_time, makespan)。"""
    s_comp = total_sms - s_comm
    if s_comp <= 0:
        tc = float("inf")
    else:
        raw = w_compute / s_comp
        if wave_unit > 0:
            # 以 wave_unit 为粒度向上取整；模拟"尾 wave 装不满"
            tc = math.ceil(raw / wave_unit) * wave_unit
        else:
            tc = raw
    if s_comm <= 0:
        tm = float("inf")
    else:
        tm = w_comm / min(s_comm, sm_sat)
    return tc, tm, max(tc, tm)


def sweep(total_sms: int, w_compute: float, w_comm: float, sm_sat: int,
          wave_unit: float = 0.0):
    rows = []
    for s in range(0, total_sms + 1):
        tc, tm, mk = times(total_sms, w_compute, w_comm, s, sm_sat, wave_unit)
        rows.append({"s": s, "compute": tc, "comm": tm, "makespan": mk})
    return rows


def _self_test() -> None:
    S, Wc, Wm, s0 = 132, 10000.0, 3000.0, 16
    rows = sweep(S, Wc, Wm, s0)
    best = min(rows, key=lambda r: r["makespan"])
    assert 0 < best["s"] < S, best
    assert math.isinf(rows[0]["makespan"])       # 不给通信 SM
    assert math.isinf(rows[-1]["makespan"])      # 不给计算 SM
    # 最优一定不差于"全串行"（把通信和计算都压到全部 SM 上顺序执行）
    serial = Wc / S + Wm / min(S, s0)
    assert best["makespan"] <= serial + 1e-9

    # wave 量化会把最优往"给计算更多 SM"的方向推，并抬高 makespan
    rows_w = sweep(S, Wc, Wm, s0, wave_unit=50.0)
    best_w = min(rows_w, key=lambda r: r["makespan"])
    assert best_w["makespan"] >= best["makespan"] - 1e-9

    print("sm_split_model: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-sms", type=int, default=132)
    ap.add_argument("--compute-work", type=float, default=10000.0)
    ap.add_argument("--comm-work", type=float, default=3000.0)
    ap.add_argument("--sm-sat", type=int, default=16)
    ap.add_argument("--wave-unit", type=float, default=50.0)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
        return
    rows = sweep(args.total_sms, args.compute_work, args.comm_work,
                 args.sm_sat, args.wave_unit)
    best = min(rows, key=lambda r: r["makespan"])
    print(f"S={args.total_sms} Wc={args.compute_work} Wm={args.comm_work} "
          f"sat={args.sm_sat} wave={args.wave_unit}")
    print(f"{'s_comm':>7} {'T_comp':>9} {'T_comm':>9} {'T':>9}")
    for r in rows:
        if r["s"] % 4 == 0 or r["s"] == best["s"]:
            print(f"{r['s']:>7} {r['compute']:>9.1f} {r['comm']:>9.1f} {r['makespan']:>9.1f}")
    print(f"best: s_comm={best['s']} makespan={best['makespan']:.1f} "
          f"(serial={args.compute_work/args.total_sms + args.comm_work/min(args.total_sms, args.sm_sat):.1f})")


if __name__ == "__main__":
    main()
