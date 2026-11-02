#!/usr/bin/env python3
"""MoE 专家并行的通算重叠模型（CPU only）。

一个 MoE 层在 EP 下的三段：
    dispatch all-to-all  ->  expert GEMM  ->  combine all-to-all

本模型把每卡的 token 切成 C 个 chunk，dispatch/combine 走同一条通信通道，
GEMM 走计算通道；再用 `--comm-sms` 指定分给通信的 SM 数，观察
"通信 SM 太少 -> 通信成瓶颈；太多 -> GEMM 被拖慢"的经典凹陷。

注意：这是**解析模型**，不是实测。参数（带宽、峰值算力、SM 饱和点）
需要按目标机器替换；系列文章里引用的实测数字来自论文，不来自这里。

运行：
    python moe_overlap.py                 # 默认扫描
    python moe_overlap.py --chunks 8 --comm-sms 20
    python moe_overlap.py --test
"""
from __future__ import annotations

import argparse
import math


def comm_time(volume_bytes: float, bw_gbps: float, sms: int, sm_sat: int) -> float:
    """通信时间（us）。bw_gbps 是单向有效带宽；sms 为分到的 SM 数。"""
    eff = min(1.0, sms / max(1, sm_sat))
    if eff <= 0:
        return float("inf")
    return volume_bytes / (bw_gbps * 1e9 * eff) * 1e6


def gemm_time(flops: float, tflops: float, sms: int, total_sms: int) -> float:
    """GEMM 时间（us），线性 scaling + 一个 wave 量化惩罚。"""
    if sms <= 0:
        return float("inf")
    eff = sms / total_sms
    # 以 1 个 wave 为粒度向上取整；这里用 tile 数近似 wave 数
    return flops / (tflops * 1e12 * eff) * 1e6


def simulate(tokens: int, hidden: int, topk: int, inter: int, ep: int,
             chunks: int, comm_sms: int, total_sms: int,
             bw_gbps: float, tflops: float, sm_sat: int,
             skew: float = 1.0) -> dict:
    """返回 makespan / 各段总量 / 暴露时间。"""
    if chunks < 1 or ep < 1:
        raise ValueError("chunks and ep must be >= 1")
    if comm_sms <= 0 or comm_sms >= total_sms:
        return {"makespan": float("inf"), "compute_total": float("inf"),
                "comm_total": float("inf"), "exposed": float("inf"),
                "overlap": 0.0, "d": float("inf"), "c": float("inf"),
                "b": float("inf")}
    tokens_per_rank = tokens / ep
    # 每卡要处理 tokens*K/ep 个 expert-token；dispatch/combine 各传一半
    local_rows = tokens_per_rank * topk
    bytes_per_row = hidden * 2  # bf16
    dispatch_bytes = local_rows * bytes_per_row
    combine_bytes = dispatch_bytes
    flops = local_rows * (6 * hidden * inter)  # gate+up+down 的粗估

    d = comm_time(dispatch_bytes / chunks, bw_gbps, comm_sms, sm_sat)
    b = comm_time(combine_bytes / chunks, bw_gbps, comm_sms, sm_sat)
    c = gemm_time(flops / chunks, tflops, total_sms - comm_sms, total_sms)

    # 带 skew 的 chunk 计算时间：hot chunk 更慢
    cs = [c * (skew if i == 0 else 1.0) for i in range(chunks)]
    ds = [d] * chunks
    bs = [b] * chunks

    # 两台机器（comm / compute）的 list schedule，D_i -> C_i -> B_i
    done: dict[str, float] = {}
    busy = {"comm": None, "compute": None}
    t = 0.0
    pending = {}
    for i in range(chunks):
        pending[f"D{i}"] = (ds[i], ())
        pending[f"C{i}"] = (cs[i], (f"D{i}",))
        pending[f"B{i}"] = (bs[i], (f"C{i}",))

    while pending or busy["comm"] or busy["compute"]:
        for mach in ("comm", "compute"):
            if busy[mach] is not None:
                continue
            is_comm = mach == "comm"
            ready = [
                (name, dur) for name, (dur, deps) in pending.items()
                if (name[0] in "DB") == is_comm and all(x in done for x in deps)
            ]
            if ready:
                # D 优先于 B
                name, dur = sorted(ready, key=lambda x: (x[0][0] != "D", x[0]))[0]
                busy[mach] = (name, dur, t + dur)
        if not busy["comm"] and not busy["compute"]:
            cand = [done[x] for _, (_, deps) in pending.items() for x in deps if x in done]
            if not cand:
                raise RuntimeError("deadlock")
            t = min(cand)
            continue
        t = min(v[2] for v in busy.values() if v is not None)
        for mach in ("comm", "compute"):
            if busy[mach] is not None and abs(busy[mach][2] - t) < 1e-12:
                name, _, _ = busy[mach]
                done[name] = t
                pending.pop(name, None)
                busy[mach] = None

    compute_total = sum(cs)
    comm_total = sum(ds) + sum(bs)
    return {
        "makespan": t,
        "compute_total": compute_total,
        "comm_total": comm_total,
        "exposed": t - compute_total,
        "overlap": 1 - max(0.0, t - compute_total) / comm_total if comm_total else 1.0,
        "d": d, "c": c, "b": b,
    }


def _defaults():
    return dict(tokens=16384, hidden=4096, topk=2, inter=14336, ep=8,
                chunks=8, comm_sms=20, total_sms=132,
                bw_gbps=400.0, tflops=700.0, sm_sat=16, skew=1.0)


def _self_test() -> None:
    base = _defaults()
    r1 = simulate(**{**base, "chunks": 1})
    assert abs(r1["exposed"] - (r1["d"] + r1["b"])) < 1e-6
    r8 = simulate(**{**base, "chunks": 8})
    assert r8["makespan"] <= r1["makespan"] + 1e-6

    # comm_sms=0 -> 不可行（inf）
    r0 = simulate(**{**base, "comm_sms": 0})
    assert math.isinf(r0["makespan"]) or r0["makespan"] > 1e9

    # 全给通信 -> GEMM 变慢；全给计算 -> 通信变慢；中间应有最优
    rows = []
    for s in [2, 4, 8, 12, 20, 32, 48, 64]:
        r = simulate(**{**base, "comm_sms": s})
        rows.append((s, r["makespan"]))
    best = min(rows, key=lambda x: x[1])
    assert 2 < best[0] < 64, best

    # skew 让 makespan 变差
    rskew = simulate(**{**base, "skew": 1.5})
    assert rskew["makespan"] >= r8["makespan"] - 1e-6

    print("moe_overlap: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = _defaults()
    for k, v in d.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--sweep", action="store_true", help="sweep comm SMs and chunks")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
        return
    kw = {k: getattr(args, k) for k in d}
    if args.sweep:
        print(f"{'comm_sms':>8} " + " ".join(f"C={c:<6}" for c in [1, 2, 4, 8, 16]))
        for s in [4, 8, 12, 16, 20, 24, 32, 48, 64]:
            vals = []
            for c in [1, 2, 4, 8, 16]:
                r = simulate(**{**kw, "comm_sms": s, "chunks": c})
                vals.append(f"{r['makespan']:7.1f}")
            print(f"{s:>8} " + " ".join(vals))
        return
    r = simulate(**kw)
    print(f"chunks={kw['chunks']} comm_sms={kw['comm_sms']}/{kw['total_sms']} "
          f"ep={kw['ep']} tokens={kw['tokens']}")
    print(f"d={r['d']:.2f}us c={r['c']:.2f}us b={r['b']:.2f}us "
          f"makespan={r['makespan']:.1f}us exposed={r['exposed']:.1f}us "
          f"overlap={r['overlap']:.1%}")


if __name__ == "__main__":
    main()
