#!/usr/bin/env python3
"""推理侧 token 切分的收益阈值模型（CPU only）。

TokenWeave 的核心观察：把 token batch 切成两半、双流重叠 AllReduce 与
attention/FFN，只有在 token 数足够大时才划算；小 batch 下切分本身会
增加 wave 数，收益被抵消。

模型：
    c(T) = max(ceil(T*w/S / wave_unit)*wave_unit, wave_unit)
    m(T) = T*v / BW
    unsplit = c(T) + m(T)
    split   = max(2*c(T/2), 2*m(T/2)) + min(c(T/2), m(T/2))

这是两级流水的理想化模型，不含 kernel launch、RMSNorm 融合等细节，
只用来解释"阈值"从哪来。
"""
from __future__ import annotations

import argparse
import math


def compute_time(tokens: float, work_per_token: float, sms: int,
                 wave_unit: float) -> float:
    if tokens <= 0:
        return 0.0
    raw = tokens * work_per_token / sms
    return max(wave_unit, math.ceil(raw / wave_unit) * wave_unit)


def comm_time(tokens: float, bytes_per_token: float, bw_gbps: float) -> float:
    if tokens <= 0:
        return 0.0
    return tokens * bytes_per_token / (bw_gbps * 1e9) * 1e6


def times(tokens: float, work_per_token: float, bytes_per_token: float,
          sms: int, wave_unit: float, bw_gbps: float) -> tuple[float, float, float]:
    c = compute_time(tokens, work_per_token, sms, wave_unit)
    m = comm_time(tokens, bytes_per_token, bw_gbps)
    unsplit = c + m
    ch = compute_time(tokens / 2, work_per_token, sms, wave_unit)
    mh = comm_time(tokens / 2, bytes_per_token, bw_gbps)
    split = max(2 * ch, 2 * mh) + min(ch, mh)
    return unsplit, split, split / unsplit if unsplit else 1.0


def find_threshold(work_per_token: float, bytes_per_token: float, sms: int,
                   wave_unit: float, bw_gbps: float, tmax: int = 1 << 20) -> int | None:
    t = 1
    while t <= tmax:
        unsplit, split, _ = times(t, work_per_token, bytes_per_token, sms,
                                  wave_unit, bw_gbps)
        if split < unsplit:
            return t
        t *= 2
    return None


def _self_test() -> None:
    kw = dict(work_per_token=0.5, bytes_per_token=8192, sms=132,
              wave_unit=10.0, bw_gbps=400.0)
    u_small, s_small, _ = times(128, **kw)
    assert s_small >= u_small, (u_small, s_small)   # 小 batch：切分更慢
    u_large, s_large, _ = times(16384, **kw)
    assert s_large < u_large, (u_large, s_large)    # 大 batch：切分更快
    th = find_threshold(**kw)
    assert th is not None and 128 <= th <= 16384, th

    # 带宽越高，通信越不是瓶颈，阈值应不升高
    th_hi = find_threshold(**{**kw, "bw_gbps": 900.0})
    assert th_hi is not None and th_hi <= th

    print(f"decode_overlap: all tests passed (threshold≈{th} tokens, "
          f"high-bw≈{th_hi})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--work-per-token", type=float, default=0.5)
    ap.add_argument("--bytes-per-token", type=float, default=8192)
    ap.add_argument("--sms", type=int, default=132)
    ap.add_argument("--wave-unit", type=float, default=10.0)
    ap.add_argument("--bw-gbps", type=float, default=400.0)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
        return
    kw = dict(work_per_token=args.work_per_token,
              bytes_per_token=args.bytes_per_token, sms=args.sms,
              wave_unit=args.wave_unit, bw_gbps=args.bw_gbps)
    if args.scan:
        print(f"{'tokens':>8} {'unsplit':>9} {'split':>9} {'ratio':>7}")
        t = 128
        while t <= 65536:
            u, s, r = times(t, **kw)
            print(f"{t:>8} {u:>9.1f} {s:>9.1f} {r:>6.2f}x")
            t *= 2
        th = find_threshold(**kw)
        print(f"threshold≈{th} tokens")
        return
    u, s, r = times(args.tokens, **kw)
    print(f"tokens={args.tokens} unsplit={u:.1f}us split={s:.1f}us ratio={r:.2f}x")


if __name__ == "__main__":
    main()
