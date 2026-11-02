#!/usr/bin/env python3
"""从 trace 里算 overlap 账（CPU only）。

输入 CSV，四列：name,stream,type,start,end
  type ∈ {compute, comm, other}
时间单位任意（ns/us/ms 均可，只要一致）。

输出：
  - compute / comm 的总时长（区间并集，去重）
  - overlap 时长与 overlap ratio
  - comm 暴露时长
  - 每条 stream 的忙时间
  - 时间轴 ASCII（C=compute, M=comm, B=both, .=idle）

这就是文章里说的"不要只看 nsys 截图，要落成三个数"：
    overlap_ratio = overlap / comm
    exposed       = comm - overlap
    compute_share = compute / span

运行：
    python overlap_audit.py --demo
    python overlap_audit.py --csv trace.csv
    python overlap_audit.py --test
"""
from __future__ import annotations

import argparse
import csv
import io


Interval = tuple[float, float]


def merge(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    xs = sorted(intervals)
    out = [list(xs[0])]
    for a, b in xs[1:]:
        if a <= out[-1][1] + 1e-12:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total_length(intervals: list[Interval]) -> float:
    return sum(b - a for a, b in merge(intervals))


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    a, b = merge(a), merge(b)
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def audit(rows: list[dict]) -> dict:
    comp = [(float(r["start"]), float(r["end"])) for r in rows if r["type"] == "compute"]
    comm = [(float(r["start"]), float(r["end"])) for r in rows if r["type"] == "comm"]
    all_iv = comp + comm
    span = (max(b for _, b in all_iv) - min(a for a, _ in all_iv)) if all_iv else 0.0
    c_len, m_len = total_length(comp), total_length(comm)
    ov = total_length(intersect(comp, comm))
    streams: dict[str, float] = {}
    for r in rows:
        streams[r["stream"]] = streams.get(r["stream"], 0.0) + float(r["end"]) - float(r["start"])
    return {
        "compute": c_len, "comm": m_len, "overlap": ov,
        "exposed": max(0.0, m_len - ov),
        "overlap_ratio": ov / m_len if m_len else 1.0,
        "compute_share": c_len / span if span else 0.0,
        "span": span,
        "streams": streams,
    }


def print_timeline(rows: list[dict], span: float, width: int = 80) -> None:
    if span <= 0:
        return
    cells = ["."] * width
    for r in rows:
        a = int(float(r["start"]) / span * (width - 1))
        b = max(a + 1, int(float(r["end"]) / span * (width - 1)))
        for x in range(a, min(width, b)):
            old = cells[x]
            new = "C" if r["type"] == "compute" else ("M" if r["type"] == "comm" else "o")
            if old == ".":
                cells[x] = new
            elif {old, new} == {"C", "M"}:
                cells[x] = "B"
    print("timeline: " + "".join(cells))
    print("          " + "C=compute  M=comm  B=both  .=idle")


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "name": r["name"], "stream": r["stream"], "type": r["type"],
                "start": float(r["start"]), "end": float(r["end"]),
            })
    return rows


DEMO = """name,stream,type,start,end
qkv_gemm,compute,compute,0,40
allreduce_qkv,comm,comm,30,55
attn,compute,compute,40,80
allreduce_attn,comm,comm,70,95
mlp_gemm,compute,compute,80,150
allreduce_mlp,comm,comm,120,160
"""


def _self_test() -> None:
    rows = [{"name": "c", "stream": "s1", "type": "compute", "start": 0, "end": 10},
            {"name": "m", "stream": "s2", "type": "comm", "start": 5, "end": 15}]
    a = audit(rows)
    assert a["compute"] == 10 and a["comm"] == 10
    assert a["overlap"] == 5 and a["exposed"] == 5
    assert abs(a["overlap_ratio"] - 0.5) < 1e-9

    rows = [{"name": "c1", "stream": "s1", "type": "compute", "start": 0, "end": 10},
            {"name": "c2", "stream": "s1", "type": "compute", "start": 20, "end": 30},
            {"name": "m", "stream": "s2", "type": "comm", "start": 5, "end": 25}]
    a = audit(rows)
    assert a["compute"] == 20 and a["comm"] == 20
    assert a["overlap"] == 10 and a["exposed"] == 10

    # 完全覆盖
    rows = [{"name": "c", "stream": "s1", "type": "compute", "start": 0, "end": 100},
            {"name": "m", "stream": "s2", "type": "comm", "start": 10, "end": 90}]
    a = audit(rows)
    assert a["exposed"] == 0 and a["overlap_ratio"] == 1.0

    # demo CSV 能解析
    rows = list(csv.DictReader(io.StringIO(DEMO)))
    assert len(rows) == 6
    print("overlap_audit: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
        return
    if args.demo:
        rows = [{k: (float(v) if k in ("start", "end") else v)
                 for k, v in r.items()} for r in csv.DictReader(io.StringIO(DEMO))]
    elif args.csv:
        rows = load_csv(args.csv)
    else:
        ap.print_help()
        return
    a = audit(rows)
    print(f"span={a['span']:.1f} compute={a['compute']:.1f} comm={a['comm']:.1f}")
    print(f"overlap={a['overlap']:.1f} exposed={a['exposed']:.1f} "
          f"overlap_ratio={a['overlap_ratio']:.1%} compute_share={a['compute_share']:.1%}")
    print("streams: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(a["streams"].items())))
    print_timeline(rows, a["span"])


if __name__ == "__main__":
    main()
