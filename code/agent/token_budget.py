#!/usr/bin/env python3
"""上下文预算与截断策略：为什么"窗口没用满"也会丢证据。

对应专题第 4 篇《Context Engineering：预算、压缩与信息衰减》。

用法：
    python3 token_budget.py                 # 默认场景
    python3 token_budget.py --test          # 自测
    python3 token_budget.py --runs 720 --budget 2000

只依赖标准库。它复现的是 harness 里最常见的一段代码：

    tool_result = json.dumps(events)
    if len(tool_result) > budget:
        tool_result = tool_result[:budget]      # ← 一刀切

这段代码在两个层面出错：
1. JSON 被切成非法文本，模型没法解析；
2. 被切掉的是"尾部"，而尾部往往正是最新、最相关的事件。

脚本把三种策略放在同一批合成轨迹上比较：固定截断、中间截断（保留头尾）、
字段级压缩（先删低价值字段）。它不调模型，只算"证据还在不在"。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass


@dataclass
class Event:
    ts: str
    level: str
    source: str
    message: str
    # 真正能定位故障的字段，故意放在 JSON 的最后一个键上
    fault_line: int | None = None

    def as_dict(self) -> dict:
        d = {
            "ts": self.ts,
            "level": self.level,
            "source": self.source,
            "message": self.message,
        }
        if self.fault_line is not None:
            d["fault_line"] = self.fault_line
        return d


def synth_events(n: int, seed: int) -> list[Event]:
    """合成一段内核 trace：只有一条带 fault_line，且它在最后一条事件里。"""
    rng = random.Random(seed)
    levels = ["INFO", "INFO", "INFO", "WARN", "DEBUG"]
    events = []
    for i in range(n):
        events.append(
            Event(
                ts=f"12:00:{i:02d}.{rng.randrange(1000):03d}",
                level=rng.choice(levels),
                source=rng.choice(["mm/filemap.c", "fs/read_write.c", "kernel/sched/core.c"]),
                message=f"event {i}: page cache lookup, flags=0x{rng.randrange(1 << 16):04x}",
            )
        )
    events[-1].fault_line = 4242
    events[-1].message = "BUG: unable to handle page fault for address 0xdeadbeef"
    events[-1].level = "ERROR"
    return events


def raw_truncate(payload: str, budget: int) -> str:
    """harness 里最常见的一刀切：按字符数截断。"""
    return payload[:budget]


def middle_truncate(payload: str, budget: int, keep: int = 200) -> str:
    """保留头尾各 keep 个字符，中间用省略号替代。仍然是文本级操作。"""
    if len(payload) <= budget:
        return payload
    head = payload[:keep]
    tail = payload[-keep:]
    return f"{head}\n...<{len(payload) - 2 * keep} chars elided>...\n{tail}"


def field_compress(events: list[Event], budget: int) -> str:
    """字段级压缩：先丢掉调试字段，再按时间倒序保留最近事件。

    重点不是"更聪明的字符串切割"，而是先理解结构：模型要的是 fault_line，
    不是每条 INFO 的 flags。压缩应该发生在结构上，而不是字节上。
    """
    slim = []
    for e in events:
        d = e.as_dict()
        d.pop("source", None)  # 低价值字段
        if d.get("fault_line") is None:
            d.pop("level", None)
        slim.append(d)
    # 从最近的事件往前装，保证最新证据优先
    kept: list[dict] = []
    for d in reversed(slim):
        candidate = json.dumps([d] + kept, ensure_ascii=False)
        if len(candidate) > budget and kept:
            break
        kept.insert(0, d)
    return json.dumps(kept, ensure_ascii=False)


def survives(payload: str, needle: str = "fault_line") -> tuple[bool, bool]:
    """返回 (证据字段还在不在, JSON 是否还能解析)。"""
    present = needle in payload
    try:
        json.loads(payload)
        valid = True
    except json.JSONDecodeError:
        valid = False
    return present, valid


def run(runs: int, budget: int, seed: int = 7) -> dict:
    stats = {
        "raw": {"evidence": 0, "valid_json": 0},
        "middle": {"evidence": 0, "valid_json": 0},
        "field": {"evidence": 0, "valid_json": 0},
    }
    for i in range(runs):
        events = synth_events(n=24, seed=seed + i)
        payload = json.dumps([e.as_dict() for e in events], ensure_ascii=False)
        for name, out in (
            ("raw", raw_truncate(payload, budget)),
            ("middle", middle_truncate(payload, budget)),
            ("field", field_compress(events, budget)),
        ):
            evidence, valid = survives(out)
            stats[name]["evidence"] += int(evidence)
            stats[name]["valid_json"] += int(valid)
    for v in stats.values():
        v["evidence_rate"] = v["evidence"] / runs
        v["json_rate"] = v["valid_json"] / runs
    return stats


def report(stats: dict, runs: int, budget: int) -> None:
    print(f"runs={runs}  budget={budget} chars")
    print(f"{'policy':<10}{'evidence kept':>16}{'parseable JSON':>18}")
    for name in ("raw", "middle", "field"):
        s = stats[name]
        print(f"{name:<10}{s['evidence_rate']:>15.1%}{s['json_rate']:>17.1%}")
    print("\n读法：raw 策略的证据保留率接近 0，因为 fault_line 在 JSON 末尾；")
    print("middle 能保住尾部证据，但 JSON 仍然是坏的；field 两者都保住。")


def test() -> None:
    events = synth_events(24, seed=1)
    payload = json.dumps([e.as_dict() for e in events], ensure_ascii=False)
    cut = raw_truncate(payload, 300)
    present, valid = survives(cut)
    assert not present and not valid

    mid = middle_truncate(payload, 300, keep=100)
    present, valid = survives(mid)
    assert present and not valid, "中间截断保住了证据，但不是合法 JSON"

    field = field_compress(events, 300)
    present, valid = survives(field)
    assert present and valid, field

    stats = run(20, 300)
    assert stats["field"]["evidence_rate"] == 1.0
    assert stats["field"]["json_rate"] == 1.0
    assert stats["raw"]["evidence_rate"] == 0.0
    print("token_budget: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=720)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        test()
        return
    report(run(args.runs, args.budget), args.runs, args.budget)


if __name__ == "__main__":
    main()
