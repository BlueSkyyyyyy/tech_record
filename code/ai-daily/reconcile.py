#!/usr/bin/env python3
"""体检：已发布的日报 vs 上游现在的当日论文数。

HF 的 daily 列表在**当天晚上还会继续进人**，21:00 抓到的通常只有最终量的
60%~80%。本脚本对每个已发布日期重抓一次，报告「发布时写了多少 / 现在上游多少 /
新增了哪些论文号」，用来决定要不要补遗。

用法：
    python3 reconcile.py                 # 检查全部已发布日期
    python3 reconcile.py --since 2026-11-09
    python3 reconcile.py --days 3        # 只看最近 3 天
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_papers import normalize, try_sources  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "code" / "ai-daily" / "docs"
ID_RE = re.compile(r"huggingface\.co/papers/(\d{4}\.\d{4,5})")


def published(date: str) -> tuple[int | None, set[str], str | None]:
    path = DOCS / f"{date}.md"
    if not path.is_file():
        return None, set(), None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^# \d{4}-\d{2}-\d{2} · (\d+) 篇", text, re.M)
    count = int(m.group(1)) if m else None
    return count, set(ID_RE.findall(text)), None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="日报与上游当日论文数的漂移体检")
    ap.add_argument("--since", help="只查该日期之后（YYYY-MM-DD）")
    ap.add_argument("--days", type=int, help="只查最近 N 天已发布的")
    args = ap.parse_args(argv)

    dates = sorted(p.stem for p in DOCS.glob("*.md") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem))
    if args.since:
        dates = [d for d in dates if d >= args.since]
    if args.days:
        dates = dates[-args.days:]

    drift = 0
    for d in dates:
        have_count, have_ids, _ = published(d)
        try:
            entries, _src = try_sources(d)
            now = normalize(entries)
        except Exception as exc:  # 网络问题不该让体检崩掉
            print(f"{d}: 抓取失败（{exc}）")
            continue
        now_ids = {p["id"] for p in now if p["id"]}
        missing = sorted(now_ids - have_ids)
        status = "ok" if not missing else f"缺 {len(missing)} 篇"
        flag = "" if not missing else "  <-- 建议补遗"
        print(f"{d}: 发布 {have_count or '?'} 篇 / 上游 {len(now_ids)} 篇 | {status}{flag}")
        if missing:
            drift += 1
            print("   新增：", " ".join(missing[:12]) + (" …" if len(missing) > 12 else ""))
    print(f"\n{len(dates)} 个日期，{drift} 个有缺口")
    return 0


if __name__ == "__main__":
    sys.exit(main())
