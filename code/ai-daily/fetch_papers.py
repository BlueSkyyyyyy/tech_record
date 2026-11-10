#!/usr/bin/env python3
"""抓取 Hugging Face Daily Papers 某一天的论文清单，输出 JSON + 人读 brief。

数据源（按顺序尝试，任一成功即止）：
  1. https://huggingface.co/api/daily_papers?date=YYYY-MM-DD        （JSON，首选）
  2. https://hf-mirror.com/api/daily_papers?date=YYYY-MM-DD         （国内镜像）
  3. 页面 https://huggingface.co/papers/date/YYYY-MM-DD 的 Svelte
     `data-props` 里内嵌的 JSON（前两个都 404/被墙时兜底）

用法：
    python3 fetch_papers.py                          # 默认取昨天（本地时区）
    python3 fetch_papers.py --date 2026-11-06
    python3 fetch_papers.py --date 2026-11-06 --out ~/tmp/papers.json --brief ~/tmp/day.md
    python3 fetch_papers.py --test                   # 离线自测解析/排序/摘要截断

只依赖标准库。--test 不联网。
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (X11; Linux x86_64) ai-daily/1.0"
MIRROR_API = "https://hf-mirror.com/api/daily_papers?date={date}&limit={limit}"
OFFICIAL_API = "https://huggingface.co/api/daily_papers?date={date}&limit={limit}"
OFFICIAL_LATEST_API = "https://huggingface.co/api/daily_papers?limit={limit}"
MIRROR_LATEST_API = "https://hf-mirror.com/api/daily_papers?limit={limit}"
MIRROR_PAGE = "https://hf-mirror.com/papers/date/{date}"
OFFICIAL_PAGE = "https://huggingface.co/papers/date/{date}"


# --------------------------------------------------------------------------- 网络
def fetch(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_json_retry(url_template: str, date: str = "", timeout: int = 45) -> object:
    """请求 JSON；镜像偶尔会截断长响应，解析失败就降 limit / 重试一次。

    url_template 里可用 {date} 与 {limit}。
    """
    last_exc: Exception | None = None
    for limit in (100, 50, 25):
        url = url_template.format(date=date, limit=limit)
        for attempt in (1, 2):
            try:
                return json.loads(fetch(url, timeout))
            except json.JSONDecodeError as exc:
                last_exc = exc           # 截断/半包，降 limit 再来
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                last_exc = exc
                break                    # 连不上就不用换 limit 了
    raise ValueError(f"请求失败（{last_exc}）")


def _daily_date(entry: dict) -> str:
    """条目「登上 daily」的日期（YYYY-MM-DD）；缺字段返回空串。"""
    p = entry.get("paper") or entry
    return (p.get("submittedOnDailyAt") or "")[:10]


def try_sources(date: str, timeout: int = 45) -> tuple[list, str]:
    """返回 (paper 列表, 来源说明)。

    顺序有讲究，都是实测踩出来的：
    1. **最新一批（不带 date）+ 本地按日期过滤**——最可信。`?date=` 在某天批次
       刚开始收录时会返回 0（2026-11-09 实测），日期页也常常只给部分（2026-11-10
       实测只给 2 篇，实际 23 篇）。
    2. **按日期 API**——批次封板后它最准，也最省流量。
    3. **日期页内嵌 JSON**——最后兜底。
    """
    errors = []

    # 1) 最新一批，本地过滤
    for url in (OFFICIAL_LATEST_API, MIRROR_LATEST_API):
        try:
            data = fetch_json_retry(url, date, timeout)
            if not isinstance(data, list):
                errors.append(f"{url}: 返回不是 list")
                continue
            picked = [e for e in data if _daily_date(e) == date]
            if picked:
                return picked, url.split("?")[0] + " (最新一批，本地按日期过滤)"
            newest = sorted({_daily_date(e) for e in data if _daily_date(e)}, reverse=True)[:3]
            errors.append(f"{url}: 最新一批里没有 {date}（最新是 {newest}）")
        except ValueError as exc:
            errors.append(f"{url}: {exc}")

    # 2) 按日期 API
    for url in (OFFICIAL_API, MIRROR_API):
        try:
            data = fetch_json_retry(url, date, timeout)
            if isinstance(data, list) and data:
                return data, url.format(date=date, limit="")
            errors.append(f"{url}: 按日期过滤为空（上游 date 过滤滞后）")
        except ValueError as exc:
            errors.append(f"{url}: {exc}")

    # 3) 日期页
    for url in (OFFICIAL_PAGE, MIRROR_PAGE):
        try:
            page = fetch(url.format(date=date), timeout).decode("utf-8", "replace")
            data = parse_svelte_page(page)
            if data:
                return data, url.format(date=date) + " (page)"
            errors.append(f"{url}: 页面里没有 data-props JSON")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError("所有数据源都失败或为空：\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- 解析
PROPS_RE = re.compile(r'data-props="([^"]+)"')


def parse_svelte_page(page: str) -> list:
    """从 /papers/date/<d> 页面抽出论文数组。

    Svelte 把 props 放进 `data-props="{&quot;...&quot;}"`，转义一层 HTML 实体后
    就是普通 JSON；论文数组所在的 key 会变，所以按结构找：list[dict] 且每项
    有 `paper` 与 `title`。
    """
    for raw in PROPS_RE.findall(page):
        text = html.unescape(raw)
        if "paper" not in text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        for value in _walk_lists(obj):
            if value and all(isinstance(v, dict) and "paper" in v and "title" in v for v in value):
                return value
    return []


def _walk_lists(node):
    if isinstance(node, list):
        yield node
        for item in node:
            yield from _walk_lists(item)
    elif isinstance(node, dict):
        for item in node.values():
            yield from _walk_lists(item)


def _org(entry: dict) -> str:
    p = entry.get("paper") or {}
    for cand in (p.get("organization"), entry.get("organization")):
        if isinstance(cand, dict) and cand.get("fullname"):
            return cand["fullname"]
    return ""


def normalize(entries: list) -> list[dict]:
    """拍平成稳定字段；按（upvotes 降序, paper id 升序）排序。"""
    out = []
    for entry in entries:
        p = entry.get("paper") or {}
        authors = [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")]
        out.append(
            {
                "id": p.get("id", ""),
                "title": p.get("title") or entry.get("title") or "",
                "abstract": p.get("summary") or entry.get("summary") or "",
                "ai_summary": p.get("ai_summary") or "",
                "ai_keywords": p.get("ai_keywords") or [],
                "authors": authors,
                "organization": _org(entry),
                "upvotes": int(p.get("upvotes") or 0),
                "published_at": p.get("publishedAt") or "",
                "daily_at": p.get("submittedOnDailyAt") or "",
                "github": p.get("githubRepo") or "",
                "project": p.get("projectPage") or "",
                "hf_url": f"https://huggingface.co/papers/{p.get('id', '')}",
                "arxiv_url": f"https://arxiv.org/abs/{p.get('id', '')}",
            }
        )
    out.sort(key=lambda x: (-x["upvotes"], x["id"]))
    return out


def dedupe(papers: list[dict]) -> list[dict]:
    seen, kept = set(), []
    for p in papers:
        if not p["id"] or p["id"] in seen:
            continue
        seen.add(p["id"])
        kept.append(p)
    return kept


# --------------------------------------------------------------------------- 输出
def render_brief(date: str, papers: list[dict], source: str, abstract_chars: int = 1200) -> str:
    n = len(papers)
    total_up = sum(p["upvotes"] for p in papers)
    lines = [
        f"# HF Daily Papers · {date}",
        "",
        f"- 论文数：**{n}**；总 upvotes：**{total_up}**",
        f"- 数据源：`{source}`",
        f"- 抓取时间：{dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "> 本文件是**素材**，不是成稿：写日报时按 upvotes 与主题挑重点，摘要都要自己重写并保留数字。",
        "",
        "## 速览（按 upvotes 降序）",
        "",
        "| # | upvotes | id | 标题 | 机构 |",
        "|---|---|---|---|---|",
    ]
    for i, p in enumerate(papers, 1):
        title = p["title"].replace("|", "\\|")
        org = (p["organization"] or "—").replace("|", "\\|")
        lines.append(f"| {i} | {p['upvotes']} | [{p['id']}]({p['hf_url']}) | {title} | {org} |")

    lines += ["", "## 逐篇素材", ""]
    for i, p in enumerate(papers, 1):
        lines += [
            f"### {i}. {p['title']}",
            "",
            f"- id：`{p['id']}`｜upvotes：**{p['upvotes']}**｜机构：{p['organization'] or '—'}",
            f"- 链接：[HF]({p['hf_url']})｜[arXiv]({p['arxiv_url']})"
            + (f"｜[code]({p['github']})" if p["github"] else "")
            + (f"｜[project]({p['project']})" if p["project"] else ""),
            f"- 作者：{', '.join(p['authors'][:8])}" + (" 等" if len(p["authors"]) > 8 else ""),
            f"- arXiv 发布：{(p['published_at'] or '—')[:10]}｜登上 daily：{(p['daily_at'] or '—')[:10]}",
        ]
        if p["ai_summary"]:
            lines.append(f"- HF AI 摘要（机器生成，仅作参照）：{p['ai_summary']}")
        if p["ai_keywords"]:
            lines.append(f"- 关键词：{', '.join(p['ai_keywords'][:8])}")
        abstract = p["abstract"] or ""
        if len(abstract) > abstract_chars:
            abstract = abstract[:abstract_chars].rstrip() + " …"
        lines += ["", "**摘要**：", "", abstract, "", "---", ""]
    return "\n".join(lines)


def render_json(date: str, papers: list[dict], source: str) -> str:
    return json.dumps(
        {
            "date": date,
            "source": source,
            "fetched_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "count": len(papers),
            "papers": papers,
        },
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------------------------------- 自测
def _selftest() -> int:
    page = (
        '<div data-props="{&quot;items&quot;:[{&quot;paper&quot;:{&quot;id&quot;:&quot;2611.00002&quot;,'
        + '&quot;title&quot;:&quot;B&quot;,&quot;upvotes&quot;:3,&quot;summary&quot;:&quot;b&quot;},'
        + '&quot;title&quot;:&quot;B&quot;}]}"></div>'
    )
    parsed = parse_svelte_page(page)
    assert len(parsed) == 1 and parsed[0]["paper"]["id"] == "2611.00002", "Svelte 页面解析失败"
    assert parse_svelte_page("<html>nothing</html>") == [], "空页面应返回空列表"

    entries = [
        {"paper": {"id": "2611.00001", "title": "A", "upvotes": 1, "summary": "abstract-a-long"},
         "title": "A", "organization": {"fullname": "Org-A"}},
        {"paper": {"id": "2611.00002", "title": "B", "upvotes": 9, "summary": "abstract-b-long",
                   "authors": [{"name": "X"}, {"name": "Y"}], "githubRepo": "https://g/x"},
         "title": "B"},
        {"paper": {"id": "2611.00002", "title": "B dup", "upvotes": 9, "summary": "abstract-b-long"},
         "title": "B dup"},
    ]
    papers = dedupe(normalize(entries))
    assert [p["id"] for p in papers] == ["2611.00002", "2611.00001"], "排序/去重不对"
    assert papers[0]["organization"] == "" and papers[1]["organization"] == "Org-A", "机构提取不对"
    assert papers[0]["authors"] == ["X", "Y"] and papers[0]["github"] == "https://g/x"
    assert papers[0]["hf_url"].endswith("2611.00002")

    brief = render_brief("2026-01-01", papers, "test", abstract_chars=1)
    assert "| 1 | 9 |" in brief and "| 2 | 1 |" in brief and "…" in brief, "brief 渲染/摘要截断不对"
    assert json.loads(render_json("2026-01-01", papers, "test"))["count"] == 2

    print("selftest ok: 2 papers, parse/dedupe/sort/render 全通过")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="抓取 HF Daily Papers 某一天")
    ap.add_argument("--date", help="YYYY-MM-DD，默认昨天（本地时区）")
    ap.add_argument("--out", help="JSON 输出路径（默认 <date>.json）")
    ap.add_argument("--brief", help="人读素材输出路径（默认 <date>.md）")
    ap.add_argument("--no-brief", action="store_true", help="只要 JSON")
    ap.add_argument("--test", action="store_true", help="离线自测")
    args = ap.parse_args(argv)

    if args.test:
        return _selftest()

    date = args.date or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    dt.date.fromisoformat(date)  # 早失败

    try:
        entries, source = try_sources(date)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    papers = dedupe(normalize(entries))
    if not papers:
        print(f"[warn] {date} 没有论文（周末/节假日 daily 可能为空）", file=sys.stderr)

    out = Path(args.out or f"{date}.json").expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_json(date, papers, source), encoding="utf-8")
    print(f"{date}: {len(papers)} papers -> {out}  (source: {source})")

    if not args.no_brief:
        brief = Path(args.brief or f"{date}.md").expanduser()
        brief.parent.mkdir(parents=True, exist_ok=True)
        brief.write_text(render_brief(date, papers, source), encoding="utf-8")
        print(f"brief -> {brief}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
