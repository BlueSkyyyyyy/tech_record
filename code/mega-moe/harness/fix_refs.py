#!/usr/bin/env python3
"""P143-1：把 `check_refs.py` 的 `near_miss` 清单变成**可复核、可定位、可验证**的人工审核表。

`check_refs.py`（P97-1）只给一行 near_miss 摘要，`triage_refs.py`（P123-1）把它粗分成五类。
但两者都不回答复核时最关键的两个问题：

  1. **这个标识符到底在不在（被解析到的）文件里？不在的话在别的哪个文件里？**
     `absent` 类里既有「标识符只是行文顺带提到、本就不该在被引文件」的假阳性，
     也可能藏着**真正写错的符号名**（例如把 `layout/mega_moe.cuh` 里实际叫 `Workspace`
     的结构体写成另一个不存在的名字）。前者无需改，后者必须改。
  2. **若在文件里、只是行号差一行，能否给出确定性的「改到哪」建议？**
     `adjacent_code` 类几乎都是「引了 `if (env)` 守卫行、标识符其实在下一行」，
     可以把引用区间机械地扩到包含该标识符，且**只可能让引用更精确（变成超集）**。

本脚本（纯文件解析、架构无关、不加载 GPU）对每条 near_miss 输出一张审核行：
`doc:line  cited  id  verdict  proposal`，verdict ∈
  - `in_range`      —— 标识符就在被引区间内（linter 的分裂区间/子串判定漏报），无需改；
  - `widened`       —— 标识符在 `l1-1`/`l2+1`，建议把区间扩到包含它（给出 before→after）；
  - `wrong_range`   —— 标识符在文件里但离被引区间很远（需人工；给出它所在的真实行号）；
  - `wrong_file`    —— 标识符不在被引文件、但在**别的文件**里出现（给出候选 `repo/path:line`）；
  - `unknown_symbol`—— 全语料都找不到该标识符（多半是文档里写错的符号名，需人工）；
  - `cross_talk`    —— 同一行有多条引用、该标识符更贴近另一条（纯方法学假阳性）。

用法：
  python fix_refs.py                 # 打印审核表 + JSON 汇总
  python fix_refs.py --apply         # 只把 `widened` 类机械地改到文档里（默认 dry-run）
  python fix_refs.py --only widened  # 只看某一类
  python fix_refs.py --quiet         # 只打印 JSON
"""
import os
import re
import sys
import json
import argparse
from collections import Counter

import check_refs as cr
import triage_refs as tr


def all_occurrences_by_file(byname, word):
    """全语料搜 `word`，返回 [(repo_relpath, line), ...]（按出现顺序，最多若干条）。"""
    pat = re.compile(r"\b" + re.escape(word) + r"\b")
    hits = []
    for fn, paths in sorted(byname.items()):
        for p in paths:
            src = cr.read_lines(p)
            if not src:
                continue
            for k, l in enumerate(src, 1):
                if pat.search(l):
                    hits.append((cr.rel(p), k))
                    break  # 每个文件只取首个出现，够用来定位
    return hits


def proposal_for(cited, l1, l2, near_o):
    """把区间 [l1,l2] 扩到包含 near_o，返回新的 `cited:lo-hi` 文本。"""
    lo, hi = min(l1, near_o), max(l2, near_o)
    return f"{cited}:{lo}" if lo == hi else f"{cited}:{lo}-{hi}"


def review(only=None):
    byname, _ = cr.build_index()
    _, problems, _, _ = cr.audit_docs()
    rows = []
    verdicts = Counter()
    for doc, lineno, citestr, relpath, ids in problems["near_miss"]:
        p = tr.resolve_path(relpath)
        raw = open(os.path.join(cr.DOCS, doc), encoding="utf-8",
                   errors="replace").read().splitlines()[lineno - 1]
        cites = tr.citations_on(raw)
        cspans = [(c[0], c[1]) for c in cites]
        prefix = citestr.split(":")[0]
        this = None
        for c in cites:
            if c[2].endswith(prefix) or prefix.endswith(c[2]):
                this = c
                break
        if this is None or p is None:
            continue
        _, _, _, l1, l2 = this
        src = cr.read_lines(p) or []
        ispans = tr.id_spans(raw)
        for i in ids:
            bare = i.split("::")[-1].rstrip("<")
            occ = tr.nearest_occurrence(src, bare)
            span = next((s for s in ispans if s[2] == i), None)
            nearest = None
            if span is not None and cspans:
                def cdist(cs):
                    s, e = cs
                    if span[0] >= s and span[1] <= e:
                        return 0
                    return s - span[1] if span[1] <= s else span[0] - e
                nearest = min(cspans, key=cdist)
            cross = nearest is not None and (nearest[0], nearest[1]) != (this[0], this[1])
            if occ and any(l1 <= o <= l2 for o in occ):
                v = "in_range"
                prop = ""
            elif cross:
                v = "cross_talk"
                prop = ""
            elif occ and (min(occ, key=lambda o: tr.dist_to_range(o, l1, l2)) in (l1 - 1, l2 + 1)):
                near_o = min(occ, key=lambda o: tr.dist_to_range(o, l1, l2))
                v = "widened"
                prop = f"{citestr} -> {proposal_for(citestr.split(':')[0], l1, l2, near_o)}"
            elif occ:
                v = "wrong_range"
                prop = f"id@{tr.nearest_occurrence(src, bare)[:4]} in file"
            else:
                hits = all_occurrences_by_file(byname, bare)
                if hits:
                    v = "wrong_file"
                    prop = "; ".join(f"{rp}:{ln}" for rp, ln in hits[:3])
                else:
                    v = "unknown_symbol"
                    prop = ""
            verdicts[v] += 1
            if only and v != only:
                continue
            rows.append(dict(doc=doc, line=lineno, cite=citestr, id=i,
                             verdict=v, proposal=prop))
    return rows, dict(verdicts)


def apply_widened(rows_by_doc):
    """把 widened 类机械改写回文档。rows_by_doc: {doc_name: [(lineno, old, new), ...]}"""
    changed = 0
    for doc, edits in rows_by_doc.items():
        path = os.path.join(cr.DOCS, doc)
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines(keepends=True)
        # 同一行多条编辑按出现顺序做，避免相互覆盖
        for lineno, old, new in sorted(edits, key=lambda e: (e[0], -len(e[1]))):
            idx = lines[lineno - 1].find(old)
            if idx >= 0:
                lines[lineno - 1] = lines[lineno - 1][:idx] + new + lines[lineno - 1][idx + len(old):]
                changed += 1
        open(path, "w", encoding="utf-8").write("".join(lines))
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows, verdicts = review(args.only)
    if not args.quiet:
        print("=" * 96)
        print("P143-1 · near_miss 引用复核表（fix_refs.py）")
        print("=" * 96)
        print("verdict 分布：")
        for k, v in sorted(verdicts.items(), key=lambda x: -x[1]):
            print(f"  {k:16s}: {v}")
        print()
        for r in rows:
            line = f"  {r['verdict']:14s} {r['doc']}:{r['line']}  {r['cite']}  id={r['id']}"
            if r["proposal"]:
                line += f"  | {r['proposal']}"
            print(line)

    if args.apply:
        by_doc = {}
        for r in rows:
            if r["verdict"] != "widened":
                continue
            old_cite, new_cite = [s.strip() for s in r["proposal"].split("->")]
            by_doc.setdefault(r["doc"], []).append((r["line"], old_cite, new_cite))
        n = apply_widened(by_doc)
        print(f"\n[apply] 改写 {n} 处 widened 引用。")

    print()
    print(json.dumps(dict(total=len(rows), verdicts=verdicts), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
