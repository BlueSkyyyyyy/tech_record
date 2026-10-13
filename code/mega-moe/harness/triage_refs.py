#!/usr/bin/env python3
"""P123-1：把 `check_refs.py` 报出的「轻微漂移 near_miss」做**逐条定点分类**。

`check_refs.py`（P97-1）判 near_miss 的规则很粗：只要「同行出现过的某个反引号标识符
不在被引行内、但在文件里别处出现」，就算一条 near_miss。它**不区分**这个标识符到底是不是
那条引用的注脚——于是一行里有多条引用、或标识符只是行文提到的另一个符号时，就会大量误报。

本脚本（纯文件解析、架构无关）把每条 near_miss 归成五类，供人工定点复核：

  1. `in_range`      —— 标识符其实**就在（修正后的）被引区间内**，是 linter 的边界/子串判定漏掉。
                        「修正后」指把分裂区间 `` `X:l1`-`:l2` `` 先合并成 `X:l1-l2`（见 check_refs）。
  2. `same_line_other` —— 同一 Markdown 行有多条引用，该标识符按字符距离更贴近**另一条**引用
                        （典型的「一行多引用串扰」）。纯方法学假阳性。
  3. `adjacent_decl`  —— 标识符最近出现在 `l1-1` / `l2+1`，且那一行是**装饰器/注释/模板/签名头**
                        （`@lru_cache`、`// 注释`、`template <...>`、`static void f(` 等）。
                        属「引体、名在头」的书写约定，非错误。
  4. `owner_body`     —— 标识符是**定义在被引行之上的函数/结构体名**（其函数体覆盖被引区间）；
                        行文在体内引用时顺带点名 owner。非错误。
  5. `far_other`      —— 标识符最近出现处离被引区间 > `--far`（默认 5）行。行文顺带提到的无关符号
                        （多半同 `same_line_other` 一样是串扰），是**真正需要人工判断**的一类：
                        若它确实是该引用的注脚，则引用行号可能真的错了。

顺带把 `adjacent_decl` 里「被引行落在**签名下一行**」的模式单列 `--fixable`，因为这种
（`f(` 在第 N 行、body 从 N+1 行起，但作者把引用写成 N+1）是可以直接改文档修正的。

用法：
  python triage_refs.py                 # 报告 + JSON
  python triage_refs.py --far 3         # 调「远/近」阈值
  python triage_refs.py --quiet         # 只打印 JSON
"""
import os
import re
import sys
import json
import argparse
from collections import Counter, defaultdict

import check_refs as cr

CITE = cr.CITE
IDENT = cr.IDENT


def resolve_path(relpath):
    """'DeepGEMM/csrc/...' / 'DeepEP/...' / 'mega-moe/docs/...' → 绝对路径。"""
    tag, _, rest = relpath.partition("/")
    root = {"DeepGEMM": cr.DG, "DeepEP": cr.DEP, "mega-moe": cr.ROOT}.get(tag)
    if root is None:
        return None
    p = os.path.join(root, rest)
    return p if os.path.exists(p) else None


def citations_on(line):
    """返回 [(char_start, char_end, cited, l1, l2)]（已合并分裂区间）。"""
    line = cr.normalize_citations(line)
    out = []
    for m in CITE.finditer(line):
        out.append((m.start(), m.end(), m.group(1), int(m.group(2)),
                    int(m.group(3)) if m.group(3) else int(m.group(2))))
    return out


def id_spans(line):
    return [(m.start(), m.end(), m.group(1)) for m in IDENT.finditer(line)]


def is_decl_head(text):
    s = text.strip()
    return (s.startswith(("@", "//", "/*", "*", "#", "--", "//!"))
            or "template<" in s or "template <" in s
            or s.endswith(("\\", "(", ",")) and (" " in s))


def nearest_occurrence(src, word):
    pat = re.compile(r"\b" + re.escape(word) + r"\b")
    return [k + 1 for k, l in enumerate(src) if pat.search(l)]


def dist_to_range(o, l1, l2):
    if l1 <= o <= l2:
        return 0
    return o - l2 if o > l2 else l1 - o


def triage(far=5):
    byname, allfiles = cr.build_index()
    resolve = cr.make_resolver(byname)
    stats, problems, _, _ = cr.audit_docs()
    near = problems["near_miss"]

    cat = Counter()
    rows = []
    fixable = []
    for doc, lineno, citestr, relpath, ids in near:
        p = resolve_path(relpath)
        raw = open(os.path.join(cr.DOCS, doc), encoding="utf-8",
                   errors="replace").read().splitlines()[lineno - 1]
        cites = citations_on(raw)
        # 定位 near_miss 里的这条引用
        prefix = citestr.split(":")[0]
        this = None
        for c in cites:
            if c[2].endswith(prefix) or prefix.endswith(c[2]):
                this = c
                break
        if this is None or p is None:
            cat["unparsed"] += 1
            rows.append(dict(doc=doc, line=lineno, cite=citestr, ids=ids,
                             cat="unparsed", cls=[]))
            continue
        _, _, _, l1, l2 = this
        src = cr.read_lines(p) or []
        cspans = [(c[0], c[1]) for c in cites]
        ispans = id_spans(raw)
        cls = []
        for i in ids:
            occ = nearest_occurrence(src, i.split("::")[-1].rstrip("<"))
            # 该 id 在行内的字符跨度（取第一个匹配）
            span = next((s for s in ispans if s[2] == i), None)
            nearest = None
            if span is not None and cspans:
                def cdist(cs):
                    s, e = cs
                    if span[0] >= s and span[1] <= e:
                        return 0
                    return s - span[1] if span[1] <= s else span[0] - e
                nearest = min(cspans, key=cdist)
            same_other = nearest is not None and (nearest[0], nearest[1]) != (this[0], this[1])
            if not occ:
                cls.append(dict(id=i, kind="absent", occ=[]))
                cat["absent"] += 1
                continue
            d = min(dist_to_range(o, l1, l2) for o in occ)
            near_o = min(occ, key=lambda o: dist_to_range(o, l1, l2))
            if d == 0:
                kind = "in_range"
            elif same_other:
                kind = "same_line_other"
            elif near_o == l1 - 1 or near_o == l2 + 1:
                adjline = src[near_o - 1] if 1 <= near_o <= len(src) else ""
                kind = "adjacent_decl" if is_decl_head(adjline) else "adjacent_code"
                if kind == "adjacent_code":
                    fixable.append((doc, lineno, citestr, i, l1, l2, near_o))
            elif near_o < l1:
                kind = "owner_body" if near_o <= l1 else "far_other"
            elif near_o > l2:
                kind = "far_other" if near_o > l2 + far else "owner_body"
            else:  # pragma: no cover
                kind = "far_other"
            cat[kind] += 1
            cls.append(dict(id=i, kind=kind, occ=near_o, d=d))
        rows.append(dict(doc=doc, line=lineno, cite=citestr, ids=ids,
                         cat=Counter(c["kind"] for c in cls).most_common(1)[0][0],
                         cls=cls))
    return dict(total=len(near), categories=dict(cat),
                merged_ranges=stats["merged_ranges"], rows=rows, fixable=fixable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--far", type=int, default=5)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    res = triage(args.far)
    if not args.quiet:
        print("=" * 96)
        print("P123-1 · near_miss 定点分类（triage_refs.py）")
        print("=" * 96)
        print(f"分裂区间合并      : {res['merged_ranges']}")
        print(f"near_miss 总条数  : {res['total']}")
        print("分类：")
        for k, v in sorted(res["categories"].items(), key=lambda x: -x[1]):
            print(f"  {k:18s}: {v}")
        if res["fixable"]:
            print(f"\n[candidate off-by-one] {len(res['fixable'])} 条（相邻行是真代码，非注释/签名）：")
            for doc, lineno, cite, i, l1, l2, o in res["fixable"]:
                print(f"  {doc}:{lineno}  {cite}  id={i}  cited=[{l1},{l2}] occ={o}")
    out = dict(total=res["total"], merged_ranges=res["merged_ranges"],
               categories=res["categories"], fixable=len(res["fixable"]))
    print()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
