#!/usr/bin/env python3
"""P96-1：DeepGEMM / DeepEP 源码引用行号健康检查 + 算子 / API 全索引。

本专题到第 105 篇已引用约 1900 处 `文件:行号`。上游一旦改动，这些行号就会「漂移」，
而内联到博客的文档没有人会回头逐条核对。这个脚本把「哪些引用还对得上」变成可复现的数字：

  Part A  算子 / API 全索引：把 DeepGEMM 的每个 device impl 头、每个 host API 入口，
          连同它的 sha256 与「被哪些篇目覆盖」列成一张表。
  Part B  引用健康检查：从 docs/*.md 正则抽取所有 `path.ext:line[-line]`，对每个引用
          到源文件里核对（文件是否存在 / 行是否越界 / 行是否非空 / 引用同一行反引号里的
          标识符是否真的出现在被引行里）。
  Part C  漂移清单：逐条列出对不上的引用（供人工修正）。

不加载 GPU、不编译 kernel，纯文件解析，任何机器都能跑；在 kernel_lab(H100) 里跑只是为了
和专题其它实验同一环境。

用法：python exp_deepgemm_source_audit.py
"""
import os
import re
import glob
import json
import hashlib
import subprocess
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # code/mega-moe
DOCS = os.path.join(ROOT, "docs")
def _find_repo(name, env):
    cands = [os.environ.get(env, ""), os.path.expanduser(f"~/github/{name}"),
             os.path.expanduser(f"~/github/{name}"), os.path.expanduser(f"~/github/{name}")]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return os.path.expanduser(f"~/github/{name}")


DG = _find_repo("DeepGEMM", "DEEPGEMM_DIR")
DEP = _find_repo("DeepEP", "DEEPEP_DIR")
LOCAL_ROOTS = [ROOT]                              # 允许引用本专题自己的 src/harness

CITE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./\-]*\.(?:cuh|hpp|cpp|cu|py)):(\d+)(?:-(\d+))?")
# 反引号里的标识符（含 ::、<> 前的名字），用于「内容匹配」抽查
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_:]{2,})`")


def sh(args, cwd):
    try:
        return subprocess.check_output(["git", "-c", "safe.directory=*"] + args, cwd=cwd,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "?"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except Exception:
        return None


def build_file_index():
    """basename -> [abs paths]，以及 rel-path 后缀 -> abs path。"""
    byname = defaultdict(list)
    allfiles = []
    for root in (DG, DEP) + tuple(LOCAL_ROOTS):
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".git", "build", "__pycache__")]
            for fn in fns:
                p = os.path.join(dp, fn)
                if any(p.startswith(x) for x in (DG, DEP, ROOT)) and fn.rsplit(".", 1)[-1] in (
                        "cuh", "hpp", "cpp", "cu", "py"):
                    allfiles.append(p)
                    byname[fn].append(p)
    return byname, allfiles


def resolve(cited, byname):
    """按 basename 或后缀匹配解析被引文件。歧义时返回全部候选。"""
    base = os.path.basename(cited)
    cands = byname.get(base)
    if cands:
        # 若引用里带目录信息，尽量用后缀筛
        if "/" in cited:
            suf = cited.lstrip("./")
            filt = [c for c in cands if c.endswith(suf)]
            if filt:
                return filt
        return cands
    # 退化：按后缀匹配
    for name, ps in byname.items():
        for p in ps:
            if p.endswith(cited):
                return [p]
    return []


def rel(p):
    for base, tag in ((DG, "DeepGEMM"), (DEP, "DeepEP"), (ROOT, "mega-moe")):
        if p.startswith(base):
            return f"{tag}/{os.path.relpath(p, base)}"
    return p


def part_a_index(byname):
    print("=" * 100)
    print("Part A · DeepGEMM 算子 / API 全索引（device impls + host apis，含 sha256 与覆盖篇目）")
    print("=" * 100)
    dg_impls = sorted(glob.glob(os.path.join(DG, "deep_gemm/include/deep_gemm/impls/*.cuh")))
    dg_apis = sorted(glob.glob(os.path.join(DG, "csrc/apis/*.hpp")))
    doc_text = {}
    for d in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        doc_text[os.path.basename(d)] = open(d, encoding="utf-8", errors="replace").read()

    def covering(basename):
        hits = []
        for name, txt in doc_text.items():
            # 仅当提到文件名（basename）才算覆盖
            if basename in txt:
                hits.append(name.split("-")[0])
        return hits

    print(f"\n-- device impls（{len(dg_impls)} 个）--")
    print(f"{'file':46s} {'sha256':13s} {'行数':>5s}  覆盖篇目(#)")
    for p in dg_impls:
        b = os.path.basename(p)
        lines = len(read_lines(p) or [])
        cov = covering(b)
        print(f"{b:46s} {sha256(p):13s} {lines:5d}  {' '.join(cov) if cov else '(none!)'}")

    print(f"\n-- host apis（{len(dg_apis)} 个）--")
    print(f"{'file':46s} {'sha256':13s} {'行数':>5s}  覆盖篇目(#)")
    for p in dg_apis:
        b = os.path.basename(p)
        lines = len(read_lines(p) or [])
        cov = covering(b)
        print(f"{b:46s} {sha256(p):13s} {lines:5d}  {' '.join(cov) if cov else '(none!)'}")
    return doc_text


def part_bc_audit(byname, doc_text):
    print()
    print("=" * 100)
    print("Part B · 引用健康检查（docs/*.md 里所有 `file.ext:line[-line]`）")
    print("=" * 100)

    total = 0
    unresolved = []
    oob = []
    empty = []
    content_total = 0
    content_ok = 0
    content_weak_ok = 0
    content_miss = []
    hard_miss = []
    per_file = Counter()
    per_doc = Counter()
    linecache = {}

    for docname, txt in sorted(doc_text.items()):
        for lineno_in_doc, dline in enumerate(txt.splitlines(), 1):
            for m in CITE.finditer(dline):
                cited, l1, l2 = m.group(1), int(m.group(2)), m.group(3)
                l2 = int(l2) if l2 else l1
                total += 1
                per_doc[docname] += 1
                cands = resolve(cited, byname)
                if not cands:
                    unresolved.append((docname, lineno_in_doc, m.group(0)))
                    continue
                ids_all = [i.split("::")[-1].rstrip("<") for i in IDENT.findall(dline)]
                ids_all = [i for i in ids_all if not i.endswith((".cuh", ".hpp", ".cpp", ".cu", ".py"))
                           and len(i) >= 4]

                def score(c):
                    src = linecache.get(c)
                    if src is None:
                        src = linecache[c] = read_lines(c)
                    if src is None:
                        return (-99,)
                    inr = 1 if (1 <= l1 <= len(src) and l2 <= len(src)) else 0
                    seg = "\n".join(src[l1 - 1:min(l2, len(src))]) if inr else ""
                    strict = 1 if (ids_all and any(i in seg for i in ids_all)) else 0
                    weak = 1 if any(i in "\n".join(src) for i in ids_all) else 0
                    suf = 1 if ("/" in cited and c.endswith(cited.lstrip("./"))) else 0
                    return (inr, strict, weak, suf)

                p = max(cands, key=score)
                per_file[os.path.basename(p)] += 1
                if p not in linecache:
                    linecache[p] = read_lines(p)
                src = linecache[p]
                if src is None or l1 < 1 or l1 > len(src) or l2 > len(src):
                    oob.append((docname, lineno_in_doc, m.group(0), rel(p), len(src or [])))
                    continue
                seg = src[l1 - 1:l2]
                if not any(s.strip() for s in seg):
                    empty.append((docname, lineno_in_doc, m.group(0), rel(p)))
                    continue
                # 内容抽查：同一行反引号里的标识符是否出现在被引行（严格）/文件（宽松）里
                if ids_all:
                    content_total += 1
                    blob = "\n".join(seg)
                    full = "\n".join(src)
                    if any(i in blob for i in ids_all):
                        content_ok += 1
                        content_weak_ok += 1
                    elif any(i in full for i in ids_all):
                        content_weak_ok += 1          # 符号在文件里、只是不在被引行 → 需复核的轻微漂移
                        content_miss.append((docname, lineno_in_doc, m.group(0), rel(p), ids_all[:4]))
                    else:
                        hard_miss.append((docname, lineno_in_doc, m.group(0), rel(p), ids_all[:4]))

    print(f"\n总引用条数            : {total}")
    print(f"  可解析到源文件      : {total - len(unresolved)}  ({100*(total-len(unresolved))/max(total,1):.2f}%)")
    print(f"  无法解析（占位/截断）: {len(unresolved)}")
    print(f"  行号越界/文件缺失   : {len(oob)}")
    print(f"  被引行全为空        : {len(empty)}")
    print(f"内容抽查（同行有 ≥4 字符反引号标识符）: {content_total} 条")
    print(f"  · 严格命中（标识符就在被引行）    : {content_ok} ({100*content_ok/max(content_total,1):.2f}%)")
    print(f"  · 轻微漂移（符号在本文件、不在被引行）: {len(content_miss)} → 见 Part C，可复核")
    print(f"  · 宽松命中（符号在文件内）        : {content_weak_ok} ({100*content_weak_ok/max(content_total,1):.2f}%)")
    print(f"  · 行内标识符与本文件无关（多为散文里的算子名/构造参数）: {len(hard_miss)}")
    print(f"覆盖的源文件数        : {len(per_file)}")

    print("\n-- 引用最多的 20 个源文件 --")
    for b, c in per_file.most_common(20):
        print(f"  {c:4d}  {b}")

    print("\n-- 引用最多的 15 篇文档 --")
    for d, c in per_doc.most_common(15):
        print(f"  {c:4d}  {d}")

    print("\n" + "=" * 100)
    print("Part C · 漂移清单（需要人工复核的引用）")
    print("=" * 100)
    if unresolved:
        print(f"\n[无法解析] {len(unresolved)} 条：")
        for it in unresolved[:30]:
            print(f"  {it[0]}:{it[1]}  {it[2]}")
    if oob:
        print(f"\n[行号越界/文件缺失] {len(oob)} 条：")
        for it in oob[:30]:
            print(f"  {it[0]}:{it[1]}  {it[2]}  -> {it[3]} (len={it[4]})")
    if empty:
        print(f"\n[被引行全空] {len(empty)} 条：")
        for it in empty[:30]:
            print(f"  {it[0]}:{it[1]}  {it[2]}  -> {it[3]}")
    print(f"\n[轻微漂移：符号在本文件、不在被引行] {len(content_miss)} 条（建议复核；含 off-by-one / 行区间端点）：")
    for it in content_miss[:40]:
        print(f"  {it[0]}:{it[1]}  {it[2]}  -> {it[3]}  ids={it[4]}")
    print(f"\n[行内标识符与本文件无关] {len(hard_miss)} 条（多为散文里引用的算子名/构造参数，非漂移）：")
    for it in hard_miss[:20]:
        print(f"  {it[0]}:{it[1]}  {it[2]}  -> {it[3]}  ids={it[4]}")

    return dict(total=total, unresolved=len(unresolved), oob=len(oob), empty=len(empty),
                content_total=content_total, content_ok=content_ok,
                content_weak_ok=content_weak_ok,
                near_miss=len(content_miss), hard_miss=len(hard_miss), files=len(per_file))


def main():
    print("P96-1 · DeepGEMM / DeepEP 源码引用行号健康检查 + 算子 / API 全索引")
    print(f"DeepGEMM: {sh(['rev-parse','--short','HEAD'], DG)}  "
          f"({sh(['log','-1','--format=%s'], DG)})")
    print(f"DeepEP  : {sh(['rev-parse','--short','HEAD'], DEP)}  "
          f"({sh(['log','-1','--format=%s'], DEP)})")
    byname, allfiles = build_file_index()
    print(f"索引文件数: {len(allfiles)}（DeepGEMM + DeepEP + 本专题）")
    doc_text = part_a_index(byname)
    stats = part_bc_audit(byname, doc_text)
    print()
    print("=" * 100)
    print("汇总 JSON")
    print("=" * 100)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
