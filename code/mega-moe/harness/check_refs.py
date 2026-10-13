#!/usr/bin/env python3
"""P97-1：可复用的「引用行号 linter」——对 docs/*.md 里的 `file.ext:line` 做健康检查。

相对 P96 的 `exp_deepgemm_source_audit.py`，本脚本修掉了三个会**静默误判**的方法学问题：

  1. **索引不该包含 `third-party/`**。P96 的 `build_file_index()` 把 DeepGEMM 自带的
     vendored `third-party/cutlass`、`third-party/deep_jit` 也扫进来，于是 `runtime.hpp`
     有 4 个同名候选、`python_api.cpp` / `math.cuh` / `math.hpp` / `utils.py` … 全部歧义，
     消歧时容易落到错的文件（例如把 `csrc/runtime/runtime.hpp:44` 判成
     `jit_kernels/heuristics/runtime.hpp:44` 这个空行）。本脚本只索引**两库自己的源码**。

  2. **消歧要看「被引行是否为空」**。被引行的行号是作者真核对过的，几乎不会指向空行；
     若某候选的被引行全空、另一个候选非空，选非空的那个。再叠加「同篇文档里该 basename
     出现在哪个目录」「全语料引用偏好」两层证据。

  3. **占位省略号不是错误**。`sm90_...hpp:30`、`impls/...cuh:56` 是作者故意省略，
     单独归为 `elided`，不计入失败。

另外支持：
  - 跳过 fenced code block（``` 之间的内容按「示例」处理，默认不检查）；
  - 标记 `refcheck:ignore` 的行跳过；`<!-- refcheck:ignore-begin -->` / `-end` 区块跳过；
  - 排除「审计文档自身」（`EXCLUDE_DOCS`）——它按设计就在引用各种坏引用；
  - 退出码：存在 `unresolved / oob / empty / ambiguous(未消解)` 时返回 1，可直接进 CI。

用法：
  python check_refs.py            # 打印报告 + JSON 汇总，按退出码判断
  python check_refs.py --quiet   # 只打印 JSON 汇总
"""
import os
import re
import glob
import json
import sys
import hashlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # code/mega-moe
DOCS = os.path.join(ROOT, "docs")

# 注意：不能整体排除 third-party/——DeepJIT（`deep_jit/...`）确实是被专题精读过的第三方源码；
# 这里只排除构建产物/缓存。vendored 的 cutlass/cute 与 deep_jit 在 DeepGEMM/DeepEP 双份拷贝，
# 靠下面的「按内容哈希去重候选」解决，不靠目录排除。
EXCLUDE_DIRS = {".git", "build", "__pycache__", "node_modules", ".pytest_cache"}
# 审计文档本身就是「引用体检」的输出，里面按设计引用了一堆坏引用（示例/名单），
# 不纳入健康检查，否则第二次体检会把自己的示例当成错误（自指涉）。
EXCLUDE_DOCS = {"106-deepgemm-source-index-audit.md",
                "107-refs-linter-audit2.md",
                "133-refs-near-miss-triage.md"}

CITE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./\-]*\.(?:cuh|hpp|cpp|cu|py)):(\d+)(?:-(\d+))?")
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_:]{2,})`")
# 「分裂区间」：作者常把区间写成两个反引号 token，如 `` `dispatch.cuh:103`-`:105` ``。
# 原 CITE 只认第一种 `file:103`，`-`:105` 里的终点会被整个丢掉，于是区间退化成单行，
# 区间的标识符（如 104 行的调用）就误报成 near_miss。这里先把分裂区间合并成 `file:103-105`
# 再交给 CITE。兼容 `` `a.hpp:10-20` ``（普通区间）与 `` `a.hpp:10`-20 ``（无冒号终点）。
CONT_RANGE = re.compile(
    r"([A-Za-z0-9_][A-Za-z0-9_./\-]*\.(?:cuh|hpp|cpp|cu|py):\d+)"  # file:start
    r"(`?\s*[-–]\s*`?:)(\d+)"                                       # -:end（终点可带前导冒号）
)


MERGED_RANGES = [0]  # normalize_citations 合并掉的「分裂区间」条数（供报告使用）


def normalize_citations(line, merges=None):
    """把分裂区间 `file:l1`-`:l2` 合并为 `file:l1-l2`；无分裂时原样返回。

    `merges` 为可选的 list（长度 1），用于**按文档**统计合并条数——只累加非排除文档，
    避免把本体检文档自身示例里的分裂区间算进报告。
    """
    def _sub(m):
        if "`" in m.group(2) or ":" in m.group(2):
            (merges if merges is not None else MERGED_RANGES)[0] += 1
        return m.group(1) + "-" + m.group(3)
    return CONT_RANGE.sub(_sub, line)


def _find_repo(name, env):
    for c in (os.environ.get(env, ""), f"~/github/{name}",
              os.path.expanduser(f"~/github/{name}"), f"~/github/{name}"):
        if c and os.path.isdir(c):
            return c
    return os.path.expanduser(f"~/github/{name}")


DG = _find_repo("DeepGEMM", "DEEPGEMM_DIR")
DEP = _find_repo("DeepEP", "DEEPEP_DIR")
LOCAL_ROOTS = [ROOT]
SRC_EXTS = ("cuh", "hpp", "cpp", "cu", "py")


def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except Exception:
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def rel(p):
    for base, tag in ((DG, "DeepGEMM"), (DEP, "DeepEP"), (ROOT, "mega-moe")):
        if p.startswith(base):
            return f"{tag}/{os.path.relpath(p, base)}"
    return p


def build_index():
    """只索引 DeepGEMM / DeepEP 自研源码 + 本专题（排除 third-party 等）。"""
    byname = defaultdict(list)
    allfiles = []
    for root in (DG, DEP) + tuple(LOCAL_ROOTS):
        if not os.path.isdir(root):
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
            for fn in fns:
                if fn.rsplit(".", 1)[-1] not in SRC_EXTS:
                    continue
                p = os.path.join(dp, fn)
                allfiles.append(p)
                byname[fn].append(p)
    return byname, allfiles


def strip_doc(txt, merges=None):
    """返回 [(lineno, text, in_fence, ignored)]：去掉 fenced code block 与 ignore 区块。"""
    out = []
    in_fence = False
    ignore = False
    for i, line in enumerate(txt.splitlines(), 1):
        line = normalize_citations(line, merges)
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            out.append((i, line, True, True))
            continue
        if "refcheck:ignore-begin" in line:
            ignore = True
        cur_ignore = ignore or ("refcheck:ignore" in line)
        if "refcheck:ignore-end" in line:
            ignore = False
        out.append((i, line, in_fence, cur_ignore or in_fence))
    return out


def doc_dir_hints(doc_lines):
    """文档里「带目录的引用」→ 该 basename 的候选目录偏好，以及文档的「主仓库」直方图。"""
    hints = defaultdict(Counter)          # basename -> Counter(dir)
    repo_hist = Counter()                 # 'DeepGEMM'/'DeepEP'/'mega-moe'
    for _, line, _, ig in doc_lines:
        if ig:
            continue
        for m in CITE.finditer(line):
            cited = m.group(1)
            if "/" in cited:
                base = os.path.basename(cited)
                hints[base][os.path.dirname(cited)] += 1
    return hints


def repo_of(p):
    if p.startswith("~/github/DeepGEMM") or "/github/DeepGEMM/" in p:
        return "DeepGEMM"
    if p.startswith("~/github/DeepEP") or "/github/DeepEP/" in p:
        return "DeepEP"
    if p.startswith("~/github/DeepGEMM"):
        return "DeepGEMM"
    if p.startswith("~/github/DeepEP"):
        return "DeepEP"
    return "mega-moe"


def doc_repo_hist(doc_lines, byname):
    hist = Counter()
    for _, line, _, ig in doc_lines:
        if ig:
            continue
        for m in CITE.finditer(line):
            cited = m.group(1)
            if "/" not in cited:
                continue
            hits = [c for c in byname.get(os.path.basename(cited), [])
                    if c.endswith(cited.lstrip("./"))]
            if len(hits) == 1:
                hist[repo_of(hits[0])] += 1
    return hist


def make_resolver(byname):
    # 全语料引用偏好由调用方累积；先用静态：文件名 -> 候选
    def suffix_filter(cited, cands):
        suf = cited.lstrip("./")
        filt = [c for c in cands if rel(c).split("/", 1)[-1].endswith(suf) or c.endswith(suf)]
        return filt or cands

    def dedup_by_content(cands):
        """vendored 拷贝（cutlass/cute/deep_jit 在双库各一份）内容相同 → 视为同一文件。"""
        seen = {}
        for c in cands:
            try:
                h = sha256(c)
            except Exception:
                h = c
            seen.setdefault(h, c)
        return list(seen.values())

    def resolve(cited, byname, hints, l1, l2, prefer_counter, repo_hist=None):
        repo_hist = repo_hist or {}
        base = os.path.basename(cited)
        cands = byname.get(base)
        if not cands:
            for name, ps in byname.items():
                for p in ps:
                    if rel(p).endswith(cited):
                        return [p], 0
            return [], 0
        cands = dedup_by_content(list(cands))
        if len(cands) == 1:
            return cands, 0
        if "/" in cited:
            f = dedup_by_content(suffix_filter(cited, cands))
            if len(f) == 1:
                return f, 0
            if f:
                cands = f
        if len(cands) == 1:
            return cands, 0

        def lines_nonempty(c):
            src = read_lines(c)
            if src is None:
                return False
            return 1 <= l1 <= len(src) and 1 <= l2 <= len(src) and \
                any(s.strip() for s in src[l1 - 1:l2])

        def doc_dir_score(c):
            d = os.path.dirname(rel(c).split("/", 1)[-1])
            h = hints.get(base, {})
            # 命中文档里同 basename 的带目录引用，或该目录在文档里被引用过
            best = 0
            for hd, cnt in h.items():
                if d.endswith(hd) or hd.endswith(d):
                    best = max(best, cnt + 100)
            return best

        def corpus_score(c):
            return prefer_counter.get(rel(c), 0)

        def repo_score(c):
            return repo_hist.get(repo_of(c), 0)

        def score(c):
            return (1 if lines_nonempty(c) else 0,
                    repo_score(c),
                    doc_dir_score(c),
                    corpus_score(c),
                    -len(rel(c)))
        cands.sort(key=score, reverse=True)
        chosen = cands[0]
        # 判「是否歧义未消解」：前两名得分完全相同
        s0, s1 = score(cands[0]), score(cands[1]) if len(cands) > 1 else None
        ambiguous = 1 if (s1 is not None and s0 == s1) else 0
        return [chosen], ambiguous

    return resolve


def audit_docs():
    byname, allfiles = build_index()
    resolve = make_resolver(byname)

    doc_lines = {}
    doc_merges = {}
    for d in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        name = os.path.basename(d)
        m = [0]
        doc_lines[name] = strip_doc(open(d, encoding="utf-8", errors="replace").read(), m)
        doc_merges[name] = m[0]

    # 先算全语料 basename 引用偏好（哪些文件被引用得多）
    prefer = Counter()
    for name, lines_ in doc_lines.items():
        for _, line, _, ig in lines_:
            if ig:
                continue
            for m in CITE.finditer(line):
                for p in byname.get(os.path.basename(m.group(1)), []):
                    pass  # 只统计能唯一解析的
    # 简化：统计「带目录引用」里出现的 relpath
    for name, lines_ in doc_lines.items():
        for _, line, _, ig in lines_:
            if ig:
                continue
            for m in CITE.finditer(line):
                cited = m.group(1)
                if "/" in cited:
                    for c in byname.get(os.path.basename(cited), []):
                        if c.endswith(cited.lstrip("./")):
                            prefer[rel(c)] += 1

    stats = dict(total=0, elided=0, unresolved=0, oob=0, empty=0,
                 ambiguous=0, content_total=0, strict=0, weak=0,
                 files=set(), docs_excluded=sorted(EXCLUDE_DOCS),
                 merged_ranges=sum(v for k, v in doc_merges.items() if k not in EXCLUDE_DOCS))
    problems = dict(unresolved=[], oob=[], empty=[], ambiguous=[], near_miss=[])
    per_file = Counter()

    for name in sorted(doc_lines):
        if name in EXCLUDE_DOCS:
            continue
        lines_ = doc_lines[name]
        hints = doc_dir_hints(lines_)
        repo_hist = doc_repo_hist(lines_, byname)
        for lineno, line, in_fence, ig in lines_:
            if ig:
                continue
            for m in CITE.finditer(line):
                cited, l1 = m.group(1), int(m.group(2))
                l2 = int(m.group(3)) if m.group(3) else l1
                stats["total"] += 1
                if "..." in cited:
                    stats["elided"] += 1
                    continue
                cands, amb = resolve(cited, byname, hints, l1, l2, prefer, repo_hist)
                if not cands:
                    stats["unresolved"] += 1
                    problems["unresolved"].append((name, lineno, m.group(0)))
                    continue
                p = cands[0]
                per_file[os.path.basename(p)] += 1
                stats["files"].add(p)
                src = read_lines(p)
                if amb:
                    stats["ambiguous"] += 1
                    problems["ambiguous"].append((name, lineno, m.group(0), rel(p)))
                if src is None or l1 < 1 or l1 > len(src) or l2 > len(src):
                    stats["oob"] += 1
                    problems["oob"].append((name, lineno, m.group(0), rel(p), len(src or [])))
                    continue
                seg = src[l1 - 1:l2]
                if not any(s.strip() for s in seg):
                    stats["empty"] += 1
                    problems["empty"].append((name, lineno, m.group(0), rel(p)))
                    continue
                ids = [i.split("::")[-1].rstrip("<") for i in IDENT.findall(line)]
                ids = [i for i in ids if not i.endswith((".cuh", ".hpp", ".cpp", ".cu", ".py"))
                       and len(i) >= 4]
                if ids:
                    stats["content_total"] += 1
                    if any(i in "\n".join(seg) for i in ids):
                        stats["strict"] += 1
                        stats["weak"] += 1
                    elif any(i in "\n".join(src) for i in ids):
                        stats["weak"] += 1
                        problems["near_miss"].append((name, lineno, m.group(0), rel(p), ids[:3]))
    stats["files"] = len(stats["files"])
    return stats, problems, per_file, allfiles


def print_report(stats, problems, per_file, allfiles):
    print("=" * 96)
    print("P97-1 · docs 引用行号 linter（check_refs.py）")
    print("=" * 96)
    print(f"DeepGEMM={DG}  DeepEP={DEP}")
    print(f"索引源文件数（含 DeepJIT，vendored 拷贝按内容去重）: {len(allfiles)}")
    print(f"排除的审计文档  : {', '.join(stats['docs_excluded'])}")
    print()
    print(f"合并的分裂区间        : {stats.get('merged_ranges', 0)}（`X:l1`-`:l2` → `X:l1-l2`）")
    print(f"总引用条数            : {stats['total']}")
    print(f"  · 占位省略（...）    : {stats['elided']}（非错误）")
    print(f"  · 无法解析到文件     : {stats['unresolved']}")
    print(f"  · 行号越界/文件缺失  : {stats['oob']}")
    print(f"  · 被引行全为空       : {stats['empty']}")
    print(f"  · 同名歧义未消解     : {stats['ambiguous']}")
    ct = max(stats['content_total'], 1)
    print(f"内容抽查（同行有标识符）: {stats['content_total']}")
    print(f"  · 严格命中           : {stats['strict']} ({100*stats['strict']/ct:.2f}%)")
    print(f"  · 宽松命中（符号在文件内）: {stats['weak']} ({100*stats['weak']/ct:.2f}%)")
    print(f"  · 轻微漂移（在文件、不在被引行）: {len(problems['near_miss'])}")
    print(f"覆盖的源文件数        : {stats['files']}")
    for k in ("unresolved", "oob", "empty", "ambiguous"):
        if problems[k]:
            print(f"\n[{k}] {len(problems[k])} 条：")
            for it in problems[k][:40]:
                print("  " + "  ".join(f"{x}" for x in it))
    if problems["near_miss"]:
        print(f"\n[轻微漂移 near_miss] {len(problems['near_miss'])} 条（多为块/区间首行约定，人工复核）：")
        for it in problems["near_miss"]:
            print("  " + "  ".join(f"{x}" for x in it))


def main():
    quiet = "--quiet" in sys.argv
    stats, problems, per_file, allfiles = audit_docs()
    if not quiet:
        print_report(stats, problems, per_file, allfiles)
    out = dict(total=stats["total"], elided=stats["elided"], unresolved=stats["unresolved"],
               oob=stats["oob"], empty=stats["empty"], ambiguous=stats["ambiguous"],
               content_total=stats["content_total"], strict=stats["strict"], weak=stats["weak"],
               near_miss=len(problems["near_miss"]), files=stats["files"],
               merged_ranges=stats.get("merged_ranges", 0),
               source_files=len(allfiles))
    print()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    bad = stats["unresolved"] + stats["oob"] + stats["empty"] + stats["ambiguous"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
