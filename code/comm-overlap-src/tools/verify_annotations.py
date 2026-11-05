#!/usr/bin/env python3
"""校验「带注释副本」与上游源码逐行等价。

规则（本目录所有 annotated 文件的唯一硬约束）：
  * 我们只允许**插入整行注释**，不允许改动、删除、重排任何一行代码。
  * 因此：把「整行注释」（去掉前导空白后以 `#` 或 `//` 开头）过滤掉之后，
    副本的有效行序列必须与上游文件**逐行完全相同**。

用法：
    python3 tools/verify_annotations.py              # 校验全部
    python3 tools/verify_annotations.py --list       # 只看清单
    python3 tools/verify_annotations.py --root ~/github   # 指定上游仓库根

退出码 0 = 全部通过。
"""

from __future__ import annotations

import argparse
import os
import py_compile
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "MANIFEST.tsv")

# 上游仓库目录名 -> 相对 ~/github 的位置（可在 MANIFEST 里覆盖）
REPO_DIRS = {
    "triton-distributed": "Triton-distributed",
    "flux": "flux",
}


def is_comment_only(line: str) -> bool:
    s = line.strip()
    return s.startswith("#") or s.startswith("//")


def code_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln for ln in f.read().splitlines() if not is_comment_only(ln)]


def read_manifest() -> list[tuple[str, str, str]]:
    rows = []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            repo, upstream, dest = line.split("\t")
            rows.append((repo, upstream, dest))
    return rows


def check_one(repo: str, upstream_rel: str, dest_rel: str, root: str) -> tuple[bool, str]:
    repo_dir = REPO_DIRS.get(repo)
    if repo_dir is None:
        return False, f"未知仓库标记 {repo!r}"
    upstream = os.path.join(os.path.expanduser(root), repo_dir, upstream_rel)
    dest = os.path.join(BASE, dest_rel)
    if not os.path.exists(upstream):
        return False, f"上游不存在：{upstream}"
    if not os.path.exists(dest):
        return False, f"副本不存在：{dest}"

    up, dn = code_lines(upstream), code_lines(dest)
    if up == dn:
        with open(dest, "r", encoding="utf-8", errors="replace") as f:
            total = len(f.read().splitlines())
        return True, f"{len(up)} 行代码一致（副本 {total} 行，含 {total - len(up)} 行注释）"

    # 定位第一处差异，方便修
    for i, (a, b) in enumerate(zip(up, dn)):
        if a != b:
            return False, (f"第 {i + 1} 行代码不一致\n  上游: {a!r}\n  副本: {b!r}")
    return False, f"行数不一致：上游 {len(up)} 行 vs 副本 {len(dn)} 行"


def check_no_abs_path(dest_rel: str) -> list[str]:
    """仓库禁止出现宿主绝对路径（会泄露用户名）。"""
    bad = []
    dest = os.path.join(BASE, dest_rel)
    with open(dest, "r", encoding="utf-8", errors="replace") as f:
        for no, line in enumerate(f, 1):
            if "/home/" in line or "/ssd/home/" in line:
                bad.append(f"{dest_rel}:{no}: {line.strip()[:100]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/github", help="上游仓库根目录（默认 ~/github）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="只校验 dest 路径包含该子串的文件")
    args = ap.parse_args()

    rows = read_manifest()
    if args.only:
        rows = [r for r in rows if args.only in r[2]]
    if args.list:
        for repo, up, dest in rows:
            print(f"{repo:20s} {up}\n{'':20s} -> {dest}")
        return 0

    failed = 0
    for repo, up, dest in rows:
        ok, msg = check_one(repo, up, dest, args.root)
        flag = "OK  " if ok else "FAIL"
        print(f"[{flag}] {dest}\n       {msg}")
        if not ok:
            failed += 1
            continue
        bad = check_no_abs_path(dest)
        if bad:
            failed += 1
            print("[FAIL] 出现宿主绝对路径（仓库禁止）：")
            for b in bad[:10]:
                print("       " + b)
        if dest.endswith(".py"):
            try:
                with tempfile.TemporaryDirectory() as td:
                    py_compile.compile(os.path.join(BASE, dest), cfile=os.path.join(td, "x.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                failed += 1
                print(f"[FAIL] Python 语法错误：{exc}")

    print(f"\n{len(rows) - failed}/{len(rows)} 个文件通过。")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
