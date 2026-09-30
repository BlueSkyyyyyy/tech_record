#!/usr/bin/env python3
"""LeetCode 配套代码的统一自测入口。

每个 Python 文件在 `if __name__ == "__main__":` 里用 assert 自测；
每个 C++ 文件编译后运行、以退出码 0 表示通过。

用法：
    python3 scripts/run_all.py              # 跑全部（Python + C++）
    python3 scripts/run_all.py --lang py    # 只跑 Python
    python3 scripts/run_all.py --lang cpp   # 只跑 C++
    python3 scripts/run_all.py array        # 只跑路径含 "array" 的文件
    python3 scripts/run_all.py --list       # 只列出会跑哪些文件
退出码：全部通过为 0，否则为 1。
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def find_files(ext):
    out = []
    for dirpath, _dirs, files in os.walk(SRC):
        for name in sorted(files):
            if name.endswith(ext):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def run_python(path):
    proc = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, cwd=ROOT
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_cpp(path, tmpdir):
    exe = os.path.join(tmpdir, os.path.splitext(os.path.basename(path))[0])
    comp = subprocess.run(
        ["g++", "-O2", "-std=c++17", "-o", exe, path],
        capture_output=True,
        text=True,
    )
    if comp.returncode != 0:
        return comp.returncode, "", "compile failed:\n" + comp.stderr
    proc = subprocess.run([exe], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="", help="只跑路径包含该子串的文件")
    ap.add_argument("--lang", choices=["py", "cpp", "all"], default="all")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    targets = []
    if args.lang in ("py", "all"):
        targets += [("py", p) for p in find_files(".py")]
    if args.lang in ("cpp", "all"):
        targets += [("cpp", p) for p in find_files(".cpp")]
    if args.filter:
        targets = [(k, p) for k, p in targets if args.filter in p]

    if not targets:
        print("没有找到匹配的测试文件")
        return 1

    if args.list:
        for kind, path in targets:
            print(f"[{kind}] {os.path.relpath(path, ROOT)}")
        return 0

    passed, failed = 0, []
    with tempfile.TemporaryDirectory() as tmpdir:
        for kind, path in targets:
            rel = os.path.relpath(path, ROOT)
            if kind == "py":
                rc, out, err = run_python(path)
            else:
                rc, out, err = run_cpp(path, tmpdir)
            if rc == 0:
                passed += 1
                print(f"PASS  {rel}" + (f"  ({out.strip()})" if out.strip() else ""))
            else:
                failed.append(rel)
                print(f"FAIL  {rel}  (rc={rc})")
                if err.strip():
                    print("      " + err.strip().replace("\n", "\n      "))

    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        print("失败文件：")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
