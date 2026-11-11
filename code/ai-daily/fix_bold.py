#!/usr/bin/env python3
"""修 `doc_include`（readFile | markdownify）里会退化成字面星号的加粗。

坑：`**「甲」**`、`**甲。**`、`**（甲）**` 这类 `**` 包住的内容以 CJK 标点
开头或结尾时，CommonMark 的 flanking 规则判定它不是强调，页面上原样显示 `**`。
主文章正文不受影响，只有被 `doc_include` 内联的 `docs/*.md` 会中招。

修法：把收尾/开头的 CJK 标点挪到 `**` 外面（语义不变），例如
    `**甲。**`      -> `**甲**。`
    `**「甲」**`     -> `「甲」` 两侧标点外移后仍以标点收尾时，退化处理见下
    `**（甲）**`     -> `（**甲**）`
只处理 CJK 标点与全角引号；ASCII 括号不动（外移会改语义），只报警告。

用法：
    python3 fix_bold.py --check content/posts/ai-daily-xxxx/index.md   # 有退化则 exit 1
    python3 fix_bold.py --fix   code/ai-daily/docs/2026-11-11.md       # 就地修
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 这些字符出现在 ** 包住的内容开头/结尾时，会破坏强调解析
CJK_START = "「『（【《〈“‘"
CJK_END = "」』）】》〉”’、，。：；！？…—·"

BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)


def _move_out(inner: str) -> str:
    """把开头/结尾的 CJK 标点挪到 ** 外面；返回替换后的整段（含新 ** 位置）。"""
    core = inner
    pre = post = ""
    while core and core[0] in CJK_START:
        pre += core[0]
        core = core[1:]
    while core and core[-1] in CJK_END:
        post = core[-1] + post
        core = core[:-1]
    if not core.strip():
        return f"**{inner}**"          # 全是标点，不动
    return f"{pre}**{core}**{post}"


def find_bad(text: str) -> list[str]:
    out = []
    for m in BOLD_RE.finditer(text):
        inner = m.group(1)
        if not inner:
            continue
        # 只看 CJK 标点：`**[文字](链接)**` 这种以 ASCII `[` 开头的是合法写法，别误报
        if inner[0] in CJK_START or inner[-1] in CJK_END:
            out.append(inner)
    return out


def fix_text(text: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        inner = m.group(1)
        new = _move_out(inner) if inner else m.group(0)
        if new != m.group(0):
            count += 1
        return new

    return BOLD_RE.sub(repl, text), count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="修 doc_include 里退化的 ** 加粗")
    ap.add_argument("files", nargs="+", help="要检查/修的文件")
    ap.add_argument("--fix", action="store_true", help="就地修改（默认只检查）")
    args = ap.parse_args(argv)

    rc = 0
    for raw in args.files:
        path = Path(raw)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if args.fix:
            new, n = fix_text(text)
            if n:
                path.write_text(new, encoding="utf-8")
                print(f"fixed {n} 处：{path}")
            left = find_bad(new)
            if left:
                print(f"[warn] 仍有 {len(left)} 处需人工处理：{path}")
                for g in left[:5]:
                    print("   ", g[:70])
                rc = 1
        else:
            bad = find_bad(text)
            if bad:
                print(f"[bad] {len(bad)} 处退化的加粗：{path}")
                for g in bad[:5]:
                    print("   ", g[:70])
                rc = 1
    if rc == 0:
        print("bold ok")
    return rc


if __name__ == "__main__":
    sys.exit(main())
