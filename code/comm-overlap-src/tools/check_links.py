#!/usr/bin/env python3
"""检查本目录 markdown 里的相对链接是否指向真实存在的文件。

只查相对路径（`./x`、`../x`、`x/y`）与同目录锚点；不查 http(s)、`#anchor`、`{{< >}}` 短代码。
用法：python3 tools/check_links.py
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#")


def md_files() -> list[str]:
    out = []
    for root, _, files in os.walk(BASE):
        if "/.git" in root:
            continue
        for f in files:
            if f.endswith(".md"):
                out.append(os.path.join(root, f))
    return sorted(out)


def main() -> int:
    bad = 0
    for path in md_files():
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # 去掉代码块，避免把示例里的 []() 当链接
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for m in LINK.finditer(text):
            target = m.group(1).strip()
            if target.startswith(SKIP_PREFIX) or "{{<" in target:
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if not os.path.exists(resolved):
                rel = os.path.relpath(path, BASE)
                print(f"[BROKEN] {rel}: {m.group(1)}")
                bad += 1
    print(f"\n检查完成：{'全部有效' if bad == 0 else f'{bad} 个失效链接'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
