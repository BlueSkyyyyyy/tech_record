#!/usr/bin/env python3
"""Codex 风格 apply_patch 的最小实现：语法、模糊定位、失败模式。

对应专题第 5 篇《Harness 解剖（一）：Codex CLI》。

用法：
    python3 apply_patch.py --demo
    python3 apply_patch.py --test

只依赖标准库。它不是 OpenAI apply_patch 的完整移植（完整版在
codex-rs/apply-patch，约 1400 行），但保留了三个关键设计：

1. 补丁是"带上下文的文本"，不是 JSON。模型写起来 token 少，错误也少。
2. 定位是四段式模糊匹配：精确 → 去行尾空白 → 去两端空白 → Unicode 标点归一。
   最后一段让 `-` 能匹配文件里的 `—`，和 `git apply` 的行为类似。
3. 应用不是原子的：多个 hunk 顺序执行，中途失败会留下已写入的部分。
   所以 harness 必须把"失败 + 已改到哪"一起回给模型，而不是简单报错。
"""

from __future__ import annotations

import argparse
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

BEGIN = "*** Begin Patch"
END = "*** End Patch"


@dataclass
class Chunk:
    context: str | None
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    eof: bool = False


@dataclass
class Hunk:
    kind: str  # add | delete | update
    path: str
    lines: list[str] = field(default_factory=list)  # add 用
    move_to: str | None = None
    chunks: list[Chunk] = field(default_factory=list)


class PatchError(Exception):
    pass


def parse(text: str) -> list[Hunk]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != BEGIN:
        raise PatchError("first line must be '*** Begin Patch'")
    if not lines or lines[-1].strip() != END:
        raise PatchError("last line must be '*** End Patch'")
    body = lines[1:-1]
    hunks: list[Hunk] = []
    i = 0
    while i < len(body):
        line = body[i]
        if line.startswith("*** Add File: "):
            h = Hunk("add", line[len("*** Add File: "):])
            i += 1
            while i < len(body) and body[i].startswith("+"):
                h.lines.append(body[i][1:])
                i += 1
            if not h.lines:
                raise PatchError(f"{h.path}: add file needs at least one + line")
            hunks.append(h)
        elif line.startswith("*** Delete File: "):
            hunks.append(Hunk("delete", line[len("*** Delete File: "):]))
            i += 1
        elif line.startswith("*** Update File: "):
            h = Hunk("update", line[len("*** Update File: "):])
            i += 1
            if i < len(body) and body[i].startswith("*** Move to: "):
                h.move_to = body[i][len("*** Move to: "):]
                i += 1
            while i < len(body) and not body[i].startswith("*** "):
                if body[i].startswith("@@"):
                    ctx = body[i][2:].strip() or None
                    ch = Chunk(ctx)
                    i += 1
                    while i < len(body) and body[i][:1] in {" ", "+", "-", "***"}:
                        s = body[i]
                        if s.startswith("*** End of File"):
                            ch.eof = True
                            i += 1
                            break
                        if s.startswith("*** "):
                            break
                        if s.startswith(" "):
                            ch.old.append(s[1:])
                            ch.new.append(s[1:])
                        elif s.startswith("-"):
                            ch.old.append(s[1:])
                        elif s.startswith("+"):
                            ch.new.append(s[1:])
                        i += 1
                    if not ch.old and not ch.new:
                        raise PatchError(f"{h.path}: empty chunk")
                    h.chunks.append(ch)
                else:
                    i += 1
            hunks.append(h)
        else:
            i += 1
    if not hunks:
        raise PatchError("no hunks")
    return hunks


def _normalise(s: str) -> str:
    """把常见印刷体标点折叠成 ASCII；这是第四段匹配。"""
    table = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u00a0": " ",
    }
    s = "".join(table.get(c, c) for c in s)
    return unicodedata.normalize("NFKC", s).strip()


def seek(lines: list[str], pattern: list[str], start: int, eof: bool = False) -> int | None:
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None
    begin = max(start, len(lines) - len(pattern)) if eof else start
    passes = [
        lambda a, b: a == b,
        lambda a, b: a.rstrip() == b.rstrip(),
        lambda a, b: a.strip() == b.strip(),
        lambda a, b: _normalise(a) == _normalise(b),
    ]
    for eq in passes:
        for i in range(begin, len(lines) - len(pattern) + 1):
            if all(eq(lines[i + j], pattern[j]) for j in range(len(pattern))):
                return i
    return None


def apply_hunk(h: Hunk, root: Path) -> str:
    p = root / h.path
    if h.kind == "add":
        if p.exists():
            raise PatchError(f"{h.path}: already exists")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(h.lines) + "\n")
        return f"A {h.path}"
    if h.kind == "delete":
        if not p.exists():
            raise PatchError(f"{h.path}: not found")
        p.unlink()
        return f"D {h.path}"
    if not p.exists():
        raise PatchError(f"{h.path}: not found")
    text = p.read_text()
    lines = text.splitlines()
    pos = 0
    for ch in h.chunks:
        at = seek(lines, ch.old, pos, ch.eof)
        if at is None:
            # 把失败上下文带回去，模型才知道下一刀往哪切
            raise PatchError(f"failed to find context {ch.old[:1]!r} in {h.path}")
        lines[at:at + len(ch.old)] = ch.new
        pos = at + len(ch.new)
    out = "\n".join(lines)
    if text.endswith("\n"):
        out += "\n"
    if h.move_to:
        dst = root / h.move_to
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(out)
        p.unlink()
        return f"R {h.path} -> {h.move_to}"
    p.write_text(out)
    return f"M {h.path}"


def apply_patch(text: str, root: Path) -> list[str]:
    return [apply_hunk(h, root) for h in parse(text)]


DEMO = """*** Begin Patch
*** Add File: hello.py
+def hello():
+    return "hi"
*** Update File: hello.py
@@ def hello():
-    return "hi"
+    return "hello"
*** End Patch
"""


def demo() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        changes = apply_patch(DEMO, root)
        print("changes:", changes)
        print((root / "hello.py").read_text(), end="")


def test() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.py").write_text("def f():\n    return 1\n")
        # 精确匹配
        apply_patch("*** Begin Patch\n*** Update File: a.py\n@@\n-    return 1\n+    return 2\n*** End Patch\n", root)
        assert "return 2" in (root / "a.py").read_text()

        # 行尾空白 + Unicode 标点：第四段匹配
        (root / "b.py").write_text("x = \u201cold\u201d  \n")
        apply_patch("*** Begin Patch\n*** Update File: b.py\n@@\n-x = \"old\"\n+x = \"new\"\n*** End Patch\n", root)
        assert (root / "b.py").read_text().startswith('x = "new"')

        # add + delete + move
        apply_patch("*** Begin Patch\n*** Add File: c.py\n+print(1)\n*** End Patch\n", root)
        assert (root / "c.py").exists()
        apply_patch("*** Begin Patch\n*** Delete File: c.py\n*** End Patch\n", root)
        assert not (root / "c.py").exists()

        # 找不到上下文必须报错，且错误里带上下文
        try:
            apply_patch("*** Begin Patch\n*** Update File: a.py\n@@\n-nope\n+x\n*** End Patch\n", root)
        except PatchError as e:
            assert "failed to find context" in str(e)
        else:
            raise AssertionError("expected PatchError")

        # 非原子：第一个文件的 hunk 成功、第二个文件失败时，第一个已经落盘
        (root / "d.py").write_text("one\ntwo\n")
        (root / "e.py").write_text("alpha\n")
        try:
            apply_patch(
                "*** Begin Patch\n*** Update File: d.py\n@@\n-one\n+ONE\n"
                "*** Update File: e.py\n@@\n-MISSING\n+X\n*** End Patch\n",
                root,
            )
        except PatchError:
            assert (root / "d.py").read_text().startswith("ONE"), "partial write is observable"
        else:
            raise AssertionError("expected partial failure")
    print("apply_patch: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        test()
    else:
        demo()


if __name__ == "__main__":
    main()
