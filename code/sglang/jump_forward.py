#!/usr/bin/env python3
"""结构化输出：跳词（jump forward）能省多少步——一个**上界估计**。

对应专题第 7 篇（结构化输出）。先读这段背景，否则会把下面的数字当成实测：

    基线 commit 1ef3335 上，SGLang 的跳词实现（constrained/outlines_jump_forward.py）
    **主路径没有调用点**：srt/managers/ 与 srt/speculative/ 下 grep 不到
    try_jump_forward；而 outlines 后端把跳词表直接短路成 None
    （constrained/outlines_backend.py:160-161 的 jump_forward_map = None），
    于是 try_jump_forward 第一行就返回 None。三个后端（xgrammar / llguidance /
    outlines）的接口都在，但没有任何一步生成会走到它。

所以本脚本模拟的是「如果**把**跳词接在生成循环上，能省多少」——是上界，不是实测。

模型：一个 40 个 key 的 JSON 对象骨架。key 名、引号、冒号、逗号、空格全是确定的，
只有每个 value 要采样。为了把账算干净，两处做了简化：

    · key 名算 1 个 token（真实里 field_00 可能切成 2–3 个）
    · value 恰好 1 个 token（真实里 3 位数要 2–3 个）

两处都让「确定 token 占比」偏高，所以步数比偏乐观——这与「上界」的定位一致。

两种模式：

    mask_only     每一步都填一次全词表掩码 + 采样 1 个 token（今天的默认路径）
    jump_forward  在「唯一出边」的状态上沿链折叠 k 个确定 token，一次 extend 追加；
                  只有 value 那个岔路口才填掩码 + 采样

对应真实代码：折叠是 outlines_jump_forward.py 的 JumpEdge / jump_forward_symbol
（沿单出边链拼字符串），一次提交 k 个 token 的代价见文章 §4 的三笔账——
KV 要一次给 k 个 slot、FSM 只能吞合法前缀、max_new_tokens 只能事后截。

用法：
    python3 jump_forward.py                  # 两种模式的账本
    python3 jump_forward.py --keys 6         # 短一点的骨架
    python3 jump_forward.py --vocab 32000    # 换小词表，看掩码字节数
    python3 jump_forward.py --test
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

KEYS = tuple(f"field_{i:02d}" for i in range(40))
VALUE_TOKENS = tuple("0123456789")   # 每个 value 恰好一个 token


# ---------------------------------------------------------------------------
# 骨架：一串「语法位置」
# ---------------------------------------------------------------------------
def skeleton(keys: tuple[str, ...]) -> list[tuple[str, object]]:
    """把 JSON 骨架摊平成位置序列。

    每个位置是 ("forced", token) 或 ("choice", 允许的 token 元组)：

        forced  压缩 FSM 里「唯一出边」的状态——下一步只有一种可能，不需要采样
        choice  岔路口——value 的 10 个数字候选，必须填掩码再采样

    连续的 forced 就是一次跳词能折叠的段（jump_forward_symbol 沿链走出来的那一段）。
    """
    pos: list[tuple[str, object]] = [("forced", "{")]
    for i, k in enumerate(keys):
        if i:
            pos.append(("forced", ","))
            pos.append(("forced", " "))
        pos.extend([
            ("forced", '"'), ("forced", k), ("forced", '"'),
            ("forced", ":"), ("forced", " "),
        ])
        pos.append(("choice", VALUE_TOKENS))
    pos.append(("forced", "}"))
    return pos


@dataclass
class Result:
    mode: str
    forwards: int      # 前向 / 一次 extend 的次数（一个调度步一次）
    samples: int       # 采样次数
    masks: int         # 全词表掩码填充次数
    segments: int      # 折叠出来的确定段数
    values: int        # 采样出来的 value 数
    tokens: int        # 生成的 token 总数（两种模式必须相同）
    text: str

    @property
    def steps(self) -> int:
        return self.forwards


def simulate(mode: str, keys: tuple[str, ...], seed: int = 7) -> Result:
    """两种模式的唯一区别是「谁被折叠」。

    mask_only：每个位置一步，一步 = 一次 forward + 一次全词表掩码 + 一次采样。
    jump_forward：连续 forced 折成一次 extend（无采样、无掩码），岔路口才走一步。
    两边用同一个 seed、且只在 choice 位置取随机数，所以 value 序列逐位相同。
    """
    pos = skeleton(keys)
    rng = random.Random(seed)
    out: list[str] = []
    forwards = samples = masks = segments = values = 0
    i = 0

    if mode == "mask_only":
        while i < len(pos):
            kind, allowed = pos[i]
            masks += 1
            samples += 1
            forwards += 1
            if kind == "choice":
                out.append(rng.choice(allowed))
                values += 1
            else:
                out.append(allowed)
            i += 1
    elif mode == "jump_forward":
        while i < len(pos):
            if pos[i][0] == "forced":
                run: list[str] = []
                while i < len(pos) and pos[i][0] == "forced":
                    run.append(pos[i][1])
                    i += 1
                # 一次 extend 提交整段确定 token
                forwards += 1
                segments += 1
                out.extend(run)
            else:
                allowed = pos[i][1]
                masks += 1
                samples += 1
                forwards += 1
                out.append(rng.choice(allowed))
                values += 1
                i += 1
    else:
        raise ValueError(f"未知模式: {mode}")

    return Result(mode, forwards, samples, masks, segments, values, len(out), "".join(out))


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------
def _bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MiB"
    return f"{n / 1024:.0f} KiB"


def report(args):
    keys = KEYS[: args.keys]
    pos = skeleton(keys)
    n_forced = sum(1 for kind, _ in pos if kind == "forced")
    mask_bytes = ((args.vocab + 31) // 32) * 4          # 打包位掩码：一行 V/32 个 int32
    print(f"骨架：{len(keys)} 个 key、{len(pos)} 个 token（确定 {n_forced} 个，"
          f"占 {n_forced / len(pos):.1%}），value 全部单 token")
    print(f"词表 V={args.vocab:,}，一行掩码 {_bytes(mask_bytes)}（V/32 × 4 B）；"
          f"V 只影响掩码字节数，不影响步数\n")

    rs = [simulate(m, keys, args.seed) for m in ("mask_only", "jump_forward")]
    base = rs[0]
    hdr = (f"{'模式':<14}{'forward':>9}{'采样':>7}{'掩码填充':>9}{'token':>7}"
           f"{'步数比':>8}{'掩码 H2D':>11}{'FSM 判定':>11}")
    print(hdr)
    print("-" * (len(hdr) + 4))
    for r in rs:
        print(f"{r.mode:<14}{r.forwards:>9}{r.samples:>7}{r.masks:>9}{r.tokens:>7}"
              f"{r.forwards / base.forwards:>8.3f}{_bytes(r.masks * mask_bytes):>11}"
              f"{r.masks * args.vocab / 1e6:>10.1f}M")

    jf = rs[1]
    same = rs[0].text == jf.text
    print(f"\n生成结果一致：{same}（逐字节相同；json.loads 后 {len(json.loads(jf.text))} 个 key）")
    print(f"跳词步数 = {jf.segments} 个确定段 + {jf.values} 个 value = {jf.steps} 步；"
          f"mask_only 是 {base.steps} 步（每个 token 一步）")
    print(f"节省：forward {(jf.forwards / base.forwards - 1):+.1%}，"
          f"采样 {(jf.samples / base.samples - 1):+.1%}，"
          f"掩码填充 {(jf.masks / base.masks - 1):+.1%}")

    print(
        "\n读法：\n"
        "  · 跳词省的是**采样与掩码**，不是前向 FLOPs：320 个 token 的 KV 一个都没少算，\n"
        "    只是从 320 次 launch 变成 81 次，其中 41 次是「一次 extend 追加整段」。\n"
        "  · 步数降幅低于「确定 token 占比」：87.5% 的 token 是确定的，但每个确定段仍要\n"
        "    一次 forward，所以只能降到 25%。段越短（key 越多、标点越碎），收益越差。\n"
        "  · 掩码这一列才是文章 §3 的账：mask_only 每步搬一行 16 KiB，jump 只在 value 处搬。\n"
        "    CPU 侧的 FSM 判定同步从 41.9M 次降到 5.2M 次（B=1；B=256 时乘 256）。\n"
        "  · **这是上界**：当前主路径没接线（outlines_backend.py:160-161 把跳词表置 None），\n"
        "    线上真实收益是 0；要接回，还得复用投机解码的 commit_decode 通路（文章 §4）。"
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _test():
    keys = KEYS
    a = simulate("mask_only", keys)
    b = simulate("jump_forward", keys)

    # 1) 两种模式生成的 JSON 完全一致（文本 + 解析结果 + key 数）
    assert a.text == b.text
    assert json.loads(a.text) == json.loads(b.text)
    assert len(json.loads(b.text)) == len(keys)
    assert a.tokens == b.tokens == len(skeleton(keys))

    # 2) jump_forward 的 forward 次数不到 mask_only 的一半
    assert b.forwards * 2 < a.forwards, (b.forwards, a.forwards)

    # 3) 跳词步数 = 确定段数 + value 数；确定段数 = key 数 + 1（最后那个 `}`）
    assert b.steps == b.segments + b.values, (b.steps, b.segments, b.values)
    assert b.segments == len(keys) + 1
    assert b.values == len(keys)
    # mask_only 每个 token 一步，且步数 = 采样 = 掩码
    assert a.steps == a.samples == a.masks == a.tokens
    # jump 只在 value 处采样/填掩码
    assert b.samples == b.masks == b.values

    # 4) 确定性：同参数两次结果相同；换 seed 后两种模式仍然一致，但值变了
    assert simulate("jump_forward", keys).text == b.text
    assert simulate("mask_only", keys).text == a.text
    c = simulate("mask_only", keys, seed=11)
    d = simulate("jump_forward", keys, seed=11)
    assert c.text == d.text and c.text != a.text

    print("jump_forward: all tests passed")
    print(f"  mask_only {a.steps} 步 → jump_forward {b.steps} 步"
          f"（{b.steps / a.steps:.1%}，{a.forwards / b.forwards:.2f}x）")
    print(f"  采样 {a.samples} → {b.samples}，掩码 {a.masks} → {b.masks}，token 数不变（{a.tokens}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--keys", type=int, default=40, help="JSON 对象的 key 数")
    ap.add_argument("--vocab", type=int, default=131072, help="词表大小（只影响掩码字节数）")
    ap.add_argument("--seed", type=int, default=7, help="value 的采样种子")
    args = ap.parse_args()
    if args.test:
        _test()
        return
    if not 1 <= args.keys <= len(KEYS):
        ap.error(f"--keys 必须在 1..{len(KEYS)} 内（骨架只定义了 {len(KEYS)} 个 key）")
    report(args)


if __name__ == "__main__":
    main()
