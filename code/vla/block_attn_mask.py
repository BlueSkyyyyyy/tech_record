#!/usr/bin/env python3
"""π0 的 block-causal attention mask：把前缀、state、动作块拼成一张图（numpy）.

对应文章：VLA 精读（四）π0 精读。

openpi 的实现只有三行（`src/openpi/models/pi0.py:41-44`）：

    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]

`mask_ar` 为 True 表示“从这个 token 开始一个新块，后面的 token 不能反向依赖它”。
图像和语言 token 的 mask_ar 全是 0 → 前缀内部双向全连接；
state token 的 mask_ar 是 1 → 图像/语言不能看 state；
第一个动作 token 的 mask_ar 是 1，其余动作是 0 → 动作块内部全连接。

π0.5 不一样：state 被塞进 prompt 的字符串里（`tokenizer.py:24-27`），
suffix 里没有 state token，时间步改用 adaRMS 注入。所以 mask 少一行一列。

运行：

    python block_attn_mask.py           # 打印 mask + 前缀缓存的收益
    python block_attn_mask.py --test    # 自测
"""
from __future__ import annotations

import argparse

import numpy as np


def make_attn_mask(input_mask: np.ndarray, ar_mask: np.ndarray) -> np.ndarray:
    """与 openpi `make_attn_mask` 等价：返回 [N, N] 的 bool mask。"""
    input_mask = np.asarray(input_mask, dtype=bool)
    ar_mask = np.asarray(ar_mask, dtype=bool)
    cumsum = np.cumsum(ar_mask)
    attn = cumsum[None, :] <= cumsum[:, None]
    valid = input_mask[None, :] & input_mask[:, None]
    return attn & valid


def pi0_layout(n_img: int = 4, n_lang: int = 3, horizon: int = 5) -> tuple[list[str], np.ndarray, np.ndarray]:
    """返回 (token 名字, input_mask, ar_mask)。"""
    names = [f"img{i}" for i in range(n_img)] + [f"txt{i}" for i in range(n_lang)]
    ar = [0] * n_img + [0] * n_lang
    names += ["state"] + [f"act{i}" for i in range(horizon)]
    ar += [1] + [1] + [0] * (horizon - 1)
    return names, np.ones(len(names), dtype=bool), np.asarray(ar, dtype=bool)


def pi05_layout(n_img: int = 4, n_lang: int = 6, horizon: int = 5) -> tuple[list[str], np.ndarray, np.ndarray]:
    """π0.5：state 已经变成 prompt 里的字符串，suffix 只有动作。"""
    names = [f"img{i}" for i in range(n_img)] + [f"txt{i}" for i in range(n_lang)]
    ar = [0] * (n_img + n_lang)
    names += [f"act{i}" for i in range(horizon)]
    ar += [1] + [0] * (horizon - 1)
    return names, np.ones(len(names), dtype=bool), np.asarray(ar, dtype=bool)


def pretty(mask: np.ndarray, names: list[str]) -> str:
    head = "            " + " ".join(f"{n[:4]:>4}" for n in names)
    lines = [head]
    for i, n in enumerate(names):
        row = " ".join("   #" if mask[i, j] else "   ." for j in range(len(names)))
        lines.append(f"{n:>10}  {row}")
    return "\n".join(lines)


def cache_math(p: int, s: int, steps: int = 10) -> tuple[float, float, float]:
    """粗略的注意力元素计数（只算 QK^T 的矩阵元素，不含乘加）。

    无缓存：每个去噪步都把 prefix+suffix 全算一遍 → steps * (p+s)^2
    有缓存：prefix 只 prefill 一次 p^2，之后每步 suffix 查询 (p+s) → p^2 + steps*s*(p+s)
    """
    naive = steps * (p + s) ** 2
    cached = p**2 + steps * s * (p + s)
    return naive, cached, naive / cached


def report() -> None:
    names, im, ar = pi0_layout()
    mask = make_attn_mask(im, ar)
    print("== π0（state 在 suffix 里）==")
    print("图例：# 可见，. 不可见；行=query，列=key\n")
    print(pretty(mask, names))

    print("\n== π0.5（state 在 prompt 字符串里）==")
    names5, im5, ar5 = pi05_layout()
    mask5 = make_attn_mask(im5, ar5)
    print(pretty(mask5, names5))

    print("\n== 前缀缓存值多少钱（P=prefix token, S=suffix token, 10 步去噪）==")
    print(f"{'P':>6}{'S':>5}{'无缓存(元素)':>16}{'有缓存(元素)':>16}{'加速':>8}")
    for p in (256, 768, 1024, 2048):
        for s in (51,):
            naive, cached, ratio = cache_math(p, s)
            print(f"{p:>6}{s:>5}{naive:>16.3e}{cached:>16.3e}{ratio:>7.1f}x")

    print("\n注：P=768 对应 3 路 224×224 图像、SigLIP patch14 下的 256 token/路。")
    print("S=51 对应 π0 的 1 个 state token + 50 步动作 chunk。")


def _self_test() -> None:
    names, im, ar = pi0_layout()
    mask = make_attn_mask(im, ar)
    idx = {n: i for i, n in enumerate(names)}

    # 前缀内部双向
    assert mask[idx["img0"], idx["img3"]]
    assert mask[idx["img3"], idx["img0"]]
    assert mask[idx["txt0"], idx["img0"]]
    # 前缀看不到 state/动作
    for n in ("state", "act0", "act4"):
        assert not mask[idx["img0"], idx[n]]
        assert not mask[idx["txt2"], idx[n]]
    # state 看不到动作，但能看前缀
    assert mask[idx["state"], idx["txt2"]]
    for n in ("act0", "act4"):
        assert not mask[idx["state"], idx[n]]
    # 动作块内部全连接（并行去噪，不是自回归）
    assert mask[idx["act0"], idx["act4"]]
    assert mask[idx["act4"], idx["act0"]]
    # 动作能看到 state 和前缀
    assert mask[idx["act3"], idx["state"]]
    assert mask[idx["act3"], idx["img1"]]

    # π0.5：suffix 只有动作，动作仍能看全部前缀
    names5, im5, ar5 = pi05_layout()
    m5 = make_attn_mask(im5, ar5)
    i5 = {n: i for i, n in enumerate(names5)}
    assert m5[i5["act0"], i5["txt5"]]
    assert m5[i5["act4"], i5["act0"]]
    assert not m5[i5["txt0"], i5["act0"]]

    # 缓存收益随 P 增大
    for p in (256, 1024):
        naive, cached, ratio = cache_math(p, 51)
        assert cached < naive and ratio > 1
    print("[test] block_attn_mask OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _self_test()
    else:
        report()
