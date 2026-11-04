#!/usr/bin/env python3
"""GUI grounding 的坐标账：归一化、缩放、点击命中率。

对应专题第 8 篇《Computer-use 与浏览器 Agent》。

用法：
    python3 gui_ground.py
    python3 gui_ground.py --width 2560 --height 1440
    python3 gui_ground.py --test

只依赖标准库。GUI agent 的动作空间和文本 agent 不同：它输出的是
(x, y)、click、type、scroll、drag，而不是 JSON 参数。坐标错 20 像素，
就可能点到隔壁按钮。

脚本算三件事：

1. 模型输出 0-1000 归一化坐标时，映射到真实分辨率会放大多少误差；
2. 截图被缩放到模型输入尺寸（比如 1280 宽）再映射回去，误差是多少；
3. 给定点击点和目标框，命中率 / IoU 怎么算。
"""

from __future__ import annotations

import argparse


def norm_to_pixels(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """0-1000 归一化坐标 → 像素坐标。这是 Qwen-VL / UI-TARS 一类模型的常见约定。"""
    px = round(x / 1000 * width)
    py = round(y / 1000 * height)
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def pixels_to_norm(px: int, py: int, width: int, height: int) -> tuple[float, float]:
    return px / width * 1000, py / height * 1000


def resize_error(width: int, height: int, model_width: int = 1280) -> tuple[float, float, float]:
    """截图缩放到 model_width 再映射回原图，单像素误差被放大多少倍。"""
    scale = model_width / width
    model_height = round(height * scale)
    # 模型在缩放图上错 1 像素，映射回原图是 1/scale 像素
    return 1 / scale, model_height, scale


def hit(click: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    x, y = click
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def report(width: int, height: int) -> None:
    print(f"screen: {width}x{height}")
    err, mh, scale = resize_error(width, height)
    print(f"resize to 1280 wide: scale={scale:.3f}, model height={mh}, 1px error -> {err:.2f}px on screen")
    print(f"normalized 0-1000: 1 unit -> {width / 1000:.2f}px horizontally, {height / 1000:.2f}px vertically")
    # 一个 40x24 的按钮，在 1000 归一化下大约占 16x17 个单位
    box = (400, 300, 440, 324)
    for name, (nx, ny) in [("center", (420, 312)), ("edge", (440, 324)), ("off-by-8", (448, 312))]:
        px, py = norm_to_pixels(nx / width * 1000, ny / height * 1000, width, height)
        print(f"  {name:<10} -> ({px},{py}) hit={hit((px, py), box)}")
    print(f"  IoU(box, box shifted 8px) = {iou(box, (408, 300, 448, 324)):.3f}")


def test() -> None:
    # 归一化边界
    assert norm_to_pixels(0, 0, 1920, 1080) == (0, 0)
    assert norm_to_pixels(1000, 1000, 1920, 1080) == (1919, 1079)
    assert norm_to_pixels(500, 500, 1000, 1000) == (500, 500)
    # 往返
    px, py = norm_to_pixels(420, 312, 1920, 1080)
    nx, ny = pixels_to_norm(px, py, 1920, 1080)
    assert abs(nx - 420) < 1 and abs(ny - 312) < 1
    # 缩放误差：4K 缩到 1280，单像素误差放大 3 倍
    err, _, scale = resize_error(3840, 2160, 1280)
    assert abs(scale - 1 / 3) < 1e-9 and abs(err - 3) < 1e-9
    # 命中与 IoU
    assert hit((420, 312), (400, 300, 440, 324))
    assert not hit((448, 312), (400, 300, 440, 324))
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-12
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    print("gui_ground: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    test() if args.test else report(args.width, args.height)


if __name__ == "__main__":
    main()
