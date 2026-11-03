#!/usr/bin/env python3
"""Latent / DiT / 视频的 token 与算力账本（纯标准库）.

对应文章：扩散模型精读（五）Latent Diffusion、（六）DiT、（九）视频、（十一）推理。

三个容易算错的地方，本脚本逐一对账：

1. **DiT 论文的 118.6 Gflops 是 MACs，不是 FLOPs。** 按 fvcore 的惯例
   （一次乘加记 1），DiT-XL/2 @256 正好 118.6 G；换成「一次乘加 = 2 FLOPs」
   的写法是 237 GFLOPs。混用这两个口径，推理预算会差一倍。
2. **latent 的压缩率是 f^2，不是 f。** f=8 的 VAE 把 512x512x3 压成 64x64x4，
   元素数少 48 倍；token 数少 64 倍。
3. **视频的 token 数是时空三维一起压。** Wan-VAE 的 4x8x8 把 5s/24fps/720p
   的 120 帧压成 30x90x160 = 43.2 万 token；full 3D attention 的注意力矩阵
   按 bf16 存要几百 GB，所以视频模型必然用 factorized / window attention。

运行：

    python latent_budget.py          # 打印四张表
    python latent_budget.py --test   # 自测
"""
from __future__ import annotations

import argparse


def dit_macs(tokens: int, d_model: int, depth: int) -> float:
    """DiT block 的 MACs（乘加次数），口径与 DiT 论文的 Gflops 对齐。

    每个 block：
      - qkv + out proj: 4*N*d^2
      - mlp (4x):        8*N*d^2
      - attention QK/AV: 2*N^2*d
      - adaLN modulation: 6*d^2
    """
    per_block = 12 * tokens * d_model**2 + 2 * tokens**2 * d_model + 6 * d_model**2
    return per_block * depth


def human(x: float) -> str:
    for unit in ["", "K", "M", "G", "T", "P", "E"]:
        if abs(x) < 1000:
            return f"{x:.3g}{unit}"
        x /= 1000
    return f"{x:.3g}Z"


def image_table() -> None:
    print("\n== 图像：像素 vs latent vs token ==")
    print(f"{'分辨率':>10} {'像素元素':>14} {'f=8 latent':>16} {'token(p=1)':>12} {'token(p=2)':>12}")
    for h, w in [(256, 256), (512, 512), (1024, 1024)]:
        px = h * w * 3
        lh, lw = h // 8, w // 8
        lat = lh * lw * 4
        print(f"{h}x{w:<6} {human(px):>14} {f'{lh}x{lw}x4={human(lat)}':>16} "
              f"{human(lh*lw):>12} {human((lh//2)*(lw//2)):>12}")

    print("\n== DiT-XL/2 的 MACs 对账（d=1152, depth=28, patch=2）==")
    print(f"{'输入':>10} {'token':>8} {'MACs':>12} {'=2xFLOPs':>12}")
    for res in [256, 512, 1024]:
        tokens = (res // 8 // 2) ** 2
        macs = dit_macs(tokens, 1152, 28)
        print(f"{res:>10} {tokens:>8} {human(macs)+'MACs':>12} {human(2*macs)+'FLOPs':>12}")
    # DiT 论文报告的 118.6 Gflops（MACs 口径）
    ref = dit_macs(256, 1152, 28) / 1e9
    print(f"256x256 复算: {ref:.1f} G（论文 Table 2: 118.6 G）")


def video_table() -> None:
    print("\n== 视频：5s / 24fps / 720p 的时空压缩 ==")
    frames, h, w = 120, 720, 1280
    print(f"原始帧: {frames} x {h} x {w} x 3 = {human(frames*h*w*3)} 元素")
    for tf, sf, ch, name in [(4, 8, 16, "Wan-VAE 4x8x8, 16ch"),
                             (1, 8, 4, "SVD 1x8x8, 4ch"),
                             (4, 8, 4, "Open-Sora Plan 4x8x8, 4ch")]:
        t, lh, lw = frames // tf, h // sf, w // sf
        tokens = t * lh * lw
        print(f"{name:>26}: {t}x{lh}x{lw} x{ch} = {human(tokens*ch):>8} 元素, "
              f"{human(tokens):>8} token")
    t, lh, lw = frames // 4, h // 8, w // 8
    tokens = t * lh * lw
    print(f"\nfull 3D attention 的注意力矩阵（单头、bf16）:")
    print(f"  {tokens:,} token -> {tokens**2/1e9:.1f}G 元素 = {tokens**2*2/1e9:.0f} GB")
    print(f"  每个 block 的 QK/AV MACs: {human(2*tokens**2*1152)}（d=1152 时）")
    print("  -> 必须 factorized（先空间后时间）或 window / 稀疏 attention")


def latency_table() -> None:
    print("\n== 推理延迟的纸面账（假设 H100 bf16 990 TFLOPs，MFU 35%）==")
    peak_tflops, mfu = 990.0, 0.35
    print(f"{'配置':>34} {'每步FLOPs':>12} {'30步+CFG':>12} {'纸面':>10}")
    cases = [
        ("DiT-XL/2 256, 50 steps, CFG", dit_macs(256, 1152, 28), 50),
        ("DiT 1024 (p=2, d=1152, 28L), 30 steps, CFG", dit_macs(4096, 1152, 28), 30),
        ("DiT 1024 (p=1, d=1152, 28L), 30 steps, CFG", dit_macs(16384, 1152, 28), 30),
    ]
    for name, macs, steps in cases:
        flops = 2 * macs * steps * 2  # x2: CFG 双 batch
        sec = flops / (peak_tflops * 1e12 * mfu)
        print(f"{name:>34} {human(2*macs)+'F':>12} {human(flops)+'F':>12} {sec:>8.2f}s")
    print("注：只算去噪主干，不含 VAE 解码 / 文本编码 / 显存搬运；真实值通常再慢 2-4 倍。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    image_table()
    video_table()
    latency_table()

    if args.test:
        ref = dit_macs(256, 1152, 28) / 1e9
        assert abs(ref - 118.6) < 1.0, f"DiT 复算应接近 118.6 G，实际 {ref:.2f}"
        # 1024 全分辨率 attention 的二次项必须已经压过线性项
        assert 2 * 16384**2 * 1152 > 12 * 16384 * 1152**2
        # 视频 full attention 内存必须到几百 GB 量级
        tokens = 30 * 90 * 160
        assert tokens**2 * 2 / 1e9 > 100
        print("\n[test] all assertions passed")


if __name__ == "__main__":
    main()
