#!/usr/bin/env python3
"""53 融合 RoPE：与 model.py 语义对拍 + PyTorch SOTA 口径计时。

用法（在 kernel_lab 容器里）：
    cd code/kernel-opt/53-fused-rope && python3 rope_ref.py
先用 `scripts/run.sh 53-fused-rope/rope.cu dump <scenario>` 生成 dump_*_{meta,in,out}。
"""
import glob
import math
import os

import torch

torch.manual_seed(0)


def build_freq(R, base, orig, factor, beta_fast, beta_slow):
    f = torch.tensor([base ** (-2.0 * i / R) for i in range(R // 2)], dtype=torch.float64)
    if orig > 0:
        def find_dim(nr):
            return R * math.log(orig / (nr * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(find_dim(beta_fast)), 0)
        high = min(math.ceil(find_dim(beta_slow)), R - 1)
        if low >= high:
            high = low + 1
        lin = (torch.arange(R // 2, dtype=torch.float64) - low) / (high - low)
        ramp = lin.clamp(0, 1)
        smooth = 1.0 - ramp
        f = f / factor * (1 - smooth) + f * smooth
    return f


def ref_interleaved(x, freq):
    """x: [..., R] fp64. view_as_complex 语义（DeepSeek model.py）。"""
    T = x.shape[0]
    pos = torch.arange(T, dtype=torch.float64)
    ang = torch.outer(pos, freq)  # [T, R/2]
    c, s = torch.cos(ang), torch.sin(ang)
    xs = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xs[..., 0], xs[..., 1]
    y0 = x0 * c[:, None, :] - x1 * s[:, None, :]
    y1 = x0 * s[:, None, :] + x1 * c[:, None, :]
    return torch.stack([y0, y1], dim=-1).reshape_as(x)


def ref_split(x, freq):
    """HF rotate_half 语义（Qwen3）。"""
    d = x.shape[-1]
    T = x.shape[0]
    pos = torch.arange(T, dtype=torch.float64)
    ang = torch.outer(pos, freq)
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    o1 = x1 * c[:, None, :] - x2 * s[:, None, :]
    o2 = x2 * c[:, None, :] + x1 * s[:, None, :]
    return torch.cat([o1, o2], dim=-1)


def timeit(fn, nbytes, name, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / iters
    print(f"  {name:<34s} {ms:8.4f} ms  {nbytes/ms/1e6:8.1f} GB/s")
    return ms


def check(tag):
    meta = sorted(glob.glob(f"dump_{tag}_meta.txt"))
    if not meta:
        print(f"[skip] no dump for {tag}")
        return
    T, H, HD, R, PAIR, base, factor, orig, bf, bs = open(meta[0]).read().split()
    T, H, HD, R, PAIR, orig, bf, bs = int(T), int(H), int(HD), int(R), int(PAIR), int(orig), int(bf), int(bs)
    base, factor = float(base), float(factor)
    stem = meta[0][:-len("_meta.txt")]
    xin = torch.tensor(bytearray(open(stem + "_in.bin", "rb").read()), dtype=torch.uint8).view(torch.bfloat16)
    xout = torch.tensor(bytearray(open(stem + "_out.bin", "rb").read()), dtype=torch.uint8).view(torch.bfloat16)
    xin = xin.reshape(T, H, HD).to(torch.float64)
    xout = xout.reshape(T, H, HD).to(torch.float64)
    freq = build_freq(R, base, orig, factor, bf, bs)
    rope = xin[..., HD - R:].clone()
    ref = ref_interleaved(rope, freq) if PAIR == 0 else ref_split(rope, freq)
    err = (ref - xout[..., HD - R:]).abs().max().item()
    print(f"=== {tag}: T={T} H={H} HD={HD} R={R} PAIR={'split-half' if PAIR else 'interleaved'} ===")
    print(f"  [check] our kernel vs model.py-fp64: max_abs_err = {err:.3e}  {'OK' if err < 2e-2 else 'FAIL'}")

    N = T * H
    nbytes = N * R * 2 * 2
    dev = "cuda"
    x = xin.to(torch.bfloat16)[..., HD - R:].contiguous().reshape(N, R).to(dev)  # [N,R]
    pos = (torch.arange(N, device=dev, dtype=torch.float64) // H)
    ang = pos[:, None] * freq.to(dev)[None, :]  # [N, R/2]
    c = torch.cos(ang).to(torch.bfloat16)
    s = torch.sin(ang).to(torch.bfloat16)

    if PAIR == 0:
        def t_interleaved():
            a = x.reshape(N, R // 2, 2).to(torch.float32)
            cf = c.to(torch.float32); sf = s.to(torch.float32)
            y0 = a[..., 0] * cf - a[..., 1] * sf
            y1 = a[..., 0] * sf + a[..., 1] * cf
            return torch.stack([y0, y1], -1).reshape(N, R).to(torch.bfloat16)
        timeit(lambda: x.clone(), nbytes, "torch clone (copy roof)")
        timeit(t_interleaved, nbytes, "torch interleaved (model.py)")
    else:
        def t_rot_half():
            x1 = x[..., : R // 2]; x2 = x[..., R // 2:]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        timeit(lambda: x.clone(), nbytes, "torch clone (copy roof)")
        timeit(t_rot_half, nbytes, "torch rotate_half (HF/Qwen3)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA")
    check("ds-mla-q")
    check("qwen3-q")
