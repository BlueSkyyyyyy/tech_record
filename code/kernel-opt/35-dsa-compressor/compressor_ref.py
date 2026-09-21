#!/usr/bin/env python3
"""35 DSA compressor 的 PyTorch 对照脚本。

三件事：
  1) 忠实复刻 /ssd/models/DeepSeek-V4-Pro/inference/model.py:279 (class Compressor)
     的 prefill 语义（fp32 计算），作为「理论正确性 + eager 基线」。
  2) 单独计时 reference 里两个 fp32 Linear（wkv / wgate）→ 说明「两次独立 GEMM」的成本。
  3) 计时同一个算子用 bf16 合并 GEMM（cuBLAS：X[M,D] @ Wm[2C,D]^T）的成本，
     作为我们自研 proj_ws_kernel 的 cuBLAS 同口径上界。

用法：python3 compressor_ref.py [ratio] [M]
"""
import sys
import time

import torch

D = 7168
HD = 512
RD = 64


def yarn_freqs(seqlen, rope_dim=RD, base=160000.0, factor=16.0,
               original_seq_len=65536.0, beta_fast=32, beta_slow=1):
    import math
    def fdim(num_rot):
        return rope_dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))
    lo = max(math.floor(fdim(beta_fast)), 0)
    hi = min(math.ceil(fdim(beta_slow)), rope_dim - 1)
    if lo == hi:
        hi += 0.001
    ramp = torch.clamp((torch.arange(rope_dim // 2, dtype=torch.float32) - lo) / (hi - lo), 0, 1)
    smooth = 1 - ramp
    freqs = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32)
    return torch.outer(t, freqs)  # [seqlen, rope_dim/2]


def apply_rotary(x, freqs):  # x: [W, RD] last dims, freqs: [W, RD/2]
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(xc * fc).flatten(-2)


def compressor_ref(x, wkv, wgate, ape, nw, ratio, freqs):
    """x[1,M,D] fp32; wkv/wgate[coff*d,D]; ape[ratio,coff*d]; nw[d]; freqs[seqlen,RD/2] -> out[M/ratio, d]"""
    bsz, seqlen, _ = x.shape
    coff, d = wkv.shape[0] // HD, HD
    kv = x @ wkv.T
    score = x @ wgate.T
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + ape
    if ratio == 4:  # overlap
        def overlap_transform(t, value):
            b, s, _, _ = t.size()
            nt = t.new_full((b, s, 2 * ratio, d), value)
            nt[:, :, ratio:] = t[:, :, :, d:]
            nt[:, 1:, :ratio] = t[:, :-1, :, :d]
            return nt
        kv = overlap_transform(kv, 0)
        score = overlap_transform(score, float("-inf"))
    out = (kv * score.softmax(dim=2)).sum(dim=2)  # [1, W, d]
    var = out.square().mean(-1, keepdim=True)
    out = (out * torch.rsqrt(var + 1e-6) * nw).squeeze(0)  # [W, d]
    W = out.size(0)
    fc = freqs[::ratio][:W]  # [W, RD/2] on device
    xc = torch.view_as_complex(out[:, -RD:].float().unflatten(-1, (-1, 2)))
    fc = torch.polar(torch.ones_like(fc), fc)
    out = out.clone()
    out[:, -RD:] = torch.view_as_real(xc * fc).flatten(-2)
    return out


def timed(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ratio = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
    coff = 2 if ratio == 4 else 1
    C = coff * HD
    N = 2 * C
    dev = "cuda"
    torch.manual_seed(0)
    print(f"device={torch.cuda.get_device_name(0)}  ratio={ratio} coff={coff} M={M} N={N} D={D}")

    x = (torch.randn(1, M, D, device=dev) * 0.05)
    wkv = torch.randn(C, D, device=dev) * 0.05
    wgate = torch.randn(C, D, device=dev) * 0.05
    ape = torch.randn(ratio, C, device=dev) * 0.25
    nw = 1 + 0.1 * torch.randn(HD, device=dev)
    freqs = yarn_freqs(M).to(dev)

    # (1) eager 基线（fp32，忠实参考）
    ref = compressor_ref(x, wkv, wgate, ape, nw, ratio, freqs)
    t_ref = timed(lambda: compressor_ref(x, wkv, wgate, ape, nw, ratio, freqs))
    print(f"[eager fp32 reference]      {t_ref:8.4f} ms")

    # (2) 两次独立 fp32 GEMM
    t_two = timed(lambda: (x @ wkv.T, x @ wgate.T))
    print(f"[two fp32 matmul]           {t_two:8.4f} ms")

    # (3) 合并 bf16 GEMM（cuBLAS）：12C,D] @ Wm[2C,D]^T
    xb = x.bfloat16()
    Wm = torch.cat([wkv, wgate], 0).bfloat16()
    t_cublas = timed(lambda: xb @ Wm.T)
    flops = 2.0 * M * N * D
    print(f"[merged bf16 cublas GEMM]   {t_cublas:8.4f} ms  ({flops / t_cublas / 1e9:8.2f} TFLOPS)")

    # (4) 两次独立 bf16 GEMM
    wkvb, wgateb = wkv.bfloat16(), wgate.bfloat16()
    t_two_bf = timed(lambda: (xb @ wkvb.T, xb @ wgateb.T))
    print(f"[two bf16 cublas GEMM]      {t_two_bf:8.4f} ms")


if __name__ == "__main__":
    main()
