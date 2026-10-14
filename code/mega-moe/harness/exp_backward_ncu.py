"""反向性能归因实验（承接 P4-4）：用 ncu 解释 wgrad ~100 vs dgrad ~450 TFLOPS。

P4-4 在 H100 上量到四个 grouped GEMM 的吞吐差 4~5 倍（wgrad ~100 / dgrad ~450 TFLOPS），
但只报数字、没解释硬件原因。本实验把每个 kernel **单独** 跑（一个进程一个 kernel 类型），
交给 `ncu` 采集 SpeedOfLight / Occupancy / LaunchStats / Memory / Compute 五个 section，
拿到「为什么」的硬指标：tensor pipe 利用率、DRAM 字节量、算术强度、占用率、寄存器/smem。

用法（driver 见 exp_backward_ncu.sh）：
    python exp_backward_ncu.py <kind> [--iters N] [--warmup M]
kind ∈ {w2g, w1g, d2g, d1g}：
    w2g = linear2 wgrad `k_grouped`  (M=I,   N=H,  K=token)
    w1g = linear1 wgrad `k_grouped`  (M=H,   N=2I, K=token)
    d2g = linear2 dgrad `m_grouped`  (m=token, N=I, K=H)
    d1g = linear1 dgrad `m_grouped`  (m=token, N=H, K=2I)
默认 iters=1 / warmup=0：ncu 自己 replay，脚本内不重复。
形状沿用 P4-4 的 DeepSeek 单 rank 量级（T=2048,H=7168,I=2048,E=8,topk=2）。
"""
import os
import sys

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              os.path.expanduser('~/github/DeepGEMM')):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


sys.path.insert(0, os.path.join(_find_deepgemm_dir(), 'tests'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout,
)
from exp_moe_backward_kernels import quant_k_operand, m_grouped_slices  # noqa: E402

GRAN_K = 128


def build_wgrad(M, N, counts, ka, device):
    host_ks = [align(k, ka) for k in counts]
    total_k = sum(host_ks)
    A = torch.randn(total_k, M, device=device, dtype=torch.float32)
    B = torch.randn(total_k, N, device=device, dtype=torch.float32)
    a = quant_k_operand(A, host_ks)
    b = quant_k_operand(B, host_ks)
    layout = torch.tensor(host_ks, device=device, dtype=torch.int32)
    d = torch.zeros(len(counts), M, N, device=device, dtype=torch.float32)
    flops = 2 * M * N * total_k

    def f():
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (a[0], a[1]), (b[0], b[1]), d, host_ks, layout, d, recipe=(1, 1, ka))

    return f, flops, dict(M=M, N=N, sum_k=total_k, host_ks=host_ks)


def build_dgrad(N, K, counts, device):
    ma = get_mk_alignment_for_contiguous_layout()
    slices, m_total = m_grouped_slices(counts, ma)
    layout_m = torch.full((m_total,), -1, device=device, dtype=torch.int32)
    for e, (s, _) in enumerate(slices):
        layout_m[s:s + counts[e]] = e
    Am = torch.randn(m_total, K, device=device, dtype=torch.float32)
    af, asf = per_token_cast_to_fp8(Am, use_ue8m0=False, gran_k=GRAN_K)
    Bg = torch.randn(len(counts), N, K, device=device, dtype=torch.float32)
    bf = torch.empty(len(counts), N, K, dtype=torch.float8_e4m3fn, device=device)
    bsf = torch.empty(len(counts), align(N, GRAN_K) // GRAN_K, align(K, GRAN_K) // GRAN_K,
                      device=device, dtype=torch.float32)
    for e in range(len(counts)):
        bf[e], bsf[e] = per_block_cast_to_fp8(Bg[e], use_ue8m0=False, gran_k=GRAN_K)
    dm = torch.empty(m_total, N, device=device, dtype=torch.bfloat16)
    flops = 2 * m_total * N * K

    def f():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (af, asf), (bf, bsf), dm, layout_m, disable_ue8m0_cast=True)

    return f, flops, dict(m=m_total, N=N, K=K, slices=slices)


def main():
    kind = sys.argv[1]
    iters = int(sys.argv[sys.argv.index('--iters') + 1]) if '--iters' in sys.argv else 1
    warmup = int(sys.argv[sys.argv.index('--warmup') + 1]) if '--warmup' in sys.argv else 0

    assert get_arch_major() == 9, 'this experiment targets H100 (sm90)'
    device = 'cuda'

    T, H, I, E, KTOP = 2048, 7168, 2048, 8, 2
    counts = [T * KTOP // E] * E  # 512 each
    ka = GRAN_K

    if kind == 'w2g':
        f, flops, info = build_wgrad(I, H, counts, ka, device)
    elif kind == 'w1g':
        f, flops, info = build_wgrad(H, 2 * I, counts, ka, device)
    elif kind == 'd2g':
        f, flops, info = build_dgrad(I, H, counts, device)
    elif kind == 'd1g':
        f, flops, info = build_dgrad(H, 2 * I, counts, device)
    else:
        raise SystemExit(f'unknown kind {kind!r}')

    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        f()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / iters
    if os.environ.get('SKIP_EVENT_PRINT'):
        print(f'[{kind}] {info} theoretical_flops={flops / 1e9:.1f} GFLOP')
    else:
        print(f'[{kind}] {info} theoretical_flops={flops / 1e9:.1f} GFLOP '
              f'event_time={ms * 1e3:6.1f} us  {flops / (ms * 1e-3) / 1e12:5.0f} TFLOPS')


if __name__ == '__main__':
    main()
