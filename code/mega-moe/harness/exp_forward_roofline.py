"""P132-1 实验：Mega MoE **前向** single-step 的 H100 预算归因（第 15 篇 ncu 反向归因的**对偶**）。

第 15 篇（`docs/15`）用 ncu 把**反向**四个 grouped GEMM 归因清楚了：wgrad 访存受限（AI~120 ≪
拐点 591）、dgrad 计算受限（AI~600）。但**前向**从未被这样量过——第 99 篇（`docs/99`）只给了
一个端到端吞吐（L1/L2 312/303 TFLOPS），没拆「为什么」。本篇补上。

Mega MoE 的前向主体是 SM100 融合 kernel（`sm100_fp8_fp4_mega_moe.cuh`）→ **仅分析**；但它的
**两段 routed GEMM** 与 H100 的 SM90 grouped kernel 是同一数学/同一形状 family：

  linear1 (`L1`)： [m, H] · W1[e]ᵀ → [m, 2I]      m-grouped（M=token，N=2I，K=H）
  SwiGLU + quant：[m, 2I] → [m, I] fp8（per-token UE8M0 gran128）   ← 逐元素，见 src/moe_forward_epilogue.cu
  linear2 (`L2`)： [m, I] · W2[e]ᵀ → [m, H]      m-grouped（M=token，N=H，K=I）

从业界量级取 `T=2048, topk=2, E=8` → routed `m = T*topk = 4096`，每专家 512 行；`H=7168, I=2048`。
前向 L1/L2 的 FLOP 与反向 wgrad 的 w1g/w2g **完全相等**（240.5 / 120.3 GF），但**形状不同**——
于是本篇能直接对照第 15 篇的访存/计算受限结论。

用法（driver 见 exp_forward_roofline.sh）：
    python exp_forward_roofline.py <kind> [--iters N] [--warmup M] [--report]
kind ∈ {l1, l2, both}
--report 时只打印字节/AI/roofline 分析（给 ncu 段用）。
"""
import os
import sys

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              '~/github/DeepGEMM'):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


sys.path.insert(0, os.path.join(_find_deepgemm_dir(), 'tests'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout,
)
from exp_moe_backward_kernels import m_grouped_slices  # noqa: E402

GRAN_K = 128
PEAK_TFLOPS = 1979.0    # H100 SXM FP8 dense tensor-core peak（与第 15 篇同一口径）
HBM_TBPS = 3.35         # H100 HBM 带宽


def build_m_grouped(N, K, counts, device):
    """前向的一段 m-grouped FP8 GEMM：A=激活 [m,K]、B=权重 [E,N,K]、D=bf16 [m,N]。"""
    ma = get_mk_alignment_for_contiguous_layout()
    slices, m_total = m_grouped_slices(counts, ma)
    layout = torch.full((m_total,), -1, device=device, dtype=torch.int32)
    for e, (s, _) in enumerate(slices):
        layout[s:s + counts[e]] = e

    A = torch.randn(m_total, K, device=device, dtype=torch.float32)
    af, asf = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN_K)
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
            (af, asf), (bf, bsf), dm, layout, disable_ue8m0_cast=True)

    # 字节账（主项）：A fp8 + B fp8（E 个专家全触）+ D bf16 + 两侧 SF（fp32）
    bytes_a = m_total * K * 1
    bytes_b = len(counts) * N * K * 1
    bytes_d = m_total * N * 2
    bytes_sf = m_total * (align(K, GRAN_K) // GRAN_K) * 4 + \
        len(counts) * (align(N, GRAN_K) // GRAN_K) * (align(K, GRAN_K) // GRAN_K) * 4
    info = dict(m=m_total, N=N, K=K, E=len(counts), flops=flops,
                bytes_a=bytes_a, bytes_b=bytes_b, bytes_d=bytes_d, bytes_sf=bytes_sf)
    return f, info


def report(kind, info, ms=None):
    tot = info['bytes_a'] + info['bytes_b'] + info['bytes_d'] + info['bytes_sf']
    ai = info['flops'] / tot
    print(f'[{kind}] m={info["m"]} N={info["N"]} K={info["K"]} E={info["E"]}')
    print(f'    FLOP={info["flops"] / 1e9:.1f} GF   bytes: A={info["bytes_a"] / 1e6:.1f}MB '
          f'B={info["bytes_b"] / 1e6:.1f}MB D={info["bytes_d"] / 1e6:.1f}MB '
          f'SF={info["bytes_sf"] / 1e6:.2f}MB  total={tot / 1e6:.1f}MB')
    print(f'    arithmetic intensity = {ai:.0f} FLOP/B  (H100 FP8 拐点 = {PEAK_TFLOPS / HBM_TBPS:.0f})'
          f'  -> {"compute-bound" if ai > PEAK_TFLOPS / HBM_TBPS else "memory-bound"}')
    print(f'    roofline: mem lower bound = {tot / (HBM_TBPS * 1e12) * 1e6:.1f} us   '
          f'compute lower bound = {info["flops"] / (PEAK_TFLOPS * 1e12) * 1e6:.1f} us')
    if ms is not None:
        print(f'    event_time = {ms * 1e3:.1f} us   {info["flops"] / (ms * 1e-3) / 1e12:.0f} TFLOPS  '
              f'({info["flops"] / (ms * 1e-3) / 1e12 / PEAK_TFLOPS * 100:.1f}% of FP8 peak)')


def main():
    kind = sys.argv[1]
    iters = int(sys.argv[sys.argv.index('--iters') + 1]) if '--iters' in sys.argv else 1
    warmup = int(sys.argv[sys.argv.index('--warmup') + 1]) if '--warmup' in sys.argv else 0
    only_report = '--report' in sys.argv
    assert get_arch_major() == 9, 'this experiment targets H100 (sm90)'
    device = 'cuda'

    T, H, I, E, KTOP = 2048, 7168, 2048, 8, 2
    counts = [T * KTOP // E] * E  # 512 each

    builders = {'l1': ('linear1 m_grouped (N=2I, K=H)', lambda: build_m_grouped(2 * I, H, counts, device)),
                'l2': ('linear2 m_grouped (N=H, K=I)', lambda: build_m_grouped(H, I, counts, device))}
    kinds = ['l1', 'l2'] if kind == 'both' else [kind]

    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}')
    print(f'forward shape: T={T} topk={KTOP} E={E} H={H} I={I}  routed m={T * KTOP}')
    print('=' * 96)
    for k in kinds:
        name, builder = builders[k]
        f, info = builder()
        f()  # warm JIT
        torch.cuda.synchronize()
        ms = None
        if not only_report:
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
        print(f'--- {name} ---')
        report(k, info, ms)
        print()


if __name__ == '__main__':
    main()
