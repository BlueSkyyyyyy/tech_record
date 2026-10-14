#!/usr/bin/env python3
"""P142-1 实验：专家负载不均（token 分布偏斜）对 MoE grouped GEMM 计算侧的代价。

MoE 的 dgrad / 前向 L2 都是 M 轴分组的 grouped GEMM：不等长的是「每个专家收到的 token 数」。
第 4/13 篇讲了三种布局（contiguous / masked / PSUM）的**语义**，第 142 篇量了**均衡**分布下的前向
roofline，但从没量过：**当 token 分布偏斜时，grouped GEMM 到底慢多少、慢在哪里**。

本篇在 H100（sm90）上用真实 DeepGEMM FP8 kernel（linear2 dgrad 形状）对照三档：
  - NAIVE：把每个专家都补到 max_m 的稠密按专家 bmm（torch.bmm `[G,max_m,k]@[G,k,n]`），
           算满 G·max_m 行 —— 「不用 grouped kernel」时的朴素做法；
  - MASKED：`m_grouped_fp8_fp4_gemm_nt_masked`，`[G,max_m,·]` 稠密存储 + device 端 `masked_m`，
            调度器**只认领每个组 ceil(m_g/BLOCK_M) 个 M 块**（`scheduler/gemm.cuh:188-225`）；
  - CONTIGUOUS：`m_grouped_fp8_fp4_gemm_nt_contiguous`，各专家拼接、每段向上对齐到
            `mk_alignment`(=128)、gap 行 group=-1，调度器按整池 M 块领取并跳过 gap
            （`scheduler/gemm.cuh:151-156/296-309`）。

去偏斜扫描用 Dirichlet 集中度 alpha（越小越偏斜，alpha→∞ 即均衡）。量：
  A. 纯 **device kernel 时长**（torch.profiler）vs 偏斜（三档）+ 有效 FLOPs 利用率；
  B. 记帐：contiguous 物理行 = Σ align(m_g,128)（含 gap）、masked 存储行 = G·max_m、有效行 = M_total；
  C. 正确性：同一份数据下 masked 有效行与 contiguous 有效行**逐位相等**，并 vs bf16 参考；
  D. G=32（更接近单 rank 本地专家数）与两种极端分布（one-heavy / power-law）。

注意：本机是 8 卡共享节点，必须绑到空闲 GPU 才稳定（`exp_grouped_skew.sh` 自动选 util 最低的卡）。
输出重定向到 exp_grouped_skew.out.txt。
"""
import os
import sys
import traceback
import warnings

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, bench, calc_diff

def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              os.path.expanduser('~/github/DeepGEMM')):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


DEEPGEMM_DIR = _find_deepgemm_dir()
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from generators import (  # noqa: E402
    MajorTypeAB, cast_fp8_fp4_with_major, grouped_cast_fp8_fp4_with_major,
)

ALIGN = 128
warnings.filterwarnings('ignore', message='.*Profiler clears events.*')


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def reset_align():
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())


def align_up(x, a):
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------- timing
def kernel_time(fn, iters=30, warmup=10):
    """纯 device kernel 时长（torch.profiler 取 `sm90_fp8_gemm_1d2d_impl` 的 device_time）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for ev in prof.key_averages():
        if 'sm90_fp8_gemm_1d2d_impl' in ev.key:
            tot += ev.device_time_total / iters  # device_time_total 单位是 us
    return tot / 1e6


# --------------------------------------------------------------------------- build
def build(G, ms, n, k, seed=0):
    """给定每组有效 token 数 ms，构造三种布局（同一份数据）。"""
    torch.manual_seed(seed)
    aligned = [align_up(m, ALIGN) for m in ms]
    M = sum(aligned)
    max_m = max(aligned)

    groups = [torch.randn((m, k), device='cuda', dtype=torch.bfloat16) for m in ms]
    b_bf = torch.randn((G, n, k), device='cuda', dtype=torch.bfloat16)

    a_c = torch.zeros((M, k), device='cuda', dtype=torch.bfloat16)
    layout = torch.full((M,), -1, device='cuda', dtype=torch.int32)
    valid_c = torch.zeros(M, device='cuda', dtype=torch.bool)
    pos = 0
    for i, (m, al) in enumerate(zip(ms, aligned)):
        a_c[pos:pos + m] = groups[i]
        layout[pos:pos + m] = i
        valid_c[pos:pos + m] = True
        pos += al

    a_m = torch.zeros((G, max_m, k), device='cuda', dtype=torch.bfloat16)
    for i, m in enumerate(ms):
        a_m[i, :m] = groups[i]

    a_c_fp8 = cast_fp8_fp4_with_major(a_c, MajorTypeAB.KMajor, 128, False, use_ue8m0=False)
    a_m_fp8 = grouped_cast_fp8_fp4_with_major(a_m, MajorTypeAB.KMajor, 128, False, use_ue8m0=False)
    for j, m in enumerate(ms):
        a_m_fp8[1][j, m:] = 0
    b_fp8 = grouped_cast_fp8_fp4_with_major(b_bf, MajorTypeAB.KMajor, 128, False,
                                            use_ue8m0=False, use_block_cast_for_fp8=True)

    d_c = torch.empty((M, n), device='cuda', dtype=torch.bfloat16)
    d_m = torch.empty((G, max_m, n), device='cuda', dtype=torch.bfloat16)
    masked_m = torch.tensor(ms, device='cuda', dtype=torch.int)

    return dict(aligned=aligned, M=M, max_m=max_m, M_total=sum(ms),
                a_c=a_c_fp8, b=b_fp8, layout=layout, d_c=d_c, valid_c=valid_c,
                a_m=a_m_fp8, masked_m=masked_m, d_m=d_m,
                a_c_bf=a_c, a_m_bf=a_m, b_bf=b_bf)


def run_contiguous(dat):
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        dat['a_c'], dat['b'], dat['d_c'], dat['layout'], disable_ue8m0_cast=True)


def run_masked(dat, expected_m):
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        dat['a_m'], dat['b'], dat['d_m'], dat['masked_m'], expected_m, disable_ue8m0_cast=True)


def time_naive(a_m_bf, b_bf):
    return bench(lambda: torch.bmm(a_m_bf, b_bf.transpose(1, 2)), num_warmups=10, num_tests=30)


def measure(dat, expected_m):
    tc = kernel_time(lambda: run_contiguous(dat))
    tm = kernel_time(lambda: run_masked(dat, expected_m))
    tn = time_naive(dat['a_m_bf'], dat['b_bf'])
    return tc, tm, tn


# --------------------------------------------------------------------------- parts
def print_row(label, dat, tc, tm, tn, n, k):
    flop = 2 * dat['M_total'] * n * k
    sum_al = sum(dat['aligned'])
    print(f'{label:>16} {dat["max_m"]:>6} {sum_al / dat["M_total"]:>7.3f} '
          f'{len(dat["masked_m"]) * dat["max_m"] / dat["M_total"]:>8.3f} '
          f'{tc * 1e6:>7.0f}u {tm * 1e6:>7.0f}u {tn * 1e6:>7.0f}u '
          f'{flop / tc / 1e12:>8.0f} {flop / tm / 1e12:>8.0f} {flop / tn / 1e12:>8.0f} '
          f'{tm / tc:>8.2f} {tn / tm:>9.2f}')


HEADER = (f'{"case":>16} {"max_m":>6} {"Σal/M":>7} {"G·max/M":>8} '
          f'{"t_cont":>8} {"t_mask":>8} {"t_naive":>8} '
          f'{"cont_TF":>8} {"mask_TF":>8} {"naive_TF":>8} {"mask/cont":>8} {"naive/mask":>9}')


def part_A(G, n, k, M_total):
    print('=' * 104)
    print(f'A. 时间 vs 偏斜（G={G} n={n} k={k} M_total={M_total}，平均每专家 {M_total // G}）'
          f'；alpha=Inf 均衡，越小越偏斜')
    print('=' * 104)
    print(HEADER)
    rows = []
    torch.manual_seed(0)  # 固定 Dirichlet 抽样，保证可复现
    for alpha in [None, 10.0, 3.0, 1.0, 0.5, 0.3]:
        if alpha is None:
            ms = [M_total // G] * G
            ms[-1] += M_total - sum(ms)
            label = 'Inf'
        else:
            frac = torch.distributions.Dirichlet(torch.full((G,), float(alpha))).sample().tolist()
            ms = [max(1, int(round(f * M_total))) for f in frac]
            ms[0] = max(1, ms[0] + (M_total - sum(ms)))
            label = f'{alpha}'
        dat = build(G, ms, n, k)
        try:
            tc, tm, tn = measure(dat, int(M_total / G * 1.3))
        except Exception:
            traceback.print_exc()
            continue
        print_row(label, dat, tc, tm, tn, n, k)
        rows.append((label, list(ms), dat, tc, tm, tn))
        del dat
        torch.cuda.empty_cache()
    return rows


def part_C(rows):
    print()
    print('=' * 104)
    print('C. 正确性：同一份数据下 masked 有效行 == contiguous 有效行（逐位），并 vs bf16 参考')
    print('=' * 104)
    label, ms, dat, *_ = rows[-1]
    G = len(ms)
    run_contiguous(dat)
    run_masked(dat, int(dat['M_total'] / G * 1.3))
    torch.cuda.synchronize()
    dm_valid = torch.cat([dat['d_m'][i, :ms[i]] for i in range(G)], dim=0)
    dc_valid = dat['d_c'][dat['valid_c']]
    print(f'  case alpha={label}: G={G} ms={ms} total valid={sum(ms)} '
          f'physical_M(contiguous)={dat["M"]} storage_M(masked)={G * dat["max_m"]}')
    print(f'     masked valid vs contiguous valid : calc_diff={calc_diff(dm_valid, dc_valid):.3e}  '
          f'bitwise={bool(torch.equal(dm_valid, dc_valid))}')
    ref = torch.cat([(dat['a_m_bf'][i, :ms[i]].float() @ dat['b_bf'][i].float().t())
                     for i in range(G)], dim=0).to(torch.bfloat16)
    print(f'     contiguous vs bf16 ref (FP8 量化噪声): relerr={relerr(dc_valid, ref):.3e}')


def part_D():
    print()
    print('=' * 104)
    print('D. G=32 与两种极端分布')
    print('=' * 104)
    G, n, k, M_total = 32, 2048, 7168, 8192
    w = torch.tensor([1.0 / (i + 1) for i in range(G)])
    w = w / w.sum()
    cases = [
        ('balanced', [M_total // G] * G),
        ('one-heavy(60%)', [int(M_total * 0.6)] + [(M_total - int(M_total * 0.6)) // (G - 1)] * (G - 1)),
        ('power-law 1/r', [max(1, int(round(float(x) * M_total))) for x in w]),
    ]
    for _, ms in cases:
        ms[-1] = max(1, ms[-1] + (M_total - sum(ms)))
    print(HEADER)
    for name, ms in cases:
        dat = build(G, ms, n, k)
        try:
            tc, tm, tn = measure(dat, int(M_total / G * 1.3))
        except Exception:
            traceback.print_exc()
            continue
        print_row(name, dat, tc, tm, tn, n, k)
        del dat
        torch.cuda.empty_cache()


def part_E():
    print()
    print('=' * 104)
    print('E. host 编排开销：composite（event 计时，含 host）vs 纯 device kernel（G=8）')
    print('    masked 的 host 路径每次都要 get_best_config + transform_sf_pair（SFA 形状随 max_m 变）')
    print('=' * 104)
    G, n, k, M_total = 8, 2048, 7168, 8192
    print(f'{"case":>10} {"max_m":>6} {"cont_dev":>9} {"cont_evt":>9} {"mask_dev":>9} {"mask_evt":>9} '
          f'{"mask_host":>9} {"cont_host":>9}')
    for name, ms in [('balanced', [1024] * 8), ('skew', [3840, 600, 600, 700, 500, 600, 700, 652])]:
        dat = build(G, ms, n, k)
        exp = int(M_total / G * 1.3)
        tc = kernel_time(lambda: run_contiguous(dat))
        tm = kernel_time(lambda: run_masked(dat, exp))
        tc_e = bench(lambda: run_contiguous(dat), num_warmups=10, num_tests=30)
        tm_e = bench(lambda: run_masked(dat, exp), num_warmups=10, num_tests=30)
        print(f'{name:>10} {dat["max_m"]:>6} {tc * 1e6:>8.0f}u {tc_e * 1e6:>8.0f}u '
              f'{tm * 1e6:>8.0f}u {tm_e * 1e6:>8.0f}u '
              f'{(tm_e - tm) * 1e6:>8.0f}u {(tc_e - tc) * 1e6:>8.0f}u')
        del dat
        torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name()}, mk_alignment={deep_gemm.get_mk_alignment_for_contiguous_layout()}')
    reset_align()
    rows = part_A(8, 2048, 7168, 8192)
    part_C(rows)
    part_D()
    part_E()


if __name__ == '__main__':
    main()
