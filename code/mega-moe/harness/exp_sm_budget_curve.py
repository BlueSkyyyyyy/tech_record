#!/usr/bin/env python3
"""P180-1 实验：MoE grouped GEMM 的「SM 预算曲线」——给通信让出一个 SM 要付多少计算？

Mega MoE 把 dispatch/combine 的通信 warp 和 UMMA/MMA 的计算 warp 放进**同一个** persistent
kernel、**同一块** SM（`sm100_fp8_fp4_mega_moe.cuh` 的 warp specialization + `setmaxnreg`）。于是
一个绕不开的账目是：**计算腿少用一个 SM，损失多少吞吐？** 前面几篇从两个 kernel / 两条 stream 的
角度量过「重叠」（第 182/183 篇），但从没量过**单条 grouped GEMM 自己的 SM 预算曲线**——而这正是
「通信腿该分几个 SM」这个决策的分子（通信腿的成本见第 165 篇）。

关键源码事实：DeepGEMM 的 grouped GEMM 是 **persistent kernel**，grid = `num_sms`，
`kNumSMs` 是**编译期模板参数**（`scheduler/gemm.cuh:20`）：
  - 每个 block 认领块的步长是 `next_block_idx = (++iter) * kNumSMs + blockIdx.x`
    （`scheduler/gemm.cuh:180`）——`num_sms` 直接是「一波」的宽度；
  - **更隐蔽**：L2 swizzle 的分组宽度 `kNum1DBlocksPerGroup` 由 `get_num_1d_blocks_per_group`
    按 `kNumSMs` 在 `{8,16}` 里挑（`scheduler/gemm.cuh:12-27`）——**改 `num_sms` 不仅改并行度，
    还改块到 (m,n) 的映射/L2 局部性**。所以曲线不是简单的「越少越慢」。
在 host 侧，`deep_gemm.set_num_sms(k)`（`csrc/runtime/runtime.hpp:78`）把 `kNumSMs` 编进新 kernel。

本实验在 H100（sm90）上用真实 SM90 FP8 grouped kernel 扫 `num_sms ∈ {132..4}`，对三类 MoE 形状
（前向 L1 / 前向 L2(=dgrad) / K-grouped wgrad），量纯 device kernel 时长与有效 TFLOPS，给：
  A. 每条形状的 SM 预算曲线 + 「完美线性」参考线，找拐点；
  B. 边际代价：每让出一个 SM，多花多少 μs（分段）；
  C. 反推「reserve k 个 SM 给通信」的净收益上界（= 计算腿变慢 vs 通信藏起来）。

注意：本机是 8 卡共享节点，绑到利用率最低的卡。
"""
import os
import sys
import traceback
import warnings

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major

def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              '~/github/DeepGEMM'):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


DEEPGEMM_DIR = _find_deepgemm_dir()
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from generators import (  # noqa: E402
    MajorTypeAB, cast_fp8_fp4_with_major, grouped_cast_fp8_fp4_with_major,
    generate_k_grouped_contiguous,
)

ALIGN = 128
DEV_SMS = 132
NUM_SMS_GRID = [132, 120, 108, 96, 84, 72, 60, 48, 36, 24, 12, 8, 4]
warnings.filterwarnings('ignore', message='.*Profiler clears events.*')


def align_up(x, a):
    return (x + a - 1) // a * a


def reset_align():
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())


def kernel_time(fn, iters=30, warmup=15):
    """纯 device kernel 时长（torch.profiler 取 1d2d / k-grouped impl 的 device_time）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for ev in prof.key_averages():
        # m_grouped → sm90_fp8_gemm_1d2d_impl；k_grouped(wgrad) → sm90_fp8_gemm_1d1d_impl
        if 'sm90_fp8_gemm_1d' in ev.key:
            tot += ev.device_time_total / iters
    return tot / 1e6


# --------------------------------------------------------------------------- build
def build_m_grouped(G, ms, n, k, seed=0):
    """M 轴分组 contiguous：各专家拼接、每段向上对齐到 ALIGN、gap 行 group=-1。"""
    torch.manual_seed(seed)
    aligned = [align_up(m, ALIGN) for m in ms]
    M = sum(aligned)
    groups = [torch.randn((m, k), device='cuda', dtype=torch.bfloat16) for m in ms]
    b_bf = torch.randn((G, n, k), device='cuda', dtype=torch.bfloat16)

    a_c = torch.zeros((M, k), device='cuda', dtype=torch.bfloat16)
    layout = torch.full((M,), -1, device='cuda', dtype=torch.int32)
    pos = 0
    for i, (m, al) in enumerate(zip(ms, aligned)):
        a_c[pos:pos + m] = groups[i]
        layout[pos:pos + m] = i
        pos += al

    a_fp8 = cast_fp8_fp4_with_major(a_c, MajorTypeAB.KMajor, 128, False, use_ue8m0=False)
    b_fp8 = grouped_cast_fp8_fp4_with_major(b_bf, MajorTypeAB.KMajor, 128, False,
                                            use_ue8m0=False, use_block_cast_for_fp8=True)
    d = torch.empty((M, n), device='cuda', dtype=torch.bfloat16)
    return dict(kind='m', G=G, ms=ms, M=M, M_total=sum(ms), n=n, k=k,
                a=a_fp8, b=b_fp8, d=d, layout=layout)


def build_k_grouped(G, m, n, ks, seed=0):
    """K 轴分组 contiguous（nt，K-major）：A/B 每组 [M, k_g]、D[e]=A_e^T B_e、输出 [G,m,n]（fp32 累积）。"""
    torch.manual_seed(seed)
    total_k, a, b, c, d, _ref, layout, host_ks = generate_k_grouped_contiguous(
        G, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=ALIGN, accumulate=True, out_dtype=torch.float)
    return dict(kind='k', G=G, ks=ks, m=m, n=n, ksum=total_k, host_ks=host_ks,
                a=a, b=b, c=c, d=d, layout=layout)


def run(dat):
    if dat['kind'] == 'm':
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            dat['a'], dat['b'], dat['d'], dat['layout'], disable_ue8m0_cast=True)
    else:
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            dat['a'], dat['b'], dat['d'], dat['host_ks'], dat['layout'], dat['c'],
            recipe=(1, 1, 128))


def flops(dat):
    if dat['kind'] == 'm':
        return 2 * dat['M_total'] * dat['n'] * dat['k']
    # wgrad: 每组 2·m·n·k_g，K 按对齐后的物理 K 计（含 padding ≈ 物理 K）
    return 2 * dat['m'] * dat['n'] * sum(dat['ks'])


def describe(dat):
    if dat['kind'] == 'm':
        return (f"m_grouped L{'1' if dat['k'] == 7168 and dat['n'] == 4096 else '2'} "
                f"G={dat['G']} m/exp={dat['M_total'] // dat['G']} n={dat['n']} k={dat['k']} "
                f"M(phys)={dat['M']}")
    return f"k_grouped(wgrad) G={dat['G']} m={dat['m']} n={dat['n']} k_g={dat['ks'][0]}"


def scan(label, dat):
    fl = flops(dat)
    print('-' * 108)
    print(f'{label}  [{describe(dat)}]  FLOP={fl / 1e9:.1f} G   (理想线性以 num_sms=132 为基准)')
    print('-' * 108)
    print(f'{"num_sms":>8} {"t(us)":>9} {"TFLOPS":>9} {"norm_t":>8} {"t_lin(us)":>10} '
          f'{"overhead":>9} {"TF/SM":>8}')
    t132 = None
    rows = []
    for k in NUM_SMS_GRID:
        if k > DEV_SMS:
            continue
        try:
            deep_gemm.set_num_sms(k)
            run(dat)
            torch.cuda.synchronize()
            # 二次调用前 warmup 已在 kernel_time 内做（含 JIT 编译）
            t = kernel_time(lambda: run(dat))
        except Exception:
            traceback.print_exc()
            deep_gemm.set_num_sms(DEV_SMS)
            continue
        if t132 is None:
            t132 = t
        t_lin = t132 * DEV_SMS / k
        tf = fl / t / 1e12
        print(f'{k:>8} {t * 1e6:>9.1f} {tf:>9.0f} {t / t132:>8.3f} {t_lin * 1e6:>10.1f} '
              f'{t / t_lin:>9.3f} {tf / k * 1e3:>8.1f}')
        rows.append((k, t, tf))
    deep_gemm.set_num_sms(DEV_SMS)
    # 分段边际代价（每让出一个 SM 多花的 μs），在 t 随 1/k 近线性的区间
    print(f'  边际代价（每少一个 SM 的额外 μs，取相邻两点差）:')
    for (k0, t0, _), (k1, t1, _) in zip(rows, rows[1:]):
        dk = k0 - k1
        dt = (t1 - t0) * 1e6
        print(f'    {k0:>3}->{k1:<3} Δsm={dk:>3}: +{dt:>7.1f} us  ({dt / dk:>6.2f} us/SM)')
    return rows, t132


def main():
    torch.manual_seed(0)
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name()}, device_SMs={DEV_SMS}, '
          f'mk_alignment={deep_gemm.get_mk_alignment_for_contiguous_layout()}')
    reset_align()

    shapes = []
    # 前向 L1：A=[tokens,H]=[8192,7168]、W1=[2I,H]=[4096,7168]
    shapes.append(('L1', build_m_grouped(8, [1024] * 8, 4096, 7168)))
    # 前向 L2 / dgrad：A=[tokens,I]=[8192,2048]、W2=[H,I]=[7168,2048]
    shapes.append(('L2/dgrad', build_m_grouped(8, [1024] * 8, 7168, 2048)))
    # 反向 wgrad：D[e]=A_e^T B_e，A=[M,k_g]=[7168,1024]、B=[N,k_g]=[2048,1024]
    shapes.append(('wgrad', build_k_grouped(8, 7168, 2048, [1024] * 8)))

    all_rows = {}
    for label, dat in shapes:
        rows, t132 = scan(label, dat)
        all_rows[label] = (dat, rows, t132)
        del dat
        torch.cuda.empty_cache()

    print()
    print('=' * 108)
    print('C. 反推：reserve k 个 SM 给通信腿的净收益上界')
    print('   （通信腿藏起来省下的时间 = 通信墙钟；计算腿损失 = t(132-k)-t(132)。净收益 = 藏住的 - 损失的）')
    print('=' * 108)
    for label, (dat, rows, t132) in all_rows.items():
        tmap = {k: t for k, t, _ in rows}
        print(f'  [{label}]')
        base = tmap.get(DEV_SMS, t132)
        for k in (8, 16, 24, 32, 48, 64):
            if DEV_SMS - k in tmap:
                loss = (tmap[DEV_SMS - k] - base) * 1e6
                print(f'    reserve {k:>2} SM -> 计算腿 {base * 1e6:.0f}->{tmap[DEV_SMS - k] * 1e6:.0f} us '
                      f'(损失 +{loss:.1f} us)；只有当通信墙钟 > {loss:.1f} us 才值')


if __name__ == '__main__':
    main()
