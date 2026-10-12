"""P1-1 实验：MoE grouped GEMM 的三种布局（contiguous / masked / PSUM）在 H100 上实测。

目的：为 docs/01 §1 的「grouped contiguous vs masked（+ PSUM）」提供真机证据——
不是跑性能，而是把**布局的物理含义**（对齐、gap 行、masked_m）打印成可核对的数字，
并用 DeepGEMM 自带 generators.py 造数据、与 PyTorch 参考对拍残差。

约束：本机 H100（sm90）。SM100 专属路径（1D1D + UE8M0）不在此列，本脚本走 SM90 的 1D2D。
"""
import os
import sys
import traceback

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major

DEEPGEMM_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from generators import (  # noqa: E402
    MajorTypeAB, generate_m_grouped_contiguous, generate_m_grouped_masked,
)


def relerr(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).abs().mean() / b.abs().mean()).item()


def contiguous_layout():
    num_groups, exp_m, n, k = 6, 96, 256, 512
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        num_groups, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=False)
    d.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a, b, d, layout, disable_ue8m0_cast=True)
    torch.cuda.synchronize()

    # grouped_layout 只含组号（gap 为 -1），据此数每组有效行
    valid_counts = []
    for g in range(num_groups):
        valid_counts.append(int((layout == g).sum().item()))
    row_is_valid = layout >= 0
    gaps = int((~row_is_valid).sum().item())
    print(f'[contiguous] groups={num_groups} total_m={m} n={n} k={k} '
          f'layout.numel()={layout.numel()}')
    print(f'[contiguous] per-group valid M     = {valid_counts}')
    print(f'[contiguous] gap rows (layout == -1) = {gaps} / {m}')
    print(f'[contiguous] relerr(valid rows)      = {relerr(d[valid], ref[valid]):.3e}')
    return valid_counts, gaps


def masked_layout():
    num_groups, max_m, exp_m, n, k = 6, 160, 96, 256, 512
    a, b, masked_m, d, ref, valid = generate_m_grouped_masked(
        num_groups, max_m, exp_m, n, k, use_ue8m0=False, use_psum_layout=False)
    expected_m = int(exp_m * 1.2)
    d.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        a, b, d, masked_m, expected_m, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    print(f'[masked] groups={num_groups} max_m={max_m} expected_m={expected_m} n={n} k={k}')
    print(f'[masked] A shape={tuple(a[0].shape)} (dense [G, max_m, k], 无效行只是被 kernel 跳过)')
    print(f'[masked] per-group masked_m = {masked_m.tolist()}')
    print(f'[masked] relerr(valid rows) = {relerr(d[valid], ref[valid]):.3e}')
    return masked_m.tolist()


def psum_layout():
    num_groups, exp_m, n, k = 6, 96, 256, 512
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        num_groups, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=True)
    d.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a, b, d, layout, disable_ue8m0_cast=True, use_psum_layout=True)
    torch.cuda.synchronize()
    # PSUM: layout[i] 是第 i 组有效 token 的**前缀和结束行**；组起点对齐到 alignment
    align = deep_gemm.get_mk_alignment_for_contiguous_layout()
    ends = layout.tolist()
    bounds = []
    prev = 0
    for i, e in enumerate(ends):
        start = prev
        bounds.append((i, start, e))
        prev = ((e + align - 1) // align) * align
    print(f'[psum] groups={num_groups} total_m={m} n={n} k={k} alignment={align}')
    print(f'[psum] grouped_layout (prefix-sum ends) = {ends}')
    print(f'[psum] decoded per group (idx, start, valid_end) = {bounds}')
    print(f'[psum] d.numel()/n = {d.numel() // n} rows (含组间 gap 行)')
    print(f'[psum] relerr(valid rows) = {relerr(d[valid], ref[valid]):.3e}')
    return ends


def main():
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}')
    print('-' * 92)
    for name, fn in (('contiguous', contiguous_layout),
                     ('masked', masked_layout),
                     ('psum', psum_layout)):
        try:
            fn()
        except Exception as e:
            print(f'[FAIL] {name}: {type(e).__name__}: {e}')
            print('       ... ' + ''.join(traceback.format_exc().strip().splitlines()[-3:]))
        print('-' * 92)


if __name__ == '__main__':
    main()
