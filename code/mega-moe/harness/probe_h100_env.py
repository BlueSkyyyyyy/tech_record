"""P0-4 环境探测：在 H100 (sm90) 上确认 DeepGEMM 2.8.1 哪些 kernel 可编译/运行。

策略：复用 DeepGEMM 自带 tests/generators.py 造数据，逐个 kernel 试跑，
把结果（OK / FAIL + 异常尾部）打印成一张矩阵。不做性能基准（那是后续 P4 的事）。
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
    KernelType, MajorTypeAB, QuantConfig,
    generate_normal, generate_m_grouped_contiguous,
    generate_k_grouped_contiguous,
)

RESULTS = []


def run(name, fn, note=''):
    try:
        out = fn()
        RESULTS.append((name, 'OK', out or '', note))
        print(f'[ OK ] {name} {out or ""} {note}')
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'
        tail = ''.join(traceback.format_exc().strip().splitlines()[-3:])
        RESULTS.append((name, 'FAIL', msg, note))
        print(f'[FAIL] {name} {msg} {note}')
        print(f'       ... {tail}')


def dense_fp8_nt():
    m, n, k = 512, 1024, 2048
    a, b, c, d, ref = generate_normal(m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                                      False, torch.bfloat16, KernelType.Kernel1D2D, use_ue8m0=False)
    deep_gemm.fp8_gemm_nt(a, b, d, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    import torch as _t
    diff = ((d.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
    return f'm={m} n={n} k={k} relerr={diff:.2e}'


def dense_bf16_nt():
    m, n, k = 512, 1024, 2048
    a, b, c, d, ref = generate_normal(m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                                      False, torch.bfloat16, KernelType.KernelNoSF, use_bf16=True)
    deep_gemm.bf16_gemm_nt(a, b, d)
    torch.cuda.synchronize()
    diff = ((d.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
    return f'm={m} n={n} k={k} relerr={diff:.2e}'


def m_grouped_fp8_nt_contiguous():
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    m, a, b, layout, d, ref, mask = generate_m_grouped_contiguous(
        8, 512, 1024, 2048, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=False)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    diff = ((d[mask].float() - ref[mask].float()).abs().mean() / ref[mask].float().abs().mean()).item()
    return f'groups=8 m={m} n=1024 k=2048 relerr={diff:.2e}'


def k_grouped_nt_contiguous_sm90():
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    num_groups, m, n = 4, 512, 1024
    ks = [512, 384, 640, 256]
    _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=128, accumulate=True, out_dtype=torch.float)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, host_ks, layout, d, recipe=(1, 1, 128))
    torch.cuda.synchronize()
    diff = ((d - ref).abs().mean() / ref.abs().mean()).item()
    return f'groups={num_groups} m={m} n={n} ks={ks} relerr={diff:.2e}'


def k_grouped_tn_contiguous_sm100():
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    num_groups, m, n = 4, 512, 1024
    ks = [512, 384, 640, 256]
    _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=128, accumulate=True, out_dtype=torch.float)
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, host_ks, layout, d, recipe=(1, 1, 128))
    torch.cuda.synchronize()
    return 'ran (unexpected on sm90)'


def mega_moe_probe():
    buf_cls = deep_gemm.mega.SymmBuffer
    return f'SymmBuffer available: {buf_cls is not None}'


def main():
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}, cap={torch.cuda.get_device_capability(0)}')
    print('-' * 90)

    run('dense fp8_fp8 gemm_nt (SM90 1D2D)', dense_fp8_nt, 'Forward, 预期 OK')
    run('dense bf16 gemm_nt', dense_bf16_nt, 'Forward, 预期 OK')
    run('m_grouped fp8 gemm_nt_contiguous', m_grouped_fp8_nt_contiguous, 'MoE linear1/2 前向, 预期 OK')
    run('k_grouped fp8 gemm_nt_contiguous', k_grouped_nt_contiguous_sm90, 'MoE wgrad, SM90 走 nt, 预期 OK')
    run('k_grouped fp8 gemm_tn_contiguous', k_grouped_tn_contiguous_sm100, 'MoE wgrad, SM100 走 tn, 预期 FAIL')
    run('mega_moe SymmBuffer', mega_moe_probe, 'SM100 专属, 仅分析')

    print('-' * 90)
    n_ok = sum(1 for r in RESULTS if r[1] == 'OK')
    print(f'summary: {n_ok}/{len(RESULTS)} OK')


if __name__ == '__main__':
    main()
