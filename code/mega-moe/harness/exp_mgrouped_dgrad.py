"""P4-3: m-grouped dgrad API（`m_grouped_fp8_fp4_gemm_*`）语义与布局核对。

MoE 反向的 dgrad（对激活的梯度）天然是 **M 轴分组**：不等长的是每个专家收到的 token 数，
落在 M 上，而 N/K 固定，对应 DeepGEMM 的 `m_grouped_*` 族。本文在 H100（sm90）上用
linear2 的 dgrad 形状（A=dY `[M,H]`、B=W2 `[G,I,H]`、D=dA `[M,I]`，即 K=H 固定、N=I 固定）
真跑三种布局，核对 P4-3 精读结论：

  A. H100 可跑的 SM90 m-grouped dgrad API
     A1. contiguous：`grouped_layout[row]` = 组号（gap 行 -1），逐组复现 `dA_e = dY_e @ W2_e^T`；
     A2. masked：稠密 `[G,max_m,K]` + `masked_m`，只算有效行；
     A3. PSUM：`grouped_layout[i]` = 第 i 组有效 token 的**前缀和结束行**，组起点对齐；
     A4. contiguous 的 `grouped_layout` 长度与 m 不符时 host 侧直接 assert；
     A5. DeepSeek 量级 dgrad（G=8, M≈4096, N=2048, K=7168）的吞吐（TFLOPS）。

  B. 三种布局的**同一性**：对同一份 dgrad 数据分别走 masked / contiguous / PSUM，
     有效行的 dA 应一致（同一计算的三种边界表示）。

约束：SM100（1D1D + packed UE8M0 + tcgen05）为**仅分析**；本机 H100 走 SM90 的 1D2D。
输出重定向到 exp_mgrouped_dgrad.out.txt。
"""
import os
import sys
import time
import traceback

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major

def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              os.path.expanduser('~/github/DeepGEMM')):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


DEEPGEMM_DIR = _find_deepgemm_dir()
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from generators import (  # noqa: E402
    MajorTypeAB, generate_m_grouped_contiguous, generate_m_grouped_masked,
    layout_masked_to_psum,
)
from deep_gemm.utils import align  # noqa: E402


def relerr(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def reset_align():
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())


# --------------------------------------------------------------------------- A
def a1_contiguous():
    """linear2 dgrad：A=dY [M,H](K=H) @ B=W2 [G,N=I,K=H].mT -> dA [M,I]。"""
    reset_align()
    G, exp_m, n, k = 6, 128, 256, 512
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        G, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=False)
    d.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    align_v = deep_gemm.get_mk_alignment_for_contiguous_layout()
    gaps = int((layout < 0).sum().item())
    print(f'  A1 contiguous dgrad: G={G} m={m} n(I)={n} k(H)={k} alignment={align_v}')
    print(f'     grouped_layout: len={layout.numel()}, gap rows(layout==-1)={gaps}/{m} '
          f'({gaps / m * 100:.0f}%)')
    print(f'     relerr(dA valid) vs dY_e@W2_e^T = {relerr(d[valid], ref[valid]):.3e}')
    return relerr(d[valid], ref[valid])


def a2_masked():
    reset_align()
    G, max_m, exp_m, n, k = 6, 160, 96, 256, 512
    a, b, masked_m, d, ref, valid = generate_m_grouped_masked(
        G, max_m, exp_m, n, k, use_ue8m0=False, use_psum_layout=False)
    expected_m = int(exp_m * 1.2)
    d.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        a, b, d, masked_m, expected_m, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    used = int(masked_m.sum().item())
    print(f'  A2 masked dgrad: G={G} max_m={max_m} expected_m={expected_m} n={n} k={k}')
    print(f'     A shape={tuple(a[0].shape)} (稠密 [G,max_m,K]；无效行只占空间、kernel 跳过)')
    print(f'     masked_m={masked_m.tolist()} -> 有效 {used} 行 / 存储 {G * max_m} 行 '
          f'({used / (G * max_m) * 100:.0f}%)')
    print(f'     relerr(dA valid) = {relerr(d[valid], ref[valid]):.3e}')
    return relerr(d[valid], ref[valid])


def a3_psum():
    reset_align()
    G, exp_m, n, k = 6, 128, 256, 512
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        G, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=True)
    d.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a, b, d, layout, disable_ue8m0_cast=True, use_psum_layout=True)
    torch.cuda.synchronize()
    align_v = deep_gemm.get_mk_alignment_for_contiguous_layout()
    ends = layout.tolist()
    bounds, prev = [], 0
    for i, e in enumerate(ends):
        bounds.append((i, prev, e))
        prev = align(e, align_v)
    print(f'  A3 PSUM dgrad: G={G} m={m} n={n} k={k} alignment={align_v}')
    print(f'     grouped_layout(prefix-sum ends)={ends}')
    print(f'     decoded (idx, physical_start, valid_end)={bounds}')
    print(f'     relerr(dA valid) = {relerr(d[valid], ref[valid]):.3e}')
    return relerr(d[valid], ref[valid])


def a4_layout_length_mismatch():
    reset_align()
    G, exp_m, n, k = 2, 128, 256, 512
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        G, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=False)
    bad = layout[:m - 1].contiguous()   # contiguous 下 grouped_layout 必须与 m 等长
    try:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, bad, disable_ue8m0_cast=True)
        torch.cuda.synchronize()
        print('  A4 layout length mismatch: UNEXPECTEDLY PASSED')
    except Exception as e:
        msg = str(e).strip().splitlines()[-1][:140]
        print(f'  A4 contiguous grouped_layout len={bad.numel()} != m={m}: rejected as expected')
        print(f'     -> {type(e).__name__}: {msg}')


def a5_perf():
    reset_align()
    G, exp_m, n, k = 8, 512, 2048, 7168      # DeepSeek 量级：I=2048(=>N), H=7168(=>K)
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        G, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=False)

    def run():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout, disable_ue8m0_cast=True)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = 2 * m * n * k / dt / 1e12
    print(f'  A5 perf dgrad: G={G} m={m} n={n} k={k} t={dt * 1e6:.0f} us  {tflops:.0f} TFLOPS')
    return tflops


# --------------------------------------------------------------------------- B
def b_three_layouts_same_data():
    """同一份 dgrad 数据 → masked/contiguous/PSUM 三种表示，有效行 dA 应一致。"""
    reset_align()
    align_v = deep_gemm.get_mk_alignment_for_contiguous_layout()
    G, max_m, exp_m, n, k = 6, 160, 96, 256, 512

    a, b, masked_m, d_masked, ref, valid_masked = generate_m_grouped_masked(
        G, max_m, exp_m, n, k, use_ue8m0=False, use_psum_layout=False)
    expected_m = int(exp_m * 1.2)
    d_masked.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        a, b, d_masked, masked_m, expected_m, disable_ue8m0_cast=True)

    # 从 masked_m 构造 PSUM 前缀和结束；物理数据由 layout_masked_to_psum 重排。
    psum_m = torch.empty(G, device='cuda', dtype=torch.int)
    last = 0
    for j in range(G):
        psum_m[j] = last + masked_m[j]
        last = align(int(psum_m[j].item()), align_v)
    total_m = align(int(psum_m[-1].item()), align_v)

    a_psum = (layout_masked_to_psum(a[0], psum_m), layout_masked_to_psum(a[1], psum_m))
    d_psum = torch.empty((total_m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a_psum, b, d_psum, psum_m, disable_ue8m0_cast=True, use_psum_layout=True)

    # contiguous 用同一份物理数据（PSUM 重排后与 contiguous 物理布局相同），只换 grouped_layout 编码。
    layout_row = torch.full((total_m,), -1, device='cuda', dtype=torch.int32)
    last = 0
    for j in range(G):
        end = int(psum_m[j].item())
        layout_row[last:end] = j
        last = align(end, align_v)
    d_contig = torch.empty((total_m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a_psum, b, d_contig, layout_row, disable_ue8m0_cast=True)
    torch.cuda.synchronize()

    # 有效行的行索引（三种表示共享同一物理行顺序）
    valid_rows = torch.zeros(total_m, device='cuda', dtype=torch.bool)
    last = 0
    for j in range(G):
        end = int(psum_m[j].item())
        valid_rows[last:end] = True
        last = align(end, align_v)

    # masked 的有效行拉成一维，顺序与 PSUM 相同（组内连续）
    ref_valid = ref[valid_masked].reshape(-1, n)
    dm = d_masked[valid_masked].reshape(-1, n)
    dp = d_psum[valid_rows]
    dc = d_contig[valid_rows]
    print(f'  B same dgrad data through 3 layouts: G={G} total valid={ref_valid.shape[0]} '
          f'total_m(physical)={total_m} n={n} k={k}')
    print(f'     masked   vs dY@W2^T ref : relerr={relerr(dm, ref_valid):.3e}')
    print(f'     PSUM     vs dY@W2^T ref : relerr={relerr(dp, ref_valid):.3e}')
    print(f'     contiguous vs ref       : relerr={relerr(dc, ref_valid):.3e}')
    print(f'     masked vs PSUM          : relerr={relerr(dm, dp):.3e}')
    print(f'     PSUM   vs contiguous    : relerr={relerr(dp, dc):.3e}')


def main():
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}')
    print('=' * 92)
    print('A. H100 (sm90) m-grouped dgrad API 语义（linear2: A=dY, B=W2, D=dA）')
    for name, fn in (('A1 contiguous', a1_contiguous), ('A2 masked', a2_masked),
                     ('A3 PSUM', a3_psum), ('A4 layout-length', a4_layout_length_mismatch),
                     ('A5 perf', a5_perf)):
        try:
            fn()
        except Exception:
            print(f'  {name} FAILED:\n' + traceback.format_exc())
    print('=' * 92)
    print('B. 三种布局同一性（同一份 dgrad 数据）')
    try:
        b_three_layouts_same_data()
    except Exception:
        print('  B FAILED:\n' + traceback.format_exc())


if __name__ == '__main__':
    main()
