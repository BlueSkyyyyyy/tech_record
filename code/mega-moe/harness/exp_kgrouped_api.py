"""P4-2: k-grouped wgrad API (`k_grouped_fp8_gemm_*_contiguous`) 语义核对。

在 H100（sm90）上真跑 DeepGEMM 2.8.1 的 K 轴分组 wgrad kernel，验证 P4-2 精读结论：

  A. H100 可跑的 SM90 `k_grouped_fp8_gemm_nt_contiguous`
     A1. contiguous `grouped_layout`（`layout[i]`=第 i 组 pad 后的 K）逐组复现 `D[e]=A_e^T B_e`；
     A2. 空组（ks_cpu 含 0）不崩且结果不变；
     A3. `ks_cpu` 未按 `k_alignment`（=128）对齐时 host 侧直接 assert；
     A4. `k_grouped_fp8_gemm_tn_contiguous`（SM100 专属）在 sm90 调 `DG_HOST_UNREACHABLE`；
     A5. 测一组 shape 的吞吐（TFLOPS）。

  B. 两种物理布局的等价（纯 torch，float32，CPU，不依赖 SM90/SM100）
     B1. SM90 `nt`：每组按 `[M, k_g]` K-major（K 连续）拼接；
     B2. SM100 `tn`：整体按 `[sum_k, M]` MN-major（M 连续）拼接；
     两者 padding 后逐组 `A_e^T @ B_e` 都与自然布局参考一致。

输出重定向到 exp_kgrouped_api.out.txt。
"""
import os
import sys
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
    MajorTypeAB, QuantConfig, generate_k_grouped_contiguous,
)

K_ALIGN = 128


def rel(a, b):
    return ((a.float() - b.float()).abs().mean() / (b.float().abs().mean() + 1e-30)).item()


def reset_align():
    deep_gemm.set_mk_alignment_for_contiguous_layout(K_ALIGN)


# --------------------------------------------------------------------------- A
def a1_nt_contiguous():
    reset_align()
    num_groups, m, n = 4, 512, 1024
    ks = [512, 384, 640, 256]
    _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=K_ALIGN, accumulate=True, out_dtype=torch.float)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, host_ks, layout, d, recipe=(1, 1, 128))
    torch.cuda.synchronize()
    print(f'  A1 nt contiguous: groups={num_groups} m={m} n={n} logical_ks={ks}')
    print(f'     host_ks(padded)={host_ks} sum_k={sum(host_ks)} grouped_layout={layout.tolist()}')
    print(f'     relerr(D[e]) vs A_e^T B_e (fp32 accumulate) = {rel(d, ref):.3e}')
    return rel(d, ref)


def a2_empty_group():
    # SM90 nt 强制 accumulate（gemm.hpp:403 断言 c.has_value()），故用 D+= 路径：
    # 把初始 D 清零，空组应保持为 0。
    reset_align()
    num_groups, m, n = 4, 512, 1024
    ks = [512, 0, 640, 256]  # group 1 empty
    _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=K_ALIGN, accumulate=False, out_dtype=torch.float)
    d.zero_()
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, host_ks, layout, d, recipe=(1, 1, 128))
    torch.cuda.synchronize()
    print(f'  A2 empty group: logical_ks={ks} host_ks={host_ks}')
    print(f'     D[1] all-zero (empty group contributes nothing) = {bool((d[1] == 0).all().item())}')
    nonempty = [0, 2, 3]
    print(f'     relerr on non-empty groups = {rel(d[nonempty], ref[nonempty]):.3e}')
    return rel(d[nonempty], ref[nonempty])


def a3_unaligned_ks():
    # host 侧 `check_k_grouped_args`（gemm.hpp:55-76）要求每个 k 是 k_alignment 的倍数；
    # 直接用未对齐的 ks_cpu 调 API（数据本身按对齐尺寸造好），应被拒绝。
    reset_align()
    num_groups, m, n = 2, 256, 512
    ks = [384, 200]  # 200 不是 128 的倍数
    aligned = [384, 256]
    _, a, b, c, d, ref, layout, _ = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, aligned,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=K_ALIGN, accumulate=True, out_dtype=torch.float)
    try:
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, layout, d, recipe=(1, 1, 128))
        torch.cuda.synchronize()
        print('  A3 unaligned ks: UNEXPECTEDLY PASSED')
        return 'passed?'
    except Exception as e:
        msg = str(e).strip().splitlines()[-1][:140]
        print(f'  A3 unaligned ks_cpu={ks} (k_alignment={K_ALIGN}): rejected as expected')
        print(f'     -> {type(e).__name__}: {msg}')
        return msg


def a4_tn_rejected():
    reset_align()
    num_groups, m, n = 2, 256, 512
    ks = [384, 256]
    _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=K_ALIGN, accumulate=True, out_dtype=torch.float)
    try:
        deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, host_ks, layout, d, recipe=(1, 1, 128))
        torch.cuda.synchronize()
        print('  A4 tn on sm90: UNEXPECTEDLY PASSED')
    except Exception as e:
        msg = str(e).strip().splitlines()[-1][:120]
        print(f'  A4 tn on sm90: rejected as expected -> {type(e).__name__}: {msg}')


def a5_perf():
    reset_align()
    num_groups, m, n = 8, 4096, 7168
    ks = [4096] * num_groups
    _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=K_ALIGN, accumulate=True, out_dtype=torch.float)

    def run():
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, host_ks, layout, d, recipe=(1, 1, 128))

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    import time
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    logical_k = sum(ks)
    tflops = 2 * m * n * logical_k / dt / 1e12
    print(f'  A5 perf: groups={num_groups} m={m} n={n} k/group={ks[0]} '
          f't={dt * 1e6:.0f} us  {tflops:.0f} TFLOPS')
    return tflops


# --------------------------------------------------------------------------- B
def b_layout_equivalence():
    """SM90 nt (每组 [M,k] K-major) vs SM100 tn (整体 [sum_k,M] MN-major) 等价。"""
    torch.manual_seed(0)
    ks = [3, 2, 4]            # logical per-group token counts
    M, N = 5, 6
    padded = [4, 4, 4]        # align to 4 for illustration (真机是 128)
    sum_k = sum(padded)

    # natural per-group activations / upstream grads:  A_e: [k, M], B_e: [k, N]
    A = [torch.randn(k, M) for k in ks]
    B = [torch.randn(k, N) for k in ks]
    ref = torch.stack([A[e].T @ B[e] for e in range(len(ks))])  # [E, M, N]

    # SM100 tn: concat groups along K (axis 0), pad to padded, layout [sum_k, M] / [sum_k, N]
    Atn = torch.zeros(sum_k, M)
    Btn = torch.zeros(sum_k, N)
    off = 0
    for e, k in enumerate(ks):
        Atn[off:off + k] = A[e]
        Btn[off:off + k] = B[e]
        off += padded[e]
    d_tn = torch.stack([Atn[o:o + padded[e]].T @ Btn[o:o + padded[e]]
                        for e, o in enumerate(_starts(padded))])

    # SM90 nt: per-group store [M, padded_k] K-major (inner stride = padded K,
    # 与 generator `x[start:end].T.flatten()` 一致，end-start 是对齐后的尺寸),
    # 各组首尾相接；base 偏移 = 累计对齐 K * M。
    flat_a = torch.zeros(sum_k * M)
    flat_b = torch.zeros(sum_k * N)
    off = 0
    for e, k in enumerate(ks):
        pad = padded[e]
        Apad = torch.zeros(pad, M)
        Bpad = torch.zeros(pad, N)
        Apad[:k] = A[e]
        Bpad[:k] = B[e]
        flat_a[off * M:(off + pad) * M] = Apad.T.reshape(-1)   # [M, pad] row-major => K(pad) contiguous
        flat_b[off * N:(off + pad) * N] = Bpad.T.reshape(-1)
        off += pad
    d_nt = torch.empty(len(ks), M, N)
    off = 0
    for e in range(len(ks)):
        pad = padded[e]
        Ae = flat_a[off * M:(off + pad) * M].reshape(M, pad).T   # [pad, M]
        Be = flat_b[off * N:(off + pad) * N].reshape(N, pad).T
        d_nt[e] = Ae.T @ Be
        off += pad

    print(f'  B layout: logical_ks={ks} padded={padded} M={M} N={N} sum_k={sum_k}')
    print(f'     SM100 tn [sum_k,M] MN-major relerr vs ref = {rel(d_tn, ref):.3e}')
    print(f'     SM90  nt per-group [M,k] K-major relerr vs ref = {rel(d_nt, ref):.3e}')


def _starts(padded):
    off, out = 0, []
    for p in padded:
        out.append(off)
        off += p
    return out


def main():
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}, cap={torch.cuda.get_device_capability(0)}')
    print('=' * 90)
    print('A. H100 (sm90) k-grouped wgrad API 语义')
    try:
        a1_nt_contiguous()
    except Exception:
        print('  A1 FAILED:\n' + traceback.format_exc())
    try:
        a2_empty_group()
    except Exception:
        print('  A2 FAILED:\n' + traceback.format_exc())
    try:
        a3_unaligned_ks()
    except Exception:
        print('  A3 FAILED:\n' + traceback.format_exc())
    try:
        a4_tn_rejected()
    except Exception:
        print('  A4 FAILED:\n' + traceback.format_exc())
    try:
        a5_perf()
    except Exception:
        print('  A5 FAILED:\n' + traceback.format_exc())

    print('=' * 90)
    print('B. 物理布局等价（纯 torch float32，与架构无关）')
    try:
        b_layout_equivalence()
    except Exception:
        print('  B FAILED:\n' + traceback.format_exc())


if __name__ == '__main__':
    main()
