"""P177-1: FP8 K-grouped wgrad 的 SF 链路端到端 —— `布局 -> SF -> kernel`（H100 兑现）。

第 12 篇（P4-2）拆了 SM90 FP8 `k_grouped_nt` 的 kernel/API，第 16 篇（P16-1）拆了 K-grouped SF 的
**打包**（`transform_k_grouped_sf_into_required_layout` -> `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`，
SM100-only），第 186 篇（P176-1）用 BF16 MN-major 同构体证明了目标卡 FP8 `k_grouped_tn` 的**累积结构**，
并把「SF 也要走同一套变换」（`gemm.hpp:363-364`）留给第 16 篇。本篇把这条链真正**串起来**：

  A  端到端 FP8 `nt`：真实的 SM90 FP8 k-grouped kernel，走完整 `量化 -> SF -> transform -> kernel`；
     显式调用 `get_mn_major_tma_aligned_tensor` 预变换 SF，与隐式（API 内部变换）**逐位相等**；
     并核对 `kernel vs 反量化操作数上的精确 GEMM`（隔离 kernel 误差）。
  B  SF 几何：每组 SF 行数 / 偏移，对照 scheduler 的 `current_sf_k_cumsum` 公式（`scheduler/gemm.cuh:242`）
     与 kernel 的 `shape_sfa_k = ceil_div(shape_k, gran_k*4)`（`sm100_fp8_fp4_gemm_1d1d.cuh:155`）。
  C  目标卡（SM100）tn 的 SF 需求：`get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor` 在 H100 被
     arch 门控拒（`smxx_layout.hpp:224`）；host 复刻 packed_sf_k 公式，坐实 **一行 SF 恒覆盖 `gran_k*4`
     个 K 元素**（= `kSFKSpan`，kernel `:199`）——这就是「布局不同、SF 几何同构」的落点。
  D  BF16 MN-major tn（第 186 篇可跑对照）与 FP8 nt 同题对拍：两条物理布局、同一数学，误差只在精度层。

输出重定向到 exp_kgrouped_sf_chain.out.txt。
"""
import time
import traceback

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major

K_ALIGN = 128
GRAN_K = 128


def align(x, a=K_ALIGN):
    return ((x + a - 1) // a) * a


def ceil_div(x, y):
    return (x + y - 1) // y


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def bitfrac(x, y):
    xb = x.reshape(-1).view(torch.int32)
    yb = y.reshape(-1).view(torch.int32)
    return (xb != yb).float().mean().item()


def build_problem(logical_ks, m, n, seed=0):
    """造 K-grouped wgrad 的原始 bf16 操作数（组内 token 数不等、含 padding 行）。

    A_e: [k_e, m]（激活）、B_e: [k_e, n]（上游梯度）；收缩轴 = 每组 token 数 k_e。
    """
    torch.manual_seed(seed)
    host_ks = [align(k) for k in logical_ks]
    A, B = [], []
    for k in logical_ks:
        A.append(torch.randn(k, m, dtype=torch.bfloat16, device='cuda'))
        B.append(torch.randn(k, n, dtype=torch.bfloat16, device='cuda'))
    return A, B, host_ks


def ref_fp64(A, B):
    G = len(A)
    m, n = A[0].shape[1], B[0].shape[1]
    out = torch.empty(G, m, n, dtype=torch.float64, device='cuda')
    for i in range(G):
        out[i] = A[i].double().T @ B[i].double()
    return out


# --------------------------------------------------------------------------- FP8 nt
def build_fp8_nt(A, B, host_ks):
    """K-major (`[M, sum_k]` K 连续) 的 FP8 操作数 + 参考量化器产出的 per-token fp32 SF。

    完全照 `tests/generators.py:488-540` 的 K-major 分支：
      每组 x_group=[host_k, mn] 先补零，`per_channel_cast_to_fp8` 沿 K 每 128 行出一个 SF；
      data = x_fp8.T.flatten()（K 连续），sf 输入 = cat(groups_sf).T（形状 [mn, sum_sf_k]）。
    返回 (a_data, a_sf, b_data, b_sf, deq_summary)：deq 返回每组反量化后的 [k, mn] fp32。
    """
    m, n = A[0].shape[1], B[0].shape[1]
    sum_k = sum(host_ks)
    a_data = torch.zeros(sum_k * m, dtype=torch.float8_e4m3fn, device='cuda')
    b_data = torch.zeros(sum_k * n, dtype=torch.float8_e4m3fn, device='cuda')
    a_sfs, b_sfs = [], []
    a_deq, b_deq = [], []
    for g, (k, hk) in enumerate(zip([a.shape[0] for a in A], host_ks)):
        for X, mn, data, sfs, deqs in [(A[g], m, a_data, a_sfs, a_deq),
                                       (B[g], n, b_data, b_sfs, b_deq)]:
            x = torch.zeros(hk, mn, dtype=torch.bfloat16, device='cuda')
            x[:k] = X
            xf, sf = deep_gemm.per_channel_cast_to_fp8(x, use_ue8m0=False, gran_k=GRAN_K)
            off = sum(host_ks[:g])
            data[off * mn:(off + hk) * mn] = xf.T.flatten()
            sfs.append(sf)                                    # [hk/128, mn]
            deq = (xf.float() * sf.repeat_interleave(GRAN_K, dim=0))[:k]   # [k, mn]
            deqs.append(deq)
    a_sf = torch.cat(a_sfs, dim=0).T.contiguous()             # [m, sum_sf_k]
    b_sf = torch.cat(b_sfs, dim=0).T.contiguous()             # [n, sum_sf_k]
    return a_data, a_sf, b_data, b_sf, a_deq, b_deq


def run_fp8_nt(a_data, a_sf, b_data, b_sf, host_ks, G, m, n):
    d = torch.zeros(G, m, n, dtype=torch.float32, device='cuda')
    layout = torch.tensor(host_ks, dtype=torch.int32, device='cuda')
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (a_data, a_sf), (b_data, b_sf), d, host_ks, layout, d, recipe=(1, 1, GRAN_K))
    torch.cuda.synchronize()
    return d


# --------------------------------------------------------------------------- BF16 tn
def build_bf16_tn(A, B, host_ks):
    """MN-major (`[sum_k, MN]`，MN 连续) 的 BF16 操作数（第 186 篇可跑对照）。"""
    m, n = A[0].shape[1], B[0].shape[1]
    sum_k = sum(host_ks)
    a = torch.zeros(sum_k, m, dtype=torch.bfloat16, device='cuda')
    b = torch.zeros(sum_k, n, dtype=torch.bfloat16, device='cuda')
    for g, hk in enumerate(host_ks):
        off = sum(host_ks[:g])
        k = A[g].shape[0]
        a[off:off + k] = A[g]
        b[off:off + k] = B[g]
    return a, b


def run_bf16_tn(a, b, host_ks, G, m, n):
    d = torch.zeros(G, m, n, dtype=torch.float32, device='cuda')
    layout = torch.tensor(host_ks, dtype=torch.int32, device='cuda')
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, host_ks, layout, d)
    torch.cuda.synchronize()
    return d


def main():
    deep_gemm.set_mk_alignment_for_contiguous_layout(K_ALIGN)
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}, cap={torch.cuda.get_device_capability(0)}')
    print('=' * 94)

    # 故意用**未对齐**的逻辑 K -> 每组产生 padding 行，喂出「布局 -> SF」的边界
    logical_ks = [500, 300, 700, 250]
    G = len(logical_ks)
    m, n = 512, 1024
    A, B, host_ks = build_problem(logical_ks, m, n, seed=0)
    ref = ref_fp64(A, B)
    sf_k = sum(host_ks) // GRAN_K
    print(f'shape: G={G} m={m} n={n}  logical_ks={logical_ks}  host_ks(padded)={host_ks}  '
          f'sum_k={sum(host_ks)}  per-group SF rows(host_k/128)={[h // GRAN_K for h in host_ks]}')

    # ===================================================================== A
    print('-' * 94)
    print('A  端到端 FP8 nt：量化 -> SF -> transform -> kernel')
    a_data, a_sf, b_data, b_sf, a_deq, b_deq = build_fp8_nt(A, B, host_ks)
    print(f'   物理布局：A/B = [sum_k, M] 的 K-major（K 连续），展开 numel={a_data.numel()}')
    print(f'   输入 SF：a_sf={tuple(a_sf.shape)} contiguous={a_sf.is_contiguous()}  '
          f'（[mn, sum_sf_k] 约定，SM90 nt 侧）')

    d_impl = run_fp8_nt(a_data, a_sf, b_data, b_sf, host_ks, G, m, n)

    a_sf_t = deep_gemm.get_mn_major_tma_aligned_tensor(a_sf)
    b_sf_t = deep_gemm.get_mn_major_tma_aligned_tensor(b_sf)
    print(f'   显式变换后 SF：a_sf_t={tuple(a_sf_t.shape)} stride={tuple(a_sf_t.stride())} '
          f'（MN 连续 + MN TMA 对齐 stride={align(m, 4)}）')
    # 幂等：再变换一次应在 preprocess_sf 里早退
    a_sf_t2 = deep_gemm.get_mn_major_tma_aligned_tensor(a_sf_t)
    print(f'   再次变换早退：stride 相同={tuple(a_sf_t2.stride()) == tuple(a_sf_t.stride())}  '
          f'值逐位相同={bool(torch.equal(a_sf_t2.view(torch.int32), a_sf_t.view(torch.int32)))}')

    d_expl = run_fp8_nt(a_data, a_sf_t, b_data, b_sf_t, host_ks, G, m, n)
    print(f'   隐式(API 内部变换) vs 显式(预变换) SF：逐位相等={bool(torch.equal(d_impl, d_expl))}  '
          f'bit-diff={bitfrac(d_impl, d_expl):.4f}')

    ref_deq = torch.empty_like(ref)
    for g in range(G):
        ref_deq[g] = a_deq[g].double().T @ b_deq[g].double()
    print('   逐组误差：')
    for g in range(G):
        print(f'     group {g}: logical_k={logical_ks[g]:4d}  kernel-vs-反量化参考={relerr(d_impl[g], ref_deq[g]):.3e}'
              f'   kernel-vs-bf16原值={relerr(d_impl[g], ref[g]):.3e}   (量化本身 vs bf16原值={relerr(ref_deq[g], ref[g]):.3e})')

    # ===================================================================== B
    print('-' * 94)
    print('B  SF 几何：每组行数 / 偏移')
    print('   scheduler 公式 `current_sf_k_cumsum += ceil_div(align(shape_k, 128), kSFKSpan)`'
          '（scheduler/gemm.cuh:242），kSFKSpan=128（SM90 fp32）')
    cum = 0
    ok = True
    for g, hk in enumerate(host_ks):
        rows = ceil_div(align(hk, 128), 128)
        print(f'     group {g}: host_k={hk:4d}  sf_rows={rows}  sf_offset(cumsum)={cum}')
        cum += rows
    print(f'   累计 SF 行数={cum}  ==  a_sf.shape[1]={a_sf.shape[1]}  => {cum == a_sf.shape[1]}')
    ok &= (cum == a_sf.shape[1])
    # padding < 128 恒成立（host_k=align(logical,128)），故 padding 行**总与真实行同处最后一个 K-block**，
    # 不会产生「纯 padding 的 SF 行」；该 block 的 SF 由真实行 amax 决定，padding 行反量化恒为 0。
    print(f'   注意：每组 host_k-logical_k<128，padding 行总落在最后 1 个 K-block 内 ⇒ 无「纯 padding SF 行」；'
          f'a_sf 全为正={bool((a_sf > 0).all())}（真行 amax>0）')
    ok &= bool((a_sf > 0).all())

    # ===================================================================== C
    print('-' * 94)
    print('C  目标卡(SM100) tn 的 SF 需求：packed UE8M0（一行恒覆盖 gran_k*4 个 K 元素）')
    print('   kernel `shape_sfa_k = ceil_div(shape_k, kGranKA*4)`（sm100_fp8_fp4_gemm_1d1d.cuh:155），'
          'scheduler 传 kSFKSpan=kGranKA*4（:199）')
    print('   输入 SF 约定：MN-major 侧为 [sum(host_k/gran_k), mn]（K 为行）；SM90 nt 侧为 [mn, sum_sf_k]')
    for gran in (32, 128):
        cov = gran * 4
        per_group = [ceil_div(hk, cov) for hk in host_ks]
        packed = sum(per_group)
        in_rows = sum(host_k // gran for host_k in host_ks)
        same = (packed == sf_k) if gran == 32 else None
        print(f'     gran_k={gran:3d}: 每组 packed 行={per_group} 总 packed_sf_k={packed}  '
              f'输入 SF 行数={in_rows}  一行覆盖 {cov} 个 K')
        print(f'              gran32 packed 行数 == SM90 fp32 行数({sf_k}) = {same}'
              f'；gran128 时每组按 512K 向上取整，尾块 padding 会多出额外行')
    print('   H100 上调用 SM100-only 的打包 API（应被 arch 门控拒，smxx_layout.hpp:224）：')
    sf_packed_in = torch.zeros(sum(h // 32 for h in host_ks), m, dtype=torch.float, device='cuda')
    layout = torch.tensor(host_ks, dtype=torch.int32, device='cuda')
    try:
        deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            sf_packed_in, layout, host_ks, 32, 128, False)
        print('     !! 意外通过')
    except Exception as e:
        msg = [ln for ln in str(e).strip().splitlines() if ln.strip()]
        print(f'     拒绝: {type(e).__name__}: {msg[-1][:150] if msg else str(e)[:150]}')

    # ===================================================================== D
    print('-' * 94)
    print('D  FP8 nt（K-major + SF）vs BF16 tn（MN-major，无 SF）同题对拍')
    a_tn, b_tn = build_bf16_tn(A, B, host_ks)
    d_bf16 = run_bf16_tn(a_tn, b_tn, host_ks, G, m, n)
    print(f'   FP8 nt  vs fp64(bf16原值)  relerr={relerr(d_impl, ref):.3e}')
    print(f'   BF16 tn vs fp64(bf16原值)  relerr={relerr(d_bf16, ref):.3e}')
    print(f'   FP8 nt  vs BF16 tn         relerr={relerr(d_impl, d_bf16):.3e}  '
          f'bit-diff={bitfrac(d_impl, d_bf16):.4f}  (差异=FP8 量化 vs BF16)')
    print(f'   两者都输出 [G, m, n]，都满足 D[e]=A_e^T B_e；差异只在精度层，不在布局/累加结构。')

    # ===================================================================== E perf（仅参考）
    print('-' * 94)
    print('E  吞吐（真实 SM90 kernel；8 卡共享，仅量级参考）')
    pk = [4096] * 8
    pm, pn = 4096, 7168
    A2, B2, hk2 = build_problem(pk, pm, pn, seed=2)
    ad, asf, bd, bsf, _, _ = build_fp8_nt(A2, B2, hk2)
    asf_t = deep_gemm.get_mn_major_tma_aligned_tensor(asf)
    bsf_t = deep_gemm.get_mn_major_tma_aligned_tensor(bsf)
    at, bt = build_bf16_tn(A2, B2, hk2)
    cases = [
        ('FP8 nt (SF 已 TMA 对齐->内部早退)', lambda: run_fp8_nt(ad, asf_t, bd, bsf_t, hk2, 8, pm, pn)),
        ('FP8 nt (SF 原始->每次内部 transpose)', lambda: run_fp8_nt(ad, asf, bd, bsf, hk2, 8, pm, pn)),
        ('BF16 tn (无 SF)', lambda: run_bf16_tn(at, bt, hk2, 8, pm, pn)),
    ]
    for name, run in cases:
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        tf = 2 * pm * pn * sum(pk) / dt / 1e12
        print(f'   {name}: G=8 m={pm} n={pn} k/group={pk[0]}  t={dt * 1e6:.0f} us  {tf:.0f} TFLOPS')

    print('=' * 94)
    print(f'B/几何判定 ALL_OK = {ok}')


if __name__ == '__main__':
    main()
