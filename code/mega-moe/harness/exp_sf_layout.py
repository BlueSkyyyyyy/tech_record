"""P15-1 实验：DeepGEMM 块级 scale（SF）布局转换 / UE8M0 打包精读 + H100 实测。

DeepGEMM 所有 FP8/FP4 kernel 的输入都要带「每 (gran_mn × gran_k) 块一个 FP32 scale」。
裸 scale 是 [mn, ceil(k/gran_k)] 行主序，但 kernel 的 TMA 描述符要求 **MN-major + 16B 对齐**；
SM100 更进一步要求把 scale 量化成 **UE8M0**（只留 fp32 的 8 位指数）并按 4 个打包进一个 int32。
这些都在 host 侧 `transform_sf_into_required_layout` 里完成，GPU 侧由
`impls/smxx_layout.cuh` 的 `transpose_fp32` / `transpose_and_pack_fp32_into_ue8m0`
/ `pack_fp32_into_ue8m0` 三个 kernel 承载。**其中 transpose 与打包 kernel 都不是 SM100 专属**，
本机 H100(sm90) 可以真编译真运行，本脚本用真实数字回答：

  A) `get_tma_aligned_size` 语义（16B 对齐）逐点核对。
  B) `get_mn_major_tma_aligned_tensor`（SM90 transpose_fp32 kernel）：值是否等于转置、
     输出 stride 是否 (1, tma_aligned_mn)、已 MN-major 时是否走 early-return（幂等）。
  C) `get_mn_major_tma_aligned_packed_ue8m0_tensor`：CUDA kernel vs torch 参考实现
     （复刻 smxx_layout.hpp:108-131）逐位一致；UE8M0 只保留指数；4 字节 LSB-first 打包；
     TMA 对齐行距；torch 回退路径（不连续）一致。
  D) PSUM 布局变体：gap 行被跳过、有效行值正确、gap 行写安全有限码。
  E) `transform_sf_into_required_layout` 的 recipe 分派在 SM90 上的可达/拒绝分支。
  F) 端到端：真跑 `fp8_gemm_nt`（SM90 1D2D），用显式变换过的 SF 与裸 SF 对比逐位一致，
     并说明 host API 内部本就调用同一变换。

来源核对（~/github/DeepGEMM，公开版 commit 057ca59，版本 2.8.1）：
  - csrc/apis/layout.hpp:11-58  transform_sf_into_required_layout 分派
  - csrc/apis/layout.hpp:60-116 pair / k-grouped 变体
  - csrc/utils/layout.hpp:101-140  check_sf_layout
  - csrc/utils/math.hpp:23-27    get_tma_aligned_size
  - csrc/jit_kernels/impls/smxx_layout.hpp:48-216  transpose / packed ue8m0 host 编排
  - deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:10-72   transpose_fp32
  - deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:76-191  transpose_and_pack_fp32_into_ue8m0
  - deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:193-293 pack_fp32_into_ue8m0
"""
import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import (
    get_tma_aligned_size,
    get_mn_major_tma_aligned_tensor,
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    align, ceil_to_ue8m0,
)

torch.manual_seed(0)
dev = "cuda"


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def diff(a, b):
    return calc_diff(a.float(), b.float())


# -----------------------------------------------------------------------------
# A) get_tma_aligned_size
# -----------------------------------------------------------------------------
def part_a():
    sec("A) get_tma_aligned_size：16 字节对齐（math.hpp:23-27）")
    ok = True
    for elem_size, exp_div in ((4, 4), (1, 16), (2, 8)):
        for x in (1, 15, 16, 17, 31, 32, 33, 127, 128, 4095):
            got = get_tma_aligned_size(x, elem_size)
            want = align(x, 16 // elem_size)
            if got != want:
                ok = False
                print(f"  MISMATCH x={x} es={elem_size}: got={got} want={want}")
    print(f"  16B 对齐公式 align(x, 16/elem_size) 逐点核对: {'PASS' if ok else 'FAIL'}")
    print(f"  get_tma_aligned_size(4095,4) = {get_tma_aligned_size(4095,4)}  (= 4096, fp32 scale 行距)"
          f"  (4095,1) = {get_tma_aligned_size(4095,1)} (= 4096, int8)")
    return ok


# -----------------------------------------------------------------------------
# B) get_mn_major_tma_aligned_tensor (SM90 transpose_fp32)
# -----------------------------------------------------------------------------
def part_b():
    sec("B) get_mn_major_tma_aligned_tensor：行主序 -> MN-major + TMA 对齐（smxx_layout.cuh:10-72）")
    ok = True
    for mn, sf_k in ((512, 16), (4096, 56), (137, 9), (256, 64)):
        sf = torch.randn(mn, sf_k, device=dev, dtype=torch.float32)
        out = get_mn_major_tma_aligned_tensor(sf)
        aligned = get_tma_aligned_size(mn, 4)
        # 值：应为原张量的转置布局（逻辑 [mn, sf_k] 不变，只是 stride 不同）
        vdiff = diff(out, sf)
        stride_ok = (out.stride(0) == 1 and out.stride(1) == aligned)
        contig = out.is_contiguous()
        # 幂等：已 MN-major 时 early-return 返回同一对象
        out2 = get_mn_major_tma_aligned_tensor(out)
        idempotent = out2.data_ptr() == out.data_ptr()
        if vdiff > 1e-6 or not stride_ok:
            ok = False
        print(f"  mn={mn:5} sf_k={sf_k:4}: val_diff={vdiff:.2e} stride=({out.stride(0)},{out.stride(1)}) "
              f"want=(1,{aligned}) contiguous={contig} idempotent_ptr_eq={idempotent}")
    print(f"  转置 + 对齐 + 幂等: {'PASS' if ok else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# C) get_mn_major_tma_aligned_packed_ue8m0_tensor
# -----------------------------------------------------------------------------
def _torch_ref_packed_ue8m0(sf):
    """复刻 smxx_layout.hpp:108-131 的 get_mn_major_tma_aligned_packed_ue8m0_tensor_torch。"""
    sf2 = sf if sf.dim() == 3 else sf.unsqueeze(0)
    ue8m0 = sf2.contiguous().view(torch.int32).bitwise_right_shift(23).to(torch.uint8)
    num_b, mn, k = sf2.shape
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros(num_b, aligned_mn, aligned_k, dtype=torch.uint8, device=sf.device)
    padded[:, :mn, :k] = ue8m0
    padded = padded.view(-1).view(torch.int32).view(num_b, aligned_mn, aligned_k // 4)
    out = torch.empty_strided((num_b, aligned_mn, aligned_k // 4),
                              (aligned_mn * (aligned_k // 4), 1, aligned_mn),
                              dtype=torch.int32, device=sf.device)
    out = out.copy_(padded)[:, :mn]
    return out.squeeze(0) if sf.dim() == 2 else out


def part_c():
    sec("C) UE8M0 打包：只留指数 + 4 字节 LSB-first（smxx_layout.cuh:161-190）")
    ok = True
    for mn, sf_k in ((512, 128), (4096, 256), (256, 64), (1024, 32)):
        # 构造合法 SF：符号 0、尾数 0，仅指数有效（2 的幂）
        exp = torch.randint(120, 140, (mn, sf_k), device=dev, dtype=torch.int32)
        sf = (exp << 23).view(torch.float32).contiguous()
        out = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
        ref = _torch_ref_packed_ue8m0(sf)
        aligned = get_tma_aligned_size(mn, 4)
        packed_k = (sf_k + 3) // 4
        # 逐位一致
        bit_equal = torch.equal(out, ref)
        # 手动拆包验证 LSB-first：byte i = exponent of k*4+i
        view = out.contiguous().view(torch.uint8).view(mn, packed_k, 4)
        byte_ok = True
        if sf_k % 4 == 0:
            for j in range(4):
                byte_ok = byte_ok and torch.equal(view[:, :, j].to(torch.int32),
                                                  (sf.view(torch.int32) >> 23)[:, j::4])
        stride_ok = (out.stride(0) == 1 and out.stride(1) == aligned)
        if not (bit_equal and byte_ok and stride_ok):
            ok = False
        print(f"  mn={mn:5} sf_k={sf_k:4} -> packed[{mn},{packed_k}] stride={tuple(out.stride())} "
              f"(want (1,{aligned})): cuda==torch_ref {bit_equal} LSB-first {byte_ok} stride_ok {stride_ok}")
        # 展示第一个 int32 的字节分解
        if mn == 512 and sf_k == 128:
            v = out[0, 0].item() & 0xFFFFFFFF
            print(f"    例 out[0,0] = 0x{v:08x} -> 4 个 ue8m0 指数字节 = "
                  f"{[v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]}，"
                  f"对应 sf[0, 0:4] 指数 = {(sf[0, :4].view(torch.int32) >> 23).tolist()}")
    # torch 回退路径：mn % 4 != 0（smxx_layout.hpp:187 -> torch 实现 108-131）
    mn, sf_k = 137, 128
    exp = torch.randint(120, 140, (mn, sf_k), device=dev, dtype=torch.int32)
    sf = (exp << 23).view(torch.float32).contiguous()
    out_fb = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    ref_fb = _torch_ref_packed_ue8m0(sf)
    print(f"  torch 回退（mn={mn} 非 4 倍数）: cuda-vs-torch 逐位 {torch.equal(out_fb, ref_fb)}")
    ok = ok and torch.equal(out_fb, ref_fb)
    # 已 MN-major 输入 -> 走 pack_fp32_into_ue8m0（不转置，smxx_layout.cuh:193-293）
    mn, sf_k = 1024, 128
    exp = torch.randint(120, 140, (mn, sf_k), device=dev, dtype=torch.int32)
    sf = (exp << 23).view(torch.float32).contiguous()
    mn_major = get_mn_major_tma_aligned_tensor(sf)             # [mn,sf_k] stride (1,mn)
    direct = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)  # transpose_and_pack
    via_pack = get_mn_major_tma_aligned_packed_ue8m0_tensor(mn_major)  # pack only
    same = torch.equal(direct, via_pack)
    print(f"  已 MN-major 输入走 pack kernel：与 transpose_and_pack 逐位一致 {same} "
          f"(stride in = {tuple(mn_major.stride())})")
    ok = ok and same
    # 为什么 SF 必须是「只有指数」的 2 的幂：kernel 只搬 8 位指数（smxx_layout.cuh:183-186），
    # 尾数/符号被丢弃；ceil_to_ue8m0 先把任意正 SF 上取整到 2 的幂（math.py:13-16）。
    raw = torch.rand(4, 8, device=dev) * 0.5 + 1e-3
    pow2 = ceil_to_ue8m0(raw)
    is_pow2 = bool((pow2.view(torch.int32) & 0x7FFFFF == 0).all().item())
    ge_raw = bool((pow2 >= raw).all().item() and (pow2 < raw * 2).all().item())
    print(f"  ceil_to_ue8m0 结果只含指数（尾数=0）: {is_pow2}；且 raw <= pow2 < 2*raw: {ge_raw}")
    ok = ok and is_pow2 and ge_raw
    print(f"  UE8M0 打包: {'PASS' if ok else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# D) PSUM 布局变体
# -----------------------------------------------------------------------------
def part_d():
    sec("D) PSUM 布局：gap 行跳过、有效行正确（smxx_layout.cuh:106-126 / 187-189）")
    ok = True
    mn, k = 1024, 256
    packed_k = k // 4
    # 每组的逻辑 K 结束（PSUM，前缀和；组间有 m_alignment 对齐 gap）
    m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    groups_end = [300, 700, 1024]
    psum = torch.tensor(groups_end, device=dev, dtype=torch.int32)
    exp = torch.randint(120, 140, (mn, k), device=dev, dtype=torch.int32)
    sf = (exp << 23).view(torch.int32).view(torch.float32).contiguous()
    out = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, psum_layout=psum)
    ref = _torch_ref_packed_ue8m0(sf)
    # 有效行的值应与无条件打包一致；gap 行不同（kernel 写安全码/0）
    starts = [0] + [align(groups_end[i], m_alignment) for i in range(len(groups_end) - 1)]
    valid = torch.zeros(mn, dtype=torch.bool, device=dev)
    for s, e in zip(starts, groups_end):
        valid[s:e] = True
    valid_rows_equal = torch.equal(out.contiguous()[:, :][valid], ref.contiguous()[:, :][valid])
    gap_rows_equal = torch.equal(out.contiguous()[:, :][~valid], ref.contiguous()[:, :][~valid])
    gap_elems = int((~valid).sum().item())
    # gap 行必须被写（不是未初始化），且为有限值（UE8M0 0xff=NaN 被排除）
    gap_bytes = out.contiguous()[~valid].view(torch.uint8)
    no_nan = bool((gap_bytes != 0xFF).all().item())
    print(f"  m_alignment={m_alignment} groups_end={groups_end} gap_rows={gap_elems}/{mn}")
    print(f"  有效行 == 无条件打包: {valid_rows_equal}；gap 行 == 原值: {gap_rows_equal}（期望 False）")
    print(f"  gap 行无 0xFF(NaN) 码: {no_nan}")
    if not (valid_rows_equal and no_nan):
        ok = False
    print(f"  PSUM 布局: {'PASS' if ok else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# E) recipe 分派在 SM90 上的可达 / 拒绝分支（layout.hpp:34-57）
# -----------------------------------------------------------------------------
def part_e():
    sec("E) transform_sf_into_required_layout 的 recipe 分派（SM90 可达性）")
    arch = deep_gemm.testing.get_arch_major()
    print(f"  arch_major = {arch}")
    mn, k = 512, 256  # 真实 (m,k)

    def mk_sf(gran_mn, gran_k, rows=None, k_scales=None):
        rows = rows if rows is not None else (mn // gran_mn)
        k_scales = k_scales if k_scales is not None else (k // gran_k)
        exp = torch.randint(120, 140, (rows, k_scales), device=dev, dtype=torch.int32)
        return (exp << 23).view(torch.float32).contiguous()

    def try_tf(name, sf, **kw):
        try:
            out = deep_gemm.transform_sf_into_required_layout(sf=sf, mn=mn, k=k, **kw)
            print(f"  [OK  ] {name}: in{tuple(sf.shape)} -> stride={tuple(out.stride())} dtype={out.dtype}")
            return out
        except Exception as e:
            print(f"  [REJ ] {name}: {str(e).strip().splitlines()[-1][:96]}")
            return None

    # 2-tuple recipe = (gran_mn, gran_k)，此时 **不能** 传 is_sfa（layout.hpp:27-29）
    try_tf("recipe=(1,128)   SFA形状[512,2]", mk_sf(1, 128), recipe=(1, 128))
    try_tf("recipe=(128,128) SFB形状[4,2]", mk_sf(128, 128), recipe=(128, 128))
    # (1,32)：gran 32 是 SM100 专属，SM90 形状合法但最终 UNREACHABLE
    try_tf("recipe=(1,32)    [SM100 专属]", mk_sf(1, 32), recipe=(1, 32))
    try_tf("recipe=(1,32) disable_ue8m0_cast=True", mk_sf(1, 32), recipe=(1, 32),
           disable_ue8m0_cast=True)
    # 3-tuple recipe = (gran_mn_a, gran_mn_b, gran_k)，此时 **必须** 传 is_sfa（layout.hpp:23-25）
    try_tf("recipe=(1,128,128) is_sfa=True", mk_sf(1, 128), recipe=(1, 128, 128), is_sfa=True)
    try_tf("recipe=(1,1,32)   is_sfa=False", mk_sf(1, 32), recipe=(1, 1, 32), is_sfa=False)
    # 形状不符：recipe 决定 SF 粒度，传错形状会被 check_sf_layout 拦下
    try_tf("recipe=(128,128) 却传 SFA[512,2]", mk_sf(1, 128), recipe=(128, 128))
    # 契约反例：2-tuple 带 is_sfa / 3-tuple 不带 is_sfa 都断言失败
    try_tf("recipe=(1,128) is_sfa=True [契约反例]", mk_sf(1, 128), recipe=(1, 128), is_sfa=True)
    try_tf("recipe=(1,128,128) 无 is_sfa [契约反例]", mk_sf(1, 128), recipe=(1, 128, 128))
    return True


# -----------------------------------------------------------------------------
# F) 端到端：真跑 fp8_gemm_nt（SM90 1D2D）并核验 SF 变换一致
# -----------------------------------------------------------------------------
def part_f():
    sec("F) 端到端 SM90 fp8 gemm：裸 SF 与显式变换 SF 逐位一致")
    from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
    m, n, k = 1024, 512, 1024
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    a_fp8, a_sf = per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8, b_sf = per_block_cast_to_fp8(b, use_ue8m0=False)
    d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    # 默认 recipe：SM90 为 (1,128,128)（get_default_recipe）
    deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d)
    # 参考：反量化后 fp32 matmul
    ref = a.float() @ b.float().t()
    print(f"  m={m} n={n} k={k}: fp8 gemm vs fp32 ref relerr = {diff(d, ref):.4e}（FP8 量化噪声量级）")

    # 显式把 SF 变换成 kernel 要求的布局，检查与内部变换一致（幂等）
    tr_a = deep_gemm.transform_sf_into_required_layout(a_sf, m, k, (1, 128))
    tr_b = deep_gemm.transform_sf_into_required_layout(b_sf, n, k, (128, 128))
    print(f"  显式变换 SFA stride={tuple(tr_a.stride())} dtype={tr_a.dtype}")
    print(f"  显式变换 SFB stride={tuple(tr_b.stride())} dtype={tr_b.dtype}")
    tr_a2 = deep_gemm.transform_sf_into_required_layout(tr_a, m, k, (1, 128))
    print(f"  SFA 二次变换 ptr 相同（幂等）: {tr_a2.data_ptr() == tr_a.data_ptr()}")

    # 用显式变换过的 SF 再跑一次（kernel 内部会再变换，若幂等则结果逐位一致）
    d2 = torch.empty_like(d)
    deep_gemm.fp8_gemm_nt((a_fp8, tr_a), (b_fp8, tr_b), d2)
    print(f"  裸 SF vs 显式变换 SF 输出逐位相等: {torch.equal(d, d2)}")
    return True


def main():
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}  "
          f"cc {torch.cuda.get_device_capability(0)}")
    results = {}
    results["A"] = part_a()
    results["B"] = part_b()
    results["C"] = part_c()
    results["D"] = part_d()
    results["E"] = part_e()
    results["F"] = part_f()
    sec("汇总")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  ALL PASS: {all(results.values())}")


if __name__ == "__main__":
    main()
