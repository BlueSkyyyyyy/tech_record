#!/usr/bin/env python3
# P33-1 实验 driver：DeepGEMM **测试断言辅助** `tests/utils.py`（H100/sm90 真机可跑）。
#
# 前面第 31 篇拆了 device 侧 epilogue 算子、第 27 篇拆了 cuBLASLt 回退、第 17/40/41 篇拆了测试基座，
# 但 `tests/test_bf16.py` / `test_einsum.py` / `test_fp8_fp4.py` 共用的这 6 个断言/转换辅助一直没讲——
# 它们正是「每个 kernel 正确性判据」的可执行 oracle。本实验逐条核对：
#   A) convert_to_fp8                —— E4M3 精确表示 E2M1；nibble LSB-first
#   B) to_cublaslt_vec16_sf_layout   —— MXFP4 per-32 SF -> NVFP4 per-16 UE4M3 分块
#   C) add_sf_k_padding              —— UE8M0 NaN 毒化 padding
#   D) assert_direct_output_matches_fp32_accumulation  —— 真实 SM90 BF16 GEMM，direct vs 累积
#   E) assert_stochastic_bf16_matches_fp32_accumulation —— 测试 oracle 与 device 哈希逐位对拍
#   F) make_cublas_gemm + cublaslt_gemm_nt —— 真实 cuBLASLt FP8 回退
#   G) assert_psum_zero_padding      —— 真实 SM90 m-grouped PSUM + 负例
#
# 目标卡代号敏感：一律用「目标卡」。

import os
import sys
import glob
import math
import importlib.util
import subprocess

import numpy as np
import torch

import deep_gemm
from deep_gemm.testing import assert_bitwise_equal, calc_diff

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
TESTS_DIR = os.path.join(DG_ROOT, 'tests')
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)

# 直接加载 tests/utils.py（模块名固定为 dg_tests_utils，避免与它自己的 import utils 冲突）
_spec = importlib.util.spec_from_file_location('dg_tests_utils', os.path.join(TESTS_DIR, 'utils.py'))
tutils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tutils)

from generators import (                              # noqa: E402
    enumerate_m_grouped_contiguous, generate_m_grouped_contiguous,
)

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def dprint(*a):
    print(*a)


def mkbins(dtype, data: torch.Tensor) -> torch.Tensor:
    return data.detach().cpu().contiguous().numpy().view(dtype).reshape(-1)


# ===========================================================================
def part_a_convert_to_fp8():
    dprint('\n===== A) convert_to_fp8：E4M3 精确表示 E2M1 =====')
    torch.manual_seed(0)
    # 造任意 packed FP4 字节（每个字节两个 nibble，LSB-first），覆盖全部 16 个 E2M1 码
    lo = torch.arange(16, dtype=torch.uint8).repeat_interleave(16).to(torch.int64)
    hi = torch.arange(16, dtype=torch.uint8).repeat(16).to(torch.int64)
    packed = (lo | (hi << 4)).to(torch.uint8).view(torch.int8)
    R, C = 16, 16
    packed = packed.reshape(R, C)
    sf_dummy = torch.ones(R, C, C, dtype=torch.float, device=DEV)  # 函数不读 sf
    packed = packed.to(DEV)

    conv_fp8, _ = tutils.convert_to_fp8((packed, sf_dummy))

    # 参考：解包 nibble -> _dequantize_from_fp4_e2m1 -> 精确转 E4M3
    from deep_gemm.utils.math import _dequantize_from_fp4_e2m1
    u = packed.view(torch.uint8)
    lo_n = (u & 0x0f).to(torch.int8)
    hi_n = ((u >> 4) & 0x0f).to(torch.int8)
    # 独立参考：数值用参考量化器反量化，符号位（含宽零 -> -0.0）单独 OR 进 E4M3 码
    # （`_dequantize_from_fp4_e2m1` 按 P37 语义无负零，故这里显式补 E4M3 的符号位）
    def to_fp8_with_sign(nib):
        val = _dequantize_from_fp4_e2m1(nib).to(torch.float8_e4m3fn).view(torch.uint8)
        return val | ((nib.view(torch.uint8) & 8) << 4)
    ref_lo = to_fp8_with_sign(lo_n)
    ref_hi = to_fp8_with_sign(hi_n)
    ref = torch.stack((ref_lo, ref_hi), dim=-1).flatten(-2).view(torch.float8_e4m3fn)
    assert_bitwise_equal(conv_fp8, ref, 'convert_to_fp8 exhaustive 256 codes')
    report('A1 convert_to_fp8 全 256 个字节码 vs 独立参考逐位一致', True,
           f'{conv_fp8.numel()} elems')

    # 幂等 / 已是 FP8 时原样返回
    id_out, _ = tutils.convert_to_fp8((conv_fp8, sf_dummy))
    report('A2 已 FP8 输入幂等', torch.equal(id_out, conv_fp8), '')

    # MN-major（stride(-1)!=1）分支会原样转置再处理，输出形状为输入的转置（×2）；这里验证
    # 其与「先转置成连续再转换」的转置逐位一致（即该分支只是搬运 stride，不改变元素值）
    d = packed.reshape(R, C).mT.contiguous().mT      # 非连续，stride(-1)=R
    mnc, _ = tutils.convert_to_fp8((d, sf_dummy))
    inner = d.mT.contiguous()                        # 连续副本，值 = d 的转置
    inner_conv, _ = tutils.convert_to_fp8((inner, sf_dummy))
    report('A3 MN-major 分支 = 转置后转换再转置', torch.equal(mnc, inner_conv.mT),
           f'out shape={tuple(mnc.shape)}')

    # 逐元素语义：解包 nibble（LSB-first）后反量化，应与转换结果数值一致
    from deep_gemm.utils.math import _dequantize_from_fp4_e2m1
    u = packed.reshape(R, C).view(torch.uint8)
    low = (u & 0x0f).to(torch.int8)
    high = ((u >> 4) & 0x0f).to(torch.int8)
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    deq = _dequantize_from_fp4_e2m1(unpacked)
    report('A3b 解包 LSB-first + 反量化 == 转换结果数值', torch.equal(conv_fp8.float(), deq),
           'low nibble -> even column')

    # 抽样打印 LUT 映射
    lut_vals = []
    for code in range(8):
        b = torch.tensor([[code]], dtype=torch.int8, device=DEV)
        o, _ = tutils.convert_to_fp8((b, torch.ones(1, 1, 1, device=DEV)))
        lut_vals.append(hex(o.view(torch.uint8).flatten()[0].item()))
    dprint(f'    E2M1 idx 0..7 -> E4M3 码 {lut_vals}')


# ===========================================================================
def part_b_cublaslt_vec16():
    dprint('\n===== B) to_cublaslt_vec16_sf_layout：MXFP4 per-32 -> NVFP4 per-16 UE4M3 =====')
    torch.manual_seed(1)
    mn, sf_k32 = 137, 20                    # 故意非 128 / 4 倍数，触发 padding
    exp32 = torch.randint(-9, 9, (mn, sf_k32), device=DEV)
    sf = torch.pow(2.0, exp32.float())      # per-32（MXFP4）正 2 的幂 scale

    out = tutils.to_cublaslt_vec16_sf_layout(sf)
    num_mn_tiles, num_k_tiles = math.ceil(mn / 128), math.ceil(sf_k32 * 2 / 4)
    exp_shape = (num_mn_tiles, num_k_tiles, 32, 4, 4)
    report('B1 输出形状/类型（5D 分块）', tuple(out.shape) == exp_shape and out.dtype == torch.uint8,
           f'shape={tuple(out.shape)} dtype={out.dtype}')

    # 独立复刻（逐行等价 tests/utils.py:98-110）
    def ue4m3_encode(e):
        return torch.where(e >= -6, (e + 7) << 3, 1 << (e + 9)).to(torch.uint8)
    ue4m3 = ue4m3_encode(exp32).repeat_interleave(2, dim=1)     # 宽 = 2*sf_k32
    mn2, sf_k16 = ue4m3.shape
    padded = torch.zeros((num_mn_tiles * 128, num_k_tiles * 4), dtype=torch.uint8, device=DEV)
    padded[:mn2, :sf_k16] = ue4m3
    ref = padded.view(num_mn_tiles, 4, 32, num_k_tiles, 4).permute(0, 3, 2, 1, 4).contiguous()
    report('B2 与独立复刻逐位一致', torch.equal(out, ref), '')

    # padding 区全 0：把 5D 还原回 2D 再看
    back = out.permute(0, 3, 2, 1, 4).reshape(num_mn_tiles * 128, num_k_tiles * 4)
    ok_pad = bool((back[mn2:, :] == 0).all()) and bool((back[:, sf_k16:] == 0).all())
    report('B3 padding 区全 0', ok_pad,
           f'MN pad {num_mn_tiles*128-mn2} 行 / K pad {num_k_tiles*4-sf_k16} 列')

    # 逐列解码 UE4M3 应还原原 fp32 scale（2 的幂），零误差
    def ue4m3_to_float(c):
        c = c.to(torch.int32)
        e4, m = (c >> 3) & 0xf, c & 0x7
        return torch.where(e4 == 0, m.float() * (2.0 ** -9), (1 + m / 8.0) * torch.pow(2.0, e4 - 7))
    dec = ue4m3_to_float(ue4m3_encode(exp32))
    ok_dec = float((dec - sf).abs().max()) == 0.0
    report('B4 UE4M3 编解码回原 2 的幂 scale 零误差', ok_dec,
           f'max|dec-sf|={float((dec - sf).abs().max()):.1e}')
    dprint(f'    exp 范围 [{int(exp32.min())},{int(exp32.max())}]，MN tile {num_mn_tiles}，K tile {num_k_tiles}')


# ===========================================================================
def part_c_sf_k_padding():
    dprint('\n===== C) add_sf_k_padding：UE8M0 NaN 毒化 padding =====')
    torch.manual_seed(2)
    for k_grouped in (False, True):
        mn, sf_k = 100, 9
        sf = torch.randn(mn, sf_k, device=DEV) if not k_grouped else torch.randn(sf_k, mn, device=DEV)
        # k_grouped 时 mn = size(-1)
        pad = 4
        out = tutils.add_sf_k_padding(sf, padding=pad, k_grouped=k_grouped)
        vals_ok = torch.equal(out, sf)
        mn_axis = sf.size(-1 if k_grouped else -2)
        pitch = (mn_axis + 3) // 4 * 4 + pad
        if k_grouped:
            stride_ok = out.stride() == (pitch, 1)
        else:
            stride_ok = out.stride() == (1, pitch)
        report(f'C k_grouped={k_grouped} 值不变 + stride={out.stride()}', vals_ok and stride_ok,
               f'pitch={pitch}')

        # 读底层 storage 确认 padding 全 -1（UE8M0 NaN 毒化）
        raw = torch.empty(out.untyped_storage().nbytes(), dtype=torch.uint8, device=DEV)
        raw.set_(out.untyped_storage())
        raw_f = raw.view(torch.float32).reshape(-1, pitch)
        poison = (raw_f[:, mn_axis:] == -1.0).all().item()
        report(f'C2 k_grouped={k_grouped} padding 全 -1（毒化）', bool(poison), '')

    # 3D 变体
    sf3 = torch.randn(4, 100, 9, device=DEV)
    out3 = tutils.add_sf_k_padding(sf3, padding=4)
    report('C3 3D 变体值不变', torch.equal(out3, sf3) and out3.dim() == 3, f'stride={out3.stride()}')


# ===========================================================================
def part_d_direct_vs_accumulate():
    dprint('\n===== D) assert_direct_output_matches_fp32_accumulation：真实 SM90 BF16 GEMM =====')
    deep_gemm.use_deterministic_algorithms(True)
    m, n, k = 512, 512, 512
    torch.manual_seed(3)
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)

    d_fp32 = torch.empty(m, n, device=DEV, dtype=torch.float32)
    deep_gemm.bf16_gemm_nt(a, b, d_fp32)
    base = torch.zeros(m, n, device=DEV, dtype=torch.float32)
    deep_gemm.bf16_gemm_nt(a, b, base, c=base)
    ok = torch.equal(d_fp32, base)
    report('D1 direct fp32 == 从零累积 D=GEMM+0 逐位', ok,
           f'{m}x{n}x{k}')

    d_bf16 = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(a, b, d_bf16)
    ok_bf16 = torch.equal(d_bf16, base.to(torch.bfloat16))
    report('D2 direct bf16 == fp32 基线一次 bf16 cast', ok_bf16, '')

    # 真实 helper：应通过
    def launch(output, acc):
        deep_gemm.bf16_gemm_nt(a, b, output, c=acc)
    try:
        tutils.assert_direct_output_matches_fp32_accumulation(d_fp32, launch, 'P33 D')
        passed = True
        msg = 'no error'
    except AssertionError as e:
        passed = False
        msg = str(e)[:80]
    report('D3 真实 helper 通过', passed, msg)

    # 负例：篡改 1 bit，helper 必须断言失败
    bad = d_fp32.clone()
    bad.view(torch.int32)[0, 0] ^= 1
    try:
        tutils.assert_direct_output_matches_fp32_accumulation(bad, launch, 'P33 D-neg')
        caught = False
    except AssertionError as e:
        caught = True
    report('D4 篡改 1 bit 后 helper 正确报错', caught, '')


# ===========================================================================
def part_e_stochastic_oracle(bin_path):
    dprint('\n===== E) assert_stochastic_bf16_matches_fp32_accumulation：oracle vs device 哈希 =====')
    if not os.path.exists(bin_path):
        report('E0 found sr binary', False, f'{bin_path} missing (skip)')
        return
    work = '/tmp/opencode/p33_sr'
    os.makedirs(work, exist_ok=True)
    W = 64                      # 每行 quartet 数 -> 模型形状 (M, N=4W)
    M = 1024
    num_values = M * 4 * W      # 4 的倍数
    subprocess.run([bin_path, work, str(num_values), str(W)], check=True)
    inp = np.fromfile(os.path.join(work, 'sr_in.bin'), dtype=np.uint32)
    rb = np.fromfile(os.path.join(work, 'sr_rb.bin'), dtype=np.uint32)
    out = np.fromfile(os.path.join(work, 'sr_out.bin'), dtype=np.uint16)

    fp32_in = torch.from_numpy(inp.copy()).view(torch.float32).to(DEV)
    rb_cu = torch.from_numpy(rb.astype(np.int64)).to(DEV)
    out_cu = torch.from_numpy(out.copy()).view(torch.bfloat16).to(DEV)

    Q = num_values // 4
    vals = (fp32_in.view(torch.int32).to(torch.int64) & 0xffffffff).view(Q, 4)
    quartet_n_idx = (torch.arange(Q, device=DEV, dtype=torch.int64) % W)

    # tests/utils.py:65-78 的 torch 公式（逐行复刻）
    h = (vals[..., 0] * 0x5671d42b + vals[..., 1] * 0x9995e499 +
         vals[..., 2] * 0xace1b8a5 + vals[..., 3] * 0xe153538d + quartet_n_idx) & 0xffffffff
    h = (h ^ (h >> 23)) * 0x7feb352d & 0xffffffff
    h ^= h >> 16
    h1 = h * 0x846ca68b & 0xffffffff
    h2 = h * 0xd35a2d97 & 0xffffffff
    h1 ^= h1 >> 11
    h2 ^= h2 >> 11
    random_bits = torch.stack((h1 & 0xffff, h1 >> 16, h2 & 0xffff, h2 >> 16), dim=-1).view(-1)
    rounded = (vals.view(-1) >> 16) + (((vals.view(-1) & 0xffff) + random_bits) >> 16)
    expected = rounded.to(torch.int64).to(torch.uint16).view(torch.bfloat16)

    n_rb = int((random_bits != rb_cu).sum())
    report('E1 torch 公式随机 bit vs device 逐位', n_rb == 0, f'mismatch={n_rb}/{num_values}')
    n_out = int((expected != out_cu).sum())
    report('E2 torch 公式随机舍入 bf16 vs device 逐位', n_out == 0, f'mismatch={n_out}/{num_values}')

    # 真实 helper：把 device 输出当作 direct_output，launch 用同一批 fp32 输入（形状 (M, 4W)）
    fp32_mat = fp32_in.view(M, 4 * W)
    out_mat = out_cu.view(M, 4 * W)
    def launch(output, acc):
        output.copy_(fp32_mat)
        acc.copy_(fp32_mat)
    try:
        tutils.assert_stochastic_bf16_matches_fp32_accumulation(out_mat, launch, 'P33 E')
        passed, msg = True, 'no error'
    except AssertionError as e:
        passed, msg = False, str(e)[:100]
    report('E3 真实 helper 通过（device 输出 == oracle 期望）', passed, msg)

    # 对比：RNE 固定倾向（无偏性对照）
    rne = fp32_mat.view(torch.int32).to(torch.int64) >> 16
    rne_bf16 = rne.to(torch.int64).to(torch.uint16).view(torch.bfloat16)
    frac = (fp32_mat.view(torch.int32).to(torch.int64) & 0xffff).float() / 65536.0
    up = (out_mat.view(torch.int16).to(torch.int64) != rne) & \
         (out_mat.view(torch.int16).to(torch.int64) > rne)
    dprint(f'    E: 随机舍入向上比例={float(up.float().mean()):.4f}，平均小数部分={float(frac.mean()):.4f}')
    dn = int((rne_bf16 != out_mat).sum())
    report('E4 与 RNE 有差异（随机性存在）', dn > 0, f'differ from RNE={dn}/{num_values}')


# ===========================================================================
def part_f_cublaslt(make_cublas_gemm):
    dprint('\n===== F) make_cublas_gemm + cublaslt_gemm_nt：真实 cuBLASLt FP8 =====')
    deep_gemm.use_deterministic_algorithms(False)
    m, n, k = 512, 512, 1024
    torch.manual_seed(4)
    a_f32 = torch.randn(m, k, device=DEV)
    b_f32 = torch.randn(n, k, device=DEV)
    a = a_f32.to(torch.float8_e4m3fn)
    b = b_f32.to(torch.float8_e4m3fn)
    d = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    deep_gemm.cublaslt_gemm_nt(a, b, d)
    ref = a_f32.to(torch.float8_e4m3fn).float() @ b_f32.to(torch.float8_e4m3fn).float().t()
    diff = calc_diff(d, ref)
    report('F1 cublaslt_gemm_nt FP8 语义正确', diff < 1e-5, f'calc_diff={diff:.2e}')

    # make_cublas_gemm 分派（FP8 分支）
    fn = make_cublas_gemm((a, None), (b, None), d, None)
    d2 = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    # 用真实 kernel 输出做等价基线
    d_copy = d.clone()
    fn2 = make_cublas_gemm((a, None), (b, None), d2, None)
    fn2()
    report('F2 make_cublas_gemm 返回可调用且 FP8 分支正确', calc_diff(d2, ref) < 1e-5,
           f'calc_diff={calc_diff(d2, ref):.2e}')

    # int8/NVFP4 分支：MXFP4 per-32 SF 经 to_cublaslt_vec16_sf_layout 转 NVFP4 per-16 后
    # 喂真实 cublaslt_nvfp4_gemm_nt（把 B/F 串起来；H100 走 cuBLASLt，非 SM100 专属）
    try:
        from deep_gemm.utils.math import _dequantize_from_fp4_e2m1
        torch.manual_seed(40)
        def make_fp4(mn):
            packed = torch.randint(0, 256, (mn, k // 2), device=DEV, dtype=torch.uint8).view(torch.int8)
            sf32 = torch.pow(2.0, torch.randint(-6, 7, (mn, k // 32), device=DEV).float())
            return packed, sf32
        ap, asf32 = make_fp4(m)
        bp, bsf32 = make_fp4(n)
        asf16 = tutils.to_cublaslt_vec16_sf_layout(asf32)   # uint8, [ceil(m/128),ceil(k/64),32,4,4]
        bsf16 = tutils.to_cublaslt_vec16_sf_layout(bsf32)
        dn = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
        deep_gemm.cublaslt_nvfp4_gemm_nt((ap, asf16), (bp, bsf16), dn)

        def deq(packed, sf32):
            u = packed.view(torch.uint8)
            lo = (u & 0x0f).to(torch.int8)
            hi = ((u >> 4) & 0x0f).to(torch.int8)
            unp = torch.stack((lo, hi), dim=-1).flatten(-2)
            vals = _dequantize_from_fp4_e2m1(unp)
            gidx = torch.arange(k, device=DEV) // 32
            return vals * sf32[:, gidx]
        ref_nv = deq(ap, asf32) @ deq(bp, bsf32).t()
        diffn = calc_diff(dn, ref_nv)
        report('F3 真实 cublaslt_nvfp4_gemm_nt（per-16 UE4M3 SF 分块布局）', diffn < 2e-2,
               f'calc_diff={diffn:.2e}')
    except Exception as e:
        msg = str(e)
        unsupported = ('not supported' in msg) or ('): 15 ' in msg)
        report('F3 cublaslt_nvfp4：本机 cuBLASLt 不支持（环境，SKIP）' if unsupported
               else 'F3 cublaslt_nvfp4 失败', unsupported, f'{type(e).__name__}: {msg[:70]}')


# ===========================================================================
def part_g_psum_zero_padding():
    dprint('\n===== G) assert_psum_zero_padding：真实 SM90 m-grouped PSUM =====')
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    dprint(f'    mk_alignment = {deep_gemm.get_mk_alignment_for_contiguous_layout()}')

    # 真实 SM90 m-grouped contiguous + PSUM：kEnsureZeroPadding 在 SM90 上取默认 true，
    # 且生成器把 A 的 padding 行置零 -> GEMM 后 D 的 padding 自然为 0，helper 应通过。
    picked = None
    for cfg in enumerate_m_grouped_contiguous(torch.bfloat16):
        _, _, num_groups, empg, n, k, major_a, major_b, use_psum, _ensure = cfg
        if use_psum and major_a.is_k_major():
            picked = cfg
            break
    if picked is None:
        report('G0 找到 PSUM 用例', False, 'none')
        return
    _, _, num_groups, empg, n, k, major_a, major_b, use_psum, _ = picked
    dprint(f'    num_groups={num_groups} expected_m/group={empg} n={n} k={k}')
    torch.manual_seed(5)
    m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(
        num_groups, empg, n, k, major_a, major_b, use_bf16=True, use_psum_layout=True)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, use_psum_layout=True)
    diff = calc_diff(d[valid_mask], ref_d[valid_mask])
    report('G1 真实 SM90 PSUM kernel 有效区语义正确', diff < 1e-5, f'calc_diff={diff:.2e} m={m}')
    try:
        tutils.assert_psum_zero_padding(a, d, grouped_layout, 'BF16')
        passed, msg = True, 'no error'
    except AssertionError as e:
        passed, msg = False, str(e)[:100]
    report('G2 真实 helper 通过（PSUM padding 全零）', passed, msg)

    # 负例：把某组 padding 首行写非零，helper 必须报错
    gl = grouped_layout.cpu().tolist()
    pad_group = next((i for i, cm in enumerate(gl)
                      if cm < ((cm + alignment - 1) // alignment) * alignment), None)
    if pad_group is not None:
        cm = gl[pad_group]
        a2 = a.clone()
        a2[cm] = 1.0
        try:
            tutils.assert_psum_zero_padding(a2, d, grouped_layout, 'BF16')
            caught = False
        except AssertionError:
            caught = True
        report('G3 非零 padding 被检出', caught, f'group={pad_group} logical_end={cm}')


# ===========================================================================
def main():
    dprint(f'GPU: {torch.cuda.get_device_name(0)}  arch={torch.cuda.get_device_capability(0)}')
    dprint(f'DeepGEMM root: {DG_ROOT}')
    dprint(f'deep_gemm {deep_gemm.__version__}  torch {torch.__version__}')

    part_a_convert_to_fp8()
    part_b_cublaslt_vec16()
    part_c_sf_k_padding()
    part_d_direct_vs_accumulate()
    part_e_stochastic_oracle(os.environ.get('TEST_UTILS_SR_BIN', ''))
    part_f_cublaslt(tutils.make_cublas_gemm)
    part_g_psum_zero_padding()

    total, ok = len(RESULTS), sum(RESULTS)
    dprint(f'\n===== P33-1 tests/utils.py 断言辅助：{ok}/{total} PASS=====')
    sys.exit(0 if ok == total else 1)


if __name__ == '__main__':
    main()
