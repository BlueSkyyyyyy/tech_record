"""P3-3 实验：Mega MoE L1/L2 epilogue 的数值原语在 H100(sm90) 上的纯 Python 核对。

Mega MoE 的 MMA 与 epilogue（`sm100_fp8_fp4_mega_moe.cuh`）本身是 SM100 专属，
本机跑不了；但其**数值语义**可以在 H100 上用位运算逐位复现：

  1) `get_ue8m0_sf_exp`（`deep_gemm/include/deep_gemm/common/math.cuh:114-135`）：
     用一次整数加法进位替 `ceil`，把 per-32 amax 映射成 UE8M0 幂次 scale。
     这里逐位复现并与「最小 e 使 qmax*2^e >= amax」的参考对齐。
  2) `get_ue8m0_sf_inv`（`math.cuh:137-147`）：scale 与 inverse 的 biased exponent
     之和恒为 254，故乘积精确为 1（验证量化无额外误差）。
  3) 按 32 元素分组的 block-scale FP8 量化-反量化往返：用 kernel 的幂次 scale
     对比「理想」absmax/qmax 浮点 scale 的数值误差与是否溢出。
  4) L1 epilogue 的 in-place SwiGLU（`...cuh:1108-1134`）：gate/up 先 round 到
     bf16 再做 silu，乘以 topk 权重后的数值，与全 fp32 参考对比。

来源核对（`~/github/DeepGEMM`，公开版 commit `057ca59`）：
  - `deep_gemm/include/deep_gemm/common/math.cuh:110-147`
  - `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1058-1219`
不涉及 SM100 kernel，仅验证 epilogue 的**数值原语**。
"""
import math
import sys

import numpy as np


# ----------------------------------------------------------------------------
# 1) `get_ue8m0_sf_exp` 的逐位复现（输入 FP32 amax，输出 biased exponent）
# ----------------------------------------------------------------------------
def ue8m0_sf_exp(amax, quant='e4m3'):
    """复刻 math.cuh:114-135。amax 为 float32 标量/数组，返回 uint32 biased exp。"""
    x = np.asarray(amax, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.uint64)
    if quant == 'e4m3':
        mantissa_bits, qmax_mantissa, qmax_exp, min_sf_exp = 23, 0x60, 8, 105
    else:  # e2m1
        mantissa_bits, qmax_mantissa, qmax_exp, min_sf_exp = 23, 0x40, 2, 1
    mask = (1 << mantissa_bits) - 1
    qmax_mantissa <<= (mantissa_bits - 7)
    rounded = (bits + mask - qmax_mantissa) >> mantissa_bits
    out = np.maximum(rounded, np.uint64(min_sf_exp + qmax_exp)) - qmax_exp
    return out.astype(np.uint32)


def ue8m0_sf_exp_ref(amax, quant='e4m3'):
    """参考：最小整数 e 使 qmax * 2^e >= amax；再 clamp 到下界，返回 biased exp。"""
    qmax = 448.0 if quant == 'e4m3' else 6.0
    min_e = -22 if quant == 'e4m3' else -126
    x = np.asarray(amax, dtype=np.float64)
    out = np.empty_like(x)
    pos = x > 0
    ratio = np.where(pos, x, 1.0) / qmax
    # e = ceil(log2(ratio))；用 frexp 精确取，避免 log2 边界抖动
    mant, expn = np.frexp(ratio)          # ratio = mant * 2^expn, mant in [0.5,1)
    e = np.where(mant == 0.5, expn - 1, expn).astype(np.int64)  # ceil(log2)
    e = np.maximum(e, min_e)
    bi = (e + 127)
    out = np.where(pos, bi, 127 + min_e)
    return out.astype(np.uint32)


def ue8m0_sf_inv(sf_exp):
    """复刻 math.cuh:137-147 的 FP32 分支：scale = 2^(sf_exp-127)，inv = 2^(127-sf_exp)。"""
    return ((254 - np.asarray(sf_exp, dtype=np.uint64)) << 23).astype(np.uint32).view(np.float32)


def part1_bit_trick():
    print('== 1) `get_ue8m0_sf_exp` 位运算 vs 参考 (ceil(log2(amax/qmax))) ==')
    ok = True
    for quant in ('e4m3', 'e2m1'):
        rng = np.random.default_rng(0)
        # 覆盖亚正规附近、幂次边界、普通范围
        vals = np.concatenate([
            np.array([0.0, 1e-30, 1e-4, 447.0, 448.0, 449.0, 895.0, 896.0, 1e6], dtype=np.float32),
            (rng.random(20000).astype(np.float32) * 2000.0),
            np.exp2(rng.integers(-20, 20, size=5000)).astype(np.float32),
        ])
        got = ue8m0_sf_exp(vals, quant)
        want = ue8m0_sf_exp_ref(vals, quant)
        bad = int((got != want).sum())
        if bad:
            idx = np.nonzero(got != want)[0][:5]
            for i in idx:
                print(f'   [{quant}] amax={vals[i]!r} got={got[i]} want={want[i]}')
        print(f'   [{quant}] {len(vals)} 个样本，不符 {bad} 个')
        ok &= bad == 0
    return ok


# ----------------------------------------------------------------------------
# 2) scale / inverse 乘积恒为 1
# ----------------------------------------------------------------------------
def part2_inv():
    print('\n== 2) UE8M0 scale 与 inverse 乘积 ==')
    sf = np.arange(105, 137, dtype=np.uint32)  # e4m3 的合法 exponent 区间
    scale = ((sf.astype(np.uint64)) << 23).astype(np.uint32).view(np.float32)
    inv = ue8m0_sf_inv(sf)
    prod = scale.astype(np.float64) * inv.astype(np.float64)
    err = np.abs(prod - 1.0).max()
    print(f'   sf_exp 范围 [{sf.min()},{sf.max()}]，scale*inv 最大偏差 {err:.3e}')
    return err == 0.0


# ----------------------------------------------------------------------------
# 3) 按 32 元素分组的 block-scale FP8 量化-反量化
# ----------------------------------------------------------------------------
def quant_block_fp8(xn, use_kernel_sf=True):
    """xn: numpy [rows, K]，按 K 方向 32 元素一组量化到 e4m3（torch FP8），返回反量化值与 SF。"""
    import torch
    x = torch.from_numpy(xn)
    rows, K = x.shape
    assert K % 32 == 0
    g = x.reshape(rows, K // 32, 32)
    amax = g.abs().amax(dim=2)                          # [rows, K/32]
    qmax = 448.0
    if use_kernel_sf:
        sf_exp = ue8m0_sf_exp(amax.numpy().astype(np.float32), 'e4m3').astype(np.int64)
        scale = np.exp2(sf_exp - 127.0).astype(np.float32)                 # 2^e
    else:
        scale = np.where(amax.numpy() > 0, amax.numpy() / qmax, 1.0).astype(np.float32)
    scale_t = torch.from_numpy(scale)
    q = (g / scale_t[:, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dq = q.to(torch.float32) * scale_t[:, :, None]
    overflow = int((g.abs() > (qmax * scale_t[:, :, None]) + 1e-3).sum())
    return dq.reshape(rows, K).numpy(), scale, overflow


def part3_blockscale():
    print('\n== 3) block-scale FP8 量化往返（K=256，按 32 分组，torch FP8） ==')
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((64, 256)).astype(np.float32)
         * rng.uniform(0.1, 8.0, size=(64, 1)).astype(np.float32))

    def relerr(a, b):
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))

    dq_k, sf_k, ov_k = quant_block_fp8(x, use_kernel_sf=True)
    dq_i, sf_i, ov_i = quant_block_fp8(x, use_kernel_sf=False)
    print(f'   kernel 幂次 scale : max abs err {np.abs(dq_k - x).max():.3e}, '
          f'相对 {relerr(dq_k, x):.3e}, 溢出 {ov_k}')
    print(f'   理想 amax/qmax    : max abs err {np.abs(dq_i - x).max():.3e}, '
          f'相对 {relerr(dq_i, x):.3e}, 溢出 {ov_i}')
    # 幂次 scale 一定 >= 理想 scale（不溢出），但最多浪费 2x 动态范围
    ratio = float((sf_k / np.where(sf_i > 0, sf_i, 1.0)).max())
    print(f'   幂次 scale / 理想 scale 最大比值 (应 <= 2): {ratio:.4f}')
    return ov_k == 0 and ratio <= 2.0 + 1e-6


# ----------------------------------------------------------------------------
# 4) L1 in-place SwiGLU 数值（bf16 round + silu*up*topk_weight）
# ----------------------------------------------------------------------------
def bf16_round(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    # round-to-nearest-even 到 bf16（保留高 16 位）
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((bits + rounding_bias) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def part4_swiglu():
    print('\n== 4) L1 epilogue in-place SwiGLU 数值 ==')
    rng = np.random.default_rng(2)
    M, I = 256, 128
    gate = rng.standard_normal((M, I)).astype(np.float32) * 3.0
    up = rng.standard_normal((M, I)).astype(np.float32) * 3.0
    w = rng.uniform(0.5, 1.5, size=(M, 1)).astype(np.float32)

    # 全 fp32 参考
    ref = (gate / (1.0 + np.exp(-gate))) * up * w

    # kernel：先 round 到 bf16，再以 fp32 做 silu，乘 topk 权重
    gb = bf16_round(gate)
    ub = bf16_round(up)
    got = (gb / (1.0 + np.exp(-gb.astype(np.float32)))) * ub.astype(np.float32) * w

    rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    print(f'   bf16-round gate/up vs 全 fp32：max abs {np.abs(got - ref).max():.3e}, 相对 {rel:.3e}')

    # 若不做 bf16 round 的对照
    noref = (gate / (1.0 + np.exp(-gate))) * up * w
    print(f'   （不做 bf16 round 的自身 vs 参考应恒 0: {np.abs(noref - ref).max():.3e}）')
    return rel < 5e-3


def main():
    ok = True
    ok &= part1_bit_trick()
    ok &= part2_inv()
    ok &= part3_blockscale()
    ok &= part4_swiglu()
    print(f'\nALL_OK = {ok}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
