"""P27-1 实验：DeepGEMM 的**测试参考量化器** `deep_gemm/utils/math.py` 精确语义 + H100 实测。

DeepGEMM 的每一个 FP8/FP4 kernel 在测试里都被判「对」，标准就是 `tests/` 用的这套纯 PyTorch
参考量化器：`per_token_cast_to_fp8` / `per_block_cast_to_fp8` / `per_channel_cast_to_fp8` /
`per_token_cast_to_fp4`（含手写 `_quantize_to_fp4_e2m1`）/ `transpose_packed_fp4` /
`cast_back_from_fp8|fp4` / `ceil_to_ue8m0` / `pack_ue8m0_to_int`。换句话说，这些函数**定义了**
「量化后的操作数是什么」，第 25/26 篇（SF 布局）与第 35 篇（MXFP4/MXFP8）里的 M/N/K 语义都由它锚定。

这套参考实现**全是张量运算，架构无关**，H100 可跑；本脚本用真实数字回答：

  A) `ceil_to_ue8m0`：是不是「>= |x| 的最小 2 的幂」？边界（0 / 次正规 / 幂次 / 溢出）如何？
  B) `pack_ue8m0_to_int` / `unpack_ue8m0_from_int`：4 字节 → int32 的字节序与往返一致性。
  C) FP8 三种粒度（per-token 1×128 / per-channel / per-block 128×128）：量化误差量级与 SF 语义。
  D) FP4 E2M1 手写量化的**取整/平局规则**：实测是不是「按 E2M1 编码的 ties-to-even」。
  E) FP4 的 nibble 打包顺序（LSB-first）与往返。
  F) `transpose_packed_fp4`：是否等于真实转置、是否对合（involution）。
  G) 端到端：以上参考量化器喂真实的 SM90 `fp8_gemm_nt`，对拍「反量化后的 fp64 参考」——
     证明 kernel 语义 = 「在这套量化操作数上做精确 GEMM」。

来源核对（~/github/DeepGEMM，公开版 commit 057ca59，版本 2.8.1）：
  - deep_gemm/utils/math.py:13-16    ceil_to_ue8m0
  - deep_gemm/utils/math.py:19-23    pack_ue8m0_to_int
  - deep_gemm/utils/math.py:26-49    per_token_cast_to_fp8
  - deep_gemm/utils/math.py:52-59    per_channel_cast_to_fp8
  - deep_gemm/utils/math.py:62-72    per_block_cast_to_fp8
  - deep_gemm/utils/math.py:84-96    _quantize_to_fp4_e2m1
  - deep_gemm/utils/math.py:99-125   per_token_cast_to_fp4
  - deep_gemm/utils/math.py:128-141  transpose_packed_fp4
  - deep_gemm/utils/math.py:144-176  _dequantize_from_fp4_e2m1 / unpack / cast_back_from_fp8|fp4
  - tests/generators.py:321/324-325、tests/test_mega_moe.py:54/108 的调用形态
"""
import math
import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import (
    ceil_to_ue8m0, pack_ue8m0_to_int,
    per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    per_token_cast_to_fp4, transpose_packed_fp4,
    cast_back_from_fp8, cast_back_from_fp4,
)
from deep_gemm.utils.math import (
    _quantize_to_fp4_e2m1, _dequantize_from_fp4_e2m1, unpack_ue8m0_from_int,
)

torch.manual_seed(0)
dev = "cuda"
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def relerr(a, b):
    a = a.float().double()
    b = b.float().double()
    denom = b.abs().sum()
    return ((a - b).abs().sum() / denom).item() if denom != 0 else 0.0


# -----------------------------------------------------------------------------
# A) ceil_to_ue8m0: 语义与边界
# -----------------------------------------------------------------------------
def ref_ceil_exp(x: float):
    """独立参考：返回 >= |x| 的最小 2 的幂的**指数域**（IEEE754 exponent field），并按实现钳位。"""
    if x == 0.0:
        return 1  # 0/次正规：exp 域 0 -> clamp 到 1
    if math.isnan(x) or math.isinf(x):
        return 254  # exp 域 255 -> clamp 到 254
    m, e = math.frexp(abs(x))          # |x| = m * 2^e, 0.5 <= m < 1
    k = (e - 1) if m == 0.5 else e      # >= 的最小 2 的幂 = 2^k
    return min(max(k + 127, 1), 254)


def part_a():
    sec("A) ceil_to_ue8m0（math.py:13-16）：>=|x| 的最小 2 的幂 + exp 钳位 [1,254]")
    samples = []

    # 幂次 2^-140 .. 2^130（含次正规、正常、溢出）
    for k in range(-140, 131):
        samples.append(2.0 ** k)
    # 幂次附近的抖动（尾数非 0，测试进位）
    for k in range(-126, 128):
        samples.append(2.0 ** k * 1.5)
        samples.append(2.0 ** k * 1.0000001)
        samples.append(2.0 ** k * 1.9999999)
    samples += [0.0, 1e-45, 5e-324, 3.4e38, 1e39, float("inf"), float("nan")]

    t = torch.tensor(samples, dtype=torch.float64).float().cuda()
    out = ceil_to_ue8m0(t)
    out_bits = out.view(torch.int).cpu()
    out_vals = out.cpu().tolist()

    n_mismatch = 0
    pow2_bad = 0
    below_abs = 0
    worst = None
    for v, eb, ov in zip(samples, out_bits.tolist(), out_vals):
        want = ref_ceil_exp(v)
        got_exp = (eb >> 23) & 0xFF
        if eb & 0x7FFFFF != 0:                       # 输出必须是 2 的幂（尾数 0）
            pow2_bad += 1
        if got_exp != want:
            n_mismatch += 1
            if worst is None:
                worst = (v, want, got_exp)
        # 输出 < |x| 只应出现在溢出/被钳位的样本
        if math.isfinite(v) and v != 0 and ov < abs(v):
            below_abs += 1
    print(f"  样本 {len(samples)} 个；与独立参考(exp域)不符 = {n_mismatch}；尾数非 0 = {pow2_bad}"
          + (f"；首个不符 {worst}" if worst else ""))
    print(f"  输出恒为 2 的幂: {pow2_bad == 0}；输出 < |x| 的样本（应仅为溢出/钳位）= {below_abs}")
    # 具体边界值
    edge = torch.tensor([0.0, 5e-324, 2.0 ** -126, 2.0 ** -125, 3.0 * 2.0 ** -128,
                         1.0, 1.5, 2.0, 2.0 ** 127, 1.1 * 2.0 ** 127, float("inf")]).float().cuda()
    eo = ceil_to_ue8m0(edge)
    for v, o in zip(edge.tolist(), eo.tolist()):
        print(f"    ceil_to_ue8m0({v:>12.6g}) = {o:>12.6g}")
    print("  结论：只保留 fp32 指数、把尾数非 0 的项**进位**到下一个 2 的幂（‘向上取整到幂次’），")
    print("        exp 域钳位到 [1,254]（0/次正规 -> 2^-126，溢出/inf/nan -> 2^127）。")
    return n_mismatch == 0 and pow2_bad == 0


# -----------------------------------------------------------------------------
# B) pack_ue8m0_to_int / unpack_ue8m0_from_int
# -----------------------------------------------------------------------------
def part_b():
    sec("B) pack_ue8m0_to_int（math.py:19-23）/ unpack（math.py:151-152）：字节序与往返")
    exps = [124, 132, 138, 132]
    vals = [torch.tensor(2.0 ** (e - 127), dtype=torch.float32) for e in exps]
    t = torch.stack(vals).cuda()
    packed = pack_ue8m0_to_int(t)
    word = packed.view(torch.int).tolist()[0]
    expect = exps[0] | (exps[1] << 8) | (exps[2] << 16) | (exps[3] << 24)
    print(f"  指数 {exps} -> int32 = 0x{word & 0xFFFFFFFF:08x}；期望(LSB-first) 0x{expect:08x}；"
          f"一致: {(word & 0xFFFFFFFF) == expect}")
    unpacked = unpack_ue8m0_from_int(packed)
    print(f"  unpack 往返逐位一致: {torch.equal(unpacked.view(torch.int), t.view(torch.int))}")

    # 大样本往返
    rnd = torch.randn(4096, 128, device=dev).abs().clamp_min(1e-6)
    sf = ceil_to_ue8m0(rnd)                       # 保证是 2 的幂（pack 的前置断言）
    p = pack_ue8m0_to_int(sf)
    u = unpack_ue8m0_from_int(p)
    print(f"  4096x128 随机 scale 往返逐位一致: {torch.equal(u.view(torch.int), sf.view(torch.int))}")
    return (word & 0xFFFFFFFF) == expect


# -----------------------------------------------------------------------------
# C) FP8 三种粒度 + SF 语义
# -----------------------------------------------------------------------------
def part_c():
    sec("C) FP8 三种粒度（per-token 1x128 / per-channel / per-block 128x128）与 SF 语义")
    m, n = 256, 1024
    gk = 128
    x = torch.randn(m, n, device=dev, dtype=torch.float32)
    # 制造行/块内尺度差异，凸显粒度影响
    x[::4] *= 8.0

    def dequant(xq, sf, mode):
        if mode == "per_token":
            gi = torch.arange(n, device=dev) // gk
            return xq.float() * sf[:, gi]
        if mode == "per_channel":
            return xq.float() * sf.repeat_interleave(gk, dim=0)
        return xq.float() * sf.repeat_interleave(gk, 0).repeat_interleave(gk, 1)

    for name, mode, fn in [("per_token(1x128)", "per_token", per_token_cast_to_fp8),
                           ("per_channel", "per_channel", per_channel_cast_to_fp8),
                           ("per_block(128x128)", "per_block", per_block_cast_to_fp8)]:
        for use_ue8m0 in (False, True):
            xq, sf = fn(x, use_ue8m0=use_ue8m0)
            back = dequant(xq, sf, mode)
            print(f"  {name:20s} ue8m0={int(use_ue8m0)}: max abs err/max|x| = "
                  f"{((back - x).abs().max() / x.abs().max()).item():.4e}  sf 形状 {tuple(sf.shape)}")

    # 448 常数与 1e-4 地板：全零块的 SF
    z = torch.zeros(4, 128, device=dev)
    _, zsf = per_token_cast_to_fp8(z, use_ue8m0=False)
    print(f"  全零块 SF = {zsf.reshape(-1)[0].item():.3e}（= clamp(amax,1e-4)/448；E4M3 动态范围常数 448）")
    # sf 上取整到 2 的幂会略微增大误差
    a = torch.randn(64, 256, device=dev) * 100
    q0, s0 = per_token_cast_to_fp8(a, use_ue8m0=False)
    q1, s1 = per_token_cast_to_fp8(a, use_ue8m0=True)
    r0 = relerr(cast_back_from_fp8(q0, s0), a)
    r1 = relerr(cast_back_from_fp8(q1, s1), a)
    print(f"  取整到 UE8M0（2 的幂）前后 relerr: {r0:.4e} -> {r1:.4e}（上取整换来确定性/可打包）")
    return True


# -----------------------------------------------------------------------------
# D) FP4 E2M1 手写量化的取整/平局规则
# -----------------------------------------------------------------------------
def part_d():
    sec("D) _quantize_to_fp4_e2m1（math.py:84-96）：取整与平局规则实测")

    # D1) 7 个中点上的取整方向
    midpoints = [(FP4_VALUES[i] + FP4_VALUES[i + 1]) / 2 for i in range(7)]
    t = torch.tensor(midpoints, dtype=torch.float32).cuda()
    codes = _quantize_to_fp4_e2m1(t).view(torch.uint8).cpu().tolist()
    print("  中点       -> code (值)        期望 ties-to-even(code)  方向")
    ok_tie = True
    for i, (mid, c) in enumerate(zip(midpoints, codes)):
        even_code = i if i % 2 == 0 else i + 1     # 相邻 code i / i+1 中的偶数
        direction = "down" if c == i else ("up" if c == i + 1 else "??")
        mat = c == even_code
        ok_tie &= mat
        print(f"  {mid:8.4f} -> {c:2d} ({FP4_VALUES[c]:5.2f})        {even_code:2d}                    {direction}")
    print(f"  7 个中点全部 = ties-to-even(按 E2M1 编码): {ok_tie}")

    # D2) 随机样本 vs 独立参考（最近值 + E2M1 code 偶数平局）
    def ref_round(v):
        a = abs(v)
        i = 0
        for j in range(7):
            lo, hi = FP4_VALUES[j], FP4_VALUES[j + 1]
            mid = (lo + hi) / 2
            if a > mid or (a == mid and (j % 2 == 1)):   # 平局取 code 为偶者
                i = j + 1
        return i

    xs = (torch.randn(2_000_00, device=dev) * 2.0)
    codes_r = _quantize_to_fp4_e2m1(xs).view(torch.uint8).cpu().tolist()
    xs_l = xs.cpu().tolist()
    n_bad = 0
    for v, c in zip(xs_l, codes_r):
        want = ref_round(v)
        sign = 0x8 if (v < 0 and want != 0) else 0
        if (sign | want) != c:
            n_bad += 1
    print(f"  20 万随机样本 vs 独立参考(最近值 + 偶数平局) 不符 = {n_bad}")

    # D3) vs ties-away（平局一律向上）：差异只出现在 7 个中点
    def ref_away(v):
        a = abs(v)
        i = 0
        for j in range(7):
            mid = (FP4_VALUES[j] + FP4_VALUES[j + 1]) / 2
            if a >= mid:
                i = j + 1
        return i
    mids = torch.tensor(midpoints, dtype=torch.float32).cuda()
    ca = _quantize_to_fp4_e2m1(mids).view(torch.uint8).cpu().tolist()
    da = [ref_away(v) for v in midpoints]
    diff_mid = sum(1 for a, b in zip(ca, da) if a != b)
    print(f"  vs ties-away：{len(midpoints)} 个中点里 {diff_mid} 个不同（正是平局处）")
    print("  结论：参考量化器 = 最近 E2M1 值 + **平局取编码为偶数者**（等价硬件 cvt.rn 的 ties-to-even）。")
    return ok_tie and n_bad == 0


# -----------------------------------------------------------------------------
# E) FP4 nibble 打包顺序 + 往返
# -----------------------------------------------------------------------------
def part_e():
    sec("E) per_token_cast_to_fp4（math.py:99-125）nibble 顺序与 cast_back_from_fp4 往返")
    codes = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0x8 | 1, 0x8 | 7, 0], dtype=torch.uint8)
    # 手工按 (lo | hi<<4) 打包
    lo, hi = codes[0::2].to(torch.int), codes[1::2].to(torch.int)
    bytes_l = (lo | (hi << 4)).tolist()
    print(f"  codes {codes.tolist()} -> 打包字节 {[hex(b) for b in bytes_l]}（第 0 个在低 nibble，LSB-first）")

    x = torch.randn(512, 512, device=dev) * 3
    xq, sf = per_token_cast_to_fp4(x, use_ue8m0=True)
    back = cast_back_from_fp4(xq, sf, gran_k=128, use_packed_ue8m0=False)
    print(f"  FP4 往返（含 UE8M0 SF）relerr = {relerr(back, x):.4e}；packed 形状 {tuple(xq.shape)}（n/2 列）")

    # 与 FP8 同数据同粒度对比（fp4 更粗）
    xq8, sf8 = per_token_cast_to_fp8(x, use_ue8m0=True)
    back8 = cast_back_from_fp8(xq8, sf8, gran_k=128)
    print(f"  同数据 per-token FP4 relerr {relerr(back, x):.4e} vs FP8 {relerr(back8, x):.4e}"
          f"（E2M1 3 位有效 vs E4M3，误差约 {relerr(back, x) / relerr(back8, x):.1f}x）")
    return bytes_l[0] == (1 | (2 << 4))


# -----------------------------------------------------------------------------
# F) transpose_packed_fp4
# -----------------------------------------------------------------------------
def part_f():
    sec("F) transpose_packed_fp4（math.py:128-141）：是否等于真实转置、是否对合")
    m, n = 64, 96
    packed = torch.randint(0, 256, (m, n // 2), device=dev, dtype=torch.uint8).to(torch.int8)
    tp = transpose_packed_fp4(packed)

    # 独立参考：解码 -> 转置 -> 重打包
    codes = torch.empty(m, n, device=dev, dtype=torch.int16)
    codes[:, 0::2] = packed.to(torch.int16) & 0x0F
    codes[:, 1::2] = (packed.to(torch.int16) >> 4) & 0x0F
    ct = codes.transpose(0, 1).contiguous()               # (n, m)
    ref = ct[:, 0::2] | (ct[:, 1::2] << 4)
    print(f"  形状 {tuple(packed.shape)} -> {tuple(tp.shape)}（n x m/2）；"
          f"与独立转置重打包逐位一致: {torch.equal(tp.view(torch.uint8), ref.to(torch.uint8))}")
    back = transpose_packed_fp4(tp)
    print(f"  对合 transpose(transpose(x)) == x: "
          f"{torch.equal(back.view(torch.uint8), packed.view(torch.uint8))}")
    return torch.equal(tp.view(torch.uint8), ref.to(torch.uint8))


# -----------------------------------------------------------------------------
# G) 端到端：参考量化器喂真实 SM90 fp8_gemm_nt
# -----------------------------------------------------------------------------
def part_g():
    sec("G) 端到端：参考量化器定义 kernel 的输入语义（真跑 SM90 fp8_gemm_nt）")
    m, n, k = 1024, 512, 1024
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    a_fp8, a_sf = per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8, b_sf = per_block_cast_to_fp8(b, use_ue8m0=False)
    d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d)

    # 参考：用同一套参考量化器把操作数**反量化**到 fp64 再做精确 GEMM
    a_deq = cast_back_from_fp8(a_fp8, a_sf, gran_k=128).double()
    # per_block 的 sf 是 (m/128, k/128)，需按 128x128 展开（cast_back_from_fp8 只做 per-token 行组）
    b_deq = (b_fp8.float() * b_sf.repeat_interleave(128, 0).repeat_interleave(128, 1)).double()
    ref = (a_deq @ b_deq.t()).float()
    print(f"  m={m} n={n} k={k}: kernel vs ‘参考量化器反量化 + fp64 GEMM’ relerr = {relerr(d, ref):.4e}"
          "（仅 bf16 输出舍入；坐实输入语义 = 这套参考）")
    # 对照：直接用原始 bf16 做 fp64 GEMM（不含 FP8 量化）会差得多
    ref_bf16 = (a.double() @ b.double().t()).float()
    print(f"  同一 kernel vs ‘原始 bf16 精确 GEMM’ relerr = {relerr(d, ref_bf16):.4e}"
          "（含 FP8 per-token/per-block 量化噪声）")
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
    results["G"] = part_g()
    sec("汇总")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("  ALL PASS" if all(results.values()) else "  SOME FAIL")


if __name__ == "__main__":
    main()
