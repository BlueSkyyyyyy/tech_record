"""P135-1 实验：MoE wgrad 的「强制累积」代价——SM90 `k_grouped_nt` 的 fp32 D 回读。

为什么要做这篇（补的缺口）：
  第 15 篇（P7-1）用 ncu 归因反向时给出一条**根因**：SM90 的 wgrad 走 `k_grouped_fp8_gemm_nt_contiguous`
  而它**被 API 强制累积**（`csrc/apis/gemm.hpp:403` 断言 `c.has_value()`；1D1D kernel 的 `D` 必是
  float 且走 `SM90_TMA_REDUCE_ADD_2D`），于是 `D` 被**回读**，把访存流量几乎翻倍 → 算术强度 AI~120
  ≪ H100 FP8 roofline 拐点 591 → 访存受限。第 22 篇（P22-1）也顺带量到「稠密 1D2D 比 1D1D 快 ~1.5×」。
  但**从没有一次对照实验**把这条根因**量化**出来：把**同一个 wgrad 数学**换一种**不强制累积**的
  kernel（`m_grouped` 的 bf16 直写）写出来，看它能快多少。

本篇用一个**数学完全相同**的等价实现来量化它：
  `dW2[e] = A_e^T B_e`（K 轴按专家分组，收缩维=token）是 `k_grouped_nt` 的 wgrad；
  把 A、B 转置成 `[M, K]` / `[N, K]`（K=全部 token）后，**同一个 `dW2`** 就是
  `m_grouped_nt`（M 轴只有一个组）的 dgrad 形状 —— 该 kernel 的 `D` 是 **bf16 直写、无需累积**。
  两者 FLOP 完全相同、操作数字节完全相同，只差：① 是否回读 fp32 `C`；② 输出 fp32 vs bf16；
  ③ K 是否被切成 8 组（每组只有 `sum_k/8` 个 `BLOCK_K`）。

这正好把「强制累积」拆成**两个可分离的因子**，本篇分别量：
  因子 ①（稠密下也成立）fp32 输出 + 回读：`fp8_gemm_nt` 1D1D(fp32,acc) vs 1D2D(bf16) 同形状。
  因子 ②（K 分组造成的）小 K：`k_grouped_nt` vs 稠密 1D1D 同一总形状 [M,N,sum_k]。
  合成：真实 wgrad `k_grouped_nt` vs 等价 dgrad `m_grouped_nt`。

可跑性：SM90 FP8 kernel（H100 可跑）。SM100 的 wgrad `k_grouped_tn_contiguous`（MN-major）为
**仅分析**（见第 12 篇）。本机 8 卡共享，数值/性能为**无争用单进程**测得，但仍受同节点其它任务影响，
报数时注明。

输出重定向：harness/exp_wgrad_accum_cost.out.txt
"""
import os
import sys
import time
import traceback

import torch

import deep_gemm

DG_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
sys.path.insert(0, os.path.join(DG_DIR, 'tests'))

from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout,
)

GRAN = 128
DEV = 'cuda'


# --------------------------------------------------------------------------- helpers
def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def _bench(fn, warm=10, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 1e3 / iters


def quant_k_operand(X, host_ks):
    """k_grouped 的操作数：MN-major 的逐 channel（沿 K）SF。

    X: [total_k, mn] fp32，各组位于累计 host_ks（gap 行为 0）。返回 (data 1D fp8, sf [mn, total/gran])。
    """
    tk, mn = X.shape
    data = torch.zeros(tk * mn, dtype=torch.float8_e4m3fn, device=X.device)
    sfs = []
    off = 0
    for hk in host_ks:
        xg = X[off:off + hk].contiguous()
        xf, sf = per_channel_cast_to_fp8(xg, use_ue8m0=False, gran_k=GRAN)
        data[off * mn:(off + hk) * mn] = xf.T.reshape(-1)
        sfs.append(sf)
        off += hk
    return data, (torch.cat(sfs).T if sfs else X.new_zeros((mn, 0)))


def make_k_grouped(M, N, host_ks):
    """生成一组 k_grouped 操作数（A[total_k,M]、B[total_k,N]）。"""
    tk = sum(host_ks)
    A = torch.randn(tk, M, device=DEV)
    B = torch.randn(tk, N, device=DEV)
    a = quant_k_operand(A, host_ks)
    b = quant_k_operand(B, host_ks)
    layout = torch.tensor(host_ks, device=DEV, dtype=torch.int32)
    d = torch.zeros(len(host_ks), M, N, device=DEV, dtype=torch.float32)
    return A, B, a, b, layout, d


def k_grouped_wgrad(a, b, d, host_ks, layout):
    """真实的 MoE wgrad：被强制累积（C=D，1D1D fp32）。"""
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (a[0], a[1]), (b[0], b[1]), d, host_ks, layout, d, recipe=(1, 1, GRAN))


def make_m_grouped(M, N, total_k, ma):
    """等价的 dgrad 形状：M 轴单组、K=全部 token，D=bf16 直写、无累积。"""
    A2 = torch.randn(M, total_k, device=DEV)
    B3 = torch.randn(1, N, total_k, device=DEV)
    a2 = per_token_cast_to_fp8(A2, use_ue8m0=False, gran_k=GRAN)
    b2f, b2sf = per_block_cast_to_fp8(B3[0], use_ue8m0=False, gran_k=GRAN)
    b2 = (b2f.unsqueeze(0), b2sf.unsqueeze(0))
    m_total = align(M, ma)
    layout_m = torch.zeros(m_total, device=DEV, dtype=torch.int32)
    dm = torch.empty(m_total, N, device=DEV, dtype=torch.bfloat16)
    return a2, b2, layout_m, dm


def m_grouped_wgrad(a, b, d, layout_m):
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, layout_m, disable_ue8m0_cast=True)


# --------------------------------------------------------------------------- parts
def part_a_correctness():
    print('=' * 92)
    print('Part A · 同一 wgrad 数学的两种写法是否等价（正确性 sanity，fresh D）')
    print('-' * 92)
    M, N, G, k_e = 2048, 7168, 8, 512
    host_ks = [align(k_e, GRAN)] * G
    A, B, a, b, layout, d = make_k_grouped(M, N, host_ks)
    d.zero_()                                      # C=D 就地累加，调用前必须清零
    k_grouped_wgrad(a, b, d, host_ks, layout)
    torch.cuda.synchronize()
    k_res = d.sum(0)                                # [M,N] fp32
    # 用【同一份】A、B 造等价的 dgrad 形状（M 轴单组，K=全部 token）
    ma = get_mk_alignment_for_contiguous_layout()
    A2 = A.t().contiguous()                         # [M, sum_k]
    B3 = B.t().unsqueeze(0).contiguous()            # [1, N, sum_k]
    a2 = per_token_cast_to_fp8(A2, use_ue8m0=False, gran_k=GRAN)
    b2f, b2sf = per_block_cast_to_fp8(B3[0], use_ue8m0=False, gran_k=GRAN)
    b2 = (b2f.unsqueeze(0), b2sf.unsqueeze(0))
    m_total = align(M, ma)
    layout_m = torch.zeros(m_total, device=DEV, dtype=torch.int32)
    dm = torch.empty(m_total, N, device=DEV, dtype=torch.bfloat16)
    m_grouped_wgrad(a2, b2, dm, layout_m)
    torch.cuda.synchronize()
    m_res = dm[:M].float()
    print(f'  k_grouped (fp32, K 分 8 组) vs m_grouped (bf16, K 单组) relerr={rel(k_res, m_res):.3e}')
    print(f'  （~3e-2 即 FP8 量化噪声，证明两者是同一 wgrad）')
    return rel(k_res, m_res)


def _report(tag, t, fl, extra=''):
    print(f'  {tag:<34} {t * 1e6:8.1f} us   {fl / t / 1e12:6.0f} TFLOPS{extra}')
    return t


def _g_dense(M, N, k_e, G, mode):
    """G 个独立稠密 GEMM（同 wgrad 的 G× 输出规模），返回 (fn, flops)。

    mode='1d2d'：bf16 直写、无回读；mode='1d1d'：fp32 强制累积。
    固定 k_e 与 G 即固定输出规模 G*M*N，用来隔离「fp32+回读」而不引入 K 分组。
    """
    As, Bs, operands = [], [], []
    for _ in range(G):
        A = torch.randn(M, k_e, device=DEV)
        B = torch.randn(N, k_e, device=DEV)
        operands.append((A, B))
    if mode == '1d2d':
        ds = [torch.empty(M, N, device=DEV, dtype=torch.bfloat16) for _ in range(G)]
        quant = [(per_token_cast_to_fp8(A, use_ue8m0=False),
                  per_block_cast_to_fp8(B, use_ue8m0=False)) for A, B in operands]

        def fn():
            for (a, b), d in zip(quant, ds):
                deep_gemm.fp8_gemm_nt(a, b, d)
    else:
        ds = [torch.zeros(M, N, device=DEV, dtype=torch.float32) for _ in range(G)]
        quant = [(per_token_cast_to_fp8(A, use_ue8m0=False),
                  per_token_cast_to_fp8(B, use_ue8m0=False)) for A, B in operands]

        def fn():
            for (a, b), d in zip(quant, ds):
                deep_gemm.fp8_gemm_nt(a, b, d, d, recipe=(1, 1, 128))
    return fn, 2 * M * N * k_e * G


def part_b_wgrad_vs_dgrad():
    print('=' * 92)
    print('Part B · DeepSeek wgrad 形状：真实 k_grouped(fp32,acc) vs 等价 dgrad 形状(bf16,直写)')
    print('-' * 92)
    ma = get_mk_alignment_for_contiguous_layout()
    T, topk, E = 2048, 2, 8
    sum_k = T * topk
    k_e = sum_k // E
    host_ks = [align(k_e, GRAN)] * E
    res = {}
    for tag, M, N in (('dW2 (M=I,N=H)', 2048, 7168), ('dW1 (M=H,N=2I)', 7168, 4096)):
        fl = 2 * M * N * sum_k
        A, B, a, b, layout, d = make_k_grouped(M, N, host_ks)
        t_k = _report(f'{tag} k_grouped fp32/acc', _bench(lambda: k_grouped_wgrad(a, b, d, host_ks, layout)), fl)
        a2, b2, layout_m, dm = make_m_grouped(M, N, sum_k, ma)
        t_m = _report(f'{tag} m_grouped bf16/直写(单组)', _bench(lambda: m_grouped_wgrad(a2, b2, dm, layout_m)), fl)
        print(f'    -> k_grouped / m_grouped(单组) = {t_k / t_m:.2f}x  '
              f'（注意 m_grouped 单组把各专家【求和】成一个 [M,N]，输出只有 1×；见下）')
        # 同输出规模对照：G 个独立稠密 GEMM，输出 G*M*N
        fn_b, _ = _g_dense(M, N, k_e, E, '1d2d')
        _report(f'{tag} {E}×稠密 1D2D bf16 (G×输出)', _bench(fn_b), fl)
        fn_a, _ = _g_dense(M, N, k_e, E, '1d1d')
        _report(f'{tag} {E}×稠密 1D1D fp32/acc (G×输出)', _bench(fn_a), fl)
        res[tag] = (t_k, t_m)
    return res


def part_c_factor_decomposition():
    print('=' * 92)
    print('Part C · 三个可分离因子（同输出规模对照，逐层隔离）')
    print('-' * 92)
    M, N, E, k_e = 2048, 7168, 8, 512
    sum_k = E * k_e
    fl = 2 * M * N * sum_k
    # 因子①：纯累积（输出 1×、计算受限的形状下 fp32+回读 vs bf16 直写）
    A = torch.randn(M, sum_k, device=DEV)
    B = torch.randn(N, sum_k, device=DEV)
    a1 = per_token_cast_to_fp8(A, use_ue8m0=False)
    b1 = per_token_cast_to_fp8(B, use_ue8m0=False)
    d1 = torch.zeros(M, N, device=DEV, dtype=torch.float32)
    t_1d1d = _report('[因子①] 稠密 1D1D fp32/acc (输出 1×)',
                     _bench(lambda: deep_gemm.fp8_gemm_nt(a1, b1, d1, d1, recipe=(1, 1, 128))), fl)
    a2 = per_token_cast_to_fp8(A, use_ue8m0=False)
    b2 = per_block_cast_to_fp8(B, use_ue8m0=False)
    d2 = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    t_1d2d = _report('[因子①] 稠密 1D2D bf16/直写 (输出 1×)',
                     _bench(lambda: deep_gemm.fp8_gemm_nt(a2, b2, d2)), fl)
    print(f'    -> 因子①（fp32+回读 vs bf16 直写，同输出规模）= {t_1d1d / t_1d2d:.2f}x')
    # 因子②：K 分组 kernel 开销（同 G× 输出规模、同为 fp32/acc）
    host_ks = [align(k_e, GRAN)] * E
    _A, _B, ak, bk, layout, dk = make_k_grouped(M, N, host_ks)
    t_kg = _report('[因子②] k_grouped fp32/acc (G× 输出)',
                   _bench(lambda: k_grouped_wgrad(ak, bk, dk, host_ks, layout)), fl)
    fn_g1d1d, _ = _g_dense(M, N, k_e, E, '1d1d')
    t_g1d1d = _report(f'[因子②] {E}×稠密 1D1D fp32/acc (G× 输出)',
                      _bench(fn_g1d1d), fl)
    print(f'    -> 因子②（K 分组 vs 逐专家稠密、同 G× 输出规模）= {t_kg / t_g1d1d:.2f}x')
    # 因子③：在 G× 输出规模下，fp32 累积 vs bf16 直写
    fn_g1d2d, _ = _g_dense(M, N, k_e, E, '1d2d')
    t_g1d2d = _report(f'[因子③] {E}×稠密 1D2D bf16 (G× 输出)',
                      _bench(fn_g1d2d), fl)
    print(f'    -> 因子③（同 G× 输出规模下 fp32/acc vs bf16 直写）= {t_g1d1d / t_g1d2d:.2f}x')
    print(f'    -> 合成（k_grouped / G×稠密 1D2D）= {t_kg / t_g1d2d:.2f}x')
    return t_1d1d, t_1d2d, t_kg


def part_d_per_group_k_sweep():
    print('=' * 92)
    print('Part D · per-group K 扫描：K 被切多小，k_grouped 掉多少')
    print('-' * 92)
    M, N, E = 2048, 7168, 8
    ma = get_mk_alignment_for_contiguous_layout()
    print(f'  {"per-group k":>12} {"sum_k":>8} {"k_grouped us":>14} {"TFLOPS":>8} {"m_grouped(1组) us":>18} {"TFLOPS":>8}')
    for k_e in (128, 256, 512, 1024, 2048, 4096):
        host_ks = [align(k_e, GRAN)] * E
        sum_k = sum(host_ks)
        A, B, a, b, layout, d = make_k_grouped(M, N, host_ks)
        fl = 2 * M * N * sum_k
        t_k = _bench(lambda: k_grouped_wgrad(a, b, d, host_ks, layout))
        a2, b2, layout_m, dm = make_m_grouped(M, N, sum_k, ma)
        t_m = _bench(lambda: m_grouped_wgrad(a2, b2, dm, layout_m))
        print(f'  {k_e:>12} {sum_k:>8} {t_k * 1e6:>14.1f} {fl / t_k / 1e12:>8.0f} '
              f'{t_m * 1e6:>18.1f} {fl / t_m / 1e12:>8.0f}')


def part_e_traffic_model():
    print('=' * 92)
    print('Part E · 访存账 + 算术强度（AI）：为什么强制累积把 wgrad 推成访存受限')
    print('-' * 92)
    M, N, E, k_e = 2048, 7168, 8, 512
    sum_k = E * align(k_e, GRAN)
    fl = 2 * M * N * sum_k
    op = (M + N) * sum_k                 # A+B fp8 操作数
    sf = (M + N) * sum_k // GRAN * 4     # 粗略 SF
    out_fp32 = E * M * N * 4             # 每专家一个 fp32 [M,N]
    rows = [
        ('k_grouped 实际 (fp32+回读)', op + sf + out_fp32, out_fp32, 'A^T B 每专家一份 + 读回 C'),
        ('理想 wgrad (fp32, 无回读)', op + sf, out_fp32, '同输出规模、无 C 回读'),
        ('理想 wgrad (bf16, 无回读)', op + sf, E * M * N * 2, '同输出规模、bf16 直写'),
        ('m_grouped 单组 (求和, 输出 1×)', op + sf, M * N * 2, '各专家求和，非 wgrad'),
    ]
    print(f'  {"":<30} {"读 MB":>9} {"写 MB":>9} {"合计 MB":>10} {"FLOP":>9} {"AI":>8}')
    for tag, rd, wr, note in rows:
        print(f'  {tag:<30} {rd / 1e6:>9.1f} {wr / 1e6:>9.1f} {(rd + wr) / 1e6:>10.1f} '
              f'{fl / 1e12:>8.2f}T {fl / (rd + wr):>8.1f}   {note}')
    print('  注：AI=每字节 DRAM 流量对应的 FLOP；H100 FP8 计算 roofline 拐点约 591 FLOP/B。')
    print(f'  → 实际 wgrad 的流量里，C 回读占 {out_fp32 / (op + sf + 2 * out_fp32) * 100:.0f}%，'
          f'每专家 fp32 输出（读+写）占 {(2 * out_fp32) / (op + sf + 2 * out_fp32) * 100:.0f}%。')


def main():
    print(f'deep_gemm {deep_gemm.__version__}, device={torch.cuda.get_device_name(0)}, '
          f'sms={deep_gemm.get_num_sms()}')
    for name, fn in (('Part A', part_a_correctness), ('Part B', part_b_wgrad_vs_dgrad),
                     ('Part C', part_c_factor_decomposition), ('Part D', part_d_per_group_k_sweep),
                     ('Part E', part_e_traffic_model)):
        try:
            fn()
        except Exception:
            print(f'{name} FAILED:\n' + traceback.format_exc())
    print('=' * 92)
    print('done')


def ncu_main(which):
    """给 ncu 单次 launch 用：`python exp_wgrad_accum_cost.py --ncu k|m|d1|d2`。

    每次只发射**一次**目标 kernel（先 warmup 若干次让 JIT/cache 命中，ncu 只 profile 最后一次）。
    """
    M, N, E, k_e = 2048, 7168, 8, 512
    host_ks = [align(k_e, GRAN)] * E
    sum_k = sum(host_ks)
    ma = get_mk_alignment_for_contiguous_layout()
    if which == 'k':      # Wgrad 真实路径：k_grouped fp32/强制累积
        A, B, a, b, layout, d = make_k_grouped(M, N, host_ks)
        for _ in range(3):
            k_grouped_wgrad(a, b, d, host_ks, layout)
        torch.cuda.synchronize()
        d.zero_()
        k_grouped_wgrad(a, b, d, host_ks, layout)
    elif which == 'm':    # 等价 dgrad 形状：m_grouped bf16/直写，M 轴单组
        a2, b2, layout_m, dm = make_m_grouped(M, N, sum_k, ma)
        for _ in range(3):
            m_grouped_wgrad(a2, b2, dm, layout_m)
        torch.cuda.synchronize()
        m_grouped_wgrad(a2, b2, dm, layout_m)
    elif which == 'd1':   # 稠密 1D1D fp32/acc
        A = torch.randn(M, sum_k, device=DEV)
        B = torch.randn(N, sum_k, device=DEV)
        a = per_token_cast_to_fp8(A, use_ue8m0=False)
        b = per_token_cast_to_fp8(B, use_ue8m0=False)
        d = torch.zeros(M, N, device=DEV, dtype=torch.float32)
        for _ in range(3):
            deep_gemm.fp8_gemm_nt(a, b, d, d, recipe=(1, 1, 128))
        torch.cuda.synchronize()
        deep_gemm.fp8_gemm_nt(a, b, d, d, recipe=(1, 1, 128))
    elif which == 'd2':   # 稠密 1D2D bf16/直写
        A = torch.randn(M, sum_k, device=DEV)
        B = torch.randn(N, sum_k, device=DEV)
        a = per_token_cast_to_fp8(A, use_ue8m0=False)
        b = per_block_cast_to_fp8(B, use_ue8m0=False)
        d = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
        for _ in range(3):
            deep_gemm.fp8_gemm_nt(a, b, d)
        torch.cuda.synchronize()
        deep_gemm.fp8_gemm_nt(a, b, d)
    else:
        raise SystemExit(f'unknown {which}')
    torch.cuda.synchronize()


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--ncu':
        ncu_main(sys.argv[2])
    else:
        main()
