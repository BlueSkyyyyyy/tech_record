"""P7-2 实验：HyperConnection prenorm GEMM（`tf32_hc_prenorm_gemm`）在 H100(sm90) 上的实测。

Mega mHC / HyperConnection 里，每个 token 需要「把残差投影到 mix 系数」并同时拿到
RMSNorm 的分母 `sqr_sum = sum_k A^2`。DeepGEMM 把这件两件事融进一个 kernel：

    D[m, n] = sum_k A[m, k] * B[n, k]      （A=bf16 激活 / B=fp32 权重 / D=fp32）
    S[m]    = sum_k A[m, k]^2              （与上面共享对 A 的那一次读）

本 kernel 有 **SM90 实现**（`sm90_tf32_hc_prenorm_gemm.cuh`），所以本机 H100 可以真跑。
本脚本用真实数字回答：

  1) 数值正确性：D 与 S 对拍 torch **真 fp32** 参考（含 split-K 求和），用归一化 Frobenius 误差。
  2) TF32 语义：kernel 的 MMA 是 TF32（10-bit 尾数）。A 是 bf16（8-bit 尾数，上转无损），
     B 是 fp32（24-bit 尾数）——用「截断」与「就近舍入(RNE)」两种 TF32 模拟，
     判断硬件到底是截断还是舍入。
  3) S 融合的价值：S 在对 A 做 bf16→fp32 转换时顺手平方累加，不额外读 A；
     对比 torch 单独做 `(a.float()**2).sum(-1)` 的耗时（= 再读一遍 A 的代价）。
  4) split-K 分解：num_splits=16 时 D 是 3D `[16,m,n]`，逐 split 覆盖一段 K；核对可加性。
  5) 参数契约：sm90 限制 `n<=32 and n%8==0`、`k%block_k==0`（block_k=64）等在 host 端被拒。
  6) 性能：TFLOPS / 有效带宽，并贴合 roofline 说明本形状是**访存受限**（N 很小、K 很大）。

来源核对（`~/github/DeepGEMM`，公开版 commit `057ca59`）：
  - `csrc/apis/hyperconnection.hpp:10-55`（host 入口 + 形状/dtype 校验；arch9→sm90, arch10→sm100）
  - `csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp:15-114`（block_m=64/block_k=64、
    num_math_threads=128、num_tma_threads=128、`n<=32`、swizzle、stage 预算、launch）
  - `deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh`
      123-146 TMA 载入 warp；147-285 math warp（bf16→fp32 寄存器 A、`__bfloat1622float2`、
      201-202 顺手累加平方；205-231 TF32 WGMMA；233-242 warp 内归约写 sqr_sum；246-284 CD 上屏）
  - `deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh`（SM100 专属，仅分析：
      A 先由 cast/reduce warpgroup 转 fp32 写 TMEM，UMMA 走 TS 形式，`SM100_MMA_TF32_TS`）
"""
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, count_bytes


def rne_tf32(x: torch.Tensor) -> torch.Tensor:
    """把 fp32 就近舍入到 TF32（保留 10-bit 尾数），round-to-nearest-even。"""
    b = x.float().contiguous().view(torch.int32)
    lsb = (b >> 13) & 1
    rounded = b + 0x0FFF + lsb
    rounded = rounded & ~0x1FFF
    return rounded.view(torch.float32)


def trunc_tf32(x: torch.Tensor) -> torch.Tensor:
    """把 fp32 截断到 TF32（直接丢低 13 位）。"""
    b = x.float().contiguous().view(torch.int32)
    return (b & ~0x1FFF).view(torch.float32)


def nerr(x: torch.Tensor, y: torch.Tensor) -> float:
    """归一化 Frobenius 误差 ||x-y||_2 / ||y||_2（对整体张量，非逐元素）。"""
    x, y = x.double(), y.double()
    denom = y.norm()
    if denom == 0:
        return (x - y).norm().item()
    return ((x - y).norm() / denom).item()


def time_fn(fn, warmup=10, iters=50):
    """用 cuda event 计时，返回中位数（秒）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    return ts[iters // 2] / 1e3


def run_case(m, n, k, num_splits):
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float, device='cuda')
    if num_splits is None:
        d = torch.empty((m, n), dtype=torch.float, device='cuda')
        s = torch.empty((m,), dtype=torch.float, device='cuda')
    else:
        d = torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
        s = torch.empty((num_splits, m), dtype=torch.float, device='cuda')

    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
    final_d = d if num_splits is None else d.sum(0)
    final_s = s if num_splits is None else s.sum(0)

    # torch 参考用**真 fp32**（关掉 TF32 matmul），避免 cuBLAS 也用 TF32 造成混淆
    ref_d = (a.float() @ b.T)
    ref_s = a.float().square().sum(-1)
    tf32_trunc = a.float() @ trunc_tf32(b).T
    tf32_rne = a.float() @ rne_tf32(b).T

    return dict(
        a=a, b=b, d=d, s=s, final_d=final_d, final_s=final_s,
        ref_d=ref_d, ref_s=ref_s, tf32_trunc=tf32_trunc, tf32_rne=tf32_rne,
        cd_diff=calc_diff(final_d, ref_d), cs_diff=calc_diff(final_s, ref_s),
        d_vs_trunc=nerr(final_d, tf32_trunc),
        d_vs_rne=nerr(final_d, tf32_rne),
        d_vs_fp32=nerr(final_d, ref_d),
        s_relerr=nerr(final_s, ref_s),
    )


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f'GPU: {torch.cuda.get_device_name(0)}  arch sm{torch.cuda.get_device_capability(0)[0]}')
    print(f'deep_gemm: {deep_gemm.__version__}  (TF32 matmul for torch ref: DISABLED)\n')

    print('=' * 84)
    print('1) 正确性 + TF32 语义（D 对拍真 fp32 / 截断TF32 / RNE-TF32；S 对拍 fp32 精确和）')
    print('   D.vs_X 为归一化 Frobenius 误差 ||Δ||/||ref||')
    print('=' * 84)
    shapes = [(13, 24, 28672), (137, 24, 7680), (4096, 24, 7168), (8192, 24, 28672),
              (4096, 32, 7168), (4096, 8, 7168), (64, 24, 7168)]
    print(f'{"m":>6} {"n":>4} {"k":>7} {"splits":>6} | {"D.vs_fp32":>10} {"D.vs_trunc":>11} '
          f'{"D.vs_rne":>9} | {"S.vs_fp32":>10} {"cd_diff":>9} {"cs_diff":>9}')
    for (m, n, k) in shapes:
        for num_splits in (None, 16):
            r = run_case(m, n, k, num_splits)
            print(f'{m:>6} {n:>4} {k:>7} {str(num_splits):>6} | '
                  f'{r["d_vs_fp32"]:>10.3e} {r["d_vs_trunc"]:>11.3e} {r["d_vs_rne"]:>9.3e} | '
                  f'{r["s_relerr"]:>10.3e} {r["cd_diff"]:>9.2e} {r["cs_diff"]:>9.2e}')
    print('\n  说明：D 是 TF32 GEMM，对真 fp32 参考有 ~几 e-4 的归一化误差（10-bit 尾数）；')
    print('        与截断/RNE 两种 TF32 模拟比，谁更小就说明硬件用的是哪种（见下方判定）。')

    r = run_case(4096, 24, 7168, None)
    verdict = '截断(truncate)' if r['d_vs_trunc'] <= r['d_vs_rne'] else '就近舍入(RNE)'
    print(f'\n  判定：kernel D 与「{verdict}」参考更接近 '
          f'(trunc={r["d_vs_trunc"]:.3e} vs rne={r["d_vs_rne"]:.3e}; '
          f'比值 rne/trunc={r["d_vs_rne"]/r["d_vs_trunc"]:.2f})')

    print('\n' + '=' * 84)
    print('2) S（sqr_sum）是「搭车」算出来的：与 fp64 精确平方和逐元素对比')
    print('=' * 84)
    r = run_case(4096, 24, 7168, None)
    exact_f64 = r['a'].double().square().sum(-1)
    abs_err = (r['final_s'].double() - exact_f64).abs()
    print(f'  经:k=7168, 行和量级 ~7e3；max|Δ| = {abs_err.max().item():.3e}, '
          f'max rel = {(abs_err / exact_f64.abs()).max().item():.3e}')
    print('  （bf16 值平方在 fp32 中可精确表示；误差仅来自 128 线程 × 2 累加器的求和顺序）')

    print('\n' + '=' * 84)
    print('3) 融合价值：单独算 sqr_sum（再读一遍 A）vs 融合进 GEMM')
    print('=' * 84)
    m, n, k = 8192, 24, 28672
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float, device='cuda')
    d = torch.empty((m, n), dtype=torch.float, device='cuda')
    s = torch.empty((m,), dtype=torch.float, device='cuda')
    t_fused = time_fn(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s))
    t_reduce = time_fn(lambda: a.float().square().sum(-1))
    t_matmul = time_fn(lambda: a.float() @ b.T)
    a_bytes = a.numel() * a.element_size()
    print(f'  m={m} n={n} k={k}: A 读一遍 = {a_bytes/1e6:.1f} MB （GEMM 与 sqr_sum 都要读它）')
    print(f'  融合 kernel (D+S)  : {t_fused*1e6:7.1f} us')
    print(f'  单独 sqr_sum       : {t_reduce*1e6:7.1f} us  (再读一遍 A，融合即省下这部分)')
    print(f'  torch 真 fp32 matmul: {t_matmul*1e6:7.1f} us')
    print(f'  融合 / 单独 matmul  : {t_fused/t_matmul:.2f}x')

    print('\n' + '=' * 84)
    print('4) split-K 可加性 + 逐 split 覆盖的 K 段')
    print('=' * 84)
    g = run_case(4096, 24, 7168, 16)
    d_ns = run_case(4096, 24, 7168, None)
    bk = 64
    k_blocks = (7168 + bk - 1) // bk
    per, rem = divmod(k_blocks, 16)
    print(f'  num_splits=16, K=7168, block_k={bk}: k_blocks={k_blocks}, '
          f'每 split {per} 块(前 {rem} 个多 1 块)')
    print(f'  sum_splits(D) vs 非 split D: calc_diff={calc_diff(g["final_d"], d_ns["final_d"]):.2e}, '
          f'nerr={nerr(g["final_d"], d_ns["final_d"]):.2e}')
    print(f'  sum_splits(S) vs 非 split S: nerr={nerr(g["final_s"], d_ns["final_s"]):.2e}')
    print(f'  逐 split D 的 Frobenius 范数（16 个，应大致均匀，尾段因 remainder 略小）：')
    print('   ', [f'{g["d"][i].norm().item():.1f}' for i in range(16)])

    print('\n' + '=' * 84)
    print('5) 参数契约（sm90 host 端断言，均应在调用前被拒）')
    print('=' * 84)
    def expect_reject(desc, fn):
        try:
            fn()
            print(f'  [未拒绝!] {desc}')
        except Exception as e:
            msg = str(e).replace('\n', ' ')[:100]
            print(f'  [拒绝] {desc}: {msg}')
    m, k = 64, 4096
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b_good = torch.randn((24, k), dtype=torch.float, device='cuda')
    b_bad = torch.randn((64, k), dtype=torch.float, device='cuda')       # n=64 > 32
    k_bad = k + 8                                                        # 4104，非 64 倍数
    a_kbad = torch.randn((m, k_bad), dtype=torch.bfloat16, device='cuda')
    b_kbad = torch.randn((24, k_bad), dtype=torch.float, device='cuda')
    d = torch.empty((m, 24), dtype=torch.float, device='cuda')
    d64 = torch.empty((m, 64), dtype=torch.float, device='cuda')
    s = torch.empty((m,), dtype=torch.float, device='cuda')
    expect_reject('n=64 (>32)', lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b_bad, d64, s))
    expect_reject(f'k={k_bad} 非 64 倍数', lambda: deep_gemm.tf32_hc_prenorm_gemm(a_kbad, b_kbad, d, s))
    deep_gemm.tf32_hc_prenorm_gemm(a, b_good, d, s)
    print('  [通过] n=24, k=4096（合法）正常执行')

    print('\n' + '=' * 84)
    print('6) 性能 + roofline（N 很小、K 很大 → 访存受限，A 的读取主导流量）')
    print('=' * 84)
    print(f'{"m":>6} {"n":>4} {"k":>7} {"splits":>6} | {"us":>8} {"TFLOPS":>8} {"GB/s":>8} {"AI(F/B)":>8}')
    for (m, n, k) in [(4096, 24, 7168), (8192, 24, 28672), (137, 24, 7680), (4096, 32, 7168)]:
        for num_splits in (None, 16):
            torch.manual_seed(0)
            a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            b = torch.randn((n, k), dtype=torch.float, device='cuda')
            if num_splits is None:
                d = torch.empty((m, n), dtype=torch.float, device='cuda'); s = torch.empty((m,), dtype=torch.float, device='cuda')
            else:
                d = torch.empty((num_splits, m, n), dtype=torch.float, device='cuda'); s = torch.empty((num_splits, m), dtype=torch.float, device='cuda')
            t = time_fn(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits))
            flops = 2 * m * n * k
            bytes_ = count_bytes(a, b, d, s)
            print(f'{m:>6} {n:>4} {k:>7} {str(num_splits):>6} | {t*1e6:>8.1f} '
                  f'{flops/t/1e12:>8.1f} {bytes_/t/1e9:>8.1f} {flops/bytes_:>8.1f}')
    print('\n  roofline: H100 HBM ~3.35 TB/s；本形状 AI≈20-25 FLOP/B，远低于 TF32 拐点，')
    print('            故有效带宽越接近 HBM 峰值越优，TFLOPS 被 AI×带宽封顶。')

    print('\n完成。')


if __name__ == '__main__':
    main()
