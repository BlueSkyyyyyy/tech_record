"""P179-1: Mega MoE 的 BLOCK_M 选择到底买到什么 —— 用 H100 真机 GEMM 量「tile 收益」。

背景：Mega MoE 主 kernel 的 `BLOCK_M`（沿每个专家 token 数 M 的 tile）由 host 启发式
`get_block_config_for_mega_moe`（`csrc/jit_kernels/heuristics/mega_moe.hpp:88-127`）从候选集
`kCandidateBlockM = {8,16,32,64,96,128,192,240}`（`layout/mega_moe.cuh:13-16`）里挑，规则的核心是
`:97` 的注释：

    Every M block costs roughly 128 extra rows (its tasks reload the weight tiles),
    so use the fewest blocks per expert and the smallest tile covering them:
    one block up to 192 rows, then blocks of up to 240 rows.

第 178 篇只量了这个启发式的**输出**（topk 变时 block_m 怎么跳），没量 tile 本身的时延。主 kernel 是
SM100 专属、H100 跑不了，但「一个 M block 的真实代价」可以用 H100 上**真实 SM90 FP8 grouped GEMM**当代理
量出来：`m_grouped_fp8_fp4_gemm_nt_masked`（`assert` 与 L1/L2 同形：A=[tokens,H]，B=W 权重），且它的
`expected_m` 形参可以**强制 block_m**（`sm90.hpp:34-35` MGroupedMasked 候选 {64,128}）。

三个真机测量（H100 sm90，DeepGEMM 2.8.1，L1 形状 K=H=7168,N=2I=4096 / L2 形状 K=I=2048,N=H=7168）：

  Part B · tile 代价曲线（G=1）：固定一个专家，扫 M=1..256，强制 block_m∈{64,128}。看
           「块内行几乎免费、跨块才跳」的阶梯 —— 直接验证 `:97` 的「块=权重重载」模型。
  Part C · block_m 的收益：同一 M=128，bm64（2 块）vs bm128（1 块）的真实时延比。
  Part D · 多专家税：固定总行数=512，把行摊给 G∈{1,2,4,8,16,32} 个专家（每专家 = 一份独立权重），
           看 per-expert（=per-weight-stream）固定开销如何吃掉效率 —— `:99`「many local experts」的量化。

约束：Mega MoE 主 kernel（UMMA/UTCCP/NVLink 直写、240 行 tile）SM100 专属、**仅分析**；本机所有数字
来自 SM90 FP8 `m_grouped_fp8_fp4_gemm_nt_masked` 代理（block_m 只能到 64/128）。8 卡节点共享，
报数已多次重复取 min，仍可能与独占基线有差异。输出重定向到 exp_block_m_tile.out.txt。
"""
import math
import os
import sys
import time
import traceback

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              '~/github/DeepGEMM'):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


sys.path.insert(0, os.path.join(_find_deepgemm_dir(), 'tests'))
from deep_gemm.utils import align, per_token_cast_to_fp8, per_block_cast_to_fp8  # noqa: E402

GRAN_K = 128


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def _bench(fn, iters=60, warmup=15, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


# ---------------------------------------------------------------- Part A: 启发式
def py_block_m(num_ranks, num_experts, num_tokens, num_topk):
    """逐行重写 csrc/jit_kernels/heuristics/mega_moe.hpp:88-127 的 block_m 选择。"""
    num_expected_tokens = float(num_tokens) * num_ranks * num_topk / num_experts
    num_covered_tokens = num_expected_tokens + math.sqrt(num_expected_tokens)
    block_m = 16 if num_expected_tokens <= 10 else 32
    if num_expected_tokens > 24:
        num_blocks = int(math.ceil(num_covered_tokens / 240))
        if num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14:
            num_blocks = 2
        for candidate in (64, 128, 192, 240):
            block_m = candidate
            if num_blocks * candidate >= num_covered_tokens:
                break
    return block_m, num_expected_tokens, num_covered_tokens


def part_a():
    print('=' * 100)
    print('Part A · BLOCK_M 启发式（真实 pybind 绑定 vs Python 重写）')
    print('=' * 100)
    print('候选集 kCandidateBlockM = {8,16,32,64,96,128,192,240}  (layout/mega_moe.cuh:13-16)')
    print(f'{"R":>2} {"E":>4} {"T":>6} {"topk":>4} {"exp_tok":>8} {"covered":>8} '
          f'{"py_bm":>5} {"C++ bm":>6} {"ok":>3}')
    R = 8
    mismatch = 0
    for (E, T) in [(64, 8192), (256, 8192), (256, 1024)]:
        for topk in (2, 4, 8):
            bm, exp_tok, cov = py_block_m(R, E, T, topk)
            c_bm = deep_gemm.get_block_m_for_mega_moe(R, E, T, T, topk, 'fp8xfp4')
            ok = bm == c_bm
            mismatch += (not ok)
            print(f'{R:>2} {E:>4} {T:>6} {topk:>4} {exp_tok:>8.1f} {cov:>8.1f} '
                  f'{bm:>5} {c_bm:>6} {"ok" if ok else "MISMATCH":>3}')
    print(f'mismatch={mismatch}')
    print(':97 的代价模型：「每个 M block ≈ 128 行的额外代价（它要重载权重 tile）」')


# ---------------------------------------------------------------- proxy GEMM
def make_ops(G, K, N, max_m, device='cuda'):
    torch.manual_seed(0)
    W = torch.randn(G, N, K, device=device)
    b_f8 = torch.empty(G, N, K, dtype=torch.float8_e4m3fn, device=device)
    b_sf = torch.empty((G, align(N, GRAN_K) // GRAN_K, align(K, GRAN_K) // GRAN_K),
                       device=device, dtype=torch.float32)
    for e in range(G):
        b_f8[e], b_sf[e] = per_block_cast_to_fp8(W[e], use_ue8m0=False, gran_k=GRAN_K)
    A = torch.randn(G * max_m, K, device=device)
    a_f8, a_sf = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN_K)
    a_f8 = a_f8.reshape(G, max_m, -1)
    a_sf = a_sf.reshape(G, max_m, -1)
    d = torch.empty(G, max_m, N, dtype=torch.bfloat16, device=device)
    return (a_f8, a_sf), (b_f8, b_sf), W, A, d


def run(G, K, N, max_m, M, expected_m, ops):
    (a_f8, a_sf), (b_f8, b_sf), _W, _A, d = ops
    mm = torch.full((G,), M, device='cuda', dtype=torch.int32)

    def fn():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((a_f8, a_sf), (b_f8, b_sf), d, mm,
                                                   expected_m, disable_ue8m0_cast=True)
    return _bench(fn)


def part_b(tag, K, N):
    print()
    print('=' * 100)
    print(f'Part B · tile 代价曲线（G=1，{tag}，K={K} N={N}，权重 {N*K/1e6:.1f}MB/专家）')
    print('=' * 100)
    max_m = 4096
    ops = make_ops(1, K, N, max_m)
    print(f'{"forced bm":>9} {"M":>5} {"#blocks":>7} {"t(us)":>8} {"Δprev(us)":>9} {"TFLOPS":>7} {"rel_row":>8}')
    for expM, bm in [(16, 64), (128, 128)]:
        prev = None
        for M in [1, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256]:
            t = run(1, K, N, max_m, M, expM, ops)
            nb = math.ceil(M / bm)
            fl = 2 * M * N * K
            d = '' if prev is None else f'{(t - prev) * 1e6:9.1f}'
            print(f'{bm:>9} {M:>5} {nb:>7} {t*1e6:>8.1f} {d:>9} {fl/t/1e12:>7.0f} '
                  f'{t*1e9/M:>8.1f}')
            prev = t
    print('读法：同一 block 内（#blocks 不变）时延**几乎不变**——块内行被权重流掩盖；只有 #blocks 跳 1')
    print('      才跳一次 ≈ 权重流代价。这正是 :97「块=权重重载」的真机形态。')


def part_c():
    print()
    print('=' * 100)
    print('Part C · block_m 的收益：同一份工作，tile 越大块越少')
    print('=' * 100)
    K, N, max_m = 7168, 4096, 4096
    ops = make_ops(1, K, N, max_m)
    print(f'{"M":>5} {"bm=64 (expM=16)":>16} {"bm=128 (expM=128)":>18} {"speedup":>8}')
    for M in [96, 128, 160, 192, 256]:
        t64 = run(1, K, N, max_m, M, 16, ops)
        t128 = run(1, K, N, max_m, M, 128, ops)
        print(f'{M:>5} {t64*1e6:>13.1f}us {t128*1e6:>15.1f}us {t64/t128:>7.2f}x')


def part_d():
    print()
    print('=' * 100)
    print('Part D · 多专家税：固定总行数=512，摊给 G 个专家（每专家一份独立权重）')
    print('=' * 100)
    K, N = 7168, 4096
    print(f'{"G":>3} {"M/专家":>7} {"总行":>5} {"t(us)":>8} {"TFLOPS":>7} {"rows/专家成本":>12}')
    base = None
    for G, M in [(1, 512), (2, 256), (4, 128), (8, 64), (16, 32), (32, 16)]:
        ops = make_ops(G, K, N, M)
        t = run(G, K, N, M, M, 128, ops)
        tot = G * M
        if base is None:
            base = t
        print(f'{G:>3} {M:>7} {tot:>5} {t*1e6:>8.1f} {2*tot*N*K/t/1e12:>7.0f} {t/base:>11.2f}x')
    print('读法：G 越大 → 每个专家块越小、权重流次数越多，per-expert 固定开销吃满 → 效率单调塌。')
    print('      这是 :99「a single 240-row block pays off with many local experts」的量化背景。')


def correctness():
    print()
    print('=' * 100)
    print('正确性抽查 · 代理 kernel vs 反量化 fp32 参考（G=1, L1, M=128, block_m=128）')
    print('=' * 100)
    K, N, max_m = 7168, 4096, 4096
    ops = make_ops(1, K, N, max_m)
    (a_f8, a_sf), (b_f8, b_sf), W, A, d = ops
    d.zero_()
    mm = torch.full((1,), 128, device='cuda', dtype=torch.int32)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((a_f8, a_sf), (b_f8, b_sf), d, mm, 128,
                                               disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    a_deq = (a_f8[0, :128].float().reshape(128, -1) *
             a_sf[0, :128].float().repeat_interleave(GRAN_K, dim=1)[:, :K])
    b_deq = (b_f8[0].float() * b_sf[0].float().repeat_interleave(GRAN_K, 1)
             .repeat_interleave(GRAN_K, 0)[:N, :K])
    ref = a_deq @ b_deq.T
    print(f'kernel vs dequantized fp32 relerr = {rel(d[0, :128].float(), ref):.3e}')


def main():
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}')
    for fn in (part_a, part_b, part_c, part_d, correctness):
        try:
            if fn is part_b:
                fn('L1', 7168, 4096)
                fn('L2', 2048, 7168)
            else:
                fn()
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main()
