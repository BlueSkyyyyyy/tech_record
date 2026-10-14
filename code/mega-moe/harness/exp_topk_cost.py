"""P178-1: MoE 的 top-k 到底买到了什么 —— 前向/反向成本与 grouped GEMM 效率的 top-k 扫描。

背景：DeepSeek-V3 用 topk=8（+1 共享）。所有「通信量模型」（第 01 篇）和「专家容量」
（`num_topk`）都随 k 线性放大，直觉上 topk 越贵越线性。但 topk 还悄悄改了另一件事：
Mega MoE 的 **BLOCK_M 启发式**以「每专家期望 token 数 = T·R·topk/E」（`heuristics/mega_moe.hpp:94`）
为输入，topk 一变，kernel 自己挑的 tile 也变（16/32/64/128/192/240）。而 wgrad 的收缩维 K
正是「每专家 token 数」。于是 topk 的边际代价不是简单线性。

本实验在三层各取真机/真绑定数字（H100 sm90，DeepGEMM 2.8.1）：

  Part A · host 启发式（架构无关，真实 pybind 绑定）
    - `deep_gemm.get_block_m_for_mega_moe(...)` 随 topk 的**阶梯**；
    - `_C.get_symm_buffer_size_for_mega_moe(...)` 的对称缓冲字节数随 topk 的**线性**；
    - 用 Python 逐行重写 `heuristics/mega_moe.hpp:88-127` 对拍（0 不符）。

  Part B · 真实 SM90 FP8 grouped GEMM（本机可跑）
    - 固定 T/H/I/E，扫 topk：dgrad（m_grouped，M=T·topk）与 wgrad（k_grouped，ΣK=T·topk）；
    - 报时间/TFLOPS，并给「每个 topk 的边际代价」与 wgrad 效率随 k 的提升。

  Part C · 通信量模型（+可选 NCCL all-to-all 真机）
    - dispatch ~ T·topk·H·1B、combine ~ T·topk·H·2B；用第 01/166 篇实测带宽折算。

约束：Mega MoE 主 kernel（tcgen05/UMMA/UTCCP/NVLink 直写）SM100 专属、**仅分析**；
本机所有 kernel 数字来自 SM90 FP8 `m_grouped`/`k_grouped`。输出重定向到 exp_topk_cost.out.txt。
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
from deep_gemm.utils import (  # noqa: E402
    align, per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    get_mk_alignment_for_contiguous_layout,
)

GRAN_K = 128


# --------------------------------------------------------------------------- helpers
def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def _bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ------------------------------ Part A: host heuristic (reimplementation of the C++ rule)
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
    return block_m, num_expected_tokens


def part_a():
    print('=' * 100)
    print('Part A · host 启发式（真实 pybind 绑定 vs Python 重写，架构无关）')
    print('=' * 100)
    R = 8
    H = 7168
    I = 2048
    mismatch = 0
    # 三种规模，前两种 block_m 全 240，第三种（每专家期望 token 少）能走完 64→128→192→240 阶梯
    for (E, T, tmax) in [(64, 8192, 8192), (256, 8192, 8192), (256, 1024, 1024)]:
        print(f'\n-- E={E} T={T} num_max_tokens_per_rank={tmax} num_ranks={R} mma_type=fp8xfp4 --')
        print(f'{"topk":>4} {"exp_tok/expert":>14} {"py_block_m":>11} {"C++ block_m":>12} {"ok":>3} '
              f'{"num_bytes(MB)":>13} {"Δbytes(MB)":>11}')
        prev_bytes = None
        for topk in [1, 2, 3, 4, 6, 8, 12, 16]:
            py_bm, exp_tok = py_block_m(R, E, T, topk)
            c_bm = deep_gemm.get_block_m_for_mega_moe(R, E, tmax, T, topk, 'fp8xfp4')
            num_bytes, _slice = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
                R, E, tmax, topk, H, I, 'fp8xfp4', 'swiglu', 0)
            delta = '' if prev_bytes is None else f'{(num_bytes - prev_bytes) / 1e6:9.2f}'
            prev_bytes = num_bytes
            ok = 'ok' if py_bm == c_bm else 'MISMATCH'
            mismatch += (py_bm != c_bm)
            print(f'{topk:>4} {exp_tok:>14.1f} {py_bm:>11} {c_bm:>12} {ok:>3} '
                  f'{num_bytes / 1e6:>13.2f} {delta:>11}')
    print(f'\nPython 重写 vs C++ 绑定：mismatch={mismatch}（逐行重写 heuristics/mega_moe.hpp:88-127）')
    print('\n读法：① `py_block_m` 随 topk 走**阶梯**（每专家期望 token 数 ∝ topk 跨过 24/192 等阈值）；')
    print('      ② `num_bytes` 随 topk **先近似线性、后饱和**——ring 容量由 live pool block 数封顶。')


# ------------------------------ Part B: real grouped GEMM sweep
def quant_k_operand(X, host_ks):
    total_k, mn = X.shape
    data = torch.zeros(total_k * mn, dtype=torch.float8_e4m3fn, device=X.device)
    deq = torch.empty_like(X)
    sfs = []
    off = 0
    for hk in host_ks:
        xg = X[off:off + hk].contiguous()
        xf8, sf = per_channel_cast_to_fp8(xg, use_ue8m0=False, gran_k=GRAN_K)
        data[off * mn:(off + hk) * mn] = xf8.T.reshape(-1)
        deq[off:off + hk] = xf8.float() * sf.repeat_interleave(GRAN_K, dim=0)
        sfs.append(sf)
        off += hk
    sf_all = torch.cat(sfs).T if sfs else X.new_zeros((mn, 0))
    return data, sf_all, deq


def run_k_grouped(counts, M, N, device, want_deq=False):
    host_ks = [align(k, GRAN_K) for k in counts]
    total_k = sum(host_ks)
    A = torch.randn(total_k, M, device=device, dtype=torch.float32)
    B = torch.randn(total_k, N, device=device, dtype=torch.float32)
    a = quant_k_operand(A, host_ks)
    b = quant_k_operand(B, host_ks)
    layout = torch.tensor(host_ks, device=device, dtype=torch.int32)
    d = torch.zeros(len(counts), M, N, device=device, dtype=torch.float32)

    def fn():
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (a[0], a[1]), (b[0], b[1]), d, host_ks, layout, d, recipe=(1, 1, GRAN_K))

    t = _bench(fn)
    deq, d_single = None, None
    if want_deq:
        # 单次 fresh 运行（不能用上面累积了 iters 次的 d），再与反量化参考比较
        d_single = torch.zeros(len(counts), M, N, device=device, dtype=torch.float32)
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (a[0], a[1]), (b[0], b[1]), d_single, host_ks, layout, d_single,
            recipe=(1, 1, GRAN_K))
        torch.cuda.synchronize()
        deq = torch.empty_like(d_single)
        off = 0
        for e, k in enumerate(counts):
            hk = host_ks[e]
            deq[e] = a[2][off:off + hk].T @ b[2][off:off + hk]
            off += hk
    fl = 2 * M * N * total_k  # 用 host_ks（对齐后）——即 kernel 实际计算的 K
    return t, fl, deq, d_single


def run_m_grouped(counts, N, K, E, device):
    ma = get_mk_alignment_for_contiguous_layout()
    total_m = sum(align(c, ma) for c in counts)
    A = torch.randn(total_m, K, device=device, dtype=torch.float32)
    layout = torch.full((total_m,), -1, device=device, dtype=torch.int32)
    off = 0
    for e, c in enumerate(counts):
        layout[off:off + c] = e
        off += align(c, ma)
    a_f8, a_sf = per_token_cast_to_fp8(A, use_ue8m0=False, gran_k=GRAN_K)
    B = torch.randn(E, N, K, device=device, dtype=torch.float32)
    b_f8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=device)
    b_sf = torch.empty((E, align(N, GRAN_K) // GRAN_K, align(K, GRAN_K) // GRAN_K),
                       device=device, dtype=torch.float32)
    for e in range(E):
        b_f8[e], b_sf[e] = per_block_cast_to_fp8(B[e], use_ue8m0=False, gran_k=GRAN_K)
    d = torch.empty(total_m, N, device=device, dtype=torch.bfloat16)

    def fn():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a_f8, a_sf), (b_f8, b_sf), d, layout,
                                                   disable_ue8m0_cast=True)

    t = _bench(fn)
    fl = 2 * total_m * N * K  # 用对齐后的 m_total——kernel 实际计算的行数（含段尾 padding）
    return t, fl, total_m


def part_b(T=4096, H=7168, I=2048, E=64, topks=(1, 2, 3, 4, 6, 8)):
    print()
    print('=' * 100)
    print(f'Part B · 真实 SM90 FP8 grouped GEMM 随 topk（T={T} H={H} I={I} E={E}）')
    print('=' * 100)
    device = 'cuda'
    print(f'{"topk":>4} {"tok/expert":>10} | {"dgrad m(aligned)":>15} {"pad":>5} {"t(us)":>7} {"TFLOPS":>7} '
          f'| {"wgrad sumK":>11} {"pad":>5} {"t(us)":>7} {"TFLOPS":>7}')
    rows = []
    for topk in topks:
        counts = [(T * topk) // E] * E
        # dgrad: dA_e = dY_e @ W2_e^T  (M=token, N=I, K=H)
        td, fld, m_total = run_m_grouped(counts, I, H, E, device)
        # wgrad: dW2[e] = A_e^T dY_e  (M=I, N=H, K=token)
        tw, flw, _, _ = run_k_grouped(counts, I, H, device)
        exact = sum(counts)
        sumk = sum(align(c, 128) for c in counts)
        rows.append((topk, td, fld, tw, flw, exact, m_total, sumk))
        print(f'{topk:>4} {T * topk // E:>10} | {m_total:>15} {m_total / exact:>5.2f} '
              f'{td * 1e6:>7.0f} {fld / td / 1e12:>7.0f} | {sumk:>11} {sumk / exact:>5.2f} '
              f'{tw * 1e6:>7.0f} {flw / tw / 1e12:>7.0f}')
    print('\n-- 归一化：把 topk=1 的时间/FLOPs 当基准 --')
    print(f'{"topk":>4} {"dgrad t/t0":>11} {"dgrad FLOPs/t0":>15} {"dgrad TFLOPS":>13} '
          f'{"wgrad t/t0":>11} {"wgrad FLOPs/t0":>15} {"wgrad TFLOPS":>13}')
    t0d, f0d = rows[0][1], rows[0][2]
    t0w, f0w = rows[0][3], rows[0][4]
    for topk, td, fld, tw, flw, *_ in rows:
        print(f'{topk:>4} {td / t0d:>11.2f} {fld / f0d:>15.2f} {fld / td / 1e12:>13.0f} '
              f'{tw / t0w:>11.2f} {flw / f0w:>15.2f} {flw / tw / 1e12:>13.0f}')
    print('\n读法：topk 增大 → 每专家 token 数跨越 M/K 对齐粒度，dgrad 的段尾 padding（pad 列）')
    print('      收缩、wgrad 的 K padding 收缩 ⇒ **两者 TFLOPS 都随 topk 上升**；但 FLOPs 仍线性涨。')
    # correctness spot-check (topk=4)
    print('\n-- 正确性抽查（topk=4）：wgrad kernel（单次 fresh）vs 反量化参考 --')
    counts = [(T * 4) // E] * E
    tw, flw, deq, d = run_k_grouped(counts, I, H, 'cuda', want_deq=True)
    print(f'   dW2 kernel vs deq  relerr={rel(d, deq):.3e}  (sum_k={sum(align(c, 128) for c in counts)})')


# ------------------------------ Part C: communication model
def part_c(T=4096, H=7168):
    print()
    print('=' * 100)
    print('Part C · 通信量模型（dispatch/combine 随 topk 线性放大；带宽取第 01 篇实测）')
    print('=' * 100)
    BW = 393e9  # GB/s，第 01 篇 8×H100 NCCL a2a 渐近有效带宽
    print(f'{"topk":>4} {"dispatch MB":>12} {"combine MB":>12} {"total MB":>10} '
          f'{"~time(us) @393GB/s":>18}')
    for topk in [1, 2, 3, 4, 6, 8]:
        disp = T * topk * H * 1 / 1e6      # fp8 1B/元素
        comb = T * topk * H * 2 / 1e6      # bf16 2B/元素
        total = disp + comb
        print(f'{topk:>4} {disp:>12.1f} {comb:>12.1f} {total:>10.1f} '
              f'{total * 1e6 / BW * 1e6:>18.1f}')
    print('  （对比：Part B 的 GEMM 时间随 topk 也线性，但 wgrad 效率随之提升——两股力量相抵。）')


def main():
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}')
    try:
        part_a()
    except Exception:
        traceback.print_exc()
    try:
        part_b(E=64)
        part_b(E=8)
    except Exception:
        traceback.print_exc()
    try:
        part_c()
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    main()
