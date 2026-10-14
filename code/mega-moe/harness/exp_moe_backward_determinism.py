#!/usr/bin/env python3
"""P170-1: MoE 反向 grouped GEMM 的可复现性边界（H100 实测）。

问题：MoE 训练要「逐位可复现」，该在哪里下手？`use_deterministic_algorithms` 够吗？

用 H100(sm90) 上真实可跑的 DeepGEMM kernel 把「谁不确定」逐层定位：

  Part A 反向 grouped GEMM 本体是**结构性 bit-确定的**：
         k_grouped(wgrad) / m_grouped(dgrad) 跑 20 次、开关 True/False、改 num_sms，逐位 0 差。
         （源码依据：scheduler/gemm.cuh 无 split-K、无跨 tile 归约；见文档 §2。）
  Part B 唯一带「契约级不确定」的 GEMM 路径是**稠密 cuBLASLt 回退**（split-K reduce）：
         kineto 抓到 det=False 走 `nvjet`+`reduce`、det=True 走自研 kernel。
  Part C wgrad / dgrad 的收缩轴不同，决定谁对「谁来排 token」敏感：
         C2 实测真实 m_grouped：组内 token 行置换 → 输出行逐位不变（dgrad 每行独立）。
         C1 用**固定累加序参考**证明 wgrad 的收缩轴是 token，置换 token 序改 fp32 步长（数学不变、bit 变）。
  Part D 把 C1 放到 MoE 语境：dispatch 的原子槽位分配（DeepEP dispatch.cuh:341）决定 token 行序，
         于是「kernel 稳定 ≠ 端到端可复现」；量化 wgrad 抖动幅度与 dgrad/逐 token 量的不变性。

输出重定向到 exp_moe_backward_determinism.out.txt。
"""
import os
import sys
import traceback

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major

DGH = None
for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
          os.path.expanduser('~/github/DeepGEMM'), os.path.expanduser('~/github/DeepGEMM')):
    if p and os.path.isdir(os.path.join(p, 'tests')):
        DGH = p
        break
sys.path.insert(0, os.path.join(DGH, 'tests'))

from generators import (  # noqa: E402
    MajorTypeAB, generate_k_grouped_contiguous, generate_m_grouped_contiguous,
)

K_ALIGN = 128
dev = 'cuda'


def reset_align():
    deep_gemm.set_mk_alignment_for_contiguous_layout(K_ALIGN)


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def bitdiff(x, y):
    return int((x != y).sum().item())


# --------------------------------------------------------------------------- data
def build_k_grouped():
    num_groups, m, n = 8, 512, 1024
    ks = [1024, 768, 640, 512, 896, 384, 768, 512]
    _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
        use_ue8m0=False, gran_k=128, quant_config=None,
        use_psum_layout=False, k_alignment=K_ALIGN, accumulate=True, out_dtype=torch.float)
    return dict(a=a, b=b, c=c, d=d, ref=ref, layout=layout, host_ks=[int(x) for x in host_ks],
                num_groups=num_groups, m=m, n=n, ks=ks)


def build_m_grouped():
    G, exp_m, n, k = 8, 128, 256, 512
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        G, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=False)
    return dict(a=a, b=b, layout=layout, d=d, ref=ref, valid=valid, G=G, n=n, k=k)


def run_k_grouped(D):
    D['d'].zero_()
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        D['a'], D['b'], D['d'], D['host_ks'], D['layout'], D['d'], recipe=(1, 1, 128))
    torch.cuda.synchronize()
    return D['d'].clone()


def run_m_grouped(D, a=None):
    aa = D['a'] if a is None else a
    D['d'].zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(aa, D['b'], D['d'], D['layout'], disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    return D['d'].clone()


def rep(fn, det, runs=20):
    deep_gemm.use_deterministic_algorithms(det)
    outs = [fn() for _ in range(runs)]
    return sum(bitdiff(outs[0], o) for o in outs[1:])


# --------------------------------------------------------------------------- A
def part_a(DK, DM):
    print('=== Part A: 反向 grouped GEMM 本体是结构性 bit-确定的 ===')
    for det in (True, False):
        print(f'  A  k_grouped(wgrad) det={det}: 20 次逐位差 = {rep(run_k_grouped, det) if False else rep(lambda: run_k_grouped(DK), det)}')
    for det in (True, False):
        print(f'  A  m_grouped(dgrad) det={det}: 20 次逐位差 = {rep(lambda: run_m_grouped(DM), det)}')
    deep_gemm.use_deterministic_algorithms(True)
    base_k = run_k_grouped(DK)
    base_m = run_m_grouped(DM)
    for nsm in (60, 80, 100, 132):
        deep_gemm.set_num_sms(nsm)
        print(f'  A  num_sms={nsm:3d}: k_grouped 逐位差={bitdiff(base_k, run_k_grouped(DK))}, '
              f'm_grouped 逐位差={bitdiff(base_m, run_m_grouped(DM))}')
    deep_gemm.set_num_sms(132)


# --------------------------------------------------------------------------- B
def part_b():
    print('\n=== Part B: 唯一带契约级不确定的 GEMM 路径是稠密 cuBLASLt 回退 ===')
    from deep_gemm.testing import bench_kineto
    m = n = 512
    k = 32768
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)

    def run(det):
        deep_gemm.use_deterministic_algorithms(det)
        deep_gemm.bf16_gemm_nt(a, b, d)
        torch.cuda.synchronize()

    deep_gemm.use_deterministic_algorithms(True)
    t_own = bench_kineto(lambda: run(True), 'gemm', suppress_kineto_output=True)
    t_nvjet, t_reduce = bench_kineto(lambda: run(False), ('nvjet', 'reduce'), suppress_kineto_output=True)
    print(f'  B  m=n=512 k=32768 bf16:')
    print(f'     det=True  (自研 kernel, 命中 "gemm"): {t_own*1e6:8.1f} us')
    print(f'     det=False (cuBLASLt 回退): nvjet={t_nvjet*1e6:8.1f} us + reduce={t_reduce*1e6:8.1f} us '
          f'(reduce>0 即 split-K 归约被发射)')
    if t_nvjet > 0 and t_own > 0:
        print(f'     自研/cuBLASLt 时间比 = {t_own/max(t_nvjet+t_reduce,1e-12):.3f}x')
    for det in (True, False):
        outs = []
        deep_gemm.use_deterministic_algorithms(det)
        for _ in range(10):
            run(det)
            outs.append(d.clone())
        print(f'     det={det}: 10 次逐位差 = {sum(bitdiff(outs[0], o) for o in outs[1:])}')
    deep_gemm.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------- C
def part_c(DK, DM):
    print('\n=== Part C: 收缩轴决定谁对 token 序敏感 ===')

    # C2: 真实 m_grouped dgrad：组内 token 行置换 → 输出行逐位不变
    base_m = run_m_grouped(DM)
    a0, asf = DM['a'][0].clone(), DM['a'][1].clone()
    layout, valid = DM['layout'], DM['valid']
    m = a0.shape[0]
    idx = torch.arange(m, device=dev)
    row_perm = idx.clone()
    i = 0
    nsg = 0
    while i < m:
        if layout[i].item() >= 0:
            j = i
            while j < m and layout[j].item() == layout[i].item():
                j += 1
            seg = idx[i:j]
            torch.manual_seed(100 + nsg)
            row_perm[i:j] = seg[torch.randperm(seg.numel(), device=dev)]
            nsg += 1
            i = j
        else:
            i += 1
    shuf_m = run_m_grouped(DM, (a0[row_perm], asf[row_perm]))
    inv = torch.empty_like(row_perm)
    inv[row_perm] = idx
    print(f'  C2 真实 m_grouped(dgrad)：组内 token 行置换后按行对齐，逐位差='
          f'{bitdiff(base_m, shuf_m[inv])}/{base_m.numel()}  ← dgrad 每行独立')

    # C1: wgrad 的收缩轴是 token —— 固定累加序参考（数学不变、fp32 步长变）
    T, mm, nn = 2048, 512, 512
    torch.manual_seed(0)
    S = torch.randn(T, mm, device=dev, dtype=torch.bfloat16).float()
    B = torch.randn(T, nn, device=dev, dtype=torch.bfloat16).float()

    def wgrad_blocks(x, y, blk=128):
        D = torch.zeros(mm, nn, device=dev, dtype=torch.float32)
        for s in range(0, T, blk):
            D = D + x[s:s + blk].t() @ y[s:s + blk]
        return D

    base = wgrad_blocks(S, B)
    # 参考实现本身逐位稳定？
    print(f'  C1 固定累加序参考自身 10 次逐位差 = '
          f'{sum(bitdiff(base, wgrad_blocks(S, B)) for _ in range(10))}')
    diffs = []
    for seed in range(5):
        torch.manual_seed(1000 + seed)
        p = torch.randperm(T, device=dev)
        o = wgrad_blocks(S[p], B[p])
        d = bitdiff(base, o)
        diffs.append(d)
        print(f'  C1 wgrad token 序样本 {seed}: 逐位差={d}/{base.numel()}, '
              f'relerr={relerr(o, base):.3e}  ← 数学不变、bit 变')
    print(f'  C1 5 个 token 序: 逐位差 min={min(diffs)} max={max(diffs)}；'
          f'relerr 量级 ~1e-7（训练可接受，但逐位不可复现）')
    return S, B


# --------------------------------------------------------------------------- D
def part_d(S, B):
    print('\n=== Part D: MoE 层级 —— dispatch 行序决定 wgrad 能否逐位复现 ===')
    T, mm, nn = S.shape[0], S.shape[1], B.shape[1]

    def wgrad_blocks(x, y, blk=128):
        D = torch.zeros(mm, nn, device=dev, dtype=torch.float32)
        for s in range(0, T, blk):
            D = D + x[s:s + blk].t() @ y[s:s + blk]
        return D

    def per_token_stat(x):
        return x.float().norm(dim=1)  # 逐 token 量：与 token 序无关

    base = wgrad_blocks(S, B)
    maxrel, maxbit = 0.0, 0
    for seed in range(5):
        p = torch.randperm(T, device=dev)
        o = wgrad_blocks(S[p], B[p])
        maxrel = max(maxrel, relerr(o, base))
        maxbit = max(maxbit, bitdiff(base, o))
    print(f'  D  dispatch 行序随机（5 样本）：wgrad 逐位差 max={maxbit}/{base.numel()}, '
          f'relerr max={maxrel:.3e}')
    # 逐 token 量不变
    ok = True
    p = torch.randperm(T, device=dev)
    s0 = per_token_stat(S)
    s1 = per_token_stat(S[p])[torch.argsort(p)]  # 还原顺序
    ok = bitdiff(s0, s1) == 0
    print(f'  D  对照：逐 token 量（dgrad 类，行独立）在乱序下逐位不变 = {ok}')
    print('  结论：kernel 稳定 ≠ 端到端可复现；wgrad 的 token 累加序由 dispatch 决定'
          '（DeepEP dispatch.cuh:341 的 atomicAdd 槽位分配）。')


def main():
    print(f'get_arch_major() = {get_arch_major()}')
    reset_align()
    torch.manual_seed(0)
    DK = build_k_grouped()
    DM = build_m_grouped()
    print(f'k_grouped: G={DK["num_groups"]} m={DK["m"]} n={DK["n"]} host_ks={DK["host_ks"]} '
          f'sum_k={sum(DK["host_ks"])}')
    print(f'm_grouped: G={DM["G"]} n={DM["n"]} k={DM["k"]} m={DM["a"][0].shape[0]}')
    part_a(DK, DM)
    part_b()
    S, B = part_c(DK, DM)
    part_d(S, B)


if __name__ == '__main__':
    try:
        main()
        print('\nDONE')
    except Exception:
        traceback.print_exc()
        print('FAILED')
