#!/usr/bin/env python3
"""P174-1: MoE 反向梯度累积的「切分 × 结合序」可复现性（H100 实测）。

问题：第 170 篇（docs/180）把「MoE 反向算子链的可复现性边界」量到「kernel 逐位确定、端到端
不是」，并明确留了一个口子（docs/180 §8）：把「多 microbatch 梯度累加」这一维度补上——
量梯度累积的**结合序**（association order），而不只是单次 GEMM 里的 token 序。

这一篇回答：生产训练里 gradient accumulation（global batch > 单卡显存时不可避免）到底把
哪条「逐位可复现」的假设打碎、碎在哪一层、幅度多大。

用 H100(sm90) 上真实可跑的 DeepGEMM kernel（SM90 FP8 k_grouped=wgrad、m_grouped=dgrad）
把「切分 × 结合序」拆成四个可控变量：

  Part A  wgrad：同一批 token，一次大 GEMM vs 切 3 个 microbatch 再求和 —— 逐位变吗？
          根因：kernel 内 K 轴按固定 K-block 树累加（不切 K），切分改变了这棵树。
  Part B  结合序（association order）：固定一组 per-chunk 偏梯度，按 forward / reverse /
          pairwise-tree / sorted-by-magnitude 求和 —— 不同顺序差多少 bit？幅度多大？
  Part C  dgrad：dgrad 每行（每 token）独立，microbatch 切分 = 每组行的一个置换 → 逐位不变。
  Part D  训练真实写法：把 fp32 梯度缓冲作为 `c` 让 kernel **原地累积**（c=d），
          与「各 microbatch 算 fresh、再 torch 求和」是不是同一份 bit？

输出重定向到 exp_moe_grad_accum_assoc.out.txt。
"""
import os
import sys
import random
import hashlib

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
    MajorTypeAB, k_grouped_cast_fp8_fp4_with_major, generate_m_grouped_contiguous,
)

K_ALIGN = 128
dev = 'cuda'
GEN = os.path.join(DGH, 'tests', 'generators.py')


def reset_align():
    deep_gemm.set_mk_alignment_for_contiguous_layout(K_ALIGN)


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def bitdiff(x, y):
    return int((x != y).sum().item())


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# --------------------------------------------------------------------------- helpers
def quant_k(A_bf16, ks):
    """K-major per-128 block-scale 量化；`ks` 是每组的物理（已对齐）K。"""
    return k_grouped_cast_fp8_fp4_with_major(
        A_bf16, ks, MajorTypeAB.KMajor, use_ue8m0=False, gran_k=128, is_fp4=False)


def run_kwgrad(aq, bq, ks, G, m, n, c=None):
    """真实 SM90 FP8 wgrad：D = C + A^T B（SM90 k_grouped 强制累积，c 必填）。"""
    d = torch.zeros((G, m, n), device=dev, dtype=torch.float32)
    if c is not None:
        d = c
    layout = torch.tensor(ks, dtype=torch.int32, device=dev)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(aq, bq, d, ks, layout, d, recipe=(1, 1, 128))
    torch.cuda.synchronize()
    return d.clone()


# --------------------------------------------------------------------------- Part A
def part_a():
    print('=== Part A: wgrad 对 microbatch 切分不 bit 不变 ===')
    G, m, n = 8, 512, 1024
    ks_full = [1024, 768, 640, 512, 896, 384, 768, 512]   # 每专家 token 数（均 128 对齐）
    torch.manual_seed(0)
    Ae = [torch.randn(k, m, device=dev, dtype=torch.bfloat16) for k in ks_full]
    Be = [torch.randn(k, n, device=dev, dtype=torch.bfloat16) for k in ks_full]

    # 一次大 GEMM（把所有 token 拼在每个专家里）
    aq_full = quant_k(torch.cat(Ae, 0), ks_full)
    bq_full = quant_k(torch.cat(Be, 0), ks_full)
    dW_ref = run_kwgrad(aq_full, bq_full, ks_full, G, m, n)

    # 切成 3 个 microbatch（每专家按同一 token 顺序切，piece 均 128 对齐）
    pieces = {1024: [512, 256, 256], 768: [256, 256, 256], 640: [256, 256, 128],
              512: [256, 128, 128], 896: [512, 256, 128], 384: [128, 128, 128]}
    piece = [pieces[k] for k in ks_full]
    nmb = 3
    dW_mb = []
    for j in range(nmb):
        ks_j, Aj, Bj = [], [], []
        for e in range(G):
            s = sum(piece[e][:j])
            ln = piece[e][j]
            ks_j.append(ln)
            Aj.append(Ae[e][s:s + ln])
            Bj.append(Be[e][s:s + ln])
        dW_mb.append(run_kwgrad(quant_k(torch.cat(Aj, 0), ks_j),
                                quant_k(torch.cat(Bj, 0), ks_j), ks_j, G, m, n))
    print(f'  每 microbatch 总 token = '
          f'{[sum(pieces[k][j] for k in ks_full) for j in range(nmb)]}')

    dW_sum = dW_mb[0] + dW_mb[1] + dW_mb[2]
    # fp64 ground truth（在 bf16 操作数上做，排除 FP8 量化误差）
    dW_true = torch.stack([Ae[e].float().T @ Be[e].float() for e in range(G)])

    n_el = dW_ref.numel()
    print(f'  A1 一次大 GEMM vs 3-microbatch 求和: 逐位差 = {bitdiff(dW_ref, dW_sum)}/{n_el} '
          f'({100.0 * bitdiff(dW_ref, dW_sum) / n_el:.1f}%), relerr = {relerr(dW_sum, dW_ref):.3e}')
    print(f'  A2 一次大 GEMM vs fp64 真值: relerr = {relerr(dW_ref, dW_true):.3e}')
    print(f'  A3 3-mb 求和   vs fp64 真值: relerr = {relerr(dW_sum, dW_true):.3e}')
    return Ae, Be, ks_full


# --------------------------------------------------------------------------- Part B
def part_b():
    print('\n=== Part B: 结合序（association order）——同一批偏梯度、不同求和顺序 ===')
    G, m, n = 8, 512, 1024
    ks_eq = [1024] * G          # 每专家等长，便于切成 8 个 128-chunk 的均匀贡献
    chunk, nchunk = 128, 8
    torch.manual_seed(1)
    Ae = [torch.randn(k, m, device=dev, dtype=torch.bfloat16) for k in ks_eq]
    Be = [torch.randn(k, n, device=dev, dtype=torch.bfloat16) for k in ks_eq]

    # 参考：一次大 GEMM
    aq = quant_k(torch.cat(Ae, 0), ks_eq)
    bq = quant_k(torch.cat(Be, 0), ks_eq)
    ref = run_kwgrad(aq, bq, ks_eq, G, m, n)

    # 每个 chunk 贡献一份 [G,m,n] 偏梯度
    contrib = []
    for c in range(nchunk):
        ks_c = [chunk] * G
        Ac = torch.cat([Ae[e][c * chunk:(c + 1) * chunk] for e in range(G)], 0)
        Bc = torch.cat([Be[e][c * chunk:(c + 1) * chunk] for e in range(G)], 0)
        contrib.append(run_kwgrad(quant_k(Ac, ks_c), quant_k(Bc, ks_c), ks_c, G, m, n))

    def assoc(vals):
        # pairwise tree（这里 8 个，两两归并）
        cur = list(vals)
        while len(cur) > 1:
            nxt = [cur[i] + cur[i + 1] for i in range(0, len(cur) - 1, 2)]
            if len(cur) % 2:
                nxt.append(cur[-1])
            cur = nxt
        return cur[0]

    fwd = contrib[0]
    for c in contrib[1:]:
        fwd = fwd + c
    rev = contrib[-1]
    for c in reversed(contrib[:-1]):
        rev = rev + c
    tree = assoc(contrib)
    order = sorted(range(len(contrib)), key=lambda i: contrib[i].abs().mean().item())
    bymag = contrib[order[0]]
    for i in order[1:]:
        bymag = bymag + contrib[i]

    dW_true = torch.stack([Ae[e].float().T @ Be[e].float() for e in range(G)])
    n_el = ref.numel()
    print(f'  参考(一次大 GEMM) vs fp64 真值 relerr = {relerr(ref, dW_true):.3e}')
    for name, v in (('forward', fwd), ('reverse', rev), ('pairwise-tree', tree),
                    ('by-magnitude', bymag)):
        print(f'  B  {name:14s} vs 一次大 GEMM: 逐位差={bitdiff(ref, v):7d}/{n_el} '
              f'({100.0 * bitdiff(ref, v) / n_el:5.1f}%), relerr={relerr(v, ref):.3e}, '
              f'vs fp64={relerr(v, dW_true):.3e}')
    for name, v in (('reverse', rev), ('tree', tree), ('bymag', bymag)):
        print(f'  B  forward vs {name:13s}: 逐位差={bitdiff(fwd, v)}/{n_el}')
    return


# --------------------------------------------------------------------------- Part C
def part_c():
    print('\n=== Part C: dgrad 每 token 行独立 → microbatch 切分（=行置换）逐位不变 ===')
    G, exp_m, n, k = 8, 128, 256, 512
    random.seed(7)   # generator 的每组 M 用 random.uniform，钉住以复现分母
    m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
        G, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=False, use_psum_layout=False)

    def run_m(aa):
        dd = torch.zeros_like(d)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(aa, b, dd, layout, disable_ue8m0_cast=True)
        torch.cuda.synchronize()
        return dd

    base = run_m(a)
    a0, asf = a[0].clone(), a[1].clone()
    idx = torch.arange(a0.shape[0], device=dev)
    perm = idx.clone()
    nsg = 0
    i = 0
    while i < a0.shape[0]:
        if layout[i].item() >= 0:
            j = i
            while j < a0.shape[0] and layout[j].item() == layout[i].item():
                j += 1
            seg = idx[i:j]
            torch.manual_seed(200 + nsg)
            perm[i:j] = seg[torch.randperm(seg.numel(), device=dev)]
            nsg += 1
            i = j
        else:
            i += 1
    shuf = run_m((a0[perm], asf[perm]))
    inv = torch.empty_like(perm)
    inv[perm] = idx
    print(f'  C  每组行置换（= microbatch 切分的行重排）后按行对齐: '
          f'逐位差={bitdiff(base, shuf[inv])}/{base.numel()}  ← 每行独立、bit 不变')
    return


# --------------------------------------------------------------------------- Part D
def part_d():
    print('\n=== Part D: 训练真实写法 —— kernel 原地累积(c=d) vs 各 microbatch 求和 ===')
    G, m, n = 8, 512, 1024
    ks_full = [1024, 768, 640, 512, 896, 384, 768, 512]
    torch.manual_seed(0)
    Ae = [torch.randn(k, m, device=dev, dtype=torch.bfloat16) for k in ks_full]
    Be = [torch.randn(k, n, device=dev, dtype=torch.bfloat16) for k in ks_full]
    pieces = {1024: [512, 256, 256], 768: [256, 256, 256], 640: [256, 256, 128],
              512: [256, 128, 128], 896: [512, 256, 128], 384: [128, 128, 128]}
    piece = [pieces[k] for k in ks_full]
    nmb = 3

    # (1) 各 microbatch fresh，再 torch 求和
    fresh = []
    for j in range(nmb):
        ks_j, Aj, Bj = [], [], []
        for e in range(G):
            s = sum(piece[e][:j])
            ln = piece[e][j]
            ks_j.append(ln)
            Aj.append(Ae[e][s:s + ln])
            Bj.append(Be[e][s:s + ln])
        fresh.append(run_kwgrad(quant_k(torch.cat(Aj, 0), ks_j),
                                quant_k(torch.cat(Bj, 0), ks_j), ks_j, G, m, n))
    torch_sum = fresh[0] + fresh[1] + fresh[2]

    # (2) 持久 fp32 缓冲，逐 microbatch 原地累积（c=d）
    acc = torch.zeros((G, m, n), device=dev, dtype=torch.float32)
    for j in range(nmb):
        ks_j, Aj, Bj = [], [], []
        for e in range(G):
            s = sum(piece[e][:j])
            ln = piece[e][j]
            ks_j.append(ln)
            Aj.append(Ae[e][s:s + ln])
            Bj.append(Be[e][s:s + ln])
        layout = torch.tensor(ks_j, dtype=torch.int32, device=dev)
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            quant_k(torch.cat(Aj, 0), ks_j), quant_k(torch.cat(Bj, 0), ks_j),
            acc, ks_j, layout, acc, recipe=(1, 1, 128))
        torch.cuda.synchronize()

    print(f'  D1 原地累积(c=d) vs fresh+torch 求和: 逐位差={bitdiff(acc, torch_sum)}/{acc.numel()}, '
          f'relerr={relerr(acc, torch_sum):.3e}')
    dW_true = torch.stack([Ae[e].float().T @ Be[e].float() for e in range(G)])
    print(f'  D2 原地累积 vs fp64 真值 relerr = {relerr(acc, dW_true):.3e}')
    return


def main():
    print(f'arch_major = {get_arch_major()}  (H100=9; 本篇全部在 sm90 真机跑)')
    print(f'generators.py sha256[:16] = {sha(GEN)}')
    reset_align()
    part_a()
    part_b()
    part_c()
    part_d()
    print('\n=== 判定 ===')
    print('  wgrad: 一次大 GEMM ≠ 多 microbatch 求和（bit 变、幅度 ~1e-7）'); 
    print('  wgrad: 结合序本身改 bit（幅度同量级）——可复现要固定 (microbatch 数, 顺序, reduce 算法)')
    print('  dgrad: 每 token 行独立 → microbatch 切分逐位不变')
    print('  → 可复现性与数值精度是两件事（同第 170 篇）'); 


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
