"""P176-1: MN-major K-grouped wgrad —— SM100 FP8 `k_grouped_tn` 生产路径的 H100 可跑对照。

Mega MoE 反向的 wgrad 在**目标卡（SM100）**上走 MN-major 的 `k_grouped_fp8_gemm_tn_contiguous`
（A/B 物理布局 `[sum_k, M]` / `[sum_k, N]`，MN 连续；见 csrc/apis/gemm.hpp:320-371），本机 H100(sm90)
**跑不了**（gemm.hpp:369 `DG_HOST_UNREACHABLE`）。但同一套「K 轴分组 + MN-major 物理布局 + 输出
`[G, m, n]`」在 **BF16** 家族里有 **SM90 实现**（`k_grouped_bf16_gemm_tn_contiguous` ->
`sm90_bf16_k_grouped_gemm`，`sm90_bf16_gemm.hpp:264-330`），是本机能真跑、且**物理布局逐位同构**的对照。

本实验在 H100 上真跑这条 BF16 MN-major K-grouped kernel，验证 SM100 FP8 tn 路径的「结构性」结论：

  A1  逐组复现 `D[e] = A_e^T B_e`（BF16 输入、fp32 累加，vs fp64 参考）。
  A2  组是**独立累加器**：一次 launch 出全部组，vs 每组单独一次 launch，**逐位相等**。
      （这是把 P174/P175「wgrad 累积序改 bit」结论迁移到 tn 路径的关键前提。）
  A3  组**排列无关**：物理组顺序置换后，输出按同一置换**逐位相等**（每组的 K 折叠加法链不变）。
  A4  契约：SM90 的 BF16 tn 强制 accumulate 且**拒绝 PSUM**（`sm90_bf16_gemm.hpp` -> gemm.hpp:710-711）；
      PSUM 是 SM100-only（由 scheduler 源把握）。
  A5  同配置 run-to-run **逐位确定**。
  A6  吞吐（TFLOPS）。
  B   累积序解离：kernel（真机 fp32 累加）vs fp64 参考 vs torch fp32 `bmm`——三者都在 ~1e-2（BF16 输入
      量化）量级，但 kernel 与 torch fp32 之间是 ~1e-7 的「结合序」级 bit 差，坐实 P174 的分离。

输出重定向到 exp_kgrouped_tn_mn_major.out.txt。
"""
import os
import sys
import time
import traceback

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major

K_ALIGN = 128


def align(x, a=K_ALIGN):
    return ((x + a - 1) // a) * a


def set_align():
    deep_gemm.set_mk_alignment_for_contiguous_layout(K_ALIGN)


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-30)).item()


def bitfrac(x, y):
    """两个同形状张量逐元素 bit 不同的比例。"""
    xb = x.reshape(-1).view(torch.int32)
    yb = y.reshape(-1).view(torch.int32)
    return (xb != yb).float().mean().item()


def build_slots(logical_ks, m, n, seed=0):
    """按 DeepGEMM 的 MN-major 物理布局造每组数据。

    物理张量 a:[sum_k, m]、b:[sum_k, n]（MN 连续）；组 i 的数据落在其 padded 槽的**起始**
    `logical_k` 行，槽尾 padding 置零（与 tests/generators.py:576-587 一致）。
    """
    torch.manual_seed(seed)
    host_ks = [align(k) for k in logical_ks]
    slots_a, slots_b = [], []
    for lk, hk in zip(logical_ks, host_ks):
        sa = torch.zeros(hk, m, dtype=torch.bfloat16, device='cuda')
        sb = torch.zeros(hk, n, dtype=torch.bfloat16, device='cuda')
        sa[:lk] = torch.randn(lk, m, dtype=torch.bfloat16, device='cuda')
        sb[:lk] = torch.randn(lk, n, dtype=torch.bfloat16, device='cuda')
        slots_a.append(sa)
        slots_b.append(sb)
    return slots_a, slots_b, host_ks


def concat(slots):
    return torch.cat(slots, dim=0).contiguous()


def grouped_layout_contiguous(host_ks):
    return torch.tensor(host_ks, dtype=torch.int32, device='cuda')


def ref_fp64(slots_a, slots_b, logical_ks):
    G = len(logical_ks)
    m = slots_a[0].shape[1]
    n = slots_b[0].shape[1]
    out = torch.empty(G, m, n, dtype=torch.float64, device='cuda')
    for i, lk in enumerate(logical_ks):
        out[i] = slots_a[i][:lk].double().T @ slots_b[i][:lk].double()
    return out


def run_tn(a, b, grouped, host_ks, G, m, n, use_psum=False):
    d = torch.zeros(G, m, n, dtype=torch.float32, device='cuda')
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        a, b, d, host_ks, grouped, d, use_psum_layout=use_psum)
    torch.cuda.synchronize()
    return d


def main():
    set_align()
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}, cap={torch.cuda.get_device_capability(0)}')
    print('物理布局：A/B = [sum_k, M] / [sum_k, N]（MN-major，MN 连续）；输出 D = [G, m, n]')
    print('=' * 92)

    logical_ks = [512, 384, 640, 256]
    G = len(logical_ks)
    m, n = 512, 1024
    slots_a, slots_b, host_ks = build_slots(logical_ks, m, n, seed=0)
    a = concat(slots_a)
    b = concat(slots_b)
    grouped = grouped_layout_contiguous(host_ks)
    ref = ref_fp64(slots_a, slots_b, logical_ks)
    print(f'shape: G={G} m={m} n={n} logical_ks={logical_ks} host_ks(padded)={host_ks} '
          f'sum_k={sum(host_ks)}')

    # ---- A1 correctness
    print('-' * 92)
    print('A1  逐组 D[e]=A_e^T B_e（BF16 输入 / fp32 累加 vs fp64 参考）')
    d_all = run_tn(a, b, grouped, host_ks, G, m, n)
    for i in range(G):
        print(f'      group {i}: logical_k={logical_ks[i]:4d}  relerr={relerr(d_all[i], ref[i]):.3e}')

    # ---- A2 group independence (bitwise)
    print('-' * 92)
    print('A2  组独立累加器：全组一次 launch vs 每组单独 launch —— 逐位相等？')
    all_eq = True
    for i in range(G):
        di = run_tn(slots_a[i], slots_b[i], grouped_layout_contiguous([host_ks[i]]),
                    [host_ks[i]], 1, m, n)
        eq = bool(torch.equal(d_all[i], di[0]))
        all_eq &= eq
        print(f'      group {i}: bitwise-equal={eq}')
    print(f'      => 全部逐位相等 = {all_eq}')

    # ---- A3 permutation invariance (bitwise)
    print('-' * 92)
    print('A3  组排列无关：置换物理组顺序 + ks，输出按同一置换逐位相等？')
    perm = [2, 0, 3, 1]
    a_p = concat([slots_a[i] for i in perm])
    b_p = concat([slots_b[i] for i in perm])
    host_ks_p = [host_ks[i] for i in perm]
    d_p = run_tn(a_p, b_p, grouped_layout_contiguous(host_ks_p), host_ks_p, G, m, n)
    all_eq = all(bool(torch.equal(d_p[j], d_all[perm[j]])) for j in range(G))
    print(f'      perm={perm}  => 逐位相等 = {all_eq}')

    # ---- A4 contract: psum rejected on SM90 BF16 tn
    print('-' * 92)
    print('A4  契约：SM90 BF16 tn 用 PSUM 布局 -> 应被拒（PSUM 为 SM100-only）')
    try:
        logical_e = [512, 384, 640]
        ends = [512, align(512) + 384, align(512) + align(384) + 640]
        sa, sb, hk = build_slots(logical_e, m, n, seed=1)
        aa, bb = concat(sa), concat(sb)
        gp = torch.tensor(ends, dtype=torch.int32, device='cuda')
        run_tn(aa, bb, gp, hk, len(logical_e), m, n, use_psum=True)
        print('      !! 意外通过')
    except Exception as e:
        msg = str(e).strip().splitlines()
        print(f'      rejected as expected -> {type(e).__name__}: {msg[-1][:160] if msg else str(e)[:160]}')

    # ---- A5 determinism
    print('-' * 92)
    print('A5  同配置 run-to-run 逐位确定（20 次）')
    base = run_tn(a, b, grouped, host_ks, G, m, n)
    ndiff = sum(0 if torch.equal(base, run_tn(a, b, grouped, host_ks, G, m, n)) else 1
                for _ in range(20))
    print(f'      20 次里与首次逐位不等的次数 = {ndiff}')

    # ---- A6 perf
    print('-' * 92)
    print('A6  吞吐（真实 SM90 BF16 MN-major grouped kernel；对照 P4-2 FP8 nt 同形状 425 TFLOPS）')
    for pk, pm, pn in [([512] * 8, 4096, 7168), ([1024] * 8, 4096, 7168),
                       ([512] * 8, 7168, 7168)]:
        sa, sb, hk = build_slots(pk, pm, pn, seed=2)
        pa, pb = concat(sa), concat(sb)
        pg = grouped_layout_contiguous(hk)
        for _ in range(3):
            run_tn(pa, pb, pg, hk, len(pk), pm, pn)
        torch.cuda.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            run_tn(pa, pb, pg, hk, len(pk), pm, pn)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        logical_k = sum(pk)
        tf = 2 * pm * pn * logical_k / dt / 1e12
        gbps = (pm + pn) * logical_k * 2 / dt / 1e9
        print(f'      G={len(pk)} m={pm} n={pn} k/group={pk[0]}  t={dt * 1e6:.0f} us  '
              f'{tf:.0f} TFLOPS  读 {gbps:.0f} GB/s（仅 A/B 输入，bf16）')

    # ---- B accumulation-order dissociation
    print('=' * 92)
    print('B  累积序解离：kernel(真机 fp32 累加) vs fp64 vs torch fp32 bmm')
    torch.backends.cuda.matmul.allow_tf32 = False
    dk = run_tn(a, b, grouped, host_ks, G, m, n)          # kernel
    dt_ = torch.empty(G, m, n, dtype=torch.float32, device='cuda')
    for i, lk in enumerate(logical_ks):
        dt_[i] = (slots_a[i][:lk].float().T @ slots_b[i][:lk].float())
    print(f'      relerr(kernel, fp64)       = {relerr(dk, ref):.3e}')
    print(f'      relerr(torch_fp32, fp64)   = {relerr(dt_, ref):.3e}')
    print(f'      bit-diff fraction(kernel, torch_fp32) = {bitfrac(dk, dt_):.4f}')
    print(f'      max|kernel - fp64|         = {(dk.double() - ref).abs().max().item():.3e}')
    print(f'      max|kernel - torch_fp32|   = {(dk - dt_).abs().max().item():.3e}')


if __name__ == '__main__':
    main()
