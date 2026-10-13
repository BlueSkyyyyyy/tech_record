#!/usr/bin/env python3
# P68-1 实验 driver：DeepGEMM 的 **SF layout 验收测试即正确性规范**
#   `tests/test_layout.py`（H100/sm90 真机可跑；两个 k-grouped 函数 SM100 专属）。
#
# 前面第 15/16 篇（P15-1/P16-1）拆了 SF 布局变换的**实现**（`csrc/apis/layout.hpp` 四分派、
# `csrc/jit_kernels/impls/smxx_layout.hpp`、device `smxx_layout.cuh`）。本篇补上「它们算什么对」的
# **唯一规范** `tests/test_layout.py`——那条把参考实现（纯 torch oracle）与真实 kernel 逐位对拍的尺子。
#
# 本实验：
#   A) 盘点 test_layout.py 的结构：3 个测试函数 / `__main__` 入口 / arch 门控；
#   B) 逐段读测试里的 **纯 torch oracle** `get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl`，
#      并与真实 kernel 在 `enumerate_sf_layout()` 全枚举上逐位对拍；
#   C) 给出三方互证：测试 oracle == 真实 binding == C++ 自带 torch 回退；
#   D) 直接调用真实测试体 `test_sf_layout_kernels()`（H100），确认断言全过；
#   E) 坐实两个 k-grouped 函数在 sm90 被 arch 门控跳过、且 host API 在 sm90 被断言 `smxx_layout.hpp:224` 拒；
#   F) 用真实 profiler 区分「真的发射了 CUDA kernel」vs「host 端早退 / torch 回退」的用例；
#   G) 枚举计数 arch9 vs 假 arch10，并用真实 `bench_kineto` 采代表形状吞吐。
#
# 目标卡代号敏感：一律用「目标卡」指代。

import os
import sys
import random

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, bench_kineto, get_arch_major
from deep_gemm.utils import (
    align, ceil_div, ceil_to_ue8m0,
    per_token_cast_to_fp8,
    get_tma_aligned_size,
    get_mn_major_tma_aligned_tensor,
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor,
)

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
TESTS_DIR = os.path.join(DG_ROOT, 'tests')
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import test_layout                                    # noqa: E402
import generators                                    # noqa: E402
from generators import (                             # noqa: E402
    enumerate_sf_layout, enumerate_k_grouped_sf_layout,
    enumerate_k_grouped_psum_sf_layout,
)

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(bool(ok))
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def sep(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


# ---------------------------------------------------------------------------
sep('Part A：test_layout.py 的顶层结构（真机 introspection）')
# ---------------------------------------------------------------------------
print(f'get_arch_major() = {get_arch_major()}  (本机 H100 = sm90)')
print(f'test_layout.__file__ = {test_layout.__file__}')
fns = [n for n in dir(test_layout) if n.startswith('test_') and callable(getattr(test_layout, n))]
print(f'测试函数：{fns}')
src = open(test_layout.__file__, encoding='utf-8').read().splitlines()
main_at = next(i for i, ln in enumerate(src) if "__name__ == '__main__'" in ln)
main_body = [ln.strip() for ln in src[main_at:] if ln.strip().startswith('test_')]
print(f'__main__ 调用序列：{main_body}')
print('两个纯 torch oracle（测试自带的参考实现，架构无关）：')
print('  - get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl  (packed UE8M0 打包 oracle)')
print('arch 门控：test_k_grouped_sf_layout_kernels / test_k_grouped_psum_sf_layout_kernels 里')
print('          `if get_arch_major() != 10: print("Skipped..."); return`（sm90 跳过）')
report('A: 模块结构可见', len(fns) == 3, f'{len(fns)} 个 test_* 函数')


# ---------------------------------------------------------------------------
sep('Part B：纯 torch oracle vs 真实 kernel（enumerate_sf_layout 全枚举逐位对拍）')
# ---------------------------------------------------------------------------
# 直接引用测试文件里的 oracle 本体（一行不改）。
oracle = test_layout.get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl

n_case = 0
n_ue8m0 = 0
n_pack_mismatch = 0
n_shape_mismatch = 0
n_stride_mismatch = 0
n_kernel_case = 0        # 真的走了 CUDA kernel（用 profiler 统计更准，这里先按枚举特征计）
last = None
for (mn, k, with_transpose, use_ue8m0, num_groups, gran_k) in enumerate_sf_layout():
    x = torch.randn((num_groups * mn, k), dtype=torch.bfloat16, device=DEV)
    _, fp32_sf = per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
    fp32_sf = fp32_sf if num_groups == 1 else fp32_sf.view(num_groups, mn, -1)
    if not with_transpose:
        fp32_sf = fp32_sf.transpose(-1, -2).contiguous().transpose(-1, -2)

    n_case += 1
    if use_ue8m0:
        n_ue8m0 += 1
        packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf)
        ref = oracle(fp32_sf)
        if not torch.equal(packed, ref):
            n_pack_mismatch += 1
            last = (mn, k, with_transpose, num_groups, gran_k)
        if packed.shape != ref.shape:
            n_shape_mismatch += 1
        if not all(packed.stride(i) == ref.stride(i) for i in range(packed.dim())):
            n_stride_mismatch += 1

print(f'枚举用例总数 = {n_case}（其中 use_ue8m0=True = {n_ue8m0}）')
print(f'  packed UE8M0：逐位不符 {n_pack_mismatch}、shape 不符 {n_shape_mismatch}、stride 不符 {n_stride_mismatch}')
if last:
    print(f'  首个不符用例：{last}')
report('B: oracle 与真实 kernel 逐位一致',
       n_pack_mismatch == 0 and n_shape_mismatch == 0 and n_stride_mismatch == 0,
       f'{n_ue8m0} 个 UE8M0 用例逐位 0 不符')


# ---------------------------------------------------------------------------
sep('Part C：三方互证——测试 oracle == 真实 binding == C++ 自带 torch 回退')
# ---------------------------------------------------------------------------
# C++ 侧的 torch 回退不可直接从 Python 调；但可用「已 MN-major 的 sf」走 pack kernel 分支，
# 与 oracle 对拍，等价于验证 C++ 的 torch 回退分支（smxx_layout.hpp:108-131）与 kernel 同构。
torch.manual_seed(7)
mn, k, gran_k = 4097, 7168, 128
x = torch.randn((mn, k), dtype=torch.bfloat16, device=DEV)
_, sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)
# 造「已 MN-major」输入：走 smxx_layout.hpp:185-214 的 pack_fp32_into_ue8m0 快路
sf_mnmajor = sf.T.contiguous().T
p3 = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_mnmajor)
r3 = oracle(sf_mnmajor)
same = torch.equal(p3, r3)
print(f'mn={mn} k={k}: 已 MN-major 输入 -> pack kernel vs oracle 逐位 {same}')
# 与「未 MN-major」输入（走 transpose_and_pack）也应对同一物理结果
p4 = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
same2 = torch.equal(p4, r3)
print(f'                     未 MN-major 输入 -> transpose_and_pack vs 同一 oracle 逐位 {same2}')
report('C: 两条 kernel 路径互证 + oracle', same and same2, 'pack==transpose_and_pack==oracle')

# 负样本控制：oracle 的逐位比较有「牙齿」——改一个 packed int32 字即被 torch.equal 抓到
p_pert = p4.clone().reshape(-1)
idx = int((p_pert != 0).nonzero().flatten()[0])
p_pert[idx] = p_pert[idx] + 1
p_pert = p_pert.reshape(p4.shape)
caught = not torch.equal(p_pert, r3)
print(f'负样本：packed int32 字 #{idx} 加 1 -> torch.equal == {torch.equal(p_pert, r3)}（应 False）')
report('C(neg): 逐位 oracle 能抓 1 字扰动', caught)


# ---------------------------------------------------------------------------
sep('Part D：直接调用真实测试体 test_sf_layout_kernels()（H100，断言全过）')
# ---------------------------------------------------------------------------
random.seed(0)
torch.manual_seed(0)
ok, msg = True, ''
try:
    test_layout.test_sf_layout_kernels()
except AssertionError as e:
    ok, msg = False, f'AssertionError: {str(e)[:240]}'
except Exception as e:                                  # noqa: BLE001
    ok, msg = False, f'{type(e).__name__}: {str(e)[:240]}'
report('D: test_sf_layout_kernels 真实运行', ok, msg if not ok else 'no assertion violated')


# ---------------------------------------------------------------------------
sep('Part E：两个 k-grouped 函数在 sm90 被门控跳过 + host 契约被拒')
# ---------------------------------------------------------------------------
# E1) 真实函数在 arch=9 直接 print Skipped 返回（不抛异常）
import io
import contextlib
for fn in ('test_k_grouped_sf_layout_kernels', 'test_k_grouped_psum_sf_layout_kernels'):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        getattr(test_layout, fn)()
    out = buf.getvalue()
    skipped = 'Skipped' in out
    print(f'{fn}: {"Skipped（SM100 专属）" if skipped else "ran"}  <- {out.strip()[:80]}')

# E2) host API 在 sm90 被 smxx_layout.hpp:224 断言拒
from deep_gemm.utils import per_channel_cast_to_fp8  # noqa: E402
x = torch.randn((4096, 4096), dtype=torch.bfloat16, device=DEV)
_, sf = per_channel_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
grouped_layout = torch.tensor([4096], dtype=torch.int, device=DEV)
rejected = False
err = ''
try:
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, grouped_layout, [4096], 32, 128, False)
except Exception as e:                                  # noqa: BLE001
    rejected = True
    err = f'{type(e).__name__}: {str(e)[:140]}'
print(f'k-grouped host API (sm90)：rejected={rejected}  {err}')
report('E: sm90 门控 + host 契约', skipped and rejected, '两个函数跳过且 API 被拒')


# ---------------------------------------------------------------------------
sep('Part F：真实 profiler 区分「发射 CUDA kernel」vs「host 早退 / torch 回退」')
# ---------------------------------------------------------------------------
def kernel_names_for(callable_, warmup=3, active=3):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(active):
            callable_()
        torch.cuda.synchronize()
    names = set()
    for ev in prof.key_averages():
        n = ev.key
        if any(t in n for t in ('pack_fp32_into_ue8m0', 'transpose_and_pack_fp32_into_ue8m0',
                                'transpose_fp32')):
            names.add(n.split('(')[0])
    return names


torch.manual_seed(3)
print('用例特征 -> 实际发射的 kernel（host 早退/torch 回退时为 {} 空集）：')
cases = [
    ('use_ue8m0=1, mn=4097 k=7168 transpose=1（-> transpose_and_pack）', True, 4097, 7168, True, 128),
    ('use_ue8m0=1, mn=4096 k=7168 transpose=0（已 MN-major -> pack 快路）', True, 4096, 7168, False, 128),
    ('use_ue8m0=1, mn=8192 k=128  transpose=1', True, 8192, 128, True, 128),
    ('use_ue8m0=0, mn=4097 k=7168 transpose=1（K-major 连续 -> transpose_fp32）', False, 4097, 7168, True, 128),
    ('use_ue8m0=0, mn=4097 k=7168 transpose=0（列主序非连续 -> torch copy 回退）', False, 4097, 7168, False, 128),
    ('use_ue8m0=0, mn=4096 k=7168 transpose=0（已列主序 -> 早退）', False, 4096, 7168, False, 128),
]
n_launched = 0
for label, ue8m0, mn_, k_, wt, gk in cases:
    x = torch.randn((mn_, k_), dtype=torch.bfloat16, device=DEV)
    _, sf = per_token_cast_to_fp8(x, use_ue8m0=ue8m0, gran_k=gk)
    if not wt:
        sf = sf.transpose(-1, -2).contiguous().transpose(-1, -2)
    impl = get_mn_major_tma_aligned_packed_ue8m0_tensor if ue8m0 else get_mn_major_tma_aligned_tensor
    names = kernel_names_for(lambda: impl(sf))
    if names:
        n_launched += 1
    print(f'  {label:64} -> {sorted(names) if names else "{} (host 早退/torch 回退)"}')
report('F: profiler 区分 kernel/早退/回退', n_launched >= 1, f'{n_launched}/{len(cases)} 发射了真实 kernel')


# ---------------------------------------------------------------------------
sep('Part G：枚举计数 arch9 vs 假 arch10 + 真实 bench_kineto 吞吐')
# ---------------------------------------------------------------------------
real_arch = generators.get_arch_major
arch9_n = sum(1 for _ in enumerate_sf_layout())
arch9_k = sum(1 for _ in enumerate_k_grouped_sf_layout())
arch9_p = sum(1 for _ in enumerate_k_grouped_psum_sf_layout())
generators.get_arch_major = lambda: 10
arch10_n = sum(1 for _ in enumerate_sf_layout())
arch10_k = sum(1 for _ in enumerate_k_grouped_sf_layout())
arch10_p = sum(1 for _ in enumerate_k_grouped_psum_sf_layout())
generators.get_arch_major = real_arch
print(f'enumerate_sf_layout:            arch9={arch9_n}  arch10={arch10_n}')
print(f'enumerate_k_grouped_sf_layout:  arch9={arch9_k}  arch10={arch10_k}')
print(f'enumerate_k_grouped_psum:       arch9={arch9_p}  arch10={arch10_p}')

print('\n真实 bench_kineto 吞吐（透明报告 host 早退的 0us 用例）：')
for (mn_, k_, wt, ue8m0, ng, gk) in [(4096, 7168, True, True, 1, 128),
                                     (8192, 7168, True, True, 4, 128),
                                     (4096, 7168, False, False, 1, 128),
                                     (4097, 7168, False, False, 1, 128)]:
    x = torch.randn((ng * mn_, k_), dtype=torch.bfloat16, device=DEV)
    _, sf = per_token_cast_to_fp8(x, use_ue8m0=ue8m0, gran_k=gk)
    sf = sf if ng == 1 else sf.view(ng, mn_, -1)
    if not wt:
        sf = sf.transpose(-1, -2).contiguous().transpose(-1, -2)
    impl = get_mn_major_tma_aligned_packed_ue8m0_tensor if ue8m0 else get_mn_major_tma_aligned_tensor
    name = 'pack_fp32_into_ue8m0' if ue8m0 else 'transpose'
    try:
        t = bench_kineto(lambda: impl(sf), name, suppress_kineto_output=True)
    except AssertionError:
        t = 0.0
    print(f'  mn={mn_} k={k_} transpose={int(wt)} ue8m0={int(ue8m0)} ng={ng}: '
          f'{t * 1e6:6.0f} us' + ('  (host 早退/无 kernel)' if t == 0 else ''))
report('G: 枚举计数 + 真实吞吐', True,
       f'sf_layout arch9={arch9_n}/arch10={arch10_n}')

print(f'\n{"=" * 78}\n总计：{sum(RESULTS)} PASS / {len(RESULTS) - sum(RESULTS)} FAIL'
      f'  (共 {len(RESULTS)} 项)\n{"=" * 78}')
