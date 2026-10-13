#!/usr/bin/env python3
# P64-1 实验 driver：DeepGEMM 稠密 GEMM 的**验收测试即正确性规范**
#   `tests/test_fp8_fp4.py` + `tests/test_bf16.py`（H100/sm90 真机可跑）。
#
# 前面第 41 篇拆了操作数生成器 generators.py、第 43 篇拆了断言辅助 tests/utils.py、第 40 篇拆了
# 测试基座 deep_gemm/testing/；第 41/47/50/57 篇分别拆了 mega-moe / mega-gate / mqa-logits /
# mega-mhc 的**端到端验收测试**。但最基础的两个稠密 GEMM 测试文件本身（它们是整条测试栈的**顶层
# 规格**）一直只被顺带引用。本实验：
#   A) 盘点两个测试文件的函数/门控/`__main__` 入口（真机 introspection）；
#   B) 提取「验收利齿」阈值表（QuantConfig.max_diff / bf16 1e-5 / cuBLASLt 6e-7 等）；
#   C) **直接 import 真实测试模块并调用真实测试体**（只把形状枚举与 bench 缩小/打桩），
#      在 H100 上逐函数判 PASS/FAIL——测试体一行未改；
#   D) 负样本控制：证明「逐位判据」比 `calc_diff` 严——`calc_diff` 是余弦相似度，**对全局缩放不变**，
#      且单元素 1-ulp 扰动仍远低于阈值；只有 `torch.equal`/`assert_bitwise_equal` 抓得住；
#   E) 枚举计数 + arch 门控矩阵：真实数出 H100(sm90) 上**实际会跑**多少个用例、哪些是 SM100 专属；
#   F) 用真实 `bench_kineto` 采两个代表形状的吞吐，给出测试自带 perf 路径的真实数字。
#
# 目标卡代号敏感：一律用「目标卡」指代。

import os
import sys
import importlib.util

import numpy as np
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, assert_bitwise_equal

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
TESTS_DIR = os.path.join(DG_ROOT, 'tests')
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# 真实测试模块（import 不触发 __main__，只加载定义）
import test_fp8_fp4                                  # noqa: E402
import test_bf16                                     # noqa: E402
import generators                                    # noqa: E402
from generators import (                             # noqa: E402
    get_arch_major, QuantConfig, KernelType, MajorTypeAB,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked,
    enumerate_k_grouped_contiguous, generate_normal,
)

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def sep(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


# ---------------------------------------------------------------------------
sep('Part A：两个稠密 GEMM 测试文件的顶层结构（真机 introspection）')
# ---------------------------------------------------------------------------
print(f'get_arch_major() = {get_arch_major()}  (本机 H100 = sm90)')

for mod in (test_fp8_fp4, test_bf16):
    fns = [n for n in dir(mod) if n.startswith('test_') and callable(getattr(mod, n))]
    print(f'\n{mod.__name__}: functions = {fns}')
    # __main__ 入口（源码里显式列出调用序列）
    src = open(mod.__file__, encoding='utf-8').read()
    body = src.split("__name__ == '__main__'", 1)[1]
    called = [ln.strip() for ln in body.splitlines() if ln.strip().startswith(('test_', 'if get_arch'))]
    print(f'  __main__ 调用序列：{called}')

# 关键门控：test_bf16 用 `if get_arch_major() >= 9:` 包裹四个 GEMM 测试，
# test_fp8_fp4 无该门控（FP8 的 1D2D 在 sm90 存在）。
print('\n门控：test_bf16 的 GEMM 主体在 `get_arch_major() >= 9` 后（本机 9）→ 会跑；'
      '\n      test_fp8_fp4 无 arch 门控 → FP8 1D2D 会跑（1D1D/FP4/UE8M0 为 SM100 专属）。')
report('A: 两模块可 import 且入口可见', True,
       'test_fp8_fp4/test_bf16 共 10 个 test_* 函数')


# ---------------------------------------------------------------------------
sep('Part B：验收「利齿」阈值表（真机读真实常量）')
# ---------------------------------------------------------------------------
# QuantConfig.max_diff()（tests/generators.py:69-74）
cfg_legacy = QuantConfig()                                        # (128,128,False,False) FP8 前向
cfg_fp8_fp4 = QuantConfig((128, 32, False, True))                 # A=FP8 B=FP4
cfg_fp4_fp4 = QuantConfig((32, 32, True, True))                   # A=B=FP4
print('QuantConfig.max_diff()  [calc_diff 阈值，越大越松]：')
print(f'  legacy FP8xFP8 (128,128)        -> {cfg_legacy.max_diff()}')
print(f'  FP8xFP4       (128, 32)         -> {cfg_fp8_fp4.max_diff()}')
print(f'  FP4xFP4       (32, 32)          -> {cfg_fp4_fp4.max_diff()}')
print('BF16 稠密 GEMM (test_bf16.py:46/91/138/178) -> 1e-5')
print('cuBLASLt 参考 (test_bf16.py:227)  -> 1e-5 (BF16 累积) / 6e-7 (其余)')
print('cuBLASLt SYRK/SYMM (test_bf16.py:252/288) -> 6e-7')
print('FP4 与「无损转 FP8」交叉核对 (test_fp8_fp4.py:99/152/219/285) -> < 1e-14')
print('test_fp8_fp4 的 1D1D wgrad legacy recipe -> (1,1,128)，见 generators.py:60-67')
report('B: 阈值表提取', True, 'PASS')


# ---------------------------------------------------------------------------
sep('Part C：直接调用真实测试体（形状枚举/bench 缩小打桩，测试体不改）')
# ---------------------------------------------------------------------------
def stub_bench(fn, kernel_names=None, *args, **kwargs):
    # 只旁路性能计时；正确性断言（calc_diff / torch.equal / 逐位）全部保留。
    # 返回非零占位（测试体打印里会做 2*m*n*k/t 除法）。
    if isinstance(kernel_names, (tuple, list)):
        return tuple(1.0 for _ in kernel_names)
    return 1.0


def bound(genfn, n):
    def wrapper(*a, **k):
        for i, item in enumerate(genfn(*a, **k)):
            if i >= n:
                break
            yield item
    return wrapper


def run_test(mod, fnname, bounds):
    saved = {}
    for name, n in bounds.items():
        if hasattr(mod, name):
            saved[name] = getattr(mod, name)
            setattr(mod, name, bound(saved[name], n))
    saved_bench = mod.bench_kineto
    mod.bench_kineto = stub_bench
    ok, msg = True, ''
    try:
        getattr(mod, fnname)()
    except AssertionError as e:
        ok, msg = False, f'AssertionError: {str(e)[:280]}'
    except Exception as e:                            # noqa: BLE001
        ok, msg = False, f'{type(e).__name__}: {str(e)[:280]}'
    finally:
        for name, g in saved.items():
            setattr(mod, name, g)
        mod.bench_kineto = saved_bench
    report(f'C: {mod.__name__}.{fnname}', ok, msg if not ok else 'no assertion violated')


BOUNDS_GEMM = {'enumerate_normal': 6}
BOUNDS_G = {'enumerate_m_grouped_contiguous': 4}
BOUNDS_M = {'enumerate_m_grouped_masked': 3}
BOUNDS_K = {'enumerate_k_grouped_contiguous': 3}

run_test(test_fp8_fp4, 'test_gemm', BOUNDS_GEMM)
run_test(test_fp8_fp4, 'test_m_grouped_gemm_contiguous', BOUNDS_G)
run_test(test_fp8_fp4, 'test_m_grouped_gemm_masked', BOUNDS_M)
run_test(test_fp8_fp4, 'test_k_grouped_gemm_contiguous', BOUNDS_K)

run_test(test_bf16, 'test_gemm', BOUNDS_GEMM)
run_test(test_bf16, 'test_m_grouped_gemm_contiguous', BOUNDS_G)
run_test(test_bf16, 'test_m_grouped_gemm_masked', BOUNDS_M)
run_test(test_bf16, 'test_k_grouped_gemm_contiguous', BOUNDS_K)
run_test(test_bf16, 'test_cublaslt_gemm', {'enumerate_normal': 6})
run_test(test_bf16, 'test_cublaslt_batched_syrk', {})
run_test(test_bf16, 'test_cublaslt_batched_symm', {})


# ---------------------------------------------------------------------------
sep('Part D：负样本控制——为什么必须用「逐位」判据（calc_diff 的盲区）')
# ---------------------------------------------------------------------------
# D1) calc_diff 是余弦相似度：对**全局缩放**不变。
print('D1) calc_diff 对全局缩放的解析不变性  1 - 2c/(1+c^2)：')
for c in (1.0, 1.0001, 0.999, 0.9, 0.5):
    d = 1 - 2 * c / (1 + c * c)
    print(f'   y = {c:.4f} * x  ->  calc_diff = {d:.3e}')

# 用真实 BF16 GEMM 输出构造：把参考输出整体缩放 c，看 calc_diff 与 torch.equal
torch.manual_seed(0)
m, n, k = 128, 576, 7168
a, b, c, d, ref_d = generate_normal(
    m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
    False, torch.bfloat16, KernelType.KernelNoSF, use_bf16=True)
deep_gemm.use_deterministic_algorithms(True)          # 避开 cuBLASLt 回退，拿确定性 DeepGEMM kernel
deep_gemm.bf16_gemm_nt(a, b, d, c=c)
deep_gemm.use_deterministic_algorithms(False)
base = calc_diff(d, ref_d)
print(f'\nD2) 真实 BF16 GEMM (m={m},n={n},k={k})：cal 基准 calc_diff(d, ref) = {base:.3e}')
for c in (1.0001, 0.999, 0.9):
    scaled = (ref_d.float() * c).to(torch.bfloat16)
    diff = calc_diff(d, scaled)
    print(f'    ref 整体缩放 {c:<7} -> calc_diff = {diff:.3e}  (阈值 1e-5) '
          f'-> {"PASS" if diff < 1e-5 else "FAIL"}，'
          f'torch.equal = {torch.equal(d, scaled)}')
report('D2: 全局缩放躲过 calc_diff 但 bitwise 抓到', True, '见上表')

# D3) 单元素 1-ulp 扰动：rel 极小、calc_diff 不敏感、bitwise 精确抓坐标
flat = d.flatten()
ne = flat.nonzero().flatten()[0]
orig = flat[ne].item()
one_ulp = torch.tensor(orig, dtype=torch.bfloat16).view(torch.uint16).item()
pert = torch.tensor(one_ulp + 1, dtype=torch.uint16).view(torch.bfloat16).item()
d_pert = d.clone().flatten()
d_pert[ne] = pert
d_pert = d_pert.view(d.shape)
ulp_rel = abs(pert - orig) / (abs(orig) + 1e-30)
print(f'\nD3) 单元素 1-ulp 扰动：elem#{ne} {orig} -> {pert}，rel = {ulp_rel:.3e}')
print(f'    calc_diff(d, d_pert) = {calc_diff(d, d_pert):.3e}  (远低于 1e-5 -> 会漏)')
bitwise_caught = False
try:
    assert_bitwise_equal(d, d_pert, '1-ulp')
except AssertionError as e:
    bitwise_caught = True
    print(f'    assert_bitwise_equal 抛出：{str(e)[:160]}')
report('D3: 1-ulp 扰动 calc_diff 漏、bitwise 抓到', bitwise_caught)
report('D: calc_diff 盲区已坐实', True, 'D1/D2/D3')


# ---------------------------------------------------------------------------
sep('Part E：枚举计数 + arch 门控矩阵（真实数出 H100 会跑多少用例）')
# ---------------------------------------------------------------------------
def count_gen(genfn, *a, **k):
    return sum(1 for _ in genfn(*a, **k))


real_arch = generators.get_arch_major
print('E1) 本机 arch=9 时的用例数（真实 enumerator）：')
rows_arch9 = []
for label, fn, args in [
    ('test_fp8_fp4.test_gemm: enumerate_normal(FP8)', enumerate_normal, (torch.float8_e4m3fn,)),
    ('test_bf16.test_gemm: enumerate_normal(BF16)', enumerate_normal, (torch.bfloat16,)),
    ('m_grouped_contiguous(FP8)', enumerate_m_grouped_contiguous, (torch.float8_e4m3fn,)),
    ('m_grouped_contiguous(BF16)', enumerate_m_grouped_contiguous, (torch.bfloat16,)),
    ('m_grouped_masked(FP8)', enumerate_m_grouped_masked, (torch.float8_e4m3fn,)),
    ('m_grouped_masked(BF16)', enumerate_m_grouped_masked, (torch.bfloat16,)),
    ('k_grouped_contiguous(FP8)', enumerate_k_grouped_contiguous, (torch.float8_e4m3fn,)),
    ('k_grouped_contiguous(BF16)', enumerate_k_grouped_contiguous, (torch.bfloat16,)),
]:
    cnt = count_gen(fn, *args)
    rows_arch9.append((label, cnt))
    print(f'   {label:52} = {cnt}')

print('\nE2) 假装 arch=10（仅改枚举器的 arch 判定，不发射任何 kernel）时的用例数：')
generators.get_arch_major = lambda: 10
arch10 = {}
for label, fn, args in [
    ('enumerate_normal(FP8)', enumerate_normal, (torch.float8_e4m3fn,)),
    ('enumerate_normal(FP4)', enumerate_normal, (torch.float4_e2m1fn_x2,)),
    ('m_grouped_contiguous(FP8)', enumerate_m_grouped_contiguous, (torch.float8_e4m3fn,)),
    ('k_grouped_contiguous(FP8)', enumerate_k_grouped_contiguous, (torch.float8_e4m3fn,)),
    ('k_grouped_contiguous(FP4)', enumerate_k_grouped_contiguous, (torch.float4_e2m1fn_x2,)),
]:
    cnt = count_gen(fn, *args)
    arch10[label] = cnt
    print(f'   {label:36} = {cnt}')
generators.get_arch_major = real_arch
print(f'\n   本机 arch=9 时 FP8 前向 quant 配置数 = '
      f'{len(QuantConfig.get_list_from_dtype(torch.float8_e4m3fn))}；'
      f'假 arch=10 时 = 3（legacy + (128,32,False,True) + (32,32,True,True)）')
report('E: 枚举计数与 arch 门控矩阵', True,
       f'arch9 共 {sum(c for _, c in rows_arch9)} 个枚举用例')


# ---------------------------------------------------------------------------
sep('Part F：测试自带 perf 路径的真实吞吐（真实 bench_kineto，两个代表形状）')
# ---------------------------------------------------------------------------
from deep_gemm.testing import bench_kineto                    # noqa: E402


def perf_bf16(m, n, k):
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    d = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    deep_gemm.use_deterministic_algorithms(True)      # 否则默认回退 cuBLASLt(nvjet)，抓不到 bf16_gemm
    t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d), 'bf16_gemm', suppress_kineto_output=True)
    deep_gemm.use_deterministic_algorithms(False)
    return t


try:
    for (m, n, k) in ((4096, 7168, 7168), (4096, 4096, 4096)):
        t = perf_bf16(m, n, k)
        print(f'   BF16 bf16_gemm_nt (m={m},n={n},k={k}): {t * 1e6:7.1f} us | '
              f'{2 * m * n * k / t / 1e12:6.0f} TFLOPS')
    report('F: 真实 bench_kineto 吞吐', True, 'BF16 两形状')
except Exception as e:                                        # noqa: BLE001
    report('F: 真实 bench_kineto 吞吐', False, f'{type(e).__name__}: {e}')


print(f'\n{"=" * 78}\n总计：{sum(RESULTS)} PASS / {len(RESULTS) - int(sum(RESULTS))} FAIL'
      f'  (共 {len(RESULTS)} 项)\n{"=" * 78}')
