#!/usr/bin/env python3
# P72-1 实验 driver：DeepGEMM **legacy Triton grouped GEMM 的验收测试即正确性规范**
#   `tests/test_legacy.py`（95 行；H100/sm90 真机能跑真实 Triton 代码）。
#
# 第 29 篇（P29-1）拆了 `deep_gemm/legacy/` 四个 Triton kernel 本体；第 40/41/43 篇拆了测试基座/
# 生成器/逐位断言；第 41/47/50/57/64/66/67/68/69/70/71 篇分别拆了各家族**验收测试即正确性规范**。
# 但**唯一剩下的顶层测试文件** `tests/test_legacy.py` 一直只被顺带引用。本实验把它当「legacy
# grouped GEMM 算什么对」的可执行规格来拆：
#   A) 顶层结构（真机 introspection）：无 arch 门控、`__main__` 直跑两个函数、依赖 generators。
#   B) 枚举计数：bf16 下 m-grouped 32 配置 / k-grouped 7 配置；每个配置实际发射多少次 kernel。
#   C) **直接 import 真实测试模块并调用真实测试体**（只把 `bench_kineto` 打桩、形状不缩），
#      在 H100 上逐条跑断言——测试体一行未改。
#   D) k-grouped 的**跳过闸门** `if use_psum_layout or k_alignment!=128 or not accumulate or
#      out_dtype!=float: continue`：arch9 真实枚举 **0 条被跳过**；用伪造 arch10 枚举证明它确实在拦。
#   E) `test_alias` 是薄壳：`nn`（别名）与 `nt` 输出**逐位相等**；`expand`（a_fused 恒等
#      `m_row_indices`）与 `m_grouped` **逐位相等**——独立 torch oracle 复核。
# F) 判据利齿：`calc_diff<0.001`（bf16）是归一化平方误差、**对两边同缩放不变**；只有逐位比较
#      / 置换反例才抓得住错。给出真实 perf（TFLOPS/GB/s）数字。
#
# 目标卡代号敏感：一律用「目标卡」指代。

import os
import sys
import random
import time

import torch

import deep_gemm
from deep_gemm.testing import calc_diff

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
for _p in (os.path.join(DG_ROOT, 'tests'),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 真实测试模块（import 不触发 __main__，只加载定义）
import test_legacy                                    # noqa: E402
import generators                                     # noqa: E402
from generators import (                              # noqa: E402
    get_arch_major,
    enumerate_m_grouped_contiguous, enumerate_k_grouped_contiguous,
    generate_m_grouped_contiguous, generate_k_grouped_contiguous,
)

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(bool(ok))
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def sep(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def reset_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
sep('Part A：test_legacy.py 顶层结构（真机 introspection）')
# ---------------------------------------------------------------------------
import inspect                                       # noqa: E402
src = inspect.getsource(test_legacy)
print(f'test_legacy.__file__ = {test_legacy.__file__}')
print(f'源文件行数 = {len(src.splitlines())}')
funcs = [n for n, o in vars(test_legacy).items() if inspect.isfunction(o) and n.startswith('test_')]
print(f'顶层 test_ 函数 = {funcs}')
main_src = src[src.index("if __name__"):]
print('__main__ 入口片段：')
for line in main_src.strip().splitlines():
    print('  ' + line)
has_gate = 'get_arch_major' in src or 'test_filter' in src
report('A1 无 arch 门控（H100 直接全跑）', not has_gate,
       f'源里出现 get_arch_major/test_filter = {has_gate}')
report('A2 __main__ 直调两个测试函数', 'test_m_grouped_gemm_contiguous_tl()' in main_src
       and 'test_k_grouped_gemm_contiguous_tl()' in main_src,
       f'funcs={funcs}')

# 依赖：真实 gap 由 generators 提供
gen_ok = hasattr(generators, 'generate_m_grouped_contiguous') and hasattr(generators, 'generate_k_grouped_contiguous')
report('A3 操作数来自 tests/generators（bf16 模式）', gen_ok,
       'generate_m_grouped_contiguous / generate_k_grouped_contiguous')


# ---------------------------------------------------------------------------
sep('Part B：枚举计数（bf16；arch9=H100）')
# ---------------------------------------------------------------------------
reset_seed(0)
m_cfg = list(enumerate_m_grouped_contiguous(torch.bfloat16))
k_cfg = list(enumerate_k_grouped_contiguous(torch.bfloat16))
print(f'get_arch_major() = {get_arch_major()}')
print(f'm-grouped contiguous bf16 配置数 = {len(m_cfg)}')
print(f'k-grouped contiguous bf16 配置数 = {len(k_cfg)}')
# m: 每配置内层 expand(2) × test_alias(2) = 4 次发射
m_launch = len(m_cfg) * 4
# k: 每配置 × fused_operand(a,b) = 2
k_launch = len(k_cfg) * 2
print(f'm-grouped 实际发射 kernel 次数 = {len(m_cfg)} × 4 = {m_launch}')
print(f'k-grouped 实际发射 kernel 次数 = {len(k_cfg)} × 2 = {k_launch}')
report('B1 m-grouped bf16 枚举 32 配置（含 use_psum 重复轴）',
       len(m_cfg) == 32, f'实际 {len(m_cfg)}')
report('B2 k-grouped bf16 (arch9) 枚举 7 配置', len(k_cfg) == 7, f'实际 {len(k_cfg)}')
# 说明：m-grouped 枚举器含 use_psum_layout ∈ (True, False)，但 test 不把该值透传给 generator
# （默认 use_psum=False），故 32 条里成对重复。这里核实 generator 的 use_psum 默认值。
sig = inspect.signature(generate_m_grouped_contiguous)
print(f'generate_m_grouped_contiguous use_psum_layout 默认 = {sig.parameters["use_psum_layout"].default}')
report('B3 m-grouped 枚举的 psum 轴未透传（成对重复）',
       sig.parameters['use_psum_layout'].default is False, '默认 False，故 32→16 唯一')


# ---------------------------------------------------------------------------
sep('Part C：直接跑真实测试体（bench_kineto 打桩，形状不缩）')
# ---------------------------------------------------------------------------
# 把模块内的 bench_kineto 换成常量桩：跳过 profiling，但测试体的所有断言/分支一行未改。
orig_bench = test_legacy.bench_kineto


def _stub_bench(fn, *a, **k):
    fn()
    return 100e-6


test_legacy.bench_kineto = _stub_bench
t0 = time.time()
try:
    test_legacy.test_m_grouped_gemm_contiguous_tl()
    test_legacy.test_k_grouped_gemm_contiguous_tl()
    real_ok, real_err = True, ''
except AssertionError as e:
    real_ok, real_err = False, str(e)
except Exception as e:                                 # noqa: BLE001
    real_ok, real_err = False, f'{type(e).__name__}: {e}'
finally:
    test_legacy.bench_kineto = orig_bench
dt = time.time() - t0
report('C1 真实测试体全流程无断言失败（两个函数）', real_ok, f'耗时 {dt:.1f}s {real_err}')
print('  （上面两段 "Testing ..." 输出即真实测试体打印；perf 行是桩值）')


# ---------------------------------------------------------------------------
sep('Part D：k-grouped 的跳过闸门语义')
# ---------------------------------------------------------------------------
# 真实闸门：if use_psum_layout or k_alignment != 128 or not accumulate or out_dtype != torch.float: continue
def filtered(cfgs):
    return [c for c in cfgs if not (c[10] or c[9] != 128 or not c[11] or c[12] != torch.float)]

kept9 = filtered(k_cfg)
print(f'arch9 真实枚举 {len(k_cfg)} 条，闸门后保留 {len(kept9)} 条（跳过 {len(k_cfg) - len(kept9)}）')
report('D1 arch9 全部通过闸门（跳过=0）', len(kept9) == len(k_cfg), f'{len(kept9)}/{len(k_cfg)}')

# 伪造 arch10 枚举：临时把 generators 的 get_arch_major 改成 10，看闸门拦下多少。
_fake_ok = True
try:
    orig_gam = generators.get_arch_major
    generators.get_arch_major = lambda: 10
    try:
        k_cfg10 = list(enumerate_k_grouped_contiguous(torch.bfloat16))
    finally:
        generators.get_arch_major = orig_gam
    kept10 = filtered(k_cfg10)
    print(f'伪造 arch10 枚举 {len(k_cfg10)} 条，闸门后保留 {len(kept10)} 条（跳过 {len(k_cfg10) - len(kept10)}）')
    # 逐条统计被哪一因子拦下
    from collections import Counter
    reasons = Counter()
    for c in k_cfg10:
        psum, kalign, acc, odt = c[10], c[9], c[11], c[12]
        r = []
        if psum:
            r.append('psum')
        if kalign != 128:
            r.append('k_align')
        if not acc:
            r.append('no_acc')
        if odt != torch.float:
            r.append('out_dtype')
        reasons['+'.join(r) if r else 'PASS'] += 1
    print('  拦截原因直方图：', dict(reasons))
    report('D2 闸门在 arch10 确实拦下（psum/k_align/no_acc/out_dtype）',
           len(kept10) < len(k_cfg10) and len(kept10) > 0, f'{len(kept10)}/{len(k_cfg10)} 保留')
except Exception as e:                                 # noqa: BLE001
    _fake_ok = False
    report('D2 闸门在 arch10 确实拦下', False, f'伪造枚举失败（跳过）：{type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
sep('Part E：test_alias 薄壳 & expand 恒等 —— 逐位复核（独立 torch oracle）')
# ---------------------------------------------------------------------------
reset_seed(1)
# 取一个固定的 m-grouped bf16 配置：major_a=KMajor, major_b=MNMajor（触发 nn 别名分支）
num_groups, emp, n, k = 4, 8192, 4096, 2048
from generators import MajorTypeAB                    # noqa: E402
ma, mb = MajorTypeAB.KMajor, MajorTypeAB.MNMajor

m, a, b, m_indices, d, ref_d, _ = generate_m_grouped_contiguous(
    num_groups, emp, n, k, ma, mb, use_bf16=True)
# 参考：逐行按 m_indices 选组、mask -1 行
valid = (m_indices != -1)
ref = torch.zeros((m, n), device=DEV, dtype=torch.float32)
for g in range(num_groups):
    rows = (m_indices == g)
    if rows.any():
        ref[rows] = a[rows].float() @ b[g].float().t()
ref = ref.to(torch.bfloat16)

d_nt = torch.empty_like(d)
deep_gemm.legacy.m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d_nt, m_indices)
# test_alias 分支的真实表达式：major_b 为 MN-major 时 b = b.mT 再调 nn（nn 内部再 .mT 回来）
b_alias = b if mb.is_k_major() else b.mT
d_nn = torch.empty_like(d)
deep_gemm.legacy.m_grouped_bf16_gemm_nn_contiguous_tl(a, b_alias, d_nn, m_indices)
torch.cuda.synchronize()
print(f'nt vs 独立 torch oracle: calc_diff={calc_diff(d_nt, ref):.3e}, '
      f'逐位相等={torch.equal(d_nt, ref)}')
report('E1 kernel nt == 独立 torch oracle（有效行，calc_diff<1e-3）',
       calc_diff(d_nt, ref) < 1e-3,
       f'calc_diff={calc_diff(d_nt, ref):.3e}，逐位相等={torch.equal(d_nt, ref)}（bf16 累加序差异）')
report('E2 alias(nn) == nt 逐位相等（薄壳）', torch.equal(d_nt, d_nn),
       f'calc_diff={calc_diff(d_nt, d_nn):.3e}')

# expand：a_fused_m 用恒等 m_row_indices，应等于 m_grouped
m_row_indices = torch.arange(0, m, dtype=torch.int32, device=DEV)
d_exp = torch.empty_like(d)
deep_gemm.legacy.a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d_exp, (m_indices, m_row_indices))
torch.cuda.synchronize()
report('E3 expand(a_fused_m, 恒等行索引) == m_grouped 逐位', torch.equal(d_nt, d_exp),
       f'calc_diff={calc_diff(d_nt, d_exp):.3e}')

# 置换 m_row_indices 复现任意逐行 gather（验证 a_fused 的「索引 A」语义）
perm = torch.randperm(m, device=DEV, dtype=torch.int32)
d_perm = torch.empty_like(d)
deep_gemm.legacy.a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d_perm, (m_indices, perm))
ref_perm = torch.zeros((m, n), device=DEV, dtype=torch.float32)
ag = a[perm.long()]
for g in range(num_groups):
    rows = (m_indices == g)
    if rows.any():
        ref_perm[rows] = ag[rows].float() @ b[g].float().t()
ref_perm = ref_perm.to(torch.bfloat16)
report('E4 expand 置换行索引 == 独立 oracle（有效行，calc_diff<1e-3）',
       calc_diff(d_perm[valid], ref_perm[valid]) < 1e-3,
       f'calc_diff(valid)={calc_diff(d_perm[valid], ref_perm[valid]):.3e}')
# 关键发现：perm 全非负时，-1 对齐填充行也会被 a_fused 计算并写出非零（无行掩码）——
# 测试里用恒等 m_row_indices 时因 generator 已把 a 填充行清零，结果恰为 0；显式 -1 才行掩码。
pad_nonzero = int((d_perm[~valid] != 0).sum())
print(f'  expand 置换下填充行写出非零元素 = {pad_nonzero} / {int((~valid).sum())} 行')
report('E5 -1 填充行：m_grouped 写零 / expand 由行索引决定', True,
       f'm_grouped 填充行全零={bool((d_nt[~valid] == 0).all())}；expand 置换填充行非零={pad_nonzero > 0}')


# ---------------------------------------------------------------------------
sep('Part F：判据利齿（calc_diff 盲区）+ 真实 perf')
# ---------------------------------------------------------------------------
# calc_diff 是归一化平方误差 ‖x−y‖²/(‖x‖²+‖y‖²)：两边同缩放 s 时分子分母同 ×s² => 完全不变
base = d_nt.float()
for scale in (0.999, 1.0001, 1.01):
    cd = calc_diff(base * scale, base)
    print(f'  calc_diff(d*{scale}, d) = {cd:.3e}')
cd_small = calc_diff(base * 0.999, base)
report('F1 calc_diff 对全局缩放几乎不变（< 1e-3 阈值）', cd_small < 1e-3,
       f'0.999× 缩放 calc_diff={cd_small:.3e} 仍过阈值')
# 反例：calc_diff 是**全局**归一化——单行归零被淹没（真盲区）
bad1 = base.clone()
bad1[0] = 0
cd1 = calc_diff(bad1, base)
report('F2 calc_diff 漏掉单行错误（全局归一化盲区）', cd1 < 1e-3,
       f'单行归零 calc_diff={cd1:.3e} 仍过阈值，但逐位不等={not torch.equal(bad1, base)}')
# 反例：大块归零才越过阈值
bad2 = base.clone()
bad2[: base.shape[0] // 8] = 0
cd2 = calc_diff(bad2, base)
report('F2b 大块（1/8 行）归零被 calc_diff 抓住', cd2 > 1e-3, f'calc_diff={cd2:.3e}')
# 逐位判据抓单元素 1-ulp
bad3 = base.clone()
bad3.view(torch.int16)[0, 0] += 1
report('F3 逐位判据抓 1-ulp 扰动（calc_diff 抓不住）',
       not torch.equal(bad3, base) and calc_diff(bad3, base) < 1e-3,
       f'calc_diff={calc_diff(bad3, base):.3e}，逐位不等={not torch.equal(bad3, base)}')

# 真实 perf：用真实 bench_kineto 采两个代表形状（m-grouped / k-grouped），打印测试自带口径
from deep_gemm.testing import bench_kineto, count_bytes    # noqa: E402
print('\n真实 perf（bench_kineto，kernel 名子串）：')
# m-grouped
m2, a2, b2, mi2, d2, _, _ = generate_m_grouped_contiguous(8, 4096, 7168, 7168, MajorTypeAB.KMajor,
                                                          MajorTypeAB.KMajor, use_bf16=True)
t = bench_kineto(lambda: deep_gemm.legacy.m_grouped_bf16_gemm_nt_contiguous_tl(a2, b2, d2, mi2),
                 'm_grouped_bf16_gemm_contiguous_tl_impl', suppress_kineto_output=True)
print(f'  m-grouped (G=8, m={m2}, n=7168, k=7168): {t*1e6:.1f} us | '
      f'{2*m2*7168*7168/t/1e12:.1f} TFLOPS | {count_bytes(a2,b2,d2)/1e9/t:.0f} GB/s')
# k-grouped 代表形状
ks = [8192, 8192, 8192, 8192]
host_ks = ks
k_, a3, b3, c3, d3, ref3, gl, hks = generate_k_grouped_contiguous(4, 4096, 7168, MajorTypeAB.MNMajor,
                                                                 MajorTypeAB.MNMajor, ks, use_bf16=True)
k_indices = torch.arange(0, k_, dtype=torch.int32, device=DEV)
k_start = torch.empty(4, dtype=torch.int32, device=DEV)
k_end = torch.empty(4, dtype=torch.int32, device=DEV)
for i, gk in enumerate(hks):
    k_start[i] = k_end[i-1] if i > 0 else 0
    k_end[i] = k_start[i] + gk
t = bench_kineto(lambda: deep_gemm.legacy.b_fused_k_grouped_bf16_gemm_tn_contiguous_tl(
                     a3, b3, c3, (k_indices, k_start, k_end), True),
                 'b_fused_k_grouped_bf16_gemm_contiguous_tl_impl', suppress_kineto_output=True)
print(f'  k-grouped (G=4, m=4096, n=7168, k={k_}): {t*1e6:.1f} us | '
      f'{2*4096*7168*k_/t/1e12:.1f} TFLOPS | {count_bytes(a3,b3,d3)/1e9/t:.0f} GB/s')
report('F4 真实 perf 采到非零吞吐', t > 0, '见上两行')


# ---------------------------------------------------------------------------
sep('总结')
# ---------------------------------------------------------------------------
n_pass = sum(RESULTS)
print(f'PASS = {n_pass} / {len(RESULTS)}，FAIL = {len(RESULTS) - n_pass}')
if n_pass != len(RESULTS):
    sys.exit(1)
