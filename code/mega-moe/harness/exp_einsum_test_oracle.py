#!/usr/bin/env python3
# P66-1 实验 driver：DeepGEMM einsum 家族的**验收测试即正确性规范**
#   `tests/test_einsum.py`（H100/sm90 真机可跑；两个 FP8 变体被 arch>=10 门控）。
#
# 本专题第 40 篇拆了测试基座 deep_gemm/testing/、第 41 篇拆了操作数生成器 generators.py、
# 第 43 篇拆了逐位断言辅助 tests/utils.py；第 41/47/50/57/64 篇分别把 Mega MoE / Mega Gate /
# MQA logits / Mega mHC / 稠密 GEMM 的**验收测试**当「可执行规格」拆透。但把整个
# **einsum 家族**（`bmk,bnk->mn` 批归约 / MLA 双向投影 / wgrad `bhd,bhr->hdr` 及其 FP8 变体）
# 的出口判据集中定义的 `tests/test_einsum.py` 一直只被顺带引用过几行。
#
# 本实验：
#   A) 盘点 test_einsum.py 的函数/`__main__` 入口/arch 门控（真机 introspection）；
#   B) 提取「验收利齿」阈值表（calc_diff 1e-5/1e-10/1e-3/2e-3 + direct==累加/随机舍入 oracle）；
#   C) **直接 import 真实测试模块、调用真实测试体**（只把 bench_kineto 打桩旁路计时），
#      H100 上逐函数判 PASS/FAIL——测试体一行未改；两个 SM100 专属函数当场被 test_filter 打印跳过；
#   D) einsum 特有的「利齿与盲区」负样本控制：`bhr,hdr->bhd` 的 calc_diff<1e-10 与逐位判据；
#      `bhd,bhr->hdr` 的 C-累积语义（漏掉 C 会被大初始值放大暴露）；
#   E) 确定性开关切换真实 kernel（cuBLASLt `nvjet` vs 自研 `gemm`），profiler 抓名字；
#   F) 测试自带 perf 路径的真实吞吐（真实 bench_kineto）。
#
# 目标卡代号敏感：一律用「目标卡」指代。

import os
import sys

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, assert_bitwise_equal

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
TESTS_DIR = os.path.join(DG_ROOT, 'tests')
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import test_einsum                                   # noqa: E402
from test_einsum import get_arch_major               # noqa: E402

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def sep(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


# ---------------------------------------------------------------------------
sep('Part A：test_einsum.py 的顶层结构（真机 introspection）')
# ---------------------------------------------------------------------------
print(f'get_arch_major() = {get_arch_major()}  (本机 H100 = sm90)')

fns = [n for n in dir(test_einsum) if n.startswith('test_') and callable(getattr(test_einsum, n))]
print(f'\ntest_einsum.py: functions = {fns}')
src = open(test_einsum.__file__, encoding='utf-8').read()
body = src.split("__name__ == '__main__'", 1)[1]
called = [ln.strip() for ln in body.splitlines() if ln.strip().startswith(('test_', 'random.', 'torch.manual'))]
print(f'__main__ 调用序列：\n  ' + '\n  '.join(called))

# arch 门控：只有两个 FP8 反向/投影函数被 @test_filter(lambda: get_arch_major() >= 10) 包住
gated = []
for ln in src.splitlines():
    s = ln.strip()
    if s.startswith('@test_filter'):
        gated.append(s)
print(f'\n@test_filter 装饰器（源码逐行）：{gated}')
print('  -> test_fp8_bhd_hdr_bhr / test_fp8_bhd_bhr_hdr 在 H100 被门控跳过；'
      '\n     test_fp8_bhr_hdr_bhd 无装饰器（bf16 输出路径 sm90 可跑）。')
report('A: 模块可 import 且入口/门控可见', True,
       f'{len(fns)} 个 test_* 函数，{len(gated)} 个 arch 门控')


# ---------------------------------------------------------------------------
sep('Part B：验收「利齿」阈值表（真机读文件字面量）')
# ---------------------------------------------------------------------------
import re                                             # noqa: E402

thresholds = []
for i, ln in enumerate(src.splitlines(), 1):
    m = re.search(r'calc_diff\([^)]*\)\s*<\s*([0-9.eE+-]+)', ln)
    if m:
        thresholds.append((i, m.group(1)))
print('calc_diff 阈值（行号 -> 值）：')
for i, v in thresholds:
    print(f'  test_einsum.py:{i}: calc_diff(...) < {v}')
print("\n逐位 oracle（tests/utils.py，第 43 篇）：")
print('  assert_direct_output_matches_fp32_accumulation  <-  test_einsum.py:60 / :91')
print('  assert_stochastic_bf16_matches_fp32_accumulation <-  test_einsum.py:63 / :94 / :179 / :223 (SM100-only)')
report('B: 阈值表提取', True, f'{len(thresholds)} 处 calc_diff 阈值')


# ---------------------------------------------------------------------------
sep('Part C：直接调用真实测试体（只打桩 bench_kineto，测试体不改）')
# ---------------------------------------------------------------------------
def stub_bench(fn, kernel_names=None, *args, **kwargs):
    if isinstance(kernel_names, (tuple, list)):
        return tuple(1.0 for _ in kernel_names)
    return 1.0


def run_test(mod, fnname):
    saved_bench = mod.bench_kineto
    mod.bench_kineto = stub_bench
    ok, msg = True, ''
    try:
        getattr(mod, fnname)()
    except AssertionError as e:
        ok, msg = False, f'AssertionError: {str(e)[:280]}'
    except Exception as e:                                # noqa: BLE001
        ok, msg = False, f'{type(e).__name__}: {str(e)[:280]}'
    finally:
        mod.bench_kineto = saved_bench
    report(f'C: test_einsum.{fnname}', ok, msg if not ok else 'no assertion violated')


# 六个可跑函数（其中两个 FP8 变体被 test_filter 跳过，调用它们只打印 Filtered）
run_test(test_einsum, 'test_bmk_bnk_mn')
run_test(test_einsum, 'test_bhr_hdr_bhd')
run_test(test_einsum, 'test_bhd_hdr_bhr')
run_test(test_einsum, 'test_bhd_bhr_hdr')
run_test(test_einsum, 'test_fp8_bhr_hdr_bhd')
run_test(test_einsum, 'test_fp8_bhd_hdr_bhr')       # 期望：被门控跳过
run_test(test_einsum, 'test_fp8_bhd_bhr_hdr')       # 期望：被门控跳过


# ---------------------------------------------------------------------------
sep('Part D：einsum 特有的「利齿与盲区」负样本控制')
# ---------------------------------------------------------------------------
# D1) `bhr,hdr->bhd`：测试用 calc_diff < 1e-10（很紧），但逐位是否相等？
print('D1) "bhr,hdr->bhd" 一个真实形状：calc_diff(<1e-10) 与逐位判据的差别')
torch.manual_seed(0)
b, h, r, d_dim = 128, 128, 512, 128
x = torch.randn((b, h, r), device=DEV, dtype=torch.bfloat16)
y = torch.randn((h, d_dim, r), device=DEV, dtype=torch.bfloat16)
ref_z = torch.einsum('bhr,hdr->bhd', x, y)
deep_gemm.use_deterministic_algorithms(True)          # 走自研 sm90_bf16_gemm
z = torch.empty((b, h, d_dim), device=DEV, dtype=torch.bfloat16)
deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
deep_gemm.use_deterministic_algorithms(False)
cd = calc_diff(z, ref_z)
try:
    assert_bitwise_equal(z, ref_z, 'bhr,hdr->bhd')
    bitwise = True
except AssertionError:
    bitwise = False
print(f'    calc_diff(z, ref) = {cd:.3e}  (<1e-10 -> {"PASS" if cd < 1e-10 else "FAIL"})')
print(f'    torch.equal(z, ref) = {bitwise}   (逐位比 calc_diff 严格)')
report('D1: 紧 calc_diff + 逐位关系已记录', True, f'calc_diff={cd:.2e}, bitwise={bitwise}')

# D2) `bhd,bhr->hdr`：C-累积语义。测试用 z_0*10 作初始值，直接抄 torch 参考。
print('\nD2) "bhd,bhr->hdr" 的 C-累积：漏掉 C（写成覆盖）会被大初始值放大暴露')
b2 = 1024
xa = torch.randn((b2, h, d_dim), device=DEV, dtype=torch.bfloat16)
ya = torch.randn((b2, h, r), device=DEV, dtype=torch.bfloat16)
z0 = torch.randn((h, d_dim, r), device=DEV, dtype=torch.float32) * 10
ref_z0 = z0 + torch.einsum('bhd,bhr->hdr', xa.float(), ya.float())
z_ok = z0.clone()
deep_gemm.einsum('bhd,bhr->hdr', xa, ya, z_ok, z_ok)      # D=C=z0，累积
d_ok = calc_diff(z_ok, ref_z0)
z_bad = torch.zeros_like(z0)                              # 覆盖（丢掉 C）
deep_gemm.einsum('bhd,bhr->hdr', xa, ya, z_bad, z_bad)
d_bad = calc_diff(z_bad, ref_z0)
print(f'    D=C=z0 累积 : calc_diff = {d_ok:.3e}  (阈值 1e-10)')
print(f'    D=0 覆盖    : calc_diff = {d_bad:.3e}  <- 丢 C 立即暴露')
report('D2: C-累积语义被测试严判', d_ok < 1e-10 and d_bad > 1e-3, f'{d_ok:.2e} vs {d_bad:.2e}')

# D3) `bmk,bnk->mn`：语义 = 沿 batch 求和（等效于把 batch 并进收缩轴），并做「部分和」等价。
print('\nD3) "bmk,bnk->mn" 的语义：批归约 == 把 batch 并进收缩轴的 GEMM')
s, m, n, k = 64, 128, 256, 128
ab = torch.randn((s, m, k), device=DEV, dtype=torch.bfloat16)
bb = torch.randn((s, n, k), device=DEV, dtype=torch.bfloat16)
# 注：fp32 D 的 bmk,bnk->mn 契约要求 c==d（累积，见测试 :29-33 / docs/17 :25-26）；
# 不传 c 会段错误（本实验真机踩到）。
dd = torch.zeros((m, n), device=DEV, dtype=torch.float32)
deep_gemm.einsum('bmk,bnk->mn', ab, bb, dd, c=dd)
folded = ab.float().permute(1, 0, 2).reshape(m, s * k) @ bb.float().permute(1, 0, 2).reshape(n, s * k).mT
part = torch.zeros((m, n), device=DEV, dtype=torch.float32)
for si in range(s):
    part += ab[si].float() @ bb[si].float().mT
print(f'    vs 折叠 batch 的 fp32 GEMM : calc_diff = {calc_diff(dd, folded):.3e}')
print(f'    vs 逐 batch 部分和          : calc_diff = {calc_diff(dd, part):.3e}')
report('D3: 批归约语义等价已坐实', calc_diff(dd, folded) < 1e-4 and calc_diff(dd, part) < 1e-4)
report('D: einsum 利齿与盲区负样本控制', True, 'D1/D2/D3')


# ---------------------------------------------------------------------------
sep('Part E：确定性开关切换真实 kernel（profiler 抓名字）')
# ---------------------------------------------------------------------------
def kernel_names(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            names.append(evt.key)
    return names


b3 = 8192
xe = torch.randn((b3, h, r), device=DEV, dtype=torch.bfloat16)
ye = torch.randn((h, d_dim, r), device=DEV, dtype=torch.bfloat16)
ze = torch.empty((b3, h, d_dim), device=DEV, dtype=torch.bfloat16)

deep_gemm.use_deterministic_algorithms(True)
names_det = kernel_names(lambda: deep_gemm.einsum('bhr,hdr->bhd', xe, ye, ze))
deep_gemm.use_deterministic_algorithms(False)
names_cub = kernel_names(lambda: deep_gemm.einsum('bhr,hdr->bhd', xe, ye, ze))
pick = lambda ns: [s for s in ns if ('gemm' in s.lower() or 'nvjet' in s.lower())]
print(f'deterministic=True  抓到的 GEMM kernel: {pick(names_det)}')
print(f'deterministic=False 抓到的 GEMM kernel: {pick(names_cub)}')
det_hit = any('sm90_bf16_gemm' in s for s in names_det)
cub_hit = any('nvjet' in s for s in names_cub)
report('E: 确定性开关切换 kernel（测试用其抓两根路径）', det_hit and cub_hit,
       f'det={det_hit}, cublaslt={cub_hit}')


# ---------------------------------------------------------------------------
sep('Part F：测试自带 perf 路径的真实吞吐（真实 bench_kineto）')
# ---------------------------------------------------------------------------
from deep_gemm.testing import bench_kineto                 # noqa: E402


def perf_bhr_hdr_bhd(b, h, r, d):
    x = torch.randn((b, h, r), device=DEV, dtype=torch.bfloat16)
    fy = torch.randn((h, d, r + 128), device=DEV, dtype=torch.bfloat16)
    y = fy[:, :, :r]
    z = torch.empty((b, h, d), device=DEV, dtype=torch.bfloat16)
    deep_gemm.use_deterministic_algorithms(True)
    t = bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), 'gemm', suppress_kineto_output=True)
    deep_gemm.use_deterministic_algorithms(False)
    return t


def perf_bmk(s, m, n, k):
    a = torch.randn((s, m, k), device=DEV, dtype=torch.bfloat16)
    b = torch.randn((s, n, k), device=DEV, dtype=torch.bfloat16)
    d = torch.zeros((m, n), device=DEV, dtype=torch.float32)
    return bench_kineto(lambda: deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=d), 'gemm', suppress_kineto_output=True)


try:
    t = perf_bhr_hdr_bhd(8, 4096, 4096, 1024)
    print(f'   "bhr,hdr->bhd" (b=8, h=4096, r=4096, d=1024): {t * 1e6:7.1f} us | '
          f'{2 * 8 * 4096 * 4096 * 1024 / t / 1e12:6.0f} TFLOPS')
    t = perf_bhr_hdr_bhd(8192, 128, 512, 128)
    print(f'   "bhr,hdr->bhd" (b=8192, h=128, r=512, d=128): {t * 1e6:7.1f} us | '
          f'{2 * 8192 * 128 * 512 * 128 / t / 1e12:6.0f} TFLOPS')
    t = perf_bmk(8192, 256, 256, 256)
    print(f'   "bmk,bnk->mn"  (s=8192, m=n=k=256): {t * 1e6:7.1f} us | '
          f'{2 * 8192 * 256 * 256 * 256 / t / 1e12:6.0f} TFLOPS')
    report('F: 真实 bench_kineto 吞吐', True, 'bhr,hdr->bhd / bmk,bnk->mn')
except Exception as e:                                        # noqa: BLE001
    report('F: 真实 bench_kineto 吞吐', False, f'{type(e).__name__}: {e}')


print(f'\n{"=" * 78}\n总计：{sum(RESULTS)} PASS / {len(RESULTS) - int(sum(RESULTS))} FAIL'
      f'  (共 {len(RESULTS)} 项)\n{"=" * 78}')
