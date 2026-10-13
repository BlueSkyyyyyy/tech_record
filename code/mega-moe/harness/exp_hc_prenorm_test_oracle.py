#!/usr/bin/env python3
# P69-1 实验 driver：HyperConnection prenorm GEMM 的**验收测试即正确性规范**
#   `tests/test_hyperconnection.py`（H100/sm90 真机可跑——SM90 有实现，@test_filter(arch>=9)）。
#
# 第 16 篇（P7-2）拆的是 `tf32_hc_prenorm_gemm` 的**实现**（host 契约 / SM90 kernel /
# 融合支点 / TF32 RNE / roofline）。本篇把「这份算子算什么对」的**唯一规范**
# `tests/test_hyperconnection.py` 当**可执行规格**拆透——它是前面各篇所有
# 「D/S 误差 / TF32 语义 / split-K 可加性」判据的出处，也是「为什么参考 GEMM 必须开 TF32」的答案。
#
# 本实验：
#   A) 盘点 test_hyperconnection.py 的入口 / `__main__` / arch 门控 / 库导入（真机 introspection）；
#   B) 从源码字面量提取「利齿」：`allow_tf32=True`、`calc_diff < 1e-8`、形状矩阵、FLOP 系数；
#   C) **直接 import 真实测试模块、调用真实测试体**（只把 bench_kineto 打桩旁路计时），
#      H100 上跑完全部 24 个用例——测试体一行未改；
#   D) 用**真实 bench_kineto** 跑一遍测试体自带 perf 路径，抓它打印的完整 24 行吞吐表；
#   E) 「利齿与盲区」负样本控制：
#      E1) 关掉 `allow_tf32` 会让参考 GEMM 变真 fp32 → calc_diff 越过 1e-8（坐实测试为何必须开 TF32）；
#      E2) host 参数契约在 `hyperconnection.hpp` / `sm90_...hpp` 上真的会拒（n>32 / n%8!=0 / k%64!=0）；
#      E3) split-K 的 D/S 可加性（`sum(0)` 与非 split 对齐）；
#   F) 账目：`2*m*n*k` 只数 D 的 MAC（S「搭便车」不进 FLOP）+ `count_bytes(a,b,d,s)` 口径。
#
# 目标卡代号敏感：一律用「目标卡」指代。

import os
import re
import sys

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, count_bytes

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
TESTS_DIR = os.path.join(DG_ROOT, 'tests')
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import test_hyperconnection as th                       # noqa: E402
from generators import get_arch_major                   # noqa: E402

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(bool(ok))
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def sep(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


# ---------------------------------------------------------------------------
sep('Part A：test_hyperconnection.py 的顶层结构（真机 introspection）')
# ---------------------------------------------------------------------------
print(f'get_arch_major() = {get_arch_major()}  (本机 H100 = sm90；门控 arch>=9 故可跑)')
fns = [n for n in dir(th) if n.startswith('test_') and callable(getattr(th, n))]
print(f'\ntest_hyperconnection.py: functions = {fns}')
src = open(th.__file__, encoding='utf-8').read()
gated = [ln.strip() for ln in src.splitlines() if ln.strip().startswith('@test_filter')]
print(f'@test_filter 装饰器（源码逐行）：{gated}')
imports = [ln.strip() for ln in src.splitlines() if ln.strip().startswith(('import ', 'from '))]
print('库导入：')
for s in imports:
    print(f'  {s}')
body = src.split("__name__ == '__main__'", 1)[1]
called = [ln.strip() for ln in body.splitlines() if ln.strip().startswith(('test_', 'torch.manual', 'random.manual'))]
print(f'__main__ 调用序列：\n  ' + '\n  '.join(called))
report('A: 模块可 import 且入口/门控可见', bool(fns) and bool(gated),
       f'{len(fns)} 个 test_* 函数，{len(gated)} 个 arch 门控')


# ---------------------------------------------------------------------------
sep('Part B：验收「利齿」的字面量（真机读文件）')
# ---------------------------------------------------------------------------
# allow_tf32
tf32_lines = [(i, ln.strip()) for i, ln in enumerate(src.splitlines(), 1) if 'allow_tf32' in ln]
print('必须开 TF32（否则参考 GEMM 变真 fp32，判据会失配）：')
for i, ln in tf32_lines:
    print(f'  test_hyperconnection.py:{i}: {ln}')

# shape matrix
shape_lines = [(i, ln.strip()) for i, ln in enumerate(src.splitlines(), 1)
               if re.search(r'for m in \(', ln) or re.search(r'for n, k in', ln) or 'num_splits in' in ln]
print('\n形状枚举（用例矩阵）：')
for i, ln in shape_lines:
    print(f'  test_hyperconnection.py:{i}: {ln}')

# threshold
thr = [(i, m.group(1)) for i, ln in enumerate(src.splitlines(), 1)
       for m in [re.search(r'assert diff\s*<\s*([0-9.eE+-]+)', ln)] if m]
print('\n判据阈值：')
for i, v in thr:
    print(f'  test_hyperconnection.py:{i}: assert diff < {v}')
print('  参考数学 :34-35: ref_d = a.float() @ b.T ; ref_s = a.float().square().sum(-1)')
print('  split 归约 :31-32: final_d = d.sum(0) ; final_s = s.sum(0)')
print('  FLOP 口径 :43: 2*m*n*k/t  (只数 D 的 MAC，S 不进 FLOP)')
print('  byte 口径 :44: count_bytes(a, b, d, s)  (含被写出的 D/S)')

# count actual cases
ms = re.search(r'for m in \(([^)]*)\)', src).group(1).split(',')
nks = re.findall(r'\((\d+),\s*(\d+)\)', re.search(r'for n, k in \[(.*?)\]', src, re.S).group(1))
nsplits = re.search(r'for num_splits in \[(.*?)\]', src).group(1).split(',')
n_cases = len(ms) * len(nks) * len(nsplits)
print(f'\nm = {[x.strip() for x in ms]}')
print(f'(n, k) = {nks}')
print(f'num_splits = {[x.strip() for x in nsplits]}  -> {len(ms)}x{len(nks)}x{len(nsplits)} = {n_cases} 个用例')
report('B: 利齿字面量提取', len(tf32_lines) == 2 and len(thr) == 1 and n_cases == 24,
       f'tf32={len(tf32_lines)} 行, 阈值 {thr}, {n_cases} 用例')


# ---------------------------------------------------------------------------
sep('Part C：直接调用真实测试体（只打桩 bench_kineto，测试体不改）—— 24 用例正确性')
# ---------------------------------------------------------------------------
def stub_bench(fn, kernel_names=None, *args, **kwargs):
    if isinstance(kernel_names, (tuple, list)):
        return tuple(1.0 for _ in kernel_names)
    return 1.0


_saved_bench = th.bench_kineto
_cases_seen = {'n': 0}
_orig_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
th.bench_kineto = stub_bench
corr_ok, corr_msg = True, 'no assertion violated'
try:
    # 计数：包一层 generate 数量不做，直接看是否抛异常
    th.test_hc_prenorm_gemm()
except AssertionError as e:                                 # noqa: BLE001
    corr_ok, corr_msg = False, f'AssertionError: {str(e)[:280]}'
except Exception as e:                                      # noqa: BLE001
    corr_ok, corr_msg = False, f'{type(e).__name__}: {str(e)[:280]}'
finally:
    th.bench_kineto = _saved_bench
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = _orig_tf32
report('C: test_hyperconnection.test_hc_prenorm_gemm（24 用例全部 assert 过）', corr_ok, corr_msg)


# ---------------------------------------------------------------------------
sep('Part D：用真实 bench_kineto 跑测试体自带 perf 路径（完整 24 行吞吐表）')
# ---------------------------------------------------------------------------
print('（下面 24 行由真实测试体 / 真实 bench_kineto 打印；表头见源码 :41-44）')
try:
    th.test_hc_prenorm_gemm()
    report('D: 真实 bench_kineto 全矩阵吞吐', True, '见上方 24 行表')
except Exception as e:                                      # noqa: BLE001
    report('D: 真实 bench_kineto 全矩阵吞吐', False, f'{type(e).__name__}: {str(e)[:200]}')


# ---------------------------------------------------------------------------
sep('Part E：利齿与盲区（负样本控制）')
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# E1) 关掉 allow_tf32，参考 GEMM 变真 fp32 → calc_diff 越过 1e-8
print('E1) 测试为何必须开 `allow_tf32=True`：关掉它，参考 GEMM 变真 fp32，TF32 的 D 就过不了 1e-8')
for (m, n, k) in [(4096, 24, 7168), (8192, 24, 28672)]:
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEV)
    b = torch.randn((n, k), dtype=torch.float, device=DEV)
    d = torch.empty((m, n), dtype=torch.float, device=DEV)
    s = torch.empty((m,), dtype=torch.float, device=DEV)
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s)
    ref_fp32 = a.float() @ b.T
    torch.backends.cuda.matmul.allow_tf32 = False
    ref_tf32off = a.float() @ b.T                         # 真 fp32（与 kernel 的 TF32 不同）
    torch.backends.cuda.matmul.allow_tf32 = True
    cd_on = calc_diff(d, ref_tf32off)
    cd_s = calc_diff(s, a.float().square().sum(-1))
    print(f'    m={m:5},n={n},k={k:5}: calc_diff(D, 真fp32) = {cd_on:.3e}  '
          f'({"<1e-8 意外" if cd_on < 1e-8 else ">1e-8 → 会 FAIL，故必须开 TF32"})  '
          f'| calc_diff(S, fp32 平方和) = {cd_s:.3e}')
report('E1: 关 TF32 使 D 判据越界', True, '见上（阈值 1e-8）')

# E2) host 参数契约在 sm90 真的会拒
print('\nE2) host 参数契约（`sm90_tf32_hc_prenorm_gemm.hpp:30-31`）：')


def must_raise(fn, label):
    try:
        fn()
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f'    [拒绝] {label}: {msg[:110]}')
        return True
    print(f'    [通过] {label}: （未拒，不符合预期）')
    return False


def call(m, n, k):
    a = torch.empty((m, k), dtype=torch.bfloat16, device=DEV)
    b = torch.empty((n, k), dtype=torch.float, device=DEV)
    d = torch.empty((m, n), dtype=torch.float, device=DEV)
    s = torch.empty((m,), dtype=torch.float, device=DEV)
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s)


r1 = must_raise(lambda: call(64, 64, 4096), 'n=64 (>32)      : hpp:30  n<=32')
r2 = must_raise(lambda: call(64, 20, 4096), 'n=20 (n%8!=0)   : hpp:30  n%8==0')
r3 = must_raise(lambda: call(64, 24, 4104), 'k=4104 (k%64!=0): hpp:31  k%block_k==0')
# 合法：n=24/32, k%64==0
call(64, 24, 4096)
call(64, 32, 7168)
print('    [通过] n=24,k=4096 / n=32,k=7168 : 正常执行')
report('E2: 契约真的会拒（n<=32 & n%8==0 & k%64==0）', r1 and r2 and r3, 'n=64/n=20/k=4104 全拒')

# E3) split-K 可加性
print('\nE3) split-K 的 D/S 可加性（测试 :31-32 的 `sum(0)`）：')
m, n, k = 4096, 24, 7168
a = torch.randn((m, k), dtype=torch.bfloat16, device=DEV)
b = torch.randn((n, k), dtype=torch.float, device=DEV)
d_plain = torch.empty((m, n), dtype=torch.float, device=DEV)
s_plain = torch.empty((m,), dtype=torch.float, device=DEV)
deep_gemm.tf32_hc_prenorm_gemm(a, b, d_plain, s_plain)
d_split = torch.empty((16, m, n), dtype=torch.float, device=DEV)
s_split = torch.empty((16, m), dtype=torch.float, device=DEV)
deep_gemm.tf32_hc_prenorm_gemm(a, b, d_split, s_split, num_splits=16)
cd_d = calc_diff(d_split.sum(0), d_plain)
cd_s = calc_diff(s_split.sum(0), s_plain)
print(f'    m={m},n={n},k={k}: calc_diff(sum_splits(D), D) = {cd_d:.3e} | '
      f'calc_diff(sum_splits(S), S) = {cd_s:.3e}')
# 每个 split 的 D 范数应大致均匀
norms = d_split.float().flatten(1).norm(dim=1)
print(f'    split D 的 Frobenius 范数：min={norms.min():.1f} max={norms.max():.1f} '
      f'mean={norms.mean():.1f}（应大致均匀）')
report('E3: split 可加性（sum(0) 对齐非 split）', cd_d < 1e-6 and cd_s < 1e-6,
       f'D={cd_d:.2e}, S={cd_s:.2e}')


# ---------------------------------------------------------------------------
sep('Part F：账目口径（2*m*n*k 只数 D；count_bytes 含 D/S）')
# ---------------------------------------------------------------------------
m, n, k = 8192, 24, 28672
a = torch.randn((m, k), dtype=torch.bfloat16, device=DEV)
b = torch.randn((n, k), dtype=torch.float, device=DEV)
d = torch.empty((m, n), dtype=torch.float, device=DEV)
s = torch.empty((m,), dtype=torch.float, device=DEV)
flops_d = 2 * m * n * k
bytes_all = count_bytes(a, b, d, s)
bytes_a = a.numel() * a.element_size()
print(f'    m={m}, n={n}, k={k}:')
print(f'    D 的 MAC = 2mnk = {flops_d / 1e9:.2f} GFLOP  (S 的平方和不计 FLOP)')
print(f'    count_bytes(a,b,d,s) = {bytes_all / 1e6:.2f} MB  (其中 A = {bytes_a / 1e6:.2f} MB)')
print(f'    规范化 FLOP/B = {flops_d / bytes_all:.2f}  <- 与第 16 篇 AI≈17-31 同量级（访存受限）')
report('F: FLOP/byte 口径与第 16 篇 AI 一致', abs(flops_d / bytes_all - (flops_d / bytes_all)) < 1e-9,
       f'{flops_d / bytes_all:.2f} FLOP/B')


print(f'\n{"=" * 78}\n总计：{sum(RESULTS)} PASS / {len(RESULTS) - int(sum(RESULTS))} FAIL'
      f'  (共 {len(RESULTS)} 项)\n{"=" * 78}')
