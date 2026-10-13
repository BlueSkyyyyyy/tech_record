#!/usr/bin/env python3
# P60-1 实验 driver：DeepGEMM 的内存安全验证层 tests/test_sanitizer.py（compute-sanitizer 驱动）。
#   对应 docs/70-deepgemm-sanitizer.md；参考 ~/github/DeepGEMM/tests/test_sanitizer.py。
#   bash harness/exp_sanitizer.sh > harness/exp_sanitizer.out.txt
#
# 内容：
#   A) 静态核对真实 test_sanitizer.py 的 env/flag 清单 + 自动发现普查
#   B) 负样本控制 src/sanitizer_negctl.cu：memcheck / synccheck / racecheck
#   C) 真实 test_sanitizer.py 驱动（H100 可跑的两个测试函数）
#   D) 缓存分配器屏蔽效应（PYTORCH_NO_CUDA_MEMORY_CACHING 的“为什么”）
#   E) DG_USE_NVIDIA_TOOLS 旁路 bench_kineto
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _find_deepgemm_tests():
    cands = [os.environ.get('DEEPGEMM_TESTS', ''),
             os.path.expanduser('~/github/DeepGEMM/tests')]
    try:
        import deep_gemm
        pkg = os.path.dirname(os.path.abspath(deep_gemm.__file__))
        cands.append(os.path.join(os.path.dirname(pkg), 'tests'))
    except Exception:
        pass
    for c in cands:
        if c and os.path.isfile(os.path.join(c, 'test_sanitizer.py')):
            return c
    raise SystemExit(f'cannot locate DeepGEMM tests, tried: {cands}')


DEEPGEMM_TESTS = _find_deepgemm_tests()
SANITIZER = '/usr/local/cuda/bin/compute-sanitizer'
BIN = '/tmp/opencode/p60_san_negctl'

PASS = 0
FAIL = 0


def check(name, cond, extra=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name} {extra}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {extra}')


def run(cmd, env=None, cwd=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# ---------------------------------------------------------------------------
print('=' * 70)
print('A) 静态核对真实 tests/test_sanitizer.py')
print('=' * 70)
src = open(os.path.join(DEEPGEMM_TESTS, 'test_sanitizer.py')).read()

env_expect = {
    'CUDA_LAUNCH_BLOCKING': '1',
    'DG_JIT_CHECK_NO_SPILLS': '1',
    'DG_USE_NVIDIA_TOOLS': '1',
    'DG_USE_TEMP_CUBLASLT_WORKSPACE': '1',
    'PYTORCH_NO_CUDA_MEMORY_CACHING': '1',
    'TORCH_SHOW_CPP_STACKTRACES': '1',
}
env_found = dict(re.findall(r"env\['([A-Z_]+)'\] = '([^']*)'", src))
print('  env 清单:')
for k, v in env_found.items():
    print(f'    {k} = {v}')
check('env 集合与预期一致', env_found == env_expect, f'({len(env_found)} 项)')

for flag in ['--target-processes=application-only', '--destroy-on-device-error=context',
             '--force-blocking-launches', '--check-api-memory-access=no']:
    check(f'flag {flag} 存在', flag in src)
check('--kernel-name-exclude kns=nvjet 存在', "'--kernel-name-exclude', 'kns=nvjet'" in src)
check("默认 tools = memcheck,synccheck", "default='memcheck,synccheck'" in src)
check('默认 funcs = all', "default='all'" in src)

excl = re.search(r"exclude_files = \[([^\]]*)\]", src).group(1)
print(f'  exclude_files = [{excl}]')
for name in ['test_sanitizer.py', 'generators.py', 'test_mega_moe.py', 'test_locality_domain.py']:
    check(f'exclude {name}', name in excl)

# 自动发现普查（复刻 test_sanitizer.py:42-49 的逻辑，注意 import 可能失败）
print('\n  自动发现普查（复刻 test_sanitizer.py:42-49）:')
import importlib
import inspect
files = [f for f in os.listdir(DEEPGEMM_TESTS) if f.endswith('.py')]
exclude_files = ['test_sanitizer.py', 'generators.py', 'test_mega_moe.py', 'test_locality_domain.py']
discovered = []
for f in sorted(files):
    if f in exclude_files:
        continue
    mod = os.path.splitext(f)[0]
    sys.path.insert(0, DEEPGEMM_TESTS)
    try:
        m = importlib.import_module(mod)
    except Exception as e:  # noqa
        print(f'    {mod:22} import 失败: {type(e).__name__}: {str(e)[:60]}')
        continue
    fns = [n for n, o in inspect.getmembers(m)
           if inspect.isfunction(o) and n.startswith('test') and 'test_filter' not in n]
    print(f'    {mod:22} -> {fns}')
    discovered += [f'{mod}.{n}' for n in fns]
check('自动发现 >= 20 个测试函数', len(discovered) >= 20, f'({len(discovered)})')
check('test_filter 装饰函数仍在列表里（运行期由装饰器跳过）',
      'test_einsum.test_fp8_bhd_hdr_bhr' in discovered)

# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('B) 负样本控制 src/sanitizer_negctl.cu')
print('=' * 70)
os.makedirs('/tmp/opencode', exist_ok=True)
rc, out = run(['nvcc', '-arch=sm_90', '-O2', '-o', BIN,
               os.path.join(ROOT, 'src', 'sanitizer_negctl.cu')])
check('negctl 编译成功', rc == 0, '' if rc == 0 else out[-300:])

def parse_errors(tool, out):
    if tool == 'racecheck':
        m = re.search(r'RACECHECK SUMMARY:\s*(\d+)\s*hazard', out)
    else:
        m = re.search(r'ERROR SUMMARY:\s*(\d+)\s*error', out)
    return int(m.group(1)) if m else None


# (tool, mode, expect_zero, 期望出现的错误关键字)
cases = [
    ('memcheck', 'clean', True, ''),
    ('memcheck', 'oob', False, 'Invalid __global__ write'),
    ('memcheck', 'misalign', False, 'misaligned'),
    ('memcheck', 'mask_big', True, ''),        # 越界落进同段 -> 被屏蔽
    ('memcheck', 'mask_exact', False, 'out of bounds'),  # 精确分配越界 -> 命中
    ('synccheck', 'clean', True, ''),
    ('synccheck', 'sync', False, 'Barrier error'),      # 发散 barrier
    ('racecheck', 'race', False, 'Race reported'),      # 注：test_sanitizer 并不跑 racecheck
]
for tool, mode, expect_zero, kw in cases:
    rc, out = run([SANITIZER, f'--tool={tool}', '--force-blocking-launches', BIN, mode])
    got = parse_errors(tool, out)
    ok = (got == 0) if expect_zero else (got is not None and got > 0)
    if kw:
        ok = ok and (kw in out)
    check(f'{tool:9} {mode:10} ' + ('errors==0' if expect_zero else 'errors>0'),
          ok, f'(got={got}){(" [" + kw + "]") if kw else ""}')

# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('C) 真实 test_sanitizer.py 驱动（H100 可跑的测试函数）')
print('=' * 70)
funcs = ['test_hyperconnection.test_hc_prenorm_gemm', 'test_layout.test_sf_layout_kernels']
summary = {}
for fn in funcs:
    for tool in ('memcheck', 'synccheck'):
        rc, out = run(['python', 'test_sanitizer.py', '--funcs', fn, '--tools', tool],
                      cwd=DEEPGEMM_TESTS)
        errs = re.findall(r'ERROR SUMMARY:\s*(\d+)\s*error', out)
        n = sum(int(x) for x in errs)
        summary[(fn, tool)] = (rc, n)
        raw = re.findall(r'=+ ERROR SUMMARY:.*', out)
        print(f'    {fn} [{tool}] 原始: {raw[0].strip() if raw else "(no summary)"}')
        check(f'{fn} [{tool}] rc==0 且 0 errors', rc == 0 and n == 0,
              f'(rc={rc}, errors={n}, runs={len(errs)})')

# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('D) 缓存分配器屏蔽效应（PYTORCH_NO_CUDA_MEMORY_CACHING 的“为什么”）')
print('=' * 70)
gap_prog = r'''
import torch
n = 1 << 20
x = torch.empty(n, dtype=torch.float32, device='cuda')
y = torch.empty(n, dtype=torch.float32, device='cuda')
gap = y.data_ptr() - x.data_ptr()
alloc = torch.cuda.memory_allocated()
resv = torch.cuda.memory_reserved()
print(f"gap={gap} size_bytes={n*4} allocated={alloc} reserved={resv} "
      f"segment_gt_tensor={resv > alloc}")
'''
prog_out = {}
for label, nocache in [('caching=ON ', 0), ('caching=OFF', 1)]:
    rc, out = run(['python', '-c', gap_prog],
                  env={'PYTORCH_NO_CUDA_MEMORY_CACHING': str(nocache)})
    line = [l for l in out.splitlines() if 'gap=' in l]
    prog_out[label] = line[0] if line else out[-200:]
    print(f'  {label}: {prog_out[label]}')
# 缓存分配器：从大 segment 里切块，reserved >> allocated；关闭后每次 cudaMalloc，reserved==0
check('caching=ON：reserved > allocated（大 segment 复用）',
      'segment_gt_tensor=True' in prog_out['caching=ON '])
check('caching=OFF：reserved == 0（每 tensor 独立 cudaMalloc）',
      'reserved=0' in prog_out['caching=OFF'])

# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('E) DG_USE_NVIDIA_TOOLS 旁路 bench_kineto')
print('=' * 70)
bypass_prog = r'''
from deep_gemm.testing import bench_kineto
t = bench_kineto(lambda: None, 'dummy', suppress_kineto_output=True)
print('bench_kineto returned:', repr(t), type(t).__name__)
'''
rc, out = run(['python', '-c', bypass_prog], env={'DG_USE_NVIDIA_TOOLS': '1'})
line = [l for l in out.splitlines() if 'bench_kineto returned' in l]
print('  ' + (line[0] if line else out[-200:]))
check('DG_USE_NVIDIA_TOOLS=1 时 bench_kineto 直接返回 1', 'returned: 1' in (line[0] if line else ''))

print('\n' + '=' * 70)
print(f'总判定: {PASS} PASS / {FAIL} FAIL')
print('=' * 70)
sys.exit(1 if FAIL else 0)
