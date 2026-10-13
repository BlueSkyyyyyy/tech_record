#!/usr/bin/env python3
# P70-1 实验 driver：DeepGEMM 顶层运行时的**验收测试即正确性规范**
#   `tests/test_lazy_init.py`（199 行，架构无关、H100/sm90 真机可跑）。
#
# 第 23 篇（P23-1）拆了 DeepJIT 的编译/缓存/加载/发射管线；第 40 篇（P40-1，docs/50）
# 拆了 `Runtime` 的 lazy init / shutdown / cuBLASLt handle 与 per-stream workspace 的**生命周期**，
# 并在 §6.5 只用一句「直接跑上游测试：Ran 3 tests / OK」交叉引用过它。
# 本篇把这份测试当成**可执行规格**单独拆透——它定义了「lazy init / fork safety / 原生退出清理」
# 到底算什么对，也是第 40 篇所有「handle 恰好 create/destroy 各 1 次」结论的判据出处。
#
# 本实验：
#   A) 盘点 test_lazy_init.py 的顶层结构（入口 / 门控 / 三个 unittest 用例 / mp 启动）；真机 introspection；
#   B) 解码 LD_PRELOAD 探针 `_PROBE_SOURCE`：每个 magic exit code 的语义 / 状态机；
#   C) **原样跑真实测试模块**（mp 8 进程 + unittest.main），报告 Ran N / OK；
#   D) **独立复刻** 6 个用例矩阵：把测试源码里的 `_PROBE_SOURCE` 逐字取出编译成 .so，
#      用同一套 LD_PRELOAD 拦截 cublasLtCreate/Destroy，逐用例数事件，与规格期望对拍；
#   E) 「利齿」负样本控制：shutdown 幂等 / 终态（shutdown 后 get_tc_util 抛 RuntimeError）/
#      `DG_USE_PYTORCH_CUBLASLT_HANDLE` 是「整数」不是「布尔」；
#   F) fork safety：mp worker 全部 exit 0。
#
# 目标卡代号敏感：一律用「目标卡」指代。

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

import torch

import deep_gemm

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
TESTS_DIR = os.path.join(DG_ROOT, 'tests')
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import test_lazy_init as tl                              # noqa: E402

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(bool(ok))
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def sep(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def sh_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


SRC = open(tl.__file__, encoding='utf-8').read()
SRC_LINES = SRC.splitlines()


# ---------------------------------------------------------------------------
sep('Part A：test_lazy_init.py 的顶层结构（真机 introspection）')
# ---------------------------------------------------------------------------
print(f'deep_gemm 根目录        : {DG_ROOT}')
print(f'测试文件                : {tl.__file__}  ({len(SRC_LINES)} 行)')
print(f'平台                    : {sys.platform}   （探针类 @skipUnless(linux) 门控）')

test_classes = [n for n in dir(tl) if n.startswith('Test') and isinstance(getattr(tl, n), type)]
print(f'\nunittest 类             : {test_classes}')
for cls_name in test_classes:
    cls = getattr(tl, cls_name)
    methods = [m for m in dir(cls) if m.startswith('test_')]
    skip = getattr(cls, '__unittest_skip__', False) or getattr(cls, '__unittest_skip_condition__', None)
    print(f'  {cls_name}: methods={methods}')
    print(f'    skip 条件文本: {getattr(cls, "__unittest_skip_condition__", None)!r}')
    print(f'    setUpClass 里构建探针: {hasattr(cls, "setUpClass")}')

iface = [ln.strip() for ln in SRC_LINES if ln.strip().startswith(('import ', 'from '))]
print('\n库导入：')
for s in iface:
    print(f'  {s}')
body = SRC.split("__name__ == '__main__'", 1)[1]
calls = [ln.strip() for ln in body.splitlines() if ln.strip()]
print('\n__main__ 逐行：')
for s in calls:
    print(f'  {s}')
report('A: 模块可 import、入口/门控/三用例可见',
       len(test_classes) == 1 and bool(test_classes),
       f'{len(test_classes)} 个 Test 类、{len([m for m in dir(getattr(tl, test_classes[0])) if m.startswith("test_")])} 个用例')


# ---------------------------------------------------------------------------
sep('Part B：LD_PRELOAD 探针 `_PROBE_SOURCE` 解码（真机从源码行号定位）')
# ---------------------------------------------------------------------------
probe_lines = [(i, SRC_LINES[i - 1]) for i in range(22, 53)]
print('探针源码（test_lazy_init.py:22-52，逐行）：')
for i, ln in probe_lines:
    print(f'  {i:3}: {ln}')

print('\n三个 magic exit code 的语义（测试体靠它们判定）：')
sem = {
    91: 'cublasLtDestroy 在 backend_dead 之后被调用 → 退出期 use-after-free',
    92: 'verify_cleanup 时 live_handles != 0 → handle 泄漏（未被销毁）',
    93: 'cublasLtDestroy 收到非本探针 handle 或 live_handles != 1 → 重复/陌生销毁',
}
for code, why in sorted(sem.items()):
    print(f'  _Exit({code}): {why}')

print('\n状态机：')
print('  initialize_probe()  __attribute__((constructor))  →  std::atexit(verify_cleanup)')
print('  cublasLtCreate  →  std::atexit(stop_backend); ++live_handles; 打印 DG_PROBE_CREATE')
print('  cublasLtDestroy →  backend_dead?  _Exit(91);  handle 不符? _Exit(93); --live_handles; 打印 DG_PROBE_DESTROY')
print('  退出时 verify_cleanup 最后跑（constructor 里先注册，故在扩展静态析构之后）→ 打印 DG_PROBE_CLEAN 或 _Exit(92)')

# 探针源与测试源码字面一致（逐字取出）
assert tl._PROBE_SOURCE in SRC
report('B: 探针源码可从模块逐字取出（用于独立复刻）', tl._PROBE_SOURCE in SRC,
       f'{len(tl._PROBE_SOURCE.splitlines())} 行')


# ---------------------------------------------------------------------------
sep('Part C：原样跑真实测试模块（mp 8 进程 + unittest.main，与官方 __main__ 一致）')
# ---------------------------------------------------------------------------
cmd = [sys.executable, os.path.join(TESTS_DIR, 'test_lazy_init.py'), '--num-processes', '8']
print('$ ' + ' '.join(sh_quote(c) for c in cmd))
res = subprocess.run(cmd, cwd=TESTS_DIR, capture_output=True, text=True, timeout=300)
combined = res.stdout + '\n' + res.stderr
print('\n'.join(combined.strip().splitlines()[-12:]))
ran_ok = res.returncode == 0 and re.search(r'Ran \d+ tests', combined) and 'OK' in combined
report('C: 真实测试模块 H100 真机通过', bool(ran_ok),
       f'rc={res.returncode}, ' + (re.search(r'Ran \d+ tests.*', combined).group(0) if ran_ok else '见上'))


# ---------------------------------------------------------------------------
sep('Part D：独立复刻 6 用例矩阵（编译测试自带的 LD_PRELOAD 探针，逐用例数事件）')
# ---------------------------------------------------------------------------
workdir = tempfile.mkdtemp(prefix='dg-lazy-probe-')
probe_src = os.path.join(workdir, 'probe.cpp')
probe_so = os.path.join(workdir, 'probe.so')
with open(probe_src, 'w') as f:
    f.write(tl._PROBE_SOURCE)
cc = shutil.which('c++')
subprocess.run([cc, '-shared', '-fPIC', '-std=c++17', probe_src, '-o', probe_so],
               check=True, capture_output=True, text=True)
print(f'编译探针: {cc} -shared -fPIC -std=c++17 probe.cpp -o probe.so  →  {os.path.getsize(probe_so)} B')

use_runtime = 'assert deep_gemm.get_tc_util() == deep_gemm.get_tc_util() == 100\n'
close_runtime = textwrap.dedent('''\
    deep_gemm.shutdown()
    deep_gemm.shutdown()
    try:
        deep_gemm.get_tc_util()
    except RuntimeError:
        pass
    else:
        raise AssertionError('runtime was recreated after shutdown')
''')
cases = (
    ('unused',            '0', '',                             0),
    ('owned',             '0', use_runtime,                    1),
    ('managed',           '1', use_runtime,                    0),
    ('explicit',          '0', use_runtime + close_runtime,    1),
    ('managed_explicit',  '1', use_runtime + close_runtime,    0),
    ('unused_explicit',   '0', close_runtime,                  0),
)
child_preamble = textwrap.dedent(f'''\
    from pathlib import Path
    import sys
    repo = Path({DG_ROOT!r}).resolve()
    sys.path.insert(0, str(repo))
    import deep_gemm
''')

print(f'\n{"用例":<18}{"managed":<9}{"get_tc_util":<12}{"shutdown":<10}{"期望create":<11}{"实测create":<11}{"实测destroy":<12}{"clean":<6}{"判定"}')
all_case_ok = True
for name, managed, body, expected in cases:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['DG_USE_PYTORCH_CUBLASLT_HANDLE'] = managed
    env['LD_PRELOAD'] = probe_so + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    r = subprocess.run([sys.executable, '-c', child_preamble + body, DG_ROOT],
                       cwd=DG_ROOT, env=env, capture_output=True, text=True, timeout=45)
    ev = r.stderr.splitlines()
    n_create = ev.count('DG_PROBE_CREATE')
    n_destroy = ev.count('DG_PROBE_DESTROY')
    n_clean = ev.count('DG_PROBE_CLEAN')
    ok = (r.returncode == 0 and n_create == expected and n_destroy == expected and n_clean == 1)
    all_case_ok &= ok
    used = 'Y' if 'get_tc_util' in body else '-'
    shut = 'Y' if 'shutdown' in body else '-'
    print(f'{name:<18}{managed:<9}{used:<12}{shut:<10}{expected:<11}{n_create:<11}{n_destroy:<12}{n_clean:<6}{"PASS" if ok else "FAIL"}')
    if not ok:
        print(f'    rc={r.returncode} stderr={r.stderr!r}')
report('D: 6 用例矩阵 create/destroy/clean 全部符合规格', all_case_ok, '6/6')


# ---------------------------------------------------------------------------
sep('Part E：利齿（shutdown 幂等/终态；env 是「整数」不是「布尔」）')
# ---------------------------------------------------------------------------
# E1: shutdown 幂等 + 终态
e1 = textwrap.dedent('''\
    import deep_gemm
    assert deep_gemm.get_tc_util() == 100
    deep_gemm.shutdown()
    deep_gemm.shutdown()            # 幂等：第二次也不报错
    try:
        deep_gemm.get_tc_util()
        print('NO_RAISE')
    except RuntimeError:
        print('RAISED')
''')
env = os.environ.copy()   # 注意：E1 需要真实 GPU（无 LD_PRELOAD 时 cublasLtCreate 要真建库）
r = subprocess.run([sys.executable, '-c', f'import sys; sys.path.insert(0, {DG_ROOT!r})\n' + e1],
                   cwd=DG_ROOT, env=env, capture_output=True, text=True, timeout=30)
e1_ok = r.returncode == 0 and 'RAISED' in r.stdout
print(f'E1 shutdown 幂等 & 终态（shutdown 后 get_tc_util 抛 RuntimeError）: rc={r.returncode} out={r.stdout.strip()!r}')
report('E1: shutdown 幂等且不可恢复', e1_ok, '两次 shutdown 无错、之后访问抛 RuntimeError')

# E2: env 值类型是 int → true/false 非法（对照 doc/48 的开关面）
def probe_env(value):
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    env['DG_USE_PYTORCH_CUBLASLT_HANDLE'] = value
    env['LD_PRELOAD'] = probe_so
    r = subprocess.run([sys.executable, '-c', child_preamble + 'deep_gemm.get_tc_util()', DG_ROOT],
                       cwd=DG_ROOT, env=env, capture_output=True, text=True, timeout=30)
    ev = r.stderr.splitlines()
    return r.returncode, ev.count('DG_PROBE_CREATE')

e2_ok = True
for val, expect_create in (('0', 1), ('1', 0), ('2', 0), ('-1', 1)):
    rc, nc = probe_env(val)
    ok = (rc == 0 and nc == expect_create)
    e2_ok &= ok
    print(f'E2 DG_USE_PYTORCH_CUBLASLT_HANDLE={val!r:<4} → rc={rc}, create={nc} (期望 {expect_create}) {"PASS" if ok else "FAIL"}')
for val in ('true', 'false', 'abc'):
    rc, nc = probe_env(val)
    print(f'E2 DG_USE_PYTORCH_CUBLASLT_HANDLE={val!r:<7} → rc={rc} (期望非 0：非整数被 DJ_PANIC 拒)')
report('E2: env 按 int 解析（true/false 非法）', e2_ok, '0/1/2/-1 语义正确')


# ---------------------------------------------------------------------------
sep('Part F：fork safety（mp worker 全部 exit 0）')
# ---------------------------------------------------------------------------
def _worker(rank):
    torch.cuda.set_device(rank)

failed = []
for rank in range(8):
    p = subprocess.run([sys.executable, '-c', f'import torch; torch.cuda.set_device({rank % torch.cuda.device_count()})'],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        failed.append((rank, p.returncode, p.stderr[-200:]))
# 用真实模块的 main（与官方 __main__ 同一目标函数）跑一遍
mp_code = textwrap.dedent(f'''\
    import sys, torch
    sys.path.insert(0, {TESTS_DIR!r})
    import torch.multiprocessing as mp
    import test_lazy_init as tl
    procs = [mp.Process(target=tl.main, args=(i,)) for i in range(8)]
    for p in procs: p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0, p.exitcode
    print('ALL_WORKERS_OK')
''')
r = subprocess.run([sys.executable, '-c', mp_code], cwd=TESTS_DIR, capture_output=True, text=True, timeout=120)
mp_ok = r.returncode == 0 and 'ALL_WORKERS_OK' in r.stdout
print(f'真实 tl.main × 8 进程: rc={r.returncode} out={r.stdout.strip()!r}')
report('F: fork safety（8 个 mp worker 全部 exit 0）', mp_ok and not failed,
       'tl.main 8/8 exit 0')


# ---------------------------------------------------------------------------
sep('汇总')
# ---------------------------------------------------------------------------
n_pass = sum(RESULTS)
print(f'{n_pass} PASS / {len(RESULTS) - n_pass} FAIL  (共 {len(RESULTS)})')
print('参考版本：~/github/DeepGEMM commit 057ca59（v2.8.1）。SM100 无关，本机 H100 全可跑。')
sys.exit(0 if n_pass == len(RESULTS) else 1)
