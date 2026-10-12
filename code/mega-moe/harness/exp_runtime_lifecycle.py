"""P40-1 实验：DeepGEMM 运行时对象（`Runtime`）的生命周期与资源所有权。

对照源码（~/github/DeepGEMM @ 057ca59 / v2.8.1）：
  - csrc/runtime/runtime.hpp        （Runtime 构造/析构、cuBLASLt handle、per-stream workspace）
  - csrc/apis/config.hpp:12-14      （shutdown() 语义）
  - deep_gemm/__init__.py:121-125   （init + atexit(shutdown)）
  - third-party/deep_jit/include/deep_jit/utils/lazy.hpp:15-38（LazyInit）
  - third-party/deep_jit/include/deep_jit/utils/env.hpp:28-54（get_env<int>）
  - csrc/jit_kernels/impls/smxx_cublaslt.hpp:73-77,95-105（handle + workspace 消费点）
  - tests/test_lazy_init.py         （上游寿命/析构测试，作为交叉验证）

本实验在 H100 (sm90) 上真跑，全部与 UMMA/TMA/WGMMA 无关：
  A. LD_PRELOAD 探针统计 cublasLtCreate/Destroy/Clean（6 组寿命用例）
  B. shutdown() 终态语义 + init 不可恢复
  C. 每 stream 一份 workspace 的真实显存账（held vs temp）
  D. DG_USE_PYTORCH_CUBLASLT_HANDLE 的整数解析（含非法值报错）
  E. 交叉验证：跑上游 tests/test_lazy_init.py
"""
import os
import subprocess
import sys
import tempfile
import textwrap

# 必须在 import deep_gemm 之前设：Runtime 单例在首次访问时读取该环境变量。
os.environ.setdefault('DG_USE_TEMP_CUBLASLT_WORKSPACE', '0')
os.environ.setdefault('DG_USE_PYTORCH_CUBLASLT_HANDLE', '0')

DEEPGEMM_DIR = os.environ.get('DEEPGEMM_DIR', os.path.expanduser('~/github/DeepGEMM'))
DEEPGEMM_DIR = os.path.realpath(
    os.path.expanduser('~/github/DeepGEMM') if not os.path.isdir(DEEPGEMM_DIR) else DEEPGEMM_DIR)
if not os.path.isdir(DEEPGEMM_DIR):
    DEEPGEMM_DIR = '~/github/DeepGEMM'

# --- 复用 tests/test_lazy_init.py 的 LD_PRELOAD 探针源（逐行照抄，见该文件 :22-52）---
PROBE_SOURCE = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static uint32_t live_handles = 0, handle_token = 0;
static bool backend_dead = false;
static void verify_cleanup() {
    if (live_handles != 0) std::_Exit(92);
    std::fputs("DG_PROBE_CLEAN\n", stderr);
}
// Registered before the extension's static destructors, so this runs last.
__attribute__((constructor)) static void initialize_probe() {
    std::atexit(verify_cleanup);
}
static void stop_backend() { backend_dead = true; }
extern "C" uint32_t cublasLtCreate(void** handle) {
    std::atexit(stop_backend);
    *handle = &handle_token;
    ++live_handles;
    std::fputs("DG_PROBE_CREATE\n", stderr);
    return 0;
}
extern "C" uint32_t cublasLtDestroy(void* handle) {
    if (backend_dead) std::_Exit(91);
    if (handle != &handle_token or live_handles != 1) std::_Exit(93);
    --live_handles;
    std::fputs("DG_PROBE_DESTROY\n", stderr);
    return 0;
}
'''

CHILD_PREAMBLE = textwrap.dedent('''\
    from pathlib import Path
    import sys
    repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repo))
    import deep_gemm
    assert Path(deep_gemm._C.__file__).absolute().parent == repo / 'deep_gemm'
''')

USE_RUNTIME = 'assert deep_gemm.get_tc_util() == deep_gemm.get_tc_util() == 100\n'
CLOSE_RUNTIME = textwrap.dedent('''\
    deep_gemm.shutdown()
    deep_gemm.shutdown()
    try:
        deep_gemm.get_tc_util()
    except RuntimeError:
        pass
    else:
        raise AssertionError('runtime was recreated after shutdown')
''')

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"PASS" if cond else "FAIL"}] {name}  {detail}')


def compile_probe():
    d = tempfile.mkdtemp(prefix='dg-rtlife-')
    src = os.path.join(d, 'probe.cpp')
    so = os.path.join(d, 'probe.so')
    with open(src, 'w') as f:
        f.write(PROBE_SOURCE)
    subprocess.run(['c++', '-shared', '-fPIC', '-std=c++17', src, '-o', so],
                   check=True, capture_output=True, text=True, timeout=60)
    return so


def run_child(probe, body, env_extra):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.update(env_extra)
    if probe:
        env['LD_PRELOAD'] = probe + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    r = subprocess.run([sys.executable, '-c', CHILD_PREAMBLE + body, DEEPGEMM_DIR],
                       cwd=DEEPGEMM_DIR, env=env, capture_output=True, text=True, timeout=60)
    lines = r.stderr.splitlines()
    return {
        'rc': r.returncode,
        'create': lines.count('DG_PROBE_CREATE'),
        'destroy': lines.count('DG_PROBE_DESTROY'),
        'clean': lines.count('DG_PROBE_CLEAN'),
        'stderr_tail': '\n'.join(lines[-4:]),
    }


def part_a(probe):
    print('\n===== Part A：cuBLASLt handle 所有权（探针统计 create/destroy/clean）=====')
    cases = [
        ('unused',          '0', '', 0),
        ('owned',           '0', USE_RUNTIME, 1),
        ('managed',         '1', USE_RUNTIME, 0),
        ('explicit',        '0', USE_RUNTIME + CLOSE_RUNTIME, 1),
        ('managed_explicit','1', USE_RUNTIME + CLOSE_RUNTIME, 0),
        ('unused_explicit', '0', CLOSE_RUNTIME, 0),
    ]
    for name, managed, body, expect in cases:
        r = run_child(probe, body, {'DG_USE_PYTORCH_CUBLASLT_HANDLE': managed})
        ok = (r['rc'] == 0 and r['create'] == expect and r['destroy'] == expect
              and r['clean'] == 1)
        check(f'{name}: managed={managed} create/destroy/clean={r["create"]}/{r["destroy"]}/{r["clean"]} '
              f'(期望 {expect}/{expect}/1) rc={r["rc"]}', ok, '' if ok else r['stderr_tail'])


def part_b():
    print('\n===== Part B：shutdown() 终态语义 + init 不可恢复 =====')
    body = textwrap.dedent('''\
        import deep_gemm
        print('B before:', deep_gemm.get_tc_util())
        deep_gemm.shutdown(); deep_gemm.shutdown()
        errs = {}
        for fn in ('get_tc_util', 'get_num_sms'):
            try:
                getattr(deep_gemm, fn)()
                errs[fn] = 'NO_ERROR'
            except RuntimeError as e:
                errs[fn] = str(e).splitlines()[0]
        deep_gemm._C.init(str(Path(deep_gemm.__file__).resolve().parent))
        try:
            deep_gemm.get_tc_util(); after = 'RECOVERED'
        except RuntimeError as e:
            after = str(e).splitlines()[0]
        print('B get_tc_util after shutdown:', errs['get_tc_util'])
        print('B get_num_sms  after shutdown:', errs['get_num_sms'])
        print('B after _C.init:', after)
    ''')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['DG_USE_PYTORCH_CUBLASLT_HANDLE'] = '1'
    r = subprocess.run([sys.executable, '-c', CHILD_PREAMBLE + body, DEEPGEMM_DIR],
                       cwd=DEEPGEMM_DIR, env=env, capture_output=True, text=True, timeout=60)
    out = r.stdout
    print(out.strip())
    check('shutdown 后 get_tc_util 抛 lazy.hpp:27 "factory != null"',
          'lazy.hpp:27' in out and 'factory != null' in out)
    check('shutdown 后 get_num_sms 抛同一错误', out.count('factory != null') >= 2)
    check('重复 shutdown 幂等（无异常）', 'Traceback' not in out)
    check('_C.init 不重建 runtime（仍报错）',
          'B after _C.init:' in out and 'RECOVERED' not in out)


def part_c():
    print('\n===== Part C：每 stream 一份 workspace 的显存账（H100）=====')
    child = textwrap.dedent('''\
        import torch, deep_gemm
        a = torch.ones((32, 64), device='cuda', dtype=torch.bfloat16)
        b = torch.ones((48, 64), device='cuda', dtype=torch.bfloat16)
        o = torch.empty((32, 48), device='cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        mib = lambda: round((torch.cuda.memory_allocated() - base) / 2 ** 20, 2)
        deep_gemm.cublaslt_gemm_nt(a, b, o); torch.cuda.synchronize()
        s0 = mib()
        deep_gemm.cublaslt_gemm_nt(a, b, o); torch.cuda.synchronize()
        s0b = mib()
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            deep_gemm.cublaslt_gemm_nt(a, b, o)
        torch.cuda.synchronize()
        s1 = mib()
        ok = torch.equal(o, torch.full_like(o, 64))
        print(f'RESULT {s0} {s0b} {s1} {int(ok)}')
    ''')
    for mode, env_val in (('held', '0'), ('temp', '1')):
        env = os.environ.copy()
        env['DG_USE_TEMP_CUBLASLT_WORKSPACE'] = env_val
        out = subprocess.run([sys.executable, '-c', CHILD_PREAMBLE + child, DEEPGEMM_DIR],
                             cwd=DEEPGEMM_DIR, env=env, capture_output=True, text=True, timeout=120).stdout
        line = [x for x in out.splitlines() if x.startswith('RESULT')]
        if not line:
            check(f'{mode}: 无结果', False, out.strip()[-200:])
            continue
        s0, s0b, s1, ok = line[-1].split()[1:]
        print(f'  {mode}: stream0#1={s0} MiB, stream0#2={s0b} MiB, stream1={s1} MiB, out_ok={ok}')
        if mode == 'held':
            check('held：首用 +16MiB（一份 workspace）', abs(float(s0) - 16.0) < 1e-6, f'={s0}')
            check('held：同 stream 复用不增长', abs(float(s0b) - 16.0) < 1e-6, f'={s0b}')
            check('held：换 stream 再 +16MiB（每 stream 一份）', abs(float(s1) - 32.0) < 1e-6, f'={s1}')
        else:
            check('temp：不持有 workspace（持续占用 0）', abs(float(s1)) < 1e-6, f'={s1}')
        check(f'{mode}：cublaslt_gemm_nt 结果正确', ok == '1')


def part_d(probe):
    print('\n===== Part D：DG_USE_PYTORCH_CUBLASLT_HANDLE 的整数解析 =====')
    # value -> (期望 create 次数, 是否应报错)
    cases = [('0', 1, False), ('1', 0, False), ('2', 0, False), ('-1', 1, False),
             ('false', None, True), ('true', None, True), ('abc', None, True)]
    for val, expect_create, expect_err in cases:
        r = run_child(probe, USE_RUNTIME, {'DG_USE_PYTORCH_CUBLASLT_HANDLE': val})
        errored = r['rc'] != 0 or 'Panic error' in r['stderr_tail'] or 'invalid value' in r['stderr_tail']
        if expect_err:
            check(f'"{val}"：非整数被拒（DJ_PANIC）', errored,
                  '' if errored else f'unexpectedly ok, create={r["create"]}')
        else:
            check(f'"{val}"：create={r["create"]}（期望 {expect_create}）',
                  (not errored) and r['create'] == expect_create)


def part_e():
    print('\n===== Part E：交叉验证上游 tests/test_lazy_init.py =====')
    r = subprocess.run([sys.executable, 'tests/test_lazy_init.py'], cwd=DEEPGEMM_DIR,
                       capture_output=True, text=True, timeout=300)
    tail = '\n'.join((r.stdout + r.stderr).splitlines()[-6:])
    print(tail)
    check('上游寿命测试 3 tests OK', r.returncode == 0 and 'OK' in tail)


def main():
    import torch
    print('=' * 72)
    print('DeepGEMM 运行时生命周期实验 (P40-1)')
    print(f'  DeepGEMM dir : {DEEPGEMM_DIR}')
    print(f'  torch        : {torch.__version__}  cuda={torch.version.cuda}')
    print(f'  GPU          : {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}')
    try:
        import deep_gemm
        print(f'  deep_gemm    : {deep_gemm.__version__}')
    except Exception as e:
        print(f'  deep_gemm import 失败: {e}')
    print('=' * 72)

    probe = compile_probe()
    part_a(probe)
    part_b()
    part_c()
    part_d(probe)
    part_e()

    print('\n' + '=' * 72)
    print(f'SUMMARY: {len(PASS)} PASS / {len(FAIL)} FAIL')
    if FAIL:
        print('FAILED:', FAIL)
    print('=' * 72)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
