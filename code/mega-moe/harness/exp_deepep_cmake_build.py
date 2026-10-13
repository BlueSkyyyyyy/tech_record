#!/usr/bin/env python3
"""P119-1 实验：在 kernel_lab(H100/sm90) 上真跑 DeepEP 的 CMake 调试构建链。

对照两组构建路径：
  A. `setup.py`（真装包路径，torch CUDAExtension，见第 124 篇）——本脚本不重跑，只做静态对照。
  B. `CMakeLists.txt`（自称 only for debugging）——本脚本真 configure / build / 读产物。

用法（在仓库根）：
  bash harness/exp_deepep_cmake_build.sh > harness/exp_deepep_cmake_build.out.txt

本脚本在容器内运行，纯构建 / 文件解析，不加载 GPU。
"""
import os
import re
import shutil
import subprocess
import sys
import json

DE = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
BUILD = os.environ.get('DEEPEP_CMAKE_BUILD', '/tmp/deepep-cmake-build-debug')
VERBOSE_LOG = '/tmp/deepep_cmake_verbose.log'


def run(cmd, cwd=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                          capture_output=True, text=True, timeout=timeout)


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def main():
    section('0. 环境坐标')
    for cmd in (['cmake', '--version'], ['nvcc', '--version']):
        r = run(cmd)
        print('$ ' + ' '.join(cmd) + ' | head -n1 ->', r.stdout.splitlines()[0] if r.stdout else r.stderr.splitlines()[0])
    r = run(['python', '-c', 'import torch;print(torch.__version__)'])
    print('torch', r.stdout.strip())
    r = run(['python', '-c', 'import torch;print(torch.utils.cmake_prefix_path)'])
    torch_cmake = r.stdout.strip()
    print('torch cmake prefix', torch_cmake)
    r = run('git -c safe.directory="*" log --oneline -1', cwd=DE)
    print('DeepEP', r.stdout.strip())

    # NCCL root：复用 DeepEP 自己的 find_pkgs（第 124 篇）
    find_pkgs_path = os.path.join(DE, 'deep_ep', 'utils', 'find_pkgs.py')
    g = {}
    exec(open(find_pkgs_path).read(), g)
    nccl_root = g['find_nccl_root']()
    print('find_nccl_root() ->', nccl_root)

    section('1. CMake configure（cmake-build-debug 约定）')
    if os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    cfg = ['cmake', '-S', DE, '-B', BUILD,
           f'-DCMAKE_PREFIX_PATH={torch_cmake}',
           f'-DNCCL_ROOT_DIR={nccl_root}']
    r = run(cfg)
    print('configure rc =', r.returncode)
    print('\n'.join((r.stdout + r.stderr).splitlines()[-8:]))
    assert r.returncode == 0, 'configure failed'
    print('[OK] configure：需显式 -DNCCL_ROOT_DIR（env 变量不算，CMakeLists:25-27 判的是 CMake 变量）')

    section('2. 目标图（cmake --build --target help）')
    r = run(['cmake', '--build', BUILD, '--target', 'help'])
    tgts = [l.split('... ')[-1].strip() for l in r.stdout.splitlines() if l.strip().startswith('...')]
    print('targets:', ', '.join(tgts[:16]))
    for key in ('_C', 'runtime_comm', 'runtime_driver', 'deep_ep_indexing_cuda'):
        print(f'  {"有" if key in tgts else "无"}  {key}')

    section('3. 编译两个 host runtime 静态库 + _C 扩展')
    # runtime_driver 首次编译，抓它用哪个编译器（g++ 而非 nvcc）
    r = run(['cmake', '--build', BUILD, '--target', 'runtime_driver', '-j4', '--verbose'])
    driver_log = r.stdout + r.stderr
    print(f'[build {"runtime_driver":18s}] {"OK" if r.returncode == 0 else "FAIL"}')
    gpp = re.findall(r'(/usr/bin/c\+\+[^\n]*driver\.cpp)', driver_log)
    print('  host .cpp 用 CXX 编译器（g++）编：', '是' if gpp else '否')
    for tgt in ('runtime_comm', '_C'):
        r = run(['cmake', '--build', BUILD, '--target', tgt, '-j4'])
        status = 'OK' if r.returncode == 0 else f'FAIL(rc={r.returncode})'
        print(f'[build {tgt:18s}] {status}')

    section('4. 编译 IDE 索引 TU（会失败——上游真实编译错误）')
    r = run(['cmake', '--build', BUILD, '--target', 'deep_ep_indexing_cuda', '-j4', '--verbose'])
    indexing_rc = r.returncode
    with open(VERBOSE_LOG, 'w') as f:
        f.write(r.stdout + r.stderr)
    print('build deep_ep_indexing_cuda rc =', indexing_rc)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if re.search(r'error:|\d+ errors? detected', l)]
    for l in errs[:8]:
        print('  ' + l.strip())

    section('5. 关键编译命令行（nvcc 诊断 flag 只作用于 .cu）')
    log = open(VERBOSE_LOG).read()
    nvcc = [l for l in log.splitlines() if '/nvcc ' in l and ' -M ' not in l]
    if nvcc:
        line = nvcc[0]
        for frag in ('-DENABLE_FAST_DEBUG', '--ptxas-options=--verbose,--register-usage-level=10,'
                     '--warn-on-local-memory-usage', '-lineinfo', '-Xcompiler=-rdynamic',
                     'compute_90,code=sm_90'):
            print(f'  {"命中" if frag in line else "未出现"}  {frag}')
    # 找 g++ 编译 driver.cpp 的行（已在 §3 抓）

    section('6. 产物')
    art = []
    for root, _, files in os.walk(BUILD):
        for fn in files:
            if fn.endswith('.so') or fn.endswith('.a'):
                p = os.path.join(root, fn)
                art.append((os.path.relpath(p, BUILD), os.path.getsize(p)))
    for p, sz in sorted(art):
        print(f'  {sz:>10d} B  {p}')
    r = run('nm -D %s | grep -i PyInit' % os.path.join(BUILD, '_C.cpython-312-x86_64-linux-gnu.so'))
    print('  _C 导出 init 符号：', r.stdout.strip())

    section('7. `cmake-build-*` 命名约定（两处 ignore）')
    gi = open(os.path.join(DE, '.gitignore')).read()
    print('  .gitignore 含 */cmake-build-*/ :', '*/cmake-build-*/' in gi)
    sp = open(os.path.join(DE, 'setup.py')).read()
    m = re.search(r"ignore_patterns\(([^)]*)\)", sp)
    print('  setup.py ignore_patterns：', m.group(1) if m else '(none)')

    section('8. DeepGEMM 对偶（同一套调试 CMake）')
    dg = os.environ.get('DEEPGEMM_ROOT', '~/github/DeepGEMM')
    dgcm = open(os.path.join(dg, 'CMakeLists.txt')).read()
    print('  DeepGEMM/CMakeLists.txt 首行注释：', dgcm.splitlines()[0].strip())
    print('  DeepGEMM 有 runtime_comm/driver 静态库：', 'runtime_' in dgcm)
    print('  DeepGEMM 有 NCCL 依赖：', 'NCCL' in dgcm)

    section('9. 汇总 JSON')
    print(json.dumps({
        'nccl_root': nccl_root,
        'configure_rc': 0,
        'artifacts': {p: sz for p, sz in art},
        'indexing_tu_rc': indexing_rc,
    }, indent=2))


if __name__ == '__main__':
    main()
