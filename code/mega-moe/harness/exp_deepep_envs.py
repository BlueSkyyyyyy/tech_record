"""P107-1 实验：DeepEP 的环境变量 / 开关面。

目标源码（`~/github/DeepEP@93eb6eb`，`deep_ep.__version__ == 2.5.0`）：
  * 运行时 `EP_*`：散落在 csrc/{buffers,runtime,kernels,utils} 与 deep_ep/{__init__,buffers,comm,utils}
  * 安装期固化的 persistent envs：setup.py `persistent_env_names`（:14-15）-> 生成的 `deep_ep/envs.py`
  * 解析器：`csrc/utils/system.hpp::get_env`（由 harness/deepep_env_probe.cpp 真机编译验证）
  * 引导闸门：`deep_ep/__init__.py::check_nccl_so:19-41`（`EP_SUPPRESS_NCCL_CHECK`）

本脚本（纯 Python / torch，架构无关）：
  A. 复用真实生成的 `deep_ep/envs.py` + 复刻 `__init__.py:8-16` 的「只补不覆盖」语义；
  B. 真跑 `check_nccl_so` 的 `EP_SUPPRESS_NCCL_CHECK` 闸门（短路 / 未抑制时抛 AssertionError）；
  C. **全库 `EP_*` 清单普查**（对两个仓库的源文件正则扫描，给出唯一变量集合与消费点计数）；
  D. 两个「跨语言/默认值」实证：`EP_BUFFER_DEBUG` 的 C++ int 解析 vs Python 真值、
     `EP_NUM_MAX_LOCAL_RANKS` 默认 16 与 engram 的 `NCCL_WIN_STRIDE` 副作用。

运行：bash harness/exp_deepep_envs.sh
"""
import ast
import hashlib
import importlib.util
import os
import re
import subprocess
import sys

import torch  # noqa: F401  —— 让 torch 把它的 libnccl 载入 /proc/self/maps，忠实复现 check_nccl_so 场景

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
DEG = os.environ.get('DEEPGEMM_ROOT', '~/github/DeepGEMM')
if not os.path.isdir(DEG):
    DEG = os.path.expanduser('~/github/DeepGEMM')

FAILS = []


def check(label, cond, extra=''):
    tag = '[PASS]' if cond else '[FAIL]'
    if not cond:
        FAILS.append(label)
    print(f'  {tag} {label} {extra}')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def read(path):
    with open(path, 'r') as f:
        return f.read()


# --------------------------------------------------------------------------- #
# A. persistent envs：真实生成的 deep_ep/envs.py + __init__.py:8-16 的应用语义
# --------------------------------------------------------------------------- #
print('== Part A: 安装期 persistent envs ==')
setup_src = read(os.path.join(DEP, 'setup.py'))
persistent_names = re.search(r'persistent_env_names\s*=\s*\((.*?)\)', setup_src, re.S).group(1)
persistent_names = re.findall(r"'([^']+)'", persistent_names)
print('  setup.py persistent_env_names:', persistent_names)
check('persistent_env_names 恰 7 条', len(persistent_names) == 7, str(persistent_names))

# 找真实生成的 envs.py（pip install 时的 build 目录）
cands = []
for root, _dirs, files in os.walk(os.path.join(DEP, 'build')):
    if 'envs.py' in files and os.path.basename(root) == 'deep_ep':
        cands.append(os.path.join(root, 'envs.py'))
cands.sort()
if not cands:
    print('  [SKIP] 未找到生成的 deep_ep/envs.py（未 pip install？）')
else:
    envs_py = cands[-1]
    print(f'  真实生成物: {envs_py}  sha256={sha256(envs_py)[:16]}')
    print('  ' + read(envs_py).replace('\n', '\n  ').rstrip())
    spec = importlib.util.spec_from_file_location('deep_ep_generated_envs', envs_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    baked = dict(mod.persistent_envs)
    check('persistent_envs 是 dict', isinstance(baked, dict), str(baked))
    check('固化键都来自 persistent_env_names 白名单',
          set(baked).issubset(set(persistent_names)), str(sorted(baked)))

    # 复刻 __init__.py:8-16：只在「环境里没设」时补上，不覆盖
    init_src = read(os.path.join(DEP, 'deep_ep/__init__.py'))
    assert re.search(r'if key not in os\.environ:\s*\n\s*os\.environ\[key\] = value', init_src), \
        '未找到 __init__.py:8-16 的「只补不覆盖」逻辑'
    child = subprocess.run([sys.executable, '-c',
                            'import os, importlib.util, sys\n'
                            f'spec=importlib.util.spec_from_file_location("m","{envs_py}")\n'
                            'm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
                            'for k,v in m.persistent_envs.items():\n'
                            '    if k not in os.environ: os.environ[k]=v\n'
                            'print(os.environ.get("EP_NCCL_ROOT_DIR"))\n'],
                           capture_output=True, text=True,
                           env={**os.environ, 'EP_NCCL_ROOT_DIR': '/explicit/wins'})
    got = child.stdout.strip()
    check('已显式设置时不被 persistent_envs 覆盖', got == '/explicit/wins', f'got={got!r}')
    child2 = subprocess.run([sys.executable, '-c',
                             'import os, importlib.util, sys\n'
                             f'spec=importlib.util.spec_from_file_location("m","{envs_py}")\n'
                             'm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
                             'for k,v in m.persistent_envs.items():\n'
                             '    if k not in os.environ: os.environ[k]=v\n'
                             'print(os.environ.get("EP_NCCL_ROOT_DIR"))\n'],
                            capture_output=True, text=True,
                            env={k: v for k, v in os.environ.items() if k != 'EP_NCCL_ROOT_DIR'})
    got2 = child2.stdout.strip()
    check('未设置时补上固化值', got2 == baked.get('EP_NCCL_ROOT_DIR', ''), f'got={got2!r}')

# --------------------------------------------------------------------------- #
# B. check_nccl_so 的 EP_SUPPRESS_NCCL_CHECK 闸门（真跑真实函数体）
# --------------------------------------------------------------------------- #
print('\n== Part B: EP_SUPPRESS_NCCL_CHECK 引导闸门 ==')
_ = torch.cuda.nccl.version()  # 强制加载 torch 链的 libnccl（复现真实 import 场景）
init_py = os.path.join(DEP, 'deep_ep/__init__.py')
tree = ast.parse(read(init_py))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_nccl_so')
fn_src = ast.get_source_segment(read(init_py), fn)
print(f'  check_nccl_so 源: {init_py}:{fn.lineno}-{fn.end_lineno}')

ns = {'os': os, 'glob': __import__('glob'), 'filecmp': __import__('filecmp')}
find_pkgs = os.path.join(DEP, 'deep_ep/utils/find_pkgs.py')
spec = importlib.util.spec_from_file_location('find_pkgs', find_pkgs)
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)
ns['find_nccl_root'] = fp.find_nccl_root
exec(compile(fn_src, init_py, 'exec'), ns)
check_nccl_so = ns['check_nccl_so']

os.environ['EP_SUPPRESS_NCCL_CHECK'] = '1'
r = check_nccl_so()
check('EP_SUPPRESS_NCCL_CHECK=1 -> 直接返回 None（绕过二进制核对）', r is None)
del os.environ['EP_SUPPRESS_NCCL_CHECK']
threw = False
try:
    check_nccl_so()
except AssertionError as e:
    threw = True
    print(f'  [INFO] 未抑制时 AssertionError: {str(e)[:110]}')
check('未抑制 -> AssertionError（本机 PyTorch NCCL != linked）', threw)

# --------------------------------------------------------------------------- #
# C. 全库 EP_* 清单普查
# --------------------------------------------------------------------------- #
print('\n== Part C: EP_* 环境变量清单普查 ==')
exts = ('.py', '.cpp', '.cuh', '.hpp', '.h', '.cu')
tokens = {}


def scan(root, repo):
    for dirpath, dirnames, files in os.walk(root):
        if any(s in dirpath for s in ('/build/', '/third-party/', '/.git/')):
            continue
        for f in files:
            if not f.endswith(exts):
                continue
            p = os.path.join(dirpath, f)
            try:
                txt = read(p)
            except Exception:
                continue
            # 只认「被引号包住的 EP_*」= 真正的环境变量名；裸 EP_* 是宏（EP_HOST_ASSERT…）
            for m in re.finditer(r'["\'](EP_[A-Z0-9_]+)["\']', txt):
                name = m.group(1)
                tokens.setdefault(name, {'repos': set(), 'files': set()})
                tokens[name]['repos'].add(repo)
                tokens[name]['files'].add(os.path.relpath(p, root))

scan(DEP, 'DeepEP')
scan(DEG, 'DeepGEMM')

real = {k: v for k, v in tokens.items() if not k.endswith('_')}
print(f'  扫描唯一「带引号」EP_* 名字: {len(real)} 个（裸 EP_* 宏已排除）')
for name in sorted(real):
    info = real[name]
    print(f'    {name:30s} files={len(info["files"]):2d} repos={sorted(info["repos"])}')

persistent = set(persistent_names)
live = sorted(set(real) - persistent)
RUNTIME_SET = {'EP_AVOID_RECORD_STREAM', 'EP_BUFFER_DEBUG', 'EP_DEFAULT_RDMA_SL',
               'EP_DISABLE_BARRIER_PROFILING', 'EP_DISABLE_GIN', 'EP_GIN_GDAKI_DEBUG',
               'EP_NUM_MAX_LOCAL_RANKS', 'EP_OVERRIDE_RDMA_SL', 'EP_REUSE_NCCL_COMM',
               'EP_SUPPRESS_NCCL_CHECK', 'EP_USE_NVIDIA_TOOLS'}
print(f'\n  persistent（安装期固化）: {sorted(persistent)}')
print(f'  「运行时开关」11 条     : {sorted(RUNTIME_SET)}')
check('带引号 EP_* 总数 == 16（7 persistent + 9 仅运行时）', len(real) == 16, str(len(real)))
check('运行时开关 11 条全部命中（含 2 条同时是 persistent 的 RDMA SL）',
      RUNTIME_SET.issubset(set(real)) and len(RUNTIME_SET) == 11,
      str(sorted(RUNTIME_SET - set(real))))

# 与 docs 里记录的清单交叉核对（README/第 86/98 篇协作面）
for v in ['EP_SUPPRESS_NCCL_CHECK', 'EP_REUSE_NCCL_COMM', 'EP_BUFFER_DEBUG',
          'EP_AVOID_RECORD_STREAM', 'EP_DISABLE_GIN', 'EP_DEFAULT_RDMA_SL',
          'EP_OVERRIDE_RDMA_SL', 'EP_NUM_MAX_LOCAL_RANKS', 'EP_USE_NVIDIA_TOOLS',
          'EP_DISABLE_BARRIER_PROFILING', 'EP_GIN_GDAKI_DEBUG']:
    check(f'清单含 {v}', v in live or v in persistent)

# --------------------------------------------------------------------------- #
# D. 两个语义实证
# --------------------------------------------------------------------------- #
print('\n== Part D: 跨语言真值 + 默认值 ==')
# D1. EP_BUFFER_DEBUG：Python 用 os.environ.get(...,0) —— 非空字符串恒真
os.environ['EP_BUFFER_DEBUG'] = '0'
py_truthy = bool(os.environ.get('EP_BUFFER_DEBUG', 0))
check('Python: EP_BUFFER_DEBUG="0" 为真（会打印）', py_truthy is True,
      'C++ get_env<int> 同值 -> 0（关）；两语言不一致')
del os.environ['EP_BUFFER_DEBUG']
check('Python: 未设时为 0（假）', not bool(os.environ.get('EP_BUFFER_DEBUG', 0)))
# 消费点坐标复核
ep_py = read(os.path.join(DEP, 'deep_ep/buffers/ep.py'))
check('ep.py 用裸 os.environ.get 读 EP_BUFFER_DEBUG',
      "os.environ.get('EP_BUFFER_DEBUG', 0)" in ep_py)

# D2. EP_NUM_MAX_LOCAL_RANKS 默认 16 + engram 的 NCCL_WIN_STRIDE 副作用
engram = read(os.path.join(DEP, 'deep_ep/buffers/engram.py'))
m = re.search(r"num_max_local_ranks = int\(os\.getenv\('EP_NUM_MAX_LOCAL_RANKS', 16\)\) if allow_hybrid_mode else 1", engram)
check('engram.py:61 默认 16 / 非 hybrid 为 1', bool(m))

def engram_registered_bytes(num_gpu_bytes, num_rdma_storage_bytes, allow_hybrid, num_max_local_ranks):
    n = num_max_local_ranks if allow_hybrid else 1
    return num_gpu_bytes + num_rdma_storage_bytes * n + (1 << 32)

def align(x, a):
    return (x + a - 1) // a * a

# 复刻 engram.py:62-66 的 NCCL_WIN_STRIDE 副作用
def win_stride_or_none(num_registered_bytes, total_gpu_bytes):
    return align(num_registered_bytes, 1 << 32) if num_registered_bytes > total_gpu_bytes else None

rb_small = engram_registered_bytes(1 << 30, 1 << 20, True, 16)
rb_big = engram_registered_bytes(200 << 30, 1 << 30, True, 16)
check('小缓冲不触发 NCCL_WIN_STRIDE', win_stride_or_none(rb_small, 80 << 30) is None,
      f'registered={rb_small/2**30:.2f} GiB')
ws = win_stride_or_none(rb_big, 80 << 30)
check('大缓冲触发 NCCL_WIN_STRIDE（对齐 4 GiB）', ws is not None and ws % (1 << 32) == 0,
      f'registered={rb_big/2**30:.1f} GiB -> {ws}')
# RDMA storage 随 local-rank 数放大 <=> 默认 16 是「保守预留 16 卡本地存储」
r16 = engram_registered_bytes(0, 1 << 30, True, 16)
r8 = engram_registered_bytes(0, 1 << 30, False, 16)
check('非 hybrid 只按 1 份 RDMA storage 预留', r8 < r16, f'hybrid={r16/2**30:.0f} vs non={r8/2**30:.0f} GiB')

print('\n' + ('ALL PASS' if not FAILS else f'FAILURES: {FAILS}'))
sys.exit(1 if FAILS else 0)
