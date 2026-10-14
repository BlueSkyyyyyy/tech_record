"""P109-1 实验：DeepEP 的 NVLink 连接自检 `check_nvlink_connections` 与 H100 P2P 拓扑实测。

目标源码（`~/github/DeepEP@93eb6eb`，`deep_ep.__version__ == 2.5.0`）：
  * `deep_ep/utils/envs.py`
      - `check_nvlink_connections:146-181`（PCIe GPU 只能两两 NVLink，故做 P2P 状态自检）
      - 消费点：`deep_ep/buffers/engram.py:67`、`deep_ep/buffers/ep.py:325`
      - 邻居（已被别的篇目覆盖）：`check_torch_deterministic:184-190`（第 118 篇）、
        `find_pkgs.py`（第 86 篇）、`get_{physical,logical}_domain_size:136-143`（第 73/86 篇）
  * 外部参考（源码注释里给出的出处）：vLLM `vllm/platforms/cuda.py` 的 P2P 检查。

本脚本（纯 Python / torch / pynvml，架构无关）**直接加载真实 `deep_ep/utils/envs.py` 源码**
（`sys.modules` 注入合成 `deep_ep._C`/`deep_ep.comm` 绕过 `import deep_ep` 的 `check_nccl_so` 闸门），
然后：
  A. H100 设备名不含 'PCIE' ⇒ `check_nvlink_connections` 在 H100 上是 **no-op**（真机调用返回 None）；
  B. 真机 NVML P2P 状态矩阵（8×8，NVML_P2P_CAPS_INDEX_{NVLINK,WRITE,ATOMICS}）——全互联的证据；
  C. 对照 `torch.cuda.can_device_access_peer`（CUDA 运行时视角），与 NVML 视角一致性；
  D. 模拟 PCIe 分支：group.size()>2 时断言 / size==2 时走完整 body（真实 pynvml）/ 假 pynvml 造非 OK 断言；
     以及 `CUDA_VISIBLE_DEVICES` 物理设备号解析的边界；
  E. 真机 P2P 拷贝带宽矩阵（调用 `src/nvlink_p2p_matrix.cu` 编出的二进制）；
  F. pynvml 生命周期（nvmlInit/nvmlShutdown 配对）与一个绑定坑（NVML_P2P_CAPS_INDEX_READ 是 tuple）。

运行：bash harness/exp_deepep_nvlink_p2p.sh
"""
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import types

import torch

DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')

P2P_BIN = os.environ.get('P2P_BIN', '/tmp/nvlink_p2p_matrix')

FAILS = []


def check(label, cond, extra=''):
    tag = '[PASS]' if cond else '[FAIL]'
    if not cond:
        FAILS.append(label)
    print(f'  {tag} {label} {extra}')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# --------------------------------------------------------------------------- #
# 0. 加载真实 deep_ep/utils/envs.py（stub 掉 _C / comm）
# --------------------------------------------------------------------------- #
print('== 环境坐标 ==')
print(f'  DeepEP root: {DEP}')
envs_path = os.path.join(DEP, 'deep_ep', 'utils', 'envs.py')
print(f'  envs.py sha256={sha256(envs_path)[:16]}')
print(f'  torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}')

_deep_ep = types.ModuleType('deep_ep')
_deep_ep.__path__ = [os.path.join(DEP, 'deep_ep')]
sys.modules['deep_ep'] = _deep_ep
sys.modules['deep_ep._C'] = types.ModuleType('deep_ep._C')
sys.modules['deep_ep.comm'] = types.ModuleType('deep_ep.comm')
_utils = types.ModuleType('deep_ep.utils')
_utils.__path__ = [os.path.join(DEP, 'deep_ep', 'utils')]
sys.modules['deep_ep.utils'] = _utils


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


envs = load(envs_path, 'deep_ep.utils.envs')
print(f'  加载成功：{sorted(n for n in dir(envs) if not n.startswith("_"))}')

NUM_DEV = torch.cuda.device_count()


# --------------------------------------------------------------------------- #
# A. H100 上 check_nvlink_connections 是 no-op
# --------------------------------------------------------------------------- #
print()
print('== Part A: H100 设备名与 check_nvlink_connections 的 no-op ==')


class RealGroupStub:
    """只提供 .size() 的 group 桩——真机调用走 no-op，不需要真通信组。"""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


name = torch.cuda.get_device_name(0)
has_pcie = 'PCIE' in name
check("H100 设备名不含 'PCIE'", not has_pcie, f'name={name!r}')
# 真机调用：因为没有 'PCIE'，函数应直接返回 None（不 import pynvml、不 all_gather）
r = envs.check_nvlink_connections(RealGroupStub(NUM_DEV))
check('check_nvlink_connections 真机返回 None（no-op）', r is None)
# 负对照：把设备名伪装成 PCIE 且 group 很大 → 应命中 :157 断言
_saved = torch.cuda.get_device_name
torch.cuda.get_device_name = lambda *a, **k: name + ' PCIE'
try:
    envs.check_nvlink_connections(RealGroupStub(NUM_DEV))
    check('PCIE + group>2 应断言', False, '未触发（异常）')
except AssertionError as e:
    check('PCIE + group>2 命中 :157 断言', 'pairwise NVLink' in str(e), f'msg={str(e)[:48]!r}')
except Exception as e:  # noqa: BLE001
    check('PCIE + group>2 命中 :157 断言', False, f'非 AssertionError: {type(e).__name__}: {e}')
finally:
    torch.cuda.get_device_name = _saved


# --------------------------------------------------------------------------- #
# B. 真机 NVML P2P 状态矩阵
# --------------------------------------------------------------------------- #
print()
print('== Part B: 真机 NVML P2P 状态矩阵（8×8） ==')
import pynvml  # noqa: E402

pynvml.nvmlInit()
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(NUM_DEV)]

CAPS = [
    ('NVLINK', pynvml.NVML_P2P_CAPS_INDEX_NVLINK),
    ('WRITE', pynvml.NVML_P2P_CAPS_INDEX_WRITE),
    ('ATOMICS', pynvml.NVML_P2P_CAPS_INDEX_ATOMICS),
]
OK = pynvml.NVML_P2P_STATUS_OK
nvlink_status = {}
n_ok = {cn: 0 for cn, _ in CAPS}
n_pairs = 0
for cn, ci in CAPS:
    print(f'  {cn}:')
    for i in range(NUM_DEV):
        row = []
        for j in range(NUM_DEV):
            st = pynvml.nvmlDeviceGetP2PStatus(handles[i], handles[j], ci)
            row.append(st)
            if cn == 'NVLINK':
                nvlink_status[(i, j)] = st
        print('    ' + ' '.join(f'{v:3d}' for v in row))
# 统计非对角对是否全 OK
for (i, j), st in nvlink_status.items():
    if i == j:
        continue
    n_pairs += 1
    if st == OK:
        n_ok['NVLINK'] += 1
check('NVML NVLINK：所有非对角对状态 == OK', n_ok['NVLINK'] == n_pairs,
      f'{n_ok["NVLINK"]}/{n_pairs}')
# 也对 W/ATOMICS 做一次非对角统计（重建）
w_ok = sum(1 for i in range(NUM_DEV) for j in range(NUM_DEV) if i != j
           and pynvml.nvmlDeviceGetP2PStatus(handles[i], handles[j], pynvml.NVML_P2P_CAPS_INDEX_WRITE) == OK)
a_ok = sum(1 for i in range(NUM_DEV) for j in range(NUM_DEV) if i != j
           and pynvml.nvmlDeviceGetP2PStatus(handles[i], handles[j], pynvml.NVML_P2P_CAPS_INDEX_ATOMICS) == OK)
check('NVML WRITE：所有非对角对 OK', w_ok == n_pairs, f'{w_ok}/{n_pairs}')
check('NVML ATOMICS：所有非对角对 OK', a_ok == n_pairs, f'{a_ok}/{n_pairs}')

# B2: pynvml 绑定坑——NVML_P2P_CAPS_INDEX_READ 是 tuple 而非 int
read_const = pynvml.NVML_P2P_CAPS_INDEX_READ
print(f'  [观察] NVML_P2P_CAPS_INDEX_READ = {read_const!r} (type={type(read_const).__name__})')
check('READ 常量是 tuple（绑定坑，DeepEP 用 NVLINK 不受影响）', isinstance(read_const, tuple))
import ctypes  # noqa: E402

try:
    pynvml.nvmlDeviceGetP2PStatus(handles[0], handles[1], read_const)
    read_threw = False
    read_err = ''
except (TypeError, ctypes.ArgumentError) as e:
    read_threw = True
    read_err = type(e).__name__
check('用 READ 常量直接调用会抛（ctypes.ArgumentError）', read_threw, f'err={read_err}')


# --------------------------------------------------------------------------- #
# C. torch.cuda.can_device_access_peer 矩阵
# --------------------------------------------------------------------------- #
print()
print('== Part C: torch.cuda.can_device_access_peer（CUDA 运行时视角） ==')
n_peer = 0
mismatch = 0
for i in range(NUM_DEV):
    for j in range(NUM_DEV):
        if i == j:
            continue
        c = torch.cuda.can_device_access_peer(i, j)
        n_peer += 1 if c else 0
        if (nvlink_status[(i, j)] == OK) != bool(c):
            mismatch += 1
check('canAccessPeer 所有非对角对为真', n_peer == n_pairs, f'{n_peer}/{n_pairs}')
check('canAccessPeer 与 NVML NVLINK 状态一致', mismatch == 0, f'mismatch={mismatch}')
pynvml.nvmlShutdown()


# --------------------------------------------------------------------------- #
# D. 模拟 PCIe 分支
# --------------------------------------------------------------------------- #
print()
print('== Part D: 模拟 PCIe 分支（真实函数体） ==')


class FakeDist:
    """为 check_nvlink_connections 提供 all_gather_object：把注入的物理设备号填满输出。"""

    def __init__(self, phys_list):
        self.phys_list = phys_list

    def all_gather_object(self, out, obj, group):
        for k in range(len(out)):
            out[k] = self.phys_list[k % len(self.phys_list)]


saved_dist = envs.dist
saved_env_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
torch.cuda.get_device_name = lambda *a, **k: name + ' PCIE'
try:
    # D2: group.size()==2，物理设备号 [0,1]（H100 真机 NVLink 相连）→ 应通过
    envs.dist = FakeDist([0, 1])
    envs.check_nvlink_connections(RealGroupStub(2))
    check('PCIE + size==2 + 真实 pynvml（GPU0,1）→ 通过', True)

    # D3: 假 pynvml 返回非 OK → 应命中 :177 断言
    fake_pynvml = types.ModuleType('pynvml')
    fake_pynvml.NVML_P2P_CAPS_INDEX_NVLINK = 2
    fake_pynvml.NVML_P2P_STATUS_OK = 0
    fake_pynvml.NVML_P2P_STATUS_ERROR = 1
    calls = {'init': 0, 'shutdown': 0, 'status': 0}

    def _nvmlInit():
        calls['init'] += 1

    def _nvmlShutdown():
        calls['shutdown'] += 1

    def _handle(i):
        return ('H', i)

    def _status(h1, h2, caps):
        calls['status'] += 1
        # 只有 (0,1) 这一对返回非 OK，其余 OK
        idx = {h1[1], h2[1]}
        return 1 if idx == {0, 1} else 0

    fake_pynvml.nvmlInit = _nvmlInit
    fake_pynvml.nvmlShutdown = _nvmlShutdown
    fake_pynvml.nvmlDeviceGetHandleByIndex = _handle
    fake_pynvml.nvmlDeviceGetP2PStatus = _status
    sys.modules['pynvml'] = fake_pynvml
    envs.dist = FakeDist([0, 1])
    try:
        envs.check_nvlink_connections(RealGroupStub(2))
        check('假 pynvml 非 OK → 命中 :177 断言', False, '未触发')
    except AssertionError as e:
        check('假 pynvml 非 OK → 命中 :177 断言', 'not connected via NVLink' in str(e),
              f'msg={str(e)[:44]!r}')
    # 假 pynvml 也应被 close（nvmlShutdown 在断言前不会执行——断言直接抛出）
    print(f'  [观察] 假 pynvml 生命周期调用：init={calls["init"]} status={calls["status"]} '
          f'shutdown={calls["shutdown"]}（断言路径下 shutdown 不执行）')
finally:
    torch.cuda.get_device_name = _saved
    envs.dist = saved_dist
    sys.modules.pop('pynvml', None)
    # 恢复真实 pynvml 供后面用
    import importlib
    import pynvml as _real_pynvml  # noqa: F401
    sys.modules['pynvml'] = importlib.import_module('pynvml')

# D4: CUDA_VISIBLE_DEVICES 物理设备号解析（函数 :164-165 的表达式）
print()
print('  -- CUDA_VISIBLE_DEVICES 解析 --')
cases = [
    ('0,1,2,3,4,5,6,7', 3, '3'),
    ('0,1,2,3', 3, '3'),
    ('0,1,', 1, '1'),           # strip(',') 去掉尾部逗号
    ('4,5,6,7', 0, '4'),
]
parse_ok = True
for cvd, cur, expect in cases:
    devices = (cvd).strip(',').split(',')
    got = devices[cur]
    parse_ok &= (got == expect)
    print(f'    cvd={cvd!r} cur={cur} -> {got!r} (expect {expect!r})')
check('CUDA_VISIBLE_DEVICES 解析全对', parse_ok)


# --------------------------------------------------------------------------- #
# E. 真机 P2P 带宽矩阵
# --------------------------------------------------------------------------- #
print()
print('== Part E: 真机 P2P 拷贝带宽矩阵（cudaMemcpyPeerAsync） ==')
if os.path.isfile(P2P_BIN):
    out = subprocess.run([P2P_BIN, '64', '30'], capture_output=True, text=True, timeout=600)
    text = out.stdout
    for line in text.splitlines():
        print('  ' + line)
    # 解析有向带宽矩阵（只取 'directed P2P bandwidth' 段，避免和 canAccessPeer 段的 src 行混淆）
    seg = text.split('== directed P2P bandwidth')
    seg = seg[1].split('== same-device')[0] if len(seg) > 1 else ''
    mat = {}
    for line in seg.splitlines():
        mm = re.match(r'src\s*(\d+)\s+(.*)$', line)
        if not mm:
            continue
        i = int(mm.group(1))
        vals = mm.group(2).split()
        for j, v in enumerate(vals):
            if v != '-':
                mat[(i, j)] = float(v)
    if mat:
        allv = sorted(mat.values())
        mn, mx = allv[0], allv[-1]
        med = allv[len(allv) // 2]
        # 每个目的列的统计（写带宽受目的 GPU 繁忙程度影响）
        col_stats = {}
        for (i, j), v in mat.items():
            col_stats.setdefault(j, []).append(v)
        slow_col = min(col_stats, key=lambda j: min(col_stats[j]))
        print(f'  [分析] all-pair min={mn:.1f} median={med:.1f} max={mx:.1f} GB/s')
        print('  [分析] 每目的列 min：' +
              ' '.join(f'dst{j}={min(col_stats[j]):.0f}' for j in sorted(col_stats)))
        print(f'  [分析] 最慢目的列 dst{slow_col}（该 GPU 被其他任务共享时写带宽下降）')
        check('P2P 所有有向对 > 150 GB/s（NVLink 路径而非 PCIe）', mn > 150, f'min={mn:.1f}')
        check('P2P 中位带宽 > 300 GB/s', med > 300, f'median={med:.1f}')
        check('除最慢目的列外，其余对 > 300 GB/s',
              all(v > 300 for (i, j), v in mat.items() if j != slow_col),
              f'slow_col=dst{slow_col}')
    else:
        check('解析 P2P 矩阵', False, text[:200])
else:
    check(f'P2P 二进制存在 ({P2P_BIN})', False, '未编译')


# --------------------------------------------------------------------------- #
# 小结
# --------------------------------------------------------------------------- #
print()
print('== 小结 ==')
print(f'  FAILS={len(FAILS)}' + (f' -> {FAILS}' if FAILS else '  ALL PASS'))
sys.exit(1 if FAILS else 0)
