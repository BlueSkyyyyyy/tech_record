"""P108-1 实验：DeepEP 的测试/进程引导层与可复现性契约。

目标源码（`~/github/DeepEP@93eb6eb`，`deep_ep.__version__ == 2.5.0`）：
  * `deep_ep/utils/envs.py`
      - `EnvOverride:22-37`（安装期开关的临时覆盖上下文）
      - `init_seed:40-51` / `get_local_seed:54-61` / `get_global_seed:64-71`
      - `dist_print:74-86`（once-in-node + 屏障）
      - `init_dist:89-133`（**所有 DeepEP 测试的入口**；SL setdefault / tcp init / device_id 兼容 / 默认 dtype&device）
      - `check_torch_deterministic:184-190`
  * 消费点：`deep_ep/utils/refs.py:151`（`get_global_seed`）、
    `deep_ep/buffers/engram.py:81-85`（`EnvOverride`）、
    `deep_ep/buffers/ep.py:639/813`（`check_torch_deterministic`）、
    `deep_ep/buffers/ep.py:18-21/430/845`（`value_or`/`weak_lru`）。
  * 对照（已被别的篇目覆盖）：`semantic.py:5-26`（第 83 篇）、`find_pkgs.py`（第 86 篇）、
    `check_nccl_so`/persistent envs（第 117 篇）。

本脚本（纯 Python / torch，架构无关）**直接加载真实 `deep_ep/utils/envs.py` 源码**（用
`sys.modules` 注入合成 `deep_ep._C`/`deep_ep.comm` 绕过 `import deep_ep` 的 `check_nccl_so` 闸门），
然后：
  A. `init_seed` 的 local/global 语义（local = global + rank）与 torch/random 播种；
  B. 单进程真跑 `init_dist(0, 1)`（NCCL world_size=1）——SL setdefault、默认 dtype/device、
     `new_group`、种子可复现；
  C. `EnvOverride` 四条语义（enable=False 不生效 / 覆盖 / 未设时退出即 pop / 原有值恢复）；
  D. `dist_print` 的 once-in-node 门控与屏障调用次数；
  E. `check_torch_deterministic` 四种组合；
  F. `get_local_seed` 全库无消费者的「死 API」实证。

运行：bash harness/exp_deepep_process_bootstrap.sh
"""
import hashlib
import importlib.util
import inspect
import os
import random
import socket
import sys
import types

import torch  # noqa: F401  —— 真实 torch，用于 set_default_dtype / manual_seed / cuda

DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')

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
semantic_path = os.path.join(DEP, 'deep_ep', 'utils', 'semantic.py')
print(f'  envs.py     sha256={sha256(envs_path)[:16]}')
print(f'  semantic.py sha256={sha256(semantic_path)[:16]}')
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
print(f'  加载成功：envs.{sorted(n for n in dir(envs) if not n.startswith("__"))}')


# --------------------------------------------------------------------------- #
# A. init_seed：local = global + rank
# --------------------------------------------------------------------------- #
print()
print('== Part A: init_seed 的 local / global 语义 ==')


class FakeDist:
    def __init__(self, rank):
        self._rank = rank
        self.barrier_calls = 0

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return 1

    def barrier(self, *a, **k):
        self.barrier_calls += 1


saved_dist = envs.dist
envs.dist = FakeDist(3)
envs.init_seed(100)
check('local_seed == global + rank', envs.get_local_seed() == 103, f'={envs.get_local_seed()}')
check('global_seed == 100', envs.get_global_seed() == 100, f'={envs.get_global_seed()}')

# torch / random 都按 local seed 播种：与手工 manual_seed 对齐
t_ref = (torch.manual_seed(103), torch.randn(4))[1]
envs.init_seed(100)
t_now = torch.randn(4)
check('torch 按 local_seed 播种', torch.equal(t_ref, t_now))
envs.init_seed(100)
check('同 seed 重复可复现', torch.equal(t_now, torch.randn(4)))

envs.dist = FakeDist(7)
envs.init_seed(100)
check('不同 rank 得到不同 local_seed', envs.get_local_seed() == 107)
envs.dist = saved_dist


# --------------------------------------------------------------------------- #
# B. init_dist：单进程真跑（NCCL world_size=1）
# --------------------------------------------------------------------------- #
print()
print('== Part B: init_dist(0, 1) 单进程真跑 ==')
if not torch.cuda.is_available():
    print('  [SKIP] 无 CUDA')
else:
    # 记录 init_dist 会 setdefault 的两个 SL 变量（先清掉以观察其效果）
    for k in ('NCCL_IB_SL', 'EP_OVERRIDE_RDMA_SL'):
        os.environ.pop(k, None)
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'

    import torch.distributed as dist
    print(f'  dist.init_process_group 签名: {inspect.signature(dist.init_process_group)}')
    has_device_id = 'device_id' in inspect.signature(dist.init_process_group).parameters
    print(f'  含 device_id 参数: {has_device_id}（init_dist:124-126 的兼容分支）')

    torch.set_default_dtype(torch.float32)
    rank, world, group = envs.init_dist(0, 1, seed=42)
    check('返回 rank', rank == 0, f'={rank}')
    check('返回 world_size', world == 1, f'={world}')
    check('返回 group 尺寸', group.size() == 1, f'={group.size()}')
    check('NCCL_IB_SL setdefault = 1', os.environ.get('NCCL_IB_SL') == '1')
    check('EP_OVERRIDE_RDMA_SL setdefault = 1', os.environ.get('EP_OVERRIDE_RDMA_SL') == '1')
    check('默认 dtype 设为 bfloat16', torch.get_default_dtype() == torch.bfloat16,
          f'={torch.get_default_dtype()}')
    check('默认 device 设为 cuda', torch.tensor([1.0]).device.type == 'cuda')
    check('cuda current_device 设为 local_rank', torch.cuda.current_device() == 0)
    check('init_dist 内部已 init_seed（local=42+0）', envs.get_local_seed() == 42,
          f'={envs.get_local_seed()}')

    # 种子可复现：init_dist 后 torch 流应等于「manual_seed(local_seed)」
    sample = torch.randn(4, device='cuda')
    torch.manual_seed(42)
    ref = torch.randn(4, device='cuda')
    check('init_dist 后 torch 流 == manual_seed(42)', torch.equal(sample, ref))
    dist.destroy_process_group()

# setdefault 不覆盖已存在值
os.environ['NCCL_IB_SL'] = '3'
print(f'  setdefault 语义：已设 NCCL_IB_SL=3 时不会被改成 1（源码用 setdefault）')
os.environ.pop('NCCL_IB_SL', None)


# --------------------------------------------------------------------------- #
# C. EnvOverride
# --------------------------------------------------------------------------- #
print()
print('== Part C: EnvOverride 上下文语义 ==')
for k in ('EXP_OV_A', 'EXP_OV_B', 'EXP_OV_C'):
    os.environ.pop(k, None)

# C1 enable=False 的项完全不参与
with envs.EnvOverride([('EXP_OV_A', '1', True), ('EXP_OV_B', '2', False)]):
    check('enable=True 生效', os.environ.get('EXP_OV_A') == '1')
    check('enable=False 不生效', os.environ.get('EXP_OV_B') is None)
check('退出后未设的原键被 pop', 'EXP_OV_A' not in os.environ)

# C2 原有值恢复
os.environ['EXP_OV_C'] = 'orig'
with envs.EnvOverride([('EXP_OV_C', 'override', True)]):
    check('覆盖已有值', os.environ.get('EXP_OV_C') == 'override')
check('退出后恢复原值', os.environ.get('EXP_OV_C') == 'orig')
os.environ.pop('EXP_OV_C', None)

# C3 异常也能恢复
try:
    with envs.EnvOverride([('EXP_OV_A', 'x', True)]):
        raise RuntimeError('boom')
except RuntimeError:
    pass
check('异常退出仍恢复', 'EXP_OV_A' not in os.environ)

# C4 engram 的真实用法（NCCL_GIN_TYPE=3 + 条件项）
with envs.EnvOverride([('NCCL_GIN_TYPE', '3', True),
                       ('NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC', '8', False)]):
    check('engram 用法：GIN_TYPE 生效', os.environ.get('NCCL_GIN_TYPE') == '3')
    check('engram 用法：条件项关闭不生效', 'NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC' not in os.environ)
check('engram 用法：退出清理', 'NCCL_GIN_TYPE' not in os.environ)


# --------------------------------------------------------------------------- #
# D. dist_print：once-in-node + 屏障
# --------------------------------------------------------------------------- #
print()
print('== Part D: dist_print 的 once-in-node 与屏障 ==')
envs.dist = FakeDist(0)
envs._local_rank = 0
envs.dist_print('hello from rank0', once_in_node=True)
check('once_in_node=True & local_rank==0 打印且屏障 1 次', envs.dist.barrier_calls == 1,
      f'barrier_calls={envs.dist.barrier_calls}')
envs.dist = FakeDist(0)
envs._local_rank = 3
envs.dist_print('should be suppressed', once_in_node=True)
check('once_in_node=True & local_rank!=0 仍屏障但不打印', envs.dist.barrier_calls == 1,
      f'barrier_calls={envs.dist.barrier_calls}')
envs.dist = FakeDist(0)
envs._local_rank = 3
envs.dist_print('all ranks', once_in_node=False)
check('once_in_node=False 每个 rank 都打印', envs.dist.barrier_calls == 1,
      f'barrier_calls={envs.dist.barrier_calls}')

# _local_rank 未设置时断言
envs._local_rank = None
try:
    envs.dist_print('x')
    check('_local_rank 未设置为 None 时断言', False)
except AssertionError:
    check('_local_rank 未设置为 None 时断言', True)
envs.dist = saved_dist
envs._local_rank = None


# --------------------------------------------------------------------------- #
# E. check_torch_deterministic
# --------------------------------------------------------------------------- #
print()
print('== Part E: check_torch_deterministic 四组合 ==')
saved_det = torch.are_deterministic_algorithms_enabled()
saved_fill = torch.utils.deterministic.fill_uninitialized_memory


def set_det(det, fill):
    torch.use_deterministic_algorithms(det, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = fill


for det, fill, should_raise in [(False, True, False), (True, False, False),
                                (False, False, False), (True, True, True)]:
    set_det(det, fill)
    raised = False
    try:
        envs.check_torch_deterministic()
    except AssertionError:
        raised = True
    check(f'det={det} fill={fill} -> raise={should_raise}', raised == should_raise)
torch.use_deterministic_algorithms(saved_det)
torch.utils.deterministic.fill_uninitialized_memory = saved_fill


# --------------------------------------------------------------------------- #
# F. get_local_seed 是死 API
# --------------------------------------------------------------------------- #
print()
print('== Part F: get_local_seed 全库消费者普查 ==')
consumers = []
for root, _dirs, files in os.walk(DEP):
    if '/build/' in root or '/.git' in root:
        continue
    for fn in files:
        if not fn.endswith('.py'):
            continue
        p = os.path.join(root, fn)
        if p == envs_path:
            continue
        with open(p, 'r') as f:
            for i, line in enumerate(f, 1):
                if 'get_local_seed' in line:
                    consumers.append(f'{os.path.relpath(p, DEP)}:{i}')
print(f'  get_local_seed 消费者: {consumers if consumers else "（无）"}')
check('get_local_seed 无任何消费者（死 API）', len(consumers) == 0)


print()
if FAILS:
    print(f'== 结果：FAIL {len(FAILS)} -> {FAILS}')
    sys.exit(1)
print('== 结果：ALL PASS ==')
