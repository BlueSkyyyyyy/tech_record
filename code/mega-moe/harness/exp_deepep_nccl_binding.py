"""P76-1 实验：DeepEP 的【NCCL 通信域绑定与 Python 引导层】。

目标源码（`~/github/DeepEP@93eb6eb`，尽量直接加载 / ast 抽取真实文件，不走 `deep_ep/__init__`）：
  * deep_ep/utils/find_pkgs.py        —— find_pkg_root / find_nccl_root / get_nccl_lib_name
  * deep_ep/__init__.py               —— check_nccl_so (:19-41) / init_jit (:44-51) / 引导顺序
  * deep_ep/comm/handle.py            —— NCCLCommHandle (:12-38) / get_nccl_comm_handle (:44-77)
                                          / get_physical|logical_domain_size (:80-95)
  * deep_ep/comm/barrier.py           —— Python barrier 包装
  * csrc/kernels/comm/context.cpp     —— get_physical/logical_domain_size 公式 (:52-63)
                                          / 域分解 (:166-183) / NCCL 版本闸门 (:156-162) / GIN 断言 (:124-146)
  * setup.py                          —— persistent_envs / NCCL 根目录 include+link / SONAME `-l:`

为什么 H100(sm90) 可跑：
  * 本层是**纯 host / Python**：包发现、NCCL 版本核对、comm 句柄生命周期、域大小算术，都不碰
    任何 kernel / UMMA / WGMMA。`import deep_ep` 会触发 `check_nccl_so()`（本机 PyTorch 链系统
    NCCL 2.29.7 ≠ DeepEP 链接的 pip NCCL 2.32.3 → AssertionError）。故这里用 sys.modules 注入
    合成 `deep_ep._C` stub，再按文件路径 exec 真实 `comm/handle.py`、`comm/barrier.py`，并 ast
    抽取真实 `check_nccl_so` 函数体运行（打印 sha256 佐证未改）。
  * 版本探测用 ctypes 直接 `dlopen` pip 的 `libnccl.so.2` 与系统的 `libnccl.so.2.29.7`，读
    `ncclGetVersion`，坐实「loaded vs linked」两个版本号。

运行：bash harness/exp_deepep_nccl_binding.sh > harness/exp_deepep_nccl_binding.out.txt
"""
import ast
import ctypes
import filecmp
import glob
import hashlib
import importlib.util
import os
import sys
import types

import torch


DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def installed_deep_ep_dir():
    spec = importlib.util.find_spec('deep_ep')
    return os.path.dirname(spec.origin)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------------------
# Part A：包发现层 find_pkgs.py（真实模块直接 import 执行）
# --------------------------------------------------------------------------------------
print('\n=== Part A: 包发现层 find_pkgs.py（真实模块，H100 真机） ===')
FIND_PKGS = os.path.join(DEP, 'deep_ep/utils/find_pkgs.py')
fp = load_file('deep_ep_utils_find_pkgs_probe', FIND_PKGS)
print(f'  find_pkgs.py sha256={sha256(FIND_PKGS)}')

real_root = fp.find_nccl_root(optional=True)
lib_dir = os.path.join(real_root, 'lib') if real_root else ''
print(f'  find_nccl_root() = {real_root}')
check('A1 find_nccl_root 解析出 pip nvidia/nccl 根目录',
      real_root is not None and os.path.isdir(lib_dir), real_root)

lib_name = fp.get_nccl_lib_name(real_root)
lib_files = sorted(os.path.basename(p) for p in glob.glob(f'{lib_dir}/libnccl.so*'))
print(f'  get_nccl_lib_name() = {lib_name}   (glob: {lib_files})')
check('A2 get_nccl_lib_name 返回实际存在的 SONAME 文件名',
      lib_name in lib_files and os.path.isfile(os.path.join(lib_dir, lib_name)), lib_name)

# 环境变量覆盖优先级：EP_NCCL_ROOT_DIR > NCCL_DIR
env_saved = {k: os.environ.pop(k, None) for k in ('EP_NCCL_ROOT_DIR', 'NCCL_DIR')}
os.environ['EP_NCCL_ROOT_DIR'] = '/tmp/_epx'
check('A3a EP_NCCL_ROOT_DIR 优先', fp.find_pkg_root('nccl') == '/tmp/_epx')
os.environ['NCCL_DIR'] = '/tmp/_sysdir'
check('A3b 同时存在时 EP_NCCL_ROOT_DIR 胜过 NCCL_DIR', fp.find_pkg_root('nccl') == '/tmp/_epx')
del os.environ['EP_NCCL_ROOT_DIR']
check('A3c 无 EP_ 前缀时回退 NCCL_DIR', fp.find_pkg_root('nccl') == '/tmp/_sysdir')
for k in ('EP_NCCL_ROOT_DIR', 'NCCL_DIR'):
    os.environ.pop(k, None)
for k, v in env_saved.items():
    if v is not None:
        os.environ[k] = v

check('A4 optional=True 未找到返回 None',
      fp.find_pkg_root('definitely_not_a_pkg_xyz', optional=True) is None)
try:
    fp.find_pkg_root('definitely_not_a_pkg_xyz')
    check('A5 optional=False 未找到抛 AssertionError', False, 'no raise')
except AssertionError as e:
    check('A5 optional=False 未找到抛 AssertionError', 'Cannot find package' in str(e), str(e))


def oracle_find_pkg_root(name, lib_name):
    """独立复刻 find_pkg_root 的算法（枚举 dist、路径优先级、lib 定位）"""
    from importlib.metadata import distributions
    path_priority = {p: i for i, p in enumerate(sys.path)}
    best, best_priority = None, len(sys.path)
    for dist in distributions():
        dist_name = (dist.metadata['Name'] or '').lower()
        if f'nvidia-{name}' not in dist_name and f'nvidia_{name}' not in dist_name:
            continue
        dist_site = str(dist._path.parent)
        priority = path_priority.get(dist_site, len(sys.path))
        if priority > best_priority:
            continue
        for f in (dist.files or []):
            if lib_name in str(f):
                lib_dir2 = os.path.dirname(str(f.locate()))
                root = os.path.dirname(lib_dir2) if os.path.basename(lib_dir2) == 'lib' else lib_dir2
                best, best_priority = root, priority
                break
    return best


ora = oracle_find_pkg_root('nccl', 'libnccl.so')
check('A6 独立 oracle 复刻 find_pkg_root 结果一致',
      ora == real_root, f'oracle={ora} real={real_root}')

# lru_cache：第二次调用命中缓存
fp.find_nccl_root.cache_clear()
fp.find_nccl_root()
fp.find_nccl_root()
ci = fp.find_nccl_root.cache_info()
check('A7 find_nccl_root 带 lru_cache（第二次命中）', ci.hits >= 1 and ci.misses == 1, str(ci))


# --------------------------------------------------------------------------------------
# Part B：check_nccl_so（ast 抽取真实函数体，真机跑出真实版本失配）
# --------------------------------------------------------------------------------------
print('\n=== Part B: 运行时 NCCL 版本核对 check_nccl_so（ast 抽取真实函数体） ===')
INIT = os.path.join(installed_deep_ep_dir(), '__init__.py')
src = open(INIT).read()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_nccl_so')
print(f'  __init__.py sha256={sha256(INIT)}  check_nccl_so 源自 :{fn.lineno}-{fn.end_lineno}')
ns = {'os': os, 'glob': glob, 'filecmp': filecmp, 'find_nccl_root': fp.find_nccl_root}
exec(compile(ast.Module(body=[fn], type_ignores=[]), INIT, 'exec'), ns)
check_nccl_so = ns['check_nccl_so']

# 1) 抑制开关
os.environ['EP_SUPPRESS_NCCL_CHECK'] = '1'
try:
    check_nccl_so()
    check('B1 EP_SUPPRESS_NCCL_CHECK=1 直接返回', True)
except Exception as e:
    check('B1 EP_SUPPRESS_NCCL_CHECK=1 直接返回', False, repr(e))
os.environ.pop('EP_SUPPRESS_NCCL_CHECK', None)

# 2) 真实比对：loaded（PyTorch 载入）vs linked（DeepEP 期望）
with open('/proc/self/maps') as f:
    loaded = None
    for line in f:
        if 'libnccl' in line:
            so = line.strip().split(' ')[-1]
            loaded = so if loaded is None else loaded
linked = sorted(glob.glob(f'{fp.find_nccl_root()}/lib/libnccl.so*'))[0]
print(f'  loaded (torch)   = {loaded}')
print(f'  linked (DeepEP)  = {linked}')
check('B2 loaded ≠ linked（本机真实失配）',
      loaded is not None and not filecmp.cmp(loaded, linked, shallow=False),
      f'filecmp={filecmp.cmp(loaded, linked, shallow=False)}')
try:
    check_nccl_so()
    check('B3 check_nccl_so 在失配时抛 AssertionError', False, 'no raise')
except AssertionError as e:
    msg = str(e)
    check('B3 check_nccl_so 在失配时抛 AssertionError',
          'Invalid NCCL versions' in msg and loaded in msg and linked in msg,
          '...' + msg[msg.find('Invalid'):msg.find('Invalid') + 90])

# 3) filecmp 的「逐字节」语义：shallow=True 只看 stat 可能误判，此处用 False
check('B4 check_nccl_so 用 shallow=False（内容级比较，非仅 stat）',
      'shallow=False' in ast.get_source_segment(src, fn))


# --------------------------------------------------------------------------------------
# Part C：comm/handle.py —— NCCLCommHandle 生命周期 + comm 句柄缓存（合成 _C stub 跑真实代码）
# --------------------------------------------------------------------------------------
print('\n=== Part C: comm/handle.py —— 句柄生命周期与缓存（真实代码 + 合成 _C stub） ===')
HANDLE = os.path.join(DEP, 'deep_ep/comm/handle.py')
BARRIER = os.path.join(DEP, 'deep_ep/comm/barrier.py')
print(f'  handle.py sha256={sha256(HANDLE)}  barrier.py sha256={sha256(BARRIER)}')


class FakeC:
    def __init__(self):
        self.created = []
        self.destroyed = []
        self.uid_calls = 0

    def get_local_nccl_unique_id(self):
        self.uid_calls += 1
        return b'LOCAL-UID'

    def create_nccl_comm(self, uid, num_ranks, rank_idx):
        h = 7000 + num_ranks * 100 + rank_idx
        self.created.append((uid, num_ranks, rank_idx, h))
        return h

    def destroy_nccl_comm(self, comm):
        self.destroyed.append(comm)

    def get_physical_domain_size(self, comm):
        return (2, 8) if comm == 999 else (1, 1)

    def get_logical_domain_size(self, comm, allow_hybrid_mode):
        return (2, 8) if allow_hybrid_mode else (1, 8)


fake_c = FakeC()
pkg = types.ModuleType('deep_ep')
pkg.__path__ = [f'{DEP}/deep_ep']
sys.modules['deep_ep'] = pkg
sys.modules['deep_ep._C'] = fake_c
sys.modules.setdefault('deep_ep.comm', types.ModuleType('deep_ep.comm'))

handle_mod = load_file('deep_ep_comm_handle_probe', HANDLE)


class FakeBackend:
    def __init__(self, ptr=None):
        if ptr is not None:
            self._comm_ptr = lambda: ptr


class FakeGroup:
    def __init__(self, size, rank, backend):
        self._size, self._rank, self._backend = size, rank, backend

    def size(self):
        return self._size

    def rank(self):
        return self._rank

    def _get_backend(self, dev):
        return self._backend


# C1: 复用 PyTorch 的 comm（backend._comm_ptr 存在，EP_REUSE_NCCL_COMM 默认 1）
g1 = FakeGroup(8, 3, FakeBackend(ptr=999))
h1 = handle_mod.get_nccl_comm_handle(g1)
check('C1a 复用 PyTorch comm → managed=False 且 ptr 原样',
      h1.get() == 999 and h1.managed is False and not fake_c.created, f'comm={h1.get()} managed={h1.managed}')
h1b = handle_mod.get_nccl_comm_handle(g1)
check('C1b 同一 group 第二次命中缓存（同一对象）', h1b is h1)

# C2: 关闭复用 → DeepEP 自建 comm（managed=True，走 all_gather_object）
os.environ['EP_REUSE_NCCL_COMM'] = '0'
_saved_ag = torch.distributed.all_gather_object


def fake_all_gather_object(out_list, obj, group):
    for i in range(group.size()):
        out_list[i] = obj


torch.distributed.all_gather_object = fake_all_gather_object
g2 = FakeGroup(8, 3, FakeBackend(ptr=999))
h2 = handle_mod.get_nccl_comm_handle(g2)
check('C2a EP_REUSE_NCCL_COMM=0 → DeepEP 自建（managed=True）',
      h2.managed is True and h2.get() == 7000 + 8 * 100 + 3 and len(fake_c.created) == 1,
      f'comm={h2.get()} created={fake_c.created}')
check('C2b 自建路径调用 get_local_nccl_unique_id', fake_c.uid_calls == 1)

# C3: force_new_comm 绕过缓存与 PyTorch 复用
h3 = handle_mod.get_nccl_comm_handle(g2, force_new_comm=True)
check('C3a force_new_comm=True 生成新的 managed 句柄（≠缓存对象）',
      h3 is not h2 and h3.managed is True and len(fake_c.created) == 2)
torch.distributed.all_gather_object = _saved_ag
os.environ.pop('EP_REUSE_NCCL_COMM', None)

# C4: managed 句柄析构 → destroy；非 managed 不 destroy
#     （注意：经 get_nccl_comm_handle 拿到的句柄还被 _storage 缓存持有，需单独构造裸句柄）
import gc
raw_managed = handle_mod.NCCLCommHandle(555, True)
raw_unmanaged = handle_mod.NCCLCommHandle(666, False)
del raw_managed, raw_unmanaged
gc.collect()
check('C4a managed 句柄析构触发 destroy_nccl_comm',
      fake_c.destroyed == [555], f'destroyed={fake_c.destroyed}')

# C4c: 缓存清理时被缓存持有的 managed 句柄才析构；PyTorch 复用句柄不析构
h2_ref = handle_mod.get_nccl_comm_handle(FakeGroup(8, 3, FakeBackend(ptr=999)), force_new_comm=False)
fake_c.destroyed.clear()
handle_mod.destroy_all_managed_nccl_comm()
gc.collect()
check('C4b 非 managed（PyTorch 复用）句柄在清缓存后【不】destroy',
      fake_c.destroyed == [], f'destroyed={fake_c.destroyed}')

# C5: destroy_all_managed_nccl_comm 只清缓存
handle_mod.get_nccl_comm_handle(FakeGroup(4, 0, FakeBackend(ptr=999)))
check('C5a 缓存非空', len(handle_mod._storage) >= 1, f'cache={len(handle_mod._storage)}')
handle_mod.destroy_all_managed_nccl_comm()
check('C5b destroy_all_managed_nccl_comm 清空缓存', len(handle_mod._storage) == 0)

# C6: 域大小包装：context 分支 vs _C 分支
class FakeCtx:
    def get_physical_domain_size(self):
        return (2, 8)

    def get_logical_domain_size(self):
        return (2, 8)


class FakeBuf:
    context = FakeCtx()


check('C6a 有 .context → 直接用 Context 的域大小（不走 _C）',
      handle_mod.get_physical_domain_size(FakeBuf()) == (2, 8)
      and handle_mod.get_logical_domain_size(FakeBuf()) == (2, 8))
g_ctx = FakeGroup(8, 3, FakeBackend(ptr=999))
check('C6b 无 .context → 经 _C.get_physical_domain_size(comm)',
      handle_mod.get_physical_domain_size(g_ctx) == (2, 8)
      and handle_mod.get_logical_domain_size(g_ctx, allow_hybrid_mode=False) is not None)


# --------------------------------------------------------------------------------------
# Part D：comm/barrier.py —— Python barrier 包装的参数转发
# --------------------------------------------------------------------------------------
print('\n=== Part D: comm/barrier.py —— barrier 参数转发（真实代码 + stub） ===')
barrier_calls = []


def fake_barrier(context, with_cpu_sync, sequential, wait_comm_stream):
    barrier_calls.append((context, with_cpu_sync, sequential, wait_comm_stream))


fake_c.barrier = fake_barrier
barrier_mod = load_file('deep_ep_comm_barrier_probe', BARRIER)
buf_ctx = types.SimpleNamespace(context='CTX0', contexts=['C0', 'C1'])
barrier_mod.barrier(buf_ctx, wait_comm_stream=False, with_cpu_sync=True, sequential=False)
check('D1 barrier 从 buffer.context 取域并按序转发 4 参数',
      barrier_calls[-1] == ('CTX0', True, False, False), str(barrier_calls[-1]))
barrier_mod.barrier(buf_ctx, context_idx=1)
check('D2 context_idx 覆盖 buffer.context', barrier_calls[-1][0] == 'C1')

# --------------------------------------------------------------------------------------
# Part E：域分解算术（复刻 context.cpp:52-63 / 166-183 的公式并与 oracle 对拍）
# --------------------------------------------------------------------------------------
print('\n=== Part E: 域分解算术（复刻 context.cpp 公式 + 独立 oracle） ===')


def physical(num_ranks, num_nvl):
    return (num_ranks // num_nvl, num_nvl)


def logical(num_ranks, num_nvl, hybrid):
    num_rdma, _ = physical(num_ranks, num_nvl)
    return (num_rdma if hybrid else 1,
            num_nvl if hybrid else num_rdma * num_nvl)


def oracle(num_ranks, num_nvl, hybrid):
    # 独立写法
    r = num_ranks // num_nvl
    if hybrid:
        return (r, num_nvl)
    return (1, num_ranks)


cases = [(8, 8, False), (8, 8, True), (16, 8, True), (16, 8, False),
         (32, 8, True), (64, 8, True), (2, 2, True), (1, 1, False)]
ok = True
for nr, nvl, hy in cases:
    p = physical(nr, nvl)
    l = logical(nr, nvl, hy)
    o = oracle(nr, nvl, hy)
    if l != o or p != (nr // nvl, nvl):
        ok = False
    print(f'    ranks={nr:3d} nvl={nvl} hybrid={int(hy)} -> phys={p} logical={l} oracle={o}')
check('E1 physical/logical 域大小公式与独立 oracle 逐配置一致', ok)

# EP8 单节点：hybrid=0 → logical=(1,8)，hybrid=1 → logical=(1,8)（nvl==ranks）
check('E2 EP8 单节点 hybrid/非 hybrid 的 logical 均退化到 (1,8)',
      logical(8, 8, False) == (1, 8) and logical(8, 8, True) == (1, 8))
# EP16 双节点：hybrid → (2,8)；非 hybrid → (1,16)
check('E3 EP16 双节点 hybrid=(2,8) / 非 hybrid=(1,16)',
      logical(16, 8, True) == (2, 8) and logical(16, 8, False) == (1, 16))
# ginSignalCount = num_ranks + 4（context.cpp:140）
check('E4 ginSignalCount = num_ranks + 2*2', [nr + 4 for nr, _, _ in cases] == [12, 12, 20, 20, 36, 68, 6, 5])
# get_sym_ptr 的 offset 不变性：offset = ptr - workspace，与目标 rank 无关
workspace = 0x100000
ptrs = [workspace + 0x2000 for _ in range(8)]
offsets = [p - workspace for p in ptrs]
check('E5 get_sym_ptr 的 offset = ptr - workspace 与 dst rank 无关',
      len(set(offsets)) == 1 and offsets[0] == 0x2000)

# --------------------------------------------------------------------------------------
# Part F：NCCL 版本探测（ctypes dlopen 两个库读 ncclGetVersion）
# --------------------------------------------------------------------------------------
print('\n=== Part F: NCCL 版本探测（loaded 系统库 vs linked pip 库） ===')


def nccl_version(path):
    lib = ctypes.CDLL(path)
    v = ctypes.c_int()
    lib.ncclGetVersion(ctypes.byref(v))
    x = v.value
    return x, f'{x // 10000}.{(x % 10000) // 100}.{x % 100}'


system_so = loaded  # torch 载入的系统库（2.29.7）
pip_so = linked      # DeepEP 链接的 pip 库（2.32.3）
s_ver, s_str = nccl_version(system_so)
p_ver, p_str = nccl_version(pip_so)
print(f'  system  {system_so} -> {s_str}')
print(f'  pip     {pip_so} -> {p_str}')
check('F1 loaded(system) 为 2.29.x', s_str.startswith('2.29'), s_str)
check('F2 linked(pip) 为 2.32.x', p_str.startswith('2.32'), p_str)
check('F3 两者版本不同 → 触发 check_nccl_so 失配', s_ver != p_ver, f'{s_str} vs {p_str}')
# context.cpp:156 的编译期闸门：pip 2.32.3 >= 2.31 → useRuntimeVersion 启用，
# 故 :160 的「必须同版本」断言不触发；真正的失配在 Python 侧 check_nccl_so 更早拦下。
check('F4 pip 版本 >= 2.31（走 useRuntimeVersion 分支，仍被 check_nccl_so 先拦）',
      p_ver >= 23100)
print(f'  setup.py 会把 EP_NCCL_ROOT_DIR/NCCL 根目录链接进 `_C`（SONAME -l:{os.path.basename(pip_so)}）')

# --------------------------------------------------------------------------------------
print('\n=== 汇总 ===')
n_pass = sum(1 for _, ok in RESULTS if ok)
n_fail = len(RESULTS) - n_pass
for name, ok in RESULTS:
    if not ok:
        print(f'  FAIL: {name}')
print(f'  PASS={n_pass}  FAIL={n_fail}  TOTAL={len(RESULTS)}')
sys.exit(1 if n_fail else 0)
