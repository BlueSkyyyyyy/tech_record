"""P113-1 实验：DeepEP 的 Python 缓存基座 —— `weak_lru` 弱引用方法缓存 + 模块级 `lru_cache`。

目标源码（`~/github/DeepEP@93eb6eb`，`deep_ep.__version__ == 2.5.0`）：
  * `deep_ep/utils/semantic.py`（27 行）
      - `value_or:5-6`（**仅 `None` 才回退默认值**）
      - `weak_lru:9-26`（**唯一一个方法级缓存**；`functools.lru_cache` 直接修饰实例方法会
        把 `self` 钉在缓存里导致实例永生，故先用 `weakref.ref(self)` 当 key）
  * 两个 `weak_lru` 消费点（`deep_ep/buffers/ep.py`）：
      - `:430-431  get_theoretical_num_sms(...)`（`maxsize=None`）
      - `:845-846  lb_get_theoretical_num_sms()`（`maxsize=None`）
  * 对照：模块级 / 静态方法级 `@functools.lru_cache()`（**没有一个用 `weak_lru`**）：
      - `deep_ep/utils/envs.py:193/200/207/237/260`（`get_sm_read_gbs` 等 5 个）
      - `deep_ep/utils/find_pkgs.py:58`（`find_nccl_root`）
      - `deep_ep/buffers/engram.py:130`（`@staticmethod @functools.lru_cache(maxsize=None)`）

本脚本（纯 Python，架构无关）**按文件路径直接加载真实 `semantic.py` / `find_pkgs.py` /
`envs.py` 源码**（`envs.py` 用 `sys.modules` 注入合成 `deep_ep._C` / `deep_ep.comm` 绕过
`check_nccl_so`，与第 118 篇同法），然后：
  A. `value_or` 的五条语义（`None`/`0`/`False`/`''`）；
  B. 用标准库 **复现泄漏**：`functools.lru_cache` 修饰实例方法 → 实例永生；
     `weak_lru` → 实例可回收；
  C. `weak_lru` 缓存键语义：命中 / 不同实例不同条 / **`weakref.ref` 相等性委托**
     （值相等但不同的实例会撞成同一条 → 结果错用）；死引用条目残留与 `maxsize` 驱逐；
     `typed` 开关；不可弱引用对象（`__slots__`）抛 `TypeError`；
  D. **全库普查**：解析 `deep_ep/**/*.py` 的 `lru_cache`/`weak_lru` 装饰器，按
     「模块函数 / 静态方法 / 实例方法」分类（解释为什么只有实例方法需要 `weak_lru`）；
  E. 真实模块级缓存：`find_nccl_root` / `get_sm_read_gbs` / `get_nvlink_gbs`
     的命中与「**缓存会吞掉环境变化，要 `cache_clear()`**」；
  F. 用真实 `weak_lru` 复刻 `lb_get_theoretical_num_sms` 的「同参只算一次」。

运行：bash harness/exp_deepep_python_caching.sh
"""
import functools
import hashlib
import importlib.util
import math
import os
import sys
import types
import weakref

import torch  # noqa: F401  —— 真实 torch（加载 envs.py 需要；本实验本身不用 CUDA）

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


def find_lru(fn):
    """从 `weak_lru` 返回的 `inner` 的闭包里取出内部 `functools.lru_cache` 包装器。"""
    for cell in fn.__closure__ or ():
        obj = cell.cell_contents
        if isinstance(obj, functools._lru_cache_wrapper):
            return obj
    return None


def load_real(path, name, stubs=None):
    for mod_name, attrs in (stubs or {}).items():
        m = types.ModuleType(mod_name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[mod_name] = m
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 0. 环境坐标 + 加载真实源码
# --------------------------------------------------------------------------- #
print('== 环境坐标 ==')
print(f'  DeepEP root: {DEP}')
sem_path = os.path.join(DEP, 'deep_ep', 'utils', 'semantic.py')
fp_path = os.path.join(DEP, 'deep_ep', 'utils', 'find_pkgs.py')
envs_path = os.path.join(DEP, 'deep_ep', 'utils', 'envs.py')
ep_path = os.path.join(DEP, 'deep_ep', 'buffers', 'ep.py')
engram_path = os.path.join(DEP, 'deep_ep', 'buffers', 'engram.py')
for p in (sem_path, fp_path, envs_path, ep_path, engram_path):
    print(f'  {os.path.basename(p):14s} sha256={sha256(p)[:16]}')
print(f'  python {sys.version.split()[0]}')

# deep_ep 包壳 + _C/comm stub（绕过 check_nccl_so）
_deep_ep = types.ModuleType('deep_ep')
_deep_ep.__path__ = [os.path.join(DEP, 'deep_ep')]
sys.modules['deep_ep'] = _deep_ep
sys.modules['deep_ep._C'] = types.ModuleType('deep_ep._C')
sys.modules['deep_ep.comm'] = types.ModuleType('deep_ep.comm')
_utils = types.ModuleType('deep_ep.utils')
_utils.__path__ = [os.path.join(DEP, 'deep_ep', 'utils')]
sys.modules['deep_ep.utils'] = _utils
sys.modules.setdefault('deep_ep.utils.math', types.ModuleType('deep_ep.utils.math'))

sem = load_real(sem_path, 'deep_ep.utils.semantic')
print(f'  加载成功：semantic.{sorted(n for n in dir(sem) if not n.startswith("_"))}')


# --------------------------------------------------------------------------- #
# A. value_or：只有 None 才回退
# --------------------------------------------------------------------------- #
print()
print('== Part A: value_or 语义（仅 None 回退）==')
check('value_or(None, 7) == 7', sem.value_or(None, 7) == 7)
check('value_or(0, 7) == 0（0 是合法值）', sem.value_or(0, 7) == 0)
check('value_or(False, 7) is False', sem.value_or(False, 7) is False)
check("value_or('', 7) == ''（空串合法）", sem.value_or('', 7) == '')
check('value_or(None, None) is None', sem.value_or(None, None) is None)


# --------------------------------------------------------------------------- #
# B. 复现泄漏：plain lru_cache 钉死实例 vs weak_lru 可回收
# --------------------------------------------------------------------------- #
print()
print('== Part B: 方法级 lru_cache 的泄漏 vs weak_lru 的可回收 ==')


class PlainCached:
    @functools.lru_cache(maxsize=None)
    def value(self, x):
        return x + 1


class WeakCached:
    @sem.weak_lru(maxsize=None)
    def value(self, x):
        return x + 1


p = PlainCached()
rp = weakref.ref(p)
p.value(1)
del p
import gc  # noqa: E402

gc.collect()
check('plain lru_cache：del + gc 后实例仍存活（被缓存钉住）', rp() is not None)
check('plain lru_cache：缓存条目永久占位 currsize==1',
      PlainCached.value.cache_info().currsize == 1,
      f'{PlainCached.value.cache_info()}')

w = WeakCached()
rw = weakref.ref(w)
w.value(1)
del w
gc.collect()
check('weak_lru：del + gc 后实例被回收', rw() is None)
check('weak_lru 保留 functools.wraps 的 __name__', WeakCached.value.__name__ == 'value')
check('weak_lru 暴露 __wrapped__（原函数）', callable(WeakCached.value.__wrapped__))


# --------------------------------------------------------------------------- #
# C. weak_lru 缓存键语义
# --------------------------------------------------------------------------- #
print()
print('== Part C: weak_lru 缓存键语义 ==')


class Id:
    # 默认 __eq__ / __hash__ = 身份
    @sem.weak_lru(maxsize=None)
    def f(self, x):
        return id(self) * 1000 + x


a1, a2 = Id(), Id()
r = a1.f(5)
check('同实例同参 → 命中（结果一致）', a1.f(5) == r)
check('同实例不同参 → 不同结果', a1.f(6) != r)
check('不同（身份）实例同参 → 互不串用', a2.f(5) != r)
ci = find_lru(Id.f).cache_info()
check('不同实例各占一条：misses==3', ci.misses == 3, f'{ci}')

# weakref.ref 的相等性委托给 referent：值相等但不同的对象会撞成同一条
class Eq:
    def __init__(self, v):
        self.v = v

    def __eq__(self, o):
        return isinstance(o, Eq) and self.v == o.v

    def __hash__(self):
        return hash(self.v)

    @sem.weak_lru(maxsize=None)
    def probe(self):
        return id(self)


e1, e2 = Eq(1), Eq(1)
check('weakref.ref 相等性委托：值相等实例 hash/eq 相同',
      weakref.ref(e1) == weakref.ref(e2) and hash(weakref.ref(e1)) == hash(weakref.ref(e2)))
check('【陷阱】值相等但不同的实例撞同一条 → e2 拿到 e1 的结果',
      e1.probe() == e2.probe() and e1.probe() == id(e1))

# 死引用条目残留 + maxsize 驱逐
class Bounded:
    @sem.weak_lru(maxsize=4)
    def f(self):
        return 1


for _ in range(20):
    b = Bounded()
    b.f()
    del b
gc.collect()
cib = find_lru(Bounded.f).cache_info()
check('死引用条目不会自动清：currsize 受 maxsize 限制 <=4', cib.currsize <= 4, f'{cib}')
check('20 个实例 20 次 miss（缓存键按实例不同）', cib.misses == 20, f'{cib}')

# typed
class Typed:
    @sem.weak_lru(maxsize=None, typed=True)
    def f(self, x):
        return type(x).__name__


t = Typed()
check('typed=True：1 与 1.0 区分', t.f(1) == 'int' and t.f(1.0) == 'float')

class Untyped:
    @sem.weak_lru(maxsize=None, typed=False)
    def f(self, x):
        return type(x).__name__


u = Untyped()
check('typed=False：1 与 1.0 等价共享（短路于第一个）', u.f(1) == u.f(1.0))

# 不可弱引用对象
class NoWeak:
    __slots__ = ('v',)

    @sem.weak_lru()
    def f(self):
        return 1


try:
    NoWeak().f()
    check('__slots__（无 __weakref__）应抛 TypeError', False)
except TypeError as exc:
    check('__slots__（无 __weakref__）抛 TypeError', 'weak reference' in str(exc), f'={exc}')


# --------------------------------------------------------------------------- #
# D. 全库普查：lru_cache / weak_lru 的三种落点
# --------------------------------------------------------------------------- #
print()
print('== Part D: deep_ep 全库缓存装饰器普查 ==')
import re  # noqa: E402

census = []  # (relpath, lineno, kind, deco)
for dp, dn, fn in os.walk(os.path.join(DEP, 'deep_ep')):
    if '__pycache__' in dp:
        continue
    for f in fn:
        if not f.endswith('.py'):
            continue
        path = os.path.join(dp, f)
        rel = os.path.relpath(path, DEP)
        lines = open(path, encoding='utf-8').read().split('\n')
        for i, line in enumerate(lines):
            if '@weak_lru' in line:
                kind = 'method(weak)'
                census.append((rel, i + 1, kind, line.strip()))
            elif '@functools.lru_cache' in line:
                if rel.endswith('semantic.py'):
                    # 这是 weak_lru 内部「怎么建缓存」的定义行，不是一次使用
                    census.append((rel, i + 1, 'definition', line.strip()))
                    continue
                # 看是否紧邻 @staticmethod / 是否在 class 内（缩进）
                static = i > 0 and '@staticmethod' in lines[i - 1]
                in_class = line.startswith('    ') and not line.startswith('        ')
                kind = 'staticmethod' if static else ('module' if not line.startswith(' ') else 'method(plain?)')
                census.append((rel, i + 1, kind, line.strip()))
# 定义处（semantic.py 的 @functools.lru_cache 参数）单列
print('  relpath:line  kind  decorator')
for rel, ln, kind, deco in census:
    print(f'  {rel}:{ln}  {kind}  {deco}')

weak_uses = [c for c in census if c[2] == 'method(weak)']
module_uses = [c for c in census if c[2] == 'module' and 'semantic.py' not in c[0]]
static_uses = [c for c in census if c[2] == 'staticmethod']
check('weak_lru 仅 2 处消费（ep.py 两个方法）', len(weak_uses) == 2, f'={[(r,l) for r,_,l,_ in weak_uses]}')
check('模块级 lru_cache 恰 6 处（envs 5 + find_pkgs 1）', len(module_uses) == 6, f'={[(r,l) for r,_,l,_ in module_uses]}')
check('静态方法级 lru_cache 恰 1 处（engram.py）', len(static_uses) == 1, f'={[(r,l) for r,_,l,_ in static_uses]}')
check('没有任何「实例方法直接用 functools.lru_cache」',
      all(c[2] != 'method(plain?)' for c in census))


# --------------------------------------------------------------------------- #
# E. 真实模块级缓存：命中 + 吞掉环境变化
# --------------------------------------------------------------------------- #
print()
print('== Part E: 真实模块级 lru_cache 的命中与副作用 ==')
fp = load_real(fp_path, 'deep_ep.utils.find_pkgs')
envs = load_real(envs_path, 'deep_ep.utils.envs')

check('find_nccl_root 带 lru_cache', hasattr(fp.find_nccl_root, 'cache_info'))
root1 = fp.find_nccl_root(optional=True)
root2 = fp.find_nccl_root(optional=True)
check('find_nccl_root 第二次命中（hits==1）', fp.find_nccl_root.cache_info().hits >= 1,
      f'{fp.find_nccl_root.cache_info()}')
check('find_nccl_root 返回稳定', root1 == root2, f'={root1}')
if root1:
    lib = fp.get_nccl_lib_name(root1)
    check('get_nccl_lib_name 返回 libnccl.so*', lib.startswith('libnccl.so'), f'={lib}')

check('get_sm_read_gbs()==180', envs.get_sm_read_gbs() == 180)
check('get_sm_write_gbs()==45', envs.get_sm_write_gbs() == 45)
check('get_sm_read_gbs 带 lru_cache（hits>=0）', hasattr(envs.get_sm_read_gbs, 'cache_info'))

# 缓存吞掉环境变化：patch subprocess.run 喂两份不同的 nvidia-smi 输出
real_run = envs.subprocess.run


class FakeCompleted:
    def __init__(self, out):
        self.stdout = out


def make_fake(gbs_values):
    def fake_run(cmd, *a, **k):
        # 与真实 nvidia-smi 一样以换行结尾（envs.py 的正则用 `^$` 定块界，缺尾换行会不匹配）
        block = 'GPU 0:\n' + '\n'.join(f'    Link {i}: {v} GB/s' for i, v in enumerate(gbs_values)) + '\n'
        return FakeCompleted(block)
    return fake_run


try:
    envs.get_nvlink_gbs.cache_clear()
    envs.subprocess.run = make_fake([100.0, 100.0])
    v1 = envs.get_nvlink_gbs(1.0)   # factor=1 便于断言
    envs.subprocess.run = make_fake([10.0, 10.0])
    v2 = envs.get_nvlink_gbs(1.0)
    check('nvlink 首次 = 200（2×100, factor=1）', abs(v1 - 200.0) < 1e-6, f'={v1}')
    check('【陷阱】换了环境输出仍返回旧值（缓存未清）', abs(v2 - 200.0) < 1e-6, f'={v2}')
    envs.get_nvlink_gbs.cache_clear()
    v3 = envs.get_nvlink_gbs(1.0)
    check('cache_clear() 后才读到新值 20', abs(v3 - 20.0) < 1e-6, f'={v3}')
finally:
    envs.subprocess.run = real_run
    envs.get_nvlink_gbs.cache_clear()


# --------------------------------------------------------------------------- #
# F. 用真实 weak_lru 复刻 lb_get_theoretical_num_sms：同参只算一次
# --------------------------------------------------------------------------- #
print()
print('== Part F: 真实 weak_lru 复刻 lb_get_theoretical_num_sms ==')


class FakeBuffer:
    calls = 0

    @sem.weak_lru(maxsize=None)
    def lb_get_theoretical_num_sms(self, nvlink_gbs, sm_read_gbs, sm_write_gbs):
        FakeBuffer.calls += 1
        raw = max(nvlink_gbs / sm_read_gbs, nvlink_gbs / sm_write_gbs)
        # 真实实现里 align(x, 2) = ceil 到偶数
        return int(math.ceil(math.ceil(raw) / 2) * 2)


fb = FakeBuffer()
r1 = fb.lb_get_theoretical_num_sms(430.3, 180.0, 45.0)
r2 = fb.lb_get_theoretical_num_sms(430.3, 180.0, 45.0)
check('同参第二次命中（只计算一次）', FakeBuffer.calls == 1, f'calls={FakeBuffer.calls}')
check('结果一致且为偶数', r1 == r2 and r1 % 2 == 0, f'={r1}')
r3 = fb.lb_get_theoretical_num_sms(430.3, 180.0, 30.0)
check('改一个参 → 新条目再算一次', FakeBuffer.calls == 2, f'calls={FakeBuffer.calls}')
check('写带宽变小 → SM 数不变大（ceil(430.3/30)=15→16）', r3 == 16, f'={r3}')


# --------------------------------------------------------------------------- #
print()
print('== 汇总 ==')
print(f'  FAILS = {len(FAILS)}')
for f in FAILS:
    print(f'    [FAIL] {f}')
print('  ALL PASS' if not FAILS else '  SOME FAILED')
sys.exit(1 if FAILS else 0)
