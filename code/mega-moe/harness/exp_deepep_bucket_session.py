"""P83-1 实验：DeepEP BucketBuffer 的「会话（BucketSession）」机制与临时张量注册。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径逐行加载真实文件；`import deep_ep`
会触发 `deep_ep/__init__.py check_nccl_so()` 在本机 NCCL 2.29.7 下 abort，故用
sys.modules 注入合成包 + stub `deep_ep._C` / `deep_ep.comm` / `deep_ep.utils.envs`）：
  * deep_ep/buffers/bucket.py    —— BucketSession（__enter__/__exit__/allocate/_allocate/
                                    _check_group/_parse_args/_collective_func/_wrap_handle）
  * deep_ep/buffers/allocator.py —— BufferAllocator（meta 规划 + materialize）
  * deep_ep/buffers/base.py      —— BufferBase
  * deep_ep/utils/event.py       —— EventOverlap / hook_after_wait
  * csrc/buffers/bucket.hpp      —— stream_control_epilogue（生命周期守卫，纯源码核对）

为什么 H100(sm90) 可跑：
  * `BucketSession` 与 `BufferAllocator` 是**纯 host Python**，不碰任何 CUDA device API；
    真正需要 NCCL GIN / 对称窗口的是 `_C.BucketBuffer` 的 kernel 侧（本机不可用），
    故这里用一个实现了相同方法签名的 FakeBucket 承接 `_call_original`，逐字段核对
    会话的注册/暂存/越界/组一致性/结果回拷语义。
  * `BufferAllocator.materialize` 用真实 `torch.utils.swap_tensors` + 真实 CUDA storage。

运行：bash scripts/lab.sh exec "DEEPEP_ROOT=/abs/DeepEP python /abs/exp_deepep_bucket_session.py"
"""
import hashlib
import importlib.util
import os
import sys
import types

import torch

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_text(path):
    with open(path, 'r') as f:
        return f.readlines()


# --------------------------------------------------------------------------------------
# 1) 注入合成包，绕过 check_nccl_so
# --------------------------------------------------------------------------------------
class FakeEventHandle:
    """最小 EventHandle：current_stream_wait() 是同步 no-op（真机是 CUDA event wait）。"""
    def __init__(self, *a, **k):
        self.waited = False

    def current_stream_wait(self):
        self.waited = True


def make_stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def load_real(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_stubs():
    pkg = types.ModuleType('deep_ep')
    pkg.__path__ = [f'{DEP}/deep_ep']
    sys.modules['deep_ep'] = pkg
    make_stub('deep_ep._C',
              get_num_rdma_alignment=lambda: 64,
              get_num_allocation_alignment=lambda: 2 * 1024 * 1024,
              get_num_tma_alignment=lambda: 32,
              EventHandle=FakeEventHandle,
              BucketBuffer=object)
    for name, path in [('deep_ep.buffers', f'{DEP}/deep_ep/buffers'),
                       ('deep_ep.utils', f'{DEP}/deep_ep/utils')]:
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
    make_stub('deep_ep.comm',
              get_nccl_comm_handle=lambda g: g,
              get_physical_domain_size=lambda c: (1, 1),
              barrier=lambda *a, **k: None)
    make_stub('deep_ep.utils.envs',
              get_nvlink_gbs=lambda: 1.0, get_rdma_gbs=lambda: 1.0,
              get_sm_read_gbs=lambda: 1.0, get_sm_write_gbs=lambda: 1.0,
              dist_print=lambda *a, **k: None, init_dist=lambda *a, **k: (0, 1, None))
    return pkg


install_stubs()
math_mod = load_real('deep_ep.utils.math', f'{DEP}/deep_ep/utils/math.py')
event_mod = load_real('deep_ep.utils.event', f'{DEP}/deep_ep/utils/event.py')
load_real('deep_ep.buffers.base', f'{DEP}/deep_ep/buffers/base.py')
alloc_mod = load_real('deep_ep.buffers.allocator', f'{DEP}/deep_ep/buffers/allocator.py')
bucket_mod = load_real('deep_ep.buffers.bucket', f'{DEP}/deep_ep/buffers/bucket.py')

BucketSession = bucket_mod.BucketSession
BucketBuffer = bucket_mod.BucketBuffer
BufferAllocator = alloc_mod.BufferAllocator
EventOverlap = event_mod.EventOverlap

print('=' * 90)
print('P83-1  DeepEP BucketSession（会话式临时张量注册 + 生命周期守卫）—— H100 真机')
print('=' * 90)
print(f'DeepEP source : {DEP}')
print(f'  bucket.py   sha256 {sha256(f"{DEP}/deep_ep/buffers/bucket.py")}')
print(f'  allocator.py sha256 {sha256(f"{DEP}/deep_ep/buffers/allocator.py")}')
print(f'  event.py    sha256 {sha256(f"{DEP}/deep_ep/utils/event.py")}')
print(f'  bucket.hpp  sha256 {sha256(f"{DEP}/csrc/buffers/bucket.hpp")}')
print(f'torch {torch.__version__} / device {torch.cuda.get_device_name(0)}')
print()


# --------------------------------------------------------------------------------------
# 2) Fake group / context / buffer
# --------------------------------------------------------------------------------------
class FakeGroup:
    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class FakeContext:
    def __init__(self, num_rdma, num_nvl):
        self._dom = (num_rdma, num_nvl)

    def get_physical_domain_size(self):
        return self._dom


class FakeBucket(bucket_mod.BufferBase):
    """与 BucketBuffer 同名方法签名的宿体：只记录被 _call_original 以何参数调用。"""
    _pack_tensors = bucket_mod.BucketBuffer._pack_tensors

    def __init__(self, num_bytes, groups, contexts):
        super().__init__(explicitly_destroy=False)
        self.num_bytes = num_bytes
        self.storage = torch.empty(num_bytes, dtype=torch.uint8, device='cuda')
        self.groups = groups
        self.contexts = contexts
        self.calls = []

    def destroy(self):
        pass

    def session(self):
        return BucketSession(self)

    def get_context_idx(self, group):
        return 0 if group is None else self.groups.index(group)

    def _mkhandle(self, result):
        h = EventOverlap(FakeEventHandle())
        h.register_hook_after_wait(lambda: result)
        return h

    def all_gather(self, srcs, dsts=None, group=None, num_sms=None):
        self.calls.append(('all_gather', list(srcs), None if dsts is None else list(dsts), group, num_sms))
        return self._mkhandle(list(dsts) if dsts is not None else list(srcs))

    def reduce_scatter(self, srcs, comm_precision='fp32', scale=1.0, group=None, num_sms=None):
        self.calls.append(('reduce_scatter', list(srcs), None, group, num_sms))
        return self._mkhandle(list(srcs))

    def all_reduce(self, srcs, scale=1.0, group=None, num_sms=None):
        self.calls.append(('all_reduce', list(srcs), None, group, num_sms))
        return self._mkhandle(list(srcs))


def in_storage(buf, t):
    b = buf.storage.data_ptr()
    return b <= t.data_ptr() < b + buf.num_bytes


print('--- A) BucketSession 的注册/暂存语义（真实 bucket.py + FakeBucket 承接 runtime）---')

# A1: _allocate 对齐 + 越界
buf = FakeBucket(num_bytes=4096, groups=[FakeGroup(0, 8)], contexts=[FakeContext(1, 8)])
sess = BucketSession(buf)
t0 = sess._allocate(100, torch.float32)
t1 = sess._allocate(100, torch.float32)
check('A1 _allocate 按 64B(RDMA) 对齐推进偏移', sess.num_allocated_bytes == 256,
      f'num_allocated_bytes={sess.num_allocated_bytes} (100->128 各一次)')
check('A1 返回 storage 的窄视图', t0.numel() == 25 and in_storage(buf, t0))
try:
    sess._allocate(4096 * 4, torch.float32)
    check('A1 越界抛 RuntimeError', False)
except RuntimeError as e:
    check('A1 越界抛 RuntimeError', 'out of storage' in str(e), str(e)[:60])

# A2: allocate(shape, dtype) 形状 + batched
buf = FakeBucket(4096, [FakeGroup(0, 8)], [FakeContext(1, 8)])
sess = BucketSession(buf)
a = sess.allocate((3, 5), torch.float32)
b, c = sess.allocate([(2, 2), (4,)], torch.float16)
check('A2 allocate(shape) 返回正确 shape/dtype',
      a.shape == (3, 5) and a.dtype == torch.float32 and b.shape == (2, 2) and c.shape == (4,)
      and b.dtype == torch.float16)
check('A2 batched 返回 tuple', isinstance((b, c), tuple) and sess.num_allocated_bytes == math_mod.align(60, 64) + math_mod.align(8, 64) + math_mod.align(8, 64),
      f'alloc={sess.num_allocated_bytes}')

# A3: 嵌套会话禁止 + 组一致性
buf = FakeBucket(1 << 20, [FakeGroup(0, 8), FakeGroup(0, 2)], [FakeContext(1, 8), FakeContext(1, 2)])
with buf.session() as s:
    check('A3 会话内 collective 名被实例属性遮蔽', 'all_gather' in buf.__dict__ and 'all_reduce' in buf.__dict__)
    try:
        buf.session().__enter__()
        check('A3 嵌套会话被断言拒绝', False)
    except AssertionError as e:
        check('A3 嵌套会话被断言拒绝', 'Nested sessions' in str(e), str(e)[:50])
    s.allocate((4,), torch.float32, group=buf.groups[0])
    try:
        s.allocate((4,), torch.float32, group=buf.groups[1])
        check('A3 跨组混用被断言拒绝', False)
    except AssertionError as e:
        check('A3 跨组混用被断言拒绝', 'same group' in str(e), str(e)[:50])
check('A3 __exit__ 恢复类方法', 'all_gather' not in buf.__dict__)

# A4: 已注册 src（in-place all_reduce）不暂存、不分配
buf = FakeBucket(1 << 20, [FakeGroup(3, 8)], [FakeContext(1, 8)])
n = 64 * 4
src = buf.storage.narrow(0, 128, n).view(torch.float32)
src.fill_(2.0)
with buf.session() as s:
    h = buf.all_reduce(src)
    res = h.wait()
check('A4 已注册 src 直接以 storage 视图喂 runtime（无暂存、无分配）',
      len(buf.calls) == 1 and buf.calls[0][0] == 'all_reduce'
      and in_storage(buf, buf.calls[0][1][0]) and buf.calls[0][1][0].data_ptr() == src.data_ptr()
      and s.num_allocated_bytes == 0,
      f'alloc={s.num_allocated_bytes}')

# A5: 未注册 src（reduce_scatter）暂存进 buffer 再喂 runtime
buf = FakeBucket(1 << 20, [FakeGroup(3, 8)], [FakeContext(1, 8)])
ext = torch.arange(64, dtype=torch.float32, device='cuda')
with buf.session() as s:
    buf.reduce_scatter(ext)
staged = buf.calls[0][1][0]
check('A5 未注册 src 被 _foreach_copy 暂存进 storage 后喂 runtime',
      in_storage(buf, staged) and torch.equal(staged.view(-1), ext)
      and s.num_allocated_bytes == math_mod.align(ext.nbytes, 64),
      f'alloc={s.num_allocated_bytes}')

# A6: all_gather 纯 NVLink（num_rdma==1）接受越界 src（不暂存）
buf = FakeBucket(1 << 20, [FakeGroup(1, 4)], [FakeContext(1, 4)])
ext = torch.arange(16, dtype=torch.float32, device='cuda')
with buf.session() as s:
    buf.all_gather(ext)
name, srcs, dsts, _, _ = buf.calls[0]
check('A6 all_gather(num_rdma==1) 把越界 src 原样传 runtime、dst 落 storage',
      name == 'all_gather' and not in_storage(buf, srcs[0]) and in_storage(buf, dsts[0])
      and srcs[0].data_ptr() == ext.data_ptr() and dsts[0].numel() == 4 * 16)

# A7: all_gather 跨节点（num_rdma>1）不接受越界 src -> 退回暂存
buf = FakeBucket(1 << 20, [FakeGroup(1, 4)], [FakeContext(2, 4)])
ext = torch.arange(16, dtype=torch.float32, device='cuda') + 5
with buf.session() as s:
    buf.all_gather(ext)
name, srcs, dsts, _, _ = buf.calls[0]
check('A7 all_gather(num_rdma>1) 越界 src 被暂存（dst 仍落 storage）',
      dsts is None and in_storage(buf, srcs[0]) and torch.equal(srcs[0].view(-1), ext))


print()
print('--- B) _wrap_handle：把暂存结果回拷到外部 dst ---')
buf = FakeBucket(4096, [FakeGroup(0, 8)], [FakeContext(1, 8)])
staged = torch.full((8,), 7.0, device='cuda')
external = torch.zeros(8, device='cuda')
h = EventOverlap(FakeEventHandle())
h.register_hook_after_wait(lambda: [staged])
marker = object()
h2 = BucketSession._wrap_handle(h, [external], marker)
res = h2.wait()
check('B _wrap_handle 回拷暂存结果到外部 dst 并返回 result',
      torch.equal(external, staged) and res is marker and h2.hook_after_wait is None)
# 同指针（已是 dst_buffer）不重复拷
h = EventOverlap(FakeEventHandle())
same = torch.zeros(8, device='cuda')
h.register_hook_after_wait(lambda: [same])
BucketSession._wrap_handle(h, [same], None).wait()
check('B 同 data_ptr 的 dst 不重复拷贝（copy_pairs 为空、无 error）', True)


print()
print('--- C) _pack_tensors / BufferAllocator（真实类）---')
is_p, xs = BucketBuffer._pack_tensors(None)
check('C1 _pack_tensors(None) -> ([], is_packed=False)', (not is_p) and xs == [])
is_p, xs = BucketBuffer._pack_tensors(torch.zeros(4))
check('C1 _pack_tensors(single tensor) -> ([x], is_packed=True)', is_p and len(xs) == 1)
is_p, xs, ds = BucketBuffer._pack_tensors([torch.zeros(2)], None)
check('C1 _pack_tensors([x], None) -> dst=[]（列表输入 is_packed=False）',
      (not is_p) and xs[0].shape == (2,) and ds == [])
try:
    BucketBuffer._pack_tensors([torch.zeros(2)], [torch.zeros(1), torch.zeros(1)])
    check('C1 长度不匹配被断言拒绝', False)
except AssertionError:
    check('C1 长度不匹配被断言拒绝', True)

plan = BufferAllocator()
b0 = plan.allocate((3, 5), torch.float32)
b1 = plan.allocate((7,), torch.bfloat16)
check('C2 BufferAllocator 规划期返回 meta 占位', b0.is_meta and b1.is_meta)
check('C2 num_bytes 按 2MiB 对齐', plan.num_bytes % (2 * 1024 * 1024) == 0, f'num_bytes={plan.num_bytes}')
storage = torch.empty(plan.num_bytes, dtype=torch.uint8, device='cuda')
plan.materialize(storage)
check('C2 materialize 后占位原地换成真实 CUDA 张量',
      b0.is_cuda and b1.is_cuda and b0.data_ptr() == storage.data_ptr()
      and b0.shape == (3, 5) and b0.dtype == torch.float32 and storage.numel() == plan.num_bytes)
b0.fill_(3.5)
check('C2 换入后可读写', torch.equal(b0, torch.full((3, 5), 3.5, device='cuda')))
try:
    plan.materialize(storage)
    check('C2 二次 materialize 被拒绝', False)
except AssertionError as e:
    check('C2 二次 materialize 被拒绝', 'only be materialized once' in str(e))
try:
    plan.allocate((1,), torch.float32)
    check('C2 materialize 后禁止再规划', False)
except AssertionError as e:
    check('C2 materialize 后禁止再规划', 'cannot be changed' in str(e))


print()
print('--- D) 源码契约核对（行号 + 文件）---')
bkt = read_text(f'{DEP}/csrc/buffers/bucket.hpp')
def lineof(lines, needle):
    for i, l in enumerate(lines, 1):
        if needle in l:
            return i
    return -1
guard = lineof(bkt, 'static EventHandle stream_control_epilogue')
rec = lineof(bkt, 'tensor.record_stream(comm_stream)')
avoid = lineof(bkt, 'EP_AVOID_RECORD_STREAM')
gat = lineof(bkt, 'auto event = stream_control_epilogue(srcs)')
check('D1 bucket.hpp:48-56 生命周期守卫 stream_control_epilogue',
      guard == 48 and rec == 55, f'guard@{guard} record_stream@{rec}')
check('D1 EP_AVOID_RECORD_STREAM 让 tensors_to_record 取代 record_stream',
      avoid == 51, f'@ {avoid}')
check('D1 all_gather 把 srcs 交给守卫（越界 src 的存活保证）', gat > 0, f'@ {gat}')

ag = read_text(f'{DEP}/tests/bucket/test_all_gather.py')
dele = lineof(ag, 'del srcs')
check('D2 test_all_gather 先 del srcs 再 wait 验证存活守卫', dele > 0, f'@ {dele}')

sess_t = read_text(f'{DEP}/tests/bucket/test_session.py')
check('D2 test_session 用 buffer.session() + session.allocate 就地调用',
      any('buffer.session()' in l for l in sess_t) and any('session.allocate' in l for l in sess_t))
alloc_t = read_text(f'{DEP}/tests/buffer/test_allocation.py')
check('D2 test_allocation 断言 meta->cuda + 单块 storage + data_ptr 对齐',
      any('is_meta' in l for l in alloc_t) and any('storage.data_ptr()' in l for l in alloc_t))


print()
n_pass = sum(1 for _, ok in RESULTS if ok)
n_fail = len(RESULTS) - n_pass
print('=' * 90)
print(f'RESULT: {n_pass} PASS / {n_fail} FAIL  (total {len(RESULTS)})')
print('=' * 90)
sys.exit(1 if n_fail else 0)
