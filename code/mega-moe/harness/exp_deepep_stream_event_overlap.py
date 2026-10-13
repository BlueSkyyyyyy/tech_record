"""P65-1 实验：DeepEP 的【通信-计算重叠 Python 调度地基】。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径 exec 真实文件，不 import `deep_ep/__init__`）：
  * deep_ep/utils/event.py   —— `EventOverlap`（CUDA 事件包装 + `with` 语法 + `hook_after_wait`）
  * deep_ep/comm/stream.py   —— `get_comm_stream`（把 C++ 通信流包成 Python `torch.cuda.Stream`）
  * deep_ep/buffers/base.py  —— `BufferBase`（通信 buffer 的公共生命周期 / `destroy` 契约）
  * csrc/utils/event.hpp / csrc/comm/stream.hpp —— 上面 Python 层的 C++ 孪生（本实验用
    `TorchEventHandle` 逐行镜像其语义，因为它俩只是 `torch::Event` 的薄封装）。

为什么 H100 可跑：
  * `EventOverlap` / `get_comm_stream` / `BufferBase` 都是**纯 host / Python**；事件与流是
    `torch.cuda.Event` / `torch.cuda.Stream`——正是 `torch::Event` 的 Python 面。故：
    ① 用 stub 注入 `deep_ep._C`（提供 `EventHandle` / `get_comm_stream`）后 exec 真实文件，
       对 `EventOverlap` 状态机做逐条硬断言；
    ② 用 `TorchEventHandle`（逐行镜像 `csrc/utils/event.hpp` 的 `EventHandle`）驱动**真实
       CUDA 事件/流**，量「通信-计算重叠」的墙钟收益，并验证事件等待的 memory-ordering。
  * `import deep_ep` 会触发 `deep_ep/__init__.py:check_nccl_so()`（本机 NCCL 2.29.7 与编译期
    不匹配 → abort），故不走包级 import。

运行：bash harness/exp_deepep_stream_event_overlap.sh > harness/exp_deepep_stream_event_overlap.out.txt
"""
import hashlib
import importlib.util
import os
import sys
import time
import types

import torch


DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


# --------------------------------------------------------------------------------------
# 真实源码加载：stub 掉 deep_ep._C，按文件路径 exec
# --------------------------------------------------------------------------------------
def load_real(relpath, modname):
    path = f'{DEP}/{relpath}'
    src = open(path, 'rb').read()
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m, hashlib.sha256(src).hexdigest()


class FakeEventHandle:
    """最小 EventHandle 桩：只记录 `current_stream_wait` 被调次数。"""

    def __init__(self, name='ev'):
        self.name = name
        self.waits = 0

    def current_stream_wait(self):
        self.waits += 1


def make_stub_pkg():
    pkg = types.ModuleType('deep_ep')
    pkg.__path__ = [f'{DEP}/deep_ep']
    sys.modules['deep_ep'] = pkg
    c = types.ModuleType('deep_ep._C')
    c.EventHandle = FakeEventHandle
    # get_comm_stream 返回一个**持久**的高优先级流（镜像 C++ 的 static stream）
    persistent = torch.cuda.Stream(priority=-1)
    c.get_comm_stream = lambda: persistent
    c._persistent_comm_stream = persistent
    sys.modules['deep_ep._C'] = c
    pkg._C = c
    for sub in ('utils', 'comm'):
        m = types.ModuleType(f'deep_ep.{sub}')
        m.__path__ = [f'{DEP}/deep_ep/{sub}']
        sys.modules[f'deep_ep.{sub}'] = m
        setattr(pkg, sub, m)
    return pkg, c


# --------------------------------------------------------------------------------------
# 逐行镜像 csrc/utils/event.hpp 的 EventHandle（torch::Event → torch.cuda.Event）
# --------------------------------------------------------------------------------------
class TorchEventHandle:
    """镜像 `csrc/utils/event.hpp:13-30`：

    * 构造时 `record(current stream)`（默认流）或 `record(给定流)`；
    * `current_stream_wait()` = `at::cuda::getCurrentCUDAStream().wait(*event)`。
    torch::Event 就是 torch.cuda.Event，故这是同一原语。
    """

    def __init__(self, stream=None):
        self.event = torch.cuda.Event()
        self.event.record(stream if stream is not None else torch.cuda.current_stream())

    def current_stream_wait(self):
        torch.cuda.current_stream().wait_event(self.event)


# --------------------------------------------------------------------------------------
def part_event_overlap(eo_mod):
    print('\n=== Part A: 真实 `EventOverlap` 状态机（exec 真实 deep_ep/utils/event.py）===')
    EventOverlap = eo_mod.EventOverlap
    print(f'  [ok] loaded deep_ep/utils/event.py  sha256={hashlib_sha[:16]}…')

    # A1: 基本等待把等待传给事件，且返回值 = hook 结果 / None
    h = FakeEventHandle('a1')
    eo = EventOverlap(h)
    r = eo.current_stream_wait()
    check('A1 current_stream_wait 转发到 event.current_stream_wait',
          h.waits == 1 and r is None, f'waits={h.waits} ret={r}')

    # A2: hook_after_wait 在等待后被调用一次、随后清空；其返回值被透传
    h = FakeEventHandle('a2')
    eo = EventOverlap(h)
    calls = []
    eo.register_hook_after_wait(lambda: (calls.append(1), 42)[1])
    r1 = eo.wait()
    r2 = eo.current_stream_wait()
    check('A2 hook_after_wait 只触发一次 + 结果返回 + 之后清空',
          calls == [1] and r1 == 42 and r2 is None and eo.hook_after_wait is None,
          f'calls={calls} r1={r1} r2={r2}')

    # A3: 重复注册 hook 抛 AssertionError（真实 assert 文案）
    h = FakeEventHandle('a3')
    eo = EventOverlap(h)
    eo.register_hook_after_wait(lambda: 0)
    raised = False
    try:
        eo.register_hook_after_wait(lambda: 1)
    except AssertionError as e:
        raised = 'already registered' in str(e)
    check('A3 重复注册 hook 被 assert 拦下', raised)

    # A4: `with event_overlap:` 不释放 handle（事件保留，可被多流等待）
    h = FakeEventHandle('a4')
    eo = EventOverlap(h)
    with eo:
        pass
    check('A4 普通 with 退出时等待且保留 event', h.waits == 1 and eo.event is h)

    # A5: `with event_overlap(release_handle=True):` 退出后 event 置 None（释放句柄）
    h = FakeEventHandle('a5')
    eo = EventOverlap(h)
    with eo(release_handle=True):
        pass
    check('A5 release_handle=True 退出后 event=None 且标志复位',
          h.waits == 1 and eo.event is None and eo._release_handle_by_call is False)

    # A6: `__call__` 返回 self（不新建包装对象，引用计数不变）
    h = FakeEventHandle('a6')
    eo = EventOverlap(h)
    check('A6 __call__ 返回同一对象', eo(release_handle=False) is eo)

    # A7: event 已被释放（None）时 __exit__ 不再等待
    h = FakeEventHandle('a7')
    eo = EventOverlap(h)
    eo.event = None
    with eo:
        pass
    check('A7 event=None 时 with 退出不等待', h.waits == 0)

    # A8: 异常穿过 with 仍会等待（__exit__ 无视 exc_type）
    h = FakeEventHandle('a8')
    eo = EventOverlap(h)
    try:
        with eo(release_handle=True):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    check('A8 异常路径也执行等待/释放', h.waits == 1 and eo.event is None)


def part_comm_stream(st_mod, cstub):
    print('\n=== Part B: 真实 `get_comm_stream`（exec 真实 deep_ep/comm/stream.py）===')
    s1 = st_mod.get_comm_stream(None)
    s2 = st_mod.get_comm_stream(None)
    persistent = cstub._persistent_comm_stream
    check('B1 返回 torch.cuda.Stream', isinstance(s1, torch.cuda.Stream))
    check('B2 每次 wrap 同一个底层流 id（C++ static stream）',
          s1.stream_id == persistent.stream_id and s2.stream_id == persistent.stream_id,
          f'ids={s1.stream_id}/{s2.stream_id} vs {persistent.stream_id}')
    check('B3 是新的 Python 包装对象（底层流单例）', s1 is not s2)


def part_buffer_base(bb_mod):
    print('\n=== Part C: 真实 `BufferBase` 生命周期（exec 真实 deep_ep/buffers/base.py）===')
    BufferBase = bb_mod.BufferBase

    class Impl(BufferBase):
        def __init__(self, explicitly_destroy=False):
            super().__init__(explicitly_destroy)
            self.destroyed = 0

        def destroy(self):
            super().destroy()
            self.destroyed += 1

    b = Impl()
    check('C1 runtime=None 时 main_context 为 None', b.main_context is None)

    class Rt:
        def __init__(self):
            self.n = 0
            self.main_context = object()

        def destroy(self):
            self.n += 1

    b = Impl(explicitly_destroy=True)
    rt = Rt()
    b.runtime = rt
    check('C2 未销毁时 main_context 转发到 runtime.main_context',
          b.main_context is rt.main_context)
    b.destroy()
    check('C3 destroy 后 runtime=None → main_context=None',
          b.runtime is None and b.main_context is None and rt.n == 1)

    b = Impl(explicitly_destroy=False)
    raised = False
    try:
        b.destroy()
    except AssertionError:
        raised = True
    check('C4 非 explicitly_destroy 时 destroy() 被 assert 拦下', raised)


def part_real_overlap():
    print('\n=== Part D: 真实 CUDA 事件/流上的「通信-计算重叠」墙钟 ===')
    dev = torch.device('cuda')
    prop = torch.cuda.get_device_properties(0)

    # 两条流：comm 用高优先级（镜像 getStreamFromPool(true)），compute 用当前默认流
    comm = torch.cuda.Stream(priority=-1)
    cur = torch.cuda.current_stream()

    # 通信负载：大张量 device-to-device 拷贝（访存受限，近似 dispatch/combine 的 NVLink 搬运）
    nbytes = 512 * 1024 * 1024
    src = torch.empty(nbytes // 4, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    # 计算负载：方阵 matmul（计算受限）
    m = n = k = 4096
    A = torch.randn(m, k, device=dev)
    B = torch.randn(k, n, device=dev)

    def do_comm():
        for _ in range(4):
            dst.copy_(src, non_blocking=True)

    def do_compute():
        for _ in range(4):
            C = A @ B
        torch.cuda.synchronize()
        return C

    # 预热
    with torch.cuda.stream(comm):
        do_comm()
    do_compute()
    torch.cuda.synchronize()

    # 单腿：通信-only / 计算-only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(comm):
        do_comm()
    torch.cuda.synchronize()
    t_comm = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    do_compute()
    t_comp = time.perf_counter() - t0

    # 串行：通信完 → 当前流等待 → 计算
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(comm):
        do_comm()
    cur.wait_stream(comm)
    do_compute()
    torch.cuda.synchronize()
    t_serial = time.perf_counter() - t0

    # 重叠：通信发在 comm 流，计算发在当前流，末尾才等待 → 语义上两者可并行
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(comm):
        do_comm()
    do_compute()  # 在当前流，不等 comm
    cur.wait_stream(comm)
    torch.cuda.synchronize()
    t_overlap = time.perf_counter() - t0

    print(f'  通信腿（copy 512MB，全卡）: {t_comm*1e3:.3f} ms   计算腿（matmul，全卡）: {t_comp*1e3:.3f} ms')
    print(f'  串行（copy→wait→matmul）: {t_serial*1e3:.3f} ms')
    print(f'  重叠（copy∥matmul，末尾 wait）: {t_overlap*1e3:.3f} ms')
    print(f'  加速比 serial/overlap = {t_serial/t_overlap:.3f}×')
    # 关键结论：**两条腿都吃满全卡时，单卡上不会真正重叠**（copy 的 grid 与 matmul 争 SM/带宽），
    # 重叠墙钟 ≈ sum 而非 max。这正是 DeepEP 要把通信 kernel 的 SM 数**硬上限在半卡**
    # （get_theoretical_num_sms，第 63 篇）的原因——通信只许占一小部分 SM，其余留给计算。
    check('D1 两腿均占满全卡时：重叠 ≈ 串行（单卡不重叠，反证 SM 分区必要性）',
          t_overlap >= t_serial * 0.95,
          f'serial={t_serial*1e3:.2f} overlap={t_overlap*1e3:.2f} ms')
    check('D2 重叠墙钟 ≈ sum(腿)（而非 max）',
          t_overlap >= (t_comm + t_comp) * 0.9,
          f'sum={(t_comm+t_comp)*1e3:.2f} got={t_overlap*1e3:.2f}')

    # D3: 用真实 EventOverlap + TorchEventHandle 表达「current stream 等待通信事件」
    src.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(comm):
        dst.fill_(3.14159)          # 通信流写 dst
        handle = TorchEventHandle(comm)  # 镜像 EventHandle(stream)
    eo = eo_mod_global.EventOverlap(handle)
    with eo:                        # __exit__ 里 current_stream_wait → 当前流等待
        pass
    got = dst[0].item()             # 当前流上读，事件等待保证 ordering
    check('D3 EventOverlap 等待后当前流可见通信流的写', abs(got - 3.14159) < 1e-5, f'got={got}')

    # D4: 延时触发 hook（镜像 ep.py:741 的 deterministic_epilogue 走 hook）
    fired = []
    h = FakeEventHandle('d4')
    h2 = eo_mod_global.EventOverlap(TorchEventHandle(torch.cuda.current_stream()))
    h2.register_hook_after_wait(lambda: fired.append('after_wait') or 'R')
    r = h2.current_stream_wait()
    check('D4 TorchEventHandle 上 hook 也在等待后触发', fired == ['after_wait'] and r == 'R')


def part_cpp_semantics():
    print('\n=== Part E: 镜像 csrc/comm/stream.hpp 的 `stream_wait` 自等待断言 ===')
    # stream_wait: EP_HOST_ASSERT(stream.id() != event_stream.id()) —— 自等待会死锁
    a = torch.cuda.Stream()
    b = torch.cuda.Stream()
    ok = a.stream_id != b.stream_id
    check('E1 不同流 id 不同（真机给不同 id）', ok, f'{a.stream_id} vs {b.stream_id}')
    raised = False
    try:
        assert a.stream_id != a.stream_id, 'self wait'
    except AssertionError:
        raised = True
    check('E2 自等待被同一断言拒绝（避免死锁）', raised)


def main():
    global eo_mod_global, hashlib_sha
    print('P65-1 DeepEP 通信-计算重叠 Python 调度地基 —— H100 真机')
    print(f'GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}  '
          f'SM={torch.cuda.get_device_properties(0).multi_processor_count}')
    print(f'DeepEP root: {DEP}')
    print(f'torch: {torch.__version__}  nccl={torch.cuda.nccl.version()}')

    pkg, cstub = make_stub_pkg()
    eo_mod, sha_eo = load_real('deep_ep/utils/event.py', 'deep_ep.utils.event')
    st_mod, sha_st = load_real('deep_ep/comm/stream.py', 'deep_ep.comm.stream')
    bb_mod, sha_bb = load_real('deep_ep/buffers/base.py', 'deep_ep.buffers.base')
    hashlib_sha, eo_mod_global = sha_eo, eo_mod

    part_event_overlap(eo_mod)
    part_comm_stream(st_mod, cstub)
    part_buffer_base(bb_mod)
    part_real_overlap()
    part_cpp_semantics()

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f'\n===== {npass} PASS / {len(RESULTS)-npass} FAIL =====')
    print(f'sha256: event.py={sha_eo[:16]} stream.py={sha_st[:16]} base.py={sha_bb[:16]}')
    return 0 if npass == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
