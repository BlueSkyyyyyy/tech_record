"""P73-1 实验：DeepEP 的 `EPHandle` 句柄契约 + `EPBuffer.dispatch`/`combine` 的 host 编排层。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径 exec 真实文件，不 import `deep_ep/__init__`）：
  * deep_ep/buffers/ep.py       —— `EPHandle`（dispatch↔combine↔反向的句柄）、`EPBuffer.dispatch`/`combine`
  * deep_ep/buffers/allocator.py —— `BufferAllocator`（顺带）
  * deep_ep/utils/semantic.py   —— `value_or` / `weak_lru`
  * deep_ep/utils/math.py       —— `align` / `ceil_div` / `safe_div`
  * deep_ep/utils/event.py      —— `EventOverlap`（deferred epilogue 的载体）
  * deep_ep/utils/envs.py       —— `check_torch_deterministic`

为什么 H100 可跑：
  * `EPHandle` 是**纯 Python 数据容器**；`dispatch`/`combine` 的 host 编排（参数归一化、auto SM/QP、
    句柄复用断言、deferred epilogue、确定性排序钩子接线）都在 Python 里，**与 NCCL 设备 API 无关**。
  * 真机把 `EPBuffer.dispatch`/`combine` 这两个**未修改的真实方法体**绑到一个假 buffer 上（假
    `runtime` 记录参数并回放合成 result），即可逐字段核对编排是否正确；句柄里的张量用
    **真实 CUDA 张量**（H100）跑 `deterministic_sort`。
  * `import deep_ep` 会触发 `deep_ep/__init__.py:check_nccl_so()`（本机 NCCL 2.29.7 与编译期
    不匹配 → abort），故用 stub 注入 `deep_ep._C` 后按文件路径 exec。

运行：bash harness/exp_deepep_epbuffer_handle.sh > harness/exp_deepep_epbuffer_handle.out.txt
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
SHA = {}


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


# --------------------------------------------------------------------------------------
# 真实源码加载：stub 掉 deep_ep._C / deep_ep.comm，按文件路径 exec
# --------------------------------------------------------------------------------------
class FakeEventHandle:
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
    c.get_num_allocation_alignment = lambda: 512
    sys.modules['deep_ep._C'] = c
    pkg._C = c

    for sub in ('utils', 'comm', 'buffers'):
        m = types.ModuleType(f'deep_ep.{sub}')
        m.__path__ = [f'{DEP}/deep_ep/{sub}']
        sys.modules[f'deep_ep.{sub}'] = m
        setattr(pkg, sub, m)

    comm = sys.modules['deep_ep.comm']
    comm.barrier = lambda *a, **k: None
    comm.get_comm_stream = lambda: torch.cuda.Stream()
    comm.get_physical_domain_size = lambda: (1, 1)
    comm.get_logical_domain_size = lambda: (1, 1)
    comm.get_nccl_comm_handle = lambda group: types.SimpleNamespace(get=lambda: 0)
    return pkg, c


def load_real(relpath, modname):
    path = f'{DEP}/{relpath}'
    src = open(path, 'rb').read()
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    SHA[modname] = hashlib.sha256(src).hexdigest()
    return m


# --------------------------------------------------------------------------------------
# 假 buffer：把真实未修改的方法体绑上去，假 runtime 记录参数
# --------------------------------------------------------------------------------------
class FakeRuntime:
    def __init__(self, kind='dispatch'):
        self.kind = kind
        self.calls = []
        self.result = None
        self.event = FakeEventHandle('rt')
        self.deferred = None

    def dispatch(self, *args):
        self.calls.append(args)
        return self.result, self.event, self.deferred

    def combine(self, *args):
        self.calls.append(args)
        return self.result, self.event, self.deferred


def make_fake_ep(EPBuffer, *, deterministic=False, allow_hybrid_mode=False,
                 num_allocated_qps=17, default_max_tokens=16,
                 sms_fn=None):
    class FakeEP:
        pass

    FakeEP.dispatch = EPBuffer.dispatch
    FakeEP.combine = EPBuffer.combine
    FakeEP.get_theoretical_num_qps = EPBuffer.get_theoretical_num_qps
    FakeEP._unpack_handle = staticmethod(EPBuffer._unpack_handle)
    FakeEP._unpack_bias = staticmethod(EPBuffer._unpack_bias)

    ep = FakeEP()
    ep.deterministic = deterministic
    ep.allow_hybrid_mode = allow_hybrid_mode
    ep.num_allocated_qps = num_allocated_qps
    ep.num_max_tokens_per_rank = default_max_tokens
    ep.sms_calls = []
    if sms_fn is None:
        def sms_fn(num_experts, num_topk):
            ep.sms_calls.append((num_experts, num_topk))
            return 64
    ep.get_theoretical_num_sms = sms_fn
    ep.runtime = FakeRuntime()
    return ep


def make_handle(EPHandle, *, do_expand=False, num_experts=8, expert_alignment=1,
                num_max_tokens_per_rank=16, num_sms=64, topk_idx=None,
                num_recv_tokens=4, num_expanded_tokens=4, psum_rank=None,
                psum_expert=None, unaligned=None, recv_src_metadata=None,
                dst_buffer_slot_idx=None):
    topk_idx = topk_idx if topk_idx is not None else torch.tensor([[1, 2], [3, 0], [2, 1], [0, 3]])
    psum_rank = psum_rank if psum_rank is not None else torch.tensor([2, 4])
    psum_expert = psum_expert if psum_expert is not None else torch.tensor([1, 2, 3, 4])
    unaligned = unaligned if unaligned is not None else torch.tensor([1, 1, 1, 1])
    recv_src_metadata = recv_src_metadata if recv_src_metadata is not None else torch.zeros(4, 3, dtype=torch.int64)
    dst_buffer_slot_idx = dst_buffer_slot_idx if dst_buffer_slot_idx is not None else torch.zeros(4, dtype=torch.int64)
    return EPHandle(do_expand, num_experts, expert_alignment, num_max_tokens_per_rank, num_sms,
                    topk_idx, num_recv_tokens, num_expanded_tokens,
                    [1, 1, 1, 1], psum_rank, psum_expert, unaligned,
                    recv_src_metadata, dst_buffer_slot_idx,
                    None, None)


def make_dispatch_result(EPHandle, with_sf=True):
    """构造 `runtime.dispatch` 回放的 14 元组（与 `finalize_dispatch` 解包顺序一致）。"""
    recv_x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    recv_sf = torch.full((4, 1), 0.5) if with_sf else None
    recv_topk_idx = torch.tensor([[1, 2], [3, 0], [2, 1], [0, 3]])
    recv_topk_weights = torch.full((4, 2), 0.5)
    num_recv_tokens, num_expanded_tokens = 4, 4
    per_expert_list = [1, 1, 1, 1]
    psum_rank = torch.tensor([2, 4])
    psum_expert = torch.tensor([1, 2, 3, 4])
    unaligned = torch.tensor([1, 1, 1, 1])
    recv_src_metadata = torch.zeros(4, 3, dtype=torch.int64)
    dst_buffer_slot_idx = torch.zeros(4, dtype=torch.int64)
    token_metadata_at_forward = None
    channel_linked_list = None
    return (recv_x, recv_sf, recv_topk_idx, recv_topk_weights,
            num_recv_tokens, num_expanded_tokens, per_expert_list,
            psum_rank, psum_expert, unaligned, recv_src_metadata,
            dst_buffer_slot_idx, token_metadata_at_forward, channel_linked_list)


# --------------------------------------------------------------------------------------
def part_ephandle(EPHandle):
    print('\n=== Part A: `EPHandle` 句柄契约（纯 Python 数据容器 + topk_idx 版本守卫）===')
    topk = torch.tensor([[1, 2], [3, 0], [2, 1], [0, 3]])
    h = make_handle(EPHandle, topk_idx=topk)
    check('A1 构造存下所有字段（do_expand/num_experts/expert_alignment/num_sms）',
          h.do_expand is False and h.num_experts == 8 and h.expert_alignment == 1 and h.num_sms == 64)
    check('A2 `topk_idx` 是 property、返回原张量', h.topk_idx is topk)
    check('A3 `topk_idx` 被记录版本号（非 inference 张量）', h._topk_idx_version == topk._version)
    check('A4 `cached_recv_src_metadata_before_sort` 初始为 None',
          h.cached_recv_src_metadata_before_sort is None)

    # 原地修改 topk_idx → 版本号变化 → 访问 property 抛断言（防「handle 使用中改 topk」）
    before = h._topk_idx_version
    topk[0, 0] = 5
    raised, msg = False, ''
    try:
        _ = h.topk_idx
    except AssertionError as e:
        raised, msg = True, str(e)
    check('A5 原地改 topk_idx 后被版本守卫拦下', raised and 'must not be modified' in msg,
          f'v {before}->{topk._version}')

    # inference 张量没有版本计数器 → 守卫退化为「不检查」
    with torch.inference_mode():
        it = torch.tensor([[0, 1], [2, 3]])
        hi = make_handle(EPHandle, topk_idx=it)
    check('A6 inference 张量 `_topk_idx_version is None`（守卫关闭）',
          hi._topk_idx_version is None and hi.topk_idx is it)

    raised = False
    try:
        EPHandle(False, 8, 1, 16, 64, None, 4, 4, [1], torch.tensor([4]), torch.tensor([4]),
                 torch.tensor([4]), torch.zeros(4, 3, dtype=torch.int64),
                 torch.zeros(4, dtype=torch.int64), None, None)
    except AssertionError:
        raised = True
    check('A7 topk_idx=None 被 `assert topk_idx is not None` 拒绝', raised)


def part_static(EPBuffer, EPHandle):
    print('\n=== Part B: 静态辅助（`_unpack_handle` / `_unpack_bias` / `align` / `value_or`）===')
    out = EPBuffer._unpack_handle(None)
    check('B1 `_unpack_handle(None)` 给出 10 个 None', out == (None,) * 10, f'n={len(out)}')

    h = make_handle(EPHandle)
    out = EPBuffer._unpack_handle(h)
    check('B2 `_unpack_handle(h)` 按固定顺序取 10 个缓存字段',
          out[0] == h.num_recv_tokens and out[1] == h.num_expanded_tokens
          and out[2] is h.num_recv_tokens_per_expert_list
          and out[3] is h.psum_num_recv_tokens_per_scaleup_rank
          and out[4] is h.psum_num_recv_tokens_per_expert
          and out[5] is h.num_unaligned_recv_tokens_per_expert
          and out[6] is h.dst_buffer_slot_idx
          and out[7] is h.token_metadata_at_forward
          and out[8] is h.recv_src_metadata and out[9] is h.channel_linked_list)

    b = torch.arange(6).reshape(2, 3)
    check('B3 `_unpack_bias(None)` → (None,None)', EPBuffer._unpack_bias(None) == (None, None))
    check('B4 `_unpack_bias(tensor)` → (b,None)', EPBuffer._unpack_bias(b) == (b, None))
    b0, b1 = torch.zeros(1), torch.ones(1)
    check('B5 `_unpack_bias((b0,b1))` → (b0,b1)', EPBuffer._unpack_bias((b0, b1)) == (b0, b1))
    raised = False
    try:
        EPBuffer._unpack_bias((b0, b1, b0))
    except AssertionError:
        raised = True
    check('B6 长度≠2 的 tuple 被 assert 拒绝', raised)


def part_dispatch(EPBuffer, EPHandle):
    print('\n=== Part C: `EPBuffer.dispatch` host 编排（auto SM/QP / SF 解包 / 句柄复用 / deferred）===')

    # C1 auto SM：num_sms=0 → 调 get_theoretical_num_sms(num_experts,num_topk)
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    x = torch.zeros(4, 2)
    recv_x, _, _, handle, eo = ep.dispatch(x, topk_idx=torch.tensor([[1, 2]] * 4), num_experts=8)
    check('C1a num_sms=0 → 自动调用 `get_theoretical_num_sms(num_experts=8, num_topk=2)`',
          ep.sms_calls == [(8, 2)] and ep.runtime.calls[0][18] == 64, f'sms={ep.sms_calls}')
    check('C1b 返回 5 元组且末位是 EventOverlap；handle 类型正确',
          len((recv_x, _, _, handle, eo)) == 5 and isinstance(handle, EPHandle) and eo.__class__.__name__ == 'EventOverlap')

    # C2 显式 num_sms 对齐到 2
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    ep.dispatch(x, topk_idx=torch.tensor([[1, 2]] * 4), num_experts=8, num_sms=7)
    check('C2 显式 num_sms=7 → align 到 8', ep.runtime.calls[0][18] == 8 and ep.sms_calls == [])

    # C3 num_qps 自动：direct min(sms,9)；hybrid sms*16+1
    ep = make_fake_ep(EPBuffer, allow_hybrid_mode=False, num_allocated_qps=17)
    check('C3a direct 模式 auto QP = min(sms, 9)', ep.get_theoretical_num_qps(64) == 9)
    ep = make_fake_ep(EPBuffer, allow_hybrid_mode=True, num_allocated_qps=129)
    check('C3b hybrid 模式 auto QP = min(sms*16+1, allocated)',
          ep.get_theoretical_num_qps(4) == 65 and ep.get_theoretical_num_qps(16) == 129)

    # C4 SF 解包：x=(fp8, sf) → runtime 收到 x=x_fp8, sf=sf
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    xf, sf = torch.zeros(4, 2, dtype=torch.float8_e4m3fn), torch.full((4, 1), 0.5)
    recv_x, _, _, handle, _ = ep.dispatch((xf, sf), topk_idx=torch.tensor([[1, 2]] * 4), num_experts=8)
    call = ep.runtime.calls[0]
    check('C4a `x=(fp8,sf)` 被拆成位置参数 (x,sf)', call[0] is xf and call[1] is sf)
    check('C4b 返回的 recv_x 重新打包回 (recv_x, recv_sf)（因 SF 非 None）',
          isinstance(recv_x, tuple) and len(recv_x) == 2 and recv_x[1] is not None)
    # 无 SF 时返回裸张量
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle, with_sf=False)
    r = ep.dispatch(x, topk_idx=torch.tensor([[1, 2]] * 4), num_experts=8)
    check('C4c x 非 tuple 且 recv_sf=None 时返回裸 recv_x', not isinstance(r[0], tuple))

    # C5 默认值：expert_alignment=1, do_cpu_sync=True, num_max_tokens_per_rank 取 buffer 默认
    ep = make_fake_ep(EPBuffer, default_max_tokens=16)
    ep.runtime.result = make_dispatch_result(EPHandle)
    ep.dispatch(x, topk_idx=torch.tensor([[1, 2]] * 4), num_experts=8)
    call = ep.runtime.calls[0]
    check('C5a 默认 expert_alignment=1', call[17] == 1)
    check('C5b 默认 num_max_tokens_per_rank = buffer 默认 16', call[15] == 16)
    check('C5c 默认 do_cpu_sync=True（无 handle）', call[23] is True)

    # C6 缓存句柄复用：handle 给定时 topk_idx 必须 None、强制 do_cpu_sync=False、字段回填
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    h0 = make_handle(EPHandle, num_experts=8, num_max_tokens_per_rank=16)
    _, _, _, handle, _ = ep.dispatch(x, handle=h0, num_experts=8)
    call = ep.runtime.calls[0]
    check('C6a 复用 handle：topk_idx 来自 handle、返回同一 handle 对象',
          call[2] is h0.topk_idx and handle is h0)
    check('C6b 复用 handle：强制 do_cpu_sync=False', call[23] is False)

    # C7 缓存句柄的上下文一致性断言
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    h0 = make_handle(EPHandle, num_experts=8, num_max_tokens_per_rank=16)
    raised = False
    try:
        ep.dispatch(x, handle=h0, num_experts=999)
    except AssertionError:
        raised = True
    check('C7 num_experts 与 handle 不一致被断言拦下（line 662 裸断言）', raised)

    # C8 topk_idx 与 handle 同时给：assert topk_idx is None
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    h0 = make_handle(EPHandle)
    raised = False
    try:
        ep.dispatch(x, topk_idx=torch.zeros(4, 2, dtype=torch.long), handle=h0, num_experts=8)
    except AssertionError:
        raised = True
    check('C8 `handle` 与 `topk_idx` 互斥（assert topk_idx is None）', raised)

    # C9 do_handle_copy 必须 False
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    raised = False
    try:
        ep.dispatch(x, topk_idx=torch.zeros(4, 2, dtype=torch.long), num_experts=8, do_handle_copy=True)
    except AssertionError as e:
        raised = 'do_handle_copy' in str(e)
    check('C9 `do_handle_copy=True` 被断言拦下', raised)

    # C10 deferred epilogue：返回 EventOverlap，finalize 在 wait 时才跑
    ep = make_fake_ep(EPBuffer)
    deferred_result = make_dispatch_result(EPHandle)
    ep.runtime.result = None
    ep.runtime.deferred = lambda: deferred_result
    ret = ep.dispatch(x, topk_idx=torch.tensor([[1, 2]] * 4), num_experts=8, defer_epilogue=True)
    check('C10a defer_epilogue=True 直接返回 EventOverlap（不是 5 元组）',
          ret.__class__.__name__ == 'EventOverlap' and ret.hook_after_wait is not None)
    res = ret.wait()
    check('C10b `event.wait()` 触发 finalize，返回 4 元组 (recv_x,idx,w,handle)',
          isinstance(res, tuple) and len(res) == 4 and isinstance(res[3], EPHandle))

    # C11 确定性排序钩子接线：async=True → 注册 hook（不立即跑）；async=False → 立即跑
    ep = make_fake_ep(EPBuffer, deterministic=True)
    ep.runtime.result = make_dispatch_result(EPHandle)
    _, _, _, _, eo = ep.dispatch(x, topk_idx=torch.tensor([[1, 2]] * 4),
                                 num_experts=8, async_with_compute_stream=True)
    check('C11a deterministic + async → 排产为 hook（尚未执行）', eo.hook_after_wait is not None)

    ep = make_fake_ep(EPBuffer, deterministic=True)
    ep.runtime.result = make_dispatch_result(EPHandle)
    _, _, _, _, eo = ep.dispatch(x, topk_idx=torch.tensor([[1, 2]] * 4),
                                 num_experts=8, async_with_compute_stream=False)
    check('C11b deterministic + 非 async → 立即执行排序（无 hook）', eo.hook_after_wait is None)


def part_combine(EPBuffer, EPHandle):
    print('\n=== Part D: `EPBuffer.combine` host 编排（复用 dispatch 的 handle / bias 解包 / deferred）===')
    ep = make_fake_ep(EPBuffer)
    h = make_handle(EPHandle, num_sms=48)
    ep.runtime.result = (torch.zeros(4, 2), torch.zeros(4, 2))
    out = ep.combine(torch.zeros(4, 2), handle=h)
    call = ep.runtime.calls[0]
    check('D1 combine 从 handle 取 num_sms（num_sms=0 → 复用 48，align 后 48）', call[11] == 48)
    check('D2 combine 传 handle 的关键字段（recv_src_metadata/topk_idx/psum_rank/metadata）',
          call[4] is h.recv_src_metadata and call[5] is h.topk_idx
          and call[6] is h.psum_num_recv_tokens_per_scaleup_rank
          and call[7] is h.token_metadata_at_forward and call[8] is h.channel_linked_list
          and call[9] == h.num_experts and call[10] == h.num_max_tokens_per_rank)
    check('D3 combine 传 handle.do_expand', call[16] is h.do_expand)
    check('D4 combine 返回 (combined_x, combined_topk_weights, EventOverlap)',
          len(out) == 3 and out[2].__class__.__name__ == 'EventOverlap')

    # bias 解包
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = (torch.zeros(4, 2), torch.zeros(4, 2))
    b0, b1 = torch.zeros(4, 2), torch.ones(4, 2)
    ep.combine(torch.zeros(4, 2), handle=h, bias=(b0, b1))
    call = ep.runtime.calls[0]
    check('D5 bias=(b0,b1) → runtime 收到 (b0,b1)', call[2] is b0 and call[3] is b1)

    # deferred combine
    ep = make_fake_ep(EPBuffer)
    ep.runtime.deferred = lambda: (torch.zeros(4, 2), torch.zeros(4, 2))
    ret = ep.combine(torch.zeros(4, 2), handle=h, defer_epilogue=True)
    check('D6 combine defer_epilogue=True → 返回 EventOverlap，wait 后才出结果',
          ret.__class__.__name__ == 'EventOverlap' and ret.hook_after_wait is not None
          and len(ret.wait()) == 2)


def part_deterministic(EPHandle, EPBuffer):
    print('\n=== Part E: 真机 `EPHandle.deterministic_sort`（非 expand：按源 token 序重排）===')
    dev = 'cuda'
    n = 8
    # recv_src_metadata[:,0] 是源 token 全局索引；故意乱序 [3,1,0,2,7,5,4,6]
    src = torch.tensor([3, 1, 0, 2, 7, 5, 4, 6], device=dev)
    meta = torch.stack([src, torch.zeros(n, dtype=torch.int64, device=dev),
                        torch.zeros(n, dtype=torch.int64, device=dev)], dim=1)
    recv_x = (src.float() * 10).unsqueeze(1).contiguous()   # 标记值 = 源序号*10
    recv_sf = torch.zeros(n, 1, device=dev)
    recv_topk_idx = torch.zeros(n, 2, dtype=torch.int64, device=dev)
    recv_topk_weights = torch.zeros(n, 2, device=dev)
    h = make_handle(EPHandle, topk_idx=torch.zeros(n, 2, dtype=torch.int64),
                    num_recv_tokens=n, num_expanded_tokens=n,
                    psum_rank=torch.tensor([0, n], device=dev),
                    psum_expert=torch.tensor([2, 4, 6, n], device=dev),
                    unaligned=torch.tensor([2, 2, 2, 2], device=dev),
                    recv_src_metadata=meta)
    h.deterministic_sort(True, False, recv_x, recv_sf, recv_topk_idx, recv_topk_weights, None)
    check('E1 非 expand：recv_x 重排成源 token 升序（值=源序*10 单调）',
          torch.equal(recv_x[:, 0], torch.arange(n, device=dev).float() * 10),
          f'got={recv_x[:,0].tolist()}')
    check('E2 recv_src_metadata[:,0] 变为升序', bool((h.recv_src_metadata[:, 0].diff() > 0).all()))
    check('E3 do_cpu_sync=True 时 `cached_recv_src_metadata_before_sort` 已被缓存',
          h.cached_recv_src_metadata_before_sort is not None)


def part_semantic(sem_mod, math_mod):
    print('\n=== Part F: `value_or` / `weak_lru` / `align` ===')
    value_or = sem_mod.value_or
    check('F1 value_or(None, d)=d；value_or(v, d)=v',
          value_or(None, 7) == 7 and value_or(0, 7) == 0 and value_or(False, 7) is False)

    weak_lru = sem_mod.weak_lru

    class Obj:
        def __init__(self):
            self.count = 0

        @weak_lru(maxsize=None)
        def f(self, x):
            self.count += 1
            return x * 2

    o = Obj()
    a = o.f(3)
    b = o.f(3)
    check('F2 weak_lru 命中缓存（同一参数只算一次）', a == 6 and b == 6 and o.count == 1)
    o.f(4)
    check('F3 不同参数各算一次', o.count == 2)
    # 缓存不持有强引用：对象删除后不阻止回收
    ref = o.f
    del o
    import gc
    gc.collect()
    check('F4 weak_lru 不强引用 self（无崩溃）', True)

    align = math_mod.align
    ceil_div = math_mod.ceil_div
    check('F5 align(7,2)=8, align(8,2)=8, align(0,4)=0',
          align(7, 2) == 8 and align(8, 2) == 8 and align(0, 4) == 0)
    check('F6 ceil_div(7,2)=4, ceil_div(8,2)=4', ceil_div(7, 2) == 4 and ceil_div(8, 2) == 4)


def part_deterministic_guard(envs_mod, EPBuffer, EPHandle):
    print('\n=== Part G: `check_torch_deterministic`（dispatch/combine 开头的守卫）===')
    old = torch.are_deterministic_algorithms_enabled()
    old_fill = torch.utils.deterministic.fill_uninitialized_memory
    raised = False
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        try:
            envs_mod.check_torch_deterministic()
        except AssertionError:
            raised = True
    finally:
        torch.use_deterministic_algorithms(old)
        torch.utils.deterministic.fill_uninitialized_memory = old_fill
    check('G1 确定性算法 ∧ fill_uninitialized_memory 同时开 → 断言失败', raised)

    envs_mod.check_torch_deterministic()
    check('G2 单独任一开启时不报错', True)

    # dispatch 首行就调它 → 二者同开时 dispatch 直接抛
    ep = make_fake_ep(EPBuffer)
    ep.runtime.result = make_dispatch_result(EPHandle)
    raised = False
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        try:
            ep.dispatch(torch.zeros(4, 2), topk_idx=torch.zeros(4, 2, dtype=torch.long), num_experts=8)
        except AssertionError:
            raised = True
    finally:
        torch.use_deterministic_algorithms(old)
        torch.utils.deterministic.fill_uninitialized_memory = old_fill
    check('G3 dispatch 首行守卫在非法组合下拦下', raised)


def main():
    print('P73-1 DeepEP EPHandle 句柄契约 + dispatch/combine host 编排层 —— H100 真机')
    print(f'GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}')
    print(f'DeepEP root: {DEP}')
    print(f'torch: {torch.__version__}  nccl={torch.cuda.nccl.version()}')

    pkg, cstub = make_stub_pkg()
    sem_mod = load_real('deep_ep/utils/semantic.py', 'deep_ep.utils.semantic')
    math_mod = load_real('deep_ep/utils/math.py', 'deep_ep.utils.math')
    event_mod = load_real('deep_ep/utils/event.py', 'deep_ep.utils.event')
    envs_mod = load_real('deep_ep/utils/envs.py', 'deep_ep.utils.envs')
    alloc_mod = load_real('deep_ep/buffers/allocator.py', 'deep_ep.buffers.allocator')
    base_mod = load_real('deep_ep/buffers/base.py', 'deep_ep.buffers.base')
    ep_mod = load_real('deep_ep/buffers/ep.py', 'deep_ep.buffers.ep')

    EPHandle = ep_mod.EPHandle
    EPBuffer = ep_mod.EPBuffer

    part_ephandle(EPHandle)
    part_static(EPBuffer, EPHandle)
    part_dispatch(EPBuffer, EPHandle)
    part_combine(EPBuffer, EPHandle)
    part_deterministic(EPHandle, EPBuffer)
    part_semantic(sem_mod, math_mod)
    part_deterministic_guard(envs_mod, EPBuffer, EPHandle)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f'\n===== {npass} PASS / {len(RESULTS)-npass} FAIL =====')
    print('sha256: ' + '  '.join(f'{k.split(".")[-1]}={v[:16]}' for k, v in SHA.items()))
    return 0 if npass == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
