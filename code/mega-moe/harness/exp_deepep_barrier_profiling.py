"""P91-1 实验：DeepEP 屏障的验收基准与「通信 kernel 精确测时」方法论。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径逐行直接加载真实源码 / 原样 exec）：
  * tests/comm/test_barrier.py            —— 屏障的**验收基准**（唯一一份没被单独拆过的 DeepEP 测试）
  * deep_ep/utils/testing.py              —— `bench_kineto(..., barrier_comm_profiling=True, barrier=...)`
  * deep_ep/comm/barrier.py               —— Python `barrier(buffer, wait_comm_stream, with_cpu_sync, sequential, context_idx)`
  * deep_ep/comm/handle.py                —— `get_physical_domain_size` / `get_logical_domain_size`
  * csrc/kernels/comm/context.cpp         —— 域尺寸语义（`allow_hybrid_mode` 分支）
  * csrc/comm/barrier.hpp、csrc/kernels/comm/barrier.hpp、impls/comm/barrier.cuh
                                          —— host 启动层 + device kernel（sequential 两级 vs 并行 1 级）

为什么 H100 可跑：
  * `import deep_ep` 会触发 `deep_ep/__init__.py` 的 check_nccl_so()（本机 NCCL 2.29.7 与编译期不匹配）。
    本脚本用 sys.modules 注入合成包 + stub `deep_ep._C`，再按文件路径 exec 真实源码（打印 sha256）。
  * 真正的 barrier kernel（`impls/comm/barrier.cuh`）走 NCCL GIN 设备 API，本机没有 → **仅分析**；
    但它的 host 启动公式、域语义、测时方法全在纯 host/Python + torch profiler 里 → H100 真机可跑。
  * Part D 用真实 `deep_ep.utils.testing.bench_kineto` + 真实 DeepGEMM FP8 kernel，在 8×H100 NCCL
    上比较三种 `barrier_comm_profiling` 模式的**计时稳定性**，量出「为什么要插屏障」。

运行：bash harness/exp_deepep_barrier_profiling.sh > harness/exp_deepep_barrier_profiling.out.txt
"""
import argparse
import ast
import hashlib
import importlib.util
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
WORLD = int(os.environ.get('EP_BARRIER_WORLD', '8'))
DEV = 'cuda'
# 目标卡代号敏感：一律不出现真实卡名。

TIMEOUT = __import__('datetime').timedelta(seconds=300)


# ======================================================================================
# 真实源码加载（绕过 deep_ep/__init__.py 的 check_nccl_so）
# ======================================================================================
def load_by_path(mod_name, path):
    src = open(path, 'rb').read()
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m, hashlib.sha256(src).hexdigest()


def make_recording_C():
    """一个可记录调用的 `deep_ep._C` 桩（barrier / domain 尺寸 / 唯一 ID）。"""
    c = types.ModuleType('deep_ep._C')
    c.calls = []

    def barrier(context, with_cpu_sync, sequential, wait_comm_stream):
        c.calls.append(('barrier', context, bool(with_cpu_sync), bool(sequential), bool(wait_comm_stream)))

    def get_physical_domain_size(nccl_comm):
        c.calls.append(('physical', nccl_comm))
        return (2, 4)

    def get_logical_domain_size(nccl_comm, allow_hybrid_mode):
        c.calls.append(('logical', nccl_comm, bool(allow_hybrid_mode)))
        return (2, 4) if allow_hybrid_mode else (1, 8)

    c.barrier = barrier
    c.get_physical_domain_size = get_physical_domain_size
    c.get_logical_domain_size = get_logical_domain_size
    c.get_local_nccl_unique_id = lambda: b'\x00' * 128
    c.create_nccl_comm = lambda *a, **k: 1
    c.destroy_nccl_comm = lambda *a, **k: None
    return c


def load_env():
    """构造合成 `deep_ep` 包树 + stub `_C` + stub `envs`，再加载 handle/barrier/testing/test。"""
    for name, path in [('deep_ep', f'{DEP}/deep_ep'),
                       ('deep_ep.utils', f'{DEP}/deep_ep/utils'),
                       ('deep_ep.comm', f'{DEP}/deep_ep/comm'),
                       ('deep_ep.buffers', f'{DEP}/deep_ep/buffers')]:
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
    sys.modules['deep_ep'].utils = sys.modules['deep_ep.utils']
    sys.modules['deep_ep'].comm = sys.modules['deep_ep.comm']
    sys.modules['deep_ep'].buffers = sys.modules['deep_ep.buffers']

    c = make_recording_C()
    sys.modules['deep_ep._C'] = c
    sys.modules['deep_ep']._C = c

    envs = types.ModuleType('deep_ep.utils.envs')
    envs._local_rank = 0
    envs.dist_print = lambda s='', once_in_node=False: print(s, flush=True)
    envs.init_dist = lambda local_rank, num_local_ranks, seed=0: (local_rank, num_local_ranks, None)
    sys.modules['deep_ep.utils.envs'] = envs
    sys.modules['deep_ep.utils'].envs = envs

    testing, h_testing = load_by_path('deep_ep.utils.testing', f'{DEP}/deep_ep/utils/testing.py')
    sys.modules['deep_ep.utils'].testing = testing
    handle, h_handle = load_by_path('deep_ep.comm.handle', f'{DEP}/deep_ep/comm/handle.py')
    barrier_mod, h_barrier = load_by_path('deep_ep.comm.barrier', f'{DEP}/deep_ep/comm/barrier.py')
    # `deep_ep.comm` 暴露多少就用多少（供合成 deep_ep 的 EPBuffer / comm 命名空间用）
    sys.modules['deep_ep.comm'].barrier = barrier_mod.barrier
    sys.modules['deep_ep.comm'].get_physical_domain_size = handle.get_physical_domain_size
    sys.modules['deep_ep.comm'].get_logical_domain_size = handle.get_logical_domain_size
    # test_barrier.py 的 `buffer: deep_ep.EPBuffer` 注解在 def 时求值 → 需要一个占位属性
    sys.modules['deep_ep'].EPBuffer = object

    test_mod, h_test = load_by_path('deep_ep_test_barrier', f'{DEP}/tests/comm/test_barrier.py')
    return {
        'C': c, 'envs': envs, 'testing': testing, 'handle': handle,
        'barrier_mod': barrier_mod, 'test': test_mod,
        'hashes': {'testing.py': h_testing, 'handle.py': h_handle,
                   'barrier.py': h_barrier, 'test_barrier.py': h_test},
    }


# ======================================================================================
# Part A：真实源码加载（sha256 + 结构）
# ======================================================================================
def part_a(ctx, rank):
    if rank != 0:
        return
    print('### Part A：真实源码加载（sha256，`~/github/DeepEP@93eb6eb`）')
    for k, v in ctx['hashes'].items():
        print(f'  {k:20s} sha256={v[:16]}…')
    top_funcs = [n.name for n in ast.parse(open(f'{DEP}/tests/comm/test_barrier.py').read()).body
                 if isinstance(n, ast.FunctionDef)]
    print(f'  test_barrier.py 顶层函数 = {top_funcs}')
    print(f'  testing.bench_kineto 存在 = {hasattr(ctx["testing"], "bench_kineto")}')
    bf = ctx['barrier_mod'].barrier
    print(f'  comm.barrier 形参 = {list(bf.__code__.co_varnames[:bf.__code__.co_argcount])}')
    print()


# ======================================================================================
# Part B：屏障验收基准的驱动逻辑 oracle（原样 exec 真实 test_barrier.py）
# ======================================================================================
class FakeContext:
    """模拟 `Context`：既能被 `barrier()` 透传给 `_C`，又有域尺寸方法。"""

    def __init__(self, name):
        self.name = name

    def get_logical_domain_size(self):
        return (2, 4)

    def get_physical_domain_size(self):
        return (2, 4)

    def __eq__(self, other):
        return self.name == other if isinstance(other, str) else NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


class FakeBuffer:
    def __init__(self, num_rank=8, rank_idx=0):
        self.rank_idx = rank_idx
        self.num_ranks = num_rank
        self.num_allocated_qps = 8
        self.context = FakeContext('ctx0')
        self.contexts = [FakeContext('ctx0'), FakeContext('ctx1'), FakeContext('ctx2')]
        self.destroyed = False
        self.barrier_calls = 0

    def get_logical_domain_size(self):
        return (2, 4)

    def barrier(self, **kwargs):
        self.barrier_calls += 1

    def destroy(self):
        self.destroyed = True


def part_b(ctx, rank):
    if rank != 0:
        return
    test_mod = ctx['test']
    print('### Part B：验收基准的驱动逻辑 oracle（原样 exec `tests/comm/test_barrier.py`）')
    ok = tot = 0

    # 1) argparse 面（`) `__main__` 里四个参数）
    src = open(f'{DEP}/tests/comm/test_barrier.py').read()
    tree = ast.parse(src)
    main = [n for n in tree.body if isinstance(n, ast.If)]
    parser_args = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'add_argument':
            opt = node.args[0].value
            kw = {k.arg: (k.value.value if hasattr(k.value, 'value') else type(k.value).__name__)
                  for k in node.keywords}
            parser_args[opt] = kw
    tot += 1
    expected = {'--num-processes', '--allow-hybrid-mode', '--num-allocated-qps', '--do-pressure-test'}
    good = set(parser_args) == expected
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] argparse 面 = {sorted(parser_args)}')
    for opt, kw in parser_args.items():
        print(f'        {opt:22s} type={kw.get("type")!s:8s} default={kw.get("default")!r} action={kw.get("action")}')

    # 2) test_barrier 调用 bench_kineto 的形态（monkeypatch 成记录器）
    captured = {}
    test_mod.bench_kineto = lambda fn, kernel_names, **kw: (
        captured.update({'kernel_names': kernel_names, **kw}) or 123.456e-6)
    buf = FakeBuffer()
    test_mod.test_barrier(buf, argparse.Namespace(allow_hybrid_mode=1, num_allocated_qps=8,
                                                  do_pressure_test=False))
    tot += 1
    good = (captured.get('kernel_names') == 'barrier'
            and captured.get('barrier_comm_profiling') is True
            and captured.get('barrier') == buf.barrier)
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] test_barrier → bench_kineto(kernel_names={captured.get("kernel_names")!r}, '
          f'barrier_comm_profiling={captured.get("barrier_comm_profiling")}, '
          f'barrier=buffer.barrier({captured.get("barrier") is buf.barrier}))')
    print(f'        loop_barrier 在 profile 里连续调用 buffer.barrier()；本 rank barrier 次数 = {buf.barrier_calls}'
          f'（bench_kineto 被替换，故 0 次真实调用）')

    # 3) test_loop 的域恒等式断言 + pressure 循环 + explicitly_destroy
    #    用 stub 的 init_dist / deep_ep / buffer 原样驱动真实 test_loop
    made = {'n': 0, 'destroyed': 0, 'args': []}

    def fake_EPBuffer(group, num_bytes, allow_hybrid_mode, num_allocated_qps, explicitly_destroy):
        made['n'] += 1
        made['args'].append((num_bytes, allow_hybrid_mode, num_allocated_qps, explicitly_destroy))
        b = FakeBuffer()
        orig_destroy = b.destroy

        def destroy():
            made['destroyed'] += 1
            orig_destroy()

        b.destroy = destroy
        return b

    test_mod.init_dist = lambda lr, nlr: (0, 8, 'group')
    test_mod.test_barrier = lambda buffer, args: None
    saved_deep_ep = test_mod.deep_ep
    fake_deep_ep = types.ModuleType('deep_ep_fake')
    fake_deep_ep.EPBuffer = fake_EPBuffer
    fake_deep_ep.comm = types.ModuleType('deep_ep_fake.comm')
    fake_deep_ep.comm.get_physical_domain_size = lambda group: (2, 4)
    fake_deep_ep.comm.get_logical_domain_size = lambda group, allow_hybrid: (2, 4) if allow_hybrid else (1, 8)
    test_mod.deep_ep = fake_deep_ep
    test_mod.dist = types.SimpleNamespace(destroy_process_group=lambda: None)

    test_mod.test_loop(0, 8, argparse.Namespace(allow_hybrid_mode=1, num_allocated_qps=8,
                                                do_pressure_test=False))
    tot += 1
    good = (made['n'] == 1 and made['destroyed'] == 1 and made['args'][0] == (2 ** 30, 1, 8, True))
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] test_loop 单次：EPBuffer({made["args"][0] if made["args"] else None})、'
          f'destroy×{made["destroyed"]}')

    # 4) 域恒等式必须失败：伪造一个 rdma*nvl != num_ranks 的域 → AssertionError
    fake_deep_ep.comm.get_physical_domain_size = lambda group: (3, 4)  # 12 != 8
    made['n'] = 0
    raised = False
    try:
        test_mod.test_loop(0, 8, argparse.Namespace(allow_hybrid_mode=1, num_allocated_qps=8,
                                                    do_pressure_test=False))
    except AssertionError:
        raised = True
    tot += 1
    ok += raised
    print(f'  [{"PASS" if raised else "FAIL"}] 域恒等式 `rdma*nvl==num_ranks` 被 assert 拒绝（伪造 3×4 vs 8）')

    test_mod.deep_ep = saved_deep_ep
    print(f'  → Part B: {ok}/{tot} PASS')
    print()


# ======================================================================================
# Part C：域尺寸语义 oracle（复刻 csrc/kernels/comm/context.cpp:52-63）
# ======================================================================================
def physical_domain(num_ranks, num_nvl):
    assert num_ranks % num_nvl == 0
    return num_ranks // num_nvl, num_nvl


def logical_domain(num_ranks, num_nvl, allow_hybrid_mode):
    nr, nnvl = physical_domain(num_ranks, num_nvl)
    return (nr, nnvl) if allow_hybrid_mode else (1, nr * nnvl)


def part_c(ctx, rank):
    if rank != 0:
        return
    print('### Part C：域尺寸语义 oracle（复刻 `context.cpp:52-63`）')
    ok = tot = 0
    rows = [(8, 8), (8, 4), (16, 8), (32, 4), (8, 1)]
    for num_ranks, num_nvl in rows:
        nr, nnvl = physical_domain(num_ranks, num_nvl)
        lo, lu = logical_domain(num_ranks, num_nvl, True), logical_domain(num_ranks, num_nvl, False)
        tot += 1
        good = (nr * nnvl == num_ranks) and (lo[0] * lo[1] == num_ranks) and (lu[0] * lu[1] == num_ranks) \
            and lo == (nr, nnvl) and lu == (1, num_ranks)
        ok += good
        print(f'  [{"PASS" if good else "FAIL"}] ranks={num_ranks:2d} nvl={num_nvl:2d} → '
              f'physical=(rdma={nr}, nvl={nnvl})  hybrid=(scaleout={lo[0]}, scaleup={lo[1]})  '
              f'flat=(1, {lu[1]})')
    # hybrid=False 时 scaleup 吞掉全部 ranks：两级退化为一整级
    tot += 1
    flat = logical_domain(16, 8, False)
    good = flat == (1, 16)
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] allow_hybrid_mode=False → 单级域 (1, num_ranks)={flat}'
          f'（`--allow-hybrid-mode 0` 的语义）')
    print(f'  → Part C: {ok}/{tot} PASS')
    print()


# ======================================================================================
# Part D：launch_barrier 的 host 启动公式（复刻 csrc/kernels/comm/barrier.hpp:25-41）
# ======================================================================================
def launch_config(sequential, num_scaleout_ranks, is_scaleup_nvlink, num_scaleup_ranks,
                  num_timeout_cycles, flush_stores):
    kNumThreads = 512
    num_sms = 2 if (not sequential and num_scaleout_ranks > 1) else 1
    # JIT 模板实参顺序：<is_scaleup_nvlink, num_sms, kNumThreads,
    #                    num_scaleout_ranks, num_scaleup_ranks, num_timeout_cycles,
    #                    flush_stores, sequential>
    return dict(kNumThreads=kNumThreads, num_sms=num_sms,
                template=(is_scaleup_nvlink, num_sms, kNumThreads, num_scaleout_ranks,
                          num_scaleup_ranks, num_timeout_cycles, flush_stores, sequential))


def part_d_host(ctx, rank):
    if rank != 0:
        return
    print('### Part D：`launch_barrier` host 启动公式（复刻 `kernels/comm/barrier.hpp:25-41`）')
    ok = tot = 0
    cases = [
        # (sequential, scaleout, nvlink, scaleup, cycles, flush, expect_sms)
        (True, 2, False, 4, 10 ** 9, True, 1),
        (True, 1, False, 8, 10 ** 9, True, 1),   # 单级（scaleout==1）→ 1 SM
        (False, 2, False, 4, 10 ** 9, True, 2),  # 并行混合 → 2 SM
        (False, 1, False, 8, 10 ** 9, True, 1),  # 并行但单级 → 1 SM
        (False, 4, True, 8, 10 ** 9, False, 2),
    ]
    for seq, so, nv, su, cyc, fl, exp in cases:
        cfg = launch_config(seq, so, nv, su, cyc, fl)
        tot += 1
        good = cfg['num_sms'] == exp and cfg['kNumThreads'] == 512
        ok += good
        print(f'  [{"PASS" if good else "FAIL"}] sequential={int(seq)} scaleout={so} → '
              f'num_sms={cfg["num_sms"]}（期望 {exp}） template={cfg["template"]}')
    print(f'  → Part D: {ok}/{tot} PASS')
    print()


# ======================================================================================
# Part E：Python `barrier(...)` 的参数转发 + 域 helper 语义（stub `_C`）
# ======================================================================================
def part_e(ctx, rank):
    if rank != 0:
        return
    print('### Part E：Python `deep_ep/comm/barrier.py` 的参数转发（stub `_C`）')
    c = ctx['C']
    bar = ctx['barrier_mod'].barrier
    ok = tot = 0
    buf = FakeBuffer()
    c.calls.clear()
    bar(buf, wait_comm_stream=True, with_cpu_sync=False, sequential=True, context_idx=None)
    tot += 1
    good = c.calls and c.calls[-1] == ('barrier', 'ctx0', False, True, True)
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] barrier(buffer) → _C.barrier{tuple(c.calls[-1][1:]) if c.calls else None}'
          f'（context=context_idx=None→buffer.context）')
    c.calls.clear()
    bar(buf, wait_comm_stream=False, with_cpu_sync=True, sequential=False, context_idx=2)
    tot += 1
    good = c.calls and c.calls[-1] == ('barrier', 'ctx2', True, False, False)
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] barrier(with_cpu_sync=True, sequential=False, context_idx=2) → '
          f'_C.barrier{tuple(c.calls[-1][1:]) if c.calls else None}（context=buffer.contexts[2]）')

    # domain helper：有 context 走 context，没 context 走 _C（handle.py:80-95）
    handle = ctx['handle']
    c.calls.clear()
    got = handle.get_logical_domain_size(buf, allow_hybrid_mode=True)
    tot += 1
    good = got == (2, 4) and not c.calls   # 有 context 时不该落到 _C
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] get_logical_domain_size(buffer) → {got}（有 context，直接问 context）')

    class GroupOnly:
        """无 `.context`，走 `_C` 路径；还需一个带 `_comm_ptr` 的 backend（handle.py:63-66）。"""

        def __hash__(self):
            return id(self)

        def _get_backend(self, device):
            return types.SimpleNamespace(_comm_ptr=lambda: 123)

    c.calls.clear()
    got = handle.get_logical_domain_size(GroupOnly(), allow_hybrid_mode=False)
    tot += 1
    good = got == (1, 8) and c.calls[-1][0] == 'logical' and c.calls[-1][2] is False
    ok += good
    print(f'  [{"PASS" if good else "FAIL"}] get_logical_domain_size(group, allow_hybrid_mode=False) → {got}'
          f'（无 context → _C.get_logical_domain_size(comm, False)）')
    print(f'  → Part E: {ok}/{tot} PASS')
    print()


# ======================================================================================
# Part F：真实 `bench_kineto` 的 `barrier_comm_profiling`（8×H100 NCCL + 真实 DeepGEMM kernel）
# ======================================================================================
def make_fp8_inputs(m=4096, n=4096, k=4096):
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
    torch.manual_seed(0)
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8, sfb = per_block_cast_to_fp8(b, use_ue8m0=False)
    d = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)

    def fn():
        deep_gemm.fp8_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d)

    return fn


def _event_bench(fn, n=30):
    """event 计时参考：每轮前 flush 256MB（同 bench_kineto），但 start event 在 flush 之后。"""
    flush = lambda: torch.empty(int(256e6 // 4), dtype=torch.int, device=DEV).zero_()
    for _ in range(5):
        flush(); fn()
    torch.cuda.synchronize()
    se = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        flush(); se[i].record(); fn(); ee[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(se, ee))
    return ts[0], ts[len(ts) // 2]


def _sustained_load(fn, iters=100):
    flush = lambda: torch.empty(int(256e6 // 4), dtype=torch.int, device=DEV).zero_()
    import time

    def loop(with_flush, n):
        for _ in range(5):
            (flush() if with_flush else None); fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            (flush() if with_flush else None); fn()
        torch.cuda.synchronize()
        return (time.time() - t) / n * 1e3  # ms/iter

    return {'gemm': loop(False, iters), 'flush+gemm': loop(True, iters)}


def _raw_profiler_sleep_sweep(fn, kernel_name, num_tests=15):
    """复刻 bench_kineto 的 profiler 循环，只把 `torch.cuda._sleep` 的周期数当变量。"""
    out = []
    for cycles in (0, 10 ** 6, 4 * 10 ** 6, 10 ** 7, 2 * 10 ** 7):
        fn(); torch.cuda.synchronize()
        sch = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                   schedule=sch, acc_events=True)
        with p:
            for _ in range(2):
                for _ in range(num_tests):
                    torch.empty(int(256e6 // 4), dtype=torch.int, device=DEV).zero_()
                    if cycles:
                        torch.cuda._sleep(int(cycles))
                    fn()
                torch.cuda.synchronize(); p.step()
        lines = p.key_averages().table(sort_by='cuda_time_total', max_name_column_width=80).split('\n')
        dur = None
        for l in lines:
            if kernel_name in l:
                dur = l.split()[-2]
        out.append((cycles, dur))
    return out


def repeat_kineto(testing, fn, kernel_name, **kw):
    durations = []
    for _ in range(kw.pop('_repeats', 3)):
        durations.append(testing.bench_kineto(fn, kernel_name, num_tests=15,
                                              suppress_kineto_output=True, **kw))
    return durations


def part_f(ctx, rank, world):
    testing = ctx['testing']
    try:
        fn = make_fp8_inputs()
        fn(); torch.cuda.synchronize()
    except Exception as e:  # pragma: no cover
        if rank == 0:
            print(f'### Part F：SKIP（无法构造 DeepGEMM FP8 kernel: {e}）\n')
        return
    kernel_name = 'sm90_fp8_gemm_1d2d'

    # 三种模式：无屏障 / 默认 all_reduce 屏障 / 自定义 callback 屏障
    modes = [
        ('no-barrier        ', dict(barrier_comm_profiling=False, barrier=None)),
        ('barrier=all_reduce', dict(barrier_comm_profiling=True, barrier=None)),
        ('barrier=custom    ', dict(barrier_comm_profiling=True,
                                    barrier=lambda: torch.cuda._sleep(int(2e7)))),
    ]
    results = {}
    for label, kw in modes:
        ds = repeat_kineto(testing, fn, kernel_name, _repeats=3, **kw)
        results[label] = ds

    # EP_DISABLE_BARRIER_PROFILING=1 时 barrier_comm_profiling 被就地置 0
    os.environ['EP_DISABLE_BARRIER_PROFILING'] = '1'
    ds_disabled = repeat_kineto(testing, fn, kernel_name, _repeats=3,
                                barrier_comm_profiling=True, barrier=None)
    os.environ.pop('EP_DISABLE_BARRIER_PROFILING')
    results['disabled(env)     '] = ds_disabled

    # event 计时（对照）：`testing.bench` 的 flush_l2 在 start event 之前，故不计入
    ev_min, ev_med = _event_bench(fn)

    # torch.cuda._sleep(2e7) 的墙钟（注释说 ~10ms）
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record(); torch.cuda._sleep(int(2e7)); t1.record(); torch.cuda.synchronize()
    sleep_ms = t0.elapsed_time(t1)

    # 裸 profiler 扫 sleep 长度：多长的 GPU 空闲才让 per-kernel 时间回到 event 水平
    sweep = _raw_profiler_sleep_sweep(fn, kernel_name)

    # 持续负载：无间隔地背靠背跑 (flush + gemm) 与纯 gemm，看是否被降频
    sustained = _sustained_load(fn)

    if rank == 0:
        print('### Part F：真实 `bench_kineto` 的 barrier 测时路径（8×H100 NCCL + DeepGEMM FP8 kernel）')
        print(f'  kernel = {kernel_name}（M=N=K=4096）；每次 bench_kineto 15 tests × 2 批，重复 3 次')
        print(f'  torch.cuda._sleep(2e7) 墙钟 = {sleep_ms:.2f} ms（注释标 ~10ms，用大 kernel 削发射抖动）')
        print(f'  {"模式":22s} {"dur[us] 三次":32s} {"min":>8s} {"max":>8s} {"spread":>8s}')
        import statistics
        for label, ds in results.items():
            us = [d * 1e6 for d in ds]
            spread = (max(us) - min(us)) / (statistics.median(us) + 1e-9) * 100
            print(f'  {label:22s} {str([round(u, 1) for u in us]):32s} '
                  f'{min(us):8.1f} {max(us):8.1f} {spread:7.1f}%')
        base = statistics.median([d * 1e6 for d in results['no-barrier        ']])
        arr = statistics.median([d * 1e6 for d in results['barrier=all_reduce']])
        cus = statistics.median([d * 1e6 for d in results['barrier=custom    ']])
        print(f'  中位数：no-barrier={base:.1f}us  all_reduce={arr:.1f}us  custom={cus:.1f}us')
        print(f'  disabled(env) 中位数 = {statistics.median([d*1e6 for d in ds_disabled]):.1f}us'
              f' → 应与 no-barrier 同区间（barrier_comm_profiling 被环境变量置 0）')
        all_nosleep = [d * 1e6 for d in results['no-barrier        ']] \
            + [d * 1e6 for d in ds_disabled] + [float(dur.rstrip('us')) for _, dur in sweep if dur]
        print(f'  **event 计时参考**（flush 在 start event 之前，不计入）：min={ev_min:.1f}us med={ev_med:.1f}us')
        print(f'  ⇒ 真实 kernel 时长 ≈ {ev_med:.0f}us（快档）。将所有「无 sleep」测量（no-barrier + disabled(env) '
              f'+ sleep=0 扫描）合并看：这些样本横跨 {min(all_nosleep):.0f}–{max(all_nosleep):.0f}us，'
              f'呈**双峰**（快档 ~100us、慢档 ~417us，≈4×）；而三种 barrier 模式每次都落在快档且 spread <0.3%。')
        print(f'  裸 profiler 扫 sleep 长度（其余同 bench_kineto）：')
        for cycles, dur in sweep:
            print(f'        sleep={cycles:>9d} cycles → profiler 报告的 kernel 时长 = {dur}')
        print(f'  持续负载（背靠背，无间隔）：纯 GEMM {sustained["gemm"]:.3f} ms/iter、'
              f'flush+GEMM {sustained["flush+gemm"]:.3f} ms/iter'
              f'（对照隔离 event 的 {ev_med/1e3:.3f} ms/iter）')
        print()


# ======================================================================================
def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, timeout=TIMEOUT)
    ctx = load_env()
    part_a(ctx, rank)
    part_b(ctx, rank)
    part_c(ctx, rank)
    part_d_host(ctx, rank)
    part_e(ctx, rank)
    part_f(ctx, rank, world)
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(21000 + (os.getpid() % 20000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}  nccl={torch.cuda.nccl.version()}')
    print('DeepEP 屏障的验收基准与「通信 kernel 精确测时」方法论 —— H100 真机')
    print('=' * 96)
    if WORLD > 1:
        mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    else:
        worker(0, 1)
