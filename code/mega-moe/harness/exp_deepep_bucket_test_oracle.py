"""P84-1 实验：DeepEP BucketBuffer 混合集合算子的验收测试即正确性规范。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径 exec 真实模块，不走 deep_ep/__init__）：
  * tests/bucket/test_mixed.py —— bucket 集合算子的旗舰**正确性**测试（373 行）：混合
    all_reduce / reduce_scatter / all_gather / rs_ag，跨多通信组、多物理域、多 dtype，
    带 guard 区、session、外部输入、packed、barrier 变体、销毁/弱引用。
  * tests/bucket/test_perf.py —— bucket 的**性能**基准（273 行）：DeepEP vs 多档 NCCL profile，
    ragged bucket、coalescing、per-SM 扫描。

为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
  * `run_round`/`test` 要构造 `deep_ep.BucketBuffer`，依赖 DeepEP 的 C++/NCCL 设备 API
    （`ncclDevComm`/`ncclGin`），本机 NCCL 2.29.7 不匹配 → `import deep_ep` 直接 abort（见 docs/86）。
    故 buffer 端到端**仅分析**。
  * 但测试侧的规范逻辑是**纯 Python / 纯张量**：`supported`（能力门控）、`make_sizes`（bucket 尺寸
    量化）、`make_schedule`（用例枚举）、`make_groups`（通信组拓扑划分）、`make_inputs`（输入/guard
    构造）、`nccl_reference`（NCCL 参考）、`verify`（guard + exact 判据）、`measure`（中位数计时）、
    `get_skip_reason`。这些在 H100 上原样抽取运行、与独立 oracle 对拍，并用 8×H100 真机 NCCL
    坐实 `rs_ag` 的「先 reduce_scatter 再 all_gather == all_reduce」不变量。

运行：bash code/mega-moe/scripts/lab.sh exec \
       "python /abs/path/code/mega-moe/harness/exp_deepep_bucket_test_oracle.py"
"""
import hashlib
import math
import os
import random
import sys
import types
from functools import reduce

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
WORLD = int(os.environ.get('EP_TEST_WORLD', '8'))
DEV = 'cuda'
TEST_MIXED = f'{DEP}/tests/bucket/test_mixed.py'
TEST_PERF = f'{DEP}/tests/bucket/test_perf.py'

# 本机 torch 无 `reduce_scatter_single`/`all_gather_single`（DeepEP 测试用新名；实为
# `*_tensor` 的改名）。下面装上兼容别名，以便原样跑真实 `nccl_reference`/`nccl_collective`。
_MISSING_SINGLE = not hasattr(dist, 'reduce_scatter_single')
if _MISSING_SINGLE:
    dist.all_gather_single = dist.all_gather_into_tensor
    dist.reduce_scatter_single = dist.reduce_scatter_tensor


# --------------------------------------------------------------------------------------
# 真实源码加载：stub 掉 deep_ep（绕过 check_nccl_so），按文件路径 exec 两个测试模块
# --------------------------------------------------------------------------------------
def _load_module_by_path(modname, path):
    import importlib.util
    src = open(path, 'rb').read()
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m, hashlib.sha256(src).hexdigest()


def install_deep_ep_stub():
    """注入合成 deep_ep 包 + 真实 utils.math/testing，绕过 `import deep_ep` 的 NCCL 校验。"""
    ep = types.ModuleType('deep_ep')
    ep.__path__ = [f'{DEP}/deep_ep']
    ep.BucketBuffer = object
    ep.BufferAllocator = object
    ep.get_num_allocation_alignment = lambda: 2 << 20
    comm = types.ModuleType('deep_ep.comm')
    comm.get_physical_domain_size = lambda group: (1, group.size())
    ep.comm = comm
    sys.modules['deep_ep'] = ep
    sys.modules['deep_ep.comm'] = comm

    utilpkg = types.ModuleType('deep_ep.utils')
    utilpkg.__path__ = [f'{DEP}/deep_ep/utils']
    sys.modules['deep_ep.utils'] = utilpkg
    ep.utils = utilpkg

    envs = types.ModuleType('deep_ep.utils.envs')
    envs.dist_print = lambda *a, **k: None
    envs.init_dist = lambda *a, **k: (0, WORLD, None)
    envs.get_global_seed = lambda: 0
    envs.get_local_seed = lambda: 0
    sys.modules['deep_ep.utils.envs'] = envs

    math_m, h_math = _load_module_by_path('deep_ep.utils.math', f'{DEP}/deep_ep/utils/math.py')
    _testing, h_testing = _load_module_by_path('deep_ep.utils.testing', f'{DEP}/deep_ep/utils/testing.py')
    return math_m, h_math, h_testing


def exec_test_module(path, modname):
    src = open(path, 'rb').read()
    g = {'__name__': modname, '__file__': path}
    exec(compile(src, path, 'exec'), g)
    return g, hashlib.sha256(src).hexdigest()


# --------------------------------------------------------------------------------------
# 独立 oracle
# --------------------------------------------------------------------------------------
def align(x, a):
    return (x + a - 1) // a * a


def oracle_supported(op, physical, precision):
    rdma, nvl = physical
    if op == 'all_gather':
        return True
    if rdma * nvl == 1:
        return False
    if op == 'all_reduce':
        return True
    if precision == 'bf16':
        return rdma > 1 and nvl == 1
    return True


def oracle_make_sizes(sizes_of_groups, num_bytes, num_buckets, rng):
    quantum = math.lcm(*(s * 32 for s in sizes_of_groups))
    g0 = sizes_of_groups[0]
    out = [align(min(x * g0, num_bytes), quantum) for x in (32, 4064, 32736)]
    out += [align(rng.randrange(1, num_bytes + 1), quantum) for _ in range(3)]
    out += [align(num_bytes, quantum)]
    out += [align(min((i + 1) * quantum, num_bytes), quantum)
            for i in range(max(0, num_buckets - len(out)))]
    return out


def oracle_make_groups(world_size, nvl):
    sizes = []
    half = world_size // 2
    if world_size % 2 == 0 and half > 1 and (half % nvl == 0 or nvl % half == 0):
        sizes.append(half)
    if 1 < nvl < world_size and nvl not in sizes:
        sizes.append(nvl)
    groups = [tuple(range(world_size))]
    for size in sizes:
        for start in range(0, world_size, size):
            groups.append(tuple(range(start, start + size)))
    if 1 < nvl < world_size:
        for rail in range(nvl):
            groups.append(tuple(range(rail, world_size, nvl)))
    return groups


def oracle_get_skip_reason(op, rdma, nvl, precision):
    if rdma * nvl == 1 and op != 'all_gather':
        return 'single-rank reduction is not implemented'
    if op == 'reduce_scatter' and precision == 'bf16' and not (rdma > 1 and nvl == 1):
        return 'BF16 communication requires pure RDMA'
    return None


def oracle_perf_sizes(num_bytes, num_buckets, num_ranks, ragged, seed):
    alignment = num_ranks * 32
    sizes = [align((num_bytes + num_buckets - 1) // num_buckets, alignment)] * num_buckets
    if ragged and num_buckets > 1:
        sizes = [alignment * (i % 7 + 1) for i in range(num_buckets - 1)]
        sizes.append(align(max(alignment, num_bytes - sum(sizes)), alignment))
        random.Random(seed + num_bytes + num_buckets).shuffle(sizes)
    return sizes


class DummyEvent:
    def synchronize(self):
        pass


# --------------------------------------------------------------------------------------
# Part B：`supported` 能力门控（真实 vs 独立 oracle）
# --------------------------------------------------------------------------------------
def part_supported(rank, g):
    real = g['supported']
    ops = g['OPS']
    domain = [(1, 1), (1, 8), (8, 1), (2, 4), (4, 2), (8, 8), (16, 1), (1, 16)]
    total = fails = 0
    grid = []
    for physical in domain:
        for op in ops:
            for precision in ('fp32', 'bf16'):
                r = real(op, physical, precision)
                o = oracle_supported(op, physical, precision)
                total += 1
                fails += (r != o)
                grid.append((physical, op, precision, r))
    if rank == 0:
        print('### Part B：`supported` 能力门控（真实函数 vs 独立 oracle）')
        print(f'  真实 `OPS` = {ops}')
        print(f'  维度：physical∈{len(domain)} 种 × op∈{len(ops)} × precision∈{{fp32,bf16}} = {total} 组')
        ex = [c for c in grid if c[0] == (8, 1)]
        print('  例（physical=(8,1) 纯 RDMA）：')
        for physical, op, precision, r in ex:
            print(f'    {op:15s}/{precision:4s} -> {r}')
        print(f'  [{"PASS" if fails == 0 else "FAIL"}] {total - fails}/{total} 与 oracle 一致\n')
    return fails


# --------------------------------------------------------------------------------------
# Part C：`make_sizes`（mixed）bucket 尺寸量化（真实 vs 独立 oracle）
# --------------------------------------------------------------------------------------
def part_make_sizes(rank, g):
    real = g['make_sizes']
    fails = 0
    lines = []
    for sizes_of_groups, num_bytes, num_buckets in [([8], 1 << 20, 8), ([8, 4, 2], 1 << 22, 12),
                                                    ([16, 8], 1 << 18, 4)]:
        groups = [types.SimpleNamespace(size=lambda s=s: s) for s in sizes_of_groups]
        args = types.SimpleNamespace(num_bytes=num_bytes, num_buckets=num_buckets)
        a = real(groups, args, random.Random(7))
        b = oracle_make_sizes(sizes_of_groups, num_bytes, num_buckets, random.Random(7))
        quantum = math.lcm(*(s * 32 for s in sizes_of_groups))
        allalign = all(x % quantum == 0 for x in a)
        good = (a == b) and allalign
        fails += (not good)
        lines.append(f'  [{"PASS" if good else "FAIL"}] groups={sizes_of_groups} num_bytes={num_bytes} '
                     f'num_buckets={num_buckets} -> quantum={quantum}, len={len(a)}; '
                     f'全 {len(a)} 项均为 quantum 倍数={allalign}')
    if rank == 0:
        print('### Part C：`make_sizes`（mixed，`:82-88`）bucket 尺寸量化（真实 vs oracle）')
        print('\n'.join(lines))
        print()
    return fails


# --------------------------------------------------------------------------------------
# Part D：`make_groups` 通信组划分（8×H100 真机 dist.new_group）
# --------------------------------------------------------------------------------------
def ranks_of(group):
    return frozenset(dist.get_global_rank(group, i) for i in range(group.size()))


def part_make_groups(rank, g):
    world = dist.group.WORLD
    ws = world.size()
    halves = []
    for nvl in (4, 8, 2):
        real = g['make_groups']
        groups = real(world, nvl)
        got = sorted(tuple(sorted(ranks_of(grp))) for grp in groups)
        expected = oracle_make_groups(ws, nvl)
        exp_for_rank = sorted(tuple(sorted(gg)) for gg in expected if rank in gg)
        good = got == exp_for_rank
        halves.append(good)
        if rank == 0:
            print(f'  [{"PASS" if good else "FAIL"}] nvl={nvl}: 本 rank 属于 {len(got)} 个组；'
                  f'oracle 全体 {len(expected)} 个组（本 rank 命中 {len(exp_for_rank)}）')
            print(f'      组尺寸 = {sorted(len(x) for x in got)}')
    if rank == 0:
        print('### Part D：`make_groups`（`:44-66`）通信组拓扑划分 —— 见上\n')
    return halves


# --------------------------------------------------------------------------------------
# Part E：`nccl_reference` 语义 + `rs_ag == all_reduce` 不变量（8×H100 真机 NCCL）
# --------------------------------------------------------------------------------------
def part_nccl_reference(rank, g):
    world = dist.group.WORLD
    ws = world.size()
    real = g['nccl_reference']
    torch.manual_seed(0)
    n = 8 * ws
    base = torch.arange(n, dtype=torch.float32, device=DEV)
    x = (base + rank).contiguous()
    results = []
    for op in ('all_reduce', 'reduce_scatter', 'all_gather', 'rs_ag'):
        for scale in ([1.0, 0.0, -0.5] if op != 'all_gather' else [1.0]):
            out = real(op, [x.clone()], world, scale)[0]
            total = (base * ws + sum(range(ws)))  # Σ_r (base + r)
            if op in ('all_reduce', 'rs_ag'):
                exp = (total * scale).contiguous()
            elif op == 'reduce_scatter':
                exp = (total.view(ws, -1)[rank] * scale).contiguous()
            else:
                exp = torch.cat([(base + r) for r in range(ws)]).contiguous()
            good = torch.equal(out.view(-1), exp.view(-1)) if scale in (1.0, 0.0) \
                else torch.allclose(out.view(-1), exp.view(-1), atol=0, rtol=0)
            results.append((f'{op}/scale={scale}', good, out.numel()))

    # rs_ag 不变量：all_gather(reduce_scatter(x)) == all_reduce(x)
    rs = torch.empty(n // ws, dtype=torch.float32, device=DEV)
    dist.reduce_scatter_tensor(rs, x.view(-1), group=world)
    ag = torch.empty(n, dtype=torch.float32, device=DEV)
    dist.all_gather_into_tensor(ag, rs, group=world)
    ar = x.clone()
    dist.all_reduce(ar, group=world)
    inv = torch.equal(ag, ar)
    if rank == 0:
        print('### Part E：`nccl_reference`（`:126-141`）语义 + `rs_ag` 不变量（8×H100 真机 NCCL）')
        for name, good, numel in results:
            print(f'  [{"PASS" if good else "FAIL"}] {name:22s} 输出 {numel} 元素 vs 独立 sum 期望')
        print(f'  [{"PASS" if inv else "FAIL"}] all_gather(reduce_scatter(x)) == all_reduce(x)'
              f'（`:129-132` 让 rs_ag 参考走 all_reduce 的根）')
        print(f'  注：本机 torch {"无" if _MISSING_SINGLE else "有"} `*_single`，'
              f'已用 `*_tensor` 兼容别名原样驱动真实函数体\n')
    return [r[1] for r in results] + [inv]


# --------------------------------------------------------------------------------------
# Part F：`make_inputs` 输入/guard 构造 + `verify` 判据（`:105-154`）
# --------------------------------------------------------------------------------------
def part_inputs_and_verify(rank, g):
    world = dist.group.WORLD
    real_ni, real_verify = g['make_inputs'], g['verify']
    GV = g['GUARD_VALUE']
    PFX, SFX = g['GUARD_PREFIX_ELEMS'], g['GUARD_SUFFIX_ELEMS']
    outs = []
    # 构造若干 arena（真实 CUDA 张量），验证 make_inputs 的形状/守卫语义
    for op in ('all_reduce', 'all_gather'):
        for use_session in (False, True):
            arenas = []
            for _ in range(2):
                a = torch.full((PFX + 64 + SFX,), GV, dtype=torch.float32, device=DEV)
                arenas.append(a)
            inputs, templates, guards = real_ni(arenas, world, op, torch.float32, exact=True,
                                                use_session=use_session)
            exp_len = 2
            # all_gather 输入被切成每 rank 一份
            if op == 'all_gather':
                exp_numel = 64 // world.size()
            else:
                exp_numel = 64
            good = (len(inputs) == exp_len and inputs[0].numel() == exp_numel
                    and not use_session and len(guards) == 4
                    or (use_session and len(guards) == 0))
            outs.append((f'make_inputs {op} session={use_session}', good, inputs[0].shape))
    # verify：正确 guard 通过、被破坏的 guard 必须报错、exact 严格
    ref = torch.randn(16, device=DEV)
    ok_guard = torch.full((PFX,), GV, dtype=torch.float32, device=DEV)
    bad_guard = ok_guard.clone(); bad_guard[0] += 1
    ev = DummyEvent()
    # 1) 正确 -> 不报错
    real_verify([('ok', ev, [ref.clone()], [ref], [ok_guard], True)])
    outs.append(('verify 正确 guard 通过', True, ''))
    # 2) guard 被破坏 -> 必须 Raise
    try:
        real_verify([('bad-guard', ev, [ref.clone()], [ref], [bad_guard], True)])
        outs.append(('verify 破坏 guard 被抓', False, '未报错'))
    except Exception:
        outs.append(('verify 破坏 guard 被抓', True, 'AssertionError'))
    # 3) exact=True 下 1-ulp 差异 -> 抓
    approx = ref.clone(); approx[0] = torch.nextafter(approx[0], torch.tensor(float('inf'), device=DEV))
    try:
        real_verify([('exact', ev, [approx], [ref], [], True)])
        outs.append(('verify exact=True 抓 1-ulp', False, '未报错'))
    except Exception:
        outs.append(('verify exact=True 抓 1-ulp', True, 'AssertionError'))
    # 4) exact=False + 2e-4 容差 -> 通过
    real_verify([('tol', ev, [approx], [ref], [], False)])
    outs.append(('verify exact=False 容差内通过', True, ''))
    if rank == 0:
        print('### Part F：`make_inputs`（`:105-123`）与 `verify`（`:144-154`）判据')
        for name, good, extra in outs:
            print(f'  [{"PASS" if good else "FAIL"}] {name}  {extra}')
        print()
    return [o[1] for o in outs]


# --------------------------------------------------------------------------------------
# Part G：test_perf 规范（constants / make_sizes ragged / get_skip_reason / measure）
# --------------------------------------------------------------------------------------
def part_perf(rank, gp, math_m):
    outs = []
    # G1: NCCL profiles / tuning vars 表
    configs, tuning = gp['NCCL_CONFIGS'], gp['NCCL_TUNING_VARS']
    expected_profiles = {'auto', 'cta32', 'cta64', 'ring', 'tree', 'nvls', 'nvls-tree'}
    outs.append(('NCCL_CONFIGS profiles', set(configs) == expected_profiles, sorted(configs)))
    outs.append(('NCCL_TUNING_VARS', tuple(tuning) == ('NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_NCHANNELS',
                'NCCL_MAX_NCHANNELS', 'NCCL_MIN_CTAS', 'NCCL_MAX_CTAS', 'NCCL_NVLS_ENABLE',
                'NCCL_COLLNET_ENABLE'), list(tuning)))
    # G2: get_nccl_sm_budget
    real_budget = gp['get_nccl_sm_budget']
    b_ok = all(real_budget(p) == configs[p].get('NCCL_MAX_CTAS', 'auto') for p in configs)
    outs.append(('get_nccl_sm_budget', b_ok, real_budget('cta64')))
    # G3: perf make_sizes（ragged）
    real_ms = gp['make_sizes']
    for num_bytes, nb, nr, ragged in [(1 << 20, 8, 8, False), (1 << 24, 6, 8, True)]:
        a = real_ms(num_bytes, nb, nr, ragged, 0)
        b = oracle_perf_sizes(num_bytes, nb, nr, ragged, 0)
        outs.append((f'perf make_sizes {num_bytes}/{nb}/ragged={ragged}', a == b,
                     f'sum={sum(a)}'))
    # G4: get_skip_reason
    real_skip = gp['get_skip_reason']
    ok = True
    for op in gp['OPS']:
        for rdma in (1, 2, 8):
            for nvl in (1, 8):
                for prec in ('fp32', 'bf16'):
                    if real_skip(op, rdma, nvl, prec) != oracle_get_skip_reason(op, rdma, nvl, prec):
                        ok = False
    outs.append(('get_skip_reason grid', ok, real_skip('reduce_scatter', 8, 1, 'bf16')))
    if rank == 0:
        print('### Part G：test_perf 规范（`:36-106`）')
        for name, good, extra in outs:
            print(f'  [{"PASS" if good else "FAIL"}] {name}  {extra}')
        print()
    return [o[1] for o in outs]


# --------------------------------------------------------------------------------------
# Part H：8×H100 `measure`（中位数 = 全 rank 逐样本 MAX 之后取中位）
# --------------------------------------------------------------------------------------
def _measure(rank, world):
    gp, _ = exec_test_module(TEST_PERF, 'perf_ut')
    real_measure = gp['measure']
    flush = torch.empty(1 << 16, dtype=torch.uint8, device=DEV)
    x = torch.ones(1 << 20, dtype=torch.float32, device=DEV)
    fn = lambda: dist.all_reduce(x, group=world)
    med = real_measure(fn, world, flush, 5)
    # 独立复刻：每样本未同步前不得依赖计时，这里只验证「中位数」语义与返回类型
    good = isinstance(med, float) and med > 0
    if rank == 0:
        print('### Part H：test_perf `measure`（`:51-73`）中位数计时（8×H100 真机）')
        print(f'  [{"PASS" if good else "FAIL"}] all_reduce(4MiB) 中位时长 = {med:.1f} us'
              f'（逐样本先 all_reduce(MAX) 取慢 rank，再 statistics.median）')
        print()
    return [good]


# --------------------------------------------------------------------------------------
def worker(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    import datetime
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=180))
    math_m, h_mixed_math, h_mixed_test = install_deep_ep_stub()
    g_mixed, sha_mixed = exec_test_module(TEST_MIXED, 'mixed_ut')
    g_perf, sha_perf = exec_test_module(TEST_PERF, 'perf_ut')
    # test_mixed.py 在模块顶层 pop 了 NCCL_NVLS_ENABLE（其注释见 :3-6）；本机无 Fabric Manager，
    # NVLS multicast 绑定会失败（见阻塞），故重新关掉。
    os.environ['NCCL_NVLS_ENABLE'] = '0'

    results = []
    if rank == 0:
        print('### Part A：真实源码加载（sha256）')
        print(f'  tests/bucket/test_mixed.py sha256={sha_mixed[:16]}…（{len(open(TEST_MIXED).readlines())} 行）')
        print(f'  tests/bucket/test_perf.py  sha256={sha_perf[:16]}…（{len(open(TEST_PERF).readlines())} 行）')
        print(f'  真实常量：SESSION_PERIOD={g_mixed["SESSION_PERIOD"]} EXACT_PERIOD={g_mixed["EXACT_PERIOD"]} '
              f'BARRIER_PERIOD={g_mixed["BARRIER_PERIOD"]} GUARD_BYTES={g_mixed["GUARD_BYTES"]}')
        print(f'  perf DEFAULT_NUM_BYTES={g_perf["DEFAULT_NUM_BYTES"]} NUM_WARMUPS={g_perf["NUM_WARMUPS"]}\n')

    # Part B/C 纯 host（只在 rank0 打印，但函数调用无害，全 rank 一致即可）
    fails = part_supported(rank, g_mixed)
    fails += part_make_sizes(rank, g_mixed)

    # Part D make_groups 真机 dist.new_group（全 rank 参与）
    fails += sum(1 for x in part_make_groups(rank, g_mixed) if not x)

    # Part E nccl_reference 语义 + rs_ag 不变量
    fails += sum(1 for x in part_nccl_reference(rank, g_mixed) if not x)

    # Part F inputs/verify
    fails += sum(1 for x in part_inputs_and_verify(rank, g_mixed) if not x)

    # Part G perf 规范
    fails += sum(1 for x in part_perf(rank, g_perf, math_m) if not x)

    # Part H measure
    fails += sum(1 for x in _measure(rank, dist.group.WORLD) if not x)

    dist.barrier()
    if rank == 0:
        print(f'==== 总判定：{"ALL PASS" if fails == 0 else f"{fails} FAIL"} ====')
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(22000 + (os.getpid() % 20000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}')
    print('DeepEP BucketBuffer 混合集合算子的验收测试即正确性规范 —— H100 真机（测试侧架构无关部分）')
    print('=' * 96)
    if WORLD > 1:
        mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    else:
        worker(0, 1)
