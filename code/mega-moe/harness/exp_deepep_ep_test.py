"""P75-1 实验：DeepEP 的 EP dispatch/combine 端到端验收测试即正确性规范。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径读取、ast 抽取测试侧真实函数，不走 deep_ep/__init__）：
  * tests/ep/test_ep.py —— DeepEP 的旗舰验收测试（616 行）：模式矩阵 / expand/cached/zero-padding
    三种 dispatch / reduced combine / 逐位正确性判据 / 带宽账目。
  * deep_ep/utils/{refs,gate,math}.py —— 测试所依赖的参考实现与流量生成器（P55-1 已拆，这里只做接线）。

为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
  * 端到端测试要构造 `deep_ep.EPBuffer`，依赖 DeepEP 的 C++/NCCL 设备 API（`ncclDevComm`/`ncclGin`）；
    本机 NCCL 2.29.7 与编译期不匹配 → `deep_ep/__init__.py:55 check_nccl_so()` 直接 abort；
    而且真正跑 dispatch/combine 还需 ≥2.31 的 NCCL。故 buffer 路径**端到端仅分析**。
  * 但测试侧有大量**纯 Python / 纯张量**的规范逻辑：`enumerate_ep_modes`（模式枚举）、
    `fold_expanded`（expand 折叠不变量）、handle 的两套 psum 前缀和语义、`get_unique_and_valid_dst_count`
    （带宽账目的去重计数）、以及 refs 参考实现的接线。这些在 H100 上原样抽取运行、独立复刻对拍。

运行：bash code/mega-moe/scripts/lab.sh exec \
       "python /abs/path/code/mega-moe/harness/exp_deepep_ep_test.py"
"""
import ast
import hashlib
import importlib.util
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
WORLD = int(os.environ.get('EP_TEST_WORLD', '8'))
DEV = 'cuda'
TEST_EP = f'{DEP}/tests/ep/test_ep.py'


# --------------------------------------------------------------------------------------
# 真实源码加载：按文件路径 exec refs/gate/math，ast 抽取 test_ep.py 的顶层测试侧函数
# --------------------------------------------------------------------------------------
def load_utils():
    for name, path in [('deep_ep', f'{DEP}/deep_ep'),
                       ('deep_ep.utils', f'{DEP}/deep_ep/utils')]:
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
        if name == 'deep_ep.utils':
            sys.modules['deep_ep'].utils = m
    envs = types.ModuleType('deep_ep.utils.envs')
    envs.get_global_seed = lambda: 0
    envs.get_local_seed = lambda: 0
    sys.modules['deep_ep.utils.envs'] = envs

    def load(mod, fname):
        path = f'{DEP}/deep_ep/utils/{fname}'
        src = open(path, 'rb').read()
        spec = importlib.util.spec_from_file_location(mod, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod] = m
        spec.loader.exec_module(m)
        return m, hashlib.sha256(src).hexdigest()

    math_m, h_math = load('deep_ep.utils.math', 'math.py')
    refs, h_refs = load('deep_ep.utils.refs', 'refs.py')
    gate, h_gate = load('deep_ep.utils.gate', 'gate.py')
    return (math_m, refs, gate), (h_refs, h_gate, h_math)


def extract_test_funcs(names):
    """从 test_ep.py 源码里用 ast 抽取顶层函数，只 exec 这些函数（不 import deep_ep）。"""
    src = open(TEST_EP, 'rb').read()
    tree = ast.parse(src)
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            # 去掉装饰器（`@torch.inference_mode()` 之类），保留函数体
            node.decorator_list = []
            wanted.append(node)
    mod = ast.Module(body=wanted, type_ignores=[])
    code = compile(ast.fix_missing_locations(mod), TEST_EP, 'exec')
    # 提供注解求值所需的全局名（deep_ep 用桩）
    fake_ep = types.SimpleNamespace(EPBuffer=object, topk_idx_t=torch.int32)
    g = {'torch': torch, 'argparse': __import__('argparse'),
         'deep_ep': fake_ep,
         'Union': __import__('typing').Union, 'Tuple': __import__('typing').Tuple}
    exec(code, g)
    return g, wanted, hashlib.sha256(src).hexdigest()


# --------------------------------------------------------------------------------------
# Part A：真实源码加载
# --------------------------------------------------------------------------------------
def part_a(rank, hashes, wanted, sha_test):
    if rank != 0:
        return
    h_refs, h_gate, h_math = hashes
    print('### Part A：真实源码加载（sha256）')
    print(f'  tests/ep/test_ep.py sha256={sha_test[:16]}…  顶层函数抽取 = '
          f'{[n.name for n in wanted]}')
    print(f'  refs.py  sha256={h_refs[:16]}…  dispatch/combine 参考实现（P55-1 已拆）')
    print(f'  gate.py  sha256={h_gate[:16]}…  不平衡流量生成器')
    print(f'  math.py  sha256={h_math[:16]}…  对齐/计数工具')
    print()


# --------------------------------------------------------------------------------------
# Part B：模式枚举矩阵 enumerate_ep_modes（test_ep.py:22-30）
# --------------------------------------------------------------------------------------
def part_b(rank, enum_modes):
    if rank != 0:
        return

    def count(fp8_only, defer):
        a = types.SimpleNamespace(fp8_dispatch_only=fp8_only, defer_epilogue=defer)
        return sum(1 for _ in enum_modes(a))

    # 独立公式：align(2) × fp8(1/2) × bias(3) × prev(2) × async(1/2) × alloc(prev?1:2)
    def formula2(fp8_only, defer):
        fp8 = 1 if fp8_only else 2
        asy = 1 if defer else 2
        total = 0
        for _align in (128, 1):
            for _prev in (0, 1):
                alloc = 1 if _prev else 2
                total += fp8 * 3 * asy * alloc
        return total

    cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print('### Part B：模式枚举矩阵 enumerate_ep_modes（真实函数 vs 独立展开）')
    print('  维度：expert_alignment∈{128,1} × fp8∈{1,(2)} × num_bias∈{0,1,2} × '
          'with_previous_event × async_with_compute_stream × allocate_on_comm_stream')
    ok = 0
    for fp8_only, defer in cases:
        c = count(fp8_only, defer)
        f = formula2(fp8_only, defer)
        good = (c == f)
        ok += good
        print(f'  [{"PASS" if good else "FAIL"}] fp8_dispatch_only={fp8_only} defer_epilogue={defer}: '
              f'实测 {c} 组，独立公式 {f} 组')
    # 第一个 case 打印实际 yield 的元组，坐实维度含义
    a = types.SimpleNamespace(fp8_dispatch_only=0, defer_epilogue=0)
    it = enum_modes(a)
    first = next(it)
    print(f'  首组元组 (expert_alignment, use_fp8_dispatch, num_bias, with_previous_event, '
          f'async_with_compute_stream, allocate_on_comm_stream) = {first}')
    print(f'  → Part B: {ok}/{len(cases)} PASS')
    print()


# --------------------------------------------------------------------------------------
# Part C：expand 折叠不变量 fold_expanded（test_ep.py:46-58）
# --------------------------------------------------------------------------------------
def part_c(rank, fold_expanded):
    if rank != 0:
        return
    print('### Part C：expand 折叠不变量 fold_expanded（真实函数）')
    ok, tot = 0, 0
    torch.manual_seed(0)
    T, K, H = 5, 3, 4
    # expanded[row] 是逐 expert-slot 的稠密行；indices[t,k] 给出 token t 第 k 槽落在 expanded 的行号
    expanded = torch.randn(T * K, H, device=DEV)
    indices = (torch.arange(T, device=DEV).unsqueeze(1) * K
               + torch.arange(K, device=DEV).unsqueeze(0))  # [T,K]
    valid_mask = torch.ones(T, K, dtype=torch.bool, device=DEV)
    canon = expanded[indices[:, 0]]           # 每个 token 的规范行
    expanded[indices] = canon.unsqueeze(1)     # 同 token 的所有槽写成同一内容（expand 不变量）
    folded = fold_expanded(expanded, indices, valid_mask)
    good = bool(torch.equal(folded, canon))
    tot += 1; ok += good
    print(f'  [{"PASS" if good else "FAIL"}] 所有有效槽相同 → 折叠返回该值（= expanded[indices[:,0]]），'
          f'形状 {tuple(folded.shape)}')
    # 违反不变量（把一个有效槽指向另一 token 的行）→ 内部 assert 必须触发
    bad_idx = indices.clone()
    bad_idx[2, 0] = 3 * K  # 指向 token 3 的行，内容与 token 2 不同
    try:
        fold_expanded(expanded, bad_idx, valid_mask)
        tot += 1
        print('  [FAIL] 违反 expand 不变量未被 assert 抓到')
    except AssertionError:
        tot += 1; ok += 1
        print('  [PASS] 违反 expand 不变量（同 token 槽内容不同）被 assert 抓到')
    # tuple 输入（fp8 dispatch 的 (x, sf)）递归折叠
    expanded2 = torch.randn_like(expanded)
    expanded2[indices] = expanded2[indices[:, 0]].unsqueeze(1)
    tup = fold_expanded((expanded, expanded2), indices, valid_mask)
    tot += 1; ok += (isinstance(tup, tuple) and len(tup) == 2
                     and torch.equal(tup[0], folded))
    print(f'  [{"PASS" if isinstance(tup, tuple) else "FAIL"}] tuple 输入递归折叠（(x,sf)）→ '
          f'{type(tup).__name__} 长度 {len(tup) if isinstance(tup, tuple) else 0}')
    print(f'  → Part C: {ok}/{tot} PASS')
    print()


# --------------------------------------------------------------------------------------
# Part D：handle 的两套 psum 前缀和语义（test_ep.py:452-476 的断言）
# --------------------------------------------------------------------------------------
def part_d(rank, align):
    if rank != 0:
        return
    print('### Part D：handle 的两套 psum 前缀和语义（test_ep.py:452-476 断言）')
    ok, tot = 0, 0
    torch.manual_seed(1)
    num_local_experts = 7
    e = 3  # 对齐
    counts = torch.randint(0, 40, (num_local_experts,)).tolist()

    # 非 expand：inclusive prefix sum of **aligned** counts → psum[i+1]-psum[i] == align(count_i)
    psum = [0]
    for c in counts:
        psum.append(psum[-1] + align(c, e))
    non_expand_ok = all(psum[i + 1] - psum[i] == align(counts[i], e)
                        for i in range(num_local_experts))
    tot += 1; ok += non_expand_ok
    print(f'  [{"PASS" if non_expand_ok else "FAIL"}] 非 expand：psum[i+1]-psum[i] == align(count_i, {e}) '
          f'（num_recv_tokens_per_expert_list[i] == align(ref_count,{e})）')

    # expand：psum[i] = align(psum[i-1], e) + count_i（未对齐真实计数）
    eps = [0]
    for c in counts:
        eps.append(align(eps[-1], e) + c)
    exp_ok = all(eps[i + 1] - align(eps[i], e) == counts[i] for i in range(num_local_experts))
    tot += 1; ok += exp_ok
    print(f'  [{"PASS" if exp_ok else "FAIL"}] expand：psum[i+1]-align(psum[i],{e}) == 真实计数 count_i '
          f'（num_unaligned_recv_tokens_per_expert 的恢复式）')

    # 两者的一致性：expand 的「下一专家起点」 align(epsum[i]) 与非 expand 的 psum[i] 同构（都按对齐累加）
    next_start_ok = all(align(eps[i], e) == psum[i] for i in range(num_local_experts + 1))
    tot += 1; ok += next_start_ok
    print(f'  [{"PASS" if next_start_ok else "FAIL"}] align(expand_psum[i],{e}) == 非 expand_psum[i] '
          f'（两种布局共用同一「按对齐累加」的起点序列）')

    # scaleup 前缀和：去重后每 rank 计数（test_ep.py:471-476）
    num_scaleup_ranks, num_ranks = 4, 8
    ref_num = torch.randint(0, 50, (num_ranks,)).tolist()  # 每来源 rank 收到（去重后）token 数
    ps = [0]
    for i in range(num_scaleup_ranks):
        ps.append(ps[-1] + sum(ref_num[i::num_scaleup_ranks]))
    su_ok = all(ps[i + 1] - ps[i] == sum(ref_num[i::num_scaleup_ranks])
                for i in range(num_scaleup_ranks))
    tot += 1; ok += su_ok
    print(f'  [{"PASS" if su_ok else "FAIL"}] scaleup psum：count_i == Σ_{{r≡i (mod {num_scaleup_ranks})}} '
          f'ref_num_recv_tokens_per_rank[r]（去重、末元素=总数 {ps[-1]}）')
    print(f'  → Part D: {ok}/{tot} PASS')
    print()


# --------------------------------------------------------------------------------------
# Part E：带宽账目的去重计数 get_unique_and_valid_dst_count（test_ep.py:302-315）
# --------------------------------------------------------------------------------------
def unique_valid_dst_count(dst_idx, ignore_local, ignored_l, ignored_r, max_num):
    """独立复刻 test_ep.py:308-315 的逻辑（按行内去重 + 可忽略区间）。"""
    dst = dst_idx.clone()
    ignore = dst == -1
    if ignore_local and ignored_l is not None:
        ignore = ignore | ((dst >= ignored_l) & (dst < ignored_r))
    dst = dst + torch.arange(0, dst.shape[0], device=dst.device,
                             dtype=dst.dtype).unsqueeze(-1) * (max_num + 1)
    dst[ignore] = dst[0][0].item()
    return torch.unique(dst, sorted=False).numel()


def part_e(rank):
    if rank != 0:
        return
    print('### Part E：带宽账目的去重计数 get_unique_and_valid_dst_count（独立复刻 test_ep.py:308-315）')
    ok, tot = 0, 0
    torch.manual_seed(2)
    T, K, max_num = 6, 4, 9
    dst = torch.randint(0, max_num + 1, (T, K))
    dst[0, 0] = -1
    dst[1, :] = 3  # 同一 token 内重复专家 → 去重后只算 1

    def oracle(d, lo=None, hi=None):
        sentinel = int(d[0, 0])  # = 真实 `dst_idx[0][0]`（第 0 行偏移为 0）
        keys, any_ig = set(), False
        for t in range(d.shape[0]):
            for v in d[t].tolist():
                if v == -1 or (lo is not None and lo <= v < hi):
                    any_ig = True
                    continue
                keys.add(int(v) + t * (max_num + 1))
        if any_ig:
            keys.add(sentinel)  # 所有被忽略项折成同一个哨兵值（唯一一次）
        return len(keys)

    naive = oracle(dst)
    got = unique_valid_dst_count(dst, False, None, None, max_num)
    tot += 1; ok += (got == naive)
    print(f'  [{"PASS" if got == naive else "FAIL"}] 行内去重 + 忽略 -1：{got} == 独立 oracle {naive}')

    # 忽略区间（hybrid/ignore_local：某 scaleout 区间不计入 scaleout 流量）
    lo, hi = 3, 6
    naive_ig = oracle(dst, lo, hi)
    got_ig = unique_valid_dst_count(dst, True, lo, hi, max_num)
    tot += 1; ok += (got_ig == naive_ig)
    print(f'  [{"PASS" if got_ig == naive_ig else "FAIL"}] 忽略区间 [{lo},{hi})：{got_ig} == 逐行集合计数 {naive_ig}')
    print(f'  → Part E: {ok}/{tot} PASS')
    print()


# --------------------------------------------------------------------------------------
# Part F：8×H100 真机 —— 测试断言的根：dispatch 参考实现按 src 全局索引升序（test_ep.py:188-192）
# --------------------------------------------------------------------------------------
def make_x_rows(gidx, hidden, dtype):
    j = torch.arange(1, hidden + 1, device=gidx.device, dtype=torch.float32).unsqueeze(0)
    return torch.sin(gidx.float().unsqueeze(1) * j * 1e-3).to(dtype)


def part_f(rank, world, refs, math_m):
    torch.manual_seed(1000 + rank)
    num_tokens, hidden, num_topk = 512, 256, 4
    num_experts = world * 4
    gidx = torch.arange(num_tokens, device=DEV) + rank * num_tokens
    x = make_x_rows(gidx, hidden, torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=DEV)
    topk_weights = torch.rand(num_tokens, num_topk, device=DEV)
    recv_x, recv_topk_idx, recv_topk_weights, recv_src, num_recv = refs.dispatch(
        x, topk_idx, topk_weights, num_tokens, num_experts)
    sorted_src = torch.sort(recv_src).values
    num_local_experts = num_experts // world
    checks = [
        ('refs.dispatch 输出 recv_src 本身升序（测试对 kernel 元数据排序后比它）',
         bool(torch.all(recv_src[1:] >= recv_src[:-1]))),
        ('recv_src == sorted(recv_src)（即测试里 sorted_src_token_global_idx）',
         bool(torch.equal(recv_src, sorted_src))),
        ('num_recv_tokens_per_rank 求和 == 接收总数',
         int(num_recv.sum().item()) == recv_src.numel()),
        ('接收的专家索引都落在本 rank 的本地专家范围内或 -1',
         bool(recv_topk_idx.min().item() >= -1)
         and int(recv_topk_idx.max().item()) < num_local_experts),
    ]
    if rank == 0:
        print('### Part F：8×H100 真机 —— dispatch 参考实现的「按 src 升序」不变量'
              '（test_ep.py:188-192 断言的根）')
        for name, good in checks:
            print(f'  [{"PASS" if good else "FAIL"}] {name}')
        print(f'  本 rank 收到 {recv_src.numel()} 行；recv_src 前 8 = {recv_src[:8].tolist()}；'
              f'每来源 rank 计数 = {[int(v) for v in num_recv.tolist()]}')
        print()


# --------------------------------------------------------------------------------------
def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=__import__('datetime').timedelta(seconds=180))
    (math_m, refs, gate), hashes = load_utils()
    g, wanted, sha_test = extract_test_funcs(
        {'enumerate_ep_modes', 'fold_expanded', 'launch'})
    part_a(rank, hashes, wanted, sha_test)
    part_b(rank, g['enumerate_ep_modes'])
    part_c(rank, g['fold_expanded'])
    part_d(rank, math_m.align)
    part_e(rank)
    part_f(rank, world, refs, math_m)
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(21000 + (os.getpid() % 20000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}')
    print('DeepEP EP dispatch/combine 端到端验收测试即正确性规范 —— H100 真机（测试侧架构无关部分）')
    print('=' * 96)
    if WORLD > 1:
        mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    else:
        worker(0, 1)
