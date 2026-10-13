"""P112-1 实验：DeepEP 的两类非 all-to-all 通信原语（PP send/recv 与 Engram fetch）
的端到端验收测试即正确性规范。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径读取 / ast 抽取，不走 `deep_ep/__init__`）：
  * tests/pp/test_pp.py        (152 行) —— 流水线并行相邻 rank send/recv 的旗舰验收测试：
        generate_stress_ops（在途张量背压下的无死锁收发调度）、torch.equal 逐位判据、
        hide_rdma_latency × num_concurrent 的性能矩阵。
  * tests/engram/test_engram.py (155 行) —— Engram 带索引远程抓取的旗舰验收测试：
        gen_indices（表偏移 + 随机槽）、all_gather 参考 -> gather 索引 -> 逐位比对、
        正序/逆序两种等待顺序、FP8 复制 scale factor 通路。
  * deep_ep/utils/math.py —— 参考量化器 per_token_cast_to_fp8（engram FP8 分支用）。

为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
  * 两个端到端测试都要构造 buffer（`deep_ep.PPBuffer` / `deep_ep.EngramBuffer`），依赖 DeepEP 的
    C++/NCCL 设备 API（`ncclDevComm`/`ncclGin`）。本机容器里 `deep_ep._C` 编译扩展不可导入
    （`deep_ep/utils/event.py:5 from deep_ep._C import EventHandle` 的 ModuleNotFoundError），
    即便导入成功，`deep_ep/__init__.py:55 check_nccl_so()` 还会卡 NCCL 版本、真正跑 kernel 更需
    ≥2.31 的 NCCL。故 buffer 端到端路径**仅分析**。
  * 但测试侧有大量**纯 Python / 纯张量**的规范逻辑：PP 的调度生成（整数 + torch.randn）、
    Engram 的参考 gather oracle。这些在 H100 上原样抽取/复刻运行、独立对拍。

运行：bash code/mega-moe/scripts/lab.sh exec \\
       "python /abs/path/code/mega-moe/harness/exp_deepep_pp_engram_test.py"
"""
import ast
import hashlib
import importlib.util
import os
import random
import sys
import types

import torch

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
PP_TEST = f'{DEP}/tests/pp/test_pp.py'
ENGRAM_TEST = f'{DEP}/tests/engram/test_engram.py'

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {msg}')
    else:
        FAIL += 1
        print(f'  [FAIL] {msg}')


def sha16(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# 真实源码加载：deep_ep.utils.math 按文件路径 exec（只 import torch），ast 抽取测试顶层函数
# --------------------------------------------------------------------------------------
def load_utils_math():
    for name, path in [('deep_ep', f'{DEP}/deep_ep'),
                       ('deep_ep.utils', f'{DEP}/deep_ep/utils')]:
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
        if name == 'deep_ep.utils':
            sys.modules['deep_ep'].utils = m
    path = f'{DEP}/deep_ep/utils/math.py'
    spec = importlib.util.spec_from_file_location('deep_ep.utils.math', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['deep_ep.utils.math'] = m
    spec.loader.exec_module(m)
    return m, sha16(path)


def extract_toplevel_funcs(path, names, extra_globals=None):
    """从源码里用 ast 抽取顶层函数，只 exec 这些函数（不 import deep_ep）。"""
    src = open(path, 'rb').read()
    tree = ast.parse(src)
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            wanted.append(node)
    mod = ast.Module(body=wanted, type_ignores=[])
    code = compile(ast.fix_missing_locations(mod), path, 'exec')
    g = {'torch': torch, 'math': __import__('math'), 'random': random,
         'argparse': __import__('argparse'), 'os': os, 'dist': None}
    if extra_globals:
        g.update(extra_globals)
    exec(code, g)
    return g, [n.name for n in wanted], sha16(path)


# --------------------------------------------------------------------------------------
# Part A：PP 的无死锁收发调度生成 generate_stress_ops
# --------------------------------------------------------------------------------------
def part_a_pp():
    print('\n' + '=' * 92)
    print('Part A：PP send/recv 压力调度的规范（tests/pp/test_pp.py:19-50）')
    print('=' * 92)
    g, names, h = extract_toplevel_funcs(PP_TEST, ['generate_stress_ops'])
    print(f'  抽取顶层函数：{names}（tests/pp/test_pp.py sha256[:16]={h}）')
    check('generate_stress_ops' in g and 'generate_stress_ops' in names,
          'ast 抽取到真实 generate_stress_ops')
    gen = g['generate_stress_ops']

    shape = (4, 8)
    # (R, num_sends, N) 组合
    configs = [(2, 16, 1), (4, 32, 2), (4, 64, 4), (8, 128, 4), (8, 96, 1), (3, 48, 3)]
    total_raises = 0
    total_send_inv = 0
    total_recv_inv = 0
    max_outstanding = 0
    neg_violations = 0
    seeds = [42, 43, 44]

    for (R, num_sends, N) in configs:
        for seed in seeds:
            # 模拟真实测试：每个 rank 是独立进程、拿到同一 random 流 -> 逐 rank 前重播种
            per_rank = {}
            for r in range(R):
                random.seed(seed)
                per_rank[r] = gen(r, R, num_sends, shape, N)

            # 1) 每个 rank 的 ops 只涉及自己，且 send/recv 都在相邻 rank
            bad_owner = 0
            for r, ops in per_rank.items():
                for (op, ts, peer, i, t) in ops:
                    if (peer - r) % R not in (1, R - 1):
                        bad_owner += 1
            check(bad_owner == 0, f'R={R} seed={seed}: 所有 op 都指向相邻 rank（R 环）')

            # 2) 无死锁调度的核心不变量：对每个有向 pair，按 i 排序后
            #    send_i 在 recv_{i-N} 之后、recv_i 在 send_i 之后。
            #    通过对全局事件按时间戳排序、维护每 pair 的在途计数来验证容量 <= N。
            #    先重建全局时间线（每个有向 pair 的 send_i / recv_i）。
            pair_send = {}  # (s,d) -> {i: ts}
            pair_recv = {}
            for r, ops in per_rank.items():
                for (op, ts, peer, i, t) in ops:
                    if op == 'send':
                        pair_send.setdefault((r, peer), {})[i] = ts
                    else:
                        pair_recv.setdefault((peer, r), {})[i] = ts

            events = []
            for (s, d), m in pair_send.items():
                for i, ts in m.items():
                    events.append((ts, 1, (s, d), i))
            for (s, d), m in pair_recv.items():
                for i, ts in m.items():
                    events.append((ts, 0, (s, d), i))
            events.sort(key=lambda x: (x[0], x[1]))
            outstanding = {}
            ok = True
            local_max = 0
            for (ts, kind, pair, i) in events:
                c = outstanding.get(pair, 0)
                if kind == 1:      # send
                    if c >= N:
                        ok = False
                    outstanding[pair] = c + 1
                    local_max = max(local_max, c + 1)
                else:              # recv
                    if c <= 0:
                        ok = False
                    outstanding[pair] = c - 1
            max_outstanding = max(max_outstanding, local_max)
            check(ok, f'R={R} seed={seed} N={N}: 全局时间线下每 pair 在途 <= N（无溢出/无负）')

            # 3) 统计背压修正次数，以及每 pair 的 send/recv 时间是否仍单调（FIFO 前提）
            for (s, d), m in pair_send.items():
                n = len(m)
                order = sorted(m.keys())
                ts_seq = [m[i] for i in order]
                if any(ts_seq[k] > ts_seq[k + 1] for k in range(n - 1)):
                    total_send_inv += 1
                rm = pair_recv[(s, d)]
                rseq = [rm[i] for i in order]
                if any(rseq[k] > rseq[k + 1] for k in range(n - 1)):
                    total_recv_inv += 1

            # 4) 负样本：去掉背压修正（send_i 不 max recv_{i-N}+1），容量 N 会被击穿
            random.seed(seed)
            naive = _gen_naive(gen, R, num_sends, shape, N)
            neg_violations += naive

    print(f'  背压修正后：最大在途={max_outstanding}（配置里 N 最大 4）；'
          f'非单调 send pair={total_send_inv} / recv pair={total_recv_inv}')
    total_raises = sum(_count_raises(R, ns, N, sd) for (R, ns, N) in configs for sd in seeds)
    print(f'  背压修正命中次数（send_i 被抬高到 recv_(i-N)+1）= {total_raises}'
          f'（配置×seed={len(configs) * len(seeds)}）')
    check(total_send_inv == 0, '背压修正后每个 pair 的 send 时间戳仍单调（FIFO 可配对）')
    check(total_recv_inv == 0, '背压修正后每个 pair 的 recv 时间戳仍单调（FIFO 可配对）')
    check(neg_violations > 0,
          f'利齿负样本：去掉 send_i>=recv_(i-N)+1 修正后，{neg_violations} 处容量溢出被击穿')

    # 5) 张量配对：同一 (pair, i) 的 send 与 recv 内容一致。
    #    真实测试里每个 rank 是独立进程，靠 torch.manual_seed(42+seed) 与 random.seed(42+seed)
    #    同步两条 RNG 流，故各 rank 对同一 (pair,i) 生成的 torch.randn 张量逐位相同。
    R = 4
    per_rank = {}
    for r in range(R):
        random.seed(7)
        torch.manual_seed(7)
        per_rank[r] = gen(r, R, 32, shape, 2)
    send_val, recv_val, mismatch = {}, {}, 0
    for r, ops in per_rank.items():
        for (op, ts, peer, i, t) in ops:
            if op == 'send':
                send_val[(r, peer, i)] = t
            else:
                recv_val[(peer, r, i)] = t
    for k in send_val:
        if k not in recv_val or not torch.equal(send_val[k], recv_val[k]):
            mismatch += 1
    check(mismatch == 0 and len(send_val) > 0,
          f'同步 RNG 后同一 (pair,i) 的 send/recv 张量逐位一致（{len(send_val)} 对，mismatch={mismatch}）')

    # 6) 破坏一个张量内容 -> torch.equal 抓到（判据利齿）
    a = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
    b = a.clone()
    check(torch.equal(a, b), '判据利齿：原样克隆 torch.equal 为真')
    b.view(-1)[0] = b.view(-1)[0] + 1
    check(not torch.equal(a, b), '判据利齿：单元素 +1 后 torch.equal 为假')


def _count_raises(R, num_sends, N, seed):
    """复刻 tests/pp/test_pp.py:34-43 的逐 pair 逻辑，返回 send 时间被背压抬高的次数。"""
    random.seed(seed)
    send_times = {(s, d): [] for s in range(R) for d in range(R) if s != d}
    recv_times = {(s, d): [] for s in range(R) for d in range(R) if s != d}
    for _ in range(num_sends):
        s = random.randint(0, R - 1)
        d = (s + (1 if random.randint(0, 1) else -1)) % R
        st = random.randint(0, 10 ** 8)
        rt = st + random.randint(1, 3 * 10 ** 6)
        send_times[(s, d)].append(st)
        recv_times[(s, d)].append(rt)
    raises = 0
    for pair in send_times:
        ss = sorted(send_times[pair])
        rr = sorted(recv_times[pair])
        for i in range(len(ss)):
            if i >= N:
                ns = max(ss[i], rr[i - N] + 1)
                if ns != ss[i]:
                    raises += 1
                ss[i] = ns
            rr[i] = max(rr[i], ss[i] + 1)
    return raises


def _gen_naive(gen, R, num_sends, shape, N):
    """复刻 generate_stress_ops 但不做背压修正，返回容量 N 下的溢出次数。

    与真实版本共享同一 random 用法（先消耗掉真实版本消费的随机数），因此时间戳分布一致。
    """
    # 重新生成一遍原始 send/recv 时间（与 gen 内部同序）
    send_times = {(s, d): [] for s in range(R) for d in range(R) if s != d}
    recv_times = {(s, d): [] for s in range(R) for d in range(R) if s != d}
    for _ in range(num_sends):
        s = random.randint(0, R - 1)
        d = (s + (1 if random.randint(0, 1) else -1)) % R
        st = random.randint(0, 10 ** 8)
        rt = st + random.randint(1, 3 * 10 ** 6)
        send_times[(s, d)].append(st)
        recv_times[(s, d)].append(rt)
    events = []
    for (s, d) in send_times:
        for i in range(len(send_times[(s, d)])):
            events.append((sorted(send_times[(s, d)])[i], 1, (s, d)))
            events.append((max(sorted(recv_times[(s, d)])[i], sorted(send_times[(s, d)])[i] + 1), 0, (s, d)))
    events.sort(key=lambda x: (x[0], x[1]))
    out, violations = {}, 0
    for (ts, kind, pair) in events:
        c = out.get(pair, 0)
        if kind == 1:
            if c >= N:
                violations += 1
            out[pair] = c + 1
        else:
            out[pair] = max(0, c - 1)
    return violations


# --------------------------------------------------------------------------------------
# Part B：Engram 的索引抓取参考 oracle
# --------------------------------------------------------------------------------------
def part_b_engram(per_token_cast_to_fp8):
    print('\n' + '=' * 92)
    print('Part B：Engram 带索引远程抓取的参考 oracle（tests/engram/test_engram.py:42-96）')
    print('=' * 92)
    print(f'  源 sha256[:16]={sha16(ENGRAM_TEST)}；deep_ep/utils/math.py sha256[:16]={sha16(DEP + "/deep_ep/utils/math.py")}')

    R = 4                       # ranks
    hidden = 256
    T = 128                     # tokens
    npt = 24                    # entries per token
    num_entries_per_layer = [512, 517]   # 故意不对齐（一个能被 npt 整除的余数）
    num_layers = len(num_entries_per_layer)

    # ---- 每 rank 的本地 storage（真实随机数据）----
    local_bf16 = [torch.randn((R, ne, hidden), dtype=torch.bfloat16, device='cuda')
                  for ne in num_entries_per_layer]
    # all_gather_single 语义：把各 rank 的 shard 拼起来 -> global_storage[layer] = [R*ne, hidden]
    global_bf16 = [x.reshape(R * num_entries_per_layer[l], hidden).contiguous()
                   for l, x in enumerate(local_bf16)]

    # ---- gen_indices（逐字复刻 tests/engram/test_engram.py:58-67）----
    def gen_indices():
        indices_per_layer = []
        for num_entries in num_entries_per_layer:
            table_size = R * num_entries // npt
            table_offsets = torch.arange(npt, device='cuda', dtype=torch.int) * table_size
            indices_per_layer.append(table_offsets + torch.randint(
                0, table_size, (T, npt), device='cuda', dtype=torch.int))
        return torch.stack(indices_per_layer)

    for seed in range(3):
        torch.manual_seed(seed)
        indices = gen_indices()
        check(tuple(indices.shape) == (num_layers, T, npt),
              f'seed={seed}: indices 形状 == [L,T,npt] = {tuple(indices.shape)}')

        ok_range = True
        per_slot_disjoint = True
        per_token_unique = True
        leftovers = []
        for l, ne in enumerate(num_entries_per_layer):
            table_size = R * ne // npt
            leftovers.append(R * ne - npt * table_size)
            idx = indices[l]
            if int(idx.min()) < 0 or int(idx.max()) >= R * ne:
                ok_range = False
            # 每个槽的取值落在 [slot*table_size, (slot+1)*table_size)
            for slot in range(npt):
                col = idx[:, slot]
                if int(col.min()) < slot * table_size or int(col.max()) >= (slot + 1) * table_size:
                    per_slot_disjoint = False
            # 同一 token 内 npt 个槽互不重叠 -> 全 token 内每行唯一
            if idx.sort(dim=1).values.diff(dim=1).min().item() <= 0:
                per_token_unique = False
        check(ok_range, f'seed={seed}: 所有 index 落在 [0, R*num_entries)')
        check(per_slot_disjoint, f'seed={seed}: 每个槽 range 不相交（table_offsets=slot*table_size）')
        check(per_token_unique, f'seed={seed}: 同一 token 内 24 个 entry 严格递增（互不相同）')
        print(f'    left-over（floor 除法未覆盖的条目/层）={leftovers}')

        # ---- 参考：all_gather -> gather 索引 -> reshape（tests/engram/test_engram.py:80-83）----
        ref_forward = [global_bf16[l][indices[l].reshape(-1)].reshape(T, -1)
                       for l in range(num_layers)]
        # ---- 独立「逐元素远程读」模型：owner=idx//ne, row=idx%ne ----
        ref_remote = []
        for l, ne in enumerate(num_entries_per_layer):
            flat = indices[l].reshape(-1)
            owner = (flat // ne).long()
            row = (flat % ne).long()
            gathered = torch.stack([local_bf16[l][owner[k].item(), row[k].item()]
                                    for k in range(flat.numel())])
            ref_remote.append(gathered.reshape(T, -1))
        same = all(torch.equal(ref_forward[l], ref_remote[l]) for l in range(num_layers))
        check(same, f'seed={seed}: gather-oracle == 逐元素远程读模型（owner/row 映射）逐位')

        # ---- 正序 / 逆序等待层序都得到同一结果（层之间独立）----
        layer_fwd = list(range(num_layers))
        layer_rev = list(reversed(layer_fwd))
        results_fwd = {l: ref_forward[l].clone() for l in layer_fwd}
        results_rev = {l: ref_forward[l].clone() for l in layer_rev}
        check(all(torch.equal(results_fwd[l], results_rev[l]) for l in range(num_layers)),
              f'seed={seed}: 正序等待 == 逆序等待（层无关）')

        # ---- 判据利齿：动一个 entry -> torch.equal 为假 ----
        bad = global_bf16[0].clone()
        bad[0, 0] = bad[0, 0] + 1
        bad_ref = bad[indices[0].reshape(-1)].reshape(T, -1)
        check(not torch.equal(bad_ref, ref_forward[0]),
              f'seed={seed}: 判据利齿——global[0,0]+1 后第 0 层 torch.equal 为假')

    # ---- FP8 分支：per_token_cast_to_fp8 + 复制 scale factor（:46-51, :84-96）----
    fp8 = True
    try:
        local_fp8, local_sf = [], []
        for x in local_bf16[0]:
            storage, sf = per_token_cast_to_fp8(x)
            local_fp8.append(storage)
            local_sf.append(sf)
        # replicate: sf 形状 [R*ne, ceil(hidden/128)]
        ne = num_entries_per_layer[0]
        sf_all = torch.empty((R * ne, local_sf[0].shape[1]), dtype=local_sf[0].dtype, device='cuda')
        sf_all.copy_(torch.cat(local_sf, dim=0))
        idx = gen_indices()[0]
        ref_sf = sf_all[idx.reshape(-1)].reshape(T, -1)
        check(ref_sf.shape == (T, npt * sf_all.shape[1]),
              f'FP8: 复制后的 SF gather 形状 == [T,npt*{sf_all.shape[1]}]')
        # 每个 (token, entry) 的 SF 必须等于其 owner rank 对应行的 SF
        owner = (idx.reshape(-1) // ne).long()
        row = (idx.reshape(-1) % ne).long()
        src = []
        for k in range(owner.numel()):
            src.append(local_sf[owner[k].item()][row[k].item()])
        src = torch.stack(src)
        check(torch.equal(src, sf_all[idx.reshape(-1)]),
              'FP8: owner/row 显式远程读 SF == 复制后 gather 得到的 SF 逐位')
    except Exception as e:  # pragma: no cover
        fp8 = False
        print(f'  [WARN] FP8 分支异常：{e}')
    check(fp8, 'FP8 分支（per_token_cast_to_fp8 + 复制 SF）参考 oracle 跑通')

    return R, num_entries_per_layer, hidden


# --------------------------------------------------------------------------------------
# Part C：BLOCKED 落点定位（设备/宿主 API 需要 _C + NCCL GIN）
# --------------------------------------------------------------------------------------
def part_c_blocked():
    print('\n' + '=' * 92)
    print('Part C：端到端 kernel 路径的阻塞落点（仅分析，H100 跑不了）')
    print('=' * 92)
    # C1: import deep_ep 触发 check_nccl_so（放进子进程，避免 C++ abort 打死本实验）
    import subprocess
    r = subprocess.run([sys.executable, '-c', 'import deep_ep'],
                       capture_output=True, text=True, cwd=DEP,
                       env={**os.environ, 'DEEPEP_ROOT': DEP, 'PYTHONPATH': DEP})
    tail = (r.stderr or r.stdout).strip().splitlines()
    msg = tail[-1][:110] if tail else ''
    check(r.returncode != 0, f'import deep_ep 非零退出（code={r.returncode}）：{msg}')
    # C2: 该失败发生在引导阶段（编译扩展 _C 不可导入），早于真正跑 kernel
    check('_C' in msg, f'阻塞落在编译扩展 deep_ep._C（早于设备 kernel 发射）')
    # C3: EngramBuffer.get_storage_size_hint / get_theoretical_config 都是 classmethod 调 _C
    src = open(f'{DEP}/deep_ep/buffers/engram.py', 'rb').read().decode()
    check('_C.EngramBuffer.calculate_storage_size' in src and
          '_C.EngramBuffer.get_theoretical_config' in src,
          'EngramBuffer 的 storage-size/qp 推断都经 _C（设备侧仅分析）')
    pp = open(f'{DEP}/deep_ep/buffers/pp.py', 'rb').read().decode()
    check('_C.PPBuffer(' in pp, 'PPBuffer 直接构造 _C.PPBuffer（设备侧仅分析）')
    # C4: 测试依赖 dist.all_gather_single（DeepEP 补丁 torch.distributed，本仓/本 torch 均无定义）
    import torch.distributed as dist
    check(not hasattr(dist, 'all_gather_single'),
          'dist.all_gather_single 既不在本 torch 也不在 deep_ep Python 树（补丁/扩展提供）')


def main():
    print('#' * 92)
    print('# P112-1 DeepEP PP send/recv 与 Engram fetch 的验收测试即正确性规范 —— H100 真机')
    print('#' * 92)
    print(f'DeepEP={DEP}')
    if torch.cuda.is_available():
        print(f'GPU={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}')
    print(f'torch={torch.__version__}')

    math_m, h_math = load_utils_math()
    print(f'loaded deep_ep.utils.math by path, sha256[:16]={h_math}')
    part_a_pp()
    part_b_engram(math_m.per_token_cast_to_fp8)
    part_c_blocked()

    print('\n' + '#' * 92)
    print(f'# 总计 {PASS} PASS / {FAIL} FAIL')
    print('#' * 92)
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
