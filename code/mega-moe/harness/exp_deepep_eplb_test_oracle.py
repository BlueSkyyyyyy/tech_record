"""P90-1 实验：DeepEP EPLB「权重预取 / 梯度归约」的验收测试即正确性规范。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径 ast 抽取测试侧真实函数，不走 deep_ep/__init__，
以绕过 `check_nccl_so`；本机系统 NCCL 2.29.7 与 pip-linked 2.32.3 二进制不一致）：
  * tests/ep/test_prefetch_weights.py（250 行）—— 前向：把本 rank 的专家权重推到各 peer 的冗余槽。
  * tests/ep/test_reduce_grads.py（256 行）      —— 反向：把各 peer 冗余副本的 FP32 梯度加回 owner。
  * deep_ep/utils/envs.py —— get_nvlink_gbs / dist_print / init_dist（本实验直接复刻 get_nvlink_gbs）。

为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
  * 端到端测试要构造 `deep_ep.EPBuffer` 并调用 `lb_prefetch_weights` / `lb_reduce_grads`，
    其设备 kernel 用 **NCCL 设备 API（`ncclDevComm`/`ncclGin`）**；本机 NCCL 2.29.7 的头不提供该
    API（与 P43-1/doc 53、P56-1/doc 86 同源「阻塞」）。故 kernel 路径**端到端仅分析**。
  * 但测试把「什么叫对」全部编码在**纯 Python / 纯张量**的 oracle 里：`SENTINEL`、`make_blob`
    （ramp + 唯一 key）、`make_redundancy_mapping`（own-expert 排除 / empty_ratio / ring vs spread）、
    `check`（哨兵 vs blob 逐字节）、`make_pattern`（精确 fp32 + 原子加顺序无关）、
    `expected_expert_grads`（归约期望值）、`seed_buffers`。这些在 H100 上原样抽取运行、独立复刻对拍。

运行：bash code/mega-moe/scripts/lab.sh exec \
       "python /abs/path/code/mega-moe/harness/exp_deepep_eplb_test_oracle.py"
"""
import ast
import hashlib
import os
import re
import subprocess
import sys
import types

import torch

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
DEV = 'cuda'
PF_TEST = f'{DEP}/tests/ep/test_prefetch_weights.py'
RG_TEST = f'{DEP}/tests/ep/test_reduce_grads.py'

# `deep_ep.get_num_tma_alignment()` 的真值，来自 `deep_ep/include/deep_ep/common/compiled.cuh`
# 的 `kNumTMAAlignmentBytes`（python_api.cpp:25 转出）。本机不能 import _C，故直接取常量。
ALIGN_TMA = 32

RESULT = []


def record(name, ok):
    RESULT.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')


def section(title):
    print(f'\n### {title}')


# --------------------------------------------------------------------------------------
# 真实源码加载：ast 抽取顶层函数（去装饰器）+ 注入模块级常量，不 import deep_ep
# --------------------------------------------------------------------------------------
def extract(path, names, glb):
    src = open(path, 'rb').read()
    tree = ast.parse(src)
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            body.append(node)
    mod = ast.Module(body=body, type_ignores=[])
    code = compile(ast.fix_missing_locations(mod), path, 'exec')
    g = {'torch': torch}
    g.update(glb)
    exec(code, g)
    return g, hashlib.sha256(src).hexdigest()


def get_nvlink_gbs(factor: float = 0.9) -> float:
    """逐字复刻 deep_ep/utils/envs.py:207-234 的 get_nvlink_gbs。"""
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],
                                capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = r'GPU \d+:.*?(?=^GPU \d+:|^$)'
        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
        assert match
        link_pattern = r'Link \d+:\s*([\d\.]+) GB/s'
        return sum(float(bw) for bw in re.findall(link_pattern, match.group(0))) * factor
    except Exception as e:
        print(f'  (get_nvlink_gbs failed: {e})')
        return 0.0


# --------------------------------------------------------------------------------------
# Part A：真实源码加载
# --------------------------------------------------------------------------------------
def part_a(g_pf, g_rg, sha_pf, sha_rg):
    section('Part A：真实源码加载（sha256）+ 抽取清单')
    print(f'  test_prefetch_weights.py sha256={sha_pf[:16]}…')
    print(f'  test_reduce_grads.py     sha256={sha_rg[:16]}…')
    for label, g, ns in [('prefetch', g_pf, ['make_blob', 'reshape_weights', 'make_redundancy_mapping', 'check']),
                         ('reduce', g_rg, ['make_pattern', 'expert_grad_key', 'redundant_expert_grad_key',
                                           'make_redundancy_mapping', 'expected_expert_grads'])]:
        got = [n for n in ns if callable(g.get(n))]
        record(f'{label} 顶层函数抽取 = {got}', len(got) == len(ns))


# --------------------------------------------------------------------------------------
# Part B：make_blob —— ramp + 唯一 key（test_prefetch_weights.py:43-53）
# --------------------------------------------------------------------------------------
def make_blob_ref(global_expert_idx, num_bytes, payload_idx, num_payloads=3, period=255):
    key = global_expert_idx * num_payloads + payload_idx
    idx = torch.arange(num_bytes, dtype=torch.int64, device='cuda') + key
    return (idx % period).to(torch.uint8)


def part_b(make_blob):
    section('Part B：make_blob —— 每 (专家,载荷) 一个唯一 key 的确定性 ramp（:43-53）')
    n = 4096
    a = make_blob(5, n, 1)
    b = make_blob(5, n, 1)
    record('确定性：同 (e,p,n) 两次调用逐字节相等', torch.equal(a, b))
    record('dtype=uint8 / shape=[n] / 设备 cuda', a.dtype == torch.uint8 and a.shape == (n,) and a.is_cuda)
    record('ramp 逐字复刻 (arange(n)+key)%255',
           torch.equal(a, make_blob_ref(5, n, 1)))
    # key 公式：key = e*NUM_PAYLOADS + p；错误专家或错误载荷 → 每个字节都不同
    c = make_blob(6, n, 1)
    d = make_blob(5, n, 2)
    record('错误专家 → 每一字节都不同（不是碰巧几分）', int((a != c).sum()) == n)
    record('错误载荷 → 每一字节都不同', int((a != d).sum()) == n)
    # 边界与周期：末尾 255 循环
    record('周期 255：a[i] == (i+key)%255（key=5*3+1=16）',
           bool(torch.equal(a[:255], ((torch.arange(255, device='cuda') + 16) % 255).to(torch.uint8))))
    print(f'  样例：e=5,p=1 前 8 字节 = {a[:8].tolist()}')


# --------------------------------------------------------------------------------------
# Part C：make_redundancy_mapping（prefetch 版，:56-92）
# --------------------------------------------------------------------------------------
def part_c(map_pf):
    section('Part C：make_redundancy_mapping —— own-expert 排除 / empty_ratio / pattern（:56-92）')
    NR, NRE, NE = 8, 8, 64
    m = map_pf(NR, NRE, NE, 0.25, 0, 'random')
    nle = NE // NR
    record('[8,8] int32 映射，落在 LSA 域内', m.shape == (NR, NRE) and m.dtype == torch.int32)
    owners = torch.div(m, nle, rounding_mode='floor')
    row = torch.arange(NR, device=m.device, dtype=torch.int32).unsqueeze(1)
    record('一个 rank 永不为自己持有冗余副本（owners != 行号）',
           bool(((m < 0) | (owners != row)).all()))
    valid = m >= 0
    record('专家 id 落在 [0, num_experts)，-1 表示空槽',
           bool((m[valid] < NE).all() and (m[valid] >= 0).all()))
    ratio = float((~valid).float().mean())
    record(f'empty_ratio≈0.25（实测 {ratio:.4f}）', abs(ratio - 0.25) < 0.08)
    record('确定性：同 seed 两次相等', torch.equal(map_pf(NR, NRE, NE, 0.25, 0, 'random'),
                                                   map_pf(NR, NRE, NE, 0.25, 0, 'random')))
    record('换 seed 得到不同映射', not torch.equal(map_pf(NR, NRE, NE, 0.25, 0, 'random'),
                                                  map_pf(NR, NRE, NE, 0.25, 1, 'random')))
    # ring 版（reduce 测试专有）：rank r 只从 r+1 借
    mr = map_pf(NR, NRE, NE, 0.0, 0, 'ring')
    owners_r = torch.div(mr, nle, rounding_mode='floor')
    expect_owner = ((torch.arange(NR, device='cpu') + 1) % NR).unsqueeze(1).to(owners_r.device)
    record('ring：rank r 的每个冗余槽都来自 rank r+1',
           bool((owners_r == expect_owner).all()))
    record('ring：无空槽（empty_ratio=0）', bool((mr >= 0).all()))


# --------------------------------------------------------------------------------------
# Part D：check —— 哨兵 vs blob 逐字节（:96-108）
# --------------------------------------------------------------------------------------
def part_d(make_blob, check, SENTINEL):
    section('Part D：check —— 未分配槽必须留 SENTINEL，分配槽必须等于 blob（:96-108）')
    NR, NRE, NE = 8, 8, 64
    nle = NE // NR
    rank = 2
    mapping = torch.full((NR, NRE), -1, dtype=torch.int32, device='cuda')
    # 给 rank 2 分配两个槽：一个指向专家 3，一个指向专家 20；其余留空
    mapping[rank, 0] = 3
    mapping[rank, 1] = 20
    nbytes = 1024
    # 构造正确的冗余权重池
    redund = torch.full((NRE, nbytes), SENTINEL, dtype=torch.uint8, device='cuda')
    for j in range(NRE):
        e = int(mapping[rank, j].item())
        if e >= 0:
            redund[j] = make_blob(e, nbytes, 0)
    ok = True
    try:
        check(redund, mapping, rank, nbytes, 0, 'good')
    except AssertionError:
        ok = False
    record('正确内容 → check 通过（含 -1 槽保持哨兵）', ok)
    # 负样本 1：分配槽写成错误专家
    bad = redund.clone()
    bad[0] = make_blob(4, nbytes, 0)
    try:
        check(bad, mapping, rank, nbytes, 0, 'neg-expert'); ok2 = False
    except AssertionError:
        ok2 = True
    record('负样本：分配槽内容错专家 → AssertionError', ok2)
    # 负样本 2：未分配槽被意外写入
    bad = redund.clone()
    bad[5] = make_blob(9, nbytes, 0)
    try:
        check(bad, mapping, rank, nbytes, 0, 'neg-sentinel'); ok3 = False
    except AssertionError:
        ok3 = True
    record('负样本：未分配槽被写入（应为哨兵）→ AssertionError', ok3)
    # 负样本 3：错误 payload（同专家不同载荷）
    bad = torch.full((NRE, nbytes), SENTINEL, dtype=torch.uint8, device='cuda')
    for j in range(NRE):
        e = int(mapping[rank, j].item())
        if e >= 0:
            bad[j] = make_blob(e, nbytes, 1)
    try:
        check(bad, mapping, rank, nbytes, 0, 'neg-payload'); ok4 = False
    except AssertionError:
        ok4 = True
    record('负样本：内容来自错误载荷 → AssertionError', ok4)


# --------------------------------------------------------------------------------------
# Part E：make_pattern —— 精确 fp32 + 原子加顺序无关（test_reduce_grads.py:23-51）
# --------------------------------------------------------------------------------------
def make_pattern_ref(hidden, key, period=4095):
    idx = torch.arange(hidden, dtype=torch.int64, device='cuda') + key
    return ((idx % period) - period // 2).to(torch.float32)


def part_e(make_pattern, expert_grad_key, redundant_expert_grad_key):
    section('Part E：make_pattern —— 小整数精确 fp32，使 reduce(原子加) 顺序无关（:23-51）')
    h = 8192
    p = make_pattern(h, 7)
    record('dtype=fp32 / device cuda', p.dtype == torch.float32 and p.is_cuda)
    record('逐字复刻 (arange(h)+key)%4095 - 2047.5', torch.equal(p, make_pattern_ref(h, 7)))
    record('取值范围 [-2047.5, 2047.5) 且为 0.5 的整数倍',
           float(p.abs().max()) < 4095 / 2 and bool(((p * 2) == (p * 2).round()).all()))
    # key 家族互斥：expert >= 0，redundant < 0
    record('expert_grad_key(e)=e >= 0', all(expert_grad_key(e) >= 0 for e in range(0, 64, 7)))
    record('redundant_expert_grad_key < 0（与 expert key 永不相交）',
           all(redundant_expert_grad_key(r, j, 8) < 0 for r in range(8) for j in range(8)))
    # 顺序无关：k 个 pattern 求和，不同加序 → torch.equal
    parts = [make_pattern(h, expert_grad_key(e)) for e in range(5)]
    s1 = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
    s2 = parts[4] + parts[3] + parts[2] + parts[1] + parts[0]
    s3 = torch.stack(parts).sum(dim=0)
    record('原子加的顺序无关性（5 个 pattern 三种加序逐位相等）',
           torch.equal(s1, s2) and torch.equal(s1, s3))
    # 相邻 12 KiB chunk 内容不同（ramp 周期 4095 vs chunk=3072 fp32）
    record('相邻 12KiB chunk（3072 fp32）内容不同（周期 4095 不被 3072 整除）',
           not torch.equal(p[:3072], p[3072:6144]))


# --------------------------------------------------------------------------------------
# Part F：expected_expert_grads —— 归约期望值（:98-114）
# --------------------------------------------------------------------------------------
def part_f(expected_expert_grads, map_rg, make_pattern, expert_grad_key,
           redundant_expert_grad_key):
    section('Part F：expected_expert_grads —— 归约期望值（seed + 所有匹配冗余梯度，:98-114）')
    NR, NRE, NE = 8, 8, 64
    nle = NE // NR
    h = 4096
    rank = 3
    mapping = map_rg(NR, NRE, NE, 0.25, 0, 'random')
    got = expected_expert_grads(mapping, rank, NR, NRE, nle, h)
    # 独立复刻：seed = expert_grad_key，逐 (remote_rank, slot) 若 mapping 指向本 rank 则加冗余 key
    ref = torch.stack([make_pattern(h, expert_grad_key(rank * nle + le)) for le in range(nle)])
    pulled = 0
    for rr in range(NR):
        if rr == rank:
            continue
        for j in range(NRE):
            e = int(mapping[rr, j].item())
            if e >= 0 and e // nle == rank:
                ref[e % nle] += make_pattern(h, redundant_expert_grad_key(rr, j, NRE))
                pulled += 1
    record('期望值与独立复刻逐位相等', torch.equal(got, ref))
    record(f'本 rank 被拉的冗余梯度数 = {pulled}（>0 才有意义）', pulled > 0)
    # 自 pull 被结构性排除：mapping[rank, *] 永远不指向 rank 自己
    owners = torch.div(mapping, nle, rounding_mode='floor')
    record('绝不从自己那行 pull（mapping 的 own-expert 排除 → 归约不会重复计 seed）',
           bool((mapping[rank] < 0).all() or (owners[rank] != rank).all().item()))


# --------------------------------------------------------------------------------------
# Part G：整轮 push/pull 语义的本地仿真（对 PA YLOAD_SWEEP / CORRECTNESS_HIDDENS 全量）
#     + 带宽账目（per-rank / critical path / % NVLink）
# --------------------------------------------------------------------------------------
def part_g(g_pf, g_rg, nvlink_gbs):
    section('Part G：push/pull 全量仿真（真实 sweep 表）+ 带宽账目')
    make_blob = g_pf['make_blob']
    check = g_pf['check']
    map_pf = g_pf['make_redundancy_mapping']
    PF_SWEEP = g_pf['PAYLOAD_SWEEP']
    NUM_PAYLOADS = g_pf['NUM_PAYLOADS']
    SENTINEL = g_pf['SENTINEL']

    make_pattern = g_rg['make_pattern']
    map_rg = g_rg['make_redundancy_mapping']
    expected_expert_grads = g_rg['expected_expert_grads']
    CORRECTNESS_HIDDENS = g_rg['CORRECTNESS_HIDDENS']

    NR, NE, NRE = 8, 64, 8
    nle = NE // NR

    # --- prefetch：push 语义 ---
    total_slots_checked = 0
    mism = 0
    for payload in PF_SWEEP:
        for round_idx in range(2):
            mapping = map_pf(NR, NRE, NE, 0.25, round_idx, 'random')
            for rank in range(NR):
                for j in range(NRE):
                    e = int(mapping[rank, j].item())
                    if e >= 0:
                        # push：owner=e//nle 把自己的权重 blob 推到 rank 的槽 j
                        got = make_blob(e, payload[0], 0)
                        want = make_blob(e, payload[0], 0)
                        total_slots_checked += 1
                        mism += int(not torch.equal(got, want))
    record(f'push 仿真：{len(PF_SWEEP)} 载荷 × 2 轮 × {NR} rank 全槽内容一致（{total_slots_checked} 槽, {mism} 不符）',
           mism == 0)
    record('每载荷均满足 TMA 对齐（payload[i] % 32 == 0，compiled.cuh kNumTMAAlignmentBytes）',
           all(n % ALIGN_TMA == 0 for payload in PF_SWEEP for n in payload))
    record('blob key 预算：num_experts*NUM_PAYLOADS <= 255（NUM_BLOB_VALUES）',
           NE * NUM_PAYLOADS <= 255)

    # --- reduce：pull 语义 ---
    pull_mism = 0
    pull_checks = 0
    for hidden in CORRECTNESS_HIDDENS:
        for round_idx in range(2):
            mapping = map_rg(NR, NRE, NE, 0.25, round_idx, 'random')
            for rank in range(NR):
                exp = expected_expert_grads(mapping, rank, NR, NRE, nle, hidden)
                # 本地仿真 pull：seed + 所有指向本 rank 的冗余梯度
                sim = torch.stack([make_pattern(hidden, rank * nle + le) for le in range(nle)])
                for rr in range(NR):
                    if rr == rank:
                        continue
                    for j in range(NRE):
                        e = int(mapping[rr, j].item())
                        if e >= 0 and e // nle == rank:
                            sim[e % nle] += make_pattern(hidden, -1 - (rr * NRE + j))
                pull_checks += 1
                pull_mism += int(not torch.equal(sim, exp))
    record(f'pull 仿真：{len(CORRECTNESS_HIDDENS)} hidden × 2 轮 × {NR} rank 期望值逐位一致（{pull_checks} 组, {pull_mism} 不符）',
           pull_mism == 0)
    record('pattern key 预算：num_experts + num_ranks*num_redundant_experts < 4095',
           NE + NR * NRE < 4095)
    record('每 hidden 均满足 TMA 对齐（hidden*4 % 32 == 0）',
           all(h * 4 % ALIGN_TMA == 0 for h in CORRECTNESS_HIDDENS))

    # --- 带宽账目（prefetch：perf payload = (64<<20, 1<<14, 1<<12)）---
    mapping = map_pf(NR, NRE, NE, 0.25, 0, 'random')
    owners = torch.div(mapping, nle, rounding_mode='floor')
    per_sender = [int(((mapping >= 0) & (owners == r)).sum().item()) for r in range(NR)]
    perf_payload = (64 << 20, 1 << 14, 1 << 12)
    bpe = sum(perf_payload)
    crit = max(per_sender) * bpe
    print(f'  prefetch 账目：每 rank 推送的冗余专家数 {per_sender}')
    print(f'    每冗余专家 {bpe/1e6:.2f} MB；critical path {crit/1e6:.1f} MB；'
          f'NVLink 上限 {nvlink_gbs:.1f} GB/s → critical path 下限 {crit/(nvlink_gbs*1e9)*1e6:.1f} us（100% 时）')
    # --- 带宽账目（reduce：perf hidden = 64<<20 bytes / 4）---
    perf_hidden = (64 << 20) // 4
    per_owner = [int(((mapping >= 0) & (owners == r)).sum().item()) for r in range(NR)]
    bpe_r = perf_hidden * 4
    crit_r = max(per_owner) * bpe_r
    print(f'  reduce 账目：每 rank 被拉的冗余专家数 {per_owner}')
    print(f'    每冗余专家 {bpe_r/1e6:.2f} MB；critical path {crit_r/1e6:.1f} MB')
    record('prefetch 与 reduce 的“关键路径”专家集合一致（同一 mapping 两端读取）',
           per_sender == per_owner)


def main():
    names_pf = ['make_blob', 'reshape_weights', 'make_redundancy_mapping', 'check']
    names_rg = ['make_pattern', 'expert_grad_key', 'redundant_expert_grad_key',
                'make_redundancy_mapping', 'expected_expert_grads']

    # 先把两个文件解析出常量（SENTINEL / PAYLOAD_NAMES / NUM_BLOB_VALUES / PAYLOAD_SWEEP 等）
    def consts(path, wanted):
        src = open(path).read()
        tree = ast.parse(src)
        g = {'range': range}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in wanted:
                    g[node.targets[0].id] = eval(compile(ast.Expression(node.value), path, 'eval'), dict(g))
        return g

    pf_consts = consts(PF_TEST, ['SENTINEL', 'PAYLOAD_NAMES', 'NUM_PAYLOADS', 'NUM_BLOB_VALUES',
                                 'PAYLOAD_SWEEP', 'PAYLOAD_WEIGHTS', 'PAYLOAD_SF0', 'PAYLOAD_SF1'])
    rg_consts = consts(RG_TEST, ['NUM_PATTERN_VALUES', 'CORRECTNESS_HIDDENS'])

    g_pf, sha_pf = extract(PF_TEST, names_pf, pf_consts)
    g_rg, sha_rg = extract(RG_TEST, names_rg, rg_consts)

    print('=' * 88)
    print('P90-1  DeepEP EPLB 权重预取 / 梯度归约的验收测试即正确性规范')
    print(f'  torch {torch.__version__}  |  cuda {torch.version.cuda}  |  device {torch.cuda.get_device_name(0)}')
    print('=' * 88)

    part_a(g_pf, g_rg, sha_pf, sha_rg)
    part_b(g_pf['make_blob'])
    part_c(g_pf['make_redundancy_mapping'])
    part_d(g_pf['make_blob'], g_pf['check'], pf_consts['SENTINEL'])
    part_e(g_rg['make_pattern'], g_rg['expert_grad_key'], g_rg['redundant_expert_grad_key'])
    part_f(g_rg['expected_expert_grads'], g_rg['make_redundancy_mapping'],
           g_rg['make_pattern'], g_rg['expert_grad_key'], g_rg['redundant_expert_grad_key'])
    part_g(g_pf, g_rg, get_nvlink_gbs())

    n_pass = sum(1 for _, ok in RESULT if ok)
    print('\n' + '=' * 88)
    print(f'总计：{n_pass}/{len(RESULT)} PASS' + ('  → ALL PASS' if n_pass == len(RESULT) else '  → 有 FAIL'))
    print('=' * 88)
    sys.exit(0 if n_pass == len(RESULT) else 1)


if __name__ == '__main__':
    main()
