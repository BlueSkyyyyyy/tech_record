"""P55-1 实验：DeepEP 的 dispatch/combine【参考实现（语义规范）】与流量/基准生成器。

目标源码（`~/github/DeepEP@93eb6eb`，逐行直接加载真实文件，不走 deep_ep/__init__）：
  * deep_ep/utils/refs.py      —— dispatch / generate_pre_combine_data / ordered_accumulate / combine
  * deep_ep/utils/gate.py      —— generate_topk_idx / generate_rank_count / get_unbalanced_scores 等
  * deep_ep/utils/testing.py   —— flush_l2_cache / bench / bench_kineto / parse_num_bytes

为什么 H100 可跑：
  * 这三个模块是**纯 PyTorch / numpy**。refs.py 只用 `dist.get_rank/get_world_size` +
    `all_to_all_single`（torch 自带 NCCL），**不碰 DeepEP 的 C++/NCCL 设备 API**；
  * 但 `import deep_ep` 会在 `deep_ep/__init__.py:55 check_nccl_so()` 里因本机 NCCL 2.29.7
    与编译期 NCCL 不匹配而 abort。故这里用 sys.modules 注入一个合成包 + stub 掉
    `deep_ep.utils.envs`（refs 只用到 `get_global_seed`），再按文件路径 exec 真实源码。
  * 8 卡 H100 真机 `mp.spawn` + NCCL 跑真实 all_to_all，对 refs.dispatch 做**独立 oracle** 对拍；
    combine/gate/testing 是单进程张量运算，rank0 直接跑。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_deepep_refs.py"
"""
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
WORLD = int(os.environ.get('EP_REFS_WORLD', '8'))
DEV = 'cuda'


# --------------------------------------------------------------------------------------
# 真实源码加载（绕过 deep_ep/__init__.py 的 check_nccl_so）
# --------------------------------------------------------------------------------------
def load_real_modules():
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
    testing, h_test = load('deep_ep.utils.testing', 'testing.py')
    return (math_m, refs, gate, testing), (h_refs, h_gate, h_test, h_math)


# --------------------------------------------------------------------------------------
# 独立 oracle（不调用 refs 的任何代码，纯 Python 复刻文档语义）
# --------------------------------------------------------------------------------------
def make_x_rows(gidx: torch.Tensor, hidden: int, dtype):
    """确定性 x：仅由 global token idx 决定，便于对端独立重建（免 all_gather 大张量）。"""
    j = torch.arange(1, hidden + 1, device=gidx.device, dtype=torch.float32).unsqueeze(0)
    return torch.sin(gidx.float().unsqueeze(1) * j * 1e-3).to(dtype)


def dispatch_oracle(all_topk, rank, num_tokens, num_topk, num_experts, num_ranks,
                    num_max_tokens_per_rank, hidden, x_dtype):
    """独立 oracle：按 (src_rank, local_idx) 升序逐 token 收集 rank 本地专家的命中行。"""
    e_local = num_experts // num_ranks
    lo, hi = rank * e_local, (rank + 1) * e_local
    src_arr, topk_arr, x_arr, cnt = [], [], [], []
    for s in range(num_ranks):
        tk = all_topk[s]
        hit = ((tk >= lo) & (tk < hi)).any(dim=1)
        idx = hit.nonzero(as_tuple=False).flatten()
        cnt.append(int(idx.numel()))
        if idx.numel() == 0:
            continue
        gidx = (idx + s * num_max_tokens_per_rank)
        src_arr.append(gidx)
        row = tk[idx].clone()
        mask = (row >= lo) & (row < hi)
        row = row - lo
        row = row.masked_fill(~mask, -1)
        topk_arr.append(row)
        x_arr.append(make_x_rows(gidx, hidden, x_dtype))
    src = torch.cat(src_arr) if src_arr else torch.empty(0, dtype=torch.int, device=DEV)
    topk = torch.cat(topk_arr) if topk_arr else torch.empty(0, num_topk, dtype=torch.int64, device=DEV)
    xs = torch.cat(x_arr) if x_arr else torch.empty(0, hidden, dtype=x_dtype, device=DEV)
    return src, topk, xs, cnt


def grouped_reduce_naive(data: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
    """按 gid 分组、组内沿 topk **严格升序**累加，和存到该组最大下标槽、其余置零。"""
    T, k, H = data.shape
    out = torch.zeros_like(data)
    for t in range(T):
        seen = {}
        order = []
        for j in range(k):
            g = int(gid[t, j].item())
            if g not in seen:
                seen[g] = []
                order.append(g)
            seen[g].append(j)
        for g in order:
            slots = seen[g]
            acc = torch.zeros(H, dtype=torch.float32, device=data.device)
            for j in slots:
                acc += data[t, j].float()
            out[t, slots[-1]] = acc.to(out.dtype)
    return out


def ordered_accumulate_naive(data: torch.Tensor, initial=None) -> torch.Tensor:
    result = (torch.zeros(data.shape[0], data.shape[2], dtype=torch.float32, device=data.device)
              if initial is None else initial.clone().float())
    for i in range(data.shape[1]):
        result = result + data[:, i, :].float()
    return result.to(data.dtype)


def combine_oracle(y, topk_idx, num_scaleout, num_scaleup, num_experts, bias,
                   reduce_in_local, reduce_in_scaleup):
    num_ranks = num_scaleout * num_scaleup
    e_local = num_experts // num_ranks
    out = y.clone()
    if reduce_in_local:
        out = grouped_reduce_naive(out, topk_idx // e_local)
    if reduce_in_scaleup:
        out = grouped_reduce_naive(out, topk_idx // (e_local * num_scaleup))
    if isinstance(bias, (tuple, list)):
        bsum = bias[0].float() + bias[1].float()
    elif bias is not None:
        bsum = bias.float()
    else:
        bsum = None
    return ordered_accumulate_naive(out, bsum)


# --------------------------------------------------------------------------------------
# Part A：真实源码加载
# --------------------------------------------------------------------------------------
def part_a(mods, hashes, rank):
    if rank != 0:
        return
    _, refs, gate, testing = mods
    h_refs, h_gate, h_test, h_math = hashes
    print('### Part A：真实源码加载（sha256）')
    print(f'  refs.py    sha256={h_refs[:16]}…  dispatch/combine 参考实现')
    print(f'  gate.py    sha256={h_gate[:16]}…  流量/负载不均衡生成器')
    print(f'  testing.py sha256={h_test[:16]}…  bench/bench_kineto/parse_num_bytes')
    print(f'  math.py    sha256={h_math[:16]}…')
    print(f'  refs 函数: {[n for n in ("dispatch","generate_pre_combine_data","ordered_accumulate","combine") if hasattr(refs, n)]}')
    print()


# --------------------------------------------------------------------------------------
# Part B：dispatch 参考实现 + 独立 oracle（8 卡真实 all_to_all）
# --------------------------------------------------------------------------------------
def part_b(rank, world, refs):
    torch.manual_seed(1000 + rank)
    num_tokens = 512
    hidden = 256
    num_experts = world * 4
    num_topk = 4
    num_max = num_tokens
    x_dtype = torch.bfloat16

    gidx = torch.arange(num_tokens, device=DEV) + rank * num_max
    x = make_x_rows(gidx, hidden, x_dtype)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=DEV)
    topk_weights = torch.rand(num_tokens, num_topk, device=DEV)

    recv_x, recv_topk_idx, recv_topk_weights, recv_src, num_recv = refs.dispatch(
        x, topk_idx, topk_weights, num_max, num_experts)

    all_topk = [torch.empty_like(topk_idx) for _ in range(world)]
    dist.all_gather(all_topk, topk_idx)
    exp_src, exp_topk, exp_x, exp_cnt = dispatch_oracle(
        all_topk, rank, num_tokens, num_topk, num_experts, world, num_max, hidden, x_dtype)

    checks = []
    # 1) 接收行按 src_token_global_idx 升序（docstring 声明）
    checks.append(('recv_src 升序', bool(torch.all(recv_src[1:] >= recv_src[:-1]))))
    # 2) 接收行本身与 oracle 完全一致
    src_eq = (recv_src.numel() == exp_src.numel()
              and bool(torch.equal(recv_src.int(), exp_src.int())))
    checks.append(('recv_src == oracle', src_eq))
    # 3) 本地专家索引掩码（越界→-1）逐位
    checks.append(('recv_topk_idx == oracle', bool(torch.equal(recv_topk_idx.int(), exp_topk.int()))))
    # 4) 激活逐位（bf16 精确）
    checks.append(('recv_x == oracle', bool(torch.equal(recv_x, exp_x))))
    # 5) 每来源 rank 计数
    checks.append(('num_recv_tokens_per_rank == oracle',
                   [int(v) for v in num_recv.tolist()] == exp_cnt))
    # 6) -1 覆盖率 = 每个 token 中不属于本 rank 的专家数
    checks.append(('越界专家被置 -1', int((recv_topk_idx == -1).sum()) ==
                   int(((exp_topk == -1).sum()))))
    # 7) 收到的 topk 权重无 NaN/Inf 且与 x 同 token 数
    checks.append(('recv_topk_weights 形状/有限',
                   recv_topk_weights.shape == (recv_x.shape[0], num_topk)
                   and bool(torch.isfinite(recv_topk_weights).all())))

    if rank == 0:
        total = int(sum(int(v) for v in num_recv.tolist()))
        print(f'### Part B：refs.dispatch 语义（{world}×H100 NCCL all_to_all，num_tokens=512, E={num_experts}, k=4）')
        print(f'  本 rank 收到 {recv_x.shape[0]} 行；全 grid 收到 {total} 行；'
              f'每来源 rank 计数 = {[int(v) for v in num_recv.tolist()]}')
        print(f'  recv_src 前 8 个全局索引 = {recv_src[:8].tolist()}')
        for name, ok in checks:
            print(f'  [{"PASS" if ok else "FAIL"}] {name}')
        print()


# --------------------------------------------------------------------------------------
# Part C：combine 参考实现 + 独立 oracle（单进程张量运算）
# --------------------------------------------------------------------------------------
def part_c(rank, world, refs):
    torch.manual_seed(7)
    num_tokens, hidden, num_topk = 64, 32, 4
    num_scaleout, num_scaleup = 2, 4
    num_experts = num_scaleout * num_scaleup * 4
    src = torch.arange(num_tokens, device=DEV) * 3 + 7
    y = refs.generate_pre_combine_data(src, num_tokens, num_topk, hidden)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=DEV)
    bias = torch.randn(num_tokens, hidden, device=DEV)
    bias_pair = (torch.randn(num_tokens, hidden, device=DEV),
                 torch.randn(num_tokens, hidden, device=DEV))

    configs = [
        ('hybrid    (local=T, scaleup=T)', True, True, None),
        ('non-hybrid(local=T, scaleup=F)', True, False, None),
        ('multi-red-disabled(F,F)       ', False, False, None),
        ('hybrid + bias                 ', True, True, bias),
        ('hybrid + bias_pair            ', True, True, bias_pair),
    ]
    results = {}
    for label, rl, rs, b in configs:
        got = refs.combine(y, topk_idx, num_scaleout, num_scaleup, num_experts, b, rl, rs)
        ref = combine_oracle(y, topk_idx, num_scaleout, num_scaleup, num_experts, b, rl, rs)
        same = bool(torch.equal(got, ref))
        results[label] = got
        if rank == 0:
            print(f'  [{"PASS" if same else "FAIL"}] {label}  '
                  f'输出 {tuple(got.shape)} {got.dtype}  与 oracle 逐位相等={same}')
    # 文档强调：num_scaleout==1 时 (T,T) 与 (T,F) 仍因 bias 不同
    y1 = refs.generate_pre_combine_data(src, num_tokens, num_topk, hidden)
    a = refs.combine(y1, topk_idx, 1, 8, 32, bias_pair, True, True)
    b_ = refs.combine(y1, topk_idx, 1, 8, 32, bias_pair, True, False)
    if rank == 0:
        print(f'  scaleout=1 下 (local,scaleup)=(T,T) vs (T,F) 不同 = '
              f'{not bool(torch.equal(a, b_))}（文档所述 bias 效应）')
        # 非法组合 (F,T) 必须被拒
        try:
            refs.combine(y, topk_idx, num_scaleout, num_scaleup, num_experts, None, False, True)
            print('  [FAIL] (F,T) 非法组合未报错')
        except AssertionError:
            print('  [PASS] (F,T) 非法组合被 assert 拒绝')
        print()


# --------------------------------------------------------------------------------------
# Part D：流量生成器 gate.py
# --------------------------------------------------------------------------------------
def part_d(rank, world, gate, math_m):
    """镜像真实 `tests/utils/test_gate.py` 的判据：practical_ratio = count[0]/min(count[1:])。"""
    if rank != 0:
        return
    print('### Part D：流量/负载不均衡生成器 gate.py（判据镜像 tests/utils/test_gate.py）')
    ok, tot = 0, 0
    num_tokens = 4096
    print('  --- generate_topk_idx：反推 rank_count ---')
    num_experts, num_ranks, num_topk = 32, 8, 4
    e_local = num_experts // num_ranks
    rc = gate.generate_rank_count(num_tokens, num_experts, num_ranks, num_topk, 2.0)
    tk = gate.generate_topk_idx(rc, num_tokens, num_experts, num_ranks, num_topk)
    rec = torch.zeros_like(rc)
    rec.scatter_add_(1, (tk // e_local).long(), torch.ones_like(tk, dtype=rc.dtype))
    tot += 1; good = bool(torch.equal(rec, rc)); ok += good
    print(f'  [{"PASS" if good else "FAIL"}] generate_topk_idx 反推 rank_count 逐元素一致')
    try:
        gate.generate_topk_idx(rc[:, :-1].contiguous(), num_tokens, num_experts, num_ranks, num_topk)
        tot += 1; print('  [FAIL] 非法 rank_count 未报错')
    except AssertionError:
        tot += 1; ok += 1; print('  [PASS] 非法 rank_count（和 != k）被 assert 拒绝')

    print('  --- get_unbalanced_scores 主扫（precise 检查 ratio<10% & 其余 rank 不平等<1.02）---')
    ineq_fail = 0
    for num_experts_per_rank in (4, 8):
        for num_ranks in (8, 16):
            num_experts = num_experts_per_rank * num_ranks
            for num_topk in (1, 2, 4, 6):
                if num_topk > num_experts:
                    continue
                for ratio in (1.0, 2.0, 4.0):
                    for precise in (True, False):
                        lower = max(1, math_m.ceil_div(num_topk, num_experts_per_rank))
                        upper = min(min(num_topk, num_ranks), int((num_ranks - 1) / ratio) + 1)
                        if lower > upper:
                            continue
                        total = torch.zeros(num_ranks, device=DEV)
                        for _ in range(num_ranks):
                            sc = gate.get_unbalanced_scores(
                                num_tokens, num_experts, num_ranks, num_topk, ratio, precise)
                            _, idx = torch.topk(sc, num_topk, dim=-1, largest=True, sorted=False)
                            cnt = torch.nn.functional.one_hot(
                                idx // num_experts_per_rank, num_ranks).any(dim=1).float().sum(dim=0)
                            total += cnt
                            assert torch.equal(
                                gate.generate_rank_count(num_tokens, num_experts, num_ranks,
                                                         num_topk, ratio).sum(dim=1),
                                torch.full((num_tokens,), num_topk, dtype=torch.int32, device=DEV))
                        pr = total[0].item() / max(total[1:].min().item(), 1)
                        ineq = total[1:].max().item() / max(total[1:].min().item(), 1)
                        tot += 1
                        good = (not precise) or (abs(pr - ratio) / ratio < 0.1 and ineq < 1.02)
                        if precise and not good:
                            ineq_fail += 1
                        ok += good
                        if precise:
                            print(f'  [{"PASS" if good else "FAIL"}] precise E/r={num_experts_per_rank} '
                                  f'R={num_ranks:2d} k={num_topk} ratio={ratio}: '
                                  f'practical={pr:.3f} inequality={ineq:.3f}')
    print(f'  （precise 未达标数={ineq_fail}；random 模式按设计不校验）')
    print(f'  → Part D: {ok}/{tot} PASS')
    print()


# --------------------------------------------------------------------------------------
# Part E：基准/工具 testing.py
# --------------------------------------------------------------------------------------
def part_e(rank, world, testing):
    if rank != 0:
        return
    print('### Part E：基准/工具 testing.py')
    ok, tot = 0, 0
    # 1) parse_num_bytes
    cases = [('1G', 1 << 30), ('64M', 1 << 26), ('512k', 1 << 19), ('1048576', 1048576),
             (' 2 GiB ', 2 << 30), ('1tb', 1 << 40)]
    allok = all(testing.parse_num_bytes(t) == v for t, v in cases)
    tot += 1; ok += allok
    print(f'  [{"PASS" if allok else "FAIL"}] parse_num_bytes 6 例: '
          f'{[testing.parse_num_bytes(t) for t, _ in cases]}')
    # 2) 非法输入被拒
    bad_ok = 0
    for bad in ['abc', '0', '-1G', '1.2.3M']:
        try:
            testing.parse_num_bytes(bad)
        except ValueError:
            bad_ok += 1
    tot += 1; ok += (bad_ok == 4)
    print(f'  [{"PASS" if bad_ok == 4 else "FAIL"}] parse_num_bytes 非法输入拒绝 {bad_ok}/4')
    # 3) bench vs bench_kineto（真实 fp8/bf16 matmul）
    torch.manual_seed(0)
    a = torch.randn(2048, 2048, device=DEV)
    bb = torch.randn(2048, 2048, device=DEV)
    fn = lambda: a @ bb
    avg, mn, mx = testing.bench(fn, num_warmups=5, num_tests=20, flush_l2=False)
    avg_f, mn_f, mx_f = testing.bench(fn, num_warmups=5, num_tests=20, flush_l2=True)
    kin = testing.bench_kineto(fn, 'gemm', num_tests=20, suppress_kineto_output=True, flush_l2=True)
    tot += 1
    shape_ok = (avg > 0 and mn <= avg <= mx + 1e-9)
    ok += shape_ok
    print(f'  [{"PASS" if shape_ok else "FAIL"}] bench(no flush) avg/min/max = '
          f'{avg*1e6:.1f}/{mn*1e6:.1f}/{mx*1e6:.1f} us')
    print(f'        bench(flush_l2) avg/min/max = {avg_f*1e6:.1f}/{mn_f*1e6:.1f}/{mx_f*1e6:.1f} us'
          f'  (flush 多 {100*(avg_f/avg-1):.1f}%)')
    print(f'        bench_kineto("gemm") = {kin*1e6:.1f} us')
    # 4) EP_USE_NVIDIA_TOOLS 旁路
    os.environ['EP_USE_NVIDIA_TOOLS'] = '1'
    skip = testing.bench_kineto(fn, ('a', 'b'), num_tests=3)
    os.environ.pop('EP_USE_NVIDIA_TOOLS')
    tot += 1; ok += (skip == (1, 1))
    print(f'  [{"PASS" if skip == (1, 1) else "FAIL"}] EP_USE_NVIDIA_TOOLS=1 → bench_kineto 返回 {skip}')
    # 5) suppress_stdout_stderr：进入时 sys.stdout 被换成 /dev/null 文件、退出后恢复
    before = sys.stdout
    with testing.suppress_stdout_stderr():
        inside_swapped = sys.stdout is not before
    restored = sys.stdout is before
    tot += 1; ok += (inside_swapped and restored)
    print(f'  [{"PASS" if inside_swapped and restored else "FAIL"}] suppress_stdout_stderr '
          f'进入换 stdout={inside_swapped} / 退出恢复={restored}')
    print(f'  → Part E: {ok}/{tot} PASS')
    print()


# --------------------------------------------------------------------------------------
def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=__import__('datetime').timedelta(seconds=180))
    mods, hashes = load_real_modules()
    math_m, refs, gate, testing = mods
    part_a(mods, hashes, rank)
    part_b(rank, world, refs)
    part_c(rank, world, refs)
    part_d(rank, world, gate, math_m)
    part_e(rank, world, testing)
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(20000 + (os.getpid() % 20000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}')
    print('DeepEP dispatch/combine 参考实现（语义规范）+ 流量/基准生成器 —— H100 真机')
    print('=' * 96)
    if WORLD > 1:
        mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    else:
        worker(0, 1)
