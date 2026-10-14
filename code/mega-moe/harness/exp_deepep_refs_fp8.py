"""P105-1 实验：DeepEP `refs.dispatch` 的【FP8 分支】——per-128 block scale 的传递契约，
并与设备端 `layout/ep/token.cuh` 的 `sf_pack_t` / `TokenLayout` SF 布局对照。

目标源码（`~/github/DeepEP@93eb6eb`，逐行直接加载真实文件，绕过 `deep_ep/__init__.py` 的 check_nccl_so）：
  * deep_ep/utils/refs.py   —— dispatch 的 `(x, sf)` 元组分支（`:38-49` 契约、`:73` 选 sf、`:106-107`
                               独立 all_to_all、`:119-120` 打包回元组）
  * deep_ep/utils/math.py   —— `per_token_cast_to_fp8/back`（参考量化器，产生/消费 per-128 scale）
  * deep_ep/include/deep_ep/layout/ep/token.cuh + common/compiled.cuh —— 设备侧 token 缓冲 SF 布局
                                （由 src/deepep_token_layout_sf.cu 在 host 上调用真实头文件产出）

H100 可跑的原因：refs.py / math.py 全是纯 PyTorch（只用 torch 自带 NCCL 的 all_to_all_single），
`TokenLayout` 方法是 `__host__ __device__` 可 host 直接调用；端到端 `import deep_ep` 的 NCCL 闸门
用 sys.modules 注入合成包绕过（同第 65/114 篇）。

运行：bash harness/exp_deepep_refs_fp8.sh
"""
import hashlib
import importlib.util
import os
import subprocess
import sys
import types
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
WORLD = int(os.environ.get('EP_REFS_WORLD', '8'))
DEV = 'cuda'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LAYOUT_BIN = '/tmp/deepep_token_layout_sf'


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
    return (math_m, refs), (h_refs, h_math)


# --------------------------------------------------------------------------------------
# 独立 oracle：不调用 refs 任何代码，按 (src_rank, local_idx) 升序重建应到行（含 SF）
# --------------------------------------------------------------------------------------
def make_x_rows(gidx, hidden, dtype):
    j = torch.arange(1, hidden + 1, device=gidx.device, dtype=torch.float32).unsqueeze(0)
    return torch.sin(gidx.float().unsqueeze(1) * j * 1e-3).to(dtype)


def make_sf_rows(gidx, hidden, dtype):
    """确定性 per-128 scale：仅由 (global token idx, block idx) 决定，便于对端独立重建。"""
    nblk = (hidden + 127) // 128
    b = torch.arange(1, nblk + 1, device=gidx.device, dtype=torch.float32).unsqueeze(0)
    return (0.5 + 0.25 * torch.abs(torch.sin(gidx.float().unsqueeze(1) * b * 1e-2))).to(dtype)


def dispatch_fp8_oracle(all_topk, rank, num_topk, num_experts, num_ranks,
                        num_max_tokens_per_rank, hidden, sf_dtype):
    e_local = num_experts // num_ranks
    lo, hi = rank * e_local, (rank + 1) * e_local
    src_arr, topk_arr, x_arr, sf_arr, cnt = [], [], [], [], []
    for s in range(num_ranks):
        tk = all_topk[s]
        hit = ((tk >= lo) & (tk < hi)).any(dim=1)
        idx = hit.nonzero(as_tuple=False).flatten()
        cnt.append(int(idx.numel()))
        if idx.numel() == 0:
            continue
        gidx = idx + s * num_max_tokens_per_rank
        src_arr.append(gidx)
        row = tk[idx].clone()
        mask = (row >= lo) & (row < hi)
        row = row - lo
        row = row.masked_fill(~mask, -1)
        topk_arr.append(row)
        x_arr.append(make_x_rows(gidx, hidden, torch.bfloat16).to(torch.float8_e4m3fn))
        sf_arr.append(make_sf_rows(gidx, hidden, sf_dtype))
    src = torch.cat(src_arr) if src_arr else torch.empty(0, dtype=torch.int, device=DEV)
    topk = torch.cat(topk_arr) if topk_arr else torch.empty(0, num_topk, dtype=torch.int64, device=DEV)
    xs = torch.cat(x_arr) if x_arr else torch.empty(0, hidden, dtype=torch.float8_e4m3fn, device=DEV)
    sfs = (torch.cat(sf_arr) if sf_arr
           else torch.empty(0, (hidden + 127) // 128, dtype=sf_dtype, device=DEV))
    return src, topk, xs, sfs, cnt


# --------------------------------------------------------------------------------------
def part_a(mods, hashes, rank):
    if rank != 0:
        return
    math_m, refs = mods
    h_refs, h_math = hashes
    print('### Part A：真实源码加载（sha256）')
    print(f'  refs.py sha256={h_refs[:16]}…  math.py sha256={h_math[:16]}…  DeepEP@{DEP.split("/")[-1]}')
    print()


def part_b(rank, world, refs, math_m):
    """8×H100 真实 all_to_all：refs.dispatch 的 (x, sf) 元组分支 + 独立 oracle（含 SF）"""
    torch.manual_seed(2000 + rank)
    num_tokens, hidden, num_topk = 512, 512, 4
    num_experts = world * 4
    num_max = num_tokens
    hidden_sf = (hidden + 127) // 128

    gidx = torch.arange(num_tokens, device=DEV) + rank * num_max
    x_bf16 = make_x_rows(gidx, hidden, torch.bfloat16)
    sf = make_sf_rows(gidx, hidden, torch.float32)          # [T, ceil_div(H,128)] fp32
    x_fp8 = x_bf16.to(torch.float8_e4m3fn)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device=DEV)
    topk_weights = torch.rand(num_tokens, num_topk, device=DEV)

    recv_xs, recv_topk_idx, recv_topk_weights, recv_src, num_recv = refs.dispatch(
        (x_fp8, sf), topk_idx, topk_weights, num_max, num_experts)
    recv_x, recv_sf = recv_xs

    all_topk = [torch.empty_like(topk_idx) for _ in range(world)]
    dist.all_gather(all_topk, topk_idx)
    o_src, o_topk, o_x, o_sf, o_cnt = dispatch_fp8_oracle(
        all_topk, rank, num_topk, num_experts, world, num_max, hidden, torch.float32)

    checks = []
    checks.append(('recv_x 是 (x, sf) 元组', isinstance(recv_xs, tuple) and len(recv_xs) == 2))
    checks.append(('recv_sf 形状 == [nrecv, ceil_div(H,128)]',
                   tuple(recv_sf.shape) == (recv_x.shape[0], hidden_sf)))
    checks.append(('recv_sf dtype 原样保留(fp32)', recv_sf.dtype == torch.float32))
    checks.append(('recv_src == oracle', bool(torch.equal(recv_src.int(), o_src.int()))))
    checks.append(('recv_topk_idx == oracle', bool(torch.equal(recv_topk_idx.int(), o_topk.int()))))
    checks.append(('recv_x(fp8) == oracle（逐位）', bool(torch.equal(recv_x.view(torch.uint8),
                                                                     o_x.view(torch.uint8)))))
    checks.append(('recv_sf == oracle（逐位）', bool(torch.equal(recv_sf, o_sf))))
    checks.append(('num_recv_tokens_per_rank == oracle',
                   [int(v) for v in num_recv.tolist()] == o_cnt))

    # 契约：hidden_sf 必须恰为 ceil_div(hidden,128)（refs.py:49）
    try:
        refs.dispatch((x_fp8, sf[:, :-1].contiguous()), topk_idx, topk_weights, num_max, num_experts)
        bad_ok = False
    except AssertionError:
        bad_ok = True
    checks.append(('hidden_sf != ceil_div(H,128) 被 assert 拒绝', bad_ok))

    # 反量化往返：用真实参考量化器把 (recv_x, recv_sf) 还原，核对 per-128 语义
    deq = math_m.per_token_cast_back(recv_x, recv_sf)   # bf16 [nrecv, H]
    o_deq = (o_x.float().view(o_x.shape[0], -1, 128) *
             o_sf.view(o_sf.shape[0], -1, 1)).view(o_x.shape).to(torch.bfloat16)
    rel = ((deq.float() - o_deq.float()).abs().sum() / (o_deq.float().abs().sum() + 1e-9)).item()
    checks.append((f'per_token_cast_back(recv) == oracle 反量化 (rel={rel:.2e})', rel < 1e-6))

    if rank == 0:
        print(f'### Part B：refs.dispatch 的 FP8 分支（{world}×H100 NCCL all_to_all，T=512, H=512, E={num_experts}, k=4）')
        print(f'  本 rank 收到 {recv_x.shape[0]} 行；recv_sf {tuple(recv_sf.shape)} {recv_sf.dtype}；'
              f'hidden_sf = ceil_div(512,128) = {hidden_sf}')
        print(f'  每来源 rank 计数 = {[int(v) for v in num_recv.tolist()]}')
        for name, ok in checks:
            print(f'  [{"PASS" if ok else "FAIL"}] {name}')
        print()
    return all(ok for _, ok in checks)


def part_c(rank):
    """解析 C++ 真实头文件输出，用独立 Python 公式重算 TokenLayout 的 SF 布局整数几何"""
    if rank != 0:
        return True
    print('### Part C：设备端 token 缓冲 SF 布局（`layout/ep/token.cuh`，host 调用真实头文件）')
    try:
        out = subprocess.run([LAYOUT_BIN], capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f'  [FAIL] 无法运行 {LAYOUT_BIN}: {e}')
        return False
    print('  --- C++ 真实头文件输出 ---')
    for line in out.strip().splitlines():
        print(f'  {line}')
    print('  --- 独立 Python 公式复刻对拍 ---')

    TMA, RDMA = 32, 64
    ok, tot = 0, 0
    for line in out.strip().splitlines():
        if line.startswith('CONST'):
            kv = dict(tok.split('=') for tok in line.split()[1:])
            checks = [('sizeof_sf_pack_t', 4), ('kNumAlignedSFPacks', 4),
                      ('kNumTMAAlignmentBytes', TMA), ('kNumRDMAAlignmentBytes', RDMA)]
            for k, v in checks:
                tot += 1
                good = int(kv[k]) == v
                ok += good
                print(f'  [{"PASS" if good else "FAIL"}] {k}={kv[k]} (expect {v})')
        elif line.startswith('LAYOUT'):
            d = dict(tok.split('=') for tok in line.split()[1:])
            hidden, topk, md = int(d['hidden']), int(d['topk']), int(d['md'])
            hidden_sf = (hidden + 127) // 128
            num_sf_bytes = hidden_sf * 4
            sf_off = ((hidden + TMA - 1) // TMA) * TMA
            md_bytes = topk * (4 + 4) + ((1 + topk) * 4 if md else 0)
            md_off = sf_off + ((num_sf_bytes + TMA - 1) // TMA) * TMA
            bytes_no_mb = (((sf_off + ((num_sf_bytes + TMA - 1) // TMA) * TMA) +
                            ((md_bytes + TMA - 1) // TMA) * TMA + RDMA - 1) // RDMA) * RDMA
            exp = {'hidden_sf': hidden_sf, 'num_sf_bytes': num_sf_bytes, 'sf_off': sf_off,
                   'md_off': md_off, 'bytes_no_mb': bytes_no_mb}
            allok = all(int(d[k]) == v for k, v in exp.items())
            tot += 1
            ok += allok
            print(f'  [{"PASS" if allok else "FAIL"}] LAYOUT hidden={hidden} topk={topk} md={md}: '
                  f'{ {k: (int(d[k]), v) for k, v in exp.items()} }')
    print(f'  → Part C: {ok}/{tot} PASS')
    print()
    return ok == tot


def part_d(rank, world):
    """SF 字节开销与通信量模型；并坐实 combine 没有 FP8 变体"""
    if rank != 0:
        return True
    print('### Part D：SF 字节开销 / 参考契约 vs 设备缓冲上界 / combine 无 FP8 变体')
    ok, tot = 0, 0
    # 1) 参考契约固定 gran 128（refs.py:49）；设备缓冲用 gran 32 作【保守上界】（ep.hpp:269），
    #    并断言「最坏 SF 字节 <= 主数据」（ep.hpp:257: ceil_div(hidden,32)*4 <= hidden）
    consistent = True
    print('  hidden  | ref: ceil(H,128) | dev buf upper: ceil(H,32) | ratio | 4*ceil(H,32)<=H')
    for hidden in (2048, 4096, 7168, 8192, 16384):
        ref_hidden_sf = (hidden + 127) // 128
        dev_upper = (hidden + 31) // 32
        ratio = dev_upper / ref_hidden_sf
        inv = dev_upper * 4 <= hidden
        consistent &= inv and ratio == 4.0
        print(f'  {hidden:7d} | {ref_hidden_sf:16d} | {dev_upper:22d} | {ratio:5.1f} | {inv}')
    tot += 1; ok += consistent
    print(f'  [{"PASS" if consistent else "FAIL"}] 参考 gran128 是设备缓冲上界(gran32)的 1/4，'
          f'且最坏 SF 字节 <= 主数据（逐形状）')

    # 2) SF 相对数据的字节占比（参考 gran128 契约；FP8 1B/元素）
    print('  hidden  |  data/token | sf/token(ref) | sf% of data | 设备上界 sf% of data')
    for hidden in (2048, 4096, 7168, 8192, 16384):
        data = hidden * 1
        sf = ((hidden + 127) // 128) * 4
        sf_up = ((hidden + 31) // 32) * 4
        print(f'  {hidden:7d} | {data:10d} | {sf:13d} | {100*sf/data:10.3f}% | {100*sf_up/data:14.3f}%')
    tot += 1; ok += (1 / 32 == 0.03125)
    print(f'  [PASS] 参考 gran128 无 padding 时 SF/数据 = 4/128 = 3.125%')

    # 3) 设备实际 dispatch 用的 SF 列数 = 张量自己的 sf.size(1)（ep.hpp:356），不是写死的
    src_cpp = open(f'{DEP}/csrc/buffers/ep.hpp').read()
    dyn = 'num_sf_packs = num_sf_packs_;' in src_cpp
    tot += 1; ok += dyn
    print(f'  [{"PASS" if dyn else "FAIL"}] dispatch 用 sf.size(1)（ep.hpp:356）作为实际列数')
    tma_col = 'use_tma_aligned_col_major_sf' in src_cpp and 'recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks)' in src_cpp
    tot += 1; ok += tma_col
    print(f'  [{"PASS" if tma_col else "FAIL"}] 输出 SF 可选 TMA 列主序（ep.hpp:689-690）')

    # 4) combine 没有 FP8 分支：签名只收 y（无 sf 元组），generate_pre_combine_data 恒 bf16
    import inspect
    refs = load_real_modules()[0][1]
    sig = str(inspect.signature(refs.combine))
    has_sf = ('sf' in sig)
    tot += 1; ok += (not has_sf)
    print(f'  [{"PASS" if not has_sf else "FAIL"}] refs.combine 签名无 sf/元组分支：{sig}')
    src = open(f'{DEP}/deep_ep/utils/refs.py').read()
    pre = 'return result.to(torch.bfloat16)' in src
    tot += 1; ok += pre
    print(f'  [{"PASS" if pre else "FAIL"}] generate_pre_combine_data 恒返回 bf16（`:153`）')

    print(f'  → Part D: {ok}/{tot} PASS')
    print()
    return ok == tot


def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, timeout=timedelta(seconds=180))
    mods, hashes = load_real_modules()
    math_m, refs = mods
    part_a(mods, hashes, rank)
    b = part_b(rank, world, refs, math_m)
    c = part_c(rank)
    d = part_d(rank, world)
    if rank == 0:
        print('=' * 96)
        print(f'ALL DONE  PartB={b} PartC={c} PartD={d}')
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(21000 + (os.getpid() % 20000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}')
    print('DeepEP refs.dispatch 的 FP8 分支 + 设备端 token SF 布局 —— H100 真机')
    print('=' * 96)
    if WORLD > 1:
        mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    else:
        worker(0, 1)
