"""P156-1 实验：路由偏斜对 EP 通信与显存的真实代价（第 142 篇「GEMM 不为偏斜所动」的通信侧对偶）。

缺口（第 152 篇/`docs/152` §8 自己点名）：
  第 152 篇量的是**单算子** grouped GEMM 在专家负载不均下的计算侧代价，结论是「GEMM 本来就
  不为偏斜所动，偏斜的真实代价在显存与通信」——但它明确写「未把 dispatch/combine 的通信不均
  纳入（见第 104 篇）」。而第 104 篇量的是**均衡**路由下的前向/反向通信量。**偏斜路由下 EP 通信
  到底贵多少**，从没被单独量过。本篇补上。

用真实 DeepEP 源码（`~/github/DeepEP@93eb6eb`，逐行加载，绕开 `deep_ep/__init__.py` 的 NCCL 阻塞）：
  * `deep_ep/utils/gate.py:32-113`  `generate_rank_count`——制造「特殊 rank 0 拿 ratio 倍流量」的
    可控 rank 级不均衡（DeepEP 每份 EP 负载测试都用它，见第 116 篇）。
  * `deep_ep/utils/refs.py:12-120`  `dispatch`——dispatch 的参考实现（语义规范），用来交叉验证
    我们算出的 per-rank 行数 = 真实 `num_recv_tokens_per_rank`。
  * `deep_ep/utils/math.py`  `ceil_div`（refs 依赖）。
  * `deep_ep/include/deep_ep/layout/ep/token.cuh:10-82`  `TokenLayout`——per-token 缓冲布局（含
    mbarrier）；`compiled.cuh:59-70` 的 `sf_pack_t=4B`、TMA 对齐 32B、RDMA 对齐 64B。
    字节公式在 Python 里复刻并与第 162 篇（P152-1）的 C++ 真机数交叉核对。

可跑边界：纯 PyTorch + NCCL ragged all-to-all（架构无关），H100 真机可跑；Mega MoE 融合 kernel 本体
SM100 专属、仅分析。

运行：bash scripts/lab.sh exec "EPSC_WORLD=8 python /abs/path/harness/exp_routing_skew_comm.py"
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
os.environ.setdefault('NCCL_NVLS_ENABLE', '0')   # 本机无 Fabric Manager，禁用 NVLS（否则 all_gather 报 NVLS bind 失败）
WORLD = int(os.environ.get('EPSC_WORLD', '8'))
H = 7168
E = 64
TOPK = 2
T_LOCAL = 2048
RATIOS = [1.0, 1.5, 2.0, 4.0, 8.0]


# --------------------------------------------------------------------------------------
# 真实 DeepEP 源码加载（绕过 deep_ep/__init__.py 的 check_nccl_so）
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
    return (math_m, refs, gate), (h_refs, h_gate, h_math)


# --------------------------------------------------------------------------------------
# 字节公式：复刻 TokenLayout::get_num_bytes（token.cuh:32-39），与第 162 篇 C++ 数交叉核对
# --------------------------------------------------------------------------------------
TMA = 32   # kNumTMAAlignmentBytes (compiled.cuh:68)
RDMA = 64  # kNumRDMAAlignmentBytes (compiled.cuh:69)


def align(x, a):
    return (x + a - 1) // a * a


def token_bytes(num_hidden_bytes, num_sf_bytes, num_topk, with_metadata, with_mbarrier=False):
    meta = num_topk * (4 + 4) + ((1 + num_topk) * 4 if with_metadata else 0)
    data = align(align(num_hidden_bytes, TMA) + align(num_sf_bytes, TMA) + align(meta, TMA), RDMA)
    return data + (align(8, TMA) if with_mbarrier else 0)     # sizeof(mbarrier)=8


def dispatch_bytes_fp8(with_mbarrier=False):
    # TokenLayout(hidden*1B, ceil_div(hidden,32)*4, topk, with_metadata=True)  (dispatch.hpp:26-29)
    return token_bytes(H * 1, align(H, 32) // 32 * 4, TOPK, True, with_mbarrier)


def combine_bytes_bf16(with_mbarrier=True):
    # TokenLayout(hidden*2B, 0, topk, with_metadata=False)  (combine.hpp:19-22)
    # 第 162 篇 C++ 真机数 14432 = 14400（数据）+ 32（每 token mbarrier）
    return token_bytes(H * 2, 0, TOPK, False, with_mbarrier)


# --------------------------------------------------------------------------------------
# Part C：真实 ragged all-to-all 计时
# --------------------------------------------------------------------------------------
def _a2a(t, rc, sc):
    rc, sc = rc.tolist(), sc.tolist()
    out = t.new_empty((int(sum(rc)),) + tuple(t.shape[1:]))
    dist.all_to_all_single(out, t.contiguous(), output_split_sizes=rc, input_split_sizes=sc)
    return out


def _bench(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3            # us


def bench_dispatch_a2a(send_counts, recv_counts):
    """用 fp8 [rows, H] 载荷代表 dispatch 的主数据段，做一次真实 ragged a2a。"""
    nsend = int(send_counts.sum().item())
    t = torch.ones((nsend, H), dtype=torch.float8_e4m3fn, device='cuda')
    return _bench(lambda: _a2a(t, recv_counts, send_counts)), nsend


# --------------------------------------------------------------------------------------
def run(rank, world):
    (math_m, refs, gate), _ = load_real_modules()
    torch.cuda.set_device(rank)
    port = os.environ.get('EPSC_PORT', '29521')
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=world)
    per_rank = E // world
    dev = 'cuda'

    def gen_rank_count(ratio, seed):
        torch.manual_seed(seed)
        return gate.generate_rank_count(T_LOCAL, E, world, TOPK, ratio)   # [T_local, world]

    print(f'\n{"="*100}\nrank {rank}/{world}: T_local={T_LOCAL} H={H} E={E} topk={TOPK}\n{"="*100}'
          if rank == 0 else '', end='')

    # ---- Part A：路由偏斜 → per-rank 行数（token 触及 / slot 命中） ----
    rows = {}
    for ratio in RATIOS:
        rc = gen_rank_count(ratio, seed=1234)                              # [T_local, world]
        # dispatch 在传输层按 (token, dst_rank) 去重（refs.py:64-67 的 mask_to_send）：
        # 每个 token 对每个「至少命中一个本地专家」的 dst rank 只发一行
        token_touch = (rc > 0).to(torch.int32)                             # [T_local, world]
        slot_hit = rc.to(torch.int32)                                      # [T_local, world]
        rows[ratio] = (token_touch.sum(0), slot_hit.sum(0))                # 各 [world]

    # 全局汇聚：每个 src rank 把它的行数贡献到各 dst rank，得到真正的 per-rank recv
    for ratio in RATIOS:
        tt, sh = rows[ratio]
        g_tt = [torch.zeros_like(tt) for _ in range(world)]
        g_sh = [torch.zeros_like(sh) for _ in range(world)]
        dist.all_gather(g_tt, tt)
        dist.all_gather(g_sh, sh)
        rows[ratio] = (torch.stack(g_tt).sum(0), torch.stack(g_sh).sum(0))

    if rank == 0:
        print('\n[Part A] 路由偏斜 → 每 rank 收到的 dispatch 行数（token 去重口径）')
        print(f'  {"ratio":>6} | {"per-rank 行数（去重）":>44} | max/mean  min/mean  gini')
        for ratio in RATIOS:
            tt, sh = rows[ratio]
            v = tt.float()
            mx, mn, mean = v.max().item(), v.min().item(), v.mean().item()
            gini = _gini(v)
            print(f'  {ratio:>6.1f} | {str(tt.tolist()):>44} | {mx/mean:6.2f}  {mn/mean:6.2f}  {gini:.3f}')
        print('  （ratio=1.0 为均衡基线；>1 时 rank 0 被系统性偏好，即 DeepEP `generate_rank_count` 的特殊 rank）')

    # ---- Part B：字节模型（每 rank dispatch/combine 字节 + straggler） ----
    b_disp = dispatch_bytes_fp8()
    b_disp_mb = dispatch_bytes_fp8(with_mbarrier=True)
    b_comb = combine_bytes_bf16()
    byte_rows = {}
    for ratio in RATIOS:
        tt, sh = rows[ratio]
        disp_recv = tt.float() * b_disp                                     # dispatch 收端字节（去重）
        # combine：每个 rank 收到自己 token 的 topk 份 = T_local*topk 行（常值）；
        # 但**发送端**热度不同——rank 0 持的专家行数多。用 slot_hit 近似 combine send 行数。
        comb_send = sh.float() * b_comb
        byte_rows[ratio] = (disp_recv, comb_send)

    if rank == 0:
        print(f'\n[Part B] 字节模型（per-token dispatch FP8={b_disp}B / 含 mbarrier={b_disp_mb}B、'
              f'combine BF16={b_comb}B）')
        print(f'  {"ratio":>6} | {"dispatch max/mean (straggler)":>32} | {"combine send max/mean":>24} | 总字节 MB')
        for ratio in RATIOS:
            dr, cs = byte_rows[ratio]
            print(f'  {ratio:>6.1f} | {dr.max().item()/dr.mean().item():>32.3f} | '
                  f'{cs.max().item()/cs.mean().item():>24.3f} | '
                  f'{(dr.sum().item()+cs.sum().item())/1e6:8.1f}')

    # ---- Part C：真实 8×H100 ragged a2a（dispatch 主数据段）----
    def measure(T, ratio, iters):
        torch.manual_seed(1234)
        rc = gate.generate_rank_count(T, E, world, TOPK, ratio)
        send_counts = (rc > 0).sum(0).to(torch.int32).contiguous()          # 到各 dst 的行数
        recv_counts = torch.empty(world, dtype=torch.int32, device=dev)
        dist.all_to_all_single(recv_counts, send_counts)
        nsend = int(send_counts.sum().item())
        t = torch.ones((nsend, H), dtype=torch.float8_e4m3fn, device=dev)
        # 共享节点取 3 次最小（抑制邻居干扰）
        t_us = min(_bench(lambda: _a2a(t, recv_counts, send_counts), iters=iters) for _ in range(3))
        local_tot = recv_counts.sum()                       # 本 rank 收到的总行数
        mx, tot_all = local_tot.clone(), local_tot.clone()
        dist.all_reduce(mx, op=dist.ReduceOp.MAX)           # straggler = 最大单 rank recv
        dist.all_reduce(tot_all, op=dist.ReduceOp.SUM)      # 全局总行数
        return (t_us, nsend, int(mx.item()), float(tot_all.item()) / world, int(local_tot.item()))

    for tag, Tc, iters in [('C1 小消息（延迟域，T_local=2048）', T_LOCAL, 20),
                           ('C2 大消息（字节域，T_local=16384）', int(os.environ.get('EPSC_TC', '16384')), 8)]:
        timings = {}
        for ratio in RATIOS:
            timings[ratio] = measure(Tc, ratio, iters)
        if rank == 0:
            print(f'\n[Part C] 真实 ragged all-to-all（dispatch 主数据段 fp8 [rows,H]）—— {tag}')
            print(f'  {"ratio":>6} | {"t_a2a us":>10} | {"t/t_base":>9} | {"straggler/mean":>15} | '
                  f'{"本rank MB":>10} | {"有效 GB/s":>10}')
            base = timings[1.0][0]
            for ratio in RATIOS:
                t_us, nsend, mx, mean, tot = timings[ratio]
                mb = nsend * H / 1e6
                gbs = nsend * H / (t_us * 1e-6) / 1e9
                print(f'  {ratio:>6.1f} | {t_us:>10.1f} | {t_us/base:>9.3f} | {mx/mean:>15.3f} | '
                      f'{mb:>10.1f} | {gbs:>10.1f}')
            print('  （straggler/mean = 最大单 rank 收 vs 全局均值；t 是单次 a2a 墙钟）')

    # ---- Part D：真实 refs.dispatch 交叉验证 per-rank recv 行数（真实 NCCL a2a，须全员调用） ----
    T_small = int(os.environ.get('EPSC_T', '512'))
    if rank == 0:
        print('\n[Part D] 真实 deep_ep.utils.refs.dispatch 交叉验证 per-rank recv 行数')
    for ratio in [1.0, 4.0]:
        torch.manual_seed(2024 + rank)          # 各 rank 独立路由
        scores = gate.get_precise_unbalanced_scores(T_small, E, world, TOPK, ratio) \
            if ratio != 1.0 else torch.rand(T_small, E, device=dev)
        _, topk_idx = torch.topk(scores, TOPK, dim=-1, largest=True, sorted=False)
        topk_idx = topk_idx.to(torch.int32)
        # 独立算：本 rank 的发送 count[dst] = Σ_token 1[token 命中 dst]（去重口径，refs.py:64-67）
        hit = torch.zeros(world, dtype=torch.int64, device=dev)
        for d in range(world):
            hit[d] = ((topk_idx // per_rank) == d).any(dim=1).sum()
        # 全局 recv[dst] = Σ_src send_cnt_src[dst]
        gathered = [torch.zeros(world, dtype=torch.int64, device=dev) for _ in range(world)]
        dist.all_gather(gathered, hit)
        model_recv = torch.stack(gathered).sum(0)

        x = torch.randn(T_small, H, dtype=torch.bfloat16, device=dev)
        w = torch.rand(T_small, TOPK, dtype=torch.float32, device=dev)
        out = refs.dispatch(x, topk_idx, w, T_LOCAL, E)         # 5 条真实 a2a
        got = out[-1].to(torch.int64)
        if rank == 0:
            # refs 返回的 [num_ranks] 是 rank0 从各 src rank 收到的行数 = Σ 的去重命中；
            # 与 gathered[src][0] 逐元素应完全相等
            per_src = torch.stack(gathered)[:, 0]
            ok = bool(torch.equal(got.cpu(), per_src.cpu()))
            print(f'  ratio={ratio:>3.1f}  refs num_recv_tokens_per_rank(rank0) = {got.tolist()} '
                  f'sum={int(got.sum())}')
            print(f'           独立模型 per-src 命中 = {per_src.tolist()}  '
                  f'match={ok}  （全局 per-dst={model_recv.tolist()}）  T_small={T_small}')
            assert ok, 'refs.dispatch 与独立去重模型不一致'

    # ---- Part E：缓冲区/显存账 + EPLB ----
    if rank == 0:
        # DeepEP dispatch 缓冲按 num_max_tokens_per_rank 最坏情况分配（与偏斜无关）
        print('\n[Part E] 显存/缓冲区（与偏斜无关）：')
        print(f'  dispatch 缓冲按 worst-case `num_max_tokens_per_rank*num_ranks` 分配 → '
              f'偏斜不改变分配、只改变实际 NVLink 流量分布')
        print(f'  单 token dispatch 记录 = {b_disp} B（含 mbarrier {b_disp_mb} B），'
              f'combine 记录 = {b_comb} B')
        print(f'  EPLB 冗余专家（第 53 篇）把「被系统性偏好」的专家复制到 peer 冗余槽，'
              f'把 straggler 拉平——这是偏斜通信代价的正解')

    dist.barrier()
    dist.destroy_process_group()


def _gini(v):
    v = v.float().sort().values
    n = v.numel()
    s = v.sum().item()
    if s == 0:
        return 0.0
    idx = torch.arange(1, n + 1, device=v.device, dtype=torch.float32)
    return float((2 * (idx * v).sum().item() - (n + 1) * s) / (n * s))


def main():
    (math_m, refs, gate), hashes = load_real_modules()
    print(f'真实 DeepEP 源码 sha256: refs.py {hashes[0][:12]}…  gate.py {hashes[1][:12]}…  '
          f'math.py {hashes[2][:12]}…')
    print(f'字节公式自检: dispatch FP8={dispatch_bytes_fp8()} B（含 mbarrier '
          f'{dispatch_bytes_fp8(with_mbarrier=True)} B）、combine BF16={combine_bytes_bf16()} B '
          f'（对齐第 162 篇 14432）')
    mp.spawn(run, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == '__main__':
    main()
