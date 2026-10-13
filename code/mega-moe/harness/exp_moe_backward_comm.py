"""P94-1 实验：MoE **反向**的通信量模型与 H100 8 卡实测。

目的：为 docs/104「MoE 反向的通信量模型」提供真机证据。前向侧的通信模型在 docs/01 §2 已经量过
（裸 all_to_all 的时延/带宽），反向侧却一直没有模型——本实验把它拆成**四个方向**并在 8×H100 上用
真实 **ragged all-to-all**（torch.distributed + NCCL，含非均匀 split）量出：

  1. dispatch        前向：本地 token 按 top-k 专家所在 rank 散射出去（FP8，1B）
  2. combine         前向：专家输出送回来源 rank 并按 token 求和（BF16，2B）
  3. combine_bwd     反向：grad_combine 按原路送回专家 rank（= dispatch 的伴随，BF16，2B）
  4. dispatch_bwd    反向：各专家 grad 送回来源 rank 并求和（= combine 的伴随，BF16，2B）

这与 DeepEP README 的两个训练 helper 一一对应：
  - `dispatch_backward`（README `:204-221`）内部调 `_buffer.combine`，注释
    「The backward pass of MoE dispatch is actually a combine.」
  - `combine_backward`（README `:242-262`）内部调 `_buffer.dispatch`（`do_expand=True`），注释
    「The backward pass of MoE combine is actually a dispatch.」
两者都带 `async_with_compute_stream=True`——**反向不做 mega-kernel 融合，但同样做通信-计算重叠**。

路由用与 DeepEP 测试一致的 `get_unbalanced_scores`/`topk` 语义的非 expand 布局：一个 token 只向
「拥有其任一 top-k 专家」的 rank 发**一行**（`refs.py:67` 的 `mask_to_send`）。

尺寸取 DeepSeek-V3 量级：H=7168，EP=8，E=64（每 rank 8 专家），topk=8。
**这不是 SM100 Mega MoE 的 kernel**（SM100 专属，本机不可跑），而是它底下那两个通信原语的**真实通信量**。

第三部分把「一次通信」与「一次等量计算」叠到两条 CUDA stream 上，量出重叠效率。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_moe_backward_comm.py"
输出重定向到 exp_moe_backward_comm.out.txt。
"""
import datetime
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ.setdefault('NCCL_NVLS_ENABLE', '0')  # 本机 NVLS multicast 绑定失败，禁用

WORLD = 8
H = 7168           # DeepSeek-V3 hidden
E = 64             # 总专家数
TOPK = 8
TOKENS = [1024, 4096, 16384]
ITERS = 30
WARMUP = 10


def _bench(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters  # ms


def build_routing(T, world, device, seed):
    """构造两种路由计数（对应 DeepEP 的 dedup 传输 + expanded 布局）：

    - **dedup**（传输层）：一个 token 只向「拥有其任一 top-k 专家」的 rank 发**一行**
      （`refs.py:67` 的 `mask_to_send`；`dispatch.cuh:103-105` 的 `ptx::deduplicate`）→ 总行数 T·d。
    - **expand**（`do_expand=True`）：每个 (token, expert) 一行 → 总行数 T·k；combine / 两个反向相位
      都在 expanded 空间搬运。

    返回 dict: dedup_counts[world], dedup_idx[Nd], expand_counts[world], expand_idx[Ne], dst[T,k], d。
    """
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(T, E, generator=g, device=device)
    topk = scores.topk(TOPK, dim=-1).indices           # [T, topk]
    epl = E // world
    dst = topk // epl                                  # [T, topk] 每个槽的专家所在 rank

    # --- dedup：token 级，是否向某 rank 发过 ---
    occurrence = torch.zeros(T, world, dtype=torch.bool, device=device)
    occurrence.scatter_(1, dst, True)
    dedup_counts = occurrence.sum(0).to(torch.int64)
    order = occurrence.t().nonzero(as_tuple=False)     # [Nd,2]=(dst,token)，天然按 dst 分组
    dedup_idx = order[:, 1].contiguous()

    # --- expand：槽级（T*k 行），按 dst 稳定分组 ---
    flat_dst = dst.reshape(-1)                         # [T*k]
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, TOPK).reshape(-1)
    sort_order = torch.argsort(flat_dst, stable=True)  # 按 dst 分组，组内保序
    expand_idx = token_ids[sort_order].contiguous()
    expand_counts = torch.bincount(flat_dst, minlength=world).to(torch.int64)

    occf = occurrence.sum(1).float().mean().item()     # 平均不同 dst rank 数 d
    return dedup_counts, dedup_idx, expand_counts, expand_idx, dst, occf


def exchange_counts(send_counts, world, device):
    recv = torch.empty(world, dtype=torch.int64, device=device)
    dist.all_to_all_single(recv, send_counts)
    return recv


def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '26712')
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=300))
    if rank == 0:
        print(f'torch {torch.__version__}  world={world}  H={H}  E={E}  topk={TOPK}')
        print('=' * 96)

    for T in TOKENS:
        dedup_counts, dedup_idx, expand_counts, expand_idx, dst, d = \
            build_routing(T, world, dev, seed=1234 + T)
        dedup_recv = exchange_counts(dedup_counts, world, dev)
        expand_recv = exchange_counts(expand_counts, world, dev)
        dl_send, dl_recv = dedup_counts.tolist(), dedup_recv.tolist()
        el_send, el_recv = expand_counts.tolist(), expand_recv.tolist()
        Nd, Ne = int(dedup_counts.sum()), int(expand_counts.sum())
        Nd_recv, Ne_recv = int(dedup_recv.sum()), int(expand_recv.sum())

        # 预置输入（本地复制，不在计时内；数据搬运的输入就长这样）
        x_fp8_in = torch.randn(Nd, H, device=dev).to(torch.float8_e4m3fn)      # dispatch 发送侧
        y_in = torch.randn(Ne_recv, H, device=dev, dtype=torch.bfloat16)       # combine 接收侧
        g_gather_in = torch.randn(Ne, H, device=dev, dtype=torch.bfloat16)     # combine_bwd 发送侧
        g_in = torch.randn(Ne_recv, H, device=dev, dtype=torch.bfloat16)       # dispatch_bwd 接收侧
        buf_send = torch.empty(Ne, H, device=dev, dtype=torch.bfloat16)        # 发送侧回程缓冲
        buf_recv = torch.empty(Ne_recv, H, device=dev, dtype=torch.bfloat16)

        # 1) dispatch（dedup 传输，FP8 1B）：总行数 T·d
        def _dispatch():
            out = torch.empty(Nd_recv, H, device=dev, dtype=torch.float8_e4m3fn)
            dist.all_to_all_single(out, x_fp8_in,
                                   output_split_sizes=dl_recv, input_split_sizes=dl_send)
            return out
        t_disp = _bench(_dispatch)

        # 2) combine（expanded，BF16 2B）：a2a 回来源 + 本地按 token 求和
        def _combine_a2a():
            dist.all_to_all_single(buf_send, y_in,
                                   output_split_sizes=el_send, input_split_sizes=el_recv)
            return buf_send

        def _combine_reduce():
            acc = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
            acc.index_add_(0, expand_idx, buf_send)
            return acc

        def _combine():
            _combine_a2a()
            return _combine_reduce()

        t_comb_a2a = _bench(_combine_a2a)
        t_comb_red = _bench(_combine_reduce)
        t_comb = _bench(_combine)

        # 3) combine_bwd = dispatch 的伴随（expanded，BF16 2B）
        def _combine_bwd():
            dist.all_to_all_single(buf_recv, g_gather_in,
                                   output_split_sizes=el_recv, input_split_sizes=el_send)
            return buf_recv
        t_comb_bwd = _bench(_combine_bwd)

        # 4) dispatch_bwd = combine 的伴随（expanded，BF16 2B）：a2a + 本地按 token 求和
        def _dispatch_bwd_a2a():
            dist.all_to_all_single(buf_send, g_in,
                                   output_split_sizes=el_send, input_split_sizes=el_recv)
            return buf_send

        def _dispatch_bwd():
            _dispatch_bwd_a2a()
            return _combine_reduce()

        t_disp_bwd_a2a = _bench(_dispatch_bwd_a2a)
        t_disp_bwd = _bench(_dispatch_bwd)

        b_disp = Nd * H * 1 / 1e6           # dispatch（dedup, fp8）
        b_exp = Ne * H * 2 / 1e6            # expanded（bf16, 每相位）
        vol_fwd = b_disp + b_exp
        vol_bwd = b_exp + b_exp

        def _gbs(mb, ms):
            return mb / 1e3 / (ms / 1e3)

        if rank == 0:
            print(f'T={T:6d}  E={E} topk={TOPK} | d={d:.3f}  dedup 行数 Nd={Nd:7d} ({Nd/T:.2f}×T)  '
                  f'expand 行数 Ne={Ne:7d} ({Ne/T:.2f}×T)')
            print(f'    [1] dispatch      FP8  {b_disp:8.1f} MB  {t_disp*1e3:9.1f} us  '
                  f'{_gbs(b_disp, t_disp):7.1f} GB/s   (dedup T·d)')
            print(f'    [2] combine       BF16 {b_exp:8.1f} MB  {t_comb*1e3:9.1f} us  '
                  f'{_gbs(b_exp, t_comb):7.1f} GB/s  (a2a {t_comb_a2a*1e3:.1f} us='
                  f'{_gbs(b_exp,t_comb_a2a):.1f}GB/s + reduce {t_comb_red*1e3:.1f} us, expand T·k)')
            print(f'    [3] combine_bwd   BF16 {b_exp:8.1f} MB  {t_comb_bwd*1e3:9.1f} us  '
                  f'{_gbs(b_exp, t_comb_bwd):7.1f} GB/s   (= dispatch 伴随, expand)')
            print(f'    [4] dispatch_bwd  BF16 {b_exp:8.1f} MB  {t_disp_bwd*1e3:9.1f} us  '
                  f'{_gbs(b_exp, t_disp_bwd):7.1f} GB/s  (a2a {t_disp_bwd_a2a*1e3:.1f} us + '
                  f'reduce {t_comb_red*1e3:.1f} us = combine 伴随)')
            print(f'    前向 a2a {(t_disp+t_comb_a2a)*1e3:9.1f} us | 反向 a2a '
                  f'{(t_comb_bwd+t_disp_bwd_a2a)*1e3:9.1f} us | 时间比='
                  f'{(t_comb_bwd+t_disp_bwd_a2a)/(t_disp+t_comb_a2a):.2f}× | 字节量比 '
                  f'B_bwd/B_fwd={vol_bwd/vol_fwd:.2f}×')
            print('-' * 96)

    # ---- 第三部分：通信-计算重叠（combine BF16 与一次等量 GEMM 叠两流）----
    T = 4096
    dedup_counts, dedup_idx, expand_counts, expand_idx, dst, _ = \
        build_routing(T, world, dev, seed=999)
    expand_recv = exchange_counts(expand_counts, world, dev)
    el_send, el_recv = expand_counts.tolist(), expand_recv.tolist()
    rows = int(expand_recv.sum())
    y_in = torch.randn(rows, H, device=dev, dtype=torch.bfloat16)
    out_send = torch.empty(int(expand_counts.sum()), H, device=dev, dtype=torch.bfloat16)
    # 等量计算：dgrad 形状 dA = dY @ W2^T, (m=rows, N=2048, K=H)
    A = torch.randn(rows, H, device=dev, dtype=torch.bfloat16)
    B = torch.randn(2048, H, device=dev, dtype=torch.bfloat16)

    def _comm_only():
        dist.all_to_all_single(out_send, y_in,
                               output_split_sizes=el_send, input_split_sizes=el_recv)
        return out_send

    t_comm_only = _bench(_comm_only)
    t_comp_only = _bench(lambda: torch.matmul(A, B.t()))

    side = torch.cuda.Stream()
    def _both():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _comm_only()
        torch.matmul(A, B.t())
        torch.cuda.current_stream().wait_stream(side)
    t_both = _bench(_both)

    if rank == 0:
        overlap = (t_comm_only + t_comp_only - t_both) / min(t_comm_only, t_comp_only)
        print('重叠实验（T=4096，combine-BF16 与等量 dgrad-GEMM 叠两流）:')
        print(f'    通信单独 {t_comm_only*1e3:9.1f} us | 计算单独 {t_comp_only*1e3:9.1f} us | '
              f'两流并发 {t_both*1e3:9.1f} us')
        print(f'    串行基线 {(t_comm_only+t_comp_only)*1e3:9.1f} us -> 重叠省 '
              f'{(t_comm_only+t_comp_only-t_both)*1e3:8.1f} us，重叠效率 '
              f'{overlap*100:5.1f}% (相对 min)')

    dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
