"""P152-1 实验：多卡 EP MoE 的真实通信字节口径（FP8/1B dispatch + BF16/2B combine）
以及 ragged a2a 的 dtype 伸缩，补第 161 篇（P151-1）的 fp32 保守上界。

缺口（第 161 篇自己点名 `docs/161-ep-fp8-e2e-moe.md:154,167-168,189`）：
  第 161 篇把 EP MoE 图接上真实 SM90 FP8 grouped kernel，但 dispatch/combine 的 a2a 是**按 fp32
  搬运**——它明确写「本实验按 fp32 搬运，是保守上界；真机 dispatch 为 FP8/1B、combine 为 BF16/2B」。
  本篇把这个 fp32 上界**换成 DeepEP 参考实现的真实字段口径**，逐字段量出字节与时间。

权威来源（DeepEP @ 93eb6eb，行号核对过）：
  * `deep_ep/utils/refs.py:38-40`  unpack SF；`hidden_sf == ceil_div(hidden,128)`
  * `deep_ep/utils/refs.py:73-89`  dispatch 的字段：x / sf / topk_idx / topk_weights / src_token_idx
  * `deep_ep/utils/refs.py:100-110` 5 条 a2a（recv_x/recv_sf/recv_topk_idx/recv_topk_weights/recv_src_token_idx）
  * `deep_ep/utils/math.py:31-39`  per_token_cast_to_fp8 → sf 是 fp32 `[m, ceil_div(n,128)]`
  * `deep_ep/layout/ep/token.cuh:10-82`  真实 per-token 缓冲布局（含 mbarrier）；sizeof 见
    `deep_ep/include/deep_ep/common/compiled.cuh:59-70`（sf_pack_t=4B、TMA=32B、RDMA=64B）
  * `README.md:279`  combine 输入必 BF16、展开后的 topk_weights 一维
  真实 header 字节账在 `src/deepep_token_layout_bytes.cu`（直接 include token.cuh 编译运行）给出。

可跑边界：纯 PyTorch + NCCL a2a（架构无关），H100 真机可跑；Mega MoE 融合 kernel 本体 SM100 专属、仅分析。

运行：bash scripts/lab.sh exec "EPCB_WORLD=8 python /abs/path/harness/exp_ep_comm_bytes.py"
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = int(os.environ.get('EPCB_WORLD', '8'))
H = 7168
E = 64
TOPK = 2
T_LOCAL = 2048


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


def build_routing(rank, world, T, per_rank, topk):
    g = torch.Generator(device='cpu').manual_seed(1234 + rank)
    lg = torch.randn(T, E, generator=g, dtype=torch.float32)
    tw, idx = torch.topk(torch.softmax(lg, dim=-1), topk, dim=-1)
    idx = idx.cuda()
    dest = (idx // per_rank).flatten()
    send_counts = torch.bincount(dest, minlength=world).to(torch.int32).cuda()
    recv_counts = torch.empty(world, dtype=torch.int32, device='cuda')
    dist.all_to_all_single(recv_counts, send_counts)
    return int(send_counts.sum().item()), int(recv_counts.sum().item()), send_counts, recv_counts


def field(name, shape, dtype, send_counts, recv_counts, tag):
    """构造 send/recv 张量并测它单独的 a2a 时间。返回 (bytes/rank, us)。"""
    nsend = int(send_counts.sum().item())
    t = torch.ones((nsend, *shape), dtype=dtype, device='cuda')
    t_us = _bench(lambda: _a2a(t, recv_counts, send_counts))
    b = nsend * t[0].numel() * t.element_size() if t.numel() else nsend
    tag.append((name, b, t_us))
    return b, t_us


def run(rank, world):
    per_rank = E // world
    nsend, nrecv, send_counts, recv_counts = build_routing(rank, world, T_LOCAL, per_rank, TOPK)
    hs = H // 128

    if rank == 0:
        print('=' * 96)
        print(f'多卡 EP MoE 真实通信字节口径（world={world}, T_local={T_LOCAL}, H={H}, E={E}, topk={TOPK}）')
        print(f'每 rank 发出 {nsend} 行（=T_local*topk={T_LOCAL*TOPK}），收到 {nrecv} 行（ragged）')
        print('=' * 96)
    # 汇总所有 rank 的 nsend/nrecv，用 max/mean 报（各 rank ragged 不同）
    cnts = torch.tensor([nsend, nrecv], device='cuda')
    dist.all_reduce(cnts, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f'（全局 max：send={int(cnts[0])} recv={int(cnts[1])} 行/rank）')

    # ---------------- Part A：dispatch 逐字段（DeepEP refs.py 真实字段）----------------
    tag = []
    if rank == 0:
        print('\n-- Part A dispatch 逐字段（DeepEP refs.py:73-89 的真实字段）--')
    field('x   (fp8)', [H], torch.float8_e4m3fn, send_counts, recv_counts, tag)
    field('sf  (fp32 [H/128])', [hs], torch.float32, send_counts, recv_counts, tag)
    field('topk_idx (int32)', [TOPK], torch.int32, send_counts, recv_counts, tag)
    field('topk_w   (fp32)', [TOPK], torch.float32, send_counts, recv_counts, tag)
    field('src_idx  (int32)', [], torch.int32, send_counts, recv_counts, tag)

    # 融合成一条扁平 uint8 a2a（真实 kernel 的连续搬运口径）
    total_b = sum(t[1] for t in tag)
    bpt = total_b // nsend
    flat = torch.ones(total_b, dtype=torch.uint8, device='cuda')
    sc_b = (send_counts * bpt).to(torch.int32)
    rc_b = (recv_counts * bpt).to(torch.int32)
    t_flat = _bench(lambda: _a2a(flat, rc_b, sc_b))
    if rank == 0:
        print(f'  {"字段":22s} {"字节/rank":>12s} {"us":>9s}   {"GB/s":>7s}')
        for name, b, us in tag:
            print(f'  {name:22s} {b:12d} {us:9.1f}   {b/us/1e3:7.1f}')
        print(f'  {"合计（可分字段）":22s} {total_b:12d} {sum(t[2] for t in tag):9.1f}')
        print(f'  {"扁平一条 a2a":22s} {total_b:12d} {t_flat:9.1f}   {total_b/t_flat/1e3:7.1f}')

    # dispatch 真实字节 = x(fp8) + sf + meta
    disp_real = sum(t[1] for t in tag)
    disp_fp32 = nsend * H * 4
    disp_bf16 = nsend * H * 2

    # ---------------- Part B：combine 真实字段（y 必 BF16, README:279）----------------
    tagc = []
    if rank == 0:
        print('\n-- Part B combine 真实字段（y 必 BF16，README.md:279；send/recv 交换）--')
    # combine 把收到的 token 送回源 rank：send=recv_counts 形状、recv=send_counts
    n_combine_send = int(recv_counts.sum().item())
    y = torch.ones((n_combine_send, H), dtype=torch.bfloat16, device='cuda')
    t_y = _bench(lambda: _a2a(y, send_counts, recv_counts))
    comb_real = n_combine_send * H * 2
    comb_fp32 = n_combine_send * H * 4
    if rank == 0:
        print(f'  y (bf16 [H])          {comb_real:12d} {t_y:9.1f}   {comb_real/t_y/1e3:7.1f}')

    # ---------------- Part C：dtype 伸缩（同一 ragged 形状，扫 fp32/bf16/fp8）----------------
    if rank == 0:
        print('\n-- Part C dtype 伸缩：同一 ragged [nsend,H] a2a，fp32 vs bf16 vs fp8 --')
    scale = []
    for dt, name in [(torch.float32, 'fp32'), (torch.bfloat16, 'bf16'), (torch.float8_e4m3fn, 'fp8')]:
        t = torch.ones((nsend, H), dtype=dt, device='cuda')
        us = _bench(lambda: _a2a(t, recv_counts, send_counts))
        b = nsend * H * t.element_size()
        scale.append((name, b, us))
    if rank == 0:
        base = scale[0][2]
        for name, b, us in scale:
            print(f'  {name:5s} {b:10d} B  {us:9.1f} us  {b/us/1e3:7.1f} GB/s  (t/t_fp32={us/base:.3f})')

    # ---------------- 校正的 EP 预算 ----------------
    if rank == 0:
        print('\n-- Part D 校正的 EP 单步通信预算（vs 第 161 篇 fp32 上界）--')
        print(f'  dispatch  FP8 真实 {disp_real/1e6:8.2f} MB/rank   '
              f'（第161篇 fp32 {disp_fp32/1e6:8.2f} MB = {disp_fp32/disp_real:.2f}x）')
        print(f'  combine   BF16 真实 {comb_real/1e6:8.2f} MB/rank   '
              f'（第161篇 fp32 {comb_fp32/1e6:8.2f} MB = {comb_fp32/comb_real:.2f}x）')
        print(f'  单步通信合计 真实 {(disp_real+comb_real)/1e6:.2f} MB vs 第161篇 fp32 '
              f'{(disp_fp32+comb_fp32)/1e6:.2f} MB（{((disp_fp32+comb_fp32))/(disp_real+comb_real):.2f}x）')
        print(f'  其中 SF 占比 dispatch = {tag[1][1]/disp_real*100:.1f}%'
              f'（DeepEP 缓冲按 ceil_div(H,32)*4B=896B 预留，见 token.cuh）')

    # ---------------- Part E：平衡锚 ----------------
    if rank == 0:
        print('\n-- Part E 平衡负载 a2a 锚（对照本机争用）--')
    for T in (512, 2048):
        inp = torch.ones(world, T, H, dtype=torch.bfloat16, device='cuda')
        out = torch.empty_like(inp)
        us = _bench(lambda: dist.all_to_all_single(out, inp), iters=20)
        b = world * T * H * 2
        if rank == 0:
            print(f'    {T:6d} tok/rank  {b/1e6:8.1f} MB/rank  {us:9.1f} us  {b/us/1e3:7.1f} GB/s')

    if rank == 0:
        print('\ndone')


def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=__import__('datetime').timedelta(seconds=300))
    if rank == 0:
        print(f'torch {torch.__version__}  world={world}  nccl={torch.cuda.nccl.version()}')
    run(rank, world)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(23100 + (os.getpid() % 8000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
