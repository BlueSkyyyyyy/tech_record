"""P131-1 实验：8×H100 上的【多卡 EP 端到端 MoE 前向/反向】与单卡稠密参考的梯度核验。

为什么需要这篇（补的缺口）：
  * 第 99 篇（P89-1）在 H100 上复刻了 Mega MoE **前向**的数值语义，但**把 dispatch/combine 的通信
    折算成单卡 gather/scatter**——零通信；
  * 第 111 篇（P101-1）做了**反向**的数值复刻，同样是单卡、零通信；
  * 第 95 篇（P85-1）用真实 `refs.dispatch` 跨 8×H100 验了 dispatch/combine 的**伴随恒等式**，
    但那只是线性代数内积，**没有把通信接进一条完整的 MoE 前向/反向图**。

  于是「EP 真的把 token 散到别的 rank、算完再收回来，前向输出与**参数梯度**是否与单卡稠密 MoE 严格
  一致」这件事一直没被端到端量过。本篇补上：用 `torch.distributed`(NCCL) 的 `all_to_all_single`
  实现 expand 版 dispatch/combine（与 DeepEP `do_expand=True` 的行空间一致），把
  `dispatch → L1 → SwiGLU → L2 → combine` 与它的 autograd 反向，逐张量与**单卡稠密参考**对拍，
  给出 dX / dWg / dW1 / dW2 的误差，并量出真实通信量与耗时。

源码坐标（`~/github/DeepEP@93eb6eb`，v2.5.0，行号已核对）：
  * `deep_ep/utils/refs.py:10-123`   dispatch 参考：`:67` mask=(expert_start<=topk<expert_end).any(dim=1)、
                                     `:76-77` 越界 slot 置 -1、`:90` src 全局索引 = rank*num_max+local、
                                     `:94/105` 两次 `all_to_all_single`（换长度 + 换数据）
  * `deep_ep/utils/refs.py:156-174`  ordered_accumulate：沿 top-k 维按序求和（combine 的退化形式）
  * `deep_ep/utils/refs.py:177-243`  combine 参考：`grouped_reduce` 分组求和、`:242` 再加 bias
  * `deep_ep/buffers/ep.py`          真实 host 入口 dispatch/combine（设备路径依赖 NCCL≥2.31，见「阻塞」）
  * `README.md:204-220/242-258`      dispatch_backward=combine / combine_backward=dispatch（伴随）
  * 本专题：`docs/95`（伴随恒等式）、`docs/99`（前向数值复刻）、`docs/111`（反向数值复刻）

可跑性：全篇是**纯 PyTorch 张量 + NCCL all_to_all_single**，架构无关（不碰 UMMA/TMA/tcgen05），
H100 真机可跑。Mega MoE 融合 kernel 本体是 SM100 专属（`sm100_fp8_fp4_mega_moe.cuh`）→ 仅分析。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_ep_e2e_moe.py"
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = int(os.environ.get('EP_E2E_WORLD', '8'))
DEV = 'cuda'


# --------------------------------------------------------------------------------------
# 元素算子
# --------------------------------------------------------------------------------------
def silu(x):
    return x * torch.sigmoid(x)


def emlp(x, w1, w2, inter):
    """x [n,H]；w1 [2I,H]；w2 [H,I]；返回 [n,H]。"""
    gate = x @ w1[:inter].t()
    up = x @ w1[inter:].t()
    h = silu(gate) * up
    return h @ w2.t()


# --------------------------------------------------------------------------------------
# 单卡稠密参考（前向 + autograd 反向）
# --------------------------------------------------------------------------------------
def dense_forward(x, wg, w1, w2, inter, topk):
    """x [T,H] 全局；wg [H,E]；w1 [E,2I,H]；w2 [E,H,I]。返回 out [T,H]、路由量。"""
    T = x.size(0)
    logits = x @ wg                       # [T,E]
    scores = torch.softmax(logits.float(), dim=-1)
    w, idx = torch.topk(scores, topk, dim=-1)   # [T,k]
    w = w / w.sum(dim=-1, keepdim=True)         # 归一化（与 EP 路径一致）
    out = torch.zeros_like(x)
    for k in range(topk):
        e = idx[:, k]                              # [T]
        for ex in e.unique().tolist():
            m = e == ex
            y = emlp(x[m], w1[ex], w2[ex], inter)
            out[m] = out[m] + w[m, k, None].to(x.dtype) * y
    return out, w, idx


# --------------------------------------------------------------------------------------
# EP 分布式：expand dispatch → grouped 专家 → combine
# --------------------------------------------------------------------------------------
def _a2a(t, recv_counts, send_counts):
    return _A2AFn.apply(t, recv_counts, send_counts)


class _A2AFn(torch.autograd.Function):
    """`all_to_all_single` 的显式 autograd 包装：**反向就是交换 split 的同一原语**。

    这正是 DeepEP 的核心事实——dispatch 的反向是 combine、combine 的反向是 dispatch
    （`README.md:204-220/242-258`、本专题 `docs/95`）：all-to-all 的雅可比是其转置，转置
    只需把 `input_split_sizes` 与 `output_split_sizes` 对调。
    """

    @staticmethod
    def forward(ctx, t, recv_counts, send_counts):
        rc, sc = recv_counts.tolist(), send_counts.tolist()
        out = t.new_empty((int(sum(rc)),) + tuple(t.shape[1:]))
        dist.all_to_all_single(out, t.contiguous(),
                               output_split_sizes=rc,
                               input_split_sizes=sc)
        ctx.rc, ctx.sc = rc, sc
        return out

    @staticmethod
    def backward(ctx, g):
        out = g.new_empty((int(sum(ctx.sc)),) + tuple(g.shape[1:]))
        dist.all_to_all_single(out, g.contiguous(),
                               output_split_sizes=ctx.sc,
                               input_split_sizes=ctx.rc)
        return out, None, None


def ep_forward(x, wg, w1_local, w2_local, inter, topk, rank, world):
    T, H = x.shape
    E = w1_local.size(0) * world
    per_rank = E // world

    logits = x @ wg
    scores = torch.softmax(logits.float(), dim=-1)
    w, idx = torch.topk(scores, topk, dim=-1)         # [T,k] global expert ids
    w = w / w.sum(dim=-1, keepdim=True)

    # ---- expand dispatch：每个 (token, k-slot) 一行，按目的 rank 分组 ----
    dest = idx // per_rank                              # [T,k] ∈ [0,world)
    gsrc = (rank * T) + torch.arange(T, device=x.device)[:, None].expand(T, topk)  # [T,k] 全局 token 号

    send_x, send_e, send_w, send_src = [], [], [], []
    send_counts = torch.zeros(world, dtype=torch.long, device=x.device)
    for d in range(world):
        m = dest == d
        cnt = int(m.sum().item())
        send_counts[d] = cnt
        if cnt == 0:
            continue
        send_x.append(x[:, None, :].expand(T, topk, H)[m])
        send_e.append(idx[m] - d * per_rank)           # 本地专家号
        send_w.append(w[m])
        send_src.append(gsrc[m])

    sx = torch.cat(send_x, 0) if send_x else x.new_empty((0, H))
    se = torch.cat(send_e, 0) if send_e else torch.empty(0, dtype=torch.long, device=x.device)
    sw = torch.cat(send_w, 0) if send_w else torch.empty(0, dtype=w.dtype, device=x.device)
    ss = torch.cat(send_src, 0) if send_src else torch.empty(0, dtype=torch.long, device=x.device)

    # 交换长度（refs.py:93-94）
    recv_counts = torch.empty(world, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)

    rx = _a2a(sx, recv_counts, send_counts)
    re = _a2a(se, recv_counts, send_counts)
    rw = _a2a(sw, recv_counts, send_counts)
    rs = _a2a(ss, recv_counts, send_counts)

    # ---- 按本地专家分组（m-grouped contiguous 的构建）----
    order = torch.argsort(re, stable=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=x.device)
    xe, we = rx[order], rw[order]
    counts = torch.bincount(re, minlength=per_rank)
    offs = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), counts.cumsum(0)])

    ye_list = []
    for e in range(per_rank):
        a, b = int(offs[e]), int(offs[e + 1])
        if b > a:
            ye_list.append(emlp(xe[a:b], w1_local[e], w2_local[e], inter) * we[a:b, None])
    ye = torch.cat(ye_list, 0) if ye_list else rx[:0]
    y = ye[inv]                                        # 回收到顺序（按源 rank 分组）

    # ---- combine：按原路送回 + 按 token 求和 ----
    ry = _a2a(y, send_counts, recv_counts)             # 注意 split 交换
    rs_back = _a2a(rs, send_counts, recv_counts)
    out_global = torch.zeros(world * T, H, dtype=torch.float32, device=x.device)
    out_global = out_global.index_add(0, rs_back, ry.float())
    return out_global[rank * T:(rank + 1) * T].to(x.dtype), w, idx


# --------------------------------------------------------------------------------------
# 进程主体
# --------------------------------------------------------------------------------------
def _init(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=__import__('datetime').timedelta(seconds=300))
    return torch.device('cuda', rank)


def run_case(rank, world, dev, tag, T, H, I, E, topk, dtype, tf32=True):
    per_rank = E // world
    old_tf32 = torch.backends.cuda.matmul.allow_tf32   # NGC 容器默认 True（TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1）
    torch.backends.cuda.matmul.allow_tf32 = tf32
    g = torch.Generator(device='cpu').manual_seed(1234)
    wg = torch.randn(H, E, generator=g, dtype=torch.float32).to(dev).to(dtype)
    w1 = (torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32) * (H ** -0.5)).to(dev).to(dtype)
    w2 = (torch.randn(E, H, I, generator=g, dtype=torch.float32) * (I ** -0.5)).to(dev).to(dtype)
    x = torch.randn(T, H, generator=g, dtype=torch.float32).to(dev).to(dtype)
    x.requires_grad_(True)

    wg = wg.clone().requires_grad_(True)
    w1_local = w1[rank * per_rank:(rank + 1) * per_rank].clone().requires_grad_(True)
    w2_local = w2[rank * per_rank:(rank + 1) * per_rank].clone().requires_grad_(True)

    # 目标（确定性，所有 rank 一致）
    tg = torch.Generator(device='cpu').manual_seed(99)
    target_global = torch.randn(world * T, H, generator=tg, dtype=torch.float32).to(dev).to(dtype)
    target = target_global[rank * T:(rank + 1) * T]

    # ---- EP 前向 + 反向 ----
    out_ep, _, _ = ep_forward(x, wg, w1_local, w2_local, I, topk, rank, world)
    loss_ep = ((out_ep.float() - target.float()) ** 2).mean()
    loss_ep.backward()
    dx_ep = x.grad.detach()
    dwg_ep = wg.grad.detach().clone()
    dw1_ep = w1_local.grad.detach().clone()
    dw2_ep = w2_local.grad.detach().clone()
    out_ep_d = out_ep.detach().float()

    # ---- 单卡稠密参考（同样只对本 rank 的 token 求 loss，梯度沿 rank 求和）----
    xd = x.detach().clone().requires_grad_(True)
    wgd = wg.detach().clone().requires_grad_(True)
    w1d = w1.detach().clone().requires_grad_(True)
    w2d = w2.detach().clone().requires_grad_(True)
    out_d, _, _ = dense_forward(xd, wgd, w1d, w2d, I, topk)   # 只算本地 T 个 token
    loss_d = ((out_d.float() - target.float()) ** 2).mean()
    loss_d.backward()

    # ---- 归一化全体梯度（EP 的 W1/W2 是分片、Wg 每 rank 只有本地 token 贡献）----
    dwg_ep_all = dwg_ep.clone(); dist.all_reduce(dwg_ep_all, op=dist.ReduceOp.SUM)
    dw1_ep_g = torch.empty_like(w1d).view(world, per_rank * 2 * I * H)
    dw1_ep_g[rank] = dw1_ep.reshape(-1)
    dist.all_gather(list(dw1_ep_g), dw1_ep.reshape(-1))
    dw1_ep_all = dw1_ep_g.view_as(w1d)
    dw2_ep_g = torch.empty_like(w2d).view(world, per_rank * H * I)
    dw2_ep_g[rank] = dw2_ep.reshape(-1)
    dist.all_gather(list(dw2_ep_g), dw2_ep.reshape(-1))
    dw2_ep_all = dw2_ep_g.view_as(w2d)
    # 稠密 W1/W2/Wg 梯度也按 rank 求和（每个 rank 只对本地 token 求，总量 = 全量）
    dist.all_reduce(w1d.grad); dist.all_reduce(w2d.grad); dist.all_reduce(wgd.grad)
    dx_d_local = xd.grad.detach()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    if rank == 0:
        mm = 'fp32' if not torch.backends.cuda.matmul.allow_tf32 else 'tf32'
        print(f'--- [{tag}] world={world} T={T} H={H} I={I} E={E} topk={topk} dtype={dtype} matmul={mm} ---')
        print(f'  forward  : out_ep vs dense  rel={rel(out_ep_d, out_d.float()):.3e}   '
              f'max|Δ|={ (out_ep_d-out_d.float()).abs().max().item():.3e}')
        print(f'  backward : dX   rel={rel(dx_ep, dx_d_local):.3e}   '
              f'dWg rel={rel(dwg_ep_all, wgd.grad):.3e}')
        print(f'             dW1  rel={rel(dw1_ep_all, w1d.grad):.3e}   '
              f'dW2 rel={rel(dw2_ep_all, w2d.grad):.3e}')
        print(f'  loss     : ep={loss_ep.item():.6e}  dense={loss_d.item():.6e}')
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return rel(out_ep_d, out_d.float())


def measure_comm(rank, world, dev):
    """全量 dispatch 的真实通信量/耗时（DeepSeek 量级 shape，扫不同 batch）。

    平衡负载下每个 (token,k-slot) 均匀落到各 rank，等价于把 `[world, T_local, H]` 沿 rank 维
    做一次 all-to-all（与第 1 篇 `exp_ep_comm.py` 同口径，便于交叉对照）。
    """
    H = 7168
    if rank == 0:
        print('  --- dispatch(expand, bf16, 平衡负载) 有效带宽 vs batch（单向；combine 同量）---')
    for T in (512, 2048, 8192, 32768):
        inp = torch.ones(world, T, H, dtype=torch.bfloat16, device=dev)
        out = torch.empty_like(inp)

        def disp():
            dist.all_to_all_single(out, inp)

        for _ in range(10):
            disp()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(50):
            disp()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / 50
        if rank == 0:
            bytes_per = world * T * H * 2
            print(f'    本地 {T:6d} tok -> 发出 {bytes_per/1e6:8.1f} MB/rank  {ms*1e3:8.1f} us  '
                  f'{bytes_per/1e9/(ms/1e3):7.1f} GB/s')


def worker(rank, world):
    dev = _init(rank, world)
    run_case(rank, world, dev, 'fp64', T=64, H=128, I=256, E=64, topk=2, dtype=torch.float64)
    run_case(rank, world, dev, 'small', T=64, H=128, I=256, E=64, topk=2, dtype=torch.float32)
    run_case(rank, world, dev, 'small-strict', T=64, H=128, I=256, E=64, topk=2, dtype=torch.float32, tf32=False)
    run_case(rank, world, dev, 'mid', T=128, H=512, I=1024, E=64, topk=4, dtype=torch.float32)
    run_case(rank, world, dev, 'bf16', T=128, H=512, I=1024, E=64, topk=4, dtype=torch.bfloat16)
    measure_comm(rank, world, dev)
    dist.barrier()
    if rank == 0:
        print('done')
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(21000 + (os.getpid() % 10000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}  port={os.environ["MASTER_PORT"]}')
    print('EP 端到端 MoE（dispatch→L1→SwiGLU→L2→combine）前向/反向 vs 单卡稠密参考')
    print('-' * 88)
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
