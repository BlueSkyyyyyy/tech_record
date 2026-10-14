"""P134-1 实验：Mega MoE **前向 step 的通信-计算重叠**在 8×H100 上的量化。

补的空档（路线图 P132/P133 结尾点名的「下一步」）：
  * 第 142 篇（P132-1）给了**前向的计算侧**预算（L1 m_grouped 204us + L1 epilogue 24us + L2 111us = 339us）；
  * 第 143 篇（P133-1）补上了**通信 + 归约侧**（L2 远端写回 ~186us + combine 归约 ~107us）并指出
    「两腿同量级（46% vs 54%）→ 融合 kernel 必须把它们**重叠**」；
  * 但那个结论是**串行口径相加**推出来的——**没有真机量过前向这台的 overlap_factor**。
    第 65 篇（P75）量的是通用「copy ∥ matmul」（全卡不重叠、限 SM 才重叠）、第 94 篇（P60）量的是
    **反向**；前向的 `通信（dispatch/combine a2a）∥ 计算（routed grouped GEMM）` 从没被量过。

本篇补上：在 **8×H100** 上用**真实 grouped GEMM**（DeepGEMM SM90 `m_grouped_fp8_gemm_nt_contiguous`，
L1+L2，DeepSeek 量级 `T=2048,topk=2,H=7168,I=2048,E=8`）× **限 SM 的 NCCL a2a**
（dispatch fp8 `T·k·H·1B` + combine bf16 `T·k·H·2B`，共 ~88MB/rank）量出 overlap_factor，
并扫描 `NCCL_MAX_NCHANNELS`（通信占用的 channel 数 ≈ SM 数）看「SM 预算 → 重叠收益」的关系。

可跑性：真实 grouped GEMM 是 SM90 kernel（**H100 可跑**）；a2a 是 `torch.distributed` NCCL
（真机通信）。Mega MoE 融合 kernel 本体 SM100 专属 → 仅分析。本机 8 卡共享、无 NVLS（fabric manager
未开）→ 通信带宽为**受争用下界**，报数时注明。

用法（driver 见 exp_forward_overlap.sh）：
    python exp_forward_overlap.py --maxch N --reps R
"""
import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument('--maxch', type=int, default=0, help='NCCL_MAX_NCHANNELS（0=不设，用 NCCL 默认）')
_ap.add_argument('--reps', type=int, default=1, help='计算腿重复次数（缩放计算量）')
_ap.add_argument('--world', type=int, default=int(os.environ.get('FOV_WORLD', '8')))
_ap.add_argument('--iters', type=int, default=20)
_args, _ = _ap.parse_known_args()

if _args.maxch and _args.maxch > 0:
    os.environ['NCCL_MAX_NCHANNELS'] = str(_args.maxch)
os.environ.setdefault('NCCL_NVLS_ENABLE', '0')   # 本机无 fabric manager，NVLS 会报错
os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('NCCL_MIN_NCHANNELS', '1')
os.environ.setdefault('NCCL_CUMEM_ENABLE', '0')

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WORLD = _args.world
T, H, I, E, KTOP = 2048, 7168, 2048, 8, 2
REPS = _args.reps
ITERS = _args.iters


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              os.path.expanduser('~/github/DeepGEMM')):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


sys.path.insert(0, os.path.join(_find_deepgemm_dir(), 'tests'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------------------
# 通信腿：dispatch(fp8) + combine(bf16) 的 balanced all_to_all_single
# --------------------------------------------------------------------------------------
def make_comm(device):
    rows = T * KTOP
    per = rows // WORLD
    dbytes = rows * H * 1        # dispatch：T·k·H·1B（fp8 激活）
    cbytes = rows * H * 2        # combine ：T·k·H·2B（bf16 结果）
    dx = torch.ones(WORLD * per, H, dtype=torch.uint8, device=device)
    dy = torch.empty_like(dx)
    cx = torch.ones(WORLD * per, H * 2, dtype=torch.uint8, device=device)
    cy = torch.empty_like(cx)

    def f():
        dist.all_to_all_single(dy, dx)
        dist.all_to_all_single(cy, cx)

    return f, dbytes + cbytes, (dx, dy, cx, cy)


# --------------------------------------------------------------------------------------
# 计算腿：真实 grouped GEMM（L1 + L2，DeepGEMM SM90 1D2D fp8）
# --------------------------------------------------------------------------------------
def make_compute(device):
    from exp_forward_roofline import build_m_grouped
    counts = [T * KTOP // E] * E
    f1, i1 = build_m_grouped(2 * I, H, counts, device)   # linear1: [m,H]·W1ᵀ→[m,2I]
    f2, i2 = build_m_grouped(H, I, counts, device)       # linear2: [m,I]·W2ᵀ→[m,H]

    def f():
        for _ in range(REPS):
            f1()
            f2()

    return f, (i1, i2)


# --------------------------------------------------------------------------------------
def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3   # 秒


def time_serial(comm_f, comp_f, comm_s, cur, iters):
    for _ in range(3):
        with torch.cuda.stream(comm_s):
            comm_f()
        cur.wait_stream(comm_s)
        comp_f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        with torch.cuda.stream(comm_s):
            comm_f()
        cur.wait_stream(comm_s)
        comp_f()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3


def time_overlap(comm_f, comp_f, comm_s, cur, iters):
    # 计算发在默认流、通信发在 comm 流，末尾才让默认流等 comm
    for _ in range(3):
        with torch.cuda.stream(comm_s):
            comm_f()
        comp_f()
        cur.wait_stream(comm_s)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        with torch.cuda.stream(comm_s):
            comm_f()
        comp_f()
        cur.wait_stream(comm_s)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3


def worker(rank, world):
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('nccl', rank=rank, world_size=world)

    comm_f, nbytes, bufs = make_comm(dev)
    comp_f, infos = make_compute(dev)
    flops = sum(i['flops'] for i in infos) * REPS

    comm_s = torch.cuda.Stream(priority=-1)
    cur = torch.cuda.current_stream()

    t_comm = timeit(comm_f, ITERS)
    t_comp = timeit(comp_f, ITERS)
    t_serial = time_serial(comm_f, comp_f, comm_s, cur, ITERS)
    t_overlap = time_overlap(comm_f, comp_f, comm_s, cur, ITERS)

    # ---- 正确性：重叠不改变 comm 输出（逐字节）----
    dx, dy, cx, cy = bufs
    for t in (dy, cy):
        t.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(comm_s):
        comm_f()
    cur.wait_stream(comm_s)
    torch.cuda.synchronize()
    ref_d = dy.clone()
    ref_c = cy.clone()
    for t in (dy, cy):
        t.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(comm_s):
        comm_f()
    comp_f()
    cur.wait_stream(comm_s)
    torch.cuda.synchronize()
    same_d = bool(torch.equal(dy, ref_d))
    same_c = bool(torch.equal(cy, ref_c))

    # ---- 汇总到 rank0 ----
    stats = torch.tensor([t_comm, t_comp, t_serial, t_overlap,
                          float(same_d and same_c)], device=dev, dtype=torch.float64)
    gathered = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(gathered, stats)
    if rank == 0:
        tcs = torch.stack(gathered)
        # 取中位数（避免共享节点上个别人拖尾）
        tc, tk, ts, to = (tcs[:, i].median().item() for i in range(4))
        ok = bool(tcs[:, 4].min().item() > 0.5)
        ovf = ts / to if to > 0 else 0.0
        hidden = (ts - to) / min(tc, tk) if min(tc, tk) > 0 else 0.0
        ideal = max(tc, tk)
        eff = ideal / to if to > 0 else 0.0
        print(f'RESULT maxch={_args.maxch or 0} reps={REPS} '
              f't_comm={tc*1e6:.1f}us t_comp={tk*1e6:.1f}us serial={ts*1e6:.1f}us '
              f'overlap={to*1e6:.1f}us ovf={ovf:.3f} '
              f'ideal_overlap={ideal*1e6:.1f}us eff_vs_ideal={eff:.3f} '
              f'hidden_frac={hidden:.3f} '
              f'comm_bw={nbytes/1e9/tc:.1f}GB/s comp_tflops={flops/1e12/tk:.0f} '
              f'bitsame={ok}')
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(26000 + (os.getpid() % 3000))
    print(f'# P134 forward overlap  maxch={_args.maxch or 0} reps={REPS} world={WORLD}')
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
