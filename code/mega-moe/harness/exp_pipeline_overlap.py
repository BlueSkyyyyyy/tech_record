"""P155-1 实验：**跨 micro-batch 流水线到底能不能藏 EP 通信**（dispatch_{i+1} ∥ compute_i）。

背景（第 144 篇 / P134-1 的「下一步」①/②）：
  * 第 144 篇在 H100 上量得一个反直觉负结果——把 **NCCL a2a 发在侧流、真实 grouped GEMM 发在主**流，
    overlap_factor ≈ 1.0（几乎不重叠），因为 NCCL 的 a2a kernel 会占满 SM 把计算挤出去；只有给
    通信一个**小 SM 预算**（第 63 篇 `get_theoretical_num_sms`），正对照里 overlap 才成立（ovf ~1.98）。
  * 但那是**同一个 micro-batch** 的两条腿并发。生产训练真正的做法是 **micro-batch 流水线**：
    让 `dispatch_{i+1}` 去盖住 `compute_i`。本篇把这个问题单独量出来：
    **流水线深度能不能把 NCCL EP 通信藏住？**

实验（8×H100，真实 SM90 grouped GEMM + NCCL a2a）：
  * 每条 micro-batch 的「真前向」= `dispatch(a2a) → L1 → L2 → combine(a2a)`；
  * **serial**：按 i 逐个 micro-batch 串行跑；
  * **pipeline**：通信发在侧流 `comm_s`、计算发在主流程，令 `K_i` 等 `D_i`、`C_i` 等 `K_i`，
    并把 `D_0..D_{M-1}` 先排好 → `D_{i+1}` 自然落在 `K_i` 的时间窗里；
  * 扫 micro-batch 数 M ∈ {1,2,4,8} × 计算重复 REPS ∈ {1,3}。

可跑性：grouped GEMM 是 SM90 kernel（**H100 可跑**）；a2a 是 `torch.distributed` NCCL（真机通信）。
Mega MoE 融合 kernel 本体 SM100 专属 → 仅分析。本机 8 卡共享、无 NVLS → 通信带宽为**受争用下界**。

用法（driver 见 exp_pipeline_overlap.sh）：
    python exp_pipeline_overlap.py [--world 8] [--iters 10]
"""
import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument('--world', type=int, default=int(os.environ.get('POV_WORLD', '8')))
_ap.add_argument('--iters', type=int, default=int(os.environ.get('POV_ITERS', '8')))
_ap.add_argument('--repeats', type=int, default=int(os.environ.get('POV_REPEATS', '5')),
                 help='每个测量重复次数，取中位数以抗共享节点争用')
_args, _ = _ap.parse_known_args()

os.environ.setdefault('NCCL_NVLS_ENABLE', '0')   # 本机无 fabric manager，NVLS 会报错
os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('NCCL_MIN_NCHANNELS', '1')
os.environ.setdefault('NCCL_CUMEM_ENABLE', '0')

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WORLD = _args.world
ITERS = _args.iters
REPEATS = _args.repeats
# 每 micro-batch 的 token 数 T_m 取 512：使 `T_m*ktop/E = 128`（= m-grouped contiguous 的 mk 对齐）
# → 每条 micro-batch 都能直接喂真实 `m_grouped_fp8_gemm_nt_contiguous`。
TM = 512
H, I, E, KTOP = 7168, 2048, 8, 2
REPS_LIST = (1, 3)
M_LIST = (1, 2, 4, 8)


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              '~/github/DeepGEMM'):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


sys.path.insert(0, os.path.join(_find_deepgemm_dir(), 'tests'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_micros(device):
    """为 M 条 micro-batch 各建 comm / compute 闭包，返回 {M: (comm_fn(dispatch[,combine]), comp_fn, bytes)}。

    采用固定 T_m：M 越大、总工作量越大（流水线稳态步数越多）。
    """
    from exp_forward_roofline import build_m_grouped

    rows = TM * KTOP                       # 1024 行/micro-batch
    per = rows // WORLD                    # 每 rank 发/收 128 行
    dbytes = rows * H * 1                  # dispatch：T_m·k·H·1B（fp8）
    cbytes = rows * H * 2                  # combine ：T_m·k·H·2B（bf16）
    counts = [rows // E] * E               # 每专家 128 行
    assert counts[0] % 128 == 0, counts

    # comm 缓冲：每条 micro-batch 一套独立缓冲（避免互相覆盖，且能逐位核对）
    cbuf = []
    for _ in range(max(M_LIST)):
        dx = torch.ones(WORLD * per, H, dtype=torch.uint8, device=device)
        dy = torch.empty_like(dx)
        cx = torch.ones(WORLD * per, H * 2, dtype=torch.uint8, device=device)
        cy = torch.empty_like(cx)
        cbuf.append((dx, dy, cx, cy))

    # compute：每条 micro-batch 一套（真形状），再按 REPS 重复
    f1, _ = build_m_grouped(2 * I, H, counts, device)   # L1: [m,H]·W1ᵀ→[m,2I]
    f2, _ = build_m_grouped(H, I, counts, device)       # L2: [m,I]·W2ᵀ→[m,H]

    def dispatch(i):
        dx, dy, _, _ = cbuf[i]
        dist.all_to_all_single(dy, dx)

    def combine(i):
        _, _, cx, cy = cbuf[i]
        dist.all_to_all_single(cy, cx)

    def compute(i, reps):
        for _ in range(reps):
            f1()
            f2()

    return dispatch, combine, compute, (dbytes + cbytes), cbuf


_FLUSH = None


def flush_l2(device):
    """清 256MB L2（与 `deep_gemm/testing/bench.py:7-33` 同口径），避免 comm/GEMM 缓冲驻留 L2 造成假快。"""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    _FLUSH.zero_()


def time_loop(run, iters, device, warmup=3):
    # 与 `deep_gemm/testing/bench.py` 同口径：计时前清一次 L2，计时区间内不清（热态稳态）。
    flush_l2(device)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        run()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3   # 秒


def robust_time(run, iters, device, repeats=REPEATS):
    """重复 repeats 次取中位数——共享 8 卡节点上会有别的自驱任务抢卡造成离群尖峰。"""
    vals = sorted(time_loop(run, iters, device) for _ in range(repeats))
    return vals[len(vals) // 2]


def make_schedules(M, reps, dispatch, combine, compute, comm_s):
    main = torch.cuda.current_stream()

    def serial_micro():
        for i in range(M):
            dispatch(i)
            compute(i, reps)
            combine(i)

    def serial_grouped():
        # 无重叠基线：所有 dispatch → 所有 compute → 所有 combine（单流顺序）。
        # 与 pipeline 的差别**只在**「是否把计算与通信放到两条流并发」。
        for i in range(M):
            dispatch(i)
        for i in range(M):
            compute(i, reps)
        for i in range(M):
            combine(i)

    def pipeline():
        ev_d = [torch.cuda.Event() for _ in range(M)]
        ev_k = [torch.cuda.Event() for _ in range(M)]
        # 1) 先把 D_0..D_{M-1} 全排到 comm 流（D_{i+1} 不依赖任何计算）
        with torch.cuda.stream(comm_s):
            for i in range(M):
                dispatch(i)
                ev_d[i].record(comm_s)
        # 2) 主流程：K_i 等 D_i（main 流顺序保证 K_i 串行）
        for i in range(M):
            main.wait_event(ev_d[i])
            compute(i, reps)
            ev_k[i].record(main)
        # 3) comm 流回填 C_i：C_i 等 K_i（comm 流顺序保证 D 先于 C）
        with torch.cuda.stream(comm_s):
            for i in range(M):
                comm_s.wait_event(ev_k[i])
                combine(i)
        main.wait_stream(comm_s)

    return serial_micro, serial_grouped, pipeline


def worker(rank, world):
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('nccl', rank=rank, world_size=world)

    dispatch, combine, compute, nbytes, cbuf = make_micros(dev)
    comm_s = torch.cuda.Stream(priority=-1)

    print(f'[rank{rank}] dispatch+combine = {nbytes / 1e6:.2f} MB/micro-batch '
          f'(d={TM * KTOP * H / 1e6:.2f} c={TM * KTOP * H * 2 / 1e6:.2f})', flush=True)

    for reps in REPS_LIST:
        for M in M_LIST:
            serial_micro, serial_grouped, pipeline = make_schedules(
                M, reps, dispatch, combine, compute, comm_s)

            t_serial = robust_time(serial_micro, ITERS, dev)
            t_grouped = robust_time(serial_grouped, ITERS, dev)
            t_pipe = robust_time(pipeline, ITERS, dev)

            def all_comm():
                for i in range(M):
                    dispatch(i)
                    combine(i)

            t_comm = robust_time(all_comm, ITERS, dev)          # dispatch + combine（全 M 条）
            t_comp = robust_time(lambda: [compute(i, reps) for i in range(M)], ITERS, dev)

            # ---- 正确性：pipeline 不改变 comm 输出（逐字节）----
            for i in range(M):
                for t in (cbuf[i][1], cbuf[i][3]):
                    t.zero_()
            torch.cuda.synchronize()
            serial_micro()
            torch.cuda.synchronize()
            ref = [(cbuf[i][1].clone(), cbuf[i][3].clone()) for i in range(M)]

            for i in range(M):
                for t in (cbuf[i][1], cbuf[i][3]):
                    t.zero_()
            torch.cuda.synchronize()
            pipeline()
            torch.cuda.synchronize()
            bitsame = all(bool(torch.equal(cbuf[i][1], ref[i][0]) and
                               torch.equal(cbuf[i][3], ref[i][1])) for i in range(M))

            stats = torch.tensor([t_serial, t_grouped, t_pipe, t_comm, t_comp, float(bitsame)],
                                 device=dev, dtype=torch.float64)
            gathered = [torch.zeros_like(stats) for _ in range(world)]
            dist.all_gather(gathered, stats)
            if rank == 0:
                s = torch.stack(gathered)
                ts, tg, tp, tc, tk = (s[:, i].median().item() for i in range(5))
                ok = bool(s[:, 5].min().item() > 0.5)
                ovf_micro = ts / tp if tp > 0 else 0.0       # 相对朴素 per-micro 串行
                ovf_grouped = tg / tp if tp > 0 else 0.0     # 相对「无重叠最优」分组串行（≤1 = 没赢）
                ideal = max(tc, tk)
                ideal_ovf = (tc + tk) / ideal if ideal > 0 else 0.0   # 理论最优重叠比
                hidden = (ts - tp) / min(tc, tk) if min(tc, tk) > 0 else 0.0
                print(f'RESULT M={M} reps={reps} '
                      f't_comm={tc * 1e6:.1f}us t_comp={tk * 1e6:.1f}us '
                      f'serial_micro={ts * 1e6:.1f}us serial_grouped={tg * 1e6:.1f}us '
                      f'pipeline={tp * 1e6:.1f}us '
                      f'ovf_vs_micro={ovf_micro:.3f} ovf_vs_grouped={ovf_grouped:.3f} '
                      f'ideal_ovf={ideal_ovf:.3f} hidden_frac={hidden:.3f} '
                      f'comm_bw={M * nbytes / 1e9 / tc:.1f}GB/s bitsame={ok}', flush=True)
            dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29500 + (os.getpid() % 2000))
    print(f'# P155 pipeline overlap  world={WORLD} T_m={TM} H={H} I={I} E={E} ktop={KTOP}')
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
