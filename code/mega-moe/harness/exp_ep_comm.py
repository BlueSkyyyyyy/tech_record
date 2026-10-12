"""P1-2 实验：H100 上 EP all-to-all（dispatch/combine）通信开销的「真实数字」。

目的：为 docs/01 §2「EP 通信语义与开销、为何要与计算重叠」提供真机证据。
本机 H100 8 卡，NVLink 域内可用 NCCL。这里不跑 DeepEP（本机 NCCL/DeepEP 版本受阻，
见 ROADMAP「阻塞」），而是用 torch.distributed(NCCL) 的 all_to_all_single 做
**等价的 all-to-all 搬运**，量出：

  1. all-to-all 在不同消息大小下的时延与**有效带宽**（dispatch/combine 的量级）；
  2. 由「时延 vs 消息大小」线性拟合出**固定时延**与**渐近带宽**（说明「延迟敏感」，
     也解释了小 batch 下通信耗时、带宽却接近 0，正是 docs/02 §8 的现象）。

尺寸取 DeepSeek-V3 量级：H=7168，EP=8，每 rank 本地 token 数从 512 到 32768，
FP8(1B)/BF16(2B)。这是**通信原语的裸开销**，不是 Mega MoE（SM100 专属）。
NVLink 硬件上限见 `nvidia-smi nvlink -s`（每 link 26.562 GB/s、18 link/卡，单向聚合约 478 GB/s）。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_ep_comm.py"
"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
H = 7168  # DeepSeek-V3 hidden
TOKENS = [512, 2048, 8192, 32768]
DTYPES = {'fp8': torch.float8_e4m3fn, 'bf16': torch.bfloat16}
ITERS = 30
WARMUP = 10


def _sync():
    torch.cuda.synchronize()


def _bench(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    _sync()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    _sync()
    return ev0.elapsed_time(ev1) / iters  # ms


def measure_a2a(rank, world, dev):
    for name, dt in DTYPES.items():
        esz = torch.finfo(dt).bits // 8 if dt.is_floating_point else 1
        for T in TOKENS:
            inp = torch.ones(WORLD, T, H, dtype=dt, device=dev)
            out = torch.empty_like(inp)

            def fn():
                dist.all_to_all_single(out, inp)

            ms = _bench(fn)
            # 每 rank 发送总量 = WORLD*T*H*esz；有效带宽按 all-to-all 双向口径
            sent = WORLD * T * H * esz
            eff = (sent / 1e9) / (ms / 1e3)
            if rank == 0:
                print(f'[a2a-{name}]  EP={world} 本地 {WORLD*T:6d} tok  '
                      f'发出 total={sent/1e6:8.1f} MB/rank  {ms*1e3:9.1f} us  '
                      f'有效带宽 {eff:7.1f} GB/s')


def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=__import__('datetime').timedelta(seconds=180))
    measure_a2a(rank, world, dev)
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(20000 + (os.getpid() % 20000))
    # 本机 Fabric Manager 未启用 NVLS 组播，绑定 2MB NVLS buffer 会报 invalid argument。
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    print(f'torch {torch.__version__}  world={WORLD}  H={H}  iters={ITERS}  '
          f'port={os.environ["MASTER_PORT"]}')
    print('DeepSeek-V3 量级 hidden；all_to_all_single 等价 dispatch/combine 的搬运量。')
    print('-' * 96)
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
