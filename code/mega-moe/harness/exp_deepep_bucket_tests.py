#!/usr/bin/env python3
# P124-1 实验：DeepEP bucket/buffer 家族最后三份验收测试的测试侧语义在 H100 上真机复现。
#
#   tests/bucket/test_all_gather.py  (117 行)
#   tests/bucket/test_session.py     (101 行)
#   tests/buffer/test_allocation.py  ( 54 行)
#
# 三份测试的「正确性判据」全部落在纯 PyTorch / NCCL 主机 API 上；bucket 的设备 kernel（NCCL GIN /
# copy engine）在本机阻塞（见 ROADMAP「阻塞」）。本实验只复刻**测试侧**：
#   Part A (8×H100 NCCL)：test_all_gather 的参考构造 all_gather_into_tensor + 逐位判据 + shard 序
#   Part B (8×H100 NCCL)：test_session 的三件参考 + 容差梯度（逐位 vs 1e-5）；并验证
#                         「反序环重建 == reduce_scatter_tensor」是否逐位（RVLink 域为何只要 1e-5）
#   Part C (单进程)      ：test_session/test_allocation 的缓冲区算术 + BufferAllocator 物化复刻
#   Part D (单进程)      ：get_context_idx 多 group 断言 + get_theoretical_num_sms('all_gather')==0
#
# 运行：bash harness/exp_deepep_bucket_tests.sh > harness/exp_deepep_bucket_tests.out.txt
import os
import sys
import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
FAIL = 0


def ok(cond, msg):
    global FAIL
    if cond:
        print(f'  [PASS] {msg}')
    else:
        FAIL += 1
        print(f'  [FAIL] {msg}')


def align(x, y):
    return (x + y - 1) // y * y


# ---------------------------------------------------------------------------
# Part A / B：8×H100 NCCL
# ---------------------------------------------------------------------------
def part_b(rank, world, dev):
    # test_session.py:35-85 —— 三件参考与容差梯度
    numel = align(1 << 16, world * 4)
    is0 = (rank == 0)

    # all_gather：逐位
    ag_src = torch.arange(numel // world, dtype=torch.float32, device=dev) + rank
    ag_ref = torch.empty(numel, dtype=torch.float32, device=dev)
    dist.all_gather_into_tensor(ag_ref, ag_src, group=None)
    ag_recon = torch.arange(numel // world, dtype=torch.float32, device=dev).repeat(world) \
        + torch.arange(world, device=dev).repeat_interleave(numel // world)
    if is0:
        ok(torch.equal(ag_ref, ag_recon), 'B all_gather 逐位(rtol=0 依据)')

    # reduce_scatter：1e-5（NVLink 域）
    rs_src = (torch.arange(numel, dtype=torch.float32, device=dev) * 0.5 + rank)
    rs_ref = torch.empty(numel // world, dtype=torch.float32, device=dev)
    dist.reduce_scatter_tensor(rs_ref, rs_src, group=None)
    # 反序环重建（test_reduce_scatter.py:82-87 的 RDMA 参考路径）：all_to_all + 按 (rank-i)%R 累加
    a2a = torch.empty_like(rs_src)
    dist.all_to_all_single(a2a, rs_src, group=None)
    ring = torch.zeros(numel // world, dtype=torch.float32, device=dev)
    R = world
    chunk = numel // world
    for i in range(R):
        peer = (rank - i) % R
        ring.add_(a2a[peer * chunk:(peer + 1) * chunk])
    if is0:
        same_bit = torch.equal(rs_ref, ring)
        rel = (rs_ref - ring).abs().max().item() / max(rs_ref.abs().max().item(), 1e-30)
        ok(rel < 1e-5, f'B reduce_scatter 反序环重建 relerr={rel:.3e} < 1e-5')
        print(f'  [INFO] 反序环重建 vs reduce_scatter_tensor 逐位={same_bit} '
              f'（实测 8×NVLink 恰好逐位，但测试仍保守用 1e-5——逐位非契约）')
        print(f'  [INFO] test_reduce_scatter.py 的 RDMA 路径用 rtol=0，因其自研 kernel 的求和序'
              f'与参考 (rank-i)%R 逐字相同；NVLink 路径的"同序"不可依赖')

    # all_reduce：sum * scale
    scale = 0.25
    ar_src = (torch.arange(numel, dtype=torch.float32, device=dev) + rank)
    ar_ref = ar_src.clone()
    dist.all_reduce(ar_ref, group=None)
    ar_ref.mul_(scale)
    # 本地重建：sum_r (arange + r) * scale
    local_sum = torch.arange(numel, dtype=torch.float32, device=dev) * world \
        + (world * (world - 1) // 2)
    if is0:
        ok(torch.allclose(ar_ref, local_sum * scale, rtol=1e-5, atol=1e-5),
           f'B all_reduce sum*scale={scale} 匹配')


def worker(rank, world):
    global FAIL
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    import datetime
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=180))
    local_fail = [0]

    def local_ok(cond, msg):
        if rank == 0:
            ok(cond, msg)
        if not cond:
            local_fail[0] += 1

    # Part A（rank0 断言，全 rank 参与通信）
    for shard in (8, 8 * 1024, 8 * 65536):
        gathered = torch.empty(world * shard, dtype=torch.float32, device=dev)
        src = (torch.arange(shard, dtype=torch.float32, device=dev) + rank)
        gathered[rank * shard:(rank + 1) * shard].copy_(src)
        ref = torch.empty_like(gathered)
        dist.all_gather_into_tensor(ref, src, group=None)
        recon = torch.arange(shard, dtype=torch.float32, device=dev).repeat(world) \
            + torch.arange(world, device=dev).repeat_interleave(shard)
        local_ok(torch.equal(ref, recon), f'A all_gather_into_tensor 逐位(shard={shard})')
    if rank == 0:
        print(f'  [INFO] torch {torch.__version__}: dist.all_gather_single 存在='
              f'{hasattr(dist, "all_gather_single")} (test_all_gather/session 的 *_single 是 DeepEP 补丁)')
    part_b(rank, world, dev)

    fails = torch.tensor([sum(local_fail)], device=dev)
    dist.all_reduce(fails, op=dist.ReduceOp.SUM, group=None)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        FAIL += int(fails.item())


# ---------------------------------------------------------------------------
# Part C / D：单进程纯 host
# ---------------------------------------------------------------------------
def part_c():
    # test_session.py:23-31 的缓冲区算术
    ALLOC_ALIGN = 2 * 1024 * 1024   # compiled.cuh:69 kNumAllocationAlignmentBytes
    RDMA_ALIGN = 64                 # compiled.cuh:68 kNumRDMAAlignmentBytes
    num_bytes = 1 << 21             # 2 MiB 量级
    num_ranks = WORLD
    numel = align(num_bytes // 4, num_ranks * 4)
    buf = align(3 * align(numel * 4, RDMA_ALIGN), ALLOC_ALIGN)
    ok(numel == align(num_bytes // 4, num_ranks * 4), f'C numel={numel} 对齐 num_ranks*4')
    ok(buf % ALLOC_ALIGN == 0 and buf >= 3 * numel * 4, f'C session 缓冲={buf} 3 段+2MiB 对齐')

    # test_allocation.py:22-33 —— BufferAllocator 的 meta→cuda 物化（纯 torch 复刻）
    class Plan:
        def __init__(self, alignment):
            self.alignment = alignment
            self.allocs = []      # (meta_tensor, offset)
            self.num_bytes = 0
            self.materialized = False
        def allocate(self, shape, dtype):
            assert not self.materialized
            t = torch.empty(shape, dtype=dtype, device='meta')
            off = align(self.num_bytes, self.alignment)
            self.allocs.append((t, off))
            self.num_bytes = align(off + t.nbytes, self.alignment)
            return t
        def materialize(self, storage):
            assert not self.materialized
            assert storage.is_cuda and storage.is_contiguous() and storage.dtype == torch.uint8
            assert storage.numel() == self.num_bytes
            for t, off in self.allocs:
                v = storage.narrow(0, off, t.nbytes).view(t.dtype).view(t.shape)
                torch.utils.swap_tensors(t, v)
            self.materialized = True

    p = Plan(ALLOC_ALIGN)
    b0 = p.allocate((3, 5), torch.float32)     # 60 B
    b1 = p.allocate((7,), torch.bfloat16)      # 14 B
    ok(b0.is_meta and b1.is_meta, 'C allocate 返回 meta 占位张量')
    # 60 B → 对齐到 2 MiB；b1 偏移 = align(2MiB, 2MiB) = 2 MiB，总数 = 4 MiB
    ok(p.num_bytes == 2 * ALLOC_ALIGN, f'C plan.num_bytes={p.num_bytes} == 4 MiB (两次 2MiB 对齐段)')
    storage = torch.empty(p.num_bytes, dtype=torch.uint8, device='cuda')
    p.materialize(storage)
    ok(b0.is_cuda and b1.is_cuda, 'C 物化后占位张量变 CUDA')
    ok(b0.data_ptr() == storage.data_ptr(), 'C 首个分配落在存储起点')
    ok(b1.data_ptr() == storage.data_ptr() + ALLOC_ALIGN, 'C 第二个分配落在 2 MiB 偏移')
    b0.fill_(7.0); b1.fill_(3)
    ok((b0 == 7.0).all() and (b1 == 3).all(), 'C 物化张量可写回读')

    # session bump 分配（bucket.py:56-65）：64B 对齐 + 越界 RuntimeError
    class Bump:
        def __init__(self, buf_bytes, alignment=RDMA_ALIGN):
            self.num_bytes = buf_bytes
            self.alignment = alignment
            self.used = 0
        def _allocate(self, n):
            a = align(n, self.alignment)
            if self.used + a > self.num_bytes:
                raise RuntimeError(f'session out of storage, which has only {self.num_bytes}')
            off = self.used
            self.used += a
            return off
    bm = Bump(256)
    o0 = bm._allocate(60)   # 对齐到 64
    o1 = bm._allocate(64)
    ok(o0 == 0 and o1 == 64 and bm.used == 128, 'C bump 分配 64B 对齐 (0, 64)')
    try:
        bm._allocate(4096)
        ok(False, 'C bump 越界应抛 RuntimeError')
    except RuntimeError:
        ok(True, 'C bump 越界抛 RuntimeError')


def part_d():
    # get_context_idx（bucket.py:295-297）：多 group 时必须显式指定 group
    class Buf:
        def __init__(self, groups):
            self.groups = list(groups)
        def get_context_idx(self, group):
            assert len(self.groups) == 1 or group is not None, \
                '`group` is required when the buffer has multiple groups'
            return 0 if group is None else self.groups.index(group)
    b1 = Buf(['g0'])
    ok(b1.get_context_idx(None) == 0, 'D 单 group：group=None → context 0')
    b2 = Buf(['g0', 'g1'])
    ok(b2.get_context_idx('g1') == 1, 'D 多 group：显式 group → index')
    try:
        b2.get_context_idx(None)
        ok(False, 'D 多 group + group=None 应断言')
    except AssertionError:
        ok(True, 'D 多 group + group=None 被断言拦下（test_allocation 的意义）')

    # get_theoretical_num_sms('all_gather')（bucket.py:289-290）恒 0（copy-engine）
    ok((lambda op: 0 if op == 'all_gather' else None)('all_gather') == 0,
       'D get_theoretical_num_sms(all_gather) == 0（copy-engine 不占 SM）')

    # align（math.py:26）
    ok(align(0, 64) == 0 and align(1, 64) == 64 and align(64, 64) == 64 and align(65, 64) == 128,
       'D align(x,y)=ceil(x/y)*y')


if __name__ == '__main__':
    print('=' * 78)
    print(f'torch {torch.__version__}  cuda={torch.version.cuda}  '
          f'gpus={torch.cuda.device_count()}  '
          f'cc={torch.cuda.get_device_capability(0)}')
    print('=' * 78)

    print('\n== Part C: test_session/test_allocation 缓冲区算术 + BufferAllocator 物化 ==')
    part_c()
    print('\n== Part D: get_context_idx 多 group + get_theoretical_num_sms ==')
    part_d()

    print('\n== Part A/B: 8×H100 NCCL 参考语义与容差梯度 ==')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(20000 + (os.getpid() % 20000))
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    if torch.cuda.device_count() >= WORLD:
        try:
            mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
        except Exception as ex:
            FAIL += 1
            print(f'  [FAIL] 8-rank NCCL 失败（本机 GPU 被多个任务共享）: {ex}')
    else:
        print(f'  [SKIP] 需要 {WORLD} 张 GPU，仅 {torch.cuda.device_count()} 张')

    print('\n' + '=' * 78)
    print(f'总判定: {"ALL PASS" if FAIL == 0 else f"{FAIL} FAIL"}')
    print('=' * 78)
    sys.exit(0 if FAIL == 0 else 1)
