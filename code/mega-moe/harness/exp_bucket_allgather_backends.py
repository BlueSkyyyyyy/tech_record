#!/usr/bin/env python3
"""P102-1 host 算术层对拍：DeepEP bucket hybrid all_gather 的 chunk 尺寸自选择。

逐行搬运 csrc/kernels/bucket/all_gather.hpp:116-148 的代价模型，并与独立复刻互证。
架构无关（纯 host/Python），H100 真机上直接跑。

真实行号（DeepEP@93eb6eb）：
  all_gather.hpp:116-118  num_shard_bytes = Σ bucket.num_bytes
  all_gather.hpp:123-126  push_t/rdma_t/forward_t
  all_gather.hpp:129-135  枚举 i∈[1, min(64, shard/4MiB)] 取 t 最小
  all_gather.hpp:137-142  num_chunk_bytes = clamp(align(ceil_div(shard,best), 8*64), 4KiB, 256MiB)
  all_gather.hpp:145-148  num_chunks = Σ ceil_div(bucket_bytes, chunk); assert <= 8192
"""
import math

K_NUM_QPS_PER_CHUNK = 8                 # all_gather.cuh:11
K_NUM_RDMA_ALIGNMENT_BYTES = 64         # compiled.cuh:68
K_NUM_MIN_CHUNK_BYTES = 4 * 1024 * 1024  # all_gather.hpp:123
K_NUM_MAX_CHUNKS = 8192                 # all_gather.cuh:12
K_NUM_MAX_CHUNK_BYTES = 256 * 1024 * 1024  # ChunkStorage::kNumBytes


def align(a, b):
    return (a + b - 1) // b * b


def choose_chunk(num_shard_bytes, num_rdma_ranks, num_nvl_ranks, nvl_bw, rdma_bw):
    """与 all_gather.hpp:116-142 逐行等价。带宽单位 bytes/s，时间单位秒。"""
    push_t = num_shard_bytes * (num_nvl_ranks - 1) / nvl_bw
    rdma_t = num_shard_bytes * (num_rdma_ranks - 1) / rdma_bw
    forward_t = push_t * (num_rdma_ranks - 1)
    num_best_chunks, best_t = 1, math.inf
    hi = min(64, num_shard_bytes // K_NUM_MIN_CHUNK_BYTES)
    for i in range(1, hi + 1):
        t = max(max(push_t, rdma_t / i) + forward_t + i * 10e-6,
                rdma_t + forward_t / i)
        if t < best_t:
            best_t, num_best_chunks = t, i
    num_chunk_bytes = max(4 * 1024, min(
        align((num_shard_bytes + num_best_chunks - 1) // num_best_chunks,
              K_NUM_QPS_PER_CHUNK * K_NUM_RDMA_ALIGNMENT_BYTES),
        K_NUM_MAX_CHUNK_BYTES))
    return num_best_chunks, best_t, num_chunk_bytes


def independent_chunk_reference(shard, rdma, nvl, nvl_bw, rdma_bw):
    """独立复刻：用浮点显式枚举，验证与上面的整数/浮点混合写法一致。"""
    push = shard * (nvl - 1) / nvl_bw
    rdma_t = shard * (rdma - 1) / rdma_bw
    fwd = push * (rdma - 1)
    cand = [1]
    for i in range(1, min(64, shard // K_NUM_MIN_CHUNK_BYTES) + 1):
        cand.append(i)
    costs = {}
    for i in cand:
        ce_bound = max(push, rdma_t / i) + fwd + i * 10e-6
        rdma_bound = rdma_t + fwd / i
        costs[i] = max(ce_bound, rdma_bound)
    best = min(cand, key=lambda i: costs[i])
    raw = math.ceil(shard / best)
    packed = ((raw + 511) // 512) * 512
    return best, max(4096, min(packed, K_NUM_MAX_CHUNK_BYTES))


def main():
    print("P102-1 host 算术层：hybrid all_gather 的 chunk 尺寸自选择（all_gather.hpp:116-148）")
    print(f"  常量: QPs/chunk={K_NUM_QPS_PER_CHUNK}, RDMA align={K_NUM_RDMA_ALIGNMENT_BYTES}B, "
          f"min chunk={K_NUM_MIN_CHUNK_BYTES}, max chunks={K_NUM_MAX_CHUNKS}")
    print()

    # 真实带宽取值（第 63/54 篇：nvidia-smi nvlink 求和 *0.9 = 430.3 GB/s；RDMA 取典型 100Gb/s=12.5GB/s 与 400Gb/s=50GB/s）
    nvl_bw = 430.3e9
    rdma_bws = {"100Gb/s": 12.5e9, "400Gb/s": 50e9}

    configs = [  # (rdma_ranks, nvl_ranks)
        (8, 1), (16, 1),          # pure RDMA（nvl=1 → 不走 hybrid 分支，仅为对照）
        (2, 8), (4, 8), (8, 8),   # hybrid EP16/32/64
        (2, 4), (4, 4),
    ]
    shards = [1 << 20, 16 << 20, 1 << 30, 8 << 30, 64 << 30]

    print("### 表：代价模型选择的（best_chunks, chunk_bytes, QP 均为 512KiB 的倍数, num_chunks <= 8192）")
    fails = 0
    for name, rdma_bw in rdma_bws.items():
        print(f"\n  --- RDMA={name} ({rdma_bw/1e9:.1f} GB/s), NVLink={nvl_bw/1e9:.1f} GB/s ---")
        for (rdma, nvl) in configs:
            if rdma <= 1 or nvl <= 1:
                continue
            for shard in shards:
                best, t, chunk = choose_chunk(shard, rdma, nvl, nvl_bw, rdma_bw)
                best2, chunk2 = independent_chunk_reference(shard, rdma, nvl, nvl_bw, rdma_bw)
                # 一个 bucket = 整个 shard
                num_chunks = (shard + chunk - 1) // chunk
                ok = (chunk == chunk2) and (chunk % 512 == 0) and (chunk >= 4096) \
                    and (chunk <= K_NUM_MAX_CHUNK_BYTES) and (num_chunks <= K_NUM_MAX_CHUNKS)
                # 「每个 QP >= 512KiB」只在 shard >= min chunk（循环真正跑起来）时是设计目标
                if shard >= K_NUM_MIN_CHUNK_BYTES:
                    ok = ok and (chunk // K_NUM_QPS_PER_CHUNK >= 512 * 1024)
                if not ok:
                    fails += 1
                print(f"    R={rdma}x{nvl} shard={shard/2**30:6.2f}GiB -> best_chunks={best:2d} "
                      f"chunk={chunk/2**20:8.2f}MiB (per QP {chunk//K_NUM_QPS_PER_CHUNK/2**10:.0f}KiB) "
                      f"num_chunks={num_chunks:5d} {'OK' if ok else 'FAIL'}")
    print(f"\n  -> 全部性质（对齐/夹取/>=512KiB/QP/独立复刻一致）: {'PASS' if fails == 0 else 'FAIL'} (fails={fails})")

    # 边界：shard 不足 4MiB → 循环不执行 → num_best_chunks=1，chunk 被夹到 4KiB 或对齐值
    print("\n### 边界：极小 shard（< min chunk）")
    for shard in [0, 1, 1023, 4096, 4097, 1 << 20, 4 << 20]:
        best, t, chunk = choose_chunk(shard, 4, 8, nvl_bw, 50e9)
        print(f"    shard={shard:>8} -> best_chunks={best} chunk={chunk} ({chunk/1024:.1f}KiB)")
    print("  （shard=0 时 ceil_div(0,best)=0 -> align 0 = 0 -> clamp 到 4096）")

    # 性质：chunk 永远落在 [4KiB, 256MiB] 且 512B 对齐（跨全 shard 区间）
    print("\n### 性质：chunk 恒在 [4KiB, 256MiB] 且 512B 对齐")
    bad = 0
    for shard in range(1 << 20, (512 << 20) + 1, 1 << 20):
        _, _, chunk = choose_chunk(shard, 4, 8, nvl_bw, 50e9)
        if not (4096 <= chunk <= K_NUM_MAX_CHUNK_BYTES and chunk % 512 == 0):
            bad += 1
    print(f"    越界/未对齐: {bad} 次 -> {'PASS' if bad == 0 else 'FAIL'}")

    print("\nDONE")


if __name__ == "__main__":
    main()
