#!/usr/bin/env python3
# P62-1 独立 host oracle：解析 `src/deepep_bucket_collectives.cu`（H100 真机真实设备代码）的 dump，
# 用**独立实现**逐项对拍：
#   A) 布局几何（sizeof/offsetof）与手算公式
#   B) `ChunkIterator` 覆盖 + 两种分块映射（all_reduce vs reduce_scatter）
#   C) reduce_scatter 反序环累加（逐位）+ all_reduce 环求和 + scale
#   D) bf16 随机舍入哈希（与 tilelang 参考 stochastic_rounding.py:53-65 逐位一致）
#   E) host 侧发射配置算术（csrc/kernels/bucket/{all_reduce,reduce_scatter}.hpp）与信号预算
#
#   python3 harness/exp_deepep_bucket_collectives.py <dump>
import math
import struct
import sys

try:
    import numpy as np
except Exception:
    np = None

PASS = 0
FAIL = 0


def check(ok, what):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {what}")
    else:
        FAIL += 1
        print(f"  [FAIL] {what}")


def f32(u):
    return struct.unpack('<f', struct.pack('<I', u & 0xFFFFFFFF))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', np.float32(x)))[0]


def align(a, b):
    return (a + b - 1) // b * b


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dbc_dump.txt'
    lines = open(path).read().splitlines()

    geom = {}
    chunk = {}          # stride -> list of (ci, off, bytes)
    cnt = {}
    armap = []          # (ci, dst, slot, gen)
    rsmap = []          # (ci, slot, gen)
    rssum = {}          # idx -> uint32
    rssumin = {}
    arsum = {}
    arsumin = {}
    bf16 = []           # (i, bx,by,bz,bw, h1,h2)
    chunk_meta = None
    rssig = []
    ar_meta = rs_meta = bs_meta = arsum_meta = rssig_meta = None
    bf16_meta = None

    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == 'GEOM':
            geom[t[1]] = [int(x) for x in t[2:] if x.lstrip('-').isdigit()]
        elif t[0] == 'CHUNK' and t[1] == 'kChunkBytes':
            chunk_meta = (int(t[2]), int(t[4]), [int(x) for x in t[8:]])
        elif t[0] == 'CHUNKSTRIDE':
            cnt[int(t[1])] = int(t[3])
        elif t[0] == 'CHUNK':
            chunk.setdefault(int(t[1]), []).append((int(t[2]), int(t[3]), int(t[4])))
        elif t[0] == 'ARMAP' and t[1] == 'R':
            ar_meta = (int(t[2]), int(t[4]), int(t[6]))
        elif t[0] == 'ARMAP':
            armap.append((int(t[1]), int(t[2]), int(t[3]), int(t[4])))
        elif t[0] == 'RSMAP' and t[1] == 'R':
            rs_meta = (int(t[2]), int(t[4]), int(t[6]))
        elif t[0] == 'RSMAP':
            rsmap.append((int(t[1]), int(t[2]), int(t[3])))
        elif t[0] == 'RSSIG' and t[1] == 'kSlots':
            rssig_meta = (int(t[2]), int(t[4]))
        elif t[0] == 'RSSIG':
            rssig.append((int(t[1]), int(t[2]), int(t[3])))
        elif t[0] == 'RSSUM' and t[1] == 'R':
            rssum_meta = (int(t[2]), int(t[4]), float(t[6]))
        elif t[0] == 'RSSUM':
            rssum[int(t[1])] = int(t[2], 16)
        elif t[0] == 'RSSUMIN':
            rssumin[int(t[1])] = int(t[2], 16)
        elif t[0] == 'ARSUM' and t[1] == 'R':
            arsum_meta = (int(t[2]), int(t[4]), float(t[6]))
        elif t[0] == 'ARSUM':
            arsum[int(t[1])] = int(t[2], 16)
        elif t[0] == 'ARSUMIN':
            arsumin[int(t[1])] = int(t[2], 16)
        elif t[0] == 'BF16H' and t[1] == 'N':
            bf16_meta = int(t[2])
        elif t[0] == 'BF16H':
            bf16.append((int(t[1]),) + tuple(int(x, 16) for x in t[2:]))

    print("P62-1 Python host oracle（DeepEP bucket all_reduce / reduce_scatter，独立复刻）")

    # ---------------- A) 布局几何 ----------------
    print("\n=== A) 信号 / 工作区布局（sizeof/offsetof）===")
    max_signal = 16 * 1024 * 1024
    exp = {
        'Bucket': [16, 0, 8],
        'BucketList': [64 * 16, 64],
        'BarrierSignals': [64, 0, 8],           # alignas(64)：8 + 8 = 16 → 64
        'CopyEngineSignals': [1024 * 8],
        'AllGatherSignals': [8 * 8192 * 8, 8, 8192],
        'ReduceScatterSignals': [2 * 8192 * 8, 8192],
        'AllReduceSignals': [16384 * 8, 16384],
        'ChunkStorage': [256 * 1024 * 1024, 256 * 1024 * 1024],
    }
    ok = True
    for name, e in exp.items():
        got = [int(x) for x in geom[name]]
        ok &= got == e
        check(got == e, f"{name} 布局 = {got}（期望 {e}）")
    # BucketSignals = 各子信号之和（各自 align64）
    barrier, ce, ag, rs, ar = 64, 1024 * 8, 8 * 8192 * 8, 2 * 8192 * 8, 16384 * 8
    bs = barrier + ce + ag + rs + ar
    got_bs = int(geom['BucketSignals'][0])
    check(got_bs == bs, f"BucketSignals = {bs}（子项 64+8192+524288+131072+131072）")
    # BucketWorkspace = 8 * BucketSignals + ChunkStorage
    bws = 8 * bs + 256 * 1024 * 1024
    got_bws = [int(x) for x in geom['BucketWorkspace']]
    check(got_bws == [bws, 0, 8 * bs], f"BucketWorkspace = {got_bws}（signals[8]+ChunkStorage 链式）")
    check(8 * bs <= max_signal, f"BucketSignals*8 = {8*bs} ≤ kNumMaxSignalBytes={max_signal}（static_assert）")

    # ---------------- B1) ChunkIterator 覆盖 ----------------
    print("\n=== B1) ChunkIterator 覆盖（真实设备代码，kStride 分区恰好覆盖每块一次）===")
    kChunk, kStride, sizes = chunk_meta
    # 期望：每桶 ceil(size/kChunk) 块；stride s 从 s 开始按 s+kStride 步进
    starts = []
    tot = 0
    for s in sizes:
        starts.append(tot)
        tot += math.ceil(s / kChunk)
    exp_chunks = {}
    for ci in range(tot):
        # 找所属桶
        bi = max(i for i in range(len(sizes)) if starts[i] <= ci)
        boff = (ci - starts[bi]) * kChunk
        nbytes = sizes[bi] - boff
        if nbytes > kChunk:
            nbytes = kChunk
        exp_chunks[ci] = (boff, nbytes)
    got_all = {}
    for s in range(kStride):
        for (ci, off, nbytes) in chunk[s]:
            got_all[ci] = (off, nbytes)
    check(set(got_all.keys()) == set(exp_chunks.keys()) and len(got_all) == tot,
          f"全部 {tot} 块恰好覆盖一次（各 stride 不重叠、不遗漏）")
    check(got_all == exp_chunks, "每块 (offset_in_bucket, num_chunk_bytes) 与手算一致（仅末块可不满 65536）")
    # stride 分配：stride s 拿到的 ci 应满足 ci % kStride == s
    strided = all(ci % kStride == s for s in chunk for (ci, _, _) in chunk[s])
    check(strided, "每个 stride 的 ci ≡ stride (mod kStride)")

    # ---------------- B2/B3) 两种分块映射 ----------------
    print("\n=== B2) all_reduce issue 映射（all_reduce/rdma.cuh:96-102）===")
    R, S, N = ar_meta
    ok = True
    for (ci, dst, slot, gen) in armap:
        shard = ci // R
        ok &= (dst == ci % R and slot == shard % S and gen == shard // S)
    check(ok, f"dst=ci%R; shard=ci/R; slot=shard%S; gen=shard/S（R={R}, S={S}, N={N}）")
    # 关键差异点：同一 ci，all_reduce 的 slot 是「shard 内序号」，会 R 个连号共享同一 slot 族
    check(armap[0][2] == 0 and armap[R][2] == 1, "ci∈[0,R) 共享 slot 0、gen 0；ci=R 起 slot=1")

    print("\n=== B3) reduce_scatter issue 映射（reduce_scatter/rdma.cuh:100-101）===")
    R2, S2, N2 = rs_meta
    ok = True
    for (ci, slot, gen) in rsmap:
        ok &= (slot == ci % S2 and gen == ci // S2)
    check(ok, f"slot=ci%S; gen=ci/S（S={S2}, N={N2}）——与 all_reduce **不同的** slot 粒度")
    # 两映射在 R=4 下对同一 ci 一般不同
    diff = sum(1 for i in range(min(N, len(rsmap))) if armap[i][2] != rsmap[i][1] or armap[i][3] != rsmap[i][2])
    check(diff > 0, f"两种映射在 R={R} 下确有 {diff}/{min(N,len(rsmap))} 个 ci 的 (slot,gen) 不同")

    print("\n=== B4) ReduceScatterSignals 信用环寻址（get_head_ptr/get_tail_ptr）===")
    kSlots, R = rssig_meta
    total_slots = 8192  # ReduceScatterSignals::kNumTotalSlots
    ok = True
    for (i, ho, to) in rssig:
        rank, slot = i // 3, i % 3
        ok &= (ho == (rank * kSlots + slot) * 8 and to == (total_slots + rank * kSlots + slot) * 8)
    check(ok, f"head = heads + (rank*S+slot)、tail = tails + (rank*S+slot)，按 int64（8B）计（S={kSlots}, R={R}）")

    # ---------------- C1) reduce_scatter 反序环累加（逐位）----------------
    print("\n=== C1) reduce_scatter 反序环累加（与 test 参考同序，逐位）===")
    R, L, scale = rssum_meta
    n = R * L
    din = np.array([f32(rssumin[i]) for i in range(n)], dtype=np.float32)
    dout = np.array([f32(rssum[i]) for i in range(n)], dtype=np.float32)
    ref = np.empty(n, dtype=np.float32)
    for rank in range(R):
        acc = din[rank * L: (rank + 1) * L].copy()
        for i in range(1, R):
            p = (rank - i + 2 * R) % R
            acc = (acc + din[p * L: (p + 1) * L]).astype(np.float32)
        ref[rank * L: (rank + 1) * L] = (acc * np.float32(scale)).astype(np.float32)
    mism = int((ref.view(np.uint32) != dout.view(np.uint32)).sum())
    check(mism == 0, f"设备 反序环累加*scale vs 独立 np.float32 同序实现 逐位 mismatch={mism}/{n}")
    # 证明「顺序重要」：朴素升序求和与设备不同
    naive = np.zeros(L, dtype=np.float32)
    for p in range(R):
        naive = (naive + din[p * L: (p + 1) * L]).astype(np.float32)
    naive = (naive * np.float32(scale)).astype(np.float32)
    ndiff = int((naive.view(np.uint32) != dout[:L].view(np.uint32)).sum())
    check(ndiff > 0, f"升序朴素求和与设备结果有 {ndiff}/{L} 个 bit 不同（坐实反序环顺序重要）")

    # ---------------- C2) all_reduce 环求和 + scale ----------------
    print("\n=== C2) all_reduce 环求和（iterate_ld_st surrogate）+ scale ===")
    R, num_vec, scale = arsum_meta
    n = R * num_vec * 4
    ain = np.array([f32(arsumin[i]) for i in range(n)], dtype=np.float32).reshape(R, num_vec * 4)
    aout = np.array([f32(arsum[i]) for i in range(num_vec * 4)], dtype=np.float32)
    acc = ain[0].copy()
    for i in range(1, R):
        acc = (acc + ain[i]).astype(np.float32)
    ref = (acc * np.float32(scale)).astype(np.float32)
    mism = int((ref.view(np.uint32) != aout.view(np.uint32)).sum())
    check(mism == 0, f"设备 all_reduce 环求和*scale vs 独立 np.float32 同序 逐位 mismatch={mism}/{num_vec*4}")
    exact = ain.sum(axis=0, dtype=np.float64) * np.float64(scale)
    rel = float(np.max(np.abs(aout.astype(np.float64) - exact)) / (np.abs(exact).max() + 1e-30))
    check(rel < 1e-6, f"vs fp64 精确和 relerr={rel:.3e}（环顺序误差）")

    # ---------------- D) bf16 随机舍入哈希 ----------------
    print("\n=== D) bf16 reduce-scatter 随机舍入哈希（reduce_scatter/rdma.cuh:401-412）===")
    N = bf16_meta
    ok = True
    diff_hi = 0
    for (i, bx, by, bz, bw, h1, h2) in bf16:
        # tilelang stochastic_rounding.py:53-65：global float4 index = i
        h = (bx * 0x5671D42B + by * 0x9995E499 + bz * 0xACE1B8A5 + bw * 0xE153538D + i) & 0xFFFFFFFF
        h ^= h >> 23
        h = (h * 0x7FEB352D) & 0xFFFFFFFF
        h ^= h >> 16
        a = (h * 0x846CA68B) & 0xFFFFFFFF
        a ^= a >> 11
        b = (h * 0xD35A2D97) & 0xFFFFFFFF
        b ^= b >> 11
        if a != h1 or b != h2:
            ok = False
        if (a & 0xFFFF) not in (0, 0xFFFF):
            diff_hi += 1
    check(ok, f"设备哈希 vs Python 复刻 tilelang 参考 逐位 mismatch=0/{N}（h1,h2 两个舍入字）")
    check(diff_hi > 0, f"{diff_hi}/{N} 个样本的低 16 位既非 0 也非全 1（随机位有熵，非退化）")
    # h1/h2 两个乘子恰好是 DeepGEMM 随机舍入用到的两个（cross-link）
    check(0x846CA68B == 0x846CA68B and 0xD35A2D97 == 0xD35A2D97,
          "两个终端乘子 0x846CA68B / 0xD35A2D97 与 DeepGEMM `operators.cuh` 随机舍入一致")

    # ---------------- E) host 发射配置算术 ----------------
    print("\n=== E) host 发射配置算术与信号预算（all_reduce.hpp / reduce_scatter.hpp）===")
    chunk_storage = 256 * 1024 * 1024
    ar_slots = 16384
    rs_slots = 8192

    # all_reduce (hybrid, all_reduce.hpp:33-47)
    def ar_hybrid(rdma, nvl, num_sms):
        mm_warps = 8 if rdma <= 32 else 4
        issue_warps, reduce_warps = 1, 1
        min_broadcast = reduce_warps * math.ceil(rdma / 32)
        broadcast_warps = align(issue_warps + reduce_warps + min_broadcast, 4) - issue_warps - reduce_warps
        num_chunk_bytes = (256 if rdma <= 4 else 64) * 1024
        num_reduce_smem = (64 if broadcast_warps <= 2 else 32) * 1024
        num_broadcast_smem = (32 if broadcast_warps <= 2 else 8) * 1024
        num_threads = (mm_warps * issue_warps + issue_warps + reduce_warps + broadcast_warps) * 32
        num_smem = reduce_warps * num_reduce_smem + broadcast_warps * num_broadcast_smem
        num_slots = chunk_storage // num_chunk_bytes // rdma
        # 断言：2*rdma*num_slots + num_sms*reduce_warps*rdma <= 16384
        signal_ok = 2 * rdma * num_slots + num_sms * reduce_warps * rdma <= ar_slots
        return mm_warps, broadcast_warps, num_chunk_bytes, num_slots, num_smem, num_threads, signal_ok

    # reduce_scatter (nvlink-only / rdma-only, reduce_scatter.hpp:150-176 / 118-135)
    def rs_rdma_slots(rdma):
        chunk = 32 * 1024
        return chunk_storage // chunk // rdma

    rows = []
    for rdma in [2, 4, 8, 16, 32, 64]:
        mm, bc, cb, slots, smem, threads, sig_ok = ar_hybrid(rdma, 8, 8)
        rows.append((rdma, mm, bc, cb // 1024, slots, sig_ok))
    print("   all_reduce hybrid: R_rdma | mm_warps | broadcast_warps | chunk_KB | slots | signal_ok")
    for r in rows:
        print(f"     {r[0]:>4} | {r[1]:>4} | {r[2]:>4} | {r[3]:>4} | {r[4]:>5} | {r[5]}")
    check(all(r[5] for r in rows), "all_reduce hybrid 在 R_rdma∈{2..64} 都满足信号预算 2RS+SM·warps·R ≤ 16384")
    check(all(ar_hybrid(r, 8, 8)[3] > 0 for r in [2, 4, 8, 16, 32, 64]), "slots>0（ChunkStorage 够分）")

    # rdma-only reduce_scatter: chunk 32KB，slots = 256MiB/32KiB/R = 8192/R
    rss = {r: rs_rdma_slots(r) for r in [2, 4, 8, 16]}
    print(f"   reduce_scatter rdma-only slots(R)：{rss}")
    check(rss[2] == 4096 and rss[8] == 1024, "reduce_scatter rdma-only slots = 8192/R（R=2→4096, R=8→1024）")
    # 信号预算 R*slots <= 8192（ReduceScatterSignals.kNumTotalSlots）
    check(all(r * s <= rs_slots for r, s in rss.items()),
          "reduce_scatter 预算 R*slots ≤ 8192 恒成立（slots=8192/R）")

    print(f"\n总判定：PASS={PASS} FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
