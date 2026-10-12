#!/usr/bin/env python3
"""P51-1：独立 host oracle，复刻 DeepEP dispatch_copy_epilogue 的纯公式部分。

- TokenLayout 的字段偏移与字节数（layout/ep/token.cuh:32-82）；
- BufferLayout 的每 rank / 总量（:102-143）；
- expand 的「按 kExpertAlignment 对齐的每专家区」基址（host 侧 exclusive 前缀和，
  csrc/buffers/ep.hpp:711-721）与逐专家计数；
- `dispatch_copy_epilogue.cuh:232-325` 零填充 dst 的算术（warp_inclusive/exclusive_sum +
  reduce_add 在单个 warp 内的语义复刻），证：padding 行落在 base_e+count_e .. base_e+align(count_e)。

CUDA 端已用真实设备代码逐位验证；本脚本只独立复核公式，不与 GPU 通信。
"""
import numpy as np

K_HIDDEN = 128
K_SFPACKS = 4
K_TOPK = 4
SF_BYTES = K_SFPACKS * 4
WITH_META = True
ALIGN_TMA = 32
ALIGN_RDMA = 64
MBAR_BYTES = 8  # sizeof(ptx::mbarrier) 在本机为 8


def align(x, a):
    return (x + a - 1) // a * a


def token_layout_offsets():
    hidden_off = 0
    sf_off = align(K_HIDDEN, ALIGN_TMA)
    meta_off = sf_off + align(SF_BYTES, ALIGN_TMA)
    num_metadata = K_TOPK * (4 + 4) + ((1 + K_TOPK) * 4 if WITH_META else 0)
    topk_idx_off = meta_off
    weights_off = topk_idx_off + K_TOPK * 4
    src_off = weights_off + K_TOPK * 4
    ll_off = src_off + 4
    data_bytes = align(align(K_HIDDEN, ALIGN_TMA) + align(SF_BYTES, ALIGN_TMA)
                       + align(num_metadata, ALIGN_TMA), ALIGN_RDMA)
    smem_bytes = data_bytes + align(MBAR_BYTES, ALIGN_TMA)
    return dict(hidden=hidden_off, sf=sf_off, metadata=meta_off, topk_idx=topk_idx_off,
                weights=weights_off, src_global=src_off, linked_list=ll_off,
                num_bytes_gmem=data_bytes, num_bytes_smem=smem_bytes,
                num_metadata=num_metadata)


def warp_reduce_add(vals):
    return int(np.sum(vals))


def warp_inclusive_sum(vals):
    return list(np.cumsum(vals))


def warp_exclusive_sum(vals):
    return list(np.cumsum(vals) - np.array(vals))


def zero_pad_dst(counts, lane_experts, alignment):
    """复刻 cuh:254-287 的 pad dst 计算（整个 warp 一次 wave 覆盖所有 lane_expert）。"""
    n = len(lane_experts)
    counts_l = [counts[e] if e is not None else 0 for e in lane_experts]
    pads_l = [0 if e is None else (alignment - counts[e] % alignment) % alignment for e in lane_experts]
    wave_num_pads = warp_reduce_add(pads_l)
    total_rows = sum(c + p for c, p in zip(counts_l, pads_l))
    # host exclusive 基址（对齐块）
    base = 0
    bases = {}
    for e in sorted(set(x for x in lane_experts if x is not None)):
        bases[e] = base
        base += align(counts[e], alignment)
    res = []
    for local_pad in range(wave_num_pads):
        pad_psum = warp_inclusive_sum(pads_l)
        owner = min(i for i in range(n) if pad_psum[i] > local_pad)  # __ffs 取最低 set lane
        ex = warp_exclusive_sum([c + p for c, p in zip(counts_l, pads_l)])
        dst = (ex[owner] + counts_l[owner] + local_pad
               - (pad_psum[owner] - pads_l[owner]))
        res.append((lane_experts[owner], dst))
    return total_rows, bases, res


def main():
    ok = True
    print("=== TokenLayout 偏移 vs C++ 实测 ===")
    o = token_layout_offsets()
    expect = dict(hidden=0, sf=128, metadata=160, topk_idx=160, weights=176,
                  src_global=192, linked_list=196)
    for k, v in expect.items():
        good = o[k] == v
        ok &= good
        print(f"  {k:12s} off={o[k]:4d}  expect={v:4d}  {'OK' if good else 'BAD'}")
    print(f"  num_bytes<false>={o['num_bytes_gmem']} (expect 256), "
          f"<true>={o['num_bytes_smem']} (expect 288)")
    ok &= (o['num_bytes_gmem'] == 256 and o['num_bytes_smem'] == 288)

    print("=== BufferLayout（单 rank×8 token）===")
    bl_per_rank = o['num_bytes_gmem'] * 8
    bl_total = bl_per_rank * 1
    print(f"  per_token={o['num_bytes_gmem']} per_rank={bl_per_rank} total={bl_total}")
    ok &= (bl_per_rank == 2048 and bl_total == 2048)

    print("=== expand 对齐分配 + 零填充 dst（复刻 cuh:232-325）===")
    # 与 C++ 相同的 topk 表：counts=[4,3,3,4]，alignment=4
    counts = {0: 4, 1: 3, 2: 3, 3: 4}
    lane_experts = [0, 1, 2, 3]  # kNumExpertsPerRank=4，一 lane 一专家
    total_rows, bases, pads = zero_pad_dst(counts, lane_experts, 4)
    print(f"  counts={[counts[e] for e in lane_experts]} base={[bases[e] for e in lane_experts]} total_rows={total_rows}")
    ok &= (total_rows == 16 and [bases[e] for e in lane_experts] == [0, 4, 8, 12])
    # 期望 padding dst：专家 1（count=3）→ base=4 → 行 7；专家 2（count=3）→ base=8 → 行 11
    exp_pads = sorted(pads)
    print(f"  padding (expert,dst)={exp_pads}")
    got = sorted((e, d) for e, d in pads)
    exp = [(1, 7), (2, 11)]
    ok &= (got == exp)
    print(f"  期望 {exp}  {'OK' if got == exp else 'BAD'}")

    print("\n==== %s ====" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
