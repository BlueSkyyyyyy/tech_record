#!/usr/bin/env python3
# P48-1 独立 host oracle：用纯 Python 复刻 DeepEP combine 的「去重 + rank/topk 双布局」选择逻辑
# （combine_utils.cuh 的 use_rank_layout / get_num_tokens_in_layout / get_max_unroll_factor，
#  combine_reduce_epilogue.cuh:73-95 的 dedup + compute_topk_slots），
# 与同一进程里编译出来的 CUDA 真实设备代码（src/deepep_combine_reduce.cu）互为独立实现。
#
#   python harness/exp_deepep_combine_reduce.py


def use_rank_layout(allow_multiple_reduction, num_ranks, num_topk):
    if not allow_multiple_reduction:
        return False
    return num_ranks <= num_topk


def get_num_tokens_in_layout(allow_multiple_reduction, num_ranks, num_topk):
    return num_ranks if use_rank_layout(allow_multiple_reduction, num_ranks, num_topk) else num_topk


def get_max_unroll_factor(length, max_unroll, warp=32):
    for i in range(max_unroll, 0, -1):
        if length % (warp * i) == 0:
            return i
    raise ValueError("no unroll factor")


def ffs(mask):
    # 1-based lowest set bit, 0 if none（对应 __ffs）
    if mask == 0:
        return 0
    return (mask & -mask).bit_length()


def compute_topk_slots(mask, num_valid_topk, fetch):
    # 逐位从最低到最高，对应 combine_utils.cuh:41-53
    slots = []
    for _ in range(num_valid_topk):
        low = ffs(mask) - 1
        fetched = fetch(low)
        mask &= mask - 1
        slots.append(fetched if low >= 0 else -1)
    return slots


def master_lane(rank_of_lane, num_ranks):
    # 每个 rank 的最高 lane（get_master_lane_idx == 31-__clz）
    m = [-1] * num_ranks
    for lane, r in enumerate(rank_of_lane):
        if r >= 0:
            m[r] = max(m[r], lane)
    return m


def model(rank_of_lane, num_ranks, num_topk, use_rank_layout_bool):
    master = master_lane(rank_of_lane, num_ranks)
    # deduplicate key = rank；master lane 保留 -> mask
    mask = 0
    for lane, r in enumerate(rank_of_lane):
        if r >= 0 and master[r] == lane:
            mask |= (1 << lane)
    # fetch: rank layout 取 rank，否则取 lane idx
    def fetch(idx):
        return rank_of_lane[idx] if use_rank_layout_bool else idx
    slots = compute_topk_slots(mask, num_topk if not use_rank_layout_bool else num_ranks, fetch)
    distinct = sorted({r for r in rank_of_lane if r >= 0})
    return slots, distinct


def main():
    print("=== host constexpr 公式（combine_utils.cuh:8-26）===")
    ok = True
    for mult in (False, True):
        for ranks, topk in [(1, 8), (2, 8), (4, 4), (8, 4), (8, 2), (16, 8)]:
            rl = use_rank_layout(mult, ranks, topk)
            ntl = get_num_tokens_in_layout(mult, ranks, topk)
            exp_rl = mult and ranks <= topk
            good = (rl == exp_rl) and (ntl == (ranks if exp_rl else topk))
            ok &= good
            print(f"  mult={str(mult):5} ranks={ranks:3} topk={topk:2} -> "
                  f"use_rank_layout={str(rl):5} num_tokens_in_layout={ntl} {'OK' if good else 'BAD'}")
    for length, mu, exp in [(64, 4, 2), (256, 4, 4), (512, 4, 4), (32, 4, 1)]:
        got = get_max_unroll_factor(length, mu)
        ok &= (got == exp)
        print(f"  get_max_unroll_factor<{length},{mu}> = {got} (expect {exp}) {'OK' if got==exp else 'BAD'}")
    print("  PART-A", "PASS" if ok else "FAIL")

    print("\n=== dedup + 双布局选择逻辑（combine_reduce_epilogue.cuh:73-95）===")
    # 配置 A：rank layout。kNumExperts=32, per rank=8, ranks=4, topk=4
    print("\n  -- A rank-layout (ranks=4, topk=4, use_rank_layout=True) --")
    EPR = 8
    casesA = {
        0: [0, 3, 16, 24],      # ranks 0,0,2,3
        1: [8, 9, 10, 11],      # ranks 1,1,1,1
        2: [0, 8, 16, 24],      # ranks 0,1,2,3
        3: [24, -1, -1, -1],    # ranks 3
        4: [16, 24, 17, 25],    # ranks 2,3,2,3
    }
    okA = True
    for t, exps in casesA.items():
        rol = [e // EPR if e >= 0 else -1 for e in exps]
        slots, distinct = model(rol, 4, 4, True)
        expect_slots = distinct + [-1] * (4 - len(distinct))
        good = (slots == expect_slots)
        okA &= good
        print(f"    t={t}: experts={exps} ranks={rol} -> slots={slots} distinct={distinct} {'OK' if good else 'BAD'}")
    print("    PART-A", "PASS" if okA else "FAIL")

    # 配置 B：topk layout。kNumExperts=64, per rank=8, ranks=8, topk=4
    print("\n  -- B topk-layout (ranks=8, topk=4, use_rank_layout=False) --")
    casesB = {
        0: [0, 8, 16, 24],      # ranks 0,1,2,3
        1: [0, 1, 8, 16],       # ranks 0,0,1,2
        2: [8, 9, 10, 11],      # ranks 1,1,1,1
        3: [24, -1, 8, -1],     # ranks 3,-,1,-
        4: [0, 8, 16, 24],      # ranks 0,1,2,3
    }
    okB = True
    for t, exps in casesB.items():
        rol = [e // EPR if e >= 0 else -1 for e in exps]
        slots, distinct = model(rol, 8, 4, False)
        # topk layout 下 slot = master lane；distinct 数 == 非 -1 slot 数
        nvalid = sum(1 for s in slots if s >= 0)
        good = (nvalid == len(distinct))
        okB &= good
        print(f"    t={t}: experts={exps} ranks={rol} -> slots={slots} distinct_ranks={distinct} {'OK' if good else 'BAD'}")
    print("    PART-B", "PASS" if okB else "FAIL")

    # master lane 展示（token 1：ranks 0,0,1,2 -> master {0:1,1:2,2:3}）
    rol = [0, 0, 1, 2]
    print(f"\n  master_lane(ranks={rol}) = {master_lane(rol, 8)}（rank0->lane1, rank1->lane2, rank2->lane3）")

    total_ok = ok and okA and okB
    print("\n=== Python oracle:", "ALL PASS ===" if total_ok else "FAIL ===")


if __name__ == "__main__":
    main()
