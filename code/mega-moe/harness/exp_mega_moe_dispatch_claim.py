#!/usr/bin/env python3
# P184-1 实验：Mega MoE dispatch「round-robin min-peeling」认领协议的独立 host/Python 复刻。
#
# 与 src/mega_moe_dispatch_claim.cu 相互独立：这里**不模拟 lane / __ballot_sync / __fns**，
# 而是按「每轮活跃 rank 各出 L=min(remaining) 个、轮内 round-robin 摊、t=offset+slot//nactive」
# 的聚合语义，**逐轮构造整条置换**（一次性建表），再与一个「逐 token while 循环」oracle 对拍。
# 两条独立实现 + 真实设备核（.cu）三方一致才说明源码语义被真正钉住。
#
# python3 -u exp_mega_moe_dispatch_claim.py

import random
from collections import Counter

PASS = 0
FAIL = 0


def chk(ok, msg):
    global PASS, FAIL
    print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def peel_permutation(counts):
    """逐轮构造整条置换：token 顺序 = 各轮内 round-robin（活跃 rank 升序）。"""
    R = len(counts)
    rem = list(counts)
    off = 0
    out = []
    while True:
        active = [r for r in range(R) if rem[r] > 0]
        if not active:
            break
        L = min(rem[r] for r in active)
        for j in range(L):
            for r in active:           # 轮内：活跃 rank 升序轮流
                out.append((r, off + j))
        for r in active:
            rem[r] -= L
        off += L
    return out


def claim_while(counts, token_idx):
    """逐 token while 循环 oracle（与 .cu 的 host oracle 同语义、独立代码路径）。"""
    R = len(counts)
    rem = list(counts)
    off = 0
    slot = token_idx
    while True:
        active = [r for r in range(R) if rem[r] > 0]
        nactive = len(active)
        L = min(rem[r] for r in active)
        nround = L * nactive
        if slot < nround:
            return (active[slot % nactive], off + slot // nactive)
        slot -= nround
        off += L
        for r in active:
            rem[r] -= L


def check_case(counts, tag, verbose=False):
    R = len(counts)
    total = sum(counts)
    perm = peel_permutation(counts)
    bad = 0
    # 1) 长度
    if len(perm) != total:
        bad += 1
    # 2) 逐 token：置换 vs while-oracle
    for i, (r, t) in enumerate(perm):
        w = claim_while(counts, i)
        if w != (r, t):
            bad += 1
            if bad <= 3:
                print(f"      perm[{i}]={r,t} while={w}")
    # 3) 双射
    want = Counter((r, t) for r in range(R) for t in range(counts[r]))
    got = Counter(perm)
    if want != got:
        bad += 1
    # 4) 频率
    freq = Counter(r for r, _ in perm)
    if any(freq[r] != counts[r] for r in range(R)):
        bad += 1
    chk(bad == 0, f"{tag}: 置换==while-oracle + 双射 + 频率 (total={total}, bad={bad})")
    if verbose:
        print(f"    counts={counts}")
        print(f"    rank 序列[前64]={[r for r, _ in perm[:64]]}")
        print(f"    前16 (rank,t)={perm[:16]}")
    return bad


def plain_map(counts, i):
    p = 0
    for r, c in enumerate(counts):
        if i < p + c:
            return (r, i - p)
        p += c
    return (0, 0)


def interleave(counts, perm, num_warps, tag):
    """每 warp 覆盖的源 rank 数 + 最长同 rank 连跑：min-peeling vs 朴素连续切段。"""
    total = len(perm)
    def stats(use_peel):
        distinct_sum = 0
        maxrun_sum = 0
        gmax = 0
        warps = 0
        for w in range(num_warps):
            last = None
            run = 0
            mrun = 0
            seen = set()
            for i in range(w, total, num_warps):
                r = perm[i][0] if use_peel else plain_map(counts, i)[0]
                seen.add(r)
                run = run + 1 if r == last else 1
                mrun = max(mrun, run)
                last = r
            distinct_sum += len(seen)
            maxrun_sum += mrun
            gmax = max(gmax, mrun)
            warps += 1
        return distinct_sum / warps, maxrun_sum / warps, gmax
    pd, pr, pmax = stats(True)
    cd, cr, cmax = stats(False)
    print(f"  {tag} (num_warps={num_warps}):")
    print(f"    [每 warp 视角] min-peeling : 平均覆盖 {pd:.1f} 个源 rank / 平均最长同 rank 连跑 {pr:.1f} / 全局最长 {pmax}")
    print(f"    [每 warp 视角] 朴素连续切段: 平均覆盖 {cd:.1f} 个源 rank / 平均最长同 rank 连跑 {cr:.1f} / 全局最长 {cmax}")

    # 并发窗口：第 k 次迭代 528 个 warp 处理 [k*W,(k+1)*W) 连续 token（物理相关量）
    def window(use_peel):
        ds = rs = 0
        gmax = 0
        single = windows = 0
        for base in range(0, total, num_warps):
            seen = set()
            last = None
            run = mrun = 0
            for i in range(base, min(base + num_warps, total)):
                r = perm[i][0] if use_peel else plain_map(counts, i)[0]
                seen.add(r)
                run = run + 1 if r == last else 1
                mrun = max(mrun, run)
                last = r
            ds += len(seen)
            rs += mrun
            gmax = max(gmax, mrun)
            if len(seen) == 1:
                single += 1
            windows += 1
        return ds / windows, rs / windows, gmax, single, windows
    wd, wr, wmax, wsingle, wn = window(True)
    zd, zr, zmax, zsingle, zn = window(False)
    print(f"    [并发窗口视角] min-peeling : 每窗口平均覆盖 {wd:.1f} 个源 rank / 平均最长连跑 {wr:.1f} / 全局最长 {wmax} / 单 rank 窗口 {wsingle}/{wn}")
    print(f"    [并发窗口视角] 朴素连续切段: 每窗口平均覆盖 {zd:.1f} 个源 rank / 平均最长连跑 {zr:.1f} / 全局最长 {zmax} / 单 rank 窗口 {zsingle}/{zn}")
    return pd, cd


def main():
    print("== P184-1 min-peeling 独立 Python 复刻 ==")

    # 显式用例
    check_case([10, 6, 0, 3, 8, 1, 0, 4], "R=8 含零+偏斜", verbose=True)
    check_case([7] * 8, "R=8 均匀")
    check_case([100, 0, 0, 0, 0, 0, 0, 0], "R=8 单 rank")
    check_case([1] * 8, "R=8 全 1")
    check_case([12, 0, 5, 3], "R=4")
    check_case([9, 0, 4, 7, 2, 0, 6, 3, 8, 1, 5, 0, 2, 9, 4, 6], "R=16")
    check_case([3] * 72, "R=72")

    # 随机 sweep（与 .cu 相同 R 集合）
    print("\n[sweep] 500 组随机 (R, counts)")
    random.seed(12345)
    bad = 0
    for _ in range(500):
        R = random.choice([8, 16, 32, 64, 72])
        skew = random.random() < 0.5
        c = [random.randrange(40) if skew else random.randrange(8) for _ in range(R)]
        if random.random() < 0.2:
            c[random.randrange(R)] = 0
        if sum(c) == 0:
            c[0] = 1
        bad += 0 if _quiet_case(c) == 0 else 1
    chk(bad == 0, f"随机 sweep 500 组全部正确 (bad={bad})")

    # 交错性
    print("\n[interleave] min-peeling vs 朴素连续切段")
    counts = [4096, 3072, 0, 1024, 5120, 2048, 256, 1280]
    perm = peel_permutation(counts)
    interleave(counts, perm, 528, "counts=[4096,3072,0,1024,5120,2048,256,1280]")
    # 单 warp（串行）视角：整个 rank 序列
    interleave(counts, perm, 1, "单 warp 串行视角")

    print(f"\n== 判定: PASS={PASS} FAIL={FAIL} ==")
    return 0 if FAIL == 0 else 1


def _quiet_case(counts):
    # 静默版 check_case（不打印 OK 行），返回 bad 计数
    global PASS, FAIL
    R = len(counts)
    perm = peel_permutation(counts)
    bad = 0
    if len(perm) != sum(counts):
        bad += 1
    for i, (r, t) in enumerate(perm):
        if claim_while(counts, i) != (r, t):
            bad += 1
    want = Counter((r, t) for r in range(R) for t in range(counts[r]))
    if Counter(perm) != want:
        bad += 1
    if bad:
        FAIL += 1
        print(f"      sweep FAIL counts={counts}")
    else:
        PASS += 1
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
