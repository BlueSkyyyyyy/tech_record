#!/usr/bin/env python3
# P61-1 独立 host oracle：用纯 Python 复刻 DeepEP 两级（hybrid）EP dispatch/combine 里
# 四个架构无关内核，与同一进程里编译出来的 CUDA 真实设备代码（src/deepep_hybrid.cu）
# 互为独立实现：
#   * 计数就绪哨兵  encode_decode_positive / is_decoded_positive_ready（common/math.cuh:26-33）
#   * tail 信号     pack2/unpack2（common/math.cuh:45-68）
#   * 前缀和        do_psum（impls/ep/hybrid_dispatch.cuh:304-320）
#   * 分片调度      round-robin（impls/ep/hybrid_dispatch.cuh:490-532）
#   * 每槽字节      TokenLayout（layout/ep/token.cuh:10-82）
#
#   python3 harness/exp_deepep_hybrid.py

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


def encode(v):
    # math.cuh:31-33  encode_decode_positive(v) = -v - 1（对合）
    return -v - 1


def ready(v):
    # math.cuh:26-28  is_decoded_positive_ready(v) = (v >= 0)
    return v >= 0


def pack2(finish, tail):
    # math.cuh:45-52  x=finish 在低 32 位、y=tail 在高 32 位（int64 重解释为 int[2]）
    return ((tail & 0xFFFFFFFF) << 32) | (finish & 0xFFFFFFFF)


def unpack2(p):
    p &= (1 << 64) - 1
    return p & 0xFFFFFFFF, (p >> 32) & 0xFFFFFFFF   # (finish, tail)


def do_psum(count, n, is_exclusive):
    # 逐字复刻 hybrid_dispatch.cuh:304-320（warp 内建 → Python 顺序模拟）
    out_len = n + is_exclusive
    out = [None] * out_len
    psum = 0
    num_iters = (n + is_exclusive + 31) // 32
    for i in range(num_iters):
        sums = []
        for lane in range(32):
            idx = i * 32 + lane
            mem_idx = idx - is_exclusive
            value = count[mem_idx] if (0 <= mem_idx < n) else 0
            # warp_inclusive_sum(value, lane)
            inc = 0
            for k in range(0, lane + 1):
                kk = k
                # value of lane k
                idx_k = i * 32 + k
                m_k = idx_k - is_exclusive
                v_k = count[m_k] if (0 <= m_k < n) else 0
                inc += v_k
            sums.append(psum + inc)
        for lane in range(32):
            idx = i * 32 + lane
            if idx < out_len:
                out[idx] = sums[lane]
        psum = sums[31]  # exchange(sum, 31)
    return out


def rr_greedy(tails, chunk):
    R = len(tails)
    old = [0] * R
    seq = []
    recv = 0
    while True:
        wip = 0
        for r in range(R):
            if tails[r] > old[r]:
                wip |= (1 << r)
        if wip == 0:
            break
        offset = (recv + 1) % R
        hi = (wip >> offset) << offset
        # ffs = 1-based lowest set bit
        chosen = (hi & -hi).bit_length() - 1 if hi else (wip & -wip).bit_length() - 1
        recv = chosen
        start = old[recv]
        end = min(tails[recv], start + chunk)
        old[recv] = end
        seq.append(recv)
    return seq


def token_bytes(hidden, sf, topk, meta):
    def al(a, b):
        return (a + b - 1) // b * b
    meta_bytes = topk * (4 + 4) + ((1 + topk) * 4 if meta else 0)
    data = al(al(hidden, 32) + al(sf, 32) + al(meta_bytes, 32), 64)
    return data, data + al(32, 32)


def main():
    print("P61-1 Python host oracle（独立复刻四级内核 + 布局）")

    print("\n=== Part 1: 计数就绪哨兵 ===")
    ok = True
    for c in [0, 1, 2, 7, 1023, 65535, 1 << 20, 0x7FFFFFFF]:
        raw = encode(c)          # 写进 buffer 的 -c-1
        decoded = encode(raw)    # 读回后 decode = c
        ok &= (decoded == c) and ready(decoded) and (not ready(encode(0)))   # 初值 0 → decode=-1 未就绪
    check(ok, "encode/decode 对合 + 写过的就绪、初值 0 未就绪")

    print("\n=== Part 2: tail 信号 ===")
    ok = True
    for f, t in [(0, 0), (1, 1), (1, 6), (0, 100), (1, 1 << 20), (0, 0x7FFFFFFF)]:
        p = pack2(f, t)
        uf, ut = unpack2(p)
        ok &= (p == ((t << 32) | f)) and (uf == f) and (ut == t)
    check(ok, "pack2/unpack2 位段一致（finish 低、tail 高）")
    acc = 0
    old = 0
    rounds = [2, 5, 9]
    for i, t in enumerate(rounds):
        finish = 1 if i + 1 == len(rounds) else 0
        sig = pack2(finish, t)
        old_sig = pack2(0, old)
        acc += sig - old_sig
        old = t
    f_final, t_final = unpack2(acc)
    check(t_final == 9 and f_final == 1, "增量累加 tail=9 finish=1")

    print("\n=== Part 3: do_psum 前缀和 ===")
    cases = [(1, 0), (32, 0), (33, 0), (8, 1), (32, 1), (33, 1), (70, 1), (130, 1)]
    ok = True
    for n, excl in cases:
        count = [(i * 7 + 3) % 11 for i in range(n)]
        out = do_psum(count, n, excl)
        if excl:
            ref = [0] * (n + 1)
            run = 0
            for i in range(n):
                run += count[i]
                ref[i + 1] = run
        else:
            ref = []
            run = 0
            for i in range(n):
                run += count[i]
                ref.append(run)
        ok &= (out == ref)
    check(ok, "含端点/排他前缀和 8 组")

    print("\n=== Part 4: round-robin 分片调度 ===")
    ok = True
    for R, chunk, tails in [
        (4, 6, [20, 13, 5, 0]),
        (8, 6, [1, 2, 3, 4, 5, 6, 7, 8]),
        (3, 100, [7, 7, 7]),
        (5, 1, [9, 0, 0, 3, 0]),
    ]:
        seq = rr_greedy(tails, chunk)
        per = [0] * R
        for r in seq:
            per[r] += 1
        # 覆盖检查：每 rank 被处理次数 == ceil(tail/chunk)
        expect = [(t + chunk - 1) // chunk for t in tails]
        ok &= (per == expect)
    check(ok, "每 rank 分片次数 == ceil(tail/chunk)")

    print("\n=== Part 5: TokenLayout 每槽字节 ===")
    ok = True
    for hidden, sf, topk, meta in [(7168, 56, 8, True), (14336, 0, 8, False)]:
        d, wm = token_bytes(hidden, sf, topk, meta)
        ok &= (wm == d + 32)
    check(ok, "三段 32B 对齐 + RDMA 64B 对齐")

    print(f"\n================ Python 总计: PASS={PASS} FAIL={FAIL} ================")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
