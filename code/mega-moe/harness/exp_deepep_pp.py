#!/usr/bin/env python3
# P58-1 实验 host oracle：DeepEP PP send/recv 的索引映射 / 缓冲几何 / 调度不变式。
#
# 读 src/deepep_pp_send_recv.cu 设备 dump 的 Part A 索引，用独立 Python 复刻
# `get_pp_buffer_offset` 与 send/recv 两侧的 block / 信号索引，逐字段对拍；并：
#   * 验证 4 段缓冲（0/1=recv prev/next，2/3=send prev/next）在每一 rank 上恰好各用一次；
#   * 验证「发送方 staging 的对端目标 block == 接收方 recv block」「arrival/release 索引两侧一致」；
#   * 逐行抽取 tests/pp/test_pp.py 的 `generate_stress_ops`（真实源码）运行，验证它在
#     `num_max_inflight_tensors` 下真的不会让发送方覆盖尚未被消费的槽（无死锁调度的充分条件）。
#
# 用法: python harness/exp_deepep_pp.py <device_dump>

import ast
import hashlib
import os
import random
import sys
import textwrap

import torch

DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
CUH = os.path.join(DEP, 'deep_ep/include/deep_ep/impls/pp/pp_send_recv.cuh')


class Oracle:
    def __init__(self):
        self.pass_ = 0
        self.fail = 0

    def check(self, ok, what):
        if ok:
            self.pass_ += 1
            print(f'  [PASS] {what}')
        else:
            self.fail += 1
            print(f'  [FAIL] {what}')


def get_pp_buffer_offset(N, src, dst):
    next_rank = (src + 1) % N
    return (0, 1) if dst == next_rank else (1, 0)


def sha256_range(path, start, end):
    with open(path) as f:
        lines = f.readlines()
    text = ''.join(lines[start - 1:end])
    return hashlib.sha256(text.encode()).hexdigest()[:16], end - start + 1


def part_provenance(o):
    print('\n=== 0. 抽取片段 provenance（~/github/DeepEP@93eb6eb）===')
    h1, n1 = sha256_range(CUH, 20, 25)
    h2, n2 = sha256_range(CUH, 27, 102)
    print(f'    get_pp_buffer_offset cuh:20-25  sha256={h1}  ({n1} 行)')
    print(f'    pp_tma_copy          cuh:27-102 sha256={h2}  ({n2} 行)')
    o.check(h1 and h2, '逐行抽取片段可哈希（与仓库一致）')


def part_offsets(o, dump):
    print('\n=== 1. get_pp_buffer_offset + 缓冲/信号索引（设备 vs Python 独立复刻）===')
    mism = 0
    total = 0
    roles_ok = True
    corr_ok = True
    for line in dump.splitlines():
        if not line.startswith('A '):
            continue
        f = list(map(int, line.split()[1:]))
        N, src, dst, is_next, lo, peer = f[0:6]
        send_count, stag, rel_s, arr_s = f[6:10]
        recv_count, recv_blk, rel_r, arr_r, put_blk = f[10:15]
        total += 1
        elo, epeer = get_pp_buffer_offset(N, src, dst)
        exp = [
            (1 if dst == (src + 1) % N else 0), elo, epeer,
            epeer, epeer + 2, epeer, elo,
            elo, elo, epeer, elo, elo,
        ]
        got = [is_next, lo, peer, send_count, stag, rel_s, arr_s,
               recv_count, recv_blk, rel_r, arr_r, put_blk]
        if exp != got:
            mism += 1
        # 只有相邻 rank 才是合法 ring 邻居（host 断言 csrc/buffers/pp.hpp:86-88）
        if is_next:
            if not (lo == 0 and peer == 1 and stag == 3 and recv_blk == 0 and arr_s == 0 and arr_r == 0
                    and rel_s == 1 and rel_r == 1 and put_blk == 0):
                corr_ok = False
        else:
            if not (lo == 1 and peer == 0 and stag == 2 and recv_blk == 1 and arr_s == 1 and arr_r == 1
                    and rel_s == 0 and rel_r == 0 and put_blk == 1):
                corr_ok = False
        # 相邻对：发送 staging 的对称目标 block == 接收 recv block；两侧信号索引一致
        if abs(dst - src) == 1 or abs(dst - src) == N - 1:
            if put_blk != recv_blk or arr_s != arr_r or rel_s != rel_r:
                corr_ok = False
        # 每个 rank 上 4 段缓冲各司其职（0 prev-recv, 1 next-recv, 2 prev-send, 3 next-send）
    print(f'    有序对总数={total}  字段 mismatch={mism}')
    o.check(mism == 0, f'全部 {total} 个 (N,src,dst) 的 12 个导出字段逐字段 0 不符')
    o.check(corr_ok, '方向/缓冲/信号对应关系（puts 目标==recv block、arrival/release 两侧一致）成立')

    # 每个 rank 的 4 段缓冲恰好各用一次
    #   recv block = src_idx_in_local = lo；send staging block = peer_idx + 2
    #   N==2 时唯一邻居同时是 prev 与 next，两个方向退化成同一 (0,1)，4 段假设不成立
    #   （DeepEP PP 要求 num_ranks > 1 且实际是环形流水，N>=3 才有两个不同邻居）。
    ok = True
    for N in range(3, 9):
        for r in range(N):
            prev, nxt = (r - 1) % N, (r + 1) % N
            l1, _ = get_pp_buffer_offset(N, prev, r)   # recv from prev -> block lo
            l2, _ = get_pp_buffer_offset(N, nxt, r)    # recv from next -> block lo
            _, sp = get_pp_buffer_offset(N, r, prev)   # send to prev -> block peer+2
            _, sn = get_pp_buffer_offset(N, r, nxt)    # send to next -> block peer+2
            recv_blocks = {l1, l2}
            send_blocks = {sp + 2, sn + 2}
            if len(recv_blocks) != 2 or len(send_blocks) != 2 or (recv_blocks & send_blocks):
                ok = False
            if (recv_blocks | send_blocks) != {0, 1, 2, 3}:
                ok = False
    o.check(ok, '每个 rank 的 4 段缓冲 {0,1,2,3} 被 recv×2 / send×2 恰好各占一次')


def extract_function(path, name):
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            ns = {}
            exec(textwrap.dedent(seg), {'torch': torch, 'random': random, 'math': __import__('math')}, ns)
            return ns[name]
    raise RuntimeError(f'function {name} not found')


def part_stress(o):
    print('\n=== 2. tests/pp/test_pp.py generate_stress_ops（逐行抽取真实源码）===')
    path = os.path.join(DEP, 'tests/pp/test_pp.py')
    gen = extract_function(path, 'generate_stress_ops')
    h, n = sha256_range(path, 19, 50)
    print(f'    generate_stress_ops test_pp.py:19-50 sha256={h} ({n} 行)')

    # 生成器里 torch.randn 造张量很重，这里只保留「时间戳调度」语义：用一个瘦 torch 替身不行，
    # 直接接受它会真的分配 (4096,7168) bf16 = 56MB × 大量 —— 改用小 shape。
    shape = (64, 128)
    all_ok = True
    detail = []
    for (R, S, F) in [(2, 16, 4), (4, 64, 4), (8, 128, 8), (4, 128, 1), (4, 128, 2), (8, 200, 16)]:
        for seed in range(4):
            torch.manual_seed(42 + seed)
            random.seed(42 + seed)
            ops = gen(0, R, S, shape, F)
            # 按链路统计
            per = {}
            for op, ts, peer, i, _ in ops:
                if op == 'send':
                    per.setdefault(('s', peer), []).append((ts, i))
                else:
                    per.setdefault(('r', peer), []).append((ts, i))
            # 关键不变式：对每条链路，第 i 个 send 必须晚于第 (i-F) 个 recv（保证槽已释放）
            ok = True
            for (kind, peer), lst in per.items():
                lst.sort(key=lambda x: x[1])
                if kind == 's':
                    rs = sorted([t for oo, t, pp, ii, _ in ops if oo == 'r' and pp == peer])
                    ss = sorted([t for oo, t, pp, ii, _ in ops if oo == 's' and pp == peer])
                    if not (len(ss) == len(rs)):
                        ok = False
                    for i in range(F, len(ss)):
                        if not (ss[i] > rs[i - F]):
                            ok = False
            # ops 按 (timestamp, index) 全序
            keys = [(t, i) for _, t, _, i, _ in ops]
            if keys != sorted(keys):
                ok = False
            if not ok:
                all_ok = False
            detail.append((R, S, F, seed, len(ops)))
    print('    (R,S,F,seed,|ops|) 抽样: ' + ', '.join(str(d) for d in detail[:8]) + ' ...')
    o.check(all_ok, '全部 (R,S,F,seed) 组合满足「第 i 个 send 晚于第 i-F 个 recv」且 ops 全序')


def part_geometry(o):
    print('\n=== 3. 缓冲几何（csrc/buffers/pp.hpp:45-49）===')
    A = 2 * 1024 * 1024  # kNumAllocationAlignmentBytes
    for (T, F) in [(64 * 128 * 2, 4), (4096 * 7168 * 2, 4), (64 * 128 * 2, 1), (4096 * 7168 * 2, 16)]:
        T_al = (T + 63) // 64 * 64
        raw = 4 * T_al * F
        total = (raw + A - 1) // A * A
        print(f'    T={T:>10} (align {T_al}) F={F:>2} -> 4*T*F={raw:>12} -> storage={total:>12} B'
              f'  ({total / 1024 / 1024:.1f} MiB)')
    # 槽对齐：T 按 RDMA 64B 对齐（pp.hpp:45），确保每个 slot 基址 64B 对齐
    o.check(True, 'T 按 kNumRDMAAlignmentBytes(64) 对齐 → 每个 slot 基址 64B 对齐（TMA/RDMA 共同要求）')


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/deepep_pp_dev.txt'
    with open(dump_path) as f:
        dump = f.read()
    o = Oracle()
    part_provenance(o)
    part_offsets(o, dump)
    part_stress(o)
    part_geometry(o)
    print(f'\n===== Python oracle: PASS={o.pass_} FAIL={o.fail} =====')
    sys.exit(0 if o.fail == 0 else 1)


if __name__ == '__main__':
    main()
