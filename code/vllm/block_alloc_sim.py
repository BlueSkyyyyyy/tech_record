#!/usr/bin/env python3
"""PagedAttention 块分配模拟：连续预留 vs 分页 + 前缀共享。

对应专题第 2、3 篇。用法：
    python3 block_alloc_sim.py            # 跑 2000 个请求的对照
    python3 block_alloc_sim.py --test     # 自测

模拟里一个块 = block_size 个 token 的 KV；请求只在"真正用到"时才占块。
连续预留模式则按 max_model_len 一次性占满，用来对比显存利用率。
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Block:
    block_id: int
    ref_cnt: int = 0
    block_hash: str | None = None


class BlockPool:
    """完全对应 vLLM 的三个结构：块数组 + 空闲链表 + 哈希表。"""

    def __init__(self, num_blocks: int) -> None:
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free: deque[int] = deque(range(num_blocks))
        self.cache: dict[str, Block] = {}
        self.peak_in_use = 0

    @property
    def in_use(self) -> int:
        return len(self.blocks) - len(self.free)

    def allocate(self, n: int) -> list[Block] | None:
        if n > len(self.free):
            return None
        out: list[Block] = []
        for _ in range(n):
            blk = self.blocks[self.free.popleft()]
            # 队头可能还挂着前缀缓存：先淘汰，再交出去。
            if blk.block_hash is not None:
                self.cache.pop(blk.block_hash, None)
                blk.block_hash = None
            blk.ref_cnt = 1
            out.append(blk)
        self.peak_in_use = max(self.peak_in_use, self.in_use)
        return out

    def free_blocks(self, blocks: list[Block]) -> None:
        # 与 vLLM 一致：无哈希 LIFO 插队头，有哈希 FIFO 追加队尾。
        for blk in blocks:
            blk.ref_cnt -= 1
            if blk.ref_cnt == 0:
                if blk.block_hash is None:
                    self.free.appendleft(blk.block_id)
                else:
                    self.free.append(blk.block_id)

    def cache_block(self, blk: Block, h: str) -> None:
        blk.block_hash = h
        self.cache[h] = blk

    def hit(self, h: str) -> Block | None:
        blk = self.cache.get(h)
        if blk is None:
            return None
        if blk.ref_cnt == 0:
            # O(n) 摘除；vLLM 用侵入式链表做到 O(1)。
            self.free.remove(blk.block_id)
        blk.ref_cnt += 1
        return blk


def requests(n: int, seed: int = 0, max_len: int = 8192) -> list[int]:
    rng = random.Random(seed)
    # 长尾：多数请求几百 token，少数接近 8k。
    return [min(max_len, max(1, int(rng.lognormvariate(6.0, 0.9)))) for _ in range(n)]


def fill(trace: list[int], pool: BlockPool, block_size: int,
         max_active: int | None = None) -> dict:
    """按 trace 顺序尽量多地把请求放进块池，返回能同时容纳的请求数。"""
    active: list[list[Block]] = []
    used = 0
    for length in trace:
        need = -(-length // block_size)
        if max_active is not None and len(active) >= max_active:
            break
        blocks = pool.allocate(need)
        if blocks is None:
            break
        active.append(blocks)
        used += need
    return {"requests": len(active), "blocks": used,
            "pool": len(pool.blocks), "util": used / len(pool.blocks)}


def contiguous_capacity(pool_blocks: int, block_size: int, max_len: int) -> int:
    return pool_blocks // (-(-max_len // block_size))


def _test() -> None:
    pool = BlockPool(4)
    a = pool.allocate(2)
    assert a is not None and pool.in_use == 2
    assert pool.allocate(3) is None  # 不够就整体失败，不做部分分配

    # 前缀共享：同一个哈希被两个请求引用，free 一次不能回收。
    pool.cache_block(a[0], "h0")
    hit = pool.hit("h0")
    assert hit is a[0] and hit.ref_cnt == 2
    pool.free_blocks([a[0]])
    assert a[0].ref_cnt == 1 and pool.in_use == 2
    pool.free_blocks([a[0], a[1]])
    assert pool.in_use == 0

    # 有哈希的块回到队尾（LRU），无哈希的回到队头（LIFO）。
    assert pool.free[-1] == a[0].block_id
    assert pool.free[0] != a[0].block_id

    # 哈希块在队尾，要等前面的块都被拿走才会被淘汰。
    b = pool.allocate(3)
    assert b is not None and "h0" in pool.cache
    b = pool.allocate(1)
    assert b is not None and "h0" not in pool.cache

    # 同样的块池，分页能同时容纳的请求数远多于连续预留。
    tr = requests(500, seed=1)
    p_pool = BlockPool(2048)
    p = fill(tr, p_pool, 16)
    c = contiguous_capacity(2048, 16, 8192)
    assert p["requests"] > c * 5, (p, c)
    assert p_pool.in_use == p["blocks"]
    print(f"block_alloc: all tests passed (paged {p['requests']} vs contiguous {c})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=27648, help="54 GiB / 2 MiB")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        _test()
        return

    tr = requests(args.requests, args.seed, args.max_len)
    print(f"trace: {len(tr)} 请求, 平均 {sum(tr)/len(tr):.0f} token, "
          f"块池 {args.blocks} 块 (block={args.block_size})")
    print(f"{'方案':<14}{'同时容纳':>10}{'占用块':>10}{'利用率':>10}")
    pool = BlockPool(args.blocks)
    p = fill(tr, pool, args.block_size)
    print(f"{'分页':<14}{p['requests']:>10}{p['blocks']:>10}{p['util']:>9.1%}")
    c = contiguous_capacity(args.blocks, args.block_size, args.max_len)
    used = c * (-(-args.max_len // args.block_size))
    print(f"{'连续预留 8k':<14}{c:>10}{used:>10}{used/args.blocks:>9.1%}")
    print(f"\n分页/连续 并发倍数: {p['requests']/max(1, c):.1f}x")

    # 再跑一遍完整生命周期，确认所有块都归还。
    pool2 = BlockPool(args.blocks)
    blocks = pool2.allocate(100)
    assert blocks is not None
    pool2.free_blocks(blocks)
    assert pool2.in_use == 0 and len(pool2.free) == args.blocks


if __name__ == "__main__":
    main()
