#!/usr/bin/env python3
"""前缀缓存模拟：哈希链、引用计数、LRU 淘汰与 cache_salt 隔离。

对应专题第 3 篇。用法：
    python3 prefix_cache_sim.py            # 系统提示复用的命中率账本
    python3 prefix_cache_sim.py --test     # 自测
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import OrderedDict

BLOCK = 16


def h_block(parent: bytes, tokens: tuple[int, ...], salt: str = "") -> bytes:
    """对应 vLLM 的 hash_block_tokens：父哈希 + 本块 token + extra_keys。"""
    payload = repr((parent, tokens, salt)).encode()
    return hashlib.sha256(payload).digest()


def chain(tokens: list[int], salt: str = "") -> list[bytes]:
    """把 token 序列切成块并返回整条哈希链（最后不满一块的丢弃）。"""
    out: list[bytes] = []
    parent = b"\x00" * 32  # NONE_HASH 的替身
    for i in range(0, len(tokens) - BLOCK + 1, BLOCK):
        parent = h_block(parent, tuple(tokens[i:i + BLOCK]), salt)
        out.append(parent)
    return out


class PrefixCache:
    """哈希 -> 块号，块号 -> 引用计数；空闲块用 OrderedDict 做 LRU。"""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.table: dict[bytes, int] = {}
        self.ref: dict[int, int] = {}
        self.lru: OrderedDict[int, None] = OrderedDict()  # 队头最久未用
        self.next_id = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _new_block(self) -> int:
        if self.next_id < self.num_blocks:
            bid = self.next_id
            self.next_id += 1
        else:
            # 淘汰 LRU 队头；它一定 ref_cnt == 0。
            bid, _ = self.lru.popitem(last=False)
            for h, b in list(self.table.items()):
                if b == bid:
                    del self.table[h]
                    break
            self.evictions += 1
        self.ref[bid] = 1
        return bid

    def acquire(self, hashes: list[bytes]) -> int:
        """返回命中的 token 数；命中块 ref_cnt++，未命中的前缀停止。"""
        hit = 0
        for h in hashes:
            bid = self.table.get(h)
            if bid is None:
                self.misses += 1
                break
            if self.ref.get(bid, 0) == 0:
                self.lru.pop(bid, None)
            self.ref[bid] = self.ref.get(bid, 0) + 1
            self.hits += 1
            hit += BLOCK
        # 为未命中的块建立新条目（简化：只登记，不模拟写入）。
        for i in range(hit // BLOCK, len(hashes)):
            h = hashes[i]
            if h not in self.table:
                self.table[h] = self._new_block()
            else:
                self.ref[self.table[h]] += 1
        return hit

    def release(self, hashes: list[bytes]) -> None:
        for h in hashes:
            bid = self.table.get(h)
            if bid is None:
                continue
            self.ref[bid] -= 1
            if self.ref[bid] == 0:
                self.lru[bid] = None

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def workload(n: int, seed: int = 0, sys_len: int = 2000,
             salt: str = "") -> list[list[int]]:
    rng = random.Random(seed)
    system = [1000 + i % 500 for i in range(sys_len)]
    out = []
    for _ in range(n):
        user = [rng.randrange(50000) for _ in range(rng.randint(80, 400))]
        out.append(system + user)
    return out


def run(reqs: list[list[int]], cache: PrefixCache, salt: str = "") -> dict:
    saved = total = 0
    for tokens in reqs:
        hashes = chain(tokens, salt)
        total += len(tokens)
        hit = cache.acquire(hashes)
        saved += hit
        cache.release(hashes)
    return {"requests": len(reqs), "saved": saved, "total": total,
            "hit_rate": cache.hit_rate(), "evictions": cache.evictions}


def _test() -> None:
    a = list(range(100))
    assert chain(a) == chain(a)
    assert chain(a) != chain(list(range(1, 101)))
    # 父哈希不同 -> 后续全不同。
    assert chain(a)[3] != chain(a, salt="t1")[3]

    # 相同前缀必须命中；不同前缀第一块就分叉。
    c = PrefixCache(64)
    assert c.acquire(chain(a)) == 0
    c.release(chain(a))
    assert c.acquire(chain(a[:64])) == 64
    assert c.acquire(chain(list(range(1, 101)))) == 0

    # salt 隔离：同一 token、不同盐，零命中。
    c2 = PrefixCache(64)
    c2.acquire(chain(a, "tenant-a")); c2.release(chain(a, "tenant-a"))
    assert c2.acquire(chain(a, "tenant-b")) == 0

    # 容量 2 块、访问 3 个不同前缀：必然发生淘汰。
    c3 = PrefixCache(2)
    for i in range(3):
        h = chain(list(range(i * 100, i * 100 + 32)))
        c3.acquire(h)
        c3.release(h)
    assert c3.evictions >= 1

    r = run(workload(20, seed=2), PrefixCache(256))
    r_salt = run(workload(20, seed=2), PrefixCache(256), salt="t1")
    assert r["saved"] > 0 and r_salt["saved"] > 0
    print(f"prefix_cache: all tests passed (hit_rate={r['hit_rate']:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--sys-len", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--salt", default="")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        _test()
        return

    reqs = workload(args.requests, sys_len=args.sys_len, salt=args.salt)
    cache = PrefixCache(args.blocks)
    r = run(reqs, cache, args.salt)
    print(f"系统提示 {args.sys_len} token, 请求 {r['requests']} 个, "
          f"块池 {args.blocks} 块")
    print(f"命中 token      : {r['saved']:,} / {r['total']:,}")
    print(f"命中率（按 token）: {r['hit_rate']:.1%}")
    print(f"省下 prefill    : 约 {r['saved']/args.sys_len:.1f} 个完整系统提示")
    print(f"淘汰次数        : {r['evictions']}")

    # 对照：每条请求都改系统提示（盐不同）时命中率归零。
    if not args.salt:
        cache2 = PrefixCache(args.blocks)
        r2 = run(workload(args.requests, sys_len=args.sys_len), cache2,
                 salt="unique-salt")
        print(f"加 cache_salt 后 : {r2['hit_rate']:.1%}（跨租户隔离的代价）")


if __name__ == "__main__":
    main()
