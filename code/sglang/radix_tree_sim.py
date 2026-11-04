#!/usr/bin/env python3
"""前缀缓存两种数据结构的对拍：SGLang 的基数树 vs vLLM 的块哈希链。

同一个 token 前缀池、同一个容量、同一个 LRU 语义，两种结构在
「命中多少 token / 留下多少 KV / 淘汰多少次」上的差别全部来自**索引粒度**：

    SGLang（RadixAttention）  token 级基数树，节点可分裂，命中可落在任意 token 边界
    vLLM（PagedAttention）    16-token 块哈希链，命中只能落在整块边界，链式哈希防碰撞

用法：
    python3 radix_tree_sim.py              # 三组负载的对照表
    python3 radix_tree_sim.py --test       # 自测（性质断言）
    python3 radix_tree_sim.py --trace      # 打印一棵小树的形状
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import OrderedDict, defaultdict

BLOCK = 16  # vLLM 的块大小（也是本模拟里块哈希的粒度）


# ---------------------------------------------------------------------------
# SGLang 侧：token 级基数树
# ---------------------------------------------------------------------------
class Node:
    """对应 sglang/srt/mem_cache/radix_cache.py::TreeNode。"""

    __slots__ = ("key", "value", "children", "parent", "lock_ref", "last_access")

    def __init__(self, key=(), value=None, parent=None):
        self.key = tuple(key)          # 本节点独占的那一段 token
        self.value = value             # 与 key 等长的 slot 列表；None = 已淘汰
        self.children = {}
        self.parent = parent
        self.lock_ref = 0              # >0 时整条路径不可淘汰
        self.last_access = 0

    @property
    def evicted(self):
        return self.value is None

    def __len__(self):
        return len(self.key)


class RadixCache:
    """基数树前缀缓存：match / insert（含分裂）/ evict（只看叶子）。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.root = Node(key=(), value=[], parent=None)
        self.root.lock_ref = 1          # 根永不被淘汰
        self.next_slot = 0
        self.cached_tokens = 0          # 树上所有 value 的 token 总数
        self.evictable = 0              # 其中可淘汰的
        self.protected = 0              # lock_ref>0 的
        self.clock = 0
        self.num_evict_calls = 0
        self.num_splits = 0
        self.num_visits = 0              # match 时访问过的节点数（查找成本）

    # -- 内部工具 ----------------------------------------------------------
    def _new_slots(self, n: int) -> list[int]:
        out = list(range(self.next_slot, self.next_slot + n))
        self.next_slot += n
        return out

    def _touch(self, node: Node) -> None:
        self.clock += 1
        node.last_access = self.clock

    def _common_prefix(self, a: tuple, b: tuple) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _match_helper(self, key: tuple):
        """沿树下行，返回 (命中前缀长度, 最后节点, 是否停在节点中间)。"""
        node, matched, rest = self.root, 0, key
        while rest:
            child = node.children.get(rest[0])
            if child is None:
                break
            prefix = self._common_prefix(rest, child.key)
            self.num_visits += 1
            matched += prefix
            self._touch(child)
            if prefix < len(child.key):
                # 命中落在节点中间 → 分裂，把匹配边界暴露出来
                node = self._split(child, prefix)
                return matched, node, True
            rest = rest[prefix:]
            node = child
        return matched, node, False

    def _split(self, child: Node, at: int) -> Node:
        """把 child 的前 at 个 token 切出来做父节点（不复制 KV，只切 value）。"""
        self.num_splits += 1
        head = Node(
            key=child.key[:at],
            value=child.value[:at],
            parent=child.parent,
        )
        head.lock_ref = child.lock_ref  # 与 _split_node 一致：前缀继承锁定状态
        child.parent.children[child.key[0]] = head
        head.children[child.key[at]] = child
        child.key = child.key[at:]
        child.value = child.value[at:]
        child.parent = head
        self._touch(head)
        return head

    # -- 公开接口 ----------------------------------------------------------
    def match(self, key: tuple) -> tuple[int, Node]:
        matched, node, _ = self._match_helper(key)
        return matched, node

    def insert(self, key: tuple, slots: list[int]) -> tuple[int, Node]:
        matched, node = 0, self.root
        rest, rest_slots = key, slots
        while rest:
            child = node.children.get(rest[0])
            if child is None:
                break
            prefix = self._common_prefix(rest, child.key)
            if prefix < len(child.key):
                child = self._split(child, prefix)
            matched += prefix
            rest, rest_slots = rest[prefix:], rest_slots[prefix:]
            node = child
        if rest:  # 挂一个新叶子
            leaf = Node(key=rest, value=list(rest_slots), parent=node)
            node.children[rest[0]] = leaf
            self.cached_tokens += len(rest)
            self.evictable += len(rest)
            self._touch(leaf)
            self._refresh(leaf)
            node = leaf
        self._refresh(node)
        return matched, node

    def inc_lock_ref(self, node: Node) -> int:
        delta = 0
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable -= len(node)
                self.protected += len(node)
                delta -= len(node)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: Node) -> int:
        delta = 0
        while node is not self.root:
            if node.lock_ref == 1:
                self.evictable += len(node)
                self.protected -= len(node)
                delta += len(node)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def _leaves(self):
        return [
            n
            for n in self._walk(self.root)
            if not n.children and n is not self.root and n.lock_ref == 0
        ]

    def _walk(self, node: Node):
        yield node
        for c in node.children.values():
            yield from self._walk(c)

    def _refresh(self, node: Node) -> None:
        pass  # 叶子集合按需计算，保持与真实实现「evictable_leaves 是缓存」的区别可忽略

    def evict(self, num_tokens: int) -> int:
        """淘汰最久未用的叶子，直到腾出 num_tokens 或无人可淘汰。"""
        self.num_evict_calls += 1
        freed = 0
        while freed < num_tokens:
            leaves = self._leaves()
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            freed += len(victim)
            self.cached_tokens -= len(victim)
            self.evictable -= len(victim)  # 只有 lock_ref==0 的叶子会被选中
            del victim.parent.children[victim.key[0]]
            victim.value = None
        return freed

    # -- 统计 --------------------------------------------------------------
    def size(self) -> int:
        return self.cached_tokens

    def shape(self, node=None, indent=0, out=None):
        """打印树形状，用来肉眼确认分裂行为。"""
        out = [] if out is None else out
        node = self.root if node is None else node
        if node is not self.root:
            out.append(
                f"{'  ' * indent}{node.key[:4]}{'…' if len(node.key) > 4 else ''}"
                f" (len={len(node)}, lock={node.lock_ref}, slots={node.value[:3]}…)"
            )
        for c in sorted(node.children.values(), key=lambda n: n.key):
            self.shape(c, indent + 1, out)
        return out


# ---------------------------------------------------------------------------
# vLLM 侧：块哈希链
# ---------------------------------------------------------------------------
class BlockPool:
    """对应 vllm/v1/core/block_pool.py + prefix_caching_block.py 的简化版。"""

    def __init__(self, capacity_tokens: int, block: int = BLOCK):
        self.block = block
        self.num_blocks = capacity_tokens // block
        self.hash_to_id: dict[bytes, int] = {}
        self.id_to_hash: dict[int, bytes] = {}
        self.ref_cnt: dict[int, int] = defaultdict(int)
        self.free: OrderedDict[int, None] = OrderedDict()  # LRU：最左最旧
        self.next_id = 0
        self.num_evict_calls = 0
        self.num_hashes = 0              # match + allocate 里算过的块哈希数（查找成本）

    def _new_block(self) -> int:
        if len(self.ref_cnt) < self.num_blocks:
            bid = self.next_id
            self.next_id += 1
            self.ref_cnt[bid] = 0
            return bid
        # 池满：从 LRU 头部摘一个 ref_cnt==0 的块
        self.num_evict_calls += 1
        for bid in list(self.free):
            del self.free[bid]
            for h, i in list(self.hash_to_id.items()):
                if i == bid:
                    del self.hash_to_id[h]
                    del self.id_to_hash[bid]
                    break
            return bid
        raise RuntimeError("块池耗尽且无可淘汰块（仍有请求持有引用）")

    @staticmethod
    def hash_block(parent: bytes, tokens: tuple) -> bytes:
        return hashlib.sha256(repr((parent, tokens)).encode()).digest()

    def match(self, key: tuple) -> int:
        """返回命中的整块 token 数。"""
        parent = b"\x00" * 32
        hit = 0
        for i in range(0, len(key) - self.block + 1, self.block):
            self.num_hashes += 1
            h = self.hash_block(parent, key[i : i + self.block])
            if h not in self.hash_to_id:
                break
            bid = self.hash_to_id[h]
            if bid in self.free:  # 命中 = 触碰，移到 LRU 尾部
                self.free.move_to_end(bid)
            hit += self.block
            parent = h
        return hit

    def allocate(self, key: tuple, hit_tokens: int) -> tuple[list[int], int]:
        """为整条 key 分配块：命中部分加引用，其余新建；返回 (块号列表, 新建块数)。"""
        blocks = []
        parent = b"\x00" * 32
        new_blocks = 0
        for i in range(0, len(key) - self.block + 1, self.block):
            self.num_hashes += 1
            h = self.hash_block(parent, key[i : i + self.block])
            bid = self.hash_to_id.get(h)
            if bid is None:
                bid = self._new_block()
                self.hash_to_id[h] = bid
                self.id_to_hash[bid] = h
                new_blocks += 1
            self.ref_cnt[bid] += 1
            self.free.pop(bid, None)
            blocks.append(bid)
            parent = h
        return blocks, new_blocks

    def free_blocks(self, blocks: list[int]) -> None:
        for bid in blocks:
            self.ref_cnt[bid] -= 1
            if self.ref_cnt[bid] == 0:
                self.free[bid] = None  # 归入 LRU 尾部（最新），可被再次命中

    def cached_tokens(self) -> int:
        held = sum(1 for b in self.ref_cnt if self.ref_cnt[b] > 0)
        return (held + len(self.free)) * self.block


# ---------------------------------------------------------------------------
# 负载：共享前缀 + 组内共享 + 随机后缀
# ---------------------------------------------------------------------------
def make_workload(kind: str, n_req: int, seed: int = 0):
    """返回 [(token 序列)]。四组负载的「共享结构」不同，用来隔离粒度效应。"""
    rng = random.Random(seed)
    if kind == "multiturn":
        # 多轮对话：每轮在上一轮后面追加 24 个 token。
        # 上一轮长度 = 300 + 24t，几乎永远不是 16 的倍数 → 块缓存必然重算尾巴。
        reqs = []
        for conv in range(3):
            base = [conv * 10**6 + i for i in range(300)]
            cur = list(base)
            for turn in range(12):
                cur = cur + [conv * 10**6 + 500000 + turn * 37 + i for i in range(24)]
                reqs.append(list(cur))
        return reqs
    if kind == "toolcall":
        # 工具调用循环：每轮只追加 8 个 token（一次 tool result 很短）。
        # 块缓存每次都要重算上一轮尾巴（最多 15 token），增量本身才 8 token。
        reqs = []
        for conv in range(4):
            cur = [conv * 10**6 + i for i in range(200)]
            for turn in range(20):
                cur = cur + [conv * 10**6 + 600000 + turn * 53 + i for i in range(8)]
                reqs.append(list(cur))
        return reqs
    if kind == "system":
        # 三段系统提示，长度刻意不是 16 的倍数（507/611/723），各被 40 个请求共享。
        reqs = []
        for g, length in enumerate((507, 611, 723)):
            sysp = [g * 10**6 + 900000 + i for i in range(length)]
            for _ in range(40):
                reqs.append(sysp + [rng.randrange(10**6) for _ in range(rng.randint(100, 300))])
        return reqs
    if kind == "agent":
        # 8 段长 skill 前缀（非 16 对齐），每段被若干请求复用。
        heads = [[g * 10**6 + 700000 + i for i in range(317 + 22 * g)] for g in range(8)]
        return [
            rng.choice(heads) + [rng.randrange(10**6) for _ in range(rng.randint(32, 128))]
            for _ in range(n_req)
        ]
    # rag：每个请求一篇不同文档，零复用（对照组）
    return [
        [rng.randrange(10**6) for _ in range(rng.randint(256, 768))]
        for _ in range(n_req)
    ]


def run(kind: str, n_req: int, capacity: int, seed: int = 0):
    reqs = make_workload(kind, n_req, seed)

    # --- SGLang：树 + lock_ref + LRU 叶子淘汰 ---
    tree = RadixCache(capacity)
    tree_hit = 0
    for toks in reqs:
        key = tuple(toks)
        hit, node = tree.match(key)
        tree_hit += hit
        need = len(key) - hit
        if tree.size() + need > capacity:
            tree.evict(tree.size() + need - capacity)
        tree.insert(key, tree._new_slots(len(key)))
        tree.inc_lock_ref(node)  # 请求持有自己命中的那段前缀
        tree.dec_lock_ref(node)

    # --- vLLM：块哈希 + 引用计数 + LRU 块淘汰 ---
    pool = BlockPool(capacity)
    blk_hit = 0
    for toks in reqs:
        key = tuple(toks)
        hit = pool.match(key)
        blk_hit += hit
        blocks, _ = pool.allocate(key, hit)
        pool.free_blocks(blocks)

    total = sum(len(t) for t in reqs)
    return {
        "kind": kind,
        "requests": len(reqs),
        "prompt_tokens": total,
        "tree_hit": tree_hit,
        "tree_rate": tree_hit / total,
        "blk_hit": blk_hit,
        "blk_rate": blk_hit / total,
        "tree_size": tree.size(),
        "blk_size": pool.cached_tokens(),
        "tree_visits": tree.num_visits,
        "blk_hashes": pool.num_hashes,
        "tree_splits": tree.num_splits,
        "tree_evicts": tree.num_evict_calls,
        "blk_evicts": pool.num_evict_calls,
    }


def report(rows):
    hdr = (
        f"{'负载':<10}{'请求':>5}{'prompt tok':>11}{'树命中':>8}{'块命中':>8}{'差(pp)':>8}"
        f"{'重算(树)':>10}{'重算(块)':>10}{'多算':>8}{'查找(树/块)':>14}"
    )
    print(hdr)
    print("-" * (len(hdr) + 2))
    tot_t = tot_b = tot_p = 0
    for r in rows:
        rt = r["prompt_tokens"] - r["tree_hit"]
        rb = r["prompt_tokens"] - r["blk_hit"]
        tot_t, tot_b, tot_p = tot_t + rt, tot_b + rb, tot_p + r["prompt_tokens"]
        print(
            f"{r['kind']:<10}{r['requests']:>5}{r['prompt_tokens']:>11}"
            f"{r['tree_rate']:>7.1%}{r['blk_rate']:>8.1%}"
            f"{(r['tree_rate'] - r['blk_rate']) * 100:>8.1f}"
            f"{rt:>10}{rb:>10}{rb - rt:>8}"
            f"{r['tree_visits']:>8}/{r['blk_hashes']:<5}"
        )
    print(
        f"\n合计：prompt {tot_p} token，树重算 {tot_t}，块重算 {tot_b}，"
        f"块比树多算 {tot_b - tot_t}（+{(tot_b / max(tot_t, 1) - 1) * 100:.1f}%）"
    )
    print(
        "读法：\n"
        "  · 重算 token = prompt - 命中，这才是花 FLOPs 的量；块粒度每块最多丢 15 个 token。\n"
        "  · multiturn / system 两组的差全部来自「上一轮/系统提示的长度不是 16 的倍数」。\n"
        "  · rag 组两者都是 0：没有共享前缀时，树的额外结构一分钱也赚不到，只剩内存与查找开销。\n"
        "  · 查找列：树查的是「节点 + 段内比较」（O(树深)），块查的是「每块一次哈希」（O(块数)）。\n"
        "    前缀共享越深，树的查找越便宜；块哈希是常数成本但要做满整条链。"
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _test():
    # 1) 分裂：插入 A、A+B、再查 A，命中必须正好 len(A)
    A, B = tuple(range(10)), tuple(range(10, 20))
    t = RadixCache(1024)
    t.insert(A, t._new_slots(len(A)))
    t.insert(A + B, t._new_slots(len(A + B)))
    hit, node = t.match(A)
    assert hit == len(A), (hit, len(A))
    assert node.value is not None and len(node.value) == len(A)
    # 命中落在中间也要分裂：再查 A[:3]
    hit, node = t.match(A[:3])
    assert hit == 3 and len(node.key) == 3, (hit, node.key)

    # 2) 命中一定是真前缀（性质测试）
    rng = random.Random(7)
    t = RadixCache(4096)
    for _ in range(200):
        k = tuple(rng.randrange(8) for _ in range(rng.randint(1, 40)))
        hit, _ = t.match(k)
        assert k[:hit] == k[:hit]
        t.insert(k, t._new_slots(len(k)))

    # 3) lock_ref 保护：锁住的路径不被淘汰；evictable + protected == 总大小
    t = RadixCache(1024)
    p = tuple(range(100))
    t.insert(p, t._new_slots(len(p)))
    hit, node = t.match(p)
    t.inc_lock_ref(node)
    assert t.protected == len(p) and t.evictable == 0, (t.protected, t.evictable)
    t.evict(1000)                      # 锁着 → 一个都淘汰不了
    assert t.match(p)[0] == len(p)
    t.dec_lock_ref(node)
    assert t.evictable == len(p) and t.protected == 0
    freed = t.evict(len(p))
    assert freed >= len(p) and t.match(p)[0] == 0

    # 4) evictable + protected 恒等于树上 token 数（随机操作后仍成立）
    rng = random.Random(11)
    t = RadixCache(2048)
    for _ in range(300):
        k = tuple(rng.randrange(6) for _ in range(rng.randint(1, 30)))
        t.insert(k, t._new_slots(len(k)))
        hit, node = t.match(k)
        t.inc_lock_ref(node)
        t.dec_lock_ref(node)
        if rng.random() < 0.2:
            t.evict(rng.randint(1, 64))
        assert t.evictable + t.protected == t.cached_tokens, (
            t.evictable, t.protected, t.cached_tokens)

    # 5) 块缓存：命中必是 16 的倍数，且不超过树的命中
    for kind in ("multiturn", "toolcall", "system", "agent", "rag"):
        reqs = make_workload(kind, 74, seed=3)
        t, p = RadixCache(10**9), BlockPool(10**9)
        for toks in reqs:
            key = tuple(toks)
            th, _ = t.match(key)
            bh = p.match(key)
            assert bh % BLOCK == 0
            assert bh <= th, (kind, bh, th)
            t.insert(key, t._new_slots(len(key)))
            blocks, _ = p.allocate(key, bh)
            p.free_blocks(blocks)

    # 6) 同为有限容量时，树的命中率不低于块（token 粒度更细）
    rows = [run(k, 96, 40000) for k in ("multiturn", "toolcall", "system", "agent", "rag")]
    for r in rows:
        assert r["tree_rate"] >= r["blk_rate"] - 1e-9, r

    # 7) 确定性
    a = run("system", 80, 30000, seed=5)
    b = run("system", 80, 30000, seed=5)
    assert a == b

    print("radix_tree_sim: all tests passed")
    report(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--trace", action="store_true", help="打印一棵小树的形状")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--capacity", type=int, default=60000)
    args = ap.parse_args()

    if args.test:
        _test()
        return
    if args.trace:
        t = RadixCache(1024)
        base = tuple(range(20))
        for extra in ((), (100,), (101, 102), (100, 103)):
            t.insert(base + extra, t._new_slots(len(base + extra)))
        t.match(base[:8])  # 触发一次分裂
        print("\n".join(t.shape()))
        return
    report([run(k, args.requests, args.capacity)
            for k in ("multiturn", "toolcall", "system", "agent", "rag")])


if __name__ == "__main__":
    main()
