---
title: "SGLang 研读（二）RadixAttention：用基数树做前缀缓存"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 3
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "RadixAttention", "前缀缓存", "KV Cache"]
categories: ["推理框架"]
---

上一篇走完了一个请求穿过三个进程的路。这一篇停在它最容易出问题的那一步：**这段 prompt 的 KV，已经算过多少？**

[导读]({{< relref "sglang-00-index" >}}) 里那句话——「共享结构本身就是一棵树」——在这里变成 837 行代码：`python/sglang/srt/mem_cache/radix_cache.py`。它是 SGLang 与 vLLM 的第一处结构性分叉，也是后面所有账本的源头：调度器排序看它、KV 池的余量算它、显存压力下淘汰的也是它。

如果没读过 vLLM 侧的对位篇，建议先看[前缀缓存：一条哈希链省下多少 prefill]({{< relref "vllm-03-prefix-caching" >}})，本篇会反复拿它当参照。

## 1. 先看树怎么长出来

一个客服服务，200 个会话共享同一段 4,000 token 系统提示；每个租户还有一段 800 token 的 skill 定义，A 和 B 前 800 token 完全一样，从第 801 个 token 开始分叉。

把它按 token 逐位对齐、相同的部分叠起来，就得到一棵树。为了后面能算账，换成一个更小的例子：**系统提示 2,000 token，100 个请求各带 200 token 不同的用户输入**：

```text
root                 key=[]            value=[]        lock_ref=1
└─ [system 2000 tok]  key=2000 tokens   value=2000 slots  lock_ref=0
   ├─ [user#0 200 tok]  ← 叶子，可淘汰
   ├─ [user#1 200 tok]
   └─ … 共 100 个叶子
```

树上只有 **102 个节点、1 次分裂、22,000 个 token**，却服务了 100 个请求。这个形状不是我画的，是把 `radix_cache.py` 的数据结构单独跑出来的（同一个 `RadixCache` 类，只把 allocator 换成桩）：100 个请求依次 `insert` 后，`root.children` 只有一个键，第二层 100 个叶子，`evictable_size_ = 22000`。

**节点里存什么**（`TreeNode`，`radix_cache.py:251`）：

| 字段 | 含义 | 为什么必须有 |
|---|---|---|
| `key: RadixKey` | 本节点**独占**的那一段 token（不是从根到这里的全部） | 边压缩：一条边是一整段，不是一位 |
| `value: Tensor` | 与 `key` 等长的 int64 slot 下标 | 树不存 KV 字节，只存「KV 在哪」 |
| `children` | `defaultdict(TreeNode)`，键是**首 `page_size` 个 token** | 分叉只看第一位，O(1) 下探 |
| `parent` | 反向指针 | `dec_lock_ref`/淘汰要从叶往上走 |
| `lock_ref` | 有多少个请求的路径经过这里 | 引用计数，>0 不可淘汰 |
| `last_access_time` / `creation_time` | `time.monotonic()` | LRU / FIFO 的排序键 |
| `hit_count` | 被多少条请求路径复用（`_inc_hit_count`，`radix_cache.py:707`） | LFU / SLRU 的排序键 |
| `priority` | 调用方给的优先级，插入时沿路径取 max | `PriorityStrategy` 用 |

注意 `value` 是 **slot 下标**，不是 KV。KV 张量一直在 `token_to_kv_pool` 里不动，树只是持有「这段 token 的 KV 落在哪几号 slot」的索引。**一条根到叶的路径，就是该请求的完整 slot 序列**：`path_device_indices()`（`radix_cache.py:646`）从节点往上收集 `value`、反转、`torch.cat`；写完 prefill 后 `checkpoint()` 把它写回请求自己的行（`req_to_token_pool.write`，`radix_cache.py:536`），再把请求手上那份**重复的 slot 释放掉**（`radix_cache.py:509`）。

这一步是「省显存」的全部秘密：`insert` 之后，同一段 token 的 KV 在池子里只剩树里那一份，请求行只是指向它。

## 2. match_prefix：往下走，遇到岔口就分裂

`match_prefix`（`radix_cache.py:369`）的核心只有一个循环（`_match_prefix_helper`，`radix_cache.py:656`）：

```python
child_key = key.child_key(self.page_size)          # 首 page_size 个 token
while len(key) > 0 and child_key in node.children.keys():
    child = node.children[child_key]
    prefix_len = child.key.match(key, page_size=self.page_size)
    if prefix_len < len(child.key):
        new_node = self._split_node(child.key, child, prefix_len)   # 岔口落在边内部
        ...
        break
    else:
        value.append(child.value); node = child; key = key[prefix_len:]
```

三个细节决定了它的成本：

1. **分叉只按第一位判断**。`child_key` 是首 `page_size` 个 token 组成的可哈希键（`radix_cache.py:223`），`page_size=1` 时就是一个 `int`。所以「走下一层」是一次 dict 查找，不是一次线性扫描。
2. **每条边走一次比较**。`RadixKey.match` → `match_at`（`radix_cache.py:183`）不是逐 token 的 Python 循环，而是**指数搜索 + 二分**：按 1、2、4、8… 倍增窗口做 C 层切片比较，找到第一个不一致的窗口再二分（`radix_cache.py:194-212`）。一条 2,000 token 的边，Python 层只跳约 22 次。
3. **复杂度是两道账**：节点跳数是 $O(d)$，token 比较量是 $O(\sum_{\text{edge}} |\text{edge}|)$，后者在 C 层做，前者才是 Python 开销。合起来近似 $O(L)$（$L$ 为命中长度），不是 $O(L \times d)$。

### 分裂：发生在岔口那一层，且只复制下标

当 `prefix_len < len(child.key)`，匹配终点落在一条边的**内部**——这条边必须被切开，否则「命中 2,000 token」这件事在树上没有对应的节点可以锚定。`_split_node`（`radix_cache.py:682`）：

```python
new_node.key   = child.key[:split_len]
new_node.value = child.value[:split_len].clone()
child.key      = child.key[split_len:]
child.value    = child.value[split_len:].clone()
new_node.children = {key[split_len:].child_key(self.page_size): child}
```

就地**插入一层**，深度 +1，别的节点一个都没动。分裂只影响这条边，不触发任何重排、重算、KV 拷贝。

「不复制 KV」要说得准确一点：它**确实复制了一份 `value`**，但 `value` 是 int64 下标数组，不是 KV 张量。按 Qwen3-8B 算（36 层 × 8 个 KV head × 128 维 × fp16 × K/V 两份 = **144 KiB/token**），2,000 token 的 KV 是 281 MiB，而切开一条 2,200 token 的边只需两次 `clone`：2,000 + 200 个 int64 = **17.6 KB**。比例约 1:16000。这也解释了为什么用 `.clone()` 而不是切片视图——每个节点持有一段独立连续的下标内存，淘汰时可以直接 `free_segment`，不会被父节点的生命周期拖住。

**只读的 match 也会分裂**。这一点容易误解：分裂不是 insert 的副作用，而是「精确锚定命中点」的必要动作。实测：先插入 2,200 token，再用 2,000 token 的 key 只做 `match_prefix`，树就从 `1 个 2200 节点` 变成 `2000 + 200 两个节点`，`_split_node` 调用 1 次。它只改结构，不搬运数据，所以代价是微秒级——本机实测一条 2,000 token 边的 `match_prefix` 约 12 µs（纯 Python 层，不含 attention 前的任何 GPU 工作）。

**分裂次数是可以提前算的**：一次 `insert` 最多分裂一次（`_insert_helper` 的 while 循环里，分裂后必然跳出），每次分裂只增加一个节点，于是 `#nodes ≤ 1 + 2 × #insert`。真正决定它的是**到达顺序**：同一批 100 个请求，若各自 200 token 的用户输入互不相同，全程只分裂 **1 次**（首次分叉），得 102 个节点、22,000 token；若这 100 个用户输入互相是前缀（长度 1000 递减到 901），则分裂 **99 次**、节点数 101，但树上只有 **3,000 token**——99 次分裂一个叶子都没多长出来（插的都是别人的前缀），KV 也只存最长那条。**树的节点数跟着分叉数走，不跟着请求数走**，这是它和块表最本质的差别。

## 3. 引用计数：lock_ref 是一条路径的属性

一个节点被请求 A 用着，不代表它能被淘汰：**它能被复用，前提是从根到它的整条路径都还在**。所以 `lock_ref` 不是节点级标记，而是路径级计数——`inc_lock_ref`（`radix_cache.py:587`）从节点一路加到根：

```python
while node != self.root_node:
    if node.lock_ref == 0:
        self.evictable_size_ -= len(node.key)
        self.protected_size_ += len(node.key)
    node.lock_ref += 1
    self._update_leaf_status(node); node = node.parent
```

实测那棵 100 请求的树：锁住一个 200 token 的叶子后，`evictable_size_` 从 22000 掉到 19800，`protected_size_` 从 0 涨到 2200——**叶子 200 + 共享的 2000 全被保护**，`inc_lock_ref` 返回的 `delta = -2200` 正是给 allocator 记账用的。

记 $\mathcal{L}$ 为树上所有非根、未淘汰的节点，$E$ 为 `evictable_size_`，$P$ 为 `protected_size_`，则任何时刻都有：

$$
E + P = \sum_{n \in \mathcal{L}} |n.\text{key}| = \text{total\_size}()
$$

四条路径各自守恒：insert 新建节点时 `E += len(key)`（`radix_cache.py:759`）；`inc_lock_ref` 在 `lock_ref` 由 0 变 1 时把 `len(key)` 从 $E$ 挪到 $P$；`dec_lock_ref`（`radix_cache.py:602`）反向挪回；`_delete_leaf` 把 `len(key)` 从 $E$ 减掉的同时，节点也从树上摘除，于是右边同步变小。跑 100 请求那棵树，这个等式在每个阶段都成立——包括全量淘汰之后（$E=0$、$P=2200$、$\text{total}=2200$）。

**root 的 `lock_ref = 1`**（`radix_cache.py:353`）是一个永久哨兵：它让「`lock_ref > 0` ⇒ 不可淘汰」这条规则对所有节点统一，`_update_leaf_status`（`radix_cache.py:795`）永远不会把根收进 `evictable_leaves`，于是 `evict()` 不可能弹到根，`_delete_leaf` 里的 `node.parent.children.pop(...)` 也就不会遇到 `parent is None`。语义上它也说得通：树上的每条路径都经过根，根的引用计数天然至少是 1，只是没人会去减它（`inc/dec` 的循环都在 `node != self.root_node` 处停下）。

这个等式在运维上有一个直接读数：`available_and_evictable_str()`（`base_prefix_cache.py:741`）打印的 `Available tokens` 是 `available_size + evictable_size`，也就是「把所有没人用的前缀都扔掉，你还有多少」。它和 `available_size` 的差，就是树替你存着的**可回收额度**——看着像已用显存，其实是随时能拿回来的。

## 4. evict：只掉叶子，用堆排序

`evict()`（`radix_cache.py:557`）做三件事：把 `evictable_leaves` 快照成一个 `(priority, node)` 堆，反复弹最小值，删掉叶子后看父节点是否变成新叶子。

**为什么必须是叶子**。删掉一个中间节点，等于把它所有子孙的前缀 KV 挖空——`path_device_indices` 走不到根，子孙的 slot 序列就断了。所以只有「没有孩子的节点」才是安全的回收单元；删完一个叶子后，父节点**可能**变成新叶子，这时才把它放进堆（`radix_cache.py:578`）：

```python
if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
    heapq.heappush(eviction_heap, (self.eviction_strategy.get_priority(x.parent), x.parent))
```

一轮能回收多少？**上限是调用时的 `evictable_size_`**，实际按整片叶子结算：一次 `evict(1)` 会弹掉一整片 200 token 的叶子，超额回收 199 个 token。这不是 bug 而是粒度：树的最小可淘汰单位是「一条不再被共享的路径尾巴」，不是「一个 token」。需要更细的粒度时，SGLang 用的是另一个办法——`split_prompt`（`radix_cache.py:490`）：prefill 结束后把 prompt 边界单独 `insert` 一次，人为制造一次分裂，让叶子落在 prompt 与生成段之间，这样「输出 KV」可以被单独淘汰而保留 prompt。

排序键来自 `evict_policy.py`：默认 LRU 直接返回 `last_access_time`（`evict_policy.py:16`）；此外还有 LFU（`(hit_count, time)`）、FIFO（`creation_time`）、`PriorityStrategy`（`(priority, time)`，`evict_policy.py:41`）、把「尾巴」当无限旧的 T-LRU，以及按命中次数分段保护的 SLRU。**堆元素是 `(priority, node)` 元组，先比 priority**；`TreeNode.__lt__`（`radix_cache.py:278`）的实际作用是让 priority 恰好相等时元组还能继续比较，而不是抛 `TypeError`。

谁在调用它？不是调度器直连，而是 allocator：`evict_to_free_tokens()`（`python/sglang/srt/mem_cache/allocator/base.py:192`）算出 `shortfall = num_tokens - available_size()`，缺口为正才 `evict_for_alloc(EvictParams(num_tokens=shortfall))`。**树只是 allocator 的备用钱包**——只有在真的分配不出来时才去掏。

## 5. 命名空间：extra_key、cache_salt 与 bigram 视图

token 相同不等于 KV 相同。SGLang 用两个字符串把树切成互不可见的域：

| 来源 | 字段 | 谁在用 |
|---|---|---|
| LoRA / 调用方自定义 | `extra_key`，`lora_id` 会被拼进去（`schedule_batch.py:572`） | 不同 adapter 的 KV 不能混 |
| 请求级盐 | `cache_salt`（`schedule_batch.py:577`） | 多租户隔离，防侧信道 |

它们的实现方式很硬：不是一个「比较时跳过」的过滤器，而是**直接编进子节点字典键**（`radix_cache.py:246`）：

```python
if self.cache_salt is not None:
    return ((self.extra_key, self.cache_salt), plain)
return plain if self.extra_key is None else (self.extra_key, plain)
```

于是两个命名空间在根下面天然分成两棵子树，`match` 根本走不进去。更保险的是 `_check_compatible`（`radix_cache.py:167`）：如果两边 `extra_key` 或 `cache_salt` 不一致，直接 `raise ValueError`，而不是悄悄返回一个更短的命中。

**`is_eagle` 会换一种 key 视图**。EAGLE 的 draft head 需要「当前 token + 下一个 token」的 bigram 上下文，所以投机解码开启时，key 被翻成 bigram 视图（`maybe_to_bigram_view`，`radix_cache.py:154`）：`len(key)` 变成 $n-1$，`child_key` 变成 `(t[i], t[i+1])` 二元组，`value` 也截到 $n-1$ 个 slot。也就是说，一个覆盖 $n$ 个原始 token 的节点只持有 $n-1$ 个 slot——**最后一个 token 的 KV 永远要重算**，这正是 draft head 需要的那个位置。这跟 vLLM 在 EAGLE 下 `drop_eagle_block` 是同一个动机，只是 SGLang 在 key 的结构里就表达掉了。

## 6. page_size>1：树的粒度被页吃掉

上面所有讨论都假设 `page_size = 1`——这是 CUDA 上的默认值（`overrides.py:1419` 的 `_page_size_default`，默认 `return` 在 `:1440`；函数体里 MUSA 和 HIP 的 `vectorized_5d` 布局会在前面就返回 64，其余平台走到最后一行 `return {"page_size": 1}`）。此外 MLA 后端会把 page_size 顶到 64（flashmla 只认 64，trtllm_mla / cutedsl_mla 只认 32 或 64），FA4 非 MLA 在 sm100 上顶到 128。一旦大于 1，`match_prefix` 的第一件事就是 `key = key.page_aligned(self.page_size)`（`radix_cache.py:411`），把 key **向下截断**到页的整数倍（`radix_cache.py:148`）。

实测一条 2,000 token 的 prompt，在不同 `page_size` 下的可复用长度：

| page_size | 落树后的边长 | 命中长度 | 白扔的 token | `child_key` 形状 |
|---|---|---|---|---|
| 1 | 2000 | 2000 | 0 | `int` |
| 16 | 2000 | 2000 | 0 | 16 元组 |
| 64 | 1984 | 1984 | 16 | 64 元组 |
| 128 | 1920 | 1920 | 80 | 128 元组 |

两条结论：**最多浪费 `page_size - 1` 个 token 的复用**（不是每页浪费，是整条 prompt 的尾部取整）；**分裂只能落在页边界**，因为 `match` 的返回值本身已经被 `// page_size * page_size` 取整（`radix_cache.py:221`）。所以 page_size 越大，树越像「块表」——节点更少、`child_key` 更贵（一个 64 元组的哈希）、命中粒度更粗。这是 SGLang 为了 kernel 效率（连续页、TMA、MLA 的 64 对齐）主动放弃的精度，和 vLLM 用 16 是同一类取舍，只是它把默认值留在了 1。

> 顺带记一笔现状：`radix_cache.py` 这条线正在收敛到 **Unified Radix Cache**（`python/sglang/srt/mem_cache/README.md:41`），Full/SWA/Mamba 三类组件共用一棵树，且 tree core 默认走 Rust 实现。本篇讲的匹配/分裂/加锁/淘汰四件事在那边语义不变，只是从「一个类」变成「组件 + 控制器」。

## 7. 与 vLLM 对照

同一道题，两种解法（vLLM 侧基线 `c7a7607`）：

| 维度 | vLLM（块哈希链） | SGLang（基数树） |
|---|---|---|
| 索引单元 | 16 token 的定长块（`vllm/config/cache.py:185`） | 任意 token 边界（`page_size=1` 时粒度 1） |
| 身份 | 链式哈希 $h_i = H(h_{i-1}, \text{tok}_i, \text{extra}_i)$（`vllm/v1/core/kv_cache_utils.py:709`） | 树上的位置：路径 + `extra_key` + `cache_salt` |
| 查表 | `dict[BlockHash] → block`（`vllm/v1/core/block_pool.py:36`），O(1) | 逐层 dict + 每条边一次段比较，O(depth) 次跳转 |
| 命中 | 逐块 `islice` 查哈希，缺一块就停（`vllm/v1/core/single_type_kv_cache_manager.py:794`） | 走一条边吃掉一整段共享前缀 |
| 表结构 | 扁平二维 `block_table`（`vllm/v1/worker/block_table.py:80`） | 树 + `parent` 指针追逐，行内 slot 由 `path_device_indices` 铺出 |
| 淘汰 | `free_block_queue` 双向链表的 LRU，块级 | `evictable_leaves` + heap，叶子级 |
| 末尾对齐 | `num_computed_tokens` 必须块对齐，最后一个 token 的重算会放大成整块（`vllm/v1/core/kv_cache_manager.py:296`） | `input_len - 1`（`schedule_batch.py:1194`），page_size=1 时只重算 1 个 token |

- **粒度红利是「尾巴」级的**。共享前缀长度不是 16 的倍数时，块结构每请求最多白扔 15 个 token；摊到几万 token 的 prompt 上只有 1–2 个百分点（配套脚本 [`code/sglang/radix_tree_sim.py`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang) 在同一负载上对拍，块比树多算约 2%）。真正拉开差距的是「每轮只追加几个 token」的场景：增量越小，取整的代价占比越高。
- **查找成本反过来差一个数量级，但要看清比的是什么**。树每下行一层吃掉一整段共享前缀，深共享负载下只访问几十个节点；块则要沿整条链逐块查——8k prompt 就是 512 次字典查找（哈希值在请求侧增量算好，`vllm/v1/request.py:290`，所以贵的是 512 次跳转而不是 512 次哈希）。代价是树每次跳转都要解引用 Python 对象、每条边都要过一遍 token 比较，而哈希是一次 dict 查找——**共享越深树越便宜，共享越浅树的额外开销越纯亏**（`rag` 这类无共享前缀的负载，树命中率 0%，只剩开销）。
- **生命周期复杂度不同**。块是不可变的：满了就打哈希，引用计数归零即可回收。树节点是可变的：会分裂、会摘除、`value` 会被切两半，所以需要 `lock_ref` 沿路径累加、需要 `evictable_leaves` 这个派生集合、需要 $E+P=\text{total}$ 这条不变量来兜底。这是「结构共享」的固定成本，不是实现瑕疵。

一句话：**vLLM 把「均匀」当一等公民（块定长、表扁平、淘汰简单），SGLang 把「共享结构」当一等公民（树、分裂、路径锁）；前者赢在常数和可预测性，后者赢在粒度与深共享下的查找量。**

## 8. 常见坑

- **改了系统提示一个空格，整棵树白长**。key 是 token 序列，tokenizer 版本、chat template 里的空白、特殊 token 都会改变序列；这不是「命中率下降」，是「根下面换了一棵子树」。
- **别再以为 `evict()` 能救急**。它只能回收 `evictable_size_`，也就是没有任何在跑请求锁住的路径。所有前缀都在被用着时，一轮 `evict()` 返回 0，调度器只能 retract（那是[调度器篇]({{< relref "sglang-04-scheduler" >}})的话题）。
- **锁是路径级的，释放也要走同一条路**。`dec_lock_ref` 传错节点（比如传了一个已经分裂出来的新节点）会一路减到别人的路径上；代码里用 `node.parent is None` 断言「这个节点不属于这棵树」（`radix_cache.py:616`）来兜底。
- **`split_prompt` 会多一次 insert**。它故意用同一个 prompt 前缀再插一次来造分裂，看起来像重复劳动，实际是给淘汰粒度做手术。不要以为那是冗余代码。
- **多租户别只靠 token 相同**。不开 `cache_salt` 时，A 租户的系统提示和 B 完全一样就会共享节点——省显存，但也是侧信道。要不要开，取决于威胁模型。
- **`page_size` 不是性能参数，是语义参数**。它同时改变命中粒度、分裂位置和 `child_key` 的哈希成本，还会被 attention 后端强制覆盖（MLA 64 / FA4 128）。调它之前先确认后端。

## 读代码时的检查点

1. 打开 `radix_cache.py:656`，把 `_match_prefix_helper` 的 `while` 循环读三遍：**什么时候下探、什么时候分裂、什么时候跳出**。这是整棵树的心跳。
2. 打开 `radix_cache.py:682`，确认 `_split_node` 只改 `key/value/parent/children/lock_ref/hit_count` 这些结构字段、只 `clone` 下标。如果哪天它开始搬 KV，那一定是把 HiCache 的 load_back 混进来了。
3. 打开 `radix_cache.py:587` 与 `radix_cache.py:602`，把 $E+P=\text{total}$ 在纸上推一遍，再去 `mem_cache/common.py` 看 `match_kv_cache`（`mem_cache/common.py:253`）和 `_commit_prefill_admission`（`schedule_policy.py:1694` 的 `req.lock = tree_cache.lock(...)`）——**锁在哪里拿、在哪里放**，决定了这套账本能不能守住。

## 小结

- 树节点存的是**一段 token + 一段 slot 下标**，不是 KV；路径拼起来才是完整 KV，`insert` 之后重复的 slot 被释放，省显存就发生在这里。
- `match_prefix` 是「逐层 dict + 每条边一次段比较」，靠指数搜索把 2,000 token 的边压成约 22 次 Python 跳转；命中终点落在边内部时 `_split_node` 就地插一层，只复制 int64 下标（17.6 KB vs 281 MiB）。
- `lock_ref` 沿路径累加，锁定一个叶子就保护了它整条祖先链；$E+P=\text{total}$ 是这套账的不变量，root 的 `lock_ref=1` 让规则统一。
- `evict()` 只能删叶子（否则子孙的前缀就断了），用 heap 按 LRU 排序，删完把可能变成叶子的父节点塞回堆；一轮上限是 `evictable_size_`，粒度是整片叶子。
- `extra_key` / `cache_salt` 编进子节点键，命名空间之间物理不通；`is_eagle` 把 key 换成 bigram 视图，节点持有的 slot 比覆盖的 token 少一个。
- `page_size>1` 把 key 向下取整到页，最多白扔 `page_size-1` 个 token 的复用，分裂也只能落在页边界。
- 与 vLLM 的分野不是「谁更快」，而是**把「共享结构」还是把「均匀性」当一等公民**。

下一篇算另一本账：这些 slot 到底从哪来、一个 token 值多少字节、`mem_fraction_static` 现在是怎么自动推出来的——[KV 池、slot 分配器与显存账本]({{< relref "sglang-03-memory-pool" >}})。

> 代码基线：`sgl-project/sglang` main @ `1ef3335`（2026-11-03）。对比基线：`vllm-project/vllm` main @ `c7a7607`。
> 论文：RadixAttention 见 <https://arxiv.org/abs/2312.07104>。
