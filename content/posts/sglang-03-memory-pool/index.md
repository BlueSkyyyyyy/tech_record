---
title: "SGLang 研读（三）KV 池、slot 分配器与显存账本"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 4
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "KV Cache", "显存", "PagedAttention"]
categories: ["推理框架"]
---

上一篇里基数树的每个节点都捏着一串 `value`，那些 value 就是 **slot id**：显存里某一层 K/V 张量的行号。这篇把 slot 的另一头补齐——池子多大、slot 从哪儿发、发出去之后谁记账。前置阅读：[RadixAttention：基数树与前缀缓存]({{< relref "sglang-02-radix-attention" >}})。

## 1. 先记住一个画面

`memory_pool.py` 里最核心的是两个形状完全不同的对象，它们只通过"整数 id"耦合：

```text
req_to_token  (GPU, int32, [max_running_requests + 1, max_context_len])
                位置:  0    1    2    3   ...  8191
  req row 17:       [ 1,   9,  33,  34,  ...  ]    ← 每格是「某层 K/V 张量的第几行」
  req row 18:       [ 1,   9,  40,  41,  ...  ]    ← 与 17 共享前两个 slot（前缀命中）
k_buffer[layer] / v_buffer[layer]  (bf16, [max_total_num_tokens + page_size, kv_head, head_dim])
```

一个二维表把「请求 × 位置」映射成 slot；每层一张大表按 slot 寻址；slot 由分配器从一段连续整数里发。整篇文章就是把这三句拆开：第 2–4 节讲池子多大，第 5–7 节讲 slot 怎么发，第 8 节讲谁先谁后。

## 2. `mem_fraction_static`：从手填常数到反推

早期这个参数是个手填常数（0.9 一类），OOM 了就下调。main 上它的默认值已经是 `None`，由 `handle_gpu_memory_settings` 在启动时反推（`arg_groups/memory_hook.py:283`）。思路写在 `:52-71` 的 docstring 里：它定义的是「权重 + KV 池」占总显存的比例，等价于先把激活和 CUDA Graph 的 buf 估出来、再从总量里扣掉（若开了 post-capture KV sizing，这一项只剩 `1536 + tp·pp/8·1024`，因为图捕获完会直接量剩余显存，见 `:293-297`）。所以要估的其实是 `reserved_mem`：

$$
\text{reserved} = 512 + 1.5 \times T_{\text{act}} + \frac{tp \cdot pp}{8} \times 1024 + R_{\text{graph}} + R_{\text{deepep}}\quad (\text{MiB})
$$

逐项对着代码看（示例：单卡 80 GiB、TP=1、Llama-3-8B、bf16）：

| 项 | 代码位置 | 示例值 (MiB) | 是什么 |
|---|---|---|---|
| `512` | `:311` | 512 | attention backend 等的常量元数据 |
| `1.5 × T_act` | `:312` | 12,288 | `T_act = max(chunked_prefill_size, 2048)`（`:299-309`），本例 8192；按 1.5 MB/token 估激活水位 |
| `tp·pp/8 × 1024` | `:314` | 128 | "每 8 路并行多留 1 GB"，`/8*1024` 就是把 GB 写成 MB |
| `R_graph` | `:315` → `:359` | 976 | decode 图 `max_bs×2`（`:370`，本例 256×2）；prefill 图 `len(bs_list)×8`（`:388`，本例 58×8） |
| 下限 | `:316-317` | 10,240 | 显存 > 60 GiB 的卡至少留 10 GiB；本例 13,904 已超过，不生效 |
| `R_deepep` | `:319` → `:404` | 0 | `moe_a2a_backend=deepep` 且开 decode 图时再 +2 GiB（`:415`） |

合计 13,904 MiB，于是 `round((81920 − 13904) / 81920, 3) = 0.830`（`:326-330`）。

**这段代码里有两套单位在打架**，读的时候必须自己换算：

1. docstring（`:68`）写的是 "estimate the size of activations and cuda graph buffers **in GB**"，实际代码全按 MiB 累加。按字面读会大 1024 倍。
2. `gpu_mem` 来自 `nvidia-smi` 的 `memory.total`（MiB，`utils/common.py:990`、`:802`），而 `available_gpu_memory` 来自 `torch.cuda.mem_get_info` 再除 `1 << 30`（GiB，`utils/common.py:576`）。同一条启动路径上两个量词；`tp·pp/8*1024` 一个表达式里同时有 GB 和 MB，阈值 `gpu_mem > 60 * 1024` 与 `10 * 1024` 也是"把 60 GB / 10 GB 写成 MB"。

## 3. 从 GiB 到 token：`max_total_num_tokens`

容易误读的一点：`mem_fraction_static` **不是**"KV 占多少"，而是"静态部分（权重 + KV）占多少"。真正的 KV 预算要减两次（`mem_cache/kv_cache_configurator.py:2884` 起）：

```python
slack_gb = pre_model_load_memory * (1 - mem_fraction_static)        # :2901
rest_memory = available_gpu_memory - slack_gb - mm_reservation_gb   # :2932
```

也就是"现在还剩多少 − 总显存里该留给激活的那一份"。权重已经加载完，自然被扣在 `available_gpu_memory` 里。`rest_memory` 转成字节后除以每 token 字节数，再按页对齐（`model_executor/pool_configurator.py:658-667`）：

```python
max_total_num_tokens = available_bytes // self._cell_size                    # :663
max_total_num_tokens = max_total_num_tokens // page_size * page_size         # :667
```

`cell_size` 就是"一个 token 在所有层上的 KV 字节"，两条公式对应两类模型：

$$
\begin{aligned}
\text{cell}_{\text{MHA}} &= n_{kv} \times (d_{\text{head}} + d_{\text{vhead}}) \times L \times s_{kv}\\
\text{cell}_{\text{MLA}} &= (r_{\text{lora}} + d_{\text{rope}}) \times L \times s_{kv}
\end{aligned}
$$

MHA 那条在 `pool_configurator.py:513-519`，$n_{kv}$ 已经按 TP 除过、按 DCP 乘过；MLA 那条在 `:383-391`，存的是 latent + rope，**不随 TP 缩小**。两个具体例子（同一张 80 GiB 卡、同一个 0.83，只换模型）：

| 模型 | 每层每 token | 层数 | 每 token | 1 GiB 装多少 token |
|---|---|---|---|---|
| Llama-3-8B：8 个 KV head，(128+128) 维，bf16 | 4,096 B | 32 | 131,072 B = 128 KiB | 8,192 |
| DeepSeek-V3 式 MLA：512+64 latent，bf16 | 1,152 B | 61 | 70,272 B = 68.6 KiB | 15,280 |

代入账本（假设权重 8.03B×2 B = 14.96 GiB、CUDA context 约 0.6 GiB）：

- **MHA 8B**：`80 − 14.96 − 0.6 = 64.44 GiB` 可用，减掉 `slack = 80 × 0.17 = 13.6 GiB`，KV 池 **50.84 GiB**，即 `50.84 × 8192 ≈ 416,505` token，约 **41 万**。
- **MLA**：同样 50.84 GiB，除以 70,272 B 得 `776,870` token，约 **78 万**。

一个反直觉的结论：671B 的 DeepSeek 每 token 的 KV 只有 8B 的 Llama 的一半多（68.6 KiB vs 128 KiB）。如果 DeepSeek 用同样 128 head 的 MHA（每层每 token 64 KiB × 61 层 = 3.81 MiB），是 MLA 的 **57 倍**。

## 4. 两张表：`ReqToTokenPool` 与 `TokenToKVPool`

`ReqToTokenPool`（`mem_cache/memory_pool.py:379`）就是第 1 节那张二维表：

```python
self._alloc_size = size + 1     # :413  多一行给 padded batch（req_pool_indices 默认 0）
self.req_to_token = torch.zeros((self._alloc_size, max_context_len), dtype=torch.int32, ...)  # :417
self.free_slots = list(range(1, self._alloc_size))                                            # :420
```

`size` 是请求数（`max_num_reqs`），`max_context_len` 是 `context_len + 4 + draft`（`kv_cache_configurator.py:1557`）。注意 `free_slots` 是 Python list，`alloc_rows` 从**尾部** pop（`:458-459`，注释解释了为什么不从前缀 pop：`free_slots[-0:]` 是整个列表）。

KV 池则是"每层一个张量"，形状在 `_kv_buffer_shapes`（`:2537-2548`）：

```python
rows = self.size + self.page_size
return ((rows, self.head_num, self.head_dim), (rows, self.head_num, self.v_head_dim))
```

多出来的 `page_size` 行是 padded token 的落点（`:2557` 的注释：slot 0 所在的那一页吸收 dummy 写）。MLA 只有一条：`[size + page_size, 1, kv_cache_dim]`（`:4759`），K/V 融合成 576 维 latent。

**为什么必须是两张表**：行索引的语义完全不同。前者的行是"一个请求"，后者的行是"一个 slot"，而 slot 在 radix 命中时是**跨请求共享**的——两个请求命中同一段前缀 slot，但在 `req_to_token` 里各占一行。硬合并就得为共享关系做二次索引，反而更慢。

新 slot 写回表也值得看一眼：`mem_cache/allocation.py:68` 的 `write_cache_indices` 用一个 Triton kernel 一次性写整个 batch 的行段；只有非 Triton 后端才退回"每请求两次 slice 赋值"的循环（`:102-113`），代价是每请求几次 `.item()` 同步。量级：4096 并发 × 8192 上下文 × int32 = **128 MiB**，还只是元数据。

## 5. `page_size = 1`：`TokenToKVPoolAllocator`

`mem_cache/allocator/token.py` 全文 96 行，一个类和它的 6 个方法就完了：

```python
def clear(self):                                             # :42
    self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, ...)   # :44

def alloc(self, need_size):                                  # :54
    if need_size > len(self.free_pages):
        return None              # need_sort 时先 merge_and_sort_free() 再判一次
    select_index = self.free_pages[:need_size]     # :61  前缀切片，O(need_size)
    self.free_pages = self.free_pages[need_size:]  # :62
    return select_index
```

"free slot 栈"其实是一个 **int64 张量**，`alloc` 从头部切一段、`free` 拼回尾部。slot 0 永远不参与分配（`arange(1, ...)`），留给 padded token。三个细节决定它在真实负载下的表现：

- **`need_sort`**：PD 分离时打开（`kv_cache_configurator.py:2701`）。此时 free 不直接回池，而是进 `release_pages`（`token.py:71`），等池子不够时才 `merge_and_sort_free` 排序合并（`base.py:270`）。省掉每次 free 的 `torch.cat` + 排序，代价是池子"看起来"比实际小。
- **`free_group`**：`base.py:256-264` 的批量延迟。SWA 逐请求淘汰时一轮几十次 free，每次 `torch.cat` 是 O(n)，累计 O(n²)；`free_group_begin/end` 把它们攒成一次 cat。`_copy_for_free_group`（`:266`）里的 `.clone()` 是为了防止调用方随后改写那个 view。
- **`available_size`**：`len(free_pages) + len(release_pages)`（`token.py:50`）。`page_size=1` 时单位就是 token；基类版本乘了 `page_size`（`base.py:236`）。

碎片账：一个 token 一个 slot，**没有内部碎片**，代价是元数据最贵——每 token 一个 int32。

## 6. paged：`PagedTokenToKVPoolAllocator`

`allocator/paged.py:136` 起。形状上的区别只有一句：`free_pages` 存的是 **page id**（`arange(1, num_pages+1)`，`:397`），`num_pages = size // page_size`（`:153`），第 0 页同样留作 padding。

```python
def alloc(self, need_size):                          # :181
    assert need_size % self.page_size == 0           # :184（debug_mode 下）
    out_pages = self.free_pages[:num_pages]          # :194  num_pages = need_size // ps (:188)
    out_indices = (out_pages[:, None] * self.page_size + torch.arange(self.page_size, ...))
    return out_indices.reshape(-1)                   # :197-200
```

`alloc_extend` / `alloc_decode` 是批量的，一次算完整个 batch 的 slot。这里有个不得不付的成本：先用**上界**判断池子够不够（`max_new_pages = extend_num_tokens // page_size + bs + 1`，`:222`），不够再用 CPU 张量算精确值（`:233`）。注释（`:226-231`）说得很直白：两个 kernel 在 `free_pages` 不够时都会读越界（Triton 那个真越界，native 那个 clamp 出重复行），所以宁可 sync 一次。

`free` 才是 page 路径的真正代价——一次 device 端 `unique(free_index // page_size)`（`:333`，内部是 sort），把散落的 token id 折算成去重后的 page id。ROCm 上首次还要 JIT rocPRIM，所以 `:159-165` 专门预热了一次。`free_segment`（`:340`）绕开它的办法很巧：起点页对齐、页内 token 连续，于是 `free_index[::page_size]` 正好每页取一个，直接 `//page_size` 就是 page id，不需要 `unique`。

| 维度 | `page_size=1`（`token.py`） | paged（`paged.py`） |
|---|---|---|
| 分配单位 | 1 个 slot | 1 页（`page_size` 个 slot） |
| 空闲池元素 | `arange(1, size+1)` | `arange(1, num_pages+1)`，值是 page id |
| `alloc` 返回 | 直接是 slot id | `page × ps + 页内偏移` 展开 |
| `free` 成本 | 一次 `cat` | `unique(free_index // ps)`（device 排序） |
| 内部碎片 | 0 | ≤ `ps − 1` / 请求 |
| `free_segment` | 退化成 `free` | `[::ps]` 免 `unique` |
| 何时走这条 | `page_size == 1` 且 DCP 关（`kv_cache_configurator.py:2811`） | 其余全部 |

## 7. 谁先谁后：fraction → token → 请求数

启动期是一条单向链，顺序不能反：

1. `mem_fraction_static` 反推（`memory_hook.py:283`，此时还没加载权重）；
2. 加载权重、量剩余显存，算 `rest_memory`（`kv_cache_configurator.py:2884`→`:2932`）；
3. 除 `cell_size` 得 `max_total_num_tokens`（`pool_configurator.py:663`）；
4. 由 token 数反推请求数（`kv_cache_configurator.py:3024-3037`）：`estimated = int(token_capacity / context_len * 512)`，夹到 `[2048, 4096]`（`:3028-3029`），再取 `min(estimated, token_capacity // 2)`（`:3037`）；
5. `ReqToTokenPool` 按第 4 步的大小开表。

注意那个 `* 512`：意思是"每装满一个上下文长度的 token 配额，就配 512 个请求位"，然后被 4096 硬顶。所以短上下文的 8B 模型无论卡多大都是 4096，128k 上下文的模型会掉到 2048 附近——**表大小被上下文长度和 token 池共同决定，不是配置项**。

调度期则有两道闸，先撞上的是 token：

- **token 闸**：`PrefillAdder`（`managers/schedule_policy.py:609`）的预算 = `allocator.available_size() + tree_cache.evictable_size()`（`mem_cache/prefill_budget.py:111`），这是准入的主判据。
- **行闸**：`scheduler.get_num_allocatable_reqs` 的 `row_budget = req_to_token_pool.available_size() + num_held_rows - beam_rows`，最后 `res = min(pp_budget, row_budget)`（`managers/scheduler.py:4229-4244`）。

结论：**吞吐上限由 token 数卡住**，请求数只是硬天花板。`--max-running-requests` 设得比 `tokens // 2` 大不会生效（`:3034` 会压回来），设得比它小才真的限制并发。

## 8. 与 vLLM 对照

先说个容易过时的点：vLLM main（`c7a7607`）的 `gpu_memory_utilization` 默认值已经**不是 0.9**，而是 `None` → `0.92`（`vllm/config/cache.py:216`、`:186`、`:489`），开 extensible KV cache 时干脆给 1.0。但它仍然是**总显存的一个固定比例常数**，与 chunk 大小无关：

```python
requested_memory = math.ceil(init_snapshot.total_memory * gpu_memory_utilization)  # v1/worker/utils.py:750
available_kv = requested_memory - non_kv_cache_memory - cudagraph_estimate          # v1/worker/gpu_worker.py:824
num_blocks = available_memory // bytes_per_block                                    # v1/core/kv_cache_planning.py:700
```

四条实质差异：

1. **常数 vs 反推**。vLLM 的 0.92 是"你告诉我留多少"，OOM 了自己调；SGLang 的 0.83 是"按 `chunked_prefill_size` 和 graph `max_bs` 算出来的"，换卡换 TP 自动变。前者的风险是用户不知道该填多少，后者的风险是启发式错一个系数就直接 OOM——而且它假设整张卡只跑这一个进程。
2. **实测一次 vs 算术一次**。vLLM 用 dummy forward 实测 `non_kv_cache_memory`（`gpu_worker.py:758-828`）；SGLang 默认路径纯算术（只有 post-capture 分支才去量）。算术便宜、可预测，实测更准、启动更慢。
3. **页粒度 16 vs 1**。vLLM 的 `block_size` 默认 16（`cache.py:185`），块表形状 `[max_num_reqs, max_num_blocks_per_req]` int32（`v1/worker/block_table.py:81`）；SGLang 默认 `page_size=1`（`arg_groups/overrides.py:1440`；MLA 类后端被强制 64（`:1193` 起），FA4 非 MLA 在 SM100 上 128）。同一份 4096 并发 × 8192 上下文：SGLang 的 `req_to_token` 要 128 MiB，vLLM 的块表只要 8 MiB——正好 16 倍。
4. **结果**。SGLang 的 KV 利用率能贴着池子上限走（一个 token 都不用凑整），代价是元数据大 16 倍、每 token 多一次查表；vLLM 的块表紧凑、kernel 寻址单位更大，代价是每请求最多 15 个 token 的槽位浪费。不过 vLLM 也在往"测完再定"走：`enable_extensible_kv_cache`（`cache.py:389` 附近）用 VMM 预留地址空间、捕获完 graph 再 commit 物理页，逻辑上和 SGLang 的 post-capture 分支是同一件事。

## 9. 常见坑

- **把 `mem_fraction_static` 当成 KV 占比**。它是"权重 + KV"的静态占比，0.83 里权重已经吃掉 15 GiB。手填 0.9 看着只多 7%，实际是把 5.6 GiB 的激活余量吃光了。
- **手填 `--mem-fraction-static` 后调大 `chunked_prefill_size`**。自动值会跟着 chunk 变大而变小（`1.5 MB/token` 那一项），手填值不会——这是"改了 chunk 就 OOM"最常见的原因。
- **`req_to_token` 随 `--context-length` 线性涨**。4096 × 131072 × 4 B = 2 GiB，长上下文下它比很多模型的 KV 池还显眼，而且是按 `max_context_len` 一次性 `zeros` 出来的。
- **`page_size > 1` 时 `--max-total-tokens` 被悄悄取整**。`// page_size * page_size`（`pool_configurator.py:667`）不报错，日志里的数和你填的不一样。
- **`free_segment` 的 `start_pos` 必须页对齐**（`base.py:333` 的 assert），PD 分离和 SWA 每轮释放都依赖这条契约。
- **paged 路径上的重复释放更隐蔽**。`free` 会 `unique(free_index // ps)`，重复释放同一个 slot 不报错，只会让 `free_pages` 里出现重复 page；要等 `SGLANG_DEBUG_MEMORY_POOL=1`（`paged.py:154`）才 assert 出来。

## 10. 读代码时的检查点

1. 打开 `arg_groups/memory_hook.py:311`，把 `reserved_mem` 每一项的单位标出来，再用 `nvidia-smi` 的总显存手算一次 frac，对一下启动日志里那个 `0.83x`——对不上就是走了你没注意的分支（post-capture、DeepEP、VLM）。
2. 在 `model_executor/pool_configurator.py:342` 的 `_compute_cell_size` 里确认模型走 MLA 还是 MHA 分支，手算 `cell_size`，用日志里 `KV size / #tokens` 的比值验一遍。
3. 在 `kv_cache_configurator.py:2811` 确认你的 `page_size` 走哪条 allocator；再看 `allocator/paged.py:222` 那个上界，就明白为什么长 prefill 偶尔会突然 sync 一下。

## 小结

- `mem_fraction_static` 从手填常数变成反推：`reserved = 512 + 1.5·T_act + tp·pp/8·1024 + graph + deepep`（MiB），80 GiB 单卡 8B 模型算出 0.830。
- 这段代码 MB/GB 混用：docstring 写 GB、代码按 MiB 加，`gpu_mem` 是 MiB 而 `available_gpu_memory` 是 GiB。
- KV 预算 = `available − total·(1 − frac)`，再除每 token 字节数：MHA 是 `n_kv·(d_head+d_vhead)·L·s`，MLA 是 `(r_lora+d_rope)·L·s`（不随 TP 缩小）。
- 8B MHA：128 KiB/token，50.84 GiB 池 → 约 41 万 token；61 层 576 维 MLA：68.6 KiB/token → 约 78 万 token。后者每 token 只有前者的 1/1.9，却是 671B 模型。
- 两张表各司其职：`req_to_token`（请求 × 位置 → slot，int32，128 MiB）和每层 KV 张量（slot × head × dim，`size + page_size` 行）。
- `page_size=1` 用张量切片 + `free_group` 批延迟；paged 用 `unique(//ps)` + 页对齐，`free_segment` 用 `[::ps]` 免排序。顺序是 `mem_fraction_static → max_total_num_tokens → max_running_requests(min(估计, tokens/2))`，调度期 token 闸先于行闸。

下一篇讲这些 token 和 slot 怎么被排成 batch：[调度器：PrefillAdder、chunked prefill 与 retract]({{< relref "sglang-04-scheduler" >}})。

> 代码基线：`sgl-project/sglang` @ `1ef3335`（[`mem_cache/memory_pool.py`](https://github.com/sgl-project/sglang/blob/1ef3335da1142c19853996d93c5c1b8bc0e84b58/python/sglang/srt/mem_cache/memory_pool.py)、[`arg_groups/memory_hook.py`](https://github.com/sgl-project/sglang/blob/1ef3335da1142c19853996d93c5c1b8bc0e84b58/python/sglang/srt/arg_groups/memory_hook.py)）；对照基线 `vllm-project/vllm` @ `c7a7607`。
