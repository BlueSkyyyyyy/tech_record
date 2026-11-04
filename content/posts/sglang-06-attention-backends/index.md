---
title: "SGLang 研读（六）Attention 后端：extend 与 decode 两条路径，以及 page table 怎么传"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 7
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "Attention", "FlashInfer", "MLA", "KV Cache"]
categories: ["推理框架"]
---

启动日志里有一行很不起眼的话：`Attention backend not specified. Use flashinfer backend by default.`（`arg_groups/overrides.py:1183-1185`）。它背后的决策树比这句话长得多：卡是 Hopper 还是 Blackwell、模型是 MHA 还是 MLA、开没开投机解码、`page_size` 是不是 1，每一个都会换掉答案。

更反直觉的是下一层：SGLang 允许你给 prefill 和 decode **指定两个不同的后端**，然后在运行时把它们拼成一个 `HybridAttnBackend`（`model_executor/model_runner_components/attention_backend_setup.py:202-230`），并按 forward mode 转发。也就是说，"attention backend"在这套代码里不是一个类，而是"一组按相位切开的类"。

这一篇接着[第五篇的两级 overlap]({{< relref "sglang-05-overlap" >}})往下走：图捕获好了，模型每层要调的那个 attention 到底拿到什么参数。前置阅读是 [KV 池与 slot 分配器]({{< relref "sglang-03-memory-pool" >}})，因为本文一半的篇幅在讲索引。

要回答四个问题：

1. `init_forward_metadata` 一次调用，要为 kernel 准备几组 index 张量？flashinfer 路径下 extend 和 decode 各要哪几个？
2. 为什么 SGLang 把 `forward_extend` 和 `forward_decode` 分成两个抽象方法，而 vLLM 只有一个带 metadata 的 `forward`？
3. `out_cache_loc` 和 `req_to_token` 是什么关系——一个 token 的两个坐标分别由谁写、谁读？
4. 投机解码的树状 mask，为什么会把某些后端直接排除掉？

## 1. 一个后端 = 三条入口 + 两个名字

`AttentionBackend` 是抽象基类（`layers/attention/base_attn_backend.py:81`）。它给模型层暴露的入口只有三条：

- `init_forward_metadata(fb)`：每个 forward **调一次**，给本层之后所有 layer 复用。默认实现是"图外 + 图内"两半的拼接（`base_attn_backend.py:153-159`）；
- `forward_extend(...)`（`:443`）：prefill / chunked prefill / target verify / draft extend / dLLM extend 共用；
- `forward_decode(...)`（`:430`）：只有 `ForwardMode.DECODE`。

模型侧只认 `forward` 一个名字：`RadixAttention.forward` 末尾调 `get_attn_backend().forward(...)`（`layers/radix_attention.py:190-192`），基类的 `forward` 再做 mode 分发（`base_attn_backend.py:385-428`）——`is_idle()` 返回空张量，`is_decode()` 走 decode，其余全落到 extend。

关键在 `is_extend()` 的定义（`model_executor/forward_mode.py:46-52`）：`EXTEND`、`MIXED`、`TARGET_VERIFY`、`DLLM_EXTEND`（以及可选的 `DRAFT_EXTEND_V2`）都算 extend。**"extend"不是一个相位，是一族**：只要 query 侧不止一个 token、或者要读一段已经算好的 prefix，它就走 extend 这条路。

第二个名字来自 CUDA Graph 的切分。`init_forward_metadata` 默认拆成两半：`init_forward_metadata_out_graph(fb, in_capture=False)`（`:171`）在图外用 host 侧逻辑跑，`init_forward_metadata_in_graph(fb)`（`:193`）在图内录一段静态 shape 的 GPU op，replay 时自动重放。这条切线的约束写得很直白：in-graph 的 body **不许** `.item()` / `.cpu()` / `.tolist()` / 动态 shape 的 `torch.empty()`（`:199-201`）。后面所有"每步要建张量"的问题，本质都是"这段逻辑能不能塞进图里"。

## 2. 谁被选中：注册表 + 一条 if 链

后端名到工厂函数的映射是一张普通 dict（`layers/attention/attention_registry.py:45`），用 `@register_attention_backend("flashinfer")` 挂上去（`:50-55`）。`--attention-backend` 的合法值就是 `arg_groups/choices.py:69-100` 那张表，共 26 个（含 3 个别名：`nsa`→`dsa`、`compressed`→`dsv4`、`tokenspeed_mla`→`cutedsl_mla`，改写见 `arg_groups/serving_hook.py:301-302` 与 `attention_registry.py:198-206`）。工厂函数顺便承担了"这个后端能不能跑这个模型"的断言：`flashinfer` 按 `use_mla_backend` 分流到 MHA 类或 MLA 类（`:80-102`），`fa3` 断言 `SM80（非 MLA）或 SM90`（`:295-302`），`trtllm_mha` 拒绝 MLA 模型（`:342-343`）。

默认值不在注册表里，在 `arg_groups/model_override_base.py:281` 的 `get_default_attn_backend`。docstring 本身就是规格说明（`:282-294`）：

| 模型 | 硬件 / 条件 | 默认 |
|---|---|---|
| MHA | Hopper 且 `topk<=1` | `fa3`（`:309-315`） |
| MHA | SM100 且 `topk<=1` | `trtllm_mha`，K/V 行宽不等则 `fa4`（`:316-329`） |
| MHA | HIP / MPS / 其他 | `aiter` / `torch_native` / `flashinfer`（有 sink 时退回 `triton`）（`:330-338`） |
| MLA | Hopper / SM100 | `fa3` / `flashinfer`（`:341-344`） |
| MLA | HIP（KV 头 16 或 128）/ 其他 | `aiter` / `triton`（`:345-355`） |

注意那个 `topk<=1` 是怎么判的：`is_no_spec_infer_or_topk_one`（`utils/common.py:4002-4008`）返回真，当且仅当 `speculative_eagle_topk is None`，或者 `topk == 1 且 page_size == 1`。**投机解码的 topk 和 page size 一起参与"选后端"**，这是后面第 6、7 节所有冲突的源头。

## 3. `init_forward_metadata` 要准备几组索引

以 flashinfer 为例。一次调用（`flashinfer_backend.py:1006`）走三个分支：`is_decode_or_idle()` → decode updater；`is_target_verify()` → verify 用的 prefill updater；其余 → 正常 extend updater。产物是两个不同的 metadata 类：`DecodeMetadata`（只有 `decode_wrappers`，`:163`）和 `PrefillMetadata`（`prefill_wrappers` + `use_ragged` + `extend_no_prefix`，`:170`）。

索引张量本身在 `__init__` 里就按 `max_bs` 预分配好了，每步只做原地填充：

| 张量 | extend | decode | 含义 |
|---|---|---|---|
| `qo_indptr` | 需要 | 不需要 | 每个请求本批 query token 数的前缀和；decode 恒为 1，由 `bs` 隐含 |
| `paged_kv_indptr` | 需要 | 需要 | 每个请求要读的 KV slot 数的前缀和（CSR 的行指针） |
| `paged_kv_indices` | 需要（有 prefix 时） | 需要 | 逐 token 的物理 slot 列表（CSR 的列下标） |
| `paged_kv_last_page_len` | 需要 | 需要 | 每行最后一页有几个 token；`page_size=1` 时恒为 1 |
| `custom_mask` / `mask_indptr` | 树 verify 时要 | 否 | 树状注意力掩码 |

分配点：`kv_indptr` 是 `(max_bs+1,)` int32（`:490-497`），`kv_last_page_len` 是 `(max_bs,)` 的 **ones**（`:509`），`qo_indptr` 是 `(max_bs+1,)`（`:518`）。两条路径各有一个 updater（`FlashInferIndicesUpdaterDecode:1615`、`FlashInferIndicesUpdaterPrefill:1922`），但填法几乎一样：

```text
decode:  kv_indptr[1:bs+1] = cumsum(seq_lens)            # :1828
         kv_indices = pack_read_stream(req_to_token)     # :1839
extend:  kv_indptr[1:bs+1] = cumsum(paged_kernel_lens)   # :2279
         qo_indptr[1:bs+1] = cumsum(seq_lens - prefix_lens) # :2299
         kv_indices = pack_read_stream(req_to_token)     # :2290
```

用公式写就是每行的宽度：

$$
\Delta \text{qo\_indptr}_i = E_i,\qquad \Delta \text{kv\_indptr}_i = L_i
$$

$E_i$ 是该请求本批的新 token 数（extend 长度），$L_i$ 是它要读的历史长度。decode 时 $E_i \equiv 1$，所以 `qo_indptr` 直接省掉。

所以准确的说法是：**4 个张量、一份按 `max_bs` 预分配的缓冲，decode 少用一个 `qo_indptr`。** 真正值得记的差别是最后一个：extend 可以选择"根本不建页表"。`use_ragged` 为真且 `extend_no_prefix` 为真时（没有任何 cached prefix），走 `prefill_wrapper_ragged.forward`，把 `k`/`v` 当稠密张量直接喂进去，完全不需要 `kv_indices`。有 prefix 时则拆成两半：ragged 算新 token、paged 算 prefix，再用 `merge_state` 合并 logits（`flashinfer_backend.py:1507-1528`）。

## 4. 为什么 extend 和 decode 是两个方法

一句话：**两者的 kernel 签名不一样，而 SGLang 选择让这个差异暴露在抽象层**。

ragged wrapper 的 `begin_forward` 长这样（`flashinfer_backend.py:2321-2327`）：

```python
wrapper_ragged.begin_forward(qo_indptr, qo_indptr, num_qo_heads, num_kv_heads, head_dim, ...)
```

两个参数都是 `qo_indptr`：Q 和"本批 K/V"用的是同一套 prefix sum，因为新 token 的 K/V 就在 `k`/`v` 里，稠密、连续。而 paged wrapper 要四个（`:2425-2432`）：`qo_indptr, kv_indptr, kv_indices, kv_last_page_len`，外加 positional 的第 8 个参数——**写死的 `1`**（page_size）。

这暴露了 SGLang 传 page table 的核心手法：**它把页表压平成了 token 粒度的 CSR，并且在两处 `begin_forward` 里都把 page_size 写死成 `1`**，物理分页由 SGLang 自己的 `req_to_token` 和分配器负责。所以 `kv_indptr` 是 token 数的前缀和，不是页数的前缀和。好处是后端不必理解 SGLang 的分页规则，代价是索引数组的长度是 $\sum L_i$ 而不是 $\sum \lceil L_i/P \rceil$（第 6 节算这笔账）。

第二条理由更硬：extend 里 K/V 的写入和读取有先后依赖。ragged 路径下，paged 那一半只覆盖 prefix（`flashinfer_backend.py:1998-2000` 的 `paged_kernel_lens = prefix_lens`），因为"新 token 的 K 在 paged pass 之后才写"（同文件 `:2070-2072` 的注释）。quote 一下这段注释的原文更清楚：*K for a ragged extend's tokens is written after the paged pass, so that pass reads the prefix alone and masks it itself.* 这个顺序在 decode 里不存在——decode 只有一个 token，写完立刻读，paged pass 天然看到全部。

第三条：`forward_extend` 是可以被"借走"的。`FlashMLABackend` 只实现 decode 侧的 kernel，`forward_extend` 遇到 `ForwardMode.EXTEND` 时直接 `return super().forward_extend(...)`（`flashmla_backend.py:530-535`），把 prefill 交给父类 `FlashInferMLAAttnBackend`；只有 `target_verify` 和 `draft_extend_v2` 才用 FlashMLA 的 decode kernel。**"MLA 的 prefill 和 decode 是两个 kernel"这句话，在代码里就是一个 `super()` 调用。**

## 5. `out_cache_loc` 与 `req_to_token`：写坐标和读坐标

同一个 token 有两个坐标。

**写坐标 `out_cache_loc`**：一个一维 int64，长度 = 本步新 token 数（`forward_batch_info.py:423`），元素是这一层 K/V 该写进哪个物理 slot。它由分配器给出（extend 走 `alloc_for_extend`，decode 走 `alloc_for_decode`，`managers/schedule_batch.py:3216`），随 `ForwardBatch` 一路带到 attention 层，最后变成 `KVWriteLoc.for_batch(...)` 的 `loc` 字段（`mem_cache/memory_pool.py:1748-1766`），交给 `set_kv_buffer`。

**读坐标 `req_to_token`**：一张 `(max_reqs+1, max_context_len)` 的 int32 表（`memory_pool.py:379`），第 0 行是 padding 行——cuda graph 的假请求 `req_pool_indices` 全指向 0（`:411-413` 的注释）。表里存的是"第 i 个请求的第 t 个 token 在哪个 slot"。

两者的结合点在调度器里，而且顺序很关键：**先写表，再 forward**。`alloc_for_extend` 拿到 `out_cache_loc` 之后立刻把这一段追加进请求自己的行（`mem_cache/allocation.py:67` 的 `write_cache_indices`）：

```python
req_to_token_pool.write((req_idx, slice(prefix_len, seq_len)),
                        out_cache_loc[pt : pt + extend_len])   # allocation.py:108-116
```

所以 attention 层 forward 的时候，`req_to_token` 里已经有本步新 token 的坐标了——这就是为什么第 4 节里 paged 那一半必须只读 prefix（否则它会读到还没写的 K），也是为什么 decode 可以放心地只读表。

后端做的事，是**把表的一行 gather 成 kernel 要的形状**。三个后端三种形状：

- flashinfer：gather 成 CSR 的 `kv_indices`，`pack_read_stream`（`mem_cache/kv/alloc/runner_ids.py:567`）内部调 `build_kv_read_table_packed`（`:615`）或 Triton kernel `create_flashinfer_kv_indices_triton`（`:634`）；
- FA：直接切表的一行，`metadata.page_table = req_to_token[req_pool_indices, :max_seq_len_k]`（`flashattention_backend.py:1197-1199`），二维、一次行 gather；
- FlashMLA：用 Triton 把 token 坐标除以 64 变成页坐标，`data // PAGED_SIZE`（`kernels/ops/attention/kv_indices.py:194-252`，`PAGED_SIZE: tl.constexpr = 64` 在 `:202`），得到一张 `(bs, ceil(max_k/64))` 的 block table，再 `paged_view(k_cache, PAGE_SIZE)`（`flashmla_backend.py:491`）把 KV 池看成页数组。

**同一个 `req_to_token`，三种消费者。** 这也是为什么 SGLang 不肯把页表做成"后端私有"：它是全局调度器和 kernel 之间唯一的公共契约。

## 6. page_size：默认 1，MLA 64，FA4 128

`--page-size` 的默认值是 `None`，解析后：HIP 的 vectorized_5d 布局给 64、MUSA 给 64、其余全给 **1**（`arg_groups/overrides.py:1419-1441`）。这就是 SGLang 和 vLLM 最直白的一处分歧。

然后各后端会把它改掉，而且是静默改（只 warning）：

| 后端 | 允许的 page_size | 出处 |
|---|---|---|
| `flashmla` | 强制 64 | `overrides.py:1194-1201` |
| `trtllm_mla` | 32 / 64 | `:1202-1210` |
| `cutedsl_mla` | 32 / 64 | `:1211-1220` |
| `hpc_ops` | 强制 64 | `:1235-1244` |
| `fa4`（非 MLA，SM100） | 强制 128 | `:1321-1343` |

MLA 的 64 不是拍脑袋。FlashMLA 的 kernel 只吃 `block_table`，它的页块大小编译期固定：`PAGE_SIZE = 64`（`flashmla_backend.py:36`），池子要被 `paged_view` 切成 `[slots/64, 64, ...]`，而 backend 里每一条 metadata 路径都写着 `assert self.page_size == PAGE_SIZE`（`:183`、`:216`、`:254`、`:368`）。这层 kernel 接口上没有 page_size 参数，只有"第 i 页从 64*i 开始"这个隐含约定，所以池子必须按 64 对齐。

这笔账可以算。设批内 $n$ 个请求、每个要读 $L_i$ 个 token、页大小 $P$、索引 int32：

$$
B_{\text{idx}} = 4\sum_{i=1}^{n} L_i \qquad\text{vs}\qquad B_{\text{blk}} = 4\sum_{i=1}^{n}\left\lceil \frac{L_i}{P}\right\rceil
$$

$n=128$、$L_i=32768$、$P=16$ 时，前者 16 MiB、后者 1 MiB；$P=1$ 时两者相同。所以 `page_size=1` 的代价不在 kernel，在索引带宽——flashinfer 路径每步要 gather 一遍 $\sum L_i$ 个 int32。这也是为什么 SGLang 在 flashinfer 上做了那么多"别每步重建"的工程：`_create_decode_wrappers` 把预分配的 `kv_indptr` / `kv_indices` / `kv_last_page_len` 直接塞进 wrapper（`flashinfer_backend.py:1159-1172`），按 `bs` 缓存进 `decode_cuda_graph_metadata`；`fast_prefill_plan`（`:193`）绕开 FlashInfer 原生 `plan()` 里那次必经的 `.to("cpu")`，改由调用方传 host 已知的 `qo_indptr_host` / `kv_indptr_host`。

## 7. 树状 mask 为什么踢掉一批后端

EAGLE 的 `topk>1` 把 draft 从"一条链"变成"一棵树"。对 attention 层来说，树意味着两件事：

1. **KV 位置不再连续**。verify 时 $bs \times topk$ 个候选 token 的 slot 是按 `out_cache_loc` 排的，不是按树的前序排的。FA 的处理是做一次排序（`flashattention_backend.py:1160-1174`）：用 `custom_mask` 的每一行当 key，把有效列排到前面，然后把 `cache_seqlens` 改成 `mask.sum(dim=1)`——**把一个散落的树压成"每行长度不同但连续"的 varlen 批**，这样稠密 kernel 才吃得下。
2. **每行的 KV 长度不同**。所以 FA 在 `topk>1` 的 verify 上开 cascade attention（`:1600-1604`），先算共享前缀、再按行补。

一个后端要支持 `topk>1`，至少得同时有"自定义 mask"和"能重排页表"两样。FlashMLA 的 decode kernel 只接受 `block_table` + `cache_seqlens`（`flashmla_backend.py:489-497`），没有 mask 入口，所以它的多步 draft backend 在构造时就抛：

```text
ValueError: Currently FlashMLA only supports topk=1 for speculative decoding
```

（`flashmla_backend.py:619-622`；FlashInfer 的 MLA 版同款，`flashinfer_mla_backend.py:1229-1232`。）

这也解释了第 2 节那条默认规则为什么会因为 topk 改答案：`fa3`/`trtllm_mha` 的默认只给 `topk<=1`。**不是性能偏好，是能力边界。**

## 8. 与 vLLM 对照

基线 vLLM `c7a7607`，代码在 `vllm/v1/attention/`。

- **抽象粒度**。vLLM 的 `AttentionMetadata` 是个空类（`v1/attention/backend.py:387`），每个 backend 自己往里塞字段；跨后端共享的是 `CommonAttentionMetadata`（`:395`），一步一份，含 `query_start_loc` / `seq_lens` / `block_table_tensor` / `slot_mapping`（`v1/attention/backend.py:402-420`）。SGLang 没有这层中间态：backend 直接吃 `ForwardBatch`，自己产 `DecodeMetadata` 或 `PrefillMetadata`。
- **页表形态**。vLLM 的 `BlockTable` 是 `(max_num_reqs, max_num_blocks_per_req)` 的持久 int32 缓冲（`v1/worker/block_table.py:49`、`:80-82`），`get_device_tensor` 只是切前 `num_reqs` 行（`:188-190`）；`slot_mapping` 是 `(max_num_batched_tokens,)`（`:85-87`）。SGLang 的 `req_to_token` 是 token 粒度（`max_context_len` 列），页表要靠 `//PAGE_SIZE` 现推。vLLM 存"页"，SGLang 存"token"，后者更通用（`page_size=1` 时直接就是页表），前者索引更小。
- **两条路径 vs 一条**。vLLM 一个 backend 一个 `forward`，混合批也只是一份 metadata（decode 就是 query 长度 1 的请求），部分 backend 再用 `reorder_batch_threshold` 把短 query 排到批首；SGLang 把它提升成两个抽象方法，甚至允许两个 backend 实例：`resolved.decode != resolved.prefill` 时构造 `HybridAttnBackend`（`attention_backend_setup.py:202-230`），`_select_backend` 按 mode 分发（`hybrid_attn_backend.py:67-87`）。
- **CUDA Graph 的耦合方向**。vLLM 是由下往上：取所有 builder 的 `AttentionCGSupport` 最小值（`ALWAYS / UNIFORM_BATCH / UNIFORM_SINGLE_TOKEN_DECODE / NEVER`，`v1/attention/backend.py:570-583`），据此决定整个引擎的 `cudagraph_mode`（`v1/worker/gpu_model_runner.py:6767-6776`）。SGLang 是分相位、按后端声明能力位：`Phase.DECODE` / `Phase.PREFILL` 各有自己的 capture 后端（`model_executor/cuda_graph_config.py:32-63`），backend 用 `supports_full_cuda_graph_chunked_prefix`、`prefill_cuda_graph_max_batch_size` 这类 flag 逐个 opt-in（`base_attn_backend.py:234-251`）。
- **每步的 CPU 工作**。vLLM 的每步 host 成本基本是 O(num_reqs + num_tokens)：切一次 block table、写一次 slot_mapping。SGLang 的 flashinfer 路径理论上是 O(bs)（`cumsum` 在 GPU 上），但 FlashInfer 自己的 `plan()` 会做 D2H，所以它得靠 `fast_*_plan` 和 `global_override_indptr_cpu` 把这次回读掐掉（`flashinfer_backend.py:1857-1867`）。反过来，SGLang 有一个显式的开关来省这一步：`needs_cpu_seq_lens`（`base_attn_backend.py:221-222`），`FlashMLABackend` 和 `TritonAttnBackend` 都声明为 `False`，于是调度器的 `FutureMap` 不必每步发布 `seq_lens_cpu`（`managers/overlap_utils.py:26-51`）。**结论不是"谁更省"，而是"SGLang 把省这笔钱做成了每个后端自己的声明"。**

### 常见坑

1. **`page_size` 会被后端悄悄改掉。** 你写 `--page-size 1`，选了 `flashmla` 就是 64（`overrides.py:1194-1201`）；SM100 上选了 `fa4` 就是 128（`:1321-1343`）。想看真实值，别信命令行。
2. **MLA 的 prefill 未必走 MLA kernel。** `FlashMLABackend.forward_extend` 对 `EXTEND` 直接 `super()`（`flashmla_backend.py:530-535`）。profile 里看到的 prefill kernel 名可能是 FlashInfer 的。
3. **`HybridAttnBackend` 是实验特性。** 启动时除了 info 还有一条 warning 明说 `experimental and unstable`（`attention_backend_setup.py:238-240`）。
4. **`topk>1` 会同时改后端和 page_size。** 默认后端判据里带 `page_size == 1`（`utils/common.py:4002-4008`），FA4 的 128 强制又对 `topk>1` 开了例外（`overrides.py:1330-1333`）。同一台机器，开不开投机解码，page_size 可能不同。
5. **`req_to_token` 的第 0 行是垃圾桶。** cuda graph 的 padding 请求都映射到它（`memory_pool.py:411-418`），调试时看到一堆 0 不是 bug。
6. **别在 `init_forward_metadata_in_graph` 里做 host 读。** 契约写在 `base_attn_backend.py:199-201`，违反它的表现是捕获成功、replay 结果错。

### 读代码时的检查点

1. 从 `layers/radix_attention.py:190` 的 `get_attn_backend().forward(...)` 进去，对着 `base_attn_backend.py:385-428` 数一遍：你这个 mode 会落到 `forward_decode` 还是 `forward_extend`？
2. 打开 `flashinfer_backend.py:1006`，看三个分支各自产出哪个 metadata 类，再去 `:1828` / `:2279` 确认 `kv_indptr` 的语义（decode 用 `seq_lens`，extend 用 `paged_kernel_lens`，ragged 时后者只等于 `prefix_lens`）。
3. 在 `runner_ids.py:567` 的 `pack_read_stream` 里确认一件事：`kv_indices` 是从 `req_to_token` 表 gather 出来的，**不是**从 `out_cache_loc` 来的。前者是历史，后者是本步。

## 小结

- 一个后端对外只有三条入口：`init_forward_metadata`（每 forward 一次）、`forward_extend`、`forward_decode`；`forward` 只是 mode 分发器（`base_attn_backend.py:385-428`）。
- "extend"是一族而不是一个相位：`EXTEND`/`MIXED`/`TARGET_VERIFY`/`DLLM_EXTEND` 都算（`forward_mode.py:46-52`），所以 verify 走的是 prefill 的 kernel。
- flashinfer 路径下 decode 要 3 个索引张量（`kv_indptr`/`kv_indices`/`kv_last_page_len`），extend 要 4 个（多一个 `qo_indptr`），无 prefix 时可以一个页表都不要（ragged 路径）。
- SGLang 给 FlashInfer 的一律是 token 粒度 CSR + `page_size=1` 常量（`flashinfer_backend.py:2425-2432`），分页是 SGLang 自己的事。
- `out_cache_loc` 是写坐标（本步新 token 的 slot），`req_to_token` 是读坐标（历史），调度器在 forward 之前就把新 token 追加进表（`allocation.py:108-116`）。
- `page_size` 默认 1，MLA 强制 64、FA4 强制 128；64 来自 FlashMLA 的块布局（`flashmla_backend.py:36`、`kv_indices.py:202`），代价是索引数组按 token 数而不是页数增长。
- 树状 mask 要求后端能重排页表并把 `cache_seqlens` 改成 `mask.sum`（`flashattention_backend.py:1160-1174`），不具备这个能力的后端在 `topk>1` 时直接构造失败。

下一篇讲结构化输出：[压缩 FSM、跳词解码与 grammar 缓存]({{< relref "sglang-07-constrained" >}})，看 attention 之外的另一个"每步都要重算、且极难塞进 CUDA Graph"的地方。

参考：

- SGLang `python/sglang/srt/layers/attention/`（`base_attn_backend.py`、`attention_registry.py`、`flashinfer_backend.py`、`flashattention_backend.py`、`flashmla_backend.py`）
- vLLM `vllm/v1/attention/backend.py`、`vllm/v1/worker/block_table.py`
