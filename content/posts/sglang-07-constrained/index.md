---
title: "SGLang 研读（七）结构化输出：压缩 FSM、跳词解码与 grammar 缓存"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 8
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "结构化输出", "Grammar", "约束解码"]
categories: ["推理框架"]
---

一个 6 字段的 JSON，输出大约 120 个 token，其中 `{"`、`"name": "`、`", "`、`"}`、`"plate"` 这类字符占了六成以上。模型每吐一个 token，都要跑一次完整前向、一次 128k 词表的掩码填充，然后采样——虽然下一个 token 其实早就被语法唯一确定了。

这就是 SGLang 起家的那个卖点：**跳词解码（jump forward）与压缩 FSM**。`python/sglang/srt/constrained/outlines_jump_forward.py:14-17` 的文件头至今还挂着那篇 compressed FSM 博客的链接，模块 docstring 第一句就是 "Faster constrained decoding"。这一篇要把这笔账算清楚——但先说结论：**接口齐全，主路径当前没接**，所以今天 SGLang 真正的优势不在跳词，而在掩码的工程化。

系列位置：前置是[调度器]({{< relref "sglang-04-scheduler" >}})与[两级 overlap]({{< relref "sglang-05-overlap" >}})，本篇的两条主线（掩码怎么填、FSM 怎么跟上调度）都建立在它们之上。对位阅读是 vLLM 侧的 [采样与结构化输出]({{< relref "vllm-08-sampling" >}})。

## 1. 两条路线：掩码与跳词

约束解码只有两种实现方式，区别在于"谁来花这笔钱"：

- **掩码（mask）**：每一步问一次 FSM「当前状态允许哪些 token」，把不允许的位置写成 `-inf`，再在剩下的集合里采样。每步都付一次全词表判定的钱，换取"任何可编译结构都能约束"。
- **跳词（jump forward）**：提前算出 FSM 里那些**只有唯一出边**的状态，把连续的单出边折叠成一段字符串，一次性追加进输出。省掉的是"采样 k 次 + 填 k 次掩码"，**不省前向**——那 k 个 token 的 KV 仍然要算（否则后面的 attention 没东西可读），只是从 k 次 decode 变成一次更宽的 extend。

| | 掩码 | 跳词 |
|---|---|---|
| 生效结构 | 任何能编译成 FSM 的约束（JSON schema / regex / EBNF / structural tag），包括自由字符串 | 只有"唯一出边"的状态：常量 key、固定标点、单值 enum |
| 每步成本 | $B \cdot V$ 次 FSM 判定 + 掩码 H2D + 一次 kernel | 0 次采样；但 k 个 token 仍要一次前向 |
| 收益形态 | 合法率 100%，速度不变甚至更慢 | 步数从 $N$ 降到 $N - J(1-1/L)$ |
| 实现状态 | 所有后端都有，线上默认路径 | 接口齐全，调度路径无调用点 |

两者不冲突：跳词只处理"确定"的那部分，剩下的岔路口仍然要靠掩码。所以真正的问题是——掩码这条路，SGLang 比别人便宜在哪。

## 2. 掩码的一生

xgrammar 后端（默认后端，`arg_groups/serving_hook.py:236-241` 把 `grammar_backend` 解析成 `xgrammar`）一次 decode 步里，掩码走六步：

1. **分配**。`_allocate_token_bitmask` 用 `torch.full(shape, -1, dtype=int32, pin_memory=True)`（`constrained/xgrammar_backend.py:66-75`），形状来自 `xgrammar.get_bitmask_shape(batch, vocab)` —— 每行 `ceil(V/32)` 个 int32。**全 1 位表示"全部允许"**，这个默认值是后面省钱的根基。pin_memory 不是随手写的：注释说明 pageable 源会让 `non_blocking` 静默降级成同步拷贝。
2. **挑行**。`update_regex_vocab_mask` 只把"有 grammar 且未 finished 且未 terminated"的请求放进 `entries`，其余行保持缓冲区刚分配的全 1（`sampling/sampling_batch_info.py:390-415`，注释在 `:404-405`）。注意分配的行数是整批 `len(self.temperatures)`，不是约束请求数——多出来的行是免费的。
3. **填**。默认 `fill_vocab_mask_batched` 就是逐行调 `grammar.fill_vocab_mask(mask, row)`（`constrained/base_grammar_backend.py:88-93`），xgrammar 侧落到 `matcher.fill_next_token_bitmask(vocab_mask, idx)`（`xgrammar_backend.py:123-124`）。llguidance 覆写了这个静态方法，用 native 的并行填充（`llguidance_backend.py:168-176`）。
4. **搬**。`move_vocab_mask` 是 `to(device, non_blocking=True)`（`xgrammar_backend.py:126-128`），因为源是 pinned 的，这次 H2D 才真的是异步。
5. **应用**。`GrammarMask.apply` 记住的是"随便哪个 grammar"——它只是拿到后端实现的句柄，不是 per-request 状态（`base_grammar_backend.py:148-158`）。非 HIP 平台走 `apply_token_bitmask_inplace_triton`（`xgrammar_backend.py:130-147`）。
6. **释放**。`_preprocess_logits` 应用完立刻 `sampling_info.grammar_mask = None`（`model_executor/model_runner.py:2154`），否则这个张量会被 `delay_sample_func` 闭包和 batch 记录一直引用，构成稳定的显存泄漏（`model_runner.py:2149-2153` 的注释）。

第 5 步的 kernel 值得单独看。它**不读 logits**，只在 bit 为 0 的位置写 `-inf`：

```python
# python/sglang/kernels/ops/grammar/bitmask_ops.py:71-78（略去换行）
packed_bitmask = tl.load(bitmask_ptr + batch_id * bitmask_strides + bitmask_offsets, ...)
bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
tl.store(logits_ptr + batch_id * logits_strides + offsets, -float("inf"), vocab_mask & bitmask)
```

三个设计点叠起来，才是"一次 kernel 搞定整批"：

- **行号即 strided 维度**。`batch_id = row_id`（不传 `indices` 时），因为掩码的第 i 行天然对应 logits 的第 i 行——scheduler 与 model_runner 在同一个进程、同一个 batch 对象上，不需要重排。
- **持久化 grid**。`grid = (NUM_SMS,)`，work item 数是 `num_rows * ceil(V/4096)`（`bitmask_ops.py:121-124`）。128k 词表、B=256 时是 8192 个 work item，却只发 132 个 program（H100），每个 program 用 `tl.range` 跨步领活。相比"每行每块一个 block"，省掉的是调度开销。
- **未约束行零写入**。那些行掩码全 1，展开后 `bitmask` 全 False，`tl.store` 一个字节都不写。所以整批过一次 kernel，成本只跟**被约束的行数 × 词表**相关，跟批里混了多少普通请求无关。

## 3. 成本模型

把一次 decode 步的掩码成本写成两段：

$$
C_{\text{mask}} \;\approx\; \underbrace{B \cdot \lceil V/32 \rceil \cdot 4}_{\text{H2D, bytes}} \;+\; \underbrace{B \cdot V \cdot c_{\text{fsm}}}_{\text{CPU scan}}
$$

第一段是带宽账，第二段是 CPU 账，量级差三个数量级：

| 项目 | B=256, V=131072 | 量级 |
|---|---|---|
| 掩码大小 | 256 × 16 KiB = **4 MiB** | 每行恒等于 V/8 字节 |
| H2D（pinned） | 4 MiB | 带宽下限约 1.2 µs（3.3 TB/s），实际十几 µs |
| apply kernel | 读 4 MiB，稀疏写 | 一次 launch，十几 µs 量级 |
| CPU 填 | 256 × 131072 ≈ **3350 万次判定** | 毫秒级（估算，随语法复杂度浮动） |

结论很直接：**GPU 那侧几乎免费，CPU 那侧是全部成本**。所以 SGLang 的优化重点不是把 kernel 写快，而是让这份 CPU 工作**不站在关键路径上**——这就是 §6 要讲的 overlap 对齐。

同一笔账换个后端就变味。outlines 后端分配的是 `torch.zeros(B, V, dtype=torch.bool)`（`outlines_backend.py:56-58`），一行 128 KiB，是打包位掩码的 8 倍；应用是 `logits.masked_fill_(vocab_mask, -inf)`（`:71-73`），整块读、整块写。所以"用哪个后端"不只是语法表达力的选择，也是这批字节数的选择。

## 4. 跳词：压缩 FSM、KV 对齐与长度语义

压缩 FSM 的构造在 `outlines_jump_forward.py:61-139`，思路朴素到可以手推：

1. 把正则编译成 byte-level FSM，再确定性化（`:69-70`）；
2. 对每个状态数出边数。终态先人为记 1 条边，因为"终态可以自然通向终止"（`:81-84`）；
3. 出边数超过 1 的状态，从跳跃表里删掉（`:98-107`）；只有恰好 1 条出边的状态留下 `JumpEdge`；
4. `jump_forward_symbol` 从这个状态开始沿链走，把沿途的单字符符号拼成一个长字符串，直到走出表（`:146-157`）。

这就是"压缩"二字的实际含义：**它没有压缩 FSM 本身，压缩的是"状态转移的步数"**——一串单出边状态被折叠成一次跳跃。落到 `{"name": "` 这种前缀上，就是一次吐完这 10 个字符（大约 3–4 个 token），而不是采样 10 次。

字节层还有一套平行的 `JumpEdge.byte`（`:114-137`），专门处理 UTF-8 多字节字符：跳词边界可能落在 continuation byte 中间，所以 outlines 用 `<0xXX>` 形式的字节 token 兜住后缀，再重新切词（`outlines_backend.py:83-105`）。重切词会改变 token 边界，这就是 `jump_and_retokenize` 存在的原因——找出新旧序列的公共前缀，回滚 grammar 到那里，再逐个 accept 新 token（`xgrammar_backend.py:178-200`），回滚深度上限是 `MAX_ROLLBACK_TOKENS = 200`（`:63`）。

**然后是必须说清楚的部分**：在这个快照里，跳词没有接在生成循环上。

- 三个后端都实现了 `try_jump_forward`：xgrammar 用 `matcher.find_jump_forward_string()`（`xgrammar_backend.py:168-172`），llguidance 用 `compute_ff_tokens()`（`llguidance_backend.py:212-217`），outlines 走自己的跳词表（`outlines_backend.py:83-105`）；`reasoner_grammar_backend.py:353-366` 做了透传。
- 但全仓库 `grep -rn "try_jump_forward"` 只有这些定义和一个转发，`srt/managers/`、`srt/speculative/` 下**没有任何调用点**。
- 更彻底的是，outlines 后端连跳词表都不构造了：`jump_forward_map = None`（`outlines_backend.py:160-161`），于是 `try_jump_forward` 第一行就返回 None（`:84-85`）。`OutlinesJumpForwardMap` 的唯一实例化地点是它自己的 `test_main`（`outlines_jump_forward.py:182`）。

所以仓库里留下的是"接口 + 一份 outlines 跳词表实现 + 每个后端几行 `try_jump_forward`"，不是线上默认路径。这一点很重要，因为它决定了下面这笔收益账的读法。

### 如果接回：收益公式与三笔账

设输出 $N$ 个 token，其中 $J$ 个是"确定"的，平均每次跳跃吐 $L$ 个，则步数变成：

$$
\text{steps} = N - J\left(1 - \frac{1}{L}\right), \qquad
\text{speedup} = \frac{N}{N - J\left(1 - 1/L\right)}
$$

取 $N=100$、$J=60$、$L=6$：步数 50，直观上快一倍。但这个数字有个前提——**一次提交 k 个 token 的机制必须存在**，而这恰恰是三件麻烦事：

1. **KV 要一次给 k 个 slot**。正常 decode 一步只分配一个槽位：`prepare_for_decode` 里写死 `alloc_for_decode(self, token_per_req=1)`（`managers/schedule_batch.py:3216`），分配器按 `bs * token_per_req` 拿 token 槽（`mem_cache/allocation.py:650-666`）。跳词/投机这类"一步多 token"必须按 k 分配，并让 `req_to_token` 的写入偏移也跟着走 k 格。
2. **FSM 只能吞下合法的前缀**。一次吐回的 run 里如果有 token 让语法终止，后面的必须丢掉——`_accept_grammar_tokens` 就是干这个的：逐个 accept，遇到 `is_terminated()` 立刻 break，返回可保留的前缀（`scheduler_components/batch_result_processor.py:939-974`）。注释写得很直白：spec 的"语法结束后缀"既不进 KV 也不返回给用户。
3. **长度上限只能事后截断**。正常路径一步一个 token，`len(output_ids) >= max_new_tokens` 就够了（`schedule_batch.py:1465`）；一次提交 k 个就可能一步越过上限，于是有 `update_finish_state(new_accepted_len=k)`（`:1439`）和 `_cap_finished_len_at_max_new_tokens`（`:1404-1414`）——把越过上限的 stop 降级成"长度结束"。**这就是"跳词改变 `max_new_tokens` 语义"的全部机制**：它不再是"第几步停"，而是"哪一步提交的 run 里停，并截到第几个"。

今天这套机制的唯一使用者是投机解码：verify 接受 run 之后 `req.kv.commit_decode(num_accept_tokens)`（`schedule_batch.py:379-381`，调用点 `batch_result_processor.py:920-922`）。跳词如果接回，复用的就是这条通路。

## 5. grammar 的生命周期与缓存

约束请求进 scheduler 的第一站是 `process_req_with_grammar`（`constrained/grammar_manager.py:154-215`）：

1. 按类型取 key：`("json", schema)` / `("regex", pattern)` / `("ebnf", grammar)` / `("structural_tag", tag)`（`:167-179`）；
2. 再让后端加变体后缀——`get_request_grammar_key` 的默认实现原样返回（`base_grammar_backend.py:226-235`），reasoner 包装层会按"提示是否结束在开放答案块"和 thinking budget 追加 `:open_answer` / `:think_budget=N`（`reasoner_grammar_backend.py:469-487`）；
3. 查缓存：命中就 `copy()` 一个模板（`base_grammar_backend.py:304-313`）；
4. 未命中就丢线程池编译，拿回一个 `Future`，请求进 `grammar_queue`（`:312-313`、`grammar_manager.py:186-193`）。

缓存命中为什么便宜：`XGrammarGrammar.copy()` 只重建 `GrammarMatcher`，直接复用缓存里的 `CompiledGrammar`（`xgrammar_backend.py:149-166`），语法编译那一步整个跳过。这也是"按 (schema, backend) 缓存"的实际形态：逻辑上键是二元组（编译产物与后端绑定，换后端必须重编），但后端在服务启动时就被 `create_grammar_backend` 固定成进程级单例（`base_grammar_backend.py:371-446`），所以物理 key 退化成一元的 `(key_type, key_string)`——`base_grammar_backend.py:206` 的类型注解写得很清楚：`Dict[Tuple[str, str], BaseGrammarObject]`。

顺带澄清一个容易记错的点：**grammar 缓存的 key 里没有 `cache_salt`，也没有租户维度**。`cache_salt` 在 SGLang 里是 KV 前缀缓存的名字空间盐（`mem_cache/utils.py:101-110` 把 `extra_key`/`cache_salt` 塞进存储 hash 的种子），跟 grammar 缓存是两套东西。grammar 缓存就是一个裸 `dict`（`base_grammar_backend.py:206`），进程内共享、无租户隔离。基础键没有容量上限，唯一的例外是 reasoning 包装层给 thinking budget 变体加的 32 条 LRU（`reasoner_grammar_backend.py:37-40`、`:502-511`）；其余全靠 `flush_cache` 清（`scheduler.py:5868` → `grammar_manager.clear()` → `backend.reset()`，`grammar_manager.py:74-76`）。真要隔离，只能靠 key 里的变体后缀。

编译是异步的，但"等编译"这件事仍然会卡住调度：

- `get_ready_grammar_requests` 在每轮调度里最多轮询 `SGLANG_GRAMMAR_POLL_INTERVAL`（默认 5 ms，`environ.py:412`），睡眠间隔是它的 1/10（`grammar_manager.py:240-260`）。**只要队列里有一个 grammar 没编完，这一轮调度就最多多花 5 ms**。
- 轮询次数上限 `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 10000`（`environ.py:413`），按注释的算法就是"最坏等待 = 次数 × max(轮询间隔, 单次前向时延)"，5 ms × 10000 ≈ 50 s 后判超时（`grammar_manager.py:268-274`）。超时**不写缓存**，因为超时可能只是执行器排队，不能证明语法非法（`:334-335` 的注释）。
- TP/DP 各 rank 并行编译，谁先编完谁先想跑，所以 ready 集合必须取**交集**、failed 集合取**并集**（`:287-290`）——一个 rank 说失败，全体 abort；一个 rank 没编完，全体等。PP 上再由 0 号 rank 单链转发这个决定（`:86-116`）。

另一个坑是**负面缓存**：编译失败返回 `InvalidGrammarObject`（`base_grammar_backend.py:190-198`），它也会被 `set_cache` 记住（`grammar_manager.py:321`），下一个同 schema 请求直接在命中分支被 abort（`:195-201`）。好处是坏 schema 不会每次都烧一遍编译；坏处是**改了上游 schema 但字符串没变**的情况救不回来，只能 flush。

## 6. 与 overlap 调度对齐

overlap 模式下，第 N 批的 forward 已经发出去了，第 N-1 批的结果还在队列里。而填掩码必须读到"第 N-1 批的 token 已经被 FSM 接受"之后的状态——**掩码是 FSM 的函数，FSM 落后一步，掩码就是错的**。SGLang 分两条路处理：

**非投机**：把采样推迟成闭包。`launch_batch_sample_if_needed` 调用 `batch_result.delay_sample_func()`（`scheduler.py:5358-5375`），而这个调用点被放在"处理完上一批结果"之后（`:2205-2209` 的注释：sample 依赖上一批结果，所以放在它后面）。掩码填充发生在 `_preprocess_logits`（`model_runner.py:2129-2154`），也就是闭包内、forward stream 上——CPU 填掩码的时间与上一批的 result processing 重叠。

**投机**：采样在 verify 里就发生了，推迟不了，于是有两种对齐方式。

- 不支持 grammar overlap 的算法：`grammar_needs_sync()` 返回真（`schedule_batch.py:2155-2164`，条件是 `has_grammar and 有 spec and spec 不支持 overlap`），`should_process_last_batch_early` 据此在调度下一批**之前**先把上一批 pop 掉处理（`scheduler.py:2217-2236`）。
- 支持的算法（EAGLE / standalone / dflash，见 `speculative/spec_info.py:165-171`）：把 `grammar_barrier = self._advance_pending_grammar` 作为 kwarg 传进 forward（`scheduler.py:4996-5002`），worker 在建 bitmask 之前调用它，把队列里所有未处理 decode 结果的 FSM 先推进（`:2238-2246`）。这样 FSM 推进（纯 CPU）就藏在 target verify 的 forward 后面。NGRAM 不在这个名单里，因为它的 draft 是主机侧查表，没有 GPU 阶段可藏（`spec_info.py:165-171` 的注释）。

投机 + grammar 的掩码成本也因此变贵：`generate_token_bitmask` 对每个请求的**草稿树做一次 DFS**（`speculative/spec_utils.py:678-745`、`:746-795`），节点是否合法直接查父节点已填好的掩码位（`parent_bitmask[token // 32] & (1 << (token % 32))`，`:703-712`），接受就填子节点的掩码、返回时 rollback。也就是说一次 verify 要填 **树节点数 × V** 次判定，不是 V 次。topk=4、深度 5 的树约 20 个节点，CPU 掩码成本直接乘 20，换来的是一次可能接受 5 个 token。

## 7. 与 vLLM 对照

两边用的是同一个 xgrammar（SGLang 钉 `xgrammar==0.2.7`，`python/pyproject.toml:100`），掩码格式也一样是每行 `V/32` 个 int32。差别全在"谁填、填在哪、怎么送到 logits"。

| 维度 | SGLang | vLLM |
|---|---|---|
| 掩码在哪填 | model_runner 进程内，`_preprocess_logits`（`model_runner.py:2129`） | scheduler 进程填（`structured_output/__init__.py:388-520`），作为 `np.ndarray` 发给 worker |
| 行对齐 | 掩码行 == logits 行，kernel 不需要索引（`bitmask_ops.py:62-65`） | 必须重排：每步 `torch.full((num_logits, V/32), -1, pin_memory=True)` 再逐请求拷行（`structured_output/utils.py:142-170`） |
| apply | 持久化 grid，跨行跨块领活（`bitmask_ops.py:121-124`） | `xgr.apply_token_bitmask_inplace` 带 `indices`（`utils.py:180`），或新 worker 里自己的 kernel（`worker/gpu/structured_outputs.py:33-95`） |
| 缓存 | 进程内裸 dict + xgrammar 编译缓存，无上限 | xgrammar 自带 cache，上限 `VLLM_XGRAMMAR_CACHE_MB`（`backend_xgrammar.py:222-223`） |
| 并行填 | 后端级 batched fill（llguidance native，`llguidance_backend.py:168-176`） | 请求数 >128 且无投机时走线程池（`__init__.py:81-88`、`:423-430`） |
| 跳词 | 三后端接口齐全 + outlines 实现，**无调用点** | 只有一行注释链接（`backend_xgrammar.py:253-254`）和一句 TODO（`backend_guidance.py:250`） |

三点值得拎出来：

1. **vLLM 的"重排"是它架构的必然代价**。调度顺序和 worker 的 batch 顺序不一致，掩码就得按 `(request, position)` 重新映射一遍，还要处理投机的 `mask_stride`。SGLang 因为 scheduler 和 model_runner 同进程、同一份 batch，这一整步不存在。
2. **跳词两边都没实现**。区别是 SGLang 把接口（`base_grammar_backend.py:119-145`）和一份 outlines 实现留在了仓库里，vLLM 留的是注释。所以"vLLM 不做跳词"的正确说法是**未实现**，而不是"不支持"——xgrammar 本身提供 `find_jump_forward_string`。
3. **低熵输出上谁更快**，取决于掩码的 CPU 时间是否在关键路径上。SGLang 用 delay-sample + grammar barrier 把它藏进 GPU 时间；vLLM 在 scheduler 里同步填。schema 越复杂、约束请求越多，这个差距越明显；但两边都没有跳词，所以"JSON 里那 60% 的常量"仍然是一个 token 一次前向。

## 8. 常见坑

1. **`grammar_backend` 默认是 xgrammar，不是 none**（`arg_groups/serving_hook.py:236-241`）。你以为没开约束解码，其实编译器和掩码缓冲区都已经初始化了；tokenizer 不被 xgrammar 支持时会整个降级成 `none` 并打 warning（`base_grammar_backend.py:411-425`），此时结构化输出静默不可用。
2. **一个没编完的 grammar 会拖慢整轮调度**，最多 5 ms（`grammar_manager.py:240-260`）。冷启动 + 复杂 schema 时，其他请求的 ITL 会跟着抖。
3. **grammar 缓存没有租户隔离，基础键也没有上限**（只有 thinking budget 变体有 32 条 LRU）。多租户服务里，不同租户的 schema 共享同一份 dict；`cache_salt` 管的是 KV 缓存不是语法缓存（`mem_cache/utils.py:101-110`）。要么在 key 里自己加维度，要么定期 flush。
4. **编译失败会被缓存**（`base_grammar_backend.py:190-198` + `grammar_manager.py:321`）。同一个非法 schema 会一直快速失败，直到 `flush_cache`。
5. **`--disable-outlines-disk-cache` 的两个默认值不一致**：`environ.py:414` 把名字注册成默认 `False`（启用磁盘缓存），而 `outlines_jump_forward.py:41` 在**模块导入时**用 `get_bool_env_var(..., "true")` 读同一个变量、兜底值却是 `true`（禁用）。正常启动路径上参数解析会先把它写成 `0`（`arg_groups/serving_hook.py:438`），所以默认启用；但任何先 import、后解析参数的场景（单测、脚本直连）拿到的是关闭——排查 outlines 缓存行为先看这里。
6. **outlines 后端的掩码是 bool 张量**，一行 128 KiB（`outlines_backend.py:56-58`），是打包位掩码的 8 倍。128k 词表、B=256 时是 32 MiB 对 4 MiB。
7. **grammar 判 terminate 之后要立刻停**。`_accept_grammar_tokens` 会在语法终止的 token 处截断整个 run（`batch_result_processor.py:939-974`），否则 spec 或跳词多吐的 token 会绕过语法跑到输出里。

## 读代码时的检查点

1. 从 `process_req_with_grammar` 进去，跟一遍 `Future` → `grammar_queue` → `get_ready_grammar_requests` → `set_cache` 的闭环，确认"缓存里存的是 copy 还是原件"（`grammar_manager.py:321`）。
2. 打开 `sampling_batch_info.py:390-415`，数清楚一次 decode 步到底分配多少行、填多少行、哪些行保持全 1。
3. 在 overlap 模式上想一遍时序：第 N 批的 logits 回来时，FSM 停在哪一步？答案决定了你该看 `should_process_last_batch_early` 还是 `grammar_barrier`。

## 小结

- 约束解码的共同内核是**掩码**：每行 `V/32` 个 int32（128k 词表 = 16 KiB/行），全 1 表示不约束；GPU 侧只写 `-inf`、不读 logits，成本在 CPU 的全词表判定（B=256 时约 3350 万次/步）。
- SGLang 的掩码便宜在工程细节：只填约束行、未约束行零写入、持久化 grid 一次 kernel、行号免重排；对比 vLLM 每步要重排一份 `(num_logits, V/32)` 的 pinned 缓冲。
- **跳词在当前快照里没有调用点**：三个真实现 + 一个透传的 `try_jump_forward` 都在，outlines 的跳词表甚至被短路成 `None`（`outlines_backend.py:160-161`）。压缩 FSM 的代码可读、可测，但不在线上路径上。
- 跳词的收益机制可以算：步数从 $N$ 降到 $N-J(1-1/L)$；代价是一次提交 k 个 token——`out_cache_loc` 要一次给 k 个槽、FSM 要截断到语法终点、长度上限只能事后截。
- grammar 缓存是**进程级裸 dict**，key 是 `(type, 原文)` 加变体后缀，没有 `cache_salt` 隔离，基础键无上限（budget 变体 32 条 LRU）；编译失败也会被缓存。
- overlap 下掩码与 FSM 的对齐有三套机制：非投机靠 delay-sample，投机靠提前 pop（`grammar_needs_sync`）或 worker 内的 grammar barrier。

配套脚本 [`code/sglang/jump_forward.py`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang)：40 个 key 的 JSON 骨架上对比「只填掩码」与「跳词」，forward 320 → 81 步、掩码填充降 87.5%，生成结果逐字节一致——**这是上界估计，线上主路径还没接线**（`--test` 自测）。

下一篇是 [投机解码：EAGLE、draft 模型与 n-gram]({{< relref "sglang-08-speculative" >}})——那里会看到本篇的 `traverse_tree` 被当成"验证树的语法过滤器"，以及一次 verify 为什么能把掩码成本乘上树节点数。

参考文档：

- [SGLang 博客：Compressed FSM](https://lmsys.org/blog/2024-02-05-compressed-fsm/)
- [xgrammar 文档](https://xgrammar.mlc.ai/docs/)
