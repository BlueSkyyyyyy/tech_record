---
title: "SGLang 研读（八）投机解码：EAGLE 草稿树、draft-extend 与验证"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 9
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "投机解码", "EAGLE", "推测采样"]
categories: ["推理框架"]
---

`batch=1` 的 decode 步里，GPU 把 16 GB 权重从 HBM 搬一遍，只换回一个 token。投机解码是这个事实的直接推论：既然一次权重搬运的边际算力近乎免费，那就让权重一次搬完，多算几个位置。

SGLang 的实现比"小模型猜、大模型验"要多一层：它猜的不是一条链，而是一棵**树**——每层展开 `topk` 个候选，把整棵树摊平成一次 target forward。这一篇拆开三段：draft-extend 出第一层候选、draft 循环长出整棵树、verify 用一张祖先矩阵一次验完。

前置阅读：[调度器]({{< relref "sglang-04-scheduler" >}})（batch 怎么攒）、[两级 overlap 与 CUDA Graph]({{< relref "sglang-05-overlap" >}})（本篇第 6 节全靠它）、[KV 池与 slot 分配]({{< relref "sglang-03-memory-pool" >}})（第 4 节那笔预留账）。

## 1. 一个 decode 步里跑了几次前向

看 `EAGLEWorkerV2.forward_batch_generation` 的 decode 分支（`python/sglang/srt/speculative/eagle_worker_v2.py:1479`），顺序是固定的三步：

```python
verify_input = self.draft_worker.draft(batch)          # ① 长树
batch_output = self.verify(batch, ...)                 # ② target 验整棵树
self.draft_worker._draft_extend_for_decode(batch, batch_output)  # ③ 补 draft KV
```

容易数错的是 ①。`draft_forward` 的循环是 `for i in range(self.speculative_num_steps)`，但循环体在最后一步**先挑 token 再 break**，不跑前向（`eagle_worker_v2.py:822`、`:833`）：

```python
input_ids, hidden_states, scores, tree_info = select_top_k_tokens(i, ...)
if i == self.speculative_num_steps - 1:
    break
logits_output = self.draft_runner.forward(forward_batch).logits_output
```

所以 **`num_steps` 步树只对应 `num_steps - 1` 次 draft 前向**（`:695` 的 `n_inner` 就是这个数）。第 0 层的候选不是这里算的，而是上一轮 ③ 的产物：draft-extend 把 target 的 hidden state 喂进 draft 模型，一次前向拿到接受链的 hidden 和第一层 `topk_p/topk_index`（`:1307-1321`）。这也解释了 EAGLE 为什么必须 `pass hidden states`：每一轮的起点都是 target 自己的表示。

一次迭代的 GPU 工作量可以写成

$$
T_{\text{iter}} \approx T_{\text{target}}(B\cdot K) + \sum_{i=1}^{k-1} T_{\text{draft}}(B\cdot g) + T_{\text{draft}}(B\cdot K)
$$

$B$ 是 batch、$g$ 是 `speculative_eagle_topk`、$K$ 是 `speculative_num_draft_tokens`、$k$ 是 `speculative_num_steps`。注意 draft 的每一步也是 $B\cdot g$ 行而不是 $B$ 行——树在第 0 层就分叉了。

| 阶段 | forward mode | query 行数 | 主要输出 | 增量显存 |
|---|---|---|---|---|
| ① draft | `DECODE` | $B\cdot g$ / 次，共 $k-1$ 次 | `parent_list`、`top_scores_index`、`draft_tokens` | 每步 `[B·g, V]` logits + draft KV 写 $k-1$ 列 |
| ② verify | `TARGET_VERIFY` | $B\cdot K$ | `next_token_logits [B·K, V]`、hidden `[B·K, H]` | logits 是最贵的一项，见第 3 节 |
| ③ draft-extend | `DRAFT_EXTEND_V2` | $B\cdot K$ | 下一轮第一层候选 + hidden | 再一份 `[B·K, V]` logits |

$k=5,g=4,K=8,B=32,V=131072$ 时，②③ 各是一份 $32\times8\times131072\times4\text{B}\approx128$ MiB 的 fp32 logits。这就是投机在**高并发**下变贵的来源：verify 的有效 batch 是 $B\cdot K$，不是 $B$。

## 2. 树怎么长出来：每层 topk 展开，最后全局挑

`spec_utils.py:524` 的 `select_top_k_tokens` 分两段。第 0 层直接把 draft-extend 给的 `topk_index` 摊平当输入，父节点记成 `arange(-1, topk)`（`spec_utils.py:480`）；之后每一层对每个父节点的 `topk` 个子候选打分：

```python
expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)   # (b, topk, topk)
topk_cs_p, topk_cs_index = fast_topk(expand_scores.flatten(start_dim=1), topk, dim=-1)
```

（`spec_utils.py:498-503`）`scores` 是沿路径连乘的累计概率，所以这一步就是 EAGLE-2 的"按累计分剪枝"：$g^2$ 个组合里留 $g$ 个。**它并不保证每个父节点留一个子节点**——`flatten(start_dim=1)` 之后是全局 top-$g$，某个父节点可能吃掉全部名额。

整棵树候选总数是 $k\cdot g$（每层 $g$ 个），但最终只保留 $K-1$ 个节点：`organize_draft_results`（`eagle_utils.py:137-166`）把它们拼成 $k\cdot g$ 宽的一行，`topk(score_list, num_draft_token - 1)` 全局挑一次，父表则用 `parents_list[:-1]`——丢掉最后一层，因为它的 token 是 `break` 之前挑的，没有对应的前向。

两个细节值得记：`score_list` 是所有层摊平之后才 topk 的，所以**浅层节点可能挤掉深层节点**——树是"按预期收益剪枝"的，不是严格分层满叉；而 `topk=1` 时这整套退化成常量：`parent_list` 恒为 `arange(-1, steps-1)`、`top_scores_index` 恒为 `arange(steps)`，初始化时一次算好反复用（`base_spec_worker.py:118-150` 的 `_rebuild_topk1_chain_buffers`，注释写明 "runtime-invariant"）。这是 DeepSeek MTP 那条链的全部优化空间。

## 3. 一次前向验整棵树：祖先矩阵

`EagleVerifyInput.custom_mask`（`eagle_info.py:22`）不是常见的 `[B, K, L]` 三维 mask，而是一个**扁平 buffer**，长度在 `build_tree_kernel_efficient` 里算（`eagle_utils.py:226-236`）：

$$
\text{mask\_numel} = S\cdot K + K^{2}\cdot B
$$

$S$ 是 `seq_lens_sum`（所有请求已提交前缀之和）。前 $S\cdot K$ 段是"每个 verify token 对各自前缀"的部分——全是 `True`，因为树根之前的内容对所有节点都可见；后 $K^2\cdot B$ 段是每个请求一张 $K\times K$ 的**祖先矩阵**：第 $i$ 行第 $j$ 列为真，当且仅当节点 $j$ 是节点 $i$ 的祖先（含自身）。布局扁平是因为每个请求的行宽不同（`seq_len_i + K`），打包存储省掉 padding。

query 侧同样摊平（`eagle_info.py:107-113`）：

```python
qo_indptr = torch.arange(0, (1 + batch_size) * self.draft_token_num,
                         step=self.draft_token_num, dtype=torch.int32, device=device)
```

每个请求固定 $K$ 行 query，步进 $K$。于是**整棵树就是一次变长 prefill**：$K$ 个节点各自是一条独立 query，causal 关系由祖先矩阵给出，不再由位置先后给出。一次前向能同时验多条路径，靠的就是这个"把树拓扑编码进 mask"的动作，而不是任何特殊的 attention 算法。

verify 之后的检索走另一组张量：`retrieve_next_token` / `retrieve_next_sibling`（`eagle_info.py:26-27`），是从根出发的"第一个孩子 / 下一个兄弟"链表。greedy 路径上 `verify_tree_greedy_func` 用它做深度优先的逐层比对（`eagle_utils.py:712`）：每层从当前节点的孩子开始横向扫，命中就下潜，未命中就停。`accept_index` 就是这条被接受的路径的节点编号，宽度固定为 `spec_steps + 1`（`eagle_info.py:52-57` 的 `max_tree_depth`）——**树有 $K$ 个节点，接受路径最长只有 $k+1$ 个**。

树的形状本身是数据相关的，所以 PP 场景要把它当数据传：`draft(batch, with_topology=True)` 会把 `parent_list/top_scores_index` 一起返回（`eagle_worker_v2.py:749-753`），注释写明"other stages cannot re-derive it"。

## 4. 接受长度与 KV 预留

接受长度只有一个定义，藏在 `eagle_sample` 的末尾（`eagle_utils.py:924`）：

```python
accept_lens = num_correct_drafts + 1
```

`num_correct_drafts` 只数被接受的 draft token，`+1` 是必然产出的 bonus：`predicts` 按节点索引写，最后一个被接受节点的 slot 里存的就是它的下一个 token（候选被接受时是候选本身，链断时是残差采样或 argmax）。之后 `new_seq_lens = batch.seq_lens + accept_lens`（`eagle_worker_common.py:764`），注意 `batch.seq_lens` 是**本轮窗口之前**的长度，窗口本身那 $K$ 个 slot 是上一步就分配好的。

KV 预留这笔账在 `mem_cache/allocation_sizing.py` 里算得极干净：

```python
# Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
return max(spec_steps * spec_topk, spec_tokens)      # get_alloc_len_per_decode, :53
return 2 * get_alloc_len_per_decode()                # get_alloc_reserve_per_decode, :70
```

两处乘法都有明确理由：`max(steps*topk, num_draft_tokens)` 是因为 topk>1 时 draft 要写 `steps*topk` 个位置而 verify 用 `num_draft_tokens` 个，取大者；`2×` 是双缓冲，注释直说是"absorbs the `kv_committed_len` lag in overlap mode"——overlap 下本轮 KV 长度还没回传，只能按两轮预留。MTP 这类 fused draft 干脆共用 target 的池（`base_spec_worker.py:196-199`："A fused draft's KV lives in the target's pool: it has none of its own"），所以上面这串乘法算的始终是同一套 slot 编号空间。

还有一笔行内 headroom（`allocation_sizing.py:96-116`）：`extra = 4 + num_draft_tokens`，`req_to_token` 的行要比 context length 宽这么多。默认 `(steps=3, topk=1, K=4)` 下每请求每步押 `max(3,4)=4` 个 slot、预留 8 个；Llama 的默认 `(5, 4, 8)` 则是 `max(20,8)=20`、预留 40。

**batch 与 $K$ 的关系是硬约束，不是建议**：`speculative_hook.py:1127-1135` 在用户没显式指定时把 `max_running_requests` 从默认值重置成 **48**，并打 warning。原因就是上面这笔乘法：每个请求每步要多押 $\Theta(K)$ 个 slot，能同时跑的请求数必须同比缩小。

注意 $K$ 与产出长度不是一回事：树有 $K$ 个节点，但接受路径被 `accept_index` 的宽度 $k+1$ 卡死。`topk` 再大也不会多产出位置，只是让"这一层能中"更可能——每层 $g$ 个候选把该层的命中率从 $\alpha$ 抬向上界 $1-(1-\alpha)^g$，总量仍封顶在 $k+1$。

## 5. 为什么现在只有 v2：两个 graph runner 的形状理由

先纠正一个常见说法。这个 commit 上**没有 `eagle_worker.py`**（v1 已经被删掉了）：`spec_info.py:350-351` 的注释是"EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER always use the V2 worker, even with overlap disabled (scheduler drives it synchronously)"，`speculative_hook.py:1139-1143` 在关掉 overlap 时打的是"Non-overlap (synchronous) spec v2 is used"。历史包里的 v1/v2 之分，现在退化成了"同一个 v2 worker 的同步/异步两种驱动方式"。

v2 真正的结构特征是**两个独立的 CUDA graph runner**，因为它们 capture 的形状根本不是一回事（`eagle_worker_v2.py:482-644`）：

| runner | capture 的 forward | 每请求宽度 | 来源 |
|---|---|---|---|
| `EAGLEDraftCudaGraphRunner` | `ForwardMode.DECODE` | `topk` | `resolve_num_tokens_per_req(phase="draft_decode")`（`eagle_draft_cuda_graph_runner.py:155`） |
| `EAGLEDraftExtendCudaGraphRunner` | `ForwardMode.DRAFT_EXTEND_V2` | `num_draft_tokens` | 同函数 `phase="draft_extend"`（`eagle_draft_extend_cuda_graph_runner.py:149`） |

收益来自两边都是"小 kernel 串行链"。draft 是 $k-1$ 次 $B\cdot g$ 行的小前向，每次 launch 只有几十微秒，host 侧 dispatch 能占掉可观比例；draft-extend 是一次变长 extend，qo 布局固定（`eagle_info.py:444-450` 的注释："Constant num_tokens_per_req qo layout (required for cuda-graph capture)"）。两者分开 capture，才能各自用自己最合适的 bucket 宽度，而不是被对方的最大宽度拖累。

draft-extend 的图默认是开的（`SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` 默认 `False`，`environ.py:1733`），而且只在少数 backend 上 capture——`eagle_worker_v2.py:573-613` 列出白名单（Triton / TRT-LLM MLA / TRT-LLM MHA / FlashInfer / QSA / DSA / DSV4 / FA / FlashMLA），不在名单里就退回 eager。还有一个容易漏的第三份图：`EaglePostVerifyGraphs` 把 verify 之后的采样、接受判定、commit 也 capture 了（`eagle_post_verify_graphs.py:3-6`），但默认关（`SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH` 跟着 `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`，后者默认 `False`，`environ.py:284`），且只在 `topk==1` 且无 DP attention 时启用（`eagle_worker_v2.py:1434-1439`）。

## 6. 与 overlap 的三处咬合

overlap 调度的前提是"本步的 CPU 工作能盖住上一步的 GPU 时间"。投机把这条链拉长成三顿 GPU 活，于是有三处必须专门处理。

**（一）发布点前移。** 正常 decode 是 worker 返回后 scheduler 才 publish。spec v2 改成在 worker 内部回调（`eagle_worker_v2.py:1551-1553`）：

```python
# Publish before draft_extend so the fence is at verify-end.
if on_publish is not None:
    on_publish(batch_output.new_seq_lens)
```

scheduler 侧对应注释把动机写得很直白："Spec_v2 fires `on_publish` mid-worker (between verify and draft_extend) so schedule prep can overlap with draft_extend"（`managers/scheduler.py:4988-4990`）。也就是说 verify 一结束就把 `new_seq_lens` 放出去，让 CPU 去准备下一批，同时 GPU 继续跑 draft-extend。

**（二）grammar 的 FSM 必须在 verify 里推进。** 结构化输出要在采样前拿到 bitmask，而 bitmask 依赖**上一批**已提交的 token——overlap 下上一批的结果还在队列里没处理。解法是给 verify 传一个 barrier：`scheduler.py:4996-5003` 在 `supports_grammar_overlap()` 为真时把 `fwd_kwargs["grammar_barrier"]` 设成 `self._advance_pending_grammar`。

`_advance_pending_grammar`（`scheduler.py:2238-2246`）把队列里没处理的 decode 结果推一遍 FSM；`run_eagle_verify` 在**启动 target forward 之前**就把 `GrammarTree` 建好（`eagle_worker_common.py:628-637`），于是"推进 FSM"这坨 CPU 工作被塞进 target verify 的 GPU 时间里。哪些算法能这么干由 `SpeculativeAlgorithm.supports_grammar_overlap` 决定（`spec_info.py:165-174`，注释直说 "NGRAM drafts from a host corpus lookup, so it stays synchronous by design"），NGRAM 不在名单里——它的 draft 是 host 侧查表，没有 GPU 时间可以藏。

**（三）WAR barrier 等的是 draft-extend，不是 verify。** scheduler 用统一内存池做共享 buffer，下一轮 schedule 的写必须排在本轮"最后一次读"之后。这个"最后一次读"由 `last_shared_read_runner` 声明（`base_spec_worker.py:248-254`），EAGLE 的实参是 draft runner（`eagle_worker_v2.py:1401-1407`），注释："eagle's draft_extend runs on the draft runner"。屏障本身在 `scheduler.py:2087-2099`，拿不到细粒度 event 就退回 `wait_stream` 的粗屏障。

这三处合起来才是"投机 + overlap"的真实成本：不是简单地在 decode 循环里插一段，而是把 publish 点、语法推进点、内存屏障点全部挪位。

## 7. 采样语义：链上精确，树上近似

SGLang 有两条 verify 采样路径，语义强度不一样，这点必须说清楚。

**链上（`topk=1` + `--speculative-use-rejection-sampling`）是精确的拒绝采样。** 判据在 `python/sglang/kernels/ops/speculative/reject_sampling.py:146-150`：

```python
q_is_prob = (q > 0.0) & (q <= 1.0)
if q_is_prob & (coin * q < p):
    num_accept += 1
```

即 $u < p/q$，与 vLLM 的概率比检验同构。拒绝后从残差重采，输出分布逐 token 等于目标模型。代价是**只支持 topk=1**：`speculative_hook.py:1262-1265` 直接断言"requires `--speculative-eagle-topk=1`"，`:1286-1290` 又补一句"rejection sampling is only implemented for the linear (topk=1) chain"。默认还是关的（`fields/spec.py:117-119`，`False`）。

**树上（默认路径）是 target-only 采样。** `tree_speculative_sampling_target_only` 的 kernel 在 `python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh`，逐层在兄弟之间累积目标概率质量：

```cuda
if (has_target_mass && (coin < prob_acc / threshold_acc || target_prob_single >= threshold_single)) {
```

（`:81`）也就是"把一棵树的当前层当作候选集，按 target 概率做逆 CDF"——没有 $p/q$ 比值。残差那一段更直接，`:92` 留着一行 `// FIXME: leverage draft probs`：它把被拒兄弟的 `draft_probs` 覆写成 `target_probs`，于是 `relu(p-q)` 只在没试过的 token 上有质量。这在"draft 的质量集中在候选集上"时是对的（EAGLE 的候选就是 draft 的 top-$g$，近似的合理性来自这里），但不是链上那种逐 token 精确。

两个阈值旋钮就是这条路径的松弛量：`speculative_accept_threshold_single` / `_acc`（`fields/spec.py:109-116`）默认都是 `1.0`，调小会**主动接受**低概率 draft，是**有损**的。

剩下三条共性约束值得单独记：

- **TP 同步**：多卡上各 rank 的 `target_probs` 会有浮点差异，导致接受长度不同、`seq_lens` 发散、下一个集合通信死锁。所以采样后广播（`eagle_utils.py:859-870`）。
- **greedy 分支**：`is_all_greedy` 或 CPU/XPU（以及未开拒绝采样的 HIP）走 argmax 比对（`eagle_utils.py:703-722`），此时 `draft_probs` 根本不构造（`eagle_worker_v2.py:787-790`：`needs_draft_probs` 为假就跳过）。
- **coin 的确定性**：设了 `sampling_seed` 就从 `murmur_hash32(seed, seq_len, col)` 派生（`eagle_utils.py:454-491`），保证 CUDA graph replay 与 batch 重排下结果一致；draft 侧的随机数用另一个 domain separation 的流，"preserves the independence required by rejection sampling"。

## 8. 与 vLLM 对照

| 维度 | SGLang（本 commit） | vLLM `c7a7607` |
|---|---|---|
| 方法清单 | `SpeculativeAlgorithm` 枚举：8 个内置算法 + `NONE`，另有插件注册（`spec_info.py:31-46`、`:67-90`） | `SpeculativeMethod` 是一组 Literal + 三组别名（`vllm/config/speculative.py:85-93`） |
| 默认参数 | `_auto_choose_speculative_params` 按 arch 给 `(steps, topk, K)`：Llama/Grok1 `(5,4,8)`，其余 `(3,1,4)`（`speculative_hook.py:1471-1502`） | 由 `--speculative-config` 显式给 `num_speculative_tokens` |
| 树状 topk>1 | **原生支持**：祖先矩阵 + `build_tree_kernel_efficient` + 兄弟链检索（第 2、3 节） | **没有**：`grep -rn "tree" vllm/v1/spec_decode/` 只剩 `llm_base_proposer.py:1630-1631` 一句 FIXME（"when using tree-based specdec, adjust number of forward-passes according to the depth of the tree"） |
| draft 次数 | `num_steps - 1` 次，另加一次 draft-extend 出第一层 | `num_speculative_tokens - 1` 次串行 draft（`llm_base_proposer.py:689`） |
| 采样语义 | 树：target-only（`:81`/`:92` FIXME，近似）；链+拒绝采样：精确 $u<p/q$，但限 topk=1 | 链 + 拒绝采样，`standard` 模式精确 $u<p/q$（`rejection_sampler_utils.py:716`），greedy 走 argmax（`:634`） |
| 抽象层 | `SpecInput/SpecOutput` 一族 + `SpecInputType` 枚举（`spec_info.py:376-453`），attention backend 靠 `is_draft_input()`/`is_verify_input()` 分派相位 | `SpecDecodeBaseProposer.propose()`（`vllm/v1/spec_decode/llm_base_proposer.py:511`）+ runner 侧 `Speculator.propose()` |
| 与 overlap | 三处挪位（publish / grammar barrier / WAR barrier），见第 6 节 | 配置层开关：`ngram`、`suffix`、`custom_class` 直接关异步调度（`vllm/config/vllm.py:1693-1702`） |

1. **树是 SGLang 的主要形态，vLLM 已经退回链式。** vLLM 那边的理由是"树省不了 verify，每个候选 token 都要算一遍 $p$"；SGLang 的选择是"反正 verify 一次能并行算 $B\cdot K$ 行，那就让这 $K$ 行是一棵树"。两种取舍的分界在 $\beta$（verify 每多验一个 token 的边际成本）：带宽瓶颈下 SGLang 的树严格更优，算力瓶颈下树的额外行就是纯亏。
2. **采样语义上 vLLM 更严格，但只在他自己支持的那条路径上。** vLLM 的 `standard` 拒绝采样对链是逐 token 精确的；SGLang 的精确版本存在但被 `topk=1` 锁住，默认走的树路径是 target-only 近似。反过来说，SGLang 的近似换来的是树——这是"分布精确"与"每步产出更多"之间的正面取舍，不是实现偷懒。
3. **抽象层的取舍方向相反。** vLLM 把差异收进 proposer（`SpecDecodeBaseProposer` 一个基类 + 若干薄子类，`eagle.py` 只有 22 行）；SGLang 把差异摊到 `SpecInput` 的子类上，让 attention backend 按 `spec_input_type` 决定用哪套 mask 语义（`spec_info.py:430-446`）。前者的扩展点是"换个模型怎么猜"，后者是"换个算法怎么验"——因为树/链的 mask 布局不同，必须让 backend 参与。

## 常见坑

1. **把 `speculative_num_draft_tokens` 当加速倍数**。它只是 verify 窗口宽度；实际每步产出是 `accept_lens = num_correct_drafts + 1`，上界 $k+1$。
2. **以为 `num_steps` 次循环就是 `num_steps` 次 draft 前向**。是 `num_steps - 1` 次（`eagle_worker_v2.py:833`），第 0 层来自上一轮 draft-extend。
3. **`topk=1` 时手动把 `num_draft_tokens` 设成别的值**。启动时会被强行改回 `num_steps + 1` 并打 warning（`speculative_hook.py:1296-1307`），`_rebuild_topk1_chain_buffers` 里还有一条 assert（`base_spec_worker.py:126-129`）。
4. **开了 `--speculative-use-rejection-sampling` 又设 `topk>1`**。直接报错；而且它和 `--speculative-accept-threshold-*` 互斥（`speculative_hook.py:1272-1281`），因为拒绝采样不接受这两个松弛阈值。
5. **用 `spec_accept_rate` 当"每层命中率"**。它的分母是 `verify_ct * (num_draft_tokens - 1)`（`spec_decoding_metrics.py:43-48`），是**槽位**数，不是真实 proposal 数；树上 $K-1$ 个槽位来自 $k\cdot g$ 个候选，两者不等。
6. **以为高并发下投机一定赚**。verify 的 batch 是 $B\cdot K$，$K=8$ 时等于把 decode 的 batch 乘 8；再叠上默认被压到 48 的 `max_running_requests`，吞吐账很容易反过来。
7. **只开 overlap 不开 draft-extend 图**。draft 段在 `num_steps > 1` 时默认有图（`eagle_worker_v2.py:513`），draft-extend 段默认也有图，但后者受 backend 白名单限制（`eagle_worker_v2.py:573-613`），换个 MLA backend 就可能静默退回 eager——排障时要看启动日志里的 "Capture draft extend CUDA graph"。

## 读代码时的检查点

1. **先确认 `SpecInput` 的具体类型**。`batch.spec_info` 是 `EagleDraftInput`、`EagleDraftExtendInput` 还是 `EagleVerifyInput`，决定这次 forward 是 `DECODE`、`DRAFT_EXTEND_V2` 还是 `TARGET_VERIFY`（`spec_info.py:430-446`）。走错相位，mask 语义就全错。
2. **跟 `custom_mask` 的 numel 对账**。它必须等于 $S\cdot K + K^2\cdot B$（`eagle_utils.py:226-230`）；CUDA graph padding 时会按需 `torch.cat` 补 `True`（`eagle_info.py:141-154`），补错了不会报错，只会算错。
3. **盯 `last_shared_read_runner` 返回谁**。它决定 WAR barrier 等的是 verify 还是 draft-extend（`base_spec_worker.py:248-254`）；PP 下非末级 stage 没有 draft worker，会退回 target runner（`eagle_worker_v2.py:1405-1407`），这是 PP + 投机最容易出 rank 不一致的地方。

## 小结

- 一轮迭代是 **draft（$k-1$ 次前向，每次 $B\cdot g$ 行）→ verify（1 次，$B\cdot K$ 行）→ draft-extend（1 次，$B\cdot K$ 行）**；第 0 层候选由上一轮 draft-extend 产出，不是 draft 循环算的。
- 树由 `select_top_k_tokens` 逐层在 $g^2$ 组合里取全局 top-$g$、再由 `organize_draft_results` 跨层全局取 top-$K-1$ 得到；`topk=1` 时退化成常量 `parent_list`。
- **一次 verify 验完整棵树的机制**：把 $K$ 个节点摊平成 $K$ 行 query（`qo_indptr` 每请求步进 $K$），用一张 $K\times K$ 的祖先矩阵 $M_{ij}$（节点 $j$ 是节点 $i$ 的祖先时为 1）顶替 causal mask，于是整棵树变成一次变长 prefill，一次前向同时给出所有候选路径的 logits。
- 接受长度 `accept_lens = num_correct_drafts + 1`，宽度被 `spec_steps + 1` 卡死；KV 每步预留 `2 * max(steps*topk, num_draft_tokens)`，`req_to_token` 行多留 `4 + K`，`max_running_requests` 默认被压到 48。
- 这个 commit 上 **v1 worker 已经不存在**，v2 同时驱动 overlap 与非 overlap；两个 graph runner（draft decode / draft-extend）分开 capture 是因为每请求宽度一个是 `topk`、一个是 `num_draft_tokens`。
- 与 overlap 的咬合在三处：publish 前移到 verify 之后、grammar FSM 在 verify 内部推进、WAR barrier 等 draft-extend。
- 采样语义按路径分叉：链 + 拒绝采样精确（$u<p/q$，限 `topk=1`）；默认的树路径是 target-only 近似，kernel 里留着 `// FIXME: leverage draft probs`。

下一篇把视角从 GPU 挪到请求入口：[前端 DSL、Router 与 cache-aware 负载均衡]({{< relref "sglang-09-frontend" >}})。

> 代码基线：`sgl-project/sglang` main @ `1ef3335d`（2026-11-03）。参数默认值以 `python/sglang/srt/arg_groups/` 为准（`fields/spec.py` 的字段默认 + `speculative_hook.py` 的解析），`server_args.py` 已拆分成包，旧笔记里的行号会失效。
