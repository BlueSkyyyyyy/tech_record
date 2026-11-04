---
title: "vLLM 研读（九）投机解码：n-gram、EAGLE 与拒绝采样"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 10
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "投机解码", "EAGLE", "拒绝采样"]
categories: ["推理框架"]
---

一个 30k token 的代码仓库问答请求，decode 阶段每吐一个 token 都要把 16 GB 权重从 HBM 搬一遍——batch=1 时 GPU 的算力单元有 95% 的时间在等显存。既然瓶颈是"权重读一次只换一个 token"，那么只要能让权重读一次换回好几个 token，墙钟时间就直接除以那个倍数。

投机解码就是这笔交易：先用一个便宜的小模型（或一段字符串匹配）猜出 $k$ 个 token，再让目标模型一次前向把 $k+1$ 个位置全算出来，从猜对的地方往后接受。

**先记住一个画面**：draft 模型逐个吐 token，像实习生先在草稿纸上写；目标模型那一次前向是主编审稿，审到第 3 个 token 发现不对，就把后面全划掉——但第 3 个 token 本身由主编当场改成正确的那个。所以一次"审稿"至少产出一个 token，运气好产出 $k+1$ 个。

这篇回答三个问题：

1. 一次 verify 到底换回几个 token，什么条件下这笔账会亏？
2. 当前 main 上有哪些 proposer，它们各自靠什么"猜"？
3. 拒绝采样凭什么保证输出分布与目标模型**完全一致**，而不是"差不多"？

## 1. 先算账：一次 verify 换回几个 token

每个 draft token 被接受的条件是"它前面所有 draft 都被接受"，所以第 $i$ 个被接受的概率是 $\alpha^i$（$\alpha$ 是单 token 接受率，假设逐位置独立）。按"期望等于各位置被接受概率之和"累加，再加一个必然产出的 bonus token：

$$
E[\text{tokens/step}] = 1 + \alpha + \alpha^2 + \cdots + \alpha^{k} = \sum_{i=0}^{k}\alpha^{i}
$$

成本侧，一次 step 的墙钟时间约等于"一次 target verify + $k$ 次 draft 前向"：

$$
T_{\text{step}} \approx 1 + (\beta + c)\,k,
\qquad
S(k,\alpha) = \frac{\sum_{i=0}^{k}\alpha^{i}}{1 + (\beta+c)k}
$$

$c$ 是 draft 单 token 成本与 target 单 token 成本之比；$\beta$ 是 verify 时每多验一个 token 的**边际**成本——纯带宽瓶颈下 $\beta \approx 0$（权重只读一次），纯算力瓶颈下 $\beta \to 1$。下面这张表取 $c=0.1$（draft 是 target 的 1/10 大小），$\alpha$ 逐位置恒定：

| $k$ | $E$ @α=0.7 | $E$ @α=0.9 | $S$ @α=0.7, β=0 | $S$ @α=0.9, β=0 | $S$ @α=0.7, β=0.4 | $S$ @α=0.9, β=0.4 |
|---|---|---|---|---|---|---|
| 1 | 1.700 | 1.900 | 1.55 | 1.73 | 1.13 | 1.27 |
| 3 | 2.533 | 3.439 | 1.95 | 2.65 | 1.01 | 1.38 |
| 5 | 2.941 | 4.686 | 1.96 | 3.12 | 0.84 | 1.34 |
| 8 | 3.199 | 6.126 | 1.78 | 3.40 | 0.64 | 1.23 |

三行结论，全部可以从表里读出来：

- **α=0.7 时 $k$ 从 3 加到 5 几乎没有收益**（1.95→1.96），加到 8 反而亏（1.78）。因为 $0.7^6 \approx 0.12$：第 6 个 draft 只有 12% 概率被接受，却照样付 $c=0.1$ 的成本。
- **α=0.9 时 $k=8$ 还在涨**（3.40，$k=10$ 也只到 3.43）。接受率高时账本收敛慢，值得多猜，但边际收益已经很薄。
- **β 一涨，最优 $k$ 立刻往回缩**：β=0.4 时 α=0.7 的最优解是 $k=1$，$k=5$ 已经亏到 0.84。

"什么时候不赚"可以写成两个不等式：

1. $k=1$ 时 $S=(1+\alpha)/(1+\beta+c)$，所以 **$\alpha < \beta + c$ 就不赚**。
2. 完全算力瓶颈（$\beta = 1$）下 $S = \sum\alpha^i / (k+1+ck) < 1$ **恒成立**——因为 $\sum_{i=0}^{k}\alpha^i < k+1$，验 $k+1$ 个位置就要付 $k+1$ 份算力，换回的东西严格更少。这是投机解码在高并发下必然失效的数学原因，不是实现问题。

> **类比的边界**：实习生写草稿、主编审稿这个比喻有个漏洞——主编改稿不需要重读一遍所有资料。真实的 verify 要把 $k+1$ 个位置的 KV 全部写进块表，$k$ 个 draft 位置在 target 的 KV cache 里是实打实占地方的，后面会算这笔显存。

## 2. 方法清单：`SpeculativeMethod` 里现在有什么

以 `vllm/config/speculative.py:85-93` 的枚举为准：`ngram`、`draft_model`、`suffix`、`custom_class`，加上三组 Literal 别名（`EagleModelTypes`、`NgramGPUTypes`、`DSparkModelTypes`）。

一个必须说清楚的事实：**两年前笔记里的 `medusa` 和 `mlp_speculator` 在当前 main 上已经不存在了**——它们不在枚举里，`grep -rn "medusa\|mlp_speculator" --include=*.py vllm/` 在这个 commit 上返回空。代码里没有留下移除说明；从留下来的方法看，方向是 EAGLE 系的 hidden-state draft 加并行 draft（DFlash/DSpark），而不是"多个 LM head 打分"。

| method | 需要 draft 权重 | 靠什么猜 | 适用负载 | 主要风险 |
|---|---|---|---|---|
| `ngram` | 否 | 本请求历史里的 n-gram 重复 | 代码编辑、引用、摘要 | 开放生成命中率≈0；CPU 侧单线程 |
| `ngram_gpu` | 否 | 同上，匹配搬到 GPU | 同上，高 QPS | 两套实现（V1 numba / V2 GPU），默认窗口与线程行为不同 |
| `suffix` | 否 | 后缀树（跨请求语料） | 模板化、重复问答 | V1 依赖 `arctic-inference`（`vllm/v1/spec_decode/suffix_decoding.py:28`） |
| `draft_model` | 是（独立小模型） | 独立模型自回归 | 通用，draft 可换 | 词表要一致；异构词表只能配 greedy draft |
| `eagle` | 是（EAGLE 头） | target 最后一层 hidden state | 通用，官方最推荐 | draft 与 target 必须同家族 |
| `eagle3` | 是（EAGLE3 头） | target 多层 hidden state 拼接 | 通用，接受率通常更高 | target 要多输出中间层，prefill 变贵 |
| `mtp` | 否（在 target 里） | target 自带的 MTP 层 | 原生支持 MTP 的模型 | 接受率由训练决定，无法换 draft |
| `dflash` | 是 | 并行 draft + 非因果 attention | 低延迟场景 | 要单独的 attention backend（`vllm/config/speculative.py:495-497`） |
| `dspark` | 是 | block 级并行 draft | Qwen3-Omni 等 | `num_speculative_tokens` 必须等于训练 block size（`vllm/config/speculative.py:236-240`）；`dspark_draft_topk` 要求 draft TP=1（`:704`） |
| `extract_hidden_states` | 否 | ——（不产 draft） | 抓 hidden state 训 EAGLE | `proposes_draft_tokens = False`（`vllm/v1/worker/gpu/spec_decode/extract_hidden_states_config.py:21`） |
| `custom_class` | 看实现 | 用户自己的 `propose()` | 实验 | 与异步调度不兼容，需 V2 runner |

## 3. n-gram：不训练模型，只找"自己说过的话"

`ngram` 的配置只有三个旋钮：`prompt_lookup_min` / `prompt_lookup_max`（匹配窗口的下界与上界）和 `num_speculative_tokens`（往后取几个）。V2 的 speculator 在用户不填时默认取 5/5（`vllm/v1/worker/gpu/spec_decode/ngram/config.py:21-24`，代码里还留着 "arbitrarily chosen" 的 TODO）；V1 的 `NgramProposer` 则把两者写成硬断言（`vllm/v1/spec_decode/ngram_proposer.py:15-16`）。

匹配算法不复杂：把该请求的 token 序列**反转**，问题变成"找最长的、等于反转序列前缀的后缀"，也就是经典 KMP 的 `lps`（longest proper prefix which is also suffix）数组（`vllm/v1/spec_decode/ngram_proposer.py:243`）。命中后从匹配位置往后取最多 $k$ 个 token 作为 draft（`:293`）：

```python
# 反转后找最长 ngram：tokens[0:v] == tokens[i+1-v:i+1]
lps = np.zeros(max_ngram, dtype=np.int32)
...
if prev_lps >= longest_ngram:
    longest_ngram = prev_lps
    position = i
```

关键限制藏在入参里：`context_token_ids = token_ids_cpu[idx, :num_tokens]`（`vllm/v1/spec_decode/ngram_proposer.py:195`）——**只在本请求自己的 token 里找**，不跨请求。所以 ngram 的本质是"这个请求接下来要写的内容，是不是它前面已经写过/读过的内容的重复"。

- **对代码类负载特别有效**：改一个函数时，函数签名、参数名、括号结构、import 块、被引用的常量都已经在 prompt 里出现过，模型接下来吐的 token 有很高比例是"复制粘贴 + 局部改动"。同理，摘要/改写类任务是抽取式复述，RAG 是在复述检索到的原文。
- **什么时候完全无效**：开放式创作、数学推理（每一步都是全新的 token 序列）、多轮闲聊。还有一个容易误判的场景——单轮问答但答案不在 prompt 里，此时 ngram 的命中率接近 0，却仍然每步都做一次匹配。
- **它的成本不是零**：`num_numba_thread_available = min(1, cpu_count // 2)` 再整除 TP（`vllm/v1/spec_decode/ngram_proposer.py:48`），也就是**每张卡只用一个线程**做匹配，代码里明确留着"把上限从 1 提到 8"的 TODO（`:46`）。batch 总 token 数低于 8192 时还会退化成单线程（`:36`）。高 QPS 下这个 proposer 会成为 CPU 侧的串行点。

## 4. EAGLE / MTP：拿 target 的 hidden state 当地基

`EagleProposer` 整个类只有 22 行（`vllm/v1/spec_decode/eagle.py`），全部内容是继承 `SpecDecodeBaseProposer` 并把 `pass_hidden_states_to_model=True` 传下去（`:20`）。对照 `DraftModelProposer` 传的是 `False`（`vllm/v1/spec_decode/draft_model.py:30`）——这一行就是 EAGLE 与"随便找个小模型"的全部区别：

**EAGLE 的 draft 输入不只有 token embedding，还有 target 的 hidden state**。每一步 draft 前向都把上一轮的 hidden state 塞进 `model_kwargs["hidden_states"]`（`vllm/v1/spec_decode/llm_base_proposer.py:738`）。EAGLE3 更进一步，用 `combine_hidden_states` 把 target 低/中/高层的 hidden state 拼起来（`:549`），所以 target 在 prefill 时必须额外吐出中间层——`extract_hidden_states` 这个"不产 draft 的方法"就是为此存在的（`vllm/v1/worker/gpu/spec_decode/extract_hidden_states_config.py:21` 把它标成 `proposes_draft_tokens = False`）。

draft 循环是清楚的串行链（`vllm/v1/spec_decode/llm_base_proposer.py:689`）：

```python
for token_index in range(self.num_speculative_tokens - 1):
    input_ids = draft_token_ids_list[-1].int()
    ...
    model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
    ret_hidden_states = self.model(**model_kwargs)
```

**$k$ 个 draft token = $k$ 次 draft 前向**，不是一次。这正是 $c\,k$ 那一项的来源，也是 `parallel_drafting`（P-EAGLE / DFlash，`vllm/config/speculative.py:561-569`）想省掉的东西：一次前向出多个 token，把 $k$ 次串行换成 1 次。

**MTP 没有"选 draft 模型"这回事**：draft 就是 target checkpoint 里的 MTP 层，`default_model` 直接返回 target 的权重路径（`vllm/v1/worker/gpu/spec_decode/mtp/config.py:51-55`）。所以 MTP 的接受率在训练时就定了，推理侧只能调 $k$。

**树去哪了**：EAGLE 论文里用 tree attention 一次提交多条候选路径。当前 main 上这条路径已经退场——`grep -rn "tree_attn\|tree_attention" vllm/` 为空，只剩两处措辞残留：`vllm/v1/attention/backend.py:756-757` 的 docstring 说 `draft_index` "for tree-based attention ... refers to the draft attempt for the i-th level in the tree"，以及 `vllm/config/speculative.py:544` 的 "non-tree speculation"。现在的 draft 是**链式**的，verify 沿链逐 token 累积。

为什么树上必须按路径逐个验证（即使实现回来了也一样）：接受概率是沿路径的**连乘**，一条路径只有"前缀全接受"才有意义；分叉之后的目标分布是条件分布 $p(\cdot\mid \text{prefix})$，残差也要按前缀重新归一化。更根本的是——树只省 draft 的 $k$ 次前向，**不省 verify**：每个候选 token 都要算一遍 $p$。所以工程上"并行 draft + 链式 verify"比"树"更划算，这也是 vLLM 的选择。

## 5. 拒绝采样：输出分布为什么不变

入口是 `RejectionSampler._verify`（`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:255`），核心 kernel 是 `rejection_sample`（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:1109`）里的串行循环 `for i in range(num_draft_tokens)`（`:598`）。两种模式在这个循环里分叉：

**greedy（温度 0）**：接受条件就是"draft 等于 target 的 argmax"（`:634`）。拒绝时不需要重采样——target argmax 已经在手，直接写进输出（`:751`）。这是精确的，因为 greedy 的目标分布是 argmax 处的单点分布。

**随机采样**：接受条件是

```python
# Probability ratio test: p(x) > u * q(x)
accepted = target_logprob > tl.log(u) + draft_logprob
```

（`:716`）也就是 $u < p(x)/q(x)$，等价于以 $\min(1, p/q)$ 的概率接受。拒绝后从残差分布重新采样（`:962-972`）：

$$
r(x) = \frac{\max\big(p(x) - q(x),\, 0\big)}{\sum_y \max\big(p(y)-q(y),\,0\big)}
$$

正确性就是三步代数。记 $\Sigma_{\min}=\sum_y\min(p(y),q(y))$，注意 $\sum_y\max(p-q,0)=1-\Sigma_{\min}$：

$$
q(x)\min\!\Big(1,\frac{p}{q}\Big) + (1-\Sigma_{\min})\cdot r(x)
= \min(p,q) + \max(p-q,\,0) = p(x)
$$

所以输出分布**逐 token 精确等于**目标模型，不是近似。

**默认配置其实更简单**：`draft_sample_method` 默认 `"greedy"`（`vllm/config/speculative.py:694`），此时 draft 分布是 one-hot，`_sample_draft_tokens` 直接返回 `draft_probs=None`（`vllm/v1/spec_decode/llm_base_proposer.py:477-483`），kernel 走 one-hot 分支——残差就是"把被拒 token 的概率抹掉的目标分布"（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:974-981`）。这解释了一个常见误解：**默认配置下不需要存 draft logits，也就没有额外显存开销**；只有 `probabilistic` 才要为 draft logits 付显存。

**bonus token**：$k$ 个 draft 全被接受时，第 $k+1$ 个 logit 行白送一个 token，kernel 里用 `is_bonus` 判断（`:884`，走"直接用 target logits"的分支 `:926-927`）。这就是公式里那个 `+1`，也是 `mean_acceptance_length` 的下界是 1.0 而不是 0 的原因。

**verify 的显存账**：一次 verify 的 logits 是 `[num_logits, V]` 的 fp32。BS=256、k=5、词表 131k 时是 $256\times6\times131072\times4 \approx 768$ MiB，vLLM 因此按 1 GiB 上限分块（`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:33` 的 `MAX_CHUNK_BYTES = 2**30`，分块逻辑在 `_verify_in_chunks`，`:306`）。词表 256k 的模型会直接切成 2 块——这是"高并发下投机更贵"的另一个隐性来源。

**变体们**：`rejection_sample_method` 有 `standard` / `synthetic` / `block` / `fly`（`vllm/config/speculative.py:622`）。`block` 是 Sun et al. 的块级验证（kernel 里引了 arXiv 2403.10444，`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:646`），接受判据从单 token 的 $p/q$ 变成带前缀联合比的 $h = \frac{r}{r + 1 - p_\tau}$；`synthetic` 按配置的接受率接受（用于压测和模拟）；`fly` 是按熵门控的延迟判定，**有损**，会改变输出分布。只有 `standard` 是无损的那一个。

## 6. 调度器集成：draft token 是"借来的" KV

调度器每步把要验的 draft 放进 `scheduled_spec_decode_tokens`（`vllm/v1/core/sched/scheduler.py:1009-1024`），条数上限是 `num_new_tokens + num_computed_tokens - num_tokens - num_output_placeholders`，超出的直接截断丢弃。

**为什么 draft 要占 target 的 KV**：draft 会在 target 的 query 范围之外写 KV。EAGLE 因此把 `num_lookahead_tokens` 定义成 `num_speculative_tokens`（`vllm/v1/worker/gpu/spec_decode/eagle/config.py:56-57`），`VllmConfig.num_lookahead_tokens` 的 docstring 把这件事说得很直白——"the drafter writes KV for positions beyond the target model's query range"（`vllm/config/vllm.py:643-647`）。这笔预留通过 `allocate_slots(..., num_lookahead_tokens=...)` 落到块表（`vllm/v1/core/kv_cache_manager.py:403-405`，注释点名 "used by spec decode proposers with kv-cache such as eagle"）。

**准入账**可以算得很具体（`vllm/v1/core/sched/scheduler.py:371-376`）：

```python
num_spec_decode_tokens = (
    1 + self.num_spec_tokens + self.num_lookahead_tokens
    if self.num_spec_tokens else 0)
self._spec_decode_step_blocks = sum(
    cdiv(num_spec_decode_tokens, manager.block_size) ...)
```

`1`（bonus）+ $k$（待验 draft）+ $k$（lookahead）= $2k+1$ 个 token。k=5、block_size=16 → 1 块/请求；k=8 → 17 个 token → 2 块/请求。这就是"开了投机，`max_num_seqs` 的实际可达值会下降"的来源：每个请求每步要多押 1–2 个块。

**验证失败后的回收**（`vllm/v1/core/sched/scheduler.py:2347-2357`）：

```python
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
...
request.num_computed_tokens -= num_rejected
request.num_output_placeholders -= num_rejected
```

被拒 token 的 KV 还躺在块里，但 `num_computed_tokens` 退回去了，下一轮会被覆盖重写。前缀缓存这边也有一条明确规则：**只采纳 verified token 的块**（`vllm/v1/core/kv_cache_manager.py:461-464`，注释原文 "we only cache the verified tokens (by capping the number at `request.num_tokens`)"），所以被拒的 KV 不会污染 prefix cache 的哈希。

**结构化输出是另一条回滚路径**：语法不允许的 draft 被 `trim_spec_tokens` 裁掉，用 `-1` 补齐占位（`vllm/v1/core/sched/scheduler.py:2909-2917`），kernel 里 `is_valid_draft = draft_sampled >= 0` 直接把占位判负、不重采样（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:606-613`）。占位数记在 `num_invalid_spec_tokens` 里。

**异步调度**：V1 runner 下这批方法能开异步——`eagle`/`eagle3`/`mtp`/`dflash` 及 MTP 一族、`ngram_gpu`、`draft_model`、`dspark`；`ngram`、`suffix`、`custom_class` 不在名单里，会被自动关掉异步调度（`vllm/config/vllm.py:1693-1702`），`disable_padded_drafter_batch=True` 和 PP>1 同理（`:1701-1704`）。V2 runner 上这段检查整个跳过（`:1691` 的 `if not self.use_v2_model_runner`）。注意这是**配置层的名单，代码没写原因**；合理的解释是异步调度下第 N 步的 sampled token 还没回来，draft 只能对着占位符跑 padded drafter batch，而 CPU 侧 proposer 与"关掉 padding"都进不了这条流水线。

## 7. 调参：先量 $\alpha$，再谈 $k$

**先估 $\alpha$**。账本那张表说明一切：α<0.6 时 $k$ 别超过 2；α 在 0.7 附近时 $k=3$ 已经饱和；α>0.85 时 $k=5\sim8$ 才吃得满。

**动态 $k$**。`num_speculative_tokens_per_batch_size` 用三段组 `(bs_start, bs_end, K)` 按并发给不同的 $K$（`vllm/config/speculative.py:578`）。官方文档把动机写得很直白：verify 的有效 batch 是 $BS\times K$，一旦越过临界点，投机就从"降 TPOT"变成"升 TPOT"（`docs/features/speculative_decoding/dynamic_speculative_decoding.md`）。该功能在 DP>1 时会被自动禁用（各 rank 选的 $K$ 不同会让 DP 集合通信发散）。

**draft 模型怎么选**。EAGLE 必须与 target 同家族（同 tokenizer 是硬约束）；`draft_tensor_parallel_size` 只能是 1 或 target 的 TP（`vllm/config/speculative.py:477-481`），实践上默认 1——draft 太小，切 TP 的通信开销比算得快。

**接受率怎么观测**。服务端三个计数器：`vllm:spec_decode_num_drafts`、`vllm:spec_decode_num_draft_tokens`、`vllm:spec_decode_num_accepted_tokens`（`vllm/v1/spec_decode/metrics.py:229-231`），还有 `num_accepted_tokens_per_pos`（同文件 `vllm/v1/spec_decode/metrics.py:255`）能看到"从第几位开始掉"。请求级可以开 `--per-request-spec-decode-metrics`，返回里的 `mean_acceptance_length = 1 + accepted/steps`（`docs/features/speculative_decoding/acceptance_metrics.md`）——它就是账本里的 $E[\text{tokens/step}]$，直接和表里的数字对。

**长上下文与高并发为什么会亏**。上下文越长，verify 时 attention 的 KV 读取占比越高，$\beta$ 被推上去；并发越高，verify 的 batch 从 $BS$ 变成 $BS\times(k+1)$，decode 从带宽瓶颈滑向算力瓶颈，$\beta \to 1$——而 $\beta=1$ 时按第 1 节的不等式，投机**恒亏**。这也是官方把投机定位成 "medium-to-low QPS、memory-bound 负载"的降时延手段的原因（`docs/features/speculative_decoding/README.md` 开头）。

## 常见坑

1. **把 `num_speculative_tokens` 当加速倍数**。它只是上限，实际每步产出是 $1+\alpha+\cdots+\alpha^k$；$k=5$ 在 α=0.5 时只有 1.97 个 token。
2. **用 ngram 跑开放式生成，然后断定"投机没用"**。ngram 只在**本请求历史**里找重复（`vllm/v1/spec_decode/ngram_proposer.py:195`），答案不在 prompt 里的任务命中率接近 0。
3. **忘了 bonus token**。`mean_acceptance_length` 最小是 1.0；把 `accepted/steps` 当成长度会少算 1。
4. **以为默认配置要存 draft logits**。默认 `draft_sample_method="greedy"`，`draft_probs=None`（`vllm/v1/spec_decode/llm_base_proposer.py:477-483`），不占显存。
5. **开着 `--enforce-eager` 测投机收益**。draft 是 $k$ 次串行小前向，正是 CUDA Graph 收益最大的形态，关掉图会同时冤枉 draft 和 target。
6. **为了排障加 `--speculative-config '{"disable_padded_drafter_batch": true}'`**，顺手把异步调度也关了（`vllm/config/vllm.py:1701-1702`），然后奇怪吞吐掉了。
7. **用 `num_draft_tokens` 当分母算接受率**。结构化输出场景下分母里混着 `-1` 占位符，得先减掉 `num_invalid_spec_tokens`。

## 小结

- 期望产出是几何级数 $1+\alpha+\alpha^2+\cdots$，成本是 $1+(\beta+c)k$；$c=0.1$ 时 α=0.7 的最优 $k$ 是 3（1.95×），α=0.9 时 $k=8$ 还有 3.40×（尚未饱和）。
- 不赚的判据有两条硬边界：$k=1$ 时 $\alpha<\beta+c$；完全算力瓶颈（$\beta=1$）下**恒不赚**，因为 $\sum\alpha^i < k+1$。
- 当前 main 的 `SpeculativeMethod` 只有 `ngram`/`ngram_gpu`/`suffix`/`draft_model`/`custom_class` 加 EAGLE 一族（含 MTP、DFlash、DSpark）；`medusa` 与 `mlp_speculator` 已经不在代码里。
- ngram 是 KMP 在本请求历史上找"最长后缀=前缀"（`vllm/v1/spec_decode/ngram_proposer.py:243`），对代码/摘要类负载有效，对开放式生成无效，且 CPU 侧被限成每卡 1 个线程（`:48`）。
- EAGLE 与 draft model 的一行之差是 `pass_hidden_states_to_model`（`vllm/v1/spec_decode/eagle.py:20` vs `vllm/v1/spec_decode/draft_model.py:30`）；$k$ 个 draft 是 $k$ 次串行前向（`vllm/v1/spec_decode/llm_base_proposer.py:689`），这是 $ck$ 那一项的物理来源。
- 拒绝采样每步只做两件事：以 $\min(1,p/q)$ 接受、以 $\max(p-q,0)$ 归一化后重采（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:716`、`:962-981`），所以输出分布逐 token 精确等于目标模型；树上多候选也必须按路径连乘验证，省不了 verify。
- 每个请求每步要押 $2k+1$ 个 KV slot（`vllm/v1/core/sched/scheduler.py:371-376`），被拒部分靠 `num_computed_tokens -= num_rejected` 回收（`:2355`），prefix cache 只采纳 verified 的块（`vllm/v1/core/kv_cache_manager.py:461-464`）。

下一篇把视角从单卡单请求拉到多卡：[分布式：TP/PP/DP/EP 与 Executor 的进程账]({{< relref "vllm-10-distributed" >}})。

> 代码基线：`vllm-project/vllm` main @ `c7a7607`（2026-11-03）。官方文档：<https://docs.vllm.ai/en/latest/features/speculative_decoding/>（方法选型表与各方法示例）、仓库内 `docs/features/speculative_decoding/acceptance_metrics.md`（接受率指标定义）。
