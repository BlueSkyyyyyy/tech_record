---
title: "vLLM 研读（八）采样：logits 处理、logprobs 与结构化输出"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 9
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "采样", "logprobs", "结构化输出", "xgrammar"]
categories: ["推理框架"]
---

一次 decode step，256 条请求的 forward 在 15 ms 内跑完，最后交到采样器手里的是一个 `256 × 131072` 的 fp32 张量——128 MiB。从这 128 MiB 到 256 个 token id，看起来只是一次 argmax 或一次 multinomial；实际上有七八道手在上面改数，每一道都能改变输出，其中几道能把这一步的时延翻倍。

先记住一个画面：`Sampler.forward` 拿到 logits 后做的第一件事，不是缩放、不是惩罚，而是**先按原始 logits 算一遍 log_softmax 并留在旁边**（`vllm/v1/sample/sampler.py:146-155`）。因为 V1 的语义是：返回给用户的 logprobs 必须是"没被任何处理器碰过"的值。这一个决定，把采样从"一步变换"变成了"两条数据流"。

这一篇回答三个问题：

1. 处理器的执行顺序为什么不能随便换，换一步会改什么？
2. `logprobs` 从 top-20 变成全量，开销差几个数量级，差在哪？
3. 结构化输出的 bitmask 是怎么从语法变成 `-inf` 的，代价在 CPU 还是 GPU？

## 1. 一条流水线，七道手

`vllm/v1/sample/sampler.py:80-115` 的类 docstring 就是权威顺序，压缩成一张流程：

```text
logits (bf16/fp16)
  ├─ [0] 若请求 logprobs: 先算 raw_logprobs（log_softmax）并保存
  ├─ [1] 转 fp32
  ├─ [2] allowed_token_ids 白名单 -> -inf
  ├─ [3] bad_words 黑名单 -> -inf
  ├─ [4] non-argmax-invariant 处理器（min_tokens、logit_bias）
  ├─ [5] penalties（repetition / frequency / presence）
  ├─ [6] 分支：all_greedy 直接 argmax 返回
  │       否则 temperature -> argmax-invariant 处理器（min_p）
  │            -> top-k / top-p -> 采样
  └─ [7] gather top-N logprobs + sampled token 的 rank
```

代码上是三段：`forward`（`sampler.py:129-211`）负责 [0]–[1] 和收尾，`apply_logits_processors`（`sampler.py:436-482`）负责 [2]–[5]，`sample`（`sampler.py:308-366`）负责 [6]。

**第一处不可交换**：greedy 分支。`all_greedy` 为真时，`sample` 在算完 argmax 后立刻返回，temperature 和 min_p 都不执行（`sampler.py:321-335`）。这不是偷懒，是正确性——min_p 声明自己是 argmax 不变的（`vllm/v1/sample/logits_processor/builtin.py:47-49`），对 argmax 没有影响。但如果 batch 里混了 greedy 和随机请求，greedy 行也必须走完整条链路，因为 kernel 是批处理的；最后再用 `torch.where` 按 `temperature < 1e-5` 挑回 greedy 结果（`sampler.py:360-365`）。所以省钱的粒度是**整批**：批里只要混进一个温度 0 的请求，min_p 和 top-k/top-p 的 kernel 就照跑，一分钱都省不掉。

**第二处不可交换**：penalties 必须在 temperature 之前。frequency/presence 是 logit 空间的加减法（`vllm/model_executor/layers/utils.py:76-79`），如果先除以 $T$ 再减，同样的 `frequency_penalty=0.5` 在 $T=0.7$ 时实际强度会被放大 $1/0.7$ 倍。V1 的选择是"惩罚作用在原始 logit 上"，与温度解耦。

**第三处不可交换**：所有 mask（allowed ids、bad words、惩罚）必须在 min_p 和 top-p 之前。min_p 的阈值是**相对当前行最大概率**的，`masked_fill` 之后行内 max 会变；顺序反了，阈值就基于一个已经被污染的分布。

## 2. 处理器对照表

| 处理器 | 代码位置 | 作用 | 开销 / 风险 |
|---|---|---|---|
| allowed_token_ids | `sampler.py:456-457` | 白名单外的 token 置 `-inf` | 首次用到才分配 `(max_num_reqs, vocab)` bool 掩码（`gpu_input_batch.py:443-451`），GPU + CPU 各一份；128k 词表 × 1024 槽位 = 128 MiB |
| bad_words | `sampler.py:459-461` | 匹配到前缀就把下一个 token 置 `-inf` | 纯 Python 双层循环，逐请求逐词匹配（`vllm/v1/sample/ops/bad_words.py:14-27`），batch 大时是 CPU 热点 |
| logit_bias / min_tokens | `sampler.py:463-465` | 加偏置 / 未到最小长度前禁 stop token | 非 argmax 不变，greedy 也要执行；只用稀疏索引写少数位置（`builtin.py:159-162`、`:279-284`），几乎免费 |
| repetition / frequency / presence | `sampler.py:467-468` | 见 §3 公式 | **每步重建 `(batch, vocab+1)` int64 计数张量**，见 §7 账本 |
| temperature | `sampler.py:339-342` | logits 除以 $T$ | 一次原地除法；greedy 行把 $T$ 钳成 1.0（`sampler.py:298-300`） |
| min_p | `vllm/v1/sample/logits_processor/builtin.py:102-116` | 概率低于 `min_p × max_p` 的 token 置 `-inf` | argmax 不变，greedy 时整段跳过；随机分支里多一次 softmax + 比较 |
| top-k | `topk_topp_sampler.py:550-556` | 只留最大的 k 个 | 需要排序（native 路径）或 pivot 选择（Triton 路径） |
| top-p | `topk_topp_sampler.py:558-565` | nucleus，累计概率到 p | 同样需要排序；p=1 时 vLLM 直接不启用（`gpu_input_batch.py:404-406`） |
| thinking budget | `sampler.py:469-481` | 思维链 token 预算到点后强制收尾 | 需要每步维护请求状态机（`vllm/v1/sample/thinking_budget_state.py`） |

顺带纠正一个常见印象：**locally typical sampling 在 V1 的当前 main 上已经没有了**。在 `vllm/` 下 grep `typical_p` 零命中，`SamplingParams` 没有这个字段，`v1/sample/logits_processor/builtin.py` 里也没有对应处理器——参数和实现都没了。

## 3. 数值细节

**temperature**：原地除，`logits.div_(temp.unsqueeze(1))`（`sampler.py:290-300`）。greedy 行先把温度钳到 1.0，避免除零，也避免 `-inf` 被放大成 NaN。

**top-p 的 nucleus**：注意 V1 是**升序**排序，然后从"小概率尾巴"这头切：

```python
# vllm/v1/sample/ops/topk_topp_sampler.py:548-565
logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
...
probs_sort = logits_sort.softmax(dim=-1)
probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
top_p_mask[:, -1] = False        # 至少留一个
logits_sort.masked_fill_(top_p_mask, -float("inf"))
```

升序 cumsum 的语义是"从最小的 token 开始累计丢弃的质量"：累计质量 ≤ $1-p$ 的那些 token 全部丢掉，剩下的质量恰好 $\ge p$——和"降序取到累计 $\ge p$"等价，但少一次翻转，而且 `out=probs_sort` 直接在原地累加，省一份显存。`top_p_mask[:, -1] = False` 是兜底：即使 $p=0$，也至少留一个 token，否则整行全 `-inf`，softmax 出 NaN。

**top-k**：升序排序后，第 $n-k$ 个位置的值就是阈值，`logits_sort < 阈值` 的全丢掉（`topk_topp_sampler.py:550-556`）。在 CPU 后端且只有 top-k 时走 `apply_top_k_only`（`topk_topp_sampler.py:571-590`），用 `torch.topk` 避免全词表排序（docstring 提醒这条路含一次设备→主机同步，会伤 async scheduling）。

**min-p**：阈值随行内最大概率浮动：

$$
\text{keep}(i) \iff p_i \ge \text{min\_p} \cdot \max_j p_j
$$

实现就是一次 softmax + `amax` + 比较（`builtin.py:106-116`）。它比 top-p 便宜（不用排序），也比 top-p 更"自适应"：模型越确定，砍得越狠。

**惩罚三项**（`vllm/model_executor/layers/utils.py:43-80`，repetition 在 CUDA kernel 里 `csrc/sampler.cu:38-43`）：

$$
\text{logit}_i \leftarrow
\begin{cases}
\text{logit}_i / r & \text{logit}_i > 0,\ r>1\\
\text{logit}_i \times r & \text{logit}_i \le 0
\end{cases}
\quad\text{（repetition，仅对出现过的 token）}
$$

$$
\text{logit}_i \mathrel{-}= f \cdot \text{count}_i, \qquad
\text{logit}_i \mathrel{-}= p \cdot \mathbb{1}[\text{count}_i > 0]
$$

三者都依赖"这个词在本请求里出现过几次"，而次数的来源是 prompt 和已生成 token 的 bin count——这正是 §7 那笔账的源头。

**精度**：logits 一进来就 `to(torch.float32)`（`sampler.py:157-158`），`compute_logprobs` 用 `log_softmax(dtype=torch.float32)`（`sampler.py:368-370`），softmax 也显式指定 fp32。也就是说**张量本身是 bf16，但所有概率运算在 fp32 里做**。默认不做 fp64 累计；只有 `use_fp64_gumbel` 打开时，指数噪声才用 fp64（`vllm/config/model.py:308`，注释里明说"显著降低吞吐"），用于保住 fp32 随机数的下尾事件。

**采样的 kernel 分支**：核心是 Gumbel-max 的指数竞赛——对每个 token 抽 $q_i \sim \text{Exp}(1)$，取 $\arg\max_i\ p_i / q_i$（`topk_topp_sampler.py:600-606`）。这一招把"按分布抽一次"变成"整行 argmax"，避免了 `torch.multinomial` 的 CPU 同步。三条路径：

- native：排序 + softmax + 上面的竞赛（`forward_native`，`topk_topp_sampler.py:295-316`）；
- FlashInfer：拒绝采样，不做全排序，只保证统计等价，**不保证逐位相同**（`topk_topp_sampler.py:634-671`）；
- Triton：`topk_topp_triton.py` 按 Qrita 那篇的 pivot 截断 + 选择，避免全词表排序（文件头 `topk_topp_triton.py:1-7`）。

## 4. logprobs：V1 先算原始值

这是 V1 和 V0 最容易踩的语义差异，代码里专门留了注释：

```python
# vllm/v1/sample/sampler.py:142-145
# NOTE(woosuk): Use the original logits (before any penalties or
# temperature scaling) for the top-k logprobs.
# This is different from the V0 sampler, which uses the logits that
# is used for sampling (after penalties and temperature scaling).
```

四种模式在 `vllm/config/model.py:102-107` 定义：`raw_logits`、`raw_logprobs`（默认）、`processed_logits`、`processed_logprobs`。后两种要等 top-k/top-p 之后再算，因此 `PROCESSED_LOGPROBS_MODES` 会直接改变 kernel 选择（见 §7）。

三种开销形态：

1. `logprobs=k`（k ≤ `max_logprobs`，默认 20）：`torch.topk` 取前 k，再 `gather` 出采样 token 的值，并用 `batched_count_greater_than` 数出它的排名（`sampler.py:372-421`）。后者是 `@torch.compile` 的小 kernel（`vllm/v1/sample/ops/logprobs.py:10-27`），专门避免 `(x >= v).sum(-1)` 产生临时张量。这里还有一处细节：`mark_unbacked(logprobs, 0)`（`sampler.py:405-412`），把 batch 维标成符号化，防止 batch 从 1 变 2 时 dynamo 重新特化、重新编译。
2. `logprobs=-1`：不排序、不排名，直接把整个 `(batch, vocab)` 的 `raw_logprobs` 塞进 `LogprobsTensors`（`sampler.py:184-188`）。形状就是 §7 那笔账。注意它要先过 `max_logprobs` 校验，默认 20 会直接拒绝，必须显式调大 `--max-logprobs`（`sampling_params.py:927-938`）。
3. `logprobs=0`：只返回采样 token 自己的 logprob——`vllm/v1/sample/metadata.py:25` 的注释写得很清楚：`None` 表示不要，`0` 表示只要采样 token。

到 engine 进程之后还有一层：`LogprobsProcessor` 要把 token id 反解成字符串，并且处理 byte-fallback 导致的多字节 UTF-8 被拆开的问题（`vllm/v1/engine/logprobs.py:91-124`、`_correct_decoded_token` 在 `:261-323`）；`cumulative_logprob` 也在这一步累加（`vllm/v1/engine/logprobs.py:113-114`）。**这部分是纯 CPU**，而且每个 token 都要 detokenize 一次——它是 logprobs 在小 batch 下也不便宜的原因。

## 5. 结构化输出：bitmask 的一生

`StructuredOutputManager` 是 engine 级单例，而且**一个 engine 只能有一个后端**：第一个请求决定 backend，后续请求换成别的会直接报错（`vllm/v1/structured_output/__init__.py:144-151`）。可选 `xgrammar`、`guidance`、`outlines`、`lm-format-enforcer`；`--structured-outputs-config.backend` 默认 `auto`，按请求内容挑后端（`docs/features/structured_outputs.md:56-59`），其中 lm-format-enforcer 已被标为 deprecated（同文件 `:23-33`）。

一次约束解码的完整链路：

1. **编译语法**。`grammar_init` 把编译任务丢进线程池，返回一个 `Future`（`__init__.py:196-205`），请求状态机因此多了一个 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 态。编译本身有缓存：`GrammarCompiler(cache_enabled=True, cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024)`（`backend_xgrammar.py:216-223`），环境变量默认 512 MiB（`vllm/envs.py:1757`）。
2. **分配 bitmask**。`allocate_token_bitmask(max_batch_size * (1 + max_num_spec_tokens))`（`__init__.py:403-412`），底层是 `xgr.allocate_token_bitmask`（`backend_xgrammar.py:208-209`）。GPU 侧的形状是 `(max_num_logits, cdiv(vocab_size, 32))`、dtype `int32`（`vllm/v1/worker/gpu/structured_outputs.py:26-28`）——**每行字节数恰好是 vocab/8**。
3. **CPU 填 mask**。每个请求、每个待采样位置调一次 `grammar.fill_bitmask(...)`，xgrammar 的 matcher 走一遍词表，把允许的 token 对应位置 1（`__init__.py:231-252`，`backend_xgrammar.py:313-314`）。不约束的行整行填 `-1`（全 1）。请求数 > 128 且无投机时走线程池并行填（`__init__.py:81-88`、`:422-454`）。
4. **送到 GPU 并应用**。numpy 数组经 pinned memory 异步 H2D，然后一个 Triton kernel 把"bit=0"的位置写成 `-inf`：`bitmask = ((packed >> arange(32)) & 1) == 0`，`tl.store(..., -inf, mask=position_is_active & bitmask)`（`vllm/v1/worker/gpu/structured_outputs.py:96-134`）。调用点在 model forward 之后、`self._sample(...)` 之前（`vllm/v1/worker/gpu_model_runner.py:4289-4293`）——也就是说它作用在已经出了 CUDA Graph 的 logits 上，不会破坏图捕获。

**jump-forward 是什么**：语法在某些位置会退化成"后面这段已经唯一确定"。比如 schema 里一个固定字段名，只要走到 `{"name": "`，接下来的字符序列就是常量。xgrammar 提供 `find_jump_forward_string` 直接给出这段字符串，框架就可以一次性把整段 token 追加进输出、跳过对应的 N 次前向——省的是"每个 token 一次 forward"的钱，相当于把结构化输出里确定的部分从 decode 降级成 prefill。

**但在当前 main 上，V1 没有接这条路径**。全仓库只有几处相关痕迹：`backend_xgrammar.py:253-254` 的一行参考链接、`backend_guidance.py:250` 的 `TODO - Add jump decoding support in the future`、以及 scheduler 里"未来支持 jump decoding"的注释（`vllm/v1/core/sched/scheduler.py:709`）。读代码时看到 `find_jump_forward_string` 的链接，不要以为已经生效。

## 6. 种子、随机数流与 batch 依赖

请求的采样类型由 `SamplingType` 决定（`sampling_params.py:871-876`）：温度 < 1e-5 → `GREEDY`；设了 `seed` → `RANDOM_SEED`；否则 `RANDOM`。只有 `RANDOM_SEED` 会在建请求时创建一个 per-request generator 并 `manual_seed(seed)`（`gpu_model_runner.py:1261-1266`），随请求状态一路带到采样器。

采样器这边，`generators` 是 `dict[req_index, torch.Generator]`。`random_sample` 的策略很务实：

```python
# vllm/v1/sample/ops/topk_topp_sampler.py:619-631
q = empty_exponential_noise_like(probs, use_fp64_gumbel)
if len(generators) != probs.shape[0]:
    q.exponential_()                      # 整块共享 RNG 填一遍
for i, generator in generators.items():
    q[i].exponential_(generator=generator)  # 有 seed 的行覆写
```

这解释了三个"为什么同一个 prompt 结果不同"：

1. **共享 RNG 流是按形状切的**。`q.exponential_()` 一次抽 `batch × vocab` 个数。batch 从 256 变成 255，第 7 行拿到的是流里另一段偏移的数据。只有带 `seed` 的行会被覆写，其余行完全受 batch 组成影响。
2. **kernel 选择会变**。FlashInfer / aiter / XPU 内核都不支持 per-request generator，检测到 `generators` 非空就直接回退 native（`topk_topp_sampler.py:329-336`、`:422-428`、`:494-500`）。所以"设 seed"和"不设 seed"走的是两条数值路径，不只是两串随机数。
3. **浮点归约顺序变了**。不同 batch 形状下 lm_head 的 GEMM、softmax 的归约树都不同，logits 末位有差异；当两个候选 token 的 logit 差在 1e-6 量级时，argmax 会翻转。`VLLM_BATCH_INVARIANT`（`vllm/envs.py:95`）就是为这个场景准备的，打开后 vLLM 会在"温度 > eps 且没设 seed"时主动警告（`sampling_params.py:610-616`）。

结论：**想要可复现，必须同时固定 seed、batch 组成和 kernel 路径**；只给 seed 只能保证"同样的负载序列下重放一致"，不保证"单独跑和混在批里跑一致"。官方文档的 [Reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/) 页面覆盖的是种子这一层，batch 组成和 kernel 路径要自己保证。

## 7. 账本

假设：词表 $V = 131072$（128k），batch $B = 256$，fp32 张量，H100 SXM（HBM ~3.3 TB/s，PCIe 4.0 x16 实测 20–25 GB/s）。

| 项目 | 每步数据量 | 每步时间量级 | 说明 |
|---|---|---|---|
| 纯采样（fp32 softmax + top-k/top-p） | 读写各 128 MiB | ~0.1 ms | HBM 带宽内，很便宜 |
| `logprobs=20` | 256 × 21 × (4+4) B ≈ 42 KiB | ~0 | 排序 + rank，可忽略 |
| `logprobs=-1` 全量 | 256 × 131072 × 4 B = **128 MiB** | D2H 约 5–7 ms | 一步就把 decode 时延翻倍 |
| 结构化 bitmask | 每行 16 KiB，256 行 = **4 MiB** | H2D ~0.2 ms | 拷贝不是瓶颈 |
| 结构化 CPU 填充 | 128k × 256 ≈ **3350 万次** token 判定 | 1–10 ms（估算，随语法复杂度浮动） | 真正的瓶颈 |
| penalties 的 bin count | 2 × 256 × 131073 × 8 B = **512 MiB** | 分配 + `scatter_add`，亚毫秒到毫秒级 | 开任何惩罚都要付 |

几处值得单独拎出来：

- **全量 logprobs 是带宽事故，不是算力事故**。`LogprobsTensors` 里 `logprobs` 就是那个 `(batch, vocab)` 张量（`vllm/v1/outputs.py:89-95`），输出阶段由 `to_cpu_nonblocking` 把它搬回 host（`vllm/v1/outputs.py:116-128`，调用点 `vllm/v1/worker/gpu/async_utils.py:145`）。128 MiB / 22 GB/s ≈ 6 ms，而 256 并发下这一步的模型 forward 本身可能只要 10–15 ms。生成 1000 个 token，就是 128 GiB 的 PCIe 流量、约 6 秒纯拷贝。
- **penalties 的计数张量是 int64 且按整词表分配**：`torch.zeros((num_seqs, vocab_size + 1), dtype=torch.long)`，prompt 一次、output 一次（`vllm/model_executor/layers/utils.py:26-40`）。源码自己都承认"currently quite inefficient, will be reworked anyhow"（`vllm/v1/sample/ops/penalties.py:25-26`）。只开 `repetition_penalty` 一个参数，也要为整个 batch 付这份显存和带宽。
- **bitmask 的字节数就是 vocab/8**：128k 词表 = 16 KiB/行，只随词表线性涨。投机解码时每个请求要占 `1 + num_spec_tokens` 行（`__init__.py:399-412`），所以开 spec decode 会让 bitmask 行数成倍增加。
- **`PROCESSED_LOGPROBS_MODES` 会改 kernel**：只要配置成 `processed_logits` / `processed_logprobs`，FlashInfer 采样就被禁用（`topk_topp_sampler.py:245-254`），因为 FlashInfer 不暴露 top-k 之后的值。这是"logprobs 拖慢一切"最直接的机制。
- **ROCm 上的融合采样 kernel 有一条长前置条件**：`mixed_sample_eligible` 要求全随机、无 top-k/top-p、无 logprobs、无惩罚、无 allowed ids、无 grammar、无 thinking budget（`sampler.py:23-50`）。任意一个特性打开，就退回通用路径。

## 8. 常见坑

1. **温度 0 会静默清空其他采样参数**。`temperature < 1e-5` 时，`top_p` 被强制成 1.0、`top_k` 成 0、`min_p` 成 0.0（`sampling_params.py:619-624`）。你传的 `top_p=0.9` 在 greedy 请求里根本不存在，实际行为就是纯 argmax；如果下游还要回显这些参数，注意它们已经被改写过。
2. **`logprobs=0` 和 `logprobs=None` 不是一回事**。前者只要采样 token 自己（`vllm/v1/sample/metadata.py:25`），后者完全不计算；很多客户端把 0 当"关掉"。
3. **V1 的 logprobs 不含惩罚和温度**。用 V0 时代的方式（拿返回值还原采样分布）做 RL 或蒸馏，会得到系统性偏差；需要一致就显式配 `logprobs_mode=processed_logprobs`（`vllm/config/model.py:298-305`），并接受 FlashInfer 被禁用。
4. **结构化输出后端不能按请求混用**。同一 engine 里 xgrammar 和 guidance 同时出现会抛 `VLLMValidationError`（`__init__.py:144-151`）。多租户场景要么统一 `--structured-outputs-config.backend`，要么拆 engine。
5. **首请求延迟藏在语法编译里**。编译走线程池 + `Future`（`__init__.py:196-205`），所以不会阻塞 engine 主循环，但第一个复杂 JSON schema 请求仍要多等一个编译往返；重复 schema 靠 512 MiB 缓存兜底（`envs.py:1757`），冷启动时缓存是空的。
6. **`bad_words` 是 Python 循环**。逐请求、逐坏词、逐前缀比对（`vllm/v1/sample/ops/bad_words.py:35-36`），batch 256、每请求 50 个坏词时，每步就是上万次 Python 层判断，且都在关键路径上。
7. **别指望"同 prompt 同输出"跨负载成立**。见 §6：共享 RNG 流按形状切、kernel 按特性切、浮点归约按 batch 切，三重因素叠加。

## 小结

- 一步采样是七段流水线：raw logprobs → fp32 → allowed ids → bad words → 非 argmax 不变处理器 → 惩罚 → (greedy 分叉) → temperature → min_p → top-k/top-p → 采样（`sampler.py:80-115`）。顺序改变结果，尤其是"惩罚先于温度""mask 先于 min_p"这两条。
- V1 的 logprobs 默认是**原始** log_softmax，不含惩罚和温度（`sampler.py:142-145`）；这与 V0 不同，做 RL 时要么显式切 `processed_logprobs`，要么接受偏差。
- top-p 的实现是升序排序 + 原地 cumsum + `probs_sum <= 1 - p` 掩码（`topk_topp_sampler.py:548-565`），比"降序累加到 p"少一次翻转和一份显存；`top_p_mask[:, -1] = False` 保证至少留一个 token。
- 采样本身很便宜（128 MiB 张量走一遍 HBM，百微秒级）；真正贵的是 penalties 的 int64 bin count（batch 256 时 512 MiB/步）和全量 logprobs（128 MiB/步的 PCIe 流量，约 6 ms）。
- 结构化输出的成本在 CPU 填 bitmask（128k 词表 × 256 请求 ≈ 3350 万次判定/步），不在 GPU 应用（4 MiB H2D，约 0.2 ms）；bitmask 字节数恒等于 vocab/8。
- jump-forward 在当前 main 上还没接（`backend_xgrammar.py:253-254` 只有注释，`backend_guidance.py:250` 是 TODO），别把 xgrammar 的能力当成 vLLM 已实现的能力。
- 可复现 = seed + 固定 batch 组成 + 固定 kernel 路径；三者缺一，同一个 prompt 在不同负载下就会给出不同 token。

下一篇进入投机解码：[n-gram、EAGLE 与拒绝采样]({{< relref "vllm-09-spec-decode" >}})——那里会看到本篇的采样器被复用成"验证器"，以及 `num_speculative_tokens` 如何让 bitmask 行数翻倍。

参考文档：

- [Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [Logits Processors](https://docs.vllm.ai/en/latest/design/logits_processors/)
