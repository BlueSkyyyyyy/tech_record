---
title: "Megatron 源码精读（二十一）：TP 通信原语与并行线性层"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "张量并行", "tensor-parallel", "collective", "线性层"]
categories: ["训练框架"]
weight: 21
series: ["megatron-code"]
---

上一篇[《通信与计算 overlap》]({{< relref "megatron-code-20-overlap" >}})讲了「怎么把通信藏进计算」。但通信到底长什么样、由谁发起，还分散在书缝里。第 3 篇[《并行拓扑》]({{< relref "megatron-code-03-parallel-topology" >}})只讲了 `parallel_state.py` 里的**拓扑如何组织**（rank、group、坐标系），没有展开张量并行里**真正干活的那批通信原语**。

这一篇补上 TP 的「肉」——张量并行（TP）的实际执行层，三块核心素材：

- `megatron/core/tensor_parallel/mappings.py`（714 行）：**集合通信原语**，封装 `copy/scatter/gather/reduce/all-to-all` 成带 autograd 的 `torch.autograd.Function`。
- `megatron/core/tensor_parallel/layers.py`（1408 行）：**并行线性层与词表嵌入**，`ColumnParallelLinear` / `RowParallelLinear` / `VocabParallelEmbedding`。
- `megatron/core/tensor_parallel/cross_entropy.py`（235 行）：**词表并行的交叉熵**。

行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`。

---

## 1. 问题背景：TP 的一拆一合，靠原语顶起

张量并行的数学本质就一句话：**把一个大矩阵乘法拆成若干小块并行算，算完再拼起来**。第 2 篇讲过原理，但落地到代码靠的是两类东西反复组合：

1. **切（split/scatter）**：把 `[.., H]` 的重量/激活切成 `[.., H/tp]`，每 rank 留一份。
2. **合（gather/reduce）**：把各 rank 的局部结果 all-gather 拼回全量，或 all-reduce 累加回全局。

这些动作如果每次都在计算图里「裸调 `torch.distributed`」，反向传播时梯度不会自动跟着 broadcast/accumulate。所以 Megatron 把它们都包装成 **`torch.autograd.Function`**：forward 做集合通信，backward 做对应的**对偶通信**（如 forward 是 gather，backward 就是其逆 reduce-scatter）。

---

## 2. 原语层：`mappings.py` 的六类操作

### 2.1 底层 `_reduce`：all-reduce 的薄封装

最基础的是 `_reduce`（`mappings.py:22-37`）。它只做两件事：`world_size==1` 时直接返回原张量（**旁路优化**，单卡 TP 即无 TP）；否则 `contiguous()` 后 `all_reduce`。注释（`mappings.py:31-33`）特别强调：输入若已 contiguous 会被**原地改写**，否则返回新张量——调用方必须用返回值承接，否则拿到的是未 reduce 的结果。

### 2.2 四个维度 helper：切与合

把张量切/合，按「沿哪一维」分成四组，全部以 `world_size==1` 旁路开头：

| helper | 维度 | 动作 |
|---|---|---|
| `_split_along_last_dim` | 最后一维 | split 后取本 rank 那份（`:40`） |
| `_split_along_first_dim` | 第一维 | `dim_size % world_size` 断言 + 切片（`:60`，`_split_along_first_dim` 在 `:72-74` 断言可整除） |
| `_gather_along_last_dim` | 最后一维 | all-gather 后按 chunk 沿最后维 `cat`（`:84`） |
| `_gather_along_first_dim` | 第一维 | all-gather 拼接，支持 `output_split_sizes` 泛化（`:118`） |
| `_reduce_scatter_along_first_dim` | 第一维 | reduce-scatter，支持 `input_split_sizes`（`:159`） |

> 一个细节：`_gather_along_last_dim` 和 `_reduce_scatter_along_last_dim` 的实现**先转成第一维问题再复用**——`_reduce_scatter_along_last_dim`（`:103-115`）把最后维 `reshape(-1, H)` 后 `split` 成 chunk、`cat` 到第一维、走 `_reduce_scatter_along_first_dim`，最后 `reshape` 回原 shape。这样「沿最后维」的复杂 gather/reduce-scatter 不用重写，统一落到 NCCL 擅长的 first-dim 路径上。

### 2.3 autograd 化的集合通信原语

真正的对外接口是下面这些 `torch.autograd.Function`，每个的 forward/backward 恰好互逆：

- `_CopyToModelParallelRegion`（`mappings.py:201`）：forward **恒等**（copy），backward `_reduce`。用于「这块先各 rank 各算各的，反向再把梯度 all-reduce 回来」。
- `_ReduceFromModelParallelRegion`（`:221`）：forward `_reduce`，backward 恒等。与上面相反。
- `_ScatterToModelParallelRegion`（`:240`）：forward 沿最后维 split，backward 沿最后维 gather。
- `_GatherFromModelParallelRegion`（`:260`）：forward 沿最后维 gather，backward 沿最后维 split。
- `_ScatterToSequenceParallelRegion`（`:280`）：forward 沿**第一维** split，backward 沿第一维 gather —— 序列并行的切分就沿 seq 维度。
- `_GatherFromSequenceParallelRegion`（`:300`）：forward 沿第一维 gather，backward 分两种：`tensor_parallel_output_grad=True` 走 reduce-scatter（下游还是 TP 会各自消费一份），`False` 走 split（下游复制计算，各 rank 只需自己那份梯度，`:340-352`）。
- `_AllToAll`（`:469`）：forward `all_to_all_single`，backward 把 `output/input_split_sizes` 对调再 all-to-all 一次（`:519-525`）——all-to-all 的对偶就是「反向分片交换」。

对外 wrapper 大多只是套一层 `get_tensor_model_parallel_group_if_none` 取默认 group 再 `apply`（如 `copy_to_tensor_model_parallel_region` `:537`、`gather_from_sequence_parallel_region` `:567`）。

### 2.4 序列并行的双向 all-to-all

`all_to_all_sp2hp`（`:659`）和 `all_to_all_hp2sp`（`:688`）是序列并行专用的重排技巧，把张量在「sequence-parallel 布局」和「hidden-parallel 布局」之间切换：

- `sp2hp`（seq → hidden）：输入 `[num_tokens/tp, H]`，先按最后维切 `tp` 块转置拼到第一维，再 all-to-all，输出 `[num_tokens, H/tp]`（`:679-684`）。
- `hp2sp` 是其逆：`[num_tokens, H/tp]` all-to-all 后 reshape、按第一维切块重拼，输出 `[num_tokens/tp, H]`（`:707-713`）。

两次 all-to-all 实现「跨 rank 交换 hidden 维度切片」——这正是后来 `context_parallel_layout`（第 16 篇）那类重排的雏形思路。

### 2.5 异步原语：为 overlap 预留的钩子

`mappings.py` 末尾新增了两个异步原语，直接服务于第 20 篇的 overlap：

- `async_gather_from_sequence_parallel_region`（`:581`）：用 `_GatherFromSequenceParallelRegionAsync`（`:373`）发起 `async_op=True` 的 all-gather，返回 `_AsyncCollectiveHandle`（`:355`），其 `wait()`（`:364`）在**首个消费者**处才真正等。
- `async_reduce_scatter_along_first_dim`（`:602`）：同理异步 reduce-scatter。

`_AsyncCollectiveHandle` 里 `_input_buffer` 字段（`:362`）专门留着——NCCL 可能还没消费完输入，必须让输入张量存活到 `wait`，否则就是经典的「通信未完成、输入被释放」崩溃。

---

## 3. 初始化：TP 属性与切分初始化

在进线性层之前，先看两个被反复调用的初始化细节。

### 3.1 TP 属性标记

`set_tensor_model_parallel_attributes`（`layers.py:110`）给张量打四个标记：`tensor_model_parallel`、`is_qkv`、`partition_dim`、`partition_stride`，默认值在 `_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS`（`:60`）。其中 `partition_stride` 用于「交错切分」（stride）场景——优化器/checkpoint 要用它判断参数到底怎么切。`param_is_not_tensor_parallel_duplicate`（`:92`）用这个标记 + `allreduce` 属性判断「某参数是否本 rank 的**非重复**分片」，这是分布式 checkpoint 去重的依据（第 10 篇的坑）。

### 3.2 切分初始化

`_initialize_affine_weight_cpu`（`:158`）体现了 TP 参数初始化的正确姿势：**先在每个 rank 上建完整 master 权重 → 统一 `init_method` 初始化 → 再沿 `partition_dim` split**，各 rank 取自己的 slice（`weight_list[rank::world_size]`，`:193`）。这样保证「跨 rank 初始化确定性」——否则各 rank 各自随机初始化，分片之间就拼不出同一份权重。

GPU 侧 `_initialize_affine_weight_gpu`（`:143`）用 `get_cuda_rng_tracker().fork()`（`:151`）切分 RNG 流来保证初始化一致性，MoE expert 则 fork 到 `get_expert_parallel_rng_tracker_name()`（`:154`）——对应第 5 篇的随机种子机制。

---

## 4. `ColumnParallelLinear`：按列切权重

### 4.1 数学与派生

`Y = XA + b` 中，把 `A` 沿**列**切成 `[A_1, ..., A_p]`（`layers.py:787-789`）。每个 rank 持有 `A_i`（shape `[output_size/tp, input_size]`），自己算 `X A_i^T`，得到 `Y` 的某一段列。`output_size_per_partition = divide(output_size, world_size)`（`:876`）。

### 4.2 forward 的切与合

核心在 `forward`（`:1000`）：

1. **切输入**：只有当 `allreduce_dgrad`/`sequence_parallel`/expert 通信都关闭时，才 `copy_to_tensor_model_parallel_region`（`:1047`）——即输入需要广播到每个 rank（`X` 是重复的完整输入）。其余情形 `X` 本身已是各 rank 该有的那份，直接用 `input_parallel = input_`（`:1045`）。
2. **矩阵乘**：走 `linear_with_grad_accumulation_and_async_allreduce`（`:672`），这把「前向 all-gather + 反向异步 all-reduce/reduce-scatter」全部内联（见第 6 节）。
3. **合输出**：`gather_output=True` 时沿最后维 `gather_from_tensor_model_parallel_region` 拼回完整 `Y`（`:1101`）；否则各 rank 保留自己的 `Y_i = X A_i`（`:1105`），交给下游 `RowParallelLinear` 消费。

### 4.3 关键开关

- `skip_bias_add`（`:807`）：不加 bias 而是把 bias 随返回值带出去，让调用方和后续 elementwise 融合（`output_bias`，`:1106`）。
- `skip_weight_param_allocation`（`weight=None` 在外部传入，`:1021-1027`）：LoRA 等靠外部喂权重。
- `allreduce_dgrad`（`:966`）= `world_size>1 and not sequence_parallel and not disable_grad_reduce`：非 SP 时反向要对输入梯度 all-reduce。
- 权重/偏置打上 `allreduce` 属性（`:928`）——expert 并行时不 allreduce（第 14 篇 MoE 的 EP 语义）。

---

## 5. `RowParallelLinear`：按行切权重

`A` 沿**行**切 `A = transpose([A_1 .. A_p])`，同时 `X` 沿第二维切 `X = [X_1, ..., X_p]`（`layers.py:1151-1152`）。每个 rank 持有 `A_i`（`[output_size, input_size/tp]`）和 `X_i`，算 `Y_i = X_i A_i^T`，此时的 `Y_i` 是**部分和**。

forward（`:1314`）：

1. **切输入**：`input_is_parallel=False` 时 `scatter_to_tensor_model_parallel_region`（`:1330`）把 `X` 按最后维切给各 rank；`True` 时输入已切好直接用。
2. **矩阵乘**：算 `Y_i`。
3. **合输出**：三选一（`:1354-1363`）——expert 通信时原样返回；sequence parallel 时 `reduce_scatter_to_sequence_parallel_region`（把部分和 reduce-scatter 成 seq 切分发到下游）；否则 `reduce_from_tensor_model_parallel_region` 全量 all-reduce 还原 `Y`。

`RowParallelLinear` 与 `ColumnParallelLinear` 常成对出现：前者的 all-reduce（或 reduce-scatter）恰好「抵消」后者的列切分。

---

## 6. 异步 grad 累加融合：`LinearWithGradAccumulationAndAsyncCommunication`

这是 `layers.py` 里性能最关键的一段（`layers.py:470-669`），`ColumnParallelLinear.forward` 最终落到这里。

### 6.1 forward：SP 下的懒 all-gather

forward（`:475`）在 `sequence_parallel=True` 时，不从下游拿全量输入，而是 `get_global_memory_buffer().get_tensor(..., "mpu")`（`:509`）拿一块全局通信 buffer，把输入 all-gather 进去（`:510`）再 `matmul`。全局 buffer 复用避免每次前向都 `torch.empty` 分配显存（第 19 篇 `get_global_memory_buffer` 的用途）。

### 6.2 backward：三段异步重叠

backward（`:522`）是三段式的：

1. **输入激活 all-gather（异步）**：SP 下先 `async_op=True` 重 gather 输入（`:549`），拿到 `total_input`，然后**先算 input 梯度** `grad_input = grad_output.matmul(weight)`（`:558`），这中间 gather 在飞。
2. **input 梯度 all-reduce（异步）**：非 SP 且 `allreduce_dgrad` 时 `all_reduce(grad_input, async_op=True)`（`:571`），接着算 weight 梯度。
3. **SP 的 reduce-scatter（异步）**：SP 时把 input 梯度 reduce-scatter 回 seq 切分（`:582`）。

最后统一 `handle.wait()`（`:562` / `:661` / `:667`）。全程贯穿一句话注释：**依赖 `CUDA_DEVICE_MAX_CONNECTIONS=1`**（`:553-554`、`:572-573`、`:585-586`）保证按调用顺序调度 kernel——这正是第 20 篇 overlap 思想的「单卡单流基线」。

### 6.3 `gradient_accumulation_fusion`

若开启，weight 梯度不单独算再累加，而是直接 `wgrad_gemm_accum_fp32/fp16`（`:612-618`）把 `grad_output^T @ total_input` 累加进 `weight.main_grad`，省掉一次加法 kernel；FSDP 下走 `te_general_gemm`（`:594`）。这需要 Apex 的 `fused_weight_gradient_mlp_cuda`（`:48-50`）。

### 6.4 wgrad deferral

`grad_output_buffer` + `wgrad_deferral_limit`（`:536-539`）能把最终 embedding 层的 wgrad **推迟**几个 microbatch，凑满再一起 GEMM——是 `defer_embedding_wgrad_compute` 的底层机制。

---

## 7. `VocabParallelEmbedding`：按词表切嵌入

词表嵌入 `torch.nn.Embedding` 的 TP 版本（`layers.py:204`）。每个 rank 只持有 `[vocab_size/tp, embedding_dim]` 的权重分片，`vocab_range_from_global_vocab_size`（`utils.py:114`）算出本 rank 负责的 `[vocab_start, vocab_end)`。

forward（`:290`）的坑在于**掩码**：输入 token id 若落在别的 rank 的词表段，本 rank 查不出 embedding，需先 `input_mask = (input_ < start) | (input_ >= end)`（`:298`）把非法 id 归零，查完再 `output_parallel[input_mask, :] = 0.0`（`:312`）掩盖——保证跨 rank 累加后结果正确。最后可选 `reduce_scatter_embeddings`（`:314`）把 `[b,s,h]` 转 `[s,b,h]` 后 reduce-scatter 成序列并行布局；`deterministic_mode` 时用 `self.weight[masked_input]` 而非 `F.embedding`（`:305-309`），因为后者的 backward 非确定。

### 7.1 词表并行交叉熵

对偶地，输出端 `VocabParallelCrossEntropy`（`cross_entropy.py:13`）在词表维度切 logits 算 loss：

1. 各 rank 算局部 `logits_max`，all-reduce MAX 得全局 max（`:130`），减掉防溢出。
2. 各 rank 只在自己词表段取 `predicted_logits`，**all-reduce SUM** 拼出真正的 target logits（`:146-148`）。
3. `sum_exp_logits` 也 all-reduce SUM 得全局 softmax 分母（`:150-152`）。
4. `loss = log(sum) - predicted`（`:74`），label smoothing 在归一化概率上做（`:159-176`）。

三个 all-reduce（max / predicted sum / partition sum）是词表并行交叉熵的固定通信开销，`_VocabParallelCrossEntropy.backward`（`:186`）里 softmax 梯度直接用 forward 存下的 `exp_logits`（已归一化）算，无需再通信。

---

## 8. 小结

- **TP 执行层 = 带 autograd 的集合通信原语（`mappings.py`）+ 并行线性层/嵌入（`layers.py`）+ 词表并行 loss（`cross_entropy.py`）**。
- **原语成对互逆**：copy/reduce、scatter/gather、split/gather、all-to-all 的 backward 是 forward 的对偶，`_GatherFromSequenceParallelRegion` 的 backward 还按 `tensor_parallel_output_grad` 二选一（`:340-352`）。
- **`ColumnParallelLinear` 按列切 + 可 gather**（`:784`），**`RowParallelLinear` 按行切 + reduce/reduce-scatter**（`:1148`），两者成对抵消。
- **异步三段 backward**（`layers.py:522-669`）把前向 gather / input-grad reduce / SP reduce-scatter 全部异步，依赖 `CUDA_DEVICE_MAX_CONNECTIONS=1`。
- **切分初始化必须「全量 init + 按 rank 取 slice」**（`layers.py:158-201`），保证跨 rank 确定性。
- **异步原语（`async_gather_from_sequence_parallel_region` 等）为第 20 篇的 overlap 预留**，`_AsyncCollectiveHandle` 的 `_input_buffer` 是防「输入被提前释放」的关键。

下一篇回到「未讲透的基础设施」另一端——**分布式 checkpoint（`dist_checkpointing/`）**：它如何把这个「TP 切碎的模型」以切分形式直接落盘、再在任意并行布局下无损恢复。

