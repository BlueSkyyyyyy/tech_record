---
title: "Megatron 源码精读（一）：整体代码结构与启动链路"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架"]
categories: ["训练框架"]
weight: 1
series: ["megatron-code"]
---

这是「Megatron-LM 源码精读」系列的第一篇。全系列以官方仓库 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 为对象，基准 commit `f713506ce`（2026-08-31）。

本篇只回答一个问题：**一条 `torchrun pretrain_gpt.py ...` 命令，是怎么一步步落到 `GPTModel.forward` 的？**

**一句话结论**：入口文件只做装配；训练主循环在 `megatron/training/`；模型与并行实现在 `megatron/core/`。三者靠四样东西串起来——配置容器、`TransformerConfig`、`ModuleSpec`、`MegatronModule` 继承链。

---

## 1. 仓库只有两层：薄入口 + 厚内核

仓库根目录直接摊着训练入口，它们都很薄（`pretrain_gpt.py` 586 行，大半还是 dataset 和 `forward_step`）。

| 入口 / 装配文件 | 作用 |
|---|---|
| `pretrain_gpt.py` | GPT 预训练 / SFT 主入口 |
| `pretrain_hybrid.py` / `pretrain_mamba.py` | Hybrid（Mamba+Attention）/ 纯 Mamba |
| `pretrain_vlm.py` | 多模态（LLaVA 等） |
| `train_rl.py` | 强化学习 |
| `model_provider.py` | 通用模型构建器（本身不含具体模型） |
| `gpt_builders.py` / `hybrid_builders.py` / `mamba_builders.py` | 各模型的 layer spec 与服务装配 |

真正的逻辑在 `megatron/` 下，分成两块：

- **`megatron/core/`**：与「训练这件事」无关的内核——模型、并行、通信、优化器、checkpoint、数据格式。这是本系列的主战场。
- **`megatron/training/`**：训练工程——参数解析、初始化、主循环、日志、断点续训、异常重跑。

一句话分工：**`core/` 提供可组装的零件，`training/` 决定这次训练怎么用这些零件。**

## 2. 启动链路：四跳

### 2.1 第一跳：入口只做「解析 + 调用」

`pretrain_gpt.py` 的 `__main__` 只有三步（`pretrain_gpt.py:560-586`）：

```python
args = parse_and_validate_args(...)
full_config = pretrain_cfg_container_from_args(args)
pretrain(
    full_config,
    train_valid_test_datasets_provider,
    ModelType.encoder_or_decoder,
    forward_step,
    model_provider=partial(model_provider, gpt_builder),
    ...
)
```

这里有两个容易看漏的变化：

1. 参数解析的产物不再是裸的 `args`，而是 **`PretrainConfigContainer`**（`megatron/training/config/container.py:259`）——把训练、模型、优化器、checkpoint、分布式等配置分组装进一个 dataclass，支持 YAML 序列化。`pretrain()` 的第一个参数就是它。
2. `model_provider` 被 `partial` 绑定了一个 builder。**模型选择不再是 `model_provider` 内部的 if/else，而是调用方把 builder 当参数传进来。**

### 2.2 第二跳：`pretrain()` 是总导演

`megatron/training/training.py:1499` 的 `pretrain()` 按固定顺序做五件事：

1. `initialize_megatron()`：建进程组、算并行拓扑、设随机种子、算 microbatch 数；
2. 调 `model_provider(...)` 建模型（PP 的每个 stage 各建一段）；
3. 用 DDP / Megatron-FSDP / Torch FSDP2 包一层（第 8 篇）；
4. 建优化器、学习率调度器、数据加载器；
5. 进入 `train()` 的 step 循环：forward → backward → optimizer step。

所以读源码时，**`pretrain()` 是「目录」**：想知道某个机制在什么时候生效，先看它在这五步里的哪一步。

### 2.3 第三跳：`model_provider` 是柯里化的转接头

`model_provider.py:21` 的签名是：

```python
def model_provider(model_builder, pre_process=True, post_process=True,
                   vp_stage=None, config=None, pg_collection=None):
```

它做三件杂事，然后把参数原样转交给 builder（`model_provider.py:65`）：

- 按需挂 OOM observer（`record_memory_history`）；
- 若启用 ModelOpt，换成量化版 builder；
- 透传 `pre_process` / `post_process`（PP 首尾 stage 标记）、`vp_stage`（虚拟流水线 stage 号）、`pg_collection`（进程组集合）。

**「路由」发生在入口文件里**：GPT 传 `gpt_builder`，Hybrid 传 `hybrid_builder`。这样 `model_provider.py` 不需要认识任何具体模型。

### 2.4 第四跳：`gpt_builder` 定 config、定 spec、拼模型

`gpt_builders.py:25` 的 `gpt_builder` 是 GPT 装配的核心。它的结构是一棵优先级从高到低的分支树：

```python
config = core_transformer_config_from_yaml(...)  # 或 from_args
if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)      # 用户自定义 spec
elif args.experimental_attention_variant is not None:
    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(...)
elif args.num_experts:
    transformer_layer_spec = get_gpt_decoder_block_spec(...)   # MoE
elif args.heterogeneous_layers_config_path is not None:
    transformer_layer_spec = get_gpt_heterogeneous_layer_spec(...)
else:
    transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
```

最后 `GPTModel(config, transformer_layer_spec, ...)` 实例化。**「用 TE 还是纯 torch」「MoE 还是 dense」「哪种 attention variant」全部在这一层收敛成一张 spec**，`GPTModel` 本身不需要知道。

## 3. 四个必须记住的抽象

| 抽象 | 一句话 | 详见 |
|---|---|---|
| `PretrainConfigContainer` | 一次训练的全部配置，按域分组 | 本篇 |
| `TransformerConfig` | 模型结构 + 并行 + 精度超参的单一来源 | 第 6、17 篇 |
| `ModuleSpec` | 「用哪个类 + 传什么参数 + 含哪些子模块」的数据化描述 | 第 17 篇 |
| `MegatronModule → LanguageModule → GPTModel` | 模型继承链：配置、初始化、embedding/output 通用逻辑 | 本篇 |

继承链本身很浅，但决定了代码放哪：

```
MegatronModule        # megatron/core/transformer/module.py：config、初始化、量化 hook
  └─ LanguageModule   # common/language_module：embedding / output / 权重共享
       └─ GPTModel    # models/gpt/gpt_model.py：decoder-only、causal mask、MTP
```

## 4. 目录地图：只列会反复用到的

`megatron/core/` 顶层是横向切面（进程组、重计算、量化、packed 序列），子目录是纵向模块：

| 路径 | 职责 | 详见 |
|---|---|---|
| `parallel_state.py` / `process_groups_config.py` | 进程组拓扑与 `ProcessGroupCollection` | 第 3、19 篇 |
| `transformer/` | TransformerBlock / Layer / MLP / attention / `moe/` | 第 14、17 篇 |
| `tensor_parallel/` | `ColumnParallelLinear` / `RowParallelLinear` / 通信原语 | 第 21 篇 |
| `pipeline_parallel/` | `schedules.py`（1F1B）、`p2p_communication.py` | 第 18、20 篇 |
| `distributed/` | DDP、`param_and_grad_buffer.py`、Torch FSDP2、`fsdp/` | 第 8、20 篇 |
| `optimizer/` | `distrib_optimizer.py`、`muon.py`、`cpu_offloading/` | 第 7、11 篇 |
| `dist_checkpointing/` | 分布式 checkpoint 底层 | 第 22 篇 |
| `datasets/` | `.bin/.idx` 格式、三级索引、数据混合 | 第 9 篇 |
| `models/` | `gpt/`、`bert/`、`T5/`、`mamba/`、`hybrid/`、`multimodal/`、`vision/`、`audio/`、`mimo/`、`bagel/` | 第 15、23 篇 |
| `fusions/` | 手写 CUDA / 融合算子 | 第 12 篇 |
| `ssm/` | Mamba、GDN/KDA 等状态空间模型 | 第 23 篇 |
| `context_parallel_layout/` | CP 的 varlen 布局（zigzag / contiguous） | 第 16 篇 |
| `inference/` | 推理引擎（含分离式推理） | — |
| `extensions/` | 第三方接入（`transformer_engine.py` 等） | 第 6 篇 |

> 老文里常见的 `resent/` 是笔误，仓库里没有这个目录；对应的是 `ssm/`。

### 4.1 `transformer/` 与 `models/gpt/` 的分工

这是最容易混的一处：

- **`transformer/` 造积木**：`TransformerBlock`、`TransformerLayer`、`MLP`、`CoreAttention`，与具体模型无关。
- **`models/gpt/gpt_layer_specs.py` 画图纸**：用 `ModuleSpec` 把这些积木按 GPT 的要求（causal mask、normalization、MLP 类型）组装成 decoder。
- **`gpt_builders.py` 按图纸施工**，再交给 `GPTModel`；`GPTModel` 依据 `pre_process` / `post_process` 决定自己带 embedding、若干 block、还是 output。

同一套内核能同时撑 GPT / T5 / BERT / Mamba / 多模态，靠的就是「spec 描述 + config 驱动」这一层。

## 5. 怎么读这个系列

25 篇按主题分三条线，不必从头读到尾：

- **并行线（建议先读）**：02 原理 → 03 拓扑 → 19 组装地图 → 21 TP 通信 → 18 PP 调度 → 20 通算重叠 → 16 CP。
- **显存线**：04 重计算 → 07 CPU offload → 08 ZeRO/FSDP → 11 优化器 → 24 显存估算。
- **系统线**：05 随机种子 → 06 TE → 09 数据 → 10/22 checkpoint → 12 融合算子 → 13 RL → 14 MoE → 15 多模态 → 17 架构 → 23 Mamba → 25 时间估算。

完整目录：

1. [整体代码结构与启动链路]({{< relref "megatron-code-01-structure" >}})
2. [模型并行的原理：TP/SP/PP/CP/DP/FSDP]({{< relref "megatron-code-02-parallel-principles" >}})
3. [并行拓扑：parallel_state.py 精读]({{< relref "megatron-code-03-parallel-topology" >}})
4. [激活重计算：recompute.py]({{< relref "megatron-code-04-recompute" >}})
5. [随机种子的设置]({{< relref "megatron-code-05-rng-seeds" >}})
6. [与 Transformer Engine 的关系]({{< relref "megatron-code-06-transformer-engine" >}})
7. [CPU offload]({{< relref "megatron-code-07-cpu-offload" >}})
8. [ZeRO-1 / FSDP 实现]({{< relref "megatron-code-08-zero-fsdp" >}})
9. [数据集处理]({{< relref "megatron-code-09-dataset" >}})
10. [checkpoint 处理]({{< relref "megatron-code-10-checkpoint" >}})
11. [优化器]({{< relref "megatron-code-11-optimizer" >}})
12. [fused 算子]({{< relref "megatron-code-12-fused-kernels" >}})
13. [强化学习]({{< relref "megatron-code-13-rl" >}})
14. [MoE 实现与优化]({{< relref "megatron-code-14-moe" >}})
15. [多模态（LLaVA）实现]({{< relref "megatron-code-15-multimodal" >}})
16. [Context Parallel 细节]({{< relref "megatron-code-16-context-parallel" >}})
17. [MCore 架构与 layer spec 机制]({{< relref "megatron-code-17-mcore-arch" >}})
18. [Pipeline 调度细节]({{< relref "megatron-code-18-pipeline-schedule" >}})
19. [并行组装地图]({{< relref "megatron-code-19-parallel-assembly" >}})
20. [通信与计算 overlap]({{< relref "megatron-code-20-overlap" >}})
21. [TP 通信原语与并行线性层]({{< relref "megatron-code-21-tp-communication" >}})
22. [分布式 checkpoint 底层]({{< relref "megatron-code-22-dist-checkpointing" >}})
23. [Mamba/SSM 与 Hybrid 模型]({{< relref "megatron-code-23-mamba-ssm-hybrid" >}})
24. [显存到底怎么算]({{< relref "megatron-code-24-memory-estimation" >}})
25. [一个 training step 的耗时怎么估算]({{< relref "megatron-code-25-step-time-estimation" >}})
26. [2026-09 → 2026-10 上游更新速览]({{< relref "megatron-code-26-updates" >}})

## 6. 小结

- 入口薄、内核厚：`pretrain_*.py` 只解析参数并调用 `pretrain()`；主循环在 `training.py`；模型与并行在 `core/`。
- 装配靠「builder + spec」：`model_provider` 不认识具体模型，`gpt_builder` 把 config 和 spec 定下来，`GPTModel` 只管按 spec 搭。
- 配置已经对象化：`args` 先变成 `PretrainConfigContainer`，再从中取 `TransformerConfig`；不要再用「全局 args 满天飞」的旧印象读新代码。
- `transformer/` 是积木，`models/gpt/` 是图纸，`gpt_builders.py` 是施工队。

---

**版本提示（2026-10）**：2026-09 之后上游新增了 `allocator/`、`context_parallel/`、`tuning/`、`inference/disaggregation/` 等子包，`transformer_layer.py` 大幅重构（新增 mHC 等结构）；但「薄入口 + builder/spec 装配 + config 容器」的主干没有变。细节见第 26 篇。

**行号基线**：commit `f713506ce`（2026-08-31）。上游 dev 已到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
