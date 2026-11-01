---
title: "Megatron 源码精读（六）：与 Transformer Engine 的关系"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "transformer-engine", "fp8"]
categories: ["训练框架"]
weight: 6
series: ["megatron-code"]
---

Megatron Core 和 Transformer Engine（下称 TE）的边界到底在哪？哪些代码是「并行与状态」，哪些是「算子与量化」？

结论：**MCore 是装配器，TE 是运算引擎。**

切分、通信、RNG、checkpoint 归 MCore；GEMM、LayerNorm、attention、FP8/FP4 归 TE。二者的接触面几乎全在 `megatron/core/extensions/transformer_engine.py` 这一个文件里（当前 3485 行）。

上一篇[随机种子的设置]({{< relref "megatron-code-05-rng-seeds" >}})留了两个钩子：`TECudaRNGStatesTracker` 和 `te_checkpoint`。本篇一起收掉。

## 1. 一句话定位：换实现不靠 if，靠 spec provider

`TransformerConfig.transformer_impl` 取 `'local'` / `'transformer_engine'` / `'inference_optimized'`，默认就是 `'transformer_engine'`（`transformer_config.py:1436`）——默认路径下 transformer 层就是 TE 在跑。

选哪套实现不靠散落各处的 `if config.transformer_impl == ...`：`get_backend`（`models/backends.py:205`）按这个字段返回一个 `BackendSpecProvider`，TE 版是 `TESpecProvider`，本地版是 `LocalSpecProvider`。

层骨架（`transformer_layer.py` / `transformer_block.py`）一行不改，换的只是注入进来的类。

本地 provider 给的是 `tensor_parallel/layers.py` 里的 `ColumnParallelLinear`/`RowParallelLinear` 与 `transformer/dot_product_attention.py` 里的 `DotProductAttention`；两套实现接口同构，差别只在算子来自 PyTorch 还是 TE。

## 2. 软依赖：没装 TE 也要能 import

文件头部（`transformer_engine.py:71-87`）先试导入，失败时 `te = MagicMock()`、`HAVE_TE = False`；每个包装类再在自己 `__init__` 里抛明确的 `ImportError`（`TELinear` 在 `transformer_engine.py:773-777`）。

这是典型的「软依赖 + 运行时报错」：装不装 TE，`import megatron` 都能成功；真正构造 TE 层时才炸，报错信息直接告诉你 `pip install transformer-engine`。

## 3. 线性层：一个 `TELinear` 承载三种并行语义

| 类 | 基类 | 并行语义 |
|---|---|---|
| `TELinear`（`737`） | `te.pytorch.Linear` | `parallel_mode` 取 `column`/`row`/`duplicated` |
| `TEColumnParallelLinear`（`1234`） | `TELinear` | 按输出维切 |
| `TERowParallelLinear`（`1479`） | `TELinear` | 按输入维切 |
| `TELayerNormColumnParallelLinear`（`992`） | `te.pytorch.LayerNormLinear` | LN 与紧邻 linear 融成一个模块 |

关键在 `__init__` 里 `parallel_mode` 的两处分流。`duplicated`（权重在各 TP rank 复制）要求 `tp_group is None`、`tp_size=1`。

参数上还要打 `allreduce=True`（走 DP 组）、并标记 `tensor_model_parallel=False`，让 DDP 知道它不是张量并行参数（`856-880`、`920-932`）。

看代码就知道这两种情况被分开处理：

```python
if parallel_mode == "duplicated":
    assert tp_group is None          # 权重复制，不该有 tp_group
    tp_size = 1
else:
    tp_size = get_pg_size(tp_group)
explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)
if explicit_expert_comm:
    output_size = divide(output_size, tp_size)  # row 模式切的是 input_size
    ...
    te_parallel_mode = None          # 通信交给 token dispatcher
    tp_size = 1
```

**expert 不是第四种 `parallel_mode`**：它的入口是上面那个 `explicit_expert_comm` 分支，先把 `output_size`/`input_size` 除以 `tp_size`，再把 TP/EP 通信交给外层 token dispatcher（`882-892`）。把 expert 说成一种 mode 是最常见的误读。

TP 通信重叠还有个连带效应：buffer 名只认 `qkv`/`proj`/`fc1`/`fc2`，遇到别的名字它直接改 `self.config.tp_comm_overlap = False` 再打 warning（`803-813`）——关掉的是整份 config，不只这一层。

另一个小坑在返回值。`te_return_bias = skip_bias_add and bias`（`786`），只有这种情况 TE 才自己返回 bias。

其余情况 MCore 让 TE 不返回 bias，自己在 `forward` 末尾补 `None`（`964-966`）——因为 TE 在 `bias=False` 时会返回零长度张量，而 MCore 全栈约定的是 `None`。

RNG tracker 也在这里注入（`862-871`，TE ≥ 1.7.0）：expert 层用 expert-parallel tracker，`duplicated` 层用 data-parallel tracker，其余传 `None`。

另外 `get_rng_state_tracker`（`910-912`）只在 tracker 已初始化时才传，那是给 TE 做参数初始化用的。

## 4. 量化：把 autocast 做成逐层开关

`te_quant_params` 由 `finish_init` 从 `config.quant_recipe` 里按模块名取出（`894-899`）。配置是两层 dataclass：`TEQuantizationRecipe`（`99`）管单层用什么 recipe、`fp8_format`、`fp8_param`、`tp_only_amax_red`；`TEQuantizationParams`（`178`）再分成训练/评估两份：训练用 `training_recipe`，评估可另给一份，不给就复用训练的。

两份 `parse_from_config` 分工不同：`TEQuantizationRecipe` 那份（`143-175`）负责校验 fp8/fp4 互斥、custom 必填工厂；`TEQuantizationParams` 那份（`189-220`）只做字典到对象的组装。

`forward` 里每次套一层 `_get_fp8_autocast_for_quant_params`（`950-966`），出口有四种：

- 该层没配量化（`te_quant_params is None`）→ `nullcontext()`，跟随外层；
- 外层量化状态与配置一致、两个 override 开关都没开 → `nullcontext()`，不做覆盖；
- 该层要求覆盖成高精度（recipe 为空）→ `fp8_autocast(enabled=False)`，强制退回 BF16；
- 其余 → `fp8_autocast(enabled=True, fp8_recipe=..., fp8_group=amax_group)`。

recipe 字符串到 TE 对象的映射在 `278-326`：`tensorwise→Float8CurrentScaling`、`blockwise→Float8BlockScaling`、`mxfp8→MXFP8BlockScaling`、`nvfp4→NVFP4BlockScaling`、`custom→_get_custom_recipe`。

amax 的归约组由 `get_amax_reduction_group(with_context_parallel=True, tp_only_amax_red=...)` 给出。权重初始化另走 `fp8_model_init`（`223-266`），只有 `fp8_param` 为真才真把参数铸成 fp8。

`fp8_format` 只认 `e4m3`/`hybrid`，其它值直接 `ValueError`（`229-234`）。

这里有个与全局 fp8 不同的限制：per-module 量化不支持 delayed scaling，`TEQuantizationRecipe.parse_from_config` 见到 `delayed` 直接报错，自定义 recipe 则必须给 `custom_recipe_factory`（`156-168`）。

全局 `--fp8-recipe delayed` 仍然可用，它走的是整模型级别的 autocast，不经过这套 per-module 配置。

还有 `is_first_microbatch`（`787`、`952-959`）：告诉 TE「这是不是新 microbatch 的第一次 forward」，参数转置缓存靠它决定何时重建。

## 5. 上一篇的伏笔：TE 的 tracker 与 checkpoint

`TECudaRNGStatesTracker`（`3117-3149`）继承 TE 的实现，只补了 `is_initialized`/`reset`/`set_states`/`add` 四个方法，把 `_is_initialized` 这个 MCore 侧约定补齐。

`is_graph_safe_cuda_rng_tracker` 直接把它判为 graph-safe（`random.py:487-496`）。

`te_checkpoint`（`3152-3176`）是纯版本适配：TE ≥ 1.5.0 用关键字参数传 `distribute_saved_activations` / `get_rng_state_tracker` / `tp_group`，更老的版本只能按位置传。

语义与 MCore 自己的 `CheckpointFunction` 一致，差别是 fp8/fp4 的元数据由 TE 管。

这正是[第四篇]({{< relref "megatron-code-04-recompute" >}})里 `chunk_runner` 在 fp8/fp4 下选中的分支，它传进去的 `get_cuda_rng_tracker` 就是本篇那个 tracker 类。

## 6. attention 封装：GQA、CP、THD 三处最容易写错

- **GQA**：`num_gqa_groups = config.num_query_groups` 需要 TE ≥ 0.11.0；低于这个版本又开了 GQA 直接报错，不会静默降级（`1639-1646`）。
- **CP**：只有 `context_parallel_size > 1` 才传 `cp_group` / `cp_global_ranks` / `cp_stream`，且 `cp_stream` 是**类级共享**的一条 stream（`1677-1690`）；`cp_comm_type='a2a+p2p'` 还要 TE ≥ 1.12 并换成 hierarchical group。
- **THD**：`packed_seq_params` 先按版本裁字段（不同 TE 版本保留集不同，`1739-1765`）；forward 里若 `pad_between_seqs=False`，用 `cu_seqlens_*_padded` 顶替有效边界（`1841-1853`）；THD 下 `causal`/`no_mask` 会被改写成 `padding_causal`/`padding`，因为 cuDNN flash attention 只认这两种。

滑窗也是按层判断的：只有 `is_layer_window_attention(window_size, window_attn_skip_freq, layer_number)` 为真才传 `window_size`，所以同一个模型里可能只有部分层是滑窗（`1711-1719`）。

`num_splits` 要 TE ≥ 2.10、`attention_bias` 要 TE ≥ 1.2，都是硬门槛（`1832-1840`、`1857-1860`）。

dynamic CP 时 `forward` 会临时换掉 `self.cp_group`，算完再 `set_context_parallel_group` 恢复（`1805`、`1926-1933`）。

## 7. spec provider 就是装配清单

`TESpecProvider` 的方法清单（`transformer_engine_spec_provider.py:45-126`）就九个：

- 线性层：`linear`、`column_parallel_linear`、`row_parallel_linear`；
- norm 与融合：`fuse_layernorm_and_linear`、`column_parallel_layer_norm_linear`、`layer_norm`；
- attention 与 MoE：`core_attention`、`grouped_mlp_modules`、`activation_func`。

两个值得记住的分支：`fuse_layernorm_and_linear()` 返回 `True`，于是 LN+linear 合并成 `TELayerNormColumnParallelLinear`。

`layer_norm()` 在 QK LayerNorm 且 TE < 1.9 时退回 Apex 的 `FusedLayerNorm`，代码注释给的理由是那个版本用 TENorm 会明显伤害收敛。

`core_attention()` 还有第三条出口：`use_flex_attention=True` 时返回 `FlexAttention`（`79-82`），`fallback_to_eager_attn` 才退回本地 `DotProductAttention`。

`grouped_mlp_modules()` 在 TE < 1.7 时打一条 warning 并退回本地 `ColumnParallelLinear`/`RowParallelLinear`（`99-112`）——MoE 的 TE 支持有版本门槛。

## 8. 小结

- MCore 装配、TE 算子：切分与状态归 Megatron，GEMM/LN/attention/量化归 TE，边界在这个扩展文件里。
- `transformer_impl` 默认 `transformer_engine`，切换靠 `get_backend` 换 provider，骨架不动。
- `TELinear` 的 `parallel_mode` 只有 `column`/`row`/`duplicated` 三种；expert 是 `is_expert` 触发的通信外置，不是第四种。
- 量化是逐层开关：四种出口（两种 `nullcontext`、强制 BF16、真开 fp8_autocast），由每层自己的 recipe 与 override 开关决定。
- `TECudaRNGStatesTracker` 与 `te_checkpoint` 都只是接口/版本适配层，真正干活的是 TE。

下一篇：[CPU offload]({{< relref "megatron-code-07-cpu-offload" >}})讲 MCore 怎么把不活跃层的权重和激活下放到 CPU。

---

**版本提示（2026-10）**：

- GTP（Generalized TP）要求 TE ≥ 2.19，TE 侧按 `split_alignment` 探测 `te.pytorch.ops.GroupedLinear` 的签名（基线里还没有这段探测，dev 加了），这些都发生在算子装配处，「MCore 装配、TE 算子」的边界没有变。
- `mxfp8` / `nvfp4` 的 recipe 分支在本篇基线（f713506ce）里**已经存在**，不是新版才加的；读量化代码时以 `Fp8Recipe` / `Fp4Recipe` 枚举为准。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已到 2026-10-29，行号会漂移，定位请以符号名为准。
