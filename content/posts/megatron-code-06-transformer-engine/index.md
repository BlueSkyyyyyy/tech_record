---
title: "Megatron 源码精读（六）：与 Transformer Engine 的关系"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "transformer-engine", "fp8"]
categories: ["训练框架"]
weight: 6
series: ["megatron-code"]
---

上一篇[《随机种子的设置》]({{< relref "megatron-code-05-rng-seeds" >}})在两处留了钩子没展开：一是 `initialize_rng_tracker` 里「装没装 TE、`use_te_rng_tracker` 开没开」决定用哪套 tracker，二是结尾预告的 `te_checkpoint` 和 FP8 元数据管理。本篇把 Megatron Core 与 Transformer Engine（下称 TE）的**边界**讲清：哪些模块走 TE、桥接层 `megatron/core/extensions/transformer_engine.py` 干了什么、FP8/FP4 的训练路径怎么接管前向反向。

---

## 1. 一句话定位：MCore 是「装配器」，TE 是「运算引擎」

Megatron Core（MCore）自己并不实现 GEMM、LayerNorm、attention 的 CUDA kernel，它负责的是**并行切分、通信编排、状态管理**这一套上层逻辑；真正算矩阵乘、算 softmax 的那部分，从某个版本起大量外包给了 NVIDIA 单独维护的 Transformer Engine。落到代码上：

- `TransformerConfig.transformer_impl`（`transformer_config.py:1436`）可取 `'local'` / `'transformer_engine'` / `'inference_optimized'`，默认就是 `'transformer_engine'`——也就是说**默认路径下 transformer 层就是 TE 在跑**。
- 选哪套实现不靠到处写 `if config.transformer_impl == ...`，而是靠一个 `BackendSpecProvider` 抽象：`transformer_engine_spec_provider.py` 里的 `TESpecProvider` 回答「TE 后端用哪个 Linear、哪个 ColumnParallel、哪个 attention」，本地实现另有对应 provider。

这一节先建立心智模型：**一套 transformer 层的骨架（`transformer_layer.py` / `transformer_block.py`）是不变的，变化的是 spec provider 注入进来的具体算子**。ME 对 TE 的所有适配，都收在 `extensions/transformer_engine.py` 这一个 3400+ 行的文件里。

## 2. 入口：`input` 到的 `HAVE_TE` 与可选依赖

TE 对 Megatron 而言是**可选依赖**，所以文件头部先做了一坨防御性导入（`transformer_engine.py:71-87`）：

```python
try:
    import transformer_engine as te
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast, fp8_model_init
    HAVE_TE = True
except ImportError:
    if TYPE_CHECKING:
        # 类型检查时假装有 TE
        ...
    else:
        from unittest.mock import MagicMock
        te = MagicMock()
        HAVE_TE = False
```

没装 TE 时 `te` 是个 `MagicMock`、`HAVE_TE=False`，于是每个包 TE 的类里都先抛一个明确的 `ImportError` 提示装 TE（如 `TELinear.__init__` 的 `transformer_engine.py:773-777`）。这是典型的「软依赖 + 运行时报错」模式，保证装不装 TE 代码都能 import 成功、只在真正用到时才炸。

## 3. 线性层的三类封装

TE 自带 `Linear`、`LayerNormLinear`、`DotProductAttention` 等算子，MCore 不能直接用，原因是它要在 TE 算子外面**再包一层并行语义**——TP 怎么切、梯度在哪 reduce、FP8 recipe 怎么选。三类封装对应三种切法：

| 封装类 | 基类 | 并行语义 |
|---|---|---|
| `TELinear`（`transformer_engine.py:737`） | `te.pytorch.Linear` | 通用，`parallel_mode` 支持 `column`/`row`/`duplicated` |
| `TEColumnParallelLinear`（`transformer_engine.py:1234`） | `TELinear` | 按输出维切，对应 Megatron `ColumnParallelLinear` |
| `TERowParallelLinear`（`transformer_engine.py:1479`） | `TELinear` | 按输入维切，对应 `RowParallelLinear` |

关键在 `TELinear.__init__` 里那段 `parallel_mode` 的分流（`transformer_engine.py:862-892`）：

- 普通 TP：`te_parallel_mode` 保持 `column`/`row`，TE 内部自己管 all-gather / all-reduce；
- `duplicated`（非张量并行，权重在各 TP rank 复制）：`tp_group_for_te=None`、`tp_size=1`，梯度之后还要在 TP 组上再 reduce（`transformer_engine.py:927-932` 设置 `param.allreduce` / `sequence_parallel` / `tensor_model_parallel=False`）；
- **expert 层**（`is_expert=True`）：`explicit_expert_comm=True`，把 `te_parallel_mode` 置 None、通信交给外层 token_dispatcher 做（`transformer_engine.py:882-892`），因为 MoE 里 TP/EP 通信不在单个 linear 内部。

另一个重要细节：RNG tracker 的注入。`transformer_engine.py:863-871` 里 expert 层用 `get_expert_parallel_rng_tracker_name()`、非并行层用 DP tracker，把上一章讲的「命名态」通过 `extra_kwargs["rng_tracker_name"]` 传给 TE（TE >= 1.7.0），这样 TE 内部的 dropout 分支用的就是 Megatron 那套随机流，而不是 TE 自己的。

## 4. 量化（FP8/FP4）是怎么接进来的

TE 的核心卖点就是 FP8。MCore 这层要解决的是「**每一层到底用不用 FP8、用哪种 recipe**」，以及「**怎么在 PyTorch autocast 之外再套一层量化的 autocast**」。这套机制的入口是 `TELinear.forward`（`transformer_engine.py:950-966`）：

```python
def forward(self, x):
    _is_first_microbatch = None if self.disable_parameter_transpose_cache else self.is_first_microbatch
    quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)
    with quant_context:
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
    self.is_first_microbatch = False
    if self.te_return_bias:
        return out
    return out, None
```

每一层 forward 前先问一句 `_get_fp8_autocast_for_quant_params`，拿到一个要么是 `fp8_autocast(enabled=True, fp8_recipe=..., fp8_group=...)`、要么是 `nullcontext()` 的上下文，包住 TE 的 `forward`。这样**量化开关是逐层可配的**。

配方的载体是两层的 dataclass：

- `TEQuantizationRecipe`（`transformer_engine.py:98`）：单层用什么 `fp8_quantization_recipe` / `fp4_quantization_recipe`，以及 `fp8_format`（`e4m3`/`hybrid`）、`fp8_param`、`tp_only_amax_red` 等开关；
- `TEQuantizationParams`（`transformer_engine.py:177`）：训练/评估各用一份 recipe（`training_recipe` / `evaluation_recipe`），由 `parse_from_config`（`transformer_engine.py:189-220`）从 `quant_recipe` 配置字典里解析。

把配置里的字符串 recipe 映射成 TE 的 recipe 对象，发生在 `_get_fp8_autocast_for_quant_recipe`（`transformer_engine.py:278-326`）：`tensorwise→Float8CurrentScaling`、`blockwise→Float8BlockScaling`、`mxfp8→MXFP8BlockScaling`、`custom→_get_custom_recipe`；同时算好 amax 的归约组 `get_amax_reduction_group(...)`，`tp_only_amax_red` 控制是否只沿 TP 归约。这一段和 `TransformerConfig` 里的 `fp8`/`fp8_recipe`（`transformer_config.py:657-670`）是一一对应的。

还需要注意 `is_first_microbatch` 这个 flag——它告诉 TE「这是不是新的 microbatch 第一个 forward」，用于把参数转置（或 amax 状态）在 microstep 层面做一次性的操作，之后复用 cache（`transformer_engine.py:952-959`）。

## 5. 上一章的伏笔：`TECudaRNGStatesTracker` 与 `te_checkpoint`

### 5.1 TE 的 RNG tracker

上一章讲到 `random.py` 的 `initialize_rng_tracker` 会按 `use_te_rng_tracker` 选 `TECudaRNGStatesTracker`。这个类定义在 `transformer_engine.py:3117-3149`，继承 `te.pytorch.distributed.CudaRNGStatesTracker`，只补了一件事——**让它和 Megatron 自带 tracker 接口对齐**：

```python
class TECudaRNGStatesTracker(te.pytorch.distributed.CudaRNGStatesTracker):
    def is_initialized(self):    # 用内部 _is_initialized 标志
        return self._is_initialized
    def reset(self):             # 复位后 _is_initialized=False
        super().reset(); self._is_initialized = False
    def set_states(self, states):
        super().set_states(states); self._is_initialized = True
    def add(self, name, seed):
        super().add(name, seed); self._is_initialized = True
```

为什么要包这一层？因为 TE 的 tracker 天然 cudagraphable（上一章 `_get/_set_all_rng_states` 里 `cloned`/`live` 的复杂性，在 TE 版这边就被基类吃掉了），Megatron 希望两套 tracker 能无缝互换，所以用 `_is_initialized` 这个额外标志补齐「我有没有被 reset 过」这一条接口语义。

### 5.2 `te_checkpoint`：重计算交给 TE

上一章的 `CheckpointFunction.backward` 是 Megatron 自己的 activation checkpointing；TE 也有一套。`te_checkpoint`（`transformer_engine.py:3152-3176`）就是一层版本适配的薄封装：

```python
def te_checkpoint(forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs):
    from transformer_engine.pytorch.distributed import checkpoint
    if is_te_min_version("1.5.0"):
        return checkpoint(forward_func, *args, distribute_saved_activations=..., get_rng_state_tracker=..., tp_group=tp_group, **kwargs)
    else:
        return checkpoint(forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args)
```

逻辑和上一章完全相同（`distribute_saved_activations` 是 distributed checkpointing 的 key，`get_rng_state_tracker` 把 RNG 快照/回退交给 TE 的 checkpoint 实现），差别只在 1.5.0 前后参数从位置参数变成了关键字参数。这也是 MCore 大量用 `is_te_min_version(...)` 做版本分流的缩影。

## 6. attention、RoPE、cross_entropy 等杂项封装

`transformer_engine.py` 后半段（`1589` 起）是一串「把 TE 能力包成 MCore 接口」的封装，模式一致：**前置版本断言 + kwargs 转发到 TE**。

- `TEDotProductAttention`（`transformer_engine.py:1589`）：包 `te.pytorch.DotProductAttention`，额外处理 `attention_type`、GQA 的 `num_gqa_groups`（`transformer_engine.py:1639-1647`）、CP 的 `cp_group`/`cp_stream`（`transformer_engine.py:1677-1701`）、窗口注意力、`packed_seq_params` 的字段裁剪（`transformer_engine.py:1739-1765`）等。forward（`transformer_engine.py:1792`）里动态切 CP 组、做 THD 的 `cu_seqlens` padded 替换，最后恢复 CP 组。
- `TEDelayedScaling`（`transformer_engine.py:3081`）：包 `te.common.recipe.DelayedScaling`，把 `fp8_margin`、`fp8_amax_compute_algo` 等 config 喂进去。
- `fused_apply_rotary_pos_emb` / `fused_apply_rotary_pos_emb_thd`（`transformer_engine.py:3243-...`）：RoPE 的 fused 版本，thd 版本带 CP 分片。
- `te_parallel_cross_entropy`（`transformer_engine.py:3351`）与 `te_general_gemm`（`transformer_engine.py:3379`）：直接把 TE 的并行 CE loss 和 `general_gemm` 暴露出来，后者支持 fp32/bf16/fp16/fp8 的 GEMM（TN/NN/NT 布局）。
- MoE 相关（`transformer_engine.py:3419-3429`）：`fused_topk_with_score_function`、`fused_moe_aux_loss` 等在 TE >= 2.7 才可用。

这些封装有个共同套路——**每个能力都 `try: from transformer_engine... except ImportError: xxx = None`**，返回值置 None，调用方自行判断；版本差异用 `is_te_min_version` 门控，保证同一份 MCore 代码能跑在多个 TE 版本上。

## 7. 一个具体的桥接例子：`spec_provider` 是组装处的开关

要理解「哪些模块走 TE」，看 `transformer_engine_spec_provider.py` 最直接。`TESpecProvider` 就是一份「用 TE 时，各算子用哪个类」的清单：

- `linear()` → `TELinear`；`column_parallel_linear()` → `TEColumnParallelLinear`；`row_parallel_linear()` → `TERowParallelLinear`（`transformer_engine_spec_provider.py:45-55`）；
- `fuse_layernorm_and_linear()` → `True`，`column_parallel_layer_norm_linear()` → `TELayerNormColumnParallelLinear`（`57-63`），即把 LayerNorm 和紧邻的 linear 融成一个 TE 模块；
- `layer_norm()` → `TENorm`（`65-75`，QK LayerNorm 在 TE<1.9 时会退化回 Apex 的 `FusedLayerNorm`，因为那个版本收敛性差）；
- `core_attention()` → `TEDotProductAttention`，也可 `fallback_to_eager_attn=True` 退回本地 `DotProductAttention`（`77-85`）；
- MoE 的 `grouped_mlp_modules()` 和 `activation_func()` → TE 的 grouped MLP 与 `TEActivationOp`（`87-126`）。

所以「默认 transformer_impl=transformer_engine」的实现，本质就是**把这份 TE spec provider 注入到 transformer layer 的组装过程**，把本地 `ColumnParallelLinear`/`RowParallelLinear`/`DotProductAttention` 逐个替换成带 TE 前缀的类。`transformer_impl='local'` 则换成本地实现 provider，骨架代码一行不改。

## 8. 小结

- **MCore 是装配器、TE 是算子引擎**：并行切分、通信、RNG 状态归 Megatron，GEMM/LayerNorm/attention/FP8 归 TE，边界收在 `extensions/transformer_engine.py` 一个文件里。
- **软依赖 + 运行时校验**：TE 可选，`HAVE_TE`/`MagicMock` 防御性导入（`transformer_engine.py:71-87`），真正用到的类里再抛 `ImportError`。
- **Linear 包三层**：`TELinear` 承载 `column`/`row`/`duplicated` 三种并行语义，`TEColumnParallelLinear`/`TERowParallelLinear` 对应本地同名类；expert 层把 TP 通信让给 token_dispatcher。
- **量化逐层可配**：`TEQuantizationRecipe`/`TEQuantizationParams` 两级配置，`forward` 里用 `fp8_autocast`（或 `nullcontext`）包住 TE 前向，`_get_fp8_autocast_for_quant_recipe`（`transformer_engine.py:278-326`）负责字符串 recipe 到 TE recipe 对象的映射。
- **上一章的伏笔在这里兑现**：`TECudaRNGStatesTracker` 补齐接口语义（`transformer_engine.py:3117`）、`te_checkpoint` 做版本分流（`transformer_engine.py:3152`）。
- **装配开关在 spec provider**：`TESpecProvider` 就是「用 TE 时各算子选哪个类」的清单，`transformer_impl` 三选一本质是换 provider。

## 9. 下一篇预告

下一篇《CPU offload》承接本篇里 TP/PP 切分与激活管理的主题，讲 Megatron 怎么把不活跃层的权重和激活下放到 CPU、`get_cpu_offload_context`（本篇 `transformer_engine.py:3192` 已见雏形）如何包装 TE 的 offload，以及双缓冲、pin memory 这些把 PCIe 带宽压榨到极致的细节。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/extensions/transformer_engine.py` 与 `megatron/core/transformer/transformer_config.py`。）
