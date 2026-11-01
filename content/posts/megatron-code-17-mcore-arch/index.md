---
title: "Megatron 源码精读（十七）：MCore 架构与 layer spec 机制"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "mcore", "架构"]
categories: ["训练框架"]
weight: 17
series: ["megatron-code"]
---

这一篇补系列缺的地基：MCore 为什么把「实例化哪个类、传什么参数、里面装哪些子模块」写成一份数据，而不是写死在 `TransformerBlock.__init__` 里？

**一句话结论：`ModuleSpec` 把类实例化这个动作数据化，框架层只剩 `build_module(spec)` 一个动作；「一份 spec 变 N 层」靠 `BaseTransformerLayer` 这个标记类判定。**

[第 16 篇]({{< relref "megatron-code-16-context-parallel" >}})讲的是 CP 怎么切数据，这一篇回到装配期：模型还没跑起来，层是怎么被拼出来的。

## 1. 问题：三种后端要共用一套骨架

HF 的 `BertLayer` 把 `BertAttention` 写死在 `__init__` 里，想换实现只能改类或继承。

MCore 却要同时满足三件互相冲突的事：同一套 block 既跑 TE（FP8、融合算子）又跑纯 torch；同一层骨架能换成 MoE、MLA、hyper-connection；下游（如 NeMo）不改框架代码就能替换某个子模块。

硬编码类名做不到。MCore 的答案是：把「用哪个类 + 传什么参数 + 里面有什么子模块」抽成 `ModuleSpec`，框架代码只写 `build_module(spec, ...)`。

## 2. 机制：`ModuleSpec` 是四字段的实例化指令

`ModuleSpec`（`spec_utils.py:12-41`）有四个字段，各答一个问题：

- `module`：用哪个类，或 `(模块路径, 类名)` 元组；
- `params`：传给构造器的关键字参数；
- `submodules`：递归的下一层 spec；
- `metainfo`：**不进构造器**的旁注，给外层装配代码读。

前三个由 `build_module` 消费，`metainfo` 不会——「谁消费哪个字段」是读这段代码的分水岭。

例如 GatedDeltaNet 把 input LN 融进 `in_proj`，spec 上就标 `metainfo={"fuse_input_layernorm": True}`；拼 layer 时外层据此把 `input_layernorm` 置成 `IdentityOp`（`experimental_attention_variant_module_specs.py:353`）。把 `metainfo` 当成构造参数，是读这段代码最常见的误解。

`build_module`（`spec_utils.py:74`）本身只有三个动作：函数原样返回（`IdentityFuncOp` 靠它）、`submodules` 整体塞进 kwargs、`params` 与调用方 kwargs 合并后实例化：

```python
# spec_utils.py:116-122（节选）
if spec_or_module.submodules is not None:
    kwargs["submodules"] = spec_or_module.submodules
return module(*args, **spec_or_module.params, **kwargs)
```

注意 `submodules` 是**整体**注入的，不拆包——拆包由各层自己的 `__init__` 负责。

`module` 还允许写成 `("megatron.core.extensions.transformer_engine", "TENorm")`，由 `import_module`（`spec_utils.py:44`）推迟到 `build_module` 时才导入。核心模型代码基本都直接传类，元组形式主要见于自定义 spec（`tests/unit_tests/transformer/test_spec_customization.py:62`）。

## 3. 代码落点：13 个插槽、一次 fan-out

### 3.1 `TransformerLayerSubmodules` 定义层内全部插槽

`TransformerLayerSubmodules`（`transformer_layer.py:252-296`）有 13 个字段：三个 layernorm 槽、三对「模块 + bda」、三个 hyper-connection 槽，最后加一张 `sharded_state_dict_keys_map`。

默认值只有两类：layernorm 与模块槽默认 `IdentityOp`，`*_bda` 默认 `IdentityFuncOp`（`transformer_layer.py:279-292`）。`IdentityOp.__init__(*args, **kwargs)` 吞掉全部构造参数（`identity_op.py:14`），空槽也能被 `config=` / `hidden_size=` / `eps=` 正常构造。

layernorm 槽更进一步：它的类型是 `LayerNormBuilder` 而不是 `ModuleSpec`，所以不走 `build_module`，而是当 builder 直接调用（`transformer_layer.py:364`）。

`*_bda` 为什么单独一个类？因为这一路要的是「先被调用一次、再返回一个函数」的两段式：`get_bias_dropout_add(...)` 返回的是 callable。

`IdentityFuncOp.forward` 于是也返回一个函数——取第一个参数、丢掉其余（`identity_op.py:35`）。用 `IdentityOp` 顶替，两段式的签名立刻对不上。

### 3.2 槽是空的，不等于这一层没有 LN

最常见的误读是：槽空着，就以为这一层没有 LN。dense TE 路径里 `linear_qkv` 用 `column_parallel_layer_norm_linear()`，即 `TELayerNormColumnParallelLinear`（LN 融进投影），于是 `input_layernorm` 保持 `IdentityOp`；`pre_mlp_layernorm` 同理，dense 时也是 `IdentityOp`，**只有 MoE 才填真 LN**（`gpt_layer_specs.py:354`）。换到 local 后端，两个槽都填真 LN（`gpt_layer_specs.py:468`、`486`）。

所以「标准 GPT 只有 pre_mlp_layernorm」是反的：dense TE 下两个槽都是空的，LN 藏在融合线性层里。权重名也得对齐——local spec 用 `sharded_state_dict_keys_map` 把 `input_layernorm.` 重写成 `self_attention.linear_qkv.layer_norm_`，让两种布局共用一个 checkpoint。

### 3.3 fan-out：一份 layer spec 变成本 stage 的 N 层

`_get_block_submodules`（`transformer_block.py:238-276`）只有三个分支，核心是第二支：

```python
# transformer_block.py:265-272（节选）
elif isinstance(spec, ModuleSpec):
    if issubclass(spec.module, TransformerBlock):
        return spec.submodules
    elif issubclass(spec.module, BaseTransformerLayer):
        num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
        return TransformerBlockSubmodules(
            layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl)
```

条件确实是 `issubclass(spec.module, BaseTransformerLayer)`；`BaseTransformerLayer`（`transformer_layer.py:298`）是个空壳标记类，docstring 明说它存在的唯一目的就是让这里判断「这个 spec 描述的是层」。`num_layers` 来自 `get_num_layers_to_build`（`transformer_block.py:88`），是**当前 PP stage** 要建的层数，不是全局 `num_layers`。

它会随 uneven PP（首/末 stage 单独层数）和 pipeline layout 变化，所以同一份 layer spec 在不同 stage 扇出的层数可以不同。

`_build_layers`（`transformer_block.py:343-387`）随后逐 spec 实例化，`build_layer` 里做两件事：算全局 layer number（`transformer_block.py:351-353`，叠加 `get_transformer_layer_offset`），以及套 FP8/FP4 量化上下文（359-371）。

量化上下文包的是**构造**而不是 forward：`is_init=True` 进的是 `fp8_model_init`，参数直接按 FP8 建出来。这也解释了为什么非得先算全局 layer number——首尾层要留在 bf16。

## 4. 工程细节与坑

- **N 层共享同一个 spec 对象**：`[spec] * num_layers` 复制的是引用，不是深拷贝。
- 正因为共享，`TransformerLayer.__init__` 可以放心把 `submodules.mlp` 原地改写成 `functools.partial`（`transformer_layer.py:432-437`），第二轮起自动跳过包装。这是有意为之，但也提醒别往 spec 上挂运行态。
- **异构层不走 fan-out**：混合 attention/MoE、mamba 混合这类逐层不同的模型，由 `get_gpt_decoder_block_spec`（`gpt_layer_specs.py:710`）直接产出 `layer_specs` 列表并按 PP 切好，`_get_block_submodules` 第一支原样返回。fan-out 只是「每层一样」时的快捷方式。
- **工厂挑层类，子模块工厂填槽**：`get_gpt_layer_with_transformer_engine_spec`（`gpt_layer_specs.py:369-377`）按 `enable_hyper_connection` 选 `HyperConnectionTransformerLayer` 还是 `TransformerLayer`，其余交给 `get_gpt_layer_with_transformer_engine_submodules`（`gpt_layer_specs.py:182`）。后者按十几个 kwarg 分支（MLA / MoE / qk_layernorm / kitchen…），TE 与 torch 的差异全收敛在这条链上。

## 5. 小结

- 装配单元是四字段的 `ModuleSpec`；`build_module` 只合并 `params` 与 `submodules` 后实例化，函数原样返回。
- `metainfo` 不进构造器，只描述 spec 自身，由外层装配逻辑读取。
- 层内 13 个插槽默认 `IdentityOp`；槽为空不代表功能缺失，dense TE 的 LN 就融在 `linear_qkv` / `linear_fc1` 里。
- fan-out 由 `issubclass(spec.module, BaseTransformerLayer)` 触发，复制的是本 stage 的层数；异构层改传显式 `layer_specs`。

---
**版本提示（2026-10）**：上游 dev 的 `transformer_layer.py` 改动很大，新增 mHC（Manifold-Constrained Hyper-Connections）层与更多 attention variant；`BaseTransformerLayer` 也不再是纯空壳，加了 `supports_mhc_connections`、`supports_wide_residual_connections` 两个标记属性，供 block 判断能否直接喂 n-stream 隐状态。spec 与 `build_module` 机制、fan-out 条件都没变，但行号漂移明显。
**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
