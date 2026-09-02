---
title: "Megatron 源码精读（十七）：MCore 架构与 layer spec 机制"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "mcore", "架构"]
categories: ["训练框架"]
weight: 17
series: ["megatron-code"]
---

这是系列的补充篇（延续[《Context Parallel 细节》]({{< relref "megatron-code-16-context-parallel" >}})）。前面 16 篇在「用」MCore，但没讲它**本身怎么运转**。本篇补上这块地基：**Megatron-MCore 如何用「模块规格（spec）」代替硬编码的类实例化，从而让同一套 Transformer 骨架既能跑 TE 的 FP8 路径、又能跑纯 torch 路径、还能换注意力/MLP 实现而不改框架代码**。核心代码在 `megatron/core/transformer/spec_utils.py`、`transformer_block.py`、`transformer_layer.py` 和 `megatron/core/models/gpt/gpt_layer_specs.py`。

搞懂这一篇，前面所有「为什么这行会走到 TE 的 column_parallel_linear」之类的疑问都能自洽。

---

## 1. 问题背景：为什么需要「spec」

普通开源 transformer 库（如 HF）写死结构：`BertLayer` 里就是 `BertAttention` + `BertIntermediate` + `BertOutput`，想换掉注意力要改 `__init__` 或子类化。Megatron 面临的问题更尖锐：

- 同一个 `TransformerBlock` 要同时支持 **Transformer Engine（FP8/融合算子）** 和 **纯 torch** 两种实现（`gpt_layer_specs.py` 里 `get_gpt_layer_with_transformer_engine_spec` vs `get_gpt_layer_local_spec`）；
- 同一个 `TransformerLayer` 要能**换成 MoE 层、MLA 层、hyper-connection 层**（第 14 篇、第 15 篇的模型都靠这个）；
- 用户/下游（如 NeMo）要在**不改框架代码**的前提下替换某个子模块（比如把 `CoreAttention` 换成 FlashAttention 或自定义 kernel）。

直接硬编码类名做不到这些。MCore 的答案是**把「用哪个类 + 传什么参数 + 含哪些子模块」抽成一个数据对象 `ModuleSpec`**，框架代码只调 `build_module(spec)`，具体类由 spec 决定。这就是「规格化 / 数据驱动装配」。

---

## 2. 现象：`ModuleSpec` 这个不起眼的数据类

`ModuleSpec`（`spec_utils.py:12-41`）就三个字段加一个 `__call__`：

```python
# spec_utils.py:29-41（节选）
@dataclass
class ModuleSpec:
    module: Union[Tuple, type]     # 模块位置 (module.path, ClassName) 或已导入的类
    params: dict = field(default_factory=lambda: {})   # 初始化参数
    submodules: object = None       # 递归的子模块 spec
    metainfo: dict = field(default_factory=lambda: {})

    def __call__(self, *args, **kwargs):
        return build_module(self, *args, **kwargs)
```

三个字段分别回答「用哪个类」「传什么参数」「里面还有什么子模块」。注意 `module` 既可以是**已导入的类**，也可以是 `(模块路径, 类名)` 的元组——后者走 `import_module`（`spec_utils.py:44-56`）做**惰性动态导入**，从而避免在 `gpt_layer_specs.py` 顶部把所有 backend 都 `import` 一遍（TE 没装时也能安全加载）。

真正干活的是 `build_module`（`spec_utils.py:74-129`）：

```python
# spec_utils.py:111-122（节选）
# 把 spec.submodules 作为 kwargs 塞进模块初始化
if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:
    kwargs["submodules"] = spec_or_module.submodules
return module(
    *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs
)
```

核心就一句：**把 `params` 和 `submodules` 两个字典合并后传给模块构造器**。于是「装配一个模块」就收敛成「构造一个 spec 然后调用它」这个单一入口，框架侧永远只写 `build_module(spec, ...)`。

---

## 3. 根因：一个 `TransformerLayerSubmodules` 定义了层内所有「插槽」

spec 是递归的：`TransformerLayer` 的 spec 里嵌套着 `self_attention`、`mlp` 等子 spec，而定义这套「插槽」的是 `TransformerLayerSubmodules`（`transformer_layer.py:251-295`）：

```python
# transformer_layer.py:279-295（节选）
@dataclass
class TransformerLayerSubmodules:
    input_layernorm: LayerNormBuilder = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_cross_attn_layernorm: LayerNormBuilder = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_mlp_layernorm: LayerNormBuilder = IdentityOp
    mlp: MlpBuilder | type[IdentityOp] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)
```

注意每个槽的默认值是 `IdentityOp`——**没填的槽就是一个 identity 占位**。这解释了一个常见困惑：为什么 `TransformerLayer` 的 forward 里既有 `input_layernorm` 又有 `pre_mlp_layernorm`，但标准 GPT 只有 `pre_mlp_layernorm`（`input_layernorm` 默认是 identity，`gpt_layer_specs.py:336` 那段就没给它赋值）。**「槽都挂在 forward 里，用不用由 spec 决定」**，这是 MCore 层结构「看起来比实际重」的根源。

`TransformerLayer.__init__` 就是逐个槽调 `build_module`（`transformer_layer.py:387-418`）：

```python
# transformer_layer.py:387-396（节选）
self.self_attention = build_module(
    submodules.self_attention, config=self.config,
    layer_number=self.layer_number, **attention_optional_kwargs, ...
)
self.self_attn_bda = build_module(submodules.self_attn_bda)
```

`__init__` 本身不含任何「self-attention 是哪种」的信息——它只是把 `submodules.self_attention` 这个 spec 实例化。换注意力实现的唯一动作，就是换掉 spec 里这个字段。

---

## 4. 解法一：一份 spec 撑起 N 层（fan-out）

`TransformerLayer.forward` 里 self-attention 是凭空出现的注意力类，那 N 层从哪来？答案在 `TransformerBlock` 的 `_get_block_submodules`（`transformer_block.py:238-276`）：

```python
# transformer_block.py:265-272（节选）
elif isinstance(spec, ModuleSpec):
    if issubclass(spec.module, TransformerBlock):
        return spec.submodules
    elif issubclass(spec.module, BaseTransformerLayer):
        num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
        return TransformerBlockSubmodules(
            layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
        )
```

关键在最后一行：**如果传进来的 spec 描述的是「一层」（`BaseTransformerLayer` 子类），就把它复制 `num_layers` 份**，变成 `layer_specs = [spec] * num_layers`。于是你用**一份 layer spec** 就装配出了一个完整的 N 层 block。

这里的 `BaseTransformerLayer`（`transformer_layer.py:298-312`）是个**空标记类**，它的存在唯一目的就是让 `_get_block_submodules` 能判断「这个 spec 是『一层』，需要 fan-out 成 N 层」，而非「已经是整个 block」。docstring 写得很直白：

```python
# transformer_layer.py:301-307（docstring）
# 主要目的是检查传入 spec 的模块是否是本类的子类，
# 从而允许在 TransformerBlock 中把这个 spec 扇出（fan-out）给所有层。
# 详见 transformer_block.py 的 `_get_block_submodules`。
```

之后 `_build_layers`（`transformer_block.py:343-387`）真正把 `layer_specs` 里每个 spec 实例化：

```python
# transformer_block.py:382-387（节选）
self.layers = torch.nn.ModuleList(
    [build_layer(layer_spec, i + 1) for i, layer_spec in enumerate(self.submodules.layer_specs)]
)
```

`build_layer`（`transformer_block.py:350-379`）里还做了两件重要的事：给每层算**全局 layer number**（`global_layer_number = layer_number + get_transformer_layer_offset(...)`，`transformer_block.py:351-353`），以及套上 FP8/FP4 量化上下文（`transformer_block.py:359-371`）。

---

## 5. 解法二：spec 工厂函数屏蔽 backend 差异

最后补「这份 spec 从哪来」。`get_gpt_layer_with_transformer_engine_spec`（`gpt_layer_specs.py:369-376`）是个纯工厂：

```python
# gpt_layer_specs.py:369-376（节选）
@copy_signature(get_gpt_layer_with_transformer_engine_submodules)
def get_gpt_layer_with_transformer_engine_spec(*args, **kwargs) -> ModuleSpec:
    enable_hc = kwargs.get('enable_hyper_connection', False)
    layer_module = HyperConnectionTransformerLayer if enable_hc else TransformerLayer
    return ModuleSpec(
        module=layer_module,
        submodules=get_gpt_layer_with_transformer_engine_submodules(*args, **kwargs),
    )
```

它只做两件事：**挑层类**（`TransformerLayer` vs `HyperConnectionTransformerLayer`）和**填子模块**。子模块由 `get_gpt_layer_with_transformer_engine_submodules`（`gpt_layer_specs.py:182-366`）按几十个 kwarg（`qk_layernorm`、`multi_latent_attention`、`num_experts`、`use_kitchen`…）分支填充：比如 MLA 走 `MLASelfAttentionSubmodules`（`gpt_layer_specs.py:281-308`）、普通走 `SelfAttentionSubmodules`（`gpt_layer_specs.py:336-351`）。所有 backend 差异（TE vs Kitchen vs eager）集中在这一条工厂链里，`TransformerLayer`/`TransformerBlock` 这些框架类对此一无所知。

---

## 6. 小结

- **spec 替代硬编码**：`ModuleSpec`（module/params/submodules 三字段）是 MCore 的装配单元，`build_module` 把它 + `params` + `submodules` 合并后实例化（`spec_utils.py:12-41`、`74-129`）。
- **惰性动态导入**：`module` 字段可为 `(路径, 类名)` 元组，`import_module` 按需加载，TE 缺装也能安全 import（`spec_utils.py:44-71`）。
- **槽 + 默认 identity**：`TransformerLayerSubmodules` 定义了层内所有插槽，未填者默认 `IdentityOp`，所以层结构「看着重、实际轻」（`transformer_layer.py:251-295`）。
- **一份 spec fan-out 成 N 层**：`_get_block_submodules` 依据 `BaseTransformerLayer` 标记把单层 spec 复制 N 份（`transformer_block.py:265-272`），`_build_layers` 逐层实例化并算全局 layer number（`transformer_block.py:343-387`）。
- **工厂屏蔽 backend 差异**：`get_gpt_layer_with_transformer_engine_spec` 挑层类 + 填子模块，TE/torch/Kitchen 全部收敛在这一条链里（`gpt_layer_specs.py:182-376`）。

一句话：**MCore 的「可插拔」不是靠继承树，而是靠「把类实例化动作数据化成 spec」**——框架类只剩 `build_module(spec)` 一个动作，所有「换什么」都变成「改 spec」这个数据操作。

下一篇补 pipeline parallel 的调度细节（interleaved / 1F1B vs eager、bubble 率、`num_microbatch` 与 `num_stages` 的关系），把第 3 篇并行拓扑里一笔带过的 schedule 展开讲。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/transformer/spec_utils.py`、`megatron/core/transformer/transformer_block.py`、`megatron/core/transformer/transformer_layer.py`、`megatron/core/models/gpt/gpt_layer_specs.py`。）
