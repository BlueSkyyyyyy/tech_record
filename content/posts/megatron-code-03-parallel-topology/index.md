---
title: "Megatron 源码精读（三）：并行拓扑——parallel_state.py 精读"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "分布式"]
categories: ["训练框架"]
weight: 2
---

这是「Megatron-LM 源码精读」系列的第三篇。前一篇《模型并行的原理》讲过 TP/SP/PP/CP/DP/FSDP 各自的动机，本篇回答一个更具体的问题：**这些并行在真实的多卡集群里，到底如何把「全局 rank 编号」映射到一张张 GPU 上？** 答案全在 `megatron/core/parallel_state.py`（分析基准 commit `f713506ce`，约 2266 行）这一个文件里。

---

## 1. 问题背景：为什么需要一个「进程组拓扑」

Megatron 同时支持**五种正交的并行**：张量并行（TP）、序列/上下文并行（SP/CP）、专家并行（EP）、数据并行（DP）、流水线并行（PP）。每一种并行都要做集合通信（all-reduce、reduce-scatter、send/recv 等），而集合通信的载体是 PyTorch 的 `ProcessGroup`——它规定了「哪几个 rank 在一起通信」。

`parallel_state.py` 的职责就是用**一行 `order` 字符串**，从世界大小 `world_size` 里推导出所有这些进程组的成员，并缓存起来供全库查询（`get_*_group()` / `get_*_rank()` / `get_*_world_size()`）。

## 2. 核心公式：一个编号，五种身份

### 2.1 混合进制分解

整个文件的灵魂是 `generate_masked_orthogonal_rank_groups`（`parallel_state.py:250-356`）。它把每个全局 rank 看成一个**多维坐标**。docstring 里写死了这个约定（`parallel_state.py:270-273`）：

```
global_rank = tp_rank
            + dp_rank * tp_size
            + pp_rank * tp_size * dp_size
```

其中 `tp_rank \in [0, tp_size)`、`dp_rank \in [0, dp_size)`、`pp_rank \in [0, pp_size)`。

这本质上是一个**混合进制数**：把 global_rank 按 `[tp_size, dp_size, pp_size]` 的权重做分解，得到它在 TP/DP/PP 三个轴上的「坐标」。`decompose` 内嵌函数（`parallel_state.py:313-331`）就是用 `idx = [(index // d) % s for s, d in zip(shape, stride)]` 把这个坐标解出来。

### 2.2 mask 机制：任意并行组合 = 选几个轴

`generate_masked_orthogonal_rank_groups` 的第二个参数是 `mask: List[bool]`（`parallel_state.py:263-268`）。mask 标记「这次想要哪几个并行轴」：

- 想要 TP 组 → mask 只把 tp 轴置 True
- 想要 TP+DP 组 → mask 把 tp 和 dp 都置 True
- 想要 DP 组 → 只把 dp 置 True

函数把「masked（在组内变动的）轴」和「unmasked（跨组区分的）轴」分开（`parallel_state.py:333-338`），对每个 group 枚举组内 rank、对每个 group 枚举组间编号，最后用 `inner_product` 拼回 global_rank（`parallel_state.py:344-355`）。这样**一组公式就能生成所有种类的并行组**，不需要为每种并行单独写 rank 排布逻辑。

## 3. `RankGenerator`：把 `order` 变成排布

在上面的公式里，轴的顺序是写死的 `[tp, dp, pp]`。但 Megatron 允许用户指定不同的展开顺序——这就是 `RankGenerator`（`parallel_state.py:444-519`）。

### 3.1 `order` 字符串

`initialize_model_parallel` 的默认 `order="tp-cp-ep-dp-pp"`（`parallel_state.py:560`）。`RankGenerator.__init__` 把它拆成 `self.ordered_size`（`parallel_state.py:470-486`）：

```python
self.name_to_size = {"tp": ..., "pp": ..., "dp": ..., "ep": ..., "cp": ...}
self.order = order.lower()
for token in order.split("-"):
    self.ordered_size.append(self.name_to_size[token])
```

它先把五个并行的 size 存进 `name_to_size`，再按 `order` 里出现的前后顺序，把这些 size 排成 `ordered_size` 列表。这个顺序决定了后面 `generate_masked_orthogonal_rank_groups` 里 `parallel_size` 的混合进制权重。

一个约束：`RankGenerator.__init__` 里 `assert ep == 1 or cp == 1`（`parallel_state.py:450-453`）——**同一个 RankGenerator 里不能同时开 EP 和 CP**，因为它们对应两个不同的 rank 网格（EP 属于 MoE 的 expert 网格，CP 属于 attention 的序列网格）。

### 3.2 `get_ranks(token)`：一句话拿任意组

`RankGenerator.get_ranks`（`parallel_state.py:503-519`）是唯一的出口：

```python
def get_ranks(self, token):
    mask = self.get_mask(self.order, token)          # token 如 'tp-dp' → mask
    ranks = generate_masked_orthogonal_rank_groups(
        self.world_size, self.ordered_size, mask
    )
    if self.rank_offset > 0: ...                      # 平移 rank 偏移
    return ranks
```

`get_mask`（`parallel_state.py:488-501`）把 `token="tp-dp"` 翻译成「在 order 序列里，tp 和 dp 这两个轴对应的 mask 位为 True」。于是整个并行拓扑的生成被归一化成「**想拿什么组就报什么 token**」。

## 4. `initialize_model_parallel`：拓扑的装配车间

`initialize_model_parallel`（`parallel_state.py:545` 起，约 800 行）是入口。它先用 `world_size` 反推各并行的实际大小，再逐个创建进程组。

### 4.1 从 world_size 反推 DP 大小

docstring 给了一个 16 GPU、TP=2、PP=4 的经典例子（`parallel_state.py:682-696`）：

| 组类型 | 组数 | 成员（示意） |
|---|---|---|
| 8 个 DP 组 | 8 | `[g0,g2] [g1,g3] [g4,g6] ...` |
| 8 个 TP 组 | 8 | `[g0,g1] [g2,g3] [g4,g5] ...` |
| 4 个 PP 组 | 4 | `[g0,g4,g8,g12] [g1,g5,g9,g13] ...` |

实现上，`model_size = tp * pp * cp`，`data_parallel_size = world_size // model_size`（`parallel_state.py:732-737`）。注意这里**没有把 EP 算进 model_size**——EP 走的是专家侧的独立 `expert_decoder_rank_generator`（`parallel_state.py:780-800`），它把 tp 换成 `expert_tensor_parallel_size`、新加一个 `ep` 轴。

### 4.2 创建顺序：先把「要 SHARP 的 dp-cp 组」建出来

有意思的细节在 `parallel_state.py:839-895`：由于 NCCL 的 SHARP（in-network reduction）硬件限制**只能作用在第一个创建的 communicator 上**，代码会**提前、先建 `dp-cp` 组**（`decoder_rank_generator.get_ranks('dp-cp')`），确保 SHARP 落到数据并行域上，建完再把 `NCCL_COLLNET_ENABLE` 置 0（`parallel_state.py:917-919`）。

### 4.3 各组的创建清单

以下是主体创建代码与对应的 token（都是同一套 `get_ranks` 驱动）：

| 组 | token | 创建位置 |
|---|---|---|
| DP（含 CP） | `dp-cp` | `parallel_state.py:844` |
| 动态 DP×CP | `dp-cp` + `create_dynamic_dp_cp_groups` | `parallel_state.py:921-935` |
| DP | `dp` | `parallel_state.py:950` |
| CP | `cp` | `parallel_state.py:972` |
| Model Parallel（TP+PP 联合） | `tp-pp` | `parallel_state.py:1001` |
| TP | `tp` | `parallel_state.py:1018` |
| PP | `pp` | `parallel_state.py:1094` |
| Embedding 组 | 每 PP 组首尾 rank | `parallel_state.py:1123-1132` |
| Position embedding 组 | 每 PP 组首个 rank | `parallel_state.py:1134-1143` |
| TP+DP（+CP） | `tp-dp` / `tp-dp-cp` | `parallel_state.py:1151-1168` |
| TP+CP | `tp-cp` | `parallel_state.py:1174` |
| 专家 MP/TP/DP | `ep` / `etp` / `edp` | `parallel_state.py:1188` 起 |

### 4.4 Embedding 组的特殊之处

embedding 和 position embedding 不是按轴自动生成的，而是调 `get_embedding_ranks` / `get_position_embedding_ranks` 回调（`parallel_state.py:1123, 1134`）。默认实现是 `default_embedding_ranks`（`parallel_state.py:522-528`）：embedding 放在**每个 PP 组的首尾两个 stage**，position embedding 只放**首个 stage**（`parallel_state.py:531-534`）。这是为了在流水线里让词表权重只在少数 rank 上存在、其他 rank 通过广播拿到。

## 5. 查询层：全库如何拿到「我是谁」

`initialize_model_parallel` 建好所有组后，把「当前 rank 属于的那一个组」缓存到 module-level 的全局单例里（`_TENSOR_MODEL_PARALLEL_GROUP`、`_DATA_PARALLEL_GROUP` 等，见 `parallel_state.py:28-146`）。此后全库通过一组 getter 查询：

- `get_tensor_model_parallel_group()` / `get_tensor_model_parallel_rank()` / `get_tensor_model_parallel_world_size()`（`parallel_state.py:1464` / `1663` / `1635`）
- `get_pipeline_model_parallel_group()` / `..._rank()` / `..._world_size()`（`parallel_state.py:1473` / `1671` / `1643`）
- `get_data_parallel_group()` / `..._rank()` / `..._world_size()`（`parallel_state.py:1482` / `1824` / `1805`）
- `get_context_parallel_group()` / `..._rank()` / `..._world_size()`（`parallel_state.py:1518` / `1845` / `1837`）
- 专家系列：`get_expert_model_parallel_group()` 等（`parallel_state.py:1870` 起）

关键语义（这也是理解 Megatron 并行排布的核心）：

1. **`get_tensor_model_parallel_rank()` 是「位」**：在 TP 组内负责哪一列/行的权重切片。
2. **`get_pipeline_model_parallel_rank()` 是「层」**：在 PP 组内是第几个 stage（配合 `is_pipeline_first_stage` / `is_pipeline_last_stage` 判断首尾，`parallel_state.py:1679` / `1689`）。
3. **`get_data_parallel_rank()` 是「样本复制份」**：同一份模型参数在哪些 rank 上各有独立样本。
4. **`get_context_parallel_rank()` 是「序列段」**：序列被切成了几段，这段是第几段。

## 6. 一组 Demo：16 GPU，TP=2, PP=4, CP=1

按 `order='tp-cp-ep-dp-pp'`（默认），16 GPU、TP=2、PP=4、CP=1、EP=1 时：

- `model_size = 2 × 4 × 1 = 8`，`dp_size = 16 // 8 = 2`
- `ordered_size = [2, 1, 1, 2, 4]`（tp, cp, ep, dp, pp）
- global_rank 的混合进制权重：tp 位权重 1、cp 1、ep 1、dp 2、pp 4

以 `tp` 组为例（mask 只开 tp 轴），`get_ranks('tp')` 得到 8 组、每组 2 个相邻 rank：`[0,1] [2,3] [4,5] ... [14,15]`——TP 切的是相邻 rank，保证它们在一个 NVLink 域内高速通信。这正是 docstring 里强调的「相邻 rank 应在同一 DGX 节点」的原因（`parallel_state.py:693-696`）。

以 `pp` 组为例（mask 只开 pp 轴），得到 2 组、每组 4 个 rank：`[0,2,4,6]` 和 `[1,3,5,7]`——PP 跨的 rank 步长是 `tp×dp = 4`，因为一个流水线 stage 内部还包含 tp×dp 个 rank。

这就是「这些并行如何排布」的完整答案：**用 order 定轴的展开顺序，用混合进制分解把每个 rank 定位到多维坐标，再用 mask 摘出任意并行组合的成员**。

## 7. 末了：还有哪些没讲

- **专家并行（EP）/ MoE 的独立网格**：`expert_decoder_rank_generator`（`parallel_state.py:792-800`）给 MoE 建了第二套 rank 拓扑，`ep` 轴与 `cp` 互斥。这部分细节留到第 14 篇 MoE 展开。
- **虚拟流水线（interleaved 1F1B）**：`_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK/WORLD_SIZE`（`parallel_state.py:74-75, 739-747`）让每个物理 GPU 持有多个 stage，docstring 用 `[1,2][9,10]` 这种标记说明 layer 的交错分配（`parallel_state.py:582-595`）。
- **`initialize_model_parallel` 后半段的 `create_all_gather_groups`**（`parallel_state.py:1356`）为参数 all-gather 建了额外组，与 FSDP/ZeRO 相关，留到第 8 篇。

## 8. 下一篇预告

下一篇《重计算原理与代码》进入 `megatron/core/recompute.py`，看 Megatron 如何用「重算激活 + 选择性 checkpoint」把显存换计算。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/parallel_state.py`。）