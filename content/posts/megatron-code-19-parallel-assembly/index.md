---
title: "Megatron 源码精读（十九）：并行组装地图"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "parallel-state", "process-group", "rank-mapping"]
categories: ["训练框架"]
weight: 19
series: ["megatron-code"]
---

这一篇回答两个互逆的问题：给定一个物理 rank，它在 tp/pp/dp/cp/ep 的哪个位置？反过来，一次 all-reduce 的参与者是哪几个 rank？

**一句话结论：整张并行地图由「一条基公式 + 一个 mask」生成——`RankGenerator` 换 mask 枚举子组，`ProcessGroupCollection` 把结果收成一个可传参的对象。**

前面第 3 篇拆过 `parallel_state.py` 的全局变量，这一篇只看「map」这一半：rank 怎么变成组。

## 1. 问题：两个问题，一张地图

每一步训练都要答两次：**谁是同组**（做一次 TP all-reduce，哪些 rank 参与），**我是谁**（`tp_rank`/`pp_rank` 决定分片边界，如 `layer = pp_rank * layers_per_stage`）。

老 `mpu` 用一堆全局变量存答案，谁用谁 import。MCore 收成三层：`RankGenerator` 做纯数学、`initialize_model_parallel` 真正 `new_group`、`ProcessGroupCollection` 统一收纳并往下传。中间那步的 `create_group`（`parallel_state.py:213`）只是 `torch.distributed.new_group` 的薄封装，顺手把组登记进 `_global_process_group_list`。

顺带澄清：第 18 篇结尾提到的 `ParallelismConfig` 在基线里并不存在（`megatron/` 全仓无此符号），配置实际由 `RankGenerator` 的构造参数和 `initialize_model_parallel` 的形参承担。

## 2. 机制：一条基公式，一个 mask

### 2.1 基公式把 rank 写成各轴下标的加权和

`generate_masked_orthogonal_rank_groups` 的 docstring 直接给出正交并行的基公式（`parallel_state.py:273`，以 tp-dp-pp 为例）：

$$
r = i_{\mathrm{tp}} + i_{\mathrm{dp}}\, N_{\mathrm{tp}} + i_{\mathrm{pp}}\, N_{\mathrm{tp}} N_{\mathrm{dp}}
$$

$N$ 是各轴大小，$i$ 是该轴上的下标。**一个 global rank 就是各轴下标的加权和**，权重恰好是前缀积。

### 2.2 stride 是前缀积，长度比轴数多一

`prefix_product`（`parallel_state.py:303-308`）返回 `[1, a0, a0a1, ...]`，比输入多一个元素：

$$
\mathrm{stride} = \left(1,\ N_{\mathrm{tp}},\ N_{\mathrm{tp}}N_{\mathrm{dp}},\ N_{\mathrm{tp}}N_{\mathrm{dp}}N_{\mathrm{pp}}\right)
$$

长度是 len+1，因为首元素恒为 1（第一个轴的步长），`stride[i]` 恰好是前 i 个轴大小的乘积；末元素等于总规模，`decompose` 里用 `stride[:-1]` 显式跳过。若 `tp=2, dp=2, pp=4`，stride 就是 `(1, 2, 4, 8)`，下标 `(1, 1, 2)` 对应 rank `1 + 2 + 8 = 11`。

### 2.3 mask 决定这次枚举哪个轴

求「某个轴的组列表」只剩四步（`parallel_state.py:333-356`）：

1. 按 mask 把 `parallel_size` 劈成组内 `masked_shape` 与组外 `unmasked_shape`，stride 同样劈开；
2. `group_size = prod(masked_shape)`，`num_of_group = world_size // group_size`；
3. 第 $g$ 个组的「组外坐标」= `decompose(g, unmasked_shape)`；
4. 组内第 $j$ 个 rank = 组内坐标的点积 + 组外坐标的点积（`inner_product`）。

`decompose(index, shape, stride)` 就是 `idx[i] = (index // stride[i]) % shape[i]`，末尾还断言 `sum(idx*stride) == index` 自校验（`parallel_state.py:313`）。于是「求任意正交子组」变成同一段代码换 mask。

`get_mask`（`parallel_state.py:488`）的做法是：token 按 `-` 拆开，在 `order` 里 `index` 到哪一位就把哪一位置 True。mask 的位序永远跟着 `order` 走，换 `order` 不必改任何 mask 代码。

`RankGenerator`（`parallel_state.py:444-519`）在这套数学外面包了三件事：EP 与 CP 互斥的 `ep == 1 or cp == 1` 断言（450）；`world_size = tp·dp·pp·cp·ep`（461）；`order` 校验——缺的轴若 size≠1 直接 `RuntimeError`，size==1 则补到末尾（473-480）。`get_ranks(token)` 把 `"tp-dp"` 这类 token 转成 mask，最后统一加 `rank_offset`（503）。

## 3. 代码落点：16 GPU、tp=2、pp=4

docstring 里的经典例子（`parallel_state.py:682`）：`tp=2, pp=4, cp=1`，则 `dp = 16 / (2×4) = 2`；默认 `order="tp-cp-ep-dp-pp"`（`parallel_state.py:560`），cp/ep 都是 1，等效顺序 `tp-dp-pp`。

| 组 | 数量 | rank 列表 | 步长 |
|---|---|---|---|
| dp | 8 | `[0,2] [1,3] [4,6] …` | `tp`=2 |
| tp | 8 | `[0,1] [2,3] …` | 1 |
| pp | 4 | `[0,4,8,12] [1,5,9,13] …` | `tp·dp`=4 |

pp 组隔 4 取 1 不是巧合：`dp` 排在 `pp` 前面，pp 的步长就是 `tp·cp·ep·dp = 4`。把 pp 组写成 `[0,2,4,6]` 是这类文章最常见的错——那等于把 dp 的步长套给了 pp。

顺带一个数量关系：三类组都是对 16 个 rank 的一次完整划分，所以「组数 × 组大小」恒等于 world size（8×2、8×2、4×4）。

## 4. 两个生成器：attention 域与 expert 域

`initialize_model_parallel` 建两个 `RankGenerator`（`parallel_state.py:769-800`）：decoder 生成器 `ep=1, cp=cp_size`；expert 生成器 `cp=1, ep=ep_size, tp=expert_tp`（缺省等于 `tp`），dp 相应重算成 `world_size // (expert_tp·ep·pp)`。

拆两个，是因为同一批物理 GPU 有两套切法，但 pipeline 必须对齐。于是有三处硬约束：

- 一个生成器里不能同时有 EP 与 CP（`ep == 1 or cp == 1` 断言）：CP 只进 decoder 生成器，EP 只进 expert 生成器；
- 两个生成器的 PP 组必须逐项相等（`parallel_state.py:808`）——同一个 microbatch 在两种域里的 pp rank 不一致，流水线就散了；
- 除非 `order` 以 `pp` 结尾，否则要求两边 dp 大小相等（`parallel_state.py:802`）。

## 5. 组装：`ProcessGroupCollection` 的三类字段

`ProcessGroupCollection`（`process_groups_config.py:27`）是 dataclass，22 个字段全部 `init=False`（`process_groups_config.py:71-140`），必须建完对象再赋值。按用途分三类记住即可，不必逐个背：

- **模型并行**：`tp`、`pp`、`mp`、`cp`、`tp_cp`、`hcp`、`embd`、`pos_embd`；
- **专家并行**：`ep`、`expt_tp`、`tp_ep`、`tp_ep_pp`、`tp_dp_cp`；
- **数据并行**：`dp`、`dp_cp`、`dp_cp_ag`、`expt_dp`、`expt_dp_ag`、`intra_dp_cp`、`intra_expt_dp`、`intra_dist_opt`、`inter_dist_opt`。

`use_mpu_process_groups(required_pgs=None)`（`process_groups_config.py:169`）靠一张 `pg_to_func` 表（`process_groups_config.py:191-250`）把字段名映射到 `partial(get_xxx_group, check_initialized=False)`；不传 `required_pgs` 就全拉。传子集是常态：attention 只要 `['tp','cp','tp_cp']`（`attention.py:333`），DSA 这类变体只要 `['tp','cp']`（`dsa.py:1442`）。

下游统一「默认回退」：构造参数 `pg_collection=None` 时自己调 `use_mpu_process_groups()`（`transformer_block.py:295`）。

另一个入口 `setup_process_groups_for_optimizer`（`process_groups_config.py:263`）专供 optimizer/DDP：返回一份 dict，必要时给 `intra_dp_cp`、`intra_expt_dp` 各配一份 gloo 版本。

## 6. 四个容易踩的坑

- **有两个字段永远是 None**：`dp_cp_ag`、`expt_dp_ag` 在 `pg_to_func` 里是 `lambda: None`。它们是给 AG/RS overlap 单独建的通信器，得用 `create_all_gather_groups()`（`parallel_state.py:1356`）创建后再手工挂上去（`training.py:2237`）。所以「一键拿全所有组」这个假设会踩空。
- **rank 不是「从全局 rank 反向分解」出来的**：`get_tensor_model_parallel_rank`、`get_data_parallel_rank`（`parallel_state.py:1663`、`1824`，cp 版本没有覆盖值）优先返回 `set_*_rank()` 写入的覆盖值，否则直接问进程组 `group.rank()`。下标来自 `new_group` 时的 rank 顺序，不是运行时再套一遍公式。
- **`hcp` 不是单个组**：层级 CP 下它是 `List[ProcessGroup]`（`process_groups_config.py:92`），其余字段都是单个 `ProcessGroup`。
- **CP 组不止一个**：开 `--dynamic-context-parallel` 时，`initialize_model_parallel` 会在 DP×CP 上预建多套 2 的幂大小的 CP 组（`parallel_state.py:937-948`），运行期按 batch 挑。别把 `get_context_parallel_group()` 当成唯一答案。

## 7. 小结

- 一条基公式加一个 mask 就是整张地图；`prefix_product` 返回的 stride 比轴数多一个。
- `RankGenerator` 管 order 校验、EP/CP 互斥和 `rank_offset`，`get_ranks(token)` 换 mask 枚举子组。
- 16 GPU / tp=2 / pp=4 时，pp 组步长是 `tp·dp=4`，dp 组步长是 `tp`，tp 组步长是 1。
- decoder 与 expert 两个生成器：PP 组必须相等，dp 大小有条件相等。
- `ProcessGroupCollection` 的 22 个字段分三类收纳，两个 `*_ag` 字段要单独创建。

---
**版本提示（2026-10）**：上游把 CP 拓扑抽到 `megatron/core/context_parallel/`，其中 `groups.py` 用一条规则算合法的动态 CP 组大小（能整除 DP×CP 的 2 的幂）；dynamic CP 已接入标准 train/eval schedule，CP 组可以按 microbatch 选。`ProcessGroupCollection` 也扩了字段。但「一条基公式 + 一个 mask」的核心没变。
**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
