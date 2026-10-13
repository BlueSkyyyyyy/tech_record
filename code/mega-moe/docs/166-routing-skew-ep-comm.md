# 路由偏斜对 EP 通信与显存的真实代价：代价集中在「hot rank」，而不是总字节

> 本文对应路线图 **P156-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 166 篇。
>
> 第 152 篇（P142-1）把「路由负载均衡」放回**计算侧**量，结论是一个反直觉的负结果：**grouped GEMM
> 本来就对专家负载不均不敏感**——偏斜只付「每段向上取整到 BLOCK_M」的零头（+6%~14%），真正的代价
> 在**显存**（`G·max_m`）与**通信**（重专家 rank）。但它明确留了一个口子（`docs/152` §8）：
>
> > 「本篇量的是**单算子**，未把 dispatch/combine 的通信不均纳入（见第 104 篇）。」
>
> 而它引用的第 104 篇（P94-1）量的是**均衡**路由下的前向/反向通信量。于是**偏斜路由下 EP 通信到底
> 贵多少、贵在哪**，从没被单独量过。本篇补上，结论是又一层「错位」：
>
> **偏斜几乎不改变 EP 通信的「总字节」——它把字节集中到被系统性偏好的那个 rank 上，代价是
> 「straggler」：hot rank 的 dispatch 收 / combine 发是均值的 2.9~4.3×。在一条健康的 NVLink 链路上，
> 这个 straggler 会像第 155 篇的 `max(comm, comp)` 一样 gate 住整个 step；只是本机这条 8 卡共享
> NCCL all-to-all 的固定成本（~4.7ms）**太大**，把 straggler 盖住了（2.9× straggler 只让墙钟 +7.9%）。**
>
> 可跑性：路由夹具、字节账、真实 `deep_ep.utils.refs.dispatch` 交叉验证全用**真实 DeepEP 源码**
> （逐行加载，绕开 `deep_ep/__init__.py` 的 NCCL 阻塞），纯 PyTorch / host，H100 真机可跑；
> 8×H100 真实 ragged all-to-all 也真机量。**Mega MoE 融合 kernel 本体 SM100 专属、仅分析**。
> 实验 `harness/exp_routing_skew_comm.{py,sh,out.txt}`（真实源码 sha256 `refs.py c5f551b4…` /
> `gate.py 5372e456…` / `math.py ed0dfb4b…`；本机 8 卡共享、无 Fabric Manager，故 `NCCL_NVLS_ENABLE=0`）。

---

## 1. 结论先行

固定一套 DeepSeek 量级配置：每 rank 本地 `T_local=2048` token、`H=7168`、`E=64` 专家、`topk=2`、
`world=8`（每 rank 8 个专家）。用 DeepEP 自带的负载夹具让**特殊 rank 0 被系统性偏好 `ratio` 倍**：

| `ratio` | dispatch 收端 max/mean | combine 发端 max/mean | dispatch 总行数 | 结论 |
|---|---|---|---|---|
| 1.0（均衡） | 1.00 | 1.04 | 30952 | 基线 |
| 1.5 | 1.41 | 1.47 | 30952 | 总字节 **不变** |
| 2.0 | 1.78 | 1.85 | 30952 | 总字节 **不变** |
| 4.0 | **2.91** | **3.03** | 30952 | 总字节 **不变** |
| 8.0 | **4.25** | 4.25 | 26344 | 极端偏斜反而**总行数下降** |

三句话：

1. **偏斜把代价集中到 hot rank**：dispatch 收、combine 发在 hot rank 上是均值的 2.9~4.3×，
   而全局总行数几乎不动（`ratio≤4` 时完全相同）。这不是「通信量变大」，是「通信量搬家」。
2. **极端偏斜甚至减少总行数**：`ratio=8` 时每 token 只触及 **1 个** rank（两个 top-k 专家落在同一
   rank），dispatch 在传输层去重后每 token 只发 1 行——总行数从 30952 掉到 26344（−15%），但
   hot rank 仍是均值的 **4.25×**。机理见 §2 的 `upper_bound_per_token`。
3. **本机墙钟被固定成本盖住**：真实 8 卡 ragged a2a 在 221.7MB/rank 时 `t_base=5203us`，
   straggler 2.9× 只让墙钟涨到 5616us（**+7.9%**）；而 27.7MB/rank 时几乎平（+1.0%）。因为单次
   a2a 的**固定成本 ~4.7ms** 主导，边际带宽高达 ~400GB/s（§4）。

第 142 篇说「偏斜的代价在显存与通信」；本篇把这半句量化：**通信的代价是 hot rank 的 straggler，
而非总字节**——这也正是 EPLB 冗余专家（第 53 篇）存在的理由。

---

## 2. 偏斜从哪来：DeepEP 的可控不均衡夹具 `gate.py`

真实 router 输出不可控、无法稳定复现「某 rank 流量是别人几倍」的最坏情况。DeepEP 因此在
`deep_ep/utils/gate.py` 里造了一个**可控的 rank 级不均衡**夹具（第 116 篇拆过它，但没把它接到
通信量上）。核心是 `generate_rank_count`（`gate.py:32-113`）：

- 先按均匀随机给每个 token 抽 `topk` 个专家、折算到 rank，量「这个 token 触及了几个不同 rank」`a`
  （`gate.py:43-52`）；
- **再把 `a` 夹到上界** `upper_bound_per_token = int(num_normal_ranks / ratio) + 1`
  （`gate.py:54`、`:57`，`num_normal_ranks = num_ranks-1` `:41`）；
- 然后让**特殊 rank 0** 出现在足量的 token 上（`special_token_count` `:62-68`、`must_mask` `:66`、
  断言 `:69`），其余 token 用同一个循环随机排列铺到普通 rank。

**这就是「为什么极端偏斜会减少总行数」**：`ratio=8`、`world=8` 时
`upper_bound = int(7/8)+1 = 1`，`a≡1`——每个 token 的 top-k 专家全落在**同一个** rank。dispatch 在
传输层按 `(token, dst_rank)` 去重（`refs.py:67` 的 `.any(dim=1)`），于是每 token 只发 1 行（均衡时
约 1.9 行）。**为了制造「rank 0 拿 ratio 倍流量」，夹具反而压低了每个 token 触及的 rank 数。**

`ratio` 的语义即 DeepEP 负载测试里「特殊 rank 相对其它 rank 的流量倍数」；`a` 的夹取保证了这一点在
实现上可行。另有连续（分数）版本的 `get_precise_unbalanced_scores`（`gate.py:116-138`，用 0.9
阈值把「选中/背景」分数分开）与只保比值不保均衡的 `get_random_unbalanced_scores`
（`gate.py:167-174`）。

---

## 3. 精确的 per-rank 行 / 字节模型

### 3.1 行数：dispatch 去重、combine 展开

- **dispatch**：参考实现 `refs.dispatch`（`refs.py:10-120`）对每个 dst rank，
  `mask_to_send = ((expert_start <= topk_idx) & (topk_idx < expert_end)).any(dim=1)`
  （`refs.py:60-69`）——**一个 token 对某个 dst rank 只发一行**（哪怕它命中该 rank 的多个专家）。
  故 rank `dst` 的收端行数 = `Σ_src #tokens(src) 命中 dst`。
- **combine**：官方训练 helper 明确 `dispatch` 的反向是 `combine`、且带 `do_expand=True`
  （第 104 篇核对过 `README.md:197-198/208/253`）。combine 在 **expanded** 空间搬：每个 rank 最终
  收回**自己 token 的 `topk` 份**（`T_local·topk` 行，常值），但**发送端**热度不同——hot rank 持有
  更多专家行，它的 combine 发端行数 = 它持有的 slot 数。

### 3.2 字节：`TokenLayout` 的权威公式

真实 per-token 记录字节来自 `deep_ep/include/deep_ep/layout/ep/token.cuh:33-40`
（`TokenLayout::get_num_bytes`）：

```
num_data_bytes = align_RDMA( align_TMA(hidden_bytes) + align_TMA(sf_bytes) + align_TMA(meta_bytes) )
num_bytes      = num_data_bytes + align_TMA( sizeof(mbarrier)=8 )
```

其中 `align_TMA=32`、`align_RDMA=64`（`common/compiled.cuh:67-68`）、
`meta = topk·(4+4) + (with_metadata ? (1+topk)·4 : 0)`（`token.cuh:25-26`）；dispatch 的构造在
`csrc/kernels/ep/dispatch.hpp:26-28`（`TokenLayout(hidden·1B, ceil_div(hidden,32)·4, topk, true)`）、
combine 在 `csrc/kernels/ep/combine.hpp:19-21`（`TokenLayout(hidden·2B, 0, topk, false)`）。

复刻后：`hidden=7168, topk=2` → **dispatch = 8128 B**（含每 token mbarrier = 8160 B）、
**combine = 14432 B**——与第 162 篇（P152-1）在 C++ 真机上量的 `8160 / 14432` **逐字节对上**。

把 §2 的偏斜行数乘上这两个常数，即得表 1 的「dispatch 收端 / combine 发端 max/mean」。**同一份字节账，
straggler 与行数的 straggler 一致**——因为每行字节是常数。

---

## 4. H100 真机：两个 regime

用真实 8×H100 ragged `all_to_all_single` 搬 dispatch 主数据段 `fp8 [rows, H]`，行数按 §2 的偏斜
分布给（每个 rank 发自己那份，`refs.py:93-110` 的「先交换 count 再搬数据」两段式）：

**C1 小消息（延迟域，T_local=2048 → 27.7 MB/rank）**

| `ratio` | `t_a2a` us | `t/t_base` | straggler/mean | 有效 GB/s |
|---|---|---|---|---|
| 1.0 | 4719 | 1.000 | 1.00 | 5.9 |
| 2.0 | 4734 | 1.003 | 1.78 | 5.9 |
| 4.0 | 4765 | 1.010 | 2.91 | 5.8 |
| 8.0 | 4731 | 1.003 | 4.25 | 3.1 |

**C2 大消息（字节域，T_local=16384 → 221.7 MB/rank）**

| `ratio` | `t_a2a` us | `t/t_base` | straggler/mean | 有效 GB/s |
|---|---|---|---|---|
| 1.0 | 5204 | 1.000 | 1.00 | 42.6 |
| 2.0 | 5369 | 1.032 | 1.78 | 41.3 |
| 4.0 | 5616 | **1.079** | 2.91 | 39.5 |
| 8.0 | 5572 | 1.071 | 4.27 | 21.1 |

**关键归因**：C1→C2 增加 194 MB/rank，墙钟只涨 ~484us → **边际带宽 ~400 GB/s**；但单次 a2a 有
**~4.7ms 的固定成本**（C1 的 27.7MB 也要 4.7ms，有效 5.9 GB/s）。这和第 162 篇
「字节缩 4× 只省 4.5% 墙钟（时延主导）」、第 155 篇「isolated `t_comm`=3180us/176MB」是同一现象：
**本机这条 NCCL a2a 远未跑满带宽，固定成本主导**。于是 2.9× 的 straggler 只贡献 +7.9% 墙钟。

**这正是「为什么 straggler 在本机被掩盖、在真机上是大事」**：在一条健康的 NVLink 链路上（边际
带宽上千 GB/s、固定成本微秒级），hot rank 多收 2.9× 字节会直接 gate 住 step——就像第 155 篇
正对照里 `t_pipe ≈ max(t_comm, t_comp)`。本机的数字只能证明**方向**（偏斜 → straggler → 墙钟单调
上升），绝对值受共享节点/no-NVLS 拖累。

> 注意 C2 的 `ratio=8` 有效 GB/s 掉到 21.1：因为它的**每 rank 总行数更少**（§2 的 `a≡1`），
> 但固定成本一样，分母（字节）变小。

---

## 5. 交叉验证：真实 `refs.dispatch`

用真实 DeepEP 参考实现 `refs.dispatch`（`refs.py:10-120`，含 5 条真实 NCCL a2a
`refs.py:105-110`）跑一遍，把它返回的 `num_recv_tokens_per_rank`（`refs.py:93-94`）与我们的
「去重命中」独立模型逐元素对拍（`T_small=512`）：

| `ratio` | `refs` rank0 从各 src 收（`refs.py`） | 独立模型 per-src 命中 | match |
|---|---|---|---|
| 1.0 | `[121,121,127,121,130,108,117,139]`（sum 984） | 同 | **True** |
| 4.0 | `[349,353,353,353,353,349,353,353]`（sum 2816） | 同 | **True** |

即 §3 的行数模型与 DeepEP 官方参考实现**逐元素一致**——模型不是我们自己编的，是 DeepEP 语义
的可执行复刻。

---

## 6. 显存 / 缓冲区：偏斜不改分配，EPLB 才是正解

DeepEP 的 dispatch 缓冲按**最坏情况** `num_max_tokens_per_rank · num_ranks` 分配
（`csrc/kernels/ep/dispatch.hpp:39/99/111/203`、`ep.py` 的 buffer size 计算）——所以 **偏斜不会
OOM、也不会改变分配**，只改变**实际 NVLink 流量的分布**（上述 straggler）。每 token 记录
dispatch 8128 B（含 mbarrier 8160）/ combine 14432 B（§3.2）。

**结构性正解是 EPLB 冗余专家**（第 53 篇，`deep_ep` 的 `lb_prefetch_weights`/`lb_reduce_grads`）：
把被系统性偏好的专家**复制到 peer 冗余槽**，让「hot expert」的流量摊到多个 rank 上，straggler
被拉平。第 53 篇给出的 host 上界 `get_theoretical_num_sms`
（`deep_ep/buffers/ep.py:431`，每 SM HBM 读/写 `180/45 GB/s`，`envs.py:197/204`）与这里的
「hot rank 单点超载」是同一问题的资源侧刻画。

结合第 139 篇（aux-loss-free bias）与第 152 篇（GEMM 不为偏斜所动）：**均衡机制的价值要从
「hot rank 的通信 + 显存」去算，而不是从 GEMM 时间去算**——本篇把这句话的通信半边坐实。

---

## 7. 与专题其余篇的关系 / 为什么这重要

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 142 篇（P142-1） | 偏斜对 grouped GEMM 的**计算侧**代价（结论：免役）；点名未含通信 | 通信侧：hot rank 的 straggler |
| 第 104 篇（P94-1） | **均衡**路由下前向/反向通信量模型与实测 | 偏斜路由下的通信量**分布** |
| 第 162 篇（P152-1） | 真实 per-token 字节口径（dispatch 8160 / combine 14432） | 把这套字节账接到偏斜行数上 |
| 第 53 篇（P43-1） | EPLB 冗余专家的预取/归约原语 | 为什么需要它：hot rank 超载 |
| 第 139 篇（P129-1） | aux-loss / z-loss / bias 三种均衡机制的算子与梯度 | 均衡的收益在通信/显存侧 |
| 第 144/155 篇（P134/P155-1） | 前向/流水线的通信-计算重叠（`max(comm,comp)`） | straggler 如何进入这个 max |

**为什么 Mega MoE 关心这个**：融合 kernel 把 dispatch/combine 切成 per-SM 的 chunk 喂进 ring
buffer（第 9/11 篇的 `num_bytes_per_pull`、第 32 篇的 host 编排），并按 locality domain 分域调度
（第 120 篇）。**偏斜会把「hot rank 的通信 warp」变成整 kernel 的尾波**——这正是 EPLB 与均衡 bias
在训练侧存在的意义。SM100 融合 kernel 本体的这套机制**仅源码分析**（`csrc/apis/mega_moe.hpp:256/367`
只在 `arch_major==10` 派发）。

---

## 8. 边界与复现

**能跑（H100 sm90 真机）**：§2 的偏斜夹具（真实 `gate.py`）、§3 的字节/行模型、§5 的真实
`refs.dispatch` 交叉验证（含 5 条真实 NCCL a2a）、§4 的 8×H100 ragged all-to-all（C1/C2 两 regime）。

**仅分析（SM100 专属）**：Mega MoE 融合 kernel 本体的 per-SM chunk 调度、NVLink L2 直写 combine、
warp 级 SM 预算（第 9/10/11/144/155 篇）。

**边界**：① 本机 8 卡共享（另有一路自驱任务占满 GPU0）、无 Fabric Manager（关 NVLS），故 §4 的
绝对带宽是**受争用下界**，但**偏斜→straggler→墙钟单调上升**的方向在 3 次取最小后稳定；② §4 只搬
dispatch 主数据段（`fp8 [rows,H]`），未叠 5 条字段 a2a（第 162 篇已证 5 条并发的固定成本同量级）；
③ 偏斜夹具是 DeepEP 测试用的**合成**最坏情况，真实 router 的 rank 级分布介于均衡与 `ratio≈4` 之间。

复现：

```
bash code/mega-moe/harness/exp_routing_skew_comm.sh \
  > code/mega-moe/harness/exp_routing_skew_comm.out.txt
```

真实源码：`~/github/DeepEP@93eb6eb`（`deep_ep/utils/{gate,refs,math}.py`、
`deep_ep/layout/ep/token.cuh`、`deep_ep/include/deep_ep/common/compiled.cuh`、
`csrc/kernels/ep/{dispatch,combine}.hpp`、`deep_ep/buffers/ep.py`、`deep_ep/utils/envs.py`）。
