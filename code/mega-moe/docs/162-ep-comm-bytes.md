# 多卡 EP MoE 的真实通信字节口径：DeepEP dispatch/combine 的字段级拆解、真实 per-token 布局与 dtype 伸缩（8×H100 实测）

> 本文对应路线图 **P152-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 162 篇。
>
> 第 161 篇（P151-1）把多卡 EP 图接上真实 SM90 FP8 grouped kernel，第一次把「EP 通信 + FP8 量化」
> 叠进同一条端到端 MoE 图，但它自己写明了通信腿是**按 fp32 搬运的保守上界**
> （`docs/161-ep-fp8-e2e-moe.md:154`、`:167-168`、`:189`）：
>
> > 真机字节口径：`nrecv·H·1B`（dispatch FP8）= 30.4 MB/rank、`nrecv·H·2B`（combine BF16）
> > = 60.8 MB/rank —— 比本实验的 fp32 121.6 MB 各小 4×/2×。
>
> 但那个「1B / 2B」是**只看主数据**的粗口径，漏掉了 SF 与每 token 的元数据（top-k 索引、权重、
> 源 token 号、链表号）；也漏掉了 DeepEP 真正把一条 token 打成**一条连续记录**这件事。
> 本篇补上：① 用 **DeepEP 真实头文件**（`deep_ep/layout/ep/token.cuh`）算出权威的 per-token
> 字节账；② 在 8×H100 上按 DeepEP 参考实现（`deep_ep/utils/refs.py`）的**真实字段**逐字段量
> a2a；③ 量 dtype 从 fp32→bf16→fp8 的伸缩——得到一个反直觉结论：**缩小 dtype 在这些小/ragged
> 尺寸下几乎不省墙钟时间**。
>
> **可跑边界**：真实 per-token 布局是纯 host/device 算术（直接 include 真实头文件，H100 编跑）；
> a2a 用 `torch.distributed`（NCCL，架构无关）。Mega MoE 融合 kernel 本体（单 kernel 内 NVLink
> 直写 + warp 级重叠）是 **SM100 专属，仅分析**。目标卡代号敏感，全文用「目标卡」指代。
> 参考坐标：`~/github/DeepEP@93eb6eb`（v2.5.0）、`~/github/DeepGEMM@057ca59`。
> 实测见 `code/mega-moe/harness/exp_deepep_token_layout_bytes.{sh,out.txt}`、
> `harness/exp_ep_comm_bytes.{sh,out.txt}`、代码 `src/deepep_token_layout_bytes.cu`、
> `harness/exp_ep_comm_bytes.py`。

---

## 1. 缺口的精确形状

第 161 篇的通信预算是这样的（`docs/161-ep-fp8-e2e-moe.md:154-168`）：把 dispatch 和 combine 都
当成 `nrecv·H·4B`（fp32）搬运，得到每 rank 121.6 MB、各 ~4945 us。它随后**口头**修正为
「真机 dispatch 为 FP8/1B、combine 为 BF16/2B」，但没有量——因为那只是「主数据字节」，
把 SF 与逐 token 元数据都忽略了，也没解释为什么真 kernel 不会按 5 条独立 a2a 发。

一篇真正的字节口径要回答三件事：

1. **一条 token 到底多少字节？**（含 SF、元数据、对齐、mbarrier——这是 buffer 预留的权威口径）
2. **这些字节是怎么发出去的？**（DeepEP 的字段与 a2a 形态）
3. **字节减少能不能换成时间？**（dtype 从 fp32→fp8 省 4×，墙钟省多少？）

## 2. DeepEP 的真实 per-token 布局（权威字节账）

DeepEP 不为每种字段单独开一个 collective，而是把**一条 token 的全部内容打成一个连续记录**，
装在一个 buffer 里。布局在 `deep_ep/layout/ep/token.cuh:10-82` 的 `layout::TokenLayout`：

| 字段 | 定义 | 大小 |
|---|---|---|
| `num_hidden_bytes` | 主数据 | `hidden · elem_size` |
| `num_sf_bytes` | block-scale | `num_sf_packs · sizeof(sf_pack_t)` |
| `num_metadata_bytes` | 元数据 | `num_topk·(sizeof(int)+sizeof(float)) + (with_metadata ? (1+num_topk)·sizeof(int) : 0)` |

（`token.cuh:25-26`）。`sizeof(sf_pack_t)=4`（一个 union，装 4 个 UE8M0 或 1 个 fp32，
`common/compiled.cuh:59-62`），`kNumTMAAlignmentBytes=32`、`kNumRDMAAlignmentBytes=64`
（`compiled.cuh:67-68`）。整条记录的字节数（`token.cuh:32-41`）：

```
num_bytes = align_RDMA( align_TMA(hidden_bytes) + align_TMA(sf_bytes) + align_TMA(metadata_bytes) )
            + align_TMA(kWithMBarrier ? sizeof(mbarrier) : 0)
```

元数据拆开看（`get_topk_idx_ptr`/`get_topk_weights_ptr`/`get_src_token_global_idx_ptr`/
`get_linked_list_idx_ptr`，`token.cuh:63-76`）：`topk_idx`（每槽 1 个 int）→ `topk_weights`
（每槽 1 个 float）→ `src_token_global_idx`（1 个 int）→ `linked_list_idx`（`num_topk` 个 int）。

两个**入口函数**（定义在 include 了 `<nccl.h>` 的 host 头里，本实验按函数体原样重述）：

- named 派发 `csrc/kernels/ep/dispatch.hpp:26-29`：
  `TokenLayout(hidden·elem_size, num_sf_packs·sizeof(sf_pack_t), num_topk, /*with_metadata=*/true)`
- combine `csrc/kernels/ep/combine.hpp:19-22`：
  `TokenLayout(hidden·elem_size, 0, num_topk, /*with_metadata=*/false)`

`csrc/buffers/ep.hpp:256-282` 给出调用口径：dispatch 的 `elem_size = fp8?1:2`、
`num_sf_packs = ceil_div(hidden,32)`（源码注释写明「An approximation」），combine 用 bf16（2B）；
buffer 取 dispatch / combine 两者的**较大者**再对齐到 2 MB。

**H100 真机跑真实头文件**（`src/deepep_token_layout_bytes.cu` 直接 include `token.cuh` 编译，
`harness/exp_deepep_token_layout_bytes.out.txt`）：

| 场景（H=7168, topk=2） | data | sf | meta | **token 字节** |
|---|---|---|---|---|
| dispatch FP8 | 7168 | 896 | 28 | **8160** |
| dispatch BF16 | 14336 | 0 | 28 | **14432** |
| combine BF16 | 14336 | 0 | 16 | **14432** |

两个自检全 OK：`num_metadata_bytes=28`（=2·4+2·4+4+2·4）与 `get_num_bytes<false>=8128`
（= align₆₄(7168+896+align₃₂(28))）都与手算一致。其他形状：H=2048 dispatch FP8 = 2400 B；
H=512/topk=4 = 672 B；H=7168/topk=8 = 8224 B（元数据涨到 100 B）。

**两个值得记住的结论**：

- `BF16/FP8 dispatch = 1.7686 ≈ 2`（不是 4）——因为 SF 与元数据不随 dtype 缩小，被
  `ceil_div(hidden,32)·4B` 的 SF 近似**放大**：FP8 下 SF 占主记录 **896/8160 = 11.0%**。
  DeepEP 为 gran-32 的 scale 预留了「每 32 个元素一个 4 字节 sf_pack_t」的**过预留**
  （真实的 packed UE8M0 只要 1/4 的空间），这是源码注释里的 "approximation"。
- **combine 与 dispatch(BF16) 同字节**（14432），因为两者主数据都是 bf16、`hidden=7168` 时
  metadata（28 vs 16）在对齐后落进同一桶。

## 3. H100 实测：按 DeepEP 真实字段逐字段 a2a

`deep_ep/utils/refs.py:73-89` 是 dispatch 的参考实现，它把每条 token 的字段全部发出去：
`x`、`sf`、`topk_idx`、`topk_weights`、`src_token_idx`，且 `:100-110` 用 **5 条独立的
`all_to_all_single`** 发。`sf` 的形状由 `math.py:31-39` 的 `per_token_cast_to_fp8` 决定：
fp32，`[num_tokens, ceil_div(hidden,128)]`。

我们在 8×H100（`world=8, T_local=2048, H=7168, E=64, topk=2`，每 rank 发 4096 行、收 ~4031 行）
按这套字段逐条量 `all_to_all_single`（`harness/exp_ep_comm_bytes.py` Part A）：

| 字段 | 字节/rank | us | 备注 |
|---|---|---|---|
| x (fp8) | 29.36 MB | 4724.6 | 主数据 |
| sf (fp32 `[H/128]`) | 0.92 MB | 4637.4 | **字节 1/32，时间几乎一样** |
| topk_idx (int32) | 32.8 KB | 4631.8 | |
| topk_weights (fp32) | 32.8 KB | 4631.8 | |
| src_idx (int32) | 16.4 KB | 4631.1 | |
| **合计（5 条独立 a2a）** | 30.36 MB | **23256.7** | |
| **扁平一条 a2a** | 30.36 MB | **4784.0** | 6.3 GB/s |

**这张表是本篇最重要的发现**：五个字段的**字节数相差 1800 倍**（29.36 MB vs 16 KB），
但每条独立 a2a 都花 **~4630 us**——说明在这个 ragged + 受争用环境里，a2a 是**固定时延主导**，
不是带宽主导。于是「5 条独立 a2a」总共 23.3 ms，而把同样的字节**打平成一个 buffer 发一次**
只要 4.8 ms（**4.9×**）。这**正是 DeepEP 把一条 token 打成一条连续记录（第 2 节的
`TokenLayout`）的工程动机**：真 kernel 只有**一次** remote 传输，而不是 N 次 collective。
第 161 篇「5 条 a2a 各 121.6 MB」其实已经把 5 条合并成了「总字节」来报，本篇把「合并 = 4.9×」
量了出来。

combine 一侧（Part B）按 README 的契约（`README.md:279`「Combine inputs must be BF16」）：
把收到的 token 送回源 rank，`y` 是 bf16 `[nrecv, H]` = **57.79 MB/rank、4796.5 us（12.0 GB/s）**。

## 4. dtype 伸缩：4× 字节只换来 4.5% 时间

在同一份 ragged 形状（`[nsend, H]`）上扫 `fp32 / bf16 / fp8`（Part C）：

| dtype | 字节/rank | us | GB/s | t/t_fp32 |
|---|---|---|---|---|
| fp32 | 117.44 MB | 4946.4 | 23.7 | 1.000 |
| bf16 | 58.72 MB | 4799.3 | 12.2 | 0.970 |
| fp8 | 29.36 MB | 4724.9 | 6.2 | **0.955** |

**字节缩 4×（fp32→fp8），墙钟只省 4.5%**。原因与第 3 节同源：这个量级的 ragged a2a 被
**固定时延 + 争用**主导（fp32 与 fp8 的差值 221 us 只占 4946 us 的 4.5%），带宽完全没吃满
（fp8 才 6.2 GB/s）。作为对照，Part E 的**平衡负载锚**（世界大小×T×H 的连续张量、无 split）：
512 tok/rank 58.7 MB → 422 us（**139.1 GB/s**）、2048 tok/rank 234.9 MB → 2048 us
（**114.7 GB/s**）。也就是说：**到了几十 MB 的量级、且 split 平衡时，a2a 才是带宽受限的，
字节才值钱**；ragged 小消息下字节不值钱。

这与第 161 篇 / 第 104 篇的「ragged split 有 2-4× 惩罚」同向，但给出更细的图景：
**ragged 的惩罚不是「多搬了字节」，而是「每 peer 的小消息 + 固定时延」，所以 FP8 化在本机
这类 ragged 通信上收益有限。**

## 5. 校正的 EP 单步通信预算

把第 161 篇的 fp32 上界换成真实字段（Part D）：

| 段 | 真实字节/rank | 第 161 篇 fp32 | 倍数 |
|---|---|---|---|
| dispatch（x fp8 + sf + meta） | **30.36 MB** | 117.44 MB | **3.87×** |
| combine（y bf16） | **57.79 MB** | 115.58 MB | **2.00×** |
| **合计** | **88.15 MB** | 233.02 MB | **2.64×** |

dispatch 里 SF 占 3.0%（用参考实现口径的 `[H/128]` fp32），元数据 <0.3%——**主数据字节就是
大头**。88.15 MB 与第 144 篇（P134-1）独立量到的「dispatch fp8 `T·k·H·1B` + combine bf16
`T·k·H·2B` = 88.1 MB/rank」**逐 MB 对上**，互为交叉验证。

**但要用第 4 节的结论读它**：真实字节比第 161 篇低 2.64×，**不代表墙钟也低 2.64×**——
在本机的 ragged + 争用环境里，第 161 篇的 4945 us/条 与本篇的 fp8 4725 us 基本相当
（差 4.5%）。第 161 篇的 `233 MB → 9892 us` 之所以能缩水，主要靠「合并成更少的 collective」
（第 3 节的 4.9×），而不是靠 dtype。

## 6. 与既有篇章的关系与边界

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 161 篇（EP e2e + FP8 kernel） | 多卡 EP 图 + 真实 FP8 kernel，通信按 fp32 上界 | 换成 DeepEP 真实字段口径，逐字段量 |
| 第 104 篇（反向通信量） | 反向 ≈1.5× 前向、ragged 2-3× 惩罚 | 从「字节模型」推进到「真实记录布局 + dtype 伸缩」 |
| 第 144 篇（前向重叠） | fp8 dispatch + bf16 combine a2a，侧流不重叠 | 独立量到同一 88.1 MB/rank（交叉验证），拆开字段/布局 |
| 第 1 篇（a2a 锚） | 平衡负载 ~393 GB/s（无 NVLS 下不同） | 平衡 vs ragged 对照，指出小消息的时延主导 |

**能跑/能核**：真实 header per-token 布局（直接 include）；8×H100 字段级 a2a、dtype 伸缩、
平衡锚。

**仅分析**：Mega MoE 融合 kernel 的 NVLink 直写、warp 级通信-计算重叠、readiness 水位线
（`sm100_fp8_fp4_mega_moe.cuh`）——**SM100 专属**。本实验量的是**NCCL 通用 a2a**，与 DeepEP
自己 JIT 的 ragged kernel 也不是一回事（后者专为 ragged 优化、且能 `get_theoretical_num_sms`
按 SM 预算分配）。

**边界/注记**：① 本机 8 卡共享（GPU0 被另一自驱任务占满），节点处于**受争用**状态——
第 3/4 节的「固定时延 ~4.6 ms」是**受争用下界**，平衡锚 114-139 GB/s 与第 1/161 篇同量级，
说明争用程度相当；② `num_sf_packs=ceil_div(hidden,32)` 是 DeepEP 源码注释里的近似预留，
比真实 packed UE8M0 大 4×；③ 本实验按 `refs.py` 的「5 条独立 a2a」形态量，真 kernel 是
「单条记录、单次传输」（第 3 节已量出 4.9× 差距）。

## 7. 一句话结论

DeepEP 的真实 per-token 记录（`token.cuh`）在 DeepSeek 形状下是：**dispatch FP8 8160 B、
BF16 14432 B、combine BF16 14432 B**（FP8 下 SF 额外占 11%）；折算到 EP 单步是
**dispatch 30.4 MB + combine 57.8 MB = 88.2 MB/rank**（第 161 篇 fp32 上界 233 MB 的
**2.64×**）。但在本机 ragged + 争用环境下，**字节≠时间**：dtype 从 fp32 缩到 fp8（4× 字节）
只省 **4.5% 墙钟**，而把「5 条独立 a2a」合并成「一条连续记录」省 **4.9×**——这正是 DeepEP
把整条 token 打成连续布局的设计动机，也是 Mega MoE 把通信折进单 kernel 而非「发多条流」的
同一思路。

## 参考

- 本专题：第 161 篇（EP e2e + FP8 kernel）、第 144 篇（前向重叠，88.1 MB/rank 交叉验证）、
  第 104 篇（反向通信量）、第 1 篇（a2a 带宽锚）、第 2/3/9/10/11 篇（Mega MoE 单 kernel）。
- 源码：DeepEP `deep_ep/utils/refs.py:38-40/73-89/100-110`、`deep_ep/utils/math.py:31-39`、
  `deep_ep/layout/ep/token.cuh:10-82`、`deep_ep/include/deep_ep/common/compiled.cuh:59-70`、
  `csrc/kernels/ep/dispatch.hpp:26-29`、`csrc/kernels/ep/combine.hpp:19-22`、
  `csrc/buffers/ep.hpp:256-282`、`README.md:279`。
- 上游仓库：https://github.com/deepseek-ai/DeepEP 、 https://github.com/deepseek-ai/DeepGEMM 。
- 实验：`harness/exp_deepep_token_layout_bytes.{sh,out.txt}`（真实 header）、
  `harness/exp_ep_comm_bytes.{sh,py,out.txt}`（8×H100 a2a）。
