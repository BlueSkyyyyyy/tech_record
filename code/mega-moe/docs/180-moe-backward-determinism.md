# MoE 反向算子链的可复现性边界：kernel 是逐位确定的，端到端不是——`use_deterministic_algorithms` 够不够

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P170-1** 增量（第 180 篇）。
> 前面第 12/13 篇拆了 wgrad/dgrad 的**实现**、第 14/99/111 篇量了**数值误差**、第 48 篇拆了
> `use_deterministic_algorithms` 这一开关的**三个落点**、第 62 篇拆了 DeepEP 的
> `EPHandle.deterministic_sort`。但它们各讲一半：**「哪些算子逐位确定、哪里才是真正的不确定源、
> 想把一次 MoE 训练步做到逐位可复现到底要做到哪一层」从来没被放到同一张图里量过。**
>
> 本篇补上。用 H100(sm90) 上真实可跑的 DeepGEMM grouped kernel，把反向算子链拆成三层量：
> **① grouped GEMM 本体**（结构性 bit-确定）、**② 稠密 cuBLASLt 回退**（唯一带契约级不确定的
> GEMM 路径）、**③ 谁来给 wgrad 排 token**（dispatch，真正的端到端风险）。核心结论一句话：
>
> **反向的 grouped GEMM 无论开关怎么打、`num_sms` 怎么改，都逐位可复现（20 次 0 差）；
> `use_deterministic_algorithms` 只关掉「稠密回退 + Mega Gate 的 split-K + mHC 的 splits」；
> 而让 MoE 训练**逐位不可复现**的真凶在 GEMM 之外——dispatch 用 `atomicAdd` 分配槽位，
> 决定了喂给 wgrad 的 token 顺序，而 wgrad 的收缩轴恰是 token。**
>
> 可跑性：反向四个 grouped GEMM 的 SM90 实现在 H100 真机可跑（第 8 篇已确认）；
> 主 Mega MoE kernel 是 **SM100 专属、仅分析**。全部数字来自
> `harness/exp_moe_backward_determinism.{py,sh,out.txt}`（H100 80GB HBM3, sm90, PyTorch 2.12, DeepGEMM 2.8.1）。
>
> 源码坐标：DeepGEMM 公开版 commit `057ca59`（v2.8.1）；DeepEP 公开版 commit `93eb6eb`（v2.5）。
> 目标卡代号敏感：全文用「目标卡」。

---

## 1. 问题：MoE 训练的可复现性，该在哪一层下手

MoE 训练的调试/回归非常依赖「逐位可复现」：同一份输入、同一个并行拓扑，两跑必须给出同一串
bit。DeepGEMM 为此提供了一个显式开关 `deep_gemm.use_deterministic_algorithms(bool)`
（`README.md:153`）。第 48 篇已经把这个开关的**语义**讲清楚——它在三个地方分别是不同含义：

1. **关掉 cuBLASLt 回退**（`csrc/apis/gemm.hpp:519-522`、`csrc/apis/einsum.hpp:81/115`）：

   ```
   :519  if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()
   :520      and epilogue_alpha.has_value()) {
   :521      cublaslt_gemm(...); return;
   :522  }
   ```

2. **禁止 Mega Gate 的 split-K**（`csrc/jit_kernels/heuristics/mega_gate.hpp:110-126`）——
   注释很直白：`deterministic mode is also batch invariant, so K is never split`（`:110`），
   于是 `num_split_k` 上界从 `kNumMaxSplitK` 退化成 `1`（`:126`）。
3. **钉死 Mega mHC 的 splits**（`csrc/apis/mega_mhc.hpp:15-16`）。
   开关本身只写在 `csrc/jit_kernels/heuristics/runtime.hpp:14-26`（默认 `false`），
   由 `csrc/apis/config.hpp:33-35` 暴露到 Python。

但「开关做了什么」不等于「不做会怎样」。有三个从没被量过的问题：

- **反向的四个 grouped GEMM**（wgrad `k_grouped_*`、dgrad `m_grouped_*`）到底确不确定？
  开关对它们有效吗？改 `num_sms` 会变吗？
- 反向里**哪条 GEMM 路径**才是开关真正要修的？
- 就算所有 GEMM 都逐位确定，**一次完整的 MoE 训练步**能逐位复现吗？如果不能，卡在哪？

本篇把这三问在 H100 上量出来。

---

## 2. 源码：为什么 grouped 反向 GEMM 是「结构性确定」的

关键不在开关，而在**调度器**。DeepGEMM 所有 GEMM（含 grouped）共用 `Scheduler` 模板
（`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`）。它的任务分解**只沿 M/N 切块、从不切 K**：

```
scheduler/gemm.cuh:91    num_blocks = num_m_blocks * num_n_blocks;          // Normal
scheduler/gemm.cuh:93    num_blocks = num_m_blocks * num_n_blocks;          // MGrouped(Contiguous/Masked)
scheduler/gemm.cuh:102   num_blocks = num_m_blocks * num_n_blocks;          // KGroupedContiguous
```

即**每个输出 tile `(m_block, n_block)` 恰好由一个 CTA 负责**，没有跨 CTA 的部分和、没有
`atomicAdd` 归并。K 维的累积发生在 CTA **内部**、按固定的 K-block 顺序推进：
`scheduler/gemm.cuh:64-79` 的 `get_next_k_group()` 逐组把物理 K 起点往前挪，
kernel 里 `sm90_fp8_gemm_1d1d.cuh:189` 算 `num_k_blocks`、`:220` 的
`for (k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx)` **顺序**把每个 K-block 累进
TMEM/寄存器累加器。可选的「累积写回」（SM90 grouped 强制 `D += C`）也走**同一个 CTA** 内的
`SM90_TMA_REDUCE_ADD_2D`（`sm90_fp8_gemm_1d1d.cuh:346`），同样固定顺序。

对照：DeepGEMM 里**唯一**用 `atomicAdd` 做跨分片归约的自家 kernel 是 einsum 的
`bmk,bnk->mn`（`deep_gemm/include/deep_gemm/impls/sm90_bmk_bnk_mn.cuh:167/171`，
第 17 篇量到 split 时 20 次跑出 ~5 万处 bit 抖动）。**MoE 反向的 grouped GEMM 不在其列。**

结论先给出：**grouped 反向 GEMM 是「结构性的 bit-确定」——与开关、与 `num_sms`、与运行次数都无关。**
下面用真机坐实。

---

## 3. Part A：真实 grouped 反向 GEMM 20 次逐位 0 差

在 H100 上真跑 DeepGEMM 2.8.1 的 `k_grouped_fp8_gemm_nt_contiguous`（wgrad 形状，G=8、
`host_ks=[1024,768,640,512,896,384,768,512]`、m=512、n=1024）与
`m_grouped_fp8_gemm_nt_contiguous`（dgrad 形状，G=8、n=256、k=512、m≈1280），
每种同一份输入连发 20 次，比较是否逐位相同。实测（`exp_moe_backward_determinism.out.txt` §A）：

| 情形 | k_grouped(wgrad) 20 次逐位差 | m_grouped(dgrad) 20 次逐位差 |
|---|---|---|
| `use_deterministic_algorithms(True)`  | **0** | **0** |
| `use_deterministic_algorithms(False)` | **0** | **0** |
| `num_sms = 60 / 80 / 100 / 132`（对照 132 基线） | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |

三点值得强调：

- **开关 True/False 对 grouped 反向 GEMM 毫无影响**——因为它既没有 cuBLASLt 回退
  （grouped 只走自研 kernel），调度器也不切 K。第 48 篇说的三个落点里，没有一个落在 grouped 上。
- **改 `num_sms` 不影响结果**——因为输出 tile 与 CTA 的对应是「无共享」的：每块输出只被算一次，
  换 SM 数只改变「哪块给哪个 CTA」，不改任何一块内部的加法顺序（这正是 batch invariance 的机器级来源）。
- **`batch invariance` 在 grouped GEMM 层是免费的**：同一条 token 在不同 batch/组形状下，
  只要它落进同一组、同一 tile，输出 bit 一致。

---

## 4. Part B：唯一带「契约级不确定」的 GEMM 路径是稠密 cuBLASLt 回退

那开关到底修的是什么？答案是**稠密路径的 cuBLASLt 回退**。用 `bench_kineto` 按 kernel 名拆
（`m=n=512, k=32768, bf16`）：

```
det=True  (自研 kernel, 命中 "gemm"):     77.8 us
det=False (cuBLASLt 回退): nvjet= 38.3 us + reduce= 2.5 us   # reduce>0 ⇒ split-K 归约被发射
自研/cuBLASLt 时间比 = 1.907x
```

- `det=True`：走自研 `sm90_bf16_gemm`（`csrc/apis/gemm.hpp:528-531` 的 `arch_major == 9` 分支），
  kernel 名命中 `gemm`；
- `det=False`：命中 `nvjet` + `reduce`——**cuBLASLt 的 split-K 归约 kernel 真的被发射了**
  （`reduce` 时间 > 0 是直接证据）。这正是 `gemm.hpp:519` 那条 gate 放行的结果。
  代价是自研 kernel 在这组形状上比 cuBLASLt **慢 1.9×**（split-K 的并行度换来的）——这就是
  「确定性 vs 吞吐」的显式取舍。

一个诚实的负面结果：**这 10 次里 `det=False` 的输出也逐位相同**。cuBLASLt 的 split-K 归约
在**契约上**不保证顺序（cutlass 的 stream-K 调度器自己注释
`deep_gemm/include/cutlass/gemm/kernel/tile_scheduler_detail.hpp:66` 把
`Nondeterministic` 列为一个显式模式），但在**这一组形状 + 这张卡**上，实际归约顺序稳定。
所以「开关防止的是**可能发生的**不确定」，不是「每次都能观测到的抖动」——要观测抖动得靠
分片数极大的形状（如第 17 篇的 `bmk_bnk_mn`，split=993）。**这点必须说清楚，否则会误以为
det=False 必然抖动。**

---

## 5. Part C：收缩轴决定谁对 token 序敏感——wgrad 敏感、dgrad 不敏感

这是 MoE 反向最容易被忽略的一点。看两个算子的收缩轴：

| 梯度 | 数学式 | 收缩轴 | 不等的轴 |
|---|---|---|---|
| wgrad `dW` | `dW_e = A_e^T B_e` | **token（K）** | 收缩维是 token |
| dgrad `dX` | `dX_t = dY_t W_e^T` | 隐藏维（K） | token 在 **M** 上、每行独立 |

**dgrad 每行只看自己那行**：实测把 `m_grouped` 每组内部的 token 行做随机置换，再按行对齐后比较，
**逐位差 = 0 / 327680**（`out.txt` §C2）。dgrad 在机器层面就是 batch-invariant 的。

**wgrad 是沿 token 的求和**：置换 token 顺序在**精确实数**下不改变 `dW_e`（加法交换律），
但会改变 fp32 的累加步长顺序 → bit 变。用固定累加序的参考（按 BLOCK_K=128 分块、顺序累加，
模拟 kernel 的 K-block 顺序）实测：

```
固定累加序参考自身 10 次逐位差 = 0
wgrad token 序样本 0: 逐位差=217862/262144, relerr=1.554e-07
wgrad token 序样本 1: 逐位差=218022/262144, relerr=1.554e-07
...
5 个 token 序: 逐位差 min=217862 max=218104；relerr ~1e-7
```

即：**约 83% 的元素 bit 变了，但相对误差只有 ~1.5e-7**——训练数值上完全无害，但**逐位可复现
失败**。这就是「同一份数据、同一个 kernel、跑两次结果不同」的数学根因，而且它**不是 kernel 的锅**：
kernel 是确定的，变的是「喂进来的 token 顺序」。而参考本身 10 次逐位差 = 0，说明变的是顺序、
不是随机数。

---

## 6. Part D：MoE 层级的真凶——dispatch 的原子槽位分配

那么「谁在给 wgrad 排 token」？是 **dispatch**。DeepEP 把 token 送进目标 rank 的对称缓冲槽位时，
槽位号来自一个**对称内存里的原子计数器**（`deep_ep/include/deep_ep/impls/ep/dispatch.cuh:341`）：

```
:341  stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter()
                                       + stored_dst_rank_idx, 1);
```

`atomicAdd` 的**返回顺序取决于 warp/CTA 的调度时序**：同一批 token、同样的 top-k，两次运行可能落进
不同的物理槽位。于是接收端看到的 token 行序在多次运行间可能不同 → 喂给 wgrad 的 token 顺序不同 →
按 §5，**wgrad 逐位不同**。把 §5 的置换放到 MoE 语境重复 5 次随机行序（`out.txt` §D）：

```
dispatch 行序随机（5 样本）：wgrad 逐位差 max=218210/262144, relerr max=1.560e-07
对照：逐 token 量（dgrad 类，行独立）在乱序下逐位不变 = True
```

**关键对照**：同样乱序下，逐 token 的量（dgrad 那一类、行独立）**逐位不变**。这精确对应
§5 的收缩轴差异——**乱序只伤「沿 token 求和的 wgrad」，不伤 dgrad**。

DeepEP 的对策正是第 62 篇拆过的 `EPHandle.deterministic_sort`
（`deep_ep/buffers/ep.py:113-199`）：dispatch 结束后用纯 Python 的
`torch.sort`（`:146`）/ `index_select` / `scatter_add_`（`:197`）把接收行序**规范回源 token 全局索引序**
（非 expand 全局排序、expand 用 `expert*1e10+src` 双键稳定排序 `:195-199`）。置上它之后，
wgrad 的 token 顺序才与硬件时序解耦、端到端才能逐位复现。**这才是 MoE 训练可复现性的真正开关**，
而它不在 GEMM 层。

（对照：Mega MoE 主 kernel 的 combine 段对 top-k 贡献做 fp32 累加是在一个固定循环里
`ptx::accumulate`（`sm100_fp8_fp4_mega_moe.cuh:1473-1499`），顺序由槽位内容而非时序决定；
主 kernel SM100 专属，此处仅分析。）

---

## 7. 结论：想逐位复现一次 MoE 训练步，该动哪里

把三层结论并排：

| 层 | 是否逐位确定 | 由谁决定 | 需要 `use_deterministic_algorithms` 吗 |
|---|---|---|---|
| grouped 反向 GEMM（wgrad/dgrad） | **是**（20 次 0 差、`num_sms` 无关） | 调度器不切 K、无跨 CTA 归约 | **不需要**（开关对它们无落点） |
| 稠密 GEMM（含 MoE 里的共享专家等） | `det=True` 时是；`det=False` 走 cuBLASLt split-K | 开关 + 形状 | **需要**（这是开关的主要用途） |
| Mega Gate / mHC | `det=True` 钉死 split/splits | 开关 | **需要** |
| **dispatch 的 token 行序** | `deterministic=True` 时是 | **DeepEP `deterministic_sort`**，不是 GEMM 开关 | **不需要，但必须开 DeepEP 那一侧** |

**三个可操作的结论：**

1. **别在 grouped GEMM 上花心思**：wgrad/dgrad 天然逐位确定，`use_deterministic_algorithms`
   对它们不产生任何效果（打开只会在稠密路径上命中自研 kernel、掉 1.9× 吞吐）。
2. **`use_deterministic_algorithms` 的收益边界很窄**：它修的是稠密回退 + Mega Gate/mHC 的
   split-K——对**纯 grouped MoE 反向**几乎没有作用。而且它是「契约上防不确定」，不保证每次
   都能观测到差异（Part B 的 10 次全稳）。
3. **端到端可复现的真正杠杆是通信编排**：DeepEP 的 `deterministic_sort` 把 dispatch 的行序
   规范回源索引序。**只关 GEMM 开关、不开 `deterministic` dispatch，MoE 训练依旧逐位不可复现**；
   反过来，开了 `deterministic` dispatch，wgrad 的 token 顺序就固定，wgrad 才逐位可复现。
   抖动幅度 ~1.5e-7（训练无害）——**可复现性与数值精度是两件事**。

一句话：**Mega MoE 反向算子链的「不确定」不在算子实现里，而在「谁按什么顺序把 token 交给算子」里。**

---

## 8. 可继续（均为增量、非承诺）

- 在真实 DeepEP `EPBuffer.dispatch` 上端到端复现这条链路（需 NCCL ≥2.32.3，见「阻塞」）：
  开/关 `deterministic` 各跑一次完整前向+反向，量 wgrad 的 0 差与非 0 差。
- 把 §5 的「收缩轴」结论在 SM100 的 `k_grouped_tn`（MN-major，仅分析）上做同一推演。
- 给 `harness/exp_moe_backward_determinism.py` 增加一个「多 microbatch 梯度累加」维度，
  量梯度累积的**结合序**（而非仅 token 序）对 master 权重 bit 的影响。
