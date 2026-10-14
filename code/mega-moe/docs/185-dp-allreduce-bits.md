# 数据并行 all-reduce 的逐位一致性：真实 NCCL 下 ring/树/world-size 都改 bit

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P175-1** 增量（第 185 篇）。
> 第 184 篇（`docs/184`）量了 MoE 反向梯度累积的「切分 × 结合序」，最后写下
> 「**DP 的 ring-reduce（顺序）与 tree-reduce（树形）在 fp32 下给出不同 bit**」——但那只是
> **纯 torch 把 8 份偏梯度手动相加**的模拟，不是真实集合通信；第 170 篇（`docs/180`）也留了口子
> 「把结论推到真实多卡」。本篇把它补上：在 H100 上用 **真实 NCCL**（torch.distributed）做 DP 的
> 梯度 all-reduce，把「哪一层决定 bit」钉到具体配置。
>
> 一句话结论：**NCCL all-reduce 对固定配置是 run-to-run 逐位确定的（20 次 0 差），但换个
> 算法（Ring↔Tree）、换个 world size、或把 all_reduce 换成 ZeRO 的 reduce_scatter+all_gather，
> 都会改动 30%~36% 的 element bit——而幅度只有 ~3e-8（比 FP8 量化误差小 ~1e6 倍）。**
> 复现性要钉住的是 **(world size, 通信算法, 集合操作种类)**，不是 `NCCL_DETERMINISTIC`/`use_deterministic_algorithms`。
>
> 可跑性：纯 `torch.distributed(NCCL)`，**H100 真机实测**，与目标卡/SM100 无关。
> 全部数字来自 `harness/exp_dp_allreduce_bits.{py,sh,out.txt}`
> （H100 80GB HBM3, sm90, 驱动 570, PyTorch 2.12, NCCL 2.29.7）。
> 源码坐标：DeepEP commit `93eb6eb`。目标卡代号敏感：全文用「目标卡」。

---

## 1. 问题：从「手动模拟」到「真实通信」

第 170/184 篇（`docs/180`/`docs/184`）把 MoE 反向的**算子级**可复现性拆得很细：

- **单次 grouped GEMM 结构性 bit-确定**（只沿 M/N 切块、不切 K，K 在 CTA 内固定顺序累加）；
- **梯度累积的「切分 × 结合序」改 bit**——但这是**纯 torch** 把 `S` 份偏梯度按 forward/reverse/
  pairwise-tree/by-magnitude 相加得到的结论。

那一步留下的问题是：**真实的多卡数据并行（DP）梯度归约**走的是集合通信库（NCCL），它内部的
归约结构既不是简单的顺序链、也不一定是教科书的 pairwise tree。NCCL 的 all-reduce 实际上是
**reduce-scatter + all-gather**：每个 rank 对某个 chunk 按**环上接收顺序**（ring）或**二叉树**（tree）
累加，再把结果散出去。于是「哪一层决定 bit」需要真机量，而不是推断。

本篇只做一件事：**在 H100 上用真实 NCCL 把 all-reduce 的 bit 一致性拆开**。它不依赖任何
DeepGEMM kernel，是架构无关的通信层事实；对 Mega MoE 的意义在 §4（wgrad 留在专家本地 rank，
跨 rank 归约只发生在共享/稠密分支与优化器同步）。

## 2. 环境事实与实验设计

**环境事实（重要）**：本机 NVSwitch 的 **fabric manager 未启用**，NCCL 默认尝试的
**NVLS（NVLink SHARP）multicast 建不起来**，会在首次 all-reduce 报
`Failed to bind NVLink SHARP (NVLS) Multicast memory ... invalid argument`。因此实验设
`NCCL_NVLS_ENABLE=0`，让 NCCL 退到 **Ring / Tree** 两条算法（正文只对比这两条，不测 NVLS）。
这也解释了为什么本机默认算法命中 **Ring**（见 §3.3）。

**设计**：4×H100（`CUDA_VISIBLE_DEVICES=1,2,3,4`），每个 rank 拿一份**确定性**的 fp32 偏和
（`make_partial`，CPU 种子 `1000*gen+rank`；与 world size 无关，同一逻辑数据切成 `world` 份），
尺寸 `2^20`（=1M fp32 ≈ 4 MB，一个展平的梯度桶）。配置：

| 变量 | 取值 |
|---|---|
| world size | 2 / 4 |
| NCCL_ALGO | 默认 / `Ring` / `Tree` |
| NCCL_PROTO | `Simple`（+ w=2 额外 `LL`）|
| 集合操作 | `all_reduce`（Part A-E）、`reduce_scatter`+`all_gather`（Part D）|
| 开关 | `NCCL_DETERMINISTIC=0/1`（Part F）|

每个配置用独立进程组（`NCCL_ALGO` 在通信子初始化时读取，不能中途改，故每配置重新 spawn）。
每个 rank 的偏和由种子本地重建（**不需要 all_gather 收集**），rank 0 另做 fp64 真值与
手动归约序对照。

## 3. 结果

### 3.1 Part A：固定配置 run-to-run 逐位确定

对同一输入、同一配置连做 20 次 `all_reduce`，比较每次输出与第一次的逐位差：

| world | 算法/协议 | 20 次间最大逐位差 |
|---|---|---|
| 2 | 默认 | **0** |
| 2 | Ring/Simple、Tree/Simple、Ring/LL | **0** |
| 4 | 默认、Ring/Simple、Tree/Simple | **0** |

**结论**：NCCL all-reduce 在**固定 (world, 算法, 协议, 拓扑)** 下是**逐位确定**的。
梯度归约的不可复现**不是**「每次跑都抖」，而是「**换配置就换 bit**」——这正是第 170/184 篇
「可复现性 ≠ 非确定性」的通信版本。

### 3.2 Part B/C：world size 与算法都改 bit

与 fp64 真值比较（`ref.float()`）：

| world | 算法 | vs fp64 逐位差 | relerr |
|---|---|---|---|
| 2 | 默认 / Ring / Tree | **0 / 1048576** | 1.987e-08 |
| 4 | 默认 / Ring | **345555 / 1048576（33.0%）** | 3.461e-08 |
| 4 | Tree | **345068 / 1048576（32.9%）** | 3.456e-08 |

同 world、不同算法两两对比：

| 对比 | 逐位差 | relerr |
|---|---|---|
| w=2 默认 vs Ring vs Tree | **0（全同）** | 0 |
| **w=4 Ring vs Tree** | **373181 / 1048576（35.6%）** | **2.967e-08** |

**为什么 w=2 全同**：2 个 fp32 相加只有**一种**加法顺序（`a+b`），且单次 fp32 加是**正确舍入**的，
所以任何算法/顺序都得到同一个、也是正确舍入的结果（vs fp64 逐位 0 差）。**至少 3 个 rank**
才可能出现「不同结合序 → 不同 bit」——这与第 184 篇「结合序在 8 份相加时才显形」一脉相承。
**w=4 时 Ring 与 Tree 差 35.6%**：这就是第 184 篇那句「ring-reduce 与 tree-reduce 不同 bit」
在**真实 NCCL** 上的确认。

### 3.3 Part D：ZeRO 的 reduce_scatter+all_gather 与 all_reduce 不同 bit

ZeRO/Megatron 的 DP 梯度归约通常不用 `all_reduce`，而是 `reduce_scatter`（每 rank 只留自己那片
梯度）+ `all_gather`（把各片拼回）。把两者比较：

| world | 算法 | all_reduce vs rs+ag 逐位差 | relerr |
|---|---|---|---|
| 2 | 默认 / Ring | **0** | 0 |
| 4 | 默认 / Ring | **334136 / 1048576（31.9%）** | **2.646e-08** |

**结论**：在 w=4 下，**「all_reduce 版 DP」与「ZeRO 版 DP」给出不同 bit**。这解释了混合并行里
「开不加 ZeRO 的分片策略」会改变 master 权重 bit——不是精度问题，是**归约结构**问题。

### 3.4 Part E：真实 NCCL 命中了哪种手动序？

把 rank 0 的 4 份偏和用几种**手动 fp32 顺序**相加，与真实 NCCL 输出比：

| 算法 | vs manual_seq | vs manual_rot1 | vs manual_tree |
|---|---|---|---|
| w=4 默认/Ring | 336177（32.1%） | 335881（32.0%） | 367287（35.0%） |
| w=4 Tree | 458653（43.7%） | **246408（23.5%）** | 311619（29.7%） |
| 手动序之间 | seq vs tree: 311720（29.7%）; seq vs rot1: 442944（42.2%）| | |

**结论**：真实 NCCL 的 bit **不落在任何简单手动序上**——Ring 与顺序链/旋转/树都差 32%~35%，
Tree 算法虽然最接近「旋转顺序链」（23.5%）但仍不同。原因是 NCCL 的 all-reduce 是
**分 chunk 的 reduce-scatter + all-gather**，每个 chunk 由不同 rank 用不同环上顺序累加，
且 tree 算法是**二叉树 + 分块**，不是「对全部 rank 偏和做一次 pairwise tree」。这告诉我们：
**不能用一个简单模型去预测 bit，只能固定配置**。

### 3.5 Part F：`NCCL_DETERMINISTIC` 不改结果

| world | 对比 | 结果 |
|---|---|---|
| 2 | Ring/Simple DET=0 vs DET=1 | 逐位 **0 差** |
| 4 | Ring/Simple DET=0 vs DET=1 | 逐位 **0 差** |

**结论**：本机 NCCL 2.29.7 在固定算法上本就 run-to-run 确定，`NCCL_DETERMINISTIC=1` 没有额外
效果。它与第 170 篇的 `torch.use_deterministic_algorithms` 结论一致：**开关修不了「换配置就换
bit」**。

## 4. DeepEP 的两条跨-rank 归约通路（源代码事实）

NCCL 是 DP 稠密梯度的通路；在 **EP（专家并行）** 里，DeepEP 的跨-rank 归约是两条**不同**的路：

1. **EP combine 的 token 归约**（`deep_ep/include/deep_ep/impls/ep/combine_utils.cuh`）：
   按 top-k 槽位**固定顺序**遍历（`:133` 的 `for k in kNumValidTopk`），把各源 rank 的 bf16
   向量累加进 fp32 累加器 `reduced[]`（`:147-151`），最后 round 回 bf16（`:158-167`）；
   累加用 `ptx::accumulate`（`common/ptx.cuh:593-601`，SM100 上是 `add.rn.f32.bf16`）。
   —— **顺序确定、但精度是 bf16**：与 NCCL 的 fp32 是两种极端。
2. **EPLB 冗余专家梯度归约**（`.../impls/ep/reduce_grads.cuh:82`）：
   `ptx::tma_store_reduce_add_f32`（`common/ptx.cuh:148`）——**硬件 `red.add`**，
   累加顺序由硬件/到达时序决定，**契约级不保证 bit**。

把它们和本篇对照：**「跨 rank 归约不可复现」并不是 NCCL 独有**——任何用硬件原子/red 的归约
（EPLB）更不确定；DeepEP combine 则靠固定 top-k 序 + bf16 换来确定性。**Mega MoE 的 wgrad
刻意留在专家本地 rank**（`csrc/apis/gemm.hpp:373-427` 的 `k_grouped_*` 不带跨 rank 语义），
反向**没有**专家权重的 all-reduce；跨 rank 归约只出现在**共享/稠密分支**与**优化器同步**上。
所以本篇的结论对 MoE 训练的可复现性边界是：**专家权重梯度不受 DP 归约影响**，受影响的是
稠密/shared 权重——而它们的 bit 由 `(world size, NCCL_ALGO, all_reduce vs rs+ag)` 决定。

## 5. 结论与实践清单

1. **run-to-run 确定**：固定配置下 NCCL all-reduce 逐位确定（20 次 0 差）；不确定来自**改配置**。
2. **world size**：w=2 所有算法全同（只有一种加法序）；**w≥3 分化**，w=4 时 Ring/Tree 差
   35.6%、vs fp64 差 ~33%。
3. **算法**：`NCCL_ALGO` 从 Ring 换 Tree 改 35.6% bit（幅度 2.97e-8）。
4. **集合操作种类**：ZeRO 的 `reduce_scatter+all_gather` 与 `all_reduce` 在 w=4 差 31.9% bit。
5. **不可用简单模型预测**：真实 NCCL 不命中顺序链/旋转/pairwise-tree 任一种。
6. **开关无效**：`NCCL_DETERMINISTIC=1` / `use_deterministic_algorithms` 不改「换配置就换 bit」。
7. **幅度无害**：所有抖动 ≲ **3.7e-8** 相对，比 **FP8 量化误差（3.6e-2）小 ~1e6 倍**、比 bf16
   输出舍入也小——**训练无损，只影响逐位回归/调试**。
8. **复现性清单**（接第 184 篇）显式钉住：**(world size, NCCL_ALGO/协议, all_reduce vs rs+ag,
   grad_accum_steps, microbatch 顺序)**。EP 侧另补：combine 的 top-k 序（固定）与 EPLB red.add
   （不保证）。

一句话：**可复现性 ≠ 数值精度**——与第 170/184 篇同调，这一篇把「跨 rank」那一格用真实 NCCL 补齐。
