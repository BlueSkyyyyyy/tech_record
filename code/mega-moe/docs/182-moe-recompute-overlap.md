# 重算的 shadow 预算能不能兑现：真机并发下的 SM 预算分区

> 第 181 篇（P171-1）回答了一个问题：MoE 前向融合主 kernel **不存** `h`（linear1 预激活、SwiGLU 反向的输入），
> 反向只能**重算**一次 linear1 GEMM；并量出这份重算的**时间预算**只有「不依赖 `h` 的反向 GEMM」
> （shadow：dgrad L2 + wgrad L2）的 `0.47×/0.73×`。但当时明确写着：**那是预算，不是实测重叠**。
>
> 本篇把这份预算放到**真机并发**下去兑现，追问一个可核对的工程问题：**要满足什么条件，重算才能真正
> 藏进 shadow？** 答案是一条**负结果 + 一条量化解释**：单流串行之外，唯一能兑现的是「把 SM 分区给两个
> kernel」，而分区带来的净收益被「限核效率损失」吃掉了绝大部分（最好 `1.09×`）——所以 Mega MoE 才选择
> **单 kernel 融合 + SM 预算分区**，而不是「两个 kernel 两条 stream」。

**约束**：Mega MoE 融合主 kernel 是 **SM100 专属**，本机 H100(sm90) 跑不了；本文只取它**架构无关的调度
机制语义**（SM 预算分区、异步张量核、`smem` 占满一个 SM）。数值实验全部在 H100 真机、用真实 SM90 FP8
grouped kernel（DeepGEMM 2.8.1）完成。目标卡代号敏感：全文用「目标卡」。

---

## 1. 为什么「两条 stream」会退化成串行：co-residency 闸门

要重叠两个 GEMM，起码得让它们**同时占住 SM**。DeepGEMM 的 SM90 FP8 grouped/dense kernel 是
**persistent** 的：发射的 block 数就是 `num_sms`（`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:125`
`.grid_dim = dim3(config.launch_config.num_sms, 1, 1)`），每个 block 循环领任务直到把整个 GEMM 做完。
而每个 block 独占一个 SM 的共享内存预算：`smem_capacity = 232448`（`csrc/jit_kernels/heuristics/sm90.hpp:14`），
当 `block_k=128`、多 stage 且 FP8 时 `pipeline_config.smem_size` 会逼近这个上限（第 148 篇已量过消费者
block 因此无法 co-reside）。

于是默认（满 SM）发射时：第一个 kernel 的 `num_sms` 个 block **铺满全部 SM**，第二个 kernel 的 block
只能等有 SM 空出来——**两个 stream 变成了「排队」而不是「并发」**。

这与 SM 数无关，与「同时占用一块 SM 的两个 block 共享资源」有关。DeepGEMM 的 kernel 都是
「一个 block 吃满一个 SM」，所以「同 SM 上跑两个 block」几乎不可能。**重叠的唯一出路是把 SM 分成两组，
一组给一个 kernel**。

> 顺带一条与本节实验无关、但要记住的坑：`set_num_sms` 的合法性要求 `num_sms % 2 == 0`
> （`csrc/jit_kernels/heuristics/config.hpp:60`），所以本文的分区点都取偶数。

---

## 2. Part A：满 SM 双 stream == 单流串行（负结果）

`harness/exp_moe_recompute_overlap.py` Part A，`T=4096, H=7168, I=2048, E=8, topk=2`（`m_total=8192`,
`total_k=8192`，工作量为 DeepSeek 量级）：RECOMPUTE = 前向 L1 routed（`m_grouped_fp8_gemm_nt_contiguous`，
第 181 篇同款），SHADOW = dgrad L2（`m_grouped`）+ wgrad L2（`k_grouped`）。两 stream 并发执行与单流串行对拍（H100 真机）：

```
   RECOMPUTE only (满 132 SM):    356.7 us   1349 TFLOPS
   SHADOW①dgrad L2 (满 132 SM):    193.3 us   1244 TFLOPS
   SHADOW②wgrad L2 (满 132 SM):    535.9 us    449 TFLOPS
   SHADOW   合计    (满 132 SM):    729.3 us    660 TFLOPS   RECOMPUTE/SHADOW = 0.49x
   serial (rec; sh) 单流            :   1091.0 us
   concurrent (两 stream, 都满 SM)  :   1102.1 us
   ⇒ overlap_factor = (rec+sh)/ovl = 0.985x     （1.0=完全串行，1.49=完美）
```

- **并发几乎 = 串行**（`overlap_factor 0.985×`，甚至因两条 stream 的发射/事件开销略慢）。
  这是**结构性**的：两个 kernel 都想占满 132 个 SM，而每个 block 独吞一个 SM 的 `smem`，第二个只能等。
- 顺便坐实第 181 篇的分项：重算 `356.7 us`（与第 181 篇的 `357 us` 逐位一致）、wgrad L2 `535.9 us`
  （第 181 篇 `536 us`）——**重算是计算受限的**（1349 TFLOPS，接近 H100 FP8 上限），
  **wgrad 是访存受限的**（449 TFLOPS，第 15 篇 ncu 量到 SM 占用仅 ~12%）。

---

## 3. Part B/C：给两个 kernel 分区 SM，重叠才成立

把 `num_sms` 当作「给这个 kernel 的 SM 预算」：`set_num_sms(N)` 后发射的 kernel 只有 `N` 个 persistent
block。于是**两个 stream 各发一个被限核的 kernel**，`2N ≤ 132` 时它们落在**不同** SM 上，真正并行。

> 实现要点：`set_num_sms` 是 host 端全局开关，但每个 kernel 在**发射那一刻**就把它烘进了 grid 维与
> 编译键（`sm90_fp8_gemm_1d2d.hpp:125` 的 grid_dim 与 JIT 的 block config），所以「先 `set_num_sms(n_rec)`
> 发 rec、再 `set_num_sms(n_sh)` 发 sh」可以给两个并发 kernel **不同**的预算；launch 走当前 stream
> （`third-party/deep_jit/include/deep_jit/backend/cuda/kernel.hpp:164-166`：默认
> `at::cuda::getCurrentCUDAStream()`），故 `torch.cuda.stream(...)` 上下文能分别路由。

### 3.1 对称分区（两个 kernel 同一 `num_sms=N`）

```
      N   rec us    sh us  serial rec+sh  concurrent  ideal=max  ovf=(rec+sh)/ovl    spd vs 满SM串行
    132    362.4    741.1         1103.5      1116.1      741.1              0.99            0.97
     96    486.2    825.9         1312.1      1269.4      825.9              1.03            0.86
     66    642.0   1020.8         1662.8      1108.1     1020.8              1.50            0.98
     44    940.5   1323.4         2263.9      1388.5     1323.4              1.63            0.78
     34   1088.4   1565.7         2654.1      1626.5     1565.7              1.63            0.67
```

- `N=66`（严格平分 132）时 `overlap_factor = 1.50×`，**几乎等于理想 `1.49×`**——两个 kernel 真的并行起来了。
- **但净收益没了**：`spd vs 满 SM 串行 = 0.98×`。因为把每个 kernel 限到一半 SM，它自己就慢了近一倍
  （`rec 362→642 us`、`sh 741→1021 us`），省下的重叠时间被限核损失抵消。
- 结论：**分区兑现了「重叠」，但没有兑现「更快」**。重叠率随 N 变小升到 `1.63×`，而净速度反而掉到 `0.67–0.78×`。

### 3.2 非对称分区（`n_rec + n_sh = 132`，按代价配预算）

重算只占 shadow 的 `0.49×`，平分 SM 是浪费——给 shadow 更多、给重算刚好够：

```
    (n_rec,n_sh)   rec us    sh us  concurrent  ideal=max  ovf=(rec+sh)/ovl    spd vs 满SM串行
      (132, 132)    358.5    725.3      1099.1      725.3              0.99            0.99
        (66, 66)    671.0   1041.0      1115.6     1041.0              1.53            0.97
        (44, 88)    905.1    800.8      1139.0      905.1              1.50            0.95
        (34, 98)   1177.6    834.6      1443.1     1177.6              1.39            0.75
        (48, 84)    800.8    819.9      1104.0      819.9              1.47            0.98
        (56, 76)    775.2    883.9       993.1      883.9              1.67            1.09
       (24, 108)   1666.3    789.0      1863.1     1666.3              1.32            0.58
```

- **最优切分 `(56, 76)`**：并发 `993 us`，比满 SM 串行 `1086 us` 快 **`1.09×`**。这是本实验能达到的
  最好结果——而它离「完美重叠」的 `t_rec(132)+t_sh(132) → max(t_rec,t_sh)` 所隐含的 **`1.49×`** 差很远。
- `(24,108)` 反而更慢：给重算的 SM 太少，它自己变成瓶颈（`1666 us`），重叠再完美也没用。

---

## 4. Part D：重叠不改正确性（逐位一致）

分区只改「用几个 SM」，不改累加顺序；grouped GEMM 本身结构性 bit 确定（第 180 篇）：
`scheduler/gemm.cuh:87-104` 只沿 M/N 切块（`num_blocks = num_m_blocks * num_n_blocks`），
K 在 CTA 内按固定顺序累加。所以并发与串行输出**逐位一致**（H100 真机）：

```
   (132,132) / (66,66) / (44,88) / (34,98) / (56,76)：
     recompute h: equal=True max|d|/max|r|=0.00e+00
     dgrad dA   : equal=True max|d|/max|r|=0.00e+00
     wgrad dW2  : equal=True max|d|/max|r|=0.00e+00
```

即「用 SM 分区换重叠」**在数值上是免费的**（代价只在速度上）。

> 一个小坑（实验实现）：`k_grouped` wgrad 是 **`D = C + AᵀB`** 的累积型（`csrc/apis/gemm.hpp`，
> 第 12/15 篇），所以对拍时累加器必须清零、且并发/串行两条路径必须**执行相同次数**——否则输出会差
> `(runs-1)×`。修正前并发路径的 3 次 warmup 让 `dW2` 恰好是串行的 4 倍（`max|d|/max|r|=3.0`），
> 正是这个数字坐实了原因。

---

## 5. 为什么净收益只有 `1.09×`：分区兑现重叠，但没兑现「闲资源」

重叠率能到 `1.5–1.67×`（near-perfect），净速度却只 `1.09×`，这中间差的是什么？

**两个独立 kernel 的 SM 分区，本质是把「总吞吐」切成两半再并发，而不是让重算去用 shadow 闲着的资源。**
而 shadow 恰恰**有大量闲资源**：wgrad L2 `449 TFLOPS`、`SM` 占用仅 ~12%（第 15 篇 ncu）——它的张量核
大部分时间在空转，瓶颈在访存。理想情况下，重算的 MMA 应该跑在这些空转的张量核上、**完全不占额外 SM**，
那样才能拿到完整的 `1.49×`。

但「两个 persistent kernel 各占一组 SM」做不到这件事：它只是把 132 个 SM 分给两拨 block，
张量核仍按 SM 归属；限核后每个 kernel 的每-SM 效率还下降了。

**这正是 Mega MoE 选择「单 kernel + warp specialization + SM/寄存器预算」的量化理由**：只有把重算和
shadow 放进**同一个 kernel 的不同 warp**、让访存受限的那条腿把张量核让出来（异步张量核 UMMA/WGMMA，
第 167 篇证 CUDA-core 腿只重叠 30–60%、换 WGMMA 后 0.62→0.93），重算才能既不走额外 SM、又不牺牲自身效率。
Mega MoE 调度器正是这么做的：

- SM 按 **locality domain** 分成 `kNumLocalityDomains` 份，任务计数按域独立
  （`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:345-360` 的
  `get_next_local_task_idx` 用 `task_count_ptr + locality_domain_idx`），只有「波次够多」时才允许跨域偷
  （`:364-369`，`kNumMinStealWaves=2`，`:225`）——这与本文的「SM 分区」同源；
- 但它的通信/计算 warp 共享同一批 SM，靠 `setmaxnreg` 与异步 UMMA 把张量核与访存腿错开
  （第 167 篇的原型量到：把 CUDA-core 计算腿换成真实 WGMMA 后重叠效率 `0.62 → 0.93`）。

**一句话**：第 181 篇的 `1.49×` 预算是真的，但它**不是**靠「两个 kernel + 两条 stream」兑现的
（最好 `1.09×`），而是靠**单 kernel 内让计算腿用异步张量核去吃访存腿的闲张量核**兑现的——
这恰是 Mega MoE 融合而非分立的根本原因。

---

## 6. 与前面各篇的关系

- **第 181 篇**给了重算的**预算**（`0.47×/0.73×`）并明确「预算 ≠ 实测重叠」；本篇把它放到真机验证：
  满 SM 双 stream **完全不重叠**（`0.985×`），分区后重叠率 `1.5–1.67×` 但净收益仅 `1.09×`。
- **第 148 篇**证侧流 overlap 被 `232448 B/SM` smem 堵死；本篇把这把「堵死」量成了
  **`overlap_factor=0.985×`（=串行）** 的硬数字。
- **第 167 篇**证单 kernel 融合要计算腿是**异步张量核**；本篇从「预算兑现」一侧补了它的必要条件：
  光靠 SM 分区不够，必须让计算腿吃访存腿的闲张量核。
- **第 180 篇**证 grouped GEMM 结构性 bit 确定；本篇的 Part D 直接复用它：分区前后输出逐位一致。
- **第 110/34 篇**拆过 Mega MoE 调度器的 locality domain 分域与工作窃取；本篇说明它**为什么**要分域
  （同一份「SM 预算分区」动机，但落在单 kernel 内部）。

---

## 7. 结论 / 实务准则

1. **满 SM 双 stream 无法重叠**：`overlap_factor = 0.985×`（≈串行）。DeepGEMM persistent kernel
   一 block 吃满一 SM 的 `smem`（`232448 B`），第二个 kernel 只能排队。
2. **SM 分区能让重叠成立**：`num_sms=N`、`2N≤132` 时两 kernel 各占一组 SM，重叠率 `1.5–1.67×`
   （对称 `N=66` 时 `1.50×` ≈ 理想 `1.49×`）。
3. **但分区不是免费的**：限核使每 kernel 变慢，**净速度最好只有 `1.09×`**（非对称最优 `(56,76)`），
   远达不到「预算」隐含的 `1.49×`。收益上限由「限核效率损失」封顶。
4. **分区不影响正确性**：并发/串行输出逐位一致（依赖第 180 篇的 grouped GEMM 结构性确定性）。
5. **正确的兑现方式是单 kernel 融合**：让重算的异步张量核去吃 shadow（访存受限、张量核 ~12% 占用）闲着的
   张量核，而不是把 SM 切两半。这就是 Mega MoE 单 mega-kernel + locality domain + `setmaxnreg` 的量化动机。

---

## 8. 边界与「仅分析」

- **SM100 专属、仅分析**：Mega MoE 融合主 kernel（`sm100_fp8_fp4_mega_moe.cuh`）的 UMMA/tcgen05/TMEM、
  以及它把通信/计算 warp 融进同一批 SM 的机制是 SM100 专属，本机 H100 跑不了；本文只取其
  **架构无关的调度语义**（SM 预算分区、异步张量核、smem 占满一个 SM）。
- **数值实验全部 H100 真机**：真实 SM90 `m_grouped_fp8_gemm_nt_contiguous` / `k_grouped_..._nt_contiguous`，
  分区粒度即 `set_num_sms`（`csrc/apis/config.hpp` / `csrc/runtime/runtime.hpp:78-91`）。
- **环境注记**：8 卡节点共享。本实验绑到空闲 GPU1；RECOMPUTE `356.7 us` 与第 181 篇独立测得的 `357 us`
  逐位吻合，说明本次测量无争用。`set_num_sms` 因 `is_cublaslt_available() == (num_sms == device_num_sms)`
  （`csrc/runtime/runtime.hpp:89-91`）会影响稠密回退与否——但本实验的四个 GEMM 全是 grouped（无稠密），
  故不受该分派影响。
- 目标卡代号敏感：全文用「目标卡」指代。

**实验**：`harness/exp_moe_recompute_overlap.{py,sh,out.txt}`。
