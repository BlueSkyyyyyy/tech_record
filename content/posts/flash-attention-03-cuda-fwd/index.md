---
title: "Flash Attention 精读（三）：FlashAttention-2/3 的 CUDA 前向实现"
date: 2026-08-28
draft: false
weight: 3
series: ["flash-attention"]
tags: ["flash-attention", "cuda", "cutlass", "hopper", "attention", "系列"]
categories: ["算子开发"]
---

[第 2 篇]({{< relref "flash-attention-02-triton-fwd" >}})的 Triton 教程版把算法讲透了，但生产环境跑的是 CUDA。本篇精读官方仓库的两套前向实现：**FA2**（`csrc/flash_attn/src/flash_fwd_kernel.h`，Ampere/Hopper 通用）和 **FA3**（`hopper/`，Hopper 专属）。数学完全同前两篇，本篇要看的是：**同一份递推式，在"没有编译器帮忙"的层面如何被榨到硬件极限**。

一个总纲先立在此处，读完回来再看会更有感觉：

> FA2 的性能来自三件套——online softmax 消灭 HBM 中间矩阵、P fragment 零拷贝复用、1.5 级 cp.async 流水；**所有 warp 干同样的活**。FA3 把 attention kernel 彻底 GEMM 化：TMA 把加载从线程剥离、GMMA 让 smem 直读、warp specialization 让加载/GEMM/softmax 三类指令真正并行，再用 persistent 调度榨 L2。分水岭是"对称分工"到"角色分工"。

## 1. FA2：`flash_fwd_kernel.h`

### 1.1 结构：grid 映射与 tile 尺寸

grid 是 `(m_block, batch, head)`——一个 thread block 拥有一个 Q tile，对整条 KV 序列做 attention（`compute_attn` L1083–1099）。tile 尺寸在 launch template 里按 head_dim 硬编码（hdim64 → `128×128`，hdim128 → `128×64`；sm86/89 非因果用 `128×32` 以便一个 SM 驻留 2 个 CTA）。hdim256 更有意思：**运行时用 `cudaDeviceGetAttribute` 探测实际 smem 上限**再选 `128×64`（128KB）或 `64×64`（96KB）——tile 尺寸本质上是贴着 smem 上限反推出来的。

所有 warp 一起做 QK^T，也一起做 PV——没有 warp 间分工，靠 MMA atom 布局切分：`Layout<Shape<Int<kNWarps>,_1,_1>>` + `Tile<Int<16*kNWarps>,_16,_16>`，每 warp 负责相邻的 16 行。

### 1.2 KV 循环：反向遍历与两段式

循环边界三层收缩（L90–98）：`n_block_max = ceil_div(seqlen_k, kBlockN)`；causal 时裁到对角线为止——**对角线以下的整个 KV tile 不进循环**；局部窗口再从左侧裁。整个 CTA 若无可做块则提前退出写 0。

两个耐人寻味的细节：

**反向遍历 KV**（从最后一个 block 往前，L138–140 注释）：最后一个 block 是唯一需要 gmem 越界 predicate 的 block，反向遍历把它放在最前面（predicate 指令集中一次），还省掉一个寄存器。

**两段式循环摊薄 mask**（L297–382）：只有对角带附近的 `n_masking_steps` 个迭代走带 mask 的循环体，其余迭代 `apply_mask<Causal_mask=false>` 被编译成空操作（`Need_masking` 常量折叠）——与 Triton 教程 off-band/on-band 同一思想，但这里是**同一个循环拆两段**而不是两次函数调用。

### 1.3 softmax 状态机（`softmax.h`）

`Softmax<2 * size<1>(acc_o)>` 持有 `row_max`/`row_sum` 两个寄存器 fragment。每个 KV tile 的更新（`softmax_rescale_o` L137–167）就是第 1 篇递推式的逐字翻译：

```cpp
scores_scale = exp2f((scores_max_prev - scores_max_cur) * softmax_scale_log2);
row_sum *= scores_scale;         // ℓ 迁移
acc_o_rowcol *= scores_scale;    // O 累加器迁移
```

epilogue 的 `normalize_softmax_lse`（L170–186）做 quad 内 shuffle 归约得到最终 $\ell$，然后 $O \leftarrow O/\ell$、$\mathrm{LSE} = m \cdot s + \log \ell$。**循环内连 row_sum 的跨线程归约都省了**（L163–165 注释：推迟到最后）——这就是 FA2 "softmax 几乎免费"的根源，下面解释为什么可以省。

### 1.4 细节一：softmax 归约不需要跨 warp 通信

m16n8k16 MMA 的 accumulator 布局里，**一行的 8 个列元素恰好由同一 quad 的 4 个线程持有**（`lane%4 = 0..3`）。于是行 max/行 sum 的归约只需：线程内归约（`thread_reduce_`）→ quad 内 4 线程蝶形 shuffle（`Allreduce<4>`，`__shfl_xor_sync`）。**一次跨 warp 通信都没有**——softmax 的通信成本被 MMA 布局本身消化了。第 5 篇会看到 Gluon 干脆让"布局"成为用户显式声明的类型。

### 1.5 细节二：P fragment 的零拷贝复用（RS mma）

SM80 HMMA 的操作数必须在寄存器，而 smem 里的 K/V 要经 ldmatrix 搬运。第一个 GEMM（QK^T）每个 k 步都走 `SM75_U32x4_LDSM_N`（ldmatrix）进寄存器再 mma，`gemm()` 里还做了软件流水（第 i 步 mma 同时预取第 i+1 步的 fragment）。

关键在第二个 GEMM（PV）：`convert_layout_acc_Aregs`（`utils.h` L199–212）把 QK^T accumulator 的布局 $(\mathrm{MMA}=4, \mathrm{MMA\_M}, \mathrm{MMA\_N})$ **重排成 HMMA A 操作数布局** $((4,2), \mathrm{MMA\_M}, \mathrm{MMA\_N}/2)$——于是算出来的 $P$（寄存器里的 exp2 结果）**零拷贝直通**第二个 GEMM 的 A 操作数（`gemm_rs`，"RS" = A 在寄存器、B 在 smem）。V 侧则用 `ldmatrix.trans` 直接读 smem 的转置视图，免掉显式转置。这是 FA2 最关键的布局技巧：两个 GEMM 之间**不经过 smem**。

### 1.6 细节三：1.5 级 cp.async 流水

FA2 的 K/V smem 只有**单份**（不是多 stage 环形缓冲），靠"算当前块、预取下一块"拼出流水（L312–346）：

1. 循环体开头 `cp_async_wait<0>() + __syncthreads()`——等上一轮发出的 K 到位；
2. 立刻发当前块的 V；
3. QK^T mma 执行期间再预取下一个 K——`cp_async_fence()` 必须留在 `if` 内（L343–345 注释：放外面同步语义就错了，产生竞态）。

gmem→smem 的拷贝 atom 是 `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`，每线程 16B（8 个 fp16）。选 `CACHEGLOBAL` 而非 `CACHEALWAYS` 的理由写在注释里：同一 CTA 不会重复读同一地址，让它走 L2。谓词化越界处理：`Is_even_MN=true`（seqlen 整除 block）时整段 predicate 编译期消除，最后一个 block 才用 `Clear_OOB_MN` 把 smem 越界行清零。

### 1.7 细节四：数值的 host 侧折叠

`params.scale_softmax_log2 = softmax_scale * M_LOG2E`（`flash_api.cpp` L138）——$1/\sqrt d$ 不进 GEMM，与 $\log_2 e$ 一起折进 exp 时的乘法，`exp2f(tensor*scale - max_scaled)` 让编译器合成一条 `ffma`。`max == -INFINITY` 时强制取 0 防 NaN（L73–76）——与 Triton 教程 `-1e6` 哨兵同一棵树上的不同枝。FA3 的 FP8 路径更进一步：max 额外减 8 使 exp2 结果落在 $[0, 256]$，用满 e4m3 的动态范围（`hopper/softmax.h` L64–69 注释）。

## 2. FA3（Hopper）：warp specialization + TMA + GMMA

FA3 把 kernel 拆成 `flash_fwd_kernel_sm90.h`（骨架）+ `mainloop_fwd_sm90_tma_gmma_ws.hpp`（1717 行主循环）+ `epilogue_fwd.hpp` + `tile_scheduler.hpp`。骨架（L179–454）：

- **线程组织**：1 个 producer warp group（128 线程，发 TMA）+ 1~3 个 consumer warp group（跑 GMMA）。进入分支前各自做寄存器再分配：producer `warpgroup_reg_dealloc<24>()` 释放寄存器、consumer `warpgroup_reg_alloc<240>()` 拿走——**寄存器总量在 warp group 间不对称转移**，SM90 的 `setmaxnreg` 独有能力，load 线程只发指令不携状态。
- **persistent**：grid = SM 数，CTA 常驻，`scheduler.get_next_work` 循环领 tile；且在 epilogue **之前**就先取下一个 work——下一个 tile 的 TMA 与本 tile 的收尾重叠。

### 2.1 TMA 与双 pipeline

Q/K/V 各有独立 TMA descriptor；K 与 V 是**各自独立的 `PipelineTmaAsync<kStages=2>`**（深度 2 双缓冲），transaction 字节数是编译期常量供 `expect_tx` 用。生产者侧 `producer_acquire → tma.copy(barrier, ...)`，单线程发射，不再消耗整个 warp 的地址计算。三个硬件级细节：

- **cluster multicast**：K 沿 cluster M 维广播，cluster 内多个 CTA 共享一次 KV 读取；
- **cache hint 政策化**：K 用 `EVICT_LAST`（同 cluster 其他 CTA 还要复用），Q 用 `EVICT_FIRST`（用过即弃）；
- **K/V 交错预取**（L860–882）：先发 K(n)，下一轮发 K(n-1)+V(n)，使 consumer 里第 n 步的 QK GMMA 与第 n-1 步的 PV GMMA 并行。

### 2.2 GMMA：SS 与 RS 的真实分野

Hopper 的 GMMA 允许 A/B 操作数以 smem descriptor 直读（SS），也可以 A 在寄存器（RS）：

- **QK^T 恒为 SS**：A/B 都用 descriptor 直读 smem——**寄存器带宽不再喂 GEMM**，ldmatrix 从关键路径上消失；
- **PV 默认 RS**：$P$ 留在寄存器直接作 A 操作数（FA2 零拷贝技巧的 GMMA 延续）；寄存器压力过大时（hdim≤64 的 192×192 tile），$P$ 经 `stmatrix` 写回 smem 走 SS——**同一 kernel 两种模式按 tile 形态切换**，选择逻辑在 `tile_size.h`。

### 2.3 softmax 与 GEMM 的重叠：双层设计

这是 FA3 相对 FA2 的质变——FA2 里 softmax 与 GEMM 在同一线程上串行。FA3 两层重叠：

**WG 间 pingpong**：两个 consumer warp group 通过 `WarpSchedulerWG1/WG2` named barrier 轮流进入 QK GEMM——一组做 QK+softmax 时另一组做 PV，softmax 的 SFU/ALU 指令与 GEMM 的 tensor core 指令**在不同 warp group 间并行**。

**WG 内 GMMA 异步 overlap**（`IntraWGOverlap`）：利用 GMMA 的异步性，`gemm<..., wg_wait=-1>` 发出 GEMM 不等完成，先做上一迭代的 softmax（scale/max），之后 `warpgroup_wait<1>` 只等 1 条 mma 在飞。`fwd_step` 的时序（L1170–1207）：QK(n) 发出 → PV(n-1) 发出 → softmax(n-1) 前半 → release K → softmax(n-1) 后半 → P 转换 → PV 等待。

相应地，FA2 的单体 `softmax_rescale_o` 被拆成四段可重组的状态机（`hopper/softmax.h`）：`max_get_scale` / `online_softmax` / `rescale_o` / `finalize`——`scores_scale` 作为一个寄存器 fragment 在 GEMM 前后传递。LargeHeadDimV（dv>256）场景甚至有**跨 warp group 的 softmax 状态交换**：WG1 算 QK+softmax，scale 经 `smem_scale` + `PFull/PEmpty` named barrier 传给专职做 PV 的 WG2。从 189 行单体到 170 行四段式，是架构变迁的最小缩影。

### 2.4 Persistent 调度与 L2 swizzle

`DynamicPersistentTileScheduler` 做 **L2-aware swizzle**：按 `swizzle = 2^floor(log2(L2 / size_one_kv_head))` 把 head/batch 分组，让同一时间窗内的 CTA 集中在少数 KV head 上，令 KV 常驻 50MB L2。FA2 完全没有这层——第 1 篇说"SRAM 越大 attention 越快"，Hopper 的故事一半在片上，一半在这 50MB L2 的调度。

### 2.5 Epilogue：TMA store

accumulator 经 STSM atom 写 `smem_o`，`fence_view_async_shared()` 后 TMA store 异步写回；`smem_o` 与 `smem_v` union 复用。一个 smem 精算的坑值得记录（mainloop L305–307 注释）：即使 `smem_p` 是空 array，放进 TensorStorage 也会让 smem 从 227KB 涨到 228KB 触发 launch 失败（H100 上限 227KB+1KB 预留），所以用条件类型——**空成员的对其开销也能杀死 kernel**。

## 3. Split-KV：长序列的并行度补丁

标准 kernel 的并行度 = `batch × head × m_blocks`。解码（小 Q、大 KV）时不够填满 GPU——split-KV 把 KV 序列切成 `num_splits` 段并行，每段算出**未归一化的部分输出**（fp32 存 `oaccum`，epilogue 不做归一化，LSE 是局部的），再由 combine kernel 归并。

**触发启发式**（`flash_api.cpp` L281–315）：`batch*heads*m_blocks >= 0.8*num_SMs` 直接不分；否则枚举 split 数算 wave 效率 $\mathrm{eff} = n_{waves}/\lceil n_{waves} \rceil$，取达到最大效率 85% 的**最小** split 数——注释里给了具体例子：48 个 block 配 108 SM，2 splits（eff 0.89）优于 3 splits（0.67）。FA3 追加一条：单个 KV head 超 50MB L2 时即使 block 数够也要按 $\lceil \text{size}/50\text{MB} \rceil$ 分裂以保 L2 复用。

**combine 的数学**：第 $s$ 段的局部量 $(O_s, L_s)$（$O_s$ 未归一、$L_s = m_s + \log \ell_s$）：

$$
L = \log \sum_s e^{L_s}, \qquad O = \sum_s e^{L_s - L}\, O_s
$$

——注意这就是 **online softmax 递推在"块"粒度的再次现身**（第 1 篇的 α 因子公式，块从 KV tile 换成了 split 段）。FA2 的 combine kernel（L1117–1299）用 smem 转置矩阵缓存各段 LSE（+1 列防 bank conflict）、全 `-inf` 时置 0 防 NaN；FA3 版本工程更重：4-stage cp.async 流水加载 partial O、`max_valid_split` 短路（超出的段 scale 必为 0，直接停止加载）、PDL 衔接前序 kernel。

## 4. Launch template：静态分派的暴力美学

`run_flash_fwd` 用 `BOOL_SWITCH` 宏把运行时 bool 翻成 `constexpr bool` 的双分支：`Is_even_MN → Is_even_K → Is_local → Return_softmax → Has_alibi → Is_softcap`，加上 `Is_dropout/Is_causal`——**理论组合数百个 kernel 实例**，靠注释里的组合缩减规则（hdim>128 强制 `Is_even_MN=false` 等）和 `FLASHATTENTION_DISABLE_*` 编译裁剪控制产物规模。smem 超 48KB 时显式 `cudaFuncSetAttribute`。

这套"把 if 梯子变成编译期特化"的做法是 CUDA kernel 的通行模式——Triton 用户在 `tl.constexpr` 参数上享受到的是同一件事的自动版。

## 5. FA2 vs FA3 总结表

| 维度 | FA2 (SM80) | FA3 (SM90) |
|---|---|---|
| 线程模型 | 4~8 warp 对称做 QK^T 与 PV | 1 producer WG + 1~3 consumer WG，寄存器 24:240 不对称分配 |
| 全局内存 | cp.async 16B/线程，线程算地址 | TMA bulk + descriptor，单线程发射，cluster multicast |
| smem 流水 | K/V 单缓冲 + 预取下一 K（1.5 级） | K/V 独立 `PipelineTmaAsync<2>`，smem_o/smem_v union |
| GEMM | HMMA，操作数 ldmatrix 进寄存器；P 零拷贝复用（RS） | QK 恒 SS（descriptor 直读）；PV 按寄存器压力 RS/SS 切换 |
| softmax | 与 GEMM 串行，单体状态机 | 四段式状态机；WG 间 pingpong + WG 内 GMMA 异步双层重叠 |
| 调度 | grid 直映 (m, b, h) | persistent + L2 swizzle + PDL |
| 写回 | smem 中转 + 向量化 store | TMA store |
| Split-KV | wave 效率 85% 启发式 | + KV head 超 50MB L2 强制分裂 |

## 6. 小结

- FA2 的工程关键词是**对称与复用**：所有 warp 同构干活，softmax 归约被 MMA 布局消化（quad shuffle 即可），两个 GEMM 之间 P fragment 零拷贝，cp.async 手工拼 1.5 级流水。
- FA3 的关键词是**分工与异步**：TMA/GMMA/warp specialization 把"加载、GEMM、softmax"三类指令流解耦到不同 warp group，softmax 从单体拆成四段以便插进 GEMM 间隙，persistent 调度把 L2 当作设计目标。
- 两代实现的分野印证了第 2 篇的观察：**硬件每提供一种新的异步原语（cp.async → TMA/GMMA → setmaxnreg），就多一层"谁在等谁"需要显式编排**。第 5 篇的 Gluon 会把这条线推到尽头——把这套编排表直接交给用户写。
- Split-KV 的 combine 公式是 online softmax 在 split 粒度的复用——好数学会被反复征用，这是本系列第三次看到同一递推。

下一篇[（四）]({{< relref "flash-attention-04-bwd" >}})：反向传播——$\Delta$ 恒等式、recompute 的代价与收益、以及 dQ 跨块归约的三种解法。
