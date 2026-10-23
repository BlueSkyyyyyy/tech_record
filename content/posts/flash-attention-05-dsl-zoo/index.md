---
title: "Flash Attention 精读（五）：同一算法的五个抽象层级 —— TileLang、Gluon 与 Liger 的启示"
date: 2026-08-28
draft: false
weight: 5
series: ["flash-attention"]
tags: ["flash-attention", "tilelang", "gluon", "triton", "liger-kernel", "系列"]
categories: ["算子开发"]
---

前四篇里，Flash Attention 的数学与工程已经拆完了。本篇换个视角：**同一个算法，在不同抽象层级的 DSL 里长什么样**。读的代码：

- [TileLang](https://github.com/tile-ai/tilelang) `examples/flash_attention/`（MHA 前向加反向，含 Blackwell sm100 版）
- [Triton Gluon](https://github.com/triton-lang/triton) `python/examples/gluon/01-attention-forward.py`（1285 行，Blackwell 专属）
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) `src/liger_kernel/ops/fused_neighborhood_attention.py`（一个"名不副实"的反面样本）

三份代码恰好构成一个光谱：**TileLang 把"内存层级调度"显式化，Gluon 把"warp 与异步协议"显式化，Liger 则示范了当稀疏模式没有被落实进循环结构时会发生什么**。加上前两篇的 Triton 教程版和 CUDA 版，正好是同一算法的五个抽象层级。

## 先记住一个画面

同一份乐谱，交给不同的编制来演奏。

- **手写 CUDA/CUTLASS** 像一支交响乐团：每个声部、每件乐器的进出都写在总谱上，表现力最强，但排练成本最高；
- **Gluon** 像你自己排的室内乐团：谁坐哪个位置、谁先起拍、用多少力气，全部由你指定；
- **TileLang** 像一台编曲机：你告诉它"这一段用弦乐、那一段用铜管"（存储层级），它负责把音符分给具体乐器；
- **传统 Triton** 像把乐谱交给一位有经验的指挥：你只写音符，配器由他决定；
- **Liger 的 neighborhood attention** 则像拿交响乐谱让口哨吹——谱子没错，但编制和曲子不匹配，效果自然不对。

> **类比的边界**：乐器换掉，音高和节奏仍然可辨；kernel 换掉抽象层级，**数值必须逐位或同量级一致**，但性能可能差一个数量级。也就是说，"同一算法"在工程上从来不是"同一份实现"，抽象层级的选择会实打实改变访存、并行度和可表达的优化。

读完这一篇，你应该能回答三个问题：

1. TileLang 把"内存层级"显式化之后，为什么代码更长却可能更快？它买的到底是什么？
2. Gluon 为什么要把教程版 120 行的核心循环展开成 1285 行？多出来的代码换来了什么控制力？
3. Liger 的 neighborhood attention 为什么"名字叫 fused，实际不 fused"？判断一个 attention 变体该不该改 kernel 结构，标准是什么？

---

## 1. TileLang：tile 级调度 DSL

### 1.1 编程模型：scope 即地址空间

TileLang 构建在 TVM TIR 之上，用户写"一个 block 内的 tile 语义"，编译器降低到 cp.async / TMA / WGMMA / tcgen05。它最有辨识度的是**分配语句直接声明存储层级**（`example_mha_fwd_bshd.py` L37–48）：

```python
Q_shared = T.alloc_shared([block_M, dim], dtype)               # 共享内存
acc_s    = T.alloc_fragment([block_M, block_N], accum_dtype)   # 寄存器 fragment（fp32 累加）
acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)       # 降精度用
acc_o    = T.alloc_fragment([block_M, dim], accum_dtype)
scores_max = T.alloc_fragment([block_M], accum_dtype)
```

对比 Triton：`tl.load(...)` 得到的"块值"物理落点完全不可见；TileLang 要求你**声明每个 tile 住在 shared 还是 fragment，并手写每次搬运**。

代价是代码更长（前向 162 行 vs Triton 教程约 100 行），收益是编译器不用猜数据放置，layout 推理可以做全局验证。**这就像装修时自己画水电图**：比"跟师傅说看着办"麻烦，但后面每个插座在哪你都知道。

计算原语是 tile 级的：

| 原语 | 作用 | 编译器的自由度 |
|---|---|---|
| `T.copy` | tile 拷贝 | 按 src/dst 的 scope 组合自动选 TMA/cp.async/ldmatrix |
| `T.gemm` | tile 矩阵乘 | `transpose_B=True` 即 $A @ B^\top$；`policy=T.GemmWarpPolicy.FullRow` 指定 warp 怎么瓜分输出 tile |
| `T.reduce_max/sum` | tile 归约 | 选择归约树与线程映射 |

外层函数是模板：`block_M / num_stages / threads` 是 Python 变量，trace 时烘焙成常量——**静态 shape 特化**，配合 `@autotune` 穷举调参（每个配置组合编译一个独立 kernel）。

### 1.2 前向：与第 1 篇公式的对照

循环外初始化（L50–57）：`T.copy(Q[...], Q_shared)` 让 Q 常驻 shared；`scores_max` 初始化为 $-\infty$；causal 时循环上界收缩到 `(bx+1)*block_M`——**块级 early-exit**，FA2 的标准优化，直接砍掉一半循环。

主循环 `for k in T.Pipelined(loop_range, num_stages=num_stages)`（L59）的循环体，逐行对照数学：

```python
T.copy(K[bz, k*block_N:(k+1)*block_N, by, :], K_shared)   # L60 异步加载 K 块
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, ...)  # L67 S = Q @ Kᵀ（累加在预填的 mask 上）
# L69-73  running max 两段更新：scores_max = max(history, rowmax(acc_s))
scores_scale[i] = exp2((m_prev - m_new) * scale)          # L74-75 α 因子（log2e 已折进 scale）
acc_s[i,j] = exp2(acc_s[i,j]*scale - m_new*scale)         # L76-77 P_block
logsum = logsum * scores_scale + scores_sum               # L78-80 ℓ 迁移 + 累加
T.copy(acc_s, acc_s_cast)                                 # L81 fp32 → fp16（对应 p.to(fp16)）
acc_o = acc_o * scores_scale + P @ V                      # L83-87 第二次 GEMM
```

两个值得学的技巧：

**掩码"先填后累加"**（L61–66）：先往 `acc_s` 里写 0 或 $-\infty$，再让 `T.gemm` 直接加在上面。$-\infty + \text{有限值} = -\infty$，exp2 后正确变 0——把 `tl.where` 式的逐元素掩码变成一次写入，省掉整块的比较指令。

**全程 exp2**（L24 预乘 `1.44269504`）：和 Triton 教程同一招——`ex2.approx.f32` 是唯一硬件指数指令，这个 trick 在任何 DSL 里都得手工做。**抽象层级再高，硬件的脾气不会变**。

epilogue（L89–92）`acc_o /= logsum` 后经 `O_shared` 中转写回——fragment → shared（stmatrix）→ global（可生成 TMA store）两跳。

### 1.3 反向：单 kernel 三梯度 + 原子加

TileLang 反向（`example_mha_bwd_bshd.py`）与 Triton 教程（第 4 篇）数学完全相同，**kernel 组织不同**：

- Triton 拆成 `_attn_bwd_dkdv` 和 `_attn_bwd_dq` 两个 kernel；
- TileLang 用**单 kernel 内联三梯度**（dv/dk/dq 共享同一次 $QK^\top$ 重算与 P 计算），省一遍 HBM 读 Q/K/V，代价是 dq 必须跨 block 归约——走 `T.atomic_add(dQ[...], dq[i,j])`（L234–235）。

这里有个 Triton 做不到的深度优化（L118–120）：

```python
def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])
```

一个从逻辑坐标到物理偏移的仿射映射，把 dQ 的内存排布**重排成与 mma fragment 输出一致的顺序**，使 atomic_add 能发出向量化 `red.global` 指令；输出后再由 postprocess kernel 拷回自然布局。dQ 用 fp32 累加保证精度。

这是"用 layout 声明换原子操作带宽"的范例——Triton 的布局对用户不可见，这类优化只能靠编译器撞大运。**TileLang 的"多写代码"，买的就是这种"我能指定物理布局"的权力**。

反向同样用前向存的 **log2 域 LSE** 直接恢复 P：`qkT[i,j] = T.exp2(qkT[i,j]*scale - lse_shared[j])`（L212–213），一步到位。

### 1.4 Blackwell（sm100）：TMEM 与显式 warp specialization

`examples/flash_attention_sm100/` 展示了 Blackwell 三件套，全部以 Python API 暴露：

- `T.alloc_tmem`：tcgen05 MMA 的累加器不放寄存器，放 256KB 的 tensor memory；
- `T.tcgen05_gemm(..., mbar=mbar_s, clear_accum=True)` 加 `T.mbarrier_wait_parity(mbar_s, k % 2)`：异步 MMA 加 **parity 语义**的 mbarrier（一次分配、交替相位复用）；
- `variant='wasp'` 时是完整的 CUTLASS 式 warp specialization：L189 的 docstring 写明角色划分——softmax warp（tid 0–127）、DMA warp（128–159）、BMM warp（160–191），10 组 mbarrier，双缓冲 K/V，`parity = (k // num_stages) & 1` 手工管理相位。

TileLang 论文（ASPLOS'25）自称的性能来源，从这些例子看是真实的：tile 原语让调度空间显式化（`T.copy` 自动 TMA、`T.gemm` 自动 WGMMA/tcgen05）、`T.Pipelined` 自动多缓冲、`T.Layout` 用户可控 swizzle、sm100 的硬件机制近乎 1:1 暴露。README 里"MLA decode 80 行追平 FlashMLA 汇编版"的说法，抽象层级选在"恰好够用"的位置。

---

## 2. Gluon：把 CUTLASS 的决策写成 Python

Triton 的传统姿态是"你写张量表达式，编译器管机器"。Gluon 是官方承认这条路线在 Blackwell 上走到头了的产物：`01-attention-forward.py` 把教程版 120 行的核心循环展开成 1285 行，**几乎每一层原本由编译器做的决策都变成了用户代码**。

导入区（L9–25）直呼硬件实体之名：`TensorDescriptor`、`mbarrier`、`tcgen05_mma`、`TensorMemoryLayout`、`float2`…… 读这份文件的感觉，像从"点菜"变成了"进后厨自己开火"。

### 2.1 warp specialization：从布尔提示到编排表

教程版是 `tl.range(..., warp_specialize=True)` 一个提示，编译器自己决定分工。Gluon 版（L931–938）：

```python
gl.warp_specialize([
    (_attn_fwd_correction, (...)),   # default partition：占父区域原有 4 个 warp
    (_attn_fwd_softmax0,      (...)),
    (_attn_fwd_softmax1,      (...)),
    (_attn_fwd_mma,           (...)),
    (_attn_fwd_load,          (...)),
    (_attn_fwd_epilogue,      (...)),
], [4, 4, 1, 1, 1], [192, 192, 24, 24, 24])   # 各 worker 的 warp 数 / 寄存器配额
```

15 个 warp 各司其职，寄存器预算极不均匀：两个 softmax partition 各 4 warp × 192 寄存器（重 SFU/向量计算），mma/load/epilogue 各 1 warp × 24 寄存器（只发异步指令，几乎不携状态）。这种"算力的不对称分配"在传统 Triton 里是编译器内部启发式，现在由用户直接排表（经 SASS `setmaxnreg` 动态再分配）。

打个比方：传统 Triton 像跟司机说一句"**开快点**"，剩下交给他的经验；Gluon 则是你自己排好整张发车时刻表——谁几点发车、走哪条车道、加多少油（寄存器），全写死在代码里。**自由度换控制力**，这就是它敢收 3 倍代码量的底气。

### 2.2 Channel：手写生产者-消费者框架

`num_stages=3` 在 Gluon 里没有对应物。替代品是 L84–174 的 `Channel` 泛型——一块多缓冲存储加 ready/empty 两组 mbarrier 加环形游标，被实例化为 `SharedMemoryChannel` 和 `TensorMemoryChannel` 两种（SMEM 和 TMEM 用同一套协议）。

流水线深度变成**逐 channel 独立调参**：Q 2 个缓冲、KV 2~8 个（host 侧按 head_dim/dtype/causal 精调）、S 即产即销 1 个。Producer 等 `empty` 返回 `ready`，consumer 反之——教科书级的 handoff 协议，90 行框架代码换来的是流水线正确性自负。

**这里的取舍很值得体会**：Triton 的 `num_stages` 是一个"全局旋钮"，Gluon 把它拆成"每个队列各自定深度"。真实流水线里，Q 只读一次、K/V 反复读、S 用完即弃，本来就不该用同一个深度——**统一旋钮是抽象的便利，逐队列调参才是硬件的真相**。

### 2.3 值得抄走的三个细节

**S 复用为 P（L385–388）**：$QK^\top$ 的 fp32 结果在 TMEM 里被 `_reinterpret` 按 fp16 打包宽度原地重解释为 P 的缓冲——softmax warp 稍后把 exp2 结果写回同一地址。TMEM 是稀缺资源，一块存储两用。

**alpha 外包给 correction warp（L644–648, L761–779）**：教程版在 softmax 循环内联做 `acc = acc*alpha`；Gluon 版把 $\alpha$ 写进 TMEM、arrive 一个 barrier，交给独立的 correction warp 组去做 O 的重缩放——softmax 的关键路径（等下一个 S）不被 rescale 拖住。

**位掩码加 R2P（L572–593）**：causal mask 不生成显式 mask 张量，而是 16 元素粒度的位掩码，配 `gl.map_elementwise` 展开——让 SASS 层能用 R2P（register-to-predicate）指令。注释致谢 Tri Dao，即从 Flash Attention CUDA 版移植的 SASS 级技巧，传统 Triton 的 `tl.where` 无法表达。EX2 旋转门（L656–658）更绝：两个 softmax partition 在指数段**轮流进入**，避免争抢 SFU 的 EX2 单元——连 warp 组对单个功能单元的争用都纳入了显式调度。

### 2.4 一张对照表

| 传统 Triton（教程 06） | Gluon |
|---|---|
| `tl.dot`，编译器选指令/布局 | 直接调 `tcgen05_mma`，指定操作数与完成屏障 |
| `num_stages=N` 一个旋钮 | 每 channel 独立缓冲数加手写 acquire/wait |
| `warp_specialize=True` 布尔提示 | 函数 × warp 数 × 寄存器配额的编排表 |
| SMEM 隐式分配 | 显式 alloc，K/V 共享池 `_reinterpret` |
| 布局编译器推导 | 三套布局系统（寄存器/SMEM/TMEM）加 `convert_layout` 手术 |
| barrier 不可见 | 40 多处 mbarrier init/prime/expect/wait/arrive |

代码量约 3 倍，知识密度完全不同。定位很诚实：L861–866 把 fp8 配置命名为 `cutlass_gluon_attention`，注释说 fp8 比 CUTLASS 快最多 150 TFLOPS——它就是"带类型系统的 CUTLASS"，把 CUDA 里运行时才炸的同步错误前移成编译期错误，服务对象是 kernel 工程师冲刺最后 20% 性能。而教程 06 仍然在维护（它自己也吸收了 TMA descriptor、`warp_specialize` 提示），`tl` 层服务 90% 用户、`gluon` 层服务 90% 之后的性能——双层路线并存。

---

## 3. Liger 的 fused_neighborhood_attention：一个反面样本

Liger-Kernel 的 `fused_neighborhood_attention.py`（1022 行）名字带 "fused"，但深读后必须先纠正预期：**它不是 flash-attention 风格的实现**——没有 online softmax、没有块级跳过、显式物化完整的 $N \times N$ 注意力矩阵。

### 3.1 算法与实现的错位

Neighborhood attention（滑窗注意力）本身是个好想法：query $i$ 只 attend 窗口 $[i - \lfloor k/2 \rfloor d,\ i + \lfloor k/2 \rfloor d]$ 内的 key（$k$ 为奇数窗口宽，$d$ 为 dilation），复杂度理论上 $O(N \cdot k)$。但实现是"分阶段调用的朴素 attention"：

1. 一个独立 kernel 生成 $[N, N]$ 的 0/1 mask 矩阵（L13–67），写进显存；
2. QK kernel（L71–193）算 $QK^\top$，逐元素**从全局内存读入 mask**（L180–182），窗口外写 $-\infty$，物化 $[B,H,N,N]$ 的 scores；
3. 调 Liger 自己的 softmax kernel（两遍式）；
4. AV kernel（L196–299）算 $P V$。

关键错位：**`kernel_size/dilation` 的唯一作用域就是那 10 行 mask 代码**。QK kernel 的 grid 是 `(B*H, cdiv(N,BM), cdiv(N,BN))`，对所有 tile 一视同仁——窗口收窄不减少任何一次 `tl.dot`、任何一次全局访存。

对比第 2 篇 Triton 教程：causal mask 有"整块跳过 / 免 mask / 部分相交"的三分类块级判断；这里有零分类，mask 是先算完整个 tile 的 dot 再逐元素 `where` 上去的。

结果：**正确但不快、显存不省**。计算量、带宽、显存全部照旧 $O(N^2)$，还多出一个 $N^2$ 的 mask 矩阵要写要读；反向保存完整 attn_weights（L869 `save_for_backward`，不做 recompute），训练显存 $O(N^2)$。集成层 docstring 宣称的 "reducing complexity from O(n²) to O(n·k)" 与 kernel 实际行为不符。

> **这不是在嘲笑 Liger。** 它说明的是一个很常见的工程陷阱：**把一个"结构上更稀疏"的算法，实现成了"数值上加 mask"**。数值 mask 只改变结果，不改变工作量；要让稀疏真正兑现，必须让循环边界、tile 遍历、显存分配跟着变。

### 3.2 错在哪：模板填空的天花板

Liger 的代码组织是一套优秀的五层模板（ops 层 kernel 加 autograd、复用库内 softmax、functional 入口、独立 nn.Module、测试内嵌纯 PyTorch 参考实现），neighborhood attention 基本是照模板"填空"出来的。

问题在于：**新变体的核心计算结构变化（滑窗 → 循环边界收缩）没有被落实进 kernel 结构，只被落实进了 mask 数值**。

同一仓库里的 `attn_res.py`（Kimi 的 attention residuals，深度方向的小 softmax）是正面例子：N ≤ 16 全留寄存器、布局为 coalesced 访存设计、注释解释"为什么"——计算结构变了，kernel 结构跟着变。

这个对比给"在 kernel 库里落地新 attention 变体"的启示：

**先问"这个变体的性能来源对应循环结构的哪一处改动"**——

| 变体 | 性能来源 | kernel 结构该改哪 |
|---|---|---|
| 滑窗 / neighborhood | 只算窗口内 | 循环边界收缩、块级跳过 |
| 块稀疏 | 只算选中的块 | tile 剪枝、索引 gather |
| 新激活 / softcap | 换一个逐元素函数 | 替换 exp 调用点，其余不动 |
| MLA | 压缩 KV、吸收投影 | 换 K/V 的表示与 GEMM 形状 |

让 kernel 结构跟着计算结构走，稀疏/变体才真正兑现。mask 数值化只换来正确性，没换来变体存在的意义。

顺带一提，Liger 的 `multi_token_attention.py`（DeepMind MTA，`out = mask₀(conv2d(softmax(mask₋∞(scores))))`）走了另一条务实路线：Triton 只写"mask 这种纯元素级胶水 kernel"（还细心地用 `-1e9` 而非 `-inf`，避免 $-\infty \times 0 = \mathrm{NaN}$ 破坏乘累加形式的 mask 合成），softmax 复用库内实现、卷积直接交给 cuDNN——**识别出"哪部分值得手写"本身是一种工程能力**。

---

## 4. 总结：五个层级的选型

| 层级 | 代表 | 用户管理什么 | 代码量（fwd） | 适合 |
|---|---|---|---|---|
| 手写 CUDA/CUTLASS | FlashAttention-2/3 | 一切 | 数千行加模板 | 极致性能、新硬件首发 |
| Gluon | triton gluon 示例 | warp 分工、mbarrier、TMEM、布局 | ~900 行 | kernel 工程师冲刺最后 20% |
| TileLang | tilelang examples | tile 的 scope/搬运/流水线深度、layout | ~160 行 | 变体原型加需要控制内存层级 |
| 传统 Triton | 教程 06 | 只有算法（block 大小/`num_stages` 两个旋钮） | ~100 行 | 90% 场景、80% 性能 |
| 拼装 | Liger neighborhood | 只有算子组合 | ~500 行 | 只求正确接入（性能另说） |

抽象的演进方向很清楚：**每一次硬件换代，都有一层"编译器猜不准"的调度决策从隐式变成显式**——Ampere 时代 `num_stages` 够用，Hopper 需要 TMA 和 warp specialization 提示，Blackwell 干脆把 mbarrier/TMEM/寄存器配额全交给你。

DSL 的竞争点不在"谁更简洁"，而在**显式化的那层抽象选得对不对**：TileLang 选了 tile/内存层级，Gluon 选了 warp/异步协议，两者都比"让编译器猜"更接近 Blackwell 的真相。而 Liger 的反例提醒我们：**抽象层级选错，连"算法变体"本身都会落空**。

---

## 5. 系列结语

前五篇走完：从 softmax 的溢出问题（[一]({{< relref "flash-attention-01-theory" >}})），到 200 行 Triton 的完整工程（[二]({{< relref "flash-attention-02-triton-fwd" >}})），到 CUDA 的 warp 分工与 Hopper 异步（[三]({{< relref "flash-attention-03-cuda-fwd" >}})），到反向的 $\Delta$ 恒等式与三种 kernel 组织（[四]({{< relref "flash-attention-04-bwd" >}})），再到 DSL 光谱（本篇）。

回头看，Flash Attention 的故事其实是两层：

**数学层**：softmax 是一个可流式计算的可交换归约——$e^{x}$ 的乘法常数不变性允许坐标系动态迁移，这让 $O(N^2)$ 的中间矩阵消失。

**工程层**：让数学落到硅上的是一整套"分块 + 异步 + 精度边界"的决策——哪些量留在寄存器（$m, \ell, \mathrm{acc}$）、哪些降精度（只降 tensor core 操作数）、哪些提前算（LSE、$\Delta$）、哪些循环跳过（causal 对角带外）。每一层的实现，无论 Triton、CUDA、TileLang 还是 Gluon，都在回答同一组问题，只是给出答案的自由度不同。

### 自检三问

1. TileLang 的 `T.copy` 与 Triton 的 `tl.load` 在"谁决定数据放在哪"上的差别，会怎样影响一个 bank conflict 优化的可表达性？
2. Gluon 把 `num_stages` 拆成"每 channel 独立深度"，为什么这比一个全局旋钮更贴近硬件？代价是什么？
3. 如果让你给 neighborhood attention 做一个真正 fused 的实现，你会改哪三处结构？（提示：循环边界、mask 生成、反向重算）

配套代码与基准脚本见仓库 [code/](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code) 目录。

到这里，这五个抽象层级已经走完。但官方仓库里还藏着两件本篇没有展开的事：**这些文件在真实仓库里如何被组织、编译、调度**，以及**反向传播在每一代硬件上到底被写成了什么样**。分别见[第 6 篇：官方仓库源码全景]({{< relref "flash-attention-06-official-code" >}})与[第 7 篇：反向传播深潜]({{< relref "flash-attention-07-bwd-deep" >}})。
