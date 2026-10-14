# MoE 反向的激活：存下来还是重算（activation checkpointing）

> 第 144 篇（P144-1）拆了 MoE 反向一直缺席的「非 GEMM 段」——SwiGLU 反向 + 激活梯度重量化，
> 并指出它的输入是前向 linear1 的**预激活** `h = x @ W1ᵀ`，**前提是前向不单独存 gate/up**；
> 第 145/156 篇（P145/P146）试图把这段折进 dgrad 的 epilogue；第 159 篇（P159-1）给了完整
> 「前向 + 反向」分段账。但**从没量过**一件事：如果前向（尤其 Mega MoE 的融合主 kernel）确实
> 不存 `h`，反向把它**重算**一次的代价是多少、显存省多少、数值上是否无损。
>
> 本篇用 **H100(sm90) 可跑的真实 SM90 FP8 grouped/dense kernel**（DeepGEMM 2.8.1）把这条
> 「存 h / 存一半 / 重算」三选一量到底。**Mega MoE 融合主 kernel 是 SM100 专属、仅分析**：
> 本文用它**架构无关的布局与数学语义**（`h` 不落地）建立问题，数值实验全在 H100 真机。

---

## 1. 问题：SwiGLU 反向需要 `h`，而融合前向不存它

MoE 的一层前向（单专家视角）：

```
linear1:  h = x @ W1ᵀ            # h ∈ R^{m × 2I} = [gate | up]  预激活
SwiGLU:   a = silu(G1) * G2      # 取 h 的前 I 列 G1、后 I 列 G2
linear2:  y = a @ W2ᵀ
```

反向（第 154 篇推导）在 SwiGLU 处需要 **`h` 的两半**：

```
dG1 = dA * silu'(G1) * G2
dG2 = dA * silu(G1)
```

即 SwiGLU 反向的输入是 **`h`（gate/up 预激活）**，不是它的输出 `a`。

**关键事实**：DeepSeek 的 Mega MoE 把 `dispatch → L1 → SwiGLU → L2 → combine` 融进**单个
mega-kernel**（第 02/10 篇）。它的 L1 epilogue 在**寄存器/TMEM 里原地做 SwiGLU 再量化**，
**从不把 `h` 落到 global memory**：

- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1108-1134`：从 TMEM 取出
  gate/up 的 `fp32_values`，`__float22bfloat162_rn` 舍入（`:1112-1113`）、可选 clamp
  （`:1116-1120`）、`silu(gate)*up`（`:1122-1134`），结果写进 `activation_values`（随后
  量化成 L2 输入）。**没有一支写回 `h` 的路径**。
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh:360-365` 列出 mega buffer 的持久缓冲：
  `l1_token_buffer`（= dispatch 后的输入激活 `x`，fp8）、`l1_sf_buffer`、`l1_topk_weights_buffer`、
  **`l2_token_buffer`（= 量化后的 `a`，L2 输入）**、`l2_sf_buffer`、`combine_token_buffer`。
  `:427-445` 分配它们；`input_token_layout=Data(hidden*num_mma_elem_bytes)`（`:389`）、
  `intermediate_token_layout=Data(intermediate_hidden*num_mma_elem_bytes)`（`:391`）。**没有任何
  buffer 是 `h`（`2*intermediate_hidden`）**。

所以反向要拿到 `h`，只有两条路：

| 方案 | 做法 | 额外显存 | 数值 |
|---|---|---|---|
| **STORE-ALL** | 前向把 `h`（bf16，`[m,2I]`）落盘 | `2·m·I·2 B` | 与真 `h` 逐位（前向输出本就 bf16） |
| **STORE-G1** | 只存 `G1`（`[m,I]`），由已存的 `a` 反推 `G2 = a / silu(G1)` | `m·I·2 B`（一半） | 反推在 `silu(G1)` 小值处放大存储误差 |
| **RECOMPUTE** | 不存 `h`；反向再发一次 linear1 grouped GEMM `h = x @ W1ᵀ` | **0**（`x`、`W1` 本就在手） | 同一确定 kernel ⇒ 逐位 |

> `x`（L1 输入，fp8）本来就要存给 wgrad（`dW1 = xᵀ dG`），`W1` 是常驻权重，所以 RECOMPUTE
> **不需要任何新的操作数**，只是多一次 GEMM。

---

## 2. Part A：重算与存储**逐位相等**（H100 真机）

`harness/exp_moe_activation_recompute.py` Part A，用真实 SM90 `m_grouped_fp8_gemm_nt_contiguous`
（前向 L1 的 grouped 入口，`csrc/apis/gemm.hpp:181`）在同一份 per-token FP8 输入上跑两次：

```
T=256 H=2048 I=1024 E=8 topk=2  m_total=1024 (M-alignment=128)
   STORE-ALL h vs RECOMPUTE h:  torch.equal = True   (max|d|=0.000e+00 bf16 ulp)
   SwiGLU 反向 dG:  torch.equal = True  rel = 0.000e+00
   STORE-G1 反推 G2:  vs 真 G2  rel(fp32 存储)=6.824e-02  rel(FP8 存储)=4.273e-01
```

- **RECOMPUTE 与 STORE-ALL 逐位相同**：同一个确定 kernel（第 170 篇已证 grouped GEMM 结构性
  bit 确定——`scheduler/gemm.cuh:91-104` 只沿 M/N 切块、不切 K），喂同一份量化输入，输出 bf16
  **bit-identical**；进而由它算出的 SwiGLU 反向 `dG` 也逐位相同（`rel=0`）。
  ⇒ **重算在数值上是免费的**，不是「近似还原」。
- **STORE-G1 不免费**：由已存的 SwiGLU 输出 `a` 反推 `G2 = a / silu(G1)`，相对误差
  fp32 存储 `6.8e-2`、**FP8 存储（生产实际存的）`4.3e-1`**——除非只存 `G1` 且把 `a` 也按 fp32
  存（那就没省多少），否则反推会被 `silu(G1)` 小值放大。

---

## 3. Part B：显存账（DeepSeek 量级）

`h` 的字节：routed `m_total·2I·2`（bf16）+ shared `T·2S·2`（`S = E_s·I`）。`m_total` 按
`M-alignment=128` 对齐（第 13 篇）。

| T | m_total | STORE-ALL | STORE-G1 | RECOMPUTE | 参照：一张 `W1`(fp8) | STORE-ALL/W1 |
|---|---|---|---|---|---|---|
| 2048 | 4096 | 50.3 MB | 25.2 MB | **0** | 234.9 MB | 21% |
| 4096 | 8192 | 100.7 MB | 50.3 MB | **0** | 234.9 MB | 43% |
| 8192 | 16384 | 201.3 MB | 100.7 MB | **0** | 234.9 MB | 86% |

（单层激活 `x`：T=2048 时 29.4 MB / 4096 时 58.7 MB。`H=7168, I=2048, E=8, topk=2, E_s=1`。）

- `h` 的显存随 **token 数线性增长**（`m_total = T·topk`），且是**整层激活量级**（T=8192 时已
  接近一整张 fp8 权重矩阵的 86%）。
- RECOMPUTE 额外 **0**：反向 wgrad L1 本就要存 `x`（fp8），它就是重算的输入。

---

## 4. Part C：重算的**时间预算**——能否藏进「不依赖 h 的反向 GEMM」

重算 = 一次前向 L1 GEMM。它必须在对 `h` 有依赖的 **SwiGLU 反向 → dgrad L1/wgrad L1** 之前完成；
而反向里有两条**不依赖 `h`** 的大 GEMM 可以给它当 shadow：

- `dgrad L2`：`dA = dY @ W2ᵀ`（`m_grouped`，与 `h` 无关）；
- `wgrad L2`：`dW2 = aᵀ dY`（`k_grouped`，`A` 操作数是量化后的 `a`，与 `h` 无关）。

H100(sm90) 真机（空闲 GPU1，`H=7168, I=2048, E=8, topk=2, E_s=1`）：

```
T=2048  H=7168 I=2048 E=8 topk=2 Es=1  m_total=4096 total_k=4096
   RECOMPUTE routed L1 (m_grouped)                192 us    1251 TFLOPS
   RECOMPUTE shared L1 (dense fp8)                 91 us    1323 TFLOPS
   SHADOW dgrad L2 (m_grouped)                     99 us    1209 TFLOPS
   SHADOW wgrad L2 (k_grouped)                    499 us     241 TFLOPS
   RECOMPUTE 合计      283 us   SHADOW 合计      599 us   RECOMPUTE/SHADOW = 0.47x

T=4096  H=7168 I=2048 E=8 topk=2 Es=1  m_total=8192 total_k=8192
   RECOMPUTE routed L1 (m_grouped)                357 us    1348 TFLOPS
   RECOMPUTE shared L1 (dense fp8)                174 us    1378 TFLOPS
   SHADOW dgrad L2 (m_grouped)                    193 us    1243 TFLOPS
   SHADOW wgrad L2 (k_grouped)                    536 us     449 TFLOPS
   RECOMPUTE 合计      531 us   SHADOW 合计      729 us   RECOMPUTE/SHADOW = 0.73x
```

- **重算的时间预算 < shadow**（T=2048 `0.47×`、T=4096 `0.73×`）。重算是一次高算术强度的
  `m_grouped` GEMM（1250–1380 TFLOPS，接近 H100 FP8 上限），而 shadow 里 `wgrad L2` 因收缩维
  `K=token` 偏小（`k=512`/组）是访存受限的（241 TFLOPS）——**一次计算受限的 GEMM 藏在一堆
  访存受限 GEMM 后面，预算上绰绰有余**。
- **注意这是「预算」不是「实测重叠」**：DeepGEMM 单流顺序发射，要真正重叠需并发机制。
  第 148 篇（P138-1）证侧流 overlap 被 DeepGEMM 的 `232448 B/SM`（`heuristics/sm90.hpp:14`）
  smem 预算堵死（消费者 block 无从 co-reside，比值恒 ~0.58×）；第 167 篇（P157-1）证**单 kernel
  融合要计算腿是异步张量核**（CUDA-core 腿只重叠 30–60%，换 WGMMA 后 0.62→0.93）。所以
  「重算可藏」成立的前提，和 Mega MoE 融合本身的前提是同一条：**异步张量核 + 单 kernel 化**。

---

## 5. 与前面各篇的关系

- **第 10 篇**（SM100 主 kernel epilogue）解释了融合前向**为什么**不存 `h`——`h` 是 TMEM 里的
  中间量，epilogue 原地 SwiGLU + 量化后即丢弃（`sm100_fp8_fp4_mega_moe.cuh:1108-1134`）。
- **第 144 篇**（SwiGLU 反向重量化）把反向这段的数值（bf16-round ~1.7e-3、+FP8 ~2.7e-2）量了，
  但**输入 `h` 从哪来**没回答；本篇回答：**重算，且逐位无损**。
- **第 145/156 篇**试图把 SwiGLU 反向折进 dgrad 的 epilogue——那条路能省一次读写，但**前提**正是
  「前向不单独存 gate/up」；本篇从另一侧量了同一个前提的代价（重算一次 L1 GEMM）。
- **第 159 篇**的训练步账里前向 L1 是一整段 GEMM；本篇说明反向还要**再付一次**同样的 GEMM
  （重算），但它有 0 显存回报、且时间预算可藏。

---

## 6. 结论 / 实务准则

1. **RECOMPUTE 支配 STORE-ALL**：额外显存 **0**、数值**逐位相等**（同一确定 kernel）、
   时间预算 `0.47–0.73×` 的 shadow。存 `h` 的唯一理由是**回避并发**（单流下重算就是纯加时）。
2. **STORE-G1 不划算**：省一半显存，但 `G2 = a / silu(G1)` 反推在 FP8 存储下误差 `4.3e-1`，
   除非 `a` 也按 fp32 存（那就不省了）。
3. **重算的输入无需额外准备**：`x`（fp8）为 wgrad 已存、`W1` 常驻；重算只是「多一次 linear1
   GEMM」。这正是 activation checkpointing 在 MoE 上的干净形态——**代价是一次 GEMM，收益是
   `h` 的全部显存**，且数值无损。
4. **落地前提与融合同源**：要把这份 shadow 预算兑现成真重叠，需要**异步张量核 + 单 kernel**
   （第 148/167 篇），这恰好也是 Mega MoE 本身成立的条件。

---

## 7. 边界与「仅分析」

- **SM100 专属、仅分析**：Mega MoE 融合主 kernel（`sm100_fp8_fp4_mega_moe.cuh`）的 UMMA/tcgen05/
  TMEM/in-place SwiGLU 与 buffer 布局（`layout/mega_moe.cuh`）是 SM100 专属，本机 H100 跑不了；
  本文只取其**架构无关的语义与布局事实**（`h` 不落地、`l1/l2_token_buffer` 无 `h`）。
- **数值实验全部 H100 真机**：用真实 SM90 `m_grouped_fp8_gemm_nt_contiguous`（forward L1 / dgrad）
  与 `k_grouped_fp8_gemm_nt_contiguous`（wgrad），逐位性与性能均为实测。
- **环境注记**：8 卡节点共享。Part C 绑到空闲 GPU1 后，`m_grouped`/`dense` 段稳定在 1200–1380
  TFLOPS；未绑核时同一段曾量出 215–363 TFLOPS（~4× 争用）并使 T=4096 的阴影比在
  `0.84–1.26×` 间抖动——**定性结论（重算预算 < shadow）在无争用下稳定**。
- 目标卡代号敏感：全文用「目标卡」指代。

**实验**：`harness/exp_moe_activation_recompute.{py,sh,out.txt}`。
