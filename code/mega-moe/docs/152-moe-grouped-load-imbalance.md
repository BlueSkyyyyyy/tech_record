# MoE grouped GEMM 在专家负载不均下的计算侧代价：grouped kernel 为什么几乎免疫

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P142 增量（第 152 篇）。
> MoE 的 dgrad / 前向 linear2 都是 **M 轴分组的 grouped GEMM**：不等长的是「每个专家收到的 token 数」，
> 落在 M 上，N/K 固定。第 4/13 篇讲了三种布局（contiguous / masked / PSUM）的**语义**，
> 第 108/120 篇拆了 grouped 调度器与 Mega MoE 任务调度器，第 129 篇讲了**路由层**的负载均衡
> （aux-loss / z-loss / bias），第 142 篇量了**均衡**分布下的前向 roofline。但一个最实际的问题
> **从没被量过**：**当 token 分布偏斜（专家冷热不均）时，grouped GEMM 到底慢多少、慢在哪里？**
> 是不是「Mega MoE 的负载均衡主要是为了让 GEMM 不空等」？
>
> **结论先行**：**不是。** DeepGEMM 的 grouped GEMM 对负载不均**几乎免疫**——H100 真机，
> G=8、偏斜到 `max_m = 3.75× 均值`（Dirichlet α=1）时，contiguous 只从 190us→201us（**+6%**）、
> masked 188us→214us（**+14%**）；G=32 极端「一个专家吃 60% token」时 contiguous 只 +13%。
> **原因**：调度器把所有专家的 M 块汇入**同一个 persistent grid**，偏斜只产生
> 「每段向上取整到 BLOCK_M」的舍入税（`Σalign(m_g,B)/M − 1`），**不产生「每专家各自一波」的尾波**。
> 真正的代价落在**不用 grouped kernel 的朴素做法**上：把每个专家都补到 `max_m` 的稠密 bmm，
> 计算量正比于 `G·max_m`，同样偏斜下比 masked 慢 **1.6×→4.9×（G=8）**、极端下 **20.9×（G=32）**。
> 另外一次性成本是**显存**：masked 要预留 `[G,max_m,·]`（极端 19.5× 有效行），contiguous 只要 `Σalign`。
>
> **可跑边界**：本篇主 kernel 用的都是 **SM90 真实 DeepGEMM kernel**（`m_grouped_*` FP8 1D2D），
> **H100 上真机实测**；Mega MoE 主 kernel 本身 SM100 专属、仅分析。
> 实验 `harness/exp_grouped_skew.{py,sh,out.txt}`。源码坐标：DeepGEMM commit `057ca59`（v2.8.1）。
> 目标卡代号敏感：全文用「目标卡」。

---

## 0. 结论先行

1. **grouped GEMM 对负载不均几乎免疫**（H100 真机，`G=8, M=8192, n=2048, k=7168`）：
   均匀 → 最大偏斜（`max_m` 1024→3840），contiguous **190→201us（+6%）**、masked **188→214us（+14%）**；
   G=32 的「一个专家吃 60%」contiguous 也只 **239→269us（+13%）**。
2. **机制**：调度器（`scheduler/gemm.cuh`）把**所有专家的 M 块汇入一个 persistent grid**——
   contiguous 直接 `num_blocks = num_m_blocks·num_n_blocks` 当一池（`:92-93`）、masked 按组
   `ceil(m_g/BLOCK_M)` 顺序追加（`:188-204`）。于是偏斜只付「**每段向上取整到 BLOCK_M**」的舍入，
   不产生「每个专家一波、尾波叠加」。实测时间正好落在 `Σalign(m_g,128)/M`（1.00→1.08）上。
3. **真正的代价在朴素做法**：把每专家补到 `max_m` 的**稠密 bmm**（算满 `G·max_m`），
   `naive/masked` 从均衡 1.6× 涨到 `max_m=3840` 的 **4.9×**（G=8），G=32「一个专家 60%」达 **20.9×**。
   ——grouped kernel 的价值不是「均衡时才快」，而是「**偏斜时也不慢**」。
4. **一眼代价是显存**：masked 预留 `[G,max_m,·]`（`G·max_m/M` 极端 **19.5×** 有效行），
   但其**计算**只认领 `ceil(m_g/BLOCK_M)` 块、padding 行不算——所以**存储大 ≠ 计算慢**。
   contiguous 物理行 `Σalign(m_g,128)`（≈有效行 + 至多几个 %）。
5. **所以训练用 contiguous、解码用 masked**（第 13 篇）：负载均衡的收益主要在
   **通信侧**（重专家 rank 的 dispatch/combine/EPLB，第 1/53/104/139 篇）与**显存容量**，
   不在「让 grouped GEMM 不空等」——后者本来就几乎不为偏斜所动。

---

## 1. 形状与上下文：dgrad 就是 M 轴分组

以 MoE 的 linear2 反向（对激活的梯度 dgrad）为例，这是第 13 篇的标准形状：

- `A = dY` `[M, K]`，`K = H`（hidden，7168）；主序 K-major；
- `B = W2` `[G, N, K]`，`N = I`（intermediate，2048）；
- `D = dA` `[M, N]`，bf16、N-major；
- 每个专家 `e` 拿到的 token 数 `m_e` 不等，`Σ m_e = M`。

DeepGEMM 的两个 host 入口（`csrc/apis/gemm.hpp`）分别对应两种把 `m_e` 编码进布局的方式：

| 布局 | 入口 | 形状契约 | 不等长怎么表达 |
|---|---|---|---|
| contiguous | `m_grouped_fp8_fp4_gemm_nt_contiguous`（`:181-250`） | `[M,K] @ [G,N,K].mT`（`:194`）、D bf16（`:209`） | 拼接 + 每段对齐 128 + gap 行 `group=-1`；`grouped_layout` 长度 = 物理 M（`:217-218`） |
| masked | `m_grouped_fp8_fp4_gemm_nt_masked`（`:269-318`） | `[G,M,K] @ [G,N,K].mT`（`:280`）、D bf16（`:295`） | 稠密 `[G, max_m, ·]` + device 端 `masked_m`（`:290`） |

`m_grouped_fp8_fp4_gemm_nn_contiguous`（`:252-267`）只是把 B 转置后转调 nt（第 46 篇），
与负载不均无关。SM90 分派：contiguous → `sm90_m_grouped_fp8_gemm_contiguous_1d2d`
（`gemm.hpp:239`，实现 `sm90_fp8_gemm_1d2d.hpp:139`）；masked → `sm90_m_grouped_fp8_gemm_masked_1d2d`
（`gemm.hpp:309`，实现 `:216`）。SM100 走 1D1D（`:243`/`:312`），**仅分析**。

---

## 2. 源码：调度器怎么让偏斜「不产生尾波」

关键在 `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` 的 `Scheduler`。它把「M 块 × N 块」的
二维工作摊到一个 **persistent grid**（`grid_dim = num_sms`，`sm90_fp8_gemm_1d2d.hpp:197-207`）上，
worker 用 `next_block_idx = (++current_iter)*kNumSMs + blockIdx.x` 领取（`:186`）。

**contiguous：一个 M 池。** 构造时 `num_blocks = num_m_blocks * num_n_blocks`（`:92-93`），
把整段物理 M（含每专家末尾的 128 对齐 gap）当成一个连续池。取址只按块首行定组：

- `get_global_idx`（`:147-156`）：`offset = grouped_layout[m_block_idx*BLOCK_M]`（`:152`），
  即「用块首行的组号代表整块」；
- `is_computation_valid`（`:299-300`）：`grouped_layout[m_offset + m_block_idx*BLOCK_M] >= 0`，
  整块落在 gap 里就跳过。

因为 gap 恰好对齐到 `mk_alignment = 128`（= BLOCK_M），**gap 永远是整块**，于是 contiguous
实际计算的行数 = `Σ align(m_g, 128)`，没有「半块浪费」。

**masked：按组顺序追加。** 构造时不设 `num_blocks`（`:95`），而是在 `get_next_block` 里顺序扫组：

```
num_m_blocks = ceil_div(masked_m[current_group_idx], BLOCK_M)   // :195
```

每组只贡献 `ceil(m_g/BLOCK_M)` 个 M 块（`:188-204`），`is_computation_valid` 用
`m_offset + m_block_idx*BLOCK_M < masked_m[g]`（`:301-302`）把最后一块的越界行掩掉。
**调度层看到的块数 ≈ Σ ceil(m_g/BLOCK_M)，与 max_m 无关**。

**两条路径的共同点**：都只付「**每段向上取整**」的税，不付「每专家一波」的尾波——
因为所有块都在同一个 persistent grid 里被全局 worker 抢占式地消费，冷专家的少数块会被
热专家的块「填缝」。这就是偏斜免疫的结构性原因。

**一个 SM90 特有的细节**：`get_swizzled_block_idx`（`:109-143`）为 L2 复用做 2-CTA 多播，
但当某「组」的块数为奇数时，SM90 要动态拆一次多播（`:123-131`）：

```cpp
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) { ... }
#endif
```

均匀分布时各专家块数整齐（易对齐多播），偏斜时块数参差、触发这条拆分——这是 masked
比 contiguous 多出的那点点差异（1.05–1.08×）的一个来源，但量级很小，远不是主导。

---

## 3. H100 实测 A：偏斜扫描

`G=8`、`n=2048`、`k=7168`、`M_total=8192`（平均每专家 1024）。偏斜用 Dirichlet 集中度参数 α
（α→∞ 即精确均衡，α 越小越偏斜，固定 seed 可复现）。时间取 **纯 device kernel 时长**
（`torch.profiler` 抓 `sm90_fp8_gemm_1d2d_impl`），三档为
contiguous / masked / naive（稠密按专家 `torch.bmm`，算满 `G·max_m`）：

```
            case  max_m   Σal/M  G·max/M   t_cont   t_mask  t_naive  cont_TF  mask_TF naive_TF mask/cont naive/mask
             Inf   1024   1.000    1.000     190u     188u     310u     1267     1276      777     0.99      1.64
            10.0   1792   1.062    1.750     196u     208u     479u     1229     1154      502     1.06      2.30
             3.0   2432   1.062    2.375     197u     209u     740u     1219     1152      325     1.06      3.54
             1.0   3840   1.062    3.750     201u     214u    1053u     1199     1125      228     1.07      4.93
             0.5   3072   1.078    3.000     197u     211u     825u     1219     1138      292     1.07      3.90
             0.3   2688   1.078    2.625     197u     213u     765u     1220     1129      314     1.08      3.59
```

读法：

- **contiguous**：190→201us，`Σal/M` 从 1.000 涨到 1.062，`cont_TF` 从 1267 掉到 1199（−5%）。
  **时间几乎跟着 `Σal/M` 走**，这就是「每段对齐 128」的全部税。
- **masked**：188→214us，`mask/cont` 1.06–1.08，同样跟着 `Σal/M`；`mask_TF` ~1125–1154。
  均衡时 masked 甚至与 contiguous 打平（0.99×）。
- **naive**：`310→1053us`，**正比于 `G·max_m/M`**（1.0→3.75），`naive/masked` 从 1.64× 涨到 4.93×。
  均衡时它只慢 1.6×，偏斜时慢 5×——**这正是「统一用 grouped kernel」的价值**。

三种布局都在 1200 TFLOPS 上下（`n=2048, k=7168` 的 FP8 1D2D 计算受限区间），
偏斜让有效 FLOPs 利用率最多掉 5–12%。

---

## 4. 记帐：为什么 masked 存储 19.5× 却不慢

看表里的 `G·max/M` 列：偏斜到 `max_m=3840` 时 masked 要预留 **3.75×** 有效行的存储；
G=32「一个专家 60%」时 **19.5×**。但它的**计算**只认领 `ceil(m_g/BLOCK_M)` 块（§2），
padding 行既不进 MMA 也不出 store。于是：

| 量 | 公式 | 偏斜时 |
|---|---|---|
| contiguous 物理行 / 计算行 | `Σ align(m_g, 128)` | 1.00–1.08×（≈有效行） |
| masked **存储**行 | `G · max_m` | 1.00–19.5×（大） |
| masked **计算**行 | `Σ ceil(m_g, BLOCK_M)` | ≈1.00–1.08×（同 contiguous） |
| naive **计算**行 | `G · max_m` | 1.00–19.5×（大） |

**结论**：`G·max_m` 是「**预留/寻址空间**」，不是「**计算量**」——DeepGEMM 的 masked 调度器把
前者和后者解耦了（`scheduler/gemm.cuh:188-204/301-302`）。所以「masked 因为补齐到 max_m 而慢」
是个常见误解；至少在 SM90 1D2D 上，masked 只比 contiguous 慢几个百分点。

---

## 5. 正确性：masked 有效行 ≡ contiguous 有效行

同一份 `ms`（最偏斜的 α=0.3：`[1, 64, 1607, 2633, 674, 533, 236, 2445]`）走两种布局：

```
  masked valid vs contiguous valid : calc_diff=0.000e+00  bitwise=True
  contiguous vs bf16 ref (FP8 量化噪声): relerr=3.688e-02
```

- 物理行：contiguous `8832`（`Σal`）、masked 存储 `G·max_m = 21504`，有效行 `8193`；
- 按组内顺序拉平后，masked 有效行与 contiguous 有效行 **`torch.equal` 逐位为 True**；
- vs bf16 参考的 `3.69e-2` 正是 FP8 量化噪声层（第 13 篇同形状 3.67–3.70e-2）。

即两种布局是**同一数学的两种边界表示**，负载不均不引入任何数值差异。

---

## 6. G=32 与极端分布

更接近单 rank 本地专家数的 `G=32`（`M_total=8192`，平均 256）：

```
            case  max_m   Σal/M  G·max/M   t_cont   t_mask  t_naive  cont_TF  mask_TF naive_TF mask/cont naive/mask
        balanced    256   1.000    1.000     239u     279u     384u     1008      861      627     1.17      1.37
  one-heavy(60%)   4992   1.094   19.500     269u     273u    5702u      893      882       42     1.01     20.92
   power-law 1/r   2048   1.203    8.000     278u     286u    2339u      866      842      103     1.03      8.19
```

- **contiguous 依然平坦**：239→278us（+16%），与 `Σal/M` 1.00→1.20 相称；
- **masked**：均衡时 279us（1.17×，块数多、SM90 多播对齐差使然），**偏斜反而追平**（1.01–1.03×）；
- **naive 爆炸**：`one-heavy` 的 `max_m=4992` 让稠密 bmm 达 **5702us = masked 的 20.9×**；
  power-law 也有 8.2×。极端不均下，grouped kernel 是**唯一**可行的做法。

一个反直觉点：G=32 均衡时 masked 比 contiguous 慢 1.17×，但偏斜时两者打平。
因为均衡时每组的块数（`256/128=2`）很少、多播/调度的固定开销占比高；偏斜时某些组块数变多，
反而摊薄了固定开销。**「均衡」不总是 masked 的最优工况。**

---

## 7. host 编排开销（E）

除了 device kernel，masked 的 host 路径每次调用都跑 `get_best_config` +
`transform_sf_pair_into_required_layout`（SM90 走 `transpose_fp32`），而 SFA 的形状随 `max_m` 变。
在空闲 GPU 上量「composite（event 计时，含 host）− 纯 device」：

```
      case  max_m  cont_dev  cont_evt  mask_dev  mask_evt mask_host cont_host
  balanced   1024      201u      200u      191u      197u        6u       -1u
      skew   3840      199u      203u      214u      221u        7u        4u
```

host 开销 ~6–7us，相对 ~200us 的 kernel 只有 3%，**本形状下可忽略**（若在解码小 batch、
CUDA graph 场景会放大，第 109 篇）。所以「masked 因 max_m 大而 host 慢」在本量级也不成立。

> **测量注记（重要）**：本机是 8 卡共享节点。最初在一张**被别人跑满（util 100%）**的卡上量时，
> masked 的偏斜竟出现 1.58× 的假象（时钟/争用噪声）。绑到空闲卡后稳定在 1.06–1.08×。
> `exp_grouped_skew.sh` 自动选 `utilization.gpu` 最低的卡并在输出里记录卡号。这与「已知坑」里
> 「8 卡共享节点，通信/带宽测量是受争用下界」一致——**本篇所有数字都绑定空闲 GPU。**

---

## 8. 与专题其余篇的关系、边界与结论

- 与**第 13 篇**（dgrad 三布局语义）：本篇补上它没量过的「**偏斜性能**」维度，
  并坐实「masked 只认领 `ceil(m_g/BLOCK_M)` 块」在性能上确实兑现。
- 与**第 108/120 篇**（grouped / Mega MoE 调度器）：本篇量的是同一套调度器在 M 轴不均下的表现；
  Mega MoE 主 kernel 用「任务池 + FIFO」把（专家, M 块, N 块）摊到 SM（第 120 篇），
  与本篇的 persistent grid 是同一思想，故**通信-计算重叠**里的计算腿同样对路由不均不敏感。
- 与**第 129/139 篇**（路由负载均衡）：那三件套的目标是**让通信与显存均衡**，不是救 GEMM——
  GEMM 本来就不为偏斜所动（本篇）。这解释了一个工程直觉：**aux-loss 的价值要更多地从
  通信/EPLB/显存去算，而不是从 GEMM 时间去算**。
- 与**第 142/143 篇**（前向 roofline / combine）：那两篇在均衡分布下量计算腿，本篇是它的
  「不均对偶」；结论是计算腿对不均稳健，所以「计算 54% / 通信+归约 46%」的划分不会因路由波动而崩。

**边界**：① 主 kernel 用的是 **SM90 1D2D**（H100 可跑、真机数）；SM100 的 1D1D 走
`sm100_m_grouped_fp8_fp4_gemm_{contiguous,masked}_1d1d`（`gemm.hpp:243/312`）、**仅分析**，
其调度器语义（`get_global_idx`/`is_computation_valid`）与 SM90 共用同一份 `scheduler/gemm.cuh`，
故定性结论应一致；② 本篇量的是**单算子**，未把 dispatch/combine 的通信不均纳入（见第 104 篇）；
③ `M_total=8192` 是 DeepSeek 单卡的典型 token 量级，更大 M 会让固定舍入税占比更小。

**一句话**：**grouped GEMM 不是负载均衡的痛点，朴素稠密做法才是；偏斜的真实代价是显存
（`G·max_m`）与通信（重专家 rank），计算侧只付「每段对齐 128」的零头。**

---

## 9. 复现与引用

- 实验：`harness/exp_grouped_skew.py`、`harness/exp_grouped_skew.sh`、`harness/exp_grouped_skew.out.txt`
  （H100 sm90，DeepGEMM 2.8.1，Python 计时用 `torch.profiler` 取纯 device kernel）。
- 源码（DeepGEMM commit `057ca59`，v2.8.1）：
  - `csrc/apis/gemm.hpp:181-250`（contiguous 入口、`:194/209/217-218` 契约、`:239` SM90 分派）、
    `:252-267`（nn 壳）、`:269-318`（masked 入口、`:280/290/295` 契约、`:309` SM90 分派）；
  - `deep_gemm/include/deep_gemm/scheduler/gemm.cuh:92-95`（num_blocks 设置）、
    `:147-156`（`get_global_idx`）、`:188-204`（masked 按组 `ceil(m_g/BLOCK_M)`）、
    `:296-309`（`is_computation_valid`）、`:109-143`（swizzle + `:123-131` SM90 奇数块多播修正）；
  - `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:139-214`（contiguous 发射）、`:216-283`（masked 发射）。
- 相关博客：`content/posts/mega-moe-152-moe-grouped-load-imbalance`。
- 专题内引用：第 04/13 篇（dgrad 布局与调度）、第 108 篇（`scheduler/gemm.cuh` 逐段）、
  第 120 篇（Mega MoE 任务调度器）、第 129/139 篇（路由负载均衡三件套）、
  第 142/143 篇（前向 roofline / combine 归约）、第 104 篇（反向通信量）。
