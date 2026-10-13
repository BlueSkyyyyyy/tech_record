# Mega mHC 反向的目标卡单 kernel 接线（下）：真实 H=7168 的前向 Post worker 网格与 G0 累加器直喂 K1

> 第 130 篇给了 mHC 反向的**数学**（残差多流混合、Sinkhorn 迭代 VJP、`g` 的耦合项、RMSNorm），
> 第 131 篇给了 **roofline 账**（六段、访存受限、`T=16384` 访存下界 3649us vs GEMM 计算下界 68us），
> 第 132 篇把「per-token 融合 worker」在 H100 上**真跑**了一遍（一 block 一 token，K1+G1a+K2 全在片上）。
>
> 但第 132 篇的融合 worker 仍是**粗粒度**的（`threadIdx.x = h`，一个 block 一 token，块内 `block_sum` 归约），
> 它答的是「省掉多少中间张量」，**没有**回答「目标卡上这一段的 lane 到底谁拥有哪几行、归约在哪几条 lane 之间做」。
> 本篇把这条接线的**几何**钉死：目标卡上 mHC 反向的 K1（系数反向）必须复用**前向 Post worker 的
> lane-ownership 网格**，才能与前向共享同一套 TMEM 累加器寻址与 barrier 协议。本篇在 H100(sm90) 上
> **直接 include 真实头文件**，把该网格与归约路径逐位对拍，并给出 TMEM 列预算与分段 worker 的真实吞吐。
>
> **主 kernel（UMMA/TMEM/tcgen05）是 SM100 专属，仅分析**；本篇 H100 实测的是**架构无关的几何、
> 归约与元素级吞吐**。

---

## 1. 为什么「几何」是这条接线的关键

第 132 篇的结论里有一句容易被跳过的话：**融合边界两条判据**——① 必须跨 token 归约的张量（G1b 的
`dfn` wgrad）落全局；② 必须喂独立 GEMM 的张量（`du0`）落全局；其余 `dAbase[T,D]`、G1a 的 `dz[T,D]`
可留片上。

要把「可留片上」真正落地，就得让**反向的 K1 工作分配与前向 Post worker 完全一致**，理由有三：

1. **同一张 TMEM 累加器**。前向的系数投影是**一条 TF32 TS UMMA**，累加器 `u0[BLOCK_M, kNumHCOutputs]`
   就落在 TMEM 里（`impls/sm100_mega_mhc.cuh:235-238` 的类型、`:273-275` 的 `fma`）。反向的 G0
   （`u0 = A @ fnᵀ` 的 dgrad）产出**同样的形状**的累加器；若让 K1 直接从 TMEM 读它，就省掉了
   `u0` 的 `global` 往返，也省掉了「哪个 warp 该读哪几列」的二次映射。而 TMEM 的
   `SM100_TMEM_LOAD_*` 指令是按 **32-dp 行 / 列块**寻址的，**行与列的归属必须与写入端一致**。
2. **同一套 barrier 协议**。前向 Post 的逐行平方和统计经 `hc_norm_sqr_sums[2][BLOCK_M]` 汇总
   （`layout/mega_mhc.cuh:138`），两个 Post worker 各写一半、由 WG3 相加（`impls:639-640`）。
   反向的 RMSNorm 反向要复用**完全相同**的 `q` 归一化（`impls:501` 的 `q=ΣA²`），若 lan 归属不同，
   `q` 会在不同的归约树上算出、与前向的 `g` 不一致。
3. **同一组 warpgroup 角色**。前向 6 个 warpgroup 是静态角色（`impls:78-85`）；反向若复用 Post worker，
   就不必新增角色、不必重新分配寄存器预算（`impls:88-99`）。

所以「几何」不是实现细节，而是**数值一致性的前提**：`g = rsqrt(q/D + eps)` 里的 `q` 由谁求和、
按什么顺序求和，决定了反向与前向是否落在同一个定点上。

---

## 2. 前向 Post worker 的真实 lane-ownership 网格（真实行号）

前向 kernel 的几何常量与派生（**全部逐字取自源码**）：

| 量 | 值 | 来源 |
|---|---|---|
| `kNumRoutes` | 4 | `layout/mega_mhc.cuh:11` |
| `kNumHCOutputs = kNumRoutes·(kNumRoutes+2)` | 24 | `layout/mega_mhc.cuh:12` |
| `BLOCK_M` / `BLOCK_N` / `BLOCK_K` | 64 / 24 / 64 | `layout/mega_mhc.cuh:19-21` |
| `kNumAStages` / `kNumAccumStages` | 4 / 2 | `layout/mega_mhc.cuh:27-28` |
| `kNumWarpsPerWG` | 4 | `layout/mega_mhc.cuh:30` |
| `UMMA_K = 32/sizeof(float)` | 8 | `impls/sm100_mega_mhc.cuh:30` |
| `kNumTokenRowsPerLane` | 2 | `impls:31` |
| `kNumPostWGs` | 2 | `impls:56` |
| `kNumRowsPerWarp = BLOCK_M / kNumWarpsPerWG` | 16 | `impls:108` |
| `kNumRowGroupsPerWarp = kNumRowsPerWarp / kNumTokenRowsPerLane` | 8 | `impls:109` |
| `kNumLanesPerRowGroup = 32 / kNumRowGroupsPerWarp` | 4 | `impls:110` |

由此得到两条**归属规则**（kernel 内联函数 `get_first_row_idx` 与 `hidden_pair_idx`）：

- **行归属**（`impls:111-113`）：lane `(warp w, lane l)` 拥有两行
  `first = (w mod 4)·16 + ⌊l/4⌋` 与 `first + 8`。于是 `l/4 ∈ {0..7}` 把 32 条 lane 分成 8 个
  **row group**（每组 4 条连续 lane），每组负责 `first` 与 `first+8` 两行；4 个 warp × 8 组 × 2 行
  = 64 行 = `BLOCK_M`。
- **hidden 归属**（`impls:460-461`）：`hidden_pair_idx = umma_k_idx·(UMMA_K/2) + col_pair_idx·(UMMA_K/kNumLanesPerRowGroup/2)`，
  其中 `col_pair_idx = lane & (kNumLanesPerRowGroup-1) = l mod 4`。因为
  `UMMA_K/kNumLanesPerRowGroup/2 = 8/4/2 = 1`，所以**一个 UMMA 原子内 4 条 `l mod 4` 不同的 lane
  恰好覆盖该原子的 4 个浮点对（8 个 float）**——`impls:401` 的 `DG_STATIC_ASSERT(UMMA_K % kNumLanesPerRowGroup == 0)`
  正是这条前提。

关键的一点：**两个 Post worker 不按行切，而按 UMMA 原子切**。每个 worker 处理
`umma_k_idx = worker_idx, worker_idx+2, worker_idx+4, ...`（`impls:458-459`），
**每个 worker 都覆盖全部 64 行**，各自对一半的 hidden 原子求和；两者的部分和最后由 WG3 相加
（`impls:637-641`）。所以「一个工人管一半行」的直觉在这里是**错的**——这正是要用真机对拍钉死的原因。

**逐行归约路径**（`impls:545-546`）：每条 lane 先把自己 2 行的「浮点对两半相加」得到一个标量，
再用 `math::warp_reduce_sum<kNumLanesPerRowGroup=4>` 在 **row group 内 4 条连续 lane** 上求和，
写入 `smem.hc_norm_sqr_sums[worker][row]`（`impls:553-554`）；最后 WG3 把两个 worker 的值相加
（`impls:639-640`）。`warp_reduce_sum<4>` 在实现上只做 `xor 2` 与 `xor 1` 两步
（`common/math.cuh:217-218`），即**只在 4 条连续 lane 内归约**，不会跨 row group。

---

## 3. 反向六段 → 前向机器的对偶接线

把第 131 篇的六段映射到前向 6 个 warpgroup 的静态角色（`impls:78-85`）：

| 反向段 | 内容 | 目标卡落点 | 前向对偶（行号） |
|---|---|---|---|
| K0 | 重算残差混合 `A` | Post worker | `impls:482-508`（残差多流混合 + bf16 round `:497`） |
| G0 | `u0 = A·fnᵀ`（dgrad） | **同一条 TF32 TS UMMA** | `impls:235-238/273-275`（A 在 TMEM、B=`fn` 在 SMEM） |
| K1 | 系数反向 + Sinkhorn VJP + RMSNorm bwd | Post worker + 系数 epilogue | `impls:545-554` + `epilogue/sm100_mega_mhc.cuh:17-101` |
| G1a | `dz = du0·fn`（dgrad） | Normal worker 的 per-k 累加 | `epilogue:188-244`（RMSNorm 融合 worker） |
| G1b | `dfn = du0ᵀ·A`（wgrad） | **落全局**（跨 token 归约） | — |
| K2 | 残差混合反向 | Post worker 反向 | `impls:482-508`（同一条链逆放） |

**G0 累加器直喂 K1 的接线**（本篇的核心）：

1. **TMEM 列预算**。前向的累加器从 `kAccumTmemStartColumn = kNumAStages·kNumRoutes·UMMA_K = 4·4·8 = 128`
   列开始（`impls:64-65`），占 `kNumAccumStages·BLOCK_N = 2·24 = 48` 列，整块对齐到
   `utils::get_num_aligned_tmem_cols<176>() = 256`（`impls:66-67`、`common/utils.cuh:43-50`）。
   反向的 G0 累加器只需 `BLOCK_N = 24` 列，对齐到 **32 列**。两者之和 **288 < 512**，即
   **前向 A 双缓冲（128 列）+ 反向 G0 累加器（32 列）可以在同一块 TMEM 里共存**——
   这是「G0 直喂 K1」可行的**硬前提**（若前向已把 512 列用满就没地方放 G0 了）。
2. **同一张 lane 网格**。K1 读 `u0` 时用 `SM100_TMEM_LOAD_16dp256b2x`（前向 WG3 就是这么读累加器的，
   `impls:602-606`），按 128 条 math 线程 / 32-dp 行块寻址。只要 K1 用与 Post worker **相同的
   `first_row_idx` 与 `col_pair_idx`**，读回的 24 列就天然落在「每行 4 条 lane」的 row group 里，
   与 RMSNorm 反向所需的 `q` 归约树一致。
3. **barrier**：前向的 `full_stats_barrier` / `empty_stats_barrier` 计数是
   `kNumPostWarps=8` 与 `kNumWarpsPerWG + (kIsShifted?1:0)`（`impls:159-160`）。反向复用同一套，
   只需把「谁到达」从 Post 改成 Post+K1。

`dz[T, D]`（`D = hc·H = 28672`）远超 512 列，**不能**驻留 TMEM，只能像前向 A 那样
按 `BLOCK_K=64` 的粒度**流式**（`impls:202-206` 的 per-k-block A stage）。这一点与第 131 篇
「G1a 的 `dz[T,D]` 可留片上」并不矛盾——留片上是指**不落 global**，仍要按 k-chunk 流过寄存器/SMEM。

---

## 4. H100 实测（真实头文件，架构无关）

> 实验：`harness/exp_mhc_backward_segment.sh`（编译运行 `src/mhc_backward_segment.cu`，
> 直接 include `layout/mega_mhc.cuh` / `common/math.cuh` / `common/utils.cuh`）
> + `harness/exp_mhc_backward_segment.py`（真实形状账目）。
> 原始输出：`harness/exp_mhc_backward_segment.out.txt`。设备 H100(sm90)。

### 4.1 几何核对（溯源到真实头文件，非手抄）

```
kNumRoutes=4 kNumHCOutputs=24 BLOCK_M=64 BLOCK_N=24 BLOCK_K=64 kNumWarpsPerWG=4
UMMA_K=8 kNumTokenRowsPerLane=2 kNumPostWGs=2
=> kNumRowsPerWarp=16 kNumRowGroupsPerWarp=8 kNumLanesPerRowGroup=4 kNumPostThreads=256
每行由 2 个 worker × 4 lane = 8 个 lane 负责；行覆盖错误=0 worker 覆盖错误=0  [OK]
一个 UMMA 原子内 4 lane 恰好覆盖 4 个浮点对（含 col_pair_idx 双射）: OK
```

两个 worker **各自覆盖全部 64 行**（每个 worker 的每行对应恰好 4 条 lane），与 §2 的结论一致。

### 4.2 分段归约 vs host double

设备 kernel `k_post_stats` 用真实 `math::warp_reduce_sum<4>` 复刻前向 Post 网格，逐行 `q` 与
host double 全行和对比；另做 `k_post_stats_wrong`（错用 `warp_reduce_sum<32>`）作负样本：

```
T= 256 H=  64 D=  256 | 正确网格 q vs double relerr=4.094e-08 | 错(全32-lane) relerr=2.475e+00  [OK]
T= 256 H= 128 D=  512 | 正确网格 q vs double relerr=4.609e-08 | 错(全32-lane) relerr=2.477e+00  [OK]
T= 512 H= 256 D= 1024 | 正确网格 q vs double relerr=5.102e-08 | 错(全32-lane) relerr=2.476e+00  [OK]
T=  64 H= 512 D= 2048 | 正确网格 q vs double relerr=5.874e-08 | 错(全32-lane) relerr=2.474e+00  [OK]
```

正确网格只是 fp32 累加噪声（rel 5e-8）；**全 32-lane 归约的 rel 是 2.47**——这不是数值误差，
而是**结构性错误**（把 8 个 row group 的值混成 1 个再广播），坐实「归约必须严格限制在 row group 内」。

### 4.3 TMEM 列预算

```
A 每 stage 列 = kNumRoutes*UMMA_K = 32；accum 起点 = kNumAStages*A = 128
前向 used = accum_start + kNumAccumStages*BLOCK_N = 176 -> 对齐 256
反向 G0 累加器 used = BLOCK_N = 24 -> 对齐 32
两者之和 288 <= 512: OK
```

### 4.4 分段网格 vs 朴素 worker 的真实吞吐

`k_post_stats`（256 线程/block，按前向网格）vs `k_coarse`（1 线程/行，串行扫全 `D`）：

```
T=  8192 H= 512 D= 2048  | 分段网格   77.7 us ( 863 GB/s) | 朴素  850.1 us ( 79 GB/s) | 10.93x
T= 16384 H= 512 D= 2048  | 分段网格  151.1 us ( 888 GB/s) | 朴素  828.0 us (162 GB/s) |  5.48x
T= 16384 H=1024 D= 4096  | 分段网格  352.2 us ( 762 GB/s) | 朴素 1699.1 us (158 GB/s) |  4.82x
T= 32768 H=1024 D= 4096  | 分段网格  817.2 us ( 657 GB/s) | 朴素 1659.0 us (324 GB/s) |  2.03x
```

分段网格在 H≤512 时达 **~860 GB/s**（该规模下访存受限），朴素版本受「每行一个线程、`D` 方向无并行」
拖累。真实形状 `H=7168, D=28672`（§5）：每个 Post worker 的原子数 = `H/UMMA_K/2 = 448`，
每 lane 每行每 route 累加 **3584 个 float**，×4 route = **14336 float/行**——足够长的向量化访存，
分段网格的带宽会更接近 HBM 上限。

---

## 5. 真实形状（H=7168 / hc=4）的账

```
atoms=H/UMMA_K=896  k_blocks=H/BLOCK_K=112  atoms/k-block=8
num_m_blocks=64  num_tasks=num_m_blocks*splits=1024   (T=4096, splits=16)
每个 Post worker 的原子数 = 448；每 lane 覆盖 4 浮点对/行/原子
每 lane 每行 每 route 累加 3584 floats；×4 routes = 14336 floats/行（一个 worker）
前向 WG3 的 gemm_partial = num_m_blocks*splits*BLOCK_M*NOUT = 1572864 elems = 6.291 MB
省掉的往返（写+读）= 12.583 MB
K1（含 G1a 折入）元素级 AI = 0.75 FLOP/B -> 访存受限
```

**要点**：

- `u0` 的 global scratch 只有 **6.3 MB**（`gemm_partial`），「直喂」省的字节（12.6 MB）相对
  `dfn = 2.75 MB` 等张量**并不大**。真正的收益是**省掉 `num_tasks=1024` 次 global round-trip 的
  **延迟**：每个 task 一个 split 的 24 列累加器，跨 WG 同步后立刻被同 CTA 的 K1 消费，
  不必等「写 global → 下一个 kernel 再读回」。
- **K1+G1a 的算术强度 AI≈0.75**（元素级），远低于 H100 FP32 拐点（~20 FLOP/B），
  与第 131 篇「整条反向 AI≈6.5 / 访存受限」的方向一致；目标卡上它应作为**元素级段**融进
  Post/Normal worker，而不是拆成独立 GEMM。

---

## 6. 边界与「仅分析」部分

- **主 kernel（UMMA/tcgen05/TMEM 寻址、`SM100_TMEM_LOAD_16dp256b2x`、`SM100_TMEM_STORE_16dp256b4x`、
  TF32 TS UMMA）是 SM100 专属，仅分析**（`impls/sm100_mega_mhc.cuh:52` 的 `__CUDA_ARCH__ >= 1000` 守卫；
  `mega_mhc.hpp:223` 的 `arch_major == 10` 断言）。本篇 H100 实测的是**架构无关的几何、归约、
  TMEM 列预算算术与元素级吞吐**。
- 「G0 累加器直喂 K1」的**完整接线**（同一 CTA 内 G0 的 UMMA commit → K1 的 TMEM_LOAD → barrier
  相位）需要在目标卡上实现，本篇给出的是**可核对的前提**（列预算 288<512、lane 网格与归约树一致、
  barrier 计数可复用），而非已跑通的 kernel。
- 第 132 篇的粗 worker 与本篇的分段网格是**互补**的：前者量化「省多少字节/时间」，本篇量化
  「目标卡上谁拥有哪几行、归约在哪几条 lane 之间、TMEM 够不够放」。两者合起来才是完整的接线图。

---

## 7. 参考坐标

- 前向主 kernel：`deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh`（几何 `:30-31/56/108-113`、
  TMEM `:64-67`、Post worker `:403-565`、`hidden_pair_idx:460-461`、归约 `:545-554`、
  UMMA `:235-238/269-276`、WG3 读取 `:602-606/639-641`）。
- 布局：`deep_gemm/include/deep_gemm/layout/mega_mhc.cuh`（`:11-12/19-21/27-28/30`、Workspace `:73-120`、
  `SharedStorage` `:122-168`）。
- 归约原语：`deep_gemm/include/deep_gemm/common/math.cuh`（`warp_reduce_sum:225-227`）、
  `deep_gemm/include/deep_gemm/common/utils.cuh`（`get_num_aligned_tmem_cols:43-50`）。
- 反向语义/融合账：本专题第 130 篇（`docs/130`）、第 131 篇（`docs/131`）、第 132 篇（`docs/132`）。
- 实验：`src/mhc_backward_segment.cu`、`harness/exp_mhc_backward_segment.{sh,py,out.txt}`。
- 目标卡代号敏感：文中一律写「目标卡」。Mega mHC 主 kernel 为 SM100 专属，H100 端到端跑不了。
