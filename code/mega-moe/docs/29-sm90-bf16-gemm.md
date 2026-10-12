# 29 · DeepGEMM 的 SM90 BF16 dense GEMM 主 kernel：`sm90_bf16_gemm` 精读 + H100 实测

> 前面 28 篇里，DeepGEMM 的 **FP8** SM90 kernel（`sm90_fp8_gemm_1d1d` / `1d2d`，见第 10/11/12/13/21 篇）
> 被反复拆解；**BF16** 的那条路只顺带出现过——第 17 篇（einsum 批归约）、第 18 篇（MLA 按头 batched）、
> 第 27 篇（cuBLASLt 回退）都在调用它，却没系统讲过。这一篇补上：DeepGEMM 的 **SM90 BF16 稠密 GEMM 主
> kernel** `sm90_bf16_gemm.cuh` 与它的六个 host 入口 `sm90_bf16_gemm.hpp`。
>
> 它是 DeepGEMM 里**唯一**一个同时服务 Normal / M-grouped(contiguous & masked) / K-grouped /
> Batched 五种 `GemmType` 的 SM90 kernel，也是**本机 H100 上可以真机跑**的一等公民。
>
> **本文所有行号均以 commit `057ca59`（Public release 26/09/30）核对过。**
> 配套实验：`harness/exp_bf16_sm90_gemm.py` / `.sh` / `.out.txt`（H100 真机）。

---

## 1. 它和 FP8 那条路有什么不一样

先给一句话结论：**FP8 kernel 把「block-scale（SF）」当一等公民，于是拆成 `1d1d`/`1d2d` 两套五脏俱全的
实现；BF16 没有 SF，退回成一套通用 kernel。** 具体差异：

| 维度 | FP8 `sm90_fp8_gemm_1d1d/1d2d` | BF16 `sm90_bf16_gemm` |
|---|---|---|
| SF | 有（per-1x128 / 1x128 × 128x128） | **无**（`KernelType::KernelNoSF`） |
| host 入口 | 按 A/B SF 粒度分 `1d1d`（token SF）/`1d2d`（block SF） | **一个** `sm90_bf16_gemm`，加别名/分组 wrapper |
| WGMMA 指令 | `MMA_64xNx32_F32E4M3E4M3_SS`（`K=32`） | `MMA_64xNx16_F32BF16BF16_SS`（`K=16`，`mma/sm90.cuh:91`） |
| `BLOCK_K` | 128（`128*8/8`） | **64**（`128*8/16`，见 P18-1 的 `get_pipeline_config`） |
| 覆盖 `GemmType` | 各自一份 | Normal / MGrouped{Contiguous,Masked} / KGroupedContiguous / Batched **共用** |

BF16 的 `BLOCK_K=64` 直接来自选核公式 `block_k = 128 * 8 / bits`（P18-1）：`8` 是 SSE 粒度字节数，
`bits=16`。所以 BF16 每个 K-block 只有 FP8 的一半，`shape_k` 又常常很大，**stage 数就会被选得很多**
（见 §4 的合并 stage 优化）。

还有一个结构上的关键点：**BF16 kernel 里 `WGMMA::M` 恒为 64**（`mma/sm90.cuh:89`）。`BLOCK_M` 可以
是 64/128/256，靠「多个 math warpgroup 分工 + wave」拼出来（§5），跟 FP8 用 `UMMA_M` 是一套思维。

---

## 2. 六个 host 入口（`sm90_bf16_gemm.hpp`）

`SM90BF16GemmRuntime`（`:13-75`）只有一件事：把 `GemmDesc`/`GemmConfig` 拼成模板实参，
JIT 编译 `sm90_bf16_gemm_impl<...>`（`:27-64`），再用 7 个运行期参数 launch（`:67-73`）：
`grouped_layout`、`(m,n,k)`、三张 TMA descriptor。六个 host 入口都走它：

| 入口 | 行号 | `GemmType` | major 约束 | 备注 |
|---|---|---|---|---|
| `sm90_bf16_gemm`（Normal） | `:77-130` | `Normal` | A/B 任意 K/MN | 稠密；`bf16_gemm_{nt,nn,tn,tt}` 都转调它 |
| `sm90_m_grouped_bf16_gemm_contiguous` | `:132-201` | `MGroupedContiguous`(`WithPsumLayout`) | A 必须 K-major，`k%64==0` | 不等长在 M；支持 PSUM |
| `sm90_bf16_m_grouped_gemm_masked` | `:203-262` | `MGroupedMasked` | A、B **都** K-major | `[G,max_m,·]` + `masked_m` |
| `sm90_bf16_k_grouped_gemm` | `:264-330` | `KGroupedContiguous` | A、B **都** MN-major，每组 `k%128==0` | wgrad（P4-2/P16-1） |
| `sm90_bf16_bhr_hdr_bhd` | `:332-384` | `Batched` | 3D TMA，`num_groups=h` | MLA 投影（P8-2/P18） |
| `sm90_bf16_bhd_hdr_bhr` | `:386-438` | `Batched` | A K-major、B MN-major | MLA 投影另一方向 |

注意四个 `bf16_gemm_{nt,nn,tn,tt}` 只是把参数**转置**后统一转调 `nt`（`gemm.hpp:544/554/564`），
所以它们共用同一个 kernel —— 本文实验里四者 relerr 完全一致（§6）。

`GemmType` 在 kernel 里主要影响三处：调度（P4-3）、A 的取址是否带 group 偏移（`:173`
`kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked`）、以及 epilogue 用 2D 还是 3D TMA store（§5.3）。

---

## 3. kernel 模板参数与物理坐标（`sm90_bf16_gemm.cuh`）

模板头（`:27-37`）21 个参数，顺序与 host `compile_and_launch` 一一对应：

```
kMajorA kMajorB SHAPE_M SHAPE_N SHAPE_K kNumGroups
BLOCK_M BLOCK_N BLOCK_K_ kSwizzleAMode kSwizzleBMode kSwizzleDMode
kNumStages_ kNumTMAThreads kNumMathThreads
kNumTMAMulticast kIsTMAMulticastOnA kNumSMs
kGemmType kWithAccumulation cd_dtype_t
```

H100 真机抓到的两个代表实参（`harness/exp_bf16_sm90_gemm.out.txt`）：

```
(4096,7168,7168,bf16,acc)  → ...1u, 256u,128u,64u, 128u,128u,128u, 3u,128u,256u, 2u,true, 132u,(GemmType)0,true,bfloat16_t
(4096,7168,7168,fp32)      → ...1u, 128u,160u,64u, 128u,128u,  0u, 4u,128u,256u, 2u,false,132u,(GemmType)0,false,  float
```

即：`BLOCK_M=256/128`、`BLOCK_N=128/160`、`BLOCK_K_=64`；**out=bf16 时 `kSwizzleDMode=128`，out=fp32
时 `=0`**（epilogue 两种走法，§5.3）；TMA 128 线程 + math 256 线程（=2 个 math warpgroup）；
cluster=2 且 `kIsTMAMulticastOnA` 相应为 true/false。

类型层面（`:64`）只允许 `cd_dtype_t ∈ {float, cutlass::bfloat16_t}`；`__launch_bounds__(kNumTMAThreads
+ kNumMathThreads, 1)`（`:38`）。

### smem 布局（`:71-109`）

```
[ smem_d: BLOCK_M*BLOCK_N*sizeof(cd_dtype_t) ]     // 对齐 1024B，供 STSM/回写
[ smem_a: kNumStages * BLOCK_M*BLOCK_K*2 ]         // 注意 BLOCK_K 是【合并后】的（§4）
[ smem_b: kNumStages * BLOCK_N*BLOCK_K*2 ]
[ full_barriers[kNumStages]; empty_barriers[kNumStages] ]
```

`:77-78` 有一条静态断言 `WGMMA_A_SIZE_PER_STAGE ≤ SMEM_A + SMEM_B*kNumStages`，保证即使 A 的 WGMMA
padding（`WGMMA::M*BLOCK_K` 按 fp8 字节算）也不会越界——这是从 FP8 代码继承下来的保守检查。

### 初始化与 PDL（`:81-131`）

- `:85-89` 由 **第 `kNumMathThreads/32` 号 warp**（即第一个 TMA warp）的 elect-one 线程
  `prefetch_tma_descriptor` 三张 descriptor。
- `:112-121` 由第 `kNumMathThreads/32 + 1` 号 warp 初始化 barrier：`full.init(1)`、
  `empty.init(kNumTMAMulticast * kNumMathThreads / 32)`（每个 math warp 各释放一次）。
- `:124` cluster>1 用 `cluster_sync_with_relaxed_arrive`，否则 `__syncthreads`。
- `:127-128` 寄存器再平衡：TMA warp 降到 48，math 248/224（128/256 线程）。
- **`:131` `cudaGridDependencySynchronize()`** —— 与 P3-4 的 PDL 同款：等上一个 kernel 的 grid 依赖，
  让本 kernel 的 TMA warp 能在前一个 kernel 收尾时提前铺流水。

---

## 4. 合并 stage：BF16 专属的小技巧（`:44-56`）

```cpp
constexpr uint32_t kDoMergeStages =
    kNumStages_ >= 10 and kGemmType == GemmType::Normal and
    kMajorA == Major::K and kMajorB == Major::K and kNumMathThreads == 128;
constexpr uint32_t kNumMinStages = 5;
constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;
```

当「stage 数 ≥ 10、Normal、A/B 都 K-major、只有 1 个 math warpgroup」时，把相邻的
`kNumStagesPerMerge` 个 stage **在 K 方向拼成一个更宽的 stage**：`BLOCK_K = BLOCK_K_ *
kNumStagesPerMerge`，stage 数变成 `kNumStages_/kNumStagesPerMerge`（≥5）。注释写得很直白
（`:45-46`）：**为了减少 `warpgroup_wait<0>()` 的开销**。因为 BF16 的 `BLOCK_K_` 只有 64（FP8 是 128），
K 大时 stage 数会很多，每个 stage 一次 barrier 往返，开销占比高；合并后 barrier 往返次数降到
`1/kNumStagesPerMerge`，而 K 方向的 WGMMA 总量不变。

合并后 kernel 内部要按「原子 K」（`BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge`，`:218`）来推进 WGMMA
descriptor：内层 `for k < BLOCK_K/WGMMA::K`（`:265`）用 `atom_k_idx = k*WGMMA::K/BLOCK_ATOM_K`（`:266`）
在合并后的 slab 里跨原子步进。也就是「**数据搬运与 barrier 粒度 = 合并后的 BLOCK_K；MMA 取址仍以
原子 64 为单位**」。

实验里 `(128,128,512)` 这个形状正好触发合并：真机抓到 `...64u,16u,64u,128u,128u,32u,16u,128u,128u,1u`，
`kNumStages_=16`、`kNumMathThreads=128` → `kNumStagesPerMerge=3`、`BLOCK_K=192`、`kNumStages=5`。
它的数值与未合并路径完全一致（relerr 2.2e-5，§6），说明合并是纯性能优化。

---

## 5. 三段流水线

### 5.1 TMA warp-group（`:147-209`）

只有 **第 `kNumMathThreads/32 + 2` 号 warp**（第三 TMAs warp）的 elect-one 线程发 TMA：

1. `while (scheduler.get_next_block(m_block_idx, n_block_idx))`（`:157`）——persistent 调度，`GemmType`
   决定块的枚举顺序与有效性（P4-3）。K-grouped 空组直接 `continue`（`:158-159`）。
2. 判断 multicast 是否可用（`:163-166`）：`scheduler.is_tma_multicast_valid(m_block_idx)`；只有
   `kIsTMAMulticastOnA`/`!kIsTMAMulticastOnA` 之一会拿到 `kNumTMAMulticast`，另一个恒为 1。
3. K 循环（`:169`）：先 `empty_barriers[stage_idx]->wait(phase ^ 1)`（`:171`），再按 major 发 A/B 的
   `tma::copy`（`:188-199`，2D 或 3D-batched），最后 `full_barrier.arrive_and_expect_tx(A+B)`（`:200`）。
   `num_total_k_blocks = ceil_div(shape_k, BLOCK_K)`（`:168`）用的是**合并后** BLOCK_K。
4. 退出前对 multicast 再做一轮 empty wait（`:205-208`），以免分布式 barrier 被过早析构。

`tma::copy`（`common/tma_copy.cuh:91-104`）只是对 `copy_nd` 的薄封装：多播时 leader CTA 用
`SM90_TMA_LOAD_MULTICAST_nd`（`:78-86`），并由 `num_tma_multicast` 位掩码选中 cluster 内 peer。

### 5.2 math warp-group：WGMMA（`:210-282`）

- `math_wg_idx = threadIdx.x / 128`（`:215`）——同组 128 线程是一个 warpgroup，负责 `WGMMA::M=64` 行。
- **descriptor 的构造与前进**（关键）：`a_desc/b_desc` 用 `mma::sm90::make_gmma_desc<...>` 造出
  一次（`:219-220`），取出低 32 位用 `__shfl_sync` 广播（`:221-222`）；内层每一步再调
  `advance_gmma_desc_lo<...>`（`:267-270`，`mma/sm90.cuh:237-241`）按 `(mn_idx, k_idx, atom_offset)`
  重算 start_address，避免整块重建 descriptor——这是 SM90 WGMMA 的常规套路。
- **wave**：`WAVE_BLOCK_M = BLOCK_M<=64 ? BLOCK_M : WGMMA::M*2`（`:228`）。`BLOCK_M=256` 时
  `WAVE_BLOCK_M=128`、2 个 wave；`accum[WGMMA::kNumAccum * (BLOCK_M/WAVE_BLOCK_M)]` 分两段累加器
  （`:230`，初值 0）。注意两个 warpgroup 是**跨 wave 交错**的：warpgroup 0 管每 wave 的前 64 行，
  warpgroup 1 管后 64 行（`a_desc` 的 `math_wg_idx*WGMMA::M` 偏移，`:219`）。
- WGMMA 发射（`:262-272`）：外层 wave、内层 K 原子，`WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1)`。
  第四个参数 `scale_d=1` 恒为累加——**首条不清零**，因为 `accum` 已初始化为 0。
- `warpgroup_arrive/commit_batch/wait<0>` 包裹（`:260/274-278`），随后 `empty_barrier_arrive(stage_idx)`
  释放该 stage 的 smem 给 TMA 复用（`:281`）。`empty_barrier_arrive`（`:238-245`）在 multicast 时用
  `arrive(target_cta)` 只通知存活的 peer CTA，否则 `lane_idx==0` 单点 arrive。

### 5.3 epilogue：两种输出 dtype 两条路（`:284-388`）

`do_wgmma_store`（`:234-235`）决定哪些线程参与回写（`BLOCK_M<64` 时只有部分 warp）。

**bf16 输出（`kSwizzleDMode>0`，`:302-349`）**：用 `ptx::SM90_U32x2_STSM_N<nv_bfloat162>` 做 STSM，
把 `float2` 转 bf16 后按 swizzle 地址写 smem_d。swizzle 寻址（`:313-336`）把「bank group 偏移 +
atom 偏移 + wave/warp 偏移」拼出字节地址，`col ^= row % (kSwizzleDMode/16)`（`:328`）就是标准 128B
swizzle 的异或。

**fp32 输出（`kSwizzleDMode==0`，`:350-364`）**：没有 STSM 可用，直接 `st.shared` 写 `float2`；
因为 fp32 是 4 字节不能按 bf16 的 bank 布局走，索性**不开 swizzle**（正对应 §3 抓到的 `kSwizzleDMode=0`）。

回写完成后 `tma_store_fence()` + `NamedBarrier::sync(kNumWGMMAStoreThreads, 0)`（`:365-366`），再由
前 `BLOCK_N/TMA_D_BLOCK_N` 个线程发 TMA store（`:371-387`）：
- Normal/MGrouped → `SM90_TMA_STORE_2D`；
- `Batched` → `SM90_TMA_STORE_3D`（第三维 = `scheduler.current_group_idx`）；
- `kWithAccumulation` → 换成 `..._REDUCE_ADD_2D/3D`，即 **D = C + A·Bᵀ 是 TMA reduce-add 完成的**，
  不需要额外读回 D（对比 P7-1 里 FP8 `k_grouped_nt` 强制 `D=C+AᵀB` 的额外回读）。

`TMA_D_BLOCK_N = kSwizzleDMode==0 ? BLOCK_N : kSwizzleDMode/sizeof(bf16)`（`:286`）：out=bf16 时一个
128B swizzle 原子装 64 个元素，out=fp32 时整行一次铺——静态断言（`:289-291`）保证 store 指令数不超 32。

---

## 6. H100 实测（`harness/exp_bf16_sm90_gemm.out.txt`）

环境：NVIDIA H100 80GB HBM3、cc=9.0、`device_num_sms=132`、DeepGEMM arch_major=9。
所有 BF16 dense 用例都 `use_deterministic_algorithms(True)`，强制走 DeepGEMM 自家 kernel
（否则默认交给 cuBLASLt，见 P17-1；本实验也对比了：deterministic=off → `nvjet_sm90_tst_...`，
on → `sm90_bf16_gemm_impl`）。计时用「每次调用前后 `cuda.synchronize` + CUDA event，取 100 次中位数」，
报的是**真实单次延迟**（避免 PDL 让相邻 launch 重叠而系统性低估）。

### 6.1 稠密 `bf16_gemm_nt` 精度与吞吐

| m | n | k | out | acc | relerr vs fp32 | TFLOPS |
|--:|--:|--:|:--|:--|--:|--:|
| 1024 | 1024 | 1024 | bf16 | 否 | 6.8e-5 | 208 |
| 1024 | 1024 | 1024 | fp32 | 否 | 1.0e-6 | 189 |
| 2048 | 4096 | 4096 | bf16 | 否 | 1.5e-4 | 719 |
| 2048 | 4096 | 4096 | bf16 | 是 | 2.9e-3 | 697 |
| 2048 | 4096 | 4096 | fp32 | 否 | 4.4e-6 | 567 |
| 4096 | 7168 | 7168 | bf16 | 否 | 2.0e-4 | 801 |
| 4096 | 7168 | 7168 | bf16 | 是 | 2.9e-3 | 788 |
| 4096 | 7168 | 7168 | fp32 | 否 | 8.0e-6 | 725 |
| 8192 | 8192 | 2048 | bf16 | 否 | 1.0e-4 | 765 |
| 8192 | 8192 | 2048 | fp32 | 否 | 2.1e-6 | 626 |
| 128 | 4096 | 4096 | bf16 | 否 | 1.6e-4 | 246 |
| 256 | 7168 | 2048 | bf16 | 否 | 1.0e-4 | 418 |
| 128 | 128 | 512 | bf16 | 否 | 2.2e-5 | 3（合并 stage，小形状延迟主导） |

结论：
- **out=fp32 比 out=bf16 略慢**（大形状 ~0.9×）：多了一倍的 smem 回写与 TMA store 字节，且 fp32 不能开
  swizzle（P7-1 的 wgrad 也是 fp32 输出，同款访存压力）。
- **accumulate 会略慢**（719→697、801→788），因为 D 由 C 读入由 TMA reduce-add 完成，多一份流量。
- **bf16 精度分层干净**：out=fp32 时 relerr ~1e-6（fp32 输出只做一次舍入）；out=bf16 时 ~1e-4
  （bf16 输出舍入）；`acc=是` 且 out=bf16 时 ~2.9e-3（累加又落到 bf16 D 上，重舍入）——与官方
  `tests/test_bf16.py` 的阈值（fp32 6e-7 / bf16-acc 1e-5 的 calc_diff 尺度）一致。
- 峰值 **801 TFLOPS**（`4096×7168×7168` bf16），接近 H100 bf16 dense 峰值（~989 TFLOPS）的 ~81%。

### 6.2 别名入口

`nt/nn/tn/tt` 四个入口对同一逻辑结果 relerr 分别 1.662e-3 / 1.660e-3 / 1.663e-3 / 1.664e-3
（差异来自 bf16 输入下不同的收缩顺序），确认它们只是转置转发、共用同一 kernel。

### 6.3 m-grouped（contiguous / masked）

`G=8、n=4096、k=4096`：contiguous（不等长在 M，`mk_alignment=128`）relerr 1.48e-4、**575 TFLOPS**；
masked（`[G,max_m,·]` + `masked_m`）relerr 1.50e-4。两者的 `kWithGroupOffsetA` 分别为 false/true
（`:173`），但数值等价。

### 6.4 k-grouped（wgrad，major MN/MN）

`G=4、m=4096、n=7168、logical_k=[1024,2048,512,1536]`：SM90 强制 accumulate（`c.has_value()`，
见 §7 的坑），`D=C+AᵀB` relerr **1.68e-6**、**422 TFLOPS**。给一个非 128 倍数的 K 会被 host 断言拒绝
（`k % k_alignment == 0`）。

### 6.5 batched MLA 投影

`bhr,hdr->bhd`：`b=128,h=128,r=512,d=512` relerr 4.9e-5、190 TFLOPS；`b=4096,h=8,r=4096,d=1024`
relerr 1.5e-4、**781 TFLOPS**。`bhd,hdr->bhr` relerr 4.8e-5 / 7.1e-5。两者都走 `GemmType::Batched`
的 3D TMA store（`:374-379`）。

---

## 7. 踩坑与值得记的点

1. **BF16 dense 默认不是 DeepGEMM kernel**：`bf16_gemm_nt` 在「非确定性 ∧ cuBLASLt 可用 ∧
   plain-store epilogue」时直接交给 cuBLASLt（`gemm.hpp:518-523`，P17-1）。要测自家 kernel 必须
   `use_deterministic_algorithms(True)` 或限制 `num_sms`。
2. **SM90 的 k-grouped bf16 强制 accumulate**：`k_grouped_bf16_gemm_tn_contiguous` 在 arch9 分支断言
   `c.has_value()`（`gemm.hpp:707-710`），没有 direct output（那是 SM100 才有的 `epilogue`）。
3. **计时要防 PDL**：kernel 里有 `cudaGridDependencySynchronize`（`:131`）。背靠背连续 launch 会让
   相邻 kernel 在 PDL 下重叠，kineto 单看每个 kernel 的 duration 会系统性偏小（本实验一开始就踩到，
   出现同一形状 TFLOPS 差 2× 的假象）。用「每次 sync + event」才是真实延迟。
4. **out=fp32 一定 `kSwizzleDMode=0`**：fp32 不能走 bf16 的 STSM swizzle 布局，epilogue 改成裸
   `st.shared`。这解释了为什么 host 侧 `get_storage_config` 对 FP32 输出不给 swizzle（P18-1）。
5. **合并 stage 依赖选核**：只有 stage≥10 + Normal + K/K-major + 128 math 线程才触发（`:47-51`）。
   本次 `(128,128,512)` 抓到了 `kNumStages_=16 → per-merge 3`。

---

## 8. 小结

- DeepGEMM 的 SM90 BF16 GEMM 是**一套 kernel 服务五种 `GemmType`**（Normal / MGrouped
  contiguous & masked / KGrouped / Batched），因为没有 SF，比 FP8 的 `1d1d`/`1d2d` 简单一层。
- 骨架与 FP8 SM90 完全同构：TMA warp-group 铺流水 → math warpgroup 发 `MMA_64xNx16` WGMMA →
  epilogue 经 smem 用 TMA store（`Batched` 走 3D、累积走 REDUCE_ADD）。差异在 `BLOCK_K=64`、
  fp32 输出关 swizzle、以及**合并 stage** 的 BF16 专属优化。
- H100 真机六类入口全部对拍通过，峰值 **801 TFLOPS**（bf16, `4096×7168×7168`），k-grouped wgrad
  422 TFLOPS。这也补齐了前 28 篇里 BF16 路径只被顺带提到的那块。

### 关键源文件

- `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh`（主 kernel，唯一）
- `csrc/jit_kernels/impls/sm90_bf16_gemm.hpp`（六个 host 入口 + JIT 模板拼装）
- `deep_gemm/include/deep_gemm/mma/sm90.cuh`（`BF16MMASelector`、`make/advance_gmma_desc_lo`）
- `deep_gemm/include/deep_gemm/common/tma_copy.cuh`（`copy`/`copy_nd` + multicast）
- `csrc/apis/gemm.hpp`（`bf16_gemm_*`、`m_grouped_bf16_*`、`k_grouped_bf16_*` 的分派与契约）
- 选核：`csrc/jit_kernels/heuristics/sm90.hpp`（P18-1）
