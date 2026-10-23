# MPK SM100 BF16 linear device 身体走读：`linear_sm100_mpk.cuh`（CuTe 描述符 + 手写 TMA 的混合体）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/64` 同快照）。
> CUTLASS 侧走 MPK 该快照钉死的 submodule `deps/cutlass`，commit `f3fde583`
> （`~/mirage_mpk_build/deps/cutlass`）。
> `docs/32`（P118-1）在 §5.4 只给这份身体 **10 行摘要**、`docs/33` 走的是它的 FP8 兄弟
> `fp8_group_gemm_sm100.cuh`、`docs/34` 走的是 FP8 兄弟 `linear_fp8_1d2d_sm100.cuh`。本文把它
> **逐行打开**，是「SM100 手工线性 task 家族」的第三支（也是唯一 **BF16/无 block-scale** 的一支）。
> **纯源码走读 + 1 个 host 侧结构探测**：该内核的 SMEM 描述符布局由 CuTe 布局代数
> （`make_tiled_mma`/`partition_shape_{A,B,C}`/`UMMA::tile_to_mma_shape`）算出，全是
> `CUTE_HOST_DEVICE`，故整套代数在 host 上可重算/求值；SM100 指令本体在目标卡 H100/sm90
> **不可执行**（`src/linear_sm100_bf16_layouts.cu` + `harness/linear_sm100_bf16_layouts.{sh,out.txt}`）。
> 所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）这是本专题里唯一「CuTe 高层描述符 + 手写 TMA」**合体**的 device 身体——同一块 SMEM
被两套表示同时解释。** FP8 两兄弟里，`linear_fp8_sm100.cuh` 是**全手写**（手算 SMEM 描述符、
手填指令描述符、手写 `tcgen05`），`linear_fp8_1d2d_sm100.cuh` 是**全 CuTe 高层**
（`make_tiled_mma` + `cute::gemm` 主循环）。`linear_sm100_mpk.cuh` 走第三条路：**MMA 用 CuTe 高层**
（`SM100_MMA_F16BF16_SS` → `make_tiled_mma` → `partition_shape_*` → `UMMA::tile_to_mma_shape` →
`make_fragment_A/B` → `cute::gemm`，`linear_sm100_mpk.cuh:61-68/170-219/380-391/536-542`），
**而 TMA 存取与 pipeline 全靠手写**（`smem_tma` 视图 + `tma_cp_async`/`tma_store_async` +
自管 `mbarrier`/`NamedBarrier`，`:328-347/414-482/565-724`）。于是同一段 `shared_storage.A/B/C`
**既要满足 CuTe 的 `Layout_K_SW128_Atom` 描述符、又要满足 `smem_tma<3,3,3>` 的硬件 128B swizzle
写侧**——host 探测证明两者一致（`Sw<3,4,3>`）、且**手写 stage 步长逐项等于 CuTe 布局的 stage 步长**
（A 8192 / B 1024 / C 2048，全 `OK`，§3）。

**(B)（headline）它把「1 个 CTA 跑 3 类角色、6 个 warp、2 个空转」的 warp-specialized pipeline 塞进
一个 device task**——`warp5`=TMA 生产者、`warp4`=MMA 消费者+生产者、`warps0-3`=epilogue 消费者
（`:414/483/565`），`block=256` 线程（8 warp）里 **warp6/7 无任何分支、只跟着末尾
`__syncthreads()`**（`:725`）——是 FP8 身体「`warp5 DMA`/`warp4 MMA`/`warp6 SCALE`/`warps0-3
EPILOGUE`」**去掉 SCALE 后剩下的 3 类角色**（BF16 无 block-scale，故无 scale warp）。同步用
**4 对 `mbarrier`**（`ab_full` TransactionBarrier / `ab_empty` / `acc_full` / `acc_empty`
arrival-count=4，`:249-263`）+ **2 个 `NamedBarrier`**（`TmemAllocBarrier` 160、`EpilogueBarrier`
128，`:267-270`）。host 探测还复现了两处**源码注释与事实不符**：`mma_shape_B/C` 的注释写 `_32`
（实际 `_16`，因 `MMA_N=16` 不是 32），`sA/sB/sC_layout` 的注释贴的是 **ArithTuple 坐标张量**
（不是真正的 swizzle SMEM 布局）——又一枚「按注释推断会错」。

**(C)（headline）生产布线六层全通、单测 wrapper 独立成一颗整 kernel，但 as-shipped 单测编不过。**
三条 op（`linear_sm100` / `linear_with_residual_sm100` / `splitk_linear_sm100`）→ 三个 `TaskType`
（`TASK_LINEAR_SM100=253` / `TASK_LINEAR_WITH_RESIDUAL_SM100=252` / `TASK_SPLITK_LINEAR_SM100=251`）
→ 两个 `register_*` 发射器（`task_register.cc:2013/2178`，各自生成 `linear_sm100_mpk_task_impl`
的不同实参）→ `task_header.cuh:39` 传递 include → `graph.cc:626-641` dispatch。单测侧
`tests/.../sm100_linear/runtime_kernel_wrapper_sm100.cu` 自带 `__global__` wrapper（`:43-144`）+
host TMA 描述符构造（`:146-319`）+ pybind（`:660-665`），**但它 `#include "blackwell/task_header.cuh"`
会连带拉进整个 Blackwell task 家族，其中 `sampling_sm100.cuh:137/143` 需要
`MPK_MAX_NUM_BATCHED_REQUESTS`/`MPK_MAX_NUM_BATCHED_TOKENS`**——两个宏只在 `persistent_kernel.cuh:58-59`
被**注释掉**、实由生产 JIT 生成，故 as-shipped 单测（在目标卡上用 CUDA 13.2 试编）**报未定义标识符、
编不过**；其 `setup.py` 还**硬编码 `/usr/local/cuda-12.8`**（本机不存在）、`depends` 指向**不存在的
`blackwell/utils.cuh`**——与 `docs/62` 记的 cute/hopper 单测「as-shipped 坏」同一族。

---

## 1. 定位：与 FP8 两身体的三足关系

| | `linear_fp8_sm100.cuh`（docs/32 §2） | `linear_fp8_1d2d_sm100.cuh`（docs/34） | **`linear_sm100_mpk.cuh`（本文）** |
|---|---|---|---|
| 数值格式 | FP8 block-scaled（1D1D/1D2D） | FP8 block-scaled（CuTe） | **BF16，无 scale** |
| MMA 描述符 | **手写**（`sm100_utils.cuh` 描述符 + 手填 idesc） | **CuTe 高层**（`make_tiled_mma`+`cute::gemm`） | **CuTe 高层**（`make_tiled_mma`+`cute::gemm`） |
| TMA / pipeline | 手写 | **CuTe/手写混**（另有 CUTLASS collective） | **手写**（`smem_tma` + `tma_cp_async`） |
| warp 角色 | 4 类（DMA/MMA/SCALE/epilogue） | CuTe 主循环 + epilogue | **3 类**（TMA/MMA/epilogue，无 scale） |
| 生产 | `TASK_LINEAR_FP8_SM100` 家族 | **无调用者** | `TASK_LINEAR_SM100`/`+RESIDUAL`/`SPLITK` |
| 单测 | `sm100_linear_fp8` | — | `sm100_linear`（pybind `linear_sm100_mpk`） |

**为什么值得单独走**：它是「SM100 手工 task 家族」里唯一 **BF16** 的一支（MoE 侧对应
`moe_linear_sm100.cuh`，`linear_sm100_mpk.cuh:27` 也 `#include "storage.cuh"`），且是唯一
**CuTe 描述符 + 手写 TMA 混用**的一支。它把「CuTe 能替我算什么、哪些还得我手抠」这条边界画在
**「布局代数给 CuTe、数据搬运给手写」**上。DeepGEMM 的 BF16 SM100 对应物
`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh`（432 行，commit `057ca59`）则是一颗**完整
CUTLASS 持久化 GEMM**（persistent `Scheduler`、multicast、PDL `cudaGridDependencySynchronize`、
swapAB epilogue，`:118-160/172-227`）——MPK 这份是它「**砍到单 CTA、去调度器、去 multicast、
去 PDL、只留 TMA+UMMA+epilogue 主干**」的 device 内嵌版（§6 有静态对应）。

---

## 2. 模板契约与「一个 task = 一个 CTA」

入口（`:31-50`）：

```cpp
template <typename T_, typename TMA_A, typename TMA_B, class BiasTensor,
          typename TMA_OUT, int MMA_M, int MMA_N, int BATCH_SIZE,
          int OUTPUT_SIZE, int REDUCTION_SIZE, bool NOBIAS, bool SplitK,
          int NUM_AB_STAGE = 8, int NUM_ACC_STAGE = 2, int NUM_C_STAGE = 4>
__device__ __noinline__ void linear_sm100_mpk_task_impl(
    const TMA_A &tma_a, const TMA_B &tma_b, BiasTensor mBias, const TMA_OUT &tma_out);
```

- **`__noinline__`**（`:46`）：整段身体不内联——在 megakernel 里，这段代码是被 JIT 生成的常驻
  kernel 体调用的一个「叶子函数」。这是「一个 device task 靠 `__noinline__` 与其它 task 隔开、
  共享同一份指令缓存」的写法（对上 `docs/04` §8/§11 的代码足迹讨论）。
- **`MMA_M=128, MMA_N=16` 是编译期常量**（生产 `task_register.cc:2058-2059` 硬编码）。`BATCH_SIZE`/
  `OUTPUT_SIZE`/`REDUCTION_SIZE` 也是模板参数（`:337-338`/`:47-50`），即**运行时形状被编进内核**——
  与 `docs/32` §5.3 的 `kNumSMs=1`、`docs/33` §6.3 的「TMEM 不 relinquish」同族：都是把「一个 task
  = 一颗 kernel」这个原本的 launch 契约，压进一个常驻 kernel 的单个 CTA。
- **单 CTA、无 cluster**：`mma_coord_vmnk = (0, _, _, _)`（`:54-57`），`mma_v = get<0>(...) = 0`，
  `cta_mma = tiled_mma.get_slice(0)`（`:283-284`）。没有 2SM/peer-CTA（对照组：`docs/36`/`docs/52`
  的 MLA 2SM）。

---

## 3. 布局代数：CuTe 描述符侧与 `smem_tma` 写侧的双重表示

### 3.1 MMA tile 与分区（`:61-188`）

```cpp
auto tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<T_,T_,float,MMA_M,MMA_N,
                                 UMMA::Major::K, UMMA::Major::K>{});   // :61-68
auto bM = tile_size<0>(tiled_mma);   // 128
auto bN = tile_size<1>(tiled_mma);   // 16
auto bK = tile_size<2>(tiled_mma) * Int<4>{};  // 16*4 = 64  (:73-75)
```

host 探测（`linear_sm100_bf16_layouts.cu`）实测 `tile_size = (_128,_16,_64)`：**UMMA-K 对 16-bit
类型是 16**，乘 4 = `bK=64`（4 条 UMMA 覆盖一个 K-tile），不是某些注释会让人误读的 256。

`partition_shape_A/B/C`（`:170-188`）实测：

| | 源码注释（`:193/195/197`） | 实际（host 实测） |
|---|---|---|
| `mma_shape_A` | `((_128,_16),_1,_4,_8)` | `((_128,_16),_1,_4,_8)` ✓ |
| `mma_shape_B` | `((_32,_16),_1,_4,_8)` | `((_16,_16),_1,_4,_8)` ✗（`_32`→`_16`） |
| `mma_shape_C` | `((_32,_128),_1,_1,_4)` | `((_16,_128),_1,_1,_4)` ✗（`_32`→`_16`） |

**`_32` 是陈旧注释**（大概从某个 `MMA_N=32` 的历史配置拷来）。这不是「按注释推断会错」的孤例——
下面 3.3 的坐标注释更离谱。

### 3.2 swapAB 约定：A=权重、B=输入，UMMA-M=输出维

`mA` = 权重 `(OUTPUT_SIZE, REDUCTION_SIZE)`（`:92-97`），`mB` = 输入 `(BATCH_SIZE, REDUCTION_SIZE)`
（`:98-103`），`mC` = `(BATCH_SIZE, OUTPUT_SIZE)`（`:104-108`）。`local_tile` 用 `mma_tiler=(bM,bN,bK)`
切：`gA` 走 M 维（输出维）、`gB` 走 N 维（batch/token 维）（`:112-127`，源码注释 `:304-307` 明说
「`size<0>(mma_tiler)=bM` 对应权重 TMA 维、`size<1>=bN` 对应输入 TMA 维」）。所以 **`C = 输入·权重ᵀ`，
UMMA-M 是输出维、UMMA-N 是 token 维**——与 `docs/33`（`fp8_group_gemm`）headline、DeepGEMM 的
`kSwapAB` 同一约定（DeepGEMM `sm100_bf16_gemm.cuh:68` `UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N`）。
`cd_tiler = (bN,bM,bK)`（`:89-90`）用于切 C/Bias/输出（`:128-148`）。

### 3.3 SMEM 布局（`:200-219`）与两处注释失真

```cpp
auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_A);
auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_B);
// sC：先用 INTER atom 造一个 unswizzled fake，再 composition 成一个 [16,128] 连续段
```

host 实测：

- `sA_layout = Sw<3,4,3> o ... ((_128,_16),_1,_4,(_1,_8)):((_64,_1),_0,_16,(_0,_8192))`
- `sB_layout = Sw<3,4,3> o ... ((_16,_16),_1,_4,(_1,_8)):((_64,_1),_0,_16,(_0,_1024))`
- `sC_layout = Sw<0,4,3> o ... ((_16,_128),_1,_1,(_1,_4)):((_128,_1),_0,_0,(_0,_2048))`

而源码 `:221-231` 的（注释掉的）debug print 贴的是
`sA_layout: ArithTuple(_0,0) o ((_128,_16),_1,_4,8):((_1@1,_1@0),_0,_16@0,_8@0)` 之类——**那是
`mA/mB` 坐标张量（`make_coord_tensor`）的形式，不是 SMEM 布局**。这段注释是**从上游坐标张量的
print 拷过来、没更新**的：真正的 SMEM 布局带 `Sw<3,4,3>`（SW128 swizzle）与 `smem_ptr`。
**A/B 用 SW128、C 用 `Sw<0,4,3>`（恒等 swizzle）**——对上 epilogue 侧 `smem_tma<T_,0,M,S,...>`
（`OutputSmem`，`B=0` 不 swizzle，`:346`）。

### 3.4 双表示的 ABI 一致性（host 实测，本文的硬结论）

关键是：**`smem_tma` 写侧的手写 stage 步长必须等于 CuTe 描述符侧的 stage 步长**，否则 UMMA 会读错
buffer。内核手写推进指针（`:452-455` 与 `:664-665`）：

- A：`smem_wr_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE` = `stage*128*64 = 8192`
- B：`smem_wr_buffer * MMA_N * TILE_SIZE` = `stage*16*64 = 1024`
- C：`c_smem_wr_buffer_idx * MMA_N * OUTPUT_ATOM_SIZE` = `stage*16*128 = 2048`

host 探测把 CuTe 布局的 `Tiles_K`/C-stage 步长求出来对照：**A 8192 / B 1024 / C 2048，逐项 `OK`**。
这条等式是这两套表示能共存的前提；它成立，说明「CuTe 给描述符、手写给搬运」这条路走得通
（对比 `docs/32` FP8 全手写身体里同一件事是靠人工对齐保证的）。

### 3.5 SMEM 预算

host 实测 `cosize`：A=131072 B（8 stage×128×64×2B）、B=16384 B、C=16384 B，**A+B+C=163840 B=160 KiB**。
单测 wrapper 申请 **224 KiB** 动态 SMEM（`runtime_kernel_wrapper_sm100.cu:266`）——比布局 cosize 多
~64 KiB（barrier 只占几百字节）。这多出来的部分是**保守预留**（未在源码里给出公式）；本文如实记
「cosize 160 KiB vs 申请 224 KiB」，不臆断缺口来源。

---

## 4. 三条流水

### 4.1 warp5：TMA 生产者（`:414-482`）

- 双层 `m_tile`×`n_tile` 循环（`:417-418`），每个 (m,n) 内跑 `k_tile_count` 个 K-tile（`=cute::size<4>(tCgA)`，`:407`）。
- 环形 stage 缓冲 `smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE`（`:424`），
  phase 按「跨过 stage 0 翻转」（`:425-426/438-440`）。
- **乐观 peek**：先 `try_wait_barrier(ab_empty, phase)`，命中就跳过 `cute::wait_barrier`（`:428-446`）——
  与 FP8 身体 `docs/33` §1 的乐观 peek + lookahead 同一套惯用法。
- `elect_one_sync()` 单线程发两条 TMA（`:448-467`）：`tma_a.tma_cp_async`（权重，坐标 `(k_tile*64,
  m_tile*128)`，`:449-450`）、`tma_b.tma_cp_async`（输入，坐标 `(k_tile*64, n_tile*16)`，`:451`）；
  `set_barrier_transaction_bytes` 用 `tma_transaction_bytes`（`:456-458`）。
- **`kClampedBN`**（`:303-311`）：transaction bytes 里输入侧乘 `min(MMA_N, BATCH_SIZE)`——**当
  `BATCH_SIZE<MMA_N` 时 TMA 输入 box 被硬件 clamp**，故字节数要按 clamp 后的尺寸算，否则 barrier
  永远等不到（与 `docs/32` §2.5 的同类守卫同族）。
- `input_weight_smem.set_ptr(shared_weight + stage*128*64)`、`input_smem.set_ptr(shared_input +
  stage*16*64)`（`:452-455`）——这就是 §3.4 的手写 stage 推进。

### 4.2 warp4：MMA 消费者+生产者（`:483-564`）

- 先过 `tmem_allocation_result_barrier`（等 warp0 分配 TMEM，`:487`），`tCtAcc.data() =
  shared_storage.tmem_base_ptr`（`:488`）。
- 每 (m,n)：`acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE`（`:495`），等 `ab_full` 与
  `acc_empty`（`:507-513`），`accumulate_ = Zero` 初始化累加器（`:516`）。
- K-tile 循环内：`for k_block in size<2>(tCrA)`（=4）发 `cute::gemm(tiled_mma, tCrA(...),
  tCrB(...), tCtAcc_Slice)`（`:536-542`），首块后 `accumulate_ = One`；`umma_arrive(ab_empty)`
  通知 DMA 可覆写（`:544-545`）。
- 跑完 K 后 `umma_arrive(acc_full)`、`num_tiles_executed++`（`:559-561`）。

### 4.3 warps0-3：epilogue 消费者（`:565-724`）

- warp0 负责 `tmem_allocator.allocate(num_tmem_columns=MMA_N*NUM_ACC_STAGE=32, ...)`（`:569-570`，
  `num_tmem_columns≤512` 由 `task_register.cc:2067` 断言）；经 `tmem_allocation_result_barrier`
  广播（`:572`）。
- `NumericConverter<AccType=float, TypeBias=T_=bf16>` 与 `<TypeC=bf16, AccType=float>`（`:579-580`）。
- **T2R**：`make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, ...)` 把 TMEM 累加器读进寄存器（`:582-595`）。
- **残差/偏置**（`:633-665`）：`NOBIAS` 编译期短路；否则用 `gBiasCoord` + `elem_less` 造谓词，
  `copy_if` 带**双重重界守卫**（`threadIdx.x < OUTPUT_SIZE`，且 M/N 维 allowed）读残差；
  **broadcast bias** 时 upstream 用 `stride=0`（`task_register.cc:2141-2145`）让每行读同一行。
- 累加残差 → 转回 bf16（`:680-690`）→ **R2S** `cute::copy(tCrC, sC_epi_slice)` 落 SMEM（`:693-695`）。
- **S2G TMA**（`:698-715`）：`tma_store_fence` + `EpilogueBarrier` 同步后，warp0 单线程发
  `tma_out.tma_store_async`（普通）或 **`tma_out.tma_reduce_add_async`（`SplitK`，`:703-706`）**——
  split-K 靠 **TMA reduce-add** 把各 CTA 的部分和累进同一输出，而非另开 reduce kernel；
  `tma_store_wait<NUM_C_STAGE-1>` 做 store 流水（`:713-714`）。

### 4.4 收尾：TMEM 不 relinquish（`:725-733`）

```cpp
if (warp_idx == 0) {
  // don't do relinquish for megakernel
  // tmem_allocator.release_allocation_lock();
  tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
}
```

与 `docs/32` §5.3（`kNumSMs=1`）、`docs/33` §6.3（`fp8_group_gemm` 的 TMEM free 不 relinquish）
**逐字同款**：为「常驻 kernel 内复用同一 CTA」让步——不释放 allocation lock，让下一个 task/CTA
能继续 rasterize。这是 C2（常驻）对 device task 的直接让步，也是这族 task 与「独立单 kernel」
的接口契约。

---

## 5. 生产布线（补齐 `docs/32` §5.4 的摘要）

### 5.1 `TaskType` 与 op 名

- `TASK_SPLITK_LINEAR_SM100=251` / `TASK_LINEAR_WITH_RESIDUAL_SM100=252` / `TASK_LINEAR_SM100=253`
  （`runtime_header.h:138-140`）。
- op 名 `"linear_sm100"` / `"splitk_linear_sm100"` / `"linear_with_residual_sm100"`（`graph.cc:626-641`）。
- 实证：`task_register.cc` 里 `register_linear_sm100_task`/`register_splitk_linear_sm100_task` 各 1 处
  调用，`linear_sm100_mpk_task_impl` 2 处实例化（两发射器各一）；`graph.cc` 10 处 op 名匹配；
  `task_header.cuh:39` 1 处 include（host 探测末尾 audit 输出）。

### 5.2 两个发射器的差异（`task_register.cc:2013-2176` vs `:2178-2324`）

两者生成**同一模板的不同实参**：

- 公共常量：`MMA_M=128, MMA_N=16, bK=64, num_ab_stages=8, num_acc_stages=2, num_c_stages=4`；
  TMA `(B,M,S)=(3,3,3)`（128B swizzle）；`TILE_SIZE=64`、`OUTPUT_ATOM_SIZE=128`（`:2058-2075`）。
- **TMA 指针来源**：`TMA_A` 取 `input_tma_desc_ptrs[1][0]`（权重）、`TMA_B` 取 `[0][0]`（输入）
  （`:2130-2135`；splitk 版 `:2285-2290`）——坐实「A=权重」。
- **普通版** `reduction_size = input_ops[0]->dtensor.dim[1]`、GMEM_STRIDE_ROW 用 `reduction_size`
  （`:2049/2086`）；**splitk 版** `reduction_size = input_ops[0]->output_tensors[0].dim[1]`、
  `reduction_stride = dtensor.dim[1]`、GMEM_STRIDE_ROW 用 `reduction_stride`（`:2203-2204/2241/2259`）
  ——**split-K 的输入是「被切开的部分 K」**，故步长与逻辑 K 分离。
- **`NOBIAS` 的取反**：普通版 `(with_residual && rank_with_residual) ? "false" : "true"`（`:2160`），
  即「本 rank 加残差 ⇒ NOBIAS=false」；splitk 版 `with_residual ? "false" : "true"`（`:2313`）。
- `SplitK` 实参：普通版 `false`、splitk 版 `true`（`:2161/2314`）。
- 残差 rank 门控沿用 `docs/51` 的「TP 下残差只加一次」：`rank_with_residual = (params[0]==1)`
  （`:2016-2026`），`broadcast_bias` 额外把 stride 置 0（`:2139-2145`）。

### 5.3 单测 wrapper（`tests/runtime_python/blackwell/sm100_linear/`）

- `runtime_kernel_wrapper_sm100.cu`：自带 `__global__ linear_sm100_mpk_wrapper`（`:43-144`，
  `__launch_bounds__(256,1)`）——**把生产里的「JIT 生成 kernel 体」换成手写的 standalone kernel**；
  host 侧 `launch_linear_sm100_mpk` 手工造三个 `CUtensorMap`（`fill_tma_desc<bfloat16,3,3,3,2>`，
  `:192-233`），`smemBytes=224*1024`、`grid(1,1,1)`（`:263-266`）。
- 数字锚点：`BATCH_SIZE=1, OUTPUT_SIZE=128, REDUCTION_SIZE=768`（`:336-338`）；splitk
  `REDUCTION_SIZE=2048`（`:645-647`）。pybind 暴露 `linear_sm100_mpk` / `linear_splitk_sm100`
  （`:660-665`）。测试 `test_matmul_mpk.py` 用 `torch.matmul(x, wᵀ)` 做参考、`rtol/atol=1e-2`。

### 5.4 两次「as-shipped 坏了」（静态/试编事实）

- **试编事实**：在目标卡容器（CUDA 13.2）用 `nvcc -arch=sm_100a -c` 编该 wrapper，**报
  `sampling_sm100.cuh:137/143` 的 `MPK_MAX_NUM_BATCHED_REQUESTS`/`MPK_MAX_NUM_BATCHED_TOKENS`
  未定义**——`task_header.cuh` 把整个 Blackwell task 家族（含 sampling）拉进来，而这两个宏只在
  `persistent_kernel.cuh:58-59` 被**注释掉**、实由生产 JIT 提供（`runtime_header.h:445-448` 只用到
  名字、不定义）。standalone 单测不带这些宏 ⇒ 编不过。（SM100 指令在 H100 上不可执行，本文**只到
  试编、未运行**。）
- **静态事实**：`setup.py` 硬编码 `nvcc_path=/usr/local/cuda-12.8`（本机/容器均无此路径，容器是
  13.2），且 `depends` 里列了**不存在的** `tasks/blackwell/utils.cuh`（快照里只有
  `sm100_utils.cuh`）。与 `docs/62` 记的 `tasks/cute/hopper` 单测 `setup.py` 缺 include 是同一族
  「快照自带的 standalone 单测不保证 as-shipped 可构建」。

---

## 6. 与 DeepGEMM BF16 SM100 的静态对应

`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh`（432 行，commit `057ca59`）：

| 维度 | DeepGEMM `sm100_bf16_gemm` | **MPK `linear_sm100_mpk`** |
|---|---|---|
| 调度 | persistent `Scheduler`（`get_next_block`，`:158/175`） | **无调度器**：单 CTA 顺序双层 m/n 循环 |
| multicast / cluster | `kNumMulticast∈{1,2}`、`cluster_sync`（`:99/203-206`） | 无（单 CTA/cluster(1,1,1)） |
| group/batched | `GemmType` 多态 + `grouped_layout`（`:158`） | 无（`Normal` 一种） |
| PDL | `cudaGridDependencySynchronize()`（`:153`） | 无（device task 内嵌，非独立 launch） |
| warp 角色 | warp0 TMA / warp1 barrier / warp2 tmem（`:107/126/146`）+ epilogue warps | warp5 TMA / warp4 MMA / warps0-3 epilogue |
| swapAB | `kSwapAB`（`:32/68/76`） | **恒 swapAB**（A=权重、M=输出维） |
| epilogue | `sm100_store_cd_swap_ab` collective | 手写 TMEM→reg→SMEM→TMA + `tma_reduce_add` |

一句话：**MPK 这份是 DeepGEMM 同代 BF16 kernel 的「单 CTA、去调度/去 multicast/去 PDL」内嵌
简化版**——它保留的正是「一个 device task 在常驻 kernel 里自己跑完一颗 GEMM」所需的最小主干，
去掉的恰是那些「多 CTA 协同/独立 launch」才需要的东西。与 `docs/33` 记的「MPK 调度器 = DeepGEMM
调度器的严格子集」是同一现象在 BF16 侧的镜像（不过那份是 FP8 MoE，本文是 BF16 dense）。

---

## 7. 判据映射（C1–C5）

与 `docs/07`/`docs/29`–`docs/64` 同口径：**一个 device task 身体只贡献 C3（融合范围/单 task 可
表达）**，本身不拿 C1/C2/C4/C5。

- **贡献 C3**：把「BF16 dense linear（可选 +残差、可选 split-K）」编成一个可被 `register_task`
  生成、可被常驻 worker 调度的 `__device__` 函数（`__noinline__`）。
- **不贡献 C1/C2**（单次 launch / 常驻）：由运行层拿（`docs/06`/`docs/22` §8）。
- **不贡献 C4/C5**：task 内 `mbarrier`/`NamedBarrier` 是 **CTA 内跨 warp 同步**，不是跨 task/跨 SM
  的去中心化调度。
- **唯一与常驻有关的 device 侧事实**：TMEM `free` 不 `release_allocation_lock`（`:727-733`）——
  为 C2 让步的适配，与 `docs/32` §5.3、`docs/33` §6.3 同族。

---

## 8. 局限（诚实标注）

- **SM100 指令/硬件本体未执行**：所有 CuTe 布局是**在 host 上对真实 CUTLASS 头求值**得到的
  （`src/linear_sm100_bf16_layouts.cu`），属「trait 规定的布局」，**不是** SM100 上观测到的硬件
  行为；UMMA/TMA 的实际吞吐、descriptor 在 SM100 上的解释均**未验证**。目标卡 H100/sm90 无 SM100。
- **只到试编、未跑内核**：`nvcc -arch=sm_100a -c` 该单测 wrapper 在 `sampling_sm100.cuh` 处因缺宏
  失败（§5.4），本文**未绕过/未修改快照**去强行构建；因此「该 task 数值正确」只有上游单测的
  `rtol/atol=1e-2`（`test_matmul_mpk.py`）作凭据，本文**未复算**。
- **SMEM 预算 224 KiB vs cosize 160 KiB 的缺口未解释**：只记两个数，未追缺口来源（可能保守预留）。
- **`warp6/7` 空转是静态推断**：`:414-724` 只对 warp0-5 分支，但「block=256 ⇒ 8 warp」是按
  `__launch_bounds__(256,1)`（单测）/生产 `task_config`（`docs/33` §5 记的 `(6,1,...)`）推的；
  生产实际 `NUM_THREADS`（`docs/45` 记的 128/256 口径张力）未逐一坐实 ⇒ 「2 个空转 warp」是
  **静态推断、非运行时判定**。
- **DeepGEMM 对应是静态对照**：SM100 路径本机不具备，§6 只给源码并置，未对跑。
- **未与 `moe_linear_sm100.cuh`（827 行 BF16 MoE linear）对走**：它是同快照里另一份 BF16 身体，
  留待后续。

---

## 9. 参考坐标（可核对）

- **主文件**：`include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh`（737 行）。
  - 入口 `linear_sm100_mpk_task_impl` `:46-735`；includes `:22-27`；warp_idx `:51`；
    `mma_coord_vmnk` `:54-57`；`tiled_mma` `:61-68`；`bM/bN/bK` `:69-75`；`mma_tiler` `:77`；
    `mA/mB/mC/mBiasCoord` `:92-110`；`gA/gB/gBias/gC` `:112-148`；`mma_shape_{A,B,C}` `:170-188`；
    `sA/sB/sC_layout` `:200-219`；`SharedStorage` `:233-240`；SMEM 对齐/取址 `:242-246`；
    barrier 初始化 `:249-263`；NamedBarrier `:267-270`；`tCsA/tCsB/sC_epi` `:273-277`；
    `cta_mma` `:283-288`；`kClampedBN`/`tma_transaction_bytes` `:303-311`；`smem_tma` 实例化
    `:328-347`；`tCrA/tCrB` `:380-383`；`acc_shape`/`tCtAcc` `:384-391`；`k_tile_count` `:407`；
    `TmemAllocator` `:409-410`；**warp5 TMA** `:414-482`；**warp4 MMA** `:483-564`；
    **warps0-3 epilogue** `:565-724`（TMEM alloc `:569-573`、T2R `:582-595`、bias `:633-665`、
    S2G `:698-715`）；收尾不 relinquish `:725-733`；注释掉的 debug print `:190-198/221-231`。
- **支撑**：`tasks/blackwell/storage.cuh`（`PipedSharedStorage` `:25-55`）；`tasks/hopper/
  smem_layout_tma.cuh`（`smem_tma` `:19-129`，`get_swizzled_offset` `:51-80`）；`tasks/hopper/
  tma_2d.cuh`（`tma_cp_async` `:68-91`、`tma_store_async` `:138-152`、`tma_reduce_add_async`
  `:154-167`）；`tasks/hopper/barrier.cuh`（`try_wait_barrier` `:99`）；`tasks/blackwell/
  sm100_ptx.cuh`/`sm100_utils.cuh`（`docs/57`）。
- **生产**：`src/kernel/task_register.cc`（`register_linear_sm100_task` `:2013-2176`、
  `register_splitk_linear_sm100_task` `:2178-2324`、常量 `:2058-2075`、TMA 指针 `:2130-2138`）；
  `src/kernel/graph.cc:626-641`；`include/mirage/persistent_kernel/runtime_header.h:138-140`（251/252/253）、
  `:445-448`（仅名字）；`tasks/blackwell/task_header.cuh:39`（include）；`persistent_kernel.cuh:58-59`
  （宏被注释）。
- **单测**：`tests/runtime_python/blackwell/sm100_linear/runtime_kernel_wrapper_sm100.cu`
  （wrapper `:43-144`、launch `:146-319`、splitk `:353-628`、pybind `:660-665`）、`setup.py`、
  `test_matmul_mpk.py` / `test_matmul_splitk.py`。
- **DeepGEMM 对应**（`~/github/DeepGEMM`，commit `057ca59`）：
  `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:36`（入口）、`:68`（`UMMA_N`）、
  `:118-160`（SharedStorage/barrier/scheduler）、`:172-227`（TMA load warp）、`:153`（PDL）。
- **本文实验**：`src/linear_sm100_bf16_layouts.cu` + `harness/linear_sm100_bf16_layouts.sh`
  + `harness/linear_sm100_bf16_layouts.out.txt`（目标卡 H100/sm90a，host 侧，device 3；
  CUTLASS `deps/cutlass f3fde583`）。
- **论文快照**：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （实现质量/device task 内部，同 `docs/28`–`docs/64`）；「手工 kernel 可包成 task」见 `docs/05`。
- **同专题相关篇**：`docs/32`（本文补其 §5.4 摘要）、`docs/33`（FP8 MoE 兄弟）、`docs/34`（FP8 CuTe
  兄弟）、`docs/51`/`docs/58`/`docs/59`（SMEM/TMEM 布局即 ABI）、`docs/57`（SM100 共享原语）、
  `docs/60`/`docs/62`/`docs/63`（CUTLASS trait 并置 + host 探测方法论）、`docs/22` §8（C 判据总映射）。
- **交叉专题**：DeepGEMM MoE/mega 家族见 `../mega-moe/`（本专题不重复）。
