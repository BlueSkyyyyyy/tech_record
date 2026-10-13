# MPK SM100 BF16 MoE linear device 身体走读：`moe_linear_sm100.cuh`（cp.async gather + register scatter 的 grouped GEMM）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/65` 同快照）；
> 本地 `/ssd/mirage_mpk_build`。CUTLASS 侧走快照钉死的 submodule `deps/cutlass`，commit `f3fde583`。
>
> 本文走读 **BF16 SM100 MoE linear** device 身体 `tasks/blackwell/moe_linear_sm100.cuh`（827 行）——
> `docs/65`（P151-1）逐行打开的是它的 **dense BF16 兄弟** `linear_sm100_mpk.cuh`（737 行），
> 并在 §8「局限」明文留了「未与 `moe_linear_sm100.cuh`（827 行 BF16 MoE linear）对走」这条尾巴；
> `docs/33`（P119-1）走的是它的 **FP8 MoE 兄弟** `fp8_group_gemm_sm100.cuh`（1567 行）。
> 三足里本文补上中间这一支。
> **纯源码走读 + 1 个 host 侧结构探测**：该内核的 SMEM 描述符布局由 CuTe 布局代数算出，全是
> `CUTE_HOST_DEVICE`，故整套代数在 host 上可重算/求值；SM100 指令本体在目标卡 H100/sm90
> **不可执行**（`src/moe_linear_sm100_layouts.cu` + `harness/moe_linear_sm100_layouts.{sh,out.txt}`）。
> 所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）同一族里，这一支把「MoE 的三件事」全塞进一个 device task，且为此换了数据通路。**
dense 兄弟 `linear_sm100_mpk.cuh` 的 A（权重）和 B（输入）都走 **TMA**、输出走 **TMA store**；
MoE 这一支只保留 **A 走 TMA**，而把 **B（输入激活）改成逐 token 的 `cp.async` gather**
（`:540-553`，按 routing index 选 token、`topk_idx>0` 才拷），输出改成 **register 直接 scatter**
到 `[batch, topk, N]`（`:802-809`，**无 SMEM/TMA 落点**）。三件事是：① **专家分组**——每个 CTA 只做
`expert_offset, expert_offset+EXPERT_STRIDE, …` 这组激活专家（`:486-488`）；② **token 路由 gather**——
B 的每个 `(token, k)` 列由 `tRoutingIndex(token)` 决定拷不拷（`:540-543`）；③ **route-aware 输出**
——只把被路由的 `(token, topk)` 写回（`:806`）。**未路由的 token 在 SMEM 里是陈旧数据，但会被 MMA 读到、
又在 scatter 时被丢弃**——这就是「gather + 谓词 + scatter」这套边界成立的代价（§4.1/§4.3）。

**(B)（headline）`moe_linear_sm100.cuh` 与 `linear_sm100_mpk.cuh` 共享同一套 CuTe 描述符代数、
只有 MoE 多出 `sB_cp_layout`；host 探测证明两处 ABI 一致、且 MoE-only 的 cp.async stage 步长也对。**
两支用**完全相同**的 `make_tiled_mma(SM100_MMA_F16BF16_SS)`、`partition_shape_{A,B}` /
`UMMA::tile_to_mma_shape(Layout_K_SW128_Atom)`（host 实测 `mma_shape_A=((_128,_16),_1,_4,_8)`，
`mma_shape_B=((_16,_16),_1,_4,_8)`）——连**两处陈旧注释都一样**：源码 :193/:196 把 `mma_shape_B/C` 的
首维写成 `_32`，实测是 `_16`（原因是 `MMA_N=16`，注释大概从某 `MMA_N=32` 的历史配置拷来，与 `docs/65` §3.1
是同一枚「按注释推断会错」）。MoE **唯一**多出的布局是 `sB_cp_layout`
（host 实测 `(_16,_64,_1,_8):(_64,_1,_0,_1024)`，`:204-214`）——把 `sB` 的 swizzle atom 组合成一个
**给 `cp.async` 用的 gather 视图**（`tensor_cp_sB()`，`:308`）。**ABI 一致性照旧成立**：
A stage 步长 `OUTPUT_ATOM*TILE=8192`、B stage 步长 `MMA_N*TILE=1024`，与 CuTe 布局 stage 步长
逐项 `OK`（§3.4）。SMEM 预算 A+B=`147456 B`（**MoE 不再单独给 C 开 SMEM**——C 在 TMEM、经寄存器 scatter），
`MoESharedStorage` ≈`148196 B`；单测 wrapper 照样申请 **224 KiB**（差的 ~64 KiB 是保守预留，源码无公式）。

**(C)（headline）生产布线六层全通、且 BF16/FP8/Hopper 三支对 `EXPERT_STRIDE` 用了三种约定。**
op `moe_w13_linear_sm100`/`moe_w2_linear_sm100` → `TaskType 254/255`（`runtime_header.h:141-142`）→
`register_moe_linear_sm100_task`（`task_register.cc:3003`，两处 op 分派 `graph.cc:687/704`）→
`task_header.cuh:54` 传递 include → 前端 `moe_w13_linear_layer`/`moe_w2_linear_layer`
（`persistent_kernel.py:2031/2282`）→ 四个 builder（DeepSeek-V3 `builder.py:976/1054`、GLM-4.6
`:339/347`、GPT-OSS `:349/358`、Inkling `:544/564`）。**`expert_stride` 的口径三支不同**：
本支（BF16 SM100）**硬编码** `w13?10:8`（`task_register.cc:3059`，与单测 `test_w13_linear.py:15`
的 `expert_stride=10`/`test_w2_linear.py:15` 的 `8` 一致）；FP8 SM100 版改成 **`bgraph.grid_dim.x`**
（`:3298`，注释明说「must match grid_dim.x」）；Hopper SM90 版又**硬编码** `w13?5:4`
（`:3664`）。而 DeepSeek-V3 builder 给 BF16 路径传的是 `grid_dim=(NUM_EXPERTS,1,1)`（`:976-984`）。
**三种约定并存是快照事实，本文只记不判**（§5.2）——但它是一枚「同一个 `expert_offset=bid.x` 语义
在三代/两精度上有三套 stride 来源」的静态落差。

---

## 1. 定位：三足关系（dense BF16 / MoE BF16 / MoE FP8）

| | `linear_sm100_mpk.cuh`（docs/65） | **`moe_linear_sm100.cuh`（本文）** | `fp8_group_gemm_sm100.cuh`（docs/33） |
|---|---|---|---|
| 数值格式 | BF16 dense（可选 +残差 / split-K） | **BF16 grouped MoE** | FP8 block-scaled grouped MoE |
| 输入 | A=权重(TMA)、B=输入(TMA)、输出(TMA store) | **A=权重(TMA)、B=输入(`cp.async` gather)、输出=寄存器 scatter** | A/B 都 `cp.async`/TMA、输出 scatter |
| 专家分组 | 无 | **有**（`expert_offset`/`EXPERT_STRIDE`/`mMask`） | 有 |
| warp 角色 | 3 类（TMA/MMA/epilogue，warp6/7 空转） | **3 类**（DMA(5)/MMA(4)/epilogue(0-3)） | 4 类（DMA/MMA/SCALE/epilogue） |
| 尺度流水 | 无 | 无 | **有**（SCALE warp + UTCCP→TMEM） |
| SMEM C 缓冲 | 有（`sC_layout`） | **无**（C 在 TMEM，经寄存器 scatter） | 无 |
| inline 提示 | `__noinline__` | **`__forceinline__`** | — |
| 行数 | 737 | **827** | 1567 |

**为什么值得单独走**：它是「SM100 手工 linear task 家族」里 **BF16 × grouped-MoE** 的一支，
也是唯一用 **`cp.async` gather 输入 + register scatter 输出** 的 BF16 身体——把「MoE 的稀疏/路由」
这一维（dense 兄弟完全没有）用最少的机械代价（不加 warp、不加尺度流水）编进一个 task。

**与 `docs/33` 的分工**：`fp8_group_gemm_sm100` 已经覆盖「block-scaled FP8 + SCALE warp + UTCCP/TMEM」；
本文只补 **BF16 版**，不重复 FP8 的尺度/UTCCP 细节。

---

## 2. 模板契约与「一个 task = 一个 CTA」

入口（`:31-59`）：

```cpp
template <typename T_, typename TMA_A, class InputTensor, class BiasTensor,
          class IndicesTensor, class MaskTensor, class OutputTensor,
          int MMA_M, int MMA_N, int BATCH_SIZE, int OUTPUT_SIZE, int ORIG_OUTPUT_SIZE,
          int REDUCTION_SIZE, int NUM_EXPERTS, int NUM_TOPK, int EXPERT_STRIDE,
          bool W13_LINEAR, bool NOBIAS,
          int NUM_AB_STAGE = 8, int NUM_ACC_STAGE = 2, int NUM_C_STAGE = 4>
__device__ __forceinline__ void moe_linear_sm100_task_impl(
    const TMA_A &tma_a, InputTensor mInput, BiasTensor mBias,
    IndicesTensor mRoutingIndices, MaskTensor mMask, OutputTensor mOutput,
    int const expert_offset);
```

- **`__forceinline__`（`:52`）**，而 dense 兄弟是 `__noinline__`（`docs/65` §2）——同族两支对「task 身体是否
  内联进常驻 kernel 体」给了**相反**的提示。这是快照事实（本文只记不判）；对照 `docs/04` §8/§11 的代码足迹讨论。
- **`expert_offset` 是运行期入参**（不是模板参）：来自 `task_desc->task_metadata.expert_offset`
  （`runtime.cc:358` 置成 `bid.x`），配合编译期 `EXPERT_STRIDE` 决定本 CTA 处理哪些激活专家（`:486-488`）。
- **`W13_LINEAR`**（`:47`）是编译期开关：w13（gate+up）时输入是 2D `[batch,K]`、直接按 token 取；
  w2（down）时输入是 3D `[batch,topk,K]`、按 `topk_idx-1` 取第几份（`:544-552`）。
- **`NOBIAS`** 编译期短路 per-expert bias（`:754-773/789-794`）。
- **单 CTA、无 cluster**：`mma_coord_vmnk=(0,_,_,_)`（`:63-66`）、`cta_mma=tiled_mma.get_slice(0)`（`:315`）。
  `grid_dim=(1,1,1)` 的 standalone wrapper（`sm100_moe/...:325`）+ 生产 `kNumSMs` 语义与 `docs/32` §5.3 同族。

---

## 3. 布局代数（host 探测，`harness/moe_linear_sm100_layouts.out.txt`）

### 3.1 MMA tile 与分区（`:71-184`）

与 dense 兄弟逐字相同的构造：`SM100_MMA_F16BF16_SS<T_,T_,float,128,16,Major::K,Major::K>`
→ `bK=tile_size<2>*4=64`（`:79-85`）。host 实测 `tile_size=(_128,_16,_64)`。

`partition_shape_{A,B,C}`（`:166-184`）实测：

| | 源码注释（`:190/193/196`） | 实际（host 实测） |
|---|---|---|
| `mma_shape_A` | `((_128,_16),_1,_4,_8)` | `((_128,_16),_1,_4,_8)` ✓ |
| `mma_shape_B` | `((_32,_16),_1,_4,_8)` | `((_16,_16),_1,_4,_8)` ✗（`_32`→`_16`） |
| `mma_shape_C` | `((_32,_128),_1,_1,_4)` | `((_16,_128),_1,_1,_4)` ✗（`_32`→`_16`） |

与 `docs/65` §3.1 **同一处陈旧注释**（`_32` 来自历史 `MMA_N=32` 配置）。

### 3.2 swapAB 约定同 dense 兄弟

`mA` 是权重坐标张量 `(OUTPUT_SIZE, REDUCTION_SIZE, NUM_EXPERTS)`（`:103-107`，stride 里专家维是
`E<1>*OUTPUT_SIZE`）；`gA` 走 M 维（输出维）、`gB` 走 N 维（token 维）（`:128-144`）。故
**`C = 输入·权重ᵀ`，UMMA-M=输出维、UMMA-N=token 维**（对照 `docs/33`/`docs/65` 同一约定）。

### 3.3 SMEM 布局与 MoE-only 的 `sB_cp_layout`

`cases`：`sA/sB` 用 `Layout_K_SW128_Atom`（SW128，`Sw<3,4,3>`），`sB_cp_layout`（`:204-214`）把
`sB_layout.layout_a()` 组合成 `(_16,_64,_1,_8):(_64,_1,_0,_1024)`（host 实测）——**这就是 `cp.async`
写侧看到的 B 视图**，与 `tCsB` 的描述符视图共用同一块 SMEM（`tensor_sB`/`tensor_cp_sB`，
`storage.cuh:158-164`）。`sC_layout`（`:216-230`）仍构造了（`(_16,_128),_1,_1,(_1,_4):(_128,_1),_0,_0,(_0,_2048)`），
但 **MoE 的 C 不在 SMEM**——`sC_layout` 在本文用例里**并非必需**（dense 兄弟用它做 epilogue 中转；
MoE 直接 register scatter）。**这是同一段布局代码在 MoE 里的「半死」处**（只记不判）。

host 实测的 `sB_cp` stage 步长 `layout(0,0,0,1)=1024`，与手工 cp.async 环形 stage 步长
`MMA_N*TILE=1024` 一致；A stage `8192` 与 `weight_smem` 手推一致——**A/B 两处 ABI 一致性照旧成立**。

### 3.4 `copyB` 与 barrier

`cp_async_group_size = 32/MMA_N = 2`（`:355`），`copyB` 的线程布局是 `(MMA_N=16, cp_group=2)=32` 线程
（= 一个 warp，即 warp5），每 lane 每 k-tile 搬 16 B（`uint128`，`:372-376`）。这解释了
**`b_full` barrier 的 arrival count 为什么是 32**（`:271-276`，一个 warp5 lane 一次
`cpasync_barrier_arrive_noinc`，`:555-556`）——而 `a_full`（TMA）是**事务字节** barrier（arrival count 1）、
`ab_empty`（`umma_arrive`）arrival 1、`acc_full` arrival 1、`acc_empty` arrival 4（`:277-292`）。
`tma_transaction_bytes_A=sizeof(T)*128*64=16384 B`（`:341-342`）。

---

## 4. 三条流水（DMA(5) / MMA(4) / epilogue(0-3)）

### 4.1 warp5：TMA（权重）+ `cp.async`（输入 gather）生产者（`:466-571`）

- 专家循环（`:486-488`）：`activated_expert_offset` 从 `expert_offset` 起、步长 `EXPERT_STRIDE`，
  `expert_idx = mMask[activated_expert_offset]`（`num_activated_experts = mMask[NUM_EXPERTS]`，`:438-439`）。
- `m_tile`×`n_tile` 双层循环（`:491-492`）内跑 `k_tile_count` 个 K-tile（`:437`）。
- **A（权重）走 TMA**（`:523-537`）：`weight_smem.set_ptr(shared_weight + stage*128*64)`，
  TMA 坐标 `(k_tile*64, m_tile*128 + expert_idx*ORIG_OUTPUT_SIZE)`——**专家维直接折进权重的行偏移**。
- **B（输入）走 `cp.async` gather**（`:540-553`）：`token_idx = n_tile*MMA_N + lane/cp_group`；
  `topk_idx = (token_idx<BATCH_SIZE) ? tRoutingIndex(token_idx) : 0`；
  **仅当 `token_idx<BATCH_SIZE && topk_idx>0` 才 `cute::copy(copyB, tBgB(...n_tile,k_tile[,topk_idx-1]), tBsB(...stage))`**
  （w13 取 2D、w2 取第 `topk_idx-1` 份）。未路由/越界 token **不拷**，SMEM 留旧值。
- 乐观 peek（`try_wait_barrier(ab_empty, …)`，`:501-503/558-562`）+ 环形 `stage` 与 phase（`:496-499`）同族。

### 4.2 warp4：MMA 消费者+生产者（`:572-664`）

- 先过 `tmem_allocation_result_barrier`（等 warp0 分配 TMEM，`:576-577`），`tCtAcc.data()=tmem_base_ptr`。
- 每 (expert,m,n)：`acc_buf_idx=num_tiles_executed%NUM_ACC_STAGE`（`:587-588`），peek `a_full`/`b_full` 后等
  `acc_empty`（`:599-608`）；K-tile 循环内 `for k_block in size<2>(tCrA)`（=4）发
  `cute::gemm(tiled_mma, tCrA(_,_,k,smem_rd_buffer), tCrB(_,_,k,smem_rd_buffer), tCtAcc_Slice)`
  （`:632-638`，首块 `accumulate_=Zero` 后 `One`）。`umma_arrive(ab_empty)` 通知 DMA 可覆写（`:640-641`），
  跑完 K 后 `umma_arrive(acc_full)`、`num_tiles_executed++`（`:658-660`）。
- **注意**：`b_full` 由 32 个 `cp.async` lane 到达（§3.4），**与 token 是否真被路由无关**——未路由时
  lane 仍 arrive（`:555`），故 barrier 一定过、MMA 用陈旧 B 算，数值由 epilogue 丢弃（§0(A)）。

### 4.3 warps0-3：epilogue 消费者（register scatter）（`:665-816`）

- warp0 `tmem_allocator.allocate(num_tmem_columns=MMA_N*NUM_ACC_STAGE=32, &tmem_base_ptr)`（`:669-671`，
  `num_tmem_columns≤512` 由 `task_register.cc:3069` 断言），经 `tmem_allocation_result_barrier` 广播。
- **T2R**：`make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, ...)` 把 TMEM 累加器读进寄存器（`:681-691`）。
- 每 (expert,m,n)：**per-expert bias**（`:724-773`）用 `gBias(_,_,n_tile,m_tile,expert_idx)` +
  `out_col_valid` 谓词读（越界置 0）；等 `acc_full` 后 T2R copy（`:775-780`），`epilogue_wg_barrier` +
  `elect_one_sync` arrive `acc_empty`（`:783-787`），累加 bias、转回 bf16（`:789-799`）。
- **register scatter 输出**（`:802-809`）：`out_col = m_tile*MMA_M + threadIdx.x`（`:750`），
  `for i in [0,MMA_N)`：`n_idx=n_tile*MMA_N+i`，`topk_idx=tRoutingIndex(n_idx)`，
  **仅当 `n_idx<BATCH_SIZE && topk_idx>0 && out_col_valid` 才 `mOutput(n_idx, topk_idx-1, out_col)=tCrC[i]`**
  ——`mOutput` 布局是 `[batch, topk, N]`（`task_register.cc` 生成，stride `(topk*output_stride, output_stride, 1)`）。
  **`:801` 的注释写「R2G store, use cp.async.bulk here」——即当前实现是普通寄存器 store、TMA store 是 TODO**；
  这是「注释说要用、代码还没用」的一处落差（只记不判）。
- 收尾（`:815-823`）：`__syncthreads()` 后 warp0 `tmem_allocator.free(...)`，**不 `release_allocation_lock`**
  （`:820-821` 注释「don't do relinquish for megakernel」）——与 `docs/65` §4.4 / `docs/32` §5.3 /
  `docs/33` §6.3 **逐字同款**：C2（常驻）对 device task 的直接让步。

---

## 5. 生产布线（补 `docs/32` §5.4 / `docs/65` §8 的尾巴）

### 5.1 六层坐标

- **TaskType**：`TASK_MOE_W13_LINEAR_SM100=254` / `TASK_MOE_W2_LINEAR_SM100=255`
  （`runtime_header.h:141-142`）。
- **op 名**：`"moe_w13_linear_sm100"` / `"moe_w2_linear_sm100"`（`graph.cc:687-701` / `:704-718`）；
  `task_config` 的 `num_inputs` 随有无 5th bias 在 4/5 间切（`:692-694/709-711`）。
- **发射器**：`register_moe_linear_sm100_task(bgraph,params,w13_linear)`（`task_register.cc:3003`），
  末尾发射 `kernel::moe_linear_sm100_task_impl<cute::bfloat16_t, TMA_A, ...>`（`:3171`），
  w13→`TASK_MOE_W13_LINEAR_SM100`、w2→`TASK_MOE_W2_LINEAR_SM100`（`:3198/3200`）。
  常量：`MMA_M=128, MMA_N=16, bK=64, num_ab_stages=8, num_acc_stages=2, num_c_stages=4`
  （`:3061-3069`）；TMA `(B,M,S)=(3,3,3)`；`TILE_SIZE=64, OUTPUT_ATOM=128`。
  `TMA_A` 取 `input_tma_desc_ptrs[1][0]`（权重，`:3103` 附近）；`mInput` 依 `w13_linear` 选 2D/3D layout
  （`:3155-3168`）。
- **include**：`tasks/blackwell/task_header.cuh:54`（host 探测 audit=1）。
- **前端**：`moe_w13_linear_layer`（`persistent_kernel.py:2031`，`register_task(...,"moe_w13_linear_sm100",params)` `:2070`）、
  `moe_w2_linear_layer`（`:2282`，`register_task(...,"moe_w2_linear_sm100",params)` `:2319`）；bias 只在
  `target_cc==100` 允许（`:2052/2078`）。
- **builder**（host 探测 audit=4 个文件）：DeepSeek-V3 `builder.py:976/1054`、GLM-4.6 `:339/347`、
  GPT-OSS `:349/358`、Inkling `:544/564`。

### 5.2 `EXPERT_STRIDE` 的三套来源（同 §0(C)）

| 路径 | stride 来源 | 快照 |
|---|---|---|
| BF16 SM100（本文） | **硬编码** `w13?10:8` | `task_register.cc:3059` |
| FP8 SM100（docs/33） | **`bgraph.grid_dim.x`** | `task_register.cc:3298`（注释「must match grid_dim.x」） |
| Hopper SM90（docs/07） | **硬编码** `w13?5:4` | `task_register.cc:3664` |

而 `expert_offset` 在运行期一律 `bid.x`（`runtime.cc:358`，覆盖 W13/W2 × {SM100,SM90,FP8} 六种 TaskType）。
**三套 stride 约定并存**；DeepSeek-V3 对 BF16 路径传 `grid_dim=(NUM_EXPERTS=256,1,1)`（`builder.py:976-984`）。
本文**只记不判**（不臆断「哪套对」）——但这条落差是「同一 `expert_offset=bid.x` 语义在不同代/精度上
用了三种 stride 源」的静态证据。

### 5.3 单测 wrapper（`tests/runtime_python/blackwell/sm100_moe/`）

- `runtime_kernel_wrapper_sm100.cu`：自带 `__global__ moe_linear_sm100_wrapper`（`:150-207`，`__launch_bounds__(256,1)`）
  + host TMA 描述符构造 + `launch_moe_linear_sm100<...>`（`:208-...`）；pybind 暴露
  `moe_w13_linear_sm100`/`moe_w2_linear_sm100`（`:701/704`）。测试 `test_w13_linear.py`（`:15` stride=10）/
  `test_w2_linear.py`（`:15` stride=8）用 `torch.matmul` 逐专家参考。
- **同族复用**：`tests/runtime_python/blackwell/sm100_fp8_moe/runtime_kernel_wrapper_bf16.cu` 也
  `moe_linear_sm100_task_impl<bfloat16,...>`（`:135`），文件头自陈是「DeepSeek-V3 配置、2D grid
  `(expert_stride, n_splits)`」的 BF16 参考 wrapper——**FP8 MoE 目录反过来复用了本文的 BF16 身体**。
- **as-shipped 隐患（静态事实，同 `docs/65` §5.4）**：`sm100_moe/setup.py` 的 `depends` 列了
  **不存在的** `tasks/blackwell/utils.cuh`（快照只有 `sm100_utils.cuh`）；扩展名 `name='runtime_kernel_blackwell'`
  与 `sm100_linear` 目录**撞名**（`docs/30` §2.1 记过同一撞名）。本文**未构建/未运行**（SM100 本机不可执行）。

---

## 6. 三足对照（与 dense 兄弟 / FP8 MoE / Hopper 对应）

### 6.1 与 dense 兄弟 `linear_sm100_mpk.cuh`（docs/65）

| 维度 | dense（docs/65） | **MoE（本文）** |
|---|---|---|
| `TMA_A` | 权重 | 权重（专家维折进行偏移 `expert_idx*ORIG_OUTPUT_SIZE`） |
| 输入 B | TMA | **`cp.async` gather**（route 谓词） |
| 输出 | TMA store（可选 `tma_reduce_add` split-K） | **register scatter**（`[batch,topk,N]`；TMA store 是 TODO `:801`） |
| C SMEM 缓冲 | 有（`sC_layout` 参与 epilogue 中转） | **无**（C 只在 TMEM） |
| 额外布局 | — | **`sB_cp_layout`**（cp.async 视图） |
| bias | 前向残差（`NOBIAS`） | per-expert bias（`stride=0` 广播行） |
| inline | `__noinline__` | `__forceinline__` |
| 循环 | m×n | expert×m×n |

### 6.2 与 FP8 MoE `fp8_group_gemm_sm100.cuh`（docs/33）

FP8 版是「1 CTA 跑 3 条 warp 特化流水 + SCALE warp + UTCCP→TMEM」的 1567 行重身体；本文是 BF16
的「3 类角色、无尺度流水」轻身体。两者共享「专家分组 + route-aware gather/scatter」的语义，
但 **FP8 把 B 的载入也编进 DMA warp 的两套描述符（A/B）、本文把 B 交给一个 warp 的 `cp.async` gather**。
`docs/33` 的 UTCCP/scale TMEM 细节本文不重复。

### 6.3 与 Hopper 对应 `hopper/moe_linear_swapAB_hopper.cuh`（`TASK_MOE_W13/W2_LINEAR_SM90=161/162`）

同语义（`expert_offset`/`EXPERT_STRIDE`/`mMask`/route-aware 输出，`:362-365`），但走 Hopper 的
wgmma/TMA、`expert_stride` 硬编码 5/4（§5.2）。本文是它在 SM100 上的兄弟。

---

## 7. 判据映射（C1–C5）

与 `docs/07`/`docs/29`–`docs/65` 同口径：**一个 device task 身体只贡献 C3（融合范围 / 单 task 可表达）**，
本身不拿 C1/C2/C4/C5。

- **贡献 C3**：把「BF16 grouped-MoE linear（w13 或 w2、可选 per-expert bias）」编成一个可被
  `register_task` 生成、可被常驻 worker 调度的 `__device__` 函数——**融合范围 = 一个专家的
  gather-GEMM-scatter 段**。
- **不贡献 C1/C2**（单次 launch / 常驻）：由运行层拿（`docs/06`/`docs/22` §8）。
- **不贡献 C4/C5**：task 内 `mbarrier`/`NamedBarrier` 是 **CTA 内跨 warp 同步**，不是跨 task/跨 SM 的
  去中心化调度。
- **唯一与常驻有关的 device 侧事实**：TMEM `free` 不 `release_allocation_lock`（`:819-823`），
  为 C2 让步，与 `docs/32`/`docs/33`/`docs/65` 同族。

---

## 8. 局限（诚实标注）

- **SM100 指令/硬件本体未执行**：所有 CuTe 布局是**在 host 上对真实 CUTLASS 头求值**得到的
  （`src/moe_linear_sm100_layouts.cu`），属「trait 规定的布局」，**不是** SM100 上观测到的硬件行为；
  UMMA/TMA/`cp.async` 在 SM100 上的实际吞吐/描述符解释均**未验证**。目标卡 H100/sm90 无 SM100。
- **未构建、未跑内核、未复算数值**：单测 `test_w13_linear.py`/`test_w2_linear.py` 的参考是
  `torch.matmul` 逐专家，本文**未复算**；as-shipped 隐患（`utils.cuh` 缺失/扩展撞名）只作静态记录。
- **`b_full` 的 `cp.async.noinc` 语义未展开到 PTX 级**：本文按 `:271-276`（count 32）+ `:555-556`
  （每 lane `cpasync_barrier_arrive_noinc`）记为「cp.async 完成驱动的 arrival-counted barrier」，
  未逐条核对 `cp.async.mbarrier.arrive.noinc` 的 pending-count 语义（快照事实，不判）。
- **`sC_layout` 在 MoE 里是否为死代码**：`sC_layout`（`:216-230`）仍在，但 MoE 的 C 走 TMEM+寄存器
  scatter；是否被 `MoESharedStorage` 的某处间接使用**未穷举**（对照 `docs/32` §3「已实现未接线」族）。
- **`EXPERT_STRIDE` 三套来源只记不判**（§5.2）：未追 DeepSeek-V3 `grid_dim=(NUM_EXPERTS,1,1)` 与
  硬编码 stride 10/8 在运行时如何各自成立。
- **未与 `fp8_group_gemm_sm100.cuh` 的 SF 段对走**（那属 `docs/33` 范围）；未与 Hopper 对应逐行对走。

---

## 9. 参考坐标（可核对）

- **主文件**：`include/mirage/persistent_kernel/tasks/blackwell/moe_linear_sm100.cuh`（827 行）。
  - 入口 `moe_linear_sm100_task_impl` `:52-59`；includes `:22-27`；`warp_idx` `:60`；
    `mma_coord_vmnk` `:63-66`；`num_tmem_columns` `:68`；`tiled_mma` `:71-78`；`bM/bN/bK` `:79-85`；
    `mma_tiler` `:87`；`mma_coord/cd_tiler` `:98-100`；`mA` `:103-107`；`gA/gB/gBias` `:128-146`；
    `mma_shape_{A,B,C}` `:166-184`；陈旧注释 `:190/193/196`；`sA/sB_layout` `:200-203`；
    **`sB_cp_layout` `:204-214`**；`sC_layout` `:216-230`；`MoESharedStorage` using `:248-255`；
    extern shared 对齐 128 `:257-261`；barrier 初始化 `:264-293`（`a_full` tx/1、`b_full` tx/**32**、
    `ab_empty`/1、`acc_full`/1、`acc_empty`/4）；NamedBarrier `:297-301`；`copyB` `:372-376`；
    `tma_transaction_bytes_A` `:341-342`；`WeightSmem` `:362-369`；`tCrA/tCrB` `:410-413`；`tCtAcc` `:414-421`；
    `k_tile_count`/`num_activated_experts` `:437-439`；`TmemAllocator` `:441-442`；
    **warp5 DMA** `:466-571`（专家循环 `:486-488`、TMA A `:523-537`、cp.async B gather `:540-553`、
    arrive `:555-556`）；**warp4 MMA** `:572-664`；**warps0-3 epilogue** `:665-816`（TMEM alloc `:669-671`、
    T2R `:681-691`、bias `:724-773`、scatter R2G `:802-809`）；收尾不 relinquish `:819-823`。
- **支撑**：`tasks/blackwell/storage.cuh`（`MoESharedStorage` `:132-165`，`tensor_cp_sB` `:161-164`）；
  `tasks/hopper/smem_layout_tma.cuh`（`smem_tma`）；`tasks/hopper/tma.cuh`（`tma_cp_async`）；
  `tasks/hopper/barrier.cuh`（`try_wait_barrier`）；`deps/cutlass`（`cpasync_barrier_arrive_noinc`
  `include/cutlass/arch/barrier.h:747-759`）。
- **生产**：`src/kernel/task_register.cc`（`register_moe_linear_sm100_task` `:3003-3201`、`expert_stride` `:3059`、
  常量 `:3061-3069`、emission `:3171`；FP8 对照 `:3204`/`expert_stride=grid_dim.x` `:3298`；
  Hopper 对照 `:3611`/`expert_stride` `:3664`）；`src/kernel/graph.cc:687-718`；
  `include/mirage/persistent_kernel/runtime_header.h:141-142`（254/255）；
  `tasks/blackwell/task_header.cuh:54`；`src/kernel/runtime.cc:351-359`（`expert_offset=bid.x`）。
- **前端/builder**：`python/mirage/mpk/persistent_kernel.py:2031-2071/2282-2320`；
  `python/mirage/mpk/models/deepseek_v3/builder.py:976-1054`、`glm4_moe/builder.py:339-347`、
  `gpt_oss/builder.py:349-358`、`inkling/builder.py:544-564`。
- **单测**：`tests/runtime_python/blackwell/sm100_moe/runtime_kernel_wrapper_sm100.cu`
  （wrapper `:150-207`、launch `:208-...`、pybind `:701/704`）、`test_w13_linear.py`/`test_w2_linear.py`、
  `setup.py`；`tests/runtime_python/blackwell/sm100_fp8_moe/runtime_kernel_wrapper_bf16.cu:135`（复用本文身体）。
- **本文实验**：`src/moe_linear_sm100_layouts.cu` + `harness/moe_linear_sm100_layouts.sh`
  + `harness/moe_linear_sm100_layouts.out.txt`（目标卡 H100/sm90a，host 侧，device 3；CUTLASS `deps/cutlass f3fde583`）。
- **论文快照**：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （实现质量/device task 内部，同 `docs/28`–`docs/65`）。
- **同专题相关篇**：`docs/65`（本文补其 §8 尾巴，dense BF16 兄弟）、`docs/33`（FP8 MoE 兄弟）、
  `docs/32` §5.4（本文补其摘要）、`docs/30` §2（单测 wrapper/撞名）、`docs/51`/`docs/57`（SMEM ABI /
  `MoESharedStorage`）、`docs/58`（TMA swizzle）、`docs/59`（TMEM 列）、`docs/22` §8（C 判据总映射）。
- **交叉专题**：DeepGEMM MoE/mega 家族见 `../mega-moe/`（本专题不重复）。
