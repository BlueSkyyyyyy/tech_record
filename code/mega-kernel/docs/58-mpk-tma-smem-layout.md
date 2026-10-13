# MPK TMA 版 SMEM 布局 `smem_tma` 走读：一个模板参数 (B,M,S) 同时喂给 TMA / wgmma / 软件三处硬件字段——以及 1024 B 对齐契约

> 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/57` 同快照）。
> 本文是**纯源码走读（资料分析）+ 1 个 H100 实测小实验**：在目标卡（H100 / sm90a）上用真实 TMA
> 验证 `smem_tma` 的偏移函数与硬件 128 B swizzle 逐位一致，并实测其**基址对齐依赖**。
> 实验源码 `src/tma_smem_layout.cu`，原始输出 `harness/tma_smem_layout.out.txt`。

---

## 0. 定位：这是 `docs/51` 明留的尾巴

`docs/51-mpk-smem-layout.md` 走了 Ampere 的 `tasks/ampere/smem_layout.cuh`（`smem_row`/`smem_col`/
`smem_row_tiled`/`SwizzleOffsetCalculator`）与 `tasks/common/dmem_layout.cuh`，并在 §6 明确写下：

> 「Hopper 身体改用 TMA（`hopper/tma.cuh`，其 SMEM 描述符自带 swizzle 字段），
> `multitoken_paged_attention_hopper.cuh` **不 include `smem_layout.cuh`**（`grep` 证实）」

这句话是对的，但它漏了一层：Hopper/Blackwell 身体**不 include `smem_layout.cuh`，却 include 了另一份布局**
`tasks/hopper/smem_layout_tma.cuh`（129 行，`struct smem_tma`）。它被 **10 个 device 身体**共用：

| 目录 | include `smem_layout_tma.cuh` 的身体 |
|---|---|
| `hopper/` | `linear_hopper.cuh`、`linear_swapAB_hopper.cuh`、`moe_linear_swapAB_hopper.cuh`、`multitoken_paged_attention_hopper.cuh` |
| `blackwell/` | `linear_sm100_mpk.cuh`、`moe_linear_sm100.cuh`、`fp8_group_gemm_sm100.cuh`、`linear_fp8_1d2d_sm100.cuh`、`topk_softmax_sm100.cuh`、`topk_sigmoid_sm100.cuh` |
| `cute/hopper/` | `gemm_ws_mpk.cuh` |
| 单测 | `tests/runtime_python/blackwell/sm100_fp8_moe/runtime_kernel_wrapper.cu` |

名字带 `hopper/`，却被一堆 `blackwell/` 身体用——这本身又是一例「目录/后缀不等于专属」（同族见
`docs/55`/`docs/56`/`docs/57`）。`docs/51` 讲的是 Ampere 手写布局（**软件自己算 SMEM 偏移**），
本文讲的是 TMA 版布局（**描述符：软件与硬件共同解释同一块 SMEM**）——两者的契约性质不同，这正是
它值得单列的原因。

---

## 1. `smem_tma` 逐行

### 1.1 模板参数与「TMA box」语义

```cpp
template <typename T, int B, int M, int S, size_t ROW_, size_t INNER_COL_, size_t OUTER_COL_>
struct smem_tma {
  ...
  static constexpr size_t STRIDE_ROW       = INNER_COL_;
  static constexpr size_t STRIDE_OUTER_COL = ROW_ * INNER_COL_;
};
```

（`smem_layout_tma.cuh:19-39`）。与 `docs/51` 的 `smem_row<T,B,M,S,ROW,COL,STRIDE>` 相比：

- `smem_row` 是**行主序平铺** `logical_idx = row*STRIDE + col`（`STRIDE` 与 `COL` 解耦，允许 padding）；
- `smem_tma` 是 **TMA box 平铺**：外列维（`OUTER_COL`）是**块间**维，块内是 `ROW_ × INNER_COL_` 的
  **行主序**子矩阵，`STRIDE_ROW = INNER_COL_`、`STRIDE_OUTER_COL = ROW_*INNER_COL_`——即多个 box
  首尾相接。这正是 TMA 一次投递一个 `INNER_COL × ROW` tile、多次投递按 `SMEM_STRIDE_` 拉开所对应的
  物理排布（见 §3.1）。

### 1.2 swizzle 偏移函数（与 `smem_row` 同公式）

```cpp
get_swizzled_offset(logical_idx):          // :51-80
  if (B == 0) return logical_idx;          // B=0 ⇒ 恒等（无 swizzle）
  block_idx      = logical_idx >> (M+S+B);
  in_block_idx   = logical_idx & ((1<<(M+S+B))-1);
  irow           = in_block_idx >> (M+S);          // B 位：块内行（8 行原子）
  icol           = (in_block_idx >> M) & ((1<<S)-1);
  icol          ^= irow;                            // ← swizzle：16B 块 XOR 行
  offset_in_bank = in_block_idx & ((1<<M)-1);
  phy_offset     = (block_idx<<(M+S+B)) + (irow<<(M+S)) + (icol<<M) + offset_in_bank;
```

与 `smem_row`（`docs/51` §2.2）唯一的实现差别：这里用 `+` 拼接，`smem_row` 用 `|`（第 64–65 行 vs
`smem_layout.cuh:78`）——因各段位域不相交，二者数值等价，**只记不判**。`(B,M,S)=(3,3,3)` 时
`M+S+B=9`，原子 = `2^9` 元素 × `sizeof(T)`；bf16 时 = **8 行 × 64 列 × 2 B = 1024 B**，恰是硬件
128 B swizzle 的「8 行 × 128 B」原子（`icol` 是 8 个 16 B 块、`irow` 是 8 行）。

### 1.3 3D / 2D / 1D 三种访问

- **3D** `operator()(row, inner_col, outer_col)` / `at(...)`（`:82-98`）：`logical_idx =
  row*STRIDE_ROW + outer_col*STRIDE_OUTER_COL + inner_col`。
- **2D** `operator()(row, col)` / `at(row,col)`（`:101-119`）：把 `col` 拆成
  `outer_col=col/INNER_COL`、`inner_col=col%INNER_COL` 后走同一公式。
- **1D** `at(idx)` / `operator[](idx)`（`:121-128`）：直接对 `idx` 做 swizzle。

本实验在设备侧核对了 **2D 与 3D 访问逐位一致**（`get(out(24,64))` 与 `at(row,inner,0)` 对比，
`harness/tma_smem_layout.out.txt` 各 case 的 `2D-vs-3D mismatch=0`）。

### 1.4 `MIRAGE_DEBUG_HOPPER` 调试块

`:66-77` 在 `#ifdef MIRAGE_DEBUG_HOPPER` 下打印 `block_idx/irow/icol/offset_in_bank/phy_offset`
与 `base_ptr[logical_idx]` vs `base_ptr[phy_offset]`。`tma_2d.cuh:61`、`tma_3d.cuh:65`、`tma_4d.cuh:65`
有同宏的打印——该宏是这套 TMA 栈统一的调试开关（本快照**无任何文件 `#define` 它**，需手动开）。

---

## 2. 核心：一个 (B,M,S) 同时喂给三处硬件字段

`smem_tma` 的模板参数 `(B,M,S)` 不是布局层私有约定——它被**同时**编码进三处：

**① TMA tensor map 的 swizzle**（`hopper/tma_2d.cuh:266-270`）：

```cpp
constexpr CUtensorMapSwizzle tma_swizzle =
    (B == 1 ? CU_TENSOR_MAP_SWIZZLE_32B
     : B == 2 ? CU_TENSOR_MAP_SWIZZLE_64B
     : B == 3 ? CU_TENSOR_MAP_SWIZZLE_128B
              : CU_TENSOR_MAP_SWIZZLE_NONE);
```

**② wgmma matrix descriptor 的 swizzle mode + LBO/SBO**（`hopper/wgmma.cuh:46-91`，节选）：

```cpp
if (SMEM::b == 3) {  // 128B
  base_desc |= encode(16) << 16;    // LBO
  base_desc |= encode(1024) << 32;  // SBO
  base_desc |= 1llu << 62;          // wgmma swizzle mode
} else if (SMEM::b == 2) { ... 512 ... 2llu<<62; }
  else if (SMEM::b == 1) { ... 256 ... 3llu<<62; }
  else                   { ... 128/256 ... 0llu<<62; }
```

**③ 布局自身的 `get_swizzled_offset`**（§1.2）。

也就是说：`smem_tma<bf16,3,3,3,...>` 这一个类型，**同时**决定了「TMA 把数据按 128 B swizzle 写进
SMEM」「wgmma 按 128 B swizzle 从 SMEM 读」「软件按 `icol^=irow` 算地址」三件事。三者必须**同 B**，
否则任何一方都会算错——第 5 节的负对照实验正是把「B 不匹配」的后果测了出来。

生产里 B/M/S 是**全仓统一**的。`src/kernel/task_register.cc:1194-1196` 对 Hopper linear 家族硬编码：

```cpp
constexpr int B = 3; constexpr int M = 3; constexpr int S = 3;
constexpr int TMA_CP_ASYNC_SIZE = 64;   // SMEM_COL = 64 元素 = 128 B（128B swizzle 的内维）
```

`hopper/linear_hopper.cuh:79-81` 同样写死 `B=M=S=3`。所以 MPK 的 Hopper TMA 路径**只用 128 B swizzle**，
`B=1/2`（32/64 B）在本快照里是**已实现但未使用**（同族「实现不止一遍/有实现未启用」的另一种形态：
这里是「实现了三个档，只用一个」）。

---

## 3. 使用者视角：`smem_tma` 是怎么被用的

### 3.1 `linear_hopper` 的 A/B/C 缓冲

`hopper/linear_hopper.cuh:144-181` 为输入/权重/残差/输出各定义一个 `smem_tma` 别名，例如：

```cpp
using InputSmem  = smem_tma<T, B, M, S, SMEM_M_SIZE, INPUT_TMA_TILE_SIZE,
                            TILE_SIZE / INPUT_TMA_TILE_SIZE>;
using WeightSmem = smem_tma<T, B, M, S, OUTPUT_ATOM_SIZE, WEIGHT_TMA_TILE_SIZE,
                            TILE_SIZE / WEIGHT_TMA_TILE_SIZE>;
```

- `OUTER_COL = TILE_SIZE / INPUT_TMA_TILE_SIZE` 对应 TMA wrapper 的 `SMEM_REPEAT_COL_`
  （`tma_2d.cuh:40-45, 73-89`）：池化 warp 用 `SMEM_REPEAT_COL` 循环发多个 TMA，每个 box 用
  `SMEM_STRIDE_` 拉开——**必须**等于布局的 `STRIDE_OUTER_COL`，否则软件寻址与硬件落点错位。
- TMA 载入的起点是 `input_smem(0,0)`（`:252`），即布局的物理 0 偏移；wgmma 描述符
  `A_DESC a_desc(input_smem(0,0))`（`:284`）也是同一个物理指针——**布局、TMA 目的地、wgmma 起点
  三者在源码里就是同一个地址**。

### 3.2 `multitoken_paged_attention_hopper` 的 Q/KV tile

`hopper/multitoken_paged_attention_hopper.cuh:254-267`：

```cpp
using QOSmem = smem_tma<T, 3,3,3, MAX_TOKENS*NUM_QO_PER_KV, 64, (HEAD_DIM+63)/64>;
using KVSmem = smem_tma<T, 3,3,3, KV_TILE_SIZE,            64, (HEAD_DIM+63)/64>;
using Q_DESC = wgmma::mma_descriptor<QOSmem>;
using K_DESC = wgmma::mma_descriptor<KVSmem, false>;
using V_DESC = wgmma::mma_descriptor<KVSmem, true>;   // MN-major
```

同一 `KVSmem` 类型喂给两个 descriptor：`K_DESC`（MN-major=false）与 `V_DESC`（MN-major=true）——
对应 `wgmma.cuh:52-85` 的 `MNmajor` 两条分支（同一 `b`，但 LBO/SBO 与 swizzle mode 位不同）。即
**同一个布局类型，被 K 和 V 两种 MMA 消纳方式共享**，靠 `mma_descriptor<SMEM, MNmajor>` 的第二个
模板参数区分。

---

## 4. 实验：H100 上验证 `smem_tma` == 硬件 128 B TMA swizzle

> 源码 `src/tma_smem_layout.cu`；原始输出 `harness/tma_smem_layout.out.txt`；目标卡 H100 / sm90a，
> 固定空闲 device 3。做法：`cuTensorMapEncodeTiled` 建 2D bf16 tensor map（`SWIZZLE_128B`），
> 核内 `cp.async.bulk.tensor.2d` 载入一块 `boxRow×64` 到 SMEM，再用**逐字复制**的 `smem_tma` 布局
> 读回逻辑 tile，与源逐位比对；`boxRow∈{8,16,32,64}`、非零 tile 偏移、以及四组负对照。

结果（`max|Δ|` 为 0 表示逐位还原）：

| 组 | 配置 | `max|Δ|` | 判定 |
|---|---|---|---|
| A1–A4 | 128B swizzle + `smem_tma<3,3,3>`，boxRow 8/16/32/64 | **0.0** | ✔ 布局与硬件一致（含 8 行原子重复） |
| B1 | 128B swizzle + **恒等**布局 `smem_tma<0,*>`（B 不匹配） | 224.0（448/512 错） | ✘ B 必须匹配 |
| C1 | NONE swizzle + 恒等布局 | **0.0** | ✔ 无 swizzle 时恒等正确 |
| D1 | **NONE** swizzle + `smem_tma<3,3,3>`（反向不匹配） | 224.0（448/512 错） | ✘ 反向也错 |
| E1 | 128B + `smem_tma<3,3,3>`，tile 偏移 @(24,64) | **0.0** | ✔ 与 tile 全局坐标无关 |
| **F1** | 128B + `<3,3,3>`，SMEM 目的基址 `mod 1024 = 0` | **0.0** | ✔ |
| **F2** | 同上，基址 `mod 1024 = 128` | 222.0 | ✘ 整体平移一行 |
| **F3** | 同上，基址 `mod 1024 = 256` | 166.0 | ✘ 平移两行 |
| **F4** | 同上，基址偏移 1024（`mod 1024 = 0`） | **0.0** | ✔ 回到对齐即正确 |

物理 dump（A1）给出了 swizzle 的直接证据：逻辑 `row r` 的第 `c` 个 16 B 块落在物理块 `c ^ (r&7)`
（即 `icol ^= irow`）；`row0` 是恒等、`row1` 交换相邻 16 B 块、`row7` 整行 8 块反转。

**结论**：`smem_tma<3,3,3>` 的 `get_swizzled_offset` 与硬件 `CU_TENSOR_MAP_SWIZZLE_128B` 的
「16 B 块 XOR 行」模式**逐位一致**；`B` 必须与 TMA swizzle 匹配（B1/D1）；且**这对错是「静默错」**
（不报错、不崩溃，只是矩阵错位）。

---

## 5. 实验发现的关键坑：swizzle 是「按 SMEM 绝对地址」算的，基址须对齐 1024 B

F2/F3 揭示：TMA 的 swizzle **不是**按「tile 内局部行」算的，而是按 SMEM 的**绝对地址位**。

- 硬件：128 B swizzle 对 SMEM 字节地址取「`[7:9]` 位（8 行原子内的行号）XOR 进 `[4:6]` 位（16 B 块号）」
  （`fp8_group_gemm_sm100.cuh:606-612` 的源码注释原话：「the hardware to misinterpret the swizzled
  layout when the dynamic smem base has **non-zero bits in positions [7:9]**」）。
- 软件：`smem_tma::get_swizzled_offset` 按 `logical_idx` 的**局部**行算 `icol^=irow`。

二者只在**基址 1024 B 对齐**（`[7:9]` 位为 0）时一致。基址 `mod 1024 = 128` 时，硬件把「绝对行号」
当作 `local_row + 1`，XOR 模式整体平移一行——正是 F2 测到的 `max|Δ|≠0`。

**MPK 生产代码显式做了这件事**（这也是我第一版实验用 `extern __shared__ __align__(128)` 会失败的
原因）：

- `hopper/linear_hopper.cuh:88`：`smem = (uintptr_t(smem_ptr) + 1023) / 1024 * 1024;`
  且此后每个 stage 缓冲偏移都以 `+1023)/1024*1024` 上取整（`:92-107`）。
- `hopper/linear_swapAB_hopper.cuh:94`、`hopper/multitoken_paged_attention_hopper.cuh:239` 同写法；
  后者还在 `:173` 注释「since smem is 1024 bytes aligned, S_Q_OFFSET is set to zero」。
- `blackwell/linear_fp8_sm100.cuh:146-149` 用
  `DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 && ...)` 静态断言每 stage 大小 1024 对齐、
  `:100` 用 `extern __shared__ __align__(1024)`。
- `blackwell/fp8_group_gemm_sm100.cuh:606-613`、`cute/hopper/gemm_ws_mpk.cuh:98-99` 手动上取整。
- `blackwell/mla_prefill_tp8_sm100.cuh:223-229` 的注释点破**为什么需要手动做**：
  「the dynamic SMEM starts after ~7 KB of static shared (**TaskDesc buffers**), leaving the
  extern `__shared__` base only 128-aligned. Round `sb` up to 1024...」

**这条坑对 megakernel 的意义**：常驻 kernel 里 `extern __shared__` 的基址取决于**前面静态 SMEM
（TaskDesc 缓冲等）的大小**——也就是说，往常驻 kernel 里**多加/改一个 task**（或改 `TaskDesc` 布局），
可能把动态 SMEM 基址的 `[7:9]` 位从 0 变成非 0，从而**静默**破坏所有未显式对齐的 TMA/`smem_tma`
task 的地址计算。这与 `docs/51` §7 的「布局类型就是 SMEM 的 ABI」是同一层约束的**地址侧**补充：
不只要 `(B,M,S)` 兼容，还要**基址对齐**兼容。生产代码快照里每个 TMA task 都自己 `+1023` 上取整，
正是为了不被邻居的静态 SMEM 尺寸带偏——看似冗余的「每文件一遍」，实为**去中心化地各自兜底**。

---

## 6. 与 `docs/51` 的对照：两种布局层的契约性质不同

| 维度 | `smem_row`（Ampere，`docs/51`） | `smem_tma`（Hopper/Blackwell，本文） |
|---|---|---|
| 布局语义 | 行/列主序平铺，`STRIDE` 与 `COL` 解耦 | TMA box：`ROW×INNER_COL` 子矩阵 × `OUTER_COL` 块 |
| swizzle 谁算 | **只有软件**（`load_smem`/`ldsm` 都按布局算偏移） | 软件 + TMA 硬件 + wgmma 硬件**三方共同** |
| 对齐约束 | 16 B（`cp.async`）/按 `ldmatrix` 模式 | **基址 1024 B**（地址位 `[7:9]` 参与 swizzle） |
| 错配后果 | 编译期可见的错、或 `ldsm` 读到错位 | **静默错**（TMA 照常完成、wgmma 照常算） |
| 消费者 | 20 个手写 attention/GEMM 身体 | 10 个 TMA 身体（Hopper + Blackwell） |

一句话：`smem_row` 是「软件独占解释权的 SMEM 布局」；`smem_tma` 是「软硬共管、以硬件 swizzle 为准」
的**描述符型布局**。后者的正确性无法只靠读软件代码保证，必须知道硬件按什么位算——这正是本文在
目标卡上把它测出来的原因。

---

## 7. 判据映射与 megakernel 视角

- **判据贡献：只 C3（融合范围）**。`smem_tma` 是 device body 内部「生产者（TMA）→ 消费者（wgmma）
  SMEM 交接」的底层契约，不涉及 host 提交（C1）/常驻（C2）/设备内协调（C4）/去中心化调度（C5）。
- **对 megakernel 的正面含义**：一个 `(B,M,S)` 类型把「TMA、wgmma、软件」三处对齐到同一约定，
  使得**段内融合（T2）**的生产者/消费者可在编译期被证明一致——这是 MPK 敢把整图 task 编进一个
  kernel 而不会错位的基石之一（配合 `docs/07` 的 warp-specialization 流水）。
- **对 megakernel 的负面含义（成本侧）**：地址侧的对齐是**全局耦合**的——`extern __shared__` 基址
  由所有静态 SMEM 用户共同决定。常驻 kernel 融合的 task 越多，越容易把某 task 的 TMA 基址带偏；
  生产解法是**每个 TMA task 各自 `+1023` 上取整**（去中心化兜底），代价是每 task 至多浪费 ~1 KB
  SMEM。这与 `docs/12`/`docs/14`（共驻足迹）是同一「资源足迹」主题的 SMEM 地址侧切面。

---

## 8. 局限（诚实标注）

- **实验只覆盖 128 B swizzle（B=3）**：MPK 生产只用 `B=3`（`task_register.cc:1194-1196` 硬编码），
  故 B=1/2（32/64 B）的 `M,S` 取值未验证；F 组只测了 bf16/单 box，未测 `OUTER_COL>1`
  （多 box 拼接，靠 `tma_2d` wrapper 的 `SMEM_REPEAT_COL` 循环）。
- **只走读一个头 + 4 个相邻文件**：`smem_tma` 的 10 个消费者只核了 `linear_hopper` 与
  `multitoken_paged_attention_hopper` 两处的实例化语义，其余只列坐标、未逐处核对
  `OUTER_COL` 与 `SMEM_REPEAT_COL` 的一一对应。
- **未跑 MPK 本体**：本文验证的是「逐字复制的 `smem_tma` 在目标卡上与硬件 TMA 一致」，
  未重跑 `docs/09` 的端到端 `mirage`（`B2` z3-solver 版本互斥仍在）。
- **`tma_2d/3d/4d` 的 wrapper 语义未逐行**：本文只用到 2D `SMEM_REPEAT=1` 的情形；
  `SMEM_REPEAT_ROW`、3D/4D box、`GMEM_STRIDE_COL_` 等未展开。
- **对齐坑只作静态记录**：`fp8_group_gemm_sm100.cuh:606` 的机制注释与本文实验相互印证，
  但「加一个 task 会否真的把某生产路径带偏」本机不可编译 SM100、未实证（只记不判）。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`。
- `include/mirage/persistent_kernel/tasks/hopper/smem_layout_tma.cuh`（129 行）：
  `smem_tma` 定义 `:19-129`、`get_swizzled_offset` `:51-80`、3D 访问 `:82-98`、
  2D 访问 `:101-119`、1D 访问 `:121-124`、`operator[]` `:126-128`、`MIRAGE_DEBUG_HOPPER` `:66-77`。
- B→三处硬件字段：`tasks/hopper/tma_2d.cuh:266-270`（tensor map swizzle）、
  `tasks/hopper/wgmma.cuh:46-91`（mma_descriptor swizzle/LBO/SBO，`MNmajor` 两条分支 `:52-85`）、
  `smem_layout_tma.cuh:51-80`（软件）。
- 生产硬编码：`src/kernel/task_register.cc:1194-1196`（B=M=S=3、`TMA_CP_ASYNC_SIZE=64`）、
  `:1208-1240`（TMA_A/B/OUT 的 `smem_tma` 形状）。
- 1024 B 对齐契约：`tasks/hopper/linear_hopper.cuh:88,92-107`、`linear_swapAB_hopper.cuh:94`、
  `multitoken_paged_attention_hopper.cuh:173,177-239`、`tasks/cute/hopper/gemm_ws_mpk.cuh:98-99`、
  `tasks/blackwell/linear_fp8_sm100.cuh:100,146-149`、
  `tasks/blackwell/fp8_group_gemm_sm100.cuh:606-613`、`tasks/blackwell/mla_prefill_tp8_sm100.cuh:223-229`。
- 实验：`src/tma_smem_layout.cu`、`harness/tma_smem_layout.sh`、`harness/tma_smem_layout.out.txt`
  （目标卡 H100/sm90a，device 3）。
- 相关走读：`docs/51`（Ampere SMEM/DMEM 布局，本文补其 §6 的 Hopper TMA 缺口）、
  `docs/47`（Hopper `linear_hopper` task 内流水，本文补其 SMEM 描述符一层）、
  `docs/07`（`linear_hopper` 精读）、`docs/37`/`docs/56`/`docs/57`（`_sm100` 后缀不等于专属）。
  实验交叉引用：`docs/04` §19（TMA multicast，本文用同一 `cp.async.bulk.tensor` 原语但目标是布局正确性）。
- 相关坑（`ROADMAP.md`「已知坑」）：16 B `cp.async` SMEM 对齐、「同义实现多套并存」族；
  本文新增一条 **TMA swizzle 基址 1024 B 对齐**坑。
- **与 `mega-moe` 专题的交叉引用**：DeepGEMM 侧 TMA swizzle 由 CUTLASS 的 `Swizzle<B,M,S>` 在
  类型层统一表达、SMEM 切分由 `layout/*.cuh` 的 stage 结构体完成（见 `../mega-moe/docs/92-smem-layout.md`）；
  本文讲的是 **MPK 自研手写路径**里同一个 `(B,M,S)` 如何被 TMA/wgmma/软件三方共管，**不重复**。
