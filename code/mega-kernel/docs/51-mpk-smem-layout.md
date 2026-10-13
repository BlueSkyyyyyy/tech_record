# MPK SMEM/DMEM 布局原语走读：共享内存 swizzle 的「三套写法」与全局显存逻辑包装

> 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/50` 同快照）。
> 本文是**纯源码走读（资料分析），本轮无新实验**。
>
> 本篇补掉一条被三处同时留下的尾巴：`docs/44` §10 明写「`smem_row`/`ldsm`/`dmem_row` 布局定义
> （`docs/38`/`docs/39` 已覆盖）**未逐行**」，`docs/48` §7 与 P135-1 §10 又各记一次。`docs/38` §4/§5
> 已经给了 `copy_sm80.cuh`（`cp.async`/`ldmatrix`）与 `dmem_layout.cuh`（三个逻辑包装）的**摘要**，
> 但把 `tasks/ampere/smem_layout.cuh`（514 行，全部手写 attention/GEMM 的共享内存布局层）整体列为
> 「只走 `mma.cuh`」没展开。本文把它逐段打开，并把「布局层」放回 megakernel 视角。
>
> 这是 megakernel **C3（融合范围）** 的底层接口契约：常驻 kernel 里每个 task 的 device 身体都要把
> 数据在 SMEM 上按「拷贝原语写的形状」和「MMA 读的形状」对齐，布局层就是这层约定的集中地。判据贡献
> **只 C3**（device body 内部的实现砖块，不涉及 host 提交/常驻/设备内协调/去中心化调度）。

---

## 0. 定位：为什么布局层值得单列

`docs/04` §39–§56 花了十几节实测「worker 取 task / 数据面载体」，其 device 侧本体是
`common/copy_sm80.cuh` 那 104 行（`docs/38` §4）。但**拷贝原语**只负责「把字节搬进 SMEM」，
它搬进去的**地址**由上一层布局决定：`cp.async`/`ldmatrix`/TMA 都对 SMEM 的**swizzle 模式**敏感。

- `cp.async` 16 B 拷贝本身对 swizzle 无要求，但写进 SMEM 的 `(row,col)→offset` 映射必须与随后
  `ldmatrix` 期望的一致，否则 MMA 读到的是错位的矩阵。
- `ldmatrix.sync.aligned.m8n8.x4`（`copy_sm80.cuh:76-103` 的 `ldsm`/`ldsm_t`）**硬编码**了「每行
  16 B、8 行一组」的取数模式；它只能正确读入「按 128 B swizzle 排布」的 SMEM。

所以在 MPK 里，**布局类型（`smem_row<T,B,M,S,…>` 的模板参数）就是 SMEM 的 ABI**：同一个模板，
`B=M=S=0` 给出**线性无 swizzle**，`B=S=3` 给出**128 B swizzle**——用类型编码硬件布局模式。所有
20 个手写 attention/GEMM 身体（`grep` `smem_layout.cuh` 的 include 者，见 §6）都从这里选布局。

---

## 1. 文件结构：一个结构体三种参数化

`tasks/ampere/smem_layout.cuh`（514 行）在 `namespace kernel` 下定义 **7 个布局结构体 + 1 个偏移
计算器**。按「怎么表达 128 B swizzle」分成**三套写法**：

| 写法 | 结构体（行号） | swizzle 表达 | 使用者 |
|---|---|---|---|
| **A. 位切片 B/M/S** | `smem_row`（`:90-197`）、`smem_col`（`:206-328`） | `idx` 手工按 `B`/`M`/`S` 三段位移后 `icol ^= irow` | 绝大多数身体（25+ 处实例化） |
| **B. 8×64 原子** | `smem_row_tiled`（`:27-81`） | 硬编码 `AtomRBits=3`/`AtomCBits=6`，`c_swz = c_in ^ ((r_in & MASK_B) << M)` | 四个 Ampere paged-attention 变体（§2） |
| **C. CUTLASS 风格** | `SwizzleOffsetCalculator`（`:330-344`）+ `smem_row_2drow`（`:354-406`）/`smem_row_2dcol`（`:416-459`）/`smem_col_2drow`（`:469-512`） | `SwizzleOffsetCalculator::get_phy_offset(idx) = idx ^ ((idx & MASK_YYY) >> S)` | `ampere/linear.cuh`（§3） |

三套写的是**同一件事**（都是 128 B swizzle），但参数化方式、命名、静态断言完全不同。这与
`docs/20`「同一套硬件惯用法（wgmma/mbarrier/UMMA）在仓库里实现了两遍」是同一现象：布局层也
**并行存在多次实现**——这本身就是本专题反复记的工程坑「同名/同义实现多套并存」的又一实例。

---

## 2. 写法 A：位切片 B/M/S（`smem_row` / `smem_col`）

### 2.1 模板参数与语义

```cpp
template <typename T, int B, int M, int S, size_t ROW_, size_t COL_, size_t STRIDE>
struct smem_row { ... };
```

（`:83-97`）。`ROW_`/`COL_` 是逻辑形状，`STRIDE` 与 `COL` **解耦**（允许 padded）；`B`/`M`/`S` 是
swizzle 的位宽参数——**与 CUTLASS `cute::Swizzle<B,M,S>` 同一套约定**。`smem_row` 行主序
（`logical_idx = row*STRIDE + col`，`:109/117/134`），`smem_col` 列主序
（`logical_idx = col*STRIDE + row`，`:226/239/253`）。

### 2.2 swizzle 公式（2D 访问）

`smem_row::operator()(row,col)`（`:132-148`）逐行拆：

```cpp
size_t logical_idx = row * STRIDE + col;
size_t block_idx     = logical_idx >> (M + S + B);       // 每 2^(M+S+B) 一个块
size_t in_block_idx  = logical_idx & ((1 << (M + S + B)) - 1);
size_t irow          = in_block_idx >> (M + S);          // B 位：块内行
size_t icol          = (in_block_idx >> M) & ((1 << S) - 1); // S 位：块内列块
icol ^= irow;                                            // ← swizzle：列块 XOR 行
size_t offset_in_bank = in_block_idx & ((1 << M) - 1);   // M 位：bank 内偏移
size_t phy_offset = (block_idx << (M + S + B)) | (irow << (M + S)) | (icol << M) | offset_in_bank;
```

这就是标准的「块内 `icol ^= irow`」128 B swizzle。三个特例值得记牢：

- **`B=M=S=0` ⇒ 恒等映射**。此时 `M+S+B=0`，`in_block_idx = idx & 0 = 0`，`irow=icol=0`，
  `phy_offset = (idx<<0) = idx`。故 `smem_row<T,0,0,0,…>` 是**线性无 swizzle** 布局。生产里
  `InputSmem = smem_row<T, 0,0,0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>`（`norm_linear_new.cuh:178`）
  正是「SMEM 平铺、给 `cp.async` 写、不经 ldmatrix」的输入缓冲。
- **`B=S=3` ⇒ 128 B swizzle**。`M+S+B=9`（512 元素/块），`irow∈[0,8)`、`icol` 3 位、`offset_in_bank`
  3 位。生产里 `QOSmem = smem_row<T, 3,3,3, MAX_TOKENS*NUM_QO_PER_KV, HEAD_DIM, HEAD_DIM>`
  （`attention_sm100.cuh:251-252`）是「给 `ldsm` 读」的 swizzled 布局。
- **`ZeroBufferSmem = smem_row<T, 0,0,0, 1, 8, 8>`**（`attention_sm100.cuh:250`、
  `dflash_attention_sm100.cuh:288`、`silu_mul_linear.cuh:160` 等）：小的 1×8 线性缓冲，`zero_buffer`,
  配合 `load_smem_with_predict`（`copy_sm80.cuh:59-74`）做**谓词化零填充**的源地址。

### 2.3 三处「只读/调试」辅助没有外部调用者

`smem_row` 除 `operator()`/`at` 外还提供 `get_offset_in_bank`（`:107-114`）、`get_phy_offset`
（`:115-130`）、以及 1D `at(logical_idx)`（`:168-181`）/`operator[]`（`:183-196`）。全仓 `grep`：
`get_offset_in_bank` / `get_phy_offset` / `smem_row::operator[]` **在 `smem_layout.cuh` 之外零调用者**
（`get_phy_offset` 与 `SwizzleOffsetCalculator` 同名但后者是另一路）。它们与
`docs/16`/`docs/20`/`docs/32` 记录的「有定义无调用者」同族，但**性质不同**——这里更像**调试/
推导辅助 API**（`operator()` 内联展开时会被优化掉），本文**只记静态事实、不判取舍**。

---

## 3. 写法 C：`SwizzleOffsetCalculator` 与 `*_2d*` 布局

### 3.1 偏移计算器直接引 CUTLASS

```cpp
template <int B, int M, int S>
struct SwizzleOffsetCalculator {
  static constexpr size_t MASK_YYY = ((1<<B)-1) << (M + S);
  __device__ static size_t get_phy_offset(size_t logical_idx) {
    // refer to https://github.com/NVIDIA/cutlass/blob/e6e2cc.../cute/swizzle.hpp#L76-L79.
    return logical_idx ^ ((logical_idx & MASK_YYY) >> S);
  }
};
```

（`:330-344`）。注意源码在 `:340-341` **显式给出 CUTLASS `cute/swizzle.hpp` 的 commit 锚点**——
这是本快照里少见的「布局层直接对标 CUTLASS」写法（其余两套是 MPK 自己推的）。

### 3.2 三种拆分布局

- `smem_row_2drow<T,B,M,S,ROW,INNER_COL,OUTER_COL>`（`:354-406`，注释 *"Row-major layout with
  split column dimension: OUTER_COL x INNER_COL"*）：把列维拆成 `OUTER_COL×INNER_COL` 再 swizzle；
  `logical_idx = outer_col*INNER_COL + row*(INNER_COL*OUTER_COL) + inner_col`（`:387-388`）。
- `smem_row_2dcol`（`:416-459`）：`INNER_COL = 128/sizeof(T)`（**写死「一行 128 B」**，`:423`），
  带 `stage` 维（注释 *"stage was in row dim"*，`:442`），`STRIDE_STAGE = SIZE` 做多 stage 流水。
- `smem_col_2drow`（`:469-512`）：列主序版，`INNER_ROW = 128/sizeof(T)`，注释 *"stage was in col dim"*
  （`:507`）。

**唯一消费者**：`ampere/linear.cuh:150-151` 的 `smem_row_2dcol`（A/输入缓冲）与 `smem_col_2drow`
（权重缓冲）——即**整合的 GEMM 身体用 CUTLASS 风格布局，其余手写 attention 用位切片写法**。三套
写法里这一套「最贴近 CUTLASS 语义、但只被一个文件用」。

---

## 4. 写法 B：`smem_row_tiled`（8×64 原子）

`smem_row_tiled`（`:27-81`）把原子写死：

```cpp
static constexpr uint32_t AtomRBits = 3;                    // 8 rows
static constexpr uint32_t AtomCBits = 6;                    // 64 cols
static constexpr uint32_t MASK_B   = (1u << B) - 1u;
static_assert(B <= AtomRBits, "B must fit in the 3 row-in-atom bits");
static_assert(M_PLUS_S <= AtomCBits, "M+S must not exceed 6 for 64-col atoms");
```

（`:36-50`）。`operator()(row,col)`（`:63-76`）先算 `tile = r_outer*ColTiles + c_outer`，块内
`c_swz = c_in ^ ((r_in & MASK_B) << M)`，`idx = (tile << 9) | (r_in << 6) | c_swz`。对 bf16 来说
64 列 = 128 B，故 `get_128B_aligned_ptr`（`:57-61`，`uint4*` + `col ^ (row & AtomRMask)`）给出
**按 16 B 粒度的 128 B swizzle 指针**。

**使用者（4 个 Ampere paged-attention 变体）**：`multitoken_paged_attention_32_64.cuh:230/237`、
`multitoken_paged_attention_4_16.cuh:236`、两个 `*_split_kv.cuh`（`:272/279`、`:263`）。即较新的
Ampere split-KV attention 用原子写法，而同族的 **Blackwell `attention_sm100.cuh` 反而只用位切片
写法 `smem_row<T,3,3,3,…>`**（`:251-253`）。两处都写 128 B swizzle，代码完全不同——是「同一硬件惯用法
多套实现」在**同一算子家族内跨代**的实例。

另记两处 `// using QSmem = smem_row_tiled_new<T,3,3,3,…>` 注释（`32_64_split_kv.cuh:281`、
`32_64.cuh:239`）——引用一个**快照里不存在**的 `smem_row_tiled_new`，是被注释掉的迁移残留
（同 `docs/35`/`docs/44` 的「注释指向未来/旧版」族）。

---

## 5. `dmem_layout.cuh` 逐行：全局显存的逻辑包装

`tasks/common/dmem_layout.cuh`（108 行）在 `docs/38` §5 只给了三个包装的摘要，这里补齐：

- `dmem_row<T,ROW,COL,STRIDE>`（`:17-52`）：`operator()(r,c) = &base[r*STRIDE + c]`（`:27-32`）、
  `offset(r,c)`（`:34-39`）、`at(idx)`/`at(r,c)`（`:41-51`）。**`STRIDE` 与 `COL` 解耦**是关键——
  它让 device body 能直接按 padded/带 stride 的全局布局寻址（如 NHD KV cache 的 `PAGE_STRIDE`）。
- `dmem_row_const`（`:54-81`）：同上但 `base_ptr` 为 `const`（只读张量，如 Q/K）。
- `dmem_col_const`（`:83-107`）：**列主序**，`logical_idx = col*STRIDE + row`（`:95/104`）——权重的
  列主序消费（配合 `smem_col`/`smem_col_2drow`）。

与 SMEM 层不同，DMEM 层**不含 swizzle**（HBM/L2 无 bank 冲突），纯粹是「二维逻辑 → 一维地址」
的语法糖。**注意 `dmem_row` 的模板参数顺序是 `<T,ROW,COL,STRIDE>`，而 `smem_row` 是
`<T,B,M,S,ROW,COL,STRIDE>`**（多三个 swizzle 参数）——读 device body 时若把两者参数顺序记混会
静默算错地址，是需要先看 `using` 别名的坑。

消费者（同 `docs/38`）：`attention_sm100.cuh:156-159` 的 `QKV/ KV/ KVCache/ O` 四个 `dmem_*` 别名、
`paged_k_cache_dmem(...)`/`paged_v_cache_dmem(...)`（`:295/297/351/352`）；`dflash_attention_sm100.cuh:299-301`；
以及 `silu_mul_linear.cuh:88-93`（`OutputDmem = dmem_col_const`）等。

---

## 6. 布局 × 拷贝原语 × MMA 的契约链

布局层的价值在于它是**两条路径的交点**：

```
全局张量 ──(dmem_* 寻址)──▶ cp.async/load_smem ──▶ SMEM(smem_* 布局) ──ldsm──▶ 寄存器 fragment ──mma.sync──▶ 累加器
```

- **写侧**：`load_smem(smem_ptr, gmem_ptr)`（`copy_sm80.cuh:43-57`）把 16 B 写进
  `smem_row::operator()` 算出的偏移。`attention_sm100.cuh:279` 就是
  `load_smem(q_smem(dst_row,dst_col), q_dmem(src_row,src_col))`——**左侧 swizzled、右侧带 stride**，
  布局层把两侧的地址都算好。
- **读侧**：`ldsm(src_ptr_Q, q_frag)`（`attention_sm100.cuh:527`）从
  `q_smem(...)` 取 4×8×8 bf16，其内部 PTX `ldmatrix.sync.aligned.m8n8.x4` **假定 128 B swizzle**——
  与 `B=S=3` 的 `smem_row` 一致。`ldsm_t`（trans，`:636`）同理用于 PV 的 V。

**跨代对照**：Hopper 身体改用 TMA（`hopper/tma.cuh`，其 SMEM 描述符自带 swizzle 字段），
`multitoken_paged_attention_hopper.cuh` **不 include `smem_layout.cuh`**（`grep` 证实）；Blackwell
`attention_sm100` 却**回到手写 `cp.async`+`ldsm`+`smem_row`**（`docs/38` §6 已点过：黑well目录仍用
Ampere 手写 `mma.sync`）。所以「布局层的存在」直接对应「该 task 走不走 TMA」——这是
megakernel 里 device 身体的两条实现路线在源码上的分界（同 `docs/38` §4 对 `copy_sm80` vs TMA 的判断）。

---

## 7. 判据映射与 megakernel 视角

- **判据贡献：只 C3（融合范围）**。布局层不引入 host 侧提交（C1）、常驻（C2）、设备内协调（C4）、
  去中心化调度（C5）；它是 **device body 内部**把「多算子如何在 SMEM 上交接数据」写死的砖块。
  在 megakernel 里它承担的职责是：**同一 task 内生产者（`cp.async`）与消费者（`ldsm`+`mma`）之间
  的 SMEM 交接契约**——也就是段内融合（T2）能否正确落地的最底层前提。
- **对「融合边界」的含义**：布局类型把「swizzle 模式」编进模板参数，故**一个 task 的 SMEM 布局是
  编译期固定**的。跨 task 若想复用同一块 SMEM（MPK §5.3 的 paged shared memory，见 `docs/01`
  §4），必须让相邻 task 用**兼容的 `(B,M,S)`**；一旦某 task 要求不同 swizzle，就必须换页/换布局
  ——这从源码层解释了「paged SMEM 为什么按页切而不是任意复用」的一层约束。
- **工程坑（补进 ROADMAP）**：`smem_layout.cuh` 内**三套 swizzle 实现并存**（位切片 / 8×64 原子 /
  CUTLASS 计算器），且 `dmem_*` 与 `smem_*` 模板参数顺序不同；读/移植 device body 时要先看它的
  `using ...Smem`/`...Dmem` 别名，别按名字猜布局或参数序。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：本机为目标卡 H100/sm90，但本文只读代码、未取 device 计数；所有行号为
  快照 `f9eb70c` 静态事实。三套 swizzle 的**数值等价性**只按公式推断（都表达 128 B swizzle），
  **未逐地址比对**是否对所有 `(B,M,S)` 取值完全一致。
- **`smem_col` 的消费者未全表**：本文只核了 `smem_row`/`smem_row_tiled`/`*_2d*` 的主要实例化，
  `smem_col<T,3,3,3,…>`（`silu_mul_linear.cuh:165/167`、`norm_linear_new.cuh:190/192`）只点了坐标，
  未逐处核对其 `ldsm` 读侧是否一一对应。
- **`SwizzleOffsetCalculator` 与 CUTLASS 的等价性**只凭源码注释的 CUTLASS 锚点（`:340-341`），
  未拉取该 commit 的 `swizzle.hpp` 逐行比对。
- **`smem_row` 的辅助 API（`get_phy_offset` 等）无外部调用者**只作静态记录，不判是死码还是调试残留。
- **未测性能**：swizzle 对 bank 冲突/`ldmatrix` 吞吐的量化影响属 `docs/04` 数据面实验范畴，本文不重复。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`。
- `include/mirage/persistent_kernel/tasks/ampere/smem_layout.cuh`（514 行）：
  `smem_row_tiled`（`:27-81`）、`smem_row`（`:90-197`）、`smem_col`（`:206-328`）、
  `SwizzleOffsetCalculator`（`:330-344`）、`smem_row_2drow`（`:354-406`）、`smem_row_2dcol`（`:416-459`）、
  `smem_col_2drow`（`:469-512`）。
- `include/mirage/persistent_kernel/tasks/common/dmem_layout.cuh`（108 行）：`dmem_row`（`:17-52`）、
  `dmem_row_const`（`:54-81`）、`dmem_col_const`（`:83-107`）。
- 布局 include 者（20 文件，`grep smem_layout.cuh`）：`blackwell/attention_sm100.cuh`、
  `blackwell/dflash_attention_sm100.cuh`、`ampere/{multitoken_paged_attention_32_64[_split_kv],
  multitoken_paged_attention_4_16[_split_kv], moe_linear, silu_mul_linear, linear, linear_cutlass,
  norm_linear, norm_linear_new, single_batch_extend, single_batch_gqa, single_batch_decoding}.cuh`、
  `hopper/{linear_hopper, linear_swapAB_hopper, norm_linear_hopper, multitoken_paged_attention_hopper}.cuh`、
  `deprecated/paged_attention.cuh`。
- 关键实例化：`attention_sm100.cuh:250-253`（`ZeroBuffer`/`QOSmem`/`KVSmem` = 位切片 3,3,3）、
  `:156-159`（`dmem_*`）、`:279/527/636`（`load_smem`/`ldsm`/`ldsm_t`）；`linear.cuh:150-151`
  （`smem_row_2dcol`/`smem_col_2drow`）；`smem_row_tiled` 的四个消费者见 §4。
- 相关走读：`docs/38` §4/§5（`copy_sm80.cuh`/`dmem_layout.cuh` 摘要）、`docs/07`（Hopper task 内
  流水）、`docs/44` §10（本文补的尾巴）、`docs/48` §7、`docs/01` §4（paged shared memory）。
  实验交叉引用：`docs/04` §39–§56（数据面载体实测，其 device 本体即 `copy_sm80.cuh`）。
- 相关坑（`ROADMAP.md`「已知坑」）：`cp.async.wait_group` 立即数、16 B `cp.async` SMEM 对齐、
  「同义实现多套并存」族。
- **与 `mega-moe` 专题的交叉引用**：DeepGEMM 侧也有一层「共享内存布局」，但层级不同——
  `../mega-moe/docs/92-smem-layout.md` 讲的是 **DeepGEMM `layout/*.cuh` 的「stage 结构体」**
  （host 算 `num_smem_bytes`、`SharedStorage` 成员顺序/`alignas` 固定切分、TMA swizzle 目的地对齐）。
  本文讲的是 **MPK 的「swizzle 索引函数」**（`(row,col)→offset` 怎么算）。二者都在 megakernel 的
  「SMEM ABI」这一层，但一个从**宿主切分**角度看、一个从**设备寻址**角度看，**不重复**。
