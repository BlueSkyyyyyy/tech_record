# DeepGEMM 的共享内存布局结构层：`layout/gemm.cuh` 与 `layout/mqa_logits.cuh`

> 系列第 92 篇。前面 91 篇把 Mega MoE / GEMM / MQA logits / 各种调度器与测试栈都拆过，
> 但每个 kernel 开头那句 `extern __shared__ ... smem_buffer[]` 之后的**共享内存到底怎么切**、
> host 又**怎么算出 `num_smem_bytes`**，一直没被单独讲过。本篇补上这层。
>
> 涉及文件：`deep_gemm/include/deep_gemm/layout/gemm.cuh`（67 行）、
> `deep_gemm/include/deep_gemm/layout/mqa_logits.cuh`（67 行）；
> 消费方 `impls/sm100_{bf16,fp8_fp4}_gemm_1d1d.cuh` / `impls/sm100_mqa_logits.cuh`；
> host 侧 `csrc/jit_kernels/heuristics/sm100.hpp` / `csrc/jit_kernels/impls/sm100_mqa_logits.hpp`。
> 全部行号以 `~/github/DeepGEMM@057ca59`（Public release 26/09/30）为准。

## 1. 问题：`__shared__` 是怎么被切成一个个 stage 的

DeepGEMM 的 SM100 GEMM / MQA kernel 都不声明静态 `__shared__` 变量，而是：

```cpp
extern __shared__ __align__(1024) uint8_t smem_buffer[];
auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);   // e.g. sm100_fp8_fp4_gemm_1d1d.cuh:163-164
```

`SharedStorage` 就是 from `layout/` 的一个结构体：它把「CD 输出 stage、A/B 输入 stage、SF stage、
各类 mbarrier、TMEM 指针」用一个 C++ 结构体的**成员顺序与对齐**固定下来。这么做有两个直接好处：

1. **切分即类型系统**：不再手算偏移，`smem.a[stage_idx]` 就是第 `stage_idx` 级 A；成员类型
   `a_stage_t = a_dtype_t[LOAD_BLOCK_M * BLOCK_K / pack]` 也把「FP4 每字节 2 元素」写进了类型
   （`layout/gemm.cuh:46-47` 用 `get_smem_pack_factor`，见第 44 篇 `common/packing.cuh:8-12`）。
2. **对齐由编译器保证**：`alignas(1024) cd[...]` / `alignas(kSwizzleAlignment) smem_q[...]` 一出，
   `offsetof` 必然落在 TMA / swizzle 要求的边界上——这正是第 56 篇踩到的「swizzle 目的地要对齐到
   整个 swizzle 原子」在类型层面的表达。

但仅有结构体还不够：**host 申请显存时要知道申请多少字节**，而 host 看不到 `sizeof(SharedStorage)`
（它在为 SM100 编译的 device TU 里）。于是 DeepGEMM 用了**两套策略**，本篇两条都核验：

- **精确镜像**（MQA logits）：host 端 `MQALogitsConfig::smem_size()` 逐字复刻结构体的布局，
  并在 JIT 源码里放一条 `static_assert(sizeof(...) == smem_size)` 把二者钉死。
- **保守预算**（SM100 稠密 GEMM）：host 端 `get_pipeline_config` 用**上界**（barrier 一律按
  `kNumMaxStages=32` 预留）算大小，与结构体真实布局解耦，容差自然落到后面。

## 2. `layout/mqa_logits.cuh`：精确镜像

### 2.1 结构体

`MQALogitsSharedStorage`（`layout/mqa_logits.cuh:10-65`）模板参数是
`<kNumHeads, kHeadDim, BLOCK_Q, SPLIT_KV, UMMA_N, kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t>`，
先推一串 `static constexpr`（`:17-48`），再按序放成员（`:50-64`）：

```
smem_q[qStages][QElemsPerStage]      // alignas(kSwizzleAlignment)
smem_kv[kvStages][KVElemsPerStage]   // alignas(kSwizzleAlignment)
smem_sf_q[qStages][kNumSFQ]          // alignas(128)
smem_sf_kv[kvStages][kNumSFKV]       // alignas(128)
smem_weights[qStages][...]           // alignas(128)
full/empty q & kv barriers, full/empty tmem barriers
tmem_ptr_in_smem
```

几个非平凡常量：

- `kSwizzleAlignment = 8 * kNumQKBytesPerToken`（`:24-25`）：一个 swizzle 原子覆盖 **8 个 token 行**
  的 Q/K（`kNumQKBytesPerToken = head_dim`，FP4 时为 `head_dim/2`）。故 FP8 D=128 → 1024B、
  FP8 D=64 → 512B、FP4 D=128 → 512B、FP4 D=64 → 256B——实测 `alignof` 正是这四个值（§5）。
- `kNumSFQ = align(UMMA_N,128)` / `kNumSFKV = align(SPLIT_KV,128)`（`:26-27`）：SF 的 UTCCP
  对齐单元是 128 元素（第 10/35 篇）。
- `kNumWeightBytesPerRow = align(kNumHeads*2, 16)`（`:34-35`）：权重 `[BLOCK_Q, heads]` BF16 行
  按 16B 补齐（TMA 要求），与 host `num_weight_elements_per_row()`（`sm100_mqa_logits.hpp:156-158`）
  一字不差。

六条 `DG_STATIC_ASSERT`（`:43-48`）把「UMMA_N==align(BLOCK_QH,8)」「Q/KV 每 stage 字节数整除
swizzle 原子」「swizzle 原子与 TMA 对齐都是 128 的倍数」「weight stage 128 对齐」钉在编译期。

### 2.2 host 镜像与 JIT `static_assert`

`MQALogitsConfig::smem_size()`（`csrc/jit_kernels/impls/sm100_mqa_logits.hpp:168-194`）用
`add_region(bytes, alignment)` 的**加一段、对齐一次**循环，逐段复刻上面的成员顺序：

```cpp
add_region(q_stages  * umma_n   * num_qk_bytes_per_token, swizzle_alignment);
add_region(kv_stages * split_kv * num_qk_bytes_per_token, swizzle_alignment);
add_region(q_stages  * num_sf_q  * 4, kTmaAlignment);
add_region(kv_stages * num_sf_kv * 4, kTmaAlignment);
add_region(q_stages  * num_weight_bytes_per_stage, kTmaAlignment);
add_region(num_barriers * 8, 8);        // 3*(q+kv)+2*tmem 个 mbarrier
add_region(4, 4);                       // tmem_ptr_in_smem
return align(num_smem_bytes, swizzle_alignment);
```

这跟结构体的成员顺序、每个成员的 `alignas`、以及最后按最大对齐补齐，是**同一次布局计算**。
所以 JIT 源码直接断言二者相等（`sm100_mqa_logits.hpp:254-256`）：

```
static_assert(sizeof(layout::MQALogitsSharedStorage<{...}>) == {}, "Incorrect MQA logits shared-memory size");
```

即：**`MQALogitsSharedStorage` 的 `sizeof` 就是被 launch 的 `num_smem_bytes`**（`:246` 的
`config.smem_size()`），一条 `static_assert` 保证镜像永远不漂移。这也是 host 侧唯一写「精确镜像」
的地方——一旦改了 device 结构体忘了改 host，第一次编译这条 kernel 就会编译失败。

## 3. `layout/gemm.cuh`：保守预算

### 3.1 两个结构体

`SM100BF16GemmSharedStorage`（`layout/gemm.cuh:15-35`）与
`SM100FP8FP4GemmSharedStorage`（`:42-65`）差别只在「有没有 SF」：

| 成员 | BF16 | FP8FP4 |
|---|---|---|
| `cd[kNumTMAStoreStages]` (`alignas(1024)`) | ✓ | ✓ |
| `a/b[kNumStages]` | ✓ | ✓ (除以 `pack`，FP4=2) |
| `sfa/sfb[kNumStages]` (`uint32`) | ✗ | ✓ |
| `full_barriers` | ✓ | ✓ |
| `sf_full_barriers` | ✗ | ✓ |
| `empty_barriers` | ✓ | ✓ |
| `tmem_full/empty_barriers[kNumEpilogueStages]` | ✓ | ✓ |
| `tmem_overlap_barriers` | ✗ | ✓ |
| `reserved_barriers[kNumStages]` | ✓（注释见下） | ✗ |
| `tensor_core_full_barrier` | ✓ | ✗ |
| `tmem_ptr` | ✓ | ✓ |

两条 `DG_STATIC_ASSERT`（`:21-23` / `:50-52`）都要求 CD stage 与 A/B stage 是 1024 的倍数——
因为 `cd` 是 `alignas(1024)`、且 UMMA 从 smem 取 A/B 需要 1024 对齐的 stage 起点。

BF16 结构体里 `reserved_barriers[kNumStages]` 的注释很关键（`layout/gemm.cuh:31`）：

> `// Keep the BF16 barrier layout aligned with the block-scaled GEMM layout.`

**为什么**：host 的单条预算公式（下一节）给「每 stage 三个 barrier（full + sf_full + empty）」
留了位置，但 BF16 内核没有 `sf_full`。多出来的 `reserved_barriers` 正好占住那个槽，使
**BF16 与 FP8FP4 的每 stage barrier 数都是 3**（§5 Part C 实测 `BF16=3 / FP8=3`）——
于是任何假定「每 stage barrier 步长一致」的 host 侧推理与同一份 `smem_barriers` 常量都能同时
套在两个内核上。

### 3.2 host 预算 `get_pipeline_config`

`SM100ArchSpec::get_pipeline_config`（`csrc/jit_kernels/heuristics/sm100.hpp:209-252`）里，
真正进 `num_smem_bytes` 的是 `smem_extra + num_stages * smem_per_stage`（`:242-251`）：

```cpp
const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 3 + 8;   // = 32*24 + 48 + 8 = 824
const int smem_tmem_ptr = 4;
const int smem_cd = store_m * swizzle_cd_mode * num_tma_store_stages;   // 非 swap_ab 分支
const int smem_a_per_stage = load_m * block_k * elemSize(a) / pack;
const int smem_b_per_stage = load_n * block_k * elemSize(b) / pack;
// 1D1D 才加 sfa/sfb：sf_block_m * block_k / 32
```

三点「为什么」：

1. **barrier 一律按 `kNumMaxStages=32` 预留**（`:210/221`）：这样 host 不必知道最终 `num_stages`
   就能给出上界；`num_stages` 本身也是反推的（`:244-246` 用 `smem_capacity` 除 `smem_per_stage`）。
   于是**实际 barrier 字节必然 ≤ 824**，差额变成余量。
2. **`num_tma_store_stages` 单独算**（`:193-207`）：只有 k-grouped 连续且非 BF16、CD 非 fp32 时
   才降到 1（单级 TMA store 让每 SM 至多一个 bulk store 在飞、省 smem）；否则为 2。
3. **`store_block_n` 由 `swizzle_cd_mode` 反推**（kernel 侧 `STORE_BLOCK_N = swizzle_cd_mode / sizeof(cd)`，
   `sm100_fp8_fp4_gemm_1d1d.cuh:113`）：所以 host 的 `smem_cd = store_m * swizzle_cd_mode * stages`
   恰好等于结构体的 `cd` 字节数（`store_m * store_block_n * sizeof(cd) * stages`）。

### 3.3 为什么 host 预算够用：`used` 而不是 `sizeof`

结构体最后按最大对齐（`alignas(1024)`）补齐，`sizeof` 会比「最后一个成员结束处」大出最多 1023 字节。
但 smem 里只放**一个** `SharedStorage` 实例，尾部补齐不占空间。真正要满足的是：

```
used = offsetof(SharedStorage, tmem_ptr) + sizeof(uint32_t)   <=   host smem_size
```

实测（§5 Part B）这个 slack **恰好等于 `824 − 实际 barrier 字节`**：
因为 host 用 `smem_cd + 824 + 4 + stages*per_stage`，结构体真实占用是
`cd + 实际barrier + 4 + stages*(a+b+sf)`，其余各项一一相等，所以

```
slack = 824 − 实际barrier字节
```

BF16 S8 实测 `used=295148`、host `295740`、slack `592 = 824 − 232`；FP8 S8 slack `584 = 824 − 240`；
FP4 S4 slack `680 = 824 − 144`。也就是说：**host 的保守预算被「留白的 barrier 区」正好撑住**，
`kNumMaxStages` 越大越安全。

## 4. H100 实测

实验：`src/smem_layout.cu`（`nvcc -arch=sm_90a` 直接 include 两个真实 layout 头文件，纯 host 程序），
driver：`harness/exp_smem_layout.sh`，输出：`harness/exp_smem_layout.out.txt`。
结构体只含 constexpr 推导、`alignas` 数组与 cutlass mbarrier，**不含任何 UMMA/TMA 设备指令**，
故虽然用它的 kernel 是 SM100 专属，布局本身可在 H100 上逐字段核验。**总判定 42 PASS / 0 FAIL。**

### Part A：MQA `sizeof == host 镜像`（8 组配置，全部相等）

| 配置 | `sizeof` | `alignof` | `kSwizzleAlignment` | q/kv stages |
|---|---|---|---|---|
| H=64 D=128 FP8 | 203776 | 1024 | 1024 | 3/3 |
| H=128 D=128 FP8 | 203776 | 1024 | 1024 | 3/3 |
| H=64 D=64 FP8 | 105472 | 512 | 512 | 3/3 |
| H=32/16 D=128 FP8 | 203776 | 1024 | 1024 | 3/3 |
| H=64 D=128 FP4 | 227328 | 512 | 512 | 2/8 |
| H=128 D=64 FP4 | 120832 | 256 | 256 | 2/8 |
| H=64 D=64 FP4 | 120832 | 256 | 256 | 2/8 |

`sizeof` 与 host 镜像逐字节相等；`kNumSFQ=128`/`kNumSFKV=384`/`kNumQElementsPerStage` 与独立
host 口径一致；`offsetof(smem_q/smem_kv/smem_sf_q/smem_sf_kv/smem_weights/full_q_barriers)`
全部落在声明的对齐上（如 FP8 D=128：q@0、kv@49152、sfq@196608、w@202752）。注意 FP4 用更深的
Q/KV 流水（`num_q_stages=2`、`num_kv_stages=8`，`sm100_mqa_logits.hpp:137-138`）——stage 少一半
但级数翻倍，故 `sizeof` 反而更大。

### Part B：GEMM `used <= host 保守预算`（8 组配置，全部成立）

| 内核 | 形状 | `used` | host | slack | `sizeof` | 尾部 pad |
|---|---|---|---|---|---|---|
| BF16 | L128x128 K64 S8 | 295148 | 295740 | 592 | 295936 | 788 |
| BF16 | L128x128 K64 S16 | 557484 | 557884 | 400 | 558080 | 596 |
| BF16 | L128x256 K64 S8 | 426220 | 426812 | 592 | 427008 | 788 |
| BF16 | L64x128 K64 S12 | 311628 | 312124 | 496 | 312320 | 692 |
| FP8 | L128x128 K128 S8 | 303348 | 303932 | 584 | 304128 | 780 |
| FP8 | L128x128 K128 S16 | 573876 | 574268 | 392 | 574464 | 588 |
| FP8 | L128x256 K128 S8 | 438516 | 439100 | 584 | 439296 | 780 |
| FP4 | L128x128 K256 S4 | 172180 | 172860 | 680 | 173056 | 876 |

三条规律都对上：① `slack = 824 − 实际 barrier 字节`（BF16 S8 232 → 592；FP8 S8 240 → 584；
FP4 S4 144 → 680）；② `sizeof − used` 就是尾部补齐到 1024（`788/596/…`），**不占 smem**；
③ CD 1024 对齐、A/B 1024 对齐、SF stage 字节数与 host 的 `sf_block_m * block_k / 32` 一致
（FP8 K128 → 512B/stage；FP4 K256 → 1024B/stage）。

### Part C：`reserved_barriers` 的效应

`BF16` 与 `FP8FP4` 结构体各自带 3 个 per-stage barrier（实测每 stage 都是 3），
总 barrier 字节 `BF16=232`（S8/EPI2：`3*8*8 + 2*2*8 + 8`）与 `FP8=240`（`3*8*8 + 3*2*8`）只差
`FP8` 多出来的 `tmem_overlap_barriers`（或 `BF16` 的 `tensor_core_full_barrier`）；
两者都远小于 host 的 824。这坐实了那句注释：**BF16 用 `reserved_barriers` 把每 stage barrier
数从 2 补到 3，以对齐 block-scaled GEMM 的布局**。

## 5. 设计动机小结

- **两套策略各有取舍**：MQA logits 的 stage 数少、成员少，用**精确镜像 + `static_assert`** 换取
  「host 与 device 绝不分家」；稠密 GEMM 的候选 stage 数多、还要在启发式里反推，用**保守上界**
  换取「不必先知道 `num_stages`」，代价是最大 592 字节的留白。
- **切分即契约**：`alignas` 让 TMA / swizzle / UMMA 的对齐要求变成类型属性；`a_stage_t` 的
  `/pack` 让 FP4 打包进类型。改动结构体等于改动契约，host 若不跟就会在第一次 JIT 编译时炸。
- **边界**：本篇只核验**布局**（host 可编译的部分）。使用 `sm100_fp8_fp4_gemm_1d1d_impl` /
  `sm100_mqa_logits` 这些 kernel 本体仍需 SM100（第 25/35 篇），H100 上仅分析。

## 6. 复现

```
cd code/mega-moe
bash harness/exp_smem_layout.sh > harness/exp_smem_layout.out.txt
```

源码 `src/smem_layout.cu` 直接 include `deep_gemm/include/deep_gemm/layout/{gemm,mqa_logits}.cuh`，
编译期即完成结构体实例化与 `offsetof`/`sizeof` 取址，运行期只做数值对拍。
