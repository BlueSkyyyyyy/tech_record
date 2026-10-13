# 138. DeepGEMM 启发式共用的「元素位宽 / TMA swizzle 模式」公共选择器与 `GemmDesc` 的 dtype→MmaKind 管道

> 第 28 篇（P18-1）把 `get_best_config` 的**决策骨架**讲透了（枚举 `Layout` → 打分 → 取最小，
> 再派生 storage/pipeline/launch），并把 `csrc/jit_kernels/heuristics/config.hpp` 的五个结构体列了张表。
> 但那张表只给了**字段名**，没讲这些派生量**从哪来**；而整个 `heuristics/` 目录里还有一个文件
> **从没被单独读过**：`csrc/jit_kernels/heuristics/utils.hpp`（32 行）。它是全仓最短的启发式文件，
> 只干两件事：把 `MmaKind` 翻成**逻辑位宽**、把 `(block_size, elem_size)` 翻成 **TMA swizzle 模式**。
> 这两件小事却是**每一个** SM90/SM100 kernel 的 K 块宽度和 TMA 描述符 swizzle 的**唯一出处**。
>
> 本篇把这两个函数、`GemmDesc` 的 dtype→`MmaKind` 派生管道、以及 Python 侧公开的布局别名层补到
> 与前面各篇同深度，并**在 H100（sm90）真机直接 include 真实头文件**逐一核对。
>
> 目标卡代号敏感：全文用「目标卡」指代。

参考源码（DeepGEMM commit `057ca59`，`Public release 26/09/30 (#462)`，v2.8.1）：

- `csrc/jit_kernels/heuristics/utils.hpp`（32 行）：`get_num_element_bits`（`:12-19`）、
  `get_swizzle_mode`（`:21-30`）。
- `csrc/jit_kernels/heuristics/config.hpp`（190 行）：`GemmDesc`（`:12-86`）、`is_mxf4_mma`（`:35-37`）、
  `get_mma_kind`（`:39-43`）、`get_smem_pack_factor`（`:45-47`）、`check_validity`（`:49-61`）、
  `expected_*` 回退（`:29-33`）、`ensure_zero_padding`（`:26`）。
- 调用点：`sm90.hpp:60`（block_k）、`sm90.hpp:132-139`（storage swizzle）、
  `sm100.hpp:32`（block_k）、`sm100.hpp:175-190`（storage swizzle + pack factor）；
  `sm90_tf32_hc_prenorm_gemm.hpp:33-42`、`sm100_tf32_hc_prenorm_gemm.hpp:31-40`、
  `sm90_bmk_bnk_mn.hpp:28`、`sm100_bmk_bnk_mn.hpp:27-28`。
- 对照：`heuristics/mega_moe.hpp:73-86/117/214-216`（Mega MoE **不用**共用函数，自带
  `get_num_mma_elem_bytes` 并硬编码 swizzle 128）；`impls/sm100_fp8_fp4_mega_moe.hpp:69-76`
  （宽度减半 ⇒ swizzle 128→64）。
- 配套：`csrc/utils/math.hpp:11`（`kPackedFP4 = torch::kInt8`）。
- 配套对齐算术：`csrc/utils/math.hpp:13-27`（`ceil_div`/`align`/`get_tma_aligned_size`）。
- 配套 dtype 加倍：`csrc/apis/layout.hpp:66-73`（`check_ab_fp8_fp4` 的 `k/mn *= 2`）。
- Python：`deep_gemm/utils/layout.py`（13 行）。
- 实测：`src/mma_kind_swizzle.cu` + `harness/exp_mma_kind_swizzle.{sh,out.txt}`（H100/sm90 真机）。

---

## 1. 为什么这两个 3 行函数值得单独成篇

先给「它们在链路里的位置」。第 28 篇那张图里，`get_best_config` 的第三、四步是：

```
Layout(block_m,n,k,cluster)  ──►  StorageConfig(load/store block + 三个 swizzle mode)
                             ──►  PipelineConfig(smem_size / num_stages)
                             ──►  LaunchConfig(threads / 划分)
```

其中：

- **`Layout.block_k` 的值不是枚举出来的**——它是 `sm90.hpp:60` / `sm100.hpp:32` 一行算出来的：
  ```cpp
  const int block_k = 128 * 8 / get_num_element_bits(desc.get_mma_kind());
  ```
- **三个 `swizzle_*_mode` 也不是枚举出来的**——它们在 `sm90.hpp:132-139` / `sm100.hpp:175-190`
  由 `get_swizzle_mode(内维尺寸, 元素字节)` 现算。

也就是说，`heuristics/utils.hpp` 里那两个函数，是**所有 GEMM（稠密 / m-grouped / k-grouped /
Batched / MQA logits / prenorm）K 块宽度与 TMA swizzle 的唯一出处**——第 108 篇拆的
`scheduler/gemm.cuh` 决定「每个 SM 做哪个块」，而这两个函数决定「那个块在共享内存里长什么样」。
把它们讲清，`Layout`/`StorageConfig` 才算真正闭合。

但全专题此前只**顺带引用**过它们（第 97 篇引 `get_swizzle_mode`、第 103 篇引
`get_num_element_bits`），从没解释过「为什么是 `128*8`」「为什么 swizzle 只有 4 档」
「`16` 是什么意思」。本篇补上。

---

## 2. `get_num_element_bits`：逻辑位宽，与 `128*8/bits` 的 1024-bit 不变量

`heuristics/utils.hpp:12-19`：

```cpp
static int get_num_element_bits(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 16;
        case MmaKind::MXFP8FP4: return 8;
        case MmaKind::MXF4:     return 4;
        default: DG_HOST_UNREACHABLE("Unknown MMA kind");
    }
}
```

它把 `MmaKind`（`deep_gemm/include/deep_gemm/common/types.cuh:7-11` 的三个枚举：`BF16=0`、
`MXFP8FP4=1`、`MXF4=2`）翻成**逻辑位宽**：BF16 每元素 16 bit、MXFP8FP4 每元素 8 bit、
MXF4 每元素 4 bit。

注意这是**逻辑**位宽，不是存储宽度：`MmaKind::MXF4` 的 4 bit 元素在内存里是**两个一字节打包**
（见 §4），`get_num_element_bits` 完全不关心这件打包的事。

### 2.1 为什么 `block_k = 128 * 8 / bits`——1024-bit 不变量

把上式和三种 `MmaKind` 代进去：

| `MmaKind` | `get_num_element_bits` | `block_k = 128*8/bits` | `block_k * bits` | `block_k` 的**字节**宽 |
|---|---:|---:|---:|---:|
| `BF16` | 16 | 64 | 1024 | 64×2 = **128 B** |
| `MXFP8FP4` | 8 | 128 | 1024 | 128×1 = **128 B** |
| `MXF4` | 4 | 256 | 1024 | 256×0.5 = **128 B** |

**不变量是 `block_k * bits == 1024`，也就是「每个 K 块的一行恰好 128 字节」**。这不是巧合：

- TMA swizzle 的最大档就是 **128 B**（§3）。要让 K 块内维取到最大 swizzle，它必须 ≥128 B 且是
  128 的倍数；`block_k*bits = 1024 bit = 128 B` 正好卡在边界上。
- 同一份 kernel 源码要通吃 BF16/FP8/FP4，唯一随 `MmaKind` 变的就是「每元素占多少 bit」，
  于是把 `block_k` 反比于位宽，**让所有 `MmaKind` 的 K 块字节宽一致（128 B）**，kernel 里的
  swizzle/寻址逻辑才能一份代码通吃。

### 2.2 与 Mega MoE 的局部对偶：`128 / elem_bytes`

`heuristics/mega_moe.hpp` 里有一份**平行实现**（`:80-82`）：

```cpp
static int get_num_mma_elem_bytes(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::BF16 ? 2 : 1;
}
```

并由此算 block_k（`:117`）：`const int block_k = 128 / get_num_mma_elem_bytes(mma_kind);`

它用的是「**smem 每元素字节数**」（BF16 2、其余 1），而不是「逻辑位宽」。两条路给出：

| `MmaKind` | 共享 `128*8/bits` | Mega `128/elem_bytes` | 一致？ |
|---|---:|---:|---|
| `BF16` | 64 | 64 | ✅ |
| `MXFP8FP4` | 128 | 128 | ✅ |
| `MXF4` | **256** | **128** | ❌（但 Mega MoE **永远不用** `MXF4`） |

分歧之所以「无害」，是因为 Mega MoE 只支持 `fp8xfp8` / `fp8xfp4`（`parse_mma_kind`，
`heuristics/mega_moe.hpp:73-78`），两者都归到 `MmaKind::MXFP8FP4`——FP4 权重被**在 smem 里
展开成 8-bit**，所以「每元素 1 字节」成立，`block_k=128` 与共享函数一致。Mega MoE 之所以另起一套
`get_num_mma_elem_bytes`，是因为它关心的是「**smem 里**搬多少字节」，而不是「MMA 逻辑位宽」。

H100 实测（Part A）：三条共享路径 `block_k*bits` 全为 **1024**；Mega 路径在 `MXF4` 上确实给出
**128 vs 256** 的分歧，其余一致。

---

## 3. `get_swizzle_mode`：`{128,64,32,16}` 里最大的可整除者

`heuristics/utils.hpp:21-30`：

```cpp
template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving
    // 16B actually means non-swizzling (but interleaving)
    for (const int& mode: {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}
```

含义只有三句：

1. **凑出内维字节宽** `block_size * elem_size`（`block_size` 传的是「内维的元素个数」，
   见 §6 的调用点——K-major 传 `block_k`、MN-major 传 `load_block_m/n`）。
2. 返回 `{128, 64, 32, 16}` 里**第一个**能整除它的值——因为从大到小遍历，就是**最大的可整除档**。
3. 若连 16 都除不尽则 `DG_HOST_UNREACHABLE`。

### 3.1 「16 = 非 swizzle 仅交错」与 `>0` 的含义

注释点明两件反直觉的事：

- **返回值 `>0` 的统一含义是「有交错（interleaving）」**，不是「有 swizzle」。`16` 这一档
  **并不做 swizzle XOR**，只是把元素按 16 B 交错排布（TMA 的「无 swizzle」模式）。
- **128/64/32 才是真 swizzle**（对应 `cute::Swizzle<3,4,3>`/`<3,3,3>`/`<2,4,3>` 那几档 XOR
  键不同宽）。这就是为什么第 56 篇讲 TMA 描述符时会看到 `if (swizzle_mode != 0) smem_inner_dim
  = swizzle_mode / elem_size;`——**0 才是「完全不交错」，而 `get_swizzle_mode` 从不返回 0**
  （它的下界是 16）。

### 3.2 为什么下界是 16，且实际不会 `UNREACHABLE`

TMA 的原子是 16 B。只要 `block_size * elem_size` 是 16 的倍数，就一定命中某一档。
`block_size` 来自枚举出的 block（都是 16 的倍数）或 `block_k`（64/128/256），`elem_size`
∈ {1,2,4,8}，乘积必为 16 的倍数——所以 `UNREACHABLE` 只是防御。

H100 实测（Part B）对 `block_size ∈ [1,1024] × elem ∈ {1,2,4,8}` 共 **4096** 组穷举，与
「闭式定义 = `{128,64,32,16}` 里最大可整除者」**0 处不符**；`UNREACHABLE` 恰好出现在
`block_size*elem` 不能被 16 整除的 **3136** 组（与闭式预测完全一致）。典型档位：

```
bs=16  e1=16 e2=32  e4=64  e8=128
bs=32  e1=32 e2=64  e4=128 e8=128
bs=64  e1=64 e2=128 e4=128 e8=128
bs=128 e1=128 ...                  (全 128)
```

---

## 4. 两个「位宽」不是同一个位宽：逻辑 bits vs 存储 element_size

这是本篇最值得记住的一点。同一个 kernel 里同时存在两个「元素宽度」概念：

- `get_num_element_bits(MmaKind)` 用**逻辑位宽**（BF16 16 / MXFP8FP4 8 / MXF4 4）算 `block_k`；
- `get_swizzle_mode(inner, elem_size)` 用**存储字节** `c10::element_size(dtype)` 算 swizzle。

对 FP4 二者冲突：逻辑 4 bit，但存储是**打包**的——`csrc/utils/math.hpp:11` 把 `kPackedFP4`
定义成 `torch::kInt8`，因为两个 FP4 塞进一个字节，所以 `element_size == 1`（不是 0.5）。
SM100 的处理是：**在调 `get_swizzle_mode` 前先把逻辑 `block_k` 除以 `get_smem_pack_factor()`
（MXF4 为 2），换算成物理 smem 字节数**（`sm100.hpp:176-184`）：

```cpp
const int pack_factor = desc.get_smem_pack_factor();          // MXF4 => 2, else 1
const int swizzle_mode_a = get_swizzle_mode(
    (desc.major_a == cute::UMMA::Major::K ? layout.block_k : load_block_m) / pack_factor,
    c10::elementSize(desc.a_dtype));                          // packed FP4 的 elemSize == 1
```

把三条 K-major 路径代进去，结果**全部落在 128**——这正是 §2.1 那个不变量想要的效果：

| `MmaKind` | dtype | `block_k` | `elem_size` | `pack_factor` | `block_k/pack` | swizzle |
|---|---|---:|---:|---:|---:|---:|
| BF16 | bf16 | 64 | 2 | 1 | 64 | **128** |
| MXFP8FP4 | fp8 | 128 | 1 | 1 | 128 | **128** |
| MXF4 | packed fp4(int8) | 256 | 1 | 2 | 128 | **128** |

H100 实测（Part D）三种 K-major 路径的 SM90/SM100 swizzle **全为 128**。但 **MN-major 的 FP4 是例外**：
`load_block_m / pack_factor = 128 / 2 = 64` ⇒ swizzle **64**（实测 `sm100_swizzle_mn=64`）——
因为 MN-major 时内维是 M 而不是 K，宽度只有 128 个逻辑元素、打包后 64 字节，够不上 128 B。
这与 §6 里 Mega MoE 的「宽度减半 ⇒ swizzle 减半」是同一个机制。

---

## 5. `GemmDesc` 的 dtype→`MmaKind` 派生管道与 `check_validity`

`heuristics/config.hpp` 的 `GemmDesc` 不只是「字段容器」——它有三个**派生**函数和一个**闸门**，
它们和 `Heuristics` 一起决定「这次调用合法不合法、走哪种 MMA」：

```cpp
bool is_mxf4_mma() const { return a_dtype == kPackedFP4 and b_dtype == kPackedFP4; }   // :35-37
MmaKind get_mma_kind() const {                                                          // :39-43
    if (a_dtype == torch::kBFloat16) return MmaKind::BF16;
    return is_mxf4_mma() ? MmaKind::MXF4 : MmaKind::MXFP8FP4;
}
int get_smem_pack_factor() const { return is_mxf4_mma() ? 2 : 1; }                      // :45-47
```

三个要点的「为什么」：

1. **`get_mma_kind` 只看 `a_dtype == kBFloat16`**（不查 `b_dtype`）。这是「BF16 GEMM 必两端 bf16」
   这一契约在前置检查里被强制（见下），所以只看一侧即可。
2. **只有 `fp4×fp4` 才是 `MXF4`**；`fp8×fp4` 落 `MXFP8FP4`。这与 Mega MoE 一致
   （`parse_mma_kind` 把 `fp8xfp8`/`fp8xfp4` 都归 `MXFP8FP4`）——即「A 是 FP8 时，`MmaKind` 与
   权重是否 FP4 无关」。
3. **`get_smem_pack_factor` 就是 §4 里那个「÷2」**：只有 `MXF4`（两端都打包 FP4）才有 2。

### 5.1 `check_validity`：四类硬约束

`config.hpp:49-61`：

```cpp
void check_validity() const {
    if (get_mma_kind() == MmaKind::BF16) {
        DG_HOST_ASSERT(a_dtype == kBFloat16 and b_dtype == kBFloat16);
    } else {
        DG_HOST_ASSERT(a_dtype == kFloat8_e4m3fn or a_dtype == kPackedFP4);
        DG_HOST_ASSERT(b_dtype == kFloat8_e4m3fn or b_dtype == kPackedFP4);
    }
    // FP8 D implies dynamic per-32 UE8M0 SFD, only exposed for batched GEMMs
    DG_HOST_ASSERT(cd_dtype == kBFloat16 or cd_dtype == kFloat or
                   (cd_dtype == kFloat8_e4m3fn and gemm_type == GemmType::Batched and
                    not with_accumulation));
    DG_HOST_ASSERT(num_sms % 2 == 0);   // :60
}
```

- **dtype 组合**：BF16 路径必须两端 bf16；block-scaled 路径两端各 ∈ {FP8, packed FP4}。
- **FP8 输出 D** 只在 `Batched` 且**非累积**时开放（因为它是「动态 per-32 UE8M0 的 SFD 输出」，
  见第 31 篇的 epilogue 量化）。
- **`num_sms % 2 == 0`**：与 SM90/SM100 的 2-CTA cluster、TMA multicast 成对语义绑定
  （`sm90.hpp:71-79` 的 cluster 枚举、第 108 篇 `get_swizzled_block_idx` 的成对修补）。

H100 实测（Part C）用真实 `GemmDesc` 逐例核对（负例打印真实 `file:line`）：

| 组合 | `get_mma_kind` | 合法 |
|---|---|---|
| bf16/bf16/bf16 | `BF16(0)` | ✅ |
| fp8/fp8/bf16、fp8/fp8/float | `MXFP8FP4(1)` | ✅ |
| fp4/fp4/bf16 | `MXF4(2)`，pack=2 | ✅ |
| fp8/fp4/bf16 | `MXFP8FP4(1)`，pack=1 | ✅ |
| fp8/fp8/**fp8**，Normal | — | ❌ `config.hpp:57` |
| fp8/fp8/fp8，Batched acc=F | — | ✅ |
| fp8/fp8/fp8，Batched acc=T | — | ❌ `config.hpp:57` |
| **bf16**/fp8/bf16 | `MXFP8FP4` | ❌ `config.hpp:51`（`b_dtype` 断言失败） |
| bf16/bf16/bf16，`num_sms=131` | — | ❌ `config.hpp:60` |

### 5.2 `expected_*` 回退与 `ensure_zero_padding`

- `expected_m/n/k/num_groups`（`:29-33`）的 **0 表示回退到真实形状**（`get_expected_m()` 等）；
  这让「启发式按期望形状选核、kernel 按真实形状执行」解耦（第 28 篇已用它解释 m-grouped PSUM）。
- `ensure_zero_padding`（`:26`）是 **SM100 m-grouped PSUM** 的「gap 行是否保证为零」契约
  （第 12/13/91 篇的 PSUM padding）。

---

## 6. 调用点全景：谁在用它、传什么内维

`get_swizzle_mode` 的**第二个参数是「内维的元素数」**，所以每个调用点都要先判 major：

| 调用点 | 内维 | 传参 |
|---|---|---|
| `sm90.hpp:133-136` | K-major→`block_k`，MN-major→`load_block_m/n` | `get_swizzle_mode(inner, elementSize(dtype))` |
| `sm90.hpp:138-139` | C/D 输出 | `cd_dtype != float ? get_swizzle_mode(store_block_n, elem) : 0`（FP32 输出**不 swizzle**，直接给 0） |
| `sm100.hpp:177-184` | 同上，但 K-major 先 `/pack_factor` | + MXF4 的 ÷2 |
| `sm90/sm100_tf32_hc_prenorm_gemm.hpp` | prenorm 的 A/B/C/D | `block_k`/`block_n`，`sizeof(float)`（TF32） |
| `sm90/sm100_bmk_bnk_mn.hpp:27-28` | 批归约 GEMM | `block_k`/`block_n` 与 `a/d.element_size()` |

一个值得记下的细节：SM90 的 **C/D 输出在 FP32 时 swizzle 直接给 0**（`:138-139` 的三元），
而 `get_swizzle_mode` 从不返回 0——**0 是「完全不对齐/不交错」的哨兵**，只在「不 swizzle 的
FP32 输出」这一处显式写入。这解释了第 56 篇里 `swizzle_mode != 0` 这个判断的来源。

### 6.1 Mega MoE 对照：为什么它硬编码 128

`heuristics/mega_moe.hpp:214-216`：

```cpp
// NOTES: FP8 activations, FP8 weights, and FP4 weights unpacked to 8-bit in smem all use 128B swizzle
const int swizzle_acts_mode = 128;
const int swizzle_weights_mode = 128;
```

Mega MoE 的 A/B 内维**恒为 128 B**（FP8 激活 1 B/元素 × 128；FP4 权重在 smem 展开成 8-bit ⇒
128×1 B），所以它直接写死 128，不调 `get_swizzle_mode`。**它把「结论」内联了**——因为它的
`get_num_mma_elem_bytes`（§2.2）已经保证了这一点。这不是重复，而是「一条链上的两个等价表达」。

出现**例外**时仍要现算：`impls/sm100_fp8_fp4_mega_moe.hpp:69-76` 里 L1 输出（post-SwiGLU）
宽度减半：

```cpp
// Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
// so the swizzle mode is also halved (128 -> 64).
const auto tensor_map_l1_output = make_tma_2d_desc(..., config.block_n / 2, ...,
                                                   config.swizzle_acts_mode / 2);   // 128 / 2
```

即**内维减半 ⇒ swizzle 减半**——与 §4 里 SM100 MN-major FP4 的 64、§3 的闭式完全同构。

---

## 7. Python 公开布局别名层 `deep_gemm/utils/layout.py`

这 13 行是这套对齐/swizzle 工具**对用户唯一可见的那一面**（`deep_gemm/utils/layout.py:1-13`），
它把 `_C` 里的 7 个布局函数重导出，并加两个别名：

```python
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
```

- 导出的 7 个：`get_tma_aligned_size`、`get_mn_major_tma_aligned_tensor`、
  `get_mn_major_tma_aligned_packed_ue8m0_tensor`、`get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`、
  `set_/get_/get_theoretical_mk_alignment_for_contiguous_layout`。
- **两个别名指向同一个 getter**：m-grouped 的对齐在 M 轴、k-grouped 在 K 轴，但**底层是同一把
  `mk_alignment` 旋钮**，所以三个名字是同一个函数对象（第 26/91 篇的 K-grouped SF 打包、
  第 108 篇的 `get_next_k_group` 都读它）。

H100 实测（Python 段，纯 ast 解析）：导入恰 **7** 个、两个别名都指向
`get_mk_alignment_for_contiguous_layout`（`PY_OK`）。

---

## 8. H100 实测小结

`src/mma_kind_swizzle.cu`（**直接 include 真实 `heuristics/utils.hpp` + `heuristics/config.hpp`
+ `utils/math.hpp`**，nvcc `-arch=sm_90a`，链 torch）共 **43 PASS / 0 FAIL**：

- **Part A**：三种 `MmaKind` 的 `block_k*bits` 全为 **1024**（`128*8/bits` 的不变量）；Mega 局部
  `128/elem_bytes` 在 `MXF4` 上与共享函数分歧（128 vs 256）、其余一致。
- **Part B**：`get_swizzle_mode` 在 **4096** 组 `(block_size,elem)` 上与闭式定义 **0 不符**；
  `UNREACHABLE` 恰好 3136 组（= 乘积不被 16 整除者）。
- **Part C**：`is_mxf4_mma`/`get_mma_kind`/`get_smem_pack_factor` 在 4 组 dtype 上全对；
  `check_validity` 的 10 例接受/拒绝矩阵全对，负例打印真实 `config.hpp:49-61`。
- **Part D**：三条 K-major 路径 SM90/SM100 swizzle **全 128**；MN-major FP4 = 64；`128→64` 半宽效应成立。
- **Part E**：`get_tma_aligned_size(x,elem) == align(x,16/elem)`（elem 1/2/4）全对。
- **Python 段**：`utils/layout.py` 的 7 导入 + 2 别名结构全对。

---

## 9. 边界与结论

- **能跑/能核**：`heuristics/utils.hpp`、`heuristics/config.hpp`、`utils/math.hpp` 全是**架构无关的
  纯 host 代码**（无 UMMA/TMA 指令），H100(sm90) 直接 include 真实头文件即可编译运行——本篇的
  43 项断言全部是真实核对。
- **不能核**：这些 swizzle/block_k 选完后，真正用它们去造 TMA 描述符、发射 UMMA 的部分是
  **SM90/SM100 专属设备代码**（第 56 篇的 TMA 描述符层、第 10/110 篇的 MMA）——本篇只保证
  「选出来的数字对」，不保证「设备端怎么用」。
- **结论**：`heuristics/utils.hpp` 这两个 3 行函数，是「**K 块 128 字节 + 最大 swizzle**」这一
  全仓不变量的唯一出处；`GemmDesc` 的 dtype→`MmaKind` 管道则把这套数字按 dtype 选出来并通过
  `check_validity` 闸门。至此，`csrc/jit_kernels/heuristics/` 目录（`config.hpp` / `common.hpp` /
  `utils.hpp` / `sm90.hpp` / `sm100.hpp` / `mega_moe.hpp` / `mega_gate.hpp`）**全部被专篇覆盖**：
  第 28 篇讲决策骨架、第 3/5 篇讲两个 mega 启发式、本篇补上最底层的公共工具。

> 下一篇增量（opt-in）：可把 `get_tma_aligned_size` 与 `csrc/apis/layout.hpp:101-140` 的
> `check_sf_layout`（TMA 对齐 stride 检查）串成「SF 布局的字节级对齐账」，或从第 133 篇剩余的
> `adjacent_code` 队列挑关键路径定点复核。
