# DeepGEMM 的 TMA 描述符构造层：`runtime_utils.hpp` 的 `make_tma_*`

> 前面 55 篇里，每个 SM90 kernel 的每一行 `cp.async.bulk.tensor` 前面都站着一个 `CUtensorMap`
> —— 它是硬件读张量时唯一知道「张量长什么样、往 smem 里怎么摆」的东西。可这个 128 字节的
> 不透明对象**从哪里来、里面被填了什么**，一直被当成黑盒。本篇补上这一层：
> `csrc/jit_kernels/impls/runtime_utils.hpp`（410 行）里那组 `make_tma_*` 封装。
>
> 它是**纯 host、架构无关**的代码：三方库 `cuTensorMapEncodeTiled` 负责真正的编码，DeepGEMM
> 只做「从 `torch::Tensor` + major + block 形状，翻译成编码器参数」这一件事。因此 H100
> （sm90）能**真机逐字节验证**——本文所有数字都来自 `src/tma_descriptor.cu` +
> `harness/exp_tma_descriptor.{sh,out.txt}`（nvcc 直接 include 真实 `runtime_utils.hpp`）。
>
> 参考坐标：DeepGEMM `057ca59`。核心文件行号均在本机核对。

---

## 1. 为什么需要这一层：描述符 = 一次「编码器参数翻译」

TMA（Tensor Memory Accelerator）的搬运指令 `cp.async.bulk.tensor.*` 不吃普通指针，只吃一个
**128 字节的 tensormap**。硬件按这个对象里的字段决定：全局基准地址、张量的每一维大小与步长、
每次搬运的「盒子」（box）大小、元素步长、交错、smem 的 swizzle 模式、L2 提升、越界填充。

编码器是 CUDA driver 的 `cuTensorMapEncodeTiled`。它本身很好用，但从 DeepGEMM 的视角看，直接调它
有三类麻烦：

1. **参数是「几何」而不是「张量」**：GEMM 关心 `(M,N,K)`、major、`BLOCK_M/N/K`，而编码器关心
   `globalDim/globalStrides/boxDim` 这些索引维度的量，还要区分 inner/outer。两者的换算容易错。
2. **swizzle 和下层的 smem 布局强耦合**：`mode_into_tensor_map_swizzle` 必须和 kernel 里
   `make_smem_desc`（第 49 篇）描述的 swizzle 字节数一致，否则 MMA 读到的数据是错的。
3. **FP4/FP8 的 dtype 与打包语义**：`float_e2m1` 在 smem 里是「解包成 16-bit 槽」还是「保持 4-bit
   紧凑」，决定 `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` 还是 `ALIGN8B`，进而决定 box 维度。

`runtime_utils.hpp` 只干翻译：从 23 行的 `get_inner_outer_dims` 到 402 行的
`make_tma_sf_desc`，全部是 `static` 帮助函数，最终都落到 `make_tma_2d_desc:119-159`（或 3/4/5D）
里那一次 `lazy_cuTensorMapEncodeTiled` 调用。

---

## 2. 输入规整：major → `(inner, outer)`

一切从两个小函数开始：

```
get_inner_outer_dims(major, k, mn)  →  major==K  ? (k, mn) : (mn, k)      // :23-25
get_non_contiguous_dim(major)       →  major==K  ? -2      : -1           // :27-29
```

含义：TMA 的维度 0（`globalDim[0]`，内层、连续）永远是「元素相邻的那一维」。
K-major 的矩阵每行沿 K 连续，所以 inner=K、outer=M/N；MN-major 反过来。`get_non_contiguous_dim`
则给出 `stride()` 该取哪一维当作「外层步长」——这正是 `make_tma_a_desc` 等封装在调用点
（如 `sm100_fp8_fp4_gemm_1d1d.hpp:127`）里传的
`a.stride(get_non_contiguous_dim(major_a))`。

> 这条 `inner/outer` 的二分贯穿所有封装，也解释了为什么 `nn/tn/tt` 入口（第 46 篇）在 host 侧
> 只是把操作数转置后再走同一条 `nt` 路径：**major 转换 = inner/outer 对调 = 描述符里前两维对调**。

`get_compiled_dim:31-40` 是另一件事：它决定某个维度是否作为**编译期常量**进入 JIT 模板
（第 23/33 篇），`ignore_compile_dims` 打开时一律返回 0（第 48 篇）。它只在拼 JIT 源码时用，
不参与描述符编码。

---

## 3. dtype 与 swizzle 映射

### 3.1 dtype

`aten_dtype_to_tensor_map_dtype:85-100`：

| `at::ScalarType` | 条件 | `CUtensorMapDataType` | 枚举值 |
|---|---|---|---|
| `kFloat` | `allow_tf32` | `TFLOAT32` | 11 |
| `kInt` | — | `INT32` | 3 |
| `kFloat` | — | `FLOAT32` | 7 |
| `kBFloat16` | — | `BFLOAT16` | 9 |
| `kFloat8_e4m3fn` | — | `UINT8` | 0 |
| `kPackedFP4`(=`kInt8`,`utils/math.hpp:11`) | `fp4_unpacked_smem=true` | `16U4_ALIGN16B` | 14 |
| `kPackedFP4` | `fp4_unpacked_smem=false` | `16U4_ALIGN8B` | 13 |

实测（本机 driver 570 / CUDA 13.2）逐条命中：`bf16=9 fp32=7 tf32=11 e4m3=0 fp4_un=14 fp4_pk=13`。

两点「为什么」：
- **FP8 映射成 `UINT8`**：TMA 只搬字节，E4M3 就是 1 字节，硬件不解释浮点语义。
- **FP4 有两个形态**：`ALIGN16B` 是「在 smem 里解包成每元素占 16-bit 槽」的形式——这样
  tcgen05/WGMMA 的 `float_e2m1_unpacksmem_t` 能直接读；`ALIGN8B` 是保持紧凑 4-bit。二者决定了
  box 内层维度怎么算（见 §4）。

### 3.2 swizzle

`mode_into_tensor_map_swizzle:102-117`：

```
base != 0 → 断言 base==32 且 mode==128 → SWIZZLE_128B_ATOM_32B   (枚举 4)
base == 0 → 0/16→NONE(0)、32→32B(1)、64→64B(2)、128→128B(3)
```

实测 `0→0 16→0 32→1 64→2 128→3`，`base=32→4`。

- **`16` 与 `0` 都映射成 `SWIZZLE_NONE`**：16 字节以下没有 bank-conflict 问题，硬件不支持 16B
  swizzle，退化成无 swizzle。所以第 18/32 篇里「swizzle 0/16」在描述符层其实一模一样。
- **`base=32` 的 `128B_ATOM_32B` 是 CUDA 13 的新「原子」swizzle**（cute 把它映射到
  `SWIZZLE_BASE_32B`，`cute/arch/copy_sm90_desc.hpp:258`），主要服务 tcgen05 的 32B atom。
  **本机（H100 / driver 570）实测：`cuTensorMapEncodeTiled` 直接返回
  `CUDA_ERROR_INVALID_VALUE`**——所以 `base=32` 路径在 sm90 上不可用（对应 SM100 专属，仅分析）。

---

## 4. `make_tma_2d_desc`：三个非平凡规整

`make_tma_2d_desc:119-159` 是所有 2D 描述符的公共出口。除了一次 `cuTensorMapEncodeTiled`
（`:153-157`）外，它做三件容易忽略的规整：

**(1) swizzle 覆盖内层 box（`:130-131`）**

```
if (swizzle_mode != 0)
    smem_inner_dim = swizzle_mode / elem_size;
```

传进来的 `smem_inner_dim` 只要 swizzle 非 0 就被**无条件覆盖**成 `swizzle字节数/元素字节`。
这就是「内层 box 必须正好等于一个 swizzle 行」的落地处，也是 `make_tma_2d_desc` 的调用方
（如 `make_tma_a_desc`）传 `smem_inner=0` 也不报错的原因。

**(2) FP4 的两种 smem 形态（`:133-140`）**

```
assert(not fp4_unpacked_smem or gmem_inner_dim % 128 == 0);   // ALIGN16B 内层必须是 128 的倍数
if (not fp4_unpacked_smem and swizzle_mode != 0)
    smem_inner_dim = swizzle_mode * 2;                        // ALIGN8B 紧凑：box 元素数翻倍
```

解包形态（ALIGN16B）要求内层 128 对齐（`float_e2m1_unpacksmem_t` 的 `.b4x16_p64` 需要 64B 粒度）；
紧凑形态（ALIGN8B）因为每字节 2 元素，box 内层的**元素数**要乘 2 才等于 swizzle 字节数。

**(3) 16 字节对齐断言（`:128-129`）**

```
assert(data_ptr % 16 == 0);
assert(outer_stride_bytes % 16 == 0);
```

TMA 的全局维度步长以**字节**为单位且必须 16B 对齐；`gmem_strides[0] = outer_stride * elem_size`
（`:145`）。此外 `elementStrides` 恒为 `{1,1}`、`interleave=NONE`、`L2_PROMOTION_L2_256B`、
`FLOAT_OOB_FILL_NONE` 都是写死的（`:146,156-157`）——DeepGEMM 不做 im2col/交错/越界填充。

3D/4D/5D 版本（`:161/204/249`）是同构展开：dims/strides 数组变大，`smem_dim_0` 的覆盖与 FP4 规整
逐字相同，只是传给编码器的 `rank` 变成 3/4/5。

---

## 5. 上层封装：五个角色

真正被 kernel 调用的是下面这些「按角色命名」的薄封装，它们把 `(M,N,K)/block/major` 翻译成
`make_tma_*D_desc` 的参数：

| 函数 | 行号 | 维度 | inner/outer 来源 | 关键点 |
|---|---|---|---|---|
| `make_tma_a_desc` | `:325-344` | 2D | `get_inner_outer_dims(major, shape_k, shape_m*num_groups)` | `num_groups>1` 时断言 `major==K`（`:331`） |
| `make_tma_b_desc` | `:346-365` | 2D | `get_inner_outer_dims(major, shape_k, shape_n)` | `num_groups` 恒并入 **outer**（`:358`） |
| `make_tma_cd_desc` | `:367-382` | 2D | 固定 `(shape_n, shape_m*num_groups)` | 输出**恒 N-major**（第 46 篇） |
| `make_tma_sf_desc` | `:384-402` | 2D | 仅 `major==MN`，`shape_mn` 经 TMA 对齐 | 无 swizzle、按 `gran_k` 分组、4-bit 打包因子 4（`:398`） |
| `make_tma_weights_2d/3d_desc` | `:297-323` | 4D | `(k, n_per_domain, num_domains, 1)` | locality domain 分域（第 24/34 篇） |

几个「为什么」：

- **groups 的折叠方式不同**：A 先把 `shape_m` 乘成 `shape_m*num_groups` 再按 major 拆 inner/outer，
  且 `num_groups>1` 时断言 `major==K`（`:331`）——K-major 下 `groups` 落在 **outer**；B 则把
  `num_groups` 直接乘到 **outer**（`:358`）。这与第 10/12 篇的 wgrad tensormap 原地改基址
  （`+k_start*M`）是同一套坐标：组就是 M/N 维上的一段连续区。
- **C/D 恒 N-major**：输出必须 N-major 是 DeepGEMM 的全局约定（第 46 篇 `check_major_type_cd`），
  编码器这边直接钉死 `(shape_n, shape_m*num_groups)`。
- **SF 无 swizzle 且 `gran_k` 折 4**：`:398` 的
  `ceil_div(shape_k, gran_k * (dtype==kFloat ? 1 : 4))`——FP32 scale 每个 `gran_k` 一个，打包 UE8M0
  则每 4 个 `gran_k` 打成一个 int32（第 25/26 篇的 `pack_fp32_into_ue8m0`）。`shape_mn` 先经
  `get_tma_aligned_size`（`utils/math.hpp:23`）对齐再作为外层 stride。
- **weights 的 4D 形状**：`(k, n_per_domain, num_domains, 1)` 把「SM 局部域」当成一个显式维度
  （第 24/34 篇），这样同一份描述符能让不同域的 SM 各自取自己那半权重。

---

## 6. 真机实验

`src/tma_descriptor.cu` 用 nvcc 直接 include 真实 `runtime_utils.hpp`，分三部分：

### Part 1：参数翻译等价（wrapper vs 直连编码）

对每个 wrapper，我们用**自己的代码**按 §4 的规则重算编码器参数，直接调
`cuTensorMapEncodeTiled`，再和 wrapper 的输出做 **128 字节逐字节比较**。结果：

```
2d_bf16_sw128/64/32/0/16   MATCH
2d_fp32_tf32                MATCH
2d_e4m3_sw128               MATCH
2d_sw128_base32             THREW (invalid argument；SM100 专属)
a_K/a_MN/a_K_g2            MATCH
b_K_g2/b_MN_g1             MATCH
cd_Nmajor_g2                MATCH
sf_mn_fp32                  MATCH (shape_mn=132 = align(131,4))
weights2d_4d                MATCH (num_domains=2)
dtype bf16/fp32/tf32/e4m3/fp4_un/fp4_pk   枚举 9/7/11/0/14/13 全对
```

**16 个包装用例全部逐字节相等**，坐实了「wrapper = 一组确定的编码器参数翻译」这一说法：
`make_tma_*` 没有藏任何额外的字段操作。

### Part 2：真实 TMA 载入 vs `cute::Swizzle` 预测（核心发现）

造一个 bf16 `[64,128]` 张量（值 = `row*128+col`，可按位模式反解坐标），用真实
`make_tma_2d_desc` 造描述符，发一条
`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`（mbarrier 等待），
再把 smem 拷回 host，与 `cute::Swizzle<B,4,3>` 预测的物理布局逐字节比对：

```
swizzle=0    mismatch=0/1024
swizzle=32   mismatch=0/256
swizzle=64   mismatch=0/512
swizzle=128  mismatch=0/1024
```

**结论**：TMA 硬件写入的 smem 物理布局精确等于 `cute::Swizzle<B,4,3>`（B=1/2/3 对应
32/64/128B）作用在「行优先线性字节偏移」上的结果——即 16B 块的索引与 `(offset>>7)&mask` 异或。

**但这里踩到一个真正的坑**：如果 smem 目的地只满足「128B 对齐」（把目标从原子对齐的 `+0`
挪到 `+128`），同一描述符写出来的布局会**整体平移一个原子行**：

```
swizzle=128 smem_off=+128(仅128B对齐)
   按「基址=原子对齐」预测 mismatch=512/1024
   按「基址 bits[7:10)=1」 预测 mismatch=0/1024
```

原因：**swizzle 的 XOR 键取自 smem 的「绝对」字节地址 bits[7:10)**，而不是相对 box 起点的偏移。
所以目的地必须对齐到**整个 swizzle 原子（1024B）**，而不只是 swizzle 宽度（128B）——否则即使
编码器接受了这个（满足 128B 对齐的）地址，物理布局也会整体 XOR 平移。真实 kernel 里 smem 句柄
都按 1024B 对齐分配，正是为了这个。这是一条**看代码看不出来、只有真机跑才会暴露**的约束。

### Part 3：128 字节描述符指纹

变一个输入、看哪些字节变化（用于理解字段布局）：

```
diff swz128->64     bytes: 9 55 65 73
diff swz128->0      bytes: 9 72 73
diff dtype->fp32    bytes: 1 8 9 12 55
diff dtype->e4m3    bytes: 1 9 12 55
diff gi128->256     bytes: 32
diff go64->128      bytes: 36
diff so8->16        bytes: 56 65
diff stride128->256 bytes: 12
diff addr(ptr)      bytes: 1 2
```

可读出几条结构：**地址字段**落在低字节（换基址只动 byte 1-2）；**dtype** 动 byte 1/8/9；
**swizzle** 动 byte 9；全局维度动 32/36，外层步长动 12；box 维度动 55/56/65。字段并非全在固定
64-bit 字里（硬件把多个小字段紧挨着打包），所以**别去手工拼描述符**——一律走 `make_tma_*`。

---

## 7. 与其它篇的关系

- **第 49 篇（SM90 WGMMA 描述符）** 讲的是**另一套** 64 位描述符（`GmmaDescriptor`，smem→tensor
  core）。本篇的是 **128 字节 tensormap**（global→smem），两者的 swizzle 必须一致：TMA 按
  `CU_TENSOR_MAP_SWIZZLE_*` 写、WGMMA 按 `LayoutType::B*` 读，字节数对不上就是错的。
- **第 25/26 篇（SF 布局转换/打包）** 处理 SF 的**值布局**；本篇的 `make_tma_sf_desc` 处理 SF 的
  **搬运描述**（MN-only、无 swizzle、`gran_k` 折 4）。
- **第 10/12/13 篇（grouped GEMM）** 里「按组原地改 tensormap」改的就是这些描述符的低 32 位
  （基址/stride），改完还要 `tensormap.cp_fenceproxy`（第 38 篇）。
- **第 24/34 篇（locality domain）** 的分域 TMA 描述符由 `make_tma_weights_*_desc` 的 `num_domains`
  维承载。
- **第 36 篇（SM100 BF16）** 的合并 stage 与 **第 32 篇（SM90 1D1D/1D2D）** 的 swizzle 起点
  24/26、`kSwizzleCDMode` 都直接对应 `mode_into_tensor_map_swizzle` 的输出。

---

## 8. 结论与边界

- **本质**：`make_tma_*` 是一层**确定性的参数翻译**——输入 `(tensor, major, block 形状, swizzle)`，
  输出一个 128B 描述符。16 个包装用例与独立直连编码逐字节相等。
- **三个隐藏规整**：swizzle 无条件下覆盖内层 box；FP4 的 ALIGN16B/ALIGN8B 决定 box 元素数
  （后者 ×2）；全局步长以字节计且必须 16B 对齐。
- **一条真机才暴露的约束**：TMA swizzle 的 XOR 键取**绝对** smem 地址 bits[7:10)，故目的地要对齐到
  整个 swizzle 原子（128B swizzle → 1024B），否则布局整体平移。`cute::Swizzle<B,4,3>` 在原子对齐下
  精确预测硬件布局（0/256、0/512、0/1024 全对）。
- **边界**：本篇只测了 H100(sm90) 上可编码/可执行的路径。`base=32` 的 `128B_ATOM_32B` 与 FP4 的
  实际端到端载入属 SM100 专属，本机驱动对 `base=32` 直接拒绝；FP4 的 smem 形态规则
  （§4 的 `swizzle*2`、ALIGN16B 的 128 对齐）为源码精读。

> 实验：`harness/exp_tma_descriptor.sh` → `harness/exp_tma_descriptor.out.txt`（H100，32 项全 ALL PASS）。
> 源码：`src/tma_descriptor.cu`。
