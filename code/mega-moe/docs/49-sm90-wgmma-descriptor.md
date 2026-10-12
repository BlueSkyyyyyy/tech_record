# SM90 WGMMA 层：共享内存描述符、选择器与异步组

> 本专题前面已经拆过 SM90 的四个 kernel：BF16 dense（`docs/29`）、FP8 dense `1D1D/1D2D`（`docs/32`）、
> wgrad/dgrad grouped（`docs/12/13`）、MQA logits（`docs/19/30`）、TF32 prenorm（`docs/16`）。它们**每一行
> MMA 代码都在调同一套底座**——`deep_gemm/include/deep_gemm/mma/sm90.cuh` 的描述符/选择器与
> `deep_gemm/include/deep_gemm/ptx/wgmma.cuh` 的 `fence/commit/wait`。这篇把这个底座单独讲透：
> 一个 64 位的共享内存描述符是怎么把「一块 swizzle 过的 smem」喂给 WGMMA 的、三种数据类型的选择器
> 为什么 `M=64`、`K` 分别取 `16/32/8`、以及 `wgmma.commit_group` + `wgmma.wait_group N` 的异步组语义。
>
> **H100（sm90a）可跑真机**：描述符构造与选择器都是 SM90 设备代码，本机直接编译运行**真实 WGMMA 指令**，
> 13 项描述符位域 + 5 个真实 GEMM 全部逐位对拍通过（`harness/exp_wgmma_desc.sh`）。

## 1. 谁在用这套底座

`grep make_smem_desc / make_gmma_desc` 覆盖了全部 SM90 主 kernel（行号真实核对）：

| kernel | 描述符构造 | 行号 |
|---|---|---|
| BF16 dense | `make_gmma_desc<K/MN,...>` + `advance_gmma_desc_lo` | `impls/sm90_bf16_gemm.cuh:219-220/267-270` |
| FP8 `1D1D` | `make_smem_desc(ptr, 1)` 每 k 重建 | `impls/sm90_fp8_gemm_1d1d.cuh:303-304` |
| FP8 `1D2D` | `make_smem_desc(ptr, 1)` | `impls/sm90_fp8_gemm_1d2d.cuh:221-222` |
| bmk einsum | `make_smem_desc(ptr, 1)` | `impls/sm90_bmk_bnk_mn.cuh:146-147` |
| MQA logits | `make_smem_desc(ptr, to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim*8)` | `impls/sm90_fp8_mqa_logits.cuh:256-261` |
| TF32 prenorm | `make_smem_desc(ptr, 1)`（只看 B） | `impls/sm90_tf32_hc_prenorm_gemm.cuh:222-224` |

选择器（`BF16MMASelector` / `FP8MMASelector` / `TF32MMASelector`）被每个 kernel 通过
`using WGMMA = ...` 引入，例如 `sm90_bf16_gemm.cuh:59`（`WGMMA::M/N/K/kNumAccum` 在 kernel 里到处出现）。

## 2. 64 位 GmmaDescriptor 的位域

描述符是 `cute/arch/mma_sm90_desc.hpp:80-126` 的一个 `union`，5 个字段（`:107-126`）：

```
bit [0,14)   start_address_        smem 地址 >> 4（低 4 位被丢弃 = 16B 粒度）
bit [16,30)  leading_byte_offset_  LBO >> 4
bit [32,46)  stride_byte_offset_   SBO >> 4
bit [49,52)  base_offset_          仅 SWIZZLE_128B/64B 有效（DeepGEMM 恒 0）
bit [62,64)  layout_type_          INTERLEAVE=0 / B128=1 / B64=2 / B32=3（:48-53）
```

`make_smem_desc`（`mma/sm90.cuh:194-209`）就是这五个字段的直接填装，默认 `LBO=0, SBO=1024`：

```cpp
desc.bitfield.start_address_ = (uint_ptr >> 4);
desc.bitfield.layout_type_ = layout_type;
desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
desc.bitfield.base_offset_ = 0;
```

**为什么 `layout_type` 与字节 swizzle 是一个值**：`to_gmma_layout_type`（`:222-235`）把
`kSwizzleMode`（**字节数** 0/16/32/64/128）映射成枚举，其中 `0/16 → INTERLEAVE`、`32 → B32`、
`64 → B64`、`128 → B128`。所以 FP8 kernel 直接写 `make_smem_desc(ptr, 1)` 其实就是在说「这块 smem
按 **128B swizzle** 摆」。这与它们的 TMA 约束一致——`sm90_fp8_gemm_1d1d.hpp:102-103` 断言
`swizzle_a_mode == block_k`，而 fp8 的 `BLOCK_K=128`（字面 128 字节），故 B128、`SBO=1024=8×128`。

MQA logits 是个例外：它的 swizzle 粒度取决于 head dim，于是用
`to_swizzle_cute_type<kHeadDim>()`（`:281-291`，32/64/128 → B32/B64/B128）并显式给 `SBO=kHeadDim*8`。

## 3. 两种「原子」步长：`get_inner_block_atom_size` / `get_gmma_desc_stride_k`

- `get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype>`（`:211-214`）：
  `kSwizzleMode==0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype)`——即「一个 swizzle 原子在 MN 方向有多少元素」。
- `get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype>`（`:216-220`）：
  K-major 取 **1**（K 连续），MN-major 取上面的原子大小。

这两个函数解释了后面 `make_gmma_desc` 里 LBO/SBO 的数值来源。

## 4. `make_gmma_desc`：K-major 与 MN-major 两个分支

`mma/sm90.cuh:243-279` 的 `make_gmma_desc<kMajorMode, BLOCK_MN, BLOCK_K, kSwizzleMode, dtype>`
按主序分两支（`num_non_contiguous = 128/16 = 8`）：

**K-major（`:249-259`）**——先断言 `kSwizzleMode == BLOCK_K * sizeof(dtype)`，即 **K-major 必须把
`BLOCK_K` 一整行铺满 `kSwizzleMode` 字节**（bf16 `BLOCK_K=64`→128B、fp8 `BLOCK_K=128`→128B）：

```
SBO = 8 * BLOCK_K * sizeof(dtype)     // 8 行 × 一整行字节
LBO = 0                               // K 方向只有一个原子
```

**MN-major（`:260-278`）**——`BLOCK_MN_ATOM = get_inner_block_atom_size(...)`：

```
SBO = 8 * BLOCK_MN_ATOM * sizeof(dtype)              // K 方向两步一个原子
LBO = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype)        // MN 方向
若 kSwizzleMode==16（interleave）则交换 SBO/LBO      // :274-275
```

并在 MN-major 分支里 `DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0)`（`:265`）——**起始地址必须落在
原子边界上**。这些都被实测精确命中（§7 的 D06/D07）。

## 5. `advance_gmma_desc_lo`：只改低 32 位的高效步进

`mma/sm90.cuh:237-241`：

```
base + ((offset + mn_idx*BLOCK_K + k_idx*stride_k) * sizeof(dtype)) >> 4
```

返回的是**描述符的低 32 位**（`start_address` + `LBO` 所在字），所以调用方直接
`desc.reg32_[0] = advance(...)`（`sm90_bf16_gemm.cuh:267-270`），**不重算 LBO/SBO/layout**。
三个用法体现它的三段语义：

1. **K-major 内 K 步进**：`k_idx=(k*WGMMA::K)%BLOCK_K`，`stride_k=1`，偏移是「元素×元素字节 >>4」。
   bf16 `K=16` 一步 = `16×2=32B` → `>>4 = +2`（实测 D10）；`k_idx=4`（8B）会被 `>>4` 截断成 +0（D28）。
2. **多 warpgroup 的 MN 偏移**：`mn_idx = math_wg_idx*WGMMA::M`，`+mn_idx*BLOCK_K` 使第 2 个 warpgroup
   指向第 64 行之后（D05：mn_idx=16 → +0x80）。
3. **device 内合并 stage 的原子偏移**：`offset = atom_k_idx*BLOCK_M*BLOCK_ATOM_K`
   （`sm90_bf16_gemm.cuh:266-270`）把描述符推进到第 `atom_k_idx` 个合并原子，是 `docs/29`
   「合并 stage」（`BLOCK_K = BLOCK_K_ * kNumStages/8`，减少 `warpgroup_wait<0>`）能用的关键。

## 6. 选择器与 `ptx/wgmma.cuh` 的异步组

### 6.1 三个选择器

| 选择器 | 行号 | `M` | `K` | `kNumAccum = M*N/128` | 指令形态 |
|---|---|---|---|---|---|
| `FP8MMASelector<N>` | `:32-75` | 64 | 32 | N/2 | `MMA_64xNx32_F32E4M3E4M3_SS_TN` |
| `BF16MMASelector<N, kMajorA, kMajorB>` | `:101-148` | 64 | 16 | N/2 | `MMA_64xNx16_F32BF16BF16_SS<...>` |
| `TF32MMASelector<N>` | `:168-192` | 64 | 8 | N/2 | `MMA_64xNx8_F32TF32TF32_RS_TN` |

三个共性：`M` 恒为 **64**（一个 warpgroup 的 WGMMA 原语行数）、`kNumAccum = M*N/128 = N/2`（每线程
持有的 fp32 累加器个数）。区别：**FP8 `K=32`、BF16 `K=16`、TF32 `K=8`**——`K` 越大一步吞的收缩维越多，
这正是 fp8 kernel 的 `BLOCK_K=128` 只需 4 条 WGMMA、而 bf16 要 4 条（`128/16`）的原因。

`BF16MMASelector` 额外把主序编进指令（`:107-108` 用 `to_sm90_major<kMajorA/kMajorB>()` 选
`Major::K/MN` 的指令模板），所以 A/B 是 K-major 还是 MN-major 直接决定 WGMMA 走哪条硬件路径。
`FP8MMASelector` 只有 `_SS_TN`（A/B 都 smem、K-major）——这就是 `docs/46`「SM90 FP8 必须 A/B 都
K-major」的底层原因（`fp8_fp4_requires_k_major` 在 arch9 恒真）。

`FP8MMA/BF16MMA/TF32MMARS` 的 `wgmma()`（`:22-24/:85-87/:158-160`）只是把
`MMA::fma(desc_a, desc_b, d[0]...d[N/2-1], scale_d)` 展开；`scale_d` 是 **cute 的 `ScaleOut::One/Zero`**
——`Zero` 让该条指令**首次清零累加器**（kernel 里 `k==0` 传 0，其余传 1）。

### 6.2 `ptx/wgmma.cuh`（`:7-23`）

```cpp
warpgroup_arrive();          // wgmma.fence.sync.aligned：寄存器写序，首条 wgmma 前必调
warpgroup_commit_batch();    // wgmma.commit_group.sync.aligned：把已发指令封成一组
warpgroup_fence_operand(r);  // 空 asm "+f"(r)：对累加器寄存器加编译器屏障
warpgroup_wait<N>();         // wgmma.wait_group.sync.aligned N：等到「未完成组数 ≤ N」
```

**异步组语义**：WGMMA 是异步的，`commit_group` 把一批指令入队，`wait_group N` 允许最多 N 组仍在飞。
kernel 的标准用法是「`arrive` → `commit` → `wait<0>`」（`sm90_bf16_gemm.cuh:260/274/278`）；而多级
TMA 流水里会攒到 `wait<kNumStages-1>` 再放行各级。`warpgroup_fence_operand` 每轮都成对出现在
`commit` 前后（`:258-259` / `:275-277`），防止编译器把累加器 load/store 重排出 WGMMA 之外。

**实测**：把 K 循环拆成两批 `commit` + 中间 `wait<1>`（允许 1 组未完成）后结果仍逐位正确
（`bf16s` 用例），说明「分批提交 + 留 1 组在飞」是安全的重叠方式。

## 7. 累加器的寄存器→(m,n) 映射

WGMMA 的 fp32 累加器在寄存器里，落盘/后续 SwiGLU 前必须知道「第 `lane` 个线程的第 `r` 个寄存器
对应矩阵哪一格」。cute 的 `CLayout_64xN`（`cute/atom/mma_traits_sm90_gmma.hpp:433-435`）：

```
Layout<Shape<Shape<_4,_8,_4>, Shape<_2,_2,N/8>>,
       Stride<Stride<_128,_1,_16>, Stride<_64,_8,_512>>>
```

把 stride 解成 `offset = n*64 + m`（**列主序**）可读出映射：

```
lane = t0 + 4*t1 + 32*t2   (t0=lane%4, t1=(lane/4)%8, t2=lane/32)
r    = r0 + 2*r1 + 4*r2    (r0=r%2,   r1=(r/2)%2,  r2=r/4)
m = t1 + 8*r1 + 16*t2        // 64 行
n = r0 + 2*t0 + 8*r2         // N 列
```

直觉：**同一 `lane` 的 4 个寄存器对应 2 个 M 位置 × 2 个 N 位置**；`t2=lane/32` 把 4 个 warp 分到
M 的 4 段 16 行，`r2` 负责把 N 的 8 列一组往外扩。本实验就是按这张表把 `accum[]` 写回 `D[m][n]`，
结果与参考 GEMM **逐位相等**（= 映射正确）。

## 8. H100 实测（`harness/exp_wgmma_desc.sh` / `.out.txt`）

源码：`src/wgmma_desc.cu`。设备代码原样调用真实描述符与选择器；输入输出落盘，Python 独立复刻
位域公式并判对。头文件 sha256：`mma/sm90.cuh = 548fe0c6…`、`ptx/wgmma.cuh = dad341b3…`。

**A) 描述符位域（13 项全对）**

| 用例 | 结论 |
|---|---|
| `make_smem_desc(p,0)` | INTERLEAVE, LBO=0, SBO=1024 |
| `make_smem_desc(p,1)` | B128, LBO=0, SBO=1024 |
| 显式 `LBO=0x1230/SBO=0x4560` | 位域 `0x123/0x456`，base_offset=0 |
| `make_gmma_desc<K,64,64,128,bf16>` | B128/LBO=0/SBO=8×64×2=1024，与 `make_smem_desc(p,1)` **同值** |
| 同上 `mn_idx=16` | `start += 16×64×2 >> 4 = 0x80` |
| `make_gmma_desc<MN,64,64,128,bf16>` | LBO=64×64×2>>4=`0x200`，SBO=8×64×2>>4=`0x40` |
| `make_gmma_desc<MN,64,64,16,bf16>` | atom=16/2=8，swap 后 LBO=`0x8`、SBO=`0x40`、INTERLEAVE |
| `make_gmma_desc<K,64,128,128,fp8>` | SBO=8×128×1>>4=`0x40`，B128 |
| `advance(base,0,16,0)` | `+2`（bf16 一步 K=16=32B>>4）；`k_idx=4` 被 `>>4` 截断 `+0` |
| `advance(base,0,0,64*64)` | `+0x200`（合并 stage 原子偏移） |
| `advance(mn_base,0,4,0)` | `+0x20`（MN-major `stride_k=64`） |

**B) 选择器静态属性**：`BF16MMA<128>` M=64/N=128/K=16/kNumAccum=64；`FP8MMA<128>` K=32；
`TF32MMA<128>` K=8；`get_gmma_desc_stride_k` K-major=1 / MN-major=64；`get_inner_block_atom_size`
sw=32→16、sw=0→64。全部命中。

**C) 真实 WGMMA 数值（5 用例逐位相等）**：手工按 TMA 128B swizzle 铺 smem
（`chunk' = chunk XOR (row%8)`），调真实选择器 + 真实 `wgmma`：

```
bf16  (N=128,K=64)   exact=True  maxdiff=0
bf16b (N=64, K=64)   exact=True  maxdiff=0
fp8   (N=128,K=128)  exact=True  maxdiff=0
bf16s (N=128,K=64, 分两批 commit+wait<1>)  exact=True  maxdiff=0
mn    (A 为 MN-major, N=128,K=64)          exact=True  maxdiff=0
```

这同时坐实了三件事：① 描述符的 LBO/SBO/layout 与 TMA 128B swizzle 的物理布局严格对应；
② `advance_gmma_desc_lo` 的 K 步进/原子偏移公式正确；③ §7 的累加器映射正确。**总判定 `ALL PASS`**。

## 9. 与 SM100 的对照

`docs/35` 拆过 SM100 的 `make_smem_desc` / `make_umma_desc`（UMMA + TMEM），那套描述符里
**scale 因子描述符（SF）单独一块、经 UTCCP 搬进 TMEM**，`SM100_MMA_MXF8F6F4` 的 `K` 也变成 32/64。
本篇的 SM90 WGMMA 是它的上一代：**没有 TMEM、累加器在寄存器、SF 由 kernel 手工乘**、A/B 只能用
smem（SS）或 A 在寄存器（TF32 RS）。两代共有的一条是「描述符把 smem 的 swizzle 讲给 tensor core」，
但 SW90 把 swizzle 编码进 64 位描述符的 `layout_type + LBO + SBO`，比 SM100 的 `swizzle atom` 描述更紧凑。

## 10. 边界与局限

- 本篇只覆盖 **WGMMA 描述符/选择器/异步组**本身；每个 kernel 的 TMA 铺流水、epilogue STSM+store 见
  `docs/19/29/32`。
- 未做：`TF32MMASelector` 的 `RS` 路径数值（需要 A 在寄存器、布局见 `mma_traits:447-452`，
  属 `docs/16` TF32 prenorm 的范畴）；`kSwizzleMode` 取 32/64 的真实数值（本机所有 SM90 kernel 都走
  B128，MQA 的 B32/B64 属 `docs/19`）。
- 实验用的是「手工铺 swizzle」而非 TMA，等价性由「描述符语义 + 与 kernel 相同的 `advance` 调用」
  保证；真实 TMA 铺出的字节布局与手工公式一致（否则逐位对拍不可能全 0）。
