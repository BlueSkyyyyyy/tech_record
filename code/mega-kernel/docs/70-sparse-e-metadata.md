# 结构稀疏（2:4）元数据 E 的内容走读：位编码、A 物理槽置换与 gmem E 布局

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/69` 同快照）；
> CUTLASS 侧走 MPK 该快照钉死的 submodule `deps/cutlass`，commit `f3fde583`
> （`~/mirage_mpk_build/deps/cutlass`）。本文补掉 `docs/62`（P148-1）§6 与 `docs/63`（P149-1）
> §7 **明确留的两条尾巴**：前者只走到 E 的 TMEM trait 形状（`tmem_e_frg`）与「低 1 位 `sparse_id2`」，
> 后者只走到 A 操作数的**类型/SMEM 布局**（`sparse_elem`/`SpAtom`/`sparse_smem_desc`）。两文都写着
> 「E 的**硬件**编码（2:4 索引位序、每 atom 的 bit 打包）未展开」「哪些元素被丢弃、E 的索引如何选到
> A 的偶数/奇数槽」**未验证**——本文从 CUTLASS 源码把这两件事钉死，并在 host 上**逐字重放**压缩器的
> 位编码算术（`src/sparse_e_metadata.cu` + `harness/sparse_e_metadata.sh` + `harness/sparse_e_metadata.out.txt`）。
> **纯源码走读 + 1 个 host 侧算术探针**：`SparseConfig` 的常量、CUTE 布局、压缩器的 switch/loop 都是
> `CUTE_HOST_DEVICE`，故整套代数在 host 上可判定；SM90/SM100 张量核本体在本机（目标卡 H100/sm90）
> 不可执行。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）E 的内容是「每 2:4 chunk 一个 4-bit nibble」，直接编码「保留的是哪两个位置」——
即 E→A 的物理槽置换，而不是别的。** 对 16-bit/8-bit A（每 chunk 4 个逻辑元素、保留 2 个），编码器
`MetadataOneChunk2to4`（`sm90_sparse_gemm_compressor.hpp:255-286`）对第 `k` 个物理槽返回的码**恰好等于
被保留元素的逻辑下标**（`case 0..3 → 0b00..0b11`），并 `storage_ |= bits << (2*elt_phy_idx)`——于是
一个 nibble 的 `bits[0:2]` = 物理槽 0 的来源位置、`bits[2:4]` = 物理槽 1 的来源位置。host 重放全部
`C(4,2)=6` 种 2:4 模式（§4）：`[x 0 0 x]→0xC`、`[0 x x 0]→0x9`、`[0 x 0 x]→0xD`……**E 就是一张 4-bit 的
选择表**。TF32 是特例：它每 chunk 只有 **2** 个逻辑元素、保留 1 个，用另一个编码器 `MetadataOneChunk1to2`
（`:226-252`）给一个 **4-bit** 码（`keep 0 → 0b0100`、`keep 1 → 0b1110`）。

**(B)（headline）A 的压缩是 E 的镜像：只存被保留的值，2 个一拍紧挨着放；实测压缩后字节数恰为一半。**
`PhysicalElemsAPerChunk = LogicalElemsAPerChunk / 2`（`sm1xx_sparse_config.inl:109`），压缩后 A 的行内
K 轴被声明成 `(2, K/2)`（`Shape<ElementASparsity, K/ElementASparsity>`，`fill_layoutA`），K-major 步长
`(1, 2)`（对紧挨、chunk 间距 2）、MN-major 步长 `(1, M*2)`。host 用 `StructuredSparseCompressorUtility`
对 M=64/K=128 实测：**raw=16384 B、compressed=8192 B（0.50×）**，物理 `k=64`（§5）。所以「2:4 稀疏把 A
减半」既体现在类型层（`sparse_elem<2,T>`，`docs/63`），也体现在 **布局的 chunk 数减半**。另记一个
<32-bit 专属的**单非零 corner case**（§4.3）：只有唯一非零落在最后一格时才被规范化成 `{0,3}`；重放显示
编码器**从不产出 `pos0 >= pos1`** 的位置对（疑为硬件对 metadata 组合的合法性约束，只记不判）。

**(C)（headline）gmem 里的 E 不是扁平字节数组，而是一个 swizzled CuTe atom；且 MPK 零引用，DeepGEMM 的
「sparse」是另一种东西。** SM100 的 E 布局原子 `TensorEAtom` 按数据类型分支——F16 `((8,2,8),(16,2,4))`、
TF32 `((8,2,8),(8,2,4))`、F8 `(128,128)`、F4 `(128,256)`（`sm1xx_sparse_config.inl:116-131`），M 轴恒 128、
K 轴按 E 元素类型换形；SM90 侧源码**直接引 PTX 定义**
（`sparse-wgmma-metadata-64n32-f16bf16`，`sm90_sparse_config.inl:101`）。consumer audit：**MPK 对
`Sm1xx/Sm90GemmSparseConfig`、`StructuredSparseCompressor` 零引用**；而 **DeepGEMM 里名为 sparse 的两处
（`sm100_sparse_mqa_logits.hpp`、`apis/attention.hpp`）是 block-sparse MQA logits**（按 KV block 选择、
metadata = schedule entry），**与 2:4 结构稀疏无关**——即这条 SM100 结构稀疏栈在 MPK 与 DeepGEMM 里都
**没被用于 tensor-core 2:4**。

---

## 1. 定位：本文在「稀疏三段栈」里的位置

`docs/62`/`docs/63` 把 `tcgen05.mma.sp` 的操作数拆成三段：**A（`sparse_smem_desc` + `SpAtom` + `sparse_elem`，
`docs/63`）**、**E（`tmem_e_frg` trait，`docs/62`）**、**指令描述符 `sparse_flag_` 位（两文）**。本文补的
是横跨三者的**「内容/语义」**：

| 段 | `docs/62`/`docs/63` 走到了 | 本文补 |
|---|---|---|
| A | 类型（`sparse_elem<2,T>`）、SMEM 布局（`SpAtom`）、描述符（=dense） | **压缩后 A 的 gmem 布局与字节数**（`fill_layoutA`，§3/§5） |
| E | TMEM trait 形状（`tmem_e_frg`）、低 1 位 `sparse_id2` | **E 的 gmem 布局原子 + nibble 位编码**（§3/§4） |
| 编译器 | 无 | **压缩器 `SM90StructuredSparseCompressor` 把 dense A→(compressed A, E) 的算法**（§4） |

一句话：`docs/62` 的 `sparse_id2` 是 **E 的 TMEM 地址低位**（选 E 的哪一半，`docs/62` §3.2），而本文的
nibble 是 **E 的内容**（选 A 的哪两个槽）——两者是两个层级，合起来才是「指令怎么找到 E、E 怎么找到 A」。

---

## 2. `SparseConfig`：chunk 大小与 E 的 bit 预算

### 2.1 SM100（`Sm1xxGemmSparseConfig`）

`Sm1xxGemmSparseConfig<ElementAMma, LayoutA, ElementEMma>`（`sm1xx_sparse_config.inl:28`）按 A/E 的元素类型
分类（`IsF16BF16/IsTF32/IsI8/IsF8F6F4/IsF4`，`:68-91`），派生四个关键量。host 实测（`out.txt` §i）：

| dtype | `IsK` | `LogicalAPerChunk` | `PhysicalAPerChunk` | `EBits/chunk` | `EBits/Aelem` | `EAtomM` | `EAtomK` | `EAlignK` |
|---|---|---|---|---|---|---|---|---|
| F16/BF16 | 1 | 4 | 2 | 4 | 2 | 128 | 128 | 128 |
| TF32 | 1 | **2** | **1** | 4 | **4** | 128 | **64** | 64 |
| F8/I8 | 1 | 4 | 2 | 4 | 2 | 128 | 128 | 128 |
| F4 | 1 | **8** | 4 | 4 | 2 | 128 | **256** | 256 |

- `LogicalElemsAPerChunk`（`:106-108`）：CF4 一 chunk 是 8 个 4-bit nibble（一个 uint8 装 2 个），TF32 是 2
  （32-bit 元素，chunk 只横跨 2 个），其余 4——**同一个「2:4」在类型层有不同 chunk 宽度**。
- `PhysicalElemsAPerChunk = Logical/2`（`:109`）：**压缩率恒 2**。
- `ElementEBitsPerChunk = 4`（`:112`，一个 nibble），`ElementEBitsPerElementAMma = TF32 ? 4 : 2`（`:113`）
  ——这是 §4 两个编码器的由来。

### 2.2 SM90（`Sm90GemmSparseConfig`）

`Sm90GemmSparseConfig<ElementAMma, GmmaMajorA, ElementEMma, MinTileShapeK>`（`sm90_sparse_config.inl:45`）
给出同构的常量（host 实测 §ii）：16-bit `lAPerChunk=4/pAPerChunk=2/BlockK=32/NumK=4`、32-bit
`2/1/16/8`、8-bit `4/2/64/2`。差别只在 E 布局原子的写法和 `BlockK = 512/sizeof_bits<A>`（`:104`，「一个
metadata block」的宽度）——与 `docs/63`/`docs/62` 的 `K = 512/sizeof_bits<A>` 同源。**源码在此处直接给出
PTX 定义链接**（`:101`）：

```
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sparse-wgmma-metadata-64n32-f16bf16
```

这是本文所谈 nibble 编码的权威定义处（本文只从 CUTLASS 实现侧重放，未逐位对 PTX 表核验，见 §7）。

---

## 3. E 与压缩 A 的 gmem 布局

### 3.1 E 布局原子 `TensorEAtom`（`sm1xx_sparse_config.inl:116-131`）

host 实测（`out.txt` §iii）：

```
F16    EAtom = ((_8,_2,_8),(_16,_2,_4)):((_128,_16,_2048),(_1,_1024,_32))     cosize=16384 E elems
TF32   EAtom = ((_8,_2,_8),(_8,_2,_4)):((_64,_8,_1024),(_1,_512,_16))        cosize= 8192
F8     EAtom = (_128,_128):(_128,_1)                                          cosize=16384
F4     EAtom = (_128,_256):(_256,_1)                                          cosize=32768
```

要点：**(i)** M 轴恒 128（与 `docs/62` 的「E 的 M 恒 128」一致）；**(ii)** F8/F4 用**扁平** `(_128,K)`
原子，F16/TF32 用**嵌套**原子（M 拆 `(8,2,8)`、K 拆成带小维的形式）——即 E 的物理排布也随元素宽度换形，
不是一个通用「每 chunk 一字节」的数组。SM90 侧同构（`:108-119`，host 实测 §ii）：16-bit
`((8,2,4),(16,2,4))`、32-bit `((8,2,4),(8,2,8))`、8-bit 扁平 `(64,128)`。

> **与 `docs/62` 的关系**：`docs/62` 走的是 **TMEM** 里的 `tmem_e_frg`（指令 `[%9]` 操作数的地址布局），
> 本文走的是 **gmem** 里的 `TensorEAtom`（压缩器写出、TMA 读入的布局）——同一条 E 的两个存储层级。

### 3.2 压缩 A 的布局（`fill_layoutA`，`:236-281`）

`fill_layoutA` 直接构造稀疏 A 的布局（不能走 dense 的 `TagToStrideA`，源码注释 `:238-251`）。K-major：

```
Shape< M, Shape<2, K/2>, L >   Stride< K, Stride<1, 2>, ... >
```

host 实测（M=64,N=1,K=128,L=1，`out.txt` §iv）：`layoutA = (64,(_2,64),1):(128,(_1,_2),0)`——**K 轴被分成
`(2, K/2)`：前 2 是「一拍两个被保留的值」（步长 1，紧挨），后 `K/2` 是 chunk 序号（步长 2）**。MN-major 是
`(64,(_2,64),1):(_2,(_1,128),0)`（M 步长 = sparsity=2）。`fill_layoutE`（`:288`）则把 M 轴原子堆叠 `(EAtom, M/EAtomM)`。

---

## 4. 编码器：dense A → E 的 nibble（核心）

压缩器 `SM90StructuredSparseCompressor`（`sm90_sparse_gemm_compressor.hpp:62`，SM90/SM100/SM120 共用同一
实现，见 `sparse_gemm_compressor.hpp:264-319` 的 selector）。`structure_sparse_compress`（`:297-477`）每
CTA 逐行处理，每行按 chunk 切（`:349-351`），对每个 chunk：

```
:412-424  for elt_log_idx in 0..OneChunkSizeA-1:
            if masked(elem_A) != 0:  non_zero_elt_log_idx[non_zero_cnt] = elt_log_idx;
                                     tACsAC[non_zero_cnt] = elem_A; non_zero_cnt++;
:428-446  if sizeof(ElementAMmaRaw) < 32:          // 单非零 corner case（见 §4.3）
            ...
:448-453  MetadataOneChunk m;  for elt_phy_idx: m.set_metadata_bits(non_zero_elt_log_idx[elt_phy_idx], elt_phy_idx);
          tEsE[0] = m.storage();
```

即：**先把「第几个物理槽 ← 哪个逻辑位置」记进 `non_zero_elt_log_idx[]`，再把该位置编码进 nibble**。

### 4.1 `MetadataOneChunk2to4`：16/8-bit（`:255-286`）

```cpp
switch (elt_log_idx) { case 0: return 0b00; case 1: return 0b01;
                       case 2: return 0b10; case 3: return 0b11; }
storage_ |= metadata_bits() << (2 * elt_phy_idx);
```

**码 = 逻辑下标本身（2 bit）**，物理槽 0 放 `bits[0:2]`、槽 1 放 `bits[2:4]`。host 重放全部 6 种 2:4 模式
（`out.txt` §v）：

```
dense chunk            kept(pos)   nibble
[a b 0 0]  -> {0,1}    ab   0x4   0b0100
[a 0 c 0]  -> {0,2}    ac   0x8   0b1000
[a 0 0 d]  -> {0,3}    ad   0xC   0b1100
[0 b c 0]  -> {1,2}    bc   0x9   0b1001
[0 b 0 d]  -> {1,3}    bd   0xD   0b1101
[0 0 c d]  -> {2,3}    cd   0xE   0b1110
```

**这就是 E→A 槽映射的完整真值表**：硬件读 nibble，把物理槽 `k` 的值放回逻辑位置 `nibble[2k:2k+2]`，其余
位置填零。（「硬件确实这样解题」这一句系由压缩器的逆推得，**未在 SM100 上验证**，见 §7。）

### 4.2 `MetadataOneChunk1to2`：TF32（`:226-252`）

TF32 的 chunk 只有 2 个逻辑元素、1 个物理（`LogicalAPerChunk=2`），码是 **4 bit**（`EBits/Aelem=4`）：
`keep 逻辑 0 → 0b0100`、`keep 逻辑 1 → 0b1110`（host 重放 §v）。代码里的两个 4-bit 常量（`:234/:236`）与
16/8-bit 的「码 = 下标」**不是同一套编码**——同样一个「保留 1 个」的语义，在 TF32 上是两个魔数。它们的
**确切硬件含义无法只从 CUTLASS 源码推出**（只记不判）。

### 4.3 单非零 corner case（`sizeof<32` 才走，`:428-446`）

对 <32-bit 类型，若一个 chunk 只有 1 个非零，代码走两条分支：唯一非零在**最后一格**（`pos==3`）时，把值
搬到物理槽 1、记录 `{0,3}`；否则记录 `{真实位置, 3}` 并在槽 1 放零。host 重放（§v）：

```
only a@0: [a 0 0 0] -> pos {0,3}  0xC   (物理 [a,0])
only b@1: [0 b 0 0] -> pos {1,3}  0xD   (物理 [b,0])
only c@2: [0 0 c 0] -> pos {2,3}  0xE   (物理 [c,0])
only d@3: [0 0 0 d] -> pos {0,3}  0xC   (物理 [0,d])   <- 唯一被「规范化」的一支
```

**观察**：编码器产出的位置对**永不出现 `pos0 >= pos1`**（`{3,0}` 被改写成 `{0,3}` 并把数据搬到槽 1）。
一个自然的解释是「硬件要求 metadata 的两个位置递增」——但这只是**假设**（只记不判，SM100 上不可验）。
这一支也解释了为什么 A 的物理顺序（`tACsAC`）与位置表是**两个同步维护的量**：改 metadata 时数据也搬。

---

## 5. 字节账：压缩 A 恰为一半，E 是额外的小开销

`StructuredSparseCompressorUtility`（`sparse_gemm_compressor.hpp:60`）给出定量的字节数
（`get_compressed_tensor_A_bytes:155` / `get_raw_tensor_A_bytes:166` / `get_tensor_E_bytes:177`）。
host 实测（M=64,N=1,K=128,L=1，`out.txt` §ivb）：

```
A logical M=64 K=128 ; physical m=64 k=64
A bytes: raw=16384  compressed=8192  (ratio 0.50)
E bytes=2048        (metadata M=128 K=16)
```

- **压缩 A = 一半**（`k_phys = K/2`），与类型层 `sizeof_bits<sparse_elem<2,T>>` 减半（`docs/63`）一致。
- **E = 2048 B**，对 16 KB 的 dense A 是 ~12.5%（`metadata M=128` 是 E 原子 M=128 的对齐，`K=16 = 128/8`
  是 `ElementEMmaSparsity=8` 的除）。即 2:4 的收益是「A 减半 − 一小笔 E」——这正是 `tcgen05.mma.sp`
  存在的意义（用 metadata 换 bandwidth）。

---

## 6. Consumer audit：MPK 零引用，DeepGEMM 的「sparse」是另一回事

`harness/sparse_e_metadata.out.txt` 末尾：

```
SM100 SparseConfig (Sm1xxGemmSparseConfig) defs   : 1
SM90  SparseConfig (Sm90GemmSparseConfig) defs    : 1
E encoder structs (MetadataOneChunk*to*)         : 2
StructuredSparseCompressor selector arms          : 3
MPK (f9eb70c) refs Sm1xx/Sm90GemmSparseConfig    : 0
DeepGEMM (057ca59) refs GemmSparseConfig/compressor: 0
DeepGEMM "sparse" files (block-sparse, NOT 2:4)  : 2
```

- **MPK 零引用**：与 `docs/62`/`docs/63` 的结论一致——MPK 的 SM100 device 身体只走 **dense + block-scaled**
  两支，**结构稀疏与 WS 都没接**。
- **DeepGEMM「sparse」≠ 2:4**：DeepGEMM `csrc` 里名为 sparse 的两处（`jit_kernels/impls/sm100_sparse_mqa_logits.hpp`
  与 `apis/attention.hpp`）是 **block-sparse MQA logits**——即「每 query 选 top-k 个 KV block」的
  **block-level 稀疏**（metadata = `ScheduleEntry`/`KVBlockInfo`，参 `get_num_metadata_bytes:20-35`），
  与 `tcgen05.mma.sp` 的 **2:4 结构稀疏在类型/指令层毫无关系**。这是一个容易踩的**同名陷阱**：两个生产
  代码库都出现「sparse」，但一个是「没有」、另一个是「另一种」。

---

## 7. 与相邻文档的关系 + 判据映射

- `docs/62`（`tmem_e_frg`）：本文补它的**内容侧**——`docs/62` 是 E 的 **TMEM 地址布局** +「低 1 位
  `sparse_id2`」，本文是 E 的 **gmem 布局 + nibble 位编码**（`sparse_id2` 选 E 的哪一半、nibble 选 A 的哪两个）。
- `docs/63`（A 操作数）：本文补它的**压缩侧**——`docs/63` 是 A 的**类型/SMEM 布局**（`sparse_elem`/`SpAtom`），
  本文是 A 的 **gmem 压缩布局 + 字节减半**与「E 如何指回 A 的槽」。两文合起来 = `tcgen05.mma.sp` 的
  **A↔E 双向契约**。
- `docs/60`（block-scaled SF）与 `docs/62`/`docs/63` 同族：都是「CUTLASS 契约 vs MPK 手工路径」的对照；
  本文把 **稀疏栈的最后一环（编码器 + 字节账）** 也钉死。
- **判据映射**：本文只贡献 **C3（设备体内布局与操作数契约）**——它是「一个 task 的 device body 内部/其
  输入准备阶段的稀疏 A、E 契约」，不涉及 launch/常驻（C1/C2）、不涉及去中心化调度（C5）。

---

## 8. 局限（诚实标注）

- **SM90/SM100 指令本体未执行**：`TensorEAtom`/`fill_layoutA` 是**在 host 上对真实 CUTLASS 头求值**得到的
  （`src/sparse_e_metadata.cu`），属「trait/布局规定的形状」；编码器是**按源码 switch/loop 逐字重放**，
  输出的是 **CUTLASS 压缩器会写出的 E**，**不是** SM90/SM100 上观测到的硬件行为。
- **硬件解题方式未验证**：「硬件读 nibble 把物理槽放回逻辑位置」是由压缩器的逆推得，未在硬件上核；
  `TensorEAtom` 的 swizzle 与 TMA 读入的对应也未验证。
- **nibble 语义未与 PTX 表逐位对核**：源码指向 PTX `sparse-wgmma-metadata-64n32-f16bf16`；本文只从 CUTLASS
  实现侧重放，**未逐位对照 PTX 定义表**。
- **TF32 的 4-bit 码只记不判**：`0b0100`/`0b1110` 两个魔数的硬件含义无法从源码推出。
- **单非零 corner case 的「递增」解释是假设**：观测到编码器只产出 `pos0<pos1`，但「这是硬件合法性约束」
  未被任何文档/代码证实。
- **consumer audit 是 grep 级**：计数可能受宏/注释影响；DeepGEMM 的 sparse 定性来自文件/接口名与
  `ScheduleEntry` 结构，未逐行读完其 device 身体。

---

## 9. 参考坐标（可核对）

- CUTLASS（`deps/cutlass` `f3fde583`）：
  - `include/cutlass/gemm/collective/builders/sm1xx_sparse_config.inl`：`Sm1xxGemmSparseConfig:28`、
    类型分类 `IsF4:68 / IsF8F6F4:70 / IsI8:84 / IsF16BF16:86 / IsTF32:89`、
    `LogicalElemsAPerChunk:106-108`、`PhysicalElemsAPerChunk:109`、`ElementEBitsPerChunk:112`、
    `ElementEBitsPerElementAMma:113`、`TensorEAtom_MMA_{F4,S8orF8F6F4,F16,TF32}:116/119/122/125`、
    `TensorEAtom:128-131`、`TensorEAlignmentM/K:138-139`、`static_assert(16B):140`、`IsKMajor:147`、
    `TensorAAlignmentK:148`、`TensorAAlignmentM:156`、`deduce_layoutA:165`、`deduce_layoutE:190`、
    `deduce_layoutA_tag:208`、`fill_layoutA:236-281`、`fill_layoutE:288`；
  - `include/cutlass/gemm/collective/builders/sm90_sparse_config.inl`：`Sm90GemmSparseConfig:45`、
    `LogicalElemsAPerChunk:92`、`PhysicalElemsAPerChunk:93`、`ElementEBitsPerChunk:96`、
    `ElementEBitsPerElementAMma:97`、**PTX 链接注释`:101`**、`BlockK:104`、`TensorEAtom_{32,16,8}bit:108/111/114`、
    `TensorEAtom:117-119`；
  - `include/cutlass/transform/kernel/sm90_sparse_gemm_compressor.hpp`：`SM90StructuredSparseCompressor:62`、
    `ElementEChunk:87`、`ElementESparsityPerChunk:89`、`MaxThreadsPerBlock:109`、
    `MetadataOneChunk1to2:226-252`、`MetadataOneChunk2to4:255-286`、`using MetadataOneChunk:289`、
    `structure_sparse_compress:297-477`（chunk 切分 `:349-351`、找非零 `:412-424`、corner case `:428-446`、
    写 E `:448-453`）；
  - `include/cutlass/transform/kernel/sparse_gemm_compressor.hpp`：`StructuredSparseCompressorUtility:60`、
    `get_metadata_{m,k}_physical:119/128`、`get_tensorA_k_physical:137`、`get_compressed_tensor_A_bytes:155`、
    `get_raw_tensor_A_bytes:166`、`get_tensor_E_bytes:177`、`fill_layout{A,E}_from_compressor:184/188`、
    `StructuredSparseCompressorSelector`（Sm90/Sm100/Sm120 三臂 `:264-319`）。
- MPK（`f9eb70c`）：对 `Sm1xx/Sm90GemmSparseConfig`、`StructuredSparseCompressor` **零引用**（grep 0 次）。
- DeepGEMM（`~/github/DeepGEMM` `057ca59`）：`csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp`
  （`get_num_metadata_bytes:20-35`、`launch_sm100_sparse_mqa_logits:38`）与 `csrc/apis/attention.hpp`
  ——**block-sparse MQA logits，非 2:4**。
- 本文实验：`src/sparse_e_metadata.cu` + `harness/sparse_e_metadata.sh` + `harness/sparse_e_metadata.out.txt`
  （目标卡 H100/sm90a，host 侧，未 launch kernel）。
