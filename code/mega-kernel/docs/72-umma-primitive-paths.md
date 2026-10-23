# UMMA 原语多实现并置审计：DeepGEMM 自建 `ptx::` 层 vs MPK 的几条通路 vs CUTLASS atom

> 快照与坐标：MPK = `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（有完整 checkout 在
> `~/mirage_mpk_build`）；CUTLASS = MPK 该快照钉死的 submodule `deps/cutlass` commit `f3fde583`
> （`~/mirage_mpk_build/deps/cutlass`）；DeepGEMM = `~/github/DeepGEMM` commit `057ca59`
> （`Public release 26/09/30`），其 vendored CUTLASS 在 `deep_gemm/include/cute`、submodule 在
> `third-party/cutlass`。本文是 `docs/57`（P143-1，只走 MPK 的 `tasks/blackwell/` 原语层）与
> ROADMAP「下一步候选」点名的**「DeepGEMM 自建 PTX UMMA 封装层与 MPK/CUTLASS 三条 UMMA 原语通路的
> 并置对照」**，也是 `docs/62`/`docs/71`（WS 角度）之外的**全形态并置**。**纯源码走读 + 1 个 host 侧
> 文本审计探针**：SM100 指令/TMEM 本体在本机（目标卡 H100/sm90）不可执行；本文用 host 程序
> `src/umma_paths_audit.cu` 抽读各层源文件里的 `tcgen05.mma` 指令串、出能力矩阵、核对逐字相等性
> （`harness/umma_paths_audit.sh` + `harness/umma_paths_audit.out.txt`）。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）「同一颗张量核指令、几条独立封装」在能力上是严格包含关系：CUTLASS atom（40 个
distinct opcode）⊃ DeepGEMM 自建 `ptx::`（11 个）⊃ MPK 三条通路（4 个）。** 生产实现各自只把「自己
真正用到的形态」抄进自建层——DeepGEMM 自建层没有 sparse（`.sp`）、没有 `tf32`、没有 `i8`、没有
`mxf4nvf4`、没有 2SM-tf32；MPK 连 `tf32`/`i8`/`sparse` 都没有，只保留 `f16`（手写 attention）与
`mxf8f6f4.block_scale`（FP8 线性）。host 审计（§5.1）给出完整 opcode×层矩阵；唯一「自建层有、CUTLASS
没有」的是 DeepGEMM 那枚**从未被调用的 `.mma.ws`**（`docs/71`）。

**(B)（headline）DeepGEMM 自建层的本质是「CUTLASS atom 减去三件机械约束」，不是新指令封装。**
头注释直接自称 *"UMMA versions with relaxed assertions"*（`tcgen05.cuh:11`）。逐条：**(1) 去掉内的
`cute::elect_one_sync()` 守卫**——CUTLASS atom 的 `fma` 内部先 `if (cute::elect_one_sync())` 再发指令
（`mma_sm100_umma.hpp:105`），自建层把「谁发指令」交回调用者（DeepGEMM 调用点自己写
`if (cute::elect_one_sync())`，§5.2）；**(2) 去掉 `#if defined(CUTE_ARCH_TCGEN05_*_MMA_ENABLED)` 编译
门**（CUTLASS `:104`；自建层无）；**(3) 对非 block-scale 的 dense 形态，去掉 CUTLASS 恒传的全零
`disable-output-lane` 掩码 `{%5,%6,%7,%8}`**（CUTLASS `:106/:111`；自建层直接用不带掩码的 PTX 形式）。
模板化也一并去掉——CUTLASS 是 `SM100_MMA_F16BF16_SS<a,b,c,M,N,major...>` 模板；自建层是**非模板
struct**，M/N/类型全由调用方传进来的 `runtime_instr_desc`（CUTLASS 的 `make_runtime_instr_desc`）
承载。TMEM 地址/描述符/累加谓词的约定与 CUTLASS 完全一致（`idescE>>32`、`setp.ne.b32 p, %4, 0`）。

**(C)（headline）同一颗 `mxf8f6f4.block_scale.cta_group::1` 在四个互相独立的代码库里逐字相同（host
证实），构成「同一条 PTX 被抄了四遍」的最强证据。** 四个副本——DeepGEMM 自建 `ptx::SM100_MMA_MXF8F6F4_SS`
（`tcgen05.cuh:46-65`）、CUTLASS `cute::SM100_MMA_MXF8F6F4_SS`（`mma_sm100_umma.hpp:990-1026`）、MPK
`kernel::sm100::SM100_MMA_MXF8F6F4_SS`（`sm100_utils.cuh:127-150`）、MPK
`mirage::blackwell::linear_fp8_sm100::sm100::SM100_MMA_MXF8F6F4_SS`
（`linear_fp8_sm100_sm100_utils.cuh:182-205`）——的 asm 串归一化后**完全一致**
（`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;`），2SM 版
（`cta_group::2`）在提供它的三份里也逐字一致。`docs/57` 说 MPK+CUTLASS「三份封同一颗」，加上
DeepGEMM 自建层是**第四份**。

---

## 1. 定位：本文补哪一层

`docs/57` 走完 MPK 的 SM100 共享原语层（`sm100_ptx.cuh` / `sm100_utils.cuh` / `storage.cuh` /
`tensor_init.cuh`），headline 说「同一组张量核原语实现了不止一遍」。`docs/71` 从 **WS** 这一个角度
把 DeepGEMM 自建层（9 struct）打开、核了调用者数与 WS 死码。本文补的是**把「几条通路」摆在一起**：

| 层 | 文件 | distinct `tcgen05.mma` opcode | 谁在跑 |
|---|---|---|---|
| CUTLASS atom（母本）| `cute/arch/mma_sm100_umma.hpp`（36 个 `SM100_MMA_*` 模板） | **40** | 两条高层路径 + 少数生产实现 |
| DeepGEMM 自建 `ptx::` | `deep_gemm/ptx/tcgen05.cuh`（9 struct） | **11** | DeepGEMM SM100 mega 家族（13 文件） |
| MPK `sm100_ptx` | `blackwell/sm100_ptx.cuh`（裸函数） | **1**（f16） | MPK 手写 MLA decode/MTP |
| MPK `kernel::sm100` | `blackwell/sm100_utils.cuh`（struct） | **1**（mxf8f6f4） | MPK FP8 linear（1d2d / group-gemm） |
| MPK `mirage::…::sm100` | `blackwell/linear_fp8_sm100_sm100_utils.cuh`（struct） | **2**（mxf8f6f4 1SM/2SM） | MPK FP8 linear 身体 |
| MPK MLA 私有 `ptx` | `mla_mtp_decode_tp{2,4,8}_sm100.cuh` | **1**（f16，各抄一份） | MPK MLA-MTP TP 三兄弟 |

（opcode 计数取自 `harness/umma_paths_audit.out.txt` §1；「一行里 CUDA 版本分支」的 `#if` 两形态算两个
opcode，故 DeepGEMM 自建 9 struct → 11 opcode，见 §3。）

---

## 2. CUTLASS atom 是「母本」：40 个 opcode 的能力全集

`cute/arch/mma_sm100_umma.hpp`（1777 行）定义 **36 个 `SM100_MMA_*` 模板结构体**（其中 **12 个**带
`_SPARSE`），覆盖 SM100 `tcgen05.mma` 的几乎全部形态：`tf32`/`f16`/`i8`（S8）/`f8f6f4`/
`mxf8f6f4.block_scale`/`mxf4(.block_scale)`/`mxf4nvf4.block_scale`，每个再有 `SS`（A、B 都在 SMEM）、
`TS`（A 在 TMEM，asm 用 `[%1]`）、`2x1SM`（`cta_group::2`）、`_SPARSE`（`.sp`，多一个 E 的 TMEM 操作数
`[%9]`）等正交变体。host 审计抽到 **40 个 distinct opcode**、其中 **18 个带 `.sp`**、**10 个带 TMEM-A
操作数 `[%1]`**、**28 个带 `disable-output-lane` 掩码 `{...}`**（`harness/umma_paths_audit.out.txt` §3）。

几个关键签名（作为自建层的对照母本）：

```cpp
// cute/arch/mma_sm100_umma.hpp:97-116  SM100_MMA_F16BF16_SS<a,b,c,M,N,major...>
CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a, uint64_t const& desc_b,
                                 uint32_t const& tmem_c, uint32_t const& scaleC,
                                 uint64_t const& idescE) {
  if (cute::elect_one_sync()) {                 // (1) 内部 elect
    uint32_t mask[4] = {0,0,0,0};               // (2) disable-output-lane，恒 0
    asm volatile("... tcgen05.mma.cta_group::1.kind::f16 "
                 "[%0], %1, %2, %3, {%5, %6, %7, %8}, p; ..."
                 :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                    "r"(uint32_t(idescE>>32)), "r"(scaleC),
                    "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])); } }

// :141-159  SM100_MMA_TF32_TS  A 来自 TMEM → asm 用 [%1]
```

block-scale 的 CUTLASS atom（`SM100_MMA_MXF8F6F4_SS:990-1026`）则**不含掩码**、多两个 scale TMEM
操作数：

```cpp
// :1002-1020
fma(desc_a, desc_b, tmem_c, scaleC, idescE, tsfa_addr, tsfb_addr)
  -> "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;"
```

---

## 3. DeepGEMM 自建 `ptx::` 层：9 struct、11 opcode、含一枚死 WS

`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`（219 行，`namespace deep_gemm::ptx`）定义 9 个
`SM100_MMA_*` struct——**非模板**、只有一个 `static void fma(...)`、体内一段 `asm volatile`：

| struct | 行 | opcode | 调用者（除定义） |
|---|---|---|---|
| `SM100_MMA_F16BF16_SS` | `:12-27` | `.cta_group::1.kind::f16` | 3 |
| `SM100_MMA_F16BF16_2x1SM_SS` | `:29-44` | `.cta_group::2.kind::f16` | 3 |
| `SM100_MMA_MXF8F6F4_SS` | `:46-65` | `.cta_group::1.kind::mxf8f6f4.block_scale` | 3 |
| `SM100_MMA_MXF8F6F4_2x1SM_SS` | `:67-86` | `.cta_group::2.kind::mxf8f6f4.block_scale` | 2 |
| `SM100_MMA_F8F6F4_SS` | `:88-104` | `.cta_group::1.kind::f8f6f4` | **0** |
| `SM100_MMA_F8F6F4_2x1SM_SS` | `:106-122` | `.cta_group::2.kind::f8f6f4` | **0** |
| `SM100_MMA_MXF4_SS` | `:124-146` | `.kind::mxf4.block_scale.block32`（CUDA≥12.9）/`.scale_vec::2X` | 3 |
| `SM100_MMA_MXF4_2x1SM_SS` | `:148-170` | 同上 `.cta_group::2` 两形态 | 1 |
| `SM100_MMA_F16BF16_WS_SS` | `:172-187` | `.mma.ws.cta_group::1.kind::f16` | **0** |

要点：

- **11 个 opcode 来自 9 个 struct**，因为两个 MXF4 struct 各有一个
  `#if (__CUDACC_VER_MAJOR__ > 12) || (…>= 9)` 分支（`:137-141` / `:161-165`）：CUDA ≥ 12.9 用
  `mxf4.block_scale.block32`，否则用 `mxf4.block_scale.scale_vec::2X`。
- **`f8f6f4`（非 block-scale 的 FP8/FP6/FP4）与 WS 三枚零调用者**（`docs/71` 已核）。被调用的
  六个形态正好对上 DeepGEMM SM100 家族：`sm100_bf16_gemm.cuh:297`、`sm100_bmk_bnk_mn.cuh:174`、
  `sm100_bf16_mega_gate.cuh:217`、`sm100_bf16_mega_moe.cuh:795`（F16/BF16）、
  `sm100_fp8_fp4_gemm_1d1d.cuh:413-414`、`sm100_fp8_fp4_mega_moe.cuh:918`、`sm100_mqa_logits.cuh:372`、
  `sm100_sparse_mqa_logits.cuh:373`（MXF8F6F4/MXF4）。
- 除 9 个 MMA struct 外还有 4 个 helper：`tmem_load_32dp32b<kNumValues>`（`:190-204`，转调 CUTLASS
  `cute::SM100_TMEM_LOAD_32dp32b{4..128}x`）、`umma_arrive_no_elect`（`:206-209`，`tcgen05.commit`
  且**不像 CUTLASS 那样内部 elect**）、`tcgen05_before/after_thread_sync`（`:211-217`，两条
  `tcgen05.fence`）。**没有 alloc/dealloc/描述符构造**——那些 DeepGEMM 仍调 CUTLASS `cute::UMMA::` /
  `cutlass::arch::` 与自建 `deep_gemm::mma::sm100::advance_umma_desc_lo`（见 §5.2 的「分层」）。

---

## 4. MPK 的几条 UMMA 通路

MPK 没有一统的「UMMA 封装层」，而是**按用途各存一份**（`docs/57` headline 1）：

1. **`sm100_ptx.cuh`（裸函数，`:69-80`）**：`tcgen05_mma(taddr, a_desc, b_desc, idesc, acc)` 直接发
   `tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;`——**只有 f16 一种**，供手写 MLA
   decode/MTP（`mla_decode_sm100.cuh:197/385`、`mla_mtp_decode_sm100.cuh:270/549`）。
2. **`kernel::sm100::SM100_MMA_MXF8F6F4_SS`（`sm100_utils.cuh:127-150`）**：block-scaled FP8，1SM
   一种，被 `linear_fp8_1d2d_sm100.cuh:132`（作 `mma_issue_t`，调用于 `:579`）与
   `fp8_group_gemm_sm100.cuh:1271` 用。
3. **`mirage::blackwell::linear_fp8_sm100::sm100::{SM100_MMA_MXF8F6F4_SS, SM100_MMA_MXF8F6F4_2x1SM_SS}`
   （`linear_fp8_sm100_sm100_utils.cuh:182-230`）**：与 (2) **近乎逐行重复**的另存一套（含 2SM 版），
   被 `linear_fp8_sm100.cuh:487` 用。
4. **MLA 三兄弟各自的私有 `namespace ptx`**：`mla_mtp_decode_tp2/4/8_sm100.cuh` 里**各重定义一份**
   `tcgen05_mma`（tp2 `:109-114`、tp4 `:98-103`、tp8 `:95-100`），既不复用 `sm100_ptx.cuh` 也不互复用
   （`mla_mtp_decode_tp2` 的私有 `namespace ptx` 跨 `:64-127`，含 `elect_sync`/`mbar_*`/`make_desc`）。
5. **另有消费 CUTLASS atom 的两处**：BF16 GEMM 主体走 `cute::SM100_MMA_F16BF16_SS<T,…>`
   （`linear_sm100_mpk.cuh:62`、`moe_linear_sm100.cuh:72`、`fp8_group_gemm_sm100.cuh:513`）；且
   `linear_fp8_1d2d_sm100.cuh:132/134` **同一个 task 里同时用两份**（`kernel::sm100::…` 当
   `mma_issue_t` 直发、`cute::SM100_MMA_MXF8F6F4_SS<T,…>` 当 `mma_plumbing_t` 供
   `make_tiled_mma`/`cute::gemm` 走 CuTe 描述符代数）——是「同一颗指令、两条用途」的最直接证据。

---

## 5. 并置对照

### 5.1 能力矩阵（host 审计，`umma_paths_audit.out.txt` §1/§3）

| 能力 | CUTLASS atom | DeepGEMM 自建 | MPK `sm100_ptx` | MPK `kernel::sm100` | MPK `mirage::…::sm100` | MPK MLA 私有 |
|---|---|---|---|---|---|---|
| distinct opcode | **40** | **11** | 1 | 1 | 2 | 1 |
| `.sp` 稀疏 | 18 | 0 | 0 | 0 | 0 | 0 |
| `tf32` | ✓ | 0 | 0 | 0 | 0 | 0 |
| `i8`/S8 | ✓ | 0 | 0 | 0 | 0 | 0 |
| `mxf4nvf4` | ✓ | 0 | 0 | 0 | 0 | 0 |
| TMEM-A `[%1]`（TS） | 10 | 0 | 0 | 0 | 0 | 0 |
| `disable-output-lane {}` | 28 | 0 | 0 | 0 | 0 | 0 |
| `.ws`（weight-stationary）| 0 | **1**（死码）| 0 | 0 | 0 | 0 |
| 内部 `elect_one_sync` | ✓ | **无** | 无 | 无 | 无 | 无 |
| `CUTE_ARCH_*` 编译门 | ✓ | **无** | 无 | 无 | 无 | 无 |
| 模板参数（M/N/major）| ✓ | **无**（idesc 承载）| 无 | 无 | 无 | 无 |

一句话：**自建层是母本的「去模板、去守卫、去掩码、去编译门」子集**；三份 MPK 通路是更小的子集；
唯一「子集里多出来的」是 DeepGEMM 那枚 WS 死码。

### 5.2 签名/操作数形状差异（逐条）

以 `kind::f16` 为例，把四份摆在一起：

```cpp
// CUTLASS 母本  mma_sm100_umma.hpp:105-115
if (cute::elect_one_sync()) {
  uint32_t mask[4] = {0,0,0,0};
  asm(... "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;"
      :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
         "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}
// DeepGEMM 自建  tcgen05.cuh:19-25 （调用点自己 elect，见 body）
asm("... tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(desc>>32)), "r"(scale_c));
// MPK sm100_ptx  sm100_ptx.cuh:69-80 （同样无掩码、无内部 elect）
asm("... tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc));
```

三处细节值得记：

- **「谁 elect」搬到了调用者**：DeepGEMM 的调用点自己写 `if (cute::elect_one_sync())`
  （`sm100_bf16_gemm.cuh:301`、`sm100_bf16_mega_moe.cuh:785`），全 `impls/` 里共 **115 处**
  `elect_one_sync`（audit 段）。这正是自建 wrapper 去掉内部 elect 的前提。
- **`disable-output-lane` 掩码被省略**：母本恒传 `{0,0,0,0}`（等于「不屏蔽任何输出通道」），自建层
  直接用**不带掩码**的 PTX 形式——语义等价、编码更短。**block-scale 形态母本本就不带掩码**，故
  §5.3 的四份 `mxf8f6f4` 能逐字相等。
- **模板化被"外移"到主机**：M/N/major/类型在母本是模板参、在自建层由调用方传入的
  `runtime_instr_desc` 承载（其高位 32 位即 `tcgen05` 的指令描述符，低位 32 位给稀疏 E 地址）；
  自建层里 `desc >> 32` 与母本 `idescE >> 32` 是**同一约定**。

### 5.3 跨代码库逐字相等（host 证实）

`umma_paths_audit.out.txt` §4：把四份 `mxf8f6f4.block_scale` 的完整指令串（含操作数）归一化后比较：

```
cta_group::1 : DeepGEMM homegrown / CUTLASS atom / MPK kernel::sm100 / MPK mirage::...::sm100
               -> identical among the 4 copies that provide it: YES
cta_group::2 : DeepGEMM homegrown / CUTLASS atom /      MPK mirage::...::sm100
               -> identical among the 3 copies that provide it: YES   (kernel::sm100 无 2SM 版)
```

四条独立演进、分别由 DeepGEMM/CUTLASS/MPK 两个团队维护的代码路径里，这枚 asm 串**一个空格都不差**——
「这是一条被抄写的指令」而非「三套独立实现」。

---

## 6. 对 megakernel 的含义

- **对 `docs/57` 的补强**：`docs/57` headine 说「SM100 原语实现了不止一遍」，并把 MPK 内部数成「两遍
  变三遍」（`kernel::sm100` / `mirage::…::sm100` / CUTLASS）。加上 DeepGEMM 自建层，`tcgen05.mma` 的
  `mxf8f6f4.block_scale` 形态在**本专题覆盖的四个代码库里有四份逐字相同的拷贝**。这解释了为什么读
  生产 mega-kernel 的 device 身体时，「它发的是哪条指令」几乎总有多种等价写法——**指令是硬件的、
  封装是团队的**。
- **对 `docs/62`/`docs/71` 的关系**：`docs/71` 从 WS 一角切入 DeepGEMM 自建层，本文把同一层从
  **能力全集/裁剪三件套/跨库相等**三个角度补全；`docs/71` 的「同一颗 `kind::f16` 在三处各封装一遍」
  在本文扩成一张 40→11→4→1 的包含链。
- **对 `docs/02`（生产案例）的关系**：DeepGEMM 的 9 struct 正是 **Mega MoE / Mega Gate / Mega mHC**
  的 UMMA 入口（`docs/71` §3.1 的 13 文件清单）；MPK 各通路分别是**手写 MLA decode/MTP**（`sm100_ptx`）
  与 **FP8 linear 家族**（`kernel::sm100` / `mirage::…::sm100`）的入口。即「生产级段级 mega-kernel」
  与「自动化整图 megakernel」在选择张量核原语时，**都退化成「拷一份自己用到的形态」
  （+ 必要时调 CUTLASS 母本）**——这正是「megakernel 把框架职责下沉成 task 间约定」在**指令封装层**
  的同构现象。
- **判据映射**：本文只贡献 **C3（多算子范围 / 设备体内指令契约）**。它讲的是「一个 task 的 device
  body 内部发哪条 UMMA、封装来自哪份拷贝」，不涉及 launch/常驻（C1/C2）、不涉及设备内调度（C4/C5）。
  与 `docs/57`/`docs/62`/`docs/63`/`docs/71` 同层。

---

## 7. 局限（诚实标注）

- **纯文本审计，SM100 指令本体未执行**：本机（目标卡 H100/sm90）无 `tcgen05`/TMEM，无法编译或运行
  这些 wrapper；本文的 opcode/操作数矩阵是**源文件字符串的静态抽取**，不是硬件观测。语义差异
  （如「省略 `disable-output-lane` 是否与传全零完全等价」）按 PTX 规格推断，未在 SM100 上验证。
- **opcode 粒度是「字符串」**：不同的 `#if` 分支、宏展开前的文本都按字面计数；`mxf4` 的两形态、
  `mxf4nvf4` 的 `block16/block32/scale_vec::4X` 在「能力」上是近似同族，本文的 11/40 计数只作
  量级对照、不宜当精确规格。
- **CUTLASS 的 40 是「一文件内」的数**：`mma_sm100_umma.hpp` 之外（SM120/MXFP 其它头、CuTe DSL）
  还有别的 UMMA 定义，本文只取这一份母本。
- **「谁在跑」是调用者 `grep` 级**：受宏/别名/注释影响；运行期是否真走到该分支（如 `#if CUDA≥12.9`）
  未在编译期验证。
- **copy 相等性只覆盖 `mxf8f6f4.block_scale`**：`kind::f16` 因母本带掩码而**不逐字相等**（自建层去掉了
  掩码）——这不是「不一致」而是 §5.2 的裁剪，本文已区分说明；未对其余 8 个自建 struct 逐一做跨库比较。
- **未判意图**：DeepGEMM 为何保留 WS/F8F6F4 死码、MPK 为何在 `kernel::sm100` 与 `mirage::…::sm100`
  之间复制——本文只记「有定义、调用者数」这一静态事实（`docs/71` §7 同条）。

---

## 8. 参考坐标（可核对）

- DeepGEMM（`057ca59`，`~/github/DeepGEMM`）：
  - 自建层 `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh:1-219`——9 struct（§3 表）+ `tmem_load_32dp32b:190-204`
    + `umma_arrive_no_elect:206-209` + `tcgen05_before/after_thread_sync:211-217`；头注释
    *"UMMA versions with relaxed assertions"*（`:11`）；
  - 调用者：`impls/sm100_bf16_gemm.cuh:297/301`、`sm100_bmk_bnk_mn.cuh:174`、
    `sm100_bf16_mega_gate.cuh:217`、`sm100_bf16_mega_moe.cuh:785/795`、
    `sm100_fp8_fp4_gemm_1d1d.cuh:413-414`、`sm100_fp8_fp4_mega_moe.cuh:918`、
    `sm100_mqa_logits.cuh:372`、`sm100_sparse_mqa_logits.cuh:373`；13 文件 `#include` 该头（`docs/71` §3.1）。
- CUTLASS 母本（`f3fde583`，三处 checkout 同 md5）：`cute/arch/mma_sm100_umma.hpp`——36 个
  `SM100_MMA_*` 模板，`SM100_MMA_TF32_SS:46`、`SM100_MMA_F16BF16_SS:86-121`、`SM100_MMA_TF32_TS:127`、
  `SM100_MMA_F16BF16_2x1SM_SS:422`、`SM100_MMA_F8F6F4_SS:954`、`SM100_MMA_MXF8F6F4_SS:990-1026`、
  `SM100_MMA_MXF4_SS:1358`、`SM100_MMA_MXF4_2x1SM_SS:1500`；掩码/elect/guard 见 `:104-115`。
- MPK（`f9eb70c`）：
  - `include/mirage/persistent_kernel/tasks/blackwell/sm100_ptx.cuh:69-80`（`tcgen05_mma`）、
    `:82-86`（`tcgen05_commit`）、`:62-67`（`make_desc`）；
  - `blackwell/sm100_utils.cuh:127-150`（`kernel::sm100::SM100_MMA_MXF8F6F4_SS`）、
    `:116-123`（`make_runtime_instr_desc_with_sf_id`）；
  - `blackwell/linear_fp8_sm100_sm100_utils.cuh:182-205`/`:207-230`（`mirage::…::sm100` 两套）；
  - 消费者：`linear_fp8_1d2d_sm100.cuh:132/134/579`、`linear_fp8_sm100.cuh:487`、
    `fp8_group_gemm_sm100.cuh:513/1271`、`linear_sm100_mpk.cuh:62`、`moe_linear_sm100.cuh:72`；
  - MLA 私有 ptx 副本：`mla_mtp_decode_tp2_sm100.cuh:64-127`（`tcgen05_mma:109-114`）、
    `mla_mtp_decode_tp4_sm100.cuh:98-103`、`mla_mtp_decode_tp8_sm100.cuh:95-100`；消费
    `sm100_ptx` 的：`mla_decode_sm100.cuh:197/385`、`mla_mtp_decode_sm100.cuh:270/549`。
- 本文实验：`src/umma_paths_audit.cu` + `harness/umma_paths_audit.sh` +
  `harness/umma_paths_audit.out.txt`（目标卡 H100/sm90，host 侧文本审计 + grep audit，未 launch kernel）。
- 前置走读：`docs/57`（MPK SM100 原语层）、`docs/62`（TMEM fragment trait 家族）、
  `docs/63`（稀疏 A 操作数通路）、`docs/71`（WS 指令落地现状）。
- 相关坑（`ROADMAP.md`「已知坑」）：SM100 共驻 device 适配、常驻 kernel 内 TMEM alloc/dealloc 同 warp、
  `_sm100` 后缀 ≠ SM100 专属。
