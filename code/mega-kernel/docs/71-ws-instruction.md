# WS（weight-stationary）UMMA 指令落地现状对照：CUTLASS `f3fde583` 的 WS 布局 + DeepGEMM 自建 PTX 封装里的死 `tcgen05.mma.ws`

> 快照与坐标：MPK = `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`；CUTLASS = MPK 该快照
> 钉死的 submodule `deps/cutlass` commit `f3fde583`（有完整 submodule 的 checkout 在
> `/ssd/mirage_mpk_build/deps/cutlass`）；DeepGEMM = `~/github/DeepGEMM` commit `057ca59`
> （`Public release 26/09/30`），其 vendored CUTLASS 在 `deep_gemm/include/cute`、submodule 在
> `third-party/cutlass`。本文是 `docs/62`（P148-1）与 ROADMAP「下一步候选」点名的
> **「WS 指令落地后的对照（CUTLASS `f3fde583` 无 `SM100_MMA_*_WS*`，可核 DeepGEMM 侧 WS 用法）」**。
> **纯源码走读 + 1 个 host 侧结构探针 + grep audit**：SM100 指令/TMEM 本体在本机（目标卡 H100/sm90）
> 不可执行；`UMMA::tmem_*_frg::make` 与 `UMMA::InstrDescriptor` 都是 `CUTE_HOST_DEVICE`，可在 **host**
> 上用快照钉死的真实 CUTLASS 头求值（`src/ws_instruction_probe.cu` + `harness/ws_instruction_probe.sh`
> + `harness/ws_instruction_probe.out.txt`）。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）三份 CUTLASS 是同一个 commit（`f3fde583`），逐字节相同——故 `docs/62` 的结论对 DeepGEMM 的 CUTLASS 同样成立。**
DeepGEMM 的 vendored 副本（`deep_gemm/include`）、其 submodule（`third-party/cutlass`）、以及 MPK 的
`deps/cutlass` 三处，对 `cute/atom/mma_traits_sm100.hpp`、`cute/arch/mma_sm100_desc.hpp`、
`cute/arch/mma_sm100_umma.hpp` 的 **md5 完全相同**（`0f6fc14f…` / `6cf440a0…` / `a05326ed…`）。
于是 `docs/62` 的三条静态事实——WS 的 TMEM fragment 布局 `tmem_frg_ws`/`tmem_e_frg_ws` **齐全**、
指令描述符只有 `max_shift_ : 2` 位为 WS 预留、**全仓 0 个 `SM100_MMA_*_WS*` atom**——在
DeepGEMM 这边**逐条一致**（WS atom 计数仍为 0）。

**(B)（headline）但 DeepGEMM 在 CUTLASS 之外自建了一层 PTX UMMA 封装，里面「有」WS 指令——只是无人调用。**
`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`（9 个 struct）被 **SM100 mega 家族共享 include**
（mega_gate / bf16_mega_moe / fp8_fp4_mega_moe / mega_mhc / …，共 13 个 impls/epilogue 文件）。
其中 `SM100_MMA_F16BF16_WS_SS:172-187` 是这层里唯一的 WS 项，它相对 `SM100_MMA_F16BF16_SS` **只差
asm 里一个 `.ws` token**（`tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p`），函数签名
（4 个操作数、无 `tmem_sfa/sfb`、无 mask）逐字相同。caller audit：这 9 个 struct 中 **3 个零调用者**
（`SM100_MMA_F8F6F4_SS`、`SM100_MMA_F8F6F4_2x1SM_SS`、`SM100_MMA_F16BF16_WS_SS`），且**全 DeepGEMM
树 `.mma.ws` 只此 1 处**（就是这条定义）。这枚 WS 封装是在 `7f2a703`（`Public release 26/04 —
Introducing Mega MoE…`）随 Mega MoE 一起加的，从加入起就没被任何 kernel 引用过。

**(C)（headline）WS 的「布局备好、指令未落地」在 CUTLASS 与 DeepGEMM 两侧同时成立，三条证据闭环。**
(a) 全 DeepGEMM 自建代码（`deep_gemm/include/deep_gemm` + `csrc`）对 `MaxShift`/`max_shift` **0 引用**，
MPK 自建代码（`include`+`src`+`python`）同样 **0 引用**——`max_shift_` 只存在于 vendored CUTLASS 的
枚举/位域定义里；(b) `FrgTypeC/D = UMMA::tmem_frg_ws`、`FrgTypeE = UMMA::tmem_e_frg_ws` 在两个 CUTLASS
副本里**均 0**（只有 trait 定义 + `MakeTensor` 特化，无消费者）；(c) `struct SM100_MMA.*_WS` 在两个
CUTLASS 副本里**均 0**。即：**WS 的 TMEM 布局（CUTLASS 侧）和 WS 指令 wrapper（DeepGEMM 自建侧）都
写好了，但这一代（CUTLASS `f3fde583` / DeepGEMM `057ca59`）没有任何 kernel 用它**。

---

## 1. 候选由来：`docs/62` 停在哪

`docs/62`（P148-1）走 `mma_traits_sm100.hpp` 的 TMEM fragment 家族时，把 weight-stationary 的两面
（`tmem_frg_ws`、`tmem_e_frg_ws`）逐行打开并得到：

- WS 的 C/D 布局是 **1x4 / 2x2 / 4x1** 数据通路（M=32/64/128），WS 稀疏 E 比 `tmem_e_frg` 多一个 M
  子块维；
- consumer audit：`FrgTypeC/D = tmem_frg_ws`、`FrgTypeE = tmem_e_frg_ws` **均 0**；
- **全仓 0 个 `SM100_MMA_*_WS*` 原子**，只有指令描述符的 `max_shift_ : 2` 位（注释
  「Maximum shift for WS instruction」）为 WS 预留了比特。

结论是「**WS 的 TMEM 布局先写好了，WS 指令还没来**」。本文要做的是把它从「CUTLASS 这一份快照的静态
事实」推广成「**跨 checkout 可复现、并在生产实现（DeepGEMM）里也能核**」——即候选里的「DeepGEMM 侧
WS 用法」。

---

## 2. 三方 CUTLASS 同源验证（headline A）

`harness/ws_instruction_probe.out.txt` 的 audit 段打印三处 md5，逐字节相同：

| 文件 | DeepGEMM vendored | DeepGEMM `third-party/cutlass` | MPK `deps/cutlass` |
|---|---|---|---|
| `cute/atom/mma_traits_sm100.hpp` | `0f6fc14f4b294a4999236d9f0bfe7074` | 同 | 同 |
| `cute/arch/mma_sm100_desc.hpp` | `6cf440a0e0e5e2479fae970dafa55085` | 同 | 同 |
| `cute/arch/mma_sm100_umma.hpp` | `a05326ed24ecaf87068cae15a381e39e` | 同 | 同 |

含义：DeepGEMM 把 CUTLASS 以「vendored 副本」形式放进 `deep_gemm/include/cute`（并被 `deep_gemm/include/cutlass`
的 `version.h` 标为 **4.2.1**），同时 `third-party/cutlass` submodule 也钉在同一 commit；MPK 的
`deps/cutlass` 是第三份同 commit 的 checkout。**这三份是同一个 CUTLASS**。于是下面两处 WS 静态事实
（trait 定义行号、atom 计数）在两个 CUTLASS 副本里必然一致——本文用 `src/ws_instruction_probe.cu`
的 host 求值 **在 DeepGEMM 副本上复算了一遍** `docs/62` 的 WS 布局，得同样形状（§5）。

---

## 3. DeepGEMM 自建 PTX UMMA 封装层（headline B）

### 3.1 这层是什么

`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh` 定义 `namespace deep_gemm::ptx` 下 **9 个 `SM100_MMA_*`
struct**，每个只有 `static void fma(...)`，内部是一段 `asm volatile` 直接发 `tcgen05.mma...`。文件头注释
自称 *"UMMA versions with relaxed assertions"*——即它是 CUTLASS `cute::SM100_MMA_*` 之外的**自建薄封装**
（去掉 CUTLASS 的静态断言、按 DeepGEMM 自己的签名固定操作数）。它被 13 个生产文件 `#include`，覆盖
**整个 SM100 mega 家族**：

```
sm100_bf16_mega_gate.cuh    sm100_bf16_mega_moe.cuh   sm100_fp8_fp4_mega_moe.cuh
sm100_mega_mhc.cuh          sm100_bf16_gemm.cuh       sm100_fp8_fp4_gemm_1d1d.cuh
sm100_mqa_logits.cuh        sm100_sparse_mqa_logits.cuh  sm100_tf32_hc_prenorm_gemm.cuh
sm100_bmk_bnk_mn.cuh        epilogue/sm100_store_cd.cuh  epilogue/sm100_store_cd_swap_ab.cuh
                            epilogue/sm100_mega_gate.cuh
```

注意 DeepGEMM **不是**完全弃用 CUTLASS atom：`sm100_mega_mhc.cuh:235` 与
`sm100_tf32_hc_prenorm_gemm.cuh:169` 走的是 CUTLASS 的 `cute::SM100_MMA_TF32_TS<...>`（TF32 TS 形态），
而 F16/BF16、MXF8F6F4、MXF4 等形态走自建 `ptx::` 封装。**WS 封装只存在于自建那一支**。

### 3.2 9 个 struct 与调用者数（`harness/ws_instruction_probe.out.txt`）

| struct | 行 | 发出的 `tcgen05.mma` | 调用者数（除定义） |
|---|---|---|---|
| `SM100_MMA_F16BF16_SS` | `:12-27` | `.cta_group::1.kind::f16` | 3 |
| `SM100_MMA_F16BF16_2x1SM_SS` | `:29-44` | `.cta_group::2.kind::f16` | 3 |
| `SM100_MMA_MXF8F6F4_SS` | `:46-65` | `.kind::mxf8f6f4.block_scale` | 3 |
| `SM100_MMA_MXF8F6F4_2x1SM_SS` | `:67-86` | `.cta_group::2.kind::mxf8f6f4.block_scale` | 2 |
| `SM100_MMA_F8F6F4_SS` | `:88-104` | `.cta_group::1.kind::f8f6f4` | **0** |
| `SM100_MMA_F8F6F4_2x1SM_SS` | `:106-122` | `.cta_group::2.kind::f8f6f4` | **0** |
| `SM100_MMA_MXF4_SS` | `:124-146` | `.kind::mxf4.block_scale.block32` | 3 |
| `SM100_MMA_MXF4_2x1SM_SS` | `:148-170` | `.cta_group::2.kind::mxf4.block_scale.block32` | 1 |
| `SM100_MMA_F16BF16_WS_SS` | `:172-187` | **`.cta_group::1.kind::f16`（带 `.ws`）** | **0** |

调用者例（`grep` 出）：`sm100_bf16_gemm.cuh:297`（`conditional_t<kNumMulticast==1, F16BF16_SS, F16BF16_2x1SM_SS>`）、
`sm100_bf16_mega_gate.cuh:217`（同型二选一）、`sm100_bf16_mega_moe.cuh:795`（`F16BF16_2x1SM_SS::fma`）、
`sm100_fp8_fp4_mega_moe.cuh:918`（`MXF8F6F4_2x1SM_SS::fma`）、`sm100_mqa_logits.cuh:372` /
`sm100_sparse_mqa_logits.cuh:373`（`conditional_t<kIsFP4, MXF4_SS, MXF8F6F4_SS>`）。

### 3.3 WS 封装与 F16BF16 封装的唯一差别

```cpp
// SM100_MMA_F16BF16_SS  (deep_gemm/ptx/tcgen05.cuh:12-27)
"tcgen05.mma.cta_group::1.kind::f16  [%0], %1, %2, %3, p;"
// SM100_MMA_F16BF16_WS_SS (deep_gemm/ptx/tcgen05.cuh:172-187)
"tcgen05.mma.ws.cta_group::1.kind::f16  [%0], %1, %2, %3, p;"
```

函数签名**完全相同**（`fma(desc_a, desc_b, tmem_c, scale_c, desc)`），操作数列表也相同（`[%0]`=TMEM C、
`%1`=`desc_a`、`%2`=`desc_b`、`%3`=指令描述符低 32 位、`p`=`scale_c` 谓词）。差别只有 **`.ws`** 一个
token——这与 §5 的语义一致：WS 是同一颗 F16 UMMA 的 **weight-stationary 变体**（数据通路按 WS 重排），
操作数形状不变。**这枚 wrapper 是「照抄 `SM100_MMA_F16BF16_SS` 加 `.ws`」写出来的**，此后无人接线。

---

## 4. WS 的三条「未落地」证据（headline C）

`harness/ws_instruction_probe.out.txt` 的 audit 段（DeepGEMM `057ca59`；MPK `f9eb70c`）：

| 证据 | DeepGEMM | MPK |
|---|---|---|
| `.mma.ws` 出现处 | **1**（`ptx/tcgen05.cuh:183`，即定义本身） | **0** |
| 自建代码里 `MaxShift`/`max_shift` | **0** | **0** |
| `struct SM100_MMA.*_WS`（CUTLASS 副本） | **0** | **0** |
| `FrgTypeC/D = UMMA::tmem_frg_ws` | **0** | **0** |
| `FrgTypeE = UMMA::tmem_e_frg_ws` | **0** | **0** |

即：
- **指令侧**：WS 指令**只**出现在 DeepGEMM 一枚从未被调用的自建 wrapper 里；`max_shift_` 描述符位
  （WS 唯一的描述符痕迹）在任何生产代码里都没被赋值（`NoShift` 恒为默认）；
- **布局侧**：WS 的 TMEM fragment（C/D、稀疏 E）在 CUTLASS 里**定义完整、`MakeTensor` 定制点齐全、
  零 FrgType 消费者**；
- **两端都没有 `SM100_MMA_*_WS*` atom**——CUTLASS 提供的 WS「类型契约」与 DeepGEMM 提供的 WS
  「指令 wrapper」是**两处互相不知道对方存在的死代码**。

---

## 5. WS 指令的语义与 layout 对照（host 侧复算）

`src/ws_instruction_probe.cu` 用 **DeepGEMM 的 vendored CUTLASS**（与 MPK 同 commit）在 host 上求值，
结果与 `docs/62` 在 MPK CUTLASS 上所得**逐项一致**（跨 checkout 复现）：

### 5.1 `max_shift_ : 2` 是唯一的描述符 WS 痕迹（`:433`，bit [30,32)）

```
NoShift (0)    -> desc=0x00000000  bits[30,32)=0
MaxShift8 (1)  -> desc=0x40000000  bits[30,32)=1
MaxShift16 (2) -> desc=0x80000000  bits[30,32)=2
MaxShift32 (3) -> desc=0xc0000000  bits[30,32)=3
default InstrDescriptor = 0x00000000 (max_shift=0)
```

`UMMA::MaxShift` 枚举at deep_gemm vendored `cute/arch/mma_sm100_desc.hpp:251-255`
（`NoShift/MaxShift8/MaxShift16/MaxShift32`），`make_instr_desc`（`:475`）/`make_runtime_instr_desc`（`:509`）
都把它当一个模板参数、默认 `NoShift`——**除描述符 2 bit 外，WS 在 CUTLASS 的指令描述符里没有别的字段**。

### 5.2 WS C/D `tmem_frg_ws_1sm`（1x4 / 2x2 / 4x1）

```
C/D ws M=128 (4x1): ((_128,_128)):((_65536,_1))                   // N 不分段
C/D ws M=64  (2x2): ((_64,(_64,_2))):((_65536,(_1,_4194304)))     // N 后半段 stride 4194304 = 64*65536
C/D ws M=32  (1x4): ((_32,(_32,_4))):((_65536,(_1,_2097152)))     // N 四分各 stride 2097152 = 32*65536
```

即 WS 把 M 折成 4x1/2x2/1x4 数据通路后，**N 的后半段整块落到相差 64（M=64）或 32（M=32）个数据通路
的子分区上**——与 `docs/60` 的 dense `tmem_frg`「半子分区交织」同套 DP 复用思想，只是按 WS 的数据通路
形态重排（逐值与 `docs/62` 实验 C 一致）。

### 5.3 WS 稀疏 E `tmem_e_frg_ws`

```
E_ws fp16 M=128: (((_8,_2,_8),(_16,_2)),_1,_1)   // 与 tmem_e_frg M=128 同形
E_ws fp16 M=64 : (((_8,_2,_4),(_16,_2),_2))       // M 多一个 (…,4,2) 子块维，末维 stride 64*DP_b
E_ws fp16 M=32 : (((_8,_2,_2),(_16,_2),_4))       // 末维 stride 32*DP_b
```

即 WS 稀疏 E 相对 `tmem_e_frg` **只在 M<128 时多一个 M 子块维**（与 `docs/62` §4.1 同）。

---

## 6. 与 mega 家族 / 专题判据的关系

- **对 `docs/62` 的关系**：本文是它的**跨 checkout 复现 + 生产侧补充**——`docs/62` 只看了 MPK 的
  CUTLASS；本文证明 DeepGEMM 的 CUTLASS 与之逐字节同源，并把 WS 指令在**生产实现侧**的落地现状
  （自建 wrapper 里的死 `.ws`）钉死。
- **对 `docs/57`/`docs/63`（SM100 原语「实现不止一遍」）的关系**：DeepGEMM 的 `ptx/tcgen05.cuh` 是
  「SM100 UMMA 原语多实现」的又一份——它既不是 MPK 的 `tasks/blackwell/sm100_ptx.cuh`，也不是 CUTLASS
  的 `cute::SM100_MMA_*`，而是 DeepGEMM 自己的第三态（9 个 struct，被 mega 家族共享）。**同一颗
  `tcgen05.mma.kind::f16` 在这三处各封装一遍**，WS 只在 DeepGEMM 这第三态里出现过一次。
- **对 mega 家族的关系**：这层 wrapper 正是 `docs/02` 里 DeepGEMM **Mega MoE / Mega Gate / Mega mHC**
  的 UMMA 入口——即「生产级段级 mega-kernel 用哪套张量核原语」。WS 指令在这条生产路径上是**零引用**：
  「weight-stationary」在 DeepGEMM mega 家族里**目前不是一条真实数据通路**，而是一个预留位。
- **判据映射**：本文只贡献 **C3（多算子范围 / 设备体内布局与指令契约）**——它讲的是一个 task 的
  device body 内部「用哪条张量核指令、描述符哪几位」。不涉及 launch/常驻（C1/C2）、不涉及去中心化
  调度（C5）。与 `docs/60`/`docs/62`/`docs/63` 同层。

---

## 7. 局限（诚实标注）

- **SM100 指令/TMEM 本体未执行**：WS 的 `tcgen05.mma.ws` 未在 SM100 上跑（本机是目标卡 H100/sm90，
  无 SM100）；本文的 layout 是**在 host 上对真实 CUTLASS 头求值**得到的「trait 规定的形状」，`max_shift_`
  是**描述符位域的 host 编码**，均**不是**硬件观测。WS 语义（`1x4/2x2/4x1` 到底怎么被硬件消费、
  `max_shift` 三种编码的物理含义）**未验证**。
- **「WS 未落地」只覆盖本快照/本 commit**：结论对 CUTLASS `f3fde583` + DeepGEMM `057ca59` 成立；
  更新的 CUTLASS/DeepGEMM 可能已加 WS atom/用法（本专题钉死快照，不做时间外推）。
- **caller audit 是 `grep` 级**：`SM100_MMA_*` 的引用可能受宏/别名/注释影响；本文用 `\b…\b` 精确匹配
  并排除定义文件，但 `deep_gemm/include` 之外（如 `build/` 产物、生成的 `.cu`）未纳入。
- **一处行号落差**：`docs/62` 记 `tmem_e_frg_ws:671-811` / `tmem_frg_ws:876-959`，本文在 **同 md5** 的
  vendored 副本上核对为 `:672` 起 / `:877` 起（起始行差 1，属 `docs/62` 抄写偏移，静态事实本身一致）。
- **DeepGEMM 的 `.mma.ws` 只有一枚**：本文无法断定它「打算用在哪、为何留着」——只记「有定义、零调用者」
  这一静态事实，不作意图判断。

---

## 8. 参考坐标（可核对）

- DeepGEMM（`057ca59`，`~/github/DeepGEMM`）：
  - `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh:1-219`：`namespace deep_gemm::ptx` 的 9 个
    `SM100_MMA_*` struct——`F16BF16_SS:12-27`、`F16BF16_2x1SM_SS:29-44`、`MXF8F6F4_SS:46-65`、
    `MXF8F6F4_2x1SM_SS:67-86`、`F8F6F4_SS:88-104`、`F8F6F4_2x1SM_SS:106-122`、`MXF4_SS:124-146`、
    `MXF4_2x1SM_SS:148-170`、**`F16BF16_WS_SS:172-187`**（`.mma.ws` 在 `:183`）；文件头注释
    *"UMMA versions with relaxed assertions"*（`:11`）；
  - 13 个 `#include <deep_gemm/ptx/tcgen05.cuh>` 的 impls/epilogue（§3.1 表）；
  - `deep_gemm/include/cutlass/version.h:38-40`：`CUTLASS_MAJOR 4 / MINOR 2 / PATCH 1`；
  - `deep_gemm/include/cute/atom/mma_traits_sm100.hpp:672`（`tmem_e_frg_ws`）、`:875`（WS C/D 注释）、
    `:877`（`tmem_frg_ws`）、`:963`（`tmem_frg_ws_1sm`）、`:979`/`:1001`（`MakeTensor` 特化）；
  - `deep_gemm/include/cute/arch/mma_sm100_desc.hpp:251-255`（`MaxShift` 枚举）、`:433`（`max_shift_ : 2`）、
    `:475`/`:509`（`make_instr_desc`/`make_runtime_instr_desc` 的 `max_shift` 模板参）；
  - `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:297`、`sm100_bf16_mega_gate.cuh:217`、
    `sm100_bf16_mega_moe.cuh:795`、`sm100_fp8_fp4_mega_moe.cuh:918`、`sm100_mqa_logits.cuh:372`、
    `sm100_sparse_mqa_logits.cuh:373`（`ptx::SM100_MMA_*` 调用者）；
    `sm100_mega_mhc.cuh:235`、`sm100_tf32_hc_prenorm_gemm.cuh:169`（走 CUTLASS `cute::SM100_MMA_TF32_TS`）；
  - WS 封装加入 commit：`7f2a703`（`Public release 26/04`，Introducing Mega MoE）；`ptx/tcgen05.cuh`
    最后改动 `66081d4`（`Public release 26/09`）。
- CUTLASS（`f3fde583`，三处 checkout 同 md5，见 §2）：`cute/atom/mma_traits_sm100.hpp`、
  `cute/arch/mma_sm100_desc.hpp`、`cute/arch/mma_sm100_umma.hpp`（`SM100_MMA_TF32_TS:127` 起）。
- MPK（`f9eb70c`）：自建代码对 `mma.ws`/`MaxShift`/`max_shift`/`SM100_MMA_*_WS` 均 **0 引用**
  （`include`+`src`+`python`）；SM100 UMMA 走 CUTLASS atom（`linear_sm100_mpk.cuh` /
  `moe_linear_sm100.cuh` / `fp8_group_gemm_sm100.cuh` 用 `SM100_MMA_F16BF16_SS` 等）。
- 本文实验：`src/ws_instruction_probe.cu` + `harness/ws_instruction_probe.sh` +
  `harness/ws_instruction_probe.out.txt`（目标卡 H100/sm90a，host 侧结构探测 + grep audit，未 launch kernel）。
