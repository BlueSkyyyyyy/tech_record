# DeepGEMM 自建 PTX UMMA 封装层逐条对照：9 个 struct、两个 helper 与 MPK 手写 `tcgen05_mma`/`tcgen05_commit` 的并置

> 快照与坐标：DeepGEMM = `~/github/DeepGEMM` commit `057ca59`（`Public release 26/09/30`，容器内
> `~/github/DeepGEMM`）；MPK = `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`
> （完整 checkout `/ssd/mirage_mpk_build`）；CUTLASS = MPK 的 submodule `deps/cutlass` commit
> `f3fde583`（`docs/71` 已用 md5 证明它与 DeepGEMM vendored 副本、其 `third-party/cutlass` 逐字节相同）。
> 本文补 `docs/72`（P158-1）§7 明留、ROADMAP「下一步候选」点名的**三条「UMMA 原语/描述符对照」未完面**：
> **(1)** DeepGEMM 自建 `ptx/tcgen05.cuh` 9 struct 里**除 `mxf8f6f4.block_scale` 外**的跨库逐条对照；
> **(2)** 两个 helper（`tmem_load_32dp32b` / `umma_arrive_no_elect`）与 CUTLASS 同名/近名物的对照；
> **(3)** MPK `sm100_ptx` 的 `tcgen05_mma`/`tcgen05_commit` 与 DeepGEMM/CUTLASS 对应物的**操作数形状**并置。
> **纯源码走读 + 1 个 host 侧文本审计探针**：SM100 的 `tcgen05`/TMEM 本体在目标卡 H100/sm90 不可执行，
> 但各层只是「asm 串 + 一套操作数约定」，可在 host 上抽串、归一化、逐字比对（`src/umma_homegrown_audit.cu`
> + `harness/umma_homegrown_audit.sh` + `harness/umma_homegrown_audit.out.txt`）。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）DeepGEMM 自建层的 8 个非 WS struct，去掉「三件机械约束」后与 CUTLASS atom **逐字
一致**——它们不是新封装，而是 atom 的「裸 asm 版」。** host 探针给出精确对照表（`§2`）：
`F16BF16_SS` / `F16BF16_2x1SM_SS` / `F8F6F4_SS` / `F8F6F4_2x1SM_SS` 四条 dense 形态 **verbatim=N 但
strip_mask=Y**（homegrown 去掉了 CUTLASS 恒传的全零 disable-output-lane 掩码 `{%5,%6,%7,%8}` 后完全相等，
如 `§A` 的 f16 逐字样例）；`MXF8F6F4_SS` / `MXF8F6F4_2x1SM_SS` / `MXF4_SS` / `MXF4_2x1SM_SS` 四条
block-scaled 形态 **verbatim=Y**（本就无掩码，逐字相等，`docs/72` §4 已证 1SM 四份一致）。唯一
**无 CUTLASS 对端**的是 `F16BF16_WS_SS`（`.mma.ws`，CUTLASS `f3fde583` 全仓 0 个 `SM100_MMA_*_WS*`，
`docs/71` 已证）。「三件机械约束」的头注释原话是 `:11` *"UMMA versions with relaxed assertions"*。

**(B)（headline）两个 helper 的「减法」各有出处：`tmem_load_32dp32b<K>` 是 CUTLASS 八个 `copy` 结构
在编译期挑一个；`umma_arrive_no_elect` 只是 CUTLASS `umma_arrive` 去掉内部 `elect_one_sync`，
而 CUTLASS 自己也有一个 `..._no_elect` 版本（做稀疏用）。** host 探针（`§B`）：`tmem_load_32dp32b<K>`
把 `K∈{4,8,16,32,64,128}` 映射到 `cute::SM100_TMEM_LOAD_32dp32b{4,8,16,32,64,128}x`（前五个带
`kNumValues ==` 前缀，`128` 是 `conditional_t` 链的 else 分支），六个目标 struct 在 `copy_sm100.hpp`
都存在（`docs/71` §C 已核 `tmem_load_32dp32b:190-204`）；`umma_arrive_no_elect`（`:206-209`）的 asm 与
CUTLASS `umma_arrive`（`barrier.h:766-777`）**逐字相同**，差在 CUTLASS 版内部 `if (elect_one_sync())`
（`:769`）；CUTLASS 另有 `umma_arrive_multicast_no_elect`（`:833-847`，注释 *"Temporary solution for
sparse kernel"*）——**「去掉 elect 的 arrive」在两侧都是被显式承认的形态**。附带发现：
`tcgen05.fence::before/after_thread_sync` 这套 fence 在 **CUTLASS `f3fde583` 全仓 0 处**
（探针 `§B` 计数），DeepGEMM 自建层与 MPK 手写层**各自重造了一遍**（DeepGEMM `:211-217`、
MPK `linear_fp8_sm100_sm100_utils.cuh:174-180`）。

**(C)（headline）MPK 手写的 `sm100_ptx::tcgen05_mma`/`tcgen05_commit` 与 DeepGEMM 自建层「同一高度」、
比 CUTLASS 更低一层——它连 struct 都不包，是两个 `__device__` 裸函数。** host 探针（`§C`）：
`sm100_ptx::tcgen05_mma`（`sm100_ptx.cuh:69-80`）的 asm 与 DeepGEMM `SM100_MMA_F16BF16_SS`（`:23`）
**raw 逐字相等**、与 CUTLASS `SM100_MMA_F16BF16_SS`（`:111`）**strip_mask 后相等**；
`sm100_ptx::tcgen05_commit`（`:82-86`）与 DeepGEMM `umma_arrive_no_elect`、CUTLASS `umma_arrive` 的
asm **squash 后相等**。即「同一颗 `tcgen05.mma.cta_group::1.kind::f16`」在**四个代码库**里有一份
相同的操作数形状：CUTLASS（带 elect+掩码）→ DeepGEMM 自建（裸）→ MPK `sm100_ptx`（裸）。差别只在
**谁负责 elect**（三者的调用者都自己写 `elect_one_sync`/`elect_sync`，探针 `§6`：DeepGEMM `impls/`
里 **115 处** elect、MPK `mla_decode_sm100.cuh:170/182` 用 `wid==0&&elect_sync()`）。

---

## 1. 定位：本文补哪一层、与谁分工

| 篇 | 覆盖 |
|---|---|
| `docs/57`（P143-1） | MPK `tasks/blackwell/` 的**共享原语层**，headline「原语实现不止一遍」|
| `docs/69`（P154-1） | MPK 两份**描述符**副本（`_sm100_utils` 全量 vs `sm100_utils` 子集）+ 位域编码 |
| `docs/72`（P158-1） | `tcgen05.mma` **指令串集合**的多路并置（CUTLASS 40 opcode ⊃ DeepGEMM 自建 11 ⊃ MPK 4）|
| `docs/73`（P159-1） | SMEM **描述符的构造算术**四路并置（母本/全量/子集/手写）|
| **本文** | **封装层的逐 struct / 逐 helper 对照**：把 `docs/72` 停在「集合/opcode 级」的结果，落到「每个 struct 与对端差在哪几个 operand、哪几条约束」；并补 `docs/72` §7 明留的三条未完面 |

`docs/72` 已经证明集合包含链与「`mxf8f6f4.block_scale` 四份逐字一致」；本文**不重复**该结论，只补三件
`docs/72` 未做的事：**(1)** 9 struct **逐条**给出 CUTLASS 对端（`docs/72` 只给 opcode 计数）；
**(2)** 两个 helper（`tmem_load_32dp32b`/`umma_arrive_no_elect`）的减法溯源；**(3)** MPK 手写
`tcgen05_mma`/`commit` 与两侧的**操作数形状**并置（`docs/72` 只说「存在 4 个 opcode」）。

---

## 2. 9 个 homegrown struct 逐条对照（`harness/umma_homegrown_audit.out.txt` §A）

DeepGEMM `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`（219 行，`namespace deep_gemm::ptx`）：

| homegrown struct（行号） | opcode | CUTLASS 对端 | CUTLASS 有 elect | CUTLASS 有掩码 | verbatim | strip 后 |
|---|---|---|---|---|---|---|
| `SM100_MMA_F16BF16_SS` `:12-27` | `.mma.cta_group::1.kind::f16` | `SM100_MMA_F16BF16_SS` `:86-121` | Y | **Y** | N | **Y** |
| `SM100_MMA_F16BF16_2x1SM_SS` `:29-44` | `.mma.cta_group::2.kind::f16` | `SM100_MMA_F16BF16_2x1SM_SS` `:422-455` | Y | **Y** | N | **Y** |
| `SM100_MMA_MXF8F6F4_SS` `:46-65` | `.mma.cta_group::1.kind::mxf8f6f4.block_scale` | `SM100_MMA_MXF8F6F4_SS` `:990-1026` | Y | N | **Y** | Y |
| `SM100_MMA_MXF8F6F4_2x1SM_SS` `:67-86` | `.mma.cta_group::2.kind::mxf8f6f4.block_scale` | `SM100_MMA_MXF8F6F4_2x1SM_SS` `:1277-1311` | Y | N | **Y** | Y |
| `SM100_MMA_F8F6F4_SS` `:88-104` | `.mma.cta_group::1.kind::f8f6f4` | `SM100_MMA_F8F6F4_SS` `:954-985` | Y | **Y** | N | **Y** |
| `SM100_MMA_F8F6F4_2x1SM_SS` `:106-122` | `.mma.cta_group::2.kind::f8f6f4` | `SM100_MMA_F8F6F4_2x1SM_SS` `:1199-1233` | Y | **Y** | N | **Y** |
| `SM100_MMA_MXF4_SS` `:124-146` | `.mma.cta_group::1.kind::mxf4.block_scale.block32` | `SM100_MMA_MXF4_SS` `VS=32` 支 `:1401-1421` | Y | N | **Y** | Y |
| `SM100_MMA_MXF4_2x1SM_SS` `:148-170` | `.mma.cta_group::2.kind::mxf4.block_scale.block32` | `SM100_MMA_MXF4_2x1SM_SS` `:1500-` | Y | N | **Y** | Y |
| `SM100_MMA_F16BF16_WS_SS` `:172-187` | `.mma.ws.cta_group::1.kind::f16` | **(none)** | - | - | N | N |

**读法**：
- **dense 四条（f16 / f8f6f4，含 2x1SM）**：CUTLASS 在 asm 里恒传 `{%5,%6,%7,%8}`（1SM）/12 个掩码
  占位（2x1SM）的 `disable-output-lane`，homegrown **去掉**了它 → `verbatim=N`、`after-strip=Y`。
  operand 计数：`F16BF16_SS` homegrown **4** vs CUTLASS **9**（4 个真实 + 4 掩码 + 1 `p`），
  `F16BF16_2x1SM_SS` 4 vs 13，`F8F6F4_SS` 4 vs 8，`F8F6F4_2x1SM_SS` 4 vs 12。
- **block-scaled 四条**：本就没有掩码，homegrown 与 CUTLASS `strip_mask` **逐字相同**（operand 6=6），
  即 `docs/72` §4 的「四份逐字一致」在逐 struct 层面成立。
- **`SM100_MMA_MXF4_SS` 只对应 CUTLASS 的 `VS=32` 支**：CUTLASS 该 struct 用 `if constexpr (VS==16)`
  走 `mxf4nvf4.block_scale.block16`（`:1388`，CUDA≥12.9）/`scale_vec::4X`（`:1390`）、`VS==32` 走
  `mxf4.block_scale.block32`（`:1409`）/`scale_vec::2X`（`:1411`，旧 CUDA）；homegrown 把 `>=12.9` 的
  `block32` 与旧 `scale_vec::2X` **都写成无条件分叉**（`:137-141`）——即它**只 copy 了 VS=32 那半**，
  没有任何 `mxf4nvf4`/VS=16 形态（与 `docs/72`「MPK 无 `mxf4nvf4`」同源，这里是 DeepGEMM 自建层自己也砍了）。

---

## 3. 「三件机械约束」的精确形态

DeepGEMM 自建层相对 CUTLASS atom 去掉的三件（`docs/72` §5.2 已命名，本文给逐 struct 证据）：

1. **去内部 elect**：CUTLASS 每个 `fma` 体内 `if (cute::elect_one_sync())`（如 `:105/:769/:1012/:1382`）；
   homegrown **无**（9 struct 的 `fma` 体都不含 `elect_one_sync`）。责任上移到调用点——
   host 探针 `§6`：DeepGEMM `impls/` 里 `elect_one_sync` 调用点 **115 处**（每个调用点自己 elect）。
2. **去编译门**：CUTLASS 用 `#if defined(CUTE_ARCH_TCGEN05_*_MMA_ENABLED)` 包住 asm，
   homegrown 直接 `asm volatile`（仍保留 `CUTLASS_DEVICE` 标记）。
3. **去 disable-output-lane 掩码**：仅 dense 形态有——CUTLASS 恒传全零 `{%5,%6,%7,%8}`，
   homegrown 省掉（`§A` 的 strip 对比证明省掉后逐字相等）。

`verbatim` 只对 block-scaled 四条件为真、dense 四条因掩码为假——这正是 `docs/72` §5.2「裁剪口径」
所预言的结果，本文把它**逐 struct** 落到「差在哪 4–5 个 operand 占位」。

---

## 4. 两个 helper 的减法溯源

### 4.1 `tmem_load_32dp32b<K>` vs CUTLASS 八个 `copy` struct（探针 §B）

DeepGEMM `:190-204` 是一个模板函数，用 `cute::conditional_t` 链把 `K` 折到
`cute::SM100_TMEM_LOAD_32dp32b{4,8,16,32,64,128}x` 上、再用 `index_sequence` 展开
`Loader::copy(tmem_addr, values[Is]...)`。探针证明六个目标 struct 在 `copy_sm100.hpp` 都存在
（`SM100_TMEM_LOAD_32dp32b{1,2,4,8,16,32,64,128}x` 及各自 `_16b` 版共 16 个，本次映射用其中 6 个）。
它与 CUTLASS 的关系不是「改写」而是「**编译期选择器 + 变参展开**」：调用方写
`ptx::tmem_load_32dp32b<kNumChunkHeads>(addr, buf)`，CUTLASS 则要显式写
`SM100_TMEM_LOAD_32dp32b8x::copy(addr, b0..b7)`。

### 4.2 `umma_arrive_no_elect` vs CUTLASS `umma_arrive`

asm（两边逐字相同）：

```cpp
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];
```

DeepGEMM `:206-209` 直接 `asm volatile(...)`；CUTLASS `barrier.h:766-777` 先 `cast_smem_ptr_to_uint`
再 `if (cute::elect_one_sync())`。CUTLASS 另有 `umma_arrive_multicast_no_elect`（`:833-847`）与
`umma_arrive_multicast_2x1SM_no_elect`（`:852-866`），注释明说 *"Temporary solution for sparse kernel"*
——**「去掉 elect」的 arrive 不是 DeepGEMM 独有，CUTLASS 自己为稀疏留了口子**。DeepGEMM 把这一档
**提升为默认**（它 9 struct 都不 elect），代价是**每个调用点自负 elect 责任**。

### 4.3 附带：`tcgen05.fence::before/after_thread_sync` 在 CUTLASS 里根本没有

探针 `§B` 计数显示 `tcgen05.fence::before/after_thread_sync` 在 **CUTLASS `f3fde583` 全仓 0 处**
（`barrier.h`、`mma_sm100_umma.hpp` 都 0；CUTLASS 只提供 `tcgen05.wait::ld/st` 于
`fence_view_async_tmem_load/store` `:885-907`）。而 DeepGEMM 自建层 `:211-217` 与 MPK
`linear_fp8_sm100_sm100_utils.cuh:174-180` **各写了一对同名 fence**；MPK 的 MLA 身体甚至**直接内联**它
（`mla_decode_sm100.cuh:189` 等，不经 helper）。这是一处「CUTLASS 没提供的原语、两个使用方各自补齐」。

---

## 5. MPK 手写 `sm100_ptx` 与两侧并置（探针 §C）

MPK `tasks/blackwell/sm100_ptx.cuh`（89 行，`namespace kernel::sm100_ptx`）有两枚与 UMMA 有关的裸函数：

```cpp
// :69-80
__device__ __forceinline__ void tcgen05_mma(int taddr, uint64_t a_desc, uint64_t b_desc,
                                            uint32_t idesc, int acc) {
  asm volatile("{... setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;} ...");
}
// :82-86
__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ...);
}
```

host 探针 `§C` 的三方比对：

| 物 | asm（归一化） | 与 MPK |
|---|---|---|
| MPK `sm100_ptx::tcgen05_mma` | `tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;` | — |
| DeepGEMM `SM100_MMA_F16BF16_SS` `:23` | 同上 | **raw 逐字相等 Y** |
| CUTLASS `SM100_MMA_F16BF16_SS` `:111` | `...[%0], %1, %2, %3, {%5,%6,%7,%8}, p;` | **strip_mask 后相等 Y** |
| MPK `sm100_ptx::tcgen05_commit` | `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];` | — |
| DeepGEMM `umma_arrive_no_elect` | 同上（单字面量） | **squash 后相等 Y** |
| CUTLASS `umma_arrive` | 同上（在 `elect` 内） | **squash 后相等 Y** |

**形态差异**：CUTLASS 是**带模板参数的 struct**（`SM100_MMA_MXF8F6F4_SS<a,b,c,sf,M,N,...>`，
由 `make_tiled_mma` 选）；DeepGEMM 是**非模板的无条件 struct**（`struct SM100_MMA_F16BF16_SS`）；
MPK 是**连 struct 都不要的裸函数**（固定 `kind::f16`、无模板、无 elect、无掩码）。这一梯度正对应
`docs/73` 的「描述符构造从模板到常数」——**指令封装同样从模板退化到常数**。
MPK 之所以能退到裸函数：它只需要一种形态（MLA decode/MTP 的 f16 QK/PV），调用方（`wid==1 &&
elect_sync()`，`mla_decode_sm100.cuh:182`）自己 elect、自己管 `idesc`/`scale_c`。

MPK 里 `tcgen05_mma`/`tcgen05_commit`（共享 `sm100_ptx` 版）的消费者是 MLA decode/MTP 家族
（探针 `§6`：`mla_decode_sm100.cuh:197/200/202/385/389/391`、`mla_mtp_decode_sm100.cuh:270/273/275/
549/553/555`）；而 TP2/4/8 三兄弟**各在自己私有 `namespace ptx` 重抄一份**（`mla_mtp_decode_tp2_
sm100.cuh:109/122`、`tp4:98/111`、`tp8:95/108`，`docs/67` §8 已记）。

---

## 6. caller audit（探针 §A 末段 / §6）

- 9 struct **调用者计数**（`ptx::NAME`，excl. def）：`F16BF16_SS` 3、`F16BF16_2x1SM_SS` 3、
  `MXF8F6F4_SS` 3、`MXF8F6F4_2x1SM_SS` 2、`MXF4_SS` 3、`MXF4_2x1SM_SS` 1；
  **`F8F6F4_SS` 0、`F8F6F4_2x1SM_SS` 0、`F16BF16_WS_SS` 0**——即 `docs/71` 报的「9 struct 中 3 个零调用者」
  在逐 struct 上坐实。**全部 9 struct 都活在一个 219 行头文件里，但只有 6 个在生产 impl 里被点名。**
- 具体调用点：`sm100_bf16_gemm.cuh:297`、`sm100_bmk_bnk_mn.cuh:174`、`sm100_bf16_mega_gate.cuh:217`、
  `sm100_bf16_mega_moe.cuh:795`、`sm100_mqa_logits.cuh:372`、`sm100_sparse_mqa_logits.cuh:373`、
  `sm100_fp8_fp4_gemm_1d1d.cuh:413-414`、`sm100_fp8_fp4_mega_moe.cuh:918`——即 DeepGEMM 的
  **SM100 GEMM 与整个 mega 家族共用这一份自建 UMMA 层**（`ptx/tcgen05.cuh` 被 **13 个**文件 include）。
- 两个 helper 的调用点：`tmem_load_32dp32b` 在 `sm100_mqa_logits.cuh:44`、
  `sm100_sparse_mqa_logits.cuh:609`、`epilogue/sm100_mega_gate.cuh:64`；
  `umma_arrive_no_elect` 在 `sm100_mqa_logits.cuh:421/424`、`sm100_sparse_mqa_logits.cuh:409/411`。

---

## 7. 对 megakernel 的含义

- **对 `docs/72` 的补强**：`docs/72` 把「能力包含链」（40→11→4 opcode）讲清；本文给出**逐 struct 的
  差异落点**——dense 差在**一个 disable-output-lane 掩码**、其余只剩 **elect 责任归属**。即
  DeepGEMM 自建层不是「另一套指令」，而是「**atom 的裸 asm 版 + 调用点自负 elect**」；
  MPK 手写层再退一步成「**裸函数、固定一种形态**」。四层同源、逐级裁剪（CUTLASS 模板 → DeepGEMM
  无条件 struct → MPK 裸函数）。
- **对 `docs/73` 的呼应**：指令层与描述符层走**同一条退化曲线**——`docs/73` 是「模板 `make_umma_desc`
  → MPK 手写常数 `make_desc`」，本文是「模板 atom → MPK 裸 `tcgen05_mma`」。两者合起来说明：当一个
  常驻 megakernel 的某个 task 只需**一种固定形态**的 UMMA 时，把它折成常数/裸函数比走模板**更轻**，
  代价是**失去通用性、并把 elect/掩码/描述符的责任交给调用点**（`docs/64` 的 `kNumSMs=1`、`docs/67`
  的 flat-grid 同属「薄 task 的固定形态适配」）。
- **对「少而重 vs 多而杂」的呼应**：DeepGEMM **极重**的段级 mega-kernel（Mega MoE/Gate）走的是
  通用 `cute::make_tiled_mma`/`partition_shape`，只在**最内层**用这份「裸 ptx UMMA」；MPK **薄**的
  attention task 则直接调裸函数。**「重 task 用高层描述符、薄 task 用裸指令」是同一套「固定形态即降封装」的选择**
  ——与 `docs/73` §7 的结论一致。
- **判据映射**：本文只贡献 **C3（多算子范围 / 设备体内指令与操作数契约）**。它讲的是「一个 task 的
  device body 发 UMMA 时用哪份封装、那份封装与 CUTLASS 差在哪几个 operand」，不涉及 launch/常驻
  （C1/C2）、不涉及设备内调度（C4/C5）。与 `docs/57`/`docs/72`/`docs/73` 同层。

---

## 8. 局限（诚实标注）

- **纯 host 侧文本审计，SM100 本体未执行**：本文的「逐字相等」是 asm 串的**文本归一化后**相等，不是
  在 SM100 上观测到的硬件行为；`.mma.ws`/`mxf4nvf4` 等无法在目标卡 H100/sm90 上验证。
- **`strip_mask` 是启发式**：它按 `, {%...}` 正则去掉 CUTLASS 的 disable-output-lane 掩码；若其它库
  用不同书写（如空格/换行不同）可能误判。本文只对 `f3fde583`/`057ca59` 两个固定快照操作。
- **operand 计数用 `%N` 占位符数近似**：对 `{%5,%6,%7,%8}` 这类掩码会各算一次，读表时须知道
  「homegrown 4 vs CUTLASS 9」的差是掩码而非真实操作数。
- **`verbatim=Y` 不等于语义完备**：asm 相同不排除 struct 外围（`static_assert`、模板参数、`using`）
  的差异；本文只对齐 asm 与 fma 签名。
- **「谁是母本」按 `docs/72` 的判断**（DeepGEMM 早于 MPK 该 commit 公开、MPK 自陈 adapted）；本文不
  重开该论证，只复核逐 struct 关系。
- **caller audit 是 `grep` 级**：计数受命名空间/`using` 影响，只记静态事实；运行期是否真走到某
  struct 未在编译期验证（3 个零调用者也是静态判断）。
- **MPK 的 `sm100_ptx` 与 DeepGEMM 自建层无直接抄袭关系可证**：两者都近似 CUTLASS 裸 asm，
  「相同」更多是「同一硬件契约 + 同样的裁剪」的收敛，不是代码派生（本文不主张派生）。

---

## 9. 参考坐标（可核对）

- 快照：DeepGEMM @ `057ca59`（`~/github/DeepGEMM`）；MPK @ `f9eb70c`（本地 `/ssd/mirage_mpk_build`）；
  CUTLASS `deps/cutlass` @ `f3fde583`（`docs/71` md5 已证与 DeepGEMM 副本逐字节相同）。
- **DeepGEMM 自建层**（`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`，219 行，13 文件 include）：
  头注释 `:11`；9 struct——`F16BF16_SS:12-27`、`F16BF16_2x1SM_SS:29-44`、`MXF8F6F4_SS:46-65`、
  `MXF8F6F4_2x1SM_SS:67-86`、`F8F6F4_SS:88-104`、`F8F6F4_2x1SM_SS:106-122`、`MXF4_SS:124-146`
  （`>=12.9` 分叉 `:137-141`）、`MXF4_2x1SM_SS:148-170`、`F16BF16_WS_SS:172-187`；
  helpers——`tmem_load_32dp32b:190-204`、`umma_arrive_no_elect:206-209`、
  `tcgen05_before/after_thread_sync:211-217`。
- **CUTLASS atom**（`deps/cutlass/include/cute/arch/mma_sm100_umma.hpp`）：`F16BF16_SS:86-121`
  （elect `:105`、mask `:106`、asm `:111`）、`F16BF16_2x1SM_SS:422-455`（asm `:446`）、
  `F8F6F4_SS:954-985`（asm `:975`）、`MXF8F6F4_SS:990-1026`（elect `:1012`、asm `:1017`）、
  `F8F6F4_2x1SM_SS:1199-1233`、`MXF8F6F4_2x1SM_SS:1277-1311`（asm `:1302`）、
  `MXF4_SS:1358-1422`（VS=16 asm `:1388/:1390`、VS=32 asm `:1409/:1411`）、`MXF4_2x1SM_SS:1500-`。
  另 `cutlass/arch/barrier.h`：`umma_arrive:766-777`（elect `:769`）、`umma_arrive_2x1SM:780-792`、
  `umma_arrive_multicast:795-810`、`umma_arrive_multicast_no_elect:833-847`、`fence_view_async_tmem_load/
  store:885-907`。`cute/arch/copy_sm100.hpp`：`SM100_TMEM_LOAD_32dp32b{1,2,4,8,16,32,64,128}x` + `_16b`。
- **MPK 手写层**（`tasks/blackwell/sm100_ptx.cuh`，89 行）：`make_desc:63-67`、`tcgen05_mma:69-80`、
  `tcgen05_commit:82-86`、`elect_sync:27-34`、`mbar_init/wait/tx:36-57`；消费者
  `mla_decode_sm100.cuh:170/182/197/200/202/385/389/391`、`mla_mtp_decode_sm100.cuh:270/273/275/549/553/555`；
  私有副本 `mla_mtp_decode_tp2_sm100.cuh:109/122`、`tp4:98/111`、`tp8:95/108`；fence 副本
  `linear_fp8_sm100_sm100_utils.cuh:174-180`。
- 本文实验：`src/umma_homegrown_audit.cu` + `harness/umma_homegrown_audit.sh` +
  `harness/umma_homegrown_audit.out.txt`（目标卡 H100/sm90，host 侧文本审计 + grep audit，未 launch
  kernel；SM100 本体不可执行）。
- 前置/相关走读：`docs/57`（MPK SM100 原语层）、`docs/67`（MLA-MTP TP 三兄弟的私有 `ptx` 副本）、
  `docs/69`（描述符位编码）、`docs/71`（WS 指令落地现状）、`docs/72`（UMMA 指令多实现并置）、
  `docs/73`（描述符构造多路并置）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及封装层**
  （device task 内部实现质量，同 `docs/28`–`docs/73`）。
- 交叉专题：DeepGEMM MoE 前向/反向正确性验证见 `../mega-moe/`（本专题不重复）。
