# SM100 的 UMMA 指令底座 `ptx/tcgen05.cuh`：13 个 SM100 kernel 脚下的最后一块裸指令层

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P100 增量（第 110 篇）。
> 第 49 篇（`docs/49`）把**所有 SM90 kernel 每一行 MMA 都在调**的底座拆清了：64 位
> `GmmaDescriptor`、三个 WGMMA 选择器、`ptx/wgmma.cuh` 的异步组。第 35 篇（`docs/35`）拆了 SM100
> 唯一的稠密 GEMM 后端，顺带过了 `mma/sm100.cuh` 的**共享内存描述符**构造。但夹在它们中间、
> **13 个 SM100 设备文件都 `#include` 的那份 `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`（219 行）**
> 从没被单独读过——它才是 SM100 上所有 `tcgen05.*` 裸指令（`mma` / `commit` / `fence` / `ld`）
> 的唯一封装点，是第 49 篇的 SM100 对偶。
>
> 本篇把这份文件逐段拆透，并用它**最简单的消费者**——第 17 篇（`docs/17` §6）只当「对照」数了三行
> 的 SM100 `bmk,bnk->mn` 批归约 GEMM（`sm100_bmk_bnk_mn.cuh`）——把抽象指令落回一次真实 GEMM。
> **tcgen05 指令本身是 SM100 专属，仅分析**（第 3 节给出 ptxas 在 `sm_90a` 下逐条拒绝的硬证据）；
> 但它依赖的**内存几何**（STSM swizzle 整数公式、TMA reduce-add）在 H100(sm90a) 上真机可跑，
> 第 6 节给真实数字。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`（v2.1.1.post3 / Public release 26/09/30 #462）。
> 本机 H100 80GB HBM3 / sm90 / 132 SM，CUDA 13.2，PyTorch 2.12。

## 1. 为什么是这份文件：一段「没有它谁都动不了 tensor core」的地板

SM100（Blackwell）把 Hopper 的 WGMMA 换成 **UMMA / tcgen05**：tensor core 不再读寄存器累加器，
而是由**单个 warp 的单个 lane** 发一条 `tcgen05.mma`，累加器住在 **TMEM（Tensor Memory）** 里，
并以 `tcgen05.commit` 把完成事件推进 mbarrier。这套新契约把「发 MMA、等 MMA、读累加器」三件事
全部换成了新指令，且在 CUTLASS 里散落在 `cute/arch/copy_sm100.hpp` / `cute/arch/mma_sm100_umma.hpp`。

DeepGEMM 没有直接照抄 CUTLASS 的封装，而是**自己写了一份最小封装** `ptx/tcgen05.cuh`：

```
$ grep -rln "#include <deep_gemm/ptx/tcgen05.cuh>" deep_gemm/include/deep_gemm/{impls,epilogue}
impls/sm100_bf16_gemm.cuh              impls/sm100_fp8_fp4_gemm_1d1d.cuh
impls/sm100_bf16_mega_gate.cuh         impls/sm100_fp8_fp4_mega_moe.cuh
impls/sm100_bf16_mega_moe.cuh          impls/sm100_mega_mhc.cuh
impls/sm100_bmk_bnk_mn.cuh             impls/sm100_mqa_logits.cuh
impls/sm100_sparse_mqa_logits.cuh      impls/sm100_tf32_hc_prenorm_gemm.cuh
epilogue/sm100_mega_gate.cuh           epilogue/sm100_store_cd.cuh
epilogue/sm100_store_cd_swap_ab.cuh
```

**13 个文件**——也就是所有 SM100 设备代码。文件头注释只有一句 `/// UMMA versions with relaxed
assertions`（`:11`）：CUTLASS 的版本带一堆 static_assert，DeepGEMM 自己发指令时想省掉这些检查
（合法性由它自己的 `make_umma_desc`/`make_instr_desc` 和 kernel 侧断言保证）。文件里只有三类东西：

| 类别 | 行号 | 内容 |
|---|---|---|
| 9 个 `tcgen05.mma` 封装 | `:12-187` | 6 个「数据 kind」× {1SM, 2SM} + 1 个 `ws` 变体 |
| TMEM load | `:190-204` | `tmem_load_32dp32b<N>`，按宽度选 `Nx` 变体 |
| commit + fence | `:206-217` | `umma_arrive_no_elect`、`tcgen05_before/after_thread_sync` |

## 2. 九个 `*_SS::fma`：operand 契约与 PTX 语义

先看最简单也是本篇消费者真正用的那个：

```
struct SM100_MMA_F16BF16_SS {                                  // :12-27
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a, uint64_t const& desc_b,        // A / B 的 smem 描述符（各 64 位）
        uint32_t const& tmem_c,                                // 累加器 TMEM 地址
        uint32_t const& scale_c,                               // 是否累积（0=清零）
        uint64_t const& desc) {                                // 高 32 位 = instruction descriptor
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"                        // :22  p = (scale_c != 0)
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"   // :23
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
               "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};
```

四个非平凡点，全都在这一条 asm 里：

1. **`desc >> 32` 才是 instruction descriptor**（`:25`）。`tcgen05.mma` 的最后一个操作数是一个
   **32 位** `idesc`；DeepGEMM 把它打包进 `desc` 的**高 32 位**再右移取出。因为 `mma/sm100.cuh`
   的 `make_runtime_instr_desc_with_sf_id`（`mma/sm100.cuh:144-148`）正是
   `static_cast<uint64_t>(desc) << 32` —— 低 32 位留给将来的 smem 描述符拼接，高 32 位就是 idesc。
2. **`setp.ne.b32 p, scale_c, 0`**（`:22`）：`tcgen05.mma` 末尾的 predicate 是「**是否累积**」。
   `p=0` 时硬件先**清零**累加器 C 再算。于是 kernel 里控制「首条清零、其余累加」只要传
   `scale_c`：`sm100_bmk_bnk_mn.cuh:174` 传的正是 `s > 0 or k > 0`。
3. **`cta_group::1` vs `::2`**：`_SS` 后缀是「A、B 都在 SMEM」。`cta_group::1` 是单 CTA 用一块
   tensor core；`::2`（CUTLASS 称 2x1SM）让**一个 CTA 对（cluster 里的两个）共用**一条 UMMA，
   累加器横跨两个 CTA 的 TMEM。这就是 DeepGEMM 里 `kNumMulticast==1 ? _SS : _2x1SM_SS` 的分叉
   （`sm100_bf16_gemm.cuh:297`、`sm100_fp8_fp4_gemm_1d1d.cuh:413-414`）。
4. **`kind::`** 决定数据类型通路：`f16`（bf16/fp16）、`f8f6f4`（FP8/FP6/FP4 混合、无 block scale）、
   `mxf8f6f4.block_scale`（FP8/FP6/FP4 + MX block scale）、`mxf4.block_scale`（纯 FP4 + scale）。
   `block_scale` 版多两个操作数 `tmem_sfa`/`tmem_sfb`（`:53-54`）——从 TMEM 指进 scale 的地址。

九个封装的差异可以压成一张表（行号 = `fma` 的 struct 起始）：

| struct | 行号 | cta_group | kind | scale 操作数 |
|---|---|---|---|---|
| `SM100_MMA_F16BF16_SS` | `:12` | 1 | `f16` | 无 |
| `SM100_MMA_F16BF16_2x1SM_SS` | `:29` | 2 | `f16` | 无 |
| `SM100_MMA_MXF8F6F4_SS` | `:46` | 1 | `mxf8f6f4.block_scale` | SFA+SFB |
| `SM100_MMA_MXF8F6F4_2x1SM_SS` | `:67` | 2 | `mxf8f6f4.block_scale` | SFA+SFB |
| `SM100_MMA_F8F6F4_SS` | `:88` | 1 | `f8f6f4` | 无 |
| `SM100_MMA_F8F6F4_2x1SM_SS` | `:106` | 2 | `f8f6f4` | 无 |
| `SM100_MMA_MXF4_SS` | `:124` | 1 | `mxf4.block_scale` | SFA+SFB |
| `SM100_MMA_MXF4_2x1SM_SS` | `:148` | 2 | `mxf4.block_scale` | SFA+SFB |
| `SM100_MMA_F16BF16_WS_SS` | `:172` | 1 | `f16` + `.ws` | 无 |

**谁用哪个**（真实调用点）：

| 封装 | 调用点 |
|---|---|
| `F16BF16_SS` / `F16BF16_2x1SM_SS` | `sm100_bf16_gemm.cuh:297`、`sm100_bf16_mega_gate.cuh:217`、`sm100_bf16_mega_moe.cuh:795`、`sm100_bmk_bnk_mn.cuh:174` |
| `MXF8F6F4_SS` / `MXF8F6F4_2x1SM_SS` | `sm100_fp8_fp4_gemm_1d1d.cuh:414`、`sm100_fp8_fp4_mega_moe.cuh:918`、`sm100_mqa_logits.cuh:372`、`sm100_sparse_mqa_logits.cuh:373` |
| `MXF4_SS` / `MXF4_2x1SM_SS` | `sm100_fp8_fp4_gemm_1d1d.cuh:413`、`sm100_mqa_logits.cuh:372`、`sm100_sparse_mqa_logits.cuh:373` |
| `F8F6F4_SS` / `F8F6F4_2x1SM_SS` | **公开仓库无调用点** |
| `F16BF16_WS_SS` | **公开仓库无调用点** |

两个「休眠」封装值得记一笔：`f8f6f4`（无 scale 的混合低精度）与 `.ws`（weight-stationary，
`:183` 的 `tcgen05.mma.ws`）在公开版里**没有任何调用点**——它们是给未来变体/上游内部版本预留的，
和 `mma/sm100.cuh` 里那些 `#if` 分支一样属于「宁可多写、不留缺口」的底座风格。

`MXF4_SS` 还有一处**编译期版本分支**（`:137-141`、`:161-165`）：CUDA ≥ 12.9 用
`kind::mxf4.block_scale.block32`，否则退回 `kind::mxf4.block_scale.scale_vec::2X`——同一语义、两种
PTX 拼写，因为早期 ptxas 只认后者。本机 CUDA 13.2 走 `block32` 分支。

## 3. TMEM load、commit 与 fence

### 3.1 `tmem_load_32dp32b<N>`（`:190-204`）

TMEM 不能像寄存器那样直接寻址，必须用 `tcgen05.ld` 读回寄存器。CUTLASS 提供一族
`SM100_TMEM_LOAD_32dp32b{4,8,16,32,64,128}x`（32 datapath lanes × 32-bit pattern × N 次重复）。
DeepGEMM 把它包成一个按 `kNumValues` **编译期选型**的模板：

```
template <uint32_t kNumValues>
CUTLASS_DEVICE void tmem_load_32dp32b(const uint32_t tmem_addr, uint32_t* values) {
    DG_STATIC_ASSERT(kNumValues == 4 or 8 or 16 or 32 or 64 or 128, ...);   // :192-194
    using Loader = cute::conditional_t<kNumValues == 4,  cute::SM100_TMEM_LOAD_32dp32b4x,
                   ... 64 → ...64x, 128 → ...128x>>>>>;                     // :195-200
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, values[Is]...);
    }(cute::make_index_sequence<kNumValues>{});                            // :201-203
}
```

`kNumValues` 只能取 2 的幂次中的 6 档（硬件没有 1x/2x 的 32-bit 变体）。调用点很少——**只有 MQA
logits 与 Mega Gate 用**（`sm100_mqa_logits.cuh:44`、`sm100_sparse_mqa_logits.cuh:609`、
`epilogue/sm100_mega_gate.cuh:64`）：因为它们的输出是「打分」而非标准 GEMM tile，读回模式不同。
标准 GEMM 的 epilogue 走的是 `sm100_store_cd.cuh` 里另一套 `SM100_TMEM_LOAD_*`（16dp256b 等）。

### 3.2 `umma_arrive_no_elect`（`:206-209`）

```
CUTLASS_DEVICE void umma_arrive_no_elect(cutlass::arch::ClusterTransactionBarrier& barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))));
}
```

`tcgen05.commit` 是 UMMA 的**完成事件**：硬件在所有已发 MMA 完成后，对目标 mbarrier `arrive`。
名字里的 `_no_elect` 是关键——CUTLASS 的 `cutlass::arch::umma_arrive` **内部先做 `elect_one`**
（只让一个 lane 发），而 DeepGEMM 的这个版本**不做 elect**，由调用方自己保证只在需要的 lane 上调。
看 MQA logits 的用法就很清楚（`sm100_mqa_logits.cuh:421`）：

```
if (cute::elect_one_sync()) { ... ptx::umma_arrive_no_elect(smem.full_tmem_barriers[...]); }  // 外部 elect
```

而 `sm100_bmk_bnk_mn.cuh:180` 直接调 `cutlass::arch::umma_arrive(...)`（自带 elect）——两种风格并存，
`_no_elect` 只是给「已经在 elect 分支里」的调用方省一次判断。

### 3.3 两条 fence（`:211-217`）

```
CUTLASS_DEVICE void tcgen05_before_thread_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); } // :211-213
CUTLASS_DEVICE void tcgen05_after_thread_sync()  { asm volatile("tcgen05.fence::after_thread_sync;"); }  // :215-217
```

TMEM 与 `tcgen05` 异步代理之间需要显式 fence：

- **`after_thread_sync`**：在**等待 mbarrier 之后**、**读 TMEM 或发 MMA 之前**调用。它建立
  「mbarrier 观察到的异步完成 → 后续 tcgen05 操作可见」的顺序。几乎每个 `barrier->wait(...)`
  后面都紧跟一条（如 `sm100_bf16_gemm.cuh:263`、`sm100_bmk_bnk_mn.cuh:154/161/200`）。
- **`before_thread_sync`**：在**用 warp/CTANamedBarrier 同步之前**、**对 TMEM 的操作之后**调用。
  例如 epilogue 读完 TMEM 后要通知「TMEM 可复用」：`sm100_store_cd.cuh:133/139` 在
  `tmem_overlap_barrier->arrive()` / `tmem_empty_barrier->arrive()` 前各插一条。

一句话：**`after` 在等之后、读之前；`before` 在读之后、同步之前**。`tcgen05.commit` 本身**隐含**
了 `before_thread_sync`（`sm100_bmk_bnk_mn.cuh:179` 的注释），所以发完 MMA 直接 commit 即可。

## 4. 消费者逐段：`sm100_bmk_bnk_mn.cuh`（最简单的 SM100 kernel）

`bmk,bnk->mn` 是「把 batch 维并进收缩轴」的批归约 GEMM（数学、host 契约、SM90 版见第 17 篇）。
SM100 版（`sm100_bmk_bnk_mn.cuh`，271 行）**只用 `SM100_MMA_F16BF16_SS` 一个封装**，没有 SF/UTCCP，
是观察 tcgen05 底座最干净的样本。整段 `#if __CUDA_ARCH__ >= 1000`（`:25`，SM100 专属）。

- **配置与 smem**（`:29-77`）：`LAYOUT_AD_M=128`、`kNumTMAStoreStages=2`（`:29-30`）；
  静态断言钉死 `BLOCK_M==128 & BLOCK_N==128 & BLOCK_K==64 & swizzle==128`（`:35-36`）；
  smem 依次放 CD（`:42-43`，`BLOCK_M * 128B * 2 级`）、A、B（`:44-45`）、barrier
  （`:69-73`：`2*kNumStages + 1` 个）、TMEM 指针 4B（`:76`）。
- **warp 角色**（`:80-93`）：warp 1 初始化 mbarrier（`full/empty` 各 `init(1)`、`tmem_full init(1)`，
  `:83-86`）；warp 2 调 `cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem)`
  （`:92`）——**TMEM 是显式分配的资源**，必须由一个 warp 先 alloc 再让所有人用。
- **block 索引 + PDL**（`:97-106`）：`num_total_stages` 处理尾分片（`:103`）；
  `cudaGridDependencySynchronize()`（`:106`）接前驱 kernel。
- **TMA 生产者**（`:108-130`）：warp 0 单 lane 发 `tma::copy`，A/B 行坐标 `+ s_idx*SHAPE`（`:122-123`）
  ——把 batch 维折算成 2D 行偏移（同第 17 篇 §5.4）。
- **MMA warp**（`:131-183`）：`instr_desc = cute::UMMA::make_instr_desc<bfloat16, bfloat16, float,
  UMMA_M, UMMA_N, K, K>()`（`:138`）；`make_umma_desc`（`:141-142`）；**每个 lane 预存一个 stage 的
  描述符 lo**（`:143-144`），循环里用 `__shfl_sync` 取回（`:165-166`）——这样只用一个 32 位寄存器就
  流转了 `kNumStages` 个 smem 偏移。核心那几条：

  ```
  full_barriers[stage_idx]->wait((s/kNumStages) & 1);   // :160
  ptx::tcgen05_after_thread_sync();                     // :161  ← 等之后、发之前
  ... advance_umma_desc_lo<...>(..., k*UMMA_K)          // :170-173
  ptx::SM100_MMA_F16BF16_SS::fma(a_desc, b_desc, 0, s > 0 or k > 0, runtime_instr_desc);  // :174
  cutlass::arch::umma_arrive(empty_barriers[stage_idx]); // :180  通知消费者
  ... tmem_full_barrier                                     // :182  全部完成后通知 epilogue
  ```

  注意 `tmem_c` 参数恒为 `0`（累加器从 TMEM 列 0 开始），`scale_c = s>0 or k>0` 实现「首条清零」。
- **TMEM 唯一性检查**（`:188-189`）：`DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem)==0)`
  ——因为用了 `Allocator1Sm`，本 kernel 禁止两个 CTA 共享同一 SM 的 TMEM。
- **epilogue**（`:191-258`）：等 `tmem_full_barrier`（`:199`）→ 逐 atom `SM100_TMEM_LOAD_32dp32b4x`
  读回 4 个 float 再 `st_shared`（`:245-248`）→ `tma_store_fence` + `NamedBarrier` 后由 warp 0 发
  **`cute::SM90_TMA_REDUCE_ADD_2D::copy`**（`:255`）把 smem tile **加**回全局 D。这就是第 17 篇 §6
  说的「TMA reduce-add 取代 atomicAdd」，也是 SM100 与 SM90 版最本质的差别。

host 包装层（`csrc/jit_kernels/impls/sm100_bmk_bnk_mn.hpp`）：block 128×128×64、`num_threads=128`
（`:19-22`，SM90 是 384——因为 UMMA 只需单 warp 发）；`split_factor = ceil_div(s*(k/64),
max(num_sms/num_mn_blocks, 1))`（`:34`）；stage 从 4 起按 `smem_capacity` 回退（`:38-55`）；
**D 描述符是 `(n, m)` 方向**（`:69`，`make_tma_2d_desc(d, n, m, block_n, block_m, n, swizzle_cd_mode)`）
——因为 TMA reduce-add 的坐标 `(n_idx, m_idx)` 与 smem 里 `[M][N]` 排布一致。

## 5. 与第 49 篇（SM90 WGMMA 底座）的对照

| 维度 | SM90（第 49 篇） | SM100（本篇） |
|---|---|---|
| 指令 | `wgmma.mma_async` | `tcgen05.mma` |
| 描述符 | 64 位 `GmmaDescriptor`（smem 地址 + LBO/SBO + layout_type） | `mma/sm100.cuh` 的 `SmemDescriptor`（同 64 位）+ 独立 32 位 **instruction descriptor** |
| 谁发 | 每个 warpgroup（128 线程）发 | **单 CTA 的单个 warp 的单 lane** |
| 累加器 | 寄存器（`CLayout_64xN`） | **TMEM**（显式 `Allocator1Sm` 分配） |
| 完成事件 | `wgmma.commit_group` + `wgmma.wait_group<N>` | `tcgen05.commit` → mbarrier `arrive` |
| fence | `wgmma.fence` | `tcgen05.fence::before/after_thread_sync` |
| 描述符推进 | `advance_gmma_desc_lo`（低 32 位三段步进） | `advance_umma_desc_lo`（低 32 位，`byte>>4`） |
| 选择器 | `FP8/BF16/TF32MMASelector`（`K`=32/16/8） | `kind::f16/f8f6f4/mxf8f6f4.block_scale/mxf4.block_scale` |

最大的范式变化是「**累加器从寄存器搬进 TMEM**」：SM90 每个 math 线程都持有自己的累加器寄存器；
SM100 累加器是**一块 512 列的共享资源**，必须显式 alloc/free，读回要 `tcgen05.ld`。第 35 篇讲的
「512 列 TMEM 预算 + overlap 技巧」正由此而来。

## 6. H100 实测（`harness/exp_tcgen05_umma.{sh,out.txt}` + `src/tcgen05_umma.cu`）

tcgen05 指令本身无法在 H100 上运行，但**两条硬事实**可以真机验证。

### 6.1 Part A：tcgen05 指令的 ptxas 编译矩阵（硬证据）

把 `tcgen05.cuh` 的每个封装抽成一个最小 `__global__`，分别编 `sm_90a` / `sm_100a` cubin：

```
  f16bf16_ss     sm_90a=REJECT sm_100a=OK  | Instruction 'tcgen05.mma' not supported on .target
  f16bf16_2x1    sm_90a=REJECT sm_100a=OK  | Instruction 'tcgen05.mma' not supported on .target
  mxf8f6f4_ss    sm_90a=REJECT sm_100a=OK  | Instruction 'tcgen05.mma' not supported on .target
  mxf4_ss        sm_90a=REJECT sm_100a=OK  | Instruction 'tcgen05.mma' not supported on .target
  f16bf16_ws     sm_90a=REJECT sm_100a=OK  | Instruction 'tcgen05.mma.ws' not supported on .target
  tmem_load_4    sm_90a=OK     sm_100a=OK
  umma_arrive    sm_90a=REJECT sm_100a=OK  | Instruction 'tcgen05.commit' not supported on .target
  fences         sm_90a=REJECT sm_100a=OK  | Instruction 'tcgen05.fence' not supported on .target
```

五个关键观察：

1. **`tcgen05.mma` / `.mma.ws` / `.commit` / `.fence` 在 `sm_90a` 被 ptxas 逐条拒绝**——
   `Feature '.cta_group::1' not supported`（mma/commit）与 `Instruction '...' not supported on
   .target 'sm_90a'`（fence）。这是「SM100 专属」最硬的证据（同第 62/92 篇对 `cvt.rs`/`red.async`
   的判法）。
2. **`tmem_load` 是唯一例外**：`tcgen05.ld` 在 `sm_90a` 下**编得过**。原因不是硬件支持，
   而是 CUTLASS 在 `cute/arch/copy_sm100.hpp:3474` 把 asm 包在
   `#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)` 里，否则落到 `CUTE_INVALID_CONTROL_PATH`
   （一个 host 断言）。**编得过 ≠ 跑得动**：这行在 H100 上会 trap。DeepGEMM 自己的封装则
   **裸写 asm、不做 arch 守卫**——所以 `tcgen05.cuh` 里除 `tmem_load` 外每一条都会在
   `sm_90a` 编译期就炸，这是「宁可编译期失败、不要运行期静默」的取舍。
3. `tcgen05.cuh` 与 `mma/sm100.cuh` 这两个头文件在 `sm_90a` 下**可被 include**（主实验 TU 编译
   通过），说明它们本身没有 arch 依赖，只有被实例化的 asm 才有。

### 6.2 Part 1：STSM swizzle 整数公式 == `cute::Swizzle<3,4,3>`

`sm100_bmk_bnk_mn.cuh:223-249` 的 epilogue 手工算 `(row, col8)` 的 smem 字节偏移
（`row = lane`，`col ^= row % (kSwizzleCDMode/16)`，`offset = warp*32*128 + row*128 + col*16`）。
把这段整数公式逐行抽出，对**全部 128×8 个 (row, col8)** 与 `cute::Swizzle<3,4,3>` 枚举比对：

```
(row,col8) 双射检查 mismatch=0/1024
16B 组内展开后碰撞=0/16384
```

即 `Swizzle<3,4,3>` 的 `(B=3, M=4, S=3)` 正是这段代码在做的事：16B 基元（M=4）+ 8 行周期
（S=3，行 stride=2⁷）+ 3 位 XOR（B=3）。这解释了为什么 CUTLASS TMA 用 128B swizzle 时 smem tile
的物理布局就等于这段公式。

### 6.3 Part 2/3：真实 TMA reduce-add（H100 sm90 可跑）

reduce-add 那一步不用 SM100 专属指令——`cp.reduce.async.bulk.tensor...add.f32` **Hopper 就支持**
（CUTLASS `SM90_TMA_REDUCE_ADD_2D`，`cute/arch/copy_sm90_tma.hpp:1304`）。真机：

```
Part 2 (plain, 128x128 fp32): D 初值 0.5，连续 reduce-add 3 次 1.0
  0.5 + 3x(1.0) -> min=3.5000 max=3.5000   （若为覆盖写应为 1.0）
Part 3 (swizzle-128, 对齐 bmk D 描述符方向): D[:,0:32]+=2.0, D[:,32:64]+=3.0
  mismatch=0/16384
```

Part 2 坐实 **reduce-add 是「加」不是「写」**（0.5→3.5，反例覆盖写会得 1.0）；Part 3 用
`Swizzle<3,4,3>` 铺 smem、以 bmk 的 `(n,m)` 方向 TMA 描述符发 reduce-add，两种列偏移各打对，
`mismatch=0`——**同时验证了 §6.2 的 swizzle 公式与 §4 的 D 描述符方向**。

## 7. 结论与边界

- **`ptx/tcgen05.cuh` 是 SM100 上所有设备代码的裸指令地板**：9 个 `tcgen05.mma` 封装
  （6 个 data kind × {1SM,2SM} + 1 个 WS）、`tmem_load_32dp32b`、`umma_arrive_no_elect`、两条
  `tcgen05.fence`。它与第 49 篇（SM90 WGMMA 底座）严格对偶：**描述符→指令描述符拆成两个 32 位、
  累加器从寄存器搬进 TMEM、完成事件从 `wait_group` 换成 `commit`→mbarrier、fence 换成
  `before/after_thread_sync`**。
- **`desc >> 32` + `setp.ne.b32` 是本层最浓缩的两行**：前者把 32 位 instruction descriptor 从
  64 位 `desc` 的高位取出，后者把 `scale_c != 0` 变成 `tcgen05.mma` 的「积累加/清零」predicate——
  全 kernel 的「首条清零、其余累加」都只是传对这个布尔。
- **`F16BF16_WS_SS` 与 `F8F6F4_*` 在公开仓库无调用点**：底座按最大集写，变体留给未来。
- **SM100 批归约 GEMM 的位置**：相对第 17 篇 §6，本篇把 `sm100_bmk_bnk_mn.cuh` 从「三行对照」
  落到「一个只用 `F16BF16_SS` 的最小消费者」，并补了 host 包装层的 `split_factor` / stage 回退 /
  **D 描述符 `(n,m)` 方向**这三处细节。
- **边界**：tcgen05 指令本身 SM100 专属、仅分析（§6.1 的 ptxas 矩阵是唯一能在 H100 上给出的
  「是/否」证据）；但**它脚下的内存几何**（STSM swizzle、TMA reduce-add）架构无关，§6.2/§6.3 在
  H100 上真机验证——这也说明「SM100 kernel 慢/对」的相当一部分其实由 Hopper 就已存在、且能真机
  复现的访存行为决定。

## 附录：实验复现

```
bash code/mega-moe/harness/exp_tcgen05_umma.sh > code/mega-moe/harness/exp_tcgen05_umma.out.txt
```

环境：H100 80GB HBM3 / sm90 / CUDA 13.2 / PyTorch 2.12 / DeepGEMM `057ca59`（editable 安装）。
源文件 sha256（前 16 位）：`tcgen05.cuh 7760eef80c6a38ee`、`sm100_bmk_bnk_mn.cuh 49f167d1a23a65ba`、
`sm100_bmk_bnk_mn.hpp 02e180967718a626`、`mma/sm100.cuh 439c57d8aa4914c0`。总判定
**PASS=7 / FAIL=0**。
