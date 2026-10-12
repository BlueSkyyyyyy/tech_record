# 36. DeepGEMM 的 SM100 BF16 主 kernel 与 tensor-core 利用率控制（`set_tc_util`）

> 本篇是本专题「稠密 GEMM 主 kernel」矩阵的**最后一格**，外加一个很少有人注意到的
> **运行时开关**。前几篇已经凑齐：SM90 的 FP8（第 22 篇，1D1D/1D2D）、SM90 的 BF16（第 19 篇）、
> SM100 的 FP8/FP4（第 35 篇）；只差 **SM100 的 BF16**——`sm100_bf16_gemm.cuh`。
> 它是 SM100 上所有 **BF16** GEMM 的唯一后端：稠密 / m-grouped（dgrad）/ k-grouped（wgrad）/
> batched（MLA einsum `bhr,hdr->bhd`、`bhd,hdr->bhr`）共用一套模板（host 六个入口，
> `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:82/138/210/268/335/387`）。与第 35 篇的
> `sm100_fp8_fp4_gemm_1d1d` 是**同一套 UMMA+TMEM 骨架**，只把「块级缩放（SF）」这条路整段拿掉。
>
> 它也藏着本专题一直没见过的一个机制：**`deep_gemm.set_tc_util`（tensor-core 利用率控制）**。
> 这是一个公开的 Python 开关（`README.md:151`），却**只有这一个 kernel**真正实现它——
> 通过在每次 UMMA 之后插入 `clock64` 空转来把 tensor core 的占空比**按比例压低**，
> 源码注释写明意图是「let tensor cores relax for lower possibility of frequency drop」
> （`sm100_bf16_gemm.cuh:325`）。本篇把它逐行拆开，并在 H100 上跑真机复刻。
>
> **重要**：kernel 主体仍是 **SM100 专属，仅分析**（`sm100_bf16_gemm.cuh:43`
> 整段 `#if __CUDA_ARCH__ >= 1000`；UMMA/TMEM 全是 SM100 硬件）。但 `set_tc_util` 的
> **语义层**与**节流机制**（公式 + `clock64` 空转，不依赖 UMMA）都能在 H100 上真跑：
> 一是验证它**对 SM90 kernel 完全无效**（证明它只编进 SM100 BF16），二是把节流机制复刻成
> 独立 kernel 量出「利用率 vs 吞吐」。原始输出见
> `harness/exp_tc_util.{sh,out.txt}`、`harness/exp_tc_util_throttle.{sh,out.txt}`。
>
> 目标卡代号敏感：全文用「目标卡」指代。

---

## 1. 全景：现在的「稠密 GEMM 主 kernel」矩阵

| 架构 | dtype | kernel | 累加器 | SF | 装甲 (本篇) |
|---|---|---|---|---|---|
| SM90 | FP8 | `sm90_fp8_gemm_1d1d` / `1d2d` | 寄存器 | 1D1D / 1D2D | 第 22 篇 |
| SM90 | BF16 | `sm90_bf16_gemm` | 寄存器 | 无 | 第 19 篇 |
| SM100 | FP8/FP4 | `sm100_fp8_fp4_gemm_1d1d` | TMEM | block-scale(UTCCP) | 第 35 篇 |
| **SM100** | **BF16** | **`sm100_bf16_gemm`** | **TMEM** | **无** | **本篇** |

一句话：`sm100_bf16_gemm` = 第 35 篇那套 **UMMA + TMEM + 五类 warp** 的骨架，**减去 SF/UTCCP**，
再加上两个它独有的东西——**device 内合并 stage（把 `BLOCK_K` 放大）** 与
**tensor-core 利用率节流**。因为 BF16 没有 SF 通路，host 也能复用同一套包装
（第 35 篇 §2.3 已点出 `MmaKind::BF16` 的 SF 块尺寸返回 `{0,0}`）。

---

## 2. host 全链路：`set_tc_util` 从 Python 到模板参数

先把「开关」这条线走通（所有行号核对自 `~/github/DeepGEMM`，commit `057ca59`）：

- Python 绑定：`csrc/apis/config.hpp:21-25` —— `set_tc_util(int)` / `get_tc_util()`。
  同时导出的还有 `set_num_sms`、`set_pdl`（`:26-31`）等「运行时开关」。
- 运行时状态：`csrc/runtime/runtime.hpp:22` —— `int num_sms = 0, tc_util = 0;`（**默认 0**）。
  `:93-96` `set_tc_util` 断言 `0 <= util <= 100`；`:98-100` `get_tc_util()` **把 0 翻成 100**
  ——所以「默认」对外表现为 **100（不节流）**，而内部用 0 当「未设置」哨兵。
  （`csrc/apis/config.hpp:24` 的 `get_tc_util` 就是把 runtime 的这个值透出去。）
- 描述符字段：`csrc/jit_kernels/heuristics/config.hpp:22` —— `GemmDesc` 里
  `int num_sms, tc_util;`，`:77` 打印时带 `tc_util=...`。**每个** kernel 的 `GemmDesc` 都填它
  （如 `sm90_fp8_gemm_1d1d.hpp:96`、`sm100_fp8_fp4_gemm_1d1d.hpp:119`、`sm100_bf16_gemm.hpp:99`）。
- **但只有 SM100 BF16 把它当模板实参编进 kernel**：`sm100_bf16_gemm.hpp:70`
  （`compile_and_launch` 的微源码实例化里最后一个实参就是 `args.gemm_desc.tc_util`）；
  对应 kernel 模板最后一位 `uint64_t kTensorCoreUtilControl`（`sm100_bf16_gemm.cuh:35`）。
  反观 `sm90_bf16_gemm.hpp:43` 与 `sm100_fp8_fp4_gemm_1d1d.hpp` 的实例化模板列表里**都没有**它
  ——`tc_util` 只进了 `GemmDesc`（打印/JIT 微源码之外的元数据），**没进微源码文本**，
  因此对这些 kernel 既不改行为、也不改 JIT 缓存键。

> **为什么只有 BF16？** BF16 主循环的每条 UMMA 更「轻」（`UMMA_K=16`）而 stage 更多，
> 满负荷持续发射 `tcgen05.mma` 更容易把 tensor core 长时间顶在满功率上；而 FP8/FP4 的
> block-scale UMMA 本身重、且 SF/UTCCP 已经自然分词。于是「让 tensor core 歇一歇」的
> 钩子只加在 BF16 这一条最「密」的发射序列上。这是从代码结构能读出的动机，非官方文档。

---

## 3. kernel 骨架：与 FP8/FP4（第 35 篇）的差异

模板签名（`sm100_bf16_gemm.cuh:17-35`）与第 35 篇几乎同构，逐项差异：

- **无 SF**：没有 `sf_dtype` / `gran_k` 模板参数，主循环里没有 UTCCP、没有
  `tmem_overlap_barriers`、没有 `make_instr_desc_block_scaled`。
- **`UMMA_K = 16`**（`:69`）——BF16 的硬件 K 步长；FP8 是 32、FP4 是 64（第 35 篇）。
  `BLOCK_K % UMMA_K == 0`（`:74`）。
- **`BLOCK_K_ == 64` 静态钉死**（`:72`）：host 的 `block_k = 128*8/bits`
  （`heuristics/sm100.hpp:32`）对 BF16（16 bit）= 64。
- **`kDoMergeStages` 合并 stage**（`:44-53`）——本节重点，见 §4。
- **`kTensorCoreUtilControl` 节流**（`:141-142, 326-343`）——§5。
- **`kSwapAB`**：dgrad 的 m-grouped 强制 `swap_ab=true`（`heuristics/sm100.hpp:37-44`，
  `block_m` = contiguous 对齐、`block_n=128`），所以 kernel 里 UMMA 的 A/B 与
  `instr_desc` 的 major 顺序要按 `kSwapAB` 互换（`:237-240`），并用
  `update_instr_desc_with_umma_n` 按有效 M 动态缩 N（`:284-287`）——与第 35 篇一致。
- **MMA 指令**：`SM100_MMA_F16BF16_SS`（1-CTA）/ `SM100_MMA_F16BF16_2x1SM_SS`（2-CTA）
  （`ptx/tcgen05.cuh:12` / `:29`；`kind::f16`），发射循环 `:301-318`。
- **C/D 支持 BF16 或 FP32，含 accumulation**（`:63`），SF 路径的 FP8 输出对不存在。
- **PDL**：`:153` `cudaGridDependencySynchronize()`——支持 Programmatic Dependent Launch
  （与 `set_pdl` 配套），BF16 稠密也吃这一套。

共享存储布局 `SM100BF16GemmSharedStorage`（`layout/gemm.cuh:15-34`）里除了 full/empty/
tmem_full/tmem_empty 四种 barrier，还专门为节流留了一个
**`Barrier tensor_core_full_barrier;`**（`layout/gemm.cuh:33`，注释「the last barrier is for
tensor core utilization control」，`heuristics/sm100.hpp:220`），并在 host 的 smem 预算里
计入（`heuristics/sm100.hpp:221` 的 `+ 8`）。

---

## 4. 重点一：device 内合并 stage（`BLOCK_K` 从 64 放大到最多 256）

```cpp
// sm100_bf16_gemm.cuh:46-53
constexpr bool kDoMergeStages =
    kNumStages_ >= 8 and kGemmType == GemmType::Normal and
    kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
constexpr uint32_t kNumMinStages = 8;
constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;   // 64 → 64*m
constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;
```

- host 只按 `BLOCK_K_=64` 算 stage 数（`heuristics/sm100.hpp:200-250` 的
  `num_stages = min((smem_cap - extra)/per_stage, 32)`）。因为 BF16 每 stage 只装 64 个 K
  的 A/B，smem 足够塞下很多 stage（常到 16~32）。
- 但 UMMA 的发射开销（`umma_arrive` / commit / 屏障等待）与 K block 数成正比。于是 kernel
  **把 `kNumStagesPerMerge = kNumStages_/8` 个 stage 合成一个 `BLOCK_K`**，让 K block 数
  减到约 1/`m`，同时保留至少 8 个 stage 做流水。`:45` 注释直说
  「this is for reducing the `umma_arrive()` overhead」。
- 合并后每个「原子块」仍是 64 个 K：`BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge`（`:244`），
  描述符按 `(atom_k_idx, inner_k_idx)` 两段推进（`:304-309`
  `advance_umma_desc_lo<...>(base, atom_k_idx*LOAD*BLOCK_ATOM_K, inner_k_idx)`）。
  这解释了为什么发射循环里同时有 `umma_k_idx` 与 `atom_k_idx/inner_k_idx` 两级索引
  （`:303-317`）——**物理上仍是 stage 粒度，逻辑上按合并后的大 `BLOCK_K` 迭代**。
- 触发条件很窄：**仅 `GemmType::Normal`（稠密）且 A/B 都 K-major**。m-grouped 的 swap-AB
  （B 变 MN-major）与 k-grouped 都不合并——与 `DG_STATIC_ASSERT(... kKAlignment % BLOCK_K == 0)`
  （`:74`）呼应：k-grouped 的 `k_alignment` 是 128 的倍数，若 `BLOCK_K` 被放大到 192/256 会破坏对齐。

对照：第 19 篇 SM90 BF16 **也有**「合并 stage」技巧（`BLOCK_K *= kNumStagesPerMerge`），
但 SM90 合并是为了减少 `warpgroup_wait<0>`；SM100 这里合并是为了减少 **UMMA 提交/屏障开销**，
且条件更严（必须 K-major）。这是同一种思想在两代硬件上的不同落点。

---

## 5. 重点二：tensor-core 利用率控制（节流）

### 5.1 机制逐行

```cpp
// sm100_bf16_gemm.cuh:326-343（MMA issue warp，每次 K block 之后）
DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
if constexpr (kTensorCoreUtilControl < 100) {
    umma_arrive(reinterpret_cast<uint64_t*>(&smem.tensor_core_full_barrier));
    __syncwarp();
    smem.tensor_core_full_barrier.wait(tensor_core_phase);   // 等最后一条 UMMA 真正做完
    tensor_core_phase ^= 1;
    // Sleep for certain cycles
    constexpr static uint64_t kNumUMMACycles  = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
    const auto start_clock = clock64();
    if (cute::elect_one_sync())
        while (clock64() - start_clock < kNumDummyCycles) {}   // 单线程空转
    __syncwarp();
}
```

四步：

1. 用 `tensor_core_full_barrier` **等这一批 UMMA 真正算完**（否则空转没意义——要等 tensor
   core 真的停下来再歇）。
2. 由 `(UMMA_M, UMMA_N, BLOCK_K)` 估出这一批的 **UMMA 周期数** `kNumUMMACycles`：
   `2·M·N·K / 8192`（`2` 是乘加；`8192 = 128×64` 是 SM100 单个 tensor-core 每周期 MAC 数的
   量级——一个**经验常数**，把「工作量」折成「周期」）。
3. 按目标利用率算空转周期：`kNumDummyCycles = (100-util)/util · kNumUMMACycles`。于是
   `work : (work+dummy) = util : 100`，**目标占空比精确等于 `util`**。
4. 只让 `elect_one_sync()` 选出的**一个 lane** 自旋（与「只有一个 warp 发 UMMA」对称），
   其余 lane 在 `__syncwarp()` 等它。

初始化时（`:141-142`）只在 `kTensorCoreUtilControl < 100` 才 init 这个 barrier，否则整段
被 `if constexpr` 编掉、零开销。`kNumUMMACycles` 的周期当量的量级可手算：
`(UMMA_M=256, UMMA_N=128, BLOCK_K=256) → 2·256·128·256/8192 = 2048` 周期。

### 5.2 为什么需要它

注释（`:325`）是唯一动机说明：**「Let tensor cores relax for lower possibility of frequency drop」**。
持续把 tensor core 顶满会拉高功耗、触发降频（进而**全局**降速）。主动在每次 UMMA 后空转一小段，
把占空比降到 `util%`，让功耗有回落窗口——用**少量吞吐换取更稳的频率**。对延迟不敏感、
但对**功耗/频率稳定性**敏感的场景（如长时间推理、多租户共卡）有用。
它是一个**近似**控制：`kNumUMMACycles` 是估算值，且空转超调（见 §6.2）。

---

## 6. H100 实测（架构无关部分跑真机）

### 6.1 `set_tc_util` 在 SM90 上完全无效（证明「只编进 SM100 BF16」）

`harness/exp_tc_util.sh` 用**独立冷 JIT 缓存**（`DG_JIT_CACHE_DIR` 指向临时目录）跑
`exp_tc_util.py`：

- **API 语义**：默认 `get_tc_util()=100`；`set(75)→75`、`set(0)→100`（0 是「关闭」哨兵）；
  `set(-1)`/`set(101)` 都在 `runtime.hpp:94` 被断言拒绝。
- **SM90 BF16 主 kernel**（`m=n=k=4096`，H100 真实设备代码）在不同 `tc_util` 下：

| tc_util | 延迟 us | TFLOPS | relerr |
|---|---|---|---|
| 100 | 182.4 | 753 | 1.5e-4 |
| 50 | 182.5 | 753 | 1.5e-4 |
| 10 | 182.7 | 752 | 1.5e-4 |
| 100（复测） | 182.5 | 753 | 1.5e-4 |

  完全不变。**SM90 FP8（1D2D，4096³）** 同理：`tc_util=100/50/10` 都是 **107.2/107.3/107.3 us、
  1282/1281/1281 TFLOPS**。说明 `set_tc_util` 在这两个 kernel 上**改了 runtime 状态但不改行为**。
- **编译缓存**：冷缓存下同一 shape 首次调用 **2558 ms**（JIT 编译）；`set_tc_util(33)` 后再调
  同一 shape 只要 **2.37 ms**（磁盘缓存命中）——坐实 `tc_util` **不在** SM90 的 JIT 微源码/缓存键里。

### 6.2 节流机制复刻（`src/tc_util_throttle.cu`）

把 §5.1 的公式与空转宏**逐行搬到** H100 独立 kernel：用一个固定长度的 FFMA「工作突发」
代表一次 UMMA 的周期占用，按同一公式空转，量「有效利用率」与「墙时间」。

- **公式层**：对若干真实 SM100 配置，`cyc/(cyc+dummy)` **精确等于 `util/100`**
  （如 `(256,128,256)→2048` 周期；`util=50` 空转 2048、`util=25` 空转 6144，占比正好 50%/25%）。
- **真机层**（132 SM 全压满，200k 迭代）：

| util | t(ms) | work/total 实测 | SM_MHz | t/t_100 |
|---|---|---|---|---|
| 100 | 3472.6 | 0.999 | 1993 | 1.00 |
| 75 | 6449.5 | 0.645 | 1914 | 1.86 |
| 50 | 12848.9 | 0.320 | 1433 | 3.70 |
| 25 | 27047.5 | 0.190 | 1846 | 7.79 |
| 10 | 32046.9 | 0.047 | 1995 | 9.23 |

  三点观察：
  1. `work/total` 随 `util` **单调下降**、墙时间大幅拉长 —— 机制确实按比例压低吞吐；
  2. 但实测值**系统性低于** `util/100`（`util=50` 只到 ~0.32，而非 0.50）：
     因为 `while(clock64()-start<dummy)` 每次读 `clock64` 有固定延迟，**空转超调**；
  3. H100 的 **SM 频率并未随降载单调抬升**（`util=50/25` 反而更低）。所以「避免降频」这个
     意图**无法**用 H100 上的 FFMA 合成负载复现——它依赖 SM100 tensor-core 的功耗/热行为。
     H100 上能确认的只是**节流机制本身按比例压低吞吐**这一件事。

---

## 7. 结论与边界

- `sm100_bf16_gemm` 补齐了「稠密 GEMM 主 kernel」矩阵的最后一格：与第 35 篇的 FP8/FP4 是同一套
  UMMA+TMEM 骨架，**去掉 SF/UTCCP**，`UMMA_K=16`、`BLOCK_K_=64`，并新增两个独有机制。
- **合并 stage**：仅 K-major 的 `GemmType::Normal` 且 stage≥8 时把 `BLOCK_K` 放大
  `kNumStagesPerMerge` 倍（64→最多 256），目的是少发 `umma_arrive`；与第 19 篇 SM90 的
  merge 是同一思想的不同动机。
- **`set_tc_util`**：一个从 Python 直通到**模板参数**的运行时开关，但**只有这个 kernel** 实现：
  每次 UMMA 后按 `(100-util)/util·kNumUMMACycles` 空转、把占空比压到 `util%`，意图避免降频。
  H100 实测坐实：① 对 SM90 bf16/fp8 **零影响、不改缓存键**；② 机制复刻后吞吐随 `util`
  单调下降，但空转超调使实际利用率偏低、频率效应不可复现。
- **边界**：kernel 主体 SM100 专属，仅分析；H100 只验证了架构无关的**开关语义**与**节流机制**。

---

## 8. 引用

- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>
  - kernel：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh`
  - host：`csrc/jit_kernels/impls/sm100_bf16_gemm.hpp`、`csrc/apis/gemm.hpp`
  - 运行时开关：`csrc/apis/config.hpp`、`csrc/runtime/runtime.hpp`、`README.md`（Utilities 段）
  - 共享存储 / 启发式：`deep_gemm/include/deep_gemm/layout/gemm.cuh`、
    `csrc/jit_kernels/heuristics/sm100.hpp`、`csrc/jit_kernels/heuristics/config.hpp`
  - MMA：`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`、`deep_gemm/include/deep_gemm/mma/sm100.cuh`
- 版本：DeepGEMM `057ca59`（Public release 26/09/30）。
- 本专题对照篇：第 19 篇（SM90 BF16 主 kernel）、第 22 篇（SM90 FP8 1D1D/1D2D）、
  第 35 篇（SM100 FP8/FP4 主 kernel）、第 31 篇（SM100 epilogue 算子）。
