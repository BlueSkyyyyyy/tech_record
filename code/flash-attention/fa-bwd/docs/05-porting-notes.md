# FA 反向：目标卡抽象层与移植注意事项（P4-2）

> 目的：把现有 fp16 / bf16 / fp8 三套反向实现（单文件 + 两文件）里**与具体 GPU 强相关**
> 的部分切出来，列成「可替换层」。换到**目标 AI 卡**（非 H100/sm90，指令集可能不同）时，
> 只需按本清单逐层替换，host 侧的数值对拍 / 性能口径 / 数据 I/O 全部复用。
>
> 代码事实来源（写作时已读文件核对）：
> `src/fp16/fa_bwd_fp16_kernels.cuh`、`src/bf16/fa_bwd_bf16_kernels.cuh`、
> `src/fp8/fa_bwd_fp8_kernels.cuh`、`src/*/fa_bwd_*_main.cu`、
> `scripts/{lab,run,ncu}.sh`、`harness/fa_bwd_bench.py`；
> bound / 性能结论见 `04-numerics-and-perf-summary.md` 与 `03-fp8-bwd-impl.md`。

## 先记住一个画面

把移植想成**搬家**。

- 家具（数学、数据流、对拍脚本）可以整体搬走；
- 但新房的**承重墙、插座位置、门框宽度**不一样——这些必须按新房重做。

对应到 kernel：**L0–L2（数据、host、算法）是家具，L3（计算指令）与 L4（smem 布局）是承重墙**。搬家的失败，十有八九不是家具坏了，而是承重墙没对上。

> **类比的边界**：搬家搬错了可以重摆；kernel 移植错了往往**不报错，只算错**——访存越界读到别的 tile、bank conflict 只是变慢、scale 折错只是数值系统性偏大。所以本篇把"正确性锚点"（§3）单独列成一节：**它们与硬件无关，但错了就静默错**。

读完这一篇，你应该能回答：

1. 六个抽象层里，哪几层可以原样搬走，哪几层必须重写？判断依据是什么？
2. `ldmatrix` 的行距为什么要 padding？fp16 不 padding、bf16 却要 +2，差别在哪？
3. 移植后数值不对，应该按什么顺序排查？先查指令、先查布局，还是先查 scale 记账？

### 结论速查

| 问题 | 结论 | 详见 |
|---|---|---|
| 必须重写 | L3 计算指令、L4 smem 布局 | §1、§2 |
| 可以复用 | L0 数据/口径、L1 host、L2 数据流、L6 同步 | §1、§2 |
| 半可移植 | L5 量化数学（口径可搬，intrinsic 要换） | §2 L5 |
| 最容易静默错 | scale 沿归约维的折叠、S 累加器清零、fp8 解码 | §3 |
| 移植顺序 | 指令集 → L3 → L4 → smem 上限 → 解码 → 对拍 → 对标 | §4 |

---

## 1. 抽象层总览

现有实现是「**三段式 kernel 流水 + 一个量化前置**」，host 侧只做 I/O 与计时。按可移植性从强到弱分成 6 层：

```
┌──────────────────────────────────────────────────────────────────────┐
│ L0 数据/口径层   npy(q,k,v,do,o,ref_*,te_*)  ·  FLOPs 口径  ·  复现命令 │  跨卡不变
├──────────────────────────────────────────────────────────────────────┤
│ L1 host/launcher  grid/block 维度 · 动态 smem 属性 · 计时与对拍         │  基本不变
├──────────────────────────────────────────────────────────────────────┤
│ L2 算法/数据流层  preprocess(D,LSE) → main(1colblock) → convert         │  跨卡不变
├──────────────────────────────────────────────────────────────────────┤
│ L3 计算层        dot-product(标量) / mma.m16n8k32 / ldmatrix            │  ★换卡要换
├──────────────────────────────────────────────────────────────────────┤
│ L4 存储层        smem 布局 + 行距 padding + bank 冲突规避               │  ★随 L3 换
├──────────────────────────────────────────────────────────────────────┤
│ L5 精度/量化层   fp8 转换 intrinsics + rowwise scale 折算               │  半可移植
├──────────────────────────────────────────────────────────────────────┤
│ L6 同步层        __syncthreads / atomicAdd / __launch_bounds__          │  基本不变
└──────────────────────────────────────────────────────────────────────┘
```

**关键判断**：L0–L2、L6 是与硬件解耦的骨架；真正需要为目标卡重写的是 **L3（计算指令）** 以及被它牵连的 **L4（smem 布局）**；L5 的量化**数学口径**可移植，但 fp8 转换 intrinsic 按厂商不同。

这也解释了为什么本项目的路线是「先标量 golden（L3=FFMA）→ 再张量核（L3=mma）」：**L3 可替换，是同一套 L0–L2 代码的两个后端**。先写标量版不是为了"简单"，而是为了在换 L3 之前先把 L0–L2 的正确性钉死。

---

## 2. 逐层接口清单（换卡要改什么）

### L0 数据 / 口径层（跨卡不变）

| 项 | 位置 | 说明 |
|---|---|---|
| 输入/输出 npy | `~/proj/output/fa-bwd/<case>/` | CPU fp32 保存（fp16/bf16 无损）；`q,k,v,do,o` + `ref_{o,dq,dk,dv}` + `te_*` + `meta.json` |
| 新 shape dump | `harness/fa_bwd_bench.py dump` | 任意实现 load 同一份输入做逐元素比对 |
| FLOPs 口径 | `4·B·S²·H·D`（causal 不折半） | ours/FA/TE 同口径，横向可比 |
| 峰值常量 | `agent_skills/kernel-opt.md` | H100：HBM 3.35 TB/s；FP16/BF16 TC 989；FP8 1978.8 TFLOPS |
| 计时口径 | ours=CUDA event；FA/TE=CUPTI 纯 device | 见 `04` §0 |

> 移植到目标卡时，**只换「峰值常量」与「计时口径换算」**；npy 与对拍脚本不动。这一层是整个移植工作的"锚"——只要输入输出还是同一份 npy，你永远知道自己是"算错了"还是"算慢了"。

### L1 host / launcher（基本不变）

| 接口 | fp16 / bf16 | fp8 | 换卡动作 |
|---|---|---|---|
| preprocess grid | `(S, H, B)`，block `THREADS=128` | 同 | 不变 |
| main grid | `(ceil(S/BM), H, B)` | 同 | 不变 |
| convert grid | `min(ceil(n/256), 65535)` | 同 | 不变 |
| 动态 smem | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, SMEM_BYTES)` | 同 | 按新卡 smem 上限复核 |
| arch flag | `scripts/run.sh` 的 `ARCH=sm_90` | 同 | **改 `ARCH`**；WGMMA 类指令必须 `-gencode=arch=compute_XXa,code=sm_XXa`（见 kernel-opt 坑） |
| 自测对拍 | `diff_stat`：`max_abs` + `max_rel` | 同 | 不变 |

> `cudaFuncSetAttribute` 的返回值必须检查；smem 超上限时 launch 静默失败会给出「鬼数据」（`05` 环境、kernel-opt 坑「扫参前先查返回值」）。fp8 当前 `SMEM_BYTES=80128 B`（78.25 KiB）。

**为什么 L1 "基本不变"？** 因为 grid 形状由算法决定（Q 块数 × head × batch），与指令集无关。唯一要动的是 smem 上限和 arch flag——这两项是"硬件参数"，不是"算法参数"。

### L2 算法 / 数据流层（跨卡不变，是最应保留的资产）

所有 dtype 共用同一数学流水（`docs/00` §1）：

```
preprocess: LSE = m + log(l)  (online softmax，Q·K 点积)
            D   = rowsum(dO ∘ O)
main:       for each K/V col-block (1colblock):
              S  = scale·Q·Kᵀ            # recompute，不物化 O(S²) 的 P
              P  = exp(S − LSE)
              dP = dO·Vᵀ
              dS = P ∘ (dP − D)
              dV += Pᵀ·dO
              dQ += scale·dS·K
              dK += scale·dSᵀ·Q
convert:    fp32 累加缓冲 → 目标精度输出
```

**causal 处理**：`ncols = causal ? min(S, m0+BM) : S`，对角块 mask `!(causal && jg > qi)`。这套骨架在 L3 换成任何计算后端时都不变。

**"1colblock" 是什么意思？** 一个 CTA 负责一个 K/V 列块（`BN` 列），沿 Q 方向逐块累加。它决定了：$dK/dV$ 在块内就能攒完，$dQ$ 则要跨块汇总。**换卡不会改变这个数据流**，只会改变"用哪条指令算 GEMM"。

### L3 计算层（★换卡必换）

| 后端 | 入口 | 指令 | 适用 |
|---|---|---|---|
| 标量 golden | `fa_bwd_fp8_onefile.cu` | FFMA + `__half2float` | 任意卡（正确性基线） |
| 张量核 mma | `mma_block<...>` / `mma_e4e4/e5e4/e4e5` | `mma.sync.aligned.m16n8k32`（FP8）/ `m16n8k16`（FP16/BF16） | sm80+（含 sm90 兼容路径） |
| （未做）wgmma | — | `wgmma.mma_async` | sm90a 专属 |
| （未做）tcgen05 | — | Blackwell | sm100+ |

移植时**唯一必须重写的是 `mma_block` 及 `mma_e4e4/e5e4/e4e5` 这几个 `__device__` 函数**：把「warp 级 m16n8k32 + ldmatrix 取片段」换成目标卡的等价张量核原语。其余（S/P/dS 的 epilogue、atomic 累加、scale 折算）保持不动。

- **无张量核的卡**：直接退回标量 golden（同文件里已实现），只改 L4 padding。
- **有但不同的张量核**：重写片段布局（见 L4）与累加器到逻辑行的映射（现有 `g=lane>>2`、`c2=(lane&3)*2`、`q>=2 → +8`）。

> **移植工作量的 80% 在这里**。而且"有张量核"不等于"能直接复用"：不同厂商的 fragment 布局、累加器映射、甚至 MMA 形状都可能不同。先写一个"只做一次 m16n8k16 并打印结果"的冒烟测试，再接主循环。

### L4 存储层（★随 L3 换）

smem 布局按 dtype 不同，是移植最易踩坑处：

| dtype | 关键常量 | 行距 padding | 目的 |
|---|---|---|---|
| fp16 | `BM=64,BN=32,THREADS=128`；smem `98304 B` | 无（Q/K/V 行距 `kHeadDim=128` 个 half=256B） | ptxas 用 `LDS.64/HADD2` 成对读，冲突仅 4.6-way，不 padding |
| bf16 | 同；smem `98560 B` | **`kKVStride = kHeadDim+2`**（K/V 行距 260B） | 消 9.5-way bank conflict（89% 多余 wavefront），main 3.1–4.7× |
| fp8 | `BM=64,BN=32,THREADS=128,WN=2`；smem `80128 B` | `ASLD=+16`(144B)、`KTS/QTS/DSS2` 均 16B 整数倍 | `ldmatrix` 一次读 8 行，行距须为 16B 整数倍并错开 bank |

**规则**：任何用 `ldmatrix` 的 smem 行主序数组，行距都要 +（16B 的整数倍，常用 +8 元素）以消 8 行同 bank；`ldmatrix.x4.trans` 取 B 片段可让硬件顺带转置。fp16/bf16 的 QK 是「lane↔行」标量读，只需满足 `stride_words mod 32 != 0`。

**为什么 fp16 不用 padding、bf16 却要 +2？** 因为两者的读取方式不同：fp16 走 `LDS.64/HADD2` 成对读，硬件本身把冲突摊掉了；bf16 走标量读，9.5-way 冲突必须先靠行距错开。**同一个 bank 冲突问题，在不同 dtype 上要用不同解法**——这就是为什么 L4 必须跟着 L3 一起换。

### L5 精度 / 量化层（半可移植）

| 项 | fp8 实现 | 换卡注意 |
|---|---|---|
| 转换 intrinsic | `__nv_cvt_float_to_fp8` / `__nv_cvt_fp8_to_halfraw`（E4M3/E5M2） | 非 NVIDIA 卡无此 intrinsic，需替换 |
| **忌讳** | `float(fp8)` 在该工具链返回**位模式**而非数值 | 换卡/换编译器后必须用最小复现重新验证 |
| rowwise scale | `quantize_row_kernel`（amax/448 或 /57344） | 数学口径可移植 |
| scale 折算 | `Ap=P·dos`、`dS2=dS·ks`、`dS3=dS·qs`（折叠进 A/B 操作数） | 纯数学，可移植；见 §3 |
| 精度口径 | Q/K/V=E4M3、dO=dS2=dS3=E5M2、Ap=E4M3、P/dS/D/LSE=fp32 | 对齐 TE 2.14 |

**"半可移植"的含义**：`amax/448` 这个公式到哪都一样，但"怎么把 fp32 变成 8 个 bit"是厂商 intrinsic。换卡时**公式照搬、转换函数重写**，再用"已知数值的位模式"验证解码。

### L6 同步层（基本不变）

- `__syncthreads()` 分隔 smem 写/读阶段（现有实现的主循环节拍）。
- dK/dV 跨 Q 块用 `atomicAdd`（非确定性）；dQ 每 CTA 独占行、直接写。
- `__launch_bounds__(THREADS)` 控寄存器；注意**强制提 occupancy 可能 spill 负优化**（kernel-opt 坑）。

---

## 3. 必须照搬的「正确性锚点」（与硬件无关的数学约定）

移植后最容易出错的不是指令，而是下面这些**与硬件无关、但错了就静默错**的约定。每一条都配了"错了会怎样"，方便对号入座：

1. **每进入新 KV tile 必须清 S 累加器**（mma 是累加语义）；O/dQ/dK/dV 才跨块保留。
   *错了会怎样*：第二个 tile 的 S 叠在第一个上，softmax 全乱，但只在 `S>BN` 时暴露。
2. **rowwise scale 沿归约维变化 → 必须折叠操作数**：
   - 5 个 GEMM 里 S/dP 的 scale 不在归约维 → epilogue 乘 `qs·ks` / `dos·vs`；
   - dV/dQ/dK 的 rowwise scale 在归约维上 → 定义 `Ap=P·dos`、`dS2=dS·ks`、`dS3=dS·qs`，B 用**未乘 scale 的原始 fp8**，折算因子退化为「每输出行一个」。
   - 详见 `fa_bwd_fp8_kernels.cuh:12-25` 的记账注释。
   *错了会怎样*：结果整体偏大/偏小（系统性偏差），不是随机噪声。
3. **scale 数组个数要与 smem 布局精确匹配**（P3-4 踩坑：`kNScale` 少算 `3*BM+4*BN`，越界覆盖 `Ps`，症状 S=512 对、S≥1024 dV 错 ~O(1)）。
   *教训*：这类错误只在特定 shape 暴露，**对拍必须覆盖多个 S**。
4. **FP8 转换不能用 `float(fp8)`**；host 侧读 fp8 原始字节要用 `reinterpret_cast<const unsigned char*>`。
5. **causal 短序列也要对拍**（L=64 单 tile 才暴露行号越界，长序列被 softmax 掩盖）。
6. **任何带 `if (i>=tot) return` 的辅助 kernel 必须 grid-stride**（P4 前踩坑）。

### 移植失败的典型症状

| 症状 | 最可能的原因 | 先查哪 |
|---|---|---|
| 结果全 0 / 鬼数据 | smem 超限、launch 静默失败、累加器没清零 | L1、§3.1 |
| 小 S 对、大 S 错 | scale 数组越界、行号越界、S 累加器未清 | §3.2、§3.3 |
| 数值对但慢 3–5× | bank conflict、无流水、occupancy 过低 | L4、L6 |
| 数值系统性偏大/偏小 | scale 折错位置、漏乘 `sa*sb` | §3.2 |
| 解码出奇怪数值 | 用了 `float(fp8)` 或位模式误读 | L5 |

---

## 4. 换卡移植步骤（可执行 checklist）

1. **确认目标卡指令集**：是否支持张量核？FP8？smem 上限？→ 决定 L3 后端。
2. **改 `scripts/run.sh` 的 `ARCH`**（及 gencode；WGMMA/新指令需 `compute_XXa`）。
3. **移植 L3**：重写 `mma_block` + 3 个 dtype 组合的 `mma_*`；无张量核则退回标量 golden。
4. **复核 L4**：按新 warp 片段布局重算 smem 行距；`ldmatrix` 类指令的 bank 规则用 ncu `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 验证。
5. **复核 L1**：`SMEM_BYTES` 是否超新卡上限；`cudaFuncSetAttribute` 返回值必查。
6. **复核 L5**：替换 fp8 转换 intrinsic，用「已知数值位模式」最小复现验证解码。
7. **数值对拍**：用 L0 同一份 npy，跑 `run.sh <file> --dir=<case>`，比 `ref_*/te_*`。
8. **性能对标**：`harness/fa_bwd_bench.py bench` 重测目标卡 FA/TE 基线（若可用），同口径算 TFLOPS 与峰值占比。
9. **ncu 剖析**：确认 bound（DRAM/L1TEX/Compute/occupancy/waves），写入文档。

**验收标准**（建议写进移植记录）：

- 数值：`max_abs(ours vs ref)` 与 H100 上同 dtype 同量级；单/两文件逐位一致。
- 功能：MHA / GQA / MQA / causal / 非 causal / 变长 至少各一个 case 通过。
- 性能：同 shape 的 main kernel 时间与 H100 的差距能归因到具体 bound（而不是"就是慢"）。
- 回归：确定性开关、`--check` 门禁、ncu 脚本都能在新卡跑通。

---

## 5. 当前实现与目标卡的已知差距（移植前先读）

> **重要：本节是 P4-2 阶段的快照**，写于 cp.async/TMA、`--det`、D=256 之前。其中"无 cp.async/TMA""仅 D=128/512""非确定性"三条**已被后续工作改变**；保留原文是为了让移植者看到"当时的起点"，最新状态见 `01`/`01b`/`03` 与 §4.3（catalog）。**不要把本节当作当前能力清单。**

- **head_dim 已部分模板化**：fp16/bf16 用 `BwdTraits<HD,BM>`，fp8 用 `Fp8Cfg<HD,BM,BN>`，现支持 `D=128`（MHA/GQA）与 `D=512`（MLA）；`main.cu` 对不支持的 D 报错。要覆盖 FA2 的 64/96/192/256 需再补对应 `BM/BN` 与 `Fp8Cfg` 实例（fp8 的 `GEMM3/4/5` 还要求 `HD%128==0`）。
- **无 `cp.async` / TMA 流水**：main kernel 是「同步载入→算→同步」的朴素循环，Waves<0.5、1 CTA/SM，属**延迟受限**；目标卡若延迟更高，会更慢，需先补流水。
- **非确定性**：dK/dV 用 `atomicAdd` 跨 Q 块累加。要确定性需换 `dQ_accum` 分块缓冲 + convert（FA2 做法）——这也与「提并行度」的 backlog 是同一件事。
- **fp8 端到端瓶颈是 preprocess**（S=4096：71.0 ms vs main 10.2 ms），LSE 的 O(S²) 点积未分块/向量化。移卡后此瓶颈相对更重（若目标卡标量性能弱）。
- **wgmma/TCGEN05 未做**：sm90 只用 sm80 兼容的 `mma.sync`；换到只有新指令的卡时必须写 wgmma/tcgen05 后端，这是移植的最大工作量。
- **峰值常量是 H100 的**：目标卡数值需替换，否则峰值占比无意义。

**这些差距里，移植时最该先处理哪条？** 按"是否阻塞正确性"排序：

1. smem 上限与 launch 返回值（阻塞：会静默出错）；
2. fp8 解码 intrinsic（阻塞：数值全错）；
3. L3 指令（阻塞：没有它跑不起来）；
4. 流水/occupancy（不阻塞正确性，但决定性能上限）；
5. 峰值常量（只影响报告，不影响运行）。

---

## 6. 参考

- 数学与优化手段：`00-fa-bwd-optimization-catalog.md`
- 各 dtype 实现与 ncu：`01-fp16-bwd-impl.md`、`01b-bf16-bwd-impl.md`、`02-fp8-bwd-design.md`、`03-fp8-bwd-impl.md`
- 数值 / 性能汇总：`04-numerics-and-perf-summary.md`
- 环境 / 踩坑：`../../kernel-opt/agent_skills/kernel-opt.md`、`agent_guide.md`
- 状态与 backlog：`../ROADMAP.md`
