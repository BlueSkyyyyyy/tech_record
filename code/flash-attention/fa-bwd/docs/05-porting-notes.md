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

---

## 1. 抽象层总览

现有实现是「**三段式 kernel 流水 + 一个量化前置**」，host 侧只做 I/O 与计时。按可移植性
从强到弱分成 6 层：

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

**关键判断**：L0–L2、L6 是与硬件解耦的骨架；真正需要为目标卡重写的是 **L3（计算指令）**
以及被它牵连的 **L4（smem 布局）**；L5 的量化**数学口径**可移植，但 fp8 转换 intrinsic
按厂商不同。这也解释了为什么本项目的路线是「先标量 golden（L3=FFMA）→ 再张量核（L3=mma）」：
L3 可替换，是同一套 L0–L2 代码的两个后端。

---

## 2. 逐层接口清单（换卡要改什么）

### L0 数据 / 口径层（跨卡不变）

| 项 | 位置 | 说明 |
|---|---|---|
| 输入/输出 npy | `/home/xieminglin/proj/output/fa-bwd/<case>/` | CPU fp32 保存（fp16/bf16 无损）；`q,k,v,do,o` + `ref_{o,dq,dk,dv}` + `te_*` + `meta.json` |
| 新 shape dump | `harness/fa_bwd_bench.py dump` | 任意实现 load 同一份输入做逐元素比对 |
| FLOPs 口径 | `4·B·S²·H·D`（causal 不折半） | ours/FA/TE 同口径，横向可比 |
| 峰值常量 | `agent_skills/kernel-opt.md` | H100：HBM 3.35 TB/s；FP16/BF16 TC 989；FP8 1978.8 TFLOPS |
| 计时口径 | ours=CUDA event；FA/TE=CUPTI 纯 device | 见 `04` §0 |

> 移植到目标卡时，**只换「峰值常量」与「计时口径换算」**；npy 与对拍脚本不动。

### L1 host / launcher（基本不变）

| 接口 | fp16 / bf16 | fp8 | 换卡动作 |
|---|---|---|---|
| preprocess grid | `(S, H, B)`，block `THREADS=128` | 同 | 不变 |
| main grid | `(ceil(S/BM), H, B)` | 同 | 不变 |
| convert grid | `min(ceil(n/256), 65535)` | 同 | 不变 |
| 动态 smem | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, SMEM_BYTES)` | 同 | 按新卡 smem 上限复核 |
| arch flag | `scripts/run.sh` 的 `ARCH=sm_90` | 同 | **改 `ARCH`**；WGMMA 类指令必须 `-gencode=arch=compute_XXa,code=sm_XXa`（见 kernel-opt 坑） |
| 自测对拍 | `diff_stat`：`max_abs` + `max_rel` | 同 | 不变 |

> `cudaFuncSetAttribute` 的返回值必须检查；smem 超上限时 launch 静默失败会给出「鬼数据」
> （`05` 环境、kernel-opt 坑「扫参前先查返回值」）。fp8 当前 `SMEM_BYTES=80128 B`（78.25 KiB）。

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

**causal 处理**：`ncols = causal ? min(S, m0+BM) : S`，对角块 mask `!(causal && jg > qi)`。
这套骨架在 L3 换成任何计算后端时都不变。

### L3 计算层（★换卡必换）

| 后端 | 入口 | 指令 | 适用 |
|---|---|---|---|
| 标量 golden | `fa_bwd_fp8_onefile.cu` | FFMA + `__half2float` | 任意卡（正确性基线） |
| 张量核 mma | `mma_block<...>` / `mma_e4e4/e5e4/e4e5` | `mma.sync.aligned.m16n8k32`（FP8）/ `m16n8k16`（FP16/BF16） | sm80+（含 sm90 兼容路径） |
| （未做）wgmma | — | `wgmma.mma_async` | sm90a 专属 |
| （未做）tcgen05 | — | Blackwell | sm100+ |

移植时**唯一必须重写的是 `mma_block` 及 `mma_e4e4/e5e4/e4e5` 这几个 `__device__` 函数**：
把「warp 级 m16n8k32 + ldmatrix 取片段」换成目标卡的等价张量核原语。其余（S/P/dS 的
epilogue、atomic 累加、scale 折算）保持不动。

- **无张量核的卡**：直接退回标量 golden（同文件里已实现），只改 L4 padding。
- **有但不同的张量核**：重写片段布局（见 L4）与累加器到逻辑行的映射（现有
  `g=lane>>2`、`c2=(lane&3)*2`、`q>=2 → +8`）。

### L4 存储层（★随 L3 换）

smem 布局按 dtype 不同，是移植最易踩坑处：

| dtype | 关键常量 | 行距 padding | 目的 |
|---|---|---|---|
| fp16 | `BM=64,BN=32,THREADS=128`；smem `98304 B` | 无（Q/K/V 行距 `kHeadDim=128` 个 half=256B） | ptxas 用 `LDS.64/HADD2` 成对读，冲突仅 4.6-way，不 padding |
| bf16 | 同；smem `98560 B` | **`kKVStride = kHeadDim+2`**（K/V 行距 260B） | 消 9.5-way bank conflict（89% 多余 wavefront），main 3.1–4.7× |
| fp8 | `BM=64,BN=32,THREADS=128,WN=2`；smem `80128 B` | `ASLD=+16`(144B)、`KTS/QTS/DSS2` 均 16B 整数倍 | `ldmatrix` 一次读 8 行，行距须为 16B 整数倍并错开 bank |

**规则**：任何用 `ldmatrix` 的 smem 行主序数组，行距都要 +（16B 的整数倍，常用 +8 元素）
以消 8 行同 bank；`ldmatrix.x4.trans` 取 B 片段可让硬件顺带转置。fp16/bf16 的 QK 是
「lane↔行」标量读，只需满足 `stride_words mod 32 != 0`。

### L5 精度 / 量化层（半可移植）

| 项 | fp8 实现 | 换卡注意 |
|---|---|---|
| 转换 intrinsic | `__nv_cvt_float_to_fp8` / `__nv_cvt_fp8_to_halfraw`（E4M3/E5M2） | 非 NVIDIA 卡无此 intrinsic，需替换 |
| **忌讳** | `float(fp8)` 在该工具链返回**位模式**而非数值 | 换卡/换编译器后必须用最小复现重新验证 |
| rowwise scale | `quantize_row_kernel`（amax/448 或 /57344） | 数学口径可移植 |
| scale 折算 | `Ap=P·dos`、`dS2=dS·ks`、`dS3=dS·qs`（折叠进 A/B 操作数） | 纯数学，可移植；见 §3 |
| 精度口径 | Q/K/V=E4M3、dO=dS2=dS3=E5M2、Ap=E4M3、P/dS/D/LSE=fp32 | 对齐 TE 2.14 |

### L6 同步层（基本不变）

- `__syncthreads()` 分隔 smem 写/读阶段（现有实现的主循环节拍）。
- dK/dV 跨 Q 块用 `atomicAdd`（非确定性）；dQ 每 CTA 独占行、直接写。
- `__launch_bounds__(THREADS)` 控寄存器；注意**强制提 occupancy 可能 spill 负优化**
  （kernel-opt 坑）。

---

## 3. 必须照搬的「正确性锚点」（与硬件无关的数学约定）

移植后最容易出错的不是指令，而是下面这些**与硬件无关、但错了就静默错**的约定：

1. **每进入新 KV tile 必须清 S 累加器**（mma 是累加语义）；O/dQ/dK/dV 才跨块保留。
2. **rowwise scale 沿归约维变化 → 必须折叠操作数**：
   - 5 个 GEMM 里 S/dP 的 scale 不在归约维 → epilogue 乘 `qs·ks` / `dos·vs`；
   - dV/dQ/dK 的 rowwise scale 在归约维上 → 定义 `Ap=P·dos`、`dS2=dS·ks`、`dS3=dS·qs`，
     B 用**未乘 scale 的原始 fp8**，折算因子退化为「每输出行一个」。
   - 详见 `fa_bwd_fp8_kernels.cuh:12-25` 的记账注释。
3. **scale 数组个数要与 smem 布局精确匹配**（P3-4 踩坑：`kNScale` 少算 `3*BM+4*BN`，
   越界覆盖 `Ps`，症状 S=512 对、S≥1024 dV 错 ~O(1)）。
4. **FP8 转换不能用 `float(fp8)`**；host 侧读 fp8 原始字节要用
   `reinterpret_cast<const unsigned char*>`。
5. **causal 短序列也要对拍**（L=64 单 tile 才暴露行号越界，长序列被 softmax 掩盖）。
6. **任何带 `if (i>=tot) return` 的辅助 kernel 必须 grid-stride**（P4 前踩坑）。

---

## 4. 换卡移植步骤（可执行 checklist）

1. **确认目标卡指令集**：是否支持张量核？FP8？smem 上限？→ 决定 L3 后端。
2. **改 `scripts/run.sh` 的 `ARCH`**（及 gencode；WGMMA/新指令需 `compute_XXa`）。
3. **移植 L3**：重写 `mma_block` + 3 个 dtype 组合的 `mma_*`；无张量核则退回标量 golden。
4. **复核 L4**：按新 warp 片段布局重算 smem 行距；`ldmatrix` 类指令的 bank 规则
   用 ncu `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 验证。
5. **复核 L1**：`SMEM_BYTES` 是否超新卡上限；`cudaFuncSetAttribute` 返回值必查。
6. **复核 L5**：替换 fp8 转换 intrinsic，用「已知数值位模式」最小复现验证解码。
7. **数值对拍**：用 L0 同一份 npy，跑 `run.sh <file> --dir=<case>`，比 `ref_*/te_*`。
8. **性能对标**：`harness/fa_bwd_bench.py bench` 重测目标卡 FA/TE 基线（若可用），
   同口径算 TFLOPS 与峰值占比。
9. **ncu 剖析**：确认 bound（DRAM/L1TEX/Compute/occupancy/waves），写入文档。

---

## 5. 当前实现与目标卡的已知差距（移植前先读）

- **head_dim 已部分模板化**：fp16/bf16 用 `BwdTraits<HD,BM>`，fp8 用 `Fp8Cfg<HD,BM,BN>`，
  现支持 `D=128`（MHA/GQA）与 `D=512`（MLA）；`main.cu` 对不支持的 D 报错。要覆盖 FA2 的
  64/96/192/256 需再补对应 `BM/BN` 与 `Fp8Cfg` 实例（fp8 的 `GEMM3/4/5` 还要求 `HD%128==0`）。
- **无 `cp.async` / TMA 流水**：main kernel 是「同步载入→算→同步」的朴素循环，
  Waves<0.5、1 CTA/SM，属**延迟受限**；目标卡若延迟更高，会更慢，需先补流水。
- **非确定性**：dK/dV 用 `atomicAdd` 跨 Q 块累加。要确定性需换 `dQ_accum` 分块缓冲 +
  convert（FA2 做法）——这也与「提并行度」的 backlog 是同一件事。
- **fp8 端到端瓶颈是 preprocess**（S=4096：71.0 ms vs main 10.2 ms），LSE 的 O(S²)
  点积未分块/向量化。移卡后此瓶颈相对更重（若目标卡标量性能弱）。
- **wgmma/TCGEN05 未做**：sm90 只用 sm80 兼容的 `mma.sync`；换到只有新指令的卡时
  必须写 wgmma/tcgen05 后端，这是移植的最大工作量。
- **峰值常量是 H100 的**：目标卡数值需替换，否则峰值占比无意义。

---

## 6. 参考

- 数学与优化手段：`00-fa-bwd-optimization-catalog.md`
- 各 dtype 实现与 ncu：`01-fp16-bwd-impl.md`、`01b-bf16-bwd-impl.md`、
  `02-fp8-bwd-design.md`、`03-fp8-bwd-impl.md`
- 数值 / 性能汇总：`04-numerics-and-perf-summary.md`
- 环境 / 踩坑：`../../kernel-opt/agent_skills/kernel-opt.md`、`agent_guide.md`
- 状态与 backlog：`../ROADMAP.md`
