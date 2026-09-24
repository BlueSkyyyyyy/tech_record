# flash-attention 反向（fa-bwd）开发 · 路线图（活文档）

> 目标：借鉴 flash-attention / TransformerEngine 的实现，在**目标 AI 卡**（当前在 H100 sm90 上开发验证）上，
> 把 flash-attention **反向**整理/重写成**单文件**与**两文件**两种形式（fp16 / bf16 / **fp8 最重点**），
> 编译、数值对拍（ref / TE）、ncu 剖析、与 TE 性能对标，并产出分析文档。
> 这是与 `../../kernel-opt` 类似的自驱任务，但**时限更短**（autopilot 轮数少）。

## 交付物（Definition of Deliverable）

1. **代码**（`src/<dtype>/`）
   - `fa_bwd_<dtype>_onefile.cu`：**单文件**自包含（preprocess + main kernel + launcher + 自测 main），
     大量注释说明每个优化手段。
   - `fa_bwd_<dtype>_kernels.cuh` + `fa_bwd_<dtype>_main.cu`：**两文件**版（device 代码 / host+launcher）。
   - 三个 dtype：`fp16`、`bf16`、`fp8`（fp8 用 E4M3/E5M2 + rowwise scaling，参考 TE）。
2. **数值 I/O dump**（`/home/xieminglin/proj/output/fa-bwd/<case>/`）
   - 输入 `q,k,v,do` 与各实现输出 `o,dq,dk,dv`，**CPU npy（fp32 保存，fp16/bf16 无损）** + `meta.json`，
     方便任意实现 load 同一份输入做逐元素比对。
3. **对拍报告**：ref（PyTorch fp32 autograd）vs 我们实现 vs FA vs TE，给出 max abs/rel diff 与容差判断。
4. **性能报告**：CUPTI 纯 device 时间，我们 vs FA2 vs TE；标注峰值占比。
5. **ncu 剖析**：bound 在哪（HBM / L2 / smem / Tensor Core / occupancy / stall），逐项证据。
6. **文档**（`docs/`）：优化手段梳理（已有 `00-...catalog.md`）、每个 dtype 的实现分析、对拍与性能总结、目标卡 porting 说明。

## 环境 / 运行

- 复用 kernel-opt 的实验容器 `kernel_lab`（含 CUDA 13.2 / nvcc / ncu / PyTorch / **flash_attn 2.7.4** / **TE 2.14**）。
- 参考实现探测：`docker exec kernel_lab python .../harness/probe_refs.py`（已通过：FA/TE 反向均与 fp32 ref 吻合）。
- dump + 对拍 + 基准：`harness/fa_bwd_bench.py {dump|bench|all}`（已通过：S=4096 causal，FA ~132 TF、TE ~230 TF）。
- 编译运行自己的 kernel：`scripts/run.sh src/fp16/fa_bwd_fp16_onefile.cu`；剖析：`scripts/ncu.sh ...`。

## 参考坐标

| 资源 | 位置 | 用途 |
|---|---|---|
| FA2 bwd（fp16/bf16, sm80） | `~/github/flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h` 等 | 主要参照（结构清晰、无 TMA 依赖） |
| FA3 bwd（sm90） | `~/github/flash-attention/hopper/flash_bwd_kernel_sm90.h` 等 | 进阶：TMA+wgmma+warp specialization |
| TE fused_attn_bwd（含 FP8） | `code/te-perf/bench_te.py` + 容器内 TE 2.14 | FP8 口径与性能基准 |
| FA 反向参考（python） | `code/flash-attention/{ref_impl.py,bwd_variants_ref.py}` | 数学参考 |
| 本地资料 | `~/github/{cutlass,FlashMLA,...}`, `CUDA_C_Programming_Guide.pdf` | 抄优化手法 |

## 任务清单

状态：`[ ]` · `[~]` · `[x]` · `[-]`。

### P0 基础（已完成）

- [x] 探测 FA / TE 反向接口可用性、与 fp32 ref 对拍（`harness/probe_refs.py`）
- [x] dump/对拍/基准脚手架 `harness/fa_bwd_bench.py`；dump 目录 `/home/xieminglin/proj/output/fa-bwd`
- [x] 优化手段梳理初稿 `docs/00-fa-bwd-optimization-catalog.md`

### P1 fp16 反向（先打通全流程）

- [x] **P1-1** 单文件 `fa_bwd_fp16_onefile.cu`：preprocess(D+LSE) + main(1colblock) + launcher + 自测；编译运行
- [x] **P1-2** 与 ref/FA/TE 数值对拍（读 dump 的输入，比 ref 输出），记录 max diff
      → S=512 / S=4096 均 max_abs ~1.5–2.2e-3，与 FA/TE 同量级
- [x] **P1-3** ncu 剖析：bound = smem 访问（L1/TEX 53.5%、75% 多余 wavefront）+ occupancy（96KB smem→1 CTA/SM）；DRAM 仅 0.21%；与 FA2/TE 性能对比完成
- [x] **P1-4** 两文件版拆分为 `_kernels.cuh` + `_main.cu`，行为一致（逐指标核对无差异）
- [x] **P1-5** `docs/01-fp16-bwd-impl.md`：实现与优化逐条说明

### P2 bf16 反向

- [x] **P2-1** `src/bf16/fa_bwd_bf16_onefile.cu`：以 fp16 单文件为模板做 dtype 参数化
      （`__half`→`__nv_bfloat16`）；编译运行 + 对拍 ref/FA/TE；ncu；
      **附带 bf16 专属优化：K/V smem 行距 +2 padding**（消 9.5-way bank conflict，main 3.1–4.7×）
- [x] **P2-2** bf16 两文件拆分为 `fa_bwd_bf16_kernels.cuh` + `fa_bwd_bf16_main.cu`，行为一致
- [x] **P2-3** bf16 文档（`docs/01b-bf16-bwd-impl.md`：P2-1 + P2-2 两文件一节）
### P3 fp8 反向（最重点）

- [x] **P3-1** `docs/02-fp8-bwd-design.md`：dO/dP/dQKV 的量化与 scaling 布局（对齐 TE 口径）
- [x] **P3-2** 单文件 fp8 反向（golden：E4M3/E5M2 + rowwise scale + fp32 累加，标量）；编译运行
      → `src/fp8/fa_bwd_fp8_onefile.cu`。**说明**：本阶段为正确性优先的标量 golden，
      真实量化但不做 mma；mma.m16n8k32 版留给 P3-4（与 P1/P2 先标量后张量核的路线一致）。
- [x] **P3-3** 对拍：vs fp32 ref 与 vs TE fp8；给出误差统计（3 个 shape）
      → max_abs 全部 ~0.3–0.9，与 TE 同量级；`docs/03-fp8-bwd-impl.md` §2
- [x] **P3-4** ncu + 性能 vs TE fp8；找 bound 并继续优化
      → **张量核版完成**：`src/fp8/fa_bwd_fp8_mma_onefile.cu`（5 个 GEMM 全 mma.m16n8k32）。
      先做 `fa_bwd_fp8_mma_smoke.cu` 验证 4 种 dtype 组合布局（误差 ~1e-6）；再合入反向，
      rowwise scale 通过 `Ap/dS2/dS3` 折叠操作数折算。main 相对 golden **13–20×**，
      数值与 TE 同量级、S=4096 时优于 TE-vs-ref。ncu：bound 已从 smem 冲突变为
      **低 occupancy/并行度**（L1/TEX 76%→19%，No Eligible 91.7%、Waves 0.48、1 CTA/SM）。
      详见 `docs/03-fp8-bwd-impl.md` §7。剩余（pipeline/提 occupancy/dQ 缓冲）转 backlog。
- [x] **P3-5** 两文件版 `fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_main.cu`（以 mma 单文件为源，行为逐指标一致）
      → 数值逐位相同（S=512 2.426/2.975/3.735e-1；S=1024H32 2.400/4.195/3.536e-1；
      S=4096 2.635/2.643/3.216e-1），ncu `Executed Instructions=25,543,552`/128 regs/80.13KB 与单文件一致；
      `docs/03-fp8-bwd-impl.md` §8

### P4 文档 / 汇总

- [x] **P4-1** `docs/04-numerics-and-perf-summary.md`：数值表 + 性能表（我们/FA/TE，各 shape）
      → 汇总 fp16/bf16/fp8 三 dtype、单/两文件形态：对拍表（vs ref/FA/TE）、性能表（TFLOPS+峰值占比）、
      ncu bound 小结；本轮重跑两文件版与 FA/TE 基线，原始输出 `src/fa_bwd_{ours,refbench}_summary.out.txt`。
- [x] **P4-2** `docs/05-porting-notes.md`：目标卡抽象层与移植注意事项
      → 6 层抽象（L0 数据/口径、L1 host、L2 算法数据流、L3 计算、L4 存储、L5 精度量化、L6 同步）+
      逐层接口清单（换卡只换 L3 计算指令与被牵连的 L4 smem 布局）+ 正确性锚点 +
      可执行移植 checklist + 当前实现与目标卡的已知差距（hdim 编译期常量/无流水/非确定性/
      preprocess 瓶颈/无 wgmma-tcgen05）。本轮重跑两文件 fp8 kernel 确认仍可编译运行（数值与
      `04` 表逐位一致），原始输出 `src/fp8/fa_bwd_fp8_main_p42_verify_s512.out.txt`。

### P5 生产形状：GQA / MQA / MLA（用户指定，已 dump 待实现）

- [x] **P5-0** harness 支持 GQA/MQA（`Hkv`）与 MLA（`Dv`），新增 7 个生产形状并 dump（fp16/bf16/fp8）到
  `/home/xieminglin/proj/output/fa-bwd/`；补数值+性能分析到 `docs/04` §7（FA/TE 基线）
- [x] **P5-1** ours 支持 **GQA/MQA**（Q 头共享 KV 头）：fp16 先行，再 bf16/fp8；对拍 dump 的 4 个 GQA/MQA 形状
      → **已完成（第十六轮）**：fp16 单/两文件均支持 `Hkv`（Q 头 `h`→KV 头 `h/(H/Hkv)`），
      4 个形状 dq/dk/dv vs ref 同 fp16 噪声量级（dk/dv 与 FA/TE 同量级或更小）；MHA 回归逐位不变。
      `docs/01-fp16-bwd-impl.md` §8。
- [x] **P5-2** ours 支持 **head_dim=512（MLA 主注意力）**：解决 smem/寄存器容量（K/V 分块变小、Q 常驻等）
      → **已完成（第十七轮）**：head_dim 改为模板参数 `HD` + `BM` 随容量选择（`HD=128→BM=64` 逐位回归；
      `HD=512→BM=16`，smem 135.17KB，1 CTA/SM），单/两文件同步。3 个 MLA case 对拍 ref 全部 fp16 噪声
      （1.3–2.9e-3），ncu bound = smem+bank conflict+低 occupancy；MLA 反向 FA/TE 均不支持，性能数字
      仅 ours（0.21–0.63 TF）。`docs/01-fp16-bwd-impl.md` §9。
- [x] **P5-3** MLA 对拍（只有 fp32 ref 可对）与性能数字；对标 FlashMLA 思路
      → fp16（P5-2/§9）、bf16（第二十轮）、fp8（第二十一轮）三 dtype 的 MLA 对拍与 ours 性能数字
      均已给出；FlashMLA 式分块/流水优化（降 smem 冲 2 CTA/SM、张量核、持久化）留 backlog
- [x] **P5-4** fp8 GQA/MQA 对拍（vs TE FP8）与性能
      → **已完成（第十八轮）**：fp8 反向（单/两文件）支持 GQA/MQA（`Hkv`，映射 `hkv=h/(H/Hkv)`）；
      4 个 shape 对拍 ref/TE 同量级、性能 ~7–9% TE FP8。`docs/03` §13、`docs/04` §7.4。

### 可选

- [~] SM90 TMA+wgmma 版本（对标 FA3）：**wgmma 部分已完成**（O9a/O9b/O9b-2/O17/O18，
  fp16/bf16/fp8，且 **O22/O23 已把 Hopper 路径默认化**）；**TMA 部分已落地三种 dtype 的 LSE**
  （**O30 fp16** 4D-TMA + 2×K=64 chunk，**O31 bf16** dtype 参数化，**O32 fp8** 单 chunk/UINT8，
  均 1.06–1.36×、数值逐位不变）；**主 kernel 的 Q/K/V/dO TMA 化：fp16 已完成（O33，第七十四轮）**
  ——**逐 atom TMA 复现 SW128 交织布局**（描述符零改动），`--maintma`、main **1.04×**、`red`
  逐字节不变、端到端为 FA3 的 3.77×（见 `docs/01` §14s）。**剩余**：bf16/fp8 的对应 dtype
  参数化（fp8 SW128 的 `k/16` atom 下标）与 BN=64 的 `wgmma2` 几何；fp8 尚有 dS3-Ap smem 复用。见 backlog。
- [ ] 变长（cu_seqlens / varlen）覆盖

## 每项的 Definition of Done

- [ ] 代码能编译、运行、**结果正确**（与 ref 在对应精度容差内；fp16 ~1e-3、bf16 ~1e-2、fp8 另定）
- [ ] 原始输出留存（`*.out.txt` / logs）；dump 落在 `/home/xieminglin/proj/output/fa-bwd`
- [ ] 有 ncu 证据说明 bound；有性能数字与对标
- [ ] 文档/路线图更新，commit & push

## 工作循环（agent 自驱）

1. 读本文「下一步」与 `docs/00-fa-bwd-optimization-catalog.md`。
2. 写/改 `src/<dtype>/*`，`scripts/run.sh` 编译运行，`scripts/ncu.sh` 剖析。
3. dump 输入/输出到 output，跑对拍脚本比对 ref/FA/TE。
4. 更新文档与本文件，commit & push。

## 无人值守（autopilot）

`scripts/autopilot.sh {start|status|stop}`：循环启动 opencode 无头会话，每轮完成一个 P 项。
**时限更短**：默认 `MAX_ROUNDS=14`。agent 不要自行创建 `AUTOPILOT_STOP`；只有用户 stop 才停。
「下一步」做完后可自行补充具体任务（如 GQA/变长/更深优化），但保持聚焦。

## 阻塞

- **fp16/bf16 main 的「放大 BM 到 256」被寄存器文件卡死（O17b，第五十四轮）。**
  512 线程 @1 CTA/SM 时每线程寄存器上限 = `65536/512 = 128`，而本算法必须持有
  dQ 寄存器累加器 `dqacc[2][8][4]=64` + GEMM1/2 两条 wgmma 累加器（各 32，wait0 后同时读）
  = 128，必然 spill；spill 的 local 流量（uncoalesced、占 L2 ~48%）比省下的 red 更贵。
  实测 red 精确减半（51.9M→26.7M）但 main S4096 反而 0.83×。**除非先把 dQ 累加器
  「搬出寄存器」或把 Q/dO+P/dS 的驻留量压下来，否则 BM>128 的 wgmma 反向在本卡不可行。**
  详见 `docs/01` §14k、原始输出 `src/fp16/fa_bwd_fp16_o17b_sweep.out.txt`。
- **fp8 GEMM3/4/5 的 Hopper `wgmma`（O9c-2b，MN-major 描述符转置读）——硬件层面不成立。**
  查证 CUTLASS `include/cute/arch/mma_sm90_gmma.hpp`：**所有 fp8 wgmma 变体只有 `_SS_TN`
  （A/B 均 K-major），没有 `.trans_a/.trans_b` 立即数**（asm 尾部是 `p, scale_D, scaleA, scaleB`，
  对照 fp16 是 `p,1,1,tnspA,tnspB`）。即 fp16 O9b-2 靠 `tnsp=1` 做的「K-major tile 转置读」
  在 fp8 ISA 上不存在；只能物理存转置布局，而 fp8 SW128 atom（8 行×128 fp8）要求归约维≥128
  ⇒ `BM=BN=128`、smem >110KB、1–2 CTA/SM，且 GEMM5 与 GEMM3/4 需两种主序。加之 fp16 O9b-2
  已实测 GEMM3/4/5 上 wgmma **中性偏负**（墙在 L2 原子/occupancy，不在 GEMM 指令），
  **决定不再尝试**；fp8 主 kernel 的 L1/TEX 墙改走非转置手段（fold 向量化 / TMA operand /
  dK/dV 去原子）。详见 `docs/03` §23.6。

## 当前进度

- 2026-09-22：脚手架就绪。`probe_refs.py` 验证 FA2.7.4 / TE2.14 反向与 fp32 ref 吻合；
  `fa_bwd_bench.py` dump（CPU npy）+ CUPTI 基准可用。基线（fp16, causal, S=4096,H=16,B=1）：
  FA ~132 TFLOPS、TE ~230 TFLOPS。优化手段梳理初稿完成。
- 2026-09-22（第二轮）：**P1-1/P1-2/P1-3/P1-5 完成**。
  - 单文件 `src/fp16/fa_bwd_fp16_onefile.cu` 跑通：preprocess(LSE+D) + main(1colblock,recompute P) + convert。
  - 数值对拍 vs fp32 ref：S=512 时 dq/dk/dv max_abs 1.67/1.68/1.90e-3；S=4096 时 1.50/1.57/2.23e-3，
    与 FA/TE 同量级（fp16 噪声），无系统误差。原始输出见 `src/fp16/*.out.txt`。
  - ncu（main, S=512）：DRAM 0.21%、L2 0.67%、Compute 8.45%、**L1/TEX 53.5%（75% 多余 wavefront）**、
    occupancy 6.25%（96KB smem 卡 1 CTA/SM）、waves 0.48。bound = **smem 访问 + 低 occupancy**，非带宽/算力。
  - 性能：ours 0.60 TF(S512) / 0.99 TF(S4096)，仅为 FA 的 ~0.8%、TE 的 ~0.4%（正确性优先的标量实现）。
  - 文档 `docs/01-fp16-bwd-impl.md` 完成；修复 `scripts/lab.sh` 相对路径、`scripts/ncu.sh` 的 `--` 分隔。
- 2026-09-22（第三轮）：**P1-4 完成（fp16 全部收尾）**。
  - 拆成两文件：`src/fp16/fa_bwd_fp16_kernels.cuh`（device：preprocess/main/convert + 常量）
    + `src/fp16/fa_bwd_fp16_main.cu`（host：npy 读取/launcher/自测）；kernel 代码逐字未改。
  - 逐指标核对与单文件**无差异**：S=512 dq/dk/dv max_abs 1.671/1.680/1.899e-3；
    S=4096 1.499/1.572/2.225e-3；ncu DRAM 0.21% / L1TEX 53.52% / Compute 8.45% / occ 6.25%。
    原始输出见 `src/fp16/fa_bwd_fp16_main_{s512,s4096,ncu_main}.out.txt`。
   - `docs/01-fp16-bwd-impl.md` 增加「两文件版（P1-4）」一节。
- 2026-09-22（第四轮）：**P2-1 完成（bf16 单文件 + 一处 bf16 专属优化）**。
  - `src/bf16/fa_bwd_bf16_onefile.cu`：fp16 单文件 dtype 参数化（算法/线程映射逐字一致）。
  - 对拍 vs fp32 ref：S=512 dq/dk/dv max_abs 6.89/8.11/13.65e-3；S=4096 8.90/8.08/14.94e-3，
    与 FA/TE 同量级（bf16 噪声 ~1e-2），dq/dk 略优于 FA/TE，无系统误差。
  - **发现**：同款代码 bf16 的 main 比 fp16 慢 2.9×（S4096 main 197 vs 68 ms）——
    ptxas 对 `__bfloat162float` 走「LDS.U16 标量取半字 + SHF.L」，K/V 行（256B 行距，
    128B bank 周期整数倍）出现 **9.5-way bank conflict、89% 多余 wavefront、L1/TEX 78.5%**。
  - **优化**：K/V smem 行距 +2 元素（256→260B，`65 mod 32=1`）⇒ 冲突归零：
    S=512 main 5.76→1.88 ms（3.1×），S=4096 197.1→42.2 ms（4.7×）；L1/TEX 78.5%→24.9%。
    现在 bf16 main 已快过 fp16 同款；数值逐位一致。
  - ncu（padding 后）：DRAM 0.26%、L1TEX 24.9%、Compute 13.5%、occ 6.25%（1 CTA/SM）、
    waves 0.48、No Eligible 78%。**bound = 延迟/并行度**（fixed-latency stall 37.3%），
    不再是 smem 访问；下一步靠降 smem/提 occupancy + 张量核。
  - 性能：ours 0.69 TF(S512) / 1.23 TF(S4096)，约 FA 的 0.5–2%、TE 的 0.5%（标量实现）。
    S=4096 时 **preprocess 69.4ms > main 42.2ms**（LSE 重算 O(S²) 未分块）成新瓶颈。
   - 文档 `docs/01b-bf16-bwd-impl.md`；原始输出 `src/bf16/*.out.txt`。
- 2026-09-22（第五轮）：**P2-2/P2-3 完成（bf16 全部收尾）**。
  - 拆成两文件：`src/bf16/fa_bwd_bf16_kernels.cuh`（device：常量含 `kKVStride` padding +
    preprocess/main/convert）+ `fa_bwd_bf16_main.cu`（host：npy/launcher/自测）；kernel 代码逐字未改。
  - 逐指标核对与单文件**无差异**：S=512 dq/dk/dv max_abs 6.892/8.110/13.65e-3；
    S=4096 8.895/8.078/14.94e-3；ncu DRAM 0.25% / L1TEX 24.89% / Compute 13.52% / occ 6.25% /
    waves 0.48 / regs 52 / smem 98.56KB / Executed Instructions 247,182,666（与单文件完全相同）。
    原始输出见 `src/bf16/fa_bwd_bf16_main_{s512,s4096,ncu_main}.out.txt`。
  - `docs/01b-bf16-bwd-impl.md` 增加「两文件版（P2-2）」一节。

- 2026-09-22（第六轮）：**P3-1/P3-2/P3-3 完成（fp8 单文件 golden 打通）**。
  - 设计 `docs/02-fp8-bwd-design.md`：Q/K/V=E4M3 rowwise、dO/dP=E5M2 rowwise、P=E4M3(scale=1)、
    dS/LSE/D/累加 fp32；相对 TE 的差异（本版 dS/输出保留 fp32）与误差来源逐条写清。
  - 实现 `src/fp8/fa_bwd_fp8_onefile.cu`（四段式：quantize_row + preprocess + main + convert）：
    真实做 FP8 量化/反量化（`__nv_cvt_float_to_fp8`/`__nv_cvt_fp8_to_halfraw`），fp32 标量累加
    模拟张量核。**关键坑**：FP8 转换不能用 `float(fp8)`（该工具链返回位模式）。
  - `harness/fa_bwd_bench.py` 扩展 `--dtype fp8`：dump FP8 case（TE FP8 参考 `te_*`）+ TE FP8 bench。
  - 对拍（max_abs）：S=512 ours-vs-ref 4.8e-1/6.4e-1/3.3e-1，TE-vs-ref 4.9e-1/4.0e-1/5.9e-1；
    S=4096 ours-vs-ref 4.6e-1/5.0e-1/3.8e-1，TE-vs-ref 3.8e-1/3.7e-1/6.7e-1。ours-vs-TE 同量级
    ⇒ 与 TE 口径一致，无系统误差（ref 梯度 amax 3–6，相对量级 ~8–20%）。
  - ncu（main, S=512）：DRAM 0.06%、L2 0.26%、Compute 4.98%、**L1/TEX 75.96%**（90% 多余
    wavefront）、occupancy 6.25%（68KB smem 卡 1 CTA/SM）、Waves 0.32、**69% stall = MIO scoreboard**。
    bound = **smem 访问（bank conflict + FP8 解码）+ 低 occupancy**。
  - 性能（ours total / TE FP8 / FP8 峰值占比）：S=512 7.17ms·0.30TF / 0.072ms·29.8TF；
    S=1024H32 37.5ms·0.46TF / 0.136ms·126TF；S=4096 265.9ms·0.52TF / 0.451ms·304TF（15.4%）。
    golden 约 TE 的 ~0.3%、峰值的 ~0.02%；mma 优化是 P3-4。
  - 文档 `docs/03-fp8-bwd-impl.md`；原始输出 `src/fp8/*.out.txt`（含 ncu/refbench）。

- 2026-09-22（第七轮）：**P3-4 完成（fp8 张量核版，main 13–20×）**。
  - 先写 `src/fp8/fa_bwd_fp8_mma_smoke.cu`（P3-4a）：用最小 GEMM 验证 `mma.m16n8k32` 的
    片段/`ldmatrix`/rowwise 折回，覆盖 E4M3×E4M3、E5M2×E4M3、E4M3×E5M2、E5M2×E5M2
    四种组合，误差 ~1e-6（fp32 舍入）。
  - 实现 `src/fp8/fa_bwd_fp8_mma_onefile.cu`：5 个 GEMM 全部 mma；因 rowwise scale 沿归约维
    变化，引入折叠操作数 `Ap=P·dos`、`dS2=dS·ks`、`dS3=dS·qs`（B 用未乘 scale 的原始 fp8），
    折算因子退化为每输出行一个；`dP` 不再量化（只进 fp32 的 dS）。smem 行距 padding（144/48/80）。
  - **踩坑**：scale 数组个数数错（`kNScale` 应为 `3*BM+4*BN`），`sds2` 越界覆盖 `Ps`，
    症状是 S=512 对、S≥1024 dV 错 ~O(1)。用「dV 内重算 P 比对 `Ps`」计数器定位。
  - 对拍（max_abs, ours-vs-ref，causal）：S=512 2.43/2.98/3.74e-1；S=1024H32 2.40/4.20/3.54e-1；
    S=4096 2.64/2.64/3.22e-1。与 TE 同量级，S=4096 时 ours-vs-ref 优于 TE-vs-ref。
  - ncu（main, S=512）：DRAM 0.92%、**L1/TEX 19.15%**（golden 75.96%）、Compute 4.76%、
    occupancy 6.25%（128 regs/80KB smem→1 CTA/SM）、Waves 0.48、No Eligible 91.7%。
    **bound 已从 smem 冲突变为低 occupancy/延迟受限**。
  - 性能：main golden→mma：S=512 5.90→**0.452ms（13.1×）**、S=1024H32 29.16→**1.802ms（16.2×）**、
    S=4096 198.48→**10.188ms（19.5×）**；端到端 total 1.70/10.78/80.55ms（preprocess 成新瓶颈）。
    main 相对 FP8 峰值 0.24/0.48/0.68%，相对 TE FP8 main 约 4–16%。
  - 文档 `docs/03-fp8-bwd-impl.md` §7；原始输出 `src/fp8/fa_bwd_fp8_mma_*`。

- 2026-09-22（第八轮）：**P3-5 完成（fp8 两文件拆分，fp8 三种形态全部收尾）**。
  - 拆成两文件：`src/fp8/fa_bwd_fp8_kernels.cuh`（device：常量/smem 布局/fp8 转换/mma+ldmatrix 封装
    /quantize_row/preprocess/main/convert）+ `fa_bwd_fp8_main.cu`（host：npy/launcher/自测）；
    以 **mma 单文件**为源，device 代码逐字未改。
  - 逐指标核对与单文件**无差异**：S=512 dq/dk/dv max_abs 2.426/2.975/3.735e-1；
    S=1024H32 2.400/4.195/3.536e-1；S=4096 2.635/2.643/3.216e-1；main 0.454/1.814/10.164 ms。
    ncu 与单文件逐项相同（DRAM 0.93% / L1TEX 19.00% / Compute 4.75% / occ 6.25% / Waves 0.48 /
    128 regs / 80.13KB / Executed Instructions 25,543,552）。
    原始输出见 `src/fp8/fa_bwd_fp8_main_{s512,s1024h32,s4096,ncu_main}.out.txt`。
  - `docs/03-fp8-bwd-impl.md` §8 记录拆分与核对。

- 2026-09-22（第九轮）：**P4-1 完成（数值 + 性能汇总文档）**。
  - 新增 `docs/04-numerics-and-perf-summary.md`：三 dtype（fp16/bf16/fp8）× 单/两文件形态的
    对拍表（vs fp32 ref / FA2.7.4 / TE2.14）、性能表（TFLOPS + 峰值占比）、ncu bound 小结、复现命令。
  - 本轮**重跑实测**并留档：`src/fa_bwd_ours_summary.out.txt`（两文件版 fp16/bf16/fp8 三 shape
    计时+对拍）、`src/fa_bwd_refbench_summary.out.txt`（FA/TE CUPTI 基线），并用 `dump` 补齐
    fp8 三 shape 的 TE-vs-ref 数值。
  - 关键数字：fp16 S=4096 ours 1.50/1.57/2.23e-3（FA 1.88/1.73/1.97e-3）；
    bf16 S=4096 ours 8.90/8.08/14.94e-3（FA 14.4/13.3/16.3e-3）；
    fp8 S=4096 ours-vs-ref 2.64/2.64/3.22e-1 **优于 TE-vs-ref** 3.76/3.69/6.69e-1。
    性能：fp16 ours 0.99 TF vs FA 129.9 / TE 230.3；bf16 ours 1.23 TF vs FA 131.2 / TE 232.2；
    fp8 ours 1.70 TF vs TE 302.5（峰值 1978.8）。三 dtype 均非带宽/算力 bound，
    墙是**低 occupancy + 并行度**、端到端墙是 **preprocess**。

- 2026-09-22（第十轮）：**P4-2 完成（目标卡移植说明；P 项全部收口）**。
   - 新增 `docs/05-porting-notes.md`：把 fp16/bf16/fp8 三套实现切为 6 个可替换层
     （L0 数据/口径、L1 host/launcher、L2 算法数据流、L3 计算、L4 存储、L5 精度量化、L6 同步），
     给出逐层接口清单：**换卡只需换 L3 的计算指令（`mma_block`/`mma_e4e4/e5e4/e4e5` 或退回标量）
     与被牵连的 L4 smem 行距**，L0–L2/L6 骨架复用。
   - 含「正确性锚点」（累加器清零、rowwise scale 折叠、scale 数组计数、fp8 转换坑、短序列对拍、
     grid-stride）、可执行移植 checklist（9 步）、以及当前实现与目标卡的差距
     （`kHeadDim=128` 编译期常量 / 无 cp.async 流水 / 非确定性 atomic / preprocess 瓶颈 /
     无 wgmma-tcgen05 后端 / 峰值常量是 H100 的）。
   - 本轮重跑两文件 fp8 kernel（`src/fp8/fa_bwd_fp8_main.cu`，S=512）确认仍可编译运行：
     `smem=80128 B`、main 0.454 ms、dq/dk/dv vs ref 2.426/2.975/3.735e-1（与 `04` 表逐位一致）。
     原始输出 `src/fp8/fa_bwd_fp8_main_p42_verify_s512.out.txt`。

- 2026-09-22（第十一轮）：**O1 完成（preprocess 的 mma 分块 LSE，端到端 7×）**。
  - 旧 preprocess（每 (s,h) 行一个 block、标量扫 K）在 S=4096 时 70.8ms >> main 10.2ms。
  - 新增 `lse_mma_kernel`：`mma.m16n8k32` E4M3×E4M3 分块 Q·Kᵀ（每 CTA 64 行 × 64 列、4 warp），
    fp32 累加器直接 online-softmax，LSE 的行 max/sum 在 warp 内按 lane 组 `shfl_xor` 归约；
    另把 `D=rowsum(dO∘O)` 拆成独立 `delta_kernel`。两文件版与 mma 单文件同步修改、device 代码逐字一致。
  - **数值与 P3-5 逐位相同**（S=512 2.426/2.975/3.735e-1；S=1024H32 2.400/4.195/3.536e-1；
    S=4096 2.635/2.643/3.216e-1），确认只换算法数据流、未改数学口径。
  - **性能**：preprocess 1.1991→0.0850ms（14.1×）/ 8.8033→0.2162ms（40.7×）/
    70.7674→**1.1370ms（62.2×）**；total 1.71→0.60 / 10.72→2.21 / 80.86→**11.47ms（7.05×，11.98 TF）**。
    同 session TE FP8 基线 0.0723/0.1381/0.4537ms ⇒ 端到端 ours/TE 8.4×/16.0×/25.3×
    （P3-5 时 23.5×/77.7×/178×）。S=4096 端到端瓶颈回落 main（9.92ms，占 87%）。
  - ncu（lse, S=4096）：DRAM 0.46% / Compute 39.4% / L1TEX 20.6% / L2 5.35% / occ 29.5%
    （理论 43.8%，被 72 regs 卡）/ Waves 1.11（1 满波+100 尾 block）/ No Eligible 42.1%、
    long_scoreboard 主导 ⇒ **bound = 延迟/并行度 + 尾波**。S=512 时 grid=128<132 SM、Waves 0.14。
  - 原始输出 `src/fp8/fa_bwd_fp8_main_o1_{s512,s1024h32,s4096}.out.txt`、
    `..._mma_onefile_o1_s512.out.txt`、`..._o1_ncu_{lse_s512,lse_s4096,delta_s4096}.out.txt`、
     `..._o1_tebench.out.txt`。文档 `docs/03-fp8-bwd-impl.md` §9。

- 2026-09-22（第十二轮）：**O2 完成（fp8 main 降 smem 提 occupancy，3 CTA/SM）**。
  - 发现 per-tile 的 `dS3`（GEMM4 的 A）与 `Ap`（GEMM3 的 A）各只活一小段：`dS3` 可放进
    `Ks`（只用于 GEMM1）、`Ap` 可放进 `Vs`（只用于 GEMM2），两处写入都在对应 GEMM 之后的
    sync 之后，天然无竞争。smem 80.13→**75.01 KB**；驱动自动选 233.47 KB carveout。
  - ncu（main, S=4096）：Block Limit Shared Mem 2→**3**、theoretical occ 12.5%→**18.75%**、
    achieved 11.8%→16.85%、active warps/sched 1.91→2.69、No Eligible 82.83%→79.64%、
    Duration 10.27→**8.85 ms**；Waves 3.88→2.59（尾波 233/396）。
  - 性能：main S=1024H32 1.814→**1.526 ms（1.19×）**、S=4096 9.924→**8.562 ms（1.16×）**、
    S=512 0.451→0.461 ms（**不变**，grid=128<132 SM 是 grid-bound）；total 11.47→**10.07 ms**。
    数值与 P3-5/O1 逐位相同；单文件 `fa_bwd_fp8_mma_onefile.cu` 与两文件同步。
  - 原始输出 `src/fp8/fa_bwd_fp8_main_o2_{s512,s1024h32,s4096,ncu_main_s512,ncu_main_s4096,tebench}.out.txt`、
    `src/fp8/fa_bwd_fp8_mma_onefile_o2_{s512,s1024h32,s4096}.out.txt`；文档 `docs/03-fp8-bwd-impl.md` §10。

- 2026-09-22（第十三轮）：**O3 完成（fp8 main K/V 向量化 + 寄存器预取流水，main 1.15–1.18×）**。
  - 把 per-tile K/V 的「逐字节标量全局读→sync→算」改成**向量化 4B 读 + 寄存器双缓冲预取**
    （新增 `kv_prefetch`/`kv_commit`；每线程预取 `KVU=8` 个 `uint32`，落盘时同写 Kt 转置）。
    关键约束：**不加 smem**（O2 后 3 CTA/SM 已用满 carveout，加双缓冲会掉回 2 CTA），
    故用寄存器（128→**168**，无 spill，`65536/(3×128)=170` 仍 3 CTA/SM）；barrier 数不变。
  - **数值与 P3-5/O1/O2 逐位相同**：S=512 2.426/2.975/3.735e-1；S=1024H32 2.400/4.195/3.536e-1；
    S=4096 2.635/2.643/3.216e-1。单/两文件同步修改、device 代码逐字一致（差异 <1% 噪声）。
  - **性能**：main S=512 0.4655→**0.3934ms（1.18×）**、S=1024H32 1.5255→**1.3231ms（1.15×）**、
    S=4096 8.6697→**7.5092ms（1.15×）**；total 0.6214→0.5402 / 1.9214→1.6978 / 10.086→**9.060ms**。
    main-only 5.46/12.98/18.30 TF（峰值 0.28/0.66/0.93%）；同 session TE 0.0723/0.1377/0.4513ms
    ⇒ main ours/TE 18.4%/10.4%/6.0%，端到端 13.4%/8.1%/5.0%。
  - **消融归因**（额外编「只向量化、不流水」版）：向量化贡献 S512/S4096 = 1.16×/1.13×，
    寄存器预取只再贡献 ~1.02–1.03×。**收益主要来自把标量 K/V 读改成 4B 向量化读。**
  - ncu（main, S=4096）：Duration 8.85→**7.57ms**、occ 16.85→16.71%（仍 3 CTA/SM）、
    waves 2.59、L1TEX 51.99→65.59%。**`long_scoreboard` 2.94→2.21（全局延迟被压下）**，
    新主导变为 **barrier 3.34→5.13（5 个 `__syncthreads`/tile）+ short_scoreboard 3.61→4.12**
    （smem→mma 的 `ldmatrix` 依赖）⇒ bound = **CTA barrier + smem 依赖**。
  - 原始输出 `src/fp8/fa_bwd_fp8_{main,mma_onefile}_o3_*.out.txt`、
    `..._o3_ncu_main_{s512,s4096}.out.txt`、`..._o3_stall_s4096.out.txt`、
    `..._o2base_stall_s4096.out.txt`、`..._o3_ablation_{s512,s4096}.out.txt`、
    `..._o3_tebench.out.txt`。文档 `docs/03-fp8-bwd-impl.md` §11。

- 2026-09-22（第十四轮）：**O4a 完成（消 barrier + fold 全线程并行，main 1.21–1.54×）**。
   - 三处**逐位等价**改动：① 删 GEMM1/GEMM2 之间的 `__syncthreads`——GEMM2 只读 `dOs/Vs`，
     其 epilogue 读回的 `Ps[r*BN+c]` 正是**本线程** GEMM1 刚写的同一地址（两次 `mma_block`
     的 `(wm,wn)` 与累加器映射一致），无线程间依赖；② 合并 prologue 的两处 barrier（Q/dO
     与 tile0 K/V 写在互不重叠 smem）；③ fold 改用**全 128 线程**（原仅 32/64 线程空等）：
     `Ap/dS3` 每 warp 8 行×4 lane、`dS2` 每 warp 16 行×2 lane，`__shfl_xor_sync` 归约 amax
     （`fmaxf` 可交换结合 ⇒ 值逐位相同）。主 kernel `__syncthreads` **7→5**。
   - **数值与 O3 逐位相同**（S=512 2.426/2.975/3.735e-1；S=1024H32 2.400/4.195/3.536e-1；
     S=4096 2.635/2.643/3.216e-1；vs TE 也逐位不变）。单/两文件 main kernel 函数体逐字相同。
   - **性能**：main S=512 0.3947→**0.2566ms（1.54×）**、S=1024H32 1.3202→**1.0642（1.24×）**、
     S=4096 7.7391→**6.4192（1.21×）**；端到端 total 0.5403→**0.4030**、1.7278→**1.4408**、
     9.0992→**7.8488ms（17.51 TF）**。main-only 8.37/16.14/21.41 TF（峰值 0.42/0.82/1.08%）；
     同 session TE 0.0723/0.1376/0.4537 ⇒ main ours/TE **28.2%/12.9%/7.1%**。
   - ncu（main, S=4096）：Duration 7.57→**6.55ms**、**barrier 5.13→0.46（基本清零）**、
     short_scoreboard 4.12→**5.40（新主导）**、long_scoreboard 2.21→3.35、
     L1TEX 65.59→**78.51%**、L2 53.67→62.01%、Compute 13.79→15.95%、occ 16.71→17.02%（仍 3 CTA/SM）、
     Waves 2.59。S=512：Duration 406→**272µs**、barrier 0.33、achieved occ 6.25%（grid-bound）。
     **新墙 = short_scoreboard（smem→mma）+ L1/TEX**；下一步 O4b（`ldmatrix.trans` 消 `Kt/Qt/dOt`）
     或 O2b（小 S grid / S=4096 尾波）。
   - 原始输出 `src/fp8/fa_bwd_fp8_main_o4a_{s512,s1024h32,s4096,ncu_main_s4096,ncu_main_s512,stall_s4096,tebench}.out.txt`、
     `src/fp8/fa_bwd_fp8_mma_onefile_o4a_{s512,s4096}.out.txt`；文档 `docs/03-fp8-bwd-impl.md` §12。

- 2026-09-22（第十五轮）：**P5-0 完成（GQA/MQA/MLA 形状接入 + 分析）**。
  - `harness/fa_bwd_bench.py` 支持 `kv=`（GQA/MQA 的 KV 头数）与 `Dv=`（MLA 的 v 维），
    ref 走 fp32 autograd（GQA 广播 KV 头）、新增 TE FP8 反向路径（E4M3/E5M2 rowwise）。
  - 新增 7 个生产形状（`REQUESTED_SHAPES`），dump 到 `/home/xieminglin/proj/output/fa-bwd/`
    共 **21 个 case（fp16/bf16/fp8）**：4 个 GQA/MQA（d=128）+ 3 个 MLA（d=512）。
  - 实测：GQA/MQA（d=128）FA≈154–187 TF、TE≈240–279 TF（fp16/bf16），TE FP8≈170–189 TF；
    **GQA/MQA 数值 FA/TE 均与 ref 同量级**（fp16 ~1e-3、bf16 ~1e-2、fp8 O(1)）。
  - **MLA（head_dim=512）FA 与 TE 反向均不支持**（FA 限 head_dim≤256；TE 训练 bwd 限 256），
    只有 fp32 ref；需 ours 支持后才能给 MLA 性能数字。
  - 分析写入 `docs/04-numerics-and-perf-summary.md` §7；原始输出 `src/fa_bwd_bench_requested.out.txt`。

- 2026-09-22（第十六轮）：**P5-1 完成（fp16 反向支持 GQA/MQA，单/两文件）**。
  - 映射口径与 `ref_attn` 的 `repeat_interleave` 对齐：第 `h` 个 Q 头用 KV 头 `h/(H/Hkv)`。
    单/两文件同步改：`preprocess_kernel`/`fa_bwd_fp16_kernel` 增加 `int Hkv` 入参、K/V 行索引用
    `((b*S+j)*Hkv+hkv)*D`；`dk/dv`（及累加缓冲）按 `B*S*Hkv*D` 分配，`convert_kernel` 收 `n_q/n_kv`；
    host 从 `k.npy` shape[2] 读 `Hkv`（`Hkv==H` 时逐式退化，MHA 行为不变）。
  - **对拍（ours-vs-ref，fp16 causal）**：h32kv4 2.134/3.078/3.963e-3；h40kv8 1.580/2.380/3.999e-3；
    h64kv4 1.974/5.704/4.938e-3；h64kv1(MQA) 1.780/7.586/7.517e-3——均 fp16 噪声量级，
    **dk/dv 与 FA/TE 同量级或更小**，无系统误差；MHA 回归（S512 1.671/1.680/1.899e-3，
    S4096 1.499/1.572/2.225e-3）与 P1 记录逐位一致。
  - ncu（main, h32kv4 S1024）：DRAM 0.07% / **L1TEX 74.96%** / L2 0.97% / Compute 15.08% /
    occ 11.5%（理论 12.5%，1 CTA/SM, Block Limit Shared Mem 2）/ No Eligible 78.16% / Waves 1.94 /
    54 regs ⇒ bound 与 MHA 标量版一致：**smem bank conflict + 低 occupancy**。
  - 性能（CUPTI）：ours total 19.32/23.38/34.85/34.91 ms（0.89–0.99 TF，峰值 989 的 ~0.1%）；
    同 shape FA 155–188 TF、TE 240–279 TF（`fa_bwd_bench.py bench --requested --dtype fp16`）。
  - 原始输出 `src/fp16/fa_bwd_fp16_main_p51_gqa.out.txt`、`..._onefile_p51_gqa.out.txt`、
    `..._p51_ncu_gqa_kv4.out.txt`、`src/fa_bwd_bench_requested_fp16_p51.out.txt`；
    文档 `docs/01-fp16-bwd-impl.md` §8。

- 2026-09-22（第十七轮）：**P5-2 完成（fp16 反向支持 MLA head_dim=512，单/两文件）**。
  - `fa_bwd_fp16` 的 head_dim 改为模板参数：新增 `BwdTraits<HD,BM>`（`WM_ROWS/WN_ROWS/NCH=HD/32`、
    `smem_bytes`），`fa_bwd_fp16_kernel<HD,BM>`；`preprocess_kernel` 增加运行时 `HD`；三处 head-dim
    分段循环 `kk<4`→`kk<NCH`、`acc[4]`→`acc[NCH]`。单/两文件 device 代码同源，逐位一致。
  - **容量决定 `BM`**：`dQs[BM*HD]` fp32 是大头，`HD=512,BM=64` 总 smem 336KB 超限；故
    **`HD=128→BM=64`（回归逐位不变）**、**`HD=512→BM=16`（smem 135.17KB，1 CTA/SM）**。
    host 按 `D` 分派模板实例并各自设 `MaxDynamicSharedMemorySize`。
  - **对拍（ours-vs-ref，fp16 causal，D=Dv=512）**：S=256H2 1.638/1.582/1.753e-3；
    S=512H4 2.324/2.916/1.724e-3；S=1024H2 1.250/1.454/2.058e-3——均 fp16 噪声量级。
    FA/TE 反向不支持 head_dim=512（`fa=NA`/`te=NA`），只有 fp32 ref 可对。
    MHA D=128 回归逐位不变（S=512 1.671/1.680/1.899e-3；S=4096 1.499/1.572/2.225e-3）。
  - ncu（main, S=1024H2 D=512）：DRAM 0.10% / **L1TEX 53.21%** / L2 1.74% / Compute 7.16% /
    occ 6.25%（理论 6.25%，135.17KB smem 卡 1 CTA/SM）/ Waves 0.97 / 48 regs / No Eligible 85.5% /
    MIO scoreboard 36%、shared load 76.5% 多余 wavefront ⇒ bound 与 D=128 标量版同：
    **smem bank conflict + 低 occupancy**。
  - 性能（CUDA event）：preprocess/main/total = 0.211/1.058/1.278 ms（S256H2，0.21 TF）、
    1.291/2.080/3.491 ms（S512H4，0.62 TF）、2.463/4.143/6.789 ms（S1024H2，0.63 TF），
    峰值占比 ~0.02–0.06%（标量 + 1 CTA/SM）。同 session GQA/MQA FP16 基线 FA 155–187 TF、TE 240–278 TF。
  - 原始输出 `src/fp16/fa_bwd_fp16_{main,onefile}_p52_mla_*.out.txt`、
    `..._p52_reg_*.out.txt`、`..._p52_ncu_main_s1024h2.out.txt`、`src/fa_bwd_bench_requested_fp16_p52.out.txt`；
     文档 `docs/01-fp16-bwd-impl.md` §9。

- 2026-09-22（第十八轮）：**P5-4 完成（fp8 反向支持 GQA/MQA，单/两文件）**。
  - 映射口径同 fp16：第 `h` 个 Q 头用 KV 头 `hkv=h/(H/Hkv)`。单/两文件同步改：
    `lse_mma_kernel` 加 `Hkv`（K/ks 索引用 `*Hkv+hkv`）；`kv_prefetch` 入参 `(H,h)`→`(Hkv,hkv)`；
    `fa_bwd_fp8_mma_kernel` 加 `Hkv`，K/V/Kt 与 `dk_acc/dv_acc` 用 `Hkv/hkv`、Q/dO/dQ 用 `H/h`；
    `convert_kernel` 改收 `(nq,nkv)`；host 从 `k.npy` shape[2] 读 `Hkv`，按 `n_q/n_kv` 分配/启动/对拍。
  - 顺带修 `harness/fa_bwd_bench.py` 的 TE 导入顺序（`import transformer_engine` 先于
    `transformer_engine_torch`），否则 fp8 TE 基线报 `ModuleNotFoundError`。
  - **对拍（ours-vs-ref，fp8 causal，B1 S1024 D128）**：h40kv8 2.869/5.390/7.107e-1；
    h32kv4 2.517/5.408/7.072e-1；h64kv4 2.760/8.456/1.226；h64kv1(MQA) 4.097e-1/1.519/2.127。
    与 TE-vs-ref 同量级，**多数情形 ≤ TE**（kv1 dk/dv 1.52/2.13 vs TE 2.22/2.60）。无系统误差。
    MHA 回归逐位不变（S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；S4096
    2.635/2.643/3.216e-1）；单文件 GQA kv4 与两文件逐位相同。
  - ncu（main, h32kv4 S1024）：Duration 1.07ms、DRAM 0.93% / **L1TEX 70.54%** / L2 50.31% /
    Compute 13.46% / occ 18.75%（achieved 15.83%，168 regs，Block Limit Shared Mem=3）/
    Waves 1.29 / No Eligible 80.59% / **short_scoreboard 42.5%** ⇒ bound 与 MHA fp8 一致：
    **smem→mma 依赖 + L1/TEX**，非带宽/算力。
  - 性能（ours total / TE FP8 / 峰值占比）：h40kv8 1.635ms·13.1TF / 0.243ms·176.9TF；h32kv4
    1.365·12.6 / 0.201·170.9；h64kv4 2.369·14.5 / 0.361·190.1；h64kv1 2.293·15.0 / 0.401·171.4。
    ours/TE = 7.4/7.4/7.6/8.7%。
  - 原始输出 `src/fp8/fa_bwd_fp8_main_p53_{mha_s512,gqa}.out.txt`、
    `src/fp8/fa_bwd_fp8_p53_onefile_gqa_mha.out.txt`、`src/fp8/fa_bwd_fp8_main_p53_ncu_gqa_kv4.out.txt`、
    `src/fp8/fa_bwd_bench_requested_fp8_p53.out.txt`；文档 `docs/03-fp8-bwd-impl.md` §13、`docs/04` §7.4。

- 2026-09-22（第十九轮）：**P5-3（bf16 GQA/MQA）完成**。
  - bf16 反向（单/两文件）加 `Hkv`（映射同 fp16），`dk/dv` 按 `B*S*Hkv*D` 分配。
  - 对拍（ours-vs-ref，bf16 causal，B1 S1024 D128）：h40kv8 1.011/1.885/3.150e-2；
    h32kv4 9.631/20.19/30.94e-3；h64kv4 1.319/2.716/3.152e-2；h64kv1 1.066/3.385/5.891e-2，
    与 FA/TE 同量级、多数更小。MHA 回归逐位不变。
  - 性能（CUPTI）：ours total 15.5–28.4ms（1.11–1.21 TF）vs FA 155–189 / TE 241–281 TF（峰值 0.11–0.12%）。
  - 文档 `docs/01b-bf16-bwd-impl.md` §6c、`docs/04` §7.5；原始输出
    `src/bf16/fa_bwd_bf16_{main,onefile}_p53_gqa.out.txt`、`..._p53_ncu_*.out.txt`。

- 2026-09-22（第二十轮）：**P5-3（bf16 MLA head_dim=512）完成**。
  - bf16 反向复用 fp16 的 `HD/BM` 模板改造：`BwdTraits<HD,BM>`（`NCH=HD/32`、`KVStride=HD+2`）、
    `preprocess_kernel` 加运行时 `int HD`、main 改 `template<HD,BM>` 且 `acc[4]`→`acc[NCH]`；
    host `launch_bwd_main<HD,BM>` 按 `D` 分派——**`128→BM=64`（回归）、`512→BM=16`**（135.42KB smem，1 CTA/SM）。
    单/两文件同源，device 代码逐字一致。
  - **对拍（ours-vs-ref，bf16 causal，D=Dv=512）**：S=256H2 8.240/7.905/15.04e-3；
    S=512H4 10.43/10.33/13.85e-3；S=1024H2 5.152/7.742/15.57e-3——均 bf16 噪声（~1e-2）。
    FA/TE 反向不支持 head_dim=512（`fa=NA`/`te=NA`）；MHA D=128 回归逐位不变
    （S512 6.892/8.110/1.365e-2）。单/两文件逐位一致。
  - 性能（CUDA event）：preprocess/main/total = 0.213/0.748/0.972 ms（S256H2，0.28 TF）、
    1.297/1.461/2.872（S512H4，0.75 TF）、2.471/2.905/5.522（S1024H2，0.78 TF），峰值占比 0.03–0.08%。
    bf16 MLA 的 main 比 fp16 MLA 快 1.4–1.6×（padding 消冲突）。
  - ncu（main, S1024H2）：DRAM 0.14% / L1TEX 26.68% / L2 2.33% / Compute 13.07% /
    occ 6.25%（135.42KB smem，1 CTA/SM）/ Waves 0.97 / 48 regs / **bank conflicts ~0（padding 生效）** /
    stall long_scoreboard 1.71 + wait 1.35 ⇒ bound = **全局访存延迟 + 低并行度**（与 fp16 MLA 的
    smem 冲突不同）；非带宽/算力。
   - 原始输出 `src/bf16/fa_bwd_bf16_main_p53_mla_*.out.txt`、`..._onefile_p53_mla_*.out.txt`、
     `..._p53_ncu_mla_{s1024h2,stall_s1024h2}.out.txt`；文档 `docs/01b` §6d、`docs/04` §7.6。

- 2026-09-22（第二十一轮）：**P5-3（fp8 MLA head_dim=512）完成 — P5-3 全部收口**。
   - `src/fp8/`（单/两文件）把 head_dim 模板化：新增 `Fp8Cfg<HD,BM_,BN_>`（派生 `ASLD/KTS/QTS/
     DSS2/KVU/kNScale/smem_bytes/lse_smem_bytes/use_prefetch`）；`lse_mma_kernel<HD>`、
     `delta_kernel<HD>`、`fa_bwd_fp8_mma_kernel<HD,BM,BN>`；`mma_block` 的 k-loop 由 `#pragma unroll`
     改 `#pragma unroll 4`（HD=128 仍全展开）。
   - **两处 head_dim 角色不同**：GEMM1/2 的 HD 是归约维（k-loop 4→16 步）；GEMM3/4/5 的 HD 是
     输出 N 维 → 原「2warp×64=128 列铺满」必须加**N-tile 循环**（`HD/128=4` 遍，B 按 `d0*stride`
     偏移、写回列加 `d0`）。另修 `quantize_row_kernel` 支持 D>128（线程内 grid-stride，D=128 逐位不变）；
     `O3` 寄存器预取在 `KVU>8`（HD=512 时 KVU=32）时禁用、改 `kv_load_direct` 4B 向量化读。
   - host 按 `D` 分派 `<128,64,32>` / `<512,64,32>` 并各设动态 smem 上限；单文件由两文件 device
     代码拼接生成（逐字一致）。**HD=512 smem=228608B（223.2KB，上限 232448B 刚好放下），1 CTA/SM**。
   - **对拍（ours-vs-ref，fp8 causal，D=Dv=512，FA/TE 反向后端均不支持）**：S=256H2
     2.356/2.290/3.441e-1；S=512H4 2.415/2.992/4.481e-1；S=1024H2 2.232/3.337/3.602e-1——与 MHA
     fp8 同量级；**按 head_dim 每 128 维分段的 max_abs 均匀**，证明 N-tile 四段都对。MHA d128
     回归**逐位不变**（S512 2.426/2.975/3.735e-1），单/两文件逐位一致。
   - 性能（event）：preprocess/main/total = 0.107/0.162/0.308 ms（S256H2）、0.196/0.318/0.591
     （S512H4）、0.371/0.564/1.022（S1024H2）；total 0.87–4.20 TF（峰值 0.04–0.21%）。fp8 张量核
     MLA 的 main 比 fp16/bf16 标量 MLA 快 **6.5–7.3×**。
   - ncu（main, S=1024H2）：Duration **626.8µs**、DRAM 1.98% / L1TEX 26.28% / Compute 5.13%、
     255 regs + spill（local 137KB/shared 122KB）、223.2KB smem → **1 CTA/SM、occ 6.25%**、Waves 0.97、
     No Eligible 90.73% ⇒ **bound = 低 occupancy/并行度**（`unroll 4` 让 Duration 676.6→626.8µs）；
     要冲 2 CTA/SM 须先消 `Kt/Qt/dOt` 三个转置副本（≈107KB），即 backlog **O4b**。
   - 原始输出 `src/fp8/fa_bwd_fp8_main_p53_mla_final.out.txt`（含分段）、
     `..._mma_onefile_p53_mla_s1024h2.out.txt`、`..._p53_ncu_mla_unroll4_s1024h2.out.txt`；
     文档 `docs/03` §14、`docs/04` §7.7。

- 2026-09-22（第二十二轮）：**O2b + O4d 收口（上一条 WIP 里的 split-K 自动切块调参 + P/S 行距 padding）**。
   - 背景：HEAD 的 WIP commit（`fp8 main O4b/O4d WIP`）已把两处代码写进 `src/fp8/`，但未进
     ROADMAP/docs，且 O2b 的自动切块启发式（cap=4、按「铺满一个波」）在大 S/MLA 上明显次优。本轮正式收口。
   - **O2b**：ksplit sweep（k=1/2/4/8/16 × 6 shape，同 session）——d128 在 grid≈4096 最优、
     MLA（1 CTA/SM）在 grid≈132 最优。新启发式 `TARGET=(D==128)?4096:132; k=clamp(TARGET/base,1,16)`
     向下取 2 的幂。同 session main 相对旧自动档：S512 k4→k16 **1.08×**、S1024H32 k1→k8 **1.21×**、
     S4096 k1→k4 **1.19×**、MLA S256H2 k4→k16 **1.56×**。ncu：S512 grid 128→2048、
     achieved occ **6.25%→17.83%**（`long_scoreboard` 被压下）。
   - **O4d**：`PSS=BN+1=33`（奇数行距）。S=4096 ksplit=1：`op_ld` 冲突 199.9M→**77.3M（−61%）**、
     `op_st` 231.6M→198.4M（−14%）、总多余 wavefronts 613.6M→**456.0M（−26%）**，
     main 6.56→**5.85 ms（1.12×）**；smem 73.2→73.8KB，仍 3 CTA/SM。
   - 合并（新自动档）：total/main = 0.2916/0.1439（S512）、1.2000/0.8084（S1024H32）、
     **6.5265/4.9084 ms（21.06 TF，S4096）**；MLA 0.2490/0.1030、0.5892/0.3194、1.0114/0.5716。
     同 session TE FP8 0.1009/0.2061/0.5903 ms ⇒ 端到端 ours/TE 2.89×/5.82×/11.06×。
     数值与 P3-5/O1–O4a **逐位相同**（单/两文件一致；GQA 回归不变）。
   - **ncu（main, S4096, ksplit=4）**：Duration 5.00ms、DRAM 1.41% / **L2 81.53%** / L1TEX 69.91% /
     Compute 21.70%、achieved occ 18.27%（3 CTA/SM）、**Waves 10.34**、No Eligible 76.63%、
     stall long 4.46 + short 3.96 ⇒ **bound = L2 带宽 + 延迟**（split-K 复读 Q/dO + 全局 atomic）。
     下一步 **O4c**（atomic→分块 accum）。
    - 原始输出 `src/fp8/fa_bwd_fp8_main_o2b_*`、`..._o4d_ncu_mem_*`、`..._o2b_o4d_ncu_full_s4096*`、
      `..._o2b_o4d_tebench.out.txt`；文档 `docs/03` §15、`docs/04` §2.3/§3。

- 2026-09-23（第二十三轮）：**O4c 完成（向量化归约，fp8 main 1.19–1.45×）**。
   - 用 `lts__t_sectors_op_*` 把 O2b+O4d 后的 **L2 81.5%** 拆开：**全局 `red`（dQ/dK/dV 的
     `atomicAdd`）占 L2 扇区 408.9 M / 444 M = 92%**，DRAM 仅 1.4%；L1 每 red 请求 8 扇区
     （mma.m16n8 累加器 uncoalesced）。即所谓「L2 带宽」实为**原子归约吞吐**，非数据带宽。
   - 改动（单/两文件逐字一致）：把累加器里**相邻两列（q=0/1 与 q=2/3，同行同 scale）**
     打包成一次 `atomicAdd(float2*)`（`red.global.add.v2.f32`），新增 helper `red_add2`；
     三处 epilogue（GEMM3 dV / GEMM4 dK / GEMM5 dQ）同步改，smem/regs/几何不变。
   - **数值与 O2b+O4d 逐位相同**：S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；
     S4096 2.635/2.643/3.216e-1；MLA S1024H2 2.232/3.337/3.602e-1；GQA h32kv4 2.517/5.408/7.072e-1。
   - **性能**（event）：main base→O4c = 0.1439→**0.1090**（1.32×）/ 0.8084→**0.6772**（1.19×）/
     4.9552→**3.5330**（1.39×）/ MLA 0.5716→**0.3933**（1.45×）；total 0.2916→0.2556 /
     1.2000→1.0590 / 6.5833→**5.1601**ms。同 session TE 0.1014/0.2059/0.5894ms ⇒ main ours/TE
     **93%/30%/17%**（S512 main 已几乎追平 TE 整条反向），端到端 2.52×/5.14×/8.75×。
   - **ncu（main, S4096）**：red requests/sectors 34.1M/272.6M → **17.0M/136.3M（0.50×）**、
     L2 red 408.9→**204.5M**、**L2 81.53%→57.85%**、Duration 5.00→**3.60ms**、
     long_scoreboard 4.46→1.44；**新墙 = L1/TEX 81.3%（`ldmatrix`/smem + 残余 red）+
     short_scoreboard 3.50**；occ/regs/smem/bank-conflict 不变（168 regs/75KB/3 CTA/SM）。
   - 原始输出 `src/fp8/fa_bwd_fp8_main_o4c_{s512_h16_d128,s1024_h32_d128,s4096_h16_d128,
     s1024_h2_d512,s1024_h32_d128_kv4}.out.txt`、`src/fp8/fa_bwd_fp8_mma_onefile_o4c_s4096.out.txt`、
      `src/fp8/fa_bwd_fp8_main_o4c_ncu_s4096.out.txt`、`..._o4c_tebench.out.txt`；文档 `docs/03` §16。

- 2026-09-23（第二十四轮）：**O4b 完成（转置副本 → `ldmatrix.x2.trans` + K 配对，main 1.17–1.76×）**。
   - **先纠正 O4b 前提**：fp8 里 A/B 需要相反主序、且 `ldmatrix.trans` 的配对方向是 N
     （「2 字节=1 b16」），所以 `Kt/Qt/dOt` **不能整个消掉**（每个张量必须存两种主序）。
     真实收益 = 「逐字节 scatter 写 → 4B `__byte_perm` 交织写」+ 配对数组无 `+16` 行距放大。
   - **第一步（smoke）**：新增 `src/fp8/fa_bwd_fp8_trans_smoke.cu`，证明 `[K/2][N]` K 配对
     布局 + `ldmatrix.x2.trans` 与现有 `[N][K]` + `ldmatrix.x2` 读到的 B 片段 **逐位一致
     （bitwise_diff=0/128，PASS）**。
   - **合入（单/两文件 device 逐字一致）**：`Fp8Cfg` 的 `KTS/QTS → PSLD(HD+8)`、
     `KVU → NPU`、`fp8_bytes` 改 `qp/kp`；新增 `ldmatrix_x2_trans` + `mma_block_bt`；
     Q/dO 与 K/V 载入改「行对 unit」（Qp/dOp/Kp 用 `__byte_perm(q0,q1,0x5140/0x7362)` 交织，
     Ks/Vs 两行一次写）；GEMM3/4/5 的 B 改 `mma_block_bt`；O3 寄存器预取保留
     （NPU*4<=16 时启用）。
   - **数值与 O4c 逐位相同**：S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；
     S4096 2.635/2.643/3.216e-1；MLA S1024H2 2.232/3.337/3.602e-1；MLA S256H2 2.356/2.290/3.441e-1；
     GQA h32kv4 2.517/5.408/7.072e-1。单/两文件一致。
   - **性能（同 session A/B，main）**：S512 0.1091→**0.0733（1.49×）**、S1024H32 0.6765→**0.4552
     （1.49×）**、S4096 3.4553→**2.9473（1.17×）**、MLA S1024H2 0.3896→**0.3251（1.20×）**、
     MLA S256H2 0.0916→**0.0522（1.76×）**、GQA h32kv4 0.5923→**0.4404（1.34×）**；
     端到端 ours/TE = **2.18×/4.18×/7.90×**（S512 main 达 TE 整条反向的 **73%**）。
     smem **75520→70656B（d128）/ 229120→205824B（MLA）**。
   - **ncu（main, S4096）**：Duration 3.60→**3.00ms**、**smem `op_st` bank conflict 206.4M→69.4M
     （−66%）**、`op_ld` 68.4→69.3M、**L1/TEX 81.30%→69.69%**、L2 57.85→**69.12%**（上升成并列墙）、
     Compute 29.4→34.4%、short 3.50→**2.73**、long 1.44→**1.16**、occ 18.2%（168 regs/70.66KB，3 CTA/SM）；
     red 请求/扇区不变（17.0M/204.5M）。S=512：Duration **78.2µs**、L1/TEX 47.2%。
     MLA S1024H2：Duration **359.8µs**、smem 205.8KB、regs 255、occ 6.25%（1 CTA/SM）。
     **bound = L1/TEX 69.7% + L2 69.1%（残余 red）+ short 2.73**。
   - 原始输出 `src/fp8/fa_bwd_fp8_trans_smoke.out.txt`、`src/fp8/fa_bwd_fp8_main_o4b_*.out.txt`、
     `src/fp8/fa_bwd_fp8_mma_onefile_o4b_s4096_h16_d128.out.txt`、
     `src/fp8/fa_bwd_fp8_main_o4b_ncu_{s512,s4096,mla_s1024h2}.out.txt`、
     `src/fp8/fa_bwd_fp8_o4b_tebench.out.txt`；文档 `docs/03` §17、`docs/04` §2.3/§3。

- 2026-09-23（第二十五轮）：**O7（部分）完成（dQ 归约去 RMW：CTA 内寄存器累加 + 单次 flush，main 1.10×）**。
   - 用 `l1tex/lts__t_sectors_op_red` 拆 O4b 后的 204.5M red：dQ（GEMM5）的 epilogue 每个 nt tile 都
     对**本 CTA 同一 `(r,c)`**（GEMM5 的 warp/累加器映射与 nt 无关）做一次跨 CTA `atomicAdd`——
     一个元素被 RMW `ntiles` 次（S=4096 ksplit=4 平均 ~16）；dK/dV 每元素在一个 CTA 内只写一次
     （每个 nt 覆盖不同 KV 行），其 red 是跨 mblk/hkv 的 CTA 竞争，本轮未动。
   - **改动**（单/两文件 device 逐字一致）：dQ epilogue 改成**先折算 `sds2[r]*scale`、再累进寄存器
     `dqacc[2][8][4]`**，nt 循环后每元素只 flush 一次 `red_add2`（dQ 归约 `O(ntiles)→O(1)`）。
     因为 `sds2[r]` 逐 tile 变化，**必须先折算再累加**。仅 HD=128 启用（HD=512 需 4 份累加器
     =256 regs，不划算）；`REGDQ` 做成模板参数、host 按 `(S/BN)/2/ksplit ≥ 4` 选择。
   - 朴素版 168→**254 regs / 2 CTA/SM**（S=512/1024 反慢）；用 `__launch_bounds__(128,3)`
     压回 **168 regs（+48B spill）/ 3 CTA/SM**（HD=512 用 `,1`，255 regs 不变）。
   - **数值与 O4b 逐位相同**（S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；
     S4096 2.635/2.643/3.216e-1；MLA S1024H2 2.232/3.337/3.602e-1；GQA 2.517/5.408/7.072e-1）。
   - **性能**（event）：main S=4096 2.9473→**2.6736ms（1.10×）**、total 4.6162→**4.3410ms**；
     S=512 0.0733→0.0740、S=1024H32 0.4552→0.4604、GQA 0.4404→0.4463、MLA 0.3251→0.3277
     （均走原路，±1% session 噪声）。同 session TE FP8 0.1006/0.2054/0.5863/0.2014ms ⇒
     端到端 ours/TE = 2.18×/4.21×/**7.41×**（O4b 7.90×）；main-only S=4096 51.4 TF（峰值 2.60%）。
   - **ncu（main, S=4096）**：Duration 3.00→**2.69ms**、**L1 red 17.0M→9.04M、L2 red
     204.5M→108.5M（0.53×）**、**L2 69.12%→43.8%**、L1/TEX 69.7→64.4%、Compute 34.4→37.7%、
     short 2.73→1.89、long 1.16→1.09、barrier 0.35→0.22、occ 18.2%→18.1%（168 regs/70.66KB,
     3 CTA/SM）。**新墙 = L1/TEX 64.4% + short 1.89 + 残余 L2 43.8%（dK/dV 跨 CTA red）**。
    - 原始输出 `src/fp8/fa_bwd_fp8_main_o7_sweep.out.txt`、`..._mma_onefile_o7_sweep.out.txt`、
      `..._main_o7_ncu_s4096.out.txt`、`..._o7_tebench.out.txt`；文档 `docs/03` §18、`docs/04` §2.3/§3。

- 2026-09-23（第二十六轮）：**O5 完成（fp16 main 张量核 `mma.m16n8k16`，11.8–14.9×）**。
  - 新增 `src/fp16/fa_bwd_fp16_mma_onefile.cu`（单文件）+ `fa_bwd_fp16_mma_kernels.cuh` /
    `fa_bwd_fp16_mma_main.cu`（两文件，device 逐字同源）。5 个 GEMM 全部
    `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` + `ldmatrix`，与 fp8 张量核版（P3-4）
    同构但**无 rowwise scale**：① S=QKᵀ/② dP=dO·Vᵀ 用 `ldmatrix.x2`（B=[N][K]）；③ dV=PᵀdO/
    ④ dK=dSᵀQ/⑤ dQ=dS·K 用 `ldmatrix.x2.trans`（B=[K][N]）。P/dS 就地转 half（`PsT/dSsT/dSs`），
    D/LSE 用 fp32；**dQ 寄存器累加后直写**（无 split-K → 每 Q 块唯一 CTA，无需跨 CTA atomic）；
    dK/dV 仍 `atomicAdd(float2*)`（O4c）。smem 66.56KB（行距 +16B padding）、**168 regs、0 spill、
    3 CTA/SM**。本版只支持 HD=128（MHA/GQA），MLA 张量核留 backlog。
  - **数值**（ours-vs-ref，fp16 causal，max_abs）：MHA S512 1.671/1.771/1.899e-3、S4096
    1.883/1.734/1.966e-3；GQA/MQA S1024 kv4 2.134/3.305/3.850e-3、kv8 2.008/2.931/3.891e-3、
    kv4(h64) 2.348/5.704/3.893e-3、MQA kv1 2.292/7.934/7.517e-3——**全部与 ref/FA/TE 同量级**。
    单文件与两文件**逐位一致**。
  - **性能**（同 session A/B，main）：S=512 2.279→**0.192ms（11.8×）**、S=4096 67.55→**4.555ms
    （14.9×）**（22.4/60.4 TF，峰值 2.3%/6.1%）。端到端 S=4096 135.6→**73.34ms（1.8×）**，
    但被**标量 preprocess** 拖住（68.70ms，占 94%）；S1024 GQA/MQA total 9.5–18.7ms。
    同 session 纯反向 FA3/TE/FA2：GQA/MQA S1024 FA3 356–438/TE 306–354/FA2 218–259 TF；
    MHA S4096 FA3 848/TE 624/FA2 378 TF ⇒ ours main 约 FA3 的 7–16%。
  - **ncu（main, S4096）**：Duration 4.55ms、DRAM 1.41% / L1TEX 33.3% / L2 24.9% / Compute 17.4%、
    occ 18.75%（168 regs/66.56KB，3 CTA/SM）、Waves 2.59；**`long_scoreboard` 占 11.64 cycle 的
    63.4%**（全局读延迟）⇒ **bound = 全局访存延迟**（每 tile 同步 global→smem，无 cp.async/预取）。
    S=512（grid=128<132 SM）纯 grid-bound（Waves 0.32、achieved occ 6.25%）。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_onefile_o5_{s512_h16_d128,s4096_h16_d128,prodshapes}.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_main_o5_sweep.out.txt`、`..._o5_ncu_main_{s512,s4096}.out.txt`、
    `..._o5_fa3_te_baseline.out.txt`；文档 `docs/01-fp16-bwd-impl.md` §10、`docs/04` §2.1。

- 2026-09-23（第二十七轮）：**O5b 完成（bf16 main 张量核 `mma.m16n8k16`，9.4–9.9×；O5 全部收口）**。
  - 把 O5 的 fp16 张量核版做 **dtype 参数化**到 bf16：新增 `src/bf16/fa_bwd_bf16_mma_kernels.cuh`
    + `fa_bwd_bf16_mma_main.cu`（两文件）与 `fa_bwd_bf16_mma_onefile.cu`（单文件，由两文件拼接、
    device 逐字一致）。仅 `__half`→`__nv_bfloat16`、`f16.f16`→`bf16.bf16`、
    `__half2float/__float2half`→`__bfloat162float/__float2bfloat16`；5 个 GEMM 映射、smem 布局、
    ldmatrix（`.b16`，bf16 位宽同构）、padding（`LD=HD+8/LDP=BM+8/LDS=BN+8`）、dQ 寄存器累加、
    dK/dV `red_add2`（O4c）全部不变。只支持 HD=128（MHA/GQA）；MLA 张量核留 backlog。
  - **数值**（ours-vs-ref，bf16 causal，max_abs）：MHA S512 9.00/12.6/13.65e-3、S4096
    15.1/13.4/16.3e-3；GQA kv4 12.0/21.3/31.6e-3、kv8 12.3/19.3/31.5e-3、kv4(h64)
    13.5/30.9/44.2e-3、MQA kv1 11.9/45.6/72.0e-3——**全部 bf16 噪声量级、多数 ≤ FA/TE**。
    单文件与两文件**逐位一致**（S512 三个 max_abs 完全相同）。
  - **性能**（same-session event；main-only TF 口径 `4·B·S·H·S·(D+Dv)`）：main MHA S512
    1.88→**0.190 ms（9.9×）**、S4096 42.2→**4.51 ms（9.4×）**（22.6/60.9 TF）；GQA/MQA S1024
    main 0.639–1.119 ms（49–68 TF）。total 1.434 / 73.43 / 9.49–18.78 ms。
    同 session 纯反向 FA3 356–860 / TE 307–622 / FA2 217–379 TF ⇒ main ours/FA3 = **5.7–15.5%**。
    （幅度小于 fp16 的 11.8–14.9×，因 bf16 标量基线已做过 padding 优化、更快。）
  - **端到端仍被标量 preprocess 拖住**：S=4096 preprocess 68.79ms（占 94%）vs main 4.51ms；
    GQA S1024 亦 ~93% ⇒ 下一项 **O8（preprocess mma，对齐 fp8 O1）**。
  - **ncu（main, S=4096）**：Duration 4.55ms、DRAM 1.41% / L1TEX 33.24% / L2 24.84% /
    Compute 17.33% / occ 18.75%（168 regs/66.56KB，3 CTA/SM）/ Waves 2.59、No Eligible 77.39%；
    **`long_scoreboard` 7.35（~63%）** ⇒ **bound = 全局访存延迟**（每 tile 同步 global→smem、无
    cp.async/预取），与 fp16 张量核版逐项一致；S=512 纯 grid-bound（Waves 0.32）。
  - 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o5b_{s512,s4096,gqa}.out.txt`、
    `fa_bwd_bf16_mma_onefile_o5b_s512.out.txt`、`..._o5b_ncu_main_{s512,s4096}.out.txt`、
    `..._o5b_stall_s4096.out.txt`、`fa_bwd_bf16_mma_o5b_fa3_te_baseline.out.txt`；
    文档 `docs/01b-bf16-bwd-impl.md` §6e、`docs/04` §2.2。

- 2026-09-23（第二十八轮）：**O8 完成（fp16/bf16 preprocess 的 mma 分块 LSE，端到端 4.4–13.2×）**。
  - 动机：O5/O5b 把 main 换成张量核后，端到端被**标量 preprocess（LSE+D）**拖住
    （S=4096 preprocess 68.7ms ≈ main 的 15×、占 94%）。照搬 fp8 的 O1 解决。
  - **改动**（单/两文件 device 代码逐字一致，脚本核对 `device O8 block identical: True`）：
    删掉旧 `preprocess_kernel`，新增 **`lse_mma_kernel<HD>`**（`mma.m16n8k16` QKᵀ；
    LBM=64 行 Q × LBN=64 列 K/CTA，4 warp 各 16 行；P 不物化，fp32 累加器直接 online-softmax，
    行 max/sum 在同 row 的 4 个 lane 间 `shfl_xor` 归约；smem `(64+64)·(128+8)·2=34816B`）
    + 独立 **`delta_kernel<HD>`**（D=`rowsum(dO∘O)`，O(S·H·D)）。QKᵀ 的 k-loop 分块顺序与
    main GEMM1 相同 ⇒ LSE 与 main 的 P 自洽。
  - **数值与 O5/O5b 逐位相同**（fp16 S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、
    GQA kv4 2.134/3.305/3.850e-3；bf16 S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、
    kv4 12.01/21.25/31.56e-3），证明只换算法数据流、未改数学口径；单/两文件一致。
  - **性能**（CUDA-event，同 session）：fp16 **preprocess** S512 1.209→**0.071ms（17.0×）**、
    S4096 68.70→**0.986ms（69.7×）**、GQA kv4 8.802→0.186（47.4×）；**端到端 total**
    S512 1.440→**0.326ms（4.42×，6.59 TF）**、S4096 73.34→**5.581ms（13.1×，24.63 TF）**、
    GQA kv4 9.459→0.871（10.9×，19.72 TF）。bf16 同构（preprocess S4096 68.79→0.993ms 69.3×，
    total 73.43→5.573ms 13.2×，24.66 TF）。
  - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）：FA3 S4096 0.3246ms/847TF、
    TE 0.4441/619（fp16）；bf16 FA3 0.3217/854、TE 0.4415/623。ours total 现为 **FA3 的 2.9–4.7%
    （TFLOPS；O5 时 ~0.4%）**，时间比 10.6–17.2×。S512 MHA 单测 FA3 0.0265ms/162TF（ours total 12.3×）。
  - **ncu（lse, S=4096）**：fp16 DRAM 1.06% / L1TEX 23.42% / L2 5.23% / **Compute 39.53%** /
    occ 28.49%（80 regs 卡理论 37.5%）/ Waves 1.29 / No Eligible 42.88%，
    stall `long_scoreboard 2.17 + wait 1.34` ⇒ **bound = 全局访存延迟 + 低 occupancy/尾波**
    （与 fp8 O1 一致）；bf16 逐项一致。`delta_kernel` 仅 42.8µs（L1TEX 73.5%/Compute 71.9%/occ 71.9%）。
  - **结论**：端到端瓶颈从 preprocess 翻转为 **main**（S4096 main 4.47ms 占 ~80%）；
    S512 时 convert（0.064ms）已与 preprocess 同量级。**下一步 O6（main cp.async/预取）**。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o8_*.out.txt`、`..._mma_onefile_o8_s512.out.txt`、
    `..._o8_ncu_{lse,delta}_s4096.out.txt`、`..._o8_stall_lse_s4096.out.txt`、`..._o8_fa3_te_baseline.out.txt`、
    `..._o8_s512_fa3_te.out.txt`；`src/bf16/fa_bwd_bf16_mma_main_o8_*.out.txt`、
    `..._onefile_o8_s512.out.txt`、`..._o8_ncu_lse_s4096.out.txt`、`..._o8_fa3_te_baseline.out.txt`；
    文档 `docs/01-fp16-bwd-impl.md` §11、`docs/01b-bf16-bwd-impl.md` §6f、`docs/04` §2.1/§2.2/§3。

- 2026-09-23（第二十九轮）：**O6 完成（fp16/bf16 main 的 `cp.async` 双缓冲，main 2.26–2.41×）**。
  - 动机：O8 把 preprocess 打下后，main 成端到端第一瓶颈，ncu `long_scoreboard` 63%（每个
    tile 同步全局读 K/V，无预取）。O6 对齐 fp8 的 O3 思路，但 fp16/bf16 字节翻倍、寄存器预取
    要 32 regs（会丢 3 CTA/SM），故改用 **`cp.async.cg` 双缓冲 smem**（不占寄存器）。
  - **改动**（单/两文件 device 逐字一致）：新增 `cp_async16`（16B，L2-only）与
    `kv_issue_async<HD,BN>`（按 8 个 half/bf16=16B 的 unit 发 K/V，行越界普通写 0）；
    `Ks/Vs` 双缓冲（`stage=nt&1`），prologue 发 tile0、循环里 `wait_group 0`+`__syncthreads`
    后发下一 tile、再做 5 个 GEMM；**每 tile barrier 3→2**（尾 barrier 由下轮循环首承担）。
    用模板 `PIPE` 同 kernel 共存原版/O6 以便**同 session A/B**；`__launch_bounds__(THREADS,PIPE?2:3)`。
  - **代价**：smem `66.56→83.97KB`、3→2 CTA/SM（182 regs）。
  - **实测（同 session A/B，main-only，event）**：fp16 S512 0.1912→**0.0847ms（2.26×）**、
    S4096 4.510→**1.873ms（2.41×）**、GQA kv4 0.635→**0.375ms（1.69×）**；bf16 同构
    0.1896→**0.0837（2.27×）** / 4.485→**1.872（2.40×）** / 0.638→**0.377（1.69×）**。
    端到端 fp16 S4096 **5.581→3.043ms（45.16 TF）**、bf16 **5.573→3.036ms（45.27 TF）**；
    main 仍占 ~62%，preprocess 33%。同 session 纯反向 FA3 MHA S4096 fp16 0.3253ms/845TF、
    bf16 0.3210/856 ⇒ **ours total 为 FA3 的 5.3–5.4%**（O8 时 2.9%、时间比 13–17×，现 9.3–9.5×）。
  - **数值与 O5/O5b/O8 逐位相同**（S512 fp16 1.671/1.771/1.899e-3；S4096 1.883/1.734/1.966e-3；
    GQA kv4 2.134/3.305/3.850e-3；bf16 S512 9.00/12.6/13.65e-3、S4096 15.1/13.4/16.3e-3；
    GQA kv4 12.0/21.25/31.56e-3）——只改搬运、不改数学。单/两文件逐位一致。
  - **ncu（main, S=4096）**：fp16 Duration 4.55→**1.95ms**、**L1/TEX 33.3→65.2%**、L2 24.9→60.1%、
    Compute 17.4→27.1%、occ 18.75%(3 CTA)→**11.8%(2 CTA)**、Waves 2.59→3.88；
    **stall `long_scoreboard` 7.35→1.12**（墙被打掉），新主导 = **`wait` 1.95（fixed-latency）+
    `short_scoreboard` 0.79（smem→ldmatrix）+ L1/TEX 65%**，barrier 仅 0.10。bf16 逐项一致。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o6_*.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_onefile_o6_s512.out.txt`、`..._o6_ncu_s4096.out.txt`、
    `..._o6_stall_s4096.out.txt`、`src/fp16/fa_bwd_fp16_o6_fa3_te_baseline.out.txt`；
    `src/bf16/fa_bwd_bf16_mma_{main,onefile}_o6_*.out.txt`、`..._o6_ncu_s4096.out.txt`、
    `src/bf16/fa_bwd_bf16_o6_fa3_te_baseline.out.txt`；文档 `docs/01` §12、`docs/01b` §6g、`docs/04` §2.1/§2.2/§3。

- 2026-09-23（第三十轮）：**O6b 完成（K/V 降 smem 回 3 CTA/SM + A 转置读，fp16/bf16 单/两文件）**。
  - 背景：O6 的墙是 `wait`+`short_scoreboard`+L1/TEX，且代价是 smem 83.97KB → **2 CTA/SM**。
    O6b 在不牺牲流水的前提下把 smem 压回 3 CTA/SM 预算（≤77.8KB）并**减少 smem 访存量**。
  - **改动（单/两文件 device 逐字一致，脚本核对通过）**：① 新增 smoke
    `src/fp16/fa_bwd_fp16_atrans_smoke.cu` 证明「A 存 `[K][M]` + `ldmatrix.x4.trans`
    （地址 bit3/bit4 互换）」与「A `[M][K]` + `ldmatrix.x4`」**逐位一致**（`bitwise_diff=0/128`）；
    于是 GEMM3/GEMM4 的 A 直接从 `Ps/dSs[BM][BN]` 读，**删掉 `PsT/dSsT` 两份转置副本**
    （`mma_block_f16/bf16` 加 `ATRANS`）。② **只双缓冲 K**，V 单缓冲且在 GEMM2 后的 barrier
    之后才发下一 tile（V 的唯一消费者是 GEMM2，延迟由 GEMM3/4/5 盖住），省一整个 V 缓冲。
  - **smem 83.97→71.17KB**、Block Limit Shared Mem 2→**3**、理论 occ 12.5%→**18.75%**
    （168 regs、0 spill）。host 加自动档：`grid=(S/64)*H*B ≥ 396`（3 CTA/SM 一个满波）用 O6b(2)，
    否则 O6(1)；`--nopipe/--pipe/--pipe2` 可强制。
  - **数值与 O5/O8/O6 逐位相同**：fp16 S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、
    GQA kv4 2.134/3.305/3.850e-3；bf16 S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、
    GQA kv4 12.01/21.25/31.56e-3。单/两文件逐位一致。
  - **性能（同 session A/B，event，main-only）**：fp16 S4096 O6 1.900→**O6b 1.860ms**、
    GQA kv4 0.380→**0.357ms**；S512（grid=128 单波）O6 0.0832 < O6b 0.0866 ⇒ 走 O6。bf16 同构。
    端到端 fp16 total S4096 **2.9517ms（46.56 TF）**、GQA kv4 **0.5720ms（30.0 TF）**；
    bf16 2.9605/0.5764。
  - **对标**（同 session 纯反向 `fa_vs_te_bwd_only.py`）：FA3 MHA S4096 fp16 0.3248ms/846、
    bf16 0.3205/858；GQA kv4 FA3 0.0823ms/417 ⇒ ours total 为 FA3 的 **5.5%（fp16）/5.4%（bf16）**、
    GQA kv4 **7.2%**（时间比 9.1× / 6.9×）。
  - **ncu（main, S4096, PIPE=2）**：fp16 Duration 1.86ms、DRAM 3.47% / **L1TEX 71.87%** /
    **L2 63.09%** / Compute 29.77% / occ **16.90%**（71.17KB, 3 CTA/SM）/ Waves 2.59；stall
    `wait 1.88 + long_scoreboard 1.79 + short 0.81 + not_selected 0.37`。bf16 逐项一致。
    **结论**：occ 从 11.8%→16.9%、Duration ~5%，但 V 预取晚让 `long_scoreboard` 1.09→1.79，
    **墙已是 L1/L2 吞吐 + `wait`**（加 warp 收益有限）⇒ 下一步 **O7（去 atomic 降 L1/L2）**。
  - 原始输出 `src/fp16/fa_bwd_fp16_atrans_smoke.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_main_o6b_{s512,s4096,gqa_kv4}.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_onefile_o6b_{s512,s4096}.out.txt`、
    `..._o6b_ncu_main_s4096.out.txt`、`..._o6b_stall_s4096.out.txt`；
    `src/bf16/fa_bwd_bf16_mma_main_o6b_*.out.txt`、`..._onefile_o6b_*.out.txt`、
     `..._o6b_ncu_main_s4096.out.txt`；`src/fa_bwd_o6b_fa3_te_baseline.out.txt`；
     文档 `docs/01` §12b、`docs/01b` §6h、`docs/04` §2.1/§2.2/§3。

- 2026-09-23（第三十一轮）：**O8b 完成（fp16 LSE 预处理负载均衡 + `cp.async` 双缓冲，端到端 1.26×）**。
  - 动机：O6/O6b 把 main 打到 1.86ms 后，`lse_mma_kernel` 仍 1.03ms、占端到端 **34%**；
    ncu 显示 Compute 39.5% / DRAM 1.1% / `long_scoreboard 2.19 + wait 1.34` / Waves 1.29
    （ncu 报尾波可达 50%——因果下第 `mblk` 个 CTA 做 `mblk+1` 个 K tile，重块排最后）。
  - **改动**（单/两文件 device 逐字一致，脚本核对 `device identical: True`）：新增
    `lse_mma_kernel_bal<HD,PIPE>`：① **镜像配对**——每 CTA 处理 `m=blockIdx.x` 与
    `m'=nblk-1-m`，工作量恒 `nblk+1`，grid.x 64→32；② `PIPE=1` K 用 `cp.async.cg` 16B 双缓冲
    （prologue 发 tile0、循环首 wait+sync 后发下一 tile）；`PIPE=0` 退回同步标量读用于消融。
    仅 causal 使用，非 causal 走 O8 原版；mask 由 `!(causal&&jg>qi)` 写为 `jg<=qi`。
  - **消融（同 session，lse-only）**：O8 0.9853ms → 镜像配对(单缓冲) 0.4457ms（**2.20×**）
    → 镜像配对+cp.async 0.3484ms（**2.81×**）；S=512 1.38×、GQA kv4 2.09×。
    **收益主要来自负载均衡，cp.async 再叠加 1.28×**。
  - **数值与 O5/O8/O6/O6b 逐位相同**（S512 1.671/1.771/1.899e-3；S4096 1.883/1.734/1.966e-3；
    GQA kv4 2.134/3.305/3.850e-3），单/两文件逐指标一致。
  - **性能**（同 session，端到端）：preprocess S4096 1.0094→**0.3870ms（2.6×）**；
    total S4096 2.9517→**2.3364ms（58.8 TF，峰值 5.9%）**、GQA kv4 0.5720→0.5183ms。
    同 session 纯反向 FA3 S4096 0.3241ms/848TF ⇒ **ours total 为 FA3 的 6.9%**（O6b 5.5%）、
    时间比 7.2×。
  - ncu（lse, S4096）：Duration 1.03ms→**354µs**、`long_scoreboard` **2.19→0.34**、
    Waves **1.29→0.97**、Compute 39.5%→**60.4%**、occ 28.5%→22.9%（52.2KB smem→4 CTA/SM）；
    新墙 = **Compute 60% + smem 依赖（wait 1.57 + short 1.33）**。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o8b_{s512,s4096,gqa_kv4}.out.txt`、
    `..._mma_onefile_o8b_s4096.out.txt`、`..._o8b_ncu_lse_bal_s4096.out.txt`、
    `..._o8b_stall_lse_bal_s4096.out.txt`、`src/fp16/fa_bwd_fp16_o8b_fa3_te_baseline.out.txt`；
    文档 `docs/01` §13、`docs/04` §2.1/§3。

- 2026-09-23（第三十二轮）：**O8b-bf16 完成（bf16 LSE 负载均衡 + `cp.async`，端到端 1.26×；O8b 全部收口）**。
   - 把 fp16 的 O8b **逐字 dtype 参数化**到 bf16：新增 `lse_mma_kernel_bal<HD,PIPE>`
     （`__half`→bf16、`__float2half`→`__float2bfloat16`），① 镜像配对（`m` 与 `nblk-1-m`，
     工作量恒 `nblk+1`、grid.x 64→32）；② `PIPE=1` K 用 `cp.async.cg` 16B 双缓冲。仅 causal
     走新 kernel，非 causal 走 O8 原版。单/两文件 device 代码逐字一致（脚本核对）。
   - **消融（同 session，lse-only）**：O8 0.9696 → 镜像配对(单缓冲) 0.4524ms（**2.14×**）
     → 镜像配对+cp.async 0.3503ms（**2.77×**）；S=512 1.38×、GQA kv4 2.06×。
     收益主要来自负载均衡，cp.async 再叠加 ~1.29×（与 fp16 一致）。
   - **数值与 O5b/O8/O6/O6b 逐位相同**（S=512 9.001/12.61/13.65e-3；S=4096 15.10/13.40/16.31e-3；
     GQA kv4 12.01/21.25/31.56e-3），单/两文件逐指标一致。
   - **性能**（同 session，端到端）：preprocess S4096 0.993→**0.392ms**；total S4096
     2.961→**2.343ms（58.67 TF，O6b 46.4 TF）**、S512 0.186→**0.160ms**、GQA kv4
     0.576→**0.482ms（35.66 TF）**。同 session 纯反向 FA3 S4096 **0.3194ms/861TF**、TE 0.4419/622；
     GQA kv4 FA3 0.0825ms/416 ⇒ ours total 为 FA3 的 **6.8%**（O6b 5.4%）、时间比 7.3×；
     GQA 8.6%、5.8×。
   - ncu（lse_bal, S=4096）：Duration **356.4µs**、DRAM 3.07% / L1TEX 32.40% / L2 26.77% /
     **Compute 60.40%**、64 regs / 52.22KB smem（**4 CTA/SM**，occ 22.91%）/ Waves **0.97**；
     stall `wait 1.57 + short 1.33 + not_selected 0.79 + long 0.34`，与 fp16 O8b 逐项一致。
     新墙 = **Compute 60% + smem 依赖**。
   - 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o8b_{s512,s4096,gqa_kv4}.out.txt`、
     `..._mma_onefile_o8b_{s512,s4096}.out.txt`、`..._o8b_ncu_lse_bal_s4096.out.txt`、
     `..._o8b_stall_lse_bal_s4096.out.txt`、`src/bf16/fa_bwd_bf16_o8b_fa3_te_baseline.out.txt`；
      文档 `docs/01b` §6i、`docs/04` §2.2/§3。

- 2026-09-23（第三十三轮）：**O6c 完成（fp16 主 kernel tile 几何参数化 + 小网格并行度自适应，main 最多 1.13×）**。
  - **先证伪一个假设**：O6b ncu 报 Waves 2.59、尾波最多 33%、SM active cycles 偏差 22.8%/26.5%，
    于是做了「不改数学、只重排 `blockIdx.x→mblk` 双射」的负载均衡（`sched=1` 交错 / `sched=2` 逆序）。
    **实测只有 0–2%、且不同 session 正负号不稳** ⇒ GPU block 调度本就是动态的，静态重排无收益
    （`sched` 开关保留、默认 0，不作为优化）。
  - **真杠杆 = 小网格的并行度**：S=512/H16 时 `grid=128 < 132 SM`，每 SM 只有 1 个 CTA（4 warp）。
    把 `BM` 从 64→32 让 grid 翻倍到 256（每 SM 2 CTA）、并行度翻倍；代价是 dK/dV 跨 CTA 原子量随
    `BM` 反比翻倍，故只在 `S≤1024` 用。
  - **改动（单/两文件 device 逐字一致，`diff` 核对 `device region identical: True`）**：
    ① `fa_bwd_fp16_mma_kernel<HD,BM,BN,PIPE>` 的 2×2 warp 几何全部由 `(BM,BN)` 派生
    （`GM1/GN1=BM/2,BN/2`；`GMV/GNV=BN/2,HD/2`；`GMQ/GNQ=BM/2,HD/2`；`MT*=WARP_M/16,WARP_N/8`），
    所有 `16/32/64` 字面量与 `acc/pval/dqacc` 维度改为按 `MT*` 定义；`BM=64,BN=32` 展开与 O6b 同构。
    ② host `launch_cfg` 自动档 + CLI `--bm/--bn/--pipe` 覆盖：`grid<132 且 S≤1024`→`(32,32,1)`；
    否则 `BM=64`（`grid≥396`→`PIPE=2`，否则 `PIPE=1`）；`S≥4096`→`BN=64`。
  - **配置 A/B（同 session，event，main-only，ms）**：S=512 (64,32,2) 0.0858 → (64,64,2) 0.0814 →
    **(32,32,1) 0.0776（1.106×）**；S=1024 kv4/kv8 仍 (64,32,2) 最优；S=4096 (64,64,2) 1.8220 vs
    (64,32,2) 1.8535（**1.017×**）。(32,32,*) 在大 S 明显更差（原子量翻倍）。
  - **端到端**：S=512 0.1598→**0.1504ms（1.06×，14.3 TF）**；S=4096 2.347→2.349ms（持平，main +1.7%）。
    **数值与 O5/O8/O6/O6b/O8b 逐位相同**（S512 1.671/1.771/1.899e-3；S4096 1.883/1.734/1.966e-3；
    kv4 2.134/3.305/3.850e-3；kv8 2.008/2.931/3.891e-3）。
  - **ncu**：S=512 原 (64,32,1) Duration **99.3µs**/occ **6.24%**；新 (32,32,1) **83.9µs（−15.5%）**/
    occ **10.99%**、L1TEX 26.3→43.5%（每 SM 干活更多）。S=4096 `BN=64`：**L1/TEX 71.87%→57.27%**
    （Q/dO 复用翻倍）但 smem 105KB → **2 CTA/SM**（occ 18.75→12.5%），净 +1.5%；**要同时拿到低
    L1 与高 occupancy 必须先把 smem 砍到 ≤77.7KB（BN=64 需砍 ~28KB），留待 O9**。
  - **对标**（`harness/fa_vs_te_bwd_only.py fp16`）：S=4096 MHA FA3 0.3240ms/848TF、TE 0.4440/619
    ⇒ ours total 6.9%（时间 7.25×，与 O8b 持平）；S=512 MHA FA3 0.0265ms/81TF、TE 0.0321/67
    ⇒ ours total 17.6%（时间 5.7×，O8b 时 12.3×）。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o6c_*.out.txt`、`..._onefile_o6c_*.out.txt`、
    `..._o6c_ncu_*.out.txt`、`src/fp16/fa_bwd_o6c_fa3_te_*.out.txt`；文档 `docs/01` §13b。

- 2026-09-23（第三十四轮）：**O6c(bf16) 完成（bf16 主 kernel tile 几何参数化 + 小网格并行度自适应）**。
  - 把 fp16 的 O6c **逐字 dtype 参数化**到 bf16（`__half`→bf16、`mma_f16`→`mma_bf16`、
    `__float2half`→`__float2bfloat16`）：
    `fa_bwd_bf16_mma_kernel<HD,BM,BN,PIPE>` 的 2×2 warp 几何全部由 `(BM,BN)` 派生
    （`GM1/GN1=BM/2,BN/2`；`GMV/GNV=BN/2,HD/2`；`GMQ/GNQ=BM/2,HD/2`；`MT*=WARP/16,WARP/8`），
    `pval/dqacc/acc` 按 `MT*` 定义，`__launch_bounds__=(THREADS,(BN>32)?2:3)`；host 自动档
    `grid<132 且 S≤1024`→`(32,32,1)`、`S≥4096`→`BN=64`、`grid≥396`→`PIPE=2`。单文件由两文件
    device 段重新拼接（脚本核对 device 段**逐字一致**）。
  - **配置 A/B（同 session，event，main-only，ms）**：S=512 (64,32,2) 0.0886 → (64,64,2) 0.0835
    → **(32,32,1) 0.0800（1.108×，另一 session 1.18×）**；S=1024 kv4 仍 (64,32,2) 最优（1.00×）；
    S=4096 (64,64,2) 1.8349 vs (64,32,2) 1.8672（**1.018×**）。自动档选中：S=512⇒(32,32,1)、
    S=1024 kv4⇒(64,32,2)、S=4096⇒(64,64,2)。
  - **端到端**：S=512 0.1603→**0.1501ms（1.06×，14.31 TF）**、S=4096 2.3427→2.3692ms
    （58.01 TF，session 噪声持平）、GQA kv4 0.4817→0.4838ms（35.51 TF）。`[O6c A/B]` sched=0/1/2
    0.0845/0.0849/0.0846（S512）、1.8805/1.8547/1.8403（S4096）⇒ 再次证伪静态 mblk 重排。
  - **数值与 O5b/O8/O6/O6b/O8b 逐位相同**：S=512 9.001/12.61/13.65e-3；S=4096 15.10/13.40/16.31e-3；
    GQA kv4 12.01/21.25/31.56e-3。单/两文件逐指标一致（MHA S512/S4096 的 dq/dk/dv 相同）。
  - **ncu（main）**：S=512 `(32,32,1)` Duration **83.9µs**、L1TEX 43.94% / L2 44.89% / Compute
    11.91%、145 regs / 59.90KB、理论 occ 18.75%、achieved **10.99%**、Waves 0.65；stall
    `long 3.23 + wait 1.89 + short 1.08`。S=4096 `(64,64,2)` Duration **1.86ms**、**L1TEX 57.05%**
    （O6b 71.71%）/ L2 63.02% / Compute 26.77%、242 regs / 105.47KB、**2 CTA/SM**（occ 11.82%，
    O6b 16.86%），stall `wait 1.94 + long 1.45 + short 0.46` ⇒ `BN=64` 降 L1/TEX 但掉 occupancy，
    净 +1.5%；**墙 = L1/L2 吞吐 + `wait`**。
  - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：FA3 MHA S=4096 **0.3195ms/860TF**、
    TE 0.4360/630（FA2 0.7278/378）；S=512 FA3 0.0265/162、TE 0.0324/132；GQA kv4 FA3 0.0822/418、
    TE 0.1116/308。ours total 为 FA3 的 **6.7%（S4096，时间 7.41×）/ 8.8%（S512，5.70×）/
    8.5%（GQA kv4，5.89×）**。
  - 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o6c_{s512_h16,s4096_h16,s1024_h32_kv4}.out.txt`、
    `..._onefile_o6c_{s512_h16,s4096_h16}.out.txt`、`..._o6c_ncu_{s512_main,s4096_main}.out.txt`、
     `..._o6c_stall_{s512,s4096}.out.txt`、`src/bf16/fa_bwd_o6c_fa3_te_{baseline,s512}.out.txt`；
    文档 `docs/01b` §6j、`docs/04` §2.2。

- 2026-09-23（第三十五轮）：**O7c 完成（fp16/bf16：LSE/D 预装寄存器 + dK/dV float4 试错）**。
  - **负结果（先证伪一条路）**：O7 的「去原子/减 red 事务数」——把 dK/dV 的 `red.global.add`
    从 float2 提到 **float4**（`mma.m16n8` 一个 quad 的 `c2=0/2/4/6` 恰是同 row 连续 8 列，
    用两次 `shfl_down 1` 打包成两个 float4，偶 lane 落 `REDG.E.ADD.F32x4`，事务数减半）。
    四个几何 **全部变慢 1–7%**（fp16 S4096 (64,32,2) −6.9% / (64,64,2) −2.6%；bf16 −3~−5%）
    ⇒ **该归约不是事务数 bound**，`shfl` + 「半 lane 落 store」的代价 > 省下的事务。
  - **正结果（本轮真实收益）**：`lse`/`delta` **只依赖 CTA 自己的 Q 行、与 K tile 无关**，
    原实现每个 K tile 的 GEMM1/2 epilogue 都按 `qi` 去 global 读（ncu：global load 每 thread
    仅 4.4/32B）。改在 `nt` 循环前一次性装进 `lse_r[MTM1][2]/del_r[MTM1][2]`，循环内零 global 读。
  - **改动（单/两文件 device 逐字一致，脚本核对 `identical: True`）**：kernel 加模板开关
    `R4`（float4 归约，默认 false）与 `PREL`（预装，默认 true）；host 加 `--r4=` / `--prel=`
    与 2×2 A/B 段。fp16/bf16 同步。
  - **性能（同 session A/B，event，main-only）**：**PREL 相对 base**——fp16 (64,64,2) S4096
    1.8464→**1.5576ms（+18.5%）**、(64,32,2) 1.7990→**1.6081（+11.9%）**、S512 (64,64,2)
    0.0809→**0.0698（+16.0%）**、GQA kv4 S1024 (64,32,2) 0.3627→**0.3044（+19.1%）**；
    bf16 同构 1.8297→**1.5566（+17.5%）** / 0.3580→**0.3046（+17.5%）**。端到端 fp16
    **total S4096 2.3762→2.0935ms（1.13×）**、S512 0.1504→0.1495、GQA kv4 0.4344；
    bf16 total S4096 2.3692→**2.0986ms（1.13×）**。
  - **数值与 O5/O5b/O8/O6/O6b/O8b/O6c 逐位相同**（fp16 S512 1.671/1.771/1.899e-3、
    S4096 1.883/1.734/1.966e-3、GQA kv4 2.134/3.305/3.850e-3；bf16 S512 9.001/12.61/13.65e-3、
    S4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3）。
  - **ncu（main, S=4096, (64,64,2)）**：fp16 Duration 1.99→**1.61ms**、**L1/TEX 57.5→55.7%**、
    **L2 58.7→71.6%（新墙）**、Compute 27.0→23.4%、DRAM 4.0%、regs 242→250 / smem 105.47KB /
    仍 2 CTA/SM（occ 11.8%）、`long_scoreboard` 1.79→**1.38**、wait 1.88→2.00、short 0.81→0.88、
    mio 0.28→0.49；bf16 逐项一致。**结论：墙从 L1/TEX 移到 L2（残余 dK/dV `red`）+ `wait` +
    低 occupancy（105KB smem / 250 regs 锁死 2 CTA/SM）**；继续抠 red 宽度/tile 几何边际很小，
    **下一步转 O9（wgmma+TMA，唯一能同时降 smem 与提 occupancy 的杠杆）**。
  - **对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）**：fp16 S4096 MHA FA3
    0.3235ms/850TF、TE 0.4438/619 ⇒ ours total 时间 **6.47×**（真反向 FLOPs 口径 131TF≈15.4%）；
    GQA kv4 S1024 FA3 0.0828/415 ⇒ 5.25×。bf16 S4096 FA3 0.3193/861 ⇒ 6.57×、GQA kv4 5.25×。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o7c_{s512,s4096,gqa_kv4}.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_onefile_o7c_{s512,s4096}.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_main_o7c_ncu_s4096.out.txt`、`..._o7c_stall_s4096.out.txt`、
    `src/fp16/fa_bwd_fp16_o7c_fa3_te_baseline.out.txt`；`src/bf16/` 同构文件（`..._o7c_*`）；
    改动前基线 `src/fp16/fa_bwd_fp16_mma_main_o7c_base_{s512,s4096}.out.txt`；
    文档 `docs/01` §14、`docs/01b` §6k、`docs/04` §2.1/§2.2/§3/§6。

- 2026-09-23（第三十六轮）：**MLA（head_dim=512）张量核反向完成（O5c，fp16+bf16，单/两文件）**。
  - 动机：P5-2/P5-3 的 fp16/bf16 MLA 反向一直是**标量 golden**（135KB smem、1 CTA/SM）。
    fp8 早在第二十一轮就把 MLA 上了张量核（main 比标量 MLA 快 6.5–7.3×）；本轮把 fp16/bf16
    补齐，消除 MLA 的标量短板。
  - **改动（单/两文件 device 代码逐字同源）**：`fa_bwd_{fp16,bf16}_mma_kernel<HD,...>` 从
    `static_assert(HD==128)` 扩到 **HD=128/512**：① GEMM1/2 归约维是 HD ⇒ 只是 k-loop 从
    8 步变 32 步；② GEMM3/4/5 输出 N 维是 HD ⇒ 加 **N-tile 循环**（`NTW=WN*64=128` 列/遍，
    `NDT=HD/NTW` 遍），B（`[K][HD]` 转置布局）基址/写回列号 `+hd0`；③ dQ 的 HD 列放不进寄存器 ⇒
    HD>128 时在 GEMM5 epilogue **直接全局累加**（每 `(qi,列)` 由唯一线程拥有：唯一 CTA + 唯一
    warp + 唯一 N-tile ⇒ **非原子 RMW 无竞争**，`dq_acc` 由 host `memset(0)`）。
    host 按 `D∈{128,512}` 分派 LSE/delta/主 kernel 并各设 smem；MLA 自动档 `BM=32,BN=32,PIPE=1`。
  - **数值（ours-vs-ref，fp16/bf16 causal，D=Dv=512）**：fp16 S256H2 1.638/1.582/1.753e-3、
    S512H4 2.516/2.916/1.724e-3、S1024H2 1.987/1.712/1.848e-3；bf16 S512H4
    8.753/1.082e-2/1.740e-2 等——均对应 dtype 噪声量级。**MHA D=128 回归逐位不变**
    （fp16 S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3；bf16 S512
    9.001/12.61/13.65e-3），单/两文件逐指标一致。
  - **性能（同 session CUDA event）**：fp16 main 标量→张量核 1.065→**0.204**（5.2×）/
    2.120→**0.385**（5.5×）/4.214→**0.725ms**（5.8×），total 1.288→0.300 / 3.507→0.536 /
    6.817→**0.939ms（7.3×）**；bf16 main 0.753→**0.204**（3.7×）/1.488→**0.385**（3.9×）/
    2.959→**0.725ms**（4.1×）。配置 A/B `(32,32,PIPE=1)` 比 `(32,32,0)` 快 **1.67×**
    （K 双缓冲）。FA/TE 反向不支持 head_dim=512，无对标基线（仅 fp32 ref）。
  - **ncu（main，S=1024 H2 D=512，fp16/bf16 逐项相同）**：Duration **769µs**、
    DRAM 0.83% / L1/TEX 40.1% / L2 13.9% / Compute 2.2%、168 regs / **207.36KB smem → 1 CTA/SM**、
    occ 6.25%、**Waves 0.48**、No Eligible 91.6%、stall **`long_scoreboard` 7.68** + wait 2.23、
    bank conflict `op_ld` 仅 1.13M ⇒ **bound = 全局访存延迟 + 低并行度**（grid=64 < 132 SM、
    1 CTA/SM），不再是标量 MLA 的 smem bank conflict。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_p5mla_{s256h2,s512h4,s1024h2}.out.txt`、
    `..._mma_onefile_p5mla_s512h4.out.txt`、`..._mma_main_p5mla_reg_d128_s512.out.txt`、
    `src/fp16/fa_bwd_fp16_main_p5mla_*.out.txt`（标量基线）、
    `..._mma_main_p5mla_ncu_{sol,stall}_s1024h2.out.txt`；`src/bf16/` 同构文件（`..._p5mla_*`）；
     文档 `docs/01` §14b、`docs/01b` §6l、`docs/04` §7.8。

- 2026-09-23（第三十七轮）：**O10 完成（fp16/bf16：Q/dO 载入向量化 + `cp.async` 重叠 + 打包写回，
  端到端 1.04–1.32×）**。
  - 动机（O7c ncu 的 `Source Counters`）：① 主 kernel prologue 的 Q/dO 仍是逐元素
    `LDG.U16 + STS.U16`（global 平均仅用满 26.4/32 B/sector，且每元素做整除取模），其同步延迟
    **串在** K/V 的 `cp.async` 之前；② `lse_mma_kernel_bal` 的 Q 同样是标量读；③ dQ 写回为两次
    4B store（global stores 仅用满 16/32 B/sector）。
  - **改动（单/两文件 device 代码逐字一致，脚本核对 `identical: True`）**：① 新增
    `qdo_issue_async<HD,BM>`（16B `cp.async.cg` 发 Q/dO，行越界写 0），主 kernel `PIPE>=1`
    prologue 改用它（`PIPE==0` 保持同步标量读），与 K/V 各自 `commit_group`、循环首
    `wait_group 0` 一并等待 ⇒ Q/dO 延迟与 K/V 重叠；② `lse_mma_kernel_bal` 的 Q 改 `issue_q`
    （PIPE=1 cp.async / PIPE=0 标量）；③ dQ 写回（含 HD>128 直接累加）相邻两列打包 `float2`。
  - **数值与 O5/O5b/O8/O6/O6b/O8b/O6c/O7c 逐位相同**（fp16 S512 1.671/1.771/1.899e-3、
    S4096 1.883/1.734/1.966e-3、GQA kv4 2.134/3.305/3.850e-3、MLA S1024H2 1.987/1.712/1.848e-3；
    bf16 S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3），单/两文件逐指标一致。
  - **性能（同 session A/B：改动前二进制 vs O10，CUDA event）**：fp16 **total** S=512
    0.1485→**0.1256ms（1.18×）**、S=4096 2.0965→**2.0044（1.05×）**、GQA kv4 0.4360→**0.3767
    （1.16×）**、MLA S256H2 0.2990→**0.2272（1.32×）**、MLA S512H4 0.5309→**0.4357（1.22×）**、
    MLA S1024H2 0.9417→**0.8230（1.14×）**；main GQA kv4 0.3062→**0.2620（1.17×）**、
    MLA S256H2 1.22×。bf16 同构（S512 1.15×、S4096 1.04×、GQA kv4 1.15×、MLA S256H2 1.31×，
    total）。收益主要来自 **LSE 的 Q 向量化**（S=512 preprocess 0.0536→0.0405）与
    **主 kernel Q/dO 的 `cp.async` 重叠**（小网格更明显）；大 S 收益最小（prologue 占比小）。
  - **ncu（main, S=4096, (64,64,2)）**：fp16 Duration 1.61→**1.48ms**、Executed Instructions
    350.9M→**342.1M（−2.5%）**、`long_scoreboard` 1.38→**0.89**、`mio_throttle` 0.49→0.70、
    shared load 多余 wavefront 15.4%→**12.2%**；regs 250 / smem 105.47KB / 2 CTA/SM 不变。
    bf16 逐项相同（Executed Instructions 342,052,864）。**墙仍 = `wait`（mma 依赖）+ L2 + 低 occ**，
    只有 O9 能同时解。
  - **对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）**：fp16 MHA S=4096 FA3
    0.3242ms/848TF、TE 0.4413/623、FA2 0.7287/377 ⇒ ours total 时间 **6.18×**（O7c 6.47×）；
    GQA kv4 S=1024 FA3 0.0821ms/418TF ⇒ 4.59×。bf16 同量级。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o10_{ab,allshapes,ncu_s4096,ncu_s512,stall_s4096}.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_onefile_o10_*.out.txt`、`src/fp16/fa_bwd_fp16_o10_fa3_te_baseline.out.txt`；
    `src/bf16/` 同构文件（`..._o10_*`）；文档 `docs/01` §14c、`docs/01b` §6m、`docs/04` §2.1/§2.2。

- 2026-09-23（第三十八轮）：**O11 完成（fp8 LSE 负载均衡 + `cp.async` 双缓冲，preprocess 3.1×；
  另附 fp16/bf16/fp8 快速 exp/log）**。
  - 动机：fp16/bf16 在 **O8b** 已把 LSE 做「镜像配对 + `cp.async` 双缓冲」（lse 2.8×），
    但 **fp8 的 `lse_mma_kernel`（O1）从未做这两件事**——`S=4096` fp8 preprocess 仍 **1.20ms**
    （占端到端 ~29%），而 fp16/bf16 同 shape 只有 0.35ms。
  - 新增 `lse_mma_kernel_bal<HD,PIPE>`（`src/fp8/fa_bwd_fp8_kernels.cuh`，单文件同源）：
    ① 镜像配对（每 CTA 做 `m` 与 `nblk-1-m`，工作量恒 `nblk+1`，grid.x 减半）；
    ② K/Q 的 `cp.async.cg` 16B 双缓冲（fp8 一行 `HD` 字节 = `HD/16` 个 16B unit）。
    仅 causal 走新 kernel，非 causal 走 O1 原版。新增 `Fp8Cfg::lse_smem_bytes_bal0/1`。
  - 另附 **快速 exp/log**：`fexp`/`flog`（`__expf`/`__logf`，MUFU，相对误差 ~2^-21）替换
    fp16/bf16/fp8 三套件里 softmax 热点的 9/9/5 处 `expf`/`logf`，`FAST_EXP` 宏 A/B。
  - **数值与 O7 逐位相同**：S=4096 2.635/2.643/3.216e-1；S=512 2.426/2.975/3.735e-1；
    GQA h32kv4 2.517/5.408/7.072e-1；MQA kv1 4.097e-1/1.519/2.127；MLA S1024H2
    2.232/3.337/3.602e-1。单/两文件逐指标一致（S=4096 total 3.308 vs 3.310ms）。
  - **性能（event）**：lse 消融 S=512 1.67× / S1024 kv4 2.01× / MQA kv1 2.40× /
    S=4096 **2.71×**（0.936→0.345ms）；**preprocess S=4096 1.20→0.388ms（3.1×）**；
    **端到端 4.13→3.31ms（1.25×，41.5 TF）**；纯反向（去 quant）≈3.13ms ⇒ ours/TE FP8
    6.7×→**5.3×**。fast-exp A/B（fp16 S4096）：total 2.0053→1.9697（1.8%）、preprocess
    0.3712→0.3439（**8.0%**）、main 不变。
  - **同轮证伪两条**：fp16 主 kernel `(BM=32,BN=64,PIPE=2)`（dK/dV 原子量翻倍，main 慢 1.75×）、
    `cp.async .L2::256B` 提示（无变化）。
  - ncu（lse_bal, S=4096）：Duration **357µs**、DRAM 1.48% / L1/TEX 28.2% / L2 15.0% /
    **Compute 61.2%** / occ 23.2% / **Waves 0.65** ⇒ 新墙 = **Compute 61% + 网格不足一个波**
    （与 fp16/bf16 O8b 一致）。
  - 原始输出 `src/fp8/fa_bwd_fp8_main_o11_lsebal.out.txt`、
    `src/fp8/fa_bwd_fp8_main_o11_ncu_lsebal_s4096.out.txt`、
    `src/fp8/fa_bwd_fp8_mma_onefile_o11_s4096.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_main_o11_{ab_fastexp,s4096,s512}.out.txt`、
    `src/bf16/fa_bwd_bf16_mma_main_o11_{s512,s4096}.out.txt`、`src/fa_bwd_o11_fa3_te_baseline.out.txt`；
    文档 `docs/03` §19、`docs/01` §14d、`docs/01b` §6n、`docs/04` §2.3/§3。

- 2026-09-23（第三十九轮）：**O9a 完成（Hopper `wgmma` 数据通路建立 + LSE 上验证，单/两文件）**。
  - 动机：O10 后 fp16/bf16 main 的墙是 **`wait`（mma 依赖）+ L2**；`mma.m16n8k16` 在 Hopper 上走
    SM80 兼容路径（LSE ncu 还有 L1/TEX 32–38% 的 ldmatrix）。O9 要把它换成 Hopper 原生
    `wgmma`。本轮做**风险最小的第一步**——先把 LSE 的单个 QKᵀ 换掉，验证 SW128+描述符+累加器映射。
  - **冒烟** `src/fp16/fa_bwd_fp16_wgmma_smoke.cu`：证明 (1) SW128 K-major 布局 + wgmma 描述符
    自洽；(2) `wgmma.m64n64k16.f32.f16.f16` 的累加器布局与 `mma.m16n8` 的 `acc[j][q]` **逐字同构**
    （`d[j*4+q]`）。实测 `max_abs=0.000e+00` PASS。
  - **集成**（`lse_mma_kernel_bal_wgmma<HD,PIPE>`，仅 HD=128/causal）：Q/K 改 **SW128 tile**
    （`cp.async` 落 swizzled 地址）、8 条 wgmma 完成 64×64×128 的 QKᵀ，镜像配对/online-softmax/
    行归约同 O8b。fp16/bf16 单/两文件 device 代码逐字一致；默认走 mma 版，`--lsewgm=1` 切换。
  - **构建**：CUDA 13 需 `-gencode=arch=compute_90a,code=sm_90a`；`scripts/run.sh`/`ncu.sh`
    改为「`ARCH` 为空串时只用 `NVCC_FLAGS` 的 gencode」，kernel 用 `__CUDA_ARCH_FEAT_SM90_ALL`
    包 wgmma asm，使默认 `sm_90` 构建仍可用。
  - **性能**（同 session，LSE-only，event）：fp16 S=4096 O8b 0.3004→**wgmma 0.2871ms（1.046×）**、
    S=512 0.0338→0.0330、GQA kv4 0.0666→0.0642；bf16 S=4096 0.3006→**0.2856ms（1.054×）**。
    端到端 fp16 S=4096 total **1.953ms（70.1 TF）**（O10 2.004）。
  - **ncu**（S=4096，同 session 对照 O8b）：Duration 303.62→**286.18µs**、**L1/TEX 37.62→18.05%**
    （ldmatrix 被 SS 直读取代）、L2 31.62→20.25%、Executed Ipc 2.36→**2.52**；regs 62、
    smem 52.22→50.18KB、occ ~23%、Waves 0.97。**新墙 = Compute ~60% + 发射**（softmax epilogue）；
    wgmma 只打掉访存那一半 ⇒ LSE 收益有限（1.05×）。
  - **数值与 O8b 逐位相同**（fp16 S4096 1.883/1.734/1.966e-3、S512 1.671/1.771/1.899e-3、
    GQA 2.134/3.305/3.850e-3；bf16 S4096 1.510/1.340/1.631e-2）；单/两文件逐指标一致。
  - **对标**（同 session 纯反向 `fa_vs_te_bwd_only.py`）：FA3 MHA S4096 fp16 0.3240ms/848TF、
    bf16 0.3194/861 ⇒ ours total 仍 ~6.0×（与 O8b/O10 持平，LSE 非主因）。
  - 原始输出 `src/fp16/fa_bwd_fp16_wgmma_smoke.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_{main,onefile}_o9_*.out.txt`、`..._o9_gqa_kv4.out.txt`、
    `src/bf16/fa_bwd_bf16_mma_{main,onefile}_o9_s4096.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_main_o9_ncu_lse_{wgmma,mma}_s4096.out.txt`、
    `src/{fp16,bf16}/fa_bwd_*_o9_fa3_te_baseline.out.txt`；文档 `docs/01` §14e、`docs/01b` §6o、
    `docs/04` §2.1/§2.2/§3。

- 2026-09-23（第四十轮）：**O13 完成（fp16/bf16 主 kernel auto tile 重新标定 + 端到端 memset/convert 冗余）**。
  - 动机：O6c 的 auto tile 是**在 O7c-PREL 之前**定的（`grid<132 且 S≤1024` → `(32,32,1)`）；
    PREL/O10 之后每 CTA 固定开销下降、`BM=64` 反超。复测 S=512 MHA：`(64,64,2)` 主 kernel
    **0.0695→0.0566ms（1.23×）**，端到端 1.09–1.12×。
  - **改动（单/两文件 device 与 host 逐字一致）**：① 取消 `BM=32` 分支（`--bm=32` 仍可覆盖）；
    `BN=64` 判据改为 `S≥4096 || grid≤256 || grid>600`（`256<grid≤600` 保留 `BN=32`，避开
    `BN=64` 105KB smem→2 CTA/SM 的「2 个波」坏量化点，实测 grid=512 的 S1024/GQA-kv4：
    BN=32 0.2534 vs BN=64 0.2706ms）；② HD=128 时 dQ 是覆盖写 ⇒ `cudaMemset(d_dq_acc)` 加
    `if (D==512)` 守卫（仅 MLA 的 GEMM5 RMW 需要）；③ `convert_kernel` 改 **`float4` 读 + `half2` 写**。
  - **数值与 O5~O10 逐位相同**（fp16 S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、
    GQA kv8 2.008/2.931/3.891e-3、MLA S512H4 2.516/2.916/1.724e-3；bf16 S512 9.001/12.61/13.65e-3、
    S4096 15.10/13.40/16.31e-3）；单/两文件一致。
  - **性能（同 session A/B，event）**：fp16 total S512 0.1255→**0.1121（1.12×）**、main 0.0721→
    **0.0574（1.26×）**；S1024 kv8 0.4555→0.4414、kv1 0.6354→0.6109、kv4(h64) 0.6635→0.6261；
    kv4(h32) 与 S4096 不变。bf16 S512 total 0.1267→**0.1112（1.14×）**、main 1.27×。convert 桶
    S512 0.0157→0.0132、S4096 0.1037→0.0939ms。
  - **ncu（main, S512）**：旧 `(32,32,1)` Duration 83.9µs/occ 10.99%/L1TEX 43.5% → 新 `(64,64,2)`
    **59.7µs / occ 6.23% / L1TEX 20.2% / L2 36.1% / wait 2.00 + long 1.01 + short 0.46**；
    `grid=128<132 SM` ⇒ **尾波/grid-bound**。S=4096（不变）：1.54ms、occ 11.85%、L1TEX 46.6%、
    **L2 73.7%**、Compute 22.5% ⇒ 墙仍是 **L2（dK/dV 原子）+ `wait`**。
  - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）：FA3 MHA S4096 fp16 **0.3255ms/845TF**、
    bf16 0.3193/861；GQA kv8 FA3 0.1217/353 ⇒ ours total 时间 **6.03×**（O10 6.18×）。
  - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o13_{s512,s4096,gqa_kv8}.out.txt`、
    `..._mma_onefile_o13_s512.out.txt`、`..._o13_ncu_main_{s512,s4096}.out.txt`、
    `src/bf16/fa_bwd_bf16_mma_{main,onefile}_o13_*.out.txt`、`src/fa_bwd_o13_fa3_te_baseline.out.txt`；
    文档 `docs/01` §14f、`docs/01b` §6p、`docs/04` §2.1/§2.2/§3。

- 2026-09-23（第四十一轮）：**O9b（第一步）完成（fp16 主 kernel 的 GEMM1/GEMM2 上 Hopper `wgmma`）**。
   - 动机：O9a 把 LSE 的单个 QKᵀ 换成 wgmma，但 LSE 是 epilogue bound、收益有限；主 kernel 的
     墙是 **`wait`（mma 依赖）+ L2（dK/dV 原子）+ 2 CTA/SM**。O9b 把主 kernel 的 `S=QKᵀ`、`dP=dO·Vᵀ`
     （两个最大、最热的 GEMM，A/B 都 K-major、天然 SS）换成 `wgmma.m64n64k16`。
   - **前置冒烟** `src/fp16/fa_bwd_fp16_wgmma_main_smoke.cu`（**PASS，max_abs=0**）：验证把 Q/dO/K/V
     存成 **SW128 K-major** 后，① wgmma QKᵀ；② `dV=Pᵀ·dO` 的 A=P（ATRANS）+ B=dO 从 SW128 转置读；
     ③ `dQ=P·K` 的 B=K 从 SW128 转置读——**`ldmatrix.x2.trans` 能从 SW128 tile 逐位读出转置数据**
     （SW128 只在 16B 粒度置换），于是 GEMM3/4/5 可与 GEMM1/2 共用同一份 SW128 布局。
   - **实现**（单/两文件 device 逐字一致，`diff` 核对 `WGMMA DEVICE BLOCK IDENTICAL`；`#ifdef FA_WGMMA`
     包裹，默认 `sm_90` 构建不编译、行为不变）：新增 `fa_bwd_fp16_wgmma_kernel<HD>`（仅 HD=128/BM=BN=64/
     fp16）——Q/dO/K/V 全 SW128（K 双缓冲、V 单缓冲后段预取）+ 手动 1024B 对齐；`wgmma_mn64_issue` 把
     GEMM1/2 两组异步 mma 一起发、统一 `wait0` 重叠；GEMM3/4/5 仍 mma、B 用新 helper `mma_block_swb`
     从 SW128 读。host 加 `--wgmma=0/1`，自动档 `sel=(64,64)` 且 D=128 才走 wgmma（GQA `BN=32` 回退）。
   - **数值与 O5~O13 逐位相同**：MHA S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3；
     GQA kv4（回退 mma）2.134/3.305/3.850e-3。
   - **性能（同 session A/B，main-only，event）**：mma `(64,64,2)` vs wgmma——S4096 1.5103→
     **1.4435ms（1.046×）**、S512 0.0570→**0.0522ms（1.092×）**；两文件端到端 S4096 total
     **1.8822ms**（`4BS²HD` 口径 73.0 TF，真反向 FLOPs ≈146 TF）。同 session 纯反向 FA3 S4096
     **0.3255ms/844TF**、TE 0.4449/618 ⇒ ours total 时间 **5.8×**（O13 6.03×）。
   - **ncu（main, S4096）**：Duration 1.54→**1.43ms**、**smem 105.5→101.4KB**、**`wait` 1.94→1.50**、
     `long_scoreboard` 1.45→1.25、short 0.46→0.56、**L2 74.4% 仍封顶**、occ 11.88%（仍 2 CTA/SM）。
     **结论**：wgmma 打掉 GEMM1/2 的 `wait`/ldmatrix，但 **GEMM3/4/5 仍是 mma、dK/dV 仍是跨 CTA 原子**，
     L2 与 occupancy 没变 ⇒ 收益真实但有限。**下一步 O9b-2**：GEMM3/4/5 也上 wgmma（P/dS 进 smem/SW128、
     dKV 转置 B 按 FA3 `dKV_swapAB`）＋ TMA 化 K/V ＋ 双缓冲 P/dS 做跨-tile 流水；另补 bf16 版。
   - 原始输出 `src/fp16/fa_bwd_fp16_wgmma_main_smoke.out.txt`、
     `src/fp16/fa_bwd_fp16_mma_main_o9b_{s512,s4096}.out.txt`、
     `src/fp16/fa_bwd_fp16_mma_onefile_o9b_{s512,s4096}.out.txt`、
     `src/fp16/fa_bwd_fp16_mma_main_o9b_ncu_s4096.out.txt`、`src/fa_bwd_o9b_fa3_te_baseline.out.txt`；
     文档 `docs/01` §14g、`docs/04` §2.1/§3。

- 2026-09-23（第四十二轮）：**O9b-bf16 完成（bf16 主 kernel GEMM1/2 上 Hopper wgmma，单/两文件）**。
   - 动机：O9b 第四十一轮只补了 fp16 的「主 kernel GEMM1/2 → wgmma」；bf16 侧此前只有 **O9a**
     （LSE 的单个 QKᵀ 上 wgmma，§6o），主 kernel 5 个 GEMM 仍全是 `mma.m16n8k16`。本轮把 fp16
     O9b 逐字 dtype 参数化到 bf16（同为 2 字节，SW128/描述符/`m64n64k16` 累加器映射逐字节同构，
     仅 `f16.f16`→`bf16.bf16`）。
   - **改动**（单/两文件 device 逐字同源，脚本核对 `device O9b block identical: True`）：新增
     `fa_bwd_bf16_wgmma_kernel<HD>` + `wgmma_mn64_issue`/`mma_block_swb`/`kv_issue_async_sw`/
     `qdo_issue_async_sw`（`#ifdef FA_WGMMA`）；Q/dO/K/V 存 **SW128 K-major**、GEMM1/2 两组
     `wgmma.m64n64k16` 一起 issue/统一 `wait0`；GEMM3/4/5 仍 mma、其转置 B 用 `ldmatrix.x2.trans`
     从同一 SW128 tile 读；host 加 `--wgmma=0/1`，自动档仅 `D==128 && sel==(64,64)` 走 wgmma
     （GQA 的 BN=32 自动回退 mma）。构建：`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a
     -DFA_WGMMA"`。
   - **数值与 O5b~O13 逐位相同**：MHA S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、
     GQA kv4 12.01/21.25/31.56e-3；单/两文件逐指标一致。
   - **性能**（同 session `[O9b A/B]`，CUDA event，main-only）：MHA S=512 mma 0.0571–0.0573→
     **wgmma 0.0524–0.0526ms（1.088–1.089×）**、S=4096 1.4861→**1.4515–1.4552ms（1.021–1.024×）**；
     GQA kv4 S1024 走 BN=32 原 mma 路径（`--wgmma` 不生效）。端到端 bf16 total S=4096 **1.880ms
     （73.1 TF）**、S=512 **0.1085ms（19.8 TF）**、GQA kv4 **0.376ms（45.7 TF）**。
   - **ncu（main, S=4096）**：Duration **1.46ms**、DRAM 4.44% / **L2 72.99%** / L1/TEX 54.26% /
     Compute 26.51%、242 regs / 101.38KB smem → **2 CTA/SM**（occ 11.86%）、Waves 3.88；stall
     `wait` 1.50 + `long_scoreboard` 1.24 + short 0.56，与 fp16 O9b 逐项一致。**bound = L2（残余
     dK/dV 跨 CTA 原子）+ `wait` + 2 CTA/SM**，与 fp16 结论相同。
   - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：MHA S=4096 FA3
     0.3191ms/861TF、TE 0.4418/622、FA2 0.7307/376 ⇒ ours total 时间 **5.89×**；GQA kv4 S1024
     FA3 0.0823ms/417TF。
   - 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o9b_{s512,s4096,gqa_kv4}.out.txt`、
     `..._mma_onefile_o9b_{s512,s4096}.out.txt`、`..._o9b_ncu_s4096.out.txt`、
     `..._o9b_stall_s4096.out.txt`、`src/bf16/fa_bwd_bf16_o9b_fa3_te_baseline.out.txt`；
     文档 `docs/01b` §6q、`docs/04` §2.2/§3。

- 2026-09-23（第四十三轮）：**O9b-2 第一步完成（fp16 主 kernel GEMM3/4/5 上 wgmma：MN-major 转置读；
  数值逐位正确、性能中性）**。
  - 动机：O9b（第四十一轮）只把 GEMM1/2 上 wgmma，收尾墙 = `wait`（GEMM3/4/5 的 mma）+ L2（dK/dV 原子）
    + 2 CTA/SM。O9b-2 要把 GEMM3/4/5（dV=PᵀdO、dK=dSᵀQ、dQ=dS·K）也上 wgmma。
  - **关键手段（本步最大收获）**：把「K-major SW128 存储的 tile」用 **MN-major 描述符 + `tnsp=1`**
    读 = **读它的转置**（FA3 `dKV_swapAB` 同思路）。对手写 `sw128_off` 布局精确推导：LBO=64、
    SBO=(W/64)*64、转置 k16 slab 地址 = `base + s*2*SBO*16`；wgmma 尾部立即数 `p,1,1,tnspA,tnspB`。
    前置冒烟 **`src/fp16/fa_bwd_fp16_wgmma_bwd_smoke.cu`** 对 4 种转置组合逐位 PASS（`max_abs=0`）。
  - **实现**（单/两文件 device 逐字一致，`scripts/sync_onefile_device.py` 核对 `identical: True`）：
    P/dS 改 **SW128 K-major**（4B 直写 `pds_store_sw128`）；5 个 GEMM 全 wgmma（GEMM3/4 用 A=Pᵀ/dSᵀ
    的 MN 描述符、B=dO/Q 的 MN 描述符；GEMM5 的 A=dS 用 K-major、B=K 的 MN 描述符），按 N 半分两遍、
    每遍三条一起发 `wait0`；dQ 寄存器累加重映射 `dqacc[nh][j][q]`。smem **101.4→99.33KB**、230 regs、
    0 spill、仍 2 CTA/SM。host `--wgmma=1`（默认 sm_90 构建不编译，行为不变）。
  - **数值与 O5~O13/O9b 逐位相同**：S=512 1.671/1.771/1.899e-3；S=4096 1.883/1.734/1.966e-3；
    GQA kv4 2.134/3.305/3.850e-3；单/两文件一致。
  - **性能（同 session A/B，main-only，ms）—— 中性偏负**：S=512 mma `(64,64,2)` 0.0584 vs 全 wgmma
    0.0604；S=4096 mma 1.471–1.484 vs 全 wgmma **1.500–1.523**；GQA kv4 mma 0.2482 vs 0.2664。
    **全 wgmma 没有跑赢 mma 最优档，也略慢于 O9b**；端到端 S=4096 total ≈1.91–1.97ms（~72 TF）。
    同 session 纯反向 FA3 S=4096 0.3241ms/848TF、TE 0.4396/625 ⇒ ours total ~6.0×。
  - **ncu（main，`--launch-count 1`）**：S=4096 Duration 1.49–1.54ms、L1/TEX 38.6–40.3%、
    **L2 69.1–71.4%（仍为墙）**、Compute 22.5–23.5%、Tensor 11.6–12.1%、230 regs/99.33KB、occ 11.83%
    （2 CTA/SM）；stall **`wait 1.43 + long 1.96 + short 0.29`**（O9b：`wait 1.50 + long 1.25 + short 0.56`）。
    即全 wgmma 把 GEMM3/4/5 的 `ldmatrix` 也消了（short→0.29），但 **墙不在 GEMM 指令**：L2 的 dK/dV
    跨 CTA 原子 + ~99KB smem 锁死的 2 CTA/SM 才是瓶颈。**结论：O9b-2 的 GEMM3/4/5 wgmma 本身不是杠杆**
    （中性），但它建立了「MN-major 转置读」数据通路，是后续 TMA/流水/去原子的地基。
  - 原始输出 `src/fp16/fa_bwd_fp16_wgmma_bwd_smoke.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_main_o9b2_{s512,s4096,gqa_kv4}.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_onefile_o9b2_{s512,s4096,gqa_kv4}.out.txt`、
    `src/fp16/fa_bwd_fp16_mma_main_o9b2_ncu_{s512,s4096}.out.txt`、
    `src/fp16/fa_bwd_fp16_o9b2_fa3_te_baseline.out.txt`；文档 `docs/01` §14h。

- 2026-09-23（第四十四轮）：**O9b-2-bf16 完成（bf16 主 kernel GEMM3/4/5 也上 wgmma：MN-major
  转置读；数值逐位正确、性能中性）**。
  - 动机：第四十三轮只把 **fp16** 的 GEMM3/4/5（`dV=Pᵀ·dO`、`dK=dSᵀ·Q`、`dQ=dS·K`）用
    **MN-major 描述符（对 K-major SW128 tile 的转置读，FA3 `dKV_swapAB` 同思路）** 上了 wgmma；
    **bf16** 侧此前只到 **O9b**（§6q，GPU1/2 用 wgmma，其余仍 mma）。本轮把 fp16 O9b-2 逐字
    dtype 参数化到 bf16（同为 2 字节，SW128/描述符/u128 步长 `LBO=64`/`SBO=(W/64)*64`
    逐字节同构，仅 `f16.f16`→`bf16.bf16`）。
  - **改动**（单/两文件 device 逐字一致，`scripts/sync_onefile_device.py` 核对 `identical: True`）：
    新增 `wgmma_m64n64k16_bf16_t<TA,TB>` + `trans_k16_addr`/`desc_k16_mn`/`desc_k16_k` +
    `pds_store_sw128`（bf16）；P/dS 改 **SW128 K-major**；5 个 GEMM 全 wgmma（GEMM3/4/5 按 N 半
    `nh=0/1` 分两遍、每遍三条一起发统一 `wait0`）；`dqacc[nh][j][qq]` 寄存器累加；smem
    **99.33KB→2 CTA/SM**（`launch_bwd_wgmma` smem 公式改 `2*BM*BN*sizeof(bf16)`）。
  - **数值与 O5b~O13 的 mma / O9b 逐位相同**（MHA S=512 9.001/12.61/13.65e-3、S=4096
    15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3），单/两文件逐指标一致。
  - **性能（同 session `[O9b A/B]`，event，main-only）**：S=512 mma 0.0568→全 wgmma 0.0592ms
    （0.960×）、S=4096 1.4896→**1.5221ms（0.979×）**、GQA kv4（BN=32 回退）0.2697→0.2698；
    端到端 total S=4096 **1.9239ms（71.4 TF）**、S=512 0.1148ms、GQA kv4 0.3734ms（46.0 TF）。
    **全 wgmma 未跑赢 mma 最优档**（与 fp16 O9b-2 结论一致）。
  - **ncu（main, S=4096）**：Duration **1.48ms**、DRAM 4.35% / L1TEX 40.05% / **L2 71.60%** /
    Tensor 12.03% / Compute 23.37%、230 regs / 99.33KB / occ 11.86%（**2 CTA/SM**）；
    `short_scoreboard` **0.29**（O9b 的 0.56→0.29，GEMM3/4/5 的 ldmatrix 被打掉），
    但 **L2 仍 ~70%、occupancy 仍 2 CTA/SM** ⇒ 墙 = **dK/dV 跨 CTA 原子 + 99.33KB smem**，
    不在 GEMM 指令。**与 fp16 O9b-2 ncu 逐项一致**（Duration 1.49ms、L1TEX 40.25%、L2 71.38%、
    Tensor 12.09%、short 0.29）。
  - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：MHA S=4096 FA3
    **0.3206ms/858TF**、TE 0.4368/629、FA2 0.7307/376 ⇒ ours total 时间 **6.00×**（FA3）；
    GQA kv4 S=1024 FA3 0.0823ms/417TF ⇒ 4.54×。
  - 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o9b2_{s512,s4096,gqa_kv4}.out.txt`、
    `..._mma_onefile_o9b2_{s512,s4096}.out.txt`、`..._o9b2_ncu_{s512,s4096}.out.txt`、
    `src/bf16/fa_bwd_bf16_o9b2_fa3_te_baseline.out.txt`；文档 `docs/01b` §6r。

- 2026-09-23（第四十五轮）：**O7e 完成（fp8 main：fold 4B 向量化写 + REGDQ 下关 O3 预取；
  重定位瓶颈：L1/TEX 才是墙、O7b 不再是头号杠杆）**。
  - **先重新定位瓶颈**（本轮最大价值）：用 `MemoryWorkloadAnalysis_Tables` 拆 O7 后的 fp8 main，
    发现 **L1/TEX 71.07% 才是第一墙**（L2 已被 O4c/O7 从 81%/69% 压到 **43.74%**、Compute 39.1%、
    DRAM 2.6%）。两项最突出：① **register spill 占 L1TEX sector 12.09%**（`REGDQ` 的
    `dqacc[2][8][4]`=64 fp32 + O3 寄存器预取 16 uint32 抢 168-reg/3-CTA 预算，ptxas 强制 spill）；
    ② **shared store bank conflict 3.5-way / 占 store wavefront 69.8%**（fold 段 `Ap/dS3/dS2`
    全是逐 1 字节 `st.shared.u8`）。⇒ **O7b（去 dK/dV 跨 CTA red）针对的 L2 墙已不是头号杠杆**，
    真正的杠杆是 L1/TEX 的数据通路（`ldmatrix`/转置副本/smem 访存）⇒ 下一步转 **fp8 `wgmma`（O9c）**。
  - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
    ① fold 里每线程的 16 个元素在 smem 中连续（`Ap/dS3`：`m=sub4*16+t`；`dS2[m][j]`：
    `j=sub2*16+t`），**16 次 1B 写折成 4 次 4B `st.shared.u32`**（`QTS/DSS2` 均 4 对齐），
    store 指令 ÷4、**store 冲突 68.6M→36.3M（−47%）**；② 新增 `kPrefetch = use_prefetch && !kRegDq`，
    **`REGDQ` 生效时关 O3 预取**（退回 O4b 的 4B 同步读，让出 16 regs）；小 S 仍保留预取。
  - **数值与 O7/O4b/O11 逐位相同**（S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；
    S4096 2.635/2.643/3.216e-1；MLA S1024H2 2.232/3.337/3.602e-1），单/两文件逐指标一致。
  - **性能（event，同 session A/B）**：main **S=4096 2.6135→2.5215ms（1.036×）**、端到端
    3.3687→**3.2661ms（42.1 TF）**；S=1024H32 0.4604→0.4414（~1.04×）、S=512/MLA 持平
    （走原路）。main-only S=4096 54.5 TF（峰值 2.76%）。
  - **ncu（main, S=4096）**：Duration 2.66→**2.57ms**、**L1/TEX 71.07→66.06%**、spill 占 L1TEX
    sector 12.09→**5.74%**、shared store 冲突 **68.6→36.3M**；L2 43.74→45.28%、Compute 39.1→40.9%、
    168 regs / 70.66KB / 3 CTA/SM 不变。**墙仍是 L1/TEX（`ldmatrix`+smem）**。
  - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`，FA2/FA3/TE 三列）：MHA S=4096
    FA3 fp16 0.3245ms/847TF、TE fp16 0.4441/619、FA2 0.7286/377；bf16 FA3 0.3201/859；
    TE FP8 0.5899ms/466TF ⇒ ours fp8 main 2.5215ms = TE FP8 整条反向的 ~4.3×
    （S=512 main 0.072ms 已快过 TE FP8 0.101ms）。
  - 顺带修 `scripts/sync_onefile_device.py` 的单文件 device 区**结束边界**（fp8 的 host 头
    `#include <algorithm>` 在 `struct NpyF32` 之前，旧脚本会误覆盖导致单文件编译失败）。
  - 原始输出 `src/fp8/fa_bwd_fp8_o7e_sweep.out.txt`（单/两文件 ×4 shape 计时+对拍）、
    `src/fp8/fa_bwd_fp8_main_o7e_ncu_tables_s4096.out.txt`、
    `src/fp8/fa_bwd_fp8_main_o7b_base_ncu_full_s4096.out.txt`（O7 基线）、
    `src/fa_bwd_o7e_fa3_te_baseline.out.txt`；文档 `docs/03` §20、`docs/04` §2.3。

- 2026-09-23（第四十六轮）：**O9c 第一步完成（fp8 Hopper `wgmma` 数据通路：冒烟 + LSE 上验证）**。
   - 动机：O7e（第四十五轮）用 ncu 把 fp8 main 第一墙定位为 **L1/TEX 66–71%（`ldmatrix`+smem）**，
     只有 wgmma 的 SS 直读 smem 能同时消 `ldmatrix` 并压 smem。O9c 与 fp16/bf16 的 O9 系列同构：
     本步先在**风险最小的 LSE（单 GEMM、无转置、无 scale 折叠）**上建立 fp8 的 SW128 +
     `wgmma.m64n64k32` 数据通路，主 kernel 上 wgmma 留作 O9c-2。
   - **前置冒烟 `fa_bwd_fp8_wgmma_smoke.cu`**：最小 `C=Q·Kᵀ`（[64][128] fp8）覆盖
     `m64n64k32.f32.e4m3.e4m3`（QKᵀ 口径）与 `...e5m2.e4m3`（GEMM2 口径）；用
     `sw128_off_fp8`/`make_desc_sw128_fp8`/`sw128_k32_addr`，累加器映射与 mma `acc[j][q]` 同构。
     实测两种组合 **max_abs=0.000e+00（逐位 PASS）**。
   - **实现**（单/两文件 device 代码同源逐字一致，`sync_onefile_device.py` 核对 `identical: True`；
     `#ifdef FA_WGMMA` 包裹，默认 sm_90 构建不含、行为不变）：新增 fp8 SW128 布局/描述符、两条
     `wgmma.m64n64k32` asm（fp8 尾部 `p, scaleA, scaleB`，scale 取 1、rowwise scale 在 epilogue 乘）、
     `lse_mma_kernel_bal_wgmma<HD,PIPE>`（与 O11 `lse_mma_kernel_bal` 数学完全一致：同 E4E4、
     online-softmax、4-lane `shfl`、镜像配对 + `cp.async` 双缓冲，只把 4 warp×m16n64 的 mma 换成
     1 warpgroup 的 4 条 wgmma）；`Fp8Cfg` 加 `lse_smem_bytes_balw0/1`。host 加 `launch_lse_bal_wgmma`
     + CLI `--lsewgm` + O9c A/B 段（单文件 host 同步改）。
   - **性能（同 session A/B，event）**：LSE-only mma→wgmma（双缓冲）S=512 0.0389→**0.0365（1.065×）**、
     S=1024H32 0.0791→**0.0691（1.144×）**、S=4096 0.3437→**0.2697（1.275×）**、GQA kv4
     0.0779→**0.0677（1.150×）**、MQA kv1 0.1013→**0.0778（1.302×）**。端到端 total S=512
     0.1780→**0.1690**、S=1024H32 0.7064→0.6902、S=4096 3.2943→**3.1925ms（43.05 TF）**、
     GQA kv4 0.6295→0.6157。**单缓冲 wgmma 反而慢**（S4096 0.477 vs 0.344）⇒ 必须配 `cp.async`
     双缓冲（与 fp16 O9a 同）。
   - **数值**：最终 dq/dk/dv vs ref 与 O7e 同水平（S=512 2.426/2.975/3.736e-1；S=1024H32
     2.400/4.195/3.535e-1；S=4096 2.634/2.643/3.217e-1；GQA kv4 2.517/5.399/7.065e-1）；LSE 本身
     vs mma 版 max_abs 5.6–7.4e-4（fp32 求和次序差，远小于 fp8 容差）。单/两文件逐指标一致。
   - **ncu（lse, S=4096）**：mma Duration 355.87µs / Compute 61.59% / L2 15.05% / 77 regs /
     Block Limit 6 / 理论 occ 37.5% → wgmma **279.07µs / Compute 59.45% / L2 12.48% / 64 regs /
     Block Limit 8 / 理论 occ 50%**；主 stall 都是 fixed-latency `wait`（2.4→2.2 cyc）⇒
     **bound = Compute ~60% + softmax epilogue + 网格不足一个波**（与 fp16 O9a 一致），
     LSE 端到端收益有限（1.02–1.05×）。
   - **对标（同 session）**：TE FP8 纯反向（`fa_bwd_bench.py bench --dtype fp8`）S=512 0.1010ms、
     S=1024H32 0.2059、S=4096 0.5894、GQA kv4 0.2009、MQA kv1 0.4008 ⇒ ours（`--lsewgm`）端到端
     ours/TE = **1.67× / 3.35× / 5.42× / 3.06× / 2.65×**（S=4096 由 O7e 的 5.56×→5.42×）。
     FA2/FA3/TE fp16 三列（`fa_vs_te_bwd_only.py fp16`）：MHA S=4096 FA3 **0.3238ms/849TF**、
     TE 0.4443/619、FA2 0.7251/379。
   - **局限/下一步**：本步只换 LSE，**主 kernel 仍是 `mma.m16n8k32`（第一墙 L1/TEX 66–71% 未动，
     main 占端到端 79%）** ⇒ **O9c-2** 把 GEMM1/2（`S=QKᵀ`、`dP=dO·Vᵀ`）上 `wgmma.m64n64k32`
     （Q/dO/K/V 存 SW128；GEMM3/4/5 的 B 从同一 tile 用 `ldmatrix.x2.trans` 转置读，对齐 fp16 O9b）。
   - 原始输出 `src/fp8/fa_bwd_fp8_wgmma_smoke.out.txt`、
     `src/fp8/fa_bwd_fp8_o9c_lse_sweep.out.txt`（两文件 ×5 shape × default/`--lsewgm` + O9c A/B）、
     `src/fp8/fa_bwd_fp8_main_o9c_ncu_lsewgm_s4096.out.txt`、`..._ncu_lse_mma_s4096.out.txt`、
      `src/fp8/fa_bwd_fp8_o9c_tebench.out.txt`、`src/fp8/fa_bwd_fp8_o9c_fa3_te_baseline.out.txt`；
      文档 `docs/03` §21、`docs/04` §2.3。

- 2026-09-23（第四十七轮）：**O9c-2 第一步完成（fp8 主 kernel GEMM1/2 上 `wgmma.m64n32k32`，
  单/两文件；main 1.04–1.06×、数值与 mma 版同量级）**。
   - 动机：O9c（第四十六轮）只在 LSE 上 wgmma，收益有限；第一墙（ncu L1/TEX 66–71%）在**主
     kernel**（main 占端到端 79%）。O9c-2 把 GEMM1/2（`S=scale·QKᵀ`、`dP=dO·Vᵀ`，A/B 都
     K-major）换成 wgmma。
   - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     `fa_bwd_fp8_mma_kernel` 增加模板开关 `bool WGMMA=false`（默认路径不变）。WGMMA 模式下
     `Fp8Cfg::qs_sw_bytes/ks_sw_bytes` 让 Q/dO/K/V 从行主序 `ASLD` 改为 **SW128 K-major tile**
     （`sw128_off_fp8` 4B 写、动态 smem 手动 1024B 对齐、宿主多给 1024B slack）；
     **smem 70.66→68.61KB**。新增 `wgmma.m64n32k32.f32.{e4m3,e5m2}.e4m3` asm + issue-only
     `wgmma_mn32_issue<KIND>`（两条异步 mma 一起发、统一 `wait0`）；累加器与 m64n64/mma.m16n8
     同构。**fold + GEMM3/4/5 + dQ(O7 red) 全部逐字复用 mma 版**（GEMM3/4/5 的 B 仍是 O4b 的
     K 配对布局 + `ldmatrix.x2.trans`）。host 加 `--wgmma`（HD=128，REGDQ 两档）与
     `[O9c-2 A/B]` 同 session 计时 + 逐元素对拍。
   - **数值**（ours-vs-ref，fp8 causal，max_abs）：S512 2.426/2.976/3.732e-1；S1024H32
     2.399/4.176/3.535e-1；GQA kv4 2.517/5.338/7.178e-1；S4096 2.635/2.644/3.216e-1——
     **与 O4c/O7e mma 版同量级**（无系统误差）。`max_abs(wgmma-vs-mma)` = 4.0e-2/1.5e-1/2.6e-2
     （累加次序不同使 fp32 S/dP 略变、fold 的逐行量化偶有跨档）；单/两文件逐指标一致。
   - **性能**（同 session A/B，main-only，event）：S512 0.0803→**0.0773ms（1.038×）**、
     S1024H32 0.4553→**0.4326（1.052×）**、GQA kv4 0.4365→**0.4130（1.057×）**、
     S4096 2.5891→**2.4508ms（1.056×）**。端到端 total S512 0.1757（12.2 TF）、S1024H32
     0.6799（25.3）、GQA 0.6024（28.5）、S4096 **3.1374ms（43.8 TF）**。同 session TE FP8
     0.1009/0.5893ms ⇒ 端到端 ours/TE = **1.74× / 5.32×**（O7e 5.56×）。同 session 纯反向
     FA3 MHA S4096 0.3237ms/849TF、TE 0.4397/625、FA2 0.7290/377。
   - **ncu（main, S=4096, `--wgmma`, REGDQ=1）**：Duration 2.57→**2.46ms**、**L1/TEX
     66.06→65.19%**、L2 45.28→48.03%、Compute 40.86→40.23%、DRAM 2.82%、168 regs / 68.61KB /
     **3 CTA/SM**、occ 18.11%、Waves 10.34；stall `wait 1.51 + short 1.49 + long 1.18 +
     not_selected 0.35`；shared load 冲突 49.4M、store 39.0M。**结论**：wgmma 只打掉 GEMM1/2
     的 `ldmatrix`，**第一墙仍是 L1/TEX（GEMM3/4/5 的 `ldmatrix` + fold 的 smem）**；要再降得把
     GEMM3/4/5 也上 wgmma。
   - **局限/下一步**：fp8 的 GEMM3/4/5 B 是转置操作数，**fp8 的 `ldmatrix.x2.trans` 要求沿 K
     配对存储（§17）与 SW128 的「沿 K 连续 16B」不兼容**，故本轮保留配对布局。后续：① 推导
     fp8 `wgmma` 的 **MN-major 描述符转置读**（fp16 O9b-2 已验证）并冒烟；② TMA 化 Q/K/V/dO
     直写 SW128（省寄存器预取/地址运算）。
   - 原始输出 `src/fp8/fa_bwd_fp8_main_o9c2_sweep.out.txt`（两文件 ×4 shape × `--wgmma` + A/B）、
     `src/fp8/fa_bwd_fp8_main_o9c2_ncu_s4096.out.txt`、
     `src/fp8/fa_bwd_fp8_main_o9c2_ncu_stall_s4096.out.txt`、
     `src/fp8/fa_bwd_fp8_o9c2_tebench.out.txt`、`src/fa_bwd_fp8_o9c2_fa3_te_baseline_fp16.out.txt`；
     文档 `docs/03` §22。

- 2026-09-23（第四十八轮）：**O12 完成（fp8 主 kernel LSE/D 预装寄存器，main 1.04–1.13×）；
  并查证 O9c-2b 为硬件阻塞**。
   - **O12（O7c-PREL for fp8）**：fp16/bf16 早在 O7c（第三十五轮）就把 `lse`/`delta` 预装寄存器，
     但 fp8 的 GEMM1/2 epilogue 一直**按 `qi` 逐元素 global 读**（ncu：S=4096 uncoalesced global
     多余扇区 38.0M / 23%、global load 仅 9.8/32 B/sector）。新增模板开关 `PREL=true`（`--prel=0`
     可关），在 `nt` 循环前把本线程行槽（mma 4 个 / wgmma 2 个）的 LSE/D 装进 `lse_r[4]/del_r[4]`，
     epilogue 用寄存器；WGMMA 与 mma 两路径共用。单/两文件 device 逐字一致（`sync_onefile_device.py`
     核对 `identical: True`）。
   - **数值**：`PREL` on/off 逐元素 dq 差 4.8e-7–1.1e-3（dQ 跨 CTA `atomicAdd` 次序，非逻辑差）；
     **vs fp32 ref 与历史逐位一致**（S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；
     S4096 2.635/2.643/3.216e-1；GQA kv4 2.517/5.408/7.072e-1；MQA kv1 4.097e-1/1.519/2.127；
     MLA S1024H2 2.232/3.337/3.602e-1）。单/两文件一致。
   - **性能（同 session A/B，main-only）**：S512 0.0719→**0.0690（1.042×）**、S1024H32
     0.4374→**0.4057（1.078×）**、S4096 2.4950→**2.2679（1.100×）**、GQA kv4 1.079×、
     MQA kv1 1.108×、MLA S1024H2 1.010×；`--wgmma` 路径 S4096 2.3851→**2.1204（1.125×）**。
     `--wgmma` 端到端 S4096 **2.872ms（47.9 TF）**、S1024H32 0.6502（26.4）、S512 0.1735（12.4）。
   - **ncu（main,S=4096,mma）**：Duration 2.60→**2.34ms**、Executed Instructions 1008.7→**923.7M
     （−8.4%）**、**uncoalesced global 38.0M→4.70M（−87.6%，23%→5%）**、L1/TEX 65.9%/L2 48.1%/
     Compute 40.9%、regs 168/occ 18.1% 不变。新墙仍 = **L1/TEX 66% + 残余 L2（dK/dV red）**。
   - **O9c-2b 阻塞**：查证 fp8 wgmma 无转置操作数（CUTLASS 全为 `_SS_TN`、无 `tnsp`），
     MN-major 转置读在 fp8 上不存在；物理转置需 `BM=BN=128`、smem >110KB。加之 fp16 O9b-2
     实测 GEMM3/4/5 wgmma 中性偏负 ⇒ **不再尝试**（写入「阻塞」，`docs/03` §23.6）。
   - 原始输出 `src/fp8/fa_bwd_fp8_main_o12_sweep.out.txt`、
     `src/fp8/fa_bwd_fp8_mma_onefile_o12_sweep.out.txt`、
     `src/fp8/fa_bwd_fp8_main_o12_wgmma_sweep.out.txt`、
      `src/fp8/fa_bwd_fp8_main_o12_ncu_{prel0,prel1}_s4096.out.txt`、
            `src/fp8/fa_bwd_fp8_o12_tebench{,_req}.out.txt`；文档 `docs/03` §23。

- 2026-09-23（第四十九轮）：**O14 完成（fp8 输入量化 warp-per-row 向量化 + convert float4；
  端到端 1.05–1.09×）**。
   - 动机：O12 后端到端（S=4096）分解 main 76% / preprocess 13% / **quant 6%** / convert 5%；
     S=1024H32 时 quant(0.10) + convert(0.06) 占 **~24%**。旧 `quantize_row_kernel` 是
     **每行一个 CTA**（128 线程扫 D=128/512 个元素 + `__shared__ sh[128]` + 7 次
     `__syncthreads`），S=4096 grid=65536、每 CTA 只搬 512B，比带宽下限慢 ~6.5×。
   - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     新增 `quantize_row_warp_kernel<VPT,E5M2>`——**每 warp 一行**、lane `float4` 读
     `D/32` 个元素进寄存器、`__shfl_xor_sync` 树求 amax、`uchar4` 写回，**无 smem/无 barrier**；
     只实例化 `D%4==0 && D/32∈{4,16}`（128/512），其它回退旧 kernel。行 amax 用 `fmaxf`
     （可交换结合）⇒ **8 个量化输出逐字节 bitwise mismatch=0**。另把 `convert_kernel` 改
     `float4`（**中性**，S4096 0.146→0.138ms，session 噪声内，留作负结果）。
   - **性能（同 session A/B，event）**：quant（q/k/v/dO 一组）**S512 0.0324→0.0151（2.15×）**、
     **S1024H32 0.1006→0.0413（2.44×）**、**S4096 0.1825→0.0689（2.65×）**、GQA kv4
     0.0623→0.0281（2.22×）、MQA kv1 0.1002→0.0410（2.44×）、MLA S256H2 0.0140→0.0115（1.22×）、
     MLA S1024H2 0.0206→0.0142（1.46×）。端到端 total **S512 0.1776→0.1642（1.082×，13.1 TF）**、
     **S1024H32 0.6737→0.6180（1.090×，27.8 TF）**、**S4096 3.0617→2.9210ms（1.048×，47.1 TF）**、
     GQA kv4 0.5969→0.5660（1.055×）、MQA kv1 1.0137→0.9564；MLA 持平。同 session TE FP8
     0.1006/0.2056/0.5861/0.2021/0.4024 ⇒ 端到端 ours/TE = **1.63×/3.01×/4.98×/2.80×/2.38×**
     （O12 S4096 为 5.22×）。
   - **ncu（quant，S=4096，单 launch）**：Duration 46.18→**15.42µs**、**DRAM 24.29→71.31%**、
     L1/TEX 73.73→26.50%、Compute 71.80→51.49%、**Executed Instructions 34.6M→7.93M（−77%）**、
     Waves 31.03→7.76 ⇒ 墙从 **smem 归约 + 标量加载** 移到 **DRAM 带宽（已到 elementwise 上限）**。
   - **数值与 O12 逐位相同**（S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；
     S4096 2.635/2.643/3.216e-1；GQA kv4 2.517/5.408/7.072e-1；MQA kv1 4.097e-1/1.519/2.127；
     MLA S1024H2 2.232/3.337/3.602e-1），单/两文件一致。
   - 原始输出 `src/fp8/fa_bwd_fp8_main_o14_sweep.out.txt`、
     `src/fp8/fa_bwd_fp8_mma_onefile_o14_sweep.out.txt`、
     `src/fp8/fa_bwd_fp8_main_o14_ncu_quant_{old,new}_s4096.out.txt`、
      `src/fp8/fa_bwd_fp8_o14_tebench{,_base3,_req}.out.txt`、
      `src/fp8/fa_bwd_fp8_o14_fa3_te_baseline_fp16.out.txt`；文档 `docs/03` §24、`docs/04` §2.3。

- 2026-09-23（第五十轮）：**O7e-2 完成（fp8 main fold 的 shared-load bank conflict 修复，
  单/两文件；main 1.02–1.04×、shared load 冲突 −53.5%）**。
   - 动机：O7e/O9c-2/O12 之后 fp8 main 第一墙仍是 **L1/TEX ~65%**，但一直没拆开「读」与「写」。
     本轮用 `--set full` 的 `Memory Workload Analysis Tables` 拆开：**shared loads 93.1M 请求、
     62.8M bank conflict（2.1-way，占 load 波前 32%）** 才是第一来源（shared store 冲突 35.9M
     是第二）。**先证伪**「写指令数」这条路：只把 fold 写从 4B 折成 16B（`st.shared.v4.u32`）
     仅 1.007×（S4096）、S512 持平、且增大 spill ⇒ 墙在**读冲突**。
   - **根因**：fold 的 Ap/dS3 段读 `Ps[m*PSS+j]`（`PSS=33`）的 bank = `(m+j) mod 32`，原按
     `m=sub4*16+t` 分工 ⇒ 4 个 lane 组的 m 相差 16、`16*PSS≡16 (mod32)` ⇒ `sub4=0/2`、`1/3`
     两两同 bank，**恒 2-way conflict**。数学上 `16*PSS mod32 ∈ {0,16}`，改 padding 消不掉，
     必须改 lane→m 映射。
   - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     把每 lane 的 16 个 m 从「`sub4*16+t`」改成「两半 `sub4*8 + t + half*32`」⇒ 固定 `t` 时
     4 组 lane 起始 m 相差 8、bank `{0,8,16,24}+{0..7}` 恰铺满 0..31 ⇒ **无冲突**；`fmaxf`
     可交换结合 ⇒ amax 与量化结果**逐位不变**。输出：每 lane 两段各 8 个连续 m 用
     **8B `st.shared.v2.u32`**、dS2 段用 **16B `st.shared.v4.u32`**。模板开关 `F16B`、
     CLI `--f16b=0` 做同 session A/B。
   - **性能（同 session A/B，event，main-only）**：S512 0.0690→**0.0679（1.017×）**、
     S1024H32 0.4036→**0.3921（1.029×）**、GQA kv4 0.3916→**0.3792（1.033×）**、
     MQA kv1 0.7109→**0.6834（1.040×）**、S4096 2.3115→**2.2180（1.042×）**。端到端 total
     S=4096 **2.8942ms（47.5 TF，ours/TE FP8 4.98×→4.92×）**、S=1024H32 0.6056、S512 0.1632、
     GQA kv4 0.5617、MLA S1024H2 0.5298。同 session 纯反向 FA3 MHA S4096 fp16 0.3242ms/848TF、
     TE 0.4429/621（fp8 无 FA 基线）。
   - **ncu（main, S=4096，同 binary `--f16b` 0/1）**：**shared load 冲突 62.77M→29.18M
     （−53.5%）**、总多余 wavefronts 79.87M→**41.53M**、**L1/TEX 64.80→59.97%**、
     Duration 2.41→**2.27ms（−5.8%）**；L2 47.0→49.9%、Compute 40.6→41.8%、regs 168 / occ 18.08%
     / Waves 10.34 不变；shared load 请求数不变（证明是映射而非请求数）。**新墙 = L1/TEX 60%
     （`ldmatrix` 读 + fold 残余）+ L2 50%（dK/dV 跨 CTA red）+ 寄存器 spill（~2.7M local）**。
   - **数值与 O7/O12/O14 逐位一致**（S512 2.426/2.975/3.735e-1；S1024H32 2.400/4.195/3.536e-1；
     S4096 2.635/2.643/3.216e-1；GQA kv4 2.517/5.408/7.072e-1；MQA kv1 4.097e-1/1.519/2.127；
     MLA S1024H2 2.232/3.337/3.602e-1；A/B max_abs 仅 1e-7 量级，atomic 次序），单/两文件一致。
   - 原始输出 `src/fp8/fa_bwd_fp8_main_o7e2_sweep.out.txt`、
     `src/fp8/fa_bwd_fp8_mma_onefile_o7e2_sweep.out.txt`、
      `src/fp8/fa_bwd_fp8_main_o7e2_ncu_s4096{,_f16b0}.out.txt`、
      `src/fp8/fa_bwd_fp8_o7e2_tebench.out.txt`、`src/fa_bwd_o7e2_fa3_te_baseline_fp16.out.txt`；
      文档 `docs/03` §25、`docs/04` §2.3/§3。

- 2026-09-23（第五十一轮）：**O15a 完成（TMA+SW128 数据通路冒烟，逐位 PASS）+ O16 负结果
  （分段 `wgmma.wait_group` 重叠 epilogue，实测中性）；把 fp16 main 的墙定量钉在 L2 原子**。
   - **先量化墙**：ncu 拆 S=4096 主 kernel（`fa_bwd_fp16_wgmma_kernel<128>`，1.54ms）的 L2 扇区：
     **`red`（dK/dV 跨 CTA `atomicAdd`）= 102,236,160 / 139,841,608 = 73.1%**、`read` 25.6%、
     `write` 1.1%；`L2 Hit 96.2%`、DRAM 4.19%、L1/TEX 48.0%、Compute 23.7%、occ 11.9%（230 regs /
     99.33KB / 2 CTA/SM）、Waves 3.88；stall `long 2.01 + wait 1.43 + barrier 0.84 + short 0.29`。
     ⇒ **main 是 L2 原子字节数 bound**，任何只改「搬运/等待」的优化都动不了它。
   - **O15a（TMA 通路）**：新增 `src/fp16/fa_bwd_fp16_tma_smoke.cu`——`cuTensorMapEncodeTiled`
     (`CU_TENSOR_MAP_SWIZZLE_128B`) + `cp.async.bulk.tensor.2d` + mbarrier 把 [64][128] fp16
     tile 搬进 smem，**逐字节比对 == kernel 的 `sw128_off` K-major SW128**，再用 `wgmma.m64n64k16`
     消费算 `QKᵀ`。实测 `byte-mismatch=0`、`max_abs=0.000e+00`、**PASS**。
     **关键发现**：TMA box 内维 128B = fp16 的 **64 元素** ⇒ **HD=128 的 K-major tile 必须拆成
     2 个 K=64 chunk**（两块独立 8KB、wgmma 用两个 `SBO=1024` 描述符），因为 SW128 canonical
     `[rg][kg][rr][kk]` 无法由 TMA 的连续 box 一次写出（同 kernel-opt 23/28 篇「BK 锁 64」）。
   - **O16（负结果）**：给 wgmma 主 kernel 加模板开关 `OW` + CLI `--ow=`，用 `wait_group<1>`
     （S=QKᵀ 完成即做 P epilogue，与仍在飞的 dP 重叠）和 `wait_group<2>/<1>/wait0`（dV/dK/dQ
     逐个收，red 与后一条 wgmma 重叠）；**只改等待时机**。同 session A/B：S=4096 `wait0` 1.5124
     → `wait_group` 1.5064（1.004×）/ 另一 session 1.5062→1.5193（0.991×）⇒ **中性**；
     `max|diff| dq=0`（逐位）、`dk/dv ~9e-5~1.4e-4`（仅 atomic 次序）。**`--ow=` 默认关**、保留 A/B。
   - **结论/下一步**：这轮把 fp16/bf16 main 的路线收敛——**唯一真杠杆 = 跨 warpgroup 归约
     （BM=128、2 warpgroups：一个 KV 元素只被 `nblk/2` 个 CTA 贡献，dK/dV red 字节砍半）**；
     TMA 通路已建好但需先把 HD=128 tile 改成 2×K=64 chunk 才能落进主 kernel。见 `docs/01` §14i。
    - 原始输出 `src/fp16/fa_bwd_fp16_tma_smoke.out.txt`、
      `src/fp16/fa_bwd_fp16_mma_main_o16_s4096.out.txt`、`/tmp` ncu 拆解（L2 扇区、stall）；
      文档 `docs/01` §14i、`docs/04` §3、`docs/08` §5。

- 2026-09-23（第五十二轮）：**O17 完成（fp16 跨 warpgroup 归约：BM=128 + 2 warpgroups，
  dK/dV 的 red 字节砍半；main S4096 1.57×）**。
   - 动机：第五十一轮用 ncu 把 fp16 main 的墙钉死——**`red`（dK/dV 跨 CTA `atomicAdd`）占 L2
     扇区 73.1%、DRAM 仅 4.2%** ⇒ L2 原子字节数 bound；O16（错开 wait）与 O7c（float4 归约）
     均证明「动等待/事务数」无效。**唯一杠杆 = 减少每个 KV 元素的贡献 CTA 数**：BM 64→128 后
     `nblk=S/BM` 减半 ⇒ red 字节砍半。
   - **前置冒烟** `src/fp16/fa_bwd_fp16_wgmma2_smoke.cu`：验证对 **[128][*] K-major SW128 tile
     用 MN-major 描述符按 `s=0..7` 读转置**（dV=PᵀdO / dK=dSᵀQ / dQ=dS·K 两个 m64 半），
     三项 **max_abs=0.000e+00（逐位 PASS）**。
   - **实现**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     新增 `fa_bwd_fp16_wgmma2_kernel<HD>`（`#ifdef FA_WGMMA`，仅 HD=128，`__launch_bounds__(256,1)`）。
     `wg=tid>>7`，两组各持自己 64 行 Q/dO/P/dS（SW128 tile 按 8-rowgroup 偏移）；GEMM1/2 各 wg
     `wgmma.m64n64`；中段 barrier 后 **wg0** 做 GEMM3/4——`for nh: for s in 0..7` 把两个 m64 半
     连续喂**同一累加器**（= 全 BM 之和）再 `red_add2`，**wg1 同时做自己的 GEMM5**；GEMM5 各 wg
     寄存器累加后一次 `float2` 写回。`kv_issue_async_sw`/`qdo_issue_async_sw` 加模板 `NT=256`。
     host 加 `launch_bwd_wgmma2` + CLI `--wg2` + `[O17 A/B]`（mma/O9b/wg2 同 session + 逐元素差）。
     smem **149.5KB → 1 CTA/SM**（256 线程=8 warps，与 O9b 的 2×4 相同），regs 200、0 spill。
   - **数值 vs ref 与历史逐位一致**（fp16 causal）：S512 1.671/1.771/1.899e-3；S4096
     1.883/1.734/1.966e-3；GQA h32kv4 S1024 2.134/3.305/3.850e-3；MQA h64kv1 2.292/7.934/7.517e-3。
     `max|diff|` wg2-vs-mma：dq **0**（无跨 CTA 原子）/ dk/dv 8.8e-5–5.9e-3（仅 atomic 次序）。
     单/两文件逐指标一致。
   - **性能（同 session A/B，CUDA event，main-only）**：S512 0.0573→**0.0517（1.11×）**、
     S4096 1.5310→**0.9784（1.57×）**、GQA kv4 0.2636→**0.1789（1.47×）**、
     MQA kv1 0.4326→**0.2812（1.54×）**。端到端 S4096 **1.4268ms（96.3 TF）**、S512 0.1071、
     GQA 0.2925、MQA 0.4555。**main-only S4096 140.5 TF**。同 session 纯反向 FA3 MHA S4096
     0.3241ms/848TF、TE 0.4458/617、FA2 0.7285/377 ⇒ ours 端到端为 FA3 的 **4.4×**（O9b ~6.0×）。
   - **ncu（main, S=4096，同 session O9b vs O17）**：`lts__t_sectors_op_red`
     **102,236,160 → 51,904,512（0.508×）**、`read` 35.9M→17.95M（0.50×）、`write` 1.576M 不变、
     Duration **1.48→0.996ms**、**L2 71.76→54.72%**、L1/TEX 40.40→36.85%、DRAM 4.36→6.47%、
     Compute 23.58→27.48%、regs 230→200、smem 100.35→149.5KB、occ 11.89→12.41%；bank conflict 0。
     **机制假设被 ncu 完全证实**；新墙仍是 L2（red 占余下 L2 扇区 ~72.6%）⇒ 下一步 O17b（BM=256）。
   - 原始输出 `src/fp16/fa_bwd_fp16_wgmma2_smoke.out.txt`、
     `src/fp16/fa_bwd_fp16_mma_main_o17_s4096.out.txt`、`..._mma_onefile_o17_s512.out.txt`、
     `src/fp16/fa_bwd_fp16_mma_main_o17_ncu_wg2_s4096.out.txt`、`..._o17_ncu_o9b_s4096.out.txt`、
       `src/fp16/fa_bwd_fp16_o17_fa3_te_baseline.out.txt`；文档 `docs/01` §14j、`docs/04` §2.1/§3。

- 2026-09-23（第五十三轮）：**O17-bf16 完成（bf16 跨 warpgroup 归约，BM=128 + 2 warpgroups，
  dK/dV 的 red 字节砍半；单/两文件）**。
   - 把第五十二轮的 fp16 O17 **逐字 dtype 参数化**到 bf16（同为 2 字节，SW128 布局/描述符/
     `wgmma.m64n64k16` 累加器映射逐字节同构，只差 `f16.f16`→`bf16.bf16`）。新增
     `fa_bwd_bf16_wgmma2_kernel<HD>`；给 `kv_issue_async_sw`/`qdo_issue_async_sw` 加模板参数
     `NT=THREADS`（2-wg 版传 256）；**只让 wg0 做 GEMM3/4**，把两个 m64 半（128 行）连续喂同一
     `wgmma.m64n64k16` 累加器 ⇒ 每个 KV 元素只 `red` 一次；wg1 并行做自己的 GEMM5。单/两文件
     device 代码逐字一致（`sync_onefile_device.py` 核对 `identical: True`）；host 加
     `launch_bwd_wgmma2` + CLI `--wg2` + `[O17 A/B]`；默认不变（需 `--wg2`，D=512 忽略）。
   - **数值与 O5b~O13 历史值逐位一致**：MHA S512 9.001/12.61/13.65e-3、S4096
     15.10/13.40/16.31e-3、GQA kv8 12.33/19.30/31.50e-3、GQA kv4 12.01/21.25/31.56e-3、
     MQA kv1 11.90/45.58/71.96e-3；`dq` 逐位相同（无跨 CTA 原子）、dk/dv 仅 atomic 次序
     （max|diff| ~1e-4~6e-3）。单/两文件逐指标一致。
   - **性能（同 session A/B，event，main-only）**：S512 mma 0.0577→**wg2 0.0521（1.109×）**、
     S4096 1.4892→**0.9821（1.516×，139.9 TF）**、GQA kv8 1.504×、GQA kv4 1.506×、
     MQA kv1 1.531×。端到端（`--wg2`）S=4096 **1.4309ms（96.05 TF）**、S512 0.1075ms、
     GQA kv8 0.3334 / kv4 0.2892 / MQA kv1 0.4519ms。
   - **ncu（main, S=4096，同 session O9b vs O17）**：**`lts__t_sectors_op_red`
     102,236,160→51,904,512（0.508×）**、`read` 0.50×、Duration 1.48→**0.997ms**、
     **L2 71.61%→54.63%**、L1/TEX 40.25→36.68%、Compute 23.49→27.36%、regs 230→200 /
     smem 100.35→149.50KB / occ 11.89→12.41%、bank conflict 0 ⇒ **与 fp16 O17 逐项一致，
     机制假设被 ncu 完全证实**；新墙仍是 L2（red 占 ~72.6%）。
   - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：MHA S4096 FA3
     **0.3217ms/855TF**、TE 0.4422/622、FA2 0.7343/374 ⇒ ours total 时间 **4.45×**；
     GQA kv8 FA3 0.1214/354 ⇒ 2.75×；GQA kv4 FA3 0.0825/417 ⇒ 3.51×；MQA kv1 FA3
     0.1567/439 ⇒ 2.88×。
   - 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o17_{s512_h16_d128,s4096_h16_d128,
     s1024_h32_d128_kv4,s1024_h40_d128_kv8,s1024_h64_d128_kv1}.out.txt`、
     `src/bf16/fa_bwd_bf16_mma_onefile_o17_{s512,s4096}.out.txt`、
     `..._o17_ncu_{wg2,o9b}_s4096.out.txt`、`src/bf16/fa_bwd_bf16_o17_fa3_te_baseline.out.txt`；
     文档 `docs/01b` §6s、`docs/04` §2.2/§3。

- 2026-09-23（第五十四轮）：**O17b 尝试完成（BM=256 / 4 warpgroups）—— 负结果 + 寄存器墙**。
   - 动机：O17 的 red 仍占 L2 ~72.6%（O17 ncu），BM 再翻倍应让 red 再砍半。
     新增 `fa_bwd_fp16_wgmma4_kernel<HD,SEQ>`（单/两文件，`#ifdef FA_WGMMA`，仅 HD=128，
     512 线程 = 4 wg）；host `--wg4`/`--wg4seq` + `[O17b A/B]`，默认路径不变。
   - **资源账（先算后做，本项关键结论）**：512 线程 @1 CTA/SM ⇒ `65536/512 = **128 regs/线程**`；
     而 dQ 寄存器累加器（`dqacc[2][8][4]=64`）+ GEMM1/2 两条 wgmma 累加器（各 32，wait0 后都在用）
     已 = 128，必然 spill。smem：Q64+dO64+K16+V16+P32+dS32 = **224KB**（K/V 只能单缓冲）。
   - **机制被证实**：ncu `lts__t_sectors_op_red` **51.90M→26.74M（0.515×，精确减半）**。
     但 **`write` 扇区 1.57M→14.5–34M**（ptxas 报 128 regs + 溢出；ncu：local 占 L2 ~48%，
     每次 spill 只用 1/32 B/sector），read 也 18.0M→28.7–35.1M。
   - **性能（同 session A/B，main-only，ms）**：S512 0.0522→0.0958（0.55×）、GQA kv4 S1024
     0.1779→0.1919（0.93×）、MQA kv1 0.2869→0.3429（0.84×）、S4096 0.9946→1.1956（0.83×）；
     SEQ 版（串行 GEMM1/2 + 读回 half P）消了部分静态 spill、S4096 回到 0.95×，但**读回 half P
     使 dS 多一层 fp16 舍入（dk 差 2.6e-2～7.3e-2）⇒ 不能作正确路径**。ovlp 版数值与 O17 在
     fp16 噪声内（dk/dv 6–9e-5，仅 atomic 次序）。
   - **结论**：BM=256 在本卡上被**寄存器文件**卡死（不是 smem）；「放大 BM」这条路走不通。
     ncu 同时把 O17 与 O17b 同 session 对照（Duration 0.997 vs 1.20ms、L1TEX 48.8 vs 51.8%、
     L2 54.7 vs 40.7%、regs 200 vs 128、occ 12.4 vs 24.9%、Waves 3.88 vs 1.94）。
   - 原始输出 `src/fp16/fa_bwd_fp16_o17b_sweep.out.txt`、
     `src/fp16/fa_bwd_fp16_main_o17_ncu_s4096_samesession.out.txt`、
     `src/fp16/fa_bwd_fp16_main_o17b_ncu_s4096.out.txt`、
      `src/fp16/fa_bwd_fp16_main_o17_o17b_ncu_red_s4096.out.txt`；文档 `docs/01` §14k。

- 2026-09-23（第五十五轮）：**O17-2 完成（fp16/bf16：O17 的 GEMM3/GEMM4 拆分到两个 warpgroup，
  负载再平衡；数值逐位不变）**。
   - 动机：O17（BM=128、2 wg）的 phase B 里**只有 wg0 串行做 GEMM3(dV)+GEMM4(dK)**（4 条串行
     `red` 链），wg1 只做自己的 GEMM5 ⇒ **张量工作量 wg0:wg1 = 3:1**，wg1 在 GEMM3/4 期间
     barrier 空等（ncu `barrier` stall 高）。这正是 O17b（BM=256）已采用、BM=128 版漏掉的
     「dV/dK 分给两个 wg」。
   - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`；
     fp16/bf16 逐字 dtype 同构）：`fa_bwd_{fp16,bf16}_wgmma2_kernel<HD, bool SPLIT=true>`，
     `if constexpr (SPLIT)` 里 `if (wg==0) {/*GEMM3 dV*/} else {/*GEMM4 dK*/}`，两者都仍把两个
     m64 半（`s=0..7`）连续喂**同一累加器**对全 BM=128 归约 ⇒ **每 KV 元素仍只 `red` 一次**
     （red 字节不变）；GEMM5 仍各 wg 算自己 64 行。host 加 `--wg2split=0/1`（默认 1）做同 session A/B。
   - **数值**：split-vs-nosplit `max|diff|` dq=**0**、dk/dv ~3e-5~9e-4（仅 atomic 次序）；
     **vs fp32 ref 与历史逐位一致**（fp16 S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、
     GQA kv4 2.134/3.305/3.850e-3、kv8 2.008/2.931/3.891e-3、MQA kv1 2.292/7.934/7.517e-3；
     bf16 S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3）。
   - **性能（同 session A/B，event，main-only）**：fp16 S512 1.004×、**S4096 1.013×**、
     GQA kv4 **1.021×**、kv8 1.007×、MQA 1.005×；bf16 S4096 **1.038×**、GQA kv4 **1.051×**、
     kv8 1.012×、MQA 1.003×、S512 0.996×（噪声）。端到端 fp16/bf16 S4096 **1.946/1.942ms
     （70.6/70.8 TF）**。
   - **ncu（main, S=4096，同 binary `--wg2split` 0/1）**：**`lts__t_sectors_op_red` 51,904,512
     完全不变**、Duration 997.7→**982.4µs**、**stall `barrier` 1.61→0.46（−3.5×）**、
     `wait` 1.18→1.19、long 0.40→0.75、short 0.26→0.40；regs 200 / smem 148.5KB / occ 12.5% /
     Waves 3.88。⇒ **收益来自消 wg1 的 barrier 空等（3:1→1:1），不是减少原子**；墙仍是
     **L2 red + `wait`**，与 O7b 正交。
   - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）：FA3 MHA S4096 fp16
     **0.3247ms/846TF**、TE 0.4451/618、FA2 0.7275/378；bf16 FA3 0.3204/858 ⇒ ours total
     时间 **5.99×（fp16）/6.06×（bf16）**（O17 5.97×）。
   - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o17_2_sweep.out.txt`、
     `src/fp16/fa_bwd_fp16_mma_onefile_o17_2_s4096.out.txt`、
     `..._o17_2_ncu_{wg2,red}_s4096.out.txt`、`..._o17_nosplit_ncu_red_s4096.out.txt`、
      `src/bf16/fa_bwd_bf16_mma_main_o17_2_sweep.out.txt`、
      `src/bf16/fa_bwd_bf16_mma_onefile_o17_2_s4096.out.txt`、
      `src/fa_bwd_o17_2_fa3_te_baseline_{fp16,bf16}.out.txt`；
      文档 `docs/01` §14l、`docs/01b` §6t、`docs/04` §2.1/§2.2/§3。

- 2026-09-23（第五十六轮）：**O18 完成（fp16：BN=128 版 wgmma2，tile 数减半）**。
  把 O17/O17-2 的 kv-tile 从 `BN=64` 翻到 **`BN=128`**（`m64n128k16`，`docs/01` §14k.7 item 3）——
  per-CTA tile 数减半 ⇒ `__syncthreads`/`cp.async.wait`/wgmma commit-wait 序列减半。前置冒烟
  `fa_bwd_fp16_wgmma2b_smoke.cu` 逐位 PASS；`fa_bwd_fp16_wgmma2b_kernel`（单/两文件，`--wg2bn`，
  230 regs/224KB/1 CTA/SM）。**main MHA S4096 1.029×（142.9 TF）、S512 1.028×**，GQA/MQA
  中性（保持 O17）；ncu Duration 982.4→**951.1µs**、**`red` 逐字节不变**（BN 不动归约结构）。
  详见 `docs/01` §14m、`docs/04` §2.1/§3。

- 2026-09-23（第五十七轮）：**O18-bf16 完成（bf16：BN=128 版 wgmma2，单/两文件）**。
  - 把第五十六轮的 fp16 O18 **逐字 dtype 参数化**到 bf16（同为 2 字节，SW128 布局/描述符/
    `m64n128k16` 累加器映射逐字节同构，仅 `f16.f16`→`bf16.bf16`）：新增
    `wgmma_m64n128k16_bf16_t<TA,TB>` + `wgmma_mn128_issue` + `fa_bwd_bf16_wgmma2b_kernel<HD,SPLIT>`；
    host 加 `--wg2bn`（默认 `SPLIT=1`，GEMM3(dV)→wg0、GEMM4(dK)→wg1，都对全 BM=128 归约）。
    单/两文件 device 代码逐字一致（`sync_onefile_device.py` 核对 `identical: True`）。
  - **数值 vs ref 与历史逐位一致**：MHA S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、
    GQA kv4 12.01/21.25/31.56e-3、kv8 12.33/19.30/31.50e-3、kv4(h64) 13.51/30.91/44.20e-3、
    MQA kv1 11.90/45.58/71.96e-3；`max|diff|` wg2b-vs-wg2 dq `1.8–3.6e-7`、dk/dv `6e-5–8.2e-4`
    （仅 atomic 次序）。单/两文件逐指标一致。
  - **性能（同 session A/B，event，main-only）**：MHA S=512 O17 0.0507→**O18 0.0492ms
    （1.030×，43.6 TF）**、S=4096 0.9799–0.9836→**0.9558–0.9560ms（1.025–1.029×，143.8 TF）**；
    GQA kv4 0.996×（中性）、kv8 1.005×、kv4(h64) 1.002×、MQA kv1 1.007×；串行版(BN128)
    普遍更慢（0.97–1.00×）⇒ 保留 SPLIT。端到端（`--wg2bn`）S=4096 **1.381ms（99.5 TF，
    O17 1.431ms ⇒ 1.036×）**、S512 0.104、GQA kv4 0.291 / kv8 0.332 / kv4(h64) 0.447 / MQA 0.447ms。
  - **ncu（main, S4096, 同 binary `--wg2bn` vs `--wg2`）**：Duration 982.2→**952.5µs（1.031×）**、
    **`lts__t_sectors_op_red` 51,904,512 逐字节不变**、read 18.07M→18.37M、write 1.576M→1.574M、
    L1/TEX 44.2→44.2% / L2 54.6→56.9% / DRAM 4.4→6.7%、Compute 27.4→23.5%、
    regs 200→**255** / smem 148.5→**230.4KB** / occ 12.5%（1 CTA/SM）/ Waves 3.88；
    stall barrier 0.46→0.93、wait 1.19→1.25、long 0.75→**0.60**、short 0.40→0.43；bank conflict 0。
    ⇒ **收益来自 tile 数减半后的 barrier/commit-wait 序列减半**；墙仍是 **L2（red 占 ~72%）+ 1 CTA/SM**。
  - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`，FA2/FA3/TE 三列）：
    MHA S4096 FA3 **0.3209ms/857TF**、TE 0.4423/621、FA2 0.7292/377 ⇒ ours total 时间 **4.30×**
    （O17 4.45×）；GQA kv4 FA3 0.0826/416 ⇒ 3.52×（O17 3.51×）；kv8 FA3 0.1214/354 ⇒ 2.73×；
    kv4(h64) FA3 0.1602/429 ⇒ 2.79×；MQA kv1 FA3 0.1571/437 ⇒ 2.85×。**全 shape 小幅改善**。
  - 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o18_{s512,s4096}.out.txt`、
    `src/bf16/fa_bwd_bf16_mma_main_o18_gqa_{kv4,kv8,kv4h64,kv1}.out.txt`、
    `src/bf16/fa_bwd_bf16_mma_onefile_o18_{s512,s4096}.out.txt`、
    `src/bf16/fa_bwd_bf16_mma_main_o18_ncu_s4096.out.txt`、
    `src/bf16/fa_bwd_bf16_mma_main_o18_ncu_red_{s4096,o17_s4096}.out.txt`、
     `src/fa_bwd_o18_fa3_te_baseline_bf16.out.txt`；文档 `docs/01b` §6u、`docs/04` §2.2/§3。

- 2026-09-24（第五十八轮）：**O7e-3 完成（fp8 main GEMM1/2 epilogue `Ps/Ss` 的 store/回读
  bank conflict 修复；单/两文件；main 1.007–1.024×，数值逐位不变）**。
  - **先拆**：用 `--page source --csv` 按 CUDA 源码行聚合 `L1 Wavefronts Shared Excessive`，
    发现 shared-store 多余 wavefronts 的 **~98%** 来自 GEMM1/2 epilogue 的
    `Ps/Ss[r*PSS+c]` 标量写（25.56M + 12.78M）与 `Ss` 里对 `Ps` 的同模式回读（12.78M，进 LDS）。
    根因：`r=R0+g`（`g=lane>>2`）、`c=C0+2l`（`l=lane&3`），bank=`(g·PSS+2l) mod32`；
    PSS=33（≡1）时 `{g+2l}` 大量重合 ⇒ **4-way**。
  - **改**：`Fp8Cfg::PSS = BN + FA_PSS_EXTRA`（默认 5 ⇒ 37；`FA_PSS_EXTRA=1` 为旧 33，供 A/B）。
    37 仍 ≡1 mod4 ⇒ O7e-2 的 fold 掩码读沿用、无冲突；`bank=(5g+2l)` 降到 **2-way**。
    smem 70.7→72.7KB（仍 3 CTA/SM）；**只改 smem 地址，数值逐位不变**。单文件由
    `sync_onefile_device.py` 同步（`device region identical: True`）。
  - **性能（同 session A/B，event，main/total ×）**：S512 1.016/1.015、S1024H32 1.010/1.005、
    S4096 1.011/1.007、GQA kv4 1.007/1.012、MLA S1024H2 **1.024/1.005**（另一次首测 S4096
    main 2.2832→2.2269 = 1.025）。端到端 S4096 total 2.8585→**2.8375ms（48.4 TF，峰值 2.4%）**。
  - **ncu（S=4096）**：shared store 冲突 29.46M→**12.33M（−58%）**、load 冲突 29.18M→**20.59M
    （−29%）**、L1/TEX 59.97%→**55.83%**、L2 49.8%、Compute 43.1%、DRAM 3.2%、occ 18.1%
    （168 regs / 72.7KB / 3 CTA/SM）、Waves 10.34。**但 Duration 2.27→2.28ms 持平**，
    stall = `wait` **1.56** + `short_scoreboard` **1.50** + `long` 0.77（issue 45.9%）
    ⇒ **证伪「L1/TEX 是限速器」**：真正的墙是 **mma 依赖延迟 + 3 CTA/SM**。
  - 对标（同 session）：TE FP8 S512 0.1008 / S1024H32 0.2061 / S4096 0.5887 / GQA kv4
    0.2013ms；FA3 fp16 MHA S4096 0.3251ms/846TF、GQA kv4 0.0827/415（fp8 无 FA 基线）。
  - 原始输出 `src/fp8/fa_bwd_fp8_o19_pss_ab.out.txt`（2 文件 ×5 shape × PSS 33/37）、
    `src/fp8/fa_bwd_fp8_main_o19_ncu_s4096.out.txt`（`--set full`）、
    `src/fp8/fa_bwd_fp8_o19_tebench.out.txt`、`src/fp8/fa_bwd_fp8_o19_fa3_te_baseline_fp16.out.txt`；
    详见 `docs/03` §26。

- 2026-09-24（第五十九轮）：**O19 完成（fp8 跨 warpgroup 归约，BM=128、2 wg、256 线程）——负结果 + 机制判决**。
  - 动机：O7e-3（§26）把 fp8 main 第一墙定位为 **mma 依赖延迟（`wait`+`short_scoreboard`）+ 3 CTA/SM**，
    候选杠杆列为「降 L2 的 dK/dV red（49.8%）」或「提 occupancy」。fp16/bf16 的 O17 已证明跨 wg 归约能
    把 red 精确砍半、main 1.5×，故把同一机制移植到 fp8 做**判决**。
  - 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：新增
    `fa_bwd_fp8_wg2_kernel<HD,128,32>`（`__launch_bounds__(256,1)`，256 线程 = 2 wg）。phase A 每 wg 算
    自己 64 行 Q 的 S/dP（几何与 BM=64 版同构）；fold 后 GEMM3/4/5 的重叠维**整块 128 行**从 smem 读，
    每个输出元素在本 CTA 内只被一个 warp `red` 一次 ⇒ **跨 CTA red 减半**；dQ 仍走 REGDQ。host 加
    `--wg2`/`--ksplit2=` 与 `[O19 A/B]`。smem **131.33KB → 1 CTA/SM**、regs 217、无 O3 预取（`kv_load_pair_nt`）。
  - **数值**：与 mma 版同为 fp8 噪声（S4096 vs ref dq/dk/dv 2.635/2.767/3.313e-1；dq 的
    `max_abs(wg2-vs-mma)=1.2e-7`，dk/dv 0.04–0.12 仅原子归约次序+贡献 CTA 数不同）。单/两文件逐位一致。
  - **性能（同 session A/B，main-only，ms）**：S4096 mma 2.2662 vs **wg2 2.9014（0.781×）**、
    S1024H32 0.4096→0.5661（0.724×）、S512 0.0761→0.1130（0.673×）、GQA kv4 0.3869→0.5254（0.736×）；
    `ksplit2` sweep 最好 2.886ms，**全线更慢**。
  - **ncu（main, S4096，同 session）**：`lts__t_sectors_op_red` **108.48M→64.29M（0.593×）**、
    `read` 32.19M→17.29M（0.537×）、**L2 49.91%→22.61%**（L2 那一半确实被打掉），**但** achieved occ
    18.09%→**12.49%（8 vs 12 warp/SM）**、Compute 42.34→34.13%、issue 43→34%、No Eligible 54→64%、
    Duration 2.28→**2.92ms**；stall wait/short/long 1.56/1.50/0.77→1.27/1.21/0.40。
  - **结论（判决 O7e-3 的开放问题）**：**fp8 main 的墙不是 L2 red，而是「mma 依赖延迟 + occupancy」**——
    把 L2 打掉一半也补不回 1 CTA/SM 的并行度损失。fp8 右侧**不应再走「减 red / 放大 BM」**，真正杠杆是
    **提 occupancy**（168 regs→≤128、72.7KB smem→≤58KB 才 4 CTA/SM）或**减 mma 依赖 stall**（softmax/fold
    与 mma 的重叠）。`red` 减半只在 fp16/bf16（red 占 L2 73%、且 2→1 CTA/SM 换得回来）成立。
   - 原始输出 `src/fp8/o19_wg2_ab_sweep.out.txt`（同 session A/B ×4 shape + 数值）、
     `src/fp8/o19_ncu_wg2_s4096.out.txt`（wg2 `--set full`）、
     `src/fp8/o19_ncu_wg2_red_s4096.out.txt` / `src/fp8/o19_ncu_mma_red_s4096.out.txt`（red/stall 对照）；
     详见 `docs/03` §27、`docs/04` §2.3。

- 2026-09-24（第六十轮）：**O20 完成（fp8 mma 主路径 GEMM1/GEMM2 epilogue 融合，消 `Ps` 回读；
  S4096 main 1.026×）**。
  - 动机：O7e-3（§26）把 fp8 main 的 shared-store 多余 wavefronts 拆到源码行时发现，除了
    `Ps/Ss` 写的 4-way 冲突外，**`Ss` 对 `Ps` 的回读（12.78M 多余 wavefronts）本身没被消掉**。
    mma 路径里 GEMM1 epilogue 写 `Ps`、GEMM2 epilogue 又读回同一 `(r,c)`——而两次 `mma_block` 的
    线程/累加器映射完全一致（O4a 已证），这个 `P` 本就在**本线程寄存器**里。wgmma 路径（O9c-2）
    早已用 `pval[16]`，只有 mma 路径还在绕 smem。
  - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：新增编译
    开关 `FA_FUSE_EPI`（默认 1，`-DFA_FUSE_EPI=0` 供 A/B）——mma 分支 GEMM1 前声明
    `preg[2][2][4]`，GEMM1 epilogue 写 `Ps` 的同时存 `preg`，GEMM2 epilogue 直接用 `preg`。
    只改 smem 访问路径，**数值逐位不变**（同线程、同 `(r,c)`、同 `p`）。
  - **数值**：FUSE on/off 的 dq/dk/dv **逐位相同**（S512 2.426/2.975/3.735e-1；S1024H32
    2.400/4.195/3.536e-1；GQA kv4 2.517/5.408/7.072e-1；kv8 2.869/5.390/7.108e-1；
    MLA S1024H2 2.232/3.337/3.602e-1；S512H4 2.415/2.992/4.481e-1；S4096 2.635/2.643/3.216e-1）。
  - **性能**（同 session A/B，event，main-only）：**S4096 2.1997→2.1433ms（1.026×）**、
    S1024H32 1.015×、kv8 1.017×；S512/kv4/MLA 中性（0.99–1.01×，`preg` 保活略增 spill）。
    端到端 S4096 **total 2.77ms（49.6 TF，峰值 3.2% 口径按 main-only 64.1 TF）**；同 session
    TE FP8 0.5909ms/465TF ⇒ 端到端时间约 **4.7× TE**（O7e-3 4.85×）。
  - **ncu（main, S=4096，同 session，`--set full -c 1` + 定向 metrics）**：Duration 2.25→**2.21ms**、
    **shared 多余 wavefronts 15.97M→11.71M、总 wavefronts 169.07→160.55M**、
    **`op_ld` bank conflict 20.55M→16.54M（−19.5%）**、**`short_scoreboard` 1.50→1.41**、
    long 0.76→0.68；regs/smem/occ 不变（168/72.70KB/18.75%）。**证实收益来自消 `Ps` 回读**；
    墙仍是 **`wait` 1.55（mma 依赖延迟）+ 3 CTA/SM + 残余 L2 red**，与 O7e-3/O19 判决一致。
  - **对标**（同 session）：FA3 fp16 MHA S4096 0.3241ms/848TF、TE fp16 0.4451/618（fp8 无 FA 基线，
    仅口径参照）；TE FP8 0.5909ms/465TF。
  - 原始输出 `src/fp8/fa_bwd_fp8_main_o20_fuse_ab.out.txt`（两文件 ×7 shape × FUSE 0/1）、
    `src/fp8/o20_ncu_fuse{0,1}_s4096.out.txt`、`src/fp8/o20_onefile_s4096.out.txt`、
    `src/fp8/o20_tebench_fp8.out.txt`、`src/fp8/o20_fa3_te_baseline_fp16.out.txt`；
    详见 `docs/03` §28、`docs/04` §2.3。

- 2026-09-24（第六十二轮）：**O22 完成（fp8 Hopper 路径默认化：LSE + 主 kernel GEMM1/2 的 wgmma；
  端到端 1.06–1.09×）** + 三组候选杠杆判决（负结果）。
  - 动机：fp8 main 的墙（O7e-3/O19/O20/O21 一致）是 **mma 依赖延迟（`wait`+`short_scoreboard`）+
    3 CTA/SM**；而 O9c（LSE）/O9c-2（主 kernel GEMM1/2）早已实现 Hopper `wgmma`，只是默认关。
    本轮把 **`-DFA_WGMMA` 构建下 `lsewgm`/`wgmma` 默认开**（新增 `--lsewgm=0/1`、`--wgmma=0/1`
    同 binary A/B；`sm_90` 构建行为不变）。单/两文件 device 逐字一致（脚本核对 `DETACHED...`
    `DEVICE REGION IDENTICAL`）。
  - **性能（同 session A/B，event，端到端 total）**：S512 0.1421→**0.1341（1.060×）**、
    S1024H32 0.5623→**0.5308（1.059×）**、S4096 2.6937→**2.4837（1.085×）**、GQA kv4
    0.5296→**0.4947（1.071×）**。收益 = LSE wgmma（S4096 preprocess 0.3957→0.3176，1.246×）
    + 主 kernel wgmma（main 2.1752→2.0432，1.065×）。S4096 **2.48ms / 55.3 TF**（main-only
    67.3 TF，峰值 3.4%），为 TE FP8（同 session 0.5899ms/465.9TF）的 **4.21×**（O21b 4.5×）。
    数值 vs ref 与历史逐位同级（S512 2.426/2.972/3.733e-1；S4096 2.635/2.644/3.216e-1；
    GQA kv4 2.517/5.339/7.173e-1；MLA S1024H2 2.232/3.337/3.602e-1）；单/两文件一致。
  - **同轮否决的三组杠杆**：① `--regdq=0`——O21 ncu 的 local spill（68B st/380B ld、占 L1TEX
    9.57%）来自 `dqacc[2][8][4]`，但关掉后 dQ 每 nt tile 发一次跨 CTA `atomicAdd`，main
    **on 2.136 vs off 2.480ms（on 快 1.16×）** ⇒ **保留 REGDQ**；② `FA_ILV`（GEMM1/2 mma 交错，
    数值逐位不变）三次 A/B 2.180 vs 2.185ms ⇒ 中性偏负，默认 0；③ ksplit 重标定（S4096
    k=2/4/8/16=2.341/2.175/2.157/2.256）与 `PSS` 扫描（32–64 无法消 2-way store 冲突）⇒ 均维持原值。
  - **ncu（main, S=4096, wgmma 默认）**：Duration 2.05ms；stall `wait 1.53 + short 1.57 + long 0.66`
    ⇒ 仍是 **mma 依赖延迟（GEMM3/4/5 仍 mma，fp8 wgmma 无转置操作数）+ 3 CTA/SM**。
  - 原始输出 `src/fp8/o22_wgmma_default_{s4096,shapes,onefile}.out.txt`、
    `o22_mma_baseline_shapes.out.txt`、`o22_regdq_ab_s4096.out.txt`、`o22_ilv_ab_s4096.out.txt`、
    `o22_ksplit_s4096.out.txt`、`o22_ncu_stall_{s4096,wgmma_s4096}.out.txt`、
     `o22_fa3_te_baseline_fp16.out.txt`、`o22_te_fp8_bench.out.txt`；
     详见 `docs/03` §30、`docs/04` §2.3。

- 2026-09-24（第六十三轮）：**O23 完成（fp16/bf16 Hopper 快路默认化：主 kernel wgmma2/wgmma2b +
  LSE wgmma；端到端 1.08–1.43×）**。
   - 动机：O9a/O17/O18 的 fp16/bf16 Hopper 快路全是 **opt-in**（要传 `--wg2`/`--wg2bn`/`--lsewgm`），
     `run_main` 里 `wg2_sel` 默认 0 ⇒ 不传 flag 就退回慢 1.4–1.5× 的 mma。fp8 早在 **O22**
     就把 `-DFA_WGMMA` 构建下的 LSE + 主 kernel GEMM1/2 wgmma 默认化，fp16/bf16 一直没做。
   - **改动**（单/两文件 host 逐字一致，device 代码未动）：`--wg2=`/`--wg2bn=`/`--lsewgm=` 解析加
     `wg_forced`/`lse_forced` 标记（用户显式指定则尊重）；`run_main` 前加自动段——`#ifdef FA_WGMMA`
     且 D==128 时，`S>=4096`→`wg2bn`（BN=128，O18）否则 `wg2`（BN=64，O17）；`causal && D==128`
     →`lse_wgm=1`（非 causal 自动落回 O8 原版）。新增 `[O23] main backend = … | lse = …` 打印。
     纯 `sm_90` 构建行为**完全不变**（恒选 mma）；`--wg2=0 --wg2bn=0`/`--lsewgm=0` 保留回归对照。
   - **数值与历史逐位一致**：fp16 S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、
     GQA kv4 2.134/3.305/3.850e-3、MQA kv1 2.292/7.934/7.517e-3；bf16 S512 9.001/12.61/13.65e-3、
     S4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3；单/两文件逐指标一致。
   - **性能（同 session 端到端 total A/B，CUDA event，ms；旧默认=mma）**：fp16 MHA S512
     0.1132→**0.1045（1.08×）**、GQA kv4 S1024 0.3751→**0.2864（1.31×）**、MQA kv1
     0.6024→**0.4425（1.36×）**、MHA S4096 1.9450→**1.3641（1.43×）**；bf16 MHA S512
     0.1119→**0.1058**、GQA kv4 0.3763→**0.2875**、MQA kv1 0.6022→**0.4438**、MHA S4096
     1.9429→**1.3666（1.42×）**；MLA（D=512）不变。main-only：fp16 S4096 mma 1.5111 vs O18
     0.9601（**144.1 TF**）；bf16 同构 143.1 TF。LSE wgmma 再叠加 ~1.3%（S4096 preprocess
     0.3446→0.3270）。
   - **ncu（默认路径=`wgmma2b`，S=4096，`-c 1`）**：`lts__t_sectors_op_red=51,904,512`（与 O18
     逐字节相同）、255 regs / 231.42KB smem / achieved occ 12.48% / L2 56.58% / Compute 23.55%；
     stall `wait 1.25 + long 0.60 + barrier 0.93` ⇒ **墙仍是 L2 的 dK/dV 跨 CTA `red` + 1 CTA/SM**。
   - **对标**（同 session 纯反向 `fa_vs_te_bwd_only.py`，FA2/FA3/TE）：fp16 MHA S4096 FA3
     0.3251ms/846TF、TE 0.4444/619 ⇒ ours total 1.3641ms = **FA3 4.20× / TE 3.07×**（O18 4.30×）；
     GQA kv4 FA3 0.0831/413 ⇒ 3.45×；MQA kv1 FA3 0.1566/439 ⇒ 2.83×。bf16 MHA S4096 FA3
     0.3210/856 ⇒ 4.26×；GQA kv4 3.48×；MQA kv1 2.83×。
   - 原始输出 `src/fp16/fa_bwd_fp16_mma_main_o23_{s4096,shapes}.out.txt`、
     `..._mma_main_o23b_s4096.out.txt`、`..._mma_onefile_o23b_s4096.out.txt`、
     `..._o23_ncu_default_s4096.out.txt`、`src/bf16/fa_bwd_bf16_mma_*_o23*`、
     `src/fa_bwd_o23_default_ab.out.txt`、`src/fa_bwd_o23_shapes_final.out.txt`、
     `src/fa_bwd_o23_fa3_te_baseline_{fp16,bf16}.out.txt`；文档 `docs/01` §14n、`docs/01b` §6v、
     `docs/04` §2.1/§2.2、`docs/08` §5。

- 2026-09-24（第六十四轮）：**O7b 完成（fp16 确定性 dK/dV：partial + 二次归约）——机制成立、
  逐位可复现，但净负（S4096 main+reduce 0.810×）；`red` 51.9M→0**。
   - 动机：O17 后 fp16 main 仍是 **L2 的 dK/dV 跨 CTA `atomicAdd`** bound（`red` = 51,904,512，
     占余下 L2 扇区 ~72.6%）；O17b（放大 BM）/O7c（float4）已证伪「减事务数/放大宽度」。O7b 直接
     **不用原子**：每个 (Q 块, KV 行) 的贡献写进带 `mblk` 下标的 partial（非原子覆盖写，每元素本
     CTA 只写一次），再由 `dkv_reduce_kernel` 按 `mblk` 升序 + Q 头广播组求和 ⇒ **确定性反向**。
   - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     `fa_bwd_fp16_wgmma2b_kernel<HD,SPLIT,DET=false>` 加 `DET` 开关 + `dk_part/dv_part/nblk`
     参数；四处 dK/dV epilogue 从 `red_add2` 改 `dkv_det_store`（float2 覆盖写）；新增
     `dkv_reduce_kernel<HD>`；host `--det=1` opt-in A/B（仅 `FA_WGMMA && D==128`）。默认路径不变。
   - **踩坑**：partial 初版按 `hkv` 索引，GQA/MQA 下同一 KV 头的多个 Q 头互相覆盖（race），
     GQA kv4 `max|diff|=6.3`（错）；改按 Q 头 `h` 分片、reduce 对广播组 `G=H/Hkv` 求和后修复。
   - **数值（fp16 causal）**：DET 两次运行 **bitwise-diff=0**（可复现）；DET-vs-atomic dk/dv
     2.38e-7/4.77e-7（S512）、7.15e-7/9.54e-7（S4096）、2.86e-6/3.81e-6（GQA kv4，仅加法次序）；
     ours-vs-ref 与历史逐位同级（S4096 1.883/1.734/1.966e-3）。
   - **性能（同 session A/B，event，main[+reduce]）**：S512 0.0551→**0.0538（1.023×）**、
     S1024 GQA kv4 0.1865→0.2100（0.888×）、S4096 0.9598→**1.1849ms（0.810×）**。
   - **ncu（S=4096）**：atomic main `red=51,904,512`/write 1.574M/Duration 958.6µs →
     **DET main `red=0`**/write 53.48M/**Duration 802.0µs（1.20×）**；reduce Duration **384.9µs**、
     **DRAM Throughput 90.4%（带宽 bound）**、read 51.9M 扇区 ⇒ **主 kernel 去掉原子 RMW 反而快
     1.20×，但二次归约是一趟 DRAM 扫描，把收益吃回**。⇒ 保留 `--det=1` opt-in（确定性），
     **不作为性能杠杆**；真正消 red 需 **cluster 分布式归约**（SM 间 smem 合并、不落全局内存）。
   - 对标（同 session 纯反向 `fa_vs_te_bwd_only.py fp16`）：MHA S4096 FA3 0.3249ms/846TF、
     TE 0.4449/618、FA2 0.7292/377；默认端到端仍为 FA3 ~4.2×（O23，不受 O7b 影响）。
    - 原始输出 `src/fp16/fa_bwd_fp16_main_o7b_det_{s512,s4096,gqa_kv4}.out.txt`、
      `src/fp16/fa_bwd_fp16_mma_onefile_o7b_det_{s512,s4096}.out.txt`、
      `src/fp16/fa_bwd_fp16_main_o7b_ncu_{atomic_s4096,det_s4096,reduce_s4096}.out.txt`、
      `src/fa_bwd_o7b_fa3_te_baseline_fp16.out.txt`；文档 `docs/01` §14o。

- 2026-09-24（第六十五轮）：**O24 完成（fp16/bf16：preprocess `delta` 向量化 + dQ 直写 fp16，
  端到端 1.02–1.08×）**。
   - 动机：O23 后 fp16 MHA S=4096 端到端 = preprocess 0.327 + main 0.943 + convert 0.096 ≈ 1.366ms。
     main 的墙是 **L2 的 dK/dV 跨 CTA `red`**（已无搬运/等待类杠杆），但 preprocess(24%) 与
     convert(7%) 里还有两处纯工程浪费：① `delta_kernel` 旧版**每 (s,h,b) 行一个 128 线程 CTA +
     `__shared__` 树归约 + log2(THREADS) 次 `__syncthreads`**（ncu S=4096：Compute 72% / L1TEX 74%、
     42.5µs）；② D=128 的 dQ 由主 kernel 寄存器累加后**唯一拥有**（无跨 CTA 原子），却仍写 fp32
     `dq_acc` 再由 convert 读回转 fp16。
   - **改动**（单/两文件 device 与 host 逐字一致）：① **`delta_warp_kernel<HD>`** 改 **warp-per-row**、
     lane 沿 HD 以 `__half2`（4B）coalesced 读、`__shfl_xor_sync` 树归约，**无 smem/无 barrier**、
     grid-stride；② `fa_bwd_fp16_wgmma2/2b_kernel` 加 `__half* dq_h`，非空时 dQ 直接
     `__floats2half2_rn` 写 `dq`（与 convert 的 RN 相同 ⇒ **逐位不变**），host 在 D==128 且选中
     wgmma2/wgmma2b 时令 `convert` 的 `n_q=0`；其余路径（mma/wgmma/wgmma4/MLA RMW）不变。
     `--deltawarp=0`/`--dqdirect=0` 做同 binary A/B。
   - **数值与历史逐位一致**（fp16 S512 1.671/1.771/1.899e-3；S4096 1.883/1.734/1.966e-3；
     GQA kv4 2.134/3.305/3.850e-3；MQA kv1 2.292/7.934/7.517e-3；bf16 S512 9.001/12.61/13.65e-3、
     S4096 15.10/13.40/16.31e-3），单/两文件一致。
   - **性能（同 session A/B，event，端到端 total）**：fp16 MHA S4096 1.3802→**1.3309ms（1.037×）**、
     S512 0.1052→**0.1001（1.051×）**、GQA kv4 S1024 0.2892→**0.2712（1.066×）**、MQA kv1
     0.4436→**0.4119（1.077×）**；bf16 S4096 1.3650→**1.3388（1.020×）**、S512 1.053×、
     GQA kv4 1.053×、MQA kv1 1.069×。`delta` 单项 S4096 0.0425→**0.0127ms（3.35×）**。
   - **ncu（S=4096）**：旧 delta Duration 42.53µs / DRAM 24.94% / L1TEX 73.68% / Compute 72.16% /
     inst 23.2M / Waves 31.03 → 新 **14.50µs / DRAM 72.87%（带宽 bound）/ L1TEX 25.84% /
     Compute 29.42% / inst 4.19M（−82%）/ Waves 7.76** ⇒ 墙从 smem 归约+标量加载移到 **DRAM 带宽**
     （elementwise 上限），与 fp8 O14 的 `quantize_row_warp_kernel` 结论一致。
   - **对标**（同 session 纯反向 `fa_vs_te_bwd_only.py`）：fp16 MHA S4096 FA3 **0.3246ms/847TF**、
     TE 0.4405/624、FA2 0.7293/377 ⇒ ours total 时间 **4.10×**（O23 4.20×）；GQA kv4 3.28×；
     bf16 FA3 0.3202/859 ⇒ 4.18×。
   - 原始输出 `src/fp16/fa_bwd_fp16_main_o24_sweep.out.txt`、
     `src/fp16/fa_bwd_fp16_mma_onefile_o24_s4096.out.txt`、
     `src/fp16/fa_bwd_fp16_main_o24_ncu_delta_{old,warp}_s4096.out.txt`、
     `src/bf16/fa_bwd_bf16_main_o24_sweep.out.txt`、
     `src/bf16/fa_bwd_bf16_mma_onefile_o24_s4096.out.txt`、
     `src/fa_bwd_o24_fa3_te_baseline_{fp16,bf16}.out.txt`；文档 `docs/01` §14p、`docs/01b` §6w、
     `docs/04` §2.1/§2.2/§3、`docs/08` §5。

- 2026-09-24（第六十六轮）：**O25 完成（fp16：cluster 分布式归约 dK/dV）——机制成立、数值逐位
  正确，但净负（S4096 main 0.13×）；「消 red」三条路至此全部证伪**。
   - 动机：O7b（§14o）留下明确结论——真正消 red 只能上 **cluster 分布式归约**（SM 间 smem 内
     合并偏和、不落全局 partial、不做二次 DRAM 扫描）。O25 实现并判决这条路。
   - **机制冒烟** `src/fp16/fa_bwd_fp16_cluster_reduce_smoke.cu`（cluster=2、
     `red.shared::cluster.add.f32` + `mapa.shared::cluster` + `barrier.cluster`，多 tile 复用
     累加器）：`cluster-vs-ref max_abs=1.4e-06`（仅 fp32 加法次序）⇒ **PASS**。
   - **实现**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     `fa_bwd_fp16_wgmma2_kernel` 加 `int CL=1`（默认逐字不变）；cluster 沿 bx 配对相邻 mblk，
     leader 的 smem 加两个 `[BN][HD]` fp32 合并累加器（+64KB）；6 处 dK/dV red 经 `dkv_red`
     分流到 `red.shared::cluster.add.f32`；每 tile 尾 `cluster_sync → leader flush(一次全局
     red) → cluster_sync`；host `--cluster[=2]`（`cudaLaunchKernelEx`），开启时强制
     `wgmma2(BN=64)`（BN=128 的 2b 放不下合并累加器）。
   - **数值**：S512 `1.671/1.771/1.899e-3`、S4096 `1.883/1.734/1.966e-3`——与 O17~O24 **逐位
     一致**；单/两文件一致。
   - **性能（同 session A/B，event，main/total ms）**：S512 main 0.0528→**0.3443（6.5×）**、
     S4096 0.9855→**7.3178（7.4×）**；total 0.1002→0.3924 / 1.3597→7.8881。同 session 纯反向
     FA3 MHA S4096 **0.3238ms/849TF**、TE 0.4407/624、FA2 0.7253/379 ⇒ 默认 total 为 FA3 4.2×，
     cluster 版退化到 24×。
   - **ncu（main, S4096，同 session/同 binary，`-c 1`）**：`lts__t_sectors_op_red`
     **51,904,512→26,738,688（0.515×，精确减半）**、`read` 0.645×（机制被证实），而
     Duration 989.7µs→**7.64ms（7.7×）**、**stall `long_scoreboard` 0.75→5.83、`short` 0.41→4.19**
     ——逐元素远程 smem 原子延迟极高 + 每 tile leader flush 串行（另一 SM 在 `cluster_sync` 空等）。
   - **结论**：非原子 DSM（`st.async` + leader 求和）可避延迟但需 `CL×[BN][HD]` smem（CL=2
     +128KB，放不下）；逐元素先 CTA 内合并再 cluster 又退回 O7b 结构。**保留 `--cluster` opt-in**
     （默认关）。**fp16/bf16 main 的 L2 red 墙在本卡暂无便宜解法**（「放大 BM」§14k、
     partial/reduce §14o、cluster §14q 三条路全证伪）。
   - 原始输出 `src/fp16/fa_bwd_fp16_cluster_reduce_smoke.out.txt`、
     `src/fp16/fa_bwd_fp16_o25_s512_ab.out.txt`、`..._o25_s4096_ab.out.txt`、
     `..._mma_onefile_o25_s512.out.txt`、`..._mma_main_o25_ncu_s4096.out.txt`、
       `..._o25_fa3_te_baseline.out.txt`；文档 `docs/01` §14q。

- 2026-09-24（第六十七轮）：**O26 完成（fp8 `delta_kernel` → warp-per-row 向量化，对齐 fp16/bf16 O24）**。
   - 动机：fp16/bf16 在 **O24（第六十五轮）** 已把 `delta_kernel`（D=rowsum(dO∘O)）改成
     **warp-per-row 向量化**（S=4096 42.5→14.5µs，3.35×），但 **fp8 的 delta 一直是旧版**
     （每行一个 128 线程 CTA + `__shared__` 树归约 + 7×`__syncthreads`）——O11（fp8 LSE）/
     O14（fp8 quant）都漏了它。ncu S=4096：旧 delta `Duration 42.94µs`、`Compute 74.93%`、
     `Waves 31.03`、grid 65536（一行仅 128 个乘加，固定开销远大于计算）。
   - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     新增 `delta_warp_kernel<HD>`——每 warp 一行，lane 沿 HD 以 `float4`(O)/`uchar4`(dO)
     各读 4 个元素，`__shfl_xor_sync` 树归约，**无 smem / 无 barrier**、grid-stride；逐元素
     同式 `O[d]*(deq_e5m2(dO[d])*dos[row])`。host 加 `--deltawarp=0/1`（默认 1）与 `[O26 A/B]`。
   - **数值**：`max_abs(new-vs-old) = 1.9e-6~8.6e-6`、`max_rel ~1e-3`（仅 fp32 求和次序，
     warp 树 vs smem 树）；**vs fp32 ref 与历史同水平**（S512 2.426/2.972/3.733e-1；S4096
     2.635/2.644/3.216e-1；GQA kv4 2.517/5.339/7.173e-1；kv8 2.869/5.367/7.032e-1；
     MQA kv1 4.101e-1/1.572/2.126；MLA S1024H2 2.232/3.337/3.602e-1）。单/两文件逐指标一致。
   - **性能（同 session A/B，event，`-DFA_WGMMA` 默认构建）**：delta **S512 2.39× / S1024H32
     3.22× / S4096 0.0433→0.0151ms（2.87×）** / GQA kv4 3.05× / kv8 3.16× / MQA kv1 3.18× /
     MLA 1.33–1.51×；**preprocess S=4096 0.3176→0.2946ms（1.078×）**；端到端 S4096
     **2.4837→2.4637ms（1.008×，55.8 TF）**、S512 0.1284（16.7 TF）、S1024H32 0.5218（32.9）。
     同 session TE FP8 S512 0.1007 / S4096 **0.5904ms/465.6TF** ⇒ 端到端 ours/TE S4096 **4.17×**
     （O22 4.21×）。同 session 纯反向 fp16：MHA S4096 FA3 0.3237ms/849TF、TE 0.4419/622、FA2 0.7233/380。
   - **ncu（delta，S=4096，同 binary `--deltawarp` 0/1，`--set full -c 1`）**：Duration
     42.94→**17.34µs**、**DRAM 31.13→77.01%**、Compute 74.93→27.14%、Waves 31.03→7.76、
     grid 65536→16384 ⇒ 墙从 **smem 归约 + Compute 75%** 移到 **DRAM 带宽 77%（elementwise 上限）**，
     与 fp16 O24 的 delta（72.87%）/ fp8 O14 的 quant（71.31%）结论一致。
    - 原始输出 `src/fp8/fa_bwd_fp8_main_o26_sweep.out.txt`（两文件 ×9 shape × `[O26 A/B]` + 对拍）、
      `src/fp8/fa_bwd_fp8_mma_onefile_o26_sweep.out.txt`（单文件）、
      `src/fp8/o26_ncu_delta_{old,new}_s4096.out.txt`、`src/fp8/o26_te_fp8_bench.out.txt`、
      `src/fa_bwd_o26_fa3_te_baseline_fp16.out.txt`；文档 `docs/03` §31、`docs/04` §2.3/§3。

- 2026-09-24（第六十八轮）：**O27 完成（fp8 fold 量化「逐元素精确除法」→「每行 rcp + 乘法」，
  main 1.08–1.18×）**。
   - 动机：fold 把 fp32 的 `P`/`dS` 按 rowwise amax 量化成 fp8 的 `Ap`/`dS3`/`dS2`，对**每个
     `(m,j)` 元素**都算一次 `Ps*[dos] / scA`（`dS3` 用 `/sc3`、`dS2` 用 `/sc2`），而 `scX=amax/fp8max`
     是**每输出行一个**的常量。ptxas 默认 `-prec-div` ⇒ 每个元素一条精确除法（~10+ 指令），
     fold 段（CUDA-core，夹在 GEMM1/2 与 GEMM3/4/5 之间、张量核空转）是纯浪费。
   - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
     `fa_bwd_fp8_mma_kernel<..., bool RCP>` 新增模板参数（默认 true）——每行广播 `scX` 后再算一次
     `__frcp_rn(scX)` 并广播，元素处用新 helper `folddiv<RCP>(x, sc, inv)`（`true`→乘法、`false`→除法）；
     `scX` 仍原样存 `sA/sds3/sds2` 供反量化。host 加 `--foldrcp=0/1` + `[O27 A/B]`。
   - **数值**：vs fp32 ref 与历史**逐位一致**（S512 2.426/2.972/3.733e-1；S1024H32 2.399/4.177/3.535e-1；
     S4096 2.635/2.644/3.216e-1；GQA kv4 2.517/5.339/7.173e-1；kv8 2.869/5.367/7.032e-1；MQA
     4.101e-1/1.572/2.126；MLA S256H2 2.356/2.290/3.441e-1 等 9 shape 全部相同）；
     `[O27 A/B] max_abs(rcp-vs-div)` ~1e-4–2e-3（fp8 cvt 边界舍入 + dk/dv atomic 次序）。
   - **性能（同 session A/B，event，main-only）**：**S4096 2.043→1.758ms（1.162×）**、
     S512 1.077×、S1024H32 1.092×、GQA kv4 1.083×、kv8 1.152×、MQA 1.178×；端到端 S4096
     2.4637→**2.1770ms（63.1 TF）**、S512 0.1284→**0.1246**、S1024H32 0.5218→**0.4895**、
     GQA kv4 0.4875→**0.4566**、kv8 0.5606→**0.5010**、MQA 0.7926→**0.6956**、MLA S256/S512/S1024
     0.1169/0.2972/0.5175→**0.1094/0.2688/0.4653**。同 session TE FP8 0.5904ms/465.6TF ⇒ 端到端
     ours/TE **3.69×**（O26 4.17×）。
   - **ncu（main, S=4096, `-c 1`）**：Duration 2.06→**1.78ms**、**executed inst 852.6M→735.2M
     （−13.7%）**、L1/TEX 57→61.6%、L2 54.5→63.2%、Compute 43.4→43.7%、168 regs/70.66KB/3 CTA/SM；
     stall wait 1.53→1.48、short 1.57→1.43、long 0.67→0.87 ⇒ 第一墙仍是 **mma 依赖延迟 + 3 CTA/SM**
     （O7e-3/O19/O20/O22 结论未变），但 fold 段被显著削短。
   - 原始输出 `src/fp8/o27_main_sweep.out.txt`、`o27_foldrcp_ab_s4096.out.txt`、
     `o27_ncu_{main,rcp,stall}_s4096.out.txt`、`o27_te_fp8_bench.out.txt`、
     `o27_fa3_te_baseline_fp16.out.txt`；文档 `docs/03` §32、`docs/04` §2.3/§3。
    - **下一步修正**：原记「fp16/bf16 的 fold 同样含逐元素精确除法」**有误**——逐字核对
      `src/fp16/fa_bwd_fp16{mma_,}_kernels.cuh` / `src/bf16/...` 后确认：**fp16/bf16 张量核版没有
      rowwise scale fold**（`_mma_` 版注释开宗明义「fp16 无量化，也就没有 fold」）；逐元素除法只
      存在于 **fp8** 的 fold / 输入量化里。故 ④ 这条作废，fp8 的 fold 除法 O27 已收口。

- 2026-09-24（第六十九轮）：**O28 完成（fp8 fold 转换指令向量化 `cvt...x2`，MLA main ~1.03×，
  d128 中性；同时纠正 O27 的「④ fp16/bf16 fold」误记）**。
    - 动机：O27 证明 fold 的**指令数**在 main 关键路径上（去掉精确除法得到 1.08–1.18×）。fold 里
      每元素还有 **3 次 fp8 转换**（Ap=E4M3、dS3/dS2=E5M2），旧实现逐元素
      `__nv_cvt_float_to_fp8` + `<<`/`|` 拼 4B（SASS 一条 `F2FP` + `LOP3/PRMT`）。
    - **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：
      新增 `cvt2_e4m3/cvt2_e5m2`（封装 `__nv_cvt_float2_to_fp8x2`，即 `cvt.rn.satfinite.*x2.f32`，
      SASS 走 `F2FP...PACK_AB_MERGE_C` 直拼 32 位）与 `foldpack4<RCP,E5>(x0..x3,sc,inv)`；fold 的
      三处量化（Ap/dS3 的 F16B 及旧 4×4B 两分支、dS2）全部改走它。新增开关 `FA_CVT2`（默认 1），
      `-DFA_CVT2=0` 退回标量版做同 binary A/B。
    - **数值**：同批 fp32 折算 + 同舍入 ⇒ fp8 字节逐位相同，**vs fp32 ref 与历史逐位一致**
      （S4096 2.635/2.644/3.216e-1；MLA S1024H2 2.232/3.337/3.602e-1）；单/两文件一致。
    - **性能（同 session A/B，event，main-only）**：**d128 全中性**（S512/S1024/S4096/GQA/MQA
      0.99–1.01×）；**MLA 有正收益**——S512H4 0.1423→**0.1386（1.027×）**、S1024H2
      0.2727→**0.2653（1.028×）**（`NDT=HD/128=4`，fold 跑 4 遍、占比大不被完全掩盖）。
    - **ncu（main，`-c 1`，同 binary CVT2 0/1）**：d128 S4096 executed inst **735,418,304→
      703,469,504（−4.3%）**、Duration 1.80→**1.77ms**、stall `wait` 1.48→1.55/`short` 1.42→1.58、
      occ 18.11% 不变 ⇒ **指令降了但 d128 是纯 mma 依赖延迟 bound，省下的指令无处兑现**；
      MLA S1024H2 Duration 326.75→**320.54µs**、inst 24.72M→24.45M（−1.1%）。
    - 原始输出 `src/fp8/fa_bwd_fp8_o28_ab.out.txt`（9 shape × CVT2 0/1）、
      `src/fp8/o28_ncu_main_{s4096,mla_s1024h2}_ab.out.txt`、
      `src/fp8/o28_main_{s4096,mla_s1024h2}_full.out.txt`、`o28_te_fp8_bench.out.txt`、
      `o28_fa3_te_baseline_fp16.out.txt`；文档 `docs/03` §33、`docs/04` §2.3。

- 2026-09-24（第七十轮）：**O29 完成（fp8 自动 split-K 重新标定，端到端 1.01–1.32×；主 kernel
  最多 1.20×）+ GEMM3/4 指令级交错判决（负结果）**。
    - 动机：`ksplit` 自动档一直是 **O2b（第二十二轮）** 的固定 `TARGET=(D==128)?4096:132`；
      但 O2b 之后主 kernel 数据通路被改过多次（O3/O4b/O9c-2/O20/O22/O27/O28），最优点已漂移：
      d128 在 `base_grid` 小时**过切**（S1024 base=512，`use_regdq` 因此被关、dQ 逐 tile red 更多），
      S=4096 **欠切**，MLA **严重欠切**。
    - **改动**（仅 host 自动档，单/两文件 host 同步；device 代码与数学口径不变，只改 fp32 加法次序）：
      `D==128 → S>=2048 ? 8192 : max(2048, 4*base_grid)`；`D==512 → S/2`。`--ksplit=N` 仍可强制。
    - **sweep（10 个 fp8 case，main-only，event）**：最优 k 与新 auto 一致/紧邻。main：
      **S4096 1.7579→1.7370（1.012×）**、**S1024H32 0.3404→0.2988（1.139×）**、
      **kv4 0.3286→0.2742（1.198×）**、**MLA S512H4 0.1402→0.1215（1.154×）**、
      **MLA S1024H2 0.2652→0.2004（1.324×）**；kv8/MQA/h64kv4/S512/S256H2 持平。
      端到端 total 同向：S4096 2.1770→**2.1481ms**、S1024H32 0.4895→**0.4464**、
      GQA kv4 0.4566→**0.4056**、MLA S1024H2 0.4653→**0.3900ms**，**全 shape 不回退**。
    - **ncu（main，S1024H32，同 binary k=8 vs k=4）**：Duration 332.4→**304.9µs**、
      `lts__t_sectors_op_red` **26.74M→16.42M（0.61×）**、**L2 83.09%→57.61%**、
      `short_scoreboard` 2.40→**1.50** ⇒ 收益来自「少切 + 自动开 `use_regdq`」把 dQ 原子扇区
      降到 0.61×、L2 压力解除。**数值 vs ref 与历史逐位同级**（S512 2.426/2.972/3.733e-1；
      S4096 2.635/2.644/3.216e-1；S1024H32 2.399/4.177/3.535e-1；kv4 2.517/5.339/7.173e-1；
      kv8 2.869/5.367/7.032e-1；h64kv4 2.761/8.427/1.233；MQA 4.101e-1/1.572/2.126；
      MLA S256H2 2.356/2.290/3.441e-1；S512H4 2.415/2.992/4.481e-1；S1024H2 2.232/3.337/3.602e-1），
      单/两文件逐指标一致。
    - **同轮负结果（`FA_ILV34`）**：先连发 GEMM3(dV)/GEMM4(dK) 两条独立 mma 再 epilogue
      （对齐 O22 `FA_ILV` 思路、数值逐位不变），但两个 `MTM34×8×4` 累加器同时存活使 ptxas 在
      170-reg 预算下**溢出增加** ⇒ S4096 0.95×、S1024H32 0.96×。**再次证明「在 3 CTA/SM 的寄存器
      预算内加 ILP」会被 spill 吃掉**（与 O7c/O17b 一致）；开关保留默认关。
    - 原始输出 `src/fp8/fa_bwd_fp8_o29_ksplit_sweep.out.txt`、
      `src/fp8/fa_bwd_fp8_o29_ncu_s1024h32.out.txt`、`o29_te_fp8_bench.out.txt`、
      `o29_fa3_te_baseline_fp16.out.txt`、`fa_bwd_fp8_o29_ilv34_s4096.out.txt`；
      文档 `docs/03` §34、`docs/04` §2.3。

- 2026-09-24（第七十一轮）：**O30 完成（fp16：LSE 的 4D-TMA 载入 Q/K；LSE-only 1.30–1.36×、
  指令数 −28.7%、数值逐位不变；O15a 的 TMA 通路首次落进真 kernel）**。
  - 动机：O15a（第 51 轮）已证 `cuTensorMapEncodeTiled(SWIZZLE_128B)` 写出的 smem 与
    `sw128_off` **逐字节相同**，且 **HD=128 的 K-major tile 必须拆成 2×K=64 chunk**（TMA box
    内维 128B=64 个 fp16），但那条通路一直只停在冒烟。O23 把 Hopper 快路默认化后，端到端里
    剩下最大的、还没上 TMA 的搬运就是 **LSE 的 Q/K 载入**（逐 16B `cp.async` + 地址运算）。
  - **改动**（单/两文件 device 逐字一致，手工重建单文件并核对 `device identical: True`）：
    新增 `mbar_init/arrive_expect/wait`、`tma_load_4d`（`cp.async.bulk.tensor.4d`）与
    `wgmma_qkt64_tma`（两 chunk 各 `SBO=1024` 的 4+4 条 `wgmma.m64n64k16`），全部 `FA_HAS_WGMMA`
    包裹；新增 `lse_mma_kernel_bal_tma<HD,PIPE=1>`（smem = Q + 2×K + 3 mbarrier；K 双缓冲、
    逐 barrier 相位计数），数学与 `lse_mma_kernel_bal_wgmma` **完全一致**。host 加
    `make_lse_map`（4D 描述符 `dims={D,S,H,B}`、box `{64,64,1,1}`）、CLI `--lsetma=0/1`。
    TMA 路径需驱动 API ⇒ 用 **`-DFA_TMA`** 开关整体包裹：`-DFA_WGMMA -DFA_TMA -lcuda` 构建下
    D==128/causal 默认开；纯 `-DFA_WGMMA` 或 `sm_90` 构建**不引用驱动符号、无需 `-lcuda`**。
  - **数值**：TMA-vs-wgmma 的 LSE **`max_abs=0.000e+00`（逐位相同）**，5 shape × 单/两文件全部；
    `dq/dk/dv vs ref` 与历史逐位一致（S512 1.671/1.771/1.899e-3；S4096 1.883/1.734/1.966e-3；
    GQA kv4 2.134/3.305/3.850e-3；kv8 2.008/2.931/3.891e-3；MQA kv1 2.292/7.934/7.517e-3）。
  - **性能（同 session A/B，CUDA event）**：LSE-only **S512 0.0328→0.0249（1.32×）、S4096
    0.2878→0.2128（1.35×）、GQA kv4 1.31×、kv8 1.30×、MQA kv1 1.32×**；端到端 total
    S4096 **1.2553ms（109.5 TF）**、S512 0.0942、GQA kv4 0.2575、kv8 0.2905、MQA 0.3945
    （单/两文件一致）。preprocess S4096 0.2271ms（O24 0.3002）。
  - **ncu（lse, S=4096，同 session，`-c 1`）**：Duration 286.30→**214.56µs（1.33×）**、
    **Executed Instructions 165.30M→117.87M（−28.7%）**、regs 62→58、smem 50.18→50.24KB、
    occ 23.02→23.11%、DRAM 3.77→5.05%、L1TEX 17.43→17.69%、L2 20.27→26.77%、Compute
    60.93→58.30%；stall `wait 2.26→2.33 + short 0.77→0.86 + long 0.04→0.17` ⇒ 收益纯来自
    **指令数**，**墙不变 = Compute ~58% + `wait`（softmax/mma 固定延迟）**。
  - **对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）：FA3 MHA S4096
    **0.3244ms/847TF**、GQA kv4 0.0825/417、kv8 0.1212/354、MQA kv1 0.1562/440 ⇒ ours total
    时间比 **3.87×**（O24 4.10×）/ GQA kv4 3.12× / kv8 2.40× / MQA 2.53×，**全 shape 小幅改善**。
  - 原始输出 `src/fp16/fa_bwd_fp16_o30_lse_tma_sweep.out.txt`（单/两文件 ×5 shape ×
    `[O30 A/B]` + 对拍）、`src/fp16/fa_bwd_fp16_lse_tma_ncu_s4096.out.txt`、
    `src/fp16/fa_bwd_fp16_lse_wgmma_ncu_s4096.out.txt`、`src/fp16/fa_bwd_fp16_o30_fa3_te_baseline.out.txt`；
    文档 `docs/01` §14r、`docs/04` §2.1、`docs/08` §5。

- 2026-09-24（第七十二轮）：**O31 完成（bf16：LSE 的 4D-TMA 载入，对齐 fp16 O30）**。
  把 fp16 O30 逐字节 dtype 参数化到 bf16（`wgmma...bf16` + tensormap `BFLOAT16`），
  LSE-only **1.32–1.35×**、指令数 **−28.7%**、端到端 **1.065–1.071×**（MHA S4096
  1.3390→**1.2577ms**，109.3 TF，FA3 的 3.94×），`max_abs(tma-vs-wgmma)=0` 逐位一致。
  详见 `docs/01b` §6x。

- 2026-09-24（第七十三轮）：**O32 完成（fp8：LSE 的 4D-TMA 载入，对齐 fp16 O30 / bf16 O31）**。
  - **fp8 与 fp16 的关键差异**：fp8 一行 128 元素 = **128B = SW128 atom 整行** ⇒ Q/K 各只需
    **一次** 4D-TMA（box `{128,64,1,1}`，dtype=`UINT8`；fp16 需 2×K=64 chunk）。新增
    `lse_mma_kernel_bal_tma<HD,PIPE=1>`（单/两文件 device 逐字一致，`sync_onefile_device.py`
    核对 `identical: True`）+ `mbar_*`/`tma_load_4d` 封装；`Fp8Cfg::lse_smem_bytes_tma1`；
    host `make_lse_map_fp8` + `--lsetma=0/1`（`-DFA_TMA` 构建下 D==128/causal 默认开）。
    rowwise scale 仍走标量 global 读（TMA 带不了标量数组）。
  - **数值**：TMA-vs-wgmma 的 LSE **`max_abs=0.000e+00`（逐位相同）**，6 shape × 单/两文件；
    `dq/dk/dv vs ref` 与历史（O9c/O22/O27/O29）**逐位一致**（S512 2.426/2.972/3.733e-1、
    S4096 2.635/2.644/3.216e-1、S1024H32 2.399/4.177/3.535e-1、kv4 2.517/5.339/7.173e-1）。
  - **性能（同 session A/B，event）**：LSE-only **S512 1.076× / S1024H32 1.075× / kv4 1.077× /
    kv8 1.079× / MQA 1.091× / S4096 1.101×**；端到端 total S4096 **2.1303ms（64.52 TF）**、
    S512 0.1219、S1024H32 0.4468、kv4 0.4035、kv8 0.4891、MQA 0.6839。同 session TE FP8
    0.1009/0.2059/0.5876 ⇒ ours/TE **1.21×/2.17×/3.63×**（kv4 1.99×、kv8 2.02×、MQA 1.70×）。
    收益小于 fp16 O30（1.3×）：fp8 LSE 本来就用 `cp.async` 且每元素 1B，地址运算占比小。
  - **ncu（LSE, S4096）**：Duration **251.97µs**、Compute **56.79%** / L1TEX 25.00% / L2 14.60% /
    DRAM 2.08%、occ 23.55%（61 regs/26.43KB）、Waves 0.48；stall TMA-vs-wgmma：
    `long_scoreboard 2.20→0.37`、`short_scoreboard 2.25→1.05`、`wait 2.68→2.34`、
    `barrier 0.53→0.32`、`mio 0.67→0.03` ⇒ **新墙 = Compute ~57% + `wait`（softmax/mma 固定
    延迟）**，与 fp16 O30/bf16 O31 一致。第一墙仍是主 kernel（mma 依赖延迟 + 3 CTA/SM）。
  - 原始输出 `src/fp8/fa_bwd_fp8_o32_sweep.out.txt`、`..._o32_onefile_sweep.out.txt`、
    `..._o32_ncu_lse_tma_s4096.out.txt`、`..._o32_ncu_stall_lse_{tma,wgmma}_s4096.out.txt`、
     `src/fp8/fa_bwd_fp8_o32_tebench.out.txt`；文档 `docs/03` §35。
- 2026-09-24（第七十四轮）：**O33 完成（fp16：主 kernel 的 Q/K/V/dO 改用 4D-TMA，逐 atom）**。
  - **关键思路**：O30–O32 只把 LSE 的 Q/K 上了 TMA；主 kernel 的 Q/dO（prologue）与 K/V
    （每 tile）仍用逐 16B `cp.async` + `sw128_off` 地址运算。HD=128 的 K-major tile 的交织
    布局无法用一个 2D TMA box 复现（O15a），故**逐 atom 发 TMA**：一个 `[8 行][64 列]` box
    = 一个 1024B SW128 atom，dst 放到 `sw128_off(rg*8,kg*64,HD)`，**原样复现交织布局** ⇒
    **所有 wgmma 描述符零改动**、搬的字节与 cp.async 逐字节相同。
  - **实现**：`tma_fill_sw128<R,HD>`（逐 atom `tma_load_4d`）+ `fa_bwd_fp16_wgmma2b_tma_kernel`
    （与 wgmma2b 几何/数据流/描述符逐字相同，只换载入为 TMA + mbarrier：Q/dO 一次性、
    K 双缓冲两 barrier、V 单缓冲后段预取）；host `make_main_map`（box `{64,8}`）+
    `launch_bwd_wgmma2b_tma` + `--maintma` opt-in；单/两文件 device 逐字一致。
  - **数值**：`[O33 A/B] max|diff|` **dq `0.00e+00`（逐位）**、dk/dv ~2e-5（仅跨 CTA
    `atomicAdd` 次序）；vs ref 与历史相同（S4096 1.883/1.734/1.966e-3）。
  - **性能（同 session event）**：main **0.9638→0.9249ms（142.6→148.6 TF，1.042×）**、单文件
    1.044×；端到端 total **1.2289ms / 111.8 TF**。ncu：Duration 968.96→**923.87µs**、
    **`red` 51,904,512 逐字节不变**、regs 255/smem 230.5KB/occ 12.5%（1 CTA/SM）全不变
    ⇒ **TMA 只省搬运，动不了主墙（dK/dV 的 L2 `red`）**。
  - **对标**（同 session 纯反向 FA2/FA3/TE）：FA3 MHA S4096 0.3263ms/842TF、TE 0.4443/619
    ⇒ ours total **3.77×**（O30 3.87×）。原始输出 `src/fp16/fa_bwd_fp16_o33_*`；`docs/01` §14s。
  - **限制/下一步**：只挂到 BN=128 的 wgmma2b（S≥4096 默认快路）；bf16/fp8 的对应 TMA 与
    BN=64 的 `wgmma2` 几何待做；fp16 main 的墙仍是 L2 red（三条消 red 路已证伪）。

## 为什么 ours 比 FA/TE 慢这么多（归因）

「按 flash-attention 实现」指的是**算法与数据流照 FA**（preprocess 求 D、1colblock、recompute P、
dQ 累加/dK/dV 归约、causal 特化），**但计算后端与性能工程没有照 FA**。实测证据（读
`src/fp16/fa_bwd_fp16_kernels.cuh`）：

| 维度 | FA2/FA3 实现 | ours（fp16/bf16 golden） | 影响 |
|---|---|---|---|
| 5 个 GEMM | **Tensor Core**：`mma.m16n8k16` / `wgmma` + `ldmatrix` | **标量 CUDA core**：`__half2float(q)*__half2float(k)` + fp32 FMA | 峰值差 ~15×（67 vs 989 TFLOPS），实际差 ~65× |
| 数据搬运 | `cp.async`/TMA + 多级软流水/双缓冲 | 同步 `__syncthreads` 后读，无预取 | 延迟受限（ncu: L1/TEX 53.5%、Compute 8.4%、occ 6.25%） |
| dK/dV 归约 | 专用累加 + convert（deterministic 可选） | 每元素 `atomicAdd` | 竞争 + 非确定性 |
| preprocess LSE/D | 分块、向量化 | fp8 已 mma（O1，14–62×）；fp16/bf16 已 mma（O8，17–70×） | S=4096 时 preprocess 曾 > main |
| grid/occupancy | 大 grid、wave 饱满 | grid 小、waves 0.48、1 CTA/SM | SM 空转 |

**根本原因**：fp16/bf16 版是**正确性优先的标量 golden**（用来先把数学/对拍跑通），
真正的性能杠杆（张量核）只在 fp8 版实现了（main 13–20×）。所以差距是**预期内**的，
不是算法错了——数值对拍与 FA/TE 同量级就是证明。

**补齐路径（按回报排序）**：

- [x] **O5（最大杠杆）** 把 fp8 的 `mma.m16n8k16` + `ldmatrix` 后端移植到 fp16/bf16 反向
      （模板已有，改 dtype 与 smem 布局/padding）；预估 main **10–20×**。
      → **fp16 已完成（第二十六轮）**：`fa_bwd_fp16_mma_{onefile.cu, kernels.cuh+main.cu}`，
      5 个 GEMM 全 `mma.m16n8k16`；main S=512 11.8× / S=4096 14.9×（22.4/60.4 TF）。
      → **bf16（O5b）已完成（第二十七轮）**：`fa_bwd_bf16_mma_{onefile.cu, kernels.cuh+main.cu}`，
      由 fp16 版 dtype 参数化（`f16.f16`→`bf16.bf16`），布局/padding 逐字同构；main S=512 9.9× /
      S=4096 9.4×（22.6/60.9 TF）。**O5 fp16/bf16 全部收口。**
- [x] **O6** `cp.async` 双缓冲（对齐 fp8 的 O3）。**已完成（第二十九轮，fp16/bf16）**：
      `cp.async.cg` 16B 双缓冲 K/V、每 tile barrier 3→2；main **2.26–2.41×**（S4096
      4.51→1.87ms）、端到端 5.58→**3.04ms（45 TF）**，数值与 O5/O8 逐位相同；
       ncu `long_scoreboard` 7.35→1.12，新墙 = `wait`(fixed-latency)+`short_scoreboard`。
       代价 smem 66.56→83.97KB、3→2 CTA/SM。详见 `docs/01` §12、`docs/01b` §6g。
- [x] **O6b**：把 O6 的 K/V 双缓冲 smem 压回 3 CTA/SM。**已完成（第三十轮，fp16/bf16）**：
      ① GEMM3/GEMM4 的 A 改 `ldmatrix.x4.trans` 从 `Ps/dSs[BM][BN]` 直读、删掉 `PsT/dSsT`
      （smoke `fa_bwd_fp16_atrans_smoke.cu` 逐位一致）；② 只双缓冲 K、V 单缓冲在 GEMM2 后预取。
      smem **83.97→71.17KB、3 CTA/SM、occ 11.8%→16.9%**；main S4096 1.900→**1.860ms**、
      GQA kv4 0.380→**0.357ms**（S512 grid=128 单波 O6 反快，host 按网格自动选）。
      数值与 O5/O8/O6 逐位相同。ncu 新墙 = **L1/TEX 71.9% + L2 63.1% + `wait`**。
      详见 `docs/01` §12b、`docs/01b` §6h。
- [x] **O6c** 主 kernel tile 几何参数化 + 小网格并行度自适应。**已完成（第三十三轮 fp16 / 第三十四轮 bf16）**：
      2×2 warp 几何全部由 `(BM,BN)` 派生；host 自动档 `grid<132 且 S≤1024`→`(BM=32,BN=32,PIPE=1)`
      （提并行度）、`S≥4096`→`BN=64`；fp16 S=512 main **1.106×**、端到端 1.06×，S=4096 `BN=64` main 1.017×。
      同时**证伪**了「静态 mblk 重排负载均衡」（0–2%、正负不稳，`sched` 保留默认 0）。ncu：S=512
      achieved occ 6.24%→10.99%、Duration −15.5%；S=4096 `BN=64` L1/TEX 71.9→57.3% 但掉到 2 CTA/SM。
      **bf16 同款改造已完成（第三十四轮，单/两文件 device 逐字同构）**：S=512 main 0.0886→0.0800ms
      （1.11×，另一 session 1.18×）、端到端 1.06×，S=4096 1.018×、GQA 不变；数值逐位相同。
      详见 `docs/01` §13b、`docs/01b` §6j。
      > **O13（第四十轮）已修正本项的过时 auto 档**：O7c-PREL 之后 `(BM=32,BN=32,PIPE=1)` 不再最快
      > （S=512 MHA `(64,64,2)` 快 1.23×），改为取消 `BM=32` + `BN=64` 判据 `S≥4096||grid≤256||grid>600`。
      > 详见 `docs/01` §14f。
- [~] **O7** dQ/dK/dV 去 `atomicAdd`，改分块 `*_accum` + convert（确定性 + 消竞争）。
      **fp16/bf16 侧已做 O7c（第三十五轮）并证伪「减 red 事务数」**：float4 归约全几何变慢 1–7%
      （非事务数 bound）；真正有效的是 **LSE/D 预装寄存器**（main +14–19%、端到端 S4096 1.13×），
      但新墙是 **L2 71.6%（残余 dK/dV red）+ `wait` + 低 occupancy（2 CTA/SM）** ⇒ 转 **O9**。
      完整 dQ/dK/dV 去原子（分块 `*_accum`+convert）仍列 backlog（回报低于 O9）。详见 `docs/01` §14。
- [x] **O8** preprocess 的 LSE/D 改 mma 分块（对齐 fp8 的 O1）。**已完成（第二十八轮，fp16/bf16）**：
      `lse_mma_kernel<128>`（`mma.m16n8k16` QKᵀ + online-softmax + 4-lane `shfl`）+ 独立
      `delta_kernel<128>`；preprocess 17–70×、端到端 **4.4–13.2×**（S4096 total 73.3→5.58ms，
      24.6 TF；瓶颈回落 main），数值与 O5/O5b 逐位相同。详见 `docs/01` §11、`docs/01b` §6f。
- [x] **O8b** LSE 预处理负载均衡 + `cp.async` 双缓冲。**已完成（第三十一/三十二轮，fp16+bf16）**：
      镜像配对（`m` 与 `nblk-1-m`，工作量恒 `nblk+1`）+ K 的 `cp.async.cg` 16B 双缓冲；
      **lse S4096 0.985→0.348ms（2.81×，fp16）/ 0.970→0.350ms（2.77×，bf16）**、
      端到端 **2.952→2.336ms（58.8 TF，FA3 的 6.9%，fp16）/ 2.961→2.343ms（58.7 TF，6.8%，bf16）**，
       `long_scoreboard` 2.19→0.34、Waves 1.29→0.97，数值逐位相同。详见 `docs/01` §13、`docs/01b` §6i。
- [x] **O11** fp8 LSE 补上 O8b + 三 dtype 快速 exp/log。**已完成（第三十八轮）**：
      `lse_mma_kernel_bal<HD,PIPE>`（镜像配对 + `cp.async.cg` 16B 双缓冲；fp8 = 1B/元素，
      `HD/16` 个 unit）——lse **S4096 0.936→0.345ms（2.71×）**、preprocess **1.20→0.388ms（3.1×）**、
      端到端 **4.13→3.31ms（1.25×）**，数值与 O7 逐位相同；另把 fp16/bf16/fp8 的 `expf/logf`
      换 `__expf/__logf`（preprocess 再 ~8%）。详见 `docs/03` §19、`docs/01` §14d、`docs/01b` §6n。
- [~] **O9**（对标 FA3）TMA + `wgmma` + warp specialization 多级流水。
      **O9a 已完成（第三十九轮）**：建立 Hopper `wgmma + SW128` 数据通路（冒烟 PASS +
      `lse_mma_kernel_bal_wgmma`，L1/TEX 37.6→18.1%、LSE 1.05×、数值逐位相同）；
      但 LSE 是 softmax epilogue bound，收益有限 ⇒ **O9b**。
      **O9b 第一步已完成（第四十一轮，fp16）**：主 kernel 的 **GEMM1/2（`S=QKᵀ`、`dP=dO·Vᵀ`）**
      换成 `wgmma.m64n64k16`（Q/dO/K/V 存 SW128；GEMM3/4/5 的转置 B 用 `ldmatrix.x2.trans` 从
      同一 SW128 tile 读；冒烟逐位 PASS）。main **S512 1.092×、S4096 1.046×**（数值与 O13 逐位相同）；
      ncu `wait` 1.94→1.50、smem 105.5→101.4KB，但 **GEMM3/4/5 仍 mma、dK/dV 仍跨 CTA 原子 ⇒
       L2 74% 与 2 CTA/SM 没变**，收益有限。**bf16 版已完成（第四十二轮）**：把 fp16 O9b
       逐字 dtype 参数化到 bf16（同为 2 字节，SW128/描述符/累加器映射逐字节同构），main
       **S512 1.089× / S4096 1.021×**、数值与 O5b~O13 逐位相同、ncu 逐项一致（详见 `docs/01b` §6q）。
        **O9b-2（第一步已完成，第四十三轮）**：GEMM3/4/5 用 **MN-major 描述符（对 K-major SW128 tile
        转置读，FA3 `dKV_swapAB` 同思路）** 上了 wgmma，5 个 GEMM 全 wgmma；冒烟
        `fa_bwd_fp16_wgmma_bwd_smoke.cu` 逐位 PASS、数值与 O5~O13 逐位相同；**但性能中性偏负**
        （S4096 1.50–1.52 vs mma 1.47–1.48），ncu 证实**墙不在 GEMM 指令**（short_scoreboard 0.56→0.29），
        而是 **L2 的 dK/dV 跨 CTA 原子（~70%）+ 99KB smem 锁死的 2 CTA/SM**。详见 `docs/01` §14h。
         **bf16 版 O9b-2 已完成（第四十四轮）**：把 fp16 O9b-2 逐字 dtype 参数化到 bf16，数值与
         O5b~O13 逐位相同、main S4096 0.979×/S512 0.960×、ncu 与 fp16 逐项一致（详见 `docs/01b` §6r）。
          **O9b-2b（下一步，第五十一轮已重定）**：**先做跨 warpgroup 归约（BM=128、2 warpgroups，
          dK/dV red 字节砍半）**——这是打掉 L2 原子墙的唯一真杠杆（`docs/01` §14i）。TMA 化
          Q/K/V/dO 通路已由 **O15a** 建好（冒烟逐位 PASS），但要落进主 kernel 需先把 HD=128 的
          K-major tile 改成 **2×K=64 chunk**（TMA box 内维 128B=64 元素，`docs/01` §14i.2）。
          > **O7e（第四十五轮）修正了 fp8 侧的优先级**：O7b 针对的 L2 墙已不是头号杠杆
          > （见下），**fp8 应先做 `wgmma`（O9c）**；fp16/bf16 的 O9b-2b 仍按原计划。
- [~] **O9c（fp8 main 的 Hopper wgmma）**：把 fp8 `mma.m16n8k32` 换 `wgmma.m64nNk32`
      （SS 直读 smem 描述符，消 `ldmatrix`/配对副本、压 smem 冲更高 occupancy）。
      依据：O7e 定位 fp8 main 第一墙 = **L1/TEX 66–71%（`ldmatrix`+smem）**，只有 wgmma 能同时
      消 L1 访存与降 smem。可复用 fp16/bf16 O9a/O9b 的 SW128 数据通路（fp8 SW128 与 bf16 逐字节同构，
      仅 asm 尾部操作数 `p, scaleA, scaleB` 不同，见 `agent_skills/kernel-opt.md`）。
      **第一步已完成（第四十六轮）**：`fa_bwd_fp8_mma_smoke` 式冒烟 `fa_bwd_fp8_wgmma_smoke.cu`
      （`m64n64k32` e4m3×e4m3 / e5m2×e4m3，逐位 PASS）+ **LSE 上验证**
      `lse_mma_kernel_bal_wgmma<HD,PIPE>`（SW128 + 镜像配对 + `cp.async` 双缓冲）：event LSE
      S=4096 **1.275×**、ncu Duration 355.87→279.07µs（regs 77→64、理论 occ 37.5→50%）、
      端到端 S=4096 1.03×，数值与 O7e 同水平。**主 kernel 5 个 GEMM 仍 mma** ⇒ **O9c-2**
      把 GEMM1/2 上 `wgmma`（对齐 fp16 O9b）。详见 `docs/03` §21。
      **O9c-2 第一步已完成（第四十七轮）**：主 kernel GEMM1/2 换 `wgmma.m64n32k32`
      （Q/dO/K/V 存 SW128、fold 与 GEMM3/4/5 逐字复用 mma 版），main **S512 1.038× /
      S1024H32 1.052× / GQA 1.057× / S4096 1.056×**，数值与 mma 版同量级；ncu 第一墙仍是
      **L1/TEX 65%（GEMM3/4/5 的 `ldmatrix` + fold）**。**O9c-2b（GEMM3/4/5 上 wgmma）已确认
      硬件阻塞（第四十八轮）**：fp8 wgmma 无转置操作数（CUTLASS 全为 `_SS_TN`、无 `tnsp`），
      MN-major 转置读在 fp8 ISA 上不存在；物理转置需 `BM=BN=128`、smem >110KB，且 fp16 O9b-2
      已实测该路中性偏负 ⇒ 不再尝试，见「阻塞」与 `docs/03` §23.6。
      **O12 已完成（第四十八轮）**：主 kernel LSE/D 预装寄存器（fp16 O7c-PREL 的 fp8 版），
      main **1.04–1.13×**、uncoalesced global −87.6%，数值与历史逐位一致（`docs/03` §23）。
      详见 `docs/03` §22。
- [x] **O14（第四十九轮）** fp8 输入量化向量化（warp-per-row）。`quantize_row_kernel` 每行一个
      CTA（smem 归约 + 7×`__syncthreads`）改为 **warp-per-row `float4` + `__shfl_xor` 树 +
      `uchar4`**；quant **1.22–2.65×**、端到端 S=4096 **1.048×**（ours/TE FP8 5.22×→4.98×）、
      S=1024H32 1.090×、S512 1.082×；8 个量化输出**逐字节 mismatch=0**；ncu 墙移到 DRAM 71%
      （带宽上限）、指令数 −77%。另证伪 convert `float4`（中性）。详见 `docs/03` §24、`docs/04` §2.3。
- [ ] 目标：fp16/bf16 main ≥ 0.5× FA2 → 逐步逼近 FA2/TE。


## 目标形状 & FA/TE 归因（已分析，见 docs/06）

- [x] 纯反向口径基准 `harness/fa_vs_te_bwd_only.py`：FA2.7.4 217–377 TF vs TE2.14 305–618 TF（TE 1.2–1.6×）
- [x] kernel/SASS/SOL 归因：FA2.7.4=SM80 `HMMA/LDSM/LDGSTS`，TE=cuDNN SM90 `UTMALDG/WARPGROUP`(wgmma+TMA)；
      FA 对 GQA/MQA 多一个 `reduce` kernel。详见 `docs/06`。
- [x] FA3（SM90）编译并三方对比：**FA3 > TE > FA2**（详见 docs/06）：GQA/MQA FA3 355–438TF vs TE 307–356TF；MHA S4096 FA3 850TF vs TE 618TF。
- [ ] 后续：ours 对标升级为 Hopper 路线（O5 mma → O9 wgmma+TMA）；GQA KV 归约放进 kernel。

## 下一步（明确到可执行）

> **当前冲刺（按序，把 ours 性能推到 FA3/TE 水平；这是最高优先级，别再被其它任务打断）**：
>
> **O17 已完成（第五十二轮，fp16）——唯一真杠杆落地**：`fa_bwd_fp16_wgmma2_kernel<HD>`
> （BM=128、2 warpgroups、256 线程，`--wg2`，`sm_90a`+`-DFA_WGMMA`）：两组各算自己 64 行 Q 的
> P/dS，**dK/dV 只由 wg0 对全 BM=128 归约**（两个 m64 半进同一 wgmma 累加器），每个 KV 元素只
> `red` 一次 ⇒ **ncu 实测 `red` 102.2M→51.9M（0.508×）、`read` 0.50×、L2 71.8%→54.7%、
> Duration 1.48→1.00ms**；main S4096 **1.57×（140.5 TF）**、GQA 1.47×、MQA 1.54×、S512 1.11×，
> 端到端 S4096 1.427ms、**为 FA3 的 4.4×**（O9b ~6.0×）；数值 vs ref 逐位一致。前置冒烟
> `fa_bwd_fp16_wgmma2_smoke.cu` 逐位 PASS。
> **O17-bf16 已完成（第五十三轮）**：把 fp16 O17 逐字 dtype 参数化到 bf16
> `fa_bwd_bf16_wgmma2_kernel<HD>`（`--wg2`）：单/两文件 device 逐字一致、数值 vs ref 历史逐位
> 一致；ncu 与 fp16 逐项一致（`red` 102.2M→51.9M=0.508×、`read` 0.50×、L2 71.6%→54.6%、
> Duration 1.48→1.00ms）；main S4096 **1.516×（139.9 TF）**、GQA/MQA 1.50–1.53×、S512 1.11×，
> 端到端 S4096 **1.4309ms（96.05 TF）、为 FA3 的 4.45×**。详见 `docs/01b` §6s、`docs/04` §2.2/§3。
> **O17b 已做（第五十四轮）—— 负结果**：BM=256 / 4 warpgroups 把 red 精确再砍半
> （51.90M→26.74M），但 **512 线程把每线程寄存器上限压到 128**，dQ 累加器（64）+ 两条 wgmma
> 累加器（64）就吃满、必然 spill；local 流量占 L2 ~48%，净 Duration 反而 +21%（S4096）。
> ⇒ **「放大 BM」在本卡走不通**，寄存器文件是硬墙（不是 smem）。详见 `docs/01` §14k。
> **O17-2 已完成（第五十五轮，fp16/bf16）**：把 O17 phase B 的 GEMM3(dV)→wg0、GEMM4(dK)→wg1
> （原版只有 wg0 串行做 dV+dK，张量工作量 3:1、wg1 空等）。**red 字节完全不变**（51.9M），
> **stall `barrier` 1.61→0.46（−3.5×）**、Duration 997.7→**982.4µs**；main S4096 **1.013×（fp16）/
> 1.038×（bf16）**、GQA kv4 1.021×/1.051×，数值与历史**逐位一致**。详见 `docs/01` §14l、`01b` §6t。
> **O18 已完成（第五十六轮，fp16）**：把 O17/O17-2 的 kv-tile 从 BN=64 翻到 **BN=128**
> （`m64n128k16`，§14k.7 item 3）——tile 数减半 ⇒ barrier/`cp.async.wait`/wgmma commit-wait
> 序列减半。冒烟 `fa_bwd_fp16_wgmma2b_smoke.cu` 逐位 PASS；`fa_bwd_fp16_wgmma2b_kernel`
> （单/两文件，`--wg2bn`，230 regs/224KB/1 CTA/SM）。**main MHA S4096 1.029×（142.9 TF）、
> S512 1.028×**，GQA/MQA 中性（保持 O17）；ncu Duration 982.4→**951.1µs**、**`red` 逐字节不变**
> （BN 不动归约结构）。详见 `docs/01` §14m、`docs/04` §2.1/§3。
> **O18-bf16 已完成（第五十七轮）**：把 fp16 O18 逐字 dtype 参数化到 bf16
> （`fa_bwd_bf16_wgmma2b_kernel`，`--wg2bn`），数值与历史逐位一致；**main MHA S4096
> 1.025–1.029×（143.8 TF）/ S512 1.030×**，GQA/MQA 中性；ncu `red` 逐字节不变、Duration
> 982.2→952.5µs。详见 `docs/01b` §6u、`docs/04` §2.2/§3。
> **O23 已完成（第六十三轮）**：fp16/bf16 的 Hopper 快路（主 kernel wgmma2/wgmma2b + LSE wgmma）
> 在 `-DFA_WGMMA` 构建下**默认打开**（对齐 fp8 O22），端到端 **1.08–1.43×**（MHA S4096
> 1.9450→1.3641ms），`--wg2=0 --wg2bn=0`/`--lsewgm=0` 保留 mma 对照，纯 `sm_90` 不变；
> 数值逐位一致，默认档墙仍是 L2 red + 1 CTA/SM。详见 `docs/01` §14n、`docs/01b` §6v。
> **O7b 已完成（第六十四轮）——负结果 + 确定性能力**：把 dK/dV 的跨 CTA `red` 换成
> per-(b,h,mblk) partial 覆盖写 + `dkv_reduce_kernel` 二次归约（`--det=1` opt-in）。`red`
> 51.9M→**0**、主 kernel **1.20×**（958.6→802.0µs）、**结果逐位可复现**；但二次归约是纯 DRAM
> 带宽 bound（90.4%、384.9µs）⇒ `main+reduce` S4096 **0.810×**（S512 1.02×）。
> 详见 `docs/01` §14o。
> **O24 已完成（第六十五轮）**：清非-main 开销——`delta_kernel`→warp-per-row（S4096
> 42.5→**14.5µs，3.35×**、DRAM 74% bound）+ D=128 wgmma2/2b 主 kernel 直写 fp16 dQ、convert
> 跳过 dQ；端到端 fp16/bf16 S4096 1.037×/1.020×、S512 1.05×、GQA 1.05–1.08×，数值逐位不变。
> 详见 `docs/01` §14p、`docs/01b` §6w。**main 的墙（L2 red）+ 1 CTA/SM 仍未动。**
> **O25 已完成（第六十六轮）——cluster 分布式归约，负结果**：`red.shared::cluster.add.f32`
> （mapa 到 leader smem）+ 每 tile leader flush，`red` 精确减半（51.9M→26.7M）且数值逐位一致，
> 但逐元素远程原子延迟 + 每 tile 串行 flush ⇒ S4096 main **0.13×**（0.99→7.64ms）、
> `long_scoreboard` 0.75→5.83。**至此「放大 BM」（§14k）/partial+reduce（§14o）/cluster（§14q）
> 三条消 red 路全部证伪**；`--cluster` 保留 opt-in。详见 `docs/01` §14q。
> **O26 已完成（第六十七轮）**：fp8 `delta_kernel` 补上 fp16/bf16 O24 的 **warp-per-row 向量化**
> （`delta_warp_kernel<HD>`，2.39–3.22×，端到端 S4096 1.008×、55.8 TF，为 TE FP8 的 4.17×）；
> ncu 墙从 smem 归约（Compute 75%）移到 DRAM 带宽 77%（elementwise 上限）。**fp8 的 preprocess
> 三个子 kernel（quant/LSE/delta）至此全部向量化**；`docs/03` §31。剩余第一墙仍是 main 的
> **mma 依赖延迟 + 3 CTA/SM**。
> **O27 已完成（第六十八轮）**：fp8 fold 量化的「逐元素精确 fp32 除法」→「每行一次 `__frcp_rn`
> + 乘法」（新模板参数 `RCP`，默认 true；`--foldrcp=0` 供同 binary A/B）。`scX=amax/fp8max`
> 本是每输出行一个的常量，旧实现却让每个 `(m,j)` 元素都发一条精确除法（ptxas `prec-div`，
> ~10+ 指令）。**main 1.08–1.18×**（S4096 2.043→**1.758ms**、MQA 1.178×、kv8 1.152×）；
> 端到端 S4096 2.4637→**2.1770ms（63.1 TF）**，S512/S1024H32/GQA/MLA 全同向改善；
> **vs fp32 ref 的 dq/dk/dv 与历史逐位一致**（9 shape）；ncu Duration 2.06→1.78ms、
> executed inst 852.6M→**735.2M（−13.7%）**。为 TE FP8（同 session 0.5904ms/465.6TF）的
> **3.69×**（O26 4.17×）。详见 `docs/03` §32、`docs/04` §2.3/§3。
> **O29 已完成（第七十轮）**：fp8 自动 split-K **重新标定**——`ksplit` 目标由早期固定的
> `(D==128)?4096:132` 改为 `D==128 → S>=2048?8192:max(2048,4*base_grid)`、`D==512 → S/2`；
> 主 kernel **S4096 1.012× / S1024H32 1.139× / GQA kv4 1.198× / MLA S512H4 1.154× /
> MLA S1024H2 1.324×**，端到端全 shape 不回退。ncu（S1024H32 k=8→4）：`red` 扇区
> **26.74M→16.42M（0.61×）**、**L2 83.1%→57.6%**、Duration 332→305µs ⇒ 收益来自「少切 +
> 自动开 `use_regdq`」。同轮 `FA_ILV34`（GEMM3/4 交错）**负结果**（spill，0.95–0.96×）。
> 详见 `docs/03` §34、`docs/04` §2.3。
> **下一步（按回报）**：① **fp8 侧「提 occupancy」**（O19/O21/O22 一致：墙 = mma 依赖延迟 +
> 3 CTA/SM；**新查证：Qp/dOp 转置副本 17.4KB + Ps/Ss 18.9KB 使 WGMMA 版 smem 已达 68.6KB，
> 4×68.6=274KB > 232KB 上限 ⇒ 4 CTA/SM 在本卡 smem 与 168-reg 双重不可达**，此杠杆基本封死）；
> ② **fp8/主 kernel 跨-tile 软流水**（K/V 单缓冲被 dS3/Ap 复用，需先腾 smem）；
> ③ **TMA 化 operand** —— **O30 已完成 fp16 LSE 的第一步（第七十一轮）**：4D-TMA 载入 LSE 的
> Q/K（2×K=64 chunk、`SBO=1024`），LSE-only **1.30–1.36×**、指令数 −28.7%、数值逐位不变；
> 但墙不变（Compute ~58% + `wait`）。**O31 已完成 bf16 LSE 的 dtype 参数化（第七十二轮）**：
> 与 fp16 O30 逐字节同构（仅 `wgmma...bf16` + tensormap `BFLOAT16`），LSE-only **1.32–1.35×**、
> 指令数 **−28.7%**、端到端 **1.065–1.071×**（MHA S4096 1.3390→**1.2577ms**，109.3 TF，
> FA3 的 3.94×），`max_abs(tma-vs-wgmma)=0` 逐位一致；纯 `sm_90`/仅 `-DFA_WGMMA` 构建不变。
> 详见 `docs/01b` §6x。**O32 已完成 fp8 LSE 的 TMA（第七十三轮）**：fp8 一行 128B = SW128
> atom 整行 ⇒ Q/K 各一次 4D-TMA（UINT8 tensormap），LSE-only **1.06–1.10×**、`max_abs=0`、
> ncu `long_scoreboard 2.20→0.37`、新墙 = Compute ~57% + `wait`；详见 `docs/03` §35。
> **O33 已完成 fp16 主 kernel 的 Q/K/V/dO TMA（第七十四轮）**：**逐 atom TMA 复现交织布局**
> （一个 `[8,64]` box = 一个 SW128 atom ⇒ 描述符零改动），`--maintma`、main **1.042×**
> （142.6→148.6 TF）、`red` 逐字节不变、端到端为 FA3 的 3.77×；详见 `docs/01` §14s。
> **剩余**：bf16/fp8 的对应 dtype 参数化（fp8 SW128 为一整行 128B，逐 atom 更简单；fp8 尚需
> 处理 K/V 的 dS3/Ap smem 复用与 32 regs 寄存器预取）、BN=64 的 `wgmma2` 几何。
> **④（O27 新增，O28 已作废）fp16/bf16 的 fold 同理含逐元素精确除法**——**误记**：逐字核对
> `src/fp16,bf16/fa_bwd_*_kernels.cuh` 后确认 fp16/bf16 **没有 rowwise scale fold**（无量化），
> 逐元素除法只在 fp8。fp8 的 fold 除法 O27 已收口，转换指令 O28 也已向量化（MLA 1.03×、d128 中性）。
> **fp16/bf16 侧：L2 red（~72%）+ 1 CTA/SM 在本卡暂无便宜解法**（三条路已证伪），可转 TMA/软流水
> 或直接冲 fp8；fp8 侧墙是 **mma 依赖延迟 + occupancy**（O19/O21）。
> 详见 `docs/01` §14j/§14k/§14l/§14m/§14o/§14q、`docs/01b` §6s/§6t、`docs/04` §2.1/§2.2/§3。
>
> **O7e-3 已完成（第五十八轮）修正了 fp8 的瓶颈判断**：把 fp8 main 的 `Ps/Ss` epilogue
> store/回读 bank conflict 从 4-way 降到 2-way（PSS 33→37），shared store 冲突 −58%、
> L1/TEX 59.97→55.83%，**但 Duration 持平** ⇒ **fp8 main 的 L1/TEX 不是限速器**，真正的墙是
> **mma 依赖延迟（`wait` 1.56 + `short_scoreboard` 1.50，issue 45.9%）+ 3 CTA/SM**。
> 故 fp8 右侧的下一优先级应改为「**降 L2 的 dK/dV `red`（跨 CTA，49.8%）**」或
> 「**提 occupancy（须先砍 168 regs / 72.7KB smem）**」，而非继续抠 smem 冲突；
> `docs/03` §26。
> **O19 已完成（第五十九轮）对这两条做了判决——负结果**：fp8 跨 wg 归约版（BM=128、2 wg、256 线程）
> 把 dK/dV 的 `red` 砍到 **0.593×**、`read` 0.537×、**L2 49.9%→22.6%**，但代价是 **3→1 CTA/SM**
> （8 vs 12 warp/SM），同 session main **0.67–0.78×**（S4096 2.27→2.90ms）。⇒ **fp8 main 的墙不是
> L2 red，而是「mma 依赖延迟 + occupancy」**；**fp8 右侧不要再走「减 red / 放大 BM」**，下一杠杆是
> **提 occupancy（168 regs→≤128、72.7KB→≤58KB 才 4 CTA/SM）** 或 **减 mma 依赖 stall**。
> `red` 减半只对 fp16/bf16（red 占 L2 73%、且 2→1 CTA/SM 换得回来）成立。详见 `docs/03` §27。
> **O20 已完成（第六十轮）**：把「减 mma 依赖 stall」这条先做了一小步——**融合 mma 路径的
> GEMM1/GEMM2 epilogue**，`P` 留寄存器（`preg[2][2][4]`）不再回读 `Ps`（O7e-3 遗留的 12.78M
> wavefronts）。编译开关 `FA_FUSE_EPI`（默认 1）；**数值逐位不变**；ncu shared 总 wavefronts
> 169.07→160.55M、`op_ld` 冲突 −19.5%、`short_scoreboard` 1.50→1.41、Duration 2.25→2.21ms；
> **S4096 main 1.026×**、S1024H32 1.015×、kv8 1.017×，小 S/MLA 中性。墙仍是 `wait`+3 CTA/SM，
> 详见 `docs/03` §28。**fp8 若还要再上台阶，仍须解 occupancy 或做跨-tile 软流水**
> （K/V 单缓冲被 dS3/Ap 复用，跨-tile 重叠需先给 K/V 双缓冲腾 smem）。
> **O21 已完成（第六十一轮）——BN=64 负结果 + O21b 正结果**：① 先验证 fp16 O18 的「翻倍 KV-tile
> 减半串行相位」假设在 fp8 mma 路径是否成立：把主 kernel 的 BN 从 32 参数化到 64（GEMM1/2/3/4
> warp 块与 fold 的 j 覆盖全部由 `(BM,BN)` 派生；单/两文件 device 逐字一致）。**结果负**：
> S512 0.900×、S1024H32 0.920×、GQA kv4 0.920×、S4096 **0.816×**。ncu（S4096 同 session）：
> BN=32→64 把 regs 168→255、smem 72.70→**105.22KB**、occupancy **3→2 CTA/SM（18.05→12.28%）**；
> 虽 `Warp Cycles/Issued` 6.19→5.92（相位减半生效），但延迟受限 kernel 少 1/3 在飞 warp 更亏。
> ⇒ **fp16 O18 成立的前提是它从 1 CTA/SM 出发；fp8 已在 3 CTA/SM，翻倍 tile 必掉 occupancy**，
> 此路对 fp8 不成立（与 O19 一起双重证伪「放大 tile/减 red」）。② **O21b**：fp8 的 dQ/dK/dV 输出
> 本是 fp32、与累加缓冲同 dtype，`convert_kernel` 只是一趟纯拷贝 ⇒ 把 acc 指针别名到输出、main
> 直接 `atomicAdd`，消掉 convert。**端到端 S4096 2.7469→2.6497ms（1.037×）**、单文件 1.046×、
> GQA 1.037×，**数值逐位不变**；已设为默认路径（`--cvt=1` 供 A/B）。S4096 端到端 **2.65ms/52.2TF**、
> 为 TE FP8（0.5909ms/465TF）的 **4.5×**（O20 4.7×）。详见 `docs/03` §29。
> **O22 已完成（第六十二轮）——fp8 Hopper 路径默认化（正结果）+ 三组负结果判决**：把 fp8 早已实现、
> 却默认关的 Hopper `wgmma`（O9c 的 LSE + O9c-2 的主 kernel GEMM1/2）在 `-DFA_WGMMA`（`sm_90a`）
> 构建下**默认开**（新增 `--lsewgm=0/1`、`--wgmma=0/1` 同 binary A/B；`sm_90` 构建不变，device 逐字
> 一致）。同 session 端到端 total：**S512 1.060× / S1024H32 1.059× / S4096 1.085× / GQA kv4 1.071×**；
> S4096 **2.48ms / 55.3 TF**，为 TE FP8（0.5899ms/465.9TF）的 **4.21×**。同轮**否决**：`--regdq=0`
> （main 2.136 vs 2.480ms ⇒ 保留 REGDQ，local spill 比多发 dQ red 便宜）、`FA_ILV`（GEMM1/2 mma
> 交错，中性偏负）、ksplit 重标定（维持 k=4）、`PSS` 扫描（37 近最优）。**fp8 main 的墙仍 = mma 依赖
> 延迟（GEMM3/4/5 仍 mma，fp8 wgmma 无转置操作数=O9c-2b 硬件阻塞）+ 3 CTA/SM**，下一步仍是 occupancy
> （168→≤128 regs、72.7→≤58KB smem）或跨-tile 软流水。详见 `docs/03` §30、`docs/04` §2.3。
> 1. **O5 收尾**：fp16/bf16 反向用 `mma.m16n8k16`+`ldmatrix` 张量核后端。
>    进度：fp16 主 kernel **2.28→0.19 ms（512）/ 67.6→4.55 ms（4096），11.8–14.9×**；
>    **bf16 主 kernel 1.88→0.190 ms（512）/ 42.2→4.51 ms（4096），9.4–9.9×**（第二十七轮，单/两文件、
>    对拍/ncu/FA3-TE 对标齐备）。**O5 已完成**，转入 O8。
> 2. **O8 preprocess mma**：fp16/bf16 的 LSE/D 改 mma 分块（照搬 fp8 的 O1）。**已完成（第二十八轮）**：
>    `lse_mma_kernel` + `delta_kernel`，preprocess S4096 68.7→0.99ms（69.7×）、端到端 total
>    73.3→**5.58ms（13.1×，24.6 TF）**；数值与 O5/O5b 逐位相同、单/两文件一致；瓶颈已回落 **main**。
> 3. **O6**：`cp.async` 双缓冲（对齐 fp8 的 O3/O4）。**已完成（第二十九轮，fp16/bf16）**：
>    `cp.async.cg` 16B 双缓冲 K/V，main **2.26–2.41×**（S4096 4.51→1.87ms）、端到端
>    5.58→**3.04ms（45 TF，FA3 的 5.4%）**，数值逐位不变；ncu `long_scoreboard` 63%→~1 成。
>    **代价**：smem 66.56→83.97KB、3→2 CTA/SM。
> 4. **O6b**：把 K/V 双缓冲压回 3 CTA/SM（只双缓冲 K、或更省的转置布局）。**已完成（第三十轮）**：
>    K 双缓冲 + V 单缓冲后段预取 + `ldmatrix.x4.trans` 消 `PsT/dSsT`，smem 83.97→**71.17KB**、
>    3 CTA/SM；main S4096 1.900→1.860ms、GQA kv4 0.380→0.357ms（S512 单波走 O6，host 自动选）。
>    ncu 新墙 = **L1/L2 吞吐 + `wait`**（occ 已不是瓶颈）。
> 5. **O8b**：LSE 预处理的负载均衡 + `cp.async` 双缓冲（O6b 后 preprocess 又占端到端 34%）。
>    **已完成（第三十一/三十二轮，fp16+bf16）**：因果下第 `mblk` 个 CTA 做 `mblk+1` 个 K tile（尾波 50%），
>    改**镜像配对**（每 CTA 做 `m` 与 `nblk-1-m`，工作量恒 `nblk+1`）+ K 的 `cp.async.cg` 双缓冲；
>    **lse S4096 0.985→0.348ms（2.81×）/ 0.970→0.350ms（2.77×）**、端到端 **2.952→2.336ms（58.8 TF，6.9%）/
>    2.961→2.343ms（58.7 TF，6.8%）**，数值逐位相同。fp16/bf16 单/两文件均完成。
> 5b. **O6c**：主 kernel tile 几何参数化 + 小网格并行度自适应（fp16）。**已完成（第三十三轮）**：
>    2×2 warp 几何由 `(BM,BN)` 派生，host 自动档在 `grid<132 且 S≤1024` 用 `(BM=32,BN=32,PIPE=1)`
>    （grid 翻倍、并行度翻倍），`S≥4096` 用 `BN=64`；S=512 main **1.106×**/端到端 1.06×、S=4096
>    main 1.017×，数值逐位相同。**证伪**静态 mblk 重排（0–2%、正负不稳）。ncu：S=512 occ
>    6.24%→10.99%、Duration −15.5%；S=4096 `BN=64` L1/TEX 71.9→57.3% 但 2 CTA/SM，净 +1.5%。
>    **bf16 同款改造已完成（第三十四轮）**：S=512 main 0.0886→**0.0800ms（1.11×，另一 session 1.18×）**、
>    端到端 0.1603→**0.1501ms（1.06×）**、S=4096 1.018×、GQA 持平；数值逐位相同。详见 `docs/01b` §6j。
> 6. **O7c**（dQ/dK/dV 去原子 / 减 red 事务数）：**已完成（第三十五轮，fp16+bf16）并给出负结果**：
>    dK/dV 归约 float2→float4（quad `shfl`，`REDG.E.ADD.F32x4`）四个几何全变慢 1–7% ⇒ 该归约
>    **不是事务数 bound**；O7「去原子」这条路（分块 `*_accum`+convert）回报低于 O9，列 backlog。
>    同轮发现并合入真正有效的 **O7c-PREL（LSE/D 预装寄存器）**：main **+14–19%**、端到端 S4096
>    **2.376→2.094ms（1.13×）**、S512 ~持平，数值逐位相同；ncu 墙移到 **L2 71.6% + `wait` +
>    低 occupancy（105KB smem/250 regs→2 CTA/SM）**。详见 `docs/01` §14、`docs/01b` §6k。
> 7. **O9（当前第一优先级）**：`wgmma`+TMA+warp specialization，对标 FA3。
>    理由：O6c/O7c 已证明 `BN=64` 能降 L1/TEX，但 fp16/bf16 的 smem（105KB）与寄存器（250）
>    把 occupancy 锁死在 2 CTA/SM；**只有更低 smem 的数据通路（TMA 直写 smem + wgmma 免
>    `ldmatrix`/转置副本）才能同时拿低 L1/L2 与高 occupancy**。
> 目标：fp16/bf16 main 先到 FA2 水平，再逼近 FA3/TE；每步用 `harness/fa_vs_te_bwd_only.py`（纯反向、三列）验收。
>    **O9a 已完成（第三十九轮）**：先在 LSE（单 GEMM、无转置）上跑通 `wgmma.m64n64k16 + SW128`——
>    冒烟逐位 PASS、LSE L1/TEX 37.6→18.1%、1.05×、数值逐位不变；但 LSE 是 softmax epilogue bound，
>    收益有限。**O9b 第一步已完成（第四十一轮，fp16）**：主 kernel 的 GEMM1/2（`S=QKᵀ`、`dP=dO·Vᵀ`）
>    换成 wgmma（Q/dO/K/V 存 SW128；GEMM3/4/5 的转置 B 用 `ldmatrix.x2.trans` 从同一 SW128 tile 读，
>    冒烟逐位 PASS）；main S512 1.092× / S4096 1.046×、数值逐位不变，但 **GEMM3/4/5 仍 mma、
>    dK/dV 仍跨 CTA 原子 ⇒ L2 74% 与 2 CTA/SM 没变**。**O9b-2 第一步已完成（第四十三轮，fp16）**：
>    GEMM3/4/5 用 **MN-major 描述符（对 K-major SW128 tile 的转置读，FA3 `dKV_swapAB` 同思路）**
>    上了 wgmma，5 个 GEMM 全 wgmma；冒烟逐位 PASS、数值与 O5~O13 逐位相同；**但性能中性偏负**
>    （S4096 1.50–1.52 vs mma 1.47–1.48），ncu 证实 **short_scoreboard 0.56→0.29（ldmatrix 消掉）
>    而 L2 仍 ~70%，墙是 dK/dV 跨 CTA 原子 + 99KB smem 锁死的 2 CTA/SM**，不在 GEMM 指令。
>    **bf16 版 O9b 已完成（第四十二轮）**：逐字 dtype 参数化，main S512 1.089× / S4096 1.021×、
>    数值逐位相同、ncu 逐项一致（`docs/01b` §6q）。构建 wgmma 需 `-gencode=arch=compute_90a,code=sm_90a`（+ `-DFA_WGMMA`）
>    （`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh ...`）。
>    **O9c-2 第一步已完成（第四十七轮）**：fp8 主 kernel GEMM1/2 已上 `wgmma.m64n32k32`
>    （Q/dO/K/V 存 SW128；`smem 70.7→68.6KB`；fold + GEMM3/4/5 逐字复用 mma 版），
>    main **1.038–1.057×**、数值与 mma 版同量级；ncu 第一墙仍是 L1/TEX 65%。
>    **O9c-2b 已判为硬件阻塞（第四十八轮）**：fp8 wgmma 无转置操作数（CUTLASS 全 `_SS_TN`、
>    无 `tnsp`），MN-major 转置读在 fp8 ISA 上不存在；物理转置需 `BM=BN=128`、smem >110KB，
>    且 fp16 O9b-2 实测该路中性偏负 ⇒ **不再尝试**（见「阻塞」）。**O12 已完成（第四十八轮）**：
>    fp8 主 kernel LSE/D 预装寄存器（fp16 O7c-PREL 的 fp8 版），main **1.04–1.13×**、
>    uncoalesced global −87.6%，数值逐位一致（`docs/03` §23）。fp8 后续的 L1/TEX 墙改走
>    **fold 向量化 / TMA 化 operand / dK/dV 去原子**（非转置手段）。fp16/bf16 的 **O9b-2b**
>    （TMA + P/dS 双缓冲跨-tile 流水 + 压 smem 冲 3 CTA/SM）仍按原计划。
>    **O7e-2 已完成（第五十轮）**：用 ncu Memory Tables 把 fp8 main 的 L1/TEX 拆开，第一来源是
>    **shared load 冲突（93.1M 请求、2.1-way、62.8M 冲突）**——fold 读 `Ps/Ss` 列时 4 lane 组
>    m 间距 16（`16*PSS≡16`）恒撞。改 lane→m 映射为 `sub4*8+t+half*32`（bank 铺满 0..31）+ 8B/16B
>    向量写：**shared load 冲突 −53.5%、L1/TEX 64.8→60.0%、main 1.02–1.04×**，数值逐位不变。
>    同时证伪「fold 写折到 16B」（仅 1.007×、增大 spill）⇒ 墙在**读冲突**。`docs/03` §25。
>    fp8 剩余可动：**TMA 化 operand / dK/dV 去原子**（L1/TEX 60% + L2 50%）。
 >    **bf16 版 O9b-2 已完成（第四十四轮）**：
>    逐字 dtype 参数化，数值逐位相同、main S4096 0.979×/S512 0.960×、ncu 与 fp16 逐项一致（`docs/01b` §6r）。详见 `docs/01` §14g/§14h。
>
> **旁支已完成（第四十九轮 O14）**：fp8 **输入量化**改 **warp-per-row `float4` + `__shfl_xor`
> 树 + `uchar4`**（`quantize_row_warp_kernel`，无 smem/无 barrier），8 个量化输出**逐字节
> bitwise mismatch=0**；quant **1.22–2.65×**、端到端 **S4096 3.0617→2.9210ms（47.1 TF，
> ours/TE 5.22×→4.98×）/ S1024H32 1.090× / S512 1.082×**；ncu 墙从 **L1/TEX 73.7%+Compute
> 71.8%（DRAM 24%）** 移到 **DRAM 71.3%（带宽上限）**、指令数 −77%。另证伪 convert `float4`（中性）。
> 详见 `docs/03` §24、`docs/04` §2.3。
>
> **旁支已完成（第三十六轮 O5c）**：把 fp16/bf16 的 **MLA（head_dim=512）反向从标量升级为张量核**
> （`HD` 模板 128/512、GEMM3/4/5 N-tile 循环、dQ 全局累加），main 5.2–5.8×（fp16）/3.7–4.1×（bf16），
> 数值同噪声、MHA 回归逐位不变。详见 `docs/01` §14b、`docs/01b` §6l、`docs/04` §7.8。
> MLA 的下一步（降 smem 冲 2 CTA/SM / split-KV）列 backlog。
>
> **旁支已完成（第三十七轮 O10）**：fp16/bf16 的 **Q/dO 载入向量化（16B `cp.async`）+ 与 K/V
> 重叠 + dQ `float2` 写回**（`qdo_issue_async`、LSE 的 Q 同样向量化）。端到端 **1.04–1.32×**、
> 数值逐位不变；ncu `long_scoreboard` 1.38→0.89、指令数 −2.5%。这是 O9 之前对**现有数据通路**
> 的最后一次清理——**墙仍是 `wait` + L2 + 2 CTA/SM，只有 O9 的更低-smem 数据通路能继续推进**。
> 详见 `docs/01` §14c、`docs/01b` §6m。
>
> **旁支已完成（第三十八轮 O11）**：给 **fp8 的 LSE 补上 O8b**（镜像配对负载均衡 + `cp.async.cg`
> 16B 双缓冲，fp8 一行 `HD` 字节 = `HD/16` 个 16B unit）——fp8 preprocess `S=4096` **1.20→0.388ms
> （3.1×）**、端到端 **4.13→3.31ms（1.25×）**，数值与 O7 逐位相同；另把 fp16/bf16/fp8 的
> `expf/logf` 换 `__expf/__logf`（preprocess 再 ~8%）。同轮证伪两条主 kernel 假设
> （`(BM=32,BN=64,PIPE=2)` 慢 1.75×、`cp.async .L2::256B` 无变化）。详见 `docs/03` §19。
>
> **旁支已完成（第四十轮 O13）**：修正 O6c 的过时 auto tile（O7c-PREL 之后 `(32,32,1)` 不再最优）
> + 去掉 HD=128 的 dQ memset + `convert` 向量化。端到端 S512 fp16 **1.12×** / bf16 **1.14×**，
> S1024 部分 shape 1.03–1.06×，数值逐位不变；详见 `docs/01` §14f、`docs/01b` §6p、`docs/04` §2.1/§2.2/§3。

> **用户新增需求（已完成）**：让 ours 支持 P5 的生产形状（GQA/MQA + MLA head_dim=512）——
> 目前 FA/TE 做不了 MLA 反向，ML A 的性能数字只能由 ours 提供。

- [x] **P5-1（优先）GQA/MQA**：改 fp16 反向（`src/fp16/`）支持 `Hkv`（由 `k.npy` 的 head 维读出），
      K/V 索引 `h/(H/Hkv)` 映射到 KV 头；对拍 4 个 GQA/MQA dump case 的 `ref_dq/dk/dv.npy`，
      再用 `harness/fa_bwd_bench.py bench --requested` 对标 FA/TE（目标 ≥ FA）。
      **已完成（第十六轮）**，详见 `docs/01-fp16-bwd-impl.md` §8。
- [x] **P5-2（优先）MLA head_dim=512**：fp16 反向支持 D=512（smem/寄存器容量、
      Q 常驻 / 更小 K/V 分块 / 可能需 split-K 或 KV 分片），对拍 3 个 MLA case 的 ref；给出性能数字。
      **已完成（第十七轮）**：`HD` 模板化 + `BM` 随容量选择（128→64 / 512→16），单/两文件；详见
      `docs/01-fp16-bwd-impl.md` §9。
- [x] **P5-3**：bf16/fp8 复用同一 `Hkv`/`HD` 改造；fp8 GQA/MQA 对拍 vs TE FP8。
      （bf16 的 `Hkv`+`HD` 均已完成；**fp8 的 `HD`（MLA head_dim=512）也已完成（第二十一轮）**。）
      （P5-1/P5-2 的 fp16 改造已完成：`Hkv` 入参 + `hkv=h/(H/Hkv)` 映射 + `dk/dv` 按 `B*S*Hkv*D`
      分配、`convert` 收 `n_q/n_kv`；`HD`/`BM` 模板。bf16 可直接照搬，fp8 在已优化的 mma 路径上
      加同一映射与更大 head_dim 的分块。）
      **进度（第十八/十九/二十/二十一轮）**：fp8 的 `Hkv`（GQA/MQA）已完成（单/两文件，`docs/03` §13、`docs/04` §7.4）；
      **bf16 的 `Hkv`（GQA/MQA）也已完成**（`docs/01b` §6c、`docs/04` §7.5）；
      **bf16 的 `HD`（MLA head_dim=512）也已完成（第二十轮）**；
      **fp8 的 `HD`（MLA head_dim=512）也已完成（第二十一轮）**：`Fp8Cfg<HD,BM,BN>` 模板化，
      GEMM1/2 加长 k-loop、GEMM3/4/5 加 N-tile 循环（每 128 维一遍），`HD=128` 回归逐位不变、
      `HD=512` smem 223.2KB（1 CTA/SM），3 个 MLA case 对拍 ref 同 fp8 噪声（2.2–4.5e-1，分段均匀），
      main 比 fp16/bf16 标量 MLA 快 6.5–7.3×；`docs/03` §14、`docs/04` §7.7。
- [x] **P5-4** fp8 GQA/MQA 对拍（vs TE FP8）与性能 —— 第十八轮完成。
- [~] **MLA 优化（backlog）**：fp16（P5-2）、bf16（第二十轮）、fp8（第二十一轮）均已给出 MLA 的
      ref 对拍与 ours 性能数字。**fp16/bf16 上张量核已完成（第三十六轮 O5c，`docs/01` §14b / `01b` §6l）**：
      `HD` 模板扩到 128/512、GEMM3/4/5 N-tile 循环、dQ 全局累加；main 5.2–5.8×（fp16）/
      3.7–4.1×（bf16），数值同噪声、MHA 回归逐位不变。fp8 MLA 早在第二十一轮即为张量核。
      **剩余**：三者都是 1 CTA/SM、grid 不足一个波（ncu Waves 0.48、bound=延迟+低并行度）⇒
      下一步**降 smem 冲 2 CTA/SM** 与 **split-KV 提高 grid**；fp8 侧最大障碍是 `Kt/Qt/dOt`
      转置副本（O4b 已把 fp8 MLA smem 223→201KB，仍需 >116KB）。对标 FlashMLA 的分块/流水/persistent。
> 以下为既有 fp8 优化 backlog（P5 已全部收口，现在可与 MLA 优化合并推进）。

> P5-3 已完成，ROADMAP 里的「P 项」全部收口，后续为**优化 backlog**。
> **O1/O2/O3/O4a/O2b/O4d/O4c/O4b/O7 已完成**。O4c 把 L2 原子流量砍半（L2 81.5%→57.9%）；
> O4b 把转置副本换成 K 配对 + `ldmatrix.x2.trans`（`op_st` 冲突 −66%、L1/TEX 81.3%→69.7%）；
> **O7 把 dQ 的每-tile 归约折成「CTA 内寄存器累加 + 单次 flush」**（残余 red 再砍半：
> L2 red 204.5M→108.5M、**L2 墙 69.1%→43.8%**、main S=4096 1.10×，且保住 3 CTA/SM）。
> **O8（preprocess mma）与 O6（main cp.async 双缓冲）均已完成**：O5 把 fp16/bf16 main 换成
> 张量核后，O8 打掉标量 preprocess（S=4096 68.7→0.99ms），O6 用 `cp.async.cg` 双缓冲把 main 的
> **全局访存延迟 bound**（ncu `long_scoreboard` 63%）消掉（S=4096 4.51→1.87ms）——端到端
> 5.58→**3.04ms（45 TF，FA3 的 5.4%）**，数值逐位不变。**O6b 又把 K/V 双缓冲 smem
> 83.97→71.17KB（K 双缓冲 + V 单缓冲后段预取 + `ldmatrix.x4.trans` 消转置副本）回到 3 CTA/SM**，
> main S=4096 1.90→1.86ms、端到端 3.04→**2.95ms（46.6 TF）**。**当前新墙 = L1/TEX 72% +
> L2 63% 吞吐 + `wait`**。**O8b（LSE 负载均衡+cp.async）与 O6c（tile 几何参数化 + 小网格
> 并行度自适应）也已完成**：O6c 让 S=512 main 1.106×（grid 翻倍、occ 6.24→10.99%），并证伪了
> 静态 mblk 重排；`BN=64` 能把 S=4096 的 L1/TEX 71.9→57.3% 但掉到 2 CTA/SM（净 +1.5%），
> **要同时拿低 L1 与高 occupancy 须等 O9 的更低 smem 数据通路**。**O6c(bf16) 也已完成（第三十四轮）**
> （S=512 main 1.18×、端到端 1.06×，S=4096/GQA 持平，数值逐位相同）。**O7c（fp16+bf16，第三十五轮）
> 完成了「O7 去原子」这一路的判决**：dK/dV float4 归约全几何变慢 1–7%（**非事务数 bound**），
> 但同轮合入 **LSE/D 预装寄存器**（main +14–19%、端到端 S4096 2.376→**2.094ms，1.13×**，
> 数值逐位相同）；ncu 墙从 L1/TEX 移到 **L2 71.6% + `wait` + 低 occupancy（105KB smem/250 regs→
> 2 CTA/SM）**。**下一项 = O9（wgmma+TMA，当前唯一能同时降 smem 与提 occupancy 的杠杆）**；
> fp8 侧剩余 108.5M red（dK/dV 跨 mblk/hkv 竞争）可做 O7b；MLA 降 smem / wgmma+TMA 亦在列。
> 每轮挑一项做成完整增量（代码 + 实测 + ncu + 文档 + commit）。

- [x] **O1（端到端第一瓶颈）preprocess 分块/向量化**：S=4096 时 preprocess ~71ms >> main 10.2ms。
      **已完成（第十一轮）**：新增 `lse_mma_kernel`（`mma.m16n8k32` E4M3×E4M3 分块 Q·Kᵀ +
      online-softmax + warp 内 `shfl` 归约）与独立 `delta_kernel`（rowsum(dO∘O)）。
      preprocess：S=512 1.1991→**0.0850ms（14.1×）**、S=1024H32 8.8033→**0.2162ms（40.7×）**、
      S=4096 70.7674→**1.1370ms（62.2×）**；total 1.71→0.60 / 10.72→2.21 / 80.86→**11.47ms（7.05×）**。
      数值与 P3-5 逐位相同。ncu（lse, S=4096）：DRAM 0.46% / Compute 39.4% / L1TEX 20.6% /
      occ 29.5%（理论 43.8%，被 72 regs 卡）/ Waves 1.11（尾波）→ bound = 延迟/并行度 + 尾波。
      端到端瓶颈回落 main。详见 `docs/03-fp8-bwd-impl.md` §9。
- [x] **O2（fp8 main 提 occupancy）**：**已完成（第十二轮）**。把 `dS3`/`Ap` 两个 per-tile 小缓冲
      折叠进「本 tile 内已死亡」的 `Ks`/`Vs` 尾部（`Ks` 只用于 GEMM1、`Vs` 只用于 GEMM2，各有 sync 隔开）。
      smem 80.13→**75.01 KB**，驱动自动选 233.47 KB carveout，Block Limit Shared Mem 2→**3**：
      theoretical occupancy 12.5%→**18.75%**、achieved 11.8%→16.85%、active warps/sched 1.91→2.69、
      ncu Duration 10.27→**8.85 ms**。main：S=1024H32 1.814→**1.526 ms（1.19×）**、
      S=4096 9.924→**8.562 ms（1.16×）**；S=512 不变（grid=128<132 SM，grid-bound）。
      数值与 P3-5/O1 逐位相同；单文件与两文件同步。详见 `docs/03-fp8-bwd-impl.md` §10。
- [x] **O3（fp8 main 加流水）**：**已完成（第十三轮）**。K/V 改成向量化 4B 读 + 寄存器双缓冲
      预取（不加 smem，保持 3 CTA/SM，regs 128→168）；main 1.15–1.18×。消融显示收益主要来自
      向量化（1.13–1.16×），预取仅再 ~2–3%。ncu `long_scoreboard` 2.94→2.21，新墙 = CTA barrier
      （5.13）+ short_scoreboard（4.12）。详见 `docs/03-fp8-bwd-impl.md` §11。
- [x] **O4a（barrier 合并 + fold 并行）**：**已完成（第十四轮）**。三处逐位等价改动：
      ① 删 GEMM1/GEMM2 之间的 barrier（GEMM2 读回的 `Ps[r*BN+c]` 是本线程同一地址，无线程间
      依赖）；② 合并 prologue 两处 barrier；③ fold 改全 128 线程并行（warp 内 `shfl_xor`
      归约 amax，`fmaxf` 结合律 ⇒ 逐位相同）。主 kernel `__syncthreads` 7→5。
      **数值与 O3 逐位相同**；main **1.21–1.54×**（S=512 0.3947→0.2566、S=1024H32
      1.3202→1.0642、S=4096 7.7391→**6.4192**），端到端 total 0.5403→0.4030 / 1.7278→1.4408 /
      9.0992→**7.8488ms（17.51 TF）**。ncu：**barrier 5.13→0.46（基本清零）**，墙移到
      short_scoreboard（4.12→5.40）与 L1/TEX 65.59→78.51%；Duration 7.57→6.55ms。
      详见 `docs/03-fp8-bwd-impl.md` §12。
- [x] **O4b（转置副本 → `ldmatrix.x2.trans` + K 配对布局）**：**已完成（第二十四轮）**。
      **先纠正前提**：fp8 里 A/B 主序相反、且 `ldmatrix.trans` 的配对方向是 N，**转置副本
      无法整个消掉**（须存两种主序）。真实收益：把「逐字节 scatter 写」换成 **4B 交织写**
      （`__byte_perm 0x5140/0x7362`）、配对数组比 `[HD][K+16]` 更小。先写
      `src/fp8/fa_bwd_fp8_trans_smoke.cu` 验证「`[K/2][N]` K 配对 + `ldmatrix.x2.trans`」
      与现有 `ldmatrix.x2` **逐位一致（PASS）**；再合入：`Fp8Cfg` 的 `KTS/QTS→PSLD`、
      `KVU→NPU`，新增 `mma_block_bt`，Q/dO 与 K/V 载入改「行对 unit」（O3 预取保留）。
      smem **75520→70656B（d128）/ 229120→205824B（MLA）**；`op_st` bank conflict
      **206.4M→69.4M（−66%）**、L1/TEX **81.3%→69.7%**、short_scoreboard 3.50→2.73；
      main **1.17–1.76×**（S=512 0.1091→0.0733、S=1024H32 0.6765→0.4552、
      S=4096 3.4553→2.9473、MLA S1024H2 0.3896→0.3251、MLA S256H2 0.0916→0.0522、
      GQA 0.5923→0.4404），数值与 O4c **逐位相同**。新墙 = L1/TEX 69.7% + L2 69.1%（残余
      red）+ short 2.73。MLA 即使去掉三个配对数组仍 >116KB → 冲 2 CTA/SM 需继续降 smem。
      详见 `docs/03` §17、`docs/04` §2.3/§3。
- [x] **O2b**：split-K 切块数自动选择（N 方向切块，`dq/dk/dv` 跨 CTA atomic 汇总）。
       **已完成（第二十二轮）**：先 sweep（k=1/2/4/8/16）× 6 个 shape，规律是
      **d128（3 CTA/SM）目标 grid≈4096、MLA（1 CTA/SM）目标≈132（1 个波）**；新启发式
      `TARGET=(D==128)?4096:132`、`k=clamp(TARGET/base,1,16)` 向下取 2 的幂。旧自动档（cap4、
      `ceil(396/base)`）在大 S 和 MLA 小 S 上明显次优：main 相对旧自动档 d128 1.08–1.21×、
      MLA S256H2 1.56×。ncu：S=512 grid 128（occ 6.25%）→2048（occ 17.83%）。详见 `docs/03` §15。
- [x] **O4d**：`Ps/Ss` 两个 fp32 `[BM][BN]` 缓冲行距 `BN→BN+1`（33 word，奇数）消 bank conflict。
      **已完成（第二十二轮）**：S=4096 ksplit=1 `op_ld` 冲突 199.9M→77.3M（−61%）、`op_st`
      231.6M→198.4M（−14%）、总多余 wavefronts 613.6M→456.0M（−26%），main 6.56→5.85 ms（1.12×）；
      数值逐位不变。详见 `docs/03` §15。
- [x] **O4c（向量化归约，real lever）**：**已完成（第二十三轮）**。先用 `lts__t_sectors_op_*`
      拆出 O2b+O4d 后「L2 81.5%」的真身——**全局 `red`（dQ/dK/dV 的 atomicAdd）占 L2 扇区
      92%（408.9 M/444 M）**、DRAM 仅 1.4%，且 L1 每 red 请求 8 扇区（mma 累加器布局天然
      uncoalesced）。改成把 `mma.m16n8` 累加器里**相邻两列（q=0/1、q=2/3，同行同 scale）**打包
      成一次 `atomicAdd(float2*)`（`red.global.add.v2.f32`）。**red 请求/sector 各 0.50×**、
      L2 red 408.9→204.5 M、**L2 81.5%→57.9%（墙被打掉）**；main S=512/1024H32/4096 =
      1.32×/1.19×/1.39×（MLA 1.45×），数值与 O2b+O4d **逐位相同**。新墙 = **L1/TEX 81.3%
       + short_scoreboard 3.50**。详见 `docs/03` §16。
- [x] **O7（dQ 归约去 RMW）**：**已完成（第二十五轮，部分收口）**。O4c 后残余的 204.5M red 里，
      dQ（GEMM5）的 epilogue 每个 nt tile 都对**本 CTA 同一 (r,c)**（映射与 nt 无关）做一次跨 CTA
      `atomicAdd`，一个元素被 RMW `ntiles` 次。O7 把它改成**先折算 `sds2[r]*scale`、再累进寄存器
      `dqacc[2][8][4]`，nt 循环后每元素只 flush 一次 `red_add2`**（dQ 归约指令 `O(ntiles)→O(1)`）。
      * 因 `sds2[r]` 逐 tile 变化，必须**先折算再累加**（不能累加裸 mma 输出）。
      * 只对 **HD=128**（dQ 的 N 一次铺满）启用；HD=512 需 4 份累加器（256 regs）不划算，走原路。
      * 朴素版 168→**254 regs / 2 CTA/SM**（S=512/1024 反慢）；用 `__launch_bounds__(128,3)`
        压回 **168 regs（+48B spill）/ 3 CTA/SM**，并做成模板参数 `REGDQ`，host 按「平均每 CTA
        的 nt tile 数」(`(S/BN)/2/ksplit ≥ 4`) 选择，小 S 高 ksplit 走原路避免回归。
      * **数值与 O4b 逐位相同**；L1 red 17.0M→**9.0M**、L2 red 204.5M→**108.5M（0.53×）**、
        **L2 69.1%→43.8%**、short_scoreboard 2.73→1.89；main S=4096 2.9473→**2.6736ms（1.10×）**、
        端到端 4.6162→**4.3410ms**；其余 shape 走原路（±1% 噪声）。**新墙 = L1/TEX 64.4% +
        short 1.89 + 残余 L2 43.8%**。**剩余 red（dK/dV 跨 mta/hkv 竞争）转 backlog**。详见
        `docs/03` §18、`docs/04` §2.3/§3。
- [x] **O7e（第四十五轮）** fp8 main：fold 4B 向量化写 + `REGDQ` 下关 O3 预取。
      重定位 fp8 第一墙为 **L1/TEX**（见任务清单 O9c），register spill 与 store 冲突各压掉约一半，
      main S=4096 **1.036×**、数值逐位不变。详见 `docs/03` §20、`docs/04` §2.3。
- [x] **O7e-2（第五十轮）** fp8 main fold 的 shared-load bank conflict 修复：lane→m 映射
      `sub4*16+t`→`sub4*8+t+half*32`（消 `Ps/Ss` 列读的恒 2-way 冲突）+ 8B/16B 向量写；
       shared load 冲突 **−53.5%**、L1/TEX 64.8→60.0%、main 1.02–1.04×、数值逐位不变。
       另证伪「fold 写折 16B」（1.007×、增大 spill）。详见 `docs/03` §25、`docs/04` §2.3。
- [x] **O7e-3（第五十八轮）** fp8 main **GEMM1/2 epilogue `Ps/Ss` 的 store/回读 bank conflict 修复**：
       用 `--page source --csv` 按源码行把 shared-store 多余 wavefronts 拆开，发现 **~98% 来自
       `Ps/Ss[r*PSS+c]` 写（25.6M+12.8M）及其同模式回读（12.8M）**，根因 PSS=33（≡1）使
       `(g·PSS+2l) mod 32` 大量重合（4-way）。改 **`PSS=BN+5=37`**（仍 ≡1 mod 4 ⇒ fold 掩码读不冲突；
       bank=(5g+2l) ⇒ 2-way）；smem 70.7→72.7KB 仍 3 CTA/SM，**数值逐位不变**。同 session A/B
       main 1.007–1.024×、端到端 1.005–1.015×；ncu：store 冲突 29.46M→12.33M（−58%）、
       load 冲突 29.18M→20.59M（−29%）、L1/TEX 59.97→55.83%。**但 Duration 持平** ⇒ 证伪
       「L1/TEX 是限速器」：真正墙是 `wait`(1.56)+`short_scoreboard`(1.50) 的 mma 依赖延迟 +
       3 CTA/SM（issue 仅 45.9%）。详见 `docs/03` §26。
- [x] **（第五十一轮新列，最高优先级）O17：fp16 跨 warpgroup 归约（BM=128、2 warpgroups）**。
      动机（量化证据）：ncu 拆 S=4096 主 kernel 的 L2 扇区——**`red`（dK/dV 跨 CTA `atomicAdd`）
      占 73.1%**、DRAM 仅 4.2% ⇒ main 是 **L2 原子字节数 bound**；O16（重叠 epilogue）与 O7c
      （float4 归约）都证明「动等待/事务数」无效。**一个 KV 元素被 `nblk` 个 CTA 贡献，BM=128 后
      只被 `nblk/2` 个 ⇒ red 字节直接砍半**。**已完成（第五十二轮）**：
      `fa_bwd_fp16_wgmma2_kernel<HD>`（单/两文件，`#ifdef FA_WGMMA`，仅 HD=128、256 线程）——
      2 个 wg 各算自己 64 行 Q 的 P/dS；**dK/dV 只由 wg0 对全 BM=128 归约**（两个 m64 半进同一
      `wgmma.m64n64k16` 累加器 = 两半之和），每个 KV 元素只 `red` 一次；dQ 各 wg 寄存器累加。
      前置冒烟 `fa_bwd_fp16_wgmma2_smoke.cu` 验证 128 行 MN-major 转置读**逐位 PASS**。
      ncu：`red` **102.2M→51.9M（0.508×）**、`read` 也 0.50×、L2 71.8%→**54.7%**、Duration
      1.48→**1.00ms**；main-only **S4096 1.57×（140.5 TF）/ GQA 1.47× / MQA 1.54× / S512 1.11×**，
      端到端 S=4096 **1.427ms（96.3 TF，FA3 的 4.4×，O9b 时 ~6.0×）**；数值 vs ref 历史逐位一致。
       regs 200 / smem 149.5KB / 1 CTA/SM。**bf16 版（O17-bf16）已完成（第五十三轮）**：
       `fa_bwd_bf16_wgmma2_kernel<HD>` 逐字 dtype 参数化，main S4096 1.516×（139.9 TF）、
       GQA/MQA 1.50–1.53×、S512 1.11×、ncu 与 fp16 逐项一致（`docs/01b` §6s）。TMA 通路（O15a）
       与 O17b（BM=256）留后续。
       **O17-2 已完成（第五十五轮）**：把 phase B 的 **GEMM3→wg0、GEMM4→wg1**（原版 wg0 串行
       做 dV+dK，张量工作量 3:1），red 字节不变但 **`barrier` stall 1.61→0.46（−3.5×）**、
       main S4096 **1.013×（fp16）/1.038×（bf16）**、GQA kv4 1.021×/1.051×，数值逐位不变
       （`SPLIT` 模板 + `--wg2split=0/1` A/B）。详见 `docs/01` §14l、`docs/01b` §6t。
- [x] **O18：BN=128 版 wgmma2（fp16，第五十六轮）**。O17/O17-2 后 main 仍 **1 CTA/SM（12.5%）+
       延迟受限**；§14k.7 item 3 提出把 KV-tile 从 BN=64 翻到 **128**：每 CTA 的 tile 数减半 ⇒
       `__syncthreads`/`cp.async.wait`/wgmma `commit_group`+`wait0` 序列减半；GEMM1/2 用
       `m64n128k16`（一条算两倍）。先冒烟 `fa_bwd_fp16_wgmma2b_smoke.cu` 逐位验证 m64n128 的
       累加器布局与转置描述符（dV/dK/dQ/S 四项 **max_abs=0**，含 GEMM3/4 的 m64 半基址 = 存储列
       64 的 `+1024B`）。新增 `fa_bwd_fp16_wgmma2b_kernel<128,SPLIT>`（单/两文件 device 逐字一致，
       `--wg2bn`，230 regs/0 spill/224KB smem/1 CTA/SM）。**ncu（S4096）**：Duration 982.4→
       **951.1µs**、L1/TEX 49.5→44.3%、**`red` 51,904,512 逐字节不变**（BN 不动归约结构）、
       `barrier` 0.46→0.93。**main MHA S4096 0.9895→0.9618ms（142.9 TF，1.029×）/ S512 1.028×**；
       GQA/MQA 中性（0.994–0.995×，保持 O17）；数值 vs ref 逐位一致。墙仍是 **L2（red 占 ~72%）+
       `wait` + 低 occ**。详见 `docs/01` §14m、`docs/04` §2.1/§3。
- [x] **O18-bf16：BN=128 版 wgmma2（bf16，第五十七轮）**。把 fp16 O18 逐字 dtype 参数化到 bf16
       （新增 `wgmma_m64n128k16_bf16_t` / `wgmma_mn128_issue` / `fa_bwd_bf16_wgmma2b_kernel<HD,SPLIT>`，
       host `--wg2bn`；单/两文件 device 逐字一致）。**main MHA S4096 1.025–1.029×（143.8 TF）/
       S512 1.030×**，GQA/MQA 中性；数值与 O5b~O17 历史逐位一致；ncu Duration 982.2→**952.5µs**、
       **`red` 51,904,512 逐字节不变**、regs 200→255 / smem 148.5→230.4KB / occ 12.5%（1 CTA/SM）。
        端到端 S4096 **1.381ms（99.5 TF，FA3 的 4.30×）**。详见 `docs/01b` §6u、`docs/04` §2.2/§3。
- [x] **O21（第六十一轮）fp8 主 kernel BN=32→64（负结果）+ 消冗余 convert（O21b，正结果）**。
      BN 全参数化（单/两文件 device 逐字一致）：**S512 0.900×、S1024H32 0.920×、GQA 0.920×、
      S4096 0.816×**；ncu 证实 regs 168→255、smem 72.7→105.2KB、occ 3→2 CTA/SM ⇒ 延迟受限下
      少 1/3 在飞 warp 更亏（与 O19 一起证伪 fp8 的「放大 tile/减 red」）。O21b 把 acc 别名到 fp32
      输出、消掉纯 fp32→fp32 的 `convert_kernel`：端到端 **S4096 1.037× / 单文件 1.046× / GQA 1.037×**，
       **数值逐位不变**，已设为默认。详见 `docs/03` §29。
- [x] **O22（第六十二轮）fp8 Hopper 路径默认化（正结果）+ REGDQ/ILV/ksplit/PSS 判决（负结果）**。
      `-DFA_WGMMA` 构建下 `lsewgm`/`wgmma` 默认开（`--lsewgm=0/1`、`--wgmma=0/1` 供 A/B；`sm_90`
      构建不变、device 逐字一致）：同 session 端到端 **S512 1.060× / S1024H32 1.059× / S4096 1.085× /
      GQA kv4 1.071×**；S4096 **2.48ms / 55.3 TF**（main-only 67.3 TF），为 TE FP8 的 **4.21×**。
      同轮否决：`--regdq=0`（main 2.136 vs 2.480ms，保留 REGDQ）、`FA_ILV`（中性偏负）、
      ksplit 重标定（维持 k=4）、`PSS` 扫描（37 近最优）。墙仍 = **mma 依赖延迟 + 3 CTA/SM**。
      详见 `docs/03` §30、`docs/04` §2.3。
- [x] **O7b（第六十四轮）**：dK/dV 跨 CTA 归约 → **确定性反向**。fp16 `--det=1`（partial +
      `dkv_reduce_kernel`）：`red` 51.9M→0、主 kernel 1.20×、逐位可复现，但二次归约 DRAM bound
      ⇒ S4096 `main+reduce` 0.810×；保留 opt-in，详见 `docs/01` §14o。
      **① cluster 分布式归约** 已由 **O25（第六十六轮）** 做掉、并证伪（`red` 精确减半但 main
      0.13×，远程原子延迟 + 串行 flush，见 §14q）。**剩余（backlog）**：把确定性做成**默认**
      （当归约成本可接受时）——但 O7b 已示 reduce 是 DRAM bound，故暂不做。
      **O7e 已证明 fp8 侧该 L2 墙只剩 43.7% < L1/TEX 66%**；**O19/O21 又证伪 fp8 的「减 red/放大
      tile」**⇒ fp8 下一步只剩「提 occupancy（168→≤128 regs + 72.7→≤58KB smem）或减 mma 依赖
      stall」；fp16/bf16 侧见上 O17（跨 wg 归约才是真杠杆）。
- [x] **O27（第六十八轮）**：fp8 fold 量化「逐元素精确除法」→「每行 `rcp` + 乘法」。新增模板参数
      `RCP`（默认 true）+ `--foldrcp=0` 同 binary A/B；`scX=amax/fp8max` 是每行常量，旧实现每元素
      一次 `prec-div`（~10+ 指令）。**main 1.08–1.18×**（S4096 2.043→**1.758ms**）、端到端 S4096
      2.4637→**2.1770ms（63.1 TF）**；**vs fp32 ref 逐位一致**（9 shape）；ncu executed inst
      852.6M→735.2M（−13.7%）。详见 `docs/03` §32。~~fp16/bf16 的 fold 同样含逐元素除法~~（**误记，
      见「下一步」④；fp16/bf16 无 fold**）。
- [x] **O28（第六十九轮）**：fp8 fold 转换指令向量化（`__nv_cvt_float2_to_fp8x2` / SASS
      `F2FP...PACK_AB_MERGE_C`，新 helper `foldpack4` + 开关 `FA_CVT2` 默认 1）。**数值逐位
      不变**；**d128 全中性**（指令 −4.3% 但纯 mma 依赖延迟 bound）、**MLA main ~1.03×**
      （S512H4 0.1423→0.1386、S1024H2 0.2727→0.2653ms）。详见 `docs/03` §33、`docs/04` §2.3。
- [ ] （backlog）P3-3 正式化：把「ours vs ref vs TE」对拍汇总进 `harness/`，供 P4 数值表引用。

## 灵感 / backlog

- 用 `nsys` 看 preprocess + main + convert 的端到端重叠。
- 把 FA2 的 `dQ_accum` 累加缓冲 vs 纯 atomic 做对比实验。
- deterministic 模式的代价量化。
- fp8：对比「只量化 dO」vs「dO 和 P 都量化」的精度/性能权衡。
