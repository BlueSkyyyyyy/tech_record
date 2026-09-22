# flash-attention 反向（fa-bwd）开发 · 路线图（活文档）

> 目标：借鉴 flash-attention / TransformerEngine 的实现，在目标 AI 卡（当前在 H100 sm90 上验证）上，
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

### 可选

- [ ] SM90 TMA+wgmma 版本（对标 FA3）
- [ ] causal / 非 causal / GQA / 变长（cu_seqlens）覆盖

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

（当前无）

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

## 下一步（明确到可执行）

> P4-2 完成后，ROADMAP 里的「P 项」已全部收口，后续为**优化 backlog**（按回报排序）。
> **O1/O2 已完成**，下一项从 **O3（fp8 main 加流水）** 起做，或先做小 S 的 grid / 尾波。
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
- [ ] **O3（fp8 main 加流水）**：K/V（及 Q/dO）用 `cp.async` 双缓冲，把「同步载入→算」改成
      重叠流水；O2 后仍是 `No Eligible 79.6%`（`long_scoreboard`+`barrier` 主导），验证
      `long_scoreboard` 下降。可顺带做 **O2b**：小 S 的 grid 太小（S=512 grid=128<132 SM）
      与 3 CTA 下的尾波（waves 2.59，partial 233/396）——考虑小 S 用 BM=32 或 N 方向切块。
- [ ] **O2c（fp8 main → 4 CTA/SM）**：需再砍 ~17 KB smem，等价于消除 `Kt/Qt/dOt` 三个转置副本
      （fp8 的 `ldmatrix.trans` 因为「2 字节=1 b16」的配对转置，布局不是 drop-in，需最小复现验证）。
- [ ] **O4（确定性与归约）**：dK/dV 的 `atomicAdd` 换 `dK/dV_accum` 分块缓冲 + convert
      （对齐 FA2 做法），顺带消 atomic 竞争、便于 deterministic 口径。
- [ ] （backlog）P3-3 正式化：把「ours vs ref vs TE」对拍汇总进 `harness/`，供 P4 数值表引用。

## 灵感 / backlog

- 用 `nsys` 看 preprocess + main + convert 的端到端重叠。
- 把 FA2 的 `dQ_accum` 累加缓冲 vs 纯 atomic 做对比实验。
- deterministic 模式的代价量化。
- fp8：对比「只量化 dO」vs「dO 和 P 都量化」的精度/性能权衡。
