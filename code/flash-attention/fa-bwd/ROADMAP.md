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

- [ ] SM90 TMA+wgmma 版本（对标 FA3）
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

## 为什么 ours 比 FA/TE 慢这么多（归因）

「按 flash-attention 实现」指的是**算法与数据流照 FA**（preprocess 求 D、1colblock、recompute P、
dQ 累加/dK/dV 归约、causal 特化），**但计算后端与性能工程没有照 FA**。实测证据（读
`src/fp16/fa_bwd_fp16_kernels.cuh`）：

| 维度 | FA2/FA3 实现 | ours（fp16/bf16 golden） | 影响 |
|---|---|---|---|
| 5 个 GEMM | **Tensor Core**：`mma.m16n8k16` / `wgmma` + `ldmatrix` | **标量 CUDA core**：`__half2float(q)*__half2float(k)` + fp32 FMA | 峰值差 ~15×（67 vs 989 TFLOPS），实际差 ~65× |
| 数据搬运 | `cp.async`/TMA + 多级软流水/双缓冲 | 同步 `__syncthreads` 后读，无预取 | 延迟受限（ncu: L1/TEX 53.5%、Compute 8.4%、occ 6.25%） |
| dK/dV 归约 | 专用累加 + convert（deterministic 可选） | 每元素 `atomicAdd` | 竞争 + 非确定性 |
| preprocess LSE/D | 分块、向量化 | 标量逐行（fp16/bf16 未修；fp8 已用 mma 修，14–62×） | S=4096 时 preprocess 曾 > main |
| grid/occupancy | 大 grid、wave 饱满 | grid 小、waves 0.48、1 CTA/SM | SM 空转 |

**根本原因**：fp16/bf16 版是**正确性优先的标量 golden**（用来先把数学/对拍跑通），
真正的性能杠杆（张量核）只在 fp8 版实现了（main 13–20×）。所以差距是**预期内**的，
不是算法错了——数值对拍与 FA/TE 同量级就是证明。

**补齐路径（按回报排序）**：

- [ ] **O5（最大杠杆）** 把 fp8 的 `mma.m16n8k16` + `ldmatrix` 后端移植到 fp16/bf16 反向
      （模板已有，改 dtype 与 smem 布局/padding）；预估 main **10–20×**。
- [ ] **O6** `cp.async` 双缓冲 + 降 smem 提 occupancy（对齐 fp8 的 O2/O3）。
- [ ] **O7** dQ/dK/dV 去 `atomicAdd`，改分块 `*_accum` + convert（确定性 + 消竞争）。
- [ ] **O8** preprocess 的 LSE/D 改 mma 分块（对齐 fp8 的 O1）。
- [ ] **O9**（对标 FA3）TMA + `wgmma` + warp specialization 多级流水。
- [ ] 目标：fp16/bf16 main ≥ 0.5× FA2 → 逐步逼近 FA2/TE。


## 目标形状 & FA/TE 归因（已分析，见 docs/06）

- [x] 纯反向口径基准 `harness/fa_vs_te_bwd_only.py`：FA2.7.4 217–377 TF vs TE2.14 305–618 TF（TE 1.2–1.6×）
- [x] kernel/SASS/SOL 归因：FA2.7.4=SM80 `HMMA/LDSM/LDGSTS`，TE=cuDNN SM90 `UTMALDG/WARPGROUP`(wgmma+TMA)；
      FA 对 GQA/MQA 多一个 `reduce` kernel。详见 `docs/06`。
- [ ] 后续：ours 张量核化（O5）与 wgmma/TMA（O9）；GQA KV 归约放进 kernel。

## 下一步（明确到可执行）

> **用户新增需求（优先）**：让 ours 支持 P5 的生产形状（GQA/MQA + MLA head_dim=512）——
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
- [ ] **MLA 优化（backlog）**：fp16（P5-2）、bf16（第二十轮）、fp8（第二十一轮）均已给出 MLA 的
       ref 对拍与 ours 性能数字；下一步是**优化**——fp16/bf16 都是标量 + 1 CTA/SM（~135KB smem），
       目标是**上张量核**（对齐 fp8 mma 路径）并把 smem 降下来冲 2 CTA/SM；fp8 MLA 已是张量核但
       1 CTA/SM（223KB smem，255 regs + spill），瓶颈在 occupancy。共同的最大障碍是 `Kt/Qt/dOt`
       三个转置副本（fp8 MLA 里 ≈107KB），正好是 **O4b（`ldmatrix.trans` 消转置副本）**；其次是
       preprocess 在 MLA（D=512）下占比升高（S=1024H2 时 0.371ms vs main 0.564ms）。
       对标 FlashMLA 的分块/流水/persistent。
> 以下为既有 fp8 优化 backlog（P5 已全部收口，现在可与 MLA 优化合并推进）。

> P5-3 已完成，ROADMAP 里的「P 项」全部收口，后续为**优化 backlog**。
> **O1/O2/O3/O4a/O2b/O4d 已完成**，下一项从 **O4c（`atomicAdd` → 分块 accum，消 L2 原子流量）**
> 起做（O2b+O4d 后 ncu 显示 bound 已变成 **L2 带宽 81.5% + long/short scoreboard**，正是 atomics 的锅）；
> 之后是 O4b（fp8 `ldmatrix.trans` 消转置副本 → 冲 4 CTA/SM，同时是 fp8 MLA 冲 2 CTA/SM 的关键）。
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
- [ ] **O4b（消转置副本）**：把 `Kt/Qt/dOt` 三个转置副本换成 fp8 `ldmatrix.trans`
      （既减 smem 冲 4 CTA/SM = 原 O2c，又减 smem 往返）。**注意**：fp8 的 `ldmatrix.trans`
      因为「2 字节=1 b16」的配对转置，布局不是 drop-in，需先写最小复现验证（对照 `42` 篇
      的 `ldmatrix.x4.trans` 用法）。
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
- [ ] **O4c（确定性与归约）**：dK/dV 的 `atomicAdd` 换 `dK/dV_accum` 分块缓冲 + convert
      （对齐 FA2 做法），顺带消 atomic 竞争、降 L2 流量（O2b+O4d 后 ncu L2 81.5%）、便于
      deterministic 口径。**（下一项）**
- [ ] （backlog）P3-3 正式化：把「ours vs ref vs TE」对拍汇总进 `harness/`，供 P4 数值表引用。

## 灵感 / backlog

- 用 `nsys` 看 preprocess + main + convert 的端到端重叠。
- 把 FA2 的 `dQ_accum` 累加缓冲 vs 纯 atomic 做对比实验。
- deterministic 模式的代价量化。
- fp8：对比「只量化 dO」vs「dO 和 P 都量化」的精度/性能权衡。
