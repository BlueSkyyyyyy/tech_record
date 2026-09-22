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
- [ ] **O9**（对标 FA3）TMA + `wgmma` + warp specialization 多级流水。
- [ ] 目标：fp16/bf16 main ≥ 0.5× FA2 → 逐步逼近 FA2/TE。


## 目标形状 & FA/TE 归因（已分析，见 docs/06）

- [x] 纯反向口径基准 `harness/fa_vs_te_bwd_only.py`：FA2.7.4 217–377 TF vs TE2.14 305–618 TF（TE 1.2–1.6×）
- [x] kernel/SASS/SOL 归因：FA2.7.4=SM80 `HMMA/LDSM/LDGSTS`，TE=cuDNN SM90 `UTMALDG/WARPGROUP`(wgmma+TMA)；
      FA 对 GQA/MQA 多一个 `reduce` kernel。详见 `docs/06`。
- [x] FA3（SM90）编译并三方对比：**FA3 > TE > FA2**（详见 docs/07）：GQA/MQA FA3 355–438TF vs TE 307–356TF；MHA S4096 FA3 850TF vs TE 618TF。
- [ ] 后续：ours 对标升级为 Hopper 路线（O5 mma → O9 wgmma+TMA）；GQA KV 归约放进 kernel。

## 下一步（明确到可执行）

> **当前冲刺（按序，把 ours 性能推到 FA3/TE 水平；这是最高优先级，别再被其它任务打断）**：
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
>
> **旁支已完成（第三十六轮 O5c）**：把 fp16/bf16 的 **MLA（head_dim=512）反向从标量升级为张量核**
> （`HD` 模板 128/512、GEMM3/4/5 N-tile 循环、dQ 全局累加），main 5.2–5.8×（fp16）/3.7–4.1×（bf16），
> 数值同噪声、MHA 回归逐位不变。详见 `docs/01` §14b、`docs/01b` §6l、`docs/04` §7.8。
> MLA 的下一步（降 smem 冲 2 CTA/SM / split-KV）列 backlog。

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
- [ ] （backlog）O7b：dK/dV 的跨 CTA 归约（分块 `*_accum` + convert，或按 KV 列块常驻 / Q 块累加），
      消剩余 108.5M red 并得到确定性反向。
- [ ] （backlog）P3-3 正式化：把「ours vs ref vs TE」对拍汇总进 `harness/`，供 P4 数值表引用。

## 灵感 / backlog

- 用 `nsys` 看 preprocess + main + convert 的端到端重叠。
- 把 FA2 的 `dQ_accum` 累加缓冲 vs 纯 atomic 做对比实验。
- deterministic 模式的代价量化。
- fp8：对比「只量化 dO」vs「dO 和 P 都量化」的精度/性能权衡。
