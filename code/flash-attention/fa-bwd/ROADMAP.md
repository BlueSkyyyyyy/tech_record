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

- [ ] **P3-1** `docs/02-fp8-bwd-design.md`：dO/dP/dQKV 的量化与 scaling 布局（对齐 TE 口径）
- [ ] **P3-2** 单文件 fp8 反向（mma e5m2/e4m3，rowwise scale，fp32 累加）；编译运行
- [ ] **P3-3** 对拍：vs fp32 ref（容差）与 vs TE fp8；给出误差统计
- [ ] **P3-4** ncu + 性能 vs TE fp8；找 bound 并继续优化
- [ ] **P3-5** 两文件版 + `docs/03-fp8-bwd-impl.md`

### P4 文档 / 汇总

- [ ] **P4-1** `docs/04-numerics-and-perf-summary.md`：数值表 + 性能表（我们/FA/TE，各 shape）
- [ ] **P4-2** `docs/05-porting-notes.md`：目标卡抽象层与移植注意事项

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

## 下一步（明确到可执行）

- [ ] **P3-1**：`docs/02-fp8-bwd-design.md`：对齐 TE 口径写清 dO/dP/dQKV 的量化与 scaling 布局
      （dO E5M2 + rowwise scale，P/S 走 E4M3，fp32 累加，`scale_a*scale_b` 折回）。
- [ ] 之后按 P3-2..P3-5 推进 fp8 单文件 → 对拍 → ncu/性能 → 两文件 + 文档（**fp8 为最重点**）。
- [ ] （backlog，性能）**preprocess 已成 S=4096 端到端瓶颈**（69ms > main 42ms）：对 LSE 点积分块 +
      向量化（`__ldg`/float4），或把 LSE 并入前向摊薄；main 侧继续降 smem 提 occupancy、
      上张量核（mma）+ 流水；fp16 也可同步加 K/V padding。

## 灵感 / backlog

- 用 `nsys` 看 preprocess + main + convert 的端到端重叠。
- 把 FA2 的 `dQ_accum` 累加缓冲 vs 纯 atomic 做对比实验。
- deterministic 模式的代价量化。
- fp8：对比「只量化 dO」vs「dO 和 P 都量化」的精度/性能权衡。
