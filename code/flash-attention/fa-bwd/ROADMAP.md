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

- [ ] **P1-1** 单文件 `fa_bwd_fp16_onefile.cu`：preprocess(D) + main(1colblock) + launcher + 自测；编译运行
- [ ] **P1-2** 与 ref/FA/TE 数值对拍（读 dump 的输入，比 ref 输出），记录 max diff
- [ ] **P1-3** ncu 剖析：bound 在哪；与 FA2/TE 性能对比
- [ ] **P1-4** 两文件版拆分为 `_kernels.cuh` + `_main.cu`，行为一致
- [ ] **P1-5** `docs/01-fp16-bwd-impl.md`：实现与优化逐条说明

### P2 bf16 反向

- [ ] **P2-1..5** 同 P1（复用 fp16 骨架，dtype 参数化）

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

## 下一步（明确到可执行）

- [ ] **P1-1**：实现 `src/fp16/fa_bwd_fp16_onefile.cu`——先做**功能正确**的 FA2 风格单文件反向：
  `preprocess` 求 D=rowsum(dO∘O)；主 kernel 每 Q 块固定、遍历 K 块，recompute S/P，算 dV/dP/dS/dQ/dK；
  dQ 用全局 fp32 累加缓冲（atomicAdd），dK/dV 用 atomicAdd；支持 causal。用 `scripts/run.sh` 编译，
  用 dump 的输入跑，和 `ref_dq/dk/dv` 对拍。
- [ ] 之后按 P1-2 … 逐步推进；fp8 是重点。

## 灵感 / backlog

- 用 `nsys` 看 preprocess + main + convert 的端到端重叠。
- 把 FA2 的 `dQ_accum` 累加缓冲 vs 纯 atomic 做对比实验。
- deterministic 模式的代价量化。
- fp8：对比「只量化 dO」vs「dO 和 P 都量化」的精度/性能权衡。
