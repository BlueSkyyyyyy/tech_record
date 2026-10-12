# DeepSeek Mega MoE 前向 / 反向算子实现解析 · 路线图（活文档）

> 目标：以 **DeepGEMM** / **DeepEP** 为参考，把 DeepSeek 的 **Mega MoE**（dispatch + linear1 + SwiGLU +
> linear2 + combine 融合的单 mega-kernel、通信与计算重叠）以及 MoE 的**反向**（wgrad/dgrad grouped GEMM）
> 逐层讲透。本专题为长期自驱任务，由 `scripts/autopilot.sh` 驱动（与仓库内其它自驱任务并行）。
>
> **重要约束**：Mega MoE 主实现是 **SM100（Blackwell）**，本机 H100（sm90）跑不了端到端。
> 因此以**源码精读 + 线上资料**为主线，数值实验只做 H100 上可行的部分（grouped GEMM / backward / 通信原语）。

## 交付物（Definition of Deliverable）

1. **分析文档**（`docs/`，内联到博客）
   - MoE / EP 基础；Mega MoE 的整体数据流与「通信-计算重叠」机制。
   - `sm100_fp8_fp4_mega_moe.cuh` 主体 kernel 的逐段精读（warp specialization、symmetric buffer、
     ring buffer / task 调度、TMA、tcgen05 MMA、epilogue）。
   - MoE **反向**：wgrad（`k_grouped_*`）与 dgrad（`m_grouped_*`）的实现解析，以及 gate/router 反向。
   - 相关 mega-kernel：Mega Gate、Mega mHC。
2. **可运行实验**（`src/`、`harness/`）
   - H100 上可跑的 grouped GEMM / backward 数值实验（对照 PyTorch 参考），报误差与性能。
   - 通讯原语（DeepEP）在 H100 上的可行实验（如环境允许）。
3. **实测证据**：能跑的部分给真实数字与原始输出；跑不了的部分明确标注「SM100 专属，仅分析」。

## 环境 / 参考坐标

- 容器 `kernel_lab`（`scripts/lab.sh`），GPU H100 sm90。Mega MoE 端到端需 SM100（本机不可跑）。
- `~/github/DeepGEMM`：Mega MoE、grouped GEMM、backward wgrad；`~/github/DeepEP`：通信。
- 线上资料：DeepSeek 官方博客 / GitHub PR（如 `#316` Mega MoE、`#95` MoE backward）。
- **目标卡代号敏感**：一律写「目标卡」。

## 任务清单

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。

### P0 侦察与基础

- [x] **P0-1** 读通 DeepGEMM：`csrc/apis/mega_moe.hpp`、`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、
      `scheduler/mega_moe.cuh`、`layout/mega_moe.cuh`、`tests/test_mega_moe.py`，产出数据流脑图。
      → `docs/02-mega-moe-architecture.md`（commit `057ca59` 行号核实）。
- [x] **P0-2** 补充线上资料：DeepGEMM README/release notes 中 Mega MoE 段落、PR #316；记录引用链接。
      → `docs/01` §5（版本时间线与链接）、`docs/02` §8（PR #316 基准表与解读）。
- [x] **P0-3** 访问 `~/github/DeepEP`：dispatch/combine kernel 与 symmetric memory 的接口与数据流。
      → `docs/07-deepep-dispatch-combine.md`（真实行号核对，DeepEP commit `93eb6eb` / v2.5.0）。
- [x] **P0-4** 环境探测：`kernel_lab` 里能否编译/跑 DeepGEMM 的 SM90 部分（`pip install -e ~/github/DeepGEMM`，
      注意 `DG_SKIP_CUDA_BUILD`）；确认 H100 上可用哪些 kernel。已知：CUDA 13.2 / PyTorch 2.12 达标。
      → `docs/08-h100-environment.md`（真机数字）；实验 `harness/probe_h100_env.py`、`harness/probe_deepep.py`。
      结论：dense/grouped FP8、BF16、**SM90 版 MoE wgrad（`k_grouped_*_nt_contiguous`）** 均可跑；
      `k_grouped_*_tn_contiguous` 与 `fp8_fp4_mega_moe` SM100 专属（`gemm.hpp:369` 拒绝）。DeepEP 可装可 import
      但运行期受阻（NCCL 2.29.7 vs 需求 ≥2.31），见「阻塞」。

### P1 MoE / EP 基础（`docs/01-moe-ep-basics.md`）

- [x] **P1-1** MoE 计算图：router → topk → grouped GEMM（linear1/SwiGLU/linear2）→ combine；
      解释 grouped contiguous vs masked 两种布局。→ `docs/01` §1（含真实行号：`gemm.hpp:181/209-246/269-318`、
      `runtime.hpp:57-64`、README `76-86`），并补 **PSUM（前缀和）第三布局**；H100 实测三种布局数值等价
      （`harness/exp_grouped_layout.py` + `.out.txt`）。
- [x] **P1-2** EP（专家并行）通信：all-to-all dispatch / combine 的语义与开销；为何要与计算重叠。
      → `docs/01` §2 重写：通信量模型（dispatch ~$T{\cdot}d{\cdot}H{\cdot}1$B、combine ~$T{\cdot}k{\cdot}H{\cdot}2$B，
      K 倍放大）+ 串行 vs 重叠量化 + **H100 实测**（`harness/exp_ep_comm.py`，8×H100 NCCL
      all-to-all：渐近 ~393 GB/s、固定时延 ~50–90 us，达 NVLink 硬件上限 ~82%）。

### P2 Mega MoE 架构（`docs/02-mega-moe-architecture.md`）

- [x] **P2-1** 整体设计：单 kernel 融合 5 段；symmetric buffer、`transform_weights_for_mega_moe` 布局。
      → `docs/02` §2-§3（数据流脑图 + `MegaMoEBuffer` 布局）+ §3.4（权重布局：gate/up 8 粒度交错、
      SF 的 UTCCP 4x32 转置、SM100 packed UE8M0 + MN-major 索引），含纯 PyTorch 重排核对
      `harness/exp_weight_transform.py`（0 误差）。
- [x] **P2-2** 「通信-计算重叠」机制：ring buffer / pool blocks / task 调度（`scheduler/mega_moe.cuh`）、
      dispatch warp 与 MMA/epilogue warp 的分工。→ `docs/02` §4-§5。
- [x] **P2-3** 与「先通信后计算」「chunked 分步」方案的对比（为什么更快）。→ `docs/02` §7。

### P3 主体 kernel 精读（`docs/03-mega-moe-kernel.md`）

- [x] **P3-1** 模板参数与 block config（`get_block_config_for_mega_moe`）：BLOCK_M/N/K、cluster、stage、
      thread 划分（dispatch / non-epilogue / epilogue）。
      → `docs/03-mega-moe-kernel.md`（真实行号核对）；H100 对拍 `harness/exp_block_config.py` / `.out.txt`。
- [x] **P3-2** warp specialization 与同步（mbarrier）；TMA 载入激活/权重/scale。
      → `docs/09-mega-moe-mbarrier.md`（真实行号核对）：五类 warp 两层解耦、全 mbarrier 清单与 init 计数、
      `advance_pipeline` 相位翻转 / 生产者 phase=1 约定、A/B loader 坐标与 2-CTA multicast、task 槽双缓冲
      与「仅 epilogue release」、跨 SM/rank barrier、L1→L2 位图依赖；H100 真机验证相位约定 +
      NSTAGES=1/4 流水重叠（`src/mbarrier_pipeline.cu`、`harness/exp_mbarrier_pipeline.{sh,out.txt}`）。
- [x] **P3-3** linear1 → SwiGLU → linear2 的 MMA 与 epilogue（tcgen05、UE8M0 scale、FP8xFP4）。
      → `docs/10-mega-moe-mma-epilogue.md`（真实行号核对）：leader CTA 单 warp 发 UMMA、A/B 互换
      与 2-CTA（`UMMA_M=256/UMMA_N=BLOCK_M`）、动态缩 N（`valid_m` 16 对齐）、routed/shared 双
      instruction descriptor、UTCCP 把 SF 从 smem 搬进 TMEM（`kernel:894-907`）、UMMA
      `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`（`tcgen05.cuh:67-86`）与 accumulate 首条清零、FP4 打包因子
      （`packing.cuh:8-12`）、commit multicast 通知 empty/tmem_full；epilogue TMEM 双缓冲 + 提前
      `release_task_info`/`release_tmem`、L1 in-place SwiGLU（bf16 round）+ UE8M0 量化（
      `math.cuh:114-147`）+ TMA 转置上屏 + L2 K-block bitmask 依赖、L2 BF16 NVLink 直写远端
      combine buffer、combine top-k fp32 归约。H100 数值实验 `harness/exp_ue8m0_sf.py`（+`.out.txt`）：
      位运算 SF 25009 样本 0 不符、scale×inv 精确 1、幂次 scale/理想 ≤1.9977 且 0 溢出、bf16-round
      SwiGLU 相对误差 4.5e-3。配套博客 `content/posts/mega-moe-07-mma-epilogue`。
- [x] **P3-4** dispatch / combine 段与 NVLink 读写；locality domain 的作用。
      → `docs/11-mega-moe-dispatch-combine.md`（真实行号核对）：dispatch（`:342-689`）分布式前缀和
      计数 + 写远端源索引 + NVLink barrier + 无锁 round-robin 认领（`:480-528`）+ 分 chunk TMA
      远端 pull（`:552-576`）与远端 weight/SF（`:579-599`）+ 清 workspace；combine L2 直写远端
      （`:1346-1351`）+ grid-index readiness 水位线（`:1405-1441`）+ fp32 chunk 归约（`:1447-1519`）；
      `SymBuffer::map` / TMA / sys 原子 / rel-acq / `grid_sync` / `nvlink_barrier` 原语表；
      locality domain 分域调度（`scheduler:344-370`）与片上 pointer-chase 探针标定
      （`csrc/apis/locality_domain.hpp`）。H100 实测 `src/nvlink_access.cu` +
      `harness/exp_nvlink_access.{sh,out.txt}`：远端裸延迟 ~991ns ≈ 本地 ~350ns 的 **2.83×**、
      远端读/写带宽 ~368 GB/s ≈ 本地 ~2.85 TB/s 的 **1/8**，同 socket/跨 socket 无差异（NV18 平面）。

### P4 反向实现解析（`docs/04-moe-backward.md`）

- [x] **P4-1** MoE 反向数学：linear2/linear1 的 dgrad 与 wgrad，SwiGLU 反向，router/topk 反向。
      → `docs/04-moe-backward.md`：逐层反向推导 (1)–(8) + 「wgrad=k_grouped / dgrad=m_grouped」形状角色表
      （`gemm.hpp:320-371`/`:55-76`/`:340-349`、`generators.py:583-587`、README `:80`）；H100 float64 对拍
      `harness/exp_moe_backward_math.py`（+`.out.txt`）：手写反向 vs autograd 四项梯度 relerr ≤2.9e-16、
      中心差分 ≤5.7e-8，k_grouped 逐组 `A_e^T@B_e` 复现 wgrad relerr ≤1.3e-16。配套博客
      `content/posts/mega-moe-09-backward`。
- [x] **P4-2** DeepGEMM 的 wgrad API `k_grouped_fp8_gemm_tn_contiguous`（K 轴分组、M/N 固定）精读。
      → `docs/12-moe-backward-wgrad.md`（真实行号核对）：`tn`（SM100/MN-major `[sum_k,M]`，仅分析）
      与 `nt`（SM90/K-major 每组 `[M,k_g]`）同一数学 `D[e]=A_e^T B_e`；`ks_cpu`/`grouped_layout`
      （contiguous=物理 K / PSUM=逻辑 K 结束，PSUM 仅 SM100）、`check_k_grouped_args`（`gemm.hpp:55-76`）、
      SM90 按组原地改写 tensormap（`sm90_fp8_gemm_1d1d.cuh:193-215` + `tensor_map_buffer`）、
      SF 逐组打包 UE8M0 4→int32（`smxx_layout.hpp:218-283`）、调度器 K 分组切块
      （`scheduler/gemm.cuh:71-82/226-246`）。H100 实测 `harness/exp_kgrouped_api.py`（+`.out.txt`）：
      nt 4 组 relerr 1.98e-2（FP8 噪声）、空组 OK、未对齐 ks 被拒、tn 在 sm90 被拒、
      `m=4096,n=7168`（8 组）**425 TFLOPS**；纯 torch 布局等价 tn 0 误差 / nt 3.3e-8。
- [x] **P4-3** dgrad（`m_grouped_*`）与 contiguous/masked 布局在训练中的用法。
      → `docs/13-moe-backward-dgrad.md`（真实行号核对）：dgrad 天然是 **M 轴分组**（不等长=token_数 落在 M，
      N/K 固定，`gemm.hpp:194/280`）；两个 host 入口 `m_grouped_fp8_fp4_gemm_nt_contiguous`（`:181-250`，
      contiguous=长度 m 逐行组号 / PSUM=长度 G 前缀和结束，`:213-220`）与 `..._nt_masked`（`:269-318`，
      `[G,max_m,·]` 稠密 + device 端 `masked_m`）；**PSUM 仅作用于 SFA**（`layout.hpp:69`、`gemm.hpp:230`）。
      调度器三 `GemmType`（`scheduler/gemm.cuh:92-105/188-225`），`get_global_idx`/`is_computation_valid`
      统一取址与跳过 gap（`:151-156/296-309`）；**段必须对齐 BLOCK_M 的 kernel 侧原因**：contiguous 用
      `grouped_layout[m_block_idx*BLOCK_M]`（M block 首行组号）代表整块（`:152`），SM90 TMA multicast
      peer 一致性也依赖它（`:287-289`）。SM90 1d2d kernel：A/D/SFA 在 masked 下是 3D（组号=第 3 维）、
      contiguous 下 2D（`sm90_fp8_gemm_1d2d.hpp:178-194/245-261`），`kWithGroupOffsetA=masked`
      （kernel `:189-194`）；SFB 逐组基址（`:240`）。**训练分工**：前向/dgrad 用 contiguous、解码 CUDA
      graph 用 masked、DeepEP 动态输出用 PSUM 桥接。H100 实测 `harness/exp_mgrouped_dgrad.py`（+`.out.txt`）：
      contiguous/masked/PSUM relerr 3.67–3.70e-2（FP8 噪声）、A4 布局长度不符被拒（`gemm.hpp:218`）、
      `G=8,m=4352,n=2048,k=7168` **366 TFLOPS**；三种布局两两 **逐位相等（0 误差）**。配套博客
      `content/posts/mega-moe-11-dgrad`。
- [ ] **P4-4** 在 H100 上做 grouped GEMM / backward 的数值对拍实验（vs PyTorch）。

### P5 相关 mega-kernel（`docs/05-mega-gate-mhc.md`）

- [ ] **P5-1** Mega Gate（`mega_gate.hpp` / `sm100_bf16_mega_gate.cuh`）：做什么融合、怎么实现。
- [ ] **P5-2** Mega mHC（`mega_mhc.hpp` / `sm100_mega_mhc.cuh`）：HyperConnection 相关。

### P6 汇总

- [ ] **P6-1** `docs/06-summary.md`：数据流图、关键机制、与朴素实现的开销对比、反向要点一页纸。

## 每轮增量流程（autopilot）

1. 读本文件「任务清单」选**一个未完成子项**（优先 P0→P1→…）。
2. 写/改 `docs/<NN>-*.md`（会被内联到博客，**里面不要写 {{< >}} 短代码**）与必要的 `src/`、`harness/`。
3. 能跑的用 `scripts/lab.sh exec "..."` 跑出真实数字；**跑不了的（SM100 专属）明确标注「仅分析」**。
4. 精读源码贴**真实核对过的行号**；网络资料给可点击链接（不要编造 URL）。
5. `/tmp/hugo_bin/hugo --gc --minify` 确认无 ERROR；`git add` 只加本专题路径；commit 前缀 `megamoe:`；
   `git pull --rebase` 后 push（网络失败只本地 commit 并在本文件记「待推送」）。
6. 更新本文件状态与「当前进度/下一步」。
7. **不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。**

## 已知坑

- 目标卡代号敏感：写「目标卡」。
- Mega MoE 是 SM100 专属；H100 上**不要**试图跑 `fp8_fp4_mega_moe`，会编译/运行失败。
- 被内联的 `docs/*.md` 里**不要写 `{{< >}}` 短代码**；站内链接只在 `content/posts` 里用 `relref`。
- 行号引用必须真实核对过。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。
- DeepGEMM editable 安装后，JIT 需要 `deep_gemm/include/{cute,cutlass}`；editable 不会自动拷，
  要手动 `cp -r third-party/cutlass/include/{cute,cutlass} deep_gemm/include/`（详见 `docs/08` §2）。
  另需 `apt-get install -y libdw-dev`，并确保 `third-party/deep_jit` 子模块与 gitlink 一致（`2efdab4`）。

## 当前进度 / 下一步

- 当前进度：**P0/P1/P2/P3 全部完成；P4 进行中（P4-1、P4-2、P4-3 完成）。**
  P4-3：新建 `docs/13-moe-backward-dgrad.md`——dgrad 的 M 轴分组 API 精读。**全景**：dgrad
  （`dA=dY·W2^T`、`dX=dG·W1^T`）不等长在 **M（token）轴**、N/K 固定，故复用前向的 `m_grouped` 族；
  A（激活）必须 K-major、D 固定 BF16（`gemm.hpp:197/209`）。**两入口**：
  `m_grouped_fp8_fp4_gemm_nt_contiguous`（`:181-250`，contiguous=长度 m 逐行组号 / PSUM=长度 G
  前缀和结束，判别在 `:213-220`）与 `..._nt_masked`（`:269-318`，`[G,max_m,·]` 稠密 + device 端
  `masked_m`）；**PSUM 只作用于 SFA**（`layout.hpp:69`、`gemm.hpp:230`），因为 gap 只出现在激活侧。
  **调度器**三 `GemmType`（`scheduler/gemm.cuh:92-105`）：masked/PSUM 的 `num_m_blocks` 随组变
  （`:188-225`），contiguous 恒定；`get_global_idx`（`:151-156`）与 `is_computation_valid`
  （`:296-309`）统一取址/跳 gap。**为什么段要对齐 `BLOCK_M`**：contiguous 用 M block **首行**组号
  `grouped_layout[m_block_idx*BLOCK_M]` 代表整块（`:152`），不对齐就跨组；SM90 multicast peer
  一致性也依赖它（`:287-289`）。**SM90 kernel**：masked 下 A/D/SFA 是 3D（组号=第 3 维）、contiguous
  下 2D（`sm90_fp8_gemm_1d2d.hpp:178-194/245-261`），`kWithGroupOffsetA=masked`（kernel `:189-194`）；
  SFB 逐组基址（`:240`）。**训练分工**：前向/dgrad→contiguous、解码 CUDA graph→masked、
  DeepEP 动态输出→PSUM 桥接；shared expert 走稠密路径；dgrad 需按 token 跨专家 scatter-add，
  wgrad 留专家本地无需 all-reduce。H100 实测 `harness/exp_mgrouped_dgrad.py`（+`.out.txt`）：
  contiguous/masked/PSUM relerr 3.67–3.70e-2、A4 布局长度不符被拒（`gemm.hpp:218`）、
  `m=4352,n=2048,k=7168`（G=8）**366 TFLOPS**；三种布局**两两逐位相等（0 误差）**。配套博客
  `content/posts/mega-moe-11-dgrad`。
  P4-2：新建 `docs/12-moe-backward-wgrad.md`——wgrad 的 K 轴分组 API 精读。**两条路径一个数学**：
  `k_grouped_fp8_gemm_tn_contiguous`（SM100，A/B 为 `[sum_k,M]`/`[sum_k,N]` **MN-major**，
  靠组坐标偏移取数，`sm100_fp8_fp4_gemm_1d1d.hpp:325-403`，**仅分析**）与
  `k_grouped_fp8_gemm_nt_contiguous`（SM90，每组 `[M,k_g]` **K-major** 拼接，`gemm.hpp:373-427`）。
  **SM90 为什么按组原地改 tensormap**：base tensormap 只能按首个非空组的 K 造
  （`sm90_fp8_gemm_1d1d.hpp:159-195`），kernel 里切组时 patch 全局地址 `+k_start*M`
  与 inner-dim stride，并用 `tma_desc_commit/wait_group` + release/acquire 防竞态
  （`sm90_fp8_gemm_1d1d.cuh:193-215`，对应 PR #343）。**参数契约**：`ks_cpu` 每组 K 必须
  `k_alignment` 对齐（`check_k_grouped_args`，`gemm.hpp:55-76`）；`grouped_layout` 在
  contiguous=物理 K、PSUM=逻辑 K 结束（PSUM 仅 SM100，`scheduler/gemm.cuh:71-82`）；
  SM90 `nt` 强制 accumulate（`gemm.hpp:403`）。**SF**：SM100 逐组把 FP32 打包成 MN-major
  UE8M0 int32（每 4 个 gran_k 打 1 个，`smxx_layout.hpp:218-283`）。调度器按组切 K 块 +
  独立计数 `current_sf_k_cumsum`（`scheduler/gemm.cuh:226-246`）。H100 实测
  `harness/exp_kgrouped_api.py`（+`.out.txt`）：nt 4 组 relerr 1.98e-2、空组 D 全零、
  未对齐 ks 被拒（`gemm.hpp:69`）、tn 在 sm90 被拒（`gemm.hpp:369`）、`m=4096,n=7168`（8 组）
  4528us **425 TFLOPS**；纯 torch 布局等价：tn 0 误差、nt 3.3e-8。配套博客
  `content/posts/mega-moe-10-wgrad`。
  P4-1：新建 `docs/04-moe-backward.md`——MoE 反向数学。**全景**：`dz`→combine 反向
  （`dY=w⊙dz`、`q=<Y,dz>`）→linear2（`dW2=A^T dY` wgrad / `dA=dY W2^T` dgrad）→SwiGLU
  （`dG_G=dA⊙G_U⊙silu'`、`dG_U=dA⊙silu`）→linear1（`dW1=x^T dG` / `dX=dG W1^T`）→router。
  **router/topk 反向**：归一化 topk 权重把 softmax 概率除以 `P=Σ_{i∈S}p_i`，得
  `dp_j=(q_j P-A)/P²`（`A=Σ q_i p_i`）再走 softmax Jacobian；**坑**：`P` 必须是原始 softmax 概率
  在选中集上的和（不是归一化权重之和，后者恒为 1），第一版因此让 `dx`/`dWg` 偏到 1e-2。**映射**：
  wgrad 的不等长轴是收缩维 K（token）→ `k_grouped`；dgrad 的不等长轴是 M（token）→ `m_grouped`，
  给出了 (M,N,K) 角色表与 `gemm.hpp`/`generators.py` 真实行号。H100 实验
  `harness/exp_moe_backward_math.py`（+`.out.txt`，float64）：四项梯度 vs autograd relerr ≤2.9e-16、
  中心差分 ≤5.7e-8，按 `k_alignment=128` 逐组 `A_e^T@B_e` 复现 `dW1/dW2` relerr ≤1.3e-16。
  配套博客 `content/posts/mega-moe-09-backward`。**反向无 mega-kernel**：融合价值主要在前向重叠；
  反向由独立 grouped GEMM 组成，wgrad 天然留守专家本地 rank。
  （P0/P1/P2/P3 进展见下）
  P3-4：新建 `docs/11-mega-moe-dispatch-combine.md`——通信两段逐段精读 + H100 访存实测。
  **dispatch**：分布式前缀和一次原子加（高位计数、低位负载，`kernel:374-381`）→ `grid_sync` 后
  SM0 push `peer_grid_idx`（`:399-404`）→ `nvlink_barrier`（`:422-428`）→ 无锁 round-robin
  min-peeling 认领 token（`:480-528`）→ 环空等待（`:544-550`）→ 分 chunk `tma_load_1d` 远端 pull
  token（`:552-576`）+ 远端 topk weight（`:579-581`）+ 远端 SF 逐块搬并 UTCCP 转置写位（`:584-599`）
  → 写元数据/`red_add_rel` 通知（`:602-618`）→ 清 workspace 并与 combine 重叠（`:621-689`）。
  **combine**：L2 epilogue 读回 `cached_src_metadata`（`:1281-1290`）后经 `sym_buffer.map` 直写
  远端 combine buffer（`:1346-1351`）；readiness 用「每 launch 唯一 grid index」当水位线
  （peer push `:403` ↔ `st_rel_sys`:1419 ↔ `ld_acq_sys` 自旋 `:1436-1441`）；归约双缓冲 TMA 拉
  top-k、fp32 累加（`:1477-1490`）、BF16 写 `y`（`:1512-1518`）。**原语表**：`SymBuffer::map`
  （`layout/sym_buffer.cuh:33-40`）、`grid_sync`/`nvlink_barrier`（`comm/barrier.cuh:33-92`，加减式
  相位信号免复位）。**locality domain**：`kNumDeviceLocalityDomains=2`（`types.cuh:54`），
  调度器按域分任务计数 + 少量任务不偷域（`scheduler:344-370`、`kNumMinStealWaves=2`），域标签由
  片上 pointer-chase 延迟探针标定（`csrc/apis/locality_domain.hpp:74-142`）。H100 实测
  `src/nvlink_access.cu` + `harness/exp_nvlink_access.{sh,out.txt}`：远端裸延迟 ~991ns ≈ 本地
  ~350ns 的 **2.83×**，远端读/写带宽 ~368 GB/s ≈ 本地 ~2.85 TB/s 的 **1/8**，同/跨 socket 相同。
  P3-3：新建 `docs/10-mega-moe-mma-epilogue.md`——计算本体与 epilogue 逐段精读。**MMA**：只有
  leader CTA 的一个 warp 发 UMMA（`kernel:819-938`）；`A/B 互换 + 2-CTA` 使 `UMMA_M=256`、
  `UMMA_N=BLOCK_M`（token 维），按 `valid_m` 动态缩 N（`update_instr_desc_with_umma_n`，
  `mma/sm100.cuh:150-153`）；routed(FP8/FP4) 与 shared(FP8) 两个 block-scaled descriptor
  （`:825-850`）；`UTCCP_4x32dp128bit_2cta` 把 SFA/SFB 从 smem 搬进 TMEM（`:894-907`）；
  `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`（`ptx/tcgen05.cuh:67-86`）内层按 `gran_k=32` 拆 4 条，
  首条非累加清零、其余累加（`:920`）；FP4 打包因子 2（`packing.cuh:8-12`）；commit multicast
  同时通知 `empty_barriers` 与（末 K block 时）`tmem_full_barriers`。**Epilogue**：TMEM 双缓冲，
  读完最后有效块即 `release_tmem`，`release_task_info` 提前（`:990`）；L1 分支 in-place SwiGLU
  （bf16 round，gran-8 交错权重让 gate/up 成对，`:1108-1135`）+ amax warp-pair 归约 + UE8M0
  量化（`get_ue8m0_sf_exp/inv`，`math.cuh:114-147`）+ STSM + TMA 转置上屏（就是 L2 激活）+
  `l2_full_mask` bitmask 依赖；L2 分支 BF16 经 `sym_buffer.map` NVLink 直写远端 combine buffer；
  combine 段 TMA 拉 top-k 贡献、fp32 累加、bf16 写 `y`。H100 数值实验 `harness/exp_ue8m0_sf.py`
  （+`.out.txt`）：位运算 `get_ue8m0_sf_exp` 在 25009 样本（含 0/亚正规/幂次边界）上 0 不符、
  `scale×inv` 精确为 1、幂次 scale/理想 ≤1.9977 且 0 溢出、bf16-round SwiGLU 相对误差 4.5e-3。
  配套博客 `content/posts/mega-moe-07-mma-epilogue`。
  P3-2：新建 `docs/09-mega-moe-mbarrier.md`——**两层解耦**（TMA↔MMA 的 full/empty，MMA↔epilogue 的
  tmem_full/empty）+ 调度器 task 槽（task_info full/empty）的完整 mbarrier 清单与 init 计数
  （`full.init(2*2)` / `empty.init(1)` / `tmem_empty.init(2*kNumEpilogueThreads)`，`kernel:209-285`）、
  `advance_pipeline` 相位翻转（`:307-314`）与「生产者 phase=1（buffer 初空）」约定、
  A/B loader 的坐标与 2-CTA multicast + FP4 half-bytes（`:690-818`）、MMA 的 UTCCP/UMMA/commit
  （`:819-938`）、epilogue 的 `tmem_full.wait`→`release_task_info`（`:986-990`）与
  「仅 epilogue release」的执行顺序论证、跨 SM `grid_sync` / 跨 rank `nvlink_barrier`
  （`comm/barrier.cuh:33-92`）、L1→L2 位图依赖（`scheduler:140-166`）。H100 真机实验
  `src/mbarrier_pipeline.cu`（`harness/exp_mbarrier_pipeline.sh` + `.out.txt`）：**实测坐实**相位约定
  （全新 mbarrier `try_wait(parity=1)` 立即成功、`(0)` 阻塞），并量到多级流水重叠——96 MiB
  `cp.async.bulk` 在 NSTAGES 1→4 下串行 5.58→3.44 ms（带校验 1.62×），去校验 2.29→0.79 ms
  （2.88×，41→118 GiB/s，延迟受限）。配套博客文章 `content/posts/mega-moe-06-mbarrier`。
  P3-1：新建 `docs/03-mega-moe-kernel.md`——一次 launch 的物理坐标（`grid=num_sms`、
  `cluster=2`、512 线程/16 warp，`impls/sm100_fp8_fp4_mega_moe.hpp:149-198`）、BLOCK_N=128/BLOCK_K=128(fp8,bf16=64)、
  BLOCK_M 启发式（`heuristics/mega_moe.hpp:88-127`，成本模型「M block 重载权重 ~128 行」+ 路由 `+sqrt` 余量）、
  store block、SF 的 UTCCP 128 对齐、`get_pipeline_config_for_mega_moe` smem 预算→`num_stages`
  （`:129-195`，`smem_capacity=232448`）、五类 warp 角色（dispatch/TMA-A/TMA-B/MMA/调度/epilogue+combine，
  `kernel:341-948`）、寄存器两档再平衡（每档恰好 64512）。H100 真机对拍：`harness/exp_block_config.py`
  复刻启发式并与 `deep_gemm.get_block_m_for_mega_moe` 逐点比较，9 组形状 block_m **全部一致**；
  配套博客文章 `content/posts/mega-moe-05-kernel`。
  （P0/P1/P2 进展见下）
  P1-2：重写 `docs/01` §2——EP all-to-all 语义 + 「K 倍放大」通信量模型（dispatch $T{\cdot}d{\cdot}H$、
  combine $T{\cdot}k{\cdot}H{\cdot}2$）+ 串行/重叠量化对比；新增 §2.3 **H100 实测**：
  `harness/exp_ep_comm.py` / `.out.txt` 在 8×H100 上测 NCCL `all_to_all_single`（H=7168、EP=8），
  渐近有效带宽 ~393 GB/s（达 `nvidia-smi nvlink` 硬件上限 ~478 GB/s 的 ~82%）、固定时延 ~50–90 us，
  论证小 batch 延迟敏感与重叠必要性。
  P1-1：细化 `docs/01` §1——MoE 计算图 + 三种 grouped 布局（contiguous / masked / PSUM）的真实 API 与行号，
  解释对齐（SM90=128 / SM100=256）与 gap 行的「为什么」；新增 §1.5 H100 实测（`harness/exp_grouped_layout.py`，
  三布局 relerr 一致 ≈3.7e-2）。
  （P0 进展见下）
  P0-1：读通 DeepGEMM Mega MoE 五个源文件，产出 `docs/02-mega-moe-architecture.md`（真实行号核对），P2-2/P2-3 一并覆盖。
  P0-2：`docs/01` §5 记录仓库/版本时间线/关键结论/DeepEP 侧真实链接；`docs/02` §8 补 PR #316 EP8 基准与解读。
  P0-3：精读 DeepEP v2.5 对称显存/dispatch/combine，产出 `docs/07-deepep-dispatch-combine.md`。
  P0-4：在 `kernel_lab`（H100）实测 DeepGEMM 2.8.1 可用 kernel 矩阵——dense/grouped FP8、BF16、
  **SM90 版 MoE wgrad（`k_grouped_*_nt_contiguous`）** 均 OK；`..._tn_contiguous`（wgrad TN）与
  `fp8_fp4_mega_moe` 在 sm90 被 `gemm.hpp:369` 拒绝（SM100 专属）。DeepEP 2.5 可构建/import，
  但 `EPBuffer` 因 NCCL 版本（PyTorch 链 2.29.7 vs 需求 ≥2.31/2.32.3）运行期段错误。
  产出 `docs/08-h100-environment.md` + `harness/probe_h100_env.py` / `harness/probe_deepep.py`（含 *.out.txt）。
  P2-1：补 `docs/02` §3.4——`transform_weights_for_mega_moe` 的三件事：L1 gate/up 按 8 粒度交错
  （`mega/__init__.py:112`，配合 L1 epilogue 的 `SM100_TMEM_LOAD_16dp256b1x` 做 in-place SwiGLU，
  kernel `:1066`-`:1134`）、SF 的 UTCCP 4x32 转置（`mega/__init__.py:129` ↔ kernel
  `transform_sf_token_idx` `:145`-`:149`）、SM100 packed UE8M0 + MN-major SF 索引
  （`smxx_layout.hpp:133`、kernel `:774`/`:788`）；L2 权重不交错只转 SF。实验
  `harness/exp_weight_transform.py`（+`.out.txt`，H100）：行号映射/SwiGLU 等价/UTCCP 映射全部 0 误差，
  与 deep_gemm 内部实现逐元素一致。**P2 全部完成。**
- **下一步：继续 P4 反向**：P4-4（H100 上把 SM90 `k_grouped_fp8_gemm_nt_contiguous`（wgrad）与
  `m_grouped_fp8_gemm_nt_contiguous`（dgrad）接到真实 MoE 的完整反向，与 PyTorch autograd 做
  **kernel 级**数值对拍，出误差与吞吐；P4-2/P4-3 已完成 API 语义与布局等价前置核对）。
  之后推进 P5-1（Mega Gate）/P5-2（Mega mHC）。

## 阻塞

- **DeepEP 运行期实验受阻（环境，非内核）**：本机 PyTorch 链接系统 **NCCL 2.29.7**，而 DeepEP 2.5 需要
  **NCCL ≥ 2.31 的设备通信 API**（`ginSupport/ginMinStride/ginType`，`csrc/kernels/comm/context.cpp:71-142`）。
  装 `nvidia-nccl-cu13 2.32.3` + `LD_PRELOAD` 可过 import 检查，但 `EPBuffer` 构造仍在
  `ep.py:304` 段错误；用系统 2.29.7 头重建则编译失败。**需一个链接 NCCL ≥ 2.32.3 的 PyTorch 环境**。
  因此在修复前，DeepEP 的 H100 dispatch/combine 实测暂不做（源码精读见 `docs/07`）。
