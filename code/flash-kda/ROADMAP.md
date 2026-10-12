# Kimi Delta Attention（KDA）前向 / 反向算子实现解析 · 路线图（活文档）

> 目标：以 **FlashKDA**（`~/github/FlashKDA`）为参考，把 **Kimi Delta Attention（KDA）** 的
> **前向**与**反向**算子实现逐层讲透：数学（gated delta rule 的分块形式）→ CUDA kernel 结构
> （K1 prepare / K2 recurrence）→ 数值对拍 → 性能与 `ncu` → 反向推导与实现。
> 本专题为长期自驱任务，由 `scripts/autopilot.sh` 驱动（与仓库内其它自驱任务并行）。

## 交付物（Definition of Deliverable）

1. **分析文档**（`docs/`，内联到博客）
   - KDA 数学与分块（chunk）公式、前向递推、反向（adjoint / 反向递推）的完整推导。
   - FlashKDA 前向 **K1**（`fwd_kernel1.cuh`）与 **K2**（`fwd_kernel2.cuh`）的逐阶段精读（贴真实行号）。
   - 反向实现解析（FLA `chunk_kda` autograd 或自研），逐式说明。
   - 数值对拍报告 + 性能/ncu 报告 + 小结。
2. **可运行参考实现**（`src/`）
   - `src/kda_ref.py`：纯 PyTorch/NumPy 的 KDA 前向（逐 token delta rule）与分块版，可自测。
   - `src/kda_bwd_ref.py`：反向的逐步推导实现，与 `torch.autograd` / FLA 对拍。
3. **对拍/基准脚本**（`harness/`）：探测 FlashKDA / FLA 可用性、dump 输入输出、报告 max abs/rel、CUPTI 计时。
4. **实测证据**：所有性能/误差数字来自本机实测，原始输出留 `*.out.txt`。

## 环境 / 参考坐标

- 容器 `kernel_lab`（`scripts/lab.sh`），GPU H100 sm90。
- FlashKDA：`~/github/FlashKDA`（forward，`CHUNK=16`，`D=128`，SM90+）。
- FLA（反向参照）：容器内 `pip install flash-linear-attention` 后 `from fla.ops.kda import chunk_kda`。
- 注意：**目标卡代号敏感**，一律写「目标卡」。

## 任务清单

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。

### P0 侦察与环境（先做）

- [x] **P0-1** 读通 FlashKDA：`csrc/flash_kda.cpp`（API/形状）、`fwd.h`（launch 模板）、
      `smxx/fwd_kernel1.cuh`、`smxx/fwd_kernel2.cuh`、`smxx/utils.cuh`、`fwd_launch.cu`、deep-dive，
      产出结构脑图（`docs/01` §5，含真实行号）。
- [x] **P0-2** 环境打通：`scripts/lab.sh` 起容器；确认 import torch/FlashKDA；`pip install flash-linear-attention`
      并确认 `chunk_kda` 可跑（拿反向参照）。
- [x] **P0-3** `harness/probe_refs.py`：跑通 FlashKDA forward 与 FLA `chunk_kda`，确认两者量级一致的相同输入。

### P1 数学与分块公式（`docs/01-kda-theory.md`）

- [ ] **P1-1** 逐 token 的 gated delta rule 定义：状态 `S∈R^{D×D}`、log-decay `g`、写入门 `β`、
      输出 `o_t = q_t^T S_t`；解释 KDA 相对 GLA/DeltaNet 的区别（per-channel decay）。
- [ ] **P1-2** chunk（CHUNK=16）分块形式推导：块内 `L`（下三角）、`Mqk`、`INV`（块内注意力矩阵求逆）、
      `k_decayed/q_decayed/k_restored/g_total` 的来历与作用。
- [ ] **P1-3** 前向两段式：为什么拆 K1（token 并行）+ K2（head 并行）而非单 kernel（引用 deep-dive）。
- [ ] **P1-4** 数学自检：分块公式与逐 token 递推在 CPU 上逐元素一致（`src/kda_ref.py` 两种实现互拍）。

### P2 前向 K1 精读（`docs/02-kda-fwd-k1.md`）

- [ ] **P2-1** 阶段拆解：TMA 载入 → QK L2 norm → gate 激活+cumsum → decay_apply → L/Mqk →
      8×8 fp32 前代求逆 + 16×16 bf16-HMMA 合并 → 写 workspace；贴行号。
- [ ] **P2-2** 关键优化逐条：smem union 复用、`__launch_bounds__(256,8)`、base-2 `ex2.approx`、
      `tanh.approx` sigmoid、前代求逆的精度取舍。
- [ ] **P2-3** workspace 布局逐字段解释（对上 `flash_kda.cpp:get_workspace_size`）。

### P3 前向 K2 精读（`docs/03-kda-fwd-k2.md`）

- [ ] **P3-1** warp specialization（LOAD / MMA / STORE）与 TMA pipeline。
- [ ] **P3-2** 八个 Phase（dual GEMM k@s/q@s → cast → u=(v-u)β; u=INV@u → Mqk@U+out → store →
      s_acc update）逐条讲清数据流；`MOVM_T` 寄存器内转置消 smem 往返。
- [ ] **P3-3** 状态更新 `s_acc = s_acc*g_total + k_restored^T @ U` 的分块含义与 bf16 状态精度验证。

### P4 数值对拍（`docs/04-numerics.md`）

- [ ] **P4-1** 参考实现：`src/kda_ref.py` 逐 token 递推（fp32）作为 golden。
- [ ] **P4-2** FlashKDA forward vs golden：多 shape（不同 T/H）、多 seed，报 max abs/rel。
- [ ] **P4-3** 与 FLA `chunk_kda` 三方对比；给误差表。

### P5 反向的推导与实现（`docs/05-kda-bwd.md`）

- [ ] **P5-1** 反向数学：从 `o_t=q_t^T S_t` 与状态递推逐 token 求 adjoint（`dq,dk,dv,dβ,dg, dstate`）。
- [ ] **P5-2** `src/kda_bwd_ref.py`：逐 token 反向参考实现，与 `torch.autograd`（对 `kda_ref.py`）对拍一致。
- [ ] **P5-3** 与 FLA `chunk_kda` 反向（`torch.autograd` 穿 `chunk_kda`）对拍；报 max abs/rel。
- [ ] **P5-4** 反向分块化（chunked backward）推导：如何复用前向的 `L/INV/Mqk`、状态反向传递。

### P6 反向 kernel 解析/实现（`docs/06-kda-bwd-kernel.md`）

- [ ] **P6-1** 若 FLA 提供 CUDA 反向，逐 kernel 精读；否则基于 P5-4 给出自研 kernel 的设计。
- [ ] **P6-2** （可选）在 `kernel_lab` 上写最小可运行反向 kernel 并与参考对拍。

### P7 性能与剖析（`docs/07-perf.md`）

- [ ] **P7-1** FlashKDA forward CUPTI 计时（不同 T/H），与 FLA 对比。
- [ ] **P7-2** `ncu` 剖析 K1/K2 bound（smem/TensorCore/occupancy），逐项证据。

### P8 小结 / 一页纸

- [ ] **P8-1** `docs/08-summary.md`：公式、结构、误差、性能、反向要点一页纸汇总。

## 每轮增量流程（autopilot）

1. 读本文件「任务清单」选**一个未完成子项**（优先 P0→P1→… 顺序，但可并行做数值/精读）。
2. 写/改 `docs/<NN>-*.md`（会被内联到博客）与必要的 `src/`、`harness/` 代码。
3. 用 `scripts/lab.sh exec "..."` 在 `kernel_lab` 里跑，结果留 `*.out.txt`。
4. 数值不吻合要贴出输入/误差并排查，不可含糊带过。
5. `/tmp/hugo_bin/hugo --gc --minify` 确认无 ERROR；`git add` 只加本专题路径；commit 前缀 `kda:`；
   `git pull --rebase` 后 push（网络失败只本地 commit 并在本文件记「待推送」）。
6. 更新本文件状态与「当前进度/下一步」。
7. **不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。**

## 已知坑

- 目标卡代号敏感：写「目标卡」。
- 被内联的 `docs/*.md` 里**不要写 `{{< >}}` 短代码**；站内链接只在 `content/posts` 里用 `relref`。
- 行号引用必须真实核对过；宁可不标也不标错。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。

## 当前进度 / 下一步

- 当前进度：P0-1～P0-3 完成。
  - P0-1：通读 FlashKDA host/launch/K1/K2/utils 五个源文件与 deep-dive，
    在 `docs/01-kda-theory.md` §5 写出结构脑图（行号已核对）；`docs/01` 同时含数学定义与 CHUNK=16 分块公式初稿。
  - P0-2：`kernel_lab` 容器（H100 sm90，torch 2.12）确认可用；FlashKDA 的 `cutlass` 子模块
    浅克隆后以 `FLASH_KDA_CUDA_ARCHS=90a pip install -e . --no-build-isolation` 编译成功，
    `import flash_kda` OK；`pip install flash-linear-attention==0.5.2` 后 `fla.ops.kda.chunk_kda` 可用。
  - P0-3：`harness/probe_refs.py`（结果 `harness/probe_refs.out.txt`）在同一输入上对拍三路：
    FlashKDA `fwd`、FLA `chunk_kda`、FLA `fused_recurrent_kda`(fp64 gold)。两个 case
    （`g=N(0,4)`、`g=0`，B1 T512 H4 D128，非零初态）下：`chunk_kda`/`flash_kda` 输出对 gold 的
    max_abs≈6e-4～8e-4、max_rel≈4e-3～9e-3，两者互差 max_abs≈8e-4～1e-3，量级（|o|max≈0.15）一致，
    确认三方实现同一套 KDA 数学。
- **下一步：P1-1/P1-2** 正式写 `docs/01` 的逐 token 定义与 chunk 公式（补全 KDA vs GLA/DeltaNet 的
  区分点），随后 **P1-4** 落地 `src/kda_ref.py`（逐 token 递推 vs 分块两种实现 CPU 互拍）。
  反向（P5）可用已装好的 FLA `chunk_kda` autograd 作对拍参照。

## 阻塞

- （空）
