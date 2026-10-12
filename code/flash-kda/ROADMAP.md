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

- [x] **P1-1** 逐 token 的 gated delta rule 定义：状态 `S∈R^{D×D}`、log-decay `g`、写入门 `β`、
      输出 `o_t = q_t^T S_t`；解释 KDA 相对 GLA/DeltaNet 的区别（per-channel decay）。
- [x] **P1-2** chunk（CHUNK=16）分块形式推导：块内 `L`（下三角）、`Mqk`、`INV`（块内注意力矩阵求逆）、
      `k_decayed/q_decayed/k_restored/g_total` 的来历与作用。
- [x] **P1-3** 前向两段式：为什么拆 K1（token 并行）+ K2（head 并行）而非单 kernel（引用 deep-dive）。
- [x] **P1-4** 数学自检：分块公式与逐 token 递推在 CPU 上逐元素一致（`src/kda_ref.py` 两种实现互拍）。

### P2 前向 K1 精读（`docs/02-kda-fwd-k1.md`）

- [x] **P2-1** 阶段拆解：TMA 载入 → QK L2 norm → gate 激活+cumsum → decay_apply → L/Mqk →
      8×8 fp32 前代求逆 + 16×16 bf16-HMMA 合并 → 写 workspace；贴行号。
- [x] **P2-2** 关键优化逐条：smem union 复用、`__launch_bounds__(256,8)`、base-2 `ex2.approx`、
      `tanh.approx` sigmoid、前代求逆的精度取舍。
- [x] **P2-3** workspace 布局逐字段解释（对上 `flash_kda.cpp:get_workspace_size`）。

### P3 前向 K2 精读（`docs/03-kda-fwd-k2.md`）

- [x] **P3-1** warp specialization（LOAD / MMA / STORE）与 TMA pipeline。
- [x] **P3-2** 八个 Phase（dual GEMM k@s/q@s → cast → u=(v-u)β; u=INV@u → Mqk@U+out → store →
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
- workspace：分配（`get_workspace_size`）恒用**上界** `ceil(T/16)+N`，而非 varlen launch 用**精确**
  `N·ceil(T_seq/16)`（更小）；`ws_tile_prefix` 偏移按 launch 的 total_tiles 算，勿按分配上界猜。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。

## 当前进度 / 下一步

- 当前进度：P0 全部完成，P1 全部完成，P2 全部完成，P3-1 完成（K2 执行结构：warp specialization + TMA pipeline），
  P3-2 完成（K2 六个 Phase 数据流 + `MOVM_T` 寄存器转置，含 GPU 实测）。
  - P0-1～P0-3：见下（历史记录）。
  - P1-1/P1-2：`docs/01` 写清逐 token gated delta rule、KDA vs GLA/DeltaNet 对照表（DeltaNet 有 β 无 α、
    GLA 有 α 无 β、KDA 两者兼有且 α 逐通道），以及 CHUNK=16 分块公式（`L/Mqk/INV/k_decayed/...`）。
  - P1-3：`docs/01` §4 给出 K1/K2 拆分动机（引用 deep-dive §2）。
  - P1-4：新增 `src/kda_ref.py`（纯 PyTorch，`kda_token_recurrent` golden + `kda_chunked_fwd` 分块，均支持
    初态/非整除 T）。自测（`harness/kda_ref_selftest.out.txt`）：fp64 下两种实现逐元素**完全相等**（max_abs=0），
    fp32 下 max_abs~1e-9~1e-7（机器精度）。新增 `harness/check_kda_ref.py` 与 FLA `chunk_kda` 对拍
    （`harness/check_kda_ref.out.txt`）：bf16 下 o/S 的 max_rel≈3e-3~6e-3。踩坑：FLA
    `transpose_state_layout=True` 会把 `initial_state` 也当 V-first 重解释，曾导致仅 t=0 误差 0.1；默认布局
    `[N,H,K,V]` 与本 ref 的 S 同布局。
  - P2-1：新增 `docs/02-kda-fwd-k1.md`：逐阶段精读 K1（tile 定位/TMA 载入/L2 norm/gate+cumsum/decay_apply/
    L,Mqk/tril+beta/块内求逆/写 workspace），全部贴真实行号；讲清 union 复用「先入寄存器再覆写」的时序、
    varlen 二分、beta 8 对齐、列并行前缀和免同步、Schur 补求逆代数。新增 `harness/check_inv_identity.py`
    （纯 PyTorch 复现 8×8 前代+Schur 补），实测 fp64 max_abs≤2.2e-16、bf16 量化≈2.5e-3~5.9e-3
    （`harness/check_inv_identity.out.txt`）。新增博客 post `flash-kda-02-fwd-k1` 并更新 index 目录。
  - P2-2：`docs/02` 新增 §11「关键优化逐条量化」，全部本机实测：
    (a) smem union：`harness/smem_probe.cu` 编译期 `cosize`/`sizeof` 实测 Phase A=16384B、
    Phase B=14336B、union 省 14336B(≈14KB)，含另两处小 union 共省 18688B；`sizeof(SharedStorageK1)=21248B`。
    (b) `__launch_bounds__(256,8)`：对已编译 `.so` 跑 `cuobjdump -res-usage`（`harness/k1_res_usage.out.txt`）
    实测 K1 两变体均 `REG:32 STACK:0 LOCAL:0`（正好压 32 寄存器上限、零 spill）；8×256=2048=max 线程、
    8×21248=166KB≤228KB，即 8 CTA/SM 占满。
    (c) `ex2.approx`/`tanh.approx`：`harness/approx_probe.cu`（`approx_probe.out.txt`）实测 ex2 max rel
    1.4e-7、tanh max abs 7.6e-6、sigmoid max abs 3.9e-6；门控衰减端到端 max rel≤1.9e-5（lb=-5），
    比 bf16 q/k 舍入(~4e-3)小两个量级。
    (d) 前代求逆精度：`harness/check_inv_precision.py`（`check_inv_precision.out.txt`）对比 A纯fp32/B kernel
    (seed fp32+bf16 合并)/C bf16 seed/D Neumann(fp16/bf16)：完全共线 c=0.9 时 Neumann fp16=0.58、bf16=2.83
    而 kernel 路径仅 1.6e-3，量化确认「放弃 Neumann 级数」的稳定性收益。
  - P2-3：`docs/02` 新增 §12「workspace 逐字段字节账与 SoA 指针切分」，全部本机实测与源码核对：
    (a) 每 tile 六字段字节（`utils.cuh:70–76`）= 3×4096 + 512 + 2×512 = **13824 B**，与 host 硬编码
    `flash_kda.cpp:20` 一致，且各字段均 128B 对齐（`flash_kda.cpp:16–18` static_assert）；解释为何
    workspace 13824B 比 K1 smem Phase B 14336B 少 512B（$L$ 的 fp32 512B 不必落盘，只存求逆后的 bf16 INV）。
    (b) SoA 六基址前缀和偏移（`fwd_launch.cu:64–71`）、`ws_idx=head*total_tiles+tile`（`fwd_kernel1.cuh:516`）
    head-major 让 K2 单 CTA 顺读；讲清「为何 SoA 而非 AoS」（各字段 TMA 布局/对齐/descriptor 解耦）。
    (c) 总分配恒取上界 `tiles=ceil(T/16)+N`（`flash_kda.cpp:14`），非 varlen launch 用精确 `N·ceil(T_seq/16)`
    （`flash_kda.cpp:180`）并给不等式证明 ≤ 上界；点明 `ws_tile_prefix` 偏移必须按 **launch** 的 total_tiles
    而非分配上界（坑）。新增 `harness/check_workspace_layout.py`（`check_workspace_layout.out.txt`）三查：
    字段预算/对齐、9 组 `get_workspace_size` 对闭式（含 varlen N=2/4/7）全 OK、varlen 实跑
    `lens=[30,16,5,50]` 在预测偏移 304128 读出 `tile_prefix=[0,2,3,4,8]` 与期望一致、其后 108B canary 未被动。
  - P3-1：新增 `docs/03-kda-fwd-k2.md` 前向 K2 的**执行结构**（warp specialization + TMA pipeline），
    全部贴真实行号：
    (a) warp 分工（`fwd_kernel2.cuh:169–197`）：4 MMA warp（tid 0–127）+ 1 LOAD（128–159）+ 1 STORE（160–191）
    = 192 线程（`fwd_launch.cu:186`）；讲清为何 K1「高占用短 kernel」与 K2「专职分工深流水」取向相反。
    (b) 流水线构造（`utils.cuh:86–143`）：输入 `PipelineTmaAsync<3>`（LOAD=Producer、MMA=Consumer，
    `num_producers=1`/`num_consumers=128`；`transaction_bytes=17984`）、输出 `PipelineAsync<2>`
    （MMA=Producer 128 / STORE=Consumer 1）；`pipeline_init_wait(1)`。
    (c) producer/consumer 循环与 barrier：LOAD producer_acquire/producer_tail（`347,421`）、MMA
    store_acquire(436)/consumer_wait(437)/commit(737)/consumer_release(738)、`compute_barrier`
    `NamedBarrier(128,0)`（`427,733`）作为 4 MMA warp 的 rendezvous（因 `s_acc` 跨 warp 共享）。
    (d) 状态搬运：独立 `state_acc_tma_barrier`（`111,240–317,786–835`）与 union（`99–107`，fp32 状态缓冲
    65536B 比流水线 62144B 大，故 union 取 65536B）；尾块手动 store（`757–766`）。
    (e) 实测（`harness/smem_k2_probe.cu`、`k2_res_usage.out.txt`）：每输入 stage 17984B、`sizeof
    (SharedStorageK2<128,16;3,2>)=98432B`；K2 全部 14 个实例 `REG:74–80 STACK:0 LOCAL:0`（无 spill）；
    占用率瓶颈是 smem → 2 CTA/SM（196864B≤233472B）。新增博客 post `flash-kda-03-fwd-k2` 并更新 index。
  - P3-2：`docs/03` 新增 **Part II「MMA warp 六个 Phase 数据流」**（§8–§17），全部贴真实行号：
    (a) §8 记号与 MMA 骨架：讲清 K2 里 `s_acc = Sᵀ`（$[V,K]$，对上 `torch_ref.py:227–239` 的 `.t()`）、
    TN 约定下 `C=(M,N)`/`B=(N,K)` 的 lane→下标映射为何相反（`mma_traits_sm80.hpp:53`），
    以及为何「C 片段当 B 必须做 8×8 转置」→ `movmatrix`。
    (b) §9–§14 逐 Phase 数据流：Phase 1 dual GEMM（手动软件流水 `544–568`）、Phase 2 cast+载 v/INV/β
    （β 按 token 逐行，`group_id=lane/4`）、Phase 3 $\tilde v\to P\tilde v$（`MOVM_T` 把 C 片段转 B，
    转置被 gemm 代数「吃掉」`614–620`）、Phase 4 $M_{qk}U+$out（复用 U）、Phase 5 store、
    Phase 6 状态更新（`s_acc_T`/`k_restored_t` 转置视图 + LDSM_T/STSM_T，`g_total` 逐 key 缩放 `684–719`）。
    (c) §15 数据搬运预算表 + **GPU 实测** `harness/movm_probe.cu`（`harness/movm_probe.out.txt`）：
    用与 K2 相同片段验证 **`MOVM_T` 路径 $U=P\tilde v$ vs host fp32 参考 max_abs=1.5e-8、max_rel=8.7e-8**；
    naive `STSM+LDSM_N`（不转置）max_rel=1.26（全错），**证明转置是数学必需**；延迟微基准
    4×`MOVM_T`=18.0 cyc vs `STSM+syncwarp+LDSM`=21.8 cyc（1.21×），真正收益是省 4096B U 缓冲与
    Phase 3→4→6 对 U 复用的零同步。
- **下一步：P3-3** 讲状态更新 `s_acc = s_acc*g_total + k_restored^T @ U` 的分块含义，
  验证 bf16 状态存储 + fp32 FMA 更新的精度（对照 deep-dive §3）；再进 P4 数值对拍（golden 已有
  `src/kda_ref.py`），反向（P5）可用已装好的 FLA `chunk_kda` autograd 对拍。

## 阻塞

- （空）
