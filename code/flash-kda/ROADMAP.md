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
- [x] **P3-3** 状态更新 `s_acc = s_acc*g_total + k_restored^T @ U` 的分块含义与 bf16 状态精度验证。

### P4 数值对拍（`docs/04-numerics.md`）

- [x] **P4-1** 参考实现：`src/kda_ref.py` 逐 token 递推（fp32/fp64）作为 golden（P1-4 交付）。
- [x] **P4-2** FlashKDA forward vs golden：多 shape（不同 T/H）、多 seed、多门控区间，报 max abs/rel。
- [x] **P4-3** 与 FLA `chunk_kda` 三方对比；给误差表。

### P5 反向的推导与实现（`docs/05-kda-bwd.md`）

- [x] **P5-1** 反向数学：从 `o_t=q_t^T S_t` 与状态递推逐 token 求 adjoint（`dq,dk,dv,dβ,dg, dstate`）。
- [x] **P5-2** `src/kda_bwd_ref.py`：逐 token 反向参考实现，与 `torch.autograd`（对 `kda_ref.py`）对拍一致。
- [x] **P5-3** 与 FLA `chunk_kda` 反向（`torch.autograd` 穿 `chunk_kda`）对拍；报 max abs/rel。
- [x] **P5-4** 反向分块化（chunked backward）推导：如何复用前向的 `L/INV/Mqk`、状态反向传递。

### P6 反向 kernel 解析/实现（`docs/06-kda-bwd-kernel.md`）

- [x] **P6-1** FLA `chunk_kda` 提供 **Triton** 反向（非 CUDA）：逐 kernel 精读其 5 个 kernel 分解，
      并与 P5-4 的 chunked backward 逐式对照；同时基于 FlashKDA 前向 workspace 给出目标卡 CUDA 反向设计。
- [x] **P6-2** （可选）在 `kernel_lab` 上写最小可运行反向 kernel（复刻状态伴随 `dhu`/B1）并与参考对拍。
- [x] **P6-3** 关闭 P6-2 的「非整除 T」边界：定位 host 侧 `E_tot` 记账假设整除的 bug（尾块 58 行
      未被寻址），改为按最后有效 token 取 $E^{tot}$；非整除 T 实测 Triton↔FLA 仍 1e-8~4e-5；
      澄清「K 的 64 分段」是 FLA 同 program 内展开、非正确性约束。

### P7 性能与剖析（`docs/07-perf.md`）

- [x] **P7-1** FlashKDA forward CUPTI 计时（不同 T/H），与 FLA 对比。
- [x] **P7-2** `ncu` 剖析 K1/K2 bound（smem/TensorCore/occupancy），逐项证据。
- [x] **P7-3** K1/K2 访存字节拆分（DRAM 读/写逐项归因 workspace vs `v` vs `o`/state），
      解决 P7-2 待验证 (b)。
- [x] **P7-4** 归因 K1「global load 4/32 bytes/sector」，解决 P7-2 待验证 (a)（静态 SASS 计数 +
      `SourceCounters` 逐行 + DRAM 占比）。
- [x] **P7-5** 关闭 P7-2/P7-4 待验证 (c)：**K2 突破 2 CTA/SM 后 L2 墙是否后移**——实验=否。
      改造 `SharedStorageK2`（fp32 状态缓冲改按 `StateFP32` 条件预留）+ 扫 pipeline stage，编译 6 个变体，
      实测占用率/ncu SOL/K2 时间。结论：smem 被无条件预留的 65536 B fp32 缓冲钉死为 98432 B（只减
      stage 不动 smem）；真进 3 CTA/SM 的配置必须砍 pipeline 深度，反而慢（`cond(2,1)` +14%、
      `cond(1,2)` +40%）；K2 在 grid 填满时 `lts` 已 93–94% SOL，是 L2 带宽 bound，占用率无可用余量。

### P8 小结 / 一页纸

- [x] **P8-1** `docs/08-summary.md`：公式、结构、误差、性能、反向要点一页纸汇总
      （含踩坑速查、交付物索引、待验证项；博客 post `flash-kda-08-summary` 并更新 index）。

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

- 当前进度：**P0–P8 全部完成 + P6-3/P7-3/P7-4/P7-5 补测**。P0/P1/P2/P3 完成，P4 前向三方对拍完成；P5 反向推导/显式与分块参考/三方核验完成；P6 FLA Triton 反向 5 kernel 精读 + B1 状态伴随最小 kernel（P6-2）+ 非整除 T 边界补测（P6-3）完成；P7 前向性能 + ncu bound 剖析 + K1/K2 访存字节拆分（P7-3）+ K1 标量 global load 归因（P7-4）+ K2 占用率实验（P7-5）完成；**P8-1 完成**（`docs/08-summary.md`：一页纸汇总，见下）。
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
  - P6-1：新增 `docs/06-kda-bwd-kernel.md` + `harness/check_bwd_kernel_map.py`
    （`harness/check_bwd_kernel_map.out.txt`）+ 博客 post `flash-kda-06-bwd-kernel` 并更新 index。
    **Part A（记号映射实测，fp32）**：跑 FLA 前向子函数，取它给反向保存的 `qg,kg,Aqk,Akk,w,v_new,h`
    与 `src/kda_ref.py` 分块符号对拍，钉死映射——`qg=q_dec/scale`(7e-8)、`kg=k_rest`(5e-7)、
    `tril(Aqk)=M_qk`(8.5e-4)、**`Akk=P=(I+L)^{-1}`**(1.5e-4，而非 $P\,diag(β)$，后者 rel≈1)、
    `w=P·diag(β)·k_dec`(7e-4)、`v_new=U`(7e-4)、`h=S_in`(块0 逐位相等、末块 1.8e-3)。
    **Part B（kernel 分解实测）**：`torch.profiler` 抓真跑 $B1,T512,H4,D128$ bf16 的
    `chunk_kda` fwd+bwd，CUDA kernel 级确认反向=3 专用 kernel（`dAv`/`dhu_blockdim64`/`intra`）
    + `recompute_w_u_fwd_kda_kernel`(×2) + `chunk_local_cumsum`(×2)，耗时与前向同阶。
    文档逐 kernel 精读并贴真实行号：编排 `chunk_bwd.py:434-615`（重算 463-506 / dAv 510 / dhu 541 /
    wy_dqkg 557 / intra 576 / 反向 cumsum 599 / gate 604），`dAv` 43-114（Aqk 存未 mask 全阵、用时
    mask 89-90）、`dhu` 359-671（逆序 476，式 7.13 在 585-587）、`wy_dqkg` 133-310（253-256、264/281、
    302-304 的 $P$ 伴随）、`intra` 395-667、前向 `inter_solve_fused` 43-379。**设计**：目标卡 CUDA
    反向复用 K1 workspace 六字段（`utils.cuh:70-76`：`kd/qd/kr/gt/INV(P)/Mqk`），只缺 `beta`/门 logits
    → 补一个 B0 门控重算，再落 B3 intra / B2 dqkg / B1 state-adjoint（B1 直接镜像前向 K2，逆序扫）。
    精度待验证项：$P$ 的伴随（式 7.5）与 bf16 存储 $\bar S$。
    **踩坑**：验证 $P$ 时用 fp32 `torch.linalg.solve(I+L,I)` 稠密解在未归一化随机 k 上会把单位下三角
    逆的第一行都解错（残差 $10^5$），须改用前代（forward substitution，即 kernel 的 8×8 前代思路）；
    另 FLA 的 `g_cumsum` 是**块内局部** log2 累计、`Aqk` 存的是未 mask 全阵。

- **下一步：专题 P0–P8 全部完成 + P6-3/P7-3/P7-4/P7-5 补测**，进入收尾/维护。后续可选（有明确需求时再做）：
  更完整的 CUTLASS/TMA 版反向 B1/B2/B3 工程化；P7-2 遗留的 nsys 流量拆分（P7-3 已用 ncu DRAM 计数替代、
  待验证 (b) 已关闭）。**P7-5 已用实验关闭 P7-2/P7-4 待验证 (c)**（K2 突破 2 CTA/SM 后 L2 墙不后移，
  见下）。
  - P7-5：`docs/07` 新增 §8「K2 突破 2 CTA/SM 后 L2 墙会不会后移？」+ `harness/_build_k2_occ.py`
    （把 `fwd_kernel2.cuh` 的 union 第二臂改 `ReserveFP32 = StateFP32` 并宏参 stage，编译 6 个变体：
    orig/plain(3,2)/plain(2,1)/cond(3,2)/cond(2,2)/cond(2,1)/cond(1,2)）+ `harness/bench_k2_occ.py`
    （`bench_k2_occ.out.txt`）+ `harness/probe_k2_occ_ncu.sh` & `harness/ncu_driver_occ.py`
    （`probe_k2_occ_ncu.out.txt`）+ `harness/_smem_probe.py`（`smem_k2_occ_probe.out.txt`）。**实测**：
    (a) **smem 被无条件预留的 65536 B fp32 状态缓冲钉死为 98432 B**——`plain(2,1)`（只减 stage）仍 98432、
    2 CTA/SM；去掉 fp32 缓冲的 `cond(3,2)` 只剩 3200 B（95232）；要摸 3 CTA/SM 必须同时砍 OUT/IN 到 1，
    且 ncu 用「dynamic+driver 1.02KB/block」算，`cond(2,2)`=78,208 B×3=234,624 > 233,472 仍落回 2。
    (b) **2→3 CTA/SM 没有加速**：`B8 T2048 H64`（grid 512）orig K2=773µs、`cond(2,2)`=775µs；
    唯一真进 3/SM 的 `cond(2,1)` 反 **慢 14%（885µs）且 L2 SOL 从 94.0% 掉到 81.4%**；`cond(1,2)`（IN=1）
    K2=1086µs。设计点 `B1 T8192 H64` grid=64<132 SM，2/3 CTA/SM 帽根本不起作用，`cond(2,2)` 793µs≈orig 801µs。
    (c) orig K2 已 `lts__throughput=93–94%`、`dram=55%`、`sm=34%`、`warps_active=18.6%`——**L2 带宽 bound**，
    ncu「Est. Local Speedup 81.25%」是纯启发式，被实验证伪（占用率↑反而↓）。所有变体输出与上游**逐位相同**
    （`max|dO|=0`）。**结论**：L2 墙不后移；K2 无占用率维度余量，要更快只能减 L2 流量（workspace/`v` 复用）
    或改并行结构。同步更新 `docs/08` 待验证项与性能小结。博客 post `flash-kda-07-perf` 标题更新。
  - P7-4：`docs/07` 新增 §7「K1 那条 4/32 bytes/sector 的 global load 是谁」+ `harness/probe_k1_gload.sh`
    （`harness/probe_k1_gload.out.txt`），关闭 P7-2 待验证 (a)。**静态**（`cuobjdump -sass` 按函数切段）：
    K1 `IsVarlen=false` 只有 **1 条标量 `LDG.E`**（+ 5 条 TMA `UTMALDG` = q/k/g/beta/dt_bias），
    `IsVarlen=true` 5 条（多出的 4 条 = 二分查找 `tile_prefix`/`cu_seqlens`，`fwd_kernel1.cuh:179/183/185/186`）。
    非 varlen 那条的 SASS 前驱 `S2R R2,SR_CTAID.Y` + `LDC A_log_ptr` + `IMAD.WIDE R2*4`
    ⇒ **`fwd_kernel1.cuh:258` 的 `A_log_ptr[head_idx]`**。**动态**（`ncu --section SourceCounters`）：
    整段 SASS 唯一 `Address Space=Global` 的 load 即它，`Instructions=2048`、`L1 Tag Requests=2048`、
    `Sectors Global=2048 == Ideal 2048`、`Excessive=0`——`B=1,T=1024,H=4` grid=256 CTA×8 warp=2048 个 warp，
    **每 warp 恰好 1 条**（`smsp__inst_executed_op_global_ld=2048` 佐证）。**为什么是 4/32**：地址在
    warp 内一致（`head_idx=blockIdx.y`）⇒ 硬件按**广播**处理，1 sector/warp 已是该访问的最小代价，
    `ncu` 的「每线程利用字节」启发式对广播标量 load 误报。**字节预算**：设计点该 load = 262144 warp
    × 32B = 8.0MB vs K1 DRAM 读 405MB = **1.98%**（小 shape 1.97%），且 K1 只占前向 ~8% ⇒ 端到端 <0.2%。
    **勘误**：§5.2 记的「估 50% 局部加速空间」未在本次 `MemoryWorkloadAnalysis_Tables` 输出中复现
    （紧随的 `Est. Local Speedup: 44.94%` 属 shared-load bank conflict），且与 `Excessive=0` 矛盾。
    结论：§5.4「K1 = barrier stall + L2」不变，无需优化。
  - P7-3：`docs/07` 新增 §6「K1/K2 的访存账」+ `harness/check_traffic.py`
    （`harness/check_traffic.out.txt`）：用 `ncu`（默认每次 replay 前清 L2，故 `dram__bytes_*`
    = 唯一去重访存字节）对 K1/K2 的读/写逐项归因，与闭式预测对拍。四组 shape
    （1×1024×4 / 4×1024×64 / 1×8192×64 / 8×2048×8）实测：**DRAM 读全部落在预测的
    1.000–1.010**——K1 读 = `q,k,g,β`（**不读 v**，100%）、写 = workspace；**K2 读 = workspace
    74.9–76.7% + `v` 22.2–22.7% + β 0.2% + h0 0.4–2.8%**，写 = `o`（94–98.5%）+ 终态。
    修正 P7-2 §5.3「K2 只读 workspace」的说法：`v` 占 K2 读流量约 22.5%（每 chunk
    `tma_load_v/beta`，`fwd_kernel2.cuh:325/326,358/367`）。方法学坑：**TMA 密集 kernel 不能拿
    `lts__t_sectors` 当字节**（实测读 ×1.5、写 ×3 系统性放大），且小 shape 的 DRAM 写会因 dirty
    行滞留 L2 读成 0，须放大到 workspace ≫ L2（ratio 回升 0.82–0.96）才有意义。同步更新
    `docs/08` 待验证项。
  - P8-1：新增 `docs/08-summary.md`（一页纸汇总：一句话总览 / 数学两套形式 / 前向 K1-K2 结构对比表 /
    数值误差汇总表 / 性能汇总表 / 反向推导-分块-设计要点 / 8 条踩坑速查 / 交付物索引 / 待验证项），
    数字全部复用 `docs/01`–`07` 的本机实测，未新增实验。新增博客 post `flash-kda-08-summary`
    并更新 `flash-kda-00-index` 目录（原「8. ncu 剖析反向 kernel（待写）」替换为本篇），hugo 构建无 ERROR。
    踩坑：新 post 的 `date` 若晚于系统当前时间（本轮 23:03 CST，先写了 23:20）会被 Hugo 当 future
    页面排除，`relref` 报 `REF_NOT_FOUND`——把 date 调到当前时间之前即修复（buildFuture 默认 false）。

  - P7-2：`docs/07` 新增 §5「ncu 剖析：K1/K2 到底 bound 在什么上」+ `harness/probe_ncu.sh`
    （`harness/probe_ncu.out.txt`）+ `harness/ncu_driver.py`，更新博客 post `flash-kda-07-perf` 标题。
    ncu 2026.1.1、每 kernel 单独 profile（`-k regex -c 1 --launch-skip 3`，replay 12–20 passes，
    脚本裁掉超长 CUTLASS demangled 名）。三个 shape（T=1024 H=4 / T=8192 H=64 / B4 T=1024 H=64）
    实测结论：(a) **K1/K2 都不是 TensorCore bound**——K1 tensor pipe 全程 ≤1.6%、K2 22–24%
    （每活跃 SM），FMA 5–12%，故 §1.3 的 `TF/s` 只是相对算力；(b) 放大后两者都撞 **L2 带宽**：
    K1 T=8192 H=64 SOL 最大 pipe 是 L2 88.2%（DRAM 52.1%，occupancy 98.4%），K2 在 grid 填满
    （B4 H64，256 CTA→occ 18.06%）时 L2 **91.6%**（DRAM 53.0%、L1TEX 59.5%、tensor 仍 24.1%）
    →KDA 前向硬成本是 13824B/tile workspace + D×D 状态的搬运（K1 写入 L2、K2 读回 L2，
    流量 1.7–2.0GB），不是乘法；(c) **K1 第二瓶颈是 barrier stall**（大 shape 27.0 cyc/issue，
    ncu 称占发射间隔 43.5cyc 的 61%），来自 `fwd_kernel1.cuh:425/476/487/508/514` 一圈
    `__syncthreads()`（union 复用 smem 的先同步再覆写）+ shared **2.7-way/2.3-way bank conflict**
    （占 shared wavefront 45–54%）；小 shape（T=1024 H=4 grid 256）则因 grid 太小 occ 仅 24.2%；
    (d) **K2 被 shared memory 钉在 2 CTA/SM（98.43KB/block，theoretical occ 18.75%）**，设计点
    T=8192 H=64 只有 64 CTA < 132 SM → achieved occ 9.4%、ncu 报「grid < SM，估 51.5% 加速空间」，
    这正是 §2.4「per-CTA 串行延迟受限」的硬件证据；K2 的 stall 以 `wait`（固定延迟 MMA 依赖，
    ~1.1）为主、几乎无 barrier（warp-specialized）。**待验证**：K1「global load 仅用 4/32 字节/sector」
    具体落在哪条 load；K2 的 L2 流量中 workspace 读 vs $o$/state 写占比（需 nsys/插桩）。
  - P7-1：新增 `docs/07-perf.md` + `harness/bench_fwd.py`（`harness/bench_fwd.out.txt`）+ 博客 post
    `flash-kda-07-perf` 并更新 index。**统一计时口径**：`wall`（一个 CUDA event 对包 `REPS=50` 次取均值、
    3 轮取中位，含 host 派发）+ `dev`（`torch.profiler` CUPTI CUDA-activity，纯 kernel 时间），
    `CUDA_VISIBLE_DEVICES` 隔离到空闲卡。18 组 case（T=128…8192、H=1/2/4/8/16/64、B=1…8、T=1023）
    实测结论：(a) **FLA `chunk_kda` 的 wall 被 host 派发钉死在 ~0.86ms**（T=128 时 dev 仅 58µs，
    wall 780µs），因其 Python 包装拆成 ~10 个独立 kernel launch；FlashKDA 单 host 调用（2 kernel），
    `wall≈dev`（差 ~0.1%）→ 中小 shape FlashKDA wall 领先 2×–21×（**赢在省 host 税**）。
    (b) 纯 GPU 时间 `dev` 上 FLA 在中等 shape 反而更小（T=2048 H=4：122.9 vs 204.8µs），因 FLA 的
    状态递推 kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 沿 T 并行，而 FlashKDA K2 沿 T 串行、
    并行度只有 `B·H`；`B·H` 大时（B8 T2048 H8）FlashKDA **wall 与 dev 同时领先 2.2×/2.6×**，
    T=8192 H=16 因 K2 只有 16 CTA 而打平（0.98×），H=64 反超 1.84×。(c) K2 沿 T 串行的定量证据：
    固定 B=1,H=4 时 dev/T ≈ 100ns/token 恒定（1024→108.7、2048→204.8、4096→402.8µs），固定 T=1024 扫
    H=1→16 dev 仅 106→126µs（per-CTA 串行延迟受限而非吞吐受限）。(d) kernel 级分解：FlashKDA=3 kernel
    （K2 89%/K1 8%/init），FLA=10 kernel 拼装；T=8192 H=64 下 FLA 的非递推工作（intra 422+inter_solve 424+
    recompute_w_u 358+o 340+l2norm 177+cumsum 147+elementwise 63≈1.93ms）远大于 FlashKDA 的 K1 480µs，
    印证 K1「一次算完落盘」的融合收益。**方法学坑**：背靠背相邻 CUDA event 对的**中位数会读到真值一半**
    （用 `torch.mm(4096³)` 对照：median 174µs 而 sum/N=CUPTI=341µs），必须用「一个 event 对包 N 次取均值」。
    平台：H100 80GB HBM3、torch 2.12、FLA 0.5.2、triton 3.6、bf16、D=128。待验证项：FLA host 税组成
    未用 nsys 拆开（仅由 wall 恒定推断 host-bound）；K2 的 smem/TensorCore 占用留 P7-2。
  - P6-2：新增 `docs/06` §6「P6-2 最小可运行反向 kernel——B1 状态伴随实测」+ `harness/check_bwd_dhu_kernel.py`
    （`harness/check_bwd_dhu_kernel.out.txt`）。在 H100 用 Triton 写出**最小可运行的状态伴随（B1/`dhu`）kernel**：
    grid `(cdiv(V,BV), B·HV)`，沿 block 逆序，每块 `dU=k_rest@dS+dU_in`（=FLA `dv2`）、
    `dS ← E_tot⊙dS + q_dec^T do·scale − w^T dU`、落盘 `dh[c]`，四行与 `fla/ops/common/chunk_delta_h.py:576-587`
    逐行同构。三方对拍（fp32，两组配置 B1 T256 H2 D128 BT64 / T512 H4 D128 BT32）：**Triton↔FLA `dhu`
    max_rel 4.6e-5 / 3.7e-5**（`dh`/`dv2`/`dh0` 全 ≤4e-5），**torch 显式递推↔FLA 1.1e-3 / 1.2e-2**——因
    FLA 与 Triton 同走 tf32 `tl.dot`、torch 走 ieee fp32，证明**`dhu` 的误差底噪是 tf32（~1e-3）而非 fp32**；
    与 `src/kda_bwd_ref.py::kda_chunked_backward` 的 `dh0` 对到 **3.3e-4 / 5.2e-3**（Part A2），钉死 B1 即
    `05` §7.3 的跨块状态伴随链。**受控消融**（回答 §5.3 待验证项）：bf16 carry / bf16 落盘 `dh` 的相对误差
    均 **2.5e-3~2.7e-3**，因 B2 对 `dh` 线性，该误差原样传给 `dk/dβ/dA`，落在 bf16 GEMM 底噪内→**bf16 存状态
    伴随可行**。**踩坑**：`dhu` 初始 `dS` 来自终态梯度 `dht`（此处为 0）而非初态 `h0`（`h0` 只进 B2 的 `detot`）；
    第一版误用 `h0`，指纹是「`dh0` 对但**仅最末块**的 `dh`/`dv2` 全错」。计时：107 µs（T256 H2）/191 µs
    （T512 H4），朴素实现、未优化，价值在钉死契约与误差底噪（非整除 T、K 的 64 分段**待验证**）。
  - P6-3：新增 `docs/06` §6.5 + `harness/check_bwd_dhu_boundary.py`（`harness/check_bwd_dhu_boundary.out.txt`），
    关闭 P6-2 的「非整除 T」待验证。**根因**：kernel 本体已 T-general（`m_t=o_t<T` 把 pad 行置零），
    只有 host wrapper 用 `.view(B,NT,BT,H,K)` 折块、假设 T 整除 BT；$T=250$ 尾巴 58 行时直接
    `RuntimeError: shape '[1,4,64,2,128]' invalid for input of size 64000`，即**尾块的 $E^{tot}$ 从未被寻址**。
    FLA 前向量不给 pad（`g_cum` $[B,T,H,K]$），其 `dhu` 靠 mask 处理非整除 T，是合法参照。**修法**：
    $E^{tot}_c=2^{g\_cum[\min((c+1)BT,T)-1]}$（块内局部 cumsum，末有效行即该块门控和）。实测 fp32
    Triton↔FLA：T=256（可整除）**逐位回归 P6-2**（dh 4.60e-5/dv2 2.07e-5/dh0 1.82e-5）；非整除
    T=250/100/513、B2T130H3、T250BT32 六组 `dh/dv2/dh0` 全落 **9e-8~4e-5**；单块 T=31 时 `dh≡0`、
    `dv2` 逐位等于 FLA（$dU=k\cdot0+dv$，tf32 下 0 精确）；另检 FLA `dhu` 的 `grid=(cdiv(V,BV),N*HV)`，
    **K 不占 grid**、`K=64/128/…` 是同 program 内 `o_k1..o_k4` 展开→「K 的 64 分段」非正确性缺口。
  - P5-4：新增 `docs/05` §7「反向分块化（chunked backward）推导」+ `src/kda_bwd_ref.py::kda_chunked_backward`
    + `harness/check_bwd_chunked.py`（`harness/check_bwd_chunked.out.txt`）。把逐 token adjoint（式 5.2–5.8）
    折叠成块内形式：输出支路 $S_{in}+=qd^\top\bar o$、状态更新支路 $S_{in}+=E^{tot}\odot\bar S_{out}$、
    $U=P u'$、$P=(I+L)^{-1}$ 的伴随 $\bar L=-P^\top\bar P P^\top$（**复用前向 $P$，不再求逆**）、
    两 GEMM $A_L=kd\,ki^\top$/$A_{qk}=qd\,ki^\top$、衰减链与反向前缀和，块间只传 $D\times D$ 状态伴随
    （式 7.13）。对拍三方：逐 token 显式 adjoint（P5-2 已证与 autograd 逐位相等）/autograd 穿
    `kda_chunked_fwd` 本体/autograd 穿 `kda_token_recurrent`。13 组 case（$T=16\to512$、$D=12\to128$、
    $g\in\{0,1,2,4\}$、chunk $\{15,16\}$、含初态/终态上游）实测：**fp64 六项 $max\_rel\sim10^{-15}$（机器精度，
    含 $D=128,T=512$）**；fp32（GPU）$\sim10^{-3}$、bf16（GPU）$\sim10^{-2}\sim3\times10^{-1}$。**踩坑**：$S_{in}$
    出现在输出/状态更新/pred 三处，只有状态更新那支带 $E^{tot}$，第一版把输出支路的 $qd^\top\bar o$
    并进状态伴随再整体乘 $E^{tot}$，指纹是「`dq` 逐位对、其余五项全错($max\_rel\approx0.3$)」，改成独立
    累加器即 fp64 归零。fp32 的 $\sim10^{-3}$ 经定位是**参考实现块内求逆走 fp32 `torch.linalg.solve`**：
    同输入下 `kda_chunked_fwd` 在 GPU 上与前向逐 token 就差 $\sim2.5\times10^{-4}$（CPU 仅 $\sim2\times10^{-7}$，
    已写进 harness 输出），梯度再放大；「本实现 vs autograd(chunked)」仅 $\sim5\times10^{-4}$ 证明差异来自
    $P$ 而非反向公式。更新博客 post `flash-kda-05-bwd` 标题。
  - P5-3：新增 `harness/check_bwd_fla.py`（`harness/check_bwd_fla.out.txt`）。用
    `use_qk_l2norm_in_kernel=False`/`use_gate_in_kernel=False`/`use_beta_sigmoid_in_kernel=False`
    让两侧吃**同一份** bf16 操作数（归一化后 q/k、`sigmoid(beta)`、自然对数衰减 `g_fla=g_log2·ln2`，
    默认 $[N,H,K,V]$ 布局），损失 $L=\langle o,\bar o\rangle+\langle S_T,\bar S_T\rangle$，
    参考实现跑 fp64。5 组 case（$T=128/512/1024/1023$、$H=2/4/8$、$g=N(0,1/2)$、含 $g=0$）实测：
    FLA chunked 反向 vs 显式 adjoint 的六项 `max_rel` 全部落在 bf16 底噪 **4.1e-3~8.2e-3**
    （`dq 4.1e-3/dk 5.9e-3/dv 8.2e-3/dg 6.8e-3/dbeta 7.6e-3/dh0 7.2e-3`），且前向 sanity
    $o$/$S$ 误差 3e-3~8.5e-3 与反向同阶（反向误差≈前向误差，无算法级偏差/无随 $T$ 漂移）；
    fp64 与 fp32 显式参考结果到打印精度相同→误差由 FLA bf16 主导。**踩坑**：FLA 的 `g` 是自然对数
    衰减、参考是 log2，换算 `g_fla=g_log2·ln2` 导致 $\partial L/\partial g_{\log2}=(\partial L/\partial g_{fla})·\ln2$；
    误写成除 $\ln2$ 时 `dg` 的 `max_rel` 稳定停在 0.52（$\approx1-1/\ln^2 2$）、其余五项正常，
    改回乘 $\ln2$ 即降到 3e-3~7e-3。更新 `docs/05` 新增 §6「与 FLA 反向对拍」、小结顺延 §7，
    更新博客 post `flash-kda-05-bwd` 标题与 index 目录。
  - P5-1/P5-2：新增 `docs/05-kda-bwd.md` + `src/kda_bwd_ref.py` + `harness/check_bwd_ref.py`
    （`harness/check_bwd_ref.out.txt`）。**P5-1**：从逐 token gated delta rule 推导完整 adjoint——
    状态伴随合成 $G_t=\bar S_t^{rec}+\text{scale}\,q_t^\top\bar o_t$、更新支路
    （$\bar S_{t-1}{+}{=}\alpha_t\odot_{row}G$、$\bar k_t{+}{=}G\delta_t$、$\bar\delta_t=G^\top k_t$、
    $\bar\alpha_t{+}{=}\sum_v G_{:v}\odot(S_{t-1})_{:v}$）、delta/pred 支路、衰减链
    $\bar g=\bar\alpha\odot\ln2\cdot\alpha$，以及门激活链（$g=C\sigma(u)$，$u=a(g^{logit}+dt\_bias)$）
    到 `g_logits/A_log/dt_bias`。**P5-2**：`src/kda_bwd_ref.py` 用纯 `torch`（**不调 autograd**）
    实现反向核心循环（`kda_bwd_ref.py:75–106`）与 `kda_gate_backward`。实测 16 组 case
    （T=1..512、D=16..64、g={0,1,2,4}、多 seed）fp64 下 `dq/dk/dv/dg/dbeta/dh0` 与 `torch.autograd`
    **逐元素完全相等（max_abs=0）**，fp32 下 max_abs≤5.7e-6/max_rel≤6.4e-7（机器精度）；中心差分独立
    核验（fp64，ε=1e-6）六项 $\bar q,\bar k,\bar v,\bar g,\bar\beta,\bar h_0$ 差 ~1e-10；gate 链对
    `kda_gate` autograd max_rel≤1.1e-7。反向结构特征：token 维串行（状态伴随沿时间前传），块内可并行。
    新增博客 post `flash-kda-05-bwd` 并更新 index 目录（hugo 构建无 ERROR）。
  - P4-3：新增 `harness/check_three_way.py`（`harness/check_three_way.out.txt`）：在同一批输入上对
    fp64 golden（`kda_ref.py::kda_token_recurrent`）/ FlashKDA `flash_kda.fwd`（CHUNK=16）/ FLA
    `chunk_kda`（chunk=64）做三方对拍，状态统一在 $S^\top$ 布局比。16 组 case（T=64..1024/1023、
    H=1/2/8、B=3、seed=0..3、g=N(0,0/1/2/4)）实测：`o` 三方 `max_rel` 均 $3\times10^{-3}\sim8\times10^{-3}$、
    状态 $5\times10^{-3}\sim1.1\times10^{-2}$，同处 bf16 底噪；**`flash↔fla` 的 `o` 误差（4.88e-04/3.65e-03）
    小于各自 vs golden（8.37e-04/6.22e-03）**，用**共模抵消**解释（两 bf16 实现共享同一份 bf16 输入量化，
    互相之差把 bf16 输入舍入减掉，只剩算法/累加序差异）；`g=0` 无衰减时状态误差三方都抬到 $\sim1.4\times10^{-2}$
    且不再共模（状态舍入噪声相互独立），与 §4/P3-3 §18 互洽。更新 `docs/04` 加 §6「三方对比」并把小结顺延为 §7。

  - P4-1/P4-2：新增 `docs/04-numerics.md` 与 `harness/check_fwd_numerics.py`
    （`harness/check_fwd_numerics.out.txt`）：以 `src/kda_ref.py::kda_token_recurrent` fp64 逐 token
    递推为 golden，对编译好的 FlashKDA 前向做端到端对拍。17 组 case（T=16/17/64/128/512/1024/1023、
    H=1/2/8、B=3、seed=0..3、g=N(0,0/1/2/4)）实测：`o` 峰值相对误差 **4.0e-3~7.6e-3**、终态
    **5.0e-3~1.4e-2**，与 bf16 有效精度一致；`o.max_abs` 随 T 不变（峰值误差在早期 token，无累积漂移）；
    `g=0` 无衰减时状态误差升到 1.4e-2（bf16 累加），与 P3-3 §18 受控消融互洽。
    **踩坑（重要布局约定）**：FlashKDA 的 `initial_state`/`final_state` 是 **$S^\top$==[N,H,V,K]**，
    与 `kda_ref.py` 的 $S$==[K,V] 相反；方阵不触发形状检查，错布局只表现为「初态处误差大、有衰减时又收敛」。
    新增 `harness/probe_state_layout.py`（`harness/probe_state_layout.out.txt`）三情形复现：零初态
    o max_abs=2.3e-4；随机初态未转置时 t=0 误差 1.6e-1 逐 token 收敛到 1.1e-3；直接比状态
    $S$ vs fin=0.511、$S^\top$ vs fin=3.98e-3，钉死布局。

  - P3-3：`docs/03` 新增 **Part III（§17–§20）**，全部贴真实行号：
    (a) §17 分块含义：从逐 token gated delta rule 展开得 `S_C = diag(E_tot) S_0 + k_rest^T U`（式 17.2），
    `k_restored = k⊙2^{-g_cum}⊙E_tot`（K1 `fwd_kernel1.cuh:453–468`）承载「写入点到块末剩余衰减」；
    澄清 kernel smem `g_total` 已是 `E_tot=2^{sum g}`（`fwd_kernel1.cuh:330,353–356`；`gate_scale=lower_bound·log2e`
    `flash_kda.cpp:128`），K2 逐 key 列缩放（`fwd_kernel2.cuh:684–685/706–707/718–719`），与
    `kda_ref.py:135` 行缩放互为转置写法；块内前缀和沿 token、按 key 列独立（`fwd_kernel1.cuh:311–330`）。
    (b) §18 新增 `harness/check_state_bf16.py`（纯 PyTorch 受控消融，fp32 工作精度、fp64 golden）：
    隔离「bf16 状态存储」与「bf16 GEMM 操作数」——T=1024 g=1（衰减区）下 o max_rel：fp32 3.8e-07、
    store_bf16 3.5e-04、gemm_bf16(st_fp32) 1.7e-03、kernel_bf16 1.7e-03；把状态 fp32→bf16 存储的**边际**
    输出变化仅 o max_abs=2.8e-05。g=0（无衰减病态）时 bf16 存储边际升到 o ~2e-3。fp32-FMA vs bf16-FMA
    更新算术：g=4 逐位相同、g=1 输出差 1.2e-4。结论支持 deep-dive §3「有衰减 regime 下 bf16 存储对输出
    无可测量损失」，与 `probe_refs.out.txt` 端到端 FLA/FlashKDA 误差（状态 max_rel ~6–9e-3）量级互洽。
    (c) §19 状态落盘 bf16/fp32 两条路径（`fwd_kernel2.cuh:786–801`、`804–835`，union `99–107`）；
    §20 行号速查。更新博客 post `flash-kda-03-fwd-k2` 标题。

## 阻塞

- （空）
