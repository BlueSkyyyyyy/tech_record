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
- [x] **P6-4** B2 `wy_dqkg` 最小可运行 Triton kernel（对应 FLA `chunk_kda_bwd_wy_dqkg_fused`）：
      独立 torch 参考 vs FLA 六项梯度 5.8e-4~2.1e-3（=tf32 底噪，证代数一致）；本 kernel vs FLA
      逐位相同（含非整除 T）；bf16 操作数下 `dq/dv/db/dA` 2.3e-3~6.4e-3、`dk/dg` 3.3e-2~5.0e-2；
      记录运行期 FLA 路由到 TileLang 后端（fp32 编译失败）的陷阱（`docs/06` §6.6）。
- [x] **P6-5** B3 `intra` 最小可运行 Triton kernel（对应 FLA `chunk_kda_bwd_kernel_intra`）：
      独立 torch 参考 vs FLA `dq/dk/db/dg` 3.0e-4~8.8e-4（=tf32 底噪）；本 kernel vs FLA `dq/dk`
      逐位相同（max_abs=0）、`db/dg` ~1e-7（atomic-add 顺序）；含非整除 T；bf16 四顶抬到
      1.1e-2~4.8e-2；朴素计时与 FLA 同速。至此 B1/B2/B3 三条 GEMM 支路全部位级可核验
      （`docs/06` §6.7）。
- [x] **P6-6** B0 门控重算最小可运行 Triton kernel（对应 FLA `kda_gate_chunk_cumsum` +
      `kda_gate_bwd`）：前向重算 β/`g_cum` 与 torch/FLA ~2e-7（含非整除 T、g=0、B=2），β 与 FLA
      逐位相同；反向门控链（反向前缀和 + `dg_logits/dA/dbias`）与 autograd ~3e-7；**交叉验证到
      FlashKDA K1 当场算的 workspace `kGTotal`（E^tot）max_rel 1.6e-5~8.1e-5**（差=K1 tanh.approx
      sigmoid）。坑：`log2e` 必须乘在反向前缀和上（漏掉则误差稳停 0.3069=1−ln2）；FLA
      `kda_gate_bwd` 输出随输入 dtype 是 bf16（B0 vs FLA 表面 3e-3 实为 bf16 出口）。至此 §5.2
      的 B0/B1/B2/B3 四段全部有最小可核验 kernel（`docs/06` §6.8）。
- [x] **P6-7** 补齐反向第 5 个 kernel `dAv`（输出支路）并做**端到端组装**核对：
      新增 `harness/check_bwd_e2e.py` + `docs/06` §6.9。`dAv` 与 FLA 逐位相同；把
      `dAv→B1→B2→B3→反向 cumsum(B0)` 串成完整反向 vs `torch.autograd` 穿 `chunk_kda`，
      六项梯度全落 tf32 底噪（dq 逐位、dk≤1.5e-5、dv≤3.9e-4、dg≤1.1e-5、dbeta≤1.1e-4、
      dh0≤5.1e-6，含非整除 T）；Part C（FLA 自分解==autograd 逐位）+ Part D（swap 后 B2/B3==FLA
      到 1e-7）把残差钉死到 B1 的 tf32 舍入。两个组装才暴露的坑：B2 吃 `v_new` 而非 `u`、
      反向 cumsum 不要额外乘 ln2（FLA `chunk_local_cumsum` 默认 `scale=RCP_LN2`）。B1 补 `dht` 入口。
- [x] **P6-8** 把整条反向**直接架在 FlashKDA 自己的 K1 workspace 上**（不再借助 FLA 前向中间量）：
      新增 `harness/check_bwd_ws_reuse.py` + `docs/06` §6.10。按 head-major SoA 偏移切六字段
      （`utils.cuh:70-76`）+ B0 重算，逐块重建 $S_{in}/U/w/h/Aqk/Akk$，跑 `dAv→B1→B2→B3→B0`。
      Part A（workspace 重算前向 vs FlashKDA 自身 `out`/`final_state`）4.5e-3~8.9e-3；Part B（七个
      操作数 vs `kda_ref` 独立重算）1.3e-4~8.1e-3；Part C（六项梯度 vs autograd）最坏 6.4e-3
      （6 组含非整除 T、$g=0$、$B=2$）。四个集成坑：字段展平须按 `(NT,CHUNK,H)` 重排（错则 qg/kg rel≈1）、
      `INV` 即 $P$ 不需转置、golden 须吃激活操作数、非整除 T 按 K1 补零尾块。

### P7 性能与剖析（`docs/07-perf.md`）

- [x] **P7-1** FlashKDA forward CUPTI 计时（不同 T/H），与 FLA 对比。
- [x] **P7-2** `ncu` 剖析 K1/K2 bound（smem/TensorCore/occupancy），逐项证据。
- [x] **P7-3** K1/K2 访存字节拆分（DRAM 读/写逐项归因 workspace vs `v` vs `o`/state），
      解决 P7-2 待验证 (b)。
- [x] **P7-4** 归因 K1「global load 4/32 bytes/sector」，解决 P7-2 待验证 (a)（静态 SASS 计数 +
      `SourceCounters` 逐行 + DRAM 占比）。
- [x] **P7-6** 关闭 P7-1/§4 待验证「FLA host 税的组成」：用 `process_time`（+warmup）实测
      FLA 前向 host CPU **固定 ~0.82–0.86 ms/次、与 T 无关**（T=16…2048），拆开 =
      `chunk_kda_fwd` ~525 µs + `l2norm_fwd`×2 ~154 µs（**单次 77 µs，而裸 Triton launch 仅 2.4 µs**）
      + `fused_beta_sigmoid` ~63 µs + autograd/dispatch 杂项 ~95 µs；`FLA_DISABLE_BACKEND_DISPATCH=1`
      掉 ~190 µs。nsys 稳态：FLA **12 个 kernel/次**（9 `cuLaunchKernelEx`+3 `cudaLaunchKernel`，
      driver 中位 ~6.3 µs）、FlashKDA **3 个**；结论＝那 0.8 ms 几乎全是 FLA 每 kernel 一次的 Python
      launch 包装，与 GPU 计算（~82 µs）/autograd/分配无关（`docs/07` §9）。
- [x] **P7-7** 反向性能与 host 税剖析：把 P6-7/P6-8 组装的 workspace 驱动反向链
      （`dAv→B1(dhu)→B2(wy_dqkg)→B3(intra)→反向 cumsum(B0)`）与 `torch.autograd` 穿
      `chunk_kda` 放在统一口径（event-pair `wall` + CUPTI `dev`）下比，只计反向。新增
      `harness/bench_bwd.py`（`harness/bench_bwd.out.txt`）+ `docs/07` §10。四组 shape
      （1×512×4 / 2×1024×8 / 8×2048×8 / 1×250×3 非整除）实测：ours wall/dev **0.41–0.76×**、
      **0.60–0.79×** FLA 反向（前向对照 `fwd` dev 63.5/128.2/668.7/42.9µs，端到端 fwd+bwd
      dev 0.5–0.8×）。**关键归因**：FLA 反向 profile 里含**前向状态重算**
      `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（B2T1024H8 578µs）+`recompute_w_u`（32.7µs）
      =FLA 反向 dev 的 **46.2%**（T512H4 时 21.8%），我们复用 K1 workspace 从不重算——把 §5.2
      「复用 K1 workspace」的设计收益从正确性（P6-8）兑现成性能。**短板**：B1 `_dhu_kernel`
      是瓶颈（占 ours dev 55.5–62.8%）且比 FLA `dhu` 慢 3.9×，是下一步工程化首要目标
      （应镜像前向 K2 的 warp specialization）。host 税非主要矛盾（ours 110–260µs、
      FLA 500–730µs/次，大 shape 全被隐藏）。
- [x] **P7-8** 复核 P7-7 的反向计时：P7-7 表在**未隔离 GPU**（GPU 0 被另一容器打满）时测得，
      两侧绝对值被抬高约 2×，关键两行不可复现（ours B1 @B2T1024H8 539.8→**214.3**、FLA `fwd_h`
      578→**83.6**）；比值结论不变但「重算占 FLA 反向 dev 46.2%」更正为 **17.8%**。用控制变量脚本
      （`check_bwd_dhu_dtype.py`）拆开 B1 与 FLA `dhu` 的两个混淆——**chunk 粒度（16 vs 64）** 与
      **操作数 dtype（fp32/tf32 vs bf16）**：同 BT 同 dtype 下 ours/FLA = **1.1–1.9×**（非 3.9×）；
      让 B1 吃 workspace 原生 bf16（不 `.float()`）→ **2.2–2.5×**，B2T1024H8 已快过 FLA `dhu`。
      修正归因并重排优化优先级。
- [x] **P7-9** 把 P7-8 §11.4 的「B1 吃 workspace 原生 bf16」杠杆**落到真实组装链**（此前只在孤立
      kernel 上手工验证）：新增 `harness/check_bwd_b1_bf16_e2e.py` + `docs/07` §12。只换 B1 的 mma
      操作数（状态仍 fp32、`dh/dv2` 输出保持 fp32 以兼容 B2/B3 的 fp32 操作数），正确性整链仍落 bf16
      底噪（六项 vs autograd ≤8.3e-3，与 P6-8 同阶；操作数换 bf16 自身增量 `dq` 逐位 0、其余 ≤3.8e-3）；
      反向 dev 降到 fp32-B1 链的 **0.71–0.91×**、FLA 反向的 **0.51–0.60×**；B1 kernel 快 **1.68–2.09×**
      （$B2\,T1024\,H8$ 208→119µs，已快过 FLA `dhu` 137.4µs）。§11.5 优化优先级第 (1) 步完成。
- [x] **P7-10** 兑现 §11.5 第 (2) 步「B1 更大 chunk 重组」：从 **16-chunk K1 workspace 免前向重算**地
      恢复 64-chunk 操作数（`qg/kg/kd/ki/E64` 逐子块重标定 + 重算 `P64/w64`），喂 bf16 B1 核 @BT=64。
      Part A：`qg/kg/w/g_cum` vs FLA forward@64 落 bf16 底噪（1e-3~1.7e-2，含非整除 T=250/513）；Part B：
      B1@64 vs FLA `dhu`@64 `dh≤1.55e-2/dv2≤1.87e-3/dh0≤1.44e-2`；Part C：**B1 快 2.1–2.3×**
      （T512H4 45.5→19.5µs 等）。硬伤：torch 原型恢复 ~2.0–5.7ms（`A64/P64/w64` 占 2.0–3.7ms），须融成
      Triton pass；且整链不干净（B2/B3 仍 16、B1@64 只落 64 边界 `dh`）。见 `docs/07` §13。
- [x] **P7-11** 兑现 P7-10 的「融成 Triton pass」：把 16→64 re-chunk 恢复写成**单个** Triton kernel
      （grid `(NG,B·H)`；重标定 `qg/kg/g_cum64` + 6 个非对角 $A_{ij}$ + 4×4 块前代合并 + `w64` 一次算完）。
      关键：$P64$ 的**对角块直接复用落盘 $P16$**（workspace `INV`），**不重算任何 16×16 求逆**；非对角
      $A_{ij}=kd16_i(kr_j2^{e_{ij}})^\top$ 把对偏移 $e_{ij}\le0$ 折进更小的 `kr`，绕开 §13.4 的 fp32 溢出。
      Part A：`qg/kg/g_cum` vs torch 原型 1e-7~8e-6、`w` 1.6e-2（=bf16 `P16` 的精度上限，同 §13）；vs
      FLA fwd@64 落 bf16 底噪（含非整除 T=250/513、g=0）。Part B：pass→bf16 B1@64 vs FLA `dhu`@64
      `dh≤1.44e-2/dv2≤1.87e-3/dh0≤1.20e-2`（与 P7-10 在 torch 原型上同阶）。Part C：kernel-only dev
      **41–953µs（ieee）/ 16–492µs（tf32）**，为 torch 原型 full（2.1–5.6ms）的 **0.11–0.25×**。新增
             `harness/check_bwd_rechunk_pass.py`（`..._pass.out.txt` + `..._pass_tf32.out.txt`）。见 `docs/07` §14。
- [x] **P7-12** 闭合 §13.5 (ii)：把整条反向 `dAv→B1→B2→B3→反向 cumsum` **统一到 64 块**且仍只吃 FlashKDA
       16-chunk K1 workspace（免前向重算）——在 P7-11 的 `qg/kg/w/g_cum/P64` 之外，从同一批 16 块字段
       补齐 `h64=h16[0::4]`（状态粒度无关）、`Aqk64`（对角=落盘 `Mqk`，非对角 `qd16_i(kr_j2^{e})^T`）、
       `U64=P64 diag(β)(v-kd64 h64)`。Part A：七项前向量 vs FLA forward@64 落 bf16 底噪（qg 8.3e-3 …
       P64 3.0e-2，含非整除 T=250/513、g=0；`Aqk` 因 FLA 未初始化三角+溢出不入表，靠整链核验）。
       Part B：整链@64 vs `torch.autograd` 六项 `dq 9.8e-3/dk 7.9e-3/dv 1.3e-2/dbeta 6.0e-3/dg 7.7e-3/
       dh0 8.3e-3`，与 16 链同阶。**Part C（关键结论）**：整链@64 反向 **dev 反而慢 2.0–2.5×**——
       逐 kernel 归因 B1@64 快 2.1–2.2×，但 B2 `wy_dqkg`@64 慢 1.8–2.9×、B3 `intra`@64 慢 4.8–5.5×（块内
       intra 工作 $\propto T\cdot BT$），B1 的省被整个吃掉。**故 §13.5 (ii) 的答案是「不要把整链跟到 64」**，
               反向维持 16 块链；B1@64 仅作孤立加速手段。见 `docs/07` §15。
- [x] **P7-13** 用 `ncu` 剖析反向瓶颈 **B1**（此前只剖析过前向 K1/K2）：新增
      `harness/ncu_bwd_driver.py` + `harness/probe_bwd_ncu.sh`（`harness/probe_bwd_ncu.out.txt`）。
      B1 三种形态（ours-BT16/ours-BT64/FLA-`dhu`）**都不是任何硬件单元的 bound**（tensor 4.5–14.9%、
      DRAM ≤28.6%、L2 ≤39.8%），achieved occupancy 一律 ~6.2% 是**小 grid（64/16 block ≪ 132 SM）**
      的症状、非旋钮；本质是**延迟/并行度受限**。同 BT=64 下 ours 比 FLA 慢 2.14×，差距=`long_scoreboard`
      3.88–5.29/8.19–9.68（≈47–55% issue 间隔在等全局加载）vs FLA 1.06/5.06，源于 ours **单条 `[128,BV]`
      依赖链 + `num_stages=1` 无流水**，而 FLA 拆 K 为两条 64 链（`chunk_delta_h.py:401-415/585-604`）
      + `num_stages∈{2,3,4}` 流水（`:350-352`，180KB smem/block、occ 6.25% 仍更快）。ours BT16→64 快
      1.81× = 串行步数 ×4 + `dh` 落盘 DRAM 写 ×6（20.8→3.43MB）。**设计输入**：目标卡 CUDA 版优先做
      (1) 镜像 K2 的 TMA 输入流水、(2) D=128 状态拆 2 条独立累加链，而非提占用率。见 `docs/07` §16。
- [x] **P7-5** 关闭 P7-2/P7-4 待验证 (c)：**K2 突破 2 CTA/SM 后 L2 墙是否后移**——实验=否。
      改造 `SharedStorageK2`（fp32 状态缓冲改按 `StateFP32` 条件预留）+ 扫 pipeline stage，编译 6 个变体，
      实测占用率/ncu SOL/K2 时间。结论：smem 被无条件预留的 65536 B fp32 缓冲钉死为 98432 B（只减
      stage 不动 smem）；真进 3 CTA/SM 的配置必须砍 pipeline 深度，反而慢（`cond(2,1)` +14%、
      `cond(1,2)` +40%）；K2 在 grid 填满时 `lts` 已 93–94% SOL，是 L2 带宽 bound，占用率无可用余量。

- [x] **P7-14** 在 bf16 Triton B1 原型上**先验证 §16.5 的两条设计输入**，避免盲目 CUDA 重写：
      新增 `harness/check_bwd_b1_ilp.py`（`harness/check_bwd_b1_ilp.out.txt`）+ `harness/probe_bwd_ilp_ncu.sh`
      / `ncu_bwd_driver.py --variant`（`harness/probe_bwd_ilp_ncu.out.txt`，空闲卡）。四变体
      （base / `tl.range` 流水 / 拆 K 两条链 / 两者）。**Part A**：四变体与 FLA `dhu` 数值同阶（含非整除
      T=250），拆链甚至把 `dv2/dh0` 精度略降。**Part B（关键）**：`B2T1024H8`@BT=64 内核-only
      base 45.2µs（1.96×FLA）→ 流水 ns=3 **23.0µs（1.00×FLA）**、`B8T2048H8` **0.85×**、`B1T512H4` 0.97×；
      **拆链单独 51.1µs（比 base 还慢 13%）、与流水叠加 25.6µs（比纯流水慢）**。部署粒度 BT=16 流水亦
      1.4–2.0×。**`ncu`**：流水把 `long_sb` 5.27→1.46→**0.96**（≈FLA 1.06）、`tensor%` 6.8→15.9、DRAM
      13.5→30.6%，smem 65.5→181.3KB（≈FLA 180.2）占用率 12.5→6.25% 却最快（再证非占用率受限）。
      **结论**：§16.5 第 (2) 条（拆累加链）在本原型上**冗余**（流水已填满 ILP），降级为「不推荐」；
      第 (1) 条（镜像 K2 的 TMA 输入流水）是**全部**收益来源。目标卡 CUDA B1 第一优先级＝照抄 K2 流水。
      见 `docs/07` §17。
- [x] **P7-15** 用**手写纯 CUDA** B1（`nvcuda::wmma` bf16 m16n16k16，fp32 状态，per-warp V 列，
  `cp.async` 预取深度 `NSTAGE` 为超参）补 §17.4 留下的边界（「手写 CUDA 显式化预取后是否还需拆链」）：
  新增 `harness/b1_cuda_probe.cu` + `harness/check_bwd_b1_cuda.py`（`harness/check_bwd_b1_cuda.out.txt`）
  + `harness/ncu_b1_cuda_driver.py`；`docs/07` §18。**Part A**：与 FLA `dhu` **逐位相同
  （`dh/dv2/dh0` max_abs=0）**，BT=32/64、含非整除 T=250（两侧同为 bf16 mma + fp32 累加且逐条 op 序
  相同）。**Part B**：本原型绝对速度差 FLA `dhu` **4.3–9.6×**、`NSTAGE` 预取只值 **2–6%**（小 shape
  0.92–0.98×），**大 shape 反而 1.40–1.45× 变慢**（NS=2/3 的 smem 到 168–226KB → 占用 2→1 CTA/SM）。
  **结论**：裸 wmma 版**根本没到 load-bound**（每 CTA 2 warp、~108KB smem→每 SM 2 warp、3 GEMM/chunk
  仅 ~1.9 TFLOP/s≈峰值 0.2%），故 §17 的「流水杠杆」**不是可单独加上的开关**——它只在 kernel **已是 K2
  形状**（warp-specialized + TMA 大块搬运 + 足够占用）时才兑现；「拆链」在 CUDA 里无从单独消融（warp 内
  `[K,16]` 状态本就是 8 条独立 mma m-tile，与 §17「拆链冗余」一致）。**两坑**：(a) warp-local 的
  `S/duf/du` 切片必须用 warp 内 lane（步长 32）索引，用 block 级 `lane+blockDim.x` 会漏写一半（全零输入
  通过、`dv2/dh0` 差 O(1)）；(b) ring buffer 的 stage 号须按处理序 `i%NSTAGE` 而非 chunk 号 `c%NSTAGE`
   （NSTAGE=1 恒等、藏住 bug）。
- [x] **P7-16** 用 `ncu` 剖析 P7-15 的**手写 CUDA B1**（此前只剖析过 Triton B1），把 §18.4「非 load-bound
   但瓶颈在占用率/发射/同步」的**定性归因量化并修正**。新增 `harness/ncu_b1_cuda_driver.py` 参数化改造
   （`--which ours|fla --bt --ns --no-build`）+ `harness/probe_b1_cuda_ncu.sh`
   （`harness/probe_b1_cuda_ncu.out.txt`，7 配置：ours BT64 NS=1/3、`-DB1_SYNC`、BT16、大 shape + FLA
   `dhu` 对照）。**坑**：`nvcc` 在 ncu 下会挂起（每次 run 卡 >20 min、只留 header），改为 ncu 外预编译 +
   `--no-build`；ours 路径直接合成操作数、不碰 FLA/Triton（后者 setup 拉起几十 kernel 撑爆 profile）。
   **结论（三个）**：(1) **不是全局加载**——`cp.async` 已把 `long_sb` 压到 0.66（NS=1）/0.57（NS=3），换成
   同步写（SYNC）则暴涨到 **3.96**、Duration 111.6→**183.3µs（+64%）**，即预取确实在隐藏加载、且已无事可
   做（与 §18 的 NS 只值 2–6% 自洽）；(2) **不是同步、不是占用率**——`barrier` 仅 0.02–0.09 拍（§18.4 猜的
   「同步」要更正），theoretical occ 与 FLA 同为 **6.25%**、大 shape achieved 6.09% vs FLA 6.19% 仍慢
   **4.9×**；(3) **真瓶颈 = `wait`（mma 固定延迟 1.80 拍≈36% 发射间隔）+ `short_sb`（smem 操作数取数
   0.87–1.56）+ 76% shared-load bank conflict**（ours 827520/1083520；FLA ldmatrix 仅 154/18586=0.8%），
   叠加 FLA tensor 密度 **14.1% vs ours 2.28%**。**设计输入修正**：目标卡 CUDA B1 第一优先级＝**换掉
   wmma-from-smem（ldmatrix/TMA 整块 + 提 mma 密度，消 bank conflict）**，`cp.async`/TMA 预取只是「不退化」
   保底（与 §17 的 Triton 结论相反，因裸 wmma 的 global 等待本就被同步+小 tile 掩盖）。见 `docs/07` §19。
- [x] **P7-17** 用**最便宜的受控实验**隔离 CUDA B1 的 bank-conflict 项，检验 §19.5「消 76% bank
  conflict 是量级项」的说法：只把 smem 行 stride 各 +8 bf16（宏 `B1_PAD`，退化版 swizzle），
  同源编译两次。新增 `harness/check_bwd_b1_pad.py`（`harness/check_bwd_b1_pad.out.txt`）+
  `harness/probe_b1_cuda_pad_ncu.sh`（`harness/probe_b1_cuda_pad_ncu.out.txt`）+ `docs/07` §20。
  **Part A**：pad 与 nopad、与 FLA `dhu` **逐位相同（`dh/dh0/dv2` max_abs=0）**。**Part B/C**：
  pad 一律快 **10–23%**（BT=64 小 shape 105.0→84.5）且 smem **+29%**，但 **B8T2048H8 BT=64 pad 反慢
  1.21×**——143.6 KB > 116.7 KB 门槛把占用率从 2 打到 1 CTA/SM。**Part D（ncu）**：bank-conflict
  从 **76%→19–25%**（残留=fp32 状态标量访问/`e`）、L1/TEX 43–65%→22–24%、shared-load wavefront
  总量 1.08M→315K（证明 ~3.2× 是纯重放），**但 `wait` 纹丝不动（1.79→1.77 等）**——`wait` 是 mma 固定
  流水延迟、与 smem 冲突无关；`Eligible Warps/Scheduler` 仍 0.15–0.26。**结论（修正 §19.5）**：
  bank conflict 是 **~1.2× 分项、不是量级项**；量级差（xFLA 4.9–8×）来自**并行度/mma 密度**（tensor
  2.3% vs FLA 14%）。目标卡 CUDA B1 优先级改为：**(1) 先补并行度/warp-specialization（量级项）、
    (2) bank conflict 用零成本 XOR swizzle/`ldmatrix` 而非 padding（padding 的 +29% smem 会反噬占用率）、
    (3) 预取保底**。
- [x] **P7-18** 在动「K2 形状重写」之前，用**最便宜的受控实验**检验 §20.6 第一优先级「并行度/mma 密度
    是量级项」里有多少是**免费的**：同一份 `b1_cuda_probe.cu` 上加两个免费旋钮——(A) warp 内 mma ILP
    （编译宏 `B1_ILP1`/`B1_ILP2`，把 op1/op2·op3 的 `for m: for kk` 换成 `for kk: for m` 暴露独立累加链，
    逐位不变）、(B) CTA 形状（`B1_BV`/`B1_NW`，`BW=16` 约束下扫 `BV=16/32/64`）。新增
    `harness/check_bwd_b1_ilp_cuda.py`（`harness/check_bwd_b1_ilp_cuda.out.txt`）+
    `harness/probe_b1_cuda_ilp_ncu.sh`（`harness/probe_b1_cuda_ilp_ncu.out.txt`）+ `docs/07` §21。
    **Part A**：六份 build 与 base **逐位相同（max_abs=0，含非整除 T=250）**。**Part B**：`ilp1` 完全无效
    （1.00–1.02×）、`ilp2` 稳定快 **6–12%（0.88–0.94×）**、叠加无额外收益；`bv16`/`bv64` 反慢
    （占用率被 smem 钉死，多塞 warp/grid 换不到可用 warp）。**Part D**：`ilp2` REG 168 仍**零 spill**。
    **`ncu`**：加 16 条独立 mma 链后 `wait` 仅 1.82→1.76、tensor 3.5%→3.9%（FLA 14%）、
    `Eligible Warps/Scheduler` 仍 **0.16**——**每调度器只挂 ~1 个 active warp，warp 内 ILP 填不满**。
    **结论（修正 §20.6 措辞）**：免费杠杆（bank conflict ~1.2× + 指令级 ILP ~1.1×）都只是分项；量级项的
    落点是**提高每 warp 的有效 mma 密度**（`ldmatrix`/TMA 整块喂数 + 流水，把 2.3% 抬到 FLA 的 14%），
    **不是**简单加 warp 数或加指令级并行。见 `docs/07` §21。

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
- workspace 字段是 head-major `[H, total_tiles, CHUNK, D]`；展平到 FLA 的 token-major
  `[B,T,H,D]` 必须 `permute(0,1,3,2,4)`（按 `(NT,CHUNK,H)`），直接 reshape 会交错 head/chunk 轴
  （指纹：`qg/kg` 相对误差≈1.0）。`INV` 即 $P[o_t,o_A]$，不需转置。

## 当前进度 / 下一步

- 当前进度：**P0–P8 全部完成 + P6-3/P6-4/P6-5/P6-6/P6-7/P6-8/P7-3/P7-4/P7-5/P7-6/P7-7/P7-8/P7-9/P7-10/P7-11/P7-12/P7-13/P7-14/P7-15/P7-16/P7-17/P7-18 补测**。
  - P7-18（本轮）：在动「K2 形状重写」前，用最便宜的受控实验检验 §20.6 第一优先级「并行度/mma 密度是
    量级项」里有多少**免费**。同一份 `b1_cuda_probe.cu` 加两个免费旋钮：warp 内 mma ILP（`B1_ILP1`/
    `B1_ILP2`，op 循环 `for m: for kk`→`for kk: for m` 暴露独立累加链，**逐位不变**）与 CTA 形状
    （`B1_BV`/`B1_NW`）。新增 `harness/check_bwd_b1_ilp_cuda.py`（`..._ilp_cuda.out.txt`）+
    `harness/probe_b1_cuda_ilp_ncu.sh`（`..._ilp_ncu.out.txt`）+ `docs/07` §21。**Part A** 六 build
    逐位相同；**Part B** `ilp1` 无效、`ilp2` **快 6–12%**、`bv16/bv64` 反慢；**Part D** `ilp2` REG168
    零 spill；**ncu** 16 条独立链后 `wait` 1.82→1.76、tensor 3.5→3.9%（FLA 14%）、`Eligible W/S` 仍
    **0.16**（每调度器只挂 ~1 warp，ILP 填不满）。**结论**：免费杠杆都是分项（conflict ~1.2× + ILP
    ~1.1×）；量级项落点＝**提高每 warp 的有效 mma 密度**（`ldmatrix`/TMA + 流水，2.3%→14%），
    不是加 warp 数/加指令级并行——修正 §20.6 措辞。
  - P7-17：用最便宜的受控实验（smem 行 stride 各 +8 bf16，宏 `B1_PAD`，退化版 swizzle）隔离
    CUDA B1 的 bank-conflict 项，**修正 §19.5**。新增 `harness/check_bwd_b1_pad.py`
    （`harness/check_bwd_b1_pad.out.txt`）+ `harness/probe_b1_cuda_pad_ncu.sh`
    （`harness/probe_b1_cuda_pad_ncu.out.txt`）+ `docs/07` §20。**Part A**：pad 与 nopad、与 FLA `dhu`
    **逐位相同**（max_abs=0）。**Part B**：pad 快 10–23%（BT64 小 shape CUPTI 105.0→84.5），**除
    B8T2048H8 BT=64 反慢 1.21×**（smem 111→143.6 KB，2→1 CTA/SM）。**Part D（ncu）**：bank-conflict
    76%→19–25%、L1/TEX 65%→23%、shared-load wavefront 1.08M→315K，**但 `wait` 不变（1.79→1.77）**。
    **结论**：bank conflict 是 ~1.2× 分项、非量级项；量级差来自并行度/mma 密度。目标卡 CUDA B1 优先级 =
    (1) 补并行度/warp-specialization、(2) 零成本 XOR swizzle/`ldmatrix` 消冲突（不用 padding）、(3) 预取保底。
  - P7-16：`ncu` 剖析 P7-15 的手写 CUDA B1，把 §18.4 归因量化并**修正**：新增
    `harness/ncu_b1_cuda_driver.py`（参数化 `--which/--bt/--ns/--no-build`）+ `harness/probe_b1_cuda_ncu.sh`
    （`harness/probe_b1_cuda_ncu.out.txt`）+ `docs/07` §19。7 配置（ours BT64 NS=1/3、`-DB1_SYNC`、BT16、
    大 shape + FLA `dhu` 对照）。**结论**：ours 的 `cp.async` 已把 `long_sb` 压到 0.66/0.57（换取同步写 = 3.96、
    慢 64%），**非 load-bound**；`barrier` 仅 0.02（**§18.4 猜的「同步」更正**）、occupancy 与 FLA 同为
    **6.25%** 却慢 **4.9×**；真瓶颈 = **`wait`（mma 固定延迟 1.80≈36%）+ `short_sb`（smem 取数）+ 76% smem
    bank conflict**（FLA ldmatrix 0.8%），叠加 FLA tensor 14.1% vs ours 2.28%。**设计输入修正**：目标卡
    CUDA B1 第一优先级＝换掉 wmma-from-smem（ldmatrix/TMA 整块 + 提 mma 密度），预取只是保底。**坑**：
    `nvcc` 在 ncu 下挂起，须 ncu 外预编译 + `--no-build`；ours 路径勿碰 FLA/Triton（setup 撑爆 profile）。
  - P7-15：新增 `docs/07` §18 + 手写纯 CUDA B1 原型 `harness/b1_cuda_probe.cu` +
    `harness/check_bwd_b1_cuda.py`（`harness/check_bwd_b1_cuda.out.txt`）+ `harness/ncu_b1_cuda_driver.py`。
    补 §17.4 的边界（手写 CUDA 显式化预取后是否还需拆链）。**Part A**：与 FLA `dhu` **逐位相同**
    （`dh/dv2/dh0` max_abs=**0**，BT=32/64、含非整除 T=250）。**Part B（关键）**：本原型绝对速度差 FLA
    `dhu` **4.3–9.6×**，`NSTAGE` 预取只值 **2–6%**（小 shape 0.92–0.98×），**大 shape 反而 1.40–1.45×
    变慢**（smem 168–226KB → 占用 2→1 CTA/SM）。**结论**：裸 wmma 版根本没到 load-bound（每 CTA 2 warp、
    每 SM 2 warp、3 GEMM/chunk 仅 ~1.9 TFLOP/s≈峰值 0.2%），故 §17 的「流水杠杆」**不是可单独加上的
    开关**——只在 kernel 已是 K2 形状（warp-specialized + TMA 大块搬运 + 足够占用）时才兑现；「拆链」在
    CUDA 里无从单独消融（warp 内状态本就是 8 条独立 mma m-tile）。smem 足迹 `BT=16/32/64`、`NS=1` =
    58.9/76.3/111.1 KB。两坑见 `docs/07` §18.5。**待验证**：本原型是研究用最小实现，`xFLA` 绝对数字
    不代表目标卡天花板，只可下「裸 wmma 版非 load-bound」的定性结论。
  - P7-14：新增 `docs/07` §17 + `harness/check_bwd_b1_ilp.py`（`harness/check_bwd_b1_ilp.out.txt`）+
    `harness/probe_bwd_ilp_ncu.sh` / `ncu_bwd_driver.py --variant`（`harness/probe_bwd_ilp_ncu.out.txt`，
    空闲卡）。**在动 CUDA 重写前，先用 bf16 Triton B1 原型量清 §16.5 两条设计输入各值多少**。四变体：base
    （单链 ns=1）/ `tl.range` 流水 / 拆 K 两条链 / 两者叠加。**Part A**：四变体 vs FLA `dhu` 数值同阶
    （含非整除 T=250），`num_stages` 不改结果、拆链把 `dv2/dh0` 精度略降（`1.221e-4/7.451e-9`）。**Part B
    （关键）**：`B2T1024H8`@BT=64 内核-only base 45.2µs（1.96×FLA）→ 流水 ns=3 **23.0µs（1.00×FLA，追平）**、
    `B8T2048H8` **0.85×（反超）**、`B1T512H4` 0.97×；**拆链单独 51.1µs（比 base 慢 13%）、与流水叠加
    25.6µs（比纯流水慢）**；部署粒度 BT=16 流水亦 1.4–2.0×（`B2T1024H8` 99→50.9）。**`ncu` 佐证**：流水把
    `long_sb` 5.27→1.46→**0.96（≈FLA 1.06）**、`tensor%` 6.8→15.9（FLA 14.9）、DRAM 13.5→30.6%（FLA 28.6%）；
    smem 65.5→**181.3KB（≈FLA 180.2）**、理论占用率 12.5→**6.25%** 却最快（再证 §16.2 非占用率受限）。
    **结论**：**§16.5 第 (2) 条「拆 2 条累加链」在 Triton 自动流水下是冗余的**（流水已填满 ILP，拆链单独/
    叠加都更慢），降级为「不推荐」；**第 (1) 条「镜像 K2 的 TMA 输入流水」是全部收益来源**——单靠它就把
    B1 从 1.96× 拉到 1.00×/0.85× FLA。故目标卡 CUDA B1 的**第一优先级就是照抄 K2 流水**，拆链暂缓。
    边界（待验证）：手写 CUDA/TMA 把预取显式化后是否仍需手工拆链增 ILP，须在目标卡再测。
  - P7-13：新增 `docs/07` §16 + `harness/ncu_bwd_driver.py` + `harness/probe_bwd_ncu.sh`
    （`harness/probe_bwd_ncu.out.txt`，空闲卡）。补上「**反向只计过时、没量过硬件 bound**」这块：用 `ncu`
    剖析反向最大单 kernel **B1**（状态伴随 / FLA `dhu`）的三种形态。**结论一**：B1 不 bound 在任何硬件单元
    （ours tensor 4.5–6.8%、FLA 14.9%；DRAM ≤28.6%、L2 ≤39.8%），achieved occ 一律 ~6.2% 是**小 grid
    （64/16 block ≪ 132 SM）**的症状、非旋钮；本质**延迟/并行度受限**（`ncu` 的 occupancy/「grid too small」
    建议是启发式误报，同 §8.4）。**结论二**：同 BT=64 ours 慢 FLA 2.14×（53.8 vs 25.2µs），差在 ours 的
    `long_scoreboard` 3.88–5.29/8.19–9.68（≈47–55% issue 间隔等全局加载）vs FLA 1.06/5.06——ours 是
    **单条 `[128,BV]` 依赖链 + `num_stages=1` 无流水**，FLA 拆 K 为两条 64 链（`chunk_delta_h.py:401-415/585-604`）
    + `num_stages∈{2,3,4}` 流水（`:350-352`，180KB smem/block、theo occ 6.25% 仍更快）。**结论三**：ours
    BT16→64 快 1.81× = 串行步数 ×4 + `dh` 落盘 DRAM 写 ×6（20.8→3.43MB，L2 ×1.68）。**设计输入**：
    目标卡 CUDA B1 优先 (1) 镜像 K2 的 TMA 输入流水、(2) D=128 状态拆 2 条独立累加链、(3) 不为小 shape
    拆 BV、(4) 维持 16 块链。这把 §11.5/§14.5/§15.5「下一步=镜像 K2 warp specialization」的**动机量化**了。P0/P1/P2/P3 完成，P4 前向三方对拍完成；P5 反向推导/显式与分块参考/三方核验完成；P6 FLA Triton 反向 5 kernel 精读 + B1 状态伴随最小 kernel（P6-2）+ 非整除 T 边界补测（P6-3）+ B2 `wy_dqkg` 最小 kernel（P6-4）+ B3 `intra` 最小 kernel（P6-5）+ **B0 门控重算最小 kernel（P6-6）** + **P6-7 补齐 `dAv` 并把 B0/B1/B2/B3/dAv 组装成端到端反向核对** + **P6-8 把整条反向直接架在 FlashKDA K1 workspace 上端到端跑通**完成，至此 §5.2 的 **B0/B1/B2/B3 四段全部有最小可核验 kernel，四段能组装成完整反向链，且该链已直接吃 FlashKDA 自己的 workspace**；P7 前向性能 + ncu bound 剖析 + K1/K2 访存字节拆分（P7-3）+ K1 标量 global load 归因（P7-4）+ K2 占用率实验（P7-5）+ **FLA host 税组成归因（P7-6）** + **反向性能与 host 税剖析（P7-7）**完成；**P8-1 完成**（`docs/08-summary.md`：一页纸汇总，见下）。
  - P7-12：新增 `docs/07` §15 + `harness/check_bwd_ws_reuse64.py`
    （`harness/check_bwd_ws_reuse64.out.txt` + `..._timing.out.txt`，空闲卡）。**闭合 §13.5 (ii)**：把整条反向
    `dAv→B1→B2→B3→反向 cumsum` **统一到 64 块**、仍只吃 FlashKDA 16-chunk K1 workspace。在 P7-11 的
    `qg/kg/w/g_cum/P64` 外，从同批 16 块字段补齐 `h64=h16[0::4]`（**状态是物理量、与分块粒度无关**）、
    `Aqk64`（对角=落盘 `Mqk`，非对角 `qd16_i(kr_j2^{e})^T`，指数≤0 折进 `kr`）、
    `U64=P64 diag(β)(v64−kd64 h64)`（`kd64=kd16·2^{off}`）。非整除 T 全部零填充到 `Tp=NG·64` 并按
    `T=Tp` 跑链（pad token 惰性），golden 吃同一 pad 输入。**Part A** 七项前向量 vs FLA forward@64 落
    bf16 底噪（`qg` 8.3e-3/`kg` 7.0e-3/`w` 8.7e-3/`g_cum` 2.6e-3/`U64` 1.8e-2/`P64` 3.0e-2/`h` 6.9e-3，
    含非整除 T=250/513、g=0；`Aqk` 因 FLA 未初始化三角 + 深衰减溢出不入表，靠整链核验）。**Part B** 整链@64
    vs `torch.autograd` 六项 `dq 9.8e-3/dk 7.9e-3/dv 1.3e-2/dbeta 6.0e-3/dg 7.7e-3/dh0 8.3e-3`（与 P6-8
    的 16 链同阶）。**Part C（关键结论）**：整链@64 反向 **dev 反而慢 2.0–2.5×**（T512H4 119.7→240.0 等）——
    逐 kernel 归因：`_dhu`（B1）@64 **快 2.1–2.2×**，但 `wy_dqkg`（B2）@64 慢 1.8–2.9×、`intra`（B3）@64
    慢 **4.8–5.5×**（块内 intra 工作 $\propto BT^2\cdot NT\propto T\cdot BT$），B1 的省被整个吃掉。**答案：
    §13.5 的两个选项都不合算**（「B1@64 吐子块 `dS`」在 FLOPs 上等价于再跑一遍 B1@16；「B2/B3 一并重组到
    64」净亏）→ **反向维持 16 块链（P6-8/P7-9），B1@64 仅作孤立状态伴随加速**；但本节的 64 前向量重建
    （§15.1）已把 §13/§14「64 操作数免重算恢复」从 B1 扩展到**全部**前向量。四个坑：FLA `initial_state`
    布局 `[K,V]` vs harness 的 $S^\top$=`[V,K]`（误用则 `h[0]` 互为转置）、门控 ln2 单位（多乘一次指纹=整体
    ×1.4427）、非整除 T 的 `gate_tok[:, :NT16]` 截尾、FLA `Aqk` 原始缓冲不可直接对拍。
  - P7-11：新增 `docs/07` §14 + `harness/check_bwd_rechunk_pass.py`
    （`harness/check_bwd_rechunk_pass.out.txt` + `harness/check_bwd_rechunk_pass_tf32.out.txt`，空闲卡）。
    把 P7-10 的 16→64 恢复融成**单个 Triton kernel**：grid `(NG,B·H)`，每 program 一个 64-group 的一个
    head。**核心洞察**：$L64$ 的对角 16×16 块就是 K1 已求逆的 $L16$（workspace `INV`=$P16$），所以
    $P64$ 用 **4×4 块前代**从 `P16`+6 个非对角块直接拼出、**零新求逆**（同 FLA
    `merge_16x16_to_64x64_inverse_kernel`）；非对角 $A_{ij}=kd16_i(kr_j2^{e_{ij}})^\top$（$e_{ij}\le0$
    折进更小的 `kr`，绕开 §13.4 坑 (1) 的 fp32 溢出），$w64=P64\,\mathrm{diag}(\beta_t)kd64$。**Part A**：
    `qg/g_cum` vs torch 原型 1e-7、`kg` 8e-6、`w` 1.6e-2（=**bf16 落盘 `P16`** 的精度上限，同 §13/P7-10），
    vs FLA fwd@64 落 bf16 底噪。**Part C（端到端）**：pass 操作数 → P7-9 的 bf16 B1@64 vs FLA `dhu`@64
    `dh≤1.44e-2/dv2≤1.87e-3/dh0≤1.20e-2`（含非整除 T=250/513、g=0），与 P7-10 在 torch 原型上同阶。
    **Part B**：kernel-only dev **41–953µs（ieee）/ 16–492µs（tf32）**，为 torch 原型 full
    （2147–5628µs）的 **0.11–0.25×**（4–9× 更快）；`pass wall`（299–1420µs）另含 harness 侧
    gather/`permute`/`b0_gate`，真实集成融进 K1/B0，故以 `dev` 为口径。**坑**：(a) 输出 pad 行的
    `g_cum` 必须**不 mask**（B1 取 padding 后的 chunk 末 token 算 `etot`），且输出须 `torch.zeros`——
    否则 pad 行未初始化，Part C 读到 NaN（指纹：仅非整除 T=250 的 `dh/dv2/dh0` 全 NaN）；(b) K1 门控的
    carry 只能补到 `K1` 的 tile 末 `NT·16`，**再往外**的重组 pad 子块门控和为 0（补过头会让 `kg` 整个
    尾块错，指纹同 §13.4 但出现在尾块本身）。**局限**：`w` 精度被 bf16 `P16` 顶在 1.6e-2；整链 B2/B3
    仍 16 块（§13.5 ii 未解）。
  - P7-10：新增 `docs/07` §13 + `harness/check_bwd_b1_rechunk.py`（`harness/check_bwd_b1_rechunk.out.txt`，空闲卡）。
    兑现 §11.5 第 (2) 步「B1 更大 chunk 重组」：**从 16-chunk K1 workspace 免前向重算地恢复 64-chunk 操作数**。
    **代数**：分块递推对任意 chunk 精确 ⇒ 64 块量只是 16 块的**逐子块重标定**——`q_dec64=qd·2^{off_s}`、
    `k_rest64=kr·2^{G64−off_s−G16_s}`（指数≤0、有界）、`ki64=ki16·2^{−off_s}`、`E_tot64=Π E16`；
    **唯一新算**是 64 块内逆 `P64=(I+tril(kd64@ki64^T,−1)β)^{-1}` 与 `w64=P64 diag(β) kd64`。
    **Part A**：`qg/kg/w/g_cum` vs FLA forward@64 全落 bf16 底噪（1e-3~1.7e-2，含非整除 T=250/513、g=0）。
    **Part B**：B1@64（bf16 核，token 轴补零 mask）vs FLA `dhu`@64，`dh≤1.55e-2/dv2≤1.87e-3/dh0≤1.44e-2`。
    **Part C**：**B1 快 2.1–2.3×**（T512H4 45.5→19.5、B2T1024H8 122.2→54.8、B8T2048H8 393.2→187.4、
    T250H3 23.4→10.8 µs；相对 FLA 仍 1.5–2.4× 常数因子）。**硬伤**：torch 原型恢复 ~2.0–5.7 ms（纯重标定
    0.47–1.48 ms + `A64/P64/w64` 2.0–3.7 ms），远大于 B1，须融成 K1 式 Triton pass（数字为高估上界、待验证）；
    且整链不干净（B2/B3 仍 16、B1@64 只落 64 边界 `dh`）。**坑**：(a) `ki16=kr·2^{−G16}` 溢 fp32（须逐子块
    对把偏移折进较小因子做 `A64`）；(b) 非整除 T 零填充丢「累积进位」污染尾块前一子块（指纹 `kg` 尾块前 4
    token `max_rel≈0.83`），须先补携带值再差分；(c) workspace 字段 `[B,NT,H,CHUNK,D]` 展平须 `permute(0,1,3,2,4)`。
    **观察**：`w64`≠重排 `w16`（`max_rel≈1.0`），强衰减下 `E64` 小到 `3e-86`，64 块被末子块主导。
  - P7-9：新增 `docs/07` §12 + `harness/check_bwd_b1_bf16_e2e.py`（`harness/check_bwd_b1_bf16_e2e.out.txt`）。
    把 P7-8 §11.4 的「B1 吃 workspace 原生 bf16」从**孤立 kernel 的手工实验**落进 P6-8 的**真实组装链**
    （此前 `reconstruct` 把整套 workspace `.float()`，B1 一直跑 tf32）。只换 B1 的 mma 操作数 dtype
    （状态 `b_dh` 仍 fp32 累加、`dh`/`dv2` 输出**保持 fp32**——因为 B2 `wy_dqkg` 把 `dv` 与 fp32 `v`
    放进同一 `tl.dot`、`dh` 与 fp32 `h` 进同一 `tl.sum`，落 bf16 会直接报 dtype mismatch；加速来自
    mma 操作数、与输出 dtype 无关）。**Part A 正确性**（P6-8 同 6 组 case，含非整除 T=250/513、g=0、
    B=2）：bf16-B1 链 vs `torch.autograd`(FLA) 六项 `dq 3.42e-3/dk 3.02e-3/dv 6.80e-3/dbeta 6.01e-3/
    dg 7.06e-3/dh0 8.27e-3`，与 fp32-B1 基线**同阶**；操作数换 bf16 自身增量 `dq` **逐位 0**、其余
    `dk≤1.03e-3/dv≤3.76e-3/dbeta≤2.30e-3/dg≤1.08e-3`，B1 的 `dh` bf16-vs-fp32 跨 case 6.4e-4–3.9e-3。
    **Part B 性能**（空闲卡、wall+dev 同 P7-8 口径）：B1 kernel 快 **1.68–2.09×**（T512H4 101.5→53.2、
    B2T1024H8 208.2→118.8、B8T2048H8 651.9→387.0、T250H3 46.9→22.4；B2T1024H8 的 118.8 已快过空闲卡
    FLA `dhu` 137.4），**整链 dev 降到 fp32-B1 链的 0.71–0.91×、FLA 反向的 0.51–0.60×**。**坑**：
    小 shape（T512H4）wall 反而略高（320 vs 306），是 harness 每个 timed call 对 fp32 `dv` 做一次
    `.to(bf16)` 的 host 税被放大（小 shape 整链 wall 本就 host 主导，P7-6），真实实现里 `do/dv` 与
    workspace 同为 bf16、无需运行期转换 → 以 dev 为口径。至此 §11.5 优化优先级第 (1) 步完成，
    下一步是第 (2) 步「更大 chunk 重组消粒度损失」。
  - P6-7：新增 `docs/06` §6.9 + `harness/check_bwd_e2e.py`（`harness/check_bwd_e2e.out.txt`），
    并给 `harness/check_bwd_dhu_kernel.py` 的 B1 kernel 补 `dht`（终态上游梯度）入口（`HAS_DHT` 分支）。
    **Part A**：补齐反向第 5 个 kernel **`dAv`（输出支路）**（对应 FLA `chunk_kda_bwd_kernel_dAv`，
    `chunk_bwd.py:44-114`），`dAqk=do·U^⊤·scale`、`dv=M_qk·do`；4 组配置（T=256/512/1024/250）
    下与 FLA **逐位相同（max_abs=0）**。**Part B（关键）**：把 `dAv→B1(dhu)→B2(wy_dqkg)→B3(intra)
    →反向 cumsum(B0)` 串成完整反向，与 `torch.autograd` 穿 `chunk_kda` 对拍（P5-3 同款 setup：全
    `use_*_in_kernel=False`、同一份 fp32 操作数、损失含 `final_state` 上游 → `dht≠0`）：六项梯度
    **全落 tf32 底噪**——`dq` 逐位、`dk≤1.51e-5`、`dv≤3.90e-4`、`dg≤1.10e-5`、`dbeta≤1.09e-4`、
    `dh0≤5.06e-6`，**含非整除 T=250**。**Part C**：FLA 自己的子 kernel 按 `chunk_kda_bwd` 顺序手串
    与 autograd **逐位相同**（证明 golden=分解）。**Part D（swap）**：把 FLA 的 B1/dAv/dAkk 输出喂
    我们的 B2/B3 再比 FLA 分解，`dk`/反 cumsum `dg` 到 **8.4e-8~1.2e-7** → 证明 B2/B3 在相同输入下
    与 FLA 代数一致，Part B 残差来自 B1 独立 tf32 舍入经衰减链放大（如 T=512：dh 9.9e-6、dv2 8.3e-5
    → dk 1.5e-5、dg 1.1e-5）。**两个组装才暴露的坑**：(a) **B2 要吃 `v_new`（`fwd_h` 输出）而非 `u`
    （`fwd_intra` 输出）**，误用指纹=「dq 逐位、dk/dg 随 T 大面积错」；(b) **反向 cumsum 不要额外乘 ln2**
    （FLA `chunk_local_cumsum` 默认 `scale=RCP_LN2`，其 reverse 已直接给 `ga.grad`），误乘指纹=
    「dg 与其余五项差一个均匀常数因子」（线性拟合 `gold/ours≈1.44=RCP_LN2`、max_rel 稳停 0.44）。
     回归：B1 自身自测（`check_bwd_dhu_kernel.py`）改后仍与旧数字一致（dh 4.6e-5 等），无退化。
  - P7-8：新增 `docs/07` §11「复核 §10：GPU 竞争污染与两个混淆变量」+ `harness/check_bwd_dhu_dtype.py`
    （`harness/check_bwd_dhu_dtype.out.txt`）+ `harness/bench_bwd_idle.out.txt`（`bench_bwd.py` 钉到
    空闲卡 `CUDA_VISIBLE_DEVICES=1` 重跑）。**动机**：P7-7 的两个关键数字事后不可复现——ours B1
    @B2T1024H8 报 539.8µs（→「慢 FLA 3.9×」）、FLA 前向重算 `fwd_h` 报 578.0µs（→「重算占 FLA
    反向 dev 46.2%」）；重测机器上 GPU 0 正被另一容器打到 100%。**空闲卡重跑**：主表 ours dev 是 FLA
    反向的 0.61–0.80×（比值同 §10），但绝对值低约 2×（B2T1024H8 ours dev 972.8→483.0、FLA
    1322.8→652.3）；逐 kernel 只有两行对不上且都是「带串行递推的大头」：**ours B1 539.8→214.3**、
    **FLA `fwd_h` 578.0→83.6**（其 FLA `dhu` 136.9→137.4 几乎没变）。重算占比更正为
    (83.6+32.8)/652.3 = **17.8%**（T512H4 为 21.9%）。**控制变量**（`check_bwd_dhu_dtype.py`）：造一个
    只改 `tl.dot` 操作数 dtype 的 bf16-mma 克隆 kernel（状态仍 fp32 累加，仅点积内 `.to(bf16)`，与
    FLA 逐字一致），同操作数扫 {ours,FLA}×{fp32/tf32,bf16}×{BT=32,64}（CUPTI，空闲卡）：**同 BT 同
    dtype 下 ours/FLA = 1.13–1.95×**（fp32 1.13–1.65×、bf16 1.24–1.95×），即 B1 与 FLA `dhu` 只差
    一个常数因子；**chunk 粒度**（BT=16 vs 64）损失约 1.3–1.6×。**最便宜杠杆**：B1 操作数保持
    workspace 原生 bf16（P6-8 的 `reconstruct` 把整套 `.float()` 了，使 B1 掉进最慢的 tf32 路径）→
    T512H4 91.3→**37.2µs（2.45×）**、B2T1024H8 215.0→**97.9µs（2.20×，已快过 FLA `dhu` 137.4）**；
    Part A 正确性 ours-bf16 vs FLA-bf16（同 BT=64）`dh/dv2` max_abs 6.1e-5–2.4e-4、`dh0` 7.5e-9–1.7e-7。
    **更正**：B1 差距性质从「实现差 4×」变为「chunk 粒度 + 常数因子」；优化优先级重排为
    (1) 吃原生 bf16 → (2) 更大 chunk 重组 → (3) 最后才镜像 K2 warp specialization。**方法学修正**：
    本专题基准此后统一 `CUDA_VISIBLE_DEVICES=<空闲卡>`。
  - P6-8：新增 `docs/06` §6.10 + `harness/check_bwd_ws_reuse.py`（`harness/check_bwd_ws_reuse.out.txt`）。
    把整条反向**直接架在 FlashKDA 自己的 K1 workspace 上**（此前 P6-2～P6-7 都吃 FLA 中间量）：
    按 head-major SoA 偏移（`fwd_launch.cu:64-71`，`n_ht=H·total_tiles`）切出六字段 + B0 重算，
    逐块重建 $S_{in}/U/w/h/Aqk/Akk$，跑 `dAv→B1→B2→B3→反向 cumsum(B0)`。**Part A**（workspace
    重算前向 `o=q_dec S_in+M_qk U`、`S_out=E_tot S_in+k_rest^T U` vs FlashKDA 自身 `out`/`fin`）
    `o` max_rel 4.5e-3~8.9e-3、终态 4.5e-3~8.0e-3；**Part B**（七个操作数 vs `kda_ref` 独立重算，fp32，
    CHUNK=16）1.3e-4~8.1e-3；**Part C**（六项梯度 vs `torch.autograd`，P6-7 setup 同一份激活操作数）
    最坏 `dq 3.42e-3 / dk 2.98e-3 / dv 6.34e-3 / dbeta 5.97e-3 / dg 7.05e-3 / dh0 6.42e-3`——全落
    bf16 输入 / tf32 `tl.dot` 底噪，6 组 case 含**非整除 T=250/513、g=0、B=2**。至此「复用 K1 workspace」
    的设计前提**端到端成立**（不再只是 §6.8 Part E 的 `kGTotal` 点对点）。**四个只有架到真 workspace
    才暴露的坑**：(a) 字段是 `[H,total_tiles,CHUNK,D]` head-major，展平到 FLA `[B,T,H,D]` 必须
    `permute(0,1,3,2,4)` 按 `(NT,CHUNK,H)` 重排，错 reshape 指纹=「qg/kg rel≈1.0」；(b) `INV` 就是
    $P[o_t,o_A]$（与 `Akk` 同布局）**不需转置**；(c) golden 必须吃**激活后**操作数（P6-7 的
    `use_*_in_kernel=False`），否则与我们的激活单位梯度差一个 l2norm/sigmoid Jacobian（指纹=`dq/dbeta`
    大面积 0.5+）；(d) 非整除 T 按 K1 补零尾块（`v/β/g` pad 到 `NT·CHUNK` 再切回 T），配合 §6.5 的
    `E_tot` 末有效 token 取法，非整除与整除同精度。

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

- **下一步：专题 P0–P8 全部完成 + P6-3/P6-4/P6-5/P6-6/P6-7/P6-8/P7-3/P7-4/P7-5/P7-6/P7-7/P7-8/P7-9/P7-10/P7-11/P7-12/P7-13/P7-14/P7-15/P7-16/P7-17/P7-18 补测**，进入收尾/维护。后续可选（有明确需求时再做）：
  - **下一步工程化＝把 B1 做成目标卡 CUDA**（唯一剩下的实际工作）。P7-13（`docs/07` §16）已用 `ncu` 把它的
    硬件动机量化清楚：B1 是**延迟/并行度受限**（非 tensor/带宽/占用率），与 FLA `dhu` 的 2.14×（同 BT=64）
    差距＝**没有 loads 流水（long_scoreboard 47–55%）+ 单一 `[128,BV]` 依赖链**，FLA 用「K 拆两条 64 链 +
    `num_stages` 流水」补上。**P7-14（`docs/07` §17）在 bf16 Triton 原型上把这两条设计输入分开量了**：
    单靠**流水**就把 B1 从 1.96× 拉到 **1.00×/0.85× FLA**（`long_sb` 5.27→0.96≈FLA），而**拆链单独负收益、
    叠加更慢**——故 CUDA 版优先级修正为：**(1) 镜像前向 K2 的 TMA 输入流水（`docs/03` §8/§10）是唯一必需项 →
    (2) 不为小 shape 拆 BV → (3) 维持 16 块链**；**「拆 2 条累加链」降级为不推荐**（Triton 自动流水已填满
    ILP）。这也是 `docs/06` §5.2 早写下的设计，现在有了硬件证据 + 原型验证。**P7-15（`docs/07` §18）用
    手写纯 CUDA B1 补了「显式预取后是否还需拆链」的边界**：裸 wmma 版与 FLA `dhu` **逐位相同**（含非整除
    T），但**根本没到 load-bound**（每 CTA 2 warp、~108KB smem，3 GEMM/chunk ~1.9 TFLOP/s），故 `cp.async`
    预取只值 2–6%、大 shape 反被占用率损失反噬（1.40–1.45×）——**结论：流水杠杆只在 kernel 已是 K2 形状
    （warp-specialized + TMA 大块搬运 + 足够占用）时才兑现；裸 wmma 版要先把结构做到 K2**。手写 CUDA/TMA
    显式化预取后是否仍需手工拆链增 ILP，仍须在目标卡上、且以 K2 结构为基座再测（待验证）。**P7-16（`docs/07`
    §19）用 `ncu` 剖析这支 CUDA B1，把上面的归因量化并修正**：裸 wmma 版真正卡在 **`wait`（mma 固定延迟
    1.80≈36% 发射间隔）+ `short_sb`（smem 取数）+ 76% shared-load bank conflict**，而**不是** global 加载
    （`cp.async` 已把 `long_sb` 压到 0.66；换同步写则 3.96、慢 64%）、**不是**同步（`barrier` 仅 0.02）、
    **不是**占用率（与 FLA 同为 6.25% theo 却慢 4.9×）。故目标卡 CUDA B1 的**第一优先级＝换掉
    wmma-from-smem（ldmatrix/TMA 整块 + 提 mma 密度，消 bank conflict）**，`cp.async`/TMA 预取只是「不退化」
    保底；「拆链」仍不作为独立手段。**P7-17（`docs/07` §20）用 smem padding 受控实验修正了上条的排序**：
    把 bank conflict 从 76% 打到 19%（数值逐位不变）端到端**只快 10–23%**（大 shape BT=64 因 smem +29%
    掉占用率反慢 1.21×），且 ncu 显示 **`wait` 纹丝不动**——bank conflict 是 ~1.2× 分项、**不是量级项**；
    量级差来自**并行度/mma 密度**。故目标卡 CUDA B1 真实第一优先级＝**(1) 补并行度/warp-specialization
    （量级项）、(2) 用零成本 XOR swizzle/`ldmatrix` 消冲突（勿用 padding）、(3) 预取保底**。
    **P7-18（`docs/07` §21）在动「K2 形状重写」前，用最便宜的受控实验检验「并行度/mma 密度是量级项」里
    有多少免费**：同一份 `b1_cuda_probe.cu` 加 warp 内 mma ILP（`B1_ILP1`/`B1_ILP2`，op 循环换序暴露独立
    累加链，**逐位不变**）与 CTA 形状（`B1_BV`/`B1_NW`）。**Part A** 六 build 逐位相同；**Part B** `ilp1`
    无效、`ilp2` 快 **6–12%**、`bv16/bv64` 反慢；**ncu** 加 16 条独立链后 `wait` 仅 1.82→1.76、tensor
    3.5→3.9%（FLA 14%）、`Eligible Warps/Scheduler` 仍 **0.16**（每调度器只挂 ~1 active warp，warp 内 ILP
    填不满）。**结论（修正 §20.6 措辞）**：免费杠杆（bank conflict ~1.2× + 指令级 ILP ~1.1×）都只是分项；
    量级项的落点＝**提高每 warp 的有效 mma 密度**（`ldmatrix`/TMA 整块喂数 + 流水，2.3%→14%），**不是**
    简单加 warp 数或加指令级并行——即「K2 形状重写」的必要性被这两轮受控实验**双重确认**。
  §5.2 的 **B0/B1/B2/B3/dAv 五段已全部做成最小 kernel，P6-7 已把四段组装成端到端反向核到
  tf32 底噪，P6-8 又把整条反向直接架在 FlashKDA K1 workspace 上端到端核到 bf16 底噪**；
  **P7-7 又把它和 FLA 反向在同一口径下比（wall/dev 0.40–0.80× 反超，因复用 workspace 省掉
  FLA 反向里重算前向状态那一块）**；**P7-8 复核发现 P7-7 的表受 GPU 竞争污染（关键两行不可复现），
  并把下一阶段的工程化目标从「B1 慢 3.9× → 重写」更正为「B1 只差 FLA 1.6–1.9×，性质是 CHUNK=16
  粒度 + 常数因子；最便宜杠杆是先吃 workspace 原生 bf16（2.2–2.5×），再考虑更大 chunk 重组，
  最后才是镜像 K2 的 warp specialization」**；`dAv/B2/B3` 已领先或接近，非瓶颈。剩余的是更完整的
  CUTLASS/TMA + warp-specialization 版反向工程化（重点＝B1）。P7-2 遗留的 nsys
  流量拆分（P7-3 已用 ncu DRAM 计数替代、待验证 (b) 已关闭）。
  **P7-5 已用实验关闭 P7-2/P7-4 待验证 (c)**（K2 突破 2 CTA/SM 后 L2 墙不后移，见下）。
  **P7-6 已用 `process_time`+nsys 关闭 §4 待验证「FLA host 税组成」**（0.8 ms ≈ 12 次 Triton kernel
  的 Python launch 包装 + ~190 µs `dispatch` + ~75 µs driver launch，与 GPU/autograd/分配无关，见下）。
  **P7-7：新增 `docs/07` §10 + `harness/bench_bwd.py`（`harness/bench_bwd.out.txt`），见下。P7-8：
  更正其被污染的数字并重排 B1 优化优先级（`docs/07` §11 + `harness/bench_bwd_idle.out.txt` +
  `harness/check_bwd_dhu_dtype.py`），见上。P7-9：把 §11.4 的 bf16-B1 杠杆落进真实组装链
  （`docs/07` §12 + `harness/check_bwd_b1_bf16_e2e.py`），B1 快 1.68–2.09×、整链 dev 0.71–0.91×，
  见上；§11.5 优先级第 (1) 步完成。**P7-10：兑现第 (2) 步（`docs/07` §13 +
  `harness/check_bwd_b1_rechunk.py`）——64-chunk 操作数可从 16-chunk workspace 免重算恢复（逐子块重标定
  + 重算 64 块内逆 `P64/w64`），B1@64 正确且快 **2.1–2.3×**；但恢复须融 Triton pass、整链需 B2/B3 一并
  重组，故下一步＝把恢复融成 K1 式 pass 并决定整链是否跟到 64；第 (3) 步镜像 K2 仍排最后。**
  **P7-11：兑现 P7-10 的「融成 pass」（`docs/07` §14 + `harness/check_bwd_rechunk_pass.py`）——单个
  Triton kernel 完成重标定 + 6 个非对角 $A_{ij}$ + 复用落盘 $P16$ 的 4×4 块前代合并 + `w64`，
  kernel-only 41–953µs（ieee）/16–492µs（tf32），为 torch 原型的 0.11–0.25×；B1@64 端到端仍落 bf16
  底噪。**恢复侧的硬伤已消。** **P7-12 已闭合 §13.5 (ii)**：整条反向能统一到 64 块端到端跑通（六项梯度落
  bf16 底噪），但**实测反而慢 2.0–2.5×**（B1@64 快 2.1–2.2×，被 B2/B3@64 的 4–5× 涨盖过）→ 答案是**不要把
  整链跟到 64**，反向维持 16 块链，B1@64 仅作孤立加速手段；且 64 前向量（含 `h64/U64/Aqk64/P64`）可从 16
  workspace 免重算恢复（§15.1）。剩余的是把这些最小 kernel 工程化为目标卡 CUDA（CUTLASS/TMA +
  warp-specialization，重点＝B1）；另有 `w` 精度受 bf16 `P16` 限制（1.6e-2）。**
  - P6-6：新增 `docs/06` §6.8 + `harness/check_bwd_b0_gate.py`
    （`harness/check_bwd_b0_gate.out.txt`）。把 §5.2 的 **B0 `gate_bwd`**（对应 FLA
    `kda_gate_chunk_cumsum` `gate.py:375` + `kda_gate_bwd` `gate.py:252`）做成最小可运行 Triton
    kernel：**前向重算** β 与 g_cum（`inner=exp(A_log)(g+dt_bias)`、`gate=LB·σ(inner)`、`g_cum=
    cumsum_chunk(gate)·log2e`，与 K1 `fwd_kernel1.cuh:321-323/327-330` 同式）。**Part A/B**：g_cum
    与 torch/FLA 对到 **max_rel 1.5e-7~3.9e-7（含非整除 T250/513、g=0、B=2）**，β 与 FLA **逐位相同**。
    **Part C**：反向前缀和（`APPLY_GATE=False,REVERSE=True`）vs FLA `chunk_local_cumsum(reverse)` **1.2e-7~2.1e-7**。
    **Part D**：门控反向 `dg_logits/dA/dbias` vs **torch.autograd 2.3e-7~1.3e-6**。**Part E（关键）**：
    用 `flash_kda._fwd_raw` + 自备 workspace 读 K1 当场写下的 `kGTotal`（E^tot，SoA 偏移按
    `fwd_launch.cu:64-71`），B0 的 exp2(g_cum[last]) 对上 **max_rel 1.6e-5~8.1e-5**——差即 K1 的
    `tanh.approx` sigmoid（`fwd_kernel1.cuh:323`），**证明 B0 复现了前向 kernel 的门控**。**Part F**：
    计时 B1T512H4 27.3µs(FLA 50.4)/B4T2048H8 46.6(67.9)/B1T2048H64 91.6(64.7)，同量级。
    **踩坑**：(a) **log2e 必须乘在反向前缀和上**（g_cum=log2e·cumsum(gate)），漏掉指纹=归一化误差
    稳停 **0.3069=1−ln2**；(b) **FLA `kda_gate_bwd` 末尾 `dg.type_as(g)` 把输出转回输入 dtype(bf16)**，
    故「B0 vs FLA」表面 3e-3 实为 bf16 出口（round 到 bf16 后残差 max≤5.6e-4、mean~1e-7），B0 保 fp32 故
    与 autograd 1e-7。至此 §5.2 的 **B0/B1/B2/B3 四段全部有最小可运行 kernel**。
  - P6-5：新增 `docs/06` §6.7 + `harness/check_bwd_intra_kernel.py`
    （`harness/check_bwd_intra_kernel.out.txt`）。把 §5.2 的 **B3 `bwd_intra`**（对应 FLA
    `chunk_kda_bwd_kernel_intra`，`chunk_intra.py:395-667`，非 `SAFE_GATE` 分支）做成最小可运行
    Triton kernel（grid `(NK*NC,NT,B*HV)`、`BC=16`、`BK=32`，四阶段 A/B/C/D 按真实行号 473-660
    对拍）。**Part A**：独立纯 torch（ieee fp32）逐块参考 vs FLA Triton 内核，`dq/dk/db/dg`
    相对误差 **3.0e-4~8.8e-4**＝tf32 底噪（替身证明代数一致）。**Part B**：本 kernel vs FLA
    **`dq`/`dk` 逐位相同（max_abs=0）**、`db`/`dg` ~1e-7（`db` 跨 K-块用 `atomic_add`、加法顺序
    与 FLA 的 `[NK,...].sum(0)` 不同），**含非整除 T=250/513**。**Part C**：全 bf16 操作数 →
    四项抬到 **1.1e-2~4.8e-2**（A/B/C/D 四段把 `exp2(g-g)` 直接乘在操作数上，量化被指数放大，
    印证 `chunk_intra.py:597` 的「不要 bf16」注释；比 B2 高约一个量级）。**Part D**：朴素计时
    B1 T512 H4 D128 → 86.9µs（FLA 86.4µs）、B2 T2048 H8 D128 → 591.1µs（FLA 592.0µs），同速。
    **踩坑**：(a) 漏掉 `chunk_intra.py:558` 的 `b_dg2 = b_q*b_dq2`（衰减链的 $\bar g{+}{=}\bar q^{dec}\odot q$），
    指纹是「`dq/dk/db` 全对、只有 `dg` 大面积错（max_rel≈1.0）」；(b) Part C/D 对 `dAqk/dAkk`
    是转置读，其**行下标是 chunk-内局部量 `i_i*BC+o_i` 而非绝对 `i_ti`**，误用后**只有第 0 个
    chunk 对、其余全错**（`i_t=0` 时两者相等，故单 chunk 测试漏检）；(c) `db` 的跨 K-块归约用
    `atomic_add` 与 FLA 数学等价但非逐位（~1e-7）。
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
  - P7-6：`docs/07` 新增 §9「FLA 那 0.8 ms『host 税』到底花在哪」+ `harness/probe_nsys.sh`
    （内部 `probe_nsys.py` 四模式 `micro`/`cpu`/`profile`/`loop`）+ `harness/probe_nsys.out.txt`，
    关闭 §4 待验证「FLA host 税组成」。**主口径 `time.process_time`**（单线程下==纯 CPU，已用忙循环
    校验 `process_time≡wall`；`wall` 在队列填满后把 CPU 阻塞也算进去、`torch.profiler` CPU 活动会放大
    device 时间，二者都不适合）。**实测**：(a) FLA 前向 host CPU **固定 ~0.82–0.86 ms/次、与 T 无关**
    （T=16→821.5、2048→862.2 µs），FlashKDA 对照 ~34 µs/次（**~24×**）；(b) 分解（T=1024，dispatch ON）：
    `chunk_kda_fwd` 529 + `l2norm_fwd`×2 154（**单次 77 µs**）+ `fused_beta_sigmoid` 63 + autograd/dispatch
    杂项 ~95；`FLA_DISABLE_BACKEND_DISPATCH=1`（`fla/ops/backends/__init__.py:28`）总量掉 ~190 µs（22%），
    且 `dispatch.wrapper` **每次调用都重跑**（`__init__.py:167`）；(c) **裸 Triton/torch launch 仅 2.4/2.0 µs**、
    裸 `torch.empty(_like)` 0.07 µs ⇒ 多出的几十 µs 是 FLA 在 Triton 上叠的每调用 Python 包装
    （`fla_cache_autotune`/`CachedAutotuner`/`AutotuneKey.build`）；(d) nsys 稳态（capture-range N=200，
    用中位数避开首个调用 2.5 ms lazy 长尾）：FLA **12 kernel/次**（9 `cuLaunchKernelEx`+3 `cudaLaunchKernel`，
    driver 中位 ~6.3 µs）、FlashKDA **3**（~5.6 µs）；稳态 GPU kernel 合计 FLA ~82 µs vs FlashKDA ~109 µs
    （B1T1024H4）。**结论**：0.8 ms ≈ 12 次 Triton kernel 的 Python launch 包装 + ~190 µs `dispatch`
    + ~75 µs 真 driver launch，与 GPU 计算 / autograd / 分配无关。**踩坑**：(1) **不预热会把 Triton 首次
    JIT+autotune 的几十 ms 摊成 3843 µs/次的假 launch 税**，必须先 warmup 再计；(2) nsys 的 CUDA API
    **总量**被首个调用的 lazy 初始化长尾污染（`cuLaunchKernelEx` avg 183 µs 而 med 5.5 µs），要**看中位数**。
    同步更新 `docs/08` 待验证项。
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
  - P6-4：新增 `docs/06` §6.6 + `harness/check_bwd_dqkg_kernel.py`
    （`harness/check_bwd_dqkg_kernel.out.txt`）。把 §5.2 的 **B2 `bwd_dqkg`** 做成最小可运行 Triton
    kernel（grid `(NT,B·H)`、BK=BV=64、逐 `i_k`/`i_v` 分块、`i_k==0` 分支存 `dv2`/`db`/`dA` 第一支、
    `tl.debug_barrier()`、求逆伴随两次 bf16 `tl.dot` 照搬 FLA `chunk_bwd.py:133-310`）。**Part A**：
    独立纯 torch（ieee fp32）逐块参考 vs FLA Triton 内核，六项 `dq/dk/dv/db/dg/dA` 相对误差
    **5.8e-4~2.1e-3**＝tf32 底噪（替身证明代数一致：转置写反会是 O(1) 误差）。**Part B**：本 kernel
    vs FLA **逐位相同（max_abs=0）**，含非整除 T=250/513（`wy_dqkg` per-chunk 独立，无 B1 那样的
    host 非整除 bug）。**Part C**：全 bf16 操作数 → `dq/dv/db/dA` **2.3e-3~6.4e-3**、`dk/dg`
    **3.3e-2~5.0e-2**（门控/衰减链小量相减+指数放大，印证 §3.3「不要 bf16」注释）。**Part D**：
    朴素计时 T512H4D128 97.8µs（FLA 88.4µs）、T2048H8D128 496.8µs（FLA 452.8µs）。**踩坑**：本机
    FLA 0.5.x 把 `@dispatch('kda')` 路由到 **TileLang** 后端（`fla/ops/kda/backends/tilelang/
    chunk_bwd_dqkg.py`），其 fp32 路径编译失败（无 fp32 MMA）；对拍须 `FLA_DISABLE_BACKEND_DISPATCH=1`
    钉回文档精读的 Triton 内核。另踩坑：早先非整除测试的巨值（3.6e6）是自己 `[:, :T]` 切片造成
    **非连续**张量、破坏 FLA 假设的 batch stride，`.contiguous()` 后归位——非 FLA 的 bug。
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
