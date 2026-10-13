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
- [x] **P6-9** 关闭 §5.3 最后一条精度待验证（$P$/`Akk` 的存储精度）：
      新增 `harness/check_bwd_p_dtype.py`（`harness/check_bwd_p_dtype.out.txt`）+ `docs/06` §6.11。
      **Part A**（dtype 审计）：FLA 前向 `Akk` 开成 `k.dtype`（`chunk_intra.py:817-819`）——fp32 入
      →fp32、bf16 入→bf16，**生产时 $P$ 本就 bf16**，与目标卡 K1 的 bf16 `kINV` 一致；并更正 §3.3
      把 `302-303` 的 `b_A.to(b_A.dtype)`（恒等 no-op）误读成「强制 bf16」（`not bf16` 注释在
      `chunk_intra.py:597`，指块内衰减链而非求逆伴随）。**Part B**（孤立解析）：$-P^\top\bar P P^\top$
      对 bf16($P$) 放大不到 3×（$P$ 量化 2.1e-4 → 伴随量 4.9e-4）。**Part C**（孤立 B2）：只换
      `A=bf16(P).fp32`，六项 `dq` 逐位 0、`dk 4.9e-6/dv 4.7e-5/db 6.5e-5/dg 2.0e-6/dA 4.9e-4`。
      **Part D**（整链，四组含非整除 T=250）：$P$ 降 bf16 的**归因增量** `dq/dh0=0`、其余
      **≤1.3e-4**，比 bf16 操作数底噪（§6.6 Part C 的 3e-2~5e-2）低 2–3 个数量级。
      **结论**：**无须为 $P$ 增存 fp32 副本**（撤回 §5.3/§5.4「增存 fp32 版」建议），直接用 bf16 `kINV`。

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
- [x] **P7-19** 兑现 §21.6 的量级项落点，把手写 CUDA B1 从「wmma-from-smem」改成 **K2 形状**：
    运行态 $S$ **常驻寄存器**（不再有 fp32 smem 主副本与 fp32 staging 往返），mma 换裸
    `mma.sync.m16n8k16`、操作数用 `ldmatrix(.x4/.trans/.x2.trans)` 从 **XOR 填充后无冲突**的 smem 整块喂入；
    `scale` 折进 `do` 的 bf16 操作数、`-w^T` 用符号位 XOR，使 $S$ 更新只需**一组累加器寄存器**。新增
    `harness/b1_reg_probe.cu` + `harness/check_bwd_b1_reg.py`（`harness/check_bwd_b1_reg.out.txt`）+
    `harness/ncu_b1_reg_driver.py`/`harness/probe_b1_reg_ncu.sh`（`harness/probe_b1_reg_ncu.out.txt`）+
    `docs/07` §22。**Part A**：vs FLA `dhu` 落 bf16 底噪（`dh/dv2` max_abs=4.88e-4=1 ulp bf16、
    `dh0` max_rel~2.1e-3，含非整除 T=250；不再逐位——`scale` 折进 bf16 操作数所致）。**Part B**：
    vs 旧 CUDA 基线稳定快 **2.3–2.7×**（`xbase` 0.37–0.43），`xFLA` 从 §18 的 4.3–9.6× 收到
    **2.10–3.75×**（BT=64；BT=16 部署粒度 81.8 vs 旧 214.4µs）；`NSTAGE>1` 仍几乎无效、大 shape 反噬
    （smem）。**Part C**：smem BT64/NS1 111.1→**81.4KB**（‑27%）、BT16 58.9→**30.0KB**（‑49%）。
    **Part D（`ncu`）**：bank-conflict 小 shape **827392→0**、大 shape **53.1M→0.56M**（XOR stride）；
    tensor **2.3%→5.8%**（小）/ **3.5%→8.3%**（大，FLA 12.2%）；`wait` 1.79→**1.54**/1.82→**1.53**；
    但 `short_sb` 略升、占用率仍 6.25%、`Eligible W/S` 0.14–0.19——**残差在 `wait`/并行度/TMA 深度**。
    **结论**：K2 形状（寄存器常驻 + `ldmatrix`/`mma.sync` + XOR swizzle）兑现约 **2.4×**、把量级项吃掉
    大头但没吃干净；下一步是在其上补 **TMA + warp-specialization 深流水**（学 K2/§17），而非再加 ILP
    （§21）或 padding（§20）。**三坑**：(a) `ldmatrix.x4.trans` 子块次序与非 trans 不同（`(0,0),(0,1),
    (1,0),(1,1)` vs `(0,0),(1,0),(0,1),(1,1)`），用错指纹=`max_rel≈1.0`；(b) B 操作数必须 `.trans`
    （否则每 lane 拿相邻 N 而非相邻 K，整片错位）；(c) `scale` 折进 bf16 操作数换来单累加器、但丢逐位。
- [x] **P7-20** 兑现 §22.5 的下一步：在 K2 形状（P7-19）之上补 **TMA + warp-specialization**——
      q/k/w/dO/dV 由**专职 producer warp** 发 `cp.async.bulk.tensor`（5D 张量图，box `{64,1,1,BT,1}`，
      **128 B swizzle `Swizzle<3,4,3>`**，box 内维 ≤128 B 故把一行拆成两个 `[BT,64]` atom），
      用 **per-stage `mbarrier`**（`full`/`empty`，`try_wait.parity`）做 producer/consumer 握手，
      稳态循环**无 `__syncthreads()`**；`e` 改由 compute warp 直读 global。新增
      `harness/b1_tma_probe.cu` + `harness/check_bwd_b1_tma.py`（`harness/check_bwd_b1_tma.out.txt`）+
      `harness/ncu_b1_tma_driver.py`/`harness/probe_b1_tma_ncu.sh`（`harness/probe_b1_tma_ncu.out.txt`）+
      `docs/07` §23。**Part A**：vs FLA 落 bf16 底噪（4.88e-4/5.0e-3），且 **vs P7-19 reg 内核逐位相同
      （`dh/dv2/dh0` max_abs=0）**（含非整除 T=250，靠 TMA OOB 零填充）。**Part B**：新增同形状
      `reg64`（`BV=64/NW=4`）对照隔离 TMA+warp-spec 本身——`BT=64` 三档 **xreg64=0.81–0.85×**（约 +15–20%），
      `BT=32` 打平、`BT=16` 反略慢；`xFLA` 从 §22 的 2.10–3.75× 收到 **1.73–3.80×**
      （`B8T2048H8 BT=64` 达 **1.73×**，全表最好）。**Part C**：smem 反而**变大**（BT64/NS1 81.4→**103.4KB**，
      理论占用率 12.5→7.8%）——省了 padding、赔了 stage 粒度。**Part D（`ncu`）**：achieved 占用率
      **6.32→7.75%**、tensor **8.85→10.73%**（逼近 FLA 12.12%）、`short_sb` 1.62→0.91、`barrier` 0.14→0，
      但 `long_sb` 2.25→**3.52**（等 `full[s]` 完成）、bank-conflict 0→26.5 万（占 shared wavefront 3.3%）。
      **结论**：TMA + warp-spec 是真实但**次线性**的一步，把 K2 形状再推 ~15%、**没有**吃掉 2× 残差；
      残差重定位到 **`long_sb`（TMA/mbarrier 完成延迟）+ 浅 pipeline + swizzle 读冲突**，下一步是**加深
      pipeline / K 拆半喂**或改用 CUTLASS `PipelineTmaAsync`。**三坑**：(a) `mbarrier.arrive` 到达数必须
      **严格等于** init count——每 warp 全 32 lane arrive 一个 count=NW 的 barrier 会**硬件 fault**
      （`Unknown Error` 定位在 `mbarrier.arrive`），须 lane 0 arrive 一次；(b) TMA 128 B swizzle 是
      `b^(((b>>7)&7)<<4)`，`ldmatrix` 漏套则「仅行 ≡0 mod 8 正确」（`dv2` 逐 token 的 mod-8 指纹）；
      (c) TMA 搬来的 `dV` 是 swizzled 的，逐标量读 `du+=dv` 必须走 `sw_read` 逆变换，且勿漏 `warp*BW`。
- [x] **P7-21** 把 TMA B1 残差的第一块钉死：§23 相对 §22 的 K2 形状把 `long_sb` 从 2.25 抬到 3.52，
     回读代码发现 TMA 版在 mma 循环里**逐 chunk 直读 global `E^tot`**（reg 版是 `cp.async` 预取，
     `b1_reg_probe.cu:150`），是 §23.1「省一次 TMA」引入的**关键路径 global 标量读**。新增编译开关
     `-DB1T_ETMA`：给 `Stage` 加 `float e[KD]`、用一张 **3D 张量图**（`{KD,NT*H,B}`、`SWIZZLE_NONE`、
     box `{KD,1,1}`）每 chunk 多发一条 `cp.async.bulk.tensor.3d`，compute 改读 smem。新增
     `harness/b1_tma_probe.cu` 的 `B1T_ETMA` 分支 + `harness/check_bwd_b1_tma.py` 的 tmae 列 +
     `ncu_b1_tma_driver.py --which etma` / `probe_b1_tma_ncu.sh`（`docs/07` §24）。**Part A**：tmae
     与基线 TMA **逐位相同（max_abs=0）**（含非整除 T=250）、vs FLA 仍 bf16 底噪。**Part B**：kernel-only
     **0.77–0.98×** 基线 TMA（`NS=1`/BT=16/32 收益最大），**全表最好一行 xFLA=1.63**
     （B8T2048H8 BT=64 **NS=1**：218.5→**195.0µs** vs FLA 119.4）。**Part D（ncu）**：`long_sb`
     4.35→**3.86**（NS1）/3.46→**2.69**（NS2）/2.66→**2.05**（T512）、`wait` 1.60→1.24、tensor
     9.68→**10.83%**（逼近 FLA 12.21%）；**占用率正式排除**——FLA achieved 占用 6.15% 比 ours
     7.7% 还低却快 1.7×，且 `NS=1` 摸到 2 CTA/SM（theo 15.62%）也不成比例加速（grid 128<132 SM）。
     **结论**：`E^tot` 进流水削掉 `long_sb` 0.6–0.8（证明那块就是 per-chunk 全局读）、「加深 pipeline」
     作废（`E^tot` 修好后最优回到 `NS=1`），残差（1.63×）＝每 warp 发射效率（`wait`/`short_sb`），
     下一步＝CUTLASS `PipelineTmaAsync` 或转向 Triton。**坑**：3D TMA 的 `globalStrides` 从 **dim1**
     起算（首版把 dim0 元素大小也算进去→`cuTensorMapEncodeTiled` 报错后**静默**用坏图启动、并污染
     后续 CUDA 状态使 reg 内核 `cudaFuncSetAttribute` 报 `rc=-3`）；fp32 `[128]` 行=512B>128B 故必须
      `SWIZZLE_NONE`。

- [x] **P7-22** 把 §17 的 Triton 软件流水从**孤立原型**落进**真实组装链**（§12 的链此前写死
      `num_stages=1`）：只把 B1 的反向 chunk 循环 `range(NT-1,-1,-1)` 换成前向
      `tl.range(0,NT,num_stages=NS)`（`c=NT-1-cc`，算术一字未动）。新增
      `harness/check_bwd_b1_pipe_e2e.py`（`harness/check_bwd_b1_pipe_e2e.out.txt`，空闲卡）
      + `docs/07` §25。**Part A**：`NS∈{2,3,4}` 链与 `NS=1` 链 `dq/dk/dv/dg/dh0` **逐位相同
      （max_abs=0）**、`dbeta` ≤1.5e-8（B2 `db` 的 `atomic_add` 序），vs FLA 最坏 max_rel 与 §12
      逐位一致（`dq 3.42e-3 … dh0 8.27e-3`，含非整除 T=250/513、g=0、B=2）。**Part B（链级）**：
      `NS=3` 让 B1 **在所有 shape 反超 FLA `dhu`**（`T512H4` 54.4→**34.9** vs FLA 54.0、`B2T1024H8`
      122.1→**61.9** vs 137.1=`2.2×`、`B8T2048H8` 387.3→**318.9** vs 567.0=`1.78×`、`T250H3`
      23.5→**19.0** vs 27.7），整条反向 `dev/FLA` 从 §12 的 **0.51–0.61×** 收到 **0.48–0.54×**
      （`NS≥3` 到平台）。**Part C（孤立 BT×NS 扫描）**：BT=16 部署粒度亦有效（`B2T1024H8`
      143→80），`BT=64/NS=4` **OOM**（`Required 243200 > 232448`）。**Part D（方法学）**：受控诊断
      证明**孤立小 grid launch 的 CUPTI 归因比链内背靠背高 ~1.3×**（`T512H4 NS3` 孤立 49.7 vs 链内
      归因 35.1）——固定启停/排空被算进 device-active 窗口，故**链级口径一律用链内归因**。**结论**：
      兑现「直接接受 Triton 版」这条路线的链级落地，**Triton 侧 B1 工程化收敛**；链级瓶颈从 B1
       **明确转向 B2/B3**（`B8T2048H8` 上 B1 只占 ~0.32/2.22 ms，且 §15 已证整链跟到 64 净亏）。
- [x] **P7-23** 第一次把链级新瓶颈 **B2/B3 单独量清楚**（此前 `ncu` 只剖析过 B1）：新增
      `harness/check_bwd_b2b3_e2e.py`（`harness/check_bwd_b2b3_e2e.out.txt`，空闲卡）+
      `harness/probe_b2b3_ncu.sh`/`harness/ncu_b2b3_driver.py`（`harness/probe_b2b3_ncu.out.txt`）+
      `harness/probe_b2b3_hdh_bf16.py`（`harness/probe_b2b3_hdh_bf16.out.txt`）+ `docs/07` §26。
      **Part A**：组装链按 kernel 归因，`B8T2048H8` 上 **B2+B3 = 66.6%**（B2 906 µs/B3 580 µs，
      B1 只剩 13.8%），坐实 §25「别再压 B1」。**Part B**：免费旋钮已用尽——`num_stages` 对 B2/B3
      **完全无效**（短归约/标量循环，与 B1 的长串行循环相反）；tiling 已最优（B2 `BK=BV=64`；
      B3 `BC=16`，`BC=32`→2022 µs、`BK=BV=128`→37.8 ms）；唯一免费杠杆是 **B3 `num_warps=2`**
      （1.16–1.27×）。**Part C**：B3 `nw=2` 落链 `dq/dk/dv/dg` **逐位相同（max_abs=0）**、`dbeta≤1.5e-8`、
      vs FLA 最坏 max_rel 与 §25.2 逐位一致；B3 内核 0.79–0.87×、整链 dev **0.94–0.98×**。**Part D
      （`ncu`）**：**B2 = DRAM-bound**（DRAM 59% SOL、L2 60%、tensor 4.9%、`long_sb` 6.61）+ **寄存器
      压占用**（247 regs → Block Limit Reg 2 → occ 12.4%）；**B3 = 无 TensorCore 的 FMA/L1-bound**
      （tensor **0.05%**、SM 82%、L1/TEX 71%、occ 49%），本质是逐元素外积、不走 MMA；B3 `BT=64` 掉到
      **2.29 ms**（REG 201、occ 12.4%）解释 §15 的 4.8–5.5×。**Part E（结构性杠杆）**：B2 的 `h/dh`
      换 bf16 只值 **1.08–1.16×**（非量级项）且给梯度加 ~1.5–2e-3。**结论**：B2/B3 性质不同须分别
      处理（B2 减 DRAM/寄存器；B3 MMA 化），本轮为**纯剖析 + 免费旋钮收口**，结构性重写留下一步。
- [x] **P7-24** 兑现 §26.6 第 (2) 条：**B3 `intra` 对角块 MMA 化**。关键观察——部署粒度
      `BT=BC=CHUNK=16 ⟹ NC=1 ⟹ i_i≡0`，Parts A/C（仅有的 `tl.dot`）从不执行，B3 **全部**是对角
      `for j` 外积循环，这正是 §26 的 `tensor 0.05%`。恒等式：把门控指数移出收缩，
      `dq2=2^{g}⊙(Mq@(k·2^{-g}))`、`dkt=2^{-g}⊙(Sq@(q·2^{g}))`（`Mq=(a≥j)⊙dAqk`、`Sq=(a≤j)⊙dAqkᵀ`）
      ——FLA **`SAFE_GATE=True` 分支早已这么做**（`chunk_intra.py:520-526/620-637`），默认
      `safe_gate=False`（`chunk.py:47`）才是逐元素。新增 `harness/check_bwd_b3_mma.py`
      （`..._correctness/…_timing/…_chain.out.txt`）+ `harness/ncu_b3_mma_driver.py` /
      `harness/probe_b3_mma_ncu.sh`（`..._b3_mma_ncu.out.txt`）+ `docs/07` §27。**Part A**：8 组
      case（`BT=16/64`、非整除 `T=250/513`、`g=0`）四项 vs baseline/FLA 最坏 `max_rel` 1.0e-3~1.1e-3
      =tf32 底噪。**Part B**：孤立 kernel `dev` **0.51–0.67×**（BT=16/64、nw=2/4 均如此）。**Part C**：
      落链后 vs FLA 六项最坏 `dq 3.42e-3…dh0 8.27e-3` **逐值同 §25.2/§26.3**；自身增量
      `dq 1.42e-3/dk 1.09e-3/dv 0/dbeta 3.0e-5/dg 6.08e-3`；链 `dev` **0.88–0.91×**
      （B8T2048H8 2216→1957 µs）。**Part D（ncu）**：`tensor 0.05%→2.61%（×52）`、`fma 38.8→10.1%`、
      SM `82→27%`、Duration **527→257 µs（0.49×）**，但转成 **L2/访存 bound**（L2 43→74%、DRAM
      28→57%、`long_sb` 3.22→16.71）；DRAM%翻倍×Duration 减半＝总字节不变（Mem 941 GB/s→1.92 TB/s）。
      **结论**：B3 结构性重写兑现（kernel 1.5–2×、链 −10%），但 tensor 只到 2.6%——B3 下一杠杆是
      **减对角平铺重复读/L1 局部性**而非再加 MMA；链级瓶颈转到 **B2**（下一步＝§26.6 第 (1) 条）。
      **踩坑**：对角 `dAqk` 平铺的列是 **chunk-内局部**下标 `i_i*BC+o_i`（非绝对 `i_ti+o_i`），
      用错指纹=`i_t=0`（T≤BT 单 chunk）对、其余全错到 O(1)。
- [x] **P7-25** 兑现 §26.6 第 (1) 条里最便宜、**算术不变**的那一项——**B2 `wy_dqkg` 的 register-spill**。
      对未改动的 B2 直接读编译期寄存器：部署 `nw=4` 卡在 Triton 255 上限并**溢出 12 个**（`nw=2` 溢出
      190；`ncu` 回读 247 regs/thread、`Block Limit Registers=2`）。根因＝`i_k` 循环**顶部**就加载
      `q/k/g`（`check_bwd_dqkg_kernel.py:244-249`），但它们只在 `i_v` 循环**之后**才用到
      （`:290-305`），故 `b_q/b_k/b_g` 在整个 `i_v` 循环里一直活着、与 `b_h/b_dh`+三累加器叠加超 255。
      改法＝把这三个 `tl.load` 挪到 `i_v` 循环之后（**运算/累加序一字未改**）。新增
      `harness/check_bwd_b2_struct.py`（`harness/check_bwd_b2_struct.out.txt`）+
      `harness/ncu_b2_struct_driver.py` / `harness/probe_b2_struct_ncu.sh`
      （`harness/probe_b2_struct_ncu.out.txt`）+ `docs/07` §28。**Part A**：late vs baseline
      **逐位相同（六项 `max_abs=0`）**，7 组 case 含非整除 `T=250/513`、`B=2`、`BT=16/64`；
      vs FLA 仍只在 `db 1.7e-7/dg 6.1e-8`（atomic 序）。**Part B/C**：`nw=4` 溢出 12→**0**、`nw=8`
      254→224；孤立 kernel `dev` **0.84–0.89×**（`B8T2048H8` 896→800）、`do_bench` 0.89–0.90×。
      **Part D**：落链（pipe B1 `NS=3`+MMA B3）vs FLA 六项最坏 `dq 3.42e-3…dh0 8.27e-3` **逐值同
      §25/§26/§27**；自身增量 `dq/dk/dv/dg` **逐位 0**、`dbeta ≤3.7e-8`；链 `dev` **0.95–0.97×**
      （`B8T2048H8` 1961→1862）。**Part E（ncu）**：regs/thread 247→**210**、但 `Block Limit
      Registers` 仍 **2**（`210×128×3>65536`，须 ≤170 才到 3 blocks）、占用率 12.36→12.41% **未动**、
      DRAM 字节**逐位不变**（777.6→778.0 MB）——收益是**去掉溢出流量**而非提占用/减字节。**口径提醒**：
      ncu 自报 `Duration` 给 late 反而略慢（与 CUPTI/`wall`/`do_bench` 三口径一致反），同 §25.5
      窗口不同源，故只用 CUPTI/do_bench 比值 + ncu 静态寄存器/字节。**结论**：零数值代价的 B2 结构性
      收益（链 −3~5%）；B2 根本 bound（DRAM + 2 blocks/SM）未解，下一步＝去 `i_k` 重复读 / `dh` 不落盘
      （B1→B2 融合）。
- [x] **P7-26** 把 §28.7 留给 B2 的两条结构项（`h/dh` bf16、`i_k` 去重）在**真实组装链 + 部署 dtype**
      下核验，各给一个明确结论（本轮为**收口/反例**轮）。新增 `harness/check_bwd_b2_hdh.py`
      （`harness/check_bwd_b2_hdh.out.txt`）+ `harness/ncu_b2_hdh_driver.py` /
      `harness/probe_b2_hdh_ncu.sh`（`harness/probe_b2_hdh_ncu.out.txt`）+ `docs/07` §29。
      **关键事实**：部署链里 B2 操作数**全 fp32**（`_prep_b` 只转 B1 用的 `qg/kg/w`），故
      §26.4/§28.6 的 ncu（driver 默认 bf16）**不是部署 mix**——新 ncu 全 fp32 实测 B2 读
      **1.63 GB**（非 778 MB），其中 `h`+`dh`=1.07 GB=**66%**。**(1) `h/dh` bf16**：隔离
      正确性 `dq≤1.6e-3/dk≤2.3e-3/dv=0逐位/db≤2.1e-5/dg≤1.9e-3`、CUPTI `dev` 大 shape
      **0.90–0.91×**、ncu 读 1.63→**1.03 GB（−37%）**但 occ 不动（12.4%）且 SOL 只 53–71%
      （进一步证明**非带宽受限**）；**落链反而 1.07–1.24× 变慢**——因 harness 每次调用对 `h/dh`
      做 `.to(bf16)`（两个 ~GB 级 elementwise kernel），而**内核归因 B2 本身是 0.90–0.97×**。
      故该项「**有条件成立：需前向/B1 的 bf16 出口端到端对齐，收益上界 ≈ 内核 10% ≈ 链 ~4%**」。
      **(2) `i_k` 去重**：V 侧重读理论省 256 MB=总读的 16%，但 L2（3.02 GB）≫ DRAM 已吸收；
      `BK=128`（无 i_k 循环）反而读升 1.76 GB、写爆 1.49 GB、时长 1.56 ms（溢出 292）——测不出净收益，
      且去重须改累加结构、寄存器反弹，**在 Triton 原型不做**。**(3) 顺手复核**：唯一进 3 blocks/SM
      的 `BK=BV=32`（162 reg）反慢 1.25× → 占用率不是旋钮。至此 B2 廉价/半廉价杠杆全走完，唯一剩余
      真结构项＝**B1→B2 融合（`dh` 不落盘）**或与 K2 形状（寄存器常驻 + TMA，§22–24）合做。
- [x] **P7-27** 兑现 §29.5 的「唯一剩下的真结构项」——**B1→B2 融合原型**（`dh` 留在寄存器、不落盘）。
      设计：B2 对整条 `V` 收缩且后段有非线性收尾，故无原子的最干净原型取 **`BV=V`**（一个 program 管
      一个 `(b,h)`、逆序扫全块、`b_dS[K,V]` 常驻，每块 B1 完立刻做 B2、`dh` 不落盘；grid `(B·H,)`
      =**K2 形状**）。新增 `harness/check_bwd_fused.py`（`harness/check_bwd_fused.out.txt`，A 正确性 +
      B 计时 + C `V` 分区扫描）+ `harness/probe_fused_regs.py`（`harness/probe_fused_regs.out.txt`）+
      `harness/ncu_bwd_fused_driver.py` / `harness/probe_bwd_fused_ncu.sh`
      （`harness/probe_bwd_fused_ncu.out.txt`）+ `docs/07` §30。**Part A**：融合 B1+B2 与未融合
      B1+B2 在同一份操作数上 **`dq/dk/dv/dg/dh0` 逐位相同（`max_abs=0`）**、`db ≤1.2e-7`；融合链 vs
      FLA 六项最坏 `dq 3.62e-3 … dh0 7.33e-3`（=bf16/tf32 底噪，同 P6-8/P7-26）。**Part B（关键）**：
      融合**反而慢 14–30×**（`B1+B2` 1.37→**19.13 ms** @`B8T2048H8`；整链 8.8–21.7×）。**Part C/归因**：
      (1) 融合内核 **溢出 1786**（regs 255 上限、B1 0 / B2 12）；(2) `ncu` DRAM **读 12.92 GB / 写
      18.93 GB（总 31.9 GB，未融合一对仅 2.7 GB）**——多出的 ~30 GB 全是溢出流量，把原意省的 `dh`
      写+读（~1.4 GB）彻底淹没；(3) 融合强制的 `BV=V` 分区**本身**让 B1 单独就慢 **2.6–5.6×**
      （grid `4·B·H`→`1·B·H`）。**结论**：B1→B2 融合**代数无损但朴素 `BV=V` 单 kernel 不可行**；
      要做必须**改并行结构**＝用 §22–24 的 **K2 形状**（状态由多 MMA warp 分片常驻寄存器 + TMA 大块
      搬运 + 同 CTA 内 B1→B2 交接），而非把整条 `[K,V]` 塞进一个 monolithic Triton program。
      即「融合是唯一剩下的真结构项」成立，但**其前置是 K2 形状**、不是独立开关（同 §15/§17/§20）。
      **待验证**：K2 形状内融合（寄存器分片 + TMA + 同 CTA 交接）须在目标卡以 K2 为基座再测。
- [x] **P7-28** 兑现 §27.7「B3 下一杠杆」里**与 kernel 无关**的那块税：组装链让 B3 在**全新置零**的
      `dq/dk/dg` 上跑、再在 kernel 外物化三个整张 `[B,T,H,K]` 的求和（`check_bwd_b1_pipe_e2e.py:162-166`），
      白读零缓冲 + 再读写一遍 B3 产出 + 3 个 memset。这不是 FLA 的契约：FLA `chunk_kda_bwd_intra`
      把传入 `dq/dk/dg` 当**累加器**、写到独立 `dq2/dk2/dg2`（`chunk_intra.py:554-560/657-667`）。
      改为 B3 **就地累积进 B2 的缓冲**（grid `i_k`×子块轴划分每个 `(token,k)`、无跨 program 竞争；
      `db` 仍 `atomic_add` 叠在 B2 已 store 的值上），**kernel 算术逐行不动**。新增
      `harness/check_bwd_b3_accum.py`（`harness/check_bwd_b3_accum.out.txt`）+ `docs/07` §31。**Part A**：
      就地 vs「零填充 + 链外求和」`dq/dg` **逐位 0**、`dk/db ~1e-7`（纯 fp32 加法次序）；并验证 FLA 契约
      （FLA intra 吃非零 acc == FLA(零)+acc）同组数。**Part B**：就地链 vs 现链自身增量 `dq/dv/dg`
      **逐位 0**、`dk/dbeta ≤9.2e-8`，vs FLA 六项最坏 `dq 3.42e-3…dh0 8.27e-3` **逐值同 §25.2/§26.3**。
      **Part C**：链 `dev` **0.84–0.88×**、`wall` **0.76–0.86×**；per-kernel 归因坐实机制——`B8T2048H8`
      elementwise 归因 **316→49 µs**、每次少 **8 个** elementwise/memset kernel；`ip dev/FLA` 0.36–0.41×。
      **结论**：先消掉这块「链外 elementwise 税」（B3 链时间的 ~16%），下一步才是 §27.7 的 kernel 内
      对角平铺 L1 局部性优化。**约束**：B3 契约改为消费 B2 缓冲（其后 B2 输出不可再用；若 B2/B3 融合须重审）。
- [x] **P7-29** 检验 §27.7/§31.3 遗留的「减少对角平铺重复读 / L1 局部性」并给出**空结论**，同时找到
      真正的杠杆。回读 P7-24 MMA kernel 定位两处具体冗余读：**(R1)** 对角 key 行 `j=i_ti+o_i` 即
      `o_c`，故 `p_kd/p_gd`（`check_bwd_b3_mma.py:142-143`）与 `p_k/p_g`（`:98-99`）地址逐字节相同、
      mask 相同，kernel 却重读一遍；**(R2)** grid `(NK*NC,NT,B*H)`、`BK=32` 使 `NK=4` 个 program 各自
      重读同一份只依赖 `(chunk,head)` 的 `[BC,BC]` 对角 `dAqk/dAkk`（`:135-138`）。新增
      `harness/check_bwd_b3_l1.py`（`..._correctness/…_timing/…_chain.out.txt`）+ `harness/ncu_b3_l1_driver.py`
      / `harness/probe_b3_l1_ncu.sh`（`probe_b3_l1_ncu.out.txt`）+ `docs/07` §32。**Part A**：`reuse`（R1
      消重读）/`bkK`（R2，`BK=K`）与 P7-24 **逐位相同**（`dq/dk/dg max_abs=0`，`db` ~1e-7 atomic 序）、
      `loop` 仅差 fp32 加法序（≤2.4e-6）；vs FLA 均 tf32 底噪。**Part B（空结论）**：`reuse` 恒 **1.00×**
      ——R1 被 Triton CSE 掉、根本不生成重复 load；`BK=64` 部署 `BT=16` 大 shape 只 **0.95×**、
      `BK=128` 中性（R2 被 L2 吸收）。**Part B（真杠杆）**：新增 `_intra_kernel_loop`（grid `(NT,B*H)`、
      program 内 `tl.range(0,NK,num_stages=NS)`），大 shape **0.79–0.81×**，且 `NS=1` 反比 grid 版
      **慢 1.02–1.27×**——收益**全部**来自软件流水而非循环本身。**Part C**：落链六项梯度 vs FLA
      `dq 3.42e-3…dh0 8.27e-3` **逐值同 §25/§26/§27**、自身增量 `dq/dv` 逐位 0，链 `dev` **0.97×**
      （`B8T2048H8` 1940→1875）。**Part D（ncu）**：四档 **DRAM 字节逐位相同**（读 311 MB/写 182 MB）
      ——重复读不进 DRAM；`long_scoreboard 16.74→5.72→3.98`（`ns` 越大越低）、`Duration 257→195`、
      占用率反降（48.6→18–30%）、`L2 82–88%` 成最高 pipe。**结论**：B3 kernel 级优化收敛（已彻底
      L2/DRAM bound），再往下只能靠 B2/B3 **融合**把 `dq/dk/dg` 累加器留寄存器，而非 B3 内 tiling。
- [x] **P7-30** 兑现 §32.6/§31.3/§30.5 的「B2/B3 融合」：部署粒度 `NC=1` 下 B3 只剩**对角** Parts B/D
      （两个 `tl.dot`，§27.1），把它折进 B2 的 `(NT,B*H)` program（B2 结束时已有完整 `dAkk`=`b_dA`），
      把 intra 贡献加到 B2 的 `dq/dk/dg` 寄存器上再存一次，`dA` 不再落盘。新增
      `harness/check_bwd_b2b3_fused.py`（`check_bwd_b2b3_fused.out.txt`）+ `harness/ncu_b2b3_fused_driver.py`
      / `harness/probe_b2b3_fused_ncu.sh`（`probe_b2b3_fused_ncu.out.txt`）+ `docs/07` §33。**Part A**：
      融合 vs「B2+B3(MMA 对角)」同操作数 **`dq`/`dv2` 逐位相同**、`dk/db/dg` ≤2.2e-7（fp32 加法序，
      含非整除 `T=250/513`、`K/V=64/128`）。**Part B（关键约束）**：留 `[BT,K]` 累加器只能 `BK=K`，
      而 `BK=K` 必溢出（`B8T2048H8`：`BV=64/nw=4` 溢 **182**、`nw=8` 溢 60、`BV=128/nw=4` 溢 222），
      **唯一解法＝`BV=V`（去掉 `i_v` 循环）+ `nw=8`** 把溢出压到 **42**；此时孤立 dev 相对部署基线
      （B2 nw=4 + B3 nw=4）**0.47–0.85×**。**Part C**：落链（对比 P7-28 就地累积链）六项梯度 vs FLA
      **逐值同 §25/§26**（`dq 3.417e-3…dh0 8.267e-3`）、自身增量 `dq/dv` 逐位 0、`dk/dg ≤1.1e-7`、
      `dbeta ≤2.4e-7`；链 `dev` **0.83–0.93×**、`wall` **0.90–0.92×**（`B8T2048H8` 1682→1541µs；
      逐 kernel：融合 1053µs vs `wy_dqkg 902 + intra 295 = 1197µs`）。**Part D（ncu）**：DRAM 字节
      **2.56→1.96 GB（−23%）**、L2 **4.40→3.66 GB（−17%）**、tensor `5.74→6.20%`、`long_sb 8.97→5.76`、
      Duration 1193→1050µs；仍 255 寄存器 + 165.9KB smem 钉 1 block/SM（与 B2 同，非占用率翻身，
      纯减字节）。**结论**：融合**成立**（第三次尝试，前两次 B1→B2 朴素融合是反例），但**唯一使能条件
      是 `BK=K`+`BV=V`**（结构前提而非可选旋钮）；§32 的「B2/B3 融合」至此兑现。**闭包**：B1 由 §25
      流水解决、B2/B3 由本轮融合解决，Triton 侧反向三元组工程化收敛。
- [x] **P7-31** 量化 §30.5 遗留的「K2 形状内 B1→B2 融合」**收益上界**（不建融合 kernel）：新增
      `harness/check_bwd_dh_roundtrip.py`（`check_bwd_dh_roundtrip.out.txt`）+ 编译开关 `B1T_NODH`
      （`b1_tma_probe.cu`，删 `dh` 全局 store、保留 op1 用的 `sdh` 镜像）+ `build_kernel(nodh=)`
      （`check_bwd_dqkg_kernel.py`，B2 不读 `dh`）+ `harness/ncu_dh_roundtrip_driver.py` /
      `harness/probe_dh_roundtrip_ncu.sh`（`probe_dh_roundtrip_ncu.out.txt`）+ `docs/07` §34。
      **Part A**：`dv2/dh0` 逐位不变下，B1 去掉 `dh` store 快 **0.28–0.67×**（BT=64 占 33%，
      BT=32 NS=2 占 **72%**）；`ncu` 定位为**不是带宽**（DRAM read 不变、write 仅 −62 MB）而是
      **逐标量 `st.global` 打满 store 队列 → `lg_throttle` 0.67→0.06 + L2 流量放大 ~3×**（866→418 MB），
      `tensor%` 10.8→16.3。**Part B**：B2 不读 `dh` 只值 **~5%**（`nodh/base 0.92–0.95`，ncu read −92 MB、
      Duration 755→717、长 `long_sb`＝DRAM bound）。**Part C（关键）**：往返上界 ≈ B1+B2 的 **~10%**，
      其中 **~2/3 在 B1 侧且是 `lg_throttle`（store 效率）而非融合专属**——**只要把 `dh` 落盘换成
      TMA/专职 STORE warp（镜像前向 K2 的 STORE warp）就能在不做融合下吃掉 B1 侧 33%**；融合相对
      「高效 `dh` store」的净增量只剩 B2 侧 ~5%（同 §26/§29）。**结论（修正 §30.5）**：目标卡 CUDA 反向
      下一优先级＝**B1 的 `dh`/`dv2` 改 TMA store**，而非 B1→B2 融合。**待验证**：TMA store 变体能否
      复现这 33%（预期能，TMA 有独立引擎、不求 LSU 发射）。
- [x] **P7-48** 拆开 §50.5 留下的唯一待查线索——B2 K2-mma 核的「L2 字节比 identical-op Triton 高 35%
      （3.03 vs 2.24 GB）」。给 `b2_mma_probe.cu` 加**受控字节消融**：`-DB2M_NOHDH`/`-DB2M_NODOV`
      （跳过操作数 `cp.async`、smem 零填充）、`-DB2M_VECST`（输出改 64-bit `float2` store）、
      `-DB2M_NOASYNC`（同 16B、同访问模式，换成普通走 L1 的 `float4` load）。新增
      `harness/check_bwd_b2_l2.py`（`check_bwd_b2_l2.out.txt`，空闲卡）+ `harness/probe_b2_l2_ncu.sh`
      （`probe_b2_l2_ncu.out.txt`）+ `docs/07` §51。**结论（0.79 GB 全部拆开、无第三项）**：
      **(a) 读侧 +0.64 GB = `cp.async` 的 `.BYPASS` 路径属性**——`h/dh` 与 `do/vn/dv` **各自** 1.50×
      （`h/dh` 50.33 M 扇区=1.61 GB vs 数据 1.07 GB；`do/vn/dv` 9.44 M vs 6.29 M），DRAM 读两侧都 1.28 GB
      不变（多出的是 L2 re-touch）；`noasync` 同请求数（2,490,368）把 L1TEX 扇区 18→**16/请求**、srcunit 读
      59.77 M→**39.85 M（1.0×）** → 是 BYPASS 路径**非** 16B 宽度/访问模式（扇区级成因**待验证**）。
      **(b) 写侧 +0.09 GB = 25 条逐标量 32-bit `STG.E`**（4 lane 各写 4B 到 sector 的 `{0,8,16,24}` →
      每次 store 只覆盖半 sector，l1tex st **12.71 M=1.98×**）；`vecst` 合 `float2` 后 l1tex st/srcunit 写
      **12.71/9.34 M→6.42 M**，与 Triton 逐位相同。**叠满 `noasync_vecst`=2.23 GB=Triton 逐值**。
      **Part A**：`vecst` vs `base` `dq/dk/dw/dgk` **逐位相同（max_abs=0，含非整除 T=250）**、vs torch 同落
      tf32 底噪；SASS `base` 25×STG.E(32b) → `vecst` 1×STG.E+**12×STG.E.64**。**Part B**：`vecst` CUPTI `dev`
      **0.99×（B8）/0.98×（B2）/0.87×（T512）/0.89×（T250）**（大 shape L2/DRAM 双 bound 只值 ~1%、小 shape
      store 指令减半值 10–13%）；`noasync` 字节降到 2.39 GB 但 **Duration +21%**（丢流水）→ 读放大是 async 的
      代价、不能用普通 load 换。**设计输入**：(1) 目标卡 CUDA B2 **一律向量化 store**（免费、逐位、小 shape
      10%+）；(2) 读侧下一杠杆＝**TMA 输入**（普通 load 净亏 21%，与 §20/§24 收敛）；(3) 省字节须落在**最紧
      pipe**（`vecst` 后 L2 反降到 85%、不再最紧）。**边界**：读侧 1.5× 微架构成因待验证；`noasync*` 破坏
      流水/可能有竞态，仅作字节诊断；bf16/大 `BVK` 未测。
- [x] **P7-49** 兑现 §51.6 的「读侧下一杠杆 = TMA 输入」：把 B2 K2-mma 核（`b2_mma_probe.cu`）的操作数
      路径从 `cp.async`（`.BYPASS`）换成 **`cp.async.bulk.tensor`（TMA）+ SWIZZLE_128B**，看能否既消掉
      §51.4 的 +0.64 GB L2 读放大、又**不像 `noasync` 那样丢流水（−21%）**。新增 `-DB2M_TMA`：`do/vn/dv`
      用 4D 张量图 `{V,T,H,B}`/box `{BVK,BT,1,1}`、`h/dh` 用 5D 张量图 `{V,K,H,NT,B}`/box `{BVK,K,1,1,1}`，
      单线程每 stage 发 5 条 TMA + per-stage `mbarrier`（`expect_tx`），OOB 尾行由 TMA 零填充。新增
      `harness/check_bwd_b2_tma.py`（`check_bwd_b2_tma.out.txt`，空闲卡）+ `harness/probe_b2_tma_ncu.sh`
      （`probe_b2_tma_ncu.out.txt`）+ `docs/07` §52。**Part A**：`tma`/`tma_vecst` 与 `base` 的 `dq/dk/dw/dgk`
      **逐位相同（max_abs=0，含非整除 T=250）**、vs torch 落 tf32 底噪（与 §49.2 同）。**ncu（`B8T2048H8`）**：
      读放大**整条消失**——srcunit 读 **59.77M→39.85M（=DRAM 扇区数 1.0×，与 `noasync` 预测逐值吻合）**、
      l1tex 全局读请求 **=0**（SASS `base` 10×`LDGSTS`→`tma` 10×`UBLKCP`），DRAM 读 1.28 GB 不变；L2 3.02→
      **2.35 GB**、`tma_vecst`→**2.25 GB（Triton 2.23）**。**且兑现时间**：`base` 最紧 pipe 是 L2 89.0%>DRAM
      84.6%，删读放大后 L2 降 87.5%、**DRAM 89.0% 成最高 pipe**，Duration **519.8→494.2 µs**；`tma_vecst`
      491.0 µs（比 identical-op Triton 核 ~551 快 **1.12×**）。**CUPTI `dev`**：`tma` 0.951×（B8）/0.955×（B2）/
      0.933×（T512）/0.955×（T250），`tma_vecst` 0.946/0.926/0.794/0.857——**全 shape 净赢**（对比 `noasync`
      的 −21%）。**结论**：TMA 保留异步、走独立引擎，故**消放大且净赚 ~5%**；目标卡 CUDA B2 操作数路径应直接上
      TMA；「省字节落最紧 pipe」再得正例。**边界**：128 B swizzle 强绑 `BVK==32`（`BVK=64` 需拆 atom，未做）、
      bf16/非 GEMM 段未覆盖；本核 `tma_vecst` 已贴 DRAM 地板 1.11×。

- [x] **P7-50** 关闭 §52.6 的硬边界——**把 B2 TMA 输入从 `BVK==32` 扩到 `BVK=64`**（§49.5 的
      cp.async 最优分块恰是 `BVK=64`，此前两收获无法叠加）。给 `b2_mma_probe.cu` 加 `TMA_SWZ128`
      宏（`BVK==32` 走 `SWIZZLE_128B`、`BVK==64` 改 **`SWIZZLE_NONE`** 稠密 256 B 行；`rd`/两张张量图
      按宏二选一，`b2_mma_probe.cu:117-122/148/460/477`，**算术一字未动**）。新增
      `harness/check_bwd_b2_tma64.py`（`check_bwd_b2_tma64.out.txt`，空闲卡）+ `harness/probe_b2_tma64_ncu.sh`
      （`probe_b2_tma64_ncu.out.txt`）+ `docs/07` §53。**Part A**：`tma64` 与 `base64` **逐位相同
      （`dq/dk/dw/dgk` max_abs=0）**、`base32==base64` 的 `dq/dk/dw` 也逐位（证 k8 累加序不随 `BVK`
      变；`dgk` 2.9e-6=fp32 分组）、`tma64` vs torch 3.6e-4（tf32 底噪、比 Triton 更贴真 fp32）。
      **Part B（关键）**：`tma64/base64` **恒 2.5–3.0×（慢）**、`tma64/base32` 2.6–2.8×——`SWIZZLE_NONE`
      把「最优点 3%」倒赔成 −180%。**Part D（ncu）**：`tma64` 读侧修复**仍在**（L2 2.35 GB=`tma32`、
      srcunit 39.85M、l1tex 全局读=0、SASS `LDGSTS`→`UBLKCP`），但 **bank conflict 8.39M→272.6M（32×）**、
      shared-load wavefront 37.7M→302.0M（8×）、DRAM SOL 30%、Duration 1470 µs。**根因**：稠密行 stride
      256 B=64 float，`64≡0 (mod 32)` ⟹ 固定列下 128 个 `h` 行全落同 bank（mma 片段恰是固定列扫行）；
      `base` 的 `BVP=BVK+4`（68 mod 32=4）与 `tma32` 的行内 swizzle 都消失，唯独 `SWIZZLE_NONE` 两者皆无。
      **结论**：`SWIZZLE_NONE` 捷径**正确但不可用**（证伪）；`TMA+BVK=64` 只能走 §52.6 另一条路——
      **把 tile 拆成两个 128 B swizzle `[·,32]` atom**（未做，唯一待验证）。当前可用最优=cp.async@64
      （0.964×）或 TMA@32（0.949×），**二者尚未叠加**。**方法学**：跨 `BVK` 先用逐位校验判布局解码
      （`base32==base64` 即证累加序不随 `BVK` 变），再单独归因时间差；初版 `#if` 误用 `constexpr`（非宏）
      致 `tma32` 也走稠密、被逐位校验立刻暴露。**该唯一待验证（两个 128 B swizzle atom 拆分）已由 P7-51
      关闭（答案：逐位正确、无 bank 冲突、全程最快，见 `docs/07` §54）。**
- [x] **P7-51** 兑现 §53.5 的**唯一**待验证路径——`TMA + BVK=64` 改走**两个 128 B swizzle atom 拆分**
      （而非 `SWIZZLE_NONE` 稠密 256 B 行）：把 B2 K2-mma 核的 TMA 操作数布局从「一块 `[rows][BVK]`、box 宽
      `BVK`」改成 **atom-major `[NAT=BVK/32][rows][32]`**（每个 32-float/128 B 宽的 atom 自己的 TMA box、
      恒 `SWIZZLE_128B`），`rd` 按 `col>>5` 选 atom 后做 `Swizzle<3,4,3>` 解码，**算术一字未动**
      （`b2_mma_probe.cu:130-165/260-278/463-485`）。新增 `harness/check_bwd_b2_tma_atom.py`
      （`harness/check_bwd_b2_tma_atom.out.txt`，空闲卡）+ `harness/probe_b2_tma_atom_ncu.sh`
      （`harness/probe_b2_tma_atom_ncu.out.txt`）+ `docs/07` §54。**Part A**：`atom64` 与 `base64`
      `dq/dk/dw/dgk` **逐位相同（max_abs=0，含非整除 T=250）**、与 `tma32` 的 `dq/dk/dw` 也逐位（证
      `BVK=32/64` 拼出同一 `k` 收缩序）、vs torch tf32 底噪 3.6e-4（比 Triton 更贴 fp32）；4 组 shape。
      **Part B**：`atom64/base32` dev **0.853–0.974×**、`atom64/base64` **0.927–0.976×**——**§49.5「BVK=64
      最优分块」× §52「TMA 读侧修复」两收获首次叠加**（对照 §53.3 `SWIZZLE_NONE` 版 2.5–3.0×）。**Part D
      （ncu）**：读侧修复仍在（L2 3.02→**2.39 GB**、srcunit 59.77M→**39.85M**、l1tex 全局读 **=0**、SASS
      `LDGSTS`→`UBLKCP` 10→20），且 bank conflict **8.39M**（=base64，**不再是 `SWIZZLE_NONE` 的 272.6M**），
      DRAM SOL **90.6%**（最高）、Duration **485.9 µs**（518.9→485.9，全档最短）。**结论**：§53.5 待验证成立，
      目标卡 CUDA B2 取 `BVK=64` 时用 **NAT 个 128 B swizzle atom**、勿用 `SWIZZLE_NONE`；可与 `-DB2M_VECST`
      无冲突叠加。**边界**：只测 `BVK∈{32,64}`、fp32、`BT=16`；`atom64` 的 L2 比 `tma32` 略高 0.04 GB 未归因。

- [x] **P7-52** 兑现 §49.7「补非 GEMM 段」的**第一刀**：把 B2 的 `i_k==0` **输出支路**
      `dv2 = beta·(A@dv)`（B2 里唯一不依赖门控链的**完整**输出）落进 K2 形状 CUDA B2 核。给
      `b2_mma_probe.cu` 加 `-DB2M_OUT`（+ 独立符号 `b2out_launch`）：`acc2[2][4]` warp 私有
      `[16(o_A),16(v)]` tile；**收缩维是 chunk 行 `o_t` 而非 V**，故 `Akk`（`[B,T,H,BT]`，`o_A` 连续）
      喂 mma A 片段只需**索引重映射**（`:448-451`）、B 操作数**复用 GEMM 段已 staged 的 `dv` tile**
      （`:456-458`），**不新增任何 `dv` 字节、无 smem 转置**；`beta[o_A]` 直读（`:530/534`）；`BVK` 的
      stage 归属按 `local0=warp*BW-it*BVK` 守卫（`:441-444`）。新增 `harness/check_bwd_b2_out.py`
      （`harness/check_bwd_b2_out.out.txt`，空闲卡）+ `harness/ncu_b2_out_driver.py` /
      `harness/probe_b2_out_ncu.sh`（`harness/probe_b2_out_ncu.out.txt`）+ `docs/07` §55。**Part A**：
      四组 shape（含非整除 `T=250`）`dv2` 落 tf32 底噪（`out`/`out_tma`/`out_tma64` vs torch
      **3.63e-4**，**比 Triton B2 的 5.96e-4 更贴真 fp32**）、`dvb` 4.40e-4；`out` vs Triton 8.78e-4
      （=两侧 tf32 分组差）；**`out` vs `out_tma64` 逐位相同（`max_abs=0`）**（把 §54 的布局正确性在新
      算术上再证一遍）。**Part B**：输出支路只给 GEMM 核心加 **+12–25%**（`B8T2048H8` `out_tma`
      555.0 vs `core_tma` 492.3=`1.13×`）；含 GEMM 的 CUDA 核到全 Triton B2 的 **0.62–0.74×**；
      `out_tma64` 反回退（`BVK=64` 时 warp 只在 1 个 stage 有活）→ **输出支路偏好小 `BVK`**（与 GEMM
      段相反）。**ncu（`B8T2048H8`)**：增量＝**DRAM 写 +130.6 MB**（`dv2`+`dvb` 两个 `[B,T,H,V]`）＋
      `Duration +60.4 µs`（比 ≈2.16 TB/s，仍冲 HBM），DRAM SOL 仍 **87%**、occ/tensor **一动不动**
      （24.63%/7.52%）；新增 global 读＝A 的 4.19 M sectors，bank conflict 8.73→9.41 M（小头）。
      **结论**：§49.7 第一刀成立、独立可证、便宜；**关键使能是索引重映射**（收缩维在 chunk 行则不转置）；
      剩余大头＝**门控链 + `dkgb=A@dw` + `dA` 的 `dA`/`db` 后处理**（下一步）。**边界**：`dA`/`db`/门控链
      未进 CUDA、同 CTA 交接未做、`out` store 未接 `-DB2M_VECST`。

- [x] **P7-53** 兑现 §55.6 的「补非 GEMM 段」**第二刀**——`dkgb = A @ (-dw)`（B2 `i_k==0` 的另一个
      chunk-行收缩段，喂 `db`/`dg`/`dk`/`dA`）落进 K2 形状 CUDA B2 核。给 `b2_mma_probe.cu` 加
      `-DB2M_DKGB`（+ 符号 `b2dkgb_launch`）：**与 §55 输出支路同构**（`Akk` 喂 mma 的 A 操作数走同一套
      **索引重映射、无转置**，`b2_mma_probe.cu:516-519` vs `:546-549` 逐字相同）；差别是 B 操作数换成
      **新 staged 的 `dw`**（`Stage::dwm`，k-tiled，TMA 复用 `make_map_dov`）并**在 B-fragment 上取负**
      （`b_dw = -b_dw`，`b2_mma_probe.cu:555-556`）。新增 `harness/check_bwd_b2_dkgb.py`
      （`check_bwd_b2_dkgb.out.txt`，空闲卡）+ `harness/ncu_b2_dkgb_driver.py` / `harness/probe_b2_dkgb_ncu.sh`
      （`probe_b2_dkgb_ncu.out.txt`）+ `docs/07` §56。**Part A**：`dkgb` vs torch(fp32，同一份 tf32 `dw`
      操作数) 落 tf32 底噪 **3.64e-4**（`max_abs` 最坏 1.01e-3），且 **`dkgb`≡`dkgb_tma`≡`dkgb_tma64`
      逐位相同（`max_abs=0`，含非整除 `T=250`）**——把 §54 布局正确性在**第三段新算术**上再证一遍。
      **Part B**：`dkgb` 段只给 GEMM 核心加 **+9–13%**（`B8T2048H8` 544.1 vs `core_tma` 492.7=1.10×，
      比输出支路 +12–25% 还略低），CUDA 核到全 Triton B2 的 **0.63–0.70×**；`dkgb_tma64` 大 shape 反回退
      （同 §55 输出支路偏好小 `BVK`）。**ncu（`B8T2048H8`）**：增量＝**DRAM 读 +0.07 GB（`dw`）+ 写
      65.4 MB（`dkgb`）**＋`Duration +48.6 µs`（≈2.78 TB/s，仍冲 HBM），DRAM SOL 仍 **88.7%**、L2 **88.0%**、
      occ/tensor 不动（24.51%/7.68%）；新增 global 读＝A 的 2.10 M sectors、`dw` 走 TMA（`srcunit` +2.39M）；
      bank conflict 8.72→9.18 M（小头）；SASS `LDS` 88→96 / `STG` 25→33 / `UBLKCP` 10→12。**结论**：
      第二刀成立、独立可证、便宜；目标卡 CUDA B2 里凡「收缩维落在 chunk 行」的段（`A@dv`、`A@dw`、
      最终 `A^T diag(beta) A^T`）都走**同一套「原地读 `Akk` + staged 操作数」**；`dw` 这类中间量生产形态应
      **就地交接**（本轮为独立可证而落盘，多读/写 67 MB）。**边界**：`dA` 后处理项（`dw@kg^T`、`A^T diag A^T`）、
      `db`（`sum_k dkgb·kg` 需跨 warp 归约）与门控链未进 CUDA；本段 store 未接 `-DB2M_VECST`。
- [x] **P7-54** 兑现 §55.6/§56.6「补非 GEMM 段」**第三刀**——`dA` 的两个 GEMM 贡献（最终 `-A^T diag(beta) A^T`
      sandwich **之前**）落进 K2 形状 CUDA B2 核：`dApre[o_t,o_u] = Σ_v dv·v` (base, `chunk_bwd.py:264`) `+ Σ_k dw·kg`
      (`:280-281`, `kg=k·exp2(g)`)。给 `b2_mma_probe.cu` 加 `-DB2M_DA`（+ 符号 `b2da_launch`）：两个收缩维都**不是
      `BT`**——base 按 warp 拥有 **V 列**（同输出支路分区，`local0` 守卫）标量 `dv[r,v]*v[c,v]`、`dw@kg^T` 按 warp 拥有
      **输出 k 列**（`accw` 从 mma 片段镜像到 `swall`）标量点积；8 warp 私有 `[BT,BT]` 部分和经 smem 归约后落盘。
      新增 `harness/check_bwd_b2_da.py`（`check_bwd_b2_da.out.txt`，空闲卡）+ `harness/ncu_b2_da_driver.py` /
      `harness/probe_b2_da_ncu.sh`（`probe_b2_da_ncu.out.txt`）+ `docs/07` §57。**Part A**：`dApre` vs torch 落
      **fp32 底噪 5.31e-7**（比 §55 `dvb`/§56 `dkgb` 的 tf32 3.6e-4 低 3 个量级，因本段用**标量 FMA** 非 mma）、
      `da`≡`da_tma`≡`da_tma64` **逐位相同（max_abs=0，含非整除 T=250，含新 staged 的 `v`）**。**Part B（关键）**：
      本段**贵得多**——`da/core` **1.49–1.84×**、`da_tma/core_tma` **1.33–1.61×**（远高于前两刀的 +10–25%），根因＝
      收缩维是完整的 `V`/`K`（各 128，8× MAC）+ 标量 FMA；含本段 CUDA 核到全 Triton B2 仍 **0.72–0.81×**。
      **Part D（ncu）**：`l1tex` shared wavefronts **38.10→69.57M（+82%）**、bank conflict **8.74→24.81M（×2.8）**、
      SASS `LDS 88→287`、`DRAM SOL 88.87→76.17%`、`tensor 8.04→6.00%`、`Duration 494.9→660.4µs`（增量≈标量 smem
      流量、非带宽：新增 DRAM 读写只 ~0.21GB / 165µs≈1.27TB/s）。**设计输入（更新 §56.5）**：「收缩维落在 chunk 行」
      的段走「原地读 `Akk` + staged 操作数」；「收缩维是全维 `V`/`K`」的段（本轮 `dApre`）**应像三条主 GEMM 一样用
      `mma.sync`**（`X@Y^T` 形式可复用 GEMM 核 fragment 读法 + warp 分片归约），本轮为**先证代数**取标量实现。
      **边界**：**未落** `-A^T diag(beta) A^T` sandwich、`db` 的 `sum_k dkgb·kg`、门控链（§49.7/§55.6/§56.6 仍未关闭）。

- [x] **P7-55** 兑现 §55.6/§56.6/§57.6「补非 GEMM 段」**第四刀**——把 §57 的 `dApre` **就地**接上最终
      `-A^T diag(beta) A^T` sandwich，得到**完整 B2 `dA` 输出**：`dAm=where(m_A, dApre·beta[o_u],0)`、
      `dA=where(m_A,-(b_A@(dAm@b_A)),0)`（`b_A[p,j]=A[o_t=j,o_A=p]`；FLA `chunk_bwd.py:300-304`）。给
      `b2_mma_probe.cu` 加 `-DB2M_SAND`（需 `-DB2M_DA`；新符号 `b2sand_launch`，输出 `dApre`+`dA`）：`b_A`/`beta`
      每 chunk 载入 smem（`sA/sbeta/daB`，~3KB），两个 `[BT,BT]@[BT,BT]` 的 **fp32 标量**矩阵乘 + 一次 `beta` 缩放，
      **不重读 `dApre`**（直接吃 §57 的 `da16`）。新增 `harness/check_bwd_b2_sand.py`（`check_bwd_b2_sand.out.txt`，
      空闲卡）+ `harness/ncu_b2_sand_driver.py` / `harness/probe_b2_sand_ncu.sh`（`probe_b2_sand_ncu.out.txt`）+
      `docs/07` §58。**Part A**：`dA` vs torch（**同一份 `dApre`**）**逐位相同（max_abs=0）**、四种 stage 布局
      （非 TMA/TMA@32/TMA@64）`dA`/`dApre` 都逐位、vs FLA 最终 `dA` 落 **tf32 底噪 1.56e-3~2.34e-3**（含非整除
      T=250）。**Part B**：sandwich 很便宜——`sand/da` **1.02–1.06×**、`sand_tma/da_tma` **1.03–1.10×**（远低于
      §57 的 +33–61%，因无新 `[BT,V]/[BT,K]` 收缩）；含完整 `dA` 的 TMA CUDA 核到全 Triton B2 **0.77×**。**ncu**：
      增量＝`dA` 落盘 **+8.0MB** + `sA/beta` 标量读（+0.39M sector）+ `LDS 287→334`/`STG 26→31`，`Duration
      655.8→703.5µs（+7.3%）`，bank conflict/占用率/tensor% 全不动。**关键副产物（更正 §57）**：接上 sandwich 对最终
      `dA` 比 FLA 时暴露 §57 的**符号错误**——FLA `b_dw=-b_dw`（`chunk_bwd.py:280`）故第二项是 **`- dw@kg^T`**
      （§57 写成 `+`，其自身参考两侧同号故未现形）；已改正 `.cu`（`dapart=dbpart-dkpart`）与 §57 参考/公式。
      **边界**：`db` 已落；门控链当时未落（**已由 P7-57 关闭**）。

- [x] **P7-56** 兑现 §55.6/§56.6/§57.6/§58.6「补非 GEMM 段」**第五刀**——写入门梯度
      `db[o] = Σ_v dvb[o,v]·v[o,v] + Σ_k dkgb[o,k]·kg[o,k]` 落进 K2 形状 CUDA B2（`docs/07` §59）。`dvb`(P7-52 §55)
      与 `dkgb`(P7-53 §56) 已在同一核、按相同 warp 列划分算出，故 `db` 是**纯跨 warp 归约**：给
      `b2_mma_probe.cu` 加 `-DB2M_DB`（需 `-DB2M_OUT -DB2M_DKGB`；诊断 `-DB2M_NODB`；新符号 `b2db_launch`）。`dvb·v`
      项的原始 `v` 需**持久镜像** `sh->dbv[warp][BT][BW]`（逐 V-tile 的 `vv` 会在 `acc2` 完整前被覆写，
      `b2_mma_probe.cu:750-762`），`dkgb·kg` 项直接读持久 `skg`；每 lane 归约 `acc2/acc3` 片元，行组内 `shfl_xor`
      再 8 warp smem 归约落盘（`:848-878`）。新增 `harness/check_bwd_b2_db.py`（`check_bwd_b2_db.out.txt`，空闲卡）+
      `harness/ncu_b2_db_driver.py` / `harness/probe_b2_db_ncu.sh`（`probe_b2_db_ncu.out.txt`）。**Part A**：`db` vs
      torch（**同一份 `dvb/dkgb`**）**1.3e-7~1.8e-7**（=纯 fp32 归约序）、vs torch（从操作数）2.0e-4~2.3e-4、
      vs FLA `db` 落 **tf32 底噪 5.5e-4~6.3e-4**（与 `dv2` vs FLA 同阶）、**四布局（非 TMA/TMA@32/TMA@64）逐位相同
      （max_abs=0，含非整除 T=250）**。**Part B**：`db` 是最便宜的一刀——`db_nodb→db` **1.02–1.07×**、
      `tma_nodb→tma` **1.03–1.13×**（无新 global 读、无新收缩）；含 OUT+DKGB+DB 的 TMA CUDA 核到全 Triton B2
      **0.85–1.04×**。**ncu**：增量＝`db` 落盘 `+1.4MB` + 读 smem `dbv/skg`（`LDS 104→124`/`STG 49→50`），l1tex
      全局读 sector **不变**、占用率/tensor%/带宽都不动（`Duration 742→762µs，+2.7%`，核仍 DRAM/L2 双 bound）。
      **坑（真实踩到）**：harness 起初把 `dw` 输入与 `dw_out` 输出设成**同一张量**（`dw=dv@h` 的 global round-trip），
      流水/落盘覆写致 `dkgb` 大面积错（`max_rel≈2`）；改独立缓冲后 DB build 的 `dkgb` 与 §56 专用 build **逐位相同**
       ——正是「`dw` 应同 CTA 交接、不走 global」的实证动机。**边界**：门控链当时未落（**已由 P7-57 关闭**）。

- [x] **P7-57** 兑现 §55.6/§56.6/§57.6/§58.6/§59.6「补非 GEMM 段」**第六刀**——**门控链**（最终
      `dq/dk/dg`）落进 K2 形状 CUDA B2（`docs/07` §60）。把三条 raw GEMM 输出（`accq/acck/acck`）+
      raw `dgk=[Σ_V h⊙dh]`（§49）+ `dkgb=A@(-dw)`（§56）按门控标定成最终输出：`dq=dq_raw*exp2(g)*scale`、
      `dk_mid=dk_raw*exp2(gn-g)`、`kdk=k*dk_mid`、`dgk=dgk_raw*exp2(gn)+Σ_t kdk`、
      `dg=q*dq-kdk+(t==last)*dgk+k*exp2(g)*dkgb*beta`、`dk=dk_mid+dkgb*exp2(g)*beta`。唯一归约是
      `Σ_t kdk`（chunk 内逐 k-列和，**warp 内 `shfl_xor` over groupID**，因 mma 16 行铺在 8 lane×2 行），
      `g==0` lane 写 `sdgkt[KD]`、`last` 行读回；symbols `b2gate_launch`。新增 `harness/check_bwd_b2_gate.py`
      （`check_bwd_b2_gate.out.txt`，空闲卡）+ `harness/ncu_b2_gate_driver.py` / `harness/probe_b2_gate_ncu.sh`
      （`probe_b2_gate_ncu.out.txt`）。**Part A**：`dq` vs torch（**同一 raw**）**逐位相同（max_abs=0）**、
      `dk`/`dg` 落 **fp32 序 4.3e-8 / 2.2e-7**、vs FLA 三项全落 **tf32 底噪 1.0e-3**；三布局（非 TMA/TMA@32）
      `dq`/`dk` **逐位**、`dg` 仅 TMA@64 差 1 ulp（含非整除 T=250）。**Part B**：门控是前五刀**最贵**的一刀
      `gate/nodg` **1.30–1.38×**（对照 `db` +2~13%）；`gate_tma/B2` **0.73–0.78×**。**ncu**：增量＝**+199 MB
      输出写（`dqg+dkg+dg`）+ 9.2× global load sector（片元式散读 `q/k/g`）+ `exp2f`（XU 0.20→1.51 M）**，
      占用/bank conflict/tensor 不动，核仍 DRAM/L2 双 bound。**设计输入**：chunk 内逐列和的落点＝warp 内 `shfl`
      + 一个 `[KD]` smem tile；门控成本回收杠杆＝**把 `q/k/g` 合流载入 smem**（生产 B2 里 `k/g` 已为 `kg` 载入）。
      至此 §5.2 的 **B2 六项 `dq/dk/dv2/db/dg/dA` 第一次全部存在于 K2 形状 CUDA 核**。**坑**：`m_last` 项按
      lane 条件（`g+8==lastrow`）写会让 `g=7` 的行 7/15 都被加——指纹「`dq/dk` 逐位对、`dg` 只在 行%16∈{7,15}
      错」；改逐行坐标即归零。**边界**：`B2M_GATE` 要求 `B2M_DKGB` 且 standalone（不与 OUT/SAND 组合）、
       `q/k/g` 片元式 global 读未合流、标量落盘。**下一步＝六项全搬完后收尾/维护**；有需求时再合流 `q/k/g`。

- [x] **P7-58** 兑现 §60.5「门控链的下一杠杆＝把 `q/k/g` 合流载入 smem 收回 +30%」（当时标**待验证**），
      结论＝**证伪（净亏）**（`docs/07` §61）。给 `b2_mma_probe.cu` 加 **`-DB2M_QKG`**（需 `B2M_GATE`；
      诊断 **`-DB2M_QKGP`** 只把 staging 行距 `KD`(128 字) 加到 `KD+4` 消固定列 row-aliasing 冲突）：
      chunk 开头一次合流 `[BT,KD]` staging 读 `q/k/g`（`b2_mma_probe.cu:643-654`，与 `kg` 同构），门控改读
      `sh->sq/skb/sgb[rowrel][col]`（`:960-976`，**算术一字未动**）。新增 `harness/check_bwd_b2_gate_qkg.py`
      （`check_bwd_b2_gate_qkg.out.txt`，空闲卡）+ `harness/probe_b2_gate_qkg_ncu.sh`
      （`probe_b2_gate_qkg_ncu.out.txt`）。**Part A**：`dqg/dkg/dg` vs P7-57 `gate`/`gate_qkgp`/`gate_tma_qkg`/
      `gate_tma_qkgp` **全逐位相同（max_abs=0）**、`gate_tma64_qkg` 仅 `dg` 1 ulp（含非整除 T=250）——合流逐位无损。
      **Part B（关键）**：合流**一律更慢**——非 TMA `qkg/gate` **1.11–1.51×**、TMA `tqkg/tma` **1.08–1.19×**；
      pad 行距只把大 shape 1.51→1.49、1.09→1.08，**不改符号**。**ncu**：两独立机制——(1) 非 TMA 加 24 KB
      常驻 tile 把 dyn smem 92.7→117.3 KB、占用率上限 **2→1 block/SM**（`warps_active` 24.2→12.4%，pad 后
      bank conflict 19.9M→10.5M 回基线仍 1.49×）；(2) TMA 版占用率不掉（2 blocks），global ld sector 确从
      **19.40M→12.85M（−34%，预期收益兑现）**、pad 后 bank conflict 回基线 11.46M，但 `Duration +8.7%`、
      DRAM 读不变（1.56→1.55 GB）——本核 **DRAM/L2 双 bound（~86/88% SOL）**，删 L1 散读（被 L2 吸收）
      **不动瓶颈**，staging 多一遍 smem 写+同步反成净负。**结论**：**修正 §60.4**——门控 +30% 的成本是
      **三份 `[B,T,H,K]` 输出写（199 MB，交付物）+ `exp2f` XU + 本核本就 DRAM/L2 bound**，**不是** raw
      `q/k/g` 的散读合流度（散读真实存在于 L1 计数但被 L2 吸收、不进端到端）。**设计输入（更新）**：门控链
      **不必**为 `q/k/g` 单设 smem staging；`q` 保持直读 global 更划算；若合流须不增 smem 峰值且不掉占用率
      ——但本轮已证即便做到仍 +8.7%，故**不作为优化项**。**方法学**：把「ncu 计数里的散读」当成本前，须用
      「换算子只改取数来源、occupancy 不变」的受控对拍验证它是否在关键路径上。
- [x] **P7-60** 兑现 §62.6 留的**唯一剩余真结构项**（「六段合成单一生产 launch」，此前六段各是一个独立
      编译变体 + 独立 launcher）：给 `b2_mma_probe.cu` 加 `-DB2M_PROD`，在段旗标定义前把
      `B2M_OUT/DKGB/DA/SAND/DB/GATE` 六旗标全部置 1，新增 `b2prod_launch`（一次 launch 产出 B2 全部六项
      梯度 + raw `dw/dgk`）。**六段代码体一行未动**——它们的累加器/smem/输出两两不相交；唯一共享改动是
      `beta` 由三处声明改为一处（`#if B2M_GATE && !B2M_OUT` / `#if B2M_SAND && !B2M_OUT && !B2M_GATE`
      让位，`:588-591/595-597/610-614`），故**既有六个隔离 launcher 签名全部不变**。新增
      `harness/check_bwd_b2_prod.py`（`check_bwd_b2_prod.out.txt`，空闲卡）+ `harness/ncu_b2_prod_driver.py` /
      `harness/probe_b2_prod_ncu.sh`（`probe_b2_prod_ncu.out.txt`）+ `docs/07` §63。**Part A**：四组 shape
      （含非整除 `T=250`）`prod` 的**每个**输出 vs 对应隔离 build（`dv2/dvb/dkgb/db` vs `db`、
      `dApre/dA` vs `sand`、`dqg/dkg/dg` vs `gate`、`dw/dgk` vs `core`）**逐位相同（`max_abs=0`）**，
      `prod_tma` vs `prod` 也逐位。**Part B**：`prod` regs 70–80→**140 零 spill**、smem **133.8 KB**
      （1 block/SM，occ 12.5%）。**Part C（关键）**：一次 `prod` launch vs 老设计 3 次 launch
      （`db`+`sand`+`gate`，各重做一遍同一 GEMM 核）**0.467–0.526×**（`B8T2048H8` 2450→1288µs；b2 核
      45→15），省下的正是 **2 份 GEMM 核 + 2 次 launch**。**Part D**：`prod` 门控 `dqg` vs torch **逐位**、
      `dkg/dg` ~1e-6（fp32 序）、`dv2` ~1.2e-4（tf32 底噪）。**`ncu`**：合并后 DRAM SOL 45%、tensor 3%、
      occ 12.5%——**非 DRAM-bound、是标量/发射受限**，坐实 ~2× 来自核复用而非换瓶颈 regime。
      **边界**：只验证「合并==隔离」与 launch 摊薄，未改段内实现；下一步＝§49.7 的同 CTA 交接（`dw`/`dv2`
      不落盘就地喂下游段），本节把「六段共存一核」这一前提做实。
- [x] **P7-61** 兑现 §49.7/§63.6 的**最后一条真结构项**——**同 CTA `dw` 交接**（此前 DKGB 段消费的是
      同一个核里 GEMM 基座刚写进全局的 `dw`，即 `dw = dv@h^T` 的全局往返）：给 `b2_mma_probe.cu` 加
      `-DB2M_HDW`（需 `B2M_DKGB`），把核心的 `accw` 留在核内、镜像进 `swall`，DKGB 的 mma 改**在
      it-loop 之后**从该镜像读（`accw` 只有所有 V-tile 累加完才完整，故必须后移）；`Stage::dwm` /
      `prefetch` cp16 / TMA `dwm` 张量图 / 核心 `dw_out` store 全部条件关掉。新增
      `harness/check_bwd_b2_hdw.py`（`check_bwd_b2_hdw.out.txt`，空闲卡）+ `harness/probe_b2_hdw_ncu.sh`
      / 复用 `ncu_b2_prod_driver.py`（`probe_b2_hdw_ncu.out.txt`）+ `docs/07` §64。**Part A**：四组 shape
      （含非整除 `T=250`）`prod_hdw` 十项输出 vs `prod` **逐位相同（`max_abs=0`）**、`prod_hdw_tma` vs
      `prod_hdw` 逐位；**Part A2**：喂**全零 `dw`** 结果**逐位不变**——`dw` 输入证实未被读（交接自证）。
      **Part B**：regs 140→**110** 零 spill、smem 133.8→**129.2 KB**（TMA 123.0→118.9）。**Part C/D**：
      CUPTI `dev`/`wall` 大 shape **略慢 +2.8%/+4%**（`B8T2048H8` 1324 vs 1288µs）、小 shape 噪声内——**wash**；
      **Part E（关键）**：隔离的「核心+DKGB」里交接**稳定快 7–9%**（`B8` 570→522µs）。**机制**：`dw` 往返
      在 `prod` 里被 GEMM 流水**藏住**（原 DKGB 段在 stage 内与后续 stage 重叠），交接后移成**串行尾巴**，
      故省字节换不来时间；隔离段里 staging 无从隐藏，交接净赚。**`ncu`**：字节确实降（DRAM −126 MB、
      L2 −0.33 GB、bank conflict 不增、occ/tensor 不动），但 ncu `Duration` 给 **−7%** 与 CUPTI/wall **反向**
      （同 §63.5 窗口不同源，ncu 清 L2 使那 60 MB `dw` 读变成真 DRAM 读）。**结论**：`dw` 从生产 ABI 消失
      （不预算/不传参/不落盘），是**内存足迹优化、非速度杠杆**；六段齐全的生产核里 quintessentially wash，
      DKGB 主导段里 7–9%。**判据（新）**：省字节前先问「这笔访存是否本就被流水藏住」。`dv2`/`v` 喂 `db`
      与 `dApre` 入 sandwich 早已是核内（P7-55/P7-56），故本轮只动 `dw`；至此 B2 六段的中间量**要么本就
      核内、要么可核内**，K2 形状 CUDA 化在**结构上闭合**。
- [x] **P7-62** 兑现 §60.5 点名的 K2 形状 CUDA 化**最后一条**结构项——**B3 `intra` 折进 K2 形状**：
      部署 `BT=BC=CHUNK=16 ⟹ NC=1` 下 B3 只剩对角 Parts B/D（§27.1），本轮把 §27 的**对角 MMA 恒等式**
      从 Triton 搬进 CUDA，做成 **K2 形状**核（`grid (NT,B*H)`、8 warp、warp `w` 拥有输出 k 列
      `[w*16,w*16+16)`），四个 `[BT,BT]@[BT,KD]` 走 `mma.sync.m16n8k8.tf32` + fp32 门控。新增
      `harness/b3_intra_probe.cu` + `harness/check_bwd_b3_intra_cuda.py`（`check_bwd_b3_intra_cuda.out.txt`，
      空闲卡）+ `harness/ncu_b3_intra_cuda_driver.py` / `harness/probe_b3_intra_cuda_ncu.sh`
      （`probe_b3_intra_cuda_ncu.out.txt`）+ `docs/07` §65。**Part A**：5 组 shape（含非整除 `T=250`、
      单 chunk `T=64`、`g=0`）CUDA vs Triton MMA B3 四项 `dq/dk/db/dg` 最坏 `max_rel` **1.2–1.5e-3**
      （=§27 记录 tf32 底噪）、CUDA vs torch golden 更紧 **≤6.4e-4**（CUDA 的 `beta`/门控走 fp32 标量、
      只有矩阵乘落 tf32）。**Part B**：孤立 `dev` 比 Triton MMA B3 快 **1.3–1.45×**（`B8T2048H8` 240.2 vs
      349.5µs=`0.69×`、`T512H4` 6.6 vs 9.5=`0.70×`）。**Part C**：smem 26.6KB、8 blocks/SM 上限。
      **Part D（ncu）**：`B8T2048H8` Duration 295.6→**239.4µs（0.81×）**、DRAM 61.3→**50.4%**、L1 命中
      18.5→**51.4%**、`long_sb` 19.89→**11.37**、regs 72→59——优势在**访存侧**（`q/k/g` 一次合流 smem、
      8 warp 共享），两核都非 tensor/core bound（对角 intra 部署粒度总量小）。**结论**：结构闭合 +
      比 Triton 版快 1.3–1.45×。**坑**：门控循环漏乘 `dk2` 的 `2^{g}`（指纹＝只 `dq` 对、`dk/db/dg` 同比例
      偏大）。**下一步（未做）**＝并进生产 launch（在 `-DB2M_PROD` 里用核内 `dA`+`dAqk`+已 staging 的
      `q/k/g` 把对角贡献加到门控段最终 `dq/dk/dg` 并进 `db`），须处理寄存器/smem 峰值与累加器共享；
      不改本轮已证代数。**该下一步已由 P7-63 关闭**（逐位无损、单次 launch；但大 shape 非净赢，见 `docs/07` §66）。
- [x] **P7-63** 兑现 §65.6 / P7-62 的「下一步」——**把 B3 对角 intra 折进生产 launch**（`-DB2M_INTRA`，
      要求 `-DB2M_PROD`；新入口 `b2prod_intra_launch` 多一个 `dAqk` 入参）：用 SAND 的**核内** `daB`
      （掩码 `-X2` = `dAkk`，不再落盘读回）+ `dAqk` + 合流 `q/k/g`，四个 `[BT,BT]@[BT,KD]` tf32 mma 产出
      `idq/idk/idg`，**寄存器 merge** 进门控段最终的 `dq/dk/dg`（片元布局逐位同构）、`db` 的 `Σ_k dkpre·k`
      并进既有跨 warp 归约；为读到 `daB` 把 DA+SAND 两块整体前移至 GATE/DB 之前（段内算术一字未动）。新增
      `harness/check_bwd_b2_intra.py`（`check_bwd_b2_intra.out.txt`，空闲卡）+ `harness/ncu_b2_intra_driver.py`
      / `harness/probe_b2_intra_ncu.sh`（`probe_b2_intra_ncu.out.txt`）+ `docs/07` §66。**Part A**：6 组 shape
      （含非整除 `T=250`、单 chunk `T=64`）折进 vs「`prod` + 独立 B3」`dqg/dkg` **逐位（max_abs=0）**、
      `dg` ≤4.8e-7 / `db` ≤9.5e-7（fp32 加法结合序）、**未触及输出全逐位不变**。**Part B**：regs 146→**144**、
      **零 spill**、smem 133760→**159872 B**、仍 1 block/SM。**Part C（关键）**：折进 vs 两次 launch 大 shape
      **1.04–1.06×（略慢）**、小 shape **0.88–0.89×（净赢）**（`B8T2048H8` 1538.1→**1636.7µs**）——折进把 intra
      变成每 CTA 关键路径尾部的**串行尾巴**。**Part D（ncu）**：DRAM 字节几乎不动、占用率/tensor% **完全不变**，
      代价＝bank conflict +55%、L2 +70 MB、global ld sector +6.6 M。**结论**：结构闭合兑现（单 kernel、`dA`
      不落盘），但**目标卡应保持 B2/B3 为两个 kernel**，仅小 batch/短序列（launch 受限）时折进净赢。见 `docs/07` §66。
- [x] **P7-64** 把 K2 形状 CUDA B2 **生产核**（`-DB2M_PROD [-HDW] [-INTRA]`）**第一次落进完整反向链**
       （P7-48…P7-63 的验证全在段间/合成操作数上，从未端到端）：新增 `harness/check_bwd_prod_e2e.py`
       （`check_bwd_prod_e2e.out.txt`，空闲卡）+ `docs/07` §67。组装
       `dAv(Triton)→CUDA-B1→CUDA-B2-prod[→Triton B3 intra]→反向 cumsum`，只吃 K1 workspace；两变体
       `pc`（`prod_hdw` + 独立 Triton intra）、`pci`（`prod_hdw_intra` 单 launch）。**Part A（正确性）**：6 组
       shape（含非整除 `T=250`/`B2 T=513`、`g=0`、`B=2`）六项梯度 vs FLA 落 bf16/tf32 底噪
       （`dq 3.1e-3/dk 3.4e-3/dv 6.8e-3/dbeta 6.0e-3/dg 7.9e-3/dh0 9.4e-3`，与 `cf` 逐值同阶）；
       `dv`/`dh0` 与 §45 最优链 `cf` **逐位相同（max_abs=0）**。**Part B（计时，关键/诚实）**：整核**不是净赢**
       ——`pci/cf` **1.27–1.73×**（`B8T2048H8` `prod_intra` **1606.9µs** vs 融合 Triton `_fused_b2b3_kernel`
       **998µs**；`prod_hdw` 无 intra = 1265.8µs，与 §63 的 1288 互证；`prod_intra` 与 §66 的 1636 互证）。
       差**不是**输出体量（多落盘的诊断中间量 ~155MB≈65µs≈4%），而是**六段非 GEMM 成本在同一个
       1 block/SM、tensor 3% 的核里累积**（§57 `dA` +33–61% + §60 门控 +30–38% 叠加），追不上 Triton
       把 B2/B3 融进一张 `tl.dot` 图 + 寄存器常驻的形态。`pc/FLA` 0.46–0.71、`pci/FLA` 0.47–0.59。
       **结论（更新 §66.6）**：目标卡最优反向仍是 §45 的 `dAv→B1→融合 Triton B2/B3→cumsum`（CUDA 只在
       B1 净赢）；CUDA B2 生产核当前是**正确但更慢**的研究产物，下一杠杆＝把六段**重新融合回寄存器**
       （同 CTA 交接、消 `dA`/门控中间量落盘），但收益上界已被端到端钉在 ~1.6×。**边界**：`B2M_TMA`
       与 `B2M_INTRA` 未打通（TMA 分支未传 `dAqk_in`）；`pc` 的独立 Triton `intra` 计时含链外加法。
- [x] **P7-65** 把 §67.4 的「六段非 GEMM 成本在同一个 1 block/SM 核里累积」**逐段归因**（此前每段成本都在
       **各段独立**的 kernel/launch 里量，生产形态下的边际不可见）：给 `b2_mma_probe.cu` 加
       `-DB2M_PROD -DB2M_HDW -DB2M_LEVEL=k`（k∈0..6，`b2_mma_probe.cu:150-176`），**保留所有 flag/launch
       签名/staging**、只按逆依赖序删段的**函数体**（`1=OUT 2=DKGB 3=DA 4=SAND 5=DB 6=GATE(+INTRA)`，
       INTRA 随 GATE）——与 P7-52…P7-57 的「独立 launcher」（连 kernel 签名一起换）根本不同，故能共存于
       生产核内做增量扫描；`LEVEL=6` 即 P7-60 生产核、`LEVEL=0` 是带全套 staging 的裸 GEMM 核。新增
       `harness/check_bwd_b2_prod_levels.py`（`check_bwd_b2_prod_levels.out.txt`，空闲卡）+
       `harness/probe_b2_levels_ncu.sh`（`probe_b2_levels_ncu.out.txt`）+ `docs/07` §68。**Part A**：各 level
       存活输出与 level 6 **逐位相同（max_abs=0，四组 shape 含非整除 T=250）**（消融纯 compute-only）。
       **Part B/C**（`dev` µs，`B8T2048H8`，regs 31→112）：`OUT 151(24.7%)/DKGB 79(13%)/DA 184(30%)/
       SAND −3/DB 36(5.9%)/GATE 204(33.5%)`——总非 GEMM 开销 **651.6µs**（与 §67 的 1265.8 互证），**OUT+DA+
       GATE 三段吃 83%**；曲线**≈加性**、与各段独立数字同阶（修正 §67.4「累积」措辞）。**`ncu`**（逐值吻合
       CUPTI，7/7）：**OUT 成本=交付物写 +130.8MB**（`dv2+dvb` 两个 `[B,T,H,V]`）、**GATE=交付物写 +198.2MB**
       （`dqg+dkg+dg`）+散读 70MB+`exp2f`、**DA=标量数学**（只写 8MB 却把 bank-conflict 9.4M→20.8M ×2.2、
       tensor 纹丝不动 1.61%）；SAND/DB 本就 ~0/36µs 且中间量早已核内（§55/§56/§64）。**结论（修正 §67.4）**：
       「把六段**重新融合回寄存器**」上界只有 SAND+DB ≈ 40µs（=§67.3 的 ~4%），**不是 1.6×**——三段的钱花在
       **交付物 IO 与 DA 的标量数学**上，重融合删不掉；**唯一有量级的单项杠杆＝把 DA 从标量 FMA 改成
       `mma.sync`**（DA 是唯一仍标量、收缩维=全维 V/K 的非 GEMM 段，§57.6 早已写下该设计输入），把 184µs
       压到 OUT 那样的 mma 量级即可补掉与 Triton 998µs 的 264µs 缺口大半。目标卡默认反向仍推荐 §45 的
       `dAv→B1→融合 Triton B2/B3→cumsum`。
- [x] **P7-66** 兑现 §68.5 点名的「唯一量级杠杆」——**把 DA（`dApre = dv@v^T - dw@kg^T`）从标量 FMA
       改成 `mma.sync`**：给 `b2_mma_probe.cu` 加 `-DB2M_DAMMA`（需 `B2M_DA`），每 warp 用
       `mma.sync.m16n8k8.tf32` 算**整块** `[BT,BT]` partial（2 n-tile × 每项 2 k-step，收缩自己的
       `BW=16` 宽切片），A/B 操作数复用已 staged 的 `dv/swall` 与 `vv/skg`（读法与 GEMM 核逐字同构、
       tf32），8 个 partial 仍过 `dapart→da16` 归约（`:990-1006/1176-1201`）。新增
       `harness/check_bwd_b2_da_mma.py`（`check_bwd_b2_da_mma.out.txt`，空闲卡）+
       `harness/probe_b2_damma_ncu.sh`（`probe_b2_damma_ncu.out.txt`）+ `docs/07` §69。**Part A**：
       `da_mma` `dApre` vs torch **3.42e-4=tf32 底噪**（标量 6.4e-7=fp32 底噪）、`da_mma` vs 标量
       3.42e-4、**三种布局逐位相同（含非整除 T=250）**。**Part B（孤立）**：`da_mma/da` **0.87–0.92×**、
       `da_mma_tma/da_tma` **0.82–0.94×**。**Part C（生产核，关键）**：`-DB2M_PROD -DB2M_HDW` 下
       非 DA 七项输出**逐位不变**、只 `dApre/dA` 落 tf32；`prod(mma)/prod(sc)` **0.93–0.95×**
       （`B8T2048H8` 1279.5→**1213.5**）。**§68 的 L2→L3 累积消融**测生产态 DA 边际：**185.4→83.3µs
       （0.45×）**、其余 shape **0.32–0.48×**——**§68.5 的杠杆方向正确、段内砍 2–3×**。**但整核只
       +5–7%**（DA 只占核 ~14%），`1214/998=1.22×` **未拉进 Triton 融合核**。**`ncu`**：`Duration`
       1.27→1.21ms、bank conflict 23.46→17.30M（**只削 ~40%**，因 tf32 操作数无 16-bit `ldmatrix`、
       片段仍 row-major 直读）、`tensor%` 3.37→3.81、DRAM/L2 不变、regs 112→104、occ 不动。
       **结论（修正 §68.5）**：`mma` 化 DA 值得做（免费、正确、段内 2–3×）但**不是翻盘点**（整核上界
       ~5–7%）；§67/§68 的默认建议不变。**坑**：`B2M_GATE` 下 raw `dq/dk` 缓冲**未初始化**（只存门控
       后的 `dqg/dkg`），误比 raw `dq/dk` 读出 O(1) 垃圾、误判成「mma 改坏门控」，须先查 flag 下哪些
       缓冲真被写。见 `docs/07` §69。
- [x] **P7-67** 关闭 §67.4（P7-64）记录的边界「**`B2M_TMA` 与 `B2M_INTRA` 未打通（TMA 分支未传
       `dAqk_in`）**」：`b2prod_launch_impl` 的 `#if B2M_TMA_ON` 分支启动内核时漏了 `B2M_INTRA` 下
       内核签名末尾的 `const float* dAqk`（`b2_mma_probe.cu:707-709`），故 `-DB2M_TMA -DB2M_INTRA`
       编不过（对照：删修复后 `error: too few arguments in function call`）。修复＝TMA 分支与非 TMA
       分支写同一段 `#if B2M_INTRA (const float*)dAqk_in, #endif` 守卫（**算术一字未动**）；宿主
       `b2prod_intra_launch` 本就 TMA-agnostic（张量图在 impl 内用指针现编码，`:1912-1919`），
       harness 侧无需改 launcher。新增 `harness/check_bwd_b2_intra_tma.py`
       （`harness/check_bwd_b2_intra_tma.out.txt`，空闲卡）+ `docs/07` §70。**Part A**：`prod_intra_tma`
       vs `prod_intra` 六组 shape（含非整除 `T=250`、单 chunk `T=64`）**11 项输出 `max_abs=0`
       （BIT-IDENTICAL）**——TMA 只换 stage 布局、不改任何 GEMM/门控/sandwich 算术（`dAqk` 两布局都
       global 直读）。**Part B**：regs 144→128、smem 159.9→**149.1 KB**（−6.7%）、但 +12B spill；两者
       都仍超 2 block/SM 的 smem 门槛、占用率仍 1 block/SM。**Part C（关键/诚实）**：CUPTI `dev`
       TMA **一律慢 8–12%**（`B8T2048H8` 1642.7→1788.3；与 §66.4 的 1636.7 互证基线口径）、wall 大
       shape ~9%。**结论**：边界关闭、开关随时可叠加且数值无损，但**目标卡 CUDA B2 默认取非 TMA**
       ——TMA 在**孤立 GEMM 核**净赚 ~5%（§52），到**六段齐全的生产核**（§67/§68 已证非 DRAM bound）
       反号成慢 8–12%（per-stage `mbar_wait+__syncthreads` 成净负）；再次印证「同一旋钮在孤立核与整核
       可反号」（§61/§64）、「省字节要落最紧 pipe」（§51.6）。**待验证**：TMA × `-DB2M_DAMMA`、
       TMA 版落 P7-64 整链（`pci` 变体）未做端到端。
- [x] **P7-68** 关闭 §70.6（P7-67）记录的两条**待验证**——(1) `-DB2M_TMA × -DB2M_DAMMA` 在生产核里
       组合、(2) TMA 折叠版落 P7-64 整链（`pci` 变体）——并**修正 §70.5 对 TMA 生产代价的归因**。
       新增 `harness/check_bwd_b2_tma_damma.py`（`check_bwd_b2_tma_damma.out.txt`，空闲卡）+
       `harness/probe_b2_td_ncu.sh`（`probe_b2_td_ncu.out.txt`）+ `docs/07` §71。四个生产变体
       `sc / sc_tma / mma / mma_tma`（全部 `-DB2M_PROD -DB2M_HDW`；**`-DB2M_HDW` 下 `dw` 是 scratch，
       逐位比对须排除**）。**Part A**：`*_tma` vs 其非 TMA 双胞胎 **8 项输出全逐位（max_abs=0）**；
       `mma* vs sc*` 非 DA 8 项逐位、只 `dApre/dA` 落 tf32 底噪（1.2–1.9e-3）；6 组含非整除 T=250/单
       chunk。**Part B（关键）**：`sc_tma/sc` **1.039–1.057**（TMA 仍慢 4–6%），但 `mma_tma/mma`
       **0.978–1.006**（`B8T2048H8` = **1.000**）——**DAMMA 把 TMA 的符号从 +4~6% 拨回 ~0%**。
       **Part C**：`pci_tma` vs `pci` 六项梯度**逐位**、vs FLA 与 §67.2 同阶；链级 `pci_tma/pci`
       **1.006–1.024**；`pci/cf` **1.6–1.7×** 复核 §67.3。**`ncu`（机制）**：TMA 两列 L2 字节都低
       ~0.67 GB，但 bank conflict 行为分叉——**标量 DA 下 TMA 把冲突 23.46M→28.31M（+4.85M）、mma DA
       下 TMA 一动不动（17.30M 两列相同）**（`sc` 23.46M / `mma` 17.30M 与 §69 逐值吻合）。**结论**：
       生产核 TMA 的代价**本体就是标量 DA 段多出的 4.85M bank conflict**（非 per-stage 同步开销，
       修正 §70.5）；DAMMA 打开后 TMA 免费且省 L2 → 设计输入由「默认非 TMA」更新为「**随 DA 实现
       切换**：mma DA 可同时开 TMA，标量 DA 则非 TMA」。**新判据**：`ncu` 的 bank-conflict 增量要
       按段实现形态读（同开关在标量/mma 段一负一零）。**边界/待验证**：链级只测到 `pci_tma`（标量
       DA 折叠版），`pci_mma/_tma`（DAMMA 进链）未做端到端。
- [x] **P7-69** 关闭 §71.6（P7-68）记录的最后一条边界——`pci_mma`/`pci_mma_tma`（**DAMMA 进链**）
       端到端。四个折叠核（`-DB2M_PROD -DB2M_HDW -DB2M_INTRA`）只差 DA 实现 × stage：`pci`（标量/
       cp.async）/`pci_tma`/`pci_mma`（`mma.sync.tf32`）/`pci_mma_tma`，各自链成
       `dAv→CUDA-B1→CUDA-B2-prod_intra→反向 cumsum`，4 组 TCASES（含非整除 `T=250`）。新增
       `harness/check_bwd_b2_damma_e2e.py`（`check_bwd_b2_damma_e2e.out.txt`，空闲卡）+ `docs/07` §72。
       **Part A**：`pci_mma_tma` vs `pci_mma` **六项梯度全部逐位（max_abs=0）**；`pci_mma` vs `pci`
       `dq/dv/dh0` **逐位 = 0**、`dk≤5.45e-6/dbeta≤7.99e-6/dg≤9.89e-7`；四个变体 vs FLA 的 worst-case
       **逐值完全相同**（`dq 3.949e-3/dk 3.196e-3/dv 9.500e-3/dbeta 5.456e-3/dg 4.659e-3/dh0 8.889e-3`）。
       **关键副产物（修正 §71.2 的量级读法）**：捕获原始 B2 输出发现**真实链里 DA 的 tf32 改写量本身
       就小 3 个数量级**（`dApre≤8.14e-6/dA≤6.26e-6`，且 `dqg` 逐位 0；而 §71.2 用 O(1) 合成操作数测得
       1.2–1.9e-3）——tf32 误差随操作数幅度走，链里 DA 操作数（B1 bf16 出口的 `dv`、bf16 派生的
       `v_new`/`h`）幅度小，故链级残留 ≈ 核级改写量（无放大、无额外衰减）。**Part B（关键）**：链
       `dev` `pci_mma/pci` **0.975–1.001**（DAMMA 值 ~1–2.5%，`B8T2048H8` 2423→2362µs）、
       `pci_mma_tma/pci_mma` **0.980–0.992**（**DAMMA 打开后 TMA 反号成快 ~1–2%**，兑现 §71.3 预测）、
       `pci/FLA` 0.45–0.59、`pci/cf` ~1.68–1.73（复核 §67.3）。**结论**：DAMMA 进链数值无损、性能
       ~1–2.5%，其上叠 TMA 再 ~1–2% 且免费，与 §71.6 设计输入一致；目标卡反向默认仍是 `cf`。
       **边界**：仅 `K=V=128/BT=16/fp32 操作数`；只测折叠形态（非折叠 `pc` 的 DAMMA 未单独落链）。
- [x] **P7-70** 关闭 §57.6/§58.6/§62.6 留下的**最后一条**「store 未向量化」边界——`[BT,BT]` 的
      `dA`/`dApre`（§59 明确没做的「小张量 + 一 thread 一 entry」那两处）：给 `b2_mma_probe.cu` 的
      `dApre` store（`-DB2M_DA` 段）与最终 `dA` sandwich store（`-DB2M_SAND` 段）改成**按相邻列成对**
      经 `st2` 写出（`if idx<BT*BT` → `if idx<BT*(BT/2)`、`v0/v1` 两掩码各管 `jj`/`jj+1`），**算术一字未动**。
      新增 `harness/check_bwd_b2_vecst_da.py`（`harness/check_bwd_b2_vecst_da.out.txt`，空闲卡）。
      **关键区别**：§51/§59 的 store 是**片段跨步**（半 sector、`l1tex st`=1.98×），而 `dA`/`dApre` 是
      `col=idx%BT` 型**相邻列**、**本就完全合并（扇区已最小）**——向量化只**砍 store 指令数**（不砍字节）。
      **Part A**：standalone `da`/`sand` base vs vecst 四组 shape（含非整除 `T=250`）**逐位相同（max_abs=0）**。
      **Part B**：孤立段 `dev` **0.93–0.97×**（`sand` B8 1008→974、T512 14.7→13.7）。**Part C（关键）**：
      生产核（`-DB2M_PROD -DB2M_HDW`）**十项输出逐位相同**、`dev` **0.932–0.960×**（`B8T2048H8`
      1262.5→**1208.5**、`B2T1024H8` 164.9→158.2）——与本核「非 DRAM bound、发射受限」的历次归因
      （§61/§63/§68）一致，故少一半 store 指令即兑现时间。**Part D**：SASS `STG.E`(32b) **37→19**、
      `STG.E.64` **0→18**。**结论**：B2 生产核里**所有** store 至此都已是向量化/已评估形态；
      **方法学（新）**：同叫「标量 store」危害可以相反——先看**线程→地址映射**（是否连续）、再看核是
      **扇区 bound** 还是**发射 bound**。**边界**：只测到 `prod_hdw`（非 TMA/非 INTRA 折叠），未与
      `pci*` 整链、`-DB2M_TMA` 交叉重测（机制同、store 与 stage 布局无关）；是研究形态内部免费收尾、
      不改变 §45 的 `cf` 默认部署。**该边界已由 P7-71 关闭**（八变体三向交叉、孪生逐位、链级 1–3%）。
- [x] **P7-71** 关闭 §73.6（P7-70）的边界：把 `-DB2M_VECST` 落进**完整反向链**、并与 TMA / DAMMA
      **三向交叉**。八份 fold 内核 = `{scalar DA, mma DA} × {非 TMA, TMA} × {标量 store, VECST}`
      （全 `-DB2M_PROD -DB2M_HDW -DB2M_INTRA`），各链成 `dAv→CUDA-B1→CUDA-B2-prod_intra→反向 cumsum`，
      只吃 FlashKDA 16-chunk K1 workspace。新增 `harness/check_bwd_b2_vecst_e2e.py`
      （`harness/check_bwd_b2_vecst_e2e.out.txt`，空闲卡）+ `docs/07` §74。**Part A**：四对孪生
      （`pci_vec/pci`、`pci_tma_vec/pci_tma`、`pci_mma_vec/pci_mma`、`pci_mma_tma_vec/pci_mma_tma`）
      六项梯度**全部逐位相同（`max_abs=0`，4 组含非整除 `T=250`/`B=2`）**；八变体 vs FLA worst-case
      `max_rel` **逐值完全同 §72**（`dq 3.949e-3…dh0 8.889e-3`）。**Part B**：链 `dev` VECST
      **0.968–1.003×**（`B2`/`B8`/`T250` 落 0.968–0.995、最小 shape `T512` 噪声内），且**非 TMA > TMA**（`pci*` 0.968–0.998 vs `pci_tma*` 0.985–1.003，
      因 TMA 版把访存压力从 LSU 发射挪走、store 本就不是其最紧项）；`pci_tma/pci` 1.002–1.027、
      `pci_mma_tma/pci_mma` 0.976–0.987（DAMMA 把 TMA 反号为正，兑现 §71/§72）、`cf/pci` 0.67–0.78。
      **结论**：边界关闭——B2 生产核所有 store 变体（片段跨步/相邻列）在**隔离核与整链**两处都核验
      逐位无损；VECST 是研究形态内部免费收尾，**不改变 §45 的 `cf` 默认部署**。**边界**：仅 `BT=16`、
      `K=V=128`、fp32 操作数。
- [x] **P7-59** 收口 §51.6「向量化 store 一律做」在 **B2 非 GEMM 四段**上留下的边界（P7-52/53/56/57 每轮
      边界都写着「本段 store 未接 `-DB2M_VECST`」）：给 `b2_mma_probe.cu` 加一个 `st2()` 成对 store 助手
      （`:96`；`B2M_VECST` 下一条 `STG.E.64`，否则两条标量），落到输出支路 `dv2/dvb`（`:934-940`）、
      `dkgb`（`:952-953`）、门控链 `dqg/dkg`（先写 `dqgv/dkgv[2][4]`，循环后成对存 `:1011-1014`）与 `dg`
      （`:1054-1055`）——**算术一字未动**。新增 `harness/check_bwd_b2_vecst_seg.py`
      （`harness/check_bwd_b2_vecst_seg.out.txt`，空闲卡）。**Part A**：四段 base vs vecst 同源对照，四组
      shape（含非整除 `T=250`）**全部逐位相同（`max_abs=0`）**；门控 vecst vs FLA `dqg 1.051e-3/dkg 9.466e-4/
      dg 1.010e-3` **逐值同 P7-57**（tf32 底噪）。**Part B（关键）**：CUPTI `dev` vecst/base **16/16 全为正**
      ——小 shape（`T512/T250`）**0.835–0.919×**、大 shape（`B8T2048H8`）**0.983–0.990×**，`db` 段（一次
      向量化 `dv2/dvb/dkgb` 三个大张量）大 shape 也 **0.940×**。**机制**：`db` build 的 SASS `STG.E`
      **50→2**、`STG.E.64` 0→**24**（残留 2 条标量为 `dgk` 列归约与 `db_out` thread-per-row，非成对形态）。
      见 `docs/07` §62。**边界**：`[BT,BT]` 的 `dA`/`dApre` 未向量化（小张量 + thread-per-entry 布局）；
      仅 `K=V=128`、`BT=16`、fp32。至此 B2 全核（含四段非 GEMM）的 token-major store 全部 64-bit 向量化。
- [x] **P7-32** 兑现 §34.4 的**待验证**：把 B1 的 `dh` 落盘从逐标量 `st.global` 换成 **TMA store**，
      验证能否复现 §34.1「删掉 store」收益的那 33%。给 `b1_tma_probe.cu` 加 `-DB1T_TMADH`：per-warp
      `[KD,BW]=[128,16]` 未 padding tile 写入加进 `Shared` 的**双缓冲** `dho[2][NW][KD][BW]`，lane 0 发
      `cp.async.bulk.tensor.5d.global.shared::cta.bulk_group`（`make_map_dh`：dims `{VD,KD,H,NT,B}`、
      box `{BW,KD,1,1,1}`、`SWIZZLE_NONE`）+ `wait_group.read` 复用；稳态循环仍无 `__syncthreads()`。
      新增 `harness/check_bwd_b1_tma_store.py`（`check_bwd_b1_tma_store.out.txt`）+ `probe_b1_tma_store_ncu.sh` /
      `ncu_dh_roundtrip_driver.py --tma`（`probe_b1_tma_store_ncu.out.txt`）+ `docs/07` §35。**Part A**：16 组
      （`BT=32/64`、`NS=1/2`、含非整除 `T=250`）TMA vs scalar `dh/dv2/dh0` **逐位相同（max_abs=0）**、vs FLA 仍
      bf16 1 ulp。**Part B**：CUPTI 恢复占比 `(base−tma)/(base−nodh)` = **0.58–0.84（均值 ~0.72）**，tma/base
      0.40–0.78、base→tma 提速 **1.28–2.48×**。**Part D（ncu）**：`lg_throttle` **0.66→0.04** / **0.89→0.05**
      （落回 nodh 的 0.05/0.03）——**直接坐实 §34.1 的「逐标量 store 打满队列」就是真凶**；DRAM write
      **92.1→91.4 / 7.12→7.17 基本不变**（不是省带宽）、L2 字节 860→487 / 161→86（**−43%**，消掉标量 2 B
      store 的 ~3× 放大）；theoretical occ 腰斩 15.62→7.81% 但 achieved 7.77% 不动（再排除占用率）。
      **结论**：§34.4 答案是**能**——TMA store 在**不做任何融合**下吃掉那 33% 的大头；目标卡 CUDA 反向第一
      优先级＝B1 `dh`（及 `dv2`）改 TMA store（独立、低风险、逐位无损），B1→B2 融合净增量只剩 B2 侧 ~5%。
      **待验证**：`dv2` 的 TMA store 推广未实测（本节只改 `dh`，与 §34.1 口径一致）。

- [x] **P7-33** 兑现 §35.5 的**待验证**：把 TMA store 从 `dh` 推广到同一逐标量循环里的 **`dv2`**（输出支路）。
      给 `b1_tma_probe.cu` 加 `-DB1T_TMADV`：per-warp `[BT,BW]` 未 padding tile 写进 `Shared` 的双缓冲
      `duo[2][NW][BT][BW]`，lane 0 发一条 `cp.async.bulk.tensor.4d`（`make_map_dv2`：dims `{VD,T,H,B}`、
      box `{BW,BT,1,1}`、`SWIZZLE_NONE`），OOB 尾行由 TMA 按 `T` 边界自动裁（免显式 `tr<T`）；`dh`/`dv2`
      **共用一个 bulk group**（每块一次 commit，`wait_group.read 1` 语义不变）。另加 `-DB1T_NODV2` 消融。
      新增 `harness/check_bwd_b1_tma_dv2.py`（`check_bwd_b1_tma_dv2.out.txt`）+ `ncu_b1_tma_dv2_driver.py` /
      `probe_b1_tma_dv2_ncu.sh`（`probe_b1_tma_dv2_ncu.out.txt`）+ `docs/07` §36。**Part A**：16 组
      （`BT=32/64`、`NS=1/2`、含非整除 `T=250`）tma2 vs scalar `dh/dv2/dh0` **逐位相同（max_abs=0）**、vs FLA
      仍 bf16 1 ulp。**Part B**：`tma2/tma` = **0.84–0.95（均值 ~0.89，16/16 组为正）**、`tma2/base` =
      **0.39–0.71（均值 0.58）**、对「两条全删」上界恢复 **0.62–0.83（均值 0.75）**；`base−nodv2` 只 7–16%
      （证实 `dv2` 是「小块」）。**Part D（ncu）**：`lg_throttle` **0.66→0.04 / 0.90→0.07**（nodv2 列仍 0.83/0.92，
      证明 `dv2` 标量写也是反压源）、DRAM write **91.6→91.2 / 7.49→7.14**（不是带宽）、L2 字节 **tma 再 −15%**
      （490.7→416.2 / 85.8→75.5，−52% vs base）、tensor **15.48/16.89%**。**坑**：4D 张量图 `globalStrides`
      按 **dim 从 1** 起算，首版把 T/H 两维 stride 写反——`H==1` 全对、只有 `H>1` 时 `dv2` 错（`dh/dh0` 仍逐位），
      指纹=「仅 `dv2` 错且仅多 head 错」。**结论**：§35.5 答案=**能且逐位无损**——`dh`+`dv2` 全改 TMA 后
       `lg_throttle` 归零、恢复全删上界 75%，B1 下一块瓶颈只剩每 warp 发射效率（`wait` 1.40 / tensor 15.5%）。
- [x] **P7-34** 复核 `docs/07` 里**两个 FLA `dhu` baseline**（同一条
      `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`，`B2T1024H8` 一处报 ~23µs、一处报 ~137µs）
      并更正 §25。新增 `harness/check_bwd_fla_dhu_baseline.py`（`..._baseline.out.txt`，空闲卡）+
      `docs/07` §37。**根因＝operand dtype 而非 direct/chain**：直接调用 FLA `dhu` bf16 = **13.1/23.3/120.0µs**、
      fp32 = **53.0/127.6/525.4µs**（`B1T512H4/B2T1024H8/B8T2048H8`，fp32/bf16 = **4.0–5.5×**）；链内 fp32
      `dhu`（54.0/137.5/571.8）与直接 fp32 逐一对上、改 bf16 重跑同链即回 ~23µs。机制＝fp32 载入 2× 字节
      + `tl.dot` 落 **tf32**（`chunk_delta_h.py:359-671` 的操作数都 `.to(b_k.dtype)`）。整条 FLA 反向链
      **bf16 dev 只有 fp32 的 0.54–0.58×**。**更正 §25**：与 FLA **matched bf16** 下，部署粒度 BT=16 的
      bf16 B1（NS3 34.9/61.9/318.9）比 FLA `dhu` **慢 2.3–2.5×**（非旧写的反超 1.46–2.2×）、整链 `dev/FLA`
      **0.86–0.95×**（非 0.48–0.54×）——与 §22–24 CUDA 原型的 1.7–3.8× 自洽（差距＝**16 块粒度**，孤立
      BT=64 时 ours≈FLA）。诚实定位＝**与 FLA 反向同档（~0.9×）且省掉 FLA 的前向状态重算**；工程重点
      （B1 粒度 + store、B2/B3 融合）不变。**注意**：§10/§11 当时我方 B1 亦为 fp32（fair）、§17–24/§34–36
      两侧均 bf16（fair），只有 §25 跨 dtype。
- [x] **P7-35** 把「K2 形状 CUDA B1」的累计收益**收口成 matched-bf16 的 `xFLA`**（§22–§24 报过、
      §35/§36 的 store 轮只报相对收益），并在加 `BT=16` 覆盖时**发现手写 pipeline 的竞态**。新增
      `harness/check_bwd_b1_xfla.py`（`check_bwd_b1_xfla.out.txt`，空闲卡）+ `docs/07` §38。三个变体
      `base`/`tma`/`tma2`（=ETMA/+TMADH/+TMADV）。**Part A**：`tma2` vs `base` `BT=32/64 NS=1/2`（含非整除
      `T=250`）**逐位相同**、vs 直接 bf16 FLA `dhu` 仍 1 ulp。**Part B/C（关键）**：`xFLA`（ours/FLA bf16，
      同 `BT=64`）——大 shape `B8T2048H8` **1.15（NS=1）/0.83（NS=2）**（§24 标量 store 时 **1.63**，即 TMA
      store 兑现「大 shape 追平 FLA」）、小 shape（`B1T512H4/B2T1024H8/B1T250H3`）仍 **2.0–2.7×**（残差＝
      每 warp 发射效率，非 store）；**BT=16 无法在探针测 `xFLA`**（FLA 前向 `chunk_kda_fwd_intra` 只支持
      chunk 32/64，直接 `BT=16` 报 `ValueError`），部署粒度口径仍归 §37.3。**Part D（踩坑/新发现）**：
      run-to-run 确定性扫描暴露 **`base` kernel 在 `NSTAGE≥2` 的大 shape 上不可复现**（`BT=16 NS=2` 3.64e-2、
      `BT=32 NS=3` 4.52e-3），而 **`NS=1` 在所有 `BT` 稳定**——即 §35/§36 的「16 组逐位相同」是**观测到而非
      有保证**；根因（手写 `full`/`empty` parity 在 `NS≥2` 的覆写窗口）**待验证**。**回授**：§24「最优回到
      `NS=1`」同时是**唯一有保证正确**的配置，多 stage 收益须换 CUTLASS `PipelineTmaAsync` 才安全。
      **新方法学**：多 stage 手写流水必须做 run-to-run 确定性检查（与 §25.5 归因窗口、§37 dtype 对齐并列）。
- [x] **P7-36** 定位并**修掉** P7-35 遗留的 `NSTAGE≥2` run-to-run 竞态（§38.3 只说「parity 待验证」）。
      新增 `harness/check_bwd_b1_race.py`（`check_bwd_b1_race.out.txt`，空闲卡）+ `docs/07` §39；
      探针加 `-DB1T_EPROXY`（`fence.proxy.async.shared::cta`，`b1_tma_probe.cu:59-61`）与
      `_PRO`/`_CON`/`_ECOPY` 变体。**Part A（9 变体 run-to-run 扫描，`B8T2048H8 NR=40`）**：`base` 在
      `BT=16 NS=2`=39/40、`BT=32 NS=3`=19/40 racy 且**每次扫描计数都在抖**；`noetma`/`dropread`
      （不碰复用的 `st->e`）=0，`proxy`/`pro`/`con`=`fence.proxy.async`（两侧任一侧单独加都够）**全 0/40**，
      `esync`/`evol`**修不了**，`sync`（`__syncthreads`）能修但把流水折回串行。**Part B（程序化）**：
      racy `base` 29/59 run 失败，错误是**一整个 compute warp 的 16 列 V-slice**（`[0,16)`/`[64,80)`）
      跨**连续一段 chunk**（K-row 直方图最高在 `m=3` 的 56–61 行×29）＝**producer 用下一轮 TMA 覆写了
      该 warp 还在读的 stage 缓冲**（WAR）。**根因**：`full`/`empty` mbarrier 只在 **generic proxy** 内
      定序；consumer 普通 `ld` 读 `st->e`、producer `cp.async.bulk.tensor` 回填同一缓冲（**async proxy**），
      缺 `fence.proxy.async`。**Part C**：`proxy/base`=**0.96–1.02（噪声内，修竞态免费）**，而放弃 ETMA
      回读 global（`noetma`）要贵 **4–27%**。**Part D**：`proxy` `NS=2/3` vs `base` `NS=1` `dh`
      **逐位相同（max_abs=0，12 格）**→ 多 stage 数值不受影响。**撤回 §38.4 回授**：多 stage 安全**不需要**
      换 CUTLASS `PipelineTmaAsync`，一条 fence 即可。**边界**：本轮在标量 store 的 `-DB1T_ETMA` 上做，
      `+TMADH/+TMADV`（`tma2`）store 侧的复用握手同样建议补 fence（**待验证**）。
- [x] **P7-37** 关闭 §39.5 的 `tma2` **store 侧**待验证：`tma2`（`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV`）
      在输出侧新增一对 generic↔async——compute warp 用 generic store 写 `dh`/`dv2` tile 进 smem、
      `cp.async.bulk.tensor`（async proxy）读出去（**RAW**），与输入侧 `st->e` 的 WAR（§39）机制对称。
      加 `-DB1T_SPROXY`：在 `dh`（`b1_tma_probe.cu:402-403`）/`dv2`（`:473-474`）两处**写完 tile、
      `__syncwarp()` 之后、lane 0 发 store 之前**各插一条 `fence_proxy_async()`（全 32 lane 执行，
      同 CUTLASS `tma_store_fence`）。新增 `harness/check_bwd_b1_tma2_race.py`
      （`check_bwd_b1_tma2_race.out.txt`，空闲卡）+ `docs/07` §40。**Part A（关键/诚实）**：同一 kernel
      `NR=40` 确定性扫描（`B8T2048H8`）——`scalar` 三次大配置全 racy（39/40），而 **`tma2` 全族
      （含两侧都没 fence 的 `tma2`）全 `0/40`**：**输出侧 RAW/WAR 在本探针观测不到**（与 §35/§36「逐位
      相同」的观测一致；推断是 `cp.async.bulk` store 的节奏把 §39 输入侧覆写窗口拱出敏感区）。**但
      「观测 clean≠有保证 clean」**（§38.3 教训），故部署仍取两侧 fence。**Part B**：`tma2_full`
      vs `scalar_p` 16 格 `dh/dv2/dh0` **逐位相同（max_abs=0，含非整除 T=250）**、vs 直接 bf16 FLA
      `dhu` 仍 1 ulp（4.88e-4/2.44e-4）——fence 纯 ordering、不改数值。**Part C**：`full/tma2` =
      **0.997–1.081（噪声内）**，store 侧 fence **免费**；`scalar_p/tma2_full` 仍 1.42–2.93（§35/§36
      TMA store 收益不受影响）。**Part D**：`tma2_full` `NS=2/3` vs `NS=1` `dh` **逐位相同**。**结论**：
      部署配置＝`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY -DB1T_SPROXY`、`NSTAGE` 可 >1；
      「输入/输出两个方向的 TMA 复用缓冲都需 `fence.proxy.async`」收口。**唯一剩余边界（待验证）**：
      更深 pipeline / 不同 store tile 形状改变写→store 时序时，输出侧可观测性可能变化，届时仍以
      `tma2_full` 为准。

- [x] **P7-38** 关闭 P7-37（§40.6）留下的**唯一边界**：把 pipeline 加深、store 时序推极端，看
      输出侧（及 TMA 形状下输入侧）`fence.proxy.async` 竞态是否变得可观测。给 `b1_tma_probe.cu`
      新增 `NSTAGE=4/5/6` 实例化（BT=16/32）与输出缓冲深度宏 `B1T_SDEPTH`（`dho`/`duo` 按
      `i % SDEPTH` 复用、`cp.async.bulk.wait_group.read SDEPTH-1` 释放；`SDEPTH=1` 单缓冲极端）。新增
      `harness/check_bwd_b1_deep_ns.py`（`check_bwd_b1_deep_ns.out.txt`，空闲卡）+ `docs/07` §41。
      **Part A（关键）**：`NR=80` 确定性扫描 `B8T2048H8`，覆盖 `BT`×`NS`（16:1–6/32:1–4）×
      `SDEPTH∈{1,2,3,4}` × 4 变体——标量对照 `scalar` 稳定复现输入侧竞态（`BT16 NS2/3/4`=**79/80**、
      `BT32 NS3`=**79/80**，`NS=1` 免疫），而 **`tma2` 全族（含两侧都没 fence 的）在全部 60 个组合上
      **0/80**。**答案：更深 pipeline / 不同 store 时序仍观测不到**（扫有功率，故非盲点）。细节：
      `scalar` 的 racy 概率随 `NS` **反降**（`NS=2/3/4`79/80→`NS=5`18/80→`NS=6`6/80）＝强时序相关，
      单一配置的 clean 不能外推。**Part B**：`tma2_full` 深配置 vs `scalar_p` `dh/dv2/dh0` **逐位
      相同（含非整除 T=250）**、vs bf16 FLA `dhu` `BT=64` 仍 1 ulp。**Part C（新正面数据）**：TMA
      store + `BT=16/32` 上**深 pipeline 有正收益**——`B8T2048H8` BT16 `NS=1→4` 300.7→177.8µs
      （1.69×）、BT32 `NS=1→3` 203.4→124.6µs（1.63×）；`SDEPTH` 非旋钮（≤3%）。**Part D**：深
      `NS/SDEPTH` vs `NS=1` `dh` 逐位相同。**对目标卡的设计输入**(1)两侧 fence 保留（免费、逐位
      无损、本探针观测不到，无需再留不确定项）；(2)`NSTAGE` 建议 ≥4（**修正 §24「回到 NS=1」**，那是
      BT=64/标量 store 的结论）。**方法学**：clean 只在被扫过的配置上成立；性能旋钮最优值依赖
      store 路径 + BT。

- [x] **P7-39** 第一次把 K2 形状的**手写 CUDA B1 落进完整反向链**（此前 §15–§41 全是孤立
      kernel vs kernel，组装链的 B1 一直是 Triton）。新增 `harness/check_bwd_b1_cuda_e2e.py`
      （`check_bwd_b1_cuda_e2e.out.txt`，空闲卡）+ `docs/07` §42。**接口**：给 `b1_tma_probe.cu`
      补 `dht`（终态伴随）入口——kernel 参数 + 寄存器状态按 [K,V] 片段布局初始化
      （`b1_tma_probe.cu:258/341-357`），新增独立符号 `b1t_launch_dht`（`:739`）、`b1t_launch`
      仍传 `nullptr`，语义对齐 Triton B1 的 `HAS_DHT`。**Part A（drop-in）**：同一份 workspace
      操作数、`dht=dfin≠0`，CUDA-B1 vs Triton-B1 `dh/dv2/dh0` `max_abs` 1.4e-5~3.7e-4
      （=**1 bf16 ulp**，CUDA 落盘 bf16）；去掉 `dht` 后 `dh` 差 3.8e-2~4.5e-2（证明入口生效）。
      **Part B（链级正确性）**：CUDA-B1 链（`BT=16`、`NSTAGE=4`）vs FLA autograd 六项最坏
      `dq 3.423e-3/dk 3.308e-3/dv 6.795e-3/dbeta 6.012e-3/dg 7.064e-3/dh0 9.365e-3`，与
      Triton-B1 链**同阶**（`dq`/`dh0` 两链**逐位相同**、其余 ≤2.3e-3），全落 bf16 底噪；6 组含
      非整除 T=250/513、g=0、B=2。**Part C（链级 wall）**：CUDA-B1 链 0.94–1.03× Triton-B1 链
      （`B8T2048H8` **1.5→0.94×** 快 6%、`T512H4` 1.03×、`B2T1024H8` 1.02×）；`T250H3` 1.65×
      回归＝host 侧 `launch_impl` **每次重编码张量图**（`b1_tma_probe.cu:660-684`）打在小 shape
      host-bound 链上，**待验证**（生产须一次编码复用）。**Part D（部署粒度 `BT=16` 单核）**：
      event-pair wall（孤立 µs 级 kernel 的 CUPTI 归因本机**双峰** ~41/130µs，故弃用）——小 shape
      CUDA 与 Triton 同档/略慢（1.05–1.63×），`B8T2048H8` 快 **1.9×**（0.53×）；「手写 CUDA 是否
      比 Triton 快」**依赖 shape**。**三坑**：(a) 早期把 CUDA 的 bf16 `dh`/`dv2` 整体 `.float()`
      喂 B2 = 凭空 elementwise 税（`B8T2048H8` 链 wall 8600 vs 7047），应让 B2 直接吃 bf16 `dh`、
      只有进 `tl.dot` 的 `dv` 需 fp32；(b) launch 每次重编码张量图（见上）；(c) 孤立 µs 级 kernel
      的 CUPTI 归因双峰。**结论**：目标卡 CUDA 反向**第一次端到端可跑**（B1=CUDA、其余 Triton），
      drop-in 数值等价、链级不拖慢；链瓶颈仍是 B2/B3（§26），路线「先接受 Triton、CUDA 逐件替换」成立。

- [x] **P7-40** 关闭 P7-39（§42.4/§42.6 坑 b）留下的 host 税**待验证**：把「CUDA-B1 链在
      `T=250 H=3` 上的 1.65× 回归＝`launch_impl` 每次重编码张量图」这个假设直接测掉。给
      `b1_tma_probe.cu` 加 **`-DB1T_MAPCACHE`**（`MapSet`+`encode_maps`，`b1_tma_probe.cu:661-687`；
      指针/shape 全不变的 encode-once 复用，`:700-714`）+ 探测符号 `b1t_encode_only`（只编码不 launch，
      `:788`）/ `b1t_mapcache_miss`（`:796`）。新增 `harness/check_bwd_b1_launch_tax.py`
      （`harness/check_bwd_b1_launch_tax.out.txt`，空闲卡）+ `docs/07` §43。**Part A**：纯编码
      **3.8–4.0µs/次**（8 张图），但加进 launch 后（队列反压主导）cache 只省 **0.15–0.75µs**
      ⇒ 编码不是那笔税；cache 命中率 **199/200**。**Part B**：cache vs no-cache `dh/dh0/dv2`
      **逐位相同（max_abs=0，含非整除 T=250/513）**。**Part C（关键）**：空闲卡重跑链 wall，
      `T=250 H=3` 实为 **1.049×**（非 1.65×），`T512H4` 1.051×、`B8T2048H8` 0.951×；
      且 §42.4 的 Triton baseline 245.9µs 是**低离群**（本轮同链测到 308/323/386µs，跨轮 ±25%）。
      **结论**：**撤回**「张量图重编码＝host 税」的归因，1.65× 不可复现（基线采样产物）；一次编码复用
      免费、逐位、生产该做，但**不解决任何已测性能问题**，不再当优化项。**方法学（新）**：`process_time`
      在 launch 队列打满时读的是 **GPU 吞吐速率**、非纯 CPU；隔离纯 CPU 须用**不发 launch** 的入口。
      见 `docs/07` §43。

- [x] **P7-41** 兑现 §24.5/§36 的「换 CUTLASS `PipelineTmaAsync`」（K2 形状 CUDA B1 的最后一条
      工程化选项），结论是**空杠杆**：给 `b1_tma_probe.cu` 加 `-DB1T_CUTLASS`，**只**把手写
      `full`/`empty` mbarrier 握手换成 `cutlass::PipelineTmaAsync<NSTAGE>`（TMA 拷贝、swizzle、
      `mma.sync`、`dh`/`dv2` TMA store、`fence.proxy.async` 一字未改）。新增
      `harness/check_bwd_b1_cutlass.py`（`..._cutlass.out.txt`，空闲卡）+ `harness/ncu_b1_cutlass_driver.py` /
      `probe_b1_cutlass_ncu.sh`（`..._cutlass_ncu.out.txt`）+ `docs/07` §44。**Part A**：四变体
      （base/hnd/cut/cut2）16 格（`BT=32/64`、`NS=1/2`、含非整除 `T=250`）——`cut` vs `base`、
      `cut2` vs `hnd`（=`tma2_full`）**逐位相同（`dh/dv2/dh0` max_abs=0）**、vs bf16 FLA `dhu` 仍 1 ulp。
      **Part B（关键）**：`cut2/hnd = 0.985–1.003`（16/16 噪声内，均值 ~0.995）——**换 CUTLASS 一点没变快**，
      §24.5 的残差（`wait`/`short_sb`）不是 pipeline 协议实现开销。**Part C**：`base`（未加 fence 的手写
      pipeline）在 `B8T2048H8` `BT16/32 NS=3` 仍 racy（3.76e-2/1.57e-3），而 `hnd`/`cut2` 全 18 格 run-to-run
      **0.00e+00**、`cut2` vs `hnd` 逐位 0——CUTLASS 的 phase/parity 记账**天然**避开手写 off-by-one（§36 靠
      fence 修掉）。**Part D（ncu）**：`BT=64 NS=2` 逐指标重合（B8 Duration 99.04→98.82、`wait` 1.42→1.38、
      `short_sb` 0.61/0.63、`tensor` 22.29/22.37、regs 165→164、smem 同、occ 7.77/7.79）。**结论**：§24.5 的
      选项 (a) **关闭为空结论**——手写 mbarrier 已到该结构上限；选项 (b)「直接接受 Triton 版」是正确路线；
      CUTLASS 的价值在**稳健/可维护**（天然确定、代码更短），不在性能。目标卡剩余投入应去**链级 B2/B3**。

- [x] **P7-42** 把两条**分别收敛**的反向优化**合成一条链**（此前从未同时出现）：CUDA B1 只在 §42 落
      过链（那条链用**未融合** B2/B3），B2/B3 融合只在 §33 落过链（那条链用 **Triton** pipe B1）。新增
      `harness/check_bwd_cudab1_fused_e2e.py`（`harness/check_bwd_cudab1_fused_e2e.out.txt`，空闲卡）+
      `docs/07` §45。拼成 `dAv -> CUDA B1(dht,NSTAGE) -> fused B2+B3 -> reverse cumsum`（三变体
      `cu`=CUDA+未融合、`fu`=Triton+融合、`cf`=CUDA+融合）。**Part A**：`cf` vs FLA 六项 worst-case
      `dq 3.417e-3/dk 3.301e-3/dv 6.795e-3/dbeta 6.012e-3/dg 7.054e-3/dh0 9.365e-3` = 与 §25/§26/§42
      **逐值一致**（tf32/bf16 底噪）；位级签名证实归因——`cf` vs `fu` 的 `dq`/`dh0`、`cf` vs `cu` 的
      `dv`/`dh0` **逐位相同（max_abs=0）**（`dv/dh0` 由 B1 定、`dq/dk/dg/dbeta` 由 B2/B3 形式定）。
      **Part B（关键）**：(1) **两条优化可加**——`cf` 在**所有** shape 优于 `cu`（`cf/cu` **0.67–0.71×**，
      融合在 CUDA B1 之上再砍 ~30%，非零和）；(2) **但最优 B1 随 shape 交叉**——`cf/fu` 只在大 shape
      取胜（`B8T2048H8` **0.91×**，dev 1537→1402），小 shape **1.05–1.16× 略慢**，根因＝B1 单核头对头
      （CUDA B1@BT=16 小 grid 40.8/77.9/22.8 vs Triton 34.9/61.7/19.0，大 shape 才 184.2 vs 317.9=1.73×）；
      (3) `NSTAGE≥3` 最优（`cf1` 明显最慢）。两条路线 `dev/FLA` 都只 **0.31–0.37×**（整链反向比 FLA 反向
      dev 快 ~2.7–3.2×，且不含其前向重算）。**结论**：目标卡 CUDA 反向最终形态＝`dAv→B1→融合 B2/B3→
      反向 cumsum`，**B1 按 grid 切换实现、B2/B3 一律融合**；合成实验（本轮新增方法学）证明可加、并暴露
      单链看不到的「B1 随 shape 切换」部署约束；剩余大头＝把融合 B2/B3 从 Triton 搬成 K2 形状 CUDA。
- [x] **P7-43** 动「K2 形状 CUDA 化」前先给 go/no-go 与收益上界（**不建 CUDA 核**）：新增
      `harness/check_bwd_b2b3_bound.py`（`harness/check_bwd_b2b3_bound.out.txt`，空闲卡）+
      `harness/probe_b2b3_bound_ncu.sh`（`harness/probe_b2b3_bound_ncu.out.txt`）+ `docs/07` §46。
      **Part A**（deploy 基线）：`B8T2048H8` 融合核 `dev 1003µs`、`regs 255/spills 34/smem 114KB`、
      占用率 **1 block/SM（12.5%）被 regs 与 smem 同时钉死**。**Part B**（`maxnreg` 受控扫 255→80）：
      占用率**一动不动**（smem 卡住），每档都换算成线性 spill（34→320）与变慢（最坏 **3.05×**）
      ——「降寄存器腾占用率」在 Triton 走不通。**Part C（关键）**：同指针/掩码/字节、**无 `tl.dot`** 的
      traffic 等价核（0 spill、0 smem、3 block/SM、带宽饱和 ~3.0 TB/s）**509µs**，融合核 **988µs**
      高它 **1.94×**（`T512H4` 2.43×）；ncu 字节口径地板 586µs（1.79×）互证。**Part D（ncu）**：
      Active/Eligible Warps per Scheduler **1.99 / 0.18**、issue 0.14/cyc（每 7 周期一条）、
      `long_sb` 5.80=41.7%、`Block Limit Registers=1 且 Shared Mem=1`——**发射饥饿**（不是带宽，
      不是占用率旋钮）。**结论**：融合 B2/B3 是延迟受限、有 **~1.9×** 头寸，且该头寸**在 Triton 里够不着**
      （regs+smem 双钉、`BV=V` 无循环可流水）→ **「K2 形状 CUDA 化」被数据支持**（首位动作＝把
      `dq/dk/dw`/`dA` 累加器按 warp 分片常驻寄存器，TMA 流水其次）。**新方法学**：用「同字节、去 compute 的
       traffic 等价核」测地板，一次给出延迟受限判据 + 收益上界，比 ncu「Est. Speedup」启发式可靠
       （后者 K2/B1 两次误报，本轮 44% 也明确不可由降寄存器兑现）。
- [x] **P7-44** 兑现 §46.6 首位动作的**第一种读法（放弃 `tl.dot`、手写高占用 SIMT fp32）并以反面结论收口**：
       新增 `harness/b2_core_probe.cu`（`b2_core_launch`）+ `harness/check_bwd_b2_core.py`
       （`..._b2_core.out.txt`，空闲卡）+ `harness/probe_b2_core_ncu.sh` / `ncu_b2_core_driver.py`
       （`..._b2_core_ncu.out.txt`）+ `docs/07` §47。隔离 B2 的 **GEMM 核**（`dq/dk/dw/dgk` 四条 op，
       不含 finalize/`dA`/`dv2`/`db`/intra）。**设计**：block 256、thread 固定列 `kk=tid%K` + 8 行
       `tbase..tbase+7`，24 个 fp32 寄存器累加器、38KB smem、V 方向 4 tile。**Part A**：手写核 vs 真
       fp32 torch **`dq/dk/dw` 逐位相同（max_abs=0）**、`dgk` 5.2e-7（替身证明代数无误）；同 op Triton 核
       =tf32 底噪 8.4–9.8e-4。**Part B/C（关键，反面）**：手写 SIMT 核 `dev` **全面慢同 op Triton 核 3.11×
       （B8T2048H8 1727.7 vs 556.2）/2.78×/4.44×/5.20×**，仅 ~7.4 TFLOP/s（峰值 ~11%），
       **比完整 Triton B2 还慢 1.93×**；同 op Triton GEMM 核（556µs）**已是完整 B2（903µs）的 62%**。
       **Part D（ncu，两核字节完全相同作对照：读 1.28GB/写 201MB）**：手写核 **L1/TEX 89.6% bound**
       （`long_sb` 9.42=59%、FMA 14%、DRAM 25%、0.12% bank conflict、2 block/SM/24.8% occ）；Triton 核
       **DRAM 79.8% bound（2.67 TB/s）**、tensor 7.4%、1 block/SM/12.5% occ。机制＝SIMT 每 24 FMA 发 26
       次 shared 读（算术强度 ~0.9 op/word）→ 撞 L1；`tl.dot` 的 mma 把强度提高一量级 → 直接撞带宽。
       **结论（修正/收窄 §46.6）**：(1) **K2 形状端口必须保留 `ldmatrix`/`mma.sync` tensor core**——
       「用 SIMT 省掉 `tl.dot`」被显式证伪；(2) **GEMM 段不值得重写**（Triton 同 op 核 552µs vs 同字节
       HBM 地板 442µs 只 **1.25×**，头寸仅 25%）；(3) **§46 的 1.94× 重新定位到非 GEMM 段**（融合核比 GEMM 核
       多 0.48GB≈146µs 地板却多 436µs）**与 `h/dh` 流量**（fp32 1.07GB＝GEMM 核读的 84%）；(4) 动作顺序改为
       「非 GEMM 段寄存器常驻分片 + 同 CTA 交接 + 少搬 `h/dh`」先于 TMA 流水。**新方法学**：SIMT/tensor 决策
       先看**算术强度（op/word）**，`ncu` 的 L1/TEX SOL + `long_scoreboard` 是直接读数。**两坑**：
       `dgk` 输出错用 `KD*VD` large base（应 `((b*NT+c)*H+h)*K`，非法访存在下一次 triton 载入才冒头）；
       torch 参考须 `allow_tf32=False`（否则 cublas tf32 把逐位=0 污染成 3e-4）。
- [x] **P7-45** 兑现 §47.6 的显式缺口——把融合 B2/B3 的「非 GEMM 段 + `h/dh` 流量」从**差值推断**变成
       **受控字节消融**：同一份融合核上**只把选定的输入 `tl.load` 换成 `tl.zeros`**（`tl.dot` 全保留、
       store 全保留、程序形状不变），干净分开「字节」与「compute/发射」。新增
       `harness/check_bwd_b2b3_seg.py`（`harness/check_bwd_b2b3_seg.out.txt`，空闲卡）+
       `harness/probe_b2b3_seg_ncu.sh` / `ncu_b2b3_seg_driver.py`（`harness/probe_b2b3_seg_ncu.out.txt`）+
       `docs/07` §48。变体 `full`/`no_hdh`/`bf16_hdh`/`no_qkg`/`no_odv`/`no_all`。**Part A**：被门控的
       `full` 与未改动融合核 5 组 shape（含非整除 T=250/513、K=64）**逐位相同（max_abs=0）**。**Part B
       （`B8T2048H8`，1745MB）**：`full` 1051µs；`no_hdh`（去 1074MB）**590µs=0.56×**；`bf16_hdh`（h/dh
       字节砍半）**ncu 1.04ms＝与 full 持平（wash）**；`no_qkg`（只去 201MB）**762µs=0.73×**；`no_all`
       （所有大操作数不读）**170µs=0.16×**。**`ncu`（关键）**：`full` DRAM 只 **55.9% SOL**、tensor 6.21%、
       issue 0.14/cyc、255 regs/166KB smem/occ 12.49% → **延迟/发射受限**；`h/dh`＝**1.09GB＝DRAM 读的 55%**；
       `no_hdh` **occ 纹丝不动（12.46%）**、DRAM% 反降到 38% → 收益**既非占用率也非带宽**；`bf16_hdh` 省
       537MB 却 **Duration 不变**；`no_qkg` **regs 255→165** 才是不成比例加速的真因。**结论（收窄 §47.5）**：
       融合核头寸**既不是纯字节、也不是占用率，而是「大操作数（尤其 h/dh）常驻寄存器 → spill/发射饥饿」**；
       **`h/dh` bf16 不作为独立杠杆**（本实验为 wash，与 §29 上界 ~10% 自洽）；K2 形状端口首位动作＝
       **把大操作数从寄存器保活里拿掉、只留 warp 分片的小累加器**，判据补「**减的是字节还是寄存器保活？**」。
       **坑**：第一版「关 `tl.dot`＋秩-1 归约替身」触发 Triton 跨 lane 布局转换 → `no_post` **4964 spills
       /32 regs/慢 50×**（编译器病态，非测量），故改「只渡字节」；`bf16_hdh` 的 `dev/wall` 含 harness
       `h/dh.to(bf16)` 税（同 §26.5/§29.1），干净口径取 `ncu Duration`；`no_all` 的 `ncu` 记录为空（卡住）。
- [x] **P7-46** 兑现 §47.6/§48.4 的首位动作：在**保留 `mma.sync` tensor core** 前提下，把 B2 的 V-归约
       GEMM 核写成 **K2 形状**（`h/dh/do/vn/dv` 按 V 分块经 `cp.async` 流式喂 smem、寄存器里只剩每 warp
       `[16,16]` 的小累加器）。新增 `harness/b2_mma_probe.cu`（`b2mma_launch`，`mma.sync.m16n8k8.tf32`）+
       `harness/check_bwd_b2_mma.py`（`harness/check_bwd_b2_mma.out.txt`，空闲卡）+ `harness/probe_b2_mma_ncu.sh`
       / `harness/ncu_b2_mma_driver.py`（`harness/probe_b2_mma_ncu.out.txt`）+ `docs/07` §49。**Part A**：四量
       落 tf32 底噪（vs torch `dq 3.6e-4…dw 3.4e-4/dgk 1.6e-7`），且 CUDA(tf32) 比 Triton(tf32) 更贴 fp32。
       **Part B/C（关键/正面）**：同 op、同 fp32 操作数，`B8T2048H8` **CUDA 517µs vs Triton `dev` 554µs＝
       0.93×**（离 §47 地板 442µs **1.17×**，Triton 是 1.25×）、`B2T1024H8` 0.87×——**本专题第一个跑赢
       identical-op Triton 核的 CUDA 反向核**；小 shape（`T512H4`/`T250H3`，grid 128/48）反慢 1.4×（固定
       开销主导，核应随 shape 切换）。**`ncu`**：regs **226→70**、Block Limit Reg **1→3**、occ **12.4→24.6%**、
       `long_sb` 4.11→**3.09**、DRAM **79.9→84.6% SOL（2.68→2.84 TB/s）**、L2 成最高 pipe（88.8%）——
       坐实「移出大操作数常驻」是一阶量（§48.4），新核贴 DRAM 地板。**残差**：L2 字节比 Triton **高 35%**
       （cp.async 16B 粒度放大）+ 直接 smem 读 mma 片段致 **8.81M bank conflict**（Triton swizzle/`ldmatrix`
       为 0）→ 下一杠杆＝降 L2 流量/消冲突。**扫描**：`BVK/NS` 六档，大 shape 最优 `64/2`（0.90×）。**结论**：
       §48.4 首位动作方向正确且首次兑现正收益；K2 形状三要素（tensor core + 流式操作数 + 小累加器分片）
       被 §47 反例与本轮正例双向钉死；剩余头寸仍在非 GEMM 段，本核可作其 GEMM 段基座。

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
- **跨实现的 `xFLA` 必须对齐 dtype/chunk/归因窗口三者**：FLA `dhu` 的 fp32 operands 比 bf16 慢
  **4.0–5.5×**（fp32 载入 2× + `tl.dot` 落 tf32；`chunk_delta_h.py:359-671` 操作数 `.to(b_k.dtype)`），
  整条 FLA 反向链 bf16 只有 fp32 的 0.54–0.58×。`docs/07` §25 曾拿 **bf16 我方** 比 **fp32 FLA** 得出
  错的「反超」；正确口径见 §37（P7-34）。
- workspace 字段是 head-major `[H, total_tiles, CHUNK, D]`；展平到 FLA 的 token-major
  `[B,T,H,D]` 必须 `permute(0,1,3,2,4)`（按 `(NT,CHUNK,H)`），直接 reshape 会交错 head/chunk 轴
  （指纹：`qg/kg` 相对误差≈1.0）。`INV` 即 $P[o_t,o_A]$，不需转置。
- **手写 TMA 流水的「缓冲复用」必须显式 `fence.proxy.async`**（P7-36）：`full`/`empty` mbarrier
  只在 **generic proxy** 内定序，consumer 用普通 `ld` 读、producer 用 `cp.async.bulk.tensor`（**async
  proxy**）回填同一 stage 缓冲时，WAR 未定序 → 间歇性读到被覆写的字节。指纹＝**整段 warp V-slice
  错、跨连续 chunk、`NSTAGE≥2` 且 `NSTAGE=1` 免疫**；`__threadfence_block`/`volatile`/`__syncwarp`
  修不了，`fence.proxy.async.shared::cta`（`b1_tma_probe.cu:59-61`）修得掉且免费。
- **`fence.proxy.async` 要覆盖两个方向**（P7-37）：不止输入侧 `st->e` 的 WAR（consumer `ld` 读
  被 TMA 回填的缓冲，§39），输出侧的 `dh`/`dv2` tile 也有 **RAW**（generic store 写 tile →
  `cp.async.bulk.tensor...bulk_group` 读出去）——凡「generic 写/读」与「async proxy（TMA）读/写」
  共用同一块 smem，都要显式 fence。部署取 `-DB1T_EPROXY -DB1T_SPROXY`（代价噪声内、数值逐位不变）。
  诚实边界：输出侧 RAW 在 `X_FLASHKDA` 探针 **`NR=40` 观测不到**（`tma2` 全族 0/40），只有输入侧在
  标量 store 基线可复现——「观测 clean≠有保证 clean」（§38.3）。
- **竞态的「racy/clean」是强时序相关的**（P7-38，§41.2）：标量基线 racy 概率随 `NS` **反降**
  （`BT16 NS=2/3/4`≈79/80 → `NS=5` 18/80 → `NS=6` 6/80），且 `tma2` 全族在 `NS≤6`/`SDEPTH≤4`
  仍 0/80。故 **clean 只在被扫过的配置上成立、不可外推**；确定性扫描要 `NR` 足够大 + 覆盖部署最强
  变体本身。
- **`NSTAGE` 最优值依赖 store 路径 + BT**（P7-38，§41.4）：§24 在 BT=64/标量 `dh` store 上得
  「最优回到 `NS=1`」；换成 TMA store + `BT=16/32` 后 **深 pipeline 有正收益**（`B8T2048H8`
  1.6–1.7×，`NS=4` 附近最优）。跨配置不可外推，须实测。
- **`process_time` 测「host 税」在 launch 队列打满时读的是 GPU 吞吐速率、非纯 CPU**（P7-40，§43.2）：
  想把 host 端一项成本（如 `cuTensorMapEncodeTiled`）单独称出来，必须用**不发 launch**的入口
  （本专题的 `b1t_encode_only`）；否则编码等 CPU 工作被藏在队列等待里（纯编码 3.8–4µs 加进 launch
  后只看得见 ≤0.75µs）。跨实现的「小 shape host 回归」先做 run-to-run 复现（P7-40 的 1.65× 即
  Triton   基线抽到低离群的产物，空闲卡实为 ~1.05×）。
- **判 SIMT vs tensor 决策先看「算术强度 op/word」**（P7-44，§47.4）：手写 SIMT fp32 GEMM 每做 24 个 FMA
  要发 26 次 shared 读（强度 ~0.9 op/word）→ **必撞 L1/TEX**（实测 89.6% SOL bound，DRAM 只 25%），
  比同 op 的 `tl.dot`（mma 把强度抬高一个量级、直接 DRAM bound）慢 3–5×。`ncu` 的 L1/TEX SOL +
  `long_scoreboard` 是这条判据的直接读数；「寄存器常驻 + 高占用」**不能**替代 tensor core 的算术强度。
- **torch 参考做对拍前先关 `torch.backends.cuda.matmul.allow_tf32=False`**（P7-44）：默认 cublas 走 tf32，
  会把「手写 fp32 核 vs torch」从**逐位=0** 污染成 ~3e-4（看起来像 bug）；而 `tl.dot` 默认就是 tf32，
  与 torch-enable-tf32 同底噪。
- **Triton 里「关掉一个 `tl.dot` 但保字节」不能用秩-1 归约替身**（P7-45）：`tl.sum(X,1)[:,None]*tl.sum(Y,0)`
  要读同样操作数，但 Triton 为这些跨 lane 归约做**布局转换**（MMA layout ⇄ blocked），实测 `no_post`
  变体 **4964 spills / 32 regs / 慢 50×**——是**编译器病态**、不是测量。要分离「字节 vs compute」应改用
  **只渡输入字节**（`tl.load` → `tl.zeros`，`tl.dot` 全留），既不扰 compute 图也不触发 DCE。
- **同一个「h/dh 换 bf16」实验，`wall`/CUPTI `dev` 与 `ncu Duration` 会差一个数量级**（P7-45）：若
  harness 在计时闭包里做 `.to(bf16)`（本专题 `check_bwd_b2b3_seg.py`、§26.5/§29.1 亦然），两个 elementwise
  kernel 被算进 `dev`，把「wash」夸大成「1.49× 慢」。**核内改 dtype 的干净口径一律取 `ncu Duration`**。
- **tf32 的 mma 片段没有 16-bit `ldmatrix` 可走**（P7-46）：`mma.sync.*.tf32` 的操作数是 32-bit，
  `ldmatrix` 只支持 `.b16`，故手写 tf32 K2 核只能**直接从 smem 按 fragment 布局读**——本专题
  `b2_mma_probe.cu` 因此有 **8.81M shared-load bank conflict**（Triton 的 swizzle+`ldmatrix` 为 0）。
  要消它须走 CUTLASS 式手工 smem 重排（或改 bf16 用 `ldmatrix`），不能只靠 padding（§20 已证 padding
  反噬占用率）。
- **`cp.async` 的 L2 读放大是 `.BYPASS` 路径属性、不是 16B 宽度**（P7-48，§51.4；更正 §49.6/P7-46 的
  旧归因）：`b2_mma_probe.cu` 的 `LDGSTS.E.BYPASS.128` 读路径比**同 16B、同访问模式**的普通 `float4`
  `ld.global` 多 **1.5×** L2 读扇区（`h/dh`、`do/vn/dv` 各 1.5×；L1TEX 计 18 vs 16 扇区/请求、L2 srcunit
  计 24 vs 16），DRAM 读两侧不变（多出的是 L2 re-touch）。**故「放宽 cp.async 到 32B/128B」是错的**——
  16B 已是 cp.async 上限且宽度无关；也不能换成普通 load（丢流水 **+21%**）。要同时「大块搬运 + 深流水」
  只能上 **TMA**（`cp.async.bulk.tensor`）。
- **逐标量 32-bit `STG.E` 让 L2 写字节翻倍**（P7-48，§51.4）：4 个 lane 各写 4B 到同一 sector 的
  `{0,8,16,24}` 偏移 → **每次 store 指令只覆盖半个 sector**，l1tex st 扇区 = 数据扇区的 **1.98×**。
  合成 **64-bit `float2` store**（`b2_mma_probe.cu` 的 `-DB2M_VECST`）后立刻回到 1.0×、与 Triton 逐位。
  与 B1 §35/§36「逐标量 store 打满队列」同课——**store 一律向量化**（本专题多次踩到）。
- **「L2 字节」不是唯一指标，省字节要落最紧 pipe**（P7-48，§51.6）：B2 K2-mma 核部署 shape 是
  **DRAM 84.7% + L2 85–89% 双 bound**，`vecst` 消掉写放大后 L2 反降到 85%（不再最紧）；与 §50 的
  `dgk`/冲突「省了却不兑现时间」同构。改访存前先看 ncu 哪个 pipe 最高。
- **同一笔「读放大」用普通 load 消是净亏、用 TMA 消是净赚**（P7-49，§52.3）：`cp.async` `.BYPASS` 的
  1.5× L2 读放大，`noasync`（换同步 `float4`）能消掉但**−21%**（丢异步预取）；`cp.async.bulk.tensor`（TMA）
  消掉同样多（59.77M→39.85M 扇区）+ **净 +5%**——因为 TMA 保留异步、且走独立引擎（l1tex 全局读请求=0）。
  **测「换访存机制」不能只看字节，要同口径比机制本身的开销**。
- **同核「先算后写、再从 global 读回」的中间操作数，输入与输出缓冲不能共用**（P7-56，§59.6）：`b2_mma_probe.cu`
  的 DKGB 段吃 global `dw`（`=dv@h`，由本核同批算出），harness 起初把 `dw` 输入指针与 `dw_out` 输出指针设成
  **同一张量**——流水/落盘覆写致 `dkgb` 大面积错（`max_rel≈2`）、`db` 随之错，且**只对照 dkgb 专用 build 才定位**
  （对自身 torch 参考比是「循环自洽」、查不出）。改独立 `dw_out` 缓冲后与 §56 专用 build **逐位相同**。凡原型里
  有「中间量走 global round-trip」的两趟契约，对拍必须用**独立缓冲**，且**与能独立核验的上一刀交叉比**。
- **ncu 里的「散读 sector 放大」不一定是端到端头寸**（P7-58，§61.4；更正 §60.4 的归因）：门控链的
  `q/k/g` 片元式散读在 l1tex 全局读 sector 上是 **9.2×**，但把它换成合流 smem staging（`-DB2M_QKG`，
  global sector **−34%**）后端到端**反慢 +8.7%**（大 shape）——因为本核 **DRAM/L2 双 bound**，那 1.3 GB
  L1 散读**被 L2 吸收**（DRAM 读不变），删它不动瓶颈；staging 多一遍 smem 写+同步是净负。**判据**：把
  ncu 计数当成本前，须用「换算子**只改取数来源**、occupancy 不变」的受控对拍确认它在关键路径上。另本核
  加常驻 `[BT,KD]` tile 直接把 dyn smem 顶过 occupancy 阈值（非 TMA 92.7→117.3 KB，**2→1 block/SM**）
  ——小核「往 smem 搬东西」的第一代价是**峰值 smem 顶掉占用率**，先算 `smem/block × blocks ≤ 227KB`。
- **把多个「各自独立编译、彼此不组合」的段合成一个核时，最先撞的是共享入参的重复声明**（P7-60，§63.1）：
  `b2_mma_probe.cu` 的 `beta` 原在 `B2M_OUT`/`B2M_GATE`/`B2M_SAND` 三处各声明一次（单独编译各出现一次、
  合法），并集后成同名重复参数直接编译失败。改法是让后两处**在已被前一处声明时让位**
  （`#if B2M_GATE && !B2M_OUT` 等），从而**既有隔离 launcher 签名一字不变**、只有并集 build 看到新顺序。
  另：`b2prod_launch` 的参数表必须严格按宏展开后的 kernel 签名排（含 TMA 七张图与 `scale` 夹在中间），
  ctypes `argtypes` 的 `c_float` 位置数错会在第 N 个实参报 `wrong type`。
- **「省字节」的第二个前提：这笔访存是否本就被流水藏住**（P7-61，§64.4/§64.6）：六段 `prod` 里 GEMM
  基座算出的 `dw` 走「写全局→DKGB 段读回」，同 CTA 交接（`-DB2M_HDW`）删掉这趟往返（DRAM −126 MB、
  L2 −0.33 GB）却**端到端略慢（CUPTI/wall +2.8%/+4%）**——因为原 DKGB 段在 stage 内与后续 stage 的
  GEMM/TMA **重叠**、那趟往返被藏住，交接迫使 DKGB 后移成**串行尾巴**丢掉了重叠；隔离的「核心+DKGB」
  里 staging 无从隐藏，交接才净赚 **7–9%**。故「省字节」除要看是否落最紧 pipe（§51.6）外，还要看该
  访存是否在关键路径上——**同一改动在隔离段与整核里可以反号**。另：此类改动 ncu `Duration` 与
  CUPTI/wall 可**反向**（ncu 清 L2 使被藏的读变成真 DRAM 读，同 §63.5/§25.5），口径仍取 CUPTI/wall。
- **把 B3 对角 intra 的四个 `tl.dot` 手工翻译成 CUDA `mma` 时，逐支路的 `2^{±g}` 门控别漏乘**（P7-62，
  §65.3）：`dq2`/`dk2` 共享同一 B 操作数（`KG`）与同一门控因子 `2^{g}`，`dkt` 用 `2^{-g}`——第一版只在
  门控循环里写了 `adq *= 2^{g}` 而漏了 `adk *= 2^{g}`。指纹＝**同一 B 操作数、同一门控下只有 `dq` 对、
   `dk`/`db`/`dg` 同比例偏大**（误差随 `g` 幅度放大，因 `db` 也取自 `dk2`）。
- **「结构闭合」≠「性能净赢」**（P7-63，§66.4/§66.6）：把 B3 对角 intra 折进生产 launch（省一次 launch、
  `dA` 不落盘）**大 shape 反而略慢 1.04–1.06×**——独立 B3 是一批只做 intra 的 CTA、其工作量在 executor
  层被更好地流水掉，折进却把它变成**每个 CTA 关键路径尾部的串行尾巴**（并多扫一遍 `q/k/g` 的 smem
  staging、bank conflict +55%）；只有小 shape/短序列（launch 受限）才净赢 0.88–0.89×。判据同 §64：**该活
  在隔离核里能重叠、并进生产核后是否仍在关键路径之外**。另：折进的 `dg`（折在 `m_last` 项**之前**）与
  `db`（折成 `Σ_w(dbrow+idbrow)`）只改 fp32 **结合序**，故与「两核相加」差 ulp（≤9.5e-7）而非逐位。
- **核内复用 `dAkk` 时掩码别漏**（P7-63，§66.7）：`daB` 存的是 SAND 未掩码的 `X2`，intra 读它当 `dAkk`
  必须自带 `(a>j) & valid` 掩码，否则 `Mk/Sk` 把上三角也吃进去（指纹＝`dq` 对、`dk/dg` 错到 O(1)）。
- **「单核化/结构闭合」不等于「端到端更快」——要用整链计时钉住**（P7-64，§67.3）：K2 形状 CUDA B2 的
  每一段相对「GEMM 核心」都只贵一两成（§55–§60），但六段叠进同一个 1 block/SM、tensor 3%、发射受限的核
  后，整核比 **Triton 融合 B2/B3 慢 ~1.6×**。段级「CUDA 赢 Triton」的结论**不能外推**到整核（Triton 会把
  中间量留在寄存器/融进一张 `tl.dot` 图）。凡「把独立核合起来」的改动，必须以**整链 dev** 收口（同 §63/§66）。
- **CUDA `<x>_probe.cu` 是 fp32/tf32 核**（P7-64）：落链时 B1 的 bf16 `dh`/`dv2` 需转 fp32 才能喂
  `b2_mma_probe.cu`（无 bf16 入口）；且其 `-DB2M_TMA` 分支未传 `-DB2M_INTRA` 的 `dAqk_in`
  （`b2_mma_probe.cu:1920-1931`），两宏不可同时用。
- **对拍 `b2_mma_probe.cu` 的门控/生产输出前，先查该 flag 下哪些缓冲真被写**（P7-66，§69.7）：
  `B2M_GATE` 生产形态只存门控后的 `dqg/dkg`，**raw `dq`/`dk` 缓冲不会被写（未初始化）**——误把 raw
  `dq/dk` 当输出比，会读出 O(1) 的垃圾差（本轮首版因此把「mma 改坏门控」当成真 bug，实际改对输出集后
  全部逐位）。同理「孤立 `b2da_launch` 逐位」不能外推到「生产 launch 逐位」——两条链的缓冲契约不同。
- **「段内砍 2×」与「整核砍 2×」是两回事**（P7-66，§69.4/§69.6）：`mma` 化 DA 把段内边际 185→83µs
  （0.45×），但整核只 0.93–0.95×——段只占核 ~14%。判「某段优化能否翻盘」须先乘段占比（同 §67.3/§68.3）。
- **同叫「标量 store」，危害可以相反**（P7-70，§73.1/§73.6）：`-DB2M_VECST` 的收益**不总**来自「消半 sector」
  ——§51/§59 的片段跨步 store（4 lane 各写 sector 的 `{0,8,16,24}`）是**扇区**问题（`l1tex st`=1.98×），向量化
  改**字节**；而 `dA`/`dApre` 的 `col=idx%BT` 型**相邻列** store **本就完全合并（扇区已最小）**，向量化只**砍指令数**
  （`STG.E.32` 37→19、`STG.E.64` 0→18），收益靠**本核发射受限**兑现（生产核 0.93–0.96×）。判「该不该/为何值钱」
  先看**线程→地址映射**（是否连续），再看核是**扇区 bound**还是**发射 bound**——别把 P7-48 的「半 sector」当通解。
- **同一个 store 旋钮的收益随「访存是否走 TMA」而缩**（P7-71，§74.2）：`-DB2M_VECST` 在整链上
  基本恒正（`B2`/`B8`/`T250` 1–3%，最小 shape 噪声内），但**非 TMA 版收益更大**（`pci*` 0.970–0.986）**、TMA 版更小**（`pci_tma*` 0.984–0.997）
  ——TMA 把操作数访存从 LSU 发射挪到独立引擎后，store 指令数本就不再是最紧的发射项。判「向量化 store
  值多少」除了看线程→地址映射与 bound 类型（§73.6），还要看**同核里访存的发射压力是否已被 TMA 分流**。


## 当前进度 / 下一步

- 当前进度：**P0–P8 全部完成 + P6-3…P6-9 / P7-3…P7-71 补测**。
  - P7-71（本轮）：**关闭 §73.6（P7-70）的边界——`-DB2M_VECST` 落进完整反向链、并与 TMA/DAMMA
    三向交叉**。八份 fold 内核 `{scalar/mma DA}×{非TMA/TMA}×{标量 store/VECST}` 各链成
    `dAv→CUDA-B1→CUDA-B2-prod_intra→反向 cumsum`。新增 `harness/check_bwd_b2_vecst_e2e.py`
    （`check_bwd_b2_vecst_e2e.out.txt`，空闲卡）+ `docs/07` §74。**Part A**：四对孪生六项梯度**全部
    逐位相同（max_abs=0，含非整除 T=250/B=2）**、八变体 vs FLA worst-case **逐值同 §72**。**Part B**：
    链 `dev` VECST **0.968–1.003×**（`B2`/`B8`/`T250` 0.968–0.995、`T512` 噪声内）、且**非 TMA > TMA**（TMA 版 store 本就不是最紧发射项）；
    `pci_tma/pci` 1.002–1.027、`pci_mma_tma/pci_mma` 0.976–0.987（DAMMA 把 TMA 反号为正，兑现
    §71/§72）。**结论**：B2 生产核所有 store 变体在隔离核与整链两处都核验逐位无损；研究形态内部
    免费收尾，**不改变 §45 的 `cf` 默认部署**。**下一步**：专题维持维护态——§45 的 `cf` 仍是目标卡
    反向默认，CUDA B2 生产核整体慢 ~1.7×（§67–§72 已反复归因、明确不再追）；CUDA B2 研究形态的
    store/DA/stage 旋钮至此**全部关闭**，有明确新需求（新 shape/dtype/融合目标）再动。
  - P7-70（上轮）：**关闭 §57.6/§58.6/§62.6 留下的最后一条「store 未向量化」边界**——`[BT,BT]`
    的 `dA`/`dApre` store（§59 明确没做的那两处）改按**相邻列成对**（`st2`），**算术一字未动**。新增
    `harness/check_bwd_b2_vecst_da.py`（`check_bwd_b2_vecst_da.out.txt`，空闲卡）+ `docs/07` §73。
    **关键区别**：§51/§59 是**片段跨步**（半 sector）store，向量化改**字节**；`dA`/`dApre` 是
    `col=idx%BT` 型**相邻列**、扇区**本就最小**，向量化只**砍指令数**。**Part A**：standalone
    `da`/`sand` 四组 shape（含非整除 `T=250`）**逐位相同（max_abs=0）**。**Part B**：孤立段 `dev`
    **0.93–0.97×**。**Part C（关键）**：生产核（`-DB2M_PROD -DB2M_HDW`）**十项输出逐位相同**、`dev`
    **0.932–0.960×**（`B8T2048H8` 1262.5→**1208.5**）——与「本核非 DRAM bound、发射受限」一致。
    **Part D**：SASS `STG.E`(32b) 37→19、`STG.E.64` 0→18。**结论**：B2 生产核所有 store 至此都已
    向量化/已评估；**方法学（新）**：同叫「标量 store」先看线程→地址映射（是否连续）、再看核是
    扇区 bound 还是发射 bound。**边界**：只测 `prod_hdw`，未与 `pci*` 整链/`-DB2M_TMA` 交叉；
    研究形态内部免费收尾，**不改变** §45 的 `cf` 默认部署。**下一步**：专题维持维护态——§45 的
    `cf`（`dAv→CUDA-B1→融合 Triton B2/B3→cumsum`）仍是目标卡反向默认，CUDA B2 生产核整体
    慢 ~1.7×（§67–§72 已反复归因、明确不再追）；有需求时可把 `-DB2M_VECST` 与 `pci*`/TMA 交叉收尾。
  - P7-69（上轮）：**关闭 §71.6 的最后一条边界——`pci_mma`/`pci_mma_tma`（DAMMA 进链）端到端**。
    新增 `harness/check_bwd_b2_damma_e2e.py`（`check_bwd_b2_damma_e2e.out.txt`，空闲卡）+ `docs/07` §72。
    四个折叠核（`-DB2M_PROD -DB2M_HDW -DB2M_INTRA`）只差 DA 实现 × stage，各链成
    `dAv→CUDA-B1→CUDA-B2-prod_intra→反向 cumsum`。**Part A**：`pci_mma_tma` vs `pci_mma` **六项梯度
    逐位（max_abs=0）**；`pci_mma` vs `pci` `dq/dv/dh0` 逐位 0、`dk≤5.45e-6/dbeta≤7.99e-6/dg≤9.89e-7`；
    四变体 vs FLA worst-case **逐值完全相同**。**关键副产物**：捕获原始 B2 输出证明**真实链里 DA 的
    tf32 改写量本身只有 ~5–8e-6**（`dqg` 逐位 0），比 §71.2 合成操作数的 1.2–1.9e-3 小 3 个数量级
    （tf32 误差随操作数幅度走）→ **修正 §71.2 的量级读法**。**Part B（关键）**：链 `dev`
    `pci_mma/pci` **0.975–1.001**、`pci_mma_tma/pci_mma` **0.980–0.992**（**DAMMA 打开后 TMA 反号成
    快 ~1–2%**，兑现 §71.3）、`pci/FLA` 0.45–0.59、`pci/cf` ~1.68–1.73。**结论**：DAMMA 进链数值无损、
    ~1–2.5%，叠 TMA 再 ~1–2% 且免费；目标卡反向默认仍是 `cf`。**边界**：仅 `K=V=128/BT=16/fp32`；
    非折叠 `pc` 的 DAMMA 未单独落链。**下一步（可选收尾）**：`-DB2M_VECST` 接 `pci*` 的 `dA`/`dApre`
    小张量 store（§62 遗留）；或 CUDA B2 生产核与 Triton `cf` 的 ~1.7× 差已被 §67–§72 反复钉在
    「六段非 GEMM 成本在同核累积」，明确**不再追**——专题维持 `cf` 默认部署、CUDA 只在 B1 净赢。
  - P7-68（上轮）：**关闭 §70.6 的两条待验证（TMA × DAMMA 生产态、TMA 折叠版落整链）并修正 §70.5
    对 TMA 生产代价的归因**。新增 `harness/check_bwd_b2_tma_damma.py`（`check_bwd_b2_tma_damma.out.txt`，
    空闲卡）+ `harness/probe_b2_td_ncu.sh`（`probe_b2_td_ncu.out.txt`）+ `docs/07` §71。四变体
    `sc/sc_tma/mma/mma_tma`（`-DB2M_PROD -DB2M_HDW`；HDW 下 `dw` 是 scratch 须排除）。**Part A**：
    `*_tma` vs 非 TMA 双胞胎 **8 项全逐位**、`mma* vs sc*` 只 `dApre/dA` 落 tf32（1.2–1.9e-3）。
    **Part B（关键）**：`sc_tma/sc` **1.039–1.057**（TMA 仍慢 4–6%），`mma_tma/mma` **0.978–1.006**
    （`B8` = **1.000**）——**DAMMA 把 TMA 的符号拨回 ~0%**。**Part C**：`pci_tma` vs `pci` 逐位、
    链级 `pci_tma/pci` **1.006–1.024**、`pci/cf` **1.6–1.7×** 复核 §67.3。**`ncu`**：TMA 两列 L2 都
    低 ~0.67 GB，但 bank conflict **标量 DA 下 23.46M→28.31M（+4.85M）、mma DA 下不动（17.30M）**
    （与 §69 逐值吻合）→ **生产核 TMA 的代价本体就是标量 DA 的 bank conflict**；设计输入更新为
    「**随 DA 实现切换**：mma DA 可同时开 TMA，标量 DA 则非 TMA」。**新判据**：bank-conflict 增量按
    段实现形态读。**边界/待验证**：`pci_mma/_tma`（DAMMA 进链）未做端到端。
  - P7-67（上轮）：**关闭 §67.4（P7-64）记录的边界「`B2M_TMA` 与 `B2M_INTRA` 未打通」**。根因＝
    `b2prod_launch_impl` 的 `#if B2M_TMA_ON` 分支漏传内核签名末尾的 `dAqk`（`:707-709`），
    `-DB2M_TMA -DB2M_INTRA` 编不过（删修复后 `error: too few arguments`）。修复＝TMA 分支补
    `#if B2M_INTRA (const float*)dAqk_in, #endif`（算术一字未动）；宿主 launcher 本就 TMA-agnostic。
    新增 `harness/check_bwd_b2_intra_tma.py`（`check_bwd_b2_intra_tma.out.txt`，空闲卡）+ `docs/07` §70。
    **Part A**：`prod_intra_tma` vs `prod_intra` 六组 shape（含非整除 T=250、单 chunk T=64）**11 项输出
    逐位相同（max_abs=0）**。**Part B**：regs 144→128、smem 159.9→149.1 KB（−6.7%）+12B spill，仍
    1 block/SM。**Part C（关键/诚实）**：CUPTI `dev` TMA **一律慢 8–12%**（B8T2048H8 1642.7→1788.3，
    与 §66.4 的 1636.7 互证）。**结论**：边界关闭、开关可叠加且数值无损，但**目标卡 CUDA B2 默认取非
    TMA**（孤立 GEMM 核净赚 ~5% 的旋钮，到六段齐全的生产核——已证非 DRAM bound——反号成慢 8–12%）。
    加 `-DB2M_DAMMA`（需 `B2M_DA`）：每 warp 用 `mma.sync.m16n8k8.tf32` 把 `dApre = dv@v^T - dw@kg^T`
    的每个 `[BT,BT]` partial 整块算出（2 n-tile × 每项 2 k-step，收缩自身 `BW=16` 切片），A/B 复用
    `dv/swall` 与 `vv/skg`（读法同 GEMM 核、tf32），partial 仍过 `dapart→da16` 归约。新增
    `harness/check_bwd_b2_da_mma.py`（`check_bwd_b2_da_mma.out.txt`，空闲卡）+ `harness/probe_b2_damma_ncu.sh`
    （`probe_b2_damma_ncu.out.txt`）+ `docs/07` §69。**Part A**：`dApre` 落 **tf32 底噪 3.42e-4**（标量
    fp32 6.4e-7）、三种布局**逐位**（含非整除 T=250）。**Part B（孤立）**：`da_mma/da` 0.87–0.92×。
    **Part C（生产核，关键）**：非 DA 七项输出**逐位不变**、`prod(mma)/prod(sc)` **0.93–0.95×**
    （`B8T2048H8` 1279.5→**1213.5**）；**§68 的 L2→L3 消融**给出生产态 DA 边际 **185.4→83.3µs（0.45×）**、
    其余 shape **0.32–0.48×**——**杠杆方向正确、段内砍 2–3×**，**但整核只 +5–7%**（DA 占核 ~14%），
    `1214/998=1.22×` **未拉进 Triton 融合核**。**`ncu`**：bank conflict 只削 ~40%（tf32 无 16-bit
    `ldmatrix`）、`Duration` 1.27→1.21ms、tensor 3.37→3.81、DRAM/L2/occ 不动。**结论（修正 §68.5）**：
    `mma` 化 DA 值得做（免费、正确、段内 2–3×）但**不是翻盘点**；§67/§68 默认建议不变（目标卡默认反向
    仍是 §45 的 `dAv→B1→融合 Triton B2/B3→cumsum`）。**坑**：`B2M_GATE` 下 raw `dq/dk` 缓冲未初始化
    （只存 `dqg/dkg`），误比会读出 O(1) 垃圾。
  - P7-65（上轮）：**把 §67.4 的「六段非 GEMM 成本在同一个 1 block/SM 核里累积」逐段归因**（此前每段
    都在各段独立的 kernel/launch 里量）。给 `b2_mma_probe.cu` 加 `-DB2M_PROD -DB2M_HDW -DB2M_LEVEL=k`
    （保留所有 flag/签名/staging、只按逆依赖序删段函数体，`b2_mma_probe.cu:150-176`）。新增
    `harness/check_bwd_b2_prod_levels.py`（`check_bwd_b2_prod_levels.out.txt`，空闲卡）+
    `harness/probe_b2_levels_ncu.sh`（`probe_b2_levels_ncu.out.txt`）+ `docs/07` §68。**Part A**：各 level
    存活输出 vs level 6 **逐位相同**（四组 shape）。**Part B（关键）**：`B8T2048H8` 上 `OUT 151(24.7%)/
    DKGB 79/DA 184(30%)/SAND −3/DB 36/GATE 204(33.5%)`，总非 GEMM **651.6µs**，**OUT+DA+GATE 占 83%**，
    曲线**≈加性**（与各段独立数字同阶）。**`ncu`**：OUT/GATE 的成本=**交付物字节**（+130.8 / +198.2 MB
    最终梯度写）、DA=**标量数学**（bank-conflict ×2.2、tensor 不动）。**结论（修正 §67.4）**：目标卡的
    「重新融合回寄存器」上界仅 ~40µs（~4%）、**不做**；**唯一有量级杠杆＝`mma.sync` 化 DA**。
  - P7-64（上轮）：**把 K2 形状 CUDA B2 生产核第一次落进完整反向链**（此前验证全在段间/合成操作数）。
    新增 `harness/check_bwd_prod_e2e.py`（`check_bwd_prod_e2e.out.txt`，空闲卡）+ `docs/07` §67。组装
    `dAv(Triton)→CUDA-B1→CUDA-B2-prod[→Triton B3 intra]→反向 cumsum`（只吃 K1 workspace），两变体
    `pc`（`prod_hdw`+独立 intra）/`pci`（`prod_hdw_intra` 单 launch）。**Part A**：6 组 shape（含非整除
    `T=250`/`B2 T=513`、`g=0`、`B=2`）六项 vs FLA 落 bf16/tf32 底噪（与 `cf` 逐值同阶），`dv`/`dh0` 与
    §45 最优链 `cf` **逐位相同**。**Part B（关键）**：整核**不是净赢**——`pci/cf` **1.27–1.73×**
    （`B8T2048H8` `prod_intra` 1606.9µs vs 融合 Triton 998µs），差来自六段非 GEMM 成本在同核累积。
    **结论（更新 §66.6）**：目标卡最优反向仍是 §45 的 `dAv→B1→融合 Triton B2/B3→cumsum`；CUDA B2 生产核
    是**正确但更慢**的研究产物，下一杠杆＝把六段重新融合回寄存器（收益上界已钉在 ~1.6×）。
  - P7-63（上轮）：**兑现 §65.6/P7-62 的「下一步」——把 B3 对角 intra 折进生产 launch**（`-DB2M_INTRA`，
    需 `-DB2M_PROD`；新入口 `b2prod_intra_launch` 多 `dAqk` 入参）。用 SAND 的**核内** `daB`（掩码 `-X2`
    = `dAkk`，不落盘读回）+ `dAqk` + 合流 `q/k/g`，四个 `[BT,BT]@[BT,KD]` tf32 mma 产出 `idq/idk/idg`，
    寄存器 merge 进门控最终 `dq/dk/dg`（片元同构）、`db` 并进既有跨 warp 归约；DA+SAND 前移至 GATE/DB
    之前（段内算术一字未动）。新增 `harness/check_bwd_b2_intra.py`（`check_bwd_b2_intra.out.txt`）+
    `ncu_b2_intra_driver.py` / `probe_b2_intra_ncu.sh`（`probe_b2_intra_ncu.out.txt`）+ `docs/07` §66。
    **Part A**：6 组 shape 折进 vs「`prod`+独立 B3」`dqg/dkg` **逐位=0**、`dg ≤4.8e-7`/`db ≤9.5e-7`
    （fp32 结合序）、未触及输出全逐位不变。**Part B**：regs 146→144、零 spill、smem 133760→159872 B。
    **Part C**：大 shape **1.04–1.06×（略慢）**、小 shape **0.88–0.89×（净赢）**。**Part D**：占用率
    /tensor% 不动，代价＝bank conflict +55% + L2 +70 MB + 串行尾巴。**结论**：结构闭合（单 kernel、`dA`
    不落盘），但目标卡**保持 B2/B3 两 kernel**，仅小 batch/短序列折进净赢。
  - P7-62（上轮）：**兑现 §60.5 的 K2 形状 CUDA 化最后一条结构项——B3 `intra` 折进 K2 形状**。部署
    `NC=1` 下 B3 只剩对角（§27.1），把 §27 的对角 MMA 恒等式搬进 CUDA、做成 K2 形状核（`grid (NT,B*H)`、
    warp-per-k-column、四个 `mma.sync.tf32` + fp32 门控）。新增 `harness/b3_intra_probe.cu` +
    `harness/check_bwd_b3_intra_cuda.py`（`check_bwd_b3_intra_cuda.out.txt`）+ `ncu_b3_intra_cuda_driver.py`
    / `probe_b3_intra_cuda_ncu.sh`（`probe_b3_intra_cuda_ncu.out.txt`）+ `docs/07` §65。**Part A**：5 组
    shape（含非整除 `T=250`/单 chunk/`g=0`）vs Triton MMA B3 四项 `max_rel` **1.2–1.5e-3**（tf32 底噪）、
    vs torch golden **≤6.4e-4**。**Part B**：孤立 `dev` 快 **1.3–1.45×**。**Part D（ncu）**：
    Duration 295.6→239.4µs、DRAM 61.3→50.4%、L1 命中 18.5→51.4%、`long_sb` 19.9→11.4。**结论**：
    结构闭合 + 比 Triton 快 1.3–1.45×。**下一步（未做）**＝并进 `-DB2M_PROD` 生产 launch（用核内
    `dA`+`dAqk`+已 staging 的 `q/k/g` 加到门控段最终 `dq/dk/dg`、进 `db`），须处理寄存器/smem 峰值与
    累加器共享；不改本轮已证代数。
  - P7-61（上轮）：**兑现 §49.7/§63.6 最后一条真结构项——同 CTA `dw` 交接**（`-DB2M_HDW`）。给
    `b2_mma_probe.cu` 加 `-DB2M_HDW`（需 `B2M_DKGB`）：核心的 `dw`（`accw`）留在核内、镜像进 `swall`，
    DKGB 的 mma 改在 it-loop 之后从镜像读（`accw` 须全部 V-tile 累加完才完整，故后移）；`Stage::dwm` /
    `prefetch` cp16 / TMA `dwm` 图 / 核心 `dw_out` store 条件关掉，`swall` 分配由 `B2M_DA` 扩到
    `B2M_DA||B2M_HDW`。新增 `harness/check_bwd_b2_hdw.py`（`check_bwd_b2_hdw.out.txt`，空闲卡）+
    `harness/probe_b2_hdw_ncu.sh`（`probe_b2_hdw_ncu.out.txt`）+ `docs/07` §64。**Part A**：四组 shape
    （含非整除 `T=250`）`prod_hdw` 十项输出 vs `prod` **逐位相同（`max_abs=0`）**、TMA 布局亦逐位；
    **Part A2**：喂**全零 `dw`** 结果**逐位不变**（`dw` 输入证实未被读）。**Part B**：regs 140→110
    零 spill、smem 133.8→129.2 KB（TMA 123.0→118.9）。**Part C/D**：CUPTI `dev`/`wall` 大 shape 略慢
    **+2.8%/+4%**、小 shape 噪声内（wash）；**Part E（关键）**：隔离「核心+DKGB」里交接 **快 7–9%**
    （B8 570→522µs）。**机制**：`dw` 往返在 `prod` 里被 GEMM 流水藏住、后移 DKGB 丢 stage 内重叠；隔离
    段里 staging 无从隐藏故净赚。**`ncu`**：DRAM −126 MB / L2 −0.33 GB、bank conflict/occ/tensor 不动；
    ncu `Duration` −7% 与 CUPTI/wall 反向（窗口不同源，同 §63.5）。**结论**：`dw` 从生产 ABI 消失
    （不预算/不传参/不落盘），是**内存足迹优化、非速度杠杆**；判据＝「省字节前先问这笔访存是否本就被
    流水藏住」。`dv2`/`v` 喂 `db`、`dApre` 入 sandwich 早已核内；至此 B2 六段中间量要么本就核内、要么
    可核内，K2 形状 CUDA 化**结构上闭合**。
  - P7-60（上轮）：**兑现 §62.6 唯一剩余真结构项——六段合成单一生产 launch**（`-DB2M_PROD`）。在段旗标
    定义前把六个旗标全置 1、新增 `b2prod_launch`，六段**代码体一行未动**（累加器/smem/输出两两不相交），
    唯一共享改动是 `beta` 由三处声明改一处（既有隔离 launcher 签名不变）。新增
    `harness/check_bwd_b2_prod.py`（`check_bwd_b2_prod.out.txt`）+ `ncu_b2_prod_driver.py` /
    `probe_b2_prod_ncu.sh`（`probe_b2_prod_ncu.out.txt`）+ `docs/07` §63。**Part A**：四组 shape（含非整除
    `T=250`）`prod` 每个输出 vs 对应隔离 build **逐位相同（`max_abs=0`）**、`prod_tma` vs `prod` 逐位。
    **Part B**：`prod` regs 140 **零 spill**、smem 133.8 KB（1 block/SM）。**Part C（关键）**：一次 `prod`
    launch vs 老设计 3 次 launch **0.467–0.526×**（`B8T2048H8` 2450→1288µs；b2 核 45→15，省下 2 份 GEMM 核
    +2 次 launch）。**Part D**：门控 `dqg` vs torch 逐位、`dkg/dg`~1e-6（fp32 序）、`dv2`~1.2e-4（tf32）。
    **`ncu`**：合并后 DRAM SOL 45%、tensor 3%、occ 12.5%——非 DRAM-bound、标量/发射受限，证 ~2× 来自核复用。
    **该下一步（§49.7 同 CTA 交接 `dw`）已由 P7-61 收口**（逐位无损、`dw` 从 ABI 消失；六段生产核里 wash、
    隔离段里 7–9%）。
  - P7-59：**收口 §51.6「向量化 store 一律做」在 B2 非 GEMM 四段上留下的边界**（P7-52/53/56/57
    每轮边界都写「本段 store 未接 `-DB2M_VECST`」）。给 `b2_mma_probe.cu` 加 `st2()` 成对 store 助手
    （`:96`：`B2M_VECST` 下一条 `STG.E.64`、否则两条标量），落到 `dv2/dvb`（`:934-940`）、`dkgb`
    （`:952-953`）、门控 `dqg/dkg`（改先写 `dqgv/dkgv[2][4]`，循环后成对存 `:1011-1014`）与 `dg`
    （`:1054-1055`）——**算术一字未动**。新增 `harness/check_bwd_b2_vecst_seg.py`
    （`harness/check_bwd_b2_vecst_seg.out.txt`，空闲卡）。**Part A**：四段 base vs vecst 同源对照、四组
    shape（含非整除 `T=250`）**全部逐位相同（`max_abs=0`）**；门控 vecst vs FLA `dqg 1.051e-3/dkg 9.466e-4/
    dg 1.010e-3` 逐值同 P7-57。**Part B**：CUPTI `dev` vecst/base **16/16 全为正**，小 shape `T512/T250`
    **0.835–0.919×**、大 shape `B8T2048H8` 0.983–0.990×、`db` 段大 shape 也 **0.940×**。**机制**：`db` build
    SASS `STG.E` **50→2**、`STG.E.64` 0→**24**。见 `docs/07` §62。**边界**：`dA`/`dApre` 未向量化（小张量 +
    thread-per-entry）；仅 `K=V=128/BT=16/fp32`。**下一步＝维护/收尾**：§49.7–§60.5 的六段已全进 CUDA、合流
    已证伪（P7-58）、四段 store 已向量化（P7-59），K2 形状 CUDA B2 的廉价/半廉价项全部关闭。当时点名的
    唯一剩余真结构项「六段**合成单一生产 launch**」已由 **P7-60** 关闭（`-DB2M_PROD`，逐位无损、0.47–0.53×）；
    另一条「B3 intra 折进 K2 形状」仍待（须先合并 kernel 签名、处理寄存器/smem 峰值）。
    结论＝证伪（净亏）**（`docs/07` §61）。给 `b2_mma_probe.cu` 加 `-DB2M_QKG`（需 `B2M_GATE`；诊断
    `-DB2M_QKGP` 把 staging 行距 `KD`→`KD+4` 消固定列 row-aliasing 冲突）：chunk 开头一次合流 `[BT,KD]`
    staging 读 `q/k/g`（`:643-654`，与 `kg` 同构），门控改读 `sh->sq/skb/sgb[rowrel][col]`（`:960-976`，
    算术一字未动）。新增 `harness/check_bwd_b2_gate_qkg.py`（`check_bwd_b2_gate_qkg.out.txt`，空闲卡）+
    `harness/probe_b2_gate_qkg_ncu.sh`（`probe_b2_gate_qkg_ncu.out.txt`）。**Part A**：`dqg/dkg/dg` vs
    P7-57 `gate`/`gate_qkgp`/`gate_tma_qkg`/`gate_tma_qkgp` **全逐位相同（max_abs=0）**、`gate_tma64_qkg`
    仅 `dg` 1 ulp（含非整除 T=250）——**合流逐位无损**。**Part B（关键）**：合流**一律更慢**——非 TMA
    `qkg/gate` **1.11–1.51×**、TMA `tqkg/tma` **1.08–1.19×**；pad 行距只把大 shape 1.51→1.49、1.09→1.08，
    **不改符号**。**ncu**：两独立机制——(1) 非 TMA 三张常驻 `[BT,KD]` tile（+24KB）把 dyn smem 92.7→117.3KB、
    **占用率上限 2→1 block/SM**（`warps_active` 24.2→12.4%），pad 后 bank conflict 19.9M→10.5M 回基线仍
    1.49×；(2) TMA 版占用率不掉（2 blocks），global ld sector 确从 **19.40M→12.85M（−34%，预期收益兑现）**、
    pad 后 bank conflict 回基线 11.46M，但 `Duration +8.7%`、DRAM 读不变（1.56→1.55GB）——本核 **DRAM/L2
    双 bound（~86/88% SOL）**，删 L1 散读（被 L2 吸收）**不动瓶颈**，staging 多一遍 smem 写+同步反成净负。
    **结论**：**修正 §60.4**——门控 +30% 的成本是 **三份 `[B,T,H,K]` 输出写（199MB，交付物）+ `exp2f` XU +
    本核本就 DRAM/L2 bound**，**不是** raw `q/k/g` 的散读合流度（散读真实存在于 L1 计数但被 L2 吸收、
    不进端到端）。**设计输入（更新）**：门控链**不必**为 `q/k/g` 单设 smem staging；`q` 保持直读 global
    更划算；若合流须不增 smem 峰值且不掉占用率——但本轮已证即便做到仍 +8.7%，故**不作为优化项**。
    **方法学（新）**：把「ncu 计数里的散读」当成本前，须用「换算子只改取数来源、occupancy 不变」的受控
    对拍验证它是否在关键路径上；否则会把一个被 L2 吸收的分项误判为端到端头寸。**至此 §60.5 的两条
    「下一步」（收尾/维护 + 合流 `q/k/g`）全部关闭**：六项已全进 CUDA、合流被证伪，专题进入维护态。
  - P7-57（上轮）：**兑现 §55.6/§56.6/§57.6/§58.6/§59.6「补非 GEMM 段」第六刀——门控链（最终
    `dq/dk/dg`）落进 K2 形状 CUDA B2**（`docs/07` §60）。给 `b2_mma_probe.cu` 加 `-DB2M_GATE`（需
    `-DB2M_DKGB`；诊断 `-DB2M_NODG`；新符号 `b2gate_launch`）。门控标定全在寄存器（warp 拥有 k-列切片，
    各 mma 片元独立）；唯一归约 `Σ_t kdk` 是 **warp 内 `shfl_xor` over groupID**（mma 16 行 = 8 lane×2 行），
    `g==0` lane 写 `sdgkt[KD]`、`last` 行读回；`B2M_GATE` 下不再落 raw `dq/dk`（生产形态）。新增
    `harness/check_bwd_b2_gate.py`（`check_bwd_b2_gate.out.txt`）+ `ncu_b2_gate_driver.py` / `probe_b2_gate_ncu.sh`
    （`probe_b2_gate_ncu.out.txt`）。**Part A**：`dq` vs torch（同一 raw）**逐位（max_abs=0）**、`dk`/`dg`
    **fp32 序 4.3e-8/2.2e-7**、vs FLA 三项 **tf32 底噪 1.0e-3**；非 TMA/TMA@32 逐位、TMA@64 仅 `dg` 差 1 ulp
    （含非整除 T=250）。**Part B**：门控是前五刀**最贵**的一刀（`gate/nodg` **1.30–1.38×**），`gate_tma/B2`
    **0.73–0.78×**。**ncu**：增量＝+199 MB 输出写 + **9.2× global-load sector（片元式散读 `q/k/g`）** +
    `exp2f`（XU 0.20→1.51 M）；占用/冲突/tensor 不动、仍 DRAM/L2 双 bound。**至此 B2 六项
    `dq/dk/dv2/db/dg/dA` 全部进 CUDA**。**坑**：`m_last` 项按 lane 条件写会错加行 7/15，须按行坐标。
    **下一步＝专题收尾/维护**；有需求时把 `q/k/g` 合流载入 smem 收回门控的 +30%（待验证）。
  - P7-56（上轮）：**兑现 §55.6/§56.6/§57.6/§58.6「补非 GEMM 段」第五刀——写入门梯度
    `db = Σ_v dvb·v + Σ_k dkgb·kg` 落进 K2 形状 CUDA B2**（`docs/07` §59）。`dvb`(§55)/`dkgb`(§56) 已在同一核、
    相同 warp 列划分下算出，故 `db` 是**纯跨 warp 归约**：给 `b2_mma_probe.cu` 加 `-DB2M_DB`
    （需 `-DB2M_OUT -DB2M_DKGB`；诊断 `-DB2M_NODB`；新符号 `b2db_launch`）；`dvb·v` 项的 `v` 需持久镜像
    `sh->dbv[warp][BT][BW]`（`b2_mma_probe.cu:750-762`），`dkgb·kg` 直接读持久 `skg`，行组 `shfl_xor` + 8 warp
    smem 归约落盘（`:848-878`）。新增 `harness/check_bwd_b2_db.py`（`check_bwd_b2_db.out.txt`，空闲卡）+
    `harness/ncu_b2_db_driver.py` / `harness/probe_b2_db_ncu.sh`（`probe_b2_db_ncu.out.txt`）。**Part A**：`db` vs
    torch（同一 `dvb/dkgb`）**1.3e-7~1.8e-7**、vs FLA 落 **tf32 底噪 5.5e-4~6.3e-4**、**四布局逐位相同（max_abs=0，
    含非整除 T=250）**。**Part B**：`db` 段只值 **+2~8%（非 TMA）/ +2~13%（TMA）**；含 OUT+DKGB+DB 的 TMA CUDA 核到
    全 Triton B2 **0.85–1.04×**。**ncu**：增量＝`db` 落盘 `+1.4MB` + 读 smem `dbv/skg`（`LDS 104→124`），全局读
    sector/占用率/带宽都不动（`Duration +2.7%`）。**坑**：harness 起初把 `dw` 输入与 `dw_out` 输出设成**同一张量**
    （`dw=dv@h` 的 global round-trip），被覆写致 `dkgb` 大面积错；改独立缓冲后与 §56 专用 build **逐位相同**——正是
    「`dw` 应同 CTA 交接、不走 global」的实证动机。**下一步＝门控链**（落完后 B2 六项全进 CUDA）。
  - P7-55（上轮）：**兑现 §55.6/§56.6/§57.6「补非 GEMM 段」第四刀——最终 `dA` sandwich `-A^T diag(beta) A^T`
    落进 CUDA，补全 B2 `dA` 支路；并更正 §57 的 `dw` 符号**（`docs/07` §58）。给 `b2_mma_probe.cu` 加
    `-DB2M_SAND`（需 `-DB2M_DA`；+ 符号 `b2sand_launch`，输出 `dApre`+`dA`）：`b_A`/`beta` 每 chunk 载入 smem
    （`sA/sbeta/daB` ~3KB），两个 `[BT,BT]@[BT,BT]` 的 **fp32 标量**矩阵乘 + 一次 `beta` 缩放，**不重读 `dApre`**
    （直接吃 §57 的 `da16`）。新增 `harness/check_bwd_b2_sand.py`（`check_bwd_b2_sand.out.txt`，空闲卡）+ 
    `harness/ncu_b2_sand_driver.py` / `harness/probe_b2_sand_ncu.sh`（`probe_b2_sand_ncu.out.txt`）。**Part A**：
    `dA` vs torch（**同一份 `dApre`**）**逐位相同（max_abs=0）**、四种 stage 布局（非 TMA/TMA@32/TMA@64）`dA`/`dApre`
    都逐位、vs FLA 最终 `dA` 落 **tf32 底噪 1.56e-3~2.34e-3**（含非整除 T=250）。**Part B**：sandwich 很便宜
    （`sand/da` **1.02–1.06×**、`sand_tma/da_tma` **1.03–1.10×**，远低于 §57 的 +33–61%，因无新 `[BT,V]/[BT,K]`
    收缩）；含完整 `dA` 的 TMA CUDA 核到全 Triton B2 **0.77×**。**ncu**：增量＝`dA` 落盘 +8.0MB + `sA/beta` 标量读
    + `LDS 287→334`/`STG 26→31`，`Duration 655.8→703.5µs（+7.3%）`，bank conflict/占用率/tensor% 全不动。
    **关键副产物**：接上 sandwich 对最终 `dA` 比 FLA 时暴露 §57 的**符号错误**——FLA `b_dw=-b_dw`
    （`chunk_bwd.py:280`）故 `dApre` 第二项是 **`- dw@kg^T`**（§57 写成 `+`，其自身参考两侧同号故未现形）；
    已改正 `.cu`（`dapart=dbpart-dkpart`）与 §57 参考/公式（§57 数值不变，两侧同步翻转）。**边界**：`db` 的
    `sum_k dkgb·kg`（跨 warp 归约）已落；门控链当时未落（**已由 P7-57 关闭**）。
  - P7-54（上轮）：**兑现 §55.6/§56.6「补非 GEMM 段」第三刀——`dA` 的两个 GEMM 贡献（sandwich 之前）
    `dApre = Σ_v dv·v + Σ_k dw·kg` 落进 K2 形状 CUDA B2**（`docs/07` §57）。给 `b2_mma_probe.cu` 加
    `-DB2M_DA`（+ `b2da_launch`）：两个收缩维都**不是 `BT`**——base 按 warp 拥有 **V 列**（同输出支路分区）
    标量 `dv[r,v]*v[c,v]`、`dw@kg^T` 按 warp 拥有 **输出 k 列**（`accw` 镜像到 `swall`）标量点积；8 warp
    私有 `[BT,BT]` 部分和经 smem 归约后落盘。新增 `harness/check_bwd_b2_da.py`（`check_bwd_b2_da.out.txt`，
    空闲卡）+ `harness/ncu_b2_da_driver.py` / `harness/probe_b2_da_ncu.sh`（`probe_b2_da_ncu.out.txt`）。
    **Part A**：`dApre` vs torch 落 **fp32 底噪 5.31e-7**（比前两刀的 tf32 3.6e-4 低 3 量级，因用标量 FMA）、
    `da`≡`da_tma`≡`da_tma64` **逐位相同（max_abs=0，含非整除 T=250）**。**Part B（关键）**：本段**贵得多**——
    `da/core` **1.49–1.84×**、`da_tma/core_tma` **1.33–1.61×**（远高于 +10–25%），根因＝收缩维是完整的
    `V`/`K`（各 128，8× MAC）+ 标量 FMA；含本段 CUDA 核到全 Triton B2 仍 **0.72–0.81×**。**ncu**：shared
    wavefronts **+82%**、bank conflict **×2.8**、`LDS 88→287`、`DRAM SOL 88.87→76.17%`、`tensor 8.04→6.00%`、
    `Duration 494.9→660.4µs`（增量≈标量 smem 流量、非带宽）。**设计输入**：「收缩维 = 全维 `V`/`K`」的段应像
    三条主 GEMM 一样用 `mma.sync`（本轮先证代数取标量）。**下一步＝`-A^T diag(beta) A^T` sandwich + `db`
    的 `sum_k dkgb·kg`（跨 warp 归约）+ 门控链（`dg`/`dk` 的 `exp2(g)` 标定）**，把 B2 全核搬进 CUDA。
  - P7-53（上轮）：**兑现 §55.6「补非 GEMM 段」第二刀——`dkgb = A @ (-dw)` 落进 K2 形状 CUDA B2**
    （`docs/07` §56）。给 `b2_mma_probe.cu` 加 `-DB2M_DKGB`（+ `b2dkgb_launch`）：与 §55 输出支路同构
    （A 操作数走同一套索引重映射、无转置），B 操作数换成新 staged 的 `dw`（`Stage::dwm`，k-tiled）并在
    B-fragment 取负。新增 `harness/check_bwd_b2_dkgb.py`（`check_bwd_b2_dkgb.out.txt`，空闲卡）+
    `harness/ncu_b2_dkgb_driver.py` / `harness/probe_b2_dkgb_ncu.sh`（`probe_b2_dkgb_ncu.out.txt`）。
    **Part A**：`dkgb` vs torch **3.64e-4**（tf32 底噪）、`dkgb`≡`dkgb_tma`≡`dkgb_tma64` **逐位相同
    （max_abs=0，含非整除 T=250）**。**Part B**：段只给 GEMM 核心加 **+9–13%**、CUDA 核到全 Triton B2
    **0.63–0.70×**。**ncu**：增量＝DRAM 读 +0.07 GB(`dw`) + 写 65.4 MB(`dkgb`) +48.6 µs（≈2.78 TB/s，
    仍 DRAM-bound 88.7% SOL）、occ/tensor 不动。**下一步＝补 `dA` 后处理项（`dw@kg^T`、`A^T diag A^T`）+
    `db`（`sum_k dkgb·kg`，需跨 warp 归约）+ 门控链（`dg`/`dk` 的 `exp2(g)` 标定）**，把 B2 全核搬进 CUDA。
  - P7-52（上轮）：**兑现 §49.7「补非 GEMM 段」第一刀——B2 输出支路 `dv2=beta·(A@dv)` 落进 K2 形状 CUDA B2**
    （`docs/07` §55）。给 `b2_mma_probe.cu` 加 `-DB2M_OUT`（+ 符号 `b2out_launch`）：`Akk` 喂 mma 只需**索引
    重映射**（收缩维是 chunk 行 `o_t` 而非 V，故无 smem 转置）、B 操作数**复用已 staged 的 `dv`**（不新增
    `dv` 字节）。新增 `harness/check_bwd_b2_out.py`（`check_bwd_b2_out.out.txt`，空闲卡）+
    `harness/ncu_b2_out_driver.py` / `harness/probe_b2_out_ncu.sh`（`probe_b2_out_ncu.out.txt`）。**Part A**：
    `dv2`/`dvb` 落 tf32 底噪（vs torch **3.63e-4/4.40e-4**，比 Triton B2 的 5.96e-4 更贴真 fp32），
    `out`≡`out_tma64` **逐位相同**（含非整除 T=250）。**Part B**：输出支路只给 GEMM 核心 **+12–25%**
    （`B8T2048H8` `out_tma` 555.0 vs `core_tma` 492.3）；含 GEMM 的 CUDA 核到全 Triton B2 **0.62–0.74×**；
    `BVK=64` 反回退→输出支路偏好小 `BVK`。**ncu**：增量＝DRAM 写 **+130.6 MB**（`dv2`+`dvb`）+60.4 µs
    （≈2.16 TB/s，仍 DRAM-bound 87% SOL）、occ/tensor 不动。**下一步＝补门控链 + `dkgb=A@dw` + `dA`/`db`
    后处理**（把 B2 全核搬进 CUDA）。
  - P7-51（上轮）：**兑现 §53.5 的唯一待验证路径——`TMA + BVK=64` 用两个 128 B swizzle atom 拆分，
    正确、无冲突、且全程最快**（`docs/07` §54）。把 B2 K2-mma 核的 TMA 操作数布局改成 atom-major
    `[NAT=BVK/32][rows][32]`（每 atom 自己的 128 B swizzle TMA box，`rd` 按 `col>>5` 选 atom），算术一字未动。
    新增 `harness/check_bwd_b2_tma_atom.py`（`check_bwd_b2_tma_atom.out.txt`，空闲卡）+
    `harness/probe_b2_tma_atom_ncu.sh`（`probe_b2_tma_atom_ncu.out.txt`）。**Part A**：`atom64`≡`base64`
    （`dq/dk/dw/dgk` max_abs=0，含非整除 T=250）、`atom64`≡`tma32` 的 `dq/dk/dw` 逐位、vs torch 3.6e-4。
    **Part B**：`atom64/base32` dev **0.853–0.974×**、`atom64/base64` 0.927–0.976×——**§49.5「BVK=64 最优分块」
    × §52「TMA 读侧修复」两收获首次叠加**（对照 `SWIZZLE_NONE` 版 `tma64/base64` 的 2.5–3.0×）。**ncu**：
    L2 3.02→2.39 GB、srcunit 59.77M→39.85M、l1tex 全局读=0、**bank conflict 8.39M（非 SWIZZLE_NONE 的 272.6M）**、
    DRAM SOL 90.6%（最高）、Duration 518.9→**485.9 µs**。**结论**：目标卡 CUDA B2 取 `BVK=64` 时用 NAT 个 128 B
    swizzle atom、勿用 `SWIZZLE_NONE`；可与向量化 store 无冲突叠加。
  - P7-50（上轮）：**关闭 §52.6 的硬边界——把 B2 TMA 输入从 `BVK==32` 扩到 `BVK=64`，结论是
    `SWIZZLE_NONE` 捷径「正确但不可用」**（`docs/07` §53）。给 `b2_mma_probe.cu` 加 `TMA_SWZ128`
    宏（`BVK==32` 走 `SWIZZLE_128B`、`BVK==64` 改 `SWIZZLE_NONE` 稠密 256 B 行；`rd`/两张张量图按
    宏二选一，算术一字未动）。新增 `harness/check_bwd_b2_tma64.py`（`check_bwd_b2_tma64.out.txt`，空闲卡）
    + `harness/probe_b2_tma64_ncu.sh`（`probe_b2_tma64_ncu.out.txt`）。**Part A**：`tma64≡base64`
    **逐位相同（四位 max_abs=0）**、`base32==base64` 的 `dq/dk/dw` 也逐位（证 k8 累加序不随 `BVK` 变，
    `dgk` 2.9e-6=fp32 分组）、`tma64` vs torch 3.6e-4（tf32 底噪、比 Triton 更贴真 fp32）。**Part B**：
    `tma64/base64` **2.5–3.0×（慢）**——`SWIZZLE_NONE` 把最优点 3% 倒赔成 −180%。**ncu**：读侧修复仍在
    （L2 2.35 GB=`tma32`、srcunit 39.85M、l1tex 全局读=0、`LDGSTS`→`UBLKCP`），但 **bank conflict
    8.39M→272.6M（32×）**、shared-load wavefront 37.7M→302.0M、DRAM SOL 30%、Duration 1470 µs。**根因**：
    稠密行 stride 256 B=64 float，`64≡0 (mod 32)` ⟹ 固定列下 128 个 `h` 行全落同 bank（mma 片段恰是
    固定列扫行）。**结论**：`TMA+BVK=64` 只能走 §52.6 另一条路——**拆两个 128 B swizzle `[·,32]` atom**
    （未做，唯一待验证）；当前可用最优=cp.async@64（0.964×）或 TMA@32（0.949×），二者未叠加。**方法学**：
    跨 `BVK` 先用逐位校验判布局解码（初版 `#if` 误用 `constexpr` 非宏致 `tma32` 也走稠密、被逐位校验暴露）。
  - P7-49（上轮）：**兑现 §51.6 的「读侧下一杠杆 = TMA 输入」——B2 K2-mma 核的读放大整条消失、且净赢 ~5%**
    （`docs/07` §52）。给 `b2_mma_probe.cu` 加 `-DB2M_TMA`：操作数 `do/vn/dv`（4D 图 `{V,T,H,B}`/box
    `{BVK,BT,1,1}`）与 `h/dh`（5D 图 `{V,K,H,NT,B}`/box `{BVK,K,1,1,1}`）改 **`cp.async.bulk.tensor` +
    SWIZZLE_128B**，单线程每 stage 发 5 条 + per-stage `mbarrier`（`expect_tx`），OOB 尾行 TMA 零填充。
    新增 `harness/check_bwd_b2_tma.py`（`check_bwd_b2_tma.out.txt`）+ `harness/probe_b2_tma_ncu.sh`
    （`probe_b2_tma_ncu.out.txt`）。**Part A**：`tma`/`tma_vecst` 与 `base` `dq/dk/dw/dgk` **逐位相同
    （max_abs=0，含非整除 T=250）**、vs torch 落 tf32 底噪。**ncu（`B8T2048H8`）**：读放大**整条消失**——
    srcunit 读 **59.77M→39.85M（=DRAM 扇区数 1.0×，与 §51.4 `noasync` 预测逐值吻合）**、l1tex 全局读请求
    **=0**（SASS 10×`LDGSTS`→10×`UBLKCP`）、DRAM 读 1.28 GB 不变；L2 3.02→**2.35 GB**、`tma_vecst`→
    **2.25 GB（Triton 2.23）**。`base` 最紧 pipe L2 89.0%>DRAM 84.6%，删读放大后 **DRAM 89.0% 成最高 pipe**、
    Duration **519.8→494.2 µs**；`tma_vecst` **491.0 µs**（比 identical-op Triton 核 ~551 快 **1.12×**）。
    **CUPTI `dev`**：`tma` 0.951×/0.955×/0.933×/0.955×，`tma_vecst` 0.946×/0.926×/0.794×/0.857×（B8/B2/T512/T250）
    ——**全 shape 净赢**（对比 `noasync` 的 −21%：TMA 保留异步 + 走独立引擎）。**结论**：目标卡 CUDA B2 操作数
    路径应直接上 TMA；「省字节落最紧 pipe」再得正例（§50 的 `dgk` 是反例）。**边界**：128 B swizzle 强绑
    `BVK==32`（`BVK=64` 需拆 atom，未做）、bf16/非 GEMM 段未覆盖；本核 `tma_vecst` 已贴 DRAM 地板 1.11×。
  - P7-48（上轮）：**闭环 §50.5 唯一的待查线索——B2 K2-mma 核的「L2 字节 +35%（3.03 vs 2.24 GB）」**，
    用受控字节消融拆成两条独立、可分别消除的机制。给 `b2_mma_probe.cu` 加 `-DB2M_NOHDH`/`-DB2M_NODOV`
    （跳操作数 `cp.async`、零填充）、`-DB2M_VECST`（输出 `float2` store）、`-DB2M_NOASYNC`（同 16B 同访问
    模式换普通 `float4` load）。新增 `harness/check_bwd_b2_l2.py`（`check_bwd_b2_l2.out.txt`，空闲卡）+
    `harness/probe_b2_l2_ncu.sh`（`probe_b2_l2_ncu.out.txt`）+ `docs/07` §51。**(a) 读侧 +0.64 GB＝
    `cp.async` `.BYPASS` 路径属性**（`h/dh`、`do/vn/dv` 各 1.50×，DRAM 读不变＝L2 re-touch；`noasync`
    同请求数把 srcunit 读 59.77M→39.85M＝1.0×，故非 16B 宽度/访问模式）；**(b) 写侧 +0.09 GB＝25 条逐标量
    32-bit `STG.E` 半 sector 写**（`vecst` 后 l1tex st 12.71M→6.42M，与 Triton 逐位相同）。**叠满
    `noasync_vecst`＝2.23 GB＝Triton 逐值（无第三项）**。**Part A**：`vecst` vs `base` **逐位相同
    （max_abs=0，含非整除 T=250）**、vs torch 同落 tf32 底噪。**Part B**：`vecst` CUPTI `dev` 0.99×（B8）/
    0.98×（B2）/0.87×（T512）/0.89×（T250）；`noasync` 字节虽降但 **+21%**（丢流水）。**设计输入**：
    (1) 目标卡 CUDA B2 **一律向量化 store**（免费逐位、小 shape 10%+）；(2) 读侧下一杠杆＝**TMA 输入**；
    (3) 省字节须落**最紧 pip**（`vecst` 后 L2 反降 85%、不再最紧）。**边界**：读侧 1.5× 微架构成因待验证；
    `noasync*` 破坏流水/可能竞态仅作诊断；bf16/大 `BVK` 未测。
  - P7-47（上轮）：**更正 §49.4.3 的 bank conflict 归因：B2 K2-mma 核那 8.81M smem bank conflict
    ≈97% 来自 `dgk` 列归约（16-`kcol` 别名），不是 mma 片段读**。受控消融给 `b2_mma_probe.cu` 加
    `-DB2M_NODGK`（删整段 `dgk`）与 `-DB2M_DGKCF`（无冲突映射：一条指令只用 8 个 `kcol`，`c=l&7`/`s=l>>3`
    使 bank `4c+s` 铺满 `0..31`，两半 `ph=0/1` 覆盖 `kcol` 0–7/8–15，`__shfl_xor` 归约）。新增
    `harness/check_bwd_b2_dgk.py`（`check_bwd_b2_dgk.out.txt`，空闲卡）+ `harness/probe_b2_dgk_ncu.sh`
    （`probe_b2_dgk_ncu.out.txt`）+ `docs/07` §50。**Part A**：`nodgk`/`dgkcf` 的 `dq/dk/dw` 与 `base`
    **逐位相同**（受控）；`dgkcf` 的 `dgk` vs torch 仍 **1.31e-7**（与 base 同）。**Part B/ncu**（`B8T2048H8`）：
    冲突 **8.81M→0.29M/0.33M（−96%）**、shared wavefronts 42.65M→25.45M/34.09M；但 **Duration
    519.9/518.5/520.1µs、L2 字节三变体全 3.03GB**——部署主 shape **DRAM 84.6%/L2 88.7% bound，零兑现**。
    **小 shape**（延迟主导）`nodgk` 省 **20–23%**、`dgkcf` 省 **3–4%**（残差是循环本身）。**结论**：更正归因
    （fragment 读因 `BVP=36` 的 `4g+t` 展开本就无冲突）；`-DB2M_DGKCF` 免费位级安全但只值小 shape；§49.4.3
    的「L2 +35%（3.03 vs 2.24GB）」与 `dgk`/冲突**无关**，是下一条独立待查线索（`cp.async` 粒度/`h/dh` L2）。
  - P7-46（上轮）：**「K2 形状 CUDA 化」第二刀＝B2 GEMM 核保留 `mma.sync` + 流式操作数，首次正面**。
    在**保留 tensor core** 前提下把 B2 的 V-归约 GEMM 核写成 K2 形状（`h/dh/do/vn/dv` 按 V 分块经
    `cp.async` 流式喂 smem、寄存器只留每 warp `[16,16]` 小累加器）。新增 `harness/b2_mma_probe.cu`
    （`b2mma_launch`，`mma.sync.m16n8k8.tf32`）+ `harness/check_bwd_b2_mma.py`（`..._b2_mma.out.txt`）+
    `harness/probe_b2_mma_ncu.sh` / `ncu_b2_mma_driver.py`（`..._b2_mma_ncu.out.txt`）+ `docs/07` §49。
    **Part A**：四量落 tf32 底噪（vs torch `dq 3.6e-4/dgk 1.6e-7`），CUDA(tf32) 比 Triton 更贴 fp32。
    **Part B/C（关键）**：同 op 同 fp32 操作数，`B8T2048H8` **517µs vs Triton 同 op 核 554µs＝0.93×**
    （离地板 442µs **1.17×**，Triton 1.25×）、`B2T1024H8` 0.87×——**本专题首个跑赢 identical-op Triton 的
    CUDA 反向核**；小 shape 反慢 1.4×（固定开销、按 shape 切换）。**ncu**：regs **226→70**、occ **12.4→
    24.6%**、`long_sb` 4.11→**3.09**、DRAM **79.9→84.6% SOL**、L2 88.8% 成最高 pipe——坐实「移出大操作数
    常驻」是一阶量；残差＝L2 字节高 35% + **8.81M bank conflict**（Triton 为 0）。**扫描**：大 shape 最优
    `BVK=64/NS=2`（0.90×）。**结论**：K2 形状（tensor core + 流式操作数 + 小累加器分片）三要素由 §47 反例
    与本轮正例双向钉死；剩余头寸仍在非 GEMM 段，本核可作其 GEMM 段基座。
  - P7-45（上轮）：**把融合 B2/B3 的「头寸在字节还是发射」问清楚**——受控**字节消融**（同核只把选定
    输入 `tl.load` 换 `tl.zeros`，`tl.dot`/store/形状不动）。新增 `harness/check_bwd_b2b3_seg.py`
    （`harness/check_bwd_b2b3_seg.out.txt`）+ `harness/probe_b2b3_seg_ncu.sh` / `ncu_b2b3_seg_driver.py`
    （`harness/probe_b2b3_seg_ncu.out.txt`）+ `docs/07` §48。变体 `full`/`no_hdh`/`bf16_hdh`/`no_qkg`/
    `no_odv`/`no_all`。**Part A**：门控 `full` 与未改动融合核 5 组 shape **逐位相同（max_abs=0）**。
    **Part B（`B8T2048H8` 1745MB）**：`full` 1051µs；`no_hdh`（−1074MB）**590µs=0.56×**；`bf16_hdh`
    （h/dh 字节砍半）ncu **1.04ms=与 full 持平**；`no_qkg`（只 −201MB）**762µs=0.73×**；`no_all`（大操作数
    全不读）**170µs=0.16×**。**`ncu`**：`full` DRAM 仅 **55.9% SOL**、tensor 6.21%、issue 0.14/cyc、
    255 regs/166KB smem/occ 12.49% → **延迟/发射受限**；`h/dh`=**1.09GB=DRAM 读的 55%**；`no_hdh` occ
    **不动**、DRAM% 反降到 38%（既非占用率也非带宽）；`bf16_hdh` 省 537MB 却 Duration 不变；`no_qkg`
    **regs 255→165** 是不成比例加速真因。**结论（收窄 §47.5）**：融合核头寸＝**「大操作数（尤其 h/dh）
    常驻寄存器 → spill/发射饥饿」**，**不是**纯字节、**不是**占用率；**`h/dh` bf16 非独立杠杆**（wash，与
    §29 上界 ~10% 自洽）；K2 形状端口首位动作＝**把大操作数从寄存器保活里拿掉、只留 warp 分片小累加器**，
    判据补「**减的是字节还是寄存器保活？**」。**坑**：第一版「关 `tl.dot`＋归约替身」触发跨 lane 布局
    转换 → `no_post` **4964 spills/慢 50×**（编译器病态），故改「只渡字节」；`bf16_hdh` 的 `dev/wall`
    含 harness `.to(bf16)` 税（同 §26.5/§29.1），干净口径取 `ncu Duration`；`no_all` 的 ncu 记录为空。
  - P7-44（上轮）：**K2 形状 CUDA 端口的第一刀，以「SIMT 是反向方向」收口**——把 §46.6 首位动作的
    「放弃 `tl.dot`、手写高占用 SIMT fp32」读法试掉。新增 `harness/b2_core_probe.cu` +
    `harness/check_bwd_b2_core.py`（`..._b2_core.out.txt`）+ `harness/probe_b2_core_ncu.sh` /
    `ncu_b2_core_driver.py`（`..._b2_core_ncu.out.txt`）+ `docs/07` §47。隔离 B2 GEMM 核（`dq/dk/dw/dgk`
    四条 op）。**Part A**：手写核 vs 真 fp32 torch `dq/dk/dw` **逐位相同（max_abs=0）**（替身证明代数无误）。
    **Part B/C（反面）**：手写 SIMT 核全面慢同 op Triton 核 **3.11/2.78/4.44/5.20×**（`B8T2048H8` 1727.7
    vs 556.2µs）、仅 7.4 TFLOP/s、比完整 Triton B2 还慢 1.93×；同 op Triton GEMM 核（556µs）已是完整 B2
    （903µs）的 **62%**。**Part D（ncu，两核字节相同对照）**：手写核 **L1/TEX 89.6% bound**（`long_sb`
    9.42、2 block/SM、24.8% occ），Triton 核 **DRAM 79.8% bound**（2.67TB/s、1 block/SM、tensor 7.4%）。
    **结论**：**端口必须保留 `mma.sync` tensor core**（SIMT 判死）；**GEMM 段不值得重写**（Triton 同 op 核
    552µs vs 同字节地板 442µs 只 **1.25×**）；**§46 的 1.94× 重定位到非 GEMM 段 + `h/dh` 流量**（fp32
    1.07GB＝GEMM 核读的 84%）；动作顺序＝非 GEMM 段寄存器分片 + 同 CTA 交接 + 少搬 `h/dh` 先于 TMA 流水。
    **新方法学**：SIMT/tensor 决策先看算术强度（op/word）。
  - P7-43（上轮）：**给「融合 B2/B3 → K2 形状 CUDA」一个 go/no-go 与收益上界（不建 CUDA 核）**——
    §45.3 点名的剩余大头，但动 900 行 CUDA 前先量清。新增 `harness/check_bwd_b2b3_bound.py`
    （`..._bound.out.txt`）+ `harness/probe_b2b3_bound_ncu.sh`（`..._bound_ncu.out.txt`）+ `docs/07` §46。
    **Part A**：deploy `B8T2048H8` 融合核 `dev 1003µs`、`regs 255/spills 34/smem 114KB`、占用率
    **1 block/SM（regs 与 smem 同时钉死）**。**Part B**（`maxnreg` 255→80）：占用率**不动**、只有线性
    spill（34→320）与变慢（最坏 **3.05×**）→「降寄存器腾占用率」在 Triton 走不通。**Part C（关键）**：
    同字节、**无 `tl.dot`** 的 traffic 等价核（0 spill/0 smem/3 block/SM/带宽饱和 ~3.0 TB/s）**509µs**，
    融合核 988µs 高它 **1.94×**（`T512H4` 2.43×；ncu 字节口径 1.79× 互证）。**Part D（ncu）**：
    Eligible Warps/Scheduler **0.18**、issue 0.14/cyc、`long_sb` 41.7%、regs+smem 双 1 →**发射饥饿**。
    **结论**：融合 B2/B3 延迟受限、有 **~1.9×** 头寸且 **Triton 里够不着** → **K2 形状 CUDA 化被数据支持**
    （首位动作＝`dq/dk/dw`/`dA` 累加器按 warp 分片常驻，TMA 流水其次）。**新方法学**：用「同字节、去
    compute 的 traffic 等价核」测地板（延迟判据 + 收益上界 + 占用率能否兑现三合一）。
  - P7-42（上轮）：**把两条分别收敛的反向优化合成一条链**——CUDA B1（§15–§41）此前只在 §42 落过链
    （那条链用**未融合** B2/B3），B2/B3 融合（§31–§33）此前只在 §33 落过链（那条链用 **Triton** pipe B1），
    **两条最新收获从未同时出现**。新增 `harness/check_bwd_cudab1_fused_e2e.py`
    （`harness/check_bwd_cudab1_fused_e2e.out.txt`，空闲卡）+ `docs/07` §45；拼成
    `dAv -> CUDA B1(dht,NSTAGE) -> fused B2+B3 -> reverse cumsum`（三变体 `cu`=CUDA+未融合、
    `fu`=Triton+融合、`cf`=CUDA+融合）。**Part A**：`cf` vs FLA 六项 worst-case `dq 3.417e-3/dk 3.301e-3/
    dv 6.795e-3/dbeta 6.012e-3/dg 7.054e-3/dh0 9.365e-3` 与 §25/§26/§42 **逐值一致**；位级签名
    （`cf` vs `fu` 的 `dq`/`dh0`、`cf` vs `cu` 的 `dv`/`dh0` **逐位 0**）证实「`dv/dh0` 由 B1 定、
    `dq/dk/dg/dbeta` 由 B2/B3 形式定」。**Part B**：(1) **两条优化可加**——`cf/cu` **0.67–0.71×**（全 shape，
    融合在 CUDA B1 之上再砍 ~30%）；(2) **最优 B1 随 shape 交叉**——`cf/fu` 只在大 shape 胜（`B8T2048H8`
    **0.91×**），小 shape **1.05–1.16× 略慢**（CUDA B1@BT=16 小 grid 慢、大 shape 184.2 vs 317.9=1.73×）；
    (3) `NSTAGE≥3` 最优。两条路线 `dev/FLA` 均 **0.31–0.37×**（比 FLA 反向 dev 快 ~2.7–3.2×）。**结论**：
    目标卡 CUDA 反向最终形态＝`dAv→B1→融合 B2/B3→反向 cumsum`，**B1 按 grid 切换、B2/B3 一律融合**；
    剩余大头＝把融合 B2/B3 从 Triton 搬成 K2 形状 CUDA。**方法学（新）**：多条分别收敛的优化必须做合成实验。
  - P6-9（上轮）：**关闭 §5.3 最后一条精度待验证（$P$/`Akk` 的存储精度）**——受控消融「反向 B2 读到的
    $P$ 存 fp32 vs bf16」。新增 `harness/check_bwd_p_dtype.py`（`harness/check_bwd_p_dtype.out.txt`）+ `docs/06` §6.11。
    **Part A**（dtype 审计）：FLA 前向 `Akk` 开成 `k.dtype`（`chunk_intra.py:817-819`），fp32 入→fp32、bf16 入→bf16，
    **生产时 $P$ 本就是 bf16**（与目标卡 K1 的 bf16 `kINV` 一致）；并更正 `docs/06` §3.3 把 `302-303` 的
    `b_A.to(b_A.dtype)`（恒等 no-op）误读成「强制 bf16」——`not bf16` 注释在 `chunk_intra.py:597`，指块内衰减链。
    **Part B**：$-P^\top\bar P P^\top$ 对 bf16($P$) 放大不到 3×（$P$ 量化 2.1e-4 → 伴随量 4.9e-4）。
    **Part C**（孤立 B2，只换 `A=bf16(P).fp32`）：`dq` 逐位 0、`dk 4.9e-6/dv 4.7e-5/db 6.5e-5/dg 2.0e-6/dA 4.9e-4`。
    **Part D**（整链四组含非整除 T=250）：$P$ 降 bf16 的**归因增量** `dq/dh0=0`、其余 **≤1.3e-4**，比 bf16
    操作数底噪（§6.6 Part C 的 3e-2~5e-2）低 2–3 个数量级。**结论**：**无须为 $P$ 增存 fp32 副本**（撤回
    §5.3/§5.4「增存 fp32 版」建议），目标卡直接用 bf16 `kINV`。至此前向 K1 workspace 的**字段精度**全部有据可依。
  - P7-41（上轮）：**兑现 §24.5/§36「换 CUTLASS `PipelineTmaAsync`」＝空杠杆**——给 `b1_tma_probe.cu`
    加 `-DB1T_CUTLASS`，**只**换 pipeline 管理（`Shared` 用 `PipelineTmaAsync<NSTAGE>::SharedStorage`
    `:243-244`；`make_producer_start_state`+`producer_acquire/get_barrier/commit` `:318-371`；
    `consumer_wait/release` `:415-428/614-615`），TMA/swizzle/`mma.sync`/TMA store/两侧 fence 一字未改。
    新增 `harness/check_bwd_b1_cutlass.py`（`..._cutlass.out.txt`，空闲卡）+ `harness/ncu_b1_cutlass_driver.py`
    / `probe_b1_cutlass_ncu.sh`（`..._cutlass_ncu.out.txt`）+ `docs/07` §44。**Part A**：`cut`≡`base`、
    `cut2`≡`hnd`（=`tma2_full`）**逐位相同**（16 格，含非整除 T=250）、vs bf16 FLA `dhu` 1 ulp。**Part B**：
    `cut2/hnd = 0.985–1.003`（均值 ~0.995）——**换 CUTLASS 零性能**。**Part C**：`base`（未加 fence）在
    `B8T2048H8 BT16/32 NS=3` 仍 racy（3.76e-2/1.57e-3），`hnd`/`cut2` 全 18 格确定、`cut2`≡`hnd`——CUTLASS
    的 parity 记账天然避开手写 off-by-one。**Part D（ncu）**：逐指标重合（B8 Duration 99.04→98.82、
    `wait` 1.42→1.38、`tensor` 22.29/22.37、regs 165→164）。**结论**：§24.5 选项 (a) 关闭为空——手写
    mbarrier 已到上限；选项 (b)「接受 Triton」是正确路线；CUTLASS 值在稳健/可维护，不在性能；剩余投入去链级 B2/B3。
  - P7-40（本轮）：**关闭 P7-39 §42 的 host 税待验证**——把「`T=250 H=3` 的 1.65× 链回归＝
    `launch_impl` 每次重编码张量图」直接测掉。给 `b1_tma_probe.cu` 加 `-DB1T_MAPCACHE`（`MapSet`+
    `encode_maps`，`b1_tma_probe.cu:661-687`；encode-once 复用 `:700-714`）+ 探测符号
    `b1t_encode_only`（只编码不 launch，`:788`）/ `b1t_mapcache_miss`（`:796`）。新增
    `harness/check_bwd_b1_launch_tax.py`（`..._launch_tax.out.txt`，空闲卡）+ `docs/07` §43。
    **Part A**：纯编码 **3.8–4.0µs/次**，加进 launch 后（队列反压主导）cache 只省 **0.15–0.75µs**
    ⇒ 编码不是那笔税；cache 命中 **199/200**。**Part B**：cache vs no-cache `dh/dh0/dv2`
    **逐位相同（max_abs=0，含非整除 T=250/513）**。**Part C（关键）**：空闲卡链 wall `T250H3`
    实为 **1.049×**（非 1.65×）、`T512H4` 1.051×、`B8T2048H8` 0.951×；§42.4 的 Triton baseline
    245.9µs 是**低离群**（本轮同链 308/323/386µs，±25%）。**结论**：**撤回** host 税归因，1.65×
    不可复现；一次编码复用免费/逐位/生产该做，但不解决已测性能问题。**方法学**：`process_time`
    在队列打满时读的是 GPU 吞吐速率、非纯 CPU，隔离纯 CPU 须用不发 launch 的入口。
  - P7-39（上轮）：**第一次把 K2 形状的手写 CUDA B1 落进完整反向链**（此前全是孤立 kernel
    vs kernel，组装链 B1 一直是 Triton）。给 `b1_tma_probe.cu` 补 `dht`（终态伴随）入口
    （`b1_tma_probe.cu:258/341-357`，新增符号 `b1t_launch_dht` `:739`，语义对齐 Triton 的
    `HAS_DHT`），新增 `harness/check_bwd_b1_cuda_e2e.py`（`check_bwd_b1_cuda_e2e.out.txt`，空闲卡）
    + `docs/07` §42。**Part A**：同 workspace 操作数、`dht≠0`，CUDA-B1 vs Triton-B1 `dh/dv2/dh0`
    `max_abs` 1.4e-5~3.7e-4（=1 bf16 ulp）；去 `dht` 后 `dh` 差 3.8e-2~4.5e-2（入口生效）。
    **Part B**：CUDA-B1 链 vs FLA 六项最坏 `dq 3.423e-3/dk 3.308e-3/dv 6.795e-3/dbeta 6.012e-3/
    dg 7.064e-3/dh0 9.365e-3`，与 Triton-B1 链同阶（`dq`/`dh0` 两链逐位相同）。**Part C**：链级
    wall CUDA-B1 0.94–1.03× Triton-B1（`B8T2048H8` 快 6%）；`T250H3` 1.65× 回归＝launch 每次重编码
    张量图的 host 税（**待验证**）。**Part D**：部署粒度 `BT=16` 单核，小 shape CUDA 1.05–1.63×、
    大 shape **0.53×**（1.9× 快）——依赖 shape。**结论**：目标卡 CUDA 反向第一次端到端可跑
    （B1=CUDA、其余 Triton），drop-in 数值等价、链级不拖慢；瓶颈仍是 B2/B3。
  - P7-38（上轮）：**关闭 P7-37（§40.6）留下的唯一边界**——把 pipeline 加深（`NSTAGE` 到 6）、
    store 时序推极端（新增 `-DB1T_SDEPTH=N`，输出 tile 缓冲 1–4 深 / `i % SDEPTH` 复用），看
    输出侧（及 TMA 形状下输入侧）`fence.proxy.async` 竞态是否变得可观测。新增
    `harness/check_bwd_b1_deep_ns.py`（`check_bwd_b1_deep_ns.out.txt`，空闲卡）+ `docs/07` §41。
    **Part A（关键）**：`NR=80` 确定性扫描 `B8T2048H8`，覆盖 `BT`×`NS`（16:1–6/32:1–4）×
    `SDEPTH∈{1,2,3,4}` × 4 变体——标量对照 `scalar` 稳定复现输入侧竞态（`BT16 NS2/3/4`=**79/80**、
    `BT32 NS3`=**79/80**，`NS=1` 免疫），**`tma2` 全族（含两侧都没 fence 的）在全部 60 个组合上
    **0/80**。**答案：更深 pipeline / 不同 store 时序仍观测不到**（扫描有功率，非盲点）。细节：
    `scalar` racy 概率随 `NS` **反降**（`NS=5`18/80、`NS=6`6/80）＝强时序相关，单一配置 clean 不可外推。
    **Part B**：`tma2_full` 深配置 vs `scalar_p` `dh/dv2/dh0` **逐位相同（含非整除 T=250）**、vs
    bf16 FLA `dhu` `BT=64` 仍 1 ulp。**Part C（新正面数据）**：TMA store + `BT=16/32` 上**深 pipeline
    有正收益**（`B8T2048H8` BT16 `NS=1→4` 300.7→177.8µs=1.69×、BT32 `NS=1→3` 203.4→124.6µs=1.63×；
    `SDEPTH` 非旋钮）。**Part D**：深 `NS/SDEPTH` vs `NS=1` `dh` 逐位相同。**结论**：两侧 fence 保留
    （免费、逐位无损、本探针观测不到，不再留不确定项）；**`NSTAGE` 建议 ≥4（修正 §24 的「回到 NS=1」**，
    那是 BT=64/标量 store 的结论）。
  - P7-37（上轮）：**关闭 §39.5 的 `tma2` store 侧待验证**——给 `b1_tma_probe.cu` 加
    `-DB1T_SPROXY`（写完 `dh`/`dv2` tile 后、`cp.async.bulk.tensor` store 前一条
    `fence.proxy.async`，与 §39 输入侧对称）。新增 `harness/check_bwd_b1_tma2_race.py`
    （`check_bwd_b1_tma2_race.out.txt`，空闲卡）+ `docs/07` §40。**Part A（关键/诚实）**：`NR=40`
    确定性扫描（`B8T2048H8`）`scalar` 三次大配置全 racy（39/40），而 **`tma2` 全族（含无 fence 的
    `tma2`）全 0/40**——**输出侧 RAW/WAR 在本探针观测不到**；「观测 clean≠有保证 clean」，部署仍取
    两侧 fence。**Part B**：`tma2_full` vs `scalar_p` 16 格 `dh/dv2/dh0` **逐位相同（含非整除
    T=250）**、vs 直接 bf16 FLA `dhu` 1 ulp（4.88e-4/2.44e-4）。**Part C**：`full/tma2`
    **0.997–1.081（噪声内，免费）**；`scalar_p/tma2_full` 1.42–2.93（TMA store 收益不变）。
    **Part D**：`tma2_full` `NS=2/3` vs `NS=1` `dh` 逐位相同。**结论**：目标卡 CUDA B1 部署配置＝
    `-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY -DB1T_SPROXY`（`NSTAGE` 可 >1）；输入/输出两
    方向的 TMA 复用缓冲都需 `fence.proxy.async` 收口。**唯一边界（待验证）**：更深 pipeline / 不同
    store tile 形状改变写→store 时序时其可观测性可能变化，届时仍以 `tma2_full` 为准。
  - P7-36（上轮）：**定位并修掉 P7-35 遗留的 `NSTAGE≥2` run-to-run 竞态**——根因是手写 TMA 流水
    复用的 stage 缓冲缺一条 **`fence.proxy.async`**（generic proxy 的普通 `ld` 读 vs async proxy 的
    TMA 回填，WAR 未定序），不是 §38.3 猜的 `full`/`empty` parity。新增 `harness/check_bwd_b1_race.py`
    （`check_bwd_b1_race.out.txt`，空闲卡）+ `docs/07` §39；探针加 `-DB1T_EPROXY`/`_PRO`/`_CON`/`_ECOPY`。
    **Part A**（9 变体、`B8T2048H8` `NR=40` 逐次比 `dh`）：`base` racy（`16/2`=39/40、`32/3`=19/40，且
    每次扫描计数抖动），`noetma`/`dropread`=0，**`proxy`/`pro`/`con` 全 0/40**（两侧任一侧单独加都够），
    `esync`/`evol` 修不了，`sync`（`__syncthreads`）能修但折回串行。**Part B**（程序化）：失败 run 的
    错误是**一整个 compute warp 的 16 列 V-slice 跨连续若干 chunk**（`[0,16)`×320 / `[64,80)`×240、
    K-row 最高在 `m=3` 的 56–61 行×29）＝producer TMA 覆写了该 warp 仍在读的 stage 缓冲（WAR）。
    **Part C**：`proxy/base`=**0.96–1.02（免费）**，放弃 ETMA 回读 global（`noetma`）=贵 **4–27%**。
    **Part D**：`proxy` `NS=2/3` vs `base` `NS=1` `dh` **逐位相同（max_abs=0，12 格）**。**撤回 §38.4
    回授**：多 stage 安全**不需要** CUTLASS `PipelineTmaAsync`，一条 fence 即可；`-DB1T_EPROXY` 即修复。
    **边界（待验证）**：本轮在标量 store 的 `-DB1T_ETMA` 上做，`tma2` 的 store 侧复用握手同样建议补 fence。
  - P7-35（上轮）：把「K2 形状 CUDA B1」的累计收益**收口成 matched-bf16 的 `xFLA`**，并**发现手写
    pipeline 在 `NSTAGE≥2` 的 run-to-run 竞态**。新增 `harness/check_bwd_b1_xfla.py`
    （`check_bwd_b1_xfla.out.txt`，空闲卡）+ `docs/07` §38。三变体 `base`/`tma`/`tma2`。**正确性**：
    `tma2` vs `base`（`BT=32/64 NS=1/2`、含非整除 `T=250`）**逐位相同**、vs 直接 bf16 FLA `dhu` 1 ulp。
    **`xFLA`（ours/FLA bf16，同 `BT=64`）**：大 shape `B8T2048H8` **1.15（NS=1）/0.83（NS=2）**（§24
    标量 store 时 1.63 → TMA store 兑现「大 shape 追平 FLA」）；小 shape `B1T512H4 2.30/B2T1024H8 2.69/
    B1T250H3 2.24`（残差＝每 warp 发射效率，§24.5 上限）。**BT=16 无法在探针测**（FLA 前向只支持
    chunk 32/64），部署口径仍归 §37.3。**踩坑（新）**：run-to-run 扫描暴露 `base` kernel 在 `NSTAGE≥2`
    大 shape 上不可复现（`BT=16 NS=2` 3.64e-2、`BT=32 NS=3` 4.52e-3），`NSTAGE=1` 全 `BT` 稳定 →
    §35/§36 的「16 组逐位相同」属**观测而非保证**；根因（`full`/`empty` parity 在 `NS≥2`）**待验证**。
    **回授**：§24「最优回到 `NS=1`」同时是**唯一有保证正确**的配置；多 stage 收益须换 CUTLASS
    `PipelineTmaAsync` 才安全。**新方法学**：多 stage 手写流水须做 run-to-run 确定性检查。
  - P7-34（上轮）：**测量学复核**——澄清 `docs/07` 里两个 FLA `dhu` baseline 并更正 §25（P7-22）的
    计时结论。同一条 `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 在 §17–36 报 ~23µs（**bf16 直接
    调用**）、在 §10/§11/§25 报 ~137µs（**fp32 链内**）。新增 `harness/check_bwd_fla_dhu_baseline.py`
    （`check_bwd_fla_dhu_baseline.out.txt`，空闲卡）+ `docs/07` §37。**根因＝operand dtype**：直接调用
    FLA `dhu` bf16 = **13.1/23.3/120.0µs**、fp32 = **53.0/127.6/525.4µs**（fp32/bf16 **4.0–5.5×**）；
    链内 fp32 `dhu` 与直接 fp32 逐一对上、同链改 bf16 即回 ~23µs；机制＝fp32 载入 2× 字节 + `tl.dot`
    落 **tf32**。整条 FLA 反向链 **bf16 dev 只有 fp32 的 0.54–0.58×**。**更正 §25**：与 FLA matched bf16
    下，部署粒度 BT=16 的 bf16 B1（NS3 34.9/61.9/318.9）**慢 FLA `dhu` 2.3–2.5×**（非「反超 1.46–2.2×」）、
    整链 `dev/FLA` **0.86–0.95×**（非 0.48–0.54×）。诚实定位＝**与 FLA 反向同档（~0.9×）且省掉其前向
    重算**；工程重点（B1 粒度+store、B2/B3 融合）不变。§10/§11（两侧同 fp32）与 §17–24/§34–36（两侧同
    bf16）仍 fair，只有 §25 跨 dtype。
  - P7-33（上轮）：兑现 §35.5 的**待验证**——把 TMA store 从 `dh` 推广到同一逐标量循环里的 **`dv2`**
    （`-DB1T_TMADV`：per-warp `[BT,BW]` tile + 双缓冲 `duo` + `cp.async.bulk.tensor.4d`，与 `dh` 共用一个
    bulk group）。`dh/dv2/dh0` 与 scalar **逐位相同**（16 组，含非整除 `T=250`）；`tma2/tma` =
    **0.84–0.95（均值 ~0.89，全为正）**、`tma2/base` **0.39–0.71**、对「两条 store 全删」上界恢复
    **0.62–0.83（均值 0.75）**；`ncu` 把 `lg_throttle` **0.66→0.04 / 0.90→0.07**（nodv2 列仍 0.83/0.92，
    证明 `dv2` 标量写也是反压源）、DRAM write 基本不变（**不是带宽**）、L2 再 **−15%**、tensor 抬到
    **15.5/16.9%**。**坑**：4D 张量图 `globalStrides` 按 dim 从 1 起算，首版写反 T/H stride，`H==1` 全对、
    只有 `H>1` 时 `dv2` 错（指纹=「仅 `dv2`、仅多 head」）。**结论**：§35.5 答案=**能且逐位无损**；
    `dh`+`dv2` 全改 TMA 后 B1 下一块瓶颈只剩每 warp 发射效率（`wait` 1.40 / tensor 15.5%）。新增
    `harness/check_bwd_b1_tma_dv2.py`（`check_bwd_b1_tma_dv2.out.txt`）+ `ncu_b1_tma_dv2_driver.py` /
    `probe_b1_tma_dv2_ncu.sh`（`probe_b1_tma_dv2_ncu.out.txt`）+ `docs/07` §36。
  - P7-32（上轮）：兑现 §34.4 的**待验证**——把 B1 的 `dh` 落盘从逐标量 `st.global` 换成 **TMA store**
    （`-DB1T_TMADH`：per-warp `[KD,BW]` 未 padding tile + 双缓冲 `dho` + `cp.async.bulk.tensor.5d`），
    验证能否复现 §34.1「删掉 store」收益的那 33%。`dv2/dh0/dh` 与 scalar **逐位相同**；CUPTI 恢复占比
    `(base−tma)/(base−nodh)` = **0.58–0.84（均值 ~0.72）**、base→tma 提速 **1.28–2.48×**；`ncu` 把
    `lg_throttle` **0.66→0.04 / 0.89→0.05**（落回 nodh 档，**坐实「逐标量 store 打满队列」是真凶**）、
    DRAM write 基本不变（**不是省带宽**）、L2 字节 **−43%**（消掉 ~3× 写放大）、theoretical occ 腰斩但
    achieved 不动。**结论**：答案是**能**——TMA store 在**不做融合**下吃掉那 33% 大头；目标卡 CUDA 反向
    第一优先级＝B1 `dh`（及 `dv2`）改 TMA store；B1→B2 融合净增量只剩 B2 侧 ~5%。新增
    `harness/check_bwd_b1_tma_store.py`（`check_bwd_b1_tma_store.out.txt`）+ `harness/probe_b1_tma_store_ncu.sh` /
    `ncu_dh_roundtrip_driver.py --tma`（`probe_b1_tma_store_ncu.out.txt`）+ `docs/07` §35。
    **待验证**：`dv2` 的 TMA store 推广未实测（本轮只改 `dh`，与 §34.1 口径一致）。
  - P7-31（上轮）：量化 §30.5 遗留的「K2 形状内 B1→B2 融合」**收益上界**（不建融合 kernel）。用
    受控消融把 `dh` 落盘往返拆两半：**Part A（B1 侧）** 给 `b1_tma_probe.cu` 加 `-DB1T_NODH` 删掉
    `dh` 全局 store（保留 op1 用的 `sdh` 镜像）、`dv2/dh0` 逐位不变下 B1 快 **0.28–0.67×**（BT=64 占
    33%、BT=32 NS=2 占 **72%**）；`ncu` 定位**不是带宽**（DRAM read 不变、write 仅 −62 MB）而是
    **逐标量 `st.global` 打满 store 队列 → `lg_throttle` 0.67→0.06 + L2 流量放大 ~3×**（866→418 MB）、
    `tensor%` 10.8→16.3。**Part B（B2 侧）** `wy_dqkg` 不读 `dh`（`build_kernel(nodh=True)`）只值
    **~5%**（ncu read −92 MB、Duration 755→717、长 `long_sb`＝DRAM bound）。**Part C（关键）**：
    往返上界 ≈ B1+B2 的 **~10%**，其中 **~2/3 在 B1 侧且是 `lg_throttle`（store 效率）而非融合专属**
    → **只要把 `dh`/`dv2` 落盘换成 TMA/专职 STORE warp（镜像前向 K2 的 STORE warp）就能不融合地吃掉
    B1 侧 33%**；融合净增量只剩 B2 侧 ~5%（同 §26/§29）。**结论（修正 §30.5）**：目标卡 CUDA 反向下一
    优先级＝**B1 `dh` 改 TMA store**，而非 B1→B2 融合。新增 `harness/check_bwd_dh_roundtrip.py`
    （`check_bwd_dh_roundtrip.out.txt`）+ `harness/ncu_dh_roundtrip_driver.py` /
    `harness/probe_dh_roundtrip_ncu.sh`（`probe_dh_roundtrip_ncu.out.txt`）+ `docs/07` §34。
    **待验证**：TMA store 变体能否复现这 33%。
  - P7-30（上轮）：兑现 §32.6/§31.3/§30.5 的「B2/B3 融合」，把 `dq/dk/dg` 累加器留寄存器。部署
    粒度 `NC=1` 下 B3 只剩对角 Parts B/D（两个 `tl.dot`），折进 B2 的 `(NT,B*H)` program（B2 结束
    时已有完整 `dAkk`），加完 intra 再存一次、`dA` 不落盘。新增 `harness/check_bwd_b2b3_fused.py`
    （`check_bwd_b2b3_fused.out.txt`）+ `harness/ncu_b2b3_fused_driver.py` / `harness/probe_b2b3_fused_ncu.sh`
    （`probe_b2b3_fused_ncu.out.txt`）+ `docs/07` §33。**Part A**：融合 vs「B2+B3」同操作数 `dq/dv2`
    逐位、`dk/db/dg` ≤2.2e-7（fp32 序，含非整除 T=250/513）。**Part B**：留 `[BT,K]` 累加器只能
    `BK=K`，而 `BK=K` 必溢出（`BV=64/nw=4` 182 → `nw=8` 60 → `BV=128/nw=4` 222）；**唯一解法＝
    `BV=V`+`nw=8`** 压到 **42**，孤立 dev 相对部署基线 **0.47–0.85×**。**Part C**：落链（对比 P7-28
    就地链）六项 vs FLA **逐值同 §25/§26**、自身增量 `dq/dv` 逐位 0、`dk/dg ≤1.1e-7`；链 `dev`
    **0.83–0.93×**、`wall` **0.90–0.92×**（`B8T2048H8` 1682→1541）。**Part D（ncu）**：DRAM 字节
    **2.56→1.96 GB（−23%）**、L2 **−17%**、Duration 1193→1050µs；仍 1 block/SM（非占用率翻身，
    纯减字节）。**结论**：融合成立，但**唯一使能条件是 `BK=K`+`BV=V`**（结构前提，非可选旋钮）。
    B1（§25 流水）+ B2/B3（本轮融合）→ Triton 侧反向三元组工程化收敛。见 `docs/07` §33。
  - P7-29（上轮）：检验 §27.7/§31.3 遗留的「减少对角平铺重复读 / L1 局部性」并给**空结论**、同时
    找到真杠杆。回读 P7-24 kernel 定位两处具体冗余读——**(R1)** 对角 key 行 `j=i_ti+o_i` 即 `o_c`，
    `p_kd/p_gd`（`check_bwd_b3_mma.py:142-143`）与 `p_k/p_g`（`:98-99`）地址/mask 逐字节相同却被重读；
    **(R2)** grid `(NK*NC,NT,B*H)`、`BK=32` 令 `NK=4` 个 program 各自重读只依赖 `(chunk,head)` 的
    `[BC,BC]` 对角 `dAqk/dAkk`（`:135-138`）。新增 `harness/check_bwd_b3_l1.py`
    （`..._correctness/…_timing/…_chain.out.txt`）+ `harness/ncu_b3_l1_driver.py` / `harness/probe_b3_l1_ncu.sh`
    （`probe_b3_l1_ncu.out.txt`）+ `docs/07` §32。**Part A**：`reuse`/`bkK` 与 P7-24 **逐位相同**
    （`dq/dk/dg max_abs=0`）、`loop` 仅差 fp32 加法序。**Part B**：`reuse` 恒 **1.00×**（R1 被 CSE）、
    `BK=64` 部署大 shape 只 **0.95×**（R2 被 L2 吸收）——两处冗余读都不是杠杆；而新增
    `_intra_kernel_loop`（grid `(NT,B*H)` + `tl.range(0,NK,num_stages)`）大 shape **0.79–0.81×**，
    且 `NS=1` 反比 grid 版慢——收益全部来自**软件流水**。**Part C**：落链六项梯度 vs FLA 逐值同
    §25–27、自身增量 `dq/dv` 逐位 0，链 `dev` **0.97×**。**Part D（ncu）**：四档 **DRAM 字节逐位
    相同**、`long_scoreboard 16.74→3.98`、`Duration 257→195`、占用率反降、`L2 82–88%` 成最高 pipe。
    **结论**：B3 kernel 级优化收敛（已彻底 L2/DRAM bound），下一步只能靠 B2/B3 **融合**（§30/§31.3）。
  - P7-28（上轮）：兑现 §27.7「B3 下一杠杆」里**与 kernel 无关**的那块税——组装链让 B3 在**全新置零**的
    `dq/dk/dg` 上跑、再在 kernel 外物化三个整张 `[B,T,H,K]` 求和（`check_bwd_b1_pipe_e2e.py:162-166`）。
    这不是 FLA 契约：FLA `chunk_kda_bwd_intra` 把传入 `dq/dk/dg` 当**累加器**写到独立缓冲
    （`chunk_intra.py:554-560/657-667`）。改为 B3 **就地累积进 B2 缓冲**（grid 划分每个 `(token,k)`、
    无竞争；`db` 仍 `atomic_add`），**kernel 算术逐行不动**。新增 `harness/check_bwd_b3_accum.py`
    （`harness/check_bwd_b3_accum.out.txt`）+ `docs/07` §31。**Part A**：就地 vs「零+链外求和」
    `dq/dg` **逐位 0**、`dk/db ~1e-7`（fp32 加法次序）；FLA 契约同组数。**Part B**：就地链 vs 现链
    `dq/dv/dg` **逐位 0**、`dk/dbeta ≤9.2e-8`，vs FLA 六项最坏 `dq 3.42e-3…dh0 8.27e-3`（逐值同 §25/§26）。
    **Part C**：链 `dev` **0.84–0.88×**、`wall` **0.76–0.86×**；per-kernel 归因 elementwise
    **316→49 µs**（`B8T2048H8`）、每次少 **8 个** elementwise/memset kernel。**结论**：先消掉 B3 链时间
    的 ~16%「链外 elementwise 税」，下一步才是 §27.7 的 kernel 内对角平铺 L1 局部性优化。**约束**：B3
    契约改为消费 B2 缓冲（其后 B2 输出不可再用）。见 `docs/07` §31。
  - P7-27（上轮）：兑现 §29.5「唯一剩下的真结构项」——**B1→B2 融合原型**（`dh` 不落盘）。
    无原子的最干净形态取 **`BV=V`**（一 program 管一个 `(b,h)`、逆序扫全块、`b_dS[K,V]` 常驻、
    每块 B1 完立刻做 B2；grid `(B·H,)`=**K2 形状**）。新增 `harness/check_bwd_fused.py`
    （`harness/check_bwd_fused.out.txt`，A 正确性 + B 计时 + C `V` 分区扫描）+ `harness/probe_fused_regs.py`
    （`harness/probe_fused_regs.out.txt`）+ `harness/ncu_bwd_fused_driver.py` / `harness/probe_bwd_fused_ncu.sh`
    （`harness/probe_bwd_fused_ncu.out.txt`）+ `docs/07` §30。**Part A**：融合 B1+B2 与未融合 B1+B2
    **`dq/dk/dv/dg/dh0` 逐位相同（`max_abs=0`）**、`db ≤1.2e-7`；融合链 vs FLA 六项最坏
    `dq 3.62e-3 … dh0 7.33e-3`（bf16/tf32 底噪，同 P6-8/P7-26）。**Part B**：融合**反而慢 14–30×**
    （`B1+B2` 1.37→**19.13 ms** @`B8T2048H8`）。**归因**：融合内核**溢出 1786**（regs 255；B1 0/B2 12）；
    `ncu` DRAM **读 12.92 GB / 写 18.93 GB（31.9 GB，未融合一对仅 2.7 GB）**——多出的 ~30 GB 全是
    溢出流量，淹没原意省的 `dh` 写+读（~1.4 GB）；且融合强制的 `BV=V` 分区**本身**让 B1 单独慢
    **2.6–5.6×**（grid `4·B·H`→`1·B·H`）。**结论**：融合**代数无损、朴素 `BV=V` 单 kernel 不可行**；
    要做必须**用 §22–24 的 K2 形状**（状态多 MMA warp 分片常驻 + TMA 大块搬运 + 同 CTA 内 B1→B2 交接），
    而非一个 monolithic Triton program——「融合是唯一剩下的真结构项」成立但**前置是 K2 形状**。见 `docs/07` §30。
  - P7-26（上轮）：把 §28.7 留给 B2 的两条结构项（`h/dh` bf16、`i_k` 去重）在**真实组装链 +
    部署 dtype** 下核验（**收口/反例**轮）。新增 `harness/check_bwd_b2_hdh.py`
    （`harness/check_bwd_b2_hdh.out.txt`）+ `harness/ncu_b2_hdh_driver.py` /
    `harness/probe_b2_hdh_ncu.sh`（`harness/probe_b2_hdh_ncu.out.txt`）+ `docs/07` §29。
    **关键**：部署链 B2 操作数**全 fp32**（`_prep_b` 只转 B1 的 `qg/kg/w`），故 §26.4/§28.6 的 ncu
    （driver 默认 bf16）**不是部署 mix**——全 fp32 实测 B2 读 **1.63 GB**（非 778 MB），`h`+`dh`
    =1.07 GB=**66%**。**(1) `h/dh` bf16**：隔离 `dev` 大 shape **0.90–0.91×**、读 −37%（1.63→1.03 GB）
    但 occ 不动、SOL 仅 53–71%（证明**非带宽受限**）；**落链反而 1.07–1.24× 变慢**（harness 每次
    调用做 `.to(bf16)`），而**内核归因 B2 自身 0.90–0.97×** → 该项「**有条件成立：需前向/B1 的 bf16
    出口对齐，收益上界 ≈ 内核 10% ≈ 链 ~4%**」。**(2) `i_k` 去重**：理论省 16% 但 L2（3.02 GB）已吸收，
    `BK=128` 反因溢出写爆 1.49 GB/时长 1.56 ms → **Triton 原型不做**。**(3) 占用率不是旋钮**：唯一
    3 blocks/SM 的 `BK=BV=32` 反慢 1.25×。至此 B2 廉价/半廉价杠杆全走完，唯一剩余真结构项＝
    **B1→B2 融合（`dh` 不落盘）**或与 K2 形状（寄存器常驻 + TMA，§22–24）合做。见 `docs/07` §29。
  - P7-25：兑现 §26.6 第 (1) 条里最便宜、**算术不变**的 B2 结构项——**register-spill**。
    未改动的 B2 部署 `nw=4` 卡 Triton 255 上限并溢出 12 个（`nw=2` 溢出 190；ncu 247 regs、
    `Block Limit Registers=2`）；根因＝`q/k/g` 在 `i_k` 循环顶部加载却只在 `i_v` 循环**之后**才用
    （`check_bwd_dqkg_kernel.py:244-249` vs `:290-305`），活得贯穿整个 `i_v` 体。改法＝把这三个
    `tl.load` 挪到 `i_v` 循环之后（运算一字未改）。新增 `harness/check_bwd_b2_struct.py`
    （`harness/check_bwd_b2_struct.out.txt`）+ `harness/ncu_b2_struct_driver.py` /
    `harness/probe_b2_struct_ncu.sh`（`harness/probe_b2_struct_ncu.out.txt`）+ `docs/07` §28。
    **Part A**：late vs baseline **六项逐位相同（`max_abs=0`）**（7 组含非整除 T=250/513、B=2、BT=16/64）。
    **Part B/C**：`nw=4` 溢出 12→**0**、`nw=8` 254→224；孤立 kernel `dev` **0.84–0.89×**、`do_bench`
    0.89–0.90×。**Part D**：落链 vs FLA 六项最坏 **逐值同 §25/§26/§27**；自身增量 `dq/dk/dv/dg`
    **逐位 0**、`dbeta ≤3.7e-8`；链 `dev` **0.95–0.97×**（B8T2048H8 1961→1862）。**Part E（ncu）**：
    regs 247→210、`Block Limit Registers` 仍 **2**（须 ≤170 才到 3 blocks）、占用率 **未动**、DRAM
    字节**逐位不变**——收益＝去溢出流量。**结论**：零数值代价的 B2 收益（链 −3~5%）；B2 根本 bound
    （DRAM + 2 blocks/SM）未解，下一步＝去 `i_k` 重复读 / B1→B2 融合。见 `docs/07` §28。
  - P7-24（上轮）：兑现 §26.6 第 (2) 条——**B3 `intra` 对角块 MMA 化**。核心观察：部署粒度
    `BT=BC=16 ⟹ NC=1 ⟹ i_i≡0`，Parts A/C（B3 里仅有的 `tl.dot`）从不执行，B3 全部是对角 `for j`
    逐元素外积循环（=§26 的 `tensor 0.05%` 成因）。恒等式：门控指数移出收缩，`dq2=2^{g}⊙(Mq@(k·2^{-g}))`、
    `dkt=2^{-g}⊙(Sq@(q·2^{g}))`——FLA `SAFE_GATE=True` 分支早已这么做（`chunk_intra.py:520-526/620-637`），
    默认 `safe_gate=False` 才是逐元素。新增 `harness/check_bwd_b3_mma.py`
    （`..._correctness/…_timing/…_chain.out.txt`）+ `harness/ncu_b3_mma_driver.py` /
    `harness/probe_b3_mma_ncu.sh`（`..._b3_mma_ncu.out.txt`）+ `docs/07` §27。**Part A**：8 组
    case 四项 vs baseline/FLA 最坏 1.0e-3~1.1e-3=tf32 底噪。**Part B**：孤立 kernel `dev` **0.51–0.67×**。
    **Part C**：落链 vs FLA 六项最坏 `dq 3.42e-3…dh0 8.27e-3` **逐值同 §25.2/§26.3**（不退化）、
    自身增量 `dq 1.42e-3/dk 1.09e-3/dv 0/dbeta 3.0e-5/dg 6.08e-3`、链 `dev` **0.88–0.91×**
    （B8T2048H8 2216→1957 µs）。**Part D（ncu）**：`tensor 0.05%→2.61%（×52）`、`fma 38.8→10.1%`、
    SM `82→27%`、Duration **527→257 µs（0.49×）**，但转成 **L2/访存 bound**（L2 43→74%、DRAM 28→57%、
    `long_sb` 3.22→16.71；DRAM%×Duration 不变=总字节不变）。**结论**：B3 结构性重写兑现，但 tensor
    只到 2.6%——B3 下一杠杆是**减对角平铺重复读/L1 局部性**；链级瓶颈转到 **B2**。见 `docs/07` §27。
    **踩坑**：对角 `dAqk` 平铺的列是 chunk-内局部 `i_i*BC+o_i`（非绝对 `i_ti+o_i`），用错指纹=
    `i_t=0` 对、其余全错 O(1)。
  - P7-23（上轮）：第一次把链级新瓶颈 **B2/B3 单独量清楚**。新增
    `harness/check_bwd_b2b3_e2e.py`（`..._b2b3_e2e.out.txt`，空闲卡）+ `harness/probe_b2b3_ncu.sh` /
    `harness/ncu_b2b3_driver.py`（`..._b2b3_ncu.out.txt`）+ `harness/probe_b2b3_hdh_bf16.py`
    （`..._hdh_bf16.out.txt`）+ `harness/probe_bwd_chain_split.py`/`probe_b2b3_tiling.py`（诊断留档）
    + `docs/07` §26。**Part A**：组装链按 kernel 归因，`B8T2048H8` 上 **B2+B3 = 66.6%**（B2 906/B3 580，
    B1 13.8%）。**Part B**：免费旋钮已用尽——`num_stages` 对 B2/B3 **完全无效**（短归约/标量循环）、
    tiling 已最优、只有 B3 `num_warps=2` 值 1.16–1.27×。**Part C**：B3 `nw=2` 落链 `dq/dk/dv/dg`
    **逐位相同**、vs FLA 最坏 max_rel 与 §25.2 逐位一致，整链 dev **0.94–0.98×**。**Part D（ncu）**：
    **B2 = DRAM-bound（59% SOL）+ 247 寄存器压占用（12.4%）+ `long_sb` 6.6**；**B3 = 无 TensorCore 的
    FMA/L1-bound（tensor 0.05%、SM 82%）**（B3@64 掉到 2.29 ms 解释 §15）。**Part E**：B2 `h/dh` 换
    bf16 只值 1.08–1.16×（非量级项，+1.5–2e-3 精度）。**结论**：B2/B3 性质不同须分别处理（B2 减
    DRAM/寄存器、B3 MMA 化），本轮为**纯剖析 + 免费旋钮收口**，结构性重写留下一步。见 `docs/07` §26。
  - P7-22（上轮）：把 §17 的 Triton 软件流水从孤立原型落进真实组装链（§12 的链此前写死
    `num_stages=1`）——只把 B1 反向 chunk 循环换成前向 `tl.range(0,NT,num_stages=NS)`（算术一字
    未动）。新增 `harness/check_bwd_b1_pipe_e2e.py`（`..._pipe_e2e.out.txt`，空闲卡）+ `docs/07` §25。
    **Part A**：`NS∈{2,3,4}` 与 `NS=1` 链 `dq/dk/dv/dg/dh0` **逐位相同**、`dbeta` ≤1.5e-8；vs FLA
    最坏 max_rel 与 §12 逐位一致（含非整除 T、g=0、B=2）。**Part B**：`NS=3` 让 B1 **全 shape
    反超 FLA `dhu`**（`B2T1024H8` 122→**61.9** vs 137=`2.2×`、`B8T2048H8` 387→**319** vs 567=
    `1.78×`），整链 `dev/FLA` 从 **0.51–0.61×** 收到 **0.48–0.54×**。**Part C**：BT=16 亦有效，
    `BT=64/NS=4` OOM。**Part D**：孤立小 grid kernel 的 CUPTI 归因比链内背靠背高 ~1.3×（方法学）。
    **结论**：兑现「直接接受 Triton 版」的链级落地，Triton 侧 B1 工程化收敛，链级瓶颈转向 B2/B3。
    见 `docs/07` §25。
  - P7-21（上轮）：把 TMA B1 残差的第一块钉死。§23 相对 §22 的 K2 形状把 `long_sb` 从 2.25 抬到
    3.52，回读发现 TMA 版在 mma 循环里**逐 chunk 直读 global `E^tot`**（reg 版 `cp.async` 预取，
    `b1_reg_probe.cu:150`）——§23.1「省一次 TMA」引入的关键路径 global 标量读。新增 `-DB1T_ETMA`：
    `Stage` 加 `float e[KD]`、3D 张量图（`{KD,NT*H,B}`、`SWIZZLE_NONE`、box `{KD,1,1}`）每 chunk 多发
    一条 `cp.async.bulk.tensor.3d`，compute 改读 smem。新增 `harness/b1_tma_probe.cu` 的 `B1T_ETMA`
    分支 + `harness/check_bwd_b1_tma.py` tmae 列 + `ncu_b1_tma_driver.py --which etma` /
    `probe_b1_tma_ncu.sh` + `docs/07` §24。**Part A**：tmae vs 基线 TMA **逐位相同（max_abs=0）**、
    vs FLA bf16 底噪（含非整除 T=250）。**Part B**：kernel-only **0.77–0.98×** 基线 TMA，
    **全表最好 xFLA=1.63**（B8T2048H8 BT=64 **NS=1**：218.5→**195.0µs** vs FLA 119.4）。
    **Part D（ncu）**：`long_sb` 4.35→**3.86**（NS1）/3.46→**2.69**（NS2）/2.66→**2.05**（T512）、
    `wait` 1.60→1.24、tensor 9.68→**10.83%**（FLA 12.21%）；**占用率正式排除**（FLA achieved 6.15%
    比 ours 7.7% 还低却快 1.7×；`NS=1` 摸到 2 CTA/SM 也不成比例加速）。**结论**：`E^tot` 进流水削掉
    `long_sb` 0.6–0.8、「加深 pipeline」作废（修好后最优回到 `NS=1`）、残差 1.63× ＝每 warp 发射效率。
    下一步＝CUTLASS `PipelineTmaAsync` 或转向 Triton。见 `docs/07` §24。
  - P7-20（上轮）：兑现 §22.5 的下一步，在 K2 形状（P7-19）之上补 **TMA + warp-specialization**——
    q/k/w/dO/dV 由**专职 producer warp** 发 `cp.async.bulk.tensor`（5D 图、box `{64,1,1,BT,1}`、
    **128 B swizzle `Swizzle<3,4,3>`**），**per-stage `mbarrier`** 握手、稳态循环无 `__syncthreads()`、
    `e` 直读 global。新增 `harness/b1_tma_probe.cu` + `harness/check_bwd_b1_tma.py`
    （`..._b1_tma.out.txt`）+ `harness/ncu_b1_tma_driver.py` / `harness/probe_b1_tma_ncu.sh`
    （`..._b1_tma_ncu.out.txt`）+ `docs/07` §23。**Part A**：vs P7-19 reg 内核 **逐位相同（max_abs=0）**、
    vs FLA 落 bf16 底噪（含非整除 T=250，靠 TMA OOB 零填充）。**Part B**：新增同形状 `reg64`（BV=64/NW=4）
    对照——`BT=64` **xreg64=0.81–0.85×**（+15–20%）、`BT=32` 打平、`BT=16` 略慢；`xFLA` 收到
    **1.73–3.80×**（`B8T2048H8 BT=64` 1.73× 全表最好）。**Part C**：smem 反大（81.4→103.4KB，理论占用
    12.5→7.8%）。**Part D（ncu）**：achieved 占用 **6.32→7.75%**、tensor **8.85→10.73%**（FLA 12.12%）、
    `short_sb`/`barrier` 降，但 `long_sb` **2.25→3.52**、bank-conflict 0→26.5 万。**结论**：真实但**次线性**
    （~15%），残差重定位到 `long_sb`/浅 pipeline/swizzle 读冲突 → 下一步加深 pipeline 或换 CUTLASS
    `PipelineTmaAsync`。
  - P7-19：兑现 §21.6 的量级项落点，把手写 CUDA B1 改成 **K2 形状**——运行态 $S$ **常驻寄存器**
    （不再有 fp32 smem 主副本/staging 往返），mma 换裸 `mma.sync.m16n8k16` + `ldmatrix`（XOR 填充无冲突），
    `scale` 折进 `do` 的 bf16 操作数使 $S$ 只需一组累加器。新增 `harness/b1_reg_probe.cu` +
    `harness/check_bwd_b1_reg.py`（`..._b1_reg.out.txt`）+ `harness/ncu_b1_reg_driver.py` /
    `harness/probe_b1_reg_ncu.sh`（`..._b1_reg_ncu.out.txt`）+ `docs/07` §22。**Part A** vs FLA 落 bf16
    底噪（max_abs 4.88e-4=1 ulp）；**Part B** vs 旧基线快 **2.3–2.7×**、`xFLA` 从 ~8× 收到 **2.1–3.75×**
    （BT=16 部署粒度 81.8 vs 旧 214.4µs）；**Part C** smem 111.1→81.4KB（BT64）；**Part D（ncu）**
    bank-conflict **827392→0**、tensor **2.3→5.8%**、`wait` 1.79→1.54。**结论**：K2 形状兑现约 **2.4×**、
    把量级项吃掉大头；残差在 `wait`/并行度/TMA 深度 → 下一步在其上补 TMA + warp-specialization 流水。
  - P7-18：在动「K2 形状重写」前，用最便宜的受控实验检验 §20.6 第一优先级「并行度/mma 密度是
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

- **下一步：专题 P0–P8 全部完成 + P6-3…P6-9 / P7-3…P7-68 补测**，进入收尾/维护。
  **P7-68（本轮）已关闭 §70.6（P7-67）的两条待验证（TMA × DAMMA 生产态、TMA 折叠版落整链），并
  修正 §70.5 的归因**：`sc_tma/sc` 仍 1.039–1.057，但 `mma_tma/mma` 0.978–1.006——`ncu` 证 TMA 的
  生产代价本体是**标量 DA 段多出的 4.85M bank conflict**（标量 23.46M→28.31M、mma 17.30M 不动，与
  §69 逐值吻合），DAMMA 打开后 TMA 免费且省 L2；设计输入由「默认非 TMA」更新为「**随 DA 实现切换**」。
  链级 `pci_tma/pci` 1.006–1.024、六项逐位。**唯一遗留（低优先）**：`pci_mma/_tma`（DAMMA 进链）未做
  端到端；按机制预计 ~1%。
  **P7-67（上轮）已关闭 §67.4 的「`B2M_TMA` 与 `B2M_INTRA` 未打通」边界**：根因＝生产 launch 的
  `#if B2M_TMA_ON` 分支漏传内核末尾的 `dAqk`（`b2_mma_probe.cu:707-709`），`-DB2M_TMA -DB2M_INTRA`
  编不过（删修复后 `error: too few arguments`）。修复＝TMA 分支补同款 `#if B2M_INTRA` 守卫（算术一字
  未动）；宿主 launcher 本就 TMA-agnostic。新增 `harness/check_bwd_b2_intra_tma.py`
  （`check_bwd_b2_intra_tma.out.txt`）+ `docs/07` §70。**11 项输出逐位相同（max_abs=0，6 组含非整除
  T=250/单 chunk）**；regs 144→128、smem 159.9→149.1KB（+12B spill）、仍 1 block/SM；**CUPTI `dev`
  TMA 一律慢 8–12%**（B8T2048H8 1642.7→1788.3，与 §66.4 的 1636.7 互证）。**结论**：边界关闭、开关
  可叠加且数值无损，但**目标卡 CUDA B2 默认取非 TMA**（孤立 GEMM 核净赚 ~5% 的旋钮在生产核反号）。
  **P7-66 已兑现 §68.5 点名的「唯一量级杠杆」——`mma.sync` 化 DA**（`-DB2M_DAMMA`，需
  `B2M_DA`）：每 warp 用 `mma.sync.m16n8k8.tf32` 整块算 `dApre=dv@v^T - dw@kg^T` 的 `[BT,BT]`
  partial，partial 仍过 `dapart→da16` 归约。非 DA 七项输出**逐位不变**、只 `dApre/dA` 落 tf32 底噪；
  生产态 DA 边际（§68 的 L2→L3 消融）**185.4→83.3µs（0.45×）**、其余 shape 0.32–0.48×，**段内砍
  2–3×**；**但整核只 0.93–0.95×**（DA 占核 ~14%，`B8T2048H8` 1279.5→1213.5，`1214/998=1.22×`
  未拉进 Triton 融合核）。**结论（修正 §68.5）**：`mma` 化 DA **值得做但非翻盘点**；默认反向仍推荐 §45
  的 `dAv→B1→融合 Triton B2/B3→cumsum`（`docs/07` §69）。至此 §68.5 列出的**唯一**剩余量级杠杆已
  闭合；CUDA B2 生产核的剩余头寸全在**交付物 IO**（OUT/GATE 的 329MB 最终梯度写）与 tf32 片段读的
  smem 冲突上，均非「换一条指令」可解，**有明确需求时再动**。
  **P7-65 已把 §67.4 的「六段非 GEMM 累积」逐段归因**（`-DB2M_PROD -DB2M_HDW -DB2M_LEVEL=k`，
  只删段函数体、保留签名）：`B8T2048H8` 上 `OUT 151/DA 184/GATE 204` 三段吃非 GEMM 开销的 83%、曲线≈
  加性；`ncu` 证 OUT/GATE 成本=交付物字节、DA=标量数学。默认反向推荐 §45 的
  `dAv→B1→融合 Triton B2/B3→cumsum`（`docs/07` §68）。
  **P7-63 已兑现 P7-62/§65.6 的「唯一剩下的工程步」（B3 对角 intra 折进生产 launch）**：`-DB2M_INTRA`
  （需 `-DB2M_PROD`）/ `b2prod_intra_launch` 把核内 `dAkk`（SAND 的 `daB` 掩码）+ `dAqk` + 合流 `q/k/g`
  折进单次 launch，折进 vs「`prod`+独立 B3」`dqg/dkg` **逐位**、未触及输出逐位不变；但**大 shape 略慢
  （1.04–1.06×）、小 shape 净赢（0.88–0.89×）**，故**目标卡保持 B2/B3 两 kernel**（`docs/07` §66）。
  **P7-62 已兑现 §60.5 的 K2 形状 CUDA 化最后一条结构项（B3 intra 折进 K2 形状）**：新增 K2 形状 CUDA
  核 `b3_intra_probe.cu`，vs Triton MMA B3 落 tf32 底噪（含非整除 T、g=0）、孤立 `dev` 快 1.3–1.45×；
  **P7-61 已收口 §49.7/§63.6 的最后一条真结构项（同 CTA `dw` 交接）**：`prod_hdw` 十项输出 vs `prod`
  逐位相同、`dw` 从生产 ABI 消失、隔离段快 7–9% 但六段生产核里 wash；B2 六段的中间量至此**要么本就核内、
  要么可核内**，K2 形状 CUDA 化在**结构上闭合**（剩余头寸＝tensor 3% 的标量/发射受限，不在中间量搬运上）。
  后续可选（有明确需求时再做）：
  - **P7-57（`docs/07` §60）补非 GEMM 段第六刀——门控链（最终 `dq/dk/dg`）落进 CUDA**：给
    `b2_mma_probe.cu` 加 `-DB2M_GATE`（需 `-DB2M_DKGB`；诊断 `-DB2M_NODG`；+ `b2gate_launch`）。门控标定全在
    寄存器，唯一归约 `Σ_t kdk` 是 **warp 内 `shfl_xor` over groupID**，`g==0` lane 写 `sdgkt[KD]`、`last` 行读回；
    `B2M_GATE` 下不落 raw `dq/dk`。`dq` vs torch（同一 raw）**逐位**、`dk`/`dg` **fp32 序 4.3e-8/2.2e-7**、vs FLA
    **tf32 底噪 1.0e-3**；非 TMA/TMA@32 逐位、TMA@64 仅 `dg` 差 1 ulp；段 **+30~38%**（前五刀最贵，因片元式散读
    `q/k/g`：ncu global-load sector **9.2×**、+199 MB 输出写、`exp2f`），`gate_tma/B2` **0.73–0.78×**。**至此 B2 六项
    `dq/dk/dv2/db/dg/dA` 全搬进 CUDA。** 新增 `harness/check_bwd_b2_gate.py`（`check_bwd_b2_gate.out.txt`）+
    `harness/ncu_b2_gate_driver.py` / `harness/probe_b2_gate_ncu.sh`（`probe_b2_gate_ncu.out.txt`）。**坑**：`m_last`
    项按 lane 条件写会错加行 7/15，须按行坐标（`rowrel=(i<2?g:g+8)`）。**后续可选**：把 `q/k/g` 合流载入 smem 收回
    门控的 +30%（待验证）；把六段叠进单核做完整 K2 形状 CUDA B2。
  - **P7-56（`docs/07` §59）补非 GEMM 段第五刀——写入门梯度 `db = Σ_v dvb·v + Σ_k dkgb·kg` 落进 CUDA**：给
    `b2_mma_probe.cu` 加 `-DB2M_DB`（需 `-DB2M_OUT -DB2M_DKGB`；诊断 `-DB2M_NODB`；+ `b2db_launch`）。`dvb`(§55)/
    `dkgb`(§56) 已在同一核、相同 warp 列划分下算出，故 `db` 是**纯跨 warp 归约**：`dvb·v` 项的 `v` 需持久镜像
    `sh->dbv[warp][BT][BW]`（`:750-762`），`dkgb·kg` 直接读持久 `skg`，行组 `shfl_xor` + 8 warp smem 归约落盘
    （`:848-878`）。`db` vs torch（同一 `dvb/dkgb`）**1.3e-7~1.8e-7**、vs FLA 落 **tf32 底噪 5.5e-4~6.3e-4**、
    四布局**逐位相同（max_abs=0，含非整除 T=250）**；段只值 **+2~8%（非 TMA）/ +2~13%（TMA）**、含 OUT+DKGB+DB
    的 TMA CUDA 核到全 Triton B2 **0.85–1.04×**；ncu 增量＝`db` 落盘 `+1.4MB` + 读 smem `dbv/skg`（`LDS 104→124`），
    带宽/占用/tensor% 不动（`Duration +2.7%`）。**坑**：harness 起初把 `dw` 输入与 `dw_out` 输出设成同一张量、被
    覆写致 `dkgb` 大面积错，改独立缓冲后与 §56 专用 build 逐位。新增 `harness/check_bwd_b2_db.py`
    （`check_bwd_b2_db.out.txt`）+ `harness/ncu_b2_db_driver.py` / `harness/probe_b2_db_ncu.sh`
    （`probe_b2_db_ncu.out.txt`）。**下一步**：**门控链**（`dg` 的 `q·dq`/`kdk`/`m_last·dgk`/`kg·dkgb·beta`
    与 `dk += dkgb·gb`），落完后 B2 六项 `dq/dk/dv2/db/dg/dA` 全搬进 CUDA。
  - **P7-55（`docs/07` §58）补非 GEMM 段第四刀——最终 `dA` sandwich `-A^T diag(beta) A^T` 落进 CUDA**：给
    `b2_mma_probe.cu` 加 `-DB2M_SAND`（需 `-DB2M_DA`；+ `b2sand_launch`），`b_A`/`beta` 每 chunk 载入 smem
    （`sA/sbeta/daB`），两个 `[BT,BT]@[BT,BT]` 的 **fp32 标量**矩阵乘 + 一次 `beta` 缩放，不重读 `dApre`；`dA`
    vs torch（同一 `dApre`）**逐位相同**、四布局逐位、vs FLA 落 tf32 底噪 **1.56e-3~2.34e-3**；sandwich 只值
    `sand/da` **1.02–1.06×**、`sand_tma/da_tma` **1.03–1.10×**（ncu Duration 655.8→703.5µs）。**副产物：更正
    §57 的 `dw` 符号**（FLA `b_dw=-b_dw`，故 `dApre` 第二项为 `- dw@kg^T`，§57 写成 `+`；其自身参考两侧同号
    故未现形，接上 sandwich 对最终 `dA` 比 FLA 才暴露）。新增 `harness/check_bwd_b2_sand.py`
    （`check_bwd_b2_sand.out.txt`）+ `harness/ncu_b2_sand_driver.py` / `harness/probe_b2_sand_ncu.sh`
    （`probe_b2_sand_ncu.out.txt`）。**下一步**：`db` 的 `sum_k dkgb·kg`（需 §55 `dvb`、§56 `dkgb` 的跨 warp
    归约）与门控链，把 B2 六项全搬进 CUDA。
  - **P7-54（`docs/07` §57）补非 GEMM 段第三刀——`dA` 的两个 GEMM 贡献 `dApre = Σ_v dv·v - Σ_k dw·kg`
    （sandwich 之前）落进 CUDA**：给 `b2_mma_probe.cu` 加 `-DB2M_DA`（+ `b2da_launch`），两个收缩维都**不是
    `BT`**——base 按 warp 拥有 **V 列**标量 `dv[r,v]*v[c,v]`、`dw@kg^T` 按 warp 拥有 **输出 k 列**（`accw`
    镜像到 `swall`）标量点积，8 warp 私有 `[BT,BT]` 部分和经 smem 归约后落盘。`dApre` vs torch 落 **fp32
    底噪 5.31e-7**（比前两刀 tf32 3.6e-4 低 3 量级）、`da`≡`da_tma`≡`da_tma64` **逐位相同**；本段**贵得多**
    （`da_tma/core_tma` **1.33–1.61×**，收缩维是全维 `V`/`K` + 标量 FMA）、CUDA 核到全 Triton B2 **0.72–0.81×**；
    ncu shared wavefronts **+82%**、bank conflict **×2.8**、`LDS 88→287`、`DRAM SOL 88.87→76.17%`。新增
    `harness/check_bwd_b2_da.py`（`check_bwd_b2_da.out.txt`）+ `harness/ncu_b2_da_driver.py` /
    `harness/probe_b2_da_ncu.sh`（`probe_b2_da_ncu.out.txt`）。**下一步**：`-A^T diag(beta) A^T` sandwich、`db`
    （跨 warp 归约）、门控链；并把「收缩维 = 全维」的段改 `mma.sync`。
  - **P7-53（`docs/07` §56）兑现 §55.6「补非 GEMM 段」第二刀——`dkgb = A @ (-dw)` 落进 CUDA**：给
    `b2_mma_probe.cu` 加 `-DB2M_DKGB`（+ `b2dkgb_launch`），与 §55 输出支路同构（A 操作数走同一套索引
    重映射、无转置；`b2_mma_probe.cu:516-519` vs `:546-549` 逐字相同），B 操作数换成新 staged 的 `dw`
    （`Stage::dwm`，k-tiled，TMA 复用 `make_map_dov`）并在 B-fragment 取负（`:555-556`）。`dkgb` vs torch
    **3.64e-4**（tf32 底噪）、`dkgb`≡`dkgb_tma`≡`dkgb_tma64` **逐位相同（max_abs=0，含非整除 T=250）**；
    段只给 GEMM 核心加 **+9–13%**、CUDA 核到全 Triton B2 **0.63–0.70×**；ncu 增量＝DRAM 读 +0.07 GB(`dw`)
    + 写 65.4 MB(`dkgb`) +48.6 µs（≈2.78 TB/s，仍 DRAM-bound 88.7% SOL）、occ/tensor 不动。新增
    `harness/check_bwd_b2_dkgb.py`（`check_bwd_b2_dkgb.out.txt`）+ `harness/ncu_b2_dkgb_driver.py` /
    `harness/probe_b2_dkgb_ncu.sh`（`probe_b2_dkgb_ncu.out.txt`）。**下一步**：`dA` 后处理项、`db`
    （跨 warp 归约）、门控链。
  - **P7-52（`docs/07` §55）补非 GEMM 段第一刀——输出支路 `dv2 = beta·(A@dv)` 落进 CUDA**：加
    `-DB2M_OUT`（+ `b2out_launch`），A 操作数索引重映射（无转置）、B 复用已 staged 的 `dv`、不新增 `dv`
    字节。`dv2`/`dvb` vs torch **3.63e-4/4.40e-4**（比 Triton 5.96e-4 更贴 fp32）、三种布局**逐位相同**；
    段只给核心 **+12–25%**、仍 DRAM-bound。新增 `harness/check_bwd_b2_out.py`（`check_bwd_b2_out.out.txt`）
    + `harness/ncu_b2_out_driver.py` / `harness/probe_b2_out_ncu.sh`（`probe_b2_out_ncu.out.txt`）。
  - **P7-51（`docs/07` §54）兑现 §53.5 的唯一待验证路径——`TMA + BVK=64` 用两个 128 B swizzle atom 拆分**：
    把 K2-mma 核的 TMA 操作数改成 atom-major `[NAT=BVK/32][rows][32]`（每 atom 自己的 128 B swizzle box，
    `rd` 按 `col>>5` 选 atom），算术一字未动。`atom64`≡`base64` **逐位相同**（含非整除 T=250）、≡`tma32` 的
    `dq/dk/dw` 逐位；`atom64/base32` dev **0.853–0.974×**、`atom64/base64` 0.927–0.976×——**§49.5「BVK=64 最优
    分块」× §52「TMA 读侧修复」两收获首次叠加**（对照 `SWIZZLE_NONE` 版 2.5–3.0×）。ncu：L2 3.02→2.39 GB、
    srcunit 59.77M→39.85M、l1tex 全局读=0、**bank conflict 8.39M（非 SWIZZLE_NONE 的 272.6M）**、DRAM SOL
    90.6%、Duration 518.9→**485.9 µs**。新增 `harness/check_bwd_b2_tma_atom.py`（`check_bwd_b2_tma_atom.out.txt`）
    + `harness/probe_b2_tma_atom_ncu.sh`（`probe_b2_tma_atom_ncu.out.txt`）。**设计输入**：目标卡 CUDA B2 取
    `BVK=64` 时用 NAT 个 128 B swizzle atom、勿用 `SWIZZLE_NONE`；可与 `-DB2M_VECST` 叠加。**边界**：只测
    `BVK∈{32,64}`、fp32、`BT=16`。
  - **P7-48（`docs/07` §51）闭环 §50.5 的唯一待查线索：B2 K2-mma 核的「L2 字节 +35%（3.03 vs 2.24 GB）」**，
    受控字节消融拆成两条独立、分别可消的机制——**(a) 读侧 +0.64 GB＝`cp.async` `.BYPASS` 路径属性**
    （`h/dh`、`do/vn/dv` 各 1.50×、DRAM 读不变＝L2 re-touch；`-DB2M_NOASYNC` 同 16B 同访问模式的普通
    `float4` load 把 srcunit 读 59.77M→**39.85M＝1.0×**，证明非 16B 宽度/访问模式）；**(b) 写侧 +0.09 GB＝
    25 条逐标量 32-bit `STG.E` 半 sector 写**（`-DB2M_VECST` 合 `float2` 后 l1tex st **12.71M→6.42M**、与
    Triton 逐位相同）。**叠满 `noasync_vecst`＝2.23 GB＝Triton 逐值（无第三项）**。`vecst` vs `base` `dq/dk/dw/
    dgk` **逐位相同（含非整除 T=250）**；CUPTI `dev` **0.99×（B8）/0.98×（B2）/0.87×（T512）/0.89×（T250）**；
    `noasync` 字节降但 **+21%**（丢流水）。新增 `harness/check_bwd_b2_l2.py`（`check_bwd_b2_l2.out.txt`）+
    `harness/probe_b2_l2_ncu.sh`（`probe_b2_l2_ncu.out.txt`）。**设计输入**：目标卡 CUDA B2 一律**向量化 store**、
    读侧下一杠杆＝**TMA 输入**、省字节要落**最紧 pipe**。**边界**：读侧 1.5× 微架构成因待验证；`noasync*` 仅诊断。
  - **P7-47（`docs/07` §50）更正 §49.4.3 的 bank conflict 归因：8.81M 冲突 ≈97% 来自 `dgk` 列归约**（16-`kcol`
    别名），不是 mma 片段读。受控消融（`-DB2M_NODGK` 删 `dgk` / `-DB2M_DGKCF` 无冲突映射）：`dq/dk/dw` 逐位，
    冲突 **8.81M→0.29M/0.33M**、wavefronts 42.65M→25.45M/34.09M；但部署主 shape **Duration/L2 字节纹丝不动**
    （DRAM 84.6%/L2 88.7% bound），小 shape `dgkcf` 省 **3–4%**、`nodgk` 省 20–23%。新增
    `harness/check_bwd_b2_dgk.py`（`..._b2_dgk.out.txt`）+ `harness/probe_b2_dgk_ncu.sh`（`..._b2_dgk_ncu.out.txt`）。
    **其「新待查线索」（§49.4.3 的 L2 +35% 与 `dgk`/冲突无关）已由 P7-48/§51 关闭（答案：cp.async BYPASS 读侧
    +0.64 GB + 逐标量 store +0.09 GB）。**
  - **P7-46（`docs/07` §49）「K2 形状 CUDA 化」第二刀＝B2 GEMM 核保留 `mma.sync` + 流式操作数，首次正面**：
    保留 tensor core，把 `h/dh/do/vn/dv` 按 V 分块经 `cp.async` 流式喂 smem、寄存器只留每 warp `[16,16]`
    小累加器。`B8T2048H8` **517µs vs identical-op Triton 554µs＝0.93×**（离地板 1.17×、Triton 1.25×）、
    `B2T1024H8` 0.87×（小 shape 反慢 1.4×，按 shape 切换）；ncu regs 226→70、occ 12.4→24.6%、DRAM 79.9→
    84.6% SOL。残差＝L2 字节高 35% + 8.81M bank conflict。**下一步（真正工程化）**：把本核作 GEMM 段基座，
    向其中补非 GEMM 段（同 CTA 交接 `dA`/`dv2`/`db`）+ 降 L2 流量/消冲突；验证小 shape 是否需单独短核。
    新增 `harness/b2_mma_probe.cu` + `harness/check_bwd_b2_mma.py`（`..._b2_mma.out.txt`）+
    `harness/probe_b2_mma_ncu.sh` / `ncu_b2_mma_driver.py`（`..._b2_mma_ncu.out.txt`）。
  - **P7-45（`docs/07` §48）把融合 B2/B3 的「头寸在字节还是发射」用受控字节消融问清楚**：同核只把
    选定输入 `tl.load` 换 `tl.zeros`（`tl.dot`/store/形状不动）。`full` 1051µs、`no_hdh`（−1074MB）
    **590µs=0.56×**、`bf16_hdh`（h/dh 砍半）**1.04ms=wash**、`no_qkg`（−201MB）**0.73×**、`no_all`
    **170µs=0.16×**；ncu：`full` DRAM 55.9% SOL、tensor 6.21%、issue 0.14/cyc、1 block/SM。
    **结论**：头寸＝**大操作数（h/dh）常驻寄存器→spill/发射饥饿**，非纯字节、非占用率；`h/dh` bf16
    非独立杠杆；K2 形状端口首位动作＝**大操作数出寄存器保活、只留 warp 分片小累加器**。新增
    `harness/check_bwd_b2b3_seg.py`（`..._seg.out.txt`）+ `harness/probe_b2b3_seg_ncu.sh` /
    `ncu_b2b3_seg_driver.py`（`..._seg_ncu.out.txt`）。
  - **P7-44（`docs/07` §47）K2 形状 CUDA 端口的第一刀＝把「放弃 `tl.dot`、手写 SIMT」证伪**：新增
    `harness/b2_core_probe.cu` + `harness/check_bwd_b2_core.py`（`..._b2_core.out.txt`）+
    `harness/probe_b2_core_ncu.sh` / `ncu_b2_core_driver.py`（`..._b2_core_ncu.out.txt`）。隔离 B2 GEMM 核
    （`dq/dk/dw/dgk`）。手写核 vs 真 fp32 torch `dq/dk/dw` **逐位=0**（代数无误）；但全面慢同 op Triton 核
    **3.11/2.78/4.44/5.20×**（`B8T2048H8` 1727.7 vs 556.2µs）、仅 7.4 TFLOP/s、比完整 B2 还慢 1.93×；
    同 op Triton GEMM 核已是完整 B2 的 **62%**。ncu（两核字节相同）：手写核 **L1/TEX 89.6% bound**、
    Triton 核 **DRAM 79.8% bound**（`tl.dot` 抬高算术强度 → 直接撞带宽）。**结论**：端口**必须保留
    `mma.sync`**（SIMT 判死）；GEMM 段不值得重写（同 op 核 552µs vs 同字节地板 442µs 只 1.25×）；§46 的
    1.94× 重定位到**非 GEMM 段 + `h/dh` 流量**（fp32 1.07GB＝GEMM 核读的 84%）。**下一步（真正工程化）**＝
    **保留 `mma.sync` 的 K2 形状 B2/B3**（累加器 warp 分片 + TMA/`ldmatrix` 整块 + 非 GEMM 段同 CTA 交接），
    并按 §47.5 优先减 `h/dh` 字节。**方法学**：SIMT/tensor 决策先看算术强度（op/word）。
  - **P7-43（`docs/07` §46）给「融合 B2/B3 → K2 形状 CUDA」go/no-go 与收益上界**（不建 CUDA 核）：
    新增 `harness/check_bwd_b2b3_bound.py`（`..._bound.out.txt`）+ `harness/probe_b2b3_bound_ncu.sh`
    （`..._bound_ncu.out.txt`）。deploy `B8T2048H8` 融合核 `dev~1000µs`、`255 regs/34 spills/114KB smem`、
    **1 block/SM（regs+smem 双钉）**；`maxnreg` 255→80 占用率不动、只加 spill（34→320）与变慢（最坏 3.05×）；
    **同字节、无 dot 的 traffic 等价核 509µs**（带宽饱和 ~3.0 TB/s）→ 融合核高它 **1.94×**；ncu `Eligible
    Warps/Scheduler 0.18`、issue 0.14/cyc、`long_sb` 41.7% = 发射饥饿。**结论**：有 ~1.9× 延迟头寸、Triton
    够不着 → **K2 形状 CUDA 化被数据支持**；**下一步（真正的工程化）**＝把 `dq/dk/dw [BT,K]` + `dA [BT,BT]`
    累加器按 warp 分片常驻寄存器（先），TMA 输入流水（次），即 B1 §19–§24 的配方。**新方法学**：同字节去
    compute 的 traffic 等价核测地板。
  - **P7-42（`docs/07` §45）把两条分别收敛的反向优化合成一条链**：CUDA B1 与 B2/B3 融合此前从未同时在
    一条链里（§42 链用未融合 B2/B3、§33 链用 Triton B1）。拼成 `dAv -> CUDA B1 -> fused B2+B3 -> 反向
    cumsum`（`harness/check_bwd_cudab1_fused_e2e.py`）。**Part A**：`cf` vs FLA 六项 worst-case 与
    §25/§26/§42 **逐值一致**、位级签名（`dq/dh0` vs `fu`、`dv/dh0` vs `cu` 逐位 0）。**Part B**：两条
    优化**可加**（`cf/cu` **0.67–0.71×**），但**最优 B1 随 shape 交叉**（`cf/fu` 大 shape 0.91×、小 shape
    1.05–1.16×），`NSTAGE≥3` 最优；两条路线 `dev/FLA` 均 **0.31–0.37×**。**结论**：最终形态＝`dAv→B1→
    融合 B2/B3→反向 cumsum`，B1 按 grid 切换、B2/B3 一律融合；剩余大头＝把融合 B2/B3 搬成 K2 形状 CUDA。
    **新增方法学**：多条分别收敛的优化必须做**合成实验**（本轮既证明可加，又暴露单链看不到的部署约束）。
  - **P7-41（`docs/07` §44）关闭 §24.5/§36 的「换 CUTLASS `PipelineTmaAsync`」——空杠杆**：`-DB1T_CUTLASS`
    只换 pipeline 管理，`cut`≡`base`/`cut2`≡`hnd` **逐位相同**、`cut2/hnd` **0.985–1.003**（零性能）、
    `ncu` 逐指标重合（`wait` 1.42→1.38 纹丝不动）。**手写 mbarrier 已到该结构上限**；§24.5 选项 (b)「直接
    接受 Triton 版」是正确路线；CUTLASS 的价值是**稳健/可维护**（天然确定、代码更短），不是性能。附带：
    CUTLASS 的 parity 记账天然避开手写版 `NS≥2` 的 WAR 竞态（`base` 在 `BT16/32 NS=3` 仍 racy）。
  - **P7-40（`docs/07` §43）关闭 §42 的 host 税待验证**：加 `-DB1T_MAPCACHE`（张量图 encode-once
    复用，`b1_tma_probe.cu:661-687/700-714`）+ 探测符号 `b1t_encode_only`/`b1t_mapcache_miss`，
    新增 `harness/check_bwd_b1_launch_tax.py`（`..._launch_tax.out.txt`）。**纯编码 3.8–4.0µs/次**，
    但加进 launch（队列反压主导）只省 **0.15–0.75µs**；cache 命中 **199/200**、数值**逐位无损**；
    空闲卡链 wall `T250H3` 实为 **1.049×**（非 1.65×）、`B8T2048H8` 0.951×，§42.4 的 Triton
    baseline 245.9µs 是**低离群**（±25%）。**结论**：**撤回**「张量图重编码＝host 税」的归因；
    一次编码复用免费/生产该做但不解决性能问题；小 shape ~5% 差是 BT=16 CUDA 核本身（非 host）。
    下一步工程重点仍是 B2/B3（链瓶颈 ~66%）与 CUDA B1 一起做成目标卡 CUDA。
  - **P7-39（`docs/07` §42）第一次把 K2 形状手写 CUDA B1 落进完整反向链**：补 `dht` 入口
    （`b1t_launch_dht`）、验证 CUDA-B1 是 Triton-B1 的逐位 drop-in（`dh/dv2/dh0` 1 bf16 ulp），
    CUDA-B1 链六项梯度 vs FLA 与 Triton 链同阶（`dq`/`dh0` 逐位相同），链级 wall 0.94–1.03× 不拖慢
    （`B8T2048H8` 快 6%）。**遗留**：(1) ~~`launch_impl` 每次 `cuTensorMapEncodeTiled` 的 host 税
    在小 shape（`T=250`）显形为 1.65× 回归~~ **已由 P7-40 关闭（结论：不可复现、非编码所致）**；
    (2) 部署粒度 `BT=16` 下 CUDA B1 相对 Triton 单核「依赖 shape」（小 shape 同档/略慢、大 shape
    快 1.9×）。下一步工程重点仍是把 B2/B3（链瓶颈，占 ~66%）与这个 CUDA B1 一起做成目标卡 CUDA。
  - **~~P7-36 的边界（待验证）~~（已解决）**：`tma2` 的 TMA store 侧复用握手（`dho`/`duo` + `cp.async.bulk`
    store + `bulk_group`）**已由 P7-37 补 `-DB1T_SPROXY`、并由 P7-38（`docs/07` §41）关闭边界**——把
    pipeline 加深到 `NS=6`、输出缓冲加深到 `SDEPTH=4`（含单缓冲极端）后，`tma2` 全族在 60 个配置上仍
    **0/80**（标量对照稳定 racy），即两侧 `fence.proxy.async` 在本探针**观测不到但编程模型必需**，
    部署保留、免费、逐位无损；且 TMA store + `BT=16/32` 上深 pipeline 有正收益（`NSTAGE≥4`）。
  - **P7-34（`docs/07` §37）澄清两个 FLA `dhu` baseline 并更正 §25**：同一条
    `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 的 ~23µs（bf16 直接调用）与 ~137µs（fp32 链内）
    差的是 **operand dtype**（fp32 载入 2× + `tl.dot` 落 tf32，实测 fp32/bf16 = 4.0–5.5×），非 direct/chain。
    与 FLA **matched bf16** 下：bt16 B1 **慢 FLA `dhu` 2.3–2.5×**、整链 `dev/FLA` **0.86–0.95×**（更正
    §25 的「反超」与 0.48–0.54×）。诚实定位＝**与 FLA 反向同档（~0.9×）且省掉 FLA 前向状态重算**；
    工程重点（B1 粒度+store、B2/B3 融合）不变。**惯例（新增）**：`docs/07` 的 `xFLA` 此后一律
    matched-dtype（引 FLA 绝对时间必标 dtype）。新增 `harness/check_bwd_fla_dhu_baseline.py`
    （`check_bwd_fla_dhu_baseline.out.txt`）。**下一步**：审计 `docs/07` §26–33 是否还有别的跨 dtype
    计时（已查：均为链内部分比或正确性，暂未发现）。
  - **P7-33（`docs/07` §36）把 TMA store 从 `dh` 推广到 `dv2`**：`-DB1T_TMADV`（per-warp `[BT,BW]` tile +
    双缓冲 `duo` + `cp.async.bulk.tensor.4d`，与 `dh` 共用一个 bulk group，OOB 尾行由 TMA 按 `T` 裁）。
    `dh/dv2/dh0` 与 scalar **逐位相同**（16 组，含非整除 `T=250`）；`tma2/tma` = **0.84–0.95（均值 ~0.89，
    全为正）**、`tma2/base` **0.39–0.71**、对全删上界恢复 **0.62–0.83（均值 0.75）**；`ncu` `lg_throttle`
    **0.66→0.04 / 0.90→0.07**（nodv2 仍 0.83/0.92）、DRAM write 不变（**不是带宽**）、L2 再 **−15%**、
    tensor **15.5/16.9%**。坑：4D 张量图 `globalStrides` 按 dim 从 1 起算，写反 T/H stride 时 `H==1` 全对、
    仅 `H>1` 时 `dv2` 错。**结论**：§35.5 答案=**能且逐位无损**；B1 的 `dh`+`dv2` 全改 TMA 后下一块瓶颈
    只剩每 warp 发射效率（`wait` 1.40 / tensor 15.5%）。**下一步**：此专题实际工作已全部收敛（两条 store
    优化 + 全部 bound 定位），仅剩「CUTLASS `PipelineTmaAsync` 换手写 mbarrier」这一更大工程，留待有需求时做。
  - **P7-32（`docs/07` §35）兑现 §34.4 待验证＝把 B1 的 `dh` 落盘改成 TMA store**：`-DB1T_TMADH`
    （per-warp `[KD,BW]` tile + 双缓冲 + `cp.async.bulk.tensor.5d`），`dh/dv2/dh0` 与 scalar **逐位相同**；
    CUPTI 恢复占比 **0.58–0.84（均值 ~0.72）**、base→tma **1.28–2.48×**；`ncu` 把 `lg_throttle`
    **0.66→0.04 / 0.89→0.05**（坐实真凶）、DRAM write 基本不变（**不是带宽**）、L2 **−43%**、theoretical occ
    腰斩但 achieved 不动。**答案：能**——TMA store 不融合即吃掉那 33% 大头。**下一步**：目标卡 CUDA 反向
    第一优先级＝B1 `dh`（及 `dv2`）改 TMA store；B1→B2 融合降到最后（净增量只剩 B2 侧 ~5%）。
    **该待验证（`dv2` 推广）已由 P7-33 关闭（答案：能且逐位无损）。**
  - **P7-31（`docs/07` §34）量化「K2 形状内 B1→B2 融合」收益上界**：受控消融拆 `dh` 落盘往返——
    B1 去掉 `dh` store（`-DB1T_NODH`）快 **0.28–0.67×**（`dv2/dh0` 逐位），`ncu` 证明是**逐标量
    `st.global` 的 `lg_throttle`（0.67→0.06）+ L2 放大 ~3×**、**不是带宽**；B2 不读 `dh` 只值 **~5%**。
    **往返上界 ≈ B1+B2 的 ~10%，且 ~2/3 是 B1 侧 store 效率（非融合专属）**。**下一步（修正 §30.5）**：
    目标卡 CUDA 反向优先把 **B1 的 `dh`/`dv2` 落盘改成 TMA/专职 STORE warp**（镜像前向 K2 的 STORE
    warp），而非 B1→B2 融合（融合净增量只剩 ~5%）。**该待验证已由 P7-32 关闭（答案：能）。**
  - **P7-30（`docs/07` §33）B2/B3 融合兑现**：把 B3 对角（`NC=1` 只剩 Parts B/D）折进 B2，
    `dq/dk/dg` 累加器留寄存器、`dA` 不落盘；`dq/dv2` 逐位、其余 ≤2.2e-7（fp32 序），链 `dev`
    **0.83–0.93×**、`wall` **0.90–0.92×**、DRAM 字节 **−23%**。**唯一使能条件＝`BK=K`+`BV=V`+`nw=8`**
    （把 `BK=K` 的溢出压到 42），是结构前提而非可选旋钮。（其「下一步：K2 形状融合」已由 P7-31 量化，
    上界 ~10% 且多为 store 效率问题 → 降级。）
  - **P7-28（`docs/07` §31）把 B3 的「链外 elementwise 税」消掉**：组装链让 B3 在**零缓冲**上跑、
    再在 kernel 外物化 3 张 `[B,T,H,K]` 求和（`check_bwd_b1_pipe_e2e.py:162-166`）。FLA 的
    `chunk_kda_bwd_intra` 把传入 `dq/dk/dg` 当**累加器**（`chunk_intra.py:554-560/657-667`）。改成
    B3 **就地累积进 B2 缓冲**（grid 划分每个 `(token,k)`、`db` 仍 atomic），kernel 算术逐行不动：
    `Part A` `dq/dg` 逐位 0、`dk/db ~1e-7`（加法次序）；`Part B` 落链 `dq/dv/dg` 逐位 0、vs FLA 逐值
    同 §25/§26；`Part C` 链 `dev` **0.84–0.88×**、`wall` **0.76–0.86×**、elementwise 归因
    **316→49 µs**（`B8T2048H8`）。**下一步**＝§27.7 剩下的 kernel 内对角平铺 L1 局部性优化。
  - **P7-27（`docs/07` §30）把 §29.5「唯一剩下的真结构项」＝ B1→B2 融合做成原型并证伪朴素形态**：
    融合 B1+B2 在代数/数值上**无损**（与未融合 `dq/dk/dv/dg/dh0` **逐位相同**），但朴素 `BV=V`
    单 kernel **溢出 1786、DRAM 流量爆 17×（31.9 GB vs 2.7 GB）、端到端慢 14–30×**，且融合强制的
    `BV=V` 分区**本身**让 B1 单独慢 2.6–5.6×。**下一步（真结构项）＝用 §22–24 的 K2 形状**
    （状态多 MMA warp 分片常驻寄存器 + TMA 大块搬运 + 同 CTA 内 B1→B2 交接）来实现融合，
    **而不是把整条 `[K,V]` 状态塞进一个 monolithic Triton program**（待验证，须在目标卡以 K2 为基座测）。
  - **P7-26（`docs/07` §29）把 B2 的两条结构项收口**：部署链 B2 实为**全 fp32**，读 1.63 GB
    （`h/dh`=66%）；`h/dh` bf16 只值 B2 内核 ~10% 且**需原生 bf16 出口**（harness 运行期转换落链净负）；
    `i_k` 去重被 L2 吸收不划算；占用率非旋钮。（其遗留的「B1→B2 融合」已由 P7-27 证伪朴素形态。）
  - **P7-23（`docs/07` §26）第一次把链级新瓶颈 B2/B3 量清楚**：组装链按 kernel 归因，
    `B8T2048H8` 上 **B2+B3 = 66.6%**（B2 906/B3 580/B1 只 13.8%）。免费旋钮已用尽（`num_stages`
    对 B2/B3 **无效**、tiling 已最优、只有 B3 `num_warps=2` 值 1.16–1.27×，落链 `dq/dk/dv/dg`
    **逐位相同**、整链 0.94–0.98×）。**`ncu`**：**B2 = DRAM-bound（59% SOL）+ 247 寄存器压占用
    （12.4%）+ `long_sb` 6.6**；**B3 = 无 TensorCore 的 FMA/L1-bound（tensor 0.05%、SM 82%）**
    （B3@64 掉到 2.29 ms）。B2 `h/dh` 换 bf16 只值 1.08–1.16×（非量级项）。**下一步（需新 kernel）**：
    依 B2/B3 两种不同性质分别做**结构性重写**——B2 减 DRAM 流量（`h/dh` bf16/drop 按 `i_k` 重读/
    拆 tile 降寄存器）与 B3 的 MMA 化（把逐元素外积改走 `tl.dot`），维持 BT=16 块链。
  - **P7-22（`docs/07` §25）把 §17 的 Triton 软件流水从孤立原型落进真实组装链**：§12 的链此前写死
    `num_stages=1`，本轮只把 B1 反向 chunk 循环换成前向 `tl.range(0,NT,num_stages=NS)`（算术一字
    未动）。`NS=3` 下 B1 **全 shape 反超 FLA `dhu`**（`B2T1024H8` 122→**61.9 µs** vs 137=`2.2×`、
    `B8T2048H8` 387→**319** vs 567=`1.78×`），整链 `dev/FLA` 从 **0.51–0.61×** 收到 **0.48–0.54×**，
    且 `dq/dk/dv/dg/dh0` 与 `NS=1` **逐位相同**（`dbeta` ≤1.5e-8）。**方法学**：孤立小 grid kernel 的
    CUPTI 归因比链内背靠背高 ~1.3×，链级口径一律用链内归因。**结论**：兑现「**直接接受 Triton 版**」
    这条路线（§24.5 的第二选项）的链级落地——Triton 侧 B1 工程化收敛，**链级瓶颈从 B1 转向 B2/B3**。
  - **P7-21（`docs/07` §24）把 TMA B1 残差的第一块钉死**：定位到 §23 相对 §22 的 `long_sb` 抬升
    （2.25→3.52）来自 §23.1「省一次 TMA」而在 mma 循环里**逐 chunk 直读 global `E^tot`**；用一条
    512 B 的 `cp.async.bulk.tensor.3d`（`-DB1T_ETMA`）搬回 smem 后 `long_sb` 降 0.6–0.8、`wait`
    降 0.16–0.36、tensor 逼近 FLA，**算术逐位不变**，kernel-only 0.77–0.98×、**全表最好 xFLA=1.63**
    （198.8µs vs FLA 119.8，B8T2048H8 BT=64 **NS=1**）。**同时排除两条路线**：(a)「加深 pipeline」
    作废（`E^tot` 修好后最优回到 `NS=1`，2 CTA/SM 比深 stage 值）；(b)「占用率」作废（FLA achieved
    6.15% 比 ours 7.7% 还低却快 1.7×）。**残差 1.63× 已定位到每 warp 发射效率**（`wait` 1.24 vs
    FLA 0.96、`short_sb` 0.90 vs 0.50）——手写 mbarrier 最小实现的结构上限。**下一步工程化＝要么换
    CUTLASS `PipelineTmaAsync`（把 producer/consumer 与 `cp.async.bulk` 完成延迟交给成熟实现），
    要么直接接受 Triton 版**（§14/§17 已到 1.0×/0.85×）；不再在 stage 深度/占用率/ILP/padding 上找。
  - **P7-20（`docs/07` §23）已兑现 P7-19 留下的「TMA + warp-specialization」**：专职 producer warp 发
    `cp.async.bulk.tensor`（128 B swizzle）、per-stage `mbarrier` 握手、无 `__syncthreads`，正确性
    **逐位保持**，achieved 占用 6.32→7.75%、tensor 8.85→10.73%、`short_sb`/`barrier` 降，`BT=64` 只快
    **~15–20%**（同形状 `xreg64=0.81–0.85×`）、`xFLA` 收到 **1.73×**。**结论：TMA+warp-spec 是次线性一步，
    没吃掉 2× 残差**；残差重定位到 **`long_sb`（TMA/mbarrier 完成延迟）+ 浅 pipeline + swizzle 读的
    bank-conflict**。**接续本轮的下一步工程化＝在这个 TMA 骨架上 (1) 加深 pipeline（更多 stage / K 方向拆
    两半喂）、(2) 把手写 mbarrier 换成 CUTLASS `PipelineTmaAsync`（K2 已用的成熟实现）**；不再在
    ILP/padding/加 warp 上找。
  - **P7-19（`docs/07` §22）已把「K2 形状重写」的第一步兑现**：手写 CUDA B1 从 wmma-from-smem 改成
    寄存器常驻状态 + `mma.sync`/`ldmatrix` + XOR swizzle，比旧基线快 2.3–2.7×、`xFLA` 从 ~8× 收到 2.1–3.75×。
    P7-20 已在其上补完 TMA 输入路径（见上）。
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
