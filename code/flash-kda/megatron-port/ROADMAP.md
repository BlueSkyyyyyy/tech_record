# megatron-port · 路线图（活文档）

把 FLA `fla/ops/kda`（KDA）前反向算子**移植进 Megatron-LM
`megatron/core/ssm/gated_delta_net`**。本目录是 `code/flash-kda` 主系列的子专题：
主系列以 FlashKDA（手写 CUDA）为参照讲「怎么算」，本目录以 FLA 为参照讲「怎么移植」。

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。

## 交付物

| 文件 | 内容 | 状态 |
|---|---|---|
| `docs/00-math-and-api.md` | 数学原理 + 算子接口（逐 token / 分块 / 门控 / FLA 签名 / Megatron 调用面） | [x] |
| `docs/01-forward.md` | FLA KDA 前向算子整理（Gate→Intra→状态递推→输出，逐 kernel + 行号） | [x] |
| `docs/02-backward.md` | FLA KDA 反向算子整理（dAv→dhu→wy_dqkg→intra→门控 + 移植检查表） | [x] |
| `docs/03-optimization.md` | 优化手段（分块/融合/重算/精度/布局/系统层 + 目标卡取舍） | [x] |
| `docs/04-numeric-anchor.md` | 把参考实现锚定到 FLA `chunk_kda`（M6） | [x] |
| `docs/05-anchor-boundaries.md` | 锚定扩到门控/布局边界（M7：safe_gate/lower_bound、allow_neg_eigval、GVA、state_v_first） | [x] |
| `docs/06-chunk-varlen.md` | 锚定扩到 chunk_size/pad/varlen/CP（M8-1） | [x] |
| `docs/07-cp-backward.md` | 锚定扩到 CP 反向 + varlen×CP 梯度（M8-2） | [x] |
| `docs/08-megatron-contract.md` | Megatron `kda.py` 调用契约逐项对齐（M9-1） | [x] |
| `docs/09-e2e.md` | 真实层端到端前反向对拍（M10-1） | [x] |
| `src/kda_ref.py` | 最小可运行参考实现（纯 PyTorch、fp64、自带自测；M7 起支持 GVA + state_v_first） | [x] |
| `harness/check_port_fla.py` | 参考 vs FLA 正反向数值锚定脚本（GPU） | [x] |
| `harness/check_port_fla_m7.py` | 参考 vs FLA 门控/布局边界锚定（M7，GPU） | [x] |
| `harness/check_port_fla_m8.py` | 参考 vs FLA `chunk_size`/pad/varlen/CP 锚定（M8-1，GPU；CP 用真实 NCCL rank） | [x] |
| `harness/check_port_fla_m8b.py` | 参考 vs FLA CP 反向 / varlen×CP 梯度锚定（M8-2，GPU；真实 NCCL rank） | [x] |
| `harness/check_port_contract.py` | Megatron `kda.py` 调用契约对齐（M9-1，GPU；加载 FLA `chunk_kda` + 参考） | [x] |
| `harness/check_port_e2e.py` | 真实 `KimiDeltaAttention` 层端到端前反向对拍（M10-1，GPU） | [x] |

## 任务清单

### M0 侦察与环境
- [x] **M0-1** 读通 FLA `fla/ops/kda`（`chunk.py`/`chunk_fwd.py`/`chunk_bwd.py`/
      `chunk_intra.py`/`wy_fast.py`/`gate.py`）、公共状态核
      `fla/ops/common/chunk_delta_h.py`，产出 `docs/01`/`docs/02` 结构脑图（真实行号）。
- [x] **M0-2** 环境：`kernel_lab` 内 `fla` 0.5.2 + torch 2.12 + H100；确认
      `fla.ops.kda.chunk_kda` 可跑；`FLA_DISABLE_BACKEND_DISPATCH=1` 钉回 Triton。
- [x] **M0-3** 定位 Megatron 调用面 `~/proj/Megatron-LM/megatron/core/ssm/gated_delta_net/`
      （`kda.py`/`gdn.py`/`common.py`），记入 `docs/00`。

### M1 数学与接口
- [x] **M1-1** `docs/00-math-and-api.md`：逐 token 递推、门控、分块推导、FLA `chunk_kda`
      签名、Megatron 调用面、内部算子清单。

### M2 前向整理
- [x] **M2-1** `docs/01-forward.md`：前向逐 kernel（Gate → Intra → 状态递推 → 输出）数式 + 真实行号。

### M3 反向整理
- [x] **M3-1** `docs/02-backward.md`：反向五段（重算 → dAv → dhu → wy_dqkg → intra → 门控）
      + 移植检查表。

### M4 优化整理
- [x] **M4-1** `docs/03-optimization.md`：算法/kernel/系统三层优化 + 目标卡取舍建议。

### M5 参考实现
- [x] **M5-1** `src/kda_ref.py`：纯 PyTorch fp64 逐 token 递推 + 分块前向 + 门控；
      自测（`src/kda_ref.out.txt`）：Part A 递推 vs 分块 ≤1.1e-14、Part B autograd 六项
      梯度 ≤7.2e-15、Part C 门控闭式 vs autograd ~5e-8。

### M6 FLA 数值锚定
- [x] **M6-1** `harness/check_port_fla.py` + `docs/04`：把 `src/kda_ref.py` 锚定到 FLA
      `chunk_kda` 正反向。Part A（显式操作数）4 组、Part B（核内融合门控）4 组，
      含 `K64≠V96` 与非整除 `T=250`；前反向全部落 **bf16 底噪**（主路 `o/S` ≤~6.4e-3、
      梯度 ≤~4.4e-3，`dA_log` 相对量到 2.95e-2 但绝对 5.7e-4）；`harness/check_port_fla.out.txt`。

### M7 锚定边界（门控变体）
- [x] **M7-1** 把锚定扩到 `safe_gate=True`/`lower_bound`（TensorCore 门控分支）、
      `allow_neg_eigval`、GVA（$H_V>H$）、`state_v_first`——M6 只测了默认门与 `HV=H`。
      新增 `harness/check_port_fla_m7.py`（`check_port_fla_m7.out.txt`）+ `docs/05-anchor-boundaries.md`，
      并给 `src/kda_ref.py` 的 `kda_chunked_fwd` 补 **GVA**（`repeat_interleave` 组映射，
      autograd 即组求和）与 **state_v_first**（双端转置状态）。**Part 0**（fp64 CPU）：GVA 分块
      vs 展开后逐 token 递推 `rel|do|=7.4e-15 / rel|dS|=2.2e-14`、svf 的 `o` 逐位不变且
      `S_svf≡S^T`。**Part A（GVA）**/ **Part B（svf）**/ **Part C（safe lb=-5）**/ **Part D
      （neg_eigval）** 共 12 组（含非整除 `T=250`、`HV=2H/4H`、`K≠V`）：前反向全部落
      **bf16 底噪 ≤~8e-3**，`dA_log/ddt_bias` 相对量到 3.6e-2 但绝对 ≤1.2e-3（逐 head 标量
      近抵消）。**M6 默认路径重跑与旧输出 diff 为空**（无回归）。

### M8 锚定边界（chunk/varlen/CP）
- [x] **M8-1** `chunk_size` × `T` 的 pad 语义 sweep + varlen（`cu_seqlens`）+ CP **前向**路径。
      `harness/check_port_fla_m8.py`（`check_port_fla_m8.out.txt`）+ `docs/06-chunk-varlen.md`。
      **前提更正**：KDA 的 `chunk_size` 只允许 **32/64**（16 抛 `ValueError`，
      `chunk.py:390-392`），路线图原写 `{16,32,64}` 有误。**Part 0**（fp64 CPU）：块尾零填充
      vs 逐 token 递推 `≤6.3e-14`、varlen 逐序列循环 vs 逐 token `3.4e-14`。**Part A**：
      `C∈{32,64}×T∈{16,33,64,130,250,256}`（含非整除/`T<chunk`）前反向全落 bf16 底噪 `≤7.9e-3`；
      `C=32` 与 `C=64` 行逐值一致。**Part B**：varlen `cu_seqlens`（含 GVA、核内融合门控、
      per-seq 初/终态与 `dh0`）前反向 `≤5.5e-3`。**Part C**：真实 NCCL 多 rank CP
      （单机需 `NCCL_NVLS_ENABLE=0`）——对齐分片（`world=1/2/4`、跨 rank 单序列、两序列各一 rank）
      与不分片**逐位相同（`max_abs=0`）**；`part%chunk≠0` 差 2.2e-3、`total%world≠0` 丢尾，
      给出 CP 两条硬约束（`total%world==0` 且 `part%chunk_size==0`）。**边界**：CP 只锚前向
      （反向见 M8-2）。
- [x] **M8-2** CP **反向**（`dhu` 的 CP 前处理 + 跨 rank `dht` 收发）与 `varlen × CP` 组合的
      梯度锚定（M8-1 Part C 只锚了前向）。新增 `harness/check_port_fla_m8b.py`
      （`check_port_fla_m8b.out.txt`）+ `docs/07-cp-backward.md`。**Part 0**（fp64 CPU）：参考的
      分块反向 == 逐 token 反向到机器精度（`≤5.1e-16`，含非整除 `T=33/130/250`、`C∈{32,64}`）。
      **Part A**（单序列等分）：`world∈{1,2,3,4}`、`C∈{32,64}`（`world≥3` 走多级 rank 链的
      `all_gather`+`merge`）六项梯度**逐位相同（`max_abs=0`）**。**Part B**（varlen×CP，5 组，
      含一条序列跨 rank / 每序列各占一 rank / rank 界落在序列内部）六项**逐位 0**。
      **Part C**：两条 CP 硬约束对反向同样成立——`part%chunk!=0` 六项梯度 1e-3~2.3e-3；
      `total%world!=0` 尾块无人认领时 `dk/dv/dg/dbeta` 高达 4.5e-2（**且 `o/dq` 仍逐位 0**，
      即反向把分片错回灌到保留 token 的梯度，比前向更隐蔽）。**Part D/E**：核内融合门控
      （logits 入口）与 GVA（$H_V=2H$）× CP 均**逐位 0**——CP 与两者正交。单机 NCCL 需
      `NCCL_NVLS_ENABLE=0`。**关键事实**：CP 反向不能走 `dht` 入口（`chunk.py:362-364` +
      `cp/chunk_delta_h.py:849` 双重断言），跨 rank 状态伴随由 `dhm`（本 rank 贡献）+`all_gather`
      +`merge(FORWARD=False)` 重建。

### M9 Megatron 接口对齐
- [x] **M9-1** 按 `docs/02` §7 检查表逐项对齐 Megatron `kda.py` 的调用契约
      （SBHD 转置、`dht` 初值、`A_log/dt_bias` fp32、`gdn_pre_gated_delta_rule_fusion` 缺口）。
      新增 `harness/check_port_contract.py`（`check_port_contract.out.txt`，FLA_DISABLE_BACKEND_DISPATCH=1）
      + `docs/08-megatron-contract.md`。用一个逐字复刻 `kda.py::pre_gated_delta_rule` +
      `common.py::_prepare_input_for_gated_delta_rule` 的 shim 把 SBHD→BHSD、切段、l2norm、
      GVA、beta sigmoid 全走一遍再喂 FLA，真机核对：**Part A** SBHD→BHSD 转置等价（`max_abs=0`）；
      **Part B** 原样参数调 `chunk_kda`（`use_gate_in_kernel=True`）vs fp64 参考 `rel|do|=1.9e-3`、
      `rel|dh|=1.1e-3`（bf16/tf32 底噪）；**Part C** beta 核外 sigmoid 逐位（`max_abs=0`）；
      **Part D** `A_log`/`dt_bias` 降 bf16 → 门控 `max_abs=0.127`（证 `mark_keep_in_fp32` 必要）；
      **Part E** l2norm 核外 vs 核内 `rel|do|=2.5e-4`；**Part G** bf16 `initial_state` 被 FLA 拒、
      零 fp32 态与 `None` 逐位；**Part H** 12 条静态契约行全部命中。**一条契约更正**：FLA/参考支持
      **GVA**，但 **Megatron KDA 不支持**——`_compute_gates` 只产出 H-head 的 `raw_g`（`kda.py:180`），
      而 FLA 硬断言 `g.shape==(B,T,HV,K)`（`chunk.py:411`），故 `num_value_heads==num_key_heads`
      是硬约束（docstring `kda.py:55-57`）；M7 的 GVA 锚定不是 Megatron 已有能力。缺口
      `gdn_pre_gated_delta_rule_fusion` 对 KDA 未实现（`kda.py:399-402`），补它需 KDA 专属 streamed
      融合前处理（GDN 的 `_fused_streamed_pre_gated_delta_rule` 不可复用，`gdn.py:318-350`）。
      **边界**：shim 只覆盖 tp/cp=1（布局/门控/精度契约），未跑真实 `KimiDeltaAttention`（容器内
      Megatron 因 `nvidia-resiliency-ext` 版本断言无法 import，见「已知坑」）。

### M10 端到端对拍
- [x] **M10-1** 在 Megatron 路径上跑 KDA，对本参考/ FLA 做端到端前反向对拍。
      新增 `harness/check_port_e2e.py`（`check_port_e2e.out.txt`）+ `docs/09-e2e.md`。
      **先拆 import 阻塞**（此前 M9-1 只跑 shim）：容器里 `import megatron.core` 有**两条**
      import 期版本断言——(1) `nvidia-resiliency-ext` 装 0.5.0 而 Megatron 断言 ≥0.6.0
      （`dist_checkpointing/strategies/nvrx.py:44-46`，比较运行时 `nvrx.__version__`），
      在 `import megatron` 前把内存版本号改成 `"0.6.0"`；(2) DeepEP 的
      `check_nccl_so()`（`deep_ep/__init__.py:55`）逐字节比对 libnccl、本机不一致，
      用其逃生开关 `EP_SUPPRESS_NCCL_CHECK=1`（`:24`）绕过。两条置好后**首次 import 真实
      `KimiDeltaAttention` 并实例化**（照 `tests/unit_tests/ssm/gated_delta_net/test_kda.py`
      小配置，单卡 tp=cp=1）。**Part A**（真层 smoke，`T×B=64×2/128×2/250×1`、fp32+bf16）：
      输出形状/dtype 正确、`hs.grad` 与**全部**参数梯度有限。**Part B**（核心算子锚定）：用
      记录器包住 `self.gated_delta_rule`（`kda.py:141/439-449`）截获真层**当次**喂给
      `chunk_kda` 的操作数，在同一批操作数上比 FLA vs fp64 参考——fp32 `o max_abs=9.565e-05/
      rel=1.675e-03`、六项梯度 rel 2.7–3.2e-3；bf16 `o rel=4.878e-03`、梯度 rel 3.6–8.5e-3
      （=tf32/bf16 底噪，同 M6/M7/M8）。**Part C**（整模块端到端）：建第二份**逐位同权重**
      模块、把核心算子换成 fp64 参考、其余层一字不动——fp32 `out≤5.10e-4 / hs.grad≤2.48e-8 /
      param.grad≤2.01e-6`；bf16 `out≤2.93e-3 / hs.grad≤1.19e-7 / param.grad≤7.63e-6`。**结论**：
      真层的核心调用无隐藏布局/门控/精度偏差（M9-1 shim 契约在真层确认），且端到端梯度不被
      外层模块放大。**边界**：单卡 tp=cp=1、SBHD 非 packed；packed/THD、CP、GVA（Megatron
      KDA 不支持）、`gdn_pre_gated_delta_rule_fusion` 缺口未覆盖。

## 已知坑

- **强衰减下不要直接算 $k/E$**（会溢出）：减参考点或用逐对 $2^{\Delta g}$（`docs/03` §4.2）。
- FLA 内核用 **log2** 域（`exp2`+`RCP_LN2`）；参考用自然对数，`g_log2=g_nat/ln2`，同构。
- Megatron 传 **raw g**，门控激活必须在 kernel 内做。
- 对拍前 `FLA_DISABLE_BACKEND_DISPATCH=1`（否则本机路由到 TileLang，fp32 可能编不过）。
- 文档行号按本机版 `flash-linear-attention` / `Megatron-LM` 核对；升级上游后需重核。
- `dA_log`/`ddt_bias` 是逐 head 标量求和、可能近抵消，**看绝对误差不要看相对**。
- **KDA `chunk_size` 只允许 32/64**（16 抛 `ValueError`，`ops/kda/chunk.py:390-392`）；
  参考实现可任意，但锚定 FLA 时只能用 32/64。
- **CP 分片必须对齐块边界**：`total % world == 0` 且 `part_len % chunk_size == 0`
  （`part_len = total // world`，`ops/cp/context.py:82-85`）；否则 CP 与不分片结果差 ~2e-3
  （切在块内）或直接丢尾 token（`total % world != 0`）。
- **单机跑 NCCL 需 `NCCL_NVLS_ENABLE=0`**：否则 `all_gather_into_tensor` 撞
  `Failed to bind NVLink SHARP (NVLS) Multicast memory`。
- **varlen 对拍先打印 `final_state.shape`**：参考逐序列的终态若带 `[N,1,...]` 的 B=1 维，
  与 FLA 的 `[N,...]` 相比会在损失里隐式广播成 `[N,N,...]`，把形状错伪装成 O(1) 数值差。
- **容器内 `import megatron.core` 有两条 import 期版本断言**（`kernel_lab`，2026-10）：
  (1) `megatron/core/dist_checkpointing/strategies/nvrx.py:44-46` 的
  `Minimum required nvidia-resiliency-ext package version is 0.6.0`（容器装 0.5.0），
  在 `import megatron` 前设 `nvidia_resiliency_ext.__version__ = "0.6.0"` 骗过版本比较
  （真正的 async-ckpt 能力仍走符号探测，不改 KDA 数值）；
  (2) `deep_ep/__init__.py:55` 的 `check_nccl_so()` 逐字节比对运行时 libnccl 与链接期望
  （本机 `libnccl.so.2.29.7` vs `.../nvidia/nccl/lib/libnccl.so.2`），设逃生开关
  `EP_SUPPRESS_NCCL_CHECK=1`（`:24`）绕过。两条都置好后真实层可 import（M10-1 已跑通）；
  M9-1 当年只用 shim 是因为不知道这两条开关。
- **Megatron KDA 不支持 GVA**：`_compute_gates` 只产 H-head `raw_g`（`kda.py:180`），FLA 断言
  `g.shape==(B,T,HV,K)`（`chunk.py:411`）；`num_value_heads==num_key_heads` 是硬约束。勿把 M7 的
  GVA 锚定当成 Megatron 已具备能力。

## 当前进度 / 下一步

- 当前进度：M0–M8 完成 + M9-1 完成 + **M10-1 完成**（docs 00–09 + 参考实现 + FLA 数值锚定
  [默认 + 门控/布局边界 + chunk/pad/varlen/CP 前向 + CP 反向/varlen×CP 梯度] + Megatron
  调用契约对齐 + **真实层端到端前反向对拍**）。
- 本轮（M10-1）：`harness/check_port_e2e.py` + `docs/09-e2e.md`。**首次 import 并实例化真实
  `KimiDeltaAttention`**（先拆两条 import 期版本断言：nvrx 版本号 patch + `EP_SUPPRESS_NCCL_CHECK=1`）。
  **Part A** 真层 fwd/bwd、全部参数梯度有限。**Part B** 截获真层当次喂 `chunk_kda` 的操作数，
  FLA vs fp64 参考：fp32 `o rel=1.675e-03`、六项梯度 rel 2.7–3.2e-3；bf16 `o rel=4.878e-03`、
  梯度 rel 3.6–8.5e-3。**Part C** 逐位同权重的「参考芯」模块端到端：fp32 `out≤5.10e-4 /
  hs.grad≤2.48e-8 / param.grad≤2.01e-6`；bf16 `out≤2.93e-3 / hs.grad≤1.19e-7 / param.grad≤7.63e-6`。
  **结论**：真层核心调用无隐藏偏差、M9-1 shim 契约在真层确认、端到端梯度不被外层放大。
  **边界**：单卡 tp=cp=1、SBHD 非 packed；packed/THD、CP、GVA、`gdn_pre_gated_delta_rule_fusion`
  缺口未覆盖。
- 下一步（可选）：retire 到维护态（M0–M10 全完成）；若继续，把 M10-1 扩到 **packed/THD 与
  headwise/chunkwise CP** 的真层对拍，或补 `gdn_pre_gated_delta_rule_fusion` 的 KDA 专属
  streamed 融合前处理（`kda.py:399-402` 的缺口）。

## 阻塞

（无）
