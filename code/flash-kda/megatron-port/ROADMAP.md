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
| `src/kda_ref.py` | 最小可运行参考实现（纯 PyTorch、fp64、自带自测；M7 起支持 GVA + state_v_first） | [x] |
| `harness/check_port_fla.py` | 参考 vs FLA 正反向数值锚定脚本（GPU） | [x] |
| `harness/check_port_fla_m7.py` | 参考 vs FLA 门控/布局边界锚定（M7，GPU） | [x] |
| `harness/check_port_fla_m8.py` | 参考 vs FLA `chunk_size`/pad/varlen/CP 锚定（M8-1，GPU；CP 用真实 NCCL rank） | [x] |

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
- [ ] **M8-2** CP **反向**（`dhu` 的 CP 前处理 + 跨 rank `dht` 收发）与 `varlen × CP` 组合的
      梯度锚定（M8-1 Part C 只锚了前向）。

### M9 Megatron 接口对齐
- [ ] **M9-1** 按 `docs/02` §7 检查表逐项对齐 Megatron `kda.py` 的调用契约
      （SBHD 转置、`dht` 初值、`A_log/dt_bias` fp32、`gdn_pre_gated_delta_rule_fusion` 缺口）。

### M10 端到端对拍
- [ ] **M10-1** 在 Megatron 路径上跑 KDA，对本参考/ FLA 做端到端前反向对拍。

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

## 当前进度 / 下一步

- 当前进度：M0–M7 完成 + **M8-1 完成**（docs 00–06 + 参考实现 + FLA 数值锚定
  [默认 + 门控/布局边界 + chunk/pad/varlen/CP 前向]）。
- 本轮（M8-1）：`harness/check_port_fla_m8.py` + `docs/06-chunk-varlen.md`。**更正前提**：
  KDA 的 `chunk_size` 只有 **32/64**（16 抛 `ValueError`，`chunk.py:390-392`）。Part 0（fp64 CPU）
  证明块尾零填充与 varlen 逐序列循环都只是重排、与逐 token 递推 `≤6.3e-14`；Part A
  `C∈{32,64}×T∈{16,33,64,130,250,256}`、Part B varlen（含 GVA、核内融合门控、per-seq
  初/终态）前反向全落 **bf16 底噪 ≤7.9e-3**；Part C 真实 NCCL 多 rank CP——**对齐分片时与
  不分片逐位相同（`max_abs=0`）**，并测出 CP 的两条硬约束（`total%world==0` 且
  `part%chunk_size==0`，否则差 2.2e-3 / 丢尾）。单机 NCCL 需 `NCCL_NVLS_ENABLE=0`。
- 下一步：**M8-2**（CP 反向 + varlen×CP 梯度锚定）、M9（Megatron `kda.py` 调用契约对齐）、
  M10（端到端对拍）。

## 阻塞

（无）
