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
| `src/kda_ref.py` | 最小可运行参考实现（纯 PyTorch、fp64、自带自测；M7 起支持 GVA + state_v_first） | [x] |
| `harness/check_port_fla.py` | 参考 vs FLA 正反向数值锚定脚本（GPU） | [x] |
| `harness/check_port_fla_m7.py` | 参考 vs FLA 门控/布局边界锚定（M7，GPU） | [x] |

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
- [ ] **M8-1** `chunk_size∈{16,32,64}` × `T` 的 pad 语义 sweep；varlen（`cu_seqlens`）；
      CP 路径。

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

## 当前进度 / 下一步

- 当前进度：M0–M7 完成（docs 00–05 + 参考实现 + FLA 数值锚定 [默认 + 门控/布局边界]）。
- 本轮（M7）：`harness/check_port_fla_m7.py` + `docs/05-anchor-boundaries.md`。把 M6 的锚定
  扩到四个边界——`safe_gate`/`lower_bound`、`allow_neg_eigval`、GVA（$H_V>H$）、`state_v_first`；
  并给参考实现补 GVA（`repeat_interleave`）与 state_v_first（双端转置）。Part 0（fp64 CPU）
  GVA 与展开递推自洽 `7.4e-15`、svf 精确；A–D 共 12 组前反向全落 bf16 底噪 ≤~8e-3
  （`dA_log/ddt_bias` 相对 3.6e-2 但绝对 ≤1.2e-3）；M6 输出重跑逐位不变。
- 下一步：M8（`chunk_size∈{16,32,64}`×`T` 的 pad 语义 sweep、varlen `cu_seqlens`、CP 路径）、
  M9（Megatron `kda.py` 调用契约对齐）、M10（端到端对拍）。

## 阻塞

（无）
