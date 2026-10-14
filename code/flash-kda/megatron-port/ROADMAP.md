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
| `docs/10-packed-thd.md` | 真层 packed / THD（varlen）端到端对拍（M10-2） | [x] |
| `docs/11-padded-thd.md` | 真层 packed 的 padding（`cu_seqlens_q_padded`）语义对拍（M10-3） | [x] |
| `docs/12-cp-e2e.md` | 真层 context-parallel（cp>1）数值锚定 vs fp64 参考（M11-1） | [x] |
| `docs/13-thd-cp.md` | 打通 packed THD *多序列* CP（逐序列 zigzag 分片 + 预建 route；M11-2） | [x] |
| `docs/14-pre-gated-fusion.md` | `gdn_pre_gated_delta_rule_fusion` 的 KDA 缺口：结构差异 + KDA 专属 fused pre-GDR 契约与锚定（M12-1） | [x] |
| `docs/15-fused-pregdr-kernel.md` | KDA 专属 fused pre-GDR 的 Triton 核族实现 + 数值锚定/整层 drop-in/计时（M13-1） | [x] |
| `src/kda_ref.py` | 最小可运行参考实现（纯 PyTorch、fp64、自带自测；M7 起支持 GVA + state_v_first） | [x] |
| `src/kda_pregdr_fused.py` | KDA 专属 fused pre-GDR 的 Triton 核族（conv+SiLU / L2-norm / beta sigmoid）+ autograd 包装 + 自测（M13-1） | [x] |
| `harness/check_port_fla.py` | 参考 vs FLA 正反向数值锚定脚本（GPU） | [x] |
| `harness/check_port_fla_m7.py` | 参考 vs FLA 门控/布局边界锚定（M7，GPU） | [x] |
| `harness/check_port_fla_m8.py` | 参考 vs FLA `chunk_size`/pad/varlen/CP 锚定（M8-1，GPU；CP 用真实 NCCL rank） | [x] |
| `harness/check_port_fla_m8b.py` | 参考 vs FLA CP 反向 / varlen×CP 梯度锚定（M8-2，GPU；真实 NCCL rank） | [x] |
| `harness/check_port_contract.py` | Megatron `kda.py` 调用契约对齐（M9-1，GPU；加载 FLA `chunk_kda` + 参考） | [x] |
| `harness/check_port_e2e.py` | 真实 `KimiDeltaAttention` 层端到端前反向对拍（M10-1，GPU） | [x] |
| `harness/check_port_e2e_packed.py` | 真层 packed / THD（varlen）端到端对拍（M10-2，GPU） | [x] |
| `harness/check_port_e2e_padded.py` | 真层 packed 的 padding 语义对拍（M10-3，GPU） | [x] |
| `harness/check_port_e2e_cp.py` | 真层 cp>1（headwise/chunkwise）前反向锚定 vs fp64 参考（M11-1，GPU；真 NCCL） | [x] |
| `harness/check_port_e2e_cp_thd.py` | packed THD 多序列 CP 锚定（逐序列 zigzag 分片 + `--shard global` 受控对拍；M11-2，GPU；真 NCCL） | [x] |
| `harness/check_port_pre_gated_fusion.py` | `gdn_pre_gated_delta_rule_fusion` KDA 缺口的复现 + KDA 专属 fused pre-GDR 参考锚定 / drop-in（M12-1，GPU） | [x] |
| `harness/check_port_pregdr_kernel.py` | KDA 专属 fused pre-GDR **Triton 核族**的数值锚定（vs eager/纯 torch 参考）+ 整层 drop-in + wall/dev 计时（M13-1，GPU） | [x] |

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
- [x] **M10-2** 把 M10-1 的真层对拍扩到 **packed / THD（varlen）**路径：多序列沿 token 维
      拍平、`cu_seqlens` 切分、`batch==1`（`kda.py:272-289`），三组 pack 含非 64 整除段长
      （`[0,64,128]`/`[0,50,130]`/`[0,64,100,250]`）。新增 `harness/check_port_e2e_packed.py`
      （`check_port_e2e_packed.out.txt`）+ `docs/10-packed-thd.md`。参考芯改成 **packed-aware**
      （按相邻 `cu_seqlens` 逐序列循环调 `kda_chunked_fwd`，与 FLA varlen 语义一致，M8-1）。
      **Part A** 三组 pack × fp32/bf16 真层 fwd/bwd 全部有限、输出 `[T_total,1,H]` 正确。
      **Part B**（核心算子）截获真层当次喂 `chunk_kda` 的操作数（`cu_seqlens=[0,64,128]`、
      `q=(1,128,4,32)`），FLA vs 逐序列 fp64 参考：fp32 `o rel=1.993e-03`、六项梯度 rel
      3.2–3.6e-3；bf16 `o rel=3.676e-03`、梯度 rel 4.3–9.4e-3——落 tf32/bf16 底噪，**同时
      证明 packed 段间状态隔离无泄漏**（若串状态误差会 O(1)）。**Part C**（整模块）FLA 芯 vs
      参考芯：fp32 `out≤5.23e-4 / hs.grad≤2.45e-8 / param.grad≤1.95e-6`；bf16 `out≤2.93e-3 /
      hs.grad≤1.34e-7 / param.grad≤5.25e-6`，与 M10-1 SBHD 同量级。**Part D** packed == unpacked
      自洽（`[T,B,H]` vs `transpose(0,1).reshape(T*B,1,H)`、`cu_seqlens=[0,T,2T]`）：bf16 输出与
      `hs.grad` **逐位相同（0）**、fp32 `out≤3.16e-5 / hs.grad≤2.4e-9`（差仅来自归约顺序）、
      param.grad ≤2.4e-7。**结论**：packed 的 `cu_seqlens` 记账（切段/隔离/conv1d 跨段）在核心
      与整模块两个粒度都无隐藏偏差；M9-1 shim 契约在 packed 真层再确认。**边界**：单卡 tp=cp=1、
      packed 无 padding（`cu_seqlens_padded` 未用）、cp>1 的 packed CP（zigzag）未覆盖。
- [x] **M10-3** 关闭 M10-2（`docs/10` §7）明确留下的边界——**带 padding 的 packed / THD**
      （`cu_seqlens_q_padded` 走 `_resolve_cu_seqlens` 的 padding 分支，`common.py:482-506`；层把
      *padded* 边界交给核心与 causal-conv1d，`kda.py:275-281/447/580/598-607`）。新增
      `harness/check_port_e2e_padded.py`（`check_port_e2e_padded.out.txt`）+ `docs/11-padded-thd.md`。
      两组用例（真实 `[0,50,110]/[0,50,100,220]` → padded `[0,64,128]/[0,64,128,256]`）。**Part A**
      带 padding 真层 fwd/bwd 全有限。**Part B** 核心锚定（FLA vs 按 padded 边界逐段处理的 fp64
      参考）：fp32 `o rel=2.558e-03`、梯度 rel 2.7–3.6e-3；bf16 `o rel=3.876e-03`、梯度 rel
      2.6–7.2e-3（=tf32/bf16 底噪）。**Part C（归因）** 逐操作数翻转 pad：只有门控 **`g`** 会动真实
      token 输出（fp32 1.29e-5@t=49、bf16 1.53e-5@t=48），`q/k/v/beta` **逐位 0**——pad 的 `g` 参与
      块内累计衰减的稳定化参考量（chunk 尾舍入）。**Part D（关键安全属性）** 逐序列改自身 pad 尾：
      **跨序列隔离精确 `max_abs=0`**（fp32/bf16、2 用例 ×3 seed），自身 slot 尾影响最后几个真实
      token `9.1e-5`（fp32）/`1.95e-3`（bf16）。**Part E** 整模块 padded（FLA 芯 vs fp64 参考芯）
      fp32 `out=5.07e-4/hs.grad=2.27e-8/param.grad=1.64e-6`、bf16 `out=2.93e-3/…/5.72e-6`（同
      M10-1/M10-2）。**Part F** 带 padding 的真实 token vs 无 padding 布局 `out≤1.08e-4`(fp32)/
      `1.95e-3`(bf16)。**结论**：带 padding 的 packed 路径数值健全、**跨序列隔离精确**，pad 对真实
      token 的影响仅剩「自身 slot 尾经门控 `g` 的块尾舍入」、低于部署底噪。**边界**：单卡 tp=cp=1、
      `pad_between_seqs` 布局、cp>1 padded CP 未覆盖。

### M11 context-parallel（cp>1）端到端
- [x] **M11-1** 把 M10 的真层对拍扩到 **cp>1**（M10-1/2/3 的「边界」都写着单卡 cp=1；M8 只在
      **FLA 算子**级锚过 CP）。新增 `harness/check_port_e2e_cp.py`（真 NCCL，`mp.spawn`）+
      `docs/12-cp-e2e.md`。两段式：Stage 1 建 cp=1 真层、把核心换成 fp64 参考芯（M10-2 的
      `make_ref_gdr`）在全序列上跑前反向 dump 黄金；Stage 2 起 `world` 个 NCCL rank、用同权重
      重建真层、喂同一全输入的 **zigzag 分片**（`get_pos_emb_on_this_cp_rank`，Megatron 单测同款
      配方），把 rank-local 张量对 Stage 1 张量的同分片比。**实测**（`T=256`、`world=2/4`、fp32/bf16、
      `headwise`/`chunkwise`、packed 单序列/SBHD B=2）：`o`/`hs.grad`/`param.grad` 相对误差
      bf16 **4e-3–1e-2**、fp32 **0.9e-3–1.6e-3**，与 cp=1 的 M10-1/2 bf16 整模块锚定同量级，
      `world=2/4` 同阶（跨 rank 状态传递未引入偏差）。**方法学（两条，新增）**：(i) CP 对拍损失必须
      count-invariant（`sum` 而非 `mean`，否则 cp=2 输入梯度整条差 2×，实测 `|grad_cp|/|grad_ref|=1.99`）；
      (ii) 复制型参数的梯度在 CP 下是**部分和**，对全序列参考比前须在 CP 组 `all_reduce(SUM)`
      （不归约时 `in_proj.weight` 相对误差停在 0.55–0.64，归约后回底噪）。**遗留边界**：packed THD
      **多序列**（plain `cu_seqlens`、无 prebuilt THD CP routes）对不上（`o rel` O(1)）且触发真层
      fallback `FutureWarning`（`context_parallel_layout/conversion.py:120`）——属**输入构造不符真层 CP
      契约**，非 KDA 算错；Megatron 自己的多序列 CP 单测用 **SBHD + padded**（即本篇 validated 的
       sbhd 行）。`gdn_pre_gated_delta_rule_fusion`（`kda.py:399-402`）、GVA 仍未覆盖。
- [x] **M11-2** 关闭 M11-1 明确留下的**最后一条边界——packed THD 多序列 CP**（`docs/13-thd-cp.md`）。
      新增 `harness/check_port_e2e_cp_thd.py`（`check_port_e2e_cp_thd.out.txt`，空闲卡；`mp.spawn`
      真 NCCL）。**根因更正**：M11-1 的 O(1) 不是「缺 prebuilt route / 需 padded 布局」——`conversion.py`
      本就会从 `cu_seqlens` 重建等价 route；真错在**输入分片粒度**。Megatron 的 THD CP zigzag 是
      **逐序列**切 `2*cp_size` 块（`routes.py:63/_build_thd_layout_segments:79-90`），而 M11-1 用的
      `get_pos_emb_on_this_cp_rank`（`rope_utils.py:181`）按**整条 token 轴** zigzag——单序列重合、
      多序列不同（`cu=[0,64,128],cp=2`：整轴给 rank0 `{0..31,96..127}`、逐序列给
      `{0..15,48..63,64..79,112..127}`）。修法＝**逐序列 zigzag 分片** + 构造 batch 时调公开 API
      `prebuild_thd_cp_partition_routes`（`routes.py:257`）消除 fallback `FutureWarning`。**实测**：
      `world=2/4` × `headwise/chunkwise` × fp32/bf16 × 3 组 pack（`[0,64,128]`/`[0,64,160]` 非均匀/
      `[0,64,128,192,256]`）：bf16 `o/hs.grad/param.grad ~5e-3–8e-3`、fp32 `~1e-3`，与 M11-1 单序列/
      SBHD 多序列及 M10 的 cp=1 整模块锚定**同量级**；`FutureWarning` 归零。**受控对拍**（同脚本
      `--shard global`）复现 M11-1 的 O(1)（`o rel 1.10`/`0.92`）→ 一眼分开「算法错」与「输入构造错」。
      **方法学（新增）**：CP 对拍的**分片粒度**须与被锚定拓扑对齐（THD=逐序列、SBHD=沿序列维）。
       **边界**：单卡 tp=1 / packed B=1 / `pad_between_seqs` 与 `cu_seqlens_q_padded` 未涉 /
       `gdn_pre_gated_delta_rule_fusion` 与 GVA 仍未覆盖。

### M12 pre-GDR 融合缺口（KDA vs GDN）
- [x] **M12-1** 关闭 M9-1/M11-2 反复点名、路线图唯一剩余的可选缺口——**`gdn_pre_gated_delta_rule_fusion`
      对 KDA 未实现**（`kda.py:399-402` / 配置期 `transformer_config.py:1936-1938`）。**未改 Megatron-LM
      一行**；新增 `harness/check_port_pre_gated_fusion.py`（`check_port_pre_gated_fusion.out.txt`，空闲卡）+
      `docs/14-pre-gated-fusion.md`。**Part A（复现）**：两道 guard 均实测触发同一句
      `NotImplementedError`（配置期 + 运行期）。**Part B（结构差异，真实行号）**：GDN 融合入口
      `fused_streamed_pre_gated_delta_rule`（`fused_pre_gated_delta_rule.py:2571`）与 KDA pre-GDR 有**四处
      会静默算错的硬差异**——(1) 投影段：GDN `["query","key","value","z","beta","alpha"]`（`gdn.py:52`）
      vs KDA `["query","key","value","g","gate"]`（`kda.py:133`），KDA 的 `g` 段宽 **`qk_dim`＝per-channel**、
      `beta` 走**独立 `beta_proj`**（`kda.py:100-110`）；(2) 门控激活位置：GDN 在 pre-GDR 核内算
      `g=-exp(A_log)*softplus(alpha+dt_bias)`（`fused_pre_gated_delta_rule.py:500-508`），KDA **原样返回
      `raw_g`** 并置 `use_gate_in_kernel=True`（`kda.py:121`）把激活留给 `chunk_kda`；(3) 门控参数形状：
      KDA `dt_bias_dim=qk_dim_local_tp`（per-channel）、`a_log_dim=num_k_heads`（`kda.py:141-142`），GDN 两者
      都 per-value-head（`gdn.py:61-62`）；(4) 融合核输入契约假定单投影 + `causal_conv1d`。**Part C（契约与
      锚定）**：给出 KDA 专属 fused pre-GDR 的三 scope 契约（conv / qk-l2norm+split / raw_g+beta，且
      `gate` 以 `qkvfg` strided view 返回），用纯 torch 参考 `kda_pre_gdr_fused_ref` 对真实层 eager
      `pre_gated_delta_rule` 逐张量锚定——同批 FLA 原语时 **`v/gate/g` 逐位(=0)**、`k` 在 bf16 亦逐位、
      `q/beta` ≤1 ulp（fp32 `q 8.94e-08`、bf16 `q 1.95e-03`＝亚 ulp），参考 **run-to-run 自洽=0**；换
      torch conv/l2norm 后残差归因到原语（`v` bf16 7.81e-03、`q/k` bf16 ~3.9e-03）。**Part D（drop-in）**：
      逐位同权重、只替换 `pre_gated_delta_rule`，整层前反向——fp32 `out 6.48e-05 / hs.grad 2.65e-04 /
      param.grad rel 1.82e-04`、bf16 `out 9.77e-04 / hs.grad 3.91e-03 / param.grad rel 3.31e-03`（与
      M10-1/M11-2 整模块 bf16 锚定同阶，证明融合预处理对核心与整层梯度数值透明）。**Part E（布局账）**：
      六操作数全为新分配（fp32 322 KiB / bf16 161 KiB），其中只有 `gate` 的 eager `.contiguous()` 会实打实
      物化（bf16 32 KiB）——融合省的是 launch + 中间激活读写，不是「最终操作数拷贝」。**边界**：tp=cp=1、
       SBHD、GVA 因子=1、未写 Triton 融合核、未计时；容器未装外部 `causal-conv1d`，故 GDN 融合入口本机也
       不可运行。**下一步（可选）**：按 §5 契约实现 KDA 专属 Triton 融合核并接到 `kda.py:399` 分支，或维持维护态。

### M13 fused pre-GDR 实现
- [x] **M13-1** 兑现 `docs/14` §7 的「下一步」——按 `docs/14` §5 契约把 KDA 专属 fused pre-GDR
       从纯 torch 参考（M12-1）兑现成一个**可运行的 Triton 核族**。新增 `src/kda_pregdr_fused.py`
       （三个 kernel：`_kda_conv_silu_kernel`/`_kda_l2norm_kernel`/`_kda_beta_sigmoid_kernel`
       + host 编排 `_forward_raw` + autograd 包装）+ `harness/check_port_pregdr_kernel.py`
       （`check_port_pregdr_kernel.out.txt`，空闲卡）+ `docs/15-fused-pregdr-kernel.md`。**未改
       Megatron-LM 一行**。**Part A**（vs 真层 eager `pre_gated_delta_rule`，5 组 shape 含非整除
       `T=250`）：fp32 `v/gate/beta/g` **逐位 0**、`q/k` ≤1.2e-7；bf16 worst **≤1.95e-3（=0.5 ulp）**、
       半数 shape 前向**逐位 0**。**Part B**（vs 纯 torch 原语参考）：fp32 1.8e-7、bf16 3.91e-3（=conv
       后端舍入，与 M12-1 归因一致）。**Part C**（整层 drop-in，同权重只换 `pre_gated_delta_rule`）：
       fp32 `out 1.23e-4 / hs.grad 3.04e-4 / param.grad rel 2.08e-4`、bf16 `out 逐位 0 /
       hs.grad 7.81e-3 / param.grad rel 3.55e-3`（同 M12-1 阶）。**Part D**（孤立计时）：
       `wall` eager 恒 **~450 µs（host launch 税、与 T 无关）** → Triton **~190 µs（2.4–2.8×）**；
       纯设备 `dev` 小 shape 略慢（0.61–0.81×）、大 shape 反超（`B8T2048` 1.16–1.20×）。**踩坑**：
       初版 l2norm「一行一 program」在所有 shape `dev` 0.60–0.74×，块化 `[BT,BD]` 后大 shape 抬到
       1.16–1.20×。**边界**：前向 only（Part C 梯度走参考 backward）、单卡 tp=cp=1/SBHD/GVA=1、
       `conv_bias=False`/packed/CP 未覆盖、未接 `kda.py:399` guard。**下一步（可选）**：原生 Triton
       反向（`causal_conv1d_bwd`+`l2norm_bwd`+sigmoid bwd）并接到 `kda.py:399`，或 retire 维护态。

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
- **CP 对拍的损失必须 count-invariant**（M11-1，`docs/12` §2）：真层 CP 下每个 rank 只持有
  `T/cp` 个 token，`out.float().square().mean()` 会按**本地 token 数**缩放损失，使 cp=2 的输入
  梯度整条差 **2×**（实测 `|grad_cp|max=3.36e-05` vs `|grad_ref|max=1.69e-05`）。对拍一律用
  `out.float().sum()`；或对参考乘 `cp`。指纹＝`o` 完全对（permutation-invariant 的平方和损失）、
  但 `hs.grad` 相对误差恰为 ~1.0 且 `|grad_cp|/|grad_ref|≈cp`。
- **CP 下复制型参数的梯度是部分和**（M11-1，`docs/12` §2）：每个 CP rank 只算自己 token 分片的
  参数梯度，全序列梯度＝跨 CP 组 `all_reduce(SUM)`（数据并行同款记账）。直接拿 rank0 的
  `p.grad` 对 cp=1 参考比，`in_proj.weight` 相对误差停在 **0.55–0.64**；`all_reduce` 后回底噪。
  指纹＝**误差恰约 1/cp 量级、且所有复制型参数一致偏小**。
- **packed THD 多序列 CP 要 prebuilt THD CP routes**（M11-1，`docs/12` §4）：`PackedSeqParams`
  只给 `cu_seqlens`（`qkv_format="thd"`）+ 逐 token zigzag 分片时，多序列 CP 对不上全序列 golden
  （`o rel` O(1)）并打印 `context_parallel_layout/conversion.py:120` 的 fallback `FutureWarning`
  （"should not be used on the steady-state forward path"）。这是**输入构造不符真层 CP 契约**，不是
  KDA 算错；多序列 CP 用 **SBHD（`[T,B,H]`、沿序列维分片）**（Megatron 自己单测的形态）即可。

## 当前进度 / 下一步

- 当前进度：M0–M8 完成 + M9-1 完成 + M10-1/M10-2/M10-3 完成 + M11-1 完成 + M11-2 完成 +
  M12-1 完成 + **M13-1 完成**
  （docs 00–15 + 参考实现 + FLA 数值锚定 [默认 + 门控/布局边界 + chunk/pad/varlen/CP 前向 +
  CP 反向/varlen×CP 梯度] + Megatron 调用契约对齐 + **真实层端到端前反向对拍 [SBHD +
  packed/THD + 带 padding 的 packed/THD + cp>1 headwise/chunkwise + **packed THD 多序列 CP**]** +
  `gdn_pre_gated_delta_rule_fusion` 的 KDA 缺口分析 + 专属 fused pre-GDR 契约锚定/drop-in +
  **KDA 专属 fused pre-GDR 的 Triton 核族实现 + 数值/整层 drop-in/wall-dev 计时**）。
- 本轮（M13-1）：`src/kda_pregdr_fused.py` + `harness/check_port_pregdr_kernel.py` +
  `docs/15-fused-pregdr-kernel.md`。把 `docs/14` §5 的契约兑现成可运行 Triton 核族
  （conv+SiLU / L2-norm / beta sigmoid 三 kernel），**未改 Megatron-LM 一行**。**Part A** vs eager：
  fp32 `v/gate/beta/g` 逐位 0、`q/k` ≤1.2e-7；bf16 worst ≤1.95e-3（0.5 ulp）、半数逐位 0。
  **Part B** vs 纯 torch 参考：fp32 1.8e-7 / bf16 3.9e-3（conv 后端舍入）。**Part C** 整层 drop-in：
  fp32 param.grad rel 2.1e-4、bf16 out 逐位 0 / rel 3.5e-3。**Part D**：wall 从 eager ~450µs
  （host 税）降到 ~190µs（2.4–2.8×）、大 shape 纯设备 1.16–1.20×。**下一步（可选）**：原生 Triton
  反向并接 `kda.py:399` guard，或 retire 维护态（M0–M13 全完成）。
- 本轮（M12-1）：`harness/check_port_pre_gated_fusion.py` + `docs/14-pre-gated-fusion.md`。关闭
  路线图唯一剩余的可选缺口——`gdn_pre_gated_delta_rule_fusion` 对 KDA 未实现（`kda.py:399-402`
  + `transformer_config.py:1936-1938`）。**未改 Megatron-LM 一行**。**Part A** 复现两道 guard；
  **Part B** 逐行号给出 GDN 融合入口（`fused_pre_gated_delta_rule.py:2571`）与 KDA pre-GDR 四处
  会静默算错的硬差异（投影段 `gdn.py:52` vs `kda.py:133`、门控激活位置 `fused...:500-508` vs
  `kda.py:121` 的 `use_gate_in_kernel`、门控参数形状 `kda.py:141-142` per-channel vs `gdn.py:61-62`
  per-head、单投影 + `causal_conv1d` 契约）；**Part C** 用纯 torch 参考 `kda_pre_gdr_fused_ref`
  对真实层 eager `pre_gated_delta_rule` 锚定：同批 FLA 原语时 `v/gate/g` 逐位、`k` bf16 逐位、
  `q/beta` ≤1 ulp（fp32 `q 8.94e-08`、bf16 `q 1.95e-03`），参考自洽=0；**Part D** 逐位同权重
  drop-in 整层前反向——fp32 `out 6.48e-05 / param.grad rel 1.82e-04`、bf16 `out 9.77e-04 /
  param.grad rel 3.31e-03`（与 M10/M11 整模块 bf16 锚定同阶）；**Part E** 布局账（只有 `gate`
  的 `.contiguous()` 实打实物化）。**下一步（可选）**：按 `docs/14` §5 契约写 KDA 专属 Triton
  融合核并接到 `kda.py:399` 分支，或 retire 到维护态（M0–M12 全完成）。
- 本轮（M11-2）：`harness/check_port_e2e_cp_thd.py` + `docs/13-thd-cp.md`。关闭 M11-1 留的
  **packed THD 多序列 CP** 边界，**并更正根因**：O(1) 不是「缺 prebuilt route」而是**输入分片粒度**
  ——真层 THD CP zigzag 是**逐序列**切块（`routes.py:79-90`），M11-1 用的整轴 zigzag
  （`rope_utils.py:181`）对多序列不同。修法＝逐序列 zigzag 分片 + `prebuild_thd_cp_partition_routes`
  消掉 `FutureWarning`。**实测** `world=2/4`×`headwise/chunkwise`×fp32/bf16×3 组 pack（含非均匀
  序列长）：bf16 `~5e-3–8e-3`、fp32 `~1e-3`（与 M11-1 同量级），`FutureWarning` 归零；
  `--shard global` 受控对拍复现 M11-1 的 O(1)。**下一步（可选）**：retire 到维护态（M0–M11 全完成）；
  若继续，补 `gdn_pre_gated_delta_rule_fusion` 的 KDA 专属 streamed 融合前处理（`kda.py:399-402`）。
- 本轮（M11-1）：`harness/check_port_e2e_cp.py` + `docs/12-cp-e2e.md`。把 M10 的真层对拍扩到
  **cp>1**：两段式（cp=1 fp64 参考芯 dump 全序列黄金；`mp.spawn` 起 `world` 个 NCCL rank 重建真层、
  喂全输入的 zigzag 分片、对黄金的同分片比）。**实测**（`T=256`、`world=2/4`、fp32/bf16、
  headwise/chunkwise、packed 单序列/SBHD B=2）：`o`/`hs.grad`/`param.grad` 相对误差 bf16
  `4e-3–1e-2`、fp32 `0.9e-3–1.6e-3`，`world=2/4` 同阶。**两条方法学**：CP 对拍损失须 count-invariant
  （`sum` 而非 `mean`，否则 cp=2 输入梯度差 2×）；复制型参数梯度在 CP 下是部分和、对参考比前须
  `all_reduce(SUM)`（否则相对误差停在 0.55–0.64）。**遗留边界**：packed THD **多序列**（plain
  `cu_seqlens`、无 prebuilt routes）对不上、触发真层 fallback `FutureWarning`（输入构造不符 CP
  契约，非算法错；多序列 CP 用 SBHD）。
- 本轮（M10-3）：`harness/check_port_e2e_padded.py` + `docs/11-padded-thd.md`。关闭 M10-2 的
  「packed 无 padding」边界：给 THD 批次带 `cu_seqlens_padded`（真实 `[0,50,110]/[0,50,100,220]`
  → padded `[0,64,128]/[0,64,128,256]`），pad 落在每个 padded slot 尾部、被层当作段内 token
  交给核心与 conv1d。**Part A** 全有限。**Part B** 核心锚定 fp32 `o rel=2.56e-3`/bf16 `3.88e-3`
  （梯度 rel ≤7.2e-3）。**Part C** 归因：只有门控 `g` 的 pad 会动真实 token（~1.3e-5），
  `q/k/v/beta` 逐位 0。**Part D（关键）** 跨序列隔离精确 `max_abs=0`，自身 slot 尾影响
  `9.1e-5`(fp32)/`1.95e-3`(bf16)。**Part E** 整模块 padded：fp32 `out=5.07e-4`、bf16 `2.93e-3`。
  **Part F** padded↔unpadded 真实 token `≤1.08e-4`(fp32)/`1.95e-3`(bf16)。**结论**：带 padding 的
  packed 路径数值健全、跨序列隔离精确，pad 对真实 token 的影响仅剩自身 slot 尾的块尾舍入、低于
  部署底噪。**边界**：单卡 tp=cp=1、`pad_between_seqs` 布局、cp>1 的 packed CP（zigzag）未覆盖。
- 下一步（可选）：retire 到维护态（M0–M13 全完成）。原列的「打通 packed THD 多序列 CP」**已由
  M11-2 完成**；「`gdn_pre_gated_delta_rule_fusion` 的 KDA 缺口」**已由 M12-1 关闭**（结构差异
  + 专属 fused pre-GDR 契约锚定/drop-in，见 `docs/14`）；「按 `docs/14` §5 契约把 KDA 专属 fused
  pre-GDR 实现成 Triton 核族」**已由 M13-1 完成**（三 kernel 核族 + 数值锚定 + 整层 drop-in +
  wall/dev 计时，见 `docs/15`）。若继续，仅剩「原生 Triton 反向 + 接到 `kda.py:399` guard」
  （属实现工程，非分析缺口）。

## 阻塞

（无）
