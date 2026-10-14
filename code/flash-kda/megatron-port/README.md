# megatron-port — 把 FLA KDA 算子移植进 Megatron-LM `gated_delta_net`

本目录服务于一个具体工程：**把 `fla/ops/kda`（Kimi Delta Attention，KDA）的前反向算子
移植进 `Megatron-LM/megatron/core/ssm/gated_delta_net`**（本机 `~/proj/Megatron-LM`），
参考实现来自 `flash-linear-attention`。它是本仓库 `flash-kda` 主系列的一个子目录：
主系列以 FlashKDA（手写 CUDA）为参照讲「怎么算」，本目录以 FLA 为参照讲「怎么移植」。

## 交付物

| 文件 | 内容 |
|---|---|
| [`docs/00-math-and-api.md`](docs/00-math-and-api.md) | **数学原理 + 算子函数接口**：逐 token 递推、门控、分块形式推导、FLA `chunk_kda` 签名、Megatron 调用面、内部算子清单 |
| [`docs/01-forward.md`](docs/01-forward.md) | **FLA KDA 前向算子整理**：Gate → Intra → 状态递推 → 输出，逐 kernel 数式 + 真实行号 |
| [`docs/02-backward.md`](docs/02-backward.md) | **FLA KDA 反向算子整理**：dAv → 状态伴随 → Q/K/G/B/A 伴随 → 块内 → 门控，附移植检查表 |
| [`docs/03-optimization.md`](docs/03-optimization.md) | **算子优化手段**：分块 / 融合 / 重算 / 数值稳定 / tiling-autotune / 布局 / 变长-CP-graph / 移植取舍 |
| [`src/kda_ref.py`](src/kda_ref.py) | **最小可运行参考实现**（纯 PyTorch、CPU、fp64）：逐 token 递推 + chunked 前向 + 门控，自带自测 |

## 参考坐标

| 资源 | 位置 | 用途 |
|---|---|---|
| FLA KDA | `flash-linear-attention/fla/ops/kda/` | 前反向主参照（`chunk_fwd.py` / `chunk_bwd.py` / `chunk_intra.py` / `wy_fast.py` / `gate.py`） |
| FLA 公共状态核 | `flash-linear-attention/fla/ops/common/chunk_delta_h.py` | 状态递推 `fwd_h` 及其伴随 `dhu` |
| FLA 输出核 | `flash-linear-attention/fla/ops/gla/chunk.py` | `chunk_gla_fwd_o_gk`（KDA 复用） |
| Megatron KDA | `Megatron-LM/megatron/core/ssm/gated_delta_net/kda.py` | `KimiDeltaAttention`（调用面） |
| Megatron GDN | `Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py` | `GatedDeltaNet` + 确定性 torch 参考 |
| Megatron 入口 | `Megatron-LM/megatron/core/ssm/gated_delta_net/common.py` | `chunk_gated_delta_rule` / `HAVE_FLA` 等导出 |

## 运行参考实现

```bash
cd code/flash-kda/megatron-port
python3 src/kda_ref.py            # 见 src/kda_ref.out.txt
```

自测（fp64、CPU）覆盖三部分：

- **Part A**：逐 token 递推 vs chunked 前向（$o$ 与终态），相对误差 $\le1.1\times10^{-14}$。
- **Part B**：autograd 穿过 chunked 前向 vs 穿过逐 token（六项梯度），相对误差 $\le7.2\times10^{-15}$。
- **Part C**：门控闭式梯度 vs autograd，相对误差 $\sim5\times10^{-8}$（gate 内部 fp32，与 FLA 一致）。

## 约定与坑

- 布局一律 `[B, T, H, *]`（FLA 公开 API）；Megatron SBHD 在调 kernel 前转置。
- FLA 内核用 **log2** 域（`exp2` + `RCP_LN2`）；参考实现用自然对数，同构。
- **强衰减下不要直接算 $k/E$**（会溢出）：减参考点或用逐对 $2^{\Delta g}$。
- Megatron 传 **raw g**，门控激活必须在 kernel 内做。
- 文档里的行号均按本机版 `flash-linear-attention` / `Megatron-LM` 核对；升级上游后需重核。

## 与主系列的关系

- `flash-kda/docs/*`：FlashKDA 手写 CUDA 前向 + 反向工程化（性能/`ncu`/TMA）。
- 本目录 `megatron-port/docs/*`：FLA 算子的**算法与接口**层面，面向移植。
- 两套文档共享同一套数学（主系列 `docs/01`，本目录 `docs/00`）。
