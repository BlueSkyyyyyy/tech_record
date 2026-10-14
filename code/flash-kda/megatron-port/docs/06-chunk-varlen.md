# 06 · 把锚定扩到 chunk_size / pad / varlen / CP（M8-1）

> 系列第 7 篇。前一篇 [05](05-anchor-boundaries.md) 把锚定扩到门控/布局四个边界，但
> **`chunk_size`、非整除 `T` 的 pad、变长（`cu_seqlens`）、上下文并行（CP）** 这四条从未测过。
> 本篇把它们逐条锚定到 FLA `chunk_kda`，并顺手更正路线图里的一处前提错误
> （`chunk_size∈{16,32,64}` → 实际只允许 `{32,64}`）。
>
> 行号按 **`kernel_lab` 容器内 pip 包 `fla==0.5.2`**（`/usr/local/lib/python3.12/dist-packages/fla`）
> 实测核对；与 `docs/00–05` 引用的 `~/proj/flash-linear-attention` clone 行号可能不同（上游在动）。

---

## 1. 为什么这四条是「边界」

M6/M7 的锚定矩阵固定了 `chunk_size=64`、`T` 整除、`B` 个等长序列、无 CP。
FLA `chunk_kda` 公开签名（`ops/kda/chunk.py:178`）里还有四个入口改写**调度语义**，
每一个都对应一条「移植时必须与 Megatron 对齐」的契约：

| 入口 | 语义 | 核对到的源码（fla 0.5.2） |
|---|---|---|
| `chunk_size` | **KDA 只允许 32 / 64**；16 直接 `ValueError` | `ops/kda/chunk.py:390-392` |
| 非整除 `T` | 参考实现零填充到 `ceil(T/C)`；FLA **掩码尾块** | 见 §2 |
| `cu_seqlens`（varlen） | `B` 必须为 1、输入展平；`initial_state`/`final_state` 按**每条序列** `[N,HV,K,V]` | `chunk.py:371-381`、`:65-70`（`prepare_chunk_indices`） |
| `cp_context`（CP） | **禁止** `initial_state`/`output_final_state`；`cu_seqlens` 被 context 覆盖；按 token 分片到各 rank | `chunk.py:362-369`；`ops/cp/context.py:82-152` |

三点值得先说清：

- **`chunk_size=16` 被拒不是笔误**。路线图 M8-1 原写「`chunk_size∈{16,32,64}`」，
  实测 `chunk_kda(..., chunk_size=16)` 抛
  `ValueError: chunk_size must be either 32 or 64 for KDA, got 16.`。这也与 [00](00-math-and-api.md) §4
  早先记的「KDA 的 `chunk_size` 只允许 32/64」一致——本轮的贡献是**把它显式测出来**并修正清单。
- **pad 语义是「参考补零 = FLA 掩码」**。参考实现把每条序列补到 `chunk_size` 的整数倍
  （`../src/kda_ref.py:199-206`），补零 token 的 `g=0`（cumsum 不变）、`q/k/v=0`、`beta=0`
  → 不产生写入，`o` 的实 token 与终态都不受影响。FLA 不补零，而是在尾块用掩码。
  两者必须在 fp64 意义上等价（§2 Part 0 验证）。
- **CP 是「契约 + 分片约束」两件事**。契约层（`chunk.py:362-369`）禁止初/终态、覆盖 `cu_seqlens`；
  分片层（`ops/cp/context.py:82-152`）按 `part_len = total // world_size` 把 token 切成
  `[part_len*r, part_len*(r+1))`，并算出 `is_first_rank/is_last_rank/pre_num_ranks/post_num_ranks`
  驱动跨 rank 的状态收发（`ops/cp/chunk_delta_h.py:747-828`）。**这套分片只有在
  `total % world == 0` 且 `part_len % chunk_size == 0` 时才与不分片的前向数值一致**（§5 实测）。

---

## 2. Part 0：参考内部自洽（fp64 CPU）

不依赖 FLA，先证明「补零」与「逐序列循环」两种调度在参考里是同构的。

**(a) chunked（零填充）== 逐 token 递推**（`rel|do|`、`rel|dS|`，非整除 `T`）：

| `chunk` | `T=33`(pad 31) | `T=130`(pad 30) | `T=250`(pad 6) |
|---|---|---|---|
| 32 | 1.7e-15 / 5.7e-15 | 1.1e-14 / 8.0e-15 | 6.3e-14 / 4.3e-14 |
| 64 | 2.6e-15 / 2.5e-15 | 5.0e-15 / 5.1e-15 | 1.9e-14 / 1.9e-14 |

**(b) varlen（逐序列循环）== 逐 token 递推**：`lens=[250,130,64]`、`chunk=64` →
`rel|do|=3.4e-14`、`rel|dS|=3.5e-14`。

即：**块尾补零、序列切分都不改变代数**，两条调度只是重新组织计算顺序。

---

## 3. Part A：`chunk_size × T` 的 pad 语义 sweep

喂两侧**同一份 bf16 操作数**（`q/k` 已 l2norm、`g` 由 `kda_gate` 激活），
FLA 关核内门控并传匹配的 `chunk_size`；参考跑 fp64、匹配同一 `chunk_size`。
损失 $L=\langle o,\bar o\rangle+\langle S,\bar S\rangle$，下方为 `max_rel`：

| `C` | `T` | pad | `o` | `S` | `dq` | `dk` | `dv` | `dg` | `dbeta` | `dh0` |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 16 | 16 | 5.3e-3 | 2.1e-3 | 2.7e-3 | 2.5e-3 | 4.7e-3 | 4.4e-3 | 3.2e-3 | 3.2e-3 |
| 32 | 33 | 31 | 4.3e-3 | 2.0e-3 | 3.4e-3 | 2.8e-3 | 2.5e-3 | 3.4e-3 | 5.2e-3 | 1.3e-3 |
| 32 | 64 | 0 | 4.5e-3 | 2.0e-3 | 2.8e-3 | 3.1e-3 | 3.9e-3 | 2.5e-3 | 4.7e-3 | 1.6e-3 |
| 32 | 130 | 30 | 3.2e-3 | 4.9e-3 | 3.0e-3 | 3.7e-3 | 4.2e-3 | 1.8e-3 | 9.5e-4 | 2.1e-3 |
| 32 | 250 | 6 | 7.9e-3 | 3.2e-3 | 3.2e-3 | 3.4e-3 | 3.6e-3 | 3.5e-3 | 1.4e-3 | 1.1e-3 |
| 32 | 256 | 0 | 5.4e-3 | 2.0e-3 | 2.9e-3 | 3.4e-3 | 4.2e-3 | 2.3e-3 | 1.7e-3 | 1.3e-3 |
| 64 | 16 | 48 | 5.3e-3 | 2.1e-3 | 2.7e-3 | 2.5e-3 | 4.7e-3 | 4.4e-3 | 3.2e-3 | 3.2e-3 |
| 64 | 33 | 31 | 4.3e-3 | 2.0e-3 | 3.4e-3 | 2.8e-3 | 2.5e-3 | 1.7e-3 | 5.2e-3 | 1.3e-3 |
| 64 | 64 | 0 | 4.5e-3 | 2.0e-3 | 2.8e-3 | 3.1e-3 | 3.9e-3 | 2.5e-3 | 4.7e-3 | 1.6e-3 |
| 64 | 130 | 62 | 3.2e-3 | 4.9e-3 | 3.0e-3 | 3.7e-3 | 4.2e-3 | 1.8e-3 | 9.5e-4 | 2.1e-3 |
| 64 | 250 | 6 | 7.9e-3 | 3.2e-3 | 3.2e-3 | 3.4e-3 | 3.6e-3 | 3.5e-3 | 1.4e-3 | 1.1e-3 |
| 64 | 256 | 0 | 5.4e-3 | 2.0e-3 | 2.9e-3 | 3.4e-3 | 4.2e-3 | 2.3e-3 | 1.7e-3 | 1.3e-3 |

**读数**：
- 全部落在 **bf16 底噪 ≤7.9e-3**，覆盖 `T<chunk`（16）、`pad≈T`（33）、非整除（130/250）、
  整除（64/256）——**FLA 的尾块掩码与参考的零填充在 fp64 意义上等价**。
- `C=32` 与 `C=64` 的行**逐值一致到打印精度**：bf16 操作数量化主导残差，块大小带来的
  代数差在舍入以下。（`dg` 在小 `T` 偶有 1 ulp 抖动，其它项完全相同。）

---

## 4. Part B：varlen（`cu_seqlens`）

把多条序列**展平成 `B=1`**，`cu_seqlens` 给出各段边界；`initial_state`/`final_state`
是**每条序列一份**的 `[N,HV,K,V]`。参考按序列循环（§2 已证明其与逐 token 递推一致）。

| case | `o` | `S` | `dq` | `dk` | `dv` | `dg` | `dbeta` | `dh0` |
|---|---|---|---|---|---|---|---|---|
| `lens=[250,130,64]` C=64 explicit | 3.9e-3 | 3.1e-3 | 3.5e-3 | 2.8e-3 | 5.5e-3 | 3.2e-3 | 2.0e-3 | 1.3e-3 |
| `lens=[128,64,96,64]` C=32 explicit | 3.8e-3 | 1.1e-3 | 3.1e-3 | 3.1e-3 | 4.0e-3 | 2.2e-3 | 4.2e-3 | 2.4e-3 |
| `lens=[64,250,250]` C=64 **GVA x2** | 3.0e-3 | 3.6e-3 | 1.8e-3 | 3.6e-3 | 5.0e-3 | 3.9e-3 | 1.9e-3 | 2.9e-3 |
| `lens=[250,130,64]` C=64 **fused gate** | 5.1e-3 | 2.8e-3 | 3.2e-3 | 2.6e-3 | 4.9e-3 | 2.7e-3† | 5.3e-3† | 4.2e-3 |

† fused 行的 `dg`/`dbeta` 是 `dg_logit`/`dbeta_logit`（核内门控/`sigmoid` 的入口梯度），
另有两项逐 head 标量梯度 `dA_log`（abs 3.0e-4）、`ddt_bias`（abs 4.4e-3）。
四项主路 + `dh0` 全落 **bf16 底噪 ≤5.5e-3**，**逐序列的 `S` 与 `dh0` 都正确**
（若 `cu_seqlens` 的段映射或 per-seq 终态 gather 写错，`S`/`dh0` 会错到 O(1)）。

一个**真实踩到的坑**（harness 侧，不是 kernel）：参考的 `_ref_varlen` 首版把每段
终态 `si`（形状 `[1,HV,K,V]`）用 `torch.stack` 叠成 `[N,1,HV,K,V]`，与 FLA 的 `[N,HV,K,V]`
相比时在损失里**广播**成 `[N,N,HV,K,V]`，于是 `dk/dv/dbeta` 出现 ~2× 的假差
（`S` 更是 O(1)）。修正为 `torch.stack([si[0] ...])` 后全部归零。
**教训**：varlen 对拍前先打印两侧 `final_state.shape`，别让隐式广播把形状错伪装成数值错。

---

## 5. Part C：CP（真实 NCCL 多 rank，单机 8 卡）

在单机上 `torch.multiprocessing.spawn` 起 `world` 个 rank、`init_process_group("nccl")`，
每 rank 取自己的 token 段跑 `chunk_kda(..., cp_context=ctx)`，再拼回全序列与
**不分片**的 `chunk_kda` 结果比。**单机跑 NCCL 需 `NCCL_NVLS_ENABLE=0`**
（否则 `all_gather` 撞 NVLink SHARP：`Failed to bind NVLS Multicast memory`）。

| `lens` | `world` | `C` | `part` | aligned | `got_T`/`ref_T` | `max_abs` | `max_rel` |
|---|---|---|---|---|---|---|---|
| `[256]` | 1 | 64 | 256 | ✓ | 256/256 | 0.000e+00 | 0.000e+00 |
| `[256]` | 2 | 64 | 128 | ✓ | 256/256 | 0.000e+00 | 0.000e+00 |
| `[256]` | 4 | 64 | 64 | ✓ | 256/256 | 0.000e+00 | 0.000e+00 |
| `[128,128]` | 2 | 64 | 128 | ✓ | 256/256 | 0.000e+00 | 0.000e+00 |
| `[250]` | 2 | 64 | 125 | ✗ | 250/250 | 1.221e-04 | 2.165e-03 |
| `[130]` | 4 | 32 | 32 | ✗ | **128**/130 | 0.000e+00 | 0.000e+00 |

**读数与三条 CP 契约**：

1. **分片对齐时 CP 与不分片逐位相同（`max_abs=0`）**——`world=1/2/4`、单序列跨 rank、
   两序列各占一 rank，都不差一个 bit。跨 rank 的状态传递（`pre_num_ranks/post_num_ranks`
   驱动的 `merge_fwd_bwd_kernel`，`ops/cp/chunk_delta_h.py:807-828`）与不分片路径数值等价。
2. **`part % chunk_size != 0` 会错**：`lens=[250] world=2`，`part=125` 把 64-块切在边界内
   （全局块界是 `0,64,128,192,250`，rank 界 `125` 落在 `[64,128)` 中间），CP 与不分片差 **2.2e-3**
   （≫ bf16 底噪）。→ **CP 要求 `part_len` 是 `chunk_size` 的整数倍**。
3. **`total % world != 0` 会丢尾**：`lens=[130] world=4`，`part=32`，`part*4=128 < 130`，
   末尾 2 个 token 无人认领（`got_T=128`），前 128 逐位相同。→ **CP 要求 `total % world == 0`**。
   （两者合起来即 `total % (world·chunk_size) == 0` 是「不打乱块」的充分条件。）

CP 的**反向**本轮未锚定（`chunk_kda_bwd` 的 CP 分支有独立的
`chunk_gated_delta_rule_bwd_dhu_pre_process`，`ops/cp/chunk_delta_h.py:831-`）；列为 M8-2。

---

## 6. 结论、移植要点与后续

**结论**：M8-1 把参考实现从「固定 chunk=64、等长、无 CP」扩到
`chunk_size∈{32,64}` × 非整除/`T<chunk` 的 pad 语义、`cu_seqlens` 变长（含 GVA / 核内融合门控）、
以及真实多 rank CP，**全部锚定到 FLA bf16 底噪**（主路 ≤7.9e-3，CP 对齐时逐位 0）。
并更正路线图前提：**KDA 的 `chunk_size` 只有 32/64，没有 16**。

**移植要点（补进 [02](02-backward.md) §7 / [00](00-math-and-api.md) §5.2）**：

1. **pad 不用手做**：Megatron 的 packed 序列长度任意，FLA 内核用尾块掩码处理，参考的零填充
   与之等价；移植时**不要**自己 pad 到整块（会改变 `cu_seqlens` 语义）。
2. **varlen 是 Megatron 的默认路径**：`kda.py:275-289` 用 `packed_seq_params` 解析
   `cu_seqlens_q`（`B=1`、`thd` 布局），`kda.py:447` 直接透传；`initial_state`/`output_final_state`
   保持 `None`/`False`（`kda.py:440-441`），与本轮 CP/训练契约一致。
3. **CP 的两条硬约束**：`total % world == 0` 且每段是 `chunk_size` 整数倍。Megatron 的
   `cp_partition_mode="zigzag"`（`kda.py:260-270` 强制）本质上就是对 token 做等分，
   移植/自研 kernel 时须保证分片落在块边界，否则会出现本轮 §5 的 2.2e-3 偏差或丢尾。
4. **CP 下不能有初/终态**：`chunk.py:362-364` 强断言；Megatron 也确实传 `initial_state=None`、
   `output_final_state=False`（`kda.py:440-441`），跨 chunk 的状态由 CP context 的跨 rank 收发补。

**未覆盖（下一步）**：
- **M8-2**：CP **反向**（`dhu` 的 CP 前处理 + 跨 rank `dht` 收发）与 `varlen × CP` 组合的
  梯度锚定（本轮 Part C 只锚了前向）。
- **M9**：Megatron `kda.py` 调用契约逐项对齐（SBHD 转置、`dht` 初值、`A_log/dt_bias` fp32、
  `gdn_pre_gated_delta_rule_fusion` 缺口）。
- **M10**：Megatron 路径端到端前反向对拍。

运行：

```
FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
    ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla_m8.py
# 输出存 harness/check_port_fla_m8.out.txt
# Part 0/§2 为 fp64 CPU；Part A/B 为 GPU；Part C 会 spawn 真实 NCCL rank（单机 8 卡）
```
