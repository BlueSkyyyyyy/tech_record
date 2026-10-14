# 13 · 打通 packed THD *多序列* CP（M11-2）

> 系列第 14 篇。M11-1（`docs/12`）把真实 `KimiDeltaAttention` 的 CP 数值锚定做到 cp>1，
> 覆盖 `headwise`/`chunkwise`、`world=2/4`、**packed 单序列**与 **SBHD 多序列**，并把
> **packed THD *多序列***明确留成边界：`cu=[0,64,128]` 下误差抬到 O(1)，真层打印
> `conversion.py` 的 fallback `FutureWarning`，当时的结论是「需要 prebuilt THD CP route 或
> 改用 padded 布局」。
>
> 本篇把这个边界打通，**并更正根因**：真正的错在**输入分片方式**（按整条 token 轴做 zigzag
> 而不是**逐序列** zigzag），与 route 是否预建无关。
>
> 脚本：`harness/check_port_e2e_cp_thd.py`；原始输出：`harness/check_port_e2e_cp_thd.out.txt`。
> 行号按本机版核对：`Megatron-LM/megatron/core/context_parallel_layout/routes.py`、
> `.../conversion.py`、`.../utils.py`、`megatron/core/packed_seq_params.py`、
> `megatron/core/ssm/gated_delta_net/kda.py`、`megatron/core/models/common/embeddings/rope_utils.py`；
> 容器内 `fla==0.5.2`、`torch 2.12`、H100。

---

## 1. 先说结论

| 分片方式 | cu=[0,64,128], cp=2, bf16, headwise | chunkwise |
|---|---|---|
| 整轴 zigzag（M11-1 的配方） | `o rel 1.10` / `hs.grad 0.93` | `o rel 0.92` / `hs.grad 1.04` |
| **逐序列 zigzag（本篇）** | `o rel 5.36e-3` / `hs.grad 5.65e-3` | `o rel 5.84e-3` / `hs.grad 7.81e-3` |

同一份脚本、同一组权重、只换分片函数：整轴 zigzag 复现了 M11-1 的 O(1)，逐序列 zigzag 落回
bf16 底噪（`docs/12` 的 cp=1/单序列锚定量级）。**packed THD 多序列 CP 本身没有数值问题**，
此前的 O(1) 是测试输入构造错位。

---

## 2. 根因：THD CP 的 zigzag 是**逐序列**的，不是整条 token 轴

Megatron 的 THD CP *zigzag* 布局定义在
`context_parallel_layout/routes.py::_build_thd_layout_segments`（`:63`），其 zigzag 分支
（`:79-90`）是：

```python
for seq_start, seq_end in zip(cu[:-1], cu[1:]):
    seq_len = seq_end - seq_start
    chunk_len = seq_len // (2 * cp_size)
    first_chunk  = cp_rank
    second_chunk = 2 * cp_size - cp_rank - 1
    segments.append((seq_start + first_chunk  * chunk_len, chunk_len, ...))
    segments.append((seq_start + second_chunk * chunk_len, chunk_len, ...))
```

即：**每个 packed 序列各自**切成 `2*cp_size` 个等块，rank `r` 拿第 `r` 块和第
`2*cp_size-1-r` 块；多个序列依次拼接。这和 M11-1 用的
`get_pos_emb_on_this_cp_rank`（`rope_utils.py:181`）不同——后者把**整条 `T_total` 轴**
按 `2*cp_size` 等分再选块。

两者的差别在**单序列时为零**（`cu=[0,T]` 时逐序列就是整轴），在**多序列时为正**。以
`cu=[0,64,128]`、`cp_size=2` 为例（`cp_rank=0`）：

| 布局 | rank0 拿到的全局 token |
|---|---|
| 整轴 zigzag（M11-1） | `{0..31, 96..127}` |
| 逐序列 zigzag（真层） | `{0..15, 48..63, 64..79, 112..127}` |

真层在模块边界收发的正是**逐序列 zigzag**分片（`kda.py:263-270` 的
`"headwise CP requires zigzag layout"`），`chunkwise` 则先把它转成 contiguous
（`kda.py:234-241`）。把整轴 zigzag 的分片喂进去，行顺序根本不是 route 期望的顺序，误差自然
是 O(1)。

**route 本身从来不是问题**：`_redistribute_thd_layout`（`conversion.py:326`）在
`thd_cp_partition_route is None` 时会自己用 `cu_seqlens` 重建一份等价 route
（`conversion.py:352-356`），结果与预建 route 一致。M11-1 看到的 `FutureWarning`
（emit 在 `routes.py:242-252`，调用点 `conversion.py:120-122`）只是「未预建」的兼容提示，
**不改变数值**。所以 M11-1 §4 的「需要 prebuilt routes / padded 布局」措辞要更正为：
**需要的是逐序列（per-sequence）zigzag 的输入分片**；预建 route 只是消掉那条警告。

---

## 3. 修法

`harness/check_port_e2e_cp_thd.py` 相对 M11-1 只改两处（其余两段式框架一字未动）：

1. **逐序列 zigzag 分片** `per_seq_zigzag_local_indices(cu_list, cp_size, cp_rank)`，逐字复刻
   `_build_thd_layout_segments` 的 zigzag 分支，再 `index_select` 到输入 / 参考输出 / 参考梯度上；
2. **预建 route**：构造 `PackedSeqParams` 时带上 `cp_group`，并调用公开 API
   `prebuild_thd_cp_partition_routes(psp, cp_group)`（`routes.py:257`），落实「在构造 batch 时
   预建 THD CP route」的稳态契约（`utils.py::finalize_packed_seq_params:30` 是同一条路径）。

脚本另带一个 `--shard {perseq,global}` 开关：`global` 明确用 M11-1 的整轴配方，用于**受控对拍**
（同一脚本、同一权重、只换这一个变量）——这就是第 1 节里 O(1) 那一行的来源。

约束与 M8-1 一致（`_validate_thd_route_partitioning`，`routes.py:41`）：`total_tokens % cp_size == 0`
且**每个序列长度 `% (2*cp_size) == 0`**；`chunkwise` 还要求 zigzag→contiguous 后每个 rank 的
连续分片 `total/cp` 是 `chunk_size=64` 的整数倍。

---

## 4. 结果：`world=2/4` × `headwise/chunkwise` × fp32/bf16 × 3 组 pack

`cu` 三组：`[0,64,128]`（两序列各 64，world=2）、`[0,64,160]`（两序列 64/96，world=2，
长度不等）、`[0,64,128,192,256]`（四序列各 64，world=4）。下表为相对最大误差
（`max_abs / max|ref|`），与 `docs/12` 同口径：

| dtype | arm | world | cu | o rel | hs.grad rel | param.grad rel |
|---|---|---|---|---|---|---|
| bf16 | headwise | 2 | 0,64,128 | 5.36e-3 | 5.65e-3 | 7.58e-3 |
| bf16 | headwise | 2 | 0,64,160 | 5.84e-3 | 7.10e-3 | 8.03e-3 |
| bf16 | headwise | 4 | 0,64,128,192,256 | 5.84e-3 | 7.53e-3 | 6.43e-3 |
| bf16 | chunkwise | 2 | 0,64,128 | 5.84e-3 | 7.81e-3 | 7.69e-3 |
| bf16 | chunkwise | 2 | 0,64,160 | 5.84e-3 | 5.68e-3 | 6.02e-3 |
| bf16 | chunkwise | 4 | 0,64,128,192,256 | 5.84e-3 | 7.53e-3 | 6.13e-3 |
| fp32 | headwise | 2 | 0,64,128 | 1.26e-3 | 1.24e-3 | 1.15e-3 |
| fp32 | headwise | 2 | 0,64,160 | 1.03e-3 | 1.15e-3 | 1.25e-3 |
| fp32 | headwise | 4 | 0,64,128,192,256 | 1.02e-3 | 1.21e-3 | 9.82e-4 |
| fp32 | chunkwise | 2 | 0,64,128 | 1.26e-3 | 1.24e-3 | 1.15e-3 |
| fp32 | chunkwise | 2 | 0,64,160 | 9.93e-4 | 1.14e-3 | 1.31e-3 |
| fp32 | chunkwise | 4 | 0,64,128,192,256 | 1.02e-3 | 1.21e-3 | 9.82e-4 |

bf16 全表 `o`/`hs.grad`/`param.grad` 落在 **~5e-3–8e-3**，fp32 落在 **~1e-3**——与 M11-1 的
packed 单序列 / SBHD 多序列、以及 cp=1 的 M10-1/M10-2 整模块锚定**完全同量级**。

几点值得单列：

- **非均匀序列长度**（`[0,64,160]`）与均匀序列同量级 ⟹ 逐序列 cut 的记账对
  「每序列长度不同」也成立（这正是整轴 zigzag 会出错的最直接场景）。
- **`world=4` 与 `world=2` 同阶**，跨 rank 状态传递（headwise 的 head-a2a / chunkwise 的
  `cp_context`）不引入新偏差；四序列 pack 的段间隔离无泄漏（若有串状态误差会是 O(1)）。
- **`headwise` 与 `chunkwise` 同阶**，两条 CP 路各自在 M8-1 算子级证过，在真层多序列上再确认。
- **`FutureWarning` 归零**：预建 route 后输出里
  `futurewarning: THD PackedSeqParams is missing` 出现 **0 次**（M11-1 每次 layout 转换都打）。

---

## 5. 结论与边界

- **packed THD 多序列 CP 打通**：逐序列 zigzag 分片 + 预建 route 后，真实
  `KimiDeltaAttention` 在 `world=2/4`、`headwise/chunkwise`、fp32/bf16、3 组 pack（含非均匀
  序列长）上，前反向对 fp64 参考芯（全序列）锚定到 **bf16 ~5e-3–8e-3 / fp32 ~1e-3** 底噪。
- **根因更正**：O(1) 来自**输入分片方式**（整轴 vs 逐序列 zigzag），不是 route 缺失——
  `conversion.py` 本就会从 `cu_seqlens` 重建等价 route。M11-1 的「需要 prebuilt routes /
  padded 布局」应改为「需要逐序列 zigzag 分片」。
- **方法学（新增）**：凡是「CP 分片 vs 参考对拍」的实验，分片的**粒度**要和被锚定拓扑的粒度
  对齐——Megatron 的 THD CP 是**逐序列** zigzag，SBHD 是沿序列维 zigzag，二者对单序列重合、
  对多序列不同；受控对拍（同脚本只换分片函数）能一眼分开「算法错」与「输入构造错」。
- **边界**：单卡 `tp=1`、packed `B=1`、`pad_between_seqs` 未涉、`cu_seqlens_q_padded` 未用、
  每序列等块（`%2*cp`）的约束未破；`gdn_pre_gated_delta_rule_fusion`（`kda.py:399-402`，KDA
  未实现）与 GVA（Megatron KDA 本就不支持）仍未覆盖。
- **复现**：`harness/check_port_e2e_cp_thd.py`（`mp.spawn` 真 NCCL，需 `NCCL_NVLS_ENABLE=0`；
  RNG 由 `model_parallel_cuda_manual_seed` + `torch.manual_seed` 固定）；
  原始输出 `harness/check_port_e2e_cp_thd.out.txt`。
