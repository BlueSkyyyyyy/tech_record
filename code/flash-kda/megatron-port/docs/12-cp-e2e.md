# 12 · 真实层的 context-parallel（cp>1）数值锚定（M11-1）

> 系列第 13 篇。M8-1/M8-2 锚定的是 **FLA 算子**在 CP 下的语义（`build_cp_context` + 真
> NCCL 多 rank）；M10-1/M10-2/M10-3 锚定的是**真实 `KimiDeltaAttention` 模块**，但都停在
> **cp=1**（`docs/09`/`10`/`11` 的「边界」都写着「单卡 tp=cp=1」）。本篇（M11-1）把两者拼起来：
> 用**真的 `context_parallel_size = world` 进程组**跑真实层，前反向分别对 **fp64 参考芯**在
> **全序列**上的结果做数值锚定，覆盖 `headwise` / `chunkwise` 两种 `linear_cp_mode`、`world=2/4`、
> packed THD 与 SBHD 两种输入布局、fp32/bf16。
>
> 脚本：`harness/check_port_e2e_cp.py`；原始输出：`harness/check_port_e2e_cp.out.txt`。
> 行号按本机版核对：`Megatron-LM/megatron/core/ssm/gated_delta_net/kda.py`、
> `megatron/core/context_parallel_layout/conversion.py`，容器内 `fla==0.5.2`。

---

## 1. 真层里 CP 有两条路，别再混淆

真实 `KimiDeltaAttention.forward`（`kda.py:167-357`）里 CP 分两个轴（`kda.py:216-227`）：

- **headwise CP**（`linear_cp_mode="headwise"`，本模块默认）：模块边界仍是**逐 token 的 zigzag
  分片**（`seq_len_post_headwise = seq_len_local * sp_size * cp_size_headwise`，`kda.py:246`），
  模块内部用 **head 维 all-to-all**（`a2a_cp_to_hp`，`kda.py:380/395`）把「token 分片」换成
  「head 分片 + 全 token」。硬要求 **zigzag 布局**（`kda.py:263-270` 抛
  `"headwise CP requires zigzag layout"`）。
- **chunkwise CP**（`linear_cp_mode="chunkwise"`）：把输入先转 **contiguous**（`kda.py:234-241`
  的 `convert_module_input_tensors_cp_partition_mode(target_partition_mode="contiguous")`），再用
  FLA 的 `build_cp_context`（`kda.py:293-318`，cu_seqlens × cp group）把跨 rank 的**状态传递**
  交给 `chunk_kda` 的 `cp_context`（同 M8-1 的算子级路径）。额外约束：**SBHD 输入且 batch>1**
  时直接报错、要求 `micro_batch_size == 1`（`kda.py:404-408`）。

两条路在**模块边界上收发的都是「每个 CP rank 一份 zigzag token 分片」**，所以对拍时**输入与
输出都用同一套分片**（Megatron 自己的
`test_kda_cp2_packed_with_physical_padding_matches_cp1` 用的就是
`get_pos_emb_on_this_cp_rank(t, 0, cp_group)`，本脚本沿用）。

---

## 2. 为什么是「两段式」对拍（一个进程做不到）

CP 的**定义**就是「整条序列的状态被切到多个 rank，rank 间要交换状态」。而我们的 fp64 参考芯
（`src/kda_ref.py::kda_chunked_fwd`，M5/M10）**没有跨 rank 状态交换**——把它塞进一个 cp>1 的
模块里只会得到错的、只算半条序列的结果。所以本脚本走两段：

1. **Stage 1（cp=1）**：建一个 cp=1 的真层，把 `gated_delta_rule` 换成 fp64 参考芯
   （M10-2 的 `make_ref_gdr`，packed-aware），在**全序列**上跑前反向，dump 出
   `o` / `hs.grad` / 参数梯度 + 冻结权重；
2. **Stage 2（cp=world）**：`mp.spawn` 起 `world` 个 NCCL rank，用**同一份权重**重建真层，
   每个 rank 喂**同一个全输入的 zigzag 分片**，跑前反向，把 rank-local 张量对 **Stage 1 张量
   的同一分片**比。

两个实现细节值得写下来：

- **损失必须用 count-invariant 的 `out.float().sum()`，不能用 `.mean()`**。第一版用
  `.mean()`：cp=2 的每个 rank 只在本地 `T/cp` 个 token 上取均值，损失被本地元素数缩放，
  输入梯度整条差 **2×**（实测 `|grad_cp|max=3.36e-05` vs `|grad_ref|max=1.69e-05`，比值 1.99）。
  换 `sum()` 后梯度与 rank 数无关。
- **复制型参数的梯度必须先在 CP 组内 `all_reduce(SUM)`**。每个 CP rank 只持有 token 分片，
  它算出的参数梯度是**部分和**；全序列梯度 = 跨 CP rank 求和（数据并行的同款记账）。不归约时
  `in_proj.weight` 的相对误差停在 **0.55–0.64**；归约后降到 **底噪**（见下）。

---

## 3. 结果：cp=2/4 × headwise/chunkwise × packed/SBHD × fp32/bf16

`T=256`（packed 单序列 `cu=[0,256]`；SBHD `[256,2,128]`），`world=2/4`（`part=128/64` 均被
`chunk_size=64` 整除，满足 M8-1 的 CP 对齐约束），`bfloat16`/`float32`。下表 `o`/`hs.grad`/
`param.grad` 都是**相对最大误差**（`max_abs / max|ref|`）：

| dtype | layout | arm | world | o rel | hs.grad rel | param.grad rel |
|---|---|---|---|---|---|---|
| bf16 | packed | headwise | 2 | 9.85e-3 | 6.88e-3 | 4.02e-3 |
| bf16 | packed | headwise | 4 | 5.36e-3 | 6.55e-3 | 7.04e-3 |
| bf16 | packed | chunkwise | 2 | 6.64e-3 | 6.55e-3 | 7.04e-3 |
| bf16 | packed | chunkwise | 4 | 7.14e-3 | 5.24e-3 | 7.04e-3 |
| bf16 | sbhd | headwise | 2 | 6.28e-3 | 7.58e-3 | 5.10e-3 |
| bf16 | sbhd | headwise | 4 | 6.28e-3 | 7.58e-3 | 7.65e-3 |
| fp32 | packed | headwise | 2 | 1.14e-3 | 1.23e-3 | 1.62e-3 |
| fp32 | packed | headwise | 4 | 1.39e-3 | 1.23e-3 | 1.62e-3 |
| fp32 | packed | chunkwise | 2 | 1.14e-3 | 1.23e-3 | 1.62e-3 |
| fp32 | packed | chunkwise | 4 | 1.37e-3 | 1.17e-3 | 1.39e-3 |
| fp32 | sbhd | headwise | 2 | 1.46e-3 | 1.01e-3 | 1.25e-3 |
| fp32 | sbhd | headwise | 4 | 1.46e-3 | 9.17e-4 | 1.25e-3 |

对照量级：bf16 全表 `o`/`hs.grad`/`param.grad` 都落在 **4e-3–1e-2**，与 cp=1 的
M10-1/M10-2 bf16 整模块锚定（`out≈3e-3`、梯度 4.3–9.4e-3）同量级；fp32 全表落在
**0.9e-3–1.6e-3**，与「fp32 模块走 tf32 的 FLA 芯 vs fp64 参考」的底噪一致。绝对量级示例
（bf16 packed headwise world=2）：`o max_abs=3.9e-3`、`hs.grad max_abs=1.17e-2`、
`param.grad max_abs=5.6e-1`（`in_proj.weight`，其自身量级很大，故看相对）。

**world=2 与 world=4 的误差同阶**（不随 rank 数增长）——说明跨 rank 的状态传递（headwise 的
head-a2a / chunkwise 的 `cp_context`）都没有引入额外偏差。

几个可以单独说清的点：

- **headwise vs chunkwise 同阶**：两条 CP 路各自由 M8-1 在算子级证过，本轮在真层上确认两者
  都不给整模块加可测误差。
- **SBHD 多序列（B=2）是 headwise**（Megatron 的受支持多序列 CP 布局，也是它自己单测用的
  形态）：`o rel 6.3e-3`、梯度底噪，与 packed 单序列同阶。
- **packed 单序列**同时覆盖了 chunkwise（`cu=[0,T]`，T 被 world 与 64 整除）。

---

## 4. 边界：packed THD **多序列**（plain `cu_seqlens`）对不上，且本就不是稳态路径

把 packed 换成**多序列**（`cu=[0,64,128]`、`world=2`）后，误差突然抬到 **O(1)**：

| arm | o rel | hs.grad rel | param.grad rel |
|---|---|---|---|
| headwise | 1.44 | 0.88 | 0.57 |
| chunkwise | 0.92 | 1.04 | 0.96 |

同时真层在每次 layout 转换时打印：

```
FutureWarning: THD PackedSeqParams is missing precomputed context-parallel layout routes.
  This lookup will attempt to build them from packed_seq_params.cp_group as a compatibility
  fallback. ... it should not be used on the steady-state forward path.
  Callers should prebuild THD CP routes when constructing the batch ...
                       （megatron/core/context_parallel_layout/conversion.py:120）
```

也就是说：**「plain `PackedSeqParams(cu_seqlens=..., qkv_format="thd")` + 逐 token zigzag 分片」
这一组输入构造，并不是 packed THD 多序列 CP 的受支持路径**——真层明确要求调用方预先构建 THD
CP route（否则走 fallback）。Megatron 自己的多序列 CP 单测
（`test_kda_cp2_packed_with_physical_padding_matches_cp1`）用的是 **SBHD + `cu_seqlens_padded`**
形态（每 batch 行一条序列、沿序列维分片），也就是本篇 Part A 的 sbhd 行。

**结论**：packed THD 的 **多序列** CP 需要「prebuilt THD CP routes / 合适的 padded 布局」，
本篇未打通，**明确列为遗留边界**（不是「KDA 算错」，而是**输入构造不符合真层的 CP 契约**）。
这与 M10-3 的发现方向一致：padding/布局语义由真层 `_resolve_cu_seqlens`
（`common.py:482-506`）与 layout route 决定，`pad_between_seqs` 之外的多序列 CP 布局要单独测。

---

## 5. 结论与边界

- **真实 `KimiDeltaAttention` 在 cp=2/4 的 headwise 与 chunkwise CP 下前反向数值健全**，对
  fp64 参考芯（全序列）锚定到 **bf16 4e-3–1e-2 / fp32 0.9e-3–1.6e-3** 底噪，`world=2/4` 同阶。
- **packed 单序列**与 **SBHD 多序列（headwise）** 两种输入布局都被锚定；SBHD 多序列正是
  Megatron 自己多序列 CP 单测用的形态。
- **方法学（新增，值得记）**：（i）CP 对拍损失必须 count-invariant（`sum` 而非 `mean`），
  否则输入梯度按本地 token 数缩放；（ii) 复制型参数的梯度在 CP 下是**部分和**，对全序列参考比
  之前必须先在 CP 组 `all_reduce(SUM)`，否则误差停在 ~0.6 而非底噪。
- **遗留边界**：packed THD **多序列** CP（plain `cu_seqlens`、无 prebuilt routes）对不上、且触发
  真层的 fallback `FutureWarning`——需要按真层契约预建 THD CP route 或改用 padded 布局，本篇
  未打通；`gdn_pre_gated_delta_rule_fusion`（`kda.py:399-402`，KDA 未实现）与 GVA
  （Megatron KDA 本就不支持）仍未覆盖。
- **复现**：`harness/check_port_e2e_cp.py`（RNG 由 `model_parallel_cuda_manual_seed` 与
  `torch.manual_seed` 固定；`mp.spawn` 真 NCCL，需 `NCCL_NVLS_ENABLE=0`）；
  原始输出 `harness/check_port_e2e_cp.out.txt`。
