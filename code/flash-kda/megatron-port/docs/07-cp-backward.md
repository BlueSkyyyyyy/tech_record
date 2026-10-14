# 07 · 把锚定扩到 CP 反向与 varlen×CP（M8-2）

> 系列第 8 篇。前一篇 [06](06-chunk-varlen.md) 把锚定扩到 `chunk_size` / pad / varlen / CP，
> 但 **CP 只锚了前向**（Part C）。基础反向（`docs/02`）也只在无 CP 的等长路径上验过。
> 本篇补上缺口：**上下文并行（CP）的反向**（`dhu` 的 CP 前处理 + 跨 rank `dht` 收发）
> 与 **`varlen × CP` 组合**的梯度锚定。
>
> 行号按 **`kernel_lab` 容器内 pip 包 `fla==0.5.2`**
> （`/usr/local/lib/python3.12/dist-packages/fla`）实测核对；与 clone 版可能不同（上游在动）。

---

## 1. 为什么 CP 反向是独立于前向的一段

前向的 CP 装配在**状态递推的输入侧**：`chunk_gated_delta_rule_fwd_h_pre_process`
（`ops/cp/chunk_delta_h.py:747-828`）把每个 rank 自己块内的**状态贡献** `hm=[he|m]`
（`[HV,K,V+K]`，前 `V` 列是状态贡献 `he`、后 `K` 列是衰减转移 `m`）算出来，
`all_gather` 后由 `merge_fwd_bwd_kernel(..., FORWARD=True)`（`:808-828`、merge 定义 `:332`）
按 `pre_num_ranks` 把**前驱** rank 的贡献折进来，得到本 rank 的 `initial_state`。

反向的 CP 装配在**状态递推的输出侧**，是一条**完全独立**的代码路径
（`ops/kda/chunk_bwd.py:522-539`）：

```
522:    if cp_context is not None:
523:        # initial_state is None in the CP mode
524:        # We only need to compute dht of current rank and pass it to the backward kernel
525:        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(...)
```

`chunk_gated_delta_rule_bwd_dhu_pre_process`（`ops/cp/chunk_delta_h.py:831-920`）做的事与
前向 `..._fwd_h_pre_process` **对偶**，但方向相反：

| 维度 | 前向 `fwd_h_pre_process`（:747-828） | 反向 `bwd_dhu_pre_process`（:831-920） |
|---|---|---|
| 传递量 | 状态 $S$（`hm=[he\|m]`，`[HV,K,V+K]`） | 状态伴随 $\bar S$（`dhm=[he\|m]`，`[HV,K,V+K]`） |
| 合并对象 | **前驱** rank 的贡献（`pre_num_ranks`，:813） | **后驱** rank 的贡献（`post_num_ranks`，:902） |
| 合并方向 | `cur_rank = rank - num_ranks + idx`（从最早的前驱起） | `cur_rank = rank + num_ranks - idx`（从最晚的后驱起） |
| 结果 | 本 rank 的 `initial_state` | 本 rank 的 `dht`（状态伴随入口） |
| 写回 | `initial_state[0]`（`h=initial_state[0]`，:812） | `dht[-1]`（`h=dht[-1]`，:899） |

`merge_fwd_bwd_kernel` 的 CP 分支（`:448-477`）只做一件事：沿 rank 链累乘转移、
把各 rank 的状态贡献叠成边界状态

$$h \leftarrow M\,h + he \quad(\texttt{STATE\_V\_FIRST} 时转置),\qquad
\text{后向按后驱优先顺序迭代 } \texttt{cur\_rank}=rank+num\_ranks-idx.$$

**为什么必须独立**：CP 契约（`ops/kda/chunk.py:362-364`）**禁止**用户传初/终态——
`assert initial_state is None`、`assert output_final_state is False`——所以反向也**不能**通过
`dht` 入口传入跨 rank 的状态伴随；它只能由本 rank 用 `dhm`（自己的 token 对边界状态伴随的
贡献）经 `all_gather`+`merge` **重建**。`pre_process` 首行也断言 `dht is None`
（`ops/cp/chunk_delta_h.py:849`）。`expand_h0/compress_h0`（`:923-936`）是配套的压缩：
多序列（varlen）时只有**第一条序列**的状态可能跨 rank 非零，其余 rank 内部自洽，故只留
`h0[:1]`、反向时再 `expand` 回 `[N,...]`。

---

## 2. 反向 CP 的完整装配（逐段 + 行号）

`chunk_kda_bwd`（`ops/kda/chunk_bwd.py:434-615`）在 CP 下的顺序：

1. **重算前向中间量**：`recompute_w_u_fwd`（`:473-482`）拿 `w,u,qg,kg`；
   若 CP，先 `initial_state = expand_h0(initial_state, context)`（`:488`）。
2. `chunk_gated_delta_rule_fwd_h`（`:492-500`）在 CP 下用**已重建的** `initial_state`
   重算块级 `h`（供 `wy_dqkg` 用）。
3. **输出支路** `chunk_kda_bwd_dAv`（`:510-521`）→ `dAqk, dv`。
4. **CP 前处理** `chunk_gated_delta_rule_bwd_dhu_pre_process`（`:525-536`）→ `dht, initial_state=None`。
   内部：`dhm = q.new_zeros(HV,K,V+K)`（`cp/chunk_delta_h.py:864`）、
   `dht = q.new_zeros(N,HV,K,V)`（:866-869）、
   若非首 rank 跑 `pre_process_bwd_kernel_merged`（:870-893）、
   `ag_dhm = all_gather_into_tensor(dhm, group)`（:895，`comm.py:19`）、
   若非末 rank 跑 `merge_fwd_bwd_kernel(FORWARD=False)` 写 `dht[-1]`（:897-903）、
   返回 `(dht, None)`（:920）。
5. **普通状态伴随** `chunk_gated_delta_rule_bwd_dhu`（`:541-556`）——与本 rank 内
   无 CP 时同一 kernel，只是 `dht` 由第 4 步注入。
6. `chunk_kda_bwd_wy_dqkg_fused`（`:557-574`）→ `dq,dk,dv,db,dg,dAkk`，
   再 `chunk_kda_bwd_intra`（`:576-586`）→ 块内回传，最后反向 cumsum + `kda_gate_bwd`。

**要点**：CP 只改变**第 4 步**（边界状态伴随的来源）。第 3/5/6 步逐字复用普通路径——
这就是为什么后文的「CP == 无 CP」在**分片对齐**时能做到逐位相同（§4）。

---

## 3. Part 0：参考内部自洽（fp64 CPU）

CP 的整个论证建立在「token 只是被重新分组、逐 token 的伴随不变」之上。先在参考实现里
证明：**分块反向（零填充）== 逐 token 反向**，`max_rel`（`src/kda_ref.py`，
`kda_token_recurrent` vs `kda_chunked_fwd` 的 autograd 梯度，fp64 CPU）：

| `chunk` | `T=33`(pad 31) | `T=130`(pad 30) | `T=250`(pad 6) |
|---|---|---|---|
| 32 | `q/k/v/beta` 0，`g` 5.1e-16 | `g` 2.1e-16 | `g` 2.1e-19 |
| 64 | `g` 2.0e-16 | `g` 2.8e-16 | `g` 5.0e-18 |

即块尾补零对反向**只重排计算顺序**，六项梯度到 fp64 机器精度一致（`g` 的尾巴是
浮点归约序差异，其余四项逐位 0）。这解释了为什么 CP 分片在对齐时可以逐位复现：
每个 rank 内它跑的仍是同一套块。

---

## 4. Part A：CP 反向（单序列，等分）

同一份 bf16 操作数：一侧在**不分片**全序列上跑 `chunk_kda` + backward；另一侧
`torch.multiprocessing.spawn` 起 `world` 个真实 NCCL rank（`init_process_group("nccl")`，
单机需 `NCCL_NVLS_ENABLE=0`），每 rank 只拿 `[rank*part,(rank+1)*part)` 的 token 段、
传 `cp_context`，损失 $L=\langle o,\bar o\rangle$（CP 禁终态）。把各 rank 的
`dq/dk/dv/dg/dbeta` 沿 token 维拼回全序列与不分片比，下面为 `max_rel`：

| `lens` | `world` | `C` | `part` | `o` | `dq` | `dk` | `dv` | `dg` | `dbeta` |
|---|---|---|---|---|---|---|---|---|---|
| `[256]` | 1 | 64 | 256 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[256]` | 2 | 64 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[192]` | 3 | 64 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[256]` | 4 | 64 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[256]` | 2 | 32 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[256]` | 4 | 32 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |

**全部 `max_abs = max_rel = 0`（逐位相同）**，覆盖 `world=1/2/3/4`（`world≥3` 时
`pre/post_num_ranks>1`，走**多级** rank 链的 `all_gather`+`merge`）与 `C∈{32,64}`。

**为什么能逐位相同**：分片对齐意味着每个 rank 的 chunk 序列与无 CP 路径的对应段**逐块相同**；
`pre_process` 用同一 kernel（`chunk_gated_delta_rule_bwd_dhu`）在同一 fp32 状态伴随上跑，
唯一的外部输入 `dht` 由 `dhm` 在 fp32 里重建——与无 CP 内核在块边界寄存器里携带的值**同源**。
故每个 chunk 上的指令流与数值路径完全一致，结果自然逐位相同。

---

## 5. Part B：varlen × CP

多条序列展平成 `B=1`、按 token 等分到各 rank；每 rank 的 `cu_seqlens` 由 `cp_context`
按 rank 区间**裁剪**（`ops/cp/context.py:83-110`）。参考侧用**全序列** `cu_seqlens`。

| `lens` | `world` | `C` | `part` | `o` | `dq` | `dk` | `dv` | `dg` | `dbeta` |
|---|---|---|---|---|---|---|---|---|---|
| `[128,128]` | 2 | 64 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[64,64,64,64]` | 4 | 64 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[128,128]` | 2 | 32 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[128,64,64]` | 4 | 32 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[192,64]` | 2 | 64 | 128 | 0 | 0 | 0 | 0 | 0 | 0 |

覆盖「一条序列跨 rank」（`[128,128] world=2`）、「每序列各占一 rank」（`[64,64,64,64]`）、
以及**一条 rank 内既有序列尾又有下一条序列头**的边界（`[192,64] world=2`，rank 界 128 落在
第一条序列内部、第二条序列完整在 rank 1）。全部**逐位相同**——说明 `cp_context` 的
per-rank `cu_seqlens` 裁剪与跨 rank 状态收发在多序列下也正确。

---

## 6. Part C：两条 CP 硬约束在反向上同样成立

前向（[06](06-chunk-varlen.md) §5）已给出两条约束，本篇确认**反向同样被它们打破**：

| 情形 | `lens` | `world` | `C` | `part` | aligned | `o` | `dq` | `dk` | `dv` | `dg` | `dbeta` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `part % C != 0` | `[250]` | 2 | 64 | 125 | ✗ | 1.0e-3 | 1.2e-3 | 1.9e-3 | **2.3e-3** | 6.4e-4 | 1.3e-3 |
| `total % world != 0` | `[130]` | 4 | 32 | 32 | ✗ | 0 | 0 | 3.6e-2 | **4.5e-2** | 1.2e-2 | 7.4e-3 |

- **`part % chunk != 0`**（rank 界切在块内）：前向 `o` 差 2.2e-3（[06] §5），反向六项梯度
  同量级（1e-3~2.3e-3），**因块被横切、块内 `dAqk/dAkk` 的伴随不完整**。
- **`total % world != 0`**（尾块无人认领）：`[130] world=4` 只认领 128 个 token，反向
  `dk/dv/dg/dbeta` 反而**更大**（最高 4.5e-2）。一个反直觉的细节：此时 `o` 与 `dq` **仍逐位为 0**
  （前 128 个 token 的前向输出与 `q` 的伴随只依赖本 rank 内部），但**状态伴随链断了**——
  末位 rank 少算了一段状态反馈，`dk/dv/dg/dbeta` 全错。**这比前向更危险**：前向只在尾部丢 token，
  反向会把「状态外部性」的缺失回灌到**保留 token 的梯度**上。

即 CP 的两条契约 `total % world == 0` 且 `part % chunk_size == 0`
（`ops/cp/context.py:83-85`）**对前反向都是硬约束**。

---

## 7. Part D/E：核内融合门控与 GVA × CP

两条 Megatron 会真实踩到的组合：

- **Part D（融合门控，logits 入口）**：`use_gate_in_kernel=True` + `use_beta_sigmoid_in_kernel=True`，
  传原始 logits + `A_log/dt_bias`，`[256] world=2 C=64` 与 `[128,128] world=2 C=64`
  六项（此路径为 `dg_logit/dbeta_logit`）**逐位 0**。CP 契约与门控融合**无交互**。
- **Part E（GVA，$H_V=2H$）**：v/g/beta 头数为 q/k 的 2 倍，`[256] world=2` 与
  `[128,128] world=4`（C=64）六项**逐位 0**。GVA 的头映射（`i_h = i_hv // (HV//H)`）在
  CP 前处理里用同一套索引（`cp/chunk_delta_h.py:525`），跨 rank 状态按 `HV` 头独立收发，
  与 GVA 的组求和（`chunk_bwd.py:594-596`）正交。

---

## 8. 结论、移植要点与下一步

**结论**：M8-2 把数值锚定补到 **CP 反向**与 **`varlen × CP`** 梯度：
对齐分片时 CP 与不分片**逐位相同（`max_abs=0`）**，覆盖单/多序列、`world=1/2/3/4`、
`C∈{32,64}`、融合门控、GVA；两条 CP 分片约束在反向上同样成立（且反向会把分片错
回灌到保留 token 的梯度，比前向更隐蔽）。参考实现的块反向与逐 token 反向在 fp64 下
机器精度一致（Part 0）。

**移植要点（补进 [02](02-backward.md) §7）**：

1. **CP 反向不能走 `dht` 入口**：`chunk.py:362-364` 与 `cp/chunk_delta_h.py:849` 双重
   断言禁止；跨 rank 的状态伴随由 `dhm`（本 rank 贡献）+ `all_gather` + `merge` 重建。
   自研 kernel 时须复刻这条「本 rank 算 `[he|m]` 部分 → 全局 gather → 按后驱优先 merge」。
2. **多序列只压一条 `h0`**（`compress_h0`，:923）：只有第一条序列的状态可能跨 rank；
   其它序列在 rank 内闭合。反向 `expand_h0` 再补零成 `[N,...]`。
3. **CP 与门控/GVA 正交**：融合门控、GVA 的头映射都不改 CP 的状态收发，可自由叠加。
4. **分片必须落块边界**（`total % world == 0` 且 `part % chunk == 0`）：Megatron 的
   `cp_partition_mode="zigzag"`（`kda.py:260-270`）须保证此性质；否则反向梯度（不只尾块）出错。

**未覆盖（下一步）**：
- **M9-1**：按 `docs/02` §7 检查表逐项对齐 Megatron `kda.py` 的调用契约
  （SBHD 转置、`dht` 初值、`A_log/dt_bias` fp32、`gdn_pre_gated_delta_rule_fusion` 缺口）。
- **M10-1**：Megatron 路径端到端前反向对拍。

运行：

```
CUDA_VISIBLE_DEVICES=3,4,5,6,7 FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
    ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla_m8b.py
# 输出存 harness/check_port_fla_m8b.out.txt
# Part 0 为 fp64 CPU；Part A/B/D/E 会 spawn 真实 NCCL rank（单机多卡）
```
