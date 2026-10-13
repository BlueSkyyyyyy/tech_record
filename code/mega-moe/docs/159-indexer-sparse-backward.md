# 稀疏（块选择）MQA logits（lightning indexer）前向的反向：压缩输出的伴随 + 支持集散播（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的第 159 篇（P149 增量）。
> 第 157 篇（P147-1）拆了 lightning indexer 的**连续版**训练反向、第 158 篇（P148-1）拆了**分页版**
> （页表逆映射 + 别名累加）。但生产 long-context 训练真正跑的是**稀疏版**：只在一小撮**选中的 KV 块**
> 内打分，输出**压缩**成 `[T, num_max_sparse_blocks·SB]`。DeepGEMM 的稀疏前向
> `fp8_fp4_sparse_mqa_logits` / `fp8_fp4_paged_sparse_mqa_logits` 是 **SM100 专属**、且只输出 logits；
> 稀疏版反向一直没人讲过。本篇补上：① 稀疏前向的**块选择 + 压缩**语义（真实行号）；
> ② **压缩输出是 dense logits 的一个「选择」（gather），故其伴随是 scatter**——未选中/越界位置梯度恒 0；
> ③ 支持集上的四路梯度 + 跨行**别名散播**；④ 「块选择」天然是 stop-gradient（块由另一个打分器选、不可微）。
> 数学 / 支持集 / 别名 / KL 梯度**架构无关**，全部在 **H100(sm90)** 真机用 torch.autograd + 中心差分对拍，
> 并给一个可跑的 **CUDA 稀疏反向后端**与**真实 SM90 dense kernel 前向锚**。

源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。论文坐标：DeepSeek-V3.2 技术报告
[arXiv:2512.02556](https://arxiv.org/abs/2512.02556) §2.1。

## 1. 定位：连续反向（157）、分页反向（158）都有了，稀疏反向还缺

前两篇把 indexer 反向的两条路径补齐：

- **连续**（第 157 篇）：query token 的有效 KV 区间是 `[ks_t, ke_t)`，KV 是一段连续张量
  （`tests/test_attention.py:78-101` 的 `ref_mqa_logits`），反向只需沿连续下标散播；
- **分页**（第 158 篇）：KV 在物理页池里、经 block table 间接寻址，反向要沿**页表逆映射**散播回物理槽。

但真正投产的 **DeepSeek-V3.2 稀疏注意力（DSA）** 走的是第三条：先用 indexer 从整段上下文里**选出一小撮
KV 块**（每 query 2048 个 KV、块粒度 `sparse_block_kv ∈ {8,16}`），**只在选中块内**算细粒度 logits、并把
结果**压缩**传给后续。稀疏前向在 DeepGEMM 里是独立一族 kernel（README「V3.2 MQA kernels for the indexer」）；
它们**只输出 logits**，训练侧反向要自己补。

稀疏相对分页多了一层结构：**支持集（support）不再是整段窗口，而是被选中块覆盖的稀疏子集**。
这带来两个连续 / 分页都没有的反向语义点：

1. **压缩输出 = 对 dense logits 的选择**：输出第 `c` 列对应 dense logits 的某个 token；反向要把它
   **scatter** 回 dense 窗口，其余位置梯度恒 0（§3、§4）；
2. **「选中哪些块」不可微**：块索引由上游（一个独立打分器 / top-k）给出，反向对它没有梯度；梯度只在
   **选中块内、且落在有效窗口内**的 token 上（§6）。

## 2. 稀疏前向语义（真实行号）

### 2.1 测试侧参考（唯一正确性规范）

稀疏的正确性规范是 `tests/test_attention.py` 的两段纯 torch 代码：`make_sparse_kv_block_indices`
（`:442-475`，构造选中块号）与压缩展开（`:655-666`）。后者逐行读：

```
kv_offsets   = arange(sparse_block_kv)                                    (:655)
block_offsets = context_starts % sparse_block_kv                          (:657)   # 未对齐 ks 的块内平移
token_indices = (sparse_indices.unsqueeze(-1) * sparse_block_kv
                 + block_offsets[:, None, None] + kv_offsets).flatten(1)  (:658-659)
num_sparse_tokens = num_sparse_blocks * sparse_block_kv                   (:661)
valid_mask  = (arange(C) < num_sparse_tokens[:, None])                    (:662)
valid_mask &= token_indices >= context_starts[:, None]                    (:663)
valid_mask &= token_indices < context_lens[:, None]                       (:664)
sparse_logits = sparse_logits[valid_mask]                                 (:665)   # 只留有效项
full_logits   = full_logits.gather(1, token_indices - context_starts)[valid_mask]  (:666-668)
```

即：压缩输出第 `c = l·SB + kvj` 列（`l<num_sparse_blocks`）对应 dense（logical）token

$$
\mathrm{token}_c = \mathrm{sparse\_indices}[r,l]\cdot SB + (ks_r \bmod SB) + kvj,
$$

只有 `ks_r \le \mathrm{token}_c < ke_r` 时有效。三个与反向绑定的细节：

1. **`sparse_indices` 是绝对块号**（以 `SB` 为单位）；`context_start//SB` 是窗口首块，`block_offsets`
   把块内起点平移到 `ks_r`（未对齐 ks 时块边界与窗口边界错开）——`csrc/apis/attention.hpp:304-306`
   的注释「with unaligned ks, block i starts at `i·sparse_block_kv + ks % sparse_block_kv`」即此。
2. **块数是逐行的稀疏量** `num_sparse_blocks[r] = min(num_available_blocks, num_max_sparse_blocks)`
   （`:454-455`）；不足 `num_max_sparse_blocks` 的槽用**最后一个选中块**填充（`:470`）——故压缩输出的
   **padding 列会重复映射到已选块**（反向要按「任一 support」OR，不能按「最后一列」覆盖）。
3. **`num_max_sparse_blocks % 4 == 0`、`sparse_block_kv ∈ {8,16}`**（host `attention.hpp:381-382`）。

### 2.2 设备侧：选择、压缩与物理块映射

主 kernel `deep_gemm/include/deep_gemm/impls/sm100_sparse_mqa_logits.cuh`（688 行，SM100 专属、
第 22/136/137 篇已拆前向）里，**压缩列映射**落在 math warp 的写回处：

```
// Map split-local slots back to compressed-logits columns
q_slot_offset = (packed_slot_offsets >> (q_token_offset * kNumSparseSlotBits)) & kInvalidSparseSlot  (:628)
output_col_idx = (q_slot_base + q_slot_offset) * SPARSE_BLOCK_KV + token_in_kv_block                   (:631)
```
（`:627-633`）。`(slot)*SPARSE_BLOCK_KV + token_in_kv_block` 正是 §2.1 的压缩列——**块号被折叠进列号**，
这就是「压缩」。

**物理块三态映射**在 metadata kernel `scheduler/sm100_sparse_mqa_logits_metadata.cuh:359-386`：
`write_merged_kv_block`（`:359`）里 `physical_kv_block_idx` 三选一——

- 分页：`block_table[...]*kNumKVBlocksPerPage + logical%kNumKVBlocksPerPage`（`:383`）；
- 连续：`logical * (kUseUnalignedKs ? 1 : SPARSE_BLOCK_KV)`（`:385`，未对齐 ks 时按 token、否则按块）。

host 入口 `csrc/apis/attention.hpp:370-395`（连续）与 `:398-441`（分页）：二者都断言
`arch_major==10`（`:380/405`），输出分配成
`torch.empty({align(num_q_tokens,kBlockQ), logits_stride})`（`:398`，`kBlockQ=2`，见
`layout/sparse_mqa_logits.cuh:17`），再 `index` 切到 `[num_q_tokens, num_max_sparse_blocks*SB]`（`:396-399`）。

## 3. 稀疏反向：压缩的伴随 + 支持集四路梯度

设上游 `dG[r,c]` 是压缩输出第 `c` 列（仅有效项非 0），`dI_full[r,\mathrm{token}]` 是 dense 窗口的梯度。
**压缩是选择（gather）**：`G[r,c] = I[r, token_c]`，故其伴随是**把 `dG` 散播（scatter）到被选中的 token**：

$$
\boxed{\ dI_{\mathrm{full}}[r,t]\ \mathrel{+}=\sum_{c:\ \mathrm{token}_c=t} dG[r,c]\ },\qquad
dI_{\mathrm{full}}[r,t]=0\quad\text{若 } t \text{ 未被任何有效列选中}
$$

即**未选中 / 越界位置的梯度恒 0**。此后打分算子的四路反向与连续 / 分页**逐式相同**（只是求和范围收窄到支持集）：

$$
\frac{\partial \mathcal{L}}{\partial w_{r,h}}=\sum_{t\in supp(r)} dI_{\mathrm{full}}[r,t]\,\mathrm{relu}(z_{r,t,h}),\qquad
\frac{\partial \mathcal{L}}{\partial z_{r,t,h}}=dI_{\mathrm{full}}[r,t]\,w_{r,h}\,\mathbf{1}[z>0],
$$

$$
\frac{\partial \mathcal{L}}{\partial q_{r,h,d}}=\sum_{t} \frac{\partial\mathcal{L}}{\partial z_{r,t,h}}\,\hat{k}_{t,d},\qquad
\boxed{\ \frac{\partial\mathcal{L}}{\partial kv_{t}}[t]\ \mathrel{+}=\sum_{r,h:\,t\in supp(r)} \frac{\partial\mathcal{L}}{\partial z_{r,t,h}}\,q_{r,h,d}\,\mathrm{scale}_t\ }
$$

以及 `dscale_t = Σ_{r,h} (∂L/∂z_{r,t,h})·(q_{r,h}·kv_t)`（`khat = kv·scale`，对 scale 求导要用未缩放的 `kv`）。

H100 实测（`harness/exp_mqa_logits_sparse_backward.py`：Part A/B/C/D，fp64 autograd + 中心差分）：

- **Part A 伴随恒等式** `<dG, gather(I)> == <scatter(dG), I>`：`lhs = rhs = -2.692850`（逐位）；散播后
  未选中位置 `max|dI_full| = 0.0`。压缩每行 96 列（有效 64）对应 dense 窗口 512 列 → **8.0× 稀疏**。
- **Part B 支持集 KL 梯度** `dI=(softmax_{supp}(I)-p)·supp`：autograd relerr **2.208e-16**、支持集内行和
  **1.2e-16**、支持集外严格 **0**；反例（不 mask 支持集、对所有压缩列 softmax）relerr **3.050e-01**
  ——坐实「支持集掩码」是语义的一部分，不是可选优化。
- **Part C 四路反向**：`dq/dw` relerr **0.0**、`dkv` **4.119e-17**、`dscale` **8.379e-17**；中心差分
  **9.257e-10**。

## 4. 支持集里的别名：scatter-accumulate 仍是硬要求

稀疏不影响别名结构，只把它收窄：同一个 token 会被**多行**选中——

- **`next_n` 共享块**：同一序列的多个 next_n 槽复用同一段上下文块（前向里 `make_sparse_kv_block_indices`
  对同 request 的后续 token 做「保留 80% 上一 token 选中块」，`:461-469`）；
- **跨序列共享块**：prefix caching / beam search 的块复用。

Part D 实测（`S=4, N=3, SB=16, NBmax=6`，非首槽复用首槽选中块）：支持集 **768 个逻辑引用 → 224 个 token**，
被多引用的 token **224 个、最大共享度 9**。若用「写」（last-writer-wins）：

- 被共享 token 上最大绝对差 **2.198**（真实贡献被覆盖）；
- 相对正确累加的全局 **relerr = 0.8599**。

而 scatter-accumulate 与 autograd 逐位一致（relerr **6.138e-17**）。CUDA 后端里这条落在
`atomicAdd(&dkv[token*D+d], coeff*qd*scale)`；设备端 `[A]` 的「写」对照 relerr **0.887**。

**结论：稀疏反向的 `dkv` 必须做成支持集上的 scatter-accumulate；这不是优化，是正确性。**
注意一个坐标陷阱：压缩输出的 **padding 列会重复映射到已选块**（§2.1 细节 2），所以「哪些 token 被覆盖」
必须按**任一 support 列**判定，按「最后一列覆盖」会把已选 token 误标为未选（本实验踩过，见 §7）。

## 5. 稀疏 logits 上的 KL 蒸馏梯度

第 157/158 篇的结论在稀疏版**逐式不变**，只是 softmax / KL 要在**压缩支持集**内做：

$$
\frac{\partial\mathcal{L}^I}{\partial G_{r,c}}=(q_{r,c}-p_{r,c})\cdot\mathbf{1}[c\in supp(r)],\qquad
q_{r,:}=\mathrm{softmax}_{c\in supp(r)}(G_{r,:}).
$$

Part B 已验证：autograd 2.208e-16、支持集内行和 1.2e-16、支持集外 0。稀疏阶段（论文式 (4)）的语义
是「先在 top-k 选中块内重新归一化、再算 `q_S-p_S`」，与第 157 篇 §5 的 top-k 直通一致。

## 6. 「块选择」是 stop-gradient

`sparse_indices` 由上游一个独立打分器（或 top-k 块选择）给出，**不在** indexer 的计算图里：反向对它没有
梯度，也不用回传「哪块该选」。这与第 157 篇的 top-k 直通同构——**选择不可微、梯度只在选中集内**。稀疏版把
这一层从「token 级 top-k」升级成「**块级 top-k**」：粒度是 `SB ∈ {8,16}` 个 token 一块，选择面小一个数量级
（生产里每 query 2048 KV ≈ 128 块 / window 32K）。

由此得到一条工程结论：**稀疏反向的算力与显存都正比于「选中块数」而非「上下文长度」**——这正是 indexer
能被设计成「小头数 + 块稀疏」的前提。

## 7. CUDA 稀疏反向后端

`src/mqa_indexer_backward_sparse.cu`（架构无关，H100 真机 `ALL PASS`）：`grid = P`（每 query 行一 block）、
`block = (32, H)`，每 head 一个 warp。主 kernel 遍历「该行选中的块 × 块内 token」，现场判 support
（`token < start || token >= cl` 跳过），沿 lane 跨步点积 + `__shfl_xor` 归约得 `z`；`dq` 累寄存器、
`dw` 在 lane0；`dkv` 按 token 走 `atomicAdd`；`dscale` 因目标按 token 而非行，单独一趟逐 token 原子加。
自带 `[A]` 正确性 + `[B]` 吞吐：

```
[A] dq 5.906e-08 / dw 7.028e-08 / dkv 4.533e-08 / dkv_scale 6.127e-08  => ALL PASS
    (对照) 用「写」而非累加，dkv relerr = 8.870e-01
[B] S=64, N=1, H=32, D=128, sel=2048, blk=64     0.754 ms/call
    S=64, N=4, H=32, D=128, sel=2048, blk=64     1.522 ms/call
    S=256,N=1, H=32, D=128, sel=2048, blk=64     1.522 ms/call
```

（吞吐口径同第 157/158 篇，`macs = 4·P·selected_tokens·H·D / time`，打印值即该公式结果。）
**瓶颈仍是 `dkv` 的跨 `(r,h)` scatter 归约**（每 token 被 H 个头累加 → 原子争用）。稀疏把散播量从
`O(Σ_r cl_r)` 降到 `O(Σ_r selected_tokens_r)`，但生产实现仍会按物理块做局部归约以消除原子争用。
实验机器为共享 8 卡节点，绝对时延为受争用下界。

## 8. 真实 SM90 dense kernel 前向锚

稀疏前向 kernel 是 SM100 专属（`attention.hpp:380/405` 断言 `arch_major==10`），H100 上无法真跑。但
**压缩 = 选择** 这条语义可以在 H100 上用**真实的连续 dense kernel** 验证：Part E 用
`deep_gemm.fp8_fp4_mqa_logits`（SM90 分支，`q` 纯 E4M3、KV 用官方 `per_custom_dims_cast_to_fp8` 的
per-token scale、`H=32, D=128`、`ks=0, ke=KV`）算出 dense logits，再按 §2.1 的公式 gather 选中 token：

- 真实 kernel 压缩输出 vs 模拟参考（fp8 反量化）压缩输出 **relerr = 8.372e-04**（容差 5e-3）；
- 压缩支持集是窗口的**严格子集**：**5120 有效项 / 32768 窗口**（跳过所有未选中块）——坐实「压缩 = 选择」。

这给出一条与 SM100 无关的可核对锚：**稀疏前向的输出，就是把 dense logits 按选中块 gather 并压缩列号**；
其反向按 §3 就是同一 gather 的伴随（scatter）。

## 9. 小结

- **稀疏前向** = dense/paged 打分 + **块选择**（`sparse_indices`，`:442-475`）+ **压缩输出**
  （`output_col_idx=(slot)·SB+token_in_block`，kernel `:627-633`；host 输出 `[T, num_max_sparse_blocks·SB]`，
  `attention.hpp:396-399`）。块选择是 stop-gradient。
- **稀疏反向** = dense 反向 + **压缩的伴随（scatter）** + 支持集掩码：
  `dI_full = scatter(dG)`（未选中位置 0）→ 四路梯度只在支持集内 → `dkv/dscale` 沿支持集 scatter-add。
- **支持集别名是硬要求**：next_n 共享块 + 跨序列共享块使一个 token 被多行引用；「写」丢 ~86% 贡献
  （CUDA 设备端 88.7%），scatter-add 与 autograd 逐位一致。
- **KL 蒸馏梯度** `dG=(softmax_supp(G)-p)·supp` 在稀疏版逐式不变（支持集内归一化、支持集外 0）。
- **CUDA 稀疏后端** `[A] ALL PASS`；瓶颈仍是 `dkv` 的跨 `(r,h)` scatter 归约。
- **真实 SM90 dense kernel 锚** 验证「压缩 = 选择」：gather 后 relerr 8.4e-4、支持集 5120/32768。
- **边界**：SM100 的 MXFP4/MXFP8、packed UE8M0、varlen 与稀疏主 kernel 仍属 SM100 专属、仅分析
  （第 19/22/23/24/60/136/137 篇）；本篇的数学 / 支持集 / 压缩伴随 / 别名 / KL 梯度**架构无关**，H100 真机可跑。

实验与原始输出：`harness/exp_mqa_logits_sparse_backward.{py,sh,out.txt}` +
`src/mqa_indexer_backward_sparse.cu`（H100 真机 `PASS=16 / FAIL=0`、设备端 `[A] ALL PASS`；
`mqa_indexer_backward_sparse.cu` sha256 `e216cc6d…`）。
