# 分页 KV 的 MQA logits（lightning indexer）反向：页表逆映射散播 + 别名累加（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的第 158 篇（P148 增量）。
> 第 157 篇（P147-1）把 lightning indexer 的**连续版**训练反向拆透了（KL 蒸馏梯度 + 打分算子四路反向 +
> top-k 直通 + FP8 STE）。但生产推理 / long-context 训练用的是**分页 KV cache**——KV 存在物理页池里、
> 每个 query 行经 **block table** 间接寻址、还有自己的 context 长度。DeepGEMM 的分页前向
> `fp8_fp4_paged_mqa_logits`（SM90/SM100 两条实现）只输出 logits，分页版反向一直没人讲过。
> 本篇补上：① 分页前向的寻址语义（真实行号）；② 反向梯度如何**沿页表逆映射散播回物理 KV**；
> ③ 为什么「一个物理槽被多行引用」使 **scatter-accumulate（原子加）成为正确性硬要求**而非优化；
> ④ 分页 logits 上的 KL 蒸馏梯度。全部在 **H100(sm90)** 真机用 torch.autograd + 中心差分对拍，
> 并给一个可跑的 **CUDA 分页反向后端**与**真实 SM90 分页 kernel 前向锚**。

源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。论文坐标：DeepSeek-V3.2 技术报告
[arXiv:2512.02556](https://arxiv.org/abs/2512.02556) §2.1。

## 1. 定位：连续反向有了，分页反向还缺

第 157 篇的反向针对**连续**布局：query token 的有效 KV 区间是 `[ks_t, ke_t)`，KV 是一段连续张量
（`tests/test_attention.py:78-101` 的 `ref_mqa_logits`）。但线上/长上下文场景里 KV 是**分页**的：

- KV 驻留在物理页池 `kv_cache[num_blocks, block_kv, 1, dim]`（`tests/test_attention.py:355`、
  `csrc/apis/attention.hpp:204-226` 把它从融合布局里切出来）；
- 每个 query 行 `r=(seq i, next_n 槽 j)` 经 **block table** 找到自己那段页
  （`tests/test_attention.py:238-241`、kernel `deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh:191-198`）；
- 每行有**独立的 context 长度** `context_lens[i,j]`，token `j` 只看前缀 `[0, context_lens[i,j])`
  （`tests/test_attention.py:228-229`）。

分页前向在 DeepGEMM 里是独立 kernel family（README「V3.2 MQA kernels for the indexer」），
但它们**都不产生梯度**；框架侧（训练）要自己补反向。分页相对连续多了两层结构：**页表间接寻址**
与**每行掩码**——反向因此多了一个连续版不存在的正确性陷阱（§4）。

## 2. 分页前向语义（真实行号）

### 2.1 测试侧参考（唯一正确性规范）

`tests/test_attention.py:225-253` 的 `ref_paged_mqa_logits` 是权威定义。逐段读：

```
q       [batch_size, next_n, num_heads, dim]            (:230)
context_lens [batch_size, next_n]，token j 看 [0, context_lens[i,j])  (:228-229)
logits  [batch_size*next_n, max_model_len]，先填 -inf       (:232)

对每个序列 i：                                                   (:233)
  context_len = q_context_lens.max()                             (:235)
  num_blocks  = ceil(context_len / block_size)                   (:238)
  block_idxs  = block_tables[i][:num_blocks]                     (:239)
  kv_slice    = kv_cache[block_idxs]                             (:240)  # 页 gather
  kx = kv_slice.permute(2,3,0,1).reshape(kv_heads, dim, -1)      (:241)  # 拼成逻辑连续 KV
  s  = qx @ kx                                                   (:243)  # [H, next_n, total_tokens]
  mask = k_offsets < q_context_lens[:,None]                      (:247)  # 每行自己的掩码
  s = relu(s) * weight_slice                                     (:249)
  s = s.sum(dim=0)                                               (:250)
  logits[...] = where(mask, s, -inf)                             (:251)
```

三个与反向绑定的细节：

1. **页 gather 只取 `ceil(context_len/block_size)` 块**（`:238-239`）：`context_len` 是**该序列内所有
   `next_n` 槽的最大值**，所以块的集合覆盖全序列、但**每一行只用到其中前缀部分**（由 `mask` 决定）。
   一个物理块会被同一序列的多个 `next_n` 槽引用——这就是 §4 的别名来源。

2. **权重按行取**：`weight_slice = weights[i*next_n:(i+1)*next_n,:].transpose(0,1)`（`:236`），
   即 `weights[(i*next_n + n), h]`——与连续版一致（`tests/test_attention.py:85`）。

3. **无效位置是 `-inf`**（`:232/251`），在训练里当掩码：反向只对被引用的有效位置收梯度（§3）。

### 2.2 设备侧主 kernel

SM90 分页主 kernel `deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`：

- 签名 `:30-38`（`context_lens`、`block_table`、`indices`、`schedule_meta` 加 4 个 TMA 描述符；
  `:39` 断言 `not kIsVarlen`——SM90 不支持 varlen，varlen 是 SM100 的事）；
- 调度器构造 `:138-140`（`SM90PagedMQALogitsScheduler<...>`，work 单位 = 256 KV token 一个 split，
  见第 30 篇）；
- **页表读取** `:191-198`：每个 lane 读一个块号
  `block_table[q_idx * block_table_stride + (kv_idx + kv_group_idx + lane_idx*kNumMathWarpGroups)]`
  （`:196`），越界补 0（`:195-196` 三元），再 `__shfl_sync` 广播（`:198`）；
- **KV TMA** `:204-211`：`tma::copy<...>(&tensor_map_kv, ..., 1, kv_block_idx)`（`:206-207`）按块号
  拉 KV，KV scale 单独一条 TMA（`:208-209`）；
- **输出行偏移** `:255`：`kv_offset = q_idx*kNextN*logits_stride + ((kv_idx+kv_group_idx)*BLOCK_KV + sub_warp_offset)`；
- **归约与写回** `:294-329`：沿头维 `relu` + 加权求和、warp 内 `__shfl_xor` 归约（`:317-323`），
  最后 `logits[kv_offset + i*logits_stride + v]`（`:327-328`）——**注意它对该 KV 块内所有位置都写**，
  不做 context 掩码（`:326` 注释「we have redundant writes here」）。这正是 Part E 发现的：
  kernel 输出的无效区不是 `-inf`，而是计算值（§7）。

host 侧 `csrc/apis/attention.hpp:174-278` 是入口 `fp8_fp4_paged_mqa_logits`：页池/融合布局切分
（`:204-226`）、block table 契约（`:229-235`）、输出分配与 stride 对齐（`:258-263`，`split_kv=256` @ SM90
`:258`）、SM90 分派 `:274-277`。约束：SM90 只支持 `block_kv==64`（`:201-202`）、heads ∈ {32,64}（`:127`）。
元数据 `get_paged_mqa_logits_metadata:136-172`（SM90 非 varlen `:164-166`）。

## 3. 分页反向：页表逆映射 + 四路梯度

设上游 `dI[r,s] = dL/dI[r,s]`（仅有效位置非 0），`phys(r,s) = block_table[i, s//BS]*BS + s%BS`
是逻辑位置 `s` 的物理槽。链式求导与连续版**逐式相同**，唯一差别是 `kv` 那一路要**逆映射回物理槽**：

$$
\frac{\partial \mathcal{L}}{\partial w_{r,h}}=\sum_s dI_{r,s}\,\mathrm{relu}(z_{r,s,h}),\qquad
\frac{\partial \mathcal{L}}{\partial z_{r,s,h}}=dI_{r,s}\,w_{r,h}\,\mathbf{1}[z_{r,s,h}>0],
$$

$$
\frac{\partial \mathcal{L}}{\partial q_{r,h,d}}=\sum_s \frac{\partial\mathcal{L}}{\partial z_{r,s,h}}\,\hat{k}_{\mathrm{phys}(r,s),d},\qquad
\boxed{\ \frac{\partial \mathcal{L}}{\partial kv_{\mathrm{phys}}}[p]\ \mathrel{+}=\sum_{r,h,s:\ \mathrm{phys}(r,s)=p}\frac{\partial\mathcal{L}}{\partial z_{r,s,h}}\,q_{r,h,d}\ }
$$

其中 `khat = kv_phys * scale`，故对**反量化前的** `kv_phys` 求导还要乘回 `scale`：

$$\frac{\partial \mathcal{L}}{\partial kv_{\mathrm{phys}}[p]}=\mathrm{scale}_p\cdot\sum_{r,h,s:\,\mathrm{phys}(r,s)=p}\frac{\partial\mathcal{L}}{\partial z_{r,s,h}}\,q_{r,h,d}.$$

三个非平凡点：

1. **`dq` 沿 `s`（每行独立、输出 query 维），`dkv_phys` 沿 `(r,h)` 汇聚（scatter/atomic）**——与连续版
   同构，但 `s` 现在索引的是**逻辑位置**，要经页表翻译。
2. **`dkv_phys` 是「页表逆映射上的散射」**：同一个物理槽会被多行引用，累加是必须的（§4）。
3. **掩码即梯度**：无效位置 `dI=0`，所以页里未被任何有效行引用的槽梯度天然为 0；`context_lens`
   是反向的稀疏结构。

H100 实测（`harness/exp_mqa_logits_paged_backward.py` Part B，fp64 autograd）：`dq/dw` relerr **0.0**、
`dkv_phys` relerr **7.59e-17**；中心差分 3 个物理槽条目最差 **1.57e-09**。

## 4. 别名累加：分页反向的正确性硬要求

连续版里 KV 的每一维只被一个逻辑位置引用，所以 `dkv[s] = ...`（写）与累加等价。分页版**不成立**：

- **`next_n > 1` 的前缀重叠**：同一序列的每个 `next_n` 槽都看前缀 `[0, cl)`，于是**同一个物理槽被
  多个 query 行引用**（`tests/test_attention.py:340-348` 的 context 增长正是如此）；
- **跨序列共享页**：前缀缓存 / beam search / 块复用会让不同序列的 `block_table` 指向同一物理块
  （测试里 `block_table` 从随机池分配、`tests/test_attention.py:356-361`）。

Part C 实测（`S=4, N=3, BS=64, L=512`，序列 1 复用序列 0 的页）：**2816 个逻辑引用 → 2112 个物理槽，
被多引用的槽 704 个、最大共享度 2**。若用「写」（last-writer-wins，即朴素 `dkv[phys] = ...`）：

- 只在被引用的槽上最大绝对差 **3.88**（真实贡献被覆盖）；
- 相对正确累加的全局 **relerr = 0.489**（近一半贡献丢失）。

而 scatter-accumulate 与 autograd 逐位一致（relerr **1.10e-16**）。CUDA 后端里这条就落在
`atomicAdd(&dkv[phys*D+d], coeff*qd*scale)`；设备端 `[A]` 对照坐实「写」的 relerr **0.839**。
**结论：分页反向的 `dkv` 必须做成 scatter-accumulate；这不是优化，是正确性。**

## 5. 分页 logits 上的 KL 蒸馏梯度

第 157 篇 §3 的结论在分页版**逐式不变**，只是 softmax / KL 要在**每行的有效集**内做（`mask`）：

$$\frac{\partial\mathcal{L}^I}{\partial I_{r,s}}=(q_{r,s}-p_{r,s})\cdot\mathbf{1}[s<cl_r],\qquad q_{r,:}=\mathrm{softmax}_{s\in[0,cl_r)}(I_{r,:}).$$

Part D 实测：解析 `softmax-p` vs autograd relerr **1.70e-16**；有效集内行和 **1.32e-16**；
无效位置梯度严格 **0.0**。稀疏阶段（论文式 (4)）则是「先在 top-k 选中集 `S_t` 内重新归一化、再算
`q_S-p_S`」，与第 157 篇 §5 的 top-k 直通一致。

## 6. CUDA 分页反向后端

`src/mqa_indexer_backward_paged.cu`：`grid = P`（每 query 行一 block）、`block = (32 lane, H)`，
每个 head 一个 warp——沿 lane 跨步点积 + `__shfl_xor` 归约得 `z_{r,s,h}`；`dq` 累寄存器（每 block 独占
一行 `r`）、`dw` 在 lane0 累加；`dkv_phys` 沿**页表逆映射** `phys = block_table[i, s/BS]*BS + s%BS`
走全局 `atomicAdd`（`coeff*qd*scale`）。自带 `[A]` 正确性 + `[B]` 吞吐：

```
[A] dq 7.48e-08 / dw 7.91e-08 / dkv_phys 3.45e-08  => ALL PASS
    (对照) 用「写」而非累加，dkv_phys relerr = 0.839
[B] S=64,N=1,H=32,BS=64,avg=8192    27.2 ms/call  ~0.3 GFLOP/s
    S=64,N=4,H=32,BS=64,avg=8192    53.7 ms/call  ~0.6 GFLOP/s
    S=256,N=1,H=32,BS=64,avg=8192   55.1 ms/call  ~0.6 GFLOP/s
```

**为什么这么慢（~0.3–0.6 GFLOP/s）**：与第 157 篇的连续后端同一根因——`dkv` 是跨 `(r,h)` 的
**scatter 归约**（scatter 量 `O(Σ_r cl_r · H · D)`），本实现用逐元素 `atomicAdd` 只用于验证语义 + 给量级；
生产实现会按物理块做局部归约。这再次坐实「indexer 头数被设计得很小」（论文 §2.1「a small number of
heads」）的工程动机。**分页并不改变这个瓶颈，只是把归约的目标地址从连续下标换成页表逆映射。**

## 7. 真实 SM90 分页 kernel 前向锚

Part E 在 H100 上真跑 DeepGEMM 的 `fp8_fp4_paged_mqa_logits`（SM90 分支，`block_kv=64`、`heads=32`、
q 纯 E4M3、KV 用官方 `kv_cache_cast_to_fp8` 的融合布局 `tests/test_attention.py:270-282`）：

- 真实 kernel vs 模拟参考（fp8 反量化后跑 `ref_paged_mqa_logits`）`calc_diff = 2.51e-09`（tol 1e-3）；
- **发现**：kernel 对**无效区（`s >= context_lens`）不写 `-inf`**，而是写了块内计算值
  （实测无效区落在 `[-3.15, 967.0]`）。这与参考/连续版不同，也正是官方测试在比较前必须
  `masked_fill(ref_neginf_mask, 0)` 的原因（`tests/test_attention.py:397-408`）——**分页反向只对有效位置
  收梯度**，与 kernel 的冗余写无关。

这条「kernel 侧不保证 -inf、由调用方掩码」是分页与连续语义的一处真实差异：连续版
（`ref_mqa_logits:96-98`）把 `[ks,ke)` 外填 `-inf`，分页 kernel 则把整块算完、把掩码留给上层。

## 8. 小结

- 分页前向 = 连续打分 + **页表间接寻址** + **每行 context 掩码**（`tests/test_attention.py:225-253`；
  kernel `sm90_fp8_paged_mqa_logits.cuh:191-211/327-328`）。
- 分页反向 = 连续反向 + **`dkv` 沿页表逆映射 scatter-accumulate**；`dq/dw` 与连续版同构。
- **别名累加是硬要求**：`next_n` 前缀重叠 + 跨序列共享页使一个物理槽被多行引用；「写」丢 ~49% 贡献
  （CUDA 设备端 83.9%），scatter-add 与 autograd 逐位一致。
- KL 蒸馏梯度 `dL/dI = (softmax(I)-p)·mask` 在分页版逐式不变（有效集内归一化、无效位 0）。
- CUDA 分页后端 `[A] ALL PASS`；吞吐与连续版同量级（~0.3–0.6 GFLOP/s），瓶颈仍是 `dkv` 的 scatter 归约。
- 真实 SM90 分页 kernel 前向与参考一致（`calc_diff 2.51e-9`），并坐实「kernel 不保证无效区 -inf」。
- **边界**：SM100 的 MXFP4/MXFP8/packed UE8M0 与 varlen 版主 kernel 仍属 SM100 专属、仅分析
  （第 19/22/23/24 篇）；本篇的数学 / 页表语义 / 别名累加 / KL 梯度**架构无关**，H100 真机可跑。

实验与原始输出：`harness/exp_mqa_logits_paged_backward.{py,sh,out.txt}` +
`src/mqa_indexer_backward_paged.cu`（H100 真机 `PASS=14 / FAIL=0`、设备端 `[A] ALL PASS`）。
