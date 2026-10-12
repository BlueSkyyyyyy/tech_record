# DeepGEMM 稀疏注意力 indexer 的 MQA logits：连续 / 分页路径（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P9 增量（第 19 篇）。
> 前面 18 篇都在 MoE 及其反向、或 MoE 相邻的融合算子里打转；本篇转向 DeepSeek 模型的另一条
> 关键计算链——**稀疏注意力的轻量 indexer 打分（MQA logits）**。它和 MoE 没有数据依赖，
> 但同属「DeepGEMM 把 DeepSeek 模型里的关键 primitive 收进同一份代码库」这件事，
> 也是理解 DeepSeek V3.2 稀疏注意力（lightning indexer）如何用 Tensor Core 做「打分」的核心。
>
> 为什么值得单独一篇：
> - 它是一条**被 DeepGEMM README 明确列为独立 kernel family** 的算子（非 MoE），
>   和 grouped GEMM / mega-kernel 共享同一套 JIT、TMA、WGMMA、PDL 基础设施；
> - 它在 **SM90（H100）上可跑**（连续版 + 分页版都有实现），可以留下真实数字；
> - 它的数学是「**共享 K 的 MQA 点积 → ReLU → 按头加权求和**」，和标准 attention 的
>   softmax 注意力完全不同——没有 softmax、没有 online-softmax 状态，只用一次 GEMM + 归约，
>   因此实现上也完全不同（无 running max、无 rescale）。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。
> **SM90 连续 / 分页两条路径 H100 可跑**；SM100 的 MXFP4/MXFP8 变体与 sparse（top-k 块）变体
> **仅分析**（`fp8_fp4_sparse_mqa_logits` 在 `attention.hpp:380` 直接断言 `arch_major == 10`）。

## 1. 定位：它算什么

DeepSeek V3.2 的稀疏注意力（lightning indexer）在正式 attention 之前，先用一个轻量网络给
「每个 query token × 每个 KV token」打一个相关性分数，再据此 top-k 选 token。这个打分就是
**MQA logits**：query 有 `num_heads` 个头，但 KV 只有一份（Multi-Query Attention），
每个头算一个点积，ReLU 后按可学习的 per-head 权重加权求和，得到标量分数。

DeepGEMM README `:100-108` 给出逐 token 定义：

```
kv_j    = kv[0][j, :] * kv[1][j].unsqueeze(1)   # [head_dim]，KV 每 token 一个 scale
out_ij  = q[i, :, :] @ kv_j                      # [num_heads]，每头点积
out_ij  = out_ij.relu() * weights[i, :]          # [num_heads]，ReLU 后加权
out_ij  = out_ij.sum()                           # 标量 logit
```

对整个 batch，输出是 `logits[seq_len, max_seqlen_k]`：**每行压缩存储自己的有效 KV 区间**
（从第 0 列开始，长度 `cu_seq_len_k_end[i] - cu_seq_len_k_start[i]`，见 README `:99`）。

和标准 attention 的差别（这决定了 kernel 形态）：

| 维度 | 标准 attention | MQA logits（indexer） |
|---|---|---|
| Q 形状 | `[seq, heads, dim]` 多头 | `[seq, heads, dim]` 多头，但 **K/V 单头** |
| 输出 | softmax 概率 + weighted V | **标量分数**（无 softmax） |
| 归约 | online-softmax（running max/sum） | 直接 `sum_h relu(·)·w`，**无 rescale 状态** |
| 精度 | fp16/bf16 常见 | Q/KV **FP8**，每 KV token 一个 **fp32 scale**（SM90） |
| 实现 | FlashAttention 风格 | 一次 WGMMA + warp 内两级归约 |

## 2. host 契约与两条入口（`csrc/apis/attention.hpp`）

### 2.1 共享的 Q / weights 检查

`check_mqa_logits_q_and_weights`（`:18-44`）是连续/分页共用的前置检查，几个「为什么」：

- `:22-24` `is_fp4 = q.dtype == kPackedFP4`、`is_mx_sf = q_sf.has_value()`；
  **FP4 必须配 MX scale**（FP4 block scaling 不会省）。
- `:27` `head_dim ∈ {32, 64, 128}`（FP4 时不含 32：半字节打包后 32 会退化成 16）。
- `:32` **`is_mx_sf == (arch_major == 10)`**：这是 SM90 与 SM100 最本质的差异——
  SM90 的 Q 用**逐 token fp32 scale**（非 MX），SM100 用 **packed UE8M0 MX scale**（`:36` int32）。
- `:40-43` weights `[seq_len, num_heads]`：`stride(1)==1` 且 **行 stride 按 16 字节对齐**
  （`:42`），因为权重行是 **TMA 直接载入** 的；dtype 由 arch 决定
  （`:43`，SM90=`float`，SM100=`bfloat16`）。测试里 `to_mqa_weights`（`test_attention.py:29-36`）
  就是把 `num_heads` 向上取整到 16 字节边界再补零。

### 2.2 连续入口 `fp8_fp4_mqa_logits`（`:67-134`）

- `:74-75` 输入是 `q=(q_fp,q_sf)`、`kv=(kv_fp,kv_sf)` 两个 tuple。
- `:80` **`out_dtype`：SM100=bf16，SM90=fp32**。
- `:83-96` 形状：`q_fp` 逻辑 `[seq_len, num_heads, head_dim]`，`kv_fp` 逻辑 `[seq_len_kv, head_dim]`。
  `:92-96` **`kv_sf` 的 dtype 由 `is_mx_sf` 决定**：SM90 是 `float`（每 token 一个），
  SM100 是 `int32`（MX 打包）。
- `:99-106` `cu_seq_len_k_start/end` 都是长度 `seq_len` 的 int32，描述每行的有效 KV 区间。
- `:110-117` 输出分配：`block_q = get_mqa_logits_block_q(num_heads)`、`block_kv`（SM100=384，SM90=256）。
  **`stride_logits` 按 1024 字节对齐**（`:114-115`），这是 TMA 写回的对齐要求。
- `:124-129` SM90 分派：`qk_dtype == float8_e4m3fn`、**`num_heads == 32 or 64`**、无 `schedule_meta`。

> 一个关键点（`:110` + `sm90_fp8_mqa_logits.cuh:44`）：
> `get_mqa_logits_block_q(num_heads) = 128 / num_heads`，于是
> `BLOCK_Q * num_heads == 128` 恒成立。**WGMMA 的 M 维被钉死在 128**，靠调 `block_q`
> 适配头数：64 头 → `block_q=2`，32 头 → `block_q=4`。即「一个 block 处理 128 个 (token,head) 行」。

### 2.3 分页入口 `fp8_fp4_paged_mqa_logits`（`:174-282`）

用于 decode：KV 以 **block table 分页**的形式存在。

- `:200-207` `fused_kv_cache` 是 `[num_kv_blocks, block_kv, 1, head_dim + sf_bytes]` 的
  **融合布局**——KV 值和它的 scale 拼在同一行。
- `:212-223` **用 `from_blob` 把一块连续内存切出 `kv_cache` 与 `kv_cache_sf` 两个视图**：
  `kv_cache_sf` 的基址是 `data_ptr + block_kv*head_dim`，也就是每 token 的 scale 紧跟在
  `head_dim` 个 e4m3 之后。**SM90 的 `sf_bytes = sizeof(float)`**（`:204`）。
- `:226-229` `block_table [batch_size, max_block_len]` 给出每个序列的物理块编号。
- `:250-255` `context_lens` 是 `[batch_size, next_n]` 的 2D 张量（每个 query token 各自长度）。
- `:258-264` 输出 `[batch_size*next_n, max_context_len]`，同样 1024 字节对齐。
- `:272-277` SM90 分派：`block_kv == 64`（`attention.hpp:165`）、`next_n==1 or 2`、
  `indices` 必须是 None（**SM90 不支持 varlen**，SM100 才把 varlen 请求摊平成 1D）。

### 2.4 SM90 vs SM100 差异一览

| 项 | SM90（H100，本机可跑） | SM100（目标卡，仅分析） |
|---|---|---|
| Q dtype | E4M3 + **逐 token fp32 scale（非 MX）** | MXFP4/MXFP8 + **packed UE8M0 MX scale** |
| KV scale | 每 token **1 个 fp32**（整条 head_dim 共享） | MX 打包 int32，granularity 32 |
| 输出 dtype | **fp32** | **bf16** |
| num_heads | **仅 32 / 64** | 8/12/16/20/32/64（连续） |
| block_kv / split | block_kv=256（连续）；split_kv=256（分页） | split_kv=384 |
| 调度 | 连续：persistent round-robin；分页：`SM90PagedMQALogitsScheduler` | `schedule_meta`（先跑 metadata kernel 做负载均衡） |
| varlen | 不支持 | 支持（`indices`） |
| sparse（top-k 块） | 无 | `fp8_fp4_sparse_mqa_logits`（仅 SM100） |

`get_mqa_logits_metadata`（`:46-65`）只服务 SM100：它先跑一个 metadata kernel，
产出「每 SM 从哪个 (Q block, KV split) 开始」的均衡调度表；SM90 连续版不需要，
用 `block_q_idx += num_sms` 的 persistent 轮转即可（见下）。

## 3. SM90 连续 kernel 精读（`sm90_fp8_mqa_logits.cuh`）

模板参数（`:22-27`）：`kNumHeads, kHeadDim, BLOCK_Q, BLOCK_KV, kNumQStages, kNumKVStages, kNumSMs, 线程数, logits_dtype`。
host 侧固定（`sm90_fp8_mqa_logits.hpp:21-23`）：`num_specialized_threads=128`（TMA warp）、
`num_q_stages=num_kv_stages=3`、`num_math_threads=512`（4 个 warpgroup）。
`BLOCK_KV = num_math_threads/2 = 256`（`:248` 的静态断言），所以**连续版 KV 块固定 256**。

### 3.1 warp 分工与 barrier

- `:48-55` 只有 TMA warp（`threadIdx.x/32 == kNumMathThreads/32`）的 elect-one 线程
  `prefetch_tma_descriptor` 四个描述符。
- `:89-113` barrier 初始化：`full_q/full_kv` 计数 **1**（单个 TMA 生产者），
  `empty_q/empty_kv` 计数 **kNumMathThreads**（512 个消费者各自 arrive）。这与正式 GEMM 的
  「多 stage 生产者/消费者」约定一致。
- `:116-117` 寄存器两档：TMA warp **32** 个、math warp **112** 个（TMA 只需发地址，
  math 需要累加器 + 权重 + 归约临时）。
- `:97` `is_tma_load_warp`：**只有第一个 TMA warp 真正发 TMA**（`kNumMathThreads <= tid < kNumMathThreads+32`），
  其余 TMA 线程在 `:162-163` 直接 `return`。因为连续版只有一个 Q 流、一个 KV 流，不需要多组。

### 3.2 调度与有效区间聚合（`:119-152`）

- `:120-124` persistent 调度：`block_q_idx = sm_idx; block_q_idx += kNumSMs`，一轮处理一个 Q block。
- `:126-143` `load_schedule` 对 block 内所有 query 行求**有效 KV 区间的并集**
  `[start, end)`，`end` 取 `min(seq_k_end, seq_len_kv)`。`start = start/4*4`（`:139`）
  是因为 **KV scale 的 TMA 描述符要求起始地址 4 对齐**（scale 为 fp32，4 字节）。
  任务量 = `ceil_div(end-start, BLOCK_KV)` 个 KV 块。
- `:145-152` KV 流水用**全局递增的 `num_total_kv_blocks`** 取 stage/phase，
  这样跨 Q block 的 KV 流是连续的环形 buffer，不会每个 Q block 都重置相位。

### 3.3 PDL：与上一个 kernel 重叠（`:154-155`）

`cudaGridDependencySynchronize()` 是 **Programmatic Dependent Launch** 的同步点：
本 kernel 可以随上一个 kernel（如量化/DeepEP 输出）一起 launch，先做 TMA prefetch，
等上游写完再继续。这与 MoE backward/einsum 系列里的 PDL 用法一致——DeepGEMM 把
「kernel 间空隙」也吃掉了。

### 3.4 TMA 生产者（`:157-205`）

- `:166-170` `issue_tma_q`：一次载入**整个 Q block + 对应 weights**（Q 的 `[BLOCK_Q*kNumHeads, kHeadDim]`
  和 weights 的 `[BLOCK_Q, kNumHeads]`），`arrive_and_expect_tx` 一次性登记 Q+weights 字节数。
- `:176-204` 主循环：先 prefetch 下一个 Q（`load_schedule(1)` 拿到下一 block 的 stage/phase），
  `empty_q_barriers` 等待消费者释放；再为当前 block 的每个 KV 块发 TMA KV + KV scale。
  Q 是「每 block 载入一次」，KV 是「每块载入」。

### 3.5 Math warpgroup：WGMMA + 两级归约（`:206-321`）

- `:215` `float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads/4]`。累加器覆盖
  `[BLOCK_Q*kNumHeads, BLOCK_KV]` 的 tile（WGMMA M=128，N=`BLOCK_KV`=256）。
- `:228-233` 从 smem 读出每行每头的权重，按 `(j/2)*8 + (j&1) + (lane%4)*2` 的**交错下标**
  取——这是为了让 WGMMA 累加器里同 warp 的每 2 个头对下降到一个 lane，配合后面的归约。
- `:250-263` WGMMA 内层：`desc_a` 指向 **KV**（A 操作数）、`desc_b` 指向 **Q**（B 操作数），
  `[BLOCK_KV, kHeadDim] @ [kHeadDim, BLOCK_Q*kNumHeads] -> [BLOCK_KV, BLOCK_Q*kNumHeads]`。
  **Q 在 B 侧、KV 在 A 侧** 是这里的关键：因为 MQA 下 KV 是共享的单头，让它当「行」、
  Q（多头的 128 行）当「列」，正好把每头的点积排成累加器的列。
- `:273-310` epilogue **不用 TMA、直接寄存器归约 + 全局写**：
  1. `:282-293` **层内归约**：`fmaxf(·,0)`（ReLU）乘权重后，对每头的那部分累加器求和，
     最后 `sum[0]+sum[1]`、`sum[2]+sum[3]` 得到两个 KV 位置的部分和
     （`v_0_offset=lane/4+0`、`v_1_offset=lane/4+8`，即每 lane 负责 KV 维上隔 4 的两个位置）。
  2. `:294-295` **乘 KV 的逐 token scale**（fp32），这正是「KV scale 在归约之后作用」的体现。
  3. `:298-303` **跨 lane 归约**：两次 `__shfl_xor_sync`（offset 1、2）把 4 个 lane 的
     部分和合并——因为 4 个 lane 覆盖了同一 KV 位置的 4 个不同 head 组。
  4. `:306-310` 写回：地址含 `- seq_k_start[i]` 的**压缩偏移**（每行有效区间从列 0 开始），
     并且只写落在 `[start, end)` 内的位置。`stride_logits` 保证行对齐。

> 与标准 attention 对比：这里**没有 online-softmax**，`accum` 只做一次 WGMMA 就被消费，
> 归约完全在寄存器/shfl 里完成。这就是 indexer 打分便宜的原因：一次 GEMM，无迭代 rescale。

## 4. SM90 分页 kernel 精读（`sm90_fp8_paged_mqa_logits.cuh`）

分页版把 KV 从「连续 `[seq_len_kv, head_dim]`」换成「block table 指向的物理块」，
其余数学相同。host `sm90_fp8_paged_mqa_logits.hpp`：

- `:100` `split_kv = 256`；`:157` `mma_m = 64`；`:158` `num_math_warp_groups = split_kv / mma_m = 4`。
  于是 `num_math_threads = 4*128 = 512`，**每个 math warpgroup 负责一个 64 的 KV 子块**
  （`kernel:53` 静态断言 `SPLIT_KV == BLOCK_KV * kNumMathWarpGroups`）。
- `:165-179` Q/KV/KV-scale/weights 的 TMA 描述符，KV 走 **3D TMA**（第三维是物理块号）。
- `:192` `next_n == 1 or 2`。
- `:194` `128 % num_heads == 0`（与连续版同一约束）。

kernel 侧（`sm90_fp8_paged_mqa_logits.cuh`）：

- `:39` **`DG_STATIC_ASSERT(not kIsVarlen, ...)`**：SM90 分页不支持 varlen。
- `:92-94` warp 角色：`kv_group_idx` 把每个 math warpgroup 与一个 TMA warp 配对
  （`kv_group_idx = (tid>=kNumMathThreads) ? (tid-kNumMathThreads)/32 : warpgroup_idx`），
  **每个 KV 组有独立的 KV smem 流水与 barrier**（`:96-106`），Q 流水则全局共享（`:88-90`）。
- `:137-139` 调度器 `SM90PagedMQALogitsScheduler`：按 (q_idx, kv_split_idx) 分配任务，
  读 `context_lens`/`schedule_meta`/`indices`。
- `:157-163` `issue_tma_q` 只在 `kv_group_idx==0` 时发（Q 全局一次）。
- `:191-198` **block table 的读法**：用一次 warp 内 `__shfl_sync` 广播一个物理块号
  （`kv_block_idx_storage`），每 lane 取一个；这样 `block_table` 的读取不需要每 lane 单独访存。
- `:205-211` TMA KV 用 3D 描述符的第三维参数 `kv_block_idx` 定位物理块，KV scale 同理。
- `:255` `kv_offset` 预先把 `q_idx` 的 `next_n`、`kv_group_idx` 的块偏移算好。
- `:300-329` 归约/存储与连续版一致，**唯一区别是写回不做 `- seq_k_start` 压缩偏移**
  （分页输出本身就是 `[q_token, aligned_context_len]` 的稠密行，越界列由调用方 mask）。

## 5. H100 实测（H100 80GB HBM3，132 SM，CUDA 13 / torch 2.12）

脚本 `harness/exp_mqa_logits.py`、输出 `harness/exp_mqa_logits.out.txt`（原始数字）。
反量化参考用「先量化 Q/KV 再反量化」的模拟参考（`sim`），以及全精度 `fp32` 参考。

### 5.1 连续路径语义（`fp8_fp4_mqa_logits`）

| SQ | SKV | H | D | CP | vs_ref(diff) | vs_sim(diff) | -inf mask 一致 | relerr(sim) |
|---|---|---|---|---|---|---|---|---|
| 2048 | 8192 | 64 | 128 | 1 | 3.29e-2 | 1.64e-3 | True | 3.33e-2 |
| 2048 | 8192 | 64 | 128 | 0 | 3.72e-2 | 2.28e-3 | True | 3.75e-2 |
| 8192 | 65536 | 64 | 128 | 1 | 3.78e-2 | 1.78e-3 | True | 3.95e-2 |
| 2048 | 8192 | 32 | 128 | 1 | 4.58e-2 | 1.96e-3 | True | 4.62e-2 |
| 2048 | 8192 | 64 | 64 | 1 | 3.72e-2 | 1.99e-3 | True | 3.75e-2 |
| 2048 | 8192 | 64 | 32 | 1 | 3.17e-2 | 1.62e-3 | True | 3.12e-2 |

- `vs_sim`（对「量化后再反量化」的参考，隔离 FP8 量化误差）只有 **~2e-3**，说明
  **kernel 的 WGMMA + 归约实现是正确的**，剩余误差来自 fp32 归约顺序 / ReLU 前后舍入；
- `vs_ref`（对未量化的 fp32 参考）~3–5e-2，这就是 **FP8 E4M3 量化本身的误差**；
- **越界掩码逐位一致**（`-inf mask=True`），且重复 20 次输出**逐位相等**（确定性）；
- `num_heads=32/64`、`head_dim=32/64/128` 全部通过。

### 5.2 连续路径性能

有效计算量按 `2 * cost * H * D`（cost = 有效 token 数）。

| SQ | SKV | H | D | us | TFLOPS | GB/s | relu/cyc/SM |
|---|---|---|---|---|---|---|---|
| 2048 | 8192 | 64 | 128 | 483.8 | **497.1** | 159 | 7.7 |
| 8192 | 65536 | 64 | 128 | 17213.0 | 479.1 | 121 | 7.5 |
| 2048 | 65536 | 64 | 128 | 4520.4 | 478.9 | 123 | 7.5 |
| 8192 | 65536 | 32 | 128 | 9446.7 | 436.5 | 218 | 6.8 |
| 2048 | 8192 | 64 | 64 | 380.9 | 315.7 | 179 | 9.8 |

FP8 WGMMA 在 H100 上 ~480–500 TFLOPS，接近 H100 FP8 密集峰值的一半量级；
`D=64` 掉到 ~316（收缩维变浅，WGMMA 效率下降）。注意这只是「打分」的算力，
真正的稀疏注意力收益来自**后续只对 top-k token 做完整 attention**。

### 5.3 分页路径

| BSZ | nextN | H | D | block_kv | avgL | vs_ref(diff) | vs_sim(diff) | us | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 2 | 64 | 128 | 64 | 4096 | 3.68e-2 | 2.21e-3 | 8.6 | 368.4 |
| 64 | 1 | 32 | 128 | 64 | 4096 | 3.15e-2 | 2.39e-3 | 9.9 | 227.6 |
| 32 | 2 | 64 | 64 | 64 | 8192 | 4.13e-2 | 2.89e-3 | 9.7 | 323.5 |

分页版 `vs_sim` 同样 ~2e-3，语义正确；吞吐 ~230–370 TFLOPS（分页 KV 读 + 块寻址有一定开销）。

### 5.4 契约检查（SM90 全部按预期被拒）

| 非法输入 | 断言位置 | 结果 |
|---|---|---|
| 传 `schedule_meta` | `attention.hpp:125` | 拒绝 |
| 传 MX `q_sf` | `attention.hpp:32` | 拒绝 |
| `num_heads=48` | `attention.hpp:127` | 拒绝 |
| `kv_sf` 用 int32 | `attention.hpp:96` | 拒绝 |
| `weights` 用 bf16 | `attention.hpp:43` | 拒绝 |
| 分页 `block_kv=128` | `attention.hpp:165` | 拒绝 |

## 6. 与 MoE 专题的关系 / 小结

- **同源基础设施**：MQA logits 与 Mega MoE / grouped GEMM 共用 DeepJIT、TMA、
  WGMMA（SM90）/ tcgen05 UMMA（SM100）、PDL、mbarrier 流水。读它的 kernel 可以
  反向印证前面各篇讲的「生产者-消费者 barrier 计数」「TMA 对齐要求」「PDL 吃掉 kernel 间空隙」。
- **它不属于 MoE**：与专家路由、dispatch/combine、grouped GEMM 无数据依赖，是
  DeepSeek 稀疏注意力的 indexer 打分。放在本专题是因为它同属 DeepGEMM 的
  「DeepSeek 关键 primitive 库」。
- **可跑 vs 仅分析**：H100 上连续 / 分页两条 SM90 路径都能跑，本轮的数值与性能是真实数字；
  MXFP4/MXFP8、varlen、`schedule_meta` 负载均衡、sparse（top-k 块）变体都是 **SM100 专属，仅分析**。
- **SM90 分页的 `next_n<=2` 与 `block_kv==64`**、**连续版 `num_heads∈{32,64}`** 是当前实现的
  硬约束，部署时若 head 数不满足需走别的路径。

### 引用

- DeepGEMM README `#### V3.2 MQA kernels for the indexer`（`:88-110`）：
  <https://github.com/deepseek-ai/DeepGEMM>
- DeepGEMM release note `2025.09.28: weighted ReLU MQA logits for the lightning indexer for DeepSeek v3.2`（README `:19`）。
