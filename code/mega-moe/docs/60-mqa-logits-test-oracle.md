# MQA logits 的端到端验收测试与正确性 oracle（`tests/test_attention.py`）

> 第 19 篇（P9-1）拆过 DeepSeek 稀疏注意力 indexer 第一级打分的**主 kernel**
> （`fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`，SM90 有实现）；第 30 篇（P20-1）
> 拆过 SM90 分页调度器；第 22/23/24 篇拆过 SM100 的 sparse / dense / paged 调度器。
> 但**「MQA logits 算什么对」的唯一规范**——`tests/test_attention.py`（715 行）——一直没被单独拆过。
> 它也是第 19 篇里那些 TFLOPS / GB/s / relu-per-cyc 数字的共同出处。
>
> 与 Mega MoE（第 51 篇）、Mega Gate（第 57 篇）不同：MQA logits 的 **SM90 用例
> （`fmt='fp8'`、fp32 logits）在 H100 上真的能跑**——只有 SM100 的 mxfp4/mxfp8 与 sparse 用例
> 被 `@test_filter` 跳过。所以本篇既能精读「正确性 oracle」，又能**直接在 H100 上把真实测试跑通**。

---

## 1. 测试全貌：三个入口、两层架构分支

`tests/test_attention.py` 有三个用例函数：

| 用例 | 行号 | H100(arch9) 是否执行 | 验证对象 |
|------|------|----------------------|----------|
| `test_mqa_logits` | `:104-222` | **执行**（`fmt='fp8'`） | 连续 KV 的加权 ReLU MQA 打分 |
| `test_paged_mqa_logits` | `:299-439` | **执行**（`fmt='fp8'`） | 分页 KV cache 的同一打分 |
| `test_sparse_mqa_logits` | `:476-704` | **跳过**（`@test_filter(get_arch_major()==10)` `:476`） | 稀疏块内的二级打分（第 22 篇） |

两种架构的用例由三个 helper 共同决定：

- `mqa_logits_formats()`（`:45-48`）：arch10 → `[('mxfp4', bf16), ('mxfp8', bf16)]`；
  arch9 → `[('fp8', float)]`。**SM90 的 logits 是 fp32，SM100 是 bf16**。
- `mqa_logits_heads(is_mxfp4)`（`:51-56`）：arch10 heads `(8,12,16,20,32,64)`、
  head_dim `(64,128)`（mxfp4）或 `(32,64,128)`（mxfp8）；arch9 heads `(32,64)`、
  head_dim `(32,64,128)`。
- `quantize_mqa_q(q, fmt)`（`:59-70`）：fp8 走最简单的 `q.to(torch.float8_e4m3fn)`、
  该 token 无 scale（返回 `(q_fp, None)`）；mxfp4/mxfp8 走 `per_token_cast_to_fp4/fp8`
  （`gran_k=32`、packed UE8M0，第 27 篇的参考量化器）。

H100 实测（`harness/exp_mqa_logits_test.out.txt` §A）：`arch_major=9`，
`mqa_logits_formats()==[('fp8', float32)]`、heads `(32,64)`、head_dims `(32,64,128)` 全部印证。

---

## 2. 连续版 oracle：`ref_mqa_logits`（`:78-101`）

### 2.1 语义

```python
w = weights.transpose(0, 1).contiguous()          # [M,H] -> [H,M]（w[h,m]）
for n0 in range(0, seq_len_kv, kv_chunk):          # 沿 KV 分块，限制 score 临时张量大小
    score = torch.einsum('mhd,nd->hmn', q, k[n0:n1])          # [H, M, chunk]
    chunk_logits = torch.einsum('hmn,hm->mn', score.relu(), w)  # 对 head 求和 -> [M, chunk]
    mask = (cols >= ks[:,None]) & (cols < ke[:,None])
    logits[:, n0:n1] = chunk_logits.masked_fill(~mask, -inf)
    cost += mask.sum()
```

即 $out_{ij}=\sum_h \mathrm{relu}(\langle q_{i,h}, kv_j\rangle)\cdot w_{i,h}$，
仅在 $j\in[ks_i, ke_i)$ 有效、其余置 `-inf`；`cost` 是有效位置计数（供 FLOPs 核算）。

这里有两个**容易漏掉**的语义点：

1. **权重按 head 加权、不是对 head 求和后加权**：`score.relu()` 是 `[H,M,chunk]`，
   乘 `w[...,None]` 是 `[H,M,1]`，最后 `sum(dim=0)` 才对 head 求和。顺序不能换——
   先 `sum` 再乘就丢了 per-head 权重。
2. **KV 分块只是显存技巧**：`kv_chunk = max(1, 256MiB // (seq_len·H·4))`（`:88`）。
   分块不应改变结果，这正是 oracle 要保证的。

H100 实测 §B：把 `ref_mqa_logits` 与我们独立写的「三重循环朴素实现」对拍，
有效区间 `calc_diff=1.110e-16`（纯 fp64 舍入），`-inf` 掩码**逐位一致**，
`cost == 有效位置数 == sum(ke-ks) == 878`，且每行恰在 `[ks_i,ke_i)` 内有效。

### 2.2 `generate_ks_ke_tests`（`:106-121`，嵌套）与 CP 分片

测试用两种方式造 `[ks, ke)`：

- `disable_cp=True`：`ks=0`、`ke=arange(seq_len) + (seq_len_kv - seq_len)`——
  每个 query 看「错位着的定长窗口」。
- `disable_cp=False`（context-parallel）：`chunk_size=seq_len//2`、`cp_size=seq_len_kv//seq_len`、
  `cp_id=cp_size//3`。前半 query 的 `ke=cp_id·chunk+i`，后半 `ke=(2·cp_size-1-cp_id)·chunk+i`，
  `ks` 恒 0。即模拟「一个 CP rank 持有首尾两段 KV」。

H100 实测 §C（把该嵌套函数按源码逐行复刻后在 CPU 上验证）：`(2048,8192)` 下
`cp_size=4, cp_id=1`，前半 `ke∈[1024,2048)`、后半 `ke∈[6144,7168)`，与源码公式一致。
这是 CP 训练里「每个 rank 的 KS/KE 边界」的生成规范。

### 2.3 输出是**压缩**的：`[seq_len, max_seqlen_k]`

kernel 只输出有效区间内的 logits，形状是 `(seq_len, max_seqlen_k)`，
`max_seqlen_k = (ke-ks).max()`（`:160-164`）。测试再把它**展开**回稠密
（`:185-189`）：

```python
assert logits.size() == (seq_len, max_seqlen_k)
tmp = torch.full((seq_len, seq_len_kv), -inf)
for i in range(seq_len):
    tmp[i, ks[i]:ke[i]] = logits[i, :ke[i]-ks[i]]
```

kernel 侧的写回偏移就是 `- seq_k_start[i]`（`impls/sm90_fp8_mqa_logits.cuh:307-310`）。
这解释了为什么 `fp8_fp4_mqa_logits` 的输出是「压缩向量」而不是「稠密矩阵」：
indexer 只需保留有效窗口内的打分。

### 2.4 判据：`-inf` 掩码逐位 + 两层 `calc_diff`

```python
assert torch.equal(neginf_mask, ref_neginf_mask)                 # :194 -inf 掩码逐位
diff           = calc_diff(logits, ref_logits)                   # :199 vs 原始 bf16 参考
simulated_diff = calc_diff(logits, simulated_logits)             # :200 vs 隔离量化参考
assert diff < diff_tol                                           # :202
assert simulated_diff < simulated_diff_tol                       # :203
```

`mqa_logits_tolerances(fmt)`（`:74-75`）给出**两层容差**：

| fmt | `diff_tol`（vs 原始） | `simulated_diff_tol`（vs 隔离量化） |
|-----|----------------------|-------------------------------------|
| `fp8` | `1e-3` | `5e-6` |
| mxfp4/mxfp8 | `0.02` | `3e-5` |

这正是第 27 篇「参考量化器」思想的应用：`simulated_logits` 是把「同样的量化」
先在 torch 里做一遍、再在**反量化后的操作数**上算出的参考——它隔离了「量化本身带来的误差」，
所以阈值可以卡到 `5e-6`；而 `ref_logits` 是原始 bf16 输入上的参考，阈值放宽到 `1e-3`。
**两层判据把「kernel 实现误差」与「FP8 量化误差」分开**。

另外每个用例都做**逐位自洽**（`:173-175`）：同一输入连跑 20 次，
用 `assert_bitwise_equal` 检查输出逐位不变——这是第 40 篇讲的「确定性」判据。

---

## 3. 分页版 oracle：`ref_paged_mqa_logits`（`:225-253`）与 KV cache 布局

分页版的参考是**按 sequence 逐个做 matmul**（`:233-251`）：对每个序列 `i`，
把它的 block table 指到的 KV 块拼起来 `[kv_heads, dim, total_tokens]`，
`q[i]`(next_n 个 query) 与之打分、`relu`、乘权重、对 head 求和，
再按 `context_lens_nextn` 掩码。**MQA 意味着 `kv_heads=1`**（KV cache 第 3 维是 1）。

### 3.1 三种 KV cache 融合布局

分页 KV cache 不是裸张量，而是把「量化值 + scale」打成一条融合布局，三种格式：

| helper | 行号 | 元素字节 | 布局 |
|--------|------|----------|------|
| `kv_cache_cast_to_fp8` | `:270-282` | `head_dim + 4` | 每 token `head_dim` 个 E4M3 值，后接 **1 个 fp32 scale** |
| `kv_cache_cast_to_mxfp8` | `:285-296` | `head_dim + 4` | 每 token `head_dim` 个 E4M3 值 + packed UE8M0 scale（4B） |
| `kv_cache_cast_to_mxfp4` | `:256-267` | `head_dim//2 + 4` | FP4 打包（每字节 2 元素）+ packed UE8M0 scale（4B） |

SM90 fp8 的 scale 是**每 token 一个 fp32**（`x_amax/448`，clamp `1e-4`，`:274-276`），
SM100 的 mxfp 是 per-32 的 UE8M0 packed（`per_token_cast_to_fp4/fp8(..., gran_k=32,
use_packed_ue8m0=True)`）。三种布局尾部都留 4 字节放 scale——**这 4 字节的 padding 是 TMA 对齐的需要**。

H100 实测 §F：fp8 布局 `(3,64,1,132)`，把前 `64·128` 字节重新解成 E4M3 值、
后 `64·4` 字节解成 fp32，`值 · scale` 再 cast 回 bf16 **逐位等于** `kv_cache_cast_to_fp8` 的返回；
scale **逐位等于** `amax/448`。mxfp4/mxfp8（纯 torch、架构无关）往返相对误差 `0.1157 / 0.0269`。

### 3.2 `to_mqa_weights` 的 TMA 16B 行 padding（`:29-36`）

```python
element_size = weights.element_size()
stride = ceil_div(weights.size(1) * element_size, 16) * 16 // element_size
storage = torch.empty((weights.size(0), stride), ...)
result = storage[:, :weights.size(1)]          # 是 storage 的窄 view
```

注释 `:30` 说得很直白：**行的字节数补齐到 16 的倍数**，因为权重是按行 TMA load 的
（SM90 fp32 / SM100 bf16）。H100 实测 §D：`h=5` fp32 → `stride=8`（`5·4=20` 不是 16 的倍数，
补到 24 字节 → `/4=6`？实测 stride=8 即 32 字节，公式 `ceil(20/16)=2`、`2·16/4=8`）——验证了公式；
`h∈{32,64}` 恰好已对齐，前 `h` 列逐位等于输入。

---

## 4. 稀疏版 oracle（SM100 专属，仅分析）

`test_sparse_mqa_logits`（`:477-704`）的核心判据是**逐位**的（`:669`）：

```python
assert_bitwise_equal(sparse_logits, full_logits, 'sparse MQA logits')
```

即「在稀疏块内打分」必须**逐位等于**「全量打分后按同样的 token 索引 gather」
（`:655-668` 用 `sparse_indices·SB + block_offset + kv_offset` 造索引，
`full_logits.gather(...)`）。这要求稀疏 kernel 与 dense kernel 的累加顺序完全一致。
`make_sparse_kv_block_indices`（`:442-473`，纯 torch、架构无关）造稀疏块表：
同一请求内**保留上一 token 约 80% 的块**（`:461`，RNG）、其余替换——模拟真实 indexer 的
「块级稀疏在时间上的连续性」。H100 实测 §I：输出形状 `(num_q, num_max_sparse_blocks)`、
每 token 块数 `∈[1,8]`、跨行确有重叠。

---

## 5. 判据工具（第 40 篇）

测试从 `deep_gemm.testing` 借三件套（`:8-13`）：`bench_kineto`（计时）、`calc_diff`
（归一化平方误差）、`assert_bitwise_equal`（逐位 + 首个差异定位）。
H100 实测 §G：相同张量 `assert_bitwise_equal` 通过；改一个元素即抛
`AssertionError`（`num_mismatch=2`、定位到字节）；`calc_diff(x,x)=0`、
小扰动 `4.9e-9`。

---

## 6. H100 实测：真实测试跑通 + oracle 独立验证

`harness/exp_mqa_logits_test.py` 直接 `import` 真实 `tests/test_attention.py` 作为模块，
复用它的 helper，并**调用真实的 `test_mqa_logits()` / `test_paged_mqa_logits()`**
（`DG_MQA_NUM_CASES=2` 限制用例数）。结果 **42 PASS / 0 FAIL**（完整输出见
`harness/exp_mqa_logits_test.out.txt`）：

| 组 | 验证内容 | 结果 |
|----|----------|------|
| A | 模块常量与枚举：formats/heads/tolerances/dtype_tag、`sample_mqa_cases` 的 env 抽样子集与可复现性 | 8 PASS |
| B | `ref_mqa_logits` vs 朴素三重循环（`1.110e-16`）、`-inf` 掩码逐位、`cost==sum(ke-ks)` | 4 PASS |
| C | `generate_ks_ke_tests`（源码复刻）的 disable_cp / CP 分片语义 | 3 PASS |
| D | `to_mqa_weights` 的 16B 行 padding（含 `h=5→stride 8` 非平凡用例） | 7 PASS |
| E | `quantize_mqa_q` fp8：`(q_fp,None)`、`q_sim==q_fp.to(bf16)`、往返 rel `0.0266` | 3 PASS |
| F | KV cache 融合布局：fp8 逐位重构、scale `==amax/448`；mxfp4/8 布局与往返 | 7 PASS |
| G | `assert_bitwise_equal` / `calc_diff` 行为 | 4 PASS |
| H | **真实测试**：`test_mqa_logits`/`test_paged_mqa_logits` 在 H100 上无异常完成，`test_sparse_mqa_logits` 被 arch==10 门控跳过 | 3 PASS |
| I | `make_sparse_kv_block_indices` 语义 | 3 PASS |

真实测试跑出的性能（arch9 fp8，SM90 kernel）：

```
 > Fmt=fp8, DType=FP32, SQ=8192, SK= 8192, H=32, D= 32, CP=1:  100 TFLOPS,  689 us, 209 GB/s, 6.2 relu/cyc/SM
 > Fmt=fp8, DType=FP32, SQ=8192, SK= 8192, H=32, D= 64, CP=1:  191 TFLOPS,  719 us, 212 GB/s, 5.9 relu/cyc/SM
 > Fmt=fp8, DType=FP32, PAGE_KV=64, BSZ=64, NextN=1, H=32, D=64, L=262144:  46 TFLOPS, 1526 us, Metadata=4 us, 800 GB/s
 > Fmt=fp8, DType=FP32, PAGE_KV=64, BSZ=64, NextN=2, H=64, D=64, L= 65536:  122 TFLOPS,  435 us, Metadata=4 us, 708 GB/s
```

注意 `scheduled 0 us`：连续版在 arch9 **没有** `schedule_meta`（`get_mqa_logits_metadata`
只在 arch10 可用，`csrc/apis/attention.hpp:49`），所以连续版的 H100 用例走的是
默认（非 scheduled）路径；分页版的 metadata 走 SM90 自己的调度器（第 30 篇）。

---

## 7. kernel 侧对应速查

| 测试概念 | 行号 | kernel 实现 |
|----------|------|-------------|
| 连续版主 kernel | `test_mqa_logits` `:104` | `impls/sm90_fp8_mqa_logits.cuh`（SM90，第 19 篇）/ `impls/sm100_mqa_logits.cuh`（SM100，仅分析） |
| 分页版主 kernel | `test_paged_mqa_logits` `:299` | `impls/sm90_fp8_paged_mqa_logits.cuh`（SM90）/ `impls/sm100_paged_mqa_logits.cuh`（SM100，仅分析） |
| 压缩输出偏移 `-seq_k_start` | `:185-189` | `sm90_fp8_mqa_logits.cuh:307-310` |
| `schedule_meta`（连续版） | `:178-182` | `get_mqa_logits_metadata` 仅 arch10（`attention.hpp:49`）；SM90 分页有自己调度器（第 30 篇） |
| 分页 metadata | `:391` | `get_paged_mqa_logits_metadata`（`attention.hpp:136-172`） |
| sparse 主 kernel | `:625-650` | `sm100_sparse_mqa_logits` / `sm100_paged_sparse_mqa_logits`（第 22/12 篇，SM100 专属） |
| KV cache 三种融合布局 | `:256-296` | SM90 E4M3+fp32 scale / SM100 MXFP4/MXFP8 + packed UE8M0 |

host 契约（`csrc/apis/attention.hpp`）：`check_mqa_logits_q_and_weights`（`:18-44`）断言
`is_mx_sf == (arch_major == 10)`（`:32`）与 weights dtype `bf16`(arch10) / `float32`(arch9)（`:43`）；
输出 dtype `arch10?bf16:fp32`（`:80/192`）；分页 block_kv 约束 `:201-202`；
varlen 仅 arch10（`:233`）。

---

## 8. 结论与边界

- **MQA logits 的「正确」由 `tests/test_attention.py` 定义**，且是**分层的**：
  `-inf` 掩码逐位、20 次重复逐位自洽、`calc_diff` 对「原始参考」与「隔离量化参考」
  给两层阈值（fp8 `1e-3/5e-6`）。稀疏版更硬——直接 `assert_bitwise_equal` 对齐全量 gather。
- **输出是压缩的** `[seq_len, max_seqlen_k]`，kernel 写回偏移 `-seq_k_start`；
  测试把它展开回稠密 `[seq_len, seq_len_kv]` 再比。
- **oracle 覆盖了三种 KV cache 融合布局**（SM90 E4M3+fp32 scale、SM100 MXFP8、MXFP4+packed UE8M0）
  与 per-token 权重行 TMA 16B padding——这些是 kernel 输入契约的可执行参考。
- **H100 真机 42 PASS / 0 FAIL**：真实 `test_mqa_logits` / `test_paged_mqa_logits`
  在 SM90 上真正执行并通过（`test_sparse_mqa_logits` 被 `@test_filter(arch==10)` 跳过）；
  oracle 的每一项（参考数学、CP 分片、掩码、cost、布局、判据工具）都独立复核过。
- **边界**：SM100 的 mxfp4/mxfp8 用例、sparse 用例与全部 SM100 主 kernel（第 22/23/24 篇）
  本机不可跑，本篇只做源码精读；SM100 的分页调度器与 sparse metadata 语义已在第 22/24 篇
  用「原样抽取设备代码」在 H100 上验证过。
