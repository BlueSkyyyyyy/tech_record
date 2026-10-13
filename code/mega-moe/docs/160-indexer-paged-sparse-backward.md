# 分页稀疏 MQA logits（lightning indexer）前向的反向：页表逆映射 × 支持集散播（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的第 160 篇（P150 增量）。
> 第 157 篇（P147-1）拆了 lightning indexer 的**连续稠密**训练反向、第 158 篇（P148-1）拆了
> **分页稠密**反向、第 159 篇（P149-1）拆了**连续稀疏（块选择）**反向。indexer 反向的
> 2×2 矩阵（连续/分页 × 稠密/稀疏）只剩最后一格——**分页 × 稀疏**，而它正是生产 long-context
> 训练真正跑的形态：KV 住在物理页池里（block table 间接寻址）、只用 indexer 选中的一小撮 KV 块
> 打分、输出**压缩**成 `[T, num_max_sparse_blocks·SB]`。DeepGEMM 的分页稀疏前向
> `fp8_fp4_paged_sparse_mqa_logits` 是 **SM100 专属**、且只输出 logits；它的训练反向一直没人讲过。
>
> 本篇补上：① 分页稀疏前向的**支持集 + 页表逆映射**语义（真实行号）；② 反向要**依次**做两级
> 逆映射——`support（选中块 ∩ 有效窗口）→ logical KV token →（页表逆映射）→ physical slot`，
> 再 scatter-accumulate；③ **两层别名**（next_n 前缀重叠 + 跨请求共享物理页）都要求累加而非覆盖；
> ④ 「块选择」天然是 stop-gradient（块由上游打分器选、不可微）。数学 / 页表 / 支持集 / 别名
> **架构无关**，全部在 **H100(sm90)** 真机用 `torch.autograd` + 中心差分对拍，并给一个可跑的
> **CUDA 分页稀疏反向后端**与**真实 SM90 分页稠密 kernel 前向锚**。

源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。论文坐标：DeepSeek-V3.2 技术报告
[arXiv:2512.02556](https://arxiv.org/abs/2512.02556) §2.1。

## 1. 定位：indexer 反向 2×2 矩阵的最后一格

前面三篇把 indexer 反向的两条轴各自补齐：

| | 稠密（整窗口） | 稀疏（块选择 + 压缩输出） |
|---|---|---|
| **连续 KV** | 第 157 篇（P147-1）：`fp8_fp4_mqa_logits` | 第 159 篇（P149-1）：`fp8_fp4_sparse_mqa_logits` |
| **分页 KV** | 第 158 篇（P148-1）：`fp8_fp4_paged_mqa_logits` | **本篇（P150-1）：`fp8_fp4_paged_sparse_mqa_logits`** |

四条前向在 DeepGEMM 里是独立一族 kernel（README `:88`「V3.2 MQA kernels for the indexer」），
**都只输出 logits**，训练反向要自己补。分页稀疏相对另外三条多出来的结构是**两级间接寻址**：

1. **支持集（support）**：输出不是整段窗口，而是**被选中块覆盖的稀疏子集**——
   压缩输出第 `c` 列对应一个 logical KV token，未选中块/越界位置「看不见」→ 梯度恒 0（第 159 篇）；
2. **页表（page table）**：logical KV token 还要经 block table 映射到**物理页池里的槽**——
   反向散播的目标是物理槽，一个物理槽会被多行引用（第 158 篇）。

本篇把二者**组合**，并证明：反向的两级逆映射不会引入新的数学，只是把第 158/159 篇的
「页表逆映射」和「支持集散播」**串联**；但两层别名叠加后，`dkv/dscale` 的 scatter-add
不再只是「更安全」（第 148/159 篇），而是**正确性的硬要求**——实测两层的最大共享度 9。

## 2. 分页稀疏前向语义（真实行号）

### 2.1 四个 host 入口

`csrc/apis/attention.hpp` 里 indexer 家族的 host 入口：

| 入口 | 行号 | arch 门控 |
|---|---|---|
| `fp8_fp4_mqa_logits`（连续稠密） | `:67-134` | SM90/SM100 |
| `fp8_fp4_paged_mqa_logits`（分页稠密） | `:174-282` | SM90/SM100 |
| `get_sparse_mqa_logits_metadata`（连续稀疏元数据） | `:306-332` | **仅 SM100**（`:313`） |
| `get_paged_sparse_mqa_logits_metadata`（分页稀疏元数据） | `:337-367` | **仅 SM100**（`:344`） |
| `fp8_fp4_sparse_mqa_logits`（连续稀疏） | `:370-403` | **仅 SM100**（`:380`） |
| `fp8_fp4_paged_sparse_mqa_logits`（分页稀疏） | `:406-442` | **仅 SM100**（`:414`） |

分页稀疏的两个入口都断言 `arch_major == 10`（`:344`/`:414`），故本机 H100 不可跑。其 host 契约
（`:337-367` 元数据、`:406-442` 主体）把「页表」显式收进来：

- 元数据入口 `get_paged_sparse_mqa_logits_metadata`（`:337-367`）额外吃 `context_lens`、
  `block_table`、`indices`、`page_kv`，并断言 `page_kv % sparse_block_kv == 0`（`:353`）、
  `block_table` 逐行连续（`:351-352`）、`sparse_block_kv ∈ {8,16}`（`:354`）、
  `block_table.stride(0)` 入 uint32（`:352`）；它把这两者一路传进
  `launch_sm100_sparse_mqa_logits_metadata(/*is_paged=*/true, ...)`（`:362-365`）。
- 主体入口 `fp8_fp4_paged_sparse_mqa_logits`（`:406-442`）吃的是 `fused_kv_cache`
  （`[num_kv_pages, page_kv, 1, head_dim(+sf)]` uint8，`:425-434` 断言页 stride 512 对齐、页基址
  16 对齐），按 `(is_fp4 ? head_dim/2 : head_dim) + sizeof(int)` 切 KV 与 SF；输出
  `num_output_tokens = num_max_sparse_blocks * sparse_block_kv` 列的 bf16 logits（`:438-439`）。

### 2.2 支持集规范（测试侧唯一正确性规范）

分页稀疏的「算什么对」由 `tests/test_attention.py:442-672` 定义。支持集与压缩展开逐字在
`:655-668`：

```
kv_offsets      = arange(sparse_block_kv)                                   # :656
block_offsets   = context_starts % sparse_block_kv                          # :657
token_indices   = sparse_indices[...,None]*sparse_block_kv                  # :658-659
                  + block_offsets[:,None,None] + kv_offsets                 #   -> logical KV token
num_sparse_tokens = num_sparse_blocks * sparse_block_kv                     # :661
valid_mask      = (col < num_sparse_tokens)                                 # :662
valid_mask     &= token_indices >= context_starts                           # :663
valid_mask     &= token_indices < selected_ctx_len                          # :664
sparse_logits   = sparse_logits[valid_mask]                                 # :665
full_logits     = ref_logits.gather(1, token_indices - context_starts)[mask] # :666-668
assert_bitwise_equal(sparse_logits, full_logits)                            # :669
```

对分页用例，测试把 `context_starts` 置 0、并把 `sparse_indices` 当作**请求内的 logical 稀疏块号**
（`make_sparse_kv_block_indices:442-475` 在 `context_start//sparse_block_kv` 起、`ceil_div(len,SB)`
个块里选；同请求后续 token 保留上一 token 选中块的 ~80%，`:461-469`）。压缩输出第
`c = l*SB + kvj` 列 = 稠密 logits 在**同一 logical token**处的取值——**压缩就是带支持集的选择
（gather）**，未选中块根本不进输出。

分页用例的 block table 是**逐 query 行**的：测试先建 request 级页表
`request_block_table[batch_size, max_pages]`（`test_attention.py:608-612`），再用每 query 行的
`indices` 索引成 `[num_q_tokens, max_pages]`（`:614` `request_block_table[indices.long()]`），
与 DeepGEMM 的 varlen 展平一致（`:233` 断言 `is_varlen == (arch_major == 10)`）。

### 2.3 页表逆映射（physical slot 的来源）

KV 本体在物理页池里，支持集里的是 **logical** token。反向必须把 logical token 翻成 physical slot。
翻法在 metadata kernel `scheduler/sm100_sparse_mqa_logits_metadata.cuh:378-386`（逐字）：

```
kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV                             # :379
logical_page_idx    = logical_kv_block_idx / kNumKVBlocksPerPage            # :380
block_table_idx     = q_token_base * block_table_stride + logical_page_idx  # :382
physical_kv_block_idx = block_table[block_table_idx] * kNumKVBlocksPerPage  # :383
                      + logical_kv_block_idx % kNumKVBlocksPerPage
# 非分页（连续）分支：physical_kv_block_idx = logical_kv_block_idx * (kUseUnalignedKs?1:SPARSE_BLOCK_KV)  # :385
```

`physical_kv_block_idx` 以 `SPARSE_BLOCK_KV` 为单位，乘回 `SB` 加 `token_in_kv_block` 即物理 token。
等价的**逐 token**写法（本实验用的、也是反向直接用得上的）：

```
phys = block_table[q_row, logical_token // PAGE_KV] * PAGE_KV + logical_token % PAGE_KV
```

两者严格等价：`logical_kv_block_idx // (PAGE/SB) = (logical_block*SB + kvj) // PAGE = token // PAGE`
（`kvj < SB` 且 `SB | PAGE`），`(logical_kv_block_idx % (PAGE/SB))*SB + kvj = token % PAGE`。
本实验用 200 个随机点逐点核对，**mismatch=0**（Part B）。注意 `block_table` 按 **q token** 索引
（`:382` 的 `q_token_base * block_table_stride`），这是 varlen 展平的结果。

压缩输出的列布局在主体 kernel `impls/sm100_sparse_mqa_logits.cuh:631`：

```
output_col_idx = (q_slot_base + q_slot_offset) * SPARSE_BLOCK_KV + token_in_kv_block   # :631
```

`SPARSE_BLOCK_KV` 见 `layout/sparse_mqa_logits.cuh`（`KVBlockInfo:63-72` 把
`physical_kv_block_idx` 与两个 `q*_slot_offset` 打包）。

## 3. 反向 = 支持集散播 × 页表逆映射

把分页稀疏前向看成两个算子的复合：

```
I_paged[r, s]   = 打分(q_r, KV_phys(s))                 # 分页打分（第 158 篇）
G[r, c]         = I_paged[r, logical_token(c)]  if c ∈ support(r)   # 支持集选择（第 159 篇）
```

`G` 是下游损失（对主注意力分布的 KL 蒸馏，arXiv §2.1）的输入。反向因此是**两个伴随的复合**：

1. **支持集选择（gather）的伴随 = scatter**：`dI_paged[r, logical_token(c)] += dG[r, c]`，
   未选中/越界位置梯度恒 0（第 159 篇）；
2. **页表逆映射**：`dI_paged` 落在 logical token，但 KV 的存储是物理池 → 散播目标必须是
   `phys(logical_token)`（§2.3）。

合起来，四路梯度（`dI_s[r,c]` 是支持集内 KL 梯度 `softmax_support(I) - p`）：

```
dz_{r,s,h}        = dI_s[r,c(s)] · w_{r,h} · 1[z > 0]
dL/dw_{r,h}       = Σ_{s∈support(r)} dI_s[r,c(s)] · relu(z_{r,s,h})
dL/dq_{r,h,d}     = Σ_{s∈support(r)} dz_{r,s,h} · kv_phys(s),d · scale_phys(s)
dL/dkv_phys[phys]+= Σ_{r,h,s : phys(s)=phys} dz_{r,s,h} · q_{r,h,d} · scale_phys(s)      (scatter-add)
dL/dscale[phys]  += Σ_{r,h,s : phys(s)=phys} dz_{r,s,h} · (q_{r,h}·kv_phys(s))           (scatter-add)
```

四点值得点明：

- **两级逆映射，仍是一套数学**：`support` 只决定「哪些 (r,c) 参与」，`页表` 只决定「累加到哪个
  物理槽」；`dq/dw` 与前三条（连续/分页/稀疏）**同构**。真正变的是 `dkv/dscale` 的**落地地址**。
- **两层别名都必须 scatter-add**：
  - next_n 前缀重叠：同一请求的多个 q token 命中同一批块（测试里 `sparse_indices` 请求级共享）；
  - 跨请求共享物理页：prefix sharing / beam search，或页池随机分配后的碰撞；
  一个物理槽因此被更多 `(r,h)` 累加。用「写」（last-writer-wins）会静默丢贡献。
- **「选中哪些块」是 stop-gradient**：块索引由上游（一个独立打分器 / top-k）给出，反向对它没有
  梯度；梯度只在**选中块内、且落在有效窗口内**的 token 上。这在分页/连续稀疏里已成立，分页稀疏
  只是多套一层页表。
- **不确定的物理槽数**：分页稀疏的 `dkv` 目标空间是**页池**（`num_pages × PAGE_KV`），不是请求
  的 logical 窗口；页池被多请求共享时，`dscale` 的原子加争用比分页稠密更高。

## 4. H100(sm90) 真机实测

实验代码：`src/mqa_indexer_backward_paged_sparse.cu`（CUDA 反向后端，架构无关：
fp32 + `__shfl_xor` 归约 + 原子加）、`harness/exp_mqa_logits_paged_sparse_backward.py`
（纯 torch fp64 对拍 autograd + 中心差分 + 抽取官方 `tests/test_attention.py` 参考函数）、
driver `harness/exp_mqa_logits_paged_sparse_backward.sh`。原始输出
`harness/exp_mqa_logits_paged_sparse_backward.out.txt`（内核 sha256 `1e929dbf…`）。
环境：H100 80GB HBM3、CUDA 13.2、PyTorch 2.12。

### 4.1 CUDA 后端 [A] 正确性 + [B] 吞吐

CUDA 后端 `grid = P`（每 query 行一 block）、`block = (32, H)`：每 `h` 一个 warp，沿 lane 对 `d`
跨步点积 + `__shfl_xor` 归约；`dq` 累寄存器；`dkv`/`dscale` 沿**物理槽**走 global `atomicAdd`。

- **[A] 正确性**（`S=5,N=3,H=8,D=64,SB=16,PAGE=64`，页池 + 两层别名）vs host fp64 逐元素参考：
  `dq` **5.81e-08** / `dw` **6.90e-08** / `dkv` **5.06e-08** / `dkv_scale` **5.75e-08**
  → **ALL PASS**。对照：若用「写」而非 scatter-accumulate，`dkv` 与正确值 **relerr=9.22e-01**。
- **[B] 吞吐**（`~2.97–5.87 GFLOP/s`，本机 8 卡共享、**受争用下界**）——瓶颈与前三条一致，仍是
  `dkv` 跨 `(r,h)` 的 scatter 原子归约；分页只把目标地址从连续下标换成页表逆映射 + 物理槽聚合。

### 4.2 Python 语义对拍（Parts A–F，**PASS=17 / FAIL=0**）

- **Part A 支持集 = 选择语义**：可微分分页稀疏「支持集 logits」== 官方 `ref_paged_mqa_logits`
  （`tests/test_attention.py:225-253`，AST 原样抽取）在选中 logical 位置的取值，relerr
  **1.571e-08**（= ref 输出本身是 fp32 的舍入地板）；支持集是窗口的**严格子集**
  （640 项 / 7040）。
- **Part B 页表逆映射恒等式**：逐 token 版 == metadata kernel 按块版，**200 随机点 mismatch=0**；
  物理槽全部落在页池范围内。
- **Part C 支持集内 KL 梯度**：`dL/dI = softmax_support(I) - p` vs autograd **2.847e-16**；
  行和 = 0（`2.9e-16`）；**支持集外梯度恒 0**。
- **Part D 打分算子反向**（两级逆映射 + 物理池 scatter-add）vs autograd：
  `dq` **0.000e+00** / `dw` **0.000e+00** / `dkv` **5.70e-18** / `dscale` **7.84e-17**；
  `dkv` 中心差分 **0.000e+00**。
- **Part E 两层别名**：960 个 `(r,c)` 支持引用 → **240 个物理槽**，**全部被多引用、最大共享度 9**；
  naive「写」在被共享槽上最大绝对差 **2.477**、全局 relerr **8.934e-01**，而 scatter-add 与
  autograd **2.52e-17** ——坐实两层别名下「写」必错。
- **Part F 真实 SM90 分页稠密 kernel 前向锚**：用真实 `fp8_fp4_paged_mqa_logits`（SM90 分支）
  算出稠密 logits，再按支持集 gather 与「反量化 q/kv 的模拟参考」比较，relerr **1.176e-03**
  （= bf16 输出舍入 + FP8 量化层），坐实「分页稀疏压缩 = 对分页稠密 logits 的支持集选择」。

### 4.3 与前三篇的坐标对照

| 维度 | 第 157 连续稠密 | 第 158 分页稠密 | 第 159 连续稀疏 | **第 160 分页稀疏** |
|---|---|---|---|---|
| logical 窗口 | 连续 `[ks,ke)` | 连续 `[0,cl)` | 选中块 ∩ 窗口 | 选中块 ∩ 窗口 |
| 输出 | 整窗口 | 整窗口 | 压缩（选择） | 压缩（选择） |
| KV 存储 | 连续张量 | 物理页池 | 连续张量 | 物理页池 |
| `dkv` 目标 | logical 下标 | 页表逆映射 | logical 下标 | **支持集 → 页表逆映射** |
| 别名来源 | next_n 前缀 | next_n + 跨请求共享页 | next_n 共享块 | **next_n + 跨请求共享页** |
| `dkv` 散播量 | `O(Σcl_r)` | `O(Σcl_r)` | `O(Σ selected_r)` | `O(Σ selected_r)` |

四篇合起来，indexer 反向的 2×2 矩阵闭合。

## 5. 结论

- **分页稀疏反向 = 第 158 篇的页表逆映射 ∘ 第 159 篇的支持集散播**，数学完全复用：`dq/dw` 同构，
  `dkv/dscale` 只是「落地地址」变成「支持集 logical token → 页表 → 物理槽」。
- **两层别名叠加使 scatter-add 成为正确性硬要求**：实测支持集里每个物理槽被多引用（最大共享度
  9），用「写」全局 relerr 0.89。
- **主 kernel 与四个 host 入口 SM100 专属、仅分析**（本机 H100 只跑其**架构无关**的语义：支持集
  规范、页表逆映射恒等式、四路梯度、别名累加、真实 SM90 分页稠密前向锚）；生产形态的完整
  数值要在目标卡上跑 `fp8_fp4_paged_sparse_mqa_logits` 才能给出。
- **边界**：Mega MoE 融合 kernel（UMMA/tcgen05/TMEM/NVLink 直写）与 DeepEP 真实设备路径
  （NCCL GIN，本机 NCCL 2.29.7 不匹配，见 ROADMAP「阻塞」）仍为 **SM100 专属 / 仅分析**；
  本篇复刻的是 indexer 反向的**语义**，不复刻 SM 内重叠。

## 参考

- DeepGEMM（公开版 `057ca59`）：`csrc/apis/attention.hpp`、`tests/test_attention.py`、
  `deep_gemm/include/deep_gemm/{scheduler/sm100_sparse_mqa_logits_metadata.cuh,
  impls/sm100_sparse_mqa_logits.cuh, layout/sparse_mqa_logits.cuh}`。
- DeepSeek-V3.2 技术报告：<https://arxiv.org/abs/2512.02556>（lightning indexer，§2.1）。
- 本专题：`docs/157-mqa-logits-indexer-backward.md`（连续稠密）、
  `docs/158-indexer-paged-backward.md`（分页稠密）、`docs/159-indexer-sparse-backward.md`
  （连续稀疏）、`docs/19-attention-mqa-logits.md`（前向）。
