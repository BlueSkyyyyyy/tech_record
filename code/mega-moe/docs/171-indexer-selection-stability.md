# Lightning indexer 的 top-k KV（块）选择在量化前向下稳不稳（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的第 171 篇（P161 增量）。
> 第 19/22/23/24 篇拆了 indexer 的前向打分（连续 / 分页 / SM100 dense / sparse 调度），
> 第 157~160 篇把**训练反向**（连续 / 分页 / 稀疏 / 分页稀疏）补齐了四路梯度与 KL 蒸馏。
> 但那四篇都把一个前提当成了「给定」：**支持集 `S_t = Top-k(I_t,:)` 本身**。
> 生产里这个 `I_t` 是用 **MXFP4 / MXFP8 的量化打分算子**算出来的（README `:88-110`），
> 于是有一个前向/反向都没问过的问题：**量化把 logits 扰动多少，进而改变 top-k 选择多少？**
> 选错的支持集意味着反向把梯度发给了错误的位置（第 159 篇 §6 已说明「选块不可微」，
> 梯度只在选中集上）。
>
> 本篇在 **H100(sm90)** 上用**真实 SM90 `fp8_fp4_mqa_logits` kernel 锚定**，再以纯 torch 复刻
> FP8 / MXFP8 / MXFP4 三档量化，量化 top-k 选择的稳定性，并给出**正确的诊断量**。
> 主 kernel 的 MXFP4 UMMA 部分 **SM100 专属、仅分析**；量化语义与选择统计**架构无关**。
> 目标卡代号敏感：全文用「目标卡」指代。

参考坐标：DeepGEMM 公开版 commit `057ca59`（v2.8.1）；`csrc/apis/attention.hpp`、
`tests/test_attention.py`、`deep_gemm/utils/math.py`、README `:88-110`。
论文坐标：DeepSeek-V3.2 技术报告 [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) §2.1
（式 (3) dense warm-up / 式 (4) sparse training）。

---

## 1. 定位：前向、反向都有了，「选择」这一环从没量过

DeepSeek Sparse Attention（DSA）的 indexer 三段：

1. **打分**：`I_{t,j} = Σ_h w_{t,h}·relu(<q_{t,h}, kv_j·sf_j>)`（README `:100-108`、
   `tests/test_attention.py:78-101` 的 `ref_mqa_logits`）；
2. **选择**：每 query 从上下文中选 `Top-k(I_t,:)`（论文式 (4)，fine-grained token selection；
   稀疏 kernel 的压缩块路径又把它按 `sparse_block_kv ∈ {8,16}` 聚成块，第 159 篇 §2）；
3. **训练**：只对选中集算 KL 蒸馏（论文式 (3)/(4)）。

第 1 步的量化档在 SM100 生产里是 MXFP4/MXFP8（`csrc/apis/attention.hpp:22-29`：
`is_fp4 = q_fp.scalar_type() == kPackedFP4`、`is_mx_sf == (arch_major == 10)`）；
第 2、3 步的**正确性完全取决于 `I` 的排序**。若量化让排序抖动，训练时算出来的 `S_t`
就和推理时（甚至和另一档精度时）不一致——梯度去到了不同位置。**这一环从没量过。**

本机 H100(sm90) 只能跑最弱的一档（`attention.hpp:126-127`：SM90 只接受 `qk_dtype == e4m3`
且 `num_heads == 32 or 64`，Q 无 MX scale），但「量化 → logits 扰动 → 排序抖动」这条链
和 kernel 架构无关，可以在 H100 上用真实 kernel 做锚、用 torch 复刻三档。

## 2. 三档量化语义（真实行号）

`tests/test_attention.py` 的 `quantize_mqa_q`（`:59-72`）是唯一正确性规范：

| 档 | Q | KV | 说明 |
|---|---|---|---|
| **fp8**（SM90） | **直接** `q.to(float8_e4m3fn)`，`q_sf = None`（`:62`） | 逐 token fp32 scale（`per_custom_dims_cast_to_fp8`） | 512 以下无 MX |
| **mxfp8**（SM100） | 逐 32 元素 e4m3 + **UE8M0**（`per_token_cast_to_fp8`，`gran_k=32`，`:66-68`） | 同左 | |
| **mxfp4**（SM100） | 逐 32 元素 **E2M1** + UE8M0（`per_token_cast_to_fp4`，`:66-68`） | 同左 | `head_dim ∈ {64,128}`（`:54`） |

量化细节（`deep_gemm/utils/math.py`）：e4m3 的 scale `sf = amax/448`（`:36`/`:78`）；
E2M1 的 scale `sf = amax/6.0`（`:109`），E2M1 只有 `{0,±0.5,±1,±1.5,±2,±3,±4,±6}`
八个值（`:84-97` 的 `_quantize_to_fp4_e2m1`，中点 `0.25/0.75/1.25/1.75/2.5/3.5/5.0`，
round-half-to-even）；scale 都过 `ceil_to_ue8m0`（`:13-16`，只保留 2 的幂）。
**注意 fp8 档 Q 是裸 e4m3 无 scale**（`:62`）——靠 `|q|≤448` 的假设。

## 3. 实验设计（H100，真实 kernel + torch 复刻）

`harness/exp_indexer_selection_stability.py`（`bash harness/exp_indexer_selection_stability.sh`
跑出 `.out.txt`）。规模 `T=1536, C=32768, H=32, D=64, Top-k=2048, block=16`（SM90 要求
`num_heads∈{32,64}`，故取 32）。q/kv ~ `N(0,1)` bf16，weights fp32；单序列 `ks=0, ke=C`。
四部分：

- **A 锚定**：真实 `deep_gemm.fp8_fp4_mqa_logits` vs torch 复刻的 fp8 档（应极近）；
- **B token 级**：三档 vs FP32 teacher 的 top-2048 recall / Jaccard / 翻转数，logit 噪声 σ，
  并加 **iid 噪声对照**（同 σ 的独立高斯）＋随机对照验证指标；
- **C 块级**：把 token 分数按 `max` 压成 `block=16` 的块分数再 top-k（compressed indexer）；
- **D 概率质量代价**：teacher 选中集里被量化选择丢掉/错补的**概率质量**（用一份独立注意力
  分数当 teacher 分布 `p`）；
- **E 上下文缩放**：`C ∈ {8192, 32768, 131072}` 固定 `k=2048`。

## 4. 结果

### 4.1 Part A：kernel 锚定

```
kernel logits: shape=(1536, 32768) dtype=torch.float32
kernel vs torch-fp8 ref: max-abs-rel = 2.499e-04
```

真实 SM90 kernel 与 torch 复刻 fp8 档的最大相对差 **2.5e-4**（fp32 累加顺序差），
说明后面的 torch 复刻链是可信的。

### 4.2 Part B：token 级选择稳定性

```
[fp8   ] noise(σ=1.1590, 排序相关 σ_r=1.1544 max=8.929) | top-2048 recall=0.9662 jaccard=0.9346 flipped=70/2048
[mxfp8 ] noise(σ=1.1943, 排序相关 σ_r=1.1897 max=10.504) | recall=0.9650 jaccard=0.9324 flipped=72/2048
[mxfp4 ] noise(σ=5.0918, 排序相关 σ_r=5.0743 max=43.524) | recall=0.8507 jaccard=0.7404 flipped=306/2048
对照: iid 噪声(σ=1.154) recall=0.9656 | 随机 recall=0.0624
```

- **FP8 与 MXFP8 几乎同档**：recall **~0.966**，2048 个选中里翻转 **70~72** 个（~3.5%）。
  有意思的是 MXFP8（带 UE8M0 逐 32 scale）比裸 fp8 噪声只大 3%（σ 1.159→1.194）。
- **MXFP4 明显更差**：σ_r **5.07**（≈4.4×），recall **0.851**，翻转 **306/2048**（~15%）。
  E2M1 只有 3 个尾数比特、且 `head_dim=64` 每 token 只有 2 个 scale，量化误差主导。
- **iid 对照**：给 teacher 直接加同 σ=1.154 的**独立高斯**噪声，recall=**0.9656**，
  与真实 fp8 量化（0.9662）**几乎逐位相等**。⇒ 量化噪声在「排序」意义上**就是 iid 高斯**，
  没有额外的相关性红利也没有惩罚。随机对照 0.0624（≈ k/C=0.0625）证明指标正确。

### 4.3 关键洞察：margin 不是诊断量，**候选池**才是

teacher 在 rank 2048 处的相邻间隔（margin）**极小**：

```
teacher boundary margin: mean=0.0069 median=0.0046 p10=0.0007 p90=0.0160
teacher margin / fp8 σ_r: median=0.004（远小于 1，却高 recall）
corr(row-recall, margin) = +0.05（几乎无关）
[fp8  ] 阈值±σ 内候选池=355.2/行, 实际翻转=70.0/行
[mxfp4] 阈值±σ 内候选池=1573.5/行, 实际翻转=306.0/行
corr(row-recall, 候选池) = -0.41 / -0.46
```

**反直觉但正确**：`margin≈0.0069 ≪ σ≈1.15`，按「噪声大于间隔就乱选」的朴素直觉，
recall 应当崩到 ~0.5；实测却是 0.966。原因：**决定翻转的是「阈值附近 ±σ 内的分数个数」
（候选池），不是相邻间隔。** 在 `C=32768` 上取 6.25% 分位，阈值处分数密度
`≈ 1/(C·margin) ≈ 145`，±σ 内约有 `2σ·145 ≈ 334` 个候选（实测 355），其中约 20%
真正跨过阈值 → **~70 个翻转**，与实测完全对上（`3194×σ` 的早期估算亦一致）。
所以**正确的诊断量是 `候选池 ≈ 2σ_r · C · pdf(threshold)`**；margin 与 recall 的
相关系数只有 +0.05，而候选池是 −0.4。

### 4.4 Part C：块级选择（compressed indexer）

```
block=16, num_blocks=2048, top-128 blocks
[fp8  ] block top-128 recall=0.9497 flipped=7/128
[mxfp8] block top-128 recall=0.9477 flipped=7/128
[mxfp4] block top-128 recall=0.7841 flipped=28/128
```

块级（块内取 `max`）比 token 级翻转更少（fp8：7/128），因为一块只要**有一个** token
分数高就保住；但 `block=16` 的粒度更粗，MXFP4 下仍有 **22%** 的块被换掉。

### 4.5 Part D：翻转的概率质量代价

```
teacher top-k 质量=0.0626
[fp8  ] 丢掉=2.11e-03 错补=2.12e-03 相对丢失=3.37e-02
[mxfp8] 丢掉=2.17e-03 错补=2.19e-03 相对丢失=3.47e-02
[mxfp4] 丢掉=9.37e-03 错补=9.32e-03 相对丢失=1.50e-01
```

「相对丢失」= teacher 选中集承载的概率质量里、落到量化选中集之外的比例。
FP8/MXFP8 **~3.4%**，MXFP4 **~15%**——与 recall 互补（1−recall）完全自洽，
且**丢掉与错补近似相等**（`2.11e-3 vs 2.12e-3`），说明翻转是「等量进出」的边界洗牌，
不是系统性偏移。

### 4.6 Part E：上下文长度缩放（k=2048 固定）

```
C=  8192 (top-2048/8192=0.250):  fp8=0.978  mxfp8=0.977  mxfp4=0.903
C= 32768 (top-2048/32768=0.062):  fp8=0.966  mxfp8=0.965  mxfp4=0.851
C=131072 (top-2048/131072=0.016):  fp8=0.957  mxfp8=0.955  mxfp4=0.811
```

上下文越长、选中比例越小，recall 越低但**衰减很慢**（fp8 0.978→0.957，MXFP4 0.903→0.811），
因为阈值右移到分数分布尾部、密度反而变小。长上下文下 FP8 仍稳，MXFP4 掉到 ~0.81。

## 5. 训练含义与边界

- **选择是自洽的，只要训练与推理用同一档**：论文式 (4) 的 `S_t` 用 indexer 自己的 logits
  选，训练与推理同档时选择一致；量化误差带来的 3.4%（FP8）/15%（MXFP4）翻转是**同一
  精度下的固有抖动**，不是训练-推理不一致。
- **跨档会错位**：若训练用 FP8 选、推理用 MXFP4 选，两套 `S_t` 在长上下文下最多差
  ~2%（fp8 0.957 vs mxfp4 0.811 → 差异 ~15%）。生产要么锁死同档，要么接受这部分支持集漂移。
- **MXFP4 的选择代价是真实存在的**：logits 本身的 MXFP4 误差（本篇）远大于 MXFP8；
  doc 19 的容差表也印证（`tests/test_attention.py:74-75`：fp8 容差 `1e-3` vs mxfp4 `0.02`）。
  DeepGEMM 把 indexer 的 Q 从 FP8 降到 MXFP4 换 2× 带宽时，**排序精度是隐藏代价**。
- **块级比 token 级稳**：压缩 indexer 用 `max` 聚合天然抗噪（fp8 7/128）；这可能是 DSA
  最终用块选择而非纯 token 选择的一个未被写明的鲁棒性理由。

**边界/未做**：① 本机无 SM100，MXFP4/MXFP8 主 kernel 的 UMMA/UTCCP 部分**仅分析**，
量化用的是官方 `tests/test_attention.py` 的 torch 复刻；② 输入取 iid 高斯，真实
hidden 分布有重尾/低秩结构，翻转率可能不同；③ `num_heads=32/head_dim=64` 是 SM90 锚定档，
生产 indexer 头数/维度不同；④ 未接真实 DeepEP/dispatch，选择翻转对**下游稀疏注意力输出**
的端到端影响未量。

## 6. 结论

把 indexer 从「前向打分 + 反向梯度」补上中间那一环——**选择**——并在 H100 上量出：
**FP8/MXFP8 的 top-k 选择 recall ~0.966、FP4 ~0.851；决定稳定性的不是相邻间隔而是
阈值附近 ±σ 的候选池；真实量化噪声在排序意义上等价于同 σ 的 iid 高斯。**
这给出 DSA sparse training 的一条隐含正确性边界：**支持集的精度必须与打分精度自洽**。

配套实验：`harness/exp_indexer_selection_stability.{py,sh,out.txt}`（H100 sm90 真机）。
