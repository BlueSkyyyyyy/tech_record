# indexer 打分的混合精度分配：KV 主导带宽、Q 几乎免费

> 本篇是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的增量（P164-1）。
> 承接第 171~173 篇（`docs/171`、`docs/172`、`docs/173`）：那三篇把 indexer 的 **量化选择稳定性**、
> **选择翻转对下游稀疏注意力的影响**、**真实头结构（per-head softmax / GQA）/ 训练反向** 都量了，
> 但始终把 Q 与 KV **当成同一档精度**（要么都 FP8、要么都 MXFP8、要么都 MXFP4）。生产里这两个
> 操作数**根本不对称**——KV 是逐 token 存下来的长上下文缓存 `[C, head_dim]`，Q 只有当前 token 的
> `[T, num_heads, head_dim]`（`T ≪ C`）。本篇把**精度分配**这件事单独拆开：谁主导带宽、谁主导误差、
> 混档能拿回什么。
>
> **本机 H100（sm90）只能跑 fp8 档真实 kernel**（`fp8_fp4_mqa_logits` 在 SM90 上要求
> `Q 裸 e4m3 + KV 逐 token fp32 scale`，`attention.hpp:124-130`）；MXFP8/MXFP4 是 SM100 专属，
> 用纯 torch 复刻（逐 32 元素 + UE8M0，`deep_gemm/utils/math.py`）以**同一套语义**分析，
> 这部分**仅分析**。所有数字见 `harness/exp_indexer_mixed_precision.{py,sh,out.txt}`。

## 0. 结论先行

在三档精度（`fp8` = Q 裸 e4m3 / KV 逐 token fp32 scale、`mxfp8`、`mxfp4`，后两者逐 32 元素 + UE8M0）
的 `Q × KV` 组合上，H100 实测（`T=256, C=65536, H=32, D=64, top-2048`）：

1. **KV 是带宽主导项**：`C/T·H = 8` 时 KV 已占总字节的 **~89%**（真实长上下文 `C=128K` 更极端）；
   把 KV 从 fp8 换 mxfp4 让**总字节降到 0.55×**。
2. **KV 换 MXFP8 几乎免费**：`fp8/fp8 → fp8/mxfp8` 字节 0.97×、top-k recall `0.9616 → 0.9602`
   （翻转 79→82），**可以无脑换**。
3. **只有把 Q 也压到 MXFP4 才真正伤精度**：`fp8/mxfp4` recall `0.8743` vs 全 `mxfp4` `0.8293`
   —— 省 Q 那 **~10% 的字节**要付 **4.5 个 recall 点**，而 Q 换成 mxfp4 本身只省 **0.05×** 总字节。
   **最优 MXFP4 配方是「Q 保 8 位、KV 压 MXFP4」**。
4. **误差上 Q 与 KV 近乎对称**（只量化 Q 的 score σ=`0.850` vs 只量化 KV 的 `0.800`，mxfp4 各 `3.65`/
   `3.65`）：不能靠「只压 KV」白嫖——但**字节上 Q 便宜得多**，所以省位该省在 KV 上。
5. **bf16 的 Q 在 fp8 与 mxfp8 下实际不可分**（逐元素最大差 `9.77e-4` ≈ 1 个 e4m3 次正规台阶，
   top-k 与 σ 同到 4 位）：因为逐 32 的 2 的幂 scale 与 e4m3 的指数**对齐**，round-trip 只在次正规区留差。
6. **下游是二阶的**（复现第 172 篇）：`tau=0.25`（支持集捕获质量 0.977）下，仅换支持集的输出损失
   `1.64e-2`；量化配方只把它抬到 `1.66e-2`（fp8/fp8）/ `1.79e-2`（fp8/mxfp4）/ `1.93e-2`（全 mxfp4）
   —— recall 掉 13.8% 只换来输出误差 **+18%**。
7. **逐 indexer 头分配精度没用**：只压单个 indexer 头时 score σ 的 max/min 仅 `1.16×`，且与 `|w_h|`
   相关 `0.86` —— 逐头权重**已经**编码了重要性，没有新杠杆。

## 1. 背景：为什么 Q 与 KV 必须分开看

indexer 的打分（lightning indexer，DeepSeek-V3.2）在 README `:88-110` 有定义：

```python
kv_j   = kv_fp[j, :] * kv_sf[j]              # 逐 token 的 KV，一行一个 scale
out_ij = relu(q[i, :, :] @ kv_j) * weights[i, :]
out_ij = out_ij.sum()                        # 压缩成标量
```

即 `out_ij = Σ_h w[i,h] · relu(<q[i,h,:], kv_j>)`。生产里：

- `kv` 是**整个上下文**的缓存，形状 `[C, head_dim]`（`C` = 上下文长度，几十 K 到 128K）；
- `q` 只有**当前 token**，形状 `[T, num_heads, head_dim]`（`T` = 本轮 query token，通常 ≪ `C`）。

这个打分算子是**访存受限**的（FLOP 少、要读完整个 KV cache），所以**字节就是时间**，而字节几乎全在
KV 上。于是「精度分配」的正确提法是：**Q 与 KV 各自贡献多少 logit 误差、各自占多少字节**，
把位花在「便宜」的一侧。

host 契约（`csrc/apis/attention.hpp`）把两条路分开：

- 共享检查 `check_mqa_logits_q_and_weights:18-44`：
  `head_dim ∈ {32,64,128}`（`:27`）、Q 必须连续且是 `e4m3`/packed FP4（`:28-29`）、
  **`is_mx_sf == (arch_major == 10)`**（`:32`）——即 MX scale 是 SM100 专有，SM90 走另一套；
  `weights` 的行 stride 必须 16B 对齐（`:42-43`，TMA 逐行载入）。
- 连续入口 `fp8_fp4_mqa_logits:67-134`：`out_dtype` SM100 是 bf16、SM90 是 fp32（`:80`）；
  KV 的 scale 类型随 `is_mx_sf`：SM90 是 **fp32**、SM100 是 **int32（packed UE8M0）**（`:96`）；
  输出压缩成 `[seq_len, max_seqlen_k]`、行 stride 1024B 对齐（`:113-117`）；
  SM90 只接受 `e4m3`、`num_heads ∈ {32,64}`、无 `schedule_meta`，派发到
  `sm90_fp8_mqa_logits`（`:124-130`）。

量化器在 `deep_gemm/utils/math.py`：

- `ceil_to_ue8m0:13-16`：把 `amax/quant_max` 向上取到 2 的幂（fp32 位域，指数 + 尾数非零进位）。
- `per_token_cast_to_fp8:26-49`：`gran_k` 分块求 amax → `sf = ceil_to_ue8m0(amax/448)`（`:35-37`）
  → `x/sf` 转 e4m3。**`sf` 是 2 的幂**，故 `x/sf` 只是把 `x` 的指数平移——见 §4 第 5 点。
- `per_custom_dims_cast_to_fp8:75-81`：**SM90 的 KV 路径**——沿除第 0 维外的所有维求 amax，
  即**逐 token 一个 fp32 scale**（不是 2 的幂、也不分块）。
- `per_token_cast_to_fp4:99-125`：`gran_k` 分块、`sf = ceil_to_ue8m0(amax/6)`（`:109-110`）、
  量化到 E2M1（`_quantize_to_fp4_e2m1:84-96`，7 个中点 ties-to-even），nibble 偶数下标在低 4 位。
- `cast_back_from_fp8/fp4:155-176` 把量化结果反量化回 fp32 供参考复刻。

> 注：MX 档的 SF 在真实 kernel 里是 **packed int32**（4 个 UE8M0 字节 → 1 个 int32，TMA 对齐，
> `per_token_cast_to_fp8:40-47`）。本实验的字节账按「1 字节 / 32 元素」的理想紧凑口径算，
> 对 `head_dim=128` 恰好一致；对 `head_dim=64` 会因补齐到 4 的倍数略有出入，但 SF 占比 <3%，
> 不改变任何结论。

## 2. 实验设计

用真实 SM90 `fp8_fp4_mqa_logits` 锚定 fp8 档，再以**纯 torch 复刻**三档量化，对 `Q × KV`
的 3×3 组合逐一算 logits、top-k recall 与噪声，并把支持集接进下游注意力。

- 形状 `T=256, C=65536, H=32, D=64, top-2048, dv=64`（上下文是 query 头数的 8 倍，KV 主导字节）。
- 量化语义与 kernel 一致：`fp8` 档 Q 裸 e4m3（无 scale）、KV `per_custom_dims_cast_to_fp8((0,))`
  逐 token fp32 scale；`mxfp8`/`mxfp4` Q/KV 都逐 32 元素 + UE8M0（`gran_k=32`）。
- 「teacher」= 无量化 fp32 logits 的 top-k，作为选择质量基准。

## 3. H100 实测

### Part A：锚定

真实 `fp8_fp4_mqa_logits` vs 同档 torch 参考：`max-abs-rel = 2.722e-04`（与第 171/173 篇同量级），
teacher logit `std = 31.89`、`|max| = 227.5`。

### Part B：Q × KV 精度矩阵

```
  Q\KV      fp8                         mxfp8                       mxfp4
  fp8       rec=0.9616 flip=79 sig=1.1663 rec=0.9602 flip=82 sig=1.2012 rec=0.8743 flip=258 sig=3.7419
  mxfp8     rec=0.9616 flip=79 sig=1.1663 rec=0.9602 flip=82 sig=1.2012 rec=0.8743 flip=258 sig=3.7419
  mxfp4     rec=0.8772 flip=251 sig=3.7334 rec=0.8771 flip=252 sig=3.7437 rec=0.8293 flip=350 sig=5.1196
```

- 同档对角线：`fp8=0.9616`、`mxfp8=0.9602`、`mxfp4=0.8293`。
- **KV 决定大格局**：看第一行（Q=fp8），KV 从 fp8→mxfp8 几乎不动（0.9616→0.9602），
  KV→mxfp4 才掉到 0.8743。
- **Q 与 fp8/mxfp8 等价**：第三行（Q=mxfp4）两列表几乎相同（0.8772/0.8771），
  说明 Q 压到 mxfp4 才是 Q 侧的分水岭。
- **混档 vs 同档**：`Q=fp8,KV=mxfp4 = 0.8743` 明显好于 `Q=mxfp4,KV=mxfp4 = 0.8293`。

### Part C：logit 误差归因

```
  [fp8  ] 只 Q : relL2=2.667e-02  sig=0.8505      [fp8  ] 只 KV: relL2=2.510e-02  sig=0.8005
  [mxfp8] 只 Q : relL2=2.667e-02  sig=0.8505      [mxfp8] 只 KV: relL2=2.665e-02  sig=0.8501
  [mxfp4] 只 Q : relL2=1.145e-01  sig=3.6508      [mxfp4] 只 KV: relL2=1.144e-01  sig=3.6479
```

**误差上 Q 与 KV 近乎对称**：只量化 Q（fp8）使 score σ=`0.850`、只量化 KV 使 σ=`0.800`；两者都压
mxfp4 时各 `3.65`，且「两者都量化」的 σ（Part B）≈ 各自 σ 的平方和开方（`1.166 ≈ √(0.850²+0.800²)`），
方差近似可加。**这意味着不能靠「只压 KV」把误差减半**——Q 也贡献一半的噪声。但见下：**字节上
Q 比 KV 便宜得多**，所以位该花在 KV 上、省也该省在 KV 上。

### Part D：存储字节账

```
  combo     Q bytes   KV bytes  total     KV share      相对 fp8/fp8     recall
  fp8/fp8   0.524MB   4.456MB   4.981MB   89.5%         1.00x            0.9616
  fp8/mxfp8 0.524MB   4.325MB   4.850MB   89.2%         0.97x            0.9602
  fp8/mxfp4 0.524MB   2.228MB   2.753MB   81.0%         0.55x            0.8743
  mxfp8/mxfp4 0.541MB 2.228MB   2.769MB   80.5%         0.56x            0.8743
  mxfp4/mxfp4 0.279MB 2.228MB   2.507MB   88.9%         0.50x            0.8293
```

- KV 占总字节 **89.5%**（C 越大越极端）。
- `fp8→mxfp8` 的 KV：字节 0.97×、recall 只降 0.0014 —— **免费**。
- `fp8→mxfp4` 的 KV：字节 **0.55×**、recall 降 0.087 —— **真正的带宽-精度交易**。
- Q 从 fp8 换 mxfp4：总字节只从 2.753MB 降到 2.507MB（**省 0.05×**），却把 recall 从 0.8743 砸到
  0.8293（**-4.5 点**）—— **极不划算**。

一句话：**KV 压缩是收益来源，Q 保 8 位是廉价保险**。

### Part E：下游稀疏注意力输出误差（vs 稠密全上下文）

把支持集接进真实注意力（支持集逐 token 共享、per-head softmax、GQA `G=4`，复用第 172/173 篇的
语义），对比量化支持集 vs 稠密全上下文的输出：

```
  tau=0.2 (捕获质量=0.977)  仅选择 loss=1.64e-02  fp8/fp8=1.66e-02  fp8/mxfp4=1.79e-02  mxfp4/mxfp4=1.93e-02
  tau=0.5 (捕获质量=0.591)  仅选择 loss=5.38e-01  fp8/fp8=5.39e-01  fp8/mxfp4=5.55e-01  mxfp4/mxfp4=5.70e-01
  tau=1.0 (捕获质量=0.201)  仅选择 loss=3.11e+00  fp8/fp8=3.12e+00  fp8/mxfp4=3.15e+00  mxfp4/mxfp4=3.18e+00
```

- **稀疏近似本身是主项**，量化配方只是它上面的二阶修正：在尖锐工作点（`tau=0.2`，捕获 97.7% 质量），
  仅换支持集的损失就 `1.64e-2`；量化配方把它抬到 `1.66e-2`~`1.93e-2`。
- **recall 掉 13.8%（0.9616→0.8293）只让下游误差 +18%**（1.64e-2→1.93e-2），且 `fp8/mxfp4` 与
  全 mxfp4 的差距（1.79e-2 vs 1.93e-2）在输出上很小。这与第 172 篇「选择翻转是二阶」一致。
- 平坦度（`tau`）越大，支持集捕获质量越低、下游误差越大——**主权变量是「稀疏捕获了多少质量」，
  不是量化本身**。

### Part F：逐 indexer 头分配精度有用吗？

indexer 分数是 `Σ_h w[i,h]·relu(<q[i,h],kv>)`。只把**单个** indexer 头的 Q 压到 mxfp4、看 score σ：

```
  逐头 sigma: min=0.5855 max=0.7114 最敏 4 头=0.6912 最钝 4 头=0.5950 比=1.16x
  corr(|w_h|, sens_h) = 0.858
```

各头敏感度**几乎均匀**（max/min `1.16×`），且与 `|w_h|` 强相关（`0.86`）——**逐头权重已经编码了
重要性**，再按敏感度给不同头分配精度**没有新杠杆**。精度该在 **Q vs KV 这一对**上分，而不是在头之间分。

## 4. 三条「为什么」

1. **为什么 KV 主导字节**：`q` 是 `[T, H, D]`、`kv` 是 `[C, D]`。当 `C ≳ T·H`（生产里 `C` 是长上下文、
   `T` 是当前 token），KV 元素数就超过 Q。`C/T·H = 8` 时 KV 已占 89.5%，`C=128K` 更极端。
2. **为什么「只压 KV」不能省掉一半误差**：score 对 Q 与 KV 的噪声贡献**对称**（Part C，σ 0.850 vs
   0.800）——每个 logit 的方差 ≈ `Σ_h w²·D·(var(q)·var(δk) + var(δq)·var(k))`，Q 项与 KV 项同阶。
   但 Q 便宜（字节少），所以在 Q 上省位是「高代价低收益」。
3. **为什么 bf16 的 Q 在 fp8 与 mxfp8 下不可分**：`per_token_cast_to_fp8` 的 `sf = ceil_to_ue8m0(amax/448)`
   是 **2 的幂**（`math.py:37`），`x/sf` 只把 `x` 的指数整体平移；对 e4m3 这种浮点格式，
   「平移指数 → 舍入 → 平移回来」在正规区**与直接舍入完全等价**，差异只出现在 e4m3 次正规区
   （实测逐元素最大差 `9.77e-4`，正好 1 个次正规台阶）。故 fp8 与 mxfp8 对 Q 的实际效果不可分。

## 5. 与前面各篇的关系

- 第 19 篇（`docs/19`）拆了 indexer 前向 kernel 本体（SM90 连续 / 分页、Q 在 B 侧 / KV 在 A 侧、
  逐 token fp32 KV scale）；本篇章说明**这个 scale 的位置与粒度**正是「Q 与 KV 精度不对称」的来源。
- 第 171 篇（`docs/171`）量三档**同档**量化的选择稳定性；本篇章把它展开成 `Q × KV` 二维，
  并给出可执行的**混档**配方与字节账。
- 第 172 / 173 篇（`docs/172`、`docs/173`）证明选择翻转对下游是**二阶**的；本篇 Part E 在新的
  `Q × KV` 轴上复现同一结论。
- P15/P25 篇（`docs/25`、`docs/35`）拆了 SF 布局与 UE8M0 打包；本篇的字节模型建立在同一套
  `ceil_to_ue8m0` / packed int32 语义上。

## 6. 结论与边界

- **结论**：indexer 打分的精度分配有一条清晰的最优：**KV 是收益来源（fp8→mxfp8 免费、→mxfp4 换
  ~0.45× 字节），Q 保 8 位是廉价保险**（压 Q 省 <0.06× 字节却砸 4.5 个 recall 点）。误差上 Q/KV
  对称、字节上极不对称，所以「省位省在 KV」。逐头分配精度没有价值。
- **边界 1**：本机 H100（sm90）只能跑 **fp8 档**真实 kernel（`attention.hpp:124-130`）；MXFP8/MXFP4
  是 SM100 专属，用纯 torch 复刻（`math.py` 语义）分析，**仅分析**。所有「时间∝字节」的论断基于
  indexer 打分**访存受限**这一前提（FLOP 极低、要读完 KV cache），未在本机直接测 MXFP4 内核墙钟。
- **边界 2**：字节账按理想紧凑 SW（1 B / 32 元素）；真实 packed int32 + TMA 对齐会让 SF 略大，
  对 `head_dim=128` 恰好一致，对 `head_dim=64` 有 <3% 出入，不改变结论。
- **边界 3**：下游用第 172/173 篇的 `rho` 建模（indexer 是**另训**小模型、与主注意力对齐但不相同），
  非真实训练出的 indexer。`rho=0.98`/`tau=0.2` 是「已对齐 + 尖锐」的生产工作点。

## 参考

- 实验：`harness/exp_indexer_mixed_precision.{py,sh,out.txt}`（H100 sm90 真机）。
- 源码：`deep_gemm/utils/math.py:13-49/75-125/155-176`、
  `csrc/apis/attention.hpp:18-44/67-134`、`tests/test_attention.py:29/59-72/78-101`、
  `deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh`、README `:88-110`。
