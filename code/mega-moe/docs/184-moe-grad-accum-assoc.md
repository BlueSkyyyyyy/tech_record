# MoE 反向梯度累积的「切分 × 结合序」可复现性：wgrad 敏感、dgrad 免疫

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P174-1** 增量（第 184 篇）。
> 第 170 篇（`docs/180`）把「MoE 反向算子链的可复现性边界」量到「**kernel 逐位确定、端到端不是**」，
> 并在 §8 明确留了一个口子：
>
> > 「给 `harness/exp_moe_backward_determinism.py` 增加一个『**多 microbatch 梯度累加**』维度，
> > 量梯度累积的**结合序**（而非仅 token 序）对 master 权重 bit 的影响。」
>
> 本篇把这一格补上。生产训练里 **gradient accumulation**（global batch 超过单卡显存时不可避免）
> 会做两件事：把同一批 token **切成若干 microbatch** 分别算，再把各份梯度 **按某个顺序相加**。
> 这两件事都不改数学，却都改 fp32 的舍入路径。核心结论一句话：
>
> **wgrad 对「切分」和「结合序」都敏感——切 3 个 microbatch 就有 43.5% 的输出 bit 变、
> 换个求和顺序变到 52%~67%；dgrad 对两者完全免疫（逐位 0 差）。**
> 但变的是哪一位：相对误差只有 **3.5e-8~6.6e-8**（FP8 量化误差 3.6e-2 的 **~1e6 分之一**）。
> 所以「**可复现性与数值精度是两件事**」（同第 170 篇）：训练要逐位复现，就得固定
> **(microbatch 数, 每份的顺序, 跨 rank 归约的算法)**，而不是靠 `use_deterministic_algorithms`。
>
> 可跑性：wgrad/dgrad 都是 H100(sm90) 真机可跑的 DeepGEMM FP8 kernel（`k_grouped`/`m_grouped`）。
> Mega MoE 主 kernel 是 **SM100 专属、仅分析**——本篇只用它架构无关的**反向数学与 grouped 布局**。
> 全部数字来自 `harness/exp_moe_grad_accum_assoc.{py,sh,out.txt}`
> （H100 80GB HBM3, sm90, 驱动 570, PyTorch 2.12, DeepGEMM 2.8.1）。
>
> 源码坐标：DeepGEMM 公开版 commit `057ca59`（v2.8.1）。目标卡代号敏感：全文用「目标卡」。

---

## 1. 问题：梯度累积把哪条「逐位可复现」的假设打碎

第 170 篇的结论是：单次反向里，**grouped GEMM 本体结构性 bit-确定**（不切 K、同 CTA 累积、
`SM90_TMA_REDUCE_ADD` 写回），真正的不确定源在「**谁按什么顺序把 token 交给 wgrad**」（dispatch
的原子槽位分配）。它量的是**同一次 GEMM 内**的 token 序。

但生产训练还有第二个、更常触发的序：**梯度累积**。设想 global batch = `B`，单卡放不下，于是切成
`S` 个 microbatch：

```
for s in range(S):
    loss_s = model(microbatch_s); loss_s.backward()   # 累积进 param.grad
```

`param.grad` 是 fp32 的持久缓冲，每步做的是 `grad += dW_s`。于是难免三个问题：

1. **「切 `S` 份再求和」与「一次算完」是不是同一份 bit？**（K 轴的累加树被切开了）
2. **换个求和顺序（`dW_0+dW_1+…` vs `…+dW_1+dW_0` vs 树形）是不是同一份 bit？**
   —— 这正是数据并行 **all-reduce** 的 ring-reduce（顺序）与 tree-reduce（树形）之争。
3. **dgrad 会不会一样敏感？**

第 170 篇的 §C 给了一半答案（收缩轴决定谁敏感：wgrad 收缩轴是 token、dgrad 是逐行独立），
但没把「累积顺序」这一独立变量单独拎出来量。本篇就用真实 kernel 把它量清楚。

---

## 2. 机制：为什么 wgrad 敏感、dgrad 免疫

### 2.1 wgrad 的 K 累加树（`k_grouped_*`）

`k_grouped_fp8_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:373-427`）**强制累积**——`c` 必填
（`gemm.hpp:403`），SM90 走 `k_grouped_* → sm90_fp8_gemm_1d1d`（`gemm.hpp:418-422`）。
它的调度与累加是：

- **不切 K**：每个任务 = 一个 `(m_block, n_block)`，`num_blocks = num_m_blocks * num_n_blocks`
  （`scheduler/gemm.cuh:100`）；K 不跨 CTA 拆、没有跨 tile 的原子归约。
- **按组切 K**：`get_next_block`（`scheduler/gemm.cuh:185-245`）对 k-grouped 逐组推进
  （`:226-245`，`get_next_k_group` `:71-82` 维护 `current_k_start` / `current_shape_k`），
  每组 K 由本 CTA 独立算。
- **K 轴顺序累加**：math warpgroup 对组内 K 逐 K-block 走 WGMMA（`sm90_fp8_gemm_1d1d.cuh:301-311`），
  每块做 `final_accum += scale_a·scale_b·accum`（`:318-325`）——`final_accum` 在
  `:267` 声明为一份寄存器，**按 K-block 顺序串行累加**。
- **写回**：`SM90_TMA_REDUCE_ADD_2D`（`:346-348`）把 `final_accum` **加到** D（组维度已体现在地址
  `current_group_idx * shape_m + …`）——即 `D_g = C_g + Σ_k`。

所以对「某个专家的权重梯度」而言，它的 fp32 值就是**一条按 K-block 排的加法链**。把同一批 token
切成两个 microbatch、分别算再相加，等于把这条链在中间截断、改成 `(链A) + (链B)`——**结合律在浮点里
不成立**，于是 bit 变。

### 2.2 dgrad 的逐行独立（`m_grouped_*`）

dgrad = `dA = dY · W2ᵀ`（`docs/13`）。它的不等长轴是 **M（token）**，输出每一行只依赖那一个 token：
`dA[i,:] = Σ_k dY[i,k]·W2[:,k]`。microbatch 切分只是把 token 行**重新分组/重排**，不改变任何一行的
计算。`m_grouped` 调度器里 `is_computation_valid` 只对 M-grouped 系列定义、k-grouped 的分支是
`DG_TRAP`（`scheduler/gemm.cuh:296-309`），也从侧面说明「组」在 M 轴、逐行独立。故 dgrad 免疫。

---

## 3. Part A：同一批 token，「一次大 GEMM」vs「切 3 个 microbatch 求和」

配置（真实 SM90 FP8）：`G=8` 专家、`m=512`、`n=1024`，每专家 token 数
`ks=[1024,768,640,512,896,384,768,512]`（均 128 对齐）。把每个专家的 K 按同一 token 顺序切成
3 份（每份 128 对齐），得到每 microbatch 总 token `[2432, 1664, 1408]`：

```
A1 一次大 GEMM vs 3-microbatch 求和: 逐位差 = 1823583/4194304 (43.5%), relerr = 3.472e-08
A2 一次大 GEMM vs fp64 真值:        relerr = 3.634e-02
A3 3-mb 求和   vs fp64 真值:        relerr = 3.634e-02
```

- **43.5% 的输出 element 逐位不同**，但 relerr 只有 **3.5e-8**——比 fp32 机器精度（~6e-8）还小，
  是典型的「末位抖动」。
- 两者对 fp64 真值的误差都是 **3.6e-2**——那是 **FP8 量化误差**（`per_channel_cast_to_fp8` 的
  per-128 块 scale，`deep_gemm/utils/math.py:52-59`），与切分无关。

> 一句话：**切分把「逐位」打碎，但把「数值」毫发无伤**；累积序噪声比量化噪声小 **~1e6 倍**。

---

## 4. Part B：结合序（association order）——同一批偏梯度、不同求和顺序

固定每专家 1024 token，切成 8 个 128-chunk，每个 chunk 是一份 `[G,m,n]` 偏梯度；用真实 kernel 算
这 8 份（K 不跨 chunk），再按四种顺序求和（`association order`）：

```
参考(一次大 GEMM) vs fp64 真值 relerr = 3.631e-02
B  forward        vs 一次大 GEMM: 逐位差=2167886/4194304 ( 51.7%), relerr=4.816e-08, vs fp64=3.631e-02
B  reverse        vs 一次大 GEMM: 逐位差=2796726/4194304 ( 66.7%), relerr=6.637e-08, vs fp64=3.631e-02
B  pairwise-tree  vs 一次大 GEMM: 逐位差=2638321/4194304 ( 62.9%), relerr=5.982e-08, vs fp64=3.631e-02
B  by-magnitude   vs 一次大 GEMM: 逐位差=2740445/4194304 ( 65.3%), relerr=6.425e-08, vs fp64=3.631e-02
B  forward vs reverse : 逐位差=2583312/4194304
B  forward vs tree    : 逐位差=2207924/4194304
B  forward vs bymag   : 逐位差=2518891/4194304
```

- **顺序本身就是 bit 差异的独立来源**：`forward` 与 `reverse`/`tree`/`by-magnitude` 两两之间也差
  200 万~260 万个 element。哪怕每份偏梯度都已算好、只是换加法顺序，bit 也不复现。
- **幅度仍然极小**：所有顺序对 fp64 的 relerr 都在 3.631e-2 附近抖动（小数点后第 7 位），彼此 relerr
  4.8e-8~6.6e-8。
- `pairwise-tree` **不比** `forward` 更准/更差（5.98e-8 vs 4.82e-8，同量级）——树的深度不改变这个尺度。

> 这一组直接回答**数据并行**的问题：`ring all-reduce`（顺序归约）与 `tree all-reduce`（树形归约）
> 在 fp32 下给出**不同 bit**。想跨不同拓扑可复现，就必须固定归约算法（或用 bit-确定的 reduce）。

---

## 5. Part C：dgrad 对 microbatch 切分完全免疫

dgrad（`m_grouped_fp8_gemm_nt_contiguous`，`G=8, n=256, k=512`）把每组 token 行做一次置换
（= microbatch 切分后的行重排），跑完再按行对齐：

```
C  每组行置换（= microbatch 切分的行重排）后按行对齐: 逐位差=0/327680  ← 每行独立、bit 不变
```

**逐位 0 差**。这印证第 170 篇 §C2，并把结论从「组内 token 置换」推广到「microbatch 切分」：
dgrad 的每一行是一个独立 token 的梯度，切分只改行的摆放，不改任何一行的值。

> 训练含义：**dgrad 的梯度累积天然可复现**（无论怎么切、怎么排）；需要小心的只有 **wgrad**。

---

## 6. Part D：训练真实写法——kernel 原地累积 vs 各 microbatch 求和

生产代码通常不「各 microbatch 算 fresh、再 torch 求和」，而是让梯度缓冲常驻、**原地累积**
（DeepGEMM 里即 `c = d`）。实测：

```
D1 原地累积(c=d) vs fresh+torch 求和: 逐位差=0/4194304, relerr=0.000e+00
D2 原地累积 vs fp64 真值 relerr = 3.634e-02
```

**逐位 0 差**——因为两者都是「zeros → +dW_0 → +dW_1 → +dW_2」的**同一条左结合 fp32 加法链**，
`SM90_TMA_REDUCE_ADD_2D`（`sm90_fp8_gemm_1d1d.cuh:346`）的 `D = C + Δ` 正好是这个语义。
所以：

- 训练里「kernel 原地累积」与「先算后加」**可以放心互换**（只要顺序一致）；
- 真正决定 bit 的是 **microbatch 的切分与顺序**，不是「累积发生在 kernel 内还是 kernel 外」。

---

## 7. 结论与训练建议

把第 170 篇（token 序）和本篇（累积序）拼起来，「一次 MoE 训练步逐位可复现」的完整前提是：

| 层级 | 是否 bit-确定 | 想复现要固定什么 |
|---|---|---|
| wgrad GEMM 本体（同 token 序） | 是（第 170 篇） | —— |
| dgrad GEMM（任意 token 序/切分） | 是（本篇 §5） | —— |
| wgrad 的 **microbatch 切分** | 否（§3，43.5% bit 变） | microbatch 数、每份边界 |
| **累加顺序**（含 DP all-reduce） | 否（§4，52–67% bit 变） | 求和顺序 / reduce 算法 |
| dispatch 的 token 行序（第 170 篇） | 否 | `EPHandle.deterministic_sort` |

**建议**：训练复现性清单应显式钉住 **`(grad_accum_steps, microbatch 顺序, DP reduce 算法)`**，
而不是只打 `use_deterministic_algorithms`——后者只修「稠密回退 + Mega Gate/mHC 的 split-K」
（第 170 篇 §B/§8），对 grouped 反向的累积序**毫无落点**。

**幅度**：所有抖动都 ≲ **7e-8** 相对，比 FP8 量化误差 **~1e6 倍**小、比 bf16 输出舍入（~1e-3）也小。
**训练无害，只影响「逐位回归/调试」**。这也再次印证：**可复现性 ≠ 数值精度**。

**与 Mega MoE 的关系**：Mega MoE 把 wgrad 天然留在**专家本地 rank**（`docs/14`，反向无 all-reduce），
所以跨 rank 的归约序噪声只出现在**共享专家 / 稠密分支**与优化器状态同步上；而**前向的
dispatch/combine 单 mega-kernel 融合**正是为了把「通信序」也钉进一个可控的 kernel 里
（`docs/02`/`docs/11`）——这与本篇「序要显式可控」的结论同源。

---

## 8. 边界与诚实说明

- wgrad/dgrad 用**真实 SM90 FP8 kernel**（H100 真机）；**Mega MoE 主 kernel（SM100/UMMA/tcgen05）
  仅分析**，本篇只用它架构无关的反向数学与 grouped 布局。
- Part A/B 的「切分」都保证了每份 128 对齐（kernel 契约 `k % k_alignment == 0`，
  `csrc/apis/gemm.hpp:69`）——真实训练里若某专家某 microbatch 的 token 数不被 128 整除，会被 padding
  到对齐（`check_k_grouped_args` 强制），等价地把分析缩小到对齐后的粒度。
- Part B 的 8 个 chunk 是**等长**的理想化分布（便于构造丰富的结合序）；Part A 用**不等长**的真实
  路由分布（`ks=[1024,…,512]`）。
- dgrad 的「行置换」测试是 microbatch 切分的等价形式（都是行重排）；分母 `327680` 已用
  `random.seed(7)` 钉住（`generators.py:383` 的 `random.uniform`）。
- 全部为**单进程、单卡**测量；跨 rank all-reduce 的序用 §4 的顺序/树形**模拟**，未接真实 NCCL
  （本机 DeepEP 运行期受阻，见 ROADMAP「阻塞」）。

---

## 9. 参考与复现

- 实验：`harness/exp_moe_grad_accum_assoc.sh` → `harness/exp_moe_grad_accum_assoc.out.txt`；
  Python `harness/exp_moe_grad_accum_assoc.py`（`tests/generators.py` sha256[:16] `1934de0420e1e43b`）。
- 上游源码行号（DeepGEMM `057ca59`）：
  `csrc/apis/gemm.hpp:373-427`（`k_grouped_fp8_gemm_nt_contiguous`）、`:403`（`c` 必填）、
  `:55-76`（`check_k_grouped_args`，`:69` K 对齐）、
  `deep_gemm/include/deep_gemm/scheduler/gemm.cuh:100`（不切 K）、`:71-82`/`:226-245`（逐组推进）、
  `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:267/301-311/318-325/346-348`（K 轴顺序累加
  + TMA reduce-add）、
  `deep_gemm/include/deep_gemm/common/types.cuh:31-37`（`is_k_grouped_contiguous`）、
  `deep_gemm/utils/math.py:52-59`（`per_channel_cast_to_fp8`）、
  `tests/generators.py:488-544/551-608/378-419`（K/M-grouped 布局生成器）。
- 前序：第 170 篇（`docs/180`，反向算子链可复现性边界，本篇接其 §8）、
  第 12/13 篇（`docs/12`/`docs/13`，wgrad/dgrad 的 grouped 布局）、
  第 14 篇（`docs/14`，反向四 GEMM 端到端）、第 15 篇（`docs/15`，反向 ncu 归因）。
- 线上资料：DeepGEMM [PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316)（Mega MoE 融合 kernel）。
