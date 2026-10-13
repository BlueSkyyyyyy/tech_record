# Mega MoE 前向复刻（下）：路由专家 + 共享专家 + combine 的同图 H100 数值闭环

> 第 99 篇（P89-1）用 SM90 `m_grouped_fp8_gemm_nt_contiguous` 把 **路由专家**的前向流水线
> （dispatch → linear1 → SwiGLU → linear2 → combine）在 H100 上逐段搭起来、做了误差分解；
> 第 44 篇（P34-1）单独拆过 **共享专家**分支的 host/kernel/调度**语义**，并只在 **bf16 稠密**
> 路径上跑了它的数值。两条线各缺一半：路由那条**没带共享项**，共享那条**没和路由同图一起跑**。
>
> 本篇把两条通路**放进同一个 H100 前向**：共享专家走 **dense FP8** `fp8_gemm_nt`（共享权重是 2D
> 张量、没有专家维），路由走 `m_grouped_fp8_gemm_nt_contiguous`，combine 把共享结果**按
> `kNumTopk` 号槽、fp32 相加一次、不带权重**。量出端到端误差、分层误差、共享分支的吞吐，并把
> 「共享项只加一次」的**反例**坐实。所有数字来自 `harness/exp_mega_moe_forward_shared.{py,sh,out.txt}`
> （H100 sm90，DeepGEMM 2.8.1）。
>
> **Mega MoE 主 kernel 是 SM100 专属、本机跑不了**，本篇只用其**架构无关的数学与量化语义**复刻；
> 主 kernel 的 UMMA/tcgen05/NVLink 部分**仅分析**。目标卡代号敏感：全文用「目标卡」指代。

参考坐标：DeepGEMM commit `057ca59`（Public release 26/09/30，v2.8.1）；
`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、
`.../scheduler/mega_moe.cuh`、`.../layout/mega_moe.cuh`、`csrc/apis/mega_moe.hpp`、
`tests/test_mega_moe.py`。

---

## 1. 为什么需要这一篇

DeepSeek-V3/R1 的 MoE 层是

```
y = Σ_{i∈topk} w_i · Expert_i(x) + SharedExpert(x)
```

路由专家稀疏激活、共享专家**对所有 token 恒定全量激活**。Mega MoE 把两者**融进同一个
mega-kernel**，共享专家被设计成三处「借道」（第 44 篇）：

1. **借道 input buffer**：`shared_l1_acts ≡ x`（`layout/mega_moe.cuh:415`），零拷贝复用激活；
2. **借道调度器**：加 `SharedLinear1/2` 两个相位 + 两个每域计数器，共享 L1 填进 dispatch 空档
   （`scheduler/mega_moe.cuh:446-448`），共享 L2 按 `is_shared_l2_early` 决定提前或补尾
   （`:443-444`）；
3. **借道 combine**：结果写进 combine buffer 的 `kNumTopk` 槽（`kernel:1433/1449-1450`），被
   fp32 累加器**无权重加一次**。

第 99 篇的 H100 复刻只走了路由那条，第 44 篇的数值只跑了共享的 **bf16 稠密**、且路由项是**模拟**的
而非真实 grouped kernel。本篇补的正是「**两条真实 FP8 kernel 同图 + combine 相加**」这一格——
它才对应 Mega MoE 真正的整层前向。

---

## 2. 共享专家在前向里的三条语义（真实行号）

### 2.1 L1 沿 N 变宽、L2 沿 K 变宽

共享专家是一个有 `S = E_s·I` 个中间通道的前馈层（`E_s = num_shared_experts`，
`S = shared_intermediate_hidden`）。调度器把形状关系写成编译期常量
（`scheduler/mega_moe.cuh:181-185`）：

```
SHARED_L1_SHAPE_N = L1_SHAPE_N · kNumSharedExperts   // = 2·E_s·I  （N 轴）
SHARED_L1_SHAPE_K = L1_SHAPE_K                        // = H
SHARED_L2_SHAPE_N = L2_SHAPE_N                        // = H
SHARED_L2_SHAPE_K = L2_SHAPE_K · kNumSharedExperts   // = E_s·I    （K 轴）
```

**为什么 L1 在 N、L2 在 K**：L1（gate+up）输出自然是 `2·E_s·I`（N 轴宽）；L2 把 `E_s·I` 个中间
激活压回 `H`，所以**收缩维 K** 是 `E_s·I`。host 侧也据此断言（`csrc/apis/mega_moe.hpp:216-221`）：

```cpp
num_shared_experts = shared_intermediate_hidden / intermediate_hidden;   // :216
DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);   // :218
DG_HOST_ASSERT(shared_l1_n == shared_intermediate_hidden * 2);           // :219
DG_HOST_ASSERT(shared_l2_n == hidden);                                   // :221
```

且共享权重**恒为 FP8 E4M3**（`:222-223`，即使路由权重是 FP4）——与 kernel 里
`shared_b_dtype_t = float_e4m3_t`（`kernel:154`）一致。本篇实验取 `S = E_s·I`，正是这条。

### 2.2 共享 L1 的 SwiGLU **不乘 top-k 权重**

路由专家的 top-k 权重是在 L1 epilogue 里乘进激活的（`kernel:1135`）。共享专家**没有**这个权重：
L1 epilogue 的权重缓存初值是 1.0（`kernel:1068`）

```cpp
float stored_cached_weight = 1.0f;                       // :1068
...
if (not task_info.is_shared() and ...)                   // :1094  只有路由才重载
    stored_cached_weight = *buffer.l1_topk_weights_buffer...
```

所以共享项的 SwiGLU 输出恒等于 `silu(gate)·up`，不带任何路由概率。

### 2.3 共享结果进 combine 的 `kNumTopk` 槽、只加一次、不带权重

L2 epilogue 给每个 token 写 combine 的源元数据；共享专家的目标槽**固定为 `kNumTopk`**、目标
rank/token 是本地（`kernel:1287-1288`）：

```cpp
cached_src_metadata[j] = task_info.is_shared()
    ? layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk)
    : *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
```

combine 段每个 lane 读一个槽，共享槽就是 `kNumTopk`（`kernel:1432-1433`），而且**先取共享槽**
（`kernel:1449-1450` 的 `slot_idx = (mask >> kNumTopk) ? kNumTopk : __ffs(mask)-1`），在**同一个
fp32 累加器**里与路由项相加（`:1472-1490`），最后 bf16 写回 `y`（`:1492-1519`）。combine buffer
的最外层维就是 `num_topk + (num_shared_experts > 0 ? 1 : 0)`（`layout/mega_moe.cuh:448`）。

测试侧的权威对照是 `tests/test_mega_moe.py` 的 `get_baseline_shared_bias`（`:277-302`）：共享
专家 `x → L1 → SwiGLU → L2` 得到 `y`，再作为 **`combine` 的 `bias`** 传入（`:333`
`ep_buffer.combine(l2_y, handle, bias=shared_y)`）——这正是「写进额外槽、被加一次」的 host 对照。

---

## 3. H100 复刻方案

主 kernel 跑不了，但每一段的算子在本机都有等价物：

| 段 | Mega MoE（SM100，仅分析） | 本篇 H100 复刻（SM90，真实 kernel） |
|---|---|---|
| 路由 L1/L2 | `SM100_MMA_MXF8F6F4_2x1SM_SS`，SF gran 32 | `m_grouped_fp8_gemm_nt_contiguous`（1D2D，gran 128） |
| 共享 L1 | 同 UMMA，N=`2·E_s·I` | **dense** `fp8_gemm_nt`（共享权重 2D，无专家维） |
| 共享 L2 | 同 UMMA，K=`E_s·I` | **dense** `fp8_gemm_nt` |
| SwiGLU | bf16 round，路由乘 w、共享不乘 | torch：路由 `silu(g)·u·w`、共享 `silu(g)·u` |
| combine | NVLink 直写 + fp32 相加（共享槽 `kNumTopk`） | torch：路由项 `index_add` + 共享项 `+ y_shared`（一次、无权重） |

**数值口径**：端到端 `fp8-kernel vs fp64`；分层 `kernel+布局` 用「把喂进 kernel 的 FP8 操作数
反量化后在 fp32 重算」隔离（与第 99 篇一致）；共享分支的 dense GEMM 用同样口径单列。

---

## 4. 实验 A：小配置的端到端误差与反例

三组小配置（`S = E_s·I`）：

```
T=  64 H=  512 I= 256 E= 8 topk=2 Es=1 S=  256 | m_total=  1024 共享FLOP占比= 5.9%
   路由+共享 vs fp64: rel=6.3437e-02 cos=0.997983
   分层(各自 vs fp64):  路由=6.2934e-02  共享=6.3838e-02
   分层(kernel+布局 vs 反量化):  路由 L1/L2=1.4705e-03/1.4688e-03  共享 L1/L2=1.4661e-03/1.4733e-03
   反例(共享项多加):  加两次=7.1925e-01  加topk次=7.1925e-01
T= 256 H= 1024 I= 512 E= 8 topk=2 Es=2 S= 1024 | m_total=  1024 共享FLOP占比=33.3%
   路由+共享 vs fp64: rel=6.3025e-02 cos=0.998014
   分层(各自 vs fp64):  路由=6.3376e-02  共享=6.2684e-02
   分层(kernel+布局 vs 反量化):  路由 L1/L2=1.4703e-03/1.4725e-03  共享 L1/L2=1.4709e-03/1.4733e-03
   反例(共享项多加):  加两次=8.2730e-01  加topk次=8.2730e-01
T= 512 H= 1024 I= 512 E=16 topk=4 Es=1 S=  512 | m_total=  3200 共享FLOP占比=13.8%
   路由+共享 vs fp64: rel=6.3060e-02 cos=0.998011
   分层(各自 vs fp64):  路由=6.3381e-02  共享=6.2830e-02
   分层(kernel+布局 vs 反量化):  路由 L1/L2=1.4691e-03/1.4723e-03  共享 L1/L2=1.4690e-03/1.4727e-03
   反例(共享项多加):  加两次=7.1560e-01  加topk次=2.1384e+00
```

读法：

- **两条通路的量化地板一致**：路由 6.29–6.34e-2、共享 6.27–6.38e-2，端到端合计仍在 **6.28–6.34e-2**、
  cos ≥ 0.9979——与第 99 篇的纯路由 6.28–6.34e-2 **完全同量级**。「加不加共享项」几乎不改误差，
  因为**误差由 FP8 量化格式决定**，共享专家和路由专家用的是同一套 per-token/per-128 E4M3 量化。
- **kernel 本身是对的**：共享与路由的 `kernel+布局` 误差都只有 **1.47e-3**（bf16 输出舍入地板），
  说明 6.3e-2 里的 99% 来自量化、不是 kernel 或布局。
- **反例坐实「共享项只加一次、不带权重」**：把共享项多加一次（`route + 2·shared`）→ rel 跳到
  **0.72–0.83**；topk=4 时叠加 `topk` 次 → **2.14**。这与第 44 篇 combine 层的反例（relerr 8.75e-1）
  同一结论，但在**完整前向图**里再验一遍。topk=2 时「加两次」与「加 topk 次」数值相同（都=2），
  故两栏相等——这也是自洽检查。

---

## 5. 实验 B：DeepSeek 量级 + 两条分支的吞吐对照

配置 `T=2048, H=7168, I=2048, E=8, topk=2, E_s=1`（`S=2048`，共享 FLOP 占整层 **31.4%**）：

```
   m_total=4480  S=2048  共享FLOP占比=31.4%
   路由+共享 vs fp64: rel=6.2796e-02 cos=0.998027
   分层: 路由=6.2832e-02 共享=6.2782e-02
   kernel+布局:  路由 L1/L2=1.4714e-03/1.4687e-03  共享 L1/L2=1.4711e-03/1.4692e-03
   共享分支吞吐: L1 T=2048 N=4096 K=7168 t=402us 299 TFLOPS | L2 N=7168 K=2048 t=215us 280 TFLOPS | 激活量化(参考实现) t=859us
   路由分支吞吐(对照): L1 m=4480 N=4096 K=7168 t=844us 312 TFLOPS | L2 N=7168 K=2048 t=428us 308 TFLOPS
   FLOPs: 路由=394.60G  共享=180.39G
```

- 端到端 **6.28e-2**、cos **0.998**，与小配置同量级——误差是**格式属性而非规模属性**。
- **共享分支的吞吐与路由同级**：dense FP8 的 L1/L2 是 **299/280 TFLOPS**，路由 grouped 是
  **312/308 TFLOPS**。共享用稠密、路由用 grouped，但**形状不同、不是同一算账**：路由处理的是
  被 dispatch 打散、按专家分组的 `m_total=4480` 行；共享处理原始 `T=2048` 个 token，但 N/K 更宽
  （L1 N=4096、L2 K=2048）。两者都落在 ~300 TFLOPS，说明这条 H100 上**稠密与 grouped 在
  DeepSeek 量级下吃的是同一档 tensor-core 吞吐**。
- **共享专家的 FLOP 占比 31.4%**（180.4G / 575.0G），与 DeepSeek 把共享专家中间维设成「1 个路由
  专家中间维」（`E_s=1`）时的账一致——**共享专家不是零头，它占整层近三分之一的计算**。这也是
  Mega MoE 必须把它融进同一 kernel、并且**让它的 L1 去填 dispatch 空档**（`scheduler:446-448`）
  的原因：它大到值得专门调度，又因为输入已在本地而**天然可与通信重叠**。

---

## 6. 一个容易误读的数字：859us 的「激活量化」

输出里 `激活量化(参考实现) t=859us` 是**测试侧参考量化器** `per_token_cast_to_fp8`
（`deep_gemm/utils/math.py:26-38`）的开销——它是多趟 torch 实现（pad + `amax` + 除法 + `.to(fp8)` +
`.contiguous()`），**不是** Mega MoE 的量化路径。真机 kernel 的 L1 输出量化是在 epilogue 的寄存器里
就地做的：warp-pair 归约求 amax → `get_ue8m0_sf_exp/inv`（`common/math.cuh:114-147`）→ `__fmul2_rn`
→ STSM（`kernel:1160-1212`，第 10 篇已拆），**没有一趟全局往返**。所以这 859us 不能当成融合 kernel 的
量化成本，只说明「用 torch 参考实现做同一件事有多贵」。第 163 篇量过官方 TileLang SwiGLU 算子
（带权重 + UE8M0）是 **122us/686 GB/s**，量级才接近设备侧。

---

## 7. H100 实测证据台账

| 项 | 数字 |
|---|---|
| 路由+共享 端到端 vs fp64（4 组形状） | rel 6.28–6.34e-2，cos ≥ 0.9979 |
| 分层路由 / 共享 vs fp64 | 6.27–6.34e-2（两条同量级） |
| kernel+布局（路由 L1/L2、共享 L1/L2） | 1.469–1.474e-3 |
| 反例：共享项加两次 / 加 topk 次 | 0.72–0.83 / 2.14（R1 与 R3 配置） |
| 共享分支吞吐（DeepSeek 量级） | L1 299 TFLOPS / L2 280 TFLOPS（402 / 215 us） |
| 路由分支吞吐（同次运行对照） | L1 312 TFLOPS / L2 308 TFLOPS（844 / 428 us） |
| 共享专家 FLOP 占比（E_s=1） | 31.4%（180.4G / 575.0G） |

原始输出：`harness/exp_mega_moe_forward_shared.out.txt`。

---

## 8. 边界与「仅分析」

- **SM100 专属、仅分析**：`sm100_fp8_fp4_mega_moe.cuh` 主 kernel 本体、UMMA/tcgen05/UTCCP/TMEM、
  dispatch/combine 的 NVLink 直写、共享 L1 与 dispatch 的**时间重叠**、packed FP4 路由权重路径。
- **H100 真跑**：SM90 `m_grouped_fp8_gemm_nt_contiguous`（路由）、dense `fp8_gemm_nt`（共享）、
  参考量化器（第 27 篇）、m-grouped contiguous 布局（第 13 篇），以及本篇搭出的**整层前向与三层对拍**。
- **复刻差异**（会体现在误差量级上）：① 真机 SM100 gran 32（`mega_moe.hpp:177-178` 断言
  `rk==32`），本篇 H100 是 **gran 128**——这是数量级估算、不是逐位复现；② dispatch/combine 的
  **通信**被替换成单卡 gather/scatter（通信是精确搬运、不引入舍入）；③ 共享专家的 SF 布局/UTCCP
  映射（第 44 篇）在数值上不改变数学，故本篇不复刻。

---

## 9. 结论

一句话：**把共享专家和路由专家放进同一个 H100 前向里，两条通路的误差都落在同一个 FP8 量化地板
（~6.3e-2）、kernel 与布局各自只贡献 ~1.47e-3，共享项「按 `kNumTopk` 槽、fp32 相加一次、不带
权重」这条语义在完整前向图里被反例坐实；共享专家占整层 ~31% 的 FLOP、吞吐与路由同级（~300
TFLOPS），这正是它值得被融进同一个 mega-kernel 并去填 dispatch 空档的量化依据。** 至此
Mega MoE 的**路由专家 + 共享专家 + combine 三条数据通路**在 H100 上有了同一张图上的数值闭环。

## 参考

- DeepGEMM（`057ca59`，Public release 26/09/30）：`csrc/apis/mega_moe.hpp`、
  `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`、
  `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`、
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、`tests/test_mega_moe.py`。
- 仓库：https://github.com/deepseek-ai/DeepGEMM
- 本专题：`docs/10`（MMA/epilogue，UE8M0 量化）、`docs/20`（BF16 Mega MoE）、`docs/44`
  （共享专家分支）、`docs/99`（前向路由分支的 H100 复刻与误差分解）、`docs/163`（TileLang
  SwiGLU 算子）。
