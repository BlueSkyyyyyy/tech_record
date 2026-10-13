# 把 SwiGLU 反向落地成 SM100 dgrad 的 epilogue_class operator：契约缺口精读 + H100 原型

> 本专题第 156 篇，路线图 **P146-1**。第 155 篇（P145-1）证明了「把反向的 SwiGLU 段折进 dgrad 的
> epilogue」**语义无损、永不更差**（每元素 10→6 B、净省 ~13–15us），但只做了 SM90 的**输出侧原型**，
> 并留下口子①：**在 SM100 上用新写的 `epilogue_class` operator 真正落地**。本篇把它推进到
> 「落地到底要改几处契约」这一层——结论是**只加一个 operator 结构体不够**。
>
> **可跑性**：`epilogue_class` 机制、真实量化 store 分支（`math.cuh`）、契约尺寸账都是架构无关的，
> H100(sm90) 用**真实 DeepGEMM 头文件**跑通；SM100 专属的 TMEM/TMA/UTCCP 部分仅按行号对照。
> 实验：`src/moe_backward_activation_epilogue.cu` + `harness/exp_moe_backward_activation_epilogue.{sh,py,out.txt}`。
> 全部行号对着 `~/github/DeepGEMM`（commit `057ca59`）真实核对过。

---

## 1. 一句话结论

- **能落地，但不是「加一个 struct」**：`epilogue::operators` 家族的算子**只能加行为、不能加状态**
  （`operators.cuh:9-11` + kernel 的 `sizeof` 断言），而 SwiGLU 反向需要**逐元素输入 gate/up** 和
  **两个输出 + 两个 SFD**——签名对不上。
- **H100 实测：真实量化 store 分支与被测语义逐位一致**——用真实 `common/math.cuh` 复刻
  `sm100_store_cd.cuh:144-198` 的 FP8 分支，套到「dact → SwiGLU 反向 → dgate/dup」上，对**手写独立参考**
  `FP8 字节不符 0 / 16 777 216`、`SF 不符 0 / 524 288`（§4）。
- **契约缺口是可量化的**：`EpilogueOperatorArgs` 24 B → 提议扩展 64 B（+40 B，**每个** SM100 GEMM 都涨）；
  单输出分支下 `d(up)` 非零元素 **0 / 8 388 608**（不是精度问题，是第二个张量根本写不出去）。
- **收益仍是第 155 篇那句**：这段只占反向 1.10%，落地代价（改 3 处契约 + 扩全部 GEMM 的参数）**远大于**收益。

---

## 2. `epilogue_class` 的完整接线（真实行号）

SM100 的每个 GEMM 都**恰好带一个** epilogue class，它是「host 对象 → device 算子」的桥：

| 层 | 位置 | 作用 |
|---|---|---|
| host 基类 | `epilogue_class.hpp:19-39` | `get_epilogue_operator_type()`（类型字符串）、`make_epilogue_operator_args()`（运行状态）、`check()`、`output_sfd()` |
| 4 个实现 | `:41-48` Identity / `:50-65` Alpha / `:67-89` FP8 量化 / `:91-106` 随机舍入 | 每个返回一个不同的类型字符串 |
| 解析 | `:111-127` `resolve_epilogue_class` | `epilogue_class / alpha / sfd` **三者互斥**，缺省 Identity |
| JIT 注入 | `sm100_bf16_gemm.hpp:30-53` 模板体 + `:69` 实参 | 把类型字符串 `std::format` 进 kernel 模板实参 |
| 状态传递 | `sm100_bf16_gemm.cuh:39`、`sm100_fp8_fp4_gemm_1d1d.cuh:41` | operator 以 `__grid_constant__` 参数传入 |
| device 算子 | `operators.cuh:12-92` | `Identity` / `ScaleByAlpha` / `StochasticRoundToBF16` / `QuantizeToFP8` |
| store | `sm100_store_cd.cuh:22-218` | 真正读 TMEM、量化、写 TMA |

**关键点**：host 只提供一个**类型字符串**（`epilogue_class.hpp:23`、`:43/56/73/93`），device 只提供**行为**
（`operators.cuh` 的 `apply_values`/`store_sf`）。H100 实测（`.py` §①②）4 个类型字符串与 `operators.cuh`
的 struct **1:1 全对上**——所以「新增算子」的 host 唯一接线点就是多返回一个字符串。**但**这只是一半。

---

## 3. 契约的三个缺口（为什么落不了地）

### 3.1 「不得加状态」——输入指针只能塞进共享基类

kernel 里有硬断言（`sm100_bf16_gemm.cuh:59-60`、`sm100_fp8_fp4_gemm_1d1d.cuh:57-58`）：

```
DG_STATIC_ASSERT(sizeof(epilogue_operator_t) == sizeof(EpilogueOperatorArgs),
                 "Epilogue operators must not add state to `EpilogueOperatorArgs`");
```

`operators.cuh:9-11` 的注释也明说「add behavior (never state)」。SwiGLU 反向要读 `gate`/`up`、写第二个
`d(up)`，这些**必须**进共享的 `EpilogueOperatorArgs`（`types.cuh:47-52`）——于是**所有** SM100 GEMM 的
kernel 参数一起变胖。H100 实测（`.cu` §A）：

| | 字节数 |
|---|---|
| 现状 `sizeof(EpilogueOperatorArgs)` | **24 B** |
| 提议扩展（+ gate/up/weights/dup 指针） | **64 B**（+40 B） |

这 40 B 通过 `cuLaunchKernelEx` 的 `__grid_constant__` 参数传入，**不是免费的**：它落在**每个** kernel
（稠密/m-grouped/k-grouped/batched，SM100 的全部 GEMM）的 launch 属性上。

### 3.2 「`apply_values` 没有索引上下文」

`operators.cuh:12-15/22-29` 的算子只拿得到 `apply_values(uint32_t (&values)[kNumValues])`——**一串寄存器里的
累加器值、没有 `(row, n)`**。而 SwiGLU 反向必须按元素取 `gate[row,n]`、`up[row,n]`（否则算错）。
唯一带索引的是 **store 路径**：`sm100_store_cd.cuh:101` 的 `row_idx`、`:91` 的 `n_idx`，以及
`operators.cuh:78-91` 的 `store_sf(row_idx, group_n_idx, ...)`。

所以这个算子**无法**只用 `apply_values` 实现——它必须**改 store 路径**（`sm100_store_cd.cuh`），
不像 `ScaleByAlpha` 那样只改寄存器。

### 3.3 「单输出 + 单 SFD」——SwiGLU 反向天生是两输出

`EpilogueClass` 只表达**一个** D（一个 `tensor_map_cd`，`sm100_store_cd.cuh:32`）+ 可选**一个** SFD
（`output_sfd()` `epilogue_class.hpp:38`）。FP8 输出对还强制 `not c`（`epilogue_class.hpp:84`）——连 C 都被占。
而 SwiGLU 反向要出 `d(gate)` 与 `d(up)` **两个** FP8 张量 + **各自**的 SF（第 155 篇 §3）。

H100 实测（`.cu` §C/C2）：

- `d(gate)` 与 `d(up)` 的 per-32 SF **分布不同**：若强行共用一个 SF，`d(up)` 的等效比例偏差
  `RMS = 2^(e_dgate − e_dup) − 1 = 0.8506`；
- 走「单输出」分支时，`d(up)` **非零元素 0 / 8 388 608**——第二个张量根本不会被写。

**结论**：要落地得扩**三处**契约：`EpilogueOperatorArgs`（加指针）、`resolve_epilogue_class`/`check`
（支持双输出），以及 `sm100_store_cd.cuh` 的 `is_same_v` 分支（加一条双写，且要第二个 tensor map
或第二个 TMA descriptor）。这已经不是「加一个 operator」，而是「扩 epilogue 协议」。

---

## 4. H100 真机：真实 store 分支的逐位对拍

`src/moe_backward_activation_epilogue.cu` **包含真实** `deep_gemm/common/types.cuh` 与
`deep_gemm/common/math.cuh`，把 `sm100_store_cd.cuh:144-198` 的 FP8 分支**逐行**复刻成一个
`store_quant_group_real()`：

- `:147-150` `math::cast_into_bf16x2` 先把累加器 round 成 BF16 对；
- `:154-155` `math::get_packed_bf16_amax<16>` → `math::get_ue8m0_sf_exp` → `math::get_ue8m0_sf_inv`；
- `:158-168` `math::scale_bf16x2_into_fp8x4` 打包成 FP8x4。

把它套在「dact → SwiGLU 反向 → dgate/dup」上（`T=4096, I=2048`，即 d2g 输出形状），再与一份
**独立手写**的参考段（自己实现 UE8M0 位运算 + `__nv_fp8_e4m3` 转换，第 144/145 篇那套）对拍：

```
[B] FP8 字节不符 0 / 16777216（dgate+dup 两输出），SF 不符 0 / 524288
```

这份对拍证明两件事：① 真实 DeepGEMM 的量化 store 分支，与我们前几篇手推的段语义**逐位相同**；
② 提议算子的算术（把 store 分支接在 SwiGLU 反向之后）**可以直接复用现成的 device 原语**，
不用新写量化代码——真正的工作量在**契约**（§3），不在数值。

（SM100 专属部分：TMEM 的 `SM100_TMEM_LOAD_32dp32b32x` 读、TMA store、UTCCP 转置、以及
`sm100_store_cd.cuh:203-213` 的 TMA 发射，本机 sm90 跑不了，仅按行号对照。）

---

## 5. 落地成本 vs 收益

第 155 篇已把收益钉死：这段占反向 single-step 预算的 **1.10%**，折叠整步收益 **<0.2%**。本篇补上成本：

| 改动点 | 成本 |
|---|---|
| `EpilogueOperatorArgs` 加 5 个指针 | 所有 SM100 GEMM 的 kernel 参数 24→64 B |
| `resolve_epilogue_class`/`check` 支持双输出 | 与现有「三输入互斥」语义冲突，要重新设计 |
| `sm100_store_cd.cuh` 加双写分支 | 要第二个 tensor map / TMA descriptor；`is_same_v` 硬编码分支再加一条 |
| 前向 dump gate/up 或反向重算 | 第 155 篇 §6.1：+4 B/元素 的前向写，或一轮 L1 前向 GEMM |

**一句话**：把 SwiGLU 反向做成 epilogue operator 在**数值上完全可行**（真实原语逐位可复现），
但它要把一个「单输出、无状态、无索引」的通用 epilogue 协议改成「双输出、带逐元素输入」——
为一段占反向 1.1% 的边角工作付这个协议成本不划算。**若真要做，正确的做法不是扩 epilogue，
而是把 SwiGLU 反向留在独立的元素级 kernel 里**（第 154 篇那条路），它已经被证明无损且便宜。

---

## 6. 与其它篇的关系

- 段本身的语义、误差分层、预算：第 154 篇（P144-1）；
- 折叠的字节账与 SM90 输出侧原型：第 155 篇（P145-1）；
- `epilogue::operators` 家族与随机舍入：第 31 篇（P21-1）；
- SM100 的 `epilogue_class` / 1D1D 通路：第 27/35/26 篇；
- dgrad 的三布局与 m-grouped 入口：第 13/142 篇。

---

## 7. 坐标与复现

- 内核：`src/moe_backward_activation_epilogue.cu`（sha256 见 `.out.txt` 首行）。
- 驱动：`bash harness/exp_moe_backward_activation_epilogue.sh > harness/exp_moe_backward_activation_epilogue.out.txt`
  （段一 nvcc 编真实头 + 设备报告；段二 host 接线 Python）。
- 关键行号：`csrc/jit_kernels/impls/epilogue_class.hpp:23/38/67-89/83-86/111-127`、
  `deep_gemm/include/deep_gemm/epilogue/operators.cuh:9-15/20-30/65-92`、
  `deep_gemm/include/deep_gemm/epilogue/sm100_store_cd.cuh:32/35/38/95-102/115-124/144-198`、
  `deep_gemm/include/deep_gemm/common/types.cuh:47-52`、
  `deep_gemm/include/deep_gemm/common/math.cuh:75-147`、
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:41/57-58/62/497-560`、
  `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:39/59-60`、
  `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:30-53/69-70`、
  `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:79/85`、
  `csrc/apis/gemm.hpp:209/238/242-247/314`。
- 关联：第 10/13/26/27/31/35/142/144/145/154/155 篇。
