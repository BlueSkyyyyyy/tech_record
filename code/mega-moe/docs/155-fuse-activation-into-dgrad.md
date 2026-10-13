# 把反向的 SwiGLU 段折进 dgrad 的 epilogue：省掉 d(act) 的一轮全局往返（H100 实测）

> 本专题第 155 篇，路线图 **P145-1**。第 154 篇（P144-1）把「夹在两个 dgrad GEMM 之间的元素级段」
> ——把 L2 dgrad 出来的 `d(act)` 反解成 gate/up 的激活梯度、再量化成 FP8——单独量清楚了：它在
> `T=2048,I=2048` 是 20.1us，在反向 single-step 预算里只占 **1.10%**，并留下一个口子（可继续①）：
> **把这段折进邻近 GEMM 的 epilogue（对照第 148/149 篇的「同 kernel 融合」），量它能不能藏住。**
> 本篇把它做成能核对的字节账 + H100 真机输出侧原型。
>
> **可跑性**：Mega MoE 融合 kernel（UMMA/tcgen05/UTCCP、`epilogue_class` 机制）是 SM100 专属、仅分析；
> dgrad 的**输出侧数据流**（写 d(act) vs 就地反解 + 重量化）架构无关，H100(sm90) 用**真实内核**跑通。
> 实验：`src/moe_backward_activation_fused.cu` + `harness/exp_moe_backward_fused.{sh,py,out.txt}`。
> 全部行号对着 `~/github/DeepGEMM` 真实核对过。

---

## 1. 一句话结论

- **能折、且语义逐位无损**：把 SwiGLU 反向 + 重量化搬进 d2g 的 epilogue，输出与「d2g 落盘 d(act) →
  独立段」**逐位相等**（FP8 字节 0/16 777 216、SF 0/524 288）。
- **省的是 d(act) 的一轮全局往返**：每元素 `子段流量 10 B → 6 B`（-40%），因为 `d(act)` 只被这一小段
  消费、没有第二个消费者（§2）。
- **但整步收益 <0.2%**：这段本就只占反向 1.1%（第 154 篇），折叠相对「d2g + 独立段」的净节省在 H100
  真机原型上是 **~13–15us（SEQ/FUSED 1.057–1.317×）**；放回整步被四个 grouped GEMM（wgrad 占 83%）淹没。
- **它是对工程习惯，不是性能杠杆**：值得做（少一轮 `HBM` 往返、少一个 launch），但不能再往里分配期望。

---

## 2. 为什么可以折：合同

反向图里 `d(act)` 的唯一消费者就是这一小段：

```
d(y) → combine_bwd(=dispatch) → L2 dgrad(d2g 写 d(act))
     → [SwiGLU 反向 + 量化]  ← 本段：读 d(act)、写 FP8 d(gate)/d(up)
     → L1 dgrad(d1g) + L1 wgrad(w1g) + L2 wgrad(w2g) → dispatch_bwd(=combine)
```

`d(act)` 不再被任何别的算子读（wgrad 的 B 操作数、d1g 的 A 操作数都是**激活的梯度** `d(gate)/d(up)`，
不是 `d(act)`）。所以它可以在产生它的 d2g epilogue 里**就地**被消费，不落盘。

三条真实约束（都核对过行号）：

1. **d2g 的输出 D 必是 bf16**（`csrc/apis/gemm.hpp:209` `DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16)`；
   SM90 走 1D2D，`sm90_fp8_gemm_1d2d.hpp:81` `not c and d == bf16`）。现实现的 epilogue 就是一句
   `SM90_TMA_STORE_2D::copy(...)`（`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh:437`）把 smem
   里的 bf16 写回全局。折叠 = 把这句 store 换成「读 gate/up → 反解 → 量化 → 写 FP8 对」。
2. **d(gate)/d(up) 必须是 FP8 + SF**：wgrad 的操作数是 `(data, sf)` 的 FP8 对（`csrc/apis/gemm.hpp:373-411`
   的 `std::pair<Tensor,Tensor>`、recipe `(1,1,128)`），d1g 的 A 操作数同样是 FP8。这正是要重量化的原因。
3. **dgrad 的输出侧和量化语义已经在同一套 `EpilogueClass` 抽象里**：SM100 的 grouped GEMM 已经收
   `epilogue_class` 参数（`csrc/apis/gemm.hpp:242-247` 的 `resolve_epilogue_class(...)`），并且有一个现成的
   **`FP8QuantizationEpilogue`**（`csrc/jit_kernels/impls/epilogue_class.hpp:67-89`，把结果量化成 E4M3 + SFD）。
   即「dgrad 直接出 FP8」这条路在 SM100 上**机制已经存在**；只是还没有一个「SwiGLU 反向」的 epilogue operator。
   SM90 分支明确拒绝自定义 epilogue：`DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100")`
   （`csrc/apis/gemm.hpp:238`；同类断言在 `:119/:308/:385/:528/:611/:664/:712`）。

---

## 3. 每元素字节账（d2g 输出形状 `[Σt=4096, I=2048]`，bf16）

| | 现实现（d2g 写 + 独立段读） | 折叠（d2g epilogue 内） |
|---|---|---|
| d2g epilogue | 写 d(act) 2 B | 读 gate+up 4 B，写 FP8 对 2 B |
| 独立段 | 读 d(act)+gate+up 6 B，写 FP8 对 2 B | —（并入上一行） |
| **合计（d2g 算力之外）** | **10 B/元素** | **6 B/元素** |

折叠省掉 `d(act)` 的「写 2 + 读 2」= **4 B/元素（-40%）**。`gate/up` 两种实现都要读，不是额外负担；
SF 每 32 个元素 1 B/张量（~0.06 B/元素）两边都一样。

换算到 d2g 形状（`n = Σt·I = 8 388 608`）：省 `4 B × 8.39 M = 33.6 MB` 的 DRAM 流量。

---

## 4. H100 真机：输出侧原型

`src/moe_backward_activation_fused.cu` 用**同一份算力**（每元素 `kIters` 次依赖 FFMA，代表 tensor-core
收缩预算、可调）搭两个只差 epilogue 的 kernel：

- `k_prod_fresh`：算出每元素 `dact_val` 后**写 bf16 d(act)**（= 现实现）；
- `k_prod_fused`：`dact_val` 不出寄存器，直接读 gate/up、SwiGLU 反向、per-32 UE8M0 量化、写 FP8 对（= 折叠）；
- `k_seg`：第 154 篇的独立段（读 dact+gate+up、写 FP8 对）。

全部架构无关，H100(sm90) 真机跑。

**（A）正确性**（`T=4096, I=2048`）：`FUSED` 对 `FRESH → 独立段` ——
**FP8 字节不符 0 / 16 777 216、SF 不符 0 / 524 288**。坐实折叠的 SwiGLU 反向 + UE8M0 语义与独立段逐位一致。

**（B）算力扫描**（空闲 GPU，热 cache，50 次均值；单位 us）：

| kIters | fresh（写 2B） | fused（读 4B+写 2B） | fused−fresh | 独立段 | SEQ=fresh+seg | SEQ/FUSED | 净节省 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0（纯访存） | 13.53 | 40.04 | 26.50 | 39.19 | 52.72 | 1.317 | 12.68 |
| 32 | 22.76 | 47.08 | 24.31 | 39.17 | 61.93 | 1.316 | 14.86 |
| 96 | 39.44 | 63.97 | 24.53 | 39.10 | 78.54 | 1.228 | 14.57 |
| 192 | 64.52 | 89.16 | 24.63 | 39.08 | 103.60 | 1.162 | 14.44 |
| 384 | 114.98 | 140.51 | 25.53 | 39.13 | 154.11 | 1.097 | 13.60 |
| 768 | 215.75 | 241.25 | 25.50 | 39.20 | 254.94 | 1.057 | 13.69 |

两个可读的事实：

1. **折叠永远不亏（`SEQ/FUSED ≥ 1`）**，净节省约 **13–15us**：它 ≈ 省掉的 `d(act)` 往返（33.6MB，
   @HBM 峰值约 10us）减去内联开销的零头。这正是 §3 字节账的直接体现，且与算力强弱无关。
2. **融合 epilogue 的边际成本（`fused−fresh ≈ 25us`）几乎不随算力变化**——`kIters 0→768` 时它从 26.5
   只降到 24.3。**在本原型里它没有藏进生产者自己的算力**：因为一个线程是「先做 FFMA 再走 epilogue」**串行**的，
   加更多 FFMA 只是加串行时间。这是**最保守的上界**——真实 GEMM 靠 warp specialization 让 epilogue 与
   下一块的 MMA 重叠（第 143/144 篇），能藏一部分，但藏不藏取决于**该 GEMM 本来有没有带宽/流水余量**，
   与第 149 篇对 wgrad 的结论同源。§5 用真实 d2g 的 ncu 数字看余量。

---

## 5. 真实 d2g 锚点与预算

**真实 SM90 d2g**（`m_grouped_fp8_gemm_nt_contiguous`，`Σt=4096, N=I=2048, K=H=7168, E=8`）：

- event 单次 launch（7 次取最小）：**125.7us / 957 TFLOPS**；
- `ncu` 单次 launch（`--kernel-name regex:sm90_fp8_gemm`，热 cache）：`gpu__time_duration 109.3us`、
  **DRAM 读 180.9 MB / 写 15.3 MB**（合计 196 MB）、**630 784** 条 GMMA。

读法：d2g 在 HBM 上只用了 `196 MB / 109 us ≈ 1.79 TB/s`（H100 峰值 3.35 TB/s 的 **53%**）——**带宽余量约
1.5 TB/s**；同时它是 compute-bound（第 15/142 篇 ncu：SM/tensor pipe 65.5%）。所以它**有余量**在 epilogue
里多读 32MB 的 gate/up（纯读下界约 10us），但 epilogue 是否真与算力重叠、露出多少，取决于实现。

**把两段拼起来**（d2g 形状，现实现 = d2g + 独立段）：

| 方案 | 估计 | 相对现实现 |
|---|---|---|
| 现实现：d2g 126us + 独立段 39us | **~166us** | 1.00× |
| 折叠·最乐观（extra 读全藏） | ~126us（= d2g 算力） | ~1.16× |
| 折叠·最悲观（extra 读全露 @HBM 峰值） | ~126 + 10us = ~136us | ~1.09× |

即使按最悲观，折叠也拿回 **~9–16%** 的「d2g + 激活段」；但这一小段本身只占反向 single-step 预算的
**1.10%**（第 154 篇：d2g 111 + 段 20 + d1g 194 + w2g 500 + w1g 997 = 1823us），**放回整步收益 <0.2%**。

---

## 6. 前提与边界

1. **gate/up 必须先在手**。前向 L1 的 epilogue（`sm100_fp8_fp4_mega_moe.cuh:1108-1219`）是在 TMEM/寄存器里
   **就地**做 SwiGLU（`:1109-1135` 读 `store_block_raw_values`、`:1112-1113` 先 `__float22bfloat162_rn` 入 bf16、
   `:1122-1135` `silu(g)·up·w`），只把 **FP8 激活 + SF TMA 落盘**（`:1219` 之后的 store），**不单独存 gate/up**。
   所以折叠的「读 gate/up」这一步，要么前向**额外 dump** 一份 gate/up（+4 B/元素的前向写），要么反向**重算**
   （又一轮 L1 前向 GEMM）——`§3` 的字节账默认「gate/up 已保存」。这是折叠的**真实前提**，不是免费项。
2. **SM90 不能自定义 epilogue**：`csrc/apis/gemm.hpp:238` 明确拒绝（"Custom epilogues require SM100"）。
   本机 H100 只能靠「独立段」；真实折叠在 SM100（`epilogue_class` 通路）才落地。
3. **SM100 已有 `FP8QuantizationEpilogue`，但没有「SwiGLU 反向」operator**：`epilogue_class.hpp:67-89`
   只做 E4M3 + SFD 量化（且是 recipe 的 per-128 粒度，非本段的 per-32），不含反解。要真折，得新写一个
   epilogue operator（把它当第 31 篇 `epilogue::operators` 家族的新成员），并让 d2g 的 epilogue 能读 gate/up。
4. **本原型是最保守上界**：它不建模 warp specialization 下的 epilogue/MMA 重叠，也不建模真实 GEMM 的
   MMA 流水与 TMA；它量的是「即使完全不重叠，折叠也不亏」这条下界。

---

## 7. 与其它篇的关系

- 段本身的语义、误差分层与预算：第 154 篇（P144-1）。
- 反向四 GEMM 性能归因：第 15/132 篇；wgrad 累积协议系列：第 135~151 篇。
- 「同 kernel 融合 vs 侧流重叠」的判据（`232448 B/SM` 堵死 co-residency）：第 148/149 篇；SM 预算分区：第 144 篇。
- dgrad 三布局与 SM100 `epilogue_class`：第 13/31 篇；SM100 grouped 的 1D1D 通路：第 35 篇。
- 前向 L1 epilogue 的**正向**：第 10/142 篇。

**一句话**：把反向的 SwiGLU 段折进 dgrad 的 epilogue 是**对的、无损的、且永不更差**（省 `d(act)` 的
一轮全局往返、每元素 10→6 B），但它是一段占反向 1.1% 的边角工作，整步收益 <0.2%——**反向的钱在四个
grouped GEMM（wgrad 占 83%）与通信上**，这一段值得顺手折，但不值得为它改内核契约。

---

## 8. 坐标与复现

- 内核：`src/moe_backward_activation_fused.cu`（sha256 见 `.out.txt` 首行）。
- 驱动：`bash harness/exp_moe_backward_fused.sh > harness/exp_moe_backward_fused.out.txt`
  （段一 nvcc + 输出侧原型；段二 `exp_moe_backward_fused.py` 字节账 + 真实 d2g anchor；段三 `ncu` DRAM 字节）。
- 关键行号：`csrc/apis/gemm.hpp:181-250/209/238/241-247/373-411`、
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`、
  `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh:437`、
  `csrc/jit_kernels/impls/epilogue_class.hpp:19/67-89`、
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1108-1219`。
- 关联：第 10/13/15/31/35/132/142/144/148/149/154 篇。
