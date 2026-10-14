# 重算能不能藏进 shadow：单 kernel 融合下的**资源账**（张量核 vs 访存）

> 第 182 篇（P172-1）把「反向重算 linear1 能否藏进 shadow」放到真机并发下量了：**两个 kernel + 两条
> stream 兑现不了**——满 SM 双 stream 退化成串行（`overlap_factor=0.985×`），把 SM 当预算分区后能重叠、
> 但限核损失把净收益压到最好 `1.09×`，远达不到预算隐含的 `1.49×`。结论是「要靠**单 kernel 内让异步
> 张量核去吃访存受限腿闲着的张量核**」。
>
> 本篇把这个结论落成一个**可核对的资源问题**：shadow 不是一个整体，它由 **dgrad L2（计算受限）** 和
> **wgrad L2（访存受限）** 两段拼成；重算本身也是**计算受限（吃张量核）**。于是问题变成：
> **单 kernel 里，重算的张量核能不能被「访存腿」吸收、却会被「第二条计算腿」挡住？**
> 答案用一条单 kernel、两 warpgroup 的 H100 微基准给出：**访存腿 × 计算腿重叠效率 `0.88`，计算腿 ×
> 计算腿只有 `0.33`**——张量核是共享资源，只有 shadow 的访存段（wgrad）能被重算免费搭车。

**约束**：Mega MoE 融合主 kernel 是 **SM100 专属**，本机 H100(sm90) 跑不了；本文只取它**架构无关**的
原语（真实 SM90 WGMMA、访存 RMW、`setmaxnreg`/warp specialization 的机制语义）。数值实验全部在
H100(sm90a) 真机完成（WGMMA 选择器、描述符均用 DeepGEMM 真实头文件）。目标卡代号敏感：全文用「目标卡」。

---

## 1. 先把「重算 × shadow」拆成资源账

第 182 篇在 `T=4096, H=7168, I=2048, E=8, topk=2`（DeepSeek 量级）下量到三段（满 132 SM、单流串行）：

| 段 | 时长 | 吞吐 | 受限类型 |
|---|---|---|---|
| RECOMPUTE（前向 L1 routed，`m_grouped`） | `356.7 us` | 1349 TFLOPS | **计算受限**（张量核） |
| SHADOW① dgrad L2（`m_grouped`） | `193.3 us` | 1244 TFLOPS | **计算受限**（张量核） |
| SHADOW② wgrad L2（`k_grouped`） | `535.9 us` | 449 TFLOPS | **访存受限**（第 15 篇 ncu：SM 占用 ~12%） |

串行三段 `1085.9 us`，第 182 篇里面向的「预算」`max(RECOMPUTE, wgrad) + dgrad = 729.2 us`（`1.49×`）
其实已经暗含一个假设：**重算能藏进整个 shadow**。但 shadow 的两段**资源不同**——dgrad 是计算受限，
和重算抢同一块张量核；只有 wgrad（访存受限、张量核大量空闲）才可能让重算搭车。

> 这正是第 182 篇那句「没让重算去吃 shadow 闲着的张量核」的精确版本。区别是：**shadow 里只有 wgrad
> 段有闲着的张量核**，dgrad 段没有。

要验证这条资源判断，最干净的办法是**把两条腿放进同一个 kernel、两个 warpgroup**（同 SM）。
这也是 Mega MoE 主 kernel 的组织方式：512 线程切成 dispatch / non-epilogue / epilogue 三组，
各自寄存器预算编译期钉死（`sm100_fp8_fp4_mega_moe.cuh:329-334` 的 `kNumDispatchRegisters=48/40/208`
（或 `96/72/168`）与 `DG_STATIC_ASSERT(... <= 64512)`），运行时用
`cutlass::arch::warpgroup_reg_dealloc<...>()`（`:344`）让出、epilogue 再 `warpgroup_reg_alloc<...>()`
（`:948`）收回。

---

## 2. 实验设计：单 kernel、两个 warpgroup、三条腿

源码 `src/recompute_shadow_fused.cu`，H100(sm90a) 真机，grid = `num_sms`（132）、block = 256 线程
（1 个 compute warpgroup + 1 个「第二腿」warpgroup）。三条腿都是架构无关原语：

- **COMP（proxy 重算 / 计算腿）**：一个 warpgroup 反复发**真实 WGMMA**
  （`64×128×64` BF16，`BF16MMASelector`，`mma/sm90.cuh:104`；描述符 `make_gmma_desc`，`:245`；
  异步组 `warpgroup_arrive/commit_batch/wait`，`ptx/wgmma.cuh:7/11/20`）。
- **TENS（proxy dgrad L2 / 第二条计算腿）**：第二个 warpgroup 发**第二组 WGMMA**（同一张量核）。
- **MEM（proxy wgrad L2 / 访存腿）**：第二个 warpgroup 对一个 512 MiB 的 global buffer 做
  **read-modify-write（RMW）**——读 + 写，proxy「读两个操作数 + 落盘权重梯度」。

三条腿都用「二分标定」调到**单跑时长相等**（≈ 346–349 us），于是重叠效率可以对称地读：
`eff = (tA+tB)/t_fused` 归一化到「理想 `max(tA,tB)` = 1、串行 `tA+tB` = 0」。
融合 kernel 用模板开关 `kDoComp`（WG0 是否做 WGMMA）与 `kLeg`（WG1 = idle/1=MEM/2=TENS）
在同一次 run 里量出「单跑」与「融合」两组。

编译产物（`-Xptxas -v`）：所有含 WGMMA 的形态都是 `90 registers`、`0 spill`；纯访存形态 `16 registers`。

---

## 3. 结果：访存腿能吸收计算腿，第二条计算腿不能

`harness/exp_recompute_shadow_fused.sh`（H100 真机，`CUDA_VISIBLE_DEVICES=1` 避开被占用的 GPU0）：

```
  计算腿 tile = 64x128x64 BF16 WGMMA，reps=5434；MEM 腿遍历 167.0 MiB
  COMP (WGMMA/张量核)   only =    345.7 us
  TENS (WGMMA/张量核)   only =    345.7 us
  MEM  (RMW/访存)       only =    347.6 us  (1007.4 GB/s 读+写)
  COMP + MEM   fused    =    370.3 us  eff=0.877  (ideal=max(345.7,347.6), serial=693.3)
  COMP + TENS  fused    =    518.6 us  eff=0.333  (ideal=max(345.7,345.7), serial=691.4)
```

两条对照读起来非常干净：

- **计算腿 × 访存腿**（COMP+MEM）：融合后 `370.3 us`，几乎等于两条腿里较长的那条（`347.6 us`），
  重叠效率 **`0.877`**——两个 warpgroup 用的是**不同资源**（张量核 vs LSU/HBM），互不排队。
- **计算腿 × 第二条计算腿**（COMP+TENS）：融合后 `518.6 us`，明显高于 `max=345.7`、且不足串行 `691.4`，
  重叠效率只有 **`0.333`**——两个 warpgroup **抢同一块张量核**，只能部分交错。

> `0.333` 不是 0，说明本基准里的单 warpgroup 并没有 100% 占满张量核（`64×128×64` 的 WGMMA 有发射/
> 排空间隙），第二条计算腿能填掉约 1/3 的孔隙。真实的 dgrad L2 跑到 1244 TFLOPS、更接近张量核上限，
> 所以真实情形下「重算 × dgrad」的冲突只会比 `0.333` **更重**（能填的孔隙更少）。

**结论一句话**：**shadow 里 dgrad 段是「哑」的**——它和重算抢张量核、无法吸收重算；**wgrad 段是
「哑铃」**——它把张量核让出来，重算几乎可以完全搭车。第 182 篇的 `1.49×` 预算之所以能成立，根子在于
`RECOMPUTE(356.7) < wgrad(535.9)`，重算恰好能塞进 wgrad 的时长里。

---

## 4. 把实测效率代回预算：`1.43× vs 1.09×`

把 `0.877` 这个**同资源无关**的重叠效率代回第 182 篇的三段真实时长（只让重算藏进 wgrad、dgrad 串行）：

```
  串行 3 段          = 1085.9 us
  理想（rec 全藏进 wgrad）= max(rec,wgrad)+dgrad = 729.2 us  → 1.49x
  用实测 eff=0.877    → 756.9 us  → 1.43x（两 kernel 两 stream 实测仅 1.09x）
```

- 单 kernel 融合能拿回 **`1.43×`**，几乎顶到预算的 `1.49×`（差的那点就是「真实张量核比微基准更饱和」）。
- 对照第 182 篇的**两 kernel 两 stream**：同一份工作最好 `1.09×`——**融合把 `1.09×` 拉到 `1.43×`**。
- 差别的根因也清楚了：两 stream 只能把 SM **切成两组**（再把吞吐各切一半，且受 `smem_capacity=232448`
  per-SM 限制、`heuristics/sm90.hpp:14`，block 无法 co-reside），而单 kernel 是让**同一 SM 上**的访存腿
  与计算腿并行——前者动 HBM、后者动张量核，互不抢。

> 这里**没有**建议「把 dgrad 也硬塞进同一个 kernel 去和重算并行」——`COMP+TENS` 的 `0.333` 说明那是
> 赔本。正确做法是**调度顺序**：让重算在 wgrad 段执行期间冲进去，dgrad 段该串行就串行。这正是
> Mega MoE 主 kernel 里 dispatch（NVLink 访存）/ non-epilogue（发 UMMA）/ epilogue（访存）三类 warp
> 并存的道理：**它们分属不同资源池**，而不是「塞进同一个资源池里抢」。

---

## 5. 与目标卡的对应：为什么这是 SM100 的根

单 kernel 融合要成立，计算腿必须是**异步张量核**——第 167 篇（P157-1）已经量过：同样「1 comm WG +
1 compute WG」，计算腿若是 CUDA-core FMA，重叠效率只有 `0.62`；换成真实 WGMMA 后升到 `0.93`。
本篇把「comm 腿」换成「访存腿」、再加一条「第二条计算腿」做隔离，得到同一结论的两半：

- 异步张量核腿（WGMMA）与访存腿可以并行（`0.877`，本篇）——所以 Mega MoE 的 dispatch（访存）、
  combine（访存）能和 UMMA 计算重叠；
- 两条张量核腿互相排队（`0.333`，本篇）——所以不能指望「重算 + dgrad」单 kernel 内双赢。

在目标卡上，这套单 kernel 由一个 `512` 线程、`cluster=2` 的 persistent grid 承载，任务由
`scheduler/mega_moe.cuh` 按 **locality domain** 分域切分（`get_next_local_task_idx` 与偷取门槛
`kNumMinStealWaves=2`，`:225`／`:344-370`），host 侧用 `set_num_sms`（`csrc/runtime/runtime.hpp:78-84`）
控制每 kernel 的 SM 预算。本篇只验证了「两腿并行的资源判据」，把这些机制串成整条融合流水线仍是
**SM100 专属、仅分析**。

**边界与诚实说明**：
- COMP/TENS 用**单个** `64×128×64` WGMMA warpgroup proxy 真实 GEMM，未复刻 UMMA/TMEM/多 stage 流水，
  故 `COMP+TENS` 的 `0.333` 是**下界式**估计（真实张量核更饱和、冲突更重）；
- MEM 腿是纯 RMW 流式访存，proxy wgrad 的「读操作数 + 写权重梯度」流量形态，不含分组切块；
- 三条腿等长是为了对称读重叠效率；真实三段的时长比（`356.7:193.3:535.9`）在 §4 的预算里代回。

---

## 6. 参考与复现

- 实验：`harness/exp_recompute_shadow_fused.sh` → `harness/exp_recompute_shadow_fused.out.txt`；
  源码 `src/recompute_shadow_fused.cu`（真实头文件
  `deep_gemm/include/deep_gemm/mma/sm90.cuh` sha256 `548fe0c6…`、
  `.../ptx/wgmma.cuh` sha256 `dad341b3…`，与本专题第 39/167 篇一致）。
- 上游源码行号（DeepGEMM `057ca59`）：`sm100_fp8_fp4_mega_moe.cuh:329-334/344/948`（寄存器预算分区）、
  `mma/sm90.cuh:104/245`（`BF16MMASelector` / `make_gmma_desc`）、`ptx/wgmma.cuh:7/11/20`（异步组）、
  `scheduler/mega_moe.cuh:225/344-370`（分域 + 偷取门槛）、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:125`
  （`grid_dim = num_sms`）、`csrc/jit_kernels/heuristics/sm90.hpp:14`（`smem_capacity=232448`）、
  `csrc/runtime/runtime.hpp:78-84`（`set_num_sms`/`get_num_sms`）。
- 前序：第 167 篇（单 kernel warp specialization 原型，第 182 篇的前置）、第 181/182 篇（存 vs 重算、
  两 kernel 两 stream 的 `1.09×`）、第 148 篇（侧流被 `232448 B/SM` 堵死）。
- 线上资料：DeepGEMM [PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316)（Mega MoE 融合 kernel）。
