# 第 167 篇：Mega MoE 的「单 kernel·warp specialization·寄存器预算分区」——H100 上做得出原型吗？

> 本专题前面把 Mega MoE 的通信-计算重叠机制拆得很细（第 2/9/10/11/165 篇），但一直缺一环：
> **它凭什么能把「通信」和「计算」放进同一个 kernel、同一块 SM，还互不拖累？** 第 144 篇量到
> 「NCCL 侧流与 GEMM 不重叠」，第 165 篇量到「micro-batch 流水线里 NCCL 的通信腿只被藏住一小部分，
> 但只要给通信一个**小 SM 预算**，重叠就到 ~99%」——而那个正对照用的仍是**两个 kernel 靠硬件
> co-residency**。Mega MoE 真正用的是**第三个方案**：同一个 grid、同一个 CTA 里用 warp specialization
> + **动态寄存器重分配**（`setmaxnreg`）把通信段和 MMA 段塞进同一块 SM。
>
> 本篇把这条路在 **H100(sm90a) 上做成可跑原型**，回答三件事：① 寄存器预算分区是不是可测的机制；
> ② 单 kernel 内融合到底比「两 kernel co-residency」好还是差；③ 如果差，**缺的到底是什么**——
> 答案指向 **异步张量核**，也正是 Mega MoE 只能长在目标卡上的原因。
>
> **边界**：Mega MoE 融合 kernel 本体是 SM100 专属（`mega_moe.hpp` 只在 arch10 派发），本篇对它**仅分析**；
> 可跑的只有架构无关的 `setmaxnreg` / warp specialization / NVLink peer copy / WGMMA 原语。

---

## 1. 真实 Mega MoE 的 warp 与寄存器分区（源码精读）

主体 kernel `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` 用
`__launch_bounds__(kNumThreads, 1)` 启动（`:56`），并把线程**静态切成三段角色**：

- dispatch warps（`warp_idx < kNumDispatchWarps`，`:342`）；
- MMA non-epilogue warps（TMA 载入 / leader CTA 的 mainloop，`:690/756/819/939`）；
- epilogue + combine warps（`:946`）。

线程数由 host 侧启发式给出：dispatch **128**、non-epilogue **128**（`csrc/jit_kernels/heuristics/mega_moe.hpp:220-221`）、
epilogue **2×128 = 256**（`.../mega_moe.hpp:126` 的返回 `2 * 128`），合计 **512 线程/CTA**。kernel 侧还对这些数
加了整除断言（`:90-92`：dispatch%128、non-epilogue==128、epilogue%128）。

关键在 `:326-335` 这段**寄存器预算**：

```
constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
constexpr uint32_t kNumDispatchRegisters    = kUseMoreEpilogueRegisters ?  48 :  96;
constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ?  40 :  72;
constexpr uint32_t kNumEpilogueRegisters    = kUseMoreEpilogueRegisters ? 208 : 168;
DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                 kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                 kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                 "Too many registers");
```

三件事值得注意：

1. **分工不均**：epilogue/combine 段要 **208** 个寄存器（它把大块累加/隐藏维结果放在寄存器里搬，见
   `kNumMaxRegistersForBuffer = 128` 的注释 `:1372`），dispatch 段只要 **48**，MMA non-epilogue 只要 **40**。
2. **总预算被钉死在 64512**（= 65536 − 1024，给硬件留一点），并且是**编译期静态断言**——也就是说这个
   划分不是「建议」，是 kernel 能否成立的前提。
3. 光有预算还不够，**运行时**用 `cutlass::arch::warpgroup_reg_dealloc/alloc` 真正把寄存器从通信段手里
   收回来、发给计算段：dispatch `dealloc<kNumDispatchRegisters>`（`:344`）、non-epilogue
   `dealloc<kNumNonEpilogueRegisters>`（`:692/758/821/941`）、epilogue `alloc<kNumEpilogueRegisters>`
   （`:948`）。

BF16 变体 `sm100_bf16_mega_moe.cuh` 用同一套骨架，只换了数值（`:284-289`：`{48,40,208}` 或 `{96,88,160}`，
预算断言同款）。Mega mHC 也是同一个 64512 闸门（`sm100_mega_mhc.cuh:98`
`DG_STATIC_ASSERT(kMaxRegisterBudget * 128 <= 64512, ...)`）。可见**「按 warp 角色分区寄存器」是这一族
mega-kernel 的公共地基**。

这个机制不是 DeepGEMM 独有。DeepEP 的 `hybrid_combine.cuh` 把 warp 分成 scale-up(NVLink) 与
forward(RDMA) 两组，各给 **40 / 216** 个寄存器（`:103-104`
`kNumRegistersForScaleupWarps = 40`、`kNumRegistersForForwardWarps = 256 - 40`），在
`:111-112` 与 `:357-358` 分别调用 `ptx::warpgroup_reg_realloc<...>()`。底层就是
`deep_ep/include/deep_ep/common/ptx.cuh:456-469`：

```
constexpr int kNumInitialRegisters = (65536 / kNumThreads / 8) * 8;
if constexpr (kNumRegs >= kNumInitialRegisters)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
else
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
```

注释写得很直白：*「Initial register adjustment assuming the budget for `__launch_bounds__(kNumThreads, 1)`」*——
即先用 launch bounds 定下 CTA 的总预算，再用 `setmaxnreg` 在 warpgroup 之间做动态再分配。

**一句话**：Mega MoE 的「通信-计算重叠」不是靠两个 kernel 抢 SM，而是靠**单 kernel 内、按角色把寄存器
预算切好、让通信 warp 瘦下来、计算 warp 胖起来**，从而在**同一块 SM** 上共存。

---

## 2. H100 能跑吗？——能跑「机制」，跑不了「本体」

- 跑不了：`fp8_fp4_mega_moe` / `bf16_mega_moe` 融合 kernel 本体（SM100 tcgen05 UMMA + TMEM +
  symmetric memory），本机 sm90 编译即拒（`mega_moe.hpp` 只在 arch10 派发）→ **仅分析**。
- 跑得了：`setmaxnreg.inc/dec.sync.aligned.u32`、warp specialization、NVLink peer copy、以及
  **SM90 的异步张量核 WGMMA**——这些都与架构相关但 sm90a 上都有。原型因此可以**用真实原语、真实设备代码**
  量出「同一 CTA 内通信 warp + 计算 warp 的重叠能力」。

原型 `src/mega_moe_like_ws.cu`（A 部分）与 `src/mega_moe_like_ws_tc.cu`（B 部分）：

- 单进程 8 卡（本机 GPU0 被其它自驻任务占满，故用 `-DWORLD=7` + `CUDA_VISIBLE_DEVICES=1..7` 避开它）；
- **通信腿**：把本 rank 的一段数据用 NVLink 远端写（`uint4` peer store）散射到各 peer 的 recv
  （与第 155 篇同款），每 rank 168 MB；
- **计算腿**：把一块 SM 的算术打满。A 部分用 CUDA-core FMA（`kAcc=144` 条独立累加链，约需 152 个
  寄存器——故意做成「吃寄存器」，复现 epilogue warp 需要 >128 的处境）；B 部分用**真实 WGMMA**
  （`deep_gemm/mma/sm90.cuh` 的 `BF16MMASelector` + `ptx/wgmma.cuh`，64×64×64/次）；
- 每 CTA = 1 个 comm warpgroup + 若干 compute warpgroup，通信段 `dealloc` 到 40、计算段 `alloc` 到
  目标值（与 Mega MoE 的 40/208 同构）。

三种对照方案：

| 方案 | 说明 |
|---|---|
| `serial` | 通信 kernel → 计算 kernel（单流，无重叠） |
| `cores_nat` / `cores_lim` | 通信 kernel ∥ 计算 kernel（两条流；`nat`=自然网格、`lim`=限块，第 155 篇正对照） |
| `fused_ws` | **单 kernel**，每 CTA 内 comm WG + comp WG 并发，寄存器按角色分区 |

---

## 3. 实验 A：寄存器分区是可测的机制（ptxas 会认）

先证「`setmaxnreg` 不只是运行时的事，**编译器看得见**」。把 512 线程、1 个 comm WG + 3 个 compute WG、
计算路径 `kAcc=144`（约需 152 个寄存器）的融合 kernel，用「有/无 `setmaxnreg`」两个模板实例分别编译，
`ptxas -v` 给出：

| 实例 | 有无 `setmaxnreg` | 计算段寄存器 | spill |
|---|---|---|---|
| 512 thr（A 形态） | 有 | 128（launch 计数） | 0 |
| 256 thr（B 形态） | 有 | **216** | 0 |
| 256 thr（B 形态） | **无** | 156 | 0 |

也就是说：**当 kernel 里出现 `setmaxnreg.inc<216>` 时，ptxas 会把计算 warpgroup 的寄存器预算按 216 来
排指令**；没有它，编译器只给 156，计算段就慢一倍。运行时实测（H100 真机，7 卡，168 MB/rank）：

```
fused B 计算腿：setmaxnreg 开 = 435.4 us (62.1 TFLOPS)
                setmaxnreg 关 = 876.4 us (30.8 TFLOPS)   off/on = 2.01×
```

在更极端的形态（512 线程、`kAcc=200`、完全复刻 Mega MoE 的 `48/40/208` 划分）下，**无 `setmaxnreg` 会
产生 2300 字节 spill 存储**、有则 **0 spill**——差距扩大到 ~10×。这正是为什么 Mega MoE 要把
`kNumEpilogueRegisters=208` 写进编译期预算并用 `alloc` 拿回来：**没有分区，epilogue 段就得 spill，
kernel 直接废掉**。

---

## 4. 实验 B：单 kernel 融合 vs 两 kernel co-residency

固定通信量、扫计算量，每方案用**自己的腿**算重叠效率
`eff = ((tc+tg)/t − 1) / ((tc+tg)/max(tc,tg) − 1)`（`eff=1` 理想、`0` 完全串行），H100 真机：

```
scale   | serial  | cores_nat | cores_lim | fusedA | fusedB
0.5     |  0.60   |   1.87    |   1.05    |  0.31  |  0.58
1.0     |  0.31   |   1.39    |   1.09    |  0.26  |  0.60
2.0     |  0.23   |   0.48    |   0.99    |  0.30  |  0.53
```

（`cores_nat` 的 `eff>1` 是共享节点噪声/腿测量偏差——它的**总量**甚至低于自己的通信腿孤立值，
说明孤立腿那次撞上了别的任务抢卡；`fused` 与 `cores_lim` 的对比在三次运行里方向一致。）

**结论**：在这台机器上，**单 kernel 内融合（fused）只重叠了 ~30–60%，而把通信、计算拆成两个
限块 kernel 靠 SM 预算做空间隔离（cores_lim）能到 ~100%**。原因不难理解——fused 让通信 warp 和
CUDA-core 计算 warp 挤在**同一块 SM** 的发射/LSU 通路上：`fusedB` 通信腿单独跑 700–730 us
（215 GB/s，比独立通信 kernel 的 ~190 GB/s 还快），但一旦计算腿开跑（同为几百 us），总量就掉到
「通信 + 部分计算」，两者互相拖累。

这与第 165 篇的结论并不矛盾，反而互补：**重叠本身可行（限块就能到 ~99%），但「把两者塞进同一块 SM」
这一步在 H100 的 CUDA-core 计算上是有代价的。**

---

## 5. 实验 C：缺的是**异步张量核**

如果「同 SM 共存」是问题，那 Mega MoE 凭什么敢这么做？关键差别在**计算用什么执行单元**：
Mega MoE 的首条 UMMA 是 `SM100_MMA_MXF8F6F4_2x1SM_SS` / `SM100_MMA_F16BF16_...`（第 10 篇已拆），
是**异步张量核**——它由张量核流水线执行，**不占 CUDA core 的发射槽和 LSU**。因此 dispatch warp
可以真正并行地推 NVLink。

H100 上对应的异步张量核是 **WGMMA**。B 部分用真实 `BF16MMASelector` + WGMMA，把计算腿做成
64×64×64、`reps=10000`，与 CUDA-core FMA 腿（`kAcc=144`、6000 轮）调到**同量级**，再和通信腿并跑：

```
comm only          =  741.1 us (203.7 GB/s)
WGMMA(异步张量核)  only =  923.9 us
FMA(CUDA core)     only =  996.2 us
comm + WGMMA       =  953.5 us   eff = 0.930   ← 通信完全藏进张量核计算
comm + FMA         = 1188.7 us   eff = 0.620   ← 通信被拖出 ~200us
```

**同为「1 comm WG + 1 compute WG、同 CTA、`setmaxnreg` 40/216」**，只是把计算腿从 CUDA-core FMA 换成
异步张量核 WGMMA，重叠效率就从 **0.62 → 0.93**。这直接坐实了假设：

> **Mega MoE 的单 kernel 融合之所以成立，是因为它的计算是异步张量核（UMMA）；通信 warp 和 MMA warp
> 用的是两套互不争抢的执行资源。** 在 H100 上用 CUDA-core 做计算腿复刻同一结构，反而比拆成两个
> kernel 更差。

这也解释了本专题反复出现的「**SM100 专属**」到底专属在哪儿：不只是 UMMA/TMEM 的指令，而是
**「异步张量核 + 寄存器预算分区 + warp specialization」三者凑齐**才让「通信-计算同 SM 重叠」这条
路线在目标卡上成立。H100（以及任何靠 CUDA core 做稠密计算的架构）上，同样的骨架要让位于空间隔离
（限块 / 分 SM）。

---

## 6. 小结

| 问题 | 结论（H100 真机） |
|---|---|
| 寄存器预算分区可测吗 | 可测。`setmaxnreg` 让 ptxas 按目标排指令：有则 216 regs/0 spill，无则 156/2× 慢（极端配置无则 2300B spill、~10×） |
| 单 kernel 融合 vs 两 kernel co-residency | H100 CUDA-core 下**融合更差**（eff ~0.5 vs ~1.0）——同 SM 争发射/LSU |
| 缺什么 | **异步张量核**。换成 WGMMA 后融合 eff 从 0.62 → **0.93** |
| 与 Mega MoE 的联系 | Mega MoE 的 UMMA 是异步张量核，所以 dispatch warp 与 MMA warp 真正并行；这正是 SM100 专属的根 |

**可跑/仅分析**：`setmaxnreg`、warp specialization、NVLink peer copy、WGMMA 均为 H100 可跑真实设备代码；
`fp8_fp4_mega_moe` 融合 kernel 本体 SM100 专属，本篇仅分析。

真机原始输出：`harness/exp_mega_moe_like_ws.out.txt`；源码：`src/mega_moe_like_ws.cu`、
`src/mega_moe_like_ws_tc.cu`；驱动：`harness/exp_mega_moe_like_ws.sh`。

## 7. 引用

- DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`（`:56` launch bounds、
  `:90-92` 线程断言、`:326-335` 寄存器预算与断言、`:344/:692/:758/:821/:941/:948` dealloc/alloc、
  `:1372` `kNumMaxRegistersForBuffer`）。
- DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh:284-289`。
- DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh:98`。
- DeepGEMM `csrc/jit_kernels/heuristics/mega_moe.hpp:126,220-221`。
- DeepEP `deep_ep/include/deep_ep/impls/ep/hybrid_combine.cuh:103-104,111-112,357-358`。
- DeepEP `deep_ep/include/deep_ep/common/ptx.cuh:456-469`。
- 本专题：`docs/165-ep-pipeline-overlap.md`（限块正对照）、`docs/144-forward-overlap.md`（NCCL 侧流不重叠）、
  `docs/10-mega-moe-mma-epilogue.md`（UMMA 异步）、`docs/20-bf16-mega-moe.md`（BF16 变体同骨架）。
