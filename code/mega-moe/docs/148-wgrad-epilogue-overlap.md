# 把 wgrad 的 S 步归约藏进 GEMM：epilogue 融合 / 侧流重叠到底行不行

> 本文对应路线图 **P138-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 148 篇。
>
> 第 147 篇（P137-1）把「wgrad 换 bf16 输出」落到端到端 **1.67×**（ALT-OPT），并断言 2.27× 不可达，
> 因为「每步 bf16 梯度至少要**串行**读一遍（~76us/步）」。它留下的口子（可继续①）是：
> **若把归约 / 优化器更新融进 GEMM 的 epilogue、或与下一步 GEMM 重叠，这 76us 能不能藏掉？**
>
> 本篇用三个架构无关的消费侧内核 + 真实 SM90 GEMM 把这条路走到能核对的数，结论是一个**双重否定的负结果**：
>
> 1. **“增量式”归约（每步就地更新 master / 累加器）字节比批量式多 2–3.3×**，且**它并不重叠**——
>    真机上 `SEQ`（批量尾收）对 `OVL`（侧流增量）的比值是 `0.54–0.84×`，即侧流增量**比串行还慢**。
> 2. **为什么不重叠**：DeepGEMM 的 GEMM 把每 SM 的共享内存用到 **232448 B**（`heuristics/sm90.hpp:14`），
>    每个 SM 只驻留 1 个 block；侧流消费者**挤不进任何 SM 槽位**。决定性控制实验：把生产者 kernel 的
>    动态 smem 从 0 扫到 227 KB，消费者被藏住的比例从 ~90% 掉到 **−0.1%（完全串行）**。
>
> 于是 2.27× 的**唯一剩余门**是**同一个 kernel 内**的 epilogue 融合（换掉 1D2D 的 fresh-write 为
> `read-modify-write` 一个 bf16 累加器）——它的**字节账其实最优**（比 ALT-OPT 省掉末次 fp32 master 的
> 940MB 读写），上界 ~2.3×、下界 ~1.7×，但**必须改 DeepGEMM 的 1D2D epilogue**，本机只做字节账与旁证。
>
> 可跑性：真实 SM90 FP8 GEMM（H100 可跑）；消费侧内核（reduce / 增量更新 / 伪 epilogue 生产者）
> **不碰 UMMA/tcgen05、纯 bf16/fp32 访存 + 浮点，架构无关**，直接编译运行真实设备代码。SM100 的
> wgrad `k_grouped_tn_contiguous` 同族 fp32 累积、**仅分析**。本机 8 卡共享，本文在**空闲 GPU** 单进程测，
> 报数注明。实验代码 `src/wgrad_epilogue_overlap.cu`、`harness/exp_wgrad_epilogue_overlap.{sh,py,out.txt}`。

---

## 1. 合同：三种「把 S 份梯度交给优化器」的架构与字节账

先固定物理量。wgrad 的输出 `dW2` 形状是 `[E=8, M=2048, N=7168]`，元素数
`G = E·M·N = 117,440,512`：**一份 bf16 = 235 MB，一份 fp32 = 470 MB**。

SM90 上 wgrad 入口 `k_grouped_fp8_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:373-427`）的两条硬约束
（第 145/146 篇已核）决定了「每步 GEMM 的输出形态」只有两种：

- **1D1D**：`recipe==(1,1,128)`（`:384`）、`C` 必填（`:403`）、输出 **fp32 且必累积**
  （`sm90_fp8_gemm_1d1d.hpp:84`），落点 `SM90_TMA_REDUCE_ADD_2D`
  （`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`）——**每步读回整份 fp32 `C`**。
- **1D2D**：`not c` 且 `d==bf16`（`sm90_fp8_gemm_1d2d.hpp:81`）——**bf16 直写、禁累积**。

「累积 S 步 + 交给优化器」因此有下面几种架构。表中字节都按 `dW2` 的**线上流量**记，`S=8`：

| 架构 | 每步写（GEMM） | 每步消费 | S=8 总字节 | 能否与 GEMM 重叠 |
|---|---|---|---|---|
| **SEQ**（= 第 147 篇 ALT-OPT，批量尾收） | fresh bf16 235 | 尾收：读 `S·235` + 读/写 fp32 master 940 | `470·S + 940` = **4700** | 否（尾收只能串行） |
| **INC-F**（侧流增量，fp32 master） | fresh bf16 235 | 每步读 `g` 235 + RMW fp32 master 940 = **1175** | `(235+1175)·S` = **11280** | 是（理论上） |
| **INC-B**（侧流增量，bf16 累加器） | fresh bf16 235 | 每步读 `g` 235 + RMW bf16 acc 470 = **705** | `(235+705)·S` = **7520** | 是（理论上） |
| **EPI-B**（**同 kernel** epilogue 融合，bf16 累加器） | 不写 fresh，直接 RMW acc 470 | —（在 epilogue 内） | `470·S` = **3760** | 天然（同一 kernel） |

两个立刻可见的事实：

1. **增量式（INC-F / INC-B）的字节比批量式（SEQ）多 2–3.3×**：批量式把「读 S 份」和「读/写 master」
   各做一遍；增量式每步都要把 master **读一遍再写一遍**（RMW），S 步下来就是 S 倍 master 流量。
2. **同 kernel 的 epilogue 融合（EPI-B）反而字节最少**：它既不写 fresh bf16 再读回，也省掉末次
   fp32 master 的 940MB——所以「融合」的正确形态**不是**侧流增量，而是**改 1D2D epilogue**。

本篇要回答的是：**增量式的额外字节能不能靠「与 GEMM 重叠」补回来？**（若不能，唯一活路就是 EPI-B。）

---

## 2. 消费侧内核：H100 实测（架构无关）

`src/wgrad_epilogue_overlap.cu` 给三个消费侧内核（8 元素/线程、128-bit 向量化访存）：
`k_fused_all`（批量，= ALT-OPT）、`k_fused_step`（增量更新 fp32 master）、
`k_acc_bf16_step`（增量 RMW bf16 累加器）。全部**不碰 UMMA/tcgen05**，H100 直接编译运行真实设备代码，
与 CPU double 对拍 `max|err| = 5.96e-8`（= fp32 求和舍入地板）。

**独跑带宽**（`budget=4224` block，空闲 GPU）：

| 内核 | 字节 | 时间 | 带宽 |
|---|---|---|---|
| 纯读一份 bf16 | 235 MB | 78.2 us | **3005 GB/s** |
| `acc_bf16` 一步 | 705 MB | 237.4 us | 2970 GB/s |
| `fused_step` 一步 | 1175 MB | 402.0 us | 2923 GB/s |
| `fused_all S=8` | 2820 MB | 914.5 us | **3084 GB/s** |
| `fused_all S=16` | 4700 MB | 1533 us | 3066 GB/s |

即：H100 HBM 峰值 ~3.35 TB/s，消费侧**稳在 ~3.0 TB/s（90%+）**，与第 147 篇同量级。

**块数预算扫描**（`fused_step`，每步 1175MB）：

| blocks | 132 | 264 | 528 | 2112 | 4224 |
|---|---|---|---|---|---|
| GB/s | 2494 | 2957 | 2878 | 2891 | 2925 |

→ **只要 ≥264 个 block（~2/SM）就能打满 HBM**；132 block（1/SM）只有 2494 GB/s（受 MLP/延迟限制）。
这一点很关键：侧流消费者**并不需要独占 SM**，少量 SM 就够用——挡路的不是吞吐，是**能不能挤进去**（§4）。

---

## 3. 真机重叠：侧流增量消费者 vs 真实 GEMM

在 8 卡节点空闲 GPU 上，用真实 SM90 `deep_gemm.fp8_gemm_nt`（1D2D bf16，`E×` 稠密，每步 8 个 kernel）
搭 S 步流水，比较：

- **SEQ**：8 步 GEMM 跑完，再一次性 `k_fused_all`（批量尾收；= 第 147 篇 ALT-OPT 的消费段）。
- **OVL**：每步 GEMM 后 `event.record` → 侧流 `wait_event` → 侧流跑增量消费者，与下一步 GEMM 重叠。

单步 GEMM = **218 us/步**。结果（`S=8`，消费者全带宽）：

| gemm_sms | gemm us/步 | SEQ8 us | OVL8-INC-F | OVL8-INC-B | SEQ/OVL-F | SEQ/OVL-B |
|---|---|---|---|---|---|---|
| 132 | 218 | 2672 | 4912 | 3597 | **0.54** | **0.74** |
| 96 | 218 | 2674 | 4903 | 3579 | 0.55 | 0.75 |
| 64 | 274 | 3081 | 5293 | 3983 | 0.58 | 0.77 |
| 48 | 341 | 3644 | 5839 | 4544 | 0.62 | 0.80 |
| 32 | 460 | 4567 | 6744 | 5452 | 0.68 | 0.84 |

**侧流增量比串行尾收还慢**（比值全 < 1）。两个原因叠在一起：

- **字节**：OVL-INC-F 每步消费 8×1175 = 9400MB，即使独占 3 TB/s 也要 3130us，再叠 GEMM 的 8×218=1744us，
  串起来正好 ≈ 4874us ≈ 实测 4912us。**几乎没有重叠**（若完全重叠应是 ~3130us）。INC-B 同理
  （5640MB→1880us + 1744 = 3624us ≈ 实测 3597us）。
- **给 GEMM 限核（`set_num_sms`）反而让重叠比例略升**（0.54→0.68），但绝对值更差——因为限核把 GEMM
  拖慢了，而消费者**始终没有真正并行**。

**“少给 GEMM 核、多给消费者”也救不了**：把 GEMM 限到 64 SM 后扫消费者块数，INC-F 对 SEQ 恒 ~0.58×，
与消费者块数无关——说明瓶颈**不是消费者吞吐**（它 264 block 就打满 HBM 了），而是**两者根本不并行**。

---

## 4. 决定性控制：把“不并行”钉到共享内存占用

为什么真实 GEMM 和侧流消费者不并行，而一个“纯计算” kernel 却能和消费者并行？看 DeepGEMM 的 smem 预算：
`heuristics/sm90.hpp:14` 写死 `smem_capacity = 232448`，而 `get_pipeline_config`（`:148-188`）把
`smem_extra + num_stages*smem_per_stage` 填到接近这个上限——**每个 SM 只驻留 1 个 GEMM block**，
侧流消费者的 block（哪怕只要一点点 smem / warp 槽位）**挤不进任何一个 SM**，于是只能在 GEMM 的 block
空出来之后才跑 → 退化成串行。

用 `src/wgrad_epilogue_overlap.cu` 里的**伪 GEMM epilogue 生产者**（`k_gen_bf16`：算 `flops` 次 FFMA
再写 bf16；**动态 smem 可调**）做隔离实验：固定 `flops=256`（生产者 ~3.35ms，计算受限），
让侧流消费者（`fused_step`）与之并发，扫生产者的动态 smem：

| prod smem | 0 | 32 KB | 64 KB | 128 KB | 192 KB | **227 KB** |
|---|---|---|---|---|---|---|
| 生产者独占 | 3351 us | 3351 | 3361 | 3351 | 3351 | 3351 |
| 串行（prod+cons） | 3753 | 3753 | 3763 | 3753 | 3753 | 3753 |
| 并发实际 | 3538 | 3527 | 3540 | 3385 | 3383 | **3754** |
| 消费者被藏住 | 53% | 56% | 56% | **92%** | **92%** | **−0.1%** |

**当生产者占用 227 KB/SM（= DeepGEMM 的 `232448`）时，消费者被藏住的比例掉到 −0.1%——完全串行**
（并发 3754us ≈ 串行 3753us）。这就是 §3 里真实 GEMM 不并行的**确切机制**：
不是 HBM、不是调度器，而是 **SM 的共享内存槽位被占满，侧流 block 无法 co-reside**。

（这也与第 144 篇（P134）的结论一致：NCCL 侧流与 grouped GEMM 不重叠，因为「必须先按 SM 预算分区」；
本篇补上了 SM90 wgrad 这条更硬的落点——`232448` smem 直接堵死 co-residency。）

---

## 5. 增量协议的数值：bf16 累加器的累积舍入

若走 EPI-B（bf16 累加器），S 步会做 S 次 bf16 舍入。实测（vs fp64 前缀和）：

| S | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| bf16 累加 relerr | 0 | 1.80e-3 | 2.68e-3 | 3.68e-3 | **5.03e-3** |
| fp32 求和 relerr | 0 | 1.03e-10 | 2.79e-10 | 5.62e-10 | 1.04e-9 |
| cos(bf16acc, ref) | 1.0 | 0.9999985 | 0.9999964 | 0.9999933 | **0.9999875** |

bf16 累加器的误差随 S 缓涨（近 `√S·ε_bf16`），`S=16` 时 **5.03e-3**——仍比 FP8 输入噪声 3.6e-2
（第 146 篇）**小 7×**。即：**数值上 bf16 累加器依然“免费”**，瓶颈只在字节/并行，不在精度。

---

## 6. 结论：唯一剩余门是同 kernel 的 epilogue 融合，及其上界

把 §1 的字节账、§3–§4 的并行实验、§5 的数值合起来：

1. **侧流重叠（INC-F / INC-B）被双重否决**：
   - 字节比批量式多 2–3.3×（每步 RMW master）；
   - 即使字节免费，DeepGEMM 的 `232448 B/SM` 占用让侧流 block **无从 co-reside**，实测**零重叠**
     （§4 的 227 KB → −0.1%）。真机 `SEQ/OVL` 恒 0.54–0.84×，**比串行尾收还慢**。
2. **唯一活路是 EPI-B（同 kernel epilogue 融合）**：把 1D2D 的「fresh-write 235MB」
   换成「RMW bf16 累加器 470MB」，S 步总字节 `470·S`——**比 ALT-OPT 还少 940MB**。
   - **上界**：若那多出来的 235MB/步（读 acc）能藏进 GEMM 的 HBM 余量（GEMM 自身 ~1.24 TB/s，
     余量 >2 TB/s），端到端 ≈ `S·t_gemm` → `REF/(S·t_gemm) = 8·499/(8·218) ≈ 2.29×`。
   - **下界**：若完全不藏，每步 +235MB/3 TB/s ≈ +78us → `8·(218+78) = 2368us` → `3992/2368 ≈ 1.69×`
     —— 与第 147 篇的批量天花板持平。
   - 数值免费（§5，5e-3 ≪ FP8 噪声）。

所以对第 147 篇「可继续①」的**最终回答**：
**把归约“融进 epilogue”只有在「同一个 kernel 内」才有意义（字节最优、无 co-residency 问题）；
侧流/外部融合在 SM90 上被 `232448 B/SM` 的占用彻底堵死。** 真正要吃到 2.27×，动作是
**改 DeepGEMM 1D2D epilogue 的输出协议（fresh bf16 → bf16 累加器 RMW）**，而不是加一条 stream。
本机无法改 SM90 kernel 做端到端验证（会改内核语义），故 EPI-B 只给字节账与上/下界；
SM100 的 wgrad `tn`（`sm100_fp8_fp4_gemm_1d1d.hpp:325`）同族 fp32 累积，**仅分析**。

---

## 7. 坐标与复现

- 内核：`src/wgrad_epilogue_overlap.cu`（`k_fused_all` / `k_fused_step` / `k_acc_bf16_step` /
  `k_gen_bf16` + ctypes API；sha256 见 `.out.txt` 首行）。
- 驱动：`bash harness/exp_wgrad_epilogue_overlap.sh > harness/exp_wgrad_epilogue_overlap.out.txt`。
- 关键行号：`csrc/apis/gemm.hpp:373-427/384/403`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`、
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`、
  `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`、
  `csrc/jit_kernels/heuristics/sm90.hpp:14/148-188`、
  `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:325`。
- 关联：第 12/15/145/146/147 篇（wgrad 路线）、第 144 篇（P134，SM 预算分区/侧流不重叠）。

## 8. 小结与下一步

- **一句话**：侧流增量归约既多花字节、又因 `232448 B/SM` 的占用而**零重叠**（实测 `SEQ/OVL` 0.54–0.84×）；
  2.27× 的唯一剩余门是**同 kernel 的 bf16-累加器 epilogue 融合**（字节最优、上界 ~2.29× / 下界 ~1.69×、
  数值免费），需改 DeepGEMM 1D2D epilogue。
- **下一步（opt-in）**：若环境允许，可在一个**可改的** SM90 GEMM 分支上做 EPI-B 的端到端验证
  （把 epilogue 的 fresh bf16 store 换成 bf16 累加器 RMW），量「多出来的 235MB/步读」到底藏没藏进去；
  或对 P96/P97/P123 的轻微漂移清单关键路径定点复核。DeepEP 端到端仍需 NCCL≥2.32.3（见路线图「阻塞」）。
