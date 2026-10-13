# 把 wgrad 的 EPI-B 落到真 kernel：多出来的 235MB/步读，到底藏住了吗

> 本文对应路线图 **P139-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 149 篇。
>
> 第 148 篇（P138-1）用「字节账 + 真机重叠」把「把 S 步归约藏进 GEMM」走到**双重否定**：侧流/外部融合在
> SM90 的 `232448 B/SM`（`csrc/jit_kernels/heuristics/sm90.hpp:14`）下根本不让 block co-reside，只剩
> **同 kernel 的 `EPI-B`**——把每步 GEMM 的 epilogue 从「fresh bf16 直写」改成「**bf16 累加器
> read-modify-write**」——一条路。它给了字节账（`470·S`，最少）和上界 ~2.29× / 下界 ~1.69×，但
> **从没在真 kernel 上落地过**（原话：「本机无法改 SM90 kernel 做端到端验证」）。
>
> 本篇发现：DeepGEMM 的 **SM90 BF16 稠密 GEMM `bf16_gemm_nt` 本来就支持 bf16 累加器 RMW**——
> `c=d`（bf16）时走 `SM90_TMA_REDUCE_ADD_2D`，epilogue 就是「读一份 bf16 D + 加 + 写回」的货真价实的
> EPI-B。于是**不需要改任何 kernel** 就能做端到端落地验证。结论：
>
> 1. **多读的那一份 235MB/步，基本完全暴露**：wgrad 形状下每步边际 `Δ ≈ 109us`，而「纯读一份 D」在
>    H100 上的实测下界只有 ~88us（`2.67 TB/s`）——`Δ` 甚至超过 100%（`157%`），说明不仅没藏住，
>    TMA reduce-add 的 RMW 还比 fresh 直写更贵。
> 2. **藏不藏只看算术强度**：把收缩维 K 从 512 拉到 4096，同样这份 RMW 读的暴露比例从 `157%` 降到
>    `47%`（一批）——**GEMM 有带宽余量才谈得上藏**，而 wgrad 恰好没有。
> 3. **端到端**：EPI-B 相对「真实 wgrad 1D1D 同构形态」（bf16 操作数 + **fp32** 累积）稳定 **1.73–1.77×**，
>    几乎正好落在第 148 篇预测的**下界 1.69×** 上；但它相对「每步 fresh bf16 + 独立归约」（第 147/148 篇
>    的 `SEQ`/`ALT-R`）在 `S≥8` 时只有 **0.96–0.97×**（**略微更慢**）——因为两者总字节在 `S→∞` 时几乎相等。
>
> 可跑性：全程真实 SM90 BF16 kernel（**H100 可跑**）；`use_deterministic_algorithms(True)` 强制走 DeepGEMM
> 自己的 `sm90_bf16_gemm`，否则默认回退 cuBLASLt（第 27 篇）。SM100 的 wgrad `tn` 同族 fp32 累积、
> **仅分析**。本机 8 卡共享，本文在**空闲 GPU** 单进程测，报数并注明抖动。
> 实验代码 `harness/exp_wgrad_epib.{py,sh,out.txt}`。

---

## 1. 为什么 EPI-B 可以在 SM90 上直接落地

第 136/145 篇核对过：SM90 上能**累加**的 GEMM 接口并不多，而且都被宿主 API 钉死了 dtype：

| 入口 | 累加器 dtype | 累加？ | 证据 |
|---|---|---|---|
| FP8 `k_grouped_*` / 稠密 1D1D | **fp32** | 必累积 | `sm90_fp8_gemm_1d1d.hpp:84` `c 必填 and d==float` |
| FP8 稠密/分组 1D2D | （bf16） | **禁**累积 | `sm90_fp8_gemm_1d2d.hpp:81` `not c and d==bf16` |
| **BF16 稠密 `bf16_gemm_nt`** | **bf16 或 fp32** | 可选 | `sm90_bf16_gemm.hpp:89/91` |

真实 wgrad 用的是第一行（FP8 1D1D，**fp32** 累积）——所以第 148 篇说要「改 1D2D 的 epilogue」。
但第三行给出了一条**不用改 kernel** 的路：`bf16_gemm_nt` 的 `GemmDesc` 里
`.cd_dtype = d.scalar_type()`（`sm90_bf16_gemm.hpp:89`）、`.with_accumulation = c.has_value()`
（`:91`）。当 `c = d`（同为 bf16）时，kernel 模板参数 `kWithAccumulation=true`，epilogue 的 store 从
`SM90_TMA_STORE_2D` 换成 **`SM90_TMA_REDUCE_ADD_2D`**（`impls/sm90_bf16_gemm.cuh:381-384`）：

```cpp
using cute_tma_t = cute::conditional_t<kWithAccumulation,
    cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
cute_tma_t::copy(&tensor_map_cd, smem_ptr, n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
```

而 smem 里那份 D 也是 bf16（`cuh:302-349` 用 `ptx::SM90_U32x2_STSM_N<nv_bfloat162>` 写）。
所以 **TMA reduce-add 在 bf16 上把「读 D + 加 + 写 D」做在一趟里**——这正是 EPI-B 的机制本体。

**诚实边界**：这是 **BF16 操作数**、不是 FP8。本篇要回答的是**访存行为**（多读一份 bf16 D 能否被藏），
由形状 / 字节数决定，与操作数精度无关；故用与 wgrad **同形状**（`dW2 = [E·M, N]`、收缩维 K=token/步）的
bf16 GEMM 是恰当的载体。FP8 的精确 wgrad 算术（1D1D fp32 累积）作为 `REF` 同构形态一起测。

宿主入口 `csrc/apis/gemm.hpp:488-535` 也确认：`deterministic` 关时默认回退 cuBLASLt
（`:519-521`），`deterministic` 开时才走 `sm90_bf16_gemm`（`:529`）——所以实验里必须
`use_deterministic_algorithms(True)` 才能命中 DeepGEMM 自己的 kernel。

---

## 2. H100 实测 · Part A：载体正确性 + profiler 坐实模板 bool 翻转

空闲 GPU，单进程。`dW2` 形状 `[E·M, N] = [16384 = E8·M2048, N=7168]`，一份 bf16 D = **235 MB**、
fp32 = **470 MB**（与第 145~148 篇同形状）。

```
fresh 一步   : rel(d, a@b)      = 1.409e-03
RMW 再一步   : rel(d, 2·(a@b))  = 1.409e-03   （= bf16 累加器，两次结果一致）
fp32 累加器  : rel(df, 2·(a@b)) = 4.075e-07
```

- `bf16_gemm_nt(a,b,d,c=d)` 再算一步得到 `2·(a@b)`、且相对误差与单步完全一致 → **确实是把第二步
  「加」进了同一份 bf16 累加器**（bf16 累加器 RMW）。
- fp32 累积器误差 `4.1e-7` 是 fp32 舍入地板，与 bf16 的 `1.4e-3` 分开两个量级。

profiler 抓到的 kernel 名（只看模板尾部的 `kWithAccumulation` 布尔）：

```
fresh: ... (deep_gemm::GemmType)0, false, cutlass::bfloat16_t> ...
RMW  : ... (deep_gemm::GemmType)0, true , cutlass::bfloat16_t> ...
```

即同一份 kernel 源码，`kWithAccumulation` 由 `false` 翻成 `true`，走 reduce-add 分支。**这就是第 148 篇
要的那个「改 epilogue」的等价物**——只是 DeepGEMM 已经把它做成了开关。

---

## 3. H100 实测 · Part B：三协议端到端（wgrad 形状）

三条协议（`S` = micro-batch 步数；每步收缩维 K=512）：

- **`SEQ`** = 第 147/148 篇的 `ALT-R`：每步 `bf16_gemm_nt` **fresh bf16 直写** + 收尾一次 `torch.sum`
  （读 S 份 + 写 1 份 fp32）。
- **`EPI-B`** = 本篇落地：每步 `bf16_gemm_nt(..., c=d_bf16)` 的 **bf16 累加器 RMW**，无独立归约。
- **`REF`** = 真实 wgrad 的 1D1D 同构形态：每步 **fp32** 累加器 RMW（fp32 D = 470 MB/份）。

单步（各 20 次，`dW2` 形状）：

```
fresh(写) 197.8us (608 TFLOPS) | EPI-B bf16 RMW(读+写) 307.2us | fp32 累积 545.1us
-> EPI-B 相对 fresh 的每步边际 Δ = 109.4us
-> 「纯读一份 bf16 D」的 HBM 理论下界 = 70.1us（@3.35TB/s）
```

端到端：

| S | SEQ (us) | EPI-B (us) | REF fp32 累积 (us) | SEQ/EPI-B | EPI-B/REF |
|---:|---:|---:|---:|---:|---:|
| 1 | 580.4 | 308.5 | 545.5 | 1.88 | **1.77** |
| 2 | 898.4 | 613.9 | 1080.3 | 1.46 | 1.76 |
| 4 | 1353.3 | 1230.2 | 2177.4 | 1.10 | 1.77 |
| 8 | 2433.7 | 2525.7 | 4436.8 | 0.96 | 1.76 |
| 16 | 4914.5 | 5081.8 | 8784.2 | 0.97 | **1.73** |

读法：

1. **`EPI-B/REF` 几乎是常数 1.73–1.77×**——这正是第 148 篇字节账里 EPI-B 相对 fp32 累积省下的
   （fp32 D 的读+写各 470MB/步 vs bf16 各 235MB/步）。它**顶到了预测下界 1.69×**。
2. **`SEQ/EPI-B` 从 S=1 的 1.88× 掉到 S≥8 的 0.96–0.97×**（EPI-B 反而略慢）。原因是 `S→∞` 时
   两者总字节几乎相等：`SEQ = 470·S + 235`（fresh 写 + 归约读）vs `EPI-B = 470·S`（每步 RMW）；
   `SEQ` 只多出末次 fp32 归约的 235MB，却把「S 步读」交给了一个**纯流式**的归约核（第 147 篇实测
   ~3.08 TB/s），比压在 GEMM epilogue 里的 TMA reduce-add 更高效。**所以 EPI-B 并不比「fresh + 高效
   归约」快**——它的价值只在于对 `REF`（fp32 累积）的 1.7×。

---

## 4. H100 实测 · Part C：藏不藏，只看算术强度

把同一份 RMW 读放到不同的收缩维 `K` 下（每项取 3 次最小值，避开共享节点的争用抖动）：

| K | t_fresh (us) | t_epib (us) | Δ (us) | Δ / 纯读下界 | AI fresh / epib | TFLOPS fresh/epib |
|---:|---:|---:|---:|---:|---:|---:|
| 512（wgrad 形状） | 196.0 | 306.3 | **110.3** | **157%** | 464 / 244 | 614 / 393 |
| 2048 | 633.5 | 746.3 | 112.8 | 161% | 1452 / 850 | 759 / 645 |
| 4096（计算受限） | 1417.7 | 1450.6 | **33.0** | **47%** | 2249 / 1452 | 679 / 663 |

带宽锚（同进程实测）：**读一份 bf16 D 235MB = 88.0us（2.67 TB/s）**、读+写 = 157.9us（2.97 TB/s）。

- wgrad 形状：`Δ = 110us > 88us`——**多读的那份 D 不仅完全暴露，还多了 TMA reduce-add 的开销**。
  换句话说，`bf16_gemm_nt` 在这一形状下**没有可藏的带宽余量**（它本身只有 ~608 TFLOPS，远低于
  峰值，属于「小 K 打不满」的中间区，`fresh` 的写带宽也远未到顶）。
- 大 K：`Δ` 降到 33us（47%）——**GEMM 越偏计算受限，这份读越藏得住**。藏得住的前提是「通信/多读的
  时间被计算时间覆盖」，而 wgrad 的收缩维仅 512（4 个 `BLOCK_K`），恰恰是最不计算受限的一档。

这解释了第 148 篇 why 的另一半：**不是「融进 kernel」这个动作本身能藏读，而是 GEMM 必须本来就有
带宽余量**。wgrad 形状没有，所以 EPI-B 的读一定暴露。

---

## 5. ncu：DRAM 读字节直接坐实「多读一份 D」

`ncu --kernel-name 'regex:sm90_bf16_gemm' --launch-count 1`，同一形状单次 launch 热 cache：

| kernel | DRAM 读 (MB) | DRAM 写 (MB) | 时长 (us) |
|---|---:|---:|---:|
| fresh（只写） | 74.78 | 215.10 | 202.5 |
| EPI-B（读+写 RMW） | **312.75** | 214.33 | 300.5 |

`312.75 − 74.78 = 237.97 MB`，**恰好是一份 bf16 `D`（235 MB）**——TMA reduce-add 的「读 D」在硬件
计数器里如实显现。时长增量 `300.5 − 202.5 = 98us`，`238MB / 98us = 2.43 TB/s`，与上文实测读带宽
`2.67 TB/s` 同量级——**这份读是「按满速露在墙上」的，一点没被计算盖住**。

---

## 6. 结论与边界

- **EPI-B 是真的、也是可跑的**：SM90 `bf16_gemm_nt(..., c=d)` 就是 bf16 累加器 RMW（`TMA_REDUCE_ADD_2D`），
  profiler 模板 bool 与 ncu 读字节双重坐实；不需要改 kernel。
- **但「多出来的 235MB/步读」藏不住**：wgrad 形状下每步边际 `Δ≈110us`，超过「纯读一份 D」的实测下界
  `88us`（暴露 157%）；`ncu` 显示这份读按 ~2.4 TB/s 满速发生。
- **能否藏取决于算术强度**：收缩维 K=512→4096，暴露比例 157%→47%。wgrad 收缩维仅 512，没有余量。
- **端到端价值**：EPI-B 对真实 wgrad 同构形态（fp32 累积）稳定 **1.73–1.77×**，几乎正好落在第 148 篇
  的**下界 1.69×**；但它对「fresh bf16 + 高效独立归约」（`SEQ`/`ALT-R`）在 `S≥8` 只有 0.96–0.97×——
  **EPI-B 并不优于「批量 fresh + 一次流式归约」**，它省的是 fp32 物化，不是那份读。
- 边界：① 载体是 BF16 操作数（FP8 1D2D 不支持累加），回答的是访存而非 FP8 精确算术；② SM100 的 wgrad
  `tn`（`sm100_fp8_fp4_gemm_1d1d.hpp:325`）同族 fp32 累积、**仅分析**；③ 8 卡共享节点，跨运行有抖动，
  Part C 取 3 次最小值。

**一句话**：第 148 篇把 2.27× 的唯一剩余门指向「同 kernel 的 EPI-B」；本篇把它真做了，量到的是——
**EPI-B 能拿回 1.7×（对 fp32 累积），但那多读的一份 bf16 D 是彻底暴露的**，因为 wgrad 的 GEMM 没有
带宽余量可藏。要再快，只剩「换 FP8 累加器 / 更小输出 dtype」这类**减少字节**的路，而不是「藏」。

---

## 7. 坐标与复现

- 驱动：`bash harness/exp_wgrad_epib.sh > harness/exp_wgrad_epib.out.txt`
  （Part A/B/C + `ncu` 单 launch 对比）。
- 关键行号：`csrc/apis/gemm.hpp:488-535/519-521/529`、
  `csrc/jit_kernels/impls/sm90_bf16_gemm.hpp:89/91`、
  `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh:302-349/381-384`、
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`、
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`、
  `csrc/jit_kernels/heuristics/sm90.hpp:14`。
- 关联：第 12/15/135/136/145/146/147/148 篇（wgrad 的反向路线）、第 27 篇（cuBLASLt 回退与
  `deterministic` 开关）、第 19 篇（SM90 BF16 主 kernel 的 TMA reduce-add）。
