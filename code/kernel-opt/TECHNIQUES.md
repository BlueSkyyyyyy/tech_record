# 算子优化技巧台账 & 性能榜（活文档）

> 本文件是「CUDA 算子调优」系列的**知识沉淀**：每个优化手段的**原理、适用场景、代码位置、实测收益、踩坑**，
> 以及各算子的**最好成绩**。每写一篇，agent **必须**在这里至少新增一条技巧条目并更新性能榜。
> 目标：让任何人（以及未来的 agent）能快速查到「这个场景该用什么手段、能带来多少收益」。

图例：收益为相对**上一版/基线**的实测提升（H100 SXM）。代码位置相对 `code/kernel-opt/`。

---

## 一、性能榜（H100 SXM，持续更新）

| 算子 | 最佳实现 | 指标 | 峰值占比 | 对标 SOTA | 出处 |
|---|---|---|---|---|---|
| copy（读+写） | float4 + grid-stride | **2997 GB/s** | 89.4% HBM | — | 04-coalescing |
| 矩阵转置 | smem tile[32][33] padding | **1651 GB/s** | 49.2% HBM | 可继续向量化 | 05-transpose |
| 归约（求和） | float4 + warp shuffle 两级归约 | **3057 GB/s** | 91.2% HBM | — | 06-reduction |
| Softmax（行 8192） | block/row + 整行 smem 缓存 | **2861 GB/s** | 85.3% HBM | — | 07-softmax |
| GEMM fp32 2048³ | reg tiling + cp.async 双缓冲 | **34.46 TFLOPS** | 51.5% (66.9) | cuBLAS 50.73 (75.8%) → 达 68% | 09/12 |
| GEMM bf16 2048³ | mma.m16n8k16 + ldmatrix + cp.async | **220.98 TFLOPS** | 22.3% (989) | cuBLAS 672.70 (68%) → 达 32.8% | 13 |
| GEMM bf16 + bias+GELU | 寄存器 epilogue 融合 | **218.61 TFLOPS** | 22.1% | 仅比纯 GEMM 慢 ~1% | 14 |
| 独立 epilogue kernel | float4 grid-stride | **~3979 GB/s** | >100%（数据驻 L2） | — | 14 |
| MLA 注意力（absorb, prefill） | **单 kernel 融合 + 共享 P（f4s）** | **170.11 TFLOPS** @Sq=Sk=1024 | 17.2% (989) | FlashMLA 640（sparse prefill, H800）→ 差距 3.76× | 16-mla-fused |
| MLA 注意力 · 长上下文 | 同上 f4s | **184.67 TFLOPS** @Sq=1024,Sk=4096 | 18.7% | 相对 15 三 kernel 5.78× | 16-mla-fused |
| GEMM bf16 4096³ · wgmma SS | SW128 swizzle 冒烟（单缓冲，未调流水） | **132.04 TFLOPS** @4096³（148.66 @2048³） | 13.4% (989) | 13 篇 `mma+ldmatrix+cp.async` 221 | 20-mla-wgmma-sw128 |
| MLA 注意力 · wgmma SS | **SW128 swizzle 融合单 kernel** | **105.30 → 119.89 TFLOPS** @Sq=1024,Sk=1k→4k | 10.6–12.1% (989) | 16 篇 wgmma INTERLEAVE 84.6 → +24.5%；f4s 170（1.62×）；FlashMLA ~640（6.1×） | 20-mla-wgmma-sw128 |
| MLA 注意力 · wgmma SS | **V 转置访存合并**（dv 最快） | **142.31 → 159.41 TFLOPS** @Sk=1k→4k | 14.4–16.1% | 20 篇 105.3→142.3（+35%）；`l1tex` 73%→46% | 21-mla-wgmma-pipe |
| **MLA 注意力 · wgmma SS（当前最佳）** | **V 转置合并 + `cp.async` 单缓冲预取 K** | **159.8 / 193.2 / 198.8 TFLOPS** @Sk=1k/4k/8k | 16.2–20.1% | Sk≥4096 反超 16 篇 f4s（183.8/173.9）；FlashMLA ~640 → ~3.3× | 21-mla-wgmma-pipe |
| DSA lightning indexer (H^I=64, d=128) | TC mma + head 合并 HG=2 + BN=128 | **311.0 TFLOPS** @S=32768 | 31.4% (989) | 标量 0.93 → 335× | 18-dsa-sparse |
| DSA exact top-k (k=1024) | radix-select 4 趟 + per-warp 直方图（流式） | **6.75 ms** @S=32768 | 95% HBM（等效 5 趟 21.5GB/6.75ms） | 单直方图 26.7ms → 3.96× | 18-dsa-sparse |
| DSA 端到端（indexer+topk+稀疏 MLA 估算） | 同上 + 稀疏 MLA 按稠密实测吞吐折算 | **18.6× vs 稠密 MLA** @S=65536 | — | 理论上限 ~30×（indexer O(S²) 封顶） | 18-dsa-sparse |
| **DSA 稀疏 MLA 消费端**（gather top-k + online softmax + 共享 P） | **cp.async 双缓冲 p2**（BH=64,DVGRP=2,KT=32） | **136.3 → 127.9 TFLOPS** @Sk=4k→64k | 13.8% (989) | FlashMLA sm90 sparse prefill 640（H800）→ ~4.8× | 19-dsa-sparse-attn |
| DSA 稀疏 attention vs 稠密 MLA | 同上 p2 | **3.0×(4k) / 25.2×(32k) / 48.6×(64k)** | 效率为稠密 f4s 的 ~75% | 稠密 f4s 168–184 TFLOPS | 19-dsa-sparse-attn |
| DSA 端到端（indexer+topk+**实测**稀疏 MLA） | 18 的 indexer/topk + 19 的稀疏 MLA | **2.5×(4k) / 13.3×(32k) / 17.3×(64k)** | — | 理论上限 ~31× | 19-dsa-sparse-attn |
| MoE grouped GEMM | *待填（23）* | — | — | DeepGEMM | — |
| **FP8 e4m3 GEMM（per-tensor，当前最佳）** | **TMA（`cp.async.bulk.tensor.2d` + SW128）+ mbarrier + warp specialization + wgmma** | **1217.3 TFLOPS** @4096×3072×7168 | 61.5% (1978) | cuBLAS FP8 1381.3 → **88.1%（差距 1.13×）**；同 shape cuBLAS bf16 800.4 的 1.52×；tensor pipe 72% | 23-fp8-gemm-tma |
| FP8 e4m3 GEMM（per-tensor，cp.async 版） | wgmma m64n128k32 + SW128 + `cp.async` 3 级 + `wait_group` 流水 | 768.3 TFLOPS | 38.8% (1978) | 换 TMA 后 +58%；同 shape cuBLAS bf16 800.7 的 96% | 22-fp8-gemm |
| FP8 e4m3 GEMM（mma.sync 对照） | `mma.m16n8k32` + ldmatrix(b16) + `cp.async` 4 级 | 266.3 TFLOPS | 13.5% | 换 wgmma 后 **+2.89×**；tensor pipe 41%、`math_pipe_throttle` 1.12 | 22-fp8-gemm |
| FP8 e4m3 GEMM（per-block，TMA） | wgmma + 每 128-k 块折算 `sa[m,kb]*sb[nblk,kb]` | **906.2 TFLOPS** | 45.8% | 22 篇 cp.async 版 519 → **+74%**；寄存器 150、occ 13.8% | 23-fp8-gemm-tma |
| FP8 e4m3 GEMM（per-block，cp.async） | wgmma + 每 128-k 块折算 | 519.0 TFLOPS | 26.2% | per-tensor 的 68%（折算切断流水）| 22-fp8-gemm |
| MuonClip NS 正交化（N=4096） | 自研 GEMM 链 + 融合 epilogue（cfg1） | **244 TFLOPS** | 24.7% (989) | cuBLAS 链 529（53.5%）→ 差距 2.17× | 17-muonclip-ns |
| MuonClip NS 单 GEMM（4096³） | 自研 128×128×64 3 级流水 | **269 TFLOPS** | 27.2% | cuBLAS 883（89.3%）→ 达其 30.5% | 17-muonclip-ns |

> 口径说明：内存算子用有效带宽（读+写按实际最小搬运量）；GEMM 用 `2MNK/时间`；
> bf16 TC 峰值按 989 TFLOPS、fp32 按 66.9 TFLOPS、FP8 按 1978 TFLOPS；对标一律「同 shape、同口径」。

---

## 二、技巧台账

### A. 测量与基准（01/03/10）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| A1 | warmup + 多次取平均 | 所有基准 | 首次调用含 module 加载/context 等冷启动 | `common/cuda_utils.cuh: bench_ms` | 冷启动 25µs vs 稳态 9µs（2.8×） | 单次测量必错 |
| A2 | 区分墙钟 / device 时间 | 小 kernel | event 计时含 launch（~1.9µs/次） | `03-measurement` | 空 kernel 1.87µs/launch | 小 kernel 墙钟大半是 launch |
| A3 | 工作集 > L2 才算 HBM | 带宽bench | H100 L2=52MB，小数据命中 L2 | `03-measurement` | 32MB 工作集算出 3954 GB/s（超峰值，假） | 用 ≥128MB |
| A4 | ncu SOL 三节定位瓶颈 | 任意 | SpeedOfLight / Memory / Occupancy | 各篇 ncu 输出 | — | `Memory Throughput` 是**缓存层级最大值**，不是 HBM；看 `DRAM Throughput` |
| A5 | 手算 occupancy 对照 | 调参 | 资源受限项取 min | `10-ncu-deep` | 手算与 ncu 一致 | 理论≠实际（调度/负载不均） |

### B. 访存 / 合并（04）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 | 坑 |
|---|---|---|---|---|---|---|
| B1 | 相邻线程访问相邻地址 | 全局访存 | 一个 warp 合并成 4 sector/请求 | `04-coalescing` | 行优先 vs 列优先 **2382→323 GB/s（7.4×）** | 跨行 stride 会打满 L2 事务 |
| B2 | float4 向量化 | 连续对齐数据 | 指令数 ÷4，事务更宽 | `02/04` | 71% → **89%** 带宽 | 需 16B 对齐 + 元素数整除 |
| B3 | 看 sectors/request | 诊断不合并 | 理想=4（128B/32B） | `04` | colwise=32 sector/req | 不合并常表现为 L2 高、DRAM 低 |

### C. 共享内存 / bank（05/13）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 | 坑 |
|---|---|---|---|---|---|---|
| C1 | smem 中转转置 | 读写必然一边不合并 | 片内换坐标，全局两边合并 | `05-transpose` | naive→smem 466→985 GB/s | — |
| C2 | padding 消 bank conflict | 列访问 smem | bank=下标%32；列访问恒同 bank | `05` | 985→**1651 GB/s**，冲突 6531万→40万 | — |
| C3 | ldmatrix 加 +8 padding | TC GEMM | 行距 128 bf16 时 8 行同 bank | `13-tensor-core` | 76→**221 TFLOPS（2.9×）**，冲突 3565万→0 | 行距 `BK+8` / `BN+8` |

### D. 归约 / 协作（06/07）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 | 坑 |
|---|---|---|---|---|---|---|
| D1 | 不要每元素 atomicAdd | 归约/累加 | 单地址 RMW 串行 | `06-reduction` | 2.3 GB/s（0.1%），且浮点结果错（rel 0.5） | — |
| D2 | 两级归约 | 归约 | 线程局部→block→每 block 一原子 | `06` | 0.1% → **88%** | — |
| D3 | warp shuffle | block 内归约 | `__shfl_down_sync` 寄存器交换 | `06/07` | 省 smem+同步；88.5%→88.8% | 内存受限时收益小 |
| D4 | 整行缓存进 smem | softmax/layernorm | 全局只读一遍 | `07-softmax` | 融合 37.4% → **85.3%** | 行必须放得下 smem |
| D5 | online softmax | 行不可缓存（attention） | 边走边更新 running max/sum | `07`（引理） | 一趟完成 max+sum | 精度注意 rescale |

### E. GEMM / Tensor Core（08–14）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 | 坑 |
|---|---|---|---|---|---|---|
| E1 | smem tiling | GEMM | 分块复用，全局访存 ÷TILE | `08-gemm` | naive 8.2% → 13.5% | 之后瓶颈转 L1/TEX 载入管道 |
| E2 | 寄存器分块 thread tile | GEMM | 一次 smem load 供多个 FMA | `09/11` | 8×8 在 25% occ 达 46.8% | 2×2(100%occ) 只有 18.8% |
| E3 | 交错映射消 bank | GEMM smem 存 A | 让线程读到的 bank 打散 | `09` | +39.2% 段 | — |
| E4 | cp.async 双缓冲/N 级流水 | GEMM | 异步拷贝 + 多级预取 | `09/12` | 45%→51.5%→**53.6%** | 3 级后饱和；`long_scoreboard` 0.91→0.03 |
| E5 | mma.m16n8k16 + ldmatrix | bf16 GEMM | 裸 PTX 控布局 | `13` | WMMA 62 → **221 TFLOPS** | WMMA 在 sm90 是负优化（L1/TEX 77%） |
| E6 | 可插拔 epilogue 融合 | GEMM+激活 | 结果留寄存器，省 C 读回 | `14` | 融合仅慢 ~1%；sep 慢 11~33% | 独立对照 kernel 要先向量化，否则稻草人 |
| E7 | ILP ↔ occupancy 可互换 | 通用 | 提高每线程独立访存 | `11` | 12.5% occ 下 ILP 1→2：29.9→53.6 TFLOPS | 强行提 occ 会 spill（慢 34×） |
| E8 | `-Xptxas -v` 复核寄存器 | 模板化 kernel | 通用循环悄悄多花寄存器 | `09/11` | 118→168 寄存器 → 算力 -35% | `if constexpr` 走单发 | 

### F. 工具 / 工程（10/14）

| # | 技巧 | 适用场景 | 原理 | 代码 | 现象 | 坑 |
|---|---|---|---|---|---|---|
| F1 | ncu 看 stall 指纹 | 定位 stall | `smsp__average_warps_issue_stalled_*` | `10` | long_scoreboard=访存依赖 | 用 `--` 分隔 ncu 与程序参数 |
| F2 | source/SASS 对照 | 依赖链分析 | `--page source --print-source cuda,sass` | `10` | STL/LDL 定位 spill | — |
| F3 | 容器 profiling 权限 | ncu | 需 SYS_ADMIN/SYS_PTRACE | `scripts/lab.sh` | 否则 ERR_NVGPUCTRPERM | 用 kernel_lab，不用 kimi26_train |
| F4 | event 与 ncu 口径差异 | 写文 | ncu replay 会拉长 | `14` | event 8.4µs vs ncu 11.26µs | 同一结论用同一口径 |

### G. 模型场景 / 注意力（15）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| G1 | MLA「吸收」成 MQA | MLA 推理/前向 | `q_nope·k_nope = (W_ukᵀ q_nope)·c_kv`，输出 `o = W_uv(Σp·c_kv)`；KV 只剩 latent+rope，所有 head 共享 | `15/mla_attn.cu` | kernel 从「H 份 KV」变成「1 份 KV + 1 个 GEMM」 | 吸收后 `DV=kv_lora`，v 上投影是另一个 GEMM；RoPE 必须留在 nope 之外 |
| G2 | prefill MLA 是算力受限，不是访存受限 | 判断优化方向 | `c_kv` 仅 `Sk·576·2` B（S=4096 时 4.7MB）常驻 50MB L2 | `15` | ncu `DRAM Throughput` 0.71%、`Compute (SM)` 83.5%；读放大降 32× 耗时不变 | 别看到「读放大」就去做 smem/复用；先看 ncu 的 DRAM% |
| G3 | 标量 FFMA 天花板 <7% | MLA/attention 选型 | `AI_ideal=2H(DC+DR+DV)/(DC+DR)≈242`，但 FFMA ridge 仅 ~20 FLOP/byte，上限 FP32 pipe 66.9 TFLOPS | `15` | naive/head2/smem 全在 15~19 TFLOPS（1.9%）；改 TC 后 115.6（6.1×） | HF=8 复用触发 255 寄存器 + 328B 栈溢出，比 HG=2 还慢 |
| G4 | 别物化 S/P（attention 融合动机） | 长上下文 attention | 三 kernel 要把 `S`(fp32)+`P`(bf16) 各写读一遍，流量 `∝H·S²` | `15` | S=4096：25.8 GB≈7.7ms，占 TC 总时长 21%；S 越大越亏 | 三个「独立好 kernel」之和 ≠ 快；中间张量落显存是隐性税 |
| G5 | **累加器 C 布局 == PV 的 A 片段布局** | flash-attention / MLA 单 kernel 融合 | `mma.m16n8k16` 的 fp32 累加器 `c0,c1`（行 `lane/4`）与 A 片段 `a0,a1` 位置完全一致；相邻两个 n8 tile 的累加器拼成 PV 的 k16 A 片段 | `16/mla_fused.cu`（`pack2`） | P 转换**零 shuffle**，只 4 次 f32→bf16 pack；三 kernel 115.6→融合 146.5（`f4`，dup2） | n8 tile 必须**相邻**成对（覆盖 KV `[0,8)`+`[8,16)`）；online softmax 的行归约只需 4-lane shuffle |
| G6 | 用 smem 共享 P 消 QK 重复（dup） | DV 必须切分时的 MLA/GQA | `O[16,DV]` fp32 寄存器墙逼着按 DV 切 warp，代价是 QKᵀ 被重复算 `dup=DV/DVW` 次。让同 row 组的 warp **按 KV 对半分工**，行 max/sum 用 tiny smem 交换、P(bf16) 写 smem 共享读 | `16/mla_fused.cu`（`mla_shared_kernel`） | `f4`(dup2) 146.5 → `f4s`(dup1) **170.1 TFLOPS（+16%）**；P 共享只花 ~8KB smem | 别把 DV 切太细：`f2s`(DVGRP=4) 虽 dup=1，但 Q 被 4 个 warp 各读一遍，反而比 `f4s` 慢（113.9 vs 170.1） |
| G7 | wgmma 的胜负手是 **smem swizzle，不是指令** | Hopper wgmma | `wgmma` 从描述符读 A/B，但无 swizzle 的 K-major `INTERLEAVE` 布局会让 smem store 撞 bank、tensor 读操作数低效 | `16/mla_wgmma.cu` | mma 融合 170 → wgmma(INTERLEAVE) 只有 **84.6 TFLOPS（更慢）**；ncu：L1/TEX 86.4%、store bank conflict 2.7e8、tensor pipe 13.1% | 必须用 **SW128 swizzle**（描述符 layout_type=1 + `base_offset` 相位）；先做小 GEMM 冒烟测试验证描述符（见 `16/wgmma_helpers.cuh`） |
| G8 | **DSA indexer = H^I 个小 GEMM 共享同一个 K**：head 合并（HG）消 K 的重复 `ldmatrix` | DSA lightning indexer | $I_{t,s}=\sum_j w_{t,j}\mathrm{ReLU}(q_{t,j}\cdot k_s)$，$k_s$ 与 head 无关，所以每个 head 都把整个 K tile 重读一遍 → L1 85% 的元凶；把 HG 个 head 一起算，每个 `(kx,nb)` 只加载一次 B，K 的 L1 流量 ÷HG | `18-dsa-sparse/dsa.cu`（`indexer_kernel<BM,BN,W,HG>`） | S=16384：HG=1/BN=64 222 → HG=2/BN=64 252 → **HG=2/BN=128 287–311 TFLOPS**（+40%）；标量基线 0.93，总计 ~330× | HG 不能贪：HG=4/BN=128 寄存器顶到 255 + spill → 崩到 **69 TFLOPS**；每改一次都 `-Xptxas -v` 核对，`ncu` 看 L1/TEX% 是否真的降了 |
| G9 | 把 head 循环放对网格顺序：**`grid.x=KV` 让 Q tile 常驻 L2** | 任何 $O(S^2)$ 的 QK 类 kernel | 若 `grid.x=query`，所有 query block 在同一 KV 轮次同时活跃，Q 工作集 = 全量 Q > L2；Q tile 被反复从 DRAM 拉 | `18-dsa-sparse/dsa.cu` | S=32768：`grid.x=query` **69.7 TFLOPS** → 交换后 **214 TFLOPS（3.1×）** | 判据：ncu `DRAM Throughput` 高而 L1/L2 不高时，先怀疑 block 调度顺序而非算法 |
| G10 | exact top-k 用 **radix-select + per-warp 私有直方图**；别默认整行塞 smem | DSA / MoE 路由 / 采样 top-k | 只要第 k 大阈值，不需全排序：保序变换成 uint32，逐 8-bit 趟统计直方图定位阈值（4 趟），再收集 `>` + 补齐 `=`。独占瓶颈是 shared `atomicAdd` 竞争 → 每 warp 私有直方图 | `18-dsa-sparse/dsa.cu`（`topk_radix_*_kernel`） | S=32768：单直方图 26.7ms → **per-warp 6.75ms（3.96×，等效 3.2TB/s / 95% HBM）** | ①保序变换的逆**不自逆**（mask 依赖符号位）：高位置位取低 31 位、否则取反，写错则 `out_val` 全错；②整行塞 smem 在长序列反而更慢（S=32768：8.23 vs 6.75ms）——动态 smem 把 occupancy 锁成 1 block/SM；先算 occupancy 再决定 |
| G11 | **`wgmma` SS 的胜负手是 SW128 swizzle**：K-major + 128B 交换，物理列 `c'=c^r` | Hopper wgmma（A/B 从 smem 描述符直读） | 无 swizzle 的 `INTERLEAVE` 布局行距只有 16B，写侧撞 bank（2.7e8 次）、读侧 tensor pipe 仅 13%；SW128 把 128B 行内 16B 列按行号异或：`byte=(rg*(K/64)+kg)*1024+(rr*8+((c)^rr))*16+(k%8)*2` | `20-mla-wgmma-sw128/wgmma_sw128.cuh` | 同份 MLA：INTERLEAVE 84.6 → **SW128 105.3 TFLOPS（+24.5%）**；冒烟 GEMM 132（4096³）；store bank conflict 2.7e8 → 4.4-way（16B 存储理论下限 4.0） | 描述符布局：bit0-13 `start_address>>4`、bit16-29 `LBO>>4`（K-major SW128 恒 1）、bit32-45 `SBO>>4`（`(K/64)*1024`）、bit62-63 `layout_type=1`；k16 步进 `+floor(s/4)*1024+(s%4)*32`（**不是** `s*32`，跨 atom 要 +1024）；先写小 GEMM 验证描述符再上大 kernel |
| G12 | **相邻两个 bf16/半精度标量存储会被 nvcc 合并成 u32 并丢高 16 位** | 手写 swizzle/转置布局时把 fp32 结果写回 half 格式 | nvcc 看到同一线程两次相邻 2B 存储且首地址 4B 对齐，会合并成 `st.shared.u32`，但只写入低 16 位，高半变 0 | `20-mla-wgmma-sw128/bf16_store_pitfall.cu` | P 写回若用两个 bf16 存储：**3553 字节错位**（一半元素变 0）；改 `__floats2bfloat162_rn` 打包 u32 一次写 → 0 字节错位 | 症状隐蔽：QK/PV 单独定点测试全过、只有 P 写回错；任何“相邻两元素写 smem”都要显式打包或用向量类型（`uint32_t`/`__nv_bfloat162`）复核 |

### I. 稀疏注意力消费端 / DSA（19）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| I1 | 稀疏 attention 的并行轴是 **query token × head block**，不是 query block | DSA / 稀疏 MLA prefill | 同一 token 的所有 head 共享同一份 top-k 索引 $I_t$，所以 CTA 取「一个 token × $B_H$ 个 head」，gather 一次喂满整个 block；若 CTA 跨多个 query 位置，各位置 $I_t$ 不同，KV 无法共享 | `19-dsa-sparse-attn/sparse_mla.cu:79`（`sparse_mla_kernel`） | 132–137 TFLOPS；$B_H{=}64$ 时 $H/B_H{=}2$ 个 block 重复 gather，靠 L2 兜住 | $B_H$ 别太小：`s3`(BH=32) 变 4× 重复 gather，掉到 66 TFLOPS |
| I2 | gather 访存：**行内合并 + 行间随机**，索引同址广播走 L1 | 任何 top-k / 稀疏 gather KV | 每行（key）是 1KB 连续；让同一行的连续线程读连续 16B（合并 + 整行利用），行间索引随机无妨；`topk_idx` 被同行线程重复读但同址广播、L1 命中 | `sparse_mla.cu:145` | 行内仍是 `uint4` 合并，未因随机行而放大事务 | 别把索引先读到寄存器再散播——索引保留在 global 让硬件广播更省寄存器 |
| I3 | 掩码**只做尾部 tile** | 变长 / causal 稀疏 attention | 有效个数 `valid` 已知，前 $\lfloor valid/KT\rfloor$ 个 tile 全有效；只有最后一块可能不足 $KT$，才需要把无效列置 $-\infty$ | `sparse_mla.cu:424` | 掩码的 smem load+分支从每 tile 降到每 CTA 一次；实测 113→**133 TFLOPS** | 判断条件用「全局列号 $k_0{+}c\ge valid$」，别用 per-key 标志数组（多一趟 smem 写） |
| I4 | **`cp.async` 双缓冲**藏 gather 的 global 延迟 | 长上下文稀疏/稠密 attention | gather 打到 DRAM 时同步加载延迟直接暴露；用 `__pipeline_memcpy_async` 预取下一 KV tile 到 ping-pong 缓冲，与当前 tile 计算重叠 | `sparse_mla.cu:368`（`sparse_mla_pipe_kernel`） | `long_scoreboard` **4.56→1.46（3.1×）**；Sk=64k 反超同步版 11%（2.28 vs 2.54 ms） | 双缓冲 smem 翻倍：`p1`(KT=64) 需 234KB > 227KB 上限，只能配 KT=32；`wait_prior(has_next?1:0)` 别写错 |

### J. 访存布局 / 软流水（21）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| J1 | **转置/列访问的旧代码，先看线程编号顺序**：让「连续维度」当 `i` 的最低位 | wgmma SS 的 V 转置、任何「读列写行」 | 20 篇转置里 `i` 的低位是 `kb`，相邻线程地址差 `8*DC*2`=8KB、每个线程独占 sector；互换高低位让 `dv` 最快后相邻线程地址差 2B，一个 warp 覆盖连续 64B | `21-mla-wgmma-pipe/mla_pipe.cu:93` | `l1tex__throughput` 73.4%→45.8%、`long_scoreboard` 4.81→2.50；**105.3→142.3 TFLOPS（+35%）**，指令/寄存器不变 | 写侧仍受 SW128 16B 散布限制（4-way，理论下限）；先看 ncu `l1tex__throughput` 与 sectors/req，别只盯算法 |
| J2 | **`cp.async` 预取不一定要双缓冲**：只要该操作数的生命周期严格早于并行执行的计算 | 融合 attention / 链式算子的软流水 | K 只被 QK 的 `wgmma` 读，`wgmma.wait_group` 后即空闲；PV 不碰 K，于是可在 softmax/P/PV 期间用 `cp.async` 覆盖写同一块 `ks` | `21-mla-wgmma-pipe/mla_pipe3.cu:170`（`cp_async16` 在 :32） | `long_scoreboard` 2.50→1.42、tensor pipe 24.4%→30.0%；159.4→**193.2 TFLOPS**（Sk=4096），长上下文反超 f4s | 必须先论证「谁读谁、何时读完」；`cp.async.wait_group` 是 per-thread，需配 `__syncthreads` 才全局可见；目标地址 16B 对齐（SW128 每行 8×bf16 恰好） |
| J3 | MN-major（转置）GMMA 描述符别硬猜 | 想免掉 V 转置直接喂 `wgmma` | CUTLASS canonical MN-major B128 = `((8,n),(8,k)):((1,LBO),(8,SBO))`，物理 `[kt/8][dv/64][8][64]` + `c'=c^r` | `21-mla-wgmma-pipe/mn128.cuh`、`mn_test2.cu` | **未跑通**：`make_gmma_desc<Major::MN>` 的 LBO/SBO 与硬件解释**相反**；cute 值直接 out-of-range，强行对调后结果盐值错乱（D 读出重排） | 盲试描述符字段费时且不可靠；应改用 CUTLASS `make_tiled_mma` 生成描述符对齐，或保持转置+合并（J1） |

### K. FP8 GEMM / 低精度（22）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| K1 | **FP8 的 `ldmatrix` 用「两个 fp8 = 一个 b16」复用** | e4m3/e5m2 的 `mma.m16n8k32` | `ldmatrix` 吃 8×8 b16；2 个相邻 fp8 = 1 个 b16，于是 16×32 fp8 = 16×16 b16 = 4 个 m8n8 矩阵，`x4` 一次取回 `a0..a3`；B 存 `[N][K]` 用 `x2` 取 `b0/b1` | `22-fp8-gemm/fp8_gemm.cu`（`ldmatrix_x4/x2`） | 朴素按坐标取 4B → ldmatrix 后同尺寸仍受发射限制，但省 4× 载入指令；smem 行距 `BK+16` 消 bank | k32 步内 `a0/a1` 是 k 0-15、`a2/a3` 是 k16-31；地址 lane 公式 `row=(lane&7)+((lane>>3)&1)*8, col=(lane>>4)*16` |
| K2 | **Hopper 上打满 FP8 峰值必须换 `wgmma`** | FP8/bf16 GEMM | `mma.sync` 走 SM80 兼容路径，发射带宽受限；`wgmma.m64n128k32` 异步、SS 直读 smem 描述符 | `22-fp8-gemm/fp8_gemm_wgmma.cu` | 同 shape：mma.sync 266 → **wgmma 768 TFLOPS（+2.89×）**；ncu `math_pipe_throttle` 1.12 → 消失 | FP8 的 wgmma asm 尾部操作数是 `p, scaleA, scaleB`（**3 个**，不是 bf16 的 5 个）；操作数编号：64 累加器后 `da=%64,db=%65,ped=%66,scA=%67,scB=%68` |
| K3 | **FP8 的 K-major SW128 atom 与 bf16 逐字节同构** | wgmma SS 的 fp8 操作数 | CUTLASS `Layout_K_SW128_Atom_Bits` 按 bit 定义、`upcast` 到 fp8 后是 8 行×128 字节（bf16 是 8 行×64 元素=8×128B）；Swizzle<3,4,3> 作用在字节上 | `22-fp8-gemm/fp8_gemm_wgmma.cu`（`sw_off`/`make_desc_sw128`） | 20 篇的 `wgmma_sw128.cuh` 原样复用；`SBO=(K/128)*1024`、k32 步进 `(s/4)*1024+(s%4)*32` | GEMM 里 `B[N][K]` 天然 K-major，**无需转置**；attention 里 V 才要转置 |
| K4 | **per-tensor 的 wgmma 主循环别每块 `wait0`** | 融合/GEMM 的 wgmma 软流水 | 累加器到最后才读时，用 `wgmma.commit_group` 每块提交、覆盖 stage 前只 `wgmma.wait_group STAGES-2`，允许 mma 跨块流水 | `fp8_gemm_wgmma.cu`（`wgmma_wait_group<N>`） | 256×128×BK128 s3：710 → **768 TFLOPS** | 循环外要补一次 `wait0` 再读 acc；wait_group 只保证「本 warp 的 mma」，覆盖 stage 前仍需 `__syncthreads` |
| K5 | **per-block 缩放：每 128-k 块折算，注意相邻列各有 scale** | DeepSeek/V4 的 e4m3 + block scale | 固定 k 块内 `sa/sb` 是标量 → 先无缩放累加 128 个 k，再 `fin += sa*sb*acc`（DeepGEMM 的 final_accum） | `fp8_gemm_wgmma.cu`（PERBLOCK） | per-block 519 vs per-tensor 768（折算每块 `wait0` 切断流水）；`weight_block=128×128` 让 `sb` 每个 n-tile 退化成 1 个标量 | **c0 与 c1 是相邻两列，`sb` 不同**：`sbv0=sb[col]` 给 c0/c2、`sbv1=sb[col+1]` 给 c1/c3；只测试常数 scale 发现不了（会「通过」），必须用逐列随机 scale |
| K6 | **采样 check 要打散到全区间，别被 `__launch_bounds__` 骗** | 扫 config 时防假阳性 | BM=256 需 4 warpgroup/512 线程，若写死 256 只会算一半行；若采样步长 `%BM` 后总是落在已算区域，就会「OK 且快一倍」 | `22-fp8-gemm/fp8_gemm_wgmma.cu` | 曾把 256×256 的假成绩当成 898 TFLOPS；修正后真值 218 | 用与 BM/BN 互质的步长（如 `i*1009`），或对每个输出块至少采一个点 |

### L. TMA / warp specialization（23）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| L1 | **TMA 2D + SW128 与 wgmma 描述符逐字节同构** | Hopper GEMM/attention 的 A/B 装载 | `CUtensorMap` 把 tile 形状/坐标/swizzle 编码进 128B 描述符，`cp.async.bulk.tensor.2d` 一条指令搬整块并按 **`CU_TENSOR_MAP_SWIZZLE_128B`** 写 smem，布局恰等于 wgmma 期望的 K-major SW128（8 行×128B，`chunk^(row%8)`） | `23-fp8-gemm-tma/fp8_gemm_tma.cu:98`（`tma_load_2d`）、`:239`（`make_tmap`） | 搬运从「每线程算 swizzle + `LDGSTS`」变成「1 线程 1 指令」；同 shape **768 → 1217 TFLOPS（+58%）** | ①`CUDA 13` 驱动枚举 **没有** `CU_TENSOR_MAP_DATA_TYPE_FLOAT8_E4M3`，用 `UINT8` 搬字节；②`BK` 锁 128（SW128 内维上限 128B）；③需 `-lcuda` 链接 `cuTensorMapEncodeTiled`；④tensormap 必须作 `const __grid_constant__` 内核参数 |
| L2 | **mbarrier 生产者/消费者 + warp specialization** | 把装载和计算解耦 | 1 个 producer warp 专职发 TMA：`mbarrier.arrive.expect_tx` 声明字节数，TMA 完成自动补齐相位；消费者每个 warpgroup 只做 wgmma，读完一个 stage 后 `arrive(empty)`（count=消费者线程数） | `fp8_gemm_tma.cu:143`（producer）、`:159`（consumer） | 释放 255 个线程的发射槽与地址寄存器；No Eligible 从 60% 降到 …，张量管线活跃度 **72%** | `empty` 的 count 要等于**所有**消费者线程数（只让 lane0 arrive 会与同 WG 另一 warp 的 `wait_group` 竞争）；覆盖 stage 前生产者的 acquire 要配 `fence.proxy.async.shared::cta` |
| L3 | **mbarrier 相位别用动态下标数组**（会掉 local memory） | 多级流水相位跟踪 | `uint32_t phase[STAGES]` + `phase[st]^=1`（`st=kb%STAGES` 运行期）→ 数组进 local memory；实际相位就是「该 stage 第 n 次使用」的奇偶，可直接算 | `fp8_gemm_tma.cu:173`（`(kb/STAGES)&1`）、`:149`（`(kb/STAGES-1)&1`） | 第一版 local memory 占 **L1TEX 47.5% 的 sector**、`long_scoreboard` 8.1，只有 988；改成常量相位后 **→ 1217（+23%）** | 症状隐蔽：`Local Memory Spilling Requests = 0` 但 ncu 报 local memory 占比高；看 `Memory Workload Analysis → local memory` 而不是只看 spill |
| L4 | **epilogue 用 `float2` 向量存储** | mma/wgmma 累加器写回 | 累加器相邻两列 `cc, cc+1` 若分两条 4B store，同 warp 相邻 lane 步长 8B，各用半 sector；合并成一次 8B `float2` 后 lane 0-3 拼成连续 32B | `fp8_gemm_tma.cu:231` | ncu 从 **50% global sector excessive** 降到接近 0 | 首地址需 8B 对齐（`cc` 是偶数）；`N` 任意时用 `if (r<M)` 掩码 |
| L5 | **算力受限 GEMM：降 L2 流量（大 BM）> 堆 occupancy** | 大 GEMM 分块选型 | BM 越大，A tile 被同列 tile 复用的次数越多，TMA 次数与 L2 压力越小 | `fp8_gemm_tma.cu`（`BM/BN` 模板） | 128×128 s3（2 CTA/SM，L2 80%）**1097** < 256×128 s4（1 CTA/SM，L2 57%）**1217** | 别默认「occupancy 越高越好」；先看 ncu `L2 Cache Throughput` 与 tensor pipe 活跃度 |

### H. 优化器 / MuonClip（17）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| H1 | 优化器算子先算 roofline | Muon/NS、Shampoo 等 | NS 正交化 = 5 步 × 3 GEMM = `30N³` FLOPs；`AI=O(N)`，工作集仅 `2N²` | `17-muonclip-ns/ns_muonclip.cu` | ncu `DRAM 13.4%`、`Compute 45%` → **算力受限**；优化目标是把 GEMM 做快 | 别一看到「优化器」就以为访存受限；先看 ncu 的 DRAM% |
| H2 | 融合 epilogue `f·x+g(GEMM)` | 链式 GEMM（NS/优化器） | mma 累加器 `c0,c1`(行 `lane/4`)/`c2,c3`(行+8) 坐标固定，结果还在寄存器时按同坐标读 `Cin` 做 `sc_c·Cin+sc_d·acc` | `17-muonclip-ns/ns_muonclip.cu`（`store_acc_epi`） | 消掉每步 2 趟 `axpby`（读 2 写 1）；N=4096 端到端 8.93→**8.44 ms（−5.5%）**；epilogue 不涨寄存器（126/0 spill） | `Cin` 与输出必须不同 buffer（`M=bA+cAA` 中 `Cin=A` 是另一 buffer，非原地别名）；尺寸越大收益越小 |
| H3 | `mma` 路径的天花板是 L2 放大 + 寄存器墙 | 大 N 方阵 GEMM | `128×128` 分块下 A/B 各被重复读 32 次（每 GEMM ~2GB L2 流量）；块放大则每线程累加器翻倍 → 126 寄存器、occupancy 23.8% | `17-muonclip-ns/ns_muonclip.cu` | 自研单 GEMM 269 TFLOPS（cuBLAS 883），卡在 `L2 76% / occ 24%`；`128×256`/`256×128` 更慢 | 出路是 `wgmma`(SS 直读 smem，免 `ldmatrix`) + TMA 压 L2；见路线图 31/38 |
| H4 | 多级 `cp.async` 的 `wait_prior` 参数 | 通用软流水 | N 级流水 prologue 预取 N−1 级、循环每轮再 commit 1 级；等「当前 stage」应 `wait_prior(N-1)` 而非 `N-2` | `17-muonclip-ns/ns_muonclip.cu` | 写成 `N-2` 会过早放行、读到半写 smem（`max_abs_err` 爆到 1e6） | 动态 smem 手搓多级缓冲时 B 的 stage 基址要放在 A 的**全部** stage 之后（`smem + STAGES×BM×LDP`），漏乘 `STAGES` 会 A/B 槽覆盖 |

---

## 三、模型场景台账（15 起填充）

| # | 模型 | 算子 | 关键 shape/参数 | 手段与结论 | 代码 | 实测 |
|---|---|---|---|---|---|---|
| M1 | DeepSeek-V3/V4、Kimi-K2.6 | MLA | V3: H=128,DC=512,DR=64,DV=512（absorb 576/512）；V4-Pro: H=128,DC=448,rope64；V4.1: H=64；Kimi: H=64,qk=192,v=128,kv_lora=512 | 吸收成 MQA。15：TC 三 kernel 115.6（S=1024）。**16：单 kernel 融合**（online softmax 进 QK epilogue、C→A 零 shuffle、KV 常驻 smem）；用 smem 共享 P 消 QK 重复 → `f4s` **170.1**（S=1024）/ **184.7**（Sk=4096）；DRAM 仅 7%，瓶颈=L1 的 ldmatrix + 12.5% occ；距 FlashMLA 640 ~3.5×。**20：wgmma SS + SW128 swizzle** → 105.3/112.9/119.9（Sk=1k/2k/4k），仍 1.62× 慢于 `f4s`；ncu：L1 70.8%、occ 12.5%、221KB smem。**21：修 V 转置访存**（迭代顺序 dv 最快，L1 73%→46%，+35%）+ **`cp.async` 单缓冲预取下一块 K**（`long_scoreboard` 4.81→1.42）→ **159.8/193.2/198.8**（Sk=1k/4k/8k），**Sk≥4096 反超 `f4s`**，距 FlashMLA ~640 收窄到 **~3.3×**；ncu：tensor 30%、L1 46%、barrier 1.17/wait 1.01 成下一道墙 | `15-mla-attn/mla_attn.cu`、`16-mla-fused/mla_fused.cu`、`20-mla-wgmma-sw128/mla_wgmma_sw.cu`、`21-mla-wgmma-pipe/mla_pipe.cu`/`mla_pipe3.cu` | 15 tc: 2.53/9.41/36.58 ms；16 f4s: 1.72/3.18/6.33 ms；20 wgmma_sw: 2.77/5.17/9.74 ms；21 pipe3: 1.83/3.28/6.06/11.75 ms（Sk=1k/2k/4k/8k） |
| M2 | DeepSeek-V4-Pro / V4.1 | DSA 稀疏注意力（一）：indexer + top-k | index_n_heads=64(V4-Pro)/32(V4.1), index_head_dim=128, index_topk=1024(V4-Pro)/512(V4.1) | 18：indexer 是「H^I 个小 GEMM 共享 K」，TC+head 合并 HG=2 达 **287–311 TFLOPS（31% 峰值）**；exact top-k 用 radix-select + per-warp 直方图 **6.75ms @S=32768**。DSA 端到端（indexer+topk+稀疏 MLA 估算）相对稠密 MLA：4k 2.9× / 16k 9.7× / 32k 14.8× / **64k 18.6×**，上限 ~30×。ncu：indexer DRAM 2.9%、L1 62%；topk IPC 3.35、issue 83% | `18-dsa-sparse/dsa.cu` | indexer S=16384 HG2/BN128 15.32ms；topk S=32768 6.75ms；dense MLA f4s S=65536 6914.8ms |
| M2b | DeepSeek-V4-Pro / V4.1 | DSA 稀疏注意力（二）：稀疏 MLA 消费端 | H=128(V4-Pro)/64(V4.1), DC=512,DR=64,DV=512, topk=1024/512 | 19：CTA = 1 token × $B_H$ head，gather top-k 的 $c_{kv}$+$k_{rope}$、尾部 tile 掩码、共享 P、`cp.async` 双缓冲。实测 132–137 TFLOPS（~75% 稠密 f4s 效率）；Sk=64k 时 sparse attention 比稠密快 **48.6×**；`long_scoreboard` 4.56→1.46。DSA 端到端（含 18 的 indexer/topk）4k 2.5× / 32k 13.3× / 64k **17.3×**；距 FlashMLA sparse prefill 640 约 4.8× | `19-dsa-sparse-attn/sparse_mla.cu` | p2: 4096/8192/16384/32768/65536 → 136.3/129.9/133.3/132.4/127.9 TFLOPS；dense f4s 6.36→110.87 ms |
| M3 | DeepSeek-V4 | MoE | 384 routed+1 shared, top-6, inter=3072 | *待填（23）* | — | — |
| M4 | DeepSeek-V4-Pro / V4.1 | FP8 GEMM | hidden=7168, moe_inter=3072；e4m3 + ue8m0, weight_block 128×128（V4.1 为 32×32 + expert fp4）, activation dynamic 1×128 | 22：真实 shape 4096×3072×7168。①`mma.m16n8k32`+ldmatrix(b16)+`cp.async` 流水 → **266 TFLOPS（13.5%）**，ncu tensor pipe 41%、`math_pipe_throttle` 1.12（发射受限）；②换 `wgmma.m64n128k32`+SW128（与 bf16 同构）+`wait_group` 流水 → **768 TFLOPS（38.8%）**，+2.89×。**23：TMA（`cp.async.bulk.tensor.2d`+SW128）+ mbarrier + warp specialization** → 同 shape **1217.3 TFLOPS（61.5%）**，达 cuBLAS FP8（1381.3）的 **88.1%**（差距 1.13×），张量管线活跃度 72%；③per-block（1×128 激活 + 128×128 权重）22 的 cp.async 版 519 → **23 的 TMA 版 906（45.8%，+74%）**，瓶颈转为「每 128-k 块折算等 mma」的流水断裂（寄存器 150、occ 13.8%）。关键坑：相位数组动态下标掉 local memory（988→1217）、epilogue 标量 store 半 sector（float2 修复） | `22-fp8-gemm/fp8_gemm.cu`、`fp8_gemm_wgmma.cu`、`23-fp8-gemm-tma/fp8_gemm_tma.cu` | 22：mma.sync 266、wgmma 768、per-block 519；23：TMA 128×128s3 1097 / 256×128s4 **1217**、per-block 906；cuBLAS FP8 1381.3 |
| M5 | Kimi-K2.6（hidden=7168/18432, moe=2048, 384 experts） | MuonClip / Newton–Schulz 正交化 | 5 步 NS，`30N³`；测 N=2048/4096/8192 | 还原成 15 个 GEMM 的链式算子。17：自研 `mma` GEMM + 融合 `f·x+g` epilogue，**244 TFLOPS（N=4096, 24.7%）**；单 GEMM 269 vs cuBLAS 883（30.5%）；端到端 cuBLAS 链 529（2.17×）。ncu：L2 76%、occ 24%、DRAM 13% | `17-muonclip-ns/ns_muonclip.cu` | 17 N=2048/4096/8192: 1.31/8.44/65.4 ms |
| M6 | Qwen3 | GQA attention | q/kv=40/8, 32/4, 64/4 | *待填（25）* | — | — |

---

## 更新约定

- 每个新技巧一行，填全「适用场景/原理/代码/实测收益/坑」，收益写**真实数字**。
- 性能榜每次刷新最佳实现与峰值占比；对标 SOTA 要写明来源与口径。
- 卡片编号延续（A/B/C…；M 前缀用于模型场景）。
