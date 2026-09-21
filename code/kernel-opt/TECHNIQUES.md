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
| DSA 稀疏注意力 | *待填（17）* | — | — | 自研 kernel | — |
| MoE grouped GEMM | *待填（21）* | — | — | DeepGEMM | — |
| FP8 GEMM | *待填（22/23）* | — | — | DeepGEMM | — |
| MuonClip 正交化 | *待填（24）* | — | — | muonclip | — |

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

---

## 三、模型场景台账（15 起填充）

| # | 模型 | 算子 | 关键 shape/参数 | 手段与结论 | 代码 | 实测 |
|---|---|---|---|---|---|---|
| M1 | DeepSeek-V3/V4、Kimi-K2.6 | MLA | V3: H=128,DC=512,DR=64,DV=512（absorb 576/512）；V4-Pro: H=128,DC=448,rope64；V4.1: H=64；Kimi: H=64,qk=192,v=128,kv_lora=512 | 吸收成 MQA。15：TC 三 kernel 115.6（S=1024）。**16：单 kernel 融合**（online softmax 进 QK epilogue、C→A 零 shuffle、KV 常驻 smem）；用 smem 共享 P 消 QK 重复 → `f4s` **170.1**（S=1024）/ **184.7**（Sk=4096）；DRAM 仅 7%，瓶颈=L1 的 ldmatrix + 12.5% occ；距 FlashMLA 640 ~3.5× | `15-mla-attn/mla_attn.cu`、`16-mla-fused/mla_fused.cu` | 15 tc S=1024/2048/4096: 2.53/9.41/36.58 ms；16 f4s: 1.72/3.18/6.33 ms |
| M2 | DeepSeek-V4 | DSA 稀疏注意力 | index 64×128, topk=1024, compress 4/128/0 | *待填（17）* | — | — |
| M3 | DeepSeek-V4 | MoE | 384 routed+1 shared, top-6, inter=3072 | *待填（20/21/29）* | — | — |
| M4 | DeepSeek-V4 | FP8 GEMM | e4m3 + ue8m0, block 128×128 | *待填（22/23）* | — | — |
| M5 | Kimi | MuonClip | Newton–Schulz 5 步正交化 | *待填（24）* | — | — |
| M6 | Qwen3 | GQA attention | q/kv=40/8, 32/4, 64/4 | *待填（25）* | — | — |

---

## 更新约定

- 每个新技巧一行，填全「适用场景/原理/代码/实测收益/坑」，收益写**真实数字**。
- 性能榜每次刷新最佳实现与峰值占比；对标 SOTA 要写明来源与口径。
- 卡片编号延续（A/B/C…；M 前缀用于模型场景）。
