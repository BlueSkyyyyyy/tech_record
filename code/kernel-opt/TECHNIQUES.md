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
| FP8 e4m3 GEMM（per-block，TMA） | wgmma + 每 128-k 块折算 `sa[m,kb]*sb[nblk,kb]`，128×128 s4 | **936.5 TFLOPS** | 47.3% | 23 篇 906.2 → +3.3%；寄存器 154、occ 13.7%（1 CTA/SM）；同配置 per-tensor 938.7 | 24-fp8-gemm-pb |
| FP8 per-block vs per-tensor（同一二进制） | 关掉折算的 256×128 s4 对照 | **1209.6 TFLOPS**；per-block 只能 936.5 | 61.2% | 差距根因=per-block 的 `fin` 多 64 寄存器→occupancy 2→1 CTA；BM=256 需 128 累加器寄存器 > 120 预算 | 24-fp8-gemm-pb |
| **MoE grouped GEMM（contiguous, prefill）** | 单 kernel 覆盖 384 expert（B 行坐标=`group*N+n`）+ TMA+wgmma | **958–1008 TFLOPS**（aligned）@tokens=16k/32k | 48–51% (1978) | per-expert loop 13.2–14.1ms → **5.59/9.58ms（2.36×/1.47×）**；反超 cuBLAS per-expert loop（graph）**1.17×** | 25-moe-grouped-gemm |
| MoE grouped GEMM（masked, decode） | 同一 kernel，早退 `block_row>=masked_m[g]` | **3.02 ms**，B-read **2802 GB/s** | **83.6% HBM / ~89% 含 D 写** | cuBLAS padded batched（graph）4.62ms → **1.53×** | 25-moe-grouped-gemm |
| **MoE grouped GEMM · per-block FP8（当前最佳）** | **grouped（B 行坐标=`group*N+n`）+ per-block 128×128 scale（TMA+wgmma+WS）** | **806.4 TFLOPS** @tokens=32768（128×128 s3） | 40.8% (1978) | 同二进制 per-tensor 980.5 → **82.2%**；稠密单 GEMM per-block 936.5 / per-tensor 1209.6 = **77.4%** → **分组让 per-block 损失更小** | 31-moe-grouped-pb |
| MoE grouped GEMM · per-block（16k / 8k） | 同上 | **790.4 / 707.7 TFLOPS**（均为 128×128 s4） | 40.0 / 35.8% | 同二进制 per-tensor 960.5 / 777.4 → **82.3% / 91.0%**；小 batch 损失更小 | 31-moe-grouped-pb |
| MoE grouped GEMM · per-block · masked decode | masked 早退 + per-block，BM=64/BN=128 s3 | **2.95 ms**，B-read **2862 GB/s** | **85.4% HBM** | per-tensor 25 篇 2.8 TB/s；BM=128 仅 75.4% → **decode 上折算免费** | 31-moe-grouped-pb |
| MoE grouped GEMM + **A-multicast**（cluster CN=2） | TMA `...multicast::cluster`（leader 广播 A，私有/共享 empty barrier 分离） | **1001.6 TFLOPS** @tokens=32768, 128×128 s3（9.6406 ms） | 50.6% (1978) | CN=1 960.7 → **+4.3%**；L2 读 sector **−18.8%**、`long_scoreboard` 11.6→8.0；**仅大 batch 有效**（8k −7.4% / 16k −2.2% / masked −3.9%） | 26-moe-cluster-multicast |
| MoE grouped GEMM · BM=256（想消 B 放大，负结果） | 256×128/256×256 tile，ALIGN=256 | 256×128 s4 **788.5 TFLOPS/useful**；256×256 **124**（崩） | 39.8% / 6.2% | 同分布 128×256 736.6（useful）→ 仅 +7%；padding 10.8%→19.6%，256×256 触发 ptxas `C7511` wgmma 串行化 | 26-moe-cluster-multicast |
| MoE gate GEMM（router，bf16） | mma.m16n8k16+ldmatrix+cp.async 双缓冲，BM=128/BN=128 | **204→257 TFLOPS** @M=16k/32k（4096×…×7168→384） | 20.7–25.9% (989) | cuBLAS 738.2 → **27.7%**；BM=256 反而 −10%（1 CTA/SM） | 27-moe-router |
| MoE gate GEMM · wgmma+cp.async | wgmma.m64n128k16 SS + SW128 + cp.async 流水，BK=64 | **337.1 / 359.6 TFLOPS** @M=16k/32k | 34.1–36.4% (989) | 27 篇 mma.sync 201.9/262.3 → **+67%/+37%**；s3（2 CTA/SM）> s4（1 CTA/SM） | 28-moe-gate-wgmma |
| **MoE gate GEMM · 当前最佳** | **wgmma SS + TMA + mbarrier + warp specialization**（1 producer warp + 2/4 consumer WG） | **547.0 TFLOPS**@16k（128×128 s3）/ **612.2**@32k（256×128 s4） | **55.3% / 61.9%** (989) | cuBLAS bf16 734.6/752.6 → **74.5% / 81.3%**（28/29 篇同口径）；相对 27 篇 **2.71×/2.33×**；tensor pipe 68.8%、L2 80.4%、occ 26%、0 bank conflict | 29-moe-gate-multicast |
| MoE gate GEMM · A-multicast（cluster.x=3，负结果） | TMA `...multicast::cluster` 广播 A（同 m-tile 的 3 个 n-tile 共享） | 468.7 TFLOPS @32k（256×128 s4） | 47.4% (989) | L2 读 sector 46.2→**34.0 M（−26%）**、L2 吞吐 80%→48.5%，但 **−23.4%**：耦合把 tensor/SM 68.8%→51.5% | 29-moe-gate-multicast |
| MoE gate GEMM · B-multicast（cluster.y=2，小幅赢） | TMA `...multicast::cluster` 广播权重 B（同 n-tile 的 2 个 m-tile 共享） | **587.2 TFLOPS** @32k（128×128 s3） | 59.4% (989) | 该 config 上 +5.6%（556.3→587.2）；L2 读 sector 71.2→64.1 M（−10%）；仍输给直接放大 BM 的 612.2 | 29-moe-gate-multicast |
| MoE router top-k（384 选 6, sqrtsoftplus+bias） | 1 warp/token，8 warp/block，6 轮 argmax + shfl 归约 | **0.031 ms** @M=16384 | compute 62.6%、occ 84.7% | CPU 对拍 top-6 集合 8/8、权重 err 6e-8 | 27-moe-router |
| MoE token permute | **1 block/token**（x 读一次、写 K 行） | **0.605 ms** @M=16384, 1.64 GB | **2717 GB/s（81.0% HBM）** | vs 1 block/行（读 x K 次）**1.53×**；torch index_select 1.50× | 27-moe-router |
| MoE token unpermute（加权反置换） | 1 block/token，bf16 累加 fp32 | **0.535 ms** @M=16384, 1.64 GB | **3075 GB/s（91.7% HBM）** | torch index_add **17.8×** | 27-moe-router |
| FP8 e4m3 GEMM（per-block，cp.async） | wgmma + 每 128-k 块折算 | 519.0 TFLOPS | 26.2% | per-tensor 的 68%（折算切断流水）| 22-fp8-gemm |
| MuonClip NS 正交化（N=4096） | 自研 GEMM 链 + 融合 epilogue（cfg1） | **244 TFLOPS** | 24.7% (989) | cuBLAS 链 529（53.5%）→ 差距 2.17× | 17-muonclip-ns |
| **fused MoE expert FFN（bf16，3 kernel）** | **gather A + 双累加器 SwiGLU + unpermute scatter** | **42.14 ms** @M=16384（un-fused 45.44） | 端到端 **1.08×**；activation 流量 −81%、总流量 −13% | 同 shape 稠密 cuBLAS 参考见下 | 30-fused-moe |
| **fused MoE expert FFN（FP8 per-block，当前最佳）** | **grouped up/gate + swiglu/quant（一行一 block）+ grouped down（TMA+wgmma+WS）** | **11.80 / 20.58 / 15.93 ms** @M=8k/16k bal、8k rand | 相对 30 篇 bf16 **1.92× / 2.21× / 2.06×**；权重字节 50.7→**25.4 GB** | K2 down **88.1% HBM**（ncu DRAM 89.3%）；K1 78%；BF16 上限参考同 30 篇 | 34-fused-moe-fp8 |
| MoE down grouped GEMM · FP8 per-block（K2，N=7168） | grouped（B 行坐标=`group*H+n`）+ per-block 128×128 + TMA/wgmma/WS | **3.39 ms** @Pp=49152 | **88.1% HBM**（10.01 GB / 3.39 ms），ncu DRAM 89.3%、L2 86.0%、156 regs | 超 31 篇同手段（31 是 N=3072 的 up/gate 形状） | 34-fused-moe-fp8 |
| MoE down + unpermute 融合 · FP8（负结果） | `red.global.add.f32` 散写 `Yf` | **3.83 ms** vs 分开 3.35+0.54=3.89… 端到端反而 0.96× | ncu DRAM 73.2%、**L2 94.4%**：散写把瓶颈从 DRAM 顶成 L2 | 30 篇 bf16 下同一技巧是 +10%；GEMM 变快后失效 | 34-fused-moe-fp8 |
| SwiGLU + per-128 动态量化 | 一行一 block + `float4` 读 + warp 内 `__shfl_xor` amax + fp8 打包 u32 | **0.44 ms** @Pp=49152（v1 1.07） | **3090 GB/s / 92% HBM**，2.4× | 读 `GU` 1.21 GB + 写 `A2` 0.15 GB；FP8 融合前先把它做快 | 34-fused-moe-fp8 |
| fused MoE · K2 down grouped GEMM（bn256） | grouped down + `red.add` unpermute，MK=128×256 | **898 TFLOPS**（nofuse）/ 858（fuse）@M=8192 | **90.8% / 86.8%** (989) | 同 shape 稠密 cuBLAS 711 → **1.26×**；ncu DRAM 79.5%、L2 88.1% | 30-fused-moe |
| fused MoE · K1 up+gate grouped GEMM | gather A + SwiGLU epilogue，BM=128/N=256/s3 | 492–513 TFLOPS @M=8k–16k | 50–52% (989)，但 **DRAM 84.5% 权重带宽受限** | roofline 下限 10.1 ms（33.8 GB 专家权重 / 3.35 TB/s），实测 13.2 ms | 30-fused-moe |
| MuonClip NS 单 GEMM（4096³） | 自研 128×128×64 3 级流水 | **269 TFLOPS** | 27.2% | cuBLAS 883（89.3%）→ 达其 30.5% | 17-muonclip-ns |
| **RMSNorm（H=7168）** | **寄存器缓存整行 + 单读单写**（`v1r`） | **2869 GB/s** | **85.6% HBM**（ncu DRAM 82.9%） | 朴素两趟 1481（44.2%）→ 2.0×；smem 单读 2334（69.6%） | 32-fused-norm |
| **融合 add+RMSNorm（H=7168）** | **寄存器缓存 + 融合残差**（`v2r`） | **2923 GB/s** | **87.2% HBM**（ncu DRAM 86.2%） | 5 pass→4 pass；离「2 读 2 写」物理极限很近 | 32-fused-norm |
| 融合 add+RMSNorm+FP8 per-128 量化 | 寄存器缓存 + shared `atomicMax` amax | 2205 GB/s | 65.8%（ncu DRAM 64.8%、SM 50.2%） | 只省 8B→7B（−12.5%）却新增 amax 归约/打包 → **净变慢** | 32-fused-norm |
| QK-Norm（Qwen3 40q/8kv×128） | 每 warp 一个 (token,head)，shfl 归约 | 2179 GB/s | 65.0% HBM | warp 只搬 256B、ILP 低；应一 warp 串多个 head | 32-fused-norm |
| **DSA compressor 合并投影（bf16）** | **`wkv`+`wgate` 拼成 `[2C,D]` 单 GEMM + TMA/wgmma/WS** | **649.8 / 655.2 TFLOPS** @M=32768（ratio=128/4，256×128×64 s4） | **65.7 / 66.3%**（989） | cuBLAS 合并 bf16 809/814 → **~80%**；两次独立 bf16 cuBLAS 0.630/1.398 → 合并 0.595/1.185（**A 只读一遍，5.5%/15%**） | 35-dsa-compressor |
| DSA compressor 融合池化（online softmax + RMSNorm + RoPE） | 一 block 一窗、线程=列、一 kernel 三合一 | **2812 / 2790 GB/s** @M=32768, 134/268MB | **83.9 / 83.2% HBM**（ncu DRAM 84/87.2%） | eager 池化 0.315 ms → **0.048 ms（~6.6×）** | 35-dsa-compressor |
| **DSA compressor 端到端** | 合并投影 + 融合池化 | **0.790 / 1.577 ms** @M=32768, ratio=128/4 | e2e 相对官方 eager **2.00× / 2.04×**（M=8192 达 2.39×/2.14×） | 池化环节 ~6.6×；投影 1.70× | 35-dsa-compressor |
| **DSA compressor 合并投影 · bf16（最佳几何）** | `256×128×64 s4`（TMA+wgmma+WS） | **692.5 TFLOPS** @M=32768,ratio=128（698.8 @ratio=4） | **70.0%**（989） | ncu：tensor pipe **76.4%** 受限、L2 58% → 35 篇的「L2 墙」是 `BM=128` 配置假象；距离 cuBLAS 合并 bf16（809）1.17× | 36-dsa-compressor-mcast |
| **DSA compressor 合并投影 · FP8 e4m3 per-block（当前最佳）** | e4m3 + 1×128 激活 / 128×128 权重块缩放，TMA+wgmma.m64n128k32 | **959.1 TFLOPS** @ratio=128 / **974.1** @ratio=4, M=32768 | **48.5% / 49.2%**（1978） | 相对 bf16 **1.385× / 1.394×**；权重字节减半；量化后池化输出最大误差 3.2%/3.8%；ncu 瓶颈翻转为 L2 73% + barrier（occ 13.8%、154 regs） | 36-dsa-compressor-mcast |
| DSA compressor 投影 · TMA cluster multicast（负） | A 沿 `cluster.x` / B 沿 `cluster.y` 广播，CN=2/4/8 | 609 / 597 TFLOPS（CN=2）→ 537 / 513（CN=8） | vs 基线 632 **−3.7% ~ −19%** | 字节省了但 tensor 活跃度被跨 CTA 耦合拖低；tensor>70% 时 multicast 无收益 | 36-dsa-compressor-mcast |
| DSA compressor 投影 · L2 persisting（负） | `cudaAccessPolicyWindow` 钉住 `Wm` 14.7MB | 618.7 TFLOPS | vs 632 **−2.2%** | hit rate 已 82%，瓶颈不在命中率 | 36-dsa-compressor-mcast |

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
| J4 | **判决：Hopper `wgmma` 的 B 操作数只认 K-major（MN-major 描述符无效）** | 想用「免转置 V」省掉 MLA 的 `c_kv` 转置 | 32 篇用 CuTe 生成**正确的** canonical MN 描述符（`desc_probe*.cu` 打印 raw=`0x4000010000400000`，LBO=64/SBO=256 与手推一致），再跑定点 GEMM：**改 LBO 字段结果完全不变**——说明硬件把 B 的 `leading_byte_offset` 当「swizzle 下忽略=1」，即按 K-major 解释；观测到的访存模式是「输出 n 每 +1 走 128B（=一行 key）」，正是 K-major | `32-mla-tma-v/mn_smoke.cu`、`desc_probe2/3/4.cu` | MN-major descriptor 下 `max_abs_err` 恒 ~0.15（ref 0.098），且 LBO 从 64→256→1024 误差**逐位不变**；只有 SBO 影响结果 → **B 必须转置成 K-major**，`wgmma` 免转置 V 这条路在 Hopper 上关闭 | 这解释了 21 篇 J3 的失败：不是字段写错，而是**硬件不支持**。MLA 的 PV 转置不可避免；要免转置只能等 Blackwell `tcgen05` 或改走 `mma.sync+ldmatrix.x4.trans`（16 篇 `f4s` 路线，dup=1 但 170 < wgmma 199） |

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

### M. per-block 缩放 / 寄存器墙（24）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| M1 | **per-block 缩放的真正代价是寄存器，不是折算 FLOPs** | e4m3 + ue8m0 + `weight_block=128×128`（DeepSeek-V4） | 每 128-k 块要 `fin += sa*sb*acc`，于是每线程比 per-tensor 多一个跨 k 保留的 fp32 `fin`（64 regs，BM=128/BN=128）。它把 288 线程的寄存器从 90 顶到 154，**occupancy 2 CTA/SM → 1 CTA/SM**，Compute 利用率 67.7% → 49.7% | `24-fp8-gemm-pb/fp8_gemm_pb.cu`（MODE=0 vs MODE=3） | 同 config 128×128 s3：per-tensor 1090.8 → per-block 913.3（−16%）；per-block 上限 936.5（128×128 s4） | 先算寄存器账：`1 CTA=65536/线程数`、`2 CTA=65536/(2×线程数)`。288 线程时 113 就是 2-CTA 门槛，154 必掉到 1 CTA |
| M2 | **1 CTA 的好几何对 per-block 不可行**：累加器+fin 可能等于整个寄存器文件 | 想把 BM 加到 256 提 A 复用 | BM=256/BN=128 有 544 线程，每线程 `acc[64]+fin[64]=128` regs；1-CTA 预算只有 120，`512×128=65536` 恰好等于整个 regfile | `24-fp8-gemm-pb/fp8_gemm_pb.cu`（RUN_PB 256x128） | per-block 256×128：**96 regs + 608B spill → 224 TFLOPS**；per-tensor 同配置 96 regs、2 CTA、1209.6 | 别只看「核函数没报错」——`-Xptxas -v` 的 `spill stores` 是判据；spill 到 local 会把内层循环打崩 |
| M3 | **主循环里读飞行中的 wgmma 累加器 → ptxas 主动串行化 wgmma** | 想用双累加器 ping-pong 把折算与下一块 wgmma 重叠 | `wgmma` 是异步的，但若同一 warp 在「还有 wgmma 未完成」时读累加器寄存器，ptxas 判定数据相关并插入 WGMMA.AR 等待，把整段 wgmma 串行化 | `24-fp8-gemm-pb/fp8_gemm_pb.cu`（MODE=1/2） | 触发 `C7514`/`C7511` 警告；ping-pong 358、pair-drain 397，**比老实 wait0 的 913 慢 2.3×** | 想重叠必须像 DeepGEMM 那样 `warpgroup_fence_operand` + `warpgroup_wait<0>`，或 1 warpgroup/248 regs 的布局；否则编译器会让「重叠」变成「排队」 |
| M4 | **scale 预取 / 强制 2 CTA 都不是 per-block 的瓶颈**（负结果，避免重复踩） | per-block 调优 | scale 全局读是 stride=KBLK 的标量读（sector 利用率低），但它在 L2 命中、且被同 n-block 的多个 CTA 复用；`__launch_bounds__(...,2)` 硬压寄存器只会换 spill | `24-fp8-gemm-pb/fp8_gemm_pb.cu` | scale 预取 smem：913 → 871（**更慢**）；强制 2 CTA：154→96 regs、spill 608B → **219.7 TFLOPS** | 别把「看似不合并的访存」当默认瓶颈；先看 ncu 的 L2/DRAM 占比，再决定要不要动它 |
| L4 | **epilogue 用 `float2` 向量存储** | mma/wgmma 累加器写回 | 累加器相邻两列 `cc, cc+1` 若分两条 4B store，同 warp 相邻 lane 步长 8B，各用半 sector；合并成一次 8B `float2` 后 lane 0-3 拼成连续 32B | `fp8_gemm_tma.cu:231` | ncu 从 **50% global sector excessive** 降到接近 0 | 首地址需 8B 对齐（`cc` 是偶数）；`N` 任意时用 `if (r<M)` 掩码 |
| L5 | **算力受限 GEMM：降 L2 流量（大 BM）> 堆 occupancy** | 大 GEMM 分块选型 | BM 越大，A tile 被同列 tile 复用的次数越多，TMA 次数与 L2 压力越小 | `fp8_gemm_tma.cu`（`BM/BN` 模板） | 128×128 s3（2 CTA/SM，L2 80%）**1097** < 256×128 s4（1 CTA/SM，L2 57%）**1217** | 别默认「occupancy 越高越好」；先看 ncu `L2 Cache Throughput` 与 tensor pipe 活跃度 |

### N. MoE grouped GEMM / 分组调度（25）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| N1 | **MoE grouped 的第一性原理是并行度，不是单 GEMM 的算力** | expert FFN（G 个变长小 GEMM 共享 N/K） | per-expert loop 每个 GEMM 只有 `N/BN × m_g/BM` 个 CTA（G=384、m_g=128 时仅 ~24 个），132 个 SM 分不到一个 CTA，pipeline 刚热就结束 | `25-moe-grouped-gemm/moe_grouped.cu` | loop 耗时对 token 数几乎不敏感（8k/16k/32k = 13.18/13.19/14.08ms）；grouped 单 kernel 后 3.68/5.59/9.58ms（**3.59×/2.36×/1.47×**） | 别先急着优化「单个 expert 的 GEMM」；先看每次 launch 的 CTA 数 vs SM 数 |
| N2 | **把 expert 编码进 B 的行坐标，一个 kernel 吃掉全部 expert** | contiguous / masked grouped GEMM | B 显存里是 `(G,N,K)`，等价于 `(G*N,K)` 的 2D 矩阵；CTA 由 `b_row = group*N + blockIdx.x*BN` 选 expert，`group` 从 `grouped_layout[m_tile*BM]` 读（contiguous）或 `blockIdx.z`（masked）。A/B 都只用 2D TMA，无需 3D | `moe_grouped.cu:200`（`b_row`）、`:209`（masked 早退） | 384 次 launch→1 次、768 张 tensormap→2 张；prefill 与 masked 共用同一 kernel（`MASKED` 模板） | 要求 contiguous 每 expert 的 M 对齐到 BM（一个 tile 只能属一个 expert）；masked 的 `block_row >= gl[group]` 整块早退 |
| N3 | **TMA 的 `boxR` 必须跟着 BN 一起建，否则死锁（不报错）** | 复用同一张 tensor map 跑不同 BN | producer 用 `expect_tx(BM*BK + BN*BK)` 声明等待字节数；若 tensor map 的 box 行数写死得更小，TMA 实际搬的字节少于 `expect_tx`，mbarrier 的 transaction 永远补不齐 | `moe_grouped_launch.h`（`RUN_CONTIG` 里按 BN 建 tmB） | BN=256 配置 100% 死锁 → 改为每个 config 建 `make_tmap_2d(B,K,G*N,128,BN)` 后恢复 | 症状是「kernel 挂死、无 CUDA error」；设备端 `printf` 到 producer 才能定位。2D 行坐标比 3D TMA 更省事，且同等性能 |
| N4 | **decode 的 masked grouped GEMM 是纯权重带宽场景** | MoE 推理 decode（每 expert 几个~几十 token） | `M_useful << 128`，但 8.45GB 权重每个 expert 必须读一遍；早退只省无用行的 A/D，不省 B | `moe_grouped.cu:209`（早退） | 实测 **2.8 TB/s（83.6% HBM）**，含 D 写 ~89% HBM；比 cuBLAS padded batched（按 max_m 全算）快 **1.53×** | 这时堆 occupancy/s4/BN=256 全部更慢（L2 竞争）；判断带宽瓶颈看 `B-read GB/s ÷ bN`，别被 TFLOPS 迷惑 |
| N5 | **分组 GEMM 的下一堵墙是 L2，不是 DRAM：B 被每个 m-tile 重读** | 小 M/expert 的 grouped GEMM | 每个 `(m_tile,n_tile)` CTA 读整块 `BN×K` 的 B，一个 expert 的 B 被其每个 m-tile 重读。总 B 流量 $\approx (M_{total}/BM)\cdot N\cdot K$，理想只需 $G\cdot N\cdot K$ | `moe_grouped.cu`（每 CTA 独立读 B） | ncu（prefill 最佳）L2 **81.9%**、DRAM 52%、tensor 61.3%、occ 13.9%、`long_scoreboard` 47.8%；B L2 流量放大 **~4.5×**（32k tokens） | 解法是 **TMA cluster multicast**（同 expert 相邻 m-tile 组 cluster，B 从 L2 取一次广播）；变长 group 无法两两配对时要像 DeepGEMM 的 `is_peer_cta_alive` 动态关 multicast |
| N6 | **TMA cluster multicast 的正确接口**：leader 单发 + 每 CTA 各自 `expect_tx` + 跨 CTA `mapa` arrive | 同一 cluster 内多 CTA 共享同一 operand（A 沿 N、B 沿 M） | `cudaLaunchKernelEx` + `clusterDim`；leader 发 `cp.async.bulk.tensor.2d...multicast::cluster [dst],[tmap,{c0,c1}],[bar],mask`；每 CTA 对自己的 full barrier `arrive.expect_tx`，multicast 把完成信号补到每个目标 CTA 的同偏移 full；释放端私有 operand 本地 arrive、共享 operand 用 `mapa.shared::cluster` 投到 leader；init 后/退出前各一次 `cluster_sync` | `26-moe-cluster-multicast/moe_cluster.cu:131`（mcast）、`:95`（arrive_cluster）、`:195`（aempty init）、`:216`（expect_tx）、`:260`（lane0→rank0 arrive） | 协议跑通、结果与 CN=1 逐位一致；CN=2 把 A 的 L2 读 sector 砍 **15–19%**；**只有大 batch 兑现成时间**：32768 tokens 960.7→**1001.6（+4.3%）**，`long_scoreboard` 11.6→8.0 | ①`grid.x % CN == 0`否则无法启动；②共享 operand 的 empty count = `CN × 消费者 **warp** 数`（只有每 warp lane0 arrive），写错**不报错只死锁**；③必须把**私有 operand 的 empty 与共享 operand 的 empty 拆成两个 barrier**，否则非 leader 的私有 operand 也被迫等全 cluster，流水被超等拖慢 |
| N7 | **`lts__throughput`（请求/延迟）≠ `lts__t_sectors`（字节）：省 L2 字节不一定提速** | 判断要不要上 cluster multicast | multicast 把 A 的字节广播出去（sector 降 15–19%），但每个 CTA 仍要为自己的 A/B 发 TMA、且 leader↔peer 握手把访存耦合更紧，因此 L2 子系统的**请求率/延迟占用**几乎不动 | `moe_cluster.cu`（ncu 对照见 `ncu_c1_c2_16384/32768.out.txt`） | 16k：sector −15% 但 `lts__throughput` 94.08%→**94.36%**、tensor 62.5%→59.7% → 净变慢；32k：sector −18.8%，终于 +4.3% | 判据：瓶颈在**字节**（大 batch、读放大高）才上 multicast；小 batch/带宽场景（masked decode 读 B 贴 82% HBM）multicast 只有负收益 |
| N8 | **更大的 M-tile 先交 padding 和寄存器税** | 想用 `BM=256` 消 B 放大 `M/(BM·G)` | 理想 $T_{L2}=M_{total}NK(1/BN+1/BM)$，涨 BM/BN 都降流量；但 A-contact 对齐 BM 使 padding 从 10.8%→**19.6%**，且 256×256 的累加器+描述符撑爆寄存器 | `moe_cluster_launch.h`（section D）、`moe_cluster.cu` | 256×128 s4 **788.5 TFLOPS/useful** vs 同分布 128×256 736.6（仅 +7%），仍不如 ALIGN=128 的 128×256；256×256 触发 ptxas `C7511`（wgmma 寄存器不足串行化）**崩到 124** | 换几何前先算寄存器账（`acc[BM/64][BN/128][64]` + 描述符）与对齐 padding；`-Xptxas -v` 看 `C7511` |

### P. MoE gate / tall-skinny GEMM（28）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| P1 | **权重天然 K-major：`wgmma` 的 B 不用转置** | MoE gate / 任何 `X[M,K]·W[N,K]ᵀ` | `wgmma` SS 要求 B 沿 K 连续；模型权重 `W_g[E,H]` 的 H 连续，恰好是 K-major，描述符直接指过去，省掉一次转置/重排 | `28-moe-gate-wgmma/gate_wgmma.cu`（`tma_load_2d` B 用 dims={H*2,E}） | 相对 27 篇用「WgT[H,E] + ldmatrix.trans」的路径，指令与 smem 都更省 | 别沿用 27 篇的 `WgT[H,E]`（那是给 `ldmatrix.trans` 用的）；只有 attention 的 V 才真的要转置 |
| P2 | **bf16 的 SW128 atom = 8 行 × 64 元素（128B）**，故 `BK` 锁 64 | bf16 的 wgmma+TMA | Swizzle<3,4,3> 作用在 **字节** 上：128B/行 ÷ 2B = 64 元素。于是描述符 `SBO=(BK/64)*1024=1024`、`k16` 步进 `(s>>2)*1024+(s&3)*32`（只有 `s∈0..3`） | `gate_wgmma.cu`（`sw128_off`/`make_desc_sw128`） | 与 20/22 篇同构，助手直接复用；TMA 用 `UINT8` 字节语义 + box 内维 128B 写出逐字节一致的布局 | `BK=128` 不可行（TMA SW128 内维上限 128B）；`BN` 也不能超 256（**TMA box 单维上限**，想用 BN=384 会被 `cuTensorMapEncodeTiled` 拒） |
| P3 | **TMA + warp specialization 让 tall-skinny GEMM 再 +68%** | 小 N / 高瘦 GEMM（N=384, K=7168） | `cp.async` 版每轮 256 线程各自算 SW128 地址 + 发 `LDGSTS`，占发射槽与寄存器（122 regs）；换成 1 producer warp 发 TMA + consumers 只做 wgmma 后降到 **90 regs / 0 spill**，装载与计算真正重叠 | `gate_wgmma.cu`（`gate_ws_kernel`） | 16k：337.1 → **565.4（+68%）**；32k：359.6 → **600.3**；达 cuBLAS 77–80% | 相位数组动态下标会掉 local memory（同 23 篇）；consumer 的 `empty` count = 消费者线程数 |
| P4 | **小 N 的 wave 尾效应：先算 wave 数再选 geometry** | 输出 tile 数接近 SM 数的 GEMM | 输出 tile = `(M/BM)(N/BN)`。16k 时 `128×3=384`、2 CTA/SM 下只有 **1.45 wave**，尾块占 ~27% 时间；32k 时 768 块 → 2.91 wave 就顺了 | `gate_wgmma.cu`（扫 BM/STAGES） | 16k 最佳是能塞 2 CTA/SM 的 `128×128 s3`；32k 最佳却是 **1 CTA/SM 的 `256×128 s4`**（BM=256 让 A 复用翻倍、L2 降） | 别以为 occupancy 越高越好：`s4/s5`（1 CTA/SM）在 16k 慢 15%，但在 32k 反而更快——**配置要按 M 选** |
| P5 | **split-K 治不了 tall-skinny 的尾效应（负结果）** | 想靠 K 维并行摊平 wave tail | 拆 K 让 grid 变大、单 CTA 变短，但同一时刻更多 CTA 抢同一批 A 的 L2 行（本例 L2 已 79%），再加 `atomicAdd` 归约与一次 memset 清零 | `gate_wgmma.cu`（`KSPLIT` 模板，`atomicAdd` epilogue） | 16k：`s3` 565.4 → `k2` **429.8**、`k4` 347.8；全线更慢 | 判据：ncu 的 **L2 利用率** 高（>75%）时，split-K 只会加剧 L2 竞争；同理 `L2_PROMOTION_L2_256B` 也变慢（600.3→576.3） |
| P6 | **广播 A vs 广播 B：先数重读次数** | `X[M,K]·W[N,K]ᵀ`（N 小 K 大），想用 cluster multicast 压 L2 | A 被重读 `N/BN` 次、B 被重读 `M/BM` 次。N=384/BN=128 → A 只 3 次；`M/BM` 却上百次。**B 才是重读大头**。用 `cluster.x` 广播 A、`cluster.y` 广播 B（`gate_mcast_kernel<...,CN,AXIS>`，AXIS 选轴） | `29-moe-gate-multicast/gate_mcast.cu` | 32k：A-mcast 把 L2 读 sector 46.2→34.0 M（−26%）却 **−23.4%**（tensor 69→51%）；B-mcast(CN=2) 在 128×128 上 556.3→**587.2（+5.6%）** | cluster 维度**整除**对应 grid 维（`grid.x=E/BN=3` → `cluster.x` 只能取 3，取 2 报 `cluster misconfiguration`）；共享/私有 operand 的 empty barrier 必须分开 |
| P7 | **「放大 BM」和「广播 B」是同一个优化，选前者** | 减少权重 B 的 m-tile 重读 | B 重读次数 = `M/BM`。BM 128→256 直接砍半，效果等同 `cluster.y=2` 广播，但**无跨 CTA 耦合**。所以 BM 已经到顶（TMA box ≤256）时 multicast 才有边际价值 | `29-moe-gate-multicast/gate_mcast.cu` | 32k：`128×128 s3` 556.3 → B-mcast 587.2；但 `256×128 s4` 无 mcast 就 **612.2**（且在 256 上再叠 B-mcast 掉到 562） | 别把 multicast 当「免费的 L2 优化」：它是「用耦合换字节」；`sm__throughput` 已 >65% 时只会更慢（见 26 篇同一结论） |
| P8 | **一个 CTA 吃下整个 N：累加器寄存器墙（负结果）** | N 很小（如 384），想免 A 重读/免 cluster | `gate_wide_kernel`：每 CTA 算 m64×3×n128，A 只读一遍。但 3 个 n128 累加器 = `3×64=192` fp32/线程，寄存器不够让 12 条 `wgmma` 同时在飞 | `29-moe-gate-multicast/gate_mcast.cu`（`gate_wide_kernel`） | `wide128s2/s3` 145–149 TFLOPS、`wide256s2` 41；ptxas 报 **`C7511` wgmma serialized（insufficient registers）**；`wide64` 不触发但也只 456 | 与 24 篇 per-block 的寄存器墙同源：**wgmma 并行度由「在飞累加器」的寄存器预算决定**；Hopper 上无解，等 tcgen05/TMEM |

### Q. fused MoE / expert FFN（30）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| Q1 | **A 按 token 号 gather，省掉 `permuted_x` 物化** | grouped GEMM 的输入来自上一段置换 | permuted 行 `p` 来自 token `row_tok[p]`；装载 A tile 时把 `block_row+r` 换成 `row_tok[block_row+r]`，每个 permuted 行仍是 H 个连续 bf16，`cp.async` 16B 一拍即可，只是源行号随机 | `30-fused-moe/moe_fused.cu`（`up_gate_kernel` 的 `load`） | 省掉 `PX[P,H]`（8192 tokens 时 705 MB）的写+读与整个 permute kernel；K1 侧路径 14.62 → 13.20 ms（**1.11×**） | gather 源行随机，要求 `X` 能在 L2 兜住（X=117 MB，实测 L1/L2 命中良好）；`row_tok` 用 `__ldg`/只读路径读，别先搬进寄存器再散播 |
| Q2 | **一个 CTA 吃下 gate+up，双累加器直接 SwiGLU** | SwiGLU MLP 的 up/gate 两个 GEMM | 让 CTA 的 N=256：前 128 列取专家 gate 权重、后 128 列取 up 权重（两块 B 行坐标差 `I`）。`acc[0]`/`acc[1]` 在寄存器里成对，epilogue 算 `silu(g)*u` 写 bf16 activation | `moe_fused.cu`（`up_gate_kernel<...,FUSE>` 的 epilogue） | 省掉 `G,U[P,I]` fp32（1.2 GB）的写+读与独立 SwiGLU kernel；M=16384 时 K1 447 → 513 TFLOPS | 相邻两列 bf16 必须 `__floats2bfloat162_rn` 打包（20 篇的 `st.shared.u32` 丢高 16 位坑）；两个累加器共 128 fp32/线程，寄存器紧、1 CTA/SM |
| Q3 | **unpermute 融进 epilogue：`red.global.add.f32` + padding 跳过** | MoE 输出加权求和回 token | 输出行 `p` 属 token `row_tok[p]`、权重 `row_w[p]`；`red.global.add.f32 [ptr], v` 是 fire-and-forget 归约（无返回依赖），把 `w*acc` 直接加到 `Yf[t]`；补齐行 `row_w=0` 时整个 scatter 跳过 | `moe_fused.cu`（`down_kernel<...,FUSE>` 的 epilogue、`red_add_f32`） | 省掉 `D[P,H]` fp32（1.4 GB）的写+读与独立 unpermute kernel；**random 路由 32% padding** 时 K2 12.04 → **10.63 ms（−12%）** | `atomicAdd` 返回旧值会产生往返依赖，结果不用时应写 `red`；6 路归约都在 L2 打架，是 DRAM 之外的一层竞争；`red` 与 `atomicAdd` 在结果不用时 ptxas 常生成同一指令，别指望天翻地覆 |
| Q4 | **prefill expert FFN 是「权重带宽」受限，融合收益被 activation 占比锁死** | 判断 MoE 该优化搬运还是权重 | 权重流量 $\approx (P_p/BM)\cdot E\cdot N\cdot K\cdot 2$ vs activation $\approx M\cdot H\cdot 2$；balanced M=8192 时权重 50.7 GB、activation 融合后仅 1.78 GB | `moe_fused.cu`（打印的流量账 + ncu） | 融合把 activation 砍 −81%，但总流量只 −13% → 端到端 1.02×（8k balanced）/ **1.08×（16k）**；K1 ncu **DRAM 84.5%**、roofline 下限 10.1 ms（实测 13.2） | 别把 27 篇「前门搬运 69%」外推到整个 MoE：前门只有 1.1 GB gate 权重，FFN 有 50 GB 专家权重，比例完全反转。先算 `E·N·K / (M·H)` |

### R. grouped + per-block FP8（31）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| R1 | **per-block 的代价是 occupancy，不是 FLOPs——在分组场景下这笔账更便宜** | grouped MoE FFN + e4m3 + `weight_block=128×128` | 每个 128-k 块要 `fin += sa*sb*acc`，每线程多一个跨 k 保留的 fp32 `fin[64]`。分组容器本身已偏 L2/带宽，张量管线的空转占比被摊薄，所以「强制 1 CTA/SM」的相对损失比稠密小 | `31-moe-grouped-pb/moe_pb.cu`（`PERBLOCK` 模板 + 同二进制 `PERBLOCK=false` 对照） | 32k：寄存器 90→**155**、occupancy 27.9%→13.9%、tensor pipe **62.4%→46.1%**；per-block 最佳 **806.4** vs per-tensor 980.5 = **82.2%**（稠密单 GEMM 同一手段是 936.5/1209.6 = 77.4%）；主导 stall 从 `long_scoreboard` 12.61 变成 `barrier` 1.16（每块 `wait0` 排空） | 先算 `65536/(288·2)=113` 的 2-CTA 门槛：90→155 必掉 1 CTA。要做同二进制对照（只切折叠）才能把「寄存器/occupancy」与「分组调度」两件事解耦 |
| R2 | **grouped 的 per-block 缩放：`sb` 退化成标量、`sa` 逐行** | `weight_block=128×128` + `BN=128` | `BN` 恰好对齐 `weight_block` 时，`sb[group, n/128, k/128]` 在一个 n-tile 内是**常数**（整 block 广播）；`sa` 则是「本线程负责的两行」`r0/r1` 各一个，`acc[j*4+0/1]` 用 `sa0`、`acc[j*4+2/3]` 用 `sa1` | `moe_pb.cu`（折叠循环） | 折算指令极少，`sbv` 一次全局标量读（L1 广播）；逐列/逐行随机 scale 才能暴露「相邻元素 scale 不同」的 bug（常数 scale 会假通过） | 行 frag 布局：`acc[j*4+0/1]` 属行 `r0=lane/4`、`acc[j*4+2/3]` 属行 `r0+8`；掩码行（`r>=mlim`）`sa` 取 0，别越界读 |
| R3 | **带宽受限的工作点（decode/masked）里 per-block 折算免费** | MoE decode（每 expert 几个~几十 token） | 整个 kernel 卡在「必须把 8.45 GB 专家权重读一遍」，`wait0`+FMA 被 DRAM 延迟藏住；此时 per-block 与 per-tensor 同样贴 HBM | `moe_pb.cu`（section C masked） | B-read **2862 GB/s（85.4% HBM）**，per-tensor 25 篇 2.8 TB/s 持平；`BM=64` 比 `BM=128` 的 75.4% 更好（早退粒度更细） | 「per-block 贵不贵」不能脱离工作点回答：prefill 算力受限（暴露 18%），decode 带宽受限（免费） |
| R4 | **`BM=64` 省寄存器买不回 B 重读（negative）** | 想让 grouped per-block 挤进 2 CTA/SM | `BM=64/BN=128` 时 `acc[64]+fin[64]=128` regs、160 线程可容 2 CTA（≤204）；但 B 被重读 `M_total/BM` 次翻倍，L2 压力暴涨 | `moe_pb.cu`（`pb64x128s3` 等） | 全线更慢：8k 587 / 16k 656 / 32k 648（vs BM=128 的 708/790/806）；显式 `minBlocks=2` 也只是 6xx | 换 `BM` 前先算 B 的 L2 重读次数 `M_total/BM`（25 篇 N5）；分组 GEMM 里 B 重读通常比 occupancy 更值钱 |
| R5 | **TMA 的 `boxR` 必须匹配 `BM`（不只是 `BN`）** | 一张 tensor map 跑多个 `BM` | producer 用 `expect_tx(BM*BK + BN*BK)` 声明字节数，`tmA` 的 box 行数若写死 128 而 kernel 跑 `BM=64`，TMA 实际搬的字节与期望不符 → **mbarrier 永远等不齐，死锁且不报 CUDA error** | `moe_pb_launch.h`（每个 config 按 `BM` 重 build `tmA`） | BM=64 第一版 100% 挂死（ncu/`nvidia-smi` 看到残留进程）；改成 `make_tmap_2d(A,K,R,128,BM)` 后正常 | 这是 25 篇 N3（`boxR` 随 `BN`）的推广：A 的 box 行数随 `BM`、B 的 box 行数随 `BN`，写多 config 扫描时描述符必须按 config 重建；device `printf` 打到 producer 才能定位 |

### S. 归一化 / 逐元素访存（32）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| S1 | **行能放进寄存器就别放 smem** | RMSNorm / LayerNorm / softmax 等「读整行→归约→再写整行」 | 每线程把自己的 `NG=ceil((H/8)/T)` 个 `uint4` 留在寄存器里，归约出 `inv` 后**就地**归一化写回；smem 版本每元素要写+读一次，把 L1/TEX 顶成瓶颈 | `32-fused-norm/norm_fused.cu`（`rmsnorm_v1r_kernel`） | H=7168：smem 版 2334 GB/s（L1 76%、DRAM 44–70%）→ **寄存器版 2869 GB/s / 85.6%**（DRAM 82.9%） | 寄存器只多 ~16 个（总共 48）；只有「整行能一次装进寄存器」时可行；`H` 大或线程少时 `NG` 会涨 |
| S2 | **纯访存算子的优化顺序：先减 pass，再消 L1，最后才谈指令** | 所有 memory-bound 算子 | 先用融合把中间张量的读写 pass 减到最少（add+norm：5→4），再看是不是 L1/smem 往返成了更早的墙，最后才抠指令 | `norm_fused.cu` | 朴素 1481（44%）→ smem 单读 2334（70%）→ 寄存器 2869（86%）→ 融合残差 **2923（87%）** | 判断「到底哪级是墙」看 ncu `DRAM vs L1TEX vs L2`；DRAM% 才是 HBM 利用率 |
| S3 | **融合的收益只看「省掉几个 pass」——FP8 量化融合常常不划算** | 把 norm/激活与动态量化融进同一 kernel | add+norm 省 1 个 pass（8B→8B 但 pass 5→4）；norm+fp8 只省 0.5 个（8B→7B），却新增每 128 元素的 `amax` 归约（shared `atomicMax`）与逐字节 fp8 打包 | `norm_fused.cu`（`add_rmsnorm_fp8_v3r_kernel`） | 融合 add+norm 2923 GB/s；再叠 fp8 **掉到 2205 GB/s（65.8%，DRAM 只有 64.8%、SM 50.2%）**，因为瓶颈换成 amax 原子竞争 | fp8 输出占比小（1/4）时省不出钱；输出占比大（如已经只写 fp8）才值。`atomicMax` 对**非负 float** 可用 `atomicMax((int*)&a,__float_as_int(v))` |
| S4 | **`float(fp8)` 在这个 CUDA 工具链下返回位模式（不是数值）** | 任何 e4m3/e5m2 精度验证 | `__nv_fp8_e4m3` 的 `operator float()` 实测返回原始 1 字节（`1.0f` → `56`），导致误差恒定停在 155% 查不出逻辑错 | `32-fused-norm/fp8_test.cu`、`fp8_test2.cu` | 显式走 `__nv_cvt_float_to_fp8` + `__nv_cvt_fp8_to_halfraw` 后误差降到 **3.31%**（正常量化误差） | 也不要写 `fp8 out = fp8(storage_byte)` 再存（重载会错）；直接把 `__nv_fp8_storage_t`(`uint8`) 写进输出字节流 |

### O. MoE routing / token 置换（27）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| O1 | **NoAux-TC routing 的 bias 只用于「选」，不用于「权重」** | DeepSeek-V4 / Kimi-K2.6 的 router | `s=sqrt(softplus(logits))`（V3/Kimi 是 sigmoid）；`choice=s+bias`；`ids=topk(choice,K)`；权重取 **未加 bias** 的 `s[ids]` 再归一化 ×`routed_scaling_factor` | `27-moe-router/moe_router.cu:219`（`router_topk_kernel`） | CPU 对拍：top-6 集合 8/8、权重 err 5.96e-8 | 用 `choice[ids]` 当权重会全错；`weight_block=128×128` 与 router 无关别混 |
| O2 | **384 选 6 的 top-k：1 warp/token + 6 轮 argmax** | 专家数几百、top-k 个位的 MoE router | 每 lane 扫 12 个存 smem，6 轮各做一次 warp argmax（5 步 `shfl_down`），lane0 把选中项置 `-inf` 防重复；比 radix-select（第 18 篇，为 S=32k 长序列设计）更省事 | `moe_router.cu:219` | 0.031 ms@16384；1 warp/block→8 warp/block 快 13–25%（occupancy 50%→84.7%） | top-k 是 compute-bound（`expf`+shuffle），不是访存；block 太小会被「每 SM 32 block」锁死占用率 |
| O3 | **把 expert 直方图融进 top-k 的 lane0** | router→permutation 之间的元数据 | top-k 选中 `sel[j]` 时顺手 `atomicAdd(&cnt[sel[j]],1)`，省掉独立 count kernel 的一次启动 + 一遍 `ids` 读取 | `moe_router.cu:270`（`COUNT` 模板）、`:571`（`run_route`） | 含 topk 的整条 `route` 从 0.0879→0.0710 ms@16384（**−19%**）、0.175→0.127 ms@32768（**−27%**） | 直方图要在 top-k 前 `cudaMemset` 清零；`cnt` 是全局 atomic，384 个地址竞争可接受 |
| O4 | **置换（permute）的胜负手：让 `x[t]` 只读一次** | MoE token permute / 任何「一变多」的 scatter | v1「一个 block 一个 permuted 行」会把 `x[t]` 按 slot 重读 **K 次**（流量 `(K+1)MH·2`）；改成「一个 block 一个 token」，读一次存寄存器、写 K 个目标行，流量降到 `(1+K)MH·2` | `moe_router.cu:333`（`permute_copy_tok_kernel`）vs `:320`（`permute_copy_pos_kernel`） | @16384：0.924→**0.605 ms（1.53×）**，1.64 GB vs 2.82 GB | 写侧变 K 条散流，HBM 效率从 91% 掉到 81%——但流量省得更多，**净时间赢**；`st.global.cs` 只快 0.6%（噪声） |
| O5 | **反向置换是纯带宽题，目标 90%+ HBM** | MoE unpermute（expert 输出按 token 加权求和） | 一个 block 一个 token：`y[t]=Σ_j w[t,j]*permuted_y[pos(t,j)]`，bf16 直接 FM 进 fp32，读 `K·M·H·2`、写 `M·H·2` | `moe_router.cu:369`（`unpermute_kernel`） | **3075 GB/s（91.7% HBM）**@16384；ncu `DRAM 91.9%`、Compute 13.4% | 不变量可自检：`y[t] == x[t]·Σ_j w[t,j]`，置换写错立刻暴露；别用 `index_add`+fp32 广播（torch 只有 173 GB/s） |
| O6 | **端到端先量「谁占大头」，再决定优化谁** | MoE 前门整链 | gate/route/permute/unpermute 逐段计时后：gate 27%、route 4%、**permute 37% + unpermute 32% = 69%** | `moe_router.cu`（`main` 汇总） | 搬运是大头，且已贴 HBM 天花板 | 唯一的杠杆是**减少字节**（把 permute 融进 grouped GEMM），不是把 90% 的 kernel 再抠 5% |

### H. 优化器 / MuonClip（17）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| H1 | 优化器算子先算 roofline | Muon/NS、Shampoo 等 | NS 正交化 = 5 步 × 3 GEMM = `30N³` FLOPs；`AI=O(N)`，工作集仅 `2N²` | `17-muonclip-ns/ns_muonclip.cu` | ncu `DRAM 13.4%`、`Compute 45%` → **算力受限**；优化目标是把 GEMM 做快 | 别一看到「优化器」就以为访存受限；先看 ncu 的 DRAM% |
| H2 | 融合 epilogue `f·x+g(GEMM)` | 链式 GEMM（NS/优化器） | mma 累加器 `c0,c1`(行 `lane/4`)/`c2,c3`(行+8) 坐标固定，结果还在寄存器时按同坐标读 `Cin` 做 `sc_c·Cin+sc_d·acc` | `17-muonclip-ns/ns_muonclip.cu`（`store_acc_epi`） | 消掉每步 2 趟 `axpby`（读 2 写 1）；N=4096 端到端 8.93→**8.44 ms（−5.5%）**；epilogue 不涨寄存器（126/0 spill） | `Cin` 与输出必须不同 buffer（`M=bA+cAA` 中 `Cin=A` 是另一 buffer，非原地别名）；尺寸越大收益越小 |
| H3 | `mma` 路径的天花板是 L2 放大 + 寄存器墙 | 大 N 方阵 GEMM | `128×128` 分块下 A/B 各被重复读 32 次（每 GEMM ~2GB L2 流量）；块放大则每线程累加器翻倍 → 126 寄存器、occupancy 23.8% | `17-muonclip-ns/ns_muonclip.cu` | 自研单 GEMM 269 TFLOPS（cuBLAS 883），卡在 `L2 76% / occ 24%`；`128×256`/`256×128` 更慢 | 出路是 `wgmma`(SS 直读 smem，免 `ldmatrix`) + TMA 压 L2；见路线图 31/38 |
| H4 | 多级 `cp.async` 的 `wait_prior` 参数 | 通用软流水 | N 级流水 prologue 预取 N−1 级、循环每轮再 commit 1 级；等「当前 stage」应 `wait_prior(N-1)` 而非 `N-2` | `17-muonclip-ns/ns_muonclip.cu` | 写成 `N-2` 会过早放行、读到半写 smem（`max_abs_err` 爆到 1e6） | 动态 smem 手搓多级缓冲时 B 的 stage 基址要放在 A 的**全部** stage 之后（`smem + STAGES×BM×LDP`），漏乘 `STAGES` 会 A/B 槽覆盖 |

### V. compressor 投影 · 压 L2 与 FP8 化（36）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| V1 | **「L2 墙」先确认你量的是不是最佳配置** | tall-skinny GEMM 的瓶颈判断 | B 重读次数 = `M/BM`。`BM=128` 时 B 被重读 256×，L2 子系统被顶到 84%；`BM=256` 砍半到 128×，L2 掉到 58%、tensor 升到 76%——瓶颈根本不是 L2 | `36-dsa-compressor-mcast/compressor_mcast.cu`、ncu `ncu_bf16_best.out.txt` | 同 shape `128×128s2` L2 84%/Compute 58%；`256×128s4` L2 58%/tensor **76.4%**，TFLOPS 540→692 | 35 篇的「L2 墙」是在 `128×128s2` 上采的。**换几何前不要下瓶颈结论**；TMA box ≤256，`BM=256` 就是上限 |
| V2 | **multicast 判据（五条）** | 判断该不该上 TMA cluster multicast | ① ncu `lts__throughput` >75%；② 瓶颈在字节（`lts__t_sectors` 读放大高）；③ `BM/BN` 已到 TMA 上限；④ `sm__pipe_tensor_cycles_active` <60%；⑤ 从 CN=2 起试 | `compressor_mcast.cu`（`AXIS=1/2,CN`）、26/29 篇 | 本 kernel tensor 76% → A/B multicast 全负（−3.7%~−19%）；26 篇 masked（纯 DRAM）也负；只有 26 篇 32k+128×128（L2 81.9%）赢 +4.3% | 「放大 BM」和「广播 B」是同一个优化且无耦合——先放大 tile，放不动了再考虑 cluster |
| V3 | **L2 hit rate 已经很高时，persisting window 没用** | 想把反复读的权重钉进 L2 | `cudaAccessPolicyWindow(hitProp=Persisting)` 只提升命中率；若 `L2 Hit Rate` 已 >80%，瓶颈在 L2→SM 的请求率/延迟，抬命中率无意义 | `compressor_mcast.cu`（`set_persist`） | `Wm` 仅 14.7MB（<50MB L2）却 **−2.2%**；ncu `L2 Hit Rate` 82% | 判据：先看 ncu `L2 Hit Rate` 与 `lts__throughput`，命中率低才谈 persisting |
| V4 | **TMA kernel 消费者侧的 `fence.proxy.async` 可删** | 所有 TMA + wgmma 的 warp-specialized kernel | TMA 写 smem 与 wgmma 读 smem **都在 async proxy**，mbarrier 的 `complete_tx` 已完成两者排序；消费者再插 generic↔async 的 fence 是多余的 | `compressor_mcast.cu`（`CFENCE` 开关） | paired 测量中性（652 vs 650，噪声内）；原理上安全 | 只有当**消费者自己用 generic 指令写过 smem** 时才需要；纯 TMA→wgmma 不需要 |
| V5 | **tensor 受限的 kernel，换数制比调流水更有效** | bf16 tensor pipe 已 ~76%、几何受 TMA box 限制 | FP8 每个周期做 2× MACs；bf16 76% 封顶时，FP8 只需 ~51% 就能把时间砍到 0.73× | `compressor_fp8.cu`（`proj_fp8_kernel`，复用 24 的 `fp8_tma_pb_kernel`） | 投影 `692.5 → 959.1 TFLOPS（1.385×，ratio128）`、`698.8 → 974.1（1.394×，ratio4）` | 换 FP8 必须按真实量化方案（V4-Pro：e4m3+ue8m0+128×128）；per-block 折算的寄存器墙依旧（见 M1/R1），FP8 只到峰值 49% |
| V6 | **FP8 per-block 让瓶颈从「等操作数」翻成「等 barrier」** | 诊断 FP8 per-block GEMM | `fin`（+64 regs）把寄存器 90→154，越过 288 线程 2-CTA 门槛，occ 2→1 CTA/SM；每 128-k 块 `wgmma.wait0` 折算时张量管线排空 | `compressor_fp8.cu`、ncu `ncu_fp8_best.out.txt` | bf16：tensor 76%、L2 58%、stall `long_scoreboard` 9.56；FP8：tensor **51%**、L2 **73%**、occ 13.8%、stall **`barrier` 1.18** | 这是 24/31 篇同一堵墙；想再上只能 DeepGEMM 的 1-warpgroup/`setmaxnreg 248` 布局（backlog） |

### T. fused MoE + FP8 / 权重带宽（34）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| T1 | **MoE prefill FFN 的最大杠杆是权重字节，不是 activation** | 判断 MoE 优化方向 | 专家权重流量 $\approx \sum_g \lceil m_g/BM\rceil \cdot N\cdot K$ 字节，与 M 弱相关；activation 只 $\propto M$。30 篇 bf16 时权重 50.7 GB、activation 1.8 GB，融合上限 13%。换 FP8 后权重直接减半 | `34-fused-moe-fp8/moe_fp8.cu` | 端到端相对 bf16（30 篇）**1.92×**（M8k bal）/ **2.21×**（M16k bal）/ **2.06×**（M8k rand）；权重 50.7→25.4 GB | 先算 `E·N·K` vs `M·H` 再决定「省钱还是省搬运」；M 越大权重重读越多，FP8 收益越大 |
| T2 | **unpermute 融合的收益会随 GEMM 变快而反转** | 把 scatter 归约融进 GEMM epilogue | bf16 时 down GEMM 8.67 ms，散写被藏住；FP8 把 GEMM 压到 3.35 ms 后，`red.global.add.f32` 的 6 路散写把瓶颈从 DRAM 顶成 L2 请求 | `34-fused-moe-fp8/moe_fp8.cu`（`UNPERM` 模板） | 30 篇 bf16 bn128：融合 **+10%**；本篇 FP8：K2f 3.83 vs 3.35+0.54，端到端 **0.96×**；ncu DRAM **73.2%**、L2 **94.4%** | 融合收益不是常数——它取决于被测算子有多快。改精度/工作点后**必须重测**融合，别照搬旧结论 |
| T3 | **per-block FP8 与「双累加器 SwiGLU」寄存器互斥** | 想把 30 篇的 gate+up 融合叠进 FP8 | `acc`(2×64) + per-block `fin`(2×64) = **256 > 255**（每线程寄存器硬上限），任何调度都救不回来；`BN=256` 让 `NSPLIT=2` 的 `acc`+`fin` 同样翻倍 | `moe_fp8.cu`（负结果） | 双累加器只存在于「per-tensor」；per-block 只能单累加器（31 篇 64+64=128 刚好）。K1 `BN=256` 实测 7.1→**21.7 ms** | 先算 `acc+fin` 的寄存器账：per-block 的 `fin[64]` 是固定开销，凡是想叠加第二个累加器/更大 BN 的融合都要先过这一关 |
| T4 | **SwiGLU + 动态量化：一行一 block、`float4` 读、warp 内 amax** | epilogue 把 fp32 激活转 fp8 per-128 | per-128 的 `amax` 天然是「一行」内的分段归约：让一个 block 处理一行、8 个 warp 轮流处理 24 个 128-列块，lane 用 `float4` 读 G/U 各 4 列，`__shfl_xor` 5 步归约得 amax，再打包 4 个 fp8 成 `uint32` 写 | `moe_fp8.cu`（`swiglu_quant_v2_kernel`） | 1.07→**0.44 ms**，**3090 GB/s / 92% HBM（2.4×）**；v1 是「128 线程各管 1 列 + block 级 smem 归约」，只有 39% | 别用 block 级 `__syncthreads` 归约 128 个元素的 amax（两次同步 + 标量读）；`float4` + warp shuffle 才打得满带宽 |

### U. DSA compressor / 门控池化（35）

| # | 技巧 | 适用场景 | 原理 | 代码 | 实测收益 / 现象 | 坑 |
|---|---|---|---|---|---|---|
| U1 | **把多个共享 A 的投影合并成一个 GEMM** | DSA compressor 的 `wkv`/`wgate`（输入同一 `x`） | 参考实现里两次 `linear` 各读一遍 A；拼成 `Wm=[wkv;wgate]`（`[2C,D]`）后 A 只 TMA 搬一次，N 翻倍 | `35-dsa-compressor/compressor.cu`（`proj_ws_kernel`） | cuBLAS 两次 bf16 GEMM 0.630→合并 **0.595 ms**（ratio128，−5.5%）；ratio4 1.398→**1.185**（−15%） | 输出 `Y[:, :C]=kv`、`Y[:, C:]=score` 的列偏移要写死对齐；合并后 N 变大，别忘 TMA `boxR` 随 `BN` 重建 |
| U2 | **门控池化可以「一个 block 一个窗、一线程一列」地并行** | Compressor 的 `softmax_i(score)·kv`（每个输出列独立） | softmax 在窗口内 token 维，**列与列互不耦合**；让 block=窗、线程 j=输出列 j，用 online softmax 单趟做完 max/sum/加权，读完全合并 | `compressor.cu`（`pool_kernel`） | 纯访存，**2790–2812 GB/s / 83–84% HBM**（ncu DRAM 87.2%） | `ratio=4` 的 overlap 语义是 8 个 slot：前 `r` 个来自上一窗的前半 dim、后 `r` 个来自本窗后半 dim，首窗前半置 `-inf`；别把 `-inf` 直接喂 `exp`（会 NaN），跳过即可 |
| U3 | **池化 + RMSNorm + RoPE 融成一个 kernel** | Compressor 输出后处理 | 池化的输出列先经 block 归约做 RMSNorm，再经 smem 交换成对元素做末 64 维 RoPE；省掉中间 `[S/r, r, 2C]` 与 5 次 launch | `compressor.cu`（`pool_kernel`） | eager 的池化+Norm+RoPE ≈0.315 ms（M=32768）→ **0.048 ms（~6.6×）** | RMSNorm 的 block 归约别用 `__shfl_xor_sync(0xffffffff,...)` 在只有 16 个活跃线程时跑（mask 含未参与 lane，UB）；让整 warp 参与、空 lane 补 0 |
| U4 | **小 N 大 K 的投影是 L2 带宽受限，不是算力受限** | `[2C,D]` 权重只有几十 MB、M 很大 | B 被 `M/BM` 个 m-tile 重读、A 被 `2C/BN` 个 n-tile 重读；`M=32768,BM=256,BN=128` 时 L2 读 ≈ 7.5 GB | `compressor.cu`（ncu） | proj ncu：`L2` **85–88%**、`Compute` 57–59%、`DRAM` 26–29%、occ 28%、No Eligible 86% | 别看到「tensor 没满」就加 occupancy——`256×128 s4`（1 CTA/SM）赢过 `128×128 s2`（2 CTA/SM）13–20%；`BM=256` 把 B 重读砍半才是对的 |
| U5 | **输出中间张量降精度（fp32→bf16）不划算（负结果）** | 想省 GEMM 写 + 池化读 | Y 从 134 MB 减到 67 MB，但 wgmma 累加器 `f32→bf16` 转换 + 更窄 store 把收益吃回去；且 GEMM 本就被 L2/算力卡住 | `compressor.cu`（`proj_ws_kernel<...,BF16OUT>`） | 池化读确实 0.048→0.038 ms，但投影 0.742→0.749 ms，**e2e 打平略负**（0.790 vs 0.787） | 和 32 篇「FP8 量化融合不划算」同源：**收益只在被优化的那一级真是瓶颈时才兑现** |


---

## 三、模型场景台账（15 起填充）

| # | 模型 | 算子 | 关键 shape/参数 | 手段与结论 | 代码 | 实测 |
|---|---|---|---|---|---|---|
| M1 | DeepSeek-V3/V4、Kimi-K2.6 | MLA | V3: H=128,DC=512,DR=64,DV=512（absorb 576/512）；V4-Pro: H=128,DC=448,rope64；V4.1: H=64；Kimi: H=64,qk=192,v=128,kv_lora=512 | 吸收成 MQA。15：TC 三 kernel 115.6（S=1024）。**16：单 kernel 融合**（online softmax 进 QK epilogue、C→A 零 shuffle、KV 常驻 smem）；用 smem 共享 P 消 QK 重复 → `f4s` **170.1**（S=1024）/ **184.7**（Sk=4096）；DRAM 仅 7%，瓶颈=L1 的 ldmatrix + 12.5% occ；距 FlashMLA 640 ~3.5×。**20：wgmma SS + SW128 swizzle** → 105.3/112.9/119.9（Sk=1k/2k/4k），仍 1.62× 慢于 `f4s`；ncu：L1 70.8%、occ 12.5%、221KB smem。**21：修 V 转置访存**（迭代顺序 dv 最快，L1 73%→46%，+35%）+ **`cp.async` 单缓冲预取下一块 K**（`long_scoreboard` 4.81→1.42）→ **159.8/193.2/198.8**（Sk=1k/4k/8k），**Sk≥4096 反超 `f4s`**，距 FlashMLA ~640 收窄到 **~3.3×**；ncu：tensor 30%、L1 46%、barrier 1.17/wait 1.01 成下一道墙 | `15-mla-attn/mla_attn.cu`、`16-mla-fused/mla_fused.cu`、`20-mla-wgmma-sw128/mla_wgmma_sw.cu`、`21-mla-wgmma-pipe/mla_pipe.cu`/`mla_pipe3.cu` | 15 tc: 2.53/9.41/36.58 ms；16 f4s: 1.72/3.18/6.33 ms；20 wgmma_sw: 2.77/5.17/9.74 ms；21 pipe3: 1.83/3.28/6.06/11.75 ms（Sk=1k/2k/4k/8k） |
| M2 | DeepSeek-V4-Pro / V4.1 | DSA 稀疏注意力（一）：indexer + top-k | index_n_heads=64(V4-Pro)/32(V4.1), index_head_dim=128, index_topk=1024(V4-Pro)/512(V4.1) | 18：indexer 是「H^I 个小 GEMM 共享 K」，TC+head 合并 HG=2 达 **287–311 TFLOPS（31% 峰值）**；exact top-k 用 radix-select + per-warp 直方图 **6.75ms @S=32768**。DSA 端到端（indexer+topk+稀疏 MLA 估算）相对稠密 MLA：4k 2.9× / 16k 9.7× / 32k 14.8× / **64k 18.6×**，上限 ~30×。ncu：indexer DRAM 2.9%、L1 62%；topk IPC 3.35、issue 83% | `18-dsa-sparse/dsa.cu` | indexer S=16384 HG2/BN128 15.32ms；topk S=32768 6.75ms；dense MLA f4s S=65536 6914.8ms |
| M2b | DeepSeek-V4-Pro / V4.1 | DSA 稀疏注意力（二）：稀疏 MLA 消费端 | H=128(V4-Pro)/64(V4.1), DC=512,DR=64,DV=512, topk=1024/512 | 19：CTA = 1 token × $B_H$ head，gather top-k 的 $c_{kv}$+$k_{rope}$、尾部 tile 掩码、共享 P、`cp.async` 双缓冲。实测 132–137 TFLOPS（~75% 稠密 f4s 效率）；Sk=64k 时 sparse attention 比稠密快 **48.6×**；`long_scoreboard` 4.56→1.46。DSA 端到端（含 18 的 indexer/topk）4k 2.5× / 32k 13.3× / 64k **17.3×**；距 FlashMLA sparse prefill 640 约 4.8× | `19-dsa-sparse-attn/sparse_mla.cu` | p2: 4096/8192/16384/32768/65536 → 136.3/129.9/133.3/132.4/127.9 TFLOPS；dense f4s 6.36→110.87 ms |
| M3 | DeepSeek-V4-Pro | MoE grouped GEMM（expert FFN） | 384 routed+1 shared, top-6, K=hidden=7168, N=moe_inter=3072；prefill 平均 M/expert=128/256/512，decode max_m=128 | 25：①per-expert loop = 384 次 launch，每 expert 仅 ~24 CTA → SM 空转，耗时 13–14ms 对 token 数不敏感；②**把 expert 编码进 B 的行坐标**（`b_row=group*N+n`，A/B 都用 2D TMA），单 kernel 调度全部 tile → prefill **3.59×/2.36×/1.47×**（tokens=8k/16k/32k），aligned 958–1008 TFLOPS；③反超 cuBLAS per-expert graph loop **1.17×**；④masked decode 是权重带宽场景，读 8.45GB 权重达 **2.8 TB/s（83.6% HBM）**，早退比 cuBLAS padded batched 快 **1.53×**；⑤contiguous 对齐到 BM=128 的 padding 浪费：小 batch 32.5% / 16k 19.8% / 32k 10.8%；⑥ncu（最佳大 batch）：L2 **81.9%** 最高、DRAM 52%、tensor 61.3%、occ 13.9%（1 CTA/SM）、`long_scoreboard` 47.8% → B 被每个 m-tile 重读，L2 流量放大 ~4.5×，下一步 **TMA cluster multicast** | `25-moe-grouped-gemm/moe_grouped.cu` | prefill 8k/16k/32k：loop 13.18/13.19/14.08ms；grouped 3.68/5.59/9.58ms；masked 3.02ms；cuBLAS graph loop 6.54ms |
| M3b | DeepSeek-V4-Pro | MoE grouped GEMM + TMA cluster multicast | 同上（384 experts top-6, K=7168, N=3072）；cluster 沿 N 广播 A | 26：`cp.async.bulk.tensor.2d...multicast::cluster` + `mapa` 跨 CTA arrive；A 只由 m-tile 决定 → 沿 N 的 cluster 一定共享 A、无 expert 边界问题。①ncu：CN=2 把 A 的 L2 读 sector 砍 15–19%、`long_scoreboard` 11.6→8.0；②**只有 32768 tokens + 128×128 赢**：960.7→**1001.6 TFLOPS（+4.3%）**；8192 −7.4%、16384 −2.2%、masked decode −3.9%（纯 DRAM 带宽场景无用）；③CN=4 全线更慢（耦合）；④BM=256 负结果（padding/寄存器，256×256 `C7511` 崩到 124）；⑤大 batch cuBLAS graph 追平（1028.8 vs 1001.6），中小 batch grouped 仍 1.16× | `26-moe-cluster-multicast/moe_cluster.cu` | 32k：c1 960.7 / c2 **1001.6** / c4 811.2；8k：867.8/804.0；16k：947.7/926.8；masked 174.2/167.5；cuBLAS graph 32k 1028.8 |
| M3c | DeepSeek-V4-Pro（Kimi-K2.6 同构） | MoE gate GEMM（router 前门） | hidden=7168, n_routed=384；`logits[M,384]=X[M,7168]·Wg[384,7168]ᵀ`，M=16k/32k | 28：27 篇的 `mma.sync`（201.9/262.3 TFLOPS，cuBLAS 的 27.7%）换成 ① `wgmma.m64n128k16` SS + SW128（B 天然 K-major 免转置）→ 337/360；② 再叠 TMA + mbarrier + warp specialization（1 producer + 2/4 consumer WG，90 regs/0 spill）→ **565.4@16k（128×128 s3）/ 600.3@32k（256×128 s4）**，达 cuBLAS **77.0%/79.8%**、相对 27 **2.80×/2.29×**。ncu（32k 最佳）：Compute 69.6%、tensor pipe **68.2%**、L2 **79.1%**、DRAM 53.4%、occ 26%、0 bank conflict → 下一堵墙是 **A 被 3 个 n-tile 重读**。负结果：split-K（429）、BM=64（420–450）、L2_256B promotion（576）、BN=384（TMA box 上限 256） | `28-moe-gate-wgmma/gate_wgmma.cu`、`cublas_gate_ref.py` | gate 16k：0.4468→0.1595 ms；32k：0.6876→0.3005 ms；cuBLAS 0.1228/0.2397 ms；ncu 16k/32k |
| M3d | DeepSeek-V4-Pro（Kimi-K2.6 同构） | MoE router（gate + top-k + 置换） | hidden=7168, n_routed=384, topk=6, `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`, `norm_topk_prob=true`, `routed_scaling_factor=2.5`（Kimi：sigmoid/topk=8） | 27：手写七 kernel 复现前门。①gate GEMM（`mma+ldmatrix+cp.async`，BM=128）**204→257 TFLOPS**，仅 cuBLAS 738 的 **27.7%**（ncu L2 60.7%/Compute 35.3%/No Eligible 57.6%）；BM=256 想省 L2 但 1 CTA/SM 反而 −10%。②top-k `sqrtsoftplus+bias` 选专家、用未加 bias 的 `s` 归一化 ×2.5；1 warp/token、8 warp/block，CPU 对拍 8/8、err 6e-8。③直方图融进 top-k 省 19–27%。④permute「1 block/token 读一次写 K 行」比「1 block/行」**1.53×**（81% HBM）。⑤unpermute 融合加权 **91.7% HBM**。⑥端到端搬运占 **69%**；对标 torch：perm 1.50×、unperm **17.8×**、端到端 **6.6×** | `27-moe-router/moe_router.cu`、`moe_router_ref.py` | M=16384：gate 0.439 + route 0.071 + permute 0.605 + unpermute 0.535 = **1.651 ms**；torch 10.95ms；M 扫 4096/8192/16384/32768 全 OK |
| M3e | DeepSeek-V4-Pro（Kimi-K2.6 同构） | MoE gate GEMM + TMA cluster multicast | hidden=7168, n_routed=384；`logits[M,384]=X[M,7168]·Wg[384,7168]ᵀ`，M=16k/32k；`grid=(3, M/BM)` | 29：在 28 的 `gate_ws_kernel` 上加 `AXIS` 选轴（`cluster.x` 广播 A / `cluster.y` 广播 B），协议同 26。①**A-multicast（CN=3）**：L2 读 sector 46.2→**34.0 M（−26%）**、L2 吞吐 80.4%→48.5%，但 duration 290→371 µs（**−23.4%**），tensor/SM 68.8%→51.5% —— 瓶颈是 tensor 不是 L2。②**B-multicast（CN=2）**：128×128 s3 从 556.3→**587.2（+5.6%）**，L2 读 sector 71.2→64.1 M；但 **256×128 s4 无 mcast 就 612.2**，比它更高——放大 BM 与广播 B 是同一优化，前者无耦合。③**wide-N（一个 CTA 算完 N=384）失败**：3 个累加器 192 regs → ptxas `C7511` 串行化 wgmma（128/256 崩到 145/41）。ncu（最佳 256×128 s4）：Compute 69.5%、L2 80.4%、DRAM 54%、occ 26.1%、`long_scoreboard` 10.85（等 wgmma 操作数），STAGES 已到 smem 上限。**结论：multicast 只在瓶颈是 L2 字节、且 BM 无法再放大时才值得，且从小 cluster（CN=2）试起** | `29-moe-gate-multicast/gate_mcast.cu` | 32k：baseline **612.2** TFLOPS（0.2947 ms, cuBLAS 81.3%）、ax3 468.7、by2 587.2；16k：baseline 547.0（cuBLAS 74.5%）；cuBLAS 752.6/734.6 |
| M4 | DeepSeek-V4-Pro / V4.1 | FP8 GEMM | hidden=7168, moe_inter=3072；e4m3 + ue8m0, weight_block 128×128（V4.1 为 32×32 + expert fp4）, activation dynamic 1×128 | 22：真实 shape 4096×3072×7168。①`mma.m16n8k32`+ldmatrix(b16)+`cp.async` 流水 → **266 TFLOPS（13.5%）**，ncu tensor pipe 41%、`math_pipe_throttle` 1.12（发射受限）；②换 `wgmma.m64n128k32`+SW128（与 bf16 同构）+`wait_group` 流水 → **768 TFLOPS（38.8%）**，+2.89×。**23：TMA（`cp.async.bulk.tensor.2d`+SW128）+ mbarrier + warp specialization** → 同 shape **1217.3 TFLOPS（61.5%）**，达 cuBLAS FP8（1381.3）的 **88.1%**（差距 1.13×），张量管线活跃度 72%；③per-block（1×128 激活 + 128×128 权重）22 的 cp.async 版 519 → **23 的 TMA 版 906（45.8%，+74%）**，瓶颈转为「每 128-k 块折算等 mma」的流水断裂（寄存器 150、occ 13.8%）。关键坑：相位数组动态下标掉 local memory（988→1217）、epilogue 标量 store 半 sector（float2 修复）。**24：per-block 诊断**——同二进制里关掉折算的 per-tensor 对照（128×128 s3 **1090.8**、256×128 s4 **1209.6**），per-block 最佳 **936.5（128×128 s4，47.3%）**；根因是 `fin` 多 64 regs → 90→154 regs，occupancy **2→1 CTA/SM**（Compute 67.7%→49.7%）；BM=256 的 per-block 累加器 `512×128=65536` 恰为整个 regfile → 96 regs + 608B spill 崩到 224；ping-pong/pair-drain 触发 ptxas `C7514/C7511` wgmma 串行化反而 358/397；scale 预取 smem 与强制 2 CTA 均为负结果 | `22-fp8-gemm/fp8_gemm.cu`、`fp8_gemm_wgmma.cu`、`23-fp8-gemm-tma/fp8_gemm_tma.cu`、`24-fp8-gemm-pb/fp8_gemm_pb.cu` | 22：mma.sync 266、wgmma 768、per-block 519；23：TMA 128×128s3 1097 / 256×128s4 **1217**、per-block 906；**24：per-block 936.5（s4）+ per-tensor 同二进制 1209.6**；cuBLAS FP8 per-tensor 1379.5 / per-row 1335.9 |
| M5 | Kimi-K2.6（hidden=7168/18432, moe=2048, 384 experts） | MuonClip / Newton–Schulz 正交化 | 5 步 NS，`30N³`；测 N=2048/4096/8192 | 还原成 15 个 GEMM 的链式算子。17：自研 `mma` GEMM + 融合 `f·x+g` epilogue，**244 TFLOPS（N=4096, 24.7%）**；单 GEMM 269 vs cuBLAS 883（30.5%）；端到端 cuBLAS 链 529（2.17×）。ncu：L2 76%、occ 24%、DRAM 13% | `17-muonclip-ns/ns_muonclip.cu` | 17 N=2048/4096/8192: 1.31/8.44/65.4 ms |
| M3f | DeepSeek-V4-Pro | fused MoE expert FFN（up+gate+Swiglu / down+unpermute） | hidden=7168, moe_inter=3072, 384 experts top-6；Pp=M·6（aligned 到 BM=128） | 30：五段流水（permute→up/gate→swiglu→down→unpermute）融成 3 kernel。**K1**：A 按 `row_tok` gather（不写 `permuted_x`）+ 一个 CTA 吃 gate/up 双累加器做 SwiGLU。**K2**：`red.global.add.f32` 把 `w·acc` 直接归约回 token（不写 `D`），padding 行 `w=0` 跳过。M=8192 balanced：un-fused **22.65 → fused 22.18 ms（1.02×）**，activation 流量 9.43→1.78 GB（**−81%**）、含权重 60.2→52.5 GB（−13%）；M=16384：45.44→**42.14（1.08×）**；random 32% padding：32.80→**30.54（1.07×）**。单 kernel：K2 bn256 **898/858 TFLOPS（90.8%/86.8% 峰值，超同 shape cuBLAS 711 的 1.26×）**、ncu DRAM 79.5%/L2 88.1%；K1 **DRAM 84.5%**（33.8 GB 专家权重 → roofline 10.1 ms，实测 13.2）。正确性：K1 err 0.09%、K2 4e-6、fused-vs-unfused 1e-7 | `30-fused-moe/moe_fused.cu`、`cublas_ffn_ref.py` | M8k bal：K1 13.20 / K2 7.57 / cast 0.15；M16k bal：25.30 / 15.00；ncu K1 DRAM 84.5%、K2 DRAM 79.5% |
| M3g | DeepSeek-V4-Pro | MoE grouped GEMM + per-block FP8 缩放（e4m3 + 128×128 weight_block） | hidden=7168, moe_inter=3072, 384 experts top-6；`sa` per 行 per 128-k，`sb` per 128×128 块（BN=128 时退化成标量）；tokens=8k/16k/32k | 31：把 25 的 grouped 骨架（`b_row=group*N+n`，单 kernel 覆盖 384 expert，TMA+wgmma+WS）加上 24 的 per-block 折算（每 128-k 块 `wgmma_wait0` → `fin += sa*sb*acc`）。①prefill per-block 最佳 **806.4 TFLOPS@32k（128×128 s3）/ 790.4@16k / 707.7@8k**，为**同二进制 per-tensor**（980.5/960.5/777.4）的 **82.2%/82.3%/91.0%**；②根因 ncu：`fin` 让寄存器 90→155 → occupancy 27.9%→13.9%（2→1 CTA/SM）→ tensor pipe **62.4%→46.1%**，主导 stall 从 `long_scoreboard` 12.61 变为 `barrier` 1.16（每块 wait0 排空）；③**分组让 per-block 损失（18%）小于稠密单 GEMM（23%）**，且 `s4` 时两者都 1 CTA、折算几乎免费（16k pb s4 790.4 > pt s4 772.4）；④masked decode 纯权重带宽，per-block 免费：BM=64 **2.95ms / 2.86 TB/s / 85.4% HBM**（BM=128 75.4%）；⑤负结果：`BM=64` 省寄存器但 B 重读翻倍全线更慢（8k/16k/32k 587/656/648），`s4` 在 32k 慢于 s3；⑥对标：cuBLAS 无 block-scale 入口（`torch._scaled_mm` 只支持 TensorWise/RowWise），DeepGEMM 因容器缺 `libdwfl` 编不出来，上界取 per-tensor grouped | `31-moe-grouped-pb/moe_pb.cu`、`moe_pb_launch.h` | 32k：pb s2/s3/s4 715.1/**806.4**/761.3；pt 980.5；ncu pb tensor 46.1%/L2 74.8%/occ 13.9%，pt tensor 62.4%/L2 88.5%/occ 27.9%；masked 2.95ms 2862 GB/s |
| M3h | DeepSeek-V4-Pro | fused MoE expert FFN（FP8 per-block，端到端） | hidden=7168, moe_inter=3072, 384 experts top-6；`e4m3 + ue8m0 + weight_block 128×128`；A 已分组连续（Pp 对齐 BM=128） | 34：把 31 的 grouped per-block GEMM 接进 30 的 MoE FFN。①两版共用同一份 TMA+wgmma+WS 的 per-block grouped kernel，差异只有「down 是否融合 unpermute」；②端到端 **11.80 ms @M8k bal**（bf16 30 篇 22.65 → **1.92×**）、**20.58 @M16k**（45.44 → **2.21×**）、**15.93 @M8k rand**（32.80 → **2.06×**）；③专家权重 50.7→**25.4 GB**；K2 down **88.1% HBM**（ncu DRAM 89.3%）、K1 78%（DRAM 80.1%、long_scoreboard 占 stall 51.3%、occ 13.9%）；④**负结果一**：unpermute 融合在 FP8 下反向（K2f 3.83 vs 3.35+0.54，L2 94.4%、DRAM 73.2%）——GEMM 变快后散写不再被藏住；⑤**负结果二**：per-block + 双累加器 SwiGLU = 256>255 寄存器，`BN=256` 崩到 21.7 ms；⑥swiglu+动态量化从 39%→**92% HBM**（一行一 block/float4/warp amax） | `34-fused-moe-fp8/moe_fp8.cu` | K1 7.07 + swi 0.44 + K2 3.36 + unperm 0.54 + cast 0.15；ncu K1 DRAM 80.1%，K2f L2 94.4% |
| M6 | Qwen3 / DeepSeek-V4 / GLM | RMSNorm / QK-Norm / 融合残差 / 融合 FP8 量化 | DeepSeek-V4-Pro H=7168 eps=1e-6；Qwen3-8B H=5120, 40q/8kv, head_dim=128；GLM-5.2 H=6144, qk_nope=192 | 32：RMSNorm 每行 14KB，**寄存器缓存整行**消掉 smem 往返（L1 76%→54%）；融合 `x+res` 省 1 个 pass；`M=8192,H=7168` 实测 v1r rmsnorm **2869 GB/s / 85.6%**、v2r add+rmsnorm **2923 GB/s / 87.2%**（ncu DRAM 86.2%）。再叠 FP8 per-128 动态量化**掉到 2205 GB/s / 65.8%**（amax 的 shared atomicMax 成瓶颈）→ 融合只在省 ≥1 个 pass 时才划算。QK-Norm（一 warp 一 head）2179 GB/s / 65% | `32-fused-norm/norm_fused.cu`、`fp8_test*.cu` | v0 1481(44%) → v1 2334(70%) → v1r 2869(86%) → v2r 2923(87%)；v3r 2205(66%)；QK-Norm 2179(65%) |
| M7 | DeepSeek-V4 / Kimi-K2.6 | MLA 的 V 转置 / MN-major 免转置 | H=128, DC=512, DR=64, DV=512 | 32：CuTe 生成的 canonical MN-major 描述符经定点 GEMM 实测，**硬件忽略 LBO、按 K-major 解释 B**（改 LBO 结果逐位不变），判决 Hopper `wgmma` 免转置 V 不可行；MLA 的 PV 转置不可避免 | `32-mla-tma-v/mn_smoke.cu`、`desc_probe*.cu` | MN 描述符下 err 恒 ~0.15；LBO 64/256/1024 结果相同 |
| M8 | DeepSeek-V4-Pro | DSA compressor（KV 门控池化，主题 17 剩余） | `hidden=7168, head_dim=512, qk_rope=64`；`compress_ratios` 在 128/4 交替，`coff=1+(ratio==4)`；`ratio=4` overlap 8-slot；`compress_rope_theta=160000`（YaRN factor16/32/1/orig65536） | 35：官方 prefill 语义取自 `/ssd/models/DeepSeek-V4-Pro/inference/model.py:279`。①**合并投影 GEMM**（`wkv`+`wgate` 拼 `[2C,D]`，A 只读一遍）+ TMA/wgmma/WS，bf16 **649.8/655.2 TFLOPS（65.7/66.3%，M=32768）**，为同 shape cuBLAS 合并 bf16（809/814）的 **~80%**；②**融合池化**（online softmax + RMSNorm + RoPE，一线程一列/一 block 一窗）**83–84% HBM**；③端到端相对官方 eager **2.00×（ratio128）/2.04×（ratio4）**（M=8192 达 2.39×/2.14×），其中池化环节 ~6.6×、投影 1.70×；④ncu：投影 **L2 85–88% 受限**（Compute 仅 57–59%、DRAM 26–29%），根因 B 被 `M/BM=128` 个 m-tile 重读（L2 读 ≈7.5GB）；⑤负结果：Y 降 bf16 打平略负；⑥正确性 CPU 全流程对拍 err ~0.2% | `35-dsa-compressor/compressor.cu`、`compressor_ref.py` | e2e M8k/16k/32k：0.195/0.402/0.790 ms（ratio128）、0.395/0.787/1.577 ms（ratio4）；pool 2812/2790 GB/s；config 扫 s4 赢 s2/s3 |
| M9 | DeepSeek-V4-Pro | DSA compressor 投影 · 压 L2 + FP8 化 | 同 M8；量化方案 `e4m3 + ue8m0 + weight_block 128×128 + 1×128 动态激活`（`config.json`） | 36：①**翻案 35 的 L2 墙**——ncu 打到最佳几何 `256×128×64 s4`：tensor pipe **76.4%** 受限、L2 仅 58%，35 的 84% 是 `128×128` 自己 `BM=128` 造成；②压 L2 五招全负：A/B cluster multicast（−3.7%~−19%，tensor>70% 时耦合拖累）、L2 persisting（−2.2%，hit rate 已 82%）、K-split 双累加链（ptxas `C7517/C7518` 串行化）、消费者 `fence.proxy.async`（中性可删）；③**换数制**：按 V4-Pro 真实量化做 FP8 e4m3 per-block → 投影 **959.1（ratio128）/ 974.1（ratio4）TFLOPS**，相对 bf16 **1.385×/1.394×**，权重字节减半；ncu 瓶颈翻转为 L2 73% + `barrier`（occ 13.8%、154 regs）；④量化后池化输出最大误差 3.2%/3.8%；⑤端到端 FP8 0.550 ms（含池化），相对 35 篇自研 bf16 e2e 1.44× | `36-dsa-compressor-mcast/compressor_mcast.cu`、`compressor_fp8.cu` | bf16 s4：692.5/698.8 TFLOPS；fp8 s4：959.1/974.1；mcast A/B：609/597（CN2）~537/513（CN8）；persist 618.7 |
---

## 更新约定

- 每个新技巧一行，填全「适用场景/原理/代码/实测收益/坑」，收益写**真实数字**。
- 性能榜每次刷新最佳实现与峰值占比；对标 SOTA 要写明来源与口径。
- 卡片编号延续（A/B/C…；M 前缀用于模型场景）。
