# CUDA 算子调优系列 · 路线图（活文档）

> 这是「CUDA 算子调优」系列持续自驱工作的**控制面板**。AI agent 每次接管时：
> 先读本文 → 读 `agent_guide.md` → 读 [agent skill](../../agent_skills/kernel-opt.md) → 从「下一步」做起。
> **配套台账：[TECHNIQUES.md](TECHNIQUES.md)**（技巧记录 + 性能榜，每篇必须往里加条目）。

## 目标

一条**由浅入深、每篇都能自己跑出来**的算子调优学习线，最终覆盖两件事：

1. **通用 CUDA 优化**（01–14 已完成）：访存、共享内存、归约、GEMM、Tensor Core、异步流水线、融合。
2. **模型场景算子**（15 起，重点）：把 DeepSeek-V4.x / Kimi-K2.6 / Qwen3 / GLM 等真实模型里的
   MLA、DSA 稀疏注意力、MoE、FP8 GEMM、MuonClip 等算子**自己写出来并推到极致**；
   建立**技巧台账 + 性能榜**，把每一次优化的手段、原理、实测收益记录下来。

**北极星指标（持续逼近）**：

- 内存受限算子：≥ 90~95% HBM 带宽；
- GEMM（bf16 TC）：≥ 60% 峰值 / ≥ 80% 同口径 cuBLAS；
- attention / MLA / DSA：对标 FlashMLA / 随模型规模可用的 SOTA 实现，给出差距百分比；
- 每个算子都要有「优化前 → 优化后」的真实数字，并进入 `TECHNIQUES.md` 性能榜。

原则：

1. **代码必须能跑**：每个实验都有 `__main__` 式自测 / 参考实现对拍，实测数据写进文章与台账。
2. **数据必须真实**：数字来自本机 H100 实测（记录环境、命令、原始输出）；不允许编造。
3. **先说人话**：讲清楚「为什么慢 → 怎么想到的 → 改了什么 → 快了多少」。
4. **对标 SOTA**：能量化就量化（cuBLAS / CUTLASS / FlashMLA / DeepGEMM / flashinfer）。
5. **持续演进**：路线图是活文档，跑出新发现就调整；**未完成项用完后自行补充新任务**。

## 硬件 / 环境 / 本地资料

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3 (SXM)，CC 9.0，132 SM，HBM ~3.35 TB/s，BF16 TC dense ~989 TFLOPS，FP8 ~1978 TFLOPS |
| 容器 | `kernel_lab`（镜像 `dsv4-inf:latest`，CUDA 13.2，含 nvcc / ncu / nsys / PyTorch） |
| 关键点 | 宿主 `RmProfilingAdminOnly=1`，普通容器跑 ncu 报 ERR_NVGPUCTRPERM；`kernel_lab` 加了 `SYS_ADMIN`/`SYS_PTRACE` |

```bash
cd ~/proj/tech_record/code/kernel-opt       # /home/xieminglin 是指向 /ssd/home/xieminglin 的软链
scripts/lab.sh up                           # 确保容器在跑（幂等）
scripts/run.sh 15-mla-attn/mla_attn.cu      # 编译 + 运行
scripts/ncu.sh 15-mla-attn/mla_attn.cu --set full --kernel-name regex:mla
```

**本地参考仓库**（`~/github/`）：`FlashMLA`、`DeepGEMM`、`DeepEP`、`cutlass`、`flashinfer`、
`flash-attention`、`tilelang`、`triton`、`Liger-Kernel`、`muonclip`、`TransformerEngine`、`vllm`、
`How_to_optimize_in_GPU`、`CUDA_C_Programming_Guide.pdf`。

**本地模型**（`/ssd/models/`，可读 `config.json` 拿真实算子形状）：

| 模型 | 关键参数（用于构造 shape） |
|---|---|
| DeepSeek-V4-Pro | MLA: heads=128, kv_head=1, head_dim=512(qk_rope=64), q_lora=1536；DSA: index_n_heads=64, index_head_dim=128, index_topk=1024, compress_ratios∈{128,4,0}；MoE: 384 routed + 1 shared, top-6, moe_inter=3072；FP8 e4m3 + ue8m0, weight_block 128×128 |
| DeepSeek-V4.1-Flash | model_type=deepseek_v41；FP8 dynamic, ue8m0, weight_block 32×32, expert_dtype fp4 |
| Kimi-K2.6 | MLA（te-perf 已测 shape `(1,4096,64,192,128)`，qk=192, v=128）；kimi_k25 配置 |
| Qwen3 系列 | GQA：q=40/kv=8、32/4、64/4，head_dim=128（te-perf 已列） |

---

## 系列大纲

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 已完成并发布 · `[-]` 暂缓。

### 第一部分~第四部分（通用优化，01–14 已完成）

- [x] **01** 开篇：执行模型 / roofline / 工具链
- [x] **02** 第一个 kernel：线程层次 / vector add 三写法
- [x] **03** 正确测量：计时陷阱 / 有效带宽 / ncu 入门
- [x] **04** 访存合并与向量化：copy 行/列 7.4×
- [x] **05** 共享内存与 bank conflict：矩阵转置 466→1651 GB/s
- [x] **06** 归约与 warp shuffle：0.1%→91%
- [x] **07** Softmax 优化：多趟→融合→smem 85.3%
- [x] **08** GEMM 入门：naive→smem tiled
- [x] **09** GEMM 进阶：寄存器分块 + cp.async 双缓冲 51.5%
- [x] **10** ncu 深潜：occupancy / stall / roofline / SASS
- [x] **11** launch 配置与 occupancy：ILP × occupancy / `__launch_bounds__`
- [x] **12** 异步拷贝与流水线：cp.async N 级软流水 53.6%
- [x] **13** Tensor Core 入门：WMMA / `mma.m16n8k16` / ldmatrix 221 TFLOPS
- [x] **14** 融合与 epilogue：GEMM+bias+GELU 仅慢 1% / split-K

### 第五部分：模型场景算子（15 起，重点）

- [x] **15 MLA 注意力（一）**：DeepSeek/Kimi MLA 数学（q_lora/kv_lora、decoupled RoPE、absorb）· 三个标量实现 + roofline（证明 prefill 算力受限）· 真实 shape 台账 · TC 三 kernel 115.6 TFLOPS
- [x] **16 MLA 注意力（二）**：FlashMLA 式单 kernel 融合。online softmax 融进 QK epilogue（S 不落盘）· 累加器 C→PV A 片段零 shuffle · KV 常驻 smem · `f4`(dup2) 146.5 → **`f4s` 共享 P 消重复 170.1 TFLOPS（S=1024）/ 184.7（Sk=4096）** · wgmma(INTERLEAVE) 仅 84.6（暴露 swizzle 才是胜负手）· 距 FlashMLA 640 从 5.7× 收到 3.5×
- [x] **17 DSA 稀疏注意力（DeepSeek-V4）**：**（一）= 文章 18**：lightning indexer（64×128）打分（287–311 TFLOPS）+ exact top-k radix-select（6.75ms @32k）；**（二）= 文章 19**：稀疏 MLA 消费端（gather top-k + 尾部掩码 + 共享 P + `cp.async` 双缓冲），132–137 TFLOPS，sparse attention 相对稠密 4k 3.0× → 64k 48.6×，DSA 端到端 4k 2.5× → 64k **17.3×**。**剩余**：compressor（ratio 4/128/0）未做，见 backlog。
- [ ] **18 RoPE 融合算子**：yarn scaling（beta_fast/slow, factor 16）+ fused rotary，对照非融合
- [ ] **19 RMSNorm / QK-Norm 融合**：DeepSeek/Kimi RMSNorm、QK-norm、fused residual+norm，冲 HBM 峰值
- [ ] **20 MoE（一）：router + top-k + permutation**：384 experts top-6，token permute/unpermute，测路由与搬运开销
- [ ] **21 MoE（二）：grouped GEMM**：变长 group 的 grouped GEMM，对照 `~/github/DeepGEMM` 的 contiguous/masked 分组
- [x] **22 FP8 GEMM（一）：per-tensor / per-block scaling**：e4m3 + ue8m0 缩放，weight_block 128×128 —— **已作为文章 22 发布**（见「当前进度」）
- [x] **23 FP8 GEMM（二）：TMA + mbarrier + warp specialization**：已完成并发布（见「当前进度」）。e4m3 per-tensor 768→**1217 TFLOPS（cuBLAS 的 88.1%）**；per-block 519→906。
- [x] **23b FP8 GEMM（三）：per-block 缩放的寄存器墙 + ptxas 序列化**（= 文章 24）：已完成并发布。同二进制 per-tensor 对照 128×128 s3 **1090.8** / 256×128 s4 **1209.6**；per-block 最佳 **936.5（128×128 s4，47.3%）**。根因：`fin` 多 64 regs（90→154）→ occupancy **2→1 CTA/SM**（Compute 67.7%→49.7%）；BM=256 的 per-block 累加器 `512×128=65536` 恰为整个 regfile（96 regs + 608B spill → 224）；ping-pong/pair-drain 触发 ptxas `C7514/C7511` 串行化（358/397）；scale 预取与强制 2 CTA 均负结果。**剩余**：1-warpgroup/248-reg 布局、CUTLASS `fence_operand`+`wait0`、cluster multicast、expert fp4，见 backlog。
- [x] **24 Muon / MuonClip 优化器算子**：Newton–Schulz 迭代做正交化（zeropower）+ clip 融合；参考 `~/github/muonclip`；N=4096 级矩阵，冲 TFLOPS → **已作为系列第 17 篇发布**（见「当前进度」）
- [ ] **25 Paged KV-cache / flash-decoding 推理注意力**：GQA/MQA、block table、变长 seqlen；参考 `flashinfer`/`vllm`
- [ ] **26 量化推理算子：W4A16 dequant-GEMM（GPTQ/AWQ）**：ERNIE/GLM/Kimi 量化部署常用；dequant 融合进 GEMM
- [ ] **27 Attention 反向（FA bwd）**：自己推 dQ/dK/dV 并写 kernel，对拍 autograd（呼应 flash-attention 系列）
- [ ] **28 融合 cross-entropy / logits**（vocab=129280）：分块在线 logsumexp，省一次全量读写
- [ ] **29 MoE（三）：fused MoE（grouped GEMM + 激活 + unpermute）** 端到端对照
- [ ] **30 选择性扫描 / SSM 算子**（Hy3 / Mamba 混合模型）：并行 scan kernel

### 第六部分：极致性能（把上面的算子推到极致）

- [~] **31 wgmma（Hopper warpgroup MMA）**：用 `wgmma.mma_async` 替换 `mma.sync.m16n8k16`，目标 GEMM ≥ 400 TFLOPS。**已做（文章 20）**：SW128 swizzle 冒烟 GEMM 132、MLA 105.3（+24.5% vs INTERLEAVE）；剩 TMA/多级流水/warp specialization 才能冲高，见「下一步」。
- [x] **32 TMA + mbarrier 多级流水**：`cp.async.bulk.tensor.2d` + tensormap，替换 cp.async —— **GEMM 上已完成（文章 23）**；attention/MLA 上仍待做（V 需转置，见 backlog）
- [x] **33 Warp specialization（生产者/消费者）+ ping-pong 调度**：**FA8 GEMM 上已完成（文章 23）**，1 producer warp + N consumer warpgroup，mbarrier full/empty 握手，张量管线活跃度 72%
- [ ] **34 Persistent kernel + Stream-K**：为 tall-skinny / 不规则 M×N 做工作分解
- [ ] **35 Split-K / parallel-K**：小 M/N、大 K 的 GEMM
- [ ] **36 Autotuning 台 + 性能回归看板**：系统扫 launch 配置/流水级数，最佳结果写入 TECHNIQUES.md
- [ ] **37 CUTLASS / CuTe 对照**：复现一个 CUTLASS 例程并在同 shape 对比
- [ ] **38 GEMM 极限冲刺**：wgmma+TMA+persistent 叠加，目标 ≥ 80% 同口径 cuBLAS（4096³ bf16）
- [~] **39 MLA / attention 极限冲刺**：叠加前述技巧，对标 FlashMLA，给出差距百分比。**已做（文章 20/21）**：SW128 swizzle 105.3 → 修 V 转置访存合并 142.3 → `cp.async` 单缓冲预取 K **159.8/193.2/198.8（Sk=1k/4k/8k）**，Sk≥4096 反超 `f4s`，距 FlashMLA ~640 约 **3.3×**；剩 TMA/双缓冲/warp specialization/消 QK dup，目标 250+（差距 ≤2.6×）

### 第七部分：更远

- [ ] **40 nsys 端到端与 compute/comm overlap**（结合 `DeepEP` 概念）
- [ ] **41 Triton / TileLang 对照实现**：同一算子多 DSL 对比（呼应 flash-attention 05）
- [ ] **42 H100 vs A100 调优差异**：smem/TC/带宽对策略的影响

## 每篇的 Definition of Done

- [ ] 文章 `content/posts/cuda-kernel-opt-NN-<slug>/index.md`，`draft: false`，含 `weight`（=NN）
- [ ] 配套代码 `code/kernel-opt/NN-<slug>/`，实测通过，原始输出留存（`*.out.txt` / `.log`）
- [ ] 文章含：背景 → 现象/数据 → 根因 → 优化 → **实测对比表** → 小结 → 下一篇预告；模型类文章要给出真实 shape 与来源
- [ ] 关键结论有 ncu/实测支撑；引用的行号/API/shape 核对过
- [ ] **`TECHNIQUES.md` 至少新增 1 条技巧 + 更新性能榜**（记录手段、原理、代码位置、实测收益、坑）
- [ ] README.md 与 code/README.md 索引更新；ROADMAP 状态与「下一步」更新
- [ ] `hugo --gc --minify` 无 ERROR；publish 技能验证线上 200

## 工作循环（agent 自驱）

1. 读本文「下一步」，选一篇；若「下一步」为空，**自己从第五/六/七部分或 backlog 里补充具体、可测的新任务**再继续。
2. 写 / 改代码到 `code/kernel-opt/NN-*/`，用 `scripts/run.sh` 跑通，`scripts/ncu.sh` 采集关键指标。
3. 写文章，把实测数字填进去（不要编）；对标 SOTA 并给出百分比。
4. 更新 `TECHNIQUES.md`（技巧 + 性能榜）。
5. 走 `agent_skills/publish.md`：构建 → commit → push → 验证线上 200。
6. 更新本文件、README/code 索引、`agent_skills/kernel-opt.md` 踩坑。
7. commit & push（每篇一次）。

## 无人值守（autopilot）

`scripts/autopilot.sh` 循环启动 opencode 无头会话，每轮独立完成一篇文章增量并自动提交推送。
启动/查看/停止：`scripts/autopilot.sh {start|status|stop}`；日志 `autopilot.log`。

**重要：agent 永远不要自己创建 `AUTOPILOT_STOP`**——只在用户执行 `autopilot.sh stop` 时才停。
「下一步」做完了就自己扩充路线图，持续把算子性能往极致推。连续失败 3 次会由循环自动停。

## 阻塞

（当前无）

## 当前进度

- 2026-09-21：搭建容器 `kernel_lab` 与 `scripts/`、`common/cuda_utils.cuh`，起草路线图。
- 2026-09-21：完成并发布 **01–08**（… / Softmax / GEMM 入门）。已 push 且线上 200。
- 2026-09-21：完成并发布 **09 GEMM 进阶**：寄存器分块 8×8 + 交错映射消 bank conflict → 39.2%；float4 向量化 → 45.0%；cp.async 双缓冲 → **51.5%**（34.46 TFLOPS）；同口径 cuBLAS 75.8%（50.73 TFLOPS），达其 ~68%。
- 2026-09-21：完成并发布 **10 ncu 深潜**：手算 occupancy、`__launch_bounds__` 反优化（spill 1.31 TB 慢 34×）、五类 stall 指纹、`Memory Throughput` 语义澄清。
- 2026-09-21：完成并发布 **12 异步拷贝与流水线**：模板化 N 级软流水（`gemm_pipe<STAGES>`）：sync 30.18% → pipe4 **53.6%**（35.84 TFLOPS）；`long_scoreboard` 0.91→0.03；达同口径 cuBLAS 70.6%。
- 2026-09-21：完成并发布 **13 Tensor Core 入门**：WMMA(62)→mma+ldmatrix(115)→+cp.async **220.98 TFLOPS / 22.3%**；关键杠杆是 smem 行距 +8 padding 消 ldmatrix bank conflict（3565 万→0，2.9×）；同口径 cuBLAS BF16 672.70（68%）。
- 2026-09-21：完成并发布 **11 launch 配置与 occupancy**：ILP×occupancy 可互相替代；`__launch_bounds__` 扫描证实强制提 occupancy 负优化；thread tile 2×2(100%occ)18.8% vs 8×8(25%occ)46.8%。
- 2026-09-21：完成并发布 **14 融合与 epilogue**：GEMM+bias+GELU 仅慢 ~1%（218.61 TFLOPS）；独立 epilogue kernel ~4 TB/s；K 越小融合越值钱（11%/26%/33%）。
- 2026-09-21：**用户要求扩展**：新增第五/六/七部分（模型场景算子 + 极致性能）；新增台账 `TECHNIQUES.md`；autopilot 改为不停机、自我扩充。
- 2026-09-21：完成并发布 **15 MLA 注意力（一）**：读 `~/github/FlashMLA` 与 `/ssd/models/*/config.json` 对齐 MLA 吸收形式（V3 MH=128/DC=512/DR=64/DV=512，V4 448/64，Kimi 64head）；三个标量实现（naive/head_reuse/smem）全在 15~19 TFLOPS（ncu 证实 `Compute` 77~83%、`DRAM` 0.7%→算力受限，KV 驻 L2）；TC 三 kernel（QKᵀ+softmax+PV）**2.53ms / 115.64 TFLOPS（11.7%）**，比标量最好 `head2` 快 6.1×，距 FlashMLA 660 约 5.2×；扫 S=1024/2048/4096 暴露 S/P 物化流量 ∝S²（S=4096 占 21%），指出融合方向。
- 2026-09-21：完成并发布 **16 MLA 注意力（二）· 单 kernel 融合**：online softmax 融进 QKᵀ epilogue（S 只留寄存器）+ 累加器 `C`→PV `A` 片段零 shuffle + KV 常驻 smem。扫 7 个配置得 `f4`(dup2) 146.5、**`f4s`（smem 共享 P，QK dup=1）170.1 TFLOPS/1.717ms（S=1024），Sk=4096 → 184.7，相对 15 三 kernel 1.47×/5.78×**；ncu：DRAM 仅 7%、L1/TEX 63%、tensor 26.7%、occupancy 12.5%、`short_scoreboard` 3.05 主导。额外实现 `wgmma` 版（SS，2 warpgroup，K-major INTERLEAVE 布局）实测 **84.6 TFLOPS**——暴露无 swizzle 布局的 smem store bank conflict（2.7e8）与操作数读取低效，结论：wgmma 胜负手是 SW128 swizzle。距 FlashMLA 640 收窄到 **3.5×**。踩坑：`S` wgmma/mma 累加器每块必须清零；CUDA 13 nvcc 不认 `-arch=sm_90a`，要 `-gencode=arch=compute_90a,code=sm_90a`（`run.sh`/`ncu.sh` 已支持 `ARCH=""`）。

- 2026-09-21：完成并发布 **文章 17（主题 24）Muon / MuonClip 的 Newton–Schulz 正交化**：还原成 5 步 × 3 GEMM = `30N³` 的链式算子（Kimi-K2.6 hidden=7168/18432、moe=2048、384 experts；DeepSeek-V4-Pro 7168）。三条路径对拍（自研 fused / cuBLAS+elementwise / fp32 参考）：N=4096 自研融合 **244 TFLOPS（24.7%）/ 8.44 ms**，融合 epilogue（`f·x+g`）比不融合快 5.5%；单 GEMM 自研 269 vs cuBLAS 883（峰值 89.3%）= 30.5%；端到端 cuBLAS 链 529 TFLOPS（2.17×）。ncu：L2 76%、occ 23.8%、Compute 45%、DRAM 13% → `mma` 路径天花板，出路 wgmma+TMA。踩坑：多级 `cp.async` 的 `wait_prior` 应为 `STAGES-1`（写成 `-2` 读到半写数据）；手搓多级 smem 时 B 的 stage 基址要乘 `STAGES`。代码 `17-muonclip-ns/`（`ns_all/sweep/single/ncu.out.txt`）。**注意：主题号与文章号已解耦，下一篇是文章 18。**
- 2026-09-21：完成并发布 **文章 18（主题 17 上半）DSA 稀疏注意力（一）lightning indexer + exact top-k**：真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`（index_n_heads=64, index_head_dim=128, index_topk=1024）。①indexer 是「H^I 个小 GEMM 共享同一个 K」：标量 0.93 → TC 222 → **HG=2+BN=128 287–311 TFLOPS（31% 峰值）**；两条杠杆 head 合并（消 K 重复 ldmatrix，ncu L1 85%→62%）、加宽 BN；反例 HG=4/BN=128 溢出→69；网格顺序 `grid.x=KV` 让 Q 常驻 L2（32k 上 69.7→214，3.1×）。②exact top-k 用 radix-select（保序 uint32 + 4 趟 8-bit 直方图 + 收集），独占瓶颈是 shared atomicAdd：单直方图 26.7ms → **per-warp 私有直方图 6.75ms（3.96×，等效 3.2TB/s/95% HBM）**；整行塞 smem 长序列反慢（8.23 vs 6.75，occ 锁死 1 block/SM）。③DSA 预算（indexer+topk+稀疏 MLA 估算）相对稠密 MLA `f4s`：4k 2.9× / 16k 9.7× / 32k 14.8× / **64k 18.6×**，上限 ~30%（被 indexer O(S²) 封顶）。代码 `18-dsa-sparse/`（`dsa_S*.out.txt`、`indexer_ncu/topk_ncu/dense_mla_f4s.out.txt`）。踩坑：保序变换的逆不自逆（mask 依赖符号位，写错 out_val 全错）。
- 2026-09-21：完成并发布 **文章 19（主题 17 下半）DSA 稀疏注意力（二）稀疏 MLA 消费端**：真实 shape 同 V4-Pro（H=128, DC=512, DR=64, DV=512, topk=1024）。CTA = 「1 个 query token × $B_H$ 个 head」（同 token 全 head 共享 $I_t$，gather 一次喂满 block）；在 16 篇 `mla_shared_kernel` 上加①逐 key gather（行内 `uint4` 合并、行间随机、`topk_idx` 同址广播）②尾部 tile 掩码（已知 `valid`，前 ⌊valid/KT⌋ 块全有效、跳过掩码，113→133）③`cp.async` 双缓冲（`long_scoreboard` 4.56→1.46，64k 反超同步 11%）。扫配置：`KT` 比 `DVGRP` 重要，`s3`(BH=32) 因 4× 重复 gather 掉到 66；最佳 **p2 = 136.3@4k / 127.9@64k TFLOPS（13.8% 峰值）**，为稠密 `f4s` 的 ~75% 效率、工作量 1/(Sk/k)。sparse attention 相对稠密实测：4k 3.0× / 8k 5.9× / 16k 12.9× / 32k 25.2× / **64k 48.6×**；DSA 端到端（+18 的 indexer/topk 折算）4k 2.5× / 32k 13.3× / **64k 17.3×**，逼近 18 的 ~31× 上限。距 FlashMLA sm90 sparse prefill 640（H800）约 **4.8×**。代码 `19-dsa-sparse-attn/`（`sparse_S32768.out.txt`、`sweep_sk.out.txt`、`sparse_S32768_K512.out.txt`、`dense_mla_f4s_1024.out.txt`、`ncu_sk8192/65536.out.txt`）。
- 2026-09-21：完成并发布 **文章 21（主题 32/33/39 上半）MLA 极限冲刺（二）：修 V 转置访存 + `cp.async` 单缓冲预取 K**：真实 shape 同 20（H=128, DC=512, DR=64, DV=512）。①**诊断**：20 篇的 V 转置 `for(i){dv=i/(KT/8);kb=i%(KT/8);...ckv[(k0+kb*8+j)*DC+dv]}` 相邻线程地址差 `8*DC*2`=8KB，完全碎片化；ncu（Sk=4096）实测 `l1tex__throughput` **73.4%**、tensor pipe 18.2%、`long_scoreboard` **4.81**（等 global）。②**修法①**：把 `i` 高低位互换（`kb=i/DV; dv=i%DV`）→ 相邻线程读相邻 `dv`（差 2B、warp 覆盖 64B），写侧 SW128 散布仍 4-way（理论下限）；指令/寄存器不变，**105.3→142.3 TFLOPS（+35%）**；ncu：L1 45.8%、tensor 24.4%、long_scoreboard 2.50。③**修法②**：论证「K 只被 QK 读、`wgmma_wait0` 后空闲、PV 不碰 K」→ 用 `cp.async.cg.shared.global` 16B 在 softmax/P/PV 期间把下一块 K 覆盖写进同一块 `ks`（**单缓冲即可**），`cp.async.wait_group 0` + `__syncthreads` 保证可见；V 仍需 PV 后才能覆盖。→ **159.8/193.2/198.8 TFLOPS（Sk=1k/4k/8k）**，ncu：long_scoreboard **1.42**、tensor **30.0%**、L2 33.6%、L1 46.3%、barrier 1.17/wait 1.01 成下一道墙；**Sk≥4096 反超 16 篇 `f4s`**（183.8/173.9），距 FlashMLA ~640 收窄到 **~3.3×**（20 篇为 6.1×）。④**岔路记录**：尝试 MN-major（免转置）GMMA 描述符（`mn128.cuh`/`mn_test2.cu`）——cute `make_gmma_desc<Major::MN>` 的 LBO/SBO 与硬件解释相反，cute 值直接 out-of-range、强行对调后结果错乱，**未跑通**（见台账 J3），故保留转置+合并。代码 `21-mla-wgmma-pipe/`（`mla_pipe.cu`、`mla_pipe3.cu`、`sweep.out.txt`、`ncu_20baseline/ncu_pipe/ncu_pipe3.out.txt`）。
- 2026-09-21：完成并发布 **文章 20（主题 16b/31 上半）MLA 极限冲刺（一）wgmma + SW128 swizzle**：把 `wgmma` 操作数布局从无 swizzle 的 K-major `INTERLEAVE` 换成 **K-major SW128**（`layout_type=1`，物理 16B 列 `c'=c^r`），依据 CUTLASS canonical 布局推导描述符（`LBO=1`、`SBO=(K/64)*1024`、k16 步进 `floor(s/4)*1024+(s%4)*32`）。①冒烟 GEMM `gemm_sw128.cu` 一次验证对：**148.66 @2048³ / 132.04 TFLOPS @4096³**；②`qk576_test.cu`（9 atom、36 k16 步）err 1.6e-7、`pv_test.cu` err 1.5e-8 分别锁死 QK/PV；③移植融合 MLA `mla_wgmma_sw.cu`：**105.30 / 112.91 / 119.89 TFLOPS（Sk=1k/2k/4k）**，比 16 篇 wgmma INTERLEAVE 84.6 **+24.5%**，逐位对齐 `f4s` 误差。④踩坑（已入台账 G12）：nvcc 把相邻两个 bf16 标量存储合并成 `st.shared.u32` 并**丢高 16 位**，P 写回 3553 字节错位；改 `__floats2bfloat162_rn` 打包 u32 → 0 错位（`bf16_store_pitfall.cu`）。⑤ncu：store bank conflict 2.7e8→4.4-way（理论下限）、L1/TEX 86%→70.8%，但 DRAM 5.1%、Compute 18.5%、occ 12.5%、205 reg、221KB smem（1 CTA/SM）、No Eligible 80.9%、47.8% 停 L1TEX（global load 未流水）→ 仍 1.62× 慢于 `f4s` 170、距 FlashMLA ~640 约 **6.1×**。结论：SW128 是必要的但不是充分的——瓶颈转为「1 CTA/SM + 单缓冲」；下一步 TMA/多级流水/warp specialization。代码 `20-mla-wgmma-sw128/`（`gemm_sw128.out.txt`、`small_tests.out.txt`、`mla_wgmma_sw.out.txt`、`mla_sw_ncu.out.txt`、`bf16_store_pitfall.out.txt`）。

- 2026-09-21：完成并发布 **文章 22（主题 22）FP8 GEMM（一）：e4m3 的 mma.sync 与 wgmma + per-tensor / per-block 缩放**：真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`（hidden=7168, moe_inter=3072, 384 experts top-6；e4m3 + ue8m0 + weight_block 128×128 + dynamic 1×128 激活），测 `M=4096,N=3072,K=7168`（180.4 GFLOP，AI≈1785 → 算力受限）。①`mma.m16n8k32` 版：FP8 片段用「两个 fp8 = 一个 b16」复用了 `ldmatrix`（`x4` 取 a0..a3、`x2` 取 b0/b1），smem 行距 BK+16 消 bank；`cp.async` 4 级流水 → **266.3 TFLOPS（13.5%）**；ncu：Compute 57%、tensor pipe 41%、`math_pipe_throttle` 1.12 + `wait` 1.59、DRAM 仅 7.9% → **发射端口受限，mma.sync 在 Hopper 打不满 FP8**。②换 `wgmma.m64n128k32`：**FP8 的 K-major SW128 atom 与 bf16 逐字节同构**（8 行×128B，只是每行 128 个 e4m3），20 篇描述符原样复用；主循环用 `wgmma.commit_group` + `wait_group(STAGES-2)` 代替每块 `wait0`（acc 到最后才读）→ **602/717/768 TFLOPS（s2/128×128s3/256×128s3）**，相对 mma.sync **+2.89×**，**达 cuBLAS FP8（1381.7）的 55.6%、同 shape cuBLAS bf16（800.7）的 96%**。③per-block（1×128 激活 + 128×128 权重，`sb` 每 n-tile 退化成标量）→ **519 TFLOPS（26.2%）**，每块折算 `wait0` 切断流水。踩坑：①相邻两列 c0/c1 各有 `sb`，共用一个 scale 只在逐列随机 scale 时暴露（常数 scale 会「假通过」）；②FP8 wgmma asm 尾部是 `p,scaleA,scaleB`（3 个，非 bf16 的 5 个）；③`__launch_bounds__` 写死 256 而 BM=256 需 512 线程时只算一半行、性能虚高且采样巧合通过（修正后 256×256 真值 218）。ncu（wgmma 版）：Compute 38.5%、occupancy 24.7%、No Eligible 59.9%、1 CTA/SM（regs+smem 双限）。代码 `22-fp8-gemm/`（`fp8_gemm.cu`、`fp8_gemm_wgmma.cu`、`cublas_fp8_ref.py`、`fp8_mma/fp8_wgmma/cublas_fp8/ncu_mma/ncu_wgmma.out.txt`）。

- 2026-09-21：完成并发布 **文章 23（主题 23/32/33）FP8 GEMM（二）：TMA + mbarrier + warp specialization**：真实 shape 同 22（取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：hidden=7168, moe_inter=3072，e4m3 + ue8m0 + weight_block 128×128）测 `M=4096,N=3072,K=7168`。①A/B 装载从 22 的「256 线程各自算 SW128 地址 + `LDGSTS.16B`」换成 **TMA `cp.async.bulk.tensor.2d` + `CU_TENSOR_MAP_SWIZZLE_128B`**——TMA 硬件写出的 smem 布局与 wgmma 的 K-major SW128 描述符逐字节一致，kernel 里零 swizzle 代码（`make_desc_sw128` 原样复用；驱动枚举无 FLOAT8，用 `UINT8` 搬字节，需 `-lcuda`；tensormap 作 `__grid_constant__` 参数）。②**warp specialization**：1 producer warp 专职 TMA（`mbarrier.arrive.expect_tx` + `full[st]`），`(BM/64)` 个 consumer warpgroup 只做 wgmma，用 `empty[st]`（count=消费者线程数）回压，消费者以 `wgmma.wait_group<STAGES-2>` 决定释放哪个 stage。③**两个关键坑**：相位数组 `phase[st]` 动态下标掉 **local memory**（ncu 报 local memory 占 L1TEX 47.5% sector、`long_scoreboard` 8.1，只有 988）→ 改成常量相位 `(kb/STAGES)&1`；epilogue 标量 store 半 sector（ncu 报 50% excessive）→ `float2` 向量存储。④结果 per-tensor **1217.3 TFLOPS（61.5%）**（22 的 768.3 → +58%），达 **cuBLAS FP8（1381.3）的 88.1%（差距 1.13×）**、超过同 shape cuBLAS bf16（800.4）的 1.52×；扫配置最佳 `256×128×BK128 s4`（smem 197KB、1 CTA/SM），对比 `128×128 s3`（2 CTA/SM 但 L2 80%）证明「降 L2 流量 > 堆 occupancy」。⑤per-block（1×128 激活 + 128×128 权重）**519 → 906 TFLOPS（+74%）**，ncu 寄存器 150、occ 13.8%，瓶颈转为每块折算等 mma。ncu（最佳）：Compute 67.8%、tensor pipe `hmma_cycles_active` **72.0%**、DRAM 27%、L2 56.7%、local mem 0、`long_scoreboard` 9.15。代码 `23-fp8-gemm-tma/fp8_gemm_tma.cu`（`fp8_tma.out.txt`、`ncu_tma256x128s4/ncu_tma128x128s3/ncu_pb128x128s3/ncu_stalls_best/cublas_ref.out.txt`）。
- 2026-09-21：完成并发布 **文章 24（主题 23b）FP8 GEMM（三）：per-block 缩放的寄存器墙 + ptxas 序列化**：真实 shape 同 22/23（`M=4096,N=3072,K=7168`，取自 `/ssd/models/DeepSeek-V4-Pro/config.json`）。①**同二进制对照**：同一 kernel 加 `MODE=3`（读 `sa/sb` 但不折算、`wait_group<STAGES-2>` 流水）作 per-tensor 基准，锁死差距——128×128 s3：per-tensor **1090.8** vs per-block **913.3**（−16.3%）；per-tensor 256×128 s4 **1209.6**。②**根因=寄存器→occupancy**：per-block 多一个跨 k 保留的 fp32 `fin`（每线程 64 regs），288 线程寄存器 **90→154**，2 CTA/SM（113 门槛）→ **1 CTA/SM**；ncu：Compute **67.7%→49.7%**、L2 57%→65%、occ 25.4%→13.7%。③**BM=256 对 per-block 不可行**：`acc[64]+fin[64]=128` regs × 544 线程 > 120 预算，实测 96 regs + 608B spill → **224 TFLOPS**；折中 BM=192 → 683–699。④**ping-pong/pair-drain 重叠折算失败**：读飞行中的 wgmma 累加器触发 ptxas `C7514`/`C7511` 把 wgmma 串行化，358/397（比老实 `wait0` 的 913 慢 2.3×），寄存器 168+spill。⑤**负结果**：scale 预取 smem 913→871（更慢）、`__launch_bounds__(288,2)` 强压 2 CTA → 219.7（spill 608B）。⑥**per-block 最佳 936.5（128×128 s4，47.3%）**，+3.3% vs 23 篇 906；同配置 per-tensor 938.7（折算在该 config 几乎免费）。对标：cuBLAS FP8 per-tensor 1379.5（本篇 67.9%）、per-row 1335.9（便宜算子，仅上界参考）。代码 `24-fp8-gemm-pb/fp8_gemm_pb.cu`（`fp8_pb.out.txt`、`ncu_pb_s4.out.txt`、`ncu_pt_s4.out.txt`）。

## 下一步（明确到可执行）

- [x] **文章 24 — 主题 23b FP8 GEMM（三）：per-block 缩放的寄存器墙 + ptxas 序列化**：已完成并发布（见「当前进度」）。产出：①同一二进制里 per-tensor 对照（MODE=3）锁死差距（128×128 s3 1090.8 vs per-block 913.3）；②定位根因=per-block 的 `fin` 多 64 寄存器 → 90→154 regs → 2→1 CTA/SM；③证明 BM=256 的 per-block 累加器恰为整个 regfile（224 TFLOPS）；④ping-pong/pair-drain 被 ptxas `C7514/C7511` 串行化（358/397）；⑤per-block 最佳 **936.5（128×128 s4，+3.3% vs 23 篇 906）**。代码 `24-fp8-gemm-pb/fp8_gemm_pb.cu`。
- [ ] **文章 25 — 主题 21 MoE（二）：grouped GEMM**（下一步，优先）：从 `/ssd/models/DeepSeek-V4-Pro/config.json`（384 routed + 1 shared, top-6, moe_inter=3072, hidden=7168）构造变长分组，复用 23/24 的 TMA + wgmma kernel 做 grouped 版本；对照 `~/github/DeepGEMM` 的 contiguous/masked 布局，给出与 SOTA 的差距。注意 DeepGEMM 本地 `_C` 扩展需 `elfutils/libdwfl-dev` 才能编译，若容器缺包则退化为「与自研 baseline + cuBLAS 对比」。
- [ ] **文章 26 — 主题 20 MoE（一）：router + top-k + permutation**（可并行）：384 experts top-6 的 router GEMM、top-k（复用 18 篇 radix-select）、token permute/unpermute，实测路由与搬运开销。

- [x] **16 MLA 注意力（二）**：已完成（见「当前进度」）。
- [x] **文章 17 — 主题 24 Muon / MuonClip 的 Newton–Schulz 正交化**：已完成并发布（见「当前进度」）。
- [x] **文章 18 — 主题 17（上半）DSA lightning indexer + exact top-k**：已完成并发布（见「当前进度」）。
- [x] **文章 19 — 主题 17（下半）DSA 稀疏 MLA 消费端**：已完成并发布（见「当前进度」）。剩余 **compressor（ratio 4/128/0）** 未做，列入 backlog。
- [x] **文章 20 — 主题 16b/31（已完成）MLA 极限冲刺（一）：SW128 swizzle + wgmma**：把 `wgmma` 的操作数 smem 布局从无 swizzle 的 K-major `INTERLEAVE` 换成 **K-major SW128（`layout_type=1` + `c'=c^r` 物理布局）**。①用 CUTLASS canonical 布局推导描述符（`LBO=1`、`SBO=(K/64)*1024`、k16 步进 `floor(s/4)*1024+(s%4)*32`）；②小 GEMM 冒烟验证 → **132 TFLOPS @4096³**；③QK576/PV 定点测试分别锁死；④移植回融合 MLA：INTERLEAVE 84.6 → **105.3 / 112.9 / 119.9 TFLOPS（Sk=1k/2k/4k，+24.5%）**；⑤踩坑：nvcc 合并相邻 bf16 标量存储丢高 16 位（3553 字节错位），改 `pack2` u32 修复。ncu：store bank conflict 2.7e8→4.4-way、L1 86%→70.8%，但 221KB smem 锁 1 CTA/SM（occ 12.5%、No Eligible 80.9%、47.8% 停 L1TEX）→ 仍 1.62× 慢于 `f4s`、距 FlashMLA ~640 约 6.1×。代码 `20-mla-wgmma-sw128/`。
- [x] **文章 21 — 主题 32/33/39（上半）MLA 极限冲刺（二）：修 V 转置访存 + `cp.async` 单缓冲预取 K**：已完成并发布（见「当前进度」）。做了两件事，均实测：①V 转置迭代顺序 `dv` 最快 → global 读合并（`l1tex` 73%→46%，105.3→142.3）；②`cp.async` 在 softmax/P/PV 期间预取下一块 K（K 只被 QK 读，`wgmma.wait_group` 后即可覆盖，**单缓冲无需双缓冲**）→ 159.8/193.2/198.8（Sk=1k/4k/8k），Sk≥4096 反超 `f4s`，距 FlashMLA ~3.3×。**TMA/双缓冲/warp specialization 仍未做**（单缓冲约束让 V 的延迟藏不住、L1 46% + barrier 是下一道墙）。
- [x] **文章 22 — 主题 22 FP8 GEMM（一）：e4m3 的 mma.sync 与 wgmma + per-tensor / per-block 缩放**：已完成并发布（见「当前进度」）。
- [x] **文章 23 — 主题 23/32/33 FP8 GEMM（二）：TMA + mbarrier + warp specialization**：已完成并发布（见「当前进度」）。
- [ ] **其他模型场景**：RoPE/RMSNorm/compressor…（见上）。模型类文章优先；20 的 SW128 与主题 31/38/39 共用基础设施，是性能主线。
  - **编号说明**：「系列大纲」里的数字是**主题编号**（稳定 ID），文章 `NN` 是**发布顺序**。主题 24（MuonClip）= 文章 17、主题 17 上半 = 文章 18、下半 = 文章 19、主题 22（FP8 GEMM 一）= 文章 22、主题 23/32/33（FP8 GEMM 二 · TMA）= 文章 23、主题 23b（FP8 GEMM 三 · per-block 寄存器墙）= 文章 24。**下一篇是文章 25**；发布时把「文章号 ↔ 主题号」记进「当前进度」。

## 灵感 / backlog（想到就记，别丢）

- **MLA MN-major（免 V 转置）**：21 篇盲猜 GMMA MN-major 描述符失败（cute 的 LBO/SBO 与硬件语义相反、结果错乱）。正确做法是用 CUTLASS CuTe `make_tiled_mma` + `partition`/`copy` 让库生成 smem 布局与描述符，再对照手动版；成功后可消掉 21 篇没藏住的 V 转置/预取开销。
- **MLA V 的 TMA 预取**：单缓冲限制让 V 的 global 延迟藏不住（长上下文 `long_scoreboard` 残留）。要么双缓冲（需先给 Q 腾 smem），要么用 TMA + mbarrier 把 V 也做成 producer/consumer。
- **消 QK dup / P 走 RS**：`wgmma` 版两个 warpgroup 各算一份完整 S（dup 2×）；改 n32 分工或让 P 从寄存器直接喂 PV（RS）可省 smem 往返与一次 `__syncthreads`。
- **FP8 per-block 的 1-warpgroup / 248-register 布局**（24 篇遗留）：DeepGEMM 用 BLOCK_M=128、**1 个 math warpgroup（128 线程）**、`setmaxnreg` 给到 248 寄存器，把 `final_accum(128)+accum(64)=192` 塞下且不 spill。我们当前 2 WG/288 线程只有 227 上限，per-block 必掉 1 CTA。试这条路看能否把 per-block 从 936 推向 1100+；配套 CUTLASS `warpgroup_fence_operand` + `warpgroup_wait<0>` 避开 ptxas 串行化。
- **FP8 per-block · TMA cluster multicast**：同一 A tile 在 N 方向广播（`cp.async.bulk.tensor.2d...multicast::cluster`），砍 L2→SM 重复装载（24 篇 per-block L2 65%）。可先在 per-tensor 上验证收益。
- **Blackwell tcgen05 对照**：tcgen05 累加器放 tensor memory（TMEM），彻底不吃寄存器，per-block 的寄存器墙消失。若有 sm_100 机器，重跑本 shape 对照。
- **DSA compressor**（主题 17 剩余）：ratio ∈ {4,128,0} 的 KV 压缩算子，V4-Pro/Lite 的层间压缩比不同，用来把超长上下文压成更少的 key，再接 indexer/top-k。
- DeepGEMM 的 JIT + contiguous grouped GEMM 与 DeepSeek-V4 的 expert 分布对齐。
- FlashMLA 的 split-KV + `m64n...` 细节，和本文 16 逐段对照。
- ~~MuonClip 的 Newton–Schulz 5 步迭代（zeropower_via_newtonschulz）在 N=4096 上的 roofline：它到底是计算受限还是访存受限？~~ → **已答（文章 17）**：算力受限（DRAM 13%），瓶颈是自研 `mma` GEMM 的 L2 放大 + 寄存器墙；下一步 wgmma/TMA。
- 用 nsys 把「GEMM + 两个 epilogue」与「融合 GEMM」对比端到端。
- 把每篇 kernel 存成 v0/v1/v2 版本序列，做「优化步骤 diff 视图」。
- 给 01 roofline 画 mermaid 图。
- 复现 te-perf 文章里的生产 shape，把算子级结论接回模型级。
