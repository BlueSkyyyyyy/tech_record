# kernel-opt — CUDA 算子调优系列配套代码

配套博客系列：`content/posts/cuda-kernel-opt-*`。路线图与进度见 [ROADMAP.md](ROADMAP.md)，
优化技巧与性能榜见 [TECHNIQUES.md](TECHNIQUES.md)。

> 系列分两阶段：**01–14 通用 CUDA 优化**；**15 起为模型场景算子**（DeepSeek-V4.x / Kimi-K2.6 / Qwen3 的
> MLA、DSA 稀疏注意力、MoE、FP8 GEMM、MuonClip 等），并把性能持续推向极致。

## 目录

| 目录 | 对应文章 | 内容 |
|---|---|---|
| `common/` | — | 公共头 `cuda_utils.cuh`（设备信息 / 计时 / 带宽换算） |
| `scripts/` | — | `lab.sh` 容器管理、`run.sh` 编译运行、`ncu.sh` 剖析 |
| `01-overview/` | 01 开篇 | 环境自检小程序 |
| `02-first-kernel/` | 02 第一个 kernel | vector add 的三种写法 |
| `03-measurement/` | 03 正确测量 | 计时陷阱 / launch 开销 / 带宽实验 |
| `04-coalescing/` | 04 访存合并与向量化 | 合并 vs 跨行访问 vs float4 的带宽对比 |
| `05-transpose/` | 05 共享内存与 bank conflict | 矩阵转置：naive / smem 无 padding / padding |
| `06-reduction/` | 06 归约与 warp shuffle | 求和：原子反面教材 / smem 树 / warp shuffle / float4 |
| `07-softmax/` | 07 Softmax 优化 | 多趟 / 融合 / shared memory 缓存整行 |
| `08-gemm/` | 08 GEMM 入门 | naive / smem tiling（TFLOPS + ncu 瓶颈）|
| `09-gemm-advanced/` | 09 GEMM 进阶 | 寄存器分块 / float4 向量化 / cp.async 双缓冲 + cuBLAS 对照 |
| `10-ncu-deep/` | 10 ncu 深潜 | occupancy 手算 / warp stall 指纹 / roofline / source-SASS 对照 |
| `11-launch-occupancy/` | 11 launch 配置与 occupancy | ILP × occupancy / `__launch_bounds__` 扫描 / thread tile 扫描（落地 09 GEMM）|
| `12-async-pipeline/` | 12 异步拷贝与流水线 | cp.async 2/3/4/5 级软流水线 + ncu stall 对照（落地 09 GEMM）|
| `13-tensor-core/` | 13 Tensor Core 入门 | WMMA / 裸 mma.m16n8k16 + ldmatrix / bank conflict padding + cuBLAS BF16 对照 |
| `14-fusion-epilogue/` | 14 融合与 epilogue | GEMM+bias+GELU/ReLU 寄存器融合 vs 独立 epilogue kernel（11%/26%/33% 提速，随 K 缩小放大）|
| `15-mla-attn/` | 15 MLA 注意力（一） | MLA 吸收成 MQA：naive/head_reuse/smem 标量版（15~19 TFLOPS，算力受限）+ TC 三 kernel（115.6 TFLOPS），扫 S=1k/2k/4k |
| `16-mla-fused/` | 16 MLA 注意力（二） | 单 kernel 融合：online softmax + 累加器 C→A 零 shuffle + KV 常驻 smem；`f4s` 共享 P 消重复 **170.1 TFLOPS**（Sk=4096 → 184.7）；附 `wgmma` 尝试（84.6，暴露 swizzle 才是胜负手）与描述符冒烟测试 |
| `17-muonclip-ns/` | 17 Muon/MuonClip NS 正交化 | 5 步 NS = 15 个 GEMM（`30N³`）：自研 `mma` GEMM + 融合 `f·x+g` epilogue（**244 TFLOPS**，N=4096）；对照 cuBLAS 链（529）、fp32 参考；单 GEMM 269 vs cuBLAS 883；ncu 定位 L2/occupancy 瓶颈 |
| `18-dsa-sparse/` | 18 DSA 稀疏注意力（一） | DeepSeek-V4 lightning indexer（H^I=64, d=128）打分 + exact top-k：TC + head 合并 HG=2 + BN=128 → **287–311 TFLOPS**（约 330× 标量）；radix-select + per-warp 直方图 top-k → **6.75ms @S=32768**（等效 3.2TB/s）；DSA 预算 64k 相对稠密 MLA **18.6×**；含 indexer/topk 的 ncu 与稠密 MLA 对照输出 |
| `19-dsa-sparse-attn/` | 19 DSA 稀疏注意力（二） | 稀疏 MLA 消费端：CTA = 1 token × $B_H$ head，逐 key gather top-k 的 `c_kv`+`k_rope`、尾部 tile 掩码、共享 P；`cp.async` 双缓冲把 gather 延迟藏起来（`long_scoreboard` 4.56→1.46）。**132–137 TFLOPS**（~75% 稠密 f4s 效率）；sparse vs 稠密 attention 4k **3.0×** → 64k **48.6×**；DSA 端到端 4k **2.5×** → 64k **17.3×**；距 FlashMLA sparse prefill 640 约 4.8× |
| `20-mla-wgmma-sw128/` | 20 wgmma + SW128 swizzle | 把 `wgmma` 操作数的 smem 布局从 `INTERLEAVE` 换成 **K-major SW128**（`layout_type=1`，16B 列 `c'=c^r`）。`wgmma_sw128.cuh` 描述符/布局/步进；冒烟 GEMM **132 TFLOPS @4096³**；QK576/PV 定点测试；`bf16_store_pitfall.cu` 复现 nvcc 合并 bf16 存储的坑。融合 MLA **105.3/112.9/119.9 TFLOPS**（Sk=1k/2k/4k），比 INTERLEAVE 84.6 **+24.5%**；ncu 指出剩余瓶颈是 1 CTA/SM + 单缓冲 |
| `21-mla-wgmma-pipe/` | 21 MLA 极限冲刺（二） | ① **V 转置访存合并**（迭代顺序 `dv` 最快，`l1tex` 73%→46%，105.3→142.3）；② **`cp.async` 单缓冲预取下一块 K**（K 生命周期早于 PV，同一 buffer 边算边覆盖，`long_scoreboard` 2.50→1.42）→ **159.8/193.2/198.8 TFLOPS**（Sk=1k/4k/8k），Sk≥4096 反超 `f4s`。附 `mn128.cuh`/`mn_test2.cu` 记录 MN-major（免转置）描述符失败尝试 |
| `22-fp8-gemm/` | 22 FP8 GEMM（一） | DeepSeek-V4 e4m3 + ue8m0 块缩放，真实 shape 4096×3072×7168。`fp8_gemm.cu`：`mma.m16n8k32` + 「两个 fp8=一个 b16」复用 `ldmatrix` + `cp.async` 流水 → **266 TFLOPS**（发射端口受限）。`fp8_gemm_wgmma.cu`：`wgmma.m64n128k32` + SW128（与 bf16 同构）+ `wait_group` 流水 → **768 TFLOPS**（+2.89×，cuBLAS 55.6%，超同 shape cuBLAS bf16 的 96%）；per-block 折算 → 519。`cublas_fp8_ref.py` 取 cuBLAS 对照；含 `ncu_*.out.txt` |
| `23-fp8-gemm-tma/` | 23 FP8 GEMM（二） | 同 shape，把 A/B 装载换成 **TMA（`cp.async.bulk.tensor.2d` + SW128，与 wgmma 描述符逐字节同构）+ mbarrier + warp specialization**（1 producer warp + 4 consumer warpgroup）。`fp8_gemm_tma.cu`：per-tensor **768 → 1217.3 TFLOPS（+58%，cuBLAS FP8 的 88.1%）**、tensor pipe 活跃度 72%；per-block 519 → **906（+74%）**。两个关键坑：相位数组动态下标掉 local memory（988→1217）、epilogue 标量 store 半 sector（`float2` 修复）。含 `fp8_tma.out.txt`、`ncu_*.out.txt`、`cublas_ref.out.txt` |
| `24-fp8-gemm-pb/` | 24 FP8 GEMM（三） | 同 shape，诊断 per-block 缩放为何到不了 per-tensor。`fp8_gemm_pb.cu` 用 `MODE` 模板参数给出 5 条路径：`0`=per-block 单累加器、`1`=ping-pong、`2`=pair-drain、`3`=同二进制 per-tensor 对照；`PRELOAD`=scale 预取 smem、`MINB`=launch_bounds minBlocks。结论：per-block 的 `fin` 多 64 fp32 regs → 90→154 regs → occupancy **2→1 CTA/SM**；BM=256 的累加器恰为整个 regfile（spill → 224）；ping-pong 触发 ptxas `C7514/C7511` 串行化（358）。per-block 最佳 **936.5（128×128 s4）**，同配置 per-tensor 1090.8/256×128 1209.6。含 `fp8_pb.out.txt`、`ncu_pb_s4.out.txt`、`ncu_pt_s4.out.txt` |
| `25-moe-grouped-gemm/` | 25 MoE grouped GEMM | DeepSeek-V4-Pro 真实 shape（hidden=7168, moe_inter=3072, 384 routed, top-6）。`moe_grouped.cu`：把 expert 编码进 B 的行坐标 `b_row=group*N+n`，A/B 都用 2D TMA，复用 23 篇的 TMA+mbarrier+warp specialization，一个 kernel 覆盖全部 expert；`MASKED` 模板切 contiguous（prefill）/ masked（decode）。prefill 相对 per-expert loop **3.59×/2.36×/1.47×**（tokens=8k/16k/32k），最佳 958–1008 TFLOPS（aligned），反超 cuBLAS per-expert graph loop **1.17×**；masked decode 读 8.45GB 权重达 **2.8 TB/s（83.6% HBM）**。坑：TMA `boxR` 不随 BN 变 → 死锁。`cublas_moe_ref.py`（cuBLAS eager+graph 参照）。含 `moe_S{8192,16384,32768}.out.txt`、`cublas_moe.out.txt`、`ncu_grp128x256s3.out.txt` |
| `26-moe-cluster-multicast/` | 26 MoE + TMA cluster multicast | 同 25 的真实 shape，给 grouped kernel 加**沿 N 的 A 广播**：`cudaLaunchKernelEx`+`clusterDim`、leader 发 `cp.async.bulk.tensor.2d...multicast::cluster`、每 CTA 各自 `arrive.expect_tx`、共享 operand 用 `mapa.shared::cluster` arrive 到 leader、私有/共享 empty barrier 分离、init 后/退出前 `cluster_sync`。协议跑通且与 CN=1 逐位一致；CN=2 把 A 的 L2 读 sector 砍 **15–19%**、`long_scoreboard` 11.6→8.0，但**只有 32768 tokens+128×128 赢 +4.3%（960.7→1001.6）**，8k −7.4% / 16k −2.2% / masked −3.9%。根因：`lts__throughput`（请求/延迟）≠ `lts__t_sectors`（字节）。BM=256 负结果（256×256 触发 `C7511` 崩到 124）。含 `moe_cluster.cu`、`moe_cluster_launch.h`、`moe_cluster_S{8192,16384,32768}.out.txt`、`ncu_c1_c2_16384/32768.out.txt` |
| `27-moe-router/` | 27 MoE router + top-k + 置换 | DeepSeek-V4-Pro 真实 shape（hidden=7168, n_routed=384, topk=6, `sqrtsoftplus`/`noaux_tc`/`scale=2.5`）。七个手写 kernel 还原前门：`gate_mma_pipe_t`（mma+ldmatrix+cp.async，helper 模板化，BM=128 vs 256）**204→257 TFLOPS**（cuBLAS 738 的 **27.7%**，ncu L2 60.7%/Compute 35.3%/No Eligible 57.6%，BM=256 掉 1 CTA/SM −10%）；`router_topk_kernel`（1 warp/token，`sqrtsoftplus+bias` 选专家、未加 bias 的 `s` 归一化 ×2.5）0.031ms@16k、CPU 对拍 8/8 err 6e-8；直方图融进 top-k 的 `COUNT` 版省 19–27%；`permute_copy_tok_kernel`「1 block/token 读一次写 K 行」比 `permute_copy_pos_kernel` **1.53×**（81% HBM）；`unpermute_kernel` 融合加权 **91.7% HBM**。端到端 1.651ms@16k（搬运占 69%），对标 torch perm 1.50× / unperm **17.8×** / e2e **6.6×**。`moe_router_ref.py`（cuBLAS+topk+index_select+index_add 参照）。含 `moe_M{4096,8192,16384,32768}.out.txt`、`check_M16384.out.txt`、`ncu_{gate,perm,unperm,topk}.out.txt`、`torch_ref_M16384.out.txt` |
| `28-moe-gate-wgmma/` | 28 MoE gate GEMM（wgmma+TMA+WS） | DeepSeek-V4-Pro 真实 shape（`X[M,7168]·Wg[384,7168]ᵀ`，M=16k/32k）的 router gate GEMM。`gate_wgmma.cu` 两条路径：`gate_async_kernel`（`wgmma.m64n128k16` SS + SW128，bf16 atom 8 行×64 元素故 BK=64；B 天然 K-major 免转置）**337/360 TFLOPS**；`gate_ws_kernel`（TMA `cp.async.bulk.tensor.2d` + mbarrier + warp specialization，1 producer + 2/4 consumer WG，`KSPLIT` 模板做 split-K 负结果）**565.4@16k（128×128 s3）/ 600.3@32k（256×128 s4）**，达 cuBLAS bf16 **77.0%/79.8%**、相对 27 篇 **2.80×/2.29×**；ncu tensor pipe 68.2%、L2 79.1%、0 bank conflict。`cublas_gate_ref.py`（torch cuBLAS 对照）。含 `gate_M{16384,32768}.out.txt`、`ncu_ws128s3_M16384/ws256s4_M32768.out.txt`、`cublas_gate.out.txt` |
| `29-moe-gate-multicast/` | 29 gate GEMM + cluster multicast | 同 28 的真实 shape，给 `gate_ws_kernel` 加 `AXIS` 选轴：`gate_mcast_kernel<...,CN,AXIS>`（AXIS=1 `cluster.x` 广播 A / AXIS=2 `cluster.y` 广播 B），协议同 26。①**A-multicast CN=3**：L2 读 sector 46.2→**34.0 M（−26%）**、L2 吞吐 80.4%→48.5%，但慢 **−23.4%**（tensor 68.8%→51.5%，证伪 28 篇「A 是墙」）；②**B-multicast CN=2**：128×128 s3 556.3→**587.2（+5.6%）**，却输给直接放大 BM 的 256×128 s4（**612.2**）——放大 BM 与广播 B 是同一优化；③**wide-N（一个 CTA 算完 N=384）**：3 累加器 192 regs → ptxas `C7511` 串行化 wgmma（145/41 TFLOPS）；④最佳 612.2@32k（cuBLAS 81.3%）/ 547.0@16k（74.5%）。含 `gate_M{16384,32768}.out.txt`、`ncu_best_c1.out.txt`、`ncu_multicast_cmp.out.txt` |
| `30-fused-moe/` | 30 fused expert FFN | DeepSeek-V4-Pro 真实 shape（hidden=7168, inter=3072, 384 experts top-6）。`moe_fused.cu` 把五段流水（permute→up/gate→swiglu→down→unpermute）融成 3 kernel，两版共用同一份 GEMM（模板 `FUSE`）：**K1** A 按 `row_tok` gather（不写 `permuted_x`）+ 一个 CTA 吃 gate/up 双累加器做 **SwiGLU**；**K2** epilogue 用 `red.global.add.f32` 把 `w·acc` 归约回 token（不写 `D`），padding 行跳过。端到端 **1.02×（M=8k bal）→ 1.08×（M=16k）/ 1.07×（random 32% pad）**，activation 流量 **−81%**、含权重总流量 −13%；单 kernel K2 bn256 **898 TFLOPS（90.8% 峰值）/ 858 fuse**，超同 shape 稠密 cuBLAS（711）1.26×；K1 ncu **DRAM 84.5%**（33.8 GB 专家权重，权重带宽受限）。正确性 K1 0.09% / K2 4e-6 / fused-vs-unfused 1e-7。`cublas_ffn_ref.py`（同 shape cuBLAS 上界）。含 `moe_M8192_bal/M16384_bal/M8192_rand.out.txt`、`ncu_k1/ncu_k2.out.txt`、`cublas_ffn_ref.out.txt` |
| `31-moe-grouped-pb/` | 31 MoE grouped + per-block FP8 | DeepSeek-V4-Pro 真实 shape（hidden=7168, inter=3072, 384 experts top-6, e4m3 + ue8m0 + `weight_block=128×128`）。`moe_pb.cu` + `moe_pb_launch.h`：25 篇 grouped 骨架（`b_row=group*N+n`，TMA+wgmma+WS）叠 24 篇 per-block 折算（每 128-k 块 `wgmma_wait0` → `fin+=sa*sb*acc`；`BN=128` 时 `sb` 退化成整块标量、`sa` 逐行）。`PERBLOCK` 模板切同二进制 per-block / per-tensor 对照。prefill **806.4 TFLOPS@32k（128×128 s3）/ 790.4@16k / 707.7@8k**，为 per-tensor 的 **82.2%/82.3%/91.0%**；ncu 同二进制：寄存器 90→**155** → occupancy 27.9%→13.9% → tensor pipe **62.4%→46.1%**，主导 stall 由 `long_scoreboard` 12.61 转为 `barrier` 1.16。masked decode 纯带宽、折算**免费**：BM=64 **2.95ms / 2.86 TB/s / 85.4% HBM**。负结果 BM=64（B 重读翻倍）。坑：TMA `boxR` 必须随 `BM` 重建（否则死锁不报错）。含 `moe_pb_S{8192,16384,32768}.out.txt`、`ncu_pb_s3/ncu_pt_s3.out.txt`、`ptxas.txt` |

| `32-fused-norm/` | 32 RMSNorm / QK-Norm 融合 | 模型场景纯访存算子（shape 取自 `/ssd/models/{DeepSeek-V4-Pro,Qwen3-8B,GLM-5.2-FP8}/config.json`）。`norm_fused.cu` 八个变体：`rmsnorm_v0`（朴素两趟 44.2%）→ `rmsnorm_v1`（smem 单读 69.6%，ncu L1TEX 76% 成墙）→ **`rmsnorm_v1r`（整行留寄存器：2869 GB/s / 85.6% HBM，DRAM 82.9%）** → **`add_rmsnorm_v2r`（融合 x+res，5→4 pass：2923 GB/s / 87.2%，DRAM 86.2%）** → `add_rmsnorm_fp8_v3r`（+FP8 e4m3 per-128 动态量化，shared `atomicMax` amax：2205 / 65.8%，SM 50.2% 成瓶颈 → 融合不划算）→ `qk_norm_v4`（一 warp 一 head，shfl 归约：2179 / 65.0%）。坑：本工具链 `float(fp8)` 返回原始位模式（`1.0f→56`），必须走 `__nv_cvt_fp8_to_halfraw`（`fp8_test*.cu` 最小复现）。同轮附带在 `32-mla-tma-v/` 判定 `wgmma` 的 MN-major 免转置不可行（LBO 被忽略，见 TECHNIQUES J4）。含 `norm_fused.out.txt`、`ncu_norm.out.txt` |
| `34-fused-moe-fp8/` | 34 fused MoE + per-block FP8 | DeepSeek-V4-Pro 真实 shape（hidden=7168, inter=3072, 384 experts top-6, e4m3 + ue8m0 + `weight_block=128×128`）。`moe_fp8.cu`：把 31 的 grouped per-block GEMM（TMA+mbarrier+WS+`wgmma` k32+SW128，每 128-k 块 `fin+=sa*sb*acc`）接进 30 的 MoE FFN 端到端，`UNPERM` 模板切「down 是否融 unpermute」。端到端 **11.80ms @M8k bal / 20.58 @M16k / 15.93 @M8k rand**，相对 30 篇 bf16 **1.92×/2.21×/2.06×**；专家权重 50.7→**25.4GB**；K2 down **88.1% HBM**（ncu DRAM 89.3%）、K1 78%（DRAM 80.1%、`long_scoreboard` 占 stall 51.3%、occ 13.9%）。两个负结果：**unpermute 融合在 FP8 下反向**（K2f 3.83 vs 3.35+0.54，ncu L2 **94.4%**/DRAM 73.2%，散写不再被藏住）、**per-block + 双累加器 SwiGLU = 256>255 寄存器**（`BN=256` 崩到 21.7ms）。附带 `swiglu_quant_v2`（一行一 block + `float4` + warp 内 `__shfl_xor` amax）**0.44ms / 3090 GB/s / 92% HBM**（v1 39%）。含 `moe_fp8_M{8192,16384}_{bal,rand}.out.txt`、`ncu_k1/k2/k2f.out.txt` |
| `35-dsa-compressor/` | 35 DSA Compressor（KV 门控池化） | DeepSeek-V4 官方推理 shape/语义（`/ssd/models/DeepSeek-V4-Pro/inference/model.py:279`；`hidden=7168, head_dim=512, rope=64`，`compress_ratios` 在 128/4 交替、`ratio=4` 为 overlap 8-slot，YaRN `compress_rope_theta=160000`）。`compressor.cu` 两个 kernel：**`proj_ws_kernel`**（把 `wkv`/`wgate` 拼成 `[2C,D]` 合并 GEMM，TMA+mbarrier+WS+`wgmma` SW128，`BF16OUT` 模板）**649.8/655.2 TFLOPS @M=32768（65.7/66.3%，ratio=128/4）**，为同 shape cuBLAS 合并 bf16（809/814）的 **~80%**；**`pool_kernel`**（一 block 一窗、一线程一列，online softmax + RMSNorm + YaRN RoPE 三合一）**2790–2812 GB/s / 83–84% HBM**。端到端相对官方 eager **2.00×（ratio128）/2.04×（ratio4）**（M=8192 达 2.39×/2.14×）。ncu：投影 **L2 85–88% 受限**（Compute 57–59%、DRAM 26–29%），根因权重 B 被 `M/BM=128` 个 m-tile 重读；负结果：Y 降 bf16 打平略负。正确性 CPU 全流程 fp64 对拍 err ~0.2%。`compressor_ref.py`（torch eager + cuBLAS 对照）。含 `comp_ratio{128,4}_M{8192,16384,32768}.out.txt`、`sweep_ratio{128,4}_M32768.out.txt`、`torch_ref.out.txt`、`ncu_proj_256x128s4/ncu_proj_r4_256x128s4/ncu_pool_r{128,4}.out.txt` |
| `36-dsa-compressor-mcast/` | 36 compressor 投影 · 压 L2 三招全负 + FP8 化 | shape 同 35（`hidden=7168, head_dim=512, rope=64`，`compress_ratios∈{128,4}`；量化 `e4m3+ue8m0+weight_block 128×128+1×128 dynamic`，取自 `config.json`）。**`compressor_mcast.cu`**：在 35 的 `proj_ws_kernel` 上加 `AXIS`/`CN` 的 **TMA cluster multicast**（leader 单发 + `mapa` 跨 CTA arrive）、**`cudaAccessPolicyWindow` L2 persisting**、K-split 双累加链实验开关。①**翻案 35 的 L2 判决**：最佳几何 `256×128×64 s4` ncu 是 **tensor pipe 76.4% 受限**（L2 仅 58%，35 的 84% 是 `128×128s2` 的 `BM=128` 造成 B 重读 256×）；②A/B multicast（CN=2/4/8：609/597→537/513 vs 基线 632，**−3.7%~−19%**）、persisting（618.7，**−2.2%**）、K-split（ptxas `C7517/C7518` 串行化）**全负**；消费者 `fence.proxy.async` 可删（中性）；③产出 multicast **五条判据**。**`compressor_fp8.cu`**：同进程跑 bf16 与 **FP8 e4m3 per-block** 两条投影路径（复用 24 的 `fp8_tma_pb_kernel`，TMA+wgmma k32+SW128，每 128-k 块 `fin+=sa*sb*acc`）→ **959.1 TFLOPS（ratio128，bf16 的 1.385×）/ 974.1（ratio4，1.394×）**，权重字节减半；ncu 对比 bf16（tensor 76%/L2 58%/stall `long_scoreboard` 9.56）vs FP8（tensor 51%/L2 73%/occ 13.8%/stall `barrier` 1.18）。量化后池化输出最大误差 3.2%/3.8%。含 `mcast_ratio128_M32768.out.txt`、`fp8_ratio{128,4}_M32768.out.txt`、`best_r{128,4}_*.out.txt`、`ncu_bf16_best/ncu_fp8_best.out.txt` |
| `37-fp8-pb-prescale/` | 37 FP8 per-block · scale 折进操作数 | shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`（`e4m3+ue8m0+weight_block 128×128+1×128 dynamic`，M=4096/16384, N=3072, K=7168）。**诊断 fold 的 23% 是「折算时张量核空转」**：每 128-k 块 `wgmma.wait_group 0` 后 64 FMA/线程（roofline 19%），ncu tensor pipe 49.8% vs 67.9%、`barrier` 1.09 vs 0.18。**两条直路撞墙**：`fp8_gemm_pb2.cu` 用 wgmma 自身 `scale_d=0`（DeepGEMM 把 `k` 当 scale_d）替代通用 FMA 清零，消掉 ptxas `C7514`；`fp8_gemm_ws.cu` 加 `setmaxnreg` + ping-pong，但 ptxas 报 `C7507` 忽略 `setmaxnreg`，仍 348（基线 940）。**第三条路 `fp8_gemm_prescale.cu`**：`ue8m0` 是 2 的幂 → `q·sa` 只是 e4m3 指数平移、精确 → per-block GEMM 退化成 per-tensor GEMM，**940.6 → 1207.9 TFLOPS（1.28×）**；融进上游 per-1×128 动态量化器（写回前多乘一次）后端到端 **0.2617→0.2194 ms（1.19×）**、M=16384 **1.188×**；折后与 fold 相对误差 **0.15%**。含 `prescale_M{4096,16384}.out.txt`、`overlap_negative.out.txt`、`scale_d_zeroing.out.txt`、`ncu_fold/out.txt`、`ncu_pt.out.txt` |

## 怎么跑

宿主机上直接调用脚本即可（自动进 `kernel_lab` 容器编译运行）：

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up
scripts/run.sh 02-first-kernel/vector_add.cu
scripts/ncu.sh 02-first-kernel/vector_add.cu --set full --kernel-name regex:add
```

环境变量：`ARCH`（默认 `sm_90`；**设为空串 `ARCH=""` 则不传 `-arch`**，用 `NVCC_FLAGS` 里的 `-gencode` —— CUDA 13 的 nvcc 不认 `-arch=sm_90a`，wgmma 需 `NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a"`）、`GPU`（默认 0）、`NVCC_FLAGS`。

## 无人值守自驱（autopilot）

`scripts/autopilot.sh` 会**反复启动 opencode 无头会话**（`opencode run --auto`），每轮推进
`ROADMAP.md` 里的一篇文章增量（写代码 → 实测 → ncu → 写文 → 发布 → 更新路线图），
适合离开电脑后持续工作：

```bash
scripts/autopilot.sh start     # 后台启动（setsid 脱离终端，关终端也不停）
scripts/autopilot.sh status    # 查看状态 + 最近日志
scripts/autopilot.sh stop      # 停止
scripts/autopilot.sh run       # 前台运行（调试）
```

- 日志：`code/kernel-opt/autopilot.log`；停止文件：`AUTOPILOT_STOP`（**只有用户 `autopilot.sh stop` 才会创建**；agent 永不自行创建，做完就自己扩充路线图继续）。
- 可调：`MAX_ROUNDS`（默认 40）、`SLEEP_BETWEEN`（默认 30s）、`TIMEOUT_PER_ROUND`（默认 5400s）。
- 连续失败 3 次会自行停止；`autopilot.lock` 防止重复启动。

## 约定

- 每个 `.cu` 都能独立编译运行，自带参考实现对拍与计时输出。
- 实测原始输出存到同目录 `*.out.txt`，供写文章引用。
- 公共代码只放 `common/`，不放实验目录。
