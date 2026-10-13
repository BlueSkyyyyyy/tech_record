# DeepSeek Mega MoE 前向 / 反向算子实现解析 · 路线图（活文档）

> 目标：以 **DeepGEMM** / **DeepEP** 为参考，把 DeepSeek 的 **Mega MoE**（dispatch + linear1 + SwiGLU +
> linear2 + combine 融合的单 mega-kernel、通信与计算重叠）以及 MoE 的**反向**（wgrad/dgrad grouped GEMM）
> 逐层讲透。本专题为长期自驱任务，由 `scripts/autopilot.sh` 驱动（与仓库内其它自驱任务并行）。
>
> **重要约束**：Mega MoE 主实现是 **SM100（Blackwell）**，本机 H100（sm90）跑不了端到端。
> 因此以**源码精读 + 线上资料**为主线，数值实验只做 H100 上可行的部分（grouped GEMM / backward / 通信原语）。

## 交付物（Definition of Deliverable）

1. **分析文档**（`docs/`，内联到博客）
   - MoE / EP 基础；Mega MoE 的整体数据流与「通信-计算重叠」机制。
   - `sm100_fp8_fp4_mega_moe.cuh` 主体 kernel 的逐段精读（warp specialization、symmetric buffer、
     ring buffer / task 调度、TMA、tcgen05 MMA、epilogue）。
   - MoE **反向**：wgrad（`k_grouped_*`）与 dgrad（`m_grouped_*`）的实现解析，以及 gate/router 反向。
   - 相关 mega-kernel：Mega Gate、Mega mHC。
2. **可运行实验**（`src/`、`harness/`）
   - H100 上可跑的 grouped GEMM / backward 数值实验（对照 PyTorch 参考），报误差与性能。
   - 通讯原语（DeepEP）在 H100 上的可行实验（如环境允许）。
3. **实测证据**：能跑的部分给真实数字与原始输出；跑不了的部分明确标注「SM100 专属，仅分析」。

## 环境 / 参考坐标

- 容器 `kernel_lab`（`scripts/lab.sh`），GPU H100 sm90。Mega MoE 端到端需 SM100（本机不可跑）。
- `~/github/DeepGEMM`：Mega MoE、grouped GEMM、backward wgrad；`~/github/DeepEP`：通信。
- 线上资料：DeepSeek 官方博客 / GitHub PR（如 `#316` Mega MoE、`#95` MoE backward）。
- **目标卡代号敏感**：一律写「目标卡」。

## 任务清单

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。

### P0 侦察与基础

- [x] **P0-1** 读通 DeepGEMM：`csrc/apis/mega_moe.hpp`、`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、
      `scheduler/mega_moe.cuh`、`layout/mega_moe.cuh`、`tests/test_mega_moe.py`，产出数据流脑图。
      → `docs/02-mega-moe-architecture.md`（commit `057ca59` 行号核实）。
- [x] **P0-2** 补充线上资料：DeepGEMM README/release notes 中 Mega MoE 段落、PR #316；记录引用链接。
      → `docs/01` §5（版本时间线与链接）、`docs/02` §8（PR #316 基准表与解读）。
- [x] **P0-3** 访问 `~/github/DeepEP`：dispatch/combine kernel 与 symmetric memory 的接口与数据流。
      → `docs/07-deepep-dispatch-combine.md`（真实行号核对，DeepEP commit `93eb6eb` / v2.5.0）。
- [x] **P0-4** 环境探测：`kernel_lab` 里能否编译/跑 DeepGEMM 的 SM90 部分（`pip install -e ~/github/DeepGEMM`，
      注意 `DG_SKIP_CUDA_BUILD`）；确认 H100 上可用哪些 kernel。已知：CUDA 13.2 / PyTorch 2.12 达标。
      → `docs/08-h100-environment.md`（真机数字）；实验 `harness/probe_h100_env.py`、`harness/probe_deepep.py`。
      结论：dense/grouped FP8、BF16、**SM90 版 MoE wgrad（`k_grouped_*_nt_contiguous`）** 均可跑；
      `k_grouped_*_tn_contiguous` 与 `fp8_fp4_mega_moe` SM100 专属（`gemm.hpp:369` 拒绝）。DeepEP 可装可 import
      但运行期受阻（NCCL 2.29.7 vs 需求 ≥2.31），见「阻塞」。

### P1 MoE / EP 基础（`docs/01-moe-ep-basics.md`）

- [x] **P1-1** MoE 计算图：router → topk → grouped GEMM（linear1/SwiGLU/linear2）→ combine；
      解释 grouped contiguous vs masked 两种布局。→ `docs/01` §1（含真实行号：`gemm.hpp:181/209-246/269-318`、
      `runtime.hpp:57-64`、README `76-86`），并补 **PSUM（前缀和）第三布局**；H100 实测三种布局数值等价
      （`harness/exp_grouped_layout.py` + `.out.txt`）。
- [x] **P1-2** EP（专家并行）通信：all-to-all dispatch / combine 的语义与开销；为何要与计算重叠。
      → `docs/01` §2 重写：通信量模型（dispatch ~$T{\cdot}d{\cdot}H{\cdot}1$B、combine ~$T{\cdot}k{\cdot}H{\cdot}2$B，
      K 倍放大）+ 串行 vs 重叠量化 + **H100 实测**（`harness/exp_ep_comm.py`，8×H100 NCCL
      all-to-all：渐近 ~393 GB/s、固定时延 ~50–90 us，达 NVLink 硬件上限 ~82%）。

### P2 Mega MoE 架构（`docs/02-mega-moe-architecture.md`）

- [x] **P2-1** 整体设计：单 kernel 融合 5 段；symmetric buffer、`transform_weights_for_mega_moe` 布局。
      → `docs/02` §2-§3（数据流脑图 + `MegaMoEBuffer` 布局）+ §3.4（权重布局：gate/up 8 粒度交错、
      SF 的 UTCCP 4x32 转置、SM100 packed UE8M0 + MN-major 索引），含纯 PyTorch 重排核对
      `harness/exp_weight_transform.py`（0 误差）。
- [x] **P2-2** 「通信-计算重叠」机制：ring buffer / pool blocks / task 调度（`scheduler/mega_moe.cuh`）、
      dispatch warp 与 MMA/epilogue warp 的分工。→ `docs/02` §4-§5。
- [x] **P2-3** 与「先通信后计算」「chunked 分步」方案的对比（为什么更快）。→ `docs/02` §7。

### P3 主体 kernel 精读（`docs/03-mega-moe-kernel.md`）

- [x] **P3-1** 模板参数与 block config（`get_block_config_for_mega_moe`）：BLOCK_M/N/K、cluster、stage、
      thread 划分（dispatch / non-epilogue / epilogue）。
      → `docs/03-mega-moe-kernel.md`（真实行号核对）；H100 对拍 `harness/exp_block_config.py` / `.out.txt`。
- [x] **P3-2** warp specialization 与同步（mbarrier）；TMA 载入激活/权重/scale。
      → `docs/09-mega-moe-mbarrier.md`（真实行号核对）：五类 warp 两层解耦、全 mbarrier 清单与 init 计数、
      `advance_pipeline` 相位翻转 / 生产者 phase=1 约定、A/B loader 坐标与 2-CTA multicast、task 槽双缓冲
      与「仅 epilogue release」、跨 SM/rank barrier、L1→L2 位图依赖；H100 真机验证相位约定 +
      NSTAGES=1/4 流水重叠（`src/mbarrier_pipeline.cu`、`harness/exp_mbarrier_pipeline.{sh,out.txt}`）。
- [x] **P3-3** linear1 → SwiGLU → linear2 的 MMA 与 epilogue（tcgen05、UE8M0 scale、FP8xFP4）。
      → `docs/10-mega-moe-mma-epilogue.md`（真实行号核对）：leader CTA 单 warp 发 UMMA、A/B 互换
      与 2-CTA（`UMMA_M=256/UMMA_N=BLOCK_M`）、动态缩 N（`valid_m` 16 对齐）、routed/shared 双
      instruction descriptor、UTCCP 把 SF 从 smem 搬进 TMEM（`kernel:894-907`）、UMMA
      `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`（`tcgen05.cuh:67-86`）与 accumulate 首条清零、FP4 打包因子
      （`packing.cuh:8-12`）、commit multicast 通知 empty/tmem_full；epilogue TMEM 双缓冲 + 提前
      `release_task_info`/`release_tmem`、L1 in-place SwiGLU（bf16 round）+ UE8M0 量化（
      `math.cuh:114-147`）+ TMA 转置上屏 + L2 K-block bitmask 依赖、L2 BF16 NVLink 直写远端
      combine buffer、combine top-k fp32 归约。H100 数值实验 `harness/exp_ue8m0_sf.py`（+`.out.txt`）：
      位运算 SF 25009 样本 0 不符、scale×inv 精确 1、幂次 scale/理想 ≤1.9977 且 0 溢出、bf16-round
      SwiGLU 相对误差 4.5e-3。配套博客 `content/posts/mega-moe-07-mma-epilogue`。
- [x] **P3-4** dispatch / combine 段与 NVLink 读写；locality domain 的作用。
      → `docs/11-mega-moe-dispatch-combine.md`（真实行号核对）：dispatch（`:342-689`）分布式前缀和
      计数 + 写远端源索引 + NVLink barrier + 无锁 round-robin 认领（`:480-528`）+ 分 chunk TMA
      远端 pull（`:552-576`）与远端 weight/SF（`:579-599`）+ 清 workspace；combine L2 直写远端
      （`:1346-1351`）+ grid-index readiness 水位线（`:1405-1441`）+ fp32 chunk 归约（`:1447-1519`）；
      `SymBuffer::map` / TMA / sys 原子 / rel-acq / `grid_sync` / `nvlink_barrier` 原语表；
      locality domain 分域调度（`scheduler:344-370`）与片上 pointer-chase 探针标定
      （`csrc/apis/locality_domain.hpp`）。H100 实测 `src/nvlink_access.cu` +
      `harness/exp_nvlink_access.{sh,out.txt}`：远端裸延迟 ~991ns ≈ 本地 ~350ns 的 **2.83×**、
      远端读/写带宽 ~368 GB/s ≈ 本地 ~2.85 TB/s 的 **1/8**，同 socket/跨 socket 无差异（NV18 平面）。

### P4 反向实现解析（`docs/04-moe-backward.md`）

- [x] **P4-1** MoE 反向数学：linear2/linear1 的 dgrad 与 wgrad，SwiGLU 反向，router/topk 反向。
      → `docs/04-moe-backward.md`：逐层反向推导 (1)–(8) + 「wgrad=k_grouped / dgrad=m_grouped」形状角色表
      （`gemm.hpp:320-371`/`:55-76`/`:340-349`、`generators.py:583-587`、README `:80`）；H100 float64 对拍
      `harness/exp_moe_backward_math.py`（+`.out.txt`）：手写反向 vs autograd 四项梯度 relerr ≤2.9e-16、
      中心差分 ≤5.7e-8，k_grouped 逐组 `A_e^T@B_e` 复现 wgrad relerr ≤1.3e-16。配套博客
      `content/posts/mega-moe-09-backward`。
- [x] **P4-2** DeepGEMM 的 wgrad API `k_grouped_fp8_gemm_tn_contiguous`（K 轴分组、M/N 固定）精读。
      → `docs/12-moe-backward-wgrad.md`（真实行号核对）：`tn`（SM100/MN-major `[sum_k,M]`，仅分析）
      与 `nt`（SM90/K-major 每组 `[M,k_g]`）同一数学 `D[e]=A_e^T B_e`；`ks_cpu`/`grouped_layout`
      （contiguous=物理 K / PSUM=逻辑 K 结束，PSUM 仅 SM100）、`check_k_grouped_args`（`gemm.hpp:55-76`）、
      SM90 按组原地改写 tensormap（`sm90_fp8_gemm_1d1d.cuh:193-215` + `tensor_map_buffer`）、
      SF 逐组打包 UE8M0 4→int32（`smxx_layout.hpp:218-283`）、调度器 K 分组切块
      （`scheduler/gemm.cuh:71-82/226-246`）。H100 实测 `harness/exp_kgrouped_api.py`（+`.out.txt`）：
      nt 4 组 relerr 1.98e-2（FP8 噪声）、空组 OK、未对齐 ks 被拒、tn 在 sm90 被拒、
      `m=4096,n=7168`（8 组）**425 TFLOPS**；纯 torch 布局等价 tn 0 误差 / nt 3.3e-8。
- [x] **P4-3** dgrad（`m_grouped_*`）与 contiguous/masked 布局在训练中的用法。
      → `docs/13-moe-backward-dgrad.md`（真实行号核对）：dgrad 天然是 **M 轴分组**（不等长=token_数 落在 M，
      N/K 固定，`gemm.hpp:194/280`）；两个 host 入口 `m_grouped_fp8_fp4_gemm_nt_contiguous`（`:181-250`，
      contiguous=长度 m 逐行组号 / PSUM=长度 G 前缀和结束，`:213-220`）与 `..._nt_masked`（`:269-318`，
      `[G,max_m,·]` 稠密 + device 端 `masked_m`）；**PSUM 仅作用于 SFA**（`layout.hpp:69`、`gemm.hpp:230`）。
      调度器三 `GemmType`（`scheduler/gemm.cuh:92-105/188-225`），`get_global_idx`/`is_computation_valid`
      统一取址与跳过 gap（`:151-156/296-309`）；**段必须对齐 BLOCK_M 的 kernel 侧原因**：contiguous 用
      `grouped_layout[m_block_idx*BLOCK_M]`（M block 首行组号）代表整块（`:152`），SM90 TMA multicast
      peer 一致性也依赖它（`:287-289`）。SM90 1d2d kernel：A/D/SFA 在 masked 下是 3D（组号=第 3 维）、
      contiguous 下 2D（`sm90_fp8_gemm_1d2d.hpp:178-194/245-261`），`kWithGroupOffsetA=masked`
      （kernel `:189-194`）；SFB 逐组基址（`:240`）。**训练分工**：前向/dgrad 用 contiguous、解码 CUDA
      graph 用 masked、DeepEP 动态输出用 PSUM 桥接。H100 实测 `harness/exp_mgrouped_dgrad.py`（+`.out.txt`）：
      contiguous/masked/PSUM relerr 3.67–3.70e-2（FP8 噪声）、A4 布局长度不符被拒（`gemm.hpp:218`）、
      `G=8,m=4352,n=2048,k=7168` **366 TFLOPS**；三种布局两两 **逐位相等（0 误差）**。配套博客
      `content/posts/mega-moe-11-dgrad`。
- [x] **P4-4** 在 H100 上做 grouped GEMM / backward 的数值对拍实验（vs PyTorch）。
      → `docs/14-moe-backward-kernels.md`：把 SM90 `k_grouped_*_nt_contiguous`（wgrad）与
      `m_grouped_*_nt_contiguous`（dgrad）接进真实 MoE 反向图。H100 实测
      `harness/exp_moe_backward_kernels.py`（+`.out.txt`）：四项梯度 `vs deq`（隔离 kernel+布局）
      relerr ~2.6e-4、`vs autograd`（含 FP8）2.5e-2~5.1e-2 且 cos>0.999；吞吐单 rank
      （T=2048,H=7168,I=2048,E=8,topk=2）wgrad ~103/106 TFLOPS、dgrad ~449/543 TFLOPS。
      配套博客 `content/posts/mega-moe-12-backward-kernels`。

### P5 相关 mega-kernel（`docs/05-mega-gate-mhc.md`）

- [x] **P5-1** Mega Gate（`mega_gate.hpp` / `sm100_bf16_mega_gate.cuh`）：做什么融合、怎么实现。
      → `docs/05-mega-gate-mhc.md`（真实行号核对）：把「BF16 打分 GEMM + softplus/sqrt + bias 排名 +
      top-k + logical→physical 映射」压进单 kernel。**三个非平凡点**：① UMMA 累加器写 TMEM、
      gate warp 用 `tmem_load_32dp32b` 读回（`epilogue:39-78`、`impls:279-303`），split-K 时先落
      global score scratch 再用 `ScoreBarrier` 跨 CTA 汇聚（`scheduler:13-39`）；② **带 bias 排名、
      不带 bias 出权重**（`impls:379-399` + `epilogue:187-216`）；③ **warp 内排序网络 + 3-bit 游标
      归并**的确定性 top-k（`epilogue:88-152`，破平取最小专家号，`redux.sync.max`+`__reduce_min_sync`）。
      任务分解 `num_logical_ctas=expert_groups*cta*splitk`（`heuristics:21,106-187`）；`sqrt_softplus`
      阈值 20 分支（`epilogue:19-23`）；重复专家分散 `(rank+t*23333)%d`（`epilogue:205`）；shared-as-routed、
      mask/fix_routing/force_random/image_bias。**SM100 专属，仅分析**（`mega_gate.hpp:158-159`）。
      H100 实测 `src/mega_gate_routing.cu` + `harness/exp_mega_gate_routing.{sh,out.txt}`：把架构无关
      路由语义逐行移植——`sqrt_softplus` vs double relerr 1.4e-7、阈值分支连续；5 个案例
      （含重平局/3-wave/4-wave padded）共 ~7 万条 top-k **0 不符**、权重 relerr ≤1.5e-7；
      重复专家分散 d∈{1,2,4,32} 下 **max/min=1.0000** 完全均衡。配套博客
      `content/posts/mega-moe-13-mega-gate`。
- [x] **P5-2** Mega mHC（`mega_mhc.hpp` / `sm100_mega_mhc.cuh`）：HyperConnection 相关。
      → `docs/05-mega-gate-mhc.md` 第二部分（真实行号核对）：残差多流混合 `new_residual = post*x + comb@residual`
      （bf16 舍入为共享边界，`epilogue:496-498`）→ mix 投影 `fn`（唯一窄输出长 K GEMM）→
      sigmoid pre/post + **Sinkhorn 双随机** comb → RMSNorm（+FP8）。**三个非平凡点**：① UMMA 走
      **TF32 TS**（A=残差在 TMEM、B=`fn` 在 SMEM，`impls:235-275`），因系数投影需高于 bf16 的尾数；
      ② Sinkhorn 用 warp 内 `shfl_xor` 逐行/逐列归约（`epilogue:63-86`，行和列和→1）；
      ③ 跨 CTA `SplitBarrier` 用「grid 序号+到达数」一次编码、免复位（`scheduler:13-49`，Norm/Mix 两个）。
      6 warpgroup 静态角色 + split-K scratch（`layout:73-120`）；normal 把 Mix+Pre+RMSNorm 融成
      per-token worker（`epilogue:139-281`）、shifted 两 token 一 task（`:291-390`）；三种 SF 布局
      （gemm col / routed row / **shared 页**，`get_shared_sf_row_idx` `epilogue:117-122`）。**SM100 专属，仅分析**
      （`mega_mhc.hpp:223`）。H100 实测 `src/mega_mhc_semantics.cu`（+`harness/exp_mega_mhc_semantics.{sh,out.txt}`）：
      残差/RMSNorm 到 bf16 ulp、mix 系数 ~2e-7、Sinkhorn 双随机到 2e-6、SF 页布局 0 不符、
      SF 打包 0/512 不符、E4M3 量化 **0.5 ulp 最优**。配套博客 `content/posts/mega-moe-13-mega-gate`。

### P6 汇总

- [x] **P6-1** `docs/06-summary.md`：数据流图、关键机制、与朴素实现的开销对比、反向要点一页纸。
      → `docs/06-summary.md`（专题总览）：前向数据流图（dispatch→L1→SwiGLU→L2→combine 单 kernel）、
      关键机制坐标表、串行/chunked/Mega 三方案开销对比（通信 K 倍放大 + 官方 PR #316 解读）、
      主体 kernel 四篇索引、反向一页纸（四 GEMM + 路由坑 + 三布局）、Mega Gate/mHC 表、
      **H100 实测证据台账**（13 项真机数字汇总）、线上资料链接、结论与边界。
      配套博客 `content/posts/mega-moe-14-summary`（索引页同步更新：13 篇并入 mHC、新增总结篇）。

### P7 增量（收尾后 opt-in）

- [x] **P7-1** H100 侧反向性能归因：用 `ncu` 解释 P4-4 量到的 wgrad ~100 vs dgrad ~450 TFLOPS。
      → `docs/15-moe-backward-profiling.md`：单 kernel/进程 + `ncu --kernel-name regex:sm90_fp8_gemm`
      采 SpeedOfLight 等五 section + 定向 metric（时长/DRAM 读写字节/GMMA 指令/SM 周期）。**硬指标**
      （单次 launch，热 cache）：w2g Duration 499.9us、计算(SM) 12.4%、DRAM 读 530.7MB/写 460.1MB；
      w1g 998.8us/12.2%/1.09GB/932.5MB；d2g 111.0us/tensor 65.5%/180.9MB/25.0MB；d1g 194.3us/75.7%/
      293.2MB/64.2MB。**归因**：算术强度 wgrad ~120（w2g 121/w1g 119）< H100 FP8 roofline 拐点 591
      → **访存受限**；dgrad ~584/673 → **计算受限**。三条根因：① 权重梯度输出大且 fp32
      （w2g 470MB / w1g 940MB；dgrad 输出仅 16/59MB bf16）；② SM90 `k_grouped_nt` 强制累积
      （`gemm.hpp:403` C 必填，D=C+AᵀB），D 被回读使流量翻倍（读≈写+操作数，坐实）；③ 收缩维 K=token
      仅 512（4 个 BLOCK_K），GMMA 数 w2g 786k≈d2g 631k 但慢 4.5×。四个 kernel 都达各自 roofline 下界
      55–63%，**下界之比 ~5×**（wgrad 访存下界 296/604us vs dgrad 计算下界 61/122us）与实测吻合。
      另记录单次 vs 持续（降频 ≈2×，比值稳定）。H100 实测 `harness/exp_backward_ncu.{py,sh,out.txt}`。
      配套博客 `content/posts/mega-moe-15-backward-profiling`（索引页新增第 15 篇）。
- [x] **P7-2** HyperConnection prenorm GEMM（`tf32_hc_prenorm_gemm`）精读 + H100 实测：把 mHC 的
      「mix 系数投影」与「RMSNorm 平方和」融进一个 TF32 kernel（**有 SM90 实现，本机可跑**）。
      → `docs/16-hc-prenorm-gemm.md`（真实行号核对）：host 入口 `hyperconnection.hpp:10-55`
      （dtype/major 契约、arch9→sm90/arch10→sm100 派发 `:47-54`），SM90 host 契约
      `sm90_tf32_hc_prenorm_gemm.hpp:21-31`（block_m=64/block_k=64、2 warpgroup、`n<=32 and n%8==0`、
      `k%64==0`），SM90 kernel `sm90_tf32_hc_prenorm_gemm.cuh`（`:107-115` split-K 网格；`:123-146`
      TMA warp；`:179-203` **A 搬运时 bf16→fp32 顺路累加平方 = 融合支点**；`:205-231` TF32 WGMMA
      A 在寄存器/B 在 smem；`:233-284` 归约+D 上屏）；SM100 分支**仅分析**（第三 warpgroup 把 A 转
      fp32 写 TMEM、UMMA 走 `SM100_MMA_TF32_TS`，`sm100_...cuh:169-173/268-342`）。H100 实测
      `harness/exp_hc_prenorm_gemm.{py,sh,out.txt}`：① **TF32 对 fp32 操作数是 RNE，不是截断**
      （D.vs_RNE 比 D.vs_trunc 小 ~30×，如 (4096,24,7168) 1.665e-5 vs 4.982e-4）；② S 与 fp64 精确和
      rel 8.3e-7（bf16 平方在 fp32 可精确表示）；③ 融合 kernel 166.7us vs torch「单独 matmul+sqr_sum」
      3683.6us（~22×），且独立 sqr_sum 的**理想单遍 A 读**也要 ~165us；④ split-K=16 可加性
      nerr 1.59e-5、小 M（137）从 1.5→5.4 TFLOPS；⑤ 契约 n>32 / k%64 被拒（hpp:30/31）；
      ⑥ 性能大 M 达 2848 GB/s ≈ HBM 峰值 85%、AI≈17–31 → **访存受限**。配套博客
      `content/posts/mega-moe-16-hc-prenorm-gemm`（索引页新增第 16 篇）。

### P8 增量（opt-in，扫 DeepGEMM 其余未覆盖算子）

- [x] **P8-1** einsum `bmk,bnk->mn` 批归约 GEMM（跨 batch/rank 的 wgrad 融合，**有 SM90 实现，本机可跑**）。
      → `docs/17-einsum-batch-reduced-gemm.md`（真实行号核对）：DeepGEMM 硬编码 einsum 家族总览
      （`einsum.hpp:166-178` 四分派、`:22-58` bmk 入口、`:147` FP8 跨 batch wgrad 仅 cuBLASLt）；
      `bmk_bnk_mn` 契约（fp32 `c==d` `:25-26` / bf16 走 fp32 workspace `:28-38` / `m,n,k` 取自 a,b
      **d 形状不校验**）；SM90 host 调度 `sm90_bmk_bnk_mn.hpp`（block 128x128x64 `:19-25`、
      split_factor `:32-35`、流水级数 `:38-53`、2D `[s*m,k]` TMA 描述符 `:64-65`）；SM90 kernel
      `sm90_bmk_bnk_mn.cuh`（warp 角色/寄存器再平衡 `:84-85`、block 索引+`:94` 尾分片、
      `:97` **PDL** `cudaGridDependencySynchronize`、TMA 生产者 `s_idx*SHAPE` 折算 batch `:104-124`、
      WGMMA `:130-158`、epilogue **`atomicAdd(float2)` 归约到 D** `:160-174`）；SM100 变体**仅分析**
      （`:255` **TMA reduce-add** 取代 atomicAdd、tcgen05/TMEM `:131-183/245-248`、单 warpgroup）。
      H100 实测 `harness/exp_einsum_bmk_bnk.{py,sh,out.txt}`：① 语义 vs `bmm+sum` FP32 3.9e-7~7.7e-5、
      BF16 ~3e-3；② **坐实「batch 并入收缩轴」**（vs `A.permute(1,0,2).reshape(m,s*k)@...` 1e-6~2e-4）
      与**「部分 wgrad 再求和」**（s=8 rank 模型 2.8e-7、省 8x 显存）；③ **非确定性**：split=993 时
      20 次运行 6.5 万个输出里 ~5.2 万 bit 级抖动、幅度 4e-7（对照 torch.sum 恒 0）；④ 性能
      比 torch 朴素快 **7~18x**、`m=n=1024` 达 **314 TFLOPS**、`m=n=256` 因 num_mn=4 原子竞争仅
      165 TFLOPS、GB/s 616~1599；⑤ 契约 k/m/n%64 被拒（hpp:24/25）、`d` 形状不校验；⑥ PDL on/off
      语义一致、耗时噪声级（`:97` 太早同步）。配套博客 `content/posts/mega-moe-17-einsum-batch-reduced-gemm`。
- [x] **P8-2** einsum 家族投影分支 `bhr,hdr->bhd` / `bhd,hdr->bhr`（**按头 batched GEMM，有 SM90
      实现，本机可跑**）：MLA 吸收投影 `A[:,h,:]@B[h]`。→ `docs/18-einsum-mla-batched-gemm.md`
      （真实行号核对）：host 分派与契约 `einsum.hpp:60-92/94-126`（三个 `stride(2)==1`、cublaslt/
      sm90/sm100 三分支），降维映射 `sm90_bf16_gemm.hpp:332-384/386-438`（`GemmType::Batched`、
      `(m,n,k)=(b,d,r)/(b,r,d)`、`num_groups=h`、major `KK`/`K·MN`），3D TMA
      `runtime_utils.hpp:161-202`（头=第三维 + 两个显式 stride）+ kernel `batch_idx`
      （`sm90_bf16_gemm.cuh:186-199`）+ store `SM90_TMA_STORE_3D/REDUCE_ADD_3D`（`:374-379`）+
      调度器 Batched（`scheduler/gemm.cuh:90/169-173/247-262`）+ 禁 multicast
      （`heuristics/sm90.hpp:63-67`）；SM100 变体与 FP8 变体（`bhd,hdr->bhr` / `bhd,bhr->hdr`
      arch10）**仅分析**。H100 实测 `harness/exp_einsum_mla.{py,sh,out.txt}`：16 组形状 bf16 输出与
      `torch.einsum` **逐位相等**（relerr fp64 2.0–3.5e-3）；头独立性（置零头 0 不影响他头）；
      `D=C+A@B`（TMA reduce-add）relerr 5.3e-3；**带 padding 权重 `fy[:,:,:r]` 与连续权重逐位相等**；
      确定性开关实测切 kernel（True→`sm90_bf16_gemm_impl<…GemmType)4…>`、False→`nvjet_sm90_*`）；
      契约非法输入被拒；性能与 torch 同速（0.84–1.09×），`b=128 h=128` 212 / `h=8 r=4096 d=1024 b=4096`
      **360 TFLOPS**。配套博客 `content/posts/mega-moe-18-einsum-mla-batched-gemm`。

### P9 增量（opt-in，DeepSeek 稀疏注意力 indexer）

- [x] **P9-1** DeepSeek 稀疏注意力 indexer 的 **MQA logits**（`fp8_fp4_mqa_logits` 连续 /
      `fp8_fp4_paged_mqa_logits` 分页；**SM90 有实现，H100 可跑**）：加权 ReLU MQA 打分。
      → `docs/19-attention-mqa-logits.md`（真实行号核对）：语义（README `:100-108`）——
      `out_ij = Σ_h relu(<q[i,h,:], kv_j>)·w[i,h]`、KV 每 token 一个 fp32 scale、输出压缩存有效区间。
      host `attention.hpp`：共享检查 `check_mqa_logits_q_and_weights :18-44`（`is_mx_sf==arch10 :32`、
      weights 行 16B 对齐 `:42`、dtype `:43`）；连续 `:67-134`（`out_dtype` `:80`、`kv_sf` dtype `:92-96`、
      `block_q=128/heads :110`、stride 1024B 对齐 `:114-117`、SM90 分派 `:124-129`）；分页 `:174-282`
      （融合 KV 布局 `:200-207`、`from_blob` 切 KV/SF `:212-223`、`block_kv==64`、SM90 无 varlen）。
      **WGMMA M 钉死 128**（`block_q*heads==128`，kernel `:44`）。SM90 连续 kernel `:22-321`：
      warp 分工/barrier 计数 1:512、寄存器 32/112、persistent 调度 + 有效区间并集（`start/4*4 :139`）、
      PDL `:155`、**Q 在 B 侧 / KV 在 A 侧**的 WGMMA `:255-263`、ReLU+权重+KV scale 的两级 warp 归约
      （intra `:282-293` + `shfl_xor` `:298-303`）、压缩偏移写回（`- seq_k_start :306-310`）。SM90 分页
      kernel：`split_kv=256`/`mma_m=64` → 4 math WG 各管 64 KV、`not varlen :39`、每 KV 组独立 TMA 流水
      `:92-106`、block table 用 `shfl` 广播 `:191-198`、3D TMA 定位物理块 `:205-211`。SM100（MXFP4/MXFP8、
      varlen、`schedule_meta`、sparse top-k 块）**仅分析**。H100 实测
      `harness/exp_mqa_logits.{py,sh,out.txt}`：连续 `vs_sim`（隔离 FP8 量化）~2e-3、`vs_ref` 3–5e-2、
      `-inf` 掩码逐位一致、20 次重复逐位相等；性能 H=64,D=128 达 **497 TFLOPS**（D=64 掉到 316）；
      分页 3 组 `vs_sim` ~2e-3、230–370 TFLOPS；6 项契约非法输入全部按预期被拒（行号见 `attention.hpp`）。
      配套博客 `content/posts/mega-moe-19-attention-mqa-logits`（索引页新增第 19 篇）。

### P10 增量（opt-in，Mega MoE 的 BF16 变体）

- [x] **P10-1** **BF16 Mega MoE**（`bf16_mega_moe` / `sm100_bf16_mega_moe.cuh`）：Mega MoE 的
      第二个主 kernel——同一套骨架（dispatch→L1→SwiGLU→L2→combine、ring、通信-计算重叠）
      换 `MmaKind::BF16`，去掉整条 block-scale 通路。**SM100 专属，仅分析**（`mega_moe.hpp:367`
      只在 arch10 派发；`kernel:68` 整段 `#if __CUDA_ARCH__>=1000`）。
      → `docs/20-bf16-mega-moe.md`（真实行号核对）。**要点**：host 入口 `mega_moe.hpp:285-369`
      （只收 BF16 权重 `:313-315`、无 `recipe`、`get_symm_buffer_size(..., "bf16xbf16", ...)` `:352`、
      `with_sf=false` → 切片 `x_sf=None`）；启发式 `heuristics/mega_moe.hpp:73-86` 的 `MmaKind`
      分流、`:88-127` 的 `block_k=128/elem_bytes`（**BF16=64、FP8=128**）与
      `store_block_m_l1` 240 行取 40（FP8/FP4 取 24）、`:223-229` 的 `num_bytes_per_pull`
      （BF16 cap 4096B）。kernel `sm100_bf16_mega_moe.cuh`：模板 `:21-51`、`with_sf=false` `:111`、
      `a/b/d=bfloat16` `:119-121`、TMEM 只放累加器 `:158-160`、**唯一 UMMA**
      `SM100_MMA_F16BF16_2x1SM_SS` `:795`（无 UTCCP/SF）、`UMMA_BLOCK_K/UMMA_K=64/16` `:128-129`；
      L1 epilogue in-place SwiGLU `:883-1027`（gate/up 先 round bf16 `:949-950`、clamp `:953-957`、
      `kFastMath` 用 `__expf`+`fast_rcp` `:961-969`、输出 bf16 `:971-972`）；L2 epilogue bf16 直写
      远端 `:1133-1138`（`cast_into_bf16_and_pack`=RNE）；combine `:1152-1308` 与 FP8 版逐行复用。
      H100 实测 `harness/exp_bf16_mega_moe.{py,sh,out.txt}` + `src/bf16_mega_moe_epilogue.cu`：
      ① `block_m` 13 组形状 BF16/FP8 **全一致**、仅 `block_k` 64/128 分化；② symm buffer
      BF16 2713.5MB vs FP8 2145.3MB（**1.265×**）、`x_sf=None`；③ `num_bytes_per_pull`
      7168→3584/3584；④ 权重重排 diff **0.0**、l2 同对象；⑤ L1 SwiGLU「bf16 in→fp32 silu→
      bf16 out」vs fp64 rel 3.89e-3（输出 1 ulp），fast-math 叠加 7.0e-3，clamp 生效；
      ⑥ L2 cast 2.1M 对 **vs RNE 不符 0**、vs 截断 1.57M。配套博客
      `content/posts/mega-moe-20-bf16-mega-moe`（索引页新增第 20 篇）。

### P11 增量（opt-in，einsum 家族的 FP8 投影）

- [x] **P11-1** **FP8 einsum MLA 投影 `bhr,hdr->bhd`**（`fp8_einsum` / SM90 `sm90_fp8_bmm`；
      **有 SM90 实现，H100 可跑**）：BF16 MLA 投影（P8-2）的 block-scale 版本。
      → `docs/21-fp8-einsum-projection.md`（真实行号核对）：host `fp8_einsum:251-298`
      把 `(b,h)` **permute 成 `(h,b,d,r)`**（`h`=batch_size、`b`=m，`:262-276`），
      只有 `bhr,hdr->bhd` 收 FP8 `(d,sfd)` 输出对（`:259-260`）；`fp8_bmm:181-249` 的形状/类型
      契约（`:203/214/216-217`）与 FP8 输出对硬约束（`:224` arch10 / `:225` 批在列连续 / `:226`
      packed UE8M0 gran32）、SF 变换 `:234-235`、arch 分派 `:238-248`（SM90 断言
      `gran_k_a==gran_k_b==128` `:246`，自定义 epilogue 仅 SM100 `:244`）。SM90 `sm90_fp8_bmm`
      （`sm90_fp8_gemm_1d2d.hpp:283-352`）**复用前向 `1d2d` 骨架的 `GemmType::Batched`**
      （`num_groups=batch_size` `:296`、3D TMA A/B/D `:311-329`、SF TMA `:331-332`）；
      kernel `sm90_fp8_gemm_1d2d.cuh`：`BLOCK_K==128` per-128 scaling `:56`、CD 仅 bf16 `:62`、
      PDL `:149`、batched TMA `batch_idx=current_group_idx` `:186-202`、**SFB 逐 batch 基址**
      由 `get_global_idx<true,SF_K>`（`scheduler/gemm.cuh:169-172`）给出 = `current_group_idx *
      shape_n_sfb*shape_k_scales`（`:240-243`）、3D store `:433-438`；Batched 调度
      `scheduler/gemm.cuh:90-91/247-259`。SM100 `sm100_fp8_bmm`（`sm100_fp8_fp4_gemm_1d1d.hpp:405-481`
      `Kernel1D1D`+FP8 输出对/自定义 epilogue）与 `bhd,hdr->bhr`/`bhd,bhr->hdr`（`:277`/`:287`）
      为 **SM100 专属，仅分析**（SM90 `:296` 不可达）。H100 实测 `harness/exp_fp8_einsum.{py,sh,out.txt}`：
      ① 语义误差**干净分两层**——vs 隔离量化的 fp32 模拟 ~2–4e-3（bf16 输出舍入）、量化本身
      ~3–4e-2；② 每头独立 GEMM vs 逐头 fp32 bmm relerr 2.497e-3、头 0 置零→输出头 0 全 0 且其余头
      逐位不变；③ SF 布局 A `[b,h,r/128]` / B `[h,d/128,r/128]`、逐头基址 stride 4*8=32；
      ④ 契约拒绝真实站点：FP8 输出对→`einsum.hpp:224`、A 非 K-major→`:206`、recipe gran64→
      `layout.hpp:119`、h 不符→`:214`、`bhd,hdr->bhr`→`:296`；⑤ 确定性开关对 FP8 路径**无效**
      （无 cuBLASLt 回退），恒命中 `sm90_fp8_gemm_1d2d_impl<... (GemmType)4= Batched ...>`；
      ⑥ 性能大收缩维 `r=4096,d=1024` 峰值 **581 TFLOPS**、小 `r=512,d=128` 仅 ~185 TFLOPS。
      配套博客 `content/posts/mega-moe-21-fp8-einsum-projection`（索引页新增第 21 篇）。

### P12 增量（opt-in，稀疏注意力 indexer 的稀疏 MQA logits）

- [x] **P12-1** **稀疏 MQA logits**（`fp8_fp4_sparse_mqa_logits` / `fp8_fp4_paged_sparse_mqa_logits`；
      indexer 第二级打分：在 `sparse_kv_block_indices` 选中的 KV 块内算细粒度 logits 并**压缩**输出
      `[T, num_max_sparse_blocks*SB]`）。**SM100 专属，仅分析**（host `attention.hpp:313/380/414`
      断言 `arch_major==10`）；语义可架构无关验证。
      → `docs/22-sparse-mqa-logits.md`（真实行号核对）：四个入口 `attention.hpp:303-442`
      （workspace `:284-301`、契约 `:380-433`）；layout `layout/sparse_mqa_logits.cuh`
      （`MetadataHeader :23-29`、`KVSplitHeader :31-61` 的两个高位标志、`KVBlockInfo :63-72`
      16+16 位打包槽、`ScheduleEntry :74-84`、`WorkspaceState :141-146` 三计数各占 L2 线、
      `kBlockQ=2 :17`）；metadata 尺寸公式 `sm100_sparse_mqa_logits.hpp:20-36` + `SPLIT_KV`
      fp8=512/fp4=640（`:15-18`）；metadata kernel `scheduler/sm100_sparse_mqa_logits_metadata.cuh`：
      有效块前缀 `:222-229`、两 token 归并去重放 packed_slots `:273-326`、物理块三态映射
      `:377-386`（对齐 `logical*SB` / 非对齐 `logical`（已含 `SB*blk+ks%SB`）/ 分页查 block_table）、
      末 CTA 建调度（`:443-472`）——连续版均分+splice `:37-73`、分页版 `:124-161`、
      `balance_wave_entries` 直方图排序+旋转 `:75-122`；主 kernel `impls/sm100_sparse_mqa_logits.cuh`：
      五类 warp、`UMMA_M=128/UMMA_N=BLOCK_Q*H/UMMA_K=32|64/SPLIT_KV=WG*128` `:160-166`、
      contiguous TMA 快路（仅 warp0 elect-one 发 TMA，其余 `arrive_count_pred` 释放）`:436-457`、
      越界尾块 `cp_async_*_zfill` `:461-505`、math warpgroup `relu×w` bf16x2 归约 `:600-625`、
      压缩列映射 `:627-633`。H100 实测 `harness/exp_sparse_mqa_logits.{py,sh,out.txt}`：
      ① metadata 尺寸公式复刻（16/16/8/16/384/8B，例 T=4096/NB=64/SB=8/fp8 连续 2,230,288 B）；
      ② 用真实 SM90 dense kernel 取全量 → 压缩输出 `gather==L` **逐位 True**、vs fp32 参考
      relerr 3.5–4.1e-2（=dense 量化噪声层）；③ 两 token 归并去重省 ~12.5%、计算压缩比 28×/7×/14×；
      ④ unaligned ks 块起点 `b*SB+ks%SB` 跨原块边界；⑤ 四个 host 入口在 sm90 全部被拒。
      配套博客 `content/posts/mega-moe-22-sparse-mqa-logits`（索引页新增第 22 篇）。

### P13 增量（opt-in，SM100 dense MQA logits 的负载均衡调度器）

- [x] **P13-1** **SM100 dense MQA logits 主 kernel + 连续版 metadata 负载均衡调度器**
      （`sm100_mqa_logits.cuh` / `scheduler/sm100_mqa_logits.cuh` / `sm100_mqa_logits.hpp`）。
      第 19 篇拆过 SM90 dense、第 22 篇拆过 SM100 sparse，本篇补中间缺席的 SM100 dense：
      主 kernel 是 block-scaled UMMA + TMEM（MXFP4/MXFP8、bf16 输出），**SM100 专属，仅分析**；
      但它的**连续版调度器与 UMMA 无关**，在 H100 上编译运行**真实设备代码**逐条验证。
      → `docs/23-mqa-logits-sm100-schedule.md`（真实行号核对）：host `attention.hpp:46-65`
      （`get_mqa_logits_metadata` 仅 SM100、`block_q=128/H`）、`:67-134`（SM100 out=bf16、split=384、
      输出 1024B 行对齐、SM90 拒 `schedule_meta`）；jit `sm100_mqa_logits.hpp:11-24` 常量
      （`kMQALogitsSplitCost=2`/`kMQALogitsSegmentCost=3`）、`:64-69` metadata 字数公式
      （`align(3*num_sms,2)+2*num_q_blocks`）、`:117-202` `MQALogitsConfig`（`umma_n=128`、
      FP4 流水更深、`smem_size` 复刻 `layout`）；`scheduler/sm100_mqa_logits.cuh`：KV span
      并集（`start/4*4` 向下对齐、尾行 clamp，`:37-54`）、偶/奇交错就地前缀和 `:56-79`、
      二分 `:82-93`、代价模型 `cost=2·splits+3·(非空)` 与前缀和 `:122-133`、
      **按代价切分给 SM（边界落在 split 粒度，非 Q block 边界）** `locate :142-155`、
      `SM100MQALogitsScheduler` 的 scheduled `:216-239` / grid-stride `:240-269` 双模式；
      分页版 `sm100_paged_mqa_logits.cuh`（不同代价 `2·BQ·blocks+tokens`、溢出回退、页坐标滑动缓存）
      对比。主 kernel `impls/sm100_mqa_logits.cuh` **仅分析**：五类 warp、`UMMA_M=128/UMMA_N=128`、
      A=KV/B=Q、UTCCP 把 SF 搬进 TMEM、`visit_tmem_head_chunks` + ReLU×w 的 bf16x2 归约、
      压缩偏移写回、`reuse_kv`（分页）。H100 实测 `src/mqa_logits_schedule.cu` +
      `harness/exp_mqa_logits_schedule.{sh,out.txt}`：**直接编译运行 `scheduler/sm100_mqa_logits.cuh`
      真实设备代码**（只需 cute/cutlass + PDL）——10 用例 **ALL PASS**：host 独立复刻与 device
      spans / 每-SM 边界**逐字段 0 不符**；scheduled 与 grid-stride 两种模式下真实 scheduler
      发射任务**恰好覆盖所有有效 split 一次**（dup/miss/bad=0；scheduled 任务数 > Q block 数，
      坐实「一个 Q block 被相邻 SM 从中间切开」）；负载均衡收益 skew S=8192
      scheduled **[185..191]** vs grid-stride **[99..352]**、8 SM **[50..55]** vs **[28..88]**；
      缓冲区字数公式与边界用例（`kstart%4≠0`、非 384 尾块、ragged tokens、空 span、8 SM）全通过。
      配套博客 `content/posts/mega-moe-23-mqa-logits-schedule`（索引页新增第 23 篇）。

### P14 增量（opt-in，SM100 dense MQA logits 的【分页版】调度器）

- [x] **P14-1** **SM100 dense MQA logits 分页版 metadata 负载均衡调度器**
      （`scheduler/sm100_paged_mqa_logits.cuh` / `sm100_mqa_logits.hpp:26-61,297-381` /
      `attention.hpp:136-282`）。第 23 篇做了连续版调度器，本篇补**分页版**：同一套主 kernel
      （block-scaled UMMA + TMEM，**SM100 专属，仅分析**），调度换成 varlen 请求模型。
      分页调度器与 UMMA 无关，在 H100 上**直接编译运行真实设备代码**逐条验证。
      → `docs/24-mqa-logits-paged-schedule.md`（真实行号核对）：host `attention.hpp:136-172`
      （`get_paged_mqa_logits_metadata` 仅 SM100、强制 varlen/next_n=1、block_kv∈{32,64,128}、
      indices 8B 对齐 `:160`；SM90 拒 varlen `:164`）、`:258-264`（out=bf16、split=384、1024B 行对齐）、
      `sm100_mqa_logits.hpp:13-18`（`kMQALogitsSplitKV=384`/`kMQALogitsSplitsPerChunk=8`）、
      `:32`（metadata 固定 `block_q=4`，注释 Tuned for H32）、`:318-333` 分页 tensor map、
      `:359/362` 实例化放开 `page_kv`/chunk。`scheduler/sm100_paged_mqa_logits.cuh`：
      `RequestInfo`（`:13-47`，请求 = indices run-length、context 取**最后一个 token** `:38`、
      Q token **均分** `:42-46`）；metadata kernel（`:50-159`）：`uint2` 两 token 扫描两遍
      （数起点→`cta_exclusive_sum`→写起点，`:71-98`）、代价模型 `split_cost=2·BQ·blocks+tokens`
      （`:110-112`）、**溢出回退**（单值 >2³² / 前缀回绕 → 借 `num_requests` 最高位标志位，
      回退 token 代价；`:114-141`）、**只在请求 split 粒度切边界**（无段启动代价，`:144-158`）；
      `SM100PagedMQALogitsScheduler`（`:162-281`）：`kNumPagesPerSplit`/`kNumCachedPages=32`
      （`:166-170`）、每 SM 用 `meta[sm]..meta[sm+1]` 作起止（`:202-205`）、
      `next_q_block` 按 **Q block→chunk→请求**推进（`:221-251`，`num_q_blocks==1` 不分块 `:232`、
      `kv_shared_with_prev` `:233`、「同一 KV 段的多个 Q block」一拍 `:239-241`）、
      每 token KV 上界（`:254-266`）、**每 lane 缓存一页的滑动窗口** `get_kv_page_coords`
      （`:269-280`，refill `:272`、越界映射 0 `:275`、`ptx::exchange` 广播 `:279`）。
      H100 实测 `src/mqa_logits_paged_schedule.cu` + `harness/exp_mqa_logits_paged_schedule.{sh,out.txt}`：
      **直接编译运行真实设备代码**——10 用例 **ALL PASS**：metadata 每-SM 边界 vs host 独立复刻
      （含 132/8 SM、空 SM）**逐字段 0 不符**；真实 scheduler 发射任务**所有 (请求,Q-block,split)
      恰好覆盖一次**（dup/miss/bad=0）；`kv_shared_with_prev`/块大小≤8/每 token KV 上界全对；
      **分页查表含 mid-refill**（PAGE_KV=32、chunk=8 单任务 96 页 > 32 缓存）与 host **0 不符**；
      **代价溢出回退**用例（4096 token、ctx=1.56e8，split 代价 4.99e9>2³²）`use_token_cost=Y`、
      total_cost=1.664e9、边界仍完全一致；坐实 metadata `block_q=4` 与主 kernel `BLOCK_Q=128/H`
      解耦（H=64 sched/meta slots **432/288**、H=128 **864/288**，均衡只估计、正确性靠 `has_work`）。
      配套博客 `content/posts/mega-moe-24-mqa-logits-paged-schedule`（索引页新增第 24 篇）。

### P15 增量（opt-in，贯穿所有 FP8/FP4 kernel 的 SF 布局前置步骤）

- [x] **P15-1** **DeepGEMM 块级 scale（SF）布局转换与 UE8M0 打包**
      （`csrc/apis/layout.hpp` / `csrc/jit_kernels/impls/smxx_layout.hpp` /
      `deep_gemm/include/deep_gemm/impls/smxx_layout.cuh` / `csrc/utils/layout.hpp`）。
      前面第 10/12/13/21 篇反复出现「SF 要先 MN-major + TMA 对齐」，本篇把这条**贯穿所有
      block-scale GEMM 的前置步骤**单独讲透。**转置/打包 kernel 都不是 SM100 专属，H100 可跑**。
      → `docs/25-sf-layout-transform.md`（真实行号核对）：host 四分派 `layout.hpp:11-58`
      （`:38-39` FP32 1×128 SM90 转置、`:42-43` FP32 128×128 SM90 仅 check、`:46-51` FP32 SM100
      转置+UE8M0 打包、`:54-55` INT SM100 仅 check，其余 `:57` UNREACHABLE）；**recipe 契约**
      （2-tuple `(gran_mn,gran_k)` 断言 `not is_sfa` `:28`；3-tuple `(a,b,k)` 断言 `is_sfa` `:24`）；
      `transform_sf_pair_into_required_layout:60-87`（SFA 传 true / SFB 传 false）；
      `get_default_recipe`（`utils/layout.hpp:86-98`，SM90 唯一 `(1,128,128)`）；
      `check_sf_layout:101-140`（`:115` dim、`:118/119` 形状随 gran、`:122-130` TMA stride、
      `:133-138` SM90 SFB 转置检查）；`get_tma_aligned_size`（`utils/math.hpp:23-27`）。
      device `smxx_layout.cuh`：`transpose_fp32:10-72`（`:36` PDL、`:64` host 幂等早退、
      `:70` 显式转置、`kSmemSFK+1` 防 bank 冲突）；`transpose_and_pack_fp32_into_ue8m0:76-191`
      （`:99` PDL、`:107-126` PSUM 有效区间预计算、`:161-190` 打包、`:183-186` 4 字节 LSB-first
      位运算、`:177-178` **只允许 2 的幂**（符号/尾数为 0）的 debug 陷阱）；
      `pack_fp32_into_ue8m0:193-293`（已 MN-major 输入，`:227-267` 沿 grouped_layout 找组与
      padding、`:279-282` 同款陷阱）。PSUM gap 行跳过（`:117-126`/`:142-149`/`:188`，写安全有限码
      而非 NaN `0xff`）。H100 实测 `harness/exp_sf_layout.{py,sh,out.txt}` 六组 **ALL PASS**：
      ① `get_tma_aligned_size` 公式逐点一致；② 转置 `val_diff=0`、stride `(1, aligned)`、**幂等**；
      ③ UE8M0 CUDA==torch 参考**逐位一致**、`0x848a847c`→指数 `[124,132,138,132]` LSB-first、
      `ceil_to_ue8m0` 尾数恒 0、`mn=137` torch 回退一致、**已 MN-major 走 pack kernel 与
      transpose_and_pack 逐位一致**（两 kernel 互证）；④ PSUM `groups_end=[300,700,1024]`
      152/1024 gap 行、有效行逐位一致、无 NaN 码；⑤ recipe 分派 `(1,128)/(128,128)` OK、
      `(1,32)` 不论 `disable_ue8m0_cast` 都 `:57 Unknown SF transformation`（SM100 专属）、
      `:24`/`:28` 契约反例被拦；⑥ 端到端 `fp8_gemm_nt`（m1024 n512 k1024）relerr 6.8e-4、
      裸 SF vs 显式变换 SF **逐位相等**（host API 内部本就调同一变换）。配套博客
      `content/posts/mega-moe-25-sf-layout-transform`（索引页新增第 25 篇）。

### P16 增量（opt-in，P15 延伸：K-grouped SF 打包）

- [x] **P16-1** **K-grouped SF 打包**：MoE wgrad 的**按组 block-scale 布局**
      （`transform_k_grouped_sf_into_required_layout` / `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`
      / `check_k_grouped_packed_ue8m0_tensor` / device `pack_fp32_into_ue8m0<kNumGroups>`）。
      第 25 篇讲了单组 SF，第 10/12 篇讲了 K 轴分组 GEMM，本篇补交叉点：**每组 SF 行独立对齐补零**。
      → `docs/26-k-grouped-sf-packing.md`（真实行号核对）：host 入口 `layout.hpp:89-116`
      （recipe 必须 `(1,1,gran_k)` `:96`；SM90 只转置 `:104-105`、SM100 才打包 `:108-109`；
      `k_alignment` 契约 SM90=128/SM100%128==0 `:100-101`）；`smxx_layout.hpp:218-283`
      （`arch==10` 断言 `:224`、contiguous 精确 `packed_sf_k=Σ ceil_div(k/gran_k,4)` `:240-248`、
      PSUM 上界 `(sf_k+3G)/4` `:249-253`、连续输出 `[packed_sf_k,mn]` `:257`）；
      `check_k_grouped_packed_ue8m0_tensor :286-317`（`gran_k==32 :293`、`stride(0)%4==0 :300`、
      `packed_sf_k>=Σ ceil_div(k,gran_k*4) :304-312`、PSUM `:313-314`）。device
      `smxx_layout.cuh:193-293`：每 warp 一个 packed 行 `:214-219`，线性扫组找 owner
      `:224-267`（PSUM 组起点 `align(prev,k_alignment)` `:246`、**break 在累加 padding 之前** `:256-261`），
      取址**减累积 padding** `:275`、LSB-first 打包 `:287`、trap 只允许 2 的幂 `:279-282`。
      **kernel 与 UMMA 无关**：H100 上**逐行原样抽取真实设备代码**编译运行（`harness/exp_kgrouped_sf_pack.{sh,out.txt}`）
      + 独立 host 参考对拍，**ALL PASS**：A) ng=1 退化；B) contiguous gran32（无/有 padding、空组、8 组）；
      C) gran128（kalign 128/256、行数 1/2/3 大量 padding）；D) PSUM（`ends={400,800,1024}` 实际 8 行、
      上界 10，尾部 2 行 poison；`ends={256,640,1024}` 实际 3 行、上界 4）——mism=0；
      E) 输出 K pitch=mn（连续）vs 非分组 TMA 对齐；F) host wrapper 在 sm90 被 `:224` 断言拒绝（RuntimeError）。
      配套博客 `content/posts/mega-moe-26-k-grouped-sf-packing`（索引页新增第 26 篇）。

### P17 增量（opt-in，DeepGEMM 的 cuBLASLt 回退与选核）

- [x] **P17-1** **cuBLASLt 回退路径与选核**（`csrc/apis/gemm.hpp` / `csrc/apis/einsum.hpp` /
      `csrc/runtime/runtime.hpp` / `csrc/jit_kernels/impls/smxx_cublaslt.hpp` /
      `csrc/jit_kernels/impls/epilogue_class.hpp`）：前面 26 篇拆的都是 DeepGEMM 自己的 JIT
      kernel，本篇补上**它什么时候不用自家 kernel**。**H100 可跑，真机实测**。
      → `docs/27-cublaslt-fallback.md`（真实行号核对）：**三与门** `gemm.hpp:518-523`
      （`not deterministic_algorithms` ∧ `is_cublaslt_available()` ∧ `epilogue_alpha.has_value()`）；
      `runtime.hpp:89-91 is_cublaslt_available` = `num_sms == device_num_sms`；epilogue→alpha 表
      （`epilogue_class.hpp:35/41-48/50-65/67-89/91-106`：Identity→1.0 所以**默认也回退**、
      FP8Quant/Stochastic→nullopt）；`bhd_bhr_hdr` **无条件** cuBLASLt（`einsum.hpp:128-148`）；
      翻译层列主序交换 `(A Bᵀ)ᵀ=B Aᵀ`（`smxx_cublaslt.hpp:115-135`）、`FAST_ACCUM` 关、`SM_COUNT_TARGET`、
      16MiB workspace（`runtime.hpp:25-26` 注释）、heuristic 只取 1 个、NVFP4 VEC16 UE4M3
      （`smxx_cublaslt.hpp:165-185`）、strided-batched syrk/symm（`:137-163`）。
      **FP8 dense/grouped 永不回退**（`gemm.hpp:108-132`/`:235-254`）。H100 实测
      `harness/exp_cublaslt_fallback.{py,sh,out.txt}`（profiler 抓真实 kernel 名）：
      ① 稠密 BF16 `bf16_gemm_nt` **默认 `nvjet_sm90_*`（cuBLASLt）**，`deterministic` 或
      `set_num_sms(124)` → `deep_gemm::sm90_bf16_gemm_impl<…>`；② `alpha=0.5`/`epilogue.Alpha(2.0)`
      仍回退；`epilogue.FP8Quantization` 在 sm90 被 `gemm.hpp:119` 拒；③ FP8 dense 恒
      `sm90_fp8_gemm_1d2d_impl`（relerr 1.66e-3）；④ einsum 三个分支的默认/确定性切换，`bhd,bhr->hdr`
      恒 `nvjet_*`；⑤ 显式 `cublaslt_gemm_nt/tt`(relerr 2e-4)/`+C`(6.6e-4)/`batched_syrk`(1.7e-3)/
      `batched_symm`(1.7e-3) 正确性；⑥ **性能打平**：8 组形状 cuBLASLt vs DeepGEMM 0.97–1.02×
      （4096×7168×7168 均 ~377 TFLOPS）——解释为何默认敢回退。配套博客
       `content/posts/mega-moe-27-cublaslt-fallback`（索引页新增第 27 篇）。

### P18 增量（opt-in，DeepGEMM 的选核启发式）

- [x] **P18-1** **`get_best_config` 选核启发式**（`csrc/jit_kernels/heuristics/`
      的 `common.hpp` / `sm90.hpp` / `sm100.hpp` / `config.hpp` / `runtime.hpp` / `utils.hpp`）：
      第 27 篇讲了「什么时候不用自家 kernel（cuBLASLt 回退）」，本篇补另一半——**用自家 kernel 时
      block_m/n/k、cluster、stage、线程数怎么选**。**纯 host 启发式、架构无关，H100 可跑**。
      → `docs/28-deepgemm-best-config.md`（真实行号核对）：`get_best_config` 三步
      （`common.hpp:18-56`「枚举候选 → 打分 → 取最小」，严格小于、并列取先者；`:43-54`
      `DG_PRINT_CONFIGS` 按 desc 去重打印）；五数据结构（`config.hpp:12-188`，`expected_*` 为 0
      则回退真实形状 `:30-33`，`check_validity` 要求 `num_sms%2==0` `:60`）。**SM90 候选枚举**
      `sm90.hpp:16-118`：block_m（小 m 才加 16/32；`cd_dtype!=Float` 才允许 256；m-grouped
      contiguous 钉死 `mk_alignment`=128 `:33`）、block_n（`lcm(16,multiple)` 步长；1D1D+FP32 输出
      `start=24` 避 bank conflict `:43-48`；上限 1D2D=192/1D1D=160/NoSF=256）、block_k 唯一
      `128*8/bits`、multicast 关闭条件（K 分组>4 或 batched `:62-67`）、cluster 只到 2、合法性过滤
      （1D2D unroll、masked 多播整除、寄存器、swizzle≥64B、stage≥3/小矩阵≥4）。**派生配置**
      `get_storage_config :120-146`（K-major 用 block_k 定 swizzle、FP32 输出不给 swizzle）、
      `get_pipeline_config :148-188`（smem 232448 硬约束、stage≤16、K-grouped 多留 4 tensormap）、
      `get_launch_config :190-200`（block_m≤64→128 math 线程否则 256）。**代价模型**
      `get_layout_info :202-239`：只建 L1/L2 周期（HBM/总算量近似不变），显式建模 multicast
      省 L2 读（`bm/cn+bn/cm`）、累积模式 D 回读翻倍、尾波效率惩罚、**单波禁多播置 INT64_MAX**；
      `compare :241-244` 单键取最小。**SM100 对照（仅分析）** `sm100.hpp`：swap AB 进入决策、
      m-grouped 单候选快路（`:34-46`）、TMEM 容量约束（`:126-132`）、强制 128B swizzle
      （`:137-141`）、k-grouped 单级 TMA store（`:193-207`）、代价模型 `num_cycles` 恒 0
      （`:271-272` TODO）改用**词典序 tie-break**（`:275-301`：单波→多播→少波→尾波→小块）。
      H100 实测 `harness/exp_best_config.py`（+`.sh`/`.out.txt`，`DG_PRINT_CONFIGS=1` 抓真实选核）：
      用 Python **逐行重写** SM90 那套，对 **20 组形状 × 15 字段逐项对拍 0 不符**
      （覆盖 Normal/1D2D、Normal/NoSF、MGroupedContiguous/Psum/Masked、KGrouped/1D1D、Batched）；
      候选排序显示大形状在 **90 个合法配置**里按 L1/L2 周期近似最优（次优差 2.7–9.3%）；
      坐实 bm=16/32 小 m 分支、大形状 bm256+cn2（BF16 输出+多播）、k-grouped FP32 的
      bn=136/152（`start=24`）、m-contig 钉 128 vs masked 可 64、`block_mult=(32,32)` 把 bn 从 16
      变 32、batched 恒 cm=cn=1。配套博客 `content/posts/mega-moe-28-deepgemm-best-config`
      （索引页新增第 28 篇）。

### P19 增量（opt-in，DeepGEMM 的 SM90 BF16 稠密 GEMM 主 kernel）

- [x] **P19-1** **SM90 BF16 dense GEMM 主 kernel**（`sm90_bf16_gemm.cuh` / `sm90_bf16_gemm.hpp` /
      `mma/sm90.cuh` / `common/tma_copy.cuh`）：第 10/11/12/13/21 篇讲的是 FP8 SM90 kernel，
      第 17/18/27 篇顺带用到 BF16 这条路却从没系统讲过。本篇补上 DeepGEMM **唯一一个同时服务
      Normal / MGrouped(contiguous & masked) / KGrouped / Batched 五种 `GemmType`** 的 SM90 kernel。
      **H100 可跑，真机实测**。
      → `docs/29-sm90-bf16-gemm.md`（真实行号核对）：host 六入口 `sm90_bf16_gemm.hpp:77-130`（Normal）、
      `:132-201`（MGroupedContiguous，A 必须 K-major/k%64）、`:203-262`（MGroupedMasked，A/B 都 K-major）、
      `:264-330`（KGroupedContiguous，MN/MN/每组 k%128）、`:332-384`/`:386-438`（两个 batched MLA），
      JIT 模板拼装 `:27-64`。kernel `sm90_bf16_gemm.cuh`：模板 21 参 `:27-37`、类型约束 `:64`、
      **合并 stage**（`kDoMergeStages` stage≥10+Normal+K/K-major+128 math → `BLOCK_K*=kNumStagesPerMerge`，
      `:44-56`，为减少 `warpgroup_wait<0>`）、smem 布局 `:71-109`、barrier init `:112-121`、PDL `:131`；
      **TMA warp** `:147-209`（第三 warp elect-one、multicast 判定 `:163-166`、按 major 发 2D/3D TMA
      `:188-199`、`expect_tx`）；**math warp** `:210-282`（`WGMMA=BF16MMASelector`：M=64/N=BLOCK_N/K=16、
      `kNumAccum=N/2`，wave `WAVE_BLOCK_M` `:228`、跨 wave 交错的两个 warpgroup `:219`、
      `advance_gmma_desc_lo` 前进 descriptor `:267-270`、`accum` 零初始化 + WGMMA `scale_d=1`）；
      **epilogue** `:284-388`（`do_wgmma_store` `:234`；bf16 输出走 STSM+128B swizzle `:302-349`，
      fp32 输出走裸 `st.shared` 且 `kSwizzleDMode=0` `:350-364`；`NamedBarrier` 后 TMA store，
      `Batched` 走 3D、累积走 `SM90_TMA_REDUCE_ADD_2D/3D` `:371-387`）。与 FP8 SM90 的对照表（WGMMA `K`
      16 vs 32、`BLOCK_K` 64 vs 128、有无 SF）。H100 实测 `harness/exp_bf16_sm90_gemm.{py,sh,out.txt}`：
      ① 稠密 7 形状 × out∈{bf16,fp32} × acc，relerr fp32 ~1e-6 / bf16 ~1e-4 / bf16-acc ~2.9e-3，
      峰值 **801 TFLOPS**（`4096×7168×7168` bf16）、fp32 略慢（关 swizzle + 双倍 store）；
      ② 别名 nt/nn/tn/tt 一致（~1.66e-3）；③ MGrouped contiguous/masked relerr 1.5e-4、contiguous
      **575 TFLOPS**；④ KGrouped MN/MN 强制 accumulate（`gemm.hpp:707-710`）relerr 1.7e-6、
      **422 TFLOPS**、非 128 倍数 K 被拒；⑤ batched MLA `bhr,hdr->bhd` relerr 1.5e-4、**781 TFLOPS**；
      ⑥ 抓到 `(128,128,512)` 触发合并 stage（`kNumStages_=16→per-merge 3`）、out=bf16 `kSwizzleDMode=128`
      / out=fp32 `=0`；⑦ deterministic off→cuBLASLt、on→`sm90_bf16_gemm_impl`（对照 P17-1）；
      ⑧ 踩坑：PDL 让背靠背 kineto 计时系统性偏小（改用 per-call sync+event）。配套博客
      `content/posts/mega-moe-29-sm90-bf16-gemm`（索引页新增第 29 篇）。

### P20 增量（opt-in，补齐 SM90 分页 MQA logits 的调度器）

- [x] **P20-1** **SM90 paged MQA logits 的 metadata 负载均衡调度器**
      （`scheduler/sm90_paged_mqa_logits.cuh`：`sm90_paged_mqa_logits_metadata` +
      `SM90PagedMQALogitsScheduler`；`csrc/apis/attention.hpp:136-172`、
      `csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp:94-135`）。第 19 篇拆了 SM90 dense 的
      连续/分页**主 kernel**，但分页调度器只一笔带过；第 23/24 篇做的是 **SM100** dense 的调度器。
      本篇补 SM90 自己的分页调度器。`split_kv=256`=4×block_kv=4×64、单 warp、反序均分；
      **与 WGMMA/TMA/UMMA 无关**，只依赖 `math.cuh`/`ptx`/PDL，故 H100 上**直接编译运行真实设备代码**。
      → `docs/30-sm90-paged-mqa-logits-schedule.md`（真实行号核对）：host 入口 `attention.hpp:136-172`
      （`[num_sms+1,2]` 元数据、SM90 非 varlen `:163-166`、`block_kv==64` `:165`）、
      metadata host 壳 `sm90_fp8_mqa_logits.hpp:94-135`（`split_kv=256 :100`、单 warp `:101`、
      `aligned_batch_size=align(batch,32) :102`、smem `:105-106`）；kernel `scheduler/sm90_paged_mqa_logits.cuh`：
      **varlen atom 配对分组**（`indices[t]==indices[t+1]` 合成 2-token atom、context 取 `[t+1]`，`:26-45`）、
      warp 内 `shfl_up` 前缀和（`:47-69`）、varlen 分配（`:72-95`，反序 `q/r/pivot` `:75-78` + 上界二分 `:79-95`、
      `<=kNumSMs` 同趟写终止哨兵）、非 varlen 分配（`:96-126`，`next_n_atom/num_next_n_atoms` `:97-98`、
      `q_atom_idx=q_idx`、`lane0` 写终止 `:122-125`）；`SM90PagedMQALogitsScheduler`：EBO 索引存储 `:130-136`、
      `atom_to_token_idx`（`kNextNAtom` / `kPadOddN`）`:151-163`、`atom_to_block_table_row` `:165-171`、
      `refresh_num_kv_and_advance`（一次 `is_paired` 得 advance+num_kv）`:179-192`、构造读 `uint2` 起止
      `:203-206`、`should_refresh_num_kv` `:215-221`、`fetch_next_task` `:223-241`、
      `exist_q_atom_idx` `:243-245`；主 kernel 消费 `impls/sm90_fp8_paged_mqa_logits.cuh:137-140`（`kNumMathWarpGroups=4`）
      / `:195-198`（逐 KV block 查 block_table）。**与 SM100 分页调度器的对照表**（varlen/代价模型/溢出/Q 粒度/KV 复用/分页坐标）。
      H100 实测 `src/sm90_paged_mqa_logits_schedule.cu` + `harness/exp_sm90_paged_mqa_logits_schedule.{sh,out.txt}`：
      **直接编译运行真实设备代码**——10 用例 **ALL PASS**：metadata 每-SM 边界 vs host 独立复刻
      （varlen atom 分组/前缀和/反序分配/二分）**逐字段 0 不符**（含 `batch=37` 补零、`varlen paired ctx`
      取 `[t+1]`）；真实 scheduler 发射任务**所有 (atom,split) 恰好覆盖一次**（dup/miss/bad=0，`tasks==segs`），
      `num_kv`/`advance`/token/block_table 行映射逐任务 0 不符；负载均衡 **±1**（`[3..4]`/`[7..8]`/`[15..16]`）；
      配对语义（`req=8` 512 token→256 atom）、少 SM/空 SM 正确。配套博客
      `content/posts/mega-moe-30-sm90-paged-mqa-logits-schedule`（索引页新增第 30 篇）。

### P21 增量（opt-in，贯穿所有 SM100 GEMM 的 epilogue 机制）

- [x] **P21-1** **SM100 GEMM 的 epilogue：store 路径与四个 epilogue 算子**
      （`deep_gemm/include/deep_gemm/epilogue/operators.cuh`、`.../sm100_store_cd.cuh`、
      `.../sm100_store_cd_swap_ab.cuh`、`csrc/jit_kernels/impls/epilogue_class.hpp`、
      `csrc/apis/epilogue_class.hpp`）：第 27 篇讲了「何时不用自家 kernel」（cuBLASLt 回退，含
      plain-store epilogue 三与门），本篇补「用自家 kernel 时尾部到底干什么」——**所有 SM100 GEMM
      共用的可插拔 epilogue**。`EpilogueClass`（`epilogue_class.hpp:19-39`）四类
      （Identity/Alpha/FP8Quantization/BF16StochasticRounding，Python `deep_gemm.epilogue.*`，
      `apis/epilogue_class.hpp:13-25`）经 `resolve_epilogue_class`（`:108-127` 三入口互斥）选一个
      device `epilogue::operators` 算子（继承 `EpilogueOperatorArgs`，`types.cuh:47-52`，
      **无状态**：`sm100_fp8_fp4_gemm_1d1d.cuh:57` 强制 `sizeof` 相等），随 kernel
      `__grid_constant__` launch（`:41`）。四算子（`operators.cuh`）：`Identity` `:12-15`；
      `ScaleByAlpha` `:20-30`（`__fmul2_rn` packed fp32x2，只缩放乘积项）；`StochasticRoundToBF16`
      `:32-65`（**四元组哈希** `partial_hash:35-39` 四常数 + `select_random_bits:41-48` 雪崩 +
      `cast_quartet:50-64` 共享 `h + quartet_n_idx`，硬件 `cvt.rs.bf16x2.f32`）；`QuantizeToFP8`
      `:69-92`（`kSFGranN=32`、`store_sf` MN-major 4 字节打包寻址）。store：`sm100_store_cd.cuh`
      `:34-50` 编译期判别+断言、`:124` `apply_values`、四条分支 `:144-174`（output SF：先 bf16 再
      per-32 `get_packed_bf16_amax`+`get_ue8m0_sf_exp`/`_inv`+`scale_bf16x2_into_fp8x4`、整字
      `sf_word`）、`:175-176` fp32、`:177-184` 随机舍入（每批 8 值=两 quartet）、`:185-193` 普通
      bf16；swap-AB `sm100_store_cd_swap_ab.cuh:192-221` 用 **`shfl_xor` 跨 lane 归约哈希**。
      接线：SM90 全部断言 `:119/238/308/385/528/611/664/712 "Custom epilogues require SM100"`；
      FP8 输出对=`FP8Quantization`（`einsum.hpp:190-200/213-220`，Batched+不累积 E4M3）。
      **SM100 专属，仅分析**（`cvt.rs` 在 sm90 ptxas 直接报 `Feature '.rs' not supported`；
      `__fmul2_rn` 仅 sm_100_rt）。**H100 实测**：`harness/exp_epilogue_operators.{sh,out.txt}` +
      `src/epilogue_operators.cu` 把 `operators.cuh:12-92` **逐行原样抽取**编译（sha256
      `d00b6b6f…`），两个 SM100 原语用标注的等价模拟：① ScaleByAlpha(0.5) 1024 值**逐位 0 不符**；
      ② 四元组哈希 n=2²⁰ 每 bit 偏离 0.5 ≤0.00118、64 桶卡方 86.1、offset 两两相等 ≈0（独立）；
      ③ 随机舍入**无偏**（中点/1/8/3/8 偏差 ~1e-5、向上比例=0.500/0.126/0.377=小数部分）vs
      **RNE 固定偏一侧**（-3.9e-3/-9.8e-4/-2.9e-3）；④ `QuantizeToFP8` D/SFD vs torch 参考
      **逐位 0/8192、0/256**（极端行 SF exp=128、全零组=地板 105），ALL PASS。
      配套博客 `content/posts/mega-moe-31-sm100-gemm-epilogue-operators`（索引页新增第 31 篇）。

### P22 增量（opt-in，补齐 SM90 FP8 稠密 GEMM 的两个主 kernel）

- [x] **P22-1** **SM90 FP8 dense GEMM 主 kernel：1D1D vs 1D2D**
      （`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh` / `sm90_fp8_gemm_1d2d.cuh`；
      `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp` / `sm90_fp8_gemm_1d2d.hpp`；
      `csrc/apis/gemm.hpp` / `layout.hpp`；`heuristics/sm90.hpp`；`mma/sm90.cuh`；`scheduler/gemm.cuh`）。
      第 29 篇补了 SM90 BF16 主 kernel，但 DeepGEMM 立身的 **FP8** 在 SM90 上是**两个**主 kernel，
      前面 10/12/13/21/25 篇只各碰片段。**H100 可跑，真机实测**。
      → `docs/32-sm90-fp8-gemm.md`（真实行号核对）：**同一数学 `D=(A·sfa)@(B·sfb)ᵀ`，只差 SFB 粒度**——
      `1D1D`（A/B 都 per-token 1D scale，`gran_n=1`；D=**float 且必须 accumulate** `hpp:84`；
      GemmType∈{Normal,KGroupedContiguous} `cuh:52`）与 `1D2D`（SFA per-token + SFB **per-128×128 块**
      `gran_n=128`；D=**bf16 禁 accumulate** `hpp:81`；GemmType∈{Normal,MGrouped(contig/psum/masked),Batched}）。
      **分派** `gemm.hpp:117-126` 按 `gran_n==1?1d1d:1d2d`；SM90 默认 recipe `(1,128,128)`（
      `utils/layout.hpp:86-98`）→ 稠密默认 **1D2D**，wgrad K 分组恒 1D1D（`gemm.hpp:239/309/422`）。
      **1D2D kernel**：`BLOCK_K==128` per-128 scaling `:56`、BF16 D `:62`、「too much B scales」
      `:57-59`、SFB 单份非流水 `:119`、empty barrier 计数 `multicast*math/32` `:134`、第三 TMA warp
      elect-one `:171`、B scale 由 **math warp 从 global 直读** `:239-248`、`num_former_iters` 处理
      BLOCK_N 跨 scale 组 `:230-235`、predicate 选 `scale_b_0/1` `:331-345`、STSM+swizzle+TMA store
      `:378-440`。**1D1D kernel**：`kNumTMAThreads==128` `:50`、多看一块 tensormap `:68`、
      **K 分组原地改 tensormap**（`tensor_map_replace_global_addr/inner_dim_stride` `:198-201` +
      commit/wait + release/acquire `:204-216`，即第 12 篇）、unroll 控制/寄存器重配
      `:150-154`、每 k 重算 descriptor `:303`、fp32 裸 `st.shared` `:334-340`、
      **`SM90_TMA_REDUCE_ADD_2D`** `:346-348`（必须 fp32+累加之根）。**十处差异对照表** + 选核
      （fp32 输出 `block_n` 起点 24/上限 160、bf16 上限 192，`sm90.hpp:42-54`；fp32 输出无 `block_m=256`
      `:29-30`）。H100 实测 `harness/exp_fp8_sm90_gemm.{py,sh,out.txt}`：① profiler 抓真实 kernel——
      默认稠密 `sm90_fp8_gemm_1d2d_impl<…,256,128,128,…,3,128,256,2,true,132,…,bfloat16_t>`、
      `recipe=(1,1,128)+acc` → `sm90_fp8_gemm_1d1d_impl<…,128,152,128,…,4,…,float>`（`BLOCK_N=152`
      正是 fp32 避 bank conflict 分支）；② 契约精确拦在 `hpp:84`（1d1d 无 c/bf16）与 `hpp:81`
      （1d2d+accumulate）；③ 语义 vs 反量化 fp32 参考：1d2d ~6.9e-4（bf16 输出舍入）、1d1d ~1.27e-4
      （fp32 输出，仅 FP8 量化误差）；④ **交叉验证**：把 1d2d 的块 scale 广播成 per-column 喂 1d1d，
      三个形状输出**逐位相等（relerr=0.000e+00）**，坐实「只差 SFB 粒度」；⑤ 性能 1d2d 恒快 ~1.5×
      （4096×7168×7168：**1407 vs 957 TFLOPS**；bf16/无 reduce-add/BLOCK_M=256/SFB 直读）。配套博客
      `content/posts/mega-moe-32-sm90-fp8-gemm`（索引页新增第 32 篇）。

### P23 增量（opt-in，贯穿所有 kernel 的 JIT 运行时）

- [x] **P23-1** **DeepJIT 运行时：编译 / 缓存 / 加载 / 发射管线**
      （`csrc/runtime/jit.hpp`、`deep_gemm/__init__.py:122`、`csrc/python_api.cpp`，
      DeepJIT `third-party/deep_jit` 的 `runtime/{runtime,config}.hpp`、`cache/{disk,memory}.hpp`、
      `utils/{hash,parser,env}.hpp`、`backend/cuda/{backend,kernel,options,device}.hpp`、
      `utils/lazy.hpp`）。前 32 篇一直把「JIT」当黑盒：本篇补上「为什么安装时不编译 CUDA、
      首次调用才现编」，以及缓存键五段构成与跨进程/跨节点复用。**架构无关，H100 真机可跑**。
      → `docs/33-deepjit-runtime.md`（真实行号核对）：DeepGEMM 接入点
      `jit.hpp:13-33`（`Config(root, "DG", "cutlass-<ver>", {include}, {"deep_gemm/"})` + 追加两条 nvcc flag；
      `deep_gemm/__init__.py:122` `_C.init(包目录)`）、`LazyInit`（`lazy.hpp:15-38`）、
      `register_python_api`（`python_api.hpp:12-20`）。`Runtime`（`runtime.hpp:19-98`）**五段式缓存键**
      （`:46-53`/`:92-98`：extra_signature + compiler_version + `get_flags()` + post_hook + 源码/include 摘要；
      源码摘要返回 hex 字符串又被外层再哈希）；两级缓存（mem `memory.hpp:9-22`；disk `<root>/cache/<tag>.<digest>/`
      `disk.hpp:79-134`，`.committed` 标记、原子 rename 发布 `:42-67`、`:` 多根 `:87-109`）。
      哈希 `hash.hpp:10-42`（双 state FNV1a + 长度前缀 `:21-24` + splitmix 输出 32 hex）与
      `parser.hpp:37-120`（只跟踪 `include_prefixes` 的递归 include、循环检测）。
      CUDA 后端 `backend.hpp:77-168`（nvcc `--cubin`、`cd` 临时目录、`check_no_spills/local_memory` 正则
      `:120-127`、dump PTX/SASS、写 `meta.json`）、`options.hpp:39-156`（默认 flag、`get_flags`、
      `update_hash` 只哈希 join(flags) → debug 的 `--generate-line-info` 也换键）、`device.hpp:68-80`（`sm_90a`/`sm_100f`）；
      加载发射 `kernel.hpp:48-171`（`cuLibraryLoadFromFile`「恰好一个 kernel」断言、`cuLaunchKernelEx` +
      cluster/PDL/cooperative 属性、默认当前流）。`get_compiled_dim`（`runtime_utils.hpp:31-40`）与
      `compiled_dims="nk"` 的调用形态（`sm90_fp8_gemm_1d2d.hpp:30-56` 微源码）。
      **H100 实测** `harness/exp_deepjit.{py,sh,out.txt}`：① 缓存条目 = `kernel.cu`(407B)+`cubin`+`meta.json`+`.committed`，
      一次 `fp8_gemm_nt` 生成 2 条（`transpose_fp32`+`sm90_fp8_gemm_1d2d`）；② **纯 Python 逐位复刻缓存键**
      `8fbcaf5d…` `MATCH=True`（递归哈希 **16 个** `deep_gemm/` 头、`nvcc --version` 原文含行尾换行）；
      ③ 编译 vs 命中：首进程 JIT ~4.98s → 二进程磁盘加载 ~0.20s（**~25×**），单 kernel 载入 0.5–2.3ms；
      ④ 抓真实 11 个 nvcc flag；⑤ `DG_JIT_DEBUG` 多 dump ptx/sass 且因 `--generate-line-info` **换键**，
      `sm90_fp8_gemm_1d2d` 168 寄存器/10 barrier/0 spill；⑥ digest 敏感性：`"nk"` 使 m 不显式进模板，
      但 m 经启发式 block config 仍区分条目，`ignore_compile_dims` 使维度行全 0；⑦ `CHECK_NO_SPILLS` rc=0；
      ⑧ FNV1a 长度前缀消歧成立。配套博客 `content/posts/mega-moe-33-deepjit-runtime`（索引页新增第 33 篇）。

### P24 增量（opt-in，Mega MoE 调度器按域分区的前置步骤）

- [x] **P24-1** **SM 局部域（locality domain）探针与内存域分配**
      （`deep_gemm/include/deep_gemm/impls/sm100_locality_domain.cuh`、
      `csrc/jit_kernels/impls/sm100_locality_domain.hpp`、`csrc/apis/locality_domain.hpp`、
      `csrc/runtime/locality_domain.hpp`、`csrc/runtime/mlopart.hpp`、`deep_gemm/utils/mlopart.py`、
      `deep_gemm/locality_domain.py`、`deep_gemm/include/deep_gemm/common/types.cuh:54`、
      `scheduler/mega_moe.cuh`）。第 03/09/11 篇反复出现 `num_locality_domains=2`（调度器按域分区、
      少量任务不跨域偷取、权重 `localize` 到各域），但「哪个 SM 属于哪个域、怎么测出来的」一直没讲。
      **探针设备代码只用 `%smid`/`ld.global.cg`/`clock64`（架构无关），H100 真机可跑**；host 逻辑纯张量。
      → `docs/34-locality-domain-probe.md`（真实行号核对）：域模型 `kNumDeviceLocalityDomains=2`
      （`types.cuh:54`）、两条实现路线（CUDA 13.4 前的 MPS MLOPart 双 device instance `:35/:52 TODO`
      对照 13.4 起 `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`）；全链路
      `mega_moe.hpp:266/373` → `sm100_*.hpp:221` → `kernel:79/945` → `scheduler:440`，
      域在调度器里的三处作用（任务计数按域 `:344-360`、`>=2` 波次才允许跨域偷 `:366-369`、
      shared L2 等所有域 L1 `:395-401`）。探针原理：一 SM 一 4KB chunk、步长 7 与 32 行互质、
      `ld.global.cg` 绕 L1、`clock64`/跳、uint16；host `flush_l2 :69-71` / `get_probe_chain :73-79` /
      `probe :82-90`（median(-1)）→ `try_probe :93-100` 两闸门（`kMinFarNearRatio=1.25 :66`、
      TPC 内一致）→ 最多 5 次（`:67`）→ `get_even :103-106` + warn。
      **`balance :126-136` 的语义陷阱**：只在「一个域占满」时给精确 50/50。本地化内存：
      `LocalityDomainAllocator`（reserve/map/release/get_locality_domain `:65-105`）+ MLOPart IPC
      （Python 起 MPS + `-mlopart`，`cuMemCreate/Export/Import` `mlopart.hpp:24-35`、`mlopart.py:94-130`）、
      粒度 2 MiB；按域 TMA 描述符 `(1,2,N/2,K)`（`runtime_utils.hpp:297-323`）。
      **H100 实测** `src/locality_domain_probe.cu` + `harness/exp_locality_domain_probe.{sh,out.txt}`：
      ① 原样 include 真实设备代码；链置换性质 YES；② 两域都在 HBM → 延迟 ~624 周期/跳、
      ratio 1.000、`clear=F` **正确拒绝**；③ domain1=host-pinned 模拟远端 → ratio **3.265**、
      准确检出并给出 TPC 一致划分；④ host 逻辑 9 个边界用例 **9 PASS/0 FAIL**（含 ratio=1.25 边界、
      TPC 不一致拒绝、`balance(50/66)=17/49` 记录陷阱）；⑤ 真实 API：`is_localization_available=False`、
      `get_granularity=2MiB`、`get_sm_locality_domains()` **逐位等于** `get_even`（66/66）。
      配套博客 `content/posts/mega-moe-34-locality-domain-probe`（索引页新增第 34 篇）。

### P25 增量（opt-in，补齐 SM100 唯一的 GEMM 后端）

- [x] **P25-1** **SM100 FP8/FP4 稠密 GEMM 主 kernel**（`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`；
      `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`；`deep_gemm/mma/sm100.cuh`；
      `heuristics/sm100.hpp`；`cute/arch/mma_sm100_desc.hpp`）。第 22 篇拆了 SM90 的 FP8 主 kernel，
      本篇补**目标卡上唯一**的 GEMM 后端——一套模板通吃稠密/m-grouped/k-grouped/batched，
      MXF8F6F4（`UMMA_K=32`）与 MXF4（`UMMA_K=64`、packed、K-major、gran32）共用骨架。
      **SM100 专属，仅分析**（kernel `:47` 整段 `__CUDA_ARCH__>=1000`）；架构无关的几何/描述符设备代码
      在 H100 真机跑。
      → `docs/35-sm100-fp8-fp4-gemm.md`（真实行号核对）：host `compile_and_launch :35-90`
      （MXF4/MXF8F6F4 只差 dtype 字符串 `:36-39`、JIT 微源码 `:40-79`、launch `:82-89`）+ 六入口
      `:101/169/254/325/483/405`（wgrad FP8 钉 MN `:337`、wgrad FP4 钉 K + gran32 `:496-497`、
      `k_alignment%block_k==0 :357/515`、psum `:181-186`、masked `:316`、k-grouped 3D CD `:370-375`、
      SF `sfa.size(0)*gran_k*4 :377-380`）+ `get_sf_block_config :93-99`。
      kernel：模板 `:24-37`、arch 守卫 `:47`、MXF4/`kPackFactor`/`BLOCK_K` 断言 `:51-89`、
      C/D+输出 SF `:60-66`、`UMMA_M=128*multicast`/`UMMA_N=BLOCK_M(swap)` `:68-85`、
      **TMEM 列预算 + overlap 技巧** `:120-132`（`accum=2*UMMA_N`；`overlap=max(x,512)-512`；对齐到 512）、
      epilogue 配置 `:106-118`；五类 warp（init `:167-192`、调度 `:197-210`、TMA `:213-300`、
      **MMA `:301-459`**、UTCCP transposer `:460-496`、epilogue `:497-561`）——**仅 leader CTA 单 warp 发 UMMA**，
      UTCCP 把 SF 搬进 TMEM（`:371-393`）、`tmem_overlap_barriers` 在 SF UTCCP 后/首条 UMMA 前等待
      （`:399-408`）、每步算 `subblock_idx/sf_id/tmem_col`（`:416-431`）、`umma_arrive` 通知 empty/tmem_full
      （`:335-350`）；gran32/128 的 stage 与 sf id（`:96-97/263-276/369-370/427-428`）；
      **SF SMEM 内 warp 转置** `:464-472`。描述符：`mma/sm100.cuh`（`make_smem_desc :14-40`、
      `get_umma_desc_stride_k :83-87`、`advance_umma_desc_lo :89-96`、`make_umma_desc :98-142`、
      `make_runtime_instr_desc_with_sf_id :144-148`、`update_instr_desc_with_umma_n :150-158`）、
      `packing.cuh:7-11`、`tma_copy.cuh:12-14`。H100 实测 `src/sm100_fp8_fp4_geometry.cu` +
      `harness/exp_sm100_fp8_fp4_geometry.{sh,out.txt}`：**真实设备代码**（SF 转置逐行抽取 sha256 `ae0401ba`；
      描述符/指令描述符用真实头文件）——A) SF warp 转置 **128/128 全对**；B/C) smem 描述符 4 例
      （K-major stride_k=1 / MN-major = swizzle atom；FP4 pack=2 偏移减半；SBO/LBO `>>4`）全对；
      D) 指令描述符 5 例（E8M0、`n_dim=N>>3`、`m_dim=M>>4`、2-bit sf_id、packed `float_e2m1_t`→`MXF4Format::E2M1=1`、
      `update_instr_desc_with_umma_n` 置 12）全对，**总判定 ALL PASS**。配套博客
      `content/posts/mega-moe-35-sm100-fp8-fp4-gemm`（索引页新增第 35 篇）。

### P26 增量（opt-in，补齐「稠密 GEMM 主 kernel」矩阵最后一格 + `set_tc_util`）

- [x] **P26-1** **SM100 BF16 主 kernel（`sm100_bf16_gemm.cuh`）与 tensor-core 利用率控制**
      （`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh`；
      `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp`；`csrc/apis/config.hpp`；
      `csrc/runtime/runtime.hpp`；`csrc/jit_kernels/heuristics/{sm100.hpp,config.hpp}`；
      `deep_gemm/include/deep_gemm/layout/gemm.cuh`；`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`）。
      第 19 篇拆了 SM90 BF16、第 22 篇 SM90 FP8、第 35 篇 SM100 FP8/FP4，只差 **SM100 BF16**——
      SM100 上所有 BF16 GEMM（稠密/m-grouped/k-grouped/batched，host 六入口
      `sm100_bf16_gemm.hpp:82/138/210/268/335/387`）的唯一后端。与第 35 篇**同一套 UMMA+TMEM 骨架、去掉 SF/UTCCP**，
      `UMMA_K=16`（`:69`）、`BLOCK_K_=64`（host `heuristics/sm100.hpp:32` 的 `128*8/bits`）。
      **两个独有机制**：① **device 内合并 stage**（`:46-53`，仅 `GemmType::Normal` 且 A/B 都 K-major 且
      stage≥8：`BLOCK_K = BLOCK_K_ * kNumStages_/8`，64→最多 256；`:45` 注释「reducing the `umma_arrive()` overhead」；
      `BLOCK_ATOM_K`（`:244`）+ 两级 `atom_k_idx/inner_k_idx`（`:304-309`）——与第 19 篇 SM90 merge 同思想不同动机）；
      ② **tensor-core 利用率控制 `set_tc_util`**（`:141-142/326-343`）：每次 UMMA 后用
      `tensor_core_full_barrier` 等其做完，再按 `kNumUMMACycles = 2·UMMA_M·UMMA_N·BLOCK_K/8192`（`:337`）与
      `kNumDummyCycles = (100-util)/util·kNumUMMACycles`（`:338`）由单 lane `clock64` 空转，把占空比压到 `util%`
      （注释 `:325`「Let tensor cores relax for lower possibility of frequency drop」）。
      **host 全链路**：Python `csrc/apis/config.hpp:21-25` → `runtime.hpp:22`（默认 0）/`:93-100`（get 把 0 翻 100）
      → `config.hpp:22` 的 `GemmDesc.tc_util`（**每个** kernel 都填）→ **只有** `sm100_bf16_gemm.hpp:70` 把它当
      模板实参 `kTensorCoreUtilControl`（`:35`）编进 kernel（`sm90_bf16_gemm.hpp:43`、`sm100_fp8_fp4_gemm_1d1d.hpp`
      的模板列表都没有 → 不影响行为/缓存键）。
      **SM100 专属，仅分析**（kernel `:43` 整段 `__CUDA_ARCH__>=1000`）；**架构无关的开关语义与节流机制 H100 真跑**：
      `harness/exp_tc_util.{sh,out.txt}` + `harness/exp_tc_util_throttle.{sh,out.txt}`、`src/tc_util_throttle.cu`——
      ① `set_tc_util` 语义（默认 100、`set(0)→100`、`-1/101` 在 `runtime.hpp:94` 被拒）；② **对 SM90 bf16 完全无效**
      （4096³：100/50/10 都 182 us、753 TFLOPS；SM90 FP8 都 107 us、1281 TFLOPS）；③ 冷缓存下切 `tc_util` 后
      第二次调用 **2558 ms→2.37 ms** 命中缓存（坐实不在 JIT 键里）；④ 节流机制复刻：公式 `cyc/(cyc+dummy)=util/100`
      精确成立，真机 `work/total` 随 util 单调下降（util=50→0.32、25→0.19），墙时间按比例拉长，但**空转超调**
      使实测利用率偏低、且 **H100 频率未随降载抬升**（降频意图依赖 SM100 tensor-core 功耗/热行为，无法复现）。
      行号全部核对过。配套博客 `content/posts/mega-moe-36-sm100-bf16-gemm-tc-util`（索引页新增第 36 篇）。

### P27 增量（opt-in，贯穿所有 FP8/FP4 kernel 的量化语义基准）

- [x] **P27-1** **测试参考量化器 `deep_gemm/utils/math.py`**：把「量化后的操作数是什么」单独讲透——
  前面第 25/26/32/35 篇反复引用 `per_token_cast_to_fp8` / `per_block_cast_to_fp8` 却从没系统拆过，
  而 DeepGEMM 全部 kernel 测试的正确性判据正是「精确等于这套参考定义的操作数上的 GEMM」。
  **纯张量运算，架构无关，H100 真机可跑**。
  → `docs/37-deepgemm-reference-quantizer.md`（真实行号核对）：`ceil_to_ue8m0:13-16`（fp32 位域：
  指数 + 尾数非 0 进位 → 「>= |x| 的最小 2 的幂」，exp 域钳位 [1,254]）、`pack_ue8m0_to_int:19-23` /
  `unpack_ue8m0_from_int:151-152`（4 字节小端、LSB-first；断言 scale 非负且尾数 0）、
  `per_token_cast_to_fp8:26-49`（1×128、`amax/448`、`clamp(1e-4)`、`align` 补零后切回、UE8M0 上取整、
  不足 4 的倍数补 1.0 再打包）、`per_channel_cast_to_fp8:52-59`、`per_block_cast_to_fp8:62-72`（128×128）、
  `per_custom_dims_cast_to_fp8:75-81`、`_quantize_to_fp4_e2m1:84-96`（7 个中点 + `round_up` 标志 =
  **按 E2M1 编码的 ties-to-even**；`sign=(x<0)&(code!=0)` 无负零）、`per_token_cast_to_fp4:99-125`
  （`amax/6`、nibble 偶数下标在低 4 位 LSB-first）、`transpose_packed_fp4:128-141`、
  `_dequantize_from_fp4_e2m1:144-148` / `cast_back_from_fp8|fp4:155-176`。
  **在测试里的用途**：`tests/test_mega_moe.py:54`（激活 per-token FP8 gran32 + packed UE8M0，再重排成
  UTCCP 4x32 交错行序 `:40-56`）、`:108`（FP4 gran32 权重）、`generators.py:317-346`（各 kernel 操作数布局）、
  `test_attention.py:60-68`（`cast_back_*` 造 simulated 参考、隔离 kernel 误差与量化误差）。
  H100 实测 `harness/exp_reference_quantizer.{py,sh,out.txt}` 七组 **ALL PASS**：
  A) `ceil_to_ue8m0` 1040 样本（幂次/次正规/溢出/inf/nan）exp 域 **0 不符**、输出恒为 2 的幂、
  仅溢出区 8 个样本 < |x|；B) 指数 `[124,132,138,132]`→`0x848a847c` LSB-first 一致、往返逐位一致；
  C) 三粒度 relerr（非 UE8M0）2.81–3.47e-2、UE8M0 上取整后 4.15e-2、全零块 SF `2.232e-07`；
  D) 7 个中点全部 ties-to-even、**20 万随机样本 vs 独立参考 0 不符**、vs ties-away 仅 4 个中点不同；
  E) nibble 字节 `0x21,0x43,…` LSB-first、FP4 往返 relerr 1.2384e-1 vs FP8 2.2504e-2（~5.5×）；
  F) `transpose_packed_fp4` 与独立转置重打包逐位一致且对合；G) 真跑 SM90 `fp8_gemm_nt` vs
  「参考量化器反量化 + fp64 GEMM」relerr **1.4217e-03**（仅 bf16 舍入）、vs 原始 bf16 3.6969e-02。
  配套博客 `content/posts/mega-moe-37-reference-quantizer`（索引页新增第 37 篇）。

### P28 增量（opt-in，贯穿所有 kernel 的 PTX 原语底座）

- [x] **P28-1** **DeepGEMM 的 PTX 原语底座：内存序 / 原子 / 共享内存搬运 / warp 通信**
       （`deep_gemm/include/deep_gemm/ptx/ld_st.cuh` / `ptx/utils.cuh` / `ptx/tma.cuh` /
       `common/tma_copy.cuh` / `common/ring_pipeline.cuh` / `comm/barrier.cuh`）。
       前面 0~27 篇把上层 kernel 全拆了，却从没系统讲过它们共同踩的**裸 PTX 原语层**——而 Mega MoE 的
       dispatch/combine 正确性、跨 SM/rank 的 readiness、`grid_sync`、tensormap 原地改全依赖它。
       **原语皆 `__device__` 内联、架构无关，H100 真机可跑**。→ `docs/38-deepgemm-ptx-primitives.md`
       （真实行号核对）：内存序阶梯表（`ld_acq[_gpu/_sys]`、`st_rel[_sys]`、`st_relaxed_sys`、
       `ld_volatile`、`atom[_rel/_sys]`、`red[_add/or/xor/_rel/_sys/_async]`，每个给真实调用点：
       `comm/barrier.cuh:42/45/75/82`、`scheduler/mega_moe.cuh:157/164/294/402/412`、
       `sm100_fp8_fp4_mega_moe.cuh:613/1243/1419/1440/245`）+ 三个「为什么」（combine 必须 `sys`、
       `red` vs `atom`、轮询用 `ld_volatile`）；`ptx/ld_st.cuh` 逐段（LDSM/STSM、`cp_async_cg[_zfill]`/`ca_zfill`、
       `mapa_shared`+`st_async_cluster`、`st_shared_bulk`、`ld_evict_first`+cache hint、`ld_gez_pred`、fence）；
       `ptx/utils.cuh`（`smid/laneid/gridid`、`bar.sync` vs `barrier.sync`、`exchange`、`redux.sync.max.f32` 仅 SM100、
       `cvt_relu_bf16x2`/`accumulate(_square)` 的 SM100 融合 `add/fma.rn.f32.bf16` 与回退）；
       `ptx/tma.cuh`（tensormap `release/acquire` + **原地改基址/stride** 对应第 12 篇 wgrad、mbarrier 五原语 +
       相位翻转、`tma_load/store_1d/2d/3d`、`tma_gather4`）；`common/tma_copy.cuh`（2~5D `copy_nd` 与
       `Load`/`Load2SM`/`LoadMulticast` 三分支）；`common/ring_pipeline.cuh`（幂次容量位运算快路 + 相位翻转）。
       **H100 实测** `src/ptx_primitives.cu` + `harness/exp_ptx_primitives.{sh,out.txt}`（真实头文件，
       sha256 `ld_st 912e9f9a…`/`utils 8a9d5e03…`/`ring_pipeline d231e0ce…`，总判定 **ALL PASS**）：
       A) `st.release.gpu`+`ld.acquire.gpu` 生产者-消费者 200 轮×64 slot×128 word **fail=0**（弱序对照也 0，
       注明单卡 L2 强、非 ISA 保证）；B) `atomic_add` vs `red_add`/`red_add_rel` 三条求和都 =524800，
       `atomic` 返回的 1024 个旧值**互不相同** 1024/1024；C) `RingPipeline` ns∈{1,2,4,8} 40 步轨迹与 Python
       复刻**逐项 0 不符**；D) `cp_async_cg<256>`+`cg_zfill<256>`(8B) 128 块、`ca_zfill`(0/4B) 64 块 **mismatch=0**；
       E)         `ld_evict_first` 256 MiB 流式读 0.210ms/**1279 GB/s** vs 普通 `ld` 0.238ms/**1129 GB/s**（约 +13%）；
       F) `cvt_relu_bf16x2` 回退 vs 独立 RNE **0/65536**、`accumulate(_square)` 与「bf16→f32 精确」一致。
       **未能测**：`redux.sync.max.f32`（SM100 专属）；`grid_sync` 整体语义需全 grid 常驻而本机 GPU 被多个
       自驱任务共享会死锁（已移除，仅源码解读）。配套博客
       `content/posts/mega-moe-38-deepgemm-ptx-primitives`（索引页新增第 38 篇）。

### P29 增量（opt-in，补齐被现代 C++ 取代的 legacy grouped GEMM）

- [x] **P29-1** **legacy Triton MoE grouped GEMM 家族**（`deep_gemm/legacy/`：`m_grouped_gemm.py` /
       `a_fused_m_grouped_gemm.py` / `a_fused_k_grouped_gemm.py` / `b_fused_k_grouped_gemm.py` /
       `tune_options.py`；`tests/test_legacy.py`、`tests/generators.py`）。第 10~13 篇拆的是现代 C++
       的 `k_grouped`/`m_grouped`（`grouped_layout` + 强制对齐），本篇补上它**取代掉的那套 A100 时代
       纯 Triton 兜底**——官方注释即「Legacy Triton kernels for A100」（`__init__.py:117`）。
       **Triton 架构无关，H100 真机可跑真机实测**。
       → `docs/39-legacy-grouped-gemm.md`（真实行号核对）：家族总览（dgrad 两个 + wgrad 两个）+
       `tune_options.py:10-28` 的三条过滤（`get_config_smem_size<=166912` A100 预算、`BLOCK_M/K<=align`）、
       `key=[]` + k-grouped 的 `restore_value=['d_ptr']`（因原地累加）。**核心是「分组表示」的差异**：
       legacy 用**逐元素索引张量**（`m_indices`/`m_row_indices`/`k_indices`）做**任意 gather**，
       现代用 `grouped_layout`（对齐连续段）。两个贯穿机制：① **BLOCK 首行定组**
       （`m_grouped:36-40`、`a_fused_m:38-42`，首行 `-1` 整块写零）；② 对齐要求
       （`m_grouped:72` `M%align==0`、`a_fused_k:80`/`b_fused_k:79` 非 gather 侧 `K%align==0`）。
       **命名坐实**：`a_fused_k:53-57` 的 `rows` 索引 **A**、`b_fused_k:54-58` 索引 **B**；
       `-1` 索引即行掩码（`(rows>=0)`）。空组 `k_start>=k_end`：`ACC=True` 保留 D / `ACC=False` 写零
       （`a_fused_k:43-47`）。legacy **不裁每组 K 尾块**（`test_legacy.py:54`）→ 强制 `k_alignment==128`、
       `ACC=True`、fp32 输出。H100 实测 `harness/exp_legacy_grouped_gemm.{py,sh,out.txt}`：**16/16 PASS**——
       `m_grouped` 有效行**逐位**等于参考 + padding 全零；identity `m_row_indices` 与 `m_grouped`
       **逐位相等**、置换 `m_row_indices` 复现任意逐行 gather（calc_diff 1.8e-10）、block 首行 `-1` 整块写零；
       `a_fused_k`/`b_fused_k` identity **逐位相等**、非 identity 分别坐实「索引 A / 索引 B」、`-1` 掩码生效；
       空组两种 `ACC` 语义；`M%align`/非 gather 侧 `K%align` 被断言拒绝。性能（Triton sm90）：
       m-grouped `4096^3`（G=8）**142.9 TFLOPS**、k-grouped `K=16384`（G=4）**107.7 TFLOPS**
       （官方更大形状 m-grouped 285–361、k-grouped 104–109）。配套博客
       `content/posts/mega-moe-39-legacy-grouped-gemm`（索引页新增第 39 篇）。

### P30 增量（opt-in，前面各篇全部数字的来源：测试基座）

- [x] **P30-1** **DeepGEMM 的测试基座**（`deep_gemm/testing/{bench,numeric,utils}.py`、
       `deep_gemm/utils/dist.py`；`tests/test_bf16.py:45/56`、`tests/test_mega_moe.py:11/76/354`）。
       前面 0~29 篇每一个「对拍 / relerr / TFLOPS GB/s」数字都出自这套基座，却从没系统拆过。
       **架构无关（非 kernel），H100 真机可跑真机实测**。
       → `docs/40-deepgemm-testing-harness.md`（真实行号核对）：`bench.py:7-33` `bench`（event 计时、
       256MB L2 flush、`high_precision` 用 `8192²` fp32 大核藏 CPU 发射）、`:79-146` `bench_kineto`
       （`DG_USE_NVIDIA_TOOLS` 旁路 `:88-90`、schedule `wait=0/warmup=1/active=1` 两批取第二批
       `:101-105`、`flush_l2` 每次清 256MB `:107-108`、`barrier` 用 `cuda._sleep(2e7)` `:109-113`、
       按名字子串匹配 + ms/us 单位换算 `:130-143`、tuple 多 kernel + `with_multiple_kernels` 断言
       `:84/121-123`）。`numeric.py:5-11` **`calc_diff` 实测恒等于归一化平方误差**
       `‖x−y‖²/(‖x‖²+‖y‖²)`（升 fp64 `:6`）：对称、∈[0,2]、`=0` 当且仅当逐元素相等、
       **同步缩放不变 / 单侧缩放不不变**、小误差 `≈ rel_l2²/2`、DC 平移敏感；`:14-21` `count_bytes`
       递归数 `(data,sf)` 对、跳过 `None`；`:24-44` `assert_bitwise_equal`（逐位 + 首个差异字节/elem/coord/
       byte_in_elem 定位）。`utils.py:6-8` `get_arch_major`（H100=9/SM100=10）、`:11-22` `test_filter`
       （假条件跳过并打印）、`:25-37` `ignore_env`（临时 pop 环境变量）；**两 wrapper 都不 `return`**
       （只适合副作用函数，实测装饰后返回值恒 `None`）。`utils/dist.py:38-66` `uneven_all_gather`
       （**交换长度→pad→all_gather→trim→cat**，MoE 测试 `:354` 拼不等长 `topk_idx` 做参考统计）、
       `:10-35` `init_dist`（tcp + 兼容 `device_id` 签名）。H100 实测
       `harness/exp_testing_harness.{py,sh,out.txt}` 六组 **ALL PASS**：① `calc_diff` 性质逐条核过；
       ② `bench` vs `bench_kineto`（真实 `fp8_gemm_nt`）：小 kernel event 含发射明显偏大
       （`high_precision` 本次 29.3→5.3us 生效、但共享 GPU 下抖动）、大 `4096³` 冷/热 L2
       （flush 300.7 vs noflush 224.8us，~34%）、tuple 分出 `transpose` 1.7/2.5us；
       `DG_USE_NVIDIA_TOOLS=1` 返回 1/(1,1) 跳过 profiling；③ `count_bytes` 递归对上；④
       `assert_bitwise_equal` 改一个 bf16 元素定位到 `(3,5)`/`byte_in_elem=0`；⑤ `get_arch_major=9`、
       `test_filter`/`ignore_env` 语义（含 wrapper 不 return）；⑥ `uneven_all_gather` 8×H100
       `n_tok=[17,512,4096,3,2048,100,777,64]` 拼接正确且各 rank 一致、`dim=1` 正确；117.4MB 时
       ~11336us（本机无 Fabric Manager 关了 NVLS，~73 GB/s）。配套博客
       `content/posts/mega-moe-40-testing-harness`（索引页新增第 40 篇）。

### P31 增量（opt-in，前面各篇「操作数长什么样」的来源：生成器）

- [x] **P31-1** **DeepGEMM 测试的操作数/布局生成器 `tests/generators.py`**（608 行；被
       `test_bf16/fp8_fp4/einsum/attention/mega_moe/layout…` 全部 `import`）。
       第 40 篇拆了判对/计时基座（`deep_gemm/testing/`），但每篇里出现的 `major_a/major_b`、
       `grouped_layout`、`use_psum_layout`、`ks_cpu` 到底长什么样从没讲清——本篇补上
       **「测试怎么造操作数」**，它是 kernel 操作数契约的可执行参考。**纯 Python、非 kernel，
       H100 真机可跑真机实测**。
       → `docs/41-deepgemm-test-generators.md`（真实行号核对）：三层测试栈（test → generators →
       `utils/math.py` 参考量化器 / `deep_gemm/testing/`）；三个基础类型（`KernelType`
       `:18-30`/`get_kernel_types :101-105` sm90 只 1D2D、sm100 只 1D1D；`MajorTypeAB :33-41`/
       `get_major_ab :108-115`；`QuantConfig :44-86`——legacy 前向 recipe `None`、wgrad `(1,1,128)`、
       B recipe 在 `is_fp4_b or is_wgrad` 时退化成 `(1,gran_k)`、`max_diff` 按精度 1e-3/1e-2/2e-2、
       `get_list_from_dtype` 仅 SM100 追加 FP8 配置）。**mk-alignment 的全局副作用**：枚举器
       `:258/:284/:297` 每次迭代都 `set_mk_alignment_for_contiguous_layout`（实测末值 384），
       test 必须显式设；sm90=128 / sm100=256（`heuristics/runtime.hpp:57-64`）。**operand major
       物理编码** `cast_fp8_fp4_with_major :318-326`：FP8 MN-major=`.T.contiguous().T`（换 stride，
       `(128,1)`→`(1,64)`，转置后逐位一致）、FP4 MN-major=`transpose_packed_fp4(x).T`（重打包，
       `(M,K/2)→(K,M/2).T`，因 K 配对变 MN 配对无法只靠 stride），**SF 与 major 无关**。
       **三种 grouped 布局**：`generate_m_grouped_contiguous :378-419`（contiguous=逐行组号+`-1`
       padding+padding 行置零 `:406`；PSUM=长度 G 逻辑结束偏移）、`generate_m_grouped_masked
       :446-485` + 两个转换器 `layout_masked_to_psum :422-431`（gap 故意不初始化）/
       `masked_valid_mask_to_psum :434-443`（gap 必须 False）；PSUM 语义等价
       `build_psum_layout_from_ks :542-548`（`align(prev)+k`）。**K 轴分组** `generate_k_grouped_
       contiguous :551-608`：contiguous=`host_ks`（物理 K）/PSUM=`logical_group_ends`，A/B 仅组
       开头 logical_k 行非零，K-major data 是 1D 展平，**SF 用 `per_channel_cast_to_fp8`
       （`(K/gran_k, MN)`，沿收缩维 K）**、每组 SF 行独立对齐补零，变体枚举含空组/单组。H100 实测
       `harness/exp_test_generators.py`（+`.sh`/`.out.txt`）八组 **41 PASS / 0 FAIL**：枚举器普查
       （normal 42/m-contig 16/m-masked 32/k-contig 7/transpose 60/sf_layout 108/k-sf 48/psum-sf 48）、
       alignment 副作用、major 编码三条、三种布局语义全对；**四个真实 SM90 kernel 端到端对拍**
       （`fp8_fp4_gemm_nt` / `m_grouped_..._contiguous` / `..._masked` / `k_grouped_..._contiguous`）
       `calc_diff` 6.7e-4/6.8e-4/6.8e-4/2.2e-4 全部 < `max_diff`。SM100 专属分支仅分析。
       配套博客 `content/posts/mega-moe-41-test-generators`（索引页新增第 41 篇）。

### P32 增量（opt-in，补齐主 kernel 之前的整层 host 编排）

- [x] **P32-1** **Mega MoE 的 host 编排层与对称缓冲区布局**（`deep_gemm/mega/__init__.py`；
      `csrc/apis/mega_moe.hpp`；`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`；
      `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`；`csrc/jit_kernels/heuristics/mega_moe.hpp`）。
      第 02/03/09/11/20 篇反复出现 `SymmBuffer`/`x_sf`/`num_ring_tokens` 却从没把「从 Python 到裸字节」
      串起来。本篇补上**主 kernel 之前的一整层 host 编排**——它**全纯 host、架构无关**，DeepGEMM 还
      pybind 暴露了 `_C.get_symm_buffer_size_for_mega_moe` 等，故 **H100 真机逐字段对拍**。
      → `docs/42-mega-moe-host-buffer.md`（真实行号核对）：Python `SymmBuffer`（`:28` Tmax 对齐
      `get_token_alignment_for_mega_moe()=kLCMCandidateBlockM=1920`；`:50-68` size-1 退化 `torch.empty`、
      `base` 复用只校验字节数、分配后 `zero_/barrier/sync`；`:70-76` 12 view；`:112-164` 权重交错/UTCCP
      转置）。C++ `get_symm_buffer_size_for_mega_moe:32-153`：ring 容量**遍历所有候选 BLOCK_M 取 max**
      （`:56-63`，`num_experts_per_rank` padding + `align 1920`）、SF ring `max get_num_sf_ring_tokens`
      （`:70-77`）、SF 断言（`:88-92`）；`slice_input_buffers:96-151` 的 12 个 view。layout
      `MegaMoEBuffer:345-457`：`Data`（行字节 + TMA 对齐标志）、`Buffer.get_end_ptr` **链式加法**无空洞、
      `Workspace:93-254` 的 `MegaMoESignals`（`:54-90`，`sizeof=21,140,864`、`:91` offset==128 static_assert）
      + 元数据、`num_max_pool_tokens:20-28`。四个非平凡点：**`x`≡`shared_l1_acts`**（`:114`）、
      **`x_sf` K-major vs `l1/l2_acts_sf` M-major**（注释 `:95`，第 10 篇 UTCCP）、**SF 行字节
      `hidden/32` 与 shape `hidden/128` int32 两口径一致**（`:393`）、**combine buffer 外层维
      `num_topk(+shared)` = K 倍放大**（`:447-449`）。scheduler `get_num_max_live_pool_blocks:48-80`
      的保守闭式（`get_num_l1_warmup_waves:17-45` warmup 避免 L1→L2 死锁 + `frontier_growth` +
      `wave_margin` + `min(total)`；注释 `:141` 指出相位翻转依赖 ring 容量）。`fp8_fp4/bf16_mega_moe`
      入口（`:155-390`）：recipe `(1,1,32)`、权重/SF 检查、shared expert、缓冲校验、
      **arch 派发只有 `arch_major==10`**（`:256/367`，H100 唯一闸门）、`DG_COMM_KERNEL_DEBUG` 清零；
      `register_apis:392-398`。H100 实测 `harness/exp_mega_moe_host_buffer.{py,sh,out.txt}`：
      **Python 逐字段复刻** `MegaMoESignals` 结构体布局 + `Workspace`/`Data`/`Buffer`/`MegaMoEBuffer`
      链式 offset + ring 公式，与真实 binding 对拍——**7 组配置 `num_bytes` 全 MATCH、全部 view
      (offset,shape,stride,dtype) 全 OK，总判定 ALL PASS**；真实 `SymmBuffer`（gloo ws=1）Tmax 512→1920、
      `l1_acts_sf` stride `(1,153600)`、`shared_l1_acts is x`、写读回、小配置复用 OK / 大配置被拒；
      ring 容量阶梯坐实 `live*BM` 随 BLOCK_M 单调升（240 最大 119280→ring 120960）、SF ring max 在
      BLOCK_M=8（1935360）。配套博客 `content/posts/mega-moe-42-mega-moe-host-buffer`（索引页新增第 42 篇）。

### P33 增量（opt-in，补齐测试栈的「更硬逐位断言」层）

- [x] **P33-1** **DeepGEMM 测试断言辅助 `tests/utils.py`**（`assert_psum_zero_padding` /
      `assert_direct_output_matches_fp32_accumulation` / `add_sf_k_padding` /
      `assert_stochastic_bf16_matches_fp32_accumulation` / `convert_to_fp8` /
      `to_cublaslt_vec16_sf_layout` / `make_cublas_gemm`）。第 40 篇拆了计时/判对基座、第 41 篇拆了
      操作数生成器，本篇补上 `test_bf16`/`test_einsum`/`test_fp8_fp4` 共用的**「比 relerr 更硬的逐位
      断言」**——它们正是每个 kernel 正确性判据的**可执行 oracle**。**架构无关，H100 真机逐位对拍**。
      → `docs/43-deepgemm-test-utils.md`（真实行号核对）：`assert_direct_output_matches_fp32_accumulation`
      `:20-38`（把同一零张量同时当 D/C → 累积退化成 `GEMM+0`；fp32 逐位、bf16 再 cast 一次逐位）；
      `assert_stochastic_bf16_matches_fp32_accumulation` `:53-79`（纯 torch `:65-78` 复刻 device
      `operators.cuh:35-48` 的 quartet 哈希，位序 `offset%2*16` ↔ `stack(h1&0xffff,h1>>16,h2&0xffff,h2>>16)`、
      乘子 `q<2?0x846ca68b:0xd35a2d97` 逐一对应）；`assert_psum_zero_padding` `:9-17`（PSUM 逻辑尾
      `current_m` 到 `align` 边界全零，A+D 都要）；`add_sf_k_padding` `:41-50`（**用 UE8M0 NaN `-1` 毒化
      padding**，暴露 kernel 用紧凑 pitch 的越界读；非 k-grouped `(1,pitch)`/k-grouped `(pitch,1)`）；
      `convert_to_fp8` `:82-95`（E4M3 精确表示 E2M1 的 LUT `:90` + 符号位 `(nibble&8)<<4` + LSB-first
      解包，`-0` 码 `0x80`）；`to_cublaslt_vec16_sf_layout` `:98-110`（MXFP4 per-32 → NVFP4 per-16 UE4M3
      `repeat_interleave(2)`，`exp>=-6?(exp+7)<<3:1<<(exp+9)`，分块 `[⌈mn/128⌉,⌈k/64⌉,32,4,4]`，契约
      `sf` 必为正 2 的幂）；`make_cublas_gemm` `:113-120`（int8=NVFP4 分支 / FP8 分支，`kPackedFP4=kInt8`
      `utils/math.hpp:11`）。device 对照 `epilogue/operators.cuh:32-63`。**H100 实测** `src/test_utils_sr.cu`
      （把 device 哈希常数搬进 CUDA）+ `harness/exp_test_utils.{py,sh,out.txt}`：**27/27 PASS**——
      ① `convert_to_fp8` 全 256 字节码 vs 独立参考逐位一致、MN-major 分支=转置后转换再转置、解包
      LSB-first；② NVFP4 分块形状 `(2,10,32,4,4)`、与独立复刻逐位、padding 全 0、UE4M3 编解码回 2 的
      幂 scale **零误差**；③ `add_sf_k_padding` 两 stride + 读底层 storage 确认 padding 全 `-1`；④ 真实
      SM90 `bf16_gemm_nt`：direct fp32 == 累积 `GEMM+0` **逐位**、direct bf16 == 基线 cast 一次、篡改
      1 bit 被 helper 检出；⑤ 随机舍入 oracle：device 哈希 vs torch 公式随机 bit/输出 **0/262144 不符**、
      CUDA 输出直接喂**真实 helper 通过**、向上比例 0.4985 ≈ 平均小数 0.4993、与 RNE 差 130674/262144；
      ⑥ cuBLASLt FP8 relerr 1.39e-6、`make_cublas_gemm` 一致；NVFP4 端到端因本机 cuBLASLt 返回
      `NOT_SUPPORTED`（`smxx_cublaslt.hpp:89`，布局本身正确）而 SKIP；⑦ 真实 SM90 PSUM
      `m_grouped_bf16_gemm_nt_contiguous`（G=4,m=34304,n=6144,k=7168）`calc_diff=0`、helper 通过、
      非零 padding 被检出。配套博客 `content/posts/mega-moe-43-deepgemm-test-utils`（索引页新增第 43 篇）。

### P34 增量（opt-in，补上 Mega MoE 一直缺席的「共享专家」分支）

- [x] **P34-1** **Mega MoE 的共享专家（shared expert）分支**（`csrc/apis/mega_moe.hpp`、
      `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`、
      `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`、
      `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、`deep_gemm/mega/__init__.py`、
      `tests/test_mega_moe.py`）。第 02/03/09/10/11/20/42 篇反复出现的「路由专家 + combine」一直漏了
      **每 token 恒激活的共享专家**——它是 Mega MoE 里唯一「不走 dispatch、却走完整 L1/SwiGLU/L2/combine」
      的通路。**主 kernel SM100 专属，仅分析**；host/调度器/UTCCP 映射/数学架构无关，H100 真机实测。
      → `docs/44-mega-moe-shared-expert.md`（真实行号核对）：**host 契约** `mega_moe.hpp:206-228`
      （共享权重是 2D `[2S,H]`/`[H,S]`、`E_s = S/I` 整除、`S%I==0`、**恒 FP8 e4m3** `:222-223`、
      SF 检查传 `nullopt` `:224-227`）+ `get_symm_buffer_size_for_mega_moe` 的 `num_shared_experts`
      （`:35/40/53/80-85`）+ 12 view 里的四个共享 view（`shared_l1_acts ≡ x` `:114`、shared SF K-major
      `:115-128`）。**布局** `layout/mega_moe.cuh`：`get_num_max_shared_sf_tokens = ceil_div(T,8)*128`
      （`:36-40`）、`MegaMoEBuffer` 的共享缓冲链式分配（`:384-424`，`shared_l1_token_buffer =
      input_token_buffer` `:415`、combine 外层维 `num_topk+1` `:448`）、`Workspace` 的两条共享任务
      计数器 + `shared_l2_full_count[kNumMaxSharedL2Blocks=32768]`（`:69-70/89/117/125`）。
      **调度** `scheduler/mega_moe.cuh`：`BlockPhase::SharedLinear1/2`（`:83-89`）、`is_shared()` =
      `phase>Linear2`（`:131-133`，`sizeof` 相等 `:136`）、形状关系
      **`SHARED_L1_SHAPE_N = L1_SHAPE_N·E_s`（N 轴）/ `SHARED_L2_SHAPE_K = L2_SHAPE_K·E_s`（K 轴）**
      （`:181-195`）、`shared_mainloop`（`:421-436`，`pool_block_idx=m_block_idx`、`expert_idx=0`、
      `valid_m` 裁尾）、`mainloop` 的**共享 L1 无条件在 `fetch_expert_recv_count` 之前发布**（`:445-452`
      重叠 dispatch）与 **`is_shared_l2_early` 启发式**（`:443-444`，`:467-471` 补尾）。**kernel**（仅分析）：
      `shared_b_dtype_t = e4m3`（`:154`，`sizeof` 相等 `:839`）、TMA tensor map 四相位选择
      （`:697-704/763-770`）、`expert_idx=is_shared()?0`（`:779`）、shared SF 无 per-expert 基址
      （`:788`）、独立 `shared_instr_desc`（`:832-836/859`）、共享 L2 等 `shared_l2_full_count`
      （`:717-720` expected `(SHARED_L2_SHAPE_K/BLOCK_N)*2`）、L1 epilogue **共享项权重恒 1.0**
      （`:1068`；只有路由才重载 top-k 权重 `:1095`）、共享 L1 输出写 `tensor_map_shared_l1_output`
      （`:1225`）、完成写 `shared_l2_full_count`（`:1242-1244`）、L2 epilogue 写 combine 的
      **`kNumTopk` 槽**（`:1287-1288`）、combine 先取共享槽（`:1431-1433/1449-1450`）。
      Test 参考（`test_mega_moe.py:277-302` 的两级 GEMM + `:333` 作 combine 的 `bias`；UTCCP 映射
      `:36-50` 与 `kernel:145-149`）。**H100 实测**：`src/mega_moe_shared_schedule.cu`（**直接 include 真实
      `scheduler/mega_moe.cuh` 编译运行设备代码**）+ `harness/exp_mega_moe_shared_expert.{sh,out.txt}`——
      ① 共享 L1/L2 的 `get_next_task_idx` 枚举**恰好覆盖所有 `(m,n)` 一次**（32/32、64/64，去重相等，每域
      只领自己那半）；② `get_num_max_live_pool_blocks`/`get_num_l1_warmup_waves`（`:17-80`）block∈{8,32,
      128,512} code vs host 逐项一致（live 8/32/114/306）、`is_shared_l2_early`（`:443-444`）真机常量判定；
      ③ host binding 5 组配置（shared∈{0,1,2}，fp8/bf16）**逐字段 ALL OK**（`shared_l1_acts is x`、shared
      SF 行数 `ceil_div(Tmax,8)*128` + 列主序、`shared_l2_acts=(Tmax,E_s·I)`、shared=0 时三 view 全
      undefined）；④ UTCCP 映射 device vs host 8 个 BLOCK_M **逐点 0 不符且单射**；⑤ 真实 SM90
      `bf16_gemm_nt`：共享前向 vs fp64 relerr **3.447e-3**、combine `Σw_k·routed_k + shared`（无权重、
      只加一次）relerr **3.448e-3**、反例（共享项重复 topk 次）**8.747e-1**。配套博客
      `content/posts/mega-moe-44-mega-moe-shared-expert`（索引页新增第 44 篇）。

### P35 增量（opt-in，DeepGEMM 里两个「cuBLASLt-only」的窄接口）

- [x] **P35-1** **Muon / 对称矩阵 batched 算子 `batched_syrk` / `batched_symm`**
      （`csrc/apis/gemm.hpp:791-832`、`csrc/apis/einsum.hpp:128-148`、
      `csrc/jit_kernels/impls/smxx_cublaslt.hpp:14-25/137-164/223-234`、
      `csrc/utils/layout.hpp:19-40`、`tests/generators.py:176-191`、`tests/test_bf16.py:247-320`）。
      第 27 篇讲了「什么时候把稠密 GEMM 交给 cuBLASLt（可回退）」，本篇补**另外两个从头到尾就是
      cuBLASLt-only 的 API**（无自研 kernel、无回退、不进选核启发式）：`deep_gemm.batched_syrk`
      （`D=A·Aᵀ`）与 `deep_gemm.batched_symm`（`D=A·B`，A 对称），面向 **Muon 优化器的
      Newton-Schulz 迭代**。**H100 可跑，真机实测**。
      → `docs/45-muon-syrk-symm.md`（真实行号核对）：**三分类表**（可回退 / 无条件 / cuBLASLt-only）；
      形状来源 `generators.py:176-191`（5120/7168 真实模型、batch 上限 16、`rows=min(m,k)`/`cols=max`）；
      host 三件套 `get_cublaslt_batched_view:791-794`（2D→3D unsqueeze）、`batched_syrk:796-810`
      （同一 `a_3d` 当 lhs/rhs）、`batched_symm:812-832`（**内部 `b_3d.transpose(1,2)`**、对转置后 B
      重新取 major、A 对称仅契约不校验）；**放宽 batch 连续性**——`get_major_type_ab<false>` /
      `check_major_type_cd<false>`（`layout.hpp:19-28` 的 `if constexpr` 跳过 `stride(0)==rows·cols`）
      + cuBLASLt `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`（`smxx_cublaslt.hpp:14-25`）；
      翻译层 `cublaslt_batched_gemm:137-164`（`trans_a` 由 `b_major`/`trans_b` 由 `a_major` 决定 =
      操作数交换 `(A Bᵀ)ᵀ=B Aᵀ`、batch offset 取 `stride(0)`、`accumulate=false`）；
      **计算类型固定 `CUBLAS_COMPUTE_32F_FAST_TF32`**（`:45-46`，只有 block-SF NVFP4 才用纯 FP32）；
      测试形态（major × batch_shape × padding 三枚举、symm 显式 `a+aᵀ`、threshold 6e-7）；
      相邻的 `bhd,bhr->hdr`（`einsum.hpp:128-148` 无条件 cuBLASLt、`:146` TODO determinism、
      `smxx_cublaslt.hpp:223-234` `m=r,n=d,k=b`/`trans_b=OP_T`/输出 R_32F）。H100 实测
      `harness/exp_muon_syrk_symm.{py,sh,out.txt}`：① 真实形状 ×{bf16,fp32}——syrk bf16 峰值
      **529 TFLOPS**（512×512×32768）典型 ~355–380、fp32 ~180，symm bf16 ~350、fp32 ~156；
      ② relerr bf16 ~1.4e-4–8.6e-4（输出舍入）、**fp32 ~1.0e-4–2.9e-4（TF32 尾数，非全精度）**；
      ③ 布局/形态：K-major 与 MN-major 都对、**2D 与 batch=1 的 3D 逐位相等**、连续/非连续 batch
      （`stride(0)=16809999`）都算对；④ 四类非法输入（D 非行主序 / 形状 / dtype / batch 数）全拒；
      ⑤ profiler 抓 `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT`（syrk）/ `..._256x128_64x4_1x2_h_bz_coopA_NNT`（symm）；
      ⑥ `einsum("bhd,bhr->hdr")` `calc_diff=3.33e-16`/relerr 2.14e-8、两次逐位相等、266.6us/8 TFLOPS/2020 GB/s
      （K=batch 极小 → 访存受限）。配套博客 `content/posts/mega-moe-45-muon-syrk-symm`（索引页新增第 45 篇）。

### P36 增量（opt-in，补齐前面各篇一直当术语用的 GEMM 入口契约）

- [x] **P36-1** **GEMM API 的输入主序（nt/nn/tn/tt）与输出累积（early_return）契约**
      （`csrc/apis/gemm.hpp`：`early_return`、fp8_fp4/bf16 的 nt/nn/tn/tt、m_grouped 的 nn 变体、
      k_grouped 的 tn/nt；`csrc/utils/layout.hpp`：`major_check`/`get_major_type_ab`/
      `check_major_type_cd`/`fp8_fp4_requires_k_major`/`check_ab_fp8_fp4`）。
      第 10/12/13/22/27/28/32/35 篇反复用 `nt`/`K-major`/`C/D 累积`/`compiled_dims` 却从没系统讲过，
      本篇把 GEMM 入口层单独拆透。**H100 真机可跑**。
      → `docs/46-gemm-major-accumulation.md`（真实行号核对）：**`nt` 是唯一实现，`nn/tn/tt` 全是
      「把操作数（含 SF）转置后转调 `nt`」的薄壳**（`gemm.hpp:135-179`/`:537-565`）；主序判定一句话
      `stride(-1)==1 ? K : MN`（`utils/layout.hpp:29-33`，`major_check :18-27` 要求两维之一连续、
      3D 默认 batch 连续）、**输出强制 N-major**（`check_major_type_cd :35-40`）；**SM90 的 FP8/FP4
      必须 A/B 都 K-major**（`fp8_fp4_requires_k_major :42-45` = arch9 恒真 / SM100 仅 FP4×FP4，
      因 SM90 WGMMA 只有 K-major B 路径），推论 `fp8_fp4_gemm_nn` 对连续 `[K,N]` B 在 H100 必被拒
      （`gemm.hpp:94`）；packed FP4 的**逻辑形状随主序翻倍**（`check_ab_fp8_fp4 :66-73`）。
      grouped：`m_grouped_*_nn_contiguous`（`:252-267`/`:625-633`）沿 `(1,2)` 转置 B、不支持 PSUM；
      wgrad 的 K-grouped 在 SM90 是 `nt`+`Major::K,K`（`:373-427`）在 SM100 是 `tn`+`Major::MN,MN`
      （`:320-371`，`:366`），BF16 只有 `tn`（`:676-722`，SM90 `:714` 且 C 必填 `:710`）。
      **`early_return`（`:17-53`）统一输出累积**：m/n==0 空转（`:21-22`）、量化 D 永不累积（`:26`）、
      C==D 原地 / C≠D 先拷（`:27-29/50-51`）、k==0 退化 copy/zero 且清零 SFD（`:40-47`）、FP8 D 必带
      SFD（`:33`）、C 同 dtype/主序（`:35-36`）；`compiled_dims` 默认随入口变（nt/nn=`nk`、tn/tt=`mn`，
      `:842/850/858/866/935/947/953`）。H100 实测 `harness/exp_gemm_major_accum.{py,sh,out.txt}`：
      ① bf16 四入口输出**逐位相等**、`nt` 用 `.T.contiguous().T` 造 MN-major A/B 也逐位相等；
      ② `fp8 nt` relerr 7.0e-4（SF 全 1 隔离纯 FP8 舍入）、`fp8 nn` 被拒在 `gemm.hpp:94`；
      ③ m_grouped nt/nn B=K/MN 都算对且逐位相等；④ 累积契约：分离 C 与原地 C==D **逐位相等**、
      k==0 全零/D==C、m/n==0 空转、C dtype/主序不符被拒；⑤ profiler 抓 `sm90_bf16_gemm_impl` 的
      主序模板实参 `(K,K)/(K,MN)/(MN,MN)/(MN,K)` 与 `compiled_dims` 化的维 `(0,384,256)/(512,384,0)`，
      默认走 cuBLASLt `nvjet_sm90_tst_..._TNT`（第 27 篇）。配套博客
      `content/posts/mega-moe-46-gemm-major-accumulation`（索引页新增第 46 篇）。

### P37 增量（opt-in，补上 DeepJIT 缓存键「第四段」与后端抽象）

- [x] **P37-1** **DeepJIT 的 post_hook 后处理机制 + 磁盘产物（meta.json / cubin）+
      CUDA↔Ascend 后端抽象 + Python API**
      （`third-party/deep_jit`：`backend/cuda/{options,backend}.hpp`、`runtime/{runtime,config}.hpp`、
      `cache/{disk,memory}.hpp`、`python_api.hpp`、`backend/ascend/{options,backend}.hpp`；README §Post hook）。
      第 33 篇把 DeepJIT 的编译/缓存/加载/发射讲清了，但缓存键「第四段」`post_hook` 一直是空列表项。
      本篇把它拆透，并补齐同属该层的后端抽象与 Python API。**架构无关，H100 真机可跑**。
      → `docs/47-deepjit-posthook.md`（真实行号核对）：① **post_hook 是什么**——`nvcc` 产出 cubin 后、
      写 meta/提交缓存**之前**，跑 `python <abs hook> <abs cubin>` 子进程并**就地改写** cubin
      （`cuda/backend.hpp:132-138`），相对路径由 `Config::python_library_root` 解析（`config.hpp:48-50`），
      是 DeepJIT 唯一的「发布前注入用户代码」接口，DeepGEMM 默认不用（`jit.hpp` 只加 nvcc flags）；
      ② **进键方式**——`get_post_hook_hash`（`cuda/options.hpp:141-156`）= `FNV1a(相对路径 + 文件内容)`，
      且**按绝对路径线程内缓存**（`:146-149`，改 hook 要重启进程），无 hook 返回空串（`:143`）；
      ③ **override 语义**（`:90-91`）——`nullopt` 是「继承」不是「关闭」、空串是「已配置」（会导致读目录报错），
      默认设了 hook 就无法 per-kernel 关闭；④ **磁盘产物与发布顺序**（`disk.hpp:111-133`/`:42-67`，
      `cuda/backend.hpp:160-167`）——hook 在 meta/commit 之前，故「hook 改的 cubin 即最终产物」「失败 hook
      不发布」「命中不重跑 hook」三条均由顺序推出；⑤ **后端抽象**——`Runtime<Backend>` 只用五个类型别名 +
      `load`/`compile`/`compiler_info`；CUDA 的 CompilerInfo 哈希是 nvcc 版本（`cuda/backend.hpp:39-41`），
      Ascend 是 bisheng+ld.lld 两个版本（`ascend/backend.hpp:37-39/64-71`）、两步编译（`:100-124`）、
      `-save-temps` dump（`:126-139`）、发射用 `num_blocks`/`ubuf`（`ascend/options.hpp:129-158`）；
      **`post_hook` 是 CUDA 专属**（Ascend `get_post_hook_hash` 恒 `""`，`ascend/options.hpp:112`）；
      ⑥ **Python API**——`register_python_api:12-20` 注册 `Runtime`（`module_local`）+ `get_jit()`（触发
      LazyInit），README `:18-21` 的两个 WIP（cache warmup from history / Python compile API）当前不可用。
      H100 实测 `harness/exp_deepjit_posthook.{sh,out.txt}`（源码 `src/deepjit_posthook.cpp`，`g++ -std=c++20`
      直接 include 真实头文件、链接 torch+libpython，真跑 DeepJIT 代码）**8 组 ALL PASS**：基线产物 4/4 且
      `meta.post_hook=null`；hook 改变 digest（`068c54b3…≠9a0a899a…`）且副作用落地、`meta.post_hook` 记录路径；
      同进程改 hook 文件哈希不变（`threadlocal_cached=1`）、新路径→新键；`nullopt` 继承=1 / 空串已配置=1；
      失败 hook 抛异常且 `published_hit=0`；破坏 hook 使产物 cubin 头=`HOOK` 且 `load_threw=1`；
      `compile()` 加载+`launch` 后副作用仍 1 行、清内存缓存磁盘命中仍 1 行（不重跑）；跨进程冷读取
      hookhash `73360241…` vs `5dcc60df…`、digest `5bb29276…` vs `e11ef00e…` 均不同。行号全部核对过。
      配套博客 `content/posts/mega-moe-47-deepjit-posthook`（索引页新增第 47 篇）。Hugo 构建 0 ERROR。

### P38 增量（opt-in，补上贯穿所有 kernel 的全局运行时/编译开关面）

- [x] **P38-1** **DeepGEMM 的全局运行时/编译开关面**（`csrc/apis/config.hpp`、`HeuristicsRuntime`、
      `Runtime`、`deep_jit::Env`、`setup.py` 的 `persistent_envs`）。前面 47 篇把一个个 kernel 拆得很细，
      但顶层那组**每个 kernel 都读的可变全局状态**从没单独讲过。本篇拆透：两份 singleton——
      `Runtime`（`num_sms`/`tc_util`/cuBLASLt handle+workspace，`runtime.hpp:20-102`）与
      `HeuristicsRuntime`（`deterministic`/`ignore_compile_dims`/`block_m/n_multiple_of`/
      `mk_alignment`，`heuristics/runtime.hpp:10-67`），经 `config::register_apis`（`config.hpp:8-48`）
      平铺成 `deep_gemm.set_*`/`get_*`。**七个开关的「状态→消费点→为什么」**：
      ① `num_sms` 是上限且 `0`=懒取设备数（`:84-85`），改小会在 `is_cublaslt_available`（`:89-91`）
      **顺带关掉 cuBLASLt 回退**（第 27 篇三与门第二项）；② `tc_util` 的 `0` 在 `:99` 当默认 100、
      越界 `:94` 被拒，只有 SM100 BF16 主 kernel 消费（第 36 篇）；③ `use_deterministic_algorithms`
      是三个正交落点——关 cuBLASLt 回退（`gemm.hpp:519`/`einsum.hpp:81,115`）+ 禁 Mega Gate split-K
      （`mega_gate.hpp:111,126`，注释即 batch invariance）+ 钉死 Mega mHC splits
      （`mega_mhc.hpp:15-17`）；④ `block_size_multiple_of` 只改候选步长（`sm90.hpp:40`、
      `sm100.hpp:55,70-72`，swap/非 swap 分支分别用 m/n）；⑤ `mk_alignment` 默认 128
      （`runtime.hpp:12`），理论值非 arch10 恒 128 / SM100 256（`:57-64`）；⑥ `ignore_compile_dims`
      经 `get_compiled_dim`（`runtime_utils.hpp:31-40`）把全部编译维置 0（一核多形状）；
      ⑦ `set_pdl` 落到 `deep_jit` 发射属性 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
      （`deep_jit/.../kernel.hpp:148-151`，kernel 内 `cudaGridDependencySynchronize`）。
      另拆 `DG_*` 环境变量的两级前缀解析（`deep_jit/utils/env.hpp:56-84`：`DG_` 优先、`DJ_` 兜底，
      `DG_JIT_DEBUG=0` 可盖 `DJ_JIT_DEBUG=1`）与安装时固化的 `persistent_envs`
      （`setup.py:148-155` → `deep_gemm/envs.py` → `__init__.py:6-13`）。
      **架构无关，H100 真机** `harness/exp_runtime_switches.{py,sh,out.txt}` **25 PASS / 0 FAIL**：
      越界/回读语义全对；profiler 抓到 `set_num_sms(64)` 进模板实参 `64u`、`deterministic` 把
      `bf16_gemm_nt`/`einsum` 从 `nvjet` 切回 `sm90_bf16_gemm_impl`；`DG_PRINT_CONFIGS` 子进程抓到
      `block_size_multiple_of=(256,256)` 使 block_n 128→256、`set_num_sms(32)` 进 `LaunchConfig`；
      模板实参 `ignore_compile_dims` 由 `<0,4096,4096>` 变 `<0,0,0>`。行号全部核对过。
      配套博客 `content/posts/mega-moe-48-runtime-switches`（索引页新增第 48 篇）。Hugo 构建 0 ERROR。

### P39 增量（opt-in，补齐所有 SM90 kernel 共同的 MMA 底座）

- [x] **P39-1** **SM90 WGMMA 层：共享内存描述符、选择器与异步组**
      （`deep_gemm/include/deep_gemm/mma/sm90.cuh`、`deep_gemm/include/deep_gemm/ptx/wgmma.cuh`、
      `cute/arch/mma_sm90_desc.hpp`、`cute/atom/mma_traits_sm90_gmma.hpp`）。
      前面第 12/13/16/19/29/30/32 篇把每个 SM90 kernel 都拆了，但**它们每一行 MMA 都在调同一套底座**
      ——描述符构造与 WGMMA 选择器/异步组——却从没单独讲过。本篇补上。**描述符与选择器都是 SM90
      设备代码，H100（sm90a）真机直接跑真实 WGMMA 指令**。
      → `docs/49-sm90-wgmma-descriptor.md`（真实行号核对）：64 位 `GmmaDescriptor`
      （`cute/arch/mma_sm90_desc.hpp:80-126`：`start_address`[0,14)、`LBO`[16,30)、`SBO`[32,46)、
      `base_offset`[49,52)、`layout_type`[62,64)=INTERLEAVE0/B128/B64/B32 `:48-53`）；
      `make_smem_desc`（`mma/sm90.cuh:194-209`，默认 LBO=0/SBO=1024）；
      `to_gmma_layout_type`（`:222-235`，把字节 swizzle 0/16/32/64/128 编码进 layout_type）；
      `get_inner_block_atom_size`（`:211-214`）/`get_gmma_desc_stride_k`（`:216-220`，K-major=1/MN-major=atom）；
      `make_gmma_desc`（`:243-279`）两分支——**K-major 断言 `kSwizzleMode==BLOCK_K*sizeof`**、
      `SBO=8·BLOCK_K·sizeof/LBO=0`（`:249-259`），**MN-major** `SBO=8·atom/LBO=BLOCK_K·atom`、
      interleave 时交换（`:260-278`，`:265` 断言 mn_idx 落在原子边界）；
      `advance_gmma_desc_lo`（`:237-241`）只改低 32 位的三段步进（K 步进 `k_idx*stride_k`、多 warpgroup
      的 `mn_idx*BLOCK_K`、合并 stage 原子偏移 `atom_k_idx*BLOCK_M*BLOCK_ATOM_K`，见
      `sm90_bf16_gemm.cuh:59/219-220/266-270`）。三个选择器 `FP8MMASelector`（`:32-75`，`K=32`）/
      `BF16MMASelector`（`:101-148`，`K=16`，主序进模板）/`TF32MMASelector`（`:168-192`，`K=8`），
      `M=64`、`kNumAccum=M*N/128=N/2`。`ptx/wgmma.cuh:7-23` 的 `arrive/commit/fence_operand/wait<N>`
      异步组语义（`wait<N>`=未完成组数≤N）。fp32 累加器 `CLayout_64xN`
      （`mma_traits_sm90_gmma.hpp:433-435`）的寄存器→(m,n) **列主序**映射（`offset=n*64+m`）。
      **H100 实测** `src/wgmma_desc.cu` + `harness/exp_wgmma_desc.{sh,out.txt}`（真实头文件，
      sha256 `mma/sm90.cuh 548fe0c6…`/`ptx/wgmma.cuh dad341b3…`，**总判定 ALL PASS**）：
      ① 描述符位域 **13 项**全对（`make_smem_desc` 的 INTERLEAVE/B128/显式 LBO·SBO、`make_gmma_desc`
      K-major/MN-major/sw16、`advance` 的 +2/+0x80/+0x200/+0x20、`>>4` 截断）；② 选择器静态属性
      `BF16 K=16`/`FP8 K=32`/`TF32 K=8`、stride_k、atom；③ **真实 WGMMA 数值 5 用例逐位相等**
      （BF16 N=128/BLOCK_N=64、FP8 N=128/BLOCK_K=128、A 为 MN-major、分两批 `commit`+`wait<1>`），
      手工铺 TMA 128B swizzle（`chunk'=chunk XOR (row%8)`）——同时坐实描述符与 swizzle 物理布局对应、
      `advance` 公式、累加器映射三者。配套博客 `content/posts/mega-moe-49-sm90-wgmma-descriptor`
      （索引页新增第 49 篇）。

### P40 增量（opt-in，补齐 singleton 只讲「状态」没讲「生命周期」）

- [x] **P40-1** **DeepGEMM 运行时对象的生命周期与资源所有权**
      （`csrc/runtime/runtime.hpp`、`csrc/apis/config.hpp:12-14`、`deep_gemm/__init__.py:121-125`、
      `third-party/deep_jit/include/deep_jit/utils/{lazy,env}.hpp`、
      `csrc/jit_kernels/impls/smxx_cublaslt.hpp:73-77`、`csrc/runtime/jit.hpp`、
      `csrc/jit_kernels/heuristics/runtime.hpp:67`、`tests/test_lazy_init.py`）。
      第 48 篇（P38）只把 `Runtime` 当「装着 `num_sms`/`tc_util`/handle 字段的 singleton」，
      本篇补它**什么时候创建、什么时候销毁、资源归谁、退出顺序为何危险、`shutdown()` 做了什么**。
      **与 UMMA/TMA/WGMMA 无关，H100 真机逐项实测**。
      → `docs/50-runtime-lifecycle.md`（真实行号核对）：`LazyInit`（`lazy.hpp:15-38`）三性质——
      首次访问才构造（`operator->` `:26-32`）、**null factory = 终态**（`LazyInit(std::nullptr_t) :21`
      + `:27` 断言）、**可整体替换即析构**（`shared_ptr` 覆盖 → `~Runtime()`）；`shutdown()`
      （`config.hpp:13`）只是一行 `runtime = LazyInit<Runtime>(nullptr)`，故**终态、幂等、`_C.init`
      不可恢复**（`init` 只调 `init_jit`，`jit.hpp:14-32`；`jit`/`heuristics_runtime` 是另外两个独立
      singleton，`jit.hpp:12`/`heuristics/runtime.hpp:67`，不被动）。`Runtime`（`runtime.hpp:20-102`）
      两类资源：① **cuBLASLt handle 默认自建**（`:41-43` 注释：`at::cuda::getCurrentCUDABlasLtHandle`
      部分 PyTorch 版本 CPU 开销大）——`cublasLtCreate`（`:50-51`）在 ctor、`cublasLtDestroy`
      （`:54-57`，`noexcept(false)`）在 dtor，`get_cublaslt_handle :59-65` 按
      `DG_USE_PYTORCH_CUBLASLT_HANDLE` 切换托管/自建；② **workspace 每 stream 一份**（key 是
      `c10::cuda::CUDAStream`，`:28,72-74`；注释 `:27`「cuBLASLt 可能异步用，不能共享」），
      16 MiB（注释 `:25`：>16 MiB 会让 cuBLAS 选更差 heuristic），`DG_USE_TEMP_CUBLASLT_WORKSPACE`
      （`:45-48,69-70`）每次现开现弃以避开 compute-sanitizer 的 `cudaErrorCudartUnloading`。
      **退出顺序**：`__init__.py:123-125` 的 `atexit.register(shutdown)`——「原生 cuBLASLt 退出回调
      可能早于 C++ 静态析构释放 handle 存储」，故在 Py/PyTorch/CUDA 尚存时提前释放，规避
      use-after-free；上游 `test_lazy_init.py:22-52` 的 LD_PRELOAD 探针正是结算这件事
      （6 组寿命用例 `:85-92/:101-113`，退出时 `live_handles==0` 才 `DG_PROBE_CLEAN`）。
      `DG_USE_PYTORCH_CUBLASLT_HANDLE` 走 `get_env<int>`（`env.hpp:46-47`），故 `true/false/yes/no`
      **反而非法**（`DJ_PANIC`）。H100 实测 `harness/exp_runtime_lifecycle.{py,sh,out.txt}`
      **24 PASS / 0 FAIL**：A) 探针 6 例 create/destroy/clean 全对（`owned` 1/1、`managed` 0/0、
      `explicit` 1/1、`unused` 0/0）；B) `shutdown` 后 `get_tc_util`/`get_num_sms` 抛
      `lazy.hpp:27 factory != nullptr`、连调两次幂等、`_C.init` 仍报错；C) 显存账
      **16/16/32 MiB**（首用 +16、同 stream 复用、换 stream 再 +16；temp 模式恒 0）；D) 整数解析
      `0/1/2/-1` 与 `false/true/abc` 被拒；E) 上游 `test_lazy_init.py` 3 tests OK。行号全部核对过。
      配套博客 `content/posts/mega-moe-50-runtime-lifecycle`（索引页新增第 50 篇）。

### P41 增量（opt-in，补上 Mega MoE 一直缺席的「验收测试即正确性规范」）

- [x] **P41-1** **Mega MoE 的端到端测试与逐位正确性判据**（`tests/test_mega_moe.py`；
      `deep_gemm/utils/math.py:26-52`；`deep_gemm/utils/dist.py:38-66`）；
      `csrc/apis/mega_moe.hpp:21-31`；`impls/sm100_fp8_fp4_mega_moe.cuh:145-149`）。
      第 40 篇拆了测试基座（`bench_kineto`/`calc_diff`）、第 41 篇拆了操作数生成器，本篇补上
      Mega MoE **唯一的端到端验收测试**——它是「正确性判据（逐位相等）」与前面各篇反复引用的
      「TFLOPS/HBM/NVLink/overlap 数字」的出处。**端到端 fused 对比 SM100 专属，仅分析**
      （oracle 还依赖 DeepEP，本机 NCCL 受阻）；**测试侧变换/判据/参考数学/账目公式架构无关，
      H100 真机实测**。
      → `docs/51-mega-moe-test-correctness.md`（真实行号核对）：**一次 launch 比两套实现**——
      `run_fused`（单 mega-kernel）vs `run_baseline`（DeepEP dispatch → `m_grouped` GEMM ×2 →
      tilelang SwiGLU → combine），**`torch.equal` 逐位验收**（`:345-346`，非 `calc_diff`：重叠只是
      调度变换、任何误差都是 bug）× `use_deterministic_algorithms(True)`（`:77`）；正确性对比被
      `is_legacy_loaded`（`:15-33` 需 DeepEP+tilelang_ops）门控（`:339`），故 oracle 依赖 DeepEP。
      三个测试侧 SF 布局变换：① `_cast_fp8_for_mega_moe`（`:53-59`）量化 + TMA 列主序对齐
      （`per_token_cast_to_fp8` packed -> `(mn,k/32/4)` int32、`x_sf_tma` stride `(1,align(mn,4))`）；
      ② `_to_shared_mega_moe_sf_layout`（`:36-50`）共享专家的 **UTCCP 4×32 交错**（与 kernel
      `transform_sf_token_idx` `:145-149` 逐项等价）+ 按 `block_m` 分块补零；③ `_copy_fp8_sf`
      （`:62-70`）padding 行**广播最后一行**。性能账目（`:368-401`）：FLOP 系数 **3=L1 gate+up+L2**、
      HBM「权重+激活读写+输出」、`routed_weight_elem_size` 随 mma_type（bf16/2、fp8/1、**fp4/0.5**）、
      NVLink `M_recv·H·3`、`t_reduction=M·H·2(1+topk)/6.5e12`（**经验常数**）、
      `overlap_factor=elapsed/(elapsed-t_reduction)`。H100 实测 `harness/exp_mega_moe_test.{py,sh,out.txt}`
      （**直接 import 真实 test 文件复用 helper**）：**37 PASS / 0 FAIL**——SF 变换对 4 组
      `block_m∈{8,64,128,240}` 与 kernel 公式逐点相等/padding 全零/列主序；逐位判据用真实 SM90
      `m_grouped_bf16_gemm_nt_contiguous`（两次逐位相等、改 1 元素即不等、组内重排对应行不变）；
      单 rank 参考数学朴素循环 vs 分组向量化 fp64 `relerr 2.7e-16`、共享项无权重只加一次
      （错乘 topk 权重和 relerr 0.687）、masked/clamp 生效；账目公式三 mma_type 复算、overlap 单调、
      实测归约读带宽 ~2.6 TB/s。配套博客 `content/posts/mega-moe-51-mega-moe-test-correctness`
      （索引页新增第 51 篇）。

- [x] **P42-1** **DeepGEMM 的 kernel 发射层：PDL / cluster / 动态共享内存与设备端 grid 依赖同步**
      （`third-party/deep_jit/include/deep_jit/backend/cuda/kernel.hpp:92-170`、`options.hpp:178-233`、
      `runtime/runtime.hpp:31/42/71-73`、`csrc/apis/config.hpp:27-31`；device 侧 24 处
      `cudaGridDependencySynchronize()`）。
      → `docs/52-kernel-launch-pdl.md`（真实行号核对）：DeepJIT 的 `Kernel::launch` 用 **driver API
      `cuLaunchKernelEx` + 至多 3 个 `CUlaunchAttribute`**（cooperative / cluster dimension / PDL）；
      `LaunchOptions` 是「全局 `default_launch_options` → 每 kernel 四字段 override」两级 merge
      （`std::optional` 里显式 `false/0` 也算「已指定」），DeepGEMM 每 kernel 只填
      `num_smem_bytes/grid/block/cluster_dim`，只有 `set_pdl` 是用户可全局切换的；`Kernel::launch`
      的 8 个 `has_value` 断言、`MAX_DYNAMIC_SHARED_SIZE` 抬高、non-portable cluster「只设不撤」坑、
      属性数组装配。设备侧 24 处 `griddepcontrol.wait` 总在「barrier/cluster 初始化之后、scheduler
      消费前序输出之前」，而 `cudaTriggerProgrammaticLaunchCompletion`（`griddepcontrol.launch_dependents`）
      全仓库**一次不用**（依赖隐式触发）。**H100 真机（PDL/cluster/dyn-smem 均非 SM100 专属）**：
      PDL 全格（132+132）2.028→1.971 ms（1.03×）、半格（66+66）2.028→**1.042 ms（1.95×）**、
      提前触发 1.042 vs 隐式末尾 2.027、缺 `griddepcontrol.wait` 静默读 stale **1,689,600/104857600**
      （有则 0）；cluster cs=2/4 rank/size bad=0、cs=16 无属性 error 912→设属性成功；100 KiB 动态
      smem 无属性 invalid argument→设属性成功；`LaunchOptions` 真实头文件 host 测试 ALL PASS。
      配套博客 `content/posts/mega-moe-52-kernel-launch-pdl`（索引页新增第 52 篇）。

### P43 增量（opt-in，补上 EP 反向通信的空档）

- [x] **P43-1** **DeepEP 的 EPLB 冗余专家：权重预取 `lb_prefetch_weights` 与梯度归约
      `lb_reduce_grads`**（`deep_ep/include/deep_ep/layout/ep/eplb.cuh`、
      `deep_ep/include/deep_ep/impls/ep/{prefetch_weights,reduce_grads}.cuh`、
      `csrc/buffers/ep.hpp:984-1075`、`csrc/kernels/ep/{prefetch_weights,reduce_grads}.hpp`、
      `deep_ep/buffers/ep.py:846-905`、`deep_ep/buffers/allocator.py`、`deep_ep/utils/envs.py:194-235`、
      `tests/ep/test_{prefetch_weights,reduce_grads}.py`；README §Expert load balancing `:330-377`）。
      专题 `docs/07` 讲了 DeepEP dispatch/combine、P4 讲了 MoE 反向 grouped GEMM，二者之间的
      **反向通信**一直空着：动态专家复制（MoonEP / UltraEP）下前向要把主人权重（+量化 scale）
      推到各 peer 冗余槽、反向要把副本 FP32 梯度累加回主人。**两个 kernel 用 NCCL 设备 API
      （`ncclDevComm`/`ncclGin`），本机 NCCL 2.29.7 不提供 → 仅分析**；但 host 公式、负载均衡
      切块、归约语义架构无关，H100 跑**真实设备代码**。→ `docs/53-deepep-eplb-redundancy.md`
      （真实行号核对）：**数据模型** `redundancy_mapping[r][c]` = 域内专家 id 或 -1（`README:341`）。
      **host** `lb_get_theoretical_num_sms`（`ep.py:846-852`）= `align(ceil(max(nvlink/180,
      nvlink/45)),2)`（每 SM HBM 读/写 180/45 GB/s，`envs.py:194-204`；与冗余映射无关，只给 SM 容量
      上界）+ `num_sms` 对齐偶数（`ep.py:43/65`）；`ep.hpp` 契约（prefetch 多权重 `num_weights<=16`、
      reduce 单权重 FP32、每专家字节 `%32==0`、TMA 对齐 `compiled.cuh:67`）；launch 每 warp 私有
      chunk（prefetch **12 KiB**、reduce **24 KiB**，`hpp:30/31`）+ `cluster_dim=2-(num_sms%2)`。
      **负载均衡切块** `eplb.cuh`：`build_tasks:29-64` 用 `cub::BlockScan` 块内压缩、条件
      `expert_idx/kNumLocalExperts==rank_idx`、任务表按映射顺序、`kNumMaxTasks=2730`（32 KiB smem）；
      `EPWeightIterator:91-140` 用**全 rank 共享一个原子计数器**按 `weight→chunk→task` 顺序窃取，
      权重单调推进（`num_skipped_chunks`）、`chunk_idx%num_tasks` 选任务、尾块 `min(CH,B-off)` 裁剪、
      `src=本地/dst=远端`（`gin.get_sym_ptr<Lsa>`）。**两个 kernel 逐行镜像**：prefetch
      `tma_load(src_ptr)→tma_store(dst_ptr)`（覆盖）、reduce `tma_load(dst_ptr)→
      tma_store_reduce_add_f32(src_ptr)`（**元素级原子加、顺序无关**，故测试用整数 pattern +
      `torch.equal`，`test_reduce_grads.py:31-33`）。H100 实测 `src/eplb_redundancy.cu` +
      `harness/exp_eplb_redundancy.{sh,out.txt}`（逐行原样抽取 `build_tasks`/`EPWeightIterator`，
      后者用等价对称指针桩）：① `build_tasks` **16/16 逐条一致**（含映射长>线程数 8×40=320>128、
      `-1` 空槽、全空/全本地、末 rank）；② iterator 4 warp 并发 **77/77 认领**、src/dst 偏移逐条
      一致、全局 counter 收尾 81=77+4（空领语义）、`bytes[0]%CH=500` 尾块正确；③ host 公式真机
      `nvidia-smi nvlink -s` → `get_nvlink_gbs=430.3 GB/s`、写侧瓶颈 `ceil(430.3/45)=10` →
      **10 SMs**，`align(num_sms,2)` 参数语义逐点对；④ 归约语义 vs 参考 relerr **0.000e+00**、
      反例（覆盖）0.782、`build_tasks` 领槽 10==非空映射 10 且无重复。配套博客
      `content/posts/mega-moe-53-deepep-eplb-redundancy`（索引页新增第 53 篇）。

### P44 增量（opt-in，补齐所有 kernel 共用的设备端数学与工具层）

- [x] **P44-1** **DeepGEMM 的设备端数学与工具层**（`deep_gemm/include/deep_gemm/common/`
      ​`math.cuh` / `utils.cuh` / `types.cuh` / `packing.cuh` / `compile.cuh` / `exception.cuh` /
      `cute_tie.cuh`）。前面 0~43 篇把一个个 kernel 都拆过，但每个 kernel 都在调**同一层没有系统
      讲过**的设备端数学与工具函数。本篇把 `common/math.cuh`（236 行）为主的一层单独拆透。
      **除 `fma2` 的 SM100 快路（`__ffma2_rn`）外全部架构无关，H100 真机跑真实设备代码**。
      → `docs/54-deepgemm-device-math-utils.md`（真实行号核对）：**算术/指针** `advance_ptr:13-16`、
      `ceil_div:19-22`、`align:29-32`（含向下对齐 `kDoCeilAlignment=false`）、`constexpr_gcd/min:39-47`；
      **快速数学** `fma2:57-66`（目标卡 `__ffma2_rn`，sm90 回退两次 `__fmaf_rn`）、`fast_rcp:68-72`；
      **bf16/FP8 打包** `cast_into_bf16x2:75-78`/`cast_into_bf16_and_pack:80-84`（RNE）、
      `get_packed_bf16_amax:88-103`（HMNMX2 树，kNumPacked 必须 2 的幂）、
      `scale_bf16x2_into_fp8x4:106-109`（幂次 scale 乘法精确 + E4M3 打包）；
      **UE8M0 SF 指数** `get_ue8m0_sf_exp:111-135`——**用整数加法的进位**完成「把 amax/quant_max 向上
      取到 2 的幂」（`rounded_exp=(amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits`，
      进位 iff `amax` 尾数超过 quant_max 尾数 1.75/1.5；E4M3 quant_max=448=1.75·2^8、E2M1=6=1.5·2^2，
      地板 `kMinSFExponent`=105/1），`get_ue8m0_sf_inv:137-147`（带偏指数互为倒数、和为 254，
      所以 `SF*SF_inv` 精确为 1）；**归约/扫描** `warp_inclusive_sum:150-159`、
      `cta_exclusive_sum:161-184`（块内互斥前缀和，MQA logits 三个调度器的偏移来源）、
      `Reduce*` 仿函数 `:186-198`（**`ReduceMax/Min<float>` 特化为 `fmaxf/fminf`**，注释 `:187-189`
      说明泛型三元会变 `FSETP+FSEL`）、`warp_reduce:200-221`（**intra = 连续 N lane 组内归约
      （偏移 `2^j`，`N≥2^{j+1}`）；inter = lane 同余 mod N 之间归约（偏移 `N·2^j`）**）。
      `common/utils.cuh`：`PatternVisitor:11-22`、`Vectorized<>::zeros:24-40`、
      `get_num_aligned_tmem_cols:42-50`（TMEM 列取整到 {32,64,128,256,512}，第 25 篇 512 列预算的落点）。
      `common/types.cuh`：`MmaKind:7-11`/`GemmType:13-21`（+`is_m_grouped_contiguous:23-29`/
      `is_k_grouped_contiguous:31-37`，PSUM 变体也归入各族）/`KernelType:39-43`/
      `EpilogueOperatorArgs:45-52`（第 21 篇无状态算子基类）/`kNumDeviceLocalityDomains:54`（第 24 篇）。
      `common/packing.cuh`：`get_smem_pack_factor:8-12`（FP4 每字节 2 元素）。
      `common/compile.cuh`（`DG_IN_CUDA_COMPILATION:5-7` 就是 `math.cuh:56` 守卫的来源）、
      `exception.cuh`（`DG_DEVICE_ASSERT:15-23` 等）、`cute_tie.cuh`（`CUTE_TIE:42-50`）。
      H100 实测 `src/device_math_utils.cu` + `harness/exp_device_math_utils.{sh,out.txt}`（**直接 include
      真实头文件**编译运行，四头 sha256 见输出）**ALL PASS**：① `get_ue8m0_sf_exp` E4M3/E2M1 ×
      FP32/BF16 **四组合 vs 真实 `deep_gemm.utils.math.ceil_to_ue8m0` 全 0/65536**（含 2 的幂、1.75/1.5
      边界、0、均匀随机）、`SF*SF_inv` 恒 1.0；② `get_packed_bf16_amax<8>` vs `abs().amax()` **0/32768**；
      ③ `cast_into_bf16_and_pack` vs torch bf16(RNE) **0/65536**；④ `scale_bf16x2_into_fp8x4` vs
      fp32×2^e→E4M3 **0/32768**；⑤ `fma2`(sm90 回退) vs fp64 单次舍入 **0/131072**、`fast_rcp` max relerr
      **1.192e-7**；⑥ `cta_exclusive_sum`/`warp_inclusive_sum` vs 参考 **0/256**；⑦ `warp_reduce`
      intra/inter 7 组 vs 独立 host 分组模型 **0/32**（组大小与偏移逐条对）；⑧ `get_num_aligned_tmem_cols`
      10 档、`Vectorized` 宽度、`get_smem_pack_factor`、`GemmType` 判定、`ceil_div/align/constexpr_*`、
      `kNumDeviceLocalityDomains` 全对。
      配套博客 `content/posts/mega-moe-54-deepgemm-device-math-utils`（索引页新增第 54 篇）。

### P45 增量（opt-in，补齐一段「公开版无调用点」的休眠观测设施）

- [x] **P45-1** **per-warp `%clock64` 追踪探针 `common/probe.cuh`**（`deep_gemm/include/deep_gemm/
      common/probe.cuh`，91 行；对照 `csrc/runtime/jit.hpp:13-33` 的 JIT flag、`impls/sm100_locality_domain.cuh`
      / 第 24/26/34/52 篇的各处 `clock64` 用法）。前面各篇多次依赖 `clock64` 做时序/标定，DeepGEMM
      自己其实有一套**通用 per-warp trace 设施**，却因 `DG_PROBE_ENABLE` JIT 从不定义而**在公开仓库里
      没有任何调用点**——本篇把这层休眠基础设施单独拆透。**全文件架构无关**（`mov %clock64`/
      `mov %globaltimer`/`st.global`/`add.u32`），H100 真机同源编两个二进制实测。
      → `docs/55-deepgemm-probe-clock64.md`（真实行号核对）：`DG_PROBE_ENABLE` 默认 0（`:8-11`）、
      `kEnabled = DG_PROBE_ENABLE and kNumRecords>0`（`:19`）→ `if constexpr` 整段消除；三个静态量
      `kNumBytesPerRecord=8`/`kNumBytes` 与 `get_num_bytes(num_streams)`（`:17-25`）；**`is_valid_buffer`
      三条安全约束**（`:27-31`：容量偶数 / `base%16==0` / **`base>>32==end>>32` 同一 4 GiB 窗口**）；
      ctor（`:33-38`，关闭时不设 cursor）；`record<off>` 写 `cursor+off*8` **不改 cursor**、`"n"` 立即数、
      「先 advance 再负偏移回填」约定（`:40-50`）；`record_payload` 的 `st.global.v4.b32` + `DG_STATIC_ASSERT(off%2==0)`
      （`:52-61`）；`mark()` 两条 8B store 写 `%clock64`+`%globaltimer` 再 `advance<2>`（`:63-75`）；
      **`advance<N>` 只加低 32 位、单条 `IADD3`**（`:77-88`）。**三处「为什么」**：① 4 GiB 窗口约束的根因
      就是 `advance` 放弃高 32 位进位（伪指针实证：起始 `0x00000001_FFFFFFF0` 加 16 → `0x00000001_00000000`）；
      ② 负偏移回填让 ptxas 保留基址加法、压低打点开销；③ `mark` 用纳秒挂钟标定 SM 周期。
      H100 实测 `src/probe_clock64.cu` + `harness/exp_probe_clock64.{sh,out.txt}`（`probe.cuh` sha256
      `a8c3ba24…`，**总判定 ALL PASS**）：A) `kEnabled<8>=1`/`<0>=0`、`get_num_bytes(3)=192`；B) trace
      记录流 payload 四字精确还原、clock 严格单调、负偏移回填 `r4≥r3 & r5≥r4`、globaltimer 单调；
      B2) `DG_PROBE_ENABLE=0` 时 trace **全零**；C) `mark` 标定 `cycles=20000044/ns=10101088` →
      **1.9800 GHz**（trace 两端 marks 1.9796 GHz）；D) `is_valid_buffer` 六例逐条命中（真指针 1、`+8` 0、
      奇数容量 0、跨 4 GiB 0、窗口内 1）；E) `advance` 伪指针精确命中 `[0x0000000100000000] [0x0]
      [0x0000fff300000008]`、关闭时 cursor 恒 0；F) 纯记录开销 **~5.7 ns/条**（4 warp×4096，开 0.0947 /
      关 0.0017 ms），fma 循环版 +33.6 ns/条（单线程受 idle clock 主导）。配套博客
      `content/posts/mega-moe-55-deepgemm-probe-clock64`（索引页新增第 55 篇）。

### P46 增量（opt-in，补齐所有 SM90 kernel 前面那个 128 字节 tensormap 的来源）

- [x] **P46-1** **DeepGEMM 的 TMA 描述符构造层 `runtime_utils.hpp` 的 `make_tma_*`**
      （`csrc/jit_kernels/impls/runtime_utils.hpp`；`csrc/utils/math.hpp` 的 `get_tma_aligned_size`、
      `csrc/utils/layout.hpp:11` 的 `kPackedFP4`；调用点 `sm100_fp8_fp4_gemm_1d1d.hpp:124-133`；
      `cute/arch/copy_sm90_desc.hpp:255-259` 的 swizzle 映射）。前面 55 篇每个 SM90 kernel 的每一行
      `cp.async.bulk.tensor` 前面都站着一个 128 字节 `CUtensorMap`，却从没讲过它**从哪来、被填了什么**。
      **纯 host、架构无关，H100 真机逐字节验证**。
      → `docs/56-tma-descriptor-layer.md`（真实行号核对）：**本质 = 确定性参数翻译**——
      `get_inner_outer_dims:23-25`（major==K?`(k,mn)`:`(mn,k)`）与 `get_non_contiguous_dim:27-29`
      给出「内层永远连续维」的二分；`aten_dtype_to_tensor_map_dtype:85-100`（e4m3→`UINT8`、
      FP4 unpacked→`ALIGN16B`/packed→`ALIGN8B`）；`mode_into_tensor_map_swizzle:102-117`
      （**0 与 16 都→NONE**、`base=32`→`128B_ATOM_32B`）。`make_tma_2d_desc:119-159` 的三个隐藏规整：
      ① **swizzle 非 0 时无条件把内层 box 覆盖成 `swizzle/elem`**（`:130-131`）；② FP4 的
      ALIGN16B 要求 `gmem_inner%128==0`、ALIGN8B 紧凑态 box 元素数 **×2**（`:133-140`）；
      ③ 全局步长以字节计且 16B 对齐（`:128-129`），外层 stride 取 `get_non_contiguous_dim`。
      3/4/5D 同构展开（`:161/204/249`）。五个角色封装：`make_tma_a_desc:325-344`
      （groups 放 inner、`num_groups>1` 断言 K-major）、`make_tma_b_desc:346-365`（groups 放 outer）、
      `make_tma_cd_desc:367-382`（**恒 N-major**）、`make_tma_sf_desc:384-402`（仅 MN、无 swizzle、
      `gran_k` 折 **4** 的进行 packed UE8M0 分组、`shape_mn` 经 `get_tma_aligned_size` 对齐）、
      `make_tma_weights_2d/3d_desc:297-323`（4D `(k, n_per_domain, num_domains, 1)` 承载 locality
      domain 分域）。**H100 实测** `src/tma_descriptor.cu` + `harness/exp_tma_descriptor.{sh,out.txt}`
      （nvcc 直接 include 真实 `runtime_utils.hpp` + torch，32 项 **ALL PASS**）：
      **Part 1** 16 个包装用例（2D/A/B/C/D/SF/weights × dtype × swizzle × major × groups）与我们
      独立复刻的编码器参数经 `cuTensorMapEncodeTiled` 输出 **逐字节相等**（坐实 wrapper 无隐藏字段操作），
      dtype 枚举 `bf16=9 fp32=7 tf32=11 e4m3=0 fp4_un=14 fp4_pk=13` 全对；
      **Part 2** 真实 `cp.async.bulk.tensor.2d` 载入与 `cute::Swizzle<B,4,3>` 预测布局
      **0/256、0/512、0/1024 全对**，并暴露一条**看代码看不出来、只有真机才现**的约束——
      **swizzle 的 XOR 键取自 smem 的绝对地址 bits[7:10)**，故目的地只满足 128B 对齐不够、
      必须对齐到整个 swizzle 原子（128B swizzle→1024B），否则布局整体平移一个原子行
      （smem_off=+128：按 base=0 预测 mismatch 512/1024、按 base[7:10)=1 预测 0/1024）；
      **Part 3** 128 字节指纹（地址落 byte1-2、dtype 落 byte1/8/9、swizzle 落 byte9、全局维
      byte32/36、box 维 byte55/56/65）——**别手工拼描述符**。`base=32` 的 `128B_ATOM_32B` 在本机
      （H100/驱动 570）被 `cuTensorMapEncodeTiled` 直接拒（`CUDA_ERROR_INVALID_VALUE`，SM100 专属）。
      配套博客 `content/posts/mega-moe-56-tma-descriptor-layer`（索引页新增第 56 篇）。

### P47 增量（opt-in，补齐 Mega Gate 一直缺席的「验收测试即正确性规范」）

- [x] **P47-1** **Mega Gate 的端到端验收测试与路由 oracle**（`tests/test_mega_gate.py`；
      `csrc/apis/mega_gate.hpp`；`deep_gemm/include/deep_gemm/layout/mega_gate.cuh`；
      `deep_gemm/include/deep_gemm/epilogue/sm100_mega_gate.cuh`；
      `csrc/jit_kernels/heuristics/mega_gate.hpp`；`csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp`）。
      第 13 篇（P5-1）拆了 Mega Gate 的**实现**、第 51 篇（P41-1）拆了 **Mega MoE** 的端到端测试，
      本篇补上 Mega Gate **自己的**验收测试——它是「Mega Gate 的输出算什么对」的唯一规范，也是
      第 13 篇反复引用的「~7 万条 top-k 0 不符」判据的出处。**主体 kernel 是 SM100 专属，仅分析**
      （`mega_gate.hpp:159` 断言 `arch_major == 10`）；**路由 oracle 纯 torch、架构无关，H100 真机实测**。
      → `docs/57-mega-gate-test-oracle.md`（真实行号核对）：测试两个入口
      `test_mega_gate_api_contract:256-313` / `test_mega_gate:204-225` 都被
      `@test_filter(get_arch_major()==10)` 门控（H100 整体跳过）；用例矩阵 `enumerate_cases:27-37`
      45 个形状（含 4-wave K 尾块 `(32768,4096,512)`）+ `GATE_CONFIGS:21-24` 11 组。
      **判据 `check_topk_weights:105-173` 逐段**：① 打分 `scores=F.softplus(logits).sqrt():117`——
      kernel `sqrt_softplus`（`epilogue:19-23`）的 threshold-20 折衷与 `F.softplus` **数值完全一致**
      （默认 `threshold=20` 同分支规则）；② 排名 `ranking=scores+ranking_bias:130-135`，image token 用
      image_bias **替换**（非叠加）bias；③ top-k 用**结构无关的 crossing 不等式**验收 `:145-156`
      （选中最小分 ≥ 未选中最大分，容差 `1e-5+1e-5·|ranking|max`），与 sort4 网络/3-bit 游标解耦；
      ④ 出权重 `scores/(sum+1e-20)*1.5:137-139` 对应 kernel `epilogue:210-213`，共享槽恒 `1.0:164-165`；
      ⑤ 判据 **`calc_diff<1e-11:141-143`（非逐位）**因参考 logits 走 `cublaslt_gemm_nt:98-102`；
      ⑥ logical→physical 逆映射 `p<E?p:p-E:121-128` **依赖 `make_physical_map:40-49` 的编号约定**
      （副本槽=`E+logical`，第 0 列直通、第 1 列前 32 行 `E+arange`），kernel 侧选副本
      `(rank+t·23333)%n_dup`（`epilogue:205`）只影响负载、不影响逆；⑦ 四个路由控制
      mask(`:167-171`)/image_bias/fix_routing(`:246-247`，host `mega_gate.hpp:137-138` 依赖
      unmapped)/force_random（无放回采样 `epilogue:218-253`）；⑧ 逐位重复 10 次
      `assert_bitwise_equal:185-191` 的根据是确定性模式**从不 split K**
      （`heuristics/mega_gate.hpp:110-111,126`）。H100 实测 `harness/exp_mega_gate_test.{py,sh,out.txt}`
      **47 PASS/0 FAIL**（直接 import 真实 test 复用 helper；用一个完全正确的 torch 代理 gate 喂
      `check_topk_weights` → `weight_diff=0`，再用扰动反例证明判据能抓错：权重偏 `1e-3`→`calc_diff
      4.995e-7`、偏 `1e-6`→`4.543e-13` 仍过（阈值严而不溢）、换第 K+1 名专家→crossing 触发、
      重复专家→唯一性触发、忽略 image_bias→53.1% token 选择不同、漏归一化→`calc_diff=0.713`；
      另 `sqrt_softplus` vs fp64 `max rel 1.369e-7`（fp32 舍入）、副本分散均衡 `0.50000`）。
      配套博客 `content/posts/mega-moe-57-mega-gate-test-oracle`（索引页新增第 57 篇）。

### P48 增量（opt-in，补齐第 7 篇一笔带过的 combine 归约侧）

- [x] **P48-1** **DeepEP combine 的归约 epilogue**（`impls/ep/combine_utils.cuh`、
      `impls/ep/combine_reduce_epilogue.cuh`、`impls/ep/combine.cuh:145-177`、`layout/ep/token.cuh`、
      `common/ptx.cuh`；`csrc/kernels/ep/combine.hpp:148-198`、`csrc/buffers/ep.hpp:925-950`）。
      `docs/07` 只把 combine 归约侧一笔带过（一句 `combine_reduce` + `use_rank_layout`），
      本篇补上「专家算完后把结果加回源 token」这一步的**逐行实现**——EP 归约侧的落点，
      与 `docs/11` 的 DeepGEMM 自研 combine 成对照。**两段设备代码不依赖 NCCL GIN/UMMA/WGMMA，
      H100 真机可跑真实设备代码**。
      → `docs/58-deepep-combine-reduce.md`（真实行号核对）：**两阶段设计**——主 combine kernel
      做搬运 + 本地归约（`combine.cuh:145-177`），`combine_reduce_epilogue_impl`
      （`combine_reduce_epilogue.cuh:24-143`）可**延迟**到 compute stream 做跨 rank 求和（host 把
      「分配输出 + 启动 epilogue」包成 lambda，`ep.hpp:925-950`；靠 **PDL**
      `cudaGridDependencySynchronize`（`:57-59`）与主 kernel 建立顺序，故 epilogue 里读 metadata
      **不能用 `__ldg`**（`:67`），host `enable_pdl=true`（`combine.hpp:182-189`）对应）。
      **两张布局** `use_rank_layout = allow_multiple_reduction && num_ranks <= num_topk`
      （`combine_utils.cuh:8-18`）→ rank 当外层维 vs topk 当外层维，主 kernel（`combine.cuh:26-27`）
      与 epilogue（`:21-23`）**共用同一 constexpr** 保证缓冲一致。**top-k 去重**：`ptx::deduplicate`
      （`ptx.cuh:544-546`，`__match_any_sync`+`bfind.u32` 取组内**最高** lane）+ `compute_topk_slots`
      （`combine_utils.cuh:41-53`，`__ffs` 从低位挤到前、槽满仍无条件 `exchange` 避免 `BRA.DIV`、
      多出槽置 `-1`）+ `ptx::ldg_with_ge_pred`（`ptx.cuh:223-236`）谓词跳过 `-1` 越界地址。
      **`combine_reduce`（`combine_utils.cuh:55-172`）两条路径**：`enable_hadd_bypass`（`:67-70`）=
      无 bias 且 `kNumValidTopk<=2 or slot[2]<0` 时 **bf16 直加**（`:73-109`，省两次类型转换），
      否则 **fp32 累加一次舍入**（`:110-171`，`ptx::accumulate` 在 sm100 是融合 `add.rn.f32.bf16`
      `ptx.cuh:594-597`、sm90 回退 `:599-601`）；`CombineVecTraits` 用 16B `int4`（`:28-39`，
      32B `longlong4_t` 因性能退化被关 `:34`）。bias 走通用路径（`:54-55/111-114`，承载共享专家/残差，
      README `:279`）；top-k 权重按 `rank` 或 `master_lane_idx` 回写（`:127-141`）。**H100 实测**
      `src/deepep_combine_reduce.cu` + `harness/exp_deepep_combine_reduce.{sh,py,out.txt}`
      （**直接 include 真实头文件编译运行真实设备代码**）**16 PASS / 0 FAIL**：
      ① host constexpr 13 组 + 4 组 unroll + sm90 `int4` 全对；② 直接调真实 `combine_reduce`：
      bypass/general/bias 三组 vs oracle **0/256**，20 万随机 bf16 值对 bypass==通用 **0 处不同**、
      20 万三元组链式 bf16 vs 一次舍入 **44195 处（22.10%）**——坐实旁路为何卡在上界 2；
      ③ 完整 `combine_reduce_epilogue_impl`：**rank-layout**（4 rank/experts32/topk4，含同 rank 重复
      `0,0,2,3`→`{0,2,3}`、全同、互异、单 rank、两两重复五类）输出 **0/1280**、权重 **0/20**、
      加 bias **0/1280**；**topk-layout**（8 rank>topk4）输出 **0/1280**、权重 **0/20**；
      ④ Python 独立 oracle 重写 constexpr + `__ffs` 去重选择逻辑 ALL PASS。配套博客
       `content/posts/mega-moe-58-deepep-combine-reduce`（索引页新增第 58 篇）。

### P49 增量（opt-in，补齐 DeepEP 所有 kernel 共用的设备端公共层）

- [x] **P49-1** **DeepEP 的设备端公共层**（`deep_ep/include/deep_ep/common/ptx.cuh` /
      `common/math.cuh` / `common/utils.cuh` / `layout/common/barrier.cuh`）。第 7 篇拆了
      dispatch/combine 数据流、第 53 篇 EPLB、第 58 篇 combine 归约，但**它们每一行
      `mbarrier`/`cp.async`/`TMA`/`red.release.sys`/`__match_any_sync` 都来自这层，却从没单独拆过**。
      **架构无关内联 asm，H100 真机可跑真实设备代码**；`multimem`/`cvt.rs` 仅语义 oracle。
      → `docs/59-deepep-ptx-primitives.md`（真实行号核对）：八组原语——① **mbarrier 五原语**
      `:49-83`（init+fence / invalidate / arrive / arrive.expect_tx / try_wait.parity + 相位翻转）
      与 DeepEP 特有计数模型：`cp.async.mbarrier.arrive` 发射时 +1、完成时 −1（净零但卡住 mbarrier）
      ⇒ dispatch 的 mbarrier init count 是 **1** 而非线程数（`dispatch.cuh:273/309/353-354`）；
      ② **三条搬运路径**：`tma_load_1d` `:122-134`（`shared::cluster` + `kEvictFirst`）、
      `tma_store_1d` `:136-146`（`bulk_group` + `kEvictNormal`）、`tma_store_reduce_add_f32`
      `:148-156`（`cp.reduce.async.bulk…add.f32` 元素级归约，`reduce_grads.cuh:82` 等）、
      `st_bulk` `:85-101`（SM100 `st.bulk`，sm90 回退）；③ **NVLink 多播 `multimem.*`** `:162-195`
      （`bucket/all_reduce/nvlink.cuh:44-50` 把每 rank NVLink 流量从 O(N) 降到 O(1)）；
      ④ **谓词化访存** `:223-362`（`*_with_{ge,gt}_pred` + `ld.global.nc` + `L1::no_allocate` +
      L2 cache hint）+ `common/utils.cuh:14-75` 的 `iterate_ld_st` 网格步进 + 尾部裁剪；
      ⑤ **内存序阶梯** `:364-478`（`red`/`red_add_rel_{gpu,sys}`/`ld_acquire_sys`/`st_release_sys`/
      `st_relaxed_{gpu,sys}`/`ld_volatile`/`fence_*`）与「跨卡为什么必须 `sys`」；
      ⑥ **`warpgroup_reg_realloc`** `:454-469`（`setmaxnreg`，初始预算 `(65536/t/8)*8`，
      `hybrid_combine.cuh:112/358` 等）；⑦ **warp 内建** `:481-560`（`exchange`/`gather`/`match`/
      `get_master_lane_idx`=`bfind.u32`/`deduplicate`/`warp_{inclusive,exclusive}_sum`/`ffs`）；
      ⑧ **浮点打包** `:562-608`（`fadd4`/`fmul4`/`accumulate` 的 SM100 packed 快路 + sm90 回退、
      `cvt_rs_bf16x2` 随机舍入）。另拆 `math.cuh:25-33` 的 `encode_decode_positive`（`-x-1` 对合，
      「未就绪=0、已就绪=负数」，`dispatch.cuh:128/189-190/226` 跨 rank 计数就靠它）、
      `math.cuh:44-66` 的 `pack2/unpack2`（低位在前）、`layout/common/barrier.cuh:11-22` 的
      `BarrierSignals`（计数 + 双相位）。**H100 真机 41 PASS / 0 FAIL**（`src/deepep_ptx_primitives.cu`
      + `harness/exp_deepep_ptx_primitives.{sh,out.txt}`，**直接 include 真实头文件、跑真实设备代码**）：
      mbarrier 相位翻转、TMA 载入/`cp.reduce.add.f32` 归约写（init+a+b=110 vs 覆盖 3）、
      `cp.async` 计数模型、谓词裁剪 + `iterate_ld_st` 1000 元素全对、`red_add×3`=768、
      **2000 轮 release-acquire 生产者-消费者坏数据=0**、warp 内建逐项对拍、`setmaxnreg.dec` 实跑；
      坐实 `cvt.rs` 在 `sm_90a` 被 ptxas 拒（`Feature '.rs' not supported`）、`sm_100a` 通过。
      配套博客 `content/posts/mega-moe-59-deepep-ptx-primitives`（索引页新增第 59 篇）。

### P50 增量（opt-in，补齐 MQA logits 一直缺席的「验收测试即正确性规范」）

- [x] **P50-1** **MQA logits 的端到端验收测试与正确性 oracle**（`tests/test_attention.py`；
      `deep_gemm/testing/{numeric,utils}.py`；`deep_gemm/utils/math.py`；调用点
      `csrc/apis/attention.hpp:18-44/45-134/136-282/303-442`；kernel
      `impls/sm90_fp8_mqa_logits.cuh:307-310`）。第 19 篇（P9-1）拆过连续/分页主 kernel、
      第 30 篇（P20-1）拆过 SM90 分页调度器、第 22/23/24 篇拆过 SM100 sparse/dense/paged 调度器，
      但「MQA logits 算什么对」的**唯一规范** `tests/test_attention.py`（715 行）一直没被单独拆过。
      **与 Mega MoE / Mega Gate 不同：MQA logits 的 SM90 用例（`fmt='fp8'`、fp32 logits）在 H100 上
      真的能跑**（只有 SM100 的 mxfp4/mxfp8 与 sparse 用例被 `@test_filter` 跳过）。
      → `docs/60-mqa-logits-test-oracle.md`（真实行号核对）：三个入口（`test_mqa_logits:104`、
      `test_paged_mqa_logits:299`、`test_sparse_mqa_logits:477` 带 arch==10 门控 `:476`）与三个
      架构分支 helper（`mqa_logits_formats:45`、`mqa_logits_heads:51`、`quantize_mqa_q:59`）。
      **连续版 oracle** `ref_mqa_logits:78-101`：`out_ij=Σ_h relu(<q_i,kv_j>)·w_{i,h}`、按 head 加权
      **先 relu·w 再对 head 求和**、按 KV 分块只改显存不改结果、`-inf` 掩码落在 `[ks_i,ke_i)`、`cost`
      为有效计数；`generate_ks_ke_tests:106-121` 的 disable_cp/CP 两种 KS/KE 生成；kernel **压缩输出**
      `[seq_len,max_seqlen_k]`（测试 `:185-189` 展开回稠密，kernel 写回偏移 `-seq_k_start`
      `sm90_fp8_mqa_logits.cuh:307-310`）。**判据两层**：`-inf` 掩码逐位（`:194`）+ 20 次逐位自洽
      （`:173-175`）+ `calc_diff` vs 原始参考（fp8 `1e-3`）与 vs **隔离量化**参考（fp8 `5e-6`，
      `mqa_logits_tolerances:74`）——把「实现误差」与「FP8 量化误差」分开。
      **分页版** `ref_paged_mqa_logits:225-253` + 三种 KV cache 融合布局（`kv_cache_cast_to_fp8:270`
      SM90 E4M3+每 token fp32 scale、`_mxfp8:285`、`_mxfp4:256` MXFP4/8+packed UE8M0）+ `to_mqa_weights:29`
      的每行 TMA 16B padding。**稀疏版**逐位判据 `:669`（稀疏块内打分 == 全量 gather）；`make_sparse_kv_block_indices:442`
      同请求保留 ~80% 上一 token 的块。host 契约 `attention.hpp:18-44`（`is_mx_sf==arch10:32`、
      weights dtype `:43`、输出 dtype `:80/192`、分页 block_kv `:201`、varlen 仅 arch10 `:233`）。
      **H100 实测** `harness/exp_mqa_logits_test.py` + `.sh`/`.out.txt`（**直接 import 真实 test 复用
      helper 并调用真实 `test_mqa_logits`/`test_paged_mqa_logits`**）**42 PASS / 0 FAIL**：
      ① 连续/分页真实测试在 H100 跑通（100/191 TFLOPS 连续、46/122 TFLOPS 分页、metadata 4us）；
      ② `ref_mqa_logits` vs 朴素三重循环有效区间 `calc_diff=1.110e-16`、`-inf` 掩码逐位、`cost==sum(ke-ks)==878`；
      ③ `generate_ks_ke_tests` CP `(2048,8192)` 前后段 `ke∈[1024,2048)/[6144,7168)`；
      ④ `to_mqa_weights` 公式逐点（含 `h=5→stride 8`）；⑤ `quantize_mqa_q` fp8 往返 rel `0.0266`；
      ⑥ fp8 KV 布局逐位重构 + scale `==amax/448`、mxfp4/8 往返 `0.1157/0.0269`；⑦ 判据工具行为；
      ⑧ sparse 被 arch==10 门控跳过；⑨ `make_sparse_kv_block_indices` 语义。SM100 主 kernel/调度器
      仅分析（第 22/23/24 篇已用真实设备代码验证其调度器）。配套博客
      `content/posts/mega-moe-60-mqa-logits-test-oracle`（索引页新增第 60 篇）。

### P51 增量（opt-in，补齐 DeepEP dispatch 的输出落盘侧）

- [x] **P51-1** **DeepEP dispatch 的拷贝 epilogue**（`impls/ep/dispatch_copy_epilogue.cuh`；
      `layout/ep/token.cuh`；`layout/ep/workspace.cuh`；`csrc/kernels/ep/dispatch.hpp:168-226`；
      `csrc/buffers/ep.hpp:402-405/666-743`；`deep_ep/buffers/ep.py:161`）。第 7 篇讲了 dispatch/combine
      数据流、第 48 篇（P48-1）拆了 **combine 的归约 epilogue**，本篇补上**对称的另一半**——主 dispatch
      kernel 把 token 搬进对称缓冲后，由**独立的第二 kernel** `dispatch_copy_epilogue_impl` 落盘成下游张量
      （`recv_x`/`recv_sf`/`recv_topk_idx`/`recv_topk_weights`/`recv_src_metadata`），并做专家区分组对齐、
      零填充与链式列表。**关键：整段不碰 NCCL 设备 API**（无 `ncclDevComm`/`ncclGin`），故
      **H100 真机直接 include 真实头文件跑真实设备代码**。→ `docs/61-deepep-dispatch-copy-epilogue.md`
      （真实行号核对）。**要点**：① **两 kernel + PDL**——`:61` `cudaGridDependencySynchronize` 与主
      dispatch 衔接（读前序数据不得用 `__ldg`，`:59-60`）；host 发射 `dispatch.hpp:188-214`（用 smem
      反推 `num_warps`、`enable_pdl=true`）。② **三种纯结构体**——`TokenLayout`（`:10-82`，hidden/sf/
      metadata 三段 32B 对齐，smem 视图多一个 mbarrier 的 32B，故 `<false>=256/<true>=288`）、
      `BufferLayout`（`:84-144`，rank×token 网格 gmem↔smem 视图）、`EPWorkspaceLayout`（`:9-112`，
      `sizeof(EPSignals)=15,831,168`）。③ **五种形态**（四个模板布尔开关 `cuh:12-22`）——非 expand
      （dst=i）/ expand 非 cached（`atomicAdd(psum_per_expert+e,1)`）/ expand cached（读
      `recv_src_metadata[+2+l]`）/ 零填充（`:232-325`，`st_bulk` 清零 smem hidden + `reduce_add`/
      `warp_inclusive_sum`/`__ffs(__ballot)` 定位 `base_e+count_e+local_pad`）/ 链式列表（`:131-136`
      `channel_linked_list[ll[master]]=i`，`:211-230` 尾部 `-1` 哨兵 + 清 `channel_scaleup_tail`）。
      ④ 输出 SF 两种 stride（行主序 vs TMA 对齐列主序，`ep.hpp:688-698`）。**H100 实测**
      `src/deepep_dispatch_copy_epilogue.cu` + `harness/exp_deepep_dispatch_copy_epilogue.{sh,py,out.txt}`
      （**直接 include 真实头文件、构造真实 `BufferLayout<false>` token 缓冲、跑真实设备代码**）
      **21 PASS / 0 FAIL**：布局偏移与字节数逐字段；非 expand 全部输出（含 SF 两种 stride、
      `metadata[2+l]` 24/24 未触碰）；expand 原子分配（专家区内 token 集合正确、计数器到 `base+count`、
      token 行 hidden 正确、padding 行 hidden/SF/权重全零、`metadata[2+l]` 指向区内）；cached
      （严格按 metadata 行、`psum` 不变）；链式列表（写槽 + 尾哨兵 + workspace 归零）。Python oracle
       独立复刻 `TokenLayout` 偏移、expand 对齐分配与零填充 dst 公式（含 `__ffs` 取**最低**满足 lane
       这一关键点）。配套博客 `content/posts/mega-moe-61-deepep-dispatch-copy-epilogue`（索引页新增第 61 篇）。

### P52 增量（opt-in，补齐 DeepEP dispatch 的「确定性」侧）

- [x] **P52-1** **DeepEP 的确定性 dispatch 排序 `EPHandle.deterministic_sort`**
      （`deep_ep/buffers/ep.py:27-205`；`impls/ep/dispatch.cuh:341`；
      `layout/ep/token.cuh:63-77`；`deep_ep/utils/envs.py:184-190`）。第 7 篇讲了 dispatch/combine
      数据流、第 48/51 篇讲了两个 epilogue，但都隐含「接收行序稳定」这一前提。本篇补上把它**抹平**的
      那一层：dispatch 的槽位由**对称缓冲里的 `atomicAdd` 计数器**分配（`dispatch.cuh:341`），
      同一批 token 多次运行的物理行序可变；`EPBuffer(deterministic=True)` 时 dispatch 结束即挂
      `deterministic_sort`（`ep.py:734-743`，`async` 时挂 `after_wait` 钩子）把顺序规范化回
      **源 token 全局索引序**。要点：① `EPHandle`（`ep.py:27-105`）冻结路由——`topk_idx` 有
      `_version` 守卫、`psum_*`/`num_unaligned_*`/`recv_src_metadata` 是排序与反向的输入；
      ② 排序键**恒取自「排序前快照」**（`ep.py:135-140`），因为 cached dispatch 不重生成元数据；
      ③ 非 expand（`ep.py:159-172`）：按 col0 全局排序 + OOB 尾块置 dtype-max + 非 cached 才排
      `recv_src_metadata`，并逆置换更新 `channel_linked_list`；④ expand（`ep.py:174-205`）：
      `expert*1e10 + (-5e9 + src)` 双键**稳定排序**（有效行 < 填充行、块内按 src 升序），
      `scatter_add_` 构键，只排 `recv_x/sf/topk_weights` 并把槽位指针用逆置换重映射；cached expand
      整个跳过。**方法纯 host/Python、架构无关**（只 `torch.sort`/`index_select`/`scatter_add_`），
      故 H100 上**逐行原样抽取真实源码运行**（`ast` 抽取，绕过本机拒绝 `import deep_ep` 的
      `check_nccl_so`；抽取片段 sha256 `22f3026c…` 与 `~/github/DeepEP@93eb6eb` 一致）。
      → `docs/62-deepep-deterministic-sort.md`（真实行号核对）。**H100 实测**
      `harness/exp_deepep_deterministic_sort.{py,sh,out.txt}` **7 PASS / 0 FAIL**：非 expand
      同步置换 + 快照 + 幂等；6 种接收顺序输出**逐位一致**且 = src 升序；OOB 压尾；expand 块内升序
      + 填充殿后 + 槽位指针逆置换自洽；cached 不变；`channel_linked_list` 逆置换；expand 跨接收顺序
      规范化顺序逐位一致。配套博客 `content/posts/mega-moe-62-deepep-deterministic-sort`（索引页新增第 62 篇）。

### P53 增量（opt-in，补齐 DeepEP dispatch/combine/bucket 每次运行都调用的同步地基）

- [x] **P53-1** **DeepEP 的多 rank 屏障与 symmetric-memory 句柄层**
      （`deep_ep/include/deep_ep/comm/barrier.cuh`、`comm/handle.cuh`、
      `layout/common/barrier.cuh`、`impls/comm/barrier.cuh`、`common/compiled.cuh`；
      host `csrc/comm/barrier.hpp`、`csrc/kernels/comm/barrier.hpp`、`csrc/kernels/comm/context.cpp`、
      `deep_ep/comm/{barrier,handle}.py`；用法 `csrc/buffers/ep.hpp:93-106`）。第 7 篇讲 dispatch/combine
      数据流、第 43 篇 EPLB、第 48/51/52 篇三个 epilogue，但每个 kernel **开始与结束**都调的
      `comm::barrier` 从没单独拆过——它是把整个 EP 通信组拉到同一同步点的地基。**三层**：① 对称窗口
      句柄 `NCCLGin`（`handle.cuh:11-255`）——`ncclCommWindowRegister` 的窗口 + `world/lsa/rail` 三域，
      `get_sym_ptr`（`:64-92`）把「本 rank offset」翻译成「目标 rank 同 offset」（对称 = 布局一致、同
      offset 即同逻辑位置），`red_add_rel`（`:96-120`）演示「能 NVLink 就不走 RDMA」的双路径统一；
      ② 屏障协议（`barrier.cuh`）——`BarrierSignals`（计数 + 双相位槽、64B 对齐）、NVLink 腿
      `nvlink_barrier_wo_local_sync`（`:88-131`）的**免复位**（计数器低 2 位选相位/符号 + 每轮对双槽
      `+1/-1`，四调用一循环、两槽自动回 0，$2^{64}$ 计数 ~57 万年溢出）、`timeout_while`（`:30-54`）
      看门狗、`get_qp_mode`（`:56-86`）的 SM↔QP 确定映射；③ 两级域编排 `gpu_barrier`（`:196-252`）——
      先 flush TMA store → grid sync → SM0 走 scaleup(NVLink)、其余 SM 并行走 scaleout(RDMA)。
      host `context.cpp` 的域分解（`:174-183`，**仅 hybrid 且 scaleup 域==NVLink 域时 `is_scaleup_nvlink`**）
      与 `ginSignalCount=num_ranks+4`（`:140`）解释状态预算。`barrier_impl`（`impls/comm/barrier.cuh:11-40`）
      顺序/并行两模式 + host `cooperative=true`。**NVLink 腿逐行抽取 + 对称指针桩在 H100 跑真实设备代码**
      （`src/deepep_comm_barrier.cu` + `harness/exp_deepep_comm_barrier.{sh,py,out.txt}`）：2~128 rank ×
      8 次调用 counter/双相位槽 vs 四相位模型**逐字段 0 不符**、奇数次非零残值坐实 ±1、SM gating
      （sms=2/4 仅 SM0）、`get_qp_mode` 5 组逐字段、`BarrierSignals` 64B——**设备 19 PASS / Python 17 PASS /
      0 FAIL**；scaleout RDMA 腿与并行混合编排为 NCCL GIN 专属，仅源码分析。配套博客
      `content/posts/mega-moe-63-deepep-comm-barrier`（索引页新增第 63 篇）。

### P54 增量（opt-in，补齐 DeepEP 与 EP all-to-all 并列的 bucket 集合通信）

- [x] **P54-1** **DeepEP bucket 集合通信：copy-engine 路径、分块布局与算法选型**
      （`deep_ep/buffers/bucket.py`；`csrc/buffers/bucket.hpp`；`csrc/kernels/bucket/{all_gather,all_reduce,reduce_scatter}.hpp`；
      `csrc/kernels/driver/driver.cpp`；`deep_ep/include/deep_ep/{layout/bucket/*,impls/bucket/*,comm/bucket.cuh}`；
      `deep_ep/utils/envs.py`）。前面第 7/48/51/53/58/59/63 篇全在 **EP 的 all-to-all** 语义下，
      本篇补 DeepEP 里**另一套并列的通信设施** `BucketBuffer`（`all_gather`/`reduce_scatter`/`all_reduce`）
      ——MoE 训练同步专家梯度/优化器状态的原语。**纯 NVLink all-gather 连 SM kernel 都不用**，交给硬件
      copy engine，故架构无关，H100 真机实测；all_reduce/reduce-scatter 内核用 NCCL GIN/multimem 仅分析。
      → `docs/64-deepep-bucket-collectives.md`（真实行号核对）：**三类算子 × 三物理域**（纯 NVLink / 纯 RDMA /
      hybrid）的分派（`bucket.py:236-293`、`bucket.hpp:86`）；**copy-engine 四原语**（`driver.cpp:14-86`）：
      `copy_engine_push=cudaMemcpyBatchAsync`（`attrs.srcAccessOrder=Stream`+`PreferOverlapWithCompute`、
      **必须非 legacy 流**）、`copy_engine_sync/wait=cuStreamBatchMemOp`（`WRITE_VALUE_64`+`WAIT_VALUE_GEQ`，
      GPU 侧阻塞，`kNumMaxBatchMemOps=255` 切分）；**纯 NVLink all-gather 协议**（`all_gather.hpp:62-108`）：
      `CopyEngineSignals.base[kNumMaxNVLRanks]` 每 rank 一槽、`get_sym_ptr` 对称偏移、per-context `seq_idx`
      单调计数（GEQ 故不需同值）、「到达屏障 → push → 完成屏障」两次 sync；**分块层**（`chunk.cuh:12-102`）：
      `ChunkStorage=256MiB`、`ChunkIterator` 固定 stride 恰好覆盖每个 chunk（仅末块可不满）、五种信号结构
      大小（`CopyEngineSignals=8192/AllGatherSignals=512KiB/AllReduceSignals=ReduceScatterSignals=128KiB`、
      `BucketSignals=794688`）；**hybrid all-gather chunk 自选**（`all_gather.hpp:120-148`）与带宽模型
      `get_theoretical_num_sms`（每 SM 读/写 180/45 GB/s、nvlink×0.9）。**H100 实测** `src/deepep_bucket_copy_engine.cu`
      + `harness/exp_deepep_bucket.{sh,py,out.txt}`：① `cudaMemcpyBatchAsync` 64 段任意长/对齐逐字节、空 no-op、
      传 `0` 流报 `invalid argument`；② mem-op write/wait 顺序 + 跨 stream 握手 + 600 个 mem-op 分批；
      ③ **真·多 GPU（每 rank 一块 H100、peer access）跨 NVLink all-gather R=2/4/8 逐字节一致**；④ 单 GPU 多 stream
      模拟 R=2/4 通过、R≥8 互锁（并发限制）；⑤ `ChunkIterator` include 真实头文件跑设备代码 5 组覆盖恰好一次；
      ⑥ 真实编译 `sizeof` 与 Python 推导逐项相等；⑦ 带宽模型 9 域配置×3 算子与独立公式逐点一致、`nvidia-smi nvlink`
      430.3 GB/s。**设备 PASS=20 / Python PASS=21 / 0 FAIL**。配套博客
      `content/posts/mega-moe-64-deepep-bucket-collectives`（索引页新增第 64 篇）。

### P55 增量（opt-in，补齐所有 DeepEP kernel 篇目的「语义规范」层）

- [x] **P55-1** **DeepEP 的 dispatch/combine 参考实现（语义规范）与流量/基准生成器**
      （`deep_ep/utils/refs.py`；`deep_ep/utils/gate.py`；`deep_ep/utils/testing.py`；
      `deep_ep/utils/math.py`；`tests/utils/test_gate.py`）。前面第 7/48/51/52/53/58/59/61/63/64 篇
      一直在拆 DeepEP 的 **kernel**，但它们**「算什么才对」的定义**从没系统讲过——本篇补上这层地基：
      `deep_ep/utils/` 三个**纯 PyTorch** 模块。**架构无关、H100 真机可跑**（`refs.dispatch` 只用
      torch 自带 NCCL 的 `all_to_all_single`，不碰 DeepEP C++/NCCL 设备 API；唯一障碍是
      `import deep_ep` 的 `check_nccl_so()`，用 `sys.modules` 注入合成包 + stub `envs` 绕过）。
      → `docs/65-deepep-refs-and-testing.md`（真实行号核对）：**`refs.dispatch:10-123`** 用「mask +
      定长 `all_to_all_single`」精确复刻 DeepEP 语义——逐 dst rank 收集（`:63-81`，不属于该 rank 的
      top-k 槽改 `-1`，故**同一 token 会发给多个 dst**）、`src_token_global_idx = rank*num_max + local`
      （`:88-90`）、交换计数再交换数据（`:92-110`）、接收侧 `-=expert_start` + 越界 `-1`
      （`:112-117`）；三大不变量（不展开 / 按 src 全局索引升序 / OOB=-1）。**`generate_pre_combine_data:126-153`**
      的素数 `131071` 定种子 + `sin` 合成、**`ordered_accumulate:156-175`** 固定顺序 fp32 累加。
      **`combine:177-243`** 的 `grouped_reduce:201-236`——组内 **stable sort**、沿 top-k **严格升序**
      累加、结果写**组最右槽**、其余置零；三配置 hybrid/non-hybrid/disabled（`:237-240`）与 `bias`
      语义（`:241`，docstring 强调 scaleout==1 时 `reduce_in_scaleup` T/F 仍因 bias 不等价）。
      **`gate.py`**：`generate_rank_count:32-113`（特殊 rank=0、`upper_bound=int((R-1)/ratio)+1`、
      `must_mask=(a==R)` 全 rank 覆盖 token）、`generate_topk_idx:4-30`、precise/random 两种
      `get_unbalanced_scores:176-180`；判据 `tests/utils/test_gate.py:39-40` 的
      `practical_ratio=count[0]/min(count[1:])`、`inequality<1.02`。**`testing.py`**：`flush_l2_cache:14-23`
      （256MB）、`bench:26-62`（event + 丢首样本 `:61`）、`bench_kineto:113-221`（两批取第二批 + 名字子串
      + `ms/us` 换算 + chrome trace 周期聚合 + `EP_USE_NVIDIA_TOOLS` 旁路 `:145`）、`suppress_stdout_stderr:74-110`
      （`dup2` 到 devnull）、`parse_num_bytes:224-239`；`math.py:5-9 calc_diff` 是**余弦相似度型**、
      与 DeepGEMM 的归一化平方误差不同。H100 实测 `harness/exp_deepep_refs.py` + `.sh`/`.out.txt`
      （真实源码路径加载，打印 sha256 `refs c5f551b4…`/`gate 5372e456…`/`testing 9a6f1407…`）：
      ① `refs.dispatch` 8×H100 all_to_all（T=512,E=32,k=4）本 rank 收 1753 行、每来源计数
      `[207,225,206,232,210,222,230,221]`，与**独立 oracle**（纯 Python 按 `(src,local)` 重建）
      **逐项 PASS**（升序 / src / topk_idx / x 逐位 / 计数 / OOB=-1）；② `refs.combine` 五种配置
      （三降约配置 + bias/bias_pair）与独立 oracle **逐位相等**，坐实 `(T,T)≠(T,F)` bias 效应、
      `(F,T)` 被 assert 拒；③ `gate.py` precise 主扫 `E/r∈{4,8}×R∈{8,16}×k∈{1,2,4,6}×ratio∈{1,2,4}`
      **98/98 PASS**（practical 与目标差 <10%、inequality<1.02）；④ `testing.py` `parse_num_bytes` 6 例 +
      4 非法、`bench` 51.4us vs `bench_kineto` 47.7us、`flush_l2` 抖动（共享 GPU 离群点）、
      `EP_USE_NVIDIA_TOOLS` 旁路、`suppress_stdout_stderr` 全 PASS。配套博客
      `content/posts/mega-moe-65-deepep-refs-and-testing`（索引页新增第 65 篇）。

### P56 增量（opt-in，补齐所有 DeepEP kernel 篇目脚下的「对称内存分配地基」）

- [x] **P56-1** **DeepEP 的对称内存分配层：GPU/CPU 弹性段、跨进程 shard 与 copy-engine 窗口的地基**
      （`csrc/kernels/comm/symmetric.hpp`；`deep_ep/buffers/allocator.py`；`csrc/buffers/base.hpp`；
      `csrc/kernels/comm/context.cpp` 的 alloc/window 段）。前面第 7/48/51/52/53/58/59/61/63/64 篇
      全都默认「有一块布局一致、可对称寻址的对称内存」，但**这块内存谁分配、怎么在 GPU 与 CPU 之间
      弹性拼接、怎么跨进程把对端 shard 映射进本地 VA** 从没系统讲过。本篇补这层地基。**纯 host-side
      CUDA VMM + NCCL 主机 API + Linux pidfd，与 UMMA/tcgen05/GIN 无关，H100 真机可跑**——直接
      `#include` 真实头文件调真实类。→ `docs/66-deepep-symmetric-memory-alloc.md`（真实行号核对）：
      三层对齐（TMA 32B / RDMA 64B / **分配 2MiB**，`compiled.cuh:67-71`）；`DeviceContext`（`:21-75`）
      与 `build_alloc_prop` 的四步（PINNED + location、requestedHandleTypes 含 POSIX_FD/FABRIC、
      GPUDirect-RDMA-with-VMM 断言、granularity 整除 2MiB）；`cumem_create_with_fallback`（`:79-93`）
      的 **FABRIC→POSIX_FD 就地清位重试**；三个分配器——`GPUSymmetricMemory`（`:120-135`，包
      `ncclMemAlloc`）、`ElasticSymmetricMemory`（`:139-190`，**reserve 一段连续 VA 后 map「GPU 主段 +
      独立 storage 段」**）、`HybridElasticSymmetricMemory`（`:193-300`，按 rank 依次把对端 shard 贴到
      `num_sym_bytes + i*storage`）；`create_shared_handle`（`:281-299`，`cuMemExportToShareableHandle`
      后 **`cuMemRelease` 靠 fd 续命**）；`context.cpp` 的 window 消费（`:186-231`：`symmetric::alloc` →
      `ncclCommWindowRegister` → `ncclGetLsaDevicePointer` → `get_sym_ptr` 的 offset 翻译）；Python
      `BufferAllocator`（`allocator.py:19-36`：`meta` 占位 + 2MiB 对齐偏移 + `swap_tensors` 落地单块
      storage）。**H100 真机**（`src/deepep_symmetric_alloc.cu`，`sha256(symmetric.hpp)=fa24a233…`）：
      **设备 PASS=55 / Python PASS=23 / 0 FAIL**——① 属性普查 `host_numa=0`、`fabric_supported=1`、
      `gpu_direct_rdma_vmm=1`、granularity **2 MiB**；② **真坑**：属性说 FABRIC 支持，但强制请求时
      `cuMemCreate` 仍被拒 → 回退分支就地清位到 `0x1`（POSIX_FD）；③ Elastic 的 GPU / CPU-NUMA 两种
      backing 都连续 VA、两段物理独立；④ `create_shared_handle` GPU/CPU 导出→导入往返；⑤ hybrid 3 个
      shard 连续映射、偏移布局、不污染主段；⑥ **真·跨进程**（`fork`+`exec` 子进程，**不能 fork-after-CUDA**）
      `pidfd_open`/`pidfd_getfd` → 子写父读、父写子读 magic 往返；⑦ `ncclMemAlloc` 2MiB 对齐 + 非对齐
      /0 字节断言负例；Python 侧真实 `allocator.py` 源码规划/落地/契约/真实 MoE 规模。window/GIN 与
      真实多 rank scaleout 编排为 NCCL 设备 API 专属，仅分析。配套博客
      `content/posts/mega-moe-66-deepep-symmetric-memory-alloc`（索引页新增第 66 篇）。

### P57 增量（opt-in，补齐 Mega mHC 一直缺席的「验收测试即正确性规范」）

- [x] **P57-1** **Mega mHC 的端到端验收测试与正确性 oracle**（`tests/test_mega_mhc.py`；
      `csrc/apis/mega_mhc.hpp`；`deep_gemm/include/deep_gemm/layout/mega_mhc.cuh`；
      `deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh`）。第 5 篇（P5-2）拆了 Mega mHC 的
      **实现**、第 51 篇（P41-1）拆了 **Mega MoE**、第 57 篇（P47-1）拆了 **Mega Gate** 的验收测试，
      本篇补上 Mega mHC **自己的**验收测试——它是「Mega mHC 的输出算什么对」的唯一规范，也是第 5 篇
      反复引用的判据出处。**主体 kernel SM100 专属，仅分析**（`mega_mhc.hpp:223` 断言 `arch_major==10`）；
      **参考实现 / SF 页布局 / 判据 / host 契约架构无关，H100 真机实测**。
      → `docs/67-mega-mhc-test-oracle.md`（真实行号核对）：两个入口都被
      `@test_filter(get_arch_major()==10)` 门控（H100 整体跳过，`:241/:318`）；`enumerate_cases:15-19`
      **28 个形状**（`sf_layout∈{col,extra}×hidden∈{4096,7168}×T∈{1,64,65,200,1025,4096,32768}`，跨
      64/512/1024/4096 边界）×normal/shifted；`make_inputs:22-56` 的「重复 token」装置（`:52-55` 把
      63/64/511/512/T-1 行克隆成第 0 行，专抓 split-K 边界串台）与 `comb_res_mix` 三遍行/列归一化
      （`:49-51`，最后一次列归一化→**列和精确=1、行和仅近似**）；`make_outputs:59-86` 三套 SF 布局
      （`col`=`y_gemm_sf` stride `(1,align(T,4))`、`extra`=`y_routed_sf` 连续 + `y_shared_sf` 页布局，
      存储行数 `ceil(T/block_m)*align(block_m,128)`）。**SF 页布局 `extra_sf_rows:96-104` 与 kernel
      `get_shared_sf_row_idx`（`epilogue/sm100_mega_mhc.cuh:116-122`）逐字等价**（四段一一对应
      `idx&~127` / `(idx&31)*4` / `(idx>>5)&3`；kernel 侧 `:252-256` 用同一 `shared_sf_row_idx`
      同写 `y_primary_sf` 与 `y_shared_sf`）。`mhc_reference:123-210` 是语义权威（`new_residual` 先
      cast bf16 = 共享数值边界 `:148`；`_sinkhorn_reference:114-120` 双随机）；独立 fp64 对拍 relerr
      1.4e-5~2.8e-3。`check_correctness:229-238`（`y_fp8` 阈值 2e-4、其余 5e-5；`comparable:225-226`
      把 int32 SF 视图成 uint8 逐字节比）。`test_mega_mhc_api_contract:241-315` 四条运行期不变式：
      确定性逐位复现（`get_num_splits:15-24` 固定 `kDefaultNumSplits=16`，`layout:36`）、CUDA Graph
      必须先预热（`get_split_barriers:26-39` 的 `:33` 断言 `CaptureStatus::None`，每 stream 一份）、
      FP8-only 调用逐位稳定（`needs_bf16_scratch:229-238`）、多 stream 独立（split barrier 按
      `CUDAStream` 分表 `:29-30`）。性能账目 `:347-357`（`logical_bytes`+`intermediate_io_bytes`
      `=2·y_bf16(+new_residual)`）。H100 实测 `src/mhc_sf_row_idx.cu`（**直接 include 真实头文件**，
      给 `__fmul2_rn` 提供 sm90 标量回退）+ `harness/exp_mega_mhc_test.{py,sh,out.txt}`：
      ① 设备代码 `get_shared_sf_row_idx` vs `extra_sf_rows` 5 block_m×6 T **32 PASS/0 FAIL**（mismatch=0
      且页内单射）；② Python **45 PASS/0 FAIL**——页布局 scatter/gather 逐位还原 + padding 全零、
      fp64 对拍、Sinkhorn 行/列和=1、`calc_diff=1.221e-4` 的对「作 fp8 放行、作 bf16 拒绝」、
      **host 契约 9 条逐条命中 `mega_mhc.hpp` 真实行号**（含「合法调用撞 `:223` arch 闸门」）。
      配套博客 `content/posts/mega-moe-67-mega-mhc-test-oracle`（索引页新增第 67 篇）。

### P58 增量（opt-in，补齐 DeepEP 与 EP all-to-all / bucket 并列的第三类通信原语：流水线并行 send/recv）

- [x] **P58-1** **DeepEP 的流水线并行 send/recv 原语 `pp_send_recv`**（
      `csrc/kernels/pp/pp_send_recv.hpp`、`csrc/buffers/pp.hpp`、`deep_ep/include/deep_ep/impls/pp/pp_send_recv.cuh`、
      `deep_ep/include/deep_ep/layout/pp/workspace.cuh`、`deep_ep/buffers/pp.py`、`tests/pp/test_pp.py`）。
      第 7/58/61 篇拆了 **EP all-to-all**、第 64 篇拆了 **bucket 集合通信**，本篇补上 DeepEP 通信设施里
      第三类：**环形流水相邻 rank、带背压、可重叠的 send/recv**。它没有计数/前缀和/算法选型，关键是
      **几何与协议**。三块：① **host 契约**（`pp.py:12-67` / `pp.hpp`）——`num_max_tensor_bytes` 按
      `kNumRDMAAlignmentBytes=64` 对齐、`num_storage_bytes=align(4·T·F, 2MiB)`、send/recv 断言「相邻 rank +
      32B 对齐 + 字节数 32 倍数」（`pp.hpp:83-88/:100-105`）、`get_num_qps=min(2,allocated)`、
      cooperative launch、单 warp（`pp_send_recv.hpp:20-23/32/50-53`）；② **数据面 `pp_tma_copy`**
      （`cuh:27-102`）——**可复用的多 SM TMA 1D 两级流水**：跨 SM 分连续 32B 块区间、smem 切 stage+mbarrier、
      `tma_load_1d`+`mbarrier::complete_tx` 填流水、稳态「等 load→`tma_store_1d`→commit→`tma_store_wait`→
      发下一 load」、收尾排空；③ **寻址面 + 协议**——`get_pp_buffer_offset`（`cuh:20-25`）编码方向，
      4 段对称缓冲（0/1=收 prev/next，2/3=发 prev/next）与 `send_count/recv_count/arrival[2][2]/release`
      索引（表见 docs）；两跳协议 `pp_send_impl`（`cuh:106-171`：背压等 `release>=count-F+1` → TMA 到本地
      staging → `grid sync` → 按 QP `NCCLGin::put` + `StrongVASignalAdd` arrival）与 `pp_recv_impl`
      （`cuh:175-234`：等 arrival → TMA 拷出 → `grid sync` → 回 release）。**H100(sm90) 真机**：逐行抽取
      `get_pp_buffer_offset`/`pp_tma_copy`（sha256 `c3a3b0d1817b3bcb`/`ef702e38dd6caa81`）跑真实设备代码
      ——`N=2..8` 全部 **168 个有序对**的 12 个导出字段逐字段 **0 不符**、每 rank 4 段缓冲恰好各占一次、
      TMA 搬运 5 种 `num_sms`×8 种字节数 **40/40 byte-exact**（含跨 stage/SM 边界；真机坑：96KB 动态
      smem 必须 `cudaFuncSetAttribute` 否则 launch 静默失败）；`PPSignals` sizeof 192/alignof 64；
      Python oracle 逐行跑 `test_pp.py:19-50` 的 `generate_stress_ops`（sha256 `56d5e37722ddc497`）
      验证「第 i 个 send 严格晚于第 i-F 个 recv」+ 计数 + 全序。**设备 PASS=4 / Python PASS=6 / FAIL=0**。
      GIN 第二跳（`put`/`signal`）与真实多 rank 端到端为 NCCL 设备 API 专属，仅分析。
       → `docs/68-deepep-pp-send-recv.md`。配套博客 `content/posts/mega-moe-68-deepep-pp-send-recv`
       （索引页新增第 68 篇）。

### P59 增量（opt-in，补齐 DeepEP 第四类通信原语：Engram 带索引的远程抓取）

- [x] **P59-1** **DeepEP 的 Engram 远程抓取原语 `engram_fetch`**（`csrc/buffers/engram.hpp`、
      `csrc/kernels/engram/engram.hpp`、`deep_ep/buffers/engram.py`、
      `deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh`、`.../engram_fetch_wait.cuh`、
      `.../layout/engram/workspace.cuh`、`tests/engram/test_engram.py`）。
      第 7/58/61 篇拆了 **EP all-to-all**、第 64 篇 **bucket 集合通信**、第 68 篇 **PP send/recv**，
      本篇补上第四类：**带索引的随机远程读（indexed RDMA gather）**——每 rank 只存自己那部分 entry，
      给全局索引把命中项从主人 rank 拉回本地、每层一个完成钩子。三个核心设计：① **数据面走 RDMA /
      SF 面走 NVLink**（entry 本体只在主人 rank → `gin.get`；FP8 SF 每 rank 有副本 → 对称指针
      `get_sym_ptr<ncclTeamTagLsa>` 直读）；② **发射/等待分离**（`fetch()` 发所有层请求、返回每层
      hook，用哪层才 `hook()` 等）；③ **请求聚合**（同 peer 攒 32 一批 + 攒够 `flush_depth` 敲门铃）。
      **布局层** `EngramLayout`（`workspace.cuh:28-74`：storage/recv 字节偏移、SF 行号/分片号）+
      `EngramWorkspace`（`:9-21`：`BarrierSignals` + `issue_counters[128][2048]`）；**发射面**
      `EngramIssueWarpLayout`（`cuh:13-39`）把 warp 切成 (peer, QP, tile)，issue 循环（`cuh:99-171`）
      用 `__ballot_sync`/`__popc` 保序压入 64 深环形队列（多出一项恰好绕回翻转后的新队头），
      `issue_batch`（`:109-131`）把 `(global_idx,entry_idx)` 翻成 `(window, src_off, dst_off, peer)`；
      门铃前用 `issue_counters` 做到达计数（`:173-205`）；SF warp（`:208-254`）走 NVLink 分片桩 +
      `sf_token_stride/sf_hidden_stride` 两布局；wait（`engram_fetch_wait.cuh:9-28`）逐 request `gin.wait`。
      **host 公式**：`calculate_storage_size`（`engram.hpp:251-271`）、`get_theoretical_config`
      （`:273-299`，读受限 QP 模型 + 溢出回退）、`write`（`:109-155`，层首尾相接 + NVL 分片）。
      **H100(sm90) 真机**：逐行抽取 `EngramIssueWarpLayout`/issue 循环/SF 主体（sha256
      `32673de73ceeb9ce`/`06b24f4417c29feb`/`6f80791f169522ad`，与 `DeepEP@93eb6eb` 一致）跑真实
      设备代码 + Python 独立复刻布局/发射/请求/SF/两个 host 公式：**设备 PASS=5 / Python PASS=32 / 0
      FAIL**（256 条索引含 43 条 OOB → 213 条请求 miss/extra/值不符全 0；SF 两种目的布局各 1024 元素
      0 不符；溢出回退 `(33, 32768, True)`）。真实 `gin.get`/`gin.wait` 与多 rank 端到端为 NCCL GIN
      设备 API 专属（README 点名 `ncclGinOptFlagsWarpGet`），仅分析。
      → `docs/69-deepep-engram-fetch.md`。配套博客 `content/posts/mega-moe-69-deepep-engram-fetch`
      （索引页新增第 69 篇）。

### P60 增量（opt-in，补齐测试栈「内存安全 / 同步正确性」维度）

- [x] **P60-1** **DeepGEMM 的内存安全验证层 `tests/test_sanitizer.py`（compute-sanitizer 驱动）**
      （`tests/test_sanitizer.py`；`deep_gemm/testing/bench.py:87-90`；`csrc/runtime/runtime.hpp:45-48`；
      `third-party/deep_jit/.../cuda/{options,backend}.hpp`；`deep_gemm/testing/utils.py:11-22`；
      负样本 `src/sanitizer_negctl.cu`）。第 40 篇拆了计时/判对基座、第 41 篇生成器、第 43 篇逐位
      断言——它们答「结果对不对」；本篇补另一个维度：**内存访问安不安全、同步用没用错**。
      **架构无关（host/Python + 内联 PTX），H100 真机可跑**。
      → `docs/70-deepgemm-sanitizer.md`（真实行号核对）：**驱动器**——`test_template:12-26`（每
      用例单开子进程 + 固定 seed + `sys.path.append`）、自动发现 `:40-49`（遍历 `test*`、排 4 文件）、
      CLI `:30-39`；**排除了谁/为什么**（`test_mega_moe`=多 rank+SM100、`test_locality_domain`=MPS
      MLOPart）。**六个 env 的「为什么」**：`CUDA_LAUNCH_BLOCKING`、`DG_JIT_CHECK_NO_SPILLS`
      （`options.hpp:47/117-118` + `backend.hpp:120-122` 扫 nvcc 输出命中 `spilled to local memory`
      即断言失败）、`DG_USE_NVIDIA_TOOLS`（`bench.py:87-90` 返回 1 跳过 kineto）、
      `DG_USE_TEMP_CUBLASLT_WORKSPACE`（`runtime.hpp:45-48` 躲 `cudaErrorCudartUnloading`）、
      `PYTORCH_NO_CUDA_MEMORY_CACHING`（缓存分配器把越界藏进同段）、`TORCH_SHOW_CPP_STACKTRACES`。
      **六个 flag 的「为什么」**：`--target-processes=application-only`（不插桩 JIT 子进程）、
      `--destroy-on-device-error=context`（斩级联）、`--force-blocking-launches`、
      `--check-api-memory-access=no`、`--kernel-name-exclude kns=nvjet`（`kns=`=mangled 名子串，
      排 cuBLASLt）。H100 实测 `harness/exp_sanitizer.{py,sh,out.txt}` **30 PASS / 0 FAIL**：
      A) 静态清单核对（env/flag/exclude/31 个自动发现函数）；B) 负样本 `src/sanitizer_negctl.cu`：
      memcheck `clean=0`/`oob` 3（`1 bytes after the nearest allocation of size 1024 bytes`）/
      `misalign` 130 / `mask_big=0`（缓存屏蔽）/`mask_exact` 3（命中）；synccheck `sync` 64
      （`Barrier error: Divergent thread(s)`，且坐实 ptxas 会 if-convert 掉朴素发散 barrier）;
      **racecheck** `race` 1（坐实 synccheck 抓不到数据竞争，而这不在默认 tools 里）；C) 真实
      `test_sanitizer.py` 驱动 `test_hc_prenorm_gemm`/`test_sf_layout_kernels` × memcheck/synccheck
      全 `0 errors` rc=0（`DG_USE_NVIDIA_TOOLS` 让 perf 打印 1000000us）；D) 缓存分配器
      `reserved 20MiB > allocated 8MiB` vs 关闭后 `reserved=0`；E) `bench_kineto` 旁路返回 1。
      **盲区**：不跑 racecheck/initcheck、排除 mega_moe/locality_domain、`kns=nvjet` 排除 cuBLASLt。
       配套博客 `content/posts/mega-moe-70-deepgemm-sanitizer`（索引页新增第 70 篇）。

### P61 增量（opt-in，补齐 EP 通信一直缺席的「两级域」hybrid kernel）

- [x] **P61-1** **DeepEP 的两级（hybrid）EP dispatch / combine**（`deep_ep/include/deep_ep/impls/ep/
       hybrid_dispatch.cuh`（674 行）/ `hybrid_combine.cuh`（627 行）；`csrc/kernels/ep/dispatch.hpp`、
       `combine.hpp`；依赖 `common/{math,ptx}.cuh`、`layout/ep/token.cuh`、`impls/ep/combine_utils.cuh`）。
       第 7/48/61 篇拆的是**单级**（`num_scaleout_ranks==1`）路径，本篇补上 Mega MoE 在 EP8/EP16 上
       真正跑的 **hybrid** kernel：同一 kernel 内同时编排 **scale-out（跨节点 RDMA，`ncclTeamTagRail`）**
       与 **scale-up（同节点 NVLink，`ncclTeamTagLsa`）**。**数据面用 NCCL GIN 设备 API → SM100 无关但
       本机 NCCL 2.29.7 不支持 → 仅分析**；四个**架构无关内核**在 H100 真机跑真实设备代码。
       → `docs/71-deepep-hybrid-dispatch-combine.md`（真实行号核对）：host 分派 `dispatch.hpp:89-114`
       （`num_scaleout_ranks==1` 走 `dispatch_impl`，否则 `hybrid_dispatch_impl` + warp 角色 3 元组）
       / `combine.hpp:64-90`（2 元组）；rank 混合进制编码 `hybrid_dispatch.cuh:60`；warp 角色
       notify（`kNumNotifyWarps=4` `dispatch.hpp:20`）/scaleout/forward（`dispatch.hpp:80-85`）。
       **dispatch 三角色**：notify（`:107-328`：本地 `atomicAdd_block` 计数 + rank 去重 `:126-135` →
       `red_add (1<<32)|count` 把 SM 到达数打包进高位 `:140-143` → SM0 等 `>>32==kNumSMs` 后
       `encode_decode_positive` 编码写远端 `:155-156` → `gin.put` 跨 scale-out `:177-188` →
       `recv_and_reduce` 汇聚 `:193-219` → `put_value`/`red_add_rel<ncclTeamTagLsa>` 传播 `:221-252`
       → 等就绪 + 对齐 `:263-300` → `do_psum` rank 含端点 / expert 排他前缀和 `:302-327`）；
       scaleout（`:329-460`：`preload_next_token` TMA+cp.async `:354-385`、按 rank 去重分槽 `:408-417`、
       `gin.put` RDMA `:445-452`、本地目的绕过 `:432-436`、间隔 tail `update_scaleout_tail` `:338-351`）；
       forward（`:461-656`：round-robin 位图选 rank `:490-494`、chunk=`kNumSlotsPerForwardChunk=6`
       `:527-530`、`gin.get_sym_ptr<ncclTeamTagLsa>` 直写 + `tma_store_1d` `:590-596`、写
       `token_metadata_at_forward` `:609-627` + 链式列表 `:475-479/638-649`、PDL `cudaTrigger...`
       `:665`）。**combine 两角色**：scaleup（`:107-351`：沿 `channel_linked_list` 迭代、三情形归约
       `:230-303`、`combine_reduce`、`update_tails` 间隔 `st.release.sys` `:122-149`）；forward
       （`:352-622`：重放 `token_metadata_at_forward` `:390-403`、等 tail 水位 `:405-447`、
       `flush_last_tma_and_issue_rdma` 重叠 `:372-388`、多次归约先 `combine_reduce` 再转发 `:496-572`、
       清 tail + `red_add_rel` scale-out 到位线 `:579-620`）。**四个架构无关机制**：就绪哨兵
       `encode_decode_positive=-c-1`（`math.cuh:26-33`，区分「未写=0」与「写了 0」）、tail 增量
       `pack2/unpack2`（`math.cuh:45-68`，**finish 低 32 / tail 高 32**）、`do_psum`、round-robin 位图。
       H100 实测 `src/deepep_hybrid.cu` + `harness/exp_deepep_hybrid.{sh,py,out.txt}`：**直接 include
       真实头文件、编译运行真实设备代码**——Part1 哨兵 8 计数（含 0/`2^20`/`INT32_MAX`）编解码对合 +
       host 4096 计数；Part2 pack2/unpack2 位段 + 增量 {2→5→9} 累加；Part3 `do_psum`（逐字搬运
       `:304-320`）含端点/排他 8 组；Part4 round-robin（逐字搬运 `:490-532`）4 组「每 rank `[0,tail)`
       恰好一次 + 序列 = 贪心 oracle」；Part5 `TokenLayout` 每槽字节（`hidden=7168,sf=56,topk=8,meta`
       → `bytes<false>=7360/bytes<true>=7392`）。**设备 7 PASS / Python oracle 6 PASS / 0 FAIL**。
       配套博客 `content/posts/mega-moe-71-deepep-hybrid-dispatch-combine`（索引页新增第 71 篇）。

### P62 增量（opt-in，补齐 P54 只当「仅分析」放过的两个 bucket 集合算子）

- [x] **P62-1** **DeepEP bucket 的 `all_reduce` / `reduce_scatter` 三后端 + 信用环 + bf16 随机舍入**
      （`csrc/kernels/bucket/{all_reduce,reduce_scatter}.hpp`；
      `deep_ep/include/deep_ep/impls/bucket/{all_reduce,reduce_scatter}/{nvlink,rdma,hybrid}.cuh`；
      `layout/bucket/{all_reduce,reduce_scatter,workspace,chunk}.cuh`、`comm/bucket.cuh`；
      `deep_ep/buffers/bucket.py`；`tests/bucket/test_{all_reduce,reduce_scatter}.py`；
      `third-party/tilelang_ops/stochastic_rounding.py`）。第 64 篇（P54-1）只把**纯 NVLink all-gather**
      （copy engine 驱动、无 SM kernel）讲透，`all_reduce`/`reduce_scatter` 因「用 NCCL GIN 设备 API」
      被留作仅分析。本篇补齐：与 all-gather 不同，这两个**要归约、必须有 SM kernel**——纯 NVLink 腿用
      `multimem.ld_reduce.add` 硬件归约（`all_reduce/nvlink.cuh:36-51` "fused reduce-scatter +
      multicast-store all-gather"，把每 rank NVLink 流量降到 O(1)），RDMA 腿反向环 + `cp.reduce.async.bulk
      .add.f32`（`ptx.cuh:148`）累加、`kWithScale` 时末 peer 留给寄存器只乘一次 scale（`all_reduce/rdma
      .cuh:153/199-242`），hybrid 三级（multimem→RDMA→broadcast，`all_reduce/hybrid.cuh:145-332`、
      `reduce_scatter/hybrid.cuh:102-278`）。**三个非平凡点**：① **两种完全不同的分块映射**——
      all_reduce rank-major（`all_reduce/rdma.cuh:96-104`：`dst=ci%R, slot=(ci/R)%S, gen=ci/R/S`）、
      reduce_scatter slot-major（`reduce_scatter/rdma.cuh:100-101`：`slot=ci%S, gen=ci/S`），R=4 时
      8191/8192 个 chunk 的 (slot,gen) 不同；② **信用环 head/tail 背压**（发送等 `head≥gen`、接收等
      `tail≥gen+1`、消费完回 `head`，`reduce_scatter/rdma.cuh:109-222`，**最后一个用槽的 chunk 免信用**
      `:137`）；③ **fp32 求和顺序**——kernel「本地先行 + 反序环 `(rank-i)%R`」与测试参考
      （`test_reduce_scatter.py:78-87`）**逐字一致**，正是 `rtol=0` 逐位判等的根因。bf16 路径
      （`reduce_scatter/rdma.cuh:235-635`，仅 RDMA-only）用**随机舍入**消除截断偏差，哈希
      `:401-412`（四常数 + fmix + 两终端乘子）与 **DeepGEMM `StochasticRoundToBF16` 同源**
      （`0x846CA68B`/`0xD35A2D97`）；真实舍入指令 `cvt.rs.bf16x2.f32`（`ptx.cuh:604`）SM100 专属。
      `cvt.rs`/`multimem`/`gin` 数据面**仅分析**；但布局几何、`ChunkIterator` 覆盖、两种映射、
      `get_head_ptr/get_tail_ptr`、`fadd4/fmul4` 归约、bf16 哈希、host 配置算术**架构无关**。
      H100(sm90a) 实测 `src/deepep_bucket_collectives.cu` +
      `harness/exp_deepep_bucket_collectives.{sh,py,out.txt}`：**直接 include 真实头文件跑真实设备代码**，
      与独立 host oracle 对拍 **PASS=30 / FAIL=0**——A) 8 个结构 `sizeof`/`offsetof` + `BucketSignals
      =794688`/`BucketWorkspace=274792960` + 16MiB 预算全对；B1) `ChunkIterator` 3 桶 7 块恰好覆盖一次、
      每块 offset/bytes 一致；B2/B3) 两种映射 0 不符 + 8191/8192 不同；B4) 信用环寻址
      `(rank*S+slot)*8` 0 不符；C1) 反序环累加×scale vs `np.float32` 同序**逐位 0/16384**、升序朴素
      922/4096 bit 不同；C2) all_reduce 环求和×scale 逐位 0/4096、vs fp64 relerr 1.40e-07；
      D) bf16 哈希 vs tilelang 参考逐位 **0/4096**、随机位有熵、乘子与 DeepGEMM 同；E) host 配置算术
      （mm/broadcast warps、chunk KB、slots）与信号预算上限全对。配套博客
      `content/posts/mega-moe-72-bucket-allreduce-reduce-scatter`（索引页新增第 72 篇）。

### P63 增量（opt-in，补齐 DeepEP dispatch/combine 的「开多少 SM / QP」自动选择模型）

- [x] **P63-1** **DeepEP dispatch / combine 的自动 SM / QP 选择模型与带宽探测层**
      （`deep_ep/buffers/ep.py:431-541/543-560/846-852`；`deep_ep/utils/envs.py:193-283`；
      `csrc/kernels/comm/context.cpp:52-63/166-183`）。前面第 7/48/58/61/63/71 篇把 dispatch/combine
      的 **kernel 本体**拆得很细，却一直没回答上游问题：**每次 dispatch/combine 开多少 SM、多少 QP？**
      本篇补上这层 host 侧**解析式 roofline**。**纯 host / Python、不碰 CUDA/NCCL 设备 API、不依赖
      SM100 → H100 真机可跑**（`import deep_ep` 的 `check_nccl_so` 用 `sys.modules` 注入合成
      `deep_ep._C`/`deep_ep.comm` stub 绕过；`ast` 从真实 `ep.py` 抽取三个方法源码运行**未修改的真实
      方法体**）。→ `docs/73-deepep-sm-qp-autotuning.md`（真实行号核对）：**带宽探测层**——每 SM HBM
      读/写 `get_sm_read_gbs=180`/`get_sm_write_gbs=45`（`:193-204`）、`get_nvlink_gbs` 解析
      `nvidia-smi nvlink -s` 求和 ×0.9（`:207-234`）、`get_rdma_gbs` 解析 `ibstat` Rate÷8（`:260-283`）、
      `check_fast_rdma_atomic_support` 看 CA type `MT4131`（`:237-257`），全部 `lru_cache`。
      **域大小**：物理域 `(num_rdma_ranks=rank数/LSA, num_nvlink_ranks=ncclTeamLsa.nRanks)`、
      逻辑域 hybrid 时等于物理域、否则 `(1, num_ranks)`（`context.cpp:52-63/174-183`）。
      **SM 模型**（`:431-541`）四步：① 期望命中组数 `get_expected_topk` 组合式（`:473-475`，强断言
      `E % num_groups == 0`）；② 「每 token 流量账」：HBM 读/写 + RDMA/NVLink，hybrid 与 direct 两分支
      （`:483-509`）；③ 取**预计耗时最长**的链路当标尺（`:511-515`）；④ 用每 SM HBM 带宽反推 SM 数，
      取四候选最大、`×1.3` 余量、`align 4`、`max(4,...)`，`prefer_overlap` 关时 `max(.,64)`，硬上限
      `device_sms//2*2`（`:517-532`）——这就是「通信—计算重叠」的预算来源。**QP 模型**（`:543-560`）：
      direct `min(num_sms,9)` 省 doorbell、hybrid `num_sms*16+1`（16 channel + 1 notify），受
      `num_allocated_qps` 钳制；默认分配 direct 17 / hybrid(+fast atomic) 65 / 否则 129（`:327-335`）。
      消费点 `ep.py:641-645`（dispatch）与 `:780` 起（combine）。**H100 真机**
      `harness/exp_deepep_sm_qp.{sh,py,out.txt}`（真实 `envs.py` sha256 `23843e16…`、`ep.py`
      `a8f72c39…`）：**16 PASS / 0 FAIL**——真实带宽探测（nvlink 430.3、rdma 0、fast-atomic False）；
      9 配置真实方法 vs 独立 oracle 逐位一致；QP 模型逐位一致；`get_expected_topk` 6 组对拍暴力枚举
      差 ≤4.4e-16；边界（4 倍数/≤132/≥4/overlap 关 ≥64/EP1=132）；**三个真实边界**：`E%num_groups==0`、
      本机无 IB 时 hybrid 自动探测 `ZeroDivisionError`、direct `min(num_sms,9)`。配套博客
       `content/posts/mega-moe-73-deepep-sm-qp-autotuning`（索引页新增第 73 篇）。

### P64 增量（opt-in，补齐整条测试栈最基础的「顶层规格」）

- [x] **P64-1** **稠密 GEMM 的验收测试即正确性规范**（`tests/test_fp8_fp4.py` 406 行 /
      `tests/test_bf16.py` 341 行；`tests/generators.py` 的枚举器与 `QuantConfig`）。
      测试栈此前拆过基座（第 40 篇 `deep_gemm/testing/`）、生成器（第 41 篇 `generators.py`）、
      逐位断言（第 43 篇 `tests/utils.py`）以及四个 mega 算子的端到端测试（第 41/47/50/57 篇），
      但**整条栈最基础、最先跑的两个顶层文件**一直只被顺带引用几行。本篇把它们当「稠密 GEMM 算什么对」
      的**可执行规格**来拆。**主体 kernel 在 H100(sm90) 上真的会跑**（FP8→1D2D、BF16→NoSF），
      FP4 / `use_alpha` / UE8M0 / 随机舍入 / PSUM 零填充为 SM100 专属（仅分析）。
      → `docs/74-dense-gemm-test-oracle.md`（真实行号核对）：**顶层结构**——`test_fp8_fp4.py:396-406`
      无 arch 门控无条件跑四函数；`test_bf16.py:326-341` 前三个被 `if get_arch_major()>=9` 包住、
      后三个 cuBLASLt 函数无条件跑。**arch 门控矩阵**（`generators.py`）：`get_kernel_types:100-104`
      （SM90 FP8 只有 1D2D）、`get_list_from_dtype:76-86`（SM90 一个 legacy 配置、SM100 加 FP8×FP4/FP4×FP4）、
      `get_ue8m0_usage:95-98`、`enumerate_normal:122-173`、`enumerate_k_grouped_contiguous:229-262`
      （SM90 强制累积+FP32 输出）、`enumerate_m_grouped_*:194-226`。**逐函数「从松到严」判据**：
      `calc_diff<max_diff`（`test_fp8_fp4.py:54`）→ 20× `torch.equal` 确定性（`:79/145/212/289`）→
      direct==FP32 累积（`:306`）→ 随机舍入 quartet 哈希（`:65/311`）→ FP4==无损 FP8 `<1e-14`
      （`:99/152/219/285`）→ SF K-stride padding 逐位（`:93/351`）→ PSUM 零填充（`:158`）→
      空组 D/NaN 不变（`:269-271`）；且 `test_gemm` 的 `test_alias` 走 `nt/nn/tn/tt` 转置薄壳。
      **`calc_diff` 盲区实证**（`numeric.py:5-13` 是余弦相似度）：对全局缩放**完全不变**
      （真实 BF16 输出缩放 0.999/1.0001 后 calc_diff 一字不变）、单元素 1-ulp 扰动不敏感
      （1.782e-8→1.487e-11），坐实逐位判据不可省。**H100 真机**
      `harness/exp_dense_gemm_test_oracle.{sh,py,out.txt}`：直接 import 真实测试模块、调用
      **真实测试体 11 个函数**（只压形状枚举、打桩 `bench_kineto`，测试体一行未改）全部断言通过；
      枚举计数 arch9 **252 用例** vs 假 arch10 更大；真实 `bench_kineto` 采 BF16 `bf16_gemm_nt`
      **235 TFLOPS（m=4096,n=7168,k=7168）/ 413 TFLOPS（方阵）**——**18 PASS / 0 FAIL**。
      配套博客 `content/posts/mega-moe-74-dense-gemm-test-oracle`（索引页新增第 74 篇）。

### P65 增量（opt-in，补齐「通信—计算重叠」的 Python / host 调度地基）

- [x] **P65-1** **DeepEP 的通信—计算重叠「Python 调度地基」**：`deep_ep/utils/event.py`
      （`EventOverlap`）、`deep_ep/comm/stream.py`（`get_comm_stream`）、`deep_ep/buffers/base.py`
      （`BufferBase`），以及 C++ 孪生 `csrc/utils/event.hpp`（`EventHandle`）、
      `csrc/comm/stream.hpp`（通信流 / `stream_wait`）、`csrc/buffers/base.hpp`，与真正的交接点
      `csrc/buffers/ep.hpp:122-181`（`stream_control_prologue/_epilogue`）。前面第 2/9/11/63/71/73
      篇从 kernel 侧讲重叠，本篇补**谁把通信发到独立 stream、谁让计算流只在必要处等**这一层。
      **纯 host / Python，H100 真机可跑**（`deep_ep._C` 用 sys.modules 桩注入，按文件路径 exec 真实
      Python；`EventHandle` 用 `TorchEventHandle` 逐行镜像后驱动真实 `torch.cuda.Event/Stream`；
      另编译 CUDA 程序镜像 `get_comm_stream`/`stream_wait` 语义）。
      → `docs/75-deepep-stream-event-overlap.md`（真实行号核对）：通信流三要点（`static` 单例 /
      `getStreamFromPool(true)` 高优先级 / Python 薄包装同 `stream_id`）；`stream_wait` 两重载 +
      自等待断言（`stream.hpp:20-23`）；`EventHandle` 只暴露默认构造 + `current_stream_wait`
      （`event.hpp:32-36`）、`tensors_to_record` 替代 `record_stream`（`EP_AVOID_RECORD_STREAM`）；
      `EventOverlap` 状态机八条（`_release_handle_by_call` / `hook_after_wait` 一次性 /
      `__call__` 返回 self / `__exit__` 无视异常）；交接点 prologue（通信流等 `previous_event` 或
      计算流 + `allocate_on_comm_stream` 切账）与 epilogue（`async_with_compute_stream` 二选一 +
      双流 `record_stream`）；`BufferBase` 生命周期（`main_context` 转发 / `destroy` 契约）。
      H100 实测 `harness/exp_deepep_stream_event_overlap.{sh,py,out.txt}` + `src/deepep_stream_event.cu`：
      **21 PASS / 0 FAIL**——`EventOverlap` 8/8、`get_comm_stream` 3/3、`BufferBase` 4/4、
      事件/流/`stream_wait` 真机断言全过；**重叠两必要条件实测**：全卡通信 + 全卡计算**不重叠**
      （512MB copy 0.392ms ∥ FFMA 7.79ms → 串行 8.173 / 重叠 8.263 / **0.989×**），
      通信腿缩到 1/4 SM（33 blocks）则**真重叠**（2.025ms ∥ 7.79ms → 串行 14.534 / 重叠 8.186 /
      **1.776×**）——正面坐实第 63 篇「半卡 SM 上限」的必要性。配套博客
       `content/posts/mega-moe-75-deepep-stream-event-overlap`（索引页新增第 75 篇）。

### P66 增量（opt-in，补齐 einsum 家族一直缺席的「验收测试即正确性规范」）

- [x] **P66-1** **DeepGEMM einsum 家族的验收测试即正确性规范**（`tests/test_einsum.py` 276 行；
      `csrc/apis/einsum.hpp` 四分派；`tests/utils.py` 的逐位 oracle；`deep_gemm/utils/math.py`
      的参考量化器）。第 40/41/43 篇拆了测试基座/生成器/逐位断言，第 41/47/50/57/64 篇拆了
      Mega MoE / Mega Gate / MQA logits / Mega mHC / 稠密 GEMM 的端到端测试，但把整个
      **einsum 家族**（`bmk,bnk->mn` + MLA 双向投影 + 权重梯度 + FP8 block-scale 变体）判据集中
      定义的 `tests/test_einsum.py` 一直只被顺带引用。本篇把它当**「einsum 算什么对」的可执行规格**。
      H100 上真正跑五个函数，两个 FP8 反向/投影被 `@test_filter(arch>=10)` 门控（仅分析）。
      → `docs/76-einsum-test-oracle.md`（真实行号核对）：**顶层结构** `:21-259`（七个函数）+ `__main__`
      `:262-276` 无条件调用七个；门控 `:194`/`:236`；`test_fp8_bhr_hdr_bhd` 无装饰器（bf16 路径
      sm90 可跑）。**逐函数**：`test_bmk_bnk_mn:21-41`（`s=129` 非对齐、fp32 强制 `c=d`（`:29`）、
      参考 `bmm.sum(0)`）、`test_bhr_hdr_bhd:44-72`（**带 padding 权重** `fy[:,:,:r]`（`:49-50`）、
      `deterministic` 双向切「自研 vs cuBLASLt」（`:54-65`）、`assert_direct_output_...`（`:60`））、
      `test_bhd_hdr_bhr:75-104`、`test_bhd_bhr_hdr:107-125`（**`z_0*10` 放大 C 累积**（`:113`））、
      FP8 `test_fp8_bhr_hdr_bhd:128-191`（A per-token 1D + B per-128×128 2D SF；输出 dtype 分岔
      `:144`；**FP8 输出对必须逐位 == 「先 BF16 D 再 per-token 量化」**（`:159-164`，仅 SM100））、
      `test_fp8_bhd_hdr_bhr:194-233` / `test_fp8_bhd_bhr_hdr:236-259`（SM100）。
      **判据清单**：`calc_diff` 阈值 `:34`=1e-5 / `:58/89/118`=1e-10 / `:181`=2e-3(fp8)、1e-3(bf16) /
      `:213/252`=1e-3；逐位 oracle `:60/91`（direct==FP32 累积）、`:63/94/179/223`（随机舍入哈希）、
      `:163/164`（FP8 输出对逐位）。
      **H100 真机** `harness/exp_einsum_test_oracle.{sh,py,out.txt}`：直接 `import test_einsum` 调用
      **真实测试体**（只打桩 `bench_kineto`，测试体一行未改）**15 PASS / 0 FAIL**——五个可跑函数断言
      全过、两个 SM100 函数当场打印 `Filtered`；D 负样本：`bhr,hdr->bhd` **`calc_diff=0` 且逐位**、
      `bhd,bhr->hdr` 丢 C 从 1.2e-13 变 4.7e-2、`bmk,bnk->mn` 两条等价 ~1e-14；E profiler 抓确定性开关
      两侧真实 kernel（`sm90_bf16_gemm_impl<...GemmType)4...>` vs `nvjet_sm90_tst_128x256...`）；
      F 真实吞吐 `bhr,hdr->bhd` 96（h=128,b=8192）/ `bmk,bnk->mn` **124 TFLOPS**。
      真机坑：fp32 `bmk,bnk->mn` 不传 `c` 段错误（契约 `einsum.hpp:25-26`）。配套博客
      `content/posts/mega-moe-76-einsum-test-oracle`（索引页新增第 76 篇）。

### P67 增量（opt-in，补齐所有 impl 共用的「TMA 发射层」与环形流水线）

- [x] **P67-1** **DeepGEMM 的共享 TMA 拷贝层 `common/tma_copy.cuh` + 环形流水线
      `common/ring_pipeline.cuh`**：第 44 篇补齐了设备端公共地板
      `common/{math,utils,types,packing,compile,exception,cute_tie}.cuh`、第 56 篇补了 128 字节
      `CUtensorMap` 的**来源**（`runtime_utils.hpp` 的 `make_tma_*`），但夹在中间那层——
      **「描述符造好之后谁发 TMA、发几次、搬到 smem 哪里」**——从没被单独讲过。它就是
      `deep_gemm/include/deep_gemm/common/tma_copy.cuh`（106 行，被 **16 个 impl** include）：
      `get_inner_block_atom_size:12-15`（`swizzle==0 ? BLOCK_INNER/pack : swizzle_bytes/sizeof(dtype)`）、
      `LoadOps<2..5>:17-37`（`Load`/`LoadMulticast`/`Load2SM` 三原语）、`copy_nd:39-77`（三条静态断言
      守坐标数/cache hint/打包混算；沿 inner 方向**逐原子发**，smem 用 storage、gmem 坐标用 logical；
      `num_tma_multicast>1` 时分化出 **sm90 多播** rank0 发 `LoadMulticast`、mask `(1<<n)-1`，
      与 **sm100 2-SM** 只给 leader 发信号）、`copy:96-104`（`kIs3DTMA` 批处理包装）。
      Mega MoE 权重是 **4D 描述符**（`sm100_fp8_fp4_mega_moe.cuh:793` 的 `tma::copy_nd<...,4>`
      坐标 `(k, 半区, 上下半, 专家号)`）。`ring_pipeline.cuh:13-38` 把「环形 stage + 相位翻转 +
      返回旧状态」标准化（2 的幂走位运算、非 2 的幂走减法，契约 `step ≤ kNumStages`），
      MQA logits / 稀疏 MQA / Mega mHC 都用 `CUTE_TIE_DECL(p.advance(), stage, phase)` 接住；
      取代第 29 篇 `sm90_bf16_gemm.cuh:138-145` 手写的 `advance_pipeline` lambda。
      → `docs/77-deepgemm-tma-copy-ring-pipeline.md`（真实行号核对）。
      **H100(sm90) 真机** `harness/exp_tma_copy_ring.{sh,out.txt}` + `src/tma_copy_ring.cu`
      （nvcc 直接 include 真实头文件跑真实设备代码，需 `--expt-relaxed-constexpr`）：
      **27 PASS / 0 FAIL**——constexpr 原子/打包（bf16 sw128→64、fp8 sw128→128、fp4 sw128→128 storage、
      pack=2，10 条）；`RingPipeline` 9 组（含非 2 的幂 3/5、step=1/2/3/N）共 245 次调用 0 不符；
      真实 `tma::copy` 的 **atom 循环**（`BLOCK_INNER=256` 展开成 4 条 `cp.async.bulk.tensor.2d`）
      拼出 **0/4096 逐字节**、3D 批次坐标 0/1024、**sm90 多播 cluster=2 两 CTA 各自 barrier 收 tx
      0/1024**；唯一例外是 `copy_nd` 的 `__CUDA_ARCH__>=1000` 2-SM 分支（SM100 专属，仅分析）。
      配套博客 `content/posts/mega-moe-77-deepgemm-tma-copy-ring-pipeline`（索引页新增第 77 篇）。

### P68 增量（opt-in，补齐 SF 布局家族一直缺席的「验收测试即正确性规范」）

- [x] **P68-1** **SF 布局家族的验收测试即正确性规范**（`tests/test_layout.py`；
      `tests/generators.py:276-308`；`deep_gemm/utils/layout.py`；`csrc/apis/layout.hpp`；
      `csrc/jit_kernels/impls/smxx_layout.hpp`）。第 25 篇（P15-1）拆了单组 SF 布局变换的**实现**、
      第 26 篇（P16-1）拆了 K 轴分组 SF 打包，但「变换后的布局算什么对」的唯一规范
      `tests/test_layout.py` 一直只被顺带引用。本篇把它当**可执行规格**拆透。**三个 kernel
      （`transpose_fp32`/`transpose_and_pack_fp32_into_ue8m0`/`pack_fp32_into_ue8m0`）非 SM100 专属，
      H100 真机可跑**；两个 k-grouped 函数 SM100 专属（arch 门控跳过），但 host 契约/枚举/纯 torch
      oracle 架构无关，H100 真机实测。→ `docs/78-sf-layout-test-oracle.md`（真实行号核对）：
      **纯 torch oracle** `get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl:20-42`
      （`x.view(torch.int) >> 23` 取 UE8M0 最高字节 `:24`、`aligned_mn=get_tma_aligned_size(mn,4)`
      `:32` 与 `aligned_k=align(k,4)` `:33` 两个对齐口径、`padded.view(-1).view(dtype=torch.int)`
      `:36` 4 字节小端合成 int32、`empty(...).mT` `:39-40` 转成 MN 连续），与 C++ 自带同构 torch 回退
      `smxx_layout.hpp:108-131` 同一条数学。三个测试函数：`test_sf_layout_kernels:45-80`（UE8M0 路
      `:54-60` 卡「值逐位+shape+stride」`torch.equal:58`；转置路 `:61-70` 卡「shape+stride+值不变」
      `torch.equal(fp32_sf,transposed_sf):70`；perf 容错吞 `AssertionError→t=0:74-77`）、
      `test_k_grouped_sf_layout_kernels:83-115`（arch 门控 `:85-87`；**每组切出套单组 oracle 再拼回**
      `:102-104`；尾块补零逐位 `:106-107`）、`test_k_grouped_psum_sf_layout_kernels:118-154`
      （逻辑 K vs 物理 K `:127-128`、PSUM `:138`、未同步/空 ks 上界 `:142-147`）。C++ 侧：
      `preprocess_sf:48-58`、`get_mn_major_tma_aligned_tensor:60-106`（早退 `:64-65`、torch copy
      回退 `:71-74`、`transpose_fp32` `:83-93`）、`get_mn_major_tma_aligned_packed_ue8m0_tensor:133-216`
      （torch 回退 `:152-153`/`:187-188`、PSUM `:141-149`、`transpose_and_pack` `:163-173`、pack 快路
      `:185-214`）、`get_k_grouped...:218-283`（arch 断言 `:224`）；`csrc/apis/layout.hpp` 四分派
      `:38-39/42-43/46-51/54-55`、k-grouped `:89-116`（arch `:100-101`、SM90 `:104-105`）。
      H100 实测 `harness/exp_sf_layout_test_oracle.{sh,py,out.txt}`（**直接 import 真实测试模块复用其
      oracle 与测试体**）**8 PASS / 0 FAIL**：① `enumerate_sf_layout` 108 用例（54 UE8M0）逐位/shape/
      stride **0 不符**；② 两条 kernel 路径（已 MN-major 走 `pack` 快路 / 未 MN-major 走
      `transpose_and_pack`）与同一 oracle 逐位相等、1 字扰动被 `torch.equal` 抓到；③ 直接调真实
      `test_sf_layout_kernels()` 断言全过；④ 两个 k-grouped 函数打印 `Skipped`、host API 被
      `smxx_layout.hpp:224` 断言拒（RuntimeError）；⑤ **profiler 坐实**同一 `impl` 调用有的发射真
      kernel（`transpose_and_pack<512u,48u,56u,128u,1u,false>` / `pack_fp32_into_ue8m0<1u,512u,128u,16u,true,false>`
      / `transpose_fp32<512u,64u,56u,128u,57u>`）、有的走 host 早退或 `out.copy_` torch 回退；
      ⑥ 枚举 arch9 108 / 假 arch10 218。配套博客
      `content/posts/mega-moe-78-sf-layout-test-oracle`（索引页新增第 78 篇）。

### P69 增量（opt-in，补齐 HyperConnection prenorm GEMM 一直缺席的「验收测试即正确性规范」）

- [x] **P69-1** **HyperConnection prenorm GEMM 的验收测试即正确性规范**
      （`tests/test_hyperconnection.py`；`deep_gemm/testing/{utils,numeric,bench}.py`；
      `deep_gemm/utils/math.py`；`csrc/apis/hyperconnection.hpp`；
      `csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp` / `sm100_tf32_hc_prenorm_gemm.hpp`）。
      第 16 篇（P7-2）拆了 `tf32_hc_prenorm_gemm` 的**实现**，本篇补上「这份算子算什么对」的
      唯一规范 `tests/test_hyperconnection.py`（57 行）。它门控极宽
      （`@test_filter(get_arch_major() >= 9)`，**H100 可跑**），用例矩阵 24 个 deterministic 形状。
      **核心看点：判据编码了实现事实**——`allow_tf32=True`（`:17-18`）不是加速，而是「kernel 是
      TF32」在参考侧的必要镜像；`ref_d = a.float() @ b.T`（`:34`）只上转 A、不动 B，精确对应
      kernel 的数据路径；FLOP 口径 `2*m*n*k`（`:43`）**不含 S**，坐实融合=搭车。判据
      `calc_diff < 1e-8`（`:38`）是「TF32 同层残差之下、关 TF32 失配之上」的分离带。
      → `docs/79-hc-prenorm-test-oracle.md`（真实行号核对）：测试顶层结构（`test_hc_prenorm_gemm:14-45`
      + `__main__:50-57`）、`@test_filter`（`testing/utils.py:11-22`）/`get_arch_major`（`:6-8`）、
      用例矩阵（`:21-23`）、参考数学与判据（`:24-38`）、FLOP/byte 账目（`:40-44`）、
      `calc_diff`（`numeric.py:5-11`）/`count_bytes`（`:14-21`）；host 契约两层
      `hyperconnection.hpp:16-18/21/26-40/43-44` + `sm90_...hpp:21-31`（SM100 差异 `n<=128`
      `sm100_...hpp:29`）。H100 实测 `harness/exp_hc_prenorm_test_oracle.{py,sh,out.txt}`
      **8 PASS / 0 FAIL**：① 直接 import 真实测试模块、驱动真实测试体（打桩 `bench_kineto` 只留
      断言）24 用例**全部 assert 过**；② 真实 `bench_kineto` 跑通测试自带 perf 路径（共享 GPU 抖动大，
      性能结论仍以第 16 篇为准）；③ **关 `allow_tf32` 使 `calc_diff(D, 真fp32)` = 2.15e-08 /
      2.39e-08 越过 1e-8**（坐实测试为何必须开 TF32）；④ SM90 host 契约真的会拒
      （`sm90_...hpp:30` 的 `n=64`/`n=20`、`:31` 的 `k=4104`；`n=24/32` 通过）；⑤ split-K 的 D/S
      可加性 `calc_diff` 1.28e-10 / 4.22e-14、16 split 范数 ±1.3%；⑥ `2mnk=11.27 GFLOP`、
      `count_bytes=473.33 MB`（A 占 469.76）、`AI=23.82 FLOP/B`。SM100 分支仅分析。
      配套博客 `content/posts/mega-moe-79-hc-prenorm-test-oracle`（索引页新增第 79 篇）。

### P70 增量（opt-in，补齐 DeepGEMM 顶层运行时一直缺席的「验收测试即正确性规范」）

- [x] **P70-1** **DeepGEMM 顶层运行时的验收测试即正确性规范**（`tests/test_lazy_init.py`；
      `csrc/runtime/runtime.hpp`；`csrc/apis/config.hpp`；`deep_gemm/__init__.py`）。
      第 23 篇（P23-1）拆了 DeepJIT 编译/缓存/加载/发射管线；第 40 篇（P40-1，`docs/50`）拆了顶层
      `Runtime` 单例的**生命周期实现**，但只在 §6.5 用一句「直接跑上游测试：Ran 3 tests / OK」引用了
      `tests/test_lazy_init.py`（199 行）。本篇把这份测试当**可执行规格**单独拆透。
      **核心是「用 `LD_PRELOAD` 探针把 cuBLASLt 换成可计数的假库」**：探针导出同名
      `cublasLtCreate/Destroy`（`:38/:45`），三个 magic exit code = 三把尺子——`_Exit(91)`
      （`backend_dead` 后仍销毁 = 退出期 use-after-free，`:46`）、`_Exit(92)`（`live_handles!=0` =
      泄漏，`:30`）、`_Exit(93)`（销毁陌生/重复 handle，`:47`）；`initialize_probe` 是
      `__attribute__((constructor))`（`:34`）故 `verify_cleanup` **最后跑**（注释 `:33`）。
      → `docs/80-lazy-init-test-oracle.md`（真实行号核对）：顶层结构（一个
      `TestNativeRuntimeShutdown:56` + Linux 门控 `:55` + `setUpClass` 两道 skip 闸门 `:60-64` + 编译
      探针 `:70-71`；`__main__:187-199` 先 8×`mp.Process` 断言 `exit 0` 再 unittest）、
      探针源码 `:22-52` 逐行解码、6 用例矩阵（`:85-92`）语义、子进程三环境
      （`CUDA_VISIBLE_DEVICES=''`:104 / `DG_USE_PYTORCH_CUBLASLT_HANDLE`:105 / `LD_PRELOAD`:106）
      + preamble 断言导入本地构建（`:99`）、`test_real_cublaslt_shutdown:116-139` 的
      `MALLOC_PERTURB_=165`+关 tcache（`:131-132`，把退出竞态变可复现崩溃）、
      `test_cuda_workspaces_are_released:141-184` 的 atexit 在 shutdown 之后跑（`:154-155`）、
      退出时至少释放 2×16 MiB（`:152`）。H100 实测 `harness/exp_lazy_init_test_oracle.{py,sh,out.txt}`
      **7 PASS / 0 FAIL**：① 原样跑真实测试模块 `Ran 3 tests in 25.1s / OK`（**unittest 输出在 stderr**）；
      ② 逐字取出 `_PROBE_SOURCE` 编译复刻 6 用例矩阵，create/destroy/clean 全对（独立佐证）；
      ③ `shutdown` 幂等且终态——之后 `get_tc_util()` 抛 `RuntimeError`（正文带 `lazy.hpp:27`）；
      **E1 必须真给 GPU**（无探针时真库报 `the library was not initialized`，`runtime.hpp:51`）；
      ④ `DG_USE_PYTORCH_CUBLASLT_HANDLE` 是**整数**不是布尔（`0→create1`、`1/2→0`、`-1→1`、
      `true/false/abc` 被 `DJ_PANIC` 拒）；⑤ fork safety 8/8 `exit 0`。与 GPU 架构无关，H100 整份可跑。
      配套博客 `content/posts/mega-moe-80-lazy-init-test-oracle`（索引页新增第 80 篇）。

### P71 增量（opt-in，补齐 SM 局部域一直缺席的「验收测试即正确性规范」）

- [x] **P71-1** **SM 局部域的验收测试即正确性规范**（`tests/test_locality_domain.py`；
      `deep_gemm/locality_domain.py`；`deep_gemm/utils/mlopart.py`；`csrc/apis/locality_domain.hpp`；
      `csrc/runtime/locality_domain.hpp`；`csrc/runtime/mlopart.hpp`；`csrc/utils/layout.hpp:142-152`；
      `deep_gemm/include/deep_gemm/common/types.cuh:54`）。
      第 34 篇（P24-1）拆了局部域的**实现**（pointer-chase 探针 / `try_probe` 两闸门 / MLOPart + CUDA VMM
      本地化分配 / 调度器按域分区），本篇补上其「对外行为算什么对」的唯一规范——`tests/test_locality_domain.py`
      （167 行，**纯 host / API 层，一个 GPU kernel 都不启动**）。没用 `unittest`，5 个顶层函数按序调用
      （`:160-167`）；两个 helper 定契约——`assert_raises`（`:19-24`）**只吞 `RuntimeError`**（把「越界」与
      「不可用」编码成两种异常）、`as_plain`（`:67-70`）是 `localize` 的逆变换。**环境分水岭**：
      `is_localization_available()` 跨语言链（`apis:25-27` → `mlopart.hpp:24-26` → `mlopart.py:63-64/55-60/
      28-37/113-130/94-110`），末端 `serve()` 断言 2 个 device（`mlopart.py:99`）；本机 MPS 只给 1 个 →
      `available=False` + 告警一次（`mlopart.py:37`），本地化分配全线不可用
      （`empty`→`apis:34-40`→`runtime:72-81/34-38`→`mlopart.py:39-44`）。**域表规范**
      `test_sm_locality_domains:32-49` 四条断言：两域非空/dtype uint8 `:39`、TPC 内两 SM 同域 `:42`、
      恰好 50/50 `:45`、只搬最少 TPC `:47`；`get_sm`（`apis:120-124`）缓存在 `Runtime`、不可用走
      `get_even`（`:103-106`，132 SM → 66/66），`balance`（`:126-136`）的 `slice(0, num_tpcs/2)` 是
      **start 不是长度** → `surplus` 取多数域 TPC 的**后半段**（保证「已 50/50 时不动」）；把它读成
      「前一半」会把均衡输入翻成 `[0,132]`，被断言③当场抓（本实验利齿负样本）。**本地化内存规范**：
      `test_empty:52-64`（连续 + 逐位拷贝 + 域越界必须 `RuntimeError`；越界分支先于 MLOPart（`runtime:36`），
      **无 MLOPart 的 H100 也能验**）、`test_localize:73-105`（`:77` 硬门控；形状 `:86`、`stride(0)` 对齐粒度且
      浪费≤一个粒度 `:88-89`、`is_localized` 三分 `:90-94` 落点 `layout.hpp:150`）。**生命周期规范**：
      `test_mlopart:108-130`（起/服务/退 + 内存比分配器活得久 + 自动重启；`release_mlopart` 先算域表再
      `mlopart::release` `apis:144-147`）、`test_without_mlopart:133-157`（「剥掉 `-mlopart` 的假 MPS」模拟
      老宿主，断言**告警一次 + 拒绝 `localize` + 域表照给 even** = 优雅退化契约）。H100 实测
      `harness/exp_locality_domain_test_oracle.{py,sh,out.txt}` **16 PASS / 0 FAIL**：原样跑两个可跑用例
      （`test_sm_locality_domains` → `[66,66]`、`test_without_mlopart` → `detected, localize refused,
      SM table available`）；独立复刻 `get_even`/`balance` 与真实 API **逐位相等**、四条规格断言全过、利齿
      变体 `[0,132]` 被③抓；契约负样本 E1–E4（`assert_raises` 只吞 RuntimeError、`is_localized(plain.mT)` 抛
      `layout.hpp:150`、`empty` 越界抛 `runtime:36`、无 MLOPart 时 `empty(合法域)`/`localize` 抛
      `AssertionError`）；跑不了的 3 用例阻塞点全部定位到具体行。**与 GPU 架构完全无关**。配套博客
       `content/posts/mega-moe-81-locality-domain-test-oracle`（索引页新增第 81 篇）。

### P72 增量（opt-in，补齐整条测试栈最后一个顶层规格文件：legacy 家族）

- [x] **P72-1** **legacy Triton grouped GEMM 的验收测试即正确性规范**（`tests/test_legacy.py`；
      `deep_gemm/legacy/{m_grouped,a_fused_m_grouped,a_fused_k_grouped,b_fused_k_grouped}_gemm.py`、
      `tune_options.py`；`tests/generators.py:118-263/378-419/551-608`；
      `csrc/jit_kernels/heuristics/runtime.hpp:18/53-54`）。第 29 篇（P29-1）拆了 legacy 四 kernel
      的**实现**，但整条测试栈里**唯一还没被单独拆过的顶层规格文件** `tests/test_legacy.py`（95 行、
      两个顶层函数、**无任何 arch 门控**、只测 bf16）一直只被顺带引用。**纯 Triton、架构无关，
      H100 真机可跑真机实测**。→ `docs/82-legacy-test-oracle.md`（真实行号核对）：顶层结构
      `:1-12/14/50/87-95`（无 `test_filter`）；**m-grouped** `:16` 解包 10 元组、
      `enumerate_m_grouped_contiguous`（`generators.py:194-215`）给 32 配置（含 `get_psum_layout_usage
      =(True,False)`，`:118-119`）×`expand(2)×test_alias(2)`=**128 次发射**，但第 9 字段
      `use_psum_layout` 被 `_` 丢弃、generator 默认 `False`（`:381`）→ 仅 16 唯一；`test_alias`
      `:24-27` 是**主序别名薄壳**（A 必 K-major `:25`、MN-major 时 `b=b.mT` `:26`，`nn`=`nt`+`B.mT`
      `m_grouped_gemm.py:82-84`，两次 `.mT` 抵消）；`expand` `:28-32` 用**恒等** `m_row_indices`
      调 `a_fused_m`（impl `:45` gather A 行、`:54` `(rows>=0)` 掩码）；`:33` 先归零 `-1` 填充行、
      `:34-35` 判据 `calc_diff<0.001`（=`generators.py:69-74` 的 legacy 默认）。**关键分支差异**：
      `m_grouped` 有「首行 `batch_id<0` 整块写零」早退（`m_grouped_gemm.py:36-40`），
      `a_fused_m` **没有**——恒等行索引下靠 generator 已清零 A 填充行（`:405`）才恰为 0，
      任意非负置换会让填充行算出非零（实测 639 行 / ~100 万非零元素）。**k-grouped** `:52-53`
      解包 13 元组、四因子跳过闸门 `:54-56`（注释「Legacy Triton kernels do not mask per-group K
      tails」）、`:61-70` 双 fused_operand（索引 A `a_fused_k` vs 索引 B `b_fused_k`，镜像 gather
      `:54`/`:56`）、`k_indices=arange` 恒等 + `k_start/k_end` 用**对齐后**组 K 前缀和、数据放每组
      物理块开头（`generators.py:567-570/576-577/585-587`）解释「不裁尾块也能对」。闸门因子表
      （psum / k_align / no_acc / out_dtype）逐条给 kernel 侧原因（空组 `ACC` 取舍 `:43-47`/`:44-48`、
      累积读回 D `:63-64`、`restore_value=['d_ptr']` `:10`、`tune_options.py:5-23` smem 166912）。
      H100 实测 `harness/exp_legacy_test_oracle.{py,sh,out.txt}` **19 PASS / 0 FAIL**：
      ① **直接 import 真实测试模块、驱动两个真实测试体**（只打桩 `bench_kineto`，形状不缩）
      16.8s 全过；② 枚举计数（m 32 / k 7）与 `use_psum` 默认值；③ 闸门：arch9 7/7 跳过 0，
      **伪造 arch10 枚举 126 条跳过 119**（四因子直方图齐全）；④ `nn`==`nt` **逐位**（0.0）、
      `expand` 恒等==`m_grouped` **逐位**（0.0）、`nt` vs 独立 torch oracle 2.99e-09、expand 置换
      有效行 3.06e-09；⑤ 判据利齿：`calc_diff` 对 `d*0.999` 仅 5.0e-07、单行归零漏（1.5e-05）、
      1/8 行归零抓（6.8e-02）、1-ulp 只有逐位抓；真实 perf m-grouped（G=8,m=33792,n=7168,k=7168）
      **223.1 TFLOPS/115 GB/s**、k-grouped（G=4,m=4096,n=7168,k=33792）**52.5 TFLOPS/33 GB/s**。
       配套博客 `content/posts/mega-moe-82-legacy-test-oracle`（索引页新增第 82 篇）。

### P73 增量（opt-in，补齐 DeepEP 一直缺席的 host Python 编排层与句柄契约）

- [x] **P73-1** **DeepEP 的 `EPHandle` 句柄契约与 `dispatch`/`combine` 的 host 编排层**
      （`deep_ep/buffers/ep.py:27-906`；`deep_ep/utils/{semantic,math,event,envs}.py`；
      `deep_ep/buffers/allocator.py`；消费方 `~/github/DeepGEMM/tests/test_mega_moe.py:307-333`）。
      DeepEP 的**设备侧**路径都已拆透（第 7/11/48/51/52/53/61/63/65 篇），但把它们拼起来的
      **host Python 编排层**与 **`EPHandle` 本身**从没被单独讲清。**纯 host / Python，H100 可跑**。
      → `docs/83-deepep-epbuffer-handle-orchestration.md`（真实行号核对）：`EPHandle`（`:27-207`）
      的两个身份——① **接口**（`test_mega_moe.py:307-333` 把 `handle.psum_num_recv_tokens_per_expert`
      当 grouped GEMM 的 PSUM 布局）、② **反向缓存**（`ep.py:613-615` 只换 `topk_weights` 复用 expand
      布局）。**字段契约最易读错点**：两种 psum 语义不同（`:43-53`）——`...per_scaleup_rank` 是**去重**
      后每 rank 含端点前缀和（末元素=总数，`deterministic_sort` 在 `do_cpu_sync=False` 时当有效行数
      `:141`）；`...per_expert` 非 expand 是普通前缀和、expand 是「`psum[i]=align(psum[i-1],alignment)
      +专家 i 未对齐真实计数`」（真实计数=`psum[i]-align(psum[i-1])`、`i+1` 起点=`align(psum[i])`）。
      `topk_idx` 是带**版本守卫**的 property（`:103-111`，inference 张量守卫关闭）。**dispatch 五段**
      （`:562-757`）：`:638` 禁 `do_handle_copy`+`:639` `check_torch_deterministic`（确定性算法 ∧
      `fill_uninitialized_memory` 同开报错，因 `torch.empty` 初始化内核会与通信流重叠）→ `:641-645`
      auto SM/QP（`num_topk` 从 `topk_idx.shape[1]` 推、`num_qps<=num_allocated_qps` 断言）→ `:647-671`
      SF 解包 + 缓存句柄三断言（`:662` 上下文一致是**裸断言**）→ `:673-699` `value_or` 回填默认 + 调
      runtime → `:702-747` `finalize_dispatch`（句柄装配/复用 + 确定性 hook vs 立即 + 返回形状归一化）；
      `:749-757` **deferred epilogue** 返回 `EventOverlap`、`wait()` 才 finalize。**combine**（`:770-843`）：
      SM 默认**复用 `handle.num_sms`**、直接喂 handle 的 6 字段、`_unpack_bias` 支持 0/1/2 bias。
      辅助基座 `value_or`/`weak_lru`（弱引用防泄漏，`semantic.py:5-26`）/`align`。H100 实测
      `harness/exp_deepep_epbuffer_handle.{py,sh,out.txt}` **51 PASS / 0 FAIL**：注入合成 `deep_ep._C`/
      `comm` 桩后**按文件路径 exec 真实文件**、把**未修改的真实方法体**绑到假 buffer 逐字段核对
      （编排不碰 NCCL，等价验证）；`deterministic_sort` 用真实 CUDA 张量在 H100 跑。实测要点：auto SM
      干净地从 `topk_idx.shape[1]` 推 `num_topk`、缓存 handle 返回同一对象且强制关 CPU 同步、
      `num_experts` 不一致被 `:662` 裸断言拦下（消息实测为**空串**）、deferred/确定性两条接线分支、
      `value_or(0/False)` 不误判。配套博客 `content/posts/mega-moe-83-deepep-epbuffer-handle-orchestration`
      （索引页新增第 83 篇）。

### P74 增量（opt-in，补齐「DeepGEMM 的 Python 绑定面」）

- [x] **P74-1** **DeepGEMM 的 Python 绑定面**（`csrc/python_api.cpp` + 11 个 `register_apis`
      在 `csrc/apis/*.hpp` + `deep_gemm/__init__.py` + `deep_gemm/mega/__init__.py`）。
      前面 83 篇把**内核/测试**拆得很细，但「它们在 Python 里到底怎么被调到」从没单独讲清。
      **整层是 pybind11 host 绑定，与 SM100/UMMA 无关，H100 可跑、真机实测**。
      → `docs/84-deepgemm-python-api-surface.md`（真实行号核对）：`python_api.cpp:24-49` 的
      `PYBIND11_MODULE(_C)` 装配（1 个 `deep_jit::register_python_api` + 11 个 `register_apis`、
      `mega::` 排在 `gemm` 之后的「C++ 裸入口先注册、Python 包装后转发」）；`_C` 顶层 **76 名**
      = 62 `m.def` + 10 `m.attr` 别名 + {`Runtime`,`get_jit`} + {`epilogue`,`locality_domain`}
      （真机对账 0 多余 0 缺失）。**三组默认值契约**：① `compiled_dims` 随主序变（`nt/nn`=`nk`、
      `tn/tt`=`mn`，`gemm.hpp:842/850/858/866`，18 入口 18/18）；② `recipe` 按 dtype 分
      （`k_grouped_fp8`=`(1,1,128)`、`k_grouped_fp4`=`(1,1,32)`、`fp8_einsum`=`(1,128,128)`，
      `gemm.hpp:895-919`/`einsum.hpp:304-308`）；③ `alpha`/FP8 输出对是 `Alpha`/`FP8Quantization`
      的**同义简写**（`epilogue_class.hpp:11-12`），真机 `alpha=2.0` ≡ `epilogue.Alpha(2.0)`
      **逐位**。**别名机制** `gemm.hpp:920-930` 的 `m.attr("新") = m.attr("旧")`（FP8/FP4 统一命名）
      = **同一 pybind 函数对象**（`is` 10/10，非复制）。**两个子模块**：`epilogue`（基类 + 4 工厂
      类，`isinstance` 继承链为真、`Alpha` 无参抛 `TypeError`）、`locality_domain`（9 函数）。
      **公开 vs 隐藏**：`__init__.py:16-127` 显式 re-export，5 个入口**故意隐藏**（`Runtime`/
      `get_jit`/`init`/`get_symm_buffer_size_for_mega_moe`/`get_token_alignment_for_mega_moe`），
      5 个是 **Python-only mega 包装**（`.mega` 把 `SymmBuffer` 拆成裸参数转调 `_C`）；无 `py::arg`
      的入口退化为 `argN`（`get_block_m_for_mega_moe` 末位是 `activation`）。两行副作用
      `_C.init(dirname)` + `atexit.register(shutdown)`（对应第 33/50 篇）。H100 实测
      `harness/exp_deepgemm_python_api.{py,sh,out.txt}` **51/51 PASS**（解析源码 vs 真机 `_C`
      逐条对账、别名同一性、`compiled_dims`/`recipe` 默认值、子模块工厂、隐藏面、`bf16_gemm_nt`
      冒烟 relerr 1.77e-5、`alpha`≡`epilogue.Alpha` 逐位）。配套博客
        `content/posts/mega-moe-84-deepgemm-python-api-surface`（索引页新增第 84 篇）。

### P75 增量（opt-in，补齐 DeepEP 侧一直缺席的「验收测试即正确性规范」）

- [x] **P75-1** **DeepEP 的 EP dispatch / combine 端到端验收测试即正确性规范**
      （`tests/ep/test_ep.py` 616 行；`deep_ep/buffers/ep.py:27-105` 的 `EPHandle` 字段契约；
      `deep_ep/utils/{refs,gate,math,testing}.py`；消费方 `~/github/DeepGEMM/tests/test_mega_moe.py:307-333`）。
      DeepGEMM 侧测试栈已全部拆过（第 40/41/43/51/57/60/67/74/76/78/79/80/81/82 篇），DeepEP 侧也拆过
      `refs/gate/testing`（P55-1）与 `EPHandle` host 编排（P73-1），但把 dispatch/combine
      **「算什么对」**定义清楚的旗舰验收测试本身从没被单独拆过。本篇补上——它是第 7/11/48/51/52/61/
      63/71/83 篇里反复引用的「逐位一致 / expand 不变量 / 两套 psum」判据的出处，也是运行时 SM/QP、
      overlap、带宽数字的来源。**端到端 buffer 路径依赖 DeepEP 的 C++/NCCL 设备 API
      （`ncclDevComm`/`ncclGin`），本机 NCCL 2.29.7 不匹配、运行需 ≥2.31 → 端到端仅分析**；
      **测试侧纯 Python/张量逻辑架构无关，H100(sm90) 真机实测**。
      → `docs/85-deepep-ep-test-oracle.md`（真实行号核对）：**顶层结构**（`enumerate_ep_modes:22-30` /
      `launch:33-43` / `fold_expanded:46-58` / `test_dispatch_combine:61-523` / `test_loop:527-567` /
      CLI `:570-616`；mp.spawn 默认 8 进程）。**模式矩阵**（六维：`expert_alignment∈{128,1}` ×
      `use_fp8_dispatch` × `num_bias∈{0,1,2}` × `with_previous_event` × `async_with_compute_stream` ×
      `allocate_on_comm_stream`，后两者受 `with_previous_event` 约束）组合数公式
      `2 × fp8 × 3 × Σ_prev(async × alloc)`；实测 72/36/36/18（`--fp8-dispatch-only`、`--defer-epilogue`
      各钉一维）。**五种 dispatch + 两种 combine**：普通（`:146-155`）/ expand（`:159-162`，
      `do_expand`+`use_tma_aligned_col_major_sf`）/ cached（`:165-172`，只传 handle 复用布局）/
      cached-expand-零填充（`:175-179`）/ FP8 激活（`:97/103-107`）；combine 普通（`:211-219`）与
      reduced（`:221-233`），配方 `(reduce_in_local,reduce_in_scaleup)`（`:116-126`）对应 P55-1 的
      hybrid/non-hybrid/disabled 三配置。**逐位判据**：`ref_dispatch`/`ref_combine`（`:110-142`）为参照，
      **排序后**比 `torch.equal(ref_recv_src_token_idx, sorted(recv_src_metadata[:,0]))`（`:188-192`，
      排除 dispatch 行序自由度）、按来源 rank 切段用 `sorted_metadata.indices` 重排再逐位比
      （`:478-507`，权重额外按 `ref_mask=ref_recv_topk_idx<0` 屏蔽）；combine `torch.equal`
      （`:510/512`）+ 结构性断言 `combined_topk_weights==topk_weights`（`:514`）。**三条不变量**：
      ① `fold_expanded`（`:46-58`，expand 同 token 各有效槽内容必须相同，`(gathered==folded)|~mask`
      全真，tuple 递归）；② handle 两套 psum 恒等式（`:452-476`，非 expand `psum[i+1]-psum[i]==
      align(count_i,a)`；expand `psum[i+1]-align(psum[i],a)==count_i`；scaleup 前缀和去重、末元素=总数）；
      ③ 零填充（`:431-435`，`[psum[i],align(psum[i],a))` 全零）与确定性双跑（`:394-411`）。
      **带宽账目** `get_combine_bytes`（`:298-343`）的 `get_unique_and_valid_dst_count`（`:302-315`，
      行内去重 + 忽略项折成同一哨兵值只计一次；三归约配方）+ bench_kineto 按名拆
      `dispatch_impl`/`dispatch_copy_epilogue_impl`、`combine_impl`/`combine_reduce_epilogue_impl`。
      `EPHandle` 契约（`ep.py:27-60`）：`handle.topk_idx is topk_idx`（`:371-373`，P73-1 版本守卫的意义）、
      `dst_buffer_slot_idx`/两套 psum/cached 三方逐位（`:442-450`）、`recv_src_metadata[:,2:]` 的 expand
      槽位（`:390-392`）、`cumulative_local_expert_recv_stats`（`:413-416/458-459`）。**H100 实测**
      `harness/exp_deepep_ep_test.{py,sh,out.txt}`（`sys.modules` 桩绕过 `check_nccl_so`、按文件路径
      exec 真实 `refs/gate/math`、`ast` 抽取顶层函数原样运行、8×H100 NCCL 跑 `refs.dispatch`）：
      **17 项 ALL PASS**——模式矩阵 4/4、`fold_expanded` 3/3、psum 恒等式 4/4、去重计数 2/2、
      8×H100 `refs.dispatch` 4/4（本 rank 收 1753 行、每来源计数 `[207,225,206,232,210,222,230,221]`）。
      `tests/ep/test_ep.py` sha256 `7346d98a17bce91e…`。配套博客
      `content/posts/mega-moe-85-deepep-ep-test-oracle`（索引页新增第 85 篇）。

### P76 增量（opt-in，补齐 DeepEP 一直缺席的「NCCL 通信域绑定 + Python 引导层」）

- [x] **P76-1** **DeepEP 的 NCCL 通信域绑定与 Python 引导层**
      （`deep_ep/__init__.py`；`deep_ep/utils/find_pkgs.py`；`deep_ep/comm/{__init__,handle,barrier}.py`；
      `csrc/python_api.cpp`；`csrc/comm/{handle,barrier,stream}.hpp`；`csrc/kernels/comm/{api.hpp,context.cpp,barrier.hpp}`；
      `setup.py`；`common/compiled.cuh`）。前面第 7/48/51/53/58/61/63/65/71/73/75 篇的 DeepEP 篇目
      **全部从一个已存在的通信组/对称窗口讲起**，但「怎么把 PyTorch 的 NCCL 通信组变成一块可对称
      寻址的窗口」这层地基从没被单独拆过——也正是本机**运行期阻塞的确切落点**。**纯 host / Python，
      H100(sm90) 真机实测**（`import deep_ep` 触发 `check_nccl_so` 失败，故用 stub + ast 抽取真实
      函数体）。→ `docs/86-deepep-nccl-comm-binding.md`（真实行号核对）：
      **引导协议** `__init__.py:8-16/19-41/44-51/54-56`（persistent_envs → `check_nccl_so` →
      `init_jit` → 之后才 import `_C` 消费者）；**包发现** `find_pkgs.py`：`find_pkg_root:9-55`
      （env 短路 `:22-25` → `sys.path` 优先级 `:27-28` → `nvidia-{name}` 匹配 `:30-33` → lib 目录上跳
      `:40-50`）、`find_nccl_root` 带 `lru_cache` `:58-69`、`get_nccl_lib_name` SONAME `:72-77`
      （pip wheel 只发 `libnccl.so.2`，故链接用 `-l:libnccl.so.2`）；**运行时核对** `check_nccl_so:19-41`
      三闸门（抑制开关 `:24-25`、`/proc/self/maps` 重复加载 `:27-32`、`filecmp(...,shallow=False)`
      **二进制级**比较 `:33-41`——为何不比版本号字符串）；**编译期链接** `setup.py:14-20/63-74/81-83/90-93`
      （persistent_envs 固化、NCCL include+lib+`-l:`+rpath、`_C` 仅三个 host 源文件）；**绑定面**
      `python_api.cpp:19-51` + `comm/handle.hpp:8-30`（Context 类 / barrier lambda / 裸 NCCL 原语）；
      **`Context` 构造** `context.cpp:75-224`：入参断言 `:94-98`（2 MiB 对齐）、**版本闸门** `:101-106/156-162`
      （`NCCL_VERSION_CODE>=2.31` → `useRuntimeVersion`，否则断言编译期==运行时）、GIN 需求 `:115-155`
      （GDAKI `:130-134`、`ginSignalCount=num_ranks+4 :140`、连接类型 `:141-146`、SL 三级 `:148-154`）、
      **两域拆分** `:166-183`（`ncclTeamLsa`、混合进制 rank、`is_scaleup_nvlink`）、对称内存+window
      `:185-223`（`ncclCommWindowRegister:199`、`ncclGetLsaDevicePointer:200`、RDMA storage windows
      `:207-223`）；`get_sym_ptr:226-232`（offset=ptr−workspace 对称翻译）、`finalize:244-266`（逆序
      释放、用后 `not finalized` 断言 `:235/:240`）。**Python comm 层** `comm/handle.py`：
      `NCCLCommHandle:12-38`（managed 语义、`__del__`）、`get_nccl_comm_handle:44-77`（缓存 `:57-60`
      / 复用 PyTorch `_comm_ptr` `:62-66` / 自建 `:68-77`、`force_new_comm` 键 `time_ns`）、域大小包装
      `:80-95`；`comm/barrier.py:4-11` + C++ `comm/barrier.hpp:9-42`（`with_cpu_sync` vs `wait_comm_stream`
      前后对齐）+ `launch_barrier` `csrc/kernels/comm/barrier.hpp:17-54`。H100 实测
      `harness/exp_deepep_nccl_binding.py` + `.sh`/`.out.txt`（真实模块 + 合成 `_C` stub + ast 抽取
      `check_nccl_so`）**35 PASS / 0 FAIL**：真实 `find_nccl_root`=`.../nvidia/nccl`、lib=`libnccl.so.2`、
      env 优先级、lru_cache、独立 oracle 一致；**真实失配** loaded 系统 `2.29.7` vs linked pip `2.32.3`
      → `check_nccl_so` 抛 `Invalid NCCL versions`；句柄复用 managed=False / 自建 managed=True / 析构
      销毁语义、缓存清理；域算术 8 配置 vs oracle、`ginSignalCount=num_ranks+4`、对称偏移不变；版本闸门
      `>=2.31` 走 `useRuntimeVersion` 但仍被 Python 闸门先拦。**结论**：阻塞=两个闸门（Python 二进制
      核对 + C++ 设备 API 版本）都要求 `_C` 与 PyTorch 共用同一份 ≥2.31 的 NCCL。配套博客
       `content/posts/mega-moe-86-deepep-nccl-comm-binding`（索引页新增第 86 篇）。

### P77 增量（opt-in，补齐 Mega MoE 的跨 SM / 跨 rank 同步地基）

- [x] **P77-1** **DeepGEMM 的跨 SM / 跨 rank 同步原语 `comm/barrier.cuh`**
      （`deep_gemm/include/deep_gemm/comm/barrier.cuh`；`layout/sym_buffer.cuh`；
      `layout/mega_moe.cuh`；`impls/sm100_fp8_fp4_mega_moe.cuh` / `sm100_bf16_mega_moe.cuh`）。
      第 11 篇在 dispatch/combine 的「原语表」只列了 `grid_sync:33-51`/`nvlink_barrier:53-92` 两行；
      第 38 篇拆 PTX 层时只把它们当内存序调用点，并注明「`grid_sync` 整体语义因共享 GPU 会死锁，
      仅源码解读」。本篇把这层**单独**拆开。**代码全架构无关（不碰 UMMA/TMA），H100(sm90) 直接
      include 真实头文件跑真实设备代码**。→ `docs/87-mega-moe-sync-primitives.md`（真实行号核对）：
      四原语 `wait_until:14-24`（`kNumTimeoutCycles=60s@2GHz :12`、`clock64` 超时 + `print_timeout`
      + `DG_DEVICE_ASSERT`）、`cluster_sync_with_relaxed_arrive:26-31`（`cute::cluster_arrive_relaxed`
      + `cluster_wait`，弱序，只用于 2-CTA TMEM 分配 `impls:237-238`）、`grid_sync:33-51`、
      `nvlink_barrier:53-92`；设备端对称指针 `SymBuffer`（`sym_buffer.cuh:9-42`：`base`+`offsets[72]`
      **相对偏移**（`:23-24`）而非绝对指针 → 同一张表可编进所有 rank 的 `__grid_constant__`；
      `map:33-40` 单行翻译、`kNumRanks==1` 快路 `:35-36`；`kNumMaxRanks=72 :7`）。
      **两个免复位设计**：① `grid_sync` 每轮增量之和**恰好 2³¹**（`kFinishSumTag=0x80000000 :38`；
      `sm_idx==0` 加 `2³¹-(N-1)`、其余 +1 `:42-43`），只翻最高位、两轮回到原值、永不需清零，
      `(new^old)&0x80000000` 即「全体到齐」（`:45`）；② `nvlink_barrier` 的 `counter&3` 编码
      `(phase,sign)`（`:69-70`），phase 交替用 `signal[2]` 两槽、sign 每两次翻转让 `signal[phase]`
      先 +`kNumRanks` 加满再 −1 清回 0（`:75/80-85`），4 次一个周期、自动归零。跨设备必须
      `red_add_rel_sys`/`ld_acq_sys`（`:75/82`）。信号住 `MegaMoESignals`（`layout/mega_moe.cuh:55/62-64`，
      `get_*_ptr` `:151-167`）；调用点 `impls:238/393-396/422-428/683-689/1412-1415/1440`（dispatch 用
      `GridSyncIndex=0`、epilogue 用 1 `:338-339`；两个 NVLink `kTag=1/2 :323-324`）。**H100 实测**
      `src/comm_barrier.cu` + `harness/exp_comm_barrier.{sh,out.txt}`：`grid_sync` 6 组 SM 数
      （1/2/8/16/64/7，含奇数）× **2000 轮 mismatch=0**，计数器按 `(r·2³¹) mod 2³²` 在 `0/2³¹` 翻转
      （免复位坐实）；`nvlink_barrier` 单卡虚拟 N∈{2,8}（offsets 全 0）状态机 **trace_bad=0**、
      两信号槽每 4 次精确归零；**真实 8 卡** UVA peer 对称缓冲 × 30 轮 **mismatch=0**（`st_rel_sys`
      写、barrier 后 `ld_acq_sys` 读全部 8 peer）；cluster relaxed **1/1**（记录，非保证）。
      另记录一个坑：`grid_sync` 是同步点不是读保护，首版每轮复用同槽产生误报、改每轮独立槽后归零。
       总判定 **ALL PASS**。配套博客 `content/posts/mega-moe-87-mega-moe-sync-primitives`
       （索引页新增第 87 篇）。

### P78 增量（opt-in，补齐 Mega MoE 一直缺席的「官方 profiling 基础设施」）

- [x] **P78-1** **DeepGEMM 自带的 Mega MoE profiling 基础设施 + 自动生成的「覆盖面清单」**
       （`scripts/run_ncu_mega_moe.sh` / `tests/test_mega_moe.py` 的 profile 模式 /
       `scripts/quick_plot_pm.py` / `scripts/generate_pyi.py` + `stubs/_C.pyi` /
       `csrc/indexing/main.cu` + `CMakeLists.txt`）。第 3/9/10/11 篇拆了主 kernel 内部、
       第 15 篇用 `ncu` 量了反向 SM90 的**平均** metric、第 41 篇讲 `test_mega_moe.py` 的**正确性**、
       第 84 篇人工枚举 Python API 面；本篇补**官方的性能验证与工具链面**。**目标 kernel SM100+DeepEP
       端到端仅分析**；但外壳全部 H100(sm90) 真机验证。
       → `docs/88-deepgemm-profiling-infra.md`（真实行号核对）：`run_ncu_mega_moe.sh`（`:8-46` 自写参数
       解析、`:48` `DG_JIT_WITH_LINEINFO=1`、`:50-53` 预热+sleep、`:55-74` ncu 参数每条为什么
       （`--kernel-name sm100_fp8_fp4_mega_moe_impl`、`--section PmSampling`+`SourceCounters`、
       `--replay-mode application`+`--communicator tcp`+`--communicator-tcp-num-peers` 三件套、
       `--clock-control none`、`--disable-pm-warp-sampling`）、`:79-88` 每 rank 一个 ncu 并行 + wait；
       `test_mega_moe.py` 的 `--ncu-profile-only`（`:233-241`）/`--local-rank-idx`（`:461-463`，NCU
       不能 wrap `mp.spawn`）/`--dump-profile-traces`（`:362-363` `bench_kineto` trace）+ 性能账目
       （`:370-410`：FLOP×3、HBM 按 `num_touched_experts`、NVLink `recv·H·3`、`t_reduction` 串行归约
       近似与 `approx_factor`）；`quick_plot_pm.py` 45 条 curated 指标 6 类（`:77-123`）、
       `_query_available_metrics` chip 查询+去 TriageCompute 前缀+别名回退（`:134-181`）、
       `_probe_metric_series` 从 raw CSV **第 11 列**取实例序列（`:201-224`，`len<=11` 缺列返回 None）、
       `_parse_metric_values` 三形态（`:184-198`）、`collect_metric_series` 只收 `len>1`（`:227-248`）、
       `plot_pm`（`:274-417`）。配套：`generate_pyi.py`（`:5-94` 正则 C++ 函数索引、`:151-343`
       `m.def` 抽取+BracketTracker、`:608-826` 类型翻译）→ `stubs/_C.pyi`（62 def = 48 真签名
       + 14 lambda 回退；**不含 10 个 `m.attr` 别名**，故 62≠运行时 76 名）；`csrc/indexing/main.cu`
       + CMake（IDE 索引单 TU，「真正编译走 JIT」），漏 `sm100_mega_mhc.cuh`/`sm100_sparse_mqa_logits.cuh`
       （17/19）。**发现不可复现 bug**：`build_cpp_function_index` 用 `rglob`、同名只留首次命中，
       `get_tma_aligned_size`（真定义 `csrc/utils/math.hpp:23`）在 `csrc/jit_kernels/impls/sm100_mqa_logits.hpp:240`
       的调用点会被误抓 → 遍历顺序不同则 pyi 不同；强制构造确定性复现「唯一退化函数」。
       H100 实测 `harness/exp_megamoe_profiling.{sh,py,out.txt}`：桩 `ncu`/`python` 拦截真实脚本捕获
       argv（num-peers=3、-o 按 rank、预热无 NCU）；真实 `.ncu-rep` 跑 `--page raw --csv`
       （12/266 列）验证取址；真实 `test_mega_moe.py` 公式抽取执行 9 项；真实 `generate_pyi.py`
       逐字节复现 + 强制退化复现；**47/47 PASS**。配套博客
        `content/posts/mega-moe-88-deepgemm-profiling-infra`（索引页新增第 88 篇）。

### P79 增量（opt-in，补齐 Mega MoE / mHC 的 JIT 启动包装层）

- [x] **P79-1** **Mega MoE / Mega mHC 的 JIT 启动包装层**（`csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp`
       157 行 / `csrc/jit_kernels/impls/sm100_mega_mhc.hpp` 149 行）：前面第 5/10/42 篇分别拆过
       **kernel 本体**与 **Python/host 缓冲区编排**，这两份夹在 `csrc/apis/*.hpp` 与 SM100 主体 kernel
       之间的包装层却一直没人单独读过。主 kernel SM100 专属、仅分析，但 wrapper 是**纯 host C++**。
       → `docs/89-mega-moe-mhc-jit-wrappers.md`（真实行号核对）：三段式结构；BF16 wrapper
       `:33-44`（形状二次推导 + `MmaKind::BF16` + `store_block_m_l1==l2` 断言）、`:46-86`
       10 个描述符（v.s. FP8 `sm100_fp8_fp4_mega_moe.hpp:50-139` 的 20 个含 SF）、`:65-86`
       共享专家**指针别名回退**、`:96-130` 20 个模板实参 + `to_string(activation_clamp)` hexfloat +
       `fast_math`、`:132-154` launch（grid=num_sms、block=512、cluster=2）。**关键差异**：L1 输出
       swizzle BF16 取 `swizzle_acts_mode`（128）/ FP8 取 `/2`（64），靠 `make_tma_2d_desc`
       `runtime_utils.hpp:130-131` 的 `smem_inner=swizzle/elem` 规整补偿（二者 smem 内层 64 元素、
       swizzle 字节数随元素大小减半；BF16 照抄 `/2` 会被压成 32）。mHC wrapper `:37-51` 三个 TMA
       lambda（`make_residual_tma_desc` **route 当第三维** + 两个显式 stride）、`:68-73` `fn` 三维
       描述符 `allow_tf32=true`→`TFLOAT32`、`:74-78` coeff swizzle 0 vs 64、`:82-107`
       `MixArgs`/`NormArgs` POD（`layout/mega_mhc.cuh:45-71`）、`:109-126` 7 个模板实参（三个 bool
       由指针空值推出）+ launch（无 cluster、768 线程）、`:56-62` `DG_PRINT_CONFIGS`；上游
       `csrc/apis/mega_mhc.hpp:89-252`（num_splits/scratch/借用 bf16 尾部/primary SF）。H100 实测
       `src/mega_moe_mhc_jit_wrapper.cu` + `harness/exp_mega_moe_mhc_jit_wrapper.{sh,out.txt}`
       （nvcc 直接 include 真实头文件 + torch）：`to_string` 7 有限值字面量 `strtof` 逐位回原值、
       ±inf→`infinity()`、NaN 被拒；22 个 `offsetof` + `MixArgs`=64/`NormArgs`=88 逐项相等；
       mHC 四个描述符与独立直连编码逐字节相等 + **真实 3D TMA 载入 route 当第三维 0/32768 不符**；
       真机 `get_mega_moe_config`（BF16 block_k=64 / FP8 128、BF16 `num_sf_ring_tokens=0`、BF16
       smem 218532 vs FP8 225700、共享回退逐字节别名）。**50/50 ALL PASS**。另一编译坑：真实头
       `DG_UNIFIED_ASSERT` 在 nvcc 单 TU host pass 落成 `asm("trap;")`，include 前钉到 host `assert`。
        配套博客 `content/posts/mega-moe-89-mega-moe-mhc-jit-wrappers`（索引页新增第 89 篇）。

### P80 增量（opt-in，补齐两库的「启动前地基」：驱动符号懒加载 / 版本闸门 / 张量视图 helper）

- [x] **P80-1** **两库的底层地基**：DeepEP `csrc/utils/lazy_driver.hpp`（60 行）+ `csrc/utils/tensor.hpp`
       （27 行）；DeepGEMM `csrc/utils/compatibility.hpp`（12 行）+ deep_jit `utils/lazy.hpp`（71 行）/
       `backend/cuda/driver.hpp`（63 行）/ `utils/exception.hpp` + 最后一个未被任何 docs 引用的
       `csrc/jit_kernels/impls/sm100_locality_domain.hpp`（41 行）。**纯 host、架构无关，H100 可跑**。
       → `docs/90-lazy-driver-compat.md`（真实行号核对）。**三个非平凡点**：① DeepEP 的
       `get_driver_handle:10-18`（`dlopen("libcuda.so.1", RTLD_LAZY|RTLD_LOCAL)` + 函数内 static）
       与宏 `DECL_LAZY_CUDA_DRIVER_FUNCTION:23-34`（尾置 `decltype(name(args...))` 复原强类型、
       函数内 `static FuncType` 缓存、首次 `dlsym`），`EP_CUDA_STRINGIFY:20-21` 为什么必须**先展开
       别名再字符串化**（`cuda.h:96` 的 `#define cuMemGetAddressRange cuMemGetAddressRange_v2`，
       故符号表写 `_v2`）；② deep_jit 版用**初始化器 IIFE + `const`**（语言级线程安全）+ `dlerror()`
       可诊断报错（`lazy.hpp:42-54/58-71`），`LazyInit<T>:15-38` 的「null factory = 终态 / 工厂只跑
       一次」；两库 handle 因 `dlopen` 引用计数**返回同一指针**；③ 版本闸门分层——`compatibility.hpp`
       `:9-12` 卡 torch≥2.3 + `CUDA_VERSION/CUDART_VERSION≥12.9`（同时卡 driver/runtime 防混装头），
       deep_jit `driver.hpp:8` 只卡 `CUDA≥12.4`；`check_cuda_driver:44-57` 用 lazy 的 GetErrorName/String
       自举。另 `tensor.hpp` 的 `get_shape<kNumDims>`（index_sequence→tuple）与 `get_data_ptr`
       （nullopt→nullptr）；`sm100_locality_domain.hpp:13-15` 探针常量 4096/128→**32 hops**。
       H100 实测 `src/lazy_driver_compat.cu` + `harness/exp_lazy_driver_compat.{sh,out.txt}`：
       两库共 **50 个驱动符号全可解析**（deep_jit 27 / DeepEP 23）、handle 同一指针、`lazy_cuDeviceGet`
       ==直接 `cuDeviceGet`、**VMM + POSIX fd 可共享句柄导出→导入→跨句柄 2 MiB mismatch=0**（对称
       内存地基）、libdw 8/8 + trace 带真实行号、`LazyInit` 工厂只跑一次、`get_shape/get_data_ptr`
       四条全过、版本真实值 2.12/13020；**负样本**伪造 `CUDA_VERSION=11000` 被 `compatibility.hpp:11`
       `static_assert` 拦下。**30 PASS / 0 FAIL + 1 编译期负样本**。配套博客
        `content/posts/mega-moe-90-lazy-driver-compat`（索引页新增第 90 篇）。

### P81 增量（opt-in，以官方 SF 契约为准核验 host 契约代码）

- [x] **P81-1** **DeepGEMM 官方 SF 契约（`docs/scaling-factor-format.md`）的代码对照与 H100 一致性核验**。
      第 25 篇讲「SF 怎么被变换」、第 26 篇讲「K-grouped SF 怎么打包」、第 78 篇讲「test_layout.py 的逐位
      oracle」，但都以实现侧出发；官方 spec 文件是 SF 张量契约的 source of truth，却从没被这样对读。
      本篇补 spec §1.1 / §2 / §3 / §5.2 / §6.2 这些**契约校验本身**（§4 分派表归第 25 篇、§6.1 MQA 耦合
      归第 19 篇、§1.2/§7.1 打包位运算归第 25 篇，均不重复）。
      → `docs/91-sf-contract-cross-api.md`（真实行号核对）：**两条路径**（`csrc/apis/layout.hpp:38-55`，
      路径 B = 只调一次 `check_sf_layout(..., true, false, torch::kInt)`）、**§1.1 形状公式**
      （`csrc/utils/layout.hpp:112-119`，`:119` 的三元 `? 1 : 4` 同时表达 fp32/int32 末维）、
      **§3 预变换 stride 契约**（`utils/layout.hpp:122-130`：group 维紧 packed `:124`、MN-major `:126`、
      `get_tma_aligned_size`=`align(mn,4)` `:127-128`、`%4==0` `:129`；对照 `math.hpp:23-27` 的
      `ALIGN_MN=16/4`）、**§5.2 K-grouped 预打包**（`smxx_layout.hpp:286-317`：`gran_k==32` `:293`、
      `stride(0)>=mn` `:299`、下界 `:312`）、**§6.2 einsum 换回**（`einsum.hpp:268-269` permute SF，
      naive reshape 破坏 MN-major 撞 `layout.hpp:124`）、**§6.3 Mega MoE 额外要求**
      （`mega_moe.hpp:177` recipe 钉死 / `:201-204` kInt / `:367` arch 门控；`deep_gemm/mega/__init__.py:112-149`
      权重变换）。**关键观察**：`check_sf_layout` / `check_k_grouped_packed_ue8m0_tensor` **不含 arch 门控**，
      故 int32 packed UE8M0 虽是 SM100 专属，其 host 校验层可在 H100 直接跑真实代码。
      H100 实测 `src/sf_contract_conformance.cu` + `harness/exp_sf_contract_conformance.{sh,out.txt}`：
      `nvcc -arch=sm_90a` 直接 include 真实头文件，A 组 6 + B 组 5 + C 组 6 + D 组 2 = **19 PASS / 0 FAIL**，
      每条负用例打印真实 `DGException` 的 `file:line`（`:110/115/117/119/124/126/128/129`、
      `smxx_layout.hpp:293/299/312/314`），并坐实 C5「预分配更大 buffer 通过」/ D2「naive reshape 后
      post-permute stride `(1,4,20)` 被拒」。配套博客 `content/posts/mega-moe-91-sf-contract-cross-api`
       （索引页新增第 91 篇）。

### P82 增量（opt-in，补齐所有 kernel 共用的「共享内存布局结构层」）

- [x] **P82-1** **DeepGEMM 的共享内存布局结构层**（`deep_gemm/include/deep_gemm/layout/gemm.cuh`；
      `.../layout/mqa_logits.cuh`；消费方 `impls/sm100_{bf16,fp8_fp4}_gemm_1d1d.cuh` /
      `impls/sm100_mqa_logits.cuh`；host `csrc/jit_kernels/heuristics/sm100.hpp:209-252` /
      `csrc/jit_kernels/impls/sm100_mqa_logits.hpp:119-194/249-268`）。第 44 篇补了设备端数学地板、
      第 67 篇补了 TMA 拷贝层、第 56 篇补了 128 字节描述符来源，但**每个 kernel 开头
      `extern __shared__ ... smem_buffer[]` 之后共享内存怎么切、host 怎么算 `num_smem_bytes`**
      一直没被单独讲过。本篇补上 `layout/` 里**最后两个没被系统拆过的文件**。两个非平凡点：
      ① **精确镜像**（MQA）——host `MQALogitsConfig::smem_size()` 用 `add_region` 循环逐段复刻
      `MQALogitsSharedStorage` 布局，JIT 源码 `:254-256` 放 `static_assert(sizeof(...) == smem_size)`
      把二者钉死，故 `sizeof == num_smem_bytes`；② **保守预算**（稠密 GEMM）——`get_pipeline_config`
      的 `smem_barriers=kNumMaxStages*8*3+2*8*3+8=824` 一律按 32 级预留、`store_block_n` 由
      `swizzle_cd_mode` 反推、`num_stages` 用 `smem_capacity` 反算，故不必先知道 stage 数就能给上界；
      `SM100BF16GemmSharedStorage` 的 `reserved_barriers`（`gemm.cuh:31-32` 注释）正是把 BF16 的每
      stage barrier 数从 2 补到 3、以对齐 block-scaled 布局。结构体只含 constexpr/`alignas`/cutlass
      mbarrier、**不含 UMMA/TMA 设备指令**，故 H100(sm90) 直接 include 真实头文件逐字段核验。
      H100 实测 `src/smem_layout.cu` + `harness/exp_smem_layout.{sh,out.txt}`：**42 PASS / 0 FAIL**——
      A) 8 组 MQA 配置 `sizeof == host 镜像`（`alignof` 恰为 `kSwizzleAlignment` ∈ {256,512,1024}、
      `offsetof` 全落对齐、派生常量一致）；B) 8 组 GEMM 配置 `used = offsetof(tmem_ptr)+4 <= host`，
      且**`slack = 824 − 实际 barrier 字节` 精确**（BF16 S8 592 / FP8 S8 584 / FP4 S4 680），
      `sizeof − used` 的尾部补齐最多 1023B 不占 smem，CD/A/B 1024 对齐、SF stage 字节 == host
      `sf_block_m*block_k/32`；C) BF16 与 FP8FP4 每 stage barrier 数都是 3、总字节 232 vs 240 均 ≤ 824。
      配套博客 `content/posts/mega-moe-92-smem-layout-struct`（索引页新增第 92 篇）。

### P83 增量（opt-in，补齐 DeepEP bucket 一直当作细节略过的「会话式临时分配 + 生命周期守卫」）

- [x] **P83-1** **DeepEP BucketBuffer 的会话机制（`BucketSession`）与临时张量注册**（
      `deep_ep/buffers/bucket.py:19-181`（`BucketSession`）+ `:184-398`（`BucketBuffer`）；
      `deep_ep/buffers/allocator.py`；`deep_ep/buffers/base.py`；`deep_ep/utils/event.py`；
      `csrc/buffers/bucket.hpp:48-56/150/213/279/314`；`tests/bucket/test_session.py`、
      `tests/buffer/test_allocation.py`、`tests/bucket/test_all_gather.py:74-80`）。
      第 64 篇拆了 bucket 三类算子 × 三物理域、第 62 篇拆了 all_reduce/reduce_scatter 三后端，
      但常规集合算子「**张量必须注册在对称 buffer 里**」（`bucket.hpp:150/213` 裸断言）这条约束
      把临时 scratch 逼到死角——本篇补上封装它的 `BucketSession`。**纯 host Python，H100 真机**。
      → `docs/93-deepep-bucket-session.md`（真实行号核对）：**会话开关**（`bucket.py:29-43`）用
      **实例属性遮蔽类方法**（`__enter__` 装 `partial(_collective_func,name)`、`__exit__` 拆、
      嵌套检测即检测该副作用 `:32`），故 `_call_original:45-46` 必须**从类**取原方法；`_parse_args:48-53`
      用 `inspect.signature().bind` 统一 ABI。**临时分配** `_allocate:55-62` 是 **64B(RDMA) 对齐的
      bump 分配器**（越界抛 RuntimeError），`allocate:90-101` 支持单/批量形状；与**规划期**
      `BufferAllocator`（meta 占位 + 2MiB 对齐 + `torch.utils.swap_tensors` 原地物化，`allocator.py:10-35`）
      互补。**`_collective_func:103-181`** 三步：`num_shards` 表 `:114-118`、越界 src 例外
      `accept_unregistered_src = name=='all_gather' and num_rdma_ranks==1 :126`、逐 bucket 三分支
      `:141-160`（已注册 src/dst 反推整块 view、否则 bump 分配）、暂存
      `if not is_src_registered and not accept_unregistered_src: copy_tasks.append :162-163` +
      `torch._foreach_copy_ :174`、发射 `:177-181`。**结果回拷** `_wrap_handle:72-88`：先取旧
      `hook_after_wait` 再置空（因 `event.py:73` `assert ... is None`）、只对 `data_ptr` 不同的外部
      dst `_foreach_copy`、返回原 result。**生命周期守卫** `BucketBuffer::stream_control_epilogue`
      （`bucket.hpp:48-56`）对 src 默认 `record_stream(comm_stream)`（`EP_AVOID_RECORD_STREAM=1`
      走 `event.tensors_to_record`，避开 CUDA Graph 不兼容），**只有 all_gather 把 srcs 交给它**
      （`bucket.hpp:314`），`test_all_gather.py:74-80` 的 `del srcs` 再 `wait()` 即验收。组一致性
      `_check_group:64-69`（一次会话只服务一个通信组）。真机实测 `harness/exp_deepep_bucket_session.{sh,py,out.txt}`：
      **按文件路径加载真实 `bucket.py`/`allocator.py`/`base.py`/`event.py`**（`sys.modules` 注入合成
      `deep_ep._C`/`comm`/`envs` 绕过 `check_nccl_so`）+ 同名方法签名的 `FakeBucket` 承接
      `_call_original`，**31 PASS / 0 FAIL**——`_allocate` 100B×2→256B/越界抛错、batched 形状、
      嵌套会话/跨组被断言拒、已注册 src `alloc=0` 零暂存、未注册 src `_foreach_copy` 暂存、
      **纯 NVLink all-gather 越界 src 原样透传 vs 跨节点退回暂存**、`_wrap_handle` 回拷与同指针免拷、
      `BufferAllocator` meta→CUDA 物化 4MiB、四份源码行号逐一命中。需要 NCCL GIN / 对称窗口的
      `_C.BucketBuffer` kernel 侧**仅分析**。配套博客
      `content/posts/mega-moe-93-deepep-bucket-session`（索引页新增第 93 篇）。

### P84 增量（opt-in，补齐 DeepEP bucket 一直缺席的「验收测试即正确性规范」）

- [x] **P84-1** **DeepEP bucket 混合集合算子的验收测试即正确性规范**
      （`tests/bucket/test_mixed.py` 373 行；`tests/bucket/test_perf.py` 273 行；
      消费方 `deep_ep/buffers/bucket.py`、`deep_ep/comm/handle.py:80`、`csrc/buffers/bucket.hpp:151/219/284`）。
      第 54 篇（P54-1）拆了 bucket 三类算子 × 三物理域与 copy-engine、第 62 篇（P62-1）拆了
      `all_reduce`/`reduce_scatter` 三后端与信用环、第 83 篇（P83-1）拆了 `BucketSession`，但它们
      组织成**一套可执行规范**的两份旗舰测试一直只被顺带引用。**端到端 BucketBuffer 依赖 NCCL GIN /
      对称窗口 → 仅分析**；**测试侧纯 Python/张量，H100(sm90) 真机实测**。
      → `docs/94-deepep-bucket-mixed-test-oracle.md`（真实行号核对）：`test_mixed.py` 在**多轮**里随机
      编排**混合**算子（`OPS=all_reduce/reduce_scatter/all_gather/rs_ag`，`:24`；`rs_ag` 是
      `reduce_scatter`+`all_gather` 的组合，`:244-265`），轮换正交维度（`:164-214`：销毁模式 /
      会话每 3 轮 / 三条**各自独立**计算流 producer·launch·consumer / exact=bf16·all_gather·每 7 步 /
      scale∈{1,0,0.25,-0.5,2} / SM（all_gather 必须 0）/ bucket 数 / packed / dtype / 外部输入）。
      五个规范组件：① **能力门控** `supported:69-79`（单 rank 只有 all_gather `rdma*nvl==1`；bf16 只在
      纯 RDMA `rdma>1 and nvl==1`），perf `get_skip_reason:101-106` 是原因串版；② **尺寸量化**
      `make_sizes:82-88`（`quantum=lcm(group_size·32)`）；③ **用例枚举** `make_schedule:91-102`
      （`all_reduce` 恒 fp32）；④ **拓扑** `make_groups:44-66`（world + half/nvl 等分段 + 纯 RDMA rail）；
      ⑤ **NCCL 参考** `nccl_reference:126-141`（`rs_ag` 参考直接走 `all_reduce`，因
      `all_gather(reduce_scatter(x))==all_reduce(x)`）+ `make_inputs:105-123`（`GUARD_BYTES=128`、
      会话时新分配 / 非会话返回 guard 视图）+ 判据 `verify:144-154`（exact 时 `rtol=atol=0` 逐位 +
      guard 恒等）。`run_round:157-314` 三条隐蔽契约：三流 event 串 order（`:228-236`）、原地指针契约
      （`:277-282` 结果 `data_ptr`==arena payload 地址）、慢快照（`:283-284`「deferred CPU checks
      cannot hide missing dependencies」）+ 销毁后弱引用断言（`:304-311`）。`test_perf.py`：NCCL profile
      矩阵 `:36-48`（auto/cta/ring/tree/nvls…，`run_profiles:218-249` 总先跑 auto、profile 必须在
      import torch 前设）、`nccl_collective:76-84` coalescing、ragged `make_sizes:91-98`、
      `measure:51-73`（先 `all_reduce(MAX)` 取每样本**最慢 rank** 时长再中位）。H100 实测
      `harness/exp_deepep_bucket_test_oracle.{sh,py,out.txt}`（按路径 exec 两个真实测试模块、stub
      `deep_ep` 绕过 `check_nccl_so`）：**33 PASS / 0 FAIL**——`supported` 64/64 组、`make_sizes`
      3/3（全为 `quantum` 倍数）、真机 `dist.new_group` 拓扑 3/3（nvl=8/4/2）、`nccl_reference`
      4op×scale vs 独立求和 10/10 + `rs_ag` 不变量、`make_inputs`/`verify` 8/8（guard 抓错、exact
      1-ulp、容差放行）、perf 规范 6/6、真实 `measure` 1/1。**环境发现**：DeepEP@93eb6eb 用的
      `reduce_scatter_single`/`all_gather_single` 在本机 torch 2.12.0a0 不存在，用 `*_tensor` 兼容别名
      驱动（继第 86 篇 NCCL 版本后又一条「测试所需环境」）。配套博客
       `content/posts/mega-moe-94-deepep-bucket-mixed-test-oracle`（索引页新增第 94 篇）。

### P85 增量（opt-in，补齐 EP 通信的反向侧：dispatch / combine 的伴随语义与路由权重梯度）

- [x] **P85-1** **MoE dispatch / combine 的反向（伴随）语义与路由权重梯度**
      （`README.md:158-289` 训练/推理四个 helper；`deep_ep/buffers/ep.py:562-757/770-843`；
      `deep_ep/utils/refs.py:10-123/156-174/177-243`；`tests/ep/test_ep.py:514`；
      `impls/ep/combine_utils.cuh:58-119`、`impls/ep/combine_reduce_epilogue.cuh:22-139`）。
      第 4 篇推了 MoE 反向的**数学**、第 12/13/14 篇拆了反向 grouped GEMM、第 43 篇拆了 EPLB
      权重预取/梯度归约、第 7/51/58/61/71 篇拆了 dispatch/combine 设备 kernel，但把前向通信接回反向
      的那层——「dispatch 的反向是 combine、combine 的反向是 dispatch 为什么成立、`w` 的梯度怎么走」
      ——一直只在第 7 篇 §8 与 README 里一句话带过。**纯线性代数 + host/Python 语义，与 SM100/NCCL
      设备 API 无关；端到端 `EPBuffer` 反向 helper 依赖 NCCL ≥2.31 → 仅分析**。→
      `docs/95-ep-backward-adjoint.md`（真实行号核对）：把 dispatch 看成**复制/置换**算子
      `D:ℝ^{T×H}→ℝ^{N×H}`、combine 的 token 求和看成 `Dᵀ`，**两原语互为转置** → 三行「反向用什么
      原语」表：`grad_x=Dᵀ(grad_recv_x)`（combine 求和）、`grad_w=Dᵀ(grad_recv_w)`（combine 权重
      reduce）、`grad_y=Cᵀ(grad_combined_x)`（dispatch 复制）；逐字对上 `README.md:204-220`
      （`dispatch_backward` 调 `combine(grad_recv_x, topk_weights=grad_recv_w)`）与 `:242-258`
      （`combine_backward` 调 `dispatch(..., do_expand=True, do_zero_padding=True)`）。**精确定义**：
      `refs.dispatch:67` 非 expand 按**目的 rank**复制（一个 token 对一个目的 rank 一行、带**整行** `w`
      `:75`）、`:85-90` 按源全局索引升序、`:50-53/:112-117` 契约；`refs.combine:177-243` 是分组**求和**
      （`combine_reduce:58-119` 设备侧同语义），README `:279` 明说权重乘法在 combine **之前**由专家
      计算融合。**一个只在复制语义下成立的巧妙设计**：按目的 rank 复制使一个 token 的 `w` 被复制 `d`
      份，朴素求和得 `d·w`，故 `combined_topk_weights` 必须**选择而非求和**（
      `combine_reduce_epilogue.cuh:128-139` 的 `master_lane_idx`+`get_topk_weights_ptr`），才能满足
      `test_ep.py:514` 的 `== topk_weights`。**训练图路由权重梯度**：`z=Σ_k w·y` →
      `∂L/∂y=w·g`（复制=dispatch）、`∂L/∂w=<y,g>`（沿 k 归约=combine 的权重 reduce）；`EPHandle`
      （`ep.py:103-111/613-615`、README `:281`）存路由账本，反向只换数据。**H100 真机 ALL PASS**
      （`harness/exp_ep_backward_adjoint.{sh,py,out.txt}`，refs.py sha256 `c5f551b4…`）：Part A 单进程
      纯张量四条恒等式 `<D(x),g>==<x,Dᵀ(g)>`（0.765600/0.765601）、`<G(w),gw>==<w,Gᵀ(gw)>`
      （−0.466352/−0.466353）、`<C(y),v>==<y,D(v)>`（−5.027745/−5.027744）、朴素求和 combine == 真实
      `refs.combine`（逐位）、权重梯度对拍 autograd（fp64 ≤1e-12）、`dispatch_backward==combine`；
      Part B **8×H100 真机 NCCL 用真实 `refs.dispatch`**（`all_to_all_single`）跨 rank 验伴随
      `<D(x),g>==<x,Dᵀ(g)>`（100.3833/100.3833）、`<G(w),gw>==<w,Gᵀ(gw)>`（−18.4716/−18.4716）、
      复制语义 `Σrecv_w==degree·w`（平均 degree 3.328）。配套博客
       `content/posts/mega-moe-95-ep-backward-adjoint`（索引页新增第 95 篇）。

### P86 增量（opt-in，补齐 P79 只对照未拆的两个 mega 家族 JIT 启动包装层）

- [x] **P86-1** **FP8/FP4 Mega MoE + Mega Gate 的 JIT 启动包装层**
      （`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` 224 行 /
      `sm100_bf16_mega_gate.hpp` 92 行；`csrc/jit_kernels/heuristics/mega_moe.hpp` /
      `mega_gate.hpp`；`deep_gemm/include/deep_gemm/layout/mega_gate.cuh`；
      `csrc/apis/mega_gate.hpp`；`sm100_fp8_fp4_mega_moe.cuh` / `sm100_bf16_mega_gate.cuh`）。
      第 89 篇（P79-1）拆了 BF16 Mega MoE 与 mHC 的包装层、只把 FP8 Mega MoE 当「对照」数了描述符
      （且计数有误），Mega Gate 的包装层从未被单独读过。本篇补齐，凑齐
      `sm100_{fp8_fp4_mega_moe, bf16_mega_moe, bf16_mega_gate, mega_mhc}.hpp` 四个包装层。
      **主体 kernel 全 SM100 专属，仅分析**；包装层纯 host，H100 真机实测。
      → `docs/96-mega-moe-gate-jit-wrappers.md`（真实行号核对）。**FP8 Mega MoE 要点**：
      host 派生 `:37-41`（多取 `num_sf_ring_tokens`）；`sf_smem_outer_dim = block_k/(kGranK*4)` `:51-52`；
      **18 个描述符**（9 routed + 9 shared，`:53-139`，更正第 89 篇的「20」），9 个 SF 全
      `Major::MN` + 无 swizzle（`make_tma_sf_desc` 断言 `:393/396`，`get_tma_aligned_size` `:398`）；
      共享专家 `E_s` 折叠（L1 N / L2 K 乘 `S`，`:94-139`，对第 44 篇 `SHARED_L1_SHAPE_N = L1_SHAPE_N·E_s`）
      且共享权重走 2D 描述符；模板 25 实参 `:174-189`（比 BF16 多 `STORE_BLOCK_M_L2`/`SF_BLOCK_M/N`/
      `kNumSFRingTokens`/`to_string(weight_dtype)`）；launch `:192-221`（grid=num_sms、block=512、cluster=2）。
      **Mega Gate 要点**：8 个模板 bool 全由指针/选项空值推出 `:63-65`；两描述符用 `make_tma_a/b_desc`
      （K-major、BLOCK_K=64）`:36-41`；`RoutingArgs` POD ABI（`sizeof=72`，offset 48 交界）`layout:31-43`；
      scratch `:30-34` + 按 stream 缓存 score barrier（capture 前 warm-up，`csrc/apis/mega_gate.hpp:13-26`）；
      配置搜索 `heuristics/mega_gate.hpp:114-166`（三维枚举 + deterministic 禁 split-K `:110-111`）；
      launch `:69-89`（grid 可不足整卡、cluster=num_mma_ctas）。H100 实测
      `src/mega_moe_gate_jit_wrapper.cu` + `harness/exp_mega_moe_gate_jit_wrapper.{sh,out.txt}`：
      **143 PASS / 0 FAIL**——4 形状 × 2 MmaKind 的 10 条配置不变量（DeepSeek-V3 规模：FP8
      `block_m=240/block_k=128/store_l1=24,store_l2=8/sf=256x128`、BF16 `block_k=64/store=40,40/sf=0`）；
      `num_shared==0` 时 4 个共享描述符与 routed **128 字节逐字节相等**；独立 `cuTensorMapEncodeTiled`
      编码器对 `l1_acts`/`shared_l2_acts`/`l1_acts_sf` **逐字节互证**；模板占位符计数 25/20/15；
      Gate 常量 + `RoutingArgs` offset + scratch 公式 + `caches_logical_count` 真值表；4 形状 ×
      deterministic 开关 config（`T=64` 关确定性 `splitK=8`、开确定性 `1`）。配套博客
      `content/posts/mega-moe-96-mega-moe-gate-jit-wrappers`（索引页新增第 96 篇）。

### P87 增量（opt-in，补齐第 16 篇只写了 4 条 bullet 的 SM100 hc_prenorm + mega 家族最后一个 JIT 包装层）

- [x] **P87-1** **SM100 HyperConnection prenorm GEMM 的 JIT 启动包装层与三段流水**
      （`csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp`、
      `deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh`；
      `csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp`、`csrc/apis/hyperconnection.hpp`）。
      第 16 篇（P7-2）拆了 `tf32_hc_prenorm_gemm` 的独立算子形态，但 **SM100 分支只写了 4 条 bullet**
      （标注「仅分析」）；第 79/86 篇补齐了另外四个 mega 家族（BF16/FP8 Mega MoE、Mega Gate、mHC）
      的 JIT 启动包装层，唯独 `sm100_tf32_hc_prenorm_gemm.hpp`（113 行）缺席。本篇补齐这最后一块。
      **host 包装层**（纯 host，H100 真机）：`block_m=block_k=64`、两个 warpgroup 各 128 线程、
      `block_n=align(n,16)` 且 **n 上限从 SM90 的 32 放宽到 128**（`:28`）、`k%64==0`；
      num_stages 预算循环（`:53-67`）与 SM90（`sm90_...hpp:55-79`）三处差异——**barrier 家族从
      2 组变 4 组 + 1 个 tmem_full**（`(num_stages*4+1)*8`）、多 4B TMEM 指针、SM90 末尾把 smem
      覆盖成满 227KB 而 SM100 保留算出的值；D 描述符在 split 时走 **3D `[num_splits,m,n]`**
      （`:46-50`）；grid=`num_splits*ceil(m/64)`、block=256、10 个编译期模板常量。
      **SM100 kernel**（仅分析）：与第 10/35 篇同源的「TMA→cast→MMA」三段流水，多出的
      **cast-and-reduce warpgroup**（`:268-342`）承担「A bf16→fp32 写 TMEM」——因 UMMA 的 TS 形态
      （`SM100_MMA_TF32_TS` `:169-170`）没有寄存器 A 操作数；并在转换同一拍用 `__ffma2_rn(v,v,sum)`
      （`:316`）累加平方和（第 16 篇「省一遍 A 的读」的落点）。四组 mbarrier 的 init 计数
      （`full_cast`=128 `:112`、其余=1 `:111/113/114/116`）、`UMMA_K=32/sizeof(float)=8`、
      逐 K `fma`（`:203`）首条清零、`umma_arrive` 一次 commit 通知 empty/empty_cast（`:207-208`）
      与末尾 `tmem_full`（`:212`）、TMEM 列预算 `BLOCK_K*kNumCastStages+BLOCK_N`（`:75`，
      `get_num_aligned_tmem_cols` `utils.cuh:42-50`）；**split-K 分解**（`:126-134`：余数分给前
      `kRemainKBlocks` 个 split、部分 D 落独立切片、`m_offset=shape_m*k_split_idx`、host 外部求和）；
      **手写银行组 swizzle** `get_swizzled_smem_offset`（`:18-34`，STSM/LDSM 共用）。
      H100 实测 `src/hc_prenorm_sm100.cu` + `harness/exp_hc_prenorm_sm100.{sh,out.txt}`
      （nvcc 直接 include 真实头文件 + torch，**21 PASS / 0 FAIL**；文件 sha256
      `sm100_...hpp bb44f58e…`/`sm90_...hpp 9570a7ce…`/`sm100_...cuh 37d28d51…`）：
      ① host 预算表——`n=8/16` 12 stage、`n=24/32` 12 stage、`n=64` SM90 拒/SM100 9 stage、
      `n=128` SM90 拒/SM100 5 stage，SM100 恰多 204B（`(2*12+1)*8+4`）；② split-K 恰好覆盖
      （含余数 1/2）+ 部分和 == 完整 GEMM（`max|Δ|=1.144e-05`）+ 3D scratch 证实；
      ③ **`get_swizzled_smem_offset<128,16>` 与 `cute::Swizzle<3,4,3>` 逐点 ALL MATCH**、
      在 32×128B atom 上是 **16B chunk 双射**、mode-64 非 shortcut 分支无碰撞；
      ④ cast-reduce 的 `warp_reduce_sum<4>` + `m_idx` 映射 64 行全对、`fma2` 平方 fp32 vs fp64
      5.901e-08；⑤ SM100 wrapper 的 2D/3D TMA 描述符用真 torch 张量（`m=4096,k=7168`、`n∈{24,128}`）
      成功构造且 2D/3D 指纹不同。配套博客 `content/posts/mega-moe-97-sm100-hc-prenorm-gemm`
      （索引页新增第 97 篇）。

### P88 增量（opt-in，补齐 DeepEP 的「任何设备代码跑起来之前」的编译配置面）

- [x] **P88-1** **DeepEP 的编译期配置面与 DeepJIT 接入**（`deep_ep/include/deep_ep/common/compiled.cuh`、
      `csrc/runtime/jit.hpp`、`csrc/utils/system.hpp`；`setup.py`、`csrc/python_api.cpp`）。
      第 33 篇拆了 DeepGEMM 的 DeepJIT 接入、第 47 篇拆了 DeepJIT post-hook、第 86 篇拆了 DeepEP 的
      Python 引导层（`__init__.py:44-51` 的 `init_jit` **调用点**），但同一个 `init_jit` 的 C++ 实现
      （`csrc/runtime/jit.hpp`）与它引出的 `compiled.cuh` 一直没被系统读过——本篇补齐。
      **两个编译面**：host `_C`（`setup.py:96-98` cxx `-DEP_NUM_TOPK_IDX_BITS`）与 device JIT
      （`jit.hpp:38` 同一宏进每个 kernel 的 nvcc），必须一致；`compiled.cuh:45-47` 缺省 64。
      **`compiled.cuh` 五件事**：① `#undef` torch 的五个 `__CUDA_NO_{HALF,BFLOAT16}*`（`:10-25`）；
      ② 256-bit LD/ST 兼容（`:33-43`，CUDART≥13 用 `longlong4_32a`）；③ `topk_idx_t=int_with_bits<
      EP_NUM_TOPK_IDX_BITS>`（`:45-57`，只特化 8/16/32/64）；④ `sf_pack_t` 4B union +
      `kNumAlignedSFPacks=16/sizeof=4`（`:59-64`，SF token 维 4 对齐的来历 `ep.hpp:693`）；
      ⑤ 32/64/2MiB 三档对齐 + 两条整除 `static_assert` + 通信上限（`:66-82`，是 `layout/ep/workspace.cuh:15-20`
      的数组维度、`bucket.hpp:86` 的 QP 划分、`context.cpp:95/137` 的断言来源）。**`init_jit`**：
      `Config` 五参（`library_root`/`"EP"`/空 `extra_signature`/`{include}`/`{"deep_ep/"}`）+
      8 条 nvcc flag（2 诊断 + `-I<nccl>` + `-DEP_NUM_TOPK_IDX_BITS` + 4 条 GIN GDAKI）+ 条件
      `EP_GIN_GDAKI_DEBUG`；`get_env`（`system.hpp:12-29`，只支持 string/int，非支持类型抛异常）
      + `LazyInit` + `register_apis`。**缓存键**：`deep_jit/runtime/runtime.hpp:92-97` 键 =
      `extra_signature + nvcc 版本 + options.update_hash(str::join(get_flags())) + post_hook +
      parser.parse_into_hash(source)`；`-I<nccl>` 在 `nvcc_flags` 里 → 进第 ③ 段（NCCL include 路径
      进键），DeepEP 自身头按内容进第 ⑤ 段（`include_prefixes={"deep_ep/"}`，路径无关）——与 DeepGEMM
      把 CUTLASS 版本编进 `extra_signature` 对照。**完全 host / 编译期、架构无关**，H100(sm90) 真机
      `src/deepep_compiled_abi.cu` + `harness/deepep_jit_config_probe.cpp` +
      `harness/exp_deepep_jit_config.{sh,out.txt}` **48 PASS / 0 FAIL**：`compiled.cuh` 默认 64 /
      覆盖 32 两个编译单元各 15 PASS、`longlong4_t` 32B/32-align（`CUDART_VERSION=13020`）、
      `get_env` 四条正常路径（含 `"123abc"`→123）+ `get_env<bool>` 抛 `EPException`（真实行号
      `system.hpp:27`）+ `Config` 五参与三条负样本（相对根 / 空前缀 / 保留字 `"DJ"`）+ flag 哈希
      （`nccl-A 7b47f9c3…` vs `nccl-B 1a00ac53…` vs `debug 63643454…`）+ Python 解析 `jit.hpp`
      9 条 flag 字面量逐字一致。配套博客 `content/posts/mega-moe-98-deepep-jit-config`
      （索引页新增第 98 篇）。

### P89 增量（opt-in，补齐「前向」一直缺席的端到端数值复刻与误差分解）

- [x] **P89-1** **Mega MoE 前向流水线在 H100 上的数值复刻与误差分解**（
      `harness/exp_mega_moe_forward_emul.{py,sh,out.txt}`；
      参考 `impls/sm100_fp8_fp4_mega_moe.cuh`、`csrc/apis/gemm.hpp`、`csrc/apis/mega_moe.hpp`；
      对照第 41 篇 `tests/test_mega_moe.py`、第 4/12/13/14 篇反向实验）。第 3/9/10/11 篇拆了
      SM100 主 kernel、第 41 篇拆了端到端验收测试、第 4/12/13/14 篇把**反向**四个 grouped GEMM
      真的接进图里跑了，但**前向**一直缺「用 H100 可跑算子逐段复刻 + 误差分解」。**主 kernel
      SM100 专属、仅分析**；前向的数学/量化语义架构无关，用 SM90 `m_grouped_fp8_gemm_nt_contiguous`
      （1D2D）+ `m_grouped_bf16_gemm_nt_contiguous` 复刻。**语义钉死**：linear1
      （`kernel:161-164`）→ SwiGLU（gate/up 先 round bf16 `:1117-1118`、clamp、`silu*up`、**乘
      top-k 权重** `:1135`）→ L1 输出 UE8M0 per-token 量化（`:1160-1192`，MN-major SF `:1205-1213`）
      → linear2 → combine（`:1412-1519`，**不再乘权重**，`ptx::accumulate` fp32 求和 `:1477-1490`）。
      **H100 实测**：① 端到端 fp8-kernel vs fp64 在 4 组形状（含 DeepSeek 量级）上 rel
      **6.28–6.34e-2**、cos ≥0.9980（bf16-torch 5.0e-3 / fp32-torch 5.1e-4）；② 分层——kernel
      vs 反量化参考端到端仅 **3.2e-4**、单层 **1.47e-3**（bf16 输出地板）→ 6.3e-2 的 99% 来自量化；
      ③ 消融 仅激活 5.70e-2 / 仅权重 4.58e-2 / 两者 6.28e-2 / 不量化 1.55e-3，**两侧同量级**；
      ④ 吞吐 L1/L2 **312/303 TFLOPS**（843/434us）；⑤ 记吞吐陷阱（每调用 empty+sync 把 312 误测
      成 54）。配套博客 `content/posts/mega-moe-99-mega-moe-forward-emulation`（索引页新增第 99 篇）。

### P90 增量（opt-in，补齐 EPLB 两个原语一直缺席的「验收测试即正确性规范」）

- [x] **P90-1** **DeepEP EPLB 冗余权重的验收测试即正确性规范**
      （`tests/ep/test_prefetch_weights.py` 250 行；`tests/ep/test_reduce_grads.py` 256 行；
      `deep_ep/buffers/ep.py:846-905`；`deep_ep/utils/envs.py:193-234`；
      `deep_ep/include/deep_ep/common/compiled.cuh:67-71`；消费方第 53 篇 P43-1）。
      第 53 篇（P43-1）把两个 EPLB 原语（前向 `lb_prefetch_weights` / 反向 `lb_reduce_grads`）
      从 host 编排到负载均衡切块讲透了，但只在 §5/§6 引用过上游两份验收测试来佐证 reduce-add 的
      顺序无关性，从没把这两份「什么叫对」的唯一规范单独拆过——它们正是第 53 篇所有判据的出处。
      **设备路径走 NCCL 设备 API（`ncclDevComm`/`ncclGin`），本机 NCCL 2.29.7 无该 API
      （同「阻塞」），端到端仅分析**；但判据几乎全在纯 Python/纯张量里，H100(sm90) 可跑真机。
      → `docs/100-deepep-eplb-test-oracle.md`（真实行号核对）：**前向 blob** 是**唯一 key 的
      确定性 ramp**（`make_blob:45-53`，`key=e*NUM_PAYLOADS+p`、`(arange+key)%255`、纯算术非 RNG
      以保跨 GPU 逐位相同）→ 错专家/错载荷/错偏移**每个字节**都暴露；`SENTINEL=0xa5`（`:16`）让
      未分配槽被写=strayed writes（`check:95-108` 两条分支 `torch.equal` 逐字节）；`PAYLOAD_SWEEP`
      （`:27-38`）10 组只打 chunk 边界（12 KiB 容量整数倍/两侧短尾/少于 warp 数）。**映射表**
      `make_redundancy_mapping`（prefetch `:56-92` / reduce `:54-95`）保证**一个 rank 绝不为本地
      专家持有冗余副本**（own-expert 排除 `:93-94`）——前向定义「往哪推」、反向变成「绝不从自己
      那行 pull」→ seed 不被重复计（`expected_expert_grads:98-114` 的 `:107` continue + `:111` owner
      判定）；`ring`（`:75-80`，r 只借 r+1）是基准 pattern，分离「流量形态」与「kernel」。**反向
      blob** 是**小整数精确 fp32**（`make_pattern:23-40`，period 4095、值 `(0..4094)-2047`）——因
      `cp.reduce.async.bulk` 是元素级原子加、落地顺序不固定，小整数使期望值**顺序无关**、判据用
      `torch.equal` 免容差；两族 key（expert≥0 `:46` / redundant<0 `:51`）永不碰撞。**判据维度**：
      哨兵分支、`with_sf=False` 未请求载荷槽必须保哨兵（host 契约 `ep.py:866-874`）、反向源池
      一个元素都不许被改（`:198-204`，TMA 源/目的弄反的落点）。H100 实测
      `harness/exp_deepep_eplb_test_oracle.{py,sh,out.txt}`（ast 抽取真实函数原样运行 + 独立复刻
      逐位对拍）**37 PASS / 0 FAIL**：`make_blob`/`make_pattern` 逐字复刻、错专家/错载荷失配=n、
      映射表 own-expert/ratio/ring/seed 确定性、`check` 三负样本被断言抓到、5 pattern 三种加序
      `torch.equal`、期望值与独立复刻逐位相等；全量仿真 10 载荷 × 2 轮 × 8 rank push **920 槽
      0 不符**、11 hidden × 2 轮 × 8 rank pull **176 组 0 不符**；带宽关键路径 prefetch 与 reduce
      **完全一致**（本机 NVLink 上限 430.3 GB/s、`random` `[2,5,6,8,9,10,4,3]` → critical path
       671 MB、下限 ~1560 us）。`tests/*.py` sha256 已认证。配套博客
       `content/posts/mega-moe-100-deepep-eplb-test-oracle`（索引页新增第 100 篇）。

### P91 增量（opt-in，补齐 DeepEP 测试套件里最后一份未覆盖的验收测试）

- [x] **P91-1** **DeepEP 屏障的验收基准与「通信 kernel 精确测时」方法论**
      （`tests/comm/test_barrier.py` 66 行；`deep_ep/utils/testing.py:113-221`；
      `deep_ep/comm/barrier.py:6-11`；`deep_ep/comm/handle.py:80-95`；
      `deep_ep/buffers/ep.py:237-240`；`csrc/comm/handle.hpp:16-22`、`csrc/comm/barrier.hpp:9-42`、
      `csrc/kernels/comm/barrier.hpp:17-54`、`csrc/kernels/comm/context.cpp:52-63`、
      `deep_ep/include/deep_ep/impls/comm/barrier.cuh:11-41`、
      `deep_ep/include/deep_ep/layout/common/barrier.cuh:11-23`）。
      第 53 篇（P53-1）拆了屏障**设备协议**、第 86 篇（P76-1）拆了**host 编排**，但
      **「什么叫这个屏障是对的/够快」的唯一规范** `tests/comm/test_barrier.py`（DeepEP 测试套件里
      `tests/{ep,bucket,pp,engram,utils,buffer,comm}` 最后一份未被单独读过的文件）一直缺席。
      **它不看数值、只测单次屏障耗时**。**设备 kernel（`barrier_impl`/`gpu_barrier`）走 NCCL GIN
      设备 API，本机 NCCL 2.29.7 无该 API（同「阻塞」）→ 端到端仅分析**；域语义/host 启动/Python
      转发/`bench_kineto(barrier_comm_profiling=...)` 测时路径**全在纯 host/Python + torch profiler**
      里，8×H100 真机复刻。
      → `docs/101-deepep-barrier-test-oracle.md`（真实行号核对）：`test_barrier:10-26`
      （`loop_barrier` 连打 1000 次 `:19-21`、`bench_kineto(..., 'barrier', barrier_comm_profiling=True,
      barrier=buffer.barrier)` `:23`——**把 profiler 的削峰屏障换成被测对象自己**，因 ring 实现的
      `all_reduce` 有额外 RTT `testing.py:169`）；`test_loop:30-52`（两条**域恒等式**断言
      `rdma*nvl==num_ranks` `:35` / `scaleout*scaleup==num_ranks` `:36`；`--do-pressure-test` 反复
      建/毁 `EPBuffer(explicitly_destroy=True)` `:38-51`；argparse 四参 `:55-66`）。**域语义**
      `context.cpp:52-63`：物理域来自 NCCL 拓扑（`ncclTeamWorld`/`ncclTeamLsa`），逻辑域由
      `allow_hybrid_mode` 决定 —— `True`→`(rdma, nvl)` 两级、`False`→`(1, num_ranks)` 单级。
      **host 链路**：`EPBuffer.barrier = comm.barrier`（`ep.py:237`）→ Python `barrier(context_idx)`
      选通信域并**重排参数**转发 `_C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)`
      （`barrier.py:10-11`）→ pybind lambda `csrc/comm/handle.hpp:16-22`（`barrier_signals!=nullptr`
      断言、`flush_stores` 硬编 `true`、当前流）→ `barrier()` 的 CPU-sync/comm-stream **双向对齐**
      （`csrc/comm/barrier.hpp:17-41`）→ `launch_barrier` 的 `kNumThreads=512` `:26`、
      `num_sms=(not sequential and num_scaleout_ranks>1)?2:1` `:30`（只有**并行混合且跨节点**才 2 SM）、
      8 模板实参 `.cooperative=true` `:36-49`。**设备 kernel**：`kSequential` 先 scaleout（仅
      `kNumScaleoutRanks>1`）再 scaleup 且**要 flush scaleout 的 RDMA 请求** `:23-34`；并行时两 SM 各跑
      `gpu_barrier`；`BarrierSignals` 的 `nvl_barrier_signals[2]` 双相位槽 `layout:11-23`（第 53 篇协议）。
      **测时方法论发现**：`bench_kineto` 的 `barrier_comm_profiling` 每轮插 `torch.cuda._sleep(2e7)`
      （实测墙钟 **13.47ms**）+ 屏障（`testing.py:154/166-173`）、`EP_DISABLE_BARRIER_PROFILING=1`
      就地置 0 `:154`。H100 实测 `harness/exp_deepep_barrier_profiling.{py,sh,out.txt}`（真实源码
      路径 exec + stub `_C`/`envs`、真实 DeepGEMM `sm90_fp8_gemm_1d2d` M=N=K=4096、8×H100 NCCL）：
      **Part B 4/4、C 6/6、D 5/5、E 4/4 PASS**；**Part F**：`no-barrier`/`disabled(env)`（无 sleep）
      的 profiler 数字在多次测量里呈**双峰**（快档 ~98–101us / 慢档 ~417–419us，≈**4×**），而
      `barrier=all_reduce`/`barrier=custom` 每轮都落在快档、spread **<0.3%**；event 计时参考
      ~104.7/106.0us。**慢档是真实 GPU 行为**：直接量持续负载（背靠背、无间隔）同一 GEMM
      **隔离 0.106 ms/iter vs 纯 GEMM 0.34 / flush+GEMM 0.66 ms/iter（3–6×）**，坐实 **H100 持续满载
      降频**（呼应第 15 篇「单次 vs 持续 ≈2×」、第 36 篇「降载不抬频」），13.5ms 空闲让频率恢复
      ——`barrier_comm_profiling` 的真实作用比 docstring「削 CPU 发射抖动」多一层「避降频」。
      配套博客 `content/posts/mega-moe-101-deepep-barrier-test-oracle`（索引页新增第 101 篇）。

### P92 增量（opt-in，补齐 Mega Gate / Mega mHC 从没被单独读过的两个分段屏障调度器）

- [x] **P92-1** **Mega Gate / Mega mHC 的跨 CTA 分段屏障调度器与工作区布局**
      （`deep_gemm/include/deep_gemm/scheduler/mega_gate.cuh`（`ScoreBarrier`）/
      `scheduler/mega_mhc.cuh`（`SplitBarrier`/`Norm`/`Mix`）；
      `layout/mega_gate.cuh:22-24/78-108`；`layout/mega_mhc.cuh:37-41/80-120`；
      `ptx/utils.cuh:23-27`（`%gridid`）；`ptx/ld_st.cuh:308/340/360`；
      调用点 `impls/sm100_bf16_mega_gate.cuh:55-56/75/268-302`、
      `impls/sm100_mega_mhc.cuh:51/173-178/323/379/583/653`、
      `epilogue/sm100_mega_mhc.cuh:17-24/292-306`；host `csrc/apis/{mega_gate,mega_mhc}.hpp:13-38`、
      `csrc/jit_kernels/impls/{sm100_bf16_mega_gate,sm100_mega_mhc}.hpp:23/84 与 29/146`；
      `heuristics/mega_gate.hpp:110-111/126`）。
      第 5 篇（Mega Gate）与第 5 篇下篇（Mega mHC）只在注释里提过「split-K 时用 `ScoreBarrier` /
      `SplitBarrier` 跨 CTA 汇聚（`scheduler:13-39/49`）」，第 87 篇拆的 `grid_sync`/`nvlink_barrier`
      是**跨 rank** 的同步原语——**同一 kernel 内按 token-block / m-block 分段**的这两个文件（共 96 行）
      从没被单独读过。**除一行 `red.async` 外全架构无关，H100(sm90) 直接 include 真实头文件跑真实设备代码**。
      → `docs/102-mega-gate-mhc-split-barrier.md`（真实行号核对）：**状态编码**「`状态 = (grid 标签)*stride
      + (到达数)`」——gate `(g+1)*kNumMaxLogicalCtas(64)+n`、mHC `(g+1)*kNumSplitBarriers(2)*kNumMaxSplits(64)+n`；
      **免复位**靠 `%gridid`（`ptx/utils.cuh:23-27`）每 launch 自增给出全新基数（对照第 87 篇 `grid_sync`
      的 2³¹ 免复位），host `torch.zeros` 只在每 stream 首次分配（`apis:13-25/26-38` 的 capture 断言强制
      CUDA Graph 前 warm-up）。**三个「为什么」**：① `wait_init` 的无符号距离（`while (ld_acq - base >= C)`）
      **同时**拒绝「上一轮残留」（距离回绕成极大）与「本轮未 arm」，却接受本轮任意部分到达（`mega_gate.cuh:27`/
      `mega_mhc.cuh:32` 同一注释）；② `arrive` 的 `red.async.release.gpu.global.add.u64` 是**唯一 SM100
      专属的一行**（ptxas `sm_90a` 报 `requires .target sm_100 or higher`；真机用同语义同步 `red` 替身）；
      ③ 状态位**进位隔离**——stride 必须严格大于最大到达数（gate `:15` 断言；mHC 用 128 > `kNumSplits`）。
      另拆 **结构对照**（槽索引 / 128B line 防 false sharing / `SplitBarrier::get_ptr:15-18` 的
      `kBarrierIndex*kNumMaxMBlocks + m_block_idx` 让 Norm/Mix 互不重叠）、两个 `Workspace` 几何
      （gate 屏障数组 8 MiB / mHC 4 MiB；score scratch 与三段 partial 偏移）、**四个角色调用链**
      （gate：worker group 内 rank0 init、其余 wait_init、每片 arrive、队首 wait；mHC：owner SM init、
      全体任务 wait_init、每 split arrive、epilogue wait）与 deterministic 差异（gate 关 split-K 使
      `kNumLogicalCtas==1` 直接跳过屏障 `:271/298/301`；mHC 只是把 splits 钉成 16）。H100 实测
      `harness/exp_mega_gate_mhc_split_barrier.{sh,out.txt}` + `src/mega_gate_mhc_split_barrier.cu`
      （真实头文件 sha256 `mega_gate.cuh 4aef186a…`/`mega_mhc.cuh bf774cfa…`）**31 PASS / 0 FAIL**：
      `ScoreBarrier` C∈{1,4,8} 连续两次 launch 不重置仍 mismatch=0、终止态到达数字段==C；`wait_init`
      三种预置状态（陈旧/arm/部分到达）逐一钉死（init_seen_before_return=1/0/0）；`SplitBarrier`
      Norm/Mix × splits∈{4,16} 四组免复位通过、`get_ptr` 偏移 `off0=mb*128`/`off1=(16384+mb)*128`；
      `Workspace` host 算术 7/7；**`arrive` 在 `sm_90a` 编不过、`sm_100a` 编得过**（S 组硬证据）。
      配套博客 `content/posts/mega-moe-102-mega-gate-mhc-split-barrier`（索引页新增第 102 篇）。

### P93 增量（opt-in，补齐第 36 篇只在一条 bullet 里带过的 SM100 `kSwapAB`（dgrad）分支）

- [x] **P93-1** **SM100 BF16 GEMM 的 `kSwapAB`（dgrad）分支：A/B 角色互换、转置写回与选核启发式**
      （`csrc/jit_kernels/heuristics/sm100.hpp:30-46/81-132/165-215/254-270`、
      `csrc/jit_kernels/heuristics/sm90.hpp:125`、`csrc/jit_kernels/heuristics/{runtime,common,config}.hpp`、
      `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:22-66/126-...`、
      `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:66-88/177/237-240/286/310-315/382-412`、
      `deep_gemm/include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh`（259 行）、
      `deep_gemm/include/deep_gemm/scheduler/gemm.cuh:177-184`）。
      第 19 篇拆了 SM90 BF16 主 kernel、第 36 篇（P36-1）拆了 SM100 BF16 主 kernel 骨架，但第 36 篇
      只在一条 bullet 里写「dgrad 的 m-grouped 强制 `swap_ab=true`」，从未讲清**它改了什么、为什么改、
      host 怎么选、转置写回怎么落地**；而它恰是 SM100 上 MoE **dgrad**（及所有 m-grouped 稠密 GEMM）
      唯一能走的形态。**为什么需要它**：SM100 UMMA 的 M 维（layout A/D 行数）被钉死在 64/128/256
      （`sm100_bf16_gemm.cuh:252-254`），而 m-grouped 的 M 是「逐组不等、按 16 对齐」的短维、N（hidden）
      才是固定大维；swap 把 A、B 操作数在 MMA 里**换轴**，让 `UMMA_N=BLOCK_M`（`:68`，且可逐块
      `update_instr_desc_with_umma_n` `:286`）去承接可变短 M、`UMMA_M=128`（`:67`）留给大 N。
      → `docs/103-sm100-swap-ab-dgrad.md`（真实行号核对）：**host 启发式**——m-grouped（含
      `MGroupedMasked`/`MGroupedContiguousWithPsumLayout`）走 `heuristics/sm100.hpp:34-46` 返回**唯一
      候选** `swap_ab=1, block_n=128, cluster_m=1`，`block_m=mk_alignment_for_contiguous_layout()`
      （默认 128、SM100 理论值 256，`runtime.hpp:11/59-67`）、`cluster_n=ceil_div(n,128)%2==0 and
      num_sms%2==0 ? 2 : 1`（`:42`）；对照非 m-grouped 完整枚举 `swap_ab∈{0,1}` 并施加
      `block_n==128`（`:121-123`）、`cluster_m==1`（`:81`）、TMEM 列预算（`:130-132`）过滤。
      **编译期常量**：`STORE_BLOCK_M=16`（`UMMA_STEP_N` 原子 `:85`）、`STORE_BLOCK_N=BLOCK_N=128`
      （`:86`，epilogue 用整 128 行 TMEM，须一个 warpgroup、`:87` `kNumUMMAStoreThreads=128`）、
      `smem_cd=16*128*sizeof(cd)*num_tma_store_stages`（`:214-215`）。**设备端**：TMA `load_block_m`
      逐块取 `get_aligned_effective_m_in_block / kNumMulticast`（`:177`）、`instr_desc` 的 major 顺序对调
      （`:237-240`）、MMA 发成 `fma(b_desc, a_desc, …)`（`:310-312`）。**转置写回**
      `epilogue/sm100_store_cd_swap_ab.cuh`：按 16 行 M 一块块搬（`:60`）、一次读 8 行 TMEM（`:69`）、
      转置 STSM（FP8 用 `SM100_U8x8_STSM_T` `:156-165`、BF16 用 `SM90_U32x4_STSM_T` `:233-238`）、
      TMA store 坐标是 `(n_idx,m_idx)`（`:249-256`）；swap 下 FP8 输出必须正好 `16×128` + 一个 warpgroup
      （`:42-48`），随机舍入在「间隔 4 条 lane」间用 `__shfl_xor_sync` 逐级归约哈希（`:206-224`）。
      **主 kernel 是 SM100 专属、仅分析**；但 host 半边是纯 C++，H100(sm90) 直接 include 真实头文件真机
      `harness/exp_sm100_swap_ab_dgrad.{sh,out.txt}` + `src/sm100_swap_ab_dgrad.cu` **56 PASS / 0 FAIL**：
      m-grouped 唯一 swap layout（3 形状，含 `n=7168`→cluster_n=2 / `n=7296`→1）；`block_m` 随
      `mk_alignment`(128/256) 变（`num_stages` **9→6**、`smem_size` **230204→205628**）；storage/
      pipeline/launch 逐字段对回公式（`swizzle a/b/cd=128`、`load_block_m=block_m/cluster_n`、
      `store_block_m=16`、`smem_extra=9020`、`num_threads=256`、`num_epilogue_threads=128`）；稠密 BF16
      枚举 **50 候选**、最优落回 `swap_ab=0`；`(A@B^T)^T==B@A^T` **逐位**（逐组 fp32 9.5e-06）；
      SM90 候选全 `swap_ab==0`（48 个）且源码 `sm90.hpp:125` 硬断言。另点明一个会咬人的数字：
      `mk_alignment` 直接进 JIT 模板实参（`sm100_bf16_gemm.hpp` 附近），配置改动会**静默改变 kernel 形态**。
      配套博客 `content/posts/mega-moe-103-sm100-swap-ab-dgrad`（索引页新增第 103 篇）。

### P94 增量（opt-in，补齐反向侧一直缺席的通信量模型与 H100 实测）

- [x] **P94-1** **MoE 反向的通信量模型与 H100 实测：反向为什么没有 mega-kernel**
      （`~/github/DeepEP/README.md:171-262`、`deep_ep/utils/refs.py:10-243`、
      `deep_ep/include/deep_ep/impls/ep/dispatch.cuh:104`、`deep_ep/buffers/ep.py:518-562`）。
      第 1 篇给了**前向**通信量模型、第 4/12/13/14 篇拆了反向**计算侧**、第 95 篇拆了反向**语义侧**
      （dispatch/combine 互为转置），但反向**通信量**从没有模型与实测。
      → `docs/104-moe-backward-communication.md`（真实行号核对）：**核心结论——反向通信不是更小、
      而是约 1.5× 前向**。原因是**布局不对称**：前向 dispatch 在传输层按目标 rank 去重
      （`dispatch.cuh:104` `ptx::deduplicate`、`refs.py:67` `mask_to_send`）→ T·d·H·1B（FP8）；
      而 combine 与**两个反向相位**都在 expanded 空间（T·k·H·2B，BF16）——官方
      `README.md:208/245` 明说「dispatch 的反向是 combine、combine 的反向是 dispatch」，且都带
      `do_expand=True`（`:197-198/253-254`）。模型 B_bwd/B_fwd=4k/(d+2k)≈1.49（k=8,d≈5.4）。
      H100 8 卡真实 ragged all-to-all 实测 `harness/exp_moe_backward_comm.{py,sh,out.txt}`
      （30 次均值）：单向带宽 **114–124 GB/s**、反向/前向**字节量比 1.49×、时间比 1.46–1.54×**
      （T∈{1024,4096,16384}）；ragged 相对 8 等分（第 1 篇 ~300+ GB/s）有 **2–3× 拆分惩罚**；
      combine 本地按 token 求和的数据量（expanded 行数×H）与通信同级（PyTorch `index_add` 朴素路径
      T=16384 达 25ms，仅示量级）；叠两流**只重叠 7.2%**（all-SM NCCL a2a 与 GEMM 争抢 SM）。
      **反向无 mega-kernel 的真因**：计算天然分四个独立 grouped GEMM、通信是两张方向相反的网；
      重叠改由通信库 `async_with_compute_stream=True`（`:217/252`）在 host 层做，且须像
      `ep.py:518-532` 限制通信 `num_sms` 才有真重叠。反向四个 GEMM（第 14 篇 ~4.15ms）与两相位 a2a
      （~8.0ms）同量级。**主 kernel / 真实 EPBuffer 仅分析**；通信原语模型与 H100 实测全可跑。
      配套博客 `content/posts/mega-moe-104-moe-backward-communication`（索引页新增第 104 篇）。

### P95 增量（opt-in，补齐 Mega MoE 的第二个 dtype 形态：FP4 路由权重）

- [x] **P95-1** **Mega MoE 的 FP4 路由权重路径（`fp8xfp4` / `kPackedFP4`）**：第 3/9/10/11 篇逐段拆了
      SM100 主 kernel、第 20 篇拆了它的 **BF16 变体**、第 35 篇拆了 SM100 稠密 GEMM 的 MXFP4/MXFP8、
      第 37 篇拆了参考量化器、第 96 篇拆了 JIT 包装层，但**同一 kernel 的第二个 dtype 形态——路由权重走
      FP4**——一直只在各处被一句「打包因子 2」带过。**主 kernel 的 MMA/TMA 路径 SM100 专属、仅分析**；
      但 FP4 的张量表示契约、权重重排、参考量化、host 配置全架构无关，H100 真机实测.
      → `docs/105-fp4-mega-moe-weights.md`（真实行号核对）：**全景** `"fp8xfp8"` 与 `"fp8xfp4"` 都映射到
      同一个 `MmaKind::MXFP8FP4`（`heuristics/mega_moe.hpp:73-78`），差异只落在路由权重张量 dtype
      （激活恒 FP8、共享专家恒 FP8、SF 恒 MN-major packed UE8M0 gran 32）。**FP4 表示**：
      `kPackedFP4 = torch::kInt8`（`utils/math.hpp:11`），每字节 2 个 e2m1，逻辑 K 靠
      `get_logical_shape`（`utils/layout.hpp:57-64`）与 TMA 描述符（`runtime_utils.hpp:301/316`）的「×2」
      恢复；host 契约 recipe 恒 `(1,1,32)`、L1/L2 同 dtype、权重 SF 必须 `kInt`（`mega_moe.hpp:176-204`）。
      **从 global 到 SMEM 的三层表示**：① 全局 2 元素/字节；② TMA 用
      `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`（`fp4_unpacked_smem=true`）搬运时把 4-bit **摊成可寻址的
      未打包形式**（`float_e2m1_unpacksmem_t`），SMEM 每元素仍 1 字节（`get_num_mma_elem_bytes=1`），故
      tile 容量/流水级数不变，只有线字节减半——barrier 的 `expect_tx` 用 `/(kIsWeightFP8?1:2)` 精确对上
      （kernel `:807-810`）；③ **因为 A=FP8、B=FP4 必须用 `MXF8F6F4`（unpacked，`UMMA_K=32`）而非
      packed `MXF4`**（`UMMA_K=64`，要求两侧都 e2m1，`heuristics/config.hpp:36`）。**权重重排**：FP8/FP4
      **共用同一条 tuple 路径**（`mega/__init__.py:146-164`），且证明 N 轴 gran-8 交错与 K 轴打包
      **可交换（逐位）**。**host 唯一 FP4 专属位**：`block_m∈(192,240]` 时 `store_block_m_l1=24`
      （BF16=40）、`block_m==240` 时 `store_block_m_l2=8`（`heuristics:112-115`，专为 FP8xFP4 prefill
      省 SMEM 换更深 TMA 流水）。H100 真机 `harness/exp_fp4_mega_moe_weights.{py,sh,out.txt}`
      **ALL_OK=True**：`symm buffer` fp8xfp4==fp8xfp8（2145.3MB）、`block_m` 与 dtype 无关、packed 往返
      逐位、`unpack∘interleave == interleave∘unpack` 逐位、FP4 权重使 MoE 输出误差 ~3.2×（量化本身
      ~4.4×）、权重显存**恰好减半**。配套博客 `content/posts/mega-moe-105-fp4-mega-moe-weights`
       （索引页新增第 105 篇）。

### P96 增量（opt-in，给前 105 篇做一次引用体检 + 全算子索引）

- [x] **P96-1** **DeepGEMM / DeepEP 算子总索引与引用行号健康检查（source-rot audit）**：
      105 篇已写 ~1900 处 `file:line`，会被内联到博客却无人回核；上游每版会漂移。本篇把「还有多少
      引用对得上」做成可复现数字，并给全局算子 → 覆盖篇目索引，**当场闭环修掉体检发现的真实错误**。
      **纯文件解析、架构无关，H100/任意机器可跑**。
      → `docs/106-deepgemm-source-index-audit.md`（真实数字）：源 `DeepGEMM 057ca59`（`#462`）/
      `DeepEP 93eb6eb`（v2.5.0），脚本 `harness/exp_deepgemm_source_audit.{py,sh}`（+`.out.txt`）。
      方法四层核对（可解析 / 行号在界 / 行非空 / 同 Markdown 行反引号标识符内容命中）+ 同名多文件
      按「(行号在界, 严格命中, 宽松命中, 路径后缀)」字典序打分消歧。**结果**：总 1904 条引用、
      可解析 **1898（99.68%）**、**行号越界 0**、无法解析 6（全为 `...` 占位）、被引行全空 10、
      内容抽查 741 条（严格 554=74.76%、宽松 646=87.18%）、覆盖 198 个源文件；**19 个 device impl +
      11 个 host api 全部至少被一篇覆盖（无 none）**；精确引用最多 `gemm.hpp`(95)/`ep.py`(66)/
      `layout.hpp`(60)。**修掉 2 处真实错误 + 1 处简写歧义**：① `docs/39`+`exp_legacy_grouped_gemm.py`
      的 `m_grouped_gemm.py:92/86-92`→`:72/61-72`（文件仅 84 行，重跑实验输出）；② `docs/81` 把
      `mlopart.py:135`→`tests/test_locality_domain.py:136/:150-151`（引错文件）；③ `docs/38` 的
      `mega_moe.cuh:1440/1419/613/1243`→全名 `sm100_fp8_fp4_mega_moe.cuh:…`（简写被解析到越界）。
      另列 10 条同名歧义 + 92 条轻微漂移（off-by-one / 同行多引用串扰的启发式假阳性）供后续复核。
      配套博客 `content/posts/mega-moe-106-source-index-audit`（索引页新增第 106 篇）。

### P97 增量（opt-in，引用体检第二遍：可复用 linter + 自指涉收口）

- [x] **P97-1** **把第 106 篇的引用体检做成可复用 linter，并收口它的 defer 项**：
      第 106 篇留下「10 条同名歧义 + 92 条轻微漂移」与「应统一写全相对路径」，但复现其脚本
      （今日 `total 1933 / unresolved 9 / oob 3 / empty 24`）发现**脚本自身有三个会静默误判的方法学
      问题**：① 索引扫进 vendored `third-party/cutlass`（与 `deep_gemm/include/cute|cutlass` 同份拷贝）
      与两份 `third-party/deep_jit` → `runtime.hpp`/`python_api.cpp` 各 4 候选、消歧易落错份；② 消歧
      排序键没利用「被引行是否为空」（作者核对过的引用极少指空行）；③ `impls/...cuh` 类占位被当错误。
      **纯文件解析、架构无关，H100/任意机器可跑**。
      → `docs/107-refs-linter-audit2.md`（真实数字）：新 linter `harness/check_refs.{py,sh,out.txt}`。
      改进四点：候选**按内容哈希去重**（同份拷贝折叠，`dedup_by_content`）、排序键
      `(被引行非空, 文档主仓库, 文档目录线索, 全语料引用偏好, 路径短)`（`repo_of`/`doc_repo_hist` 修掉
      跨仓库错配，如 DeepEP 文档里的 `csrc/python_api.cpp` 被解析到 DeepGEMM 那份）、`...` 单列
      `elided`、支持 fenced/`refcheck:ignore` 且给出 CI 退出码。**结果**：总 1699 条（排除 2 篇审计
      文档）、**无法解析/行号越界/被引行全空/同名歧义未消解 四类硬错误全 0**、严格命中 530
      （71.52%）、宽松 617（83.27%）、覆盖 236 个源文件（vendored 去重后；含 DeepJIT）。**闭环修掉
      2 处真实 off-by-one**：`docs/34` 的 `csrc/runtime/locality_domain.hpp:13`（空行）→`:16`（CUDA 13.4
      兼容层注释，与同句 `mlopart.hpp:13` 对应）；`docs/98` 的 `csrc/buffers/bucket.hpp:78`（空行）→`:77`
      （`num_storage_bytes % kNumAllocationAlignmentBytes` 断言）。**并量化「审计文档自指涉」**：把
      `docs/106` 纳入检查立刻多 `oob 3 + empty 2`，全来自它 §5 的修正名单（按设计引用坏引用）→ 体检
      不幂等，故两篇审计文档入 `EXCLUDE_DOCS`。87 条轻微漂移不逐条改写（多为「引用块/区间首行」约定
      与同行多引用串扰），保留在 `.out.txt`。配套博客 `content/posts/mega-moe-107-refs-linter-audit2`
      （索引页新增第 107 篇）。

### P98 增量（opt-in，补齐所有 GEMM / 反向 grouped GEMM 重复复用却一直没被单独读过的「任务调度器」）

- [x] **P98-1** **DeepGEMM 的通用 GEMM 任务调度器 `scheduler/gemm.cuh`**：
      `deep_gemm/include/deep_gemm/scheduler/gemm.cuh`（314 行）+ 五个调用方
      （`impls/sm90_bf16_gemm.cuh:135`、`sm90_fp8_gemm_1d1d.cuh:161`、`sm90_fp8_gemm_1d2d.cuh:153`、
      `sm100_bf16_gemm.cuh:158`、`sm100_fp8_fp4_gemm_1d1d.cuh:199`）+ `common/types.cuh:13-21` 的
      7 种 `GemmType`。前面 107 篇把它当「`scheduler/gemm.cuh:xxx`」引用了 15+ 次，却从没单独拆透；
      它正是 MoE 反向 **wgrad（k-grouped）/ dgrad（m-grouped）** 的任务枚举地基（第 4/12/13/14 篇的
      计算侧、第 26/91 篇的 SF 打包、第 103 篇的 swap-AB 尾块都指向它）。**唯一架构分支是
      `#if __CUDA_ARCH__ < 1000`（SM90 multicast 修补），其余全是整数索引、不碰 UMMA/TMA，H100(sm90)
      可直接 include 真实头文件跑真实设备代码**。
      → `docs/108-deepgemm-gemm-scheduler.md`（真实行号核对）：`get_num_1d_blocks_per_group:14-26`
      （编译期在 `{8,16}` 用「组跨度 + 每 SM 迭代数」代理代价选 `N1D`，随 `kIsMulticastOnA` 换主轴）；
      `Scheduler:30-39` + `kKAlignment%128==0` 静态断言 `:42-43`；`get_next_k_group:71-82`（非 psum
      前缀和 vs psum「起点对齐 `kKAlignment`、长度=未对齐结尾−对齐起点」）；构造 `:85-106`；
      `get_swizzled_block_idx:108-144`（L2 组 swizzle + `#if __CUDA_ARCH__<1000` 的 SM90 奇数分组
      拆最后 1 块 `:120-133`）；`get_global_idx:146-174`（7 形态 × {MN,K,SF_K} 偏移；
      `MGroupedContiguous` 读**第 4 参** `m_block_idx` 指向的 `grouped_layout[m*BLOCK_M]`，
      A 只传 3 参/ B 才传 4 参）；`get_aligned_effective_m_in_block:176-183`（swap-AB+psum+关零填充
      尾块裁到 16）；`get_next_block:185-272`（网格步进 + 六分支：Masked/psum 逐组变 `num_m_blocks`、
      k-grouped 更新 `k_start`/`sf_cumsum`、Batched `current_group_idx=nbi/num_blocks`）；SM90 专属
      `is_tma_multicast_valid:274-292`（MGroupedContiguous onM 要求 peer CTA 同组）与
      `is_computation_valid:294-309`。H100 实测 `src/gemm_scheduler.cu` +
      `harness/exp_gemm_scheduler.{sh,out.txt}`（同 `.cu` 编 `sm_90a` 运行 + `sm_100a` 仅编译覆盖；
      源 sha256 `f48dfef6…`）**51 PASS / 0 FAIL**：① 启发式 6/6 vs 暴力；② swizzle 对
      `[0,num_m*num_n)` 穷举 10 组全部双射、与独立复刻逐位相等，`7x5` 的 `fixup_hits=35` /
      `8x6=0` 坐实 SM90 修补非摆设；③ 8 种形态任务枚举是 `[0,N)` 的排列（`N` 由规范独立算出，
      含 Batched 96 / Masked 27 / psum 24），k-grouped 逐组 `(k_start,sf_cumsum)` 含不齐整端点
      `KA=256` 全 `bad=0`；④ psum 尾块有效 M=16、非尾块 128；⑤ multicast 对端一致（peer-eq=1 /
      peer-neq=0）；⑥ `get_global_idx` 七形态偏移全对（`MGroupedContiguous` 同一条调用随
      `grouped_layout` 从 256256 变 1256）。SM100 分支仅编译期覆盖、本机不可运行。配套博客
      `content/posts/mega-moe-108-deepgemm-gemm-scheduler`（索引页新增第 108 篇）。

### P99 增量（opt-in，补齐贯穿训练/推理部署的 CUDA Graph 兼容性面）

- [x] **P99-1** **DeepGEMM / DeepEP 的 CUDA Graph 兼容性面与 capture 前的 warm-up 契约**
      （`csrc/apis/mega_mhc.hpp:26-39`、`csrc/apis/mega_gate.hpp:13-26`、
      `csrc/apis/attention.hpp:284-301`、`csrc/runtime/runtime.hpp:67-76`、
      `third-party/deep_jit/.../backend/cuda/kernel.hpp:116-152/169`、
      `tests/test_mega_mhc.py:259-283/298-315`；
      DeepEP `csrc/buffers/ep.hpp:163-170`、`csrc/utils/event.hpp:15`、
      `deep_ep/utils/event.py:27-29`、`csrc/buffers/bucket.hpp:48-58`）。
      第 50/52/98 篇分别拆过 `Runtime` 生命周期、`cuLaunchKernelEx` 发射层、DeepEP 编译配置面，
      但「把算子放进 CUDA Graph 会怎样」这条贯穿部署的线索一直没被单独讲过。**主体 kernel/API
      SM100 专属、仅分析**（三处断言都在 `arch_major==10` 闸门之后）；**发射层与真实 SM90 kernel
      在 graph 下的 capture/replay H100 真机实测**。→ `docs/109-cuda-graph-capture.md`（真实行号核对）。
      **三层时间轴**是根因：host 一次性初始化 / capture（把 stream 上操作录成 DAG）/ replay；
      懒初始化落在捕获里 → 一次性工作被录进 DAG 并每次 replay 重放。三个 API 用同一模板（按
      `CUDAStream` 懒分配一块**必须为零**的 scratch + 首次分配前断言
      `c10::cuda::currentStreamCaptureStatusMayInitCtx() == CaptureStatus::None`，
      torch 头 `c10/cuda/CUDAGraphsC10Utils.h:70`）。**关键对照**：cuBLASLt workspace 也是按 stream
      懒分配，但用 `torch::empty`（无 zeroing）**故不需要断言**。发射层全部 capture-safe
      （`cuFuncSetAttribute` 是 host 侧属性不进 DAG；cooperative/cluster/PDL 三属性 +
      `cuLaunchKernelEx` 受支持）。H100 实测 `harness/exp_cuda_graph_capture.{py,sh,out.txt}`
      **18 PASS / 0 FAIL**：A) 忠实 Python 复刻三 API 契约（未 warm-up 拒绝且未分配 / warm-up 后
      capture 且复用同对象 / 新 stream 独立 / capture 内 zeroing 被重放 / `empty` capture-safe）；
      B) 真实 SM90 `bf16_gemm_nt`（自研/默认 cuBLASLt）与 `fp8_gemm_nt`（1d2d）、`set_pdl(True)`
      的 capture+replay 与 eager **逐位相等**（含「全新 stream 首次调用即在 capture 内」的 workspace
      懒分配）；C) 64×`256³` bf16 eager **12.14 us/call** vs graph replay **6.77 us/call**、**1.79×**；
      D) profiler 抓到 replay 发射的 `deep_gemm::sm90_bf16_gemm_impl<...>`。另对照 DeepEP 的
      `EP_AVOID_RECORD_STREAM`（`record_stream` 对 graph 不感知）路线。配套博客
      `content/posts/mega-moe-109-cuda-graph-capture`（索引页新增第 109 篇）。

### P100 增量（opt-in，补齐第 49 篇 SM90 WGMMA 底座的 SM100 对偶 + 第 17 篇只当对照的 SM100 分支）

- [x] **P100-1** **SM100 的 UMMA 指令底座 `ptx/tcgen05.cuh`**（`deep_gemm/include/deep_gemm/
      ptx/tcgen05.cuh` 219 行；消费者 `sm100_bmk_bnk_mn.cuh` / `sm100_bmk_bnk_mn.hpp`；
      描述符层 `mma/sm100.cuh`；`cute/arch/copy_sm100.hpp` / `copy_sm90_tma.hpp`；
      13 个 `#include` 它的设备文件）。第 49 篇（P39-1）把**所有 SM90 kernel 每一行 MMA 都在调**的
      底座拆清了（`mma/sm90.cuh` + `ptx/wgmma.cuh`），第 35 篇（P25-1）拆了 SM100 稠密 GEMM 后端并
      过了 `mma/sm100.cuh` 的 smem 描述符，但夹在中间、**13 个 SM100 设备文件都 include 的
      `ptx/tcgen05.cuh`** 从没被单独读过——它是 SM100 上所有 `tcgen05.*` 裸指令（`mma`/`commit`/
      `fence`/`ld`）的唯一封装点，是第 49 篇的严格对偶。**tcgen05 指令 SM100 专属、仅分析**
      （§6.1 给出 ptxas 逐条拒绝的硬证据）；**它脚下的内存几何（STSM swizzle / TMA reduce-add）
      H100(sm90a) 真机可跑**。
      → `docs/110-sm100-tcgen05-umma.md`（真实行号核对）：**三层**——① **9 个 `*_SS::fma` 封装**
      （`:12-187`，6 个 data kind × {1SM,2SM} + 1 个 `ws`）：`cta_group::1/2`、`kind::f16/f8f6f4/
      mxf8f6f4.block_scale/mxf4.block_scale[.block32|.scale_vec::2X]`（CUDA≥12.9 分支 `:137-141/161-165`）、
      **`desc >> 32` = 32 位 instruction descriptor**（`:25`，对应 `mma/sm100.cuh:144-148` 的 `<<32`）、
      **`setp.ne.b32 p, scale_c, 0`**（`:22`）= `tcgen05.mma` 的「累积/清零」predicate（kernel 传
      `s>0 or k>0`）；`F16BF16_WS_SS`/`F8F6F4_*` 在公开仓库**无调用点**。② **TMEM/commit/fence**——
      `tmem_load_32dp32b<N>`（`:190-204`，6 档宽度，仅 MQA logits/Mega Gate 用）、
      `umma_arrive_no_elect`（`:206-209`，不做 `elect`，由调用方保证）、`tcgen05_before/after_thread_sync`
      （`:211-217`，after=等之后读之前、before=读之后同步之前）。③ **消费者 `sm100_bmk_bnk_mn.cuh`**
      逐段——把第 17 篇 §6 的「三行对照」落到一个只用一个封装的**最小样本**：warp 角色/TMEM
      `Allocator1Sm`（`:80-93`）、每 lane 预存 stage 描述符 + `__shfl_sync`（`:143-144/165-166`）、
      `fma(...,0,s>0 or k>0,...)`（`:174`）、`ld_shared(tmem_ptr)==0` 唯一性（`:188-189`）、
      epilogue `TMEM_LOAD_32dp32b4x`+`st_shared`+**`SM90_TMA_REDUCE_ADD_2D`**（`:245-255`）；
      host 包装 `split_factor`（`:34`）/stage 回退（`:38-55`）/**D 描述符 `(n,m)` 方向**（`:69`）。
      另给与第 49 篇的**八维对照表**（描述符/发指令主体/累加器/WGMMA wait_group vs tcgen05 commit→
      mbarrier/fence/选择器）。**H100(sm90a) 实测**（`src/tcgen05_umma.cu` +
      `harness/exp_tcgen05_umma.{sh,out.txt}`）**7 PASS / 0 FAIL**：① **ptxas 编译矩阵**——`tcgen05.mma/
      .mma.ws/.commit/.fence` 在 `sm_90a` **逐条 REJECT**（`Feature '.cta_group::1' not supported` /
      `Instruction '...' not supported on .target 'sm_90a'`）、`sm_100a` 全 OK；`tmem_load` 因 CUTLASS
      arch 守卫（`copy_sm100.hpp:3474`）在 `sm_90a` **编得过但运行期 trap**（编得过≠跑得动）；
      ② STSM swizzle 整数公式（`sm100_bmk_bnk_mn.cuh:223-249` 原样抽取）对 **128×8 全枚举** ==
      `cute::Swizzle<3,4,3>`（mismatch 0/1024、16B 组内碰撞 0/16384）；③ **真实 TMA reduce-add**：
      初值 0.5 + 3×1.0 → **3.5**（坐实是加非写，覆盖写应为 1.0）；④ swizzle-128 reduce-add 以 bmk 的
      `(n,m)` 方向各打两段列 → **mismatch 0/16384**；⑤ `tcgen05.cuh`+`mma/sm100.cuh` 在 `sm_90a` 下
      可 include、`InstrDescriptor`==4B。配套博客 `content/posts/mega-moe-110-sm100-tcgen05-umma`
      （索引页新增第 110 篇）。

### P101 增量（opt-in，补齐「反向」一直缺席的端到端误差分解）

- [x] **P101-1** **MoE 反向的 H100 数值复刻与误差分解**（第 14 篇的反向是「kernel 接得对不对」、
      第 99 篇的前向是「误差从哪来」；本篇是第 99 篇的**反向对偶**——问「四项梯度的误差从哪来、
      能否可加拆开、梯度方向是否保住」）。Mega MoE 没有反向 mega-kernel（第 14 篇 §6），反向天然
      是四个独立 grouped GEMM。**主 kernel（SM100 tcgen05/packed UE8M0）仅分析**；反向的
      数学/量化/布局用 SM90 `k_grouped_*_nt_contiguous` / `m_grouped_*_contiguous` 复刻，**H100 真机**。
      → `docs/111-moe-backward-emulation.md`（真实行号核对）：先钉语义——Mega MoE 把 top-k 权重
      **提前乘进 L1 输出**（`sm100_fp8_fp4_mega_moe.cuh:1134` `__fmul2_rn(__fmul2_rn(gate,up),weights)`，
      权重 `:1087-1097` 缓存广播、`:604` 写入），combine 退化成纯 fp32 求和（`:1477-1490`），
      故反向里 `dY` 直传、`w` 只出现在 SwiGLU 反向。四个 GEMM 角色（`W1=[E,2I,H]`、`W2=[E,H,I]`）：
      `dW2=dYᵀa`（K=token，两操作数都激活）、`dA=dY·W2`（M=token，B=权重）、`dW1=dGᵀx`、`dX=dG·W1`；
      host 入口 `gemm.hpp:373/181`、K 对齐 `:69`。**两层实验**（`harness/exp_moe_backward_emul.{py,sh,out.txt}`）：
      ① **真实 kernel 锚点**（`T=64,H=512,I=256,E=8,topk=2`）——`kernel vs deq` dW **2.6e-4**（与第 14 篇
      逐位同量级）、dX 1.5e-3（dgrad 输出经两个 top-k 槽 bf16 scatter-add）、`kernel vs fp64` 2.4e-2~3.5e-2、
      cos ≥0.9994；② **量化来源分解（架构无关）**——参考量化器（激活 per-token 1D、权重 per-128 块 2D、
      `dY` per-token）在去量化操作数上做精确 GEMM，逐来源开关。**H100 实测**（fp64 手写反向 vs
      `torch.autograd.grad` 逐位 0 不符）：`none` 只剩 fp32 累加地板 ~5e-4~7e-4；`fwd_only` 就把 dX 推到
      **7.2e-2**（占 `full` 8.97e-2 的 **80%**）——**反向误差一大半是前向带进来的**（前向把已量化的
      激活/权重存下来），反向四项合计只多 ~1.7e-2。逐梯度分工：`dW2` 对反向权重量化**完全免疫**
      （`+bwd_w` 与 `fwd_only` 逐位相同，`dW2` 不含权重）、`dX` 对反向权重量化最敏感（7.23e-2→8.25e-2，
      per-128 块 2D 比 per-token 1D 粗）、`dW1` 对反向激活更敏感。DeepSeek 量级（`T=2048,H=7168,I=2048`）
      规律不变（前向主导 ~90%），最坏 `full` 下 dX cos=0.9927 / dW2 cos=0.9980，**方向全部 >0.99**。
      与前向对照（第 99 篇激活/权重「各占一半」5.7e-2/4.6e-2）：反向 ≈ 带进来的前向误差 + 新引入的
      ~1.7e-2——**实践含义：想提反向梯度精度先治前向的 block-scale 量化**。配套博客
      `content/posts/mega-moe-111-moe-backward-emulation`（索引页新增第 111 篇）。

### P102 增量（opt-in，补齐 P54 只当「仅分析」4 行带过的两个 all_gather SM kernel 后端）

- [x] **P102-1** **DeepEP bucket all_gather 的两个 SM kernel 后端（pure-RDMA / hybrid）**
      （`csrc/kernels/bucket/all_gather.hpp`、`csrc/buffers/bucket.hpp:245-330`、
      `deep_ep/include/deep_ep/impls/bucket/all_gather/{rdma,hybrid}.cuh`、
      `deep_ep/include/deep_ep/layout/bucket/{all_gather,copy_engine,workspace}.cuh`、
      `csrc/kernels/driver/driver.cpp:14-87`、`deep_ep/buffers/bucket.py:361-392`、
      `tests/bucket/test_all_gather.py`）。第 54 篇（P54-1）拆了 `BucketBuffer` 但只把**纯 NVLink
      all-gather**（copy engine、无 SM kernel）讲透；第 62 篇（P62-1）补齐了 `all_reduce`/`reduce_scatter`
      三后端——唯独 all_gather 的 **pure-RDMA 与 hybrid 两个 SM kernel** 两处都只 4 行带过。本篇补全，
      「三类算子 × 三物理域」矩阵拼完。**host** `launch_all_gather`（`all_gather.hpp:24-230`）按物理域三分支：
      pure RDMA（`:36-61`，512 线程满卡）、pure NVLink CE（`:62-108`）、hybrid（`:109-229`，`grid=num_rdma_ranks`）。
      **pure-RDMA kernel**（`rdma.cuh:11-49`）：rail 域、`global_warp_idx` 步进认领目标 rank、lane 步进 bucket、
      `gin.put<ncclTeamTagRail>` 发自己那份 shard。**hybrid kernel**（`hybrid.cuh:11-80`）：SM=目标 RDMA rank、
      warp=QP，先清 `tails` + 入口屏障 + 退避（`:34-51`），按 bucket→chunk→QP 发，每 chunk
      `num_bytes_per_qp=align(ceil_div(chunk,QPs),64)`、`qp_offset=min(warp*per,chunk)`（`:61-64`），
      **空 QP 也照发 signal**（`:66-76`）。**chunk 自选择代价模型**（`:116-148`）：`push_t/rdma_t/forward_t`
      + CE 每波 10us，枚举 `i≤64` 取最小，`clamp(align(ceil_div(B,best),8*64),4KiB,256MiB)`；
      `kNumMinChunkBytes=4MiB` = 「每 QP 至少 512KiB」NIC 经验。**完成协议**：每 `(chunk,QP)` 远端
      `tails[qp][chunk]` +1，host `copy_engine_wait` 按 `WAIT_VALUE_GEQ, target=num_rdma_ranks-1` 等。
      数据面 NCCL GIN → **仅分析**；信号布局/寻址/协议架构无关，H100(sm90a) 直接 include 真实头文件跑真实设备代码
      + 真实驱动 mem-op（`src/bucket_allgather_backends.cu` + `harness/exp_bucket_allgather_backends.{sh,py,out.txt}`）：
      `AllGatherSignals`=524288(512KiB)/`BucketSignals`=794688 + 字段偏移；4 组 `(R,SMs)` 每非己 rank 恰 1 warp、
      每桶覆盖一次；10 组 chunk QP 拆分覆盖恰好一次/64B 对齐/尾段可非 64 倍数；完成协议四场景（每槽恰 R-1 /
      per-chunk 独立 / GEQ 过计通过 / 未达阈值不通过）；Python host 算术层独立复刻一致（2 带宽 × 5 拓扑 ×
       6 shard）。配套博客 `content/posts/mega-moe-112-bucket-allgather-sm-backends`（索引页新增第 112 篇）。

### P103 增量（opt-in，补齐第 23 篇对 SM100 dense MQA logits 主 kernel 只给的一页摘要）

- [x] **P103-1** **SM100 dense MQA logits 主 kernel 逐段精读**
      （`deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`，609 行；`layout/mqa_logits.cuh`；
      `csrc/apis/attention.hpp:18-282`、`csrc/jit_kernels/impls/sm100_mqa_logits.hpp:11-155`）。
      第 19/23 篇拆的是它的**调度器**（H100 真机可跑），第 22 篇拆的是稀疏版主 kernel 概览；
      第 23 篇 §5 只给主 kernel 一页摘要（列结论、没逐段读、没真机验证其架构无关部分）。本篇补全。
      → `docs/113-mqa-logits-dense-kernel.md`（真实行号核对）：一次 launch 16 warp 五类角色
      （math 0..11 `:430-533` / SF-Q 生产者 `:278-280` / KV 生产者 `:281-283` / UTCCP `:284-364` /
      UMMA `:365-429`）；`BLOCK_Q=128/num_heads`、`UMMA_M=UMMA_N=128`、`SPLIT_KV=384`、
      **`kNumTmemStages==kNumMathWarpGroups=3`**（`:99`）让一个 KV split 的 3 段 UMMA 被 3 个
      math warpgroup 各吃一段；TMEM 三块列预算（`:101-108`）+ SF 的 128 对齐；`reuse_kv`
      （`:151-153`/`:335-343`）跳过重复转置。**五段架构无关逻辑 H100(sm90a) 真机 ALL PASS**
      （`src/mqa_logits_dense_epilogue.cu` + `harness/exp_mqa_logits_dense_epilogue.{sh,out.txt}`）：
      **A** `dispatch_num_block_tokens`（`:27-36`）运行期 token 数→`Int<max(1,min(N,BLOCK_Q))>`
      （N=0 也退化为处理 1 token；`BLOCK_Q∈{1,2,4}` 共 19 点全对）；**B** `visit_tmem_head_chunks`
      （`:39-49`）按 16/8/4 贪心切头，起点连续、宽度和恰为 H（9 组 H 与 host 复刻逐项 0 不符，
      `H%4==0` 不越读）；**C** `reduce_pairs`+`store_token`（`:473-489`）：`cvt.relu`+`__hfma2`+
      两条 `sum_0/sum_1` bf16x2 累加器把 `Σ_h relu(acc_h)·w_h` **整个在 bf16 里做**，「一次性 vs
      分段」归约**逐位相等 0/64**、device vs host 逐位重放 0/64、bf16 累加 vs fp32 参考
      `max|Δ|=1.99e-1`/`max|Δ|/Σ|term|=2.55e-3`（≈2^-9；`max|Δ|/|ref|` 可达 O(1) 纯属相消），
      压缩写回窗口与值 0 错且窗口外保留 poison；**D** `issue_paged_split`（`:184-221`）的
      `kNumSFTMAs`/`kNumKVPagesFromSFProducer` 两生产者 KV 页均分（PAGE_KV 32/64/128 → 4/8、0/6、0/3，
      `kv_from_sf+kv_from_kv==pages`）；**E** `utccp_required_smem_warp_transpose`（`:288-296`）的
      `src[i*32+lane]→dst[lane*4+i]` 4×32→32×4 转置 128 元素 0 不符。与 SM90 版八维对照表。
      **UMMA/tcgen05/UTCCP/2-SM TMA SM100 专属、仅分析**。配套博客
      `content/posts/mega-moe-113-mqa-logits-dense-kernel`（索引页新增第 113 篇）。

### P104 增量（opt-in，补齐第 65 篇只提一句的 DeepEP 语义工具层）

- [x] **P104-1** **DeepEP 的 MoE 语义工具层 `deep_ep/utils/math.py`**
      （`deep_ep/utils/math.py:5-103`；对照 `deep_gemm/testing/numeric.py:5-27`、
      `deep_gemm/utils/math.py:19-49`）。第 65 篇拆了 DeepEP `deep_ep/utils/` 的
      `refs.py`/`gate.py`/`testing.py`，却只顺带提了一句同目录 `math.py`「有个和 DeepGEMM 同名但
      不同定义的 `calc_diff`」。本篇把这 103 行单独讲透——它装的是 **DeepEP 测试与下游共用的
      MoE 语义词汇**：判据 `calc_diff`、参考量化器 `per_token_cast_to_fp8/back`、**top-k 专家去重
      oracle** `inplace_unique`、**group-limited 路由掩码** `create_grouped_scores`、记账
      `count_bytes`/`hash_tensor(s)`。→ `docs/114-deepep-utils-math.md`（真实行号核对）：
      `calc_diff:5-9`（`+1` 避 0/0、破尺度不变；余弦式≡平方误差式）、`safe_div:12-19`、
      `per_token_cast_to_fp8:31-39`（未打包/未 UE8M0/gran 硬编码 128）、
      `per_token_cast_back:43-56`（int 分支 `:51-53` 的 `view(uint8)<<23`）、
      `inplace_unique:59-71`（`:66` 按 count 排序是死代码）、`create_grouped_scores:74-79`
      （DeepSeek-V2/V3 device-limited 路由掩码）、`hash_tensor(s):82-93`、`count_bytes:96-103`。
      三个函数（`inplace_unique`/`create_grouped_scores`/`hash_tensor`）**公开仓库无调用点**。
      **`math.py` 只 `import torch`**，按路径 exec 真实源码即可（无需绕过 `check_nccl_so`）；
      H100 真机 `harness/exp_deepep_utils_math.{py,sh,out.txt}` **33/33 ALL PASS**：
      A) `calc_diff` 与 DeepGEMM 差 `+1`（`DeepEP(x,y)==DeepGEMM((x+1),(y+1))`）、全零两边 0、
      缩放 spread `6.6e-3` vs `4.6e-9`；B) scale 逐位相同但 fp8 `10/6400` 差 1、
      packed-int 解码≡UE8M0 fp32、int 分支≡fp32 分支；C) `inplace_unique`≡独立 oracle、
      `:66` 死代码、降序（对照 DeepGEMM 升序）；D) `create_grouped_scores`≡向量化复刻、
      每 token 恰保留 16 专家；E) `count_bytes`≡DeepGEMM、hash/pack 语义全对。配套博客
       `content/posts/mega-moe-114-deepep-utils-math`（索引页新增第 114 篇）。

### P105 增量（opt-in，P104 延伸：`refs.dispatch` 的 FP8 分支）

- [x] **P105-1** **DeepEP `refs.dispatch` 的 FP8 分支 + 设备端 `sf_pack_t`/`TokenLayout` SF 布局**
      （`deep_ep/utils/refs.py:10-123`、`deep_ep/utils/math.py:30-56`；
      `deep_ep/include/deep_ep/{common/compiled.cuh:59-68,layout/ep/token.cuh:10-143,
      impls/ep/dispatch.cuh:262/293-309,impls/ep/dispatch_copy_epilogue.cuh:146-178}`；
      host 编排 `csrc/buffers/ep.hpp:257/269/351-356/689-690`、`csrc/kernels/ep/dispatch.hpp:26-28`；
      `tests/engram/test_engram.py:19`）。第 65 篇只点了一句 `refs.py:46-49` 的
      `hidden_sf==ceil_div(hidden,128)`、第 114 篇拆了参考量化器返回 fp32；本篇补上 `(x, sf)`
      元组通路到设备端布局的接缝。→ `docs/115-deepep-refs-fp8-dispatch.md`（真实行号核对）：
      参考侧 `use_fp8=isinstance(x,tuple):39`、SF 列数契约 `:49`、逐 rank 选 token 时 SF 同下标
      `:73`、SF **单独一次** `all_to_all_single` 但同计数 `:107`、接收侧只掩码索引不碰 SF `:112-117`、
      打包回元组 `:120`；设备侧 `sf_pack_t` union（`compiled.cuh:59-62`，**全仓唯一 `ue8m0`
      字样**）、`TokenLayout` 三段布局 + `get_sf_ptr`/`get_metadata_ptr`（`token.cuh:34-37/55-60`）、
      kernel `cp.async` 搬 SF（`dispatch.cuh:299-307`）、epilogue 写回（`dispatch_copy_epilogue:146-178`）、
      BF16 `if constexpr` 消除（`:147`）。**关键发现（同名不同契约）**：参考固定 gran-128
      （SF 税 3.125%），设备 **dispatch 列数取 `sf.size(1)`**（`ep.hpp:356`，每单位 4 字节 `:355`、
      `:351` 注释）、**缓冲**按 gran-32 上界 `ceil_div(hidden,32)` 预留（`:269`，配 `:257` 断言
      `4*ceil(H,32)<=H`）恰是参考的 **4 倍**、输出可切 **TMA 列主序**（`:689-690`）。「一列=一
      `sf_pack_t`=128 元素」的统一解释 union 两成员（fp32 gran128 / UE8M0 gran32 打包），
      DeepEP 从不解释这 4 字节。H100 真机两项验证 `harness/exp_deepep_refs_fp8.{py,sh,out.txt}`
      + `src/deepep_token_layout_sf.cu`：① **8×H100 NCCL all_to_all**（T=512,H=512,E=32,k=4）
      `(x,sf)` 分支四项与**独立 oracle 逐位相等**、`per_token_cast_back` 反量化 rel=0、
      `hidden_sf` 不符被 assert 拒；② include 真实 `layout/ep/token.cuh`、**host 调用真实
      `TokenLayout`/`BufferLayout`**，7 组 `(hidden,topk,md)` 5 字段与 Python 独立公式**逐字段 0 不符**
      （Part C 11/11、Part D 6/6）。`combine` 无 FP8 变体（签名无 `(y,sf)`、pre_combine 恒 bf16）。
      端到端 `dispatch.cuh` 仍需 NCCL≥2.32.3（阻塞）。配套博客
      `content/posts/mega-moe-115-deepep-refs-fp8-dispatch`（索引页新增第 115 篇）。

### P106 增量（opt-in，补齐 DeepEP `utils/` 的「流量生成层」）

- [x] **P106-1** **DeepEP 的 MoE 路由分数/拓扑生成器 `deep_ep/utils/gate.py`**
      （`deep_ep/utils/gate.py:4-180`；判据/上游 `tests/utils/test_gate.py:7-53`、
      下游 `tests/ep/test_ep.py:13/77`）。第 65 篇把 `gate.py` 一句话概括为 DeepEP
      `deep_ep/utils/` 的「流量生成层」、第 114 篇拆 `math.py` 时又顺手提过它，但这份
      **让每份 EP 测试都能造出「某 rank 流量是别人几倍」最坏情况的路由夹具**从没被单独读过。
      `gate.py` 只 `import torch`，按文件路径直接 exec 即可（绕开 `deep_ep/__init__.py:39` 的
      `check_nccl_so()` 运行期阻塞），故 **H100 真机可跑真机实测**。
      → `docs/116-deepep-gate-generators.md`（真实行号核对）：**它是夹具不是算子**——真实 router
      输出不可控、无法稳定复现不均衡，于是 `gate.py` 制造**可控的 rank 级不均衡**，分两条互补
      路线：`precise`（先造 rank 级流量表 `rank_count [T,R]`、再把这几个位置填高分，承诺「特殊
      rank 0 恰拿 `ratio` 倍、其余 rank 近乎全均衡」）与 `random`（直接给专家打分、只保证比值大致）。
      四个非平凡点：① `generate_topk_idx:4-29` 是 `rank_count ↔ topk_idx` 的**精确逆映射**——
      逐 rank 独立乱序（`:16-18`）取前 `rank_count[t,r]` 个（`:22-23`）、展平后同 token 内**先按
      rank 分组**；② `generate_rank_count:32-113` 先量「每 token 触及几个不同 rank」`a`（`:49-51`）
      并 clamp 到 `int((R-1)/ratio)+1`（`:54`），再用**循环重复的同一个随机排列**（`:78` 的
      `randperm.repeat` + `:98-104` 的 `col%a` 填充）铺普通 rank——**这才是 precise 均衡的机制**；
      且 `rank_count` 元素**可以是 0..k**（多个 top-k 专家落同一 rank，实测 `max_entry=4`）；
      `must_mask=(a==num_ranks)`（`:66`）只在 `k>=R` 时非空；③ `get_precise_unbalanced_scores:116-137`
      用 **0.9 阈值**把选中/背景分数分开（`:127-136`）；④ `map_unbalanced_ratio_to_factor:148-164`
      在单标量 `factor`（rank 0 专家分数上界）上 20 轮二分（`:149-164`，度量是 `one_hot.any:158-159`）
      ——**大 R 时近似阈值开关**（`R=64` 附近 `ratio` 0.0→3.98→7.07）、**不可行目标静默饱和到上界**
      （`R=16,EPR=1,k=4` 的 `ratio` 上限 ≈ `(R-1)/(k-1)=5`，目标 8 使二分返回 `factor=99.9999`、
      实际 `ratio=4.959`，**不报错**）。**可行性闸门只在 precise 测试路径**（`test_gate.py:20-24`
      的 `lb<=ub`，下界 `max(1,⌈k/E_s⌉)`、上界与 `gate.py:54` 同源；完整矩阵 **405→379、跳过 26**），
      random 路径无保护。H100 真机 `harness/exp_deepep_gate.{sh,py,out.txt}`（`ALL PASS`，源码
      sha256 `gate.py=5372e456…`）：`generate_topk_idx` 14 例**逐位可逆**且组内有序；结构不变式
      7 组行和恒 `==k`、特殊 rank 恒为 rank 0（实测 `ratio=3.996`）；复刻 `test_gate.py:38-52`
      判据 precise **45/45** 满足 vs random **14/45**（坐实断言只在 `if precise:` 分支）；
      分数分离 selected∈`[0.9,1.0]`/bg∈`[0,0.9]`；二分单调命中可行目标、不可行静默饱和；
      同 seed 逐位可复现、不同 seed 不同。配套博客
       `content/posts/mega-moe-116-deepep-gate-generators`（索引页新增第 116 篇）。

### P107 增量（opt-in，补齐每次部署都要碰、却从没单独成篇的「环境变量 / 开关面」）

- [x] **P107-1** **DeepEP 的环境变量 / 开关面**（`csrc/utils/system.hpp:12-29` 的 `get_env<T>`；
      `setup.py:14-15/63-74` 的 `persistent_env_names` 与生成的 `deep_ep/envs.py`；
      应用点 `deep_ep/__init__.py:8-16`；引导闸门 `deep_ep/__init__.py:19-41`；以及 11 个运行时
      `EP_*` 的全部消费点）。前面 116 篇把 kernel / host 编排 / 测试基座几乎拆尽，但 DeepEP 的行为
      在**四个时间点**被环境变量塑形——`import` 时（引导闸门）、构造 `Context`/`Buffer` 时、
      **每次 API 调用**时、JIT **编译期**——这层从没单独成篇，与第 38 篇（DeepGEMM 的
      `DG_*`/`set_*`）成对照。**纯 host / Python + 编译期、架构无关，H100 真机可跑真机实测**。
      → `docs/117-deepep-env-surface.md`（真实行号核对）：**两句总览**——① 比 DeepGEMM 那套更隐式：
      `DG_*` 有 pybind 可回读，`EP_*` **全靠 `getenv`、无回读、C++ 与 Python 各解析一遍**；
      ② **两个通道**：安装期把白名单（7 条）固化进生成的 `deep_ep/envs.py`（`setup.py:63-74` 的
      `generate_default_envs`），运行时 `__init__.py:8-16` **只补不覆盖**（真实生成物只固化
      `EP_NCCL_ROOT_DIR=/tmp/nccl229`，坐实「这份 DeepEP 编到哪份 NCCL」）。**`get_env<T>` 的
      两个陷阱**：① 只支持 `int`/`string`，别的类型走 `EP_HOST_ASSERT(false)` 抛 `EPException`
      （`system.hpp:27`）；② **畸形/空值返回未初始化值（UB）**——`:22-24` 丢弃 `sscanf` 返回值，
      `abc`/空串下 `value` 不写、返回**栈垃圾**（既非默认值也非 0），g++ 真跑得 `21925`、哨兵复刻
      坐实（⇒「默认关」的开关写成 `=yes` 会**时开时关、不可复现**）。**全库清单**：带引号正则
      （排除裸 `EP_*` 宏）扫两仓库，得**恰好 16 个** = 7 persistent + 9 仅运行时，并集即 **11 条
      运行时开关**（`EP_SUPPRESS_NCCL_CHECK`/`EP_REUSE_NCCL_COMM`/`EP_DISABLE_GIN`/`EP_*_RDMA_SL`/
      `EP_BUFFER_DEBUG`/`EP_AVOID_RECORD_STREAM`/`EP_NUM_MAX_LOCAL_RANKS`/`EP_USE_NVIDIA_TOOLS`/
      `EP_DISABLE_BARRIER_PROFILING`/`EP_GIN_GDAKI_DEBUG`）。**三个真坑**（源码无注释、只能实测发现）：
      ① **`EP_BUFFER_DEBUG` 跨语言分歧**——C++ `get_env<int>` 把 `"0"` 解析成 **0（关）**，Python
      `os.environ.get(...,0)`（`buffers/ep.py:309/535`）对非空 `"0"` 恒为 **真（开）**；
      ② **`EP_NUM_MAX_LOCAL_RANKS` 默认 16 + 显存副作用**——hybrid 下按「本节点 16 卡」预留 RDMA
      storage（`engram.py:61`），超显存时 `:64-65` **顺手设全局 `os.environ['NCCL_WIN_STRIDE']`**（对齐
      4 GiB）；③ `EP_SUPPRESS_NCCL_CHECK` 的三把尺子（`__init__.py:24` 抑制 / `:28-32` 重复 libnccl
      扫描 / `:33-41` `filecmp.cmp(shallow=False)` 逐字节比库；「**加载了才比**」——没加载 libnccl 时
      `filecmp.cmp(None,…)` 抛 `TypeError` 而非断言，故上游在 `import torch` 后才跑，实验需先
      `torch.cuda.nccl.version()` 强制加载才复现断言）。另拆 `EP_AVOID_RECORD_STREAM`（源码注释
      `ep.hpp:162` 点明**只对 V2 API 生效**，V1 仍 `record_stream`）。**与第 38 篇的对照表**（入口
      pybind vs getenv、单解析 vs 双解析、有无引导闸门、安装期固化）。H100 真机
      `harness/deepep_env_probe.cpp`（g++ 直接 include 真实 `system.hpp`）+ `harness/exp_deepep_envs.{sh,py,out.txt}`
      （复用真实生成的 `envs.py` 与真实 `check_nccl_so`）**38 PASS / 0 FAIL**：Part A 9 PASS
      （真实 `get_env` 五语义 + bool 抛异常 + 畸形哨兵复刻）；Part B 29 PASS（白名单 7 条、真实生成物
      与「已设不覆盖/未设补上」两例、抑制短路/未抑制断言、全库 16 个带引号 `EP_*`、11 条运行时
      开关全命中、跨语言真值、`NCCL_WIN_STRIDE` 三例）。配套博客
      `content/posts/mega-moe-117-deepep-env-surface`（索引页新增第 117 篇）。

### P108 增量（opt-in，补齐每个 DeepEP 测试跑之前都会执行的「进程 / 环境引导层」）

- [x] **P108-1** **DeepEP 的测试 / 进程引导层**（`deep_ep/utils/envs.py` 的 `init_dist:89-133`、
      `init_seed:40-51`、`get_local_seed:54-61`/`get_global_seed:64-71`、`dist_print:74-86`、
      `EnvOverride:22-37`；消费点 `deep_ep/buffers/engram.py:81-85`、`deep_ep/utils/refs.py:6/151`、
      `deep_ep/buffers/ep.py:18-21/639/813`、`tests/**/test_*.py`）。第 117 篇拆了 DeepEP 的 `EP_*`
      环境变量面、第 40 篇拆了 **DeepGEMM 的** `init_dist`（`deep_gemm/utils/dist.py`），但 DeepEP
      **自己的**这层「每个测试第一行 `init_dist(...)`、之后 `init_seed`/`dist_print`/`EnvOverride`」
      从没单独成篇。**纯 host / Python，架构无关，H100 真机可跑**。
      → `docs/118-deepep-process-bootstrap.md`（真实行号核对）：`init_dist` 三件 host 事——① 起组
      （`rank = node_rank*num_local_ranks + local_rank` `:122`、`world_size = num_nodes*num_local_ranks`
      `:121`、`tcp://MASTER_ADDR:MASTER_PORT` `:120`、`inspect.signature` 探测 `device_id` 兼容分支
      `:124-126`），② 全局副作用（`set_default_dtype(bfloat16)` `:128`、`set_default_device('cuda')`
      `:129`、`set_cuda_device(local_rank)` `:130`），③ 播种（`init_seed(seed)` `:132`）；返回**新建的
      子组** `dist.new_group(...)` `:133`（不是 `WORLD`，集合操作）。**三个非平凡点**：`setdefault` 把
      `NCCL_IB_SL`/`EP_OVERRIDE_RDMA_SL` 压到 1（注释 `:103-104`：测试用、生产用 SL 3，尊重用户
      预置值）；`init_seed` 的 **`local = global + rank`**（`:48`）是「全局可复现、rank 间互异」的根，
      `get_global_seed` 被 combine 参考实现 `refs.py:148-152` 的 `sin(float(seed))` 消费，而
      `get_local_seed` **全库无消费者（死 API）**；`dist_print` 即使 `once_in_node` 静音也**必走 `:86`
      的 barrier**（全 rank 都要调）、`_local_rank` 仅由 `init_dist:115` 写入（`:83` 裸断言）。
      `EnvOverride` 三元 `enable` 标志（`:25` 过滤）+ 退出精确回滚（原值写回、原无则 `pop`；`__exit__`
      吞异常签名 `:32`），唯一消费者 `engram.py:81-85`。H100 实测
      `harness/exp_deepep_process_bootstrap.{sh,out.txt}`（`sys.modules` 注入合成 `deep_ep._C`/`comm`
      绕过 `check_nccl_so`，**直接加载真实 `envs.py` 并真跑 `init_dist(0,1)` NCCL world_size=1**）
      **30 PASS / 0 FAIL**：`init_seed` local/global 与 torch 流逐位；`init_dist` 七项副作用/返回值/
      种子传播 + `setdefault` 不覆盖已设值；`EnvOverride` 九项（含异常回滚、engram 条件覆盖）；
       `dist_print` 四项（静音仍屏障）；`check_torch_deterministic` 四组合；`get_local_seed` 死 API。
       配套博客 `content/posts/mega-moe-118-deepep-process-bootstrap`（索引页新增第 118 篇）。

### P109 增量（opt-in，补齐 `envs.py` 里最后一个被点名却从未单独读过的守卫）

- [x] **P109-1** **DeepEP 的 NVLink 连接自检 `check_nvlink_connections` 与 H100 P2P 拓扑**
      （`deep_ep/utils/envs.py:146-181`；消费点 `deep_ep/buffers/{ep.py:325,engram.py:67}`；
      对照 `get_physical_domain_size:136-143` → `comm/handle.py:80-85` →
      `csrc/kernels/comm/context.cpp:52-63`）。第 73 篇拆了 SM/QP 自动选型、第 86 篇拆了 NCCL 通信域
      绑定、第 118 篇拆了进程引导层，但 `envs.py` 里只剩**最后一个被反复点名、却从没单独读过**的
      函数。它是 DeepEP 对「节点内 GPU 两两 NVLink 全互联」假设的**唯一运行时自检**（对称内存 /
      `num_nvlink_ranks` 域 / `nvlink_barrier` / Mega MoE L2 远端直写 combine 都建立在它之上）。
      **纯 host / Python + NVML + CUDA 运行时，架构无关，H100 真机可跑**。
      → `docs/119-deepep-nvlink-p2p-check.md`（真实行号核对）：**三段逻辑**——① **字符串门**
      `:156` `if 'PCIE' in torch.cuda.get_device_name()`（H100 SXM 名 `NVIDIA H100 80GB HBM3`
      **不含 `PCIE`** ⇒ H100 上是 **no-op**；脆弱点，源码 `TODO :155` 自认可检查不全）；② PCIe 下
      **硬上限** `assert group.size() <= 2` `:157`（A100 PCIE 只有两两配对 NVLink），**先于任何 P2P
      探测**的粗闸门；③ 否则 `pynvml.nvmlInit()` `:159-161` → `CUDA_VISIBLE_DEVICES` 把**逻辑→物理**
      设备号翻译 `:163-165`（`int(devices[torch.cuda.current_device()])`，最易错一行）→
      `dist.all_gather_object` 汇聚 `:166-167` → 对所有**无序对** `i<j` 查
      `nvmlDeviceGetP2PStatus(..., NVML_P2P_CAPS_INDEX_NVLINK) == NVML_P2P_STATUS_OK` `:169-178`
      → `nvmlShutdown()` `:180-181`。**检查的是离散 NVLink 能力状态、不是带宽**；与权威域大小
      （NCCL 的 `ncclTeamLsa.nRanks`）分工是「**NCCL 定域、NVML 自检**」。两个消费点在 buffer 构造：
      `ep.py:325`（EPBuffer）、`engram.py:67`（构造 shared handle 前）。**H100 真机 16 PASS / 0 FAIL**
      （`harness/exp_deepep_nvlink_p2p.{sh,out.txt}` 直接加载真实 `envs.py` + `src/nvlink_p2p_matrix.cu`）：
      **A** 真机调用返回 `None`（no-op），伪装 `'… PCIE'` + group 8 → 命中 `:157`；**B** NVML 8×8 的
      `NVLINK`/`WRITE`/`ATOMICS` 三能力 **56/56 非对角对全 OK**（满互联）；**C**
      `torch.cuda.can_device_access_peer` **56/56** 且与 NVML 状态矩阵 **0 处不一致**（但 CUDA 那个
      布尔**不区分 NVLink/PCIe**，故必须借 NVML）；**D** 模拟 PCIe 分支真机跑全三条路径（no-op /
      `:157` / `:177` 断言 + all_gather），用**假 pynvml** 让 `(0,1)` 非 OK 触发
      `GPU 0 and GPU 1 are not connected via NVLink`（断言在 `nvmlShutdown` 之前抛出、异常路径下
      NVML 不显式关）；`CUDA_VISIBLE_DEVICES` 解析 4 边界全对；**E** `cudaMemcpyPeerAsync` 8×8 有向
      带宽：56 对全 **> 212 GB/s**（远高于 PCIe Gen5 ~64 GB/s，坐实每对走 NVLink、同/跨 NUMA 无差异），
      唯一异常是**目的列 `dst0`**（写到 GPU0 仅 ~213，写其他 ~373）——不是拓扑而是**该 GPU 被共享任务
      占用**，坐实「带宽探测必须与 NVML 能力探测分开」。附 **pynvml 绑定坑**：`NVML_P2P_CAPS_INDEX_READ`
      在本机版本被导出成 tuple `(0,)`、直接调用抛 `ctypes.ArgumentError`（DeepEP 用 `NVLINK` int
      不受影响）。附 vLLM 同款实现链接（源码注释 `:170`）。配套博客
       `content/posts/mega-moe-119-deepep-nvlink-p2p-check`（索引页新增第 119 篇）。

### P110 增量（opt-in，补齐四个调度器里最后一个、也是 Mega MoE 主 kernel 自己的任务调度器）

- [x] **P110-1** **DeepGEMM 的 Mega MoE 任务调度器 `scheduler/mega_moe.cuh`（481 行）**：
      本专题已单独拆过另外三个调度器——`scheduler/gemm.cuh`（第 108 篇，稠密/grouped/MoE 反向 GEMM
      共用）、`scheduler/mega_gate.cuh` + `scheduler/mega_mhc.cuh`（第 92 篇）。唯独 Mega MoE 主 kernel
      自己的任务调度器被 100+ 篇以行号引用了 15+ 次、却只在第 02 篇 §4 有一节概述。它是整条
      **dispatch + linear1 + SwiGLU + linear2 + combine** 流水线**把工作切成 task、并发给 5 类 warp 分身**
      的唯一入口，也是 host 侧 ring 容量公式 `get_num_max_live_pool_blocks` 的唯一定义处。
      **纯整数/布局运算，不碰 UMMA/TMA/tcgen05，H100(sm90) 可直接 include 真实头文件跑真实设备代码**
      （唯一 SM100 专属是 `publish_task` 的 `ptx::st_async_cluster` + cluster barrier，仅分析）。
      → `docs/120-mega-moe-scheduler.md`（真实行号核对）：**四个非平凡点**——① **行容量公式**：
      `get_num_l1_warmup_waves:17-45`（L1 预热波的两个下界：第一个 L2 波触及多 M block +
      `(ntb-1)*(l1n-l2n)` 交错残留，避免 L1→L2 死锁）+ `get_num_max_live_pool_blocks:48-80`
      （预热后活跃块 + 前沿增长 + 波余量，min 到 `ntb`）；host 用它做 ring 分配
      （`csrc/apis/mega_moe.hpp:55-63`：对 `kCandidateBlockM` 取 `max(live*block_m)` 后对齐 1920），
      是**host 与 device 之间唯一的几何契约**；② **`L2KBlockDependency:142-166`**：一位一 L1 N 块的
      位图，`arrive` 用 `ptx::red_xor_rel`（`ld_st.cuh:328-330`）**翻转**不累加、`wait` 用
      `(ptx::ld_acq_gpu(mask) ^ expected_mask)` 自旋（`expected_mask = generation&1 ? 0 : kFullMask`），
      把「L1→L2 细粒度依赖」与 **ring 世代奇偶**绑死——依赖正确性与 ring 容量强耦合（源码 `:140-141`
      注释点明）；消费者 `impls/sm100_fp8_fp4_mega_moe.cuh:723-726`、生产者（L1 epilogue 落盘后）
      `:1246`；③ **`create_task:309-342`** 不用任何 host 预建的 `grouped_layout` 表——每 lane 缓存名下
      专家 token 数（`fetch_expert_recv_count:286-307`，自旋到 `value>>32 == kNumSMs*kNumRanks` 即
      dispatch 完成）、`math::warp_inclusive_sum` 现算前缀和、`__ballot_sync` + `ptx::exchange` 在 warp
      内求出 `(local_expert_idx, owner_m_block_idx, valid_m)`；`get_pool_block_offset:272-280` 的
      `__reduce_add_sync` 是池块前缀和；④ **locality-domain 分域 + 工作窃取**（`get_next_task_idx:344-370`）：
      `kNumLocalClusters = kNumClusters/kNumLocalityDomains`、本域队列空了才偷邻域、门槛
      `kNumMinStealWaves=2`（`:225`/`:368-369`，本波任务 ≥2 个全局波才偷，避免跨 die 延迟）。
      **发布/消费顺序契约**：生产 `wait empty → get_next_task() → publish_task`（`:457-465` 注释：
      `get_next_task()` 会推进全局计数、不能早于消费者释放槽），消费端**只有 epilogue release**
      （`impls:982-990`，因为只有它知道 UMMA `tmem_full` 何时到），故 `task_info_empty_barriers` 的
      init 计数是 `2*kNumEpilogueThreads`（kernel `:276-278`，`kNumScheduleStages=2`）。
      **H100(sm90a) 真机 34 PASS / 0 FAIL**（`harness/exp_mega_moe_scheduler.{sh,out.txt}` +
      `src/mega_moe_scheduler.cu`，同一 `.cu` 兼编 `sm_90a` 运行 / `sm_100a` 仅编译覆盖；源 sha256
      `1bf76529…`）：A 行容量公式 vs 独立复刻 10 形状全对；A2 复刻 host ring 分配（DeepSeek 量级
      `block_m=240`、`ring=119280`→对齐 1920 得 **120960**）；B `TaskInfo` ABI（`sizeof=32`/`alignof=16`/
      `get_umma_aligned_valid_m(100)=112`）；C `L2KBlockDependency` 的 XOR 翻转 + 世代奇偶（`kNumL1BlockNs=32`、
      perK=2）；D/E `create_task` 8 专家 12 池块**逐块 0 不符**、`get_num_tokens` 0 不符；
      F 分域枚举双射（两域并集覆盖 `m×n`）+ 窃取阈值。主 kernel UMMA/TMA 与 cluster 写仅分析。
      配套博客 `content/posts/mega-moe-120-mega-moe-scheduler`（索引页新增第 120 篇；并修掉索引页
      重复的第 119 条）。

### P111 增量（opt-in，补齐四个 Buffer 家族共享、却从没被单独读过的生命周期契约）

- [x] **P111-1** **DeepEP 四个 Buffer 家族共享的生命周期契约 `BufferBase`**
      （`deep_ep/buffers/base.py`（27 行）+ `csrc/buffers/base.hpp`（63 行）；
      四个派生家族 `csrc/buffers/{ep,engram,pp,bucket}.hpp` + `deep_ep/buffers/{ep,engram,pp,bucket}.py`；
      `deep_ep/__init__.py:63`）。前面 120 篇把 DeepEP 的通信原语、地基、测试/部署面都拆遍了，
      但四个 Buffer 家族都继承、**却从没被单独读过**的公共基类一直被当 boilerplate。
      **纯 host、架构无关，H100(sm90) 可跑真机实测**。
      → `docs/121-deepep-bufferbase-lifecycle.md`（真实行号核对，`base.py` sha256[:16]
      `17e6d15585068396` / `base.hpp` sha256[:16] `ba2c0cab18b449f4`）。**三个非平凡点**：
      ① **两条平行继承链**——Python 的 `BufferBase`（ABC）与 C++ 的 `deep_ep::BufferBase` 是两个
      独立类：`pybind11::class_<EPBuffer, BufferBase>`（`ep.hpp:1085`/`engram.hpp:308`/`pp.hpp:127`/
      `bucket.hpp:344`）把 C++ 继承暴露给 pybind，Python `EPBuffer(BufferBase)`（`ep.py:208` 等）
      继承 ABC 后另建 `self.runtime = _C.EPBuffer(...)`（`ep.py:339`），靠 `self.runtime` 引线相连
      ——「Python 管用户可见语义、C++ 管资源释放时序」；② **`destroy_on_destruction` 三分支**
      （`base.hpp:18-29`）——默认自动销毁 + 清 `main_context`（`:20-22`）、显式模式不销毁则打印
      泄漏告警（`:25-27`，`name` 分 `EP`/`Engram`/`PP`/**小写 `bucket`**）、`destroyed` 标志
      （`:14`）把重复销毁变硬错误；`destroy()` 是纯虚（`:35`）故只能由派生析构体（`ep.hpp:98-99`
      等）调用——模板方法；③ **两个跨语言分歧**——`~BufferBase() noexcept(false)`（`:34`）是刻意
      的（`destroy()` 会抛，默认 `noexcept(true)` 会 `std::terminate`）；二次 `destroy()` Python
      **幂等**（`base.py:25` 的 `if self.runtime is not None:`）而 C++ **抛异常**（`ep.hpp:103` 的
      `EP_HOST_ASSERT(not destroyed)`）。另：`main_context` 是 **property**（`base.py:11-14`）销毁后
      自动返回 `None`；`destroy` 是**带实现的 `@abstractmethod`**（`:21-27`），四家族各自覆盖又用
      `super().destroy()` 复用（`ep.py:365-371`/`pp.py:69-72`/`bucket.py:394-398`/`engram.py:106-110`
      清理字段各异）；DeepEP 全部测试都传 `explicitly_destroy=True`（`tests/ep/test_ep.py:538` 等）
      以取确定性释放。H100 实测 `harness/exp_deepep_bufferbase_lifecycle.{sh,py,out.txt}` +
      `src/bufferbase_lifecycle.cpp` **32 PASS / 0 FAIL**：Part 1 在临时目录搭「影子 include 树」
      （真实 `base.hpp` 逐字节拷入、最小 stub 顶掉 pybind11 与 `comm/api.hpp`）g++ 编译运行真实类体
      （A/B/C/D/E/F 覆盖三分支 + 泄漏告警恰一次 + `sizeof(BufferBase)=32/alignof=8`）；Part 2 按文件
      路径加载真实 `base.py`（抽象基类不可实例化 / property 转发 / 默认模式撞 `assert explicitly_destroy`
      且未销毁 runtime / 二次销毁幂等）；Part 3 源解析四个家族五要素齐全 + 顶层 re-export。
      配套博客 `content/posts/mega-moe-121-deepep-bufferbase-lifecycle`（索引页新增第 121 篇）。

### P112 增量（opt-in，补齐 DeepEP 两类非 all-to-all 原语一直缺席的「验收测试即正确性规范」）

- [x] **P112-1** **DeepEP 的两类非 all-to-all 通信原语（PP send/recv 与 Engram fetch）的
      验收测试即正确性规范**（`tests/pp/test_pp.py`（152 行）+ `tests/engram/test_engram.py`（155 行）；
      `deep_ep/buffers/pp.py`（72 行）/ `deep_ep/buffers/engram.py`（169 行）；
      `deep_ep/utils/math.py:31-41`；`deep_ep/__init__.py:55`）。前面 121 篇已把 DeepEP 每一族的
      **验收测试**陆续做成「测试即正确性规范」（EP 第 75/85、bucket 第 83/84/94、EPLB 第 100、
      barrier 第 101 篇），唯独**流水线并行 send/recv**（第 68 篇实现）与 **Engram 远程抓取**
      （第 69 篇实现）这两族的测试从没被单独读过。本篇补上——它们定义的**不是数值而是协议/索引语义**。
      **kernel 端到端仅分析**（`_C.PPBuffer` `pp.py:37`、`_C.EngramBuffer` `engram.py:87`、
      `check_nccl_so` `__init__.py:55`、`deep_ep._C` 本机不可导入）；测试侧纯 Python/张量逻辑
      H100(sm90) 真机可跑。
      → `docs/122-deepep-pp-engram-test-oracle.md`（真实行号核对，`test_pp.py` sha256[:16]
      `a4a4368bcc9f52fb`、`test_engram.py` `23d15e87d1ee9851`、`math.py` `ed0dfb4bf9893769`）。
      **PP 侧（协议）**：`generate_stress_ops:19-50` 用整数时间戳编排**在途背压下的无死锁收发序列**
      ——dst 被 `:27` 限死在环上邻居 ±1；对每个有向 pair，第 i 次 send 必须晚于第 (i−N) 次 recv
      （`:40-42` 的 `sorted_send[i]=max(sorted_send[i], sorted_recv[i-N]+1)`），使全局时间排序下每 pair
      在途上界恰为 `num_max_inflight_tensors=N`（`:43` 再保证 recv 晚于 send → 在途 ≤ N）；`:44`
      每 pair 造一张 `torch.randn` 张量、`:45-48` 分挂 src 的 send 与 dst 的 recv（同一次随机收发对）、
      `:49` 按 `(时间戳, i)` 排序。`:78-79` **两条 RNG（`torch`+`random`）都同步**才能靠 `:89`
      `torch.equal` 判「收到的正是对端发的」；`:91-92` 用 `torch.cuda._sleep` 把时间戳差翻成真实时序
      压力。profiling（`:97-130`）：`hide_rdma_latency × num_concurrent` 矩阵、`:111` 大张量冲 L2、
      `:115` sleep 掉估算 RDMA 往返、`:126-129` 带宽账目（send 恒 2×、recv hide 时 2×）。
      **Engram 侧（索引语义）**：`:42-56` 每 rank 写本地 shard，FP8 时 `per_token_cast_to_fp8`
      （参考量化器第 114 篇）后把 SF 用 `all_gather_single` **复制到本地**（`:49-50`，注释 `:48`
      点明「任何 rank 都能本地抓到 entry 的 SF」）；`gen_indices:58-67` 把全局表按 `npt` 切成
      `table_size=(R*ne)//npt` 的**不相交槽**、每槽随机取（同 token 内严格递增；floor 除法**留空**
      `[8,4]` 行永不取到）；参考 `:80-83` all-gather 出全局表再 `global[idx]`，`:78` **正序/逆序两种
      层序**都验证（层间独立）、`:93`/`:95` `torch.equal`；profiling `:111-116` 按名拆
      `engram_fetch_impl`/`engram_fetch_wait_impl`、`:118` 总等待乘层数。**H100(sm90) 实测**
      `harness/exp_deepep_pp_engram_test.{py,sh,out.txt}` **72 PASS / 0 FAIL**：Part A 原样 ast 抽取
      `generate_stress_ops` 跑 18 组 `(R,N,seed)`——每 op 指向相邻 rank、每 pair 在途 ≤ N、背压抬高
      命中 **31 次**且修正后 send/recv 序列**仍单调**（FIFO 可配对）、**利齿负样本去掉修正后 31 处
      容量溢出被击穿**、同步 RNG 后 send/recv 张量**逐位一致**；Part B 复刻 oracle——索引形状/范围/
      槽不相交/唯一、`owner=idx//ne`/`row=idx%ne` 逐元素远程读 == gather-oracle **逐位**、正逆序一致、
      FP8 SF 复制通路逐位；Part C 定位阻塞（`import deep_ep` 非零退出落在 `deep_ep._C`、
       `dist.all_gather_single` 全仓无定义、由 DeepEP 补丁 torch 提供）。配套博客
       `content/posts/mega-moe-122-deepep-pp-engram-test-oracle`（索引页新增第 122 篇）。

### P113 增量（opt-in，补齐 DeepEP 反复引用却从没单独读过的 Python 缓存基座）

- [x] **P113-1** **DeepEP 的 Python 缓存策略：方法级 `weak_lru` 弱引用缓存 vs 模块/静态级
       `lru_cache`**（`deep_ep/utils/semantic.py`；消费点 `deep_ep/buffers/ep.py:430/845`；
       对照 `deep_ep/utils/find_pkgs.py:58`、`envs.py:193/200/207/237/260`、
       `buffers/engram.py:129-130`、`buffers/bucket.py:236`）。
       前面 122 篇把 DeepEP 的 kernel、host 编排、测试基座都拆遍了，但有 27 行 Python 被反复引用
       却从没单独讲过——`semantic.py` 的 `weak_lru`（`:9-26`），全仓库**唯一一个「方法级」缓存**。
       **H100 可跑，真机实测**（纯 Python、架构无关）。
       → `docs/123-deepep-python-caching.md`（真实行号核对）：`value_or:5-6`（**仅 `None` 回退**，
       `0`/`False`/`''` 合法，`ep.py:656-676` 消费）；`weak_lru:9-26` 逐行——`:17`
       `@functools.lru_cache(maxsize, typed)`、`:19` `func(_self(), …)` 解引用、`:21`
       `functools.wraps`、`:23` **key 是 `weakref.ref(self)` 而非 `self`**。**为什么**：
       `functools.lru_cache` 把调用参数整体当 key，实例方法第一个参数就是 `self` → 被强引用钉住、
       buffer 永不回收（`ep.py:430/845` 的 `get_theoretical_num_sms`/`lb_get_theoretical_num_sms`，
       后者默认 `subprocess.run(['nvidia-smi','nvlink','-s'])`，`envs.py:207-235`）。
       **三个真实陷阱**（实验逐条坐实）：① `weakref.ref` 相等性/哈希**委托 referent** →
       「值相等但不同」实例撞同一条缓存、第二个拿到第一个结果；② 死引用条目**不自动清**、只受
       `maxsize`（`maxsize=4` 跑 20 临时实例 `currsize==4`、`misses==20`）；③ `__slots__`
       （无 `__weakref__`）抛 `TypeError`。**全库普查**：**只有 key 含 `self` 的实例方法才需要
       `weak_lru`**——模块级 6 处（`find_pkgs.py:58`+`envs.py` 5）与静态方法 1 处
       （`engram.py:129-130` `@staticmethod`+`lru_cache`）都用标准库、全仓库**无「实例方法直接
       `functools.lru_cache`」**；对照 `BucketBuffer.get_theoretical_num_sms`（`bucket.py:236`，
       被 `:324/352/385` 每类集合通信调用）**刻意为不缓存**。**模块级缓存也咬人**：`envs.py:207`
       的 `get_nvlink_gbs` 用假输出实测（200→改环境仍 200→`cache_clear()` 后 20），坐实**吞掉环境
       变化**（与第 117 篇环境变量面互为脚注）。H100 实测
       `harness/exp_deepep_python_caching.{sh,py,out.txt}`（按路径加载真实
       `semantic/find_pkgs/envs.py`，`sys.modules` 注入合成 `deep_ep._C` 绕开 `check_nccl_so`）
       **39 PASS / 0 FAIL**：Part A `value_or` 5 条；Part B 普通 `lru_cache` 泄漏复现（实例
       `del+gc` 后仍存活、`currsize==1`）vs `weak_lru` 可回收；Part C 缓存键语义 11 条（含两个
       陷阱与 `typed`/`__slots__`）；Part D 普查 4 条；Part E 真实模块级缓存 10 条（含
       `find_nccl_root` 命中、`get_nvlink_gbs` 吞环境）；Part F 真实 `weak_lru` 复刻
       `lb_get_theoretical_num_sms` 同参只算一次。配套博客
        `content/posts/mega-moe-123-deepep-python-caching`（索引页新增第 123 篇）。

### P114 增量（opt-in，补齐「任何设备代码跑起来之前」的构建 / 打包 / 安装期地基）

- [x] **P114-1** **两库的构建 / 打包 / 安装期地基**（`deep_ep/utils/find_pkgs.py`、`DeepEP/setup.py`、
      `DeepGEMM/setup.py`、`DeepGEMM/scripts/generate_pyi.py`、`DeepEP/{build.sh,develop.sh,CMakeLists.txt}`）。
      前面 123 篇把 kernel、host 编排、测试基座、JIT 缓存键、环境变量面、生命周期契约都拆遍了，
      但有 78+140+222 行「**在任何一行设备代码跑起来之前就已定型**」的构建期代码从没单独讲过；它决定
      「wheel 链到哪份 NCCL」「哪些环境变量被烤进包」「`DG_SKIP_CUDA_BUILD` 跳过什么」「editable 为何
      找不到 `cute/cutlass`」，也正是本机 **DeepEP 运行期阻塞**的确切落点之一。**纯 host / Python、
      架构无关，H100 真机实测**。
      → `docs/124-build-packaging-foundation.md`（真实行号核对）：§1 `find_pkg_root:9-55` 四层语义
      （`:22-25` env 覆盖 `EP_NCCL_ROOT_DIR`>`NCCL_DIR`、命中不校验；`:27-38` sys.path 优先级，
      判据 `if priority > best_priority: continue` 严格大于 ⇒ 更小 priority 胜、**同 priority 枚举在后者
      覆盖前者**；`:31-33` 元数据子串匹配；`:40-46` lib_name 根目录裁剪；`:52-55` optional）；
      §2 `get_nccl_lib_name:72-78`（无版本名优先、否则 `sorted(libnccl.so.*)` 取首，真机 `libnccl.so.2`）
      + `-l:{name}`/rpath 钉住 wheel 目录；§3 持久化环境变量代码生成（DeepEP `setup.py:14-15` 7 条 /
      DeepGEMM `:148-155` 3 条）+ 运行时只补不覆盖（DeepEP `__init__.py:11-14` / DeepGEMM `:8-11`）；
      §4 `spec_from_file_location:17-20` 绕开 `import deep_ep`（顶部 `check_nccl_so`+`import _C`）——
      正是本机 `Invalid NCCL versions` 阻塞落点；§5 版本号 git 干净断言被 broad except **吞掉**
      ⇒ 脏工作区/无 git/权限拒绝都退成 `+local`（垃圾桶）；§6 `get_wheel_url:80-96` 文件名方案
      （真机 `deep_gemm-2.8.1+cu13-torch2.12-cxx11abi1-cp312-linux_x86_64.whl`）+ `CachedWheelsCommand:176-200`
      （`urllib` `timeout=1` 下载）+ `get_ext_modules:99-108` 的 `DG_SKIP_CUDA_BUILD` → `ext_modules=[]`
      （只跳 host `_C`，device kernel 全 JIT）；§7 `CustomBuildPy` 四个拷贝（`prepare_includes:157-173` 拷
      `cute/cutlass` = **editable 已知坑**、`generate_default_envs`、`generate_pyi_file:136-146`、
      `prepare_agent_files:128-134` 拷 `csrc`+`docs`），DeepEP `prepare_agent_files:54-61` 用
      `ignore_patterns('cmake-build-*')`；§8 `CMakeLists.txt:1` 自称 only-for-debugging、
      `build.sh`（bdist_wheel）vs `develop.sh`（build_ext + 软链）两条路。H100 实测
      `harness/exp_build_packaging.{py,sh,out.txt}`（按路径加载真实 `find_pkgs.py` + AST 抽取两个
      `setup.py`）：**54 PASS / 0 FAIL**——A 四层语义 14 项、B 库名取值 5 项、C envs 生成 8 项、
      D 版本号 10 项（含 D4 脏工作区被吞成 `+local`、D4b 去掉 broad except 即抛）、E wheel 名 4 项、
       F `DG_SKIP_CUDA_BUILD` 3 项、G spec 加载 4 项（含本机 `import deep_ep` 阻塞复现）、H 拷贝/过滤 6 项。
       配套博客 `content/posts/mega-moe-124-build-packaging-foundation`（索引页新增第 124 篇）。

### P115 增量（opt-in，P114 延伸：DeepGEMM 的 Python 类型桩生成器）

- [x] **P115-1** **Python 类型桩生成器 `scripts/generate_pyi.py`**（890 行纯 stdlib）：第 124 篇只在
       `setup.py:136-146` 一句话带过的 `generate_pyi_file`。这份 `.pyi` 是**用户唯一能看到的 Python
       接口契约**（IDE 补全 / 类型检查 / 文档站签名）。**纯 Python、架构无关，H100 真机实测**。
       → `docs/125-generate-pyi-stub-generator.md`（真实行号核对）：主流程五段 `:872-890`
       （`build_cpp_function_index:5-94` 真机 290 条 / `extract_m_def_statements:151-222` 真机 62 条 /
       `parse_mdef_and_attach_cpp_signatures:389-414` / `extract_cpp_signature_details:581-605` /
       `generate_pyi_file_content:828-869`）。**反直觉**：参数名/类型来自 **C++ 签名**，`py::arg` 只贡献
       个数 + 默认值。**`BracketTracker:97-148`**：四括号分别计数、`<>` 仅当 `()[]{}` 全 0 才算模板括号
       （`foo(a<b)` 安全）、`>` 须 `angle>0` 才减（`->`/`a>b` 安全、`>>` 双闭环）；`is_top_level` 四计数
       全 0 用于切顶层逗号。`parse_parameter_declaration:494-578` 从右往左顶层回扫 + 类型关键字黑名单。
       `cpp_type_to_python_type:608-696`（ORDER MATTERS）、`cpp_default_to_python_default:723-778`。
       **三处真实边界**：① `m.attr` 造的 **10 个别名**（含 `fp8_gemm_nt`/`fp4_gemm_nt`）**全无 stub**；
       ② `std::variant<int,...>` 被 `\bint\b` **过宽正则**误标 `int`；③ `default_args` 是**相对 `py::arg`
       的序位**、消费端按 **C++ 形参序位**取（当前巧合对齐）。另两瑕疵：顶层 `<<` 误计、行内尾注释
       `...; // m.def(` 仍被采集；死代码 `extract_cpp_signature_from_content:345-386`（无调用点）。
       H100 实测 `harness/exp_generate_pyi.{py,sh,out.txt}`（importlib 按路径加载**真实**脚本）：
       **45 PASS / 0 FAIL**——A 端到端两次生成逐位一致（sha `57a06883344d96b5`，612 行/62 函数）
       且与既有 `stubs/_C.pyi` 逐位一致；B/C/D 扫 m.def（含注释边界）、拆实参、建索引（首定义胜）；
       E `BracketTracker` 5 边界；F/G 18+11 映射；H 16 形参逐字；I 别名 10/10 缺；J 序位反例；K 死代码。
       配套博客 `content/posts/mega-moe-125-generate-pyi-stub-generator`（索引页新增第 125 篇）。

### P116 增量（opt-in，补齐所有 kernel 共用的「错误 / 断言层」）

- [x] **P116-1** **两库共用的错误 / 断言层**（`DeepEP/deep_ep/include/deep_ep/common/exception.cuh`、
      `DeepGEMM/deep_gemm/include/deep_gemm/common/exception.cuh` + `csrc/utils/exception.hpp` +
      `csrc/utils/compatibility.hpp` + `common/compile.cuh`）：前面 125 篇把每个 kernel / host 入口 /
      调度器 / 布局都拆过，但「每一行 kernel 发射前/后都要踩」的断言与错误检查宏一直当空气。
      → `docs/126-deepgemm-deepep-exception-layer.md`（真实行号核对）：两个异常类逐行对称
      （`EPException:11-23` / `DGException:13-24`，都只继承 `std::exception`、`what()` 返回拼串，
      **只差 `exception` vs `error` 一词**）；**五个差异** ① 单文件 vs host/device 拆两文件
      （DeepEP 94 行；DeepGEMM device 43 + host 67）② DeepEP 独有 `CUDA_DRIVER_CHECK`（走
      `lazy_cuGetError*`，须在 `namespace deep_ep` 内用）/`NCCL_CHECK`/`EPExceptionWithLineInfo`
      （仅 2 处，`ep.hpp:661`），DeepGEMM 独有 `DG_CUBLASLT_CHECK`/`DG_TRAP_ONLY_DEVICE_ASSERT`
      ③ **分派键** `EP_UNIFIED_ASSERT`=`__CUDA_ARCH__`（精确）vs `DG_UNIFIED_ASSERT`=
      `DG_IN_CUDA_COMPILATION`（`compile.cuh:5-7`，nvcc 全程定义）→ **`.cu` host 段用
      `DG_UNIFIED_ASSERT` 展开成 `asm("trap;")`、汇编报 `no such instruction: trap`**（真机负向
      编译坐实）；牵连 `layout/mega_moe.cuh:119-125` 的 `DG_HOST_ASSERT` 顺序敏感 ④ 两库都无
      `py::register_exception` → 统一翻成 Python `RuntimeError`、消息保留文件:行号（真机编译
      真实 pybind11 模块验证 5 宏）⑤ `*_STATIC_ASSERT` 在 DeepGEMM 两头 `#ifndef` 重复定义。
      设备五个断言宏各 trap（独立进程 `cudaDeviceSynchronize` 全 `cudaErrorLaunchFailure`）。
      **纯 host C++、架构无关**，H100(sm90) 真机 **36 PASS / 0 FAIL**
      （`harness/exp_ep_dg_exception.{sh,py,out.txt}` + `src/exception_layer.cu` +
      `src/exception_bridge.cpp`）。配套博客 `content/posts/mega-moe-126-exception-layer`
      （索引页新增第 126 篇）。

### P117 增量（opt-in，补齐第 4 篇只覆盖 softmax 那一路、第 5 篇只做前向的「路由反向」）

- [x] **P117-1** **Mega Gate 路由的反向**（`csrc/apis/mega_gate.hpp` /
      `deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh` /
      `deep_gemm/include/deep_gemm/epilogue/sm100_mega_gate.cuh`）：第 4 篇（P4-1）的 router 反向是
      **softmax + 硬 top-k** 那一路，第 5/13 篇（P5-1）把 **Mega Gate** 前向拆透了，但它是
      `sqrt(softplus(·))` 打分、**带 bias 排名、不带 bias 出权重**的另一套语义，且**只有前向、没有反向**
      （DeepGEMM 不实现 gate 反向，主 kernel SM100 专属）。本篇补上这套打分的梯度推导并真机对拍。
      → `docs/127-mega-gate-backward.md`（真实行号核对）：前向三件事——① 打分 `s=ssp(z)=sqrt(softplus(z))`
      带**阈值 20 分支**（`epilogue:19-24`，`z>20` 直接 `sqrt(z)`）；② **排名用 `score+bias`、出权重用不带
      bias 的 score**（`mega_gate.hpp:56-57` 注释 + `impls:397-398`），无 bias 时因 `ssp` 单调故**按 z 选 ≡
      按 s 选**（`impls:420-423`）；③ 权重 `w_i=g·s_i/(Σ_j s_j + 1e-20)`（`epilogue:210-213`，`:212` 的
      `+1e-20f`；共享专家槽恒 1.0、索引固定 `:194-195`）。反向三处非平凡：(1) 硬 top-k 是 straight-through
      → **bias 对梯度路径完全透明**（只改「谁入选」）；(2) **归一化 Jacobian**
      `ds_i = g(dy_i·S − Σ_j dy_j s_j)/S²`——与第 4 篇式 (6) 结构同构（分母换成 score 和、修正项换成 `Σ dy·s`）；
      (3) **分段导数** `ssp'`：`z>20` 为 `1/(2√z)`、否则 `sigmoid(z)/(2√softplus(z))`，`z=20` 处连续。合起来
      `dz_i=ds_i·ssp'(z_i)`（仅选中槽）、`dW_g=xᵀ dz`、`dx=dz·W_gᵀ`。**H100(sm90) 真机**
      （`harness/exp_mega_gate_backward.{sh,py,out.txt}` + `src/mega_gate_backward.cu`）：A 前向/解析导数
      vs double `2.52e-7`/`6.43e-8`、阈值探针 19.999/20/20.001 → 0.111806/0.111803/0.111801 **连续**；
      B 无 bias「按 z 选 vs 按 s 选」**0/4096 不符**、有 bias **4096/4096 不符**；C 归一化 Jacobian vs
      有限差分 `5.1e-11`；D 整 token gate 反向（T=4096,H=64,E=128,topk=6）device(fp32) vs double，
      no-bias/with-bias 的 `dz/dW_g/dx` relerr 均 **~1.1e-7**；E `eps=1e-20` 影响 **0**；F 手写反向 vs
      **torch autograd**（float64，6 组形状含/不含 bias）全部对齐到 **~2e-16** + 中心差分 `3.5e-9`。
      边界：Mega Gate SM100 专属且只有前向，本篇的「反向」按它的语义推导；数值语义架构无关故可跑，
      主 kernel UMMA/tcgen05/排序网络**仅分析**。配套博客 `content/posts/mega-moe-127-mega-gate-backward`
       （索引页新增第 127 篇）。

### P118 增量（opt-in，补齐 P78/P88 只一句话带过的 profiling 最后一公里）

- [x] **P118-1** **NCU PM-sampling 离线绘图工具 `scripts/quick_plot_pm.py`**（448 行纯 host/Python）：
      第 78 篇（P78-1）扫官方 profiling 基础设施时只带过它、第 88 篇（P88-1）拆了
      `run_ncu_mega_moe.sh` 的 NCU 编排，但「把 `.ncu-rep` 变成一眼能读的时间序列图」这最后一公里
      从没单独读过。它是内核内 `common/probe.cuh`（第 55 篇，per-warp `%clock64` 时间线）
      **互补的另一套观测**：PM sampling 看整段 kernel 执行期间 SM/L1/L2/DRAM/NVLink 的**实时**
      利用率起伏——对通信-计算重叠的 Mega MoE 尤为关键。→ `docs/128-quick-plot-pm.md`（真实行号核对）：
      五个小结构 `:21-74`（`MetricSpec` 的展示名/无后缀基础名/`kind`/`category`/`aliases`、
      `ResolvedMetricSpec`、`MetricSeries`、`CATEGORY_ORDER`、`KIND_SUFFIXES` 的 9 种后缀候选，
      `pct` 带两个回退）；45 条 curated 指标 `:77-125`（类别 Overview5/SM12/L1 8/L2 8/DRAM4/互连 8），
      metric 名三形态（`.TriageCompute.` 中缀全名 / 裸名 / 分组域前缀），`_metric_candidates:156-161`
      双候选（完整名 + `split(...,1)` 去前缀）**为跨代 chip 而设**；host 管线——
      `_run_csv_command:126-132` 的「非零且 stdout 空才 None」、`_query_available_metrics:134-153`
      只留含 `__` 的行、`resolve_metric_specs:164-181` 只查 chip 能力无需报告、
      `_parse_metric_values:184-198` 无正则拆三格式、`_probe_metric_series:201-224` 三硬约定
      （固定三行 / 目标列写死 `idx==11` / 取第一条数据行）、`collect_metric_series:227-248`
      后缀回退 + `len(values)>1` 闸门；`plot_pm:274-417`（`GridSpec` 竖向叠图共享 x 轴、按
      `CATEGORY_ORDER` 排序分组 + 类别色带、`unit=="%"` 固定 `ylim=(0,100)`、`_format_value` 的
      T/G/M/K 分档且 0 特判、输出 `.pm_sampling.png`）。H100(sm90) 真机 `harness/exp_quick_plot_pm.{sh,py,out.txt}`：
      importlib 加载真实脚本、纯 host 架构无关，**73 PASS / 0 FAIL**；`ncu 2026.1.1.0 --chip gh100`
      查得 **3728** 个 metric，`resolve_metric_specs('gh100')` = **resolved 34 / missing 11**，
      缺失 11 条逐条确认候选查无（**SM TMEM Pipe Throughput** `sm__mem_tensor_cycles_active_realtime`
      等）——坐实这份 curated 清单**面向目标卡 realtime 域**、在 H100 上仍可用 34 条；
      合成 4 序列渲染 55 KB PNG、空输入早退。**漂移**：注释 `:66` 引用的 `scripts/ncu-metrics.txt`
      在仓库缺失；`idx==11` 依赖 NCU CSV 列布局（改动会静默变 None）。
      配套博客 `content/posts/mega-moe-128-quick-plot-pm`（索引页新增第 128 篇）。

### P119 增量（opt-in，补齐 P124 只一句话带过的 DeepEP CMake 调试构建链）

- [x] **P119-1** **DeepEP 的 CMake 调试构建链**（`CMakeLists.txt` / `csrc/CMakeLists.txt` /
      `csrc/kernels/CMakeLists.txt` / `csrc/kernels/{comm,driver}/CMakeLists.txt` /
      `csrc/indexing/main.cu` / `setup.py:54-61` / `.gitignore`）与 **DeepGEMM 对偶**。
      第 124 篇（P114-1）拆「构建 / 打包 / 安装期地基」时只一句话带过「`CMakeLists.txt` 首行自称
      `only for debugging`、不参与 `pip install`」，第 78 篇又顺带提过 DeepGEMM 的 `csrc/indexing/main.cu`
      与 CMake；但 DeepEP 的 CMake 调试链本身从没被单独读过。**一套模板两个产物**：`setup.py`
      （`CUDAExtension`，只编三个 host TU）与 CMake。→ `docs/129-deepep-cmake-debug-build.md`
      （真实行号核对）：顶层 `CMakeLists.txt:1` 自述、`:6-13` 诊断 flag（`-DENABLE_FAST_DEBUG`
      全仓无 `#ifdef` 消费者、`--register-usage-level=10,--warn-on-local-memory-usage`、`CUDA_SEPARABLE_
      COMPILATION`、`-lineinfo`）、`:25-27` 的 `if (NOT NCCL_ROOT_DIR)` **判 CMake 变量**（传环境变量无效，
      必须 `-DNCCL_ROOT_DIR=`）、`:34-36` include/link、`:39/42/43/46` 两个产物目标 + 索引 TU；
      目标图三层：`csrc/CMakeLists.txt:1` → `csrc/kernels/CMakeLists.txt:1-14` 的 `add_deep_ep_library`
      工厂（`POSITION_INDEPENDENT_CODE ON`）+ `comm/driver` 各一行 + **`RUNTIME_CUDA_LIBRARIES`
      `CACHE INTERNAL` 缓存变量**（顶层 `:43` 消费）；`csrc/indexing/main.cu`（34 行）是全仓唯一 `.cu`，
      一次 include 全部 `impls/**/*.cuh` 只为 IDE 索引。`cmake-build-*` 命名约定一体两面（`.gitignore:12`
      + `setup.py:60` 的 `shutil.ignore_patterns`）。**H100(sm90) 真机**（`harness/exp_deepep_cmake_build.{sh,py,
      out.txt}`，纯构建 / 文件解析、架构无关）：configure **rc=0**；`runtime_driver`/`runtime_comm`/`_C`
      三目标全 OK（`libruntime_driver.a` **32794 B**、`libruntime_comm.a` **627820 B**、
      `_C.cpython-312-x86_64-linux-gnu.so` **1178816 B**、`nm -D` 见 `PyInit__C`）；默认 `all` 因
      `deep_ep_indexing_cuda` **rc=2** 失败——上游真实编译缺口 `bucket/reduce_scatter/rdma.cuh:619/623`
      （`longlong4_t`→`const float4`，`dst_ptr` 在 `:548` 按 `longlong4_t` 推进、`:563` 的 `values[k]` 是
      `longlong4_t`、累加却走 `:590` 的 `reinterpret_cast<float2*>`）；nvcc 诊断 flag **只作用于索引 TU**、
      三个 host `.cpp` 由 g++ 编（真机抓两条命令行坐实）。**DeepGEMM 对偶**（`CMakeLists.txt` sha256[:16]
      `8f05c0443af4daf5`）：同款 flag / `_C` / 索引 TU，但**无 NCCL、无 runtime 静态库**（device kernel
      全走 JIT）。**与本机「阻塞」的关系**：`context.cpp` 编译通过 → 问题**不在构建**、而在运行期
      加载哪份 soname（第 86/117 篇）。配套博客 `content/posts/mega-moe-129-deepep-cmake-debug-build`
       （索引页新增第 129 篇）。

### P120 增量（opt-in，补齐第 5 篇只做前向、第 117 篇只覆盖 Mega Gate 的「mHC 反向」）

- [x] **P120-1** **Mega mHC（HyperConnection）变换的反向**（`tests/test_mega_mhc.py:123-210` /
      `csrc/apis/mega_mhc.hpp` / `layout/mega_mhc.cuh` / `impls/sm100_mega_mhc.cuh` /
      `epilogue/sm100_mega_mhc.cuh`）。第 5 篇（P5-2）拆了 mHC **前向**、第 4 篇处理 **MoE router**
      反向、第 127 篇补 **Mega Gate** 反向，但 **mHC 这条「残差改写 + 系数生成」通路的反向一直没有**——
      DeepGEMM 的 Mega mHC **只有前向**（主 kernel SM100 专属），而它正是 HyperConnection 真正改写
      残差流的地方。→ `docs/130-mega-mhc-backward.md`（真实行号核对）：**前向**（权威参考
      `test:123-210`）① 残差多流混合 `new_residual = post_mix⊙x + comb_res_mix@residual`、**bf16
      舍入是共享数值边界**（`impls:486-497`，`NOTES` 在 `:496`、`__float22bfloat162_rn` 在 `:497`）；
      ② 投影 `z@fnᵀ` 乘 `g=rsqrt(Σz²/(hc·H)+eps)`（平方和 `impls:501`、尺度 `epilogue:43`）再分段
      `scales+bases`、sigmoid（`epilogue:53-56`）；③ Sinkhorn 交替行/列归一化（`epilogue:63-86`，
      softmax `:66-69`、`get_col_sum` 两次 `shfl_xor` `:71-76`、iters `:80-83`）；④ RMSNorm
      （**均方 over H，与 ②的 `hc*H` 平方和是两个量**，`epilogue:188-244`、shifted `:323-365`）；
      `kIsShifted` 只改「用哪套 pre、是否输出 pre」（`mega_mhc.hpp:78,155-157`）。**反向三处非平凡**：
      (1) **Sinkhorn 迭代 VJP**——不是一次归一化的反而是整条序列逆放，每步「除以和」同时改分子/分母
      （`dX=dY/(S+eps)−Σ(dY·X)/(S+eps)²`）；(2) **`g` 的耦合项 `dq`**——24 个投影输出被同一 `q`
      拉起，漏掉只错「经 `q` 的残差/输入梯度」；(3) 两处 bf16 边界 straight-through + 两条易漏回流
      （投影路 `dz` 要回流到 `s`）。与第 127 篇对照：top-k 权重和 / Sinkhorn 行·列 / RMSNorm 共享同一类
      「跨槽·跨维耦合」Jacobian。**H100(sm90) 真机**（`harness/exp_mega_mhc_backward.{sh,py,out.txt}` +
      `src/mega_mhc_backward.cu`）：A 手写 VJP vs **torch autograd**（float64，6 组形状 × 9~10 个张量
      梯度，normal/shifted）**最坏 relerr 1.01e-15**、Sinkhorn 反向 vs autograd `2e-16`、vs 中心差分
      `~1e-11`；B2 **负向对照**：含 `dq` 项 `dx relerr=1.09e-16`、**丢 `dq` 项 `6.18e-02`** 而
      `dfn/dgamma` 不变；C double 参考自身中心差分抽 7 个标量最差 `3.6e-11`；D 同一模板 device(fp32) vs
      double **relerr 8e-8~3.6e-7**、4×4 Sinkhorn VJP vs 有限差分 `~1e-11`。主 kernel UMMA/TMEM/
       SplitBarrier **仅分析**。配套博客 `content/posts/mega-moe-130-mega-mhc-backward`（索引页新增第 130 篇）。

### P121 增量（opt-in，把第 130 篇的 mHC 反向当成「待部署算子流水」做融合/roofline 账）

- [x] **P121-1** **Mega mHC 反向的 H100 融合账**（`src/mhc_backward_fused.cu`、
      `harness/exp_mhc_backward_fused.{py,sh,out.txt}`；语义源 `tests/test_mega_mhc.py:123-210`、
      `impls/sm100_mega_mhc.cuh`、`epilogue/sm100_mega_mhc.cuh`、`csrc/apis/mega_mhc.hpp`）。
      第 130 篇（P120-1）把 mHC 反向的**数学**补上了并证明「手写 VJP ≡ autograd」；本篇换**工程视角**：
      把这条反向拆成**六段算子流水**（K0 重算残差混合 → G0 `u0=A@fnᵀ` → K1 系数反向（含 Sinkhorn VJP）
      → G1a `dz=du0@fn` / G1b `dfn=du0ᵀ@A` → K2 残差混合反向）并量化「各段重在哪、算力还是带宽受限、
      融合能省多少、目标卡上怎么映射」。**关键观察**：`g` 耦合项 `dq·2A`（元素级）已折进 K1 的
      `dAbase`，只有 `du0@fn` 留作 GEMM——可融合的标量部分从 GEMM 里摘出去。→
      `docs/131-mhc-backward-fused-account.md`（真实行号核对）。**正确性**（`src/mhc_backward_fused.cu`，
      六段各一个真实 CUDA kernel + `k_reduce`）：device(fp32) 六段 vs host **单片 double 参考**
      （第 130 篇同款）逐项对拍，`T=256 H=64 it=2/5` worst relerr **8.84e-07/6.55e-07**（normal/shifted
      全覆盖，含 `dshift`）；真实形状（`H=7168`）手写 vs autograd 最坏 **2.36e-15**。**账（真实形状
      H=7168 hc=4 D=28672 Nout=24）**：整条反向 **AI≈6.5**、三段 GEMM **AI≈24**，均 ≪ H100 bf16 拐点
      **295** → **访存受限**；`T=16384` 整条反向访存下界 **3648.9 us**、三段 GEMM 计算下界仅 **68.4 us**
      （**相差 53×**）。**H100 实测**：① 元素级三段真实设备代码（`T=65536 H=64`）：K0 **1005.8 GB/s**、
      K2 **208.0 GB/s**、K1「内存部分」**153.2 GB/s**（`-sinkhorn` 计时归因开关），完整版掉到 **56.1 GB/s**
      ——差的 **~2.4 ms 全在 Sinkhorn**（本参考把 4×4 Sinkhorn 串行在单线程、其余线程空等；是
      `O(T·hc²·iters)` 固定成本、与 `D` 无关，真实 H 下被摊薄）；② **低基数原子是坑**：
      `dgamma/dscales/dbases` 的全局 atomicAdd（约 3.9M 次落到 3/24/64 个地址）换成「per-token 部分和 +
      并行归约」后 K1 从 **9.66 ms → 3.82 ms**（`K1r reduce` 自己仅 **152.7 us / 312.5 GB/s**）；
      ③ 真实形状：三段 GEMM 合计 **3037.3 us / ~22.3 TFLOPS**（`Nout=24` 太窄，有效带宽 ~0.93 TB/s，
      坐实访存受限；`T=4096` 738.4 us），而 torch **逐 op 元素级 62.3 ms**、**autograd 整条 119.5 ms**
      （`T=4096` 15.9/31.0 ms）——比访存下界慢 **~33×**，融合空间主要在元素级段。**目标卡仅分析**：
      三段窄 GEMM 复用 `sm100_bf16_gemm`/`sm100_fp8_fp4_gemm_1d1d`（`Nout=24` 需 padding，AI 小故
      `set_tc_util` 有意义）；三段元素级按前向 `sm100_mega_mhc.cuh` 的 per-token worker 对偶融进
      epilogue（同款 `shfl_xor` 做 Sinkhorn，把 G0 的 TMEM 累加器直接喂 K1 省一次 `u0` 往返）。
      主 kernel SM100 专属、仅分析。配套博客
      `content/posts/mega-moe-131-mhc-backward-fused-account`（索引页新增第 131 篇）。

### P122 增量（opt-in，把 P121 结尾的目标卡接线推演坐实 + per-token 融合 worker 真机化）

- [x] **P122-1** **Mega mHC 反向的目标卡接线 + per-token 融合 worker**（`src/mhc_backward_worker.cu`、
      `harness/exp_mhc_backward_worker.{py,sh,out.txt}`；源码 `impls/sm100_mega_mhc.cuh`、
      `epilogue/sm100_mega_mhc.cuh`）。第 131 篇（P121-1）把 mHC 反向拆成六段并留下一个**目标卡级**的
      接线问题（「三段元素级按前向 per-token worker 对偶融进 epilogue，把 G0 的 TMEM 累加器直喂 K1」）；
      本篇把它坐实。**前向机器（真实行号）**：残差多流混合 Post worker（`impls:482-508`，bf16 round
      `:497` 是共享数值边界、`q=ΣA²` 融合 `:501`）、系数投影**一条 TF32 TS UMMA**
      （`:235-238/273-275`，A 在 TMEM/B=`fn` 在 SMEM）、系数 epilogue（`epilogue:17-101`：split 归约得
      `hc_rms_scale :43`、仿射+sigmoid `:52-56`、Sinkhorn `:63-86` 的两次 `shfl_xor` 列和 `:71-76`、
      RMSNorm `run_normal_worker :141`/`:216-220`）。**反向六段 → 目标卡对偶表**：G0 复用前向同一条
      TF32 TS UMMA、K1 的 Sinkhorn VJP 复用同一 warp 布局逆放、K2 复用 Post worker 反向。**融合边界两条
      判据**：① 必须跨 token 归约（G1b `dfn` wgrad）→ 落全局；② 必须喂独立 GEMM（`du0`）→ 落全局；
      其余 `dAbase[T,D]` 与 G1a 的 `dz[T,D]` 可留片上。→ `docs/132-mhc-backward-sm100-wiring.md`。
      **H100 实测**（融合 worker = 一 block 一 token、K1+G1a+K2 全在片上、不写 `dAbase`/`dz`）：
      6 组形状（normal/shifted）device(fp32) vs host 单片 double 参考最坏 relerr **5.28e-07**，
      与六段 **5.92e-07** 同量级、同 worst 张量族；`T=65536 H=64` 元素级 K1+dz+K2 **6000.3 us →
      5702.8 us**（**~5%**，被串行 Sinkhorn 的 **3813.5 us** 压制）；真实形状 `T=16384 H=7168
      D=28672` 省掉 **7.516 GB** 中间张量往返（`4·T·D·4`；H100 1 GiB D2D copy 实测 **908.8 GB/s**
      → 投影 8271 us，共享环境保守下界）。**目标卡单 kernel 接线仅分析**（本机只给 H≤64 微基准；
       真实 `H=7168` 需前向 Post worker 那种 warp-pair 分段，落到 TMEM/mbarrier）。主 kernel SM100 专属、
       仅分析。配套博客 `content/posts/mega-moe-132-mhc-backward-sm100-wiring`（索引页新增第 132 篇）。

### P123 增量（opt-in，收口 P96/P97 留下的 87 条引用 near_miss defer）

- [x] **P123-1** **引用行号的 near_miss 定点复核：分裂区间合并 + 五分类 + 7 处修正**
      （`harness/check_refs.py` 增强 + 新 `harness/triage_refs.{py,sh}`）。第 106 篇（P96-1）给全专题
      约 1900 处 `file:line` 做体检、第 107 篇（P97-1）把它做成可复用 linter 并留下「10 条同名歧义 +
      87 条轻微漂移（near_miss）」的 defer；歧义已消掉，**near_miss 一直没逐条落账**。本篇收口。
      **① 修 linter 一个真缺口——分裂区间**：作者把区间写成两个反引号 token（`` `a.hpp:103`-`:105` ``）
      时，`CITE` 只认带文件名的 `file:line`、终点被整个丢掉、区间退化成单行，**偶尔把正确引用误报成
      near_miss**（`docs/01:116` 的 `ptx::deduplicate` 调用在 104、区间中段）。修法：新增 `CONT_RANGE`
      （`check_refs.py:56-71`）先合并 `file:l1`-`:l2`/`file:l1`-`l2` 为 `file:l1-l2`，`strip_doc` 扫描前
      过 `normalize_citations`（`:135`）、统计 `MERGED_RANGES`（`:62/69`）；当前语料合并 **13 条**。
      **② 五分类**（`triage_refs.{py,sh}`：`citations_on:54`/`is_decl_head:68`/`nearest_occurrence:75`/
      `triage:86`）：按「标识符最近出现行到被引区间的距离」分为 `in_range`/`same_line_other`/
      `adjacent_decl`（相邻行是装饰器/注释/模板头/签名头）/`owner_body`（引体点名 owner）/`far_other`/
      `absent`。**③ 闭环修 7 处**「引用少写定义/签名那一行」：`07:83`/`07:170`/`05:648`/`45:31`/
      `58:136`/`78:311`/`84:150`。→ `docs/133-refs-near-miss-triage.md`。
      **H100 实测**（纯文件解析、架构无关）：总引用 **2074**；near_miss **108 →（合并区间）106 →
      （7 处修正）99**，严格命中 **640 → 642 → 649（73.67%）**；**标识符级**五分类：`owner_body` 59 /
      `far_other` 25 / `absent` 22 / `same_line_other` 2 / `adjacent_decl` 1 均良性，待判的
      `adjacent_code` 仅 **6** → **约 93% 的 near_miss 是「引用没错、判据太粗」**；剩余 6 条逐条判读
      均为书写约定（引分支/引装饰器/引拒绝行）。`oob`/`unresolved`/`empty`/`ambiguous` 四类硬错误
      **恒为 0**、CI 退出码 0。配套博客 `content/posts/mega-moe-133-refs-near-miss-triage`
      （索引页新增第 133 篇）。

### P124 增量（opt-in，DeepEP 验收测试栈最后三份收口）

- [x] **P124-1** **DeepEP bucket / buffer 家族最后三份验收测试即正确性规范**
      （`tests/bucket/test_all_gather.py` 117 行 / `tests/bucket/test_session.py` 101 行 /
      `tests/buffer/test_allocation.py` 54 行；`deep_ep/buffers/bucket.py`、`deep_ep/buffers/allocator.py`、
      `deep_ep/utils/math.py:5/26`、`deep_ep/utils/testing.py:113/224`、
      `deep_ep/include/deep_ep/common/compiled.cuh:68-69`、`csrc/python_api.cpp:24/26`）。
      前面已把 DeepEP 七个子目录的验收测试逐份拆过（EP 第 75/85、bucket mixed/perf 第 84、EPLB
      第 90、barrier 第 91、pp/engram 第 112、gate 第 106），只剩这三份没读过。**bucket 设备 kernel
      （NCCL GIN / copy engine）本机阻塞 → 端到端仅分析**；测试侧全纯 PyTorch / NCCL 主机 API，
      H100 8 卡真机复现。→ `docs/134-deepep-bucket-test-oracle.md`（真实行号核对）：**① all-gather
      纯 NVLink copy-engine 语义**（`test_all_gather.py:18`；`dst.data_ptr()==gathered.data_ptr()` `:54`
      + `torch.equal` `:55`；`num_sms=0`（`bucket.py:290-291`）；「越界输入」仅 `num_rdma_ranks==1`
      `:58`；`:77` `del srcs` 测 `EventOverlap` 生命周期）。**② `test_session.py` 会话内 bump 分配**
      （缓冲区量 `:25-29`；`BucketSession.__enter__` 实例属性遮蔽 `bucket.py:29-43`、`_allocate`
      `:56-65` 64B bump+越界 `RuntimeError`；容差梯度 all_gather `rtol=0` `:62/75` vs
      reduce_scatter/all_reduce `1e-5` `:65/68`）。**③ `test_allocation.py` 多样式分配**
      （`BufferAllocator` meta→cuda `allocator.py:26-36`、`data_ptr`/`contexts`/多 group `:31-33`；
      `get_context_idx` 多 group 断言 `bucket.py:295-297`）。**H100 实测**
      `harness/exp_deepep_bucket_tests.{py,sh,out.txt}` **ALL PASS**：Part C `numel=524288`、session
      缓冲 `6291456` B、`plan.num_bytes=4194304`、第二分配偏移 2 MiB、bump `(0,64)`+越界抛错；
      Part D 多 group 断言 + `get_theoretical_num_sms(all_gather)==0`；Part A/B **8×H100 NCCL**——
      `all_gather_into_tensor` shard∈{8,8192,524288} **逐位**、all_gather 逐位、reduce_scatter 反序环
      重建 relerr **0.000e+00**（该机 8×NVLink 恰好逐位、测试仍保守用 1e-5）、all_reduce `sum*scale`
      匹配；坐实 `dist.all_gather_single` 在 stock torch 2.12 **不存在**。配套博客
       `content/posts/mega-moe-134-deepep-bucket-test-oracle`（索引页新增第 134 篇）。

### P125 增量（opt-in，收口 P122 结尾的目标卡单 kernel 接线：真实 H=7168 的 Post worker 网格）

- [x] **P125-1** **Mega mHC 反向的目标卡单 kernel 接线（下）：真实前向 Post worker 网格 + G0 累加器直喂 K1**
      （`deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh`；
      `deep_gemm/include/deep_gemm/layout/mega_mhc.cuh`；
      `deep_gemm/include/deep_gemm/common/{math,utils}.cuh`；对照 `epilogue/sm100_mega_mhc.cuh`）。
      第 132 篇（P122-1）的融合 worker 是**粗粒度**的（一 block 一 token、`threadIdx.x = h`），只答了
      「省多少中间张量」，没答「目标卡上 lane 到底谁拥有哪几行、归约在哪几条 lane 之间」。本篇把几何钉死。
      **主 kernel SM100 专属、仅分析**；架构无关的几何/归约/TMEM 列预算/元素级吞吐 H100(sm90) 真机实测。
      → `docs/135-mhc-backward-sm100-worker-wiring.md`（真实行号核对）：**几何**（`impls:30-31/56/108-113`：
      `kNumRowsPerWarp=16`/`kNumRowGroupsPerWarp=8`/`kNumLanesPerRowGroup=4`，lane `(w,l)` 拥有行
      `(w%4)·16+⌊l/4⌋` 与 `+8`）；**关键反直觉点**——两个 Post worker **按 UMMA 原子切、不按行切**
      （`impls:458-459`），**每个 worker 覆盖全部 64 行**，最后 WG3 相加（`impls:639-640`）；
      **逐行归约** `math::warp_reduce_sum<4>`（只 `xor 2`+`xor 1`，`common/math.cuh:217-218`，别名 `:225`）
      → `smem.hc_norm_sqr_sums[worker][row]`（`layout:138`）。**反向六段 → 前向机器对偶表**
      （K0/G1a/K2 复用 Post/Normal worker；G0 复用同一条 TF32 TS UMMA `impls:235-238/273-275`；
      G1b `dfn` 落全局；`dz[T,D]` 按 `BLOCK_K` 流式）。**G0 直喂 K1 的硬前提**：TMEM 列预算
      前向 `176→256` + 反向 G0 `24→32` = **288 < 512**（真实 `get_num_aligned_tmem_cols` `utils:43-50`）；
      K1 用与 Post 相同的 `first_row_idx`/`col_pair_idx` 读 `SM100_TMEM_LOAD_16dp256b2x`（前向 WG3 同款 `impls:602-606`）。
      H100 实测 `src/mhc_backward_segment.cu` + `harness/exp_mhc_backward_segment.{sh,py,out.txt}`
      （**直接 include 真实头文件**）：① 几何核对 0 错误；② 分段归约 vs host double **rel 4–6e-8**，
      **全 32-lane 归约 rel=2.47**（结构性错误）；③ TMEM 判定 ALL OK；④ 分段网格 vs 朴素吞吐
      **2.03×–10.93×**（H=512 **~860 GB/s**）；真实形状 `T=4096` 的 `u0` scratch 仅 **6.3 MB**、
      「直喂」省 12.6 MB——收益是**省 1024 次 global round-trip 的延迟**，K1+G1a **AI≈0.75** → 访存受限。
      配套博客 `content/posts/mega-moe-135-mhc-backward-worker-geometry`（索引页新增第 135 篇）。

### P126 增量（opt-in，补齐第 22 篇对 SM100 sparse MQA logits 主 kernel 只给的 40 行概览）

- [x] **P126-1** **SM100 sparse MQA logits 主 kernel 逐段精读**（`deep_gemm/include/deep_gemm/impls/
      sm100_sparse_mqa_logits.cuh`，688 行；`layout/sparse_mqa_logits.cuh`；
      `csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp`；`csrc/apis/attention.hpp:303-442`）。
      第 22 篇（P12-1）只给了 §4 约 40 行概览，而同一主 kernel 的 **dense 版**后来在第 113 篇（P103-1）做了
      完整逐段精读；本篇把稀疏版补齐到同深度。**主 kernel UMMA/tcgen05/UTCCP/2-SM TMA SM100 专属、
      仅分析**；架构无关的六段 H100(sm90) 直接 include 真实头文件跑真实设备代码。
      → `docs/136-sm100-sparse-mqa-logits-kernel.md`（真实行号核对）：**两个入口同一核心**——
      连续 `sm100_sparse_mqa_logits`（`:647-665`）与分页 `sm100_paged_sparse_mqa_logits`（`:667-686`）
      都调 `..._core_impl`（`:146-645`），唯一分叉是 `KVAccessor` 的
      `kSupportsContiguousTMA`/`kNeedsTailMask`（`:36-37/103-104`）。几何（`:157-166`）：`UMMA_M=128`、
      **`UMMA_N=BLOCK_Q*kNumHeads≤64`（与 dense 钉死 128 不同）**、`SPLIT_KV=nwg*128`、
      `split_kv=fp4?640:512`、`num_kv_stages=fp4?5:3`。**六类 warp**（`get_num_threads=(nwg+1+nwg/4)*128`）：
      math 0..nwg*4-1（`:551-641`）、Q/metadata 生产者（`:263-322`）、SF 转置（`:323-369`）、
      UMMA（`:370-415`）、KV copy（`:416-550`）、填充（`:642`）；fp8=768 线程/fp4=896。
      **六段架构无关逻辑**：A `get_swizzled_kv_chunk_idx`（`:28-32`，**gather 式 KV 走 `cp.async` 故需
      自定义 smem swizzle**：低 3 位与 bits[3,6) XOR，对合置换）；B `KVSplitHeader`/`KVBlockInfo`
      （`layout:31-72`，标记位 + 双 Q 槽 16+16 位打包）；C 连续 `get_num_valid_tokens`（`:96-98`，尾块裁剪）
      + 分页 `resolve_kv_block`（`:121-131`，`PAGE%SB==0` 保证块不跨页、SF 区在页内 KV 区之后）；
      D math warp 压缩列映射（`:588-590/627-633`，token 双射 + slot 解码 + `(base+slot)*SB+t` 双射）；
      E 资源账（`get_num_aligned_tmem_cols` raw fp8 376/fp4 428 → 512；`SharedStorage` 4 组
      fp8sb16 222208/fp4sb8 230912 `< smem_capacity 232448`）；F SF 4×32→32×4 转置（`:325-333`）。
      H100(sm90) 实测 `src/sparse_mqa_logits_kernel.cu` +
      `harness/exp_sparse_mqa_logits_kernel.{sh,out.txt}`（**直接 include 真实 layout 头**）**ALL PASS**：
      swizzle N=64/128 对合+双射+模型一致；打包 32 例 0 不符；尾块 `88→12/90→10/≥100→0`、分页 40 块 0 不符；
      压缩列 `[80,400)` 每列恰一次；线程 768/896、TMEM 8 档全对、SharedStorage 8 成员 offsetof 8/8；
      SF 转置 0/128。配套博客 `content/posts/mega-moe-136-sm100-sparse-mqa-logits-kernel`
       （索引页新增第 136 篇）。

### P127 增量（opt-in，补齐第 22 篇对 SM100 sparse MQA logits 的 metadata kernel 只给的 30 行概览）

- [x] **P127-1** **SM100 sparse MQA logits 的 metadata kernel 逐段精读**
      （`deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh`，475 行；
      `layout/sparse_mqa_logits.cuh`；`csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp:135-200`；
      `csrc/apis/attention.hpp:306-367`）。第 22 篇（P12-1）只给了 §3 约 30 行概览、第 136 篇（P126-1）
      把**主 kernel** 做了完整逐段精读；本篇把 metadata kernel 补到同深度。**关键结论：整个 metadata
      kernel 都架构无关**（只用 `ptx::` 整数/原子、`cp.async`、`math::cta_exclusive_sum`、`cute::` 下标，
      **不碰 UMMA/tcgen05/UTCCP/TMA**），故 H100(sm90) 直接 include 真实头文件、用真实 launch 参数
      **原样启动该 kernel**，与独立 host oracle 逐字段对拍。**主循环**：Q 块构造 + 有效块前缀
      （`:204-236`，分页 `indices` 线性扫回 request 头、偶数偏移配对、否则 `atomic_add(next_q_offset)`
      扔回全局游标无锁认领；`min(NB, ceil((kv_end-kv_begin)/SB))` + 对齐断言）；稀疏索引加载
      （`:246-271`：非对齐同步 `ld_evict_first` + 就地 `*SB+ks%SB`、对齐 `cp_async_cg<256>` +
      `cp_async_fence/wait`）。**归并路径分块**（`:273-291`）：输出按线程切、每线程**二分**求起点处
      消耗了多少 q0（`logical[1][q1-1] >= logical[0][q0]`），把顺序归并变成「二分定位 + 局部归并」；
      跨分区去重（`:293-298`）；打包（`:300-326`：每 16 位 `[present|slot15]`、`in_q0=in_q1⇔相等`、
      `consume_q1 = in_q1 and remaining>in_q0` 把重复项留给下一分区）。`cta_exclusive_sum` 压缩 +
      `atomic_add(num_kv_splits)` 按 Q 块一次预留连续 split、分页清零第二 token 的 `q_block_info`
      （`:328-341`）。`write_merged_kv_block`（`:359-415`）：物理块三态（分页 block_table / 非对齐
      =绝对起点 / 对齐=`logical*SB`）、`q*_slot_offset = slot - base` 只存相对偏移、split 首块写
      header、**只看最后一块**判 `is_contiguous`（TMA 快路）/`has_partial_tail`（裁剪慢路）；
      尾块 padding 成 `KVBlockInfo(0,0xFFFF,0xFFFF)`（`:431-436`）。**末 CTA 建调度**（`:443-472`）：
      `atomic_add_rel(num_finished_ctas)==gridDim.x` + `ld_acq` 取回全部写入；连续版
      `build_contiguous_schedule`（`:37-73`，均分 + 按 Q 块边界切 + `atomicMax` 波数）；分页版
      `build_paged_schedule`（`:124-161`，每 token 切至多 `kNumKVSplitsPerEntry=8` split 的有界条目）
      + `balance_wave_entries`（`:75-122`，**直方图排名排序 + 旋转**、两 wave 时第二 wave 反向使活跃
      SM 互补）。H100(sm90) 实测 `src/sparse_mqa_logits_metadata.cu` +
      `harness/exp_sparse_mqa_logits_metadata.{sh,out.txt}`（grid=1 消歧）**ALL PASS**：6 组样例
      （连续对齐 A1/A2/A3、非对齐 B1、分页 C1/C2）共 **446 个 KV 块 / 36 个 split，0 不符**——
      归并顺序、`base+offset` 反解、物理块三态、split 头全对；调度覆盖 `miss=dup=bad=0`，A3 连续
      16 split/4SM=2 wave 每 SM 恰 4、C2 分页 `balance_wave_entries` 轮转（wave0 `[1,1,4,4]`/wave1
      反向 `[4,1,0,0]`）。主 kernel 与 host `arch_major==10` 门控 SM100 专属、仅分析。
      配套博客 `content/posts/mega-moe-137-sm100-sparse-mqa-logits-metadata-scheduler`
      （索引页新增第 137 篇）。

### P128 增量（opt-in，补齐 `heuristics/` 目录里从没被单独读过的最底层公共工具）

- [x] **P128-1** **DeepGEMM 启发式共用的「元素位宽 / TMA swizzle 模式」公共选择器 +
      `GemmDesc` 的 dtype→MmaKind 派生管道 + Python 公开布局别名层**
      （`csrc/jit_kernels/heuristics/utils.hpp` 32 行、`config.hpp:35-61`、
      `deep_gemm/utils/layout.py` 13 行、`csrc/utils/math.hpp:11/13-27`）。第 28 篇把
      `get_best_config` 的决策骨架讲透并给 `config.hpp` 五结构体列了**字段表**，但没讲派生量
      **从哪来**；`heuristics/utils.hpp` 是全仓最短的启发式文件、从没被单独读过，却是**每个**
      SM90/SM100 kernel 的 K 块宽度与 TMA swizzle 的**唯一出处**。**① `get_num_element_bits`**
      （`:12-19`）把 `MmaKind`（`types.cuh:7-11`）翻成**逻辑位宽**（BF16 16 / MXFP8FP4 8 / MXF4 4）；
      `block_k = 128*8/bits`（`sm90.hpp:60`、`sm100.hpp:32`）背后是**不变量 `block_k*bits == 1024`
      =「K 块一行恰好 128 字节」**（TMA 最大 swizzle 即 128 B，故一份 kernel 通吃三种 MmaKind）。
      **② `get_swizzle_mode`**（`:21-30`）返回 `{128,64,32,16}` 里最大可整除 `block_size*elem_size`
      的档；注释点明「`>0`=交错」「**16=非 swizzle 仅交错**」→ 它**从不返回 0**，0 只在
      「FP32 输出不 swizzle」处显式写（`sm90.hpp:138-139`）。**③ 两个位宽不同源**：
      `get_num_element_bits` 用逻辑位、`get_swizzle_mode` 用存储字节 `c10::elementSize`
      （`kPackedFP4=torch::kInt8` ⇒ elemSize=1）；SM100 先 `/ get_smem_pack_factor()`（MXF4 的 ÷2，
      `sm100.hpp:176-184`）→ 三条 K-major 全落 128、**MN-major FP4=64**。**④ Mega MoE 是
      「结论内联」的平行实现**（`heuristics/mega_moe.hpp:80-82/117/214-216`）：自带
      `get_num_mma_elem_bytes`（BF16 2、其余 1、`block_k=128/elem`）并**硬编码 swizzle 128**
      （FP4 权重在 smem 展开 8-bit ⇒ A/B 内维恒 128 B），只有「宽度减半」才现算
      （`impls/sm100_fp8_fp4_mega_moe.hpp:69-76`，L1 输出 `128→64`）。**⑤ `GemmDesc` 派生管道**
      （`config.hpp:35-61`）：`is_mxf4_mma`/`get_mma_kind`（只看 a==kBFloat16）/
      `get_smem_pack_factor`（MXF4 才 2）+ `check_validity` 四类硬约束（BF16 两端 bf16；
      block-scaled 两端各 ∈{FP8,packedFP4}；**FP8-D 仅 Batched 非累积** `:57`；
      **`num_sms%2==0`** `:60`）+ `expected_*` 的 0=回退 + `ensure_zero_padding`。**⑥ Python 别名层**
      `utils/layout.py`：7 个 `_C` 布局函数 + `get_m_/get_k_alignment_for_contiguous_layout`
      **都指向同一 `get_mk_alignment_for_contiguous_layout`**（m/k 共用一把对齐旋钮）。
      H100(sm90) 真机（**直接 include 真实头文件 + torch**）**43 PASS / 0 FAIL**：
      `block_k*bits` 全 1024、`get_swizzle_mode` 在 **4096** 组上 0 不符且 `UNREACHABLE` 恰 3136 组、
      `check_validity` 10 例矩阵全对（负例打印 `config.hpp:51/57/60`）、三条 K-major swizzle 全 128 /
      MN-major FP4=64、`get_tma_aligned_size==align(x,16/elem)`、Python 段 `PY_OK`。
      至此 `csrc/jit_kernels/heuristics/` 目录（`config`/`common`/`utils`/`sm90`/`sm100`/`mega_moe`/
      `mega_gate`）**全部被专篇覆盖**。设备端用这些数字造 TMA/发 UMMA 仍 SM100/SM90 专属、仅分析。
       配套博客 `content/posts/mega-moe-138-mma-kind-swizzle-selector`（索引页新增第 138 篇）。

### P129 增量（opt-in，补齐训练侧一直缺席的「路由负载均衡」算子与反向）

- [x] **P129-1** **MoE 路由负载均衡三件套：aux load-balancing loss / router z-loss /
      aux-loss-free bias（DeepSeek-V3）及其反向**（`csrc/apis/mega_gate.hpp`、
      `impls/sm100_bf16_mega_gate.cuh`、`epilogue/sm100_mega_gate.cuh`、`tests/test_mega_gate.py`；
      公式参考 Switch 2101.03961、ST-MoE 2202.08906、DeepSeek-V2 2405.04434、DeepSeek-V3 2412.19437）。
      专题到第 138 篇已把前向、反向、Mega Gate/mHC 与三条门控反向逐个拆开，但**「路由的负载均衡」这一
      决定 MoE 能否收敛的训练侧算子一直没讲**：第 4 篇 router 反向、第 127 篇 Mega Gate 反向都把 `bias`
      当**给定常量**（stop-grad），而 DeepGEMM 的 Mega Gate 确实只**消费** `bias`（`mega_gate.hpp:41-43`）、
      **不产生**它。**① aux load-balancing loss**（$\mathcal{L}=\alpha E\sum_i f_iP_i$，$f_i$=top-k 硬计数、
      **必须 stop-grad**、$P_i$=平均门控概率可微）：梯度只走软概率
      $\partial\mathcal{L}/\partial z_{t,j}=\frac{\alpha E}{T}p_{t,j}(f_j-g_t)$；不变量 $\sum_i f_i=k$
      且均匀 $P$ 时 $\mathcal{L}=\alpha k$（与 $f$ 分布无关）。**② router z-loss**
      （$\mathcal{L}_z=\beta\overline{\mathrm{lse}^2}$，$\partial\mathcal{L}_z/\partial z=\frac{2\beta}{T}\mathrm{lse}\,p$）。
      **③ aux-loss-free bias**（DeepSeek-V3 `e_score_correction_bias`）：$b_i$ **只进排名、不进权重**、
      每步 $b_i\mathrel{+}=\gamma\operatorname{sign}(\text{target}-\text{load}_i)$、**零梯度**（控制回路）。
      **与 Mega Gate 对接（真实行号）**：`mega_gate.hpp:56-57`「rank on score+bias, weights are the
      unbiased scores」、`impls:283` `kHasScoringBias`、`impls:380-383/396-398`（排名加 bias、图像 token 换
      `image_bias`）、`impls:404`（在带 bias 分上选 top-k）、`epilogue:212-213`（权重用
      `selected_unbiased_score`）、`tests/test_mega_gate.py:130-139`（`ranking=scores+ranking_bias` /
      `reference_weights` 用无偏 `scores`）——**DeepGEMM 只做消费端，b 的产生/更新在框架训练循环里**。
      H100(sm90) 真机 `harness/exp_moe_load_balance.{py,sh,out.txt}`（纯 PyTorch float64、架构无关）
      **21 PASS / 0 FAIL**：Part A 解析梯度 vs autograd **2.20e-15** / 中心差分 7.97e-06 / 均匀 $P$ 时
      $L=\alpha k$；Part B z-loss vs autograd **4.95e-16**、放大 logits 尺度 L 2.54e-2→2.89e-1；
      Part C 无 bias 硬负载 max/min **28.04** → bias 800 步 **1.0178**（各专家占比偏差 <3.05e-4、
      过载专家 bias 更低方向正确、$\gamma$ 10× 则 40 步达 1.105）；Part D **核心结论**——aux-loss 梯度是
      **弱杠杆**（作用于软概率、对阶跃的硬 top-k 仅 8.50→5.44），bias 符号更新是**强杠杆**（直接改排名、
      800 步 28.04→1.03），**坐实 DeepSeek-V3 aux-loss-free 的工程动机**。本层纯 host/Python、架构无关。
      配套博客 `content/posts/mega-moe-139-moe-load-balance`（索引页新增第 139 篇）。

### P130 增量（opt-in，补齐 DeepEP 所有通信篇目脚下的 host 上下文对象）

- [x] **P130-1** **DeepEP 的 host 通信上下文 `comm::Context`**（`csrc/kernels/comm/context.cpp`；
      `csrc/kernels/comm/api.hpp:31-88`；`csrc/kernels/comm/symmetric.hpp`；
      `deep_ep/comm/handle.py`；`deep_ep/include/deep_ep/common/compiled.cuh:67-82`）。
      第 7/48/51/53/63/64/66/68/69/73/86/121 篇把 DeepEP 的 dispatch/combine、epilogue、
      屏障、对称内存、Buffer 家族都拆过，但它们**全部从一个已经存在的「上下文/窗口」讲起**——
      把「一个 PyTorch NCCL 通信组」变成「一块可对称寻址、可发起 GIN RDMA、可两级域编排的窗口」
      的那**一个 host 对象** `Context`（268 行）从没被单独读过。**端到端 NCCL 设备 API
      （`ncclDevCommCreate`/`ncclCommWindowRegister`）本机 2.29.7 不提供 → 仅分析**；
      但**全部 host 算术**（域分解、signal 预算、偏移布局、版本闸门判据、句柄缓存语义）
      架构无关，H100(sm90) 真机逐条实测。
      → `docs/140-deepep-comm-context.md`（真实行号核对）：`Context` 字段三组（域分解/NCCL 句柄/
      缓冲指针，`api.hpp:37-63`）+ `symmetric_memory` 是 `shared_ptr`（多 Context 共享一块内存）；
      host 引导 `get_local_unique_id/create_nccl_comm/destroy_nccl_comm`（`context.cpp:23-50`，
      comm 指针当 int64 返回）+ `handle.py:44-77` 三分支缓存（cache / 复用 torch `_comm_ptr`
      `managed=False` / 新建 `managed=True`）；三查询 `get_physical_domain_size`（`:52-57`）、
      `get_logical_domain_size`（`:59-63`，hybrid=物理域 / 单级=`(1,num_ranks)`）、
      `get_gin_min_stride`（`:65-73`）；**GIN 需求逐字段**（`:121-155`：`ginType=GDAKI`、
      `ginContextCount=num_allocated_qps`、`ginQueueDepth=1024`、**`ginSignalCount=num_ranks+4`**
      （=第 63 篇免复位屏障的槽来源）、三级连接 `CUSTOM_STRIDE/RAIL/FULL`、`ginTrafficClass` 三级
      优先级）；**NCCL 运行时版本闸门**（`:156-164`：`>=2.31` 走 `useRuntimeVersion` +
      `devCommRuntimeVersionSize`，`<2.31` 断言 `NCCL_VERSION_CODE==nccl_runtime_version`）——
      并**厘清本机「阻塞」是两个独立闸门**：本处（pip 头 2.32.3 走 runtime 分支、不要求相等）
      与更外层 `deep_ep/__init__.py:19-41` 的 `check_nccl_so` **二进制比对**（torch 载系统 2.29.7
      ≠ DeepEP 期望 pip 2.32.3）；域分解恒等式（`:166-183`，rank=`rdma*nvl+nvl_idx` 行主序）+
      对称窗口（`:195-205`，`ncclCommWindowRegister` **集合操作**免屏障、链式加法
      workspace→buffer→multimem、仅 owner 清零）+ RDMA storage 窗口（`:207-223`，每
      `gin_min_stride` 片一个 `NCCL_WIN_GIN_ONLY` 窗口 + 拷到 device）+ `get_sym_ptr` 偏移翻译
      （`:226-232`）+ `finalize()` 拆除逆序（`:244-266`）。H100 实测
      `harness/exp_deepep_context.{py,sh,out.txt}`（设备常量来自 `src/deepep_context_arith.cu`
      **直接 include 真实头文件**）：**29 PASS / 0 FAIL**——真实头文件常量 6/6（2MiB 分配对齐等）、
      `advance_ptr` 布局 8/8（storage windows/local）、域分解 **60 行逐字段 0 不符** + 乘积不变量、
      NCCL 版本闸门四组合（ctypes 读 system 2.29.7 / pip 2.32.3）+ `check_nccl_so` 两条二进制比对、
      句柄缓存三分支。设备侧 NCCL API 与端到端 dispatch/combine 仅分析。配套博客
      `content/posts/mega-moe-140-deepep-comm-context`（索引页新增第 140 篇）。

### P131 增量（opt-in，把「通信」真正放回 MoE 图：多卡 EP 端到端前向/反向）

- [x] **P131-1** **多卡 EP 端到端 MoE 前向/反向 vs 单卡稠密参考**（8×H100 真通信）。
      第 99 篇（P89-1）的前向数值复刻、第 111 篇（P101-1）的反向数值复刻都是**单卡**的——作者
      **明确把 dispatch/combine 的通信折算成单卡 gather/scatter**（第 99 篇 §2「跳过通信」）；
      第 95 篇（P85-1）虽跨 8×H100 跑了真实 `refs.dispatch`，但只验「两个原语互为转置」这条
      **线性代数恒等式**、**没把 EP 接进完整 MoE 图**、没量**参数梯度**。本篇补这个结构性空档：
      在 8×H100 上用真实 NCCL `all_to_all_single` 实现 expand 版 dispatch/combine
      （行空间与 DeepEP `do_expand=True` 一致），把 `dispatch→L1→SwiGLU→L2→combine` 与它的
      autograd 反向逐张量与**单卡稠密参考**对拍。**反向的关键事实**（第 4/14/95/104 篇的
      「反向无 mega-kernel」）被**写成代码**：`all_to_all_single` 是 in-place、**不挂 autograd**
      （实测 `out.requires_grad=False`），故手写 `_A2AFn`——`forward` 用 `send_counts/recv_counts`、
      `backward` **对调**再跑同一原语（= `README.md:204-220/242-258` 的 `dispatch_backward`=combine、
      `combine_backward`=dispatch）。**dispatch 语义**：`refs.py:10-123` 是**非 expand**（`:16`
      「Not expanded」、`:67` 按目的 rank 整行复制、`:76-77` 越界槽置 `-1`，第 95 篇度 degree≈3.3
      的来源），训练用 **expand**（`ep.py:577/619`、`README.md:253-255`）；收方 `argsort` 成
      **m-grouped contiguous**（第 11 篇的布局在此长出来）后跑 L1/L2，`w` 乘在激活里（第 99 篇 §1，
      combine 退化成纯求和）。
      → `docs/141-ep-e2e-moe.md`（真实行号核对）。H100 实测
      `harness/exp_ep_e2e_moe.{py,sh,out.txt}`（`world=8,E=64`，四组梯度 vs 稠密、沿 rank 求和）：
      **fp64** 前向 `4.3e-8` / 四梯度 `2.7e-8~4.4e-8`（= router 的 `softmax(·.float())` fp32 地板，
      **算法精确等价**）；**真 fp32**（`allow_tf32=False`）前向 `3.2e-7` / 四梯度 `1.7e-7~4.2e-7`
      （**把通信放回图里 EP ≡ 稠密到单精度**）；**tf32**（NGC 容器默认
      `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`、`fp32_precision=='tf32'`）`2.8e-4`（**不是 EP 的错**，
      是「fp32 名字下的 TF32」两实现 matmul 形状不同导致的舍入分歧；记下：**NGC 容器做 fp32 对拍
      须显式关 TF32**）；**bf16** 前向 `2.3e-3` / 反向 `4~6e-3`（输出舍入地板）；loss 逐位对上。
      真实通信量：平衡负载下等价于 `[world,T_local,H]` all-to-all（`H=7168,bf16,topk=2`），
      单向搬运 `8·T_local·topk·H·2B`——坐实第 1 篇**「K 倍放大」**；本次在**共享节点**
      （GPU0 利用率 100%）测得 **104–123 GB/s** 为**受争用下界**，同机无争用时第 1 篇对同样字节数
      量到 **303–392 GB/s**（~NVLink 上限 82%），对照说明通信带宽对「同卡还有谁在用」极敏感。
      **边界**：Mega MoE 融合 kernel（UMMA/UTCCP/TMEM/NVLink 直写与 readiness）与 DeepEP 真实
      设备路径（NCCL GIN，本机 2.29.7 不匹配，见「阻塞」）**SM100 专属 / 仅分析**；本篇复刻的是
      **语义**、不复刻 SM 内重叠。配套博客 `content/posts/mega-moe-141-ep-e2e-moe`
      （索引页新增第 141 篇）。

### P132 增量（opt-in，补齐反向 ncu 归因（第 15 篇）的对偶：前向的性能来源）

- [x] **P132-1** **Mega MoE 前向 single-step 的 H100 预算归因：前向是 dgrad 的转置对偶**。
      第 15 篇（P7-1）用 ncu 把**反向**四段归因清楚了（wgrad 访存受限 AI~120、dgrad 计算受限 AI~600），
      但**前向的性能来源从没被单独量过**（第 99 篇只给端到端吞吐 312/303 TFLOPS，没拆「为什么」）。
      本篇补上它的**对偶**：同一套 ncu 方法量前向两段 routed GEMM，并加上前向特有的 **L1 epilogue
      （in-place SwiGLU + top-k 权重 + UE8M0 量化）**的定时预算。**核心理念**：前向 L1
      （`m4096 N=2I4096 K=H7168`）与反向 **d1g**（`N=H K=2I`）是 **N↔K 转置对偶**、前向 L2（`N=H K=I`）
      与 **d2g** 同理——两者 FLOP 相同、实测时长/利用率**几乎逐格吻合**；**前向没有 wgrad 那种
      访存受限瓶颈**——wgrad 收缩维是 token（K=512，仅 4 个 BLOCK_K）且写巨大 fp32 权重梯度
      （w1g 940MB / w2g 470MB）→ AI~120 ≪ 拐点 591，而前向收缩维是隐藏维（H/I，大）、输出是 bf16
      激活 → AI 805/651 > 591 → **计算受限**。
      → `docs/142-forward-roofline.md`（真实行号核对）。形状契约
      `csrc/apis/mega_moe.hpp:188-197`（`intermediate_hidden_2 == 2*intermediate_hidden` `:197`、
      `num_tokens <= num_max_tokens_per_rank` `:194`、arch 门控 `:256/367`）；L1 epilogue
      `sm100_fp8_fp4_mega_moe.cuh:1112-1192`（gate/up 先 RNE→bf16 `:1112-1113`、clamp `:1116-1119`、
      `__expf`+`fast_rcp` `:1125-1132`、`__fmul2_rn(gate,up,weights)` `:1134`、`get_ue8m0_sf_exp/_inv`
      `:1173-1176`、`__nv_fp8x4_e4m3` `:1181`、UE8M0 定义 `common/math.cuh:114-147`）；SM90 m_grouped
      FP8 契约 `sm90_fp8_gemm_1d2d.hpp:148`（D 必 bf16）。H100 真机
      `harness/exp_forward_roofline.{py,sh,out.txt}` + `src/moe_forward_epilogue.cu`：
      ① ncu 单次 launch L1 **203.7us / Compute(SM) 76.5% / DRAM 读 345.5MB 写 31.6MB / GMMA
      1 261 568**、L2 **111.4us / 66.4% / 131.2MB+49.2MB / 622 592**（GMMA 比 2.03 = FLOP 比，
      指令效率一致；kernel 名 `sm90_fp8_gemm_1d2d_impl`）；② event 稳态 L1 204.1us/1178 TFLOPS
      （59.5%）、L2 111.1us/1082 TFLOPS（54.7%）、single-shot 229.9/139.7us；③ **L1 epilogue**
      逐行移植后 **24.2us / 3129.7 GB/s（≈93% of HBM 3.35TB/s）**、UE8M0 scale 与 fp64 参考
      **逐位同**、`max|Δ|/amax 5.5e-2`（fp8 一个 ulp）；④ **前向 step 预算 339.4us**
      （204.1+24.2+111.1），epilogue 仅占 GEMM 的 ~7%；⑤ 前向两段计算 roofline 下界 182.3us、
      实测 315.2us（GEMM only）达 **58%**，与第 15 篇反向四段「55–63%」同一完成度；
      ⑥ 前向 step vs 反向四段 single-shot（1804us）→ **反向 ≈ 前向 5.3×**，差距全在 wgrad
       （共 1498.7us，占反向 83%），前向没有对应形状。对照表：forward L1 203.7us/76.5% vs
       d1g 194.3us/75.7%、forward L2 111.4us/66.4% vs d2g 111.0us/65.5%。**仅分析**：融合 kernel 的
       UMMA/tcgen05/TMEM/UTCCP、L1 FP8 输出（2 TMA stage）；本机 8 卡节点共享，带宽数受争用。
       配套博客 `content/posts/mega-moe-142-forward-roofline`（索引页新增第 142 篇）。

### P133 增量（opt-in，补全 P132 前向预算缺的「非 GEMM 段」）

- [x] **P133-1** **Mega MoE 前向 single-step 预算（下）：L2 写回与 combine 归约，补全第 142 篇缺的
       「非 GEMM 段」**（`impls/sm100_fp8_fp4_mega_moe.cuh:1254-1357`（L2 bf16 epilogue 写回）/
       `:1365-1520`（combine 归约）；`layout/mega_moe.cuh:43-52`（`TokenSrcMetadata`）/`:326-343`
       （`get_rank_buffer`/`get_data_buffer`）/`:447-449`（`combine_token_buffer` 邮箱）；
       `common/math.cuh:80-84`（`cast_into_bf16_and_pack`）；`ptx/utils.cuh:65-74`（`ptx::accumulate`）；
       `layout/sym_buffer.cuh:33-40`（`SymBuffer::map`）；对照第 1/11/44/58/85/142 篇）。
       第 142 篇（P132-1）把前向预算算成 `L1 204.1us + L1 epilogue 24.2us + L2 111.1us = 339.4us`，
       但**只到 L2 的 GEMM 输出**——融合 kernel 里 L2 结果还要经 **L2 bf16 epilogue 写回** +
       **combine 归约** 两段才变成 `y`（第 142 篇明确把 L2 的 NVLink 直写列为「仅分析」未进预算）。
       本篇补上。**顺带澄清术语**：Mega MoE 的 combine 与 DeepEP 的 combine **语义相同、方向相反**
       （DeepEP 收方 pull 再本地求和；Mega MoE 专家 rank 主动 push 到源 rank 的
       `combine_token_buffer[topk(+shared)][token][hidden]` 邮箱、源 rank 只做本地求和；
       `get_rank_buffer` 的参数是**槽号**不是 rank、rank 在 `sym_buffer.map` 的地址里），两条腿
       字节量都是 `T·topk·H·2B`（第 1 篇模型）。**H100(sm90) 真机**（`src/moe_forward_combine.cu`
       **直接 include 真实 `common/math.cuh`/`ptx/utils.cuh`** 跑真实设备代码 +
       `harness/exp_moe_forward_combine.{sh,py,out.txt}`，头文件 sha256 `math 576e9db1…`/
       `utils 8a9d5e03…`）：A) scatter→reduce **邮箱逐位 0/12288、`y=Σ_slot bf16` 逐位 0/6144**
       （纯 torch oracle 同判、漏一槽反例 8171/8192）；B) DeepSeek 量级（`T=2048,topk=2,H=7168`）
       **combine 归约 106.5us · 827 GB/s**（读 topk 邮箱 58.7MB + 写 y 29.4MB）、
       **L2 打包参照 73.4us · 1371 GB/s**、**2 卡远端邮箱搬运 186.1us · 315.6 GB/s**（受争用下界，
       对照第 11 篇裸 peer streaming ~369 GB/s、本地 HBM 3.35 TB/s）。**结论**：L2 写回（~186us）+
       combine（~107us）**合计 ~292us，与第 142 篇计算腿 339.4us 同量级**（计算 54% / 通信+归约 46%）
       → 融合 kernel 必须把两腿**重叠**推进，第 142 篇的 339.4us 只是这台机器的**计算侧**；
       通信腿瓶颈是远端带宽/延迟（第 11 篇远端延迟 2.83×本地）。融合 kernel 的 TMEM 读回 /
       NVLink 远端直写 / readiness `st_rel_sys`+`ld_acq_sys`/`grid_sync` **SM100 专属、仅分析**。
       配套博客 `content/posts/mega-moe-143-mega-moe-forward-combine`（索引页新增第 143 篇）。

### P134 增量（opt-in，把 P142/P143 的「必须重叠」结论用真机坐实）

- [x] **P134-1** **Mega MoE 前向 step 的通信-计算重叠真机量化**（`harness/exp_forward_overlap.py`；
      `src/forward_overlap_nvlink.cu`；参考 `impls/sm100_fp8_fp4_mega_moe.cuh:329-334/342-344/393`、
      `scheduler/mega_moe.cuh:344-370`；DeepEP `deep_ep/utils/envs.py:193-204`、
      `deep_ep/buffers/ep.py:431-541`；对照第 65/94/63/142/143 篇）。
      第 142 篇给了前向**计算侧**预算（339.4us）、第 143 篇补了**通信+归约侧**（~292us）并推出
      「两腿同量级→必须重叠」，但那是**串行相加**、**从没真机量过前向的 overlap_factor**。
      本篇补上。**负结果**：NCCL a2a（dispatch fp8 + combine bf16 = **88.1MB/rank**）发**侧流** ×
      **真实 grouped GEMM**（DeepGEMM SM90 L1+L2，319.7us）发**主**流，扫 `NCCL_MAX_NCHANNELS`
      ×reps：**ovf ≈ 1.0（1.003–1.016）**、`eff_vs_ideal` 0.48–0.77、`hidden_frac`≤0.08——
      **侧流 NCCL 与 grouped GEMM 几乎不重叠**（NCCL kernel 占满 SM）；限 channel 反使 `t_comm`
      990→1985us（2.0×）而重叠收益为零（净亏）。本机 a2a **89GB/s**（受争用下界；NCCL a2a ≠ NVLink
      直连）。**正对照**：单进程 8 卡 NVLink **限块数** peer scatter ∥ 计算受限 kernel，
      `overlap ≈ max(t_comm,t_comp)`，`ovf` 从 B=4 的 1.09 升到 **B*=48 的 1.980**（t_comm≈t_comp）
      再降到 B=120 的 1.227——**存在最优 SM 预算**，且 NVLink 只需 ~48 SM 跑满 ~200GB/s。
      **为什么 Mega MoE 单 kernel + warp 解耦**：寄存器级 SM 分区（`kNumDispatchRegisters=48/96`、
      `kNumEpilogueRegisters=208/168`、总和 ≤64512 `:329-334`）。融合 kernel 本体 SM100 专属、仅分析。
      H100(sm90) 真机负结果 8 组 + 正对照 8 组。配套博客
      `content/posts/mega-moe-144-forward-overlap`（索引页新增第 144 篇）。

### P135 增量（opt-in，把 P7-1 的「wgrad 强制累积」根因量化出来）

- [x] **P135-1** **MoE wgrad 的「强制累积」代价：`k_grouped_nt` 的 fp32 D 回读到底占多少**
      （`csrc/apis/gemm.hpp:373-427/181-250/320-371`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84/94`、
      `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`；对照第 7/12/15/22/104/144 篇）。
      第 15 篇（P7-1）把 wgrad 慢归到三条根因，第二条是「SM90 `k_grouped_nt` 被强制累积、`D` 被回读、
      流量翻倍、AI~120≪拐点 591」，但**从没一次对照实验**量化它、也没拆清「回读」与「每专家 fp32 输出」
      各占多少。本篇补上。**合同**：`k_grouped_fp8_gemm_nt_contiguous` 必须 1D1D（`gemm.hpp:384` recipe
      `(1,1,128)`）、`C` 必填（`:403`）、1D1D 的 `D` 必 fp32 且累积（`sm90_fp8_gemm_1d1d.hpp:84/94`，
      kernel 走 `SM90_TMA_REDUCE_ADD_2D` `:346-349`）；dgrad 的 `m_grouped` 是 bf16 直写无回读（`:209`）。
      **方法**：DeepSeek 量级（`T=2048,topk=2,E=8`→每专家 `k=512`）上做**同输出规模/同 FLOP** 的四组对照——
      真实 `k_grouped`(fp32,acc) / `E×`稠密 1D1D(fp32,acc,同 G× 输出) / `E×`稠密 1D2D(bf16,同 G× 输出) /
      `m_grouped` 单组(各专家**求和**,另一算子)。先验同数学：`k_grouped` 的 `∑_e dW[e]` 与 `m_grouped`
      单组同一份 A/B 比 **relerr 3.68e-2**（FP8 噪声）。**反直觉核心结论**：① **K 轴分组 kernel 不是问题**
      ——`k_grouped` 比手动发 `E=8` 个稠密 1D1D 快 **14%**（因子②=0.86×，省发射/调度）；② 主因是
      **同 G× 输出规模下 fp32 累积 vs bf16 直写 = 2.58×**（因子③）；③ 输出 1× 的计算受限形状下纯累积
      只值 1.37×（因子①）。**H100(sm90) 真机**（空闲 GPU1、单进程）：真实 wgrad `k_grouped` **240 TFLOPS**
      （dW2 500.6us / `M=I=2048,N=H=7168`，dW1 997.0us / `M=H,N=2I=4096`），`E×`稠密 1D2D bf16 **537/767
      TFLOPS（2.23×）**，`m_grouped` 单组 **~1240 TFLOPS（5.2×，另一算子）**。**访存账**（解析 + ncu 吻合
      0.1%）：wgrad 总流量 **978.5MB / AI 122.9**（<591 → 访存受限）——**C 回读占 48%**、**每专家 fp32
      输出（读+写）占 96%**；去掉回读 AI→236（仍访存受限、理论上限 ~2×），换 bf16 输出 AI→439。per-group
      K 扫描（`k_e` 128→4096）wgrad **63→842 TFLOPS**（`m_grouped`(求和) 恒 ~1270）。**ncu 硬指标**（单次
      launch 热 cache）：`sm90_fp8_gemm_1d1d_impl`(wgrad) 读 528.3/write 449.8MB/486.5us、
      `sm90_fp8_gemm_1d2d_impl`(m_grouped) 41.0/19.4/99.5us。**该优化什么**：不是换 kernel，而是减小输出
      精度/改累积协议；SM100 wgrad `tn`（`gemm.hpp:320-371`）同结构、**仅分析**。
      `harness/exp_wgrad_accum_cost.{py,sh,out.txt}`。配套博客
      `content/posts/mega-moe-145-wgrad-accum-cost`（索引页新增第 145 篇）。

### P136 增量（opt-in，把 P135-1 结尾的「bf16 输出 2.2× 上界」落地并量精度代价）

- [x] **P136-1** **MoE wgrad 换 bf16 输出的精度代价：2.2× 上界的落地与「归约税」**
      （`csrc/apis/gemm.hpp:373-427/320-371`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84/94`、
      `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`、`sm90_fp8_gemm_1d1d.cuh:346-349`、
      `deep_gemm/utils/math.py:26/52/62`、`sm100_fp8_fp4_gemm_1d1d.hpp:325`；对照第 12/15/145 篇）。
      第 145 篇只量了「换 bf16 输出能快多少（2.23×）」，没量**精度代价**。本篇用真实 SM90 kernel 落地
      并拆开。**合同**：SM90 只有两种输出形态、**没有「fp32 直写」**——1D1D `k_grouped` 必 fp32 且必累积
      （`gemm.hpp:384/403`、`sm90_fp8_gemm_1d1d.hpp:84`、`SM90_TMA_REDUCE_ADD_2D`），1D2D 必 `not c` 且
      `d==bf16`（`sm90_fp8_gemm_1d2d.hpp:81`）。**Part A 单步精度分解**（`dW2 M=2048,N=7168,E=8,k_e=512`，
      逐专家 fp64 参考）：**FP8 输入量化噪声 3.63e-2**、REF fp32 累积 **1.18e-4**、**仅末次 bf16 舍入
      1.41e-3**、ALT 1D2D bf16 **1.42e-3**、REF vs ALT 3.67e-2（≈FP8 噪声）；`cos=0.9993230`、
      `GNS=3.68e-2`——**bf16 输出误差比 FP8 输入噪声小 25.8×**。**Part B 累积 S 步**：REF 1.18e-4→1.26e-4
      （几乎不涨）、ALT 1.42e-3→1.63e-3（缓涨、相对误差近恒定 ~ε_bf16）、cos 稳在 0.99932。**Part C 落地计时**
      （单步 **纯 GEMM 2.27×**：REF 500.5us/步 240 TFLOPS → ALT 220.8us/步 545 TFLOPS）后再扫 S：
      端到端比 **S=1 0.83×（额外物化一份 fp32 反而更慢）→ S=2 1.07× → S=4 1.38× → S=8 1.54× → S=16 1.52×**，
      **在 S≈8–16 封顶 ~1.55×**（归约每步读一份 bf16 `E·M·N·2B=235MB`、实测 ~2.4–2.8 TB/s=HBM 73–84%、
      已是带宽最优，归约占 ALT 32–63%），渐近上界 ≈1.58×。**结论**：bf16 输出「精度免费、收益有条件」——
      仍要物化 fp32 梯度就得赔归约税（~1.55× 封顶），**只有优化器直接吃 bf16 梯度**才拿回 2.2×；要吃到收益
      是**协议改动**（放开 `k_grouped` 输出类型 / 改累积），不是 kernel 微调。**踩坑**：k_grouped host 操作数是 **group-major `[E,mn,k_g]`**（`quant_kg` 写 `xf.T.reshape(-1)`），
      fp64 参考必须**逐组切片再拼列**（当成 `[mn,total_k]` 会让参考 O(1) 崩）。SM100 wgrad `tn` 同族 fp32 累积、
      **仅分析**。H100(sm90) 真机、空闲 GPU1 单进程。`harness/exp_wgrad_bf16_output.{py,sh,out.txt}`。配套博客
       `content/posts/mega-moe-146-wgrad-bf16-output`（索引页新增第 146 篇）。

### P137 增量（opt-in，把 P136-1 结尾的「只有优化器直接吃 bf16 才能拿回 2.2×」量化落地）

- [x] **P137-1** **MoE wgrad 的「fp32 物化 / 归约税」：融合进优化器能拉回多少 2.2×**
      （`csrc/apis/gemm.hpp:373-427/320-371`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`、
      `sm90_fp8_gemm_1d2d.hpp:81`、`sm90_fp8_gemm_1d1d.cuh:346-349`、`deep_gemm/utils/math.py:26/52/62`；
      对照第 12/15/145/146 篇）。第 146 篇只回答「bf16 输出精度免费、端到端封顶 ~1.55×」并甩下一句
      「只有优化器直接吃 bf16 才拿回 2.2×」——**从没量过**。本篇补上。**三协议**：`REF`（每步 1D1D
      fp32 累积，`gemm.hpp:384/403`）、`ALT-R`（每步 `E×` 1D2D bf16 + `torch.sum` + 优化器）、
      `ALT-OPT`（每步 bf16 + **融合进优化器**，读齐 S 份 bf16 就地求和更新 fp32 master，**不落地
      fp32 梯度**；`src/wgrad_reduce_consumer.cu` 的 `k_fused_sgd8`）。**消费侧内核架构无关**（纯
      bf16 访存 + 浮点，无 UMMA/tcgen05），H100 真机直接编译运行、CPU double 对拍 `2.384e-7`。
      **Part A（消费侧）**：手写 `reduce8`/`fused_sgd` 稳在 **~3.08 TB/s（HBM 峰值 3.35 的 92%）**、
      纯读一份 bf16（235MB）78us（3009 GB/s）；**`atomicAdd` 变体 S=1 就 23.4ms（RMW，慢 ~80×）被否决**；
      手写核比 `torch.sum` 快 10–26%（S=16 1371.8 vs 1726.8us、S=8 762.6 vs 837.4us）——**P136 的 838.6us
      「归约税」是次优实现的数**；方法论坑：fp32 store 必须显式 128-bit（标量版 S=1 只有 905 GB/s，
      换 `float4` 后 2442 GB/s）。**Part B（端到端，真实 SM90 GEMM，`dW2 [E=8,M=2048,N=7168]`、
      `sum_k=4096`）**：单步纯 GEMM **2.27×**（REF 499.0us/241 TFLOPS、ALT 219.6us/548 TFLOPS）；三协议
      `S=16`：REF 8441.5us、**ALT-R 5702.3us（1.48×）**、**ALT-OPT 5045.4us（1.67×）**；`S=1`：REF 957、
      ALT-R 1058（**0.90×**，物化 fp32 更慢）、ALT-OPT 616（**1.55×**）。**核心结论**：融合进优化器把端到端
      从 **1.48× 提到 1.67×**、正好顶到**理论天花板 1.69×**（`t_ref/(t_alt+235MB/BW)=499/(219.6+76.3)`）——
      收回「物化税」（省掉 reduce 写 fp32 + 优化器再读回共 8B/elem）；但 **2.27× 不可达**：每步 bf16
      梯度至少被消费端读一遍（~76us/步），叠加在 219.6us/步 GEMM 上。**2.27× 隐含「这份读是免费的」——
      它从来不是**（修正第 146 篇「融合=2.2×」的半句话）。SM100 `tn` 同族 fp32 累积、**仅分析**。
       H100(sm90) 真机、空闲 GPU1 单进程。`harness/exp_wgrad_reduce_consumer.{sh,py,out.txt}`（`.cu`
       sha256 `3e65de90…`）。配套博客 `content/posts/mega-moe-147-wgrad-materialization-tax`
       （索引页新增第 147 篇）。

### P138 增量（opt-in，把 P137-1 结尾的「归约能否藏进 GEMM」走到能核对的数）

- [x] **P138-1** **把 wgrad 的 S 步归约藏进 GEMM：epilogue 融合 / 侧流重叠到底行不行**
      （`csrc/apis/gemm.hpp:373-427/384/403`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`、
      `sm90_fp8_gemm_1d2d.hpp:81`、`sm90_fp8_gemm_1d1d.cuh:346-349`、
      `csrc/jit_kernels/heuristics/sm90.hpp:14/148-188`、`sm100_fp8_fp4_gemm_1d1d.hpp:325`；
      对照第 12/15/144/145/146/147 篇）。第 147 篇把 wgrad 换 bf16 输出的端到端钉在 **1.67×**，
      留下「每步 bf16 梯度必须串行读一遍（~76us/步）」这个口子，可继续①问：**把它融进 GEMM 的
      epilogue、或与下一步 GEMM 重叠，能不能藏掉？** 本篇补上，得到**双重否定的负结果**。
      **（1）字节账**：`G=1.17e8`（bf16 235MB/fp32 470MB），S 步总线上字节——批量 `SEQ`（=ALT-OPT）
      `470·S+940`、侧流增量 `INC-F`(fp32 master RMW) `1410·S`、`INC-B`(bf16 acc RMW) `940·S`，
      **同 kernel 的 epilogue 融合 `EPI-B`(bf16 acc) 反而最少 `470·S`**（省掉 fresh-write+读回与末次
      fp32 master 940MB）。**（2）消费侧内核**（`src/wgrad_epilogue_overlap.cu`，架构无关，H100 真跑真实
      设备代码，vs CPU double 5.96e-8）：纯读 235MB **3005 GB/s**、`fused_all S=8` 2820MB **3084 GB/s**、
      `fused_step` 1175MB 2923、`acc_bf16` 705MB 2970；**块数 ≥264（~2/SM）即打满 HBM**（132 block 只
      2494）。**（3）真机重叠**（真实 SM90 `fp8_gemm_nt` 1D2D bf16，单步 **218us**）：`SEQ8` 2672us 对
      `OVL8-INC-F` 4912us / `OVL8-INC-B` 3597us，比值恒 **0.54–0.84×（比串行尾收还慢）**；`set_num_sms`
      限核 32→132 只把比值 0.54→0.68、绝对值更差；扫消费者块数恒 ~0.58 → **瓶颈不是吞吐而是根本不并行**。
      **（4）决定性控制**（可调动态 smem 的伪 epilogue 生产者 `k_gen_bf16`）：smem 0→128KB 消费者被藏住
      53%→92%，**227KB 时 −0.1%（完全串行）**——坐实 **DeepGEMM 的 `232448 B/SM`（`sm90.hpp:14`）让侧流
      block 无从 co-reside**（与第 144 篇「必须先按 SM 预算分区」一致）。**（5）数值**：bf16 累加器 S=16
      relerr **5.03e-3**、cos 0.9999875，比 FP8 输入噪声 3.6e-2 小 7×（精度免费）。**结论**：唯一剩余门是
      **同 kernel 的 `EPI-B`**（字节最优，上界 `REF/(S·t_gemm)≈2.29×`、下界（+235MB/步不藏）≈1.69×，
      需改 DeepGEMM 1D2D epilogue fresh-write→bf16 累加器 RMW）；侧流/外部融合被 smem 占用堵死。
      SM100 `tn` 同族、**仅分析**。`harness/exp_wgrad_epilogue_overlap.{sh,py,out.txt}`
      （`.cu` sha256 `ea2ec399…`）。配套博客 `content/posts/mega-moe-148-wgrad-epilogue-overlap`
      （索引页新增第 148 篇）。

### P139 增量（opt-in，把 P138-1 结尾的 EPI-B「同 kernel 融合」在真 kernel 上落地验证）

- [x] **P139-1** **EPI-B 落到真 kernel：多出来的 235MB/步读，到底藏住了吗**
      （`csrc/apis/gemm.hpp:488-535/519-521/529`、`csrc/jit_kernels/impls/sm90_bf16_gemm.hpp:89/91`、
      `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh:302-349/381-384`、
      `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`、`sm90_fp8_gemm_1d2d.hpp:81`、
      `csrc/jit_kernels/heuristics/sm90.hpp:14`；对照第 12/15/19/27/135/136/145/146/147/148 篇）。
      第 148 篇把「把 S 步归约藏进 GEMM」走到双重否定，只剩**同 kernel 的 `EPI-B`**（epilogue
      fresh bf16 store → **bf16 累加器 RMW**），给了字节账（`470·S` 最少）与上/下界（~2.29×/~1.69×），
      但**从没在真 kernel 上落地**（原话「本机无法改 SM90 kernel 做端到端验证」）。**本篇发现**：
      DeepGEMM 的 **SM90 BF16 稠密 `bf16_gemm_nt` 本来就支持 bf16 累加器 RMW**——`c=d`（bf16）时
      `GemmDesc.with_accumulation=true`（`sm90_bf16_gemm.hpp:91`、`.cd_dtype=d.scalar_type()` `:89`），
      epilogue 由 `SM90_TMA_STORE_2D` 换成 **`SM90_TMA_REDUCE_ADD_2D`**（`sm90_bf16_gemm.cuh:381-384`，
      smem 亦 bf16 `:302-349`）：**不用改任何 kernel 即可端到端落地 EPI-B**（`use_deterministic_algorithms
      (True)` 强制走 DeepGEMM 自身 kernel、否则回退 cuBLASLt `gemm.hpp:519-521`）。**H100 实测**
      （空闲 GPU 单进程，`dW2 [E·M,N]=[16384,N=7168]`，一份 bf16 D=235MB，`harness/exp_wgrad_epib.{sh,py,out.txt}`）：
      ① **Part A**：`bf16_gemm_nt(a,b,d,c=d)` 两步 == `2·(a@b)`（relerr 1.409e-3 与单步一致）、fp32 累积
      4.075e-7，profile 抓到模板尾 `kWithAccumulation` 由 `false`→`true`；② **Part B** 三协议端到端
      （K=512/步，单步 fresh 197.8us/608 TFLOPS、EPI-B 307.2us、fp32 累积 545.1us）：`SEQ`（每步 fresh
      bf16 + `torch.sum`）对 `EPI-B` 在 `S=1` 是 **1.88×**、`S≥8` 只有 **0.96–0.97×（略慢）**，
      而 `EPI-B/REF`（fp32 累积）稳定 **1.73–1.77×**、几乎正好落在第 148 篇**下界 1.69×**；③ **Part C**
      藏不藏只看算术强度：同一份 RMW 读的暴露比例随 `K=512→2048→4096` 为 **157%/161%/47%**
      （wgrad K 仅 512、最不计算受限）；带宽锚 读一份 bf16 D 235MB=**88.0us/2.67 TB/s**；
      ④ **`ncu` 单 launch**：fresh DRAM 读 74.78MB/写 215.10MB/202.5us，EPI-B 读 **312.75MB**（多出
      **237.97MB = 恰好一份 D**）/写 214.33MB/300.5us，`238MB/98us=2.43 TB/s`——**这份读按满速露在墙上、
      没被计算盖住**。**结论**：EPI-B 是真的、可跑的（`TMA_REDUCE_ADD_2D`），但「多出来的 235MB/步读」
      在 wgrad 形状下**彻底暴露（157%）**，EPI-B 只拿回对 fp32 累积的 1.7×、并不优于「批量 fresh +
      一次流式归约」。载体是 BF16 操作数（FP8 1D2D `sm90_fp8_gemm_1d2d.hpp:81` 不支持累加），
       SM100 wgrad `tn` 同族 fp32 累积、**仅分析**。配套博客
       `content/posts/mega-moe-149-wgrad-epib-accumulate`（索引页新增第 149 篇）。

### P140 增量（opt-in，把 P139-1 结尾的「减少字节」落到「折进收缩维 K」这条路）

- [x] **P140-1** **MoE wgrad 的 big-K 折叠：把 S 步梯度累积折进收缩维，能不能消掉每步累积税**
      （`csrc/apis/gemm.hpp:373-427/403`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`、
      `sm90_fp8_gemm_1d2d.hpp:81`、`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346`、
      `deep_gemm/__init__.py:48/56`；对照第 12/135/145/146/147/148/149 篇）。
      第 145~149 篇把 wgrad 慢的根因量化到底（真凶=每专家一份 fp32 `[M,N]` 输出占 96%；bf16 输出快
      2.27× 但端到端封顶 ~1.55×；融合进优化器 1.67×；把归约藏进 GEMM 被 SM90 `232448 B/SM` 堵死；
      真 kernel EPI-B 多出的 235MB/步读彻底暴露 157%），收尾结论是「**要再快只剩减少字节的路**」、
      点名「把读一份 D 也省掉的分块归约」。本篇把它做成**结构性正面答案**：**「每步累积」只是
      「S 次独立 launch」的产物**——沿收缩维 K 拼接 S 步激活，`dW_e = Σ_s A_{e,s}^T B_{e,s} =
      [A_{e,1};…;A_{e,S}]^T [B_{e,1};…;B_{e,S}]` 恒等于一个 **K=S·k_e 的单次 GEMM**，累加器全程住 TMEM、
      只最后一次 fresh 落盘，**跨 launch 的累积器读写（RMW）归零**。**源码契约**：REF
      `k_grouped_fp8_gemm_nt_contiguous`（`gemm.hpp:373-427`）必传 `c`（`:403`）+ 必 fp32
      （`sm90_fp8_gemm_1d1d.hpp:84`）+ epilogue `SM90_TMA_REDUCE_ADD_2D`（`sm90_fp8_gemm_1d1d.cuh:346`）
      = 每步读回+写；big-K 走现成稠密 1D2D，契约相反 `not c` + 必 bf16（`sm90_fp8_gemm_1d2d.hpp:81`）
      = 一次写。**不需要改内核**（与第 146~149 篇都要改协议/kernel 不同），仅需现成 `fp8_gemm_nt`
      （`deep_gemm/__init__.py:48`）。**H100 实测**（`harness/exp_wgrad_bigk.{sh,py,out.txt}`，空闲 GPU，
      `dW2 [M=2048,N=7168]`、E=8、k_e=512）：**Part A** S=4 逐专家 REF-vs-BIG-K relerr **3.68e-2**、
      与两者各自 vs fp64（3.64e-2/3.69e-2）**同层**=纯 FP8 量化噪声、**同一算子**；**Part B** `S=1→8`：
      REF `644.8→4141.4us`、ALT-R（每步 fresh+归约、同 bf16）`602.6→2599.0us`、**BIG-K `220.7→870.5us`**，
      `REF/BIG-K 2.92→4.76×`、`ALT-R/BIG-K ≈ 3.0×`（恒定），BIG-K **1105 TFLOPS**；**Part C** 输出流量
      REF `S·8B·E·M·N` vs BIG-K 一次 `2B·E·M·N`（S=8 省 **7.3GB**），算术强度 REF 恒 `83~116`（≪591）
      → BIG-K `441→1792`（**越过拐点、计算受限**，治好「小 k_e 访存受限」宿疾）；**前提**：S 步激活须
      同时在手（fp8 驻留 `(M+N)·SUM_K·S`，S=8=302MB，随 S 线性涨）。**收益分解**：纯「折叠」≈3.0×，
      再叠「fp32 累积器→bf16 一次直写」到 4.76×。**遗留**：FP8 累加器路线（8B→2B）未做、误差-字节边界待量化。
      SM100 `tn`（`gemm.hpp:320-371`）同结构、**仅分析**。配套博客
      `content/posts/mega-moe-150-wgrad-bigk-fold`（索引页新增第 150 篇）。

## 每轮增量流程（autopilot）


1. 读本文件「任务清单」选**一个未完成子项**（优先 P0→P1→…）。
2. 写/改 `docs/<NN>-*.md`（会被内联到博客，**里面不要写 {{< >}} 短代码**）与必要的 `src/`、`harness/`。
3. 能跑的用 `scripts/lab.sh exec "..."` 跑出真实数字；**跑不了的（SM100 专属）明确标注「仅分析」**。
4. 精读源码贴**真实核对过的行号**；网络资料给可点击链接（不要编造 URL）。
5. `/tmp/hugo_bin/hugo --gc --minify` 确认无 ERROR；`git add` 只加本专题路径；commit 前缀 `megamoe:`；
   `git pull --rebase` 后 push（网络失败只本地 commit 并在本文件记「待推送」）。
6. 更新本文件状态与「当前进度/下一步」。
7. **不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。**

## 已知坑

- 目标卡代号敏感：写「目标卡」。
- Mega MoE 是 SM100 专属；H100 上**不要**试图跑 `fp8_fp4_mega_moe`，会编译/运行失败。
- 被内联的 `docs/*.md` 里**不要写 `{{< >}}` 短代码**；站内链接只在 `content/posts` 里用 `relref`。
- 行号引用必须真实核对过。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。
- DeepGEMM editable 安装后，JIT 需要 `deep_gemm/include/{cute,cutlass}`；editable 不会自动拷，
  要手动 `cp -r third-party/cutlass/include/{cute,cutlass} deep_gemm/include/`（详见 `docs/08` §2）。
  另需 `apt-get install -y libdw-dev`，并确保 `third-party/deep_jit` 子模块与 gitlink 一致（`2efdab4`）。
- **NGC 容器默认开 TF32**：`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` 使 `matmul.allow_tf32=True`
  （`fp32_precision=='tf32'`）。做 **fp32 数值对拍必须显式 `allow_tf32=False`**，否则会把 TF32
  舍入差（~1e-4）当成算法误差（见 `docs/141` §5）。
- **8 卡节点是共享的**：并行自驱任务会同时占用 GPU，通信/带宽测量可能是**受争用的下界**；
  报数时注明环境或与无争用基线（如第 1 篇）交叉对照。

## 当前进度 / 下一步

- 当前进度：**P0~P14 全部完成；P15~P140 各增量完成（P15-1…P41-1、P42-1…P140-1）。**
  本轮 P140-1：新建 `docs/150-wgrad-bigk-fold.md`
  （+`harness/exp_wgrad_bigk.{sh,py,out.txt}`）——把第 149 篇结尾的「减少字节」路线落到**「折进收缩维 K」**。
  **核心观察**：wgrad 的「每步梯度累积」只是「S 次独立 launch」的产物；沿 K 拼接 S 步激活后
  `dW_e = Σ_s A_{e,s}^T B_{e,s} = [A_{e,1};…;A_{e,S}]^T [B_{e,1};…;B_{e,S}]` 恒等于一个 **K=S·k_e 的单次
  GEMM**，累加器全程住 TMEM、只最后一次 fresh 落盘，**跨 launch 的累积器 RMW 归零**。**契约**：REF
  `k_grouped`（`gemm.hpp:403` 必传 `c`、`sm90_fp8_gemm_1d1d.hpp:84` 必 fp32、epilogue
  `SM90_TMA_REDUCE_ADD_2D` `sm90_fp8_gemm_1d1d.cuh:346`）每步读回+写；big-K 走现成 1D2D
  （`sm90_fp8_gemm_1d2d.hpp:81` `not c`+bf16）一次写。**不需要改内核**（对比第 146~149 篇都要改协议）。
  **H100 实测**（`dW2 [2048,7168]`、E=8、k_e=512，空闲 GPU）：Part A S=4 逐专家 REF-vs-BIG-K relerr
  **3.68e-2** == 两者 vs fp64 同层（纯 FP8 噪声，同一算子）；Part B `S=1→8`：REF `644.8→4141.4us`、
  ALT-R `602.6→2599.0us`、**BIG-K `220.7→870.5us`**，`REF/BIG-K 2.92→4.76×`、`ALT-R/BIG-K≈3.0×`（恒定），
  BIG-K **1105 TFLOPS**；Part C 输出流量 REF `S·8B·E·M·N` vs BIG-K 一次 `2B·E·M·N`（S=8 省 **7.3GB**），
  算术强度 REF 恒 `83~116`（≪591）→ BIG-K `441→1792`（越过拐点、计算受限）。**前提**：S 步激活须同时在手
  （fp8 驻留 `(M+N)·SUM_K·S`，S=8=302MB）。**收益分解**：纯「折叠」≈3.0×，再叠「fp32 累积器→bf16 直写」到 4.76×。
  **遗留**：FP8 累加器路线未做。SM100 `tn` 同结构、仅分析。索引页新增第 150 篇。
  上一轮 P139-1：新建 `docs/149-wgrad-epib-accumulate.md`
  （+`harness/exp_wgrad_epib.{sh,py,out.txt}`）——把第 148 篇留下的「同 kernel EPI-B」**在真 kernel 上落地**。
  **发现**：SM90 BF16 稠密 `bf16_gemm_nt(c=d)` 本就是 **bf16 累加器 RMW**（`sm90_bf16_gemm.hpp:91`
  `with_accumulation`；epilogue `SM90_TMA_REDUCE_ADD_2D` `sm90_bf16_gemm.cuh:381-384`），**无需改 kernel**。
  **H100 实测**（空闲 GPU，`dW2 [16384,7168]`、一份 bf16 D=235MB）：Part A RMW 两步 == `2·(a@b)`、profile
  模板 bool `false→true`；Part B 单步 fresh 197.8us/608 TFLOPS、EPI-B 307.2us、fp32 累积 545.1us，`EPI-B/REF`
  稳定 **1.73–1.77×（顶到第 148 篇下界 1.69×）**，但 `SEQ/EPI-B` 在 S≥8 只有 **0.96–0.97×（略慢）**；
  Part C 同一份 RMW 读的暴露比例 `K=512/2048/4096 → 157%/161%/47%`（藏不藏只看算术强度，wgrad K=512 最不
  计算受限）；带宽锚读一份 D 235MB=88.0us/2.67 TB/s。**`ncu`**：fresh 读 74.78MB/写 215.10MB/202.5us，
  EPI-B 读 **312.75MB（+237.97MB=恰好一份 D）**/写 214.33MB/300.5us → **多读的那份 D 按满速 2.43 TB/s
  露在墙上**。**结论**：EPI-B 真实可跑，但「多出来的 235MB/步读」在 wgrad 形状下**彻底暴露（157%）**，
  它只拿回对 fp32 累积的 1.7×、并不优于「批量 fresh + 一次流式归约」；要再快只剩「减少字节」的路。索引页新增第 149 篇。
  上一轮 P138-1：新建 `docs/148-wgrad-epilogue-overlap.md`
  （+`src/wgrad_epilogue_overlap.cu`、`harness/exp_wgrad_epilogue_overlap.{sh,py,out.txt}`）——量化第 147 篇
  留下的「每步 bf16 梯度要串行读一遍」口子能否藏进 GEMM。**字节账**：批量 ALT-OPT `470·S+940`；侧流增量
  INC-F `1410·S`/INC-B `940·S`；**同 kernel epilogue 融合 EPI-B `470·S`（最少）**。**消费核**（架构无关、
  H100 真跑、vs CPU double 5.96e-8）：纯读 3005 GB/s、fused_all S=8 3084 GB/s、≥264 block 即满带宽。
  **真机重叠**：单步 GEMM 218us，`SEQ8` 2672us 对 OVL8-INC-F 4912 / INC-B 3597 → **0.54–0.84×（比串行还慢）**；
  `set_num_sms` 限核只把比值 0.54→0.68；**决定性控制**：伪 epilogue 生产者动态 smem 0→128KB 藏住 53%→92%、
  **227KB（=DeepGEMM `sm90.hpp:14`）时 −0.1%（完全串行）**——坐实 `232448 B/SM` 让侧流 block 无从 co-reside。
  **数值**：bf16 累加器 S=16 relerr 5.03e-3、cos 0.9999875（< FP8 噪声 3.6e-2）。**结论**：唯一剩余门是**同 kernel
  epilogue 融合 EPI-B**（上界 ~2.29×、下界 ~1.69×，需改 1D2D epilogue）；侧流/外部融合在 SM90 被 smem 占用
  堵死。SM100 `tn` 同族、仅分析。索引页新增第 148 篇。
  上一轮 P137-1：新建 `docs/147-wgrad-materialization-tax.md`
  （+`src/wgrad_reduce_consumer.cu`、`harness/exp_wgrad_reduce_consumer.{sh,py,out.txt}`）——把第 146 篇
  结尾那句「只有优化器直接吃 bf16 才拿回 2.2×」**量化落地**。三协议（`REF` fp32 累积 / `ALT-R` bf16 +
  `torch.sum` + opt / `ALT-OPT` bf16 + **融合进优化器**）；消费侧内核架构无关、H100 真机跑真实设备代码
  （~3.08 TB/s、`atomicAdd` S=1 23.4ms 被否决、手写核比 `torch.sum` 快 10–26%）。**端到端**：纯 GEMM 2.27×；
  `S=16` REF 8441.5us / ALT-R 5702.3us（1.48×）/ **ALT-OPT 5045.4us（1.67×）**，`S=1` ALT-OPT 已 1.55×
  （ALT-R 0.90×）。**核心结论**：融合进优化器把端到端 1.48×→1.67×、顶到**理论天花板 1.69×**
  （收回物化税），但 **2.27× 不可达（每步 bf16 梯度至少被读一遍 ~76us/步）**——修正第 146 篇半句话。
  SM100 `tn` 同族、仅分析。索引页新增第 147 篇。
  上一轮 P136-1：新建 `docs/146-wgrad-bf16-output.md`
  （+`harness/exp_wgrad_bf16_output.{sh,py,out.txt}`）——把第 145 篇结尾的「wgrad 换 bf16 输出的
  2.2× 上界」**落地并量精度代价**。合同：SM90 只有两种输出形态、**没有 fp32 直写**（1D1D `k_grouped`
  必 fp32 且必累积 `gemm.hpp:384/403`+`sm90_fp8_gemm_1d1d.hpp:84`；1D2D 必 `not c` 且 `d==bf16`
  `sm90_fp8_gemm_1d2d.hpp:81`）。**精度分解**（逐专家 fp64）：FP8 输入噪声 3.63e-2、REF fp32 累积 1.18e-4、
  **仅末次 bf16 舍入 1.41e-3**、ALT 1.42e-3、REF vs ALT 3.67e-2、cos 0.9993230——**bf16 输出误差比 FP8
  输入噪声小 25.8× → 精度上基本免费**；累积 S=1→8 REF 几乎不涨、ALT 1.42e-3→1.63e-3（近恒定 ~ε_bf16）。
  **落地计时（S=8）**：纯 GEMM **2.27×**（499.8→220.2us/步），加 `Σ` 一份 fp32 梯度 **838.5us
  （2.80 TB/s≈HBM84%）**，**端到端只剩 1.54×** → 只有优化器直接吃 bf16 梯度才拿回 2.2×。**踩坑**：
  k_grouped 操作数缓冲是 **group-major `[E,mn,k_g]`**，fp64 参考必须逐组切片拼接。SM100 `tn` 同族、仅分析。
  索引页新增第 146 篇。
  上一轮 P135-1：新建 `docs/145-wgrad-accum-cost.md`
  （+`harness/exp_wgrad_accum_cost.{sh,py,out.txt}`）——把第 15 篇「wgrad 被强制累积（D 回读）」这条
  根因**量化并拆开**。合同：`k_grouped_fp8_gemm_nt_contiguous` 必 1D1D（`gemm.hpp:384/403`）、`D` 必
  fp32 且累积（`sm90_fp8_gemm_1d1d.hpp:84`、kernel `SM90_TMA_REDUCE_ADD_2D` `:346-349`），dgrad 的
  `m_grouped` bf16 直写无回读（`:209`）。**同输出规模/同 FLOP 四组对照**：真实 `k_grouped`(fp32,acc) /
  `E×`稠密 1D1D(fp32,acc,同 G× 输出) / `E×`稠密 1D2D(bf16,同 G× 输出) / `m_grouped` 单组(各专家**求和**,
  另一算子)——`∑_e dW[e]` 与单组 relerr 3.68e-2（同数学）。**反直觉结论**：① **K 分组 kernel 不是问题**
  （`k_grouped` 比手动发 8 个稠密 1D1D 快 **14%**，因子②=0.86×）；② 主因是**同 G× 输出规模下 fp32 累积
  vs bf16 直写 = 2.58×**（因子③）；③ 输出 1× 的计算受限形状下纯累积仅 1.37×。**H100(sm90) 真机**（空闲
  GPU1）：真实 wgrad `k_grouped` **240 TFLOPS**、`E×`稠密 1D2D bf16 **537/767（2.23×）**、`m_grouped`
  单组 **~1240（5.2×，另一算子）**。**访存账**（解析+ncu 吻合 0.1%）：978.5MB/AI 122.9（<591 访存受限），
  **C 回读 48%**、**每专家 fp32 输出 96%**。per-group K 扫描 63→842 TFLOPS。ncu：wgrad 读 528.3/write
  449.8MB/486.5us。**结论**：该动的是输出精度/累积协议，不是 kernel；SM100 `tn` 同结构、仅分析。
  索引页新增第 145 篇。
  上一轮 P134-1：新建 `docs/144-forward-overlap.md`
  （+`harness/exp_forward_overlap.{sh,py,out.txt}`、`src/forward_overlap_nvlink.cu`）——把 P142/P143
  「计算腿与通信+归约腿同量级 → 融合 kernel 必须重叠」的**串行推断**用真机坐实。**负结果**：NCCL a2a
  （dispatch fp8 + combine bf16 = 88.1MB/rank）发**侧流** × DeepGEMM **真实 grouped GEMM**（L1+L2，
  319.7us）/1163 TFLOPS）发**主**流，扫 `NCCL_MAX_NCHANNELS`×reps：**ovf ≈ 1.0（1.003–1.016）**、
  `eff_vs_ideal` 0.48–0.77、`hidden_frac`≤0.08 → **侧流 NCCL 与 grouped GEMM 几乎不重叠**（NCCL
  kernel 占满 SM）；限 channel 反使 `t_comm` 990→1985us（2.0×）而重叠收益为零（净亏）。本机 a2a
  **89GB/s**（受争用下界；NCCL a2a ≠ NVLink 直连）。**正对照**：单进程 8 卡 NVLink **限块数** peer
  scatter ∥ 计算受限 kernel，`overlap ≈ max(t_comm,t_comp)`，`ovf` 从 B=4 的 1.09 升到 **B*=48 的
  1.980**（t_comm≈t_comp，411 vs 396us）再降到 B=120 的 1.227——**存在最优 SM 预算**，NVLink 只需
  ~48 SM 跑满 ~200GB/s（正是第 63 篇 `get_theoretical_num_sms` 的物理来源）。**为什么 Mega MoE
  用单 kernel + warp 解耦**：寄存器级 SM 分区（`kNumDispatchRegisters=48/96`、`kNumEpilogueRegisters
  =208/168`、总和 ≤64512 `impls:329-334`，dispatch warp `:342-344`）。结论：**不是「并行发两条 stream」
  就能重叠，而是必须先按 SM 预算分区**。融合 kernel 本体 SM100 专属、仅分析。H100 真机负结果 8 组 +
  正对照 8 组。索引页新增第 144 篇。
  上一轮 P133-1：新建 `docs/143-mega-moe-forward-combine.md`
  （+`harness/exp_moe_forward_combine.{sh,py,out.txt}`、`src/moe_forward_combine.cu`）——第 142 篇
  （P132-1）前向 step 预算的**补全**：把预算只到 L2 GEMM 输出之后缺的两段补上——**L2 bf16 epilogue
  写回**（`impls:1254-1357`：TMEM→寄存器→`cast_into_bf16_and_pack`（RNE）→swizzle STSM→
  `sym_buffer.map` **NVLink 远端直写**源 rank 的 combine 缓冲 `:1346-1351`）与 **combine 归约**
  （`:1365-1520`：grid-index **水位线** readiness `:1405-1441` → 逐 chunk `tma_load_1d` 拉 top-k 贡献
  `:1459` → `ptx::accumulate` bf16→fp32 累加 `:1485` → `__float22bfloat162_rn` 回 bf16 `:1499` →
  TMA store 写 `y`）。**澄清术语**：Mega MoE combine 与 DeepEP combine 语义相同、方向相反
  （DeepEP 收方 pull；Mega MoE 专家 rank push 到源 rank 的 `combine_token_buffer[topk(+shared)][token]`
  邮箱 `layout/mega_moe.cuh:447-449`、源 rank 只本地求和；`get_rank_buffer` 的参数是**槽号**不是 rank，
  rank 在 `sym_buffer.map` 地址里），字节量都是 `T·topk·H·2B`（第 1 篇模型）。**H100(sm90) 真机**
  （`src/moe_forward_combine.cu` **直接 include 真实 `common/math.cuh`/`ptx/utils.cuh`**）：A) scatter→
  reduce **邮箱逐位 0/12288、`y=Σ_slot bf16` 逐位 0/6144**（纯 torch oracle 同判、漏一槽反例 8171/8192）；
  B) DeepSeek 量级（`T=2048,topk=2,H=7168`）**combine 归约 106.5us/827 GB/s**（读 topk 邮箱 58.7MB +
  写 y 29.4MB）、**L2 打包参照 73.4us/1371 GB/s**、**2 卡远端邮箱搬运 186.1us/315.6 GB/s**（受争用下界；
  对照第 11 篇裸 peer streaming ~369 GB/s、本地 HBM 3.35 TB/s）。**结论**：L2 写回（~186us）+
  combine（~107us）**合计 ~292us，与第 142 篇计算腿 339.4us 同量级**（计算 54% / 通信+归约 46%）
  → 融合 kernel 必须把两腿**重叠**推进，第 142 篇的 339.4us 只是这台机器的**计算侧**。融合 kernel 的
  TMEM/NVLink 直写/readiness 原语 **SM100 专属、仅分析**。索引页新增第 143 篇。
  上一轮 P132-1：新建 `docs/142-forward-roofline.md`
  （+`harness/exp_forward_roofline.{py,sh,out.txt}`、`src/moe_forward_epilogue.cu`）——第 15 篇
  （反向 ncu 归因）的**对偶**：把**前向**两段 routed GEMM（L1/L2 的 `m_grouped`）用同一套 ncu 方法
  量透，并加上前向特有的 **L1 epilogue（in-place SwiGLU + top-k 权重 + UE8M0 量化）**预算。
  **核心结论**：前向 L1/L2 与反向 **d1g/d2g 是 N↔K 转置对偶**（FLOP 相同、时长/利用率几乎逐格吻合：
  L1 203.7us/76.5% vs d1g 194.3us/75.7%、L2 111.4us/66.4% vs d2g 111.0us/65.5%），都是**计算受限**；
  **前向没有 wgrad 那种访存受限瓶颈**（wgrad 收缩维是 token K=512、输出巨大 fp32 权重梯度 AI~120；
  前向收缩维是隐藏维、输出 bf16 激活 AI 805/651 > 拐点 591）。**H100 实测**（SM90
  `sm90_fp8_gemm_1d2d`）：ncu L1 203.7us/SM 76.5%/DRAM 345.5+31.6MB/GMMA 1 261 568、L2 111.4us/
  66.4%；event 稳态 L1 204.1us/1178 TFLOPS、L2 111.1us/1082 TFLOPS；L1 epilogue **24.2us/3129.7 GB/s
  （≈93% HBM）**、UE8M0 scale 与 fp64 **逐位同**；**前向 step 339.4us**（epilogue 仅 ~7%）、
  计算 roofline 达 **58%**；**反向 ≈ 前向 5.3×**（差距全在 wgrad，占反向 83%）。融合 kernel
  UMMA/TMEM/UTCCP、L1 FP8 输出 **SM100 专属 / 仅分析**。索引页新增第 142 篇。
  上一轮 P131-1：新建 `docs/141-ep-e2e-moe.md`
  （+`harness/exp_ep_e2e_moe.{py,sh,out.txt}`）——把「通信」真正放回 MoE 图：在 **8×H100** 上用真实
  NCCL `all_to_all_single` 实现 expand 版 dispatch/combine，把 `dispatch→L1→SwiGLU→L2→combine`
  与其 autograd 反向逐张量与**单卡稠密参考**对拍。补的空档：第 99 篇（前向）/第 111 篇（反向）
  的数值复刻都是**单卡**（明确「跳过通信」），第 95 篇虽跨 8×H100 但只验**伴随恒等式**、没接进
  完整 MoE 图也不量参数梯度。**反向的关键事实写成代码**：`all_to_all_single` 不挂 autograd
  （实测 `out.requires_grad=False`）→ 手写 `_A2AFn`，`backward` 把 `send/recv` split **对调**再跑
  同一原语（= `README.md:204-220/242-258`）。**H100 实测**：**fp64** 前向 `4.3e-8`、四梯度
  `2.7e-8~4.4e-8`（= router `softmax(·.float())` 的 fp32 地板，**算法精确等价**）；**真 fp32**
  （`allow_tf32=False`）前向 `3.2e-7`、四梯度 `1.7e-7~4.2e-7`（**EP ≡ 稠密到单精度**）；
  **tf32**（NGC 容器默认 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`）`2.8e-4`（「fp32 名字下的 TF32」
  两实现分歧，**非 EP 的错**；记为坑）；**bf16** 前向 `2.3e-3`/反向 `4~6e-3`；loss 逐位对上。
  真实通信量坐实「K 倍放大」；**共享节点**（GPU0 100%）测得 104–123 GB/s 为**受争用下界**，
  对照第 1 篇同字节数 303–392 GB/s。Mega MoE 融合 kernel 与 DeepEP 真实设备路径（NCCL GIN，
  本机 2.29.7，见「阻塞」）**SM100 专属 / 仅分析**。索引页新增第 141 篇。
  上一轮 P130-1：新建 `docs/140-deepep-comm-context.md`
  （+`harness/exp_deepep_context.{py,sh,out.txt}`、`src/deepep_context_arith.cu`）——补齐 DeepEP 所有
  通信篇目脚下的 **host 通信上下文 `comm::Context`**（`csrc/kernels/comm/context.cpp`，268 行）。
  前面第 7/48/51/53/63/64/66/68/69/73/86/121 篇都从一个已存在的「上下文/窗口」讲起，这个对象从没被
  单独读过。**端到端 NCCL 设备 API（`ncclDevCommCreate`/`ncclCommWindowRegister`）本机 2.29.7
  不提供 → 仅分析**；**全部 host 算术架构无关，H100(sm90) 真机逐条实测**。要点：① `Context` 字段三组
  （域分解/NCCL 句柄/缓冲指针，`api.hpp:37-63`）+ `symmetric_memory` 是 `shared_ptr`（多 Context
  共享一块内存）；② **GIN 需求逐字段**（`context.cpp:121-155`：`GDAKI`、`ginContextCount=QPs`、
  **`ginSignalCount=num_ranks+4`**=第 63 篇免复位屏障槽来源、三级连接 `CUSTOM_STRIDE/RAIL/FULL`）；
  ③ **NCCL 运行时版本闸门**（`:156-164`：`>=2.31` 走 `useRuntimeVersion`+`devCommRuntimeVersionSize`、
  `<2.31` 断言版本相等）——并厘清本机「阻塞」是**两个独立闸门**（本处不要求相等；真正卡住的是
  `deep_ep/__init__.py:19-41` 的 `check_nccl_so` **二进制比对**：torch 载系统 2.29.7 ≠ 期望 pip 2.32.3）；
  ④ 域分解恒等式（`:166-183`，rank=`rdma*nvl+nvl_idx`）+ 对称窗口（`:195-205`，
  `ncclCommWindowRegister` **集合操作**免屏障、链式加法 workspace→buffer→multimem、仅 owner 清零）+
  RDMA storage 窗口（`:207-223`，每 `gin_min_stride` 片一个 `NCCL_WIN_GIN_ONLY`）+ `get_sym_ptr`
  偏移翻译（`:226-232`）+ `finalize()` 拆除逆序（`:244-266`）。H100 真机 **29 PASS / 0 FAIL**（设备常量
  来自 `src/deepep_context_arith.cu` 直接 include 真实头文件）：常量 6/6、`advance_ptr` 布局 8/8、
  域分解 **60 行逐字段 0 不符** + 乘积不变量、版本闸门四组合（ctypes 读 system 2.29.7 / pip 2.32.3）+
  两条二进制比对、句柄缓存三分支。索引页新增第 140 篇。
  上一轮 P129-1：新建 `docs/139-moe-load-balance.md`
  （+`harness/exp_moe_load_balance.{py,sh,out.txt}`）——补齐训练侧一直缺席的**路由负载均衡**算子与反向。
  第 4 篇 router 反向 / 第 127 篇 Mega Gate 反向都把 `bias` 当给定常量（stop-grad），而 DeepGEMM 的
  Mega Gate 只**消费** `bias`（`mega_gate.hpp:41-43`）、不产生它。**三种机制**：**① aux load-balancing
  loss**（$\alpha E\sum_i f_iP_i$，$f$ stop-grad、梯度只走软概率 $P$，$\partial\mathcal{L}/\partial z=
  \frac{\alpha E}{T}p(f-g)$，不变量 $\sum f_i=k$、均匀 $P\Rightarrow\mathcal{L}=\alpha k$）；**② router
  z-loss**（$\beta\overline{\mathrm{lse}^2}$，$\partial\mathcal{L}_z/\partial z=\frac{2\beta}{T}\mathrm{lse}\,p$）；
  **③ aux-loss-free bias**（DeepSeek-V3，$b_i$ 只进排名不进权重、$b\mathrel{+}=\gamma\operatorname{sign}(\cdot)$、
  零梯度）。**与 Mega Gate 对接**（`mega_gate.hpp:56-57`、`impls:283/380-383/396-398/404`、
  `epilogue:212-213`、`test_mega_gate.py:130-139`）。H100(sm90) 真机 **21 PASS / 0 FAIL**：
  aux-loss 解析梯度 vs autograd **2.20e-15**、z-loss **4.95e-16**；无 bias 硬负载 max/min **28.04** →
  bias 800 步 **1.0178**；**核心结论**——aux-loss 梯度是**弱杠杆**（8.50→5.44）、bias 是**强杠杆**
  （28.04→1.03），坐实 DeepSeek-V3 aux-loss-free 动机。纯 host/Python、架构无关。索引页新增第 139 篇。
  上一轮 P128-1：新建 `docs/138-mma-kind-swizzle-selector.md`
  （+`harness/exp_mma_kind_swizzle.{sh,out.txt}`、`src/mma_kind_swizzle.cu`）——补齐 `heuristics/`
  目录里唯一从没被单独读过、且是**每个** SM90/SM100 kernel 的 K 块宽度与 TMA swizzle **唯一出处**
  的 `heuristics/utils.hpp`（32 行）+ `config.hpp:35-61` 的 `GemmDesc` 派生管道 + Python
  `utils/layout.py`。**① `get_num_element_bits`**（逻辑位宽）与 **`block_k=128*8/bits` 的
  `block_k*bits==1024` 不变量**（K 块一行恰 128 B = TMA 最大 swizzle）；**② `get_swizzle_mode`**
  （`{128,64,32,16}` 最大可整除者；`16`=非 swizzle 仅交错，**从不返回 0**）；**③ 逻辑位 vs 存储
  字节**两个位宽 + SM100 的 `/pack_factor`（MXF4 ÷2）→ K-major 全 128、MN-major FP4=64；
  **④ Mega MoE 的平行实现**（自带 `get_num_mma_elem_bytes` + 硬编码 swizzle 128，
  `sm100_fp8_fp4_mega_moe.hpp:69-76` 的半宽 `128→64`）；**⑤ `GemmDesc` 的 dtype→MmaKind 管道 +
  `check_validity` 四类硬约束**；**⑥ Python `get_m_/get_k_` 共用一把对齐旋钮**。
  H100(sm90) 真机直接 include 真实头文件 **43 PASS / 0 FAIL**：`block_k*bits` 全 1024、
  `get_swizzle_mode` 4096 组 0 不符（UNREACHABLE 恰 3136 组）、`check_validity` 10 例矩阵全对、
  K-major swizzle 全 128 / MN-major FP4=64。至此 `heuristics/` 目录全部被专篇覆盖。设备端
  （TMA/UMMA）SM100/SM90 专属、仅分析。索引页新增第 138 篇。
  上一轮 P127-1：新建 `docs/137-sm100-sparse-mqa-logits-metadata-scheduler.md`
  （+`harness/exp_sparse_mqa_logits_metadata.{sh,out.txt}`、`src/sparse_mqa_logits_metadata.cu`）——把第 22 篇
  对 metadata kernel 只给的 30 行概览补齐到与第 136 篇（主 kernel）同深度。**关键结论：整个 metadata
  kernel 架构无关**（只用 `ptx::` 整数/原子、`cp.async`、`cta_exclusive_sum`、`cute::` 下标；不碰
  UMMA/tcgen05/UTCCP/TMA），故 H100(sm90) **原样启动真实 kernel** 并与独立 host oracle 对拍。主循环：
  Q 块构造 + 有效块前缀、稀疏索引加载（非对齐 `ld_evict_first` 就地平移 vs 对齐 `cp_async_cg<256>`）、
  **归并路径二分分块**（输出按线程切、二分求起点消耗的 q0 数）、跨分区去重 + `consume_q1` carry、
  `cta_exclusive_sum` 压缩 + `atomic_add` 按 Q 块预留连续 split、`write_merged_kv_block` 三态物理块 +
  相对槽偏移 + 首块写 header（只看末块判 contiguous/tail）、尾块 padding；末 CTA（release-acquire 计数）
  建调度：连续版均分 + Q 块边界切 + `atomicMax` 波数，分页版有界条目 + `balance_wave_entries` 直方图
  排序/旋转/两 wave 反向。H100 真机 **ALL PASS**：6 组样例 **446 块 / 36 split 0 不符**、调度覆盖
  `miss=dup=bad=0`、A3 每 SM 恰 4/C2 轮转 `[1,1,4,4]`→`[4,1,0,0]`。主 kernel + host `arch_major==10`
  门控 SM100 专属、仅分析。索引页新增第 137 篇。
  上一轮 P126-1：新建 `docs/136-sm100-sparse-mqa-logits-kernel.md`
  （+`harness/exp_sparse_mqa_logits_kernel.{sh,out.txt}`、`src/sparse_mqa_logits_kernel.cu`）——把第 22 篇
  对 SM100 sparse MQA logits 主 kernel 只给的 40 行概览补齐到与第 113 篇（dense 版）同深度。两个入口同一核心
  （唯一分叉是 `KVAccessor` 的两个静态布尔）；`UMMA_N=BLOCK_Q*kNumHeads≤64`（与 dense 钉死 128 不同）、
  `SPLIT_KV=nwg*128`、`split_kv=fp4?640:512`、`num_kv_stages=fp4?5:3`（FP4 半字节→更深流水）。六段架构无关
  逻辑 H100(sm90) 真机 **ALL PASS**（`src/sparse_mqa_logits_kernel.cu` 直接 include 真实 layout 头）：
  A gather 式 KV 的自定义 smem swizzle（低 3 位 XOR bits[3,6)，对合置换）；B metadata 标记位 + 双 Q 槽
  16+16 位打包；C 连续尾块裁剪 + 分页页内映射（`PAGE%SB==0`）；D math warp 压缩列 `(base+slot)*SB+t` 双射；
  E `SharedStorage` 4 组 fp8sb16 222208/fp4sb8 230912 `< 232448`、TMEM raw 376/428→512、线程 768/896；
  F SF 4×32→32×4 转置。主 kernel SM100 专属、仅分析。索引页新增第 136 篇。
  上一轮 P125-1：新建 `docs/135-mhc-backward-sm100-worker-wiring.md`
  （+`harness/exp_mhc_backward_segment.{sh,py,out.txt}`、`src/mhc_backward_segment.cu`）——收口 P122 结尾
  「真实 H=7168 warp-pair 分段」这句：把 mHC 反向单 kernel 接线的**几何**钉死。**前向 Post worker 网格**
  （`impls:108-113`：`kNumRowsPerWarp=16`/`kNumRowGroupsPerWarp=8`/`kNumLanesPerRowGroup=4`；lane 拥有
  行 `(w%4)·16+⌊l/4⌋` 与 `+8`；hidden `hidden_pair_idx` `:460-461`）；**反直觉点**——两 worker **按
  UMMA 原子切、不按行切**、各覆盖全部 64 行最后由 WG3 相加（`:639-640`）；归约 `warp_reduce_sum<4>`
  （`common/math.cuh:217-218`）。**G0 直喂 K1 硬前提**：TMEM `256+32=288<512`；K1 与 Post 同网格读
  `SM100_TMEM_LOAD_16dp256b2x`。H100(sm90) 真机（直接 include 真实头文件）：几何 0 错误、分段归约
  **rel 4–6e-8** vs 全 32-lane **rel 2.47**（结构性错误）、TMEM ALL OK、分段网格 vs 朴素 **2.03–10.93×**；
  真实形状 `u0` scratch 仅 6.3 MB、「直喂」省 12.6 MB（收益是省 1024 次 round-trip 延迟）、K1+G1a
  **AI≈0.75** 访存受限。主 kernel SM100 专属、仅分析。索引页新增第 135 篇。
  上一轮 P124-1：新建 `docs/134-deepep-bucket-test-oracle.md`
  （+`harness/exp_deepep_bucket_tests.{py,sh,out.txt}`）——收口 DeepEP 验收测试栈最后三份：
  `tests/bucket/test_all_gather.py`、`tests/bucket/test_session.py`、`tests/buffer/test_allocation.py`。
  **① all-gather 纯 NVLink copy-engine 语义**（原地 `data_ptr` 判据 + `torch.equal`、`num_sms=0`、
  越界输入仅纯 NVLink、`del srcs` 生命周期守卫）；**② `test_session` 会话内 bump 分配 + 容差梯度**
  （all_gather `rtol=0` vs reduce_scatter/all_reduce `1e-5`）；**③ `test_allocation` 多样式分配**
  （`BufferAllocator` meta→cuda 物化、多 group context）。H100 实测 **ALL PASS**：Part C `plan.num_bytes`
  =4194304、2 MiB 偏移、bump 越界抛错；Part A/B **8×H100 NCCL** all_gather 逐位、reduce_scatter 反序环
  重建 relerr **0.000e+00**（8×NVLink 恰好逐位、测试保守 1e-5）、all_reduce `sum*scale` 匹配；坐实
  `dist.all_gather_single` 在 stock torch 2.12 不存在。bucket 设备 kernel 仅分析。至此 DeepEP 七个子目录
  验收测试全部被单独拆过。索引页新增第 134 篇。
  上一轮 P123-1：新建 `docs/133-refs-near-miss-triage.md`（+`harness/triage_refs.{py,sh,out.txt}`、
  增强 `harness/check_refs.py`）——收口 P96/P97 留下的 near_miss defer。**① 修 linter 分裂区间缺口**
  （`` `a.hpp:103`-`:105` `` 的终点被丢、区间退化单行 → 误报；新增 `CONT_RANGE`/`normalize_citations`
  先合并，当前语料 **13 条**）；**② 把 near_miss 做五分类**（`owner_body`/`far_other`/`absent`/
  `same_line_other`/`adjacent_decl`/`adjacent_code`）；**③ 闭环修 7 处**「引用少写定义/签名行」。
  H100 纯文件解析：总引用 **2074**、near_miss **108→106→99**、严格命中 **640→642→649（73.67%）**、
  四类硬错误恒 0；标识符级分类 **约 93% 为判据噪音**、待判 `adjacent_code` 仅 6（均为书写约定）。
  索引页新增第 133 篇。
  上一轮 P122-1：新建 `docs/132-mhc-backward-sm100-wiring.md`
  （+`harness/exp_mhc_backward_worker.{py,sh,out.txt}`、`src/mhc_backward_worker.cu`）——把第 131 篇
  结尾的目标卡接线推演坐实：前向机器真实行号（Post worker `impls:482-508`、TF32 TS UMMA
  `:235-238/273-275`、Sinkhorn/RMSNorm `epilogue:17-101`）+ **反向六段 → 目标卡对偶表** + 融合边界两条
  判据（跨 token 归约/喂 GEMM 落全局）。H100 真机 **per-token 融合 worker**（一 block 一 token，
  K1+G1a+K2 全在片上、不写 `dAbase`/`dz`）：6 组形状 vs host double 最坏 **5.28e-07**（六段 5.92e-07）、
  `T=65536 H=64` **6000.3→5702.8 us**（~5%，Sinkhorn-bound）；真实形状省 **7.516 GB** 往返
  （实测 908.8 GB/s → 8271 us）。目标卡单 kernel 接线仅分析。索引页新增第 132 篇。
  上一轮 P121-1：新建 `docs/131-mhc-backward-fused-account.md`
  （+`harness/exp_mhc_backward_fused.{py,sh,out.txt}`、`src/mhc_backward_fused.cu`）——把第 130 篇的
  mHC 反向当成**待部署算子流水**做融合/roofline 账：拆成**六段**（K0 重算混合 → G0 `u0=A@fnᵀ` → K1
  系数反向（含 Sinkhorn VJP）→ G1a `dz=du0@fn` / G1b `dfn=du0ᵀ@A` → K2 混合反向），其中 `g` 耦合项
  `dq·2A` 折进 K1、只有 `du0@fn` 留作 GEMM。**结论**：整条反向 **AI≈6.5**、三段 GEMM **AI≈24** ≪ 拐点
  **295** → **访存受限**；`T=16384` 访存下界 **3648.9 us** vs GEMM 计算下界 **68.4 us**（**53×**）。
  H100(sm90) 真机六段 kernel device(fp32) vs double 最坏 **8.84e-7**；元素级三段 K0 **1005.8 GB/s**、
  K2 208.0、K1 内存部分 153.2（串行 Sinkhorn 拖到 56.1）、去低基数原子后 K1 **9.66→3.82 ms**；真实形状
  三段 GEMM **3.04 ms / ~22.3 TFLOPS**、torch 逐 op 元素级 **62.3 ms**、autograd **119.5 ms**（比访存
  下界慢 ~33×）。目标卡映射仅分析。索引页新增第 131 篇。
  上一轮 P120-1：新建 `docs/130-mega-mhc-backward.md`
  （+`harness/exp_mega_mhc_backward.{py,sh,out.txt}`、`src/mega_mhc_backward.cu`）——补齐第 5 篇
  只做前向、第 117 篇只覆盖 Mega Gate 的 **Mega mHC（HyperConnection）变换反向**：残差多流混合 +
  投影 GEMM + sigmoid/仿射 + **Sinkhorn 迭代 VJP** + RMSNorm 的反向推导。**三个非平凡点**：Sinkhorn
  交替归一化的整条序列逆放、`g` 把 24 个投影输出耦合到同一 `q` 的 `dq` 项、两处 bf16 共享边界 +
  投影路 `dz` 回流到 `s`。H100(sm90) 真机：手写 VJP vs **torch autograd**（float64，6 组形状 ×
  9~10 个张量梯度）**最坏 relerr 1.01e-15**、Sinkhorn 反向 vs 中心差分 `~1e-11`、负向对照（丢 `dq`）
  `dx relerr 6.18e-2` 而权重梯度不变、device(fp32) vs double `8e-8~3.6e-7`。主 kernel SM100 专属、
  仅分析。索引页新增第 130 篇。
  上一轮 P119-1：新建 `docs/129-deepep-cmake-debug-build.md`
  （+`harness/exp_deepep_cmake_build.{sh,py,out.txt}`）——补齐 P124 只一句话带过的 **DeepEP CMake 调试
  构建链**：一套模板两个产物（`setup.py` 只编三个 host TU vs CMake 的 `_C` + 两个 runtime 静态库 +
  索引 TU）、`RUNTIME_CUDA_LIBRARIES` 的 `CACHE INTERNAL` 解耦、nvcc 诊断 flag 只作用于 `csrc/indexing/
  main.cu`（三个 host `.cpp` 走 g++）、`cmake-build-*` 双向 ignore。H100(sm90) 真机：configure rc=0、
  三目标全 OK（`_C.so` 1178816 B、两个 `.a` 32794/627820 B），默认 `all` 因索引 TU **rc=2** 失败
  （上游 `rdma.cuh:619/623` 真实编译缺口）。DeepGEMM 对偶无 NCCL / 无 runtime 静态库。索引页新增第 129 篇。
  上一轮 P118-1：新建 `docs/128-quick-plot-pm.md`
  （+`harness/exp_quick_plot_pm.{sh,py,out.txt}`）——补齐 P78/P88 只一句话带过的 profiling
  「最后一公里」：`scripts/quick_plot_pm.py`（448 行纯 host/Python）把 `.ncu-rep` 变成按物理域
  分组的 PM-sampling 时间序列面板，与第 55 篇的内核内 `%clock64` 探针互补（一个 warp 级时间线、
  一个整卡单元实时利用率）。要点：五结构 + 45 条 curated 指标（三种 metric 名形态、`
  .TriageCompute.` 双候选跨代适配）、host 管线三约定（`idx==11` / 固定三行 / `len>1` 闸门）、
  `plot_pm` 的 `CATEGORY_ORDER` 分组与 `%` 固定 `ylim`。H100(sm90) 真机 **73 PASS / 0 FAIL**：
  `ncu 2026.1.1.0 --chip gh100` 得 3728 metric，`resolve_metric_specs('gh100')` = resolved 34 /
  missing 11（TMEM 等目标卡专属），合成渲染 55 KB PNG。漂移：注释引用的 `scripts/ncu-metrics.txt`
  在仓库缺失。索引页新增第 128 篇。
  上一轮 P117-1：新建 `docs/127-mega-gate-backward.md`
  （+`harness/exp_mega_gate_backward.{sh,py,out.txt}`、`src/mega_gate_backward.cu`）——补齐**第 4 篇只覆盖
  softmax 那一路、第 5 篇只做前向的「路由反向」**：Mega Gate 是 `sqrt(softplus(·))` 打分、**带 bias 排名、
  不带 bias 出权重**，且 DeepGEMM 只有前向。前向三件事（阈值 20 分段、排名/权重不同源、`w_i=g·s_i/(Σs+1e-20)`）；
  反向三处非平凡（硬 top-k → bias 对梯度透明、归一化 Jacobian 与第 4 篇式 (6) 同构、`ssp'` 分段导数）。
  H100(sm90) 真机：A 前向/导数 vs double 2.5e-7/6.4e-8、阈值连续；B 无 bias 单调等价 0/4096、有 bias 4096/4096；
  C Jacobian vs 有限差分 5.1e-11；D 整 token 反向 device vs double ~1.1e-7；E eps 影响 0；
  F 手写反向 vs torch autograd ~2e-16 + 中心差分 3.5e-9。索引页新增第 127 篇。
  上一轮 P116-1：新建 `docs/126-deepgemm-deepep-exception-layer.md`
  （+`harness/exp_ep_dg_exception.{sh,py,out.txt}`、`src/exception_layer.cu`、
  `src/exception_bridge.cpp`）——补齐**所有 kernel 共用的「错误 / 断言层」**：两个异常类逐行对称
  （只差 `exception`/`error` 一词）、五个差异（单文件 vs 拆分、各自的 `*_CHECK`、
  `EP_UNIFIED_ASSERT`=`__CUDA_ARCH__` vs `DG_UNIFIED_ASSERT`=`DG_IN_CUDA_COMPILATION` 的
  **`.cu` host 段 `asm("trap;")` 真实陷阱**、无 `py::register_exception` → Python `RuntimeError`
  且消息带文件行号、`*_STATIC_ASSERT` 重复定义）。纯 host C++、架构无关，H100(sm90) 真机
  **36 PASS / 0 FAIL**。索引页新增第 126 篇。
  上一轮 P115-1：新建 `docs/125-generate-pyi-stub-generator.md`
  （+`harness/exp_generate_pyi.{py,sh,out.txt}`）——补齐 P114 只一句话带过的
  **Python 类型桩生成器 `scripts/generate_pyi.py`**（890 行纯 stdlib）。要点：主流程五段
  （真机 `m.def` 62 条 / C++ 函数索引 290 条）、`.pyi` 参数名/类型取自 **C++ 签名**而 `py::arg`
  只给个数+默认值、`BracketTracker` 的四括号语义与 `<>` 边界、类型/默认值映射表、
  **三处真实边界**（`m.attr` 10 个别名全无 stub、`std::variant` 过宽正则误标 `int`、
  `default_args` 序位错配）+ 死代码。纯 stdlib、架构无关，H100(sm90) 真机 **45 PASS / 0 FAIL**
  （端到端与既有 `stubs/_C.pyi` 逐位一致 sha `57a06883344d96b5`）。索引页新增第 125 篇。
  上一轮 P114-1：新建 `docs/124-build-packaging-foundation.md`
  （+`harness/exp_build_packaging.{py,sh,out.txt}`）——补齐「任何设备代码跑起来之前」的
  **构建 / 打包 / 安装期地基**（`find_pkgs.py` + 两个 `setup.py`）。要点：`find_pkg_root:9-55`
  四层语义（env 覆盖 / sys.path 严格大优先级 + 同优先级后者胜 / 子串匹配 / lib_name 根裁剪）、
  `get_nccl_lib_name:72-78` 的 SONAME-only 兜底、持久化环境变量的**代码生成 + 只补不覆盖**、
  `spec_from_file_location` 绕开 `check_nccl_so`（=本机阻塞落点）、版本号 git 断言被 broad except
  **吞成 `+local`**、DeepGEMM wheel 文件名方案 + `DG_SKIP_CUDA_BUILD`、`prepare_includes` 的
  **editable 已知坑**。纯 host/Python、架构无关，H100(sm90) 真机 **54 PASS / 0 FAIL**。
  索引页新增第 124 篇。
  上一轮 P113-1：新建 `docs/123-deepep-python-caching.md`
  （+`harness/exp_deepep_python_caching.{sh,py,out.txt}`）——补齐 DeepEP 反复引用却从没单独读过的
  **Python 缓存基座**：`deep_ep/utils/semantic.py` 的 `weak_lru`（全仓库唯一的方法级缓存）。
  要点：`value_or` 仅 `None` 回退；`weak_lru` 把 key 从 `self` 换成 `weakref.ref(self)`（`:23`），
  让实例可被 GC（普通 `functools.lru_cache` 修饰方法会把 `self` 钉死 → buffer 泄漏）；
  三个真实陷阱（`weakref.ref` 相等性委托 referent → 值相等实例撞同一条 / 死引用条目不自动清只受
  `maxsize` / `__slots__` 抛 `TypeError`）；全库普查得判据「**只有 key 含 `self` 的实例方法才需要
  `weak_lru`**」（模块级 6 + 静态 1 用标准库、无「实例方法直接 lru_cache」；`bucket.py:236` 刻意
  不缓存）；模块级缓存**吞掉环境变化**需 `cache_clear()`。纯 Python、架构无关，H100(sm90) 真机
  **39 PASS / 0 FAIL**。索引页新增第 123 篇。
  上一轮 P112-1：新建 `docs/122-deepep-pp-engram-test-oracle.md`
  （+`harness/exp_deepep_pp_engram_test.{sh,py,out.txt}`）——补齐 DeepEP 两类非 all-to-all 通信原语
  （PP send/recv 与 Engram fetch）一直缺席的「验收测试即正确性规范」：`tests/pp/test_pp.py`（152 行）+
  `tests/engram/test_engram.py`（155 行）。要点：PP 测试定义的是**协议**——`generate_stress_ops`
  用整数时间戳编排「在途张量背压下无死锁的收发序列」（`send_i ≥ recv_(i−N)+1` 使每 pair 在途上界恰为
  `num_max_inflight_tensors=N`），靠 `torch.equal` 确认「收到的正是对端发的」、靠 `torch.cuda._sleep`
  把时间戳翻成真实时序压力；Engram 测试定义的是**索引语义**——all-gather 出全局表再 `global[idx]`
  的参考 oracle、SF 复制到本地、正序/逆序两种层序都验证。kernel 端到端仅分析（`_C.PPBuffer`/
  `_C.EngramBuffer` + `check_nccl_so`），测试侧纯 Python/张量 H100(sm90) 真机 **72 PASS / 0 FAIL**：
  18 组调度每 pair 在途 ≤ N、背压抬高 31 次且修正后仍单调、利齿负样本去掉修正后 31 处容量溢出被击穿、
  Engram `owner/row` 逐元素远程读 == gather-oracle 逐位。索引页新增第 122 篇。
  上一轮 P111-1：新建 `docs/121-deepep-bufferbase-lifecycle.md`
  （+`harness/exp_deepep_bufferbase_lifecycle.{sh,py,out.txt}`、`src/bufferbase_lifecycle.cpp`）——补齐
  DeepEP 四个 Buffer 家族都继承、却从没被单独读过的公共基类 `BufferBase`（`deep_ep/buffers/base.py`
  27 行 + `csrc/buffers/base.hpp` 63 行）。要点：Python ABC 与 C++ pybind **两条平行继承链**（靠
  `self.runtime` 引线相连）；`destroy_on_destruction` 三分支状态机（自动销毁 / 显式模式泄漏告警 /
  `destroyed` 标志把重复销毁变硬错误）+ 为什么必须由派生析构体调用（`destroy()` 纯虚）；
  `~BufferBase() noexcept(false)` 是刻意的；二次 `destroy()` 的跨语言分歧（Python 幂等 / C++ 抛异常）。
  纯 host、架构无关，**H100(sm90) 真机 32 PASS / 0 FAIL**。索引页新增第 121 篇。
  上一轮 P110-1：新建 `docs/120-mega-moe-scheduler.md`
  （+`harness/exp_mega_moe_scheduler.{sh,out.txt}`、`src/mega_moe_scheduler.cu`）——补齐四个调度器里
  **最后一个、也是 Mega MoE 主 kernel 自己**的任务调度器 `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`
  （481 行）。要点：行容量公式 `get_num_max_live_pool_blocks:48-80` + `get_num_l1_warmup_waves:17-45`
  （host ring 分配的唯一几何契约，`csrc/apis/mega_moe.hpp:55-63`）；`L2KBlockDependency:142-166` 的
  XOR 位图 + ring 世代奇偶；`create_task:309-342` 的 warp 现算池块映射（无 host 分组表）；
  locality-domain 分域 + 工作窃取 `kNumMinStealWaves=2`；发布/消费顺序契约（只有 epilogue release）。
  除 `publish_task` 的 cluster 写外全架构无关，**H100(sm90a) 真机 34 PASS / 0 FAIL**（同一 `.cu` 兼编
  sm_90a/sm_100a）。索引页新增第 120 篇（并修掉索引页重复的第 119 条）。
  上一轮 P109-1：新建 `docs/119-deepep-nvlink-p2p-check.md`（+`harness/exp_deepep_nvlink_p2p.{sh,out.txt}`、
  `src/nvlink_p2p_matrix.cu`）——DeepEP 的 NVLink 连接自检 `check_nvlink_connections`（`envs.py:146-181`）
  与 H100 P2P 拓扑；H100 真机 16 PASS / 0 FAIL。
  上一轮 P108-1：新建 `docs/118-deepep-process-bootstrap.md`
  （+`harness/exp_deepep_process_bootstrap.{sh,out.txt}`）——补齐每个 DeepEP 测试跑之前都会执行、
  却从没单独成篇的 **进程 / 环境引导层** `deep_ep/utils/envs.py`（与第 117 篇 `EP_*` 环境变量面、
  第 40 篇 **DeepGEMM 的** `init_dist` 成对照；同名函数两仓库不同物）。要点：`init_dist:89-133` 三件
  host 事（起组 / `set_default_dtype(bfloat16)`+`set_default_device('cuda')` 全局副作用 / `init_seed`）
  + 返回**新建子组**（非 `WORLD`）+ `inspect.signature` 探测 `device_id` 兼容分支；`setdefault` 把
  `NCCL_IB_SL`/`EP_OVERRIDE_RDMA_SL` 压到 1（测试不干扰生产 SL 3）；`init_seed` 的 `local=global+rank`、
  `get_global_seed` 被 combine 参考实现消费、**`get_local_seed` 是死 API**；`dist_print` 静音仍必屏障；
  `EnvOverride` 三元 `enable` + 精确回滚（唯一消费者 engram）。H100 真机 **30 PASS / 0 FAIL**（注入合成
  `deep_ep._C`/`comm` 绕过 `check_nccl_so`，真跑 `init_dist(0,1)` NCCL world_size=1）。索引页新增第 118 篇。
  上一轮 P107-1：新建 `docs/117-deepep-env-surface.md`
  （+`harness/exp_deepep_envs.{py,sh,out.txt}`、`harness/deepep_env_probe.cpp`）——补齐每次部署都要碰、
  却从没单独成篇的 **DeepEP 环境变量 / 开关面**（与第 38 篇 DeepGEMM `DG_*` 成对照）。
  要点：两个通道（安装期固化进生成的 `deep_ep/envs.py`，运行时 `__init__.py:8-16` **只补不覆盖**，
  真实生成物只固化 `EP_NCCL_ROOT_DIR=/tmp/nccl229`）+ `get_env<T>` 两陷阱（只支持 int/string；
  **畸形/空值返回未初始化栈垃圾**，`abc` 真跑得 `21925`）+ 全库带引号 `EP_*` 恰 **16 个**
  （7 persistent + 9 仅运行时，并集 11 条运行时开关）+ 三个真坑（**`EP_BUFFER_DEBUG` 跨语言分歧**：
  C++ `"0"`→关 / Python `"0"`→开；`EP_NUM_MAX_LOCAL_RANKS` 默认 16 且超显存时顺手设
  `NCCL_WIN_STRIDE`；`check_nccl_so` 的「未加载 libnccl 会 `TypeError` 而非断言」）。H100 真机
  **38 PASS / 0 FAIL**（g++ 直接 include 真实 `system.hpp` + 复用真实 `envs.py`/`check_nccl_so`）。
  索引页新增第 117 篇。
  上一轮 P106-1：新建 `docs/116-deepep-gate-generators.md`
  （+`harness/exp_deepep_gate.{py,sh,out.txt}`）——补齐第 65 篇只概括为「流量生成层」的
  **DeepEP MoE 路由夹具 `deep_ep/utils/gate.py`**（180 行）：为 EP dispatch/combine 测试造
  「某 rank 流量是别人几倍」的可控不均衡。要点：`generate_topk_idx:4-29` 是 `rank_count↔topk_idx`
  **精确逆映射**（组内先按 rank）；`generate_rank_count:32-113` 量 a 并 clamp 到 `int((R-1)/ratio)+1`、
  用**循环重复排列**铺普通 rank（precise 均衡机制）、`rank_count` 元素可为 0..k（多专家落同 rank）；
  `get_precise_unbalanced_scores` 用 **0.9 阈值**分离选中/背景；`map_unbalanced_ratio_to_factor:148-164`
  单标量 20 轮二分——大 R 近似**阈值开关**、不可行目标**静默饱和**（`R=16,EPR=1,k=4` 上限≈5、
  目标 8 返回 `factor=99.9999` 实际 4.959 不报错）；**可行性闸门只在 precise 路径**（`test_gate.py:20-24`，
  405→379 跳过 26）。H100 真机 ALL PASS：逆映射 14 例逐位、结构不变式 7 组、复刻 test_gate 判据
  precise **45/45** vs random **14/45**、同 seed 逐位可复现；`gate.py` 只 import torch（绕开 NCCL 阻塞）。
  索引页新增第 116 篇。
  上一轮 P105-1：新建 `docs/115-deepep-refs-fp8-dispatch.md`
  （+`harness/exp_deepep_refs_fp8.{py,sh,out.txt}`、`src/deepep_token_layout_sf.cu`）——补齐第 65 篇
  只点一句、第 114 篇只碰一半的 **DeepEP `refs.dispatch` 的 FP8 分支**：`(x, sf)` 元组通路（参考侧）
  到设备端 `sf_pack_t`/`TokenLayout` SF 布局（设备侧）的接缝。要点：参考固定 **gran-128**
  （`hidden_sf==ceil_div(H,128)`，SF 税 3.125%）、SF 逐 token 同行同掩码、单独一次 all_to_all（同计数）；
  设备 **dispatch 列数取 `sf.size(1)`**（`ep.hpp:356`）、**缓冲**按 gran-32 上界 `ceil_div(H,32)`
  预留（`:269`，是参考的 4 倍）、输出可切 TMA 列主序（`:689-690`）；union 两成员解释「一列=一
  `sf_pack_t`=128 元素」；DeepEP 不打包 UE8M0（全仓唯一 `ue8m0` 字样在 `compiled.cuh:61`/`ep.hpp:351`）。
  H100 真机：8×H100 NCCL all_to_all `(x,sf)` 分支四项 vs 独立 oracle **逐位相等**、
  `per_token_cast_back` rel=0、`hidden_sf` 不符被 assert 拒；include 真实 `layout/ep/token.cuh`
  **host 调用真实 `TokenLayout`/`BufferLayout`** 7 组形状 5 字段 **逐字段 0 不符**（C 11/11、D 6/6）。
  `combine` 无 FP8 变体。端到端 `dispatch.cuh` 仍需 NCCL≥2.32.3（阻塞）。索引页新增第 115 篇。
  上一轮 P104-1：新建 `docs/114-deepep-utils-math.md`
  （+`harness/exp_deepep_utils_math.{py,sh,out.txt}`）——补齐第 65 篇只提一句的 **DeepEP
  `deep_ep/utils/math.py`**（103 行）：判据 `calc_diff`/参考量化器 `per_token_cast_to_fp8,back`/
  **top-k 去重 oracle** `inplace_unique`/**group-limited 路由掩码** `create_grouped_scores`/记账。
  要点：三组「同名不同契约」跨库对照——`calc_diff` 与 DeepGEMM 只差 `+1`（避 0/0 但破尺度不变）、
  量化器未打包/未 UE8M0/gran 硬编码 128（scale 逐位同但 fp8 `10/6400` 差 1、int 分支可互喂
  packed UE8M0）、`count_bytes` 逐字相同；`inplace_unique` 的 `:66` 按 count 排序是死代码；
  `create_grouped_scores` 即 DeepSeek-V3 device-limited 路由掩码。`math.py` 只 import torch，
  H100 真机 **33/33 ALL PASS**。索引页新增第 114 篇。
  上一轮 P103-1：新建 `docs/113-mqa-logits-dense-kernel.md`
  （+`src/mqa_logits_dense_epilogue.cu`、`harness/exp_mqa_logits_dense_epilogue.{sh,out.txt}`）
  ——补齐第 23 篇 §5 对 **SM100 dense MQA logits 主 kernel**（`impls/sm100_mqa_logits.cuh`，
  609 行）只给的「一页摘要」。要点：16 warp 五类角色；`BLOCK_Q=128/num_heads`、`UMMA_M=UMMA_N=128`、
  `SPLIT_KV=384`、`kNumTmemStages==kNumMathWarpGroups=3`；TMEM 三块列预算 + SF 128 对齐；`reuse_kv`。
  **五段架构无关逻辑 H100(sm90a) 真机 ALL PASS**：A 运行期 token 数折射 `Int<max(1,min(N,BQ))>`；
  B TMEM 16/8/4 头块分解；C `cvt.relu`+`__hfma2` 两条 bf16x2 累加器的 `Σ_h relu·w`（一次性 vs 分段
  逐位相等、device vs host 逐位、bf16 vs fp32 `max|Δ|/Σ|term|=2.55e-3`、压缩写回窗口/poison 0 错）；
  D 分页两生产者 KV 页均分（4/8、0/6、0/3）；E 4×32→32×4 转置 0 不符。UMMA/tcgen05/UTCCP SM100
  专属仅分析。索引页新增第 113 篇。
  上一轮 P102-1：新建 `docs/112-bucket-allgather-sm-backends.md`
  （+`src/bucket_allgather_backends.cu`、`harness/exp_bucket_allgather_backends.{sh,py,out.txt}`）
  ——补齐 P54 只当「仅分析」4 行带过的 **DeepEP bucket all_gather 的两个 SM kernel 后端**
  （pure-RDMA `rdma_all_gather` / hybrid `hybrid_all_gather`），把「三类算子 × 三物理域」矩阵拼完。
  要点：`launch_all_gather`（`all_gather.hpp:24-230`）按物理域三分支；pure-RDMA 每 warp 一个目标 rank
  发 `gin.put`；hybrid 是 SM=目标 RDMA rank、warp=QP，每 chunk 按 8 QP 均分（64B 对齐）、空 QP 也发 signal；
  chunk 尺寸由显式代价模型自选（每 QP ≥512KiB）、完成协议是每 `(chunk,QP)` 远端 tail +1 + host
  `WAIT_VALUE_GEQ(target=num_rdma_ranks-1)`。数据面 NCCL GIN 仅分析；信号布局/寻址/协议架构无关，
  H100(sm90a) 直接 include 真实头文件跑真实设备代码 + 真实驱动 mem-op：4 组 `(R,SMs)` 分解、10 组 chunk
  QP 拆分、完成协议四场景全 OK，Python host 算术层独立复刻一致。索引页新增第 112 篇。
  上一轮 P101-1：新建 `docs/111-moe-backward-emulation.md`（+`harness/exp_moe_backward_emul.{py,sh,out.txt}`）
  ——补齐**「反向」一直缺席的端到端误差分解**（第 99 篇前向复刻的**反向对偶**；第 14 篇只答「kernel
  接得对不对」）。要点：Mega MoE 把 top-k 权重**提前乘进 L1 输出**（`kernel:1134`），combine 退化成
  纯 fp32 求和（`:1477-1490`），故反向 `dY` 直传、`w` 只在 SwiGLU 反向；四个 grouped GEMM 角色
  （`dW2=dYᵀa` 两侧激活、`dA=dY·W2` 含权重、`dW1=dGᵀx`、`dX=dG·W1` 含权重）。**H100 真机两层**：
  ① 真实 SM90 kernel 锚点 `vs deq` dW **2.6e-4**（复现第 14 篇）、`vs fp64` 2.4e-2~3.5e-2、cos≥0.9994；
  ② 参考量化器做的**可加来源分解**——`fwd_only` 就把 dX 推到 7.2e-2（占 `full` 8.97e-2 的 **80%**：
  反向误差一大半是前向带进来的）、`dW2` 对反向权重量化完全免疫、`dX` 对反向权重最敏感；DeepSeek 量级
  前向主导 ~90%、最坏 `full` 下 cos>0.99。主 kernel SM100 仅分析。索引页新增第 111 篇。
  上一轮 P100-1：新建 `docs/110-sm100-tcgen05-umma.md`（+`src/tcgen05_umma.cu`、
  `harness/exp_tcgen05_umma.{sh,out.txt}`）——补齐**第 49 篇（SM90 WGMMA 底座）的 SM100 严格对偶**：
  `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`（13 个 SM100 设备文件都 include，却从没单独读过）。
  要点：① **9 个 `*_SS::fma`**（6 data kind × {1SM,2SM} + 1 个 `ws`）——`cta_group::1/2`、`kind::`、
  **`desc >> 32` = 32 位 instruction descriptor**（对应 `mma/sm100.cuh:144-148` 的 `<<32`）、
  **`setp.ne.b32 p, scale_c, 0`** = tcgen05.mma 的「累积/清零」predicate（kernel 传 `s>0 or k>0`）、
  `MXF4` 的 CUDA≥12.9 `.block32` vs `.scale_vec::2X` 分支、`F16BF16_WS_SS`/`F8F6F4_*` 无调用点；
  ② **TMEM/commit/fence**——`tmem_load_32dp32b<N>`（6 档，仅 MQA logits/Mega Gate 用）、
  `umma_arrive_no_elect`、`tcgen05_before/after_thread_sync`（after=等之后读之前 / before=读之后同步之前）；
  ③ **消费者 `sm100_bmk_bnk_mn.cuh`**（把第 17 篇 §6 三行对照落成只用 `F16BF16_SS` 的最小样本）——
  `Allocator1Sm`/每 lane 预存描述符+`__shfl_sync`/`fma(...,0,s>0 or k>0,...)`/epilogue
  `TMEM_LOAD_32dp32b4x`+`st_shared`+**`SM90_TMA_REDUCE_ADD_2D`**；再加 host 包装的 `split_factor`、
  stage 回退、**D 描述符 `(n,m)` 方向**。**H100(sm90a) 真机 7 PASS / 0 FAIL**：**ptxas 编译矩阵**——
  `tcgen05.{mma,mma.ws,commit,fence}` 在 `sm_90a` 逐条 REJECT（`Feature '.cta_group::1' not supported`
  / `Instruction '...' not supported`）、`sm_100a` 全 OK，而 `tmem_load` 因 CUTLASS arch 守卫
  （`copy_sm100.hpp:3474`）在 `sm_90a` **编得过但运行期 trap**；STSM swizzle 整数公式对 128×8 全枚举
  == `cute::Swizzle<3,4,3>`（mismatch 0/1024）；**真实 TMA reduce-add**：0.5+3×1.0→**3.5**（是加非写）、
  以 bmk `(n,m)` 方向两段列 mismatch 0/16384。另给与第 49 篇的八维对照表（描述符/发指令主体/累加器
  TMEM/完成事件/fence/选择器…）。索引页新增第 110 篇。
  上一轮 P99-1：新建 `docs/109-cuda-graph-capture.md`（+`harness/exp_cuda_graph_capture.{py,sh,out.txt}`）
  ——补齐**贯穿训练/推理部署、却一直没被单独讲过的 CUDA Graph 兼容性面与 capture 前的 warm-up 契约**。
  要点：三层时间轴（host 一次性初始化 / capture / replay）是根因——懒初始化落在捕获里会把「一次性」
  工作录进 DAG 并每次 replay 重放；DeepGEMM 三个 API（`get_split_barriers` `mega_mhc.hpp:26-39`、
  `get_score_barriers` `mega_gate.hpp:13-26`、`get_sparse_mqa_logits_workspace` `attention.hpp:284-301`）
  用同一模板（按 `CUDAStream` 懒分配**必须为零**的 scratch + 断言 `currentStreamCaptureStatusMayInitCtx
  == CaptureStatus::None`）；**关键对照**是 cuBLASLt workspace（`runtime.hpp:67-76`）用 `torch::empty`
  **故不需要断言**；发射层（`cuFuncSetAttribute`/cooperative/cluster/PDL/`cuLaunchKernelEx`）全部
  capture-safe。**H100(sm90) 真机 18 PASS / 0 FAIL**：契约用忠实 Python 复刻验证（三 API 在 `arch==10`
  闸门后、H100 不可达），真实 SM90 `bf16`（自研+cuBLASLt+PDL）/`fp8 1d2d` 的 capture+replay 逐位相等，
  64×`256³` bf16 eager 12.14 → graph 6.77 us/call（**1.79×**），profiler 抓到 replay 真实 kernel。
  索引页新增第 109 篇。
  上一轮 P98-1：新建 `docs/108-deepgemm-gemm-scheduler.md`（+`src/gemm_scheduler.cu`、
  `harness/exp_gemm_scheduler.{sh,out.txt}`）——补齐**所有 GEMM / 反向 grouped GEMM 共用、
  被 15+ 次引用却从没单独读过的任务调度器** `scheduler/gemm.cuh`（314 行）。要点：`Scheduler` 一个模板
  覆盖 7 种 `GemmType`；三个非平凡点——① `get_num_1d_blocks_per_group` 编译期用代理代价在 `{8,16}`
  选块分组大小且随 `kIsMulticastOnA` 换主轴；② `get_swizzled_block_idx` 的 `#if __CUDA_ARCH__<1000`
  是 SM90 TMA multicast 必须成对的修补（奇数分组拆最后 1 块）；③ k-grouped 的 `get_next_k_group`
  定义「物理 K 起点对齐 `kKAlignment`、长度=未对齐结尾−对齐起点」与 SF-K 前缀和。另讲清
  `get_global_idx` 7 形态偏移、`get_aligned_effective_m_in_block` 尾块 16 对齐、SM90 的 multicast/计算
  有效性校验。**H100(sm90) 直接 include 真实头文件跑真实设备代码 51 PASS / 0 FAIL**（swizzle 穷举
  10 组双射、8 形态任务覆盖恰一次、k-grouped 含 `KA=256` 不齐整端点 `bad=0`、`fixup_hits=35` 坐实
  修补非摆设）；SM100 分支仅编译期覆盖。索引页新增第 108 篇。
  上一轮 P97-1：新建 `docs/107-refs-linter-audit2.md`
  （+`harness/check_refs.{py,sh,out.txt}`）——第 106 篇（引用体检）的第二遍：把体检重写成**可复用
  linter**，并收口 P96 的 defer 项。要点：P96 脚本自身有三个**会静默误判**的方法学问题——索引扫进
  vendored `third-party/*` 致同名爆炸、消歧排序键没用「被引行是否为空」、占位 `...` 被当错误；
  新 linter 用**内容哈希去重 + `(非空, 主仓库, 目录线索, 引用偏好, 路径短)` 排序键 + `elided` 单列**
  修掉。**H100 实测四类硬错误全 0**（无法解析/越界/空行/歧义），总 1699、严格 530、宽松 617、覆盖
  236 文件；**闭环修 2 处真实 off-by-one**（`docs/34 locality_domain.hpp:13→16`、
  `docs/98 bucket.hpp:78→77`）；**量化「审计文档自指涉」**（`docs/106` 的修正名单会被下一轮当错误 →
  入 `EXCLUDE_DOCS`）。索引页新增第 107 篇。
  上一轮 P96-1：新建 `docs/106-deepgemm-source-index-audit.md`
  （+`harness/exp_deepgemm_source_audit.{py,sh,out.txt}`）——本专题第一篇**元层级**增量：给前 105 篇
  的 ~1900 处 `file:line` 引用做一次可复现的**引用体检**，并给全局算子/API 索引。要点：四层核对 +
  同名文件打分消歧；**1904 条引用里 1898（99.68%）可解析、行号越界 0、19 个 impl + 11 个 api 全部有
  专篇覆盖**；内容严格命中 554、宽松命中 646（741 条抽查）。**闭环修掉 2 处真实错误 + 1 处简写歧义**
  （`m_grouped_gemm.py:92→72`、`mlopart.py:135→tests/test_locality_domain.py:136`、doc38 `mega_moe.cuh`
  →`sm100_fp8_fp4_mega_moe.cuh`），并重跑受影响的 legacy 实验。索引页新增第 106 篇。
  上一轮 P95-1：新建 `docs/105-fp4-mega-moe-weights.md`
  （+`harness/exp_fp4_mega_moe_weights.{py,sh,out.txt}`）——补齐 **Mega MoE 的第二个 dtype 形态：FP4 路由权重**。
  要点：`fp8xfp8`/`fp8xfp4` 同映射到 `MmaKind::MXFP8FP4`，差异只在路由权重 dtype（`kPackedFP4=int8`，
  每字节 2 个 e2m1，逻辑 K 靠「×2」恢复）；TMA 用 `16U4_ALIGN16B` 把 4-bit 摊成可寻址的未打包形式，
  SMEM 每元素仍 1 字节（tile/流水不变、线字节减半，`expect_tx /(kIsWeightFP8?1:2)` 对上）；因 A=FP8、
  B=FP4 必须走 `MXF8F6F4`（`UMMA_K=32`）而非 packed `MXF4`；权重重排 FP8/FP4 共用 tuple 路径且
  N-交错与 K-打包逐位可交换；host 唯一 FP4 专属位是 FP8xFP4 prefill 的小 store block。**主 kernel
  MMA/TMA 仅分析**；H100 真机 ALL_OK=True（packed 往返逐位、FP4 权重误差 ~3.2×、显存恰好减半）。
  索引页新增第 105 篇。
  上一轮 P94-1：新建 `docs/104-moe-backward-communication.md`
  （+`harness/exp_moe_backward_comm.{py,sh,out.txt}`）——补齐**反向侧一直缺席的通信量模型与 H100 实测**。
  要点：**反向通信不是更小、而是约 1.5× 前向**（布局不对称：前向 dispatch 去重 T·d·1B，combine/两个
  反向相位都在 expanded 空间 T·k·2B）；模型 B_bwd/B_fwd=4k/(d+2k)≈1.49；H100 8 卡真实 ragged
  all-to-all 实测字节量比 1.49×、时间比 1.46–1.54×、单向带宽 114–124 GB/s、ragged 相对等分有
  2–3× 惩罚；叠两流只重叠 7.2%→ 反向无 mega-kernel 的真因（四个独立 GEMM + 两张方向相反的网，
  重叠交给 `async_with_compute_stream` + 限制通信 `num_sms`）。索引页新增第 104 篇。
  上一轮 P93-1：新建 `docs/103-sm100-swap-ab-dgrad.md`
  （+`src/sm100_swap_ab_dgrad.cu`、`harness/exp_sm100_swap_ab_dgrad.{sh,out.txt}`）——补齐第 36 篇
  只在一条 bullet 里带过的 **SM100 BF16 GEMM `kSwapAB`（dgrad）分支**。要点：UMMA 的 M 维被钉死在
  64/128/256，而 m-grouped 的 M 是短且逐组变的维、N（hidden）才固定大；swap 在 MMA 里把 A/B 换轴，
  让 `UMMA_N=BLOCK_M`（可逐块动态）承接短 M、`UMMA_M=128` 留给大 N。host 启发式对 m-grouped（含
  mask/psum）返回**唯一** `swap_ab=1/block_n=128/cluster_m=1`、`block_m=mk_alignment`(128 或 256)，
  对照稠密会枚举 `swap_ab∈{0,1}`；常量 `STORE_BLOCK_M=16`、`STORE_BLOCK_N=128`、epilogue 128 线程；
  设备端 major 对调 + `fma(b,a)`；转置写回 `sm100_store_cd_swap_ab`（转置 STSM + `(n,m)` TMA 坐标）。
  **主 kernel SM100 专属、仅分析**；**H100(sm90) host 半边真机 56 PASS / 0 FAIL**（含 `mk_alignment`
  128→256 使 `num_stages` 9→6、`(A@B^T)^T==B@A^T` 逐位、SM90 全 `swap_ab==0`）。索引页新增第 103 篇。
  上一轮 P92-1：新建 `docs/102-mega-gate-mhc-split-barrier.md`
  （+`src/mega_gate_mhc_split_barrier.cu`、`harness/exp_mega_gate_mhc_split_barrier.{sh,out.txt}`）——
  补齐 Mega Gate / Mega mHC **从没被单独读过的两个分段屏障调度器** `scheduler/mega_gate.cuh`
  （`ScoreBarrier`）与 `scheduler/mega_mhc.cuh`（`SplitBarrier`/`Norm`/`Mix`）及其 `Workspace` 布局。
  状态编码 `(grid 标签)*stride + 到达数`、`%gridid` 免复位、`wait_init` 的无符号距离语义、
  `arrive` 的 `red.async` SM100 专属、进位隔离、128B line 防 false sharing、四个角色调用链与
  deterministic 差异。**除一行 `red.async` 外全架构无关**，H100(sm90) 直接 include 真实头文件跑真实
  设备代码 **31 PASS / 0 FAIL**（含 `arrive` 在 `sm_90a` 编不过 / `sm_100a` 编得过）。索引页新增第 102 篇。
  上一轮 P91-1：新建 `docs/101-deepep-barrier-test-oracle.md`
  （+`harness/exp_deepep_barrier_profiling.{py,sh,out.txt}`）——补齐 DeepEP 测试套件里**最后一份
  未覆盖的验收测试** `tests/comm/test_barrier.py` 与它暴露的**通信 kernel 精确测时方法论**。
  `test_barrier` 只测单次屏障耗时（`loop_barrier` ×1000、`bench_kineto(..., barrier_comm_profiling=True,
  barrier=buffer.barrier)`）；`test_loop` 有两条域恒等式断言 + pressure 建/毁循环。域语义由
  `allow_hybrid_mode` 决定（两级 `(rdma,nvl)` vs 单级 `(1,num_ranks)`）；host 链路
  `EPBuffer.barrier→comm.barrier→_C.barrier→barrier()→launch_barrier`（`flush_stores=true`、
  512 线程、`num_sms=(not sequential and scaleout>1)?2:1`）；设备 kernel sequential 两级 vs 并行。
  **设备 kernel 走 NCCL GIN → 端到端仅分析**；**H100 真机 Part B/C/D/E 全 PASS**（4/6/5/4）。
  **Part F 关键发现**：无 `_sleep(2e7)` 时 `bench_kineto` 报告的 kernel 时长呈**双峰**（快档 ~100us /
  慢档 ~417us，≈4×），`barrier_comm_profiling` 每轮落快档、spread<0.3%；直接量持续负载坐实慢档
  =**H100 持续满载降频**（隔离 0.106ms vs 背靠背 0.34–0.66ms）→ `barrier_comm_profiling` 兼具
  「削发射抖动 + 避降频」两重作用。索引页新增第 101 篇。
  上一轮 P90-1：新建 `docs/100-deepep-eplb-test-oracle.md`
  （+`harness/exp_deepep_eplb_test_oracle.{py,sh,out.txt}`）——补齐 EPLB 两个原语一直缺席的
  **验收测试即正确性规范**。前向 blob=唯一 key 的确定性 ramp（`make_blob`，错专家/错载荷每字节暴露）、
  `SENTINEL=0xa5` 哨兵、映射表 **own-expert 排除**；反向 blob=小整数精确 fp32（`make_pattern`，
  period 4095）使原子加**顺序无关**、判据 `torch.equal` 免容差。**设备路径 NCCL GIN 端到端仅分析**；
  **H100 真机 37/37 ALL PASS**：三负样本被断言抓到、10 载荷 × 2 轮 × 8 rank push 920 槽 0 不符、
  11 hidden × 2 轮 × 8 rank pull 176 组 0 不符、prefetch/reduce 关键路径一致（NVLink 430.3 GB/s、
  critical path 671 MB、下限 ~1560 us）。索引页新增第 100 篇。
  上一轮 P89-1：新建 `docs/99-mega-moe-forward-emulation.md`
  （+`harness/exp_mega_moe_forward_emul.{py,sh,out.txt}`）——补齐**前向**一直缺席的
  **端到端数值复刻与误差分解**。主 kernel SM100 专属、仅分析；用 SM90
  `m_grouped_fp8_gemm_nt_contiguous`（1D2D）+ `m_grouped_bf16_gemm_nt_contiguous` 逐段复刻
  dispatch→L1→SwiGLU→L2→combine 的数学与量化语义（权重在 L1 输出、L2 量化之前乘，combine 只求和）。
  **H100 真机**：端到端 fp8-kernel vs fp64 rel **6.28–6.34e-2**（bf16-torch 5.0e-3 / fp32 5.1e-4）；
  分层 kernel vs 反量化参考 3.2e-4、单层 1.47e-3；消融 仅激活 5.70e-2 / 仅权重 4.58e-2 /
  两者 6.28e-2 → **量化是主因、两侧同量级**；吞吐 L1/L2 **312/303 TFLOPS**。索引页新增第 99 篇。
  上一轮 P88-1：新建 `docs/98-deepep-jit-config.md`
  （+`src/deepep_compiled_abi.cu`、`harness/deepep_jit_config_probe.cpp`、
  `harness/exp_deepep_jit_config.{sh,out.txt}`）——补齐 DeepEP「任何设备代码跑起来之前」的**编译配置面**：
  `deep_ep/include/deep_ep/common/compiled.cuh` 的编译期 ABI（`topk_idx_t` 位宽 / `sf_pack_t` UE8M0
  打包单元 / 32-64-2MiB 三档对齐 / 写死成数组维度的通信上限）与 `csrc/runtime/jit.hpp::init_jit`
  的 `Config` 五参 + 8 条 nvcc flag（GIN GDAKI 家族），并讲清**为什么 NCCL include 路径进 DeepJIT
  缓存键、DeepEP 自身头按内容进键**（对照 DeepGEMM 的 `cutlass-<ver>` extra_signature）。
  **H100(sm90) 真机 48 PASS / 0 FAIL**（`compiled.cuh` 默认 64/覆盖 32 各 15 PASS、`get_env`/`Config`
  16 PASS、Python 解析 `jit.hpp` 2 PASS）。索引页新增第 98 篇。
  上一轮 P87-1：新建 `docs/97-sm100-hc-prenorm-gemm.md`
  （+`src/hc_prenorm_sm100.cu`、`harness/exp_hc_prenorm_sm100.{sh,out.txt}`）——
  补齐第 16 篇（P7-2）**只写了 4 条 bullet** 的 SM100 HyperConnection prenorm GEMM，以及第 79/86 篇
  漏掉的 mega 家族最后一个 JIT 启动包装层 `sm100_tf32_hc_prenorm_gemm.hpp`。
  **host 包装层**：`n<=128`（SM90 仅 32）、`block_n=align(n,16)`、num_stages 循环与 SM90 三处差异
  （barrier 家族 2→4 + tmem_full、+4B TMEM 指针、smem 不覆盖成满值）、split 时 D 走 3D
  `[num_splits,m,n]`、grid=`num_splits*ceil(m/64)`。**SM100 kernel（仅分析）**：「TMA→cast→MMA」
  三段流水 + **cast-and-reduce warpgroup**（A bf16→fp32 写 TMEM，因 UMMA TS 无寄存器 A 操作数，
  同拍 `__ffma2_rn` 累平方和）；四组 mbarrier init 计数、`UMMA_K=8`、TMEM 列预算、split-K 分解、
  手写银行组 swizzle。**H100(sm90) 直接 include 真实头文件 21 PASS / 0 FAIL**：
  host 预算表、split-K 恰好覆盖且部分和==完整 GEMM、**`get_swizzled_smem_offset<128,16>` 与
  `cute::Swizzle<3,4,3>` 逐点 ALL MATCH**（+双射）、cast-reduce 映射 64 行全对、2D/3D TMA 描述符
  构造。索引页新增第 97 篇。
  上一轮 P86-1：新建 `docs/96-mega-moe-gate-jit-wrappers.md`
  （+`src/mega_moe_gate_jit_wrapper.cu`、`harness/exp_mega_moe_gate_jit_wrapper.{sh,out.txt}`）——
  补齐第 89 篇（P79-1）**只当对照数了描述符、没拆**的两个 mega 家族 JIT 启动包装层：
  `sm100_fp8_fp4_mega_moe.hpp`（FP8/FP4 Mega MoE 满配版）与 `sm100_bf16_mega_gate.hpp`（Mega Gate），
  凑齐 `sm100_{fp8_fp4_mega_moe, bf16_mega_moe, bf16_mega_gate, mega_mhc}.hpp` 四个包装层。
  **FP8 Mega MoE 三个非平凡点**：① **18 个描述符**（9 routed + 9 shared；更正第 89 篇的「20」计数），
  9 个 SF 全 `Major::MN`+无 swizzle、`sf_smem_outer_dim=block_k/(gran_k*4)`，对齐契约来自 host 把 SF 造成
  「M 连续、末维 stride=M」的 MN-major 列主序；② 共享专家 `E_s` 折叠（L1 N / L2 K 乘 `S`，共享权重走 2D
  描述符），对第 44 篇 `SHARED_L1_SHAPE_N = L1_SHAPE_N·E_s`；③ 模板 25 实参（比 BF16 多
  `STORE_BLOCK_M_L2`/`SF_BLOCK_M/N`/`kNumSFRingTokens`/`to_string(weight_dtype)`）。
  **Mega Gate 三个非平凡点**：① **8 个模板 bool 全由指针/选项空值推出**，与 kernel `if constexpr` 天然一致；
  ② `RoutingArgs` POD ABI（`sizeof=72`，`unmapped_topk_idx_stride` 后 4B padding → offset 48 交界）；
  ③ 任务分解三维枚举 + deterministic 禁 split-K、grid 可不足整卡、cluster 随 `num_mma_ctas` 变，另自管
  score scratch 与按 stream 缓存的 score barrier（CUDA Graph capture 前必须 warm-up）。**主 kernel 全
  SM100 专属、仅分析**；包装层纯 host，H100(sm90) 直接 include 真实头文件 **143 PASS / 0 FAIL**：
  4 形状 × 2 MmaKind 配置不变量、`num_shared==0` 时 4 个共享描述符与 routed **128B 逐字节相等**、
  独立 `cuTensorMapEncodeTiled` 编码器对 3 个描述符**逐字节互证**、模板占位符计数 25/20/15、
  Gate 常量/ABI/scratch/`caches_logical_count` 真值表、4 形状 × deterministic 开关 config
  （`T=64` 关确定性 `splitK=8`、开确定性退回 `1`）。索引页新增第 96 篇。
  上一轮 P85-1：新建 `docs/95-ep-backward-adjoint.md`（+`harness/exp_ep_backward_adjoint.{sh,py,out.txt}`）
  ——补齐 EP 通信一直缺席的**反向侧**：MoE dispatch / combine 的**伴随恒等式**与**路由权重梯度**。
  把 dispatch 看成复制/置换算子 `D`、combine 的 token 求和看成 `Dᵀ`，**两原语互为转置**，于是反向不需要
  新通信 kernel——`dispatch_backward`（`README.md:204-220`）就是 `combine(grad_recv_x, topk_weights=
  grad_recv_w)`、`combine_backward`（`:242-258`）就是 `dispatch(grad_combined_x)`。**核心洞察**：非 expand
  按**目的 rank**复制（`refs.py:67`）使一个 token 的整行 `w` 被复制 `d` 份，朴素求和得 `d·w`，故
  `combined_topk_weights` 必须**选择而非求和**（`combine_reduce_epilogue.cuh:128-139` 的 `master_lane_idx`），
  才能满足 `test_ep.py:514` 的 `== topk_weights`；训练图把权重乘法融进专家计算（`README.md:279`），
  反向 `∂L/∂y=w·g`（复制=dispatch）、`∂L/∂w=<y,g>`（沿 k 归约=combine）。**纯语义/纯张量，H100 真机
  ALL PASS**：Part A 单进程四条恒等式 + 权重梯度对拍 autograd（fp64 ≤1e-12）；Part B **8×H100 真机 NCCL
  用真实 `refs.dispatch`** 跨 rank 验伴随 `<D(x),g>==<x,Dᵀ(g)>`（100.3833/100.3833）、`<G(w),gw>==<w,Gᵀ(gw)>`
  （−18.4716/−18.4716）、复制语义 `Σrecv_w==degree·w`（平均 degree 3.328）。端到端 `EPBuffer` 反向 helper
  依赖 NCCL ≥2.31 → 仅分析。索引页新增第 95 篇。
  上一轮 P84-1：新建 `docs/94-deepep-bucket-mixed-test-oracle.md`
  （+`harness/exp_deepep_bucket_test_oracle.{sh,py,out.txt}`）——补齐 DeepEP bucket 一直缺席的
  **「验收测试即正确性规范」**：`tests/bucket/test_mixed.py`（373 行，**混合**算子的**多轮**随机编排）
  与 `tests/bucket/test_perf.py`（273 行，NCCL profile 基准）。五个规范组件——**能力门控**
  `supported:69-79`（单 rank 仅 all_gather、bf16 仅纯 RDMA）、**尺寸量化** `make_sizes:82-88`
  （`quantum=lcm(group_size·32)`）、**用例枚举** `make_schedule:91-102`、**拓扑** `make_groups:44-66`
  （world + half/nvl + RDMA rail）、**NCCL 参考** `nccl_reference:126-141`（`rs_ag` 参考走
  `all_reduce`，因 `all_gather(reduce_scatter(x))==all_reduce(x)`）+ guard/逐位判据
  `verify:144-154`。另拆 `run_round:157-314` 三条隐蔽契约（三流 event 串 order / 原地指针契约 /
  慢快照）与 perf 的 `measure:51-73`（慢 rank 中位数）。**端到端 BucketBuffer 依赖 NCCL GIN → 仅分析**；
  **测试侧纯 Python/张量，H100(sm90) 真机 33 PASS / 0 FAIL**（`supported` 64/64、真机 `dist.new_group`
  拓扑 3/3、`nccl_reference`+`rs_ag` 不变量 11/11、`make_inputs`/`verify` 8/8、perf 规范 6/6、真实
  `measure` 1/1）；**环境发现**：DeepEP@93eb6eb 用的 `*_single` 通信 API 在本机 torch 2.12.0a0 不存在
  （用 `*_tensor` 兼容别名驱动）。索引页新增第 94 篇。
  上一轮 P83-1：新建 `docs/93-deepep-bucket-session.md`（+`harness/exp_deepep_bucket_session.{sh,py,out.txt}`）
  ——补齐 DeepEP bucket 一直当作细节略过的 **`BucketSession`（会话式临时张量注册 + 生命周期守卫）**。
  常规集合算子要求张量注册在对称 buffer 里（`bucket.hpp:150/213` 裸断言），`BucketSession` 用
  「**实例属性遮蔽类方法**（`:29-43`，嵌套检测即检测副作用）+ **64B 对齐 bump 分配器**（`:55-62`）+
  **越界 src 只对纯 NVLink all-gather 透传**（`:126`）否则 `_foreach_copy` 暂存 + **`_wrap_handle`
  结果回拷**（`:72-88`，只对 `data_ptr` 不同的外部 dst）+ **`stream_control_epilogue` 的
  `record_stream` 守卫**（`bucket.hpp:48-56/314`）」把约束藏起来。**纯 host Python，H100 真机**
  （按路径加载真实 `bucket.py`/`allocator.py`/`event.py` + 同名方法签名的 `FakeBucket` 承接 runtime）
  **31 PASS / 0 FAIL**：注册/暂存/透传三分支、`_allocate` 对齐与越界、嵌套/跨组拒绝、回拷语义、
  `BufferAllocator` meta→CUDA 物化、四份源码行号逐一命中。索引页新增第 93 篇。
  上一轮 P82-1：新建 `docs/92-smem-layout.md`（+`src/smem_layout.cu`、
  `harness/exp_smem_layout.{sh,out.txt}`）——补齐所有 SM100 kernel 共用的**共享内存布局结构层**：
  `layout/gemm.cuh`（`SM100BF16GemmSharedStorage` / `SM100FP8FP4GemmSharedStorage`）与
  `layout/mqa_logits.cuh`（`MQALogitsSharedStorage`），即 `layout/` 里最后两个没被系统拆过的文件。
  **两条策略**：① MQA **精确镜像**——host `MQALogitsConfig::smem_size()`（`sm100_mqa_logits.hpp:168-194`）
  逐段复刻结构体，JIT 源码 `static_assert(sizeof(...) == smem_size)` 钉死，故 `sizeof == num_smem_bytes`；
  ② 稠密 GEMM **保守预算**——`get_pipeline_config`（`heuristics/sm100.hpp:209-252`）barrier 恒按
  `kNumMaxStages=32` 预留（824B）、`store_block_n` 由 `swizzle_cd_mode` 反推。`reserved_barriers`
  把 BF16 每 stage barrier 数从 2 补到 3（对齐 block-scaled 布局）。结构体纯 host 可编译，H100 直接
  include 真实头文件：**42 PASS / 0 FAIL**（8 组 MQA `sizeof==镜像`、8 组 GEMM `used<=host` 且
  `slack==824−barrier` 精确、对齐/SF stage 全对）。索引页新增第 92 篇。
  上一轮 P81-1：新建 `docs/91-sf-contract-cross-api.md`（+`src/sf_contract_conformance.cu`、
  `harness/exp_sf_contract_conformance.{sh,out.txt}`）——以 DeepGEMM 官方 SF 契约
  `docs/scaling-factor-format.md` 为准，核验 host 契约代码：两条路径（自动变换 vs 预变换只校验）、
  §3 预变换 int32 stride 契约（MN-major + TMA 对齐）、§5.2 K-grouped 预打包（仅 gran_k=32、下界 `>=`）、
  §6.2 einsum「按 post-permute 坐标变换再换回」（naive reshape 撞 `layout.hpp:124`）、§6.3 Mega MoE
  四条更硬要求。**关键点**：`check_sf_layout` / `check_k_grouped_packed_ue8m0_tensor` 不含 arch 门控，
  int32 packed UE8M0 虽是 SM100 专属，host 校验层可在 H100 直接跑真实代码；**19 PASS / 0 FAIL**。
  索引页新增第 91 篇。
  上一轮 P80-1：新建 `docs/90-lazy-driver-compat.md`
  （+`src/lazy_driver_compat.cu`、`harness/exp_lazy_driver_compat.{sh,out.txt}`）——补齐两库的
  **「启动前地基」**：DeepEP `csrc/utils/lazy_driver.hpp`（dlopen libcuda + `DECL_LAZY_CUDA_DRIVER_FUNCTION`
  宏 + `EP_CUDA_STRINGIFY` 别名展开）与 `csrc/utils/tensor.hpp`（`get_shape`/`get_data_ptr`）；DeepGEMM
  `csrc/utils/compatibility.hpp`（torch≥2.3 / CUDA≥12.9 三闸门）与 deep_jit `utils/lazy.hpp`（`LazyInit<T>`
  + 两宏，与 DeepEP 版对照）、`backend/cuda/driver.hpp`（CUDA≥12.4、自举 `check_cuda_driver`）、
  `utils/exception.hpp`（libdw 懒加载 + 回溯）；并补上最后一个未覆盖文件 `sm100_locality_domain.hpp`
  （探针 4096/128→32 hops 的 host wrapper）。**纯 host、架构无关，H100 直接 include 真实头文件真机**：
  两库共 50 个驱动符号全可解析、handle 同一指针、VMM + POSIX fd 可共享句柄跨句柄 **2 MiB mismatch=0**、
  libdw 8/8 + trace 真实行号、负样本伪造 `CUDA_VERSION=11000` 被 `compatibility.hpp:11` 拦下——
  **30 PASS / 0 FAIL + 1 编译期负样本**。索引页新增第 90 篇。
  上一轮 P79-1：新建 `docs/89-mega-moe-mhc-jit-wrappers.md`
  （+`src/mega_moe_mhc_jit_wrapper.cu`、`harness/exp_mega_moe_mhc_jit_wrapper.{sh,out.txt}`）——补上
  Mega MoE / mHC 一直缺席的 **JIT 启动包装层** `sm100_bf16_mega_moe.hpp` / `sm100_mega_mhc.hpp`
  （host 张量 → 描述符 → 模板实参 → launch 的最后一公里）。**三个非平凡点**：① 描述符 swizzle 随
  元素字节/通道宽度变化（BF16 128 vs FP8 64，靠 `smem_inner=swizzle/elem` 补偿）；② 常量/开关进模板
  （`activation_clamp` hexfloat、三个 bool 由指针空值推出）；③ 回退用指针别名。**主 kernel SM100 专属、
  仅分析；wrapper 纯 host，H100(sm90) 直接 include 真实头文件真机 50/50 ALL PASS**（含真实 3D TMA
  载入 route 当第三维 0/32768 不符、真机 `get_mega_moe_config`）。索引页新增第 89 篇。
  上一轮 P78-1：新建 `docs/88-deepgemm-profiling-infra.md`
  （+`harness/exp_megamoe_profiling.{sh,py,out.txt}`）——补上 Mega MoE 一直缺席的**官方 profiling 基础设施**：
  `run_ncu_mega_moe.sh`（多进程 NCU 驱动：参数解析 / `DG_JIT_WITH_LINEINFO` / JIT 预热 / PmSampling+
  SourceCounters / 每 rank 一个 ncu 并行）、`test_mega_moe.py` 的 `--ncu-profile-only`/`--local-rank-idx`/
  `--dump-profile-traces` 与性能账目公式、`quick_plot_pm.py`（raw CSV 第 11 列取 PM 采样时间序列）。
  另补两份自动生成的「覆盖面清单」：`generate_pyi.py`→`stubs/_C.pyi`（62 def，漏 10 个 `m.attr`）、
  `csrc/indexing/main.cu`+CMake（17/19 impl，漏 mhc/sparse）。**发现一处不可复现工具 bug**：
  `build_cpp_function_index` 的 `rglob` 遍历顺序决定 `get_tma_aligned_size` 命中定义还是调用点 → pyi 不可复现。
  **目标 kernel SM100+DeepEP 端到端仅分析**，但外壳全部 H100(sm90) 真机验证，**47/47 PASS**。索引页新增第 88 篇。
  上一轮 P77-1：新建 `docs/87-mega-moe-sync-primitives.md`
  （+`src/comm_barrier.cu`、`harness/exp_comm_barrier.{sh,out.txt}`）——补上 Mega MoE 一直缺席的
  **「跨 SM / 跨 rank 同步地基」`comm/barrier.cuh`**：第 11 篇只列了原语表两行、第 38 篇只当内存序调用点
  并注明「仅源码解读」。四原语 `wait_until`（60s@2GHz 超时自旋）/ `cluster_sync_with_relaxed_arrive`
  （弱序，仅用于 2-CTA TMEM 分配）/ `grid_sync` / `nvlink_barrier` + 设备端对称指针 `SymBuffer`
  （相对偏移 → 同一张表编进所有 rank 的 `__grid_constant__`）。两个免复位设计：`grid_sync` 每轮和恰好
  2³¹（只翻最高位、两轮回原值、`(new^old)&0x80000000` 判到齐）；`nvlink_barrier` 的 `counter&3` 编码
  `(phase,sign)`、phase 交替 + sign 每两次翻转（+N 加满 / −1 清回 0，4 次一周期）。**架构无关设备代码，
  H100 直接 include 真实头文件跑真机**：grid_sync 6 组 SM 数 ×2000 轮 mismatch=0 + 计数器翻转坐实免复位；
  虚拟 N∈{2,8} 状态机 trace_bad=0；**真实 8 卡** UVA peer ×30 轮 mismatch=0；cluster relaxed 1/1。
  另记一坑：grid_sync 是同步点非读保护（首版复用同槽误报，改每轮独立槽归零）。总判定 ALL PASS。
  索引页新增第 87 篇。
  上一轮 P76-1：新建 `docs/86-deepep-nccl-comm-binding.md`
  （+`harness/exp_deepep_nccl_binding.{py,sh,out.txt}`）——补上 DeepEP 一直缺席的
  **「NCCL 通信域绑定 + Python 引导层」**：`import deep_ep` 的引导协议（persistent_envs →
  `check_nccl_so` → `init_jit` → 之后才 import `_C` 消费者）、包发现 `find_pkgs.py`
  （env 短路 / `sys.path` 优先级 / SONAME `get_nccl_lib_name`）、**运行时三闸门** `check_nccl_so`
  （抑制开关 / `/proc/self/maps` 重复加载 / `filecmp(shallow=False)` 二进制级）、编译期链接
  `setup.py`（persistent_envs、`-l:libnccl.so.2`+rpath、`_C` 仅三个 host 源）、`_C` 绑定面、
  **`Context` 构造**（2 MiB 对齐断言 / `NCCL_VERSION_CODE>=2.31` 版本闸门 / GIN GDAKI 需求 /
  `npTeamLsa` 两域拆分 / 对称 window 注册 / RDMA storage windows）、`get_sym_ptr` 对称翻译、
  `finalize` 逆序释放，以及 Python `comm` 层（`NCCLCommHandle` managed 语义、comm 句柄缓存与
  PyTorch `_comm_ptr` 复用、barrier 包装）。**纯 host / Python，H100(sm90) 真机 35 PASS / 0 FAIL**：
  真实失配 loaded 系统 NCCL **2.29.7** vs linked pip **2.32.3** → `check_nccl_so` 抛
  `Invalid NCCL versions`；把本机「运行期阻塞」钉到确切行号（两个闸门都要求 `_C` 与 PyTorch
  共用同一份 ≥2.31 的 NCCL）。索引页新增第 86 篇。
  上一轮 P75-1：新建 `docs/85-deepep-ep-test-oracle.md`
  （+`harness/exp_deepep_ep_test.{py,sh,out.txt}`）——补上 DeepEP 侧一直缺席的
  **「验收测试即正确性规范」** `tests/ep/test_ep.py`（616 行）。它是 dispatch/combine「算什么对」的
  可执行定义，也是第 7/11/48/51/52/61/63/71/83 篇反复引用的判据出处。**端到端 buffer 路径依赖
  DeepEP 的 C++/NCCL 设备 API，本机 NCCL 2.29.7 不匹配、运行需 ≥2.31 → 端到端仅分析**；
  **测试侧纯 Python/张量逻辑架构无关，H100(sm90) 真机 17 项 ALL PASS**。要点：模式矩阵
  `enumerate_ep_modes`（六维，实测 72/36/36/18）；五种 dispatch + 两种 combine；逐位判据
  （`torch.equal` + 排序后比 ref + 权重按 `-1` 屏蔽）；`fold_expanded` 不变量；handle 两套 psum
  恒等式（`align` 语义）；`get_unique_and_valid_dst_count` 去重计数；`EPHandle` 字段契约。
  实测：`sys.modules` 桩绕过 `check_nccl_so`、`ast` 抽取真实顶层函数原样运行、8×H100 真机跑
  `refs.dispatch` 坐实「按 src 升序」。索引页新增第 85 篇。
  上一轮 P74-1：新建 `docs/84-deepgemm-python-api-surface.md`
  （+`harness/exp_deepgemm_python_api.{py,sh,out.txt}`）——补上一直缺席的
  **「DeepGEMM 在 Python 里怎么被调到」**：`python_api.cpp:24-49` 的 `PYBIND11_MODULE(_C)`
  装配（1 JIT 注册 + 11 `register_apis`）+ `deep_gemm/__init__.py:16-127` 的显式 re-export。
  `_C` 顶层 **76 名** = 62 `m.def` + 10 `m.attr` 别名 + {`Runtime`,`get_jit`} +
  {`epilogue`,`locality_domain`}（对账 0 多余 0 缺失）。三组默认值：① `compiled_dims` 随主序
  （`nt/nn`=nk、`tn/tt`=mn，18/18）；② `recipe` 按 dtype（`(1,1,128)`/`(1,1,32)`/`(1,128,128)`）；
  ③ `alpha`/FP8 输出对 = `Alpha`/`FP8Quantization` 同义简写（`alpha=2.0` ≡ `epilogue.Alpha(2.0)`
  逐位）。别名 `m.attr`=**同一函数对象**（`is` 10/10）。两子模块 `epilogue`（基类+4 工厂类）/
  `locality_domain`（9 函数）；5 个隐藏入口、5 个 Python-only mega 包装（`.mega` 拆 `SymmBuffer`）；
  无 `py::arg` 入口退化 `argN`；`_C.init(dirname)`+`atexit(shutdown)`。**H100 真机 51/51 PASS**。
  索引页新增第 84 篇。
  上一轮 P73-1：新建 `docs/83-deepep-epbuffer-handle-orchestration.md`
  （+`harness/exp_deepep_epbuffer_handle.{py,sh,out.txt}`）——补上 DeepEP 一直缺席的
  **host Python 编排层与 `EPHandle` 句柄契约**。`EPHandle`（`ep.py:27-207`）的两个身份：
  ① **接口**（`test_mega_moe.py:307-333` 把 `handle.psum_num_recv_tokens_per_expert` 当 grouped
  GEMM 的 PSUM 布局）、② **反向缓存**（`:613-615` 只换 `topk_weights` 复用 expand 布局）。
  **两种 psum 语义不同**（`:43-53`，expand 的 `psum[i]=align(psum[i-1])+未对齐真实计数`）；
  `topk_idx` 是带**版本守卫**的 property（`:103-111`）。**dispatch 五段**（`:562-757`）：守卫
  （`:638-639`，确定性 ∧ `fill_uninitialized_memory` 同开报错）→ auto SM/QP（`:641-645`）→
  SF 解包+缓存句柄三断言（`:662` 是**裸断言**）→ 默认回填+runtime（`:673-699`）→
  `finalize_dispatch`（`:702-747`，确定性 hook vs 立即）；deferred epilogue（`:749-757`）。
  **combine**（`:770-843`）SM 默认复用 `handle.num_sms`、直喂 handle 6 字段、`_unpack_bias`。
  辅助 `value_or`/`weak_lru`/`align`。**H100 真机 51 PASS / 0 FAIL**：注入合成 `deep_ep._C`/`comm`
  桩后按文件路径 exec 真实文件、把未修改的真实方法体绑到假 buffer 逐字段核对（编排不碰 NCCL），
  `deterministic_sort` 用真实 CUDA 张量跑。索引页新增第 83 篇。
  上一轮 P72-1：新建 `docs/82-legacy-test-oracle.md`
  （+`harness/exp_legacy_test_oracle.{py,sh,out.txt}`）——补上整条测试栈里**最后一个还没被单独拆过的
  顶层规格文件** `tests/test_legacy.py`（95 行、两个顶层函数、**无 arch 门控**、只测 bf16）：
  legacy Triton grouped GEMM 的**分组语义规范**。m-grouped 枚举 32 配置（含未透传的 psum 轴，
  仅 16 唯一，`generators.py:118-119/381`）、`expand×test_alias` 共 128 次发射；`test_alias` 是
  主序别名薄壳（`nn`=`nt`+`B.mT`，两次抵消，逐位相等）、`expand` 用恒等行索引（逐位等于
  `m_grouped`）；判据 `calc_diff<0.001`。k-grouped 四因子闸门 `psum / k_alignment!=128 /
  not accumulate / out_dtype!=float`（注释「不裁每组 K 尾块」）**arch9 是 no-op、伪造 arch10
  拦掉 119/126**；不裁尾块也能对靠 generator 把每组物理块 gap 置零。**H100 真机 19 PASS/0 FAIL**：
  直接驱动真实测试体（只打桩 bench_kineto）全过，另做闸门直方图 / `nn`≡`nt` / expand≡m_grouped
  逐位复核 / 判据利齿 / 真实 perf。索引页新增第 82 篇。
  上一轮 P71-1：新建 `docs/81-locality-domain-test-oracle.md`
  （+`harness/exp_locality_domain_test_oracle.{py,sh,out.txt}`）——补上 SM 局部域一直缺席的
  **「验收测试即正确性规范」** `tests/test_locality_domain.py`（167 行，**纯 host / API 层**，
  一个 GPU kernel 都不启动）。5 个顶层函数按序调用（`:160-167`）；helper `assert_raises`（`:19-24`）
  **只吞 RuntimeError**（把「域越界」与「MLOPart 不可用」分成两种异常）、`as_plain`（`:67-70`）是
  `localize` 的逆变换。**环境分水岭**：`is_localization_available()` 末端 `serve()` 断言 2 个 device
  （`mlopart.py:99`），本机 MPS 只给 1 个 → False + 告警一次，本地化分配全线不可用。**域表规范**
  `test_sm_locality_domains:32-49` 四断言（TPC 成对 `:42`、恰好 50/50 `:45`、只搬最少 `:47`）；
  `balance` 的 `slice(0, num_tpcs/2)` 是 **start 不是长度** → `surplus` 取多数域 TPC **后半段**
  （保证已均衡时不动）——读成「前一半」会把 66/66 翻成 `[0,132]` 被断言③抓（本实验利齿负样本）。
  `test_empty` 的域越界契约先于 MLOPart（`runtime:36`），**无 MLOPart 也能验**；`test_localize`
  定义切片形状/`stride(0)` 粒度/`is_localized` 三分（落点 `layout.hpp:150`）；`test_without_mlopart`
  用「剥掉 `-mlopart` 的假 MPS」断言**告警一次 + 拒绝 localize + 域表照给 even**（优雅退化契约）。
  **H100 真机 16 PASS / 0 FAIL**：原样跑 2 个可跑用例（`[66,66]` / `detected, localize refused,
  SM table available`）+ 独立复刻 `get_even`/`balance` 逐位相等 + 契约负样本 E1–E4 + 3 个跑不了用例
  阻塞点定位到行。索引页新增第 81 篇。
  上一轮 P70-1：新建 `docs/80-lazy-init-test-oracle.md`
  （+`harness/exp_lazy_init_test_oracle.{py,sh,out.txt}`）——补上 DeepGEMM **顶层运行时**一直缺席的
  **「验收测试即正确性规范」** `tests/test_lazy_init.py`（199 行，架构无关、H100 整份可跑）。
  **核心看点：用 `LD_PRELOAD` 探针把 cuBLASLt 换成可计数的假库**——探针导出同名
  `cublasLtCreate/Destroy`（`:38/:45`），三个 magic exit code = 三把尺子：`_Exit(91)` 退出期
  use-after-free、`_Exit(92)` handle 泄漏、`_Exit(93)` 重复/陌生销毁；`initialize_probe` 是
  `__attribute__((constructor))`（`:34`）故 `verify_cleanup` 最后跑（`:33`）。6 用例矩阵
  （`:85-92`）在 `CUDA_VISIBLE_DEVICES=''` 下跑（探针返回假 handle 绕开 CUDA），坐实它测纯 host
  对象计数。`MALLOC_PERTURB_=165`+关 tcache（`:131-132`）把退出竞态变可复现崩溃；workspace atexit
  断言退出释放 ≥2×16 MiB（`:152`）。`__main__`（`:187-199`）先 8×`mp.Process` 断言 exit 0。
  **H100 真机 7 PASS/0 FAIL**：原样跑真实测试模块 `Ran 3 tests / OK`（**unittest 输出在 stderr**）；
  独立逐字取出探针复刻 6 用例 create/destroy/clean 全对；`shutdown` 幂等且终态（之后 `get_tc_util()`
  抛 `RuntimeError`，正文带 `lazy.hpp:27`；E1 必须真给 GPU）；env 是**整数**不是布尔
  （`true/false/abc` 被 `DJ_PANIC` 拒）；fork safety 8/8。索引页新增第 80 篇。
  上一轮 P69-1：新建 `docs/79-hc-prenorm-test-oracle.md`
  （+`harness/exp_hc_prenorm_test_oracle.{py,sh,out.txt}`）——补上 HyperConnection prenorm GEMM
  的**「验收测试即正确性规范」** `tests/test_hyperconnection.py`（57 行，`@test_filter(arch>=9)`，
  H100 可跑）。**核心看点：判据编码实现事实**——`allow_tf32=True`（`:17-18`）是「kernel 是 TF32」
  在参考侧的必要镜像（关掉它 `calc_diff(D,真fp32)` → 2.15e-08/2.39e-08 越过 1e-8）；`a.float()@b.T`
  只上转 A、不动 B，对应 kernel 数据路径；`2*m*n*k` 不含 S（融合=搭车）。**H100 真机 8 PASS/0 FAIL**：
  直接 import 真实测试模块、驱动真实测试体（打桩 `bench_kineto` 只留断言）24 用例全过；契约拒
  `n=64/n=20/k=4104`（`sm90_...hpp:30/31`）；split-K 可加性 1.28e-10/4.22e-14；`AI=23.82 FLOP/B`。
  SM100 分支（`n<=128`）仅分析。索引页新增第 79 篇。
  上一轮 P68-1：新建 `docs/78-sf-layout-test-oracle.md`
  （+`harness/exp_sf_layout_test_oracle.{sh,py,out.txt}`）——补上 SF 布局家族一直缺席的
  **「验收测试即正确性规范」** `tests/test_layout.py`：测试自带的纯 torch oracle
  `get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl:20-42`（UE8M0=`>>23`、两个对齐口径、
  4 字节小端合成 int32、`mT` 转 MN 连续）与 C++ 同构回退 `smxx_layout.hpp:108-131` 同一条数学；
  判据分两档（UE8M0 路卡「值逐位+shape+stride」、转置路卡「shape+stride+值不变」）；k-grouped 用
  「每组切出套单组 oracle 再拼回」+ 尾块补零逐位。**H100(sm90) 真机直接 import 真实测试模块复用
  oracle/测试体 8 PASS / 0 FAIL**：`enumerate_sf_layout` 108 用例（54 UE8M0）逐位 0 不符、两条
  kernel 路径互证、1 字扰动被 `torch.equal` 抓到、真实 `test_sf_layout_kernels` 断言全过、两个
  k-grouped 函数被门控跳过且 host API 被 `smxx_layout.hpp:224` 拒、**profiler 坐实**同一 `impl`
  调用有的发射真 kernel、有的 host 早退/`out.copy_` 回退；枚举 arch9 108 / 假 arch10 218。索引页新增第 78 篇。
  上一轮 P67-1：新建 `docs/77-deepgemm-tma-copy-ring-pipeline.md`
  （+`harness/exp_tma_copy_ring.{sh,out.txt}`、`src/tma_copy_ring.cu`）——补上所有 impl 共用的
  **TMA 发射层**与**环形流水线**：`common/tma_copy.cuh`（`get_inner_block_atom_size` 的原子宽度 /
  `LoadOps<2..5>` 三原语 / `copy_nd` 的原子循环与三条静态断言 / sm90 多播 vs sm100 2-SM 两分支 /
  4D 权重描述符）+ `common/ring_pipeline.cuh`（环形 stage + 相位翻转 + 返回旧状态，2 的幂走位运算）。
  **H100(sm90) 真机直接 include 真实头文件跑真实设备代码 27 PASS / 0 FAIL**：atom 循环展开成 4 条
  TMA 拼出 0/4096 逐字节、3D 批次坐标、sm90 多播 cluster=2 两 CTA 收 tx 0/1024；sm100 2-SM 分支
  仅分析。索引页新增第 77 篇。
  上一轮 P66-1：新建 `docs/76-einsum-test-oracle.md`
  （+`harness/exp_einsum_test_oracle.{sh,py,out.txt}`）——补上 DeepGEMM **einsum 家族**一直缺席的
  「验收测试即正确性规范」：`tests/test_einsum.py`（276 行）作为「einsum 算什么对」的可执行规格。
  拆透七个函数（`bmk,bnk->mn` 批归约 / MLA 双向投影 / 权重梯度 / 三个 FP8 变体）、两个
  `@test_filter(arch>=10)` 门控、`calc_diff` 阈值表（1e-5/1e-10/1e-3/2e-3）、逐位 oracle
  （direct==FP32 累积、随机舍入 quartet 哈希、**FP8 输出对==「先 BF16 D 再 per-token 量化」管线**）；
  einsum 特有规格点——带 padding 的权重视图 `fy[:,:,:r]`、`z_0*10` 放大 C 累积语义、fp32
  `bmk,bnk->mn` 强制 `c=d`（否则段错误）。**H100 真机直接 import 真实测试模块调用真实测试体
  15 PASS / 0 FAIL**（两个 SM100 函数当场打印 `Filtered`），profiler 抓确定性开关两侧真实 kernel。
  配套博客 `content/posts/mega-moe-76-einsum-test-oracle`（索引页新增第 76 篇）。
  上一轮 P65-1：新建 `docs/75-deepep-stream-event-overlap.md`
  （+`harness/exp_deepep_stream_event_overlap.{sh,py,out.txt}`、`src/deepep_stream_event.cu`）
  ——补上「通信—计算重叠」一直缺席的 **Python / host 调度地基**：`EventOverlap`
  （`deep_ep/utils/event.py:8-106`）、`get_comm_stream`（通信流 static 单例 + 高优先级 +
  Python 薄包装同 `stream_id`）、`BufferBase`（`main_context` / `destroy` 契约），以及 C++ 孪生
  `EventHandle`（`csrc/utils/event.hpp:13-30`，只暴露默认构造 + `current_stream_wait`，
  `tensors_to_record` 替代 `record_stream`）与 `stream_wait` 两重载（`csrc/comm/stream.hpp:14-27`，
  自等待断言），真正的交接点 `stream_control_prologue/_epilogue`（`csrc/buffers/ep.hpp:122-181`，
  `async_with_compute_stream` / `allocate_on_comm_stream` / 双流 `record_stream`）。**纯 host / Python、
  H100 真机**：`deep_ep._C` 桩注入 + 按文件路径 exec 真实 Python；`EventHandle` 用 `TorchEventHandle`
  逐行镜像驱动真实事件/流；另编译 CUDA 程序镜像流原语——**21 PASS / 0 FAIL**。**核心实测**：
  全卡通信 + 全卡计算**不重叠**（0.989×），通信腿缩到 1/4 SM 则**真重叠 1.776×**——正面坐实第 63 篇
  「半卡 SM 上限」是重叠的必要条件。配套博客
  `content/posts/mega-moe-75-deepep-stream-event-overlap`（索引页新增第 75 篇）。
  上一轮 P64-1：新建 `docs/74-dense-gemm-test-oracle.md`（+`harness/exp_dense_gemm_test_oracle.{sh,py,out.txt}`）
  ——补上整条测试栈最基础的**顶层规格**：`tests/test_fp8_fp4.py` / `tests/test_bf16.py`。前面拆过基座
  （第 40 篇）、生成器（第 41 篇）、逐位断言（第 43 篇）与四个 mega 算子的端到端测试（第 41/47/50/57 篇），
  唯独这两个最先跑的文件一直只被顺带引用。本篇把它当「稠密 GEMM 算什么对」的可执行规格：
  完整 `__main__` 门控、arch 门控矩阵（H100 真机枚举 **252 用例**）、每函数从松到严的判据链
  （`calc_diff` → 20× 逐位 → direct==FP32 累积 → 随机舍入 quartet 哈希 → FP4==无损 FP8 → SF K-stride
  padding → PSUM 零填充 → 空组 NaN 不变），并实证 `calc_diff` 作为余弦相似度的两处盲区（全局缩放不变 /
  单元素 1-ulp 不敏感）。**H100 真机直接 import 真实测试模块调用真实测试体 11 个函数、18 PASS / 0 FAIL**，
  真实 `bench_kineto` 采 BF16 235/413 TFLOPS。配套博客
  `content/posts/mega-moe-74-dense-gemm-test-oracle`（索引页新增第 74 篇）。
  上一轮 P63-1：新建 `docs/73-deepep-sm-qp-autotuning.md`（+`harness/exp_deepep_sm_qp.{sh,py,out.txt}`）
  ——补上 DeepEP dispatch/combine 一直缺席的**上游问题**：每次 `dispatch()`/`combine()` 开多少 SM / QP？
  答案不是超参，而是 host 侧**解析式 roofline** `EPBuffer.get_theoretical_num_sms:431-541` /
  `get_theoretical_num_qps:543-560`：把通信拆成「HBM 读 / HBM 写 / RDMA / NVLink」四本**每 token 流量账**，
  用 group-limited gate 的期望命中组数 `get_expected_topk=G·(1−C(E−E/G,k)/C(E,k))` 归一化，取**预计耗时
  最长**的链路当标尺，用每 SM HBM 带宽（读 180/写 45）换算 SM 数，三个安全阀 = `×1.3` + `align 4` +
  **硬上限半卡** `device_sms//2*2`——这就是「通信—计算重叠」的预算来源。QP 与之解耦：direct `min(n,9)`、
  hybrid `n*16+1`，预分配 direct 17 / hybrid 65 或 129。带宽探测走真实 `nvidia-smi nvlink -s`(×0.9) /
  `ibstat`(÷8)。**纯 host、H100 真机**用 `ast` 从真实 `ep.py` 抽取三个方法源码跑**未修改真实方法体**，
  与独立 oracle 对拍：**16 PASS / 0 FAIL**；并坐实三个真实边界（`E%num_groups==0`、无 IB 时 hybrid 自动
  探测 `ZeroDivisionError`、overlap 关时 ≥64）。配套博客
  `content/posts/mega-moe-73-deepep-sm-qp-autotuning`（索引页新增第 73 篇）。
  上一轮 P62-1：新建 `docs/72-bucket-allreduce-reduce-scatter.md`（+`src/deepep_bucket_collectives.cu`、
  `harness/exp_deepep_bucket_collectives.{sh,py,out.txt}`）——补上第 64 篇（P54-1）只当「仅源码分析」
  放过的两个 bucket 集合算子 `all_reduce`/`reduce_scatter`。与 all-gather（纯搬运、copy engine）不同，
  这两个**要归约、必须有 SM kernel**：纯 NVLink 腿 `multimem.ld_reduce.add` 硬件归约（流量降到 O(1)）、
  RDMA 腿反向环 + `cp.reduce.async.bulk.add.f32`、hybrid 三级（multimem→RDMA→broadcast）。三要点：
  ① 两种**完全不同**的分块映射（all_reduce rank-major / reduce_scatter slot-major，R=4 时
  8191/8192 不同）；② 信用环 head/tail 背压（最后一个用槽的 chunk 免信用）；③ fp32 反序环求和顺序
  与测试参考逐字一致（`rtol=0` 逐位判等的根因）；bf16 路径随机舍入哈希与 DeepGEMM 同源。数据面
  NCCL GIN / `cvt.rs` 仅分析；架构无关部分在 H100(sm90a) **直接 include 真实头文件跑真实设备代码**，
  与独立 host oracle 对拍 **30 PASS / 0 FAIL**（反序环累加逐位 0/16384、bf16 哈希逐位 0/4096、
  升序朴素 922/4096 bit 不同）。配套博客
  `content/posts/mega-moe-72-bucket-allreduce-reduce-scatter`（索引页新增第 72 篇）。
  上一轮 P61-1：新建 `docs/71-deepep-hybrid-dispatch-combine.md`（+`src/deepep_hybrid.cu`、
  `harness/exp_deepep_hybrid.{sh,py,out.txt}`）——补上 DeepEP 的**两级（hybrid = scale-out RDMA ×
  scale-up NVLink）EP dispatch/combine**（`hybrid_dispatch.cuh` 674 行 + `hybrid_combine.cuh` 627 行），
  即 Mega MoE 在 EP8/EP16 上真正跑的那条路（第 7/48/61 篇讲的都是 `num_scaleout_ranks==1` 的单级）。
  数据面（`gin.put/get_sym_ptr`、跨 rank barrier、TMA）走 NCCL GIN 设备 API，本机 2.29.7 不提供 →
  **仅分析**；但四个**架构无关内核**在 H100 上**直接 include 真实头文件跑真实设备代码**——计数就绪
  哨兵 `encode_decode_positive`（`-c-1` 对合，区分「未写=0」与「写了 0」）、tail 增量 `pack2/unpack2`
  （finish 低 32 / tail 高 32）、`do_psum` 含端点/排他前缀和、forward warp 的 round-robin 位图分片调度；
  **设备 7 PASS / Python oracle 6 PASS / 0 FAIL**。dispatch 三角色（notify 计数汇聚 + 前缀和 / scaleout
  RDMA 收发 / forward NVLink 转发 + 元数据）、combine 两角色（scaleup 链式列表 / forward 重放 + 转发）
  全部逐段精读。配套博客 `content/posts/mega-moe-71-deepep-hybrid-dispatch-combine`（索引页新增第 71 篇）。
  上一轮 P60-1：新建 `docs/70-deepgemm-sanitizer.md`（+`src/sanitizer_negctl.cu`、
  `harness/exp_sanitizer.{py,sh,out.txt}`）——补上测试栈的**第四个维度：内存安全 / 同步正确性**
  （前三层是第 40 篇计时判对、第 41 篇生成器、第 43 篇逐位断言，都在问「结果对不对」）。
  它不是一个测试用例，而是**驱动器** `tests/test_sanitizer.py`：把 `tests/` 下每个 `test*` 放进
  `compute-sanitizer`、**独立子进程**重跑（sanitizer 命中精确错误会销毁 context，JIT/singleton
  状态也不可跨用例复用）。拆透**六个 env + 六个 flag 的「为什么」**：`DG_JIT_CHECK_NO_SPILLS`
  在 `backend.hpp:120-122` 把 ptxas 的 "spilled to local memory" 警告**升级为断言失败**；
  `DG_USE_NVIDIA_TOOLS`（`bench.py:87-90`）跳过与 sanitizer 冲突的 kineto；
  `DG_USE_TEMP_CUBLASLT_WORKSPACE`（`runtime.hpp:45-48`）躲开 driver 先卸载的
  `cudaErrorCudartUnloading`；**最关键**的 `PYTORCH_NO_CUDA_MEMORY_CACHING`——缓存分配器会把越界
  藏进同一大段，memcheck 视为合法；`--kernel-name-exclude kns=nvjet` 用 mangled 名子串排掉
  cuBLASLt 预编译核。**H100 真机 30 PASS / 0 FAIL**：负样本控制证明这把尺子有效（`oob` 3 条
  `1 bytes after the nearest allocation of size 1024 bytes`；`misalign` 130；synccheck `sync` 64；
  **racecheck** `race` 1），并坐实两个反直觉点——缓存分配器**屏蔽**越界（`mask_big=0` vs
  `mask_exact=3`）、`synccheck` **抓不到数据竞争**（只有 racecheck 抓，而它不在默认 tools）；
  真实 `test_sanitizer.py` 驱动 `test_hc_prenorm_gemm`/`test_sf_layout_kernels` 在
  memcheck+synccheck 下**全 0 errors**。盲区：不跑 racecheck/initcheck、排除
  `test_mega_moe`/`test_locality_domain`。配套博客 `content/posts/mega-moe-70-deepgemm-sanitizer`
  （索引页新增第 70 篇）。
  上一轮 P59-1：新建 `docs/69-deepep-engram-fetch.md`（+`src/deepep_engram_fetch.cu`、
  `harness/exp_deepep_engram.{sh,py,out.txt}`）——补上 DeepEP **第四类通信原语 Engram**：
  **带索引的随机远程读（indexed RDMA gather）**，把「外部键值存储 / 记忆库 / 长上下文缓存」
  按全局索引从主人 rank 拉回本地、每层一个完成钩子。三个核心设计：① **数据面走 RDMA / SF 面走
  NVLink**（entry 本体只在主人 rank → `gin.get`；FP8 block-scale 每 rank 有副本 → 对称指针
  `get_sym_ptr<ncclTeamTagLsa>` 直读）；② **发射/等待分离**（`fetch()` 发所有层请求、返回每层
  hook，用哪层才 `hook()` 等）；③ **请求聚合**（同 peer 攒 32 一批发 `AggregateRequests`、攒够
  `flush_depth` 才敲一次门铃——瓶颈在 MPPS 不在字节）。布局层 `EngramLayout`（storage/recv 字节
  偏移、SF 行号/分片号）+ `EngramWorkspace`（`issue_counters[128][2048]`）；发射面
  `EngramIssueWarpLayout` 把 warp 切成 (peer, QP, tile)，issue 循环用 `__ballot_sync`/`__popc`
  保序压入 64 深环形队列（多出一项恰好绕回翻转后的新队头，永不需要 memmove）；门铃前用
  `issue_counters` 做到达计数。**H100(sm90) 真机**逐行抽取布局/发射/请求生成/SF 拷贝跑真实设备
  代码 + Python 独立复刻两个 host 公式：**设备 PASS=5 / Python PASS=32 / FAIL=0**（256 条索引含
  43 条 OOB → 213 条请求 miss/extra/值不符全 0；SF 两种目的布局各 1024 元素 0 不符；溢出回退
  `(33, 32768, True)`）。真实 RDMA 与多 rank 端到端需 NCCL GIN 设备 API（README 点名
  `ncclGinOptFlagsWarpGet`），仅分析。配套博客 `content/posts/mega-moe-69-deepep-engram-fetch`
  （索引页新增第 69 篇）。
  上一轮 P58-1：新建 `docs/68-deepep-pp-send-recv.md`（+`src/deepep_pp_send_recv.cu`、
  `harness/exp_deepep_pp.{sh,py,out.txt}`）——补上 DeepEP 通信设施里**第三类原语**：
  **流水线并行的相邻 rank send/recv**（前两类是第 7/58/61 篇的 EP all-to-all、第 64 篇的
  bucket 集合通信）。它服务「把一层激活/梯度在流水线相邻 stage 间前后传」这一步，没有计数/前缀和/
  算法选型，关键是**几何与协议**。三块：① **host 契约**（`deep_ep/buffers/pp.py:12-67`、
  `csrc/buffers/pp.hpp`）——`num_max_tensor_bytes` 按 `kNumRDMAAlignmentBytes=64` 对齐、
  `num_storage_bytes=align(4·T·F, 2MiB)`（4 段来自「收/发 × 前/后邻居」）、send/recv 断言
  「相邻 rank + 32B 对齐 + 字节数 32 倍数」（`pp.hpp:83-88/:100-105`）、`get_num_qps=min(2,allocated)`
  （`pp_send_recv.hpp:20-23`）、**cooperative launch + 单 warp**（`:32-53`，因 kernel 要
  `this_grid().sync()`）；② **数据面 `pp_tma_copy`**（`pp_send_recv.cuh:27-102`）——可复用的
  **多 SM TMA 1D 两级流水**：把 `num_bytes` 切成 32B 块、每 SM 分连续一段（`:50-55`），smem 切
  stage+mbarrier（`:37-41`），填流水 `tma_load_1d`+`mbarrier::complete_tx`（`:65-73`），稳态
  「等本 stage load→`tma_store_1d`→commit→`tma_store_wait`→发下一 load」（`:75-98`）、收尾排空
  （`:101`）；③ **寻址面 + 协议**——`get_pp_buffer_offset`（`:20-25`）用 `next=(src+1)%N` 编码方向，
  返回 `(local_idx, peer_idx)`；**4 段对称缓冲**（0/1=收 prev/next，2/3=发 prev/next）与
  `send_count[2]/recv_count[2]/arrival[2][2]/release[2]`（`layout/pp/workspace.cuh`）索引由
  `peer_idx+2`/`local_idx` 给出（表见 docs）；**两跳协议** `pp_send_impl`（`:106-171`：
  背压等 `release[peer]>=count-F+1`（`:133-141`）→ TMA 到本地 staging（`:142`）→ `grid sync`
  （`:144`）→ 按 QP `NCCLGin::put` + `ncclGin_StrongVASignalAdd` arrival（`:146-170`））与
  `pp_recv_impl`（`:175-234`：等 arrival（`:203-216`）→ TMA 拷出（`:218-220`）→ `grid sync`
  （`:221`）→ 回 release（`:224-233`））。**H100(sm90) 真机**：逐行抽取 `get_pp_buffer_offset`
  与 `pp_tma_copy`（片段 sha256 `c3a3b0d1817b3bcb`/`ef702e38dd6caa81`，与 `DeepEP@93eb6eb` 一致）
  直接 include 真实头文件跑真实设备代码——`N=2..8` **168 个有序对**的 12 个导出字段**逐字段 0 不符**、
  每 rank 4 段缓冲 `{0,1,2,3}` 被 recv×2/send×2 恰好各占一次、`put` 对称目标 == 接收方 recv block、
  TMA 搬运 5 种 `num_sms`×8 种字节数 **40/40 byte-exact**（真机坑：96KB 动态 smem 必须先
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 否则 launch 静默失败——初版 40 组
  `mismatch` 全是 dst 保持初值）；`PPSignals` `sizeof=192/alignof=64`（偏移
  `barrier=0/send_count=64/recv_count=80/arrival=96/release=128`）；`BarrierSignals=64B`；
  Python oracle 逐行跑 `tests/pp/test_pp.py:19-50` 的 `generate_stress_ops`
  （sha256 `56d5e37722ddc497`）6 组 `(R,S,F)`×4 种子，验证「第 i 个 send 严格晚于第 i-F 个 recv」
  + `sends==recvs` + ops 全序；缓冲几何 `4096×7168×2 bf16`：`F=4`→896 MiB、`F=16`→3584 MiB。
  **设备 PASS=4 / Python PASS=6 / FAIL=0**。GIN 第二跳（`put`/`signal`，`comm/handle.cuh:171-217`）
  与真实多 rank 端到端为 NCCL 设备 API 专属，仅分析。配套博客
  `content/posts/mega-moe-68-deepep-pp-send-recv`（索引页新增第 68 篇）。
  上一轮 P57-1：新建 `docs/67-mega-mhc-test-oracle.md`（+`src/mhc_sf_row_idx.cu`、
  `harness/exp_mega_mhc_test.{py,sh,out.txt}`）——补上 Mega mHC 一直缺席的**端到端验收测试**
  `tests/test_mega_mhc.py`（第 5 篇拆了实现、第 51/57 篇拆了 Mega MoE / Mega Gate 的测试）。
  三个核心事实：① **SF 页布局是 Mega MoE 共享专家的 4×32 交错页**——测试 `extra_sf_rows:96-104`
  与 kernel `get_shared_sf_row_idx`（`epilogue/sm100_mega_mhc.cuh:116-122`）**逐字等价**
  （`idx&~127`/`(idx&31)*4`/`(idx>>5)&3` 四段一一对应），`src/mhc_sf_row_idx.cu` 直接 include 真实
  头文件跑真实设备代码，5 个 block_m×6 个 T **mismatch=0 且页内单射**；② **两条判据分工**——
  `calc_diff`（`y_fp8` 2e-4 / 其余 5e-5，把「实现误差」与「FP8 量化/bf16 舍入」分开）保证与 torch
  参考一致，30 次重复的 `assert_bitwise_equal` 保证 kernel 自身确定性（确定性模式 split 固定=16）；
  ③ **`test_mega_mhc_api_contract` 四条运行期不变式**（确定性逐位复现 / CUDA Graph 必须先预热否则
  `CaptureStatus::None` / FP8-only 调用逐位稳定 / 多 stream 独立），根因是 split barrier **按 CUDAStream
  分表**（`mega_mhc.hpp:26-39`）。**主 kernel SM100 专属，仅分析**；参考实现 / SF 页布局 / 判据 /
  host 契约架构无关，**H100 真机设备 32 PASS + Python 45 PASS / 0 FAIL**——host 契约 9 条逐条命中
  `mega_mhc.hpp` 真实行号（`:118/:120/:121/:122/:123/:141/:204/:210`），最后的边界是「合法调用撞
  `:223` arch 闸门」。配套博客 `content/posts/mega-moe-67-mega-mhc-test-oracle`（索引页新增第 67 篇）。
  上一轮 P56-1：新建 `docs/66-deepep-symmetric-memory-alloc.md`（+`src/deepep_symmetric_alloc.cu`、
  `harness/exp_deepep_symmetric_alloc.{sh,py,out.txt}`）——补上前面所有 DeepEP kernel 篇目（第 7/48/51/
  52/53/58/59/61/63/64）脚下的**「对称内存分配地基」**：`csrc/kernels/comm/symmetric.hpp` + `allocator.py`。
  这块内存是所有 dispatch/combine/bucket 的基础，却从没系统讲过。核心「为什么」：① 三层对齐里
  **分配粒度 2 MiB** 必须整除 CUDA VMM 的 recommended granularity（`compiled.cuh:67-71`）；
  ② `ElasticSymmetricMemory` 把**GPU 主段和独立 storage 段 map 进同一段连续 VA**，因为设备端只用
  「一个基址 + 对称偏移」寻址；③ `HybridElasticSymmetricMemory` 把域内对端 rank 的 storage shard
  按序贴到 `num_sym_bytes + i*storage`，用 `pidfd_open`/`pidfd_getfd` 跨进程拿 fd；④ `create_shared_handle`
  导出 POSIX FD 后 **`cuMemRelease` 分配句柄、靠 fd 续命**。**纯 host-side CUDA VMM + NCCL 主机 API，
  H100 真机可跑**（直接 include 真实头文件）。**设备 PASS=55 / Python PASS=23 / 0 FAIL**：真坑是
  **属性说支持 FABRIC 但 `cuMemCreate` 被拒 → 回退分支清位到 0x1**；另做了真·跨进程 pidfd 共享
  （子写父读/父写子读，且踩过「不能 fork-after-CUDA、要 fork+exec」）；Python 侧加载真实 `allocator.py`
  验证「meta 占位 + 2MiB 对齐偏移 + `swap_tensors` 落地」。配套博客
  `content/posts/mega-moe-66-deepep-symmetric-memory-alloc`（索引页新增第 66 篇）。
  上一轮 P55-1：新建 `docs/65-deepep-refs-and-testing.md`（+`harness/exp_deepep_refs.{py,sh,out.txt}`）
  ——补上前面所有 DeepEP kernel 篇目（第 7/48/51/52/53/58/59/61/63/64）的**语义规范层**：
  `deep_ep/utils/refs.py`（dispatch/combine 纯 PyTorch 参考实现）+ `gate.py`（精确不平衡流量生成器）
  + `testing.py`（bench/bench_kineto/parse_num_bytes）+ `math.py`（`calc_diff` 是余弦相似度型，
  与 DeepGEMM 的归一化平方误差不同）。核心「为什么」：DeepEP 的 dispatch **不做 ragged all-to-all**，
  而是用「mask + 定长 `all_to_all_single`」精确复刻——同一 token 因 top-k 专家分散会**发给多个 dst**、
  每份把不属于该 dst 的槽置 `-1`，接收侧再按 `src_token_global_idx` 升序（这就是全 DeepEP 的坐标约定）；
  combine 的 `grouped_reduce` 用 **stable sort + 组内严格升序累加 + 结果落组最右槽**，三种降约配置
  hybrid/non-hybrid/disabled 与 bias 语义（scaleout==1 时 T/F 仍因 bias 不等价）由此定义。
  **H100 真机**（绕过 `import deep_ep` 的 `check_nccl_so`：`sys.modules` 注入合成包 + stub envs，
  按文件路径 exec 真实源码并打印 sha256）：8 卡 `refs.dispatch` 真实 all_to_all 与**独立 oracle
  逐项 PASS**（升序/src/topk_idx/x 逐位/计数/OOB=-1）；`refs.combine` 五配置与 oracle **逐位相等**；
  `gate.py` precise 主扫 **98/98 PASS**；`testing.py` 5/5 PASS。配套博客
  `content/posts/mega-moe-65-deepep-refs-and-testing`（索引页新增第 65 篇）。
  上一轮 P54-1：新建 `docs/64-deepep-bucket-collectives.md`（+`src/deepep_bucket_copy_engine.cu`、
  `harness/exp_deepep_bucket.{sh,py,out.txt}`）——补上 DeepEP 里**与 EP all-to-all 并列的另一套通信设施**
  `BucketBuffer`（`all_gather`/`reduce_scatter`/`all_reduce`），MoE 训练同步专家梯度/优化器状态的原语。
  最大亮点：**纯 NVLink all-gather 不用任何 SM kernel**，完全交给硬件 copy engine——`copy_engine_push =
  cudaMemcpyBatchAsync`（`srcAccessOrder=Stream`+`PreferOverlapWithCompute`、**必须非 legacy 流**）、
  `copy_engine_sync/wait = cuStreamBatchMemOp(WRITE_VALUE_64 + WAIT_VALUE_GEQ)`（GPU 侧跨 stream 屏障、
  255/批切分），信号是 `CopyEngineSignals.base[kNumMaxNVLRanks]`、per-context `seq_idx` 单调计数（GEQ 故
  不要求同值），一遍「到达屏障 → push 我的 shard → 完成屏障」即可；RDMA/hybrid 用 `ChunkIterator`
  （固定 stride 恰好覆盖每个 chunk、仅末块可不满）负载均衡分块，hybrid all-gather 还有显式代价模型自选
  chunk 尺寸；`get_theoretical_num_sms` 是解析式带宽模型（每 SM 读/写 180/45 GB/s、nvlink×0.9）。**H100 真机**：
  copy-engine 原语、`ChunkIterator`（include 真实头文件跑设备代码）、真实编译 `sizeof`、带宽模型全部真跑；
  并靠 `cudaDeviceEnablePeerAccess` 做了**真·多 GPU（每 rank 一块 H100）跨 NVLink peer 的 all-gather，
  R=2/4/8 逐字节一致**；单 GPU 多 stream 模拟时 R≥8 互锁，反证「重叠的正交性来自真实并发的执行单元」。
  all_reduce/reduce-scatter 内核用 `ncclDevComm`/`ncclGin`/`multimem`（本机 NCCL 2.29.7 无 GIN）仅源码分析。
  **设备 PASS=20 / Python PASS=21 / 0 FAIL**。配套博客 `content/posts/mega-moe-64-deepep-bucket-collectives`
  （索引页新增第 64 篇）。
  上一轮 P53-1：新建 `docs/63-deepep-comm-barrier.md`（+`src/deepep_comm_barrier.cu`、
  `harness/exp_deepep_comm_barrier.{sh,py,out.txt}`）——补上 DeepEP 里每个 dispatch/combine/bucket
  kernel **开始与结束**都调用、却从没单独拆过的 **`comm::barrier` 同步地基**。三层：① **对称窗口句柄
  `NCCLGin`**（`handle.cuh:11-255`）——`get_sym_ptr`（`:64-92`）把「本 rank offset」翻译成「目标 rank
  同 offset」（对称 = 布局一致），`red_add_rel`（`:96-120`）统一「能 NVLink 就不走 RDMA」；② **屏障协议**
  （`barrier.cuh`）——`BarrierSignals` 计数 + 双相位槽，NVLink 腿 `nvlink_barrier_wo_local_sync`
  （`:88-131`）用「计数器低 2 位选相位/符号 + 对双槽 ±1」实现**免复位**（四调用一循环、两槽自动回 0）、
  `timeout_while`（`:30-54`）看门狗、`get_qp_mode`（`:56-86`）的 SM↔QP 确定映射；③ **两级域编排
  `gpu_barrier`**（`:196-252`）——先 flush TMA store → grid sync → SM0 scaleup(NVLink)、其余 SM 并行
  scaleout(RDMA)。host `context.cpp` 域分解（`:174-183`，仅 hybrid 且 scaleup 域==NVLink 域时
  `is_scaleup_nvlink`）与 `ginSignalCount=num_ranks+4`（`:140`）解释状态预算。**H100 真机**：NVLink 腿逐行
  抽取 + 对称指针桩多 block 模拟 2~128 rank × 8 次调用（counter/双相位槽与四相位模型**逐字段 0 不符**，
  奇数次非零残值坐实 ±1）+ SM gating + `get_qp_mode` 5 组 + `BarrierSignals` 64B——**设备 19 PASS /
  Python 17 PASS / 0 FAIL**；scaleout RDMA 腿与并行混合编排为 NCCL GIN 专属，仅分析。配套博客
  `content/posts/mega-moe-63-deepep-comm-barrier`（索引页新增第 63 篇）。
  上一轮 P52-1：新建 `docs/62-deepep-deterministic-sort.md`（+`harness/exp_deepep_deterministic_sort.{py,sh,out.txt}`）
  ——补上 DeepEP dispatch 的**「确定性」侧**：dispatch 的槽位由对称缓冲里的 `atomicAdd` 计数器分配
  （`dispatch.cuh:341`），同一批 token 多次运行的物理行序可变；`EPBuffer(deterministic=True)` 时
  dispatch 结束即挂 `EPHandle.deterministic_sort`（`ep.py:734-743`）把顺序规范化回**源 token 全局索引序**，
  使整个 MoE 前向逐位可复现。要点：① `EPHandle` 冻结路由（`topk_idx` 有 `_version` 守卫、
  `psum_*`/`num_unaligned_*`/`recv_src_metadata` 是排序与反向的输入）；② 排序键恒取自「排序前快照」
  （cached dispatch 不重生成元数据）；③ 非 expand：全局按 col0 排序 + OOB 尾块置 dtype-max + 非 cached
  才排元数据 + 逆置换更新 `channel_linked_list`；④ expand：`expert*1e10 + (-5e9 + src)` 双键稳定排序
  （有效行 < 填充行、块内按 src 升序）、`scatter_add_` 构键、只排数据张量 + 槽位指针逆置换；cached expand
  整个跳过。**方法纯 host/Python、架构无关**，故 H100 上 `ast` **逐行抽取真实源码运行**（绕过本机拒绝
  `import deep_ep` 的 `check_nccl_so`；片段 sha256 `22f3026c…` 与 `~/github/DeepEP@93eb6eb` 一致），
  独立 oracle 对拍 **7 PASS / 0 FAIL**（含「6 种接收顺序输出逐位一致」这一确定性核心）。配套博客
  `content/posts/mega-moe-62-deepep-deterministic-sort`（索引页新增第 62 篇）。
  上一轮 P51-1：新建 `docs/61-deepep-dispatch-copy-epilogue.md`（+`src/deepep_dispatch_copy_epilogue.cu`、
  `harness/exp_deepep_dispatch_copy_epilogue.{sh,py,out.txt}`）——补上 DeepEP dispatch 的**输出落盘侧**：
  主 dispatch kernel 把 token 搬进对称缓冲后，由**独立的第二 kernel** `dispatch_copy_epilogue_impl`
  落盘成下游张量，是第 48 篇 combine 归约 epilogue 的**对称另一半**。**关键：整段不碰 NCCL 设备 API**，
  故 H100 真机**直接 include 真实头文件跑真实设备代码**。要点：① 两 kernel 靠 **PDL**
  （`cuh:61` `cudaGridDependencySynchronize`）衔接，读前序数据不得用 `__ldg`；② 三种纯结构体
  `TokenLayout`（`<false>=256/<true>=288`，差一个 32B mbarrier）/`BufferLayout`/`EPWorkspaceLayout`；
  ③ **一个 kernel 覆盖五种形态**（四个模板布尔开关）——非 expand / expand 非 cached（原子分配）/
  expand cached（读缓存 metadata，逐位可复现）/ 零填充（`st_bulk` + warp 扫描定位 padding 行）/
  链式列表（写槽 + 尾哨兵 + 清 `channel_scaleup_tail`）；④ SF 两种 stride。**H100 真机 21 PASS / 0 FAIL**
  （布局偏移逐字段、非 expand 全部输出、expand 原子分配 + 零填充、cached 不写计数器、链式列表），
  另有 Python oracle 独立复刻布局与零填充公式（含 `__ffs` 取最低满足 lane 这个关键点）。配套博客
  `content/posts/mega-moe-61-deepep-dispatch-copy-epilogue`（索引页新增第 61 篇）。
  上一轮 P50-1：新建 `docs/60-mqa-logits-test-oracle.md`（+`harness/exp_mqa_logits_test.{py,sh,out.txt}`）
  ——补上 DeepSeek 稀疏注意力 indexer 第一级打分「算什么对」的**唯一规范** `tests/test_attention.py`
  （715 行），也是第 19/22 篇里 TFLOPS/GB/s 数字的出处。**与 Mega MoE/Mega Gate 不同：MQA logits 的
  SM90 用例（`fmt='fp8'`、fp32 logits）在 H100 上真的能跑**（只有 mxfp4/mxfp8 与 sparse 被 `@test_filter`
  跳过）。要点：① 连续版 `ref_mqa_logits:78-101`（按 head 加权**先 relu·w 再对 head 求和**、KV 分块只改
  显存、`-inf` 掩码在 `[ks_i,ke_i)`、`cost` 计数）+ kernel **压缩输出** `[seq_len,max_seqlen_k]`（写回偏移
  `-seq_k_start`，`sm90_fp8_mqa_logits.cuh:307-310`）；② 判据两层——`-inf` 掩码逐位 + 20 次逐位自洽 +
  `calc_diff` vs 原始参考（fp8 `1e-3`）/ vs **隔离量化**参考（`5e-6`），把「实现误差」与「FP8 量化误差」
  分开；③ 分页版三种 KV cache 融合布局（SM90 E4M3+每 token fp32 scale、SM100 MXFP8/MXFP4+packed UE8M0）
  与 `to_mqa_weights` 的每行 TMA 16B padding；稀疏版逐位对齐全量 gather。**H100 真机 42 PASS/0 FAIL**：
  直接 import 真实 test 复用 helper 并调用真实 `test_mqa_logits`/`test_paged_mqa_logits`（跑出连续
  100/191 TFLOPS、分页 46/122 TFLOPS），oracle 每一项（参考数学 vs 朴素三重循环 1.11e-16、CP 分片、
  掩码、cost、三种布局逐位、判据工具）都独立复核。配套博客 `content/posts/mega-moe-60-mqa-logits-test-oracle`
  （索引页新增第 60 篇）。
  上一轮 P49-1：新建 `docs/59-deepep-ptx-primitives.md`（+`src/deepep_ptx_primitives.cu`、
  `harness/exp_deepep_ptx_primitives.{sh,out.txt}`）——补上 **DeepEP 所有 kernel 共用的设备端公共层**
  `common/ptx.cuh`（612 行）+ `math.cuh`/`utils.cuh`/`layout/common/barrier.cuh`；第 7/11/53/58 篇里
  每一行 `mbarrier`/`cp.async`/`TMA`/`red.release.sys`/`__match_any_sync` 都源于此却从没单独拆过。
  八组原语（mbarrier 五原语 + `cp.async.mbarrier.arrive` 的「发射+1/完成−1」计数模型 ⇒ dispatch 的
  mbarrier init count=1；TMA 三条搬运路径含 `cp.reduce.async.bulk…add.f32`；NVLink 多播 `multimem.*`；
  谓词化访存 + `iterate_ld_st`；`gpu`/`sys` 两档内存序；`warpgroup_reg_realloc`；warp 内建；浮点打包
  `fadd4`/`fmul4`/`accumulate`/`cvt_rs_bf16x2`）+ `encode_decode_positive`（未就绪=0、已就绪=负数）
  与 `BarrierSignals`。**H100 真机 41 PASS/0 FAIL**：直接 include 真实头文件跑真实设备代码——
  mbarrier 相位翻转、TMA 载入/归约写、`cp.async` 计数、谓词裁剪、2000 轮 release-acquire 坏数据=0、
  warp 内建逐项、`setmaxnreg` 实跑；`multimem`/`cvt.rs` 仅语义 oracle（实证 `cvt.rs` sm90 被拒/sm100 通过）。
  配套博客 `content/posts/mega-moe-59-deepep-ptx-primitives`（索引页新增第 59 篇）。
  上一轮 P48-1：新建 `docs/58-deepep-combine-reduce.md`——补上 `docs/07` 只一笔带过的
  **DeepEP combine 归约侧**（`combine_reduce` 家族）：专家算完后「把结果加回源 token」这一步的逐行实现，是 EP 归约侧的落点，
  与 `docs/11` 的 DeepGEMM 自研 combine 成对照。三个核心事实：① **两阶段 + PDL**——主 combine kernel 做搬运+
  本地归约，`combine_reduce_epilogue_impl` 可**延迟**到 compute stream 做跨 rank 求和（host 包 lambda
  `ep.hpp:925-950`），靠 `cudaGridDependencySynchronize`（`:57-59`）与主 kernel 建立顺序，故 epilogue 里读
  metadata **不能用 `__ldg`**；② **rank/topk 双布局**由 `use_rank_layout = allow_multiple_reduction &&
  num_ranks <= num_topk` 决定（`combine_utils.cuh:8-18`），主 kernel 与 epilogue **共用同一 constexpr**；
  ③ **top-k 去重 + bf16 直加旁路**——`ptx::deduplicate`（`__match_any_sync`+`bfind.u32` 取组内最高 lane）+
  `compute_topk_slots`（`__ffs` 从低位挤前、`-1` 占位、`ldg_with_ge_pred` 谓词跳过越界），`enable_hadd_bypass`
  在无 bias 且有效槽 ≤2 时走 bf16 直加（省两次类型转换），否则 fp32 累加一次舍入。**H100 真机 16 PASS / 0 FAIL**：
  **直接 include 真实头文件编译运行真实设备代码**——rank/topk 两套布局的完整 `combine_reduce_epilogue_impl`（真实
  对称 recv_buffer）输出与 top-k 权重 vs 独立 oracle **逐位 0 不符**（含同 rank 重复 token 的去重）；20 万随机
  bf16 值对坐实「≤2 槽时 bypass==通用逐位一致」、20 万三元组「链式 bf16 有 22.10% 会偏」——这正是旁路卡在上界 2
  的原因；Python 独立 oracle 重写 constexpr + 去重选择逻辑 ALL PASS。配套博客
  `content/posts/mega-moe-58-deepep-combine-reduce`（索引页新增第 58 篇）。
  上一轮 P47-1：新建 `docs/57-mega-gate-test-oracle.md`（+`harness/exp_mega_gate_test.{py,sh,out.txt}`）
  ——补上 Mega Gate 一直缺席的「验收测试即正确性规范」：`tests/test_mega_gate.py` 定义了
  「Mega Gate 的输出算什么对」。**主体 kernel SM100 专属，仅分析**（`mega_gate.hpp:159` arch==10，
  H100 上两个用例整体被 `@test_filter(get_arch_major()==10)` 跳过）；**路由 oracle 纯 torch，H100 真机实测**。
  判据 `check_topk_weights:105-173` 的五个要点：① `sqrt_softplus`（`epilogue:19-23` 的 threshold-20 折衷）
  与参考 `F.softplus(·).sqrt():117` **数值完全一致**（默认 `threshold=20` 同分支）；② 排名带 bias、
  出权重用无 bias 的 `scores` 归一化 `scores/(sum+1e-20)*1.5:137-139`，image token 的 bias 是**替换**非叠加；
  ③ top-k 用**结构无关的 crossing 不等式**验收 `:145-156`（选中最小分 ≥ 未选中最大分，容差
  `1e-5+1e-5·|ranking|max`），与 sort4 网络/3-bit 游标解耦；④ 判据 **`calc_diff<1e-11:141-143`（非逐位）**
  因参考 logits 走 `cublaslt_gemm_nt:98-102`；⑤ logical→physical 逆映射 `p<E?p:p-E:121-128`
  **依赖 `make_physical_map:40-49` 的编号约定**（副本槽=`E+logical`）。H100 实测
  `harness/exp_mega_gate_test.{py,sh,out.txt}` **47 PASS/0 FAIL**：直接 import 真实 test 复用 helper，
  用一个完全正确的 torch 代理 gate 喂 `check_topk_weights` → `weight_diff=0`，再用六类扰动反例证明
  判据能抓错（权重偏 1e-3→`calc_diff 4.995e-7`、偏 1e-6→`4.543e-13` 仍过、换第 K+1 名→crossing 触发、
  重复专家→唯一性触发、忽略 image_bias→53.1% 选择不同、漏归一化→`0.713`）；`sqrt_softplus` vs fp64
  `max rel 1.369e-7`（fp32 舍入）、副本分散均衡 `0.50000`。配套博客
  `content/posts/mega-moe-57-mega-gate-test-oracle`（索引页新增第 57 篇）。
  上一轮 P46-1：新建 `docs/56-tma-descriptor-layer.md`（+`src/tma_descriptor.cu`、
  `harness/exp_tma_descriptor.{sh,out.txt}`）——补上前面 55 篇每个 SM90 kernel 的每一行
  `cp.async.bulk.tensor` 前面那个 128 字节 `CUtensorMap` 的**来源**：
  `csrc/jit_kernels/impls/runtime_utils.hpp` 的 `make_tma_*` 是一层**确定性参数翻译**
  （major→inner/outer 二分、dtype 枚举、swizzle 映射、三个隐藏规整），最终只落到一次
  `cuTensorMapEncodeTiled`。**纯 host、架构无关，H100 真机逐字节验证 32 项 ALL PASS**：
  16 个包装用例与独立直连编码**逐字节相等**；真实 `cp.async.bulk.tensor.2d` 载入与
  `cute::Swizzle<B,4,3>` 预测布局 **0/256、0/512、0/1024 全对**；并暴露一条只有真机才现的约束——
  **swizzle 的 XOR 键取自绝对 smem 地址 bits[7:10)**，故目的地要对齐到整个 swizzle 原子
  （128B swizzle→1024B），否则布局整体平移（smem_off=+128：base=0 预测 512/1024、base=1 预测 0/1024）。
  `base=32` 的 `128B_ATOM_32B` 在 H100 被驱动直接拒（SM100 专属）。上一轮 P45-1：新建
  `docs/55-deepgemm-probe-clock64.md`（+`src/probe_clock64.cu`、
  `harness/exp_probe_clock64.{sh,out.txt}`）——补上一段**公开仓库里没有任何调用点、JIT 也从不启用**
  的休眠观测设施 `deep_gemm/include/deep_gemm/common/probe.cuh`（per-warp `%clock64` 追踪探针）。
  **三处「为什么」**：① **4 GiB 窗口约束的根因**——`advance<N>` 只用**一条 `IADD3` 加低 32 位**
  （伪指针实证 `0x00000001_FFFFFFF0` +16 → `0x00000001_00000000`，高 32 位永不进位），故
  `is_valid_buffer` 才要求整段落在同一 4 GiB 窗口（另两条：容量偶数、16B 对齐）；② **「先 advance
  再负偏移回填」**让 ptxas 保留基址加法、压低打点开销；③ **`mark()` 写 `%clock64`+`%globaltimer`
  两条 8B 记录做挂钟标定**（实测两组独立标定都 **1.98 GHz**）。`DG_PROBE_ENABLE` 默认 0 → `if constexpr`
  整段消除（关闭时 trace 缓冲全零、`advance` no-op）。**全文件架构无关内联 asm，H100 真机同源编
  两个二进制 + numpy 解析记录流 ALL PASS**（记录顺序/单调/回填/标定/安全判据逐条命中；纯记录开销
  ~5.7 ns/条）。上一轮 P44-1：新建 `docs/54-deepgemm-device-math-utils.md`（+`src/device_math_utils.cu`、
  `harness/exp_device_math_utils.{sh,out.txt}`）——补上前面 0~43 篇**每个 kernel 都在调、却从没系统
  讲过**的公共地板：`common/{math,utils,types,packing,compile,exception,cute_tie}.cuh`。
  **除 `fma2` 的 SM100 快路外全部架构无关，H100 真机直接 include 真实头文件跑真实设备代码 ALL PASS**。
  三个核心事实：① **UE8M0 取整用整数进位**——`get_ue8m0_sf_exp` 以
  `(amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits` 的**进位**实现「把
  amax/quant_max 向上取到 2 的幂」（进位 iff amax 尾数超过 1.75/1.5），与 Python
  `ceil_to_ue8m0` **四组合（E4M3/E2M1 × FP32/BF16）0/65536 逐位等价**；`get_ue8m0_sf_inv` 的
  带偏指数与 SF 互为倒数（和 254），故 `SF*SF_inv` 精确为 1，也解释了 `scale_bf16x2_into_fp8x4`
  乘幂次 scale 为何精确；② **归约两种语义**——`warp_reduce` 的 intra=连续 N lane 组内
  （偏移 `2^j`）、inter=lane 同余 mod N（偏移 `N·2^j`），实测 7 组 0/32；`cta_exclusive_sum`
  是块内互斥前缀和（MQA logits 调度器的偏移来源）；③ **工具层**——`get_num_aligned_tmem_cols`
  取整到 {32,64,128,256,512}（第 25 篇 512 列预算）、`get_smem_pack_factor`（FP4 每字节 2 元素）、
  `GemmType` 判定、`fma2`/`fast_rcp`（sm90 回退逐位等于 fused、rcp 误差 ≤1.19e-7）。
  上一轮 P43-1：新建 `docs/53-deepep-eplb-redundancy.md`（+`src/eplb_redundancy.cu`、
  `harness/exp_eplb_redundancy.{sh,py,out.txt}`）——补上专题「EP 反向通信」的空档：
  动态专家复制（MoonEP / UltraEP）下前向 `lb_prefetch_weights` 把主人权重（+scale）推到各 peer
  冗余槽、反向 `lb_reduce_grads` 把副本 FP32 梯度累加回主人。**两个 kernel 用 NCCL 设备 API
  （`ncclDevComm`/`ncclGin`），本机 NCCL 2.29.7 不提供 → 仅分析**；但三层架构无关机制 H100 真跑：
  ① host 解析式带宽模型 `align(ceil(max(nvlink/180, nvlink/45)),2)`（与冗余映射无关，只给 SM 容量上界；
  真机 `nvidia-smi nvlink` 430.3 GB/s → 写侧瓶颈 10 → **10 SMs**）；② 负载均衡切块——`build_tasks`
  用 cub 块内压缩按 owner 过滤映射表、`EPWeightIterator` 用**全 rank 共享一个原子计数器**按
  weight→chunk→task 窃取（**16/16 + 77/77 认领且偏移逐条一致**，逐行原样抽取真实设备代码）；
  ③ 归约语义——reduce 用 `tma_store_reduce_add_f32`（**元素级原子加、顺序无关**）而非覆盖
  （relerr 0 vs 反例 0.782）。`build_tasks`、iterator、host 公式、归约语义全部 H100 实测 ALL PASS。
  上一轮 P42-1：新建 `docs/52-kernel-launch-pdl.md`（+`src/pdl_launch.cu`、`src/pdl_launch_host.cpp`、
  `harness/exp_pdl_launch.{sh,out.txt}`）——补上前面 51 篇都踩、却从没单独讲过的**发射那一下**：
  DeepJIT `Kernel::launch` 用 driver `cuLaunchKernelEx` 装配**至多 3 个 `CUlaunchAttribute`**
  （cooperative / cluster dimension / PDL），`LaunchOptions`（`options.hpp:178-233`）是「全局默认
  → 每 kernel 四字段 override」两级 `std::optional` merge（显式 `false/0` 也算「已指定」），
  DeepGEMM 每 kernel 只填 `num_smem_bytes/grid/block/cluster_dim`、只有 `set_pdl`（`config.hpp:27-31`）
  用户可全局切换。`Kernel::launch` 的 8 个 `has_value` 断言、`MAX_DYNAMIC_SHARED_SIZE` 抬高、
  non-portable cluster「只设不撤」坑、属性数组装配。设备侧 **24 处** `cudaGridDependencySynchronize()`
  （`griddepcontrol.wait`）总在「barrier/cluster 初始化之后、scheduler 消费前序输出之前」，
  而 `cudaTriggerProgrammaticLaunchCompletion`（`griddepcontrol.launch_dependents`）全仓库**一次不用**
  （依赖隐式触发）。**H100 真机**（PDL/cluster/dyn-smem 均 sm90 支持，非 SM100 专属）：PDL 全格
  （132+132）2.028→1.971 ms（1.03×，无空闲 SM）、半格（66+66）2.028→**1.042 ms（1.95×）**、
  提前触发 1.042 vs 隐式末尾 2.027、缺 `griddepcontrol.wait` 静默读 stale **1,689,600/104857600**
  （有则 0）；cluster cs=2/4 bad=0、cs=16 无属性 error 912→设属性成功；100 KiB 动态 smem 无属性
  invalid argument→设属性成功；`LaunchOptions` 真实头文件 host 测试 ALL PASS。行号全部核对过。
  配套博客 `content/posts/mega-moe-52-kernel-launch-pdl`（索引页新增第 52 篇）。
  上一轮 P41-1：新建 `docs/51-mega-moe-test-correctness.md`（+`harness/exp_mega_moe_test.{py,sh,out.txt}`）
  ——补上 Mega MoE 一直缺席的**端到端验收测试 `tests/test_mega_moe.py`**，它是「正确性判据」与
  前面各篇反复引用的「TFLOPS/HBM/NVLink/overlap 数字」的共同出处。三个核心事实：① **逐位判据**——
  `run_fused`（单 mega-kernel）与 `run_baseline`（DeepEP dispatch→`m_grouped`×2→SwiGLU→combine）
  用 **`torch.equal`**（`:345-346`，非 `calc_diff`）验收，配合 `use_deterministic_algorithms(True)`
  （`:77`），因为「通信-计算重叠」只是调度变换、不改变任何一次乘加；② **oracle 依赖 DeepEP**——
  对比被 `is_legacy_loaded`（`:339`）门控，本机 DeepEP 受阻（第 0 篇/阻塞），故端到端在 H100 跑不了，
  本篇只验证测试侧架构无关部分；③ **测试侧三个 SF 布局变换**——`_cast_fp8_for_mega_moe`（`:53-59`）
  量化+TMA 列主序对齐、`_to_shared_mega_moe_sf_layout`（`:36-50`）共享专家的 UTCCP 4×32 交错
  （与 kernel `transform_sf_token_idx` `:145-149` 逐项等价）、`_copy_fp8_sf`（`:62-70`）padding
  广播最后一行。另拆性能账目（`:368-401`：FLOP 系数 3=L1 gate+up+L2、fp4 权重 0.5 字节、
  `t_reduction` 的 6.5e12 经验常数、`overlap_factor` 定义）。**H100 真机 37 PASS / 0 FAIL**（直接
  import 真实 test 复用它自己的 helper；逐位判据用真实 SM90 grouped BF16 kernel；参考数学 fp64
  relerr 2.7e-16）。行号全部核对过。配套博客
  `content/posts/mega-moe-51-mega-moe-test-correctness`（索引页新增第 51 篇）。
  上一轮 P40-1：新建 `docs/50-runtime-lifecycle.md`（+`harness/exp_runtime_lifecycle.{py,sh,out.txt}`）
  ——补上第 48 篇只当「字段容器」一笔带过的 `Runtime` 的**生命周期与资源所有权**。三个核心事实：
  ① **`LazyInit` 的「null factory = 终态」**（`lazy.hpp:21,27`）让 `shutdown()` 只需一行
  `runtime = LazyInit(nullptr)`（`config.hpp:13`）：首次访问才构造、整体替换即析构、之后任何
  `runtime->` 抛断言且 `_C.init` 救不回（它只重建 `jit`）；② **资源归属**——cuBLASLt handle
  默认自建（逃避 PyTorch `getCurrentCUDABlasLtHandle` 的 CPU 开销），自建自毁、托管归 PyTorch；
  workspace **每 CUDAStream 一份 16 MiB**（异步使用不能共享、>16 MiB 触发更差 heuristic），
  `DG_USE_TEMP_CUBLASLT_WORKSPACE` 为 compute-sanitizer 卸载顺序逃生阀；③ **退出顺序**——
  `atexit.register(shutdown)` 在 Py/PyTorch/CUDA 尚存时提前释放，规避「原生 cuBLASLt 退出回调
  先释放存储、再被 C++ 静态析构 `cublasLtDestroy`」的 use-after-free。**H100 真机 24 PASS / 0 FAIL**：
  LD_PRELOAD 探针 6 例 create/destroy/clean 全对、显存账 16/16/32 MiB、shutdown 终态逐条验证、
  上游 `test_lazy_init.py` 3 tests OK。行号全部核对过。配套博客
  `content/posts/mega-moe-50-runtime-lifecycle`（索引页新增第 50 篇）。Hugo 构建 0 ERROR。
  上一轮 P39-1：新建 `docs/49-sm90-wgmma-descriptor.md`（+`src/wgmma_desc.cu`、
  `harness/exp_wgmma_desc.{sh,out.txt}`）——补上前面所有 SM90 kernel **每一行 MMA 都在调、
  却从没单独讲过**的底座（`mma/sm90.cuh` + `ptx/wgmma.cuh`）。三个核心事实：① **64 位描述符**
  （`start_addr>>4`/`LBO>>4`/`SBO>>4`/`base_offset`/`layout_type`）把「一块 swizzle 过的 smem」
  讲给 tensor core，`to_gmma_layout_type` 直接把字节 swizzle 模式编码进 `layout_type`，
  所以 FP8 kernel 的 `make_smem_desc(ptr,1)` 其实是在说「B128」；② **`make_gmma_desc` 的 K/MN
  两分支**给出 LBO/SBO 的闭式（K-major 必须 `kSwizzleMode==BLOCK_K*sizeof`），
  `advance_gmma_desc_lo` 只改低 32 位的三段步进（K 步进 / 多 warpgroup MN 偏移 / 合并 stage 原子偏移）；
  ③ 三个选择器 `M=64`、`K`=FP8 32/BF16 16/TF32 8，`ptx/wgmma.cuh` 的 `arrive/commit/wait<N>` 是
  异步组语义，fp32 累加器 `CLayout_64xN` 是**列主序**的寄存器→(m,n) 映射。
  **H100（sm90a）真机**枚举：**描述符位域 13 项全对 + 5 个真实 WGMMA GEMM 逐位相等**（BF16/FP8/
  MN-major/分两批 `commit`+`wait<1>`）——13 项里包括显式 LBO/SBO、`>>4` 截断、MN-major 的 atom；
  5 个 GEMM 用手工 TMA 128B swizzle 铺 smem，逐位 0 误差。行号全部核对过。配套博客
  `content/posts/mega-moe-49-sm90-wgmma-descriptor`（索引页新增第 49 篇）。Hugo 构建 0 ERROR。
  上一轮 P38-1：新建 `docs/48-runtime-switches.md`（+`harness/exp_runtime_switches.{py,sh,out.txt}`）
  ——补上前面 47 篇每个 kernel 都读、却从没单独讲过的**顶层全局开关面**。两份可变 singleton
  （`Runtime` 管设备资源、`HeuristicsRuntime` 管选核/编译）经 `config::register_apis` 平铺成
  `deep_gemm.set_*`/`get_*`；逐个拆「状态→消费点→为什么」：`num_sms` 是上限且 `0`=懒取设备数，
  改小会在 `is_cublaslt_available` **顺带关掉 cuBLASLt 回退**（第 27 篇三与门第二项）；
  `tc_util==0` 当默认 100；`use_deterministic_algorithms` 是三个正交落点（关 cuBLASLt 回退 +
  禁 Mega Gate split-K + 钉死 mHC splits，皆出于 batch invariance）；`block_size_multiple_of`/
  `mk_alignment` 如何改 `get_best_config` 候选集；`ignore_compile_dims` 经 `get_compiled_dim` 把
  全部编译维置 0（一核多形状）；`set_pdl` 落到 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`。
  另拆 `DG_*` 环境变量的两级前缀解析（`DG_` 优先、`DJ_` 兜底）与安装时固化的 `persistent_envs`。
  **架构无关，H100 真机 25 PASS / 0 FAIL**（profiler 抓 num_sms/确定性选路、`DG_PRINT_CONFIGS`
  抓选核、模板实参抓编译维）。行号全部核对过。配套博客 `content/posts/mega-moe-48-runtime-switches`
  （索引页新增第 48 篇）。Hugo 构建 0 ERROR。
  上一轮 P37-1：新建 `docs/47-deepjit-posthook.md`（+`src/deepjit_posthook.cpp`、
  `harness/exp_deepjit_posthook.{sh,out.txt}`）——补上第 33 篇缓存键「第四段」`post_hook` 一直空着的列表项：
  DeepJIT 唯一的「编译产物发布前注入用户代码」接口。三个核心事实：① **运行时机与作用域**——`nvcc` 出 cubin 后、
  写 meta/提交缓存前，跑 `python <abs hook> <abs cubin>` 并**就地改写**（`cuda/backend.hpp:132-138`），
  相对路径基于 `python_library_root`；② **进键方式**——`FNV1a(相对路径+文件内容)` 且**按路径线程内缓存**
  （`cuda/options.hpp:141-156`），所以「改 hook 要重启进程」「跨进程才区分内容」；③ **override 语义**——
  `nullopt` 继承而非关闭、空串仍算已配置，默认设了 hook 后无法 per-kernel 关闭。再由「hook 在 meta/commit
  之前」的顺序推出三条实证：hook 改的 cubin 即最终产物、失败 hook 不发布、缓存命中不重跑 hook。另补齐
  `Runtime<Backend>` 的 CUDA↔Ascend 抽象（CompilerInfo 哈希、两步编译、dump、发射参数；`post_hook` 是
  **CUDA 专属**，Ascend 恒空哈希）与 Python API（`get_jit()` + 两个 WIP 特性当前不可用）。**H100 实测**——
  直接 `g++` include 真实 DeepJIT 头文件、链接 torch 跑真实代码，8 组 ALL PASS。行号全部核对过。配套博客
  `content/posts/mega-moe-47-deepjit-posthook`（索引页新增第 47 篇）。Hugo 构建 0 ERROR。
  上一轮 P36-1：新建 `docs/46-gemm-major-accumulation.md`（+`harness/exp_gemm_major_accum.{py,sh,out.txt}`）
  ——把前面各篇一直当术语用的 **GEMM 入口契约**单独拆透：`nt` 是唯一实现，`nn/tn/tt` 只是
  「转置操作数（含 SF）后转调 `nt`」的薄壳（转置只改 shape/stride、不改数学）；主序一句话
  `stride(-1)==1 ? K : MN`、输入双主序、输出强制 N-major；**SM90 FP8/FP4 必须 A/B 都 K-major**
  （WGMMA，`fp8_fp4_requires_k_major` 在 arch9 恒真），推论 `fp8 nn` 在 H100 必被拒；packed FP4
  逻辑形状随主序翻倍；`early_return` 统一输出累积（m/n==0 空转、k==0 退化、C==D 原地、C≠D 先拷、
  量化 D 不累积）；`compiled_dims` 随入口 `nk`/`mn`。**H100 实测**：bf16 四入口逐位相等、`nt`
  双主序通吃、`fp8 nn` 被拒 `gemm.hpp:94`、分离 C 与原地 C==D 逐位相等、profiler 抓到主序模板实参
  `(K,K)/(K,MN)/(MN,MN)/(MN,K)` 与编译维。行号全部核对过。配套博客
  `content/posts/mega-moe-46-gemm-major-accumulation`（索引页新增第 46 篇）。
  上一轮 P35-1：新建 `docs/45-muon-syrk-symm.md`（+`harness/exp_muon_syrk_symm.{py,sh,out.txt}`）
  ——补上 DeepGEMM 里**两个从头到尾就是 cuBLASLt-only 的窄接口**：`deep_gemm.batched_syrk`
  （`D=A·Aᵀ`）与 `deep_gemm.batched_symm`（`D=A·B`，A 对称），面向 **Muon 优化器的 Newton-Schulz
  迭代**。它们在第 27 篇的「可回退 / 无条件 / cuBLASLt-only」三分里属于第三类：无自研 kernel、
  不进选核启发式、唯一实现就是 `cublaslt_batched_gemm`。三处设计要点：① **2D↔3D 统一**
  （`get_cublaslt_batched_view:791-794` 把 2D unsqueeze 成 batch=1）；② **放宽 batch 连续性**
  （`get_major_type_ab<false>`/`check_major_type_cd<false>` 跳过 `layout.hpp:19-28` 里
  `stride(0)==rows·cols` 的断言，批 stride 经 `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`
  交给 cuBLASLt）；③ **强制 `CUBLAS_COMPUTE_32F_FAST_TF32`**（`smxx_cublaslt.hpp:45-46`），
  所以 fp32 输入实际只有 TF32 精度。另两个非平凡点：`batched_symm` **内部把 B 转置**
  （`:830`，「A 对称」只是契约不校验）、翻译层用 `(A Bᵀ)ᵀ=B Aᵀ` 交换操作数（`:137-164`，
  `trans_a` 由 `b_major`、`trans_b` 由 `a_major` 决定）。**H100 实测**：真实 Muon 形状
  ×{bf16,fp32}——syrk bf16 峰值 **529 TFLOPS**、典型 ~355–380、fp32 ~180；symm bf16 ~350、
  fp32 ~156；fp32 relerr ~1e-4–2.9e-4 坐实 TF32；2D 与 batch=1 的 3D **逐位相等**；连续/非连续
  batch 都算对；四类非法输入全被 `RuntimeError` 拒；profiler 抓 `nvjet_sm90_*`；相邻
  `einsum("bhd,bhr->hdr")` `calc_diff=3.33e-16`、两次逐位相等、2020 GB/s（K=batch 极小→访存受限）。
  行号全部核对过。配套博客 `content/posts/mega-moe-45-muon-syrk-symm`（索引页新增第 45 篇）。
  上一轮 P34-1：新建 `docs/44-mega-moe-shared-expert.md`（+`harness/exp_mega_moe_shared_expert.{py,sh,out.txt}`、
  `src/mega_moe_shared_schedule.cu`）——补上 Mega MoE 一直缺席的**共享专家（shared expert）分支**：
  DeepSeek 每层在 top-k 路由专家外还有一个**每 token 恒激活**的共享专家，Mega MoE 把它融进同一个
  mega-kernel，它是唯一「不走 dispatch、却走完整 L1/SwiGLU/L2/combine」的通路。**主 kernel SM100 专属，
  仅分析**（kernel `:80` 整段 `__CUDA_ARCH__>=1000`）；**host/调度器/UTCCP 映射/数学架构无关，H100 真机实测**。
  三个结构点：① **共享 L1 无条件在 dispatch 之前发布**（`scheduler:445-452`，因为 `shared_l1_acts ≡ x`），
  天然与通信重叠，共享 L2 按 `is_shared_l2_early`（`:443-444`）决定提前或补尾；② **L1 沿 N 轴变宽
  `2·E_s·I`、L2 沿 K 轴变宽 `E_s·I`**（`:181-195`），因为它是一个更宽的前馈层；③ **共享结果写进 combine
  的 `kNumTopk` 槽、被 fp32 累加器无权重加一次**（`kernel:1287-1288/1431-1433/1449-1450`，对应 test
  `:333` 作 combine `bias`）。另有三处「借道」：零拷贝复用 input buffer（`layout:415`）、两个相位两个
  每域计数器（`scheduler:83-89/421-436`）、恒 FP8 共享权重（`kernel:154`，路由可 FP4）。**H100 实测**：
  直接 include 真实 `scheduler/mega_moe.cuh` 跑设备代码——共享 L1/L2 任务枚举**恰好覆盖一次**
  （32/32、64/64）；constexpr 公式与 `is_shared_l2_early` code vs host 逐项一致；host binding 5 组配置
  逐字段 ALL OK；UTCCP 映射 device vs host 8 个 BLOCK_M 逐点 0 不符且单射；真实 SM90 `bf16_gemm_nt`
  共享前向 vs fp64 relerr **3.447e-3**、combine 语义 **3.448e-3**、反例 **8.747e-1**。行号全部核对过。
  配套博客 `content/posts/mega-moe-44-mega-moe-shared-expert`（索引页新增第 44 篇）。
  上一轮 P33-1：新建 `docs/43-deepgemm-test-utils.md`（+`harness/exp_test_utils.{py,sh,out.txt}`、
  `src/test_utils_sr.cu`）——补上测试栈里 **「比 relerr 更硬的逐位断言」层**：`tests/utils.py` 的 6 个
  辅助是 `test_bf16`/`test_einsum`/`test_fp8_fp4` 共用的**可执行 oracle**。三个最硬性质：① **direct ==
  累积**（`assert_direct_output_matches_fp32_accumulation:20-38`，把同一零张量同时当 D/C 使累积退化成
  `GEMM+0`，fp32 逐位、bf16 再 cast 一次逐位）；② **随机舍入 == 确定性 oracle**
  （`assert_stochastic_bf16_matches_fp32_accumulation:53-79`，纯 torch `:65-78` 复刻 device
  `operators.cuh:35-48` 的 quartet 哈希）；③ **PSUM padding 全零**（`assert_psum_zero_padding:9-17`）。
  另三个是构造/转换：`add_sf_k_padding:41-50`（**UE8M0 NaN `-1` 毒化 padding** 暴露紧凑 pitch 越界读）、
  `convert_to_fp8:82-95`（E4M3 精确表示 E2M1）、`to_cublaslt_vec16_sf_layout:98-110`（MXFP4 per-32 →
  NVFP4 per-16 UE4M3 分块）+ `make_cublas_gemm:113-120`。**H100 实测 27/27 PASS**：把 device 哈希常数
  搬进 `src/test_utils_sr.cu` 后与 torch oracle 随机 bit/输出 **0/262144 逐位不符、CUDA 输出直接喂真实
  helper 通过**；真实 SM90 `bf16_gemm_nt` direct==累积逐位、篡改 1 bit 被检出；真实 SM90 PSUM
  `m_grouped_bf16_gemm_nt_contiguous`（G=4,m=34304,n=6144,k=7168）`calc_diff=0` 且 helper 通过、非零
  padding 被检出；cuBLASLt FP8 relerr 1.39e-6。**环境边界**：NVFP4 端到端因本机 cuBLASLt 返回
  `NOT_SUPPORTED`（`smxx_cublaslt.hpp:89`）而 SKIP（布局本身已逐位验证正确）。行号全部核对过。配套博客
  `content/posts/mega-moe-43-deepgemm-test-utils`（索引页新增第 43 篇）。
  上一轮 P32-1：新建 `docs/42-mega-moe-host-buffer.md`（+`harness/exp_mega_moe_host_buffer.{py,sh,out.txt}`）
  ——补上主 kernel 之前的**整层 host 编排**：Python `SymmBuffer` → C++
  `get_symm_buffer_size_for_mega_moe`（ring 容量）→ `layout::MegaMoEBuffer`（链式字节分配）→
  scheduler `get_num_max_live_pool_blocks`。这层**全纯 host、架构无关**，DeepGEMM 又 pybind 暴露了
  `_C.get_symm_buffer_size_for_mega_moe`，故 **H100 真机逐字段对拍**。三个「为什么」：① Tmax 对齐 1920
  （`mega/__init__.py:28`）让任意候选 BLOCK_M 整除、ring/pool 无零头；② ring 容量**在所有候选
  BLOCK_M 上取 max**（`mega_moe.hpp:56-63`）因 host 此刻不知 kernel 会选哪个 BLOCK_M；③ `x_sf` K-major
  而 `l1/l2_acts_sf` M-major（注释 `:95`，第 10 篇 UTCCP）。四个非平凡点：`x`≡`shared_l1_acts`、
  SF 行字节 `hidden/32` 与 shape `hidden/128` 两口径一致、combine buffer 外层维=`num_topk(+shared)`
  （K 倍放大）、`MegaMoESignals` 固定 21,140,864 B（`:91` offset==128 static_assert）。**H100 实测**
  Python 逐字段复刻结构体布局 + 链式 offset + ring 公式，与真实 binding 对拍：**7 组配置 `num_bytes`
  全 MATCH、全部 view (offset,shape,stride,dtype) 全 OK，ALL PASS**；真实 `SymmBuffer`（gloo ws=1）
  Tmax 512→1920、`shared_l1_acts is x`、小配置复用 OK/大配置被拒；ring 阶梯坐实 `live*BM` 随 BLOCK_M
  单调升、SF ring max 在 BLOCK_M=8。**唯一挡住 H100 的是 `:256/367` 的 `arch_major==10` 派发**。
  行号全部核对过。配套博客 `content/posts/mega-moe-42-mega-moe-host-buffer`（索引页新增第 42 篇）。
  上一轮 P31-1：新建 `docs/41-deepgemm-test-generators.md`（+`harness/exp_test_generators.{py,sh,out.txt}`）
  ——补上前面 0~30 篇每个测试里 **「操作数到底长什么样」** 的来源 `tests/generators.py`（608 行，
  所有 `tests/test_*.py` 都 import）。它是 kernel 操作数契约的**可执行参考**：① 三个基础类型
  （`KernelType`/`MajorTypeAB`/`QuantConfig`，后者的 `get_recipes` 把 legacy 前向给 `None`、wgrad
  给 `(1,1,128)`、B 侧在 `is_fp4_b or is_wgrad` 时退化成 `(1,gran_k)`）；② **operand major 的物理
  编码**（FP8 MN-major=`.T.contiguous().T` 换 stride、FP4 MN-major=`transpose_packed_fp4(x).T`
  重打包、SF 与 major 无关）；③ **三种 grouped 布局**（contiguous 逐行组号 + `-1` padding、
  masked 3D+`masked_m`、PSUM 长度 G 逻辑结束偏移 = `align(prev)+k` 递推）；④ **K 分组**用
  `per_channel_cast_to_fp8`（`(K/gran_k,MN)`）、每组 SF 行独立对齐补零。另发现一个真坑：
  **枚举器会改写全局 mk-alignment**（`set_mk_alignment_for_contiguous_layout` 藏在每次迭代里，
  实测末值 384），故 test 必须先显式设；sm90=128 / sm100=256。**纯 Python 非 kernel，H100 真机**
  八组 **41 PASS / 0 FAIL**：枚举器普查（normal 42/m-contig 16/m-masked 32/k-contig 7/…）、
  major 三条、三种布局语义全对；**四个真实 SM90 kernel** 用生成器操作数对拍 `calc_diff`
  6.7e-4/6.8e-4/6.8e-4/2.2e-4 全部 < `max_diff`。行号全部核对过。配套博客
  `content/posts/mega-moe-41-test-generators`（索引页新增第 41 篇）。
  上一轮 P30-1：新建 `docs/40-deepgemm-testing-harness.md`（+`harness/exp_testing_harness.{py,sh,out.txt}`）
  ——补上前面 0~29 篇**全部数字的来源**：DeepGEMM 的**测试基座** `deep_gemm/testing/`
  （每个 `tests/test_*.py` 都 `import *`）。三件套：① **`calc_diff`**（`numeric.py:5-11`）——实测它
  **恒等于归一化平方误差** `‖x−y‖²/(‖x‖²+‖y‖²)`（升 fp64 保证判据干净），由此得：对称、∈[0,2]、
  `=0` 当且仅当逐元素相等、**同步缩放不变但单侧缩放不不变**（`1−calc_diff = cosθ·2ab/(a²+b²)`）、
  小误差 `≈ rel_l2²/2`（relerr 1e-2↔calc_diff 5e-5）、对 DC 平移敏感；② **`bench`**（`bench.py:7-33`，
  event 计时含发射，`high_precision` 用 `8192²` 大核藏 CPU 发射）vs **`bench_kineto`**（`:79-146`，
  profiler 取**纯 kernel 时长**：`DG_USE_NVIDIA_TOOLS` 旁路、schedule 两批取第二批、256MB `flush_l2`、
  `barrier` 用 `cuda._sleep(2e7)` 藏发射、tuple 分别计时 + 至多一行的断言、ms/us 单位换算）；
  ③ `count_bytes`（递归数 `(data,sf)` 对）与 `assert_bitwise_equal`（逐位 + 首个差异字节定位）。
  另拆 `testing/utils.py`（`get_arch_major`/`test_filter`/`ignore_env`，**两 wrapper 都不 return**）
  与 `utils/dist.py`（`uneven_all_gather` 的**交换长度→pad→all_gather→trim→cat**、MoE 测试拼不等长
  `topk_idx`；`init_dist`）。**架构无关，H100 真机**六组 **ALL PASS**：calc_diff 性质逐条核过；
  `bench` vs `bench_kineto` 小 kernel ~10×、大 `4096³` 冷/热 L2 差 ~34%（300.7 vs 224.8us）；
  `DG_USE_NVIDIA_TOOLS=1` 跳过 profiling；8×H100 `uneven_all_gather` 不等长（`[17,512,4096,3,2048,100,777,64]`）
  拼接正确且各 rank 一致。行号全部核对过。配套博客 `content/posts/mega-moe-40-testing-harness`
  （索引页新增第 40 篇）。
  上一轮 P29-1：新建 `docs/39-legacy-grouped-gemm.md`（+`harness/exp_legacy_grouped_gemm.{py,sh,out.txt}`）
  ——补上第 10~13 篇拆的**现代 C++ grouped GEMM 取代掉的那套 A100 时代纯 Triton 兜底**（`deep_gemm/legacy/`，
  官方注释即「Legacy Triton kernels for A100」，`__init__.py:117`）。核心是**「分组表示」的差异**：
  legacy 用**逐元素索引张量**（`m_indices`/`m_row_indices`/`k_indices`）做**任意 gather**，现代用
  `grouped_layout`（对齐连续段）；两个贯穿机制——**BLOCK 首行定组**（首行 `-1` 整块写零）与
  `M%128==0`/非 gather 侧 `K%128==0` 的对齐。家族 = dgrad 两个（`m_grouped`、A 逐行 gather 的 `a_fused_m`）
  + wgrad 两个（`a_fused_k`/`b_fused_k` 分别 gather A/B，命名坐实）；`-1` 索引即行掩码、空组两种 `ACC` 语义、
  原地累加需 `restore_value=['d_ptr']`、不裁 K 尾块（强制 `k_alignment==128`/`ACC=True`/fp32）。**Triton 架构无关，
  H100 真机**跑真实设备代码 **16/16 PASS**（identity 与 `m_grouped` 逐位相等、`a_fused_k`/`b_fused_k` 逐位相等、
  置换 gather、block 首行 -1 写零、掩码、空组全复现）；性能 m-grouped ~143 / k-grouped ~108 TFLOPS。
  行号全部核对过。配套博客 `content/posts/mega-moe-39-legacy-grouped-gemm`（索引页新增第 39 篇）。
  上一轮 P28-1：新建 `docs/38-deepgemm-ptx-primitives.md`（+`src/ptx_primitives.cu`、
  `harness/exp_ptx_primitives.{sh,out.txt}`）——补上前面 0~27 篇**上层 kernel 全都踩、却从没系统讲过**的
  **裸 PTX 原语层**（`ptx/{ld_st,utils,tma}.cuh`、`common/{tma_copy,ring_pipeline}.cuh`、`comm/barrier.cuh`）。
  Mega MoE 的 dispatch/combine 正确性全系于此：combine 用 `st_rel_sys`/`ld_acq_sys`（**sys 作用域**跨 rank）发布
  readiness（`mega_moe.cuh:1419/1440`）、`grid_sync` 用 `atomic_add_rel`「完成和」+ `ld_acq` 自旋
  （`comm/barrier.cuh:42/45`）、L1→L2 用 `red_xor_rel` 翻转 64 位位图 + `ld_acq_gpu` 读回（`scheduler:157/164`）。
  文档给出**内存序阶梯表**（`ld_acq[_gpu/_sys]`/`st_rel[_sys]`/`atom`/`red`/`red.async`/`ld_volatile`，逐条真实调用点）
  与 `ptx/{ld_st,utils,tma}`、`tma_copy`（`copy_nd` 2~5D + `Load`/`Load2SM`/`LoadMulticast`）、`ring_pipeline`
  （相位翻转位运算）逐段精读。**架构无关，H100 真机**跑真实设备代码 **ALL PASS**：acq/rel 生产者-消费者 0 失败、
  `atomic` 旧值互不相同 1024/1024、RingPipeline ns∈{1,2,4,8} 轨迹逐项 0 不符、`cp.async`/`zfill` 0 不符、
  `ld_evict_first` 流式读 **1279 vs 1129 GB/s**（+13%）、`cvt_relu_bf16x2` vs 独立 RNE **0/65536**。
  行号全部核对过。配套博客 `content/posts/mega-moe-38-deepgemm-ptx-primitives`（索引页新增第 38 篇）。
  上一轮 P27-1：新建 `docs/37-deepgemm-reference-quantizer.md`（+`harness/exp_reference_quantizer.{py,sh,out.txt}`）
  ——补上前面第 25/26/32/35 篇反复引用、却从没系统拆过的**量化语义基准** `deep_gemm/utils/math.py`：
  它是 DeepGEMM 全部 FP8/FP4 kernel 测试的正确性判据（「精确等于参考量化器定义的操作数上的 GEMM」）。
  `ceil_to_ue8m0` 是「>= |x| 的最小 2 的幂」（指数 + 尾数非 0 进位、exp 域钳位 [1,254]，UE8M0 只留 8 位指数）；
  `per_token/per_channel/per_block_cast_to_fp8` 三种粒度（`448`/`6` 常数、`clamp(1e-4)` 地板、补零对齐）；
  `_quantize_to_fp4_e2m1` 的 7 个中点 + `round_up` 标志 = **按 E2M1 编码的 ties-to-even**（无负零）；
  nibble 与 UE8M0 打包均 **LSB-first**；`transpose_packed_fp4` 对合。**纯张量运算非 SM100 专属，H100 真机**
  七组 **ALL PASS**：`ceil_to_ue8m0` 1040 样本 0 不符、FP4 20 万样本 0 不符、端到端 `fp8_gemm_nt` vs
  「参考量化器反量化 fp64 GEMM」relerr 1.4217e-03。行号全部核对过。
  配套博客 `content/posts/mega-moe-37-reference-quantizer`（索引页新增第 37 篇）。
  上一轮 P26-1：新建 `docs/36-sm100-bf16-gemm-tc-util.md`（+`src/tc_util_throttle.cu`、
  `harness/exp_tc_util.{sh,out.txt}`、`harness/exp_tc_util_throttle.{sh,out.txt}`）——补上「稠密 GEMM 主 kernel」
  矩阵的**最后一格**：SM100 BF16 主 kernel `sm100_bf16_gemm`（与第 35 篇同一套 UMMA+TMEM 骨架、去掉 SF/UTCCP，
  `UMMA_K=16`、`BLOCK_K_=64`，host 六入口覆盖稠密/m-grouped/k-grouped/batched），并拆掉它独有的
  **device 内合并 stage**（仅 K-major `Normal` 且 stage≥8，`BLOCK_K` 64→最多 256，少发 `umma_arrive`；
  与第 19 篇 SM90 的 merge 同思想不同动机）与 **tensor-core 利用率控制 `set_tc_util`**。
  后者是本专题一直没见过的机制：一个从 Python（`config.hpp:21-25`）经 `GemmDesc.tc_util` 直通到**模板参数**
  `kTensorCoreUtilControl`（`sm100_bf16_gemm.cuh:35`）的运行时开关，**只有这个 kernel** 实现——
  每次 UMMA 后用 `tensor_core_full_barrier` 等其做完，再按
  `kNumUMMACycles=2·UMMA_M·UMMA_N·BLOCK_K/8192`（`:337`）、`kNumDummyCycles=(100-util)/util·kNumUMMACycles`（`:338`）
  由单 lane `clock64` 空转，把占空比压到 `util%`（注释 `:325`「avoid frequency drop」）。
  **SM100 专属，仅分析**（kernel `:43` 整段 `__CUDA_ARCH__>=1000`）；**架构无关的开关语义与节流机制 H100 真跑**：
  ① `set_tc_util` 语义正确（默认 100、`set(0)→100`、越界在 `runtime.hpp:94` 被拒）；② **对 SM90 bf16/fp8 完全无效**
  （4096³ bf16 在 100/50/10 下都 182 us/753 TFLOPS；fp8 都 107 us/1281 TFLOPS）、**不改 JIT 缓存键**
  （冷缓存 2558 ms → 切 util 后 2.37 ms 命中）；③ 节流机制复刻：公式 `cyc/(cyc+dummy)=util/100` 精确成立，
  真机 `work/total` 随 util 单调下降、墙时间按比例拉长，但空转超调使实测利用率偏低、且 H100 频率不随降载抬升
  （降频意图依赖 SM100 tensor-core 功耗/热行为，无法复现）。行号全部核对过。
  配套博客 `content/posts/mega-moe-36-sm100-bf16-gemm-tc-util`（索引页新增第 36 篇）。
  上一轮 P25-1：新建 `docs/35-sm100-fp8-fp4-gemm.md`（+`src/sm100_fp8_fp4_geometry.cu`、
  `harness/exp_sm100_fp8_fp4_geometry.{sh,out.txt}`）——补上第 3/10/21/31 篇一直当「黑盒后端」引用的
  **目标卡上唯一 GEMM 后端**：`sm100_fp8_fp4_gemm_1d1d`。一套模板通吃稠密/m-grouped（dgrad，含 PSUM/masked）/
  k-grouped（wgrad，FP8 钉 MN、FP4 钉 K+gran32）/batched（MLA），**MXF8F6F4（`UMMA_K=32`）与 MXF4
  （`UMMA_K=64`、packed、K-major、gran32）共用骨架**。与 SM90（第 22 篇 WGMMA+寄存器累加+两种 SF 粒度分两个 kernel）
  的本质差异是 **UMMA + TMEM + UTCCP**：累加器放 TMEM、SF 从 SMEM 经 UTCCP 搬进 TMEM 由硬件按 `gran_k` 应用；
  **仅 leader CTA 单 warp** 发 `tcgen05.mma.block_scale`。**最值得记的「为什么」是 512 列 TMEM 预算与 overlap**：
  `accum=2*UMMA_N`（双缓冲）+ SF，`overlap=max(x,512)-512` 后对齐到正好 512（kernel `:120-132`），
  这解释了 host 合法性检查为何只写单级的 `umma_n+sf<=512`。还有 **SF 的 SMEM 内 warp 转置**（`:464-472`：
  `out[lane*4+i]=in[i*32+lane]`）与 gran32/128 的 stage/sf_id 选择（`:96-97/416-431`）。
  **SM100 专属，仅分析**（kernel `:47` 整段 `__CUDA_ARCH__>=1000`）；架构无关的几何/描述符设备代码在 H100 真机跑：
  SF warp 转置 **128/128 全对**；smem 描述符 4 例（K-major stride_k=1 / MN-major=swizzle atom、FP4 pack=2 偏移减半、
  SBO/LBO `>>4`）全对；指令描述符 5 例（E8M0、`n_dim=N>>3`/`m_dim=M>>4`、2-bit sf_id、packed `float_e2m1_t`→
  `MXF4Format::E2M1=1`、`update_instr_desc_with_umma_n` 置 12）全对，**总判定 ALL PASS**。行号全部核对过。
  配套博客 `content/posts/mega-moe-35-sm100-fp8-fp4-gemm`（索引页新增第 35 篇）。
  上一轮 P24-1：新建 `docs/34-locality-domain-probe.md`（+`src/locality_domain_probe.cu`、
  `harness/exp_locality_domain_probe.{sh,out.txt}`）——补上第 03/09/11 篇反复出现却从没讲清的
  `num_locality_domains=2` 的**前置步骤**：**哪个 SM 属于哪个域、这件事怎么测出来的**。
  探针设备代码 `sm100_locality_domain.cuh:8-24` 只依赖 `%smid`/`ld.global.cg`/`clock64`
  （**架构无关，H100 真机可跑**）：一 SM 一 4KB chunk、全身步长 7 与 32 行互质（每行恰访问一次）、
  `ld.global.cg` 绕 L1、`clock64` 除跳数。host `apis/locality_domain.hpp`：
  `probe :82-90`（median(-1)）→ `try_probe :93-100` 两闸门（far/near ≥ `1.25`（`:66`）、TPC 内一致）
  → 最多 5 次（`:67`）→ `get_even :103-106` + warn；`balance :126-136` **只在「一个域占满」时
  给精确 50/50**（函数名有欺骗性，本实验显式测出 50/66→17/49）。本地化内存走 CUDA VMM + MLOPart
  IPC（`cuMemCreate/Export/Import` + reserve/map/set_access，粒度 2 MiB），TMA 权重描述符把域当外维
  `(1,2,N/2,K)`。**H100 真机**（原样 include 真实设备代码）：两域都在 HBM → ~624 周期/跳、ratio 1.000、
  `clear=F` **正确拒绝**；domain1=host-pinned 模拟远端 → ratio **3.265**、准确检出 + TPC 一致；
  host 逻辑 9 边界用例 **9 PASS/0 FAIL**；真实 API `get_sm_locality_domains()` **逐位等于** `get_even`
  （66/66）、`is_localization_available=False`、粒度 2 MiB。行号全部核对过。
  配套博客 `content/posts/mega-moe-34-locality-domain-probe`（索引页新增第 34 篇）。
  上一轮 P23-1：新建 `docs/33-deepjit-runtime.md`（+`harness/exp_deepjit.{py,sh,out.txt}`）
  ——补上前 32 篇一直当黑盒的一层：**DeepGEMM 为什么安装时不编译 CUDA、首次调用才现编**。
  DeepGEMM 通过 `csrc/runtime/jit.hpp:13-33` 接入 DeepJIT（`Config(root,"DG","cutlass-<ver>",
  {include},{"deep_gemm/"})`，`deep_gemm/__init__.py:122` 在 import 时传包目录）。
  核心是 `Runtime`（`runtime.hpp:19-98`）的**五段式缓存键**（`extra_signature` + `compiler_version`
  + `get_flags()` + `post_hook` + 源码/include 摘要）与**内存/磁盘两级缓存**
  （`<root>/cache/<tag>.<digest>/`，`.committed` 标记 + 原子 rename 发布 + `:` 多根共享）。
  哈希是**双 state FNV1a**（长度前缀消歧，`hash.hpp:21-24`）；include 递归解析只跟踪
  `deep_gemm/` 前缀（`parser.hpp:37-120`），这解释了「400B 微源码却哈希 16 个头文件」。
  调用形态：每个 config 生成「只 include 一个 `deep_gemm/…cuh` + 强制实例化一个模板」的微源码
  （`sm90_fp8_gemm_1d2d.hpp:30-56`）；`compiled_dims="nk"` 使 m 不是显式编译维度
  （`runtime_utils.hpp:31-40`）。**H100 真机**：纯 Python **逐位复刻缓存键**（`8fbcaf5d…` MATCH、
  16 个头、nvcc 原文含行尾）；首进程 JIT ~4.98s → 二进程磁盘加载 ~0.20s（**~25×**）；
  抓到真实 11 个 nvcc flag；`DG_JIT_DEBUG` 因 `--generate-line-info` **换键**；
  digest 敏感性 + `CHECK_NO_SPILLS` rc=0；FNV1a 性质逐条验证。行号全部核对过。
  配套博客 `content/posts/mega-moe-33-deepjit-runtime`（索引页新增第 33 篇）。
  上一轮 P22-1：新建 `docs/32-sm90-fp8-gemm.md`
  （+`harness/exp_fp8_sm90_gemm.{py,sh,out.txt}`）——补齐第 29 篇（SM90 **BF16** 主 kernel）的另一半：
  DeepGEMM 立身的 **FP8** 在 SM90 上是**两个**主 kernel。一句话结论：**两者算同一数学
  `D=(A·sfa)@(B·sfb)ᵀ`，只差 B 的 block-scale 粒度**——`1D1D` 是 A/B 都 per-token（`gran_n=1`），
  D=**float 且必须 accumulate**（`sm90_fp8_gemm_1d1d.hpp:84`）、支持 Normal/KGroupedContiguous（即 wgrad，
  用 `SM90_TMA_REDUCE_ADD_2D` 累加 `cuh:346`）；`1D2D` 是 B per-128×128 块（`gran_n=128`），
  D=**bf16 禁 accumulate**（`sm90_fp8_gemm_1d2d.hpp:81`）、支持 Normal/MGrouped/Batched（即 dgrad）。
  分派在 `gemm.hpp:117-126`（`gran_n==1?1d1d:1d2d`），SM90 默认 recipe `(1,128,128)`
  （`utils/layout.hpp:86-98`）→ 普通稠密默认 **1D2D**。逐段精读两个 kernel（1D2D：SFB 单份非流水、
  B scale 由 math warp 从 global 直读、`num_former_iters` 处理跨 scale 组、STSM+swizzle＋TMA store；
  1D1D：K 分组原地改 tensormap、unroll 控制、fp32 裸 `st.shared`＋TMA reduce-add），并给「十处差异」对照表。
  **H100 真机**：profiler 抓到两 kernel 的真实模板实参（`BLOCK_M/N`、stages、multicast，`BLOCK_N=152`
  正是 fp32 避 bank conflict 分支）；契约精确拦在 `hpp:84`/`hpp:81`；语义 vs 反量化参考 1d2d ~6.9e-4
  （bf16 舍入）/1d1d ~1.27e-4（fp32 输出）；**交叉验证**把 1d2d 块 scale 广播成 per-column 喂 1d1d，
  逐位 0 误差；性能 1d2d 恒快 ~1.5×（4096×7168×7168 **1407 vs 957 TFLOPS**）。行号全部核对过。
  配套博客 `content/posts/mega-moe-32-sm90-fp8-gemm`（索引页新增第 32 篇）。
  上一轮 P21-1：新建 `docs/31-sm100-gemm-epilogue-operators.md`
  （+`src/epilogue_operators.cu`、`harness/exp_epilogue_operators.{sh,out.txt}`）
  ——补齐第 27 篇「何时不用自家 kernel（cuBLASLt 回退）」的另一半：**用自家 kernel 时 SM100 GEMM
  的尾部**。这是一套**所有 SM100 GEMM 共用的可插拔 epilogue**：host `EpilogueClass`
  （`csrc/jit_kernels/impls/epilogue_class.hpp:19-39`，四类 Identity/Alpha/FP8Quantization/
  BF16StochasticRounding，经 `resolve_epilogue_class:108-127` 三入口互斥解析，Python
  `deep_gemm.epilogue.*`）选出一个 **device 无状态算子**（`epilogue/operators.cuh`，继承
  `EpilogueOperatorArgs`，`sm100_fp8_fp4_gemm_1d1d.cuh:57` 强制 `sizeof` 相等），随 kernel
  `__grid_constant__` launch，在 `sm100_store_cd.cuh:124` / `sm100_store_cd_swap_ab.cuh:100/112`
  对从 TMEM 读回的寄存器批 `apply_values()`。四算子：`Identity`；`ScaleByAlpha`（`__fmul2_rn`，
  只缩放乘积项）；`StochasticRoundToBF16`（**四元组哈希 PRNG**：`partial_hash` 四常数 →
  `select_random_bits` 雪崩 → 四元组共享 `h + quartet_n_idx`，硬件 `cvt.rs.bf16x2.f32` 随机舍入）；
  `QuantizeToFP8`（先 bf16 舍入 → per-32 `get_packed_bf16_amax`+`get_ue8m0_sf_exp`/`_inv` →
  `scale_bf16x2_into_fp8x4` → E4M3，`store_sf` MN-major 4 字节打包）。swap-AB 版还用
  `shfl_xor` 跨 lane 归约哈希。接线：SM90 全部断言 "Custom epilogues require SM100"；FP8 输出对
  =`FP8Quantization`（`einsum.hpp:190-200`），补齐第 21 篇「SM100 独占 FP8 输出对」的落点。
  **H100 真机**（`operators.cuh:12-92` 逐行抽取，sha256 `d00b6b6f…`）：`cvt.rs` 在 sm90 明确失败
  （SM100 专属）；ScaleByAlpha 逐位 0 不符；四元组哈希 2²⁰ 样本均匀/独立；随机舍入**无偏** vs
  RNE 固定偏一侧；`QuantizeToFP8` D/SFD vs torch **逐位 0/8192、0/256**，ALL PASS。行号全部核对过。
  配套博客 `content/posts/mega-moe-31-sm100-gemm-epilogue-operators`（索引页新增第 31 篇）。
  Hugo 构建 0 ERROR。
  上一轮 P19-1：新建 `docs/29-sm90-bf16-gemm.md`（+`harness/exp_bf16_sm90_gemm.{py,sh,out.txt}`）
  ——补上**DeepGEMM SM90 BF16 稠密 GEMM 主 kernel**（前 28 篇里 BF16 只被顺带提到）。它是唯一一个
  **一套 kernel 服务 Normal / MGrouped(contiguous & masked) / KGrouped / Batched 五种 `GemmType`**
  的 SM90 kernel：没有 SF，骨架与 FP8 SM90 同构（TMA warp 铺流水 → math warpgroup 发
  `MMA_64xNx16` WGMMA → epilogue 经 smem 用 TMA store），差异在 **`BLOCK_K=64`**、
  **合并 stage**（stage≥10+Normal+K/K-major+128 math → `BLOCK_K*=kNumStagesPerMerge`，减少
  `warpgroup_wait<0>`）、**fp32 输出关 swizzle**（裸 `st.shared` vs bf16 的 STSM+128B swizzle）、
  **累积用 TMA reduce-add**（不额外读回 D）。**H100 真机**六类入口全部对拍：峰值 **801 TFLOPS**
  （bf16 4096×7168×7168）、k-grouped wgrad 422、batched MLA 781；抓到 `(128,128,512)` 触发合并 stage。
  行号全部核对过。配套博客 `content/posts/mega-moe-29-sm90-bf16-gemm`（索引页新增第 29 篇）。
  Hugo 构建 0 ERROR。
  上一轮 P18-1：新建 `docs/28-deepgemm-best-config.md`（+`harness/exp_best_config.{py,sh,out.txt}`）
  ——补上**「用自家 kernel 时怎么选 block_m/n/k/cluster/stage/线程」**（第 27 篇讲了「何时不用自家
  kernel」，两篇合起来才是完整选核）。`get_best_config` = 枚举候选 → 逐个打分 → 取最小
  （`common.hpp:18-56`）；SM90 是「**L1/L2 周期最小化**」的定量模型（multicast 省 L2 读、累积模式
  D 回读翻倍、尾波效率惩罚、单波禁多播），SM100 退化成**词典序 tie-break**（单波→多播→少波→
  尾波→小块，`num_cycles` 恒 0 TODO）。**H100 真机对拍**（`DG_PRINT_CONFIGS=1` 抓真实选核，Python
  逐行重写 SM90 那套）：**20 组形状 × 15 字段逐项 0 不符**，覆盖 7 类算子；候选排序显示大形状在
  **90 个合法配置**里近似最优。配套博客 `content/posts/mega-moe-28-deepgemm-best-config`
  （索引页新增第 28 篇）。Hugo 构建 0 ERROR。
  上一轮 P17-1：新建 `docs/27-cublaslt-fallback.md`（+`harness/exp_cublaslt_fallback.{py,sh,out.txt}`）
  ——补上**「DeepGEMM 什么时候不用自家 kernel」**这条一直没讲清的选核规则。**H100 真机实测**
  （`torch.profiler` 抓真实发射的 kernel 名）：稠密 BF16 `bf16_gemm_nt` 与按头 batched BF16 einsum
  （`bhr,hdr->bhd`/`bhd,hdr->bhr`）在「非确定性 ∧ 未限制 SM 数 ∧ plain-store epilogue」时**默认交给
  cuBLASLt**（`gemm.hpp:518-523` 三与门）；`Identity` epilogue 的 `get_alpha()=1.0`（而非 nullopt）
  是「裸 GEMM 也回退」的关键；`set_num_sms(<device)` 通过 `is_cublaslt_available()`（`runtime.hpp:89-91`）
  顺带禁用回退；`einsum("bhd,bhr->hdr")` **任何情况都走 cuBLASLt**（源码 TODO determinism）。
  翻译层用 `(A Bᵀ)ᵀ = B Aᵀ` 把 row-major 变 column-major（`smxx_cublaslt.hpp:115-135`）；还讲了
  FAST_ACCUM 关闭、SM_COUNT_TARGET、16 MiB workspace（`runtime.hpp:25-26` 注释）、NVFP4 VEC16 UE4M3、
  strided-batched syrk/symm；FP8/grouped 永不回退。实测 8 组形状 **cuBLASLt vs DeepGEMM 0.97–1.02×**
  （4096×7168×7168 均 ~377 TFLOPS）——解释默认为何敢回退。配套博客
  `content/posts/mega-moe-27-cublaslt-fallback`（索引页新增第 27 篇）。Hugo 构建 0 ERROR。
  上一轮 P16-1：新建 `docs/26-k-grouped-sf-packing.md`（+`harness/exp_kgrouped_sf_pack.{sh,out.txt}`、
  `src/kgrouped_sf_pack.cu`）——补上 **P15（单组 SF 布局）与 P4-2（K 轴分组 wgrad）的交叉点**：
  「每组 SF 行独立对齐补零」的按组 block-scale 打包。host `transform_k_grouped_sf_into_required_layout`
  的 `(1,1,gran_k)` recipe、SM90 只转置/SM100 才打包、contiguous 精确 `Σ ceil_div(k/gran_k,4)` 与
  PSUM 上界 `(sf_k+3G)/4`、已打包 INT 检查；device `pack_fp32_into_ue8m0<kNumGroups>` 的「每 warp 一个
  packed 行 → 线性扫组找 owner（break 在累加 padding 之前）→ 取址减累积 padding → LSB-first 打包」。
  **kernel 与 UMMA 无关**：H100 上把 `smxx_layout.cuh:193-293` **逐行原样抽取**后编译运行真实设备代码
  （打印片段 sha256）+ 独立 host 参考逐行对拍，**ALL PASS**：contiguous 无/有 padding、**空组**、8 组、
  gran128 大量 padding、**PSUM 尾部 poison**（上界宽松）、输出 K pitch=mn；host wrapper 在 sm90 被
  `:224` 断言拒绝。配套博客 `content/posts/mega-moe-26-k-grouped-sf-packing`（索引页新增第 26 篇）。
  Hugo 构建 0 ERROR。
  上一轮 P15-1：新建 `docs/25-sf-layout-transform.md`（+`harness/exp_sf_layout.{py,sh,out.txt}`）
  ——补上**贯穿所有 FP8/FP4 kernel 的 SF 布局前置步骤**（前面第 10/12/13/21 篇都只是顺带提到）。
  `transform_sf_into_required_layout` 的四分派、recipe 的 2-tuple（断言 `not is_sfa`）/
  3-tuple（断言 `is_sfa`）契约、`check_sf_layout` 形状/stride 检查、`get_tma_aligned_size` 16B 对齐；
  device 侧 `transpose_fp32`（MN-major + 幂等早退）、`transpose_and_pack_fp32_into_ue8m0` /
  `pack_fp32_into_ue8m0`（UE8M0 只留指数、4 字节 LSB-first、PSUM gap 行跳过、只允许 2 的幂的
  debug 陷阱）。**转置/打包 kernel 非 SM100 专属**，H100 真机六组 **ALL PASS**：转置 `val_diff=0` +
  幂等；UE8M0 CUDA==torch 参考逐位一致 + LSB-first 拆包核对；已 MN-major 输入两 kernel 互证；
  PSUM gap 行跳过且无 NaN 码；recipe 分派在 SM90 上 `(1,128)/(128,128)` OK、`(1,32)` 拒
  （`layout.hpp:57`、SM100 专属）、`:24`/`:28` 契约反例被拦；端到端 `fp8_gemm_nt` relerr 6.8e-4、
  裸 SF vs 显式变换 SF 逐位相等。配套博客 `content/posts/mega-moe-25-sf-layout-transform`
  （索引页新增第 25 篇）。Hugo 构建 0 ERROR。
  上一轮 P14-1：新建 `docs/24-mqa-logits-paged-schedule.md`（+`harness/exp_mqa_logits_paged_schedule.{sh,out.txt}`、
  `src/mqa_logits_paged_schedule.cu`）——补上 SM100 dense MQA logits 的**分页版** metadata 调度器
  （第 23 篇连续版的分页对照）。主 kernel 仍是 block-scaled UMMA + TMEM，**SM100 专属，仅分析**；
  但 `scheduler/sm100_paged_mqa_logits.cuh` 的调度逻辑（varlen 请求 run-length 分组、请求内 Q block
  **均分**、代价模型 `2·BQ·blocks+tokens`、**溢出回退借 `num_requests` 最高位**、同一请求多 Q block
  **复用同一段 KV**（`kv_shared_with_prev`+`kSplitsPerChunk`）、**每 lane 缓存一页的滑动窗口**
  `get_kv_page_coords`）**与 UMMA 无关**，在 H100 上**直接编译运行真实设备代码**：10 用例 ALL PASS——
  metadata 每-SM 边界 vs host 独立复刻逐字段 0 不符（含 132/8 SM、空 SM）；真实 scheduler 发射任务
  **所有 (请求,Q-block,split) 恰好覆盖一次**（dup/miss/bad=0）；分页查表含 **mid-refill** 与 host 0 不符；
  代价溢出回退用例（split 代价 4.99e9>2³²）`use_token_cost=Y` 边界仍一致；坐实 metadata `block_q=4`
  与主 kernel `BLOCK_Q=128/H` 解耦（H=64 sched/meta slots **432/288**、H=128 **864/288**）。
  配套博客 `content/posts/mega-moe-24-mqa-logits-paged-schedule`（索引页新增第 24 篇）。Hugo 构建 0 ERROR。
  上一轮 P13-1：新建 `docs/23-mqa-logits-sm100-schedule.md`（+`harness/exp_mqa_logits_schedule.{sh,out.txt}`、
  `src/mqa_logits_schedule.cu`）——补上 SM100 dense MQA logits 的主 kernel + **连续版负载均衡调度器**
  （第 19 篇 SM90 dense、第 22 篇 SM100 sparse 之间的缺口）。主 kernel 是 block-scaled UMMA + TMEM，
  **SM100 专属，仅分析**；但 `scheduler/sm100_mqa_logits.cuh`（KV span 并集 / 偶奇交错前缀和 /
  `cost=2·split+3` 代价模型 / split 粒度边界二分；`SM100MQALogitsScheduler` 的 scheduled 与 grid-stride
  双模式）**与 UMMA 无关**，在 H100 上**直接编译运行真实设备代码**：10 用例 ALL PASS——host 独立复刻与
  device spans/边界逐字段 0 不符；两种模式下任务恰好覆盖所有有效 split 一次（dup/miss/bad=0；scheduled
  任务数 > Q block 数，坐实「一个 Q block 被相邻 SM 从中间切开」）；负载均衡 skew S=8192
  scheduled **[185..191]** vs grid-stride **[99..352]**、8 SM **[50..55]** vs **[28..88]**；
  字数公式与边界用例全通过。配套博客 `content/posts/mega-moe-23-mqa-logits-schedule`（索引页新增第 23 篇）。
  Hugo 构建 0 ERROR。
  上一轮 P12-1：新建 `docs/22-sparse-mqa-logits.md`（+`harness/exp_sparse_mqa_logits.{py,sh,out.txt}`）
  ——补上稀疏注意力 indexer 的**第二级打分**：`fp8_fp4_sparse_mqa_logits` /
  `fp8_fp4_paged_sparse_mqa_logits`（在 `sparse_kv_block_indices` 选中的 KV 块内算细粒度 logits，
  压缩输出 `[T, num_max_sparse_blocks*SB]`）。**SM100 专属，仅分析**（host `attention.hpp:313/380/414`
  断言 `arch_major==10`）：metadata 布局（`kBlockQ=2`、`KVBlockInfo` 16+16 位打包槽、`KVSplitHeader`
  连续/尾块两标志）、metadata kernel（有效块前缀、两 token 归并去重、物理块三态映射、末 CTA 建
  负载均衡调度）、主 kernel（五类 warp、block-scaled UMMA、contiguous TMA 快路 vs `cp.async`
  尾裁剪、压缩列映射）。H100 实测（架构无关语义）：metadata 尺寸公式复刻、用真实 SM90 dense kernel
  取全量 → 压缩输出 **`gather==L` 逐位 True**、vs fp32 参考 3.5–4.1e-2（=dense 量化层）、
  去重 ~12.5%、压缩比 28×/7×/14×、unaligned ks 跨块边界、四个 host 入口在 sm90 全部被拒。
  配套博客 `content/posts/mega-moe-22-sparse-mqa-logits`（索引页新增第 22 篇）。Hugo 构建 0 ERROR。
  上一轮 P11-1：新建 `docs/21-fp8-einsum-projection.md`（+`harness/exp_fp8_einsum.{py,sh,out.txt}`）
  ——补上 BF16 MLA 投影（P8-2）的 **FP8 block-scale 版** `fp8_einsum('bhr,hdr->bhd')`
  （SM90 `sm90_fp8_bmm`，**H100 可跑**）。host `(b,h)` permute 成 `(h,b,d,r)`（`h`=batch、
  `b`=m），复用前向 `1d2d` 骨架的 `GemmType::Batched`（A per-token 1D SF + B per-128×128
  2D SF，3D TMA，逐 batch 的 SFB 基址由 `get_global_idx<SF_K>` 给出）；SM90 强制 gran 128、
  K-major、BF16 D；FP8 `(d,sfd)` 输出对与 `bhd,hdr->bhr`/`bhd,bhr->hdr` 是 SM100 专属（仅分析）。
  H100 实测：误差干净分两层（vs 隔离量化 2–4e-3、量化本身 3–4e-2）、每头独立且头 0 置零无串扰、
  SF 布局与契约逐条对上、确定性开关对 FP8 路径无效、大形状峰值 **581 TFLOPS**。
  配套博客 `content/posts/mega-moe-21-fp8-einsum-projection`（索引页新增第 21 篇）。Hugo 构建 0 ERROR。
- **下一步**：P0~P140 已完成（最新 **P140-1 把 wgrad 的 S 步累积折进收缩维 K（big-K 折叠）：每步累积税能不能消掉**：
  承接第 149 篇结尾「要再快只剩减少字节的路」，把它做成**结构性正面答案**——「每步累积」只是「S 次独立 launch」
  的产物；沿 K 拼接 S 步激活后 `dW_e = Σ_s A_{e,s}^T B_{e,s} = [A_{e,1};…;A_{e,S}]^T [B_{e,1};…;B_{e,S}]` 恒等于
  **一个 K=S·k_e 的单次 GEMM**，累加器全程住 TMEM、只最后一次 fresh 落盘、跨 launch RMW 归零。**契约**：REF
  `k_grouped`（`gemm.hpp:403` 必传 `c`、`sm90_fp8_gemm_1d1d.hpp:84` 必 fp32、`SM90_TMA_REDUCE_ADD_2D`
  `sm90_fp8_gemm_1d1d.cuh:346`）每步读回+写；big-K 走现成 1D2D（`sm90_fp8_gemm_1d2d.hpp:81` `not c`+bf16）一次写，
  **不改内核**。**H100 实测**（`dW2 [2048,7168]`、E=8、k_e=512，空闲 GPU）：Part A S=4 REF-vs-BIG-K relerr **3.68e-2**
  == vs fp64 同层（同一算子）；Part B `S=1→8`：REF `644.8→4141.4us`、ALT-R `602.6→2599.0us`、**BIG-K `220.7→870.5us`**，
  `REF/BIG-K 2.92→4.76×`、`ALT-R/BIG-K≈3.0×`（恒定）、BIG-K **1105 TFLOPS**；Part C REF 输出流量 `S·8B·E·M·N` vs
  BIG-K 一次 `2B·E·M·N`（S=8 省 **7.3GB**），算术强度 REF 恒 `83~116`→BIG-K `441→1792`（越过拐点、计算受限）。
  **前提**：S 步激活须同时在手（fp8 驻留 `(M+N)·SUM_K·S`，S=8=302MB）。**收益分解**：纯折叠≈3.0×，再叠 fp32→bf16 直写到 4.76×。
  **遗留**：FP8 累加器路线未做。SM100 `tn` 同结构、仅分析。索引页新增第 150 篇）。
  可继续（均为增量、非承诺）：
  ① 量化第 ① 条路线（**FP8 累加器**：把每步 RMW 的 8B 压到 2B）的**误差-字节边界**，与 big-K 折叠对比；
  ② 对 P96/P97/P123 的轻微漂移清单关键路径定点复核；
  ③ 把空出的 DeepEP 端到端留给 NCCL≥2.32.3 环境（见「阻塞」）。详见上）。
  上一轮 P139-1 把 wgrad 的 EPI-B 落到真 kernel：多出来的 235MB/步读，藏住了吗：
  第 148 篇留的口子（「在一个可改的 SM90 GEMM 分支上做 EPI-B 端到端验证」，可继续①）**落地了**。**发现**：
  DeepGEMM SM90 BF16 稠密 `bf16_gemm_nt(c=d)` 本就是 **bf16 累加器 RMW**（`sm90_bf16_gemm.hpp:91`
  `with_accumulation`；epilogue `SM90_TMA_REDUCE_ADD_2D` `sm90_bf16_gemm.cuh:381-384`），**无需改 kernel**。
  **H100 实测**（空闲 GPU，`dW2 [16384,7168]`、一份 bf16 D=235MB）：Part A RMW 两步 == `2·(a@b)`、profile
  模板 bool `false→true`；Part B 单步 fresh 197.8us/608 TFLOPS、EPI-B 307.2us、fp32 累积 545.1us，`EPI-B/REF`
  稳定 **1.73–1.77×（顶到第 148 篇下界 1.69×）**，但 `SEQ/EPI-B` 在 S≥8 只有 **0.96–0.97×（略慢）**；
  Part C 同一份 RMW 读的暴露比例 `K=512/2048/4096 → 157%/161%/47%`（藏不藏只看算术强度，wgrad K=512 最不
  计算受限）；带宽锚读一份 D 235MB=88.0us/2.67 TB/s。**`ncu`**：fresh 读 74.78MB/写 215.10MB/202.5us，
  EPI-B 读 **312.75MB（+237.97MB=恰好一份 D）**/写 214.33MB/300.5us → **多读的那份 D 按满速 2.43 TB/s
  露在墙上**。**结论**：EPI-B 真实可跑，但「多出来的 235MB/步读」在 wgrad 形状下**彻底暴露（157%）**，
  它只拿回对 fp32 累积的 1.7×、并不优于「批量 fresh + 一次流式归约」；要再快只剩「减少字节」的路。索引页新增第 149 篇）。
  上一轮 P138-1 把 wgrad 的 S 步归约藏进 GEMM：epilogue 融合 / 侧流重叠到底行不行：
  第 147 篇留的口子（「每步 bf16 梯度要串行读一遍」）**走到了能核对的数，是个双重否定的负结果**。
  ① **字节账**：批量 ALT-OPT `470·S+940`；侧流增量 INC-F `1410·S` / INC-B `940·S`（每步 RMW master）；
  **同 kernel epilogue 融合 EPI-B `470·S`（最少）**。② **消费核**（`src/wgrad_epilogue_overlap.cu`，架构无关、
  H100 真跑、vs CPU double 5.96e-8）：纯读 3005 GB/s、fused_all S=8 3084 GB/s、≥264 block 即满带宽。
  ③ **真机重叠**（真实 SM90 1D2D bf16，单步 218us）：`SEQ8` 2672us 对 OVL8-INC-F 4912 / INC-B 3597 →
  **0.54–0.84×（比串行还慢）**；`set_num_sms` 限核、扫消费者块数都救不了（恒 ~0.58）→ 不是吞吐而是不并行。
  ④ **决定性控制**：伪 epilogue 生产者动态 smem 0→128KB 消费者藏住 53%→92%、**227KB（=DeepGEMM
  `sm90.hpp:14` `232448`）时 −0.1%（完全串行）**——`232448 B/SM` 让侧流 block 无从 co-reside（印证第 144 篇）。
  ⑤ **数值**：bf16 累加器 S=16 relerr 5.03e-3、cos 0.9999875（< FP8 噪声 3.6e-2，精度免费）。索引页新增第 148 篇）。
  上一轮 P137-1 MoE wgrad 的「fp32 物化/归约税」量化：融合进优化器能拉回多少 2.2×：
  把第 146 篇结尾「只有优化器直接吃 bf16 才拿回 2.2×」落地。三协议（`REF` 每步 1D1D fp32 累积 /
  `ALT-R` 每步 bf16 + `torch.sum` + 优化器 / `ALT-OPT` 每步 bf16 + **融合进优化器**、不落地 fp32 梯度）；
  消费侧内核（`src/wgrad_reduce_consumer.cu`）架构无关、H100 真机跑真实设备代码 ~3.08 TB/s（HBM 92%），
  `atomicAdd` 变体 S=1 就 **23.4ms（RMW，慢 ~80×）被否决**、手写核比 `torch.sum` 快 **10–26%**（P136 的
  838.6us 归约税是次优实现）。**端到端**（真实 SM90 GEMM，`dW2 [8,2048,7168]`）：纯 GEMM **2.27×**；
  `S=16` REF 8441.5us、ALT-R 5702.3us（**1.48×**）、**ALT-OPT 5045.4us（1.67×）**；`S=1` ALT-OPT 已 **1.55×**
  （ALT-R 0.90×）。**核心结论**：融合进优化器把端到端 1.48×→**1.67×**、顶到**理论天花板 1.69×**
  （`t_ref/(t_alt+235MB/BW)`），即收回「物化税」（省掉 reduce 写 fp32 + 优化器再读回 8B/elem）；但
  **2.27× 不可达——每步 bf16 梯度至少被消费端读一遍（~76us/步），叠加在 219.6us/步 GEMM 上**。
  **2.27× 隐含「这份读免费」，它从来不是**（修正第 146 篇半句话）。SM100 `tn` 同族、仅分析。H100 真机、
  空闲 GPU1。索引页新增第 147 篇）。
  可继续（均为增量、非承诺）：
  ① **把「读一遍 bf16 梯度」这 76us/步再压**——例如把归约/优化器更新真正融进 GEMM 的 1D2D epilogue
  （消费端连那份 `[S,E,M,N]` bf16 张量都不落、边算边更新 master），量它能把 1.67× 推向 2.27× 的哪一点；
  ② 对 P96/P97/P123 的轻微漂移清单关键路径定点复核；③ 把空出的 DeepEP 端到端留给 NCCL≥2.32.3 环境
  （见「阻塞」）。详见上）。
  上一轮 P136-1 MoE wgrad 换 bf16 输出的精度代价与 2.2× 上界落地：
  第 145 篇结尾那条优化路线用真实 SM90 kernel 落地——SM90 只有两种输出形态（1D1D fp32/累积 vs 1D2D
  bf16/无回读，`sm90_fp8_gemm_1d2d.hpp:81`），没有「fp32 直写」。**精度**：逐专家 fp64 参考下 bf16 输出
  误差 1.41e-3 比 FP8 输入噪声 3.63e-2 小 **25.8×**、比 fp32 累积 1.18e-4 大 ~12×；累积 S=1→8 相对误差
  近似恒定 ~ε_bf16、cos 稳在 0.99932。**性能**：单步纯 GEMM **2.27×**，端到端 **S=8 1.54× → 封顶 ~1.55×**。
  索引页新增第 146 篇）。
  上一轮 P135-1 MoE wgrad 的强制累积代价量化（`docs/145`）：
  把第 15 篇「SM90 `k_grouped_nt` 强制累积（D 回读）→ 访存受限」这条根因用**同输出规模/同 FLOP**
  的四组对照拆开——真实 `k_grouped`(fp32,acc) / `E×`稠密 1D1D(fp32) / `E×`稠密 1D2D(bf16) /
  `m_grouped` 单组(求和)。**反直觉结论**：K 分组 kernel 本身不是问题（比手动发 8 个稠密快 14%），
  主因是每专家一份 **fp32** 权重梯度（同 G× 输出规模下 fp32/acc vs bf16 = 2.58×）；wgrad 总流量
  978.5MB/AI 122.9（<591 访存受限），C 回读占 48%、每专家 fp32 输出占 96%；真实 wgrad 240 TFLOPS
  vs 同输出规模 bf16 537（2.23×）vs 求和 1240（另一算子）。H100 真机 + ncu（吻合 0.1%）；
  SM100 `tn` 同结构、仅分析。索引页新增第 145 篇）。
  可继续（均为增量、非承诺）：
  把「wgrad 换 bf16 输出 / 改累积协议（chunk 写 fresh + 归约）」的 **2.2× 上界**在 SM90 上用真实
  kernel 试着落地并对照数值（wgrad 精度代价）；或对 P96/P97/P123 的轻微漂移清单关键路径定点复核；
  或把空出的 DeepEP 端到端留给 NCCL≥2.32.3 环境（见「阻塞」）。详见上）。
  上一轮 P130-1：新建 `docs/140-deepep-comm-context.md`
  上一轮 P129-1 MoE 路由负载均衡三件套（aux-loss / router z-loss /
  aux-loss-free bias）及其反向**：补齐训练侧一直缺席的「负载均衡」算子——第 4/127 篇把 `bias` 当常量是对的，
  但「bias 从哪来、怎么更新」与 aux/z loss 的梯度从没写过。三种机制的前向公式 + 梯度 + 与 Mega Gate 的
  对接（`mega_gate.hpp:56-57`、`impls:396-398`、`epilogue:212-213`）全部讲清，H100 真机 21 PASS / 0 FAIL：
  aux-loss 梯度 vs autograd 2.20e-15、z-loss 4.95e-16；**核心对照**——aux-loss 对硬 top-k 负载是弱杠杆
  （8.50→5.44），bias 符号更新是强杠杆（28.04→1.03），坐实 DeepSeek-V3 aux-loss-free 的工程动机。
  纯 host/Python、架构无关。索引页新增第 139 篇。
  上一轮 P128-1 DeepGEMM 启发式最底层的公共选择器
  `heuristics/utils.hpp`（元素位宽 / TMA swizzle）+ `GemmDesc` 的 dtype→MmaKind 管道 + Python 别名层**：
  第 28 篇只给 `config.hpp` 五结构体列**字段表**、`heuristics/utils.hpp` 从没被单独读过，而它是**每个**
  SM90/SM100 kernel 的 K 块宽度与 TMA swizzle 的**唯一出处**。`get_num_element_bits`（逻辑位宽）+
  `block_k=128*8/bits` 的 **`block_k*bits==1024` 不变量**（K 块一行恰 128 B = 最大 swizzle）；
  `get_swizzle_mode`（`{128,64,32,16}` 最大可整除者；`16`=非 swizzle 仅交错、**从不返回 0**）；
  **逻辑位 vs 存储字节**两个位宽 + SM100 的 `/pack_factor`（MXF4 ÷2）→ K-major 全 128、MN-major FP4=64；
  Mega MoE 的**平行实现**（自带 `get_num_mma_elem_bytes` + 硬编码 128，半宽 `128→64`）；
  `GemmDesc` 派生管道 + `check_validity`（BF16 两端 bf16 / FP8-D 仅 Batched 非累积 / `num_sms%2==0`）；
  Python `get_m_/get_k_` 共用一把对齐旋钮。H100(sm90) 真机 **43 PASS / 0 FAIL**（4096 组 swizzle 0 不符、
  UNREACHABLE 恰 3136 组、validity 10 例全对）。至此 `heuristics/` 目录全部被专篇覆盖。索引页新增第 138 篇。
  上一轮 P127-1：SM100 sparse MQA logits 的 metadata kernel 逐段精读：
   把第 22 篇对 metadata kernel 的 30 行概览补齐——**整个 kernel 架构无关**（只用 `ptx::` 整数/原子、
  `cp.async`、`cta_exclusive_sum`、`cute::` 下标；不碰 UMMA/tcgen05/UTCCP/TMA），故 H100(sm90) 原样
  启动真实 kernel 并与独立 host oracle 对拍。主循环：Q 块构造 + 有效块前缀（分页线性扫描配对 +
  `atomic_add(next_q_offset)` 无锁认领）、稀疏索引加载（非对齐 `ld_evict_first` 就地平移 vs 对齐
  `cp_async_cg<256>`）、**归并路径二分分块**、跨分区去重 + carry、`cta_exclusive_sum` + `atomic_add`
  按 Q 块预留连续 split、`write_merged_kv_block` 三态物理块 + 相对槽偏移 + 只判末块 contiguous/tail、
  尾块 padding；末 CTA（release-acquire 计数）建调度：连续版均分 + Q 块边界切 + `atomicMax`，
  分页版有界条目 + `balance_wave_entries` 直方图排序/旋转/两 wave 反向。H100 真机 **ALL PASS**：
  6 组样例 **446 块 / 36 split 0 不符**、调度覆盖 `miss=dup=bad=0`。主 kernel + host `arch_major==10`
  门控 SM100 专属、仅分析。索引页新增第 137 篇。
   上一轮 P126-1：P0~P126 已完成（**P126-1 SM100 sparse MQA logits 主 kernel 逐段精读**：
   把第 22 篇对主 kernel 的 40 行概览补齐到与第 113 篇（dense 版）同深度——两个入口同一核心、
   `UMMA_N≤64`（与 dense 钉死 128 不同）、`split_kv=fp4?640:512`、`num_kv_stages=fp4?5:3`；六类 warp；
   **gather 式 KV 走 `cp.async` 故需自定义 smem swizzle**（低 3 位 XOR bits[3,6)）与尾块零填充；
   metadata 标记位 + 双 Q 槽 16+16 位打包；分页 `PAGE%SB==0` 保证块不跨页；math warp 压缩列映射
   `(base+slot)*SB+t` 双射；`SharedStorage` 4 组 `offsetof` 8/8、TMEM raw 376/428→512、线程 768/896。
   H100(sm90) 真机（**直接 include 真实 layout 头**）**ALL PASS**，主 kernel SM100 专属、仅分析。
   索引页新增第 136 篇。
  可继续（均为增量、非承诺）：
  稀疏 MQA logits 的 metadata + 主 kernel（第 22/136/137 篇）已闭环；可选：把
  `harness/check_refs.py` 接进 CI（退出码已就绪）、定期重跑 `triage_refs.py` 复核 `adjacent_code` 队列，
  或选 DeepGEMM/DeepEP `csrc/utils/` 剩余小角落做定点复核。DeepEP 端到端仍需 NCCL≥2.32.3（见「阻塞」）。详见上）。
   上一轮 P125-1：P0~P125 已完成（最新 **P125-1 Mega mHC 反向的目标卡单 kernel 接线（下）**：
   把第 132 篇结尾的「真实 H=7168 warp-pair 分段 + G0 累加器直喂 K1」坐实成可核对几何——
   前向 Post worker 网格（`kNumRowsPerWarp=16`/`kNumRowGroupsPerWarp=8`/`kNumLanesPerRowGroup=4`、
   两 worker 按 UMMA 原子切、各覆盖全 64 行、WG3 相加）、`warp_reduce_sum<4>` 归约树、TMEM 列预算
   `256+32=288<512`。H100(sm90) 真机（**直接 include 真实头文件**）：几何 0 错误、分段归约 vs double
   **rel 4–6e-8** vs 全 32-lane **rel 2.47**（结构性错误）、分段网格 vs 朴素 **2.03–10.93×**；
   真实形状 K1+G1a **AI≈0.75** 访存受限。主 kernel SM100 专属、仅分析。索引页新增第 135 篇）。
   上一轮 P124-1：P0~P124 已完成（最新 **P124-1 DeepEP 验收测试栈最后三份收口**：
   `test_all_gather`/`test_session`/`test_allocation`；H100 8 卡真机 ALL PASS、索引页新增第 134 篇）。
  上一轮 P122-1：P0~P122 已完成（最新 **P122-1 Mega mHC 反向的目标卡接线 + per-token 融合 worker**
  上一轮 P121-1：P0~P121 已完成（最新 **P121-1 Mega mHC 反向的 H100 融合账**：
  把第 130 篇那条反向当成待部署算子流水拆成**六段**（K0/G0/K1/G1a/G1b/K2），量化 roofline：整条反向
  **AI≈6.5**、三段 GEMM **AI≈24** ≪ H100 bf16 拐点 295 → **访存受限**；`T=16384` 访存下界 3648.9 us
  vs GEMM 计算下界 68.4 us（53×）；H100 真机六段 kernel device(fp32) vs double 最坏 8.84e-7、元素级
  K0 1005.8 GB/s、去低基数原子后 K1 9.66→3.82 ms、真实形状三段 GEMM 3.04 ms/~22.3 TFLOPS 而 torch
  逐 op 元素级 62.3 ms、autograd 119.5 ms（比访存下界慢 ~33×）；目标卡映射仅分析；索引页新增第 131 篇。
  可继续（均为增量、非承诺）：
  第 127/130/131 篇已把两条门控反向（Mega Gate、mHC）的数学与算子账补齐；剩余可选：把本篇六段流水
  在 SM100 上做 kernel 级「仅分析」细化（例如 G0 的 TMEM 累加器直喂 K1 省 `u0` 往返的接线推演）、
  或对 P96/P97 的轻微漂移清单关键路径定点复核。另 DeepEP 端到端仍需 NCCL≥2.32.3（见「阻塞」）。详见上）。
  上一轮 P119-1：P0~P119 已完成（最新 **P119-1 DeepEP 的 CMake 调试构建链**：
  一套模板两个产物——`setup.py`（只编三个 host TU）vs `CMakeLists.txt`（`_C` 扩展 + `runtime_comm`/
  `runtime_driver` 两个 host 静态库 + `csrc/indexing/main.cu` 索引 TU）；`RUNTIME_CUDA_LIBRARIES` 的
  `CACHE INTERNAL` 解耦、`-DNCCL_ROOT_DIR` 判 CMake 变量、nvcc 诊断 flag 只作用于索引 TU（host `.cpp`
  走 g++）、`cmake-build-*` 双向 ignore；H100(sm90) 真机 configure rc=0、三目标全 OK
  （`_C.so` 1178816 B / `.a` 32794、627820 B），默认 `all` 因索引 TU **rc=2** 失败（上游
  `rdma.cuh:619/623` 真实编译缺口）；DeepGEMM 对偶无 NCCL / 无 runtime 静态库；索引页新增第 129 篇。
  可继续（均为增量、非承诺）：
  P96/P97 的轻微漂移清单关键路径定点复核、或两库 `csrc/utils/` 其余小工具（DeepEP `system.hpp`/
  `event.hpp`）的剩余角落。详见上）。
  上一轮 P118-1：P0~P118 已完成（最新 **P118-1 NCU PM-sampling 离线绘图工具 `scripts/quick_plot_pm.py`**：
  448 行纯 host/Python，把 `.ncu-rep` 变成按物理域分组的 PM-sampling 时间序列面板；与第 55 篇内核内
  `%clock64` 探针互补；五结构 + 45 条 curated（三种 metric 名形态、跨代双候选）+ host 管线三约定
  （`idx==11`/固定三行/`len>1` 闸门）；H100(sm90) 真机 `ncu --chip gh100` **73 PASS / 0 FAIL**，
  resolved 34 / missing 11；索引页新增第 128 篇）。详见上。
  上一轮 P115-1：P0~P115 已完成（最新 **P115-1 DeepGEMM 的 Python 类型桩生成器 `scripts/generate_pyi.py`**：
  890 行纯 stdlib，把 `csrc/**` 里的 62 条 `m.def` + C++ 签名变成 `_C.pyi`；`BracketTracker` 四括号
  语义、类型/默认值映射、三处真实边界（`m.attr` 10 个别名无 stub、`std::variant` 过宽正则、
  `default_args` 序位错配）；纯 Python、H100(sm90) 真机 **45 PASS / 0 FAIL**；索引页新增第 125 篇。
  绕开 `check_nccl_so`（=阻塞落点）+ 版本号 broad-except 吞成 `+local` + wheel 文件名/`DG_SKIP_CUDA_BUILD`
  + `prepare_includes` editable 坑；纯 Python、H100(sm90) 真机 **54 PASS / 0 FAIL**；索引页新增第 124 篇。
  可继续（均为增量、非承诺）：
  `DeepEP/build.sh`+`develop.sh` 之外的 `cmake-build-*` 调试链、`DeepGEMM/scripts/generate_pyi.py`
  的 `BracketTracker` 括号配平（本篇只带过）、或从 P96/P97 的轻微漂移清单挑关键路径定点复核。详见上）。
  上一轮 P113-1：P0~P113 已完成（最新 **P113-1 DeepEP 的 Python 缓存基座 `weak_lru`**：
  方法级弱引用 LRU 把 key 从 `self` 换成 `weakref.ref(self)` 以防 buffer 泄漏；`value_or` 仅
  `None` 回退；三个真实陷阱；全库普查判据「只有 key 含 `self` 的实例方法才需要 `weak_lru`」；
  纯 Python、H100(sm90) 真机 **39 PASS / 0 FAIL**；索引页新增第 123 篇。
  上一轮 P112-1：P0~P112 已完成（最新 **P112-1 DeepEP 两类非 all-to-all 原语（PP send/recv 与
  Engram fetch）的验收测试即正确性规范**：PP 的 `generate_stress_ops` 在途背压无死锁调度、
  Engram 的 all-gather+gather 参考 oracle 与 SF 复制；kernel 端到端仅分析、测试侧 H100(sm90)
  真机 **72 PASS / 0 FAIL**；索引页新增第 122 篇。
  上一轮 P111-1：P0~P111 已完成（最新 **P111-1 DeepEP 四个 Buffer 家族共享的生命周期契约
  `BufferBase`**：Python ABC 与 C++ pybind **两条平行继承链**、`destroy_on_destruction` 三分支
  状态机、`~BufferBase() noexcept(false)` 的用意、二次 `destroy()` 的跨语言分歧（Python 幂等 /
  C++ 抛异常）；纯 host、架构无关，H100(sm90) 真机 **32 PASS / 0 FAIL**；索引页新增第 121 篇。
  上一轮 P110-1：P0~P110 已完成（最新 **P110-1 Mega MoE 主 kernel 的任务调度器
  `scheduler/mega_moe.cuh`（481 行）**：补全四个调度器的最后一格——行容量公式
  `get_num_max_live_pool_blocks`/`get_num_l1_warmup_waves`（host ring 分配的唯一几何契约）、
  `L2KBlockDependency` 的 XOR 位图 + ring 世代奇偶、`create_task` 的 warp 现算池块映射、
  locality-domain 分域 + 工作窃取、发布/消费顺序契约（只有 epilogue release）；
  H100(sm90a) 真机 **34 PASS / 0 FAIL**；索引页新增第 120 篇（并修掉索引页重复的第 119 条）。
  上一轮 P109-1：P0~P109 已完成（最新 **P109-1 DeepEP 的 NVLink 连接自检
  `check_nvlink_connections` 与 H100 P2P 拓扑**：字符串门 / PCIe 硬上限 / NVML 8×8 能力矩阵 /
  `can_device_access_peer` / `cudaMemcpyPeerAsync` 带宽矩阵；H100 真机 16 PASS / 0 FAIL；
  索引页新增第 119 篇）。
  上一轮 P108-1：P0~P108 已完成（最新 **P108-1 DeepEP 的测试 / 进程引导层**：`init_dist` 三件 host 事
  + 返回新建子组 + `device_id` 兼容、`setdefault` 压 SL 到 1、`init_seed` 的 `local=global+rank`、
  `get_local_seed` 死 API、`dist_print` 静音仍屏障、`EnvOverride` 三元 enable + 精确回滚；
  H100 真机 30 PASS / 0 FAIL；索引页新增第 118 篇。详见上）。
  上一轮 P107-1：P0~P107 已完成（最新 **P107-1 DeepEP 的环境变量 / 开关面**：
  persistent envs 只补不覆盖、`get_env<T>` 的未初始化栈垃圾、全库 16 个带引号 `EP_*`（11 条运行时
  开关）、`EP_BUFFER_DEBUG` 跨语言分歧、`EP_NUM_MAX_LOCAL_RANKS`/`NCCL_WIN_STRIDE`、`check_nccl_so`
  三把尺子；H100 真机 38 PASS / 0 FAIL；索引页新增第 117 篇。可继续（均为增量、非承诺）：
  第 65 篇 §8 列出的 `envs.py` 带宽/域解析（第 73/86 篇已部分覆盖；envs.py 的 `init_dist`/`init_seed`/
  `EnvOverride`/`semantic.py` 仍可再收一网），或从 P96/P97 的轻微漂移清单挑关键路径定点复核。详见上）。
  上一轮 P106-1：P0~P106 已完成（最新 **P106-1 DeepEP MoE 路由夹具 `deep_ep/utils/gate.py`**：
  `generate_topk_idx` 精确逆映射、`generate_rank_count` 用循环重复排列铺普通 rank（precise 均衡）、
  `map_unbalanced_ratio_to_factor` 单标量二分的大 R 阈值开关与不可行静默饱和、可行性闸门只在
  precise 路径；H100 真机 ALL PASS，复刻 test_gate 判据 precise 45/45 vs random 14/45；
  索引页新增第 116 篇。详见上）。
  上一轮 P105-1：P0~P105 已完成（最新 **P105-1 DeepEP `refs.dispatch` 的 FP8 分支 + 设备端
  `sf_pack_t`/`TokenLayout` SF 布局**：参考 gran-128 契约 vs 设备缓冲 gran-32 上界（4 倍）、
  dispatch 列数取 `sf.size(1)`、union 两成员统一「一列=128 元素」、DeepEP 不打包 UE8M0；
  H100 真机 8 卡 all_to_all 逐位 + host 调真实 `TokenLayout` 逐字段 0 不符；`combine` 无 FP8 变体；
  索引页新增第 115 篇。详见上）。
  上一轮 P104-1：P0~P104 已完成（最新 **P104-1 DeepEP 的 MoE 语义工具层 `deep_ep/utils/math.py`**：
  第 65 篇只提一句——判据 `calc_diff`、参考量化器 `per_token_cast_to_fp8/back`、top-k 专家去重
  oracle `inplace_unique`、group-limited 路由掩码 `create_grouped_scores`、记账；三组「同名不同契约」
  跨库对照（`calc_diff` 差 `+1`、量化器不打包不 UE8M0、`count_bytes` 逐字相同）；发现 `:66` 按 count
  排序是死代码；`math.py` 只 import torch，H100 33/33 ALL PASS；索引页新增第 114 篇。可继续（均为
  增量、非承诺）：第 65 篇 §8 已列的 `refs.py` FP8 分支与 `envs.py` 带宽/域解析，或从 P96/P97 的轻微
  漂移清单挑关键路径定点复核。详见上）。
  上一轮 P103-1：P0~P103 已完成（最新 **P103-1 SM100 dense MQA logits 主 kernel 逐段精读**：
  第 23 篇 §5 只给一页摘要——16 warp 五类角色、`kNumTmemStages==3`、五段架构无关逻辑 H100 ALL PASS
  （`Int<max(1,min(N,BQ))>` 折射 / 16-8-4 头块 / bf16x2 归约 / 分页两生产者均分 / 4×32→32×4 转置）；
  UMMA/UTCCP SM100 仅分析；索引页新增第 113 篇。详见上）。
  上一轮 P102-1：P0~P102 已完成（最新 **P102-1 DeepEP bucket all_gather 的两个 SM kernel 后端
  pure-RDMA / hybrid**：第 54 篇只把纯 NVLink copy-engine 路径讲透、第 62 篇补了 all_reduce/reduce_scatter
  三后端，本篇补齐 all_gather 的跨节点两条腿——`launch_all_gather` 三物理域分派、pure-RDMA 每 warp
  一个目标 rank 的 `gin.put`、hybrid 的 SM=目标 RDMA rank / warp=QP 与每 chunk 8-QP 均分、显式代价模型
  自选 chunk（每 QP ≥512KiB）、`(chunk,QP)` tail+GEQ 完成协议；数据面 NCCL GIN 仅分析，信号布局/寻址/
  协议架构无关，H100(sm90a) 直接 include 真实头文件跑真实设备代码 + 真实驱动 mem-op，设备 4 部分 +
  Python host 算术层全 OK；索引页新增第 112 篇。详见上）。
  上一轮 P101-1：P0~P101 已完成（最新 **P101-1 MoE 反向的 H100 数值复刻与误差分解**：
  第 14 篇只答「kernel 接得对不对」、第 99 篇拆了前向误差分解——本篇补上**反向对偶**：四个 grouped
  GEMM 的角色、参考量化器做的**可加来源分解**（前向量化主导 ~80–90%、`dW2` 对反向后重量化免疫、
  `dX` 最敏感）、真实 SM90 kernel 锚点（`vs deq` dW 2.6e-4 复现第 14 篇）、DeepSeek 量级最坏
  `full` 下 cos>0.99；主 kernel SM100 仅分析，反向数学/量化/布局 H100 真机；详见上）。
  上一轮 P100-1：P0~P100 已完成（最新 **P100-1 SM100 UMMA 指令底座 `ptx/tcgen05.cuh`**：
  第 49 篇 SM90 WGMMA 底座的 SM100 对偶——9 个 `*_SS::fma`（`cta_group::1/2`、`kind::`、`desc>>32`
  =instruction descriptor、`setp.ne.b32` 累积谓词）、`tmem_load_32dp32b<N>`、`umma_arrive_no_elect`、
  两条 `tcgen05.fence`；消费者 `sm100_bmk_bnk_mn.cuh`（把第 17 篇 §6 落成最小样本）；H100 真机
  7 PASS/0 FAIL（ptxas 编译矩阵 sm90 逐条拒 / sm100 全过、STSM swizzle==`Swizzle<3,4,3>`、
  真实 TMA reduce-add 0.5+3×1.0→3.5）；详见上）。
  上一轮 P98-1：P0~P98 已完成（最新 **P98-1 DeepGEMM 通用 GEMM 任务调度器 `scheduler/gemm.cuh`**：
  一个 `Scheduler` 模板覆盖 7 种 `GemmType`，被五个 GEMM 内核复用（也是 MoE 反向 wgrad/dgrad 的
  任务枚举地基）；三个非平凡点——`get_num_1d_blocks_per_group` 编译期代理代价选 `{8,16}`、
  `get_swizzled_block_idx` 的 SM90 multicast 奇数分组修补 `#if __CUDA_ARCH__<1000`、
  `get_next_k_group` 的物理 K 起点对齐 + SF-K 前缀和；另拆 `get_global_idx`/`get_aligned_effective_m`
  /SM90 multicast 与计算有效性。H100(sm90) 直接 include 真实头文件跑真实设备代码 **51 PASS / 0 FAIL**，
  SM100 分支仅编译期覆盖；详见上）。
  上一轮 P95-1：P0~P95 已完成（最新 **P95-1 Mega MoE 的 FP4 路由权重路径**：
  `fp8xfp8`/`fp8xfp4` 同走 `MmaKind::MXFP8FP4`，差异只在路由权重 dtype（`kPackedFP4=int8`、每字节 2
  个 e2m1）；TMA `16U4_ALIGN16B` 把 4-bit 摊成未打包形式、SMEM 每元素 1 字节（线字节减半）；A=FP8、
  B=FP4 必须走 `MXF8F6F4`（`UMMA_K=32`）；权重重排 FP8/FP4 共用 tuple 路径且与打包逐位可交换；
  host 唯一 FP4 专属位是小 store block（prefill）。主 kernel 仅分析，H100 真机 ALL_OK=True；
  详见上）。
  上一轮 P94-1：P0~P94 已完成（最新 **P94-1 MoE 反向的通信量模型与 H100 实测**：
  反向通信 ≈ **1.5× 前向**（布局不对称：dispatch 去重 T·d·1B vs combine/两个反向相位 expanded
  T·k·2B），模型 4k/(d+2k)≈1.49，H100 8 卡 ragged a2a 实测字节量比 1.49×/时间比 1.46–1.54×、
  单向 114–124 GB/s；反向无 mega-kernel 的真因是「四个独立 GEMM + 两张方向相反的网」，
  重叠交给 `async_with_compute_stream` + 限制通信 `num_sms`；主 kernel 仅分析，通信原语真机可跑；
  详见上）。
  上一轮 P93-1：P0~P93 已完成（**P93-1 SM100 BF16 GEMM 的 `kSwapAB`（dgrad）分支**：
  host 启发式对 m-grouped 返回唯一 `swap_ab=1/block_n=128`、`block_m=mk_alignment`(128/256)，
  设备端 A/B 换轴 + 逐块动态 `UMMA_N`，转置写回 `sm100_store_cd_swap_ab`（16×128 store + 转置 STSM）；
  主 kernel SM100 专属仅分析，H100 host 半边真机 56 PASS / 0 FAIL；详见上）。
  上一轮 P92-1：P0~P92 已完成（**P92-1 Mega Gate / Mega mHC 的跨 CTA 分段屏障调度器与工作区布局**：
  `scheduler/mega_gate.cuh` 的 `ScoreBarrier` + `scheduler/mega_mhc.cuh` 的 `SplitBarrier`/`Norm`/`Mix`
  ——状态编码 `(grid 标签)*stride + 到达数`、`%gridid` 免复位、`wait_init` 无符号距离、`arrive` 的
  `red.async` SM100 专属、进位隔离、128B line、两个 `Workspace` 几何、四个角色调用链与 deterministic
  差异；除一行 `red.async` 外全架构无关，H100 真机 31/31 ALL PASS；详见上）。
  上一轮 P91-1：P0~P91 已完成（**P91-1 DeepEP 屏障的验收基准与「通信 kernel 精确测时」方法论**：
  `tests/comm/test_barrier.py`（测试套件里最后一份未覆盖文件）+ `bench_kineto(barrier_comm_profiling=...)`
  路径——域 `allow_hybrid_mode` 语义、host 链路、`launch_barrier` 公式、设备 sequential/并行内核；
  设备 kernel 走 NCCL GIN 仅分析；H100 真机 Part B/C/D/E 全 PASS；**Part F 发现 profiler 测时双峰
  （无 sleep 时快档 ~100us / 慢档 ~417us ≈4×，barrier 稳定落快档）且坐实为持续满载降频**；详见上）。
  上一轮 P90-1：P0~P90 已完成（最新 **P90-1 DeepEP EPLB 冗余权重的验收测试即正确性规范**：
  `tests/ep/test_prefetch_weights.py` + `test_reduce_grads.py`——前向 blob 唯一 key ramp + 哨兵、
  映射表 own-expert 排除；反向小整数精确 fp32 使原子加顺序无关、`torch.equal` 免容差；设备路径
  NCCL GIN 仅分析，H100 真机 37/37 ALL PASS（push 920 槽 / pull 176 组 0 不符、关键路径 671 MB /
  NVLink 430.3 GB/s / 下限 ~1560 us）；详见上）。
  上一轮 P89-1：P0~P89 已完成（最新 **P89-1 Mega MoE 前向流水线在 H100 上的数值复刻与误差分解**：
  用 SM90 `m_grouped_fp8_gemm_nt_contiguous`（1D2D）+ bf16 grouped kernel 逐段复刻
  dispatch→L1→SwiGLU→L2→combine；把 top-k 权重在 L1 输出/L2 量化前乘、combine 只求和；端到端
  fp8-kernel vs fp64 rel 6.28–6.34e-2（bf16 5.0e-3 / fp32 5.1e-4）、分层 3.2e-4、消融仅激活 5.70e-2
  / 仅权重 4.58e-2 → 量化主导且两侧同量级；吞吐 L1/L2 312/303 TFLOPS；主 kernel SM100 仅分析，
  H100 真机跑整条前向；详见上）。
  上一轮 P88-1：P0~P88 已完成（最新 **P88-1 DeepEP 的编译期配置面与 DeepJIT 接入**：
  `csrc/runtime/jit.hpp::init_jit` 的 `Config` 五参 + 8 条 nvcc flag（GIN GDAKI）+ `compiled.cuh`
  的编译期 ABI（`topk_idx_t`/`sf_pack_t`/对齐/通信上限）；核心洞察是「NCCL include 路径进缓存键、
  DeepEP 自身头按内容进键」；纯 host/编译期、架构无关，H100 真机 48 PASS / 0 FAIL；详见上）。
  上一轮 P87-1：P0~P87 已完成（最新 **P87-1 SM100 HyperConnection prenorm GEMM 的 JIT 启动包装层与三段流水**：
  host 包装层 `n<=128`/`block_n=align(n,16)`/barrier 家族 4 组(split 时 D 走 3D)、SM100 kernel 的
  cast-and-reduce warpgroup 与 `get_swizzled_smem_offset`；H100 真机 21 PASS / 0 FAIL；详见上）。
  上一轮 P84-1：P0~P84 已完成（最新 **P84-1 DeepEP bucket 混合集合算子的验收测试即正确性规范**：
  `tests/bucket/test_mixed.py`（373 行）+ `tests/bucket/test_perf.py`（273 行）——混合算子的多轮随机
  编排（三流 / scale / exact / session / 外部输入）+ 五个规范组件（`supported` 门控、`make_sizes`
  `quantum=lcm(g·32)`、`make_schedule`、`make_groups` 拓扑、`nccl_reference`）与逐位判据 `verify`；
  perf 的 NCCL profile 矩阵与慢 rank 中位数 `measure`；端到端依赖 NCCL GIN 仅分析，测试侧纯 Python/
  张量 H100 真机 **33 PASS / 0 FAIL**；详见上）。
  上一轮 P83-1：P0~P83 已完成（最新 **P83-1 DeepEP BucketBuffer 的会话机制（`BucketSession`）**：
  `deep_ep/buffers/bucket.py:19-181` + `csrc/buffers/bucket.hpp:48-56/150/213/279/314`——常规集合算子
  要求张量注册在对称 buffer 里，`BucketSession` 用「实例属性遮蔽类方法 + 64B bump 分配器 + 越界 src
  仅纯 NVLink all-gather 透传 + 结果回拷 + `record_stream` 生命周期守卫」把约束藏起来；纯 host Python，
  H100 真机（真实源码 + FakeBucket 承接 runtime）**31 PASS / 0 FAIL**；详见上）。
  上一轮 P82-1：P0~P82 已完成（最新 **P82-1 DeepGEMM 的共享内存布局结构层**：
  `layout/gemm.cuh` + `layout/mqa_logits.cuh`——`SharedStorage` 用「成员顺序 + `alignas`」把共享内存
  布局变成类型属性，host 两条策略（MQA 精确镜像 + JIT `static_assert` / 稠密 GEMM 保守预算 824B）；
  H100 直接 include 真实头文件逐字段核验 **42 PASS / 0 FAIL**，`used<=host` 且 `slack==824−barrier`
  精确；`reserved_barriers` 让 BF16 每 stage barrier 数对齐 block-scaled 布局；详见上）。
  上一轮 P81-1：P0~P81 已完成（最新 **P81-1 官方 SF 契约的代码对照与 H100 一致性核验**：
  以 `docs/scaling-factor-format.md` 为准核验 host 契约代码——两条路径（自动变换 vs 预变换只校验）、
  §3 int32 预变换 stride 契约、§5.2 K-grouped 预打包（仅 gran_k=32、下界 `>=`）、§6.2 einsum
  「按 post-permute 坐标变换再换回」、§6.3 Mega MoE 四条更硬要求；`check_sf_layout` /
  `check_k_grouped_packed_ue8m0_tensor` 无 arch 门控，H100 直接 include 真实头文件跑真实契约函数，
  **19 PASS / 0 FAIL**（负用例打印真实 `file:line`）；详见上）。
  上一轮 P80-1：P0~P80 已完成（最新 **P80-1 两库的「启动前地基」：驱动符号懒加载 / 版本闸门 / 张量视图 helper**：
  DeepEP `csrc/utils/lazy_driver.hpp` + `tensor.hpp`；DeepGEMM `csrc/utils/compatibility.hpp` + deep_jit
  `utils/lazy.hpp` / `backend/cuda/driver.hpp` / `utils/exception.hpp`；以及最后一个未覆盖文件
  `sm100_locality_domain.hpp`。两库两套 dlopen/dlsym 宏对照、`EP_CUDA_STRINGIFY` 别名展开、`LazyInit<T>`
  终态语义、版本闸门分层（DeepGEMM 12.9 vs DeepJIT 12.4）；纯 host、架构无关，H100 真机直接 include
  真实头文件：50 符号全解析、VMM+POSIX fd 跨句柄 2 MiB mismatch=0、libdw 8/8、伪造低版本被 static_assert
  拦下，**30 PASS/0 FAIL + 1 负样本**；详见上）。
  上一轮 P79-1：P0~P79 已完成（最新 **P79-1 Mega MoE / mHC 的 JIT 启动包装层**：
  `sm100_bf16_mega_moe.hpp` / `sm100_mega_mhc.hpp`——host 张量→TMA 描述符→JIT 模板实参→launch；
  描述符 swizzle 随元素字节/通道宽度变化的补偿、常量/开关进模板、回退指针别名、mHC 的 route 当第三维
  + `MixArgs`/`NormArgs` POD；主 kernel SM100 仅分析，wrapper 纯 host，H100 真机 **50/50 ALL PASS**；
  详见上）。
  上一轮 P78-1：P0~P78 已完成（最新 **P78-1 官方 profiling 基础设施 + 自动生成的覆盖面清单**：
  `scripts/run_ncu_mega_moe.sh` / `tests/test_mega_moe.py` 的 `--ncu-profile-only`/`--local-rank-idx` /
  `scripts/quick_plot_pm.py` / `scripts/generate_pyi.py`+`stubs/_C.pyi` / `csrc/indexing/main.cu`+CMake——
  多进程 NCU 编排、PM-sampling 时间序列解析（raw CSV 第 11 列）、62-def 类型桩（漏 10 个 `m.attr`）
  与 IDE 索引单 TU（17/19 impl）；**发现 pyi 生成器不可复现 bug**（`rglob` 顺序决定抓定义还是调用点）；
  目标 kernel SM100+DeepEP 仅分析，外壳 H100 真机 **47/47 PASS**；详见上）。
  上一轮 P77-1：**跨 SM / 跨 rank 同步原语 `comm/barrier.cuh`**：
  `deep_gemm/include/deep_gemm/comm/barrier.cuh` + `layout/sym_buffer.cuh`——`grid_sync` 的
  「和恰好 2³¹、只翻最高位、免复位」与 `nvlink_barrier` 的 `counter&3`→(phase,sign) 状态机
  (+N 加满 / −1 清回 0)、`wait_until` 超时自旋、`cluster_sync_with_relaxed_arrive`、`SymBuffer`
  相对偏移；架构无关设备代码，H100 真机 grid_sync ×2000 轮 + 真实 8 卡 ×30 轮 **ALL PASS**；详见上）。
  上一轮 P76-1：**DeepEP 的 NCCL 通信域绑定与 Python 引导层**：
  `deep_ep/__init__.py` / `find_pkgs.py` / `comm/{handle,barrier}.py` / `csrc/kernels/comm/context.cpp`
  / `setup.py`——引导协议、三闸门 `check_nccl_so`、`Context` 构造（版本闸门/GIN/两域拆分/对称
  window）、`get_sym_ptr`/`finalize`、Python comm 层；纯 host/Python，H100 真机 **35 PASS / 0 FAIL**，
  把本机「运行期阻塞」钉到行号（loaded 系统 NCCL 2.29.7 vs linked pip 2.32.3）；详见上）。
  上一轮 P75-1：**DeepEP 的 EP dispatch/combine 端到端验收测试即正确性规范**：
  `tests/ep/test_ep.py`（616 行）——模式矩阵 `enumerate_ep_modes`（六维，实测 72/36/36/18）、五种
  dispatch + 两种 combine、逐位判据（`torch.equal` + 排序后比 ref + 权重按 `-1` 屏蔽）、`fold_expanded`
  不变量、handle 两套 psum 恒等式、`get_unique_and_valid_dst_count` 去重计数、`EPHandle` 字段契约；
  端到端 buffer 仅分析，测试侧 H100 真机 **17 PASS / 0 FAIL**；详见上）。
  上一轮 P74-1：**DeepGEMM 的 Python 绑定面**：
  `csrc/python_api.cpp:24-49` + `deep_gemm/__init__.py:16-127`——`_C` 顶层 76 名对账 0 多余 0 缺失、
  `compiled_dims`/`recipe`/`alpha` 三组默认值、`m.attr` 别名同一函数对象、两子模块与隐藏面；
  H100 真机 **51/51 PASS**；详见上）。
  上一轮 P73-1：**DeepEP 的 `EPHandle` 句柄契约与 dispatch/combine 的 host 编排层**：
  `deep_ep/buffers/ep.py:27-906`——`EPHandle` 的两个身份（grouped GEMM 的 PSUM 布局
  接口 / 反向缓存）、两种 psum 语义（expand 的 `align(psum[i-1])+未对齐真实计数`）、`topk_idx` 版本守卫、
  dispatch 五段编排（守卫 / auto SM+QP / SF 解包+缓存句柄三断言 / 默认回填+调 runtime / finalize 与
  确定性钩子）+ deferred epilogue、combine 复用 `handle.num_sms`；H100 真机注入 `_C` 桩后 exec 真实
  文件、把未修改真实方法体绑假 buffer 逐字段核对 **51 PASS / 0 FAIL**；详见上）。
  上一轮 P72-1：**legacy Triton grouped GEMM 的验收测试即正确性规范**：
  `tests/test_legacy.py`（95 行、两个顶层函数、无 arch 门控、只测 bf16）——m-grouped 32 配置（仅 16
  唯一）+ `expand×test_alias` 128 次发射、k-grouped 四因子闸门；H100 真机 **19 PASS / 0 FAIL**；详见上。
  上一轮 P71-1：**SM 局部域的验收测试即正确性规范**：
  `tests/test_locality_domain.py`（167 行，纯 host / API 层，一个 GPU kernel 都不启动）——5 个顶层函数、
  `assert_raises` 只吞 RuntimeError（域越界 vs MLOPart 不可用分成两种异常）、域表四断言（TPC 成对 /
  恰好 50/50 / 只搬最少）、`balance` 的 `slice(0,n/2)` 是 start 不是长度（`surplus` 取后半段，读错被断言③抓）、
  `test_without_mlopart` 的优雅退化契约；H100 真机原样跑 2 个可跑用例 + 独立复刻逐位相等
  **16 PASS / 0 FAIL**；详见上）。
  上一轮 P70-1：**DeepGEMM 顶层运行时的验收测试即正确性规范**：
  `tests/test_lazy_init.py`（架构无关，H100 整份可跑）——用 `LD_PRELOAD` 探针把 cuBLASLt 换成可计数
  假库，三个 magic exit code 锁死退出期 use-after-free / 泄漏 / 重复销毁；6 用例矩阵（假 handle 绕开
  CUDA）+ `MALLOC_PERTURB_` 逼出退出竞态 + workspace 退出释放 ≥2×16 MiB + fork safety 8/8；
  H100 真机原样跑真实测试 + 独立复刻 **7 PASS / 0 FAIL**；详见上）。
  上一轮 P69-1：**HyperConnection prenorm GEMM 的验收测试即正确性规范**：
  `tests/test_hyperconnection.py`（`@test_filter(arch>=9)`，H100 可跑）——判据编码实现事实
  （`allow_tf32=True` 是「kernel 是 TF32」的镜像、`a.float()@b.T` 只转 A、`2*m*n*k` 不含 S）；
  H100 真机直接驱动真实测试体 24 用例全过 **8 PASS / 0 FAIL**；详见上）。
  上一轮 P68-1：**SF layout 家族的验收测试即正确性规范**：
  `tests/test_layout.py`——纯 torch oracle + 两档判据 + k-grouped 每组套单组；H100 真机 8 PASS / 0 FAIL；
  详见上）。
  上一轮 P67-1：**DeepGEMM 的共享 TMA 拷贝层与环形流水线**：
  `common/tma_copy.cuh` + `common/ring_pipeline.cuh`；H100 真机直接跑真实设备代码 **27 PASS / 0 FAIL**；
  详见上）。
  上一轮 P66-1：**DeepGEMM einsum 家族的验收测试即正确性规范**：
  `tests/test_einsum.py`——七个函数 / 两个 arch 门控 / `calc_diff` 阈值表 / 逐位 oracle（direct==FP32
  累积、随机舍入哈希、FP8 输出对==生产管线）；H100 真机直接驱动真实测试体 **15 PASS / 0 FAIL**；
  详见上）。
  上一轮 P65-1：**DeepEP 的通信—计算重叠「Python 调度地基」**：
  `EventOverlap` / `get_comm_stream` / `BufferBase` + C++ 孪生 `EventHandle` / `stream_wait` +
  交接点 `stream_control_prologue/_epilogue`；纯 host/Python，H100 真机 exec 真实代码 **21 PASS / 0 FAIL**；
  核心实测：全卡通信+全卡计算**不重叠**（0.989×）、通信缩到 1/4 SM 则 **1.776×**，坐实第 63 篇半卡上限；
  详见上）。
  上一轮 P64-1：**稠密 GEMM 的验收测试即正确性规范**：
  `tests/test_fp8_fp4.py` / `tests/test_bf16.py`——`__main__`/arch 门控矩阵（H100 枚举 252 用例）、
  逐函数「从松到严」判据链、`calc_diff` 两处盲区；H100 真机 18 PASS / 0 FAIL；详见上）。
  上一轮 P63-1：**DeepEP dispatch/combine 的自动 SM/QP 选择模型 + 带宽探测层**：
  host 侧解析式 roofline（四本每 token 流量账 + 期望命中组数 + 瓶颈链路标尺 + 每 SM HBM 带宽换算），
  纯 host、H100 真机 ast 抽取真实方法体跑 **16 PASS / 0 FAIL**；详见上）。
  上一轮 P62-1：**DeepEP bucket 的 `all_reduce`/`reduce_scatter`**：
  三后端（NVLink multimem / RDMA 反向环 / hybrid 三级）+ 信用环 head/tail 背压 + bf16 随机舍入归约；
  数据面 NCCL GIN / `cvt.rs` 仅分析，布局/映射/归约/hash/host 算术在 H100 真机跑**真实设备代码**
  **30 PASS / 0 FAIL**；见上）。
  上一轮 P61-1：**DeepEP 的两级（hybrid）EP dispatch/combine**
  （`hybrid_dispatch.cuh` + `hybrid_combine.cuh`，即 Mega MoE 在 EP8/EP16 上真正跑的两级域路径）：
  数据面 NCCL GIN 仅分析；就绪哨兵 / tail 增量 / `do_psum` 前缀和 / round-robin 分片调度四个架构无关
  内核在 H100 真机跑**真实设备代码**，设备 7 PASS / Python 6 PASS / 0 FAIL；详见上）。
  上一轮 P60-1：**DeepGEMM 内存安全验证层 `tests/test_sanitizer.py`**：
  compute-sanitizer 驱动的独立子进程测试栈 + 六 env/六 flag 的「为什么」；H100 真机 30 PASS 含
  racecheck/synccheck/memcheck 负样本与缓存分配器屏蔽实证；见上）。
  上一轮 **P53-1 DeepEP 的多 rank 屏障与 symmetric-memory 句柄层**：
  每个 dispatch/combine/bucket kernel 起止都调的 `comm::barrier`——`NCCLGin` 对称窗口句柄
  （`get_sym_ptr` 同 offset 翻译）、`gpu_barrier` 两级域编排（scaleup NVLink + scaleout RDMA）、
  `nvlink_barrier_wo_local_sync` 的**免复位双相位协议**、`get_qp_mode` 的 SM↔QP 映射、
  `ginSignalCount=num_ranks+4` 状态预算；NVLink 腿逐行抽取 + 对称指针桩在 H100 跑真实设备代码
  设备 19 PASS / Python 17 PASS / 0 FAIL；scaleout RDMA 腿与并行混合编排仅分析；见上）。
  上一轮 P52-1：**DeepEP 的确定性 dispatch 排序 `EPHandle.deterministic_sort`**：
  dispatch 槽位由 `atomicAdd` 分配、行序不确定，`deterministic=True` 时改排回源 token 索引序
  （非 expand 全局排序；expand `expert*1e10+src` 双键稳定排序）；纯 host/Python、架构无关，
  H100 上 `ast` 逐行抽取真实源码运行 **7 PASS/0 FAIL**；见上）。
  上一轮 P51-1：**DeepEP dispatch 的拷贝 epilogue**：
  `dispatch_copy_epilogue_impl`——主 dispatch 之后的第二段 kernel，把对称缓冲里的 token 落盘成下游张量，
  靠 PDL 衔接；一个 kernel 覆盖非 expand/expand(+cached)/零填充/链式列表五形态；整段不碰 NCCL，
  H100 真机直接 include 真实头文件 21 PASS/0 FAIL；见上）。
  上一轮 P49-1：**P49-1 DeepEP 的设备端公共层** `common/ptx.cuh` 等：
  mbarrier 五原语 + `cp.async.mbarrier.arrive` 计数模型、TMA 三条搬运路径（含 `cp.reduce...add.f32`）、
  NVLink 多播 `multimem.*`、谓词化访存 + `iterate_ld_st`、`gpu`/`sys` 两档内存序、
  `warpgroup_reg_realloc`、warp 内建、浮点打包 + `encode_decode_positive`/`BarrierSignals`；
  H100 真机直接 include 真实头文件跑真实设备代码 41 PASS/0 FAIL；`multimem`/`cvt.rs` 仅语义 oracle；见上）。
  上一轮 P48-1：**P48-1 DeepEP combine 的归约 epilogue**：
  补上 `docs/07` 一笔带过的 combine 归约侧（`combine_reduce` 家族）——两阶段 + PDL、rank/topk
  双布局（主 kernel 与 epilogue 共用同一 constexpr）、top-k 去重（`deduplicate`+`compute_topk_slots`+
  `-1` 谓词跳过）、bf16 直加旁路 vs fp32 累加；H100 真机直接 include 真实头文件跑真实
  `combine_reduce_epilogue_impl`（真实对称 recv_buffer）16 PASS/0 FAIL，20 万随机样本坐实旁路为何卡在
  上界 2；见上）。
  上一轮 P46-1：**P46-1 TMA 描述符构造层 `runtime_utils.hpp` 的 `make_tma_*`**：
  128 字节 `CUtensorMap` 是「确定性参数翻译」——major→inner/outer、dtype/swizzle 映射、三个隐藏规整；
  H100 真机 16 个包装用例与直连编码逐字节相等、真实 TMA 载入与 `cute::Swizzle` 预测布局全对，
  并发现 **swizzle XOR 键取自绝对 smem 地址 bits[7:10)**（目的地要对齐到整个 swizzle 原子）；见上）。
  上一轮 P45-1：**P45-1 per-warp `%clock64` 追踪探针 `common/probe.cuh`**：
  一段**公开版无调用点、JIT 也不启用**的休眠观测设施——三处「为什么」（4 GiB 窗口约束源于
  `advance` 单条 `IADD3` 只加低 32 位；负偏移回填让 ptxas 保留基址加法；`mark` 用 `%globaltimer`
  挂钟标定 SM 频率）；`DG_PROBE_ENABLE=0` 时整段消除；全架构无关内联 asm，H100 真机同源编两个
  二进制 + numpy 解析记录流 **ALL PASS**（含伪指针实证 advance 高 32 位不进位、标定 1.98 GHz、
  纯记录 ~5.7 ns/条）；见上）。
  上一轮 P44-1：**P44-1 DeepGEMM 的设备端数学与工具层**——
  补上每个 kernel 都在调、却从没系统讲过的 `common/{math,utils,types,packing,...}.cuh`——
  UE8M0 取整的整数进位技巧（与 Python `ceil_to_ue8m0` 逐位等价）、`SF*SF_inv` 精确为 1、
  bf16/FP8 打包、`fma2`/`fast_rcp`、`warp_reduce` 的 intra/inter 两种语义、`cta_exclusive_sum`
  与 TMEM 列取整；**H100 真机直接 include 真实头文件跑真实设备代码 ALL PASS**；见上）。
  上一轮 P43-1：新建 `docs/53-deepep-eplb-redundancy.md`（+`src/eplb_redundancy.cu`、
  `harness/exp_eplb_redundancy.{sh,py,out.txt}`）——补上专题「EP 反向通信」的空档（EPLB 冗余专家的
  权重预取与梯度归约；两个 kernel 用 NCCL 设备 API，本机不提供 → 仅分析，切块/公式/归约语义
  H100 真跑真实设备代码 ALL PASS）。上一轮 P42-1：DeepGEMM 的 kernel 发射层（PDL/cluster/动态
  smem + 设备端 grid 依赖同步；H100 真机半格 PDL 1.95×、缺同步读 stale 反例等）。
  上一轮 P41-1：P0~P41 已完成（新增 **P41-1 Mega MoE 的端到端测试与逐位正确性判据**：
  补上 `tests/test_mega_moe.py`——`run_fused` vs `run_baseline` 用 `torch.equal` 逐位验收、
  三个测试侧 SF 布局变换（TMA 列主序对齐 / 共享专家 UTCCP 4×32 交错 / padding 广播）、
  性能账目公式（FLOP=3、fp4 权重 0.5B、overlap factor）；**H100 真机 37 PASS**（直接 import
  真实 test 复用 helper）；端到端 fused 对比 SM100 专属且 oracle 依赖 DeepEP，仅分析；见上）。
  上一轮 P40-1：P0~P40 已完成（新增 **P40-1 DeepGEMM 运行时对象生命周期与资源所有权**：
  补上第 48 篇只当「字段容器」的 `Runtime` 的 lazy init / shutdown / cuBLASLt handle 与 per-stream
  workspace 归属——`LazyInit` 的「null factory = 终态」、`shutdown()` 一行赋值即永久置空且 `init`
  不可恢复、handle 默认自建自毁、workspace 每 stream 一份 16 MiB、`atexit(shutdown)` 规避退出
  use-after-free；**H100 真机 24 PASS**（探针 create/destroy/clean、显存账 16/16/32、上游寿命测试
  OK）；见上）。
  上一轮 P39-1：P0~P39 已完成（新增 **P39-1 SM90 WGMMA 层：共享内存描述符、选择器与异步组**：
  补上所有 SM90 kernel 每一行 MMA 都在调、却从没单独讲过的底座（`mma/sm90.cuh` + `ptx/wgmma.cuh`）——
  64 位 `GmmaDescriptor` 五段位域与 `to_gmma_layout_type` 把字节 swizzle 编码进描述符、
  `make_gmma_desc` 的 K/MN 两分支闭式、`advance_gmma_desc_lo` 三段步进、三个选择器 `M=64`/`K`=32·16·8、
  `ptx/wgmma.cuh` 异步组、`CLayout_64xN` 列主序累加器映射；**H100 真机直接跑真实 WGMMA**，
  描述符位域 13 项全对 + 5 个 GEMM 逐位相等；见上）。
  上一轮 **P38-1 DeepGEMM 的全局运行时/编译开关面**：
  两份可变全局 singleton（`Runtime`+`HeuristicsRuntime`）→ `config.hpp` pybind 面；七个开关的
  「状态/消费点/为什么」——`num_sms` 顺带关 cuBLASLt 回退、`tc_util==0` 当默认 100、
  `deterministic` 三落点（关回退 + 禁 split-K + 钉死 mHC splits，batch invariance）、
  `block_size_multiple_of`/`mk_alignment` 改候选集、`ignore_compile_dims` 一核多形状、
  `set_pdl` 落发射属性；`DG_*` 两级前缀解析 + 安装时 `persistent_envs`；H100 真机 25 PASS；
  见上）。
  上一轮 **P37-1 DeepJIT 的 post_hook 后处理 + 磁盘产物 + CUDA↔Ascend 后端抽象**：
  第 33 篇缓存键「第四段」终于补齐——`nvcc` 出 cubin 后、发布前跑 `python <hook> <cubin>` 就地改写；
  「路径+内容」FNV1a 进键且按路径线程内缓存；override `nullopt`=继承、空串=已配置；由发布顺序推出
  「hook 改的 cubin 即最终产物 / 失败不发布 / 命中不重跑」；`post_hook` 是 CUDA 专属（Ascend 恒空哈希）；
  H100 真机直接 include 真实头文件跑 8 组 ALL PASS；见上）。
  上一轮 **P36-1 GEMM API 的输入主序（nt/nn/tn/tt）与输出累积契约**：
  `nt` 唯一实现 + 三个转置薄壳；major 判定与 SM90 K-major 硬约束；`early_return` 统一累积语义；
  `compiled_dims` 随入口变化；H100 四入口逐位相等 + profiler 抓主序模板实参；见上）。
  上一轮 **P35-1 Muon / 对称矩阵 batched 算子 `batched_syrk`/`batched_symm`**：
  DeepGEMM 里两个 cuBLASLt-only 的窄接口——2D↔3D 统一、放宽 batch 连续性（strided-batch）、
  强制 FAST_TF32；H100 真机 8 个真实 Muon 形状实测 + 契约检查 + kernel 名；见上）。
  上一轮 P34-1 共享专家分支：Mega MoE 里唯一「不走 dispatch 却走
  完整计算」的通路——共享 L1 与 dispatch 重叠、L1 沿 N/L2 沿 K 变宽、结果写 combine 的额外槽无权重加一次；
  主 kernel SM100 仅分析，host/调度器/UTCCP/数学 H100 真机实测；见上）。
  P33-1 测试断言辅助 `tests/utils.py`：逐位 oracle 层（direct==累积逐位、随机舍入 oracle 与 device
  哈希逐位 0 不符、PSUM padding 全零+负例、SF 毒化、E4M3 精确表示 E2M1、NVFP4 VEC16 分块；H100 27/27
  PASS）。P32-1 Mega MoE host 编排层与对称缓冲区布局（H100 逐字段对拍 `num_bytes` 与 12 view）。
  P17-1 cuBLASLt 回退、P18-1 选核启发式、P19-1 SM90 BF16 主 kernel、
  P20-1 SM90 分页 MQA logits 调度器、P21-1 SM100 GEMM epilogue store 路径与四算子、
  P22-1 SM90 FP8 dense GEMM 的 1D1D/1D2D 两个主 kernel、P23-1 DeepJIT 编译/缓存/加载/发射管线、
  P24-1 SM 局部域探针与内存域分配、P25-1 SM100 FP8/FP4 稠密 GEMM 主 kernel、
  P26-1 SM100 BF16 主 kernel 与 `set_tc_util`、P27-1 测试参考量化器 FP8/FP4 block-scale 语义、
  P28-1 PTX 原语底座（内存序/原子/搬运/warp 通信）、P29-1 legacy A100 Triton grouped GEMM 家族、
  P30-1 测试基座 `deep_gemm/testing/` 的 `calc_diff`/`bench_kineto`/`uneven_all_gather`、
  P31-1 操作数/布局生成器 `tests/generators.py`）。
  后续若继续（均为增量、非承诺）：
  ⑱ **设备端数学与工具层已补齐（P44-1）**：`common/math.cuh`（UE8M0 整数进位取整/`SF*SF_inv`
    精确、bf16/FP8 打包、`fma2`/`fast_rcp`、`warp_reduce` intra/inter、`cta_exclusive_sum`）+
    `utils.cuh`/`types.cuh`/`packing.cuh`/`compile.cuh`/`exception.cuh`/`cute_tie.cuh`；架构无关，
    H100 真机直接 include 真实头文件跑真实设备代码 ALL PASS。至此「上层 kernel → 公共设备函数」的
    最后一层地板讲清。**`common/probe.cuh`（per-warp `clock64` 追踪基础设施）已由 P45-1 补齐**（第 55 篇）。
  ⑰ **Mega MoE host 编排层已补齐（P32-1）**：`deep_gemm/mega/__init__.py` 的 `SymmBuffer` 生命周期 +
    `csrc/apis/mega_moe.hpp` 的 `get_symm_buffer_size_for_mega_moe`（ring 容量、12 view）+ `layout/
    mega_moe.cuh` 的 `MegaMoEBuffer` 链式字节分配 + scheduler `get_num_max_live_pool_blocks`；
    全纯 host、架构无关，H100 真机 7 组配置逐字段 ALL PASS。至此从「Python 调用」到「主 kernel 入口」
    的整条 host 链已讲清（此前一直从 kernel 内部讲起）。
  ⑯ **操作数生成器已补齐（P31-1）**：`tests/generators.py` 的枚举器 / `QuantConfig` / operand
    major 物理编码 / 三种 grouped 布局 / PSUM 递推 / K 分组 per-channel SF + 按组零填充；纯 Python，
    H100 真机 41 PASS，四个真实 SM90 kernel 端到端对拍。**`enumerate_batched_syrk_symm`（Muon 形状）
    已由 P35-1 补齐**（`batched_syrk`/`batched_symm`，H100 8 个真实形状实测）。可再补：
    `deep_gemm/legacy/` 的生成器交叉、`deep_gemm/testing/bench.py` 在真实 SM100 上的
    `set_tc_util` 影响（需目标卡）。
  ⑮ **legacy Triton MoE grouped GEMM 已补齐（P29-1）**：`deep_gemm/legacy/` 的四个 kernel
    （dgrad `m_grouped`/`a_fused_m`、wgrad `a_fused_k`/`b_fused_k`）与「显式索引 gather」语义、
    与现代 C++（`grouped_layout`+对齐）的对照表；Triton 架构无关。**测试基座已补齐（P30-1）**：
    `deep_gemm/testing/`（`calc_diff` 判据 = 归一化平方误差、`bench`/`bench_kineto` 计时）与
    `deep_gemm/utils/dist.py` 的 `uneven_all_gather`/`init_dist`；`deep_gemm/testing/utils.py` 的
    `test_filter`/`ignore_env`。**操作数生成器已补齐（P31-1）**：`tests/generators.py`。
  ⑭ **「稠密 GEMM 主 kernel」矩阵已补齐（P26-1）**：SM90 FP8/1D1D-1D2D（P22）、SM90 BF16（P19）、
   SM100 FP8/FP4（P25）、SM100 BF16（P26）四格齐全；并拆清 `set_tc_util` 这一从 Python 直通模板参数的
   tensor-core 节流开关（只有 SM100 BF16 实现）。可再补 SM100 BF16 的 `kSwapAB` dgrad 分支数值细节，
   或把「节流」在真实 SM100 上量化（需目标卡）。
  ⑬ **SM100 GEMM 后端已讲清（P25-1）**：一套模板通吃稠密/m-grouped/k-grouped/batched，
   MXF8F6F4/MXF4 两种 UMMA、TMEM 列预算与 overlap、UTCCP 搬 SF、五类 warp；与 P22-1（SM90）成对照。
   至此「SM90 两个主 kernel + SM100 唯一主 kernel」都已拆解。
  ⑪ **JIT 运行时已讲清（P23-1）**：五段式缓存键、两级缓存、原子发布、递归 include 哈希、微源码调用形态、
  加载/发射（`cuLaunchKernelEx` + cluster/PDL 属性）。**post-hook 后处理机制、Ascend 后端对照、
  Python API 与两个 WIP 特性均已由 P37-1 补齐**（`docs/47`）：post_hook 的运行时机/进键/override 语义、
  磁盘产物与发布顺序、`Runtime<Backend>` 的 CUDA↔Ascend 抽象、`get_jit()` 与「历史预热 / Python 编译 API」
  两个不可用 WIP；H100 真机 8 组 ALL PASS。
  ⓪ **SM90 kernel 覆盖已接近完整**：FP8（1D1D/1D2D，P22-1）、BF16（P19-1）、wgrad/dgrad grouped
  （P4-2/P4-3）、MQA logits（P19/P20）、einsum（P8/P11）、prenorm（P7-2）、SF 布局（P15/P16）
  均已系统拆解；可补的 SM90 缺口已很少（如 `sm90_tf32_hc_prenorm_gemm`、`sm90_bmk_bnk_mn` 的更细粒度）。
  ① **sparse attention 系列（P12/P13/P14/P20 已开）**：SM90 dense + SM90 分页调度器、SM100 sparse、
  SM100 dense 连续版/分页版调度器均已做；可再补 `sm100_mqa_logits.cuh` SIMT epilogue 的更多数值特性，或
  `impls/sm100_sparse_mqa_logits.cuh` 主 kernel 的 TMA 快路 / 尾裁剪细节精读；
  ② **einsum 家族 FP8 投影（P11 已开）**：`bhr,hdr->bhd` 已做；`bhd,hdr->bhr` / `bhd,bhr->hdr`
  是 arch10 专属（仅分析）。可再补 `fp8_einsum` 的 FP8 输出对（SM100 `Kernel1D1D`）精读；
   ③ **DeepEP 运行期实验**：待「阻塞」中的 NCCL≥2.32.3 环境就绪后，在 H100 上跑真实 dispatch/combine
    并测 expanded layout 与 compute overlap（PDL / `async_with_compute_stream`）。**DeepEP 的设备端公共层
    已由 P49-1 补齐**（`common/ptx.cuh` 等，H100 真机 41 PASS）；**与运行期无关的「确定性 dispatch 排序」
    已由 P52-1 补齐**（`EPHandle.deterministic_sort`，纯 host/Python，H100 抽取真实源码 7 PASS）；
    **多 rank 屏障与 symmetric-memory 句柄层已由 P53-1 补齐**（`comm/{barrier,handle}.cuh`，NVLink 腿
    逐行抽取 + 对称指针桩 H100 真机 19 PASS）；**「对称内存」的物理分配/映射/跨进程 shard 地基已由
     P56-1 补齐**（`comm/symmetric.hpp` + `allocator.py`，H100 真机 55+23 PASS）；**bucket 集合通信
     已由 P54-1 补齐**（`all_gather` copy-engine 真机跨 NVLink peer 逐字节一致）；**流水线并行
      send/recv 已由 P58-1 补齐**（`pp_send_recv`，数据面 TMA 流水 + 寻址面索引 H100 真机 40/40
      byte-exact + 168 有序对 0 不符）；**带索引的远程抓取 Engram 已由 P59-1 补齐**（`engram_fetch`，
      布局/发射映射/请求索引/SF 拷贝 H100 真机设备 5 + Python 32 PASS）；**EP 的两级域 hybrid
      dispatch/combine 已由 P61-1 补齐**（`hybrid_dispatch.cuh`/`hybrid_combine.cuh`：数据面 NCCL GIN
      仅分析，就绪哨兵/tail 增量/`do_psum`/round-robin 四个架构无关内核 H100 真机 7+6 PASS）。DeepEP
      侧余下的运行期空档（Engram 真实 RDMA 与端到端、hybrid 的端到端）都要 NCCL GIN 设备 API，待环境。
      （bucket 的 `all_reduce/reduce_scatter` 内核已由 P62-1 补齐：布局/映射/归约/bf16 哈希/host 算术
      在 H100 真机跑真实设备代码 30 PASS，数据面仅分析。）
   ④ **Mega MoE 端到端 / 其它变体**：`fp8xfp8`/`fp8xfp4`（已在 `MmaKind::MXFP8FP4` 下统一）的
    **FP4 路由权重路径已由 P95-1 补齐**（第 105 篇：`kPackedFP4` 表示、TMA `16U4` 解包、
    `MXF8F6F4` vs `MXF4`、权重重排与打包可交换、FP4 vs FP8 精度/显存，H100 真机 ALL_OK）；
    剩余：shared expert 分支的更深精读；端到端需 SM100，当前不可行。
  ⑤ **k-grouped SF 打包**：P16-1 已完成（host 契约 + device 逐行对拍，H100 真实设备代码）。
   ⑥ **cuBLASLt 回退/选核**：P17-1 已完成。
   ⑦ **选核启发式 `get_best_config`**：P18-1 已完成（SM90 定量代价模型 + SM100 词典序，
    H100 上 Python 复刻 20 组形状 15 字段逐项对拍 0 不符）。至此「用不用自家 kernel」与「用自家
    kernel 时怎么配」都已讲清。
   ⑧ **SM90 BF16 主 kernel**：P19-1 已完成（五种 `GemmType` 一套 kernel、合并 stage、两种输出 dtype、
    TMA reduce-add；H100 六类入口实测）。与 P10-1（SM100 BF16 mega moe）、P20 篇互补，BF16 路径补齐。
    ⑨ **SM100 GEMM epilogue store 路径与四算子**：P21-1 已完成（`EpilogueClass` 选择 + 无状态算子 +
    共享 store 路径；随机舍入四元组哈希/无偏性、`QuantizeToFP8` D/SFD 逐位对拍；H100 抽取真实设备代码）。
    至此「何时不用自家 kernel / 怎么配 / 尾部干什么」三问都已讲清。
    ⑩ **SM90 FP8 两个主 kernel（1D1D/1D2D）**：P22-1 已完成（同一数学、只差 SFB 粒度；分派规则、
    十处差异对照、两个 kernel 逐段精读；H100 真机抓 kernel 实参 + 交叉验证逐位 0 误差 + 性能 ~1.5×）。
    与 P19-1（BF16）、P4-2/P4-3（wgrad/dgrad）、P11（FP8 einsum）互补，**SM90 FP8 主 kernel 路径补齐**。
    ⑫ **SM 局部域探针与内存域分配**：P24-1 已完成（pointer-chase 延迟探针 + `try_probe` 两闸门 +
    MLOPart/CUDA VMM 本地化分配 + 按域 TMA 描述符；探针设备代码架构无关，H100 真机跑真实设备代码，
    无本地化正确拒绝并回退、host-pinned 远端域被检出）。这是第 03/09/11 篇 `num_locality_domains=2`
    调度分区的前置步骤。
    若无新指令，可视为专题完结。

  （历史各轮进展见下）P5-2：`docs/05-mega-gate-mhc.md` 新增「第二部分：Mega mHC」。
  **计算图**（以 `tests/test_mega_mhc.py:123-210`
  参考为权威）：残差多流混合 `new_residual[t,j]=post_mix·x + Σ_i comb[t,j,i]·residual[t,i]`（bf16 舍入 =
  所有消费者的共享数值边界 `epilogue:496-498`）→ mix 投影 `mixes = (new_residual.flatten @ fn^T)·rsqrt(hc_sqr/(hc·H)+eps)·scales + bases`
  → `sigmoid` pre/post + **Sinkhorn 双随机** comb（`epilogue:63-86`）→ `norm_input=Σ_j mix_j·new_residual_j`
  → RMSNorm（+FP8）。**normal vs shifted** 区别只在用哪套 pre 系数与是否持久化 pre。**三个非平凡点**：
  ① UMMA 是 **TF32 TS**（A=残差在 TMEM、B=`fn` 在 SMEM，`impls:235-275`），因系数投影需要高于 bf16 的
  尾数精度，而激活已在 `new_residual` 处强制 bf16；② Sinkhorn 用 warp 内 `shfl_xor` 逐行/逐列归约，
  双随机矩阵做「无能量缩放」的残差再分配；③ 跨 CTA `SplitBarrier`（Norm/Mix 两个）用
  「grid 序号+到达数」一次编码、免复位（`scheduler:13-49`）。6 warpgroup 静态角色（`impls:78-85`）+
  split-K scratch（`layout:73-120`）；normal 融合 worker（`epilogue:139-281`）、shifted 两 token/task
  （`:291-390`）；三种 SF 布局 + `get_shared_sf_row_idx` 页交错（`epilogue:117-122`）。**SM100 专属，仅分析**
  （`mega_mhc.hpp:223`）。H100 实测 `src/mega_mhc_semantics.cu`（+`harness/exp_mega_mhc_semantics.{sh,out.txt}`）：
  残差/RMSNorm 到 bf16 ulp、mix 系数 relerr ~2e-7、Sinkhorn 行和列和稠密到 2e-6、SF 页布局 4 万余行
  **0 不符**、SF 打包 **0/512 不符**、E4M3 量化 **0.5 ulp（round-to-nearest 最优）**。
  配套博客 `content/posts/mega-moe-13-mega-gate`（已并入 mHC）。
  P5-1：新建 `docs/05-mega-gate-mhc.md`——Mega Gate（`mega_gate.hpp` / `sm100_bf16_mega_gate.cuh`）
  源码精读 + H100 语义对拍。**融合链路**：BF16 打分 GEMM（UMMA，A=weight/B=x，累加器写 TMEM，
  `impls:213-226`）→ gate warp `tmem_load_32dp32b` 读出（`epilogue:39-78`）→ split-K 时经 global
  score scratch + `ScoreBarrier` 跨 CTA 汇聚（`scheduler:13-39`）→ 每 token 排名。**三个非平凡点**：
  ① TMEM 打分直读（省 `[T,E]` global 往返）；② 排名用 `score+bias`、权重用 `score` 再归一化
  （`impls:379-399`/`epilogue:187-216`，bias 只做负载均衡）；③ warp 内 **sort4 排序网络 + 3-bit 游标
  归并**的确定性 top-k（破平=最小专家号，`epilogue:88-152`）。任务分解
  `num_logical_ctas=expert_groups*cta*splitk` 且必须同驻留（`heuristics:21,134`）；`sqrt_softplus`
  阈值 20（`epilogue:19-23`）；重复专家 `(rank+t*23333)%d`（`epilogue:205`）；shared-as-routed、
  mask/fix_routing/force_random/image_bias 四开关。**SM100 专属，仅分析**（`mega_gate.hpp:158-159`）。
  H100 实测 `src/mega_gate_routing.cu`（+`harness/exp_mega_gate_routing.{sh,out.txt}`）：架构无关路由
  语义逐行移植——`sqrt_softplus` vs double relerr 1.4e-7、阈值分支连续；5 案例（含重平局、3-wave、
  4-wave padded）**~7 万条 top-k 全 0 不符**、权重 relerr ≤1.5e-7；重复专家分散 **max/min=1.0000**。
  配套博客 `content/posts/mega-moe-13-mega-gate`。
  P4-4：新建 `docs/14-moe-backward-kernels.md`——把 H100 可跑的 SM90 grouped GEMM **接进真实 MoE 的
  完整反向图**并与 PyTorch autograd 对拍。**四个大 GEMM**：`dW2=A^T dY`、`dW1=x^T dG` 走
  `k_grouped_*_nt_contiguous`（不等长在 K=token）；`dA=dY W2^T`、`dX=dG W1^T` 走
  `m_grouped_*_nt_contiguous`（不等长在 M=token）；其余（combine/SwiGLU/router 反向）保持 torch。
  **两层对拍**：`vs deq` 四项 relerr **~2.6e-4**（仅 bf16 输出舍入）证明 kernel+布局正确；
  `vs autograd` 含 per-128 FP8 量化，2.46e-2~5.08e-2、**cos 全部 >0.999**。**吞吐**（单 rank
  T=2048,H=7168,I=2048,E=8,topk=2）：wgrad ~103/106 TFLOPS（K=token 偏小）、dgrad ~449/543 TFLOPS。
  配套博客 `content/posts/mega-moe-12-backward-kernels`。**结论**：反向无 mega-kernel——通信规模小、
  可分解为独立 grouped GEMM、wgrad 留守专家本地 rank。
- **下一步：P6-1（汇总 `docs/06-summary.md`）**——数据流图、关键机制、与朴素实现的开销对比、
  反向要点一页纸，把 P0~P5 的结论串成一篇总览。
  （P4 进展见下）
  P4-3：新建 `docs/13-moe-backward-dgrad.md`——dgrad 的 M 轴分组 API 精读。**全景**：dgrad
  （`dA=dY·W2^T`、`dX=dG·W1^T`）不等长在 **M（token）轴**、N/K 固定，故复用前向的 `m_grouped` 族；
  A（激活）必须 K-major、D 固定 BF16（`gemm.hpp:197/209`）。**两入口**：
  `m_grouped_fp8_fp4_gemm_nt_contiguous`（`:181-250`，contiguous=长度 m 逐行组号 / PSUM=长度 G
  前缀和结束，判别在 `:213-220`）与 `..._nt_masked`（`:269-318`，`[G,max_m,·]` 稠密 + device 端
  `masked_m`）；**PSUM 只作用于 SFA**（`layout.hpp:69`、`gemm.hpp:230`），因为 gap 只出现在激活侧。
  **调度器**三 `GemmType`（`scheduler/gemm.cuh:92-105`）：masked/PSUM 的 `num_m_blocks` 随组变
  （`:188-225`），contiguous 恒定；`get_global_idx`（`:151-156`）与 `is_computation_valid`
  （`:296-309`）统一取址/跳 gap。**为什么段要对齐 `BLOCK_M`**：contiguous 用 M block **首行**组号
  `grouped_layout[m_block_idx*BLOCK_M]` 代表整块（`:152`），不对齐就跨组；SM90 multicast peer
  一致性也依赖它（`:287-289`）。**SM90 kernel**：masked 下 A/D/SFA 是 3D（组号=第 3 维）、contiguous
  下 2D（`sm90_fp8_gemm_1d2d.hpp:178-194/245-261`），`kWithGroupOffsetA=masked`（kernel `:189-194`）；
  SFB 逐组基址（`:240`）。**训练分工**：前向/dgrad→contiguous、解码 CUDA graph→masked、
  DeepEP 动态输出→PSUM 桥接；shared expert 走稠密路径；dgrad 需按 token 跨专家 scatter-add，
  wgrad 留专家本地无需 all-reduce。H100 实测 `harness/exp_mgrouped_dgrad.py`（+`.out.txt`）：
  contiguous/masked/PSUM relerr 3.67–3.70e-2、A4 布局长度不符被拒（`gemm.hpp:218`）、
  `m=4352,n=2048,k=7168`（G=8）**366 TFLOPS**；三种布局**两两逐位相等（0 误差）**。配套博客
  `content/posts/mega-moe-11-dgrad`。
  P4-2：新建 `docs/12-moe-backward-wgrad.md`——wgrad 的 K 轴分组 API 精读。**两条路径一个数学**：
  `k_grouped_fp8_gemm_tn_contiguous`（SM100，A/B 为 `[sum_k,M]`/`[sum_k,N]` **MN-major**，
  靠组坐标偏移取数，`sm100_fp8_fp4_gemm_1d1d.hpp:325-403`，**仅分析**）与
  `k_grouped_fp8_gemm_nt_contiguous`（SM90，每组 `[M,k_g]` **K-major** 拼接，`gemm.hpp:373-427`）。
  **SM90 为什么按组原地改 tensormap**：base tensormap 只能按首个非空组的 K 造
  （`sm90_fp8_gemm_1d1d.hpp:159-195`），kernel 里切组时 patch 全局地址 `+k_start*M`
  与 inner-dim stride，并用 `tma_desc_commit/wait_group` + release/acquire 防竞态
  （`sm90_fp8_gemm_1d1d.cuh:193-215`，对应 PR #343）。**参数契约**：`ks_cpu` 每组 K 必须
  `k_alignment` 对齐（`check_k_grouped_args`，`gemm.hpp:55-76`）；`grouped_layout` 在
  contiguous=物理 K、PSUM=逻辑 K 结束（PSUM 仅 SM100，`scheduler/gemm.cuh:71-82`）；
  SM90 `nt` 强制 accumulate（`gemm.hpp:403`）。**SF**：SM100 逐组把 FP32 打包成 MN-major
  UE8M0 int32（每 4 个 gran_k 打 1 个，`smxx_layout.hpp:218-283`）。调度器按组切 K 块 +
  独立计数 `current_sf_k_cumsum`（`scheduler/gemm.cuh:226-246`）。H100 实测
  `harness/exp_kgrouped_api.py`（+`.out.txt`）：nt 4 组 relerr 1.98e-2、空组 D 全零、
  未对齐 ks 被拒（`gemm.hpp:69`）、tn 在 sm90 被拒（`gemm.hpp:369`）、`m=4096,n=7168`（8 组）
  4528us **425 TFLOPS**；纯 torch 布局等价：tn 0 误差、nt 3.3e-8。配套博客
  `content/posts/mega-moe-10-wgrad`。
  P4-1：新建 `docs/04-moe-backward.md`——MoE 反向数学。**全景**：`dz`→combine 反向
  （`dY=w⊙dz`、`q=<Y,dz>`）→linear2（`dW2=A^T dY` wgrad / `dA=dY W2^T` dgrad）→SwiGLU
  （`dG_G=dA⊙G_U⊙silu'`、`dG_U=dA⊙silu`）→linear1（`dW1=x^T dG` / `dX=dG W1^T`）→router。
  **router/topk 反向**：归一化 topk 权重把 softmax 概率除以 `P=Σ_{i∈S}p_i`，得
  `dp_j=(q_j P-A)/P²`（`A=Σ q_i p_i`）再走 softmax Jacobian；**坑**：`P` 必须是原始 softmax 概率
  在选中集上的和（不是归一化权重之和，后者恒为 1），第一版因此让 `dx`/`dWg` 偏到 1e-2。**映射**：
  wgrad 的不等长轴是收缩维 K（token）→ `k_grouped`；dgrad 的不等长轴是 M（token）→ `m_grouped`，
  给出了 (M,N,K) 角色表与 `gemm.hpp`/`generators.py` 真实行号。H100 实验
  `harness/exp_moe_backward_math.py`（+`.out.txt`，float64）：四项梯度 vs autograd relerr ≤2.9e-16、
  中心差分 ≤5.7e-8，按 `k_alignment=128` 逐组 `A_e^T@B_e` 复现 `dW1/dW2` relerr ≤1.3e-16。
  配套博客 `content/posts/mega-moe-09-backward`。**反向无 mega-kernel**：融合价值主要在前向重叠；
  反向由独立 grouped GEMM 组成，wgrad 天然留守专家本地 rank。
  （P0/P1/P2/P3 进展见下）
  P3-4：新建 `docs/11-mega-moe-dispatch-combine.md`——通信两段逐段精读 + H100 访存实测。
  **dispatch**：分布式前缀和一次原子加（高位计数、低位负载，`kernel:374-381`）→ `grid_sync` 后
  SM0 push `peer_grid_idx`（`:399-404`）→ `nvlink_barrier`（`:422-428`）→ 无锁 round-robin
  min-peeling 认领 token（`:480-528`）→ 环空等待（`:544-550`）→ 分 chunk `tma_load_1d` 远端 pull
  token（`:552-576`）+ 远端 topk weight（`:579-581`）+ 远端 SF 逐块搬并 UTCCP 转置写位（`:584-599`）
  → 写元数据/`red_add_rel` 通知（`:602-618`）→ 清 workspace 并与 combine 重叠（`:621-689`）。
  **combine**：L2 epilogue 读回 `cached_src_metadata`（`:1281-1290`）后经 `sym_buffer.map` 直写
  远端 combine buffer（`:1346-1351`）；readiness 用「每 launch 唯一 grid index」当水位线
  （peer push `:403` ↔ `st_rel_sys`:1419 ↔ `ld_acq_sys` 自旋 `:1436-1441`）；归约双缓冲 TMA 拉
  top-k、fp32 累加（`:1477-1490`）、BF16 写 `y`（`:1512-1518`）。**原语表**：`SymBuffer::map`
  （`layout/sym_buffer.cuh:33-40`）、`grid_sync`/`nvlink_barrier`（`comm/barrier.cuh:33-92`，加减式
  相位信号免复位）。**locality domain**：`kNumDeviceLocalityDomains=2`（`types.cuh:54`），
  调度器按域分任务计数 + 少量任务不偷域（`scheduler:344-370`、`kNumMinStealWaves=2`），域标签由
  片上 pointer-chase 延迟探针标定（`csrc/apis/locality_domain.hpp:74-142`）。H100 实测
  `src/nvlink_access.cu` + `harness/exp_nvlink_access.{sh,out.txt}`：远端裸延迟 ~991ns ≈ 本地
  ~350ns 的 **2.83×**，远端读/写带宽 ~368 GB/s ≈ 本地 ~2.85 TB/s 的 **1/8**，同/跨 socket 相同。
  P3-3：新建 `docs/10-mega-moe-mma-epilogue.md`——计算本体与 epilogue 逐段精读。**MMA**：只有
  leader CTA 的一个 warp 发 UMMA（`kernel:819-938`）；`A/B 互换 + 2-CTA` 使 `UMMA_M=256`、
  `UMMA_N=BLOCK_M`（token 维），按 `valid_m` 动态缩 N（`update_instr_desc_with_umma_n`，
  `mma/sm100.cuh:150-153`）；routed(FP8/FP4) 与 shared(FP8) 两个 block-scaled descriptor
  （`:825-850`）；`UTCCP_4x32dp128bit_2cta` 把 SFA/SFB 从 smem 搬进 TMEM（`:894-907`）；
  `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`（`ptx/tcgen05.cuh:67-86`）内层按 `gran_k=32` 拆 4 条，
  首条非累加清零、其余累加（`:920`）；FP4 打包因子 2（`packing.cuh:8-12`）；commit multicast
  同时通知 `empty_barriers` 与（末 K block 时）`tmem_full_barriers`。**Epilogue**：TMEM 双缓冲，
  读完最后有效块即 `release_tmem`，`release_task_info` 提前（`:990`）；L1 分支 in-place SwiGLU
  （bf16 round，gran-8 交错权重让 gate/up 成对，`:1108-1135`）+ amax warp-pair 归约 + UE8M0
  量化（`get_ue8m0_sf_exp/inv`，`math.cuh:114-147`）+ STSM + TMA 转置上屏（就是 L2 激活）+
  `l2_full_mask` bitmask 依赖；L2 分支 BF16 经 `sym_buffer.map` NVLink 直写远端 combine buffer；
  combine 段 TMA 拉 top-k 贡献、fp32 累加、bf16 写 `y`。H100 数值实验 `harness/exp_ue8m0_sf.py`
  （+`.out.txt`）：位运算 `get_ue8m0_sf_exp` 在 25009 样本（含 0/亚正规/幂次边界）上 0 不符、
  `scale×inv` 精确为 1、幂次 scale/理想 ≤1.9977 且 0 溢出、bf16-round SwiGLU 相对误差 4.5e-3。
  配套博客 `content/posts/mega-moe-07-mma-epilogue`。
  P3-2：新建 `docs/09-mega-moe-mbarrier.md`——**两层解耦**（TMA↔MMA 的 full/empty，MMA↔epilogue 的
  tmem_full/empty）+ 调度器 task 槽（task_info full/empty）的完整 mbarrier 清单与 init 计数
  （`full.init(2*2)` / `empty.init(1)` / `tmem_empty.init(2*kNumEpilogueThreads)`，`kernel:209-285`）、
  `advance_pipeline` 相位翻转（`:307-314`）与「生产者 phase=1（buffer 初空）」约定、
  A/B loader 的坐标与 2-CTA multicast + FP4 half-bytes（`:690-818`）、MMA 的 UTCCP/UMMA/commit
  （`:819-938`）、epilogue 的 `tmem_full.wait`→`release_task_info`（`:986-990`）与
  「仅 epilogue release」的执行顺序论证、跨 SM `grid_sync` / 跨 rank `nvlink_barrier`
  （`comm/barrier.cuh:33-92`）、L1→L2 位图依赖（`scheduler:140-166`）。H100 真机实验
  `src/mbarrier_pipeline.cu`（`harness/exp_mbarrier_pipeline.sh` + `.out.txt`）：**实测坐实**相位约定
  （全新 mbarrier `try_wait(parity=1)` 立即成功、`(0)` 阻塞），并量到多级流水重叠——96 MiB
  `cp.async.bulk` 在 NSTAGES 1→4 下串行 5.58→3.44 ms（带校验 1.62×），去校验 2.29→0.79 ms
  （2.88×，41→118 GiB/s，延迟受限）。配套博客文章 `content/posts/mega-moe-06-mbarrier`。
  P3-1：新建 `docs/03-mega-moe-kernel.md`——一次 launch 的物理坐标（`grid=num_sms`、
  `cluster=2`、512 线程/16 warp，`impls/sm100_fp8_fp4_mega_moe.hpp:149-198`）、BLOCK_N=128/BLOCK_K=128(fp8,bf16=64)、
  BLOCK_M 启发式（`heuristics/mega_moe.hpp:88-127`，成本模型「M block 重载权重 ~128 行」+ 路由 `+sqrt` 余量）、
  store block、SF 的 UTCCP 128 对齐、`get_pipeline_config_for_mega_moe` smem 预算→`num_stages`
  （`:129-195`，`smem_capacity=232448`）、五类 warp 角色（dispatch/TMA-A/TMA-B/MMA/调度/epilogue+combine，
  `kernel:341-948`）、寄存器两档再平衡（每档恰好 64512）。H100 真机对拍：`harness/exp_block_config.py`
  复刻启发式并与 `deep_gemm.get_block_m_for_mega_moe` 逐点比较，9 组形状 block_m **全部一致**；
  配套博客文章 `content/posts/mega-moe-05-kernel`。
  （P0/P1/P2 进展见下）
  P1-2：重写 `docs/01` §2——EP all-to-all 语义 + 「K 倍放大」通信量模型（dispatch $T{\cdot}d{\cdot}H$、
  combine $T{\cdot}k{\cdot}H{\cdot}2$）+ 串行/重叠量化对比；新增 §2.3 **H100 实测**：
  `harness/exp_ep_comm.py` / `.out.txt` 在 8×H100 上测 NCCL `all_to_all_single`（H=7168、EP=8），
  渐近有效带宽 ~393 GB/s（达 `nvidia-smi nvlink` 硬件上限 ~478 GB/s 的 ~82%）、固定时延 ~50–90 us，
  论证小 batch 延迟敏感与重叠必要性。
  P1-1：细化 `docs/01` §1——MoE 计算图 + 三种 grouped 布局（contiguous / masked / PSUM）的真实 API 与行号，
  解释对齐（SM90=128 / SM100=256）与 gap 行的「为什么」；新增 §1.5 H100 实测（`harness/exp_grouped_layout.py`，
  三布局 relerr 一致 ≈3.7e-2）。
  （P0 进展见下）
  P0-1：读通 DeepGEMM Mega MoE 五个源文件，产出 `docs/02-mega-moe-architecture.md`（真实行号核对），P2-2/P2-3 一并覆盖。
  P0-2：`docs/01` §5 记录仓库/版本时间线/关键结论/DeepEP 侧真实链接；`docs/02` §8 补 PR #316 EP8 基准与解读。
  P0-3：精读 DeepEP v2.5 对称显存/dispatch/combine，产出 `docs/07-deepep-dispatch-combine.md`。
  P0-4：在 `kernel_lab`（H100）实测 DeepGEMM 2.8.1 可用 kernel 矩阵——dense/grouped FP8、BF16、
  **SM90 版 MoE wgrad（`k_grouped_*_nt_contiguous`）** 均 OK；`..._tn_contiguous`（wgrad TN）与
  `fp8_fp4_mega_moe` 在 sm90 被 `gemm.hpp:369` 拒绝（SM100 专属）。DeepEP 2.5 可构建/import，
  但 `EPBuffer` 因 NCCL 版本（PyTorch 链 2.29.7 vs 需求 ≥2.31/2.32.3）运行期段错误。
  产出 `docs/08-h100-environment.md` + `harness/probe_h100_env.py` / `harness/probe_deepep.py`（含 *.out.txt）。
  P2-1：补 `docs/02` §3.4——`transform_weights_for_mega_moe` 的三件事：L1 gate/up 按 8 粒度交错
  （`mega/__init__.py:112`，配合 L1 epilogue 的 `SM100_TMEM_LOAD_16dp256b1x` 做 in-place SwiGLU，
  kernel `:1066`-`:1134`）、SF 的 UTCCP 4x32 转置（`mega/__init__.py:129` ↔ kernel
  `transform_sf_token_idx` `:145`-`:149`）、SM100 packed UE8M0 + MN-major SF 索引
  （`smxx_layout.hpp:133`、kernel `:774`/`:788`）；L2 权重不交错只转 SF。实验
  `harness/exp_weight_transform.py`（+`.out.txt`，H100）：行号映射/SwiGLU 等价/UTCCP 映射全部 0 误差，
  与 deep_gemm 内部实现逐元素一致。**P2 全部完成。**

## 阻塞

- **DeepEP 运行期实验受阻（环境，非内核）**：本机 PyTorch 链接系统 **NCCL 2.29.7**，而 DeepEP 2.5 需要
  **NCCL ≥ 2.31 的设备通信 API**（`ginSupport/ginMinStride/ginType`，`csrc/kernels/comm/context.cpp:71-142`）。
  装 `nvidia-nccl-cu13 2.32.3` + `LD_PRELOAD` 可过 import 检查，但 `EPBuffer` 构造仍在
  `ep.py:304` 段错误；用系统 2.29.7 头重建则编译失败。**需一个链接 NCCL ≥ 2.32.3 的 PyTorch 环境**。
  因此在修复前，DeepEP 的 H100 dispatch/combine 实测暂不做（源码精读见 `docs/07`）。
