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

## 当前进度 / 下一步

- 当前进度：**P0~P14 全部完成；P15~P21 各增量完成（P15-1、P16-1、P17-1、P18-1、P19-1、P20-1、P21-1）。**
  本轮 P21-1：新建 `docs/31-sm100-gemm-epilogue-operators.md`
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
- **下一步**：P0~P21 已完成（P17-1 cuBLASLt 回退、P18-1 选核启发式、P19-1 SM90 BF16 主 kernel、
  P20-1 SM90 分页 MQA logits 调度器、P21-1 SM100 GEMM epilogue store 路径与四算子）。
  后续若继续（均为增量、非承诺）：
  ① **sparse attention 系列（P12/P13/P14/P20 已开）**：SM90 dense + SM90 分页调度器、SM100 sparse、
  SM100 dense 连续版/分页版调度器均已做；可再补 `sm100_mqa_logits.cuh` SIMT epilogue 的更多数值特性，或
  `impls/sm100_sparse_mqa_logits.cuh` 主 kernel 的 TMA 快路 / 尾裁剪细节精读；
  ② **einsum 家族 FP8 投影（P11 已开）**：`bhr,hdr->bhd` 已做；`bhd,hdr->bhr` / `bhd,bhr->hdr`
  是 arch10 专属（仅分析）。可再补 `fp8_einsum` 的 FP8 输出对（SM100 `Kernel1D1D`）精读；
  ③ **DeepEP 运行期实验**：待「阻塞」中的 NCCL≥2.32.3 环境就绪后，在 H100 上跑真实 dispatch/combine
  并测 expanded layout 与 compute overlap（PDL / `async_with_compute_stream`）；
  ④ **Mega MoE 端到端 / 其它变体**：`fp8xfp8`/`fp8xfp4`（已在 `MmaKind::MXFP8FP4` 下统一）、
    shared expert 分支的更深精读；端到端需 SM100，当前不可行。
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
