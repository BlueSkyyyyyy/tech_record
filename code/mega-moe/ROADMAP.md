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

- 当前进度：**P0~P9 全部完成；P10-1 完成。**
  本轮 P10-1：新建 `docs/20-bf16-mega-moe.md`（+`harness/exp_bf16_mega_moe.{py,sh,out.txt}`、
  `src/bf16_mega_moe_epilogue.cu`）——补上 Mega MoE 的**第二个主 kernel** `bf16_mega_moe`
  （`sm100_bf16_mega_moe.cuh`）。同一套骨架（dispatch→L1→SwiGLU→L2→combine、ring、重叠）
  换 `MmaKind::BF16`，**去掉整条 block-scale（SF/UTCCP/UE8M0）通路**：`a/b/d=bfloat16`、
  `block_k=64`（FP8=128）、唯一 UMMA `SM100_MMA_F16BF16_2x1SM_SS`、L1 epilogue 直接产出
  bf16（gate/up 先 round bf16 再 silu，`kFastMath`=__expf+rcp.approx）、L2 bf16 直写远端
  combine；combine 段与 FP8 版逐行复用。host 侧 `get_symm_buffer_size(..., "bf16xbf16")`
  `with_sf=false`（切片 `x_sf=None`）。**SM100 专属，仅分析**（`mega_moe.hpp:367`；kernel:68）。
  H100 实测：block_m 13 组 BF16/FP8 全一致、buffer 1.265×、权重重排 0.0、L1 SwiGLU
  「bf16 in→fp32 silu→bf16 out」rel 3.89e-3（输出 1 ulp）、L2 cast vs RNE 不符 0。
  配套博客 `content/posts/mega-moe-20-bf16-mega-moe`（索引页新增第 20 篇）。Hugo 构建 0 ERROR。
  上一轮 P9-1：新建 `docs/19-attention-mqa-logits.md`（+`harness/exp_mqa_logits.{py,sh,out.txt}`）
  ——拆出 DeepSeek 稀疏注意力 **indexer 的 MQA logits**（`fp8_fp4_mqa_logits` 连续 /
  `fp8_fp4_paged_mqa_logits` 分页；**SM90 有实现，H100 可跑**）。语义
  `out_ij = Σ_h relu(<q[i,h,:], kv_j>)·w[i,h]`（README `:100-108`）、KV 每 token 一个 fp32 scale、
  输出按有效区间压缩。host `attention.hpp` 三类检查 + 连续/分页两入口；SM90 连续 kernel
  **Q 在 B 侧 / KV 在 A 侧**的 WGMMA（M 钉死 128 = `block_q*heads`）→ ReLU/权重/KV scale 的
  两级 warp 归约（intra + `shfl_xor`）→ 压缩偏移写回；SM90 分页 = 每 math WG 一个 64-KV 子块 +
  3D TMA 物理块寻址 + `shfl` 广播 block table。MXFP4/MXFP8、varlen、`schedule_meta`、
  sparse top-k 块均为 SM100 专属（**仅分析**）。H100 实测：连续 `vs_sim` ~2e-3（隔离量化）、
  `vs_ref` 3–5e-2、掩码逐位一致、重复逐位相等，H=64,D=128 **497 TFLOPS**；分页 230–370 TFLOPS；
  6 项契约非法输入全部被拒。配套博客 `content/posts/mega-moe-19-attention-mqa-logits`
  （索引页新增第 19 篇）。Hugo 构建 0 ERROR。
- **下一步**：P0~P10 已完成。后续若继续（均为增量、非承诺）：
  ① **sparse attention 系列**：`fp8_fp4_mqa_logits` 的 SM100 MXFP4/MXFP8、`get_mqa_logits_metadata`
  负载均衡调度、`fp8_fp4_sparse_mqa_logits`（top-k 块）、`fp8_fp4_paged_sparse_mqa_logits`
  （arch10 专属，仅分析；sparse 版还有 SM90 可试的部分吗需查）；
  ② **einsum 家族 FP8 投影**：`fp8_einsum` 的 `bhd,hdr->bhr` / `bhd,bhr->hdr`（arch10 专属，
  仅分析；`bhr,hdr->bhd` 的 fp8 版 SM90 可试跑）；
  ③ **DeepEP 运行期实验**：待「阻塞」中的 NCCL≥2.32.3 环境就绪后，在 H100 上跑真实 dispatch/combine
  并测 expanded layout 与 compute overlap（PDL / `async_with_compute_stream`）；
  ④ **Mega MoE 端到端 / 其它变体**：`fp8xfp8`/`fp8xfp4`（已在 `MmaKind::MXFP8FP4` 下统一）、
   shared expert 分支的更深精读，或 `transform_sf_into_required_layout` 通用布局；端到端需 SM100，当前不可行。
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
