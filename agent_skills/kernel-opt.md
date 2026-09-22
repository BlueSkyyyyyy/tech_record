# kernel-opt — CUDA 算子调优系列的自驱工作流程

## 触发场景

用户说「继续整理/推进算子优化系列」「CUDA 算子调优」（本仓库的长期自驱任务）。
本技能把「写代码 → 本地实测 → ncu 剖析 → 写文章 → 发布」串成可重复的循环。

> **无人值守模式**：本技能常被 `code/kernel-opt/scripts/autopilot.sh` 以无头会话反复调用，
> 每轮只做一篇文章增量，做完就停，更新路线图后本轮结束。若你是被 autopilot 唤醒的：
> 全程不要请求人工确认（别用 question 类工具），**永远不要自己创建 `AUTOPILOT_STOP`**
> （只有用户运行 `autopilot.sh stop` 才停）。**当 ROADMAP 的「下一步」做完/为空时，
> 自己从第五/六/七部分或 backlog 里补充具体、可测的新任务再继续**，持续把算子性能往极致推。

## 前置条件

- 已读根目录 `agent_guide.md`（目录约定、已知坑）。
- 已读系列控制面板 `code/kernel-opt/ROADMAP.md`（**每次先读它，从「下一步」做起，做完更新它**）。
- 已读技巧台账 `code/kernel-opt/TECHNIQUES.md`（**每篇必须往里加技巧条目 + 更新性能榜**）。
- 环境：宿主机有 `docker` 权限；镜像 `dsv4-inf:latest` 已存在（含 CUDA 13.2 / nvcc / ncu / nsys / PyTorch）。

## 环境：kernel_lab 容器（关键）

宿主机 `nvidia-smi` 显示 8×H100，但**宿主 `RmProfilingAdminOnly=1`**，普通容器里 ncu 会报
`ERR_NVGPUCTRPERM`。必须用带 `--cap-add SYS_ADMIN` 的容器——已封装为 `code/kernel-opt/scripts/lab.sh`：

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up          # 创建/启动 kernel_lab（幂等，可重复调用）
scripts/lab.sh enter       # 交互 shell
scripts/lab.sh status
```

代码在宿主机与容器内路径一致（容器挂了 `/home/xieminglin` 与 `/ssd`），所以直接在宿主侧写代码即可。

## 步骤（每篇的循环）

1. **选一篇**：读 ROADMAP「下一步」。
2. **写代码**：`code/kernel-opt/NN-<slug>/*.cu`，`#include "../common/cuda_utils.cuh"`。
   - 每个 .cu 自带参考实现对拍 + 计时；编译运行：`scripts/run.sh NN-<slug>/foo.cu`。
   - 剖析：`scripts/ncu.sh NN-<slug>/foo.cu --set full --kernel-name regex:foo`
     （ncu 参数写在前，程序参数用 `--` 分隔）。
   - 实测原始输出另存 `NN-<slug>/foo.out.txt`，写文章时引用真实数字。
3. **写文章**：按 `write-post.md`，slug `cuda-kernel-opt-NN-<slug>`，加 `weight`（=NN）与系列 tag。
   结构：背景 → 现象/数据 → 根因 → 优化 → 实测对比表 → 小结 → 下一篇预告。讲人话，配图优先。
4. **发布**：按 `publish.md`（构建 → commit → push → 验证线上 200）。
5. **回写控制面板**：更新 `ROADMAP.md` 状态与「下一步」、`README.md`/`code/README.md` 索引，
   把新踩的坑追加到本文件或 `agent_guide.md`。
6. **回写台账**：在 `TECHNIQUES.md` 新增技巧条目（原理/场景/代码/实测收益/坑）并刷新性能榜；
   模型类文章更新「模型场景台账」。
7. **模型场景优先**：第五部分（MLA/DSA/MoE/FP8/MuonClip…）是当前重点，要用 `/ssd/models/*/config.json`
   的真实参数构造 shape，并对标 `~/github/` 里的 FlashMLA / DeepGEMM / cutlass 等实现给出差距。
8. **提交**：每篇 commit & push 一次，信息格式 `post:` / `bench:` / `skill:`。**不要**创建 `AUTOPILOT_STOP`。

## 验收标准

- [ ] 配套代码在 `kernel_lab` 里实跑通过，原始输出已留存
- [ ] 文章数字全部来自实测（不是猜的），关键结论有 ncu 支撑
- [ ] ROADMAP 状态、「下一步」、索引已更新
- [ ] `hugo --gc --minify` 无 ERROR，线上页面 200

## 已知坑

- **ncu 权限**：必须 `kernel_lab`（含 SYS_ADMIN）。用 `kimi26_train` 会 `ERR_NVGPUCTRPERM`。
- **容器 root 建目录**：首次以容器 workdir 创建目录会属 root，宿主写不进去；
  用 `docker exec kernel_lab chown -R 1001:1001 <dir>` 修。
- **`/home/xieminglin` 是软链**到 `/ssd/home/xieminglin`；`readlink -f` 后路径一致，脚本已处理。
- **H100 峰值常量**（写文章换算用）：HBM ~3.35 TB/s；FP32 CUDA core ~66.9 TFLOPS；
  BF16/FP16 Tensor Core dense ~989 TFLOPS；FP8 ~1978 TFLOPS。
- CUDA event 计时**含 launch 开销**，测单 kernel 纯 device 时间用 ncu / CUPTI（见第 03 篇）。
- 小 shape 反复跑会命中 L2（H100 ~50MB），带宽会虚高，判断 HBM 效率要用大 shape。
- **`ncu` 的 "Memory Throughput" 是 L1/L2/DRAM 里最高那一级的利用率**，不等于 HBM；判断带宽瓶颈必须看 `DRAM Throughput`。
- **强制提高 occupancy 可能是负优化**：`__launch_bounds__(threads, N)` 压寄存器会 spill 到 local memory（实测 `occ<8>` spill 1.31 TB，慢 34×）。先看 `not_selected`/`no_eligible` 是否真的空发射口。
- `scripts/ncu.sh` 支持用 `--` 分隔 ncu 参数与程序参数（如 `... --set full -- 20000 occ2`）；收集 stall 用 `--metrics smsp__average_warps_issue_stalled_*_per_issue_active.ratio`。
- 源码/SASS 对照：`ncu --page source --print-source cuda,sass --metrics smsp__inst_executed.sum ...`。
- **`ldmatrix` 的 bank conflict 极其致命**：smem 行主序数组若不加 padding，`ldmatrix` 一次读 8 行会撞 bank（B 行距 128 bf16 时 8 行全在 bank 0，实测 3565 万次冲突）。给行距 **+8 个元素**（`LDA=BK+8`、`LDB=BN+8`）即可归零，同一 kernel 从 76→221 TFLOPS。写任何用 ldmatrix 的 kernel 先查 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`。
- **WMMA `load_matrix_sync` 在 sm_90 上是负优化**：抽象导致共享内存流量失控（L1/TEX 77%、Compute 17%），同尺寸只有 62 TFLOPS；手写 `mma.m16n8k16` + `ldmatrix` 能到 221。入门可讲 WMMA，但性能路径必须自己控布局。
- **`ldmatrix.x4.trans` 取 B 片段**：B 在 smem 里存成行主序 `[K][N]`，mma 需要 `.col` 布局，用转置 ldmatrix 让硬件顺带转置；地址 lane 公式：`row=(lane&7)+((lane>>3)&1)*8, col=(lane>>4)*8`。
- **静态 smem 上限 48KB**：`__shared__` 数组超过 0xc000 会 ptxas 报错；BK=64 的 tc kernel 会超。要更大需 `cudaFuncSetAttribute` + 动态 smem。
- **模板化 kernel 的「通用循环 loader」会悄悄多花寄存器**：把第 09 篇的单发 float4 载入改成
  `for (q = t; q < N4; q += T)` 的通用循环后，同一 128×128×8 GEMM 从 118/128 寄存器涨到 168，
  occupancy 25%→12.5%，算力掉 ~35%。当某配置下每线程恰好搬 1 个 float4 时用 `if constexpr`
  走单发写法可恢复。**写通用模板后务必用 `NVCC_FLAGS="-Xptxas -v"` 核对寄存器数**，别只看功能正确。
- **epilogue 融合的寄存器开销常常为 0**：第 14 篇把 `store_acc` 模板化加 bias+GELU/ReLU 后，
  三个实例（MODE=0/2/3）都是 128 寄存器、0 spill，occupancy 与瓶颈指标逐项不变。别怕融合
  「拖慢」GEMM；但每次都要用 `-Xptxas -v` 复核，激活换成更复杂的函数未必还免费。
- **对照「两 kernel vs 融合」时，独立 epilogue kernel 必须自己先向量化**（float4 + grid-stride，
  第 14 篇跑 ~4 TB/s），否则标量版只有几百 GB/s，会得到虚高的融合收益（稻草人对比）。
- **device 辅助函数要在 host 参考实现里复用**（如 GELU）时，记得标 `__host__ __device__`，
  否则 nvcc 报 `calling a __device__ function from a __host__ function is not allowed`。
- **CUDA event 口径与 ncu 口径会有差**：第 14 篇独立 epilogue kernel event 测 8.4 µs、ncu 报
  11.26 µs（含 replay）。写文章时同一结论尽量用同一口径，或两者都标注。
- **别被「读放大」骗了**：第 15 篇 MLA，`naive` 版读放大 131072×，但 `c_kv` 只有 1~5MB、常驻
  50MB L2，ncu 实测 `DRAM Throughput` 仅 0.71%——真正瓶颈是 `Compute (SM)` 83.5%。判断优化方向
  **先看 ncu 的 DRAM%**，DRAM 低就别做访存复用。
- **attention/MLA 的 prefill 是算力受限**：`AI_ideal = 2H(DC+DR+DV)/(DC+DR)`，H=128/576/512 时≈242，
  接近 bf16 TC ridge（295），但标量 FFMA ridge 只有 ~20。标量实现天花板 <7%，必须上 Tensor Core。
- **按 head 复用 KV 会把寄存器吃爆**：第 15 篇 `head_reuse<8>` 达 255 寄存器 + 328B 栈溢出，比
  `HG=2` 还慢。`-Xptxas -v` 核对；换来的复用若不减少 DRAM（见上条），纯属负优化。
- **三 kernel 物化 attention 中间量的隐性税**：`S`(fp32)+`P`(bf16) 各写读一遍，流量 `∝H·S²`。第 15
  篇 S=4096 时 25.8GB≈7.7ms，占 TC 总时长 21%。能融合就别物化。
- **flash-attention 类融合的免费午餐**：`mma.m16n8k16` 的 fp32 累加器 `c0,c1`（行 `lane/4`）和
  PV 的 A 片段 `a0,a1` 位置**完全一致**；softmax 后的 P 只要 `pack2(C_tile0.c0,c0.c1)` 等 4 次
  `__floats2bfloat162_rn` 就变成 PV 的 k16 A 片段（相邻两个 n8 tile 配对），**零 shuffle**。见 16 篇。
- **`mma`/`wgmma` 累加器是累加语义**：16 篇两次踩同一个坑——KV 分块循环里 `S` 累加器只在循环外
  清零一次，第二块起把上一块的分数也加了进去（`max_abs_err` 稳定偏大 ~18%）。**每进入一个新 KV
  tile 都要清零 `S`**（`O` 才跨块保留）。单块 shape 能过、多块就错，就是这个。
- **CUDA 13 的 nvcc 不认 `-arch=sm_90a`**：会静默退化成 `sm_90`，ptxas 报
  `Instruction 'wgmma.mma_async ...' not supported on .target 'sm_90'`。必须写
  `-gencode=arch=compute_90a,code=sm_90a`。`scripts/run.sh`/`ncu.sh` 现支持 `ARCH=""` 跳过 `-arch`。
- **wgmma 的 smem 布局决定生死**：无 swizzle 的 K-major `INTERLEAVE`（8×8 core matrix，行距 128B）
  会让「连续线程写相邻 k-block」全撞同一 bank（16 篇实测 store bank conflict 2.7e8），且 tensor
  读操作数低效——`wgmma` 版反而比 `mma+ldmatrix` 慢一倍。**要用 SW128 swizzle**
  （描述符 `layout_type=1` + `base_offset` 相位）；先写小 GEMM 冒烟测试验证描述符。
- **SW128（K-major）速查**（20 篇已跑通）：物理布局 `[row/8][k/64][8][64]`，atom 1024B，
  元素 `(row,k)` 字节偏移 = `(rg*(K/64)+kg)*1024 + (rr*8+((kk/8)^rr))*16 + (kk%8)*2`；
  描述符 `start>>4`(bit0-13)、`LBO=16B`(bit16-29，K-major 恒 1)、`SBO=(K/64)*1024`(bit32-45)、
  `base_offset=0`(bit49-51)、`layout_type=1`(bit62-63)；**k16 步进 `floor(s/4)*1024+(s%4)*32`**
  （跨 atom 是 +1024，不是一律 +32）。冒烟 GEMM 132 TFLOPS @4096³、MLA 105.3（+24.5% vs INTERLEAVE）。
- **相邻两个 bf16/半精度标量存储会被 nvcc 合并成 `st.shared.u32` 并丢高 16 位**（20 篇实测
  3553 字节错位，P 矩阵一半元素变 0）。症状极隐蔽：QK/PV 单独定点测试全过，只有写回那段错。
  修法：显式 `__floats2bfloat162_rn` 打包成一次 `uint32_t` 存储（首地址 4B 对齐时）。
- **`wgmma` SS 的 B 操作数必须 K-major**（CUTLASS 的 dense GMMA traits 里 B 全是
  `smem_desc<Major::K>`）。所以 MLA 的 PV 必须把 `c_kv` `[k][dv]` 转置成 `[dv][k]`，
  这次转置的 global gather 是额外开销——这是 wgmma 相对 `ldmatrix.x4.trans` 的一个劣势。
- **按 DV 切 warp 省寄存器会带来 QK 重复（dup）**：`O[16,DV]` fp32 的寄存器墙逼着按 `DVW` 切
  warp，`dup=DV/DVW`。16 篇用 smem 共享 `P`（同 row 组两 warp 各算一半 KV）把 dup 从 2 降到 1，
  提升 16%；但别切太细——DVGRP 越大，每个 warp 都要完整读一遍 Q，反而不划算。
- **「H 个小 GEMM 共享同一个 B」要显式复用 B**：DSA 的 lightning indexer 是 $I=\sum_j w_j\mathrm{ReLU}(q_j\cdot k)$，
  $k$ 与 head 无关；默认实现每个 head 都 `ldmatrix` 重读整个 K tile，ncu 会显示 L1/TEX ~85%、
  DRAM ~2%。把 HG 个 head 一起算（每个 `(kx,nb)` 只加载一次 B）＋加宽 BN 摊薄 Q，18 篇从 222 → 311 TFLOPS。
  **但 HG 别贪**：HG=4/BN=128 寄存器 255+spill → 69 TFLOPS。每改一次 `-Xptxas -v`。
- **`O(S²)` kernel 的 block 调度顺序决定 L2 工作集**：把输出 tile 的网格写成 `grid.x=KV块`、`grid.y=query块`，
  同一 query block 的 KV 块连续调度，Q tile 才留得住 L2。18 篇 `grid.x=query` 时 S=32768 indexer 只有
  69.7 TFLOPS，交换后 214（3.1×）。判据：ncu `DRAM Throughput` 高而 L1/L2 不高 → 先怀疑调度顺序。
- **FP8 GEMM 的 `mma.sync` 在 Hopper 上打不满**：`mma.m16n8k32` 走 SM80 兼容路径，ncu 会显示 `math_pipe_throttle` 为主 + tensor pipe 只有 ~41%（实测 266 TFLOPS / 13.5%）。要打满 FP8 峰值必须换 `wgmma.m64nNk32`（SS 直读 smem 描述符）。**FP8 的 K-major SW128 atom 与 bf16 逐字节同构**（8 行×128B，只是每行 128 个 e4m3），20 篇的 `wgmma_sw128.cuh` 原样复用；`B[N][K]` 天然 K-major，GEMM 无需转置。换 wgmma 后同 shape 266→768 TFLOPS（+2.89×）。
- **FP8 的 `ldmatrix` 可复用**：`ldmatrix` 吃 8×8 b16，「2 个相邻 fp8 = 1 个 b16」，所以 16×32 fp8 = 16×16 b16 = 4 个 m8n8 矩阵，`ldmatrix.x4` 一次取回 `a0..a3`；B 存 `[N][K]` 用 `x2` 取 `b0/b1`。smem 行距 `BK+16` 消 bank。
- **FP8 的 wgmma asm 尾部操作数与 bf16 不同**：bf16 是 `..., p, 1, 1, 0, 0`（5 个），FP8 是 `..., p, scaleA, scaleB`（3 个）；照抄 bf16 模板会报 `Arguments mismatch for instruction 'wgmma.mma_async with FP8 types'`。操作数编号：64 累加器后 `da=%64, db=%65, pred=%66, scA=%67, scB=%68`。
- **per-tensor 的 wgmma 循环别每块 `wait0`**：acc 到最后才读时用 `wgmma.commit_group` 每块提交、覆盖 stage 前只 `wgmma.wait_group STAGES-2`，允许 mma 跨块流水（710→768，+8%）；循环外补一次 `wait0` 再读 acc。
- **per-block 缩放：相邻两列 c0/c1 各有自己的 scale**：`fin += sa[row]*sb[col]*acc` 时 c0/c2 用 `sb[col]`、c1/c3 用 `sb[col+1]`。只测常数 scale 会「假通过」，必须用逐列随机 scale 才能暴露（实测误差稳定 35%）。DeepSeek `weight_block=128×128` 时 `sb` 每个 n-tile 退化成标量。
- **扫 config 时 check 采样必须打散**：`__launch_bounds__` 写死线程数 < 网格所需时只会算一部分行；若采样步长 `% BM` 后总落在已算区域，会显示 `OK` 且性能虚高一倍（曾把 256×256 的 898 当成绩，真值 218）。用与 BM/BN 互质的步长（如 `i*1009`）。
- **exact top-k 用 radix-select，别全排序**：保序变换成 uint32 → 逐 8-bit 趟直方图定位第 k 大阈值（4 趟）
  → 收集。独占瓶颈是 shared `atomicAdd` 竞争，**每 warp 私有直方图** 18 篇带来 3.96×（26.7→6.75ms）。
  坑：①保序变换的逆**不自逆**（mask 依赖符号位；高位置位取低 31 位、否则取反），写错 `out_val` 全错；
  ②「整行塞 smem」在长序列反而更慢（S=32768：8.23 vs 6.75ms）——动态 smem 把 occupancy 锁成 1 block/SM，
  先算 occupancy 再决定。
- **TMA（`cp.async.bulk.tensor.2d`）的 SW128 与 wgmma 描述符逐字节同构**（23 篇）：`CUtensorMap` 用
  `CU_TENSOR_MAP_SWIZZLE_128B`，硬件写出的 smem 布局恰是 wgmma 的 K-major SW128，kernel 里零 swizzle 代码。
  坑：①`CUDA 13` 驱动枚举**没有** `CU_TENSOR_MAP_DATA_TYPE_FLOAT8_E4M3` → 用 `UINT8` 搬字节；
  ②`BK` 锁 128；③要 `-lcuda` 链接 `cuTensorMapEncodeTiled`；④tensormap 必须作 `const __grid_constant__` 参数。
- **mbarrier 的相位别用动态下标数组**（23 篇）：`uint32_t phase[STAGES]; phase[st]^=1`（`st=kb%STAGES` 运行期）
  会把数组推到 **local memory**，ncu 报 local memory 占 L1TEX 47.5% sector、`long_scoreboard` 8.1，只有 988 TFLOPS；
  相位就是「该 stage 第 n 次使用」的奇偶，直接算 `(kb/STAGES)&1`（full）/`(kb/STAGES-1)&1`（empty）→ 1217。
  **注意别只看 `Local Memory Spilling Requests=0`，要看 `Memory Workload Analysis` 里的 local memory 占比。**
- **warp specialization 的 empty barrier count = 所有消费者线程数**：只让每 WG 的 lane0 `arrive` 会与同 WG
  另一 warp 的 `wgmma.wait_group` 竞争（不确定对方读完了），实测会偶发错；让每个消费者线程都 arrive 才稳。
- **算力受限 GEMM：降 L2 流量（大 BM）> 堆 occupancy**：23 篇 128×128 s3 有 2 CTA/SM 但 L2 80%、1097 TFLOPS；
  256×128 s4 只有 1 CTA/SM、L2 57%，反而 1217。先看 ncu `L2 Cache Throughput` 与 tensor pipe 活跃度，别默认
  occupancy 越高越好。
- **per-block FP8 缩放的代价是寄存器→occupancy，不是折算 FLOPs**（24 篇）：`fin += sa*sb*acc` 要求每线程多一个
  跨 k 保留的 fp32 `fin`。BM=128/BN=128 时 288 线程的寄存器从 90 顶到 154，越过 2-CTA 门槛 113，occupancy
  2→1 CTA/SM，Compute 67.7%→49.7%。先算 `65536/线程数`（1 CTA）与 `65536/(2×线程数)`（2 CTA）再选 config。
- **别把 wgmma 累加器攒到寄存器里做 ping-pong**（24 篇）：主循环里读「还在飞的 wgmma 累加器」会让 ptxas
  主动串行化 wgmma（警告 `C7514`/`C7511`），实测比老实 `wait0` 折算还慢 2.3×。要么像 DeepGEMM 用
  `warpgroup_wait<0>` 严格分开，要么改 1-warpgroup/248-reg 布局。
- **per-block 的好几何可能寄存器不可行**：BM=256 的 per-block 累加器 = `512×128 = 65536` 恰为整个 regfile，
  无论怎么调都 spill（96 regs + 608B → 224 TFLOPS）。换几何前先做寄存器账。
- **bf16 的 TMA SW128：`BK` 必须是 64，`BN` 不能超 256**（28 篇）：Swizzle<3,4,3> 作用在字节上，
  128B/行 ÷ 2B = 64 个 bf16，所以 TMA box 内维固定 128B、沿 K 的 `BK=64`（描述符 `SBO=1024`）。
  另外 **TMA box 单维上限 256**，`BN=384` 会被 `cuTensorMapEncodeTiled` 直接拒掉（`invalid argument`）——
  想在 N=384 上用整块 N 行不通，只能 `BN=128` 分 3 个 n-tile。
- **权重天然 K-major：gate GEMM 的 B 免转置**（28 篇）：`wgmma` SS 的 B 要沿 K 连续，`W_g[E,H]` 的 H 连续
  正好满足，直接建 tensormap 即可；别照搬 27 篇给 `ldmatrix.trans` 准备的 `WgT[H,E]`。
- **tall-skinny GEMM 先算 wave 数再选 geometry**（28 篇）：输出 tile = `(M/BM)(N/BN)`。gate（N=384）
  在 M=16384 时只有 384 个 tile、2 CTA/SM 下 **1.45 wave**，`128×128 s3`（塞得下 2 CTA）赢；
  M=32768（768 tile）时反而是 **1 CTA/SM 的 `256×128 s4`** 赢（BM=256 让 A 复用翻倍、压 L2）。
- **L2 利用率高时别上 split-K**（28 篇）：拆 K 让更多 CTA 抢同一批 A 的 L2 行，gate 实测
  `s3` 565→`k2` 430、`k4` 348，全线更慢；`L2_PROMOTION_L2_256B` 同样负优化（600→576）。
- **MoE grouped 的第一性原理是并行度**：per-expert loop（G=384）每个 GEMM 只有 `N/BN × m_g/BM` 个 CTA
  （实测 ~24 个），132 SM 空转，耗时对 token 数不敏感（25 篇 8k/16k/32k 都是 13–14ms）。把 expert 编码进
  B 的行坐标 `b_row = group*N + n`（`(G,N,K)` 等价 `(G*N,K)`），一个 kernel 调度全部 tile，小 batch 3.6×。
- **TMA 的 `boxR` 必须跟着 BN 一起建**（25 篇坑）：producer 用 `expect_tx(BM*BK + BN*BK)` 声明字节数，
  若复用了 `boxR=128` 的 tensor map 却跑 BN=256，TMA 少搬的字节永远补不齐 → **mbarrier 死锁，且不报 CUDA error**。
  症状是 kernel 直接挂死；用设备端 `printf` 打到 producer 才能定位。写多 config 扫描时，descriptor 要按 config 重建。
- **能用 2D 坐标表达分组就别上 3D TMA**：3D tensormap（`dims={K,N,G}`）语义上更「正统」，但 2D 行折叠
  （`b_row=group*N+n`）复用同一套 SW128 描述符、少一层维度推理、排错简单，性能相同（25 篇）。
- **分组 GEMM 的墙常在 L2 而不是 DRAM**：每个 `(m_tile,n_tile)` CTA 读整块 `BN×K` 的 B，一个 expert 的 B
  被它的每个 m-tile 重读，L2 流量放大 `(M_total/BM)` 倍。25 篇 ncu 实测 L2 **81.9%**、DRAM 仅 52%。
  解法是 TMA cluster multicast（同 expert 相邻 m-tile 组队广播 B）。
- **ncu 对某些配置会「看不到 kernel」**：25 篇 masked 的 `moe_kernel` 在 ncu 下反复报
  `No kernels were profiled`（app 提前 disconnect），普通运行正常——不要因此怀疑 kernel 本身，
  用吞吐指标（B-read GB/s）替代即可。
- **`timeout` 杀不掉容器里的 GPU 进程**：`timeout N scripts/run.sh …` 超时时只杀掉宿主上的
  `docker exec`，容器里的可执行文件（如 deadlock 的 kernel）会**继续占着 GPU 空转**，污染后续所有
  基准（26 篇实测被拖慢约 2×、数字全废）。症状：`nvidia-smi` 看到残留进程，宿主 `pkill` 报
  `Operation not permitted`（进程属容器 root）。修法：`docker exec kernel_lab pkill -9 -f <name>.out`。
  **跑 benchmark 前先确认 GPU 干净**，超时后也要主动清理。
- **TMA cluster multicast（26 篇）**：`cudaLaunchKernelEx` 的 `clusterDim.x=CN` 要求 `grid.x % CN == 0`；
  leader 发 `cp.async.bulk.tensor.2d...multicast::cluster` + mask，每个 CTA 仍各自 `arrive.expect_tx`；
  共享 operand 的 empty barrier count 是 `CN × 消费者 **warp** 数`（只有每 warp 的 lane0 用
  `mapa.shared::cluster` 投到 leader），**写错不报错只死锁**；必须把「私有 operand 的 empty」与
  「共享 operand 的 empty」拆成两个 barrier，否则非 leader 的私有 operand 被超等拖慢。
- **`lts__throughput`（L2 请求/延迟占用）≠ `lts__t_sectors`（L2 字节）**：TMA multicast 能砍 A 的 L2
  读 sector 15–19%，但 L2 利用率几乎不降（16k：94.08%→94.36%），因为每个 CTA 仍要发自己的 TMA、
  且 leader↔peer 握手把访存耦合更紧。判据：**瓶颈在字节（大 batch、读放大高）才上 multicast**；
  小 batch/masked decode（纯 DRAM 带宽）只有负收益。26 篇只有 32768+128×128 赢 +4.3%。
- **bench 的每个子模式都要保证「上游数据已就绪」**（27 篇坑）：`perm`/`unperm` 单独跑时不会执行
  `route_scatter`，于是 `pos[]` 是未初始化值 → kernel 读到垃圾、读写都落在少数行上（命中 L2），
  ncu 会报出「91µs 搬 1.64GB」这种不可能的数字。修法：这些模式开头补 `run_topk(); run_meta();`。
  判据：带宽超过峰值、或 ncu duration 与 `bench_ms` 差 5× 以上，先怀疑输入没准备好。
- **ncu 的 `dram__bytes_write.sum` 对 write-back L2 的写会严重偏小**（27 篇）：permute 写 1.41GB，
  ncu 只记到 2.65MB（dirty line 未回写、metrics 窗口就结束）。**带宽结论一律用 `bench_ms` 的外部口径**，
  ncu 只用来看「瓶颈在哪一级」（DRAM/L2/Compute 百分比 + occupancy）。
- **对标 torch `index_select` 注意字节口径**：它对每个出现都读一次源（27 篇 `x` 被读 K 次，
  真实流量 `2·P·H·2` 而非 `(P+M)·H·2`）；其本身也贴 HBM（92%），手写 kernel 赢在「读一次写 K 行」
  少搬 1.72× 字节，不是赢在带宽。
- **「一变多」scatter 置换：读一次写 K 行 vs 每行一读一写**（27 篇）：前者流量 `(1+K)MH·2`、HBM 效率 ~81%；
  后者流量 `(K+1)MH·2`、效率 ~91%，但净时间前者快 1.53×。**先算流量再比效率**，别被「效率掉 10 个点」骗回去。
- **TMA 的 `boxR` 要同时匹配 `BM`（A）和 `BN`（B）**（31 篇）：25 篇记录了 `tmB` 的 `boxR=BN`；
  31 篇跑 `BM=64` 时忘了把 `tmA` 的 `boxR` 从 128 改成 64，`expect_tx(BM*BK+BN*BK)` 与实际搬运字节
  不符 → **mbarrier 永远等不齐、死锁且不报 CUDA error**（`timeout` 也杀不掉容器里的进程）。
  写多 config 扫描时，A/B 两张 tensormap 都要按当前 `BM/BN` 重建；定位靠设备端 `printf` 打到 producer。
- **per-block 缩放的代价是 occupancy，不是折算 FLOPs——工作点决定它贵不贵**（24/31 篇）：`fin` 让
  寄存器 90→155、occupancy 2→1 CTA/SM。prefill（算力受限）暴露 ~18% 损失；decode/masked（纯权重
  带宽）里 `wait0`+FMA 被 DRAM 延迟藏住，**完全免费**。分组场景（31 篇）比稠密单 GEMM 损失更小
  （18% < 23%），因为 grouped 天生偏 L2/带宽。
- **纯访存算子的优化顺序：先减 pass、再消 L1、最后才抠指令**（32 篇）：RMSNorm 每行 14KB，
  朴素两趟 44% → smem 行缓存单读 70%（但 ncu `l1tex` 76% 成墙）→ **整行留寄存器** 86%
  （每线程 `NG=ceil((H/8)/T)` 个 `uint4`，只多 ~16 个寄存器）→ 融合残差 87%。判断哪一级是墙看
  ncu `DRAM vs L1TEX vs L2`，**DRAM% 才是 HBM 利用率**。
- **融合的收益只看「省掉几个 pass」**（32 篇）：add+norm 省 1 个 pass（5→4）值；norm+FP8 量化只省
  0.5 个（8B→7B）却引入每 128 元素的 `amax`（shared `atomicMax`）+ 逐字节打包，实测反而从
  87% 掉到 66%。**输出字节占比小的融合不划算。**
- **这个 CUDA 工具链下 `float(fp8)` 返回原始位模式，不是数值**（32 篇）：`1.0f` 的 e4m3 位模式是
  `0x38`，`float(fp8)` 直接给 `56`，导致精度校验误差恒定在 155% 查不出逻辑错。必须走
  `__nv_cvt_float_to_fp8` + `__nv_cvt_fp8_to_halfraw`；也**别写 `fp8 v = fp8(storage_byte)` 再存**
  （重载会选错）。最小复现见 `32-fused-norm/fp8_test*.cu`。
- **Hopper `wgmma` 的 B 操作数只认 K-major，MN-major 描述符无效**（32 篇判决，解答 21 篇 J3）：
  用 CuTe 生成正确的 canonical MN 描述符（`LBO`/`SBO` 与手推一致，raw=`0x4000010000400000`）后跑定点
  GEMM，**改 LBO（64/256/1024）结果逐位不变**——硬件在 swizzle 布局下忽略 `leading_byte_offset`、
  按 K-major 解释 B。所以 MLA 的 V 转置不可避免；免转置只能等 Blackwell `tcgen05` 或走
  `mma.sync + ldmatrix.x4.trans`。别在这上面反复试描述符。
- **跨迭代复用 wgmma 累积器做软流水会被 ptxas 主动串行化**（32 篇）：让 `S`（QK 的 fp32 累积器）
  在 softmax 消费后又直接给下一块 QK 用，ptxas 报 `C7515`（非 wgmma 指令定义了 wgmma 累积器）并插入
  `wait_group`，寄存器顶到 255 + 536B spill，MLA 从 199 掉到 147。即使改用 wgmma predicate 覆盖代替
  清零、把 softmax 输出写独立数组也无效——`O` 的 rescale 是 PV 累积器，跨迭代必然触发。要重叠得像
  DeepGEMM 那样 1 warpgroup/248 reg + `warpgroup_fence_operand`。
- **`ncu` 里 `wgmma` 的寄存器墙和 smem 几何会锁死延迟隐藏**（32 篇）：MLA 融合 kernel 的
  Q(73.7K)+K(73.7K)+P(8K)+V(65.5K)=216KB 强制 1 CTA/SM、只有 8 个 warp；而 KT 必须 ≥64（SW128 atom）、
  BM=32 又不满足 wgmma 的 m64，几何上腾不出 V 双缓冲空间。判断 attention 类 kernel 先看 smem 总量与
  CTA/SM，再谈流水。
- **「L2 墙」先确认量的是不是最佳配置**（36 篇）：tall-skinny GEMM 的 B 重读次数 = `M/BM`。
  35 篇的 `L2 84%` 是在 `BM=128`（B 重读 256×）上采的；换成 `BM=256`（重读 128×）后 ncu 变成
  **tensor pipe 76%、L2 58%**，瓶颈完全反转。**换几何/换 config 前不要下瓶颈结论**。
- **TMA cluster multicast 的判据（36 篇汇总 26/29/36）**：只有在 ① ncu `lts__throughput` >75%、
  ② 瓶颈在**字节**（`lts__t_sectors` 读放大高）、③ `BM/BN` 已到 TMA box 上限、④ tensor pipe <60%、
  ⑤ 从 `CN=2` 起试 时才值得。`tensor >70%` 时广播省字节但跨 CTA 耦合会拖低 tensor 活跃度，实测
  compressor 投影 A/B multicast 全负（−3.7%~−19%）。**「放大 BM」和「广播 B」是同一个优化且无耦合，
  优先放大 tile。**
- **`cudaAccessPolicyWindow`（L2 persisting）只在命中率低时有用**（36 篇）：compressor 权重 14.7MB
  < 50MB L2、`L2 Hit Rate` 已 82%，钉进 L2 反而 −2.2%。先用 ncu 看 `L2 Hit Rate` 与 `lts__throughput`。
- **纯 TMA→wgmma 的消费者侧不需要 `fence.proxy.async`**（36 篇）：TMA 写 smem 与 wgmma 读 smem 都在
  async proxy，mbarrier 的 `complete_tx` 已完成排序；消费者（纯计算、不写 smem）再插 generic↔async
  fence 是多余的（实测中性）。23 篇起的 TMA kernel 都可以删掉这句。
- **TMA 的 `boxR` 必须同时匹配 A 的 `BM` 和 B 的 `BN`**（36 篇复现 25/31）：写多 config 扫描时
  `tmA` 的 box 行数随 `BM`、`tmB` 随 `BN`。compressor fp8 kernel 复用了固定 `boxR=128` 的 map 去跑
  `BM=64/192`、`BN=256`，`expect_tx` 与实际搬运字节不符 → **mbarrier 死锁且不报错**（stdout 被 kill
  吞掉，看不到任何输出）。
- **基准的长短会改变结论**（36 篇）：同一 kernel `bench_ms(run, 200, 100)` 比 `bench_ms(run, 5, 30)`
  慢 ~10%（H100 共享机上持续负载会降频）。**head-to-head 必须在同一进程、同样 warmup/iters 下测**；
  跨文件的数字不能直接比。超时后用 `docker exec kernel_lab pkill -9 -f <binary>.out` 清残留进程，
  否则 100% 占用的死锁 kernel 会污染后续所有基准（36 篇实测被拖慢 ~2×）。
- **用 wgmma 指令自身的 `scale_d=0` 清零累加器，别用通用 FMA 写 `acc=0`**（37 篇）：后者会被 ptxas
  判为 `C7514`（非 wgmma 指令定义了 wgmma 累加器）并主动串行化 wgmma。DeepGEMM 的
  `WGMMA::wgmma(desc_a, desc_b, accum, k)` 把循环下标 `k` 当 `scale_d`（`k=0` 走 `ScaleOut::Zero`、
  `k>0` 累加）就是这个意思。改完后单累加器路径的 C7514 消失（但 ping-pong 仍被寄存器墙卡住）。
- **`setmaxnreg` 不是万能的**（37 篇）：内联 `setmaxnreg.inc/dec.sync.aligned.u32` 想给 math warpgroup
  多发寄存器，ptxas 可能回 **`C7507 'setmaxnreg' ignored to maintain minimum register requirements`**
  并直接忽略。别照搬「DeepGEMM 用 232/248 寄存器」的结论，它的线程几何（128 线程 TMA warpgroup +
  persistent 调度器）与你的不同；先看 `-Xptxas -v` 实际拿到多少。
- **mbarrier 的期望 count 必须与真实 arrive 次数逐一相等**（41 篇）：consumer 释放 `empty` 若写成「每个 consumer warp 的 lane0 arrive」（4 个 warp → 4 次）而 `mbar_init(empty, NCONS=1)`，会 **over-arrive** 把相位打乱 → producer 永久自旋，**不报 CUDA error、`timeout` 也杀不掉容器进程**。修法：`count = 所有 consumer 线程数`，每个 consumer 线程各自 `mbar_arrive`（复现 23 篇结论）。定位靠抽最小复现（`mbar_init/arrive/wait` + producer/consumer 两分支）。
- **小 `M` 的 split-K GEMM 先算 wave quantization**（41 篇）：并发槽 `Nc = 2 CTA/SM × 132 = 264`；`grid = (N/BN)·KSPLIT` 若不是 `Nc` 整数倍，`ceil(grid/Nc)` 会多出一整波、而最后一波可能只有几个 CTA。判据 `T ≈ ceil(grid/Nc) × (K/BK/KSPLIT)`，取最小 KSPLIT，但每项别少于 ~8 个 stage（prologue 反噬）。实测同 kernel 换 `N` 让 grid 从 544(3 波) 变 528(2 波)，**−20%**。
- **warp specialization 提 occupancy 不等于提速**（41 篇）：把 ALU（反量化/激活）与 `wgmma` 拆到不同 WG 后，occupancy 6.5%→21%、regs 156→90，但只快 1.07×——`fixed-latency` stall 高是**症状**，根因常在访存延迟/尾波。先看 ncu 确认瓶颈再决定是否上 WS。
- **ue8m0（2 的幂）的 block scale 可以精确折进 e4m3 操作数**（37 篇）：`sa=2^e` 时 `q·sa` 只是 e4m3
  的指数平移，尾数不丢。于是 per-block GEMM 可退化成 per-tensor GEMM（`Σ(q_a·sa)(q_b·sb)=Σsa·sb·q_a·q_b`），
  折算开销彻底消失：GEMM 本体 940.6→1207.9 TFLOPS（1.28×）。折 B（权重）一次性免费；折 A（激活）
  应融进上游 per-1×128 动态量化器（写回前多乘一次 sa，零额外访存），端到端 1.19×。**边界**：只对
  2 的幂 scale 精确（任意 fp32 scale 会多半个 ulp 舍入）；`A'` 与 scale 绑定，多 GEMM 共用 A 时要各存一份。
- **`ldmatrix` 能从 SW128 tile 里直接读「转置」的数据**（42 篇）：SW128 只在 **16B 粒度**做
  `c'=c^r` 置换，每个 16B chunk（8×bf16）完好；`ldmatrix` 每个 lane 只需一个 16B 地址。
  所以 `sw128_off(key, dv, DK)` 算出地址喂 `ldmatrix.x4.trans`，就能从 wgmma 用的那块 SW128
  K tile 里读「转置」的 V——MLA 的 PV 因此**免掉 V 转置**（一个 KV tile 只从 global 读一次）。
  反例：`wgmma` 的 B 只认 K-major（J4），纯 wgmma 版必须把 V 转置成 `V^T[DV,KT]`
  （每 tile 4096 次跨 1KB 标量读），实测比 `mma+ldmatrix` 更慢（181 < 190）。
- **怀疑某段 global 重载是墙，就把它短路掉看天花板**（42 篇）：保留全部骨架，只把那段 load
  用 `if(false && …)` 关掉（结果无意义）。纯 wgmma MLA decode 完整版 181 → 关掉每轮 V 重载
  **368 TFLOPS**，一步锁定「V 转置是全部瓶颈」。判据是 `long_scoreboard` 同步下降。
- **`mma16816` 的累加器指针是「4 个连续 float」**（42 篇）：39 篇的 `O` 是 `float O[NDH][4]`，
  传 `O[dn*2]` 得到一行；扁平 `float O[DVW/8*4]` 若传 `O+dn*2`，相邻两个 n8 tile 的累加器会
  **重叠**（`O+dn*2` 与 `O+dn*2+1` 只差 1 个 float）→ 结果全错。正解 `O + (dn*2)*4`。
  手搓 mma 时先写最小 GEMM 复现（`pvt_test.cu`）验证。
- **跨 WG 交换 softmax max/sum 的行号别重复加 `16*W`**（42 篇）：`r0 = 16*(W&3) + (lane>>2)`
  **已经是 CTA 内 0..63 的行号**；`red[wg*BM + 16*W + r0]` 会把它推到 96/144 越界别名。
  症状隐蔽：`L=32768` 误差只有 7e-5（softmax 鲁棒掩盖），但 `L=64`（单 tile）相对误差 13% FAIL。
  **短序列也要对拍**。
- **`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 上限恰是 232448B**（42 篇）：混合版 MLA
  decode 的 smem = `Q 73728 + 2×K 73728 + P 9216 + red`，多算 1KB 冗余就 `invalid argument`。
  缓冲块数×块大小要逐字节算，`red` 按当前模板参数的真实大小给。
- **persistent / 跨 item 流水：epilogue 一进热循环，寄存器就爆炸**（43 篇）：把
  `(item,kb)` 展平成一维 stage 流后，若 epilogue（`atomicAdd`/地址计算）写在 consumer 的
  flat 循环体内，编译器会把 `acc[64]` + epilogue 状态一起保活并软件流水，寄存器 **90→207**、
  occupancy 2→1 CTA/SM、慢 40%。改成 **item 外层 / kb 内层**、epilogue 落在内层循环之外即回到
  90；内层还要 `#pragma unroll 1`（`CHUNK=4` 展开 4 份 wgmma 会 128 regs+272B spill）。
  **定位法：把 epilogue 整段删掉再 `-Xptxas -v`，寄存器立刻露底（80）。**
- **`acc[64]` 把 wgmma 融合算子的 occupancy 锁死在 2 CTA/SM**（43 篇）：`(BM/64)(BN/128)·64`
  个 fp32 累加器 + smem 描述符至少要 ~90 regs，而 3 CTA/SM 的门槛是 `65536/(3×256)=85`。
  想靠堆 CTA 消 wave quantization 走不通（ptxas `C7602 Insufficient registers`），只能调
  work-item 粒度。先算 `regs ≈ acc + 26` 再决定。
- **持久化的 stage 流：`CHUNK ≥ STAGES` 是硬约束**（43 篇）：item 比流水环还短（如 `CHUNK=2`、
  `STAGES=3`）时 barrier 相位 `t/STAGES` 错乱，实测 **illegal memory access**；`static_assert`
  钉死。跨 item 的累加器清零用 wgmma 自身 `scale_d=0`（避 `C7514`）。
- **优化同时改了「并行度」和「每 stage 指令数」时，ncu 单次可能和 event 稳态反号**（43 篇）：
  持久化把 wave 2.58→1（event 稳态快 2.6%），但每 stage 多两个 `gid/ntiles` 整数除法，
  ncu 锁频下被放大 → ncu 单次反而显示持久化慢（Compute 47%→51%）。**decode 服务取 event
  稳态（多 warmup + 多 iters）为发布口径；ncu 用来读机制（waves/occupancy），别单独下快慢结论。**
- **wave quantization 和 K-split 归约税是一对矛盾**（43 篇）：细 K-chunk 增并行度、消尾波，
  但每 item 的 `atomicAdd` 归约字节 ∝ `M×N×(N/BN)×(nblk/CHUNK)` 随 `M` 线性涨。交叉点：
  `M≤16` 持久化+细 chunk 赢（~1.02–1.04×），`M≥32` baseline 粗 chunk 赢。扫参必须跨过
  「归约流量 ≈ 权重流量」的拐点。
- **算子的形状要跟着 work point 走，别一把 m64 打天下**（44 篇）：W4A16 的
  `AI(M)=2MNK/(0.53NK)≈3.76M`，bf16 TC ridge≈295 → 只有 `M≳78` 才轮到张量核；
  `wgmma.m64n128k16` 在 M=1 白算 63/64 行，kernel 退化成延迟受限（DRAM 11–18%）。
  M=1 换「每 warp 一行 + warp 内沿 K 并行 + `x`/scale 常驻 smem + `shfl` 归约 + `__ldcs`」
  的 GEMV：**0.0284ms / 1566.6 GB/s / 49.7% HBM，比 tensor-core M=1 快 2.81×**。
  先算 `AI` 再选 GEMM/GEMV，是比调 tile 更值钱的一步。
- **ncu 报 shared bank conflict 多 ≠ conflict 是瓶颈**（44 篇）：GEMV 里 conflicts 210 万
  （70% wavefront 多余）、ncu 给「Est. Speedup 59%」，但按经典办法做免冲突置换布局后
  conflicts 降到 5.4 K、**反而慢 6%**——因为拆散了 4 个连续 16B 读的局部性，而真正的墙是
  **ALU pipe 64.9%**（反量化的移位/掩码/I2F），LSU 很闲。**先看 SOL 里哪级 pipe 到顶，
  再决定动不动访存**；「Est. Speedup」只是上界提示。
- **低比特反量化别轻易用 smem LUT 替 ALU**（44 篇）：int4 的 `(nib)-8`+`I2F` 是 3 串 ALU，
  但 256 项 `float2` LUT 的随机 8B 读每 lane 命中 2 个 bank、warp 冲突爆炸 → 实测
  **慢 68%**（16 项一维 LUT 也慢 14%）。用 LUT 把 ALU 成本换成访存成本，只在访存不冲突
  且余量大时才划算。
- **`__int_as_float(0x4B000000|n)=2^23+n` 位技巧在量纲悬殊时不可用**（44 篇）：GEMV 里
  每项乘积被放大到 ~2e6、累加 ~7e7，f32 ulp≈8，真答案只有个位数 → **灾难性抵消**
  （err 15% FAIL）。换小 bias(16) 又因尾数 LSB≠1 不成立。只有被乘数与结果同量级才安全。
- **上 warp specialization 之前先数「目标工作占多少指令/pipe」**（48 篇）：WS 只能*重叠*、
  不能*减少*工作。W4A16 小 M 的 `mma_b16` 里反量化的 `pipe_alu`+`pipe_fma` 占 **78%**、
  张量核只占 2.3%（tensor pipe 7.3%），把它拆进 producer WG 后 ALU/FMA 指令**一条没少**、
  总指令反而 +9%，且 `long_scoreboard` **1.77→10.19（5.8×）**——生产者 warp 无事可做、独自干等。
  判据：**被拆出去的工作是不是「发射即返回」的异步操作（TMA）**；要等结果的反量化/激活不适合。
- **`long_scoreboard` 是「该不该拆 warpgroup」的指纹**（48 篇）：拆完若它不降反升，
  说明你把「有别的指令可发射来躲延迟」的 warp 拆成了「独自干等」的 warp。别只看
  `sm__warps_active`/occupancy 上升就以为变好。
- **延迟受限的小 M GEMM：`KSPLIT` 粒度比 tile 形状更值钱**（48 篇）：`mma_b16` 的
  `KSPLIT=8→16` 让 waves 2.06→4.12，`M∈[3,16]` 一致 **+5~11%**（切短 CTA 让更多独立访存流
  互填 `long_scoreboard`）；但 KSPLIT 不是越大越好，`nblk_z` 少到 3~5 个 tile 时
  prologue/drain + `atomicAdd` 归约税会吃回去（M=16 k32 比 k16 慢 13%）。
- **文章 `date` 落在未来会被 Hugo 静默跳过**（48 篇踩到）：本机时间 04:28 CST 时写了
  `date: 2026-09-22T06:30:00+08:00`，`hugo --gc` **无 ERROR** 但
  `public/posts/<slug>/` 不存在 → 线上 404。发布前必须 `ls public/posts/<slug>/` 确认；
  时间不确定就写一个「现在之前」的时刻。
- **跑基准前必须确认 GPU 上没有残留的 compute 进程**（54 篇踩到）：多次 `timeout` 触发的
  死锁 kernel 会一个个留下来占着 GPU 0，**让所有基准一致地虚慢 ~2×**（且宿主 `pkill` 对容器
  root 进程无效）。症状极隐蔽：单个 kernel「app 测 49 ms / ncu 测 22 ms」差 2.2×、cuBLAS 也
  比 torch 慢一倍。修法：每轮 timeout 后 `docker exec kernel_lab pkill -9 -f <bin>`，跑基准前
  `nvidia-smi --query-compute-apps=pid,process_name --format=csv` 确认空。**app 与 ncu 数字
  系统性差 ~2× 时，先查残留进程，别急着怀疑 clock。**
- **手搓的 grid remap（superblock swizzle / 1D 解码）必须是严格双射**（54 篇踩到）：当
  `Nt/GN` 不整除（如 1010/8）时，用「原 2D 网格 + 块内解码」会让部分 `(mt,nt)` 永远不被任何
  `blockIdx` 生成，输出留脏值、**不报错**，loss 悄悄错（11.764 vs 正确 11.7698）。正解是
  「1D 网格 = `ceil(Mt/GM)·ceil(Nt/GN)·GM·GN` + 越界 `return`」。**判据：任何 remap 改完都要
  用「对拍一个全局标量」而不是「kernel 正常退出」来验证**（本例损失值）。
- **扫参前先查 `cudaFuncSetAttribute` 的返回值，否则失败 launch 会给出「鬼数据」**（55 篇踩到）：
  smem 超 232448B 时 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 返回错误、kernel 不启动，
  但若 `bench_ms` 只按 event 计时、不管返回值，就会得到 1.4e-6 ms 的假时间 → `10^7 TFLOPS`。
  实例：`256×128` 下 `STAGES=5` 要 245KB 超限，扫参里出现 `10907119 TFLOPS`。约束
  `(B_M+B_N)*128*STAGES ≤ 227KB`；扫参里任何「超峰值 10×」的吞吐先怀疑 launch 没成功。
- **同一算子的 bf16/fp8 可以用一个 `bool FP8` 参数化**（55 篇）：SW128 的 128B atom 对 bf16（8×64）
  与 fp8（8×128）**逐字节同构**，`make_desc_sw128`/`SBO=(K/64)*1024`/k-step 地址函数全不用改，
  只差 `BK`（64/128）、TMA `rowbytes`（`K*2`/`K`）、wgmma 指令（`m64n128k16`/`m64n128k32`）。
  **但 per-row 激活 scale 必须在 epilogue 乘回**（或折进存进去的 e4m3）：漏掉时 rel-RMS 直接
  到 1.6e4；wgmma `m16n8` 里 `acc[j*4+0/1]` 同行、`acc[j*4+2/3]` 同行，同一行 4 个累加器共用一个 scale。
- **别只信 `config.json` 的 `quantization_config`，要读真实 tensor 头**（56 篇）：`quantization_config`
  只描述**默认**方案（`e4m3 + ue8m0 + 128×128`），per-tensor 的 `expert_dtype: fp4` 会把它覆盖。
  V4-Pro 的 routed experts 实际是 **FP4 e2m1 + E8M0 block-32**（权重存 `int8[I,H/2] + E8M0[I,H/32]`；
  `7168/2=3584`、`7168/32=224` 就是指纹），shared expert/注意力才是 fp8 block-128、lm_head bf16。
  写算子前先 `safetensors` 列一遍 dtype/形状，别照抄 config。
- **`e2m1 ⊂ e4m3` 且 scale 是 2 的幂 ⇒ FP4 可逐位无损折进 FP8**（56 篇，同 37 篇思路）：
  `e2m1` 的 8 个幅值 $\{0,.5,1,1.5,2,3,4,6\}$ 全是 3 位尾数可表示，乘 $2^e$ 只平移指数；
  最小 $.5\cdot2^{-8}=2^{-9}$ 恰是 e4m3 最小次正规。真实权重实测折叠 relRMS = **0.00e+00**。
  于是 34 篇的 fp8 grouped kernel 可原样消费 FP4 专家（host 侧一次预折叠），专家显存/传输减半；
  per-128×128 block 也能无损折（37 篇）。**只对 2 的幂 scale 精确**。
- **低比特 decode 的账在解码的 ALU / smem 延迟，不在像素带宽**（44/45/56 篇）：`e2m1` 不是偏置整数，
  逐 nibble 算术解码 ~7 条/nibble（int4 的 `__vsubss4` 只要 ~1.25 条/权重）→ 直接 compute-bound
  （ncu Compute 65.5%、DRAM 19.6%）。换成 **256 项 `uint16` smem LUT（一输入 byte → 两个 int8）**
  后 **1.99×**（0.0199→0.0100 ms，Compute 降到 20.5%）；LUT 的随机 `uint16` 读会撞 bank
  （59% 多余 wavefront、`short_scoreboard`），下一堵墙是免冲突布局。
- **大规模逐元素 helper kernel 必须 grid-stride**（57 篇踩到，代价最大的一次）：host 侧预处理
  （折叠 / 转置 scale / 量化）若写成固定 `<<<8192,256>>>` + `if (i >= tot) return`，只覆盖
  0.04% 的数据，其余是未初始化显存。**最毒的是 CPU 参考读同一份坏数据，对拍全过**，把 bug 藏了
  一整轮（我还误判成流水/竞态，换了两版 mbarrier 方案）。判据：任何带 `if (i>=tot) return` 的
  kernel 都要有 `for (i = ...; i < tot; i += gridDim.x*blockDim.x)`。
- **host 侧没有「读 fp8 原始字节」的便利**（57 篇复现 32 篇）：`std::vector<fp8>` 里 `ha[k]` 会走
  `__nv_fp8_e4m3` 的转换运算符，`(float)ha[k]` 给位模式、`(unsigned char)ha[k]` 先转 float 再截断。
  用 `reinterpret_cast<const unsigned char*>` 取字节，或存成 `std::vector<unsigned char>`，或手写
  e4m3 解码。**先验证 host 数据本身，再怀疑 kernel**。
- **fp8 的 SW128 与 bf16 不同**（57 篇）：fp8 一行 128B = **128 个元素**（bf16 是 64），16B chunk
  下标是 `k/16`（bf16 是 `k/8`）；照抄 bf16 公式会写错。用「手写布局 vs TMA 逐字节对拍」的最小
  复现（`swz_test.cu`）先验布局。另：SW128 置换是**纯 16B chunk 置换**，读 `uint4` 写 `uint4`
  即可（逐字节搬要慢 3.45×）。
- **wgmma 是 warpgroup 级**（57 篇）：单 warpgroup 只算 m64；BM=128 要 **2 个 warpgroup**（256 线程）
  各算一半行，A 描述符按 `wg*64*BK` 偏移。最小复现里线程数给够、但行映射只写一半，会静默漏一半输出。
- **小 N 的 GEMV/decode 先看 `Waves Per SM` 和 `Issued Ipc`，别顺着 ncu 的 bank conflict 就走**（58 篇）：
  「一行一 warp 收口」把并行度锁死在 `#warp = N/RWW`；`N=3072` 时只有 3072 warp（23/SM、
  `Waves 0.36~0.48`、occ 32.8%、`Issued Ipc Active 1.08`、Issue Slots Busy 20.9%）→ **并行度不足**。
  此时去掉 bank conflict（`__byte_perm` 位运算解码）反而慢 29%、split-K 把 waves 抬到 3.88 却慢 30%。
  判据：`Waves<1 && Ipc<1.5 && Issue Slots Busy<30% && DRAM<80%` ⇒ 改并行粒度（grouped），
  不是抠指令。唯一有效的是**软件流水深度 1→2**（0.0100→0.0090 ms）。
- **`__byte_perm`（PRMT）的 selector 是「每个 selector byte 的低 4 位、按 nibble 排布」**（58 篇）：
  输出 byte `i` 由 selector 第 `4i` 位起的 nibble 决定（0-3 选 `x` 的 byte，4-7 选 `y` 的）。
  把 4 个 byte 的低 3 位压成 selector 要 `(k&0xF)|((k>>4)&0xF0)|((k>>8)&0xF00)|((k>>12)&0xF000)`；
  直接拿展开后的 byte 当 selector 会全错。最小复现必须**穷举**（`prmt_test.cu` 扫 2^28）而不是抽点。
  且 PRMT 解码是 **ALU 密集**（~3 指令/nibble），在 ALU 已被 decode 占满的 kernel 里是负优化。
- **MoE decode 是 grouped GEMV，不是 top-k 个独立 GEMV**（59 篇）：一步 decode 要算 `B×topk`
  个 `(token,expert)` 对。按 expert 折叠进一个 kernel（`grid.y=experts`），一个 CTA 吃下某专家
  全部 m 个 token，**权重每专家只读/decode 一次、被 m 个 token 的 `dp4a` 复用**，权重复用
  `=pairs/active≈B·topk/E`。`B=256` 相对「一次 launch 但 M=1」快 **3.22×**、相对逐对 launch **5.87×**。
  要点：① 权重必须**铺满真实专家数**（否则全命中 L2、带宽虚高）；② grouped kernel 的 smem/寄存器
  要**随真实 m 缩放**（`MTMAX` 模板分派，且 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`
  要对**每个模板实例**设），按模板上限一把分配会把 occupancy 锁死；③ 三口径（逐对 launch /
  batched M=1 / grouped）能把「并行度」和「权重复用」两个变量分离。
- **「lane 沿 K 串行、每 chunk 读两个 `uint4`」的 int8 GEMV：激活要拆两平面消 8-way bank conflict**（59 篇）：
  16B 权重 chunk = 32 个 k 值；int8 激活 32 个 k = 32B = 两个 `uint4`。若按 `[token][K]` 行主序存，
  warp 内相邻 lane（`c=lane+32i`）地址差 32B → 每 8 个 lane 回到同批 bank。把 chunk 的前/后 16B
  分进 **plane0/plane1**，lane 地址变相邻 16B、warp 铺满连续 512B，零冲突。实测 batched 67.6%→74.5%、
  grouped 40.6%→59.9%。这个修复对 batched 基线也生效 → 修的是公共瓶颈。
- **群组/小 N 算子把 HBM 用到 ~60% 后，下一堵墙常常是 L1/TEX（LUT 随机读），但别急着上 PRMT**（59 篇）：
  grouped FP4 decode 的 ncu 是 DRAM 60.1% / **L1TEX 83.9%** / Compute 41.1%，残余 48% 多余 wavefront
  几乎全来自 256 项 `uint16` LUT 的随机读。换 PRMT（寄存器表）**慢 33%**——`dp4a` 已在整数管道上，
  换访存税为 ALU 税方向反了。判据：SM 利用率 41%≠ALU 空闲；先试**冲突无关的 LUT**（padding / 分半表）。
