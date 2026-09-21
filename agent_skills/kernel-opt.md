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
