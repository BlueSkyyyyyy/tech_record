---
title: "CUDA 算子调优（二十六）：MoE grouped GEMM 的 L2 墙 — TMA cluster multicast 实测只赚 4%"
date: 2026-09-21
draft: false
weight: 26
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "grouped GEMM", "FP8", "e4m3", "wgmma", "TMA", "cluster multicast", "mbarrier", "DeepSeek", "Hopper", "系列"]
categories: ["算子开发"]
---

[上一篇]({{< relref "cuda-kernel-opt-25-moe-grouped-gemm" >}})把 MoE 的 expert FFN 塞进了一个 grouped GEMM（384 个 expert 共享 N/K，B 按 `group*N+n` 折叠成行坐标），256 GB 权重、大 batch 最好到 **1008 TFLOPS**。但 ncu 也指出下一堵墙：**L2 吞吐 81.9%、DRAM 只有 52%**——瓶颈是「每个 CTA 都把整块 A/B tile 从 L2 拉进自己的 smem」造成的 L2→SM 流量，而不是 HBM。

这一篇兑现上一章的预告：**用 TMA cluster multicast 把 A tile 从 L2 只取一次、广播给同一 cluster 的多个 CTA**。我把这套协议完整跑通了（实测正确、ncu 证实 L2 读 sector 降了 15–19%），但结论很克制：

- **只有在大 batch（32768 tokens）且几何是 128×128 时，multicast 才赢**：`960.7 → 1001.6 TFLOPS（+4.3%）`，L2 读 sector 降 **18.8%**，`long_scoreboard` 从 11.6 降到 8.0。
- **小 batch 反而更慢**：8192 tokens `867.8 → 804.0（−7.4%）`、16384 tokens `947.7 → 926.8（−2.2%）`；masked decode `−3.9%`。
- 根因：multicast 切掉的是 **L2 的字节数**，但 `lts__throughput`（L2 的请求/延迟压力）**几乎不降**（16k：94.1%→94.4%），而跨 CTA 的流水耦合让 tensor 利用率掉了 2.8 个百分点。**只有当字节数真正成为瓶颈（大 batch）时，省下来的 L2 才会兑现成时间。**

顺带把 **B 的放大用更大 M-tile 消掉** 这条更"正统"的路也试了：BM=256 不但没赢，256×256 还因为 wgmma 寄存器不足被 ptxas 串行化（`C7511`）崩到 124 TFLOPS。所以这篇是一次**有明确边界的负/微正结果**：multicast 值得懂，但别默认它能救 L2。

| 场景 | 基线（CN=1） | **A-multicast（CN=2）** | 变化 |
|---|---|---|---|
| prefill @8192 | 867.8 | 804.0 | **−7.4%** |
| prefill @16384 | 947.7 | 926.8 | −2.2% |
| prefill @32768 | 960.7 | **1001.6**（9.6406 ms） | **+4.3%** |
| decode masked | 174.2 | 167.5 | −3.9% |

环境：H100 SXM 80GB（132 SM，HBM 3352 GB/s，FP8 dense 峰值 1978 TFLOPS），CUDA 13.2。
数字来自 `26-moe-cluster-multicast/moe_cluster_S{8192,16384,32768}.out.txt`，ncu 见 `ncu_c1_c2_16384.out.txt` / `ncu_c1_c2_32768.out.txt`。

---

## 1. 先算清楚「L2 墙」到底是 A 还是 B

grouped kernel 的 grid 是 `(N/BN, M_total/BM)`，每个 `(m_tile, n_tile)` CTA 都要自己读一块 A tile（`BM×K`）和一块 B tile（`BN×K`）。于是：

- **A 的读放大** = 同一块 A 被多少个 n-tile 读 = `N/BN`；
- **B 的读放大** = 同一块 B 被多少个 m-tile 读 = 平均 `M_total/(BM·G)`。

总 L2 读流量（字节）：

$$
T_{\text{L2}} = M_{\text{total}}K\cdot\frac{N}{BN} + GNK\cdot\frac{M_{\text{total}}}{BM\cdot G}
= M_{\text{total}}NK\left(\frac{1}{BN}+\frac{1}{BM}\right)
$$

有意思的是：**当 `BM=BN` 时，A 和 B 的流量恰好相等**——都等于 `M_total·N·K/128`。以 32768 tokens（`M_total=219264`）为例：

```
A: 219264 × 7168 × (3072/128) = 37.7 GB
B: 384×3072×7168 × (219264/(128×384)) = 8.45 GB × 4.46 = 37.7 GB
合计 75.4 GB（理想只需 A 1.57 + B 8.45 = 10 GB，放大 7.5×）
```

所以 multicast A 或 B 理论上都只能砍掉一半中的一半（CN=2 省 1/4 总流量）。先做 **A（沿 N 方向的 cluster）** 有个工程上的决定性优势：

> **A 只由 m-tile 决定。** 沿 N 组 cluster（`cluster.x = CN`）的 CTAs 一定共享同一个 m-tile，因此**一定共享同一块 A**，不受 expert 分组边界影响。
> B 只由 `(expert, n-tile)` 决定，沿 M 组 cluster 会跨越 expert 边界（32768 tokens 时平均每 expert 只有 4.5 个 m-tile，相邻两个常常属于不同 expert），要做 DeepGEMM 那套 `is_peer_cta_alive` 动态回退，复杂且在小 batch 完全失效。

于是本文先做 A 广播，B 广播留作后续。

---

## 2. TMA cluster multicast 的协议

普通 TMA 是「一条指令搬一块到自己 smem」。`cp.async.bulk.tensor.2d...multicast::cluster` 在此基础上多一个 **CTA mask**：由 cluster 里某一个 CTA 发起，硬件把同一块数据**复制到 mask 里每个 CTA 的同偏移 smem**，并且**在每个目标 CTA 的同偏移 mbarrier 上补 transaction 字节数**。

```cpp
// moe_cluster.cu:131
__device__ __forceinline__ void tma_load_2d_mcast(const CUtensorMap* tmap, void* dst,
                                                  int c0, int c1, uint64_t* bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
```

### 2.1 cluster 怎么起

用 `cudaLaunchKernelEx` 的 cluster attribute，而不是 `<<<>>>`：

```cpp
// moe_cluster_launch.h:66
attr[0].id = cudaLaunchAttributeClusterDimension;
attr[0].val.clusterDim.x = CN;   // A 广播沿 x（N 方向）
attr[0].val.clusterDim.y = 1;
attr[0].val.clusterDim.z = 1;
cudaLaunchKernelEx(&cfg, fn, tmA, tmB, D, gl, N, K, max_m, mlim, scale, dbase);
```

cluster 内 rank 用 `%cluster_ctarank` 读（`moe_cluster.cu:115`）。要求 `grid.x % CN == 0`：`N/BN` 是 24 或 12，对 CN ∈ {2,4} 都整除。

### 2.2 producer/consumer 的握手（关键）

multicast 下 producer/consumer 的 barrier 语义和单 CTA 不同：

```
                   rank0 (leader)                    rank1
producer:  A: multicast(mask=0b11) ──┐        A: (不发)
           B: 自己的 TMA            │        B: 自己的 TMA
                                    ▼
full[st]:  每个 CTA 各自 arrive.expect_tx(BM*BK + BN*BK)；A 的完成信号由硬件
           补到每个目标 CTA 的同偏移 full barrier，B 由自己的 TMA 补。
                                    │
consumer:  读完 stage 后：empty[st] 每个消费者线程本地 arrive（B 的释放）
                          aempty[st] 每个 warp 的 lane0 用 mapa 投到 rank0（A 的释放）
producer:  覆盖前：等 empty（只等本 CTA），rank0 额外等 aempty（全 cluster 都读完 A）
```

对应的代码：

```cpp
// 初始化：B 的 empty 是本 CTA 消费者数；A 的 aempty 是全 cluster 的消费者 warp 数
mbar_init(empty  + s, NCONS);        // moe_cluster.cu:194
mbar_init(aempty + s, CN * NCW);     // moe_cluster.cu:195

// producer：rank0 发 A multicast；B 每个 CTA 各发各的
if (rank == 0)
  tma_load_2d_mcast(&tmA, As + st*BM*BK, kb*BK, a_row, full + st, (1u<<CN)-1);
tma_load_2d(&tmB, Bs + st*BN*BK, kb*BK, b_row, full + st);

// consumer：B 本地 arrive；A 用 mapa 投递到 rank0
mbar_arrive(empty + rs);                              // moe_cluster.cu:259
if (lane == 0) mbar_arrive_cluster(aempty + rs, 0);   // moe_cluster.cu:260
```

`mapa.shared::cluster` 把「本 CTA 的 barrier 地址」映射成「目标 CTA 的同偏移地址」，是实现**跨 CTA barrier arrive** 的唯一手段：

```cpp
// moe_cluster.cu:95
mapa.shared::cluster.u32 rem, %0, %1;      // rem = 目标 CTA 的 barrier 地址
mbarrier.arrive.shared::cluster.b64 _, [rem];
```

三个必须做对的地方：

1. **barrier 初始化后要 `cluster_sync()`**（`moe_cluster.cu:200`）——远端 CTA 可能立刻来 arrive，本 CTA 的 barrier 必须已 init。`fence.mbarrier_init.release.cluster` + `barrier.cluster.arrive/wait` 一步都不能少。
2. **A 的释放和 B 的释放用两个不同的 barrier**。我第一版把「全 cluster 消费者数」套在唯一的 empty barrier 上，结果**非 leader 的 B（私有）也被迫等全 cluster**——超等会平白拖慢流水。拆成 `empty`（私有 operand）+ `aempty`（共享 operand）后语义才干净。
3. **kernel 结尾要再 `cluster_sync()`**（`moe_cluster.cu:268`）——在撤销分布式 barrier 之前，保证所有 CTA 的 remote arrive 都已落地，否则会有偶发的 barrier 反构竞争。

> 这里还有个**不报错、只是死锁**的坑：cluster 的 launch 要求 `grid.x % CN == 0`，且 **`aempty` 的计数必须等于 `CN × 消费者 warp 数`**（不是 `CN × 线程数`，因为只有每个 warp 的 lane0 去 arrive）。计数写错不会崩，只会 `mbarrier.try_wait` 永远转圈。调试时用 `cuda-gdb` 或设备端 `printf` 定位卡在哪个 barrier。

---

## 3. 实测：multicast 只在「字节数真的成为瓶颈」时兑现

固定 `K=7168, N=3072, G=384, topk=6`，扫 token 数与 cluster 尺寸。每个配置都过 CPU 参考对拍（`max_abs_err/ref < 3%`）。

### 3.1 prefill（contiguous）

| tokens | 几何 | CN=1（无广播） | **CN=2** | CN=4 |
|---|---|---|---|---|
| 8192 | 128×128 s3 | **867.8** | 804.0 | 786.2 |
| 16384 | 128×128 s3 | **947.7** | 926.8 | 889.1 |
| 32768 | 128×128 s3 | 960.7 | **1001.6** | 811.2 |
| 32768 | 128×256 s3 | 989.3 | 964.4 | — |
| 32768 | 128×128 s4 | 777.2 | 769.3 | 726.0 |

（单位 TFLOPS/aligned，峰值 1978。）

几个规律：

- **只有 32768 + 128×128 + CN=2 赢**（+4.3%）。这时每 expert 有 4.5 个 m-tile，L2 放大最狠（A/B 各 ~24×/4.5×）。
- **CN=4 全线更慢**。cluster 越大，leader 要等的 peer 越多，流水耦合越强；32768 时 CN=4 掉到 811。
- **128×256 上 multicast 没用**：BN 变大后 A 的放大已经从 24× 降到 12×，剩下的大头是 B（没广播），所以广播 A 边际很小。
- **s4（更深流水）本来就慢**（smem 到 128KB 后 1 CTA/SM，见 25 篇），加了 multicast 也救不回。

### 3.2 decode（masked）：multicast 帮不上忙

| 配置 | CN=1 | CN=2 | CN=4 |
|---|---|---|---|
| masked 128×256 s3 | **174.2** | 167.5 | 158.4 |

TFLOPS/useful；B-read 都贴着 **2.7 TB/s（~82% HBM）**。decode 是**纯权重带宽**场景，瓶颈在 DRAM 读 B（8.45 GB），而 A 很小、multicast 省的是 L2→SM，对 DRAM 无益；多出来的 cluster 耦合只有负收益。

---

## 4. ncu：为什么省了 L2 字节，时间却没跟着降

对 32768 tokens 的 128×128 s3，CN=1 vs CN=2：

| 指标 | CN=1 | CN=2 | 变化 |
|---|---|---|---|
| `lts__t_sectors_op_read`（L2 读 sector） | 2.580e9 | **2.095e9** | **−18.8%** |
| `lts__throughput`（L2 吞吐占用） | 88.22% | 88.68% | ≈ 持平 |
| `sm__pipe_tensor_op_hmma_cycles_active`（tensor） | 65.05% | 64.99% | 持平 |
| `smsp__..._stalled_long_scoreboard` | 11.59 | **8.00** | **−31%** |
| `gpu__dram_throughput` | 53.05% | 56.89% | ↑ |
| CTA 驻留 | 27.77% | 27.82% | — |

16k 的对比更能说明问题：sector 从 `1.427e9 → 1.213e9（−15%）`，但 `lts__throughput` **从 94.08% 微升到 94.36%**，tensor 反而从 62.46% 掉到 59.69%。

读法：

- multicast **确实**把 L2 读字节砍了 15–19%，也把等 L2 的 `long_scoreboard` 拖慢从 11.6 降到 8.0（这是 32768 能赢的直接原因）。
- 但 **`lts__throughput` 几乎不动**。这个指标刻画的是 L2 子系统的**请求率/延迟占用**，不是纯字节带宽：multicast 少搬了数据，却没有减少 L2 需要处理的**请求数**（每个 CTA 仍要为自己的 A/B 发 TMA，只是 A 的字节被广播了），加上 leader 与 peer 的握手把访存和计算耦合得更紧，于是 L2 占用的"天花板"没动。
- 当 kernel 真正卡在**字节搬运**（大 batch、放大高）时，省字节就能兑现（+4.3%）；当它更多卡在**延迟/请求**（小 batch）时，multicast 的耦合开销反而盖过收益。

一句话：**"L2 利用率高" ≠ "减少 L2 字节就能提速"**。判断要不要上 multicast，得先看瓶颈是 L2 的**带宽**还是**请求/延迟**。

---

## 5. 弯路：用更大的 M-tile 消 B 放大，也不行

按公式 $T_{\text{L2}}=M_{\text{total}}NK(1/BN+1/BM)$，涨 BN 和 BM 都能降流量，而 `BM=256` 能把 B 的放大从 4.5× 砍到 2.2×。但实现起来有两个硬约束：

1. **contiguous 要求每组 M 对齐到 BM**，BM=256 就得把分布对齐到 256（padding 从 10.8% 涨到 19.6%）；
2. **寄存器**：BM=256 需要 4 个 warpgroup/544 线程，`128×256` 的累加器是每线程 128 个 fp32。

实测（ALIGN=256 的分布）：

| 配置 | TFLOPS/useful @32768 |
|---|---|
| 128×256 s3（参照） | 736.6 |
| 256×128 s3 c1 | 800.2 |
| **256×128 s4 c1** | **788.5** |
| 256×256 s2/s3 | 124–131（崩） |

`256×128` 比同分布的 `128×256` 好一点，但仍远不如 ALIGN=128 上的 `128×256`（~878 useful）。而 `256×256` 直接崩溃：ptxas 报

```
ptxas info : (C7511) wgmma.mma_async instructions are serialized due to
             insufficient register resources for the wgmma pipeline
```

——累加器 + 描述符把寄存器撑爆，wgmma 被强制串行化，只剩 **124 TFLOPS**。**更大的 tile 不是免费的**：它先在 padding 和寄存器上收税。

---

## 6. 对标：cuBLAS 在大 batch 会追上来

把 25 篇的 cuBLAS per-expert（CUDA graph 录 384 次 `_scaled_mm`）参照在 32768 tokens 上重跑（`cublas_moe_ref.py --tokens 32768`）：

| 实现 | ms | TFLOPS/aligned | 相对 |
|---|---|---|---|
| cuBLAS eager loop | 9.6186 | 1017.4 | 基线 |
| cuBLAS **graph** loop | 9.5121 | 1028.8 | 1.01× |
| 本篇 grouped + multicast（CN=2） | **9.6406** | 1001.6 | 0.97× |
| 本篇 grouped（CN=1） | 10.0515 | 960.7 | 0.93× |

- **16384 tokens** 时 grouped 仍稳定领先 cuBLAS graph（`5.66 vs 6.54 ms`，**1.16×**），因为那时每个 expert 只有 ~256 行，cuBLAS 的 384 个小 GEMM 并行度不足；
- **32768 tokens** 时每个 expert ~512 行、单 GEMM 已有 96 个 CTA，cuBLAS graph 的 per-expert loop 基本吃满，追平并略超 grouped（1028.8 vs 1001.6/1015.5）。也就是说：**grouped 的护城河在中小 batch**，大 batch 时单个 GEMM 本身就够大，per-expert 循环不再吃亏。
- 与 DeepGEMM 的对比仍受容器缺 `libdwfl` 阻塞（见 ROADMAP「阻塞」）。

---

## 7. 复现

```bash
cd code/kernel-opt
scripts/lab.sh up

# 编译 + 全量扫描（CN=1/2/4 × 几何 × tokens）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 26-moe-cluster-multicast/moe_cluster.cu all 32768

# 单跑一个配置（mode ∈ loop | grp… | masked | bigm | all）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 26-moe-cluster-multicast/moe_cluster.cu grp128x128s3_c2 32768

# ncu（注意 cluster kernel 也要在 kernel_lab 里跑）
scripts/ncu.sh 26-moe-cluster-multicast/moe_cluster.cu \
  --kernel-name regex:moe_kernel --launch-count 1 \
  --metrics lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_read.sum \
  -- grp128x128s3_c2 32768
```

代码结构：

- `moe_cluster.cu`：wgmma/TMA/mbarrier + **cluster multicast** helpers（`tma_load_2d_mcast`、`mbar_arrive_cluster`、`cluster_sync`）+ `moe_kernel<BM,BN,BK,STAGES,MASKED,CN>`；
- `moe_cluster_launch.h`：`make_dist`、tensor map、cluster launch、CPU 对拍、CN/几何/tokens 扫描、BM=256 对照；
- 原始输出：`moe_cluster_S{8192,16384,32768}.out.txt`、`ncu_c1_c2_16384.out.txt`、`ncu_c1_c2_32768.out.txt`。

---

## 8. 小结

- **TMA cluster multicast 的正确姿势**：`cudaLaunchKernelEx` + `clusterDim`；leader 发 `...multicast::cluster` + CTA mask；每个 CTA 各自 `arrive.expect_tx`，multicast 会把 A 的完成信号补到每个目标 CTA 的同偏移 full barrier；释放端用 `mapa.shared::cluster` 把 arrive 投到目标 CTA；私有/共享 operand 各用一套 empty barrier；init 后与退出前各一次 `cluster_sync`。
- **收益有明确边界**：只有 **32768 tokens + 128×128 + CN=2** 赢 **+4.3%**（960.7→1001.6），L2 读 sector −18.8%、`long_scoreboard` 11.6→8.0；8192/16384/masked 全部小幅变慢。
- **诊断教训**：`lts__throughput`（请求/延迟）和 `lts__t_sectors`（字节）是两个东西。multicast 砍字节，但 L2 利用率不降——**别默认"L2 高"就等于"省 L2 字节能提速"**。判据是：瓶颈在字节（大 batch、放大高）才值得上 multicast。
- **更大的 tile 不是免费的**：BM=256 先交 padding（10.8%→19.6%）和寄存器税；256×256 触发 ptxas `C7511`，wgmma 串行化崩到 **124 TFLOPS**。
- **对标**：中小 batch grouped 领先 cuBLAS graph **1.16×**（16384）；大 batch cuBLAS graph 追平（32768，~1010–1029）。
- 坑：cluster 的 `aempty` 计数是 `CN × 消费者 **warp** 数`（不是线程数）；`grid.x % CN != 0` 无法启动；这两个都是「不报错、只死锁」。

**下一篇预告**：B 侧 multicast（沿 M 的 cluster + DeepGEMM 式 `is_peer_cta_alive` 动态回退），配上 per-block FP8 缩放；或者转向 **DSA compressor**（ratio ∈ {4,128,0}）把 DeepSeek-V4 稀疏注意力那条线收尾。
