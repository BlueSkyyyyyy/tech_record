---
title: "CUDA 算子调优（五十八）：FP4 decode GEMV 的第二轮 —— 从 34.8% 到 38.7%，以及为什么 split-K / 无 smem / 位运算解码都救不了它"
date: 2026-09-22T09:00:00+08:00
draft: false
weight: 58
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "FP4", "e2m1", "E8M0", "DeepSeek-V4", "GEMV", "dp4a", "反量化", "split-K", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 56 篇]({{< relref "cuda-kernel-opt-56-v4-fp4-moe" >}}) 钉死了一件事：DeepSeek-V4-Pro 的**路由专家
权重其实是 FP4 e2m1 + E8M0 block-32**；并且写了一个读真实 FP4 权重的 decode GEMV——算术解码
（逐 nibble 的移位/掩码/I2F）是 **compute-bound**（ncu Compute 65.5%、DRAM 19.6%），换成
**256 项 `uint16` smem LUT（1 个输入 byte → 2 个 int8）** 后用 `__dp4a` 做整数点积，拿到

```text
[56 篇] w1 [N=3072, K=7168], M=1：0.0100 ms / 1165.6 GB/s / 34.8% HBM
        纯读 roof（fp4+scale）：0.0041 ms / 2885.8 GB/s / 86.1%
```

56 篇把下一堵墙归给了 **LUT 的 shared bank conflict**（59% 多余 wavefront、`short_scoreboard` 4.54、
`long_scoreboard` 5.96），路线图里给它留的任务是「**免冲突 LUT 布局 / 直接上 IMMA，把 FP4 decode
从 35% 推向 70%+**」。

这一篇就是去做这件事。结论先放这：

> **没推到 70%。** 真实墙不是 bank conflict，而是**这一层的并行度上限**：`w1` 只有
> `N=3072` 行，一行一个 warp 就只有 **3072 个 warp**（23 warp/SM，`Waves Per SM = 0.48`，
> occupancy 32.8%），kernel 是彻底的**延迟受限**（`Issued Ipc Active` 只有 1.08、Issue Slots
> Busy 20.9%）。把 bank conflict 去掉（寄存器表 + `__byte_perm` 位运算解码）、把并行度拉高
> （split-K）、把 MLP 拉满（全量预取）、把 smem 和 `__syncthreads` 全删掉——**全部失败**。
> 唯一有效的是最朴素的一招：**把权重的预取从「提前 1 个 chunk」改成「提前 2 个」**，
> `0.0100 → 0.0090 ms`，**34.8% → 38.7%**（1.11×）。距 roof 仍有 **2.2×**，而且这个 gap
> 几乎全是「N 太小」造成的，不是代码能补的。

这一篇把一个**小算子做不快时的诊断方法**讲清楚：先看 `Waves Per SM` 和 `Issued Ipc`，
别一上来就顺着 ncu 的「Est. Speedup」去抠 bank conflict。

---

## 一、shape 与目标

shape 还是从真实 checkpoint 来（`/ssd/models/DeepSeek-V4-Pro`，layer 0 / expert 0 的 `w1`）：

| 项 | 值 |
|---|---|
| `w1` | `[N=3072, K=7168]`，routed expert 的 up 投影 |
| 权重存储 | **FP4 e2m1**，`[3072, 3584]` uint8（2 nibble/byte）|
| block scale | **E8M0**，`[3072, 224]` uint8（每 32 个 K 一个，$2^{b-127}$）|
| 权重字节 | fp4 = 11.01 MB（bf16 的 25.0%、fp8 的 50%）|
| 激活 | int8（per-128 动态量化），`[M, 7168]` |

它对应 **MoE decode 路径**：M 个 token（M=1 是纯 decode），每个 token 过 384 个专家的 top-6。
`N=3072` 是 `moe_intermediate_size`，是这一层**天然的并行度上限**。

计算：$D[m,n]=\sum_k A[m,k]\cdot W[n,k]$，`W` 的 FP4 先解码成整数（$2\times$e2m1 的值
$\{0,1,2,3,4,6,8,12\}$，正好落进 int8），激活也是 int8，用 `__dp4a` 一次算 4 个，
最后乘回激活 scale 与 E8M0 权重 scale（$0.5\cdot s_w\cdot s_a$，其中 $0.5$ 是
「$2\times$e2m1」的还原）。

目标口径：**权重字节 = FP4 原始字节 + scale 字节**，效率按 HBM 3.35 TB/s 算。

---

## 二、先诊断：`Waves Per SM = 0.48` 才是真墙

56 篇的 kernel 是「一个 warp 处理一行，warp 内 32 lane 沿 K 并行（每 lane 一个 16B chunk =
32 个 fp4）」。先把它的 ncu 关键项摆出来（`RWW=1, BN=8`，`grid = 3072/8 = 384` 个 CTA）：

```text
Waves Per SM                    0.36 ~ 0.48     ← 网格根本填不满 GPU
Achieved Occupancy              32.8%           ← 只有 21 warp/SM，上限 64
Issued Ipc Active               1.08            ← 每个 scheduler 每 cycle 才发 1 条
Issue Slots Busy                20.9%           ← 发射口 4/5 空着
DRAM Throughput                 36.9%
L1/TEX Cache Throughput         69.5%           ← 最高的那一级
Executed Instructions           2.43 M（warp 级）
```

`Waves Per SM < 1` 说明：**这个网格连一轮都填不满 SM**。把「一个 warp 一行」的约束写出来就明白了：

$$
\#\text{warp} = \frac{N}{\text{RWW}} = \frac{3072}{1} = 3072,\qquad
\text{warp/SM} = \frac{3072}{132} \approx 23
$$

而 H100 每个 SM 能驻留 64 个 warp。**并行度只有硬件容量的三分之一**，再好的指令效率也藏不住
600 ns 的 HBM 延迟。这一条直接决定了后面所有尝试的成败。

> **判据**：一个 memory-bound kernel 先看 `Waves Per SM` 与 `Achieved Occupancy`。
> 如果 `Waves < 1`、`Issued Ipc < 1.5`、`Issue Slots Busy < 30%`，而 `DRAM%` 又远低于 80%，
> 那就是**并行度不够**，此时 ncu 给的「Est. Speedup by fixing bank conflict」是陷阱——
> 去掉 bank conflict 也填不满 SM。

---

## 三、四个尝试，一张结果表

所有实验都在同一个 `.cu`、同一进程、同 shape 下测（`bench_ms(…, 5, 200)`），
对拍 CPU 参考（真实 FP4 值 × int8 激活）相对误差 `9.4e-8`，全部 OK。

| 版本 | 做法 | 耗时 | FP4 权重带宽 | HBM | 结论 |
|---|---|---|---|---|---|
| 56 基线 | LUT decode，单 chunk 无预取 | 0.0100 ms | 1165.6 GB/s | 34.8% | — |
| **v2** | LUT + **提前 1 chunk 预取** | 0.0091 ms | 1288.5 GB/s | 38.4% | **有效** |
| **v5** | LUT + **提前 2 chunk 预取**（DEPTH=2） | **0.0090 ms** | **1296.6 GB/s** | **38.7%** | **最好** |
| v5 DEPTH=3/4 | 预取更深 | 0.0092 ms | ≈1270 GB/s | 38.0% | 寄存器反噬 |
| v5 RWW=2 | 一 warp 两行，复用激活 | 0.0098 ms | 1194.7 GB/s | 35.6% | 负 |
| v2 KSPLIT=2/4/8 | 沿 K 切，`atomicAdd` 归约 | 0.0094 / 0.0098 / 0.0118 | ↓ | 37→29% | **负** |
| v4 全量预取 | 7 个 chunk 全进寄存器再解码 | 0.0096 ms | 1216.6 GB/s | 36.3% | 负（寄存器压力） |
| v2 PRMT | 位运算解码（去 LUT） | 0.0117 ms | 1001.8 GB/s | 29.9% | **负** |
| v6 无 smem | 激活走 global + PRMT + KSPLIT | 0.0114 ms | 1029.8 GB/s | 30.7% | 负 |
| v2 算术解码 | `e2m1_i8` 逐 nibble 移位 | 0.0180 ms | 651.6 GB/s | 19.4% | 负（对照） |

一个反直觉的现象：**「最朴素」的预取加深赢了，所有「看起来更聪明」的结构改动都输了。**
下面逐个解释。

---

## 四、唯一有效的：软件流水的深度

56 的循环是「读一个 16B 权重 → 立刻解码 → `dp4a`」。读和用之间只有一个指令的距离，
每个 lane 在飞的内存请求最多 1 个，延迟完全暴露。改成深度 2 的环形预取：

```cpp
uint4 ring[2];
ring[0] = __ldcs(Wp + row*(K/2) + (lane + 0)*16);
ring[1] = __ldcs(Wp + row*(K/2) + (lane + 32)*16);
#pragma unroll
for (int i = 0; i < 7; ++i) {
    uint4 w = ring[i & 1];
    if (i + 2 < 7) ring[i & 1] = __ldcs(Wp + row*(K/2) + (lane + 32*(i+2))*16);
    /* 解码 + dp4a */
}
```

每个 lane 同时有 **2 个 16B 在飞**，`long_scoreboard` 从 5.43 降到 **5.05**，
时间 `0.0091 → 0.0090`。继续加深到 `DEPTH=3/4` 反而回到 0.0092——因为多出来的
`uint4` 寄存器（每个 4 个 reg）挤压了 occupancy，收益抵消。**DEPTH=2 是甜点。**

注意这个收益很小（~1%），因为延迟不是靠单个 warp 的 MLP 填的，而是靠 warp 数——见下。

---

## 五、为什么 split-K 是负的（本轮最重要的反直觉）

`Waves Per SM = 0.48` 最直接的解法显然是**沿 K 把每个 warp 的工作切开，多起几倍的 CTA**。
`N=3072` 行不动，把 `K/2 = 3584` 字节切成 `KS` 段，`grid = (N/8, KS)`：

```text
KS=1  384 CTA   0.0091 ms   38.4%   waves 0.48
KS=2  768 CTA   0.0094 ms   37.1%   waves 0.97
KS=4 1536 CTA   0.0098 ms   35.8%   waves 1.94
KS=8 3072 CTA   0.0118 ms   29.5%   waves 3.88
```

**越切越慢，且 HBM 效率单调下降。** 我把激活的 smem 装载也改成只读本 block 的 k-slice
（避免重复读整段激活），结果一样。原因有两层：

1. **每个 block 的固定开销被摊薄**：256 项 LUT 的初始化、一次 `__syncthreads`、
   激活装载的 prologue，在 `KS=8` 时每个 block 只剩约 1 个 chunk 的活，
   prologue 反而成了主体。
2. **更本质的：总 L1TEX 事务没有减少。** 切 K 只是把同样的全局读、同样的 LUT 读
   分给了更多 warp；而墙是 L1TEX（69.5%）+ 延迟。**多起 warp 不能减少工作量**，
   只能更好地重叠——但重叠的前提是 SM 有空闲发射口，而单 warp 的 issue 本来就低。

> 这就是「并行度不足」的两种可能：**要么是真的没活可干（这里不是，总活很多），
> 要么是活被「一行的原子性」绑住了**。GEMV 的每一行必须由同一个 warp 收口
> （`warp_sum` + 一个写出/原子加），切 K 就把「收口」拆成了多次归约，归约税
> 在这么小的 kernel 里非常显眼。

---

## 六、为什么「去掉 LUT」也救不了：位运算解码

56 的 LUT 是「1 个输入 byte（2 个 e2m1 nibble）→ 2 个 int8」。它每次要读 16 次 smem
（一个 16B 权重 = 4 个 uint32 = 16 个 byte），随机索引 → **59% 多余 wavefront**。
最直接的替代是**把解码搬进寄存器**，用 `__byte_perm` 做。

### 推导：e2m1 → int8（$2\times$ 值）

e2m1 的 8 个幅值码 $c=(e\ll1)|m$ 对应的 $2\times$值为
$f=[0,1,2,3,4,6,8,12]$（恰是 int8 范围内的整数）。符号位是 nibble 的 bit3，
e4m3 那套「负号 = 原码｜0x80」在这里换成 int8 的二进制补码。

用 `__byte_perm` 查表：它用 selector 的 4 个 nibble 从两个 32-bit 源（共 8 个 byte）里各选 1 个 byte。
把 $f$ 的 8 个 byte 放进 `0x03020100`（$f_0..f_3$）和 `0x0C080604`（$f_4..f_7$），
把 $-f$ 放进 `0xFDFEFF00` / `0xF4F8FAFC`，就有：

```cpp
// e：4 个 nibble 已摊成 byte（每个 byte ∈ [0,15]）
uint32_t k = e & 0x07070707;                 // 幅值码 0..7
uint32_t s = (e >> 3) & 0x01010101;          // 符号位
uint32_t pk = (k & 0xF) | ((k>>4)&0xF0) | ((k>>8)&0xF00) | ((k>>12)&0xF000);  // 压成 4 个 nibble
uint32_t ps = (s & 0x1) | ((s>>4)&0x10) | ((s>>8)&0x100) | ((s>>12)&0x1000);
uint32_t pos = __byte_perm(0x03020100, 0x0C080604, pk);   // f[code]
uint32_t neg = __byte_perm(0xFDFEFF00, 0xF4F8FAFC, pk);   // -f[code]
uint32_t sel = 0x3210 | (ps << 2);                        // 输出 byte i 选 i+4*sign
return __byte_perm(pos, neg, sel);
```

写成最小复现 `prmt_test.cu`，对全部 $2^{32}$ nibble 组合里的采样做了穷举，`bad = 0`。
解码**完全正确，也完全不用 smem**。但实测：

```text
LUT   0.0091 ms  38.4%   （有 bank conflict，但只有 1.5 条 ALU / nibble + 0.5 次 smem 读）
PRMT  0.0117 ms  29.9%   （无 smem，但要 ~3 条 ALU / nibble + 一个 11 条指令的打包）
```

**PRMT 比 LUT 慢 29%。** 根因：`__byte_perm` 的代价在 **ALU pipe**，而打包（把 4 个 byte 的低
3 位压进一个 selector 的 4 个 nibble）要 11 条指令。整体解码变成了 ALU-bound；
LUT 虽然撞 bank，但每次 lookup 的 ALU 极少，bank conflict 被 LSU/发射口吸收掉了。

> **判据**：ncu 报的 bank conflict 多，不代表它是瓶颈。把「smem 查表」换成「寄存器位运算」，
> 是把访存成本换成 ALU 成本——**只有当 ALU pipe 还有余量时才划算**。这里 ALU pipe 已经因为
> decode 而接近饱和，换过去就是负优化。（和 [第 44 篇]({{< relref "cuda-kernel-opt-44-w4a16-decode-gemv" >}})
> 「smem LUT 替 ALU」的结论正好互为镜像：方向对了才叫优化，方向反了就都是税。）

---

## 七、把 smem 彻底删掉也没用

既然 LUT 的 bank conflict 是墙、PRMT 的 ALU 又是墙，那我试了第三条：**连 smem 都不要**——
激活直接从 global 读（`Aq` 只有 7 KB，常驻 L2），PRMT 解码，再把 `split-K` 拉满，
让每个 block 只剩「读权重 → 解码 → `dp4a`」这一件事，没有任何 prologue / `__syncthreads`：

```text
v6 无 smem：RWW=1 KS=1   0.0114 ms / 30.7%
           RWW=1 KS=8   0.0127 ms / 27.6%
           RWW=4 KS=8   0.0120 ms / 29.1%
```

还是慢。**删掉 prologue 并没有换来速度**，因为 prologue 根本不是墙——墙是
「3072 个 warp 太少 + 单 warp 的 issue 发不出去」。

---

## 八、v5 的 ncu 画像与那 2.2× 的差距

最好版本（v5，LUT + DEPTH=2）的 ncu：

```text
Duration                        11.17 us
DRAM Throughput                 36.9%     ← 远没到 HBM 墙
L1/TEX Cache Throughput         69.5%     ← 最高的一级：LUT + 激活 smem
L2 Cache Throughput             36.6%
Compute (SM) Throughput         20.9%
Issued Ipc Active               1.08
Issue Slots Busy                20.9%
Waves Per SM                    0.36
Achieved Occupancy              32.8%
Registers Per Thread            32
stall long_scoreboard           5.05      ← 等权重 global load
stall short_scoreboard          4.69      ← 等 LUT/激活 smem
stall mio_throttle              3.79
stall wait                      1.18
```

一幅典型的**延迟受限**画像：`DRAM` 只有 37%、`Compute` 21%、`L1/TEX` 69.5% 但没有到顶，
IPC 1.08。时间花在「等」上，而不是任何一级吞吐上。

那离 roof 的 2.2× 从哪来？roof kernel（`read_fp4_kernel`，1024 block × 256 thread 的
grid-stride 纯读）用的是 **26 万个线程**，而 GEMV 只有 **9.8 万个**（3072 warp）。
把两边的「在飞字节数」算一下：

$$
\text{in-flight} \approx \#\text{warp}\times 32 \times \text{DEPTH}\times 16\,\text{B}
= 3072\times 32 \times 2 \times 16 = 3.1\,\text{MB}
$$

而要打满 3.35 TB/s、延迟按 600 ns 算，需要 $3.35\times10^{12}\times 6\times10^{-7}\approx 2.0$ MB。
理论上 3.1 MB 够了——**但这是「全部 warp 同时都在飞」的理想值**。实际每个 warp 只有
7 次迭代，prologue/epilogue 占了大头；而且 `L1/TEX` 的 LUT 事务（69.5%）和全局读
抢同一个 LSU，能同时"在飞"的有效字节远低于 3.1 MB。**这就是 `Waves < 1` 的代价。**

---

## 九、工作点：M 变大时会发生什么

同一份 kernel 扫 M（激活复用同一份权重读）：

| M | 最好配置 | 耗时 | 带宽 | HBM |
|---|---|---|---|---|
| 1 | v5 LUT DEPTH=2 | 0.0090 ms | 1296.6 GB/s | 38.7% |
| 2 | v2 LUT KS=1 | 0.0108 ms | 1085.8 GB/s | 32.4% |
| 4 | v2 **PRMT** KS=1 | 0.0133 ms | 877.5 GB/s | 26.2% |

两个值得记的观察：

- **M 越大，单次权重读被摊到越多 token，但时间在涨**（0.0090→0.0108→0.0133）。
  因为激活的 `dp4a` 次数 $\propto M$，而权重读不变——kernel 从「等权重」变成「算激活」。
  **M=1 是带宽最优的工作点；M 一大就该换回 GEMM（张量核）**，呼应
  [第 46 篇]({{< relref "cuda-kernel-opt-46-w4a16-smallm-dispatch" >}}) 的分派表。
- **M=4 时 PRMT 反超 LUT**（0.0133 < 0.0140）：解码只做一次、被 4 个 token 复用，
  此时 LUT 的 smem 事务成了相对更重的负担，位运算解码翻盘。**「LUT 还是 PRMT」取决于
  解码结果被复用几次（MT）**，而不是绝对值谁快。

---

## 十、这一层到底该怎么快

把结论从「kernel 调优」抬到「系统设计」：

1. **decode 用 FP4 存是对的**（权重字节减半，见 56 篇），但 **M=1 的单专家 GEMV 天生是
   延迟受限**——`N=3072` 的行数就那么多，一行一个 warp 只有 3072 个 warp。
2. **真正能变快的是把「一行一个 warp」变成「一个 warp 连续处理多行、并让多行共享激活读」，
   同时用 `RWW` 把权重读的在飞量翻倍**——但实测 `RWW=2` 在这里是负的（0.0098），
   因为寄存器翻倍又把 occupancy 压回去了。
3. **批量：真实的 decode 是 384 个专家的 top-6**。把「同一批 token 命中的专家」合并成
   一个 grouped GEMV（B 侧按 expert 折叠）才是正路——那样每份专家权重只读一次，
   且可以把多行/多专家塞进一个 CTA，把并行度做上去。这是 56 篇末尾留的
   「把解码摊到 prefill（权重被 $B_M$ token 复用）」的同一句话。
4. **prefill 不要现场折**：[第 57 篇]({{< relref "cuda-kernel-opt-57-v4-fp4-moe-fold" >}})
   已经证明现场折 + SW128 置换的「变换税」吃掉字节红利；prefill 走 host 预折叠成 FP8
   再喂 grouped GEMM，decode 才轮到现场读 FP4。

判决：**这个 kernel 的现实上限就在 ~40% HBM 附近**。它不是「再抠一个技巧就能翻倍」的算子，
而是「形状（N 小）决定了它的天花板」——能改变天花板的只有**批量/分组**，不是单点指令优化。

---

## 十一、实测对比表

| 版本 | 做法 | 0.0090 附近？ | 耗时 | 权重带宽 | HBM | 相对 56 |
|---|---|---|---|---|---|---|
| 56 基线 | LUT，无预取 | ✗ | 0.0100 ms | 1165.6 GB/s | 34.8% | 1.00× |
| **v5（最佳）** | **LUT + DEPTH=2 预取** | ✓ | **0.0090 ms** | **1296.6 GB/s** | **38.7%** | **1.11×** |
| v2 | LUT + DEPTH=1 预取 | ✓ | 0.0091 ms | 1288.5 GB/s | 38.4% | 1.10× |
| v5 RWW=2 | 两行/warp | ✗ | 0.0098 ms | 1194.7 GB/s | 35.6% | 0.98× |
| v4 全量预取 | 7 chunk 全进寄存器 | ✗ | 0.0096 ms | 1216.6 GB/s | 36.3% | 0.99× |
| v2 KSPLIT=2 | 沿 K 切 + 原子加 | ✗ | 0.0094 ms | 1244.3 GB/s | 37.1% | 0.97× |
| v6 无 smem + PRMT + KS=8 | 删 smem/sync | ✗ | 0.0127 ms | 924.5 GB/s | 27.6% | 0.79× |
| v2 PRMT | 位运算解码 | ✗ | 0.0117 ms | 1001.8 GB/s | 29.9% | 0.85× |
| v2 算术解码 | 逐 nibble 移位 | ✗ | 0.0180 ms | 651.6 GB/s | 19.4% | 0.56× |
| roof | 纯读 fp4+scale | — | 0.0041 ms | 2885.8 GB/s | 86.1% | 4.4× 上界 |

所有版本对拍 CPU 参考（真实 FP4 值 × 量化后激活），`max_rel = 9.4e-8`，`OK`。

---

## 十二、踩坑

1. **`Waves Per SM < 1` 时不要相信「Est. Speedup」**：ncu 会把 `short_scoreboard`/bank
   conflict 标成主因，但真正的墙是 SM 没填满。先看 `Waves`、`Issued Ipc`、`Issue Slots Busy`。
2. **`__byte_perm` 的 selector 用的是「每个 byte 的低 4 位按 nibble 排布」**：selector
   第 $i$ 个 nibble（第 $4i$ 位起）决定输出 byte $i$。把 4 个 byte 的低 3 位压成 selector 时，
   我用 `(k & 0xF) | ((k>>4)&0xF0) | …` 才对；直接拿展开后的 byte 当 selector 会全错。
   最小复现必须**穷举**而不是抽几个点（`prmt_test.cu` 扫 `0..2^28`）。
3. **`__byte_perm` 的位运算解码是 ALU 密集**：推导正确、无 smem，但在 ALU 已被 decode
   占满的 kernel 里是负优化。别把它当「无脑替代 smem LUT」。
4. **模板参数别传运行时变量**：`gemv<RWW, MT, H, …>` 里 `RWW`/`MT` 必须是编译期常量，
   lambda 捕获的 `rww` 会报 `the value of parameter cannot be used as a constant`；
   分派要用嵌套宏把 literal 展开（本轮编译期踩了两次）。
5. **`.bin` 靠 `extract.py` 生成**（`56-v4-fp4-moe/extract.py`，读真实 safetensors），
   本目录用软链复用；跑之前确认软链目标在。

---

## 十三、小结与下一篇

- 56 篇的 FP4 GEMV 从 **34.8%（0.0100 ms）** 推进到 **38.7%（0.0090 ms）**，靠的是
  **把软件流水的深度从 1 调到 2**，仅此而已。
- **split-K、无 smem、位运算解码、全量预取、RWW** 全部是负结果；根因是
  `N=3072` 决定的 warp 数上限（`Waves 0.48`）——**延迟墙不是指令效率能补的**。
- 这层算子要真正提速只能改**并行粒度**：把 top-6 的多个专家合并成 grouped GEMV，
  让一个 CTA 吃多个 (token, expert) 对，把并行度从「3072 行」抬到「token×expert」量级。
- 下一篇候选：**把 FP4 decode 接成 grouped GEMV（多专家共享权重读 + 更高并行度）**，
  或回到 [第 57 篇]({{< relref "cuda-kernel-opt-57-v4-fp4-moe-fold" >}}) 留的
  **host 预排布 FP4 + chunk-local 折叠，兑现 prefill 的 1.26× 地板**。
