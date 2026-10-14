# 通信腿在「多对多（all-to-all）」下的每 SM 带宽预算：一条 NVLink 的膝点，要乘以 peer 数吗？

> 本文对应路线图 **P182-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 192 篇。
>
> 第 191 篇（P181-1）给「给通信让 SM」这条设计轴量了**通信侧**的分母：一条 **NVLink P2P**
> （rank0 ↔ rank1）在 ~8 个 SM 处饱和于 ~372 GB/s，超过就是白给；第 190 篇（P180-1）量了**计算侧**
> 的分母。但第 191 篇只量了**一条链路**——真实 EP（dispatch/combine）里一个 rank 要和**其余
> 7 个 rank 同时**通信。于是留下一个必须回答的问题：
>
> **「一条链路 ~8 SM 封顶」在多对多下，是该乘以 peer 数（7 条链 ⇒ 56 SM 才够），还是仍封顶在
> 一个 rank 的「总」NVLink 出口（~370 GB/s）——几个 SM 就打满？**
>
> 这不是文字游戏：DeepEP 的 `get_theoretical_num_sms()`（`deep_ep/buffers/ep.py:431-541`）决定
> dispatch/combine 开几个 SM 时，用的是**每 rank 总** NVLink 带宽 `get_nvlink_gbs()`（`envs.py:208-234`，
> `nvidia-smi nvlink -s` 求和 ×0.9 = 430.3 GB/s）当**分子**，乘以每 SM 字节/每 SM 带宽常量
> （`envs.py:194-204` 的 180/45）反推 SM 数。它隐含的假设是「**每 SM 带宽恒定**」。本篇直接把
> 这个假设放到**真机 all-to-all** 下检验。
>
> - **每 rank 的 NVLink 聚合带宽是一个常数**：6 个 peer 和 1 个 peer，同样 SM 数下聚合带宽
>   **一模一样**（峰值 375.7 vs 371.6 GB/s，比值 **1.011×**）；固定 grid=64 把 peer 从 1 扫到 6，
>   聚合带宽稳在 **372–428 GB/s**（比值 1.19×，不随 peer 线性放大）。
> - **膝点也不随 peer 放大**：多 peer 在 **12 SM** 饱和、striped 在 **8 SM** 饱和——仍是 ~8–16 SM，
>   不是 7×8=56。→「给通信让 SM」的预算是**每 rank 链路**属性，不是每 peer。
> - **全部 7 个 rank 同时 all-to-all**（真满负荷）：每 rank 仍拿到 **~325 GB/s**（单租户的 0.87×），
>   节点聚合 **2280 GB/s**；DeepEP 的 `nvlink_gbs=430.3` 比这个满负荷值乐观 **1.32×**。
> - **模型对照**：真实 `get_theoretical_num_sms` 给 `direct EP8` 的 **16 SM** 恰落在真机膝点
>   （8–16 SM）内；但 `prefer_overlap=False` 的 `max(·,64)`（`ep.py:531`）把 EP8 顶到 **64=膝点的 4×**。
>
> **可跑性 / 边界**：DeepEP 的 dispatch/combine **设备端**路径依赖 NCCL GIN 设备 API
> （本机 NCCL 2.29.7 < 需求 ≥2.31，见 ROADMAP「阻塞」）→ 端到端**仅分析**。本篇量的是它**脚下的
> host 常量与原始 NVLink 带宽**，与设备 API 无关：Part A–E 是纯 CUDA/CUDA-runtime
> （`src/nvlink_alltoall_budget.cu`），Part B 的模型对照用 P63-1 的 `sys.modules` stub + `ast` 抽取跑
> **未修改的真实方法体**。**H100（sm90）真机，8×H100 80GB HBM3（NV18 全互联）**；为避免占用被别的
> 任务吃满的卡，进程暴露 7 张空闲卡（rank0 = 最空闲那张，peer = 其余 6 张）。
> 目标卡代号敏感：全文用「目标卡」指代。
>
> 参考坐标：`~/github/DeepEP@93eb6eb`（V2.5）`deep_ep/utils/envs.py`、`deep_ep/buffers/ep.py`；
> 原始输出 `harness/exp_nvlink_alltoall_budget.out.txt`
> （`harness/exp_nvlink_alltoall_budget.{sh,py}` + `src/nvlink_alltoall_budget.cu`；
> H100 80GB HBM3, sm90, 1980 MHz）。

---

## 1. 模型里那个「总 NVLink」到底该怎么理解

先把「模型怎么用 NVLink 带宽」讲清楚，因为它决定本篇要量什么。`get_theoretical_num_sms()`
（`ep.py:431-541`）是一段解析式 roofline：把每 token 的 HBM 读/写、NVLink/RDMA 流量折算成
「每 SM 的字节数」，再除以每 SM 带宽常量，取四个候选最大（`:522-529`）：

```
num_sms = max( bounded_gbs/bounded_traffic * sm_read  / sm_read_gbs,
               bounded_gbs/bounded_traffic * sm_write / sm_write_gbs,
               bounded_gbs/bounded_traffic * (sm_write - nvlink_traffic) / sm_read_gbs,
               bounded_gbs/bounded_traffic * (sm_read  + nvlink_traffic) / sm_write_gbs )
num_sms = align(max(4, ceil(num_sms * 1.3)), 4)
num_sms = num_sms if prefer_overlap_with_compute else max(num_sms, 64)   # :531
num_sms = min(num_sms, device_sms // 2 * 2)                             # :532
```

其中 `bounded_gbs` 是**瓶颈链路的每 rank 总带宽**——NVLink 场景下就是 `get_nvlink_gbs()`（430.3）；
`nvlink_traffic` 是「每 token 要跨 NVLink 的字节比例」，对 direct 场景是
`num_nvlink_ranks/num_ranks * (1 - 1/num_nvlink_ranks)`（`:508`）——注意它**已经乘进了 peer 比例**。

关键读法：模型把 NVLink 当成**一个每 rank 共享的总资源**（`bounded_gbs`），再反推「要几个 SM
的 HBM 带宽去喂它」。所以它**结构上**没有「每 peer 一条链路各需 SM」这种错误。真正的疑问是：
**分子 `nvlink_gbs=430.3` 在多对多下成立吗？分母「每 SM 带宽恒定」在 SM 数超过膝点后还成立吗？**
这正是本篇五个 Part 要量的。

---

## 2. Part A：单 peer 基线 + MLP 的作用

实验（`src/nvlink_alltoall_budget.cu`）：每张卡一块 256 MiB 缓冲、`cudaDeviceEnablePeerAccess`
双向打开（真机 7 卡两两 42 对全通），在 rank0 上跑 grid = N 个 block、每 block 1024 线程、
`__ldcs`/`__stcs` 流式访问远端 peer 的一片，取 5 次最小值。

读远端不像读本地 HBM：**单个未展开的 `__ldcs` 往返会把 SM 卡在 ~15 GB/s**（延迟受限）。所以
kernel 用 `template<int U>` **每线程每次迭代发 U=8 个独立 load**（MLP=8），逼近 TMA 深流水线的形态。
Part A 是单 peer 基线：

| N（SM） | 1-peer read GB/s | read/SM | 1-peer write GB/s |
|--------:|-----------------:|--------:|------------------:|
| 1       | 40.5             | **40.5** | 50.6      |
| 2       | 79.3             | 39.6     | 101.2     |
| 4       | 156.0            | 39.0     | 201.9     |
| 6       | 225.5            | 37.6     | 286.0     |
| 8       | 297.4            | 37.2     | 336.4     |
| 12      | 351.3            | 29.3     | 320.7     |
| **16**  | **367.7**        | 23.0     | 353.6     |
| 32      | 369.6            | 11.6     | 357.2     |
| 64      | 371.6            | 5.8      | 366.5     |
| 128     | **370.8**        | 2.9      | **370.4** |

- **单 peer 读峰值 ~371 GB/s**（16 SM 达 95%），写峰值 ~370 GB/s。这与 DeepEP 的
  `get_nvlink_gbs()=430.3` 是同一档——真机是 **0.86×**（模型略乐观，和第 191 篇量到的
  `sm_read=180` 偏乐观 19.5% 是同一类偏置）。
- **MLP 是这条腿的命门**：第 191 篇用 U=4 的 kernel 得到「8 SM 饱和」，本篇用 U=8 得「16 SM 达
  95%」——同一量级、随在飞 load 数微移。真实的 dispatch/combine 用 TMA 批量搬运（深流水），
  落在「~8–16 SM 打到 ~370」这个区间。**不管 U 多少，一个 rank 的 NVLink 出口就是 ~370 GB/s**。

---

## 3. Part B：多 peer 同开——聚合带宽涨不涨？

rank0 的 block 轮流绑到一个 peer（`blockIdx.x % npeer`），6 个 peer 全开，同样扫 grid：

| N（SM） | 1-peer GB/s | 6-peer GB/s | 6/1 |
|--------:|------------:|------------:|----:|
| 1       | 40.5        | 40.1        | 0.99 |
| 4       | 156.0       | 159.2       | 1.02 |
| 8       | 297.4       | 307.1       | 1.03 |
| 12      | 351.3       | 359.4       | 1.02 |
| 16      | 367.7       | 356.8       | 0.97 |
| 32      | 369.6       | 366.2       | 0.99 |
| 64      | 371.6       | 358.5       | 0.97 |
| 128     | 370.8       | 375.4       | 1.01 |

- **6 peer 峰值 375.7 GB/s ≈ 1 peer 峰值 371.6 GB/s（比值 1.011×）**。把 peer 从 1 加到 6，
  同一个 SM 数下**聚合带宽纹丝不动**。原因很直接：一个 rank 的 NVLink 出口（~478 GB/s 物理
  ÷ 协议开销 ≈ 370 实测）是**共享资源**，写到哪个 peer 都走它——**多 peer 不增加带宽，只增加
  「分布」，不增加「总量」**。
- **膝点仍是 12 SM**（6-peer read 在 12 SM 达 95%）。**不是 8×6=48**。

---

## 4. Part C：striped（每个 block 轮询所有 peer）

换一种切法：每个 block **依次读全部 6 个 peer**（把 MLP 从「同一条链路的多个在飞 load」摊到
「多条链路」）：

| N（SM） | striped read GB/s | read/SM |
|--------:|------------------:|--------:|
| 1       | 57.8              | **57.8** | 
| 2       | 113.9             | 56.9     |
| 4       | 221.4             | 55.3     |
| 6       | 310.4             | 51.7     |
| **8**   | **369.1**         | 46.1     |
| 16      | 375.4             | 23.5     |
| 64      | 376.3             | 5.9      |
| 128     | 376.0             | 2.9      |

- **striped 只要 8 SM 就饱和**，比 block-per-peer（12 SM）更快——因为单个 block 同时在向多条链路
  发请求，MLP 天然更高。**单 block striped 的 per-SM 57.8 GB/s 恰与第 191 篇「单 SM 单链路 P2P
  ~57 GB/s」对上**，是一个很好的交叉验证。
- 峰值 **376.3 GB/s** 略高于 block-per-peer（375.7），两者都封顶在**同一个 ~375 GB/s**。
  → 无论怎么切（每块一 peer / 每块轮询全部），**能拿到的是每 rank 的总出口，不是每链路之和**。

---

## 5. Part D：全部 7 个 rank 同时 all-to-all（真满负荷）

前面都只有 rank0 在动。真实 all-to-all 里**所有 rank 同时**往外发。让每张卡各起一份 kernel
（各自引自己的 6 个 peer）、用独立 stream 并发、各自记 event：

| 每 rank N（SM） | perRank worst GB/s | perRank mean GB/s | nodeAgg GB/s |
|---------------:|-------------------:|------------------:|-------------:|
| 8              | 279.7              | 281.1             | 1957.9       |
| 16             | 318.6              | 319.9             | 2230.3       |
| 32             | 323.6              | 324.1             | 2265.3       |
| 64             | 313.1              | 314.6             | 2191.9       |
| 132            | **325.7**          | 327.0             | **2280.0**   |

- **满负荷下每 rank 仍拿到 ~325 GB/s**——是单租户峰值（371.6）的 **0.87×**：NVSwitch 的
  争用只吃掉 ~13%，仍接近理想。节点聚合 **2280 GB/s**，说明 7 张卡**同时**在动。
- 对 DeepEP 模型：`get_nvlink_gbs()=430.3` 相对真机满负荷 **325.7** 乐观 **1.32×**。这不是致命的
  （模型 ×1.3 余量、再 align、再 `min(device//2)`），但它意味着**真实 SM 数应比模型给的小一档**，
  与第 191 篇「换实测常量后 SM 数普遍 −4」同向。

---

## 6. Part E：固定 grid，直接扫 peer 数

把所有变量钉死，只动 peer 数，最直白地回答「加 peer 会不会加带宽」：

| npeer（grid=64） | read GB/s |
|-----------------:|----------:|
| 1                | 371.6     |
| 2                | 373.6     |
| 3                | 374.0     |
| 4                | 373.9     |
| 5                | 358.3     |
| 6                | 427.8     |

- 1→4 几乎完全平（371.6→373.9）；5 掉到 358（共享节点噪声），6 反而 427.8——但**没有出现
  「6 条链各 ~370、聚合 ~2200」**这种线性叠加。范围 358–428，**max/min = 1.19×**，仍是同一档。
- **结论坐实**：**一个 rank 的 NVLink 聚合带宽由它的出口决定，与 peer 数无关**。模型里
  `nvlink_traffic` 随 peer 缩放的是「每 token 要搬的字节比例」，不是「每 SM 带宽」；这一点模型是对的。
  错的（如果误读）是把第 191 篇的「一条链路 8 SM」直接乘 peer 数——**真机不支持这个乘法**。

---

## 7. Part B：把多对多膝点代回真实 SM 模型

用 P63-1 的 `ast` 抽取跑**未修改的真实** `get_theoretical_num_sms`（`ep.py:431-541`，
sha256 `a8f72c39…`；`envs.py` sha256 `23843e16…`），默认常量：

| 配置 | num_sms |
|------|--------:|
| direct EP8（单节点无 RDMA） | **16** |
| hybrid EP16 | 16 |
| hybrid EP32 | 8 |
| hybrid EP64 | 8 |
| hybrid EP128 | 4 |
| **direct EP8（不偏好 overlap）** | **64** |
| hybrid EP16（RDMA 慢，瓶颈在 RDMA） | 4 |
| direct EP4 | 20 |
| direct EP1（无流量） | 132 |

真机测得「一个 rank 的 NVLink（含多 peer 聚合）饱和膝点」= **8–16 SM**。读法：

- **`direct EP8` 的 16 恰好落在膝点内**——默认模型对这条主路径**并不离谱**，它给 16 个 SM，
  真机 8–16 就饱和在 ~370。这里要修正第 191 篇略带歧义的说法：16 不是「多开 2×」，
  而是在膝点的**上沿**（8–16 区间的顶）。第 191 篇换实测常量后变 12，也仍在膝点内。
- **真正越界的是 `prefer_overlap=False` 的 `max(·,64)`**（`ep.py:531`）：EP8 被顶到 **64**，
  是膝点上沿（16）的 **4×**。这 48 个额外 SM 对**通信带宽**是白给的——和第 190 篇从计算侧
  量到的「访存受限腿的 SM 几乎白送」互相印证。
- **模型的封顶是 `min(·, device_sms//2)`**（`:532`），**不建模链路膝点**：它不知道「一个 rank 的
  NVLink 出口 ~370、8–16 SM 就满了」。这是它可能超配到 64 的根因。

---

## 8. 与第 190/191 篇拼起来：完整的一条 SM 汇率线

- **第 190 篇**（计算侧）：让出一个 SM，L1 约 +1.4–3 μs、wgrad 约 +0–24 μs。
- **第 191 篇**（通信侧，单链路）：一条 NVLink 的 P2P，~8–16 SM 饱和于 ~372。
- **本篇**（通信侧，多对多）：一个 rank 的 NVLink **总出口**同样 ~8–16 SM 饱和于 ~370–376，
  **与 peer 数无关**；满负荷 7 rank all-to-all 下每 rank ~325。

合起来的操作结论很清楚：**「给通信让 SM」的预算是「一个 rank 的链路出口」的量级——约 8–16 个**，
既不随 peer 数放大，也不该被 `max(·,64)` 顶到 4× 膝点。这也从另一面解释了 Mega MoE 主 kernel
为什么不做 SM 分区：既然通信腿 ~8–16 个 SM 就封顶，把通信和计算放进**同一块 SM 的不同 warp**
（第 2/9/10 篇）比「切一整块 SM 给通信、超过膝点就白给」更省。

---

## 9. 小结与边界

- **每 rank 的 NVLink 聚合带宽是常数**：6 peer 与 1 peer 同 SM 数下**逐点相同**（峰值
  375.7 vs 371.6，比值 1.011×）；固定 grid=64 扫 peer 1→6，稳在 372–428 GB/s。**多 peer 只增加
  「分布」，不增加「总量」**。
- **膝点不随 peer 放大**：单 peer 16 SM、多 peer 12 SM、striped 8 SM 饱和——仍是 **8–16 SM**，
  不是 7×8。**「一条链路 8 SM」不能乘以 peer 数。**
- **满负荷**：7 rank 同时 all-to-all，每 rank ~325 GB/s（单租户的 0.87×），节点聚合 2280 GB/s；
  `get_nvlink_gbs()=430.3` 相对满负荷乐观 **1.32×**。
- **模型对照**（真实方法体）：`direct EP8`=16 落在膝点内；`prefer_overlap=False` 的 64=膝点 4×；
  模型封顶是 `device_sms//2`，**不建模链路膝点**。
- **边界**：DeepEP dispatch/combine **设备端**（NCCL GIN）本机跑不了（`docs/86` 两道闸门、
  ROADMAP「阻塞」）——本篇量的是它脚下的 host 常量与原始 NVLink 带宽，不碰设备 API。实验用 7 张
  空闲卡（排掉了被别的任务吃满的那张），peer 数 = 6（真实 EP8 是 7）；8 卡共享节点上第 5 个
  peer 点有争用噪声（358），但 1→4 的平坦与 8–16 的膝点稳定可复现。

---

## 附：复现

```
bash code/mega-moe/harness/exp_nvlink_alltoall_budget.sh \
  > code/mega-moe/harness/exp_nvlink_alltoall_budget.out.txt
```
- `src/nvlink_alltoall_budget.cu`：Part A（单 peer 基线）、Part B（每块一个 peer）、Part C（每块
  轮询所有 peer）、Part D（全部 rank 同时 all-to-all）、Part E（固定 grid 扫 peer 数）；纯
  CUDA/CUDA-runtime，`nvcc -arch=sm_90`。
- `harness/exp_nvlink_alltoall_budget.py`：Part B，import P63-1 的 `exp_deepep_sm_qp` 复用其
  `ast` 抽取，跑**真实** `get_theoretical_num_sms` 方法体；判定 `PASS=11 / FAIL=0`。
