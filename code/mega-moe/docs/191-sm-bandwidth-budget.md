# 通信腿的「每 SM 带宽」预算：DeepEP 自动 SM 模型里被钉死的 180 / 45 到底对不对？

> 本文对应路线图 **P181-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 191 篇。
>
> 第 190 篇（P180-1）把「按 SM 分区藏通信」这条设计轴的**计算侧量出了分母**：给通信让出一个 SM，
> 计算腿要付多少（grouped GEMM 的 SM 预算曲线）。但它只回答了账的一半。另一半是——**给通信让出的
> 这些 SM，换回来多少带宽？几个 SM 就把一条 NVLink / 一整个 HBM 打满？**
>
> 这个问题在 DeepEP 里其实已经被一个**硬编码的解析式模型**回答了：每个 dispatch/combine kernel
> 开几个 SM，由 `EPBuffer.get_theoretical_num_sms()`（`deep_ep/buffers/ep.py:431-541`）算出来——
> 它把「每 token 的 HBM 读/写 + NVLink/RDMA 流量」折成**每 SM 的字节数**，再除以两个**每 SM 带宽常量**
> （`:524-528`），取四个候选里最大、×1.3、align 4、`max(4,·)`（`:530-532`）。这两个常量是
> `get_sm_read_gbs()=180` / `get_sm_write_gbs()=45`（`deep_ep/utils/envs.py:193-204`），**写死的、
> 还挂着 `TODO: use architecture-specific values`**。
>
> 第 63 篇（`docs/73`）把这层模型拆得很细，却只**打印**了这两个常量、从没在真机上量过它们；
> 本篇补上真机校验。结论是一条很干净的话：
>
> - **两个常量的数量级对、比例错**：真机单 SM 读 **145 GB/s**（DeepEP 偏乐观 **19.5%**）、
>   单 SM 写 **63 GB/s**（DeepEP 偏保守 **40.8%**）；模型隐含的 **读:写 = 4:1** 实际是 **2.29:1**。
> - **更关键的是「每 SM 带宽恒定」这个隐含假设只在很小的 SM 数下成立**：HBM 读的每 SM 线性区
>   只到 **8 个 SM**，聚合峰值 **3208 GB/s**（要 ~22 个 SM 才顶满）；**一条 NVLink P2P 在
>   `N=8` 个 SM 就饱和在 ~372 GB/s**。超过膝点，再给通信加 SM 是**白给**的。
> - 用真机常量重跑**真实**的 `get_theoretical_num_sms` 方法体：多个配置的 SM 数会移动
>   （`direct EP8` 16→12、`hybrid EP64` 8→4、`direct EP4` 20→16），但**模型上限是 `device_sms//2`
>   而不是链路膝点**——它并不建模「HBM/NVLink 已被打满」这件事。
>
> **可跑性 / 边界**：DeepEP 的 dispatch/combine **设备端**路径依赖 NCCL GIN 设备 API
> （本机 NCCL 2.29.7 < 需求 ≥2.31，见 ROADMAP「阻塞」）→ 端到端**仅分析**。但本节要校验的是
> **两个纯 host 常量**与**原始 CUDA 带宽**，与 DeepEP/NCCL 设备 API 无关：`get_theoretical_num_sms`
> 用 P63-1 的 `sys.modules` stub + `ast` 抽取跑**未修改的真实方法体**（`ep.py` sha256
> `a8f72c39…`、`envs.py` sha256 `23843e16…`），Part A/B 是纯 CUDA/CUDA-runtime（`src/sm_bandwidth_budget.cu`）。
> **H100（sm90）真机，8×H100 80GB HBM3。** 目标卡代号敏感：全文用「目标卡」指代。
>
> 参考坐标：`~/github/DeepEP@93eb6eb`（V2.5）`deep_ep/utils/envs.py`、`deep_ep/buffers/ep.py`；
> 原始输出 `harness/exp_sm_bandwidth_budget.out.txt`（`harness/exp_sm_bandwidth_budget.{sh,py}` +
> `src/sm_bandwidth_budget.cu`；H100 80GB HBM3, sm90, 1980 MHz, L2 50 MB）。

---

## 1. 为什么这两个常量决定了「通信腿开几个 SM」

先把「模型怎么用这两个数」讲清楚——因为它决定了本篇要量什么。

`get_theoretical_num_sms()`（`ep.py:431-541`）是一段**解析式 roofline**，四步：

1. **期望命中组数** `get_expected_topk(num_groups)`（`:473-475`）：门控在组内选 top-k，
   平均命中多少个**不同的**专家组，组合式 `G·(1 − C(E−E/G, k)/C(E, k))`。
2. **每 token 流量账**（`:483-509`）：以 `sm_read`/`sm_write` 为单位累计「每个 token 要在本 SM 上
   发生几次 HBM 读 / 写」，`rdma_traffic`/`nvlink_traffic` 为单位累计「跨 rank 的字节比例」。
   例如「读 token」就是 `sm_read += 1/num_expected_topk`（`:484`）。
3. **取瓶颈链路当标尺**（`:511-515`）：谁先打满用谁——`bounded_gbs/bounded_traffic` 就是
   「**每 token 上，瓶颈链路能提供的字节预算**」。
4. **反推 SM 数**（`:522-529`）：四个候选（dispatch 读 / dispatch 写 / combine 读 / combine 写）
   ```
   num_sms = max( bounded_gbs/bounded_traffic * sm_read/sm_read_gbs,
                  bounded_gbs/bounded_traffic * sm_write/sm_write_gbs,
                  bounded_gbs/bounded_traffic * (sm_write - nvlink_traffic)/sm_read_gbs,
                  bounded_gbs/bounded_traffic * (sm_read + nvlink_traffic)/sm_write_gbs )
   ```
   再 `×1.3` 余量、`align 4`、`max(4,·)`、`prefer_overlap=False` 时 `max(·,64)`、
   最后 `min(·, device_sms//2)`（`:530-532`）。

关键读法：**这一步在问「需要几个 SM 的 HBM 带宽，才能喂饱瓶颈链路」**。分母正是
`sm_read_gbs`/`sm_write_gbs`——所以「通信腿开几个 SM」**完全由这两个每 SM 带宽常量决定**。
它们在 `envs.py` 里是硬编码（`:194-204`）：
```
@functools.lru_cache()
def get_sm_read_gbs() -> float:
    """Return the default per-SM HBM read bandwidth in GB/s."""
    # TODO: use architecture-specific values.
    return 180

@functools.lru_cache()
def get_sm_write_gbs() -> float:
    """Return the default per-SM HBM write bandwidth in GB/s."""
    # TODO: use architecture-specific values.
    return 45
```

这两个数被消费的点就是 dispatch auto-SM（`ep.py:643`：
`num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else ...`）
与 combine（同函数复用）。`get_theoretical_num_qps`（`:543-548`）再由此推 QP 数。

那么问题只剩：**180 / 45 在真机上成立吗？「每 SM 带宽」真是常数吗？**

---

## 2. Part A：一个 SM 能拉多少 HBM 带宽，几个 SM 打满

实验（`src/sm_bandwidth_budget.cu`，`harness/exp_sm_bandwidth_budget.sh`）：grid = N 个 block
（N ≤ 132 ⇒ 每 SM 一个 block）、每 block 1024 线程、`float4`（16 B）向量化、`__ldcs`/`__stcs`
（streaming 提示，避免 L2 驻留），流式访问 2 GiB 缓冲的一片；取 5 次最小值。

### 2.1 三条腿的每 SM 带宽

| N（SM） | read 聚合 GB/s | read/SM | write 聚合 GB/s | write/SM | copy eff2x GB/s | copy/SM |
|--------:|---------------:|--------:|----------------:|---------:|----------------:|--------:|
| 1       | 145.3          | **145.3** | 63.4          | **63.4** | 77.4            | 38.7    |
| 2       | 290.8          | 145.4   | 126.7           | 63.3     | 152.4           | 38.1    |
| 4       | 576.5          | 144.1   | 253.3           | 63.3     | 298.2           | 37.3    |
| 8       | 1040.2         | 130.0   | 488.7           | 61.1     | 558.6           | 34.9    |
| 16      | 1862.6         | 116.4   | 973.3           | 60.8     | 995.1           | 31.1    |
| 24      | 2322.1         | 96.8    | 1084.0          | 45.2     | 1270.5          | 26.5    |
| 32      | 2884.8         | 90.2    | 1625.3          | 50.8     | 1691.7          | 26.4    |
| 48      | 3007.9         | 62.7    | 2152.7          | 44.8     | 1991.8          | 20.7    |
| 64      | 3058.7         | 47.8    | 2394.1          | 37.4     | 2457.4          | 19.2    |
| 96      | 2978.4         | 31.0    | 1905.9          | 19.9     | 2272.9          | 11.8    |
| 128     | **3208.2**     | 25.1    | **3165.1**      | 24.7     | 2908.2          | 11.4    |
| 132     | 3201.6         | 24.3    | 3177.5          | 24.1     | 2907.4          | 11.0    |

参考上限（库路径）：`cudaMemset` **3247 GB/s**、`cudaMemcpy D2D` 单程 **1522 GB/s**（eff2x 3045）。
所以 HBM3 的读/写聚合峰值就在 **~3.2 TB/s**——和 H100 HBM3 的物理带宽一致。

### 2.2 单 SM 峰值：180 / 45 的对错

`N=1` 且扫 `blockDim`（`A5_SINGLESM`）：

| blockDim | read GB/s | write GB/s |
|---------:|----------:|-----------:|
| 128      | 17.5      | 63.4       |
| 256      | 34.9      | 63.4       |
| 512      | 68.0      | 63.4       |
| 1024     | **145.0** | **63.4**   |

几点：

- **单 SM 读 ~145 GB/s，单 SM 写 ~63 GB/s**。DeepEP 假设的 **180** 偏高（真机是 0.81×），
  **45** 偏低（真机是 1.41×）。两者都还在同一数量级——所以第 63 篇用 180/45 跑出来的 SM 数
  不会离谱，但**偏置方向不同**：读被高估 ⇒ 模型会**低估**所需 SM 数；写被低估 ⇒ 会**高估**。
- **读:写 = 145:63 ≈ 2.29:1，不是模型隐含的 4:1**。读对 `blockDim`（= 在飞 load 数）敏感、
  写出奇地不敏感（128 线程就顶到 63，因为 store 走 write-back、LSU 不用等）——这解释了为什么
  DeepEP 的「写只有读的 1/4」这种强不对称在真机上并不成立。
- 单 SM copy（读+写）eff2x 只有 **77 GB/s**（1024 线程）——读和写抢同一条 LSU／HBM 通道。

### 2.3 每 SM 带宽根本不是常数

看 `read/SM` 一列：`N≤4` 时稳稳 **145**；`N=8` 掉到 130；之后**单调下降**，`N=128` 只剩 25。
原因plainly：HBM 是**共享资源**，聚合带宽封顶 ~3.2 TB/s，于是
`per-SM = min(单SM峰值, 聚合峰值/N)`。把 `3208/145 ≈ 22`——**理论上要 ~22 个 SM 才把读打满**；
实测「每 SM 线性」在 **8 个 SM** 后就开始破（>10% 偏离）。

写这条腿更怪：`N=1..16` 每 SM ~61-63，`N=24..48` 在 45-51 之间抖（8 卡共享节点上剧烈争用），
`N=96` 掉到 20（明显被邻居抢占），`N=128` 才爬到 3165。**共享节点噪声大**，但峰值和单 SM 值得信。

> 一句话：DeepEP 的 `num_sms = …×sm_read/sm_read_gbs` **是一个线性外推**。它在
> 「SM 数 < ~8」时正确；一旦模型把 SM 数推到膝点之上，它假设的「每 SM 145 GB/s」就再也拿不到了
> ——多出的 SM 拿到的是被稀释后的份额，甚至 0（HBM 已满）。

---

## 3. Part B：一条 NVLink 要几个 SM 才打满

DeepEP 的 dispatch/combine 真正的瓶颈是**跨 rank 链路**（NVLink 或 RDMA）；SM 只是「把数据从
HBM 搬进链路」的泵。所以另一个问题是：**几个 SM 能把一条 NVLink P2P 打满？**

用 8 卡 UVA peer（`cudaDeviceEnablePeerAccess`，`cudaDeviceCanAccessPeer` 真机返回 1），在 rank0
上跑 grid = N 从 rank1 读 / 写 256 MiB：

| N（SM） | P2P read GB/s | P2P write GB/s |
|--------:|--------------:|---------------:|
| 1       | 57.0          | 50.6           |
| 2       | 112.6         | 101.2          |
| 4       | 218.0         | 201.9          |
| **8**   | **366.1**     | 329.6          |
| 16      | 369.6         | 349.8          |
| 32      | 371.1         | 354.9          |
| 64      | 369.4         | 366.8          |
| 128     | 372.4         | 370.9          |
| 132     | 366.4         | 366.1          |

- **单 SM 一条 NVLink P2P 只有 ~57 GB/s 读 / ~51 GB/s 写**——比单 SM HBM（145/63）**还慢**，
  说明瓶颈在链路往返/协议，不在 SM 的 LSU。所以「每 SM 带宽」对 HBM 和对 NVLink 是**两个不同的数**，
  DeepEP 模型却只按 HBM 的那个数（180/45）算，这是它**建模的粗糙点**。
- **`N=8` 就饱和在 ~366-372 GB/s**，之后无论 16/32/128/132 都不动。H100 单 GPU NVLink 4
  单方向理论 ~450 GB/s，实测 ~372（0.83×）。这与 DeepEP 从 `nvidia-smi nvlink -s` 解析出的
  `get_nvlink_gbs()=430.3`（18×26.562×0.9）是同一档（真机 P2P 是单对，总聚合对多对更高）。
- **结论**：**一条 NVLink 的通信腿 8 个 SM 就封顶**。给 dispatch/combine 开超过 ~8 个 SM，
  如果瓶颈真是这条链路，**多出来的 SM 对带宽毫无贡献**（白给）。

---

## 4. Part C：把实测常量代回真实模型

用 P63-1 的 `ast` 抽取跑**未修改的真实** `get_theoretical_num_sms`，只把 `sm_read_gbs`/`sm_write_gbs`
从左换右（`sm_read=145`/`sm_write=63`），对 9 个配置逐一对拍：

| 配置 | 默认(180/45) | 实测(145/63) | Δ |
|------|-------------:|-------------:|--:|
| direct EP8（单节点无 RDMA） | 16 | 12 | −4 |
| hybrid EP16 | 16 | 12 | −4 |
| hybrid EP32 | 8 | 8 | 0 |
| hybrid EP64 | 8 | 4 | −4 |
| hybrid EP128 | 4 | 4 | 0 |
| direct EP8（不偏好 overlap） | 64 | 64 | 0 |
| hybrid EP16（RDMA 慢，瓶颈在 RDMA） | 4 | 4 | 0 |
| direct EP4 | 20 | 16 | −4 |
| direct EP1（无流量） | 132 | 132 | 0 |

读法：

- 对**NVLink-bound / direct** 的配置（EP8、EP4），换实测常量后 SM 数**下降**——因为真机单 SM 读
  比 180 低、写比 45 高，模型重算后多数落在写候选或更小的读候选上，净效应是 **SM 数变少**。
- `direct EP8` 的 **16 → 12**：注意 16 已经**超过**第 3 节量到的 NVLink 膝点（8）——说明
  **默认模型给通信多开了 ~2× 的 SM**；换实测常量后 12 仍略高于膝点。而
  `prefer_overlap=False` 分支的 `max(·,64)`（`ep.py:531`）直接把 EP8 顶到 **64**，是膝点的 **8×**
  ——这 56 个 SM 对 comm 带宽是**白给**的（第 190 篇从计算侧也量到「访存受限腿的 SM 几乎白送」，
  两篇在这一点上互相印证）。
- **模型不建模链路膝点**：四个候选都是 `bounded_gbs/bounded_traffic × (每 SM 字节)/(每 SM 带宽)`，
  随 `sm_read_gbs` 线性缩放、没有上限项——唯一的封顶是 `min(·, device_sms//2)`（`:532`）。
  所以它能心安理得地要 64 个 SM，即便一条 NVLink 8 个就满了。**这正是「每 SM 带宽是常数」这个
  假设的直接后果**：把共享资源当成了可无限堆叠的私有资源。

---

## 5. 与第 190 篇拼起来：一条 SM 的「双向汇率」

第 190 篇量的是**计算腿**的汇率（每让出一个 SM，L1 +1.4~3 μs、wgrad +0~24 μs），
本篇量的是**通信腿**的汇率（每加一个 SM，读到 ~145 GB/s、写到 ~63 GB/s、一条 NVLink 到 ~57 GB/s，
直到 8 个 SM 饱和）。两者合起来给「按 SM 分区」这条设计轴一个可算的交换比：

- **通信侧**：一条 NVLink 的 P2P 需要 ~8 个 SM 就打满；超过后加 SM **零收益**。
- **计算侧**：访存受限腿（反向 wgrad）让 SM 近免费（~0 μs/SM），计算受限腿（L1）~1.4-3 μs/SM。
- ⇒ **最优切分点**通常落在「通信腿 ≤ 8 个 SM、其余给计算」附近；DeepEP 的
  `prefer_overlap=False`（≥64）和某些 `prefer_overlap=True` 的默认值会**越过膝点**，
  而越过之后那些 SM **既没喂饱链路、又让计算腿慢了**。

这也解释了 Mega MoE 主 kernel 为什么不走「按 SM 分区」：它把通信和计算放进**同一块 SM** 的
不同 warp（第 2/9/10 篇），让通信 warp 在访存受限时用不掉整个 SM 的空闲 LSU/HBM——
比「切一整块 SM 给通信、超过膝点就白给」更省。**SM 分区的粒度太粗**（第 190 篇的拐点、
本篇的 NVLink 膝点都说明：只有 ≤8 个 SM 的通信预算是有意义的）。

---

## 6. 小结与边界

- **两个硬编码常量**（`envs.py:193-204`）：真机单 SM 读 **145**（模型 180，偏乐观 19.5%）、
  写 **63**（模型 45，偏保守 40.8%）；模型隐含的 **4:1 读:写** 实际 **2.29:1**。
- **「每 SM 带宽是常数」只在小 SM 数下成立**：HBM 读的线性区 ≤ **8 SM**、聚合峰值 **3208 GB/s**；
  一条 **NVLink P2P 在 8 SM 饱和于 ~372 GB/s**。超过膝点，通信加 SM 是**白给**。
- **真实模型重算**（`ep.py` sha256 `a8f72c39…`）：多个配置 Δ=−4，`prefer_overlap=False`
  分支把 EP8 顶到 64（膝点的 8×）。模型的封顶是 `device_sms//2`，**不建模链路膝点**。
- **边界**：DeepEP dispatch/combine **设备端**（NCCL GIN）本机跑不了（`docs/86` 的两道闸门，
  ROADMAP「阻塞」）——本篇量的是它**脚下的两个 host 常量**与**原始 CUDA 带宽**，不碰设备 API；
  P2P 是 rank0↔rank1 单对，8 卡 all-to-all 的聚合会更高。8 卡共享节点上写带宽有争用噪声
  （`N=96` 明显受损），但单 SM 峰值与聚合峰值稳定可复现。

---

## 附：复现

```
bash code/mega-moe/harness/exp_sm_bandwidth_budget.sh \
  > code/mega-moe/harness/exp_sm_bandwidth_budget.out.txt
```
- `src/sm_bandwidth_budget.cu`：Part A（HBM 三条腿 + 单 SM 峰值扫描）、Part B（NVLink P2P）；
  纯 CUDA/CUDA-runtime，`nvcc -arch=sm_90`。
- `harness/exp_sm_bandwidth_budget.py`：Part C，import P63-1 的 `exp_deepep_sm_qp` 复用其
  `ast` 抽取，跑**真实** `get_theoretical_num_sms` 方法体；判定 `PASS=6 / FAIL=0`。
