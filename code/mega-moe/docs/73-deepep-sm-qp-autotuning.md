# DeepEP dispatch / combine 的自动 SM / QP 选择模型：带宽探测层与解析式 roofline

> 本文对应路线图 **P63-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 73 篇。
>
> 前面第 7 / 48 / 58 / 61 / 63 / 71 篇把 DeepEP 的 dispatch / combine kernel **本体**拆得很细，
> 但一直没回答一个上游问题：**每次 `dispatch()` / `combine()` 到底开多少个 SM、多少个 RDMA QP？**
> 这两个数字不是超参、也不是调出来的经验值，而是由 host 侧一段**解析式 roofline 模型**在运行时算出来的
> ——`EPBuffer.get_theoretical_num_sms` / `get_theoretical_num_qps`。
>
> **与专题的关系**：Mega MoE 的「通信—计算重叠」要成立，通信侧必须**恰好用掉它该用的那部分硬件**：
> 用少了 bandwidth-bound 的 NVLink/RDMA 跑不满，用多了又会和 Tensor Core 抢 SM（第 52 篇 PDL 的
> `num_sms` 上限、第 43 篇 EPLB 的 `lb_get_theoretical_num_sms` 都是同一主题的其它落点）。本篇补上
> DeepEP 里**决定 overlap 预算**的那一层。
>
> **可跑性**：这一层是**纯 host / Python**，不碰任何 CUDA / NCCL 设备 API，也不依赖 SM100。
> 带宽探测走真实的 `nvidia-smi` / `ibstat`；模型本体用 `ast` 从真实 `ep.py` **逐行抽取方法源码、
> 运行未经修改的真实方法体**（`harness/exp_deepep_sm_qp.{sh,py,out.txt}`），与独立 Python oracle
> 逐配置对拍，**H100 真机 16 PASS / 0 FAIL**。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - 模型：`deep_ep/buffers/ep.py:431-541`（SM 数）、`:543-560`（QP 数）、`:846-852`（EPLB 对照）
> - 消费点：`deep_ep/buffers/ep.py:641-645`（`dispatch`）、`:780` 起（`combine`）、`:882` / `:904`（LB）
> - 带宽探测：`deep_ep/utils/envs.py:193-283`
> - 域分解：`csrc/kernels/comm/context.cpp:52-63`（physical / logical）、`:166-183`（Context 构造）

## 1. 一段话总览

`EPBuffer` 在构造时（`ep.py:242-363`）先把「**我处在怎样的拓扑里**」拆成两组域大小：
**物理域** `(num_rdma_ranks, num_nvlink_ranks)`（RDMA 域 = 有多少个节点、NVLink 域 = 每节点几张卡）
与**逻辑域** `(num_scaleout_ranks, num_scaleup_ranks)`（hybrid 模式下就等于物理域；非 hybrid 模式
把所有 rank 塞进一个 scaleup 域、scaleout=1，见 `context.cpp:174-183`）。然后 `dispatch()` 每次调用
都问这两个模型要「**这次该用几个 SM / 几个 QP**」：

```
num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else align(num_sms, 2)   # ep.py:643
num_qps = self.get_theoretical_num_qps(num_sms)              if num_qps == 0 else num_qps               # ep.py:644
```

模型的直觉只有一句：**把 dispatch/combine 拆成「每 token 的 HBM 读 / HBM 写 / RDMA 流量 / NVLink 流量」
四本账，用「单位 token 的流量」(per-token traffic) 归一化，再取「谁能先成为瓶颈就用谁当标尺」，最后
用每 SM 的 HBM 带宽把流量换算成 SM 数。** 三个隐藏假设写在注释里：只支持 balanced gate
（`:454-457`）、不展开 top-k（`:451` TODO）、不单独计 HBM 往返（`:469`）。

## 2. 带宽探测层：模型里所有「每 GB/s」从哪来

模型里有四个带宽量，全部来自 `deep_ep/utils/envs.py`，且都 `@functools.lru_cache()` 缓存（探测一次、
进程内复用）：

| 函数 | 行号 | 值 / 解析方式 | 语义 |
|---|---|---|---|
| `get_sm_read_gbs()` | `:193-197` | 常量 **180** | 单 SM HBM **读**带宽（GB/s） |
| `get_sm_write_gbs()` | `:200-204` | 常量 **45** | 单 SM HBM **写**带宽 |
| `get_nvlink_gbs(factor=0.9)` | `:207-234` | `nvidia-smi nvlink -s` 求和 × **0.9** | 本节点 NVLink 总带宽 |
| `get_rdma_gbs(nic_name='mlx5_0')` | `:260-283` | `ibstat` 解析 `Rate` ÷ 8 | 单 NIC RDMA 带宽（Gb/s→GB/s） |

三个「为什么」：

1. **读 180 / 写 45** 是**每 SM** 而非整卡：H100 整卡 HBM ≈ 3.35 TB/s 分给 132 SM ≈ 25 GB/s/SM，但
   dispatch 的拷贝是**流式连续**访问、单 SM 靠足够多的 in-flight load 能跑到远超均值——180 是「单 SM
   顺序流能刷出的读上限」的经验值，近似 H100 单 SM 的 L2→SM 填充上限。读远大于写（4×）如实反映了
   「写要在 L2 分配 + 回写」的开销，也解释了后面 `sm_write` 一侧常先到瓶颈。
2. **`get_nvlink_gbs` 乘 0.9**：`nvidia-smi` 报的是链路理论速率（H100 为 18×26.562 ≈ 478.1 GB/s），
   0.9 是**协议/纠错效率折损**（见第 53/64 篇实测：本机 Σlinks=478.1 → 返回 430.3）。
3. **`get_rdma_gbs` 用 `ibstat` 而不是 NVML**：RDMA 是网卡属性，`ibstat` 报的是 `Rate`（Gb/s），
   除以 8 得 GB/s。**本机没有 IB/`ibstat`**，函数捕获异常返回 **0**——这个「探不到就 0」在后面的
   混合模式里会引发一个真实的除零坑（第 7 节）。

探测层还有两个纯检查函数：`check_fast_rdma_atomic_support`（`:237-257`，看网卡 CA type 是否
`MT4131` 或更新——决定 hybrid 默认分配多少 QP，见第 5 节）与 `check_nvlink_connections`
（`:146-181`，PCIe 卡只允许两两 NVLink）。它们一起构成模型输入的「环境感知」面。

## 3. 域大小：模型的两个自变量组从哪来

`get_theoretical_num_sms` 用的是 **物理域**（`num_rdma_ranks` / `num_nvlink_ranks`）**与逻辑域的
混合**：`self.num_nvlink_ranks` 出现在 `else` 分支（direct 模式），`self.num_scaleout_ranks` /
`self.num_scaleup_ranks` 出现在 hybrid 分支。这些字段在 `ep.py:352-358` 落定：

```
self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()   # ep.py:353
self.num_rdma_ranks,    self.num_nvlink_ranks   = self.get_physical_domain_size()  # ep.py:358
```

而 C++ 侧的定义（`context.cpp:52-63`）很干净：

- **物理域**：`num_nvl_ranks = ncclTeamLsa(comm).nRanks`（NCCL 的「本机/可达 NVLink 域」），
  `num_rdma_ranks = num_ranks / num_nvl_ranks`（节点数）。`get_physical_domain_size:52-57`。
- **逻辑域**：`allow_hybrid_mode ? (num_rdma_ranks, num_nvl_ranks) : (1, num_ranks)`
  （`get_logical_domain_size:59-63`）。即**关闭 hybrid 时假装只有一级域**，把所有 rank 都当
  NVLink 域（哪怕要绕 RDMA），这正是 `dispatch.hpp:89-114` 选择 single-level kernel 的依据。

Context 构造时（`context.cpp:166-183`）还会校验 `rank_idx == rdma_rank_idx * num_nvl_ranks +
nvl_rank_idx`（`:172`），并令 `is_scaleup_nvlink = (num_scaleup_ranks == num_nvl_ranks)`（`:183`）
——这一位决定 scaleup 腿到底走不走 NVLink（第 63 篇屏障、第 71 篇 hybrid 都用它）。

## 4. `get_theoretical_num_sms`：逐行拆解析式 roofline（`ep.py:431-541`）

### 4.1 期望命中数 `get_expected_topk`（`:473-475`）

```python
def get_expected_topk(num_groups: int) -> float:
    assert num_experts % num_groups == 0
    return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk)
                            / math.comb(num_experts, num_topk))
```

含义：一个 token 从 `E` 个专家里选 `k` 个（无放回均匀），问「**它平均会命中多少个不同的组**」。
`C(E - E/G, k)/C(E, k)` 是「一个组一个都没被选中」的概率，故 `1 - 该值` 是「某组至少命中一个」的
概率，乘组数 `G` 得期望命中组数。这是 **group-limited gate**（DeepSeek V3 的「top-k 必须来自不同
组」路由）在 balanced 假设下的解析形式——第 4 节末尾的实验用暴力枚举 `C(E,k)` 逐项验证了它
（6 组形状差 ≤ 4.4e-16）。**注意它的 `assert`**：模型把 `num_scaleout_ranks` 当 `num_groups`，
所以隐含要求 `E` 能被域大小整除（EP144 + 256 专家会被断言拒绝，见第 7 节）。

模型里用到两次（`:477-481`）：`num_expected_scaleout_topk = get_expected_topk(num_scaleout_ranks)`
（跨节点命中的组数）、`num_expected_topk = get_expected_topk(num_ranks)`（全局命中的组数，作为
**归一化分母**）。V3 真实配置 `E=256, k=8, G=8` 下 `num_expected_topk = 5.2947`。

### 4.2 四本流量账（`:483-509`）

记账单位是「**单位 token**」——把总量除以 `num_expected_topk` 就得到「每 token 平均的专家副本数」。
注释 `:453-454` 说得很直白：`1` 代表 scale-up 流量，等于「token 数 × 期望 top-k × 每 token 数据量」。
逐项拆（每条后面是该行「为什么」）：

**公共项**
- `sm_read += 1 / num_expected_topk`（`:484`）：dispatch 的 copy epilogue 从 HBM **读**回 token
  （量按期望 top-k 归一化）。

**hybrid 分支（`num_scaleout_ranks > 1`，`:487-498`）**
- `sm_write += 1 / num_expected_topk`（`:489`）：scaleout warp 把 token 写进 send buffer。
- `sm_write += (1/ne) * (nes / num_scaleout_ranks)`（`:492`）：目的 rank = 本节点的部分**本地绕过**，
  不过 RDMA 但要写。
- `rdma_traffic += (1/ne) * (nes * (1 - 1/num_scaleout_ranks))`（`:493`）：其余走跨节点 RDMA。
- `sm_read += nes / ne`（`:496`）：forward warp 要把收到的数据再读出来。
- `sm_write += 1`（`:497`）：issue scale-up（每 token 一次 NVLink 写）。
- `nvlink_traffic += 1 - 1/num_scaleup_ranks`（`:498`）：scaleup 域内除了本地那份，其余都要上 NVLink。

**direct 分支（`else`，`:499-509`）**
- `if num_rdma_ranks > 1: sm_write += 1/ne`（`:501-502`）：单级模式下若物理上仍跨节点，仍要写 send buffer。
- `sm_write += num_nvlink_ranks / num_ranks`（`:505`）：issue NVLink（按 NVLink 域占比）。
- `nvlink_traffic += (num_nvlink_ranks/num_ranks) * (1 - 1/num_nvlink_ranks)`（`:508`）：排除本地绕过。
- `rdma_traffic += (num_ranks - num_nvlink_ranks)/num_ranks`（`:509`）：跨节点的部分。

### 4.3 选瓶颈（`:511-515`）

```python
if self.num_scaleout_ranks > 1 and (rdma_traffic/rdma_gbs) > (nvlink_traffic/nvlink_gbs):
    bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
else:
    bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs
```

「哪条链路的**预计耗时**（流量÷带宽）更长，就用它当标尺」。**这决定了 SM 是为谁服务**：SM 数的目标
是让 SM 侧的 HBM 读/写**刚好喂饱/吸干**这条瓶颈链路——这是「通信—计算重叠」的预算来源。

### 4.4 由流量反推 SM 数（`:517-532`）

```python
num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count   # :519
num_sms = num_device_sms                     # default：无流量（EP=1）
if bounded_traffic > 0:
    num_sms = max(                                          # 四个候选里取最大
        bounded_gbs/bounded_traffic * sm_read  / sm_read_gbs,   # dispatch 读
        bounded_gbs/bounded_traffic * sm_write / sm_write_gbs,  # dispatch 写
        bounded_gbs/bounded_traffic * (sm_write-nvlink_traffic)/sm_read_gbs,   # combine 读
        bounded_gbs/bounded_traffic * (sm_read+nvlink_traffic)/sm_write_gbs,   # combine 写
    )
num_sms = align(max(4, math.ceil(num_sms * 1.3)), 4)                    # :530
num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)  # :531
num_sms = min(num_sms, num_device_sms // 2 * 2)                        # :532
```

逐条「为什么」：

- **四个候选**对应 dispatch 的读/写与 combine 的读/写（combine 的流量用 `sm_write - nvlink_traffic`
  与 `sm_read + nvlink_traffic` 修正，因为 combine 方向相反）。`bounded_gbs/bounded_traffic` 是
  「瓶颈链路按流量摊一次的秒数」，再乘「本侧每 token 字节 ÷ 每 SM 带宽」即得 SM 数。
- **×1.3 + `align(...,4)` + `max(4,...)`**（`:530`）：1.3 是**安全余量**（模型是 balanced 假设，
  真实 gate 有偏斜）；4 的倍数是因为 kernel 的 `cluster` / `block_m` 组织；下限 4。
- **`prefer_overlap_with_compute`**（`:531`）：默认 True 时「能少用就少用」（给计算留 SM）；置 False
  时强制 `max(num_sms, 64)`——即「不打算与计算重叠，那就多开 SM 把通信尽快做完」。这是
  `EPBuffer(...)` 的构造参数（`ep.py:257`）。
- **`min(..., device_sms//2*2)`**（`:532`）：**硬上限一半 SM**——无论流量多大都不许吃掉整卡，保证
  另一侧的 compute（或下一个 kernel）有卡可用。这也是第 52 篇 PDL「通信离开计算单元」的数值依据。

## 5. `get_theoretical_num_qps`：QP 数模型与默认分配（`ep.py:543-560`）

QP（Queue Pair）是 RDMA 的排队资源。模型很简单，但两条分支的动机完全不同：

```python
num_qps = min(num_sms, 8 + 1)          # :554  direct 模式：鼓励少 QP，减少 doorbell ringing 开销
if self.allow_hybrid_mode:
    num_qps = num_sms * 16 + 1         # :558  hybrid：每个 channel（16 个）与 notify 都想独占一个 QP
return min(num_qps, self.num_allocated_qps)   # :560  受分配上限约束
```

- **direct `min(num_sms, 9)`**：单级模式下 `num_sms` 个 SM 共享少量 QP 更省 doorbell 开销
  （注释 `:553`）。
- **hybrid `num_sms*16+1`**：hybrid kernel 里每个 SM 有 **16 个 channel**（第 71 篇的 warp 角色）+
  1 条 notify 通道，希望**每条通道独立 QP** 以并行发射。

分配上限 `num_allocated_qps` 的默认值在构造时定（`ep.py:327-335`）：

| 场景 | 默认 `num_allocated_qps` |
|---|---|
| direct（非 hybrid） | **17** |
| hybrid + 网卡支持 fast atomic（`MT4131+`） | **65** |
| hybrid 否则 | **129** |

「为什么 hybrid 要 65/129」：hybrid 的 QP 需求 `num_sms*16+1` 远大于 direct，所以必须**预分配**更多
QP；fast-atomic 网卡（MT4131+）硬件能力更强、doorbell 更便宜，可以用更少的 QP 达到同样吞吐，故 65
（否则退回 129）。**本机 `check_fast_rdma_atomic_support()`=False**（无 `ibstat`），所以若要开 hybrid
分配的是 129。

## 6. 模型在哪里被消费

- `dispatch`（`ep.py:641-645`）：`num_sms==0` → 调 `get_theoretical_num_sms(num_experts, num_topk)`；
  `num_qps==0` → 调 `get_theoretical_num_qps(num_sms)`；随后断言 `num_qps <= num_allocated_qps`
  （`:645`）。用户显式传入时只做 `align(num_sms,2)`。
- `combine`（`ep.py:780` 起）走同样的自动决策。
- **EPLB 副本通信**（`lb_prefetch_weights:854-884` / `lb_reduce_grads:886-906`）用**另一个**更窄的模型
  `lb_get_theoretical_num_sms`（`:846-852`）：`align(ceil(max(nvlink/sm_read, nvlink/sm_write)), 2)`
  ——只按「读/写专家权重」两本账估，与冗余映射无关（第 43 篇已从 EPLB 角度拆过，这里作为对照）。

## 7. H100 真机实验（`harness/exp_deepep_sm_qp.{sh,py,out.txt}`）

**方法**：`import deep_ep` 会触发 `check_nccl_so()`（本机 NCCL 2.29.7 与编译期不匹配 → abort），故用
`sys.modules` 注入合成 `deep_ep._C` / `deep_ep.comm` stub 后按文件路径 exec **真实** `envs.py`
（sha256 `23843e16…`）；再用 `ast` 从真实 `ep.py`（sha256 `a8f72c39…`）**逐行抽取**
`get_theoretical_num_sms` / `get_theoretical_num_qps` / `lb_get_theoretical_num_sms` 三个方法源码
（行号自动核对：`:431-541` / `:543-560` / `:846-852`），装进 shim 运行**未经修改的真实方法体**，
并与纯 Python 独立 oracle 对拍。**16 PASS / 0 FAIL**：

**A. 带宽探测层（真实 `nvidia-smi`）**
- `get_sm_read_gbs()=180`、`get_sm_write_gbs()=45`；`get_nvlink_gbs()=430.3`（18×26.562=478.1 ×0.9）；
  `get_rdma_gbs()=0`（无 `ibstat`）；`check_fast_rdma_atomic_support()=False`。
- lru_cache 生效（二次调用返回同一对象）。

**B. `get_theoretical_num_sms`：真实方法 vs oracle，逐配置一致**

| 配置 | num_sms |
|---|---|
| direct EP8（单节点 LSA8，无 RDMA） | 16 |
| hybrid EP16（2 节点×LSA8） | 16 |
| hybrid EP32（4 节点） | 8 |
| hybrid EP64（8 节点） | 8 |
| hybrid EP128（16 节点） | 4 |
| direct EP8，`prefer_overlap_with_compute=False` | 64 |
| hybrid EP16，RDMA 慢（12 GB/s，瓶颈切到 RDMA） | 4 |
| direct EP4 | 20 |
| direct EP1（无流量） | **132**（= device_sms） |

单调性很直观：**域越大 → 每 token 的跨域流量按 `1-1/R` 更接近饱和 → 需要的 SM 反而更少**（4~16），
而单节点小规模直接靠 HBM 带宽，会用更多 SM。

**C. `get_theoretical_num_qps` 逐配置一致**：direct 恒 `min(num_sms,9)`（本例都 9）；
hybrid 命中 `num_sms*16+1` 再被分配上限钳制（含 `EP32/64: 129`、`EP128: 65`——因为 `num_sms=4`
时 `4*16+1=65 < 129`）。

**D. `get_expected_topk` 组合数学**：6 组 `(E,k,G)` 与暴力枚举 `C(E,k)` **差 ≤ 4.4e-16**；
另坐实隐含假设 `E % num_groups == 0`——`(256,18)` 被 `assert` 拒绝，`(256,16)`/`(256,8)` 通过
（即 **EP144×256 专家无法套用该模型**）。

**E. 边界 / 单调性**：`num_sms` 恒为 4 的倍数、恒 ≤ `device_sms//2*2=132`、恒 ≥ 4；
`prefer_overlap=False` 时确实 ≥ 64；EP1 无流量直接返回 132。

**F. 真实「坑」**：本机无 IB → `get_rdma_gbs()=0`。hybrid 模式下若不显式传 `rdma_gbs`，模型会走
`if rdma_gbs == 0 and num_rdma_ranks > 1: rdma_gbs = get_rdma_gbs()`（`:463-464`）→ 得 0 →
随后 `rdma_traffic / rdma_gbs`（`:512`）**抛 `ZeroDivisionError`**。我们用真实方法体复现了这一行为
（PASS）。结论：**混合模式在没有可探测 IB 的环境里必须显式传入 `rdma_gbs`**，否则自动选 SM 会崩。

**G. 对照 `lb_get_theoretical_num_sms`**：本机 `max(430.3/180, 430.3/45)=9.5623`，
`ceil→10`，`align 2→10`（与第 43 篇 EPLB 实测一致）。

## 8. 小结与已知边界

**这套模型要记住的三个「为什么」**：

1. **它是一本「每 token 流量账」+ 一条 roofline**：把 dispatch/combine 拆成 HBM 读/写、RDMA、NVLink
   四本账，用 group-limited 的期望命中数归一化，挑出耗时最长的链路当标尺，再用**每 SM HBM 带宽**
   （读 180 / 写 45）换算 SM 数。通信—计算重叠能成立，是因为通信只被预算了「刚好够用」的 SM。
2. **三个安全阀**：`×1.3` 余量、`align 4`、以及**硬上限半卡** `device_sms//2*2`——保证即使模型低估
   偏斜，也不会让通信吃光整卡。
3. **QP 数与 SM 数解耦**：direct 少 QP 省 doorbell，hybrid 每 channel 一个 QP，预分配上限由网卡
   fast-atomic 能力决定（65 / 129 / direct 17）。

**明确标注的边界（诚实清单）**：
- **仅支持 balanced gate**（`:454-457`）：V3 的 group-limited gate 若真实分布偏离均匀，模型
  `×1.3` 只是为了兜住的近似，不是精确值。
- **不支持 `do_expand` / `allow_multiple_reduction`**（`:451` TODO）。
- **隐含 `E % num_groups == 0`**：把 `num_scaleout_ranks` 当 `num_groups`，EP144 一类拓扑会被断言拒绝。
- **本机无 IB 时 hybrid 自动探测会除零**（第 7 节 F）；线上（有 IB）由 `ibstat` 提供 `rdma_gbs` 才成立。
- 本篇只验证**模型层**（纯 host），真实 dispatch/combine 的 SM/QP 使用效果仍需 NCCL ≥ 2.32.3 环境
  （见路线图「阻塞」）。
