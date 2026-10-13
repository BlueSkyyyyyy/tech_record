# DeepEP 的 NVLink 连接自检 `check_nvlink_connections` 与 H100 P2P 拓扑

> 本文对应路线图 **P109-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的**第 119 篇**。
>
> 第 73 篇拆过 DeepEP 的 SM/QP 自动选型，第 86 篇拆过它的 NCCL 通信域绑定与 `find_pkgs`，第 118 篇
> 拆过测试/进程引导层——但 `envs.py` 里只剩**最后一个被反复点名、却从没单独读过**的函数：
> `check_nvlink_connections`（`deep_ep/utils/envs.py:146-181`）。第 86 篇一句「PCIe GPU 只能两两
> NVLink」，第 118 篇一句「构造 engram buffer 时调用」。它不长（36 行），却是 DeepEP 所有
> 「节点内 GPU 两两都能通过 NVLink 直接互访」假设的**唯一运行时自检**——对称内存、`num_nvlink_ranks`
> 域、`nvlink_barrier`、Mega MoE 的 L2 远端直写 combine 全都建立在它之上。
>
> **可跑性**：整层是**纯 host / Python + NVML + CUDA 运行时**，不碰 UMMA/TMA/NCCL 设备 API，
> 也非 SM100 专属。实验用 `sys.modules` 注入合成 `deep_ep._C`/`deep_ep.comm` 绕过 `import deep_ep`
> 的 `check_nccl_so` 闸门，然后**直接加载真实 `deep_ep/utils/envs.py` 源码**真机调用，并编译
> `src/nvlink_p2p_matrix.cu` 量 8×8 有向 P2P 带宽矩阵。**H100 真机 16 PASS / 0 FAIL**
> （`harness/exp_deepep_nvlink_p2p.{sh,out.txt}`）。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - `deep_ep/utils/envs.py`：`check_nvlink_connections:146-181`（`envs.py` sha256
>   `23843e168a6f5d43…`，与第 107/117 篇同一份）
> - 消费点：`deep_ep/buffers/engram.py:67`、`deep_ep/buffers/ep.py:325`
> - 邻居（已被别的篇目覆盖）：`check_torch_deterministic:184-190`（第 118 篇）、
>   `find_pkgs.py`（第 86 篇）、`get_{physical,logical}_domain_size:136-143`（第 73/86 篇）
> - 实验：`harness/exp_deepep_nvlink_p2p.py` + `.sh/.out.txt`、`src/nvlink_p2p_matrix.cu`

---

## 1. 为什么这层值得单独一篇

DeepEP 的**节点内**通信有一个根深蒂固的假设：同一个「节点」（LSA 域）里的所有 GPU 都能通过
**NVLink** 两两直接互访，形成一张**全互联（full mesh）**。这个假设被大量代码依赖：

- `get_physical_domain_size` 把 `ncclTeamLsa(comm).nRanks` 当作 **`num_nvl_ranks`**（NVLink 域大小，
  `csrc/kernels/comm/context.cpp:54`）——NVLink 域越大，一次能直接互访的 GPU 越多、走 RDMA 的越少；
- 对称内存（第 56 篇）在 LSA 域内用 `get_sym_ptr` 做「同 offset 翻译」，前提是每个 rank 都有一块
  布局一致的镜像，且能直接寻址；
- Mega MoE 的 combine 用 `sym_buffer.map` 把 BF16 结果**直写远端**（第 11 篇，`kernel:1346-1351`），
  `nvlink_barrier`（第 87 篇 `comm/barrier.cuh`）在 LSA 域内做免复位同步；
- Megatron / DeepEP 的 EP 分组策略：EP8 时通常一个节点 8 卡全互联、`num_nvl_ranks=8`。

但**并不是所有机器都全互联**：早年 **A100 PCIe** 卡只有**两两配对的 NVLink**（GPU0↔GPU1、
GPU2↔GPU3……），此时一个 8 卡的节点根本没法组成全互联——跨过 NVLink 的访问要么走 PCIe、
要么走 CPU。于是 DeepEP 在构造 buffer 时放了一个自检：**如果是 PCIe GPU，就断言 EP 组不超过 2**。
`check_nvlink_connections` 就是这段逻辑，源码注释（`:154-155`）把它说得很清楚：

```
# NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
# TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
```

它也是 DeepEP 里**唯一**用到 NVML（`pynvml`）做 P2P 能力探测的地方——其余的 P2P/带宽探测要么靠
`nvidia-smi`（`get_nvlink_gbs`）、要么靠 `ibstat`（`get_rdma_gbs`，第 73 篇）。这带来两个值得讲清的问题：

1. **它检查的到底是什么？** 不是带宽，而是 NVML 给出的**离散 P2P 能力状态**；
2. **H100 上它会走到吗？** 门是「设备名里有没有 `'PCIE'`」——一个**字符串判据**。

---

## 2. 逐行精读 `check_nvlink_connections`（`:146-181`）

```python
def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if 'PCIE' in torch.cuda.get_device_name():
        assert group.size() <= 2, 'PCIe GPUs only have pairwise NVLink connections'

        # noinspection PyUnresolvedReferences
        import pynvml
        pynvml.nvmlInit()

        # noinspection PyTypeChecker
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [0, ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                assert status == pynvml.NVML_P2P_STATUS_OK, \
                    f'GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink'

        # Close NVML
        pynvml.nvmlShutdown()
```

**① 字符串门（`:156`）**。不是查 `cudaDeviceProp` 或 NVML 的 P2P 能力来决定走哪条分支，而是看设备名里
有没有 `'PCIE'`。H100/H800 的 SXM 版本名字是 `NVIDIA H100 80GB HBM3`（实测），**不含 `PCIE`**，
所以整段会被**跳过**——H100 上它是个 no-op。这既是设计上的「宽松」（SXM 默认全互联、无需检查），
也是脆弱点：设备名一旦变了（或被虚拟化改写），判据就失效。源码注释里的 `TODO`（`:155`）也承认
「应该检查所有情况」，即真正的目标语义是「组内所有本地 GPU 都要有 NVLink 连接」，但当前实现
只在 PCIe 名下做。

**② PCIe 的硬上限（`:157`）**。若是 PCIe 卡，直接断言 `group.size() <= 2`——因为 A100 PCIe 的 NVLink
是**两两配对**的，EP 超过 2 就没有全互联可言。这是一个**先于任何 P2P 探测**的粗粒度闸门：它不检查
「具体哪几对连着」，而是直接把问题挡在 EP2。注意这条断言**不依赖 group 是否为真通信组**，只需要
`.size()`。

**③ NVML 生命周期（`:159-161` / `:180-181`）**。`import pynvml` 放在函数体内部（惰性导入，避免非 PCIe
路径白白依赖 NVML），`nvmlInit()` / `nvmlShutdown()` 成对——这是一个**标准的 NVML 资源生命周期**。

**④ 物理设备号（`:163-165`）**。关键的一步：`torch.cuda.current_device()` 给的是**逻辑**设备号
（当前 CUDA_VISIBLE_DEVICES 重排后的下标），而 NVML 的 `nvmlDeviceGetHandleByIndex` 要的是**物理**
设备号。代码用 `CUDA_VISIBLE_DEVICES` 环境变量把逻辑→物理翻译了一遍：
`int(devices[current_device])`。若进程没设 `CUDA_VISIBLE_DEVICES`，默认 `'0,1,2,3,4,5,6,7'`、
`.strip(',')` 去掉首尾逗号。这是**整个函数最容易出错的一行**：CUDA_VISIBLE_DEVICES 是「逻辑设备
→ 物理设备」的映射表，顺序敏感。

**⑤ 跨 rank 汇聚（`:166-167`）**。`physical_device_indices` 是长度为 `group.size()` 的**输出缓冲**，
`dist.all_gather_object` 把每个 rank 的物理设备号写进去——index `i` 对应通信组里的第 `i` 个 rank。
于是每个 rank 都拿到「组内所有 rank 各自的物理 GPU 号」，才能做两两 P2P 检查。注意这里用的是
**object** gather（走 pickle），而不是张量 gather——因为设备号是小整数、不值得开张量通信。

**⑥ 两两 P2P 检查（`:169-178`）**。对组内所有**无序对** `i < j`，用
`nvmlDeviceGetP2PStatus(handle_i, handle_j, NVML_P2P_CAPS_INDEX_NVLINK)` 查询这对 GPU 之间的 NVLink
**能力状态**，必须等于 `NVML_P2P_STATUS_OK`。这是**离散能力**（兼容/不兼容），**不是带宽**；要的是
「这对 GPU 之间有没有 NVLink 直连」，而非「NVLink 有多快」。失败时断言消息带上两个物理设备号，
便于定位。`i >= j: continue` 保证每对只查一次、也不自查（`i == j`）。

**⑦ 引用出处（`:170`）**。源码直接给了 vLLM 的对应实现链接：
<https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438>
（vLLM 用同一套 NVML P2P 检查来决定能否跨卡共享 KV cache / 做 TP）。

---

## 3. NVML P2P 能力 API 与 `pynvml` 的一个绑定坑

NVML 的 `nvmlDeviceGetP2PStatus(dev1, dev2, p2pIndex)` 返回一个枚举，`p2pIndex` 决定「查哪种能力」：

| 常量 | 值（本机 pynvml） | 含义 |
|---|---|---|
| `NVML_P2P_CAPS_INDEX_READ` | `(0,)`（**tuple，见下**） | dev1 能否从 dev2 读 |
| `NVML_P2P_CAPS_INDEX_WRITE` | `1` | dev1 能否向 dev2 写 |
| `NVML_P2P_CAPS_INDEX_NVLINK` | `2` | 两者是否有 **NVLink** 连接 |
| `NVML_P2P_CAPS_INDEX_ATOMICS` | `3` | 是否支持远端原子 |
| `NVML_P2P_CAPS_INDEX_PROP` | `4` | 是否支持 P2P 属性访问 |
| `NVML_P2P_CAPS_INDEX_UNKNOWN` | `5` | 未知 |

返回值 `NVML_P2P_STATUS_OK == 0` 表示兼容；非 0 表示不兼容（`nvmlDeviceGetP2PStatus` 的其余状态
值语义见 NVML 文档）。DeepEP 只查 `NVLINK` 这一项。

**真机发现（本机 pynvml 版本）**：`pynvml.NVML_P2P_CAPS_INDEX_READ` 被导出成了 **tuple `(0,)`**，
而 `WRITE/NVLINK/ATOMICS/PROP` 都是 `int`。把 `(0,)` 传给 `nvmlDeviceGetP2PStatus` 会直接抛
`ctypes.ArgumentError: Don't know how to convert parameter 3`。这是一个**库里只有个别枚举成员
被错误绑定**的坑，好在 DeepEP 用的是 `NVLINK`（`int 2`），完全不受影响。实测验证：`NVLINK` 查询
正常返回 `int 0`，`READ` 查询抛 `ArgumentError`。

对照 `torch.cuda.can_device_access_peer(i, j)`：CUDA 运行时只回答「i 能否访问 j 的显存」这个**布尔**，
**不区分 NVLink / PCIe**。所以 DeepEP 必须借 NVML 才能判定「是不是 NVLink」。实测两者的 8×8 矩阵
完全一致（56/56 对都为真）——在 H100 SXM 上 CUDA 的 peer 可达性和 NVML 的 NVLink 能力是一回事，
但语义上 CUDA 那份无法在 PCIe 机器上区分「peer 可达但走 PCIe」的情况。

---

## 4. 它在 DeepEP 里的位置：域大小与两个消费点

`check_nvlink_connections` 不返回任何东西，它是一道**前置断言**。两个消费点都在 buffer 构造里：

| 消费点 | 行号 | 时机 |
|---|---|---|
| `EPBuffer.__init__` | `deep_ep/buffers/ep.py:325` | 「Check PCIe GPUs」——紧接着才自动选 QP 数 |
| `EngramBuffer.__init__` | `deep_ep/buffers/engram.py:67` | 在算 `NCCL_WIN_STRIDE` 之后、建 shared handle 之前 |

它和真正的域大小探测（`get_physical_domain_size`）是**两件事**：

- `get_physical_domain_size`（`envs.py:136-138` → `comm/handle.py:80-85` →
  `csrc/kernels/comm/context.cpp:52-56`）是**运行时问 NCCL**：`ncclTeamWorld(comm).nRanks` 与
  `ncclTeamLsa(comm).nRanks`，返回 `(num_rdma_ranks, num_nvl_ranks)`，并断言
  `num_ranks % num_nvl_ranks == 0`。这里 `ncclTeamLsa`（LSA = Local SHARP/All-to-all 域，即 NCCL 认为的
  「NVLink 可直连域」）才是权威的「NVLink 域」。
- `check_nvlink_connections` 是**兜底自检**：NCCL 建的 LSA 域在 PCIe 机器上可能给出一个「看似能全互联」
  的域，DeepEP 用 NVML 再确认一遍，并在 PCIe 上直接把 EP 上限压到 2。

换句话说：**NCCL 决定域多大，NVML 自检保证这个域是真 NVLink 全互联**。`allow_hybrid_mode=False` 时
`get_logical_domain_size`（`:59-63`）会把逻辑域塌成 `(1, num_rdma_ranks*num_nvl_ranks)`（第 73 篇），
此时不再有 scaleup/scaleout 两级，也是建立在前者之上的。

---

## 5. H100 实测

实验 `harness/exp_deepep_nvlink_p2p.py`（**直接加载真实 `envs.py` 源码**，`sys.modules` 注入合成
`deep_ep._C`/`deep_ep.comm`）+ `src/nvlink_p2p_matrix.cu`（真机 P2P 带宽矩阵）。**16 PASS / 0 FAIL**：

### Part A：H100 上它是 no-op，且 PCIe 分支可被触发

- 设备名 `NVIDIA H100 80GB HBM3` **不含 `'PCIE'`** → 真机调用 `check_nvlink_connections(group)` 返回
  `None`（不走 NVML、不 all_gather）。
- 负对照：把 `torch.cuda.get_device_name` 伪装成 `'… PCIE'` 且 group 大小 8 → 命中 `:157` 断言
  `PCIe GPUs only have pairwise NVLink connections`。

### Part B：真机 NVML 8×8 P2P 状态矩阵——全互联

`NVML_P2P_CAPS_INDEX_{NVLINK,WRITE,ATOMICS}` 三个能力在 **56/56 个非对角对**上全部 `OK`（即 NVML
的判定就是「满互联」）。矩阵：

```
NVLINK:  行=GPU i，列=GPU j（0=OK）
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    ...（8 行全 0）
```

另发现 `NVML_P2P_CAPS_INDEX_READ` 是 tuple `(0,)`、直接用会 `ctypes.ArgumentError`（§3）。

### Part C：CUDA 运行时视角一致

`torch.cuda.can_device_access_peer(i, j)` 在 56/56 非对角对为真，与 NVML `NVLINK` 状态矩阵
**0 处不一致**。

### Part D：模拟 PCIe 分支（真实函数体）

- group 大小 8 + PCIE → `:157` 断言；
- group 大小 2 + 物理设备号 `[0,1]` + **真实 NVML** → 穿过完整的 all_gather + 两两 P2P 检查、通过
  （H100 上 GPU0↔GPU1 确有 NVLink）；
- 用**假 pynvml** 让 `(0,1)` 这一对返回非 OK → 命中 `:177` 断言
  `GPU 0 and GPU 1 are not connected via NVLink`。并观察到：断言会在 `nvmlShutdown()` **之前**抛出，
  `shutdown` 计数为 0——即**异常路径下 NVML 不会被显式关闭**（进程退出时由驱动回收）。
- `CUDA_VISIBLE_DEVICES` 物理设备号解析的 4 个边界（`'0,1,2,3,4,5,6,7'` / `'0,1,2,3'` / 尾部逗号
  `'0,1,'` 被 `strip(',')` 去掉 / `'4,5,6,7'`）全部正确。

### Part E：真机 P2P 拷贝带宽矩阵

`cudaMemcpyPeerAsync` 8×8 有向带宽（64 MiB × 30 iters，GB/s SI）：

```
dst→      0      1      2      3      4      5      6      7
src0      -  372.7  370.4  372.6  372.8  372.7  372.9  372.8
src1  212.7      -  373.9  374.3  374.5  374.6  376.3  374.7
src2  212.7  374.2      -  374.3  373.5  373.4  376.2  373.7
...
summary: min=212.6 max=376.4 mean=353.6 GB/s
```

**结论**：① 所有 56 个有向对都 **> 212 GB/s**，远高于 PCIe Gen5 x16 的理论 ~64 GB/s，坐实
**每一对都走 NVLink**（NV18 平面、同/跨 NUMA socket 无差异），与第 11 篇「远端 ~368 GB/s ≈ 本地
1/8」量级一致；② 唯一异常是**目的列 `dst0`**：所有源写到 GPU0 都只有 ~213 GB/s，而写到其他 GPU 都是
~373 GB/s。原因不是拓扑（NVML 判定 GPU0 与其余全 NVLink 相连），而是**这台机器被多个自驱任务共享、
GPU0 上正跑着十几个进程**——写带宽受**目的 GPU 的繁忙程度**限制。这正是「P2P 拷贝是有向的、
目的端是瓶颈」的体现，也说明**带宽探测必须和 NVML 能力探测分开**（能力是拓扑属性、带宽是运行时
状态，后者会被邻居任务干扰）。实验用「除最慢目的列外其余对 > 300 GB/s」来稳健判定。

---

## 6. 小结

- `check_nvlink_connections`（`envs.py:146-181`）是 DeepEP 对「节点内 GPU 全互联」假设的**唯一运行时
  自检**：门是**设备名含不含 `'PCIE'`**，PCIe 下断言 `EP ≤ 2`，否则用 **NVML** 对组内所有无序对查
  `NVML_P2P_CAPS_INDEX_NVLINK` 是否 `OK`。
- 它**不是**带宽探测：返回离散能力状态。它和权威的域大小（NCCL 的 `ncclTeamLsa.nRanks`，
  `get_physical_domain_size`）分工明确——**NCCL 定域、NVML 自检**。
- H100 SXM 设备名不含 `PCIE`，故这函数在 H100 上是 **no-op**；本实验用「伪装 PCIE + 真/假 pynvml」
  把它的 **全部三条代码路径**（no-op / `:157` 断言 / `:177` 断言 + all_gather）都真机跑到了。
- 真机拓扑：NVML NVLINK/WRITE/ATOMICS **56/56 全 OK**、`can_device_access_peer` **56/56**、
  P2P 带宽 **> 212 GB/s**（NVLink，非 PCIe）；唯一慢的是被共享任务占用的 GPU0 作为**目的端**时的
  写带宽（~213 vs ~373 GB/s），坐实「P2P 带宽是运行时量、会被邻居干扰」。
- 附带发现一个 **pynvml 绑定坑**：`NVML_P2P_CAPS_INDEX_READ` 在本机版本被导出成 tuple `(0,)`，
  直接调用抛 `ctypes.ArgumentError`；DeepEP 用 `NVLINK`（int）不受影响。

至此，`envs.py` 的四类内容（进程引导、环境工具、守卫/探测、带宽/域探测）全部有独立篇目覆盖。
这层是**架构无关的 host 层**：在 H100(sm90) 上 16 / 16 全过，与目标卡是否 SM100 无关。
