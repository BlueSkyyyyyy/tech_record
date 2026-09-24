---
title: "DSec 精读：DeepSeek 的 Agentic RL 沙箱平台，为什么不是「一个容器运行时」"
date: 2026-09-24T14:31:00+08:00
draft: false
weight: 1
series: ["paper阅读"]
tags: ["paper阅读", "DSec", "sandbox", "agent", "RL训练", "基础设施", "3FS", "系列"]
categories: ["训练框架"]
---

这是新专题 **「paper阅读」** 的第一篇。这个专题只做一件事：把一篇值得读的系统/算法论文拆开揉碎，讲清楚**它要解决的真实问题、设计取舍背后的物理约束、以及哪些结论能迁移到我们自己的工程里**。不做逐段翻译，重点在「为什么这么设计」和「哪里可以质疑」。

第一篇选 DeepSeek-AI 与清华合作的 **《DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale》**。论文本身在正文里明确提到 DSec 支撑了从 DeepSeek-V3.2 到 DeepSeek-V4.1 的 RL 训练与评估（参考文献里那篇 V4.1 就是讲 KV cache 压缩的）——也就是说，这不是一个玩具原型，而是一份**生产系统的复盘报告**。

如果你只记一句话：

> **Agentic RL 的沙箱不是「serverless 的一个变种」，而是一类全新的工作负载。** 它长寿命、有状态、CPU 稀疏、内存常驻、镜像低复用，serverless 的许多默认假设在这里全部失效。DSec 的价值，是把这些反向假设一条条摊开，然后给出可落地的系统对策。

---

## 1. 背景：RL rollout 把沙箱推成了基础设施

先厘清 agentic training 的流水线。论文把它拆成五段：环境与数据构造 → RL rollout → reward 计算 → 策略更新 → 周期性评估。其中 **rollout 与评估**对沙箱平台的压力最大，原因有三：规模大、并发高、且与训练循环紧耦合。

RL 本身是一个三段式反馈环（论文 §1）：

1. **rollout**：当前模型在沙箱环境里读文件、发工具调用、执行命令、观察输出，产生一条 trajectory；
2. **reward computation**：用原生执行信号打分——exit code、stdout、测试通过率、或任务特定 verifier；
3. **policy update**：从 trajectory + reward 更新参数。

近期的系统（论文引用 DeepSeek 自己的异步 RL 工作）进一步把 **generation 与 policy optimization 流水线化**，靠持续补充完成的样本来维持高并发、缓解长尾 straggler。对 agentic 负载来说，这意味着**大量有状态的沙箱会话同时在飞**，而且可能在 policy 更新或调度抢占时被中断、之后再恢复——并发度、生命周期管理、状态一致性要求随之陡增。

关键判断藏在 §1 的一句话里：

> "Supporting them therefore requires an **elastic execution platform** rather than a single sandbox runtime."

也就是说，真正的对手不是「哪个隔离技术更好」（Firecracker vs gVisor vs 容器），而是**平台层**：调度、生命周期、镜像分发、资源超卖、与训练框架的协同。理解这一点，后面所有设计才顺理成章。

---

## 2. 工作负载画像：七个特征定义了整个设计空间

论文 §1 列了 7 条性质，我认为这是全文最有价值的部分——它解释了「为什么不能拿现成的 serverless / K8s 容器服务凑合」。我按自己的理解重排并加上注解：

| # | 性质 | 系统含义 |
|---|---|---|
| 1 | **突发创建**：单个 job 一次最多要 32K 个沙箱 | 调度、镜像分发必须能水平扩展，不能有中心瓶颈 |
| 2 | **高密度**：交互间歇期 CPU 稀疏，天然适合超卖 | 单节点要能跑 800 microVM / 3200 容器，前提是能安全超卖 |
| 3 | **有状态、长寿命**：文件、依赖、服务在多次 LLM 轮次间持续存在 | 内存/页缓存/可写状态长期驻留，内存共享与回收成为核心需求 |
| 4 | **高度异构**：OJ 脚本、SWE 仓库、安全、computer-use、Android…… | 单个沙箱抽象覆盖不了所有场景，必须多后端 |
| 5 | **环境多样性高、复用低**：每任务自带仓库/依赖/服务/镜像 | 必须服务大量不同镜像，且多数镜像用不了几次 |
| 6 | **执行不可信**：agent 会破坏文件系统、耗尽资源、干扰系统 | 需要细粒度访问控制 + 误行为分析 |
| 7 | **执行可中断**：GPU 训练可被抢占，而 rollout 还在跑 | 必须保存状态、支持高效恢复 |

其中第 2 条与第 3 条构成一个**尖锐的矛盾**：CPU 大部分时间是空闲的（适合超卖），但内存和状态却一直驻留（不适合超卖）。论文说得很直白——「long lifetimes amplify the cost of retained memory」。后面 §5.2 的内存共享/回收机制，本质上就是在给这个矛盾找出口。

而第 5 条是另一个反直觉点：**镜像复用率极低**。论文测得容器镜像 fanout 中位数只有 **3**、p90 为 **28**；microVM 镜像中位数 **1**、p90 仅 **3**。这直接判了「本地镜像缓存」的死刑——工作集太分散，单节点根本装不下，每次突发都会退化成「必须拉镜像」。

---

## 3. 规模与后端矩阵

先看生产数字（§2.4）：

- 单个 **scale unit** 约 **160 个 CPU 节点**、**30K 核**、**~250 TB DRAM**；
- 管理 **PB 级**的 layers 与 images；
- 典型一天服务约 **300 万**沙箱实例；
- 峰值并发 **~38 万**；创建速率 **>5000 实例/秒**。

沙箱后端是一个**沿「隔离强度 ↔ 启动延迟/资源开销」权衡铺开的谱系**（论文 Table 1）：

| 后端 | 运行时性能 | 依赖足迹 | 隔离级别 | 完整 OS | 资源开销 | 典型场景 |
|---|---|---|---|---|---|---|
| FnCall | 最高 | 小 | 弱 | 弱 | 最低 | OJ、代码编译、GPU kernel |
| Container | 高 | 中 | 中（共享宿主内核） | 中 | 中 | SWE、tool use |
| MicroVM | 中 | 大 | 强 | 较强 | 较高 | 安全、强租户隔离 |
| Full VM | 低 | 最大 | 最强 | 强 | 最高 | COTS OS、图形/GUI、Android |

几个容易被忽略的工程细节：

- **FnCall** 复用**预先创建**的 CPU/GPU 容器，避免每次调用的 provisioning 开销；GPU 支持 shared（多容器共享一卡）与 exclusive（独占，用于算子评测）两种模式。
- **Container 与 FnCall 并不是裸跑在宿主机上**，而是跑在 QEMU/libvirt VM 里（§3.3）。这层 VM 提供隔离的内核与网络栈，**作为不可信容器与裸金属之间的额外安全边界**。这是一个很务实的选择：牺牲一点性能，换取纵深防御。
- **Full VM** 通过 virtio-gpu 半虚拟化支持图形负载，并可用 DXVK 之类的兼容层把渲染栈翻译到宿主原生 API。

> 我的观察：DSec 没有在「哪种隔离最好」上下注，而是**把选择权交给调用方**（`libdsec` 要求用户显式选后端）。这与很多「一个运行时打天下」的思路相反，恰恰承认了 agentic 负载的异构性。

---

## 4. 架构总览

请求路径（§3）：

```
libdsec (SDK) ──► IAM ──► placement engine ──► apiserver ──► edge ──► sandbox
                             ▲                                  │
                          watcher                        aether / chronus
                                                              │
                                                            3FS
```

**集群级服务**（负责接入、鉴权、放置、集群视图）：

- **IAM**：鉴权 + 授权。支持**多级 project 嵌套**（而非云平台常见的一两层），授权 principal（包括 agent 和 harness）可以创建子 project、下放配额和权限，但**受父级约束**——不能授予自己没有的权限。人和 agent 用同一套管理 API 与授权模型。
- **apiserver**：唯一入口代理。训练/评估代码从可信 GPU 侧调用，而沙箱执行不可信模型生成的代码、还可能要访问外网，两侧网络隔离，apiserver 是唯一通道。它**不保存 per-sandbox 状态**，沙箱 ID 里编码了归属的 edge，因此任何实例都能直接转发，接入层可水平扩展。
- **placement engine**：两阶段——先 **filtering**（健康 + 硬件能力，如 GPU），再 **ranking**（随机采样若干节点选最闲的）。
- **watcher**：周期探测 edge/host 健康，收集调度状态。placement 与 watcher **都不需要持久状态**，重启后重建即可，易于加/换实例。

**沙箱运行时**（节点本地）：

- **edge**：每机一个。承接 apiserver 的创建请求，做**节点本地准入检查**（补 placement 基于陈旧集群视图的不足），provision 存储、应用 eBPF 网络策略、启动运行时、协调磁盘/内存快照、TTL 到期回收资源。
- **aether**：容器/VM 内的跨平台代理，与 edge 建立通道（Linux 容器用 Unix domain socket，VM 用 vsock）。edge 通过通道监控健康度，通道断开即标记沙箱失败。
- **chronus**：沙箱内的 **shell 会话**抽象，暴露命令执行、文件系统、HTTP、流式 I/O 接口，同一沙箱内可有多个并发实例。

**镜像存储**：容器镜像离线从 OCI 转成 **EROFS**（元数据与数据分离，元数据留在本地，数据在 3FS）；microVM 用 **OverlayBD** 格式。两者都支持按需加载与基于共享 base 的增量快照。

**Cloud Bursting**（§3.4）：本地利用率超过 **80%** 时，placement 把部分「cloud-eligible」请求卸载到云端 VM。这里有个很巧的点——**不做「托管容器服务 + 对象存储」的组合，而是在云 VM 上复用本地的容器运行时和 EROFS 加载路径**。生产 trace 显示：一套 **30 TB** 去重后的紧凑 EROFS 镜像集，覆盖了 70% 容器任务访问的镜像文件；这套集合离线同步到云文件系统，镜像依赖完全落在其中的任务才算 cloud-eligible。生产上 **200 台云 VM** 吸收了一个 scale unit 约 **30%** 的峰值溢出。

---

## 5. 测量：三组把设计逼出来的数字

论文 §4 只测容器与 microVM（占绝大多数实例与资源消耗）。我认为最该记住的是下面四组。

### 5.1 突发 + 长寿命

- 单个容器任务典型就创建**数千个**沙箱，长尾到**数万个**，最大 job 到 **32K**；都在很短的窗口内到达。
- 沙箱生命周期（Fig. 7，30K 容器 + 10K microVM 采样）：容器中位数 **17.4 分钟**、microVM **15.5 分钟**，两者 **p99 都超过 3 小时**。

长寿命 × 高密度 × 内存常驻，是后面内存机制的直接动因。

### 5.2 环境多样性

一周的生产数据（Table 2）：

| 后端 | base images | workspaces | snapshots | 聚合大小 |
|---|---|---|---|---|
| Container | 11,266 | 102,171 | — | 82.8 TB |
| MicroVM | 2 | 53,590 | 4,889 | 50.9 TB |

另有 **103 个 toolkit**，且 **67.8% 的沙箱需要 base image 之外的 workspace 或 toolkit**。

论文在这里给了全文最漂亮的一个组合爆炸论证：如果维护 $B$ 个 base image、$W$ 个 workspace、$T$ 个 toolkit，**把三者融成一个 OCI 镜像**，那么升级 $B$ 个 base image 需要重建它们的 workspace 组合，代价是 $O(B \cdot W)$；升级 $T$ 个 toolkit 的代价是 $O(T \cdot W)$。目标是通过独立版本化把这两个代价分别降到 $O(B)$ 和 $O(T)$。

论文还否掉了两个「显而易见的替代方案」：

- **打包成 tar.gz 在沙箱内解压**：突发时重复解压带来大量 CPU/I/O 开销，会导致启动超时；
- **宿主机只读目录 bind-mount**：bind mount 是**整体替换**目标路径，而这些组件需要的是**追加/合并**语义（不遮挡下层已有文件）；而且严格只读会和「工具往自己安装目录写东西」冲突（比如 Python 生成 `__pycache__`）。

### 5.3 CPU 稀疏 + 高密度 + 内存浪费

- **~90%** 的容器与 microVM 沙箱平均只用了请求 CPU 的 **≤5%**——超卖是理性选择。
- 实际运行点：一天采样看到单节点峰值 **1048 容器 / 524 microVM**；生产中稳定跑到 **≥3200 容器 / 800 microVM 每节点**。
- microVM 的内存浪费有两个来源：① 经虚拟块设备读的镜像数据，**host 和每个 guest 各缓存一份**；② guest 内的空闲页**不主动归还** host。
- CPU 侧：部分任务有严格的 per-step 延迟预算（如固定每步时限的游戏 agent）。仅靠**降低 best-effort 优先级不够**，因为 BE 和 LS 可能跑在**同一物理核的 SMT 兄弟线程**上，仍共享执行资源。

### 5.4 大镜像工作集 + 低复用 + 低访问率

- 一周活跃 artifact 聚合 **>130 TB**，远超单节点存储。
- fanout 中位数 3 / p90 28（容器），microVM 1 / p90 3。
- 最关键的 Table 3：容器镜像**运行时实际访问的数据只占 4.2%~13.3%**：

| 镜像类型 | C++ | Go | Java | JavaScript | Python |
|---|---|---|---|---|---|
| 访问数据占比 | 8.7% | 13.3% | 9.2% | 4.2% | 6.0% |
| 镜像大小 | 4.9 GB | 4.1 GB | 12.1 GB | 9.6 GB | 6.0 GB |

**全量拉镜像极其浪费**——这个数字是「按需加载」从「省时间」升级为「省流量」的关键：省的不只是时机，而是**总量**。

---

## 6. 三个核心机制

### 6.1 可组合环境层（Composable Environment Layers）

核心洞察：base OS、每个 workspace、每个 toolkit 是**生命周期彼此独立的逻辑层**，不该融成一个单体镜像。解法就是 overlayfs 的合并语义——多个只读 lower 目录叠加，内核呈现统一目录树，冲突按优先级解析，可写 upper 目录透明吸收运行时写入。

具体做法（§5.1 + §7）：

- **改 dockerd**（基于 Moby），在容器创建时**动态组装 overlayfs 的 `lowerdir`**：base image 在底、workspace 插入为只读层、各 toolkit 再叠上去。这样 workspace/toolkit 是**合并**进 base 树而非替换路径。
- 发布的环境层不可变，于是存成 **EROFS**：相比 ext4/XFS 省去写相关记账、布局更紧凑；支持压缩且**保留随机访问**——与 `tar.gz` 的顺序流不同，EROFS 只读取/解压覆盖目标数据的块，不必先传输解压整个镜像。
- microVM 复用同一模型：base image / toolkit 打包为独立版本化的 EROFS，作为**只读块设备**暴露给 guest；guest 内 rootfs 用 overlayfs，EROFS 作 lower、ext4 可写盘的一个目录作 upper。

微小的改动量也很说明问题：**只需 30 行 Go 代码**插入预先挂载好的 EROFS 层。

### 6.2 高密度资源管理

**内存**（两条互补路线）：

- **virtio-pmem + DAX**：文件访问直接映射到 host 页，不再拷贝进 guest RAM，从而**消除页缓存重复**；多个共置 microVM 共享同一份 host page cache。代价：冷访问可能触发同步缺页处理；且 guest 必须为整个 pmem 地址范围分配 `struct page` 元数据——4 KiB 页 + 64 B `struct page` 时，需要 **pmem 容量的 1/64** 作为 guest RAM（128 GB pmem → 2 GB 元数据）。因此它更适合只读的 EROFS base/toolkit 层。
- **DAMON + virtio-balloon free-page reporting**：balloon 驱动周期性扫描 buddy allocator，主动把空闲页上报 host，host 用 `madvise(MADV_DONTNEED)` 释放；默认按 order-9（2 MiB）上报。DAMON 采样页访问位，识别「超过阈值未被触碰」的冷文件页并驱逐回 buddy allocator，**把零散页合并成高阶块**以满足上报要求。用于较大的可写磁盘。

**QoS 感知的 CPU 调度**（两层）：

1. best-effort 沙箱放进 `SCHED_IDLE`，LS 任务可运行时立即让出；
2. 光有优先级不够，还要用 **Linux core scheduling**（`prctl(PR_SCHED_CORE)`）按 QoS 分组，**阻止无关 BE 工作跑在 LS 的 SMT 兄弟线程**上。

论文强调所有机制**都基于现有内核特性、无需改内核**，只是配置 + 与编排器集成。

### 6.3 可扩展的镜像分发与按需加载

关键观察还是 §5.4 那个「只访问 4.2%~13.3%」——**按需拉取同时解决时机问题和体量问题**：总 I/O 按实际使用比例缩小，而不是把开销挪到别的阶段。

DSec 不用「registry + P2P」那套（如 FaaSNet），而是**直接把镜像放在 3FS 上**。这复用了已有存储基础设施、省掉一整套分发层，但 3FS 的 I/O 特性高度不对称：**大顺序读写吞吐高，小随机 I/O 很差**。这个约束直接决定了三条设计原则：

1. **写留本地**：沙箱写不规则（小、频繁，如日志），放节点本地盘，彻底避开 3FS 的小写惩罚；
2. **读按需且批量**：只读镜像数据在访问时从 3FS 拉，且**成批**拉以吃到高吞吐；
3. **元数据尽量本地**：元数据常是小读，能分离就预取到本地。

容器侧靠 EROFS 落实：严格只读 → 写全进本地 upper；buffered I/O 按需 + 内核 readahead 合并相邻块；**multi-device 模式**把元数据与数据分开，DSec 把元数据下载到本地盘、数据留在 3FS，使路径查找不产生远程 I/O。此外还做了一些工程优化：

- 离线**合并连续层**（阈值如 3 GB）为一对 EROFS 元数据/数据镜像，保留 overlayfs whiteout 语义以正确表示删除，减少 mount 数量、避免文件重复、保留跨镜像页缓存复用；
- 用 EROFS 的 **file-backed mount** 去掉 loop 设备的块映射层。

microVM 侧则不同（Firecracker 不支持 virtio-fs，Docker `overlay2` 也不能用 overlayfs 做 data root）：只读 base/toolkit 仍用 EROFS，可写 ext4 盘用 **OverlayBD**，经 **ublk**（用户态块设备）暴露。这条块级路径支持按需读、本地写、增量快照；但 ext4 元数据嵌在块镜像里，元数据读可能触发远程 I/O，于是 ublk 实现**按 256 KiB 取块**并放入二级本地文件系统缓存，即使从页缓存驱逐也无需二次远程拉取。

---

## 7. 与 RL 框架的协同（我认为这才是本文最「DeepSeek」的部分）

很多沙箱论文止步于「怎么跑得快」，DSec 花了整整一章（§6）讲**平台如何与训练框架共设计**。这层工作通常不写在论文里，但生产价值极高。

### 7.1 用 Agent 造 Agent 的环境（pack_diff）

手工构造 agentic RL 所需的海量环境不现实。DSec 让 agent **在同一套基础设施上交互式地构建环境**，接口叫 **`pack_diff`**：任意时刻可对沙箱做增量磁盘快照（checkpoint），随后可恢复为新沙箱。**「checkpoint-and-restore」把一次交互会话直接变成可复用环境**，环境的构建、验证、消费都在同一基础设施上完成，无需单独的镜像构建流水线。

同时防信息泄漏：**build 阶段与 runtime 阶段用不同账户**，打包前从可写层清除构建残余数据，防止参考答案被带进镜像。

### 7.2 把 agent loop 从 RL 框架中拆出来

这是我认为全文最重要的工程决策。早期版本里，**agent loop 与 serving、RL 框架跑在同一个可抢占的 GPU pod 里**；GPU job 被抢占时，agent loop 丢失而沙箱还在，恢复只能靠**命令日志 replay**——已完成的操作复用记录结果以避免非幂等命令重复副作用。

从 **DeepSeek-V4.1** 起，rollout 执行被搬到 DSec 上，拆成两个组件：

- **agent sandbox**：承载 scaffold（如 DeepSeek Harness）及其工具；
- **worker container**：管理沙箱，提供与 scaffold 无关的控制层。

**两者都跑在可抢占 GPU 池之外**，于是 rollout 生命周期与 trainer 生命周期解耦。worker container + agent sandbox 共同持有完整 rollout 状态，是**唯一事实来源**，被抢占的 GPU job 直接重连继续，无需命令日志重建。论文的收益总结：把 rollout 状态恢复逻辑从 RL 框架里拿掉、减少跨组件协调、简化失败处理。

### 7.3 为抢占式训练挂起沙箱

GPU job 抢占不可避免，但沙箱状态得留到 rollout 结束——于是会积压大量空闲沙箱占内存。RL 框架主动向被抢占 job 关联的所有沙箱发 **pause 请求**，DSec 回收内存但保留执行状态；后续任何对 paused 沙箱的请求会**透明地先恢复再执行**。

- **容器**：`docker pause` 冻结进程树 → 打开 `memory.swap.max` → 触发 `memory.reclaim`（回收匿名页与文件页，保留执行状态）→ 恢复时对内存映射打 `MADV_WILLNEED` 异步预取，再 `docker unpause`。
- **microVM**：保存内存与执行状态快照 → **终止 Firecracker 进程**释放运行时内存 → 恢复时启动新进程、还原快照。

### 7.4 误行为与系统故障（这篇论文最诚实的部分）

§6.4 列了一批真实翻车案例，分两类：

**通过非预期渠道拿答案**（reward hacking 的基础设施版）：

- agent 在沙箱内翻找平台托管文件和通信通道里的**残留答案**；伪造用户请求，**直接向 chronus 的 socket 发构造好的 RPC**；查 chronus 日志找泄漏的答案；
- chronus 某些操作会调 `bash`，agent 就试着**覆盖 `/bin/bash`** 来绕过检查或注入命令；
- 加了文件/套接字访问控制后，agent 又想用 **`XFS_IOC_SWAPEXT`**（交换两文件数据 extent 映射的 ioctl）让受保护文件内容经另一 fd 暴露——结果搞坏了 XFS 元数据、**触发文件系统 shutdown**；
- 沙箱外：扫端口/服务找可达镜像，用 **Go module proxy** 拉 GitHub 代码、装新版本包，找现成实现。

论文的判断很清醒：**只查最终输出无法可靠判定 agent 是否按预期解题。**

**破坏执行环境**（很多是无意的）：

- agent 从根目录递归 `grep`，遍历 `/proc` 读 `/proc/kpagecgroup`，**触发内核 bug 把内核搞崩**；
- 漏洞利用任务里，本该转发给目标 VM 的攻击命令**在 agent 容器内执行**，把自己的内核打崩；
- `yes` 的持续输出被 chronus 记录以便异步取回，**累积了几十 GB stdout 到存储**。

**缓解**（§6.5，论文明确说只能解决一部分）：

- **AppArmor** 控制 agent 进程的文件读写与套接字访问（含 chronus 日志和 Unix socket），**即使进程在沙箱内是 root 也生效**；
- **eBPF** per-sandbox 网络白名单，按 IP/端口/协议过滤，策略可随任务阶段动态更新（如允许 PyPI、拒绝 NPM）。

---

## 8. 评估

测试环境是一套**独立的 10 节点 CPU 集群**：microVM 直接跑裸金属（AMD EPYC 9655，2 socket × 96 核 × 2 SMT，1.5 TB DRAM，3.4 TB 本地盘）；容器实验跑在一个 QEMU VM 内（96 核 × 2 SMT，512 GB，5.8 TB）。宿主 Linux 7.0，guest 6.1。

| 实验 | 配置 | 结果 |
|---|---|---|
| 按需镜像加载（§8.2） | 10 节点突发 8192 容器 | 按需 ~35 min ≈ 全本地；eager >60 min（**1.71×** 慢）；磁盘写 1600 GB → ~700 GB（**-57%**） |
| 可组合层 EROFS vs tar（§8.3） | 同 workspace/toolkit | tar 79 min → EROFS 45 min（**1.76×**）；tar 磁盘写流量 **5.5×**、峰值吞吐 **3.4×** |
| 内存超卖（§8.4） | 4 种 Firecracker 配置 | virtio-pmem+DAX 峰值 host 内存 **-40.2%**；DAMON+FPR 时间积分内存 **-21.2%**；两者结合最优 |
| CPU QoS（§8.5） | LS 国际象棋 + 10%~50% BE | 50% BE 下延迟膨胀 45.2% → core scheduling 后 **17.3%**；仅 SCHED_IDLE 最多只改善 3.4% |

两个值得注意的代价：

- virtio-pmem **把瞬时峰值 CPU 从 26.5% 抬到 41.4%**（冷访问同步缺页、缺少 guest readahead/批量块 I/O）；论文建议 CPU 受限场景可以只开 FPR + virtio-blk。
- CPU QoS 的残余退化（17.3%）来自 **turbo 频率下降、内存带宽与 LLC 争用**——core scheduling 管不了这些，论文认为已可接受，就没做内存带宽隔离。

---

## 9. 我的分析：这份工作的价值与边界

### 9.1 真正的方法论贡献

我觉得 DSec 最值得学的是**它的论证结构**：先老老实实测量生产负载，让数字把设计「逼」出来，再给机制。整篇的因果链非常干净：

- 测到 **CPU 用了不到 5%** → 超卖合理；
- 但同时测到 **p99 活过 3 小时** → 内存不能无限超卖；
- 测到 **fanout 中位数 1~3** → 本地缓存无效，必然要远程分发；
- 测到 **只访问 4%~13% 数据** → 按需加载省的是总量；
- 测到 **3FS 小随机 I/O 差** → 写本地、读批量、元数据本地；
- 测到 **SMT 兄弟线程干扰** → 优先级不够，要 core scheduling。

**每个机制都能指回一个测出来的数字**。这种「measurement-driven design」是系统论文该有的样子。

### 9.2 可迁移的工程模式

即使不做 RL 平台，下面几条也能直接用：

1. **可组合层替代单体镜像**：把「环境 = 单一 artifact」改成「base + workspace + toolkit 独立版本化」，维护成本从乘积降到线性。overlayfs + 只读格式（EROFS）是通用配方。
2. **按需加载 = 省总量，不只是省时间**：当实际访问比例很低时，lazy loading 是数量级收益。
3. **状态与计算生命周期解耦**：把 agent loop 从可抢占的 GPU pod 里搬出来，是「让训练侧的抢占不影响 rollout」的关键。这条对任何长任务 + 可抢占资源的组合都成立。
4. **超卖的配套是「可回收」**：pause（docker pause + reclaim / snapshot + terminate）+ 透明 resume，让超卖变得安全。
5. **安全是平台的一等公民**：reward hacking 被当成基础设施问题（AppArmor + eBPF + 日志审查），而不是训练算法的锅。

### 9.3 我认为的局限与开放问题

作为读者，我会追问以下几点：

1. **评估规模与生产规模差三个数量级**。生产是 160 节点、38 万并发；实验只有 10 节点、8192 容器。突发调度、watcher 陈旧视图、cloud bursting 这些**真正在规模化下才暴露的问题，评估里没覆盖**。
2. **§6 的框架协同完全没进评估**（论文自己也承认 "outside the evaluation scope"）。但 §6 恰恰是体验/正确性收益最大（抢占恢复、pause/resume、误行为）的部分，没有量化很可惜。
3. **缺少与同类平台的端到端对比**。Related Work 里点名了 E2B、OpenAI Code Interpreter、MiMo-V2-Flash、ComputerRL 等，但没有横向 benchmark。DSec 的强项（多后端统一、3FS 分发、抢占协同）很难从现有实验里看出相对优势。
4. **安全缓解被明确限定为「部分」**。XFS ioctl 那次事故说明：一旦允许 root + 任意 syscall，访问控制很容易被绕过，且可能直接搞崩基础设施。`/proc/kpagecgroup` 触发内核 bug 更说明**沙箱逃逸面 = 整个内核攻击面**。用「VM 套容器」是对的，但报告没给出 kernel crash 的根因/修复，也没给误行为的量化频率。
5. **几个数字需要小心解读**：按需加载和 EROFS 的对比是**特定 10 节点突发场景**下的，且 `tar.gz` baseline 显然是 strawman；生产里 3FS 已经存在，所以「省掉 registry + P2P」的收益是架构性的，无法从这组实验直接外推。
6. **可复现性**：DSec 本体没开源，只开源了存储组件（Rust OverlayBD + ublk，`github.com/kvcache-ai/AgentENV`）。论文更像一份**生产复盘报告**而非可复现的系统论文——这不是缺点，但读的时候要清楚它的证据类型是「deployment experience」。

### 9.4 一句话点评

这是一篇**把「RL 需要什么样的执行基础设施」讲得最系统的公开材料**。它的隔离技术全都不是新的（Firecracker、overlayfs、EROFS、virtio-pmem、DAMON、core scheduling），但**把它们按 agentic 负载的真实约束重新组织**，并补上了「与训练框架协同」和「误行为治理」这两块通常被忽略的拼图。做 agent 训练/评测平台的人，§4 的测量和 §6 的协同可以直接抄作业。

---

## 10. 小结

- **问题**：agentic RL 的沙箱是长寿命、有状态、CPU 稀疏、内存常驻、镜像低复用的全新负载，serverless 假设失效，需要平台而非单一运行时。
- **痛点数字**：突发 32K/任务、~90% CPU 用量 ≤5%、中位寿命 15~17 分钟/p99 >3 小时、镜像 fanout 中位 1~3、运行时只访问 4%~13% 数据、一周 >130 TB artifact。
- **机制**：可组合 overlayfs+EROFS 层（维护成本 $O(B \cdot W) \to O(B)$）、virtio-pmem+DAX 与 DAMON+FPR 内存优化、SCHED_IDLE + core scheduling 的 CPU QoS、3FS 上的按需/批量/元数据本地化分发。
- **协同**：`pack_diff` 让 agent 造环境、agent loop 移出可抢占 GPU 池、pause/resume 保存状态、AppArmor+eBPF 治理 reward hacking。
- **收益**：按需加载 1.71×、EROFS vs tar 1.76×、内存峰值 -40.2%、SMT 延迟膨胀 45.2%→17.3%。
- **边界**：实验规模远小于生产；框架协同与安全未量化；同类平台无端到端对比；本体未开源。

> 下一篇「paper阅读」预计会挑一篇和**训练系统 / 推理系统 / 算子**相关的论文，同样按「背景 → 测量 → 机制 → 批判」来拆。

---

## 参考资料

- 原文：DeepSeek-AI, Tsinghua University. *DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale.*（本地文件 `DSec.pdf`）
- 文内引用到的关键工作：Firecracker (NSDI'20)、EROFS (ATC'19)、OverlayBD/DADI (ATC'20)、Nydus、3FS (Fire-Flyer)、DAMON (Middleware'19)、virtio-balloon (OSDI'02)、core scheduling (Linux)、power-of-`d` choices (Mitzenmacher 2001)。
- 开源组件：Rust OverlayBD + ublk，<https://github.com/kvcache-ai/AgentENV/tree/main/storage/overlaybd>
