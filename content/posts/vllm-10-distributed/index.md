---
title: "vLLM 研读（十）分布式：TP/PP/DP/EP 与 Executor 的进程账"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 11
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "张量并行", "流水并行", "数据并行", "专家并行", "Executor"]
categories: ["推理框架"]
---

4 张 H100 上 `vllm serve -tp 4` 会站起 6 个进程；8 张卡上 `vllm serve -tp 2 -dp 4` 会站起 17 个。在调度器眼里这两种部署没有差别——它只认 `max_num_batched_tokens` 和 KV block——但卡与卡之间跑的东西完全不同：前者一步要做 160 次 all-reduce、每次都全员到场；后者每个 DP rank 各跑一份完整模型，一步只在少数几个标量上碰头。

先记住一个画面：**rank 是一根轴上的坐标，Executor 是把这个坐标翻译成"谁去算、算完谁把结果交回来"的那一层**。TP 和 PP 切同一份模型（`world_size = TP × PP`，同一个 EngineCore 管），DP 切请求（每 DP 一个 EngineCore，各自一份完整权重），EP 切 MoE 专家（组跨 DP×TP，PCP 也参与）。四种切法正交叠加，不互相替代。

这一篇回答三个问题：四种并行各付什么代价、算不算得出来？进程与 rank 怎么数、数错了会怎样？Executor 那几个实现到底差在哪？

## 1. 四种并行：切什么、通信什么

| 并行 | 切什么 | 通信原语与位置 | 一步的量（70B、256 token，假设见 1.1） | 适用场景 |
|---|---|---|---|---|
| TP | 层内权重（列切 + 行切）、head 维 | 每层 2 次 all-reduce，**层内、在关键路径上** | 160 次 × 4 MiB | 单机 NVLink、模型装不下单卡 |
| PP | 按层切 stage | stage 间 p2p（`isend`/`irecv`），一步一次 | 每步每个 stage 边界一次 hidden 传递 | 跨机、层数多、显存要摊 |
| DP | 请求 | 基本不通信，只在 finish-sync / MoE 波次上碰头 | 每 N 步一次小 all-reduce | 吞吐、独立副本、EP 的底座 |
| EP | MoE 专家 | all-to-all（dispatch/combine），层内 | 每 token 每 MoE 层 2×topk×hidden | 大 MoE、多机放宽 EP |

### 1.1 账本：TP=8 时每层 all-reduce 的字节与时间

假设写清楚：Llama-3-70B 类模型，hidden = 8192、80 层、bf16；每层 2 次 all-reduce（attention 的 `o_proj` 之后、MLP 的 `down_proj` 之后），payload 是 `num_tokens × 8192 × 2 B`；H100 SXM 的 NVLink 4 标称 900 GB/s 双向，即单向 450 GB/s；ring all-reduce 每卡搬运 $2(N-1)/N \cdot S$ 字节；按链路的 70% 可达效率折算。

取 256 个 token，单次 payload：

$$S = 256 \times 8192 \times 2 = 4\ \text{MiB}$$

$N=8$ 时每卡每次搬 $2 \times 7/8 \times 4 = 7$ MiB，光速下界 $7\ \text{MiB}/450\ \text{GB/s} = 16.3\ \mu s$，按 70% 折算约 $23\ \mu s$。一步 160 次，**约 3.7 ms 纯通信**。同一时刻读一遍权重的时间下界是 $140\ \text{GB}/8 / 3.35\ \text{TB/s} = 5.2$ ms。也就是说 decode 一步里通信占了四成——这就是 vLLM 要在 all-reduce 上做三层优化的原因，也是 TP 度加上去收益会掉的原因：

| 配置 | 每卡每次 AR 字节 | 一步通信 | 一步读权重 | 一步合计 |
|---|---|---|---|---|
| TP=4 | 6 MiB | 3.2 ms | 10.4 ms | 13.6 ms |
| TP=8 | 7 MiB | 3.7 ms | 5.2 ms | 8.9 ms |
| TP=16（跨 2 机） | 7.5 MiB，一半出网 | 7.2–14 ms | 2.6 ms | 10–17 ms |

TP 从 4 翻到 8 只快了 1.5 倍，翻到 16 反而更慢。原因在表里：**payload $S$ 与 TP 度无关**（行切之后每个 rank 拿到的仍是完整 hidden 的部分和），而每卡的权重读取量按 $1/N$ 缩。通信/计算比随 $N$ 线性变差，所以 TP 有一条天花板。跨机那行假设每个 GPU 一块 400 Gb/s IB 网卡（50 GB/s 单向）、两跳串行约 87 µs；即便让 NCCL 把节点内和出网两跳重叠、按一半算，160 次也要 7.2 ms。

换个归一化更好记：**每 token 每层 16 KiB**，80 层两次就是每 token 2.5 MiB 的 all-reduce 流量。batch 变大能摊薄权重读取，摊不薄通信——解 $2.5\ \text{MiB} \cdot B / 450\ \text{GB/s} = 5.2\ \text{ms}$ 得 $B \approx 900$，超过这个 batch，TP=8 单机就进入通信主导区。

### 1.2 PP 的账不一样

PP 的通信量只有 TP 的几十分之一：一步里每个 stage 边界传一次 hidden 状态（`256 × 8192 × 2 = 4 MiB`），跨机一次 ~80 µs。贵的地方不是带宽而是**并发**。

V1 的 PP 不是训练那套 1F1B 微批。它是**按 step 流水**：`max_concurrent_batches` 被设成 `pp_size`（`vllm/config/vllm.py:605`-`615`），EngineCore 的 batch queue 因此能排 `pp_size` 个 batch（`vllm/v1/engine/core.py:221`），每个 stage 同时在不同 step 上工作。worker 侧是异步收发：非首 stage 先 `irecv`（`vllm/v1/worker/gpu_worker.py:1530`）拿到 `AsyncIntermediateTensors`，算完非末 stage 再 `isend`（`:1566`），**下一 step 开头才 wait 上一 step 的 send**（`:1521`）。所以气泡不是每层都有，而是启动/排空时的 `pp_size` 步，以及每步一个 send 的在途延迟。

代价记两条：一是同时在飞的 token 上限变成 `pp_size × max_num_batched_tokens`（`vllm/config/vllm.py:617`-`626`，对滑窗/可回收 KV 的模型直接进 KV 预留）；二是 **PP 与 async scheduling 不兼容**，V1 model runner 下 `pipeline_parallel_size > 1` 会被判成 `v1_async_sched_incompat`（`vllm/config/vllm.py:1703`），默认路径于是把 async scheduling 关掉（`:1749`-`1755`）。省了带宽，赔了 CPU/GPU 重叠。

### 1.3 DP 与 EP

DP 每卡一份完整模型，吞吐随卡数线性涨，代价是显存——70B bf16 要 140 GB，单卡放不下就没得谈。MoE 模型上 DP 不是"完全独立"：只要有一个 rank 还有活，所有 rank 都要跑前向（空的跑 dummy），这个波次由 DP coordinator 协调（`docs/serving/data_parallel_deployment.md:15`）。

EP 是 MoE 专属：attention 仍按 TP/DP 走，专家权重按 EP 度切，token 靠 all-to-all 去它该去的专家。DeepSeek-V3 类配置（hidden 7168、58 个 MoE 层、256 专家、每专家 3×7168×2048 参数）下，fp8 专家权重约 $256 \times 44\ \text{MB} \times 58 \approx 653$ GB，EP=8 摊到 82 GB/卡、EP=16 摊到 41 GB/卡——这是"671B 塞进 8 张 H200"的全部秘密。代价是流量：topk=8、bf16 时每 token 每 MoE 层 dispatch 就是 $8 \times 7168 \times 2 = 112$ KiB，加上 combine 翻倍，58 层合计每 token 约 12.7 MiB，是 TP 那 2.5 MiB 的五倍。EP 的税更重，但它只在 MoE 层、只在 DP 组内，而且能藏——第六节。

> 两个顺手的比方，边界写在这里。TP 像"每层交一次过路费"：但过路费是**按 token 线性收的**，而且不随收费站数量打折——TP 度翻倍，单次费用一分不少。EP 像"快递分拣"：分拣中心的总吞吐却取决于 routing 分布，热点专家会让某几个 rank 单独排队，这正是 EPLB 存在的原因。

## 2. 进程与 rank 布局

`docs/design/arch_overview.md:103`-`113` 那张表是权威：API Server `A`（默认 DP）、EngineCore `DP`、GPU Worker `DP × PP × TP`、DP Coordinator 在 `DP > 1` 时 1 个。两个例子背下来：4 卡 `-tp 4` 是 1+1+4 = 6 个进程；8 卡 `-tp 2 -dp 4` 是 4+4+8+1 = 17 个。API server 数量就是 `data_parallel_size`（`vllm/entrypoints/cli/serve.py:113`-`130`，external LB 时为 1）；elastic EP 会把 API server 强行压到 1（`:139`-`145`）。EngineCore 按 DP rank 起进程，`dp_rank` 就是全局 DP 序号（`vllm/v1/engine/utils.py:280`-`293`）；coordinator 只在 `needs_dp_coordinator` 为真且 `dp_rank == 0` 的主机上起一个（`vllm/v1/engine/utils.py:1231`-`1237`、`:1266`，条件见 `vllm/config/vllm.py:877`-`881`）。

rank 坐标可以直接从注释推出（`vllm/distributed/parallel_state.py:2155`，PCP=1 时）：

```python
# 布局：DP x PP x TP，从外到内
tp_rank = rank % TP
pp_rank = (rank // TP) % PP
dp_rank = rank // (TP * PP)
```

EP rank 则是"抽掉 PP 那一维之后的坐标"：EP 组 = `DP × PCP × TP`，组内成员共享同一个 `pp_rank`（`vllm/distributed/parallel_state.py:2334`-`2343`）。所以 `TP=2, PP=2, DP=2`（8 卡）时，`rank=5` 是 `tp=1, pp=0, dp=1`，它的 EP 队友是 `rank=0` 和 `rank=1`——**EP 不跨 PP**，两个 stage 各有一组专家。推 MoE 死锁时这条必须先算清。

关键的不对称：**TP/PP 在一个 EngineCore 内部，DP 在 EngineCore 外面**。所以 `world_size` 只算 TP×PP（`vllm/config/parallel.py:1056`-`1060`），DP 是"多开几个引擎"，进程间用 ZMQ 而不是 NCCL 说话。这也解释了为什么 DP 的负载均衡是个真问题（第八节），而 TP 的不是。

## 3. ParallelConfig 的字段与校验

`vllm/config/parallel.py:148`-`155` 把四个度都摆在明面上：`pipeline_parallel_size`、`tensor_parallel_size`、`prefill_context_parallel_size`、`data_parallel_size`。PCP 的注释很值得读：它**扩大进程 world size，但不增加 KV cache 的分片数**（`:152`-`154`）——这是理解 CP 的钥匙。

几条会咬人的校验，都能在启动时替你挡掉错误配置：

- `world_size` 在 `__post_init__` 里算成 `PP × TP × PCP`（`:1056`-`1060`），Executor 再用它 assert 一遍（`vllm/v1/executor/multiproc_executor.py:139`-`145`）。所以 `-tp 2 -pp 4` 要 8 张卡，不是 6 张。
- attention head 数必须被 TP 整除：`vllm/config/model.py:1573` 直接抛 `ValueError`。这是一条"模型结构约束"，不是工程选择。
- DCP 复用 TP 的 rank，所以要求 `tp % dcp == 0`（`vllm/config/parallel.py:634`-`640`）。
- PP 与 async scheduling 不兼容（V1 model runner 下，`vllm/config/vllm.py:1703`）。
- `enable_eplb` 必须先开 `enable_expert_parallel`，且 EP 组大小 `TP × PCP × DP > 1`（`vllm/config/parallel.py:600`-`607`）；elastic EP 还要求 EPLB 且禁止 PP（`:1066`-`1073`）。
- DBO 的两个阈值都不得小于微批数量（`:1272`-`1279`）。

## 4. Executor 层

`Executor.get_class`（`vllm/v1/executor/abstract.py:63`-`103`）就是一个字符串分派表：`"ray"` → `RayExecutorV2`（`vllm/v1/executor/ray_executor_v2.py`），`"mp"` → `MultiprocExecutor`（`vllm/v1/executor/multiproc_executor.py`），`"uni"` → `UniProcExecutor`（`vllm/v1/executor/uniproc_executor.py`），`"external_launcher"` 走 SPMD。注意实际文件名：不是 `uni_proc_executor.py`，也不是 `ray_distributed_executor.py`；后者在当前 main 上已经演化成 `RayExecutorV2`。

四个实现的心智模型，按"跳数"排：

| 实现 | 模型进程在哪 | `execute_model` 怎么走 | 什么时候被选中 |
|---|---|---|---|
| `UniProcExecutor` | 同进程 | 直接 `run_method`（`vllm/v1/executor/uniproc_executor.py:103`），零序列化 | `world_size == 1` |
| `MultiprocExecutor` | 子进程 | 一次入队 + 每 worker 各读一份 | 默认；单机或多机 `--nnodes` |
| `RayExecutorV2` | Ray actor | 继承 Multiproc 的 MQ + NCCL 数据面（`vllm/v1/executor/ray_executor_v2.py:239`-`247`） | 显式 `--distributed-executor-backend ray`、或已在 placement group 里 |
| `ExecutorWithExternalLauncher` | 外部启动器 | 同 UniProc，rank 从环境变量来 | SPMD / torchrun 类场景 |

`execute_model` 的形状在抽象基类里就定死了：`self.collective_rpc("execute_model", args=(scheduler_output,))`，然后 `return output[0]`（`vllm/v1/executor/abstract.py:291`-`297`）。MP 实现把这条路径压成了"一次广播 + 只收一个回包"：`SchedulerOutput` 只 `enqueue` 一次（`vllm/v1/executor/multiproc_executor.py:492`），节点内走 `/dev/shm` 里的单写多读环形缓冲（`rpc_broadcast_mq`，`:178`-`185`；跨机由 leader 再转发），所以 8 个 worker 读的是同一份序列化结果，不是 8 次 pickling。回包则只收 `output_rank` 那一个（`:496`），也就是"TP rank 0 且 PP 最后一个 stage"（`:649`-`663`，注释里那个 `world_size - tp_size` 的算法值得抄下来）。只有 KV connector 的输出需要全员聚合时，才会把所有回包拉回来再 merge（`:469`-`486`）。

```text
EngineCore                /dev/shm 环形缓冲              worker 0..N-1
    |-- enqueue(SchedOut) ------>|                          |
    |                       pickle 一次                     |
    |                            |--- 各读一份（零拷贝） -->| prepare_inputs
    |                            |                          | forward（层内 all-reduce）
    |<-- 只有 output_rank 回包 ------------------------------|
```

这张图解释了两件事：控制面每个 step 只有**一次**序列化，所以 TP=8 不会把 CPU 打满；数据面在 worker 内部，EngineCore 全程看不到 hidden 状态。

为什么默认是多进程而不是 Ray？代码给了答案：只要 `world_size_across_dp > 1` 就先假定 `"mp"`（`vllm/config/parallel.py:1152`），CUDA 上 `nnodes > 1` 也仍然是 `"mp"`（`:1156`-`1157`），只有当 `data_parallel_backend == "ray"`、或进程已经在一个 Ray placement group 里时才翻成 `"ray"`（`:1170`-`:1187`）。Ray 换来的是跨机放置和弹性伸缩，不是更快的 RPC——`RayExecutorV2` 复用的还是同一套 MQ 和 NCCL 数据面（`vllm/v1/executor/ray_executor_v2.py:240`-`243` 的 docstring 写得很直白）。反过来，`docs/serving/parallelism_scaling.md:33` 那句"多机默认走 Ray"和当前 main 的默认值已经对不上了，看代码为准。

## 5. 通信原语

`initialize_model_parallel`（`vllm/distributed/parallel_state.py:2085`）先用一句注释钉死 rank 布局：`ExternalDP x DP x PP x PCP x TP`（`:2155`），然后把这个 5 维张量按维度 transpose 成组：TP 组连续（`:2181`）、PP 组跨步（`:2305`）、DP 组再跨一步（`:2323`）、EP 组合并 `DP × PCP × TP`（`:2334`-`2363`）。**EP 组 = DP × PCP × TP** 这一点必须记住，否则算不对专家落在哪张卡。

上层是三个薄封装：`tensor_model_parallel_all_reduce` / `all_gather` / `reduce_scatter`（`vllm/distributed/communication_op.py:12`-`28`），一律转发给 `get_tp_group()`。`GroupCoordinator` 里有个容易被忽略的短路：`world_size == 1` 时直接 `return input_`（`vllm/distributed/parallel_state.py:744`-`746`）——TP=1 不只是"通信很快"，是根本没有 collective 节点。

真正选 kernel 的地方在 device communicator：`CudaCommunicator.all_reduce`（`vllm/distributed/device_communicators/cuda_communicator.py:384`-`468`）是一串带条件的降级阶梯——NCCL symmetric memory → quick all-reduce → FlashInfer → aiter（ROCm）→ **custom all-reduce** → PyTorch symmetric memory → pynccl → `torch.distributed`。前两档的门槛是尺寸而不是"开没开"：

| 档位 | 触发条件（TP=8） | 上限 |
|---|---|---|
| NCCL symmetric memory | world ≥ 4，尺寸 ≤16 KiB 或 ≥128 KiB | H100：64 MiB（`vllm/distributed/device_communicators/all_reduce_utils.py:58`-`70`） |
| custom all-reduce | world ≤ 8（2/4/6/8），同机全互联，≤8 MiB 且 16 B 对齐 | 8 MiB（`vllm/distributed/device_communicators/custom_all_reduce.py:118`、`:496`-`511`） |
| pynccl / torch.distributed | 兜底 | 无 |

所以 4 MiB 那次 all-reduce 在 TP=8 上走的是 NCCL symm mem 而**不是** custom all-reduce：`custom_ar_preferred_ranges` 给 TP=8 留的甜点区是 16 KiB 到 128 KiB（`vllm/distributed/device_communicators/all_reduce_utils.py:111`-`120`、`:123`-`156`）。这解释了一个常见困惑——"我明明没关 custom AR，profile 里却没有它的 kernel"，答：尺寸不合，走了带宽更好的那条路。

还有一个反直觉的默认值：EP 的 `all2all_backend` 默认是 `"allgather_reducescatter"`（`vllm/config/parallel.py:228`）。也就是说，不装 DeepEP 的时候，MoE 的 all-to-all 根本不是 all-to-all，而是"先用 all-gather 把所有 token 收齐、再 reduce-scatter 把结果散回去"——两个 `GroupCoordinator` 封装就够用了（`vllm/distributed/parallel_state.py:758`、`:789`）。代价是每个 rank 都要收下**全量** token，哪怕它一个专家都不负责这些 token。DeepEP 系后端才有真正的 dispatch/combine kernel，这也是第六节 DBO 强绑 DeepEP 的原因。

## 6. DBO：把 batch 劈成两半，让 all-to-all 躲进另一半的计算

`docs/design/dbo.md:5` 把动机写得没有歧义：**用 MoE 层稀疏的 all-to-all 去重叠周边的计算，只针对 DP+EP 部署**。做法是（`:9`）在 model runner 里把 batch 切成两个微批，起两个 CPU 线程各跑一半模型，在 `FusedMoEModularKernel` 内部的 yield 点上互相让路：一个在等 all-to-all 的时候，另一个在算。所有 `dbo_yield` / `dbo_maybe_run_recv_hook` 都在 `FusedMoEModularKernel.forward` 里（`:78`），所以这套机制对模型是透明的。`UBatchWrapper` 管线程生命周期和 CUDA Graph（`vllm/v1/worker/gpu_ubatch_wrapper.py:65`），`forward` 只认 `forward_context.ubatch_slices` 在不在（`:357`-`380`）。

开启条件是一串"与"：`--enable-dbo` + `--data-parallel-size > 1` + `--enable-expert-parallel`（`docs/design/dbo.md:32`） + DeepEP（`:37`，decode 主导选 `deepep_low_latency`，prefill 主导选 `deepep_high_throughput`）+ 至少 2 张可见 GPU（`:42`）。微批必须**所有 DP rank 一致**：任一行不可分就全员不分；token 数按全局最大值 padding；padding 后某个 rank 的第二个微批为空则整体放弃（`docs/design/dbo.md:52`）。阈值默认 decode 32 token、prefill 512 token（`vllm/config/parallel.py:280`、`:285`），低于阈值不分（`vllm/v1/worker/ubatch_utils.py:140`-`148`）。

为什么 all-to-all 能切、TP 的 all-reduce 不能切？关键在于"等"的东西不一样。all-reduce 的输出每个元素都需要**全部**输入，所以只能把 GEMM 按 tile 切开、让 tile 级通信和计算交错（这套结构在[通算并行（五）：张量并行]({{< relref "comm-overlap-05-tensor-parallel" >}})里拆过）；而两个微批的 all-to-all 是两份**互相独立**的数据——A 在等自己的 dispatch 回来时，B 完全可以先算自己的 attention。切分点是 batch，重叠对象是另一份数据。这就是 DBO 只要两个线程加一个 yield、而 TP 的 overlap 要重写 kernel 的原因。

风险三条，都能在代码里找到痕迹：

1. **只有 Full CUDA Graph 可用**（`docs/design/dbo.md:62`）。`UBatchWrapper` 每个形状只抓一张（微批版）图，所以没开微批时反而要把 FULL 降级成 NONE（`vllm/v1/worker/gpu_ubatch_wrapper.py:371`-`374`）。DBO + PIECEWISE 是行不通的组合。
2. **要切走一块 SM 预算**。`create_sm_control_context` 给通信留 `VLLM_DBO_COMM_SMS` 个 SM，默认 20（`vllm/v1/worker/ubatch_utils.py:87`，`vllm/envs.py:308`、`:2152`-`2160`），H100 的 132 个 SM 里剩 112 给计算，且只有 DeepGEMM 支持被这么限。
3. **阈值以下不生效**。decode 一步只有 20 个 token 时，切两半的收益抵不过多出来的线程同步和 padding，代码选择不分。

收益的量级取决于 all-to-all 在 step 里占多少——也就是第 1.3 节那 12.7 MiB/token 能不能被藏住。混合负载（prefill 与 decode 同批）下两半的形状差异大，重叠窗口会被撕碎，这是它最难调的地方。

## 7. EPLB、elastic EP、CP：只讲定位

- **EPLB**（`--enable-eplb`，`vllm/engine/arg_utils.py:1357`）：解决专家负载倾斜，按统计把热点专家复制成冗余副本再搬迁权重。EPLB 用**独立于 EP 的进程组**（`vllm/distributed/parallel_state.py:2365`-`2374`，注释解释了为什么必须隔离：避免和 MoE 前向的 collective 互相卡死）。入口 `vllm/distributed/eplb/rebalance_execute.py:669` 的 `rearrange_expert_weights_inplace`。冗余数默认由 `(逻辑专家数 + 冗余) % ep_size == 0` 推出来（`vllm/config/parallel.py:672`-`675`）；大规模部署文档建议直接给 32（`docs/serving/expert_parallel_deployment.md:251`）。
- **Elastic EP**（`--enable-elastic-ep`，`vllm/engine/arg_utils.py:1335`）：EP 组的成员可以变，靠 stateless 进程组重建（`vllm/distributed/parallel_state.py:2313`-`2321`），入口 `vllm/distributed/elastic_ep/elastic_execute.py:153` 的 `ElasticEPScalingExecutor`。它要求 EPLB 打开、禁 PP、禁 hybrid LB（`vllm/config/parallel.py:1066`-`1079`）。
- **CP**（`--prefill-context-parallel-size`，`vllm/engine/arg_utils.py:1241`；`--decode-context-parallel-size`，`:1217`）：切的是序列维，不是权重。定位见 `docs/serving/context_parallel_deployment.md:8`-`36`：PCP 摊 TTFT，DCP 摊 KV 显存。DCP 的收益可以用一句话算出来——**KV 复制倍数 = tp_size / kv_head 数**（`:36`），DeepSeek-R1 只有 1 个 kv head，`-tp 8` 就是 8 倍复制，`-dcp 8` 才把它消掉（`:44`）。

## 8. 部署建议

**单机 TP 与多机 TP 的分界，是 900 GB/s 与 50 GB/s 的分界。** NVLink 全互联的机器上，TP=8 的 all-reduce 还能压在 20 µs 量级；一旦跨机，同样一次就变成 45–90 µs，1.1 节那张表直接翻倍。经验规则：`TP ≤ 单机 GPU 数`，跨机用 PP 或 DP。没有 NVLink 的机器（文档举的例子是 L40S）连单机 TP 都不划算，应该用 PP（`docs/serving/parallelism_scaling.md:23`）。验证方法很土但有效：`NCCL_DEBUG=TRACE` 里出现 `[send] via NET/Socket` 就说明跨机 TP 走在 TCP 上，那不是能用的状态（`docs/serving/parallelism_scaling.md:218`-`219`）。

**PP 什么时候值得**：模型单机装不下、层数够多能切匀、且能接受 `pp_size` 个 batch 同时在飞带来的显存与延迟，以及 async scheduling 被关掉。它的好处是通信量小、切分不要求"整除 head"这种结构条件。

**DP 的负载均衡**：`DPCoordinator` 收集每个引擎的 waiting/running 队列长度，发布给所有 front-end 做路由（`vllm/v1/engine/coordinator.py:23`-`35`），同时维护"波次"（wave）状态让所有 rank 一起跑/一起停（`:37`-`46`）。注意 admission control 的粒度坑：`--max-num-seqs` 是**每 DP rank**，但 `--max-num-queued-reqs` / `--max-num-queued-tokens` 是**整个 server**（`docs/serving/data_parallel_deployment.md:27`-`29`），按 rank 数放大上限，否则 4 个 rank 只跑得出 1 个 rank 的量。

**EP 的显存收益**是四条路里最直接的：专家权重按 EP 度线性摊，1.3 节那 653 GB 在 EP=8 下变 82 GB/卡。前提是 attention 那边别拖后腿——大 MoE 的常规姿势是 `TP=1, DP=8, EP=8`，attention 权重在 8 个 rank 上复制，专家权重切开（`docs/serving/expert_parallel_deployment.md:41`）。

按这个顺序决策，基本不会走偏：

1. 单卡装得下：别分布式，`TP=1` 时连 collective 节点都不建（`vllm/distributed/parallel_state.py:744`-`746`）。
2. 单机装得下：`TP = 单机卡数`，先用 1.1 节的账估通信占比；层数切不匀或没有 NVLink 就退到 PP（`docs/serving/parallelism_scaling.md:23`）。
3. 单机装不下：`TP = 每机卡数`、`PP = 机器数`（`docs/serving/parallelism_scaling.md:9`），别把 TP 跨机。
4. MoE 且专家权重是大头：`TP=1, DP=N, EP=N`，显存交给 EP，再考虑 DBO 藏 all-to-all。
5. 显存够了但 KV 不够：`-dcp` 摊 KV 复制，或者干脆 P/D 分离（下一篇）。

## 9. 常见坑

1. **`-tp` 加了但没变快**。先看 all-reduce 占比：1.1 节的账在 batch 大、TP 高时必然恶化。TP 从 4 到 8 只快 1.5 倍是正常现象，不是配置错。
2. **PP 一开，吞吐掉了**。检查 async scheduling 是不是被静默关了（`vllm/config/vllm.py:1749`-`1755` 会打 warning），以及 batch queue 变成 `pp_size` 后 KV 是不是不够了。
3. **`SchedulerOutput` 太大**。广播缓冲单块默认 16 MiB（`VLLM_MQ_MAX_CHUNK_BYTES_MB`，`vllm/envs.py:260`），超过就走 overflow 分支改从 ZMQ socket 直发（`vllm/distributed/device_communicators/shm_broadcast.py:955`-`957`、`:1004`-`:1005`），零拷贝失效但不报错——大 batch + 长 prompt 时这是隐形的性能悬崖。
4. **以为 custom all-reduce 一直在跑**。TP=8 下它只覆盖 16–128 KiB；你的 4 MiB decode all-reduce 走的是 symm mem。想确认就去看 `should_custom_ar` 的三个前置条件（`vllm/distributed/device_communicators/custom_all_reduce.py:496`-`511`）。
5. **DBO 开了没效果**。多数是踩在阈值下（decode 32 / prefill 512）、或 CUDA Graph 不是 FULL、或某个 DP rank 的第二个微批为空导致全员放弃（`docs/design/dbo.md:52`）。另外 DeepSeek V4.1 Engram 明确不支持 DBO，必须关掉（`docs/features/engram.md:178`-`179`）。
6. **EPLB 一开就 OOM 或卡住**。冗余专家是要真占显存的；`num_redundant_experts` 没让 `(逻辑专家 + 冗余)` 被 `ep_size` 整除会被校验挡下（`vllm/config/parallel.py:672`-`675`）。多机还要 `ulimit -l unlimited`，否则 DeepEP 报 `cannot register cq buf`（`docs/serving/expert_parallel_deployment.md:263`）。

## 小结

- **TP 的税是固定的**：每 token 每层 16 KiB，与 batch、与 TP 度都无关；70B/TP=8/256 token 时一步 160 次 all-reduce、约 3.7 ms，占 decode 一步的四成。TP 从 4 到 8 只有 1.5 倍，到 16 跨机反而回退。
- **PP 省带宽但费并发**：一步只传一次 4 MiB hidden，可 `max_concurrent_batches` 被抬到 `pp_size`，而且 async scheduling 必须关（`vllm/config/vllm.py:1703`）。
- **进程账**：API `A`（默认 DP）+ EngineCore `DP` + Worker `DP×PP×TP` + Coordinator（DP>1 时 1 个）；`world_size` 只含 TP×PP×PCP，DP 在引擎外面。
- **Executor 的差别是跳数**：UniProc 零跳、Multiproc 一次 shm 广播 + 单回包（`vllm/v1/executor/multiproc_executor.py:492`、`:649`）、Ray 复用同一套数据面只多了放置能力。默认 mp，因为跨机也不需要 Ray。
- **all-reduce 有三档**：NCCL symm mem（≤16 KiB 或 ≥128 KiB，H100 TP=8 上限 64 MiB）、custom AR（16–128 KiB 甜点、≤8 MiB、同机全互联）、pynccl 兜底。
- **EP 摊显存、DBO 藏流量**：DeepSeek-V3 类专家权重 653 GB，EP=8 → 82 GB/卡；每 token 12.7 MiB 的 all-to-all 靠 DBO 用两个微批 ping-pong 藏进计算，代价是 20 个 SM、FULL CUDA Graph、以及一堆"必须所有 rank 一致"的前置条件。
- **KV 复制倍数 = tp_size / kv_heads**，这是 DCP 存在的全部理由；DeepSeek-R1 的 `-tp 8` 是 8 倍复制。

下一篇讲 KV 怎么离开这张卡：[KV 传输：connector、P/D 分离与 offload]({{< relref "vllm-11-kv-transfer" >}})。

---

参考：vLLM 官方文档 [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)、[Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)；仓库内 `docs/design/arch_overview.md`、`docs/design/dbo.md`。
