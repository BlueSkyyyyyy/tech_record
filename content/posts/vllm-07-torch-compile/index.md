---
title: "vLLM 研读（七）torch.compile：切分、融合与编译预算"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 8
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "torch.compile", "CUDA Graph", "Inductor", "算子融合"]
categories: ["推理框架"]
---

decode 一步只算一个 token。一个 8B 模型在 H100 上 `batch=1` 出一个 token，GPU 真正做乘加的时间约 4.8 ms——这是把 16 GB bf16 权重从 HBM 完整搬一遍的物理下限；而把这一层层的 kernel 依次塞进 CUDA stream，本身还要再花 1 ms 量级。**瓶颈不在 kernel 算得慢，在于 launch 太碎。**

**先记住一个画面**：模型的 `forward` 被 TorchDynamo 抓成一整张 FX 图，然后在每个 attention 前面剪一刀，得到若干"片"（piece）；每一片交给 TorchInductor 编译，再分别收进一个 CUDA Graph。运行时 decode 一步等于「33 次 graph replay + 32 次 attention 的 eager launch」，而不是 256 次 kernel launch。

这篇回答三个问题：

1. `-O0` 到 `-O3` 各打开什么、默认是哪一级、代价和数值风险在哪？
2. 为什么偏偏在 attention 处切，"片"和 CUDA Graph 是怎么接上的？
3. 编译预算（启动时间、显存、磁盘缓存）花在哪，怎么用日志确认自己真的命中了编译，而不是悄悄退回 eager？

> 代码基线：vLLM main，HEAD `c7a7607519`。行号以此为准；上游改动后请重新 `grep -n`。

## 1. 为什么 decode 最吃编译

prefill 一步处理几百到几千个 token，每个 kernel 都有足够多的并行工作把它"喂饱"，launch 的那几微秒被摊薄；decode 一步每个请求只出 1 个 token，GEMM 退化成 batch=1 的向量-矩阵乘，kernel 时间被压到极限，固定开销就开始显形。这笔固定开销有两层：CUDA kernel launch（约 3–5 μs）和 Python 侧的 module/op dispatch。

**账本（量级估算）**。假设：Llama-3.1-8B，bf16 权重 16 GB，H100 SXM 带宽 3.35 TB/s，TP=1，`batch=1` decode，32 层。

| 项目 | 计算 | 结果 |
|---|---|---|
| 权重搬运下限 | 16 GB ÷ 3.35 TB/s | 4.8 ms |
| eager 每层 kernel 数 | 4 个 GEMM + RoPE + attention + SiLU+Mul，两次 RMSNorm 只算 1 条 | 8 |
| eager 一步 launch 次数 | 32 层 × 8 | 256 次 |
| launch 开销 | 256 × 3–5 μs | 0.77–1.28 ms（取 1.0 ms） |
| eager 一步总时长 | 4.8 + 1.0 | ≈ 5.8 ms，launch 占 **17%** |
| piecewise 之后 | 33 次 replay + 32 次 attention launch = 65 次 | ≈ 0.26 ms，占比降到 **5%** |
| 全图 CUDA Graph | 1 次 replay | ≈ 0.004 ms，可忽略 |

这张表也解释了为什么编译在服务端的收益高度依赖负载形状：`batch=1` 的长输出（聊天、agent 循环、投机解码的验证步）里 launch 占比约 17%，而 `batch=256`、每步算几十毫秒时，同样的 1 ms 只剩几个百分点。

类比的边界：编译**不是**让 kernel 算得更快。它做两件事——把逐层 launch 折叠成一次 graph replay，以及把相邻算子融成一个 kernel 少读写一次显存。算术强度该多少还是多少，所以它救不了大 batch 的算力瓶颈。

## 2. 编译级别：`-O0` 到 `-O3`

`-O3` 只是 `--optimization-level 3` 的缩写，由参数预处理展开（`vllm/utils/argparse_utils.py:395-399`）；`optimization_level` 字段默认 `O2`（`vllm/config/vllm.py:452`）。

| 级别 | `mode` | `cudagraph_mode` | 融合 pass | 风险 |
|---|---|---|---|---|
| `-O0` | NONE | NONE | 全关 | 完全不编译，decode 付满 launch 开销；只适合调试 |
| `-O1` | VLLM_COMPILE | PIECEWISE | `fuse_norm_quant`、`fuse_act_quant`（条件性）＋ ROCm 的 `fuse_act_padding`、`fuse_mla_dual_rms_norm` | attention 仍 eager，全图不进 graph |
| `-O2`（默认） | VLLM_COMPILE | FULL_AND_PIECEWISE | 追加 `fuse_allreduce_rms`、`fuse_rope_kvcache` 与 `fuse_qk_norm_rope_kvcache`（后两者 ROCm）、`fuse_rope_kvcache_cat_mla` | FULL 图额外占显存；融合改变求和顺序 |
| `-O3` | 同 O2 | 同 O2 | 同 O2 | 与 O2 当前完全等价，别指望更快 |

四个级别的差异集中在 `vllm/config/vllm.py:270-354` 的 `OPTIMIZATION_LEVEL_00..03` 四张表里，`-O2` 与 `-O3` 两张逐字段相同——这一点代码和文档口径一致：`O3` 的 docstring 直接写 "Currently the same as -O2"（`vllm/config/vllm.py:159`），`docs/design/optimization_levels.md:78-81` 也是同样一句。所以**"`-O3` 会更激进、可能改数值"这一说法在当前 main 上不成立**。

这些开关全挂在 `PassConfig` 上，字段声明集中在 `vllm/config/compilation.py:127-154`；默认值一律是 `None`，含义是"留给级别默认值决定"，所以你可以只覆盖其中一项而不动其余。

真正会改数值的不是级别，而是融合本身：

- `fuse_allreduce_rms` 在 `VLLM_BATCH_INVARIANT` 下被强制关掉，注释写明"fused all-reduce + RMSNorm path is not batch-invariant"（`vllm/config/vllm.py:194-201`）。也就是说，同一个请求在不同 batch 组合下能否复现 bit 级一致，取决于这个开关。
- RMSNorm/SiLU 与 FP8 量化融合后，量化发生在融合 kernel 内部，中间张量不再以 bf16 落盘，数值与"先反量化再算"的路径不完全等价。

还有一条容易被忽略的规则：**用户显式设的字段永远赢过级别默认值**。`_apply_optimization_level_defaults` 只往 `None` 字段里写（`vllm/config/vllm.py:1021`、`1028`），所以 `-O2 -cc.pass_config.fuse_allreduce_rms=False` 这种组合是成立的，也是官方推荐的关单点方式。

## 3. piecewise compilation：切在 attention 上

`CompilationConfig.splitting_ops` 的默认值不是硬编码的列表，而是 `None`；`__post_init__` 之后由 `set_splitting_ops_for_v1` 填成「attention 类算子」（`vllm/config/compilation.py:1187-1196`）。这些算子的白名单在 `_attention_ops` 里，共 18 个，含 `vllm::unified_attention_with_output`、`vllm::unified_mla_attention_with_output`、`vllm::mamba_mixer2`、`vllm::sparse_attn_indexer` 等（`vllm/config/compilation.py:779-800`），最后还会补上 `vllm::unified_kv_cache_update` 与 MLA 版本（`vllm/config/compilation.py:1224-1225`）。

切分动作在 `split_graph` 里逐节点判断（`vllm/compilation/backends.py:576`、`601`），判定条件就是"这个 node 的 target 是否在 `splitting_ops` 里"（`vllm/compilation/partition_rules.py:14-34`）。为什么切在这里，而不是均匀切：

- attention 要读 KV cache、block table、attention metadata，这些东西的 shape 随 batch 和序列长度变化，塞进 CUDA Graph 的静态内存假设很难成立；
- attention 被整体包成了一个 custom op `torch.ops.vllm.unified_attention_with_output`（`vllm/model_executor/layers/attention/attention.py:814`），Dynamo 不会往里看——这正是"复杂算子还能保住全图捕获"的关键（`docs/design/torch_compile.md:179`）。

切完的结果是：一个 32 层模型得到 33 个"可捕获子图"（第一个 attention 之前、相邻两个 attention 之间、最后一个 attention 之后）加 32 个 attention 子图。每个可捕获子图都会被 `CUDAGraphWrapper` 包一层，`runtime_mode` 固定为 PIECEWISE（`vllm/compilation/backends.py:698-701`）。但**只有 3 份不同的 Inductor 代码**：首段、中间段（所有中间层共用）、末段（`docs/design/torch_compile.md:202-206`）——这是编译时间没有随层数线性爆炸的原因。

三个容易混的配置项：

| 字段 | 位置 | 作用 |
|---|---|---|
| `splitting_ops` | `vllm/config/compilation.py:502-519` | 切在哪；`[]` 表示不切（全图一个 graph，配合 FULL cudagraph） |
| `inductor_passes` | `vllm/config/compilation.py:602-607` | 追加自定义 Inductor pass，值可以是函数或限定名，在 `__post_init__` 里解析（同文件 `959-974`） |
| `custom_ops` | `vllm/config/compilation.py:480-492` | 哪些 vLLM 自研 kernel 参与，`all`/`none`/`+op`/`-op` |

`splitting_ops` 被显式设成 `[]` 时有个连带后果：`cudagraph_mode` 会从 PIECEWISE 降级成 NONE，从 FULL_AND_PIECEWISE 降级成 FULL（`vllm/config/compilation.py:1227-1252`），并且打一条 warning。很多"我明明开了 CUDA Graph 却没生效"的现场，源头就是这里。

## 4. 动态 shape 与编译预算

`torch.compile` 默认会对 shape 加 guard，vLLM 的策略是"丢掉 guard"：一张以 token 数 `num_tokens` 为唯一符号维的图，覆盖所有 batch。控制这一行为的字段是 `dynamic_shapes_config.type`，默认 `BACKED`（`vllm/config/compilation.py:346`），另外两档是 `UNBACKED`（保证不加 guard，但可能抛 data-dependent error）和 `BACKED_SIZE_OBLIVIOUS`（折中，PyTorch 侧仍是实验特性）。三档的取舍在 `docs/design/torch_compile.md:129-139` 讲得比代码注释清楚。

编译范围的上界直接由调度参数决定：`_set_compile_ranges` 把 `max_num_batched_tokens` 作为第一个端点（`vllm/config/vllm.py:2721`），再由融合阈值插入更多端点——比如 AllReduce 融合的通信量上限换算成 token 数（`vllm/config/vllm.py:2741-2746`）。换算公式是

$$
N_{\text{tokens}} = \frac{S_{\max}}{H \cdot \text{itemsize}}
$$

以 8B 模型（hidden 4096、bf16）为例：TP=2 在 SM90 上阈值 64 MB，算得 8192 token，刚好等于默认 `max_num_batched_tokens` 8192，于是 AllReduce 融合对全部 token 数生效；TP=8 的阈值只有 0.5 MB，算得 64 token，于是编译被切成 `[1, 64]` 与 `[65, 8192]` 两段，前一段才带融合。阈值表本身在 `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:104-126`。

**编译预算账本（量级估算）**：同样 32 层、`-O2`、TP=8 SM90。`PiecewiseBackend` 会被实例化 33 次，每个实例对每段 range 各编译一次，即 33 × 2 = 66 次 `compile()` 调用；去重后 Inductor 只需产出 3 份代码 × 2 段 range = 6 份产物，其余命中 Inductor 自己的 cache。单份产物的首次编译是秒级到十几秒（含 Triton 编译），合计 30–120 s 启动时间。这也是为什么 autotune 默认关闭——按具体 shape 调优"从几秒到几分钟"（`docs/design/torch_compile.md:246`）。

**padding 的代价**同样来自这里。CUDA Graph 的形状必须固定，capture size 是一张预设表：`[1, 2, 4]` 加 8 的倍数到 256，再加 16 的倍数到 `max_graph_size`（`vllm/config/vllm.py:2467-2468`）。运行时"batch 落在哪个 capture size 就用哪张图，就近向上取"（`vllm/config/vllm.py:2498-2500`），于是 batch=37 会 replay 40 的图，多出来的 3 个 token 槽位照样跑完整个 forward。

缓存在磁盘上分两层，都在 `VLLM_CACHE_ROOT`（默认 `~/.cache/vllm`，`vllm/envs.py:35`）下面：

- vLLM 自己的产物：`torch_compile_cache/<10 位 hash>/rank_i_j/`，hash 由环境、config、被 trace 的源码、编译器版本四因子算出（`vllm/compilation/backends.py:1086-1098`）；
- 进程级 `TORCHINDUCTOR_CACHE_DIR`：被 vLLM 改写到 `<cache_dir>/inductor_cache`，不再用 `/tmp`（`vllm/compilation/compiler_interface.py:32-34`）。

改一行模型代码、换个 `block_size`、升个 torch 版本，都可能让那个 10 位 hash 变掉，于是"冷启动又慢了一次"。要复现冷启动，必须同时清掉这两个目录。

## 5. custom op：把不该被看的地方藏起来

`CustomOp` 是一个分发壳：`forward` 在构造时绑定到 `_forward_method`（`vllm/model_executor/custom_op.py:133-135`），启用时按平台走 `forward_cuda`/`forward_hip`，禁用时退回 `forward_native` 让编译器去融。是否启用由 `custom_ops` 决定，`default_on()` 的语义是"含 `all` 则默认开，含 `none` 则默认关"（`vllm/model_executor/custom_op.py:293-300`）。

反直觉的一点：**用 Inductor 时 vLLM 默认把自己的 kernel 全关掉**。`VllmConfig.__post_init__` 在 `backend=="inductor"` 且 `mode != NONE` 时往 `custom_ops` 里补一个 `"none"`，否则补 `"all"`（`vllm/config/vllm.py:1891-1896`）。也就是说 `rms_norm`、`silu_and_mul`、`rotary_embedding` 这些 `@CustomOp.register(...)` 注册的算子（`vllm/model_executor/layers/layernorm.py:38`、`vllm/model_executor/layers/activation.py:115`、`vllm/model_executor/layers/rotary_embedding/base.py:14`），在编译模式下默认交给 Inductor 生成 Triton 融合代码。原因写在融合文档里：NVIDIA 上 Inductor 生成的融合 kernel 比 vLLM 自研 CUDA kernel 更快（`docs/design/fusions.md:211-213`）。

所以融合 pass 的开关是**条件性**的：`fuse_norm_quant` 只在 `rms_norm` 或 `quant_fp8` 走 custom kernel 时才开（`vllm/config/vllm.py:172-179`），`fuse_act_quant` 同理，FP4 模型例外（`vllm/config/vllm.py:182-190`）。你如果手动 `-cc.custom_ops='["all"]'`，反而会把这些融合 pass 一起打开。

真正的例外是 attention：它不是 `CustomOp`，而是用 `direct_register_custom_op` 注册的原生 torch op（`vllm/model_executor/layers/attention/attention.py:813-818`），并且把 `output` 作为入参传进去（in-place 写），这样 CUDA Graph 捕获时显存分配也被一起收进图里（`docs/design/torch_compile.md:252`）。

## 6. 融合清单与生效条件

官方表在 `docs/design/fusions.md:21-31`，数字照抄如下（"E2E Speedup" 列，条件是作者标注的硬件/并行度）：

| 融合 | 开关 | 融了什么 | 默认开启 | E2E 收益 |
|---|---|---|---|---|
| AllReduce + RMSNorm | `fuse_allreduce_rms` | all-reduce → residual add → RMSNorm（→ quant） | O2，TP>1 且 SM90/SM100 且装了 FlashInfer | 5–20% |
| Attention + Quant | `fuse_attn_quant` | attention 输出 → FP8/NVFP4 量化 | 任何级别都不默认开 | 3–7% |
| RoPE + KV cache | `fuse_rope_kvcache` | rotary embedding → KV cache 写入 | O2，仅 ROCm/AITER | 2–4% |
| QK Norm + RoPE | `enable_qk_norm_rope_fusion` | Q/K RMSNorm → RoPE | 不默认（H100 上有性能回退） | 2–3% |
| RMSNorm + Quant | `fuse_norm_quant` | RMSNorm（+residual）→ FP8/FP4 quant | O1，条件性 | 1–4% |
| SiLU+Mul + Quant | `fuse_act_quant` | SiLU+Mul → FP8/FP4 quant | O1，条件性 | 1–4% |
| MLA dual RMSNorm | `fuse_mla_dual_rms_norm` | 成对 q/kv RMSNorm → 1 个 HIP kernel | O1，仅 ROCm/AITER | 1–2% |

两点读表提示：

- **"低 token 数才划算"**是这张表的隐含维度。AllReduce 融合和 RoPE+KV cache 融合都只在小 batch 段生效，代码里就是 `compile_range.end <= max_token_num`（`vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1128-1132`、`vllm/compilation/passes/fusion/rope_kvcache_fusion.py:457-461`），RoPE 那条的阈值默认 256 token（`vllm/config/compilation.py:156`）。
- **pass 的顺序是被设计过的**，不是随便排的：先 `NoOpElimination`，再 `AllReduceFusionPass`，然后 `RMSNormQuantFusionPass`、`ActivationQuantFusionPass`，最后还有 IR lowering、clone elimination、两次 post cleanup 和 `fix_functionalization`（`vllm/compilation/passes/pass_manager.py:154-223`）。AllReduce+RMS 必须排在 RMS+Quant 之前，否则 `fused_add_rms_norm` 已经被量化 pass 吃掉，前者就匹配不上了。

## 7. 常见坑

**graph break**。Dynamo 在 fullgraph 模式下遇到不支持的 Python 就直接报错；更隐蔽的是"能跑但图断了"，性能悄悄退回 eager。最常见的触发是依赖张量数值的控制流：

```python
if data.size(0) % 128 == 0:   # 依赖动态维的分支
    foo()
else:
    bar()
```

正确做法是把分支逻辑包进 custom op（Dynamo 不 trace 进 custom op），而不是指望编译器猜（`docs/design/debug_vllm_compile.md:148-175`）。

**recompile**。shape 变化本身不触发重编译——一张动态图覆盖所有 batch；触发重编译的是**超出编译范围**或**打破了 guard 假设**。前者会撞到 `PiecewiseBackend.__call__` 里的断言 "Shape: N out of considered ranges"（`vllm/compilation/piecewise_backend.py:361-364`），后者表现为 `ConstraintViolationError`，此时应换 `unbacked` 定位（`docs/design/debug_vllm_compile.md:202-213`）。

**编译时间与显存**。时长看 `-O0`/`-O1`，或只关 autotune；显存看 FULL cudagraph——每张 capture size 的图都要独占一份中间激活。显存紧的时候 `-cc.cudagraph_mode=PIECEWISE` 通常比降 `gpu_memory_utilization` 更划算。

**`--enforce-eager` 什么时候用**。它同时关掉编译和 CUDA Graph（`vllm/config/vllm.py:1817-1819`），等价于 `-cc.mode=none -cc.cudagraph_mode=NONE`。适用场景只有三个：定位"是不是编译/图捕获引入的错误"、显存极度紧张、以及模型代码短期改不成可捕获形式。生产环境拿它当默认值是纯亏。

**怎么确认命中了**。开 `VLLM_LOGGING_LEVEL=DEBUG`，认这几行：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B
```

| 日志 | 出处 | 说明 |
|---|---|---|
| `Using cache directory: .../torch_compile_cache/<hash>/rank_0_0` | `vllm/compilation/backends.py:1125-1129` | 命中哪份缓存 |
| `Dynamo bytecode transform time: X s` | `vllm/compilation/backends.py:1188` | 图捕获耗时 |
| `Compiling a graph for compile range [...] takes X s` | `vllm/compilation/backends.py:417-419` | 真在编译（冷启动） |
| `Directly load the compiled graph(s) for compile range ... took X s` | `vllm/compilation/backends.py:311-313` | 命中缓存（热启动） |
| `Enabled custom fusions: norm_quant, act_quant, ...` | `vllm/config/compilation.py:318` | 哪些融合真的开了 |
| `fusion pass matches: {...}` | `vllm/compilation/passes/vllm_inductor_pass.py:121` | 每个 pass 实际替换了几个 pattern |

最后一行是判断"融合到底有没有生效"的唯一硬证据：开关打开不等于 pattern 匹配上，`matches` 为空就是白开。

## 8. 读代码/调参时的三个检查点

1. **先看 `custom_ops` 与融合开关的联动**。看到 `-cc.pass_config.fuse_norm_quant=True` 却没效果，去查 `enabled_custom_ops`/`disabled_custom_ops`（`vllm/config/compilation.py:1331-1345`）——`rms_norm` 走的是 Inductor 生成的原生实现时，这个 pass 本来就会被 `enable_norm_fusion` 判成 False。
2. **再看 `splitting_ops` 到底是不是 attention**。被显式设成 `[]`、或开了 `fuse_attn_quant`（走 `set_splitting_ops_for_attn_fusion`），`cudagraph_mode` 都会被动降级（`vllm/config/compilation.py:1272-1290`），此时谈"piecewise 命中率"没有意义。
3. **最后对一下编译范围端点**。端点由 `max_num_batched_tokens` 和融合阈值共同决定，谁小谁就是分界线；调大 `max_num_batched_tokens` 会把更多 token 推出融合区，同时让动态图覆盖更宽的范围——收益和启动时间要一起看。

## 小结

- decode 的瓶颈是 launch 密度：8B 模型 `batch=1` 一步约 5.8 ms，其中 1 ms 量级是 256 次 kernel launch（约 17%），piecewise 后降到 0.26 ms（约 5%）。
- `-O2` 是默认级别；`-O3` 在当前 main 上与 `-O2` 逐字段相同，代码和文档都写明"暂未更激进"，不要把它当成"更快的开关"。
- 切分点是 attention：`splitting_ops` 默认填成 18 个 attention 类算子，32 层模型切出 33 个可捕获片，但只有 3 份不同的 Inductor 代码。
- custom op 在 Inductor 下默认全关（`custom_ops` 补 `"none"`），融合 pass 因此是条件性的；attention 是特意被藏起来的那个 op，为的是保住全图捕获。
- 编译预算 ≈ 3 份代码 × range 段数；range 上界是 `max_num_batched_tokens`，融合阈值再插一刀（TP=8 SM90 时是 64 token），首次启动 30–120 s 量级。
- 判断融合是否生效只看 `fusion pass matches`；判断冷/热启动只看 `Compiling a graph` 与 `Directly load the compiled graph(s)` 这两行。

下一篇进入采样：[采样：logits 处理、logprobs 与结构化输出]({{< relref "vllm-08-sampling" >}})——看那张 `[num_tokens, vocab_size]` 的大矩阵在出模型之后还要被怎么揉一遍。

参考：官方 [torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile/) 与 [Fusion torch.compile passes](https://docs.vllm.ai/en/latest/design/fusions/)。
