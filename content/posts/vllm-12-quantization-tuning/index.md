---
title: "vLLM 研读（十二）量化、显存与调优：把模型塞进去并跑满"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 13
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "量化", "KV cache", "显存", "调优"]
categories: ["推理框架"]
---

调 vLLM 的人迟早会撞上同一堵墙：模型能起来，但并发上不去；并发上去了，p99 的 TTFT 又炸了；换了个量化 checkpoint，显存确实省了，吞吐反而掉了。这三个现象背后是同一本账——显存怎么分、参数怎么换。

先记住一个画面：**一张固定大小的桌子，上面摆五摞东西**——权重、KV cache、激活/workspace、CUDA graph、通信缓冲。vLLM 启动时做的事，本质是先把前四摞量出来，剩下的全推给 KV cache；跑起来之后，它只能在 KV 这一摞里腾挪。

> 类比的边界：桌子的面积（HBM 容量）和每摞东西的搬运速度（带宽）都是焊死的，人能做的是选"量"还是"猜"——`gpu_memory_utilization` 就是那把尺子，memory profiling 是它量出来的读数。

读完你应该能回答三个问题：

- 一张 80 GB 卡上，权重之外还剩多少能给 KV？这个数是哪几行代码算出来的？
- 量化省的是哪一摞？为什么省了显存反而可能更慢？
- 上线后盯哪几个指标，能区分"显存不够"和"调度不优"？

## 1. 五项预算与那把尺子

每卡显存写成五项之和：

$$
M_{\text{card}} = \underbrace{P \cdot b_w}_{\text{weights}} + \underbrace{B_{\text{kv}} \cdot N_{\text{tok}}}_{\text{KV pool}} + \underbrace{M_{\text{act}}}_{\text{activation}} + \underbrace{M_{\text{graph}}}_{\text{CUDA graph}} + \underbrace{M_{\text{comm}}}_{\text{NCCL/非 torch}}
$$

| 项 | 70B/8 卡（单卡） | 估算方式 | 谁来量 |
|---|---|---|---|
| weights | 17.7 GB | `P × 字节/参数`：BF16 = 2 B、FP8 = 1 B、W4 ≈ 0.5 B + scale | `model_runner.model_memory_usage` |
| KV pool | ~50 GB | `bytes_per_token × tokens`，见第 2 节 | profiling 反推 |
| activation/workspace | 1–3 GB | 与 `max_num_batched_tokens`、MoE workspace 同阶 | `transient_peak_headroom` |
| CUDA graph | 0.5–3 GB | 捕获尺寸个数 × 每张图的中间激活 | `profile_cudagraph_memory()` |
| NCCL/非 torch | 0.3–1 GB | 随 TP/EP 与 message size 增长，vLLM 不建模 | 归入 non-torch |

`gpu_memory_utilization` 的语义在 `vllm/config/cache.py:216` 定义，默认值不是常数：**可扩展 KV cache 生效时是 1.0，否则 0.92**（`vllm/config/cache.py:489` 的 `resolved_gpu_memory_utilization`）。文档字符串还写清了一件常被忽略的事——这是**每个实例**的额度，"同一张卡跑两个实例时各设 0.5"是合法用法（`vllm/config/cache.py:219`）。

额度换算成 KV 池只有四步（`vllm/v1/worker/gpu_worker.py:716` 的 `determine_available_memory`）：

```text
requested  = total_memory × util                        # vllm/v1/worker/utils.py:750
non_kv     = total_consumed + transient_peak_headroom    # vllm/utils/mem_utils.py:318-321
available  = requested − non_kv − cudagraph_estimate     # vllm/v1/worker/gpu_worker.py:824-828
num_blocks = available // bytes_per_block                # vllm/v1/core/kv_cache_planning.py:700
```

`non_kv` 不是拍脑袋：`memory_profiling` 的 docstring（`vllm/utils/mem_utils.py:242`）把一张卡的内存分成三类——别人的、torch 的、vLLM 用但不是 torch 的（NCCL 就在第三类）。量法是跑一次 `profile_run`，用 `max_num_batched_tokens` 规模的 dummy batch 把峰值激活打出来（`vllm/v1/worker/gpu_model_runner.py:6121`，注释里明说这一步顺带预分配通信缓冲），再用 `mem_get_info` 的差值算总消耗。

两个容易读错的细节：

- CUDA graph 的显存是**单独估**的（`vllm/v1/worker/gpu_worker.py:783`），`profile_cudagraph_memory()` 对每种模式只捕前 2 个尺寸做采样外推（`vllm/v1/worker/gpu_model_runner.py:6320`）。这张估算默认开启（`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认 1，`vllm/envs.py:342`）；关掉它，图占的显存会从 KV 池里被"偷"走，然后在 capture 时 OOM。
- profiling 会断言"初始空闲 ≥ profiling 后空闲"（`vllm/v1/worker/gpu_worker.py:803`）——同卡上有别的进程在启动时释放显存，这一步会直接报错，而不是给你一个偏大的 KV 池。

## 2. 账本一：两张卡上的五项分解

假设 H100 80 GB、`gpu_memory_utilization=0.92`（73.6 GB 额度）、`block_size=16`；Llama-3.1-70B 用 TP=8，Llama-3.1-8B 单卡。每 token KV 字节数按 `2 × L × n_kv × d_head × 2 B`，块大小即页大小（`vllm/v1/kv_cache_interface.py:579`）：

| 项 | 70B / TP=8 | 8B / 单卡 |
|---|---|---|
| 权重（BF16） | 141.2 GB ÷ 8 = **17.7 GB** | **16.1 GB** |
| KV 每 token | 2×80×8×128×2 = **320 KiB** | 2×32×8×128×2 = **128 KiB** |
| 每块（16 token） | 5.00 MiB | 2.00 MiB |
| activation/workspace | 2.5 GB | 1.5 GB |
| CUDA graph | 2.5 GB | 1.5 GB |
| NCCL/非 torch | 0.8 GB | 0.6 GB |
| **剩给 KV** | **≈50.1 GB** | **≈53.9 GB** |
| 块数 / 可存 token | 9,565 / 约 153 k | 25,701 / 约 411 k |

这张表能读出两件事。第一，**权重的 TP 切分是线性的，别的不是**：70B 的权重从 141 GB 掉到 17.7 GB，KV 池却只有 50 GB，因为另外四项加起来仍有 5.8 GB。第二，8B 单卡的 KV 池比 70B/8 卡更大——小模型不是"参数少了所以能跑更多并发"，是"参数少了所以 KV 池大了"。记住 **70B 上每块 5 MiB** 这个数：所有"再加 1 GB 能多几个请求"的心算都从这里出发。

## 3. 账本二：并发上限怎么估

单请求按 `max_model_len` 独占 KV 时，并发上限是：

$$
N_{\max} = \min\left(\text{max\_num\_seqs},\ \left\lfloor \frac{\text{KV bytes}}{\text{max\_model\_len} \times \text{bytes/token}} \right\rfloor\right)
$$

| `max_model_len` | 70B/8 卡 | 8B 单卡 |
|---|---|---|
| 8 k | 18 | 50 |
| 32 k | 4 | 12 |
| 128 k | 1 | 3 |

这张表解释了两个最常见的"事故"：

- **8B 模型设 `--max-model-len 131072`，并发只有 3**。H100 上默认 `max_num_seqs` 是 1024（`vllm/engine/arg_utils.py:3004-3007`），根本轮不到它约束——约束是 KV。
- **70B 在 32 k 上下文下并发 4**，四个请求就能把池子吃满，第五个开始排队或抢占。这不是引擎不行，是账算不过来。

vLLM 对"连一个请求都放不下"是硬报错的：`_check_enough_kv_cache_memory`（`vllm/v1/core/kv_cache_planning.py:72`）会告诉你需要多少 GiB、可用多少 GiB，还会估一个"能塞下的最大长度"。所以把 `max_model_len` 调大不是"反正用不到"，它直接乘在分母上。

不想让 profiling 猜，可以走两条手动路线：`--kv-cache-memory-bytes` 直接给 KV 字节数（此时忽略 `gpu_memory_utilization`，`vllm/config/cache.py:379`），或者 `--enable-extensible-kv-cache` 用虚拟地址预留、warmup 后按实际空闲提交物理页（`vllm/config/cache.py:388`）。后者也是"默认 1.0"的来源：额度只是上限，超了占的是地址空间而不是页。

## 4. 量化方法全景

`vllm/model_executor/layers/quantization/__init__.py:15` 的 `QuantizationMethods` 是这份清单的唯一权威（当前 main 上正好 30 个值）。按"粒度 / 是否要校准 / 动没动 KV"整理：

| 方法 | 权重粒度 | 激活 | 校准 | 对 KV 的影响 |
|---|---|---|---|---|
| `fp8` | per-tensor，或 128×128 block | dynamic per-token（默认） | 否（静态方案需 checkpoint 带 scale） | 无 |
| `fp8_per_tensor` / `fp8_per_block` / `fp8_per_channel` | 在线量化，三种粒度 | dynamic | 否 | 无 |
| `modelopt_fp4`（NVFP4） | per-16（E4M3 scale + 二级 fp32） | W4A4 或 W4A16 | 是（ModelOpt 离线） | 无 |
| `mxfp4` / `gpt_oss_mxfp4` | per-32（E8M0 scale） | bf16 | 是 | 无 |
| `awq_marlin` / `gptq_marlin` | 4 bit，group 128（GPTQ 可 4/8 bit、`desc_act`） | bf16 | 是 | 无 |
| `compressed-tensors` | W4A16/W8A8/W4A8/NVFP4/MXFP8 多种 | 视 scheme | 是（llm-compressor） | 无 |
| `gguf` / `bitsandbytes` | — | — | — | 已迁出仓库，见下 |

粒度不是名词游戏，它决定**scale 的额外字节**和**精度损失落在哪**。NVFP4 的 group 是 16（`vllm/model_executor/layers/quantization/modelopt.py:743`），每 16 个权重配 1 字节 E4M3 scale，等效 4.5 bit/参数；FP8 的 128×128 block 每块只有一个 fp32 scale，额外开销可以忽略。`modelopt.py:145` 的 `LINEAR_ALGOS` 把 ModelOpt 六种算法和粒度写在一起，是这段最好的注脚。

按代码纠正两点。第一，**`gguf` 和 `bitsandbytes` 不在这个 HEAD 的 `QuantizationMethods` 里**。它们已迁到仓库外插件（`vllm-gguf-plugin`、`vllm-bnb-plugin`，见 `docs/features/quantization/gguf.md:7`、`bnb.md:8`）。还在说"vLLM 内置 GGUF"的资料都过期了。
第二，**`--quantization fp8` 不再做在线量化**。`vllm/model_executor/layers/quantization/fp8.py:106` 直接抛错并让你改用 `fp8_per_tensor`——`fp8` 现在专指"读一个已经序列化成 FP8 的 checkpoint"。

`--quantization` 和模型自带 quant config 的关系是**必须一致**（`vllm/config/model.py:1397` 的 `_verify_quantization`）：不给参数时采用 checkpoint 的 `quant_method`；给了但对不上，`vllm/config/model.py:1473` 报 `does not match`。唯一例外是 override 机制——`auto_gptq`/`awq`/`mxfp4` 这类方法可以"认领"别的 checkpoint 格式，按 `override_quantization_method` 的顺序探测（`vllm/config/model.py:1410` 起的 `overrides` 列表）。

还有一个反复被问的问题：**量化权重不会缩小 KV cache**。上表最后一列全是"无"。要缩 KV 必须单独开 `--kv-cache-dtype`，那是另一条路径（第 6 节）。

## 5. 权重加载

load format 的全集在 `vllm/model_executor/model_loader/__init__.py:33` 的 `LoadFormats`，映射表紧跟其后（`:50` 的 `_LOAD_FORMAT_TO_MODEL_LOADER`）：

| `--load-format` | 走哪个 loader | 适用 |
|---|---|---|
| `auto` / `safetensors` / `pt` | `DefaultModelLoader` | 绝大多数 HF checkpoint，`auto` 优先 safetensors、回退 bin |
| `sharded_state` | `ShardedStateLoader` | 预切分 checkpoint，加载时间与 TP 无关 |
| `tensorizer` / `runai_streamer` / `instanttensor` | 各自 loader | 对象存储、流式加载、直 I/O |
| `ipc_cache` | `IpcModelLoader` | 配 `vllm preload` 的权重缓存守护进程，重启秒起 |

多卡怎么切权重，答案在 `sharded_weight_loader`（`vllm/model_executor/model_loader/weight_utils.py:1616`）：

```python
shard_size = param.data.shape[shard_axis]      # 本地参数已经是 1/TP 大小
start_idx = tp_rank * shard_size
loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
```

注意顺序：**checkpoint 里是完整张量，每个 rank 都读完整文件，然后只 narrow 出自己那一片**。TP=8 时磁盘上就被读了 8 遍——`docs/configuration/conserving_memory.md:24` 明说了这点，并建议转成预切分格式。这正是 `sharded_state` 存在的理由：每个 worker 只读自己的 shard，加载时间从"随 TP 线性增长"变成常数（`vllm/model_executor/model_loader/sharded_state_loader.py:30` 的类注释）。

加载时间粗算：70B BF16 是 141 GB，NVMe 顺序读假设 3 GB/s，单进程约 47 s；TP=8 共读同一份文件，页缓存未命中时有效带宽被抢，实测常在分钟级。三个旋钮都在 `LoadConfig` 里：`safetensors_load_strategy` 选 `lazy`（mmap，默认）/`eager`（整文件读进内存，网络盘推荐）/`prefetch`（先灌页缓存）（`vllm/config/load.py:65`），预读块大小默认 16 MiB（`vllm/config/load.py:13`），预取线程数默认 8（`vllm/config/load.py:12`）。它们不是"更快"，而是把随机读换成顺序读；文件在本地 NVMe 上时，mmap 的 `lazy` 反而最快。

## 6. KV cache 量化

KV 的字节数是三项相乘：`2 × L × n_kv × d_head × b`，FP8 只动最后一项 `b`——从 2 变 1，池子直接翻倍——70B 的 8 k 并发从 18 变成 36，这是所有量化手段里**性价比最高的一刀**，因为它同时减少了每步 decode 要读的字节。

`cache_dtype` 的可选值在 `vllm/config/cache.py:44` 的 `CacheDType`，比"fp8 二选一"复杂得多：`fp8`/`fp8_e4m3`/`fp8_e5m2`、`fp8_ds_mla`、`nvfp4`/`nvfp4_ds_mla`、`int4_per_token_head`/`int8_per_token_head`/`fp8_per_token_head`、`turboquant_*`、`octave_*`、`ultraquant_4bit`。默认 `auto` 跟模型 dtype 走；DeepSeek-V3.2 一类模型默认就是 fp8（`vllm/config/cache.py:245` 的 docstring），此时显式设 `bfloat16` 才是"不量化"。

**为什么要 per-head scale？** 因为 K 的每个 head 动态范围不同，一个 per-tensor scale 会被最活跃的 head 拽偏。vLLM 有两条 per-head 路线，别混：

- **静态 per-head**：scale 在 checkpoint 里，`q_scale=[num_heads]`、`k/v_scale=[num_kv_heads]`，需要 llm-compressor 离线校准，且只有 FlashAttention 后端支持（`docs/features/quantization/quantized_kvcache.md:15`）；
- **动态 per-token-head**：scale 运行时算，`fp8_per_token_head` 这类值由 Triton 后端在 `customize_spec` 里把 fp32 scale 内联进每个 head 的数据后面（`vllm/v1/attention/backends/triton_attn.py:308`），每 head 每 token 多 8 字节。

代价可以精确算：Llama-3-8B 每 head 的 bf16 是 `(128+128)×2 = 512 B`，换成 `fp8_per_token_head` 是 `256×1 + 8 = 264 B`，**51.6%**——不是整半，多出来的就是那 8 字节 scale。

**MLA 更敏感**，原因在存储形态。MLA 每 token 只有一个 latent（`head_size_v=0`，`vllm/v1/kv_cache_interface.py:737`），没有"多 head 平均"来摊平误差；DeepSeek 的 `fp8_ds_mla` 布局是 656 B/token：NoPE 512 维切成 **4 个 128 维 tile，每 tile 一个 fp32 的 2 的幂次 scale**，RoPE 64 维仍是 bf16（`vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py:81`）。所以 MLA 的 KV 量化在代码里就是按 128 维 tile 做的：拿通用的 per-tensor 路径去量化 MLA，精度风险要自己承担。不放心就用 `--kv-cache-dtype-skip-layers` 把指定层（或 `sliding_window` 这类层型）留在原生精度（`vllm/config/cache.py:285`）。

## 7. 调优 playbook

先看默认值怎么分档（`vllm/engine/arg_utils.py:2951` 的 `get_batch_defaults`）：≥160 GiB（B200/B300）16384/1024；≥70 GiB 且非 A100（H100/H200）**server 8192/1024**；其余 8192/2048。H100 上开箱即用就是 `max_num_batched_tokens=8192`、`max_num_seqs=1024`。

| 目标 | 主要旋钮 | 建议组合 | 别做 |
|---|---|---|---|
| 最低 TTFT（低并发、长 prompt） | `max_num_batched_tokens` ↑ | 8192→16384，prefix caching 开，`-O2` | 别把 budget 压到 2048，长 prompt 会被切成一串 chunk |
| 最低 ITL（流式体验） | `max_num_batched_tokens` ↓、`max_num_seqs` ↓ | 2048 + 适度 `max_num_seqs` + `--performance-mode interactivity` | 别在同一实例上追吞吐，两者抢同一个 budget |
| 最大吞吐（离线批处理） | 三者一起 ↑ | 16384 + `max_num_seqs` 512~1024 + `-O2`；`--performance-mode throughput` | 别开 `enforce_eager` |
| 单卡最大并发 | `max_model_len` ↓、KV 量化 | `--max-model-len` 按真实业务设，`--kv-cache-dtype fp8` | 别为了"以后可能用"设 131072 |
| 启动最快（开发循环） | `-O0` / `enforce_eager` | `-O0` 连 CUDA graph 一起关（`vllm/config/vllm.py:284`） | 别拿它跑生产 |

`-O` 的语义写死在 `vllm/config/vllm.py:147` 的 `OptimizationLevel`：`-O0` 不编译不捕图（`:284`）；`-O1` 编译 + PIECEWISE（`:305`）；`-O2` 默认，FULL_AND_PIECEWISE（`:326`）；`-O3` 目前等于 O2。图越大、捕获越多，`M_graph` 越大——这是 `-O` 与"显存不够"之间唯一的联系。

`--performance-mode` 是这一版新增的粗粒度旋钮（`vllm/config/vllm.py:458`）：`interactivity` 只捕 1..32 的细粒度尺寸，小 batch padding 少；`balanced`/`throughput` 从 1,2,4 一路按 8、16 的步长捕到 `min(2×max_num_seqs, 512)`（`vllm/config/vllm.py:2608`、`:2630`）。H100 上默认约 50 张图，图池按最大 batch 的中间激活计费。

chunked prefill 默认开着（`vllm/config/scheduler.py:134` 的 `enable_chunked_prefill = True`），它 decode 优先、把 prefill 塞进空隙；`docs/configuration/optimization.md:64` 给了口径：budget 小（2048）ITL 好，budget 大 TTFT 好，追吞吐建议 >8192。

A/B 要用 `vllm bench`，不是旧脚本——`benchmarks/benchmark_serving.py` 现在只有 17 行，是个让你改用 CLI 的弃用提示（`benchmarks/benchmark_serving.py:6`）：

```bash
vllm bench serve --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 512 --max-concurrency 64 --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --goodput ttft:500 tpot:50
```

`--max-concurrency` 是"允许多少个同时在跑"，`--request-rate` 是"多快发一个"（默认 inf = 全打在 0 时刻，`vllm/benchmarks/serve.py:1746`）；压吞吐用前者，压排队行为用后者。`--goodput` 按 SLO 过滤（`vllm/benchmarks/serve.py:1868`），报出的 `request_goodput` 才是"满足 TTFT<500 ms 且 TPOT<50 ms 的请求比例"——只报 throughput 的 A/B 没意义，把 batch 撑大总能换来吞吐。

## 8. 观测：三个指标就能定位

指标定义都在 `vllm/v1/metrics/loggers.py`：

| 指标 | 出处 | 看什么 |
|---|---|---|
| `vllm:num_requests_running` / `waiting` | `loggers.py:527`、`:537` | running 是否顶到 `max_num_seqs` |
| `vllm:num_requests_waiting_by_reason` | `loggers.py:547` | `capacity`（等显存/槽位）vs `deferred`（等 LoRA、KV 传输） |
| `vllm:kv_cache_usage_perc` | `loggers.py:619` | 稳态是否长期 >0.95 |
| `vllm:num_preemptions` | `loggers.py:738` | 非 0 就是池子不够 |
| `vllm:prefix_cache_queries` / `hits` | `loggers.py:661`、`:672` | 命中率 = hits/queries，按 token 计 |
| `vllm:time_to_first_token_seconds` / `inter_token_latency_seconds` | `loggers.py:930`、`:940` | TTFT / ITL 直方图 |
| `vllm:request_queue_time_seconds` | `loggers.py:970` | 排队时间，区分"算得慢"和"排得久" |

判定树很短。

1. **KV usage 长期 >0.95 且 preemption >0** → 显存不够。先降 `max_model_len` 或 `max_num_seqs`，再考虑 `--kv-cache-dtype fp8`，最后才加卡。
2. **KV usage <0.7、running 远小于 `max_num_seqs`、waiting 里 `capacity` 不为 0** → 不是显存，是 token budget 或调度策略：`max_num_batched_tokens` 太小、chunked prefill 切得太碎，或者长 prompt 把 budget 吃光。
3. **waiting=0、queue_time 接近 0、但 TTFT 高** → 前端或 prefill 本身慢：查 tokenize 线程池、多模态预处理、prefix cache 命中率。

第 2 条最容易被误诊成"显存不足"然后去加卡。区分办法只有一个：**running 没顶满而 waiting 有人在等，瓶颈就不在显存**。

## 9. 排障顺序

1. **先对启动日志的两行数字**：`Available KV cache memory: X GiB`（`vllm/v1/worker/gpu_worker.py:844`）和 `num_gpu_blocks`。手算一遍（第 2 节），对不上就是 profiling 被别的进程干扰了。
2. **算单请求占用**：`max_model_len × bytes/token`。它接近 KV 池时，`max_model_len` 就是元凶。
3. **看 preemption 计数**：非 0 说明调度器在拆东墙补西墙，而这条路径最贵——V1 默认 RECOMPUTE，被抢占的请求要重算 prefill。
4. **看 prefix cache 命中率**：命中率低而 QPS 高，说明每个请求都在重算同一段 system prompt；先修命中，再谈加卡。
5. **最后才动量化**：它改变的是精度和 kernel 路径，不是"免费的显存"，该作为账算完后的优化，而不是第一步。

## 10. 四个常见坑

- **OOM 有三种，别混**：启动期（`requested` 超过空闲，`vllm/v1/worker/utils.py:796` 直接报错并让你降 util）、capture 期（图没被估进去）、运行期（别的进程抢显存 / 多实例叠加）。修法完全不同。
- **`max_model_len` 设太大 = KV 池太小**：它在并发公式分母上。设成业务真实上限 + 余量，或 `--max-model-len auto` 让引擎按显存反推（`vllm/config/model.py:266`）。
- **量化后反而变慢**：`mxfp4` 的线性层没实现，会回退成未量化（`vllm/model_executor/layers/quantization/mxfp4.py:102`），权重一点没省；W4A16 只在 decode（带宽受限）赚，prefill（算力受限）要多付 dequant；`-O2` 的 `fuse_attn_quant` 依赖 `IS_QUANTIZED`，而它当前恒为 False（`vllm/config/vllm.py:164`），量化模型拿不到这个融合。
- **多实例共享一张卡**：`gpu_memory_utilization` 是 per-instance 的，两个 0.5 的实例各自 profiling；后启动的撞上对方释放显存，还会触发 `:803` 的断言。

## 小结

- 每卡显存是五项之和；能调的只有 KV，其上限 = `total × util − non_kv − cudagraph`（`vllm/v1/worker/gpu_worker.py:824`）。
- `gpu_memory_utilization` 默认不是 0.92：可扩展 KV cache 生效时是 1.0，否则 0.92（`vllm/config/cache.py:489`）。
- 70B/8 卡在 8 k 下并发约 18，8B 单卡约 50；`max_model_len` 从 8 k 提到 128 k，并发掉到 1 和 3。
- 量化权重不缩 KV，缩 KV 只有 `--kv-cache-dtype` 一个开关：fp8 让池子翻倍，per-token-head 版本额外付 8 B/head/token（≈51.6% 而非 50%）。
- 加载慢先分清"读了几遍"：默认每个 TP rank 都读完整 checkpoint，`sharded_state` 才能让加载时间与 TP 无关。
- 诊断只看三件事：KV usage、`waiting_by_reason`、preemption；running 没满而 waiting 有人，就不是显存问题。

下一篇回到导读，把十三篇的账本合起来看：[推理引擎的四个账本]({{< relref "vllm-00-index" >}})。

**官方文档**：[Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)、[Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)。
