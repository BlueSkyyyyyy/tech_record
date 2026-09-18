# TE 算子性能测试（rmsnorm / rmsnorm_bwd / fused_attn_bwd）

测试 `transformer_engine`（TE）三个算子的性能，用于与 H100 roofline 对比：

- `rmsnorm_fwd`（即 `te.rmsnorm`）
- `rmsnorm_bwd`（即 `te.rmsnorm_bwd`）
- `rmsnorm_bwd_add`（即 `te.rmsnorm_bwd_add`，add==True 的融合残差反向）
- `fused_attn_fwd` / `fused_attn_bwd`（即 `te.fused_attn_bwd`）

## 运行方式

本机 GPU 正在跑推理，**请勿在推理期间运行**。等 GPU 空闲后，进入容器执行：

```bash
# 1. 进入容器 kimi26_train
docker exec -it kimi26_train bash

# 2. 把脚本拷进容器（或在容器内直接挂载宿主路径运行）
#    脚本与本目录同步，直接在宿主侧写好的 ~/tech_record/code/te-perf/ 下
cd ~/tech_record/code/te-perf

# 3. 先做一次正确性校验（一次性跑通 API，确认无误）
python bench_te.py --check

# 4. 完整性能测试（默认 warmup=10, repeat=50，输出屏幕结果 + CSV）
python bench_te.py --warmup 10 --repeat 100 --csv te_perf.log.csv

# 只测 rmsnorm（如果只想分开测）
python bench_te.py --rmsnorm-only --warmup 10 --repeat 100 --csv rmsnorm.csv

# 只测 attention
python bench_te.py --attn-only --warmup 10 --repeat 100 --csv attn.csv
```

把 `te_perf.log.csv`（以及屏幕 `stdout` 全文）保存下来，用于后续写博客时分析 roofline 差异。

## 测试矩阵

- **rmsnorm / rmsnorm_bwd / rmsnorm_bwd_add** shape 为 `(rows, cols) = (batch*seq, hidden)`，包含
  `(128,512) (256,512) (64,1024) (128,1024) (512,512) (1024,512) (1024,1024)
   (1024,2048) (2048,1024) (2048,2048) (4096,2048) (4096,4096) (8192,2048)
   (8192,4096) (16384,2048) (32768,2048)`，精度覆盖 `fp32 / bf16 / fp16`。
- **fused_attn fwd/bwd** shape 为 `(batch, seqlen, num_heads, qk_head_dim[, v_head_dim])`，
  4 元组为普通 MHA（qk==v），5 元组为 MLA（qk≠v）。包含
  `(1,512,16,128) (1,1024,16,128) (1,2048,16,128) (2,2048,16,128) (4,2048,16,128)
   (8,2048,16,128) (1,4096,16,128) (2,4096,16,128) (1,1024,32,128)
   (4,1024,32,128) (8,1024,32,128) (4,8192,16,128)`，
  以及生产模型形状（batch=1, seq=4096）：
  - `(1,4096,64,192,128)` — Kimi-K2.6 MLA（qk=nope128+rope64=192，v=128，64 头）；
  - `(1,4096,64,128,128)` — dsv4 DSA indexer（64 头，head_dim=128）；
  - `(1,4096,32,128,128)` — dsv4.1 DSA indexer（32 头，head_dim=128）。

  精度覆盖 `bf16 / fp16`，mask 为 causal，bias 为 no_bias，training=True，dropout=0。
  其中 `(1,1024,32,128)` 额外测一组 **FP8**（fwd QKV/S/O 用 E4M3，bwd dO/dP/dQKV 用
  E5M2；QKV/dO 在计时区外预先量化，故测到的仍是纯 FP8 fused-attn kernel）。TE fused_attn
  **不支持 fp32**（`FusedAttnBackend` 仅有 F16_max512 / F16_arbitrary / FP8）。
  **限制**：dsv4 / dsv4.1 主注意力 `head_dim=512`（qk=448+64，v=512）超出 TE fused_attn
  在 H100 上的支持范围（最大 256，且 `qk=256,v=128` 也不支持），这两个模型实际用自研
  CSA/DSA 稀疏注意力 kernel，无法用 TE fused_attn 测试，故只测其 DSA indexer。

## 指标口径

- **time(us)**：每个 case 的**纯 device kernel 执行时间**（CUPTI / `torch.profiler` 实测），
  统计一次调用所 launch 的全部 kernel（含 device memset）的 device 时长之和，对 repeat 次调用取平均，
  **已排除 launch / 主机派发开销**。
- **GB/s**：按算子实际读写的最小数据量估算（读 Q/K/V + 写 O 等），忽略中间 S/P 等。
- **AI(flop/B)**：算术强度 = FLOPs / 实际搬运字节数，用于落 roofline 图。
- **%BW / %TC**：实测带宽占 HBM 峰值带宽 / 实测算力占 tensor-core 峰值的百分比。
- **TFLOPS**：attention 按 `4*b*s*h*s*d`（fwd）、`8*b*s*h*s*d`（bwd，约 2x fwd）计。
- **计时方法**：主指标用 CUPTI（`torch.profiler`）直接测 **device kernel 时间**，天然不含 launch。
  脚本开头另跑一个最小 `fill_` kernel 用 CUDA event 测 wall 值，用于展示主机派发开销
  （wall ≈ 12 us、device ≈ 1 us，即 host 开销 ~11 us/call）。注意 **launch 开销不是常数**
  （主机派发越重、gap 越大），所以不能靠"wall − 固定值"得到纯内核时间——这也是改用 CUPTI 的原因。
- **L2 常驻**：小 shape 反复调用时数据可能常驻 L2（H100 ≈ 50MB），此时 `%BW`（按 HBM 峰值折算）
  会偏高甚至 >100%，属正常现象；判断 HBM 效率请看工作集大于 L2 的大 shape。
- **H100 roofline 常量**（硬编码，脚本末尾会打印）：
  - FP16/BF16 tensor-core dense peak ≈ 989.4 TFLOPS（132 SM × 1.980 GHz）
  - FP8 tensor-core dense peak ≈ 1978.8 TFLOPS（FP8 数据点的 `%TC` 按此折算）
  - FP32 CUDA-core peak ≈ 66.9 TFLOPS
  - HBM3 带宽 ≈ 3.35 TB/s
  - roofline 拐点（ridge）：FP16/BF16 ≈ 295 FLOP/byte，FP8 ≈ 591 FLOP/byte，FP32 ≈ 20 FLOP/byte

## roofline 分析要点

- **rmsnorm\*** 是纯内存受限算子（AI ≈ 2–4 FLOP/byte，远低于拐点），性能上限 = HBM
  带宽，实测 `%BW` 应该随 shape 增大而接近 100%。
- **rmsnorm_bwd_add** 是 `rmsnorm_bwd` 的融合残差版本：forward 为 `z = rmsnorm(x) + add`，
  反向在一个 kernel 里同时算出 `dx`（含 add 的梯度）与 `dw`，比「单独 bwd + 单独 add 反向」
  少一次全量读写，等价带宽通常更高、更接近峰值。
- **fused_attn*** 随 seqlen 增长从内存受限过渡到计算受限（AI 穿越拐点），
  `%TC` 应在大 seqlen 时逼近 tensor-core 峰值。
- 与 roofline 的差距主要来自：**小 shape 数据常驻 L2**（`%BW` 会虚高）、未完全饱和的带宽、
  causal mask 的 wasted FLOPs、backward 的额外访存、以及 kernel 实现本身的效率。

## 用 ncu / Nsight Systems 剖析单个算子

脚本里的计时是「墙钟中位数」，只能拿到总耗时，看不到带宽/占用率/停顿原因等细节。
要诊断单个 kernel（例如 `rmsnorm_bwd` 4096×4096），用 Nsight 工具深入剖析：

### ncu（Nsight Compute，看单 kernel 的 GPU 内部指标）

先写一个只跑目标算子的最小脚本 `prof_single.py`（已提供），避免 bench 里一大堆 kernel 互相干扰：

```bash
# bf16 反传，单 kernel profile（--launch-skip 4 跳过前面 randn/warmup 的 kernel）
ncu --set full --kernel-name regex:"rmsnorm" \
    --launch-count 1 --launch-skip 4 \
    python prof_single.py --rows 4096 --cols 4096 --dtype bfloat16 --bwd --iters 10

# 只看关键指标（速度/DRAM带宽/SM吞吐/活跃warp/占用率）
ncu --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__mix_inst_stranded.avg.pct_of_peak_sustained_active \
    --kernel-name regex:"rmsnorm" --launch-count 1 --launch-skip 4 \
    python prof_single.py --rows 4096 --cols 4096 --dtype bfloat16 --bwd --iters 10
```

rmsnorm 是**内存受限**算子，重点读 ncu 报告的 **SpeedOfLight** 与 **Memory Workload
Analysis** 两节：

- `dram__throughput` 应接近 HBM 峰值（H100 ≈ 3.35 TB/s）
- `sm__throughput` 通常很低（被访存拖住，属正常）
- 「actual / peak DRAM 带宽」即算子的真实带宽利用率，可对照 bench 打印的 `%BW`

### Nsight Systems（nsys，看端到端时间线，不是看单算子性能）

`nsys` 记录的是**时间线**（kernel 起止、内存拷贝、API 调用、调度重叠），用来分析并发/气泡/开销，
**不用来做单算子的性能剖析**（它拿不到 SM 内部指标）。适合看：

- 每个 kernel 的启动顺序与耗时（对应 wall-clock 时间）
- kernel 之间是否有空闲气泡、是否与 H2D/D2H 拷贝重叠
- launch overhead 与任务排队情况

```bash
# 端到端跑一遍 bench，抓时间线
nsys profile --stats=true -o rmsnorm_bwd \
    python prof_single.py --rows 4096 --cols 4096 --dtype bfloat16 --bwd --iters 10
```

结论：**单算子性能用 ncu**（SM/DRAM 指标），**端到端时序/并发用 nsys**。

## 说明

- 主指标用 CUPTI（`torch.profiler`）测纯 device kernel 时间（warmup + 多次取平均），已排除 launch 开销；
  脚本仍用 CUDA event 测一个最小 `fill_` kernel 的 wall 值，仅用于展示主机派发开销的量级。
- TE 的 rmsnorm / attention 均从 `transformer_engine_torch` / `cpp_extensions.fused_attn`
  直接调用原生 kernel（等同 `te.rmsnorm*` / `te.fused_attn*` 底层实现）。
- 环境变量保持 TE 默认（`NVTE_FUSED_ATTN=1`、`NVTE_FUSED_ATTN_USE_FAv2_BWD=0`）。
