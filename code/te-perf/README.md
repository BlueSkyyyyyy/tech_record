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
- **fused_attn fwd/bwd** shape 为 `(batch, seqlen, num_heads, head_dim)`，包含
  `(1,512,16,128) (1,1024,16,128) (1,2048,16,128) (2,2048,16,128) (4,2048,16,128)
   (8,2048,16,128) (1,4096,16,128) (2,4096,16,128) (4,1024,32,128)
   (8,1024,32,128) (4,8192,16,128)`，
  精度覆盖 `bf16 / fp16`，mask 为 causal，bias 为 no_bias，training=True，dropout=0。

## 指标口径

- **time(us)**：每个 case 的中位运行时间（含 launch overhead，见下方说明）。
- **GB/s**：按算子实际读写的最小数据量估算（读 Q/K/V + 写 O 等），忽略中间 S/P 等。
- **AI(flop/B)**：算术强度 = FLOPs / 实际搬运字节数，用于落 roofline 图。
- **%BW / %TC**：实测带宽占 HBM 峰值带宽 / 实测算力占 tensor-core 峰值的百分比。
- **TFLOPS**：attention 按 `4*b*s*h*s*d`（fwd）、`8*b*s*h*s*d`（bwd，约 2x fwd）计。
- **launch overhead**：脚本会先跑一个最小 `fill_` kernel 测出单次内核启动的固定
  开销（约 3–10 us/call），小 shape 时应从 `time(us)` 中扣掉它才是纯内核执行时间。
- **H100 roofline 常量**（硬编码，脚本末尾会打印）：
  - FP16/BF16 tensor-core dense peak ≈ 989.4 TFLOPS（132 SM × 1.980 GHz）
  - FP32 CUDA-core peak ≈ 66.9 TFLOPS
  - HBM3 带宽 ≈ 3.35 TB/s
  - roofline 拐点（ridge）：FP16/BF16 ≈ 295 FLOP/byte，FP32 ≈ 20 FLOP/byte

## roofline 分析要点

- **rmsnorm\*** 是纯内存受限算子（AI ≈ 2–4 FLOP/byte，远低于拐点），性能上限 = HBM
  带宽，实测 `%BW` 应该随 shape 增大而接近 100%。
- **rmsnorm_bwd_add** 是 `rmsnorm_bwd` 的融合残差版本：forward 为 `z = rmsnorm(x) + add`，
  反向在一个 kernel 里同时算出 `dx`（含 add 的梯度）与 `dw`，比「单独 bwd + 单独 add 反向」
  少一次全量读写，等价带宽通常更高、更接近峰值。
- **fused_attn*** 随 seqlen 增长从内存受限过渡到计算受限（AI 穿越拐点），
  `%TC` 应在大 seqlen 时逼近 tensor-core 峰值。
- 与 roofline 的差距主要来自：launch overhead（小 shape）、未完全饱和的带宽、
  causal mask 的 wasted FLOPs、backward 的额外访存、以及 kernel 实现本身的效率。

## 说明

- 使用 CUDA event 计时，warmup + 多次取中位数，避免首跑抖动。
- TE 的 rmsnorm / attention 均从 `transformer_engine_torch` / `cpp_extensions.fused_attn`
  直接调用原生 kernel（等同 `te.rmsnorm*` / `te.fused_attn*` 底层实现）。
- 环境变量保持 TE 默认（`NVTE_FUSED_ATTN=1`、`NVTE_FUSED_ATTN_USE_FAv2_BWD=0`）。
