# fa-bwd — FlashAttention 反向的整理、重写与调优

借鉴 `~/github/flash-attention`（FA2/FA3）与 TransformerEngine 的实现，在 GPU 上把
**flash-attention 反向**整理成 **单文件 / 两文件** 两种形式（fp16 / bf16 / **fp8 最重点**），
编译、用 ncu 剖析 bound、与 TE 做数值与性能对标，并产出分析文档。

- 目标卡：**目标 AI 卡**；当前在 **H100 sm90** 上开发验证，代码按可替换层抽象，便于移植到目标卡。
- 生产形状（GQA/MQA/MLA，对齐目标卡配置）：见 `harness/fa_bwd_bench.py` 的 `REQUESTED_SHAPES` 与 `docs/04` §7。
- 控制面板：[ROADMAP.md](ROADMAP.md)｜优化手段梳理：[docs/00-fa-bwd-optimization-catalog.md](docs/00-fa-bwd-optimization-catalog.md)
- 数值 I/O dump 目录：**`/home/xieminglin/proj/output/fa-bwd/<case>/`**（CPU npy，便于 load 比对）

## 目录

```
fa-bwd/
  ROADMAP.md                 # 任务/进度/下一步
  docs/                      # 分析文档
  harness/
    probe_refs.py            # 探测 FA/TE 反向可用性 + 对拍（已验证）
    fa_bwd_bench.py          # dump / 对拍 / CUPTI 基准
  src/<dtype>/               # 我们的实现：onefile / two-file
  scripts/                   # lab / run / ncu / autopilot
```

## 快速开始

```bash
cd ~/proj/tech_record/code/flash-attention/fa-bwd

# 参考实现探测（FA2.7.4 与 TE2.14 反向 vs fp32 ref）
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python "$PWD/harness/probe_refs.py"

# 生成某个 case 的输入并 dump ref/fa/te 的 I/O（可在 /home/xieminglin/proj/output/fa-bwd 查看）
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python "$PWD/harness/fa_bwd_bench.py" dump \
    --dtype fp16 --shape 1 4096 16 128 causal

# 性能基准（FA vs TE）
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python "$PWD/harness/fa_bwd_bench.py" bench \
    --dtype fp16 --shape 1 4096 16 128 causal

# 编译/运行我们自己的 kernel
scripts/run.sh src/fp16/fa_bwd_fp16_onefile.cu
scripts/ncu.sh src/fp16/fa_bwd_fp16_onefile.cu --set full --kernel-name regex:bwd
```

## 基线（fp16, causal, H100）

| shape | FA 2.7.4 | TE 2.14 |
|---|---|---|
| B1 S4096 H16 D128 | ~132 TFLOPS | ~230 TFLOPS |

（CUPTI 纯 device 时间；bwd FLOPs≈4·b·s²·h·d。详见 docs。）
