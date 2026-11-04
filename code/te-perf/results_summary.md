# TE 算子性能测试结果汇总

> `kernel耗时` 为 **CUPTI 实测的纯 device kernel 执行时间**（µs）：由 `torch.profiler` 统计一次调用所 launch 的全部 kernel（含 device memset）的 device 时长之和，对 repeat 次调用取平均，**不含 launch / 主机派发开销**。设备均为 NVIDIA H100。原始数据见 `te_perf.csv` / `perf.log`（含纯 device 时间与由此计算的 GB/s、TFLOPS）；Kimi-K3 MLA 行来自 `mla_kimi_k3.csv` / `mla_kimi_k3.log`（脚本 `bench_mla_kimi_k3.py`，详见 `results_mla_kimi_k3.md`）；Kimi-K3 AttnRes 行来自 `attnres_kimi_k3.csv` / `attnres_kimi_k3.log` / `attnres_schedule.log`（脚本 `bench_attnres_kimi_k3.py`，详见 `results_attnres_kimi_k3.md`）。
>
> 注意：小 shape 反复调用时数据可能常驻 L2（H100 约 50MB），此时按 HBM 峰值折算的 `%BW` 会偏高、甚至超过 100%，属正常现象，不代表超过 HBM 带宽；判断 HBM 效率请看工作集大于 L2 的大 shape。

> **生产模型融合注意力形状**（batch=1, seq=4096）：
> - `(1, 4096, 64, 192, 128)` — Kimi-K2.6 MLA（qk=nope128+rope64=192，v=128，64 头）
> - `(1, 4096, 96, 192, 128)` — **Kimi-K3 MLA**（同几何、96 头、**NoPE**；TE 训练 fwd+bwd 均可测，见 [`results_mla_kimi_k3.md`](results_mla_kimi_k3.md)）
> - `(1, 4096, 64, 128, 128)` — dsv4 DSA indexer（64 头，head_dim=128）
> - `(1, 4096, 32, 128, 128)` — dsv4.1 DSA indexer（32 头，head_dim=128）
> - `(1, 4096, 40, 128, 128, 8)` — Qwen3-8B GQA（q=40，kv=8）
> - `(1, 4096, 32, 128, 128, 4)` — Qwen3-30B-A3B GQA（q=32，kv=4）
> - `(1, 4096, 64, 128, 128, 4)` — Qwen3-235B-A22B GQA（q=64，kv=4）
> - `(1, 4096, 64, 128, 128, 1)` — DeepSeek-V4-Pro DSA indexer（MQA：q=64 共享 1 个压缩 KV 头）
>
> **上述模型形状的 seq=1024 短序列版本**（与目标卡的 shape 配置对齐）：
> - `(1, 1024, 40, 128, 128, 8)` / `(1, 1024, 32, 128, 128, 4)` / `(1, 1024, 64, 128, 128, 4)` — Qwen3-8B / 30B-A3B / 235B-A22B GQA @ seq=1024；
> - `(1, 1024, 64, 128, 128, 1)` — DeepSeek-V4-Pro DSA indexer @ seq=1024。
>
> **MLA head_dim=512 小 shape**（目标卡对标点）：`(1, 256, 2, 512)` / `(1, 512, 4, 512)` / `(1, 1024, 2, 512)`；TE 训练 bwd 拒绝 `head_dim>256`，故仅列入推理 fwd 段。
>
> **AttnRes**（FLA `fused_attnres`，表末 6 行）：`attnres_fwd_bwd` 是一次调用的 **fwd+bwd 合计**（µs），shape 为 `(batch, seq, L, hidden)`，其中 L 是 depth source 数（Kimi-K3 的实际范围是 2…9，这里取两端 L=2 / L=9）；dtype bf16、融合输出 RMSNorm、`checkpoint_level=1`。详见下方「Kimi-K3 AttnRes 专项」。
>
> 注：dsv4 / dsv4.1 / DeepSeek-V4-Pro 主 MLA 注意力 `head_dim=512`（qk=448+64，v=512）超出 TE fused_attn 在 H100 上训练 bwd 的支持范围（最大 256），仅推理 fwd 段 `fused_attn_fwd_infer` 可测；其生产实现是自研 CSA/DSA 稀疏注意力 kernel，故这里只测其 DSA indexer / 推理 fwd。

| op name | dtype | shape | device | kernel耗时(µs) |
|---|---|---|---|---|
| rmsnorm_fwd | torch.float32 | (128, 512) | H100 | 1.9 |
| rmsnorm_fwd | torch.bfloat16 | (128, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float16 | (128, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float32 | (256, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.bfloat16 | (256, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float16 | (256, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float32 | (64, 1024) | H100 | 2.1 |
| rmsnorm_fwd | torch.bfloat16 | (64, 1024) | H100 | 2.1 |
| rmsnorm_fwd | torch.float16 | (64, 1024) | H100 | 2.1 |
| rmsnorm_fwd | torch.float32 | (128, 1024) | H100 | 2.1 |
| rmsnorm_fwd | torch.bfloat16 | (128, 1024) | H100 | 2.1 |
| rmsnorm_fwd | torch.float16 | (128, 1024) | H100 | 2.1 |
| rmsnorm_fwd | torch.float32 | (512, 512) | H100 | 2.1 |
| rmsnorm_fwd | torch.bfloat16 | (512, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float16 | (512, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float32 | (1024, 512) | H100 | 2.3 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float16 | (1024, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float32 | (1024, 1024) | H100 | 3.3 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 1024) | H100 | 2.6 |
| rmsnorm_fwd | torch.float16 | (1024, 1024) | H100 | 2.5 |
| rmsnorm_fwd | torch.float32 | (1024, 2048) | H100 | 5.3 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 2048) | H100 | 3.7 |
| rmsnorm_fwd | torch.float16 | (1024, 2048) | H100 | 3.6 |
| rmsnorm_fwd | torch.float32 | (2048, 1024) | H100 | 4.8 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 1024) | H100 | 3.5 |
| rmsnorm_fwd | torch.float16 | (2048, 1024) | H100 | 3.5 |
| rmsnorm_fwd | torch.float32 | (2048, 2048) | H100 | 7.8 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 2048) | H100 | 4.9 |
| rmsnorm_fwd | torch.float16 | (2048, 2048) | H100 | 4.9 |
| rmsnorm_fwd | torch.float32 | (4096, 2048) | H100 | 24.3 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 2048) | H100 | 7.9 |
| rmsnorm_fwd | torch.float16 | (4096, 2048) | H100 | 7.9 |
| rmsnorm_fwd | torch.float32 | (4096, 4096) | H100 | 49.8 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 4096) | H100 | 26.0 |
| rmsnorm_fwd | torch.float16 | (4096, 4096) | H100 | 26.1 |
| rmsnorm_fwd | torch.float32 | (8192, 1024) | H100 | 23.9 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 1024) | H100 | 7.5 |
| rmsnorm_fwd | torch.float16 | (8192, 1024) | H100 | 7.4 |
| rmsnorm_fwd | torch.float32 | (8192, 2048) | H100 | 49.4 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 2048) | H100 | 25.8 |
| rmsnorm_fwd | torch.float16 | (8192, 2048) | H100 | 25.5 |
| rmsnorm_fwd | torch.float32 | (8192, 4096) | H100 | 96.7 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 4096) | H100 | 51.5 |
| rmsnorm_fwd | torch.float16 | (8192, 4096) | H100 | 51.3 |
| rmsnorm_fwd | torch.float32 | (16384, 2048) | H100 | 95.8 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 2048) | H100 | 51.5 |
| rmsnorm_fwd | torch.float16 | (16384, 2048) | H100 | 51.3 |
| rmsnorm_fwd | torch.float32 | (32768, 2048) | H100 | 187.2 |
| rmsnorm_fwd | torch.bfloat16 | (32768, 2048) | H100 | 98.9 |
| rmsnorm_fwd | torch.float16 | (32768, 2048) | H100 | 98.7 |
| rmsnorm_fwd | torch.float32 | (4096, 7168) | H100 | 88.9 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 7168) | H100 | 88.7 |
| rmsnorm_fwd | torch.float16 | (4096, 7168) | H100 | 88.6 |
| rmsnorm_fwd | torch.float32 | (8192, 7168) | H100 | 169.8 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 7168) | H100 | 172.3 |
| rmsnorm_fwd | torch.float16 | (8192, 7168) | H100 | 172.1 |
| rmsnorm_fwd | torch.float32 | (16384, 4096) | H100 | 189.6 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 4096) | H100 | 100.7 |
| rmsnorm_fwd | torch.float16 | (16384, 4096) | H100 | 100.4 |
| rmsnorm_fwd | torch.float32 | (16384, 7168) | H100 | 333.0 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 7168) | H100 | 336.1 |
| rmsnorm_fwd | torch.float16 | (16384, 7168) | H100 | 335.7 |
| rmsnorm_bwd | torch.float32 | (128, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (128, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd | torch.float32 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (256, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.float32 | (64, 1024) | H100 | 5.0 |
| rmsnorm_bwd | torch.bfloat16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd | torch.float16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd | torch.float32 | (128, 1024) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float32 | (512, 512) | H100 | 5.3 |
| rmsnorm_bwd | torch.bfloat16 | (512, 512) | H100 | 5.5 |
| rmsnorm_bwd | torch.float16 | (512, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.float32 | (1024, 512) | H100 | 5.7 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 512) | H100 | 5.8 |
| rmsnorm_bwd | torch.float16 | (1024, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float32 | (1024, 1024) | H100 | 6.7 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 1024) | H100 | 5.6 |
| rmsnorm_bwd | torch.float16 | (1024, 1024) | H100 | 5.6 |
| rmsnorm_bwd | torch.float32 | (1024, 2048) | H100 | 9.8 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd | torch.float16 | (1024, 2048) | H100 | 9.0 |
| rmsnorm_bwd | torch.float32 | (2048, 1024) | H100 | 8.7 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 1024) | H100 | 6.5 |
| rmsnorm_bwd | torch.float16 | (2048, 1024) | H100 | 6.5 |
| rmsnorm_bwd | torch.float32 | (2048, 2048) | H100 | 22.1 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 2048) | H100 | 11.0 |
| rmsnorm_bwd | torch.float16 | (2048, 2048) | H100 | 11.0 |
| rmsnorm_bwd | torch.float32 | (4096, 2048) | H100 | 40.5 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 2048) | H100 | 23.1 |
| rmsnorm_bwd | torch.float16 | (4096, 2048) | H100 | 23.1 |
| rmsnorm_bwd | torch.float32 | (4096, 4096) | H100 | 77.1 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 4096) | H100 | 41.0 |
| rmsnorm_bwd | torch.float16 | (4096, 4096) | H100 | 41.4 |
| rmsnorm_bwd | torch.float32 | (8192, 1024) | H100 | 38.7 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 1024) | H100 | 21.1 |
| rmsnorm_bwd | torch.float16 | (8192, 1024) | H100 | 21.1 |
| rmsnorm_bwd | torch.float32 | (8192, 2048) | H100 | 77.0 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 2048) | H100 | 40.5 |
| rmsnorm_bwd | torch.float16 | (8192, 2048) | H100 | 40.4 |
| rmsnorm_bwd | torch.float32 | (8192, 4096) | H100 | 145.3 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 4096) | H100 | 73.8 |
| rmsnorm_bwd | torch.float16 | (8192, 4096) | H100 | 74.0 |
| rmsnorm_bwd | torch.float32 | (16384, 2048) | H100 | 145.7 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 2048) | H100 | 73.3 |
| rmsnorm_bwd | torch.float16 | (16384, 2048) | H100 | 73.2 |
| rmsnorm_bwd | torch.float32 | (32768, 2048) | H100 | 281.0 |
| rmsnorm_bwd | torch.bfloat16 | (32768, 2048) | H100 | 144.9 |
| rmsnorm_bwd | torch.float16 | (32768, 2048) | H100 | 144.8 |
| rmsnorm_bwd | torch.float32 | (4096, 7168) | H100 | 189.1 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 7168) | H100 | 131.8 |
| rmsnorm_bwd | torch.float16 | (4096, 7168) | H100 | 131.5 |
| rmsnorm_bwd | torch.float32 | (8192, 7168) | H100 | 358.7 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 7168) | H100 | 238.2 |
| rmsnorm_bwd | torch.float16 | (8192, 7168) | H100 | 238.5 |
| rmsnorm_bwd | torch.float32 | (16384, 4096) | H100 | 280.1 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 4096) | H100 | 146.5 |
| rmsnorm_bwd | torch.float16 | (16384, 4096) | H100 | 146.7 |
| rmsnorm_bwd | torch.float32 | (16384, 7168) | H100 | 690.6 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 7168) | H100 | 452.0 |
| rmsnorm_bwd | torch.float16 | (16384, 7168) | H100 | 452.4 |
| rmsnorm_bwd_add | torch.float32 | (128, 512) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (256, 512) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (256, 512) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float32 | (64, 1024) | H100 | 4.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (64, 1024) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.float16 | (64, 1024) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.float32 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.float16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.float32 | (1024, 512) | H100 | 5.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float32 | (1024, 1024) | H100 | 6.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 1024) | H100 | 5.9 |
| rmsnorm_bwd_add | torch.float16 | (1024, 1024) | H100 | 6.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 2048) | H100 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 2048) | H100 | 8.9 |
| rmsnorm_bwd_add | torch.float16 | (1024, 2048) | H100 | 9.2 |
| rmsnorm_bwd_add | torch.float32 | (2048, 1024) | H100 | 8.3 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 1024) | H100 | 6.9 |
| rmsnorm_bwd_add | torch.float16 | (2048, 1024) | H100 | 6.9 |
| rmsnorm_bwd_add | torch.float32 | (2048, 2048) | H100 | 28.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 2048) | H100 | 12.9 |
| rmsnorm_bwd_add | torch.float16 | (2048, 2048) | H100 | 12.7 |
| rmsnorm_bwd_add | torch.float32 | (4096, 2048) | H100 | 50.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 2048) | H100 | 28.9 |
| rmsnorm_bwd_add | torch.float16 | (4096, 2048) | H100 | 28.9 |
| rmsnorm_bwd_add | torch.float32 | (4096, 4096) | H100 | 93.5 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 4096) | H100 | 51.7 |
| rmsnorm_bwd_add | torch.float16 | (4096, 4096) | H100 | 52.1 |
| rmsnorm_bwd_add | torch.float32 | (8192, 1024) | H100 | 48.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 1024) | H100 | 27.3 |
| rmsnorm_bwd_add | torch.float16 | (8192, 1024) | H100 | 27.2 |
| rmsnorm_bwd_add | torch.float32 | (8192, 2048) | H100 | 94.1 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 2048) | H100 | 50.8 |
| rmsnorm_bwd_add | torch.float16 | (8192, 2048) | H100 | 50.9 |
| rmsnorm_bwd_add | torch.float32 | (8192, 4096) | H100 | 179.1 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 4096) | H100 | 95.4 |
| rmsnorm_bwd_add | torch.float16 | (8192, 4096) | H100 | 96.0 |
| rmsnorm_bwd_add | torch.float32 | (16384, 2048) | H100 | 180.5 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 2048) | H100 | 94.5 |
| rmsnorm_bwd_add | torch.float16 | (16384, 2048) | H100 | 94.2 |
| rmsnorm_bwd_add | torch.float32 | (32768, 2048) | H100 | 353.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (32768, 2048) | H100 | 181.4 |
| rmsnorm_bwd_add | torch.float16 | (32768, 2048) | H100 | 181.1 |
| rmsnorm_bwd_add | torch.float32 | (4096, 7168) | H100 | 336.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 7168) | H100 | 169.3 |
| rmsnorm_bwd_add | torch.float16 | (4096, 7168) | H100 | 168.3 |
| rmsnorm_bwd_add | torch.float32 | (8192, 7168) | H100 | 655.5 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 7168) | H100 | 314.4 |
| rmsnorm_bwd_add | torch.float16 | (8192, 7168) | H100 | 313.9 |
| rmsnorm_bwd_add | torch.float32 | (16384, 4096) | H100 | 351.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 4096) | H100 | 181.3 |
| rmsnorm_bwd_add | torch.float16 | (16384, 4096) | H100 | 182.3 |
| rmsnorm_bwd_add | torch.float32 | (16384, 7168) | H100 | 1292.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 7168) | H100 | 603.9 |
| rmsnorm_bwd_add | torch.float16 | (16384, 7168) | H100 | 603.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 11.9 |
| fused_attn_fwd | torch.float16 | (1, 512, 16, 128) | H100 | 11.9 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 18.5 |
| fused_attn_fwd | torch.float16 | (1, 1024, 16, 128) | H100 | 18.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 51.1 |
| fused_attn_fwd | torch.float16 | (1, 2048, 16, 128) | H100 | 51.4 |
| fused_attn_fwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 81.6 |
| fused_attn_fwd | torch.float16 | (2, 2048, 16, 128) | H100 | 82.5 |
| fused_attn_fwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 140.7 |
| fused_attn_fwd | torch.float16 | (4, 2048, 16, 128) | H100 | 143.2 |
| fused_attn_fwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 257.0 |
| fused_attn_fwd | torch.float16 | (8, 2048, 16, 128) | H100 | 262.9 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 148.7 |
| fused_attn_fwd | torch.float16 | (1, 4096, 16, 128) | H100 | 149.9 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 255.1 |
| fused_attn_fwd | torch.float16 | (2, 4096, 16, 128) | H100 | 260.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 32, 128) | H100 | 31.5 |
| fused_attn_fwd | torch.float16 | (1, 1024, 32, 128) | H100 | 31.2 |
| fused_attn_fwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 83.4 |
| fused_attn_fwd | torch.float16 | (4, 1024, 32, 128) | H100 | 83.8 |
| fused_attn_fwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 150.7 |
| fused_attn_fwd | torch.float16 | (8, 1024, 32, 128) | H100 | 154.4 |
| fused_attn_fwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 1869.7 |
| fused_attn_fwd | torch.float16 | (4, 8192, 16, 128) | H100 | 1939.5 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 599.3 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 609.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 96, 192, 128) | H100 | 86.5 |
| fused_attn_fwd | torch.float16 | (1, 1024, 96, 192, 128) | H100 | 87.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 96, 192, 128) | H100 | 252.4 |
| fused_attn_fwd | torch.float16 | (1, 2048, 96, 192, 128) | H100 | 256.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 96, 192, 128) | H100 | 878.0 |
| fused_attn_fwd | torch.float16 | (1, 4096, 96, 192, 128) | H100 | 896.7 |
| fused_attn_fwd | torch.bfloat16 | (1, 8192, 96, 192, 128) | H100 | 3490.0 |
| fused_attn_fwd | torch.float16 | (1, 8192, 96, 192, 128) | H100 | 3713.9 |
| fused_attn_fwd | torch.bfloat16 | (1, 16384, 96, 192, 128) | H100 | 14181.2 |
| fused_attn_fwd | torch.float16 | (1, 16384, 96, 192, 128) | H100 | 14560.6 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 96, 192, 128) | H100 | 1862.7 |
| fused_attn_fwd | torch.float16 | (2, 4096, 96, 192, 128) | H100 | 1895.5 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 472.4 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 494.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 254.7 |
| fused_attn_fwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 260.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 40, 128, 128, 8) | H100 | 307.1 |
| fused_attn_fwd | torch.float16 | (1, 4096, 40, 128, 128, 8) | H100 | 312.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 32, 128, 128, 4) | H100 | 255.7 |
| fused_attn_fwd | torch.float16 | (1, 4096, 32, 128, 128, 4) | H100 | 257.9 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 128, 128, 4) | H100 | 465.9 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 128, 128, 4) | H100 | 491.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 128, 128, 1) | H100 | 475.9 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 128, 128, 1) | H100 | 487.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 40, 128, 128, 8) | H100 | 34.8 |
| fused_attn_fwd | torch.float16 | (1, 1024, 40, 128, 128, 8) | H100 | 34.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 32, 128, 128, 4) | H100 | 30.4 |
| fused_attn_fwd | torch.float16 | (1, 1024, 32, 128, 128, 4) | H100 | 30.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 64, 128, 128, 4) | H100 | 48.0 |
| fused_attn_fwd | torch.float16 | (1, 1024, 64, 128, 128, 4) | H100 | 48.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 64, 128, 128, 1) | H100 | 47.2 |
| fused_attn_fwd | torch.float16 | (1, 1024, 64, 128, 128, 1) | H100 | 47.0 |
| fused_attn_fwd | fp8 | (1, 1024, 32, 128) | H100 | 24.4 |
| fused_attn_bwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 32.7 |
| fused_attn_bwd | torch.float16 | (1, 512, 16, 128) | H100 | 32.3 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 58.8 |
| fused_attn_bwd | torch.float16 | (1, 1024, 16, 128) | H100 | 58.6 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 164.5 |
| fused_attn_bwd | torch.float16 | (1, 2048, 16, 128) | H100 | 164.6 |
| fused_attn_bwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 267.9 |
| fused_attn_bwd | torch.float16 | (2, 2048, 16, 128) | H100 | 270.5 |
| fused_attn_bwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 470.7 |
| fused_attn_bwd | torch.float16 | (4, 2048, 16, 128) | H100 | 477.0 |
| fused_attn_bwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 899.1 |
| fused_attn_bwd | torch.float16 | (8, 2048, 16, 128) | H100 | 908.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 451.2 |
| fused_attn_bwd | torch.float16 | (1, 4096, 16, 128) | H100 | 450.9 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 801.3 |
| fused_attn_bwd | torch.float16 | (2, 4096, 16, 128) | H100 | 826.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 32, 128) | H100 | 105.6 |
| fused_attn_bwd | torch.float16 | (1, 1024, 32, 128) | H100 | 105.9 |
| fused_attn_bwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 318.6 |
| fused_attn_bwd | torch.float16 | (4, 1024, 32, 128) | H100 | 318.2 |
| fused_attn_bwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 591.9 |
| fused_attn_bwd | torch.float16 | (8, 1024, 32, 128) | H100 | 594.4 |
| fused_attn_bwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 5536.6 |
| fused_attn_bwd | torch.float16 | (4, 8192, 16, 128) | H100 | 5820.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 2161.7 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 2212.5 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 96, 192, 128) | H100 | 364.0 |
| fused_attn_bwd | torch.float16 | (1, 1024, 96, 192, 128) | H100 | 355.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 96, 192, 128) | H100 | 945.6 |
| fused_attn_bwd | torch.float16 | (1, 2048, 96, 192, 128) | H100 | 957.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 96, 192, 128) | H100 | 3177.2 |
| fused_attn_bwd | torch.float16 | (1, 4096, 96, 192, 128) | H100 | 3291.2 |
| fused_attn_bwd | torch.bfloat16 | (1, 8192, 96, 192, 128) | H100 | 11520.7 |
| fused_attn_bwd | torch.float16 | (1, 8192, 96, 192, 128) | H100 | 12205.4 |
| fused_attn_bwd | torch.bfloat16 | (1, 16384, 96, 192, 128) | H100 | 45043.4 |
| fused_attn_bwd | torch.float16 | (1, 16384, 96, 192, 128) | H100 | 46287.0 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 96, 192, 128) | H100 | 6482.9 |
| fused_attn_bwd | torch.float16 | (2, 4096, 96, 192, 128) | H100 | 6544.3 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 1562.0 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 1566.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 784.4 |
| fused_attn_bwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 825.1 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 40, 128, 128, 8) | H100 | 1001.3 |
| fused_attn_bwd | torch.float16 | (1, 4096, 40, 128, 128, 8) | H100 | 1027.4 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 32, 128, 128, 4) | H100 | 849.4 |
| fused_attn_bwd | torch.float16 | (1, 4096, 32, 128, 128, 4) | H100 | 871.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 128, 128, 4) | H100 | 1583.3 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 128, 128, 4) | H100 | 1590.5 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 128, 128, 1) | H100 | 1606.2 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 128, 128, 1) | H100 | 1644.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 40, 128, 128, 8) | H100 | 133.4 |
| fused_attn_bwd | torch.float16 | (1, 1024, 40, 128, 128, 8) | H100 | 133.1 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 32, 128, 128, 4) | H100 | 112.0 |
| fused_attn_bwd | torch.float16 | (1, 1024, 32, 128, 128, 4) | H100 | 112.6 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 64, 128, 128, 4) | H100 | 193.4 |
| fused_attn_bwd | torch.float16 | (1, 1024, 64, 128, 128, 4) | H100 | 194.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 64, 128, 128, 1) | H100 | 214.4 |
| fused_attn_bwd | torch.float16 | (1, 1024, 64, 128, 128, 1) | H100 | 215.4 |
| fused_attn_bwd | fp8 | (1, 1024, 32, 128) | H100 | 75.1 |
| fused_attn_fwd_infer | torch.bfloat16 | (1, 4096, 128, 512, 512, 1) | H100 | 23221.7 |
| fused_attn_fwd_infer | torch.float16 | (1, 4096, 128, 512, 512, 1) | H100 | 23532.5 |
| fused_attn_fwd_infer | torch.bfloat16 | (1, 256, 2, 512) | H100 | 19.3 |
| fused_attn_fwd_infer | torch.float16 | (1, 256, 2, 512) | H100 | 18.7 |
| fused_attn_fwd_infer | torch.bfloat16 | (1, 512, 4, 512) | H100 | 35.0 |
| fused_attn_fwd_infer | torch.float16 | (1, 512, 4, 512) | H100 | 35.1 |
| fused_attn_fwd_infer | torch.bfloat16 | (1, 1024, 2, 512) | H100 | 68.4 |
| fused_attn_fwd_infer | torch.float16 | (1, 1024, 2, 512) | H100 | 67.8 |
| attnres_fwd_bwd | torch.bfloat16 | (1, 1024, 2, 7168) | H100 | 109.3 |
| attnres_fwd_bwd | torch.bfloat16 | (1, 1024, 9, 7168) | H100 | 256.6 |
| attnres_fwd_bwd | torch.bfloat16 | (1, 2048, 2, 7168) | H100 | 207.6 |
| attnres_fwd_bwd | torch.bfloat16 | (1, 2048, 9, 7168) | H100 | 509.7 |
| attnres_fwd_bwd | torch.bfloat16 | (1, 4096, 2, 7168) | H100 | 395.0 |
| attnres_fwd_bwd | torch.bfloat16 | (1, 4096, 9, 7168) | H100 | 1007.4 |

## Kimi-K3 MLA 专项（96 头 / qk=192 / v=128，TE fused_attn fwd + bwd）

> 完整数据、正确性校验与口径见 [`results_mla_kimi_k3.md`](results_mla_kimi_k3.md)，脚本
> [`bench_mla_kimi_k3.py`](bench_mla_kimi_k3.py)，原始 CSV / stdout 为 `mla_kimi_k3.csv` /
> `mla_kimi_k3.log`。计时与主表同一套 CUPTI device time（warmup=10、repeat=50、b=1、
> causal / no_bias / dropout=0、训练态 `is_training=True`、反向独立计时）。

Kimi-K3 的 MLA 与 K2.6 同几何、头数 64 → 96，并改为 **NoPE**（`rotary_emb=None`，64 维 rope
槽位保留但不旋转）：训练时 `kv_b_proj` 上投影成每头 `k(128) + v(128)`，进入 fused-attn 的
shape 是 `(b, s, 96, 192, 128)`。TE 走 `F16_arbitrary_seqlen`，**训练 fwd + bwd 均支持**
（96 头本身不构成限制，限制仍是 bwd 的 `head_dim<=256`）；与 torch fp32 参考对比，
fwd / dq / dk / dv 的 cos 均 ≈ **0.999997**。FP8 训练在 `192/128` 上返回 `No_Backend`
（FP8 推理 fwd 可用）；TE 原生支持 `qk!=v`，**不需要像 HF FA2 那样把 v 从 128 padding 到 192**。

| s | fwd bf16(µs) | fwd fp16(µs) | bwd bf16(µs) | bwd fp16(µs) | bwd/fwd | fwd TFLOPS(S²) | bwd TFLOPS(S²) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 86.5 | 87.0 | 364.0 | 355.8 | 4.21× | 745 | 354 |
| 2048 | 252.4 | 256.4 | 945.6 | 957.7 | 3.75× | 1021 | 545 |
| 4096 | 878.0 | 896.7 | 3177.2 | 3291.2 | 3.62× | 1174 | 649 |
| 8192 | 3490.0 | 3713.9 | 11520.7 | 12205.4 | 3.30× | 1181 | 716 |
| 16384 | 14181.2 | 14560.6 | 45043.4 | 46287.0 | 3.18× | 1163 | 732 |
| 2×4096 | 1862.7 | 1895.5 | 6482.9 | 6544.3 | 3.48× | 1107 | 636 |

注：

- **bwd 独立计时**：只计 `fused_attn_bwd`（吃 fwd 的 `aux_ctx`，kernel 内重算 S），不含前向；
  「单层训练一步」≈ fwd + bwd。
- **TFLOPS 是完整 S² 口径**：causal kernel 实际只算约一半 tile，真实 tensor-core 利用率 ≈ 表值 ×0.5。
- **折算到 K3 的 24 个 MLA 层**（4k / bf16）：fwd 0.878 ms → **21.07 ms**，bwd 3.177 ms →
  **76.25 ms**，合计 **97.32 ms/step**；仅核心注意力，不含 q_a/q_b、kv_a/kv_b、o_proj、
  输出 gate 等投影 GEMM，也不含 69 个 KDA 层。
- **与 K2.6 对照**（4k / bf16）：64 头 fwd 599.3 µs / bwd 2161.7 µs → 96 头为 **1.46× / 1.47×**
  （头数 1.5×），单位 FLOP 吞吐持平（±2%）。
- **关键结论**：fwd 在 s≥2k 饱和（~1180 S²-TFLOPS，真实 ~59% TC）；bwd 只有 fwd 等效吞吐的
  48–63%（4k 真实 ~33%），是训练侧主要瓶颈；s=1024 时 bwd/fwd 达 4.21×（固定开销占比高）。

## Kimi-K3 AttnRes 专项（FLA `fused_attnres`，D=7168，bf16）

> 数据 `attnres_kimi_k3.csv` / `attnres_kimi_k3.log`（sweep）、`attnres_schedule.log`（逐 L 聚合）、
> `attnres_sweep_{1024,2048,4096}.csv`（seq 专项重扫），脚本
> [`bench_attnres_kimi_k3.py`](bench_attnres_kimi_k3.py)；口径同为 CUPTI device time
> （warmup=5、repeat=20），反向独立计时，融合输出 RMSNorm、`checkpoint_level=1`，FLA 为 0.6.0 的
> checkout。**本节只列实测数字；roofline、带宽利用率与后端取舍的分析见
> [`results_attnres_kimi_k3.md`](results_attnres_kimi_k3.md)。**

Kimi-K3 用 AttnRes 替换 PreNorm 的 `h = h + sublayer(h)`：每个 sublayer 的输入是 L 个 depth
source（已完成 block 的残差和 + 当前 block 的 partial sum）的 softmax 加权和。
`attn_res_block_size=12`（层）⇒ 93 层共 **186 次**调用/迭代（L=2…8 各 24 次、L=9 共 18 次）。

### 单次调用（T=4096）

| L | fwd (µs) | bwd (µs) | fwd+bwd (µs) | 峰值显存 (MB) | 次数/迭代 |
|--:|--:|--:|--:|--:|--:|
| 2 | 74 | 318 | 392 | 392 | 24 |
| 3 | 118 | 417 | 535 | 448 | 24 |
| 4 | 119 | 438 | 557 | — | 24 |
| 5 | 158 | 537 | 695 | 560 | 24 |
| 6 | 160 | 563 | 723 | — | 24 |
| 7 | 199 | 656 | 855 | — | 24 |
| 8 | 200 | 692 | 892 | 728 | 24 |
| 9 | 237 | 770 | 1007 | 784 | 18 |

`fwd+bwd` 与单独测的 `step` 相差 <2.5%，可互换使用；L=2/3/5/8/9 取自 sweep，L=4/6/7 取自
schedule 那一轮（同 shape 相差 ≤1.5 µs）；`—` 表示该 L 未单独测显存。

### 序列长度扩展（L=9）

| T | fwd (µs) | bwd (µs) | fwd+bwd (µs) | 峰值显存 (MB) |
|--:|--:|--:|--:|--:|
| 1024 | 57 | 200 | 257 | 196 |
| 2048 | 121 | 389 | 510 | 392 |
| 4096 | 237 | 770 | 1007 | 784 |
| 8192 | 451 | 1519 | 1970 | 1569 |
| 16384 | 879 | 3021 | 3900 | 3137 |

### 后端对比（T=4096，µs）

| L | Triton fwd/bwd | Gluon fwd/bwd | Gluon 峰值显存 | naive fwd/bwd (ms) |
|--:|--:|--:|--:|--:|
| 2 | 74 / 318 | 63 / 179 | 175 MB | 1.15 / 2.45 |
| 3 | 118 / 417 | 95 / 246 | 231 MB | 1.43 / 3.04 |
| 5 | 158 / 537 | 134 / 324 | 343 MB | 2.04 / 4.24 |
| 8 | 200 / 692 | 190 / 575 | 511 MB | 2.95 / 6.04 |
| 9 | 237 / 770 | 230 / 2844 | 567 MB | 3.32 / 7.51 |

naive 是 FLA 自带的 PyTorch 参考实现，按 **wall** 计时（几十个 eager kernel，CUPTI 求和不稳定），
fused 用 CUPTI。

### 融合与显存开关（T=4096，L=9）

| 配置 | fwd (µs) | bwd (µs) | 峰值显存 (MB) |
|---|--:|--:|--:|
| 基线（融合输出 RMSNorm，ckpt=1） | 237 | 770 | 784 |
| 不融合输出 norm（bf16 权重，额外一次 `rms_norm`） | 282 | — | — |
| 不融合、fp32 norm 权重（torch 退化路径） | 711 | — | — |
| 融合，ckpt=0（存 `o_pre`） | 237 | 767 | 840 |

注：

- **L=8→9**：fwd 201→234、bwd 694→773；跨到 `L2=16` 的指针表桶（桶本身的影响在噪声内）。
- **Gluon 在 L≤8 更快**（反向快 17–44%、显存低 30–55%），但 **L=9 反向 2844 µs**（T=16384 时
  11.1 ms）；K3 的 18 次 L=9 全落在这一档，生产保留 Triton 路径。
- **K3 每迭代（186 次串行）**：T=4096 为 fwd 28.8 + bwd 101.0 = **129.8 ms**；
  T=1024 / 8192 / 16384 分别为 33.6 / 254.7 / 502.1 ms。
- 相对 naive（L=9）：fwd **9.0×**、bwd **7.7×**、峰值显存 **5.2×**（0.78 vs 4.09 GB）。
