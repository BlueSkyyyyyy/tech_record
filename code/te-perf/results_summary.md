# TE 算子性能测试结果汇总

> `kernel耗时` 为 CUDA event 计时中位数（µs），已包含单次 launch 开销；`launch时间` 为实测的固定内核启动开销（~12.0 µs/call，每个 case 相同）。设备均为 NVIDIA H100。完整数据见 `te_perf.csv`。

> **生产模型融合注意力形状**（batch=1, seq=4096）：
> - `(1, 4096, 64, 192, 128)` — Kimi-K2.6 MLA（qk=nope128+rope64=192，v=128，64 头）
> - `(1, 4096, 64, 128, 128)` — dsv4 DSA indexer（64 头，head_dim=128）
> - `(1, 4096, 32, 128, 128)` — dsv4.1 DSA indexer（32 头，head_dim=128）
>
> 注：dsv4 / dsv4.1 的主 MLA 注意力 `head_dim=512`（qk=448+64，v=512）超出 TE fused_attn 在 H100 上的支持范围（最大 256），实际由自研 CSA/DSA 稀疏 kernel 承担，故无法用 TE fused_attn 测试，这里只测其 DSA indexer。

| op name | dtype | shape | device | kernel耗时(µs) | launch时间(µs) |
|---|---|---|---|---|---|
| rmsnorm_fwd | torch.float32 | (128, 512) | H100 | 18.1 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (128, 512) | H100 | 18.1 | 12.0 |
| rmsnorm_fwd | torch.float16 | (128, 512) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float32 | (256, 512) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (256, 512) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.float16 | (256, 512) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float32 | (64, 1024) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (64, 1024) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.float16 | (64, 1024) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.float32 | (128, 1024) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (128, 1024) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float16 | (128, 1024) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float32 | (512, 512) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (512, 512) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float16 | (512, 512) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.float32 | (1024, 512) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 512) | H100 | 18.1 | 12.0 |
| rmsnorm_fwd | torch.float16 | (1024, 512) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.float32 | (1024, 1024) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 1024) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.float16 | (1024, 1024) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float32 | (1024, 2048) | H100 | 18.5 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 2048) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float16 | (1024, 2048) | H100 | 18.1 | 12.0 |
| rmsnorm_fwd | torch.float32 | (2048, 1024) | H100 | 18.4 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 1024) | H100 | 18.1 | 12.0 |
| rmsnorm_fwd | torch.float16 | (2048, 1024) | H100 | 18.0 | 12.0 |
| rmsnorm_fwd | torch.float32 | (2048, 2048) | H100 | 19.4 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 2048) | H100 | 18.4 | 12.0 |
| rmsnorm_fwd | torch.float16 | (2048, 2048) | H100 | 18.5 | 12.0 |
| rmsnorm_fwd | torch.float32 | (4096, 2048) | H100 | 35.1 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 2048) | H100 | 19.5 | 12.0 |
| rmsnorm_fwd | torch.float16 | (4096, 2048) | H100 | 19.4 | 12.0 |
| rmsnorm_fwd | torch.float32 | (4096, 4096) | H100 | 60.6 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 4096) | H100 | 36.9 | 12.0 |
| rmsnorm_fwd | torch.float16 | (4096, 4096) | H100 | 36.8 | 12.0 |
| rmsnorm_fwd | torch.float32 | (8192, 1024) | H100 | 34.8 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 1024) | H100 | 19.3 | 12.0 |
| rmsnorm_fwd | torch.float16 | (8192, 1024) | H100 | 19.4 | 12.0 |
| rmsnorm_fwd | torch.float32 | (8192, 2048) | H100 | 60.2 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 2048) | H100 | 36.5 | 12.0 |
| rmsnorm_fwd | torch.float16 | (8192, 2048) | H100 | 36.3 | 12.0 |
| rmsnorm_fwd | torch.float32 | (8192, 4096) | H100 | 107.0 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 4096) | H100 | 62.4 | 12.0 |
| rmsnorm_fwd | torch.float16 | (8192, 4096) | H100 | 62.4 | 12.0 |
| rmsnorm_fwd | torch.float32 | (16384, 2048) | H100 | 106.7 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 2048) | H100 | 62.2 | 12.0 |
| rmsnorm_fwd | torch.float16 | (16384, 2048) | H100 | 62.2 | 12.0 |
| rmsnorm_fwd | torch.float32 | (32768, 2048) | H100 | 198.1 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (32768, 2048) | H100 | 109.7 | 12.0 |
| rmsnorm_fwd | torch.float16 | (32768, 2048) | H100 | 109.2 | 12.0 |
| rmsnorm_fwd | torch.float32 | (4096, 7168) | H100 | 99.5 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 7168) | H100 | 99.6 | 12.0 |
| rmsnorm_fwd | torch.float16 | (4096, 7168) | H100 | 99.6 | 12.0 |
| rmsnorm_fwd | torch.float32 | (8192, 7168) | H100 | 180.7 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 7168) | H100 | 183.3 | 12.0 |
| rmsnorm_fwd | torch.float16 | (8192, 7168) | H100 | 183.3 | 12.0 |
| rmsnorm_fwd | torch.float32 | (16384, 4096) | H100 | 200.5 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 4096) | H100 | 111.4 | 12.0 |
| rmsnorm_fwd | torch.float16 | (16384, 4096) | H100 | 110.9 | 12.0 |
| rmsnorm_fwd | torch.float32 | (16384, 7168) | H100 | 343.7 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 7168) | H100 | 346.5 | 12.0 |
| rmsnorm_fwd | torch.float16 | (16384, 7168) | H100 | 346.7 | 12.0 |
| rmsnorm_bwd | torch.float32 | (128, 512) | H100 | 21.2 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (128, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd | torch.float16 | (128, 512) | H100 | 21.2 | 12.0 |
| rmsnorm_bwd | torch.float32 | (256, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (256, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.float16 | (256, 512) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd | torch.float32 | (64, 1024) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (64, 1024) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd | torch.float16 | (64, 1024) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd | torch.float32 | (128, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (128, 1024) | H100 | 21.2 | 12.0 |
| rmsnorm_bwd | torch.float16 | (128, 1024) | H100 | 21.2 | 12.0 |
| rmsnorm_bwd | torch.float32 | (512, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (512, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd | torch.float16 | (512, 512) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.float32 | (1024, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.float16 | (1024, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.float32 | (1024, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 1024) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.float16 | (1024, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd | torch.float32 | (1024, 2048) | H100 | 23.0 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 2048) | H100 | 22.7 | 12.0 |
| rmsnorm_bwd | torch.float16 | (1024, 2048) | H100 | 22.4 | 12.0 |
| rmsnorm_bwd | torch.float32 | (2048, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 1024) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.float16 | (2048, 1024) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.float32 | (2048, 2048) | H100 | 35.4 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 2048) | H100 | 23.7 | 12.0 |
| rmsnorm_bwd | torch.float16 | (2048, 2048) | H100 | 23.6 | 12.0 |
| rmsnorm_bwd | torch.float32 | (4096, 2048) | H100 | 53.5 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 2048) | H100 | 36.4 | 12.0 |
| rmsnorm_bwd | torch.float16 | (4096, 2048) | H100 | 36.4 | 12.0 |
| rmsnorm_bwd | torch.float32 | (4096, 4096) | H100 | 90.5 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 4096) | H100 | 54.1 | 12.0 |
| rmsnorm_bwd | torch.float16 | (4096, 4096) | H100 | 54.4 | 12.0 |
| rmsnorm_bwd | torch.float32 | (8192, 1024) | H100 | 51.9 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 1024) | H100 | 34.4 | 12.0 |
| rmsnorm_bwd | torch.float16 | (8192, 1024) | H100 | 34.3 | 12.0 |
| rmsnorm_bwd | torch.float32 | (8192, 2048) | H100 | 90.0 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 2048) | H100 | 53.5 | 12.0 |
| rmsnorm_bwd | torch.float16 | (8192, 2048) | H100 | 53.7 | 12.0 |
| rmsnorm_bwd | torch.float32 | (8192, 4096) | H100 | 158.2 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 4096) | H100 | 86.7 | 12.0 |
| rmsnorm_bwd | torch.float16 | (8192, 4096) | H100 | 87.0 | 12.0 |
| rmsnorm_bwd | torch.float32 | (16384, 2048) | H100 | 158.6 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 2048) | H100 | 86.3 | 12.0 |
| rmsnorm_bwd | torch.float16 | (16384, 2048) | H100 | 86.5 | 12.0 |
| rmsnorm_bwd | torch.float32 | (32768, 2048) | H100 | 294.3 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (32768, 2048) | H100 | 158.2 | 12.0 |
| rmsnorm_bwd | torch.float16 | (32768, 2048) | H100 | 157.9 | 12.0 |
| rmsnorm_bwd | torch.float32 | (4096, 7168) | H100 | 203.7 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 7168) | H100 | 145.8 | 12.0 |
| rmsnorm_bwd | torch.float16 | (4096, 7168) | H100 | 145.5 | 12.0 |
| rmsnorm_bwd | torch.float32 | (8192, 7168) | H100 | 372.4 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 7168) | H100 | 252.4 | 12.0 |
| rmsnorm_bwd | torch.float16 | (8192, 7168) | H100 | 251.9 | 12.0 |
| rmsnorm_bwd | torch.float32 | (16384, 4096) | H100 | 293.2 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 4096) | H100 | 159.7 | 12.0 |
| rmsnorm_bwd | torch.float16 | (16384, 4096) | H100 | 159.8 | 12.0 |
| rmsnorm_bwd | torch.float32 | (16384, 7168) | H100 | 703.3 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 7168) | H100 | 468.4 | 12.0 |
| rmsnorm_bwd | torch.float16 | (16384, 7168) | H100 | 465.7 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (128, 512) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 512) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 512) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (256, 512) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (256, 512) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (256, 512) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (64, 1024) | H100 | 22.0 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (64, 1024) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (64, 1024) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (128, 1024) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 1024) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (512, 512) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (512, 512) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (512, 512) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 512) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 512) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (1024, 512) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 1024) | H100 | 22.1 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 1024) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (1024, 1024) | H100 | 22.0 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 2048) | H100 | 24.8 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 2048) | H100 | 22.8 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (1024, 2048) | H100 | 22.9 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (2048, 1024) | H100 | 22.2 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 1024) | H100 | 22.1 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (2048, 1024) | H100 | 22.1 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (2048, 2048) | H100 | 41.6 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 2048) | H100 | 25.7 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (2048, 2048) | H100 | 25.8 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (4096, 2048) | H100 | 63.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 2048) | H100 | 42.1 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (4096, 2048) | H100 | 41.7 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (4096, 4096) | H100 | 106.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 4096) | H100 | 64.9 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (4096, 4096) | H100 | 65.3 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (8192, 1024) | H100 | 61.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 1024) | H100 | 40.6 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 1024) | H100 | 40.2 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (8192, 2048) | H100 | 107.5 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 2048) | H100 | 64.1 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 2048) | H100 | 64.2 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (8192, 4096) | H100 | 191.9 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 4096) | H100 | 108.3 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 4096) | H100 | 108.6 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (16384, 2048) | H100 | 193.4 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 2048) | H100 | 107.2 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (16384, 2048) | H100 | 107.4 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (32768, 2048) | H100 | 366.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (32768, 2048) | H100 | 194.5 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (32768, 2048) | H100 | 194.1 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (4096, 7168) | H100 | 351.0 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 7168) | H100 | 182.2 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (4096, 7168) | H100 | 182.4 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (8192, 7168) | H100 | 672.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 7168) | H100 | 327.4 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 7168) | H100 | 327.3 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (16384, 4096) | H100 | 365.3 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 4096) | H100 | 194.2 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (16384, 4096) | H100 | 194.8 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (16384, 7168) | H100 | 1309.1 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 7168) | H100 | 619.0 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (16384, 7168) | H100 | 617.7 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 43.4 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 512, 16, 128) | H100 | 43.2 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 48.9 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 1024, 16, 128) | H100 | 48.8 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 81.3 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 2048, 16, 128) | H100 | 81.3 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 113.2 | 12.0 |
| fused_attn_fwd | torch.float16 | (2, 2048, 16, 128) | H100 | 114.1 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 172.3 | 12.0 |
| fused_attn_fwd | torch.float16 | (4, 2048, 16, 128) | H100 | 175.7 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 290.6 | 12.0 |
| fused_attn_fwd | torch.float16 | (8, 2048, 16, 128) | H100 | 297.8 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 180.6 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 4096, 16, 128) | H100 | 181.8 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 289.2 | 12.0 |
| fused_attn_fwd | torch.float16 | (2, 4096, 16, 128) | H100 | 292.2 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 116.0 | 12.0 |
| fused_attn_fwd | torch.float16 | (4, 1024, 32, 128) | H100 | 116.9 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 187.2 | 12.0 |
| fused_attn_fwd | torch.float16 | (8, 1024, 32, 128) | H100 | 189.6 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 1949.9 | 12.0 |
| fused_attn_fwd | torch.float16 | (4, 8192, 16, 128) | H100 | 2010.9 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 650.6 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 637.7 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 523.9 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 533.4 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 297.1 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 290.9 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 65.7 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 512, 16, 128) | H100 | 65.4 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 92.3 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 1024, 16, 128) | H100 | 91.8 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 198.1 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 2048, 16, 128) | H100 | 199.8 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 304.2 | 12.0 |
| fused_attn_bwd | torch.float16 | (2, 2048, 16, 128) | H100 | 305.4 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 506.1 | 12.0 |
| fused_attn_bwd | torch.float16 | (4, 2048, 16, 128) | H100 | 509.3 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 930.0 | 12.0 |
| fused_attn_bwd | torch.float16 | (8, 2048, 16, 128) | H100 | 937.9 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 485.6 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 4096, 16, 128) | H100 | 483.0 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 870.6 | 12.0 |
| fused_attn_bwd | torch.float16 | (2, 4096, 16, 128) | H100 | 857.2 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 352.8 | 12.0 |
| fused_attn_bwd | torch.float16 | (4, 1024, 32, 128) | H100 | 352.4 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 626.5 | 12.0 |
| fused_attn_bwd | torch.float16 | (8, 1024, 32, 128) | H100 | 628.5 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 5544.2 | 12.0 |
| fused_attn_bwd | torch.float16 | (4, 8192, 16, 128) | H100 | 5986.3 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 2192.2 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 2225.1 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 1583.1 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 1632.2 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 825.0 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 837.3 | 12.0 |
