# TE 算子性能测试结果汇总

> `kernel耗时` 为 **CUPTI 实测的纯 device kernel 执行时间**（µs）：由 `torch.profiler` 统计一次调用所 launch 的全部 kernel（含 device memset）的 device 时长之和，对 repeat 次调用取平均，**不含 launch / 主机派发开销**。设备均为 NVIDIA H100。原始数据见 `te_perf.csv` / `perf.log`（含纯 device 时间与由此计算的 GB/s、TFLOPS）。
>
> 注意：小 shape 反复调用时数据可能常驻 L2（H100 约 50MB），此时按 HBM 峰值折算的 `%BW` 会偏高、甚至超过 100%，属正常现象，不代表超过 HBM 带宽；判断 HBM 效率请看工作集大于 L2 的大 shape。

> **生产模型融合注意力形状**（batch=1, seq=4096）：
> - `(1, 4096, 64, 192, 128)` — Kimi-K2.6 MLA（qk=nope128+rope64=192，v=128，64 头）
> - `(1, 4096, 64, 128, 128)` — dsv4 DSA indexer（64 头，head_dim=128）
> - `(1, 4096, 32, 128, 128)` — dsv4.1 DSA indexer（32 头，head_dim=128）
>
> 注：dsv4 / dsv4.1 的主 MLA 注意力 `head_dim=512`（qk=448+64，v=512）超出 TE fused_attn 在 H100 上的支持范围（最大 256），实际由自研 CSA/DSA 稀疏 kernel 承担，故无法用 TE fused_attn 测试，这里只测其 DSA indexer。

| op name | dtype | shape | device | kernel耗时(µs) |
|---|---|---|---|---|
| rmsnorm_fwd | torch.float32 | (128, 512) | H100 | 2.0 |
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
| rmsnorm_fwd | torch.float32 | (1024, 512) | H100 | 2.4 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float16 | (1024, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float32 | (1024, 1024) | H100 | 3.3 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 1024) | H100 | 2.6 |
| rmsnorm_fwd | torch.float16 | (1024, 1024) | H100 | 2.5 |
| rmsnorm_fwd | torch.float32 | (1024, 2048) | H100 | 5.3 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 2048) | H100 | 3.8 |
| rmsnorm_fwd | torch.float16 | (1024, 2048) | H100 | 3.7 |
| rmsnorm_fwd | torch.float32 | (2048, 1024) | H100 | 4.8 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 1024) | H100 | 3.5 |
| rmsnorm_fwd | torch.float16 | (2048, 1024) | H100 | 3.5 |
| rmsnorm_fwd | torch.float32 | (2048, 2048) | H100 | 8.1 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 2048) | H100 | 5.0 |
| rmsnorm_fwd | torch.float16 | (2048, 2048) | H100 | 4.9 |
| rmsnorm_fwd | torch.float32 | (4096, 2048) | H100 | 24.3 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 2048) | H100 | 7.8 |
| rmsnorm_fwd | torch.float16 | (4096, 2048) | H100 | 7.7 |
| rmsnorm_fwd | torch.float32 | (4096, 4096) | H100 | 49.9 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 4096) | H100 | 26.1 |
| rmsnorm_fwd | torch.float16 | (4096, 4096) | H100 | 26.0 |
| rmsnorm_fwd | torch.float32 | (8192, 1024) | H100 | 24.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 1024) | H100 | 7.9 |
| rmsnorm_fwd | torch.float16 | (8192, 1024) | H100 | 7.6 |
| rmsnorm_fwd | torch.float32 | (8192, 2048) | H100 | 49.5 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 2048) | H100 | 25.6 |
| rmsnorm_fwd | torch.float16 | (8192, 2048) | H100 | 25.5 |
| rmsnorm_fwd | torch.float32 | (8192, 4096) | H100 | 96.6 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 4096) | H100 | 51.7 |
| rmsnorm_fwd | torch.float16 | (8192, 4096) | H100 | 51.5 |
| rmsnorm_fwd | torch.float32 | (16384, 2048) | H100 | 95.7 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 2048) | H100 | 51.6 |
| rmsnorm_fwd | torch.float16 | (16384, 2048) | H100 | 51.4 |
| rmsnorm_fwd | torch.float32 | (32768, 2048) | H100 | 187.1 |
| rmsnorm_fwd | torch.bfloat16 | (32768, 2048) | H100 | 99.1 |
| rmsnorm_fwd | torch.float16 | (32768, 2048) | H100 | 98.2 |
| rmsnorm_fwd | torch.float32 | (4096, 7168) | H100 | 89.0 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 7168) | H100 | 88.9 |
| rmsnorm_fwd | torch.float16 | (4096, 7168) | H100 | 88.8 |
| rmsnorm_fwd | torch.float32 | (8192, 7168) | H100 | 169.9 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 7168) | H100 | 172.4 |
| rmsnorm_fwd | torch.float16 | (8192, 7168) | H100 | 172.3 |
| rmsnorm_fwd | torch.float32 | (16384, 4096) | H100 | 189.8 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 4096) | H100 | 101.0 |
| rmsnorm_fwd | torch.float16 | (16384, 4096) | H100 | 100.5 |
| rmsnorm_fwd | torch.float32 | (16384, 7168) | H100 | 333.1 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 7168) | H100 | 336.0 |
| rmsnorm_fwd | torch.float16 | (16384, 7168) | H100 | 336.2 |
| rmsnorm_bwd | torch.float32 | (128, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (128, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd | torch.float32 | (256, 512) | H100 | 5.2 |
| rmsnorm_bwd | torch.bfloat16 | (256, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.float32 | (64, 1024) | H100 | 5.0 |
| rmsnorm_bwd | torch.bfloat16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd | torch.float16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd | torch.float32 | (128, 1024) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float32 | (512, 512) | H100 | 5.3 |
| rmsnorm_bwd | torch.bfloat16 | (512, 512) | H100 | 5.6 |
| rmsnorm_bwd | torch.float16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd | torch.float32 | (1024, 512) | H100 | 5.7 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 512) | H100 | 5.8 |
| rmsnorm_bwd | torch.float16 | (1024, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float32 | (1024, 1024) | H100 | 6.7 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 1024) | H100 | 5.6 |
| rmsnorm_bwd | torch.float16 | (1024, 1024) | H100 | 5.6 |
| rmsnorm_bwd | torch.float32 | (1024, 2048) | H100 | 9.8 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd | torch.float16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd | torch.float32 | (2048, 1024) | H100 | 8.5 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 1024) | H100 | 6.5 |
| rmsnorm_bwd | torch.float16 | (2048, 1024) | H100 | 6.5 |
| rmsnorm_bwd | torch.float32 | (2048, 2048) | H100 | 22.0 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 2048) | H100 | 11.1 |
| rmsnorm_bwd | torch.float16 | (2048, 2048) | H100 | 11.1 |
| rmsnorm_bwd | torch.float32 | (4096, 2048) | H100 | 40.7 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 2048) | H100 | 23.1 |
| rmsnorm_bwd | torch.float16 | (4096, 2048) | H100 | 23.1 |
| rmsnorm_bwd | torch.float32 | (4096, 4096) | H100 | 77.1 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 4096) | H100 | 41.1 |
| rmsnorm_bwd | torch.float16 | (4096, 4096) | H100 | 41.4 |
| rmsnorm_bwd | torch.float32 | (8192, 1024) | H100 | 38.9 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 1024) | H100 | 21.0 |
| rmsnorm_bwd | torch.float16 | (8192, 1024) | H100 | 21.1 |
| rmsnorm_bwd | torch.float32 | (8192, 2048) | H100 | 77.1 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 2048) | H100 | 40.5 |
| rmsnorm_bwd | torch.float16 | (8192, 2048) | H100 | 40.5 |
| rmsnorm_bwd | torch.float32 | (8192, 4096) | H100 | 145.3 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 4096) | H100 | 73.8 |
| rmsnorm_bwd | torch.float16 | (8192, 4096) | H100 | 74.0 |
| rmsnorm_bwd | torch.float32 | (16384, 2048) | H100 | 145.3 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 2048) | H100 | 73.3 |
| rmsnorm_bwd | torch.float16 | (16384, 2048) | H100 | 73.3 |
| rmsnorm_bwd | torch.float32 | (32768, 2048) | H100 | 281.2 |
| rmsnorm_bwd | torch.bfloat16 | (32768, 2048) | H100 | 144.7 |
| rmsnorm_bwd | torch.float16 | (32768, 2048) | H100 | 145.1 |
| rmsnorm_bwd | torch.float32 | (4096, 7168) | H100 | 189.4 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 7168) | H100 | 132.3 |
| rmsnorm_bwd | torch.float16 | (4096, 7168) | H100 | 131.7 |
| rmsnorm_bwd | torch.float32 | (8192, 7168) | H100 | 359.6 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 7168) | H100 | 239.5 |
| rmsnorm_bwd | torch.float16 | (8192, 7168) | H100 | 238.7 |
| rmsnorm_bwd | torch.float32 | (16384, 4096) | H100 | 280.3 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 4096) | H100 | 146.8 |
| rmsnorm_bwd | torch.float16 | (16384, 4096) | H100 | 146.9 |
| rmsnorm_bwd | torch.float32 | (16384, 7168) | H100 | 691.9 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 7168) | H100 | 454.7 |
| rmsnorm_bwd | torch.float16 | (16384, 7168) | H100 | 452.7 |
| rmsnorm_bwd_add | torch.float32 | (128, 512) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (256, 512) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float32 | (64, 1024) | H100 | 4.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (64, 1024) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.float16 | (64, 1024) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.float32 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (512, 512) | H100 | 5.3 |
| rmsnorm_bwd_add | torch.float16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.float32 | (1024, 512) | H100 | 5.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float32 | (1024, 1024) | H100 | 6.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 1024) | H100 | 5.8 |
| rmsnorm_bwd_add | torch.float16 | (1024, 1024) | H100 | 6.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 2048) | H100 | 11.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 2048) | H100 | 8.9 |
| rmsnorm_bwd_add | torch.float16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd_add | torch.float32 | (2048, 1024) | H100 | 8.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 1024) | H100 | 6.8 |
| rmsnorm_bwd_add | torch.float16 | (2048, 1024) | H100 | 6.9 |
| rmsnorm_bwd_add | torch.float32 | (2048, 2048) | H100 | 28.3 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 2048) | H100 | 12.8 |
| rmsnorm_bwd_add | torch.float16 | (2048, 2048) | H100 | 12.7 |
| rmsnorm_bwd_add | torch.float32 | (4096, 2048) | H100 | 50.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 2048) | H100 | 28.8 |
| rmsnorm_bwd_add | torch.float16 | (4096, 2048) | H100 | 28.9 |
| rmsnorm_bwd_add | torch.float32 | (4096, 4096) | H100 | 93.5 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 4096) | H100 | 52.0 |
| rmsnorm_bwd_add | torch.float16 | (4096, 4096) | H100 | 52.0 |
| rmsnorm_bwd_add | torch.float32 | (8192, 1024) | H100 | 48.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 1024) | H100 | 27.1 |
| rmsnorm_bwd_add | torch.float16 | (8192, 1024) | H100 | 27.3 |
| rmsnorm_bwd_add | torch.float32 | (8192, 2048) | H100 | 94.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 2048) | H100 | 51.1 |
| rmsnorm_bwd_add | torch.float16 | (8192, 2048) | H100 | 50.8 |
| rmsnorm_bwd_add | torch.float32 | (8192, 4096) | H100 | 179.3 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 4096) | H100 | 95.3 |
| rmsnorm_bwd_add | torch.float16 | (8192, 4096) | H100 | 95.5 |
| rmsnorm_bwd_add | torch.float32 | (16384, 2048) | H100 | 180.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 2048) | H100 | 94.6 |
| rmsnorm_bwd_add | torch.float16 | (16384, 2048) | H100 | 94.3 |
| rmsnorm_bwd_add | torch.float32 | (32768, 2048) | H100 | 354.5 |
| rmsnorm_bwd_add | torch.bfloat16 | (32768, 2048) | H100 | 181.2 |
| rmsnorm_bwd_add | torch.float16 | (32768, 2048) | H100 | 181.4 |
| rmsnorm_bwd_add | torch.float32 | (4096, 7168) | H100 | 337.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 7168) | H100 | 168.7 |
| rmsnorm_bwd_add | torch.float16 | (4096, 7168) | H100 | 168.5 |
| rmsnorm_bwd_add | torch.float32 | (8192, 7168) | H100 | 657.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 7168) | H100 | 315.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 7168) | H100 | 315.3 |
| rmsnorm_bwd_add | torch.float32 | (16384, 4096) | H100 | 352.1 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 4096) | H100 | 181.3 |
| rmsnorm_bwd_add | torch.float16 | (16384, 4096) | H100 | 182.0 |
| rmsnorm_bwd_add | torch.float32 | (16384, 7168) | H100 | 1295.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 7168) | H100 | 604.2 |
| rmsnorm_bwd_add | torch.float16 | (16384, 7168) | H100 | 603.3 |
| fused_attn_fwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 11.9 |
| fused_attn_fwd | torch.float16 | (1, 512, 16, 128) | H100 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 18.7 |
| fused_attn_fwd | torch.float16 | (1, 1024, 16, 128) | H100 | 18.8 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 51.5 |
| fused_attn_fwd | torch.float16 | (1, 2048, 16, 128) | H100 | 51.6 |
| fused_attn_fwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 81.5 |
| fused_attn_fwd | torch.float16 | (2, 2048, 16, 128) | H100 | 82.4 |
| fused_attn_fwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 141.7 |
| fused_attn_fwd | torch.float16 | (4, 2048, 16, 128) | H100 | 143.4 |
| fused_attn_fwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 257.6 |
| fused_attn_fwd | torch.float16 | (8, 2048, 16, 128) | H100 | 264.3 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 148.2 |
| fused_attn_fwd | torch.float16 | (1, 4096, 16, 128) | H100 | 150.3 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 255.2 |
| fused_attn_fwd | torch.float16 | (2, 4096, 16, 128) | H100 | 259.4 |
| fused_attn_fwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 83.8 |
| fused_attn_fwd | torch.float16 | (4, 1024, 32, 128) | H100 | 85.1 |
| fused_attn_fwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 151.6 |
| fused_attn_fwd | torch.float16 | (8, 1024, 32, 128) | H100 | 154.3 |
| fused_attn_fwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 1881.9 |
| fused_attn_fwd | torch.float16 | (4, 8192, 16, 128) | H100 | 1960.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 618.8 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 642.3 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 472.4 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 488.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 256.8 |
| fused_attn_fwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 259.4 |
| fused_attn_bwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 32.7 |
| fused_attn_bwd | torch.float16 | (1, 512, 16, 128) | H100 | 32.4 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 58.5 |
| fused_attn_bwd | torch.float16 | (1, 1024, 16, 128) | H100 | 58.6 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 164.0 |
| fused_attn_bwd | torch.float16 | (1, 2048, 16, 128) | H100 | 163.1 |
| fused_attn_bwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 269.1 |
| fused_attn_bwd | torch.float16 | (2, 2048, 16, 128) | H100 | 271.6 |
| fused_attn_bwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 470.6 |
| fused_attn_bwd | torch.float16 | (4, 2048, 16, 128) | H100 | 477.2 |
| fused_attn_bwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 890.8 |
| fused_attn_bwd | torch.float16 | (8, 2048, 16, 128) | H100 | 898.1 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 449.4 |
| fused_attn_bwd | torch.float16 | (1, 4096, 16, 128) | H100 | 450.8 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 789.1 |
| fused_attn_bwd | torch.float16 | (2, 4096, 16, 128) | H100 | 815.3 |
| fused_attn_bwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 318.0 |
| fused_attn_bwd | torch.float16 | (4, 1024, 32, 128) | H100 | 319.6 |
| fused_attn_bwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 592.5 |
| fused_attn_bwd | torch.float16 | (8, 1024, 32, 128) | H100 | 597.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 5543.5 |
| fused_attn_bwd | torch.float16 | (4, 8192, 16, 128) | H100 | 5858.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 2145.2 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 2221.9 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 1524.2 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 1570.1 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 786.7 |
| fused_attn_bwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 798.3 |
