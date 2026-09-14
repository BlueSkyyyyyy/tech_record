# TE 算子性能测试结果汇总

> `kernel耗时` 为 CUDA event 计时中位数（µs），已包含单次 launch 开销；`launch时间` 为实测的固定内核启动开销（~11.8 µs/call，每个 case 相同）。设备均为 NVIDIA H100。完整数据见 `te_perf.csv`。

| op name | dtype | shape | device | kernel耗时(µs) | launch时间(µs) |
|---|---|---|---|---|---|
| rmsnorm_fwd | torch.float32 | (128, 512) | H100 | 18.0 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (128, 512) | H100 | 17.9 | 11.8 |
| rmsnorm_fwd | torch.float16 | (128, 512) | H100 | 17.8 | 11.8 |
| rmsnorm_fwd | torch.float32 | (256, 512) | H100 | 17.6 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (256, 512) | H100 | 17.9 | 11.8 |
| rmsnorm_fwd | torch.float16 | (256, 512) | H100 | 17.6 | 11.8 |
| rmsnorm_fwd | torch.float32 | (64, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (64, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float16 | (64, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float32 | (128, 1024) | H100 | 17.8 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (128, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float16 | (128, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float32 | (512, 512) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (512, 512) | H100 | 17.8 | 11.8 |
| rmsnorm_fwd | torch.float16 | (512, 512) | H100 | 17.8 | 11.8 |
| rmsnorm_fwd | torch.float32 | (1024, 512) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 512) | H100 | 17.8 | 11.8 |
| rmsnorm_fwd | torch.float16 | (1024, 512) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float32 | (1024, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float16 | (1024, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float32 | (1024, 2048) | H100 | 18.4 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 2048) | H100 | 17.8 | 11.8 |
| rmsnorm_fwd | torch.float16 | (1024, 2048) | H100 | 17.9 | 11.8 |
| rmsnorm_fwd | torch.float32 | (2048, 1024) | H100 | 18.2 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 1024) | H100 | 18.0 | 11.8 |
| rmsnorm_fwd | torch.float16 | (2048, 1024) | H100 | 17.7 | 11.8 |
| rmsnorm_fwd | torch.float32 | (2048, 2048) | H100 | 19.2 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 2048) | H100 | 18.2 | 11.8 |
| rmsnorm_fwd | torch.float16 | (2048, 2048) | H100 | 18.2 | 11.8 |
| rmsnorm_fwd | torch.float32 | (4096, 2048) | H100 | 34.9 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 2048) | H100 | 19.3 | 11.8 |
| rmsnorm_fwd | torch.float16 | (4096, 2048) | H100 | 19.1 | 11.8 |
| rmsnorm_fwd | torch.float32 | (4096, 4096) | H100 | 60.3 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 4096) | H100 | 36.7 | 11.8 |
| rmsnorm_fwd | torch.float16 | (4096, 4096) | H100 | 36.7 | 11.8 |
| rmsnorm_fwd | torch.float32 | (8192, 1024) | H100 | 34.6 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 1024) | H100 | 19.1 | 11.8 |
| rmsnorm_fwd | torch.float16 | (8192, 1024) | H100 | 19.1 | 11.8 |
| rmsnorm_fwd | torch.float32 | (8192, 2048) | H100 | 59.8 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 2048) | H100 | 36.5 | 11.8 |
| rmsnorm_fwd | torch.float16 | (8192, 2048) | H100 | 36.3 | 11.8 |
| rmsnorm_fwd | torch.float32 | (8192, 4096) | H100 | 107.0 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 4096) | H100 | 62.3 | 11.8 |
| rmsnorm_fwd | torch.float16 | (8192, 4096) | H100 | 62.2 | 11.8 |
| rmsnorm_fwd | torch.float32 | (16384, 2048) | H100 | 106.6 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 2048) | H100 | 62.5 | 11.8 |
| rmsnorm_fwd | torch.float16 | (16384, 2048) | H100 | 62.2 | 11.8 |
| rmsnorm_fwd | torch.float32 | (32768, 2048) | H100 | 197.9 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (32768, 2048) | H100 | 109.4 | 11.8 |
| rmsnorm_fwd | torch.float16 | (32768, 2048) | H100 | 109.2 | 11.8 |
| rmsnorm_fwd | torch.float32 | (4096, 7168) | H100 | 99.5 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 7168) | H100 | 99.5 | 11.8 |
| rmsnorm_fwd | torch.float16 | (4096, 7168) | H100 | 99.4 | 11.8 |
| rmsnorm_fwd | torch.float32 | (8192, 7168) | H100 | 180.4 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 7168) | H100 | 183.2 | 11.8 |
| rmsnorm_fwd | torch.float16 | (8192, 7168) | H100 | 183.0 | 11.8 |
| rmsnorm_fwd | torch.float32 | (16384, 4096) | H100 | 199.8 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 4096) | H100 | 111.3 | 11.8 |
| rmsnorm_fwd | torch.float16 | (16384, 4096) | H100 | 110.9 | 11.8 |
| rmsnorm_fwd | torch.float32 | (16384, 7168) | H100 | 343.6 | 11.8 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 7168) | H100 | 346.8 | 11.8 |
| rmsnorm_fwd | torch.float16 | (16384, 7168) | H100 | 346.7 | 11.8 |
| rmsnorm_bwd | torch.float32 | (128, 512) | H100 | 20.8 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (128, 512) | H100 | 21.2 | 11.8 |
| rmsnorm_bwd | torch.float16 | (128, 512) | H100 | 21.0 | 11.8 |
| rmsnorm_bwd | torch.float32 | (256, 512) | H100 | 21.0 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (256, 512) | H100 | 21.0 | 11.8 |
| rmsnorm_bwd | torch.float16 | (256, 512) | H100 | 21.2 | 11.8 |
| rmsnorm_bwd | torch.float32 | (64, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (64, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd | torch.float16 | (64, 1024) | H100 | 21.2 | 11.8 |
| rmsnorm_bwd | torch.float32 | (128, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (128, 1024) | H100 | 21.1 | 11.8 |
| rmsnorm_bwd | torch.float16 | (128, 1024) | H100 | 20.9 | 11.8 |
| rmsnorm_bwd | torch.float32 | (512, 512) | H100 | 21.2 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (512, 512) | H100 | 21.2 | 11.8 |
| rmsnorm_bwd | torch.float16 | (512, 512) | H100 | 21.0 | 11.8 |
| rmsnorm_bwd | torch.float32 | (1024, 512) | H100 | 21.1 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 512) | H100 | 21.1 | 11.8 |
| rmsnorm_bwd | torch.float16 | (1024, 512) | H100 | 21.0 | 11.8 |
| rmsnorm_bwd | torch.float32 | (1024, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 1024) | H100 | 21.2 | 11.8 |
| rmsnorm_bwd | torch.float16 | (1024, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd | torch.float32 | (1024, 2048) | H100 | 22.8 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 2048) | H100 | 22.3 | 11.8 |
| rmsnorm_bwd | torch.float16 | (1024, 2048) | H100 | 22.1 | 11.8 |
| rmsnorm_bwd | torch.float32 | (2048, 1024) | H100 | 21.7 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 1024) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd | torch.float16 | (2048, 1024) | H100 | 21.2 | 11.8 |
| rmsnorm_bwd | torch.float32 | (2048, 2048) | H100 | 35.0 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 2048) | H100 | 23.1 | 11.8 |
| rmsnorm_bwd | torch.float16 | (2048, 2048) | H100 | 23.1 | 11.8 |
| rmsnorm_bwd | torch.float32 | (4096, 2048) | H100 | 53.3 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 2048) | H100 | 35.7 | 11.8 |
| rmsnorm_bwd | torch.float16 | (4096, 2048) | H100 | 36.0 | 11.8 |
| rmsnorm_bwd | torch.float32 | (4096, 4096) | H100 | 90.1 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 4096) | H100 | 53.8 | 11.8 |
| rmsnorm_bwd | torch.float16 | (4096, 4096) | H100 | 53.9 | 11.8 |
| rmsnorm_bwd | torch.float32 | (8192, 1024) | H100 | 51.6 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 1024) | H100 | 34.1 | 11.8 |
| rmsnorm_bwd | torch.float16 | (8192, 1024) | H100 | 34.0 | 11.8 |
| rmsnorm_bwd | torch.float32 | (8192, 2048) | H100 | 89.8 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 2048) | H100 | 53.1 | 11.8 |
| rmsnorm_bwd | torch.float16 | (8192, 2048) | H100 | 53.1 | 11.8 |
| rmsnorm_bwd | torch.float32 | (8192, 4096) | H100 | 157.9 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 4096) | H100 | 86.7 | 11.8 |
| rmsnorm_bwd | torch.float16 | (8192, 4096) | H100 | 86.7 | 11.8 |
| rmsnorm_bwd | torch.float32 | (16384, 2048) | H100 | 158.4 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 2048) | H100 | 86.1 | 11.8 |
| rmsnorm_bwd | torch.float16 | (16384, 2048) | H100 | 86.2 | 11.8 |
| rmsnorm_bwd | torch.float32 | (32768, 2048) | H100 | 294.0 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (32768, 2048) | H100 | 157.7 | 11.8 |
| rmsnorm_bwd | torch.float16 | (32768, 2048) | H100 | 158.1 | 11.8 |
| rmsnorm_bwd | torch.float32 | (4096, 7168) | H100 | 204.0 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 7168) | H100 | 145.7 | 11.8 |
| rmsnorm_bwd | torch.float16 | (4096, 7168) | H100 | 145.5 | 11.8 |
| rmsnorm_bwd | torch.float32 | (8192, 7168) | H100 | 372.3 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 7168) | H100 | 251.9 | 11.8 |
| rmsnorm_bwd | torch.float16 | (8192, 7168) | H100 | 251.5 | 11.8 |
| rmsnorm_bwd | torch.float32 | (16384, 4096) | H100 | 292.7 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 4096) | H100 | 159.4 | 11.8 |
| rmsnorm_bwd | torch.float16 | (16384, 4096) | H100 | 159.6 | 11.8 |
| rmsnorm_bwd | torch.float32 | (16384, 7168) | H100 | 703.6 | 11.8 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 7168) | H100 | 467.7 | 11.8 |
| rmsnorm_bwd | torch.float16 | (16384, 7168) | H100 | 466.5 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (128, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (128, 512) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (256, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (256, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (256, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (64, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (64, 1024) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (64, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (128, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 1024) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (128, 1024) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (512, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (512, 512) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (512, 512) | H100 | 21.4 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (1024, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 512) | H100 | 21.5 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (1024, 512) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (1024, 1024) | H100 | 21.8 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 1024) | H100 | 21.3 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (1024, 1024) | H100 | 21.5 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (1024, 2048) | H100 | 24.5 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 2048) | H100 | 22.5 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (1024, 2048) | H100 | 22.7 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (2048, 1024) | H100 | 21.9 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 1024) | H100 | 21.9 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (2048, 1024) | H100 | 21.6 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (2048, 2048) | H100 | 41.4 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 2048) | H100 | 25.3 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (2048, 2048) | H100 | 25.7 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (4096, 2048) | H100 | 63.3 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 2048) | H100 | 42.2 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (4096, 2048) | H100 | 41.3 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (4096, 4096) | H100 | 106.6 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 4096) | H100 | 64.7 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (4096, 4096) | H100 | 65.0 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (8192, 1024) | H100 | 61.5 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 1024) | H100 | 40.5 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (8192, 1024) | H100 | 39.9 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (8192, 2048) | H100 | 107.4 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 2048) | H100 | 63.6 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (8192, 2048) | H100 | 63.8 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (8192, 4096) | H100 | 191.8 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 4096) | H100 | 108.2 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (8192, 4096) | H100 | 108.1 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (16384, 2048) | H100 | 193.2 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 2048) | H100 | 106.7 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (16384, 2048) | H100 | 107.3 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (32768, 2048) | H100 | 366.6 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (32768, 2048) | H100 | 194.2 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (32768, 2048) | H100 | 193.8 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (4096, 7168) | H100 | 351.1 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 7168) | H100 | 182.5 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (4096, 7168) | H100 | 182.5 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (8192, 7168) | H100 | 672.6 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 7168) | H100 | 328.8 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (8192, 7168) | H100 | 326.3 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (16384, 4096) | H100 | 365.6 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 4096) | H100 | 194.4 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (16384, 4096) | H100 | 195.2 | 11.8 |
| rmsnorm_bwd_add | torch.float32 | (16384, 7168) | H100 | 1307.4 | 11.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 7168) | H100 | 619.5 | 11.8 |
| rmsnorm_bwd_add | torch.float16 | (16384, 7168) | H100 | 616.4 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 42.9 | 11.8 |
| fused_attn_fwd | torch.float16 | (1, 512, 16, 128) | H100 | 42.9 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 48.3 | 11.8 |
| fused_attn_fwd | torch.float16 | (1, 1024, 16, 128) | H100 | 48.8 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 80.7 | 11.8 |
| fused_attn_fwd | torch.float16 | (1, 2048, 16, 128) | H100 | 81.2 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 112.3 | 11.8 |
| fused_attn_fwd | torch.float16 | (2, 2048, 16, 128) | H100 | 113.7 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 172.3 | 11.8 |
| fused_attn_fwd | torch.float16 | (4, 2048, 16, 128) | H100 | 174.7 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 290.1 | 11.8 |
| fused_attn_fwd | torch.float16 | (8, 2048, 16, 128) | H100 | 296.9 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 182.7 | 11.8 |
| fused_attn_fwd | torch.float16 | (1, 4096, 16, 128) | H100 | 181.4 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 287.5 | 11.8 |
| fused_attn_fwd | torch.float16 | (2, 4096, 16, 128) | H100 | 292.7 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 115.5 | 11.8 |
| fused_attn_fwd | torch.float16 | (4, 1024, 32, 128) | H100 | 116.7 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 186.4 | 11.8 |
| fused_attn_fwd | torch.float16 | (8, 1024, 32, 128) | H100 | 188.7 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 1931.7 | 11.8 |
| fused_attn_fwd | torch.float16 | (4, 8192, 16, 128) | H100 | 2014.1 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 66.4 | 11.8 |
| fused_attn_bwd | torch.float16 | (1, 512, 16, 128) | H100 | 65.6 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 91.8 | 11.8 |
| fused_attn_bwd | torch.float16 | (1, 1024, 16, 128) | H100 | 91.9 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 198.9 | 11.8 |
| fused_attn_bwd | torch.float16 | (1, 2048, 16, 128) | H100 | 200.0 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 304.5 | 11.8 |
| fused_attn_bwd | torch.float16 | (2, 2048, 16, 128) | H100 | 305.5 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 506.3 | 11.8 |
| fused_attn_bwd | torch.float16 | (4, 2048, 16, 128) | H100 | 509.9 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 911.3 | 11.8 |
| fused_attn_bwd | torch.float16 | (8, 2048, 16, 128) | H100 | 930.1 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 484.6 | 11.8 |
| fused_attn_bwd | torch.float16 | (1, 4096, 16, 128) | H100 | 483.7 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 843.1 | 11.8 |
| fused_attn_bwd | torch.float16 | (2, 4096, 16, 128) | H100 | 833.5 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 358.2 | 11.8 |
| fused_attn_bwd | torch.float16 | (4, 1024, 32, 128) | H100 | 354.2 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 629.1 | 11.8 |
| fused_attn_bwd | torch.float16 | (8, 1024, 32, 128) | H100 | 629.5 | 11.8 |
| fused_attn_bwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 5544.3 | 11.8 |
| fused_attn_bwd | torch.float16 | (4, 8192, 16, 128) | H100 | 5886.5 | 11.8 |
