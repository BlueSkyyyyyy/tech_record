# TE 算子性能测试结果汇总

> `kernel耗时` 为 CUDA event 计时中位数（µs），已包含单次 launch 开销；`launch时间` 为实测的固定内核启动开销（~12.0 µs/call，每个 case 相同）。设备均为 NVIDIA H100。完整数据见 `te_perf.csv`。

| op name | dtype | shape | device | kernel耗时(µs) | launch时间(µs) |
|---|---|---|---|---|---|
| rmsnorm_fwd | torch.float32 | (128, 512) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (128, 512) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.float16 | (128, 512) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float32 | (256, 512) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (256, 512) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float16 | (256, 512) | H100 | 17.6 | 12.0 |
| rmsnorm_fwd | torch.float32 | (64, 1024) | H100 | 17.6 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (64, 1024) | H100 | 17.6 | 12.0 |
| rmsnorm_fwd | torch.float16 | (64, 1024) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float32 | (128, 1024) | H100 | 17.6 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (128, 1024) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float16 | (128, 1024) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float32 | (512, 512) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (512, 512) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float16 | (512, 512) | H100 | 17.6 | 12.0 |
| rmsnorm_fwd | torch.float32 | (1024, 512) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 512) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.float16 | (1024, 512) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.float32 | (1024, 1024) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 1024) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float16 | (1024, 1024) | H100 | 17.7 | 12.0 |
| rmsnorm_fwd | torch.float32 | (1024, 2048) | H100 | 18.1 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 2048) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.float16 | (1024, 2048) | H100 | 17.6 | 12.0 |
| rmsnorm_fwd | torch.float32 | (2048, 1024) | H100 | 18.2 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 1024) | H100 | 17.8 | 12.0 |
| rmsnorm_fwd | torch.float16 | (2048, 1024) | H100 | 17.9 | 12.0 |
| rmsnorm_fwd | torch.float32 | (2048, 2048) | H100 | 19.1 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 2048) | H100 | 18.3 | 12.0 |
| rmsnorm_fwd | torch.float16 | (2048, 2048) | H100 | 18.2 | 12.0 |
| rmsnorm_fwd | torch.float32 | (4096, 2048) | H100 | 34.9 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 2048) | H100 | 19.2 | 12.0 |
| rmsnorm_fwd | torch.float16 | (4096, 2048) | H100 | 19.1 | 12.0 |
| rmsnorm_fwd | torch.float32 | (4096, 4096) | H100 | 60.4 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 4096) | H100 | 36.7 | 12.0 |
| rmsnorm_fwd | torch.float16 | (4096, 4096) | H100 | 36.7 | 12.0 |
| rmsnorm_fwd | torch.float32 | (8192, 2048) | H100 | 60.0 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 2048) | H100 | 36.6 | 12.0 |
| rmsnorm_fwd | torch.float16 | (8192, 2048) | H100 | 36.4 | 12.0 |
| rmsnorm_fwd | torch.float32 | (8192, 4096) | H100 | 107.1 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 4096) | H100 | 62.3 | 12.0 |
| rmsnorm_fwd | torch.float16 | (8192, 4096) | H100 | 62.2 | 12.0 |
| rmsnorm_fwd | torch.float32 | (16384, 2048) | H100 | 106.5 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 2048) | H100 | 62.3 | 12.0 |
| rmsnorm_fwd | torch.float16 | (16384, 2048) | H100 | 62.2 | 12.0 |
| rmsnorm_fwd | torch.float32 | (32768, 2048) | H100 | 197.8 | 12.0 |
| rmsnorm_fwd | torch.bfloat16 | (32768, 2048) | H100 | 109.4 | 12.0 |
| rmsnorm_fwd | torch.float16 | (32768, 2048) | H100 | 109.2 | 12.0 |
| rmsnorm_bwd | torch.float32 | (128, 512) | H100 | 21.0 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (128, 512) | H100 | 21.1 | 12.0 |
| rmsnorm_bwd | torch.float16 | (128, 512) | H100 | 21.0 | 12.0 |
| rmsnorm_bwd | torch.float32 | (256, 512) | H100 | 21.0 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (256, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.float16 | (256, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd | torch.float32 | (64, 1024) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (64, 1024) | H100 | 21.1 | 12.0 |
| rmsnorm_bwd | torch.float16 | (64, 1024) | H100 | 21.1 | 12.0 |
| rmsnorm_bwd | torch.float32 | (128, 1024) | H100 | 20.9 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (128, 1024) | H100 | 21.0 | 12.0 |
| rmsnorm_bwd | torch.float16 | (128, 1024) | H100 | 20.9 | 12.0 |
| rmsnorm_bwd | torch.float32 | (512, 512) | H100 | 21.2 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (512, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.float16 | (512, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd | torch.float32 | (1024, 512) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 512) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.float16 | (1024, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd | torch.float32 | (1024, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 1024) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.float16 | (1024, 1024) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd | torch.float32 | (1024, 2048) | H100 | 23.0 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 2048) | H100 | 22.8 | 12.0 |
| rmsnorm_bwd | torch.float16 | (1024, 2048) | H100 | 22.3 | 12.0 |
| rmsnorm_bwd | torch.float32 | (2048, 1024) | H100 | 22.4 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 1024) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd | torch.float16 | (2048, 1024) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd | torch.float32 | (2048, 2048) | H100 | 35.6 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 2048) | H100 | 23.7 | 12.0 |
| rmsnorm_bwd | torch.float16 | (2048, 2048) | H100 | 23.6 | 12.0 |
| rmsnorm_bwd | torch.float32 | (4096, 2048) | H100 | 53.6 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 2048) | H100 | 36.4 | 12.0 |
| rmsnorm_bwd | torch.float16 | (4096, 2048) | H100 | 36.5 | 12.0 |
| rmsnorm_bwd | torch.float32 | (4096, 4096) | H100 | 89.8 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 4096) | H100 | 54.1 | 12.0 |
| rmsnorm_bwd | torch.float16 | (4096, 4096) | H100 | 54.2 | 12.0 |
| rmsnorm_bwd | torch.float32 | (8192, 2048) | H100 | 90.0 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 2048) | H100 | 53.3 | 12.0 |
| rmsnorm_bwd | torch.float16 | (8192, 2048) | H100 | 53.4 | 12.0 |
| rmsnorm_bwd | torch.float32 | (8192, 4096) | H100 | 158.0 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 4096) | H100 | 86.7 | 12.0 |
| rmsnorm_bwd | torch.float16 | (8192, 4096) | H100 | 86.9 | 12.0 |
| rmsnorm_bwd | torch.float32 | (16384, 2048) | H100 | 158.5 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 2048) | H100 | 86.0 | 12.0 |
| rmsnorm_bwd | torch.float16 | (16384, 2048) | H100 | 86.1 | 12.0 |
| rmsnorm_bwd | torch.float32 | (32768, 2048) | H100 | 293.5 | 12.0 |
| rmsnorm_bwd | torch.bfloat16 | (32768, 2048) | H100 | 157.6 | 12.0 |
| rmsnorm_bwd | torch.float16 | (32768, 2048) | H100 | 157.9 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (128, 512) | H100 | 21.6 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 512) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (256, 512) | H100 | 21.5 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (256, 512) | H100 | 21.2 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (256, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (64, 1024) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (64, 1024) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (64, 1024) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (128, 1024) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 1024) | H100 | 21.2 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 1024) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (512, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (512, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (512, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 512) | H100 | 21.3 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (1024, 512) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 1024) | H100 | 21.7 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (1024, 1024) | H100 | 21.4 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (1024, 2048) | H100 | 24.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 2048) | H100 | 22.5 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (1024, 2048) | H100 | 22.5 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (2048, 1024) | H100 | 22.0 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 1024) | H100 | 21.8 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (2048, 1024) | H100 | 22.0 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (2048, 2048) | H100 | 41.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 2048) | H100 | 25.5 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (2048, 2048) | H100 | 25.6 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (4096, 2048) | H100 | 63.7 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 2048) | H100 | 42.0 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (4096, 2048) | H100 | 41.6 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (4096, 4096) | H100 | 106.5 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 4096) | H100 | 64.3 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (4096, 4096) | H100 | 65.0 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (8192, 2048) | H100 | 107.1 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 2048) | H100 | 64.4 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 2048) | H100 | 63.7 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (8192, 4096) | H100 | 192.5 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 4096) | H100 | 108.0 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 4096) | H100 | 108.4 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (16384, 2048) | H100 | 193.2 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 2048) | H100 | 107.5 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (16384, 2048) | H100 | 106.8 | 12.0 |
| rmsnorm_bwd_add | torch.float32 | (32768, 2048) | H100 | 368.1 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (32768, 2048) | H100 | 194.0 | 12.0 |
| rmsnorm_bwd_add | torch.float16 | (32768, 2048) | H100 | 194.0 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 42.6 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 512, 16, 128) | H100 | 42.8 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 48.4 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 1024, 16, 128) | H100 | 48.4 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 81.2 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 2048, 16, 128) | H100 | 80.7 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 111.9 | 12.0 |
| fused_attn_fwd | torch.float16 | (2, 2048, 16, 128) | H100 | 114.0 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 174.0 | 12.0 |
| fused_attn_fwd | torch.float16 | (4, 2048, 16, 128) | H100 | 176.3 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 291.4 | 12.0 |
| fused_attn_fwd | torch.float16 | (8, 2048, 16, 128) | H100 | 296.1 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 178.9 | 12.0 |
| fused_attn_fwd | torch.float16 | (1, 4096, 16, 128) | H100 | 181.0 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 288.3 | 12.0 |
| fused_attn_fwd | torch.float16 | (2, 4096, 16, 128) | H100 | 291.8 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 116.0 | 12.0 |
| fused_attn_fwd | torch.float16 | (4, 1024, 32, 128) | H100 | 116.3 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 185.6 | 12.0 |
| fused_attn_fwd | torch.float16 | (8, 1024, 32, 128) | H100 | 188.7 | 12.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 1968.6 | 12.0 |
| fused_attn_fwd | torch.float16 | (4, 8192, 16, 128) | H100 | 2016.6 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 66.1 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 512, 16, 128) | H100 | 64.7 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 91.0 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 1024, 16, 128) | H100 | 91.3 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 198.0 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 2048, 16, 128) | H100 | 200.3 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 305.0 | 12.0 |
| fused_attn_bwd | torch.float16 | (2, 2048, 16, 128) | H100 | 305.6 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 506.2 | 12.0 |
| fused_attn_bwd | torch.float16 | (4, 2048, 16, 128) | H100 | 509.7 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 911.3 | 12.0 |
| fused_attn_bwd | torch.float16 | (8, 2048, 16, 128) | H100 | 945.6 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 484.5 | 12.0 |
| fused_attn_bwd | torch.float16 | (1, 4096, 16, 128) | H100 | 483.6 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 822.9 | 12.0 |
| fused_attn_bwd | torch.float16 | (2, 4096, 16, 128) | H100 | 835.3 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 353.4 | 12.0 |
| fused_attn_bwd | torch.float16 | (4, 1024, 32, 128) | H100 | 352.2 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 625.7 | 12.0 |
| fused_attn_bwd | torch.float16 | (8, 1024, 32, 128) | H100 | 627.6 | 12.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 5546.1 | 12.0 |
| fused_attn_bwd | torch.float16 | (4, 8192, 16, 128) | H100 | 5896.4 | 12.0 |
