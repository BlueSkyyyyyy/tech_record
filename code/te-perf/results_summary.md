# TE 算子性能测试结果汇总

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
| fused_attn_bwd | fp8 | (1, 1024, 32, 128) | H100 | 75.1 |
| fused_attn_fwd_infer | torch.bfloat16 | (1, 4096, 128, 512, 512, 1) | H100 | 23221.7 |
| fused_attn_fwd_infer | torch.float16 | (1, 4096, 128, 512, 512, 1) | H100 | 23532.5 |
