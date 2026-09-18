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
| rmsnorm_fwd | torch.float32 | (1024, 512) | H100 | 2.4 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float16 | (1024, 512) | H100 | 2.0 |
| rmsnorm_fwd | torch.float32 | (1024, 1024) | H100 | 3.3 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 1024) | H100 | 2.6 |
| rmsnorm_fwd | torch.float16 | (1024, 1024) | H100 | 2.5 |
| rmsnorm_fwd | torch.float32 | (1024, 2048) | H100 | 5.2 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 2048) | H100 | 3.8 |
| rmsnorm_fwd | torch.float16 | (1024, 2048) | H100 | 3.6 |
| rmsnorm_fwd | torch.float32 | (2048, 1024) | H100 | 4.6 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 1024) | H100 | 3.5 |
| rmsnorm_fwd | torch.float16 | (2048, 1024) | H100 | 3.5 |
| rmsnorm_fwd | torch.float32 | (2048, 2048) | H100 | 7.6 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 2048) | H100 | 5.0 |
| rmsnorm_fwd | torch.float16 | (2048, 2048) | H100 | 4.9 |
| rmsnorm_fwd | torch.float32 | (4096, 2048) | H100 | 24.2 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 2048) | H100 | 7.9 |
| rmsnorm_fwd | torch.float16 | (4096, 2048) | H100 | 7.6 |
| rmsnorm_fwd | torch.float32 | (4096, 4096) | H100 | 49.7 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 4096) | H100 | 26.0 |
| rmsnorm_fwd | torch.float16 | (4096, 4096) | H100 | 25.9 |
| rmsnorm_fwd | torch.float32 | (8192, 1024) | H100 | 24.1 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 1024) | H100 | 7.5 |
| rmsnorm_fwd | torch.float16 | (8192, 1024) | H100 | 7.4 |
| rmsnorm_fwd | torch.float32 | (8192, 2048) | H100 | 49.4 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 2048) | H100 | 25.7 |
| rmsnorm_fwd | torch.float16 | (8192, 2048) | H100 | 25.5 |
| rmsnorm_fwd | torch.float32 | (8192, 4096) | H100 | 96.6 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 4096) | H100 | 51.6 |
| rmsnorm_fwd | torch.float16 | (8192, 4096) | H100 | 51.5 |
| rmsnorm_fwd | torch.float32 | (16384, 2048) | H100 | 95.6 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 2048) | H100 | 51.3 |
| rmsnorm_fwd | torch.float16 | (16384, 2048) | H100 | 51.2 |
| rmsnorm_fwd | torch.float32 | (32768, 2048) | H100 | 187.3 |
| rmsnorm_fwd | torch.bfloat16 | (32768, 2048) | H100 | 98.2 |
| rmsnorm_fwd | torch.float16 | (32768, 2048) | H100 | 98.6 |
| rmsnorm_fwd | torch.float32 | (4096, 7168) | H100 | 88.9 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 7168) | H100 | 88.9 |
| rmsnorm_fwd | torch.float16 | (4096, 7168) | H100 | 88.7 |
| rmsnorm_fwd | torch.float32 | (8192, 7168) | H100 | 169.8 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 7168) | H100 | 172.3 |
| rmsnorm_fwd | torch.float16 | (8192, 7168) | H100 | 172.1 |
| rmsnorm_fwd | torch.float32 | (16384, 4096) | H100 | 189.7 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 4096) | H100 | 100.7 |
| rmsnorm_fwd | torch.float16 | (16384, 4096) | H100 | 100.5 |
| rmsnorm_fwd | torch.float32 | (16384, 7168) | H100 | 332.9 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 7168) | H100 | 335.8 |
| rmsnorm_fwd | torch.float16 | (16384, 7168) | H100 | 335.5 |
| rmsnorm_bwd | torch.float32 | (128, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (128, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd | torch.float32 | (256, 512) | H100 | 5.2 |
| rmsnorm_bwd | torch.bfloat16 | (256, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.float32 | (64, 1024) | H100 | 5.0 |
| rmsnorm_bwd | torch.bfloat16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd | torch.float16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd | torch.float32 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd | torch.bfloat16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float32 | (512, 512) | H100 | 5.3 |
| rmsnorm_bwd | torch.bfloat16 | (512, 512) | H100 | 5.5 |
| rmsnorm_bwd | torch.float16 | (512, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.float32 | (1024, 512) | H100 | 5.8 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 512) | H100 | 5.8 |
| rmsnorm_bwd | torch.float16 | (1024, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float32 | (1024, 1024) | H100 | 6.6 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 1024) | H100 | 5.6 |
| rmsnorm_bwd | torch.float16 | (1024, 1024) | H100 | 5.7 |
| rmsnorm_bwd | torch.float32 | (1024, 2048) | H100 | 10.0 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd | torch.float16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd | torch.float32 | (2048, 1024) | H100 | 8.6 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 1024) | H100 | 6.5 |
| rmsnorm_bwd | torch.float16 | (2048, 1024) | H100 | 6.5 |
| rmsnorm_bwd | torch.float32 | (2048, 2048) | H100 | 22.2 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 2048) | H100 | 11.1 |
| rmsnorm_bwd | torch.float16 | (2048, 2048) | H100 | 11.0 |
| rmsnorm_bwd | torch.float32 | (4096, 2048) | H100 | 40.6 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 2048) | H100 | 23.1 |
| rmsnorm_bwd | torch.float16 | (4096, 2048) | H100 | 23.0 |
| rmsnorm_bwd | torch.float32 | (4096, 4096) | H100 | 77.1 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 4096) | H100 | 41.1 |
| rmsnorm_bwd | torch.float16 | (4096, 4096) | H100 | 41.4 |
| rmsnorm_bwd | torch.float32 | (8192, 1024) | H100 | 38.7 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 1024) | H100 | 20.9 |
| rmsnorm_bwd | torch.float16 | (8192, 1024) | H100 | 21.0 |
| rmsnorm_bwd | torch.float32 | (8192, 2048) | H100 | 77.2 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 2048) | H100 | 40.4 |
| rmsnorm_bwd | torch.float16 | (8192, 2048) | H100 | 40.5 |
| rmsnorm_bwd | torch.float32 | (8192, 4096) | H100 | 145.2 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 4096) | H100 | 73.8 |
| rmsnorm_bwd | torch.float16 | (8192, 4096) | H100 | 74.0 |
| rmsnorm_bwd | torch.float32 | (16384, 2048) | H100 | 145.3 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 2048) | H100 | 73.3 |
| rmsnorm_bwd | torch.float16 | (16384, 2048) | H100 | 73.2 |
| rmsnorm_bwd | torch.float32 | (32768, 2048) | H100 | 281.0 |
| rmsnorm_bwd | torch.bfloat16 | (32768, 2048) | H100 | 144.8 |
| rmsnorm_bwd | torch.float16 | (32768, 2048) | H100 | 145.0 |
| rmsnorm_bwd | torch.float32 | (4096, 7168) | H100 | 189.7 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 7168) | H100 | 131.8 |
| rmsnorm_bwd | torch.float16 | (4096, 7168) | H100 | 132.2 |
| rmsnorm_bwd | torch.float32 | (8192, 7168) | H100 | 358.3 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 7168) | H100 | 238.6 |
| rmsnorm_bwd | torch.float16 | (8192, 7168) | H100 | 238.2 |
| rmsnorm_bwd | torch.float32 | (16384, 4096) | H100 | 280.3 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 4096) | H100 | 146.7 |
| rmsnorm_bwd | torch.float16 | (16384, 4096) | H100 | 146.8 |
| rmsnorm_bwd | torch.float32 | (16384, 7168) | H100 | 690.4 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 7168) | H100 | 452.2 |
| rmsnorm_bwd | torch.float16 | (16384, 7168) | H100 | 450.7 |
| rmsnorm_bwd_add | torch.float32 | (128, 512) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (256, 512) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float32 | (64, 1024) | H100 | 4.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd_add | torch.float16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd_add | torch.float32 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (512, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.bfloat16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.float16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.float32 | (1024, 512) | H100 | 5.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float32 | (1024, 1024) | H100 | 6.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 1024) | H100 | 5.8 |
| rmsnorm_bwd_add | torch.float16 | (1024, 1024) | H100 | 5.9 |
| rmsnorm_bwd_add | torch.float32 | (1024, 2048) | H100 | 12.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 2048) | H100 | 8.9 |
| rmsnorm_bwd_add | torch.float16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd_add | torch.float32 | (2048, 1024) | H100 | 8.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 1024) | H100 | 6.8 |
| rmsnorm_bwd_add | torch.float16 | (2048, 1024) | H100 | 7.0 |
| rmsnorm_bwd_add | torch.float32 | (2048, 2048) | H100 | 28.3 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 2048) | H100 | 12.6 |
| rmsnorm_bwd_add | torch.float16 | (2048, 2048) | H100 | 12.7 |
| rmsnorm_bwd_add | torch.float32 | (4096, 2048) | H100 | 50.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 2048) | H100 | 28.8 |
| rmsnorm_bwd_add | torch.float16 | (4096, 2048) | H100 | 28.8 |
| rmsnorm_bwd_add | torch.float32 | (4096, 4096) | H100 | 93.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 4096) | H100 | 51.7 |
| rmsnorm_bwd_add | torch.float16 | (4096, 4096) | H100 | 51.9 |
| rmsnorm_bwd_add | torch.float32 | (8192, 1024) | H100 | 48.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 1024) | H100 | 27.1 |
| rmsnorm_bwd_add | torch.float16 | (8192, 1024) | H100 | 27.3 |
| rmsnorm_bwd_add | torch.float32 | (8192, 2048) | H100 | 94.5 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 2048) | H100 | 50.9 |
| rmsnorm_bwd_add | torch.float16 | (8192, 2048) | H100 | 50.8 |
| rmsnorm_bwd_add | torch.float32 | (8192, 4096) | H100 | 179.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 4096) | H100 | 95.3 |
| rmsnorm_bwd_add | torch.float16 | (8192, 4096) | H100 | 95.4 |
| rmsnorm_bwd_add | torch.float32 | (16384, 2048) | H100 | 180.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 2048) | H100 | 94.6 |
| rmsnorm_bwd_add | torch.float16 | (16384, 2048) | H100 | 94.5 |
| rmsnorm_bwd_add | torch.float32 | (32768, 2048) | H100 | 353.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (32768, 2048) | H100 | 181.2 |
| rmsnorm_bwd_add | torch.float16 | (32768, 2048) | H100 | 181.1 |
| rmsnorm_bwd_add | torch.float32 | (4096, 7168) | H100 | 338.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 7168) | H100 | 168.9 |
| rmsnorm_bwd_add | torch.float16 | (4096, 7168) | H100 | 168.4 |
| rmsnorm_bwd_add | torch.float32 | (8192, 7168) | H100 | 655.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 7168) | H100 | 314.2 |
| rmsnorm_bwd_add | torch.float16 | (8192, 7168) | H100 | 314.0 |
| rmsnorm_bwd_add | torch.float32 | (16384, 4096) | H100 | 352.1 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 4096) | H100 | 181.8 |
| rmsnorm_bwd_add | torch.float16 | (16384, 4096) | H100 | 181.6 |
| rmsnorm_bwd_add | torch.float32 | (16384, 7168) | H100 | 1294.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 7168) | H100 | 603.3 |
| rmsnorm_bwd_add | torch.float16 | (16384, 7168) | H100 | 602.0 |
| fused_attn_fwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 11.9 |
| fused_attn_fwd | torch.float16 | (1, 512, 16, 128) | H100 | 11.9 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 18.6 |
| fused_attn_fwd | torch.float16 | (1, 1024, 16, 128) | H100 | 18.7 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 51.1 |
| fused_attn_fwd | torch.float16 | (1, 2048, 16, 128) | H100 | 51.5 |
| fused_attn_fwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 81.4 |
| fused_attn_fwd | torch.float16 | (2, 2048, 16, 128) | H100 | 82.3 |
| fused_attn_fwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 141.0 |
| fused_attn_fwd | torch.float16 | (4, 2048, 16, 128) | H100 | 143.9 |
| fused_attn_fwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 258.6 |
| fused_attn_fwd | torch.float16 | (8, 2048, 16, 128) | H100 | 264.1 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 148.6 |
| fused_attn_fwd | torch.float16 | (1, 4096, 16, 128) | H100 | 149.9 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 255.2 |
| fused_attn_fwd | torch.float16 | (2, 4096, 16, 128) | H100 | 259.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 32, 128) | H100 | 31.2 |
| fused_attn_fwd | torch.float16 | (1, 1024, 32, 128) | H100 | 31.2 |
| fused_attn_fwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 83.0 |
| fused_attn_fwd | torch.float16 | (4, 1024, 32, 128) | H100 | 83.8 |
| fused_attn_fwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 151.5 |
| fused_attn_fwd | torch.float16 | (8, 1024, 32, 128) | H100 | 154.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 1891.9 |
| fused_attn_fwd | torch.float16 | (4, 8192, 16, 128) | H100 | 1970.1 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 608.4 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 630.7 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 483.8 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 481.7 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 255.9 |
| fused_attn_fwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 259.9 |
| fused_attn_fwd | fp8 | (1, 1024, 32, 128) | H100 | 24.8 |
| fused_attn_bwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 32.6 |
| fused_attn_bwd | torch.float16 | (1, 512, 16, 128) | H100 | 32.5 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 58.5 |
| fused_attn_bwd | torch.float16 | (1, 1024, 16, 128) | H100 | 58.5 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 164.0 |
| fused_attn_bwd | torch.float16 | (1, 2048, 16, 128) | H100 | 164.4 |
| fused_attn_bwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 268.7 |
| fused_attn_bwd | torch.float16 | (2, 2048, 16, 128) | H100 | 270.0 |
| fused_attn_bwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 475.0 |
| fused_attn_bwd | torch.float16 | (4, 2048, 16, 128) | H100 | 477.3 |
| fused_attn_bwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 893.6 |
| fused_attn_bwd | torch.float16 | (8, 2048, 16, 128) | H100 | 917.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 450.9 |
| fused_attn_bwd | torch.float16 | (1, 4096, 16, 128) | H100 | 448.2 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 806.0 |
| fused_attn_bwd | torch.float16 | (2, 4096, 16, 128) | H100 | 828.4 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 32, 128) | H100 | 105.2 |
| fused_attn_bwd | torch.float16 | (1, 1024, 32, 128) | H100 | 105.4 |
| fused_attn_bwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 318.9 |
| fused_attn_bwd | torch.float16 | (4, 1024, 32, 128) | H100 | 318.6 |
| fused_attn_bwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 592.3 |
| fused_attn_bwd | torch.float16 | (8, 1024, 32, 128) | H100 | 595.3 |
| fused_attn_bwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 5517.2 |
| fused_attn_bwd | torch.float16 | (4, 8192, 16, 128) | H100 | 5861.9 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 2166.5 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 2219.6 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 1545.5 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 1580.5 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 797.8 |
| fused_attn_bwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 821.8 |
| fused_attn_bwd | fp8 | (1, 1024, 32, 128) | H100 | 75.0 |
