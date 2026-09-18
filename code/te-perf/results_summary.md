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
| rmsnorm_fwd | torch.float32 | (1024, 2048) | H100 | 5.3 |
| rmsnorm_fwd | torch.bfloat16 | (1024, 2048) | H100 | 3.8 |
| rmsnorm_fwd | torch.float16 | (1024, 2048) | H100 | 3.6 |
| rmsnorm_fwd | torch.float32 | (2048, 1024) | H100 | 4.8 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 1024) | H100 | 3.5 |
| rmsnorm_fwd | torch.float16 | (2048, 1024) | H100 | 3.4 |
| rmsnorm_fwd | torch.float32 | (2048, 2048) | H100 | 7.8 |
| rmsnorm_fwd | torch.bfloat16 | (2048, 2048) | H100 | 5.0 |
| rmsnorm_fwd | torch.float16 | (2048, 2048) | H100 | 4.9 |
| rmsnorm_fwd | torch.float32 | (4096, 2048) | H100 | 24.3 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 2048) | H100 | 7.8 |
| rmsnorm_fwd | torch.float16 | (4096, 2048) | H100 | 7.7 |
| rmsnorm_fwd | torch.float32 | (4096, 4096) | H100 | 49.7 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 4096) | H100 | 26.0 |
| rmsnorm_fwd | torch.float16 | (4096, 4096) | H100 | 25.9 |
| rmsnorm_fwd | torch.float32 | (8192, 1024) | H100 | 23.9 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 1024) | H100 | 7.9 |
| rmsnorm_fwd | torch.float16 | (8192, 1024) | H100 | 7.4 |
| rmsnorm_fwd | torch.float32 | (8192, 2048) | H100 | 49.3 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 2048) | H100 | 25.6 |
| rmsnorm_fwd | torch.float16 | (8192, 2048) | H100 | 25.4 |
| rmsnorm_fwd | torch.float32 | (8192, 4096) | H100 | 96.6 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 4096) | H100 | 51.6 |
| rmsnorm_fwd | torch.float16 | (8192, 4096) | H100 | 51.5 |
| rmsnorm_fwd | torch.float32 | (16384, 2048) | H100 | 95.8 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 2048) | H100 | 51.5 |
| rmsnorm_fwd | torch.float16 | (16384, 2048) | H100 | 51.2 |
| rmsnorm_fwd | torch.float32 | (32768, 2048) | H100 | 187.2 |
| rmsnorm_fwd | torch.bfloat16 | (32768, 2048) | H100 | 98.4 |
| rmsnorm_fwd | torch.float16 | (32768, 2048) | H100 | 98.1 |
| rmsnorm_fwd | torch.float32 | (4096, 7168) | H100 | 88.9 |
| rmsnorm_fwd | torch.bfloat16 | (4096, 7168) | H100 | 88.7 |
| rmsnorm_fwd | torch.float16 | (4096, 7168) | H100 | 88.6 |
| rmsnorm_fwd | torch.float32 | (8192, 7168) | H100 | 170.0 |
| rmsnorm_fwd | torch.bfloat16 | (8192, 7168) | H100 | 172.2 |
| rmsnorm_fwd | torch.float16 | (8192, 7168) | H100 | 172.1 |
| rmsnorm_fwd | torch.float32 | (16384, 4096) | H100 | 189.2 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 4096) | H100 | 100.8 |
| rmsnorm_fwd | torch.float16 | (16384, 4096) | H100 | 100.4 |
| rmsnorm_fwd | torch.float32 | (16384, 7168) | H100 | 333.1 |
| rmsnorm_fwd | torch.bfloat16 | (16384, 7168) | H100 | 335.8 |
| rmsnorm_fwd | torch.float16 | (16384, 7168) | H100 | 335.5 |
| rmsnorm_bwd | torch.float32 | (128, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (128, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd | torch.float32 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.bfloat16 | (256, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd | torch.float32 | (64, 1024) | H100 | 5.0 |
| rmsnorm_bwd | torch.bfloat16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd | torch.float16 | (64, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float32 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd | torch.bfloat16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float16 | (128, 1024) | H100 | 4.9 |
| rmsnorm_bwd | torch.float32 | (512, 512) | H100 | 5.3 |
| rmsnorm_bwd | torch.bfloat16 | (512, 512) | H100 | 5.5 |
| rmsnorm_bwd | torch.float16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd | torch.float32 | (1024, 512) | H100 | 5.8 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 512) | H100 | 5.8 |
| rmsnorm_bwd | torch.float16 | (1024, 512) | H100 | 5.4 |
| rmsnorm_bwd | torch.float32 | (1024, 1024) | H100 | 6.7 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 1024) | H100 | 5.6 |
| rmsnorm_bwd | torch.float16 | (1024, 1024) | H100 | 5.6 |
| rmsnorm_bwd | torch.float32 | (1024, 2048) | H100 | 9.9 |
| rmsnorm_bwd | torch.bfloat16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd | torch.float16 | (1024, 2048) | H100 | 9.1 |
| rmsnorm_bwd | torch.float32 | (2048, 1024) | H100 | 8.6 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 1024) | H100 | 6.5 |
| rmsnorm_bwd | torch.float16 | (2048, 1024) | H100 | 6.6 |
| rmsnorm_bwd | torch.float32 | (2048, 2048) | H100 | 22.1 |
| rmsnorm_bwd | torch.bfloat16 | (2048, 2048) | H100 | 11.2 |
| rmsnorm_bwd | torch.float16 | (2048, 2048) | H100 | 11.0 |
| rmsnorm_bwd | torch.float32 | (4096, 2048) | H100 | 40.6 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 2048) | H100 | 22.9 |
| rmsnorm_bwd | torch.float16 | (4096, 2048) | H100 | 23.1 |
| rmsnorm_bwd | torch.float32 | (4096, 4096) | H100 | 77.1 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 4096) | H100 | 41.1 |
| rmsnorm_bwd | torch.float16 | (4096, 4096) | H100 | 41.3 |
| rmsnorm_bwd | torch.float32 | (8192, 1024) | H100 | 38.8 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 1024) | H100 | 20.9 |
| rmsnorm_bwd | torch.float16 | (8192, 1024) | H100 | 21.1 |
| rmsnorm_bwd | torch.float32 | (8192, 2048) | H100 | 77.1 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 2048) | H100 | 40.5 |
| rmsnorm_bwd | torch.float16 | (8192, 2048) | H100 | 40.4 |
| rmsnorm_bwd | torch.float32 | (8192, 4096) | H100 | 145.2 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 4096) | H100 | 73.8 |
| rmsnorm_bwd | torch.float16 | (8192, 4096) | H100 | 74.1 |
| rmsnorm_bwd | torch.float32 | (16384, 2048) | H100 | 145.8 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 2048) | H100 | 73.3 |
| rmsnorm_bwd | torch.float16 | (16384, 2048) | H100 | 73.3 |
| rmsnorm_bwd | torch.float32 | (32768, 2048) | H100 | 281.0 |
| rmsnorm_bwd | torch.bfloat16 | (32768, 2048) | H100 | 144.5 |
| rmsnorm_bwd | torch.float16 | (32768, 2048) | H100 | 145.1 |
| rmsnorm_bwd | torch.float32 | (4096, 7168) | H100 | 189.2 |
| rmsnorm_bwd | torch.bfloat16 | (4096, 7168) | H100 | 132.4 |
| rmsnorm_bwd | torch.float16 | (4096, 7168) | H100 | 131.6 |
| rmsnorm_bwd | torch.float32 | (8192, 7168) | H100 | 358.4 |
| rmsnorm_bwd | torch.bfloat16 | (8192, 7168) | H100 | 238.1 |
| rmsnorm_bwd | torch.float16 | (8192, 7168) | H100 | 237.3 |
| rmsnorm_bwd | torch.float32 | (16384, 4096) | H100 | 279.9 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 4096) | H100 | 146.8 |
| rmsnorm_bwd | torch.float16 | (16384, 4096) | H100 | 146.8 |
| rmsnorm_bwd | torch.float32 | (16384, 7168) | H100 | 689.8 |
| rmsnorm_bwd | torch.bfloat16 | (16384, 7168) | H100 | 452.1 |
| rmsnorm_bwd | torch.float16 | (16384, 7168) | H100 | 452.9 |
| rmsnorm_bwd_add | torch.float32 | (128, 512) | H100 | 4.8 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float16 | (128, 512) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (256, 512) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float16 | (256, 512) | H100 | 5.1 |
| rmsnorm_bwd_add | torch.float32 | (64, 1024) | H100 | 4.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (64, 1024) | H100 | 4.8 |
| rmsnorm_bwd_add | torch.float16 | (64, 1024) | H100 | 4.9 |
| rmsnorm_bwd_add | torch.float32 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.bfloat16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float16 | (128, 1024) | H100 | 5.0 |
| rmsnorm_bwd_add | torch.float32 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (512, 512) | H100 | 5.3 |
| rmsnorm_bwd_add | torch.float16 | (512, 512) | H100 | 5.2 |
| rmsnorm_bwd_add | torch.float32 | (1024, 512) | H100 | 5.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float16 | (1024, 512) | H100 | 5.5 |
| rmsnorm_bwd_add | torch.float32 | (1024, 1024) | H100 | 6.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 1024) | H100 | 5.9 |
| rmsnorm_bwd_add | torch.float16 | (1024, 1024) | H100 | 5.9 |
| rmsnorm_bwd_add | torch.float32 | (1024, 2048) | H100 | 11.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (1024, 2048) | H100 | 8.8 |
| rmsnorm_bwd_add | torch.float16 | (1024, 2048) | H100 | 9.2 |
| rmsnorm_bwd_add | torch.float32 | (2048, 1024) | H100 | 8.6 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 1024) | H100 | 6.8 |
| rmsnorm_bwd_add | torch.float16 | (2048, 1024) | H100 | 6.8 |
| rmsnorm_bwd_add | torch.float32 | (2048, 2048) | H100 | 28.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (2048, 2048) | H100 | 12.8 |
| rmsnorm_bwd_add | torch.float16 | (2048, 2048) | H100 | 12.7 |
| rmsnorm_bwd_add | torch.float32 | (4096, 2048) | H100 | 50.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 2048) | H100 | 28.8 |
| rmsnorm_bwd_add | torch.float16 | (4096, 2048) | H100 | 28.9 |
| rmsnorm_bwd_add | torch.float32 | (4096, 4096) | H100 | 93.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 4096) | H100 | 51.7 |
| rmsnorm_bwd_add | torch.float16 | (4096, 4096) | H100 | 52.1 |
| rmsnorm_bwd_add | torch.float32 | (8192, 1024) | H100 | 48.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 1024) | H100 | 27.0 |
| rmsnorm_bwd_add | torch.float16 | (8192, 1024) | H100 | 27.1 |
| rmsnorm_bwd_add | torch.float32 | (8192, 2048) | H100 | 94.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 2048) | H100 | 50.8 |
| rmsnorm_bwd_add | torch.float16 | (8192, 2048) | H100 | 50.8 |
| rmsnorm_bwd_add | torch.float32 | (8192, 4096) | H100 | 179.9 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 4096) | H100 | 95.2 |
| rmsnorm_bwd_add | torch.float16 | (8192, 4096) | H100 | 95.6 |
| rmsnorm_bwd_add | torch.float32 | (16384, 2048) | H100 | 180.2 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 2048) | H100 | 94.3 |
| rmsnorm_bwd_add | torch.float16 | (16384, 2048) | H100 | 94.7 |
| rmsnorm_bwd_add | torch.float32 | (32768, 2048) | H100 | 353.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (32768, 2048) | H100 | 181.4 |
| rmsnorm_bwd_add | torch.float16 | (32768, 2048) | H100 | 181.4 |
| rmsnorm_bwd_add | torch.float32 | (4096, 7168) | H100 | 336.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (4096, 7168) | H100 | 168.8 |
| rmsnorm_bwd_add | torch.float16 | (4096, 7168) | H100 | 168.3 |
| rmsnorm_bwd_add | torch.float32 | (8192, 7168) | H100 | 656.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (8192, 7168) | H100 | 314.4 |
| rmsnorm_bwd_add | torch.float16 | (8192, 7168) | H100 | 314.5 |
| rmsnorm_bwd_add | torch.float32 | (16384, 4096) | H100 | 351.7 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 4096) | H100 | 181.4 |
| rmsnorm_bwd_add | torch.float16 | (16384, 4096) | H100 | 182.2 |
| rmsnorm_bwd_add | torch.float32 | (16384, 7168) | H100 | 1296.4 |
| rmsnorm_bwd_add | torch.bfloat16 | (16384, 7168) | H100 | 603.8 |
| rmsnorm_bwd_add | torch.float16 | (16384, 7168) | H100 | 602.5 |
| fused_attn_fwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 11.8 |
| fused_attn_fwd | torch.float16 | (1, 512, 16, 128) | H100 | 11.8 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 18.5 |
| fused_attn_fwd | torch.float16 | (1, 1024, 16, 128) | H100 | 18.6 |
| fused_attn_fwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 51.1 |
| fused_attn_fwd | torch.float16 | (1, 2048, 16, 128) | H100 | 51.5 |
| fused_attn_fwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 81.3 |
| fused_attn_fwd | torch.float16 | (2, 2048, 16, 128) | H100 | 82.1 |
| fused_attn_fwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 140.9 |
| fused_attn_fwd | torch.float16 | (4, 2048, 16, 128) | H100 | 142.2 |
| fused_attn_fwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 256.2 |
| fused_attn_fwd | torch.float16 | (8, 2048, 16, 128) | H100 | 264.1 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 148.9 |
| fused_attn_fwd | torch.float16 | (1, 4096, 16, 128) | H100 | 150.0 |
| fused_attn_fwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 255.9 |
| fused_attn_fwd | torch.float16 | (2, 4096, 16, 128) | H100 | 259.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 1024, 32, 128) | H100 | 31.1 |
| fused_attn_fwd | torch.float16 | (1, 1024, 32, 128) | H100 | 31.1 |
| fused_attn_fwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 83.0 |
| fused_attn_fwd | torch.float16 | (4, 1024, 32, 128) | H100 | 84.2 |
| fused_attn_fwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 150.3 |
| fused_attn_fwd | torch.float16 | (8, 1024, 32, 128) | H100 | 154.0 |
| fused_attn_fwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 1887.9 |
| fused_attn_fwd | torch.float16 | (4, 8192, 16, 128) | H100 | 1925.9 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 613.3 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 630.4 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 486.8 |
| fused_attn_fwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 498.2 |
| fused_attn_fwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 255.4 |
| fused_attn_fwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 259.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 512, 16, 128) | H100 | 32.7 |
| fused_attn_bwd | torch.float16 | (1, 512, 16, 128) | H100 | 32.5 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 16, 128) | H100 | 59.0 |
| fused_attn_bwd | torch.float16 | (1, 1024, 16, 128) | H100 | 58.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 2048, 16, 128) | H100 | 164.1 |
| fused_attn_bwd | torch.float16 | (1, 2048, 16, 128) | H100 | 163.2 |
| fused_attn_bwd | torch.bfloat16 | (2, 2048, 16, 128) | H100 | 269.0 |
| fused_attn_bwd | torch.float16 | (2, 2048, 16, 128) | H100 | 270.1 |
| fused_attn_bwd | torch.bfloat16 | (4, 2048, 16, 128) | H100 | 470.5 |
| fused_attn_bwd | torch.float16 | (4, 2048, 16, 128) | H100 | 474.3 |
| fused_attn_bwd | torch.bfloat16 | (8, 2048, 16, 128) | H100 | 887.4 |
| fused_attn_bwd | torch.float16 | (8, 2048, 16, 128) | H100 | 892.1 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 16, 128) | H100 | 448.2 |
| fused_attn_bwd | torch.float16 | (1, 4096, 16, 128) | H100 | 451.5 |
| fused_attn_bwd | torch.bfloat16 | (2, 4096, 16, 128) | H100 | 806.2 |
| fused_attn_bwd | torch.float16 | (2, 4096, 16, 128) | H100 | 801.2 |
| fused_attn_bwd | torch.bfloat16 | (1, 1024, 32, 128) | H100 | 106.0 |
| fused_attn_bwd | torch.float16 | (1, 1024, 32, 128) | H100 | 105.1 |
| fused_attn_bwd | torch.bfloat16 | (4, 1024, 32, 128) | H100 | 318.7 |
| fused_attn_bwd | torch.float16 | (4, 1024, 32, 128) | H100 | 318.7 |
| fused_attn_bwd | torch.bfloat16 | (8, 1024, 32, 128) | H100 | 592.1 |
| fused_attn_bwd | torch.float16 | (8, 1024, 32, 128) | H100 | 595.1 |
| fused_attn_bwd | torch.bfloat16 | (4, 8192, 16, 128) | H100 | 5501.7 |
| fused_attn_bwd | torch.float16 | (4, 8192, 16, 128) | H100 | 5809.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 192, 128) | H100 | 2170.7 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 192, 128) | H100 | 2208.7 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 64, 128, 128) | H100 | 1542.8 |
| fused_attn_bwd | torch.float16 | (1, 4096, 64, 128, 128) | H100 | 1574.5 |
| fused_attn_bwd | torch.bfloat16 | (1, 4096, 32, 128, 128) | H100 | 803.9 |
| fused_attn_bwd | torch.float16 | (1, 4096, 32, 128, 128) | H100 | 804.2 |
