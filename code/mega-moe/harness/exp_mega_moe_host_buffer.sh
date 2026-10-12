#!/usr/bin/env bash
# P32-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 Mega MoE 的 host 编排层与对称缓冲区布局
#   deep_gemm.mega.SymmBuffer / _C.get_symm_buffer_size_for_mega_moe / layout::MegaMoEBuffer /
#   scheduler::get_num_max_live_pool_blocks（纯 host，架构无关）。
#   bash harness/exp_mega_moe_host_buffer.sh > harness/exp_mega_moe_host_buffer.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_mega_moe_host_buffer.py"
