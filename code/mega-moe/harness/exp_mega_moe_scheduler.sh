#!/usr/bin/env bash
# P110-1 实验 driver：DeepGEMM 的 Mega MoE 任务调度器 deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh（481 行）。
#
# 该文件是「dispatch + linear1 + SwiGLU + linear2 + combine」mega-kernel 唯一的任务切分入口，
# 也是 ring buffer 容量公式 get_num_max_live_pool_blocks 的来源。它被前面 100+ 篇以行号引用了 15+ 次
# （第 02/03/09/10/11/44 篇……），却从没被单独拆透（另外三个调度器 gemm.cuh / mega_gate / mega_mhc
# 已分别由第 108/92 篇补上）。它的分工全是整数/布局运算，不碰 UMMA/TMA，故可在 H100(sm90) 上
# 直接 include 真实头文件、编译运行真实设备代码。唯一 SM100 专属的是 publish_task 的 cluster 协作
# 与主 kernel 端 UMMA/TMA —— 仅做源码分析。
#   A) get_num_max_live_pool_blocks / get_num_l1_warmup_waves（host constexpr）vs 独立复刻
#   B) TaskInfo 的 ABI（sizeof/alignof/offsetof）与 is_valid/is_shared/get_umma_aligned_valid_m
#   C) L2KBlockDependency：arrive 的 red.xor 翻转 + wait 的 (ld_acq ^ expected_mask) 相位语义
#   D) create_task 的 warp 协作映射：pool block -> (expert, owner m_block, valid_m)
#   E) get_pool_block_offset / get_num_total_pool_blocks 的前缀和语义
#   F) get_next_task_idx 的 locality-domain 分域枚举是双射 + work-steal 阈值
#   bash harness/exp_mega_moe_scheduler.sh > harness/exp_mega_moe_scheduler.out.txt
# 对应源码 src/mega_moe_scheduler.cu。同一 .cu 同时编 sm_90a（运行）与 sm_100a（仅编译期覆盖）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=${HOME}/github/DeepGEMM
SRC=${HOME}/proj/tech_record/code/mega-moe/src/mega_moe_scheduler.cu
INC="-I$DG/deep_gemm/include -I$DG/third-party/cutlass/include -I$DG/third-party/deep_jit/include -I$DG/csrc"

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
nvcc --version | tail -1
sha256sum $DG/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh \
          $DG/deep_gemm/include/deep_gemm/layout/mega_moe.cuh | awk '{n=split($2,a,"/"); print substr($1,1,16), a[n]}'
wc -l $DG/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh

echo
echo "== 编译（sm_90a 运行 + sm_100a 仅编译覆盖 SM100 分支）=="
nvcc -std=c++17 -O2 -w --expt-relaxed-constexpr \
  -gencode arch=compute_90a,code=sm_90a \
  -gencode arch=compute_100a,code=sm_100a \
  $INC "$SRC" -o /tmp/mega_moe_scheduler
echo "build OK (sm90a + sm100a)"

echo
echo "== 运行（H100, sm90a cubin）=="
/tmp/mega_moe_scheduler
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
