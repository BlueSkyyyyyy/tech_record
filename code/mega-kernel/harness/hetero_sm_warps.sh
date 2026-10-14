#!/usr/bin/env bash
# P90-1（§78）复现：补 P89-1（§77）§77.8 明留的尾巴
#   「O 只 1/2/4 且固定 O·T=1024 → 未单独扫每 SM warp 数，故『加 warp 摊』的完整验证仍缺口」。
#   在**同一 SM 角色绑定、同一逻辑工作、同一模式份额**下，把「每 SM 总线程/warp 数 Wsm=O·T」
#   变成唯一新变量：固定 T=256 扫 O∈{1,2,4,8}（Wsm=256→2048），并补 Wsm=1024 的 T 拆分
#   （T=512/O=2、T=1024/O=1）做 ·T 维交叉。检验「发射受限 × 访存受限」共驻税（§72.8 缓解 (a)
#   的『加 warp 摊』）能否随 Wsm↑ 收回，及 §73「税是全局」在高 warp 下是否仍成立。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_warps.cu
BIN=/tmp/hetero_sm_warps
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §78 SM 角色绑定 × 每 SM warp 数（Wsm=O·T）（P90-1，补 §77.8 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_warps_ptxas.txt || exit 1
grep -E "Function properties for _Z11pat_hetero|bytes stack frame|registers, used" /tmp/hetero_sm_warps_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 O=1/2/4/8 / §B solo×6 列 / §C 三映射×warp / §C2 税×Wsm / §D SM 分配 / §E coal+scalar 份额×Wsm）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar 在 Wsm=256 (O1/T256) vs Wsm=2048 (O8/T256) 下 bind 的 DRAM 字节 / 活跃 warp / 发射 ====="
if command -v ncu >/dev/null 2>&1; then
  for O in 1 8; do
    echo "--- map=1 (bind), seq 1,0,1,0 (coal+scalar), T=256 O=$O (Wsm=$((O * 256))) ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run 256 "$O" 1 1 1,0,1,0 2>&1 | grep -E "pat_hetero|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
