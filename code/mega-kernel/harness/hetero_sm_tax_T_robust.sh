#!/usr/bin/env bash
# P96-1（§84）复现：补 P95-1（§83）**§83.5 明留的两条尾巴**——
#   (tail 1)「混合 T 趋势只有 ~4%、单模式 ~3–5%，**只取 3 次 run 均值**…据此只可说『L2 停顿不进 T』
#            这条否定性结论，不可把 +4.3% 当成确定的一阶现象」；
#   (tail 2)「未做 `--section MemoryWorkloadAnalysis` 的延迟分解」。
# 做法：复用 §82 同一内核（src/hetero_sm_tax_T.cu，**新增一个 `robust` 子命令，不动 run/bench**）：
#   §A 稳健计时——每配置 S=30 次**单采样**、round-robin 交错（消除时钟/热漂移），给
#      min/median/mean/std + T512/T128、T256/T128 的 median 比与 bootstrap 95% CI（B=2000）；
#   §B ncu——把「内存延迟分解」补齐：L2/L1 命中率与吞吐、**每 warp 绝对 L2 停顿周期**
#      （`smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio`）+ 总 warp 延迟
#      （`smsp__average_warp_latency.ratio`），扫描 T 与单模式对照。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_tax_T.cu
BIN=/tmp/hetero_sm_tax_T_robust
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §84 块粒度 T 次级效应的统计稳健性 + 内存延迟分解（P96-1，补 §83.5 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a；U=8 R=32 map=bind（与 §82/§83 同配置同内核）"
echo "# DEV=$DEV"
which ncu || { echo "ncu 不可用，退出"; exit 1; }

echo
echo "===== 构建（--ptxas-options=-v，与 §82 同内核）====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_tax_T_robust_ptxas.txt || exit 1
grep -E "Function properties for _Z5pat_T|Used [0-9]+ registers|bytes spill" /tmp/hetero_sm_tax_T_robust_ptxas.txt | head -60

echo
echo "===== §A 稳健计时（S=30 单采样 + round-robin 交错 + bootstrap 95% CI）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 400 "$BIN" robust 2>&1

METRICS="gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warp_latency.ratio,\
smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,\
smsp__average_warp_latency_issue_stalled_lg_throttle.ratio,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"

run_ncu () {
  local tag="$1" T="$2" seq="$3"
  echo "----- [$tag] U=8 R=32 T=$T map=1(bind) seq=$seq -----"
  CUDA_VISIBLE_DEVICES="$DEV" timeout 400 ncu --target-processes all \
    --replay-mode application --cache-control none \
    -k "regex:pat_T" --launch-count 1 --launch-skip 0 \
    --metrics "$METRICS" \
    "$BIN" run 8 32 "$T" 1 "$seq" 2>&1 \
    | sed -n '/Metric Name/,/^$/p' | grep -E "Metric Name|----|duration|dram__|lts__|l1tex__|smsp__|GB/s"
  echo
}

echo "===== §B 内存延迟分解（ncu）：混合 scalar+l2res 扫 T ====="
for T in 128 256 512; do run_ncu "mixed scalar+l2res" "$T" "0,0,1,1"; done

echo "===== §B2 单模式对照：l2res16M solo 扫 T ====="
for T in 128 512; do run_ncu "l2res16M solo" "$T" "0,0,0,1"; done

echo "===== §B3 单模式对照：scalar4B solo 扫 T ====="
for T in 128 512; do run_ncu "scalar4B solo" "$T" "0,0,1,0"; done

echo "===== §B4 MemoryWorkloadAnalysis_Tables（混合 T=512，--print-details all）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 400 ncu --target-processes all \
  --replay-mode application --cache-control none \
  -k "regex:pat_T" --launch-count 1 --launch-skip 0 \
  --section MemoryWorkloadAnalysis_Tables --print-details all \
  "$BIN" run 8 32 512 1 0,0,1,1 2>&1 \
  | sed -n '/Memory Workload Analysis Tables/,$p' | grep -vE "^\s*$" | head -60

echo
echo "########## §84 完成 ##########"
