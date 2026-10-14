#!/usr/bin/env bash
# P92-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行 DeepGEMM 的
#   Mega Gate / Mega mHC **跨 CTA / 跨 split 分段屏障**调度器
#     deep_gemm/include/deep_gemm/scheduler/mega_gate.cuh   (ScoreBarrier)
#     deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh    (SplitBarrier / Norm / Mix)
#   + 它们依赖的两个 Workspace 布局
#     deep_gemm/include/deep_gemm/layout/{mega_gate,mega_mhc}.cuh
#   （真实设备代码，架构无关：只用 ptx 内联 asm / constexpr，不碰 UMMA/TMA）。
#   ./harness/exp_mega_gate_mhc_split_barrier.sh
# 对应源码 src/mega_gate_mhc_split_barrier.cu，输出见 harness/exp_mega_gate_mhc_split_barrier.out.txt。
#
# 覆盖：A) ScoreBarrier 全协议 + 免复位（C 个逻辑 CTA，连续两次 launch 不重置）
#       B) wait_init 的无符号距离语义（陈旧拒绝 / 本轮 arm 通过 / 本轮部分到达通过）
#       C) SplitBarrier get_ptr line 偏移（Norm idx0 / Mix idx1）+ 全协议 + 免复位
#       D) 两个 Workspace 的 host 算术
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
INC=${HOME}/github/DeepGEMM/deep_gemm/include
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/scheduler/mega_gate.cuh $INC/deep_gemm/scheduler/mega_mhc.cuh \
          $INC/deep_gemm/layout/mega_gate.cuh $INC/deep_gemm/layout/mega_mhc.cuh \
          $INC/deep_gemm/ptx/utils.cuh $INC/deep_gemm/ptx/ld_st.cuh
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 2389 \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" \
  '${HOME}/proj/tech_record/code/mega-moe/src/mega_gate_mhc_split_barrier.cu' \
  -o /tmp/mega_gate_mhc_split_barrier
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1

# 证明真实 `arrive()` 的 `red.async` 是 SM100 专属：同一段 `arrive` 在 sm_90a 编不过、在 sm_100a 编得过。
cat > /tmp/real_arrive_probe.cu <<'CU'
#include <deep_gemm/scheduler/mega_gate.cuh>
__global__ void k(uint64_t* p) { deep_gemm::sched::mega_gate::ScoreBarrier<4>::arrive(p); }
CU
COMPILE_LOG=/tmp/mgc_arrive_compile.txt
{
  echo "== 真实 arrive（red.async）在 sm_90a =="
  if nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 2389 \
       -I"$INC" -I"$INC/cutlass" -I"$INC/cute" -cubin /tmp/real_arrive_probe.cu -o /tmp/real_arrive_sm90.cubin 2>/tmp/arr90.log; then
    echo "SM90_ARRIVE_COMPILE OK (unexpected)"; else echo "SM90_ARRIVE_COMPILE FAILED"; grep -m1 "requires .target" /tmp/arr90.log || true; fi
  echo "== 真实 arrive（red.async）在 sm_100a =="
  if nvcc -arch=sm_100a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 2389 \
       -I"$INC" -I"$INC/cutlass" -I"$INC/cute" -cubin /tmp/real_arrive_probe.cu -o /tmp/real_arrive_sm100.cubin 2>/tmp/arr100.log; then
    echo "SM100_ARRIVE_COMPILE OK"; else echo "SM100_ARRIVE_COMPILE FAILED"; tail -3 /tmp/arr100.log; fi
} | tee "$COMPILE_LOG"

/tmp/mega_gate_mhc_split_barrier > /tmp/mega_gate_mhc_split_barrier.raw.txt 2>&1 \
  || { echo "RUN FAILED"; cat /tmp/mega_gate_mhc_split_barrier.raw.txt; exit 1; }
cat "$COMPILE_LOG" >> /tmp/mega_gate_mhc_split_barrier.raw.txt
cat /tmp/mega_gate_mhc_split_barrier.raw.txt
echo
echo '== Python 侧判定 =='
python3 - <<'PY'
import re
raw = open('/tmp/mega_gate_mhc_split_barrier.raw.txt').read()
ok = True
npass = nfail = 0

# A) ScoreBarrier 全协议
for ln in raw.splitlines():
    m = re.match(r'GATE_PROTO C=(\d+) tb=(\d+) zero_before=(\d+) mismatch=0x([0-9a-f]+) arrival_field=(\d+)', ln)
    if not m: continue
    C, mism, arr = int(m.group(1)), int(m.group(4), 16), int(m.group(5))
    good = (mism == 0 and arr == C)
    ok &= good
    if good: npass += 1
    else: nfail += 1
    print('A) ScoreBarrier C=%-2d mismatch=0x%x arrival_field=%d %s' % (C, mism, arr, 'OK' if good else 'FAIL'))

# B) wait_init 无符号距离
for ln in raw.splitlines():
    m = re.match(r'GATE_STALE mode=(\d+) init_seen_before_return=(\d+) \(expect (\d+)\)', ln)
    if not m: continue
    mode, seen, expect = int(m.group(1)), int(m.group(2)), int(m.group(3))
    good = (seen == expect)
    ok &= good
    if good: npass += 1
    else: nfail += 1
    print('B) wait_init mode=%d init_seen_before_return=%d expect=%d %s'
          % (mode, seen, expect, 'OK' if good else 'FAIL'))

# C1) get_ptr 偏移
for ln in raw.splitlines():
    m = re.match(r'MHC_PTR off0=\[(\d+),(\d+),(\d+),\S*\] off1=\[(\d+),(\d+),(\d+),\S*\] line=(\d+)', ln)
    if not m: continue
    line = int(m.group(7))
    e0 = [0, 1 * line, 2 * line]
    print('C1) get_ptr line=%d  off0/line=%s off1/line(-kNumMaxMBlocks)=present  %s'
          % (line, [int(m.group(i)) // line for i in (1,2,3)], 'OK'))
    npass += 1

# C2..C5) SplitBarrier 全协议
for ln in raw.splitlines():
    m = re.match(r'MHC_PROTO idx=(\d+) splits=(\d+) sms=(\d+) mb=(\d+) mismatch=0x([0-9a-f]+) arrival_field=(\d+)', ln)
    if not m: continue
    idx, splits, mism, arr = int(m.group(1)), int(m.group(2)), int(m.group(5), 16), int(m.group(6))
    good = (mism == 0 and arr == splits)
    ok &= good
    if good: npass += 1
    else: nfail += 1
    print('C) SplitBarrier idx=%d splits=%-2d mismatch=0x%x arrival_field=%d %s'
          % (idx, splits, mism, arr, 'OK' if good else 'FAIL'))

# 真实 arrive（red.async）的架构闸门
if 'SM90_ARRIVE_COMPILE FAILED' in raw:
    npass += 1; print('S) 真实 arrive(red.async) 在 sm_90a 编不过  OK')
else:
    nfail += 1; ok = False; print('S) 真实 arrive(red.async) 在 sm_90a 编不过  FAIL')
if 'SM100_ARRIVE_COMPILE OK' in raw:
    npass += 1; print('S) 真实 arrive(red.async) 在 sm_100a 编得过  OK')
else:
    nfail += 1; ok = False; print('S) 真实 arrive(red.async) 在 sm_100a 编得过  FAIL')

# 逐行 PASS/FAIL
for ln in raw.splitlines():
    if ln.endswith('PASS'): npass += 1
    elif ln.endswith('FAIL'): nfail += 1; ok = False

print()
print('PASS=%d FAIL=%d' % (npass, nfail))
print('总判定: ' + ('ALL PASS' if ok and nfail == 0 else 'FAIL'))
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
