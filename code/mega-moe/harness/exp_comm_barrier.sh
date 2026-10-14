#!/usr/bin/env bash
# P77-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行 DeepGEMM 的
#   **跨 SM / 跨 rank 同步原语** `deep_gemm/include/deep_gemm/comm/barrier.cuh`
#   + 设备端对称指针 `layout/sym_buffer.cuh`（真实设备代码，架构无关）。
#   ./harness/exp_comm_barrier.sh
# 对应源码 src/comm_barrier.cu，输出见 harness/exp_comm_barrier.out.txt。
#
# 覆盖：A) grid_sync 有序性 + kFinishSumTag 免复位（多组 SM 数，含奇数 N）
#       B) nvlink_barrier 状态机（单卡虚拟多 rank，phase/sign 交替 + ±1 自清零）
#       C) nvlink_barrier 真实多卡（UVA peer 对称缓冲，8 rank 并发 + 跨 rank 数据可见）
#       D) cluster_sync_with_relaxed_arrive（2-CTA cluster，弱序可见性）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
INC=${HOME}/github/DeepGEMM/deep_gemm/include
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/comm/barrier.cuh $INC/deep_gemm/layout/sym_buffer.cuh $INC/deep_gemm/layout/mega_moe.cuh
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 2389 \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" \
  '${HOME}/proj/tech_record/code/mega-moe/src/comm_barrier.cu' -o /tmp/comm_barrier
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
/tmp/comm_barrier > /tmp/comm_barrier.raw.txt 2>&1 || { echo "RUN FAILED"; cat /tmp/comm_barrier.raw.txt; exit 1; }
cat /tmp/comm_barrier.raw.txt
echo
echo '== Python 侧判定 =='
python3 - <<'PY'
import re
raw = open('/tmp/comm_barrier.raw.txt').read()
ok = True
TAG = 0x80000000

# A) grid_sync
seens = {}
for ln in raw.splitlines():
    m = re.match(r'GRIDSYNC G=(\d+) rounds=(\d+) mismatch=(\d+) counter_seq=([\d,]+)', ln)
    if not m: continue
    G, rounds, mism = int(m.group(1)), int(m.group(2)), int(m.group(3))
    seq = [int(x) for x in m.group(4).split(',')]
    # 期望：每轮 counter += 2^31 (mod 2^32) → 奇偶轮在 0 / 0x80000000 间翻转
    exp = [(r * TAG) & 0xffffffff for r in range(1, len(seq) + 1)]
    good = (mism == 0) and (seq == exp)
    if not good: ok = False
    print('A) grid_sync G=%-3d rounds=%-5d mismatch=%d counter_seq=%s (exp %s) %s'
          % (G, rounds, mism, seq, exp, 'OK' if good else 'FAIL'))
    seens[G] = seq

# B) 虚拟 nvlink
for ln in raw.splitlines():
    m = re.match(r'NVLVIrt N=(\d+) G=(\d+) rounds=(\d+) trace_bad=(\d+)', ln)
    if not m: continue
    bad = int(m.group(4))
    if bad != 0: ok = False
    print('B) nvlink_barrier(virtual) N=%s G=%s rounds=%s trace_bad=%d %s'
          % (m.group(1), m.group(2), m.group(3), bad, 'OK' if bad == 0 else 'FAIL'))
for ln in raw.splitlines():
    if ln.startswith('NVLVIrt_trace'):
        print('   ' + ln)

# D) cluster
for ln in raw.splitlines():
    m = re.match(r'CLUSTER dim=2 seen_peer=(\d+)/(\d+)', ln)
    if m:
        print('D) cluster_sync_with_relaxed_arrive: 对端写可见 %s/%s （relaxed：不保证，仅记录）'
              % (m.group(1), m.group(2)))

# C) 真实多卡
for ln in raw.splitlines():
    m = re.match(r'NVLRANK ranks=(\d+) G=(\d+) rounds=(\d+) mismatch=(\d+)', ln)
    if m:
        mm = int(m.group(4))
        if mm != 0: ok = False
        print('C) nvlink_barrier(real) ranks=%s G=%s rounds=%s mismatch=%d %s'
              % (m.group(1), m.group(2), m.group(3), mm, 'OK' if mm == 0 else 'FAIL'))

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
