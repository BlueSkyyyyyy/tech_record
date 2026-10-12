#!/usr/bin/env bash
# P45-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行 DeepGEMM 的
#   per-warp `%clock64` 追踪探针（`deep_gemm/include/deep_gemm/common/probe.cuh` 的真实设备代码）。
#   ./harness/exp_probe_clock64.sh
# 对应源码 src/probe_clock64.cu，输出见 harness/exp_probe_clock64.out.txt。
#
# 覆盖：A) DG_PROBE_ENABLE 开关语义（关闭后编成空）        B) record/advance/payload/mark 记录流
#       C) mark() 用 %globaltimer(ns) 标定 SM 频率          D) is_valid_buffer 安全判据（对齐/偶数/4GiB 窗口）
#       E) advance 只加低 32 位（伪指针实证）                F) 探针开销（开/关墙时间 + 纯记录吞吐）
# 全部为架构无关的设备内联 asm（mov %clock64 / mov %globaltimer / st.global），sm90 可编译可运行。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
INC=~/github/DeepGEMM/deep_gemm/include
SRC=~/proj/tech_record/code/mega-moe/src/probe_clock64.cu
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/common/probe.cuh
echo
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -DDG_PROBE_ENABLE=1 \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" "$SRC" -o /tmp/probe_on 2>&1 | grep -i error || true
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" "$SRC" -o /tmp/probe_off 2>&1 | grep -i error || true
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
echo
echo "########## DG_PROBE_ENABLE=1 ##########"
/tmp/probe_on on
echo
echo "########## DG_PROBE_ENABLE=0 (默认) ##########"
/tmp/probe_off off
echo
echo '== Python 侧解析记录流 + 判定 =='
python3 - <<'PY'
import numpy as np
ok = True

def rd_u64(p):
    return np.fromfile(p, dtype=np.uint64)

# ---- A) 开关 + 常量 ----
# (由二进制 stdout 打印，见下方汇总)

# ---- B) trace（DG_PROBE_ENABLE=1）----
tr = rd_u64('/tmp/probe_on_trace.bin')
assert tr.size == 64, tr.size
mark0_c, mark0_g = int(tr[0]), int(tr[1])
r2, r3, r4, r5 = int(tr[2]), int(tr[3]), int(tr[4]), int(tr[5])
pl0, pl1 = int(tr[6]), int(tr[7])
r8 = int(tr[8]); mark1_c, mark1_g = int(tr[9]), int(tr[10])
# payload：{a,b,c,d} 两条 8B：低32=a 高32=b；低32=c 高32=d
def lo32(x): return x & 0xffffffff
def hi32(x): return (x >> 32) & 0xffffffff
payload_ok = (lo32(pl0), hi32(pl0), lo32(pl1), hi32(pl1)) == \
             (0x11111111, 0x22222222, 0x33333333, 0x44444444)
clocks = [mark0_c, r2, r3, r4, r5, r8, mark1_c]
mono = all(clocks[i] <= clocks[i+1] for i in range(len(clocks)-1))
# 回填槽（r4/r5）必须晚于 r3，且 r8 晚于 r5（中间有 nanosleep）
fill_ok = (r4 >= r3 and r5 >= r4)
gmono = mark1_g >= mark0_g
if not (payload_ok and mono and fill_ok and gmono): ok = False
print('B) trace: payload=%s clock_monotone=%s fill_ok=%s globaltimer_mono=%s' %
      (payload_ok, mono, fill_ok, gmono))
print('   clocks r0/r2/r3/r4/r5/r8/r9 = %s' % clocks)
print('   globaltimer mark0=%d mark1=%d (Δns=%d)' % (mark0_g, mark1_g, mark1_g - mark0_g))

# ---- B2) 关闭后 trace 全零 ----
tr_off = rd_u64('/tmp/probe_off_trace.bin')
off_zero = bool((tr_off == 0).all())
if not off_zero: ok = False
print('B2) DG_PROBE_ENABLE=0 trace 全零: %s' % off_zero)

# ---- C) 标定 ----
cal = rd_u64('/tmp/probe_on_calib.bin')
c0, c1, g0, g1 = (int(cal[i]) for i in range(4))
freq = (c1 - c0) / max(1, (g1 - g0))
# trace 里的 mark 也标定一次
freq_mark = (mark1_c - mark0_c) / max(1, (mark1_g - mark0_g))
print('C) calib: cycles=%d ns=%d -> %.4f GHz ; from marks %.4f GHz' %
      (c1 - c0, g1 - g0, freq, freq_mark))
c_ok = 1.0 < freq < 3.0 and 1.0 < freq_mark < 3.0
if not c_ok: ok = False
print('   频率落在 1~3 GHz: %s' % c_ok)

# ---- E) advance 只加低 32 位 ----
adv = rd_u64('/tmp/probe_on_advance.bin')
e0, e1, e2 = (int(adv[i]) for i in range(3))
e_ok = (e0 == 0x0000000100000000 and e1 == 0x0000000000000000 and e2 == 0x0000fff300000008)
adv_off = rd_u64('/tmp/probe_off_advance.bin')
off_adv_ok = bool((adv_off == 0).all())
if not (e_ok and off_adv_ok): ok = False
print('E) advance: [0x%016x] [0x%016x] [0x%016x] 期望 [0x0000000100000000] [0x0] [0x0000fff300000008] -> %s'
      % (e0, e1, e2, e_ok))
print('   关闭时 advance 为 no-op（cursor 保持 0）: %s' % off_adv_ok)

# ---- D) is_valid_buffer ----
def parse_valid(p):
    d = {}
    for ln in open(p):
        k, v = ln.split()
        d[k] = int(v)
    return d
v = parse_valid('/tmp/probe_on_valid.txt')
exp = dict(real_aligned=1, real_misaligned8=0, odd_records=0, crosses_4gib=0,
           near_but_inside=1, inside_window=1, multi_stream_stay=1,
           num_bytes_1=512, num_bytes_3=1536)
d_ok = (v == exp)
if not d_ok: ok = False
print('D) is_valid_buffer/get_num_bytes: %s' % d_ok)
for k in exp:
    print('   %-20s got=%-6s exp=%s' % (k, v.get(k), exp[k]))

# ---- F) 开销 ----
d = open('/tmp/probe_on_overhead.txt').read().split()
e = open('/tmp/probe_off_overhead.txt').read().split()
ms_on, iters, work = float(d[0]), int(d[1]), int(d[2])
ms_off = float(e[0])
pon, pw, pi = float(d[3]), int(d[4]), int(d[5])
poff = float(e[3])
per_rec = (ms_on - ms_off) / iters * 1e6  # ns/record（单线程，受 idle clock 影响）
per_rec_p = (pon - poff) / (pw * pi) * 1e6
print('F) overhead: fma loop on=%.4f off=%.4f ms -> +%.1f ns/record (单线程, idle clock)' %
      (ms_on, ms_off, per_rec))
print('   probe_only: on=%.4f off=%.4f ms (4warp x 4096) -> %.2f ns/record(含 4 warp 并发墙时间)' %
      (pon, poff, per_rec_p))

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
