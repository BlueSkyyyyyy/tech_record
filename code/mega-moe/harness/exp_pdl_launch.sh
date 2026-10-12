#!/usr/bin/env bash
# P42-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 DeepGEMM/DeepJIT 的 **kernel 发射层**
#   —— PDL（Programmatic Dependent Launch）、cluster / non-portable cluster、dynamic shared memory，
#   以及 DeepJIT `LaunchOptions` 的 host 语义。
#   ./harness/exp_pdl_launch.sh
# 对应源码 src/pdl_launch.cu（设备侧 runtime API cudaLaunchKernelEx）与
#          src/pdl_launch_host.cpp（直接 include 真实 deep_jit options.hpp）。
# 输出见 harness/exp_pdl_launch.out.txt。
#
# 依据：DeepJIT 的 Kernel::launch 走 driver API cuLaunchKernelEx + CUlaunchAttribute
#   （third-party/deep_jit/include/deep_jit/backend/cuda/kernel.hpp:135-170）；
#   DeepGEMM 每个 kernel 入口都调用 cudaGridDependencySynchronize()（24 处）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DJ=~/github/DeepGEMM/third-party/deep_jit/include
MEGA=~/proj/tech_record/code/mega-moe

echo "== 真实头文件 sha256 =="
sha256sum $DJ/deep_jit/backend/cuda/options.hpp $DJ/deep_jit/backend/cuda/kernel.hpp \
          $DJ/deep_jit/runtime/runtime.hpp \
          ~/github/DeepGEMM/csrc/apis/config.hpp
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
nvcc --version | tail -1

echo
echo "== 编译 / 运行设备侧实验 =="
nvcc -arch=sm_90 -O3 -std=c++17 "$MEGA/src/pdl_launch.cu" -o /tmp/pdl_launch
/tmp/pdl_launch > /tmp/pdl_launch.raw.txt
cat /tmp/pdl_launch.raw.txt

echo
echo "== 编译 / 运行 LaunchOptions host 实验 =="
g++ -std=c++20 -O2 -I$DJ -I/usr/local/cuda/include "$MEGA/src/pdl_launch_host.cpp" \
    -o /tmp/pdl_launch_host -L/usr/local/cuda/lib64 -lcuda
/tmp/pdl_launch_host > /tmp/pdl_launch_host.raw.txt
cat /tmp/pdl_launch_host.raw.txt

echo
echo "== Python 侧判定 =="
python3 - <<'PY'
import re
dev = open('/tmp/pdl_launch.raw.txt').read()
hos = open('/tmp/pdl_launch_host.raw.txt').read()
L = dev.splitlines()
ok = True
def fail(m):
    global ok; ok = False; print('  !! ' + m)
def g(pat):
    m = [l for l in L if re.match(pat, l)]
    return m[-1] if m else None

# 设备信息
dm = re.search(r'DEVICE .* sm_count=(\d+)', dev)
sms = int(dm.group(1)) if dm else 0
print('设备 SM 数 =', sms)

# 1) 分量
c = g(r'PDL_COMPONENT')
if c:
    mm = re.search(r'producer_full_ms=([\d.]+) producer_half_ms=([\d.]+) pair_serial_full_ms=([\d.]+)', c)
    pf, ph, ps = map(float, mm.groups())
    print('分量: producer_full=%.3f producer_half=%.3f pair_serial_full=%.3f ms' % (pf, ph, ps))
    if not (ph > 0 and abs(pf - ph) < 0.2 * ph): fail('producer 半分/全分时间应相近')

# 2) 全格 PDL：收益有限但正确
fg = g(r'PDL full_grid')
if fg:
    mm = re.search(r'serial_ms=([\d.]+) overlap_ms=([\d.]+) speedup=([\d.]+)x fail=(\d+)', fg)
    ser, ov, sp, fl = float(mm.group(1)), float(mm.group(2)), float(mm.group(3)), int(mm.group(4))
    print('全格: serial=%.3f overlap=%.3f speedup=%.2fx fail=%d' % (ser, ov, sp, fl))
    if fl != 0: fail('全格 PDL 正确性 fail != 0')
    if not (0.95 <= sp <= 1.30): fail('全格 speedup 应接近 1（无空闲 SM），得到 %.2f' % sp)

# 3) 半格 PDL：prologue 并发
hg = g(r'PDL half_grid')
if hg:
    mm = re.search(r'serial_ms=([\d.]+) overlap_ms=([\d.]+) speedup=([\d.]+)x fail=(\d+)', hg)
    ser, ov, sp, fl = float(mm.group(1)), float(mm.group(2)), float(mm.group(3)), int(mm.group(4))
    print('半格: serial=%.3f overlap=%.3f speedup=%.2fx fail=%d' % (ser, ov, sp, fl))
    if fl != 0: fail('半格 PDL 正确性 fail != 0')
    if sp < 1.6: fail('半格 speedup 应显著 >1（空闲 SM 上并发），得到 %.2f' % sp)

# 4) 触发器位置：提前触发才重叠
tr = g(r'PDL trigger')
if tr:
    mm = re.search(r'early_ms=([\d.]+) implicit_end_ms=([\d.]+)', tr)
    e, i = float(mm.group(1)), float(mm.group(2))
    print('触发器: early=%.3f implicit_end=%.3f ms' % (e, i))
    if not (i > 1.6 * e): fail('隐式触发应≈串行（无重叠），得到 early=%.3f implicit=%.3f' % (e, i))

# 5) stale 演示：缺 griddepsync → stale；有 → 0
st = [l for l in L if l.startswith('PDL stale')]
if len(st) == 2:
    vals = {}
    for l in st:
        mm = re.search(r'grid_sync=(\d) fail=(\d+) / (\d+)', l)
        vals[int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3)))
    print('stale: grid_sync=1 fail=%d, grid_sync=0 fail=%d (out of %d)'
          % (vals[1][0], vals[0][0], vals[0][1]))
    if vals[1][0] != 0: fail('有 griddepsync 时不应有 stale')
    if vals[0][0] == 0: fail('缺 griddepsync 时应观测到 stale 读')

# 6) dynamic smem
dy = g(r'DYN maxDynamic')
d0 = g(r'DYN launch_without_attr')
d1 = g(r'DYN set_attr_err')
if dy and d0 and d1:
    print('dynsmem: ' + dy)
    print('        ' + d0)
    print('        ' + d1)
    if 'err=0' in d0 or 'invalid argument' not in d0:
        fail('未设 attribute 的 >48KiB dyn smem launch 应失败')
    mm = re.search(r'readback=(\d+) expect=(\d+)', d1)
    if mm and mm.group(1) != mm.group(2): fail('dyn smem readback 不符')
    if 'launch_err=0' not in d1: fail('设 attribute 后 dyn smem launch 应成功')

# 7) cluster
cl = [l for l in L if l.startswith('CLUSTER cs=')]
for l in cl:
    print(l)
    mm = re.search(r'cs=(\d+) grid=(\d+) launch_err=(\d+) bad=(\d+)', l)
    if int(mm.group(3)) != 0 or int(mm.group(4)) != 0: fail('cluster 发射/rank 不符: ' + l)
np = g(r'CLUSTER nonportable')
if np:
    print(np)
    if 'without_attr_err=0' in np:
        fail('16-CTA 非便携 cluster 未设 attribute 时不应成功')
    if 'with_attr_err=0' not in np:
        fail('设 nonportable attribute 后 16-CTA cluster 应成功')

# 8) host LaunchOptions
if 'HOST_JUDGE ALL PASS' not in hos: fail('LaunchOptions host 判定非 ALL PASS')
else: print('LaunchOptions host: ALL PASS')

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
