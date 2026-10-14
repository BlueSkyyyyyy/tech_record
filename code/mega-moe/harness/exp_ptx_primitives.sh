#!/usr/bin/env bash
# P28-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行 DeepGEMM **PTX 原语底座**
#   （`deep_gemm/include/deep_gemm/ptx/{ld_st,utils}.cuh`、`common/ring_pipeline.cuh` 的真实设备代码）。
#   ./harness/exp_ptx_primitives.sh
# 对应源码 src/ptx_primitives.cu，输出见 harness/exp_ptx_primitives.out.txt。
#
# 覆盖：A) st.release.gpu/ld.acquire.gpu 生产者-消费者契约 vs 裸存弱版本
#       B) atom(返回旧值) vs red/red.release 求和          C) RingPipeline::advance 相位轨迹
#       D) cp.async.cg/ zfill / ca_zfill                    E) ld_evict_first 正确性 + 流式带宽
#       F) cvt_relu_bf16x2 / accumulate(_square) 回退路径。
# `ptx::` 原语全部是 __device__ 内联、架构无关，sm90 可编译可运行。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/ptx_primitives.cu"
INC="${HOME}/github/DeepGEMM/deep_gemm/include"

CMD=$(cat <<'EOS'
set -e
INC=${HOME}/github/DeepGEMM/deep_gemm/include
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/ptx/ld_st.cuh $INC/deep_gemm/ptx/utils.cuh $INC/deep_gemm/common/ring_pipeline.cuh
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" \
  '${HOME}/proj/tech_record/code/mega-moe/src/ptx_primitives.cu' -o /tmp/ptx_primitives
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/ptx_primitives > /tmp/ptx_prim.raw.txt
cat /tmp/ptx_prim.raw.txt
echo
echo '== Python 侧独立复刻 + 判定 =='
python3 - <<'PY'
import re
raw = open('/tmp/ptx_prim.raw.txt').read()
lines = raw.splitlines()
ok = True

# A) 内存序：release/acquire(mode=1) 必须 0 失败；弱版本(mode=2) 记录观测到的失败
mo = {}
for ln in lines:
    m = re.match(r'MO mode=(\d+) rounds=(\d+) slots=(\d+) per_slot=(\d+) fail_words=(\d+)', ln)
    if m:
        mo[int(m.group(1))] = int(m.group(5))
rl = 'OK' if mo.get(1, -1) == 0 else 'FAIL'
if mo.get(1, -1) != 0: ok = False
print('A) mem-order: acqrel fail_words=%s (%s)  weak volatile fail_words=%s'
      % (mo.get(1), rl, mo.get(2)))

# B) 原子
m = re.search(r'ATOMIC expected_sum=(\d+) atom_sum=(\d+) red_sum=(\d+) red_rel_sum=(\d+) atom_old_distinct=(\d+)/(\d+)', raw)
if m:
    exp, a, r, rr, dist, tot = (int(x) for x in m.groups())
    b_ok = (a == exp and r == exp and rr == exp and dist == tot)
    if not b_ok: ok = False
    print('B) atom/red: atom_sum=%d red_sum=%d red_rel_sum=%d (exp %d) atom_old_distinct=%d/%d (%s)'
          % (a, r, rr, exp, dist, tot, 'OK' if b_ok else 'FAIL'))

# C) RingPipeline
def expected(ns, steps):
    st, ph = 0, 0
    out = []
    for _ in range(steps):
        out.append((st, ph))
        nxt = st + 1
        if (ns & (ns - 1)) == 0:
            st = nxt % ns; ph ^= nxt // ns
        else:
            st = nxt
            if st >= ns:
                st -= ns; ph ^= 1
    return out
for ln in lines:
    m = re.match(r'RING ns=(\d+) steps=(\d+)(.*)', ln)
    if not m: continue
    ns, steps = int(m.group(1)), int(m.group(2))
    got = [tuple(int(x) for x in tok.split(',')) for tok in m.group(3).split()]
    exp = expected(ns, steps)
    same = got == exp
    if not same: ok = False
    print('C) RingPipeline ns=%d steps=%d: %s' % (ns, steps, 'OK' if same else 'FAIL'))

# D) cp.async
for ln in lines:
    m = re.match(r'CPASYNC (\S+) mismatch=(\d+)/(\d+)', ln)
    if m:
        mis, tot = int(m.group(2)), int(m.group(3))
        if mis != 0: ok = False
        print('D) cp.async %s: mismatch=%d/%d (%s)' % (m.group(1), mis, tot, 'OK' if mis == 0 else 'FAIL'))

# E) ld_evict_first：两条路径数值一致（都 XOR 同一 buffer），比较带宽
ev = {}
for ln in lines:
    m = re.match(r'EVICT mode=(\d+) bytes=(\d+) ms=([\d.]+) GBps=([\d.]+)', ln)
    if m:
        ev[int(m.group(1))] = (float(m.group(3)), float(m.group(4)))
if 0 in ev and 1 in ev:
    print('E) ld_evict_first: plain ms=%.3f/%.1f GBps  evict_first ms=%.3f/%.1f GBps'
          % (ev[0][0], ev[0][1], ev[1][0], ev[1][1]))

# F) warp 数值原语
m = re.search(r'BF16 cvt_relu pack mismatch=(\d+)/(\d+)', raw)
if m:
    mis, tot = int(m.group(1)), int(m.group(2))
    if mis != 0: ok = False
    print('F) cvt_relu_bf16x2 pack vs 独立 RNE 参考: mismatch=%d/%d (%s)' % (mis, tot, 'OK' if mis == 0 else 'FAIL'))
    sm = re.search(r'BF16 sample(.*)', raw)
    if sm: print('   samples:' + sm.group(1))

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
