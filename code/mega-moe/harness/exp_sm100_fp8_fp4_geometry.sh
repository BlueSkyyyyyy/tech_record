#!/usr/bin/env bash
# P25-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行 DeepGEMM **SM100 FP8/FP4 稠密 GEMM
#   主 kernel** 里**架构无关的几何/描述符设备代码**（真实函数原样抽取/真实头文件）。
#   ./harness/exp_sm100_fp8_fp4_geometry.sh
# 对应源码 src/sm100_fp8_fp4_geometry.cu，输出见 harness/exp_sm100_fp8_fp4_geometry.out.txt。
#
# 抽取/验证的真实代码：
#   A) SF warp 内原地转置  sm100_fp8_fp4_gemm_1d1d.cuh:465-471  -> /tmp/utccp_body.inc
#   B) 描述符推进/位布局    deep_gemm/mma/sm100.cuh（真实头文件，非抽取）
#   C) 指令描述符           deep_gemm/mma/sm100.cuh + cutlass make_instr_desc_block_scaled（真实）
# SM100 主 kernel 本身在 sm90 不可编译（kernel:47 整段 __CUDA_ARCH__>=1000）；本实验只跑上述无关部分。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/sm100_fp8_fp4_geometry.cu"
INC="~/github/DeepGEMM/deep_gemm/include"
KER="$INC/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh"

CMD=$(cat <<'EOS'
set -e
INC=~/github/DeepGEMM/deep_gemm/include
KER=$INC/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh
# A) 抽取 SF warp 转置的 lambda 体（去掉签名与收尾 `};`），逐行原样
sed -n '465,471p' "$KER" > /tmp/utccp_body.inc
echo "extracted utccp body sha256: $(sha256sum /tmp/utccp_body.inc | cut -d' ' -f1) ($KER:465-471)"
echo "----- 抽取片段 -----"
cat /tmp/utccp_body.inc
echo "--------------------"
EOS
)

CMD="$CMD
nvcc -arch=sm_90 -O2 -std=c++17 --expt-relaxed-constexpr \
  -I'$INC' -I'$INC/cutlass' -I'$INC/cute' -I/tmp \
  '$SRC' -o /tmp/sm100_geom
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/sm100_geom > /tmp/sm100_geom.raw.txt
cat /tmp/sm100_geom.raw.txt
echo
echo '== Python 侧独立复刻 + 逐项对拍 =='
python3 - <<'PY'
import re

PASS = 'ALL PASS'
FAIL = 'FAIL'

raw = open('/tmp/sm100_geom.raw.txt').read()

# ---------- A) SF warp 转置 ----------
tr = {}
for ln in raw.splitlines():
    m = re.match(r'TR (\d+) (\d+)', ln)
    if m:
        tr[int(m.group(1))] = int(m.group(2))
def in_val(i):
    return (0xAAAA0000 ^ ((i * 2654435761) & 0xffffffff)) & 0xffffffff
# 设备：values[i]=in[i*32+lane] (i=0..3)，再写 out[lane*4+i]=values[i]
mism_tr = 0
for lane in range(32):
    for i in range(4):
        exp = in_val(i * 32 + lane)
        if tr[lane * 4 + i] != exp:
            mism_tr += 1
verdict_tr = PASS if mism_tr == 0 else FAIL
print('A) SF warp 转置: 128 值 mismatch=%d/128  (%s)  (out[lane*4+i] == in[i*32+lane])'
      % (mism_tr, verdict_tr))

# ---------- B/C) smem 描述符 ----------
# 位布局（cute SmemDescriptor）：start_address[0,14) leading[16,30) stride(hi)[0,14)
#   version(hi)[14,16) base_offset(hi)[17,20) lbo_mode(hi)[20] layout_type(hi)[29,32)
def decode(lo, hi):
    return dict(start=lo & 0x3FFF, lbo=(lo >> 16) & 0x3FFF,
                sbo=hi & 0x3FFF, version=(hi >> 14) & 0x3,
                base=(hi >> 17) & 0x7, lbo_mode=(hi >> 20) & 1,
                layout=(hi >> 29) & 0x7)
def atom_base(layout):
    return 32 if layout == 1 else 16   # SWIZZLE_128B_BASE32B=1 else 16
def inner_atom(block_mn, sw, sz):
    return block_mn if sw == 0 else sw // sz
# layout_type 枚举：NONE=0, 128B_BASE32B=1, 128B=2, 64B=4, 32B=6
LT = {0: 0, 16: 0, 32: 6, 64: 4, 128: 2}

cases = [
    dict(name='K-major  BM128 BK128 SW128 FP8', major='K',  BM=128, BK=128, SW=128, sz=1, pack=1),
    dict(name='MN-major BM128 BK128 SW128 FP8', major='MN', BM=128, BK=128, SW=128, sz=1, pack=1),
    dict(name='K-major  BM128 BK128 SW64  FP4', major='K',  BM=128, BK=128, SW=64,  sz=1, pack=2),
    dict(name='MN-major BM256 BK128 SW64  FP8', major='MN', BM=256, BK=128, SW=64,  sz=1, pack=1),
]
desc = {}
for ln in raw.splitlines():
    m = re.match(r'DESC (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)', ln)
    if m:
        i, lo, hi, a0, a1, a2, a32 = (int(x) for x in m.groups())
        desc[i] = dict(lo=lo, hi=hi, adv=[a0, a1, a2, a32])
ncase_desc = raw.split('DESC_NCASE')[1].split()[0]
print('B/C) smem 描述符: %d 例  (DESC_NCASE=%s)' % (len(desc), ncase_desc))
all_ok = mism_tr == 0
for i, c in enumerate(cases):
    d = desc[i]; f = decode(d['lo'], d['hi'])
    layout = LT[c['SW']]
    nb = 128 // atom_base(layout)
    if c['major'] == 'K':
        stride_k = 1
        sbo = nb * c['BK'] * c['sz'] // c['pack']
        lbo = 0
    else:
        block_mn_atom = inner_atom(c['BM'], c['SW'], c['sz'])
        sbo = nb * block_mn_atom * c['sz']
        lbo = c['BK'] * block_mn_atom * c['sz']
        stride_k = block_mn_atom
    exp_sbo, exp_lbo = sbo >> 4, lbo >> 4
    def adv_byte(offset, k):
        return ((offset + k * stride_k) * c['sz'] // c['pack']) >> 4
    ok = (f['layout'] == layout and f['sbo'] == exp_sbo and f['lbo'] == exp_lbo and
          d['adv'][0] == d['lo'] and d['adv'][1] == d['lo'] + adv_byte(0, 1) and
          d['adv'][2] == d['lo'] + adv_byte(0, 2) and d['adv'][3] == d['lo'] + adv_byte(32, 0))
    all_ok = all_ok and ok
    verdict = 'OK' if ok else 'MISMATCH'
    print('   [%d] %s: layout=%d(exp %d) SBO=%d(exp %d) LBO=%d(exp %d) version=%d '
          'advD(0,1)=%d(exp %d) advD(32,0)=%d(exp %d) %s'
          % (i, c['name'], f['layout'], layout, f['sbo'], exp_sbo, f['lbo'], exp_lbo, f['version'],
             d['adv'][1] - d['lo'], adv_byte(0, 1), d['adv'][3] - d['lo'], adv_byte(32, 0), verdict))

# ---------- D) 指令描述符 ----------
# InstrDescriptorBlockScaled 位布局：b_sf_id[4,6) a_format[7,10) b_format[10,13)
#   a_major[15] b_major[16] n_dim[17,23) scale_format[23] m_dim[24,29) a_sf_id[29,31) k_size[31]
# packed float_e2m1_t 走 MXF4Format 家族（E2M1=1），非 MXF8F6F4Format 的 E2M1=5
FMT = {'E4M3': 0, 'E2M1': 1}
icases = [
    dict(a='E4M3', b='E4M3', M=256, N=128, ma=0, mb=0, sfa=0, sfb=0),
    dict(a='E4M3', b='E4M3', M=256, N=128, ma=0, mb=0, sfa=1, sfb=2),
    dict(a='E4M3', b='E2M1', M=256, N=128, ma=0, mb=0, sfa=3, sfb=3),
    dict(a='E4M3', b='E4M3', M=128, N=64,  ma=1, mb=0, sfa=2, sfb=1),
    dict(a='E2M1', b='E2M1', M=128, N=128, ma=0, mb=0, sfa=3, sfb=0),
]
instr = {}
for ln in raw.splitlines():
    m = re.match(r'INSTR (\d+) (\d+) (\d+) (\d+) (\d+)', ln)
    if m:
        i, hi, ids, lo, nd = (int(x) for x in m.groups())
        instr[i] = dict(hi=hi, ids=ids, lo=lo, nd=nd)
ncase_instr = raw.split('INSTR_NCASE')[1].split()[0]
print('D) 指令描述符: %d 例  (INSTR_NCASE=%s)' % (len(instr), ncase_instr))
for i, c in enumerate(icases):
    d = instr[i]; hi = d['hi']
    a_sf = (hi >> 29) & 3; b_sf = (hi >> 4) & 3
    a_fmt = (hi >> 7) & 7;  b_fmt = (hi >> 10) & 7
    a_major = (hi >> 15) & 1; b_major = (hi >> 16) & 1
    n_dim = (hi >> 17) & 0x3F; scale = (hi >> 23) & 1
    m_dim = (hi >> 24) & 0x1F; k_size = (hi >> 31) & 1
    ok = (a_sf == c['sfa'] and b_sf == c['sfb'] and a_fmt == FMT[c['a']] and b_fmt == FMT[c['b']] and
          a_major == c['ma'] and b_major == c['mb'] and n_dim == (c['N'] >> 3) and scale == 1 and
          m_dim == (c['M'] >> 4) and k_size == 0 and d['lo'] == 0 and d['nd'] == (96 >> 3))
    all_ok = all_ok and ok
    verdict = 'OK' if ok else 'MISMATCH'
    print('   [%d] a=%s(fmt%d) b=%s(fmt%d) M=%d N=%d: scale_format=%d n_dim=%d(exp %d) '
          'm_dim=%d(exp %d) a_sf_id=%d(exp %d) b_sf_id=%d(exp %d) lo32=%d n_dim@umma_n=96=%d(exp 12) %s'
          % (i, c['a'], a_fmt, c['b'], b_fmt, c['M'], c['N'], scale, n_dim, c['N'] >> 3,
             m_dim, c['M'] >> 4, a_sf, c['sfa'], b_sf, c['sfb'], d['lo'], d['nd'], verdict))

print()
print('总判定: ' + (PASS if all_ok else FAIL))
PY"

"$ROOT/scripts/lab.sh" exec "$CMD"
