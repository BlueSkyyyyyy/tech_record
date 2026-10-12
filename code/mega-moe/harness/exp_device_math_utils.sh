#!/usr/bin/env bash
# P44-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行 DeepGEMM **设备端数学与工具层**
#   （`deep_gemm/include/deep_gemm/common/{math,utils,types,packing}.cuh` 的真实设备代码）。
#   bash harness/exp_device_math_utils.sh > harness/exp_device_math_utils.out.txt
# 对应源码 src/device_math_utils.cu；参考 ~/github/DeepGEMM 的 common/math.cuh 与
# deep_gemm/utils/math.py 的 ceil_to_ue8m0 参考量化器。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INC="~/github/DeepGEMM/deep_gemm/include"

CMD=$(cat <<EOS
set -e
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/common/math.cuh $INC/deep_gemm/common/utils.cuh $INC/deep_gemm/common/types.cuh $INC/deep_gemm/common/packing.cuh
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -I"$INC" -I"$INC/cutlass" -I"$INC/cute" '$ROOT/src/device_math_utils.cu' -o /tmp/device_math_utils
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/device_math_utils
echo
echo '== Python 侧独立复刻 + 判定 =='
cd '$ROOT/harness'
python3 - <<'PY'
import numpy as np, torch, math
import warnings; warnings.filterwarnings('ignore')
from deep_gemm.utils.math import ceil_to_ue8m0

def rd(p, dtype):
    return np.fromfile('/tmp/'+p, dtype=dtype)

ok = True

# ---------------- A) UE8M0 SF 指数 ----------------
amax = rd('dmutils_amax.bin', np.float32)
sf8e = rd('dmutils_sf8e.bin', np.uint32)
sf4e = rd('dmutils_sf4e.bin', np.uint32)
sf8i = rd('dmutils_sf8i.bin', np.float32)
sf4i = rd('dmutils_sf4i.bin', np.float32)
p8 = rd('dmutils_p8.bin', np.float32)
p4 = rd('dmutils_p4.bin', np.float32)
at = torch.from_numpy(amax)
# reference from the real python quantizer
sf8 = ceil_to_ue8m0(torch.clamp(at, min=1e-4) / 448.0)
ref8 = (sf8.view(torch.int32) >> 23).to(torch.int64).numpy().astype(np.uint32)
sf4 = ceil_to_ue8m0(torch.clamp(at, min=6.0 * 2**-126) / 6.0)
ref4 = (sf4.view(torch.int32) >> 23).to(torch.int64).numpy().astype(np.uint32)
m8 = int((sf8e != ref8).sum()); m4 = int((sf4e != ref4).sum())
if m8 or m4: ok = False
print('A1) get_ue8m0_sf_exp(E4M3, fp32) vs ceil_to_ue8m0  mismatch=%d/%d' % (m8, len(amax)))
print('A2) get_ue8m0_sf_exp(E2M1, fp32) vs ceil_to_ue8m0  mismatch=%d/%d' % (m4, len(amax)))
# bf16 input
abf = torch.frombuffer(open('/tmp/dmutils_amax_bf16.bin','rb').read(), dtype=torch.bfloat16)
sf8eb = rd('dmutils_sf8eb.bin', np.uint32); sf4eb = rd('dmutils_sf4eb.bin', np.uint32)
abf_f = abf.float()
ref8b = (ceil_to_ue8m0(torch.clamp(abf_f, min=1e-4)/448.0).view(torch.int32)>>23).to(torch.int64).numpy().astype(np.uint32)
ref4b = (ceil_to_ue8m0(torch.clamp(abf_f, min=6.0*2**-126)/6.0).view(torch.int32)>>23).to(torch.int64).numpy().astype(np.uint32)
mb8 = int((sf8eb != ref8b).sum()); mb4 = int((sf4eb != ref4b).sum())
if mb8 or mb4: ok = False
print('A3) get_ue8m0_sf_exp(E4M3, bf16)                mismatch=%d/%d' % (mb8, len(abf)))
print('A4) get_ue8m0_sf_exp(E2M1, bf16)                mismatch=%d/%d' % (mb4, len(abf)))
prod_ok = bool(np.all(p8 == 1.0) and np.all(p4 == 1.0))
if not prod_ok: ok = False
print('A5) SF * SF_inv == 1.0 exactly: E4M3 %s / E2M1 %s (min exp8=%d max exp8=%d min exp4=%d max exp4=%d)'
      % (bool(np.all(p8==1.0)), bool(np.all(p4==1.0)), sf8e.min(), sf8e.max(), sf4e.min(), sf4e.max()))
# exact power-of-two amax -> SF == amax/quant_max rounded up; spot check a few
print('   samples (amax, exp8, exp4):', [(float(amax[i]), int(sf8e[i]), int(sf4e[i])) for i in (0,1,2,3)])

# ---------------- B) packed bf16 amax ----------------
kPack = 8
bx = torch.frombuffer(open('/tmp/dmutils_bx.bin','rb').read(), dtype=torch.bfloat16).view(-1, kPack, 2)
bamax = torch.frombuffer(open('/tmp/dmutils_bamax.bin','rb').read(), dtype=torch.bfloat16)
ref_amax = bx.abs().amax(dim=(1, 2))
mb = int((bamax != ref_amax).sum())
if mb: ok = False
print('B) get_packed_bf16_amax<%d> vs abs().amax(): mismatch=%d/%d' % (kPack, mb, bamax.numel()))

# ---------------- C) cast_into_bf16_and_pack ----------------
cx = rd('dmutils_cx.bin', np.float32).reshape(-1, 2)
cout = rd('dmutils_cout.bin', np.int32)
ct = torch.from_numpy(cx).to(torch.bfloat16)
bits = ct.view(torch.int16).to(torch.int32).numpy()
ref_pack = ((bits[:,0] & 0xffff) | (bits[:,1] << 16)).astype(np.uint32)
mc = int((cout.view(np.uint32) != ref_pack).sum())
if mc: ok = False
print('C) cast_into_bf16_and_pack vs torch.bfloat16(RNE): mismatch=%d/%d' % (mc, cout.size))

# ---------------- D) scale_bf16x2_into_fp8x4 ----------------
lo = torch.frombuffer(open('/tmp/dmutils_lo.bin','rb').read(), dtype=torch.bfloat16).view(-1,2).float()
up = torch.frombuffer(open('/tmp/dmutils_up.bin','rb').read(), dtype=torch.bfloat16).view(-1,2).float()
se = rd('dmutils_sfexp_s.bin', np.uint32).astype(np.int64)
sout = rd('dmutils_sout.bin', np.uint32)
inv = torch.from_numpy(np.ldexp(1.0, se.astype(np.int32))).float()
vals = torch.cat([lo, up], dim=1) * inv[:, None]
ref_fp8 = vals.to(torch.float8_e4m3fn).view(torch.uint8).numpy().astype(np.uint32)
ref_pack_s = (ref_fp8[:,0] | (ref_fp8[:,1] << 8) | (ref_fp8[:,2] << 16) | (ref_fp8[:,3] << 24)).astype(np.uint32)
ms = int((sout != ref_pack_s).sum())
if ms: ok = False
print('D) scale_bf16x2_into_fp8x4 vs fp32*2^e -> E4M3(RNE): mismatch=%d/%d' % (ms, sout.size))

# ---------------- E) fma2 / fast_rcp ----------------
fa = rd('dmutils_fa.bin', np.float32).reshape(-1,2); fb = rd('dmutils_fb.bin', np.float32).reshape(-1,2)
fc = rd('dmutils_fc.bin', np.float32).reshape(-1,2); fo = rd('dmutils_fout.bin', np.float32).reshape(-1,2)
ref_fma = (fa.astype(np.float64)*fb.astype(np.float64)+fc.astype(np.float64)).astype(np.float32)
me = int((fo != ref_fma).sum())
print('E1) fma2 vs fused fma (fp64 单次舍入): mismatch=%d/%d  max|diff|=%.3e'
      % (me, fo.size, float(np.abs(fo.astype(np.float64)-ref_fma.astype(np.float64)).max())))
rx = rd('dmutils_rx.bin', np.float32); ro = rd('dmutils_rout.bin', np.float32)
rel = np.abs(ro - 1.0/rx) / (1.0/np.abs(rx))
print('E2) fast_rcp(rcp.approx.ftz): max relerr=%.3e  mean=%.3e' % (float(rel.max()), float(rel.mean())))

# ---------------- F) scan ----------------
sin = rd('dmutils_scan_in.bin', np.uint32).astype(np.int64)
incl = rd('dmutils_incl.bin', np.uint32).astype(np.int64)
excl = rd('dmutils_excl.bin', np.uint32).astype(np.int64)
total = rd('dmutils_total.bin', np.uint32).astype(np.int64)
n = sin.size
excl_ref = np.concatenate([[0], np.cumsum(sin)[:-1]])
incl_ref = np.zeros(n, np.int64)
for s in range(0, n, 32):
    incl_ref[s:s+32] = np.cumsum(sin[s:s+32])
mf = int((incl != incl_ref).sum()); mx = int((excl != excl_ref).sum())
if mf or mx or total[0] != sin.sum(): ok = False
print('F) scan: cta_exclusive_sum mismatch=%d/%d  warp_inclusive_sum mismatch=%d/%d  total=%d(exp %d)'
      % (mx, n, mf, n, total[0], sin.sum()))

# ---------------- G) warp_reduce ----------------
base = rd('dmutils_wr_base.bin', np.float32).astype(np.float64)

def offsets(N, inter):
    o = []
    if inter:
        # kIntergroupReduce=true: offsets 1,2,4,... while N <= offset
        for t in (1,2,4,8,16):
            if N <= t: o.append(t)
    else:
        # kIntergroupReduce=false: offsets 2,4,... while N >= 2*offset
        for t in (2,4,8,16,32):
            if N >= t: o.append(t // 2)
    return o

def reach(lane, offs):
    S = {lane}; frontier = {lane}
    for off in offs:
        nf = set()
        for l in S:
            nf.add(l ^ off)
        S |= nf
    return S

for tag, N, inter in [('4f',4,False),('8f',8,False),('16f',16,False),('32f',32,False),
                      ('4t',4,True),('8t',8,True),('16t',16,True)]:
    s = rd('dmutils_wr_%s_sum.bin' % tag, np.float32).astype(np.float64)
    m = rd('dmutils_wr_%s_max.bin' % tag, np.float32).astype(np.float64)
    offs = offsets(N, inter)
    ref_s = np.zeros(32); ref_m = np.zeros(32)
    for lane in range(32):
        r = sorted(reach(lane, offs))
        ref_s[lane] = base[r].sum(); ref_m[lane] = base[r].max()
    ds = np.abs(s - ref_s).max() if len(offs) else 0.0
    dm = int((m != ref_m).sum())
    if dm: ok = False
    groups = {}
    for lane in range(32):
        groups.setdefault(tuple(sorted(reach(lane, offs))), []).append(lane)
    gsz = sorted(set(len(v) for v in groups.values()))
    print('G) warp_reduce<%d,%s> offsets=%s groups=%d sizes=%s  sum max|diff|=%.3e  max mismatch=%d/32'
          % (N, {False:'intra',True:'inter'}[inter], offs, len(groups), gsz, ds, dm))

# ---------------- H) utils / constexpr ----------------
tc = rd('dmutils_tmem_cols.bin', np.uint32)
tc_ref = [32,32,64,64,128,128,256,256,512,512]
if not np.array_equal(tc, np.array(tc_ref, np.uint32)): ok = False
print('H1) get_num_aligned_tmem_cols:', list(map(int,tc)), '(exp', tc_ref, ')')
vb = rd('dmutils_vec_bytes.bin', np.uint32)
if list(map(int, vb)) != [16,8,4,2,1]: ok = False
print('H2) Vectorized<16/8/4>::vec_t bytes + smem_pack_factor(E2M1/E4M3):', list(map(int,vb)), '(exp [16,8,4,2,1])')
gf = rd('dmutils_gemm_flags.bin', np.uint32)
if list(map(int, gf)) != [1,1]: ok = False
print('H3) is_m_grouped_contiguous / is_k_grouped_contiguous 判定:', list(map(int,gf)), '(exp [1,1])')
hc = rd('dmutils_host_checks.bin', np.uint32)
hc_ref = [8,1024,896,8,1024,12,17,2]
if list(map(int,hc)) != hc_ref: ok = False
print('H4) ceil_div/align/constexpr_* / kNumDeviceLocalityDomains:', list(map(int,hc)), '(exp', hc_ref, ')')

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
