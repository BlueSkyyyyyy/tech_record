#!/usr/bin/env bash
# P154-1 实验 driver：DeepGEMM SM90 TF32 WGMMA 的 RS（寄存器 A）路径。
#   bash harness/exp_tf32_rs_wgmma.sh > harness/exp_tf32_rs_wgmma.out.txt
# 对应源码 src/tf32_rs_wgmma.cu，参考 ~/github/DeepGEMM 的
#   deep_gemm/include/deep_gemm/mma/sm90.cuh（TF32MMARS/TF32MMASelector）
#   deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh（唯一消费者）
#   cute/atom/mma_traits_sm90_gmma.hpp（ALayout_64x8）
# 补齐 docs/49（P39-1）§10 明确列为未做的「TF32MMASelector RS 路径数值」。
#
# 为什么 H100(sm90a) 可跑：TF32MMARS/TF32MMASelector、ptx/wgmma.cuh 的 fence/commit/wait
#   全是 SM90 设备代码；本机 H100 直接编译运行真实 `wgmma...f32.tf32.tf32` RS 指令。
# 本脚本：
#   (1) 静态属性 dump（M/K/kNumAccum）；
#   (2) A 寄存器布局 dump + Python 独立复刻 + 与 hc_prenorm kernel 手写取址逐项比对；
#   (3) 真实 WGMMA RS 数值（A 在寄存器 / B 在 smem 128B swizzle），四组输入隔离 A/B 的 tf32 舍入；
#   (4) 编译期负样本：TF32MMASelector<N,false>（SS）应被 static_assert 拒绝。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
INC=${HOME}/github/DeepGEMM/deep_gemm/include
CUTE=${HOME}/github/DeepGEMM/third-party/cutlass/include
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/mma/sm90.cuh $INC/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh $CUTE/cute/atom/mma_traits_sm90_gmma.hpp
nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I"$INC" -I"$CUTE" -I"$CUTE/cutlass" -I"$CUTE/cute" \
  '${HOME}/proj/tech_record/code/mega-moe/src/tf32_rs_wgmma.cu' -o /tmp/tf32_rs_wgmma -lcuda
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/tf32_rs_wgmma > /tmp/tf32_rs_wgmma.raw.txt
cat /tmp/tf32_rs_wgmma.raw.txt
echo
echo '== Python 独立复刻 + 判定 =='
python3 - <<'PY'
import re, struct
import numpy as np
raw = open('/tmp/tf32_rs_wgmma.raw.txt').read().splitlines()
ok = True; fails = []
def check(cond, msg):
    global ok
    if not cond: ok = False; fails.append(msg)
    print(('  OK  ' if cond else '  FAIL') + '  ' + msg)

# ---------- (1) 静态属性 ----------
S = {}
for ln in raw:
    m = re.match(r'S(\d+) (\d+)', ln)
    if m: S[int(m.group(1))] = int(m.group(2))
print('A) 选择器静态属性（TF32: M=64 K=8, kNumAccum=M*N/128=N/2）')
check(S[0]==64 and S[1]==8 and S[2]==4,   'TF32MMASelector<8>:   M=64 K=8 kNumAccum=4')
check(S[3]==64 and S[4]==64 and S[5]==8 and S[6]==32, 'TF32MMASelector<64>:  M=64 N=64 K=8 kNumAccum=32')
check(S[7]==64 and S[8]==128 and S[9]==8 and S[10]==64,'TF32MMASelector<128>: M=64 N=128 K=8 kNumAccum=64')
check(S[11]==128, 'TF32MMASelector<256>: kNumAccum=128')

# ---------- (2) A 寄存器布局 ----------
lay = {}
for ln in raw:
    m = re.match(r'L (\d+) (\d+) (\d+) (\d+)', ln)
    if m: lay[(int(m.group(1)), int(m.group(2)))] = (int(m.group(3)), int(m.group(4)))
def cute_formula(lane, reg):
    t0 = lane % 4; t1 = (lane//4) % 8; t2 = lane//32
    v0 = reg % 2; v1 = reg // 2
    return (t1 + 16*t2 + 8*v0, t0 + 4*v1)
def kernel_formula(lane, reg):
    # sm90_tf32_hc_prenorm_gemm.cuh:157-195：row = warp*16 + lane/4；regs 0..3 取
    # (row,eo),(row+8,eo),(row,eo+4),(row+8,eo+4)，eo=lane%4
    warp = lane//32; lane_in = lane%32; row = warp*16 + lane_in//4; eo = lane_in % 4
    return [(row,eo),(row+8,eo),(row,eo+4),(row+8,eo+4)][reg]
mism_c = sum(1 for k,v in lay.items() if v != cute_formula(*k))
mism_k = sum(1 for lane in range(128) for r in range(4) if lay[(lane,r)] != kernel_formula(lane,r))
check(len(lay)==512, '布局 dump 512 项 (128 lane x 4 reg)')
check(mism_c==0, 'device dump == cute ALayout_64x8 公式 (mismatch=%d)' % mism_c)
check(mism_k==0, 'device dump == hc_prenorm kernel 手写取址 (mismatch=%d)' % mism_k)
# 双射：每 (m,k) 恰好被一个 (lane,reg) 覆盖
cover = set(lay.values())
check(len(cover)==64*8, '布局对 64x8 双射 (覆盖 %d/512)' % len(cover))

# ---------- (3) 真实 WGMMA 数值 ----------
def rd(tag, which):
    path = '/tmp/tf32rs_%s_%s.bin' % (tag, which)
    return np.fromfile(path, dtype=np.float32)
def tf32_trunc(x):
    b = x.view(np.uint32)
    return (b & np.uint32(0xFFFFE000)).view(np.float32)
def tf32_rne(x):
    b = x.view(np.uint32).astype(np.uint64)
    lsb = (b >> np.uint64(13)) & np.uint64(1)
    rounded = b + np.uint64(0x0FFF) + lsb          # round-half-to-even
    return (rounded & np.uint64(0xFFFFE000)).astype(np.uint32).view(np.float32)

def case(tag, N, expect_a=None, expect_b=None):
    A = rd(tag,'A').reshape(64,32).astype(np.float64)
    B = rd(tag,'B').reshape(N,32).astype(np.float64)
    D = rd(tag,'D').reshape(64,N).astype(np.float64)
    Aq_t = tf32_trunc(rd(tag,'A')).reshape(64,32).astype(np.float64)
    Bq_t = tf32_trunc(rd(tag,'B')).reshape(N,32).astype(np.float64)
    Aq_r = tf32_rne(rd(tag,'A')).reshape(64,32).astype(np.float64)
    Bq_r = tf32_rne(rd(tag,'B')).reshape(N,32).astype(np.float64)
    ref_hi = A @ B.T
    ref_t  = Aq_t @ Bq_t.T
    ref_r  = Aq_r @ Bq_r.T
    d_hi = np.abs(D-ref_hi).max(); d_t = np.abs(D-ref_t).max(); d_r = np.abs(D-ref_r).max()
    print('  case %-11s N=%3d  maxdiff[full=%.3e trunc=%.3e rne=%.3e]  exact_trunc=%s exact_rne=%s'
          % (tag, N, d_hi, d_t, d_r, d_t==0.0, d_r==0.0))
    return d_t, d_r

print('C) 真实 WGMMA RS 数值（A 寄存器 / B smem 128B swizzle），对 fp64 参考')
# 整数小值：tf32 精确表示，应逐位==完整 GEMM
d_t,d_r = case('n64_int', 64);   check(d_t==0.0 and d_r==0.0, 'n64_int: 整数输入 exact (t=%.1e r=%.1e)'%(d_t,d_r))
d_t,d_r = case('n128_int',128);  check(d_t==0.0 and d_r==0.0, 'n128_int: 整数输入 exact')
# A 有小数、B 整数 → 隔离 A(寄存器) 的 tf32 舍入
d_t,d_r = case('n64_a_frac',64)
print('    n64_a_frac: A 寄存器舍入 -> rne更近=%s trunc更近=%s' % (d_r<d_t, d_t<d_r))
# A 整数、B 有小数 → 隔离 B(smem) 的 tf32 舍入
d_t,d_r = case('n64_b_frac',64)
print('    n64_b_frac: B smem 舍入   -> rne更近=%s trunc更近=%s' % (d_r<d_t, d_t<d_r))
d_t,d_r = case('n64_both',64)
d_t,d_r = case('n128_both',128)

# 单元素探针：D[0,n] 就是硬件对 B[n,0] 的 tf32 量化
def probe_check(tag, K):
    Bp = rd(tag,'B').reshape(64,K).astype(np.float32)
    Dp = rd(tag,'D').reshape(64,64).astype(np.float64)[0]
    tr = tf32_trunc(Bp[:,0]).astype(np.float64)
    rn = tf32_rne(Bp[:,0]).astype(np.float64)
    print('  --- 单元素探针 %s（BLOCK_K=%d，A 是 delta，隔离 B 的 tf32 量化）---' % (tag, K))
    for n in range(8):
        print('    n=%d  raw=0x%08x  D=%.8f  trunc=%.8f  rne=%.8f  -> %s'
              % (n, Bp[n,0].view(np.uint32), Dp[n], tr[n], rn[n],
                 'trunc' if abs(Dp[n]-tr[n]) < abs(Dp[n]-rn[n]) else 'rne'))
    pt = int(np.sum(Dp[:8] == tr[:8])); pr = int(np.sum(Dp[:8] == rn[:8]))
    return pt, pr
pt, pr = probe_check('probe', 32)
check(pt == 8, 'RS 探针 K=32：8 个边界值全部 == 截断 (trunc=%d/8 rne=%d/8)' % (pt, pr))

# (4) TMA tf32 转换探针：同一批 fp32 字节，FLOAT32 vs TFLOAT32 tensor map
print('D) TMA tensor map dtype 探针（同一 fp32 源，no-swizzle 载入 smem）')
for tag in ('f32', 'tf32'):
    src = rd('tma_' + tag, 'src'); dst = rd('tma_' + tag, 'dst')
    n = 8
    tr = tf32_trunc(src).astype(np.float64); rn = tf32_rne(src).astype(np.float64)
    dd = dst.astype(np.float64)
    same_src = int(np.sum(dd[:n] == src[:n]))
    same_tr = int(np.sum(dd[:n] == tr[:n])); same_rn = int(np.sum(dd[:n] == rn[:n]))
    print('  dtype=%-8s  smem==raw %d/%d  ==trunc %d/%d  ==rne %d/%d' % (tag, same_src, n, same_tr, n, same_rn, n))
    if tag == 'f32':
        check(same_src == n, 'TMA(FLOAT32): smem 与原字节逐位相同 (%d/%d)' % (same_src, n))
    else:
        check(same_rn == n, 'TMA(TFLOAT32): smem == RNE-to-tf32，**TMA 做了 fp32→tf32 就近舍入** (%d/%d)' % (same_rn, n))

# (5) DeepGEMM 端到端 delta 探针：tf32_hc_prenorm_gemm 的 B 是否被舍入（只有装了 deep_gemm 才跑）
print('E) DeepGEMM tf32_hc_prenorm_gemm 端到端 delta 探针（可选）')
try:
    import torch, deep_gemm, struct
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = 'cuda'
    probes = [0x3F801FFF, 0x3F803000, 0x3F805000, 0x3F807000, 0xBF801FFF, 0x3FBFFFFF, 0x40001234, 0x40402345]
    aa = torch.zeros(64, 64, dtype=torch.float32, device=dev).to(torch.bfloat16); aa[0, 0] = 1.0
    bb = torch.zeros(8, 64, dtype=torch.float32, device=dev)
    for i, u in enumerate(probes): bb[i, 0] = struct.unpack('<f', struct.pack('<I', u))[0]
    dd = torch.empty(64, 8, dtype=torch.float32, device=dev); ss = torch.empty(64, dtype=torch.float32, device=dev)
    deep_gemm.tf32_hc_prenorm_gemm(aa, bb, dd, ss)
    Dg = dd[0].cpu().numpy().astype(np.float64)
    Bg = bb[:, 0].cpu().numpy()
    trg = tf32_trunc(Bg).astype(np.float64); rng = tf32_rne(Bg).astype(np.float64)
    c = int(np.sum(Dg == rng))
    check(c == 8, 'deep_gemm B 端到端 == RNE-to-tf32 (%d/8)；TMA(TFLOAT32) 就是它的来源' % c)
except Exception as e:
    print('  (跳过：deep_gemm 不可用 %s)' % type(e).__name__)

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
if fails:
    print('失败项:'); [print('  - '+f) for f in fails]
PY

echo
echo '== (4) 编译期负样本：TF32MMASelector<N,false>（SS 形态）应 static_assert 拒绝 =='
cat > /tmp/tf32_ss_neg.cu <<'CU'
#include <deep_gemm/mma/sm90.cuh>
using namespace deep_gemm;
using T = mma::sm90::TF32MMASelector<64, false>::type;   // 期望：DG_STATIC_ASSERT(kUseRS) 触发
int main(){ return (int)sizeof(T); }
CU
if nvcc -gencode=arch=compute_90a,code=sm_90a -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
   -I"$INC" -I"$CUTE" -I"$CUTE/cutlass" -I"$CUTE/cute" -c /tmp/tf32_ss_neg.cu -o /tmp/tf32_ss_neg.o 2>/tmp/tf32_ss_neg.err; then
  echo "  FAIL  SS(N=64,false) 竟然编过"
else
  echo "  OK    SS(N=64,false) 被拒：$(grep -m1 -o 'SS mode is not supported.*' /tmp/tf32_ss_neg.err || head -1 /tmp/tf32_ss_neg.err)"
fi
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
