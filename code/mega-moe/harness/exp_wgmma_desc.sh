#!/usr/bin/env bash
# P39-1 实验 driver：DeepGEMM 的 SM90 WGMMA 层（`mma/sm90.cuh` + `ptx/wgmma.cuh`）。
#   bash harness/exp_wgmma_desc.sh > harness/exp_wgmma_desc.out.txt
# 对应源码 src/wgmma_desc.cu，参考 ~/github/DeepGEMM 的
# deep_gemm/include/deep_gemm/mma/sm90.cuh 与 deep_gemm/include/deep_gemm/ptx/wgmma.cuh。
#
# 为什么 H100(sm90a) 可跑：GmmaDescriptor 构造、`*_Selector`、fence/commit/wait 全是 SM90 设备代码；
# 本机 H100 直接编译运行真实 WGMMA 指令。本脚本：
#   (1) 描述符位域：dump 真实 make_smem_desc/make_gmma_desc/advance_gmma_desc_lo，Python 独立复刻核对；
#   (2) 真实 WGMMA 数值：手工铺 TMA 128B swizzle smem，跑 BF16/FP8/MN-major/分批 commit 四种，
#       结果按 CLayout_64xN 的寄存器->(m,n) 映射落盘，Python 对参考 GEMM。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/wgmma_desc.cu"

CMD=$(cat <<'EOS'
set -e
INC=~/github/DeepGEMM/deep_gemm/include
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/mma/sm90.cuh $INC/deep_gemm/ptx/wgmma.cuh
nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" \
  '~/proj/tech_record/code/mega-moe/src/wgmma_desc.cu' -o /tmp/wgmma_desc
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/wgmma_desc > /tmp/wgmma_desc.raw.txt
cat /tmp/wgmma_desc.raw.txt
echo
echo '== Python 独立复刻 + 判定 =='
python3 - <<'PY'
import re, numpy as np
raw = open('/tmp/wgmma_desc.raw.txt').read()
lines = raw.splitlines()
ok = True
fails = []

def check(cond, msg):
    global ok
    if not cond:
        ok = False; fails.append(msg)
    print(('  OK  ' if cond else '  FAIL') + '  ' + msg)

# ---------- (1) 描述符位域 ----------
lang = {}
for ln in lines:
    m = re.match(r'D(\d+) 0x([0-9a-f]{16})', ln)
    if m: lang[int(m.group(1))] = int(m.group(2), 16)

def fld(v):
    return dict(start=v & 0x3FFF, lbo=(v >> 16) & 0x3FFF, sbo=(v >> 32) & 0x3FFF,
                base=(v >> 49) & 0x7, layout=(v >> 62) & 0x3)

print('A) 描述符位域（bit[0,14)=addr>>4, [16,30)=LBO>>4, [32,46)=SBO>>4, [49,52)=base_offset, [62,64)=layout）')
d0, d1, d2, d3, d4, d5 = (fld(lang[i]) for i in range(6))
d6, d7, d8, d13, d28 = (fld(lang[i]) for i in (6, 7, 8, 13, 28))
check(d0['layout'] == 0 and d0['lbo'] == 0 and d0['sbo'] == 0x40, 'D00 make_smem_desc(p,0)=INTERLEAVE LBO=0 SBO=1024')
check(d1['layout'] == 1 and d1['lbo'] == 0 and d1['sbo'] == 0x40, 'D01 make_smem_desc(p,1)=B128 LBO=0 SBO=1024')
check(d2['layout'] == 1 and d2['lbo'] == 0x123 and d2['sbo'] == 0x456 and d2['base'] == 0,
      'D02 显式 LBO=0x1230/SBO=0x4560 -> 位域 0x123/0x456，base_offset=0')
check(d3['layout'] == 3 and d3['lbo'] == 1 and d3['sbo'] == 7, 'D03 B32 LBO=16/SBO=112 -> 1/7')
check(d4['layout'] == 1 and d4['lbo'] == 0 and d4['sbo'] == 0x40 and d4['start'] == d1['start'],
      'D04 make_gmma_desc<K,BM=64,BK=64,sw128,bf16>: B128/LBO=0/SBO=8*64*2=1024，start 与 D01 同')
check(d5['start'] == d4['start'] + 0x80 and d5['lbo'] == 0 and d5['sbo'] == 0x40,
      'D05 mn_idx=16 -> start+=16*64*2>>4=0x80（MN=行方向步进）')
check(d6['layout'] == 1 and d6['lbo'] == 0x200 and d6['sbo'] == 0x40,
      'D06 make_gmma_desc<MN,BK=64,sw128>: LBO=64*64*2>>4=0x200，SBO=8*64*2>>4=0x40')
check(d7['layout'] == 0 and d7['lbo'] == 0x8 and d7['sbo'] == 0x40,
      'D07 MN-major sw=16（interleave）: atom=16/2=8 -> swap 后 LBO=128>>4=0x8，SBO=1024>>4=0x40')
check(d8['layout'] == 1 and d8['lbo'] == 0 and d8['sbo'] == 0x40,
      'D08 FP8 K-major BK=128: SBO=8*128*1>>4=0x40，B128')
lo = lambda v: v & 0xFFFFFFFF
check(lo(lang[9]) == lo(lang[4]) and lo(lang[10]) == lo(lang[4]) + 2 and lo(lang[11]) == lo(lang[4]) + 0x80,
      'D09/D10/D11 advance: k_idx=0 -> +0，k_idx=16(K-major stride=1) -> +32B>>4=+2，mn_idx=16 -> +0x80')
check(lo(lang[12]) == lo(lang[4]) + 0x200, 'D12 advance offset=64*64 elem -> 8192B>>4=+0x200')
check(lo(lang[13]) == lo(lang[6]) + 0x20, 'D13 MN-major advance k_idx=4: stride_k=64 -> 4*64*2>>4=+0x20')
check(lo(lang[28]) == lo(lang[4]), 'D28 advance k_idx=4(K-major): 8B 被 >>4 截断 -> +0')

print('B) 选择器静态属性（M/K/kNumAccum = M*N/128）')
check(lang[14] == 64 and lang[15] == 128 and lang[16] == 16 and lang[17] == 64, 'BF16MMA<128>: M=64 N=128 K=16 kNumAccum=64')
check(lang[18] == 64 and lang[19] == 32 and lang[20] == 64, 'FP8MMA<128>: M=64 K=32 kNumAccum=64')
check(lang[21] == 64 and lang[22] == 8 and lang[23] == 64, 'TF32MMA<128>: M=64 K=8 kNumAccum=64')
check(lang[24] == 1 and lang[25] == 64, 'get_gmma_desc_stride_k: K-major=1, MN-major=BLOCK_MN_ATOM=64')
check(lang[26] == 16 and lang[27] == 64, 'get_inner_block_atom_size: sw=32->16, sw=0->BLOCK_MN=64')

# ---------- (2) 真实 WGMMA 数值 ----------
print('C) 真实 WGMMA 数值（A@B^T，逐位对参考整数 GEMM）')
def chk(name, nb, k):
    A = np.fromfile('/tmp/wgmma_A_%s.bin' % name, dtype=np.float32).reshape(64, k)
    B = np.fromfile('/tmp/wgmma_B_%s.bin' % name, dtype=np.float32).reshape(nb, k)
    D = np.fromfile('/tmp/wgmma_D_%s.bin' % name, dtype=np.float32).reshape(64, nb)
    R = A @ B.T
    exact = bool((D == R).all())
    check(exact, '%s: exact=%s maxdiff=%.3e' % (name, exact, float(np.abs(D - R).max())))

chk('bf16', 128, 64)    # bf16 K-major swizzle128, BLOCK_N=128
chk('bf16b', 64, 64)    # bf16, BLOCK_N=64
chk('fp8', 128, 128)    # fp8 E4M3 K-major, BLOCK_N=128, BLOCK_K=128
chk('bf16s', 128, 64)   # bf16 分两批 commit + wait<1>
chk('mn', 128, 64)      # A 为 MN-major

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
if fails:
    print('失败项:'); [print('  - ' + f) for f in fails]
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
