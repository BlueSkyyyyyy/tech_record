#!/usr/bin/env bash
# P21-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行 DeepGEMM 真实 epilogue 算子
#   （deep_gemm/include/deep_gemm/epilogue/operators.cuh:12-92 原样抽取）。
#   ./harness/exp_epilogue_operators.sh
# 对应源码 src/epilogue_operators.cu，输出见 harness/exp_epilogue_operators.out.txt。
#
# 抽取原因：operators.cuh 与 ld_st.cuh 里的 SM100 硬件原语（cvt.rs / __fmul2_rn）在 sm90 上
# 不可编译/不可用，故本 harness 逐行抽取四个 struct 到生成文件，并在 src 里补上明确标注的等价模拟。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/epilogue_operators.cu"
INC="~/github/DeepGEMM/deep_gemm/include"
OPC="$INC/deep_gemm/epilogue/operators.cuh"

CMD=$(cat <<'EOS'
set -e
INC=~/github/DeepGEMM/deep_gemm/include
OPC=$INC/deep_gemm/epilogue/operators.cuh
GEN=/tmp/epilogue_operators.inc
{
  echo 'namespace deep_gemm { namespace epilogue { namespace operators {'
  sed -n '12,92p' "$OPC"
  echo '} } }'
} > "$GEN"
echo "extracted structs sha256: $(sha256sum "$GEN" | cut -d' ' -f1) ($OPC:12-92)"
echo "cvt.rs on sm_90 (expected to fail):"
cat > /tmp/cvtrs_probe.cu <<'EOF'
#include <cstdint>
__global__ void k(uint32_t* o){ uint32_t p; float a=1.f,b=2.f; uint32_t r=0x1234u;
  asm volatile("cvt.rs.bf16x2.f32 %0, %1, %2, %3;\n":"=r"(p):"f"(a),"f"(b),"r"(r)); o[0]=p; }
EOF
nvcc -arch=sm_90 -O3 -std=c++17 -c /tmp/cvtrs_probe.cu -o /tmp/cvtrs_probe.o 2>&1 | grep -m1 "error" || true
echo
EOS
)

CMD="$CMD
nvcc -arch=sm_90 -O3 -std=c++17 \
  -I'$INC' -I'$INC/cutlass' -I'$INC/cute' -I/tmp \
  '$SRC' -o /tmp/epilogue_operators && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && /tmp/epilogue_operators
echo
echo '== E) QuantizeToFP8 的 D/SFD 与 torch 参考逐位对拍 =='
python3 - <<'PY'
import struct, torch
with open('/tmp/epilogue_quant.bin','rb') as f:
    M,N,stride,nacc = struct.unpack('<4I', f.read(16))
    acc = torch.frombuffer(bytearray(f.read(nacc*4)), dtype=torch.float32).reshape(M,N).clone()
    D = torch.frombuffer(bytearray(f.read(nacc)), dtype=torch.uint8).reshape(M,N).clone()
    sfd_words = torch.frombuffer(bytearray(f.read()), dtype=torch.uint32).clone()

# 参考：与 store_cd.cuh:144-174 同序 —— bf16 round -> per-32 amax -> UE8M0 exp -> scale -> E4M3
def ue8m0_sf_exp(amax_f32):
    b = amax_f32.view(torch.int32).to(torch.int64) & 0xffffffff
    rounded = (b + 0x7fffff - (0x60 << 16)) >> 23      # kMantissaMask=0x7fffff, kQuantMaxMantissa=0x60<<16
    return torch.clamp(rounded, 105 + 8) - 8           # kMinSFExponent+maxExp .. -maxExp

bf = acc.to(torch.bfloat16)
grp = bf.reshape(M, N//32, 32)
amax = grp.abs().amax(dim=2).to(torch.float32)          # get_packed_bf16_amax 的树形归约 = 组内 abs max
exp = ue8m0_sf_exp(amax)
sf_inv = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), (127 - exp))  # get_ue8m0_sf_inv
D_ref = (bf.reshape(M, N//32, 32).float() * sf_inv[..., None]).to(torch.float8_e4m3fn)
D_ref = D_ref.reshape(M, N).view(torch.uint8)

mism_D = int((D.view(torch.uint8) != D_ref).sum())
print(f'  D: E4M3 [M={M},N={N}]  bytewise_mismatch={mism_D}/{M*N}')

# SFD：按 store_sf 寻址（sfd + (sf_idx/4)*stride + row_idx，字节 sf_idx%4）解出 SF 字节
sf_bytes = torch.zeros(M, N//32, dtype=torch.int32)
for row in range(M):
    for g in range(N//32):
        sf_idx = g
        word = int(sfd_words[(sf_idx//4)*stride + row])
        sf_bytes[row, g] = (word >> (8*(sf_idx%4))) & 0xff
mism_sf = int((sf_bytes != exp).sum())
print(f'  SFD: UE8M0 字节 [M={M},N/32={N//32}]  mismatch={mism_sf}/{M*(N//32)}')
print(f'  极端行(×100) SF exp={int(exp[0,0])}；全零组 SF exp={int(exp[1,0])}（=最小指数 {105+8-8}）')
print('  结论：' + ('ALL PASS' if mism_D==0 and mism_sf==0 else 'FAIL'))
PY"

"$ROOT/scripts/lab.sh" exec "$CMD"
