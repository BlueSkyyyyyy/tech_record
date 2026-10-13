#!/usr/bin/env bash
# P88-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 DeepEP 的「编译期配置面 + DeepJIT 接入」
#   deep_ep/include/deep_ep/common/compiled.cuh   （编译期 ABI / 常量，架构无关）
#   csrc/runtime/jit.hpp                          （init_jit：Config 五参 + 8 条 nvcc flag）
#   csrc/utils/system.hpp                         （get_env）
#   ./harness/exp_deepep_jit_config.sh
# 对应源码：src/deepep_compiled_abi.cu（Part 1，nvcc）、harness/deepep_jit_config_probe.cpp（Part 2，g++）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
DE=~/github/DeepEP
DJ=$DG/third-party/deep_jit
SRC=~/proj/tech_record/code/mega-moe/src/deepep_compiled_abi.cu
PROBE=~/proj/tech_record/code/mega-moe/harness/deepep_jit_config_probe.cpp

echo "== 环境坐标 =="
( cd $DE && git -c safe.directory='*' log --oneline -1 )
( cd $DG && git -c safe.directory='*' log --oneline -1 )
( cd $DJ && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DE/deep_ep/include/deep_ep/common/compiled.cuh \
          $DE/csrc/runtime/jit.hpp \
          $DE/csrc/utils/system.hpp | awk '{n=split($2,a,"/"); print $1, a[n]}'
nvcc --version | tail -1
g++ --version | head -1

echo
echo "############ Part 1: compiled.cuh ABI（nvcc 直接 include 真实头文件）############"
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DE/deep_ep/include "$SRC" -o /tmp/deepep_compiled_abi_64 -lcudart
echo "> build (default EP_NUM_TOPK_IDX_BITS=64) OK"
/tmp/deepep_compiled_abi_64
echo
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu -DEP_NUM_TOPK_IDX_BITS=32 \
  -I$DE/deep_ep/include "$SRC" -o /tmp/deepep_compiled_abi_32 -lcudart
echo "> build (-DEP_NUM_TOPK_IDX_BITS=32) OK"
/tmp/deepep_compiled_abi_32

echo
echo "############ Part 2: get_env / Config / 缓存键（g++ host，deep_jit 真实头）############"
g++ -std=c++20 -O1 -w \
  -I$DJ/include -I$DE/deep_ep/include -I$DE \
  "$PROBE" -o /tmp/deepep_jit_config_probe
echo "> build OK"
/tmp/deepep_jit_config_probe

echo
echo "############ Part 3: 从 jit.hpp 源码交叉核对 flag 清单（python 解析真实文件）############"
python3 - "$DE/csrc/runtime/jit.hpp" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
# 抓所有 nvcc_flags.emplace_back("...") 的字符串字面量前缀（动态两条为 "-I" + ... / "-DEP_..." + ...）
args = re.findall(r'nvcc_flags\.emplace_back\("([^"]*)"', src)
# 动态两条以 -I / -DEP_ 开头，其余是常量
const = [a for a in args if not a.startswith(("-I", "-DEP_NUM_TOPK_IDX_BITS"))]
dynamic = [a for a in args if a.startswith(("-I", "-DEP_NUM_TOPK_IDX_BITS"))]
expected_const = [
    "--diag-suppress=39,161,174,177,186,940,3012",
    "--compiler-options=-Wno-deprecated-declarations,-Wno-abi",
    "-DNCCL_GIN_GDAKI_ENABLE=1",
    "-DNCCL_GIN_PROXY_ENABLE=0",
    "-DNCCL_GIN_GPI_ENABLE=0",
    "-DNCCL_GIN_EFA_GDA_ENABLE=0",
    "-DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1",
]
print("  提取到 emplace_back 字面量:", len(args), "条（常量", len(const), "/ 动态", len(dynamic), "）")
print("  constants:", const)
print("  dynamic chains:", dynamic)
assert len(args) == 9, f"expected 9 emplace_back, got {len(args)}"
assert const == expected_const, f"constants mismatch:\n  got {const}\n  exp {expected_const}"
assert any(a.startswith('-I') for a in dynamic), "missing -I<nccl>"
assert any(a.startswith('-DEP_NUM_TOPK_IDX_BITS=') for a in dynamic), "missing -DEP_NUM_TOPK_IDX_BITS"
# 条件 flag 必须由 EP_GIN_GDAKI_DEBUG 守卫
assert re.search(r'if \(get_env<int>\("EP_GIN_GDAKI_DEBUG", 0\)\)', src), "debug guard missing"
print("  [PASS] flag 清单与源码逐字一致；EP_GIN_GDAKI_DEBUG 为条件守卫")
print("  [PASS] 动态两条：-I<nccl_include_dir> 与 -DEP_NUM_TOPK_IDX_BITS=<bits>")
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
