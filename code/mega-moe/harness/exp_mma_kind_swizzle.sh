#!/usr/bin/env bash
# P128-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 DeepGEMM 启发式共用的
#   「元素位宽 / TMA swizzle 模式」公共选择器 + GemmDesc 的 dtype→MmaKind 派生管道 +
#   Python 公开布局别名层。
#   - csrc/jit_kernels/heuristics/utils.hpp : get_num_element_bits / get_swizzle_mode
#   - csrc/jit_kernels/heuristics/config.hpp: GemmDesc 派生量 + check_validity
#   - csrc/utils/math.hpp                   : kPackedFP4 / get_tma_aligned_size
#   - deep_gemm/utils/layout.py             : 公开布局别名层
#   ./harness/exp_mma_kind_swizzle.sh
# 对应源码 src/mma_kind_swizzle.cu；输出见 harness/exp_mma_kind_swizzle.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=${HOME}/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=${HOME}/proj/tech_record/code/mega-moe/src/mma_kind_swizzle.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DG/csrc/jit_kernels/heuristics/utils.hpp \
          $DG/csrc/jit_kernels/heuristics/config.hpp \
          $DG/deep_gemm/utils/layout.py | awk '{n=split($2,a,"/"); print $1, a[n]}'
nvcc --version | tail -1

echo
echo "== 编译（nvcc 直接 include 真实头文件 + torch）=="
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DG/csrc -I$DG -I$DG/third-party/deep_jit/include -I$DG/third-party/cutlass/include \
  -I/usr/local/cuda/targets/x86_64-linux/include/cccl -I$DG/deep_gemm/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/mma_kind_swizzle \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcublas -lcublasLt -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验（C++ 真机 include 真实头文件）=="
/tmp/mma_kind_swizzle

echo
echo "== Python 公开布局别名层（deep_gemm/utils/layout.py）=="
python3 - <<'PY'
import ast
path = "${HOME}/github/DeepGEMM/deep_gemm/utils/layout.py"
tree = ast.parse(open(path).read())
imported, aliases = [], {}
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        for a in node.names:
            imported.append(a.name)
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        aliases[node.targets[0].id] = ast.unparse(node.value)
print("PY_IMPORTED", imported)
print("PY_ALIASES", aliases)
ok = True
def chk(c, m):
    global ok
    if not c: ok = False; print("  !! " + m)
chk(len(imported) == 7, "expected 7 _C imports, got %d" % len(imported))
chk("get_tma_aligned_size" in imported, "get_tma_aligned_size imported")
chk("get_mn_major_tma_aligned_packed_ue8m0_tensor" in imported, "packed ue8m0 imported")
# 两个别名都指向同一个 getter：m/k 共用同一把 mk_alignment 旋钮
chk(aliases.get("get_m_alignment_for_contiguous_layout") == "get_mk_alignment_for_contiguous_layout", "m alias")
chk(aliases.get("get_k_alignment_for_contiguous_layout") == "get_mk_alignment_for_contiguous_layout", "k alias")
print("PY_OK" if ok else "PY_FAIL")
PY

echo
echo "ALL DONE"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
