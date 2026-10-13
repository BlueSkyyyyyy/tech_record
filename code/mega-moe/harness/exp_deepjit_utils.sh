#!/usr/bin/env bash
# P160-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 DeepJIT 的「支撑工具层」——
#   deep_jit/utils/{filesystem,command,str,uuid,json,gil,no_ref_ptr}.hpp
#   （hash/parser/env/lazy/exception 已分别在第 33/38/40/116 篇拆过，本篇补剩余这层）
#   bash harness/exp_deepjit_utils.sh > harness/exp_deepjit_utils.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DJ=~/github/DeepGEMM/third-party/deep_jit/include
SRC=~/proj/tech_record/code/mega-moe/src
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PBINC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYEXT=$(python3-config --extension-suffix)

echo "== DeepJIT 版本坐标 =="
( cd ~/github/DeepGEMM/third-party/deep_jit && git log --oneline -1 ) 2>/dev/null || true
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
g++ --version | head -1

echo
echo "== 被拆源码 sha256 =="
sha256sum $DJ/deep_jit/utils/filesystem.hpp $DJ/deep_jit/utils/command.hpp \
          $DJ/deep_jit/utils/str.hpp $DJ/deep_jit/utils/uuid.hpp \
          $DJ/deep_jit/utils/json.hpp $DJ/deep_jit/utils/gil.hpp \
          $DJ/deep_jit/utils/no_ref_ptr.hpp

echo
echo "== 消费点（谁 include / 调用它们）=="
grep -rn "write_file_sync\|fsync_dir\|safe_remove_all\|try_update_mtime\|normalize_path\|is_executable\|\bread(" \
  $DJ/deep_jit/cache/disk.hpp $DJ/deep_jit/backend/cuda/backend.hpp $DJ/deep_jit/backend/ascend/backend.hpp \
  $DJ/deep_jit/backend/cuda/options.hpp | sed 's#.*deep_jit/include/##'
echo "---"
grep -rn "call_external_command" $DJ/deep_jit/backend/cuda/backend.hpp $DJ/deep_jit/backend/ascend/backend.hpp | sed 's#.*deep_jit/include/##'
echo "---"
grep -rn "str::join" $DJ/deep_jit/backend/cuda/options.hpp $DJ/deep_jit/backend/ascend/options.hpp $DJ/deep_jit/backend/cuda/backend.hpp $DJ/deep_jit/backend/ascend/backend.hpp | sed 's#.*deep_jit/include/##'
echo "---"
grep -rn "get_uuid" $DJ/deep_jit/cache/disk.hpp | sed 's#.*deep_jit/include/##'
echo "---"
grep -rn "json::object_t\|\.dump()" $DJ/deep_jit/runtime/config.hpp $DJ/deep_jit/backend/cuda/backend.hpp $DJ/deep_jit/backend/cuda/options.hpp | sed 's#.*deep_jit/include/##'
echo "---"
grep -rn "GilScopedRelease" $DJ/deep_jit/runtime/runtime.hpp $DJ/deep_jit/backend/cuda/kernel.hpp $DJ/deep_jit/backend/cuda/backend.hpp | sed 's#.*deep_jit/include/##'
echo "---"
grep -rn "NoRefPtr" $DJ/deep_jit/backend/cuda/kernel.hpp | sed 's#.*deep_jit/include/##'

echo
echo "== 编译纯 host 程序（直接 include 真实头文件）=="
g++ -std=c++20 -O2 -g -I$DJ "$SRC/deepjit_utils.cpp" -o /tmp/deepjit_utils -ldw -ldl
echo "build OK"

echo
echo "== 运行（json/str/uuid/command/filesystem/no_ref_ptr）=="
/tmp/deepjit_utils | tee /tmp/deepjit_utils.raw.txt

echo
echo "== 编译 GilScopedRelease pybind 探针 =="
g++ -std=c++20 -O2 -shared -fPIC -I$DJ -I"$PBINC" -I"$PYINC" \
  "$SRC/deepjit_gil.cpp" -o /tmp/deepjit_gil$PYEXT -lpython3.12
echo "build OK"

echo
echo "== Python 侧判定 =="
python3 ~/proj/tech_record/code/mega-moe/harness/exp_deepjit_utils.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
