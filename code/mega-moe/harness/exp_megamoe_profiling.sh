#!/usr/bin/env bash
# P78-1 实验 driver：在 kernel_lab(H100/sm90) 上验证 DeepGEMM 自带的
#   Mega MoE profiling 基础设施 + 自动生成的「覆盖面清单」。
#   bash harness/exp_megamoe_profiling.sh > harness/exp_megamoe_profiling.out.txt
# 参考 ~/github/DeepGEMM（commit 057ca59, v2.8.1）：
#   scripts/run_ncu_mega_moe.sh、scripts/quick_plot_pm.py、tests/test_mega_moe.py
#   scripts/generate_pyi.py、stubs/_C.pyi、csrc/indexing/main.cu、CMakeLists.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DG=~/github/DeepGEMM

"$ROOT/scripts/lab.sh" exec "
set -e
GEN=/tmp/megamoe_prof
rm -rf \$GEN && mkdir -p \$GEN/work \$GEN/cap
DG=$DG
echo '== 真实源码 sha256 / 规模 =='
sha256sum \$DG/scripts/run_ncu_mega_moe.sh \$DG/scripts/quick_plot_pm.py \$DG/scripts/generate_pyi.py \$DG/stubs/_C.pyi \$DG/csrc/indexing/main.cu | sed 's#~/github/DeepGEMM/##'
wc -l \$DG/scripts/run_ncu_mega_moe.sh \$DG/scripts/quick_plot_pm.py \$DG/scripts/generate_pyi.py \$DG/tests/test_mega_moe.py \$DG/stubs/_C.pyi \$DG/csrc/indexing/main.cu
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1

echo
echo '== E1: run_ncu_mega_moe.sh 的 ncu 命令行构造（桩 ncu/python 捕获真实 argv）=='
mkdir -p \$GEN/stubbin
cat > \$GEN/stubbin/ncu <<'EOS'
#!/usr/bin/env bash
f=\$(mktemp \$CAP/ncu.XXXXXX); printf '%s\n' \"\$@\" > \$f
EOS
cat > \$GEN/stubbin/python <<'EOS'
#!/usr/bin/env bash
f=\$(mktemp \$CAP/python.XXXXXX); printf '%s\n' \"\$@\" > \$f
EOS
chmod +x \$GEN/stubbin/ncu \$GEN/stubbin/python
export CAP=\$GEN/cap
export PATH=\$GEN/stubbin:\$PATH
cd /tmp
bash \$DG/scripts/run_ncu_mega_moe.sh --num-processes 3 -o \$GEN/work --foo bar >\$GEN/run_ncu.stdout 2>&1
echo '--- 捕获到的 python 调用（第一行应为 JIT 预热）---'
for f in \$GEN/cap/python.*; do echo \"python \$(tr '\n' ' ' < \$f)\"; done
echo '--- 捕获到的 ncu 调用（每 rank 一条）---'
for f in \$GEN/cap/ncu.*; do echo \"ncu[\$(basename \$f)]: \$(tr '\n' ' ' < \$f)\"; done
echo '--- 输出目录 ---'
ls -la \$GEN/work | head -3
echo '--- run_ncu stdout ---'
cat \$GEN/run_ncu.stdout
" 2>&1

echo
echo "== E2 准备：真实 ncu --import --page raw --csv =="
"$ROOT/scripts/lab.sh" exec "
set -e
GEN=/tmp/megamoe_prof
cd /tmp
printf '%s\n' '#include <cstdio>' '__global__ void busy(float* p){ float s=0; for(int i=0;i<2000000;i++) s+=__sinf(i*1e-6f); if(threadIdx.x==0) p[0]=s; }' 'int main(){ float* p; cudaMalloc(&p,4); busy<<<1,32>>>(p); cudaDeviceSynchronize(); printf(\"done\n\"); return 0; }' > \$GEN/tiny.cu
nvcc -arch=sm_90 -O3 \$GEN/tiny.cu -o \$GEN/tiny
ncu --config-file off --set default --launch-count 1 --export \$GEN/tiny --force-overwrite \$GEN/tiny >/dev/null 2>&1
ncu --import \$GEN/tiny.ncu-rep --page raw --csv --metrics sm__maximum_warps_avg_per_active_cycle --print-metric-instances values > \$GEN/raw_present.csv 2>/dev/null
ncu --import \$GEN/tiny.ncu-rep --page raw --csv > \$GEN/raw_all.csv 2>/dev/null
echo 'raw_present.csv 第 3 行（真实取址来源）：'
sed -n '3p' \$GEN/raw_present.csv | cut -c1-200
echo \"raw_present.csv 列数: \$(sed -n '1p' \$GEN/raw_present.csv | python3 -c 'import sys,csv;[print(len(r)) for r in csv.reader(sys.stdin)]')\"
echo \"raw_all.csv   列数: \$(sed -n '1p' \$GEN/raw_all.csv | python3 -c 'import sys,csv;[print(len(r)) for r in csv.reader(sys.stdin)]')\"
" 2>&1

echo
echo "== E2/E3/E4：Python 判定（真实 parse / 真实 test 代码抽取 / 真实 pyi 生成）=="
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && MEGAMOE_GEN=/tmp/megamoe_prof python exp_megamoe_profiling.py" 2>&1
