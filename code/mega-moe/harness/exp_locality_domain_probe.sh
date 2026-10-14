#!/usr/bin/env bash
# P24-1 实验 driver：SM 局部域（locality domain）探针与内存域分配。
#
# 为什么 H100(sm90) 可跑：`deep_gemm/include/deep_gemm/impls/sm100_locality_domain.cuh`
# 的 `locality_domain_probe_chase` 只依赖 `%smid` / `ld.global.cg` / `clock64`（架构无关）；
# host 侧 `csrc/apis/locality_domain.hpp` 的 probe/try_probe/balance/get_even 是纯张量逻辑。
# 本机 MLOPart 本地化内存不可用（需 MPS `-mlopart` + 双 device instance），故：
#   (A) 原样编译运行真实 device 探针量 HBM per-hop 延迟；
#   (B) 用 host-pinned mapped memory 当「远端域」验证探针能检出 far/near 并给出稳定划分；
#   (C) C++ 复刻 host 逻辑跑边界用例；
#   (D) 真机 DeepGEMM API：localization 不可用时 get_sm_locality_domains() 是否退化为 even。
#   bash harness/exp_locality_domain_probe.sh > harness/exp_locality_domain_probe.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/locality_domain_probe.cu"
DGEMM=${HOME}/github/DeepGEMM
PY="import torch, deep_gemm as dg; \
n=torch.cuda.get_device_properties(0).multi_processor_count; \
print('get_num_locality_domains  =', dg.locality_domain.get_num_locality_domains()); \
print('get_granularity           =', dg.locality_domain.get_granularity()); \
print('is_localization_available =', dg.is_localization_available()); \
d=dg.locality_domain.get_sm_locality_domains(); b=dg.locality_domain.get_balanced_sm_locality_domains(); \
e=torch.tensor([(i//2)%2 for i in range(n)], dtype=torch.uint8, device=d.device); \
print('num_sms =', n, ' get_even =', e.tolist()[:8]); \
print('sm_domains[:8] =', d[:8].tolist(), ' sum =', int(d.sum()), ' == get_even:', bool((d==e).all())); \
print('balanced[:8] =', b[:8].tolist(), ' sum =', int(b.sum()))"
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
  '$SRC' -o /tmp/locality_domain_probe \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/locality_domain_probe \
  && echo '' && echo '=== [E] real DeepGEMM API (python) ===' \
  && python -c \"$PY\""
