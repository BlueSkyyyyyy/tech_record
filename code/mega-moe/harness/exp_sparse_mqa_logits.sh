#!/usr/bin/env bash
# P12-1 实验 driver：DeepGEMM 稀疏 MQA logits（DeepSeek 稀疏注意力 indexer 第二级打分，
# 主 kernel SM100 专属）。在 kernel_lab(H100/sm90) 上做架构无关语义验证：
#   (A) metadata 尺寸/布局公式；(B) 稀疏选择 + 压缩列映射（用真实 SM90 dense kernel 取全量）；
#   (C) unaligned ks 偏移；(D) host 契约在 sm90 被拒。
#   bash harness/exp_sparse_mqa_logits.sh > harness/exp_sparse_mqa_logits.out.txt
# 对应源码 exp_sparse_mqa_logits.py；参考 ~/github/DeepGEMM 的
# csrc/apis/attention.hpp:284-442、csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp、
# deep_gemm/include/deep_gemm/{layout,scheduler,impls}/...sparse_mqa_logits*。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_sparse_mqa_logits.py"
