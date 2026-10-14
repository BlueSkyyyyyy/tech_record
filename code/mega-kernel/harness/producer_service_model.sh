#!/usr/bin/env bash
# P75-1（§63）复现：把 §62 端到端骨架里的 **producer 写端单独抽出来**（无 worker、无背压、
#   无数据面），测「producer 服务成本 vs 服务的环数」——固定每个 producer 写的总批数 Btot，
#   扫每个 producer 连续服务的环数 wpp（chunk=Btot/wpp）。补掉 §62.10 明留的尾巴
#   「更细的 producer 服务模型（每环服务成本 vs 环数）留待后续」，把 §62.7 的『扇出』归因
#   钉死在 producer 侧（或证伪）。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/producer_service_model.cu
BIN=/tmp/producer_service_model
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §63 producer 服务模型：每环服务成本 vs 环数 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 900 "$BIN" bench
