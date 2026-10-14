#!/usr/bin/env bash
# P47-1 复现：**in-kernel 队列的「发布内存序」——不 release 到底会不会读错？release/acquire 值多少代价？**
# 补齐 P46-1（docs/04 §35.7）明留的尾巴：「入队是『先推 tail、后写数据』（无 release 发布），
# 理论上消费者可能在数据可见前读到槽；真实实现（MPK）用 release/acquire 把它做实」。
# 真实锚点：MPK 队列协议「relaxed 写数据 + release 推游标 / acquire 读游标 + relaxed 读数据」
#   （docs/06 §5.3，mpk_atoms.cuh:17-96；worker 入队 st.relaxed + CAS-release :1215-1228；
#    scheduler 出队 ld.acquire :1340 + ld.relaxed :1357-1360；派发 :1495-1502）。
# 机制：
#   §A message-passing litmus（P: data=1;flag=1 / Q: while(flag==0);r1=data）relaxed vs release+acquire；
#      正对照 store-buffering litmus（P: x=1;r1=y / Q: y=1;r2=x）验证弱内存重排可观测、检测器有效。
#   §B 单常驻 kernel 的 in-kernel 生产者→消费者环形队列（payload=PL 字、逐字校验），
#      wro（写 payload→推 tail，无 fence）vs rel（写→__threadfence→推 tail；读 tail→__threadfence→读）
#      vs pub（先推 tail 后写，= P46 §35 真实写法）vs pubslp（人为拉长发布→写窗口，检测器正对照）。
#   §C 每槽 release/acquire 的代价（小 payload 放大队列机制本身）。
# 目标卡 H100 / sm90，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/pub_order_ring.cu
BIN=/tmp/pub_order_ring
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 900 "$BIN" bench
echo
echo "===== 追加：litmus relaxed 连跑 3 次（稳定性） ====="
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" litmus 10000000 0 1 | sed 's/^/  /'; done
echo "===== 追加：litmus release/acquire 连跑 3 次（恒 0） ====="
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" litmus 10000000 1 1 | sed 's/^/  /'; done
