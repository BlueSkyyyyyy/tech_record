#!/usr/bin/env bash
# 管理 kernel 优化系列专用的 GPU 实验容器 kernel_lab。
#
# 为什么需要单独的容器：宿主机的 ncu 权限是 RmProfilingAdminOnly=1，
# 且普通容器没有 CAP_SYS_ADMIN，ncu 会报 ERR_NVGPUCTRPERM。
# 这个容器加了 --cap-add SYS_ADMIN/SYS_PTRACE，可以在里面正常跑 ncu。
#
# 用法：
#   scripts/lab.sh up      # 创建/启动容器（幂等）
#   scripts/lab.sh enter   # 进入容器交互 shell
#   scripts/lab.sh exec "命令"
#   scripts/lab.sh status
#   scripts/lab.sh rm      # 删除容器
set -euo pipefail

IMAGE="${LAB_IMAGE:-dsv4-inf:latest}"
NAME="${LAB_NAME:-kernel_lab}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

up() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker start "$NAME" >/dev/null
  else
    docker run -d --name "$NAME" --restart unless-stopped \
      --gpus all --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --cap-add SYS_ADMIN --cap-add SYS_PTRACE \
      -v /ssd:/ssd -v /home/xieminglin:/home/xieminglin \
      -w "$REPO/code/kernel-opt" \
      "$IMAGE" sleep infinity >/dev/null
  fi
  echo "container '$NAME' is up"
}

case "${1:-up}" in
  up) up ;;
  enter) up; docker exec -it "$NAME" bash ;;
  exec) shift; up >/dev/null; docker exec "$NAME" bash -lc "$*" ;;
  status) docker ps -a --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}\t{{.Image}}' ;;
  rm) docker rm -f "$NAME" ;;
  *) echo "usage: $0 {up|enter|exec <cmd>|status|rm}" >&2; exit 1 ;;
esac
