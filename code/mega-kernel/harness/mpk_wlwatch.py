"""Insert a printf-only watchdog into the MPK worker "wait for task" loop.

Unlike the trap-based variants, this one *prints and continues*, so it reveals
whether worker[0] is actually spinning in that loop (a trap in the worker
kernel cannot unblock the scheduler kernel, so a trap would still show up as a
hang). Apply, run the probe, then restore.

Run inside kernel_lab:
  python3 mpk_wlwatch.py apply|restore
"""

import os
import sys

MIRAGE_HOME = os.environ.get("MIRAGE_HOME", "/ssd/mirage_mpk_build")
CUH = os.path.join(MIRAGE_HOME,
                   "include/mirage/persistent_kernel/persistent_kernel.cuh")
BAK = CUH + ".wlwatchbak"

IF_THREAD0 = "      if (threadIdx.x == 0) {"
LOOP_HEAD = "        while (next_task_pos[queue_idx] == last_task_pos[queue_idx]) {"
BLOCK_HEAD = IF_THREAD0 + "\n" + LOOP_HEAD
NANO = "          __nanosleep(10);"


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "apply"
    if action == "restore":
        data = open(BAK).read()          # read first: never truncate on error
        open(CUH, "w").write(data)
        os.remove(BAK)
        print("[wlwatch] restored")
        return
    orig = open(CUH).read()
    s = orig.index(BLOCK_HEAD)
    e = orig.index("        assert(next_task_pos[queue_idx] + "
                   "config.per_worker_queue_len >", s)
    block = orig[s:e]
    assert block.count(NANO) == 1
    open(BAK, "w").write(orig)
    newblock = block.replace(BLOCK_HEAD, IF_THREAD0 +
                             "\n        long _dbg = 0;\n" + LOOP_HEAD)
    newblock = newblock.replace(
        NANO,
        NANO +
        "\n          if (worker_id == 0 && (++_dbg % 20000000L) == 0) {\n"
        "            printf(\"[WL] worker0 STUCK spins=%ld next=%llu last=%llu "
        "ready0=%llu\\n\", _dbg, next_task_pos[0], last_task_pos[0],\n"
        "                   ld_acquire_gpu_u64("
        "&config.worker_queue_last_ready_task_id[0]));\n"
        "          }")
    open(CUH, "w").write(orig.replace(block, newblock))
    print("[wlwatch] applied")


if __name__ == "__main__":
    main()
