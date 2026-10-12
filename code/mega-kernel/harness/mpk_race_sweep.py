"""Controlled-variant sweep to classify the MPK worker wait-loop hang.

The unmodified `f9eb70c` snapshot hangs intermittently right after launch in
`tests/runtime_python/test_mode/test_rmsnorm_testmode.py` (plain). A previous
round found that inserting an unrelated spin counter into the worker "wait for
task" loop made it converge. That perturbation could hide either

  (a) a *timing* race (lost wakeup / scheduling), or
  (b) a *compiler memory-ordering* bug: `ld_acquire_gpu_u64` in
      include/mirage/persistent_kernel/mpk_atoms.cuh has NO `"memory"`
      clobber (unlike `ld_acquire_sys_u64`), so the compiler may hoist the
      shared `next_task_pos` load out of the polling loop and miscompile it.

This script applies a matrix of *minimal, semantically equivalent* edits to
the single worker polling loop (scoped so the identical scheduler loop is NOT
touched) and runs the probe R times per variant, reporting PASS / WRONG /
HANG counts.

Variants
  plain             unmodified snapshot
  counter           prior fix: ++spin + essentially-never-taken branch
  sleep0            __nanosleep(10) -> __nanosleep(0)     (pure timing, shorter)
  sleep1000         __nanosleep(10) -> __nanosleep(1000)  (pure timing, longer)
  compiler_barrier  insert asm volatile("" ::: "memory"); (NO hardware fence)
  threadfence       insert __threadfence();                (hardware fence)
  relaxed           ld_acquire_gpu_u64 -> ld_relaxed_gpu_u64 in the loop

Interpretation
  * sleep* fixes it                     -> timing race
  * compiler_barrier fixes it           -> compiler hoist (missing clobber)
  * threadfence fixes it but
    compiler_barrier does not           -> hardware memory ordering

Run inside kernel_lab:
  MIRAGE_HOME=/ssd/mirage_mpk_build CUDA_VISIBLE_DEVICES=7 \
    python3 -u mpk_race_sweep.py --reps 3 --hang-timeout 25
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MIRAGE_HOME = os.environ.get("MIRAGE_HOME", "/ssd/mirage_mpk_build")
CUH = os.path.join(MIRAGE_HOME,
                   "include/mirage/persistent_kernel/persistent_kernel.cuh")

IF_THREAD0 = "      if (threadIdx.x == 0) {"
LOOP_HEAD = "        while (next_task_pos[queue_idx] == last_task_pos[queue_idx]) {"
BLOCK_HEAD = IF_THREAD0 + "\n" + LOOP_HEAD
END_MARK = "        assert(next_task_pos[queue_idx] + config.per_worker_queue_len >"
NANO = "          __nanosleep(10);"


def get_block(src):
    s = src.index(BLOCK_HEAD)
    e = src.index(END_MARK, s)
    return src[s:e]


def variant_plain(b):
    return b


def variant_counter(b):
    # Accumulating spin counter declared *before* the loop (so it is not a
    # fresh per-iteration variable that the compiler folds away). Mirrors
    # harness/mpk_workaround_only.patch (gated there by MPK_DBG_WATCHDOG).
    b = b.replace(BLOCK_HEAD, IF_THREAD0 +
                  "\n        unsigned long _dbg_spins = 0;\n" + LOOP_HEAD)
    b = b.replace(NANO, NANO +
                  "\n          ++_dbg_spins;\n"
                  "          if (worker_id == 0 && _dbg_spins > 20000000UL) { __trap(); }")
    return b


def variant_sleep0(b):
    return b.replace(NANO, "          __nanosleep(0);")


def variant_sleep1000(b):
    return b.replace(NANO, "          __nanosleep(1000);")


def variant_compiler_barrier(b):
    # No hardware effect; only prevents compiler hoisting/CSE across it.
    return b.replace(NANO, '          asm volatile("" ::: "memory");\n' + NANO)


def variant_threadfence(b):
    return b.replace(NANO, "          __threadfence();\n" + NANO)


def variant_relaxed(b):
    return b.replace("ld_acquire_gpu_u64", "ld_relaxed_gpu_u64")


VARIANTS = {
    "plain": variant_plain,
    "counter": variant_counter,
    "sleep0": variant_sleep0,
    "sleep1000": variant_sleep1000,
    "compiler_barrier": variant_compiler_barrier,
    "threadfence": variant_threadfence,
    "relaxed": variant_relaxed,
}


def run_one(args):
    cmd = [sys.executable, "-u", os.path.join(HERE, "mpk_race_probe.py"),
           "--workers", str(args.workers), "--schedulers", str(args.schedulers),
           "--hang-timeout", str(args.hang_timeout),
           "--device", os.environ.get("MPK_DEVICE", "0")]
    try:
        p = subprocess.run(cmd, cwd=HERE, env=os.environ,
                           capture_output=True, text=True, timeout=args.timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode()
        return 124, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--hang-timeout", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--schedulers", type=int, default=16)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    orig = open(CUH).read()
    block = get_block(orig)
    assert "worker_queue_last_ready_task_id" in block and block.count(NANO) == 1, \
        "worker loop anchors look wrong"
    backup = CUH + ".sweepbak"
    shutil.copy(CUH, backup)

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(VARIANTS)
    results = {}
    try:
        for name in names:
            fn = VARIANTS[name]
            new_block = fn(block)
            assert new_block != block or name == "plain", \
                f"variant {name} made no edit"
            modified = orig.replace(block, new_block)
            with open(CUH, "w") as f:
                f.write(modified)
            votes = []
            first_tail = []
            for r in range(args.reps):
                rc, out = run_one(args)
                v = {0: "PASS", 1: "WRONG", 124: "HANG"}.get(rc, f"rc{rc}")
                votes.append(v)
                if r == 0:
                    first_tail = [ln for ln in out.splitlines()
                                  if "[probe]" in ln]
                print(f"  [{name}] rep {r}: {v} (rc={rc})", flush=True)
            results[name] = (votes, first_tail)
            sys.stdout.flush()
    finally:
        shutil.copy(backup, CUH)
        os.remove(backup)

    print("\n==== SWEEP SUMMARY ({} reps each) ====".format(args.reps), flush=True)
    for name in names:
        votes, tail = results[name]
        c = {v: votes.count(v) for v in set(votes)}
        print(f"  {name:18s} {votes}  -> {c}", flush=True)


if __name__ == "__main__":
    main()
