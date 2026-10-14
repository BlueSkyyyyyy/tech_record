"""P7-3: split FlashKDA K1/K2 memory traffic into read vs write and attribute it
to the analytic buffers (workspace / q,k,g,beta / v / output o / state).

Motivation (docs/07 §5.3 待验证 (b)): ``ncu`` reported that at the design point
K2 is L2-bandwidth bound.  This script answers *where* that L2 traffic comes
from, i.e. how much of K2's read traffic is the per-tile workspace versus the
raw ``v``/``beta`` streams, and how much of the write traffic is the output
``o`` versus the final state.

Method
------
``ncu`` flushes the L2 before each replay pass (default ``--cache-control all``),
so the ``dram__bytes_read/write`` counters measure the *unique* bytes touched by
one kernel invocation.  We compare those to closed-form predictions:

  K1 (prepare):  reads  q,k,g,beta   (it never touches v)
                 writes workspace    = H * ceil(T/16) * 13824 bytes
  K2 (recurrence): reads workspace + v + beta + h0
                 writes o + final_state

Usage (inside kernel_lab):
    python /ssd/.../harness/check_traffic.py | tee .../check_traffic.out.txt
"""

import csv
import io
import os
import subprocess
import sys

D = 128
CHUNK = 16
PER_TILE = 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2  # 13824, utils.cuh:70-76
FLASHKDA = os.path.expanduser("~/github/FlashKDA")
DRIVER = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness/ncu_driver.py")

METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "gpu__time_duration.sum",
]


def predict(B, T, H):
    tiles = B * H * ((T + CHUNK - 1) // CHUNK)
    ws = tiles * PER_TILE
    q_ = B * T * H * D * 2
    beta = B * T * H * 2
    h0 = B * H * D * D * 2
    o = B * T * H * D * 2
    return {
        "tiles": tiles,
        "ws": ws,          # workspace bytes
        "qkg": 3 * q_ + beta,      # K1 input: q,k,g,beta
        "v": q_,           # v stream (K2 only)
        "beta": beta,
        "h0": h0,          # initial/final state
        "o": o,
    }


def profile(kernel, B, T, H):
    cmd = [
        "ncu", "-k", f"regex:{kernel}", "-c", "1", "--launch-skip", "3",
        "--csv", "--metrics", ",".join(METRICS),
        sys.executable, DRIVER, "--B", str(B), "--T", str(T), "--H", str(H),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    p = subprocess.run(cmd, cwd=FLASHKDA, env=env, capture_output=True, text=True)
    vals = {}
    rdr = csv.reader(io.StringIO(p.stdout))
    for row in rdr:
        if len(row) >= 15 and row[12] == "Metric Name":
            continue
        if len(row) >= 15 and row[12] in METRICS:
            vals[row[12]] = float(row[14])
    if not vals:
        sys.stderr.write(p.stdout[-2000:] + "\n" + p.stderr[-2000:] + "\n")
        raise RuntimeError(f"no metrics for {kernel} B={B} T={T} H={H}")
    return vals


def mb(x):
    return x / 1e6


def report(kernel, B, T, H):
    pr = predict(B, T, H)
    mv = profile(kernel, B, T, H)
    dr = mv["dram__bytes_read.sum"]
    dw = mv["dram__bytes_write.sum"]
    l2r = mv["lts__t_sectors_op_read.sum"] * 32
    l2w = mv["lts__t_sectors_op_write.sum"] * 32
    dur = mv["gpu__time_duration.sum"]

    print(f"\n===== {kernel}  B={B} T={T} H={H}  grid B*H={B*H}  "
          f"tiles={pr['tiles']}  dur={dur/1000:.2f}us =====")
    if kernel.endswith("prepare"):
        pred_r = pr["qkg"]
        parts_r = [("q,k,g,beta", pr["qkg"])]
        pred_w = pr["ws"]
        parts_w = [("workspace", pr["ws"])]
    else:
        pred_r = pr["ws"] + pr["v"] + pr["beta"] + pr["h0"]
        parts_r = [("workspace", pr["ws"]), ("v", pr["v"]), ("beta", pr["beta"]), ("h0", pr["h0"])]
        pred_w = pr["o"] + pr["h0"]
        parts_w = [("o", pr["o"]), ("final_state", pr["h0"])]

    print(f"  DRAM read  {mb(dr):9.2f} MB   (predicted {mb(pred_r):9.2f} MB, ratio {dr/pred_r:.3f})")
    for name, b in parts_r:
        print(f"      - {name:<12} {mb(b):9.2f} MB  = {100*b/pred_r:5.1f}% of read")
    print(f"  DRAM write {mb(dw):9.2f} MB   (predicted {mb(pred_w):9.2f} MB, ratio {dw/pred_w:.3f})")
    for name, b in parts_w:
        print(f"      - {name:<12} {mb(b):9.2f} MB  = {100*b/pred_w:5.1f}% of write")
    print(f"  L2   read  {mb(l2r):9.2f} MB   write {mb(l2w):9.2f} MB   (read/write = {l2r/l2w:.2f})")
    return dr, dw, l2r, l2w


def main():
    shapes = [
        (1, 1024, 4),
        (4, 1024, 64),
        (1, 8192, 64),
        (8, 2048, 8),
    ]
    print("FlashKDA K1/K2 memory traffic split (P7-3)")
    print(f"per-tile workspace = {PER_TILE} B (utils.cuh:70-76)")
    print("ncu flushes L2 per replay pass -> dram__bytes == unique bytes touched")
    for B, T, H in shapes:
        for k in ("flash_kda_fwd_prepare", "flash_kda_fwd_recurrence"):
            report(k, B, T, H)


if __name__ == "__main__":
    main()
