"""P7-110: driver for `harness/probe_cpasync_bypass.cu` -- a self-contained
microbenchmark of the *read path only* that isolates §51.7's open question:
why does `cp.async` (L1-bypass) read 1.5x more L2 srcunit sectors than the
identical 16B / identical-pattern plain `ld.global.v4`?

All variants issue the same 512 B per warp per iteration access pattern over the
same buffer; only the load instruction differs (see the .cu header).  `ncu`
collection lives in `harness/probe_cpasync_bypass_ncu.sh`
(`harness/probe_cpasync_bypass_ncu.out.txt`).

Run inside `kernel_lab`:
    CUDA_VISIBLE_DEVICES=2 python .../harness/probe_cpasync_bypass.py \
        --mode 0 --pat 1 --nunits 4096 --iters 8
"""
import argparse
import ctypes
import os
import subprocess
import sys

import torch

HARNESS = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HARNESS, "probe_cpasync_bypass.cu")

MODES = {
    0: "cp.async.cg 16B (L1-bypass)",
    1: "cp.async.ca 16B (L1)",
    2: "cp.async.ca  8B x2",
    3: "cp.async.ca  4B x4",
    4: "ld.global.v4 16B (L1)",
    5: "ld.global.cg.v4 16B (L1-bypass)",
    6: "ld.global.f32 4B x4",
}


def build(so="/tmp/probe_cpasync_bypass.so"):
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.probe_launch.restype = ctypes.c_int
    lib.probe_launch.argtypes = [ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_long, ctypes.c_int, ctypes.c_long,
                                 ctypes.c_int, ctypes.c_int]
    return lib


def run_once(lib, mode, pat, x, out, nunits, iters, pitch, grid, block):
    return lib.probe_launch(mode, pat,
                            ctypes.c_void_p(x.data_ptr()),
                            ctypes.c_void_p(out.data_ptr()),
                            nunits, iters, pitch, grid, block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--pat", type=int, default=1)
    ap.add_argument("--nunits", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pitch", type=int, default=128)
    ap.add_argument("--grid", type=int, default=264)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-build", action="store_true",
                    help="use existing /tmp/probe_cpasync_bypass.so (ncu: nvcc hangs "
                         "under ncu, so prebuild outside)")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") is None else 0)
    so = "/tmp/probe_cpasync_bypass.so"
    if not (args.no_build and os.path.exists(so)):
        so = build(so)
    lib = load(so)

    nfloats = max(args.nunits * 128, args.nunits * 4 * args.pitch)
    x = torch.randn(nfloats, device="cuda", dtype=torch.float32)
    out = torch.zeros(args.grid * args.block, device="cuda", dtype=torch.float32)

    rc = run_once(lib, args.mode, args.pat, x, out, args.nunits, args.iters,
                  args.pitch, args.grid, args.block)
    if rc != 0:
        raise SystemExit(f"probe_launch rc={rc} (mode={args.mode} pat={args.pat})")
    torch.cuda.synchronize()
    s = float(out.sum().item())

    # timed (event-pair around `reps` back-to-back launches)
    for _ in range(3):
        run_once(lib, args.mode, args.pat, x, out, args.nunits, args.iters,
                 args.pitch, args.grid, args.block)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(args.reps):
        run_once(lib, args.mode, args.pat, x, out, args.nunits, args.iters,
                 args.pitch, args.grid, args.block)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / args.reps

    print(f"mode={args.mode}({MODES[args.mode]}) pat={args.pat} "
          f"nunits={args.nunits} iters={args.iters} pitch={args.pitch} "
          f"grid={args.grid} block={args.block}")
    print(f"  out.sum={s:.4e}  wall={ms*1000:.1f} us")


if __name__ == "__main__":
    main()
