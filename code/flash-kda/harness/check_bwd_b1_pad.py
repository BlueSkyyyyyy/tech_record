"""P7-17: isolate the shared-memory bank-conflict term of the hand-written CUDA B1.

§19 (P7-16) ncu-profiled `b1_cuda_probe.cu` and found that **76% of its shared-load
wavefronts are bank-conflict replays** (FLA `dhu` uses ldmatrix: 0.8%), because every
16x16 wmma operand tile is loaded from smem whose row stride (KD=128 bf16 = 256 B = 0
mod 32 words; per-warp B stride BW=16 bf16 = 32 B) maps every row onto the same banks.
§19.5 then listed "换掉 wmma-from-smem（ldmatrix/TMA 整块 + 提 mma 密度，消 bank conflict）"
as the *quantitative* first priority.

This script is the cheapest controlled test of that hypothesis **without** a full ldmatrix
rewrite: the same kernel with the smem row strides padded by 8 elems (16 B), which shifts
consecutive rows by 4 (K, 272 B) / 12 (V, 48 B) banks and makes the 16x16 tile loads
conflict-free.  Build the same source twice (`-DB1_PAD` on/off) and compare:

  Part A  correctness of both builds vs FLA `dhu` (must be bit-identical to each other and
          to the unpadded baseline -> padding only changes smem addresses, not math)
  Part B  kernel-only CUPTI us/call: unpadded vs padded vs FLA `dhu`
  Part C  smem footprint (the 16 B/row pad grows smem -> watch occupancy)

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_pad.py
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dhu_dtype import (                                          # noqa: E402
    cuda_kernel_us, stats, make_inputs, as_dtype, fla_dhu,
)
from check_bwd_b1_cuda import load, run_cuda                               # noqa: E402
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu    # noqa: E402

D = 128
SRC_CU = os.path.join(HARNESS, "b1_cuda_probe.cu")


def build(tag, extra=()):
    so = f"/tmp/b1_cuda_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + list(extra) + [SRC_CU, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    lib = ctypes.CDLL(so)
    lib.b1_launch.restype = ctypes.c_int
    lib.b1_launch.argtypes = [ctypes.c_int, ctypes.c_int] + \
        [ctypes.c_void_p] * 9 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    lib.b1_smem_bytes.restype = ctypes.c_int
    lib.b1_smem_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    return lib


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {"nopad": build("nopad"), "pad": build("pad", ("-DB1_PAD",))}

    scale = D ** -0.5

    # ---------------------------------------------------------------- Part A
    print("\n" + "=" * 84)
    print("Part A: padding must be numerically inert (both builds vs FLA dhu)")
    print("=" * 84)
    for (B, T, H) in [(1, 512, 4), (2, 1024, 8), (1, 250, 3)]:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            outs = {}
            for tag, lib in libs.items():
                outs[tag] = run_cuda(lib, F["fi"], do, F["dv"], scale, BT, 1)
            print(f"  B={B} T={T} H={H} BT={BT}")
            for i, nm in enumerate(("dh", "dh0", "dv2")):
                stats(f"{nm}  pad  vs FLA", outs["pad"][i], F["dh" if nm == "dh"
                      else ("dh0" if nm == "dh0" else "dv2")])
            stats("dh  pad vs nopad", outs["pad"][0], outs["nopad"][0])
            stats("dv2 pad vs nopad", outs["pad"][2], outs["nopad"][2])

    # ---------------------------------------------------------------- Part C
    print("\n" + "=" * 84)
    print("Part C: smem footprint (bytes) per (BT, NSTAGE): nopad vs pad")
    print("=" * 84)
    for BT in (16, 32, 64):
        row = []
        for ns in (1, 2, 3):
            a = libs["nopad"].b1_smem_bytes(BT, ns)
            b = libs["pad"].b1_smem_bytes(BT, ns)
            row.append(f"ns={ns}: {a:6d}->{b:6d}")
        print(f"  BT={BT:3d}   " + "   ".join(row))

    # ---------------------------------------------------------------- Part B
    print("\n" + "=" * 84)
    print("Part B: kernel-only CUPTI us/call — nopad vs pad vs FLA dhu")
    print("=" * 84)
    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8)]
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (16, 32, 64):
            if BT == 16:
                F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
                fla_us = float("nan")
            else:
                F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
                fla_us = cuda_kernel_us(lambda F=F, BT=BT: chunk_gated_delta_rule_bwd_dhu(
                    q=F["fi"]["qg"], k=F["fi"]["kg"], w=F["fi"]["w"], gk=F["fi"]["g_cum"],
                    h0=h0, dht=None, do=do, dv=F["dv"], scale=scale, chunk_size=BT),
                    "bwd_kernel_dhu")
            ftag = f"{fla_us:7.1f}us" if fla_us == fla_us else "   n/a  "
            print(f"  B{B} T{T} H{H}  BT={BT}   FLA dhu = {ftag}")
            base = {}
            for tag, lib in libs.items():
                try:
                    us = cuda_kernel_us(lambda F=F, lib=lib, BT=BT: run_cuda(
                        lib, F["fi"], do, F["dv"], scale, BT, 1), "b1_dhu_cuda")
                except RuntimeError as e:
                    print(f"      {tag:6s} smem-exceeded ({e})")
                    continue
                base[tag] = us
                xtag = f"xFLA={us/fla_us:5.2f}" if fla_us == fla_us else "xFLA=  n/a"
                print(f"      {tag:6s} {us:8.1f}us   {xtag}")
            if "nopad" in base and "pad" in base:
                print(f"      -> pad/nopad = {base['pad']/base['nopad']:5.3f}x")
    print("\nDONE")


if __name__ == "__main__":
    main()
