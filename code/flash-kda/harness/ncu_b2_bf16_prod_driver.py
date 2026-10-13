"""P7-75 `ncu` driver: profile the CUDA B2 **production launch** with bf16-native operand staging
(`b2_mma_probe.cu -DB2M_PROD -DB2M_BF16`).  Dependency-free (torch + ctypes) so the profile holds
exactly one kernel; the `.so` is pre-built **outside** ncu (nvcc hangs under ncu, P7-16).

With `B2BP_BF16=1` the five streamed operands (`vn/do/dv/h/dh`) are bf16 tensors (the rest fp32),
mirroring the fp32 build where they are fp32.  Used for the DRAM/L2/tensor comparison.

    B2P_SO=/tmp/b2_bf16prod_bf16.so B2BP_BF16=1 python .../ncu_b2_bf16_prod_driver.py --shape b8t2048h8
"""
import argparse
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"

SHAPES = {"b8t2048h8": (8, 2048, 8), "b2t1024h8": (2, 1024, 8), "t512h4": (1, 512, 4),
          "t250h3": (1, 250, 3)}
BT = 16
SCALE = 1.0 / (128 ** 0.5)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="b8t2048h8", choices=list(SHAPES))
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    B, T, H = SHAPES[args.shape]
    dev = "cuda"
    K = V = 128
    NT = (T + BT - 1) // BT
    bf16 = os.environ.get("B2BP_BF16", "0") == "1"
    gg = torch.Generator(device=dev).manual_seed(0)
    mk = lambda *s: torch.empty(*s, device=dev)
    # the five streamed operands: bf16 under B2BP_BF16, else fp32
    st5 = dict(dtype=torch.bfloat16) if bf16 else dict()
    v = torch.randn(B, T, H, V, device=dev, generator=gg)             # raw v: always fp32 (vv)
    vn = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    do = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    dv = torch.randn(B, T, H, V, device=dev, generator=gg, **st5)
    dw = torch.randn(B, T, H, K, device=dev, generator=gg)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    g = -torch.rand(B, T, H, K, device=dev, generator=gg)
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev, generator=gg))
    A = torch.randn(B, T, H, BT, device=dev, generator=gg) * 0.1
    h = torch.randn(B, NT, H, K, V, device=dev, generator=gg, **st5)
    dh = torch.randn(B, NT, H, K, V, device=dev, generator=gg, **st5)
    dq, dk, dwo = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    dgk = mk(B, NT, H, K)
    dv2, dvb = mk(B, T, H, V), mk(B, T, H, V)
    dkgb = mk(B, T, H, K)
    dqg, dkg, dg = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    db = torch.zeros(B, T, H, device=dev)
    dApre, dA = mk(B, T, H, BT), mk(B, T, H, BT)

    lib = ctypes.CDLL(os.environ["B2P_SO"])
    fn = lib.b2prod_launch
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * 4 +
                   [ctypes.c_int] * 4 + [ctypes.c_void_p])
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(args.reps):
        rc = fn(ptr(v), ptr(vn), ptr(h), ptr(dh), ptr(do), ptr(dv), ptr(k), ptr(g), ptr(dw),
                ptr(dq), ptr(dk), ptr(dwo), ptr(dgk), ptr(A), ptr(beta), ptr(dv2), ptr(dvb),
                ptr(dkgb), ptr(q), ptr(k), ptr(g), ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                ptr(db), ptr(dApre), ptr(A), ptr(dA), B, T, H, NT, st)
        if rc != 0:
            raise SystemExit(f"b2prod_launch rc={rc}")
    torch.cuda.synchronize()
    print(f"b2prod_launch ok  shape={args.shape} reps={args.reps} bf16={bf16}")


if __name__ == "__main__":
    main()
