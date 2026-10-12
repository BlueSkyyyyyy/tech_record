"""P7-15: hand-written plain-CUDA B1 (`dhu`) prototype vs the Triton conclusion of P7-14.

docs/07 §17 (P7-14) measured, on a **Triton** bf16 B1 prototype, that the only lever that
matters is the load pipeline (mirror K2's TMA input pipeline): `tl.range(num_stages)` took
B1 from 1.96x to 1.00x/0.85x FLA `dhu`, while splitting the D=128 state into two accumulator
chains was redundant (and negative when combined).  The open boundary written there was:

    "手写 CUDA/TMA 把预取显式化后是否仍需手工拆链增 ILP，须在目标卡再测。"

This script answers the *prefetch* half on the lab GPU with a hand-written CUDA kernel
(`b1_cuda_probe.cu`, wmma bf16 m16n16k16, fp32 running state, per-warp V columns), in which
the `cp.async` prefetch depth is an explicit template knob `NSTAGE` (1 = issue-then-wait, no
overlap; 2/3 = prefetch 1/2 chunks ahead).  The K=128 state is handled *within* each warp as
8 independent mma m-tiles (no cross-warp reduction), so the manual "split accumulator chain"
of the Triton prototype is naturally expressible and is left as a single chain.

  Part A  correctness of the CUDA kernel vs FLA `dhu` (bf16, same BT), incl. non-divisible T
  Part B  kernel-only CUPTI timing: NSTAGE = 1/2/3, BT = 16/32/64, vs FLA `dhu`
  Part C  smem footprint per (BT, NSTAGE)

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_cuda.py
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
    cuda_kernel_us, stats, make_inputs, as_dtype, fla_dhu, cdiv,
)

D = 128
BV = 32
SO = "/tmp/b1_cuda.so"


def build():
    src = os.path.join(HARNESS, "b1_cuda_probe.cu")
    extra = os.environ.get("B1_CFLAGS", "").split()
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + [src, "-o", SO]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")


def load():
    lib = ctypes.CDLL(SO)
    lib.b1_launch.restype = ctypes.c_int
    lib.b1_launch.argtypes = [ctypes.c_int, ctypes.c_int] + \
        [ctypes.c_void_p] * 9 + \
        [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    lib.b1_smem_bytes.restype = ctypes.c_int
    lib.b1_smem_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, fi, do, dv, scale, BT, NSTAGE):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, Kk = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    last_idx = torch.clamp((torch.arange(NT, device=qg.device) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(g_cum.float()[:, last_idx]).contiguous()      # [B,NT,HV,K]
    dh = torch.empty(B, NT, HV, Kk, V, dtype=torch.bfloat16, device=qg.device)
    dh0 = torch.empty(B, HV, Kk, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty_like(do)
    q = qg.contiguous(); k = kg.contiguous(); ww = w.contiguous()
    do_ = do.contiguous(); dvv = dv.contiguous()
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b1_launch(BT, NSTAGE, ptr(q), ptr(k), ptr(ww), ptr(do_), ptr(dvv), ptr(etot),
                       ptr(dh), ptr(dh0), ptr(dv2),
                       B, T, HV, V, NT, scale, stream)
    if rc != 0:
        raise RuntimeError(f"b1_launch rc={rc} (BT={BT}, NSTAGE={NSTAGE})")
    return dh, dh0, dv2


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build()
    lib = load()
    scale = D ** -0.5
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

    # ------------------------------------------------------------------ Part C
    print("\n" + "=" * 84)
    print("Part C: smem footprint (bytes) per (BT, NSTAGE)")
    print("=" * 84)
    for BT in (16, 32, 64):
        row = []
        for ns in (1, 2, 3):
            row.append(f"ns={ns}: {lib.b1_smem_bytes(BT, ns):6d}")
        print(f"  BT={BT:3d}   " + "   ".join(row))

    # ------------------------------------------------------------------ Part A
    print("\n" + "=" * 84)
    print("Part A: CUDA B1 vs FLA dhu (bf16, same BT) — correctness")
    print("=" * 84)
    for (B, T, H) in [(1, 512, 4), (2, 1024, 8), (1, 250, 3)]:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            dh, dh0, dv2 = run_cuda(lib, F["fi"], do, F["dv"], scale, BT, 1)
            print(f"  B={B} T={T} H={H} BT={BT}")
            stats("dh  cuda vs FLA", dh, F["dh"])
            stats("dv2 cuda vs FLA", dv2, F["dv2"])
            stats("dh0 cuda vs FLA", dh0, F["dh0"])

    # ------------------------------------------------------------------ Part B
    print("\n" + "=" * 84)
    print("Part B: kernel-only CUPTI us/call, NSTAGE sweep vs FLA dhu")
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
                F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)  # operands only
                fla_us = float("nan")
            else:
                F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
                fla_us = cuda_kernel_us(lambda F=F, BT=BT: chunk_gated_delta_rule_bwd_dhu(
                    q=F["fi"]["qg"], k=F["fi"]["kg"], w=F["fi"]["w"], gk=F["fi"]["g_cum"],
                    h0=h0, dht=None, do=do, dv=F["dv"], scale=scale, chunk_size=BT),
                    "bwd_kernel_dhu")
            base_us = None
            ftag = f"{fla_us:7.1f}us" if fla_us == fla_us else "   n/a  "
            print(f"  B{B} T{T} H{H}  BT={BT}   FLA dhu = {ftag}")
            for ns in (1, 2, 3):
                try:
                    us = cuda_kernel_us(lambda F=F, BT=BT, ns=ns: run_cuda(
                        lib, F["fi"], do, F["dv"], scale, BT, ns), "b1_dhu_cuda")
                except RuntimeError as e:
                    print(f"      NSTAGE={ns}   smem-exceeded ({e})")
                    continue
                if base_us is None:
                    base_us = us
                xtag = f"xFLA={us/fla_us:5.2f}" if fla_us == fla_us else "xFLA=  n/a"
                print(f"      NSTAGE={ns}   {us:8.1f}us   {xtag}  xNS1={us/base_us:5.2f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
