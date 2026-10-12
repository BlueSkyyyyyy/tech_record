"""P7-20: TMA + warp-specialized hand-written CUDA B1 (`dhu`) — the §22.5 next step.

`b1_tma_probe.cu` takes the P7-19 K2-shape kernel (register-resident state + raw
`mma.sync`/`ldmatrix`) and replaces the operand path with a *producer-warp* TMA
pipeline: q/k/w/dO/dV are staged by `cp.async.bulk.tensor` into 128 B-swizzled
smem atoms (conflict-free `ldmatrix` *without* the +8 padding), and stage reuse is
handed back through per-stage `empty` mbarriers — no `__syncthreads()` in the loop.

  Part A  correctness vs FLA `dhu` (bf16, same BT), incl. non-divisible T
  Part B  kernel-only CUPTI timing, NSTAGE = 1/2/3, BT = 16/32/64, vs FLA `dhu`,
          the P7-19 reg kernel (`b1_reg_probe.cu`) and the wmma base (`b1_cuda_probe.cu`)
  Part C  smem footprint vs reg

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_tma.py
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
TMA_SO = "/tmp/b1_tma.so"
TMAE_SO = "/tmp/b1_tma_e.so"
REG_SO = "/tmp/b1_reg.so"
REG64_SO = "/tmp/b1_reg64.so"
BASE_SO = "/tmp/b1_cuda.so"


def build(src_name, so, extra=()):
    src = os.path.join(HARNESS, src_name)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + list(extra) + [src, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"nvcc failed: {src_name}")


def load(so):
    lib = ctypes.CDLL(so)
    for name in ("b1t_launch", "b1r_launch", "b1_launch"):
        if hasattr(lib, name):
            f = getattr(lib, name)
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 9 + \
                [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    for name in ("b1t_smem_bytes", "b1r_smem_bytes", "b1_smem_bytes"):
        if hasattr(lib, name):
            s = getattr(lib, name)
            s.restype = ctypes.c_int
            s.argtypes = [ctypes.c_int, ctypes.c_int]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run(lib, fi, do, dv, scale, BT, NSTAGE, launcher):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, Kk = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    last_idx = torch.clamp((torch.arange(NT, device=qg.device) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(g_cum.float()[:, last_idx]).contiguous()
    dh = torch.empty(B, NT, HV, Kk, V, dtype=torch.bfloat16, device=qg.device)
    dh0 = torch.empty(B, HV, Kk, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty_like(do)
    q = qg.contiguous(); k = kg.contiguous(); ww = w.contiguous()
    do_ = do.contiguous(); dvv = dv.contiguous()
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = getattr(lib, launcher)(BT, NSTAGE, ptr(q), ptr(k), ptr(ww), ptr(do_), ptr(dvv),
                                ptr(etot), ptr(dh), ptr(dh0), ptr(dv2),
                                B, T, HV, V, NT, scale, stream)
    if rc != 0:
        raise RuntimeError(f"{launcher} rc={rc} (BT={BT}, NSTAGE={NSTAGE})")
    return dh, dh0, dv2


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build("b1_tma_probe.cu", TMA_SO, extra=["-lcuda"] + os.environ.get("TMA_EXTRA", "").split())
    build("b1_tma_probe.cu", TMAE_SO, extra=["-lcuda", "-DB1T_ETMA"])
    build("b1_reg_probe.cu", REG_SO)
    build("b1_reg_probe.cu", REG64_SO, extra=["-DB1R_BV=64", "-DB1R_NW=4"])
    build("b1_cuda_probe.cu", BASE_SO)
    lib_tma = load(TMA_SO)
    lib_tmae = load(TMAE_SO)
    lib_reg = load(REG_SO)
    lib_reg64 = load(REG64_SO)
    lib_base = load(BASE_SO)
    scale = D ** -0.5
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

    # ------------------------------------------------------------------ Part C
    print("\n" + "=" * 88)
    print("Part C: smem footprint (bytes) per (BT, NSTAGE)  [tma / tmae / reg]")
    print("=" * 88)
    for BT in (16, 32, 64):
        row = []
        for ns in (1, 2, 3):
            t = lib_tma.b1t_smem_bytes(BT, ns)
            e = lib_tmae.b1t_smem_bytes(BT, ns)
            r = lib_reg.b1r_smem_bytes(BT, ns)
            row.append(f"ns={ns}: {'  n/a' if t < 0 else f'{t:6d}'}"
                       f"/{'  n/a' if e < 0 else f'{e:6d}'}/{r:6d}")
        print(f"  BT={BT:3d}   " + "   ".join(row))

    # ------------------------------------------------------------------ Part A
    print("\n" + "=" * 88)
    print("Part A: TMA+warp-specialized B1 vs FLA dhu (bf16, same BT) — correctness")
    print("=" * 88)
    for (B, T, H) in [(1, 512, 4), (2, 1024, 8), (1, 250, 3)]:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            dh, dh0, dv2 = run(lib_tma, F["fi"], do, F["dv"], scale, BT, 1, "b1t_launch")
            dhe, dh0e, dv2e = run(lib_tmae, F["fi"], do, F["dv"], scale, BT, 1, "b1t_launch")
            dh_r, dh0_r, dv2_r = run(lib_reg, F["fi"], do, F["dv"], scale, BT, 1, "b1r_launch")
            print(f"  B={B} T={T} H={H} BT={BT}")
            stats("dh  tma vs FLA", dh, F["dh"])
            stats("dv2 tma vs FLA", dv2, F["dv2"])
            stats("dh0 tma vs FLA", dh0, F["dh0"])
            stats("dh  tmae vs FLA", dhe, F["dh"])
            stats("dv2 tmae vs FLA", dv2e, F["dv2"])
            stats("dh0 tmae vs FLA", dh0e, F["dh0"])
            stats("dh  tmae vs tma", dhe, dh)
            stats("dv2 tmae vs tma", dv2e, dv2)
            stats("dh0 tmae vs tma", dh0e, dh0)
            stats("dh  tma vs reg", dh, dh_r)
            stats("dv2 tma vs reg", dv2, dv2_r)
            stats("dh0 tma vs reg", dh0, dh0_r)

    # ------------------------------------------------------------------ Part B
    print("\n" + "=" * 88)
    print("Part B: kernel-only CUPTI us/call; tma NSTAGE sweep vs reg/base and FLA dhu")
    print("=" * 88)
    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8)]
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F64 = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        for BT in (16, 32, 64):
            if BT == 64:
                fla_us = cuda_kernel_us(lambda: chunk_gated_delta_rule_bwd_dhu(
                    q=F64["fi"]["qg"], k=F64["fi"]["kg"], w=F64["fi"]["w"],
                    gk=F64["fi"]["g_cum"], h0=h0, dht=None, do=do, dv=F64["dv"],
                    scale=scale, chunk_size=64), "bwd_kernel_dhu")
                F = F64
            else:
                F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do) if BT == 32 else F64
                fla_us = float("nan")
            ftag = f"{fla_us:7.1f}us" if fla_us == fla_us else "   n/a  "
            reg_us = cuda_kernel_us(lambda F=F, BT=BT: run(
                lib_reg, F["fi"], do, F["dv"], scale, BT, 1, "b1r_launch"), "b1_reg_cuda")
            reg64_us = cuda_kernel_us(lambda F=F, BT=BT: run(
                lib_reg64, F["fi"], do, F["dv"], scale, BT, 1, "b1r_launch"), "b1_reg_cuda")
            base_us = cuda_kernel_us(lambda F=F, BT=BT: run(
                lib_base, F["fi"], do, F["dv"], scale, BT, 1, "b1_launch"), "b1_dhu_cuda")
            print(f"  B{B} T{T} H{H}  BT={BT}   FLA dhu = {ftag}   reg(BV32,NS=1) = {reg_us:7.1f}us"
                  f"   reg64(BV64,NS=1) = {reg64_us:7.1f}us   base(wmma) = {base_us:7.1f}us")
            for ns in (1, 2, 3):
                try:
                    us = cuda_kernel_us(lambda F=F, BT=BT, ns=ns: run(
                        lib_tma, F["fi"], do, F["dv"], scale, BT, ns, "b1t_launch"),
                        "b1_tma_kernel")
                except RuntimeError as e:
                    print(f"      tma NSTAGE={ns}   smem-exceeded ({e})")
                    continue
                xtag = f"xFLA={us/fla_us:5.2f}" if fla_us == fla_us else "xFLA=  n/a"
                # P7-21: E^tot staged via TMA (B1T_ETMA) — same shape, only `e` moves into smem
                try:
                    use = cuda_kernel_us(lambda F=F, BT=BT, ns=ns: run(
                        lib_tmae, F["fi"], do, F["dv"], scale, BT, ns, "b1t_launch"),
                        "b1_tma_kernel")
                    etag = f"  tmae={use:7.1f}us xTMA={use/us:5.2f}"
                except RuntimeError:
                    etag = "  tmae=  smem-exceeded"
                print(f"      tma NSTAGE={ns}   {us:8.1f}us   {xtag}  xreg={us/reg_us:5.2f}"
                      f"  xreg64={us/reg64_us:5.2f}  xbase={us/base_us:5.2f}{etag}")
    print("\nDONE")


if __name__ == "__main__":
    main()
