"""K2 occupancy experiment: FlashKDA K2 smem/occupancy vs pipeline-stage variants.

Reads the variant modules built by ``_build_k2_occ.py`` (torch cpp_extension cache)
and, for a set of shapes, reports:
  * correctness vs upstream ``flash_kda`` (max abs diff on ``o``)
  * ``wall`` (one event pair over REPS calls, median of rounds)
  * per-kernel device time (torch.profiler / CUPTI): K1 prepare, K2 recurrence, init

The shapes are chosen so that grid_k2 = B*H spans the interesting regimes:
grid < SM (grid-limited) vs grid >= 3*SM (occupancy could matter).

Usage (inside kernel_lab):
    python bench_k2_occ.py --smem           # just print smem budget per variant
    python bench_k2_occ.py                  # timing table
"""
import argparse
import math
import os
import statistics
import sys

import torch

D = 128
SCALE = 1.0 / math.sqrt(D)
LOWER_BOUND = -5.0
WARMUP = 10
REPS = 50
PROF_REPS = 20

VARIANTS = {
    "orig_upstream":      None,                       # pip-installed flash_kda_C (3,2, no cond)
    "plain_in3_out2":     (3, 2, 0),
    "plain_in2_out1":     (2, 1, 0),
    "cond_in3_out2":      (3, 2, 1),
    "cond_in2_out2":      (2, 2, 1),
    "cond_in2_out1":      (2, 1, 1),
    "cond_in1_out2":      (1, 2, 1),
}


def load_variant(name):
    spec = VARIANTS[name]
    if spec is None:
        import flash_kda_C
        return flash_kda_C
    i, o, c = spec
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    base = os.path.join("/tmp/kda_occ", "src_cond" if c else "src_plain", "csrc")
    cutlass = "~/github/FlashKDA/cutlass"
    return load(
        name={"plain_in3_out2": "kda_occ_plain_in3_out2",
              "plain_in2_out1": "kda_occ_plain_in2_out1",
              "cond_in3_out2": "kda_occ_cond_in3_out2",
              "cond_in2_out2": "kda_occ_cond_in2_out2",
              "cond_in2_out1": "kda_occ_cond_in2_out1",
              "cond_in1_out2": "kda_occ_cond_in1_out2"}[name],
        sources=[os.path.join(base, "flash_kda.cpp"),
                 os.path.join(base, "smxx", "fwd_launch.cu")],
        extra_include_paths=[os.path.join(cutlass, "include"),
                             os.path.join(cutlass, "examples", "common"),
                             os.path.join(cutlass, "tools", "util", "include"),
                             base],
        extra_cflags=["-O3", "-Wno-psabi"],
        extra_cuda_cflags=[
            "-O3", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
            "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills", "-lineinfo",
            "--threads", "32", f"-DKDA_IN_STAGES={i}", f"-DKDA_OUT_STAGES={o}"],
        verbose=False)


def make_inputs(B, T, H, device="cuda", seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16) * 2.0
    beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=device)
    dt_bias = torch.randn(H, D, device=device)
    return q, k, v, g, beta, A_log, dt_bias


def make_fn(mod, tensors):
    q, k, v, g, beta, A_log, dt_bias = tensors
    B, T, H, _ = q.shape
    out = torch.empty_like(q)
    ws = torch.empty(mod.get_workspace_size(B * T, H, B), dtype=torch.uint8, device=q.device)

    def fn():
        mod.fwd(q, k, v, g, beta, SCALE, out, ws, A_log, dt_bias, LOWER_BOUND)
    return fn, out


def time_wall(fn, reps=REPS, rounds=7):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        means.append(s.elapsed_time(e) / reps)
    return min(means) * 1e3  # us, best round to suppress co-tenant noise


def profile_kernels(fn, reps=PROF_REPS, passes=3):
    from torch.profiler import ProfilerActivity, profile
    best = None
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(passes):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        per = {}
        for ev in prof.key_averages():
            if ev.self_device_time_total > 0:
                per[ev.key] = per.get(ev.key, 0.0) + ev.self_device_time_total / reps
        if best is None:
            best = per
        else:
            for k, v in per.items():
                best[k] = min(best.get(k, 1e18), v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smem", action="store_true")
    ap.add_argument("--shapes", default="1x1024x4,1x8192x64,8x1024x64,8x2048x64")
    a = ap.parse_args()

    if a.smem:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        return

    shapes = []
    for s in a.shapes.split(","):
        B, T, H = (int(x) for x in s.split("x"))
        shapes.append((B, T, H))

    mods = {name: load_variant(name) for name in VARIANTS}
    print(f"loaded {len(mods)} variants", flush=True)

    for (B, T, H) in shapes:
        print(f"\n===== B={B} T={T} H={H}  grid_k2={B*H}  smem/SM=132 =====", flush=True)
        tensors = make_inputs(B, T, H)
        ref_fn, ref_out = make_fn(mods["orig_upstream"], tensors)
        ref_fn(); torch.cuda.synchronize()
        ref = ref_out.float().clone()
        row = []
        for name, mod in mods.items():
            fn, out = make_fn(mod, tensors)
            fn(); torch.cuda.synchronize()
            d = (out.float() - ref).abs().max().item()
            wall = time_wall(fn)
            per = profile_kernels(fn)
            k2 = sum(v for k, v in per.items() if "recurrence" in k)
            k1 = sum(v for k, v in per.items() if "prepare" in k)
            tot = sum(per.values())
            row.append((name, d, wall, k1, k2, tot))
        print(f"{'variant':18s} {'max|dO|':>9s} {'wall_us':>9s} {'k1_us':>8s} {'k2_us':>9s} {'tot_us':>9s}")
        for name, d, wall, k1, k2, tot in row:
            print(f"{name:18s} {d:9.2e} {wall:9.1f} {k1:8.1f} {k2:9.1f} {tot:9.1f}")


if __name__ == "__main__":
    main()
