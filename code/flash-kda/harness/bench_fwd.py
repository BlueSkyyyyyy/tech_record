"""P7-1: FlashKDA forward timing vs FLA ``chunk_kda`` (CUPTI + wall clock).

Two independent timings are reported for every configuration:

  * ``wall``   -- mean per-call wall time of ``REPS`` calls under one CUDA-event
                  pair (includes launch/CPU overhead), median of ``rounds`` rounds;
  * ``cupti``  -- device (kernel) time obtained from ``torch.profiler`` (which is
                  backed by CUPTI): we profile ``PROF_REPS`` calls, sum the
                  ``cuda_time_total`` of every kernel and divide by ``PROF_REPS``.
                  This strips host launch overhead and shows pure GPU work.

Configuration is swept over ``T`` and ``H`` (and a few ``B``) at the only shape
FlashKDA supports: ``D=128``, ``CHUNK=16``, bf16. FLA runs the matching
``chunk_size=64`` Triton kernel on the same tensors.

Also prints effective FLOPs/s and bytes/s using a simple analytic cost model of
the chunked forward (see ``docs/07-perf.md`` for the derivation).

Usage (inside kernel_lab):
    python ~/proj/tech_record/code/flash-kda/harness/bench_fwd.py
"""

import math
import statistics
import sys

import torch

D = 128
SCALE = 1.0 / math.sqrt(D)
LOWER_BOUND = -5.0
WARMUP = 10
REPS = 50
PROF_REPS = 20


def make_inputs(B, T, H, device="cuda", seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16) * 2.0
    beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=device)
    dt_bias = torch.randn(H, D, device=device)
    h0 = torch.randn(B, H, D, D, device=device)
    return q, k, v, g, beta, A_log, dt_bias, h0


def time_flash(fn, reps=REPS, rounds=3):
    # NB: taking the *median* of back-to-back per-call event pairs is unreliable
    # on Hopper -- CUDA event timestamps of adjacent pairs get interleaved and
    # the median reads ~half the true cost (verified against torch.mm and CUPTI;
    # only sum/N, i.e. the mean, is consistent). We therefore time `reps` calls
    # under one event pair and average, repeating `rounds` times.
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
    return statistics.median(means)


def profile_flash(fn, reps=PROF_REPS):
    """Return device (kernel) time per call, ms, from torch.profiler (CUPTI).

    CUDA-activity only: adding CPU tracing serialises/slows the kernels and
    inflates the device numbers, so host dispatch is measured via ``wall`` instead.
    """
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    dev = 0.0
    for ev in prof.key_averages():
        dev += ev.self_device_time_total
    return dev / reps / 1e3


def analytic_flops(B, T, H, D, chunk=16):
    """Rough forward FLOPs of gated-delta chunked attention (matmul-dominant).

    Counted matmuls per chunk (C=T/chunk tiles, N=B*H*C):
      K1: L = k_dec @ k^T           2*D*chunk*chunk
          INV (block inverse)        negligible (O(chunk^3) per chunk, ~0.27 MFLOP)
      K2: q@s, k@s (each D*chunk*D) 2 * 2*D*chunk*D
          U = INV@U                 2*chunk*chunk*chunk
          Mqk@U                     2*chunk*chunk*chunk
          s += k_rest^T @ U         2*D*chunk*D
    Reported as 1e9 * FLOPs.
    """
    C = (T + chunk - 1) // chunk
    n = B * H * C
    per_chunk = (4 * D * chunk * chunk          # L + INV-ish + Mqk@U
                 + 4 * D * chunk * D            # q@s, k@s
                 + 2 * D * chunk * D)           # k_rest^T @ U
    return n * per_chunk


def analytic_bytes(B, T, H, D, chunk=16):
    """Lower-bound DRAM traffic (bytes): read q,k,v,g,beta + write o (+ state)."""
    elems_in = B * T * H * (4 * D + 1)          # q,k,v,g (D each) + beta (1)
    elems_out = B * T * H * D
    return (elems_in + elems_out) * 2           # bf16 = 2 bytes


def run_case(name, B, T, H):
    import flash_kda
    from fla.ops.kda import chunk_kda

    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
    out_fk = torch.empty(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    fin_fk = torch.empty(B, H, D, D, device="cuda", dtype=torch.bfloat16)
    h0_fk = h0.bfloat16().transpose(-1, -2).contiguous()

    def fk():
        flash_kda.fwd(
            q, k, v, g, beta, SCALE, out_fk,
            A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
            initial_state=h0_fk, final_state=fin_fk,
        )

    def fla():
        chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta, scale=SCALE,
            initial_state=h0, output_final_state=True,
            use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        )

    fk()
    fla()
    torch.cuda.synchronize()

    wall_fk = time_flash(fk)
    wall_fla = time_flash(fla)
    cupti_fk = profile_flash(fk)
    cupti_fla = profile_flash(fla)

    flops = analytic_flops(B, T, H, D) / 1e9
    byts = analytic_bytes(B, T, H, D) / 1e9
    print(f"\n{name:22s} B={B} T={T} H={H} D={D}")
    print(f"  FLOPs~{flops:8.2f} GFLOP  bytes~{byts:7.3f} GB")
    print(f"  FlashKDA  wall={wall_fk*1e3:9.1f}us  dev={cupti_fk*1e3:9.1f}us  "
          f"TFLOP/s={flops/(cupti_fk/1e3):7.1f}  GB/s={byts/(cupti_fk/1e3):7.1f}")
    print(f"  FLA chunk wall={wall_fla*1e3:9.1f}us  dev={cupti_fla*1e3:9.1f}us  "
          f"TFLOP/s={flops/(cupti_fla/1e3):7.1f}  GB/s={byts/(cupti_fla/1e3):7.1f}")
    print(f"  speedup (wall FLA/Flash) = {wall_fla/wall_fk:5.2f}x   "
          f"(dev FLA/Flash) = {cupti_fla/cupti_fk:5.2f}x")
    return dict(name=name, B=B, T=T, H=H, flops=flops, byts=byts,
                wall_fk=wall_fk, cupti_fk=cupti_fk,
                wall_fla=wall_fla, cupti_fla=cupti_fla)


def kernel_breakdown(name, B, T, H):
    """Print per-kernel device-time breakdown for one config (CUPTI)."""
    import flash_kda
    from torch.profiler import ProfilerActivity, profile
    from fla.ops.kda import chunk_kda

    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
    out_fk = torch.empty(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    fin_fk = torch.empty(B, H, D, D, device="cuda", dtype=torch.bfloat16)
    h0_fk = h0.bfloat16().transpose(-1, -2).contiguous()

    def fk():
        flash_kda.fwd(q, k, v, g, beta, SCALE, out_fk, A_log=A_log, dt_bias=dt_bias,
                      lower_bound=LOWER_BOUND, initial_state=h0_fk, final_state=fin_fk)

    def fla():
        chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=SCALE, initial_state=h0,
                  output_final_state=True, use_gate_in_kernel=True,
                  use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND)

    print(f"\n{'='*94}\nKernel breakdown [{name}] B={B} T={T} H={H} (device us/call, {PROF_REPS} calls)")
    for label, fn in [("FlashKDA", fk), ("FLA chunk_kda", fla)]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROF_REPS):
                fn()
            torch.cuda.synchronize()
        print(f"  -- {label} --")
        for ev in prof.key_averages():
            us = ev.self_device_time_total / PROF_REPS
            if us > 0.5:
                print(f"     {us:10.1f} us  {ev.count // PROF_REPS:3d}x  {ev.key[:64]}")


def main():
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    cases = []
    for T in [128, 256, 512, 1024, 2048, 4096]:
        cases.append((f"T={T}", 1, T, 4))
    for H in [1, 2, 4, 8, 16]:
        cases.append((f"H={H}", 1, 1024, H))
    for B in [1, 2, 4, 8]:
        cases.append((f"B={B}", B, 1024, 4))
    cases.append(("T=1023", 1, 1023, 4))
    cases.append(("big B8 T2048 H8", 8, 2048, 8))
    for T, H in [(8192, 16), (8192, 64)]:
        cases.append((f"T={T} H={H}", 1, T, H))

    rows = []
    for name, B, T, H in cases:
        rows.append(run_case(name, B, T, H))

    for name, B, T, H in [("T=1024 H=4", 1, 1024, 4), ("T=8192 H=64", 1, 8192, 64)]:
        kernel_breakdown(name, B, T, H)

    print(f"\n{'='*94}")
    print(f"  {'case':16s} {'B':>2s} {'T':>5s} {'H':>3s} | "
          f"{'fk wall':>8s} {'fk dev':>8s} | {'fla wall':>8s} {'fla dev':>8s} | "
          f"{'fk TF/s':>8s} {'spd(wall)':>9s} {'spd(dev)':>8s}")
    for r in rows:
        print(f"  {r['name']:16s} {r['B']:>2d} {r['T']:>5d} {r['H']:>3d} | "
              f"{r['wall_fk']*1e3:8.1f} {r['cupti_fk']*1e3:8.1f} | "
              f"{r['wall_fla']*1e3:8.1f} {r['cupti_fla']*1e3:8.1f} | "
              f"{r['flops']/(r['cupti_fk']/1e3):8.1f} "
              f"{r['wall_fla']/r['wall_fk']:9.2f} {r['cupti_fla']/r['cupti_fk']:8.2f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
