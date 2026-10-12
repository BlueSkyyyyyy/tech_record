"""P7-6: decompose the host-side overhead of FLA ``chunk_kda`` vs FlashKDA.

P7-1 showed (via one CUDA-event pair wrapping ``REPS`` calls) that FLA's wall time
is pinned at ~0.86 ms for every shape while its device time varies with T, i.e.
FLA is *host bound*. That left the composition of the ~0.8 ms host tax
unmeasured (``docs/07-perf.md`` section 4). This probe splits it two ways:

  * ``host`` = ``wall`` (event-pair mean over ``REPS`` calls) - ``dev`` (CUPTI
    device time) -- the total CPU wall time not overlapped with GPU work;
  * inside ``host``: ``self_cpu_time`` of each operator from ``torch.profiler``
    (CPU activity) -- this is the Python/framework dispatch cost (Triton JIT
    launcher, autograd, tensor allocs, the CUDA API itself is attributed to the
    op that issues it).

Run one mode per invocation (so nsys can profile them separately):

    python probe_nsys.py fla    [case]
    python probe_nsys.py flash  [case]

``case`` is ``small`` (B1 T1024 H4), ``big`` (B1 T8192 H64) or ``bh``
(B8 T2048 H8). See ``probe_nsys.sh`` for the nsys CUDA-API/kernel breakdown.
"""

import math
import statistics
import sys
import time

import torch

D = 128
SCALE = 1.0 / math.sqrt(D)
LOWER_BOUND = -5.0
WARMUP = 10
REPS = 50
PROF_REPS = 20

CASES = {
    "small": (1, 1024, 4),
    "big": (1, 8192, 64),
    "bh": (8, 2048, 8),
}


def make_inputs(B, T, H, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) * 2.0
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda")
    dt_bias = torch.randn(H, D, device="cuda")
    h0 = torch.randn(B, H, D, D, device="cuda")
    return q, k, v, g, beta, A_log, dt_bias, h0


def build(mode, B, T, H):
    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
    if mode == "fla":
        from fla.ops.kda import chunk_kda

        def fn():
            chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=SCALE,
                      initial_state=h0, output_final_state=True,
                      use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
                      use_beta_sigmoid_in_kernel=True,
                      A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND)
    else:
        import flash_kda
        out = torch.empty(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        fin = torch.empty(B, H, D, D, device="cuda", dtype=torch.bfloat16)
        h0f = h0.bfloat16().transpose(-1, -2).contiguous()

        def fn():
            flash_kda.fwd(q, k, v, g, beta, SCALE, out, A_log=A_log,
                          dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                          initial_state=h0f, final_state=fin)
    return fn


def wall_time(fn, reps=REPS, rounds=3):
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


def decompose(fn):
    """Return (wall_ms, dev_ms, cpu_ms, [(op, self_cpu_us), ...])."""
    from torch.profiler import ProfilerActivity, profile
    wall = wall_time(fn)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROF_REPS):
            fn()
        torch.cuda.synchronize()
    dev = cpu = 0.0
    ops = []
    for ev in prof.key_averages():
        dev += ev.self_device_time_total
        cpu += ev.self_cpu_time_total
        if ev.self_cpu_time_total > 0:
            ops.append((ev.key, ev.self_cpu_time_total / PROF_REPS,
                        ev.count // PROF_REPS))
    ops.sort(key=lambda x: -x[1])
    return wall, dev / PROF_REPS / 1e3, cpu / PROF_REPS / 1e3, ops


def cpu_profile(mode):
    """Deterministic host-CPU decomposition via ``time.process_time``.

    ``wall`` (event-pair) and CUPTI both get noisy once the launch queue fills;
    ``process_time`` (single-threaded here) is the clean measure of "how much
    CPU does one call burn". We report it as a function of T (to show the cost
    is fixed / host-bound) and, for FLA, split the per-call CPU across its
    top-level sub-calls by wrapping them with timers.
    """
    acc, cnt = {}, {}
    if mode == "fla":
        import fla.ops.kda.chunk as ch
        for name in ("chunk_kda_fwd", "l2norm_fwd", "fused_beta_sigmoid"):
            orig = getattr(ch, name)

            def mk(orig, name):
                def w(*a, **k):
                    t = time.process_time()
                    r = orig(*a, **k)
                    acc[name] = acc.get(name, 0.0) + time.process_time() - t
                    cnt[name] = cnt.get(name, 0) + 1
                    return r
                return w
            setattr(ch, name, mk(orig, name))

    print(f"mode={mode}  process_time (pure host CPU) vs T:")
    for B, T, H in [(1, 16, 4), (1, 128, 4), (1, 512, 4), (1, 1024, 4), (1, 2048, 4)]:
        fn = build(mode, B, T, H)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        acc.clear(); cnt.clear()
        N = 100
        t0 = time.process_time()
        for _ in range(N):
            fn()
        dt = (time.process_time() - t0) / N * 1e6
        torch.cuda.synchronize()
        extra = ""
        if mode == "fla":
            extra = "  | " + " ".join(
                f"{n}={acc[n]/N*1e6:.0f}us(x{cnt[n]//N})" for n in acc)
        print(f"  B={B} T={T:5d} H={H}: cpu={dt:7.1f} us/call{extra}")


def microbench():
    """Per-launch host overhead: trivial torch vs trivial Triton kernel.

    GPU work is ~1 us, so the event-pair mean wall time is dominated by the
    host enqueue path (Python dispatcher / Triton launcher + the driver call).
    """
    import triton
    import triton.language as tl

    @triton.jit
    def addk(x, y, o, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(o + offs, tl.load(x + offs, mask=m) + tl.load(y + offs, mask=m), mask=m)

    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    o = torch.empty(4096, device="cuda")

    def tri():
        addk[(1,)](x, y, o, 4096, BLOCK=4096)

    def tor():
        torch.add(x, y, out=o)

    def tor2():
        o.copy_(x)

    print(f"per-launch wall (event-pair mean over {REPS} enqueues, tiny GPU work):")
    for name, fn in [("triton addk", tri), ("torch.add", tor), ("tensor.copy_", tor2)]:
        print(f"  {name:16s} wall = {wall_time(fn, reps=REPS, rounds=3)*1e3:8.2f} us/launch")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fla"
    if mode == "micro":
        microbench()
        return
    if mode in ("fla", "flash") and len(sys.argv) > 2 and sys.argv[2] == "cpu":
        cpu_profile(mode)
        return
    case = sys.argv[2] if len(sys.argv) > 2 else "small"
    how = sys.argv[3] if len(sys.argv) > 3 else "profile"
    B, T, H = CASES[case]
    fn = build(mode, B, T, H)
    if how == "loop":
        # Steady-state loop only (no torch.profiler) so nsys measures overhead.
        # Wrap the measured loop in a CUDA-profiler capture range so nsys can
        # exclude warmup / Triton JIT (cuLibraryLoadData etc.).
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        wall = wall_time(fn, reps=min(n, 10), rounds=1)  # quick wall (warm)
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        cudart.cudaProfilerStop()
        print(f"mode={mode} case={case} B={B} T={T} H={H} loop n={n} "
              f"wall={wall*1e3:.1f} us/call  loop_wall={dt/n*1e6:.1f} us/call")
        return
    wall, dev, cpu, ops = decompose(fn)
    print(f"GPU={torch.cuda.get_device_name()} torch={torch.__version__}")
    print(f"mode={mode} case={case} B={B} T={T} H={H}  REPS={REPS} PROF_REPS={PROF_REPS}")
    print(f"  wall  = {wall*1e3:9.1f} us/call   (event-pair mean)")
    print(f"  dev   = {dev*1e3:9.1f} us/call   (CUPTI device)")
    print(f"  host  = wall-dev = {(wall-dev)*1e3:9.1f} us/call")
    print(f"  CPU self (profiler) sum = {cpu*1e3:9.1f} us/call; "
          f"n_ops>0 = {len(ops)}; launches = {sum(c for _, _, c in ops)}")
    print("  -- top CPU ops (self_cpu us/call, count/call) --")
    for name, us, cnt in ops[:18]:
        print(f"     {us:9.2f}  {cnt:3d}x  {name[:72]}")


if __name__ == "__main__":
    main()
