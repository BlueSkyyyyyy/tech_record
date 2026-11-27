"""Shared helpers for the MoE EP / group-GEMM benchmark trio (uniEP / DeepEP / TE group-gemm).

Conventions match the rest of `te-perf`: the primary number is the **pure CUPTI
device kernel time** (sum of all kernel durations launched by one call, divided
by the repeat count), plus a wall-clock figure from CUDA events for reference.

All three benchmark scripts (bench_uniep_ep.py, bench_deepep_ep.py,
bench_te_group_gemm.py) build the *same* routing so the comparisons are
apples-to-apples:

  * `num_tokens` tokens per rank, uniform routing over `num_experts` experts,
  * `topk` **distinct** experts per token,
  * EP = world size, `num_experts // world_size` local experts per rank.
"""

import csv
import json
import os
import re
import statistics

import torch

BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------

def make_routing(num_tokens, num_experts, topk, seed):
    """Uniform random routing with distinct experts (same distribution for all frameworks).

    Returns (topk_idx int32 [T, topk], topk_weights fp32 [T, topk]).
    """
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=g)
    topk_idx = scores.topk(topk, dim=1, largest=True, sorted=False).indices.to(torch.int32).contiguous()
    topk_weights = torch.softmax(scores.gather(1, topk_idx.long()), dim=1, dtype=torch.float32).contiguous()
    return topk_idx, topk_weights


def local_m_splits(topk_idx, num_experts, world_size, group=None):
    """Per-local-expert row counts after a global EP all-to-all.

    `topk_idx` is *this rank's* routing; every rank contributes its own tokens, so
    we all-gather the per-expert histograms and sum over ranks. Returns a python
    list of length `num_experts // world_size` (this rank's experts).
    """
    counts = torch.bincount(topk_idx.reshape(-1).long(), minlength=num_experts)[:num_experts].to(torch.int64)
    if group is not None and group.size() > 1:
        gathered = [torch.zeros_like(counts) for _ in range(group.size())]
        torch.distributed.all_gather(gathered, counts, group=group)
        counts = torch.stack(gathered).sum(dim=0)
    per_expert = counts.cpu().tolist()
    epr = num_experts // world_size
    rank = group.rank() if group is not None else 0
    return per_expert[rank * epr:(rank + 1) * epr]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _cupti_device_us(prof, iters):
    """Robust pure-device time for one call.

    For every launched kernel (keyed by name + grid + block) take the *median*
    duration over the profiled iterations instead of the mean: this box has
    spurious multi-ms stalls (host scheduling) that inflate the mean of the
    profiled pass by 2x on some ops. The per-call total is
    sum_k median_k * launches_per_call_k.
    """
    samples = {}
    order = []
    for e in prof.events():
        if getattr(e, "device_type", None) is not None and "cuda" not in str(e.device_type).lower():
            continue
        t = getattr(e, "device_time", 0.0) or 0.0
        if t <= 0:
            continue
        key = (e.name, getattr(e, "grid", None), getattr(e, "block", None))
        if key not in samples:
            samples[key] = []
            order.append(key)
        samples[key].append(t)
    total = 0.0
    for key in order:
        vals = sorted(samples[key])
        med = vals[len(vals) // 2]
        total += med * max(1.0, len(vals) / iters)
    return total


def probe_quiet(group, expect_us=80.0, tries=6, sleep_s=10.0):
    """Gate the measurement on a cheap cross-rank reference op.

    This box intermittently stalls host-side by 3-7 ms per call (other tenants), which
    inflates *every* communication op in a config while leaving GEMMs untouched. An
    8-rank `all_reduce` of 4 MB should cost ~50-100 µs; if the median is several times
    that, the machine is busy -- wait and retry so a whole config is not measured
    during a stall window.
    """
    import time
    if group is None or group.size() == 1:
        return True
    buf = torch.ones(1 << 20, dtype=torch.float32, device="cuda")
    ok = True
    for attempt in range(tries):
        for _ in range(5):
            torch.distributed.all_reduce(buf, group=group)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        for i in range(10):
            starts[i].record()
            torch.distributed.all_reduce(buf, group=group)
            stops[i].record()
        torch.cuda.synchronize()
        vals = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, stops))
        med = vals[len(vals) // 2]
        ok = med < expect_us * 3
        if ok:
            if attempt and group.rank() == 0:
                print(f"  [probe] quiet after {attempt} retries (all_reduce median {med:.0f} us)", flush=True)
            del buf
            return True
        if group.rank() == 0:
            print(f"  [probe] host is stalling: all_reduce median {med:.0f} us "
                  f"(expect ~{expect_us:.0f}), retry {attempt + 1}/{tries} in {sleep_s:.0f}s", flush=True)
        time.sleep(sleep_s)
    del buf
    return False


def measure(fn, warmup=10, iters=50, barrier=None):
    """Return timing stats for one op.

    kernel_us:  CUPTI pure device time per call (sum over every launched kernel).
    wall_med:   median per-call CUDA-event time (robust to the multi-ms host-jitter
                outliers this shared box produces every few iterations).
    wall_p90 / wall_max: tail of the per-call distribution (straggler cost).
    wall_us:    mean per-call time (comparable to the legacy te-perf numbers).
    jitter:     fraction of calls slower than 3x the median (host/jitter diagnostic).
    """
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    if barrier is not None:
        barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        out = fn()
        stops[i].record()
    torch.cuda.synchronize()
    samples = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, stops))
    n = len(samples)
    wall_med = samples[n // 2]
    wall_p90 = samples[min(n - 1, int(0.9 * n))]
    wall_max = samples[-1]
    wall_mean = sum(samples) / n
    jitter = sum(1 for v in samples if v > 3 * wall_med) / n
    if barrier is not None:
        barrier()

    # CUPTI attach/detach costs show up as one huge kernel duration in the first
    # profiler session of a process; burn a throwaway session so the reported
    # numbers are not skewed (the first measured op is otherwise inflated ~7x).
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            out = fn()
        torch.cuda.synchronize()
    kernel_us = _cupti_device_us(prof, iters)
    if barrier is not None:
        barrier()
    return {"kernel_us": kernel_us, "wall_us": wall_mean, "wall_med_us": wall_med,
            "wall_p90_us": wall_p90, "wall_max_us": wall_max, "jitter": jitter, "out": out}


def measure_robust(fn, warmup=10, iters=50, barrier=None, tries=3):
    """measure() with a re-try when a host stall polluted the wall numbers.

    The box stalls for milliseconds at a time (other tenants); a comm op measured in
    such a window reports wall >> kernel. Re-measure and keep the quietest attempt.
    """
    best = None
    for attempt in range(tries):
        st = measure(fn, warmup, iters, barrier)
        if best is None or st["wall_med_us"] < best["wall_med_us"]:
            best = st
        slow = best["wall_med_us"] > max(1000.0, 3.0 * max(best["kernel_us"], 1.0))
        if not slow or attempt == tries - 1:
            return best
        print(f"  [retry] wall {best['wall_med_us']:.0f} us vs kernel {best['kernel_us']:.0f} us"
              f" -> host stall, re-measuring ({attempt + 1}/{tries - 1})", flush=True)
        import time
        time.sleep(5.0)
    return best


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def gather_rows(rows, group):
    """Collect per-rank rows on rank 0 and aggregate.

    kernel_us -> mean over ranks (kernel duration is per-device);
    wall_us   -> max over ranks (straggler, what the step actually waits for).
    Assumes every rank produced rows in the same order/keys.
    """
    world = group.size() if group is not None else 1
    if world > 1:
        payload = [None] * world
        torch.distributed.all_gather_object(payload, rows, group=group)
    else:
        payload = [rows]
    merged = []
    for i, base in enumerate(payload[0]):
        item = dict(base)
        item["kernel_us"] = statistics.fmean(p[i]["kernel_us"] for p in payload)
        item["kernel_us_max"] = max(p[i]["kernel_us"] for p in payload)
        # straggler is what a step actually waits for
        for k in ("wall_us", "wall_med_us", "wall_p90_us", "wall_max_us"):
            item[k] = max(p[i][k] for p in payload)
        merged.append(item)
    return merged


CSV_FIELDS = [
    "framework", "op", "dtype", "shape", "device",
    "tokens_per_rank", "hidden", "intermediate", "topk", "num_experts", "ep",
    "kernel_us", "kernel_us_max", "wall_us", "wall_med_us", "wall_p90_us", "wall_max_us",
    "jitter", "note",
]


def row_from_stats(stats, **meta):
    """Build a CSV row from a measure() result plus metadata."""
    row = dict(meta)
    row.update({
        "kernel_us": stats["kernel_us"],
        "kernel_us_max": stats["kernel_us"],
        "wall_us": stats["wall_us"],
        "wall_med_us": stats["wall_med_us"],
        "wall_p90_us": stats["wall_p90_us"],
        "wall_max_us": stats["wall_max_us"],
        "jitter": round(float(stats.get("jitter", 0.0)), 3),
    })
    return row


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def fmt_shape(vals):
    return "(" + ", ".join(str(v) for v in vals) + ")"


def device_name():
    """Short device tag, e.g. 'H100' (same convention as the other te-perf tables)."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    for token in name.replace("NVIDIA", "").split():
        if re.match(r"^[A-Z]+\d+[A-Za-z]*$", token):   # e.g. H100 in "H100 80GB HBM3"
            return token
    return name


def bcast_json(obj, src=0, group=None):
    """Broadcast a small json-able object from `src`."""
    world = group.size() if group is not None else 1
    if world == 1:
        return obj
    box = [obj if group.rank() == src else None]
    torch.distributed.broadcast_object_list(box, src=src, group=group)
    return box[0]


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
