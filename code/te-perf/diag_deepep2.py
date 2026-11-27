"""Second DeepEP diagnostic: why is expanded dispatch slow/jittery at H=3072?"""
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import BF16, make_routing, measure

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
import deep_ep  # noqa: E402

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", world_size=world, rank=rank, device_id=torch.device(f"cuda:{local_rank}"))
group = dist.new_group(list(range(world)))
sync = lambda: dist.barrier(group=group)

M, H, TOPK, E = (int(os.environ["DIAG_M"]), int(os.environ["DIAG_H"]), 16, 32)
buf = deep_ep.EPBuffer(group, num_max_tokens_per_rank=M, hidden=H, num_topk=TOPK,
                       prefer_overlap_with_compute=False, explicitly_destroy=True)
num_sms = buf.get_theoretical_num_sms(E, TOPK)
torch.manual_seed(1234 + rank)
idx = make_routing(M, E, TOPK, seed=1234 + rank)[0].to(deep_ep.topk_idx_t)
w = make_routing(M, E, TOPK, seed=1234 + rank)[1]
x = torch.randn(M, H, dtype=BF16, device="cuda")
sync()


def show(tag, st):
    if rank == 0:
        print(f"[M{M} H{H}] {tag:38s} kernel={st['kernel_us']:9.1f} wall_med={st['wall_med_us']:9.1f} "
              f"p90={st['wall_p90_us']:9.1f} max={st['wall_max_us']:9.1f}", flush=True)


base = dict(topk_idx=idx, topk_weights=w, num_experts=E, num_max_tokens_per_rank=M, num_sms=num_sms)

# 1. non-expanded, cpu sync
st = measure(lambda: buf.dispatch(x=x, expert_alignment=1, do_cpu_sync=True, **base), 10, 20, barrier=sync)
show("plain  align=1  cpu_sync", st)
# 2. expanded, cpu sync (current bench convention)
st = measure(lambda: buf.dispatch(x=x, expert_alignment=128, do_expand=True,
                                  use_tma_aligned_col_major_sf=True, do_cpu_sync=True, **base),
             10, 20, barrier=sync)
show("expand align=128 cpu_sync", st)
# 3. expanded, no cpu sync
st = measure(lambda: buf.dispatch(x=x, expert_alignment=128, do_expand=True,
                                  use_tma_aligned_col_major_sf=True, do_cpu_sync=False, **base),
             10, 20, barrier=sync)
show("expand align=128 no_cpu_sync", st)
# 4. expanded, async + deferred epilogue (DeepEP's own test convention)
def call_async():
    e = buf.dispatch(x=x, expert_alignment=128, do_expand=True, use_tma_aligned_col_major_sf=True,
                     async_with_compute_stream=True, defer_epilogue=True, **base)
    e.current_stream_wait()
    return e


st = measure(call_async, 10, 20, barrier=sync)
show("expand async+defer", st)
# 5. expanded without the tma sf flag
st = measure(lambda: buf.dispatch(x=x, expert_alignment=128, do_expand=True, do_cpu_sync=True, **base),
             10, 20, barrier=sync)
show("expand align=128 no_sf_flag", st)

buf.destroy()
dist.destroy_process_group()
