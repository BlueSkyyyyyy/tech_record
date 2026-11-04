#!/usr/bin/env python3
"""连续批处理调度模拟：token budget、chunked prefill 与重算式抢占。

对应专题第 4 篇。用法：
    python3 scheduler_sim.py                # 对比四种 token budget
    python3 scheduler_sim.py --test         # 自测

模型刻意简化：一步 = 一次 forward；1 个 token 的 KV 记 1 个单位，块按
block_size 向上取整。抢占采用 vLLM V1 的"丢弃 KV、之后重算"，不换出。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    prompt: int
    output: int
    arrival: int
    computed: int = 0          # 已有 KV 的 token 数
    generated: int = 0
    first_token_step: int | None = None
    last_work: int | None = None
    itls: list[int] = field(default_factory=list)  # 单位：一步的 token 工作量
    preemptions: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.output

    @property
    def done(self) -> bool:
        return self.generated >= self.output


class KV:
    def __init__(self, blocks: int, block_size: int) -> None:
        self.capacity = blocks * block_size
        self.block_size = block_size
        self.used = 0

    def need(self, n: int) -> int:
        return -(-n // self.block_size) * self.block_size

    def fits(self, n: int) -> bool:
        """只检查容量，不占用：用于准入判断。"""
        return self.used + self.need(n) <= self.capacity

    def resize(self, old: int, new: int) -> bool:
        if self.used - self.need(old) + self.need(new) > self.capacity:
            return False
        self.used += self.need(new) - self.need(old)
        return True


def simulate(prompts: list[int], outputs: list[int], budget: int,
             kv_blocks: int, block_size: int = 16,
             chunk_cap: int | None = None, reserve_full_isl: bool = True,
             max_steps: int = 20000) -> dict:
    kv = KV(kv_blocks, block_size)
    waiting = [Req(i, p, o, i // 2) for i, (p, o) in enumerate(zip(prompts, outputs))]
    running: list[Req] = []
    finished: list[Req] = []
    preemptions = 0
    step = 0
    work_clock = 0
    max_step_tokens = 0

    def preempt(victim: Req) -> bool:
        nonlocal preemptions
        if victim.computed == 0:
            return False
        kv.resize(victim.computed, 0)
        victim.computed = 0
        victim.preemptions += 1
        victim.last_work = None
        if victim in running:
            running.remove(victim)
        waiting.insert(0, victim)
        preemptions += 1
        return True

    while (waiting or running) and step < max_steps:
        step += 1

        # 准入：默认检查整段 ISL（scheduler_reserve_full_isl=True）。
        while waiting and waiting[0].arrival <= step:
            r = waiting.pop(0)
            want = r.prompt if reserve_full_isl else min(r.prompt, block_size)
            if kv.fits(want):
                running.append(r)
            else:
                waiting.insert(0, r)
                break

        tokens_left = budget
        scheduled: list[tuple[Req, int]] = []
        scheduled_ids: set[int] = set()

        for r in list(running):
            if tokens_left <= 0:
                break
            if r.done or r.computed >= r.total:
                continue
            if r.computed < r.prompt:
                want = r.prompt - r.computed
                if chunk_cap is not None:
                    want = min(want, chunk_cap)
            else:
                want = 1
            want = min(want, tokens_left)
            if want <= 0:
                continue

            # KV 不够：抢占一个"这一步还没被调度"的 running 请求。
            ok = kv.resize(r.computed, r.computed + want)
            while not ok:
                victims = [v for v in reversed(running) if v.rid not in scheduled_ids]
                victim = next((v for v in victims
                               if v is not r and v.computed > 0), None)
                if victim is None:
                    break
                preempt(victim)
                ok = kv.resize(r.computed, r.computed + want)
            if not ok:
                continue

            scheduled.append((r, want))
            scheduled_ids.add(r.rid)
            tokens_left -= want

        step_tokens = budget - tokens_left
        work_clock += step_tokens
        max_step_tokens = max(max_step_tokens, step_tokens)

        done_now = []
        for r, n in scheduled:
            was_prefill = r.computed < r.prompt
            r.computed += n
            if was_prefill and r.computed >= r.prompt:
                if r.first_token_step is None:
                    r.first_token_step = step
                r.generated += 1
                r.last_work = work_clock
            elif not was_prefill:
                r.generated += 1
                if r.last_work is not None:
                    r.itls.append(work_clock - r.last_work)
                r.last_work = work_clock
            if r.done:
                done_now.append(r)

        for r in done_now:
            running.remove(r)
            kv.resize(r.computed, 0)
            finished.append(r)

    ttfts = [(r.first_token_step - r.arrival) for r in finished if r.first_token_step]
    all_itls = [x for r in finished for x in r.itls]
    return {
        "finished": len(finished),
        "steps": step,
        "preemptions": preemptions,
        "mean_ttft": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "p99_ttft": sorted(ttfts)[int(0.99 * (len(ttfts) - 1))] if ttfts else 0,
        "mean_itl": sum(all_itls) / len(all_itls) if all_itls else 0.0,
        "max_itl": max(all_itls) if all_itls else 0,
        "max_step_tokens": max_step_tokens,
        "completed": len(finished) == len(prompts) and all(r.done for r in finished),
    }


def _test() -> None:
    # 预算永远不能超；所有请求必须完成。
    r = simulate([100] * 20, [20] * 20, budget=64, kv_blocks=256)
    assert r["completed"] and r["max_step_tokens"] <= 64, r

    # 预算越小，长 prompt 的 TTFT 越大；单步工作量越小（ITL 越稳）。
    big = simulate([2048] * 8, [8] * 8, budget=8192, kv_blocks=4096)
    small = simulate([2048] * 8, [8] * 8, budget=256, kv_blocks=4096)
    assert big["mean_ttft"] < small["mean_ttft"], (big, small)
    assert big["max_step_tokens"] > small["max_step_tokens"], (big, small)

    # 整段 ISL 预留：小 KV 池下也不会过度准入，自然不抢占。
    safe = simulate([256] * 8, [4] * 8, budget=512, kv_blocks=20,
                    reserve_full_isl=True)
    assert safe["preemptions"] == 0 and safe["completed"], safe

    # 只检查首个 chunk 的准入时，KV 池边缘会出现抢占；模拟必须收敛。
    loose = simulate([256] * 8, [4] * 8, budget=512, kv_blocks=20,
                     reserve_full_isl=False, max_steps=5000)
    assert loose["steps"] < 5000, loose

    print(f"scheduler: all tests passed "
          f"(loose preemptions={loose['preemptions']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--output", type=int, default=32)
    ap.add_argument("--kv-blocks", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--chunk-cap", type=int, default=None)
    ap.add_argument("--no-reserve-isl", action="store_true",
                    help="只按首个 chunk 准入，更容易观察到抢占")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        _test()
        return

    prompts = [args.prompt] * args.requests
    outputs = [args.output] * args.requests
    print(f"{args.requests} 请求, prompt={args.prompt}, output={args.output}, "
          f"KV={args.kv_blocks} 块")
    total_tokens = sum(prompts) + sum(outputs)
    print(f"{'budget':>8}{'完成':>6}{'步数':>7}{'抢占':>6}"
          f"{'TTFT':>8}{'p99 TTFT':>10}{'单步token':>10}{'tok/步':>8}")
    for budget in (8192, 2048, 512, 128):
        r = simulate(prompts, outputs, budget, args.kv_blocks,
                     args.block_size, args.chunk_cap,
                     reserve_full_isl=not args.no_reserve_isl)
        print(f"{budget:>8}{r['finished']:>6}{r['steps']:>7}{r['preemptions']:>6}"
              f"{r['mean_ttft']:>8.1f}{r['p99_ttft']:>10}"
              f"{r['max_step_tokens']:>10}{total_tokens/max(1,r['steps']):>8.0f}")
    print("\nTTFT 单位是步（一步 = 一次 forward）；单步 token 是延迟的代理，"
          "chunk 越大尖刺越高。")


if __name__ == "__main__":
    main()
