#!/usr/bin/env python3
"""调度装箱模拟：SGLang 式「decode 优先 + 单条 chunked」 vs vLLM 式「统一 token 预算」。

对应专题第 4 篇（调度器）。这个模型只保留三件事：

    token budget    一步最多算多少 token（vLLM: max_num_batched_tokens / SGLang: max_prefill_tokens）
    KV 容量         一共多少个 token 的 slot，装不下时 SGLang retract、vLLM preempt，两者都是「丢掉重算」
    到达与生成      prompt 长度 / 输出长度 / 到达步分布

两种策略的差别只在**剩余预算怎么花**：

    sglang  先给所有 running 各 1 个 decode token，剩余预算按 FCFS 尝试装新请求；
            如果一条都装不下，就把队列头那条长 prompt 切一块进来（chunked prefill，一次只切一条）
    vllm    running 的 decode 与 waiting 的 prefill 共用同一个预算，长 prompt 直接切块铺满预算

用法：
    python3 scheduler_sim.py                 # 默认负载对照
    python3 scheduler_sim.py --kv 12000      # 收紧 KV，看 retract/preempt
    python3 scheduler_sim.py --test
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    prompt: int
    out_len: int
    arrive: int
    done: bool = False
    # 已经算过 KV 的 token 数（prompt + 已生成）
    computed: int = 0
    generated: int = 0
    start_step: int | None = None
    first_token_step: int | None = None
    finish_step: int | None = None
    itls: list[int] = field(default_factory=list)
    last_gen_step: int = 0
    retracts: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.generated


@dataclass
class Result:
    policy: str
    ttft: list[int]
    itl: list[int]
    retracts: int
    steps: int
    tokens: int

    def p(self, xs: list[int], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return float(s[min(len(s) - 1, int(len(s) * q))])


def make_load(n: int, seed: int = 0) -> list[Req]:
    """混合负载，刻意制造「队头阻塞」：

    t=0 一条 8000 token 的长 prompt（会占住 chunked prefill 的队头），
    随后 60 步内涌入大量短请求——这正是 chunked prefill 两种切法差别最大的场景。
    """
    rng = random.Random(seed)
    reqs = [Req(rid=0, prompt=8000, out_len=64, arrive=0)]
    t = 2
    for i in range(1, n):
        if i % 6 == 0:
            prompt = rng.randint(1500, 2500)
        else:
            prompt = rng.randint(128, 384)
        reqs.append(Req(rid=i, prompt=prompt, out_len=rng.randint(16, 64), arrive=t))
        t += rng.choice([0, 1, 1, 2])
    return reqs


def sglang_victim(running: list[Req]) -> Req:
    """SGLang retract_decode 的近似：先丢「生成最少」的，同分先丢「prompt 最长」的。

    真实排序键是 (len(output_ids), -len(origin_input_ids)) 降序后从尾部 pop
    （schedule_batch.py::_get_decode_retraction_order）。
    """
    return min(running, key=lambda r: (r.generated, -r.prompt))


def vllm_victim(running: list[Req]) -> Req:
    """vLLM preempt 的近似：挑「已算 KV 最多」的那个，丢掉的代价最大但腾出的最多。

    真实实现按优先级 / 到达顺序挑（vllm/v1/core/sched/scheduler.py 的 _preempt）。
    """
    return max(running, key=lambda r: r.computed)


def simulate(policy: str, load: list[Req], budget: int, kv: int, max_running: int,
             chunk: int = 4096) -> Result:
    """一步 = 选一批请求、给每个请求若干 token、推进生成。

    关键：**prefill 中的请求与 decode 中的请求是两回事**。
    `computed < prompt` 表示 prompt 还没算完（chunked prefill 的中间态），
    它只能继续吃 prefill 预算；`computed >= prompt` 才是 decode，每步吃 1 个。
    """
    pending = sorted(load, key=lambda r: r.arrive)
    waiting: list[Req] = []
    running: list[Req] = []
    done: list[Req] = []
    step = 0
    total_tokens = 0
    retracts = 0

    def kv_used() -> int:
        return sum(r.computed for r in running)

    while len(done) < len(load) and step < 200_000:
        while pending and pending[0].arrive <= step:
            waiting.append(pending.pop(0))

        step_budget = budget
        touched: list[tuple[Req, int]] = []

        # --- 阶段 A：decode（只服务「prompt 已算完」的请求，每个 1 token） ---
        for r in list(running):
            if step_budget <= 0:
                break
            if r.computed < r.prompt:
                continue                      # 还在 prefill，交给阶段 B
            if r.computed + 1 > kv:
                continue
            touched.append((r, 1))
            step_budget -= 1
            r.computed += 1

        # --- 阶段 B：prefill ---
        if policy == "sglang":
            # B1: 继续已经切块的请求（一次只切一条，最多 chunk 个 token）
            for r in list(running):
                if r.computed >= r.prompt or step_budget <= 0:
                    continue
                take = min(r.prompt - r.computed, step_budget, chunk, kv - kv_used())
                if take > 0:
                    touched.append((r, take))
                    step_budget -= take
                    r.computed += take
                break                          # chunked prefill 一次只推进一条
            # B2: 整条装入新请求（三重约束：token / 请求数 / KV）
            for r in list(waiting):
                if step_budget <= 0 or len(running) >= max_running:
                    break
                need = r.prompt - r.computed
                if need <= step_budget and kv_used() + need <= kv:
                    waiting.remove(r)
                    running.append(r)
                    touched.append((r, need))
                    step_budget -= need
                    r.computed += need
            # B3: 一条都装不下 → 把队头切一块进来
            if step_budget > 0 and waiting and len(running) < max_running:
                r = waiting[0]
                take = min(r.prompt - r.computed, step_budget, chunk, kv - kv_used())
                if take > 0:
                    waiting.remove(r)
                    running.append(r)
                    touched.append((r, take))
                    step_budget -= take
                    r.computed += take
        else:
            # vllm：running 的欠账与 waiting 的新请求共用一个预算，长 prompt 直接铺满
            for r in list(running):
                if r.computed >= r.prompt or step_budget <= 0:
                    continue
                take = min(r.prompt - r.computed, step_budget, kv - kv_used())
                if take > 0:
                    touched.append((r, take))
                    step_budget -= take
                    r.computed += take
            for r in list(waiting):
                if step_budget <= 0 or len(running) >= max_running:
                    break
                take = min(r.prompt - r.computed, step_budget, kv - kv_used())
                if take <= 0:
                    continue
                waiting.remove(r)
                running.append(r)
                touched.append((r, take))
                step_budget -= take
                r.computed += take

        # --- 阶段 C：KV 超卖 → 丢请求（两种策略都是「丢 KV、之后重算」） ---
        guard = 0
        while kv_used() > kv and running and guard < 1000:
            guard += 1
            victim = sglang_victim(running) if policy == "sglang" else vllm_victim(running)
            running.remove(victim)
            kv_used_delta = victim.computed
            victim.computed = 0
            victim.retracts += 1
            retracts += 1
            waiting.append(victim)
            del kv_used_delta

        # --- 阶段 D：把这一步算的 token 记成输出 ---
        for r, take in touched:
            if r.start_step is None:
                r.start_step = step
            after = max(0, r.computed - r.prompt)
            before = max(0, r.computed - take - r.prompt)
            new_gen = after - before
            if new_gen > 0:
                if r.first_token_step is None:
                    r.first_token_step = step
                else:
                    r.itls.append(step - r.last_gen_step)
                r.last_gen_step = step
                r.generated += new_gen
            total_tokens += take

        # --- 阶段 E：收尾 ---
        for r in list(running):
            if r.generated >= r.out_len:
                r.finish_step = step
                running.remove(r)
                done.append(r)

        step += 1

    ttft = [(r.first_token_step - r.arrive) if r.first_token_step is not None else 9999 for r in done]
    itl = [x for r in done for x in r.itls]
    return Result(policy, ttft, itl, retracts, step, total_tokens)


def report(load, budget, kv, max_running):
    print(f"负载: {len(load)} 请求，prompt 合计 {sum(r.prompt for r in load):,} token；"
          f"budget={budget}, kv={kv}, max_running={max_running}\n")
    hdr = f"{'策略':<8}{'总步数':>7}{'TTFT p50':>10}{'TTFT p99':>10}{'ITL p50':>9}{'ITL p99':>9}{'retract':>9}{'token/步':>10}"
    print(hdr)
    print("-" * (len(hdr) + 2))
    results = []
    for p in ("sglang", "vllm"):
        r = simulate(p, [Req(r.rid, r.prompt, r.out_len, r.arrive) for r in load], budget, kv, max_running)
        results.append(r)
        print(f"{r.policy:<8}{r.steps:>7}{r.p(r.ttft, .5):>10.0f}{r.p(r.ttft, .99):>10.0f}"
              f"{r.p(r.itl, .5):>9.0f}{r.p(r.itl, .99):>9.0f}{r.retracts:>9}{r.tokens / r.steps:>10.1f}")
    print(
        "\n读法：\n"
        "  · 预算充裕时两种策略几乎重合——两边都是「decode 优先 + 预算内装箱」，这不是巧合，\n"
        "    而是 continuous batching 的唯一合理解。\n"
        "  · 差别只在两处：① 排序（SGLang 的 lpm 会把同前缀的请求排一起，vLLM 是 FCFS）\n"
        "    ② KV 紧张时牺牲谁（见下）。\n"
        "  · SGLang 先丢「生成最少」的：它的输出最短，重算最便宜；vLLM 挑「已算 KV 最多」的，\n"
        "    腾得多但重算也贵。两种都是「丢 KV 重算」，没有 swap。\n"
        "  · 把 --kv 调到接近单条长 prompt 的长度，就能看到 retract/preempt 的频率差。"
    )
    return results


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _test():
    load = make_load(120, seed=1)
    a = simulate("sglang", [Req(r.rid, r.prompt, r.out_len, r.arrive) for r in load], 2048, 200_000, 64)
    b = simulate("vllm", [Req(r.rid, r.prompt, r.out_len, r.arrive) for r in load], 2048, 200_000, 64)

    # 1) 都跑完，且没有 retract（KV 充足）
    assert len(a.ttft) == len(load) and len(b.ttft) == len(load)
    assert a.retracts == 0 and b.retracts == 0

    # 2) KV 充足时，两者的总步数应在同一量级（预算一样、算的 token 一样）
    assert abs(a.steps - b.steps) / max(a.steps, b.steps) < 0.25, (a.steps, b.steps)

    # 3) 无竞争时两种策略**收敛**：都 decode 优先、都按预算装箱，步数差在 10% 内
    assert abs(a.steps - b.steps) / a.steps < 0.10, (a.steps, b.steps)

    # 4) 真正稳定的差别是「KV 紧张时牺牲谁」：两种 victim 顺序不同
    rs = [Req(1, prompt=8000, out_len=50, arrive=0, computed=3000, generated=40),
          Req(2, prompt=300, out_len=50, arrive=0, computed=290, generated=1),
          Req(3, prompt=1000, out_len=50, arrive=0, computed=500, generated=20)]
    assert sglang_victim(rs).rid == 2          # 生成最少的先丢
    assert vllm_victim(rs).rid == 1            # 已算 KV 最多的先丢

    # 4) 收紧 KV 后两种策略都会出现 retract/preempt
    c = simulate("sglang", [Req(r.rid, r.prompt, r.out_len, r.arrive) for r in load], 2048, 12_000, 64)
    d = simulate("vllm", [Req(r.rid, r.prompt, r.out_len, r.arrive) for r in load], 2048, 12_000, 64)
    assert c.retracts > 0 and d.retracts > 0, (c.retracts, d.retracts)
    assert len(c.ttft) == len(load) and len(d.ttft) == len(load)

    # 5) 预算越大，总步数越少
    e = simulate("sglang", [Req(r.rid, r.prompt, r.out_len, r.arrive) for r in load], 4096, 200_000, 64)
    assert e.steps <= a.steps

    # 6) 确定性
    f = simulate("sglang", [Req(r.rid, r.prompt, r.out_len, r.arrive) for r in load], 2048, 200_000, 64)
    assert f.ttft == a.ttft and f.itl == a.itl and f.steps == a.steps

    print("scheduler_sim: all tests passed")
    report(load, 2048, 40_000, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--requests", type=int, default=120)
    ap.add_argument("--budget", type=int, default=2048, help="单步 token 预算")
    ap.add_argument("--kv", type=int, default=40_000, help="KV 容量（token）")
    ap.add_argument("--max-running", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.test:
        _test()
        return
    report(make_load(args.requests, args.seed), args.budget, args.kv, args.max_running)


if __name__ == "__main__":
    main()
