#!/usr/bin/env python3
"""P52-1 实验：DeepEP 的确定性 dispatch 排序 `EPHandle.deterministic_sort`。

为什么 H100(sm90) 可跑真实代码：
  * `deterministic_sort`（`deep_ep/buffers/ep.py`）是**纯 Python + torch.index_select/torch.sort**，
    完全不碰 NCCL / 对称显存 / kernel；它作用在 dispatch 已经落到本地的张量上。
  * 本机 `import deep_ep` 会在包 `__init__` 的 `check_nccl_so()` 处拒绝（PyTorch 链 NCCL 2.29.7
    vs 期望 2.32.3，见 ROADMAP「阻塞」）。因此这里**不 import 包**，改用
    `importlib.util.find_spec` 定位已安装的 `ep.py`，用 `ast` **逐行原样抽取真实的
    `EPHandle.deterministic_sort`**（打印其 sha256），再绑到一个只含必需字段的 shim 上运行。
  * 独立 host oracle 用「按 (expert, src_token_global_idx) 双键排序 + 槽位指针反置换」的朴素定义
    复刻同一语义，与真实方法逐行对拍。

运行：
  bash code/mega-moe/harness/exp_deepep_deterministic_sort.sh \
      > code/mega-moe/harness/exp_deepep_deterministic_sort.out.txt
"""
import ast
import hashlib
import importlib.util
import os
import sys
import textwrap
import types

import torch


# --------------------------------------------------------------------------------------
# 载入真实方法（不 import 包，绕过 NCCL 版本检查）
# --------------------------------------------------------------------------------------
def load_real_method():
    spec = importlib.util.find_spec("deep_ep")
    assert spec is not None, "deep_ep not installed"
    path = os.path.join(os.path.dirname(spec.origin), "buffers", "ep.py")
    src = open(path).read()
    tree = ast.parse(src)
    seg = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EPHandle":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "deterministic_sort":
                    seg = ast.get_source_segment(src, fn)
    assert seg is not None, "deterministic_sort not found"
    body = textwrap.dedent(seg)
    ns = {}
    exec("import torch\nfrom typing import Optional\n" + body, ns)  # noqa: S102 — 运行的是被测对象的真实源码
    return ns["deterministic_sort"], seg, path


DETERMINISTIC_SORT, METHOD_SRC, METHOD_PATH = load_real_method()


class ShimHandle:
    """只含 `deterministic_sort` 需要的字段；方法体由真实源码绑定。"""

    deterministic_sort = DETERMINISTIC_SORT

    def __init__(self, do_expand, recv_src_metadata, psum_num_recv_tokens_per_expert,
                 num_unaligned_recv_tokens_per_expert, psum_last, cached_metadata=None):
        self.do_expand = do_expand
        self.recv_src_metadata = recv_src_metadata
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        # `psum_num_recv_tokens_per_scaleup_rank[-1]` 仅取最后一个元素（总接收 token 数）
        self.psum_num_recv_tokens_per_scaleup_rank = torch.tensor([psum_last], dtype=torch.int64)
        self.cached_recv_src_metadata_before_sort = cached_metadata


def call_sort(handle, do_cpu_sync, is_cached_dispatch, recv_x, recv_sf,
              recv_topk_idx, recv_topk_weights, channel_linked_list):
    bound = types.MethodType(DETERMINISTIC_SORT, handle)
    bound(do_cpu_sync, is_cached_dispatch, recv_x, recv_sf,
          recv_topk_idx, recv_topk_weights, channel_linked_list)


def sha16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# 独立 oracle（朴素定义）
# --------------------------------------------------------------------------------------
def oracle_nonexpand(src_metadata, topk, N_valid):
    """非 expand：按 col0（src token 全局 idx）升序；OOB 行（>=N_valid）排最后。"""
    keys = src_metadata[:, 0].clone()
    idx = torch.arange(keys.numel())
    oob = idx >= N_valid
    keys = keys.clone()
    keys[oob] = torch.iinfo(keys.dtype).max
    order = torch.argsort(keys, stable=True)
    return order, oob


def oracle_expand(recv_src_metadata, psum_per_expert, num_unaligned, M):
    """expand：先按专家块、块内按 src idx 升序；有效槽在填充槽之前。

    返回 (order, valid_mask, expert_of_row)。
    """
    expert_start = psum_per_expert - num_unaligned
    token_idx2expert = torch.bucketize(torch.arange(M, device=recv_src_metadata.device),
                                       expert_start[1:], right=True, out_int32=False)
    slots = recv_src_metadata[:, 2:]
    src_global = recv_src_metadata[:, 0]
    valid = slots >= 0
    # 双键键值，与真实方法同构但不复用其代码：expert*BIG + (-BIG/2 + src)
    BIG = 10_000_000_000
    keys = token_idx2expert.to(torch.int64) * BIG
    src64 = src_global.unsqueeze(1).expand_as(slots).to(torch.int64)
    contrib = (-(BIG // 2) + src64)[valid]
    keys.scatter_add_(0, slots[valid].to(torch.int64), contrib)
    order = torch.argsort(keys, stable=True)
    return order, valid, token_idx2expert


def reverse_perm(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device)
    return inv


# --------------------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------------------
def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def case_nonexpand(shuffle_seed=None, num_recv_rows=14, k=2, H=6, num_valid=14):
    print("\n== 用例 A：非 expand + 非 cached（'排序即按 src 索引规范化'）==")
    torch.manual_seed(10)
    N = num_recv_rows
    # col0 = src token 全局 idx（唯一）、col1 = src rank、cols2: = 目标缓冲槽位
    src_idx = torch.arange(100, 100 + N, dtype=torch.int32)
    src_rank = torch.randint(0, 4, (N,), dtype=torch.int32)
    slots = torch.arange(N).reshape(1, -1).repeat(N, 1)  # 占位（非 expand 不用于排序键）
    meta = torch.stack([src_idx, src_rank], dim=1)
    meta = torch.cat([meta, slots[:, :k]], dim=1).contiguous()

    x = torch.arange(N, dtype=torch.float32).reshape(N, 1).repeat(1, H).clone()
    topk_idx = torch.arange(N, dtype=torch.int64).reshape(N, 1).repeat(1, k).clone()
    topk_w = torch.randn(N, k).float().clone()
    cll = torch.full((N,), -1, dtype=torch.int32)

    x0, ti0, tw0, meta0 = x.clone(), topk_idx.clone(), topk_w.clone(), meta.clone()

    order, oob = oracle_nonexpand(meta0, k, num_valid)

    # 模拟「接收顺序不确定性」：对物理行做一次置换
    perm = None
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        perm = torch.randperm(N, generator=g)
        x, ti0, tw0, meta, cll = x[perm].contiguous(), ti0[perm].contiguous(), tw0[perm].contiguous(), \
            meta[perm].contiguous(), cll[perm].contiguous()
        # 置换后 src 索引到达顺序改变；oracle 需要在置换后的物理行上算
        order, oob = oracle_nonexpand(meta.clone(), k, num_valid)

    h = ShimHandle(False, meta, torch.tensor([0]), torch.tensor([0]), num_valid)
    call_sort(h, do_cpu_sync=True, is_cached_dispatch=False, recv_x=x, recv_sf=None,
              recv_topk_idx=ti0, recv_topk_weights=tw0, channel_linked_list=cll)

    ok = True
    # 1) recv_x 按 order 重排、且为原序置换
    exp_x = (x0 if perm is None else x0[perm])[order]
    ok &= report("recv_x == 原张量[order]", torch.equal(x, exp_x))
    # 2) recv_src_metadata 已按 order 置换
    exp_meta = (meta0 if perm is None else meta0[perm])[order]
    ok &= report("recv_src_metadata 已置换且 col0 非降", torch.equal(h.recv_src_metadata, exp_meta)
                 and torch.all(h.recv_src_metadata[:-1, 0] <= h.recv_src_metadata[1:, 0]))
    # 3) topk_idx / weights 一致
    ok &= report("recv_topk_idx / weights 同步置换",
                 torch.equal(ti0, (topk_idx if perm is None else topk_idx[perm])[order])
                 and torch.allclose(tw0, (topk_w if perm is None else topk_w[perm])[order]))
    # 4) cached 快照保留排序前元数据
    ok &= report("cached_recv_src_metadata_before_sort 为排序前副本",
                 torch.equal(h.cached_recv_src_metadata_before_sort, meta0 if perm is None else meta0[perm]))

    if perm is None:
        # 5) 幂等：再排一次不变（此时 recv_src_metadata 已有序，键已规范化）
        snap = x.clone()
        h2 = ShimHandle(False, h.recv_src_metadata.clone(), torch.tensor([0]), torch.tensor([0]),
                        num_valid)
        call_sort(h2, True, False, x, None, ti0, tw0, None)
        ok &= report("幂等（二次排序不变）", torch.equal(x, snap))
    return ok


def case_nonexpand_canonical():
    print("\n== 用例 B：不同接收顺序 → 同一规范化输出（确定性核心）==")
    torch.manual_seed(10)
    k, H = 2, 6
    N = 14
    src_idx = torch.arange(100, 100 + N, dtype=torch.int32)
    src_rank = torch.randint(0, 4, (N,), dtype=torch.int32)
    meta = torch.cat([torch.stack([src_idx, src_rank], 1), torch.zeros(N, k, dtype=torch.int32)], 1)
    x0 = torch.arange(N, dtype=torch.float32).reshape(N, 1).repeat(1, H)
    ti0 = torch.arange(N, dtype=torch.int64).reshape(N, 1).repeat(1, k)
    tw0 = torch.randn(N, k).float()

    outs = []
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        x, ti, tw, meta_p = x0[perm].clone(), ti0[perm].clone(), tw0[perm].clone(), meta[perm].clone()
        h = ShimHandle(False, meta_p, torch.tensor([0]), torch.tensor([0]), N)
        call_sort(h, True, False, x, None, ti, tw, None)
        outs.append((x, ti, tw))
    same_x = all(torch.equal(outs[0][0], o[0]) for o in outs)
    same_ti = all(torch.equal(outs[0][1], o[1]) for o in outs)
    same_tw = all(torch.equal(outs[0][2], o[2]) for o in outs)
    # 规范输出应等于「按 src 索引升序」
    ref_x = x0[torch.argsort(src_idx, stable=True)]
    ok = report("6 种接收顺序输出逐位一致", same_x and same_ti and same_tw)
    ok &= report("规范输出 == 按 src 索引升序", torch.equal(outs[0][0], ref_x))
    return ok


def case_nonexpand_oob():
    print("\n== 用例 C：非 expand + OOB 尾块（do_cpu_sync=False）==")
    torch.manual_seed(11)
    k, H = 2, 4
    N, valid = 12, 9
    src_idx = torch.arange(200, 200 + N, dtype=torch.int32)
    meta = torch.cat([torch.stack([src_idx, torch.zeros(N, dtype=torch.int32)], 1),
                      torch.zeros(N, k, dtype=torch.int32)], 1)
    x = torch.arange(N, dtype=torch.float32).reshape(N, 1).repeat(1, H).clone()
    ti = torch.zeros(N, k, dtype=torch.int64)
    tw = torch.zeros(N, k)
    h = ShimHandle(False, meta.clone(), torch.tensor([0]), torch.tensor([0]), valid)
    call_sort(h, do_cpu_sync=False, is_cached_dispatch=False, recv_x=x, recv_sf=None,
              recv_topk_idx=ti, recv_topk_weights=tw, channel_linked_list=None)
    order, oob = oracle_nonexpand(meta, k, valid)
    ok = report("OOB 行被压到末尾且内部顺序正确",
                torch.equal(x[:, 0], torch.arange(N, dtype=torch.float32)[order]))
    ok &= report("末尾恰为 OOB 行集合", set(order[valid:].tolist()) == set(torch.nonzero(oob).flatten().tolist()))
    return ok


def build_expand(num_experts=4, topk=2, per_expert=(3, 5, 2, 4), align=4, H=4):
    """构造 expand 模式的 recv_src_metadata / psum / 未对齐计数。

    psum[i] = align(前面各组对齐量之和) + 本组未对齐实际数（与 EPHandle 文档一致）。
    """
    num_unaligned = torch.tensor(list(per_expert), dtype=torch.int64)
    aligned = [((c + align - 1) // align) * align for c in per_expert]
    psum = []
    acc = 0
    for i, c in enumerate(per_expert):
        psum.append(acc + c)          # 本组起始 + 实际数
        acc += aligned[i]
    psum = torch.tensor(psum, dtype=torch.int64)
    M = acc
    return psum, num_unaligned, aligned, M


def case_expand(shuffle_seed=None, cached=False):
    tag = "cached" if cached else "non-cached"
    print(f"\n== 用例 D：expand + {tag} ==")
    torch.manual_seed(12)
    num_experts, topk, H = 4, 2, 4
    psum, num_unaligned, aligned, M = build_expand(H=H)
    # 构造 N 个接收 token 到各专家的槽位
    N = 11
    slots = torch.full((N, topk), -1, dtype=torch.int32)
    src_idx = torch.arange(300, 300 + N, dtype=torch.int32)
    # 用「每个专家在其对齐块内」的偏移分配槽位
    expert_start = psum - num_unaligned
    cursor = {e: int(expert_start[e].item()) for e in range(num_experts)}
    for t in range(N):
        for j in range(topk):
            e = (t + j) % num_experts
            if cursor[e] < int(psum[e].item()):
                slots[t, j] = cursor[e]
                cursor[e] += 1
    meta = torch.cat([torch.stack([src_idx, torch.zeros(N, dtype=torch.int32)], 1), slots], 1).contiguous()

    # 真实 handle 的字段都在 GPU 上（方法内部用 device='cuda' 造 arange）
    DEV = "cuda"
    psum, num_unaligned = psum.to(DEV), num_unaligned.to(DEV)
    meta, slots = meta.to(DEV), slots.to(DEV)

    # 展开张量的内容 = 槽位号（便于追踪置换）
    x = torch.full((M, 1), -1.0).repeat(1, H)
    for t in range(N):
        for j in range(topk):
            s = int(slots[t, j])
            if s >= 0:
                x[s, :] = s
    ti = torch.full((M, topk), -1, dtype=torch.int64)
    tw = torch.zeros(M, topk)
    # 给每个槽一个唯一权重 tag，便于验证展开权重同步置换
    for s in range(M):
        tw[s, 0] = float(s)
    x, ti, tw = x.to(DEV), ti.to(DEV), tw.to(DEV)
    x0, meta0, tw0 = x.clone(), meta.clone(), tw.clone()

    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        p = torch.randperm(N, generator=g)
        meta = meta[p].clone()

    cached_meta = None if not cached else meta0.clone()  # cached 表示这是之前 non-cached 存下的
    h = ShimHandle(True, meta.clone(), psum, num_unaligned, int(psum[-1].item()), cached_metadata=cached_meta)

    if cached:
        # cached：copy epilogue 已按 metadata 摆放，排序不改变任何展开数组
        xc, tic, twc = x.clone(), ti.clone(), tw.clone()
        m_before = meta.clone()
        call_sort(h, True, True, x, None, ti, tw, None)
        ok = report("cached：展开数组与元数据均不变",
                    torch.equal(x, xc) and torch.equal(ti, tic) and torch.equal(tw, twc)
                    and torch.equal(h.recv_src_metadata, m_before))
        return ok

    order, valid, expert_of = oracle_expand(meta0, psum, num_unaligned, M)
    inv = reverse_perm(order)
    call_sort(h, True, False, x, None, ti, tw, None)

    ok = True
    # 1) 每个专家块内，有效槽按 src idx 升序
    new_expert_of = expert_of[order]
    mono = True
    for e in range(num_experts):
        rows = [i for i in range(M) if int(new_expert_of[i]) == e]
        tags = x[rows, 0].tolist()
        # 有效槽（tag>=0）应严格递增；填充槽（tag=-1）在末尾
        valid_tags = [v for v in tags if v >= 0]
        mono &= valid_tags == sorted(valid_tags)
        mono &= all(v == -1 for v in tags[len(valid_tags):])
    ok &= report("专家块内有效槽按 src 升序、填充槽殿后", mono)
    # 2) recv_x 是原张量按 order 重排
    ok &= report("recv_x == 原张量[order]", torch.equal(x, x0[order]))
    ok &= report("recv_topk_weights 同步置换", torch.equal(tw, tw0[order]))
    # 3) 槽位指针：新槽值 = inv[旧槽值]
    slots_new = h.recv_src_metadata[:, 2:]
    old_slots = meta0[:, 2:]
    exp_slots = old_slots.clone()
    exp_slots[valid] = inv[old_slots[valid]].to(exp_slots.dtype)
    ok &= report("recv_src_metadata[:,2:] 槽位指针已反置换", torch.equal(slots_new, exp_slots))
    # 4) 指针自洽：对有效行，new_slot 处的 tag 正是该行的元数据来源？用一致性检查：
    #    inv[old_slot] 应对应新位置，且该新位置的 tag == old_slot
    consistent = True
    for t in range(N):
        for j in range(topk):
            os_ = int(old_slots[t, j])
            if os_ >= 0:
                ns = int(slots_new[t, j])
                consistent &= (ns == int(inv[os_])) and (int(x[ns, 0]) == os_)
    ok &= report("指针指向新位置且该位置内容为原槽", consistent)
    return ok


def case_channel_linked_list():
    print("\n== 用例 E：非 expand + channel_linked_list 反置换 ==")
    torch.manual_seed(13)
    k, H = 2, 3
    N = 10
    src_idx = torch.arange(500, 500 + N, dtype=torch.int32)
    meta = torch.cat([torch.stack([src_idx, torch.zeros(N, dtype=torch.int32)], 1),
                      torch.zeros(N, k, dtype=torch.int32)], 1)
    x = torch.arange(N, dtype=torch.float32).reshape(N, 1).repeat(1, H).clone()
    ti = torch.zeros(N, k, dtype=torch.int64)
    tw = torch.zeros(N, k)
    cll = torch.tensor([2, -1, 7, 0, -1, 5, 9, -1, 1, 3], dtype=torch.int32)
    cll0 = cll.clone()
    order, _ = oracle_nonexpand(meta, k, N)
    inv = reverse_perm(order)
    h = ShimHandle(False, meta.clone(), torch.tensor([0]), torch.tensor([0]), N)
    call_sort(h, True, False, x, None, ti, tw, cll)
    valid = (cll0 >= 0) & (cll0 < N)
    exp = cll0.clone()
    exp[valid] = inv[cll0[valid]].to(torch.int32)
    ok = report("channel_linked_list 按逆置换更新、越界项跳过", torch.equal(cll, exp))
    return ok


def case_expand_canonical():
    print("\n== 用例 F：expand 模式下不同接收顺序 → 同一规范化展开顺序 ==")
    torch.manual_seed(21)
    DEV = "cuda"
    num_experts, topk, H = 4, 2, 4
    psum, num_unaligned, aligned, M = build_expand(H=H)
    psum, num_unaligned = psum.to(DEV), num_unaligned.to(DEV)
    N = 11
    src_idx = torch.arange(700, 700 + N, dtype=torch.int32)
    slots = torch.full((N, topk), -1, dtype=torch.int32)
    expert_start = psum - num_unaligned
    cursor = {e: int(expert_start[e].item()) for e in range(num_experts)}
    for t in range(N):
        for j in range(topk):
            e = (t + j) % num_experts
            if cursor[e] < int(psum[e].item()):
                slots[t, j] = cursor[e]
                cursor[e] += 1
    meta = torch.cat([torch.stack([src_idx, torch.zeros(N, dtype=torch.int32)], 1), slots], 1).to(DEV)
    x = torch.full((M, H), -1.0, device=DEV)
    for t in range(N):
        for j in range(topk):
            s = int(slots[t, j])
            if s >= 0:
                x[s, :] = s
    tw = torch.zeros(M, topk, device=DEV)
    for s in range(M):
        tw[s, 0] = float(s)
    x0, tw0 = x.clone(), tw.clone()

    outs = []
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        p = torch.randperm(N, generator=g)
        meta_p = meta[p].clone()
        xc, twc = x0.clone(), tw0.clone()
        h = ShimHandle(True, meta_p, psum, num_unaligned, int(psum[-1].item()))
        call_sort(h, True, False, xc, None, torch.zeros(M, topk, dtype=torch.int64, device=DEV), twc, None)
        outs.append((xc, twc))
    ok = report("6 种接收顺序展开输出逐位一致",
                all(torch.equal(outs[0][0], o[0]) for o in outs)
                and all(torch.equal(outs[0][1], o[1]) for o in outs))
    return ok


def main():
    print("=" * 78)
    print("P52-1 DeepEP `EPHandle.deterministic_sort` —— 真实源码逐行抽取 + 独立 oracle 对拍")
    print("=" * 78)
    print(f"  device            : {torch.cuda.get_device_name(0)}")
    print(f"  torch             : {torch.__version__}")
    print(f"  deep_ep ep.py     : {METHOD_PATH}")
    print(f"  deterministic_sort: {len(METHOD_SRC)} chars, sha256={sha16(METHOD_SRC)}")

    results = []
    results.append(case_nonexpand())
    results.append(case_nonexpand_canonical())
    results.append(case_nonexpand_oob())
    results.append(case_expand(cached=False))
    results.append(case_expand(cached=True))
    results.append(case_channel_linked_list())
    results.append(case_expand_canonical())

    print("\n" + "=" * 78)
    npass = sum(1 for r in results if r)
    print(f"TOTAL: {npass} PASS / {len(results) - npass} FAIL")
    print("=" * 78)
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
