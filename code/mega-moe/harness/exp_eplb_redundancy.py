#!/usr/bin/env python3
"""P43-1: DeepEP EPLB 冗余专家「预取 / 归约」的 host 侧公式与语义（H100 实测）。

本文件不 import deep_ep（本机缺 `deep_ep._C`，见 ROADMAP 阻塞）：逐行等价复刻
  * deep_ep/utils/envs.py:194-235  get_sm_read_gbs / get_sm_write_gbs / get_nvlink_gbs
  * deep_ep/utils/math.py:26-27      align
  * deep_ep/buffers/ep.py:845-852    lb_get_theoretical_num_sms 与 num_sms 的 align(.,2)
然后跑**真实 `nvidia-smi nvlink -s`** 算出目标卡上的理论 SM 数；并用纯 tensor 数值
验证 `lb_reduce_grads` 的「把各 rank 冗余副本的梯度累加回 owner」语义（与
`lb_prefetch_weights` 互为镜像）。
"""
import math
import re
import subprocess
import sys

import torch


def ceil_div(x, y):                 # deep_ep/utils/math.py:21-22
    return (x + y - 1) // y


def align(x, y):                    # deep_ep/utils/math.py:26-27
    return ceil_div(x, y) * y


def get_nvlink_gbs(factor=0.9):     # deep_ep/utils/envs.py:207-235
    try:
        out = subprocess.run(['nvidia-smi', 'nvlink', '-s'],
                             capture_output=True, text=True, check=True).stdout
        m = re.search(r'GPU \d+:.*?(?=^GPU \d+:|^$)', out, re.MULTILINE | re.DOTALL)
        assert m
        links = re.findall(r'Link \d+:\s*([\d\.]+) GB/s', m.group(0))
        assert links
        return sum(float(b) for b in links) * factor
    except Exception as e:          # noqa
        print(f'Failed to get NVLink connection speed: {e}')
        return 0


SM_READ_GBS = 180                   # envs.py:194-197
SM_WRITE_GBS = 45                   # envs.py:200-204


def lb_get_theoretical_num_sms():   # ep.py:845-852
    nvlink_gbs = get_nvlink_gbs()
    return align(math.ceil(max(nvlink_gbs / SM_READ_GBS, nvlink_gbs / SM_WRITE_GBS)), 2)


def test_num_sms():
    print('=== Part C: lb_get_theoretical_num_sms（真实 nvidia-smi nvlink）===')
    nv = get_nvlink_gbs()
    read_need = math.ceil(nv / SM_READ_GBS)
    write_need = math.ceil(nv / SM_WRITE_GBS)
    sms = lb_get_theoretical_num_sms()
    print(f'  nvlink_total(0.9x)   = {nv:.1f} GB/s')
    print(f'  ceil(nvlink/read )   = ceil({nv:.1f}/180) = {read_need}')
    print(f'  ceil(nvlink/write)   = ceil({nv:.1f}/45)  = {write_need}  <- 写侧瓶颈')
    print(f'  align(max,2)         = {sms}  (= 目标卡理论 LB SM 数)')
    assert sms % 2 == 0
    # ep.py:43/65：num_sms==0 用理论值，否则 align(num_sms,2)
    for raw in [1, 2, 3, 7, 12, 13, 0]:
        got = lb_get_theoretical_num_sms() if raw == 0 else align(raw, 2)
        print(f'  num_sms arg {raw:>2} -> {got}')
    assert align(0, 2) == 0 and align(13, 2) == 14
    print('  OK\n')
    return sms


def build_tasks(mapping, rank_idx, num_local):
    """复刻 eplb.cuh:29-64（映射顺序压缩）。mapping: [num_ranks][num_redundant]。
    返回 [(local_expert_idx, dst_nvl_rank_idx, redundant_expert_idx)]。"""
    num_redundant = len(mapping[0])
    tasks = []
    for r in range(len(mapping)):
        for c in range(num_redundant):
            e = mapping[r][c]
            if e >= 0 and e // num_local == rank_idx:
                tasks.append((e % num_local, r, c))
    return tasks


def test_redundant_grads_semantics():
    print('=== Part D: lb_reduce_grads 语义（冗余副本梯度累加回 owner）===')
    R, L, D = 4, 3, 3                       # 4 ranks × 3 local experts × 3 redundant slots
    E = R * L
    # 一个合法的冗余映射：每个 rank 的每个冗余槽放一个 LSA 域内的专家 id（-1 为空）
    mapping = [
        [0, 1, -1],
        [3, 4, 5],
        [6, -1, 7],
        [9, 10, 11],
    ]
    assert len(mapping) == R and all(len(row) == D for row in mapping)

    torch.manual_seed(0)
    # 每个专家的「本地贡献」（owner 自己算的梯度）
    owner_contrib = torch.randn(E, 8, dtype=torch.float64)
    # 每个冗余副本的贡献（holder rank 上那份拷贝算出的梯度）
    redundant_contrib = torch.randn(R, D, 8, dtype=torch.float64)

    # 参考：每个专家梯度 = owner 贡献 + 所有指向它的冗余副本贡献
    ref = owner_contrib.clone()
    for r in range(R):
        for c in range(D):
            e = mapping[r][c]
            if e >= 0:
                ref[e] += redundant_contrib[r][c]

    # 模拟 lb_reduce_grads：每个 rank 把「自己冗余槽里、属于本 rank 的副本」拉回来累加
    expert_grads = owner_contrib.clone()
    for r in range(R):
        for (local_e, dst_rank, red_idx) in build_tasks(mapping, rank_idx=r, num_local=L):
            expert_grads[r * L + local_e] += redundant_contrib[dst_rank][red_idx]

    rel = (expert_grads - ref).abs().max().item() / ref.abs().max().item()
    print(f'  模拟（build_tasks + 累加）vs 参考累加: relerr={rel:.3e}')
    assert rel < 1e-14

    # 反例：只覆盖、不累加（写成覆盖）会丢梯度
    overwrite = owner_contrib.clone()
    for r in range(R):
        for (local_e, dst_rank, red_idx) in build_tasks(mapping, rank_idx=r, num_local=L):
            overwrite[r * L + local_e] = redundant_contrib[dst_rank][red_idx]
    rel_bad = (overwrite - ref).abs().max().item() / ref.abs().max().item()
    print(f'  反例（覆盖而非累加）        : relerr={rel_bad:.3e}  <- 必须累加')

    # build_tasks 覆盖性：每个 (dst_rank, red_idx) 只会被 owner rank 领一次
    claimed = set()
    for r in range(R):
        for (local_e, dst_rank, red_idx) in build_tasks(mapping, rank_idx=r, num_local=L):
            key = (dst_rank, red_idx)
            assert key not in claimed, 'dup claim!'
            claimed.add(key)
    expected = {(r, c) for r in range(R) for c in range(D) if mapping[r][c] >= 0}
    assert claimed == expected
    print(f'  build_tasks 领走的冗余槽 {len(claimed)} 个 == 非空映射 {len(expected)} 个，无重复')
    print('  OK\n')


def main():
    print(f'torch {torch.__version__}')
    test_num_sms()
    test_redundant_grads_semantics()
    print('ALL PASS')


if __name__ == '__main__':
    sys.exit(main())
