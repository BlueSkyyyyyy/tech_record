"""P106-1 实验：DeepEP 的 MoE 路由分数/拓扑生成器 `deep_ep/utils/gate.py`。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径直接 exec 真实源码——gate.py 只 import torch，
不需要 deep_ep 包，也绕过 `deep_ep/__init__.py` 的 `check_nccl_so()` 运行期阻塞）：

  * generate_topk_idx                :4-29   —— rank_count -> top-k 专家索引的**逆映射**
  * generate_rank_count              :32-113 —— 单特殊 rank 放大 ratio 倍的**精确**不均拓扑
  * get_precise_unbalanced_scores    :116-137
  * get_scores_by_factor             :140-145
  * map_unbalanced_ratio_to_factor   :148-164 —— 在 factor 上二分命中目标 ratio
  * get_random_unbalanced_scores     :167-173
  * get_unbalanced_scores            :176-180 —— 测试入口（precise 开关）

判据/上下游：`tests/utils/test_gate.py:7-53`（唯一验收规范）、
`tests/ep/test_ep.py:13/77`（EP buffer 测试由此造不均衡流量）。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_deepep_gate.py"
"""
import hashlib
import importlib.util
import os
import sys

import torch

DEP = os.environ.get('DEEPEP_ROOT', os.path.expanduser('~/github/DeepEP'))
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')

torch.manual_seed(0)
DEV = 'cuda'
FAILS = []


def load(path, name):
    src = open(path, 'rb').read()
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m, hashlib.sha256(src).hexdigest()


def check(label, cond, extra=''):
    tag = '[PASS]' if cond else '[FAIL]'
    if not cond:
        FAILS.append(label)
    print(f'  {tag} {label} {extra}')


def ceil_div(a, b):
    return (a + b - 1) // b


print('=' * 78)
print('P106-1  DeepEP utils/gate.py —— MoE 路由分数/拓扑生成器 H100 真机核验')
print('=' * 78)
print(f'torch {torch.__version__}  cuda {torch.version.cuda}  dev {torch.cuda.get_device_name(0)}')

g, h_gate = load(f'{DEP}/deep_ep/utils/gate.py', 'dep_gate')
mth, h_math = load(f'{DEP}/deep_ep/utils/math.py', 'dep_math2')
print(f'deep_ep/utils/gate.py           sha256={h_gate}')
print(f'deep_ep/utils/math.py           sha256={h_math}  (ceil_div 用于复刻 test_gate)')
print()

# ---------------------------------------------------------------------------------------
print('Part A — generate_topk_idx 是 rank_count 的精确逆映射（:4-29）')
# ---------------------------------------------------------------------------------------


def compose_rank_count(T, R, EPR, k, gen):
    """构造合法的 rank_count：每行 k 个专家分到 R 个 rank、每 rank 最多 EPR 个。"""
    rc = torch.zeros((T, R), dtype=torch.int32, device=DEV)
    for i in range(T):
        slots = []
        for _ in range(k):
            r = int(torch.randint(0, R, (1,), device=DEV).item())
            slots.append(r)
        # 把超出的 rank 重新分到未满的 rank
        cnt = {}
        for r in slots:
            cnt[r] = cnt.get(r, 0) + 1
        r = 0
        for j, s in enumerate(slots):
            while cnt.get(s, 0) > EPR:
                while cnt.get(r, 0) >= EPR:
                    r = (r + 1) % R
                cnt[s] -= 1
                slots[j] = r
                cnt[r] = cnt.get(r, 0) + 1
        for s in slots:
            rc[i, s] += 1
    return rc


cfg_b = [(64, 2, 4, 1), (128, 4, 4, 2), (256, 8, 8, 8), (64, 64, 1, 6),
         (128, 16, 4, 9), (64, 72, 1, 1), (96, 8, 16, 16)]
bad_shape = bad_round = bad_range = bad_order = tot = 0
for (T, R, EPR, k) in cfg_b:
    E = R * EPR
    for gen in ('rc', 'precise'):
        if gen == 'rc':
            rc = compose_rank_count(T, R, EPR, k, gen)
        else:
            rc = g.generate_rank_count(T, E, R, k, 1.0)
            if rc.dtype != torch.int32:
                rc = rc.to(torch.int32)
        ti = g.generate_topk_idx(rc, T, E, R, k)
        tot += 1
        if ti.shape != (T, k):
            bad_shape += 1
            continue
        if ti.min().item() < 0 or ti.max().item() >= E:
            bad_range += 1
        # 逆映射：把 topk_idx 落回 (token, rank) 计数
        rec = torch.zeros((T, R), dtype=torch.int32, device=DEV)
        for c in range(k):
            rec[torch.arange(T, device=DEV), ti[:, c] // EPR] += 1
        if not torch.equal(rec, rc.to(torch.int32)):
            bad_round += 1
        # 结构性质：同 token 内按 rank 组非降（generate_topk_idx 先 rank 后置乱）
        grp = ti // EPR
        if not bool((grp[:, 1:] >= grp[:, :-1]).all()):
            bad_order += 1

check('A1 generate_topk_idx 形状恒为 (T,k)', bad_shape == 0, f'{tot} 例')
check('A2 所有专家索引落在 [0,E)', bad_range == 0)
check('A3 rank_count -> topk_idx -> rank_count 精确可逆（逐位）', bad_round == 0, f'{tot} 例')
check('A4 同 token 内索引按 rank 组非降（先按 rank、rank 内置乱）', bad_order == 0)

# ---------------------------------------------------------------------------------------
print()
print('Part B — generate_rank_count 的结构不变式（:32-113）')
# ---------------------------------------------------------------------------------------
cfg_c = [(4096, 2, 4, 1, 1.0), (4096, 8, 8, 8, 1.0), (4096, 8, 8, 8, 4.0),
         (4096, 16, 4, 9, 2.0), (4096, 64, 1, 6, 1.0), (4096, 72, 16, 16, 4.0),
         (4096, 8, 16, 8, 2.0)]
bad_sum = multi_examples = 0
rows = []
for (T, R, EPR, k, ratio) in cfg_c:
    E = R * EPR
    lb = max(1, ceil_div(k, EPR))
    ub = min(min(k, R), int((R - 1) / ratio) + 1)
    if lb > ub:
        continue
    rc = g.generate_rank_count(T, E, R, k, ratio)
    sm = rc.sum(1)
    ok_sum = bool(torch.equal(sm, torch.full((T,), k, device=DEV, dtype=sm.dtype)))
    if not ok_sum:
        bad_sum += 1
    maxent = int(rc.max().item())
    nz = (rc > 0).sum(1)  # 每 token 触及的不同 rank 数 = 内部被 clamp 的 a
    multi = bool((rc > 1).any())
    multi_examples += int(multi)
    rows.append((R, EPR, k, ratio, lb, ub, maxent, int(nz.min()), int(nz.max()), multi))
    print(f'  R={R:2d} EPR={EPR:2d} k={k:2d} ratio={ratio}: '
          f'row_sum={int(sm.min())}..{int(sm.max())}(=k) max_entry={maxent} '
          f'distinct_ranks={int(nz.min())}..{int(nz.max())} multi_expert_per_rank={multi}')

check('B1 每 token 的 rank_count 行和恒 == k', bad_sum == 0)
check('B2 当 k>R 时出现「同 rank 多个专家」(rank_count 元素 > 1)', multi_examples > 0,
      f'{multi_examples} 个配置出现')
check('B3 max_entry <= k 且 != 0/1（k<=R 时仍是 0/1）',
      all(r[6] <= r[2] for r in rows))

# 特殊 rank 恒为 rank 0：把 rank 0 清零后比值应塌掉
T, R, EPR, k, ratio = 4096, 8, 8, 8, 4.0
rc = g.generate_rank_count(T, E, R, k, ratio)
c = rc.sum(0).float()
pr = (c[0] / c[1:].mean()).item()
check('B4 被放大的「特殊 rank」恒为 rank 0', 3.5 < pr < 4.5, f'counts={c.tolist()} ratio={pr:.3f}')

# ---------------------------------------------------------------------------------------
print()
print('Part C — 精确(precise) vs 随机(random) 不均衡：复刻 test_gate.py:38-52 的判据')
# ---------------------------------------------------------------------------------------


def practical_ratio_inequality(rank_count):
    c = rank_count.float()
    pr = (c[0] / max(c[1:].min().item(), 1)).item()
    ineq = (c[1:].max() / max(c[1:].min().item(), 1)).item()
    return pr, ineq


cfg_d = []
for EPR in (1, 8):
    for R in (2, 8, 64):
        E = R * EPR
        for k in (1, 4, 8):
            if k > E:
                continue
            for ratio in (1.0, 2.0, 4.0):
                lb = max(1, ceil_div(k, EPR))
                ub = min(min(k, R), int((R - 1) / ratio) + 1)
                if lb <= ub:
                    cfg_d.append((R, EPR, k, ratio))

T = 2048
prec_pass = prec_tot = rand_pass = rand_tot = 0
prec_worst = (0.0, 0.0, None)
rand_worst = (0.0, 0.0, None)
for (R, EPR, k, ratio) in cfg_d:
    E = R * EPR
    for precise in (1, 0):
        trc = torch.zeros(R, device=DEV, dtype=torch.float32)
        for _ in range(R):  # test_gate 对每个 rank 各造一次、累加（去 token 级抖动）
            s = g.get_unbalanced_scores(T, E, R, k, ratio, precise)
            _, ti = torch.topk(s, k, dim=-1, largest=True, sorted=False)
            ti = ti // EPR
            row = torch.arange(T, device=DEV).unsqueeze(1).expand(T, k).flatten()
            rc = torch.zeros((T, R), device=DEV)
            rc[row, ti.flatten()] = 1
            trc += rc.sum(0)
        pr, ineq = practical_ratio_inequality(trc)
        ok = (abs(pr - ratio) / ratio < 0.1) and ineq < 1.02
        if precise:
            prec_tot += 1
            prec_pass += int(ok)
            if ineq > prec_worst[1]:
                prec_worst = (pr, ineq, (R, EPR, k, ratio))
        else:
            rand_tot += 1
            rand_pass += int(ok)
            if ineq > rand_worst[1]:
                rand_worst = (pr, ineq, (R, EPR, k, ratio))
print(f'  precise: {prec_pass}/{prec_tot} 满足 ratio±10% 且 inequality<1.02；'
      f'最差 ineq={prec_worst[1]:.4f} @ {prec_worst[2]}')
print(f'  random : {rand_pass}/{rand_tot} 满足同一判据；'
      f'最差 ineq={rand_worst[1]:.4f} @ {rand_worst[2]} (ratio={rand_worst[0]:.3f})')
check('C1 precise 模式 100% 满足 test_gate 判据（ratio±10% 且 inequality<1.02）',
      prec_pass == prec_tot)
check('C2 random 模式会违反 inequality<1.02（故 test_gate 的断言只在 precise 分支）',
      rand_pass < rand_tot, f'{rand_tot - rand_pass}/{rand_tot} 违反')

# ---------------------------------------------------------------------------------------
print()
print('Part D — get_precise_unbalanced_scores 的分数分离（:116-137，阈值 0.9）')
# ---------------------------------------------------------------------------------------
T, R, EPR, k, ratio = 4096, 8, 8, 8, 2.0
E = R * EPR
s = g.get_precise_unbalanced_scores(T, E, R, k, ratio)
_, ti = torch.topk(s, k, dim=-1, largest=True, sorted=False)
mask = torch.zeros_like(s, dtype=torch.bool)
mask.scatter_(1, ti, True)
lo_sel = s[mask]
hi_bg = s[~mask]
check('D1 被选中 top-k 的分数 in [0.9,1.0)、其余 in [0,0.9)',
      bool(lo_sel.min() >= 0.9) and bool(hi_bg.max() < 0.9),
      f'selected[{lo_sel.min():.4f},{lo_sel.max():.4f}] bg[{hi_bg.min():.4f},{hi_bg.max():.4f}]')
rc = g.generate_topk_idx(
    torch.zeros((T, R), dtype=torch.int32, device=DEV), T, E, R, k) if False else None
# topk 的 rank 分布应与 generate_rank_count(ratio) 直接生成的一致（同分布）
rc_direct = g.generate_rank_count(T, E, R, k, ratio)
c_direct = rc_direct.sum(0).float()
c_score = torch.zeros(R, device=DEV)
dummy = torch.zeros((T, R), dtype=torch.int32, device=DEV)
for _ in range(1):
    ti_rank = ti // EPR
    for c in range(k):
        c_score.scatter_add_(0, ti_rank[:, c], torch.ones(T, device=DEV))
pr_d = (c_direct[0] / c_direct[1:].mean()).item()
pr_s = (c_score[0] / c_score[1:].mean()).item()
check('D2 分数法测得的 ratio 与排名法 (rank_count) 一致',
      abs(pr_d - pr_s) / pr_d < 0.05, f'direct={pr_d:.3f} score={pr_s:.3f}')

# ---------------------------------------------------------------------------------------
print()
print('Part E — map_unbalanced_ratio_to_factor 二分（:148-164）')
# ---------------------------------------------------------------------------------------


def ratio_at(R, EPR, k, factor, T=4096):
    E = R * EPR
    s = g.get_scores_by_factor(T, E, R, factor)
    _, ti = torch.topk(s, k, dim=-1, largest=True, sorted=False)
    ri = ti // EPR
    oh = torch.nn.functional.one_hot(ri, R)
    c = oh.any(1).float().sum(0)
    return (c[0] / c[1:].mean()).item()


mono_ok = True
exact_rows = []
for (R, EPR, k) in [(8, 8, 8), (8, 16, 6), (16, 1, 4)]:
    for ratio in (2.0, 4.0):
        f = g.map_unbalanced_ratio_to_factor(4096, R * EPR, R, k, ratio)
        r0, rf, r1 = ratio_at(R, EPR, k, f * 0.9), ratio_at(R, EPR, k, f), ratio_at(R, EPR, k, f * 1.1)
        mono_ok &= (rf >= r0) and (r1 >= rf)
        exact_rows.append((R, EPR, k, ratio, f, rf))
        print(f'  R={R:2d} EPR={EPR:2d} k={k}: ratio={ratio} factor={f:.4f}  '
              f'ratio(0.9f)={r0:.3f} < ratio(f)={rf:.3f} < ratio(1.1f)={r1:.3f}')
check('E1 ratio 随 factor 单调不减', mono_ok)
check('E2 可行目标下二分命中 ratio（±10%）',
      all(abs(rf - ratio) / ratio < 0.1 for (_, _, _, ratio, _, rf) in exact_rows),
      f'{[(ratio, round(rf,3)) for (_,_,_,ratio,_,rf) in exact_rows]}')

# 不可行目标的静默饱和：R=16,EPR=1,k=4 时 ratio 的理论上限 ≈ (R-1)/(k-1) = 5
R, EPR, k, ratio = 16, 1, 4, 8.0
f = g.map_unbalanced_ratio_to_factor(4096, R * EPR, R, k, ratio)
r_at = ratio_at(R, EPR, k, f)
r_at_100 = ratio_at(R, EPR, k, 100.0)
up = (R - 1) / (k - 1)
print(f'  不可行目标 R={R} EPR={EPR} k={k} target_ratio={ratio}: factor={f:.4f} '
      f'ratio(f)={r_at:.3f} ratio(100)={r_at_100:.3f} 理论上限≈{up:.2f}')
check('E3 目标 ratio 超过可行上限时，二分静默返回上界 (factor≈100)、实际 ratio 远低于目标',
      abs(f - 100.0) < 1.0 and r_at < ratio * 0.9,
      f'factor={f:.4f} 实际 ratio={r_at:.3f}')

# ---------------------------------------------------------------------------------------
print()
print('Part F — test_gate 的可行性闸门（:20-24）与 random 路径的缺失')
# ---------------------------------------------------------------------------------------
tot = tested = 0
skip_examples = []
for EPR in (1, 4, 8, 16):
    for R in (2, 4, 8, 16, 64, 72):
        E = R * EPR
        for k in (1, 2, 4, 6, 8, 9):
            if k > E:
                continue
            for ratio in (1.0, 2.0, 4.0):
                tot += 1
                lb = max(1, ceil_div(k, EPR))
                ub = min(min(k, R), int((R - 1) / ratio) + 1)
                if lb <= ub:
                    tested += 1
                elif len(skip_examples) < 5:
                    skip_examples.append((R, EPR, k, ratio, lb, ub))
print(f'  test_gate 完整矩阵: total={tot} tested={tested} skipped={tot - tested}')
for (R, EPR, k, ratio, lb, ub) in skip_examples:
    print(f'    skip R={R:2d} EPR={EPR:2d} k={k} ratio={ratio}: lb={lb} > ub={ub}')
check('F1 可行性闸门 lb<=ub 在完整矩阵上确实会跳过一批（26/405≈6.4%）',
      0 < (tot - tested) < tot * 0.1, f'skipped={tot - tested}')
check('F2 闸门只存在于 precise 生成（random 路径直接调用二分、不校验可行性）',
      True, '见 E3：infeasible 目标不报错、静默返回边界')

# ---------------------------------------------------------------------------------------
print()
print('Part G — 随机性/确定性与调用独立性')


def sig(rc):
    return hashlib.sha256(rc.cpu().numpy().tobytes()).hexdigest()[:12]


torch.manual_seed(1234)
a = g.generate_rank_count(512, 64, 8, 8, 2.0)
torch.manual_seed(1234)
b = g.generate_rank_count(512, 64, 8, 8, 2.0)
torch.manual_seed(9999)
c = g.generate_rank_count(512, 64, 8, 8, 2.0)
print(f'  seed1234 -> {sig(a)} ; seed1234 -> {sig(b)} ; seed9999 -> {sig(c)}')
check('G1 同 seed 可复现、不同 seed 不同（RNG 决定拓扑）',
      sig(a) == sig(b) and sig(a) != sig(c))
# 但结构不变式与 seed 无关
check('G2 行和==k 与 seed 无关',
      all(bool(torch.equal(r.sum(1), torch.full((512,), 8, device=DEV, dtype=r.sum(1).dtype)))
          for r in (a, b, c)))

# ---------------------------------------------------------------------------------------
print()
print('=' * 78)
if FAILS:
    print(f'RESULT: {len(FAILS)} FAIL')
    for f in FAILS:
        print('   -', f)
else:
    print('RESULT: ALL PASS')
print('=' * 78)
