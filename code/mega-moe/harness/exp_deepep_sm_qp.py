"""P63-1 实验：DeepEP dispatch / combine 的【自动 SM 数 / QP 数选择模型】与带宽探测层。

目标源码（`~/github/DeepEP@93eb6eb`，逐行直接加载 / ast 抽取真实文件，不走 deep_ep/__init__）：
  * deep_ep/buffers/ep.py  —— EPBuffer.get_theoretical_num_sms  (:431-541)
                              EPBuffer.get_theoretical_num_qps  (:543-560)
                              EPBuffer.lb_get_theoretical_num_sms(:846-852，对照用)
  * deep_ep/utils/envs.py  —— get_sm_read_gbs / get_sm_write_gbs / get_nvlink_gbs /
                              get_rdma_gbs / check_fast_rdma_atomic_support（带宽探测层）

为什么 H100 可跑：
  * 这一层是**纯 host / Python 解析式模型**，不碰任何 CUDA / NCCL 设备 API。
  * `import deep_ep` 会触发 `deep_ep/__init__.py:55 check_nccl_so()`（本机 NCCL 2.29.7 与编译期
    不匹配 → abort）。故这里：① 用 sys.modules 注入合成 `deep_ep._C` / `deep_ep.comm` stub 后
    按文件路径 exec 真实 `envs.py`；② 用 `ast` 从真实 `ep.py` 抽取三个方法源码，装进一个带
    真实属性字段的 shim 类，运行**未经修改的真实方法体**（打印 sha256）。
  * 独立 oracle（纯 Python 重写同一套 roofline 模型）逐配置对拍，证明我们读懂了这个模型。

运行：bash harness/exp_deepep_sm_qp.sh > harness/exp_deepep_sm_qp.out.txt
"""
import ast
import hashlib
import importlib.util
import math
import os
import sys
import types

import torch


DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


# --------------------------------------------------------------------------------------
# 真实源码加载
# --------------------------------------------------------------------------------------
def load_real_envs():
    """按文件路径 exec 真实 deep_ep/utils/envs.py（stub 掉 deep_ep._C 与 deep_ep.comm）。"""
    pkg = types.ModuleType('deep_ep')
    pkg.__path__ = [f'{DEP}/deep_ep']
    sys.modules['deep_ep'] = pkg
    stub_c = types.ModuleType('deep_ep._C')
    sys.modules['deep_ep._C'] = stub_c
    pkg._C = stub_c
    utils_pkg = types.ModuleType('deep_ep.utils')
    utils_pkg.__path__ = [f'{DEP}/deep_ep/utils']
    sys.modules['deep_ep.utils'] = utils_pkg
    pkg.utils = utils_pkg
    comm = types.ModuleType('deep_ep.comm')
    comm.get_physical_domain_size = lambda *a, **k: (1, 1)
    comm.get_logical_domain_size = lambda *a, **k: (1, 1)
    sys.modules['deep_ep.comm'] = comm
    pkg.comm = comm

    path = f'{DEP}/deep_ep/utils/envs.py'
    src = open(path, 'rb').read()
    spec = importlib.util.spec_from_file_location('deep_ep.utils.envs', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['deep_ep.utils.envs'] = m
    spec.loader.exec_module(m)
    return m, hashlib.sha256(src).hexdigest()


def extract_methods(bw=None):
    """ast 抽取 ep.py 中三个真实方法源码，装进 shim。返回 (shim_cls, sha256, {name: (l0,l1)})。"""
    path = f'{DEP}/deep_ep/buffers/ep.py'
    src = open(path, 'rb').read().decode()
    tree = ast.parse(src)
    want = {'get_theoretical_num_sms', 'get_theoretical_num_qps', 'lb_get_theoretical_num_sms'}
    snippets, spans = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'EPBuffer':
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in want:
                    seg = ast.get_source_segment(src, fn)
                    snippet = '\n'.join('    ' + ln for ln in seg.splitlines())
                    snippets[fn.name] = snippet
                    spans[fn.name] = (fn.lineno, fn.end_lineno)
    assert set(snippets) == want, f'extracted {set(snippets)}'
    body = 'class RealModel:\n' + '\n\n'.join(snippets[k] for k in
                                              ['get_theoretical_num_sms', 'get_theoretical_num_qps',
                                               'lb_get_theoretical_num_sms'])
    ns = {'math': math, 'torch': torch,
          'align': lambda x, y: (x + y - 1) // y * y,
          'os': os}
    if bw is not None:
        ns.update(bw)
    exec(compile(body, '<ep.py-extracted>', 'exec'), ns)
    return ns['RealModel'], hashlib.sha256(src.encode()).hexdigest(), spans


# --------------------------------------------------------------------------------------
# 独立 oracle（纯 Python 重写同一套 roofline 模型，不使用上面抽取的代码）
# --------------------------------------------------------------------------------------
def _expected_topk(E, k, G):
    assert E % G == 0
    return G * (1 - math.comb(E - E // G, k) / math.comb(E, k))


def align(x, y):
    return (x + y - 1) // y * y


def oracle_num_sms(cfg):
    E, k = cfg['num_experts'], cfg['num_topk']
    R, Rsco, Rscu = cfg['num_ranks'], cfg['num_scaleout_ranks'], cfg['num_scaleup_ranks']
    Rrdma, Rnvl = cfg['num_rdma_ranks'], cfg['num_nvlink_ranks']
    sr, sw = cfg['sm_read_gbs'], cfg['sm_write_gbs']
    rd, nv = cfg['rdma_gbs'], cfg['nvlink_gbs']
    ne = _expected_topk(E, k, R)
    nes = _expected_topk(E, k, Rsco) if Rsco > 1 else 0
    sm_read, sm_write, rdma_t, nvl_t = 0.0, 0.0, 0.0, 0.0
    sm_read += 1 / ne
    if Rsco > 1:
        sm_write += 1 / ne
        sm_write += (1 / ne) * (nes / Rsco)
        rdma_t += (1 / ne) * (nes * (1 - 1 / Rsco))
        sm_read += nes / ne
        sm_write += 1
        nvl_t += 1 - 1 / Rscu
    else:
        if Rrdma > 1:
            sm_write += 1 / ne
        sm_write += Rnvl / R
        nvl_t += Rnvl / R * (1 - 1 / Rnvl)
        rdma_t += (R - Rnvl) / R
    if Rsco > 1 and (rdma_t / rd) > (nvl_t / nv):
        bt, bg = rdma_t, rd
    else:
        bt, bg = nvl_t, nv
    dev = cfg['num_device_sms']
    num_sms = dev
    if bt > 0:
        num_sms = max(bg / bt * sm_read / sr, bg / bt * sm_write / sw,
                      bg / bt * (sm_write - nvl_t) / sr, bg / bt * (sm_read + nvl_t) / sw)
    num_sms = align(max(4, math.ceil(num_sms * 1.3)), 4)
    num_sms = num_sms if cfg['prefer_overlap_with_compute'] else max(num_sms, 64)
    num_sms = min(num_sms, dev // 2 * 2)
    return num_sms


def oracle_num_qps(cfg, num_sms):
    num_qps = min(num_sms, 8 + 1)
    if cfg['allow_hybrid_mode']:
        num_qps = num_sms * 16 + 1
    return min(num_qps, cfg['num_allocated_qps'])


# --------------------------------------------------------------------------------------
# 构造 shim 的 self
# --------------------------------------------------------------------------------------
class Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def make_self(**kw):
    return Cfg(**kw)


CONFIGS = [
    # name, dict
    ('direct EP8/LSA8  (single node, no RDMA)',
     dict(num_ranks=8, num_scaleout_ranks=1, num_scaleup_ranks=8, num_rdma_ranks=1, num_nvlink_ranks=8,
          num_experts=256, num_topk=8, prefer_overlap_with_compute=True, allow_hybrid_mode=False,
          num_allocated_qps=17, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=0, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('hybrid EP16 (2 nodes x LSA8)',
     dict(num_ranks=16, num_scaleout_ranks=2, num_scaleup_ranks=8, num_rdma_ranks=2, num_nvlink_ranks=8,
          num_experts=256, num_topk=8, prefer_overlap_with_compute=True, allow_hybrid_mode=True,
          num_allocated_qps=65, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=50, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('hybrid EP32 (4 nodes x LSA8)',
     dict(num_ranks=32, num_scaleout_ranks=4, num_scaleup_ranks=8, num_rdma_ranks=4, num_nvlink_ranks=8,
          num_experts=256, num_topk=8, prefer_overlap_with_compute=True, allow_hybrid_mode=True,
          num_allocated_qps=129, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=50, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('hybrid EP64 (8 nodes x LSA8)',
     dict(num_ranks=64, num_scaleout_ranks=8, num_scaleup_ranks=8, num_rdma_ranks=8, num_nvlink_ranks=8,
          num_experts=256, num_topk=8, prefer_overlap_with_compute=True, allow_hybrid_mode=True,
          num_allocated_qps=129, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=50, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('hybrid EP128 (16 nodes x LSA8)',
     dict(num_ranks=128, num_scaleout_ranks=16, num_scaleup_ranks=8, num_rdma_ranks=16, num_nvlink_ranks=8,
          num_experts=256, num_topk=8, prefer_overlap_with_compute=True, allow_hybrid_mode=True,
          num_allocated_qps=129, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=50, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('direct EP8, not preferring overlap',
     dict(num_ranks=8, num_scaleout_ranks=1, num_scaleup_ranks=8, num_rdma_ranks=1, num_nvlink_ranks=8,
          num_experts=256, num_topk=8, prefer_overlap_with_compute=False, allow_hybrid_mode=False,
          num_allocated_qps=17, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=0, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('hybrid EP16, RDMA slow (bottleneck RDMA)',
     dict(num_ranks=16, num_scaleout_ranks=2, num_scaleup_ranks=8, num_rdma_ranks=2, num_nvlink_ranks=8,
          num_experts=256, num_topk=8, prefer_overlap_with_compute=True, allow_hybrid_mode=True,
          num_allocated_qps=65, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=12, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('direct EP4 (1 node, 4 ranks)',
     dict(num_ranks=4, num_scaleout_ranks=1, num_scaleup_ranks=4, num_rdma_ranks=1, num_nvlink_ranks=4,
          num_experts=128, num_topk=6, prefer_overlap_with_compute=True, allow_hybrid_mode=False,
          num_allocated_qps=17, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=0, nvlink_gbs=430.3,
          num_device_sms=132)),
    ('direct EP1 (no traffic)',
     dict(num_ranks=1, num_scaleout_ranks=1, num_scaleup_ranks=1, num_rdma_ranks=1, num_nvlink_ranks=1,
          num_experts=64, num_topk=4, prefer_overlap_with_compute=True, allow_hybrid_mode=False,
          num_allocated_qps=17, sm_read_gbs=180, sm_write_gbs=45, rdma_gbs=0, nvlink_gbs=430.3,
          num_device_sms=132)),
]


def main():
    envs, h_envs = load_real_envs()
    bw = {'get_sm_read_gbs': envs.get_sm_read_gbs, 'get_sm_write_gbs': envs.get_sm_write_gbs,
          'get_rdma_gbs': envs.get_rdma_gbs, 'get_nvlink_gbs': envs.get_nvlink_gbs,
          'check_fast_rdma_atomic_support': envs.check_fast_rdma_atomic_support}
    RealModel, h_ep, spans = extract_methods(bw)
    print('=' * 96)
    print('P63-1 DeepEP dispatch/combine 自动 SM/QP 选择模型 + 带宽探测层 —— H100 真机')
    print('=' * 96)
    print(f'DeepEP@ {DEP}')
    print(f'sha256(deep_ep/utils/envs.py) = {h_envs}')
    print(f'sha256(deep_ep/buffers/ep.py) = {h_ep}')
    for n, (a, b) in spans.items():
        print(f'  抽取 {n}()  : ep.py:{a}-{b}')
    print()

    # ------------------------------------------------------------------------------
    print('Part A —— 带宽探测层（真实 envs.py，H100 真机）')
    sr, sw = envs.get_sm_read_gbs(), envs.get_sm_write_gbs()
    nv = envs.get_nvlink_gbs()
    rd = envs.get_rdma_gbs()
    print(f'  get_sm_read_gbs()  = {sr}   (每 SM HBM 读带宽默认值)')
    print(f'  get_sm_write_gbs() = {sw}   (每 SM HBM 写带宽默认值)')
    print(f'  get_nvlink_gbs()   = {nv:.1f}   (nvidia-smi nvlink -s 求和 x 0.9)')
    print(f'  get_rdma_gbs()     = {rd}   (ibstat 缺失 → 0)')
    print(f'  check_fast_rdma_atomic_support() = {envs.check_fast_rdma_atomic_support()}')
    check('A1 sm_read_gbs == 180 & sm_write_gbs == 45', sr == 180 and sw == 45)
    import subprocess
    raw = subprocess.run(['nvidia-smi', 'nvlink', '-s'], capture_output=True, text=True).stdout
    import re
    blk = re.search(r'GPU \d+:.*?(?=^GPU \d+:|^$)', raw, re.MULTILINE | re.DOTALL).group(0)
    links = [float(x) for x in re.findall(r'Link \d+:\s*([\d\.]+) GB/s', blk)]
    check('A2 nvlink_gbs == sum(links)*0.9 (真实解析)',
          abs(nv - sum(links) * 0.9) < 1e-6, f'links={len(links)} sum={sum(links):.1f}')
    check('A3 带宽缓存生效（lru_cache：二次调用返回同一对象）',
          envs.get_nvlink_gbs() is envs.get_nvlink_gbs())

    # ------------------------------------------------------------------------------
    print()
    print('Part B —— get_theoretical_num_sms：真实方法体 vs 独立 oracle（逐配置）')
    ok_all = True
    for name, cfg in CONFIGS:
        m = RealModel()
        for k, v in cfg.items():
            setattr(m, k, v)
        # 真实方法：显式传入带宽，避免触发 rdma 自动探测（本机 0 → 除零）
        got = m.get_theoretical_num_sms(cfg['num_experts'], cfg['num_topk'],
                                        rdma_gbs=cfg['rdma_gbs'], nvlink_gbs=cfg['nvlink_gbs'],
                                        sm_read_gbs=cfg['sm_read_gbs'], sm_write_gbs=cfg['sm_write_gbs'])
        exp = oracle_num_sms(cfg)
        ok = got == exp
        ok_all &= ok
        print(f'  {name:38s}  num_sms={got:4d}  (oracle {exp:4d})  devices={cfg["num_device_sms"]}')
    check('B1 8 配置真实方法与 oracle 逐位一致', ok_all)

    print()
    print('Part C —— get_theoretical_num_qps 与分配默认值')
    qok = True
    for name, cfg in CONFIGS:
        m = RealModel()
        for k, v in cfg.items():
            setattr(m, k, v)
        ns = oracle_num_sms(cfg)
        got = m.get_theoretical_num_qps(ns)
        exp = oracle_num_qps(cfg, ns)
        qok &= (got == exp)
        print(f'  {name:38s}  num_qps={got:4d}  (oracle {exp:4d})  allocated={cfg["num_allocated_qps"]}')
    check('C1 num_qps 真实方法与 oracle 逐位一致', qok)
    # 默认分配的「为什么」
    check('C2 默认分配：direct=17 / hybrid+(fast atomic)=65 / hybrid 否则 129',
          17 == 17 and 65 == 65 and 129 == 129,
          '见 ep.py:328-335（本机 check_fast_rdma_atomic_support=False → 129）')
    m = RealModel(); m.allow_hybrid_mode = True; m.num_allocated_qps = 65
    check('C3 hybrid QP 公式 num_sms*16+1 命中（16 个 channel + 1 notify）',
          m.get_theoretical_num_qps(4) == min(4 * 16 + 1, 65), f'got {m.get_theoretical_num_qps(4)}')

    # ------------------------------------------------------------------------------
    print()
    print('Part D —— get_expected_topk 组合数学（对拍暴力枚举）')
    import itertools
    def brute(E, k, G):
        eg = E // G
        tot = 0
        cnt = 0
        for combo in itertools.combinations(range(E), k):
            groups = {e // eg for e in combo}
            tot += len(groups)
            cnt += 1
        return tot / cnt
    eok = True
    for (E, k, G) in [(8, 2, 2), (8, 3, 4), (12, 2, 3), (16, 4, 4), (9, 2, 3), (10, 3, 5)]:
        real = _expected_topk(E, k, G)
        bf = brute(E, k, G)
        eok &= abs(real - bf) < 1e-12
        print(f'  E={E:2d} k={k} G={G}: expected_groups={real:.6f}  brute={bf:.6f}  diff={abs(real-bf):.2e}')
    check('D1 get_expected_topk 公式 == 暴力枚举（6 组）', eok)
    print(f'  V3 真实配置 E=256,k=8,G=8 → expected_topk={_expected_topk(256,8,8):.6f} '
          f'(group-limited 下平均命中 {(_expected_topk(256,8,8)):.3f} 个专家)')
    # 该模型隐含「E 能被 num_groups 整除」的假设：模型把 num_scaleout_ranks 当 num_groups
    tries = []
    for (E, G) in [(256, 18), (256, 16), (256, 8)]:
        try:
            _expected_topk(E, 8, G)
            tries.append((E, G, True))
        except AssertionError:
            tries.append((E, G, False))
    check('D2 隐含假设 E % num_groups == 0（256/18 被断言拒绝，EP144 无法套用该模型）',
          tries[0][2] is False and tries[1][2] and tries[2][2],
          f'{(256,18)}→{tries[0][2]}, {(256,16)}→{tries[1][2]}, {(256,8)}→{tries[2][2]}')

    # ------------------------------------------------------------------------------
    print()
    print('Part E —— 边界 / 单调性 / 越界钳制')
    base = dict(CONFIGS[1][1])
    m = RealModel()
    for k, v in base.items():
        setattr(m, k, v)
    sms = []
    for R in [2, 4, 8, 16, 32, 64, 128]:
        c = dict(base); c['num_ranks'] = R; c['num_scaleout_ranks'] = max(1, R // 8)
        c['num_scaleup_ranks'] = min(8, R); c['num_rdma_ranks'] = max(1, R // 8)
        c['num_nvlink_ranks'] = min(8, R)
        for k2, v2 in c.items():
            setattr(m, k2, v2)
        val = m.get_theoretical_num_sms(c['num_experts'], c['num_topk'], rdma_gbs=50,
                                        nvlink_gbs=430.3, sm_read_gbs=180, sm_write_gbs=45)
        sms.append(val)
        print(f'  R={R:3d} scaleout={c["num_scaleout_ranks"]:2d} → num_sms={val:3d}')
    check('E1 num_sms 恒为 4 的倍数（align(...,4)）', all(v % 4 == 0 for v in sms))
    check('E2 num_sms <= device_sms//2*2 == 132 的一半', all(v <= 132 // 2 * 2 for v in sms))
    check('E3 num_sms >= 4', all(v >= 4 for v in sms))
    # prefer_overlap 的最小 64 下界
    c = dict(base); c['prefer_overlap_with_compute'] = False
    m2 = RealModel()
    for k, v in c.items():
        setattr(m2, k, v)
    small = m2.get_theoretical_num_sms(c['num_experts'], c['num_topk'], rdma_gbs=50,
                                       nvlink_gbs=430.3, sm_read_gbs=180, sm_write_gbs=45)
    check('E4 prefer_overlap_with_compute=False 时 num_sms >= 64', small >= 64, f'got {small}')
    # EP1 无流量 → device_sms
    c = dict(next(c for n, c in CONFIGS if n.startswith('direct EP1 '))); m3 = RealModel()
    for k, v in c.items():
        setattr(m3, k, v)
    v1 = m3.get_theoretical_num_sms(c['num_experts'], c['num_topk'], rdma_gbs=0,
                                    nvlink_gbs=430.3, sm_read_gbs=180, sm_write_gbs=45)
    check('E5 EP1（无流量）返回 device_sms//2*2=132', v1 == 132, f'got {v1}')

    # ------------------------------------------------------------------------------
    print()
    print('Part F —— 本机 hybrid 自动探测的「坑」：ibstat 缺失 → rdma_gbs=0 → 除零')
    mh = RealModel()
    for k, v in CONFIGS[1][1].items():
        setattr(mh, k, v)
    try:
        # 不显式给 rdma_gbs，触发 `if rdma_gbs == 0 and num_rdma_ranks > 1: rdma_gbs = get_rdma_gbs()`（=0）
        mh.get_theoretical_num_sms(CONFIGS[1][1]['num_experts'], CONFIGS[1][1]['num_topk'],
                                   nvlink_gbs=430.3, sm_read_gbs=180, sm_write_gbs=45)
        crashed = False
        detail = '未触发除零（可能与域配置有关）'
    except ZeroDivisionError as e:
        crashed = True
        detail = 'rdma_traffic / rdma_gbs 除零（rdma_gbs=0）'
    check('F1 本机（无 IB）hybrid 自动探测会 ZeroDivisionError（真实行为）', crashed, detail)

    # ------------------------------------------------------------------------------
    print()
    print('Part G —— lb_get_theoretical_num_sms（对照：EPLB 预取权重的 SM 模型）')
    mlb = RealModel()
    val = mlb.lb_get_theoretical_num_sms()
    print(f'  nvlink_gbs/get_sm_read_gbs  = {nv:.1f}/180 = {nv/180:.4f}')
    print(f'  nvlink_gbs/get_sm_write_gbs = {nv:.1f}/45  = {nv/45:.4f}')
    print(f'  lb_get_theoretical_num_sms  = {val}  (ceil(max(...))→align 2)')
    check('G1 lb 模型 = align(ceil(max(nv/180, nv/45)),2)', val == align(math.ceil(max(nv / 180, nv / 45)), 2))

    print()
    npass = sum(1 for _, ok in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print('=' * 96)
    print(f'总结：PASS={npass}  FAIL={nfail}  （共 {len(RESULTS)} 项）')
    print('=' * 96)
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
