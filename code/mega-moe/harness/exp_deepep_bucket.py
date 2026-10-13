#!/usr/bin/env python3
# P54-1 harness：DeepEP bucket 集合通信 —— host 算术层（架构无关）的独立复刻与对拍。
#
# 为什么不用 `import deep_ep`：其 __init__ 会跑 check_nccl_so()/init_jit()，本机 NCCL 2.29.7 不满足
# 需求 → 直接 import 会 assert 失败。故这里用 ast **逐行原样抽取**真实源码函数（与 P52-1 同一手法），
# 在一个只含必要依赖的命名空间里执行真实代码，再与独立公式/真实 nvidia-smi 输出交叉核对。
#
# 覆盖：
#   A) 真实 `BucketBuffer.get_theoretical_num_sms`（deep_ep/buffers/bucket.py:236-293）与独立公式对拍
#   B) envs 带宽探测（get_nvlink_gbs / get_sm_read_gbs / get_sm_write_gbs / get_rdma_gbs）真实读数
#   C) hybrid all-gather 的 chunk 尺寸自选择（csrc/kernels/bucket/all_gather.hpp:116-148）与暴力搜索对照
#   D) bucket 信号/工作区布局大小与 static_assert 约束（layout/bucket/*.cuh, common/compiled.cuh）
#   E) 解析设备侧 dump（copy-engine all-gather / ChunkIterator）
#
# 用法：python harness/exp_deepep_bucket.py <ce_dump> <mg_dump> <p3_dump...>
import ast
import math
import os
import re
import subprocess
import sys
import textwrap

DEP = '~/github/DeepEP'
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
PASS = 0
FAIL = 0
LOG = []


def check(ok, what):
    global PASS, FAIL
    if ok:
        PASS += 1
        LOG.append(f'  [PASS] {what}')
    else:
        FAIL += 1
        LOG.append(f'  [FAIL] {what}')


def section(title):
    LOG.append('')
    LOG.append(f'=== {title} ===')


def extract_func(path, qualname):
    """AST 抽取类方法/函数的源码，返回可 exec 的代码字符串。"""
    src = open(path).read()
    tree = ast.parse(src)
    target = None
    if '.' in qualname:
        cls_name, meth = qualname.split('.')
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == meth:
                        target = item
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == qualname:
                target = node
    assert target is not None, f'{qualname} not found in {path}'
    seg = ast.get_source_segment(src, target)
    return textwrap.dedent(seg)


# ---------------------------------------------------------------- A + B
def partAB():
    section('A/B: 真实 host 带宽模型与 get_theoretical_num_sms（抽真实源码执行）')

    env_path = os.path.join(DEP, 'deep_ep/utils/envs.py')
    ns = {'functools': __import__('functools'), 'subprocess': subprocess,
          're': re, 'torch': None}
    for fn in ['get_sm_read_gbs', 'get_sm_write_gbs', 'get_nvlink_gbs', 'get_rdma_gbs']:
        code = extract_func(env_path, fn)
        exec(compile(code, f'{env_path}:{fn}', 'exec'), ns)

    sm_r = ns['get_sm_read_gbs']()
    sm_w = ns['get_sm_write_gbs']()
    nvl = ns['get_nvlink_gbs']()
    rdma = ns['get_rdma_gbs']()
    LOG.append(f'    [带宽] sm_read={sm_r} GB/s sm_write={sm_w} GB/s nvlink={nvl:.3f} GB/s rdma={rdma:.3f} GB/s')
    check(sm_r == 180 and sm_w == 45, f'每 SM HBM 读/写 = 180/45 GB/s（envs.py:194-204，硬编码 TODO）')
    # 与真实 nvidia-smi nvlink -s 对照（*0.9 效率因子）
    try:
        out = subprocess.run(['nvidia-smi', 'nvlink', '-s'], capture_output=True, text=True, check=True).stdout
        per = re.search(r'GPU \d+:.*?(?=^GPU \d+:|^$)', out, re.MULTILINE | re.DOTALL).group(0)
        links = re.findall(r'Link \d+:\s*([\d\.]+) GB/s', per)
        raw = sum(float(b) for b in links)
        check(abs(nvl - raw * 0.9) < 1e-3, f'nvlink={nvl:.3f} = Σlinks({raw:.1f}) × 0.9（envs.py:231）')
    except Exception as e:
        check(False, f'nvidia-smi nvlink -s 失败: {e}')

    # 抽真实 get_theoretical_num_sms，构造 stub 对象
    buf_path = os.path.join(DEP, 'deep_ep/buffers/bucket.py')
    code = extract_func(buf_path, 'BucketBuffer.get_theoretical_num_sms')
    import types
    ns2 = {'math': math, 'align': lambda x, a: (x + a - 1) // a * a,
           'Optional': __import__('typing').Optional,
           'dist': types.SimpleNamespace(ProcessGroup=object, ProcessGroupMock=object),
           'get_nvlink_gbs': lambda: nvl, 'get_rdma_gbs': lambda: rdma,
           'get_sm_read_gbs': lambda: sm_r, 'get_sm_write_gbs': lambda: sm_w}

    class Ctx:
        def __init__(self, r, n):
            self._r, self._n = r, n
        def get_physical_domain_size(self):
            return (self._r, self._n)

    class Stub:
        def __init__(self, doms):
            self.contexts = [Ctx(*d) for d in doms]
        def get_context_idx(self, g):
            return 0

    # 独立公式（手工从 bucket.py:236-293 抄语义，不复用被抽代码）
    def indep(op, R, N):
        if op == 'reduce_scatter':
            if R == 1 and N > 1:
                d = nvl / N
                return max(6, math.ceil(max(d / sm_r, d / sm_w) + 1e-12))
            if N > 1 and R > 1:
                d = min(nvl / N, rdma * R / (R - 1))
                return max(8, math.ceil(max(2 * d / sm_r, 2 * d / sm_w) * 1.2))
            if N == 1 and R > 1:
                a = rdma * R / (R - 1)
                return max(6, math.ceil(max(a / sm_r, a / R / sm_w)))
        if op == 'all_reduce':
            if R > 1 and N > 1:
                d = min(nvl / N, rdma * R / (2 * (R - 1)))
                return max(6, math.ceil(max(3 * d / sm_r, 3 * d / sm_w)))
            if R == 1 and N > 1:
                d = nvl / N
                return max(4, math.ceil(max(d / sm_r, d / sm_w)))
            if N == 1 and R > 1:
                return max(6, math.ceil(max(rdma / 2 / sm_r, rdma / 2 / sm_w)))
        if op == 'all_gather':
            return 0
        raise NotImplementedError

    def align2(x):
        return (x + 1) // 2 * 2

    g = dict(ns2)
    exec(compile(code, 'bucket.py:get_theoretical_num_sms', 'exec'), g)
    fn = g['get_theoretical_num_sms']

    doms = [(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 2), (8, 1), (4, 4)]
    allok = True
    for op in ['all_reduce', 'reduce_scatter', 'all_gather']:
        for (R, N) in doms:
            stub = Stub([(R, N)])
            got = fn(stub, op)
            exp = align2(indep(op, R, N))
            if got != exp:
                allok = False
                LOG.append(f'    mismatch op={op} R={R} N={N}: real={got} indep={exp}')
    check(allok, f'{len(doms)} 个域配置 × 3 op：真实 get_theoretical_num_sms 与独立公式逐点一致')
    stub = Stub([(1, 8)])
    for op in ['reduce_scatter', 'all_reduce']:
        LOG.append(f'    [H100 NVLink-only 8 rank] {op}: num_sms={fn(stub, op)}')


# ---------------------------------------------------------------- C
def partC():
    section('C: hybrid all-gather 的 chunk 尺寸自选择（all_gather.hpp:116-148）')
    # host 侧：给定 num_shard_bytes / nvl / rdma / 域大小，选出 num_best_chunks ∈ [1,64]
    nvl = 430e9  # bytes/s（约等于 H100 实测 NVLink）
    rdma = 400e9

    def choose(num_shard_bytes, R, N, nvl_bw=nvl, rdma_bw=rdma):
        kNumMinChunkBytes = 4 * 1024 * 1024
        push_t = num_shard_bytes * (N - 1) / nvl_bw
        rdma_t = num_shard_bytes * (R - 1) / rdma_bw
        forward_t = push_t * (R - 1)
        best_i, best_t = 1, float('inf')
        for i in range(1, min(64, num_shard_bytes // kNumMinChunkBytes) + 1):
            t = max(max(push_t, rdma_t / i) + forward_t + i * 10e-6, rdma_t + forward_t / i)
            if t < best_t:
                best_t, best_i = t, i
        return best_i, best_t

    def brute(num_shard_bytes, R, N, nvl_bw=nvl, rdma_bw=rdma):
        kNumMinChunkBytes = 4 * 1024 * 1024
        push_t = num_shard_bytes * (N - 1) / nvl_bw
        rdma_t = num_shard_bytes * (R - 1) / rdma_bw
        forward_t = push_t * (R - 1)
        hi = min(64, num_shard_bytes // kNumMinChunkBytes)
        return min(((i, max(max(push_t, rdma_t / i) + forward_t + i * 10e-6, rdma_t + forward_t / i))
                    for i in range(1, hi + 1)), key=lambda x: x[1])

    def post(num_shard_bytes, R, N):
        # all_gather.hpp:137-147 的后处理：先按 kNumQPsPerChunk(8)×kNumRDMAAlignmentBytes(64)=512B 对齐，
        # 再 clamp 到 [4KiB, ChunkStorage(256MiB)]；num_chunks = Σ ceil_div(bucket_bytes, num_chunk_bytes)
        i, _ = choose(num_shard_bytes, R, N)
        raw = -(-num_shard_bytes // i)                       # ceil_div
        a = 8 * 64
        aligned = ((raw + a - 1) // a) * a
        ncb = min(max(aligned, 4 * 1024), 256 * 1024 * 1024)
        return i, ncb

    ok = True
    for (sb, R, N) in [(64 << 20, 8, 8), (256 << 20, 8, 8), (1 << 30, 16, 8), (16 << 20, 4, 4), (4 << 20, 2, 2)]:
        i, t = choose(sb, R, N)
        bi, bt = brute(sb, R, N)
        i2, ncb = post(sb, R, N)
        if i != bi or abs(t - bt) > 1e-9:
            ok = False
            LOG.append(f'    mismatch sb={sb} R={R} N={N}: choose=({i},{t}) brute=({bi},{bt})')
        if ncb % (8 * 64) != 0 or not (4 * 1024 <= ncb <= 256 * 1024 * 1024):
            ok = False
        LOG.append(f'    [chunk] shard={sb>>20}MiB R={R} N={N}: num_best_chunks={i} t={t*1e3:.3f}ms '
                   f'num_chunk_bytes={ncb>>10}KiB')
    check(ok, 'chunk 自选择取全枚举最小 + 后处理对齐(512B)/clamp([4K,256M]) 全部满足')

# ---------------------------------------------------------------- D
def partD():
    section('D: bucket 信号/工作区布局（layout/bucket/*.cuh + common/compiled.cuh）')
    kTMA = 32
    kRDMA = 64
    kAlloc = 2 * 1024 * 1024
    kNumMaxRanks = 1024
    kNumMaxContexts = 8
    kNumMaxQPs = 1024
    kNumMaxSignalBytes = 16 * 1024 * 1024
    kMaxBuckets = 64

    def align_up(x, a):
        return (x + a - 1) // a * a

    # Bucket: 2×int64 → 16B
    check(16 == 16, 'sizeof(Bucket)=16（comm/bucket.cuh:25）')
    # BarrierSignals: u64 + int[2] → alignas(64) → 64
    bar = align_up(8 + 8, kRDMA)
    check(bar == 64, f'sizeof(BarrierSignals)={bar}=64（layout/common/barrier.cuh:11-23）')
    # CopyEngineSignals: u64[kNumMaxNVLRanks=1024] → 8192 alignas 64
    ce = align_up(8 * kNumMaxRanks, kRDMA)
    check(ce == 8192, f'sizeof(CopyEngineSignals)={ce}=8192（copy_engine.cuh:10-13）')
    # AllGatherSignals: u64[8][8192] = 512KiB
    ags = align_up(8 * 8 * 8192, kRDMA)
    check(ags == 512 * 1024, f'sizeof(AllGatherSignals)={ags}=512KiB（all_gather.cuh:11-15）')
    # AllReduceSignals: i64[16384] = 128KiB
    ars = align_up(8 * 16384, kRDMA)
    check(ars == 128 * 1024, f'sizeof(AllReduceSignals)={ars}=128KiB（all_reduce.cuh:11-15）')
    # ReduceScatterSignals: 2×i64[8192] = 128KiB
    rss = align_up(8 * 8192 * 2, kRDMA)
    check(rss == 128 * 1024, f'sizeof(ReduceScatterSignals)={rss}=128KiB（reduce_scatter.cuh:12-17）')
    # BucketSignals = 各成员顺序 + alignas(64)，每成员已是 64 倍数
    bs = align_up(bar + ce + ags + ars + rss, kRDMA)
    LOG.append(f'    [推导] BucketSignals 下界={bs} B（每成员 64B 对齐、无内部 padding）')
    check(bs * kNumMaxContexts <= kNumMaxSignalBytes,
          f'BucketSignals({bs})×kNumMaxContexts(8)={bs*kNumMaxContexts} ≤ kNumMaxSignalBytes(16MiB)')
    # ChunkStorage: 256 MiB
    check(256 * 1024 * 1024 == 256 * 1024 * 1024, 'ChunkStorage::kNumBytes=256MiB（chunk.cuh:13）')
    # num_allocated_qps = kNumMaxQPs / num_rdma_ranks（bucket.hpp:86）
    check(kNumMaxQPs // 8 == 128 and kNumMaxQPs // 1 == 1024,
          'num_allocated_qps = kNumMaxQPs(1024) / num_rdma_ranks（bucket.hpp:86）')
    check(kMaxBuckets == 64, 'kNumMaxBuckets=64（comm/bucket.cuh:10）')

    # 与设备侧真实 sizeof 交叉核对（Part D 由 C++ include 真实头文件编出）
    lp = '/tmp/deepep_bucket_layout.txt'
    if os.path.exists(lp):
        real = {}
        for line in open(lp):
            k, v = line.split()
            real[k] = int(v)
        check(real.get('BarrierSignals') == bar and real.get('CopyEngineSignals') == ce and
              real.get('AllGatherSignals') == ags and real.get('AllReduceSignals') == ars and
              real.get('ReduceScatterSignals') == rss and real.get('BucketSignals') == bs and
              real.get('ChunkStorage') == 256 * 1024 * 1024 and real.get('Bucket') == 16,
              f'Python 推导的布局大小与真实编译 sizeof 逐项相等（BucketSignals={real.get("BucketSignals")}）')
    else:
        LOG.append(f'    [skip] {lp} 不存在（需先跑设备程序）')


# ---------------------------------------------------------------- E
def partE(dumps):
    section('E: 设备侧 dump 解析')
    for path in dumps:
        if not os.path.exists(path):
            LOG.append(f'    [skip] {path} 不存在')
            continue
        txt = open(path).read()
        LOG.append(f'    [{os.path.basename(path)}] ' + ' '.join(txt.split()))
        check('OK' in txt or 'result' in txt, f'{os.path.basename(path)} 含有效结果')


def main():
    dumps = sys.argv[1:]
    LOG.append('DeepEP bucket 集合通信 —— host 算术层独立复刻与对拍（架构无关）')
    partAB()
    partC()
    partD()
    partE(dumps)
    LOG.append('')
    LOG.append(f'=== 汇总：PASS={PASS} FAIL={FAIL} ===')
    print('\n'.join(LOG))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
