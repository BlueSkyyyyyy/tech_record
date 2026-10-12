#!/usr/bin/env python3
"""P18-1: DeepGEMM 的「选核」——SM90 启发式 `get_best_config` 精读 + H100 真机对拍。

参考源码（~/github/DeepGEMM，commit 057ca59）：
  csrc/jit_kernels/heuristics/common.hpp   —— get_best_config：枚举候选 → 逐个打分 → 取最优
  csrc/jit_kernels/heuristics/sm90.hpp     —— SM90ArchSpec：get_layout_candidates /
                                              get_storage_config / get_pipeline_config /
                                              get_layout_info（L1/L2 代价模型）/ compare
  csrc/jit_kernels/heuristics/sm100.hpp    —— SM100ArchSpec（另一套：swap AB、TMEM、多级 tie-break）
  csrc/jit_kernels/heuristics/config.hpp   —— GemmDesc / Layout / StorageConfig / PipelineConfig /
                                              LaunchConfig / LayoutInfo 数据结构
  csrc/jit_kernels/heuristics/runtime.hpp  —— HeuristicsRuntime（block_*_multiple_of、mk_alignment）
  csrc/jit_kernels/heuristics/utils.hpp    —— get_swizzle_mode / get_num_element_bits

方法：本脚本用 Python **独立重写** SM90ArchSpec::get_best_config 的每一行，然后在 H100 上
用 `DG_PRINT_CONFIGS=1` 让真实 C++ 对一组形状各打印一次它选出的配置（GemmDesc + GemmConfig +
LayoutInfo），再逐字段与 Python 复刻对拍。这样既验证「候选枚举 + 代价模型 + tie-break」的复刻，
也把「DeepGEMM 到底怎么选 block_m/n/k/stage/cluster」讲清楚。

注意：BF16 dense 默认走 cuBLASLt（见 P17-1），需 `use_deterministic_algorithms(True)` 才会
进入 DeepGEMM 的启发式；本脚本对 BF16 dense 显式打开 deterministic。

运行：bash harness/exp_best_config.sh > harness/exp_best_config.out.txt
"""
import os
import re
import sys
import tempfile
import traceback
from math import gcd

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major


def _find_deepgemm_dir():
    for p in (os.environ.get('DEEPGEMM_DIR'), os.path.expanduser('~/github/DeepGEMM'),
              '~/github/DeepGEMM'):
        if p and os.path.isdir(os.path.join(p, 'tests')):
            return p
    raise RuntimeError('DeepGEMM source dir not found')


DEEPGEMM_DIR = _find_deepgemm_dir()
sys.path.insert(0, os.path.join(DEEPGEMM_DIR, 'tests'))

from generators import (  # noqa: E402
    MajorTypeAB, generate_m_grouped_contiguous, generate_m_grouped_masked,
    generate_k_grouped_contiguous,
)
from deep_gemm.utils import align  # noqa: E402


# ===================================================================== 0. 常量表
# csrc/jit_kernels/heuristics/utils.hpp:12-19
_BITS = {'BF16': 16, 'MXFP8FP4': 8, 'MXF4': 4}
# c10::elementSize：Float8_e4m3fn=1B / BFloat16=2B / Float=4B
_ELEM = {'Float8_e4m3fn': 1, 'BFloat16': 2, 'Float': 4, 'Float8_e5m2': 1, 'kPackedFP4': 1}
_GEMM_TYPE = {'Normal': 0, 'MGroupedContiguous': 1, 'MGroupedMasked': 2,
              'KGroupedContiguous': 3, 'Batched': 4,
              'MGroupedContiguousWithPsumLayout': 5, 'KGroupedContiguousWithPsumLayout': 6}
_SMEM_CAPACITY = 232448  # sm90.hpp:14
_MMA_KIND = {0: 'BF16', 1: 'MXFP8FP4', 2: 'MXF4'}


def ceil_div(a, b):
    return -(-a // b)


def lcm(a, b):
    return a // gcd(a, b) * b


# ===================================================================== 1. C++ 打印解析
def parse_named(s):
    """解析 `Name(k=v, k=v, ...)`（值可嵌套），返回 (name, dict{raw_str})。"""
    s = s.strip()
    m = re.match(r'^(\w+)\(', s)
    name = m.group(1)
    assert s.endswith(')'), s
    body = s[m.end():-1]
    fields, depth, cur = [], 0, ''
    for ch in body:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            fields.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        fields.append(cur)
    d = {}
    for f in fields:
        k, _, v = f.partition('=')
        d[k.strip()] = v.strip()
    return name, d


def parse_line(line):
    """解析 `GemmDesc(...): GemmConfig(...), LayoutInfo(...)`。"""
    # 顶层按第一个 ': ' 切出 desc / 其余
    idx = line.find('): ')
    desc_str = line[:idx + 1]
    rest = line[idx + 3:]
    # rest = GemmConfig(...), LayoutInfo(...)
    depth, split = 0, None
    for i, ch in enumerate(rest):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            split = i
            break
    cfg_str = rest[:split].strip()
    info_str = rest[split + 1:].strip()
    _, desc = parse_named(desc_str)
    _, cfg = parse_named(cfg_str)
    _, layout = parse_named(cfg['layout'])
    _, storage = parse_named(cfg['storage_config'])
    _, pipeline = parse_named(cfg['pipeline_config'])
    _, launch = parse_named(cfg['launch_config'])
    _, info = parse_named(info_str)
    return desc, layout, storage, pipeline, launch, info


# ===================================================================== 2. Python 复刻
class SM90:
    smem_capacity = _SMEM_CAPACITY
    # HeuristicsRuntime 默认值（runtime.hpp:12）
    mk_alignment_for_contiguous_layout = 128

    def __init__(self, desc):
        self.d = {k: v for k, v in desc.items()}

    # --- 字段访问
    @property
    def gemm_type(self):
        return int(self.d['gemm_type'])

    @property
    def kernel_type(self):
        return int(self.d['kernel_type'])

    @property
    def mma_kind(self):
        return _MMA_KIND[int(self.d['mma_kind'])]

    @property
    def a_elem(self):
        return _ELEM[self.d['a_dtype']]

    @property
    def b_elem(self):
        return _ELEM[self.d['b_dtype']]

    @property
    def cd_elem(self):
        return _ELEM[self.d['cd_dtype']]

    @property
    def cd_is_float(self):
        return self.d['cd_dtype'] == 'Float'

    @property
    def num_sms(self):
        return int(self.d['num_sms'])

    def expected(self, which):
        v = int(self.d[f'expected_{which}'])
        if v > 0:
            return v
        return int(self.d[{'m': 'm', 'n': 'n', 'k': 'k',
                           'num_groups': 'num_groups'}[which]])

    # --- utils.hpp:22-30
    @staticmethod
    def swizzle(block_size, elem_size):
        for mode in (128, 64, 32, 16):
            if (block_size * elem_size) % mode == 0:
                return mode
        raise AssertionError('unreachable')

    # --- sm90.hpp:120-146
    def storage_config(self, L):
        load_block_m = L['block_m']
        load_block_n = L['block_n']
        store_block_m = 64 if self.kernel_type == 0 else L['block_m']  # 1D1D -> wgmma_m=64
        store_block_n = L['block_n']
        swizzle_a = self.swizzle(L['block_k'], self.a_elem)   # major_a==K（本实验都是 K-major）
        swizzle_b = self.swizzle(L['block_k'], self.b_elem)
        swizzle_cd = self.swizzle(store_block_n, self.cd_elem) if not self.cd_is_float else 0
        return dict(load_block_m=load_block_m, load_block_n=load_block_n,
                    store_block_m=store_block_m, store_block_n=store_block_n,
                    swizzle_a_mode=swizzle_a, swizzle_b_mode=swizzle_b,
                    swizzle_cd_mode=swizzle_cd)

    # --- sm90.hpp:148-188
    def pipeline_config(self, L, S):
        kNumMaxStages = 16
        smem_cd = align(L['block_m'] * L['block_n'] * self.cd_elem, 1024)
        smem_barriers = kNumMaxStages * 8 * 2
        smem_a = S['load_block_m'] * L['block_k'] * self.a_elem
        smem_b = S['load_block_n'] * L['block_k'] * self.b_elem
        smem_sfa = 0 if self.kernel_type == 2 else align(L['block_m'] * 4, 128)
        smem_sfb = 0 if self.kernel_type != 0 else align(L['block_n'] * 4, 128)
        use_uniform_sfb = 1 if L['block_k'] % L['block_n'] == 0 else 2
        smem_extra_sfb = 0 if self.kernel_type != 1 else align(
            ceil_div(int(self.d['k']), L['block_k']) * 4 * use_uniform_sfb, 8)
        smem_tensormap = 4 * 128 if self.gemm_type == 3 else 0   # sizeof(CUtensorMap)=128
        smem_extra = smem_cd + smem_barriers + smem_extra_sfb + smem_tensormap
        smem_per_stage = smem_a + smem_b + smem_sfa + smem_sfb
        num_stages = min((self.smem_capacity - smem_extra) // smem_per_stage, kNumMaxStages)
        return dict(smem_size=smem_extra + num_stages * smem_per_stage,
                    num_stages=num_stages, num_tma_store_stages=2)

    # --- sm90.hpp:190-200
    def launch_config(self, L):
        num_math = 128 if L['block_m'] <= 64 else 256
        return dict(num_sms=self.num_sms, num_sms_per_cluster=L['cluster_m'] * L['cluster_n'],
                    num_threads=128 + num_math, num_tma_threads=128, num_math_threads=num_math,
                    num_non_epilogue_threads=0, num_epilogue_threads=0)

    # --- sm90.hpp:202-239
    def layout_info(self, L):
        num_blocks = (ceil_div(self.expected('m'), L['block_m']) *
                      ceil_div(self.expected('n'), L['block_n']) *
                      self.expected('num_groups'))
        num_waves = ceil_div(num_blocks, self.num_sms)
        last = num_blocks % self.num_sms
        last_wave_util = self.num_sms if last == 0 else last
        l2_bw = min(64 * self.num_sms, int(8e6 / 1.3e3))
        l1_bw = 128 * self.num_sms
        ek = self.expected('k')
        n_l2_ab = ek * (L['block_m'] // L['cluster_n'] + L['block_n'] // L['cluster_m']) * self.a_elem
        n_l1_ab = ek * (L['block_m'] + L['block_n']) * self.a_elem
        n_l1_tc = ek * (max(64, L['block_m']) + L['block_n']) * self.a_elem + L['block_m'] * L['block_n'] * self.cd_elem
        n_l1_l2_cd = L['block_m'] * L['block_n'] * self.cd_elem * (2 if int(self.d['with_accumulation']) else 1)
        l2_cycles = (n_l2_ab + n_l1_l2_cd) * num_blocks // l2_bw
        l1_cycles = (n_l1_ab + n_l1_tc + n_l1_l2_cd) * num_blocks // l1_bw
        wave_eff = num_blocks / (num_waves * self.num_sms)
        num_cycles = int(max(l1_cycles, l2_cycles) / wave_eff)
        if L['cluster_m'] * L['cluster_n'] > 1 and num_waves <= 1:
            num_cycles = (1 << 63) - 1
        return dict(num_waves=num_waves, last_wave_util=last_wave_util, num_cycles=num_cycles)

    # --- sm90.hpp:16-118
    def candidates(self, block_m_mult=1, block_n_mult=1):
        desc = self.d
        gt, kt = self.gemm_type, self.kernel_type
        block_m_candidates = []
        if gt in (0, 4, 3):
            block_m_candidates = [64, 128]
            if int(desc['m']) <= 16:
                block_m_candidates.append(16)
            if int(desc['m']) <= 32:
                block_m_candidates.append(32)
            if not self.cd_is_float:
                block_m_candidates.append(256)
        elif gt in (1, 5):
            block_m_candidates = [self.mk_alignment_for_contiguous_layout]
        elif gt == 2:
            block_m_candidates = [64, 128]

        step = lcm(16, block_n_mult)
        start = step
        block_n_candidates = []
        if kt == 0 and self.cd_is_float:  # Kernel1D1D + Float：避免 bank conflict
            start = 24
            block_n_candidates.append(16)
        end = 256
        if kt == 1:
            end = 192
        if kt == 0:
            end = 160
        for i in range(start, end + 1, step):
            block_n_candidates.append(i)

        block_k = 128 * 8 // _BITS[self.mma_kind]
        disable_multicast = (gt == 3 and int(desc['num_groups']) > 4) or (gt == 4)

        out = []
        cmax = 1 if disable_multicast else 2
        for cluster_m in range(1, cmax + 1):
            for cluster_n in range(1, cmax + 1):
                if cluster_m * cluster_n > 2:
                    continue
                if self.num_sms % (cluster_m * cluster_n) != 0:
                    continue
                for block_m in block_m_candidates:
                    for block_n in block_n_candidates:
                        if kt == 1 and block_n > block_k and \
                           (block_n % (block_n - block_k) != 0 and block_k % (block_n - block_k) != 0):
                            continue
                        if gt in (2, 5) and ceil_div(int(desc['n']), block_n) % (cluster_m * cluster_n) != 0:
                            continue
                        if block_m > 128 and block_n > 128:
                            continue
                        L = dict(block_m=block_m, block_n=block_n, block_k=block_k,
                                 cluster_m=cluster_m, cluster_n=cluster_n)
                        S = self.storage_config(L)
                        if S['swizzle_a_mode'] % 64 != 0 or S['swizzle_b_mode'] % 64 != 0:
                            continue
                        ns = self.pipeline_config(L, S)['num_stages']
                        if ns < 3 or (block_m * block_n < 128 * 192 and ns < 4):
                            continue
                        out.append(L)
        return out

    def best(self, block_m_mult=1, block_n_mult=1):
        cands = self.candidates(block_m_mult, block_n_mult)
        bestL, bestI = cands[0], self.layout_info(cands[0])
        for L in cands[1:]:
            I = self.layout_info(L)
            if I['num_cycles'] < bestI['num_cycles']:
                bestL, bestI = L, I
        return bestL, bestI, cands


# ===================================================================== 3. 真机采集
def collect(fn):
    """把 C++ stdout（fd1）重定向到临时文件，运行 fn，返回被打印的文本。"""
    sys.stdout.flush()
    fd = os.dup(1)
    tf = tempfile.TemporaryFile(mode='w+b')
    os.dup2(tf.fileno(), 1)
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        os.dup2(fd, 1)
        os.close(fd)
    tf.seek(0)
    data = tf.read().decode('utf-8', 'replace')
    tf.close()
    return data


def collect_configs(fn):
    text = collect(fn)
    out = []
    for line in text.splitlines():
        if line.startswith('GemmDesc('):
            try:
                out.append(parse_line(line))
            except Exception:
                pass
    return out


# ===================================================================== 4. 用例
def reset():
    deep_gemm.use_deterministic_algorithms(False)
    deep_gemm.set_num_sms(deep_gemm.get_num_sms())
    deep_gemm.set_block_size_multiple_of((1, 1))


def fp8_cast_dense(m, n, k):
    from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
    a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')
    fa, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
    fb, sfb = per_block_cast_to_fp8(b, use_ue8m0=False)
    d = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
    return (fa, sfa), (fb, sfb), d


def cases():
    """返回 [(label, callable)]；每个 callable 触发一次 get_best_config。"""
    C = []

    # ---- 稠密 FP8（SM90 Kernel1D2D）
    for (m, n, k) in [(256, 512, 1024), (16, 128, 256), (32, 256, 512),
                      (8192, 7168, 7168), (128, 192, 128)]:
        def f(m=m, n=n, k=k):
            (a, sfa), (b, sfb), d = fp8_cast_dense(m, n, k)
            deep_gemm.fp8_gemm_nt((a, sfa), (b, sfb), d)
        C.append((f'dense fp8 {m}x{n}x{k} (1d2d)', f))

    # ---- 稠密 BF16（KernelNoSF，需 deterministic 才进 DeepGEMM）
    for (m, n, k) in [(256, 512, 1024), (16, 128, 256), (4096, 7168, 7168),
                      (64, 160, 384), (512, 512, 1024)]:
        def f(m=m, n=n, k=k):
            deep_gemm.use_deterministic_algorithms(True)
            a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
            b = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')
            d = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
            deep_gemm.bf16_gemm_nt(a, b, d)
        C.append((f'dense bf16 {m}x{n}x{k} (NoSF)', f))

    # ---- m-grouped contiguous / psum / masked FP8
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    for (label, G, exp_m, n, k, psum) in [('m-contig', 6, 128, 256, 512, False),
                                          ('m-psum', 6, 128, 256, 512, True),
                                          ('m-contig', 8, 256, 2048, 7168, False)]:
        def f(G=G, exp_m=exp_m, n=n, k=k, psum=psum):
            deep_gemm.set_mk_alignment_for_contiguous_layout(128)
            m, a, b, layout, d, ref, valid = generate_m_grouped_contiguous(
                G, exp_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                use_ue8m0=False, use_psum_layout=psum)
            d.zero_()
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a, b, d, layout, disable_ue8m0_cast=True, use_psum_layout=psum)
        C.append((f'{label} fp8 G={G} {exp_m}x{n}x{k}', f))
    for (G, max_m, exp_m, n, k) in [(6, 160, 96, 256, 512), (8, 256, 200, 1024, 512)]:
        def f(G=G, max_m=max_m, exp_m=exp_m, n=n, k=k):
            deep_gemm.set_mk_alignment_for_contiguous_layout(128)
            a, b, masked_m, d, ref, valid = generate_m_grouped_masked(
                G, max_m, exp_m, n, k, use_ue8m0=False, use_psum_layout=False)
            d.zero_()
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                a, b, d, masked_m, int(exp_m * 1.2), disable_ue8m0_cast=True)
        C.append((f'm-masked fp8 G={G} max_m={max_m} n={n} k={k}', f))

    # ---- k-grouped nt FP8（Kernel1D1D，float 输出）
    for (G, m, n, ks) in [(4, 512, 1024, [512, 384, 640, 256]),
                          (8, 4096, 7168, [4096] * 8),
                          (2, 128, 256, [128, 128])]:
        def f(G=G, m=m, n=n, ks=ks):
            deep_gemm.set_mk_alignment_for_contiguous_layout(128)
            _, a, b, c, d, ref, layout, host_ks = generate_k_grouped_contiguous(
                G, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
                use_ue8m0=False, gran_k=128, quant_config=None,
                use_psum_layout=False, k_alignment=128, accumulate=True, out_dtype=torch.float)
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, host_ks, layout, d, recipe=(1, 1, 128))
        C.append((f'k-grouped fp8 nt G={G} {m}x{n} ks={ks[:2]}...', f))

    # ---- 按头 batched BF16 einsum（GemmType::Batched）
    def f():
        deep_gemm.use_deterministic_algorithms(True)
        B, H, R, Dd = 16, 16, 512, 512
        A4 = torch.randn(B, H, R, dtype=torch.bfloat16, device='cuda')
        B4 = torch.randn(H, Dd, R, dtype=torch.bfloat16, device='cuda')
        D4 = torch.empty(B, H, Dd, dtype=torch.bfloat16, device='cuda')
        deep_gemm.einsum('bhr,hdr->bhd', A4, B4, D4)
    C.append(('einsum batched bf16 bhr,hdr->bhd', f))

    # ---- block_size_multiple_of 影响 block_n 候选
    def f():
        deep_gemm.set_block_size_multiple_of((32, 32))
        (a, sfa), (b, sfb), d = fp8_cast_dense(256, 512, 2048)
        deep_gemm.fp8_gemm_nt((a, sfa), (b, sfb), d)
    C.append(('dense fp8 256x512x2048 (block_mult=32,32)', f))

    return C


# ===================================================================== 5. 主流程
def fmt_layout(L):
    return (f"bm={L['block_m']},bn={L['block_n']},bk={L['block_k']},"
            f"cm={L['cluster_m']},cn={L['cluster_n']}")


def _mk_desc(**kw):
    base = dict(gemm_type=0, kernel_type=1, m=1, n=1, k=1, num_groups=1,
                major_a=0, major_b=0, mma_kind=1, a_dtype='Float8_e4m3fn',
                b_dtype='Float8_e4m3fn', cd_dtype='BFloat16', with_accumulation=0,
                num_sms=132, tc_util=100, compiled_dims='nk', ensure_zero_padding=1,
                expected_m=0, expected_n=0, expected_k=0, expected_num_groups=0)
    base.update({k: str(v) for k, v in kw.items()})
    return base


def show_ranking():
    """展示「代价模型如何在候选间排序」——泄露一点内部：把同一 desc 的候选按 num_cycles 排序。"""
    print('\n' + '=' * 100)
    print('候选排序（Python 复刻的 L1/L2 代价模型）——「为什么选它」')
    print('=' * 100)
    demos = [
        ('dense fp8 8192x7168x7168',
         _mk_desc(gemm_type=0, kernel_type=1, m=8192, n=7168, k=7168)),
        ('k-grouped fp8 nt G=8 4096x7168 (max_k=4096)',
         _mk_desc(gemm_type=3, kernel_type=0, m=4096, n=7168, k=32768, num_groups=8,
                  cd_dtype='Float', with_accumulation=1,
                  expected_m=4096, expected_n=7168, expected_k=4096, expected_num_groups=8)),
    ]
    for label, desc in demos:
        h = SM90(desc)
        cands = h.candidates()
        ranked = sorted(((h.layout_info(L), L) for L in cands),
                        key=lambda t: t[0]['num_cycles'])
        best_I, best_L = ranked[0]
        second_I, second_L = ranked[1]
        print(f'\n  {label}: 候选数={len(cands)}')
        print(f'    选出   {fmt_layout(best_L)}  waves={best_I["num_waves"]} '
              f'cycles={best_I["num_cycles"]}')
        print(f'    次优   {fmt_layout(second_L)}  waves={second_I["num_waves"]} '
              f'cycles={second_I["num_cycles"]}  '
              f'(差 {second_I["num_cycles"] - best_I["num_cycles"]}, '
              f'{(second_I["num_cycles"] / best_I["num_cycles"] - 1) * 100:.1f}%)')
        print('    Top-6：')
        for I, L in ranked[:6]:
            print(f'      {fmt_layout(L):38s} waves={I["num_waves"]:>3} '
                  f'cycles={I["num_cycles"]:>10} last_wave_util={I["last_wave_util"]}')


def main():
    print(f'deep_gemm {deep_gemm.__version__}, arch_major={get_arch_major()}, '
          f'device={torch.cuda.get_device_name(0)}, num_sms={deep_gemm.get_num_sms()}')
    print('=' * 100)
    print('SM90ArchSpec::get_best_config 复刻对拍（Python 独立重写 vs 真实 C++ DG_PRINT_CONFIGS）')
    print('=' * 100)

    all_cases = cases()
    n_cases = len(all_cases)
    ok = 0
    total = 0
    field_mismatch = []
    for cap, (label, fn) in enumerate(all_cases):
        try:
            records = collect_configs(fn)
        except Exception:
            print(f'\n[{cap + 1:02d}/{n_cases}] {label}')
            print('    CALL FAILED:\n' + traceback.format_exc())
            continue
        reset()
        if not records:
            print(f'\n[{cap + 1:02d}/{n_cases}] {label}: no config printed')
            continue
        for (desc, layout, storage, pipeline, launch, info) in records:
            total += 1
            h = SM90(desc)
            # block_mult 只在对应用例里非 1
            bm_mult, bn_mult = (32, 32) if 'block_mult' in label else (1, 1)
            L, I, cands = h.best(bm_mult, bn_mult)
            S = h.storage_config(L)
            P = h.pipeline_config(L, S)
            LA = h.launch_config(L)

            def gi(x):
                return int(x)

            checks = [
                ('block_m', L['block_m'], gi(layout['block_m'])),
                ('block_n', L['block_n'], gi(layout['block_n'])),
                ('block_k', L['block_k'], gi(layout['block_k'])),
                ('cluster_m', L['cluster_m'], gi(layout['cluster_m'])),
                ('cluster_n', L['cluster_n'], gi(layout['cluster_n'])),
                ('swizzle_a', S['swizzle_a_mode'], gi(storage['swizzle_a_mode'])),
                ('swizzle_b', S['swizzle_b_mode'], gi(storage['swizzle_b_mode'])),
                ('swizzle_cd', S['swizzle_cd_mode'], gi(storage['swizzle_cd_mode'])),
                ('store_block_m', S['store_block_m'], gi(storage['store_block_m'])),
                ('smem_size', P['smem_size'], gi(pipeline['smem_size'])),
                ('num_stages', P['num_stages'], gi(pipeline['num_stages'])),
                ('num_threads', LA['num_threads'], gi(launch['num_threads'])),
                ('num_waves', I['num_waves'], gi(info['num_waves'])),
                ('last_wave_util', I['last_wave_util'], gi(info['last_wave_util'])),
                ('num_cycles', I['num_cycles'], gi(info['num_cycles'])),
            ]
            bad = [(n, a, b) for (n, a, b) in checks if a != b]
            status = 'PASS' if not bad else 'FAIL'
            if not bad:
                ok += 1
            else:
                field_mismatch.append((label, bad))
            gt = _MMA_KIND[int(desc['mma_kind'])]
            print(f'\n[{cap + 1:02d}/{n_cases}] {label}  -> {status}')
            print(f"    desc: gt={desc['gemm_type']} kt={desc['kernel_type']} "
                  f"m={desc['m']} n={desc['n']} k={desc['k']} G={desc['num_groups']} "
                  f"major=({desc['major_a']},{desc['major_b']}) kind={gt} "
                  f"a={desc['a_dtype']} cd={desc['cd_dtype']} acc={desc['with_accumulation']} "
                  f"sms={desc['num_sms']} exp=({desc['expected_m']},{desc['expected_n']},"
                  f"{desc['expected_k']},{desc['expected_num_groups']})")
            print(f"    候选数={len(cands)}  选出 {fmt_layout(L)}  waves={I['num_waves']} "
                  f"cycles={I['num_cycles']} stages={P['num_stages']} threads={LA['num_threads']}")
            print(f"    C++      {fmt_layout(layout)}  waves={info['num_waves']} "
                  f"cycles={info['num_cycles']} stages={pipeline['num_stages']} "
                  f"threads={launch['num_threads']}")
            if bad:
                for (n, a, b) in bad:
                    print(f'      MISMATCH {n}: py={a} cpp={b}')

    print('\n' + '=' * 100)
    print(f'汇总：{ok}/{total} 配置逐字段一致；{len(field_mismatch)} 个不一致')
    if field_mismatch:
        for (label, bad) in field_mismatch:
            print(f'  {label}: {bad}')
    show_ranking()
    print('done.')


if __name__ == '__main__':
    main()
