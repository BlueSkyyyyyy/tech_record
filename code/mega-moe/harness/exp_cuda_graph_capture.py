#!/usr/bin/env python3
# P99-1 实验：DeepGEMM / DeepEP 的 CUDA Graph 兼容性面与 capture 前的 warm-up 契约。
#
# 三部分：
#   Part A  三个 API 的「per-stream 懒分配 + capture 前必须 warm-up」契约（Python 忠实复刻）
#   Part B  真实 DeepGEMM SM90 kernel 在 CUDA Graph 下的 capture/replay（自研 / cuBLASLt / FP8 / PDL）
#   Part C  为什么要把一串小 GEMM 放进 graph：发射开销实测
#   Part D  profiler 取证：graph replay 时到底发射了哪个 kernel
#
# 运行：bash harness/exp_cuda_graph_capture.sh
import torch
import deep_gemm

torch.manual_seed(2026)
DEV = "cuda"
PASS, FAIL = [], []


def check(cond, name, detail="", soft=False):
    (FAIL if not cond else PASS).append(name)
    tag = "PASS" if cond else ("SOFT" if soft else "FAIL")
    if cond:
        tag = "PASS"
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""))
    return cond


def hdr(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


def bitwise(x, y):
    return x.shape == y.shape and torch.equal(x, y)


# ----------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------
# 忠实复刻 csrc/apis/mega_mhc.hpp:26-39 的 get_split_barriers /
# csrc/apis/mega_gate.hpp:13-26 的 get_score_barriers /
# csrc/apis/attention.hpp:284-301 的 get_sparse_mqa_logits_workspace：
#   1. 按「当前 CUDAStream」在静态 map 里取一块 scratch（首次分配）；
#   2. 首次分配用 torch::zeros（会发射一条 zeroing kernel）；
#   3. 分配前断言 `c10::cuda::currentStreamCaptureStatusMayInitCtx() == CaptureStatus::None`。
# Python 里 cudaStreamIsCapturing 的等价物是 torch.cuda.is_current_stream_capturing()。
class ScratchByStream:
    def __init__(self, name):
        self.name = name
        self.by_stream = {}
        self.alloc_count = 0
        self.alloc_streams = []

    def get(self, nbytes):
        stream = torch.cuda.current_stream()
        t = self.by_stream.get(stream)
        if t is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"{self.name}: expected c10::cuda::CaptureStatus::None, "
                    f"warm up the stream before CUDA graph capture")
            t = torch.zeros(nbytes, dtype=torch.uint8, device=DEV)
            self.by_stream[stream] = t
            self.alloc_count += 1
            self.alloc_streams.append(stream.cuda_stream)
        return t


hdr("Part A  三个 API 的 per-stream 懒分配 + capture 前 warm-up 契约")

# A1 捕获状态 API
check(not torch.cuda.is_current_stream_capturing(), "A1a eager 时 is_current_stream_capturing()==False")
_in_capture = {}
with torch.cuda.stream(torch.cuda.Stream()):
    g_probe = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_probe):
        _in_capture["v"] = torch.cuda.is_current_stream_capturing()
check(_in_capture["v"] is True, "A1b capture 体内 is_current_stream_capturing()==True")

# A2 未 warm-up 的 stream 在 capture 中首次取 scratch -> 拒绝
scratch = ScratchByStream("get_split_barriers(clone)")
s_fresh = torch.cuda.Stream()
err = None
with torch.cuda.stream(s_fresh):
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=s_fresh):
            scratch.get(4096)
    except RuntimeError as e:
        err = str(e)
check(err is not None and "CaptureStatus::None" in err,
      "A2a 未 warm-up 的 stream 在 capture 中取 scratch 被拒绝", (err or "")[:110])
check(scratch.alloc_count == 0, "A2b 被拒后未发生分配（alloc_count==0）")

# A3 warm-up 后同一 stream 可 capture，且复用同一对象
with torch.cuda.stream(s_fresh):
    warm = scratch.get(4096)
check(scratch.alloc_count == 1 and warm.abs().max().item() == 0, "A3a warm-up 分配并清零 (alloc_count==1)")
ok_cap = True
with torch.cuda.stream(s_fresh):
    try:
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1, stream=s_fresh):
            scratch.get(4096)
    except RuntimeError:
        ok_cap = False
check(ok_cap, "A3b warm-up 后同一 stream 可成功 capture")
with torch.cuda.stream(s_fresh):
    again = scratch.get(4096)
check(again is warm, "A3c 同一 stream 复用同一 scratch 对象（is）")

# 不同 stream 各自分配
s_other = torch.cuda.Stream()
with torch.cuda.stream(s_other):
    other = scratch.get(4096)
check(other is not warm and scratch.alloc_count == 2,
      "A3d 新 stream 分配独立 scratch（alloc_count==2）", f"streams={scratch.alloc_streams}")

# A4 若不做 warm-up（即把 zeroing 留在 capture 里），清零会被录进 graph 并在每次 replay 重放
buf = torch.full((1 << 16,), 7.0, device=DEV)
s4 = torch.cuda.Stream()
with torch.cuda.stream(s4):
    buf.zero_()  # 先清零（模拟首次）
    g4 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g4, stream=s4):
        buf.zero_()  # 若 zeroing 落在 capture 内
buf.fill_(5.0)
g4.replay()
torch.cuda.synchronize()
check(buf.abs().max().item() == 0.0, "A4a capture 内的 zeroing 被录进 graph，replay 时重放（buf 归零）")

# A5 torch.empty 在 capture 中可从 graph-private 内存池分配（cuBLASLt workspace 的形态）
s5 = torch.cuda.Stream()
empty_ok, empty_bytes = True, 0
with torch.cuda.stream(s5):
    g5 = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g5, stream=s5):
            tmp = torch.empty(1 << 20, dtype=torch.uint8, device=DEV)  # 无 zeroing，无需断言
            tmp.fill_(3)
    except RuntimeError as e:
        empty_ok = False
        empty_bytes = str(e)[:80]
check(empty_ok, "A5 torch.empty 在 capture 中分配成功（无 zeroing，无需断言）", empty_bytes)

# ----------------------------------------------------------------------------
# Part B  真实 DeepGEMM SM90 kernel 在 CUDA Graph 下
# ----------------------------------------------------------------------------
hdr("Part B  真实 DeepGEMM SM90 kernel 的 capture / replay")
print("device:", torch.cuda.get_device_name(0), "| torch:", torch.__version__,
      "| deep_gemm:", deep_gemm.__file__)


def graph_roundtrip(name, call, fresh_stream_no_warm=False):
    """在独立 stream 上 capture 一次调用，replay 后与 eager 结果逐位比对。"""
    torch.cuda.synchronize()
    d_eager = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    call(d_eager)
    torch.cuda.synchronize()
    ref = d_eager.clone()

    s = torch.cuda.Stream()
    if not fresh_stream_no_warm:
        with torch.cuda.stream(s):
            call(d_eager)  # 该 stream 上 warm-up（模拟 host 契约）
        s.synchronize()

    d_graph = torch.empty_like(ref)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            call(d_graph)
    g.replay()
    torch.cuda.synchronize()
    ok = bitwise(ref, d_graph)
    d = d_graph - ref
    check(ok, name, f"max|Δ|={d.abs().max().item():.3e}")
    return g


# B1 自研 BF16（deterministic=True 关掉 cuBLASLt 回退）
M = N = K = 1024
a = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
b = torch.randn(N, K, dtype=torch.bfloat16, device=DEV)
deep_gemm.use_deterministic_algorithms(True)
graph_roundtrip("B1b bf16_gemm_nt（自研 sm90_bf16_gemm）capture+replay 逐位相等",
                lambda d: deep_gemm.bf16_gemm_nt(a, b, d))
deep_gemm.use_deterministic_algorithms(False)

# B2 默认（cuBLASLt 回退）—— 含「新 stream 上首次调用即在 capture 内」的 workspace 懒分配
graph_roundtrip("B2a bf16_gemm_nt（cuBLASLt 回退）capture+replay 逐位相等",
                lambda d: deep_gemm.bf16_gemm_nt(a, b, d))
graph_roundtrip("B2b cuBLASLt：全新 stream 上『首次调用即在 capture 内』成功（workspace 懒分配 capture-safe）",
                lambda d: deep_gemm.bf16_gemm_nt(a, b, d),
                fresh_stream_no_warm=True)

# B3 FP8 1D2D
a_fp8 = a.to(torch.float8_e4m3fn)
b_fp8 = b.to(torch.float8_e4m3fn)
a_sf = torch.ones(M, K // 128, dtype=torch.float, device=DEV)
b_sf = torch.ones(N // 128, K // 128, dtype=torch.float, device=DEV)
graph_roundtrip("B3 fp8_gemm_nt（sm90_fp8_gemm_1d2d）capture+replay 逐位相等",
                lambda d: deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d))

# B4 PDL 打开时
deep_gemm.set_pdl(True)
deep_gemm.use_deterministic_algorithms(True)
try:
    graph_roundtrip("B4 set_pdl(True) 下 capture+replay 逐位相等（PDL 属性 capture-safe）",
                    lambda d: deep_gemm.bf16_gemm_nt(a, b, d))
finally:
    deep_gemm.use_deterministic_algorithms(False)
    deep_gemm.set_pdl(False)

# B5 两条独立 stream 各自的 graph
deep_gemm.use_deterministic_algorithms(True)
sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
dA = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
dB = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
with torch.cuda.stream(sA):
    deep_gemm.bf16_gemm_nt(a, b, dA)
with torch.cuda.stream(sB):
    deep_gemm.bf16_gemm_nt(a, b, dB)
torch.cuda.synchronize()
ref = dA.clone()
gA, gB = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
with torch.cuda.stream(sA):
    with torch.cuda.graph(gA, stream=sA):
        deep_gemm.bf16_gemm_nt(a, b, dA)
with torch.cuda.stream(sB):
    with torch.cuda.graph(gB, stream=sB):
        deep_gemm.bf16_gemm_nt(a, b, dB)
dA.zero_(); dB.zero_()
gA.replay(); gB.replay()
torch.cuda.synchronize()
check(bitwise(ref, dA) and bitwise(ref, dB), "B5 两 stream 各自的 graph replay 结果一致")
deep_gemm.use_deterministic_algorithms(False)

# ----------------------------------------------------------------------------
# Part C  发射开销：一串小 GEMM eager vs graph replay
# ----------------------------------------------------------------------------
hdr("Part C  小 GEMM ×N：eager 连发 vs graph replay")
deep_gemm.use_deterministic_algorithms(True)
sm = sn = sk = 256
asm = torch.randn(sm, sk, dtype=torch.bfloat16, device=DEV)
bsm = torch.randn(sn, sk, dtype=torch.bfloat16, device=DEV)
dS = torch.empty(sm, sn, dtype=torch.bfloat16, device=DEV)
deep_gemm.bf16_gemm_nt(asm, bsm, dS)  # warm-up / JIT
torch.cuda.synchronize()
NUM = 64


def timed(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def eager_seq():
    for _ in range(NUM):
        deep_gemm.bf16_gemm_nt(asm, bsm, dS)


sC = torch.cuda.Stream()
with torch.cuda.stream(sC):
    for _ in range(NUM):
        deep_gemm.bf16_gemm_nt(asm, bsm, dS)
sC.synchronize()
gC = torch.cuda.CUDAGraph()
with torch.cuda.stream(sC):
    with torch.cuda.graph(gC, stream=sC):
        for _ in range(NUM):
            deep_gemm.bf16_gemm_nt(asm, bsm, dS)


def graph_seq():
    gC.replay()


t_eager = timed(eager_seq)
t_graph = timed(graph_seq)
print(f"  N={NUM} 次 256x256x256 bf16：eager {t_eager*1000:.1f}us/iter（{t_eager/NUM*1000:.2f}us/call）"
      f" | graph {t_graph*1000:.1f}us/iter（{t_graph/NUM*1000:.2f}us/call）"
      f" | 加速 {t_eager/t_graph:.2f}x")
check(t_graph < t_eager, "C1 graph replay 比 eager 连发快", f"{t_eager/t_graph:.2f}x")
deep_gemm.use_deterministic_algorithms(False)

# ----------------------------------------------------------------------------
# Part D  profiler 取证：replay 时真正发射的 kernel
# ----------------------------------------------------------------------------
hdr("Part D  profiler：graph replay 发射的 kernel 名")
names = []
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    gC.replay()
    torch.cuda.synchronize()
for evt in prof.key_averages():
    if evt.device_type == torch.autograd.DeviceType.CUDA and evt.count > 0:
        names.append((evt.key, evt.count))
# 按出现次数排序，取前几个
names.sort(key=lambda kv: -kv[1])
for k, c in names[:6]:
    print(f"  {c:>4}x  {k[:90]}")
check(any("sm90_bf16_gemm" in k or "gemm" in k.lower() for k, _ in names),
      "D1 profiler 抓到 graph 内真实 GEMM kernel")

# ----------------------------------------------------------------------------
hdr("汇总")
print(f"PASS={len(PASS)}  FAIL={len(FAIL)}")
if FAIL:
    print("FAIL 项：", FAIL)
