import os, sys, importlib.util
os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
import torch

def _imp(mod, fn):
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(os.path.dirname(here), "src", fn) if mod == "bwd" else os.path.join(here, fn)
    spec = importlib.util.spec_from_file_location(mod, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

m13 = _imp("m13", "check_port_pregdr_kernel.py")
m13._import_megatron()
bwd = _imp("bwd", "kda_pregdr_bwd.py")
m13.init_dist()
dtype = torch.float32
m13.model_parallel_cuda_manual_seed(123)
kda = m13.build_kda(m13.make_config(dtype))
dims = bwd._dims(kda)
print("dims:", dims)

cap = m13._capture(kda, dtype, 64, 2, seed=1)
q = cap["qkvfg"].detach().clone().requires_grad_(True)
be = cap["beta_in"].detach().clone().requires_grad_(True)
out = bwd.kda_pre_gdr_fused_native(kda, q, be, cap["batch"], cap["seq_len"])
print("out shapes:", [tuple(o.shape) for o in out[:6]])
# capture grads by hooking
gs = []
for o in out[:6]:
    o.register_hook(lambda g: gs.append(g))
sum(o.float().sum() for o in out[:6]).backward()
for g in gs:
    print("grad shape", tuple(g.shape), "stride", g.stride(), "contig", g.is_contiguous(), "dtype", g.dtype)
print("OK")
