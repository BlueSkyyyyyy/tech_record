// P177-1（§102）附带核实：PTX 是否有「S2S 的 cp.async.bulk.tensor」形式？
// 结论：没有。ptxas 对 `cp.async.bulk.tensor.2d.shared::cluster.shared::cta` 报
//   "State space incorrect for instruction 'cp.async.bulk.tensor'"
// 即 `.tensor` 家族只支持 `.global` 源；S2S 的非-tensor `cp.async.bulk` 只有
//   `shared::cta.shared::cluster`（线性）一种形式。
// 故「S2S 非连续/tensor 搬运」在 PTX 层不存在，只能用多笔线性 cp.async.bulk 模拟。
// 复现：nvcc -O3 -arch=sm_90a -c harness/tensor_s2s_ptx_probe.cu -o /tmp/t.o
#include <cstdint>
__global__ void k(void* p) {
  unsigned a = 0, cc = 0;
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.shared::cta [%0], [%1, {%2, %3}], [%4];"
               :: "r"(a), "l"((uint64_t)p), "r"(cc), "r"(cc), "r"(cc) : "memory");
}
