// probe: host-side e4m3 conversion + fp8 TMA/wgmma environment
#include "../common/cuda_utils.cuh"
#include <cuda.h>
#include <cuda_fp8.h>
#include <cstdio>

int main() {
  DeviceInfo d = device_info(0);
  print_device_info(d);
  // host conversion: float -> e4m3 -> float
  float xs[] = {0.f, 1.f, -2.5f, 0.026f, 448.f, 0.001f};
  for (float x : xs) {
    __nv_fp8_e4m3 q(x);
    float back = (float)q;
    std::printf("x=%12.6f  e4m3=0x%02x  back=%12.6f\n", x, (unsigned)(unsigned char)q.__x, back);
  }
  // tensor core fp8 peak via cudaGetDeviceProperties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::printf("cc=%d.%d smem/block optin=%zu\n", prop.major, prop.minor, prop.sharedMemPerBlockOptin);
  return 0;
}
