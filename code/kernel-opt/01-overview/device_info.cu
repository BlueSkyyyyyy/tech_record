// 01 开篇配套：打印本机 GPU 的关键参数，核对文章里用到的峰值常量。
//
// 运行：scripts/run.sh 01-overview/device_info.cu
#include "../common/cuda_utils.cuh"

int main() {
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  std::printf("visible devices: %d\n\n", ndev);
  for (int i = 0; i < ndev; ++i) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, i));
    DeviceInfo d = device_info(i);
    std::printf("================ device %d ================\n", i);
    print_device_info(d);
    std::printf("  warp size          : %d\n", p.warpSize);
    std::printf("  L2 cache           : %.2f MB\n", p.l2CacheSize / 1e6);
    std::printf("  shared/block(optin): %zu B\n", p.sharedMemPerBlockOptin);
    std::printf("  regs/block         : %d\n", p.regsPerBlock);
    std::printf("  asyncEngineCount   : %d\n", p.asyncEngineCount);
    int mem_clock_khz = 0, clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, i));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, i));
    std::printf("  SM clock / mem clk : %.0f MHz / %.0f MHz\n", clock_khz / 1e3,
                mem_clock_khz / 1e3);
    std::printf("  memory bus         : %d bit\n", p.memoryBusWidth);
    std::printf("  compute capability : %d.%d\n\n", p.major, p.minor);
  }
  return 0;
}
