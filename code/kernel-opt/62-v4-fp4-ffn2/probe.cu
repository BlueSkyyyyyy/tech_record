#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
__global__ void t(uint32_t sel, uint32_t* o){
  // a bytes = 0xA0..0xA3, b bytes = 0xB0..0xB3 tagged
  o[0]=__byte_perm(0x33323130u, 0x37363534u, sel);
}
int main(){
  uint32_t* d; cudaMalloc(&d,4); uint32_t o;
  for(uint32_t v=0; v<16; ++v){
    uint32_t sel = v | (v<<4) | (v<<8) | (v<<12);
    t<<<1,1>>>(sel,d); cudaMemcpy(&o,d,4,cudaMemcpyDeviceToHost);
    printf("field=%2u -> bytes %u %u %u %u\n", v, o&0xFF,(o>>8)&0xFF,(o>>16)&0xFF,(o>>24)&0xFF);
  }
}
