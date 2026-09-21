#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
using fp8 = __nv_fp8_e4m3;
__global__ void k(const float* in, float* out, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){ __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(in[i], __NV_SATFINITE, __NV_E4M3);
           __half h = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3); out[i]=__half2float(h); }
}
int main(){
  const int n=6; float h[n]={1.0f,1.13f,-0.7f,201.f,256.f,448.f}; float *d,*o;
  cudaMalloc(&d,n*4); cudaMalloc(&o,n*4); cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice);
  k<<<1,32>>>(d,o,n); cudaDeviceSynchronize();
  float r[n]; cudaMemcpy(r,o,n*4,cudaMemcpyDeviceToHost);
  for(int i=0;i<n;i++) printf("%.3f -> %.3f\n", h[i], r[i]);
  return 0;
}
