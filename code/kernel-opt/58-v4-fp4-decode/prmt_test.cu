#include <cstdio>
#include <cstdint>
__device__ __forceinline__ int e2m1_i8(unsigned n){const unsigned e=(n>>1)&3u,m=n&1u,s=(n>>3)&1u;int v=(int)(((e?2u:0u)+m)<<(e?e-1:0u));return s?-v:v;}
__device__ __forceinline__ uint32_t decode4(uint32_t e){
  const uint32_t k=e&0x07070707u;
  const uint32_t s=(e>>3)&0x01010101u;
  const uint32_t pk=(k&0x0000000Fu)|((k>>4)&0x000000F0u)|((k>>8)&0x00000F00u)|((k>>12)&0x0000F000u);
  const uint32_t ps=(s&0x1u)|((s>>4)&0x10u)|((s>>8)&0x100u)|((s>>12)&0x1000u);
  const uint32_t p1=__byte_perm(0x03020100u,0x0C080604u,pk);
  const uint32_t p2=__byte_perm(0xFDFEFF00u,0xF4F8FAFCu,pk);
  const uint32_t sel=0x3210u|(ps<<2);
  return __byte_perm(p1,p2,sel);
}
__global__ void t(){
  unsigned bad=0;
  for(unsigned long long w=0; w<(1ull<<28); w+=2654435761ull%0x10000000){
    uint32_t w32=(uint32_t)w;
    uint32_t lo=w32&0x0F0F0F0Fu, hi=(w32>>4)&0x0F0F0F0Fu;
    uint32_t a=decode4(__byte_perm(lo,hi,0x5140)), b=decode4(__byte_perm(lo,hi,0x7362));
    unsigned char* pa=(unsigned char*)&a; unsigned char* pb=(unsigned char*)&b;
    for(int i=0;i<4;i++){int got=(int)(signed char)pa[i]; int want=e2m1_i8((w32>>(4*i))&0xF); if(got!=want){if(bad<5)printf("lo w=%08x i=%d got=%d want=%d\n",w32,i,got,want); bad++;}}
    for(int i=0;i<4;i++){int got=(int)(signed char)pb[i]; int want=e2m1_i8((w32>>(4*(i+4)))&0xF); if(got!=want){if(bad<5)printf("hi w=%08x i=%d got=%d want=%d\n",w32,i,got,want); bad++;}}
  }
  printf("bad=%u\n",bad);
}
int main(){t<<<1,1>>>();cudaDeviceSynchronize();return 0;}
