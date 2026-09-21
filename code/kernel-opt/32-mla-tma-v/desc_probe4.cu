#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;
using bf16 = cutlass::bfloat16_t;
// K-major SW128 手写（元素单位，K=64）
static int k_off(int row, int k, int K){
  int rg=row>>3, rr=row&7, kg=k>>6, kk=k&63, c=kk>>3, cc=c^rr;
  return (rg*(K>>6)+kg)*512 + (rr*8+cc)*8 + (kk&7);
}
int main(){
  { auto lay = tile_to_shape(SM90::GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<_64,_64>{});
    printf("K layout: "); print(lay); printf("\n");
    int bad=0;
    for(int row=0;row<64;++row)for(int k=0;k<64;++k){int c=(int)lay(make_coord(row,k)); int m=k_off(row,k,64); if(c!=m){if(bad<5)printf("  K (row=%d,k=%d) cute=%d mine=%d\n",row,k,c,m);++bad;}}
    printf("K mismatches=%d/4096\n",bad);
  }
  { auto lay = tile_to_shape(SM90::GMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<_256,_16>{});
    printf("MN layout: "); print(lay); printf("\n");
    // 打印 n=0..71,k=0 与 n=0,k=0..15
    for(int n=0;n<72;++n) if(n%8==0) printf("  n=%d,k=0 -> %d\n",n,(int)lay(make_coord(n,0)));
    for(int k=0;k<16;++k) printf("  n=0,k=%d -> %d\n",k,(int)lay(make_coord(0,k)));
  }
  return 0;
}
