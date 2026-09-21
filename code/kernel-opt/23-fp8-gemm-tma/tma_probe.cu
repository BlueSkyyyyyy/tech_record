#include <cuda.h>
#include <cstdio>
int main(){
  CUtensorMap tm;
  void* p=nullptr; cudaMalloc(&p,1024);
  cuuint64_t dims[2]={128,128}, strides[1]={128};
  cuuint32_t box[2]={128,128}, es[2]={1,1};
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p,
      dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  printf("res=%d\n",(int)r);
  return 0;
}
