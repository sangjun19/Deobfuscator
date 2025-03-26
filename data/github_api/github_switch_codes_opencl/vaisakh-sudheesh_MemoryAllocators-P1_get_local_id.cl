// Repository: vaisakh-sudheesh/MemoryAllocators-P1
// File: opensource/llvm-project-llvmorg-19.1.1/libclc/amdgcn/lib/workitem/get_local_id.cl

#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD size_t get_local_id(uint dim) {
  switch (dim) {
  case 0:
    return __builtin_amdgcn_workitem_id_x();
  case 1:
    return __builtin_amdgcn_workitem_id_y();
  case 2:
    return __builtin_amdgcn_workitem_id_z();
  default:
    return 1;
  }
}
