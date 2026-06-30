#ifdef INTEL_MKL
#include "tensorflow/core/common_runtime/mkl_cpu_allocator.h"
namespace tensorflow {
constexpr const char* MklCPUAllocator::kMaxLimitStr;
constexpr const size_t MklCPUAllocator::kDefaultMaxLimit;
}  
#endif  