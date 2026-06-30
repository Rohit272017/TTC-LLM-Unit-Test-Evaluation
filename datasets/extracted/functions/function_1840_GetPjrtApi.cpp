#include "xla/pjrt/c/pjrt_c_api_gpu.h"
#include "absl/base/call_once.h"
#include "absl/log/initialize.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_internal.h"
#include "tsl/platform/platform.h"
const PJRT_Api* GetPjrtApi() {
#ifndef PLATFORM_GOOGLE
  static absl::once_flag once;
  absl::call_once(once, []() { absl::InitializeLog(); });
#endif  
  return pjrt::gpu_plugin::GetGpuPjrtApi();
}