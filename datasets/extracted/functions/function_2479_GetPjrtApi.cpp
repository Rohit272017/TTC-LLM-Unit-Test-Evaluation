#include "xla/pjrt/c/pjrt_c_api_cpu.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"
const PJRT_Api* GetPjrtApi() { return pjrt::cpu_plugin::GetCpuPjrtApi(); }