#include "tsl/platform/rocm_rocdl_path.h"
#include <stdlib.h>
#include "tsl/platform/path.h"
#if !defined(PLATFORM_GOOGLE) && TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif
#include "tsl/platform/logging.h"
namespace tsl {
std::string RocmRoot() {
#if TENSORFLOW_USE_ROCM
  if (const char* rocm_path_env = std::getenv("ROCM_PATH")) {
    VLOG(3) << "ROCM root = " << rocm_path_env;
    return rocm_path_env;
  } else {
    VLOG(3) << "ROCM root = " << TF_ROCM_TOOLKIT_PATH;
    return TF_ROCM_TOOLKIT_PATH;
  }
#else
  return "";
#endif
}
std::string RocdlRoot() {
  if (const char* device_lib_path_env = std::getenv("HIP_DEVICE_LIB_PATH")) {
    return device_lib_path_env;
  } else {
    return io::JoinPath(RocmRoot(), "amdgcn/bitcode");
  }
}
}  