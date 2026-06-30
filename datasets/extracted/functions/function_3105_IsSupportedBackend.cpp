#include "tensorflow/compiler/tf2xla/tf2xla_opset.h"
#include <algorithm>
#include <string>
#include <vector>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
namespace tensorflow {
const int SUPPORTED_DEVICES_NUM = 2;
static const char* const SUPPORTED_DEVICES[SUPPORTED_DEVICES_NUM] = {
    DEVICE_GPU_XLA_JIT, DEVICE_CPU_XLA_JIT};
bool IsSupportedBackend(absl::string_view device_name) {
  for (int i = 0; i < SUPPORTED_DEVICES_NUM; i++) {
    if (SUPPORTED_DEVICES[i] == device_name) return true;
  }
  return false;
}
absl::Status RegisterBackends(absl::string_view device_name) {
  if (!IsSupportedBackend(device_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat(device_name, " is not supported.  Supported devices are ",
                     absl::StrJoin(SUPPORTED_DEVICES, ", ")));
  }
  auto op_filter = [](KernelDef* kdef) {
    if (kdef->op() == "Const") {
      AddDtypeToKernelDefConstraint("dtype", DT_STRING, kdef);
    }
    if (kdef->op() == "Assert") {
      AddDtypeToKernelDefConstraint("T", DT_STRING, kdef);
    }
    return true;
  };
  if (!XlaOpRegistry::IsBackendRegistered(DEVICE_GPU_XLA_JIT)) {
    static auto gpu_backend =
        XlaBackendRegistrar(DEVICE_GPU_XLA_JIT, kGpuAllTypes, op_filter);
  }
  if (!XlaOpRegistry::IsBackendRegistered(DEVICE_CPU_XLA_JIT)) {
    static auto cpu_backend =
        XlaBackendRegistrar(DEVICE_CPU_XLA_JIT, kCpuAllTypes, op_filter);
  }
  if (!XlaOpRegistry::IsBackendRegistered(std::string(device_name))) {
    return absl::InternalError(
        absl::StrCat(device_name, " is not registered."));
  }
  return absl::OkStatus();
}
absl::StatusOr<std::vector<std::string>> GetRegisteredXlaOpsForDevice(
    absl::string_view device_name) {
  auto status = RegisterBackends(device_name);
  if (!status.ok()) return status;
  std::vector<const KernelDef*> kernel_defs =
      XlaOpRegistry::DeviceKernels(std::string(device_name), true);
  std::vector<std::string> op_names;
  op_names.reserve(kernel_defs.size());
  for (const auto& kernel_def : kernel_defs) {
    op_names.push_back(kernel_def->op());
  }
  std::sort(op_names.begin(), op_names.end());
  return op_names;
}
}  