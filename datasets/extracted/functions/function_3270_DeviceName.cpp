#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <variant>
#include <vector>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/refcount.h"
namespace tensorflow {
namespace tfrt_stub {
namespace {
string DeviceName(absl::string_view name_prefix, absl::string_view device_type,
                  int32_t task_id, size_t device_id) {
  return strings::StrCat(absl::StripSuffix(name_prefix, "0"), task_id,
                         "/device:", device_type, ":", device_id);
}
DeviceAttributes BuildDeviceAttributes(absl::string_view name_prefix,
                                       const char *device_type, int32_t task_id,
                                       size_t device_id) {
  const DeviceAttributes attrs = Device::BuildDeviceAttributes(
      DeviceName(name_prefix, device_type, task_id, device_id),
      DeviceType(device_type), Bytes(16ULL << 30), DeviceLocality(),
      strings::StrCat("device: ", device_type, " device"));
  return attrs;
}
}  
absl::StatusOr<std::unique_ptr<FallbackState>> FallbackState::Create(
    const SessionOptions &session_options,
    const tensorflow::FunctionDefLibrary &fdef_lib) {
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  return std::make_unique<FallbackState>(session_options, std::move(devices),
                                         fdef_lib);
}
absl::StatusOr<std::unique_ptr<FallbackState>>
FallbackState::CreateWithCpuDevice(
    const SessionOptions &session_options,
    const tensorflow::FunctionDefLibrary &fdef_lib) {
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddCpuDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  return std::make_unique<FallbackState>(session_options, std::move(devices),
                                         fdef_lib);
}
absl::StatusOr<std::unique_ptr<FallbackState>>
FallbackState::CreateWithMockGpuDevice(
    const SessionOptions &session_options,
    const tensorflow::FunctionDefLibrary &fdef_lib) {
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddCpuDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  auto device_attrs =
      BuildDeviceAttributes("/job:localhost/replica:0/task:0", "GPU", 0, 0);
  devices.push_back(
      std::make_unique<VirtualDevice>(session_options.env, device_attrs));
  return std::make_unique<FallbackState>(session_options, std::move(devices),
                                         fdef_lib);
}
absl::StatusOr<std::unique_ptr<FallbackState>>
FallbackState::CreateWithDeviceMgr(
    const SessionOptions &session_options,
    const tensorflow::FunctionDefLibrary &fdef_lib,
    absl::Nonnull<DynamicDeviceMgr *> device_mgr) {
  return std::make_unique<FallbackState>(session_options, device_mgr, fdef_lib);
}
FallbackState::FallbackState(const SessionOptions &session_options,
                             std::variant<std::vector<std::unique_ptr<Device>>,
                                          absl::Nonnull<DynamicDeviceMgr *>>
                                 device_mgr,
                             const tensorflow::FunctionDefLibrary &fdef_lib)
    : session_options_(session_options),
      device_manager_(
          std::holds_alternative<std::vector<std::unique_ptr<Device>>>(
              device_mgr)
              ? std::move(
                    std::get<std::vector<std::unique_ptr<Device>>>(device_mgr))
              : std::vector<std::unique_ptr<Device>>()),
      device_manager_ptr_(
          std::holds_alternative<absl::Nonnull<DynamicDeviceMgr *>>(device_mgr)
              ? std::get<absl::Nonnull<DynamicDeviceMgr *>>(device_mgr)
              : &device_manager_),
      func_lib_def_(OpRegistry::Global(), fdef_lib),
      pflr_(device_manager_ptr_, session_options.env, &session_options.config,
            TF_GRAPH_DEF_VERSION, &func_lib_def_,
            session_options.config.graph_options().optimizer_options(),
            nullptr, nullptr,
            nullptr,
            Rendezvous::Factory{[](const int64_t, const DeviceMgr *device_mgr,
                                   tsl::core::RefCountPtr<Rendezvous> *r) {
              *r = tsl::core::RefCountPtr<Rendezvous>(
                  new IntraProcessRendezvous(device_mgr));
              return absl::OkStatus();
            }}) {
  for (auto *d : device_manager_ptr_->ListDevices()) {
    device_set_.AddDevice(d);
  }
  device_set_.set_client_device(device_manager().HostCPU());
}
absl::StatusOr<std::unique_ptr<GraphExecutionState>>
FallbackState::CreateGraphExecutionState(GraphDef graph_def,
                                         bool run_placer) const {
  GraphExecutionStateOptions options;
  options.device_set = &device_set_;
  options.session_options = &session_options_;
  options.session_handle = "tfrt_fallback_handle";
  options.run_placer = run_placer;
  std::unique_ptr<GraphExecutionState> execution_state;
  TF_RETURN_IF_ERROR(GraphExecutionState::MakeForBaseGraph(
      std::move(graph_def), options, &execution_state));
  return execution_state;
}
absl::Status FallbackState::AddFunctionDef(const FunctionDef &func_def) {
  return func_lib_def_.AddFunctionDef(func_def);
}
}  
}  