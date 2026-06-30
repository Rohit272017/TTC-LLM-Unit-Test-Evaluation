#include "tensorflow/core/tfrt/common/pjrt_state.h"
#include <memory>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tf_pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_options.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_registry.h"
#include "tsl/platform/statusor.h"
namespace tensorflow {
PjRtState* PjRtState::Create() { return new PjRtState(); }
absl::StatusOr<xla::PjRtClient*> PjRtState::GetPjRtClient(
    const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    return it->second.get();
  }
  return errors::NotFound("PjRt client not found for device type ",
                          device_type);
}
absl::StatusOr<xla::PjRtClient*> PjRtState::GetOrCreatePjRtClient(
    const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    return it->second.get();
  }
  std::unique_ptr<xla::PjRtClient> pjrt_client;
  xla::PjrtClientFactoryOptions options = xla::PjrtClientFactoryOptions();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      xla::PjrtClientFactoryRegistry::Get().GetPjrtClient(
                          device_type, options));
  pjrt_client = xla::TfPjRtClient::CreateTfPjRtClient(std::move(client));
  clients_[device_type] = std::move(pjrt_client);
  return clients_[device_type].get();
}
Status PjRtState::SetPjRtClient(const DeviceType& device_type,
                                std::unique_ptr<xla::PjRtClient> client) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    unused_.push_back(std::move(it->second));
  }
  clients_[device_type] = std::move(client);
  return absl::OkStatus();
}
Status PjRtState::MovePjRtClientToUnused(const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    unused_.push_back(std::move(it->second));
    clients_.erase(it);
    return absl::OkStatus();
  }
  return errors::NotFound("PjRt client not found for device type ",
                          device_type);
}
Status PjRtState::SetPjRtGpuClientCreationInfo(
    std::unique_ptr<PjRtGpuClientCreationInfo> info) {
  absl::MutexLock lock(&mu_);
  pjrt_gpu_client_creation_info_ = std::move(info);
  return absl::OkStatus();
}
PjRtGpuClientCreationInfo* PjRtState::GetPjRtGpuClientCreationInfo() {
  absl::MutexLock lock(&mu_);
  return pjrt_gpu_client_creation_info_.get();
}
string PjRtState::DebugString() const { return "PjRtState"; }
}  