#include "xla/tsl/profiler/rpc/client/remote_profiler_session_manager.h"
#include <cstddef>
#include <memory>
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/profiler/rpc/client/profiler_client.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace profiler {
using tensorflow::ProfileRequest;
using tensorflow::RemoteProfilerSessionManagerOptions;
 std::unique_ptr<RemoteProfilerSessionManager>
RemoteProfilerSessionManager::Create(
    const RemoteProfilerSessionManagerOptions& options,
    const ProfileRequest& request, absl::Status& out_status,
    AddressResolver resolver) {
  VLOG(1) << "Creating a RemoteProfilerSessionManager.";
  auto session_manager = absl::WrapUnique(
      new RemoteProfilerSessionManager(options, request, resolver));
  out_status = session_manager->Init();
  if (!out_status.ok()) {
    return nullptr;
  }
  return session_manager;
}
RemoteProfilerSessionManager::RemoteProfilerSessionManager(
    RemoteProfilerSessionManagerOptions options, ProfileRequest request,
    AddressResolver resolver)
    : options_(options), request_(request) {
  if (resolver) {
    resolver_ = resolver;
  } else {
    resolver_ = [](absl::string_view addr) { return std::string(addr); };
  }
}
RemoteProfilerSessionManager::~RemoteProfilerSessionManager() {
  VLOG(2) << "Destroying RemoteProfilerSessionManager.";
}
absl::Status RemoteProfilerSessionManager::Init() {
  mutex_lock lock(mutex_);
  VLOG(1) << "SessionManager initializing.";
  const absl::Time session_created_ts =
      absl::FromUnixNanos(options_.session_creation_timestamp_ns());
  const absl::Time deadline =
      session_created_ts +
      absl::Milliseconds(options_.max_session_duration_ms());
  LOG(INFO) << "Deadline set to " << deadline
            << " because max_session_duration_ms was "
            << options_.max_session_duration_ms()
            << " and session_creation_timestamp_ns was "
            << options_.session_creation_timestamp_ns() << " ["
            << session_created_ts << "]";
  clients_.reserve(options_.service_addresses_size());
  ProfileRequest request = request_;
  for (auto& service_address : options_.service_addresses()) {
    std::string resolved_service_address = resolver_(service_address);
    request.set_host_name(resolved_service_address);
    auto client = RemoteProfilerSession::Create(resolved_service_address,
                                                deadline, request);
    clients_.push_back(std::move(client));
  }
  LOG(INFO) << "Issued Profile gRPC to " << clients_.size() << " clients";
  return absl::OkStatus();
}
std::vector<RemoteProfilerSessionManager::Response>
RemoteProfilerSessionManager::WaitForCompletion() {
  mutex_lock lock(mutex_);
  std::vector<RemoteProfilerSessionManager::Response> remote_responses(
      clients_.size());
  for (int32_t idx = 0; idx < clients_.size(); ++idx) {
    auto& remote_response = remote_responses[idx];
    auto* client = clients_[idx].get();
    remote_response.profile_response =
        client->WaitForCompletion(remote_response.status);
    remote_response.service_address = std::string(client->GetServiceAddress());
  }
  return remote_responses;
}
}  
}  