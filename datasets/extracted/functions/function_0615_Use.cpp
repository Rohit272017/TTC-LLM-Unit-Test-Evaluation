#include "tensorstore/internal/grpc/server_credentials.h"
#include <memory>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/util/result.h"
namespace tensorstore {
namespace {
ABSL_CONST_INIT static absl::Mutex credentials_mu(absl::kConstInit);
const internal::ContextResourceRegistration<GrpcServerCredentials>
    grpc_server_credentials_registration;
}  
bool GrpcServerCredentials::Use(
    tensorstore::Context context,
    std::shared_ptr<::grpc::ServerCredentials> credentials) {
  auto resource = context.GetResource<GrpcServerCredentials>().value();
  absl::MutexLock l(&credentials_mu);
  bool result = (resource->credentials_ == nullptr);
  resource->credentials_ = std::move(credentials);
  return result;
}
std::shared_ptr<::grpc::ServerCredentials>
GrpcServerCredentials::Resource::GetCredentials() {
  absl::MutexLock l(&credentials_mu);
  if (credentials_) return credentials_;
  return grpc::InsecureServerCredentials();
}
}  