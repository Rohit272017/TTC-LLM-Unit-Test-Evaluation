#include "tensorstore/internal/grpc/client_credentials.h"
#include <memory>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/credentials.h"  
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
namespace tensorstore {
namespace {
ABSL_CONST_INIT static absl::Mutex credentials_mu(absl::kConstInit);
const internal::ContextResourceRegistration<GrpcClientCredentials>
    grpc_client_credentials_registration;
}  
bool GrpcClientCredentials::Use(
    tensorstore::Context context,
    std::shared_ptr<::grpc::ChannelCredentials> credentials) {
  auto resource = context.GetResource<GrpcClientCredentials>().value();
  absl::MutexLock l(&credentials_mu);
  bool result = (resource->credentials_ == nullptr);
  resource->credentials_ = std::move(credentials);
  return result;
}
std::shared_ptr<::grpc::ChannelCredentials>
GrpcClientCredentials::Resource::GetCredentials() {
  absl::MutexLock l(&credentials_mu);
  if (credentials_) return credentials_;
  return grpc::InsecureChannelCredentials();
}
}  