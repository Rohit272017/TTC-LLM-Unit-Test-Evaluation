#include "tensorstore/internal/oauth2/gce_auth_provider.h"
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/internal/oauth2/bearer_token.h"
#include "tensorstore/internal/oauth2/oauth_utils.h"
#include "tensorstore/internal/oauth2/refreshable_auth_provider.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
ABSL_FLAG(std::optional<std::string>, tensorstore_gce_metadata_root,
          std::nullopt,
          "Url to used for http access metadata.google.internal. "
          "Overrides GCE_METADATA_ROOT.");
namespace tensorstore {
namespace internal_oauth2 {
namespace {
namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::internal::GetFlagOrEnvValue;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;
constexpr static auto ServiceAccountInfoBinder = jb::Object(
    jb::Member("email",
               jb::Projection(&GceAuthProvider::ServiceAccountInfo::email,
                              jb::NonEmptyStringBinder)),
    jb::Member("scopes",
               jb::Projection(&GceAuthProvider::ServiceAccountInfo::scopes)),
    jb::DiscardExtraMembers);
}  
std::string GceMetadataHostname() {
  return GetFlagOrEnvValue(FLAGS_tensorstore_gce_metadata_root,
                           "GCE_METADATA_ROOT")
      .value_or("metadata.google.internal");
}
GceAuthProvider::GceAuthProvider(
    std::shared_ptr<internal_http::HttpTransport> transport,
    const ServiceAccountInfo& service_account_info,
    std::function<absl::Time()> clock)
    : RefreshableAuthProvider(std::move(clock)),
      service_account_email_(service_account_info.email),
      scopes_(service_account_info.scopes.begin(),
              service_account_info.scopes.end()),
      transport_(std::move(transport)) {}
Result<HttpResponse> GceAuthProvider::IssueRequest(std::string path,
                                                   bool recursive) {
  HttpRequestBuilder request_builder(
      "GET", internal::JoinPath("http:
  request_builder.AddHeader("Metadata-Flavor: Google");
  if (recursive) {
    request_builder.AddQueryParameter("recursive", "true");
  }
  return transport_->IssueRequest(request_builder.BuildRequest(), {}).result();
}
Result<GceAuthProvider::ServiceAccountInfo>
GceAuthProvider::GetDefaultServiceAccountInfoIfRunningOnGce(
    internal_http::HttpTransport* transport) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto response,
      transport
          ->IssueRequest(
              HttpRequestBuilder(
                  "GET",
                  internal::JoinPath(
                      "http:
                      "/computeMetadata/v1/instance/service-accounts/default/"))
                  .AddHeader("Metadata-Flavor: Google")
                  .AddQueryParameter("recursive", "true")
                  .BuildRequest(),
              {})
          .result());
  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(response));
  auto info_response = internal::ParseJson(response.payload.Flatten());
  if (info_response.is_discarded()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Failed to parse service account response: ",
                            response.payload.Flatten()));
  }
  return jb::FromJson<ServiceAccountInfo>(info_response,
                                          ServiceAccountInfoBinder);
}
Result<BearerTokenWithExpiration> GceAuthProvider::Refresh() {
  const auto now = GetCurrentTime();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto response,
      IssueRequest(
          tensorstore::StrCat("/computeMetadata/v1/instance/service-accounts/",
                              service_account_email_, "/token"),
          false));
  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(response));
  TENSORSTORE_ASSIGN_OR_RETURN(auto result, internal_oauth2::ParseOAuthResponse(
                                                response.payload.Flatten()));
  return BearerTokenWithExpiration{std::move(result.access_token),
                                   now + absl::Seconds(result.expires_in)};
}
}  
}  