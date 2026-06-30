#include "tensorstore/internal/oauth2/google_service_account_auth_provider.h"
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/oauth2/bearer_token.h"
#include "tensorstore/internal/oauth2/oauth_utils.h"
#include "tensorstore/internal/oauth2/refreshable_auth_provider.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
namespace tensorstore {
namespace internal_oauth2 {
using ::tensorstore::Result;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;
constexpr char kOAuthV4Url[] = "https:
constexpr char kOAuthScope[] = "https:
GoogleServiceAccountAuthProvider::GoogleServiceAccountAuthProvider(
    const AccountCredentials& creds,
    std::shared_ptr<internal_http::HttpTransport> transport,
    std::function<absl::Time()> clock)
    : RefreshableAuthProvider(std::move(clock)),
      creds_(creds),
      uri_(kOAuthV4Url),
      scope_(kOAuthScope),
      transport_(std::move(transport)) {}
Result<HttpResponse> GoogleServiceAccountAuthProvider::IssueRequest(
    std::string_view method, std::string_view uri, absl::Cord payload) {
  return transport_
      ->IssueRequest(
          HttpRequestBuilder(method, std::string{uri})
              .AddHeader("Content-Type: application/x-www-form-urlencoded")
              .BuildRequest(),
          internal_http::IssueRequestOptions(std::move(payload)))
      .result();
}
Result<BearerTokenWithExpiration> GoogleServiceAccountAuthProvider::Refresh() {
  const auto now = GetCurrentTime();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto body,
      internal_oauth2::BuildSignedJWTRequest(
          creds_.private_key,
          internal_oauth2::BuildJWTHeader(creds_.private_key_id),
          internal_oauth2::BuildJWTClaimBody(creds_.client_email, scope_, uri_,
                                             now, 3600 )));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto response, IssueRequest("POST", uri_, absl::Cord(std::move(body))));
  TENSORSTORE_RETURN_IF_ERROR(HttpResponseCodeToStatus(response));
  TENSORSTORE_ASSIGN_OR_RETURN(auto result, internal_oauth2::ParseOAuthResponse(
                                                response.payload.Flatten()));
  return BearerTokenWithExpiration{std::move(result.access_token),
                                   now + absl::Seconds(result.expires_in)};
}
}  
}  