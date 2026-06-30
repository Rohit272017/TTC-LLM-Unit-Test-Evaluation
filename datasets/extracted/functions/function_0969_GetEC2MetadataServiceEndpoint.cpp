#include "tensorstore/kvstore/s3/credentials/ec2_credential_provider.h"
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/kvstore/s3/s3_metadata.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/std_optional.h"
ABSL_FLAG(std::optional<std::string>,
          tensorstore_aws_ec2_metadata_service_endpoint, std::nullopt,
          "Endpoint to used for http access AWS metadata service. "
          "Overrides AWS_EC2_METADATA_SERVICE_ENDPOINT.");
using ::tensorstore::Result;
using ::tensorstore::internal::GetFlagOrEnvValue;
using ::tensorstore::internal::ParseJson;
using ::tensorstore::internal_http::HttpRequestBuilder;
namespace jb = tensorstore::internal_json_binding;
namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {
static constexpr char kMetadataTokenHeader[] = "x-aws-ec2-metadata-token:";
static constexpr char kIamCredentialsPath[] =
    "/latest/meta-data/iam/security-credentials/";
static constexpr absl::Duration kConnectTimeout = absl::Milliseconds(200);
static constexpr absl::Duration kDefaultTimeout = absl::Minutes(5);
static constexpr char kSuccess[] = "Success";
std::string GetEC2MetadataServiceEndpoint() {
  return GetFlagOrEnvValue(FLAGS_tensorstore_aws_ec2_metadata_service_endpoint,
                           "AWS_EC2_METADATA_SERVICE_ENDPOINT")
      .value_or("http:
}
struct EC2CredentialsResponse {
  std::string code;
  std::optional<absl::Time> last_updated;
  std::optional<std::string> type;
  std::optional<std::string> access_key_id;
  std::optional<std::string> secret_access_key;
  std::optional<std::string> token;
  std::optional<absl::Time> expiration;
};
inline constexpr auto EC2CredentialsResponseBinder = jb::Object(
    jb::Member("Code", jb::Projection(&EC2CredentialsResponse::code)),
    jb::OptionalMember("LastUpdated",
                       jb::Projection(&EC2CredentialsResponse::last_updated)),
    jb::OptionalMember("Type", jb::Projection(&EC2CredentialsResponse::type)),
    jb::OptionalMember("AccessKeyId",
                       jb::Projection(&EC2CredentialsResponse::access_key_id)),
    jb::OptionalMember(
        "SecretAccessKey",
        jb::Projection(&EC2CredentialsResponse::secret_access_key)),
    jb::OptionalMember("Token", jb::Projection(&EC2CredentialsResponse::token)),
    jb::OptionalMember("Expiration",
                       jb::Projection(&EC2CredentialsResponse::expiration)));
Result<absl::Cord> GetEC2ApiToken(std::string_view endpoint,
                                  internal_http::HttpTransport& transport) {
  const std::string token_url =
      tensorstore::StrCat(endpoint, "/latest/api/token");
  const std::string request_header =
      "x-aws-ec2-metadata-token-ttl-seconds: 21600";
  const auto request_options = internal_http::IssueRequestOptions()
                                   .SetRequestTimeout(absl::InfiniteDuration())
                                   .SetConnectTimeout(kConnectTimeout);
  for (auto method : {std::string_view("POST"), std::string_view("PUT")}) {
    auto token_request = HttpRequestBuilder(method, token_url)
                             .AddHeader(request_header)
                             .BuildRequest();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto token_response,
        transport.IssueRequest(token_request, request_options).result());
    if (method == "POST" && (token_response.status_code == 405 ||
                             token_response.status_code == 401)) {
      continue;
    }
    bool is_retryable = false;
    TENSORSTORE_RETURN_IF_ERROR(
        AwsHttpResponseToStatus(token_response, is_retryable));
    return std::move(token_response.payload);
  }
  return absl::NotFoundError(
      "Failed to obtain EC2 API token from either IMDSv1 or IMDSv2");
}
}  
Result<AwsCredentials> EC2MetadataCredentialProvider::GetCredentials() {
  if (endpoint_.empty()) {
    endpoint_ = GetEC2MetadataServiceEndpoint();
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto api_token,
                               GetEC2ApiToken(endpoint_, *transport_));
  auto token_header = tensorstore::StrCat(kMetadataTokenHeader, api_token);
  auto iam_role_request =
      HttpRequestBuilder("GET",
                         tensorstore::StrCat(endpoint_, kIamCredentialsPath))
          .AddHeader(token_header)
          .BuildRequest();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto iam_role_response,
      transport_->IssueRequest(iam_role_request, {}).result());
  auto iam_role_plain_text = iam_role_response.payload.Flatten();
  bool is_retryable = false;
  TENSORSTORE_RETURN_IF_ERROR(
      AwsHttpResponseToStatus(iam_role_response, is_retryable));
  std::vector<std::string_view> iam_roles =
      absl::StrSplit(iam_role_plain_text, '\n', absl::SkipWhitespace());
  if (iam_roles.empty()) {
    return absl::NotFoundError("Empty EC2 Role list");
  }
  auto iam_credentials_request_url =
      tensorstore::StrCat(endpoint_, kIamCredentialsPath, iam_roles[0]);
  auto iam_credentials_request =
      HttpRequestBuilder("GET", iam_credentials_request_url)
          .AddHeader(token_header)
          .BuildRequest();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto iam_credentials_response,
      transport_->IssueRequest(iam_credentials_request, {}).result());
  auto iam_credentials_plain_text = iam_credentials_response.payload.Flatten();
  TENSORSTORE_RETURN_IF_ERROR(
      AwsHttpResponseToStatus(iam_credentials_response, is_retryable));
  auto json_credentials = ParseJson(iam_credentials_plain_text);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto iam_credentials,
      jb::FromJson<EC2CredentialsResponse>(json_credentials,
                                           EC2CredentialsResponseBinder));
  if (iam_credentials.code != kSuccess) {
    return absl::NotFoundError(
        absl::StrCat("EC2Metadata request to [", iam_credentials_request_url,
                     "] failed with code ", iam_credentials.code));
  }
  auto default_timeout = absl::Now() + kDefaultTimeout;
  auto expires_at =
      iam_credentials.expiration.value_or(default_timeout) - absl::Seconds(60);
  return AwsCredentials{iam_credentials.access_key_id.value_or(""),
                        iam_credentials.secret_access_key.value_or(""),
                        iam_credentials.token.value_or(""), expires_at};
}
}  
}  