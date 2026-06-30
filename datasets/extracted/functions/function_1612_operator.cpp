#include "tensorstore/kvstore/s3/s3_endpoint.h"
#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {
static constexpr char kAmzBucketRegionHeader[] = "x-amz-bucket-region";
struct S3VirtualHostFormatter {
  std::string GetEndpoint(std::string_view bucket,
                          std::string_view aws_region) const {
    return absl::StrFormat("https:
                           aws_region);
  }
};
struct S3PathFormatter {
  std::string GetEndpoint(std::string_view bucket,
                          std::string_view aws_region) const {
    return absl::StrFormat("https:
                           bucket);
  }
};
struct S3CustomFormatter {
  std::string endpoint;
  std::string GetEndpoint(std::string_view bucket,
                          std::string_view aws_region) const {
    return absl::StrFormat("%s/%s", endpoint, bucket);
  }
};
template <typename Formatter>
struct ResolveHost {
  std::string bucket;
  std::string default_aws_region;
  Formatter formatter;
  void operator()(Promise<S3EndpointRegion> promise,
                  ReadyFuture<HttpResponse> ready) {
    if (!promise.result_needed()) return;
    auto& headers = ready.value().headers;
    if (auto it = headers.find(kAmzBucketRegionHeader); it != headers.end()) {
      promise.SetResult(S3EndpointRegion{
          formatter.GetEndpoint(bucket, it->second),
          it->second,
      });
    }
    if (!default_aws_region.empty()) {
      promise.SetResult(S3EndpointRegion{
          formatter.GetEndpoint(bucket, default_aws_region),
          default_aws_region,
      });
    }
    promise.SetResult(absl::FailedPreconditionError(tensorstore::StrCat(
        "Failed to resolve aws_region for bucket ", QuoteString(bucket))));
  }
};
}  
std::variant<absl::Status, S3EndpointRegion> ValidateEndpoint(
    std::string_view bucket, std::string aws_region, std::string_view endpoint,
    std::string host_header) {
  ABSL_CHECK(!bucket.empty());
  if (!host_header.empty() && endpoint.empty()) {
    return absl::InvalidArgumentError(
        "\"host_header\" cannot be set without also setting \"endpoint\"");
  }
  if (internal_kvstore_s3::ClassifyBucketName(bucket) ==
      internal_kvstore_s3::BucketNameType::kOldUSEast1) {
    if (!aws_region.empty() && aws_region != "us-east-1") {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Bucket ", QuoteString(bucket),
          " requires aws_region \"us-east-1\", not ", QuoteString(aws_region)));
    }
    aws_region = "us-east-1";
  }
  if (endpoint.empty()) {
    if (!aws_region.empty()) {
      if (!absl::StrContains(bucket, ".")) {
        S3VirtualHostFormatter formatter;
        return S3EndpointRegion{
            formatter.GetEndpoint(bucket, aws_region),
            aws_region,
        };
      }
      S3PathFormatter formatter;
      return S3EndpointRegion{
          formatter.GetEndpoint(bucket, aws_region),
          aws_region,
      };
    }
    return absl::OkStatus();
  }
  auto parsed = internal::ParseGenericUri(endpoint);
  if (parsed.scheme != "http" && parsed.scheme != "https") {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Endpoint ", endpoint, " has invalid scheme ",
                            parsed.scheme, ". Should be http(s)."));
  }
  if (!parsed.query.empty()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Query in endpoint unsupported ", endpoint));
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Fragment in endpoint unsupported ", endpoint));
  }
  if (!aws_region.empty()) {
    S3CustomFormatter formatter{std::string(endpoint)};
    return S3EndpointRegion{
        formatter.GetEndpoint(bucket, aws_region),
        aws_region,
    };
  }
  return absl::OkStatus();
}
Future<S3EndpointRegion> ResolveEndpointRegion(
    std::string bucket, std::string_view endpoint, std::string host_header,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  assert(!bucket.empty());
  assert(transport);
  assert(IsValidBucketName(bucket));
  if (endpoint.empty()) {
    if (!absl::StrContains(bucket, ".")) {
      std::string url = absl::StrFormat("https:
      return PromiseFuturePair<S3EndpointRegion>::Link(
                 ResolveHost<S3VirtualHostFormatter>{
                     std::move(bucket), {}, S3VirtualHostFormatter{}},
                 transport->IssueRequest(
                     HttpRequestBuilder("HEAD", std::move(url))
                         .AddHostHeader(host_header)
                         .BuildRequest(),
                     {}))
          .future;
    }
    std::string url =
        absl::StrFormat("https:
    return PromiseFuturePair<S3EndpointRegion>::Link(
               ResolveHost<S3PathFormatter>{
                   std::move(bucket), {}, S3PathFormatter{}},
               transport->IssueRequest(
                   HttpRequestBuilder("HEAD", std ::move(url))
                       .AddHostHeader(host_header)
                       .BuildRequest(),
                   {}))
        .future;
  }
  std::string url = absl::StrFormat("%s/%s", endpoint, bucket);
  return PromiseFuturePair<S3EndpointRegion>::Link(
             ResolveHost<S3CustomFormatter>{
                 std::move(bucket), "us-east-1",
                 S3CustomFormatter{std::string(endpoint)}},
             transport->IssueRequest(HttpRequestBuilder("HEAD", std::move(url))
                                         .AddHostHeader(host_header)
                                         .BuildRequest(),
                                     {}))
      .future;
}
}  
}  