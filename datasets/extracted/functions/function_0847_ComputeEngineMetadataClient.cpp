#include "tsl/platform/cloud/compute_engine_metadata_client.h"
#include <cstdlib>
#include <utility>
#include "absl/strings/str_cat.h"
#include "tsl/platform/cloud/curl_http_request.h"
namespace tsl {
namespace {
constexpr char kGceMetadataHost[] = "GCE_METADATA_HOST";
constexpr char kGceMetadataBaseUrl[] =
    "http:
}  
ComputeEngineMetadataClient::ComputeEngineMetadataClient(
    std::shared_ptr<HttpRequest::Factory> http_request_factory,
    const RetryConfig& config)
    : http_request_factory_(std::move(http_request_factory)),
      retry_config_(config) {}
absl::Status ComputeEngineMetadataClient::GetMetadata(
    const string& path, std::vector<char>* response_buffer) {
  const auto get_metadata_from_gce = [path, response_buffer, this]() {
    string metadata_url;
    const char* metadata_url_override = std::getenv(kGceMetadataHost);
    if (metadata_url_override) {
      metadata_url = absl::StrCat("http:
                                  "/computeMetadata/v1/");
    } else {
      metadata_url = kGceMetadataBaseUrl;
    }
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    request->SetUri(metadata_url + path);
    request->AddHeader("Metadata-Flavor", "Google");
    request->SetResultBuffer(response_buffer);
    TF_RETURN_IF_ERROR(request->Send());
    return absl::OkStatus();
  };
  return RetryingUtils::CallWithRetries(get_metadata_from_gce, retry_config_);
}
}  