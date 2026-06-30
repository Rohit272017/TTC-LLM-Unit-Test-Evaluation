#include "tensorstore/kvstore/gcs/validate.h"
#include <iterator>
#include <string>
#include <string_view>
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/status.h"
namespace tensorstore {
namespace internal_storage_gcs {
bool IsValidBucketName(std::string_view bucket) {
  if (bucket.size() < 3 || bucket.size() > 222) return false;
  if (!absl::ascii_isdigit(*bucket.begin()) &&
      !absl::ascii_islower(*bucket.begin())) {
    return false;
  }
  if (!absl::ascii_isdigit(*bucket.rbegin()) &&
      !absl::ascii_islower(*bucket.rbegin())) {
    return false;
  }
  for (std::string_view v : absl::StrSplit(bucket, absl::ByChar('.'))) {
    if (v.empty()) return false;
    if (v.size() > 63) return false;
    if (*v.begin() == '-') return false;
    if (*v.rbegin() == '-') return false;
    for (const auto ch : v) {
      if (ch != '-' && ch != '_' && !absl::ascii_isdigit(ch) &&
          !absl::ascii_islower(ch)) {
        return false;
      }
    }
  }
  return true;
}
bool IsValidObjectName(std::string_view name) {
  if (name.empty() || name.size() > 1024) return false;
  if (name == "." || name == "..") return false;
  if (absl::StartsWith(name, ".well-known/acme-challenge")) return false;
  for (const auto ch : name) {
    if (ch == '\r' || ch == '\n') return false;
    if (absl::ascii_iscntrl(ch)) return false;
  }
  return internal::IsValidUtf8(name);
}
bool IsValidStorageGeneration(const StorageGeneration& gen) {
  return StorageGeneration::IsUnknown(gen) ||
         StorageGeneration::IsNoValue(gen) ||
         (StorageGeneration::IsUint64(gen) &&
          StorageGeneration::ToUint64(gen) > 0);
}
absl::Status GcsHttpResponseToStatus(
    const internal_http::HttpResponse& response, bool& retryable,
    SourceLocation loc) {
  auto absl_status_code = HttpResponseCodeToStatusCode(response);
  if (absl_status_code == absl::StatusCode::kOk) {
    return absl::OkStatus();
  }
  retryable = (response.status_code == 429 ||  
               response.status_code == 408 ||  
               response.status_code >= 500     
  );
  std::string error_message;
  auto payload = response.payload;
  auto payload_str = payload.Flatten();
  if (auto j_obj = internal::ParseJson(payload_str); j_obj.is_object()) {
    if (auto j_error = internal_json::JsonExtractMember(
            j_obj.template get_ptr<::nlohmann::json::object_t*>(), "error");
        j_error.is_object()) {
      if (auto j_message = internal_json::JsonExtractMember(
              j_error.template get_ptr<::nlohmann::json::object_t*>(),
              "message");
          j_message.is_string()) {
        error_message = j_message.template get<std::string>();
      }
    }
  }
  if (error_message.empty()) {
    error_message = HttpResponseCodeToMessage(response);
    if (error_message.empty()) {
      error_message = "Unknown";
    }
  }
  absl::Status status(absl_status_code, error_message);
  status.SetPayload("http_response_code",
                    absl::Cord(absl::StrFormat("%d", response.status_code)));
  if (!payload_str.empty()) {
    status.SetPayload(
        "http_response_body",
        payload.Subcord(0,
                        payload_str.size() < 256 ? payload_str.size() : 256));
  }
  if (auto id_header = response.headers.find("x-guploader-uploadid");
      id_header != response.headers.end()) {
    status.SetPayload("x-guploader-uploadid", absl::Cord(id_header->second));
  }
  MaybeAddSourceLocation(status, loc);
  return status;
}
}  
}  