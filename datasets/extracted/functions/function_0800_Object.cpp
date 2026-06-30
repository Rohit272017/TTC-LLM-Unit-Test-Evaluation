#include "tensorstore/kvstore/gcs_http/object_metadata.h"
#include <stdint.h>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_kvstore_gcs_http {
using ::tensorstore::internal_http::TryParseIntHeader;
using ::tensorstore::internal_json_binding::DefaultInitializedValue;
namespace jb = tensorstore::internal_json_binding;
inline constexpr auto ObjectMetadataBinder = jb::Object(
    jb::Member("name", jb::Projection(&ObjectMetadata::name)),
    jb::Member("md5Hash", jb::Projection(&ObjectMetadata::md5_hash,
                                         DefaultInitializedValue())),
    jb::Member("crc32c", jb::Projection(&ObjectMetadata::crc32c,
                                        DefaultInitializedValue())),
    jb::Member("size", jb::Projection(&ObjectMetadata::size,
                                      jb::DefaultInitializedValue(
                                          jb::LooseValueAsBinder))),
    jb::Member("generation", jb::Projection(&ObjectMetadata::generation,
                                            jb::DefaultInitializedValue(
                                                jb::LooseValueAsBinder))),
    jb::Member("metageneration", jb::Projection(&ObjectMetadata::metageneration,
                                                jb::DefaultInitializedValue(
                                                    jb::LooseValueAsBinder))),
    jb::Member("timeCreated", jb::Projection(&ObjectMetadata::time_created,
                                             jb::DefaultValue([](auto* x) {
                                               *x = absl::InfinitePast();
                                             }))),
    jb::Member("updated", jb::Projection(&ObjectMetadata::updated,
                                         jb::DefaultValue([](auto* x) {
                                           *x = absl::InfinitePast();
                                         }))),
    jb::Member("timeDeleted", jb::Projection(&ObjectMetadata::time_deleted,
                                             jb::DefaultValue([](auto* x) {
                                               *x = absl::InfinitePast();
                                             }))),
    jb::DiscardExtraMembers);
TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ObjectMetadata,
                                       [](auto is_loading, const auto& options,
                                          auto* obj, ::nlohmann::json* j) {
                                         return ObjectMetadataBinder(
                                             is_loading, options, obj, j);
                                       })
void SetObjectMetadataFromHeaders(
    const absl::btree_multimap<std::string, std::string>& headers,
    ObjectMetadata* result) {
  result->size =
      TryParseIntHeader<uint64_t>(headers, "content-length").value_or(0);
  result->generation =
      TryParseIntHeader<int64_t>(headers, "x-goog-generation").value_or(0);
  result->metageneration =
      TryParseIntHeader<uint64_t>(headers, "x-goog-metageneration").value_or(0);
  auto it = headers.find("x-goog-hash");
  if (it != headers.end()) {
    for (std::string_view kv : absl::StrSplit(it->second, absl::ByChar(','))) {
      std::pair<std::string_view, std::string_view> split =
          absl::StrSplit(kv, absl::MaxSplits('=', 1));
      if (split.first == "crc32c") {
        result->crc32c = std::string(split.second);
      } else if (split.first == "md5") {
        result->md5_hash = std::string(split.second);
      }
    }
  }
}
Result<ObjectMetadata> ParseObjectMetadata(std::string_view source) {
  auto json = internal::ParseJson(source);
  if (json.is_discarded()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Failed to parse object metadata: ", source));
  }
  return jb::FromJson<ObjectMetadata>(std::move(json));
}
}  
}  