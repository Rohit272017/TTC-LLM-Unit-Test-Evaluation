#include "tensorflow/compiler/jit/device_executable_persistor.h"
#include <string>
#include "absl/strings/str_cat.h"
namespace tensorflow {
std::string XlaSerializedCacheKeyToFileName(const XlaSerializedCacheKey& key) {
  static constexpr char kXlaSerializedCacheKeySeparator[] = "__";
  return absl::StrCat(
      key.prefix(), key.prefix().empty() ? "" : kXlaSerializedCacheKeySeparator,
      key.signature_fingerprint(), kXlaSerializedCacheKeySeparator,
      key.cluster_fingerprint(), kXlaSerializedCacheKeySeparator,
      key.device_type(),
      key.compiled_using_pjrt()
          ? absl::StrCat(kXlaSerializedCacheKeySeparator, "pjrt")
          : "",
      ".pb");
}
}  