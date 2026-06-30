#ifndef QUICHE_COMMON_PLATFORM_API_QUICHE_URL_UTILS_H_
#define QUICHE_COMMON_PLATFORM_API_QUICHE_URL_UTILS_H_
#include <optional>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "quiche_platform_impl/quiche_url_utils_impl.h"
namespace quiche {
inline bool ExpandURITemplate(
    const std::string& uri_template,
    const absl::flat_hash_map<std::string, std::string>& parameters,
    std::string* target,
    absl::flat_hash_set<std::string>* vars_found = nullptr) {
  return ExpandURITemplateImpl(uri_template, parameters, target, vars_found);
}
inline std::optional<std::string> AsciiUrlDecode(absl::string_view input) {
  return AsciiUrlDecodeImpl(input);
}
}  
#endif  