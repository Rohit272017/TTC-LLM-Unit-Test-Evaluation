#include "arolla/codegen/qtype_utils.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "arolla/qtype/qtype.h"
namespace arolla::codegen {
std::vector<std::pair<std::string, QTypePtr>>
NamedQTypeVectorBuilder::Build() && {
  return std::move(types_);
}
void NamedQTypeVectorBuilder::AddFromCommonPrefixWithPrevious(
    size_t length, const char* suffix, QTypePtr qtype) {
  std::string suffix_str(suffix);
  CHECK_LE(suffix_str.size(), length);
  size_t prefix_length = length - suffix_str.size();
  absl::string_view previous_name =
      types_.empty() ? "" : absl::string_view(types_.back().first);
  CHECK_LE(prefix_length, previous_name.size());
  types_.emplace_back(
      absl::StrCat(previous_name.substr(0, prefix_length), suffix_str), qtype);
}
}  