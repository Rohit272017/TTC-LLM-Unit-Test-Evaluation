#include "arolla/qtype/simple_qtype.h"
#include <cstdint>
#include <optional>
#include <string>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
namespace arolla {
absl::Status SimpleQType::InitNameMap() {
  name2index_.reserve(field_names_.size());
  for (const auto& field_name : field_names_) {
    if (bool inserted =
            name2index_.emplace(field_name, name2index_.size()).second;
        !inserted) {
      return absl::FailedPreconditionError(absl::StrCat(
          "duplicated name field for QType ", name(), ": ", field_name));
    }
  }
  return absl::OkStatus();
}
absl::Span<const std::string> SimpleQType::GetFieldNames() const {
  return field_names_;
}
std::optional<int64_t> SimpleQType::GetFieldIndexByName(
    absl::string_view field_name) const {
  if (auto it = name2index_.find(field_name); it != name2index_.end()) {
    return it->second;
  }
  return std::nullopt;
}
}  