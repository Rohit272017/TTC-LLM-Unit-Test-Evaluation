#include "internal/names.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "internal/lexis.h"
namespace cel::internal {
bool IsValidRelativeName(absl::string_view name) {
  if (name.empty()) {
    return false;
  }
  for (const auto& id : absl::StrSplit(name, '.')) {
    if (!LexisIsIdentifier(id)) {
      return false;
    }
  }
  return true;
}
}  