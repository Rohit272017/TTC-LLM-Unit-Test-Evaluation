#include "common/any.h"
#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
namespace cel {
bool ParseTypeUrl(absl::string_view type_url,
                  absl::Nullable<absl::string_view*> prefix,
                  absl::Nullable<absl::string_view*> type_name) {
  auto pos = type_url.find_last_of('/');
  if (pos == absl::string_view::npos || pos + 1 == type_url.size()) {
    return false;
  }
  if (prefix) {
    *prefix = type_url.substr(0, pos + 1);
  }
  if (type_name) {
    *type_name = type_url.substr(pos + 1);
  }
  return true;
}
}  