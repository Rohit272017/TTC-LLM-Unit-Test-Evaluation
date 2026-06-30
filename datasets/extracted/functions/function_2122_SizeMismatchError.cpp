#include "absl/status/status.h"
#include <cstdint>
#include <initializer_list>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
namespace arolla {
absl::Status SizeMismatchError(std::initializer_list<int64_t> sizes) {
  return absl::InvalidArgumentError(absl::StrCat(
      "argument sizes mismatch: (", absl::StrJoin(sizes, ", "), ")"));
}
}  