#ifndef AROLLA_UTIL_OPERATOR_NAME_H_
#define AROLLA_UTIL_OPERATOR_NAME_H_
#include "absl/strings/string_view.h"
#include "arolla/util/string.h"
namespace arolla {
constexpr bool IsOperatorName(absl::string_view name) {
  return IsQualifiedIdentifier(name);
}
#define AROLLA_OPERATOR_NAME(name)                    \
  (                                                   \
      []() {                                          \
        static_assert(::arolla::IsOperatorName(name), \
                      "Invalid operator name.");      \
      }(),                                            \
      name)
}  
#endif  