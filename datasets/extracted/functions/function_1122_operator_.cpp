#include <libaddressinput/address_problem.h>
#include <cstddef>
#include <ostream>
#include "util/size.h"
using i18n::addressinput::AddressProblem;
using i18n::addressinput::size;
using i18n::addressinput::UNEXPECTED_FIELD;
using i18n::addressinput::UNSUPPORTED_FIELD;
std::ostream& operator<<(std::ostream& o, AddressProblem problem) {
  static const char* const kProblemNames[] = {
      "UNEXPECTED_FIELD",  "MISSING_REQUIRED_FIELD", "UNKNOWN_VALUE",
      "INVALID_FORMAT",    "MISMATCHING_VALUE",      "USES_P_O_BOX",
      "UNSUPPORTED_FIELD",
  };
  static_assert(UNEXPECTED_FIELD == 0, "bad_base");
  static_assert(UNSUPPORTED_FIELD == size(kProblemNames) - 1, "bad_length");
  if (problem < 0 || static_cast<size_t>(problem) >= size(kProblemNames)) {
    o << "[INVALID ENUM VALUE " << static_cast<int>(problem) << "]";
  } else {
    o << kProblemNames[problem];
  }
  return o;
}