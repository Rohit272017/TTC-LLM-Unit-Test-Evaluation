#include "eval/public/cel_number.h"
#include "eval/public/cel_value.h"
namespace google::api::expr::runtime {
absl::optional<CelNumber> GetNumberFromCelValue(const CelValue& value) {
  if (int64_t val; value.GetValue(&val)) {
    return CelNumber(val);
  } else if (uint64_t val; value.GetValue(&val)) {
    return CelNumber(val);
  } else if (double val; value.GetValue(&val)) {
    return CelNumber(val);
  }
  return absl::nullopt;
}
}  