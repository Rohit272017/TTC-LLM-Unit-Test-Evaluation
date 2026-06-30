#include "arolla/memory/optional_value.h"
#include <cstdint>
#include "absl/strings/str_cat.h"
#include "arolla/util/bytes.h"
#include "arolla/util/repr.h"
#include "arolla/util/text.h"
namespace arolla {
ReprToken ReprTraits<OptionalValue<bool>>::operator()(
    const OptionalValue<bool>& value) const {
  return ReprToken{
      value.present ? absl::StrCat("optional_boolean{", Repr(value.value), "}")
                    : "optional_boolean{NA}"};
}
ReprToken ReprTraits<OptionalValue<int32_t>>::operator()(
    const OptionalValue<int32_t>& value) const {
  return ReprToken{value.present
                       ? absl::StrCat("optional_int32{", Repr(value.value), "}")
                       : "optional_int32{NA}"};
}
ReprToken ReprTraits<OptionalValue<int64_t>>::operator()(
    const OptionalValue<int64_t>& value) const {
  return ReprToken{value.present ? absl::StrCat("optional_", Repr(value.value))
                                 : "optional_int64{NA}"};
}
ReprToken ReprTraits<OptionalValue<uint64_t>>::operator()(
    const OptionalValue<uint64_t>& value) const {
  return ReprToken{value.present ? absl::StrCat("optional_", Repr(value.value))
                                 : "optional_uint64{NA}"};
}
ReprToken ReprTraits<OptionalValue<float>>::operator()(
    const OptionalValue<float>& value) const {
  return ReprToken{
      value.present ? absl::StrCat("optional_float32{", Repr(value.value), "}")
                    : "optional_float32{NA}"};
}
ReprToken ReprTraits<OptionalValue<double>>::operator()(
    const OptionalValue<double>& value) const {
  return ReprToken{value.present ? absl::StrCat("optional_", Repr(value.value))
                                 : "optional_float64{NA}"};
}
ReprToken ReprTraits<OptionalValue<Bytes>>::operator()(
    const OptionalValue<Bytes>& value) const {
  return ReprToken{value.present
                       ? absl::StrCat("optional_bytes{", Repr(value.value), "}")
                       : "optional_bytes{NA}"};
}
ReprToken ReprTraits<OptionalValue<Text>>::operator()(
    const OptionalValue<Text>& value) const {
  return ReprToken{value.present
                       ? absl::StrCat("optional_text{", Repr(value.value), "}")
                       : "optional_text{NA}"};
}
ReprToken ReprTraits<OptionalUnit>::operator()(
    const OptionalUnit& value) const {
  return ReprToken{value.present ? "present" : "missing"};
}
}  