#ifndef TENSORSTORE_SERIALIZATION_SPAN_H_
#define TENSORSTORE_SERIALIZATION_SPAN_H_
#include <cstddef>
#include <type_traits>
#include "absl/base/attributes.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace serialization {
template <typename T, ptrdiff_t N,
          typename ElementSerializer = Serializer<std::remove_cv_t<T>>>
struct SpanSerializer {
  [[nodiscard]] bool Encode(EncodeSink& sink, span<const T, N> value) const {
    for (const auto& element : value) {
      if (!element_serializer.Encode(sink, element)) return false;
    }
    return true;
  }
  [[nodiscard]] bool Decode(DecodeSource& source, span<T, N> value) const {
    for (auto& element : value) {
      if (!element_serializer.Decode(source, element)) return false;
    }
    return true;
  }
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS ElementSerializer element_serializer = {};
  constexpr static bool non_serializable() {
    return IsNonSerializer<ElementSerializer>;
  }
};
template <typename T, ptrdiff_t N>
struct Serializer<span<T, N>> : public SpanSerializer<T, N> {};
}  
}  
#endif  