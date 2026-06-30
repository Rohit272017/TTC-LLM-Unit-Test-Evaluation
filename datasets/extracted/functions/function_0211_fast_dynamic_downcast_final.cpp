#ifndef AROLLA_UTIL_FAST_DYNAMIC_DOWNCAST_FINAL_H_
#define AROLLA_UTIL_FAST_DYNAMIC_DOWNCAST_FINAL_H_
#include <type_traits>
#include <typeinfo>
#include "absl/base/attributes.h"
namespace arolla {
template <typename T, typename S>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline T fast_dynamic_downcast_final(S* src) {
  static_assert(std::is_pointer_v<T>);
  static_assert(
      !std::is_const_v<S> || std::is_const_v<std::remove_pointer_t<T>>,
      "cannot cast 'const' to 'non-const'");
  using DecayedT = std::decay_t<std::remove_pointer_t<T>>;
  using DecayedS = std::decay_t<S>;
  static_assert(!std::is_same_v<DecayedS, DecayedT>);
  static_assert(std::is_base_of_v<DecayedS, DecayedT>);
  static_assert(  
      std::is_convertible_v<DecayedT*, DecayedS*>);
  static_assert(std::is_polymorphic_v<DecayedS>);
  static_assert(std::is_final_v<DecayedT>);
  if (src != nullptr && typeid(*src) == typeid(DecayedT)) {
    return static_cast<T>(src);
  }
  return nullptr;
}
}  
#endif  