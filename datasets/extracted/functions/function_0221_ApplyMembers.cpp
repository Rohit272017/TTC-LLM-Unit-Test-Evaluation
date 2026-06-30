#ifndef TENSORSTORE_INTERNAL_CACHE_KEY_CACHE_KEY_H_
#define TENSORSTORE_INTERNAL_CACHE_KEY_CACHE_KEY_H_
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include "absl/base/attributes.h"
#include "tensorstore/internal/cache_key/fwd.h"
#include "tensorstore/util/apply_members/apply_members.h"
namespace tensorstore {
namespace internal {
template <typename T>
struct CacheKeyExcludes {
  T value;
  template <typename X, typename F>
  static constexpr auto ApplyMembers(X&& x, F f) {
    return f(x.value);
  }
};
template <typename T>
CacheKeyExcludes(T&& x) -> CacheKeyExcludes<T>;
template <typename... U>
void EncodeCacheKey(std::string* out, const U&... u);
inline void EncodeCacheKeyAdl() {}
template <typename T, typename SFINAE>
struct CacheKeyEncoder {
  static void Encode(std::string* out, const T& value) {
    EncodeCacheKeyAdl(out, value);
  }
};
template <typename T>
struct CacheKeyEncoder<T, std::enable_if_t<SerializeUsingMemcpy<T>>> {
  static void Encode(std::string* out, T value) {
    out->append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
};
template <>
struct CacheKeyEncoder<std::string_view> {
  static void Encode(std::string* out, std::string_view k) {
    EncodeCacheKey(out, k.size());
    out->append(k.data(), k.size());
  }
};
template <>
struct CacheKeyEncoder<std::string> : public CacheKeyEncoder<std::string_view> {
};
template <>
struct CacheKeyEncoder<std::type_info> {
  static void Encode(std::string* out, const std::type_info& t) {
    EncodeCacheKey(out, std::string_view(t.name()));
  }
};
template <typename T>
struct CacheKeyEncoder<T*> {
  static void Encode(std::string* out, T* value) {
    out->append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
};
template <typename T>
struct CacheKeyEncoder<CacheKeyExcludes<T>> {
  static void Encode(std::string* out, const CacheKeyExcludes<T>& v) {
  }
};
template <typename T>
constexpr inline bool IsCacheKeyExcludes = false;
template <typename T>
constexpr inline bool IsCacheKeyExcludes<CacheKeyExcludes<T>> = true;
template <typename T>
struct CacheKeyEncoder<
    T, std::enable_if_t<SupportsApplyMembers<T> && !IsCacheKeyExcludes<T> &&
                        !SerializeUsingMemcpy<T>>> {
  static void Encode(std::string* out, const T& v) {
    ApplyMembers<T>::Apply(
        v, [&out](auto&&... x) { (internal::EncodeCacheKey(out, x), ...); });
  }
};
template <typename... U>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void EncodeCacheKey(std::string* out,
                                                        const U&... u) {
  (CacheKeyEncoder<U>::Encode(out, u), ...);
}
}  
}  
#endif  