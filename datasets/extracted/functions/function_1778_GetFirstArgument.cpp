#ifndef TENSORSTORE_INTERNAL_META_H_
#define TENSORSTORE_INTERNAL_META_H_
namespace tensorstore {
namespace internal {
template <typename T, typename... Ts>
constexpr T&& GetFirstArgument(T&& t, Ts&&... ts) {
  return static_cast<T&&>(t);
}
inline int constexpr_assert_failed() noexcept { return 0; }
#define TENSORSTORE_CONSTEXPR_ASSERT(...) \
  (static_cast<void>(                     \
      (__VA_ARGS__) ? 0                   \
                    : tensorstore::internal::constexpr_assert_failed())) 
}  
}  
#endif  