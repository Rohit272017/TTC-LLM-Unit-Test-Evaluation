#ifndef QUICHE_COMMON_QUICHE_CALLBACKS_H_
#define QUICHE_COMMON_QUICHE_CALLBACKS_H_
#include <type_traits>
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "quiche/common/platform/api/quiche_export.h"
namespace quiche {
namespace callbacks_internal {
template <class Sig>
class QUICHE_EXPORT SignatureChanger {};
template <typename ReturnType, typename... Args>
class QUICHE_NO_EXPORT SignatureChanger<ReturnType(Args...)> {
 public:
  using Rvalue = ReturnType(Args...) &&;
  using Const = ReturnType(Args...) const;
};
}  
template <class T>
using UnretainedCallback = absl::FunctionRef<T>;
template <class T>
using SingleUseCallback = absl::AnyInvocable<
    typename callbacks_internal::SignatureChanger<T>::Rvalue>;
static_assert(std::is_same_v<SingleUseCallback<void(int, int &, int &&)>,
                             absl::AnyInvocable<void(int, int &, int &&) &&>>);
template <class T>
using MultiUseCallback =
    absl::AnyInvocable<typename callbacks_internal::SignatureChanger<T>::Const>;
static_assert(
    std::is_same_v<MultiUseCallback<void()>, absl::AnyInvocable<void() const>>);
}  
#endif  