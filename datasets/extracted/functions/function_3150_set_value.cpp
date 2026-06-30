#ifndef TENSORSTORE_UTIL_EXECUTION_ANY_RECEIVER_H_
#define TENSORSTORE_UTIL_EXECUTION_ANY_RECEIVER_H_
#include <utility>
#include "absl/base/attributes.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
namespace tensorstore {
using AnyCancelReceiver = poly::Poly<0, false, void()>;
namespace internal_sender {
template <typename E, typename... V>
using ReceiverPoly = poly::Poly<sizeof(void*) * 2, false,
                                void(internal_execution::set_value_t, V...),
                                void(internal_execution::set_error_t, E),
                                void(internal_execution::set_cancel_t)>;
template <typename E, typename... V>
using FlowReceiverPoly =
    poly::Poly<sizeof(void*) * 2, false,
               void(internal_execution::set_starting_t, AnyCancelReceiver up),
               void(internal_execution::set_value_t, V...),
               void(internal_execution::set_done_t),
               void(internal_execution::set_error_t, E),
               void(internal_execution::set_stopping_t)>;
}  
template <typename E, typename... V>
class AnyReceiver : public internal_sender::ReceiverPoly<E, V...> {
  using Base = internal_sender::ReceiverPoly<E, V...>;
 public:
  using Base::Base;
  AnyReceiver() : Base(NullReceiver{}) {}
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_value(V... v) {
    (*this)(internal_execution::set_value_t{}, std::forward<V>(v)...);
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_error(E e) {
    (*this)(internal_execution::set_error_t{}, std::forward<E>(e));
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_cancel() {
    (*this)(internal_execution::set_cancel_t{});
  }
};
template <typename E, typename... V>
class AnyFlowReceiver : public internal_sender::FlowReceiverPoly<E, V...> {
  using Base = internal_sender::FlowReceiverPoly<E, V...>;
 public:
  using Base::Base;
  AnyFlowReceiver() : Base(NullReceiver{}) {}
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_starting(AnyCancelReceiver cancel) {
    (*this)(internal_execution::set_starting_t{}, std::move(cancel));
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_value(V... v) {
    (*this)(internal_execution::set_value_t{}, std::forward<V>(v)...);
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_done() {
    (*this)(internal_execution::set_done_t{});
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_error(E e) {
    (*this)(internal_execution::set_error_t{}, std::forward<E>(e));
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_stopping() {
    (*this)(internal_execution::set_stopping_t{});
  }
};
}  
#endif  