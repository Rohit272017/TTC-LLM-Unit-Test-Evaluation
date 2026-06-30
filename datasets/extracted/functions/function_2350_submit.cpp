#ifndef TENSORSTORE_UTIL_EXECUTION_ANY_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_ANY_SENDER_H_
#include <utility>
#include "absl/base/attributes.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
namespace tensorstore {
namespace internal_sender {
template <typename E, typename... V>
using SenderPoly =
    poly::Poly<(sizeof(V) + ... + 0), false,
               void(internal_execution::submit_t, AnyReceiver<E, V...>)>;
template <typename E, typename... V>
using FlowSenderPoly =
    poly::Poly<(sizeof(V) + ... + 0), false,
               void(internal_execution::submit_t, AnyFlowReceiver<E, V...>)>;
}  
template <typename E, typename... V>
class AnySender : public internal_sender::SenderPoly<E, V...> {
  using Base = internal_sender::SenderPoly<E, V...>;
 public:
  using Base::Base;
  AnySender() : Base(NullSender{}) {}
  ABSL_ATTRIBUTE_ALWAYS_INLINE void submit(AnyReceiver<E, V...> receiver) {
    (*this)(internal_execution::submit_t{}, std::move(receiver));
  }
};
template <typename E, typename... V>
class AnyFlowSender : public internal_sender::FlowSenderPoly<E, V...> {
  using Base = internal_sender::FlowSenderPoly<E, V...>;
 public:
  using Base::Base;
  AnyFlowSender() : Base(NullSender{}) {}
  ABSL_ATTRIBUTE_ALWAYS_INLINE void submit(AnyFlowReceiver<E, V...> receiver) {
    (*this)(internal_execution::submit_t{}, std::move(receiver));
  }
};
}  
#endif  