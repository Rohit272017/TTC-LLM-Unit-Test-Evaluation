#include "xla/tsl/concurrency/async_value_ref.h"
#include <string_view>
#include <utility>
#include "absl/status/status.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/logging.h"
namespace tsl {
RCReference<IndirectAsyncValue> MakeIndirectAsyncValue() {
  return TakeRef(internal::AllocateAndConstruct<IndirectAsyncValue>());
}
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::Status status) {
  CHECK(!status.ok()) << "status must be an error";  
  return TakeRef(
      internal::AllocateAndConstruct<ErrorAsyncValue>(std::move(status)));
}
RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(std::string_view message) {
  return MakeErrorAsyncValueRef(absl::InternalError(message));
}
}  