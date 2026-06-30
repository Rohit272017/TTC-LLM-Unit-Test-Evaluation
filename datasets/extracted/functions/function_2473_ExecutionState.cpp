#include "xla/ffi/execution_state.h"
#include <utility>
#include "absl/status/status.h"
#include "xla/ffi/type_id_registry.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
namespace xla::ffi {
ExecutionState::ExecutionState()
    : type_id_(TypeIdRegistry::kUnknownTypeId),
      state_(nullptr),
      deleter_(nullptr) {}
ExecutionState::~ExecutionState() {
  if (deleter_) deleter_(state_);
}
absl::Status ExecutionState::Set(TypeId type_id, void* state,
                                 Deleter<void> deleter) {
  DCHECK(state && deleter) << "State and deleter must not be null";
  if (type_id_ != TypeIdRegistry::kUnknownTypeId) {
    return FailedPrecondition("State is already set with a type id %d",
                              type_id_.value());
  }
  type_id_ = type_id;
  state_ = state;
  deleter_ = std::move(deleter);
  return absl::OkStatus();
}
absl::StatusOr<void*> ExecutionState::Get(TypeId type_id) const {
  if (type_id_ == TypeIdRegistry::kUnknownTypeId) {
    return NotFound("State is not set");
  }
  if (type_id_ != type_id) {
    return InvalidArgument(
        "Set state type id %d does not match the requested one %d",
        type_id_.value(), type_id.value());
  }
  return state_;
}
bool ExecutionState::IsSet() const {
  return type_id_ != TypeIdRegistry::kUnknownTypeId;
}
}  