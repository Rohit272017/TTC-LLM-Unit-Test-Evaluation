#include "xla/ffi/execution_context.h"
#include <memory>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
namespace xla::ffi {
ExecutionContext::UserData::UserData(void* data, Deleter<void> deleter)
    : data_(data), deleter_(std::move(deleter)) {}
ExecutionContext::UserData::~UserData() {
  if (deleter_) deleter_(data_);
}
absl::Status ExecutionContext::Insert(TypeId type_id, void* data,
                                      Deleter<void> deleter) {
  return InsertUserData(type_id,
                        std::make_unique<UserData>(data, std::move(deleter)));
}
absl::Status ExecutionContext::InsertUserData(TypeId type_id,
                                              std::unique_ptr<UserData> data) {
  if (!data) return absl::InvalidArgumentError("User data must be not null");
  auto emplaced = user_data_.emplace(type_id, std::move(data));
  if (!emplaced.second) {
    return absl::AlreadyExistsError(
        absl::StrCat("User data with type id ", type_id.value(),
                     " already exists in execution context"));
  }
  return absl::OkStatus();
}
absl::StatusOr<ExecutionContext::UserData*> ExecutionContext::LookupUserData(
    TypeId type_id) const {
  auto it = user_data_.find(type_id);
  if (it == user_data_.end()) {
    return absl::NotFoundError(absl::StrCat("User data with type id ",
                                            type_id.value(),
                                            " not found in execution context"));
  }
  return it->second.get();
}
}  