#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/python/ifrt/future.h"
#include "tensorflow/core/framework/tensor.h"
namespace tensorflow {
namespace ifrt_serving {
absl::Status IfrtRestoreTensorRegistry::TryRegister(
    absl::string_view name, RestoredTensorInfo restored_tensor_info) {
  absl::MutexLock lock(&mutex_);
  auto& info = restored_tensors_[name];
  if (info.tensor_future.IsValid()) {
    return absl::AlreadyExistsError(
        absl::StrCat("Variable '", name, "' already registered."));
  }
  info = std::move(restored_tensor_info);
  return absl::OkStatus();
}
xla::ifrt::Future<tensorflow::Tensor>
IfrtRestoreTensorRegistry::GetRestoredTensor(absl::string_view name) const {
  absl::MutexLock lock(&mutex_);
  auto it = restored_tensors_.find(name);
  if (it == restored_tensors_.end()) {
    return xla::ifrt::Future<tensorflow::Tensor>(
        absl::NotFoundError(absl::StrCat("Variable '", name, "' not found.")));
  }
  return it->second.tensor_future;
}
absl::Status IfrtRestoreTensorRegistry::SetUsedByHost(absl::string_view name) {
  absl::MutexLock lock(&mutex_);
  auto it = restored_tensors_.find(name);
  if (it == restored_tensors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Variable '", name, "' not found."));
  }
  it->second.used_by_host = true;
  return absl::OkStatus();
}
void IfrtRestoreTensorRegistry::Freeze() {
  absl::MutexLock lock(&mutex_);
  xla::ifrt::Future<tensorflow::Tensor> release_tensor_future(
      absl::UnavailableError("Tensor is already release."));
  for (auto& [name, info] : restored_tensors_) {
    if (!info.used_by_host) {
      info.tensor_future = release_tensor_future;
    }
  }
}
absl::StatusOr<DtypeAndShape> IfrtRestoreTensorRegistry::GetDtypeAndShape(
    absl::string_view name) const {
  absl::MutexLock lock(&mutex_);
  auto it = restored_tensors_.find(name);
  if (it == restored_tensors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Variable '", name, "' not found."));
  }
  return it->second.dtype_and_shape;
}
}  
}  