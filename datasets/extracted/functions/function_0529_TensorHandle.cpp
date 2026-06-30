#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSORHANDLE_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_TENSORHANDLE_H_
#include <memory>
#include <vector>
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/cc/experimental/base/public/tensor.h"
namespace tensorflow {
namespace experimental {
namespace cc {
class TensorHandle {
 public:
  Tensor Resolve(Status* status);
  static TensorHandle FromTensor(const Tensor& tensor, const Runtime& runtime,
                                 Status* status);
  TensorHandle(TensorHandle&&) = default;
  TensorHandle& operator=(TensorHandle&&) = default;
 private:
  explicit TensorHandle(TFE_TensorHandle* handle) : handle_(handle) {}
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;
  TFE_TensorHandle* GetTFETensorHandle() const { return handle_.get(); }
  void Reset(TFE_TensorHandle* handle) { handle_.reset(handle); }
  struct TFETensorHandleDeleter {
    void operator()(TFE_TensorHandle* p) const { TFE_DeleteTensorHandle(p); }
  };
  std::unique_ptr<TFE_TensorHandle, TFETensorHandleDeleter> handle_;
};
inline Tensor TensorHandle::Resolve(Status* status) {
  TF_Tensor* tensor =
      TFE_TensorHandleResolve(handle_.get(), status->GetTFStatus());
  if (!status->ok()) {
    return Tensor(nullptr);
  }
  return Tensor(tensor);
}
inline TensorHandle TensorHandle::FromTensor(const Tensor& tensor,
                                             const Runtime& runtime,
                                             Status* status) {
  TFE_TensorHandle* tensor_handle = TFE_NewTensorHandleFromTensor(
      runtime.GetTFEContext(), tensor.GetTFTensor(), status->GetTFStatus());
  if (!status->ok()) {
    return TensorHandle(nullptr);
  }
  return TensorHandle(tensor_handle);
}
}  
}  
}  
#endif  