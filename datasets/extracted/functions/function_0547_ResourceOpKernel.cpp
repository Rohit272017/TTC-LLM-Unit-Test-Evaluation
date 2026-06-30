#ifndef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
template <typename T>
class ResourceOpKernel : public OpKernel {
 public:
  explicit ResourceOpKernel(OpKernelConstruction* context) : OpKernel(context) {
    has_resource_type_ = (context->output_type(0) == DT_RESOURCE);
    if (!has_resource_type_) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DT_STRING, TensorShape({2}), &tensor_));
    }
  }
  ~ResourceOpKernel() override {
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<T>(cinfo_.container(), cinfo_.name())
               .ok()) {
      }
    }
  }
  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    core::RefCountPtr<T> resource_ref_ptr = weak_resource_.GetNewRef();
    if (resource_ref_ptr == nullptr) {
      ResourceMgr* mgr = context->resource_manager();
      OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));
      T* resource;
      OP_REQUIRES_OK(context,
                     mgr->LookupOrCreate<T>(
                         cinfo_.container(), cinfo_.name(), &resource,
                         [this](T** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                           Status s = CreateResource(ret);
                           if (!s.ok() && *ret != nullptr) {
                             CHECK((*ret)->Unref());
                           }
                           return s;
                         }));
      core::ScopedUnref resource_unref(resource);
      OP_REQUIRES_OK(context, VerifyResource(resource));
      weak_resource_ = core::WeakPtr<T>(resource);
      resource_ = resource;
      if (!has_resource_type_) {
        auto h = tensor_.template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
    }
    if (has_resource_type_) {
      OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                  context, 0, cinfo_.container(), cinfo_.name(),
                                  TypeIndex::Make<T>()));
    } else {
      context->set_output_ref(0, &mu_, &tensor_);
    }
  }
 protected:
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  ABSL_DEPRECATED("Use get_resource() instead.")
  T* resource_ TF_GUARDED_BY(mu_) = nullptr;
  core::RefCountPtr<T> get_resource() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    return weak_resource_.GetNewRef();
  }
 private:
  core::WeakPtr<T> weak_resource_ TF_GUARDED_BY(mu_) =
      core::WeakPtr<T>(nullptr);
  virtual Status CreateResource(T** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;
  virtual Status VerifyResource(T* resource) { return absl::OkStatus(); }
  Tensor tensor_ TF_GUARDED_BY(mu_);
  bool has_resource_type_;
};
}  
#endif  