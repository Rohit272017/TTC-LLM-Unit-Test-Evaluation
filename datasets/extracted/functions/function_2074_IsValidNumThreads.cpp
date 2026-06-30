#include "tensorflow/lite/kernels/eigen_support.h"
#include <functional>
#include <memory>
#include <utility>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/lite/kernels/op_macros.h"
#ifndef EIGEN_DONT_ALIGN
#include "tensorflow/lite/util.h"
#endif  
namespace tflite {
namespace eigen_support {
namespace {
const int kDefaultNumThreadpoolThreads = 4;
bool IsValidNumThreads(int num_threads) { return num_threads >= -1; }
int GetNumThreads(int num_threads) {
  return num_threads > -1 ? num_threads : kDefaultNumThreadpoolThreads;
}
#ifndef EIGEN_DONT_ALIGN
static_assert(
    kDefaultTensorAlignment % EIGEN_MAX_ALIGN_BYTES == 0,
    "kDefaultTensorAlignment doesn't comply with Eigen alignment requirement.");
#endif  
void SetEigenNbThreads(int threads) {
#if defined(EIGEN_HAS_OPENMP)
  Eigen::setNbThreads(threads);
#endif  
}
class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  explicit EigenThreadPoolWrapper(int num_threads) {
    if (num_threads > 1) {
      pool_ = std::make_unique<Eigen::ThreadPool>(num_threads);
    }
  }
  ~EigenThreadPoolWrapper() override {}
  void Schedule(std::function<void()> fn) override {
    if (pool_) {
      pool_->Schedule(std::move(fn));
    } else {
      fn();
    }
  }
  int NumThreads() const override { return pool_ ? pool_->NumThreads() : 1; }
  int CurrentThreadId() const override {
    return pool_ ? pool_->CurrentThreadId() : 0;
  }
 private:
  std::unique_ptr<Eigen::ThreadPool> pool_;
};
class LazyEigenThreadPoolHolder {
 public:
  explicit LazyEigenThreadPoolHolder(int num_threads) {
    SetNumThreads(num_threads);
  }
  const Eigen::ThreadPoolDevice* GetThreadPoolDevice() {
    if (!device_) {
      thread_pool_wrapper_ =
          std::make_unique<EigenThreadPoolWrapper>(target_num_threads_);
      device_ = std::make_unique<Eigen::ThreadPoolDevice>(
          thread_pool_wrapper_.get(), target_num_threads_);
    }
    return device_.get();
  }
  void SetNumThreads(int num_threads) {
    const int target_num_threads = GetNumThreads(num_threads);
    if (target_num_threads_ != target_num_threads) {
      target_num_threads_ = target_num_threads;
      device_.reset();
      thread_pool_wrapper_.reset();
    }
  }
 private:
  int target_num_threads_ = kDefaultNumThreadpoolThreads;
  std::unique_ptr<Eigen::ThreadPoolDevice> device_;
  std::unique_ptr<Eigen::ThreadPoolInterface> thread_pool_wrapper_;
};
struct RefCountedEigenContext : public TfLiteExternalContext {
  std::unique_ptr<LazyEigenThreadPoolHolder> thread_pool_holder;
  int num_references = 0;
};
RefCountedEigenContext* GetEigenContext(TfLiteContext* context) {
  return reinterpret_cast<RefCountedEigenContext*>(
      context->GetExternalContext(context, kTfLiteEigenContext));
}
TfLiteStatus Refresh(TfLiteContext* context) {
  if (IsValidNumThreads(context->recommended_num_threads)) {
    SetEigenNbThreads(GetNumThreads(context->recommended_num_threads));
  }
  auto* ptr = GetEigenContext(context);
  if (ptr != nullptr) {
    ptr->thread_pool_holder->SetNumThreads(context->recommended_num_threads);
  }
  return kTfLiteOk;
}
}  
void IncrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    if (IsValidNumThreads(context->recommended_num_threads)) {
      SetEigenNbThreads(context->recommended_num_threads);
    }
    ptr = new RefCountedEigenContext;
    ptr->type = kTfLiteEigenContext;
    ptr->Refresh = Refresh;
    ptr->thread_pool_holder = std::make_unique<LazyEigenThreadPoolHolder>(
        context->recommended_num_threads);
    ptr->num_references = 0;
    context->SetExternalContext(context, kTfLiteEigenContext, ptr);
  }
  ptr->num_references++;
}
void DecrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references == 0) {
    delete ptr;
    context->SetExternalContext(context, kTfLiteEigenContext, nullptr);
  }
}
const Eigen::ThreadPoolDevice* GetThreadPoolDevice(TfLiteContext* context) {
  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to GetFromContext() not preceded by IncrementUsageCounter()");
  }
  return ptr->thread_pool_holder->GetThreadPoolDevice();
}
}  
}  