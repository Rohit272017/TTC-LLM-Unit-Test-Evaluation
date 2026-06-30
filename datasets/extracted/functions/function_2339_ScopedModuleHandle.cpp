#ifndef XLA_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_
#define XLA_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_
#include "absl/log/check.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/stream_executor.h"
namespace stream_executor {
class ScopedModuleHandle {
 public:
  ScopedModuleHandle(StreamExecutor* executor, ModuleHandle module_handle)
      : executor_(executor), module_handle_(module_handle) {}
  ScopedModuleHandle(ScopedModuleHandle&& other) {
    executor_ = other.executor_;
    module_handle_ = other.module_handle_;
    other.executor_ = nullptr;
    other.module_handle_ = ModuleHandle();
  }
  ScopedModuleHandle& operator=(ScopedModuleHandle&& other) {
    executor_ = other.executor_;
    module_handle_ = other.module_handle_;
    other.executor_ = nullptr;
    other.module_handle_ = ModuleHandle();
    return *this;
  }
  ~ScopedModuleHandle() {
    if (static_cast<bool>(module_handle_)) {
      CHECK(executor_->UnloadModule(module_handle_));
    }
  }
 private:
  StreamExecutor* executor_;
  ModuleHandle module_handle_;
  ScopedModuleHandle(const ScopedModuleHandle&) = delete;
  void operator=(const ScopedModuleHandle&) = delete;
};
}  
#endif  