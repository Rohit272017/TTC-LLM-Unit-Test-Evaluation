#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/convert/logger_registry.h"
#include <unordered_map>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
namespace tensorflow {
namespace tensorrt {
class LoggerRegistryImpl : public LoggerRegistry {
  Status Register(const string& name, nvinfer1::ILogger* logger) override {
    mutex_lock lock(mu_);
    if (!registry_.emplace(name, std::unique_ptr<nvinfer1::ILogger>(logger))
             .second) {
      return errors::AlreadyExists("Logger ", name, " already registered");
    }
    return OkStatus();
  }
  nvinfer1::ILogger* LookUp(const string& name) override {
    mutex_lock lock(mu_);
    const auto found = registry_.find(name);
    if (found == registry_.end()) {
      return nullptr;
    }
    return found->second.get();
  }
 private:
  mutable mutex mu_;
  mutable std::unordered_map<string, std::unique_ptr<nvinfer1::ILogger>>
      registry_ TF_GUARDED_BY(mu_);
};
LoggerRegistry* GetLoggerRegistry() {
  static LoggerRegistryImpl* registry = new LoggerRegistryImpl;
  return registry;
}
}  
}  
#endif  