#include "tensorflow/lite/core/acceleration/configuration/stable_delegate_registry.h"
#include <string>
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"
namespace tflite {
namespace delegates {
void StableDelegateRegistry::RegisterStableDelegate(
    const TfLiteStableDelegate* delegate) {
  auto* const instance = StableDelegateRegistry::GetSingleton();
  instance->RegisterStableDelegateImpl(delegate);
}
const TfLiteStableDelegate* StableDelegateRegistry::RetrieveStableDelegate(
    const std::string& name) {
  auto* const instance = StableDelegateRegistry::GetSingleton();
  return instance->RetrieveStableDelegateImpl(name);
}
void StableDelegateRegistry::RegisterStableDelegateImpl(
    const TfLiteStableDelegate* delegate) {
  absl::MutexLock lock(&mutex_);
  registry_[delegate->delegate_name] = delegate;
}
const TfLiteStableDelegate* StableDelegateRegistry::RetrieveStableDelegateImpl(
    const std::string& name) {
  absl::MutexLock lock(&mutex_);
  if (registry_.find(name) == registry_.end()) {
    return nullptr;
  } else {
    return registry_[name];
  }
}
StableDelegateRegistry* StableDelegateRegistry::GetSingleton() {
  static auto* instance = new StableDelegateRegistry();
  return instance;
}
}  
}  