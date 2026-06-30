#include "tensorflow/lite/delegates/gpu/android_hardware_buffer.h"
#include <dlfcn.h>
namespace tflite::gpu {
OptionalAndroidHardwareBuffer::OptionalAndroidHardwareBuffer() {
#ifdef __ANDROID__
  dlopen_handle_ = dlopen("libnativewindow.so", RTLD_NOW);
  if (dlopen_handle_ == nullptr) {
    supported_ = false;
    return;
  }
  allocate_ = reinterpret_cast<decltype(allocate_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_allocate"));
  acquire_ = reinterpret_cast<decltype(acquire_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_acquire"));
  release_ = reinterpret_cast<decltype(release_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_release"));
  describe_ = reinterpret_cast<decltype(describe_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_describe"));
  is_supported_ = reinterpret_cast<decltype(is_supported_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_isSupported"));
  supported_ =
      (allocate_ != nullptr && acquire_ != nullptr && release_ != nullptr &&
       describe_ != nullptr && is_supported_ != nullptr);
#else
  dlopen_handle_ = nullptr;
  allocate_ = nullptr;
  acquire_ = nullptr;
  release_ = nullptr;
  describe_ = nullptr;
  is_supported_ = nullptr;
  supported_ = false;
#endif
}
}  