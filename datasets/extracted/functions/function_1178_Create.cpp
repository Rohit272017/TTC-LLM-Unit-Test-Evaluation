#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h"
#ifdef __ANDROID__
#include <dlfcn.h>
#endif
#include <stdio.h>
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
LibCHandle LibCHandle::Create(Status &status) {
#ifndef __ANDROID__
#ifndef _WIN32
  return LibCHandle(nullptr, ::fmemopen);
#else   
  status = {kTfLiteError, "Windows not supported."};
  return LibCHandle(nullptr, nullptr);
#endif  
#else   
  void *libc = nullptr;
  FmemopenPtr fmemopen_ptr = nullptr;
  if (!(libc = dlopen("libc.so", RTLD_NOW | RTLD_LOCAL))) {
    status = {kTfLiteError,
              "Failed to load the libc dynamic shared object library."};
    return LibCHandle(nullptr, nullptr);
  }
  if (!(fmemopen_ptr =
            reinterpret_cast<FmemopenPtr>(dlsym(libc, "fmemopen")))) {
    status = {kTfLiteError, "Failed to dynamically load the method: fmemopen"};
    return LibCHandle(nullptr, nullptr);
  }
  status = {kTfLiteOk, ""};
  return LibCHandle(libc, fmemopen_ptr);
#endif  
}
FILE *LibCHandle::fmemopen(void *buf, size_t size, const char *mode) const {
  return fmemopen_(buf, size, mode);
}
}  
}  
}  