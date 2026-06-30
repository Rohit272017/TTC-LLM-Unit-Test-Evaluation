#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <cerrno>
#include <string>
#include "absl/strings/numbers.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/tools/logging.h"
namespace tflite {
namespace delegates {
namespace utils {
namespace {
void setLibraryPathEnvironmentVariable(const std::string& delegate_path) {
  std::string directory_path = "";
  size_t last_slash_index = delegate_path.rfind('/');
  if (last_slash_index != std::string::npos) {
    directory_path = delegate_path.substr(0, last_slash_index);
  }
  if (setenv(kTfLiteLibraryPathEnvironmentVariable, directory_path.c_str(),
              1) != 0) {
    TFLITE_LOG(WARN) << "Error setting environment variable "
                     << kTfLiteLibraryPathEnvironmentVariable
                     << " with error: " << strerror(errno);
  }
}
}  
using ::tflite::acceleration::AndroidInfo;
using ::tflite::acceleration::RequestAndroidInfo;
const TfLiteStableDelegate* LoadDelegateFromSharedLibrary(
    const std::string& delegate_path) {
  void* symbol_pointer =
      LoadSymbolFromSharedLibrary(delegate_path, kTfLiteStableDelegateSymbol);
  if (!symbol_pointer) {
    return nullptr;
  }
  return reinterpret_cast<const TfLiteStableDelegate*>(symbol_pointer);
}
void* LoadSymbolFromSharedLibrary(const std::string& delegate_path,
                                  const std::string& delegate_symbol) {
  void* delegate_lib_handle = nullptr;
  int dlopen_flags = RTLD_NOW | RTLD_LOCAL;
  int sdk_version;
  AndroidInfo android_info;
  if (RequestAndroidInfo(&android_info).ok() &&
      absl::SimpleAtoi(android_info.android_sdk_version, &sdk_version) &&
      sdk_version >= 23) {
    dlopen_flags |= RTLD_NODELETE;
    TFLITE_LOG(INFO) << "Android SDK level is " << sdk_version
                     << ", using dlopen with RTLD_NODELETE.";
  }
  setLibraryPathEnvironmentVariable(delegate_path);
  delegate_lib_handle = dlopen(delegate_path.c_str(), dlopen_flags);
  if (!delegate_lib_handle) {
    TFLITE_LOG(ERROR) << "Failed to open library " << delegate_path << ": "
                      << dlerror();
    return nullptr;
  }
  void* symbol_pointer = dlsym(delegate_lib_handle, delegate_symbol.c_str());
  if (!symbol_pointer) {
    TFLITE_LOG(ERROR) << "Failed to find " << delegate_symbol
                      << " symbol: " << dlerror();
    dlclose(delegate_lib_handle);
    return nullptr;
  }
  return symbol_pointer;
}
}  
}  
}  