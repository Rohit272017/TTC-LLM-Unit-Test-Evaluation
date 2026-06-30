#include "tensorstore/internal/testing/concurrent.h"  
#include "absl/log/absl_check.h"  
#include "absl/log/absl_log.h"    
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace tensorstore {
namespace internal_testing {
#ifdef _WIN32
TestConcurrentLock::TestConcurrentLock() {
  handle_ = ::CreateMutexA(nullptr,
                           FALSE,
                           "TensorStoreTestConcurrentMutex");
  ABSL_CHECK(handle_ != nullptr);
  if (::WaitForSingleObject(handle_, 0 ) != WAIT_OBJECT_0) {
    ABSL_LOG(INFO) << "Waiting on WIN32 Concurrent Lock";
    ABSL_CHECK(::WaitForSingleObject(handle_, INFINITE) == WAIT_OBJECT_0);
  }
}
TestConcurrentLock::~TestConcurrentLock() {
  ABSL_CHECK(::ReleaseMutex(handle_));
  ::CloseHandle(handle_);
}
#endif
}  
}  