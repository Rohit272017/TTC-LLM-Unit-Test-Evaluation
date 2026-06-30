#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif
#include <thread>  
#include <type_traits>
namespace tensorstore {
namespace internal {
void TrySetCurrentThreadName(const char* name) {
  if (name == nullptr) return;
#if defined(__linux__)
  pthread_setname_np(pthread_self(), name);
#endif
#if defined(__APPLE__)
  pthread_setname_np(name);
#endif
}
}  
}  