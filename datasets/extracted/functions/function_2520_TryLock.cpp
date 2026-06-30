#include "tensorflow/lite/experimental/acceleration/mini_benchmark/file_lock.h"
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif  
#include <string>
namespace tflite {
namespace acceleration {
bool FileLock::TryLock() {
#ifndef _WIN32
  if (fd_ < 0) {
    fd_ = open(path_.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
  }
  if (fd_ < 0) {
    return false;
  }
  if (flock(fd_, LOCK_EX | LOCK_NB) == 0) {
    return true;
  }
#endif  
  return false;
}
}  
}  