#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include <fcntl.h>
#include <string.h>
#ifndef _WIN32
#include <sys/file.h>
#include <unistd.h>
#endif
#include <fstream>
#include <sstream>
#include <string>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#ifndef TEMP_FAILURE_RETRY
#ifdef __ANDROID__
#error "TEMP_FAILURE_RETRY not set although on Android"
#else  
#define TEMP_FAILURE_RETRY(exp) exp
#endif  
#endif  
namespace tflite {
namespace acceleration {
FileStorage::FileStorage(absl::string_view path, ErrorReporter* error_reporter)
    : path_(path), error_reporter_(error_reporter) {}
MinibenchmarkStatus FileStorage::ReadFileIntoBuffer() {
#ifndef _WIN32
  buffer_.clear();
  int fd = TEMP_FAILURE_RETRY(open(path_.c_str(), O_RDONLY | O_CLOEXEC, 0600));
  int open_error_no = errno;
  if (fd < 0) {
    int fd = TEMP_FAILURE_RETRY(
        open(path_.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600));
    if (fd >= 0) {
      close(fd);
      return kMinibenchmarkSuccess;
    }
    int create_error_no = errno;
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Could not open %s for reading: %s, creating failed as well: %s",
        path_.c_str(), std::strerror(open_error_no),
        std::strerror(create_error_no));
    return kMinibenchmarkCantCreateStorageFile;
  }
  int lock_status = flock(fd, LOCK_EX);
  int lock_error_no = errno;
  if (lock_status < 0) {
    close(fd);
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not flock %s: %s",
                         path_.c_str(), std::strerror(lock_error_no));
    return kMinibenchmarkFlockingStorageFileFailed;
  }
  char buffer[512];
  while (true) {
    int bytes_read = TEMP_FAILURE_RETRY(read(fd, buffer, 512));
    int read_error_no = errno;
    if (bytes_read == 0) {
      close(fd);
      return kMinibenchmarkSuccess;
    } else if (bytes_read < 0) {
      close(fd);
      TF_LITE_REPORT_ERROR(error_reporter_, "Error reading %s: %s",
                           path_.c_str(), std::strerror(read_error_no));
      return kMinibenchmarkErrorReadingStorageFile;
    } else {
      buffer_.append(buffer, bytes_read);
    }
  }
#else  
  return kMinibenchmarkUnsupportedPlatform;
#endif
}
MinibenchmarkStatus FileStorage::AppendDataToFile(absl::string_view data) {
#ifndef _WIN32
  int fd = TEMP_FAILURE_RETRY(
      open(path_.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600));
  if (fd < 0) {
    int error_no = errno;
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not open %s for writing: %s",
                         path_.c_str(), std::strerror(error_no));
    return kMinibenchmarkFailedToOpenStorageFileForWriting;
  }
  int lock_status = flock(fd, LOCK_EX);
  int lock_error_no = errno;
  if (lock_status < 0) {
    close(fd);
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not flock %s: %s",
                         path_.c_str(), std::strerror(lock_error_no));
    return kMinibenchmarkFlockingStorageFileFailed;
  }
  absl::string_view bytes = data;
  while (!bytes.empty()) {
    ssize_t bytes_written =
        TEMP_FAILURE_RETRY(write(fd, bytes.data(), bytes.size()));
    if (bytes_written < 0) {
      int error_no = errno;
      close(fd);
      TF_LITE_REPORT_ERROR(error_reporter_, "Could not write to %s: %s",
                           path_.c_str(), std::strerror(error_no));
      return kMinibenchmarkErrorWritingStorageFile;
    }
    bytes.remove_prefix(bytes_written);
  }
  if (TEMP_FAILURE_RETRY(fsync(fd)) < 0) {
    int error_no = errno;
    close(fd);
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to fsync %s: %s",
                         path_.c_str(), std::strerror(error_no));
    return kMinibenchmarkErrorFsyncingStorageFile;
  }
  if (TEMP_FAILURE_RETRY(close(fd)) < 0) {
    int error_no = errno;
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to close %s: %s",
                         path_.c_str(), std::strerror(error_no));
    return kMinibenchmarkErrorClosingStorageFile;
  }
  return kMinibenchmarkSuccess;
#else   
  return kMinibenchmarkUnsupportedPlatform;
#endif  
}
const char kFlatbufferStorageIdentifier[] = "STO1";
}  
}  