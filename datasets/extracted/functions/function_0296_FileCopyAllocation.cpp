#include "tensorflow/compiler/mlir/lite/allocation.h"
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
namespace tflite {
#ifndef TFLITE_MCU
FileCopyAllocation::FileCopyAllocation(const char* filename,
                                       ErrorReporter* error_reporter)
    : Allocation(error_reporter, Allocation::Type::kFileCopy) {
  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(filename, "rb"), fclose);
  if (!file) {
    error_reporter_->Report("Could not open '%s'.", filename);
    return;
  }
  struct stat sb;
#ifdef _WIN32
#define FILENO(_x) _fileno(_x)
#else
#define FILENO(_x) fileno(_x)
#endif
  if (fstat(FILENO(file.get()), &sb) != 0) {
    error_reporter_->Report("Failed to get file size of '%s'.", filename);
    return;
  }
#undef FILENO
  buffer_size_bytes_ = sb.st_size;
  std::unique_ptr<char[]> buffer(new char[buffer_size_bytes_]);
  if (!buffer) {
    error_reporter_->Report("Malloc of buffer to hold copy of '%s' failed.",
                            filename);
    return;
  }
  size_t bytes_read =
      fread(buffer.get(), sizeof(char), buffer_size_bytes_, file.get());
  if (bytes_read != buffer_size_bytes_) {
    error_reporter_->Report("Read of '%s' failed (too few bytes read).",
                            filename);
    return;
  }
  copied_buffer_.reset(const_cast<char const*>(buffer.release()));
}
FileCopyAllocation::~FileCopyAllocation() {}
const void* FileCopyAllocation::base() const { return copied_buffer_.get(); }
size_t FileCopyAllocation::bytes() const { return buffer_size_bytes_; }
bool FileCopyAllocation::valid() const { return copied_buffer_ != nullptr; }
#endif
MemoryAllocation::MemoryAllocation(const void* ptr, size_t num_bytes,
                                   ErrorReporter* error_reporter)
    : Allocation(error_reporter, Allocation::Type::kMemory) {
#ifdef __arm__
  if ((reinterpret_cast<uintptr_t>(ptr) & 0x3) != 0) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "The supplied buffer is not 4-bytes aligned");
    buffer_ = nullptr;
    buffer_size_bytes_ = 0;
    return;
  }
#endif  
#if defined(__x86_64__) && defined(UNDEFINED_BEHAVIOR_SANITIZER)
  if ((reinterpret_cast<uintptr_t>(ptr) & 0x3) != 0) {
    aligned_ptr_ = ::aligned_alloc(4, num_bytes);
    if (aligned_ptr_ == nullptr) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to allocate aligned buffer");
      buffer_ = nullptr;
      buffer_size_bytes_ = 0;
      return;
    }
    memcpy(aligned_ptr_, ptr, num_bytes);
    buffer_ = aligned_ptr_;
  } else {
    buffer_ = ptr;
  }
#else   
  buffer_ = ptr;
#endif  
  buffer_size_bytes_ = num_bytes;
}
MemoryAllocation::~MemoryAllocation() {
#if defined(__x86_64__) && defined(UNDEFINED_BEHAVIOR_SANITIZER)
  if (aligned_ptr_) {
    free(aligned_ptr_);
  }
#endif
}
const void* MemoryAllocation::base() const { return buffer_; }
size_t MemoryAllocation::bytes() const { return buffer_size_bytes_; }
bool MemoryAllocation::valid() const { return buffer_ != nullptr; }
}  