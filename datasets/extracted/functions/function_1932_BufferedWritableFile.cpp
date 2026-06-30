#ifndef XLA_TSL_LIB_IO_BUFFERED_FILE_H_
#define XLA_TSL_LIB_IO_BUFFERED_FILE_H_
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "xla/tsl/lib/hash/crc32c.h"
#include "tsl/platform/cord.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/status.h"
namespace tsl {
class BufferedWritableFile : public WritableFile {
 public:
  explicit BufferedWritableFile(std::unique_ptr<WritableFile> file,
                                int64_t buffer_size = kDefaultBufferSize)
      : file_(std::move(file)) {
    buffer_.resize(buffer_size);
  }
  ~BufferedWritableFile() override { Close().IgnoreError(); }
  absl::Status Append(absl::string_view str_data) override {
    int64_t bytes_left = str_data.size();
    const char* data = str_data.data();
    while (bytes_left > 0) {
      int64_t append_bytes = std::min(
          static_cast<int64_t>(buffer_.size() - buffer_pos_), bytes_left);
      std::copy_n(data, append_bytes, buffer_.begin() + buffer_pos_);
      crc32_ = crc32c::Extend(crc32_, &buffer_[buffer_pos_], append_bytes);
      buffer_pos_ += append_bytes;
      if (buffer_pos_ == buffer_.size()) {
        TF_RETURN_IF_ERROR(file_->Append(buffer_));
        buffer_pos_ = 0;
      }
      data = data + append_bytes;
      bytes_left -= append_bytes;
    }
    return absl::OkStatus();
  }
  absl::Status Append(const absl::Cord& data) override {
    for (absl::string_view fragment : data.Chunks()) {
      TF_RETURN_IF_ERROR(Append(fragment));
    }
    return absl::OkStatus();
  }
  absl::Status Close() override {
    TF_RETURN_IF_ERROR(Flush());
    return file_->Close();
  }
  absl::Status Flush() override {
    if (buffer_pos_ > 0) {
      TF_RETURN_IF_ERROR(
          file_->Append(absl::string_view(&buffer_[0], buffer_pos_)));
      buffer_pos_ = 0;
    }
    return file_->Flush();
  }
  absl::Status Tell(int64_t* position) override {
    int64_t bytes_written;
    absl::Status status = file_->Tell(&bytes_written);
    if (status.ok()) {
      *position = bytes_written + buffer_pos_;
      return absl::OkStatus();
    } else {
      return status;
    }
  }
  absl::Status Sync() override { return file_->Sync(); }
  uint32_t crc32() const { return crc32_; }
  void reset_crc32() { crc32_ = 0; }
 private:
  static constexpr int64_t kDefaultBufferSize = 1048576;
  std::string buffer_;
  int64_t buffer_pos_ = 0;
  std::unique_ptr<WritableFile> file_;
  uint32_t crc32_ = 0;
  BufferedWritableFile(const BufferedWritableFile&) = delete;
  void operator=(const BufferedWritableFile&) = delete;
};
}  
#endif  