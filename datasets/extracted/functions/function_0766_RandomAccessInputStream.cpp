#include "xla/tsl/lib/io/random_inputstream.h"
#include <memory>
namespace tsl {
namespace io {
RandomAccessInputStream::RandomAccessInputStream(RandomAccessFile* file,
                                                 bool owns_file)
    : file_(file), owns_file_(owns_file) {}
RandomAccessInputStream::~RandomAccessInputStream() {
  if (owns_file_) {
    delete file_;
  }
}
absl::Status RandomAccessInputStream::ReadNBytes(int64_t bytes_to_read,
                                                 tstring* result) {
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Cannot read negative number of bytes");
  }
  result->clear();
  result->resize_uninitialized(bytes_to_read);
  char* result_buffer = &(*result)[0];
  absl::string_view data;
  absl::Status s = file_->Read(pos_, bytes_to_read, &data, result_buffer);
  if (data.data() != result_buffer) {
    memmove(result_buffer, data.data(), data.size());
  }
  result->resize(data.size());
  if (s.ok() || errors::IsOutOfRange(s)) {
    pos_ += data.size();
  }
  return s;
}
#if defined(TF_CORD_SUPPORT)
absl::Status RandomAccessInputStream::ReadNBytes(int64_t bytes_to_read,
                                                 absl::Cord* result) {
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Cannot read negative number of bytes");
  }
  int64_t current_size = result->size();
  absl::Status s = file_->Read(pos_, bytes_to_read, result);
  if (s.ok() || errors::IsOutOfRange(s)) {
    pos_ += result->size() - current_size;
  }
  return s;
}
#endif
static constexpr int64_t kMaxSkipSize = 8 * 1024 * 1024;
absl::Status RandomAccessInputStream::SkipNBytes(int64_t bytes_to_skip) {
  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can't skip a negative number of bytes");
  }
  std::unique_ptr<char[]> scratch(new char[kMaxSkipSize]);
  if (bytes_to_skip > 0) {
    absl::string_view data;
    absl::Status s =
        file_->Read(pos_ + bytes_to_skip - 1, 1, &data, scratch.get());
    if ((s.ok() || errors::IsOutOfRange(s)) && data.size() == 1) {
      pos_ += bytes_to_skip;
      return absl::OkStatus();
    }
  }
  while (bytes_to_skip > 0) {
    int64_t bytes_to_read = std::min<int64_t>(kMaxSkipSize, bytes_to_skip);
    absl::string_view data;
    absl::Status s = file_->Read(pos_, bytes_to_read, &data, scratch.get());
    if (s.ok() || errors::IsOutOfRange(s)) {
      pos_ += data.size();
    } else {
      return s;
    }
    if (data.size() < static_cast<size_t>(bytes_to_read)) {
      return errors::OutOfRange("reached end of file");
    }
    bytes_to_skip -= bytes_to_read;
  }
  return absl::OkStatus();
}
int64_t RandomAccessInputStream::Tell() const { return pos_; }
}  
}  