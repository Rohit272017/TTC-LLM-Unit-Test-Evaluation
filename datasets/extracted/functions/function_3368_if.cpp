#include "tensorflow/core/data/service/byte_size.h"
#include <cstddef>
#include <string>
#include "absl/strings/str_cat.h"
namespace tensorflow {
namespace data {
size_t ByteSize::ToUnsignedBytes() const { return bytes_; }
double ByteSize::ToDoubleBytes() const { return static_cast<double>(bytes_); }
double ByteSize::ToDoubleKB() const { return *this / ByteSize::KB(1); }
double ByteSize::ToDoubleMB() const { return *this / ByteSize::MB(1); }
double ByteSize::ToDoubleGB() const { return *this / ByteSize::GB(1); }
double ByteSize::ToDoubleTB() const { return *this / ByteSize::TB(1); }
std::string ByteSize::DebugString() const {
  if (*this < ByteSize::KB(1)) {
    return absl::StrCat(ToUnsignedBytes(), "B");
  }
  if (*this < ByteSize::MB(1)) {
    return absl::StrCat(ToDoubleKB(), "KB");
  }
  if (*this < ByteSize::GB(1)) {
    return absl::StrCat(ToDoubleMB(), "MB");
  }
  if (*this < ByteSize::TB(1)) {
    return absl::StrCat(ToDoubleGB(), "GB");
  }
  return absl::StrCat(ToDoubleTB(), "TB");
}
}  
}  