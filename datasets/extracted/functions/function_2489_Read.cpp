#include "langsvr/buffer_reader.h"
#include <cstring>
namespace langsvr {
BufferReader::~BufferReader() = default;
size_t BufferReader::Read(std::byte* out, size_t count) {
    size_t n = std::min(count, bytes_remaining_);
    memcpy(out, data_, n);
    data_ += n;
    bytes_remaining_ -= n;
    return n;
}
}  