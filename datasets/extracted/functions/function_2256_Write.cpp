#include "langsvr/buffer_writer.h"
namespace langsvr {
BufferWriter::BufferWriter() = default;
Result<SuccessType> BufferWriter::Write(const std::byte* in, size_t count) {
    size_t at = buffer.size();
    buffer.resize(at + count);
    memcpy(&buffer[at], in, count);
    return Success;
}
std::string_view BufferWriter::BufferString() const {
    if (buffer.empty()) {
        return "";
    }
    auto* data = reinterpret_cast<const char*>(&buffer[0]);
    static_assert(sizeof(std::byte) == sizeof(char), "length needs calculation");
    return std::string_view(data, buffer.size());
}
}  