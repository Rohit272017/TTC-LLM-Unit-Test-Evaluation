#ifndef AROLLA_MEMORY_BUFFER_H_
#define AROLLA_MEMORY_BUFFER_H_
#include <initializer_list>
#include <string>
#include <variant>
#include <vector>
#include "absl/strings/string_view.h"
#include "arolla/memory/simple_buffer.h"
#include "arolla/memory/strings_buffer.h"  
#include "arolla/memory/void_buffer.h"     
#include "arolla/util/bytes.h"
#include "arolla/util/text.h"
namespace arolla {
template <typename T>
struct BufferTraits {
  using buffer_type = SimpleBuffer<T>;
};
template <>
struct BufferTraits<std::monostate> {
  using buffer_type = VoidBuffer;
};
template <>
struct BufferTraits<std::string> {
  using buffer_type = StringsBuffer;
};
template <>
struct BufferTraits<absl::string_view> {
  using buffer_type = StringsBuffer;
};
template <>
struct BufferTraits<Text> {
  using buffer_type = StringsBuffer;
};
template <typename T>
using Buffer = typename BufferTraits<T>::buffer_type;
template <typename T>
Buffer<T> CreateBuffer(std::initializer_list<T> values) {
  return Buffer<T>::Create(values.begin(), values.end());
}
template <typename T>
Buffer<T> CreateBuffer(const std::vector<T>& values) {
  return Buffer<T>::Create(values.begin(), values.end());
}
}  
#endif  