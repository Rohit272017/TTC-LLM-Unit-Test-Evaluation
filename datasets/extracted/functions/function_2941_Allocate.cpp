#include "tensorstore/internal/ref_counted_string.h"
#include <cstring>
#include <new>
namespace tensorstore {
namespace internal {
RefCountedString& RefCountedString::operator=(
    const RefCountedString& other) noexcept {
  if (other.data_) other.header().IncrementReferenceCount();
  if (data_) header().DecrementReferenceCount();
  data_ = other.data_;
  return *this;
}
RefCountedString& RefCountedString::operator=(std::string_view s) {
  auto* data = AllocateCopy(s);
  if (data_) header().DecrementReferenceCount();
  data_ = data;
  return *this;
}
RefCountedString& RefCountedString::operator=(const char* s) {
  return *this = std::string_view(s);
}
char* RefCountedString::Allocate(size_t size) {
  if (size == 0) return nullptr;
  void* ptr = ::operator new(size + sizeof(Header));
  new (ptr) Header{size};
  return static_cast<char*>(ptr) + sizeof(Header);
}
const char* RefCountedString::AllocateCopy(std::string_view s) {
  if (s.empty()) return nullptr;
  char* data = Allocate(s.size());
  std::memcpy(data, s.data(), s.size());
  return data;
}
void RefCountedString::Header::Deallocate() const {
  ::operator delete(const_cast<Header*>(this), length + sizeof(Header));
}
}  
}  