#include "common/internal/shared_byte_string.h"
#include <cstdint>
#include <string>
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "common/allocator.h"
#include "common/internal/reference_count.h"
#include "google/protobuf/arena.h"
namespace cel::common_internal {
SharedByteString::SharedByteString(Allocator<> allocator,
                                   absl::string_view value)
    : header_(false, value.size()) {
  if (value.empty()) {
    content_.string.data = "";
    content_.string.refcount = 0;
  } else {
    if (auto* arena = allocator.arena(); arena != nullptr) {
      content_.string.data =
          google::protobuf::Arena::Create<std::string>(arena, value)->data();
      content_.string.refcount = 0;
      return;
    }
    auto pair = MakeReferenceCountedString(value);
    content_.string.data = pair.second.data();
    content_.string.refcount = reinterpret_cast<uintptr_t>(pair.first);
  }
}
SharedByteString::SharedByteString(Allocator<> allocator,
                                   const absl::Cord& value)
    : header_(allocator.arena() == nullptr,
              allocator.arena() == nullptr ? 0 : value.size()) {
  if (header_.is_cord) {
    ::new (static_cast<void*>(cord_ptr())) absl::Cord(value);
  } else {
    if (value.empty()) {
      content_.string.data = "";
    } else {
      auto* string = google::protobuf::Arena::Create<std::string>(allocator.arena());
      absl::CopyCordToString(value, string);
      content_.string.data = string->data();
    }
    content_.string.refcount = 0;
  }
}
}  