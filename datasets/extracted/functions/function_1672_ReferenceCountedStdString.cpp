#include "common/internal/reference_count.h"
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "common/data.h"
#include "internal/new.h"
#include "google/protobuf/message_lite.h"
namespace cel::common_internal {
template class DeletingReferenceCount<google::protobuf::MessageLite>;
template class DeletingReferenceCount<Data>;
namespace {
class ReferenceCountedStdString final : public ReferenceCounted {
 public:
  explicit ReferenceCountedStdString(std::string&& string) {
    (::new (static_cast<void*>(&string_[0])) std::string(std::move(string)))
        ->shrink_to_fit();
  }
  const char* data() const noexcept {
    return std::launder(reinterpret_cast<const std::string*>(&string_[0]))
        ->data();
  }
  size_t size() const noexcept {
    return std::launder(reinterpret_cast<const std::string*>(&string_[0]))
        ->size();
  }
 private:
  void Finalize() noexcept override {
    std::destroy_at(std::launder(reinterpret_cast<std::string*>(&string_[0])));
  }
  alignas(std::string) char string_[sizeof(std::string)];
};
class ReferenceCountedString final : public ReferenceCounted {
 public:
  static const ReferenceCountedString* New(const char* data, size_t size) {
    return ::new (internal::New(offsetof(ReferenceCountedString, data_) + size))
        ReferenceCountedString(size, data);
  }
  const char* data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }
 private:
  ReferenceCountedString(size_t size, const char* data) noexcept : size_(size) {
    std::memcpy(data_, data, size);
  }
  void Delete() noexcept override {
    void* const that = this;
    const auto size = size_;
    std::destroy_at(this);
    internal::SizedDelete(that, offsetof(ReferenceCountedString, data_) + size);
  }
  const size_t size_;
  char data_[];
};
}  
std::pair<absl::Nonnull<const ReferenceCount*>, absl::string_view>
MakeReferenceCountedString(absl::string_view value) {
  ABSL_DCHECK(!value.empty());
  const auto* refcount =
      ReferenceCountedString::New(value.data(), value.size());
  return std::pair{refcount,
                   absl::string_view(refcount->data(), refcount->size())};
}
std::pair<absl::Nonnull<const ReferenceCount*>, absl::string_view>
MakeReferenceCountedString(std::string&& value) {
  ABSL_DCHECK(!value.empty());
  const auto* refcount = new ReferenceCountedStdString(std::move(value));
  return std::pair{refcount,
                   absl::string_view(refcount->data(), refcount->size())};
}
}  