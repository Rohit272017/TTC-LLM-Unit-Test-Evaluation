#ifndef QUICHE_COMMON_PLATFORM_API_QUICHE_MEM_SLICE_H_
#define QUICHE_COMMON_PLATFORM_API_QUICHE_MEM_SLICE_H_
#include <cstddef>
#include <memory>
#include <utility>
#include "quiche_platform_impl/quiche_mem_slice_impl.h"
#include "absl/strings/string_view.h"
#include "quiche/common/platform/api/quiche_export.h"
#include "quiche/common/quiche_buffer_allocator.h"
#include "quiche/common/quiche_callbacks.h"
namespace quiche {
class QUICHE_EXPORT QuicheMemSlice {
 public:
  QuicheMemSlice() = default;
  explicit QuicheMemSlice(QuicheBuffer buffer) : impl_(std::move(buffer)) {}
  QuicheMemSlice(std::unique_ptr<char[]> buffer, size_t length)
      : impl_(std::move(buffer), length) {}
  QuicheMemSlice(const char* buffer, size_t length,
                 quiche::SingleUseCallback<void(const char*)> done_callback)
      : impl_(buffer, length, std::move(done_callback)) {}
  struct InPlace {};
  template <typename... Args>
  explicit QuicheMemSlice(InPlace, Args&&... args)
      : impl_{std::forward<Args>(args)...} {}
  QuicheMemSlice(const QuicheMemSlice& other) = delete;
  QuicheMemSlice& operator=(const QuicheMemSlice& other) = delete;
  QuicheMemSlice(QuicheMemSlice&& other) = default;
  QuicheMemSlice& operator=(QuicheMemSlice&& other) = default;
  ~QuicheMemSlice() = default;
  void Reset() { impl_.Reset(); }
  const char* data() const { return impl_.data(); }
  size_t length() const { return impl_.length(); }
  absl::string_view AsStringView() const {
    return absl::string_view(data(), length());
  }
  bool empty() const { return impl_.empty(); }
 private:
  QuicheMemSliceImpl impl_;
};
}  
#endif  