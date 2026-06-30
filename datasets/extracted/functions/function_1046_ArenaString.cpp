#ifndef THIRD_PARTY_CEL_CPP_COMMON_INTERNAL_ARENA_STRING_H_
#define THIRD_PARTY_CEL_CPP_COMMON_INTERNAL_ARENA_STRING_H_
#include "absl/strings/string_view.h"
namespace cel::common_internal {
class ArenaString final {
 public:
  ArenaString() = default;
  ArenaString(const ArenaString&) = default;
  ArenaString& operator=(const ArenaString&) = default;
  explicit ArenaString(absl::string_view content) : content_(content) {}
  typename absl::string_view::size_type size() const { return content_.size(); }
  typename absl::string_view::const_pointer data() const {
    return content_.data();
  }
  operator absl::string_view() const { return content_; }
 private:
  absl::string_view content_;
};
}  
#endif  