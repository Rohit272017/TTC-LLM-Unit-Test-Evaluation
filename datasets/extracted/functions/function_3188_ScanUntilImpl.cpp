#include "tsl/platform/scanner.h"
namespace tsl {
namespace strings {
void Scanner::ScanUntilImpl(char end_ch, bool escaped) {
  for (;;) {
    if (cur_.empty()) {
      Error();
      return;
    }
    const char ch = cur_[0];
    if (ch == end_ch) {
      return;
    }
    cur_.remove_prefix(1);
    if (escaped && ch == '\\') {
      if (cur_.empty()) {
        Error();
        return;
      }
      cur_.remove_prefix(1);
    }
  }
}
bool Scanner::GetResult(absl::string_view* remaining,
                        absl::string_view* capture) {
  if (error_) {
    return false;
  }
  if (remaining != nullptr) {
    *remaining = cur_;
  }
  if (capture != nullptr) {
    const char* end = capture_end_ == nullptr ? cur_.data() : capture_end_;
    *capture = absl::string_view(capture_start_, end - capture_start_);
  }
  return true;
}
}  
}  