#ifndef TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_
#define TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_
#include <cstdint>
#include <utility>
#include "absl/base/config.h"
namespace tensorstore {
#ifndef TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
#if ABSL_HAVE_BUILTIN(__builtin_LINE) && ABSL_HAVE_BUILTIN(__builtin_FILE)
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 1
#elif defined(__GNUC__) && __GNUC__ >= 5
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 1
#elif defined(_MSC_VER) && _MSC_VER >= 1926
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 1
#else
#define TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT 0
#endif
#endif
class SourceLocation {
  struct PrivateTag {
   private:
    explicit PrivateTag() = default;
    friend class SourceLocation;
  };
 public:
  constexpr SourceLocation() : line_(1), file_name_("") {}
#if TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
  static constexpr SourceLocation current(
      PrivateTag = PrivateTag{}, std::uint_least32_t line = __builtin_LINE(),
      const char* file_name = __builtin_FILE()) {
    return SourceLocation(line, file_name);
  }
#else
  static constexpr SourceLocation current() {
    return SourceLocation(1, "<source_location>");
  }
#endif
  const char* file_name() const { return file_name_; }
  constexpr std::uint_least32_t line() const { return line_; }
 private:
  constexpr SourceLocation(std::uint_least32_t line, const char* file_name)
      : line_(line), file_name_(file_name) {}
  std::uint_least32_t line_;
  const char* file_name_;
};
}  
#endif  