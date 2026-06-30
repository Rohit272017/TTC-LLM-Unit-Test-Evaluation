#include "arolla/io/wildcard_input_loader.h"
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
namespace arolla::input_loader_impl {
std::function<std::optional<std::string>(absl::string_view)> MakeNameToKeyFn(
    const absl::ParsedFormat<'s'>& format) {
  constexpr absl::string_view kUniqueString =
      "unique_string_5a7cf4c5ed2d49068302b641bad242aa";
  auto formatted = absl::StrFormat(format, kUniqueString);
  size_t prefix_end = formatted.find(kUniqueString);
  DCHECK(prefix_end != absl::string_view::npos);
  std::string prefix = formatted.substr(0, prefix_end);
  size_t suffix_begin = prefix_end + kUniqueString.size();
  DCHECK(suffix_begin <= formatted.size());
  std::string suffix = formatted.substr(suffix_begin);
  return [prefix = std::move(prefix), suffix = std::move(suffix)](
             absl::string_view name) -> std::optional<std::string> {
    if (!absl::ConsumePrefix(&name, prefix)) {
      return std::nullopt;
    }
    if (!absl::ConsumeSuffix(&name, suffix)) {
      return std::nullopt;
    }
    return std::string(name);
  };
}
}  