#include "arolla/jagged_shape/util/repr.h"
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "arolla/util/string.h"
namespace arolla {
std::string CompactSplitPointsAsSizesRepr(
    absl::Span<const int64_t> split_points, size_t max_part_size) {
  if (split_points.size() <= 1) {
    return "[]";
  }
  int64_t size = split_points[1] - split_points[0];
  if (absl::c_adjacent_find(split_points, [size](int64_t a, int64_t b) {
        return b - a != size;
      }) == split_points.end()) {
    return absl::StrCat(size);
  }
  std::ostringstream result;
  result << "[";
  bool first = true;
  const auto sizes_size = split_points.size() - 1;
  if (sizes_size <= 2 * max_part_size) {
    for (size_t i = 0; i < sizes_size; ++i) {
      result << NonFirstComma(first) << split_points[i + 1] - split_points[i];
    }
  } else {
    for (size_t i = 0; i < max_part_size; ++i) {
      result << NonFirstComma(first) << split_points[i + 1] - split_points[i];
    }
    result << NonFirstComma(first) << "...";
    for (size_t i = sizes_size - max_part_size; i < sizes_size; ++i) {
      result << NonFirstComma(first) << split_points[i + 1] - split_points[i];
    }
  }
  result << "]";
  return std::move(result).str();
}
}  