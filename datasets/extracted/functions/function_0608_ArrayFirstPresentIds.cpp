#include "arolla/array/array_util.h"
#include <cstdint>
#include <vector>
#include "arolla/array/array.h"
#include "arolla/util/unit.h"
namespace arolla {
std::vector<int64_t> ArrayFirstPresentIds(const Array<Unit>& array,
                                          int64_t max_count) {
  std::vector<int64_t> res;
  if (max_count <= 0) {
    return res;
  }
  res.reserve(max_count);
  if (array.IsDenseForm() || array.HasMissingIdValue()) {
    int64_t index = 0;
    while (index < array.size() && res.size() < max_count) {
      if (array[index].present) res.push_back(index);
      index++;
    }
  } else {
    int64_t offset = 0;
    while (offset < array.dense_data().size() && res.size() < max_count) {
      if (array.dense_data().present(offset)) {
        res.push_back(array.id_filter().IdsOffsetToId(offset));
      }
      offset++;
    }
  }
  return res;
}
std::vector<int64_t> ArrayLastPresentIds(const Array<Unit>& array,
                                         int64_t max_count) {
  std::vector<int64_t> res;
  if (max_count <= 0) {
    return res;
  }
  res.reserve(max_count);
  if (array.IsDenseForm() || array.HasMissingIdValue()) {
    int64_t index = array.size() - 1;
    while (index >= 0 && res.size() < max_count) {
      if (array[index].present) res.push_back(index);
      index--;
    }
  } else {
    int64_t offset = array.dense_data().size() - 1;
    while (offset >= 0 && res.size() < max_count) {
      if (array.dense_data().present(offset)) {
        res.push_back(array.id_filter().IdsOffsetToId(offset));
      }
      offset--;
    }
  }
  return res;
}
}  