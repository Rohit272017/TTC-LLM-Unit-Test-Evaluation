#include "xla/service/cpu/runtime_topk.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
#include "absl/base/casts.h"
#include "absl/base/dynamic_annotations.h"
template <typename T>
static void TopK(int64_t batch_size, int64_t input_size, int64_t k,
                 const T* values, T* out_values, int32_t* out_indices) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(values,
                                      input_size * batch_size * sizeof(T));
  static constexpr auto convert_to_int = [](T value) {
    uint32_t x = absl::bit_cast<uint32_t>(value);
    return static_cast<int32_t>(x) < 0 ? std::numeric_limits<int32_t>::max() - x
                                       : x;
  };
  std::vector<int32_t> temp_indices(input_size);
  for (int64_t batch = 0; batch != batch_size; ++batch) {
    std::iota(temp_indices.begin(), temp_indices.end(), 0);
    const T* values_batch = values + batch * input_size;
    auto kth_element = temp_indices.begin() + k;
    std::partial_sort(temp_indices.begin(), kth_element, temp_indices.end(),
                      [values_batch](size_t i1, size_t i2) {
                        int32_t v1 = convert_to_int(values_batch[i1]);
                        int32_t v2 = convert_to_int(values_batch[i2]);
                        if (v1 == v2) {
                          return i1 < i2;  
                        }
                        return v1 > v2;
                      });
    T* out_values_batch = out_values + batch * k;
    int32_t* out_indices_batch = out_indices + batch * k;
    std::copy(temp_indices.begin(), kth_element, out_indices_batch);
    for (int64_t i = 0; i < k; i++) {
      out_values_batch[i] = values_batch[temp_indices[i]];
    }
  }
}
ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_TopKF32(
    int64_t batch_size, int64_t input_size, int64_t k, const float* values,
    float* out_values, int32_t* out_indices) {
  TopK(batch_size, input_size, k, values, out_values, out_indices);
}