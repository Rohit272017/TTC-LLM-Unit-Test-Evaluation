#ifndef TENSORFLOW_LITE_KERNELS_TRANSPOSE_TEST_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_TRANSPOSE_TEST_UTILS_H_
#include <functional>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
namespace tflite {
template <typename T>
std::vector<T> RunTestPermutation(const absl::Span<const int> shape,
                                  const absl::Span<const int> perms) {
  const int count = absl::c_accumulate(shape, 1, std::multiplies<>{});
  std::vector<T> out(count);
  std::vector<T> input(count);
  absl::c_iota(input, static_cast<T>(0));
  const RuntimeShape input_shape(shape.size(), shape.data());
  RuntimeShape output_shape(perms.size());
  for (int i = 0; i < perms.size(); i++) {
    output_shape.SetDim(i, input_shape.Dims(perms[i]));
  }
  TransposeParams params{};
  params.perm_count = static_cast<int8_t>(perms.size());
  absl::c_copy(perms, params.perm);
  reference_ops::Transpose(params, input_shape, input.data(), output_shape,
                           out.data());
  return out;
}
}  
#endif  