#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_UTILS_H_
#include <stdint.h>
#include <algorithm>
#include <cstring>
namespace tflite {
namespace reduce_utils {
inline void RemoveSize1Dims(int* shape_out, int& out_num_dims, int* axis_out,
                            int& out_num_axis) {
  for (int64_t i = 0; i < out_num_dims;) {
    if (shape_out[i] == 1) {
      for (int64_t j = i + 1; j < out_num_dims; ++j) {
        shape_out[j - 1] = shape_out[j];
      }
      for (int64_t j = 0; j < out_num_axis; ++j) {
        if (axis_out[j] == i) {
          for (int64_t k = j + 1; k < out_num_axis; ++k) {
            axis_out[k - 1] = axis_out[k];
          }
          out_num_axis -= 1;
          break;
        }
      }
      for (int64_t j = 0; j < out_num_axis; ++j) {
        if (axis_out[j] > i) {
          axis_out[j] -= 1;
        }
      }
      --out_num_dims;
    } else {
      ++i;
    }
  }
}
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* axis_out,
                        int& out_num_axis, const int* shape_in, int* shape_out,
                        int& out_num_dims) {
  if (num_dims == 0) {
    out_num_axis = 0;
    out_num_dims = 0;
    return true;
  }
  out_num_axis = 0;
  out_num_dims = num_dims;
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    if (current < 0 || current >= num_dims) {
      return false;
    }
    bool is_dup = false;
    for (int j = 0; j < out_num_axis; ++j) {
      if (axis_out[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      axis_out[out_num_axis] = current;
      out_num_axis += 1;
    }
  }
  memcpy(shape_out, shape_in, num_dims * sizeof(int));
  std::sort(&axis_out[0], &axis_out[out_num_axis]);
  RemoveSize1Dims(shape_out, out_num_dims, axis_out, out_num_axis);
  if (out_num_axis > 0) {
    int64_t j = out_num_axis - 1;
    bool previous_here = (axis_out[j] == out_num_dims - 1);
    if (previous_here) {
      j -= 1;
    }
    for (int64_t i = out_num_dims - 2; i >= 0; --i) {
      bool current_here = j >= 0 ? (axis_out[j] == i) : false;
      if (current_here == previous_here) {
        shape_out[i] *= shape_out[i + 1];
        for (int64_t k = i + 1; k + 1 < out_num_dims; ++k) {
          shape_out[k] = shape_out[k + 1];
        }
        for (int64_t k = 0; k < out_num_axis; ++k) {
          if (axis_out[k] > i) {
            axis_out[k] -= 1;
          }
        }
        if (current_here) {
          for (int64_t k = j + 1; k + 1 < out_num_axis; ++k) {
            axis_out[k] = axis_out[k + 1];
          }
          out_num_axis -= 1;
        }
        out_num_dims -= 1;
      }
      if (current_here) {
        j -= 1;
      }
      previous_here = current_here;
    }
  }
  return true;
}
}  
}  
#endif  