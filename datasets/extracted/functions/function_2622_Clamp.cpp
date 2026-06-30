#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
#include <limits>
#include <vector>
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"
namespace tflite {
namespace strided_slice {
inline int Clamp(const int v, const int lo, const int hi) {
  TFLITE_DCHECK(!(hi < lo));
  if (hi < v) return hi;
  if (v < lo) return lo;
  return v;
}
inline void StridedSlicePadIndices(tflite::StridedSliceParams* p,
                                   int dim_count) {
  TFLITE_CHECK_LE(dim_count, 5);
  TFLITE_CHECK_GE(dim_count, p->start_indices_count);
  TFLITE_CHECK_EQ(p->start_indices_count, p->stop_indices_count);
  TFLITE_CHECK_EQ(p->stop_indices_count, p->strides_count);
  const int pad_count = dim_count - p->start_indices_count;
  for (int i = p->start_indices_count - 1; i >= 0; --i) {
    p->strides[i + pad_count] = p->strides[i];
    p->start_indices[i + pad_count] = p->start_indices[i];
    p->stop_indices[i + pad_count] = p->stop_indices[i];
  }
  for (int i = 0; i < pad_count; ++i) {
    p->start_indices[i] = 0;
    p->stop_indices[i] = 1;
    p->strides[i] = 1;
  }
  p->shrink_axis_mask <<= pad_count;
  p->ellipsis_mask <<= pad_count;
  p->new_axis_mask <<= pad_count;
  p->begin_mask <<= pad_count;
  p->end_mask <<= pad_count;
  p->begin_mask |= (1 << pad_count) - 1;
  p->end_mask |= (1 << pad_count) - 1;
  p->start_indices_count = dim_count;
  p->stop_indices_count = dim_count;
  p->strides_count = dim_count;
}
inline int StridedSliceStartForAxis(const tflite::StridedSliceParams& params,
                                    const RuntimeShape& input_shape,
                                    int32_t axis) {
  const int32_t axis_size = input_shape.Dims(axis);
  int32_t start = params.start_indices[axis];
  const int32_t stride = params.strides[axis];
  const int32_t begin_mask = (params.begin_mask & 1 << axis);
  if (start < 0) {
    start += axis_size;
  }
  if (stride > 0) {
    start = Clamp(start, 0, axis_size);
  } else {
    start = Clamp(start, -1, axis_size - 1);
  }
  if (begin_mask) {
    if (stride > 0) {
      start = 0;
    } else {
      start = axis_size - 1;
    }
  }
  return start;
}
inline int StridedSliceEndForAxis(const tflite::StridedSliceParams& params,
                                  const RuntimeShape& input_shape, int axis,
                                  int start) {
  const auto shrink_axis_mask = params.shrink_axis_mask;
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  const int axis_size = input_shape.Dims(axis);
  const bool offset = params.offset;
  if (shrink_axis) {
    if (start >= axis_size) {
      return start;
    } else {
      return start + 1;
    }
  }
  const auto* indices = params.stop_indices;
  int end = indices[axis];
  if (offset) {
    end += start;
  }
  const int32_t stride = params.strides[axis];
  const int32_t end_mask = (params.end_mask & 1 << axis);
  if (end < 0) {
    end += axis_size;
  }
  if (stride > 0) {
    end = Clamp(end, 0, axis_size);
  } else {
    end = Clamp(end, -1, axis_size - 1);
  }
  if (end_mask) {
    if (stride > 0) {
      end = axis_size;
    } else {
      end = -1;
    }
  }
  return end;
}
inline int StartForAxis(const tflite::StridedSliceParams& params,
                        const RuntimeShape& input_shape, int axis) {
  const auto begin_mask = params.begin_mask;
  const auto* start_indices = params.start_indices;
  const auto* strides = params.strides;
  const int axis_size = input_shape.Dims(axis);
  if (axis_size == 0) {
    return 0;
  }
  int start = start_indices[axis];
  if (begin_mask & 1 << axis) {
    if (strides[axis] > 0) {
      start = std::numeric_limits<int>::lowest();
    } else {
      start = std::numeric_limits<int>::max();
    }
  }
  if (start < 0) {
    start += axis_size;
  }
  if (strides[axis] > 0) {
    start = Clamp(start, 0, axis_size);
  } else {
    start = Clamp(start, -1, axis_size - 1);
  }
  return start;
}
inline int StopForAxis(const tflite::StridedSliceParams& params,
                       const RuntimeShape& input_shape, int axis,
                       int start_for_axis) {
  const auto end_mask = params.end_mask;
  const auto shrink_axis_mask = params.shrink_axis_mask;
  const auto* stop_indices = params.stop_indices;
  const auto* strides = params.strides;
  const int axis_size = input_shape.Dims(axis);
  if (axis_size == 0) {
    return 0;
  }
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  int stop = stop_indices[axis];
  if (shrink_axis) {
    return start_for_axis + 1;
  }
  if (end_mask & (1 << axis)) {
    if (strides[axis] > 0) {
      stop = std::numeric_limits<int>::max();
    } else {
      stop = std::numeric_limits<int>::lowest();
    }
  }
  if (stop < 0) {
    stop += axis_size;
  }
  if (strides[axis] > 0) {
    stop = Clamp(stop, 0, axis_size);
  } else {
    stop = Clamp(stop, -1, axis_size - 1);
  }
  return stop;
}
inline bool LoopCondition(int index, int stop, int stride) {
  return stride > 0 ? index >= stop : index <= stop;
}
inline tflite::StridedSliceParams BuildStridedSliceParams(
    int begin_mask, int end_mask, int shrink_axis_mask,
    const std::vector<int>& start_indices, const std::vector<int>& stop_indices,
    const std::vector<int>& strides) {
  tflite::StridedSliceParams op_params{};
  const int dims_count = start_indices.size();
  op_params.start_indices_count = dims_count;
  op_params.stop_indices_count = dims_count;
  op_params.strides_count = dims_count;
  for (int i = 0; i < dims_count; ++i) {
    op_params.start_indices[i] = start_indices[i];
    op_params.stop_indices[i] = stop_indices[i];
    op_params.strides[i] = strides[i];
  }
  op_params.begin_mask = begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = shrink_axis_mask;
  return op_params;
}
}  
}  
#endif  