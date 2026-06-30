#include "tensorflow/lite/kernels/internal/reference/comparisons.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
namespace tflite {
namespace reference_ops {
BroadcastComparison4DSlowCommon BroadcastComparison4DSlowPreprocess(
    const RuntimeShape& unextended_input1_shape,
    const RuntimeShape& unextended_input2_shape,
    const RuntimeShape& unextended_output_shape) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  return {RuntimeShape::ExtendedShape(4, unextended_output_shape), desc1,
          desc2};
}
}  
}  