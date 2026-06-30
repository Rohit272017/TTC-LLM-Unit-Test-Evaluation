#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
namespace tensorflow {
REGISTER_OP("CreateTRTResourceHandle")
    .Attr("resource_name: string")
    .Output("resource_handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_OP("InitializeTRTResource")
    .Attr("max_cached_engines_count: int = 1")
    .Input("resource_handle: resource")
    .Input("filename: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);
REGISTER_OP("SerializeTRTResource")
    .Attr("delete_resource: bool = false")
    .Attr("save_gpu_specific_engines: bool = True")
    .Input("resource_name: string")
    .Input("filename: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);
}  
#endif  