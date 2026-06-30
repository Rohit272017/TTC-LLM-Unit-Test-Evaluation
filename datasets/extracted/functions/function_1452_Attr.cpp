#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
namespace tensorflow {
REGISTER_OP("TRTEngineOp")
    .Attr("serialized_segment: string")
    .Attr("segment_func: func = {}")
    .Attr("InT: list({bool,int8,float16,float32,int32,resource})")
    .Attr("OutT: list({bool,int8,float16,float32,int32})")
    .Attr("input_shapes: list(shape) = []")
    .Attr("output_shapes: list(shape) = []")
    .Attr("max_cached_engines_count: int = 1")
    .Attr("max_batch_size: int = 1")
    .Attr("workspace_size_bytes: int")
    .Attr("precision_mode: {'FP32', 'FP16', 'INT8'}")
    .Attr("calibration_data: string = ''")
    .Attr("use_calibration: bool = true")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      std::vector<tensorflow::PartialTensorShape> output_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
      for (int i = 0; i < output_shapes.size(); i++) {
        ::tensorflow::shape_inference::ShapeHandle shape;
        shape_inference::ShapeHandle output_shape_handle;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            output_shapes[i], &output_shape_handle));
        c->set_output(i, output_shape_handle);
      }
      return OkStatus();
    })
    .Attr("segment_funcdef_name: string = ''")
    .Attr("cached_engine_batches: list(int) >= 0 = []")
    .Attr("fixed_input_size: bool = true")
    .Attr("static_engine: bool = true")
    .Attr("profile_strategy: string = ''")
    .Attr("use_explicit_precision: bool = false");
}  
#endif  