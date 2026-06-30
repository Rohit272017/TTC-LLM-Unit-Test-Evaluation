#include "tensorflow/lite/delegates/hexagon/builders/min_max_builder.h"
#include "tensorflow/lite/core/c/common.h"
namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus MinMaxOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  int a_tensor_id = inputs->data[0];
  int b_tensor_id = inputs->data[1];
  const auto& a_tensor = context->tensors[a_tensor_id];
  const auto& b_tensor = context->tensors[b_tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(a_tensor_id));
  AddInput(graph_builder_->GetHexagonTensorId(b_tensor_id));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, a_tensor));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, b_tensor));
  const int output_tensor_id = outputs->data[0];
  const auto& output_tensor = context->tensors[output_tensor_id];
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, output_tensor));
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);
  return kTfLiteOk;
}
TfLiteStatus MinMaxOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}
OpBuilder* CreateMinMaxBuilder(GraphBuilder* graph_builder, int op_type) {
  return new MinMaxOpBuilder(graph_builder, op_type);
}
}  
}  
}  