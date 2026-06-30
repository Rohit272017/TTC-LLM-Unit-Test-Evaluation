#include <limits>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/gru_cell.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace custom {
namespace unidirectional_sequence_gru {
namespace {
void GruImpl(const TfLiteTensor* input, const TfLiteTensor* input_state,
             const TfLiteTensor* gate_weight, const TfLiteTensor* gate_bias,
             const TfLiteTensor* candidate_weight,
             const TfLiteTensor* candidate_bias, TfLiteTensor* output,
             TfLiteTensor* output_state, TfLiteTensor* activation,
             TfLiteTensor* concat,
             tflite::CpuBackendContext* cpu_backend_context) {
  const int n_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];
  const int n_output = output->dims->data[2];
  const int n_batch_input = n_batch * n_input;
  const int n_batch_output = n_batch * n_output;
  const RuntimeShape input_shape({n_batch, n_input});
  const float* input_data = GetTensorData<float>(input);
  const RuntimeShape state_shape = GetTensorShape(input_state);
  const float* input_state_data = GetTensorData<float>(input_state);
  const RuntimeShape gate_weight_shape = GetTensorShape(gate_weight);
  const float* gate_weight_data = GetTensorData<float>(gate_weight);
  const RuntimeShape gate_bias_shape = GetTensorShape(gate_bias);
  const float* gate_bias_data = GetTensorData<float>(gate_bias);
  const RuntimeShape candidate_weight_shape = GetTensorShape(candidate_weight);
  const float* candidate_weight_data = GetTensorData<float>(candidate_weight);
  const RuntimeShape candidate_bias_shape = GetTensorShape(candidate_bias);
  const float* candidate_bias_data = GetTensorData<float>(candidate_bias);
  const RuntimeShape activation_shape = GetTensorShape(activation);
  const RuntimeShape output_shape = RuntimeShape({n_batch, n_output});
  float* output_data = GetTensorData<float>(output);
  float* output_state_data = GetTensorData<float>(output_state);
  float* activation_data = GetTensorData<float>(activation);
  const RuntimeShape concat_shape = GetTensorShape(concat);
  float* concat_data = GetTensorData<float>(concat);
  tflite::FullyConnectedParams fc_params;
  fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  fc_params.float_activation_max = std::numeric_limits<float>::max();
  fc_params.lhs_cacheable =
      IsConstantTensor(gate_weight) && IsConstantTensor(candidate_weight);
  fc_params.rhs_cacheable = false;
  for (int i = 0; i < n_time; ++i) {
    gru_cell::GruCell(
        input_shape, input_data, state_shape, input_state_data,
        gate_weight_shape, gate_weight_data, gate_bias_shape, gate_bias_data,
        candidate_weight_shape, candidate_weight_data, candidate_bias_shape,
        candidate_bias_data, output_shape, output_data, output_state_data,
        activation_shape, activation_data, concat_shape, concat_data, fc_params,
        cpu_backend_context);
    input_data += n_batch_input;
    output_data += n_batch_output;
    input_state_data = output_state_data;
  }
}
}  
enum InputTensor {
  kInput = 0,
  kInputState = 1,
  kGateWeight = 2,
  kGateBias = 3,
  kCandidateWeight = 4,
  kCandidateBias = 5,
  kInputNum = 6
};
enum OutputTensor {
  kOutput = 0,
  kOutputState = 1,
  kOutputNum = 2
};
enum TemporaryTensor {
  kActivation = 0,
  kConcat = 1,
  kTemporaryNum = 2
};
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, kTemporaryNum, scratch_tensor_index);
  return scratch_tensor_index;
}
void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, kInputNum);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, kOutputNum);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInput, &input));
  TF_LITE_ENSURE_EQ(context, input->dims->size, 3);
  const int n_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];
  const TfLiteTensor* input_state;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputState, &input_state));
  TF_LITE_ENSURE_EQ(context, input_state->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_state->dims->data[0], n_batch);
  const int n_output = input_state->dims->data[1];
  const TfLiteTensor* gate_weight;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kGateWeight, &gate_weight));
  TF_LITE_ENSURE_EQ(context, gate_weight->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, gate_weight->dims->data[0], 2 * n_output);
  TF_LITE_ENSURE_EQ(context, gate_weight->dims->data[1], n_input + n_output);
  const TfLiteTensor* gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kGateBias, &gate_bias));
  TF_LITE_ENSURE_EQ(context, gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, gate_bias->dims->data[0], 2 * n_output);
  const TfLiteTensor* candidate_weight;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kCandidateWeight,
                                          &candidate_weight));
  TF_LITE_ENSURE_EQ(context, candidate_weight->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, candidate_weight->dims->data[0], n_output);
  TF_LITE_ENSURE_EQ(context, candidate_weight->dims->data[1],
                    n_input + n_output);
  const TfLiteTensor* candidate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kCandidateBias, &candidate_bias));
  TF_LITE_ENSURE_EQ(context, candidate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, candidate_bias->dims->data[0], n_output);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutput, &output));
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = n_time;
  output_size->data[1] = n_batch;
  output_size->data[2] = n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));
  TfLiteTensor* output_state;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputState, &output_state));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, output_state,
                                     TfLiteIntArrayCopy(input_state->dims)));
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kTemporaryNum);
  node->temporaries->data[kActivation] = *scratch_tensor_index;
  TfLiteTensor* activation;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kActivation, &activation));
  activation->type = input->type;
  activation->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray* activation_size = TfLiteIntArrayCreate(2);
  activation_size->data[0] = n_batch;
  activation_size->data[1] = 2 * n_output;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, activation, activation_size));
  node->temporaries->data[kConcat] = (*scratch_tensor_index) + kConcat;
  TfLiteTensor* concat;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kConcat, &concat));
  concat->type = input->type;
  concat->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray* concat_size = TfLiteIntArrayCreate(2);
  concat_size->data[0] = n_batch;
  concat_size->data[1] = n_input + n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, concat, concat_size));
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInput, &input));
  const TfLiteTensor* input_state;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputState, &input_state));
  const TfLiteTensor* gate_weight;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kGateWeight, &gate_weight));
  const TfLiteTensor* gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kGateBias, &gate_bias));
  const TfLiteTensor* candidate_weight;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kCandidateWeight,
                                          &candidate_weight));
  const TfLiteTensor* candidate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kCandidateBias, &candidate_bias));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutput, &output));
  TfLiteTensor* output_state;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputState, &output_state));
  TfLiteTensor* activation;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kActivation, &activation));
  TfLiteTensor* concat;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kConcat, &concat));
  auto cpu_backend_context = CpuBackendContext::GetFromContext(context);
  if (gate_weight->type == kTfLiteFloat32) {
    GruImpl(input, input_state, gate_weight, gate_bias, candidate_weight,
            candidate_bias, output, output_state, activation, concat,
            cpu_backend_context);
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Unsupported combination of data types for GruCell");
    return kTfLiteError;
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_GRU() {
  static TfLiteRegistration r = {
      unidirectional_sequence_gru::Init, unidirectional_sequence_gru::Free,
      unidirectional_sequence_gru::Prepare, unidirectional_sequence_gru::Eval};
  return &r;
}
}  
}  
}  