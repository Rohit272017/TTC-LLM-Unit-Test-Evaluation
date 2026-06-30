#include <math.h>
#include <algorithm>
#include <cstddef>
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/lstm_eval.h"
#include "tensorflow/lite/kernels/op_macros.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace bidirectional_sequence_lstm {
constexpr int kInputTensor = 0;
constexpr int kFwInputToInputWeightsTensor = 1;  
constexpr int kFwInputToForgetWeightsTensor = 2;
constexpr int kFwInputToCellWeightsTensor = 3;
constexpr int kFwInputToOutputWeightsTensor = 4;
constexpr int kFwRecurrentToInputWeightsTensor = 5;  
constexpr int kFwRecurrentToForgetWeightsTensor = 6;
constexpr int kFwRecurrentToCellWeightsTensor = 7;
constexpr int kFwRecurrentToOutputWeightsTensor = 8;
constexpr int kFwCellToInputWeightsTensor = 9;    
constexpr int kFwCellToForgetWeightsTensor = 10;  
constexpr int kFwCellToOutputWeightsTensor = 11;  
constexpr int kFwInputGateBiasTensor = 12;  
constexpr int kFwForgetGateBiasTensor = 13;
constexpr int kFwCellGateBiasTensor = 14;
constexpr int kFwOutputGateBiasTensor = 15;
constexpr int kFwProjectionWeightsTensor = 16;  
constexpr int kFwProjectionBiasTensor = 17;  
constexpr int kBwInputToInputWeightsTensor = 18;  
constexpr int kBwInputToForgetWeightsTensor = 19;
constexpr int kBwInputToCellWeightsTensor = 20;
constexpr int kBwInputToOutputWeightsTensor = 21;
constexpr int kBwRecurrentToInputWeightsTensor = 22;  
constexpr int kBwRecurrentToForgetWeightsTensor = 23;
constexpr int kBwRecurrentToCellWeightsTensor = 24;
constexpr int kBwRecurrentToOutputWeightsTensor = 25;
constexpr int kBwCellToInputWeightsTensor = 26;   
constexpr int kBwCellToForgetWeightsTensor = 27;  
constexpr int kBwCellToOutputWeightsTensor = 28;  
constexpr int kBwInputGateBiasTensor = 29;  
constexpr int kBwForgetGateBiasTensor = 30;
constexpr int kBwCellGateBiasTensor = 31;
constexpr int kBwOutputGateBiasTensor = 32;
constexpr int kBwProjectionWeightsTensor = 33;  
constexpr int kBwProjectionBiasTensor = 34;  
constexpr int kFwInputActivationStateTensor = 35;
constexpr int kFwInputCellStateTensor = 36;
constexpr int kBwInputActivationStateTensor = 37;
constexpr int kBwInputCellStateTensor = 38;
constexpr int kAuxInputTensor = 39;  
constexpr int kFwAuxInputToInputWeightsTensor = 40;   
constexpr int kFwAuxInputToForgetWeightsTensor = 41;  
constexpr int kFwAuxInputToCellWeightsTensor = 42;    
constexpr int kFwAuxInputToOutputWeightsTensor = 43;  
constexpr int kBwAuxInputToInputWeightsTensor = 44;   
constexpr int kBwAuxInputToForgetWeightsTensor = 45;  
constexpr int kBwAuxInputToCellWeightsTensor = 46;    
constexpr int kBwAuxInputToOutputWeightsTensor = 47;  
constexpr int kFwOutputTensor = 0;
constexpr int kBwOutputTensor = 1;  
enum TemporaryTensor {
  kFwScratchBuffer = 0,
  kBwScratchBuffer = 1,
  kInputQuantized = 2,
  kFwActivationStateQuantized = 3,
  kBwActivationStateQuantized = 4,
  kFwCellStateQuantized = 5,
  kBwCellStateQuantized = 6,
  kInputScalingFactors = 7,
  kAuxInputScalingFactors = 8,
  kOutputStateScalingFactors = 9,
  kProductScalingFactors = 10,
  kRecoveredCellWeights = 11,
  kAccumScratchBuffer = 12,
  kInputZeroPoints = 13,
  kAuxInputZeroPoints = 14,
  kOutputStateZeroPoints = 15,
  kFwRowSums = 16,
  kBwRowSums = 17,
  kAuxInputQuantized = 18,  
  kNumTemporaryTensors = 19,
};
struct OpData {
  int scratch_tensor_index;
  bool compute_fw_row_sums = false;
  bool compute_bw_row_sums = false;
};
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  context->AddTensors(context, kNumTemporaryTensors,
                      &op_data->scratch_tensor_index);
  return op_data;
}
void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}
TfLiteStatus CheckLstmTensorDimensionsAndTypes(
    TfLiteContext* context, TfLiteNode* node, int n_input, int n_output,
    int n_cell, int input_to_input_weights_tensor,
    int input_to_forget_weights_tensor, int input_to_cell_weights_tensor,
    int input_to_output_weights_tensor, int recurrent_to_input_weights_tensor,
    int recurrent_to_forget_weights_tensor,
    int recurrent_to_cell_weights_tensor,
    int recurrent_to_output_weights_tensor, int cell_to_input_weights_tensor,
    int cell_to_forget_weights_tensor, int cell_to_output_weights_tensor,
    int input_gate_bias_tensor, int forget_gate_bias_tensor,
    int cell_gate_bias_tensor, int output_gate_bias_tensor,
    int projection_weights_tensor, int projection_bias_tensor) {
  const auto* params = reinterpret_cast<TfLiteBidirectionalSequenceLSTMParams*>(
      node->builtin_data);
  TF_LITE_ENSURE(context, params->cell_clip >= 0);
  TF_LITE_ENSURE(context, params->proj_clip >= 0);
  const TfLiteTensor* input_to_forget_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, input_to_forget_weights_tensor,
                                 &input_to_forget_weights));
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[1], n_input);
  TF_LITE_ENSURE(context, (input_to_forget_weights->type == kTfLiteFloat32) ||
                              (input_to_forget_weights->type == kTfLiteInt8) ||
                              (input_to_forget_weights->type == kTfLiteUInt8));
  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, input_to_input_weights_tensor);
  if (input_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[1], n_input);
    TF_LITE_ENSURE_TYPES_EQ(context, input_to_input_weights->type,
                            input_to_forget_weights->type);
  }
  const TfLiteTensor* input_to_cell_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, input_to_cell_weights_tensor,
                                 &input_to_cell_weights));
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[1], n_input);
  TF_LITE_ENSURE_TYPES_EQ(context, input_to_cell_weights->type,
                          input_to_forget_weights->type);
  const TfLiteTensor* input_to_output_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, input_to_output_weights_tensor,
                                 &input_to_output_weights));
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->data[1], n_input);
  TF_LITE_ENSURE_TYPES_EQ(context, input_to_output_weights->type,
                          input_to_forget_weights->type);
  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, recurrent_to_input_weights_tensor);
  if (recurrent_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[1],
                      n_output);
    TF_LITE_ENSURE_TYPES_EQ(context, recurrent_to_input_weights->type,
                            input_to_forget_weights->type);
  }
  const TfLiteTensor* recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, recurrent_to_forget_weights_tensor,
                            &recurrent_to_forget_weights));
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[0],
                    n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[1],
                    n_output);
  TF_LITE_ENSURE_TYPES_EQ(context, recurrent_to_forget_weights->type,
                          input_to_forget_weights->type);
  const TfLiteTensor* recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, recurrent_to_cell_weights_tensor,
                            &recurrent_to_cell_weights));
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[1],
                    n_output);
  TF_LITE_ENSURE_TYPES_EQ(context, recurrent_to_cell_weights->type,
                          input_to_forget_weights->type);
  const bool cifg_weights_all_or_none =
      ((input_to_input_weights != nullptr) &&
       (recurrent_to_input_weights != nullptr)) ||
      ((input_to_input_weights == nullptr) &&
       (recurrent_to_input_weights == nullptr));
  TF_LITE_ENSURE(context, cifg_weights_all_or_none == true);
  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, cell_to_input_weights_tensor);
  if (cell_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, cell_to_input_weights->type,
                            input_to_forget_weights->type);
  }
  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, cell_to_forget_weights_tensor);
  if (cell_to_forget_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, cell_to_forget_weights->type,
                            input_to_forget_weights->type);
  }
  const TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, cell_to_output_weights_tensor);
  if (cell_to_output_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, cell_to_output_weights->type,
                            input_to_forget_weights->type);
  }
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool peephole_weights_all_or_none =
      ((cell_to_input_weights != nullptr || use_cifg) &&
       (cell_to_forget_weights != nullptr) &&
       (cell_to_output_weights != nullptr)) ||
      ((cell_to_input_weights == nullptr) &&
       (cell_to_forget_weights == nullptr) &&
       (cell_to_output_weights == nullptr));
  TF_LITE_ENSURE(context, peephole_weights_all_or_none == true);
  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, input_gate_bias_tensor);
  if (use_cifg) {
    TF_LITE_ENSURE_EQ(context, input_gate_bias, nullptr);
  } else {
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, input_gate_bias->type, kTfLiteFloat32);
  }
  const TfLiteTensor* forget_gate_bias;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, forget_gate_bias_tensor, &forget_gate_bias));
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->data[0], n_cell);
  TF_LITE_ENSURE_TYPES_EQ(context, forget_gate_bias->type, kTfLiteFloat32);
  const TfLiteTensor* cell_gate_bias;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, cell_gate_bias_tensor,
                                          &cell_gate_bias));
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->type, kTfLiteFloat32);
  const TfLiteTensor* output_gate_bias;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, output_gate_bias_tensor, &output_gate_bias));
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->data[0], n_cell);
  TF_LITE_ENSURE_TYPES_EQ(context, output_gate_bias->type, kTfLiteFloat32);
  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, projection_weights_tensor);
  if (projection_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, projection_weights->type,
                            input_to_forget_weights->type);
  }
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, projection_bias_tensor);
  if (projection_bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->data[0], n_output);
    TF_LITE_ENSURE_TYPES_EQ(context, projection_bias->type, kTfLiteFloat32);
  }
  const bool projecton_tensors_consistent =
      ((projection_weights != nullptr) || (projection_bias == nullptr));
  TF_LITE_ENSURE(context, projecton_tensors_consistent == true);
  return kTfLiteOk;
}
TfLiteStatus CheckInputTensorDimensions(TfLiteContext* context,
                                        TfLiteNode* node, int n_input,
                                        int n_output, int n_cell) {
  TF_LITE_ENSURE_OK(
      context,
      CheckLstmTensorDimensionsAndTypes(
          context, node, n_input, n_output, n_cell,
          kFwInputToInputWeightsTensor, kFwInputToForgetWeightsTensor,
          kFwInputToCellWeightsTensor, kFwInputToOutputWeightsTensor,
          kFwRecurrentToInputWeightsTensor, kFwRecurrentToForgetWeightsTensor,
          kFwRecurrentToCellWeightsTensor, kFwRecurrentToOutputWeightsTensor,
          kFwCellToInputWeightsTensor, kFwCellToForgetWeightsTensor,
          kFwCellToOutputWeightsTensor, kFwInputGateBiasTensor,
          kFwForgetGateBiasTensor, kFwCellGateBiasTensor,
          kFwOutputGateBiasTensor, kFwProjectionWeightsTensor,
          kFwProjectionBiasTensor));
  TF_LITE_ENSURE_OK(
      context,
      CheckLstmTensorDimensionsAndTypes(
          context, node, n_input, n_output, n_cell,
          kBwInputToInputWeightsTensor, kBwInputToForgetWeightsTensor,
          kBwInputToCellWeightsTensor, kBwInputToOutputWeightsTensor,
          kBwRecurrentToInputWeightsTensor, kBwRecurrentToForgetWeightsTensor,
          kBwRecurrentToCellWeightsTensor, kBwRecurrentToOutputWeightsTensor,
          kBwCellToInputWeightsTensor, kBwCellToForgetWeightsTensor,
          kBwCellToOutputWeightsTensor, kBwInputGateBiasTensor,
          kBwForgetGateBiasTensor, kBwCellGateBiasTensor,
          kBwOutputGateBiasTensor, kBwProjectionWeightsTensor,
          kBwProjectionBiasTensor));
  return kTfLiteOk;
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  const auto* params = reinterpret_cast<TfLiteBidirectionalSequenceLSTMParams*>(
      node->builtin_data);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 48);
  TF_LITE_ENSURE_EQ(context, node->outputs->size,
                    params->merge_outputs ? 1 : 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, input->dims->size, 3);
  const bool time_major = params->time_major;
  const int max_time = time_major ? input->dims->data[0] : input->dims->data[1];
  const int n_batch = time_major ? input->dims->data[1] : input->dims->data[0];
  const int n_input = input->dims->data[2];
  const TfLiteTensor* fw_input_to_output_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwInputToOutputWeightsTensor,
                                 &fw_input_to_output_weights));
  const int n_fw_cell = fw_input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, fw_input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, fw_input_to_output_weights->dims->data[1],
                    n_input);
  const TfLiteTensor* bw_input_to_output_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwInputToOutputWeightsTensor,
                                 &bw_input_to_output_weights));
  const int n_bw_cell = bw_input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, bw_input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, bw_input_to_output_weights->dims->data[1],
                    n_input);
  TF_LITE_ENSURE_EQ(context, bw_input_to_output_weights->type,
                    fw_input_to_output_weights->type);
  const TfLiteTensor* fw_recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kFwRecurrentToOutputWeightsTensor,
                            &fw_recurrent_to_output_weights));
  TF_LITE_ENSURE_EQ(context, fw_recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, fw_recurrent_to_output_weights->dims->data[0],
                    n_fw_cell);
  TF_LITE_ENSURE_EQ(context, fw_recurrent_to_output_weights->type,
                    fw_input_to_output_weights->type);
  const int n_fw_output = fw_recurrent_to_output_weights->dims->data[1];
  const TfLiteTensor* bw_recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kBwRecurrentToOutputWeightsTensor,
                            &bw_recurrent_to_output_weights));
  TF_LITE_ENSURE_EQ(context, bw_recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, bw_recurrent_to_output_weights->dims->data[0],
                    n_bw_cell);
  TF_LITE_ENSURE_EQ(context, bw_recurrent_to_output_weights->type,
                    fw_input_to_output_weights->type);
  const int n_bw_output = bw_recurrent_to_output_weights->dims->data[1];
  TF_LITE_ENSURE_OK(
      context, CheckInputTensorDimensions(context, node, n_input, n_fw_output,
                                          n_fw_cell));
  const TfLiteTensor* aux_input =
      GetOptionalInputTensor(context, node, kAuxInputTensor);
  const TfLiteTensor* fw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToInputWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToCellWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToOutputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToInputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToCellWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToOutputWeightsTensor);
  const bool aux_inputs_weights_all_or_none =
      ((fw_aux_input_to_cell_weights != nullptr) &&
       (fw_aux_input_to_forget_weights != nullptr) &&
       (fw_aux_input_to_output_weights != nullptr) &&
       (bw_aux_input_to_cell_weights != nullptr) &&
       (bw_aux_input_to_forget_weights != nullptr) &&
       (bw_aux_input_to_output_weights != nullptr)) ||
      ((fw_aux_input_to_cell_weights == nullptr) &&
       (fw_aux_input_to_forget_weights == nullptr) &&
       (fw_aux_input_to_output_weights == nullptr) &&
       (bw_aux_input_to_cell_weights == nullptr) &&
       (bw_aux_input_to_forget_weights == nullptr) &&
       (bw_aux_input_to_output_weights == nullptr));
  TF_LITE_ENSURE(context, aux_inputs_weights_all_or_none);
  const bool has_aux_input = (fw_aux_input_to_forget_weights != nullptr);
  if (has_aux_input) {
    TF_LITE_ASSERT_EQ(aux_input->dims->data[0], input->dims->data[0]);
    TF_LITE_ASSERT_EQ(aux_input->dims->data[1], input->dims->data[1]);
  }
  TfLiteTensor* fw_output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFwOutputTensor, &fw_output));
  TfLiteTensor* fw_activation_state =
      GetVariableInput(context, node, kFwInputActivationStateTensor);
  TF_LITE_ENSURE(context, fw_activation_state != nullptr);
  TfLiteTensor* fw_cell_state =
      GetVariableInput(context, node, kFwInputCellStateTensor);
  TF_LITE_ENSURE(context, fw_cell_state != nullptr);
  TF_LITE_ENSURE_EQ(context, NumElements(fw_activation_state),
                    n_batch * n_fw_output);
  TF_LITE_ENSURE_EQ(context, NumElements(fw_cell_state), n_batch * n_fw_cell);
  TfLiteIntArray* fw_output_size = TfLiteIntArrayCreate(3);
  fw_output_size->data[0] = time_major ? max_time : n_batch;
  fw_output_size->data[1] = time_major ? n_batch : max_time;
  fw_output_size->data[2] =
      params->merge_outputs ? n_bw_output + n_fw_output : n_fw_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, fw_output, fw_output_size));
  const bool is_hybrid_op = IsHybridOp(input, fw_input_to_output_weights);
  TfLiteIntArrayFree(node->temporaries);
  if (is_hybrid_op) {
    node->temporaries = TfLiteIntArrayCreate(
        has_aux_input ? kNumTemporaryTensors : kNumTemporaryTensors - 1);
  } else {
    node->temporaries = TfLiteIntArrayCreate(2);  
  }
  node->temporaries->data[kFwScratchBuffer] =
      op_data->scratch_tensor_index + kFwScratchBuffer;
  TfLiteTensor* fw_scratch_buffer;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kFwScratchBuffer,
                                              &fw_scratch_buffer));
  fw_scratch_buffer->type = input->type;
  fw_scratch_buffer->allocation_type = kTfLiteArenaRw;
  const TfLiteTensor* fw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwInputToInputWeightsTensor);
  const bool fw_use_cifg = (fw_input_to_input_weights == nullptr);
  if (has_aux_input && !fw_use_cifg) {
    TF_LITE_ENSURE_EQ(context, fw_aux_input_to_input_weights->dims->data[0],
                      fw_input_to_input_weights->dims->data[0]);
  }
  TfLiteIntArray* fw_scratch_buffer_size = TfLiteIntArrayCreate(2);
  fw_scratch_buffer_size->data[0] = n_batch;
  if (fw_use_cifg) {
    fw_scratch_buffer_size->data[1] = n_fw_cell * 4 + 16;
  } else {
    fw_scratch_buffer_size->data[1] = n_fw_cell * 5 + 16;
  }
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, fw_scratch_buffer,
                                                   fw_scratch_buffer_size));
  TF_LITE_ENSURE_OK(
      context, CheckInputTensorDimensions(context, node, n_input, n_bw_output,
                                          n_bw_cell));
  TfLiteTensor* bw_activation_state =
      GetVariableInput(context, node, kBwInputActivationStateTensor);
  TF_LITE_ENSURE(context, bw_activation_state != nullptr);
  TfLiteTensor* bw_cell_state =
      GetVariableInput(context, node, kBwInputCellStateTensor);
  TF_LITE_ENSURE(context, bw_cell_state != nullptr);
  if (!params->merge_outputs) {
    TfLiteTensor* bw_output;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kBwOutputTensor, &bw_output));
    TfLiteIntArray* bw_output_size = TfLiteIntArrayCreate(3);
    bw_output_size->data[0] = time_major ? max_time : n_batch;
    bw_output_size->data[1] = time_major ? n_batch : max_time;
    bw_output_size->data[2] = n_bw_output;
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, bw_output, bw_output_size));
  }
  TF_LITE_ENSURE_EQ(context, NumElements(bw_activation_state),
                    n_batch * n_bw_output);
  TF_LITE_ENSURE_EQ(context, NumElements(bw_cell_state), n_batch * n_bw_cell);
  node->temporaries->data[kBwScratchBuffer] =
      op_data->scratch_tensor_index + kBwScratchBuffer;
  TfLiteTensor* bw_scratch_buffer;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kBwScratchBuffer,
                                              &bw_scratch_buffer));
  bw_scratch_buffer->type = input->type;
  bw_scratch_buffer->allocation_type = kTfLiteArenaRw;
  const TfLiteTensor* bw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwInputToInputWeightsTensor);
  const bool bw_use_cifg = (bw_input_to_input_weights == nullptr);
  if (has_aux_input && !bw_use_cifg) {
    TF_LITE_ENSURE_EQ(context, bw_aux_input_to_input_weights->dims->data[0],
                      bw_input_to_input_weights->dims->data[0]);
  }
  TfLiteIntArray* bw_scratch_buffer_size = TfLiteIntArrayCreate(2);
  bw_scratch_buffer_size->data[0] = n_batch;
  if (bw_use_cifg) {
    bw_scratch_buffer_size->data[1] = n_bw_cell * 4;
  } else {
    bw_scratch_buffer_size->data[1] = n_bw_cell * 5;
  }
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, bw_scratch_buffer,
                                                   bw_scratch_buffer_size));
  if (is_hybrid_op) {
    op_data->compute_fw_row_sums = true;
    op_data->compute_bw_row_sums = true;
    node->temporaries->data[kInputQuantized] =
        op_data->scratch_tensor_index + kInputQuantized;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kInputQuantized,
                                                &input_quantized));
    input_quantized->type = fw_input_to_output_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[kFwActivationStateQuantized] =
        op_data->scratch_tensor_index + kFwActivationStateQuantized;
    TfLiteTensor* fw_activation_state_quantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kFwActivationStateQuantized,
                                  &fw_activation_state_quantized));
    fw_activation_state_quantized->type = fw_input_to_output_weights->type;
    fw_activation_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(fw_activation_state_quantized->dims,
                             fw_activation_state->dims)) {
      TfLiteIntArray* fw_activation_state_quantized_size =
          TfLiteIntArrayCopy(fw_activation_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, fw_activation_state_quantized,
                                         fw_activation_state_quantized_size));
    }
    node->temporaries->data[kBwActivationStateQuantized] =
        op_data->scratch_tensor_index + kBwActivationStateQuantized;
    TfLiteTensor* bw_activation_state_quantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kBwActivationStateQuantized,
                                  &bw_activation_state_quantized));
    bw_activation_state_quantized->type = fw_input_to_output_weights->type;
    bw_activation_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(bw_activation_state_quantized->dims,
                             bw_activation_state->dims)) {
      TfLiteIntArray* bw_activation_state_quantized_size =
          TfLiteIntArrayCopy(bw_activation_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, bw_activation_state_quantized,
                                         bw_activation_state_quantized_size));
    }
    node->temporaries->data[kFwCellStateQuantized] =
        op_data->scratch_tensor_index + kFwCellStateQuantized;
    TfLiteTensor* fw_cell_state_quantized;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kFwCellStateQuantized,
                                       &fw_cell_state_quantized));
    fw_cell_state_quantized->type = fw_input_to_output_weights->type;
    fw_cell_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(fw_cell_state_quantized->dims,
                             fw_cell_state->dims)) {
      TfLiteIntArray* fw_cell_state_quantized_size =
          TfLiteIntArrayCopy(fw_cell_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, fw_cell_state_quantized,
                                              fw_cell_state_quantized_size));
    }
    node->temporaries->data[kBwCellStateQuantized] =
        op_data->scratch_tensor_index + kBwCellStateQuantized;
    TfLiteTensor* bw_cell_state_quantized;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kBwCellStateQuantized,
                                       &bw_cell_state_quantized));
    bw_cell_state_quantized->type = fw_input_to_output_weights->type;
    bw_cell_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(bw_cell_state_quantized->dims,
                             bw_cell_state->dims)) {
      TfLiteIntArray* bw_cell_state_quantized_size =
          TfLiteIntArrayCopy(bw_cell_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, bw_cell_state_quantized,
                                              bw_cell_state_quantized_size));
    }
    node->temporaries->data[kInputScalingFactors] =
        op_data->scratch_tensor_index + kInputScalingFactors;
    TfLiteTensor* input_sf;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, kInputScalingFactors, &input_sf));
    input_sf->type = kTfLiteFloat32;
    input_sf->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {n_batch};
    if (!TfLiteIntArrayEqualsArray(input_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_sf_size = TfLiteIntArrayCreate(1);
      input_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, input_sf, input_sf_size));
    }
    node->temporaries->data[kAuxInputScalingFactors] =
        op_data->scratch_tensor_index + kAuxInputScalingFactors;
    TfLiteTensor* aux_input_sf;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kAuxInputScalingFactors,
                                       &aux_input_sf));
    aux_input_sf->type = kTfLiteFloat32;
    aux_input_sf->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(aux_input_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* aux_input_sf_size = TfLiteIntArrayCreate(1);
      aux_input_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, aux_input_sf,
                                                       aux_input_sf_size));
    }
    node->temporaries->data[kOutputStateScalingFactors] =
        op_data->scratch_tensor_index + kOutputStateScalingFactors;
    TfLiteTensor* output_state_sf;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kOutputStateScalingFactors,
                                  &output_state_sf));
    output_state_sf->type = kTfLiteFloat32;
    output_state_sf->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_sf_size = TfLiteIntArrayCreate(1);
      output_state_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_sf,
                                                       output_state_sf_size));
    }
    node->temporaries->data[kProductScalingFactors] =
        op_data->scratch_tensor_index + kProductScalingFactors;
    TfLiteTensor* prod_scaling_factors;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kProductScalingFactors,
                                       &prod_scaling_factors));
    prod_scaling_factors->type = kTfLiteFloat32;
    prod_scaling_factors->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(prod_scaling_factors->dims, 1,
                                   scaling_dims)) {
      TfLiteIntArray* prod_scaling_factors_size = TfLiteIntArrayCreate(1);
      prod_scaling_factors_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, prod_scaling_factors,
                                              prod_scaling_factors_size));
    }
    node->temporaries->data[kRecoveredCellWeights] =
        op_data->scratch_tensor_index + kRecoveredCellWeights;
    TfLiteTensor* recovered_cell_weights;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kRecoveredCellWeights,
                                       &recovered_cell_weights));
    recovered_cell_weights->type = kTfLiteFloat32;
    recovered_cell_weights->allocation_type = kTfLiteArenaRw;
    int recovered_cell_dims[1] = {n_fw_cell};
    if (!TfLiteIntArrayEqualsArray(recovered_cell_weights->dims, 1,
                                   recovered_cell_dims)) {
      TfLiteIntArray* recovered_cell_weights_size = TfLiteIntArrayCreate(1);
      recovered_cell_weights_size->data[0] = n_fw_cell;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, recovered_cell_weights,
                                              recovered_cell_weights_size));
    }
    node->temporaries->data[kAccumScratchBuffer] =
        op_data->scratch_tensor_index + kAccumScratchBuffer;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, kAccumScratchBuffer, &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int n_cell = std::max(n_fw_cell, n_bw_cell);
    if (has_aux_input) {
      n_cell = std::max(n_cell, fw_aux_input_to_output_weights->dims->data[0]);
      n_cell = std::max(n_cell, bw_aux_input_to_output_weights->dims->data[0]);
    }
    int accum_scratch_dims[2] = {n_cell, n_batch};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
      accum_size->data[0] = n_cell;
      accum_size->data[1] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, accum_scratch, accum_size));
    }
    node->temporaries->data[kInputZeroPoints] =
        op_data->scratch_tensor_index + kInputZeroPoints;
    TfLiteTensor* input_zp;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kInputZeroPoints, &input_zp));
    input_zp->type = kTfLiteFloat32;
    input_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_zp_size = TfLiteIntArrayCreate(1);
      input_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, input_zp, input_zp_size));
    }
    node->temporaries->data[kAuxInputZeroPoints] =
        op_data->scratch_tensor_index + kAuxInputZeroPoints;
    TfLiteTensor* aux_input_zp;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, kAuxInputZeroPoints, &aux_input_zp));
    aux_input_zp->type = kTfLiteFloat32;
    aux_input_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(aux_input_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* aux_input_zp_size = TfLiteIntArrayCreate(1);
      aux_input_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, aux_input_zp,
                                                       aux_input_zp_size));
    }
    node->temporaries->data[kOutputStateZeroPoints] =
        op_data->scratch_tensor_index + kOutputStateZeroPoints;
    TfLiteTensor* output_state_zp;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kOutputStateZeroPoints,
                                       &output_state_zp));
    output_state_zp->type = kTfLiteFloat32;
    output_state_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_zp_size = TfLiteIntArrayCreate(1);
      output_state_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_zp,
                                                       output_state_zp_size));
    }
    int fw_row_sums_rows = fw_use_cifg ? 6 : 8;
    if (has_aux_input) {
      fw_row_sums_rows += fw_use_cifg ? 3 : 4;
    }
    const TfLiteTensor* fw_projection_weights =
        GetOptionalInputTensor(context, node, kFwProjectionWeightsTensor);
    if (fw_projection_weights != nullptr) {
      fw_row_sums_rows += ceil(static_cast<float>(n_fw_output) / n_fw_cell);
    }
    node->temporaries->data[kFwRowSums] =
        op_data->scratch_tensor_index + kFwRowSums;
    TfLiteTensor* fw_row_sums;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kFwRowSums, &fw_row_sums));
    fw_row_sums->type = kTfLiteInt32;
    fw_row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int fw_row_sums_dims[2] = {fw_row_sums_rows, n_fw_cell};
    if (!TfLiteIntArrayEqualsArray(fw_row_sums->dims, 2, fw_row_sums_dims)) {
      TfLiteIntArray* fw_hybrid_scratch_size = TfLiteIntArrayCreate(2);
      fw_hybrid_scratch_size->data[0] = fw_row_sums_dims[0];
      fw_hybrid_scratch_size->data[1] = fw_row_sums_dims[1];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, fw_row_sums,
                                                       fw_hybrid_scratch_size));
    }
    int bw_row_sums_rows = bw_use_cifg ? 6 : 8;
    if (has_aux_input) {
      bw_row_sums_rows += bw_use_cifg ? 3 : 4;
    }
    const TfLiteTensor* bw_projection_weights =
        GetOptionalInputTensor(context, node, kBwProjectionWeightsTensor);
    if (bw_projection_weights != nullptr) {
      bw_row_sums_rows += ceil(static_cast<float>(n_bw_output) / n_bw_cell);
    }
    node->temporaries->data[kBwRowSums] =
        op_data->scratch_tensor_index + kBwRowSums;
    TfLiteTensor* bw_row_sums;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kBwRowSums, &bw_row_sums));
    bw_row_sums->type = kTfLiteInt32;
    bw_row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int bw_row_sums_dims[2] = {bw_row_sums_rows, n_bw_cell};
    if (!TfLiteIntArrayEqualsArray(bw_row_sums->dims, 2, bw_row_sums_dims)) {
      TfLiteIntArray* bw_row_sums_size = TfLiteIntArrayCreate(2);
      bw_row_sums_size->data[0] = bw_row_sums_dims[0];
      bw_row_sums_size->data[1] = bw_row_sums_dims[1];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, bw_row_sums,
                                                       bw_row_sums_size));
    }
    if (has_aux_input) {
      node->temporaries->data[kAuxInputQuantized] =
          op_data->scratch_tensor_index + kAuxInputQuantized;
      TfLiteTensor* aux_input_quantized;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kAuxInputQuantized,
                                         &aux_input_quantized));
      aux_input_quantized->type = fw_input_to_output_weights->type;
      aux_input_quantized->allocation_type = kTfLiteArenaRw;
      if (!TfLiteIntArrayEqual(aux_input_quantized->dims, aux_input->dims)) {
        TfLiteIntArray* aux_input_quantized_size =
            TfLiteIntArrayCopy(aux_input->dims);
        TF_LITE_ENSURE_OK(context,
                          context->ResizeTensor(context, aux_input_quantized,
                                                aux_input_quantized_size));
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteBidirectionalSequenceLSTMParams*>(
      node->builtin_data);
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* fw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwInputToInputWeightsTensor);
  const TfLiteTensor* fw_input_to_forget_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwInputToForgetWeightsTensor,
                                 &fw_input_to_forget_weights));
  const TfLiteTensor* fw_input_to_cell_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwInputToCellWeightsTensor,
                                 &fw_input_to_cell_weights));
  const TfLiteTensor* fw_input_to_output_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwInputToOutputWeightsTensor,
                                 &fw_input_to_output_weights));
  const TfLiteTensor* fw_recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kFwRecurrentToInputWeightsTensor);
  const TfLiteTensor* fw_recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kFwRecurrentToForgetWeightsTensor,
                            &fw_recurrent_to_forget_weights));
  const TfLiteTensor* fw_recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwRecurrentToCellWeightsTensor,
                                 &fw_recurrent_to_cell_weights));
  const TfLiteTensor* fw_recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kFwRecurrentToOutputWeightsTensor,
                            &fw_recurrent_to_output_weights));
  const TfLiteTensor* fw_cell_to_input_weights =
      GetOptionalInputTensor(context, node, kFwCellToInputWeightsTensor);
  const TfLiteTensor* fw_cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kFwCellToForgetWeightsTensor);
  const TfLiteTensor* fw_cell_to_output_weights =
      GetOptionalInputTensor(context, node, kFwCellToOutputWeightsTensor);
  const TfLiteTensor* fw_input_gate_bias =
      GetOptionalInputTensor(context, node, kFwInputGateBiasTensor);
  const TfLiteTensor* fw_forget_gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwForgetGateBiasTensor,
                                 &fw_forget_gate_bias));
  const TfLiteTensor* fw_cell_gate_bias;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kFwCellGateBiasTensor,
                                          &fw_cell_gate_bias));
  const TfLiteTensor* fw_output_gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kFwOutputGateBiasTensor,
                                 &fw_output_gate_bias));
  const TfLiteTensor* fw_projection_weights =
      GetOptionalInputTensor(context, node, kFwProjectionWeightsTensor);
  const TfLiteTensor* fw_projection_bias =
      GetOptionalInputTensor(context, node, kFwProjectionBiasTensor);
  TfLiteTensor* fw_activation_state =
      GetVariableInput(context, node, kFwInputActivationStateTensor);
  TFLITE_DCHECK(fw_activation_state != nullptr);
  TfLiteTensor* fw_cell_state =
      GetVariableInput(context, node, kFwInputCellStateTensor);
  TFLITE_DCHECK(fw_cell_state != nullptr);
  TfLiteTensor* fw_output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kFwOutputTensor, &fw_output));
  const TfLiteTensor* bw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwInputToInputWeightsTensor);
  const TfLiteTensor* bw_input_to_forget_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwInputToForgetWeightsTensor,
                                 &bw_input_to_forget_weights));
  const TfLiteTensor* bw_input_to_cell_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwInputToCellWeightsTensor,
                                 &bw_input_to_cell_weights));
  const TfLiteTensor* bw_input_to_output_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwInputToOutputWeightsTensor,
                                 &bw_input_to_output_weights));
  const TfLiteTensor* bw_recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kBwRecurrentToInputWeightsTensor);
  const TfLiteTensor* bw_recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kBwRecurrentToForgetWeightsTensor,
                            &bw_recurrent_to_forget_weights));
  const TfLiteTensor* bw_recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwRecurrentToCellWeightsTensor,
                                 &bw_recurrent_to_cell_weights));
  const TfLiteTensor* bw_recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kBwRecurrentToOutputWeightsTensor,
                            &bw_recurrent_to_output_weights));
  const TfLiteTensor* bw_cell_to_input_weights =
      GetOptionalInputTensor(context, node, kBwCellToInputWeightsTensor);
  const TfLiteTensor* bw_cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kBwCellToForgetWeightsTensor);
  const TfLiteTensor* bw_cell_to_output_weights =
      GetOptionalInputTensor(context, node, kBwCellToOutputWeightsTensor);
  const TfLiteTensor* bw_input_gate_bias =
      GetOptionalInputTensor(context, node, kBwInputGateBiasTensor);
  const TfLiteTensor* bw_forget_gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwForgetGateBiasTensor,
                                 &bw_forget_gate_bias));
  const TfLiteTensor* bw_cell_gate_bias;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBwCellGateBiasTensor,
                                          &bw_cell_gate_bias));
  const TfLiteTensor* bw_output_gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kBwOutputGateBiasTensor,
                                 &bw_output_gate_bias));
  const TfLiteTensor* bw_projection_weights =
      GetOptionalInputTensor(context, node, kBwProjectionWeightsTensor);
  const TfLiteTensor* bw_projection_bias =
      GetOptionalInputTensor(context, node, kBwProjectionBiasTensor);
  TfLiteTensor* bw_activation_state =
      GetVariableInput(context, node, kBwInputActivationStateTensor);
  TFLITE_DCHECK(bw_activation_state != nullptr);
  TfLiteTensor* bw_cell_state =
      GetVariableInput(context, node, kBwInputCellStateTensor);
  TFLITE_DCHECK(bw_cell_state != nullptr);
  TfLiteTensor* bw_output = params->merge_outputs
                                ? nullptr
                                : GetOutput(context, node, kBwOutputTensor);
  TfLiteTensor* fw_scratch_buffer;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kFwScratchBuffer,
                                              &fw_scratch_buffer));
  TfLiteTensor* bw_scratch_buffer;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kBwScratchBuffer,
                                              &bw_scratch_buffer));
  const TfLiteTensor* aux_input =
      GetOptionalInputTensor(context, node, kAuxInputTensor);
  const TfLiteTensor* fw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToInputWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToCellWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToOutputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToInputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToCellWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToOutputWeightsTensor);
  const bool has_previous_bw_output = (aux_input != nullptr);
  const bool use_aux_input = (fw_aux_input_to_forget_weights != nullptr);
  TfLiteLSTMParams lstm_params = {params->activation, params->cell_clip,
                                  params->proj_clip, kTfLiteLSTMFullKernel,
                                  params->asymmetric_quantize_inputs};
  const int bw_output_offset =
      params->merge_outputs ? fw_recurrent_to_output_weights->dims->data[1] : 0;
  const auto actual_bw_output = params->merge_outputs ? fw_output : bw_output;
  const bool time_major = params->time_major;
  const bool non_stacking_mode = !use_aux_input && has_previous_bw_output;
  const TfLiteTensor* bw_input = non_stacking_mode ? aux_input : input;
  const TfLiteTensor* real_aux_input = non_stacking_mode ? nullptr : aux_input;
  switch (fw_input_to_output_weights->type) {
    case kTfLiteFloat32: {
      TfLiteStatus fw_pass_status = lstm_eval::EvalFloat(
          input, fw_input_to_input_weights, fw_input_to_forget_weights,
          fw_input_to_cell_weights, fw_input_to_output_weights,
          fw_recurrent_to_input_weights, fw_recurrent_to_forget_weights,
          fw_recurrent_to_cell_weights, fw_recurrent_to_output_weights,
          fw_cell_to_input_weights, fw_cell_to_forget_weights,
          fw_cell_to_output_weights,
          nullptr,
          nullptr,
          nullptr,
          nullptr, real_aux_input,
          fw_aux_input_to_input_weights, fw_aux_input_to_forget_weights,
          fw_aux_input_to_cell_weights, fw_aux_input_to_output_weights,
          fw_input_gate_bias, fw_forget_gate_bias, fw_cell_gate_bias,
          fw_output_gate_bias, fw_projection_weights, fw_projection_bias,
          &lstm_params,
          true, time_major, 0,
          fw_scratch_buffer, fw_activation_state, fw_cell_state, fw_output,
          false,
          false,
          false,
          false,
          CpuBackendContext::GetFromContext(context));
      TF_LITE_ENSURE_OK(context, fw_pass_status);
      TfLiteStatus bw_pass_status = lstm_eval::EvalFloat(
          bw_input, bw_input_to_input_weights, bw_input_to_forget_weights,
          bw_input_to_cell_weights, bw_input_to_output_weights,
          bw_recurrent_to_input_weights, bw_recurrent_to_forget_weights,
          bw_recurrent_to_cell_weights, bw_recurrent_to_output_weights,
          bw_cell_to_input_weights, bw_cell_to_forget_weights,
          bw_cell_to_output_weights,
          nullptr,
          nullptr,
          nullptr,
          nullptr, real_aux_input,
          bw_aux_input_to_input_weights, bw_aux_input_to_forget_weights,
          bw_aux_input_to_cell_weights, bw_aux_input_to_output_weights,
          bw_input_gate_bias, bw_forget_gate_bias, bw_cell_gate_bias,
          bw_output_gate_bias, bw_projection_weights, bw_projection_bias,
          &lstm_params,
          false, time_major, bw_output_offset,
          bw_scratch_buffer, bw_activation_state, bw_cell_state,
          actual_bw_output,
          false,
          false,
          false,
          false,
          CpuBackendContext::GetFromContext(context));
      TF_LITE_ENSURE_OK(context, bw_pass_status);
      return kTfLiteOk;
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      TfLiteTensor* input_quantized;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, kInputQuantized, &input_quantized));
      TfLiteTensor* fw_activation_state_quantized;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kFwActivationStateQuantized,
                                    &fw_activation_state_quantized));
      TfLiteTensor* bw_activation_state_quantized;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kBwActivationStateQuantized,
                                    &bw_activation_state_quantized));
      TfLiteTensor* fw_cell_state_quantized;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kFwCellStateQuantized,
                                         &fw_cell_state_quantized));
      TfLiteTensor* bw_cell_state_quantized;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kBwCellStateQuantized,
                                         &bw_cell_state_quantized));
      TfLiteTensor* prod_scaling_factors;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kProductScalingFactors,
                                         &prod_scaling_factors));
      TfLiteTensor* recovered_cell_weights;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, kRecoveredCellWeights,
                                         &recovered_cell_weights));
      TfLiteTensor* aux_input_quantized =
          use_aux_input ? GetTemporary(context, node, kAuxInputQuantized)
                        : nullptr;
      TfLiteTensor* accum_scratch;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, kAccumScratchBuffer, &accum_scratch));
      TfLiteTensor* fw_row_sums;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kFwRowSums, &fw_row_sums));
      TfLiteTensor* bw_row_sums;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kBwRowSums, &bw_row_sums));
      const int fw_row_sums_size = fw_row_sums->dims->data[0];
      const int bw_row_sums_size = bw_row_sums->dims->data[0];
      TfLiteStatus fw_pass_status = lstm_eval::EvalHybrid(
          input, fw_input_to_input_weights,
           nullptr, fw_input_to_forget_weights,
           nullptr, fw_input_to_cell_weights,
           nullptr, fw_input_to_output_weights,
           nullptr,
          fw_recurrent_to_input_weights,
           nullptr,
          fw_recurrent_to_forget_weights,
           nullptr,
          fw_recurrent_to_cell_weights,
           nullptr,
          fw_recurrent_to_output_weights,
           nullptr,
          fw_cell_to_input_weights, fw_cell_to_forget_weights,
          fw_cell_to_output_weights,
          nullptr,
          nullptr,
          nullptr,
          nullptr, real_aux_input,
          fw_aux_input_to_input_weights, fw_aux_input_to_forget_weights,
          fw_aux_input_to_cell_weights, fw_aux_input_to_output_weights,
          fw_input_gate_bias, fw_forget_gate_bias, fw_cell_gate_bias,
          fw_output_gate_bias, fw_projection_weights,
           nullptr, fw_projection_bias,
          &lstm_params,
          true, time_major, 0,
          fw_scratch_buffer, GetTemporary(context, node, kInputScalingFactors),
          GetTemporary(context, node, kAuxInputScalingFactors),
          GetTemporary(context, node, kOutputStateScalingFactors),
          prod_scaling_factors, recovered_cell_weights, input_quantized,
          aux_input_quantized, fw_activation_state_quantized,
          fw_cell_state_quantized, fw_activation_state, fw_cell_state,
          accum_scratch, fw_output,
          GetTemporary(context, node, kInputZeroPoints),
          GetTemporary(context, node, kAuxInputZeroPoints),
          GetTemporary(context, node, kOutputStateZeroPoints), fw_row_sums,
          fw_row_sums_size, &op_data->compute_fw_row_sums,
          false,
          false,
          false,
          false,
          CpuBackendContext::GetFromContext(context));
      TF_LITE_ENSURE_OK(context, fw_pass_status);
      TfLiteStatus bw_pass_status = lstm_eval::EvalHybrid(
          bw_input, bw_input_to_input_weights,
           nullptr, bw_input_to_forget_weights,
           nullptr, bw_input_to_cell_weights,
           nullptr, bw_input_to_output_weights,
           nullptr,
          bw_recurrent_to_input_weights,
           nullptr,
          bw_recurrent_to_forget_weights,
           nullptr,
          bw_recurrent_to_cell_weights,
           nullptr,
          bw_recurrent_to_output_weights,
           nullptr,
          bw_cell_to_input_weights, bw_cell_to_forget_weights,
          bw_cell_to_output_weights,
          nullptr,
          nullptr,
          nullptr,
          nullptr, real_aux_input,
          bw_aux_input_to_input_weights, bw_aux_input_to_forget_weights,
          bw_aux_input_to_cell_weights, bw_aux_input_to_output_weights,
          bw_input_gate_bias, bw_forget_gate_bias, bw_cell_gate_bias,
          bw_output_gate_bias, bw_projection_weights,
           nullptr, bw_projection_bias,
          &lstm_params,
          false, time_major, bw_output_offset,
          bw_scratch_buffer, GetTemporary(context, node, kInputScalingFactors),
          GetTemporary(context, node, kAuxInputScalingFactors),
          GetTemporary(context, node, kOutputStateScalingFactors),
          prod_scaling_factors, recovered_cell_weights, input_quantized,
          aux_input_quantized, bw_activation_state_quantized,
          bw_cell_state_quantized, bw_activation_state, bw_cell_state,
          accum_scratch, actual_bw_output,
          GetTemporary(context, node, kInputZeroPoints),
          GetTemporary(context, node, kAuxInputZeroPoints),
          GetTemporary(context, node, kOutputStateZeroPoints), bw_row_sums,
          bw_row_sums_size, &op_data->compute_bw_row_sums,
          false,
          false,
          false,
          false,
          CpuBackendContext::GetFromContext(context));
      TF_LITE_ENSURE_OK(context, bw_pass_status);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s is not currently supported.",
                         TfLiteTypeGetName(fw_input_to_output_weights->type));
      return kTfLiteError;
  }
}
}  
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM() {
  static TfLiteRegistration r = {
      bidirectional_sequence_lstm::Init, bidirectional_sequence_lstm::Free,
      bidirectional_sequence_lstm::Prepare, bidirectional_sequence_lstm::Eval};
  return &r;
}
}  
}  
}  