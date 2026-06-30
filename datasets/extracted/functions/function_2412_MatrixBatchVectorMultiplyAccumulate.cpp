#include "tensorflow/lite/kernels/lstm_eval.h"
#include <math.h>
#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include "ruy/matrix.h"  
#include "ruy/mul_params.h"  
#include "ruy/profiler/instrumentation.h"  
#include "ruy/ruy.h"  
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/op_macros.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace lstm_eval {
namespace {
void MatrixBatchVectorMultiplyAccumulate(
    const float* matrix, const float* vector, const float* result,
    float* output, int m_rows, int m_cols, int n_batch,
    CpuBackendContext* cpu_backend_context) {
  tflite::FullyConnectedParams float_fc_params;
  float_fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  float_fc_params.float_activation_max = std::numeric_limits<float>::max();
  float_fc_params.lhs_cacheable = true;
  float_fc_params.rhs_cacheable = false;
  tflite::RuntimeShape weight_shape({m_rows, m_cols});
  tflite::RuntimeShape input_shape({n_batch, m_cols});
  tflite::RuntimeShape output_shape({n_batch, m_rows});
  if (n_batch == 1) {
    tflite::optimized_ops::FullyConnected(
        float_fc_params, input_shape, vector, weight_shape, matrix,
        output_shape, result, output_shape, output, cpu_backend_context);
  } else {
    tflite::optimized_ops::FullyConnected(
        float_fc_params, input_shape, vector, weight_shape, matrix,
        output_shape, nullptr, output_shape, output, cpu_backend_context);
    for (int i = 0; i < m_rows * n_batch; ++i) {
      output[i] += result[i];
    }
  }
}
void ComputeRowSums(
    int32_t* input_to_input_row_sums, int32_t* input_to_forget_row_sums,
    int32_t* input_to_cell_row_sums, int32_t* input_to_output_row_sums,
    int32_t* aux_input_to_input_row_sums, int32_t* aux_input_to_forget_row_sums,
    int32_t* aux_input_to_cell_row_sums, int32_t* aux_input_to_output_row_sums,
    int32_t* recurrent_to_input_row_sums, int32_t* recurrent_to_forget_row_sums,
    int32_t* recurrent_to_cell_row_sums, int32_t* recurrent_to_output_row_sums,
    int32_t* projection_weights_row_sums, int32_t* row_sums, int n_cell,
    int n_input, int n_aux_input, int n_output,
    const int8_t* input_to_input_weights_ptr,
    const int8_t* input_to_forget_weights_ptr,
    const int8_t* input_to_cell_weights_ptr,
    const int8_t* input_to_output_weights_ptr,
    const int8_t* aux_input_to_input_weights_ptr,
    const int8_t* aux_input_to_forget_weights_ptr,
    const int8_t* aux_input_to_cell_weights_ptr,
    const int8_t* aux_input_to_output_weights_ptr,
    const int8_t* recurrent_to_input_weights_ptr,
    const int8_t* recurrent_to_forget_weights_ptr,
    const int8_t* recurrent_to_cell_weights_ptr,
    const int8_t* recurrent_to_output_weights_ptr,
    const int8_t* projection_weights_ptr, bool use_cifg,
    const float* aux_input_ptr, bool recurrent_to_input_is_diag = false,
    bool recurrent_to_forget_is_diag = false,
    bool recurrent_to_cell_is_diag = false,
    bool recurrent_to_output_is_diag = false) {
  if (!use_cifg) {
    tensor_utils::ReductionSumVector(input_to_input_weights_ptr,
                                     input_to_input_row_sums, n_cell, n_input);
  }
  tensor_utils::ReductionSumVector(input_to_forget_weights_ptr,
                                   input_to_forget_row_sums, n_cell, n_input);
  tensor_utils::ReductionSumVector(input_to_cell_weights_ptr,
                                   input_to_cell_row_sums, n_cell, n_input);
  tensor_utils::ReductionSumVector(input_to_output_weights_ptr,
                                   input_to_output_row_sums, n_cell, n_input);
  if (aux_input_ptr) {
    if (!use_cifg) {
      tensor_utils::ReductionSumVector(aux_input_to_input_weights_ptr,
                                       aux_input_to_input_row_sums, n_cell,
                                       n_aux_input);
    }
    tensor_utils::ReductionSumVector(aux_input_to_forget_weights_ptr,
                                     aux_input_to_forget_row_sums, n_cell,
                                     n_aux_input);
    tensor_utils::ReductionSumVector(aux_input_to_cell_weights_ptr,
                                     aux_input_to_cell_row_sums, n_cell,
                                     n_aux_input);
    tensor_utils::ReductionSumVector(aux_input_to_output_weights_ptr,
                                     aux_input_to_output_row_sums, n_cell,
                                     n_aux_input);
  }
  if (!use_cifg) {
    if (!recurrent_to_input_is_diag) {
      tensor_utils::ReductionSumVector(recurrent_to_input_weights_ptr,
                                       recurrent_to_input_row_sums, n_cell,
                                       n_output);
    }
  }
  if (!recurrent_to_forget_is_diag) {
    tensor_utils::ReductionSumVector(recurrent_to_forget_weights_ptr,
                                     recurrent_to_forget_row_sums, n_cell,
                                     n_output);
  }
  if (!recurrent_to_cell_is_diag) {
    tensor_utils::ReductionSumVector(recurrent_to_cell_weights_ptr,
                                     recurrent_to_cell_row_sums, n_cell,
                                     n_output);
  }
  if (!recurrent_to_output_is_diag) {
    tensor_utils::ReductionSumVector(recurrent_to_output_weights_ptr,
                                     recurrent_to_output_row_sums, n_cell,
                                     n_output);
  }
  if (projection_weights_ptr != nullptr) {
    tensor_utils::ReductionSumVector(
        projection_weights_ptr, projection_weights_row_sums, n_output, n_cell);
  }
}
inline float GetTensorScale(const TfLiteTensor* tensor) {
  return tensor == nullptr ? 1.0f : tensor->params.scale;
}
inline void CalculateLstmGateFloat(
    const float* input, const float* input_to_gate_weights,
    const float* aux_input, const float* aux_input_to_gate_weights,
    const float* output_state, const float* recurrent_to_gate_weights,
    const float* cell_state, const float* cell_to_gate_weights,
    const float* layer_norm_coefficients, const float* gate_bias,
    const int n_batch, const int n_input, const int n_aux_input,
    const int n_output, const int n_cell,
    const TfLiteFusedActivation activation, float* gate,
    const bool is_input_all_zeros, const bool is_aux_input_all_zeros,
    float* output, bool recurrent_is_diag, CpuBackendContext* context) {
  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);
  if (use_layer_norm) {
    std::fill_n(gate, n_cell * n_batch, 0.0f);
  } else {
    tensor_utils::VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  }
  float* accumulation_buffer = gate;
  if (!is_input_all_zeros) {
    MatrixBatchVectorMultiplyAccumulate(input_to_gate_weights, input,
                                        accumulation_buffer, output, n_cell,
                                        n_input, n_batch, context);
    std::swap(accumulation_buffer, output);
  }
  if (!is_aux_input_all_zeros) {
    MatrixBatchVectorMultiplyAccumulate(aux_input_to_gate_weights, aux_input,
                                        accumulation_buffer, output, n_cell,
                                        n_aux_input, n_batch, context);
    std::swap(accumulation_buffer, output);
  }
  if (recurrent_is_diag) {
    tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        recurrent_to_gate_weights, n_cell, output_state, n_batch,
        accumulation_buffer);
    std::swap(accumulation_buffer, output);
  } else {
    MatrixBatchVectorMultiplyAccumulate(recurrent_to_gate_weights, output_state,
                                        accumulation_buffer, output, n_cell,
                                        n_output, n_batch, context);
  }
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_gate_weights, n_cell, cell_state, n_batch, output);
  }
  if (use_layer_norm) {
    tensor_utils::MeanStddevNormalization(output, output, n_cell, n_batch);
    tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell,
                                                output, n_batch, output);
    tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, output);
  }
  tensor_utils::ApplyActivationToVector(output, n_batch * n_cell, activation,
                                        gate);
}
void UpdateLstmCellFloat(int n_batch, int n_cell, float* cell_state,
                         const float* input_gate, float* forget_gate,
                         const float* cell_gate, bool use_cifg, float clip) {
  tensor_utils::VectorVectorCwiseProduct(forget_gate, cell_state,
                                         n_batch * n_cell, cell_state);
  if (use_cifg) {
    float* scratch = forget_gate;
    tensor_utils::Sub1Vector(forget_gate, n_batch * n_cell, scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_gate, scratch, n_batch * n_cell, cell_state);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_gate, input_gate, n_batch * n_cell, cell_state);
  }
  if (clip > 0.0f) {
    tensor_utils::CwiseClipping(cell_state, n_batch * n_cell, clip);
  }
}
void CalculateLstmOutputFloat(int n_batch, int n_cell, int n_output,
                              const float* cell_state, const float* output_gate,
                              TfLiteFusedActivation activation,
                              const float* projection_weights,
                              const float* projection_bias,
                              const float proj_clip, float* output_state,
                              float* scratch, float* projection_bias_scratch,
                              CpuBackendContext* context) {
  tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell,
                                        activation, scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate, scratch, n_batch * n_cell,
                                         scratch);
  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);
  if (use_projection) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                                            projection_bias_scratch);
    } else {
      std::fill_n(projection_bias_scratch, n_batch * n_output, 0.0f);
    }
    MatrixBatchVectorMultiplyAccumulate(projection_weights, scratch,
                                        projection_bias_scratch, output_state,
                                        n_output, n_cell, n_batch, context);
    if (proj_clip > 0.0f) {
      tensor_utils::CwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  } else {
    std::copy_n(scratch, n_batch * n_output, output_state);
  }
}
void CalculateLstmGateHybrid(
    const int8_t* input, const float* input_sf, const int32_t* input_zp,
    const int8_t* input_to_gate_weights,
    const uint8_t* input_to_gate_weights_ledger,
    const float input_to_gate_weights_scale, int32_t* input_to_gate_row_sums,
    const int8_t* aux_input, const float* aux_input_sf,
    const int32_t* aux_input_zp, const int8_t* aux_input_to_gate_weights,
    const float aux_input_to_gate_weights_scale,
    int32_t* aux_input_to_gate_row_sums,
    const int8_t* output_state, const float* output_state_float,
    const float* output_state_sf, const int32_t* output_state_zp,
    const int8_t* recurrent_to_gate_weights,
    const float* recurrent_to_gate_diag,
    const uint8_t* recurrent_to_gate_weights_ledger,
    const float recurrent_to_gate_weights_scale,
    int32_t* recurrent_to_gate_row_sums,
    const float* cell_state, const int8_t* cell_to_gate_weights,
    const float cell_to_gate_weights_scale,
    const float* layer_norm_coefficients, const float* gate_bias,
    const int n_batch, const int n_input, const int n_aux_input,
    const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    float* gate,
    const bool is_input_all_zeros, const bool is_aux_input_all_zeros,
    const bool is_output_state_all_zeros, bool* compute_row_sums,
    CpuBackendContext* context,
    float* scratch0,         
    float* scratch1,         
    int32_t* accum_scratch,  
    bool recurrent_is_diag) {
  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);
  if (use_layer_norm) {
    std::fill_n(gate, n_cell * n_batch, 0.0f);
  } else {
    tensor_utils::VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  }
  if (!is_input_all_zeros) {
    if (input_to_gate_weights_ledger != nullptr) {
      std::vector<float> scales(n_batch);
      for (int i = 0; i < n_batch; i++) {
        scales[i] = input_to_gate_weights_scale * input_sf[i];
      }
      tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
          input_to_gate_weights, input_to_gate_weights_ledger, n_cell, n_input,
          input, scales.data(), n_batch, gate);
    } else {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_to_gate_weights, n_cell, n_input, input,
          input_to_gate_weights_scale, input_sf, n_batch, gate,
          nullptr, input_zp, accum_scratch,
          input_to_gate_row_sums, compute_row_sums, scratch0, context);
    }
  }
  if (!is_aux_input_all_zeros) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_gate_weights, n_cell, n_aux_input, aux_input,
        aux_input_to_gate_weights_scale, aux_input_sf, n_batch, gate,
        nullptr, aux_input_zp, accum_scratch,
        aux_input_to_gate_row_sums, compute_row_sums, scratch0, context);
  }
  if (!is_output_state_all_zeros) {
    if (recurrent_to_gate_weights_ledger != nullptr) {
      std::vector<float> scales(n_batch);
      for (int i = 0; i < n_batch; i++) {
        scales[i] = recurrent_to_gate_weights_scale * input_sf[i];
      }
      tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
          recurrent_to_gate_weights, recurrent_to_gate_weights_ledger, n_cell,
          n_output, output_state, scales.data(), n_batch, gate);
    } else {
      if (recurrent_is_diag) {
        tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(
            recurrent_to_gate_diag, n_cell, output_state_float, n_batch, gate);
      } else {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            recurrent_to_gate_weights, n_cell, n_output, output_state,
            recurrent_to_gate_weights_scale, output_state_sf, n_batch, gate,
            nullptr, output_state_zp, accum_scratch,
            recurrent_to_gate_row_sums, compute_row_sums, scratch0, context);
      }
    }
  }
  if (use_peephole) {
    float* recovered_cell_weights = scratch1;
    tensor_utils::VectorScalarMultiply(cell_to_gate_weights, n_cell,
                                       cell_to_gate_weights_scale,
                                       recovered_cell_weights);
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        recovered_cell_weights, n_cell, cell_state, n_batch, gate);
  }
  if (use_layer_norm) {
    tensor_utils::MeanStddevNormalization(gate, gate, n_cell, n_batch);
    tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell,
                                                gate, n_batch, gate);
    tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  tensor_utils::ApplyActivationToVector(gate, n_cell * n_batch, activation,
                                        gate);
}
void CalculateLstmOutputHybrid(
    int n_batch, int n_cell, int n_output, const float* cell_state,
    const float* output_gate, TfLiteFusedActivation activation,
    const int8_t* projection_weights, const uint8_t* projection_weights_ledger,
    float projection_weights_scale, const float* projection_bias,
    const float proj_clip, float* output_state, bool asymmetric_quantize_inputs,
    int32_t* projection_weights_row_sums, bool* compute_row_sums,
    CpuBackendContext* context, float* scratch0, int8_t* scratch1,
    float* scratch2, int32_t* scratch3, int32_t* scratch4) {
  tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell,
                                        activation, scratch0);
  tensor_utils::VectorVectorCwiseProduct(output_gate, scratch0,
                                         n_batch * n_cell, scratch0);
  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);
  if (use_projection) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                                            output_state);
    } else {
      std::fill_n(output_state, n_batch * n_output, 0.0f);
    }
    if (!tensor_utils::IsZeroVector(scratch0, n_batch * n_cell)) {
      tensor_utils::BatchQuantizeFloats(scratch0, n_batch, n_cell, scratch1,
                                        scratch2, scratch3,
                                        asymmetric_quantize_inputs);
      if (projection_weights_ledger != nullptr) {
        std::vector<float> scales(n_batch);
        for (int i = 0; i < n_batch; i++) {
          scales[i] = projection_weights_scale * scratch2[i];
        }
        tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
            projection_weights, projection_weights_ledger, n_output, n_cell,
            scratch1, scales.data(), n_batch, output_state);
      } else {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            projection_weights, n_output, n_cell, scratch1,
            projection_weights_scale, scratch2, n_batch, output_state,
            nullptr, scratch3, scratch4,
            projection_weights_row_sums, compute_row_sums, scratch2, context);
      }
    }
    if (proj_clip > 0.0f) {
      tensor_utils::CwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  } else {
    std::copy_n(scratch0, n_batch * n_output, output_state);
  }
}
void CalculateLstmGateInteger8x8_16(
    const int8_t* input, const int8_t* input_to_gate_weights,
    const int32_t* input_to_gate_bias, const int32_t input_to_gate_scale_a,
    const int32_t input_to_gate_scale_b,
    const int8_t* output_state, const int8_t* recurrent_to_gate_weights,
    const int32_t* recurrent_to_gate_bias,
    const int32_t recurrent_to_gate_scale_a,
    const int32_t recurrent_to_gate_scale_b,
    const int16_t* cell_state, const int16_t* cell_to_gate_weights,
    const int32_t cell_to_gate_scale_a, const int32_t cell_to_gate_scale_b,
    const int16_t* layer_norm_coefficients, const int32_t* layer_norm_bias,
    const int32_t layer_norm_input_scale_a,
    const int32_t layer_norm_input_scale_b,
    const int32_t layer_norm_variance_guard,
    const int n_batch, const int n_input, const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    int16_t* gate,
    CpuBackendContext* context,
    int32_t* scratch5) {
  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);
  std::fill_n(gate, n_batch * n_cell, 0);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input, input_to_gate_bias, input_to_gate_weights, input_to_gate_scale_a,
      input_to_gate_scale_b, n_batch, n_input, n_cell, 0, scratch5, gate,
      context);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      output_state, recurrent_to_gate_bias, recurrent_to_gate_weights,
      recurrent_to_gate_scale_a, recurrent_to_gate_scale_b, n_batch, n_output,
      n_cell, 0, scratch5, gate, context);
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_gate_weights, n_output, cell_state, n_batch,
        cell_to_gate_scale_a, cell_to_gate_scale_b, gate);
  }
  if (use_layer_norm) {
    tensor_utils::ApplyLayerNorm(
        gate, layer_norm_coefficients, layer_norm_bias,
        layer_norm_input_scale_a, layer_norm_input_scale_b,
        layer_norm_variance_guard, n_batch, n_cell, gate);
  }
  switch (activation) {
    case kTfLiteActSigmoid:
      tensor_utils::ApplySigmoid(gate, n_batch, n_cell, gate);
      break;
    case kTfLiteActTanh:
      tensor_utils::ApplyTanh(3, gate, n_batch, n_cell, gate);
      break;
    default:
      TFLITE_ASSERT_FALSE;
  }
}
void UpdateLstmCellInteger(int n_batch, int n_cell, int16_t* cell_state,
                           int32_t cell_state_scale, const int16_t* input_gate,
                           int16_t* forget_gate, const int16_t* cell_gate,
                           bool use_cifg, int16_t clip) {
  int16_t* scratch = forget_gate;
  tensor_utils::CwiseMul(forget_gate, cell_state, n_batch, n_cell, 15,
                         cell_state);
  if (use_cifg) {
    tensor_utils::Sub1Vector(forget_gate, n_batch * n_cell, scratch);
    tensor_utils::CwiseMul(scratch, cell_gate, n_batch, n_cell,
                           30 + cell_state_scale, scratch);
  } else {
    tensor_utils::CwiseMul(input_gate, cell_gate, n_batch, n_cell,
                           30 + cell_state_scale, scratch);
  }
  tensor_utils::CwiseAdd(cell_state, scratch, n_batch, n_cell, cell_state);
  if (clip > 0) {
    tensor_utils::CwiseClipping(cell_state, n_batch * n_cell, clip);
  }
}
void CalculateLstmOutputInteger8x8_16(
    int n_batch, int n_cell, int n_output, const int16_t* cell_state,
    int32_t cell_state_scale, const int16_t* output_gate,
    int32_t hidden_scale_a, int32_t hidden_scale_b, int32_t hidden_zp,
    const int8_t* projection_weights, int32_t proj_scale_a,
    int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int8_t quantized_proj_clip, int8_t* output_state,
    CpuBackendContext* context, int16_t* scratch0, int8_t* scratch1,
    int32_t* scratch2) {
  tensor_utils::ApplyTanh(15 + cell_state_scale, cell_state, n_batch, n_cell,
                          scratch0);
  const bool use_projection = (projection_weights != nullptr);
  if (use_projection) {
    tensor_utils::CwiseMul(output_gate, scratch0, hidden_scale_a,
                           hidden_scale_b, n_batch, n_cell, -hidden_zp,
                           scratch1);
    std::fill_n(output_state, n_batch * n_output, 0);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        scratch1, projection_bias, projection_weights, proj_scale_a,
        proj_scale_b, n_batch, n_cell, n_output, output_state_zp, scratch2,
        output_state, context);
    if (quantized_proj_clip > 0) {
      tensor_utils::CwiseClipping(output_state, n_batch * n_output,
                                  quantized_proj_clip);
    }
  } else {
    tensor_utils::CwiseMul(output_gate, scratch0, hidden_scale_a,
                           hidden_scale_b, n_batch, n_cell, hidden_zp,
                           output_state);
  }
}
void CalculateLstmGateInteger8x8_8(
    const int8_t* input, int32_t input_zp, const int8_t* input_to_gate_weight,
    const int32_t input_to_gate_scale_a, const int32_t input_to_gate_scale_b,
    const int32_t input_times_weights_scale_a,
    const int32_t input_times_weights_scale_b,
    const int32_t input_times_weights_zp,
    const int8_t* output_state, const int32_t output_state_zp,
    const int8_t* recurrent_to_gate_weight,
    const int32_t recurrent_to_gate_scale_a,
    const int32_t recurrent_to_gate_scale_b,
    const int32_t output_state_times_weights_scale_a,
    const int32_t output_state_times_weights_scale_b,
    const int32_t output_state_times_weights_zp,
    const int16_t* layer_norm_gate_weight,
    const int32_t layer_norm_gate_scale_a,
    const int32_t layer_norm_gate_scale_b, const int32_t* gate_bias,
    const int n_batch, const int n_input, const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    int16_t* gate,
    int8_t* scratch0, int8_t* scratch1) {
  tensor_utils::MatrixBatchVectorMultiply(
      input, input_zp, input_to_gate_weight, input_to_gate_scale_a,
      input_to_gate_scale_b, n_batch, n_input, n_cell, scratch0,
      input_times_weights_zp);
  tensor_utils::MatrixBatchVectorMultiply(
      output_state, output_state_zp, recurrent_to_gate_weight,
      recurrent_to_gate_scale_a, recurrent_to_gate_scale_b, n_batch, n_output,
      n_cell, scratch1, output_state_times_weights_zp);
  tensor_utils::TwoGateSaturatingAdd(
      scratch0, input_times_weights_zp, scratch1, output_state_times_weights_zp,
      input_times_weights_scale_a, input_times_weights_scale_b,
      output_state_times_weights_scale_a, output_state_times_weights_scale_b,
      n_batch, n_cell, gate);
  tensor_utils::ApplyLayerNormFloat(
      gate, layer_norm_gate_weight, layer_norm_gate_scale_a,
      layer_norm_gate_scale_b, gate_bias, n_batch, n_cell, gate);
  switch (activation) {
    case kTfLiteActSigmoid:
      tensor_utils::ApplySigmoidFloat(gate, n_batch, n_cell, gate);
      break;
    case kTfLiteActTanh:
      tensor_utils::ApplyTanhFloat(gate, n_batch, n_cell, -12, gate);
      break;
    default:
      TFLITE_ASSERT_FALSE;
  }
}
void CalculateLstmOutputInteger8x8_8(
    int n_batch, int n_cell, int n_output, const int16_t* cell_state,
    const int16_t* output_gate, const int8_t* projection_weights,
    int32_t proj_scale_a, int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int32_t quantized_proj_clip, int8_t* output_state,
    int16_t* scratch) {
  tensor_utils::ApplyTanhFloat(cell_state, n_batch, n_cell, -15, scratch);
  tensor_utils::CwiseMul(output_gate, scratch, n_batch, n_cell, 15 + 15 - 15,
                         scratch);
  tensor_utils::MatrixBatchVectorMultiply(
      scratch, projection_weights, proj_scale_a, proj_scale_b, projection_bias,
      n_batch, n_cell, n_output, output_state_zp, output_state);
  if (quantized_proj_clip > 0) {
    tensor_utils::CwiseClipping(output_state, n_batch * n_output,
                                quantized_proj_clip);
  }
}
inline void LstmStepFloat(
    const float* input_ptr, const float* input_to_input_weights_ptr,
    const float* input_to_forget_weights_ptr,
    const float* input_to_cell_weights_ptr,
    const float* input_to_output_weights_ptr, const float* aux_input_ptr,
    const float* aux_input_to_input_weights_ptr,
    const float* aux_input_to_forget_weights_ptr,
    const float* aux_input_to_cell_weights_ptr,
    const float* aux_input_to_output_weights_ptr,
    const float* recurrent_to_input_weights_ptr,
    const float* recurrent_to_forget_weights_ptr,
    const float* recurrent_to_cell_weights_ptr,
    const float* recurrent_to_output_weights_ptr,
    const float* cell_to_input_weights_ptr,
    const float* cell_to_forget_weights_ptr,
    const float* cell_to_output_weights_ptr,
    const float* input_layer_norm_coefficients_ptr,
    const float* forget_layer_norm_coefficients_ptr,
    const float* cell_layer_norm_coefficients_ptr,
    const float* output_layer_norm_coefficients_ptr,
    const float* input_gate_bias_ptr, const float* forget_gate_bias_ptr,
    const float* cell_gate_bias_ptr, const float* output_gate_bias_ptr,
    const float* projection_weights_ptr, const float* projection_bias_ptr,
    const TfLiteLSTMParams* params, int n_batch, int n_cell, int n_input,
    int n_aux_input, int n_output, int output_batch_leading_dim,
    float* output_state_ptr, float* cell_state_ptr, float* scratch0,
    float* scratch1, float* scratch2, float* scratch3, float* scratch4,
    float* output_ptr, bool recurrent_to_input_is_diag,
    bool recurrent_to_forget_is_diag, bool recurrent_to_cell_is_diag,
    bool recurrent_to_output_is_diag, CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("LstmStepFloat");
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  float* input_gate_scratch = scratch0;
  float* forget_gate_scratch = scratch1;
  float* cell_gate_scratch = scratch2;
  float* output_gate_scratch = scratch3;
  float* accumulation_scratch_buffer = scratch4;
  const bool is_input_all_zeros =
      tensor_utils::IsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
      (aux_input_ptr == nullptr ||
       tensor_utils::IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  if (!use_cifg) {
    CalculateLstmGateFloat(
        input_ptr, input_to_input_weights_ptr, aux_input_ptr,
        aux_input_to_input_weights_ptr, output_state_ptr,
        recurrent_to_input_weights_ptr,
        cell_state_ptr, cell_to_input_weights_ptr,
        input_layer_norm_coefficients_ptr, input_gate_bias_ptr, n_batch,
        n_input, n_aux_input, n_output, n_cell,
        kTfLiteActSigmoid, input_gate_scratch,
        is_input_all_zeros, is_aux_input_all_zeros, accumulation_scratch_buffer,
        recurrent_to_input_is_diag, context);
  }
  CalculateLstmGateFloat(
      input_ptr, input_to_forget_weights_ptr, aux_input_ptr,
      aux_input_to_forget_weights_ptr, output_state_ptr,
      recurrent_to_forget_weights_ptr,
      cell_state_ptr, cell_to_forget_weights_ptr,
      forget_layer_norm_coefficients_ptr, forget_gate_bias_ptr, n_batch,
      n_input, n_aux_input, n_output, n_cell,
      kTfLiteActSigmoid, forget_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, accumulation_scratch_buffer,
      recurrent_to_forget_is_diag, context);
  CalculateLstmGateFloat(
      input_ptr, input_to_cell_weights_ptr, aux_input_ptr,
      aux_input_to_cell_weights_ptr, output_state_ptr,
      recurrent_to_cell_weights_ptr,
      nullptr,
      nullptr, cell_layer_norm_coefficients_ptr,
      cell_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      params->activation, cell_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, accumulation_scratch_buffer,
      recurrent_to_cell_is_diag, context);
  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch,
                      forget_gate_scratch, cell_gate_scratch, use_cifg,
                      params->cell_clip);
  CalculateLstmGateFloat(
      input_ptr, input_to_output_weights_ptr, aux_input_ptr,
      aux_input_to_output_weights_ptr, output_state_ptr,
      recurrent_to_output_weights_ptr,
      cell_state_ptr, cell_to_output_weights_ptr,
      output_layer_norm_coefficients_ptr, output_gate_bias_ptr, n_batch,
      n_input, n_aux_input, n_output, n_cell,
      kTfLiteActSigmoid, output_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, accumulation_scratch_buffer,
      recurrent_to_output_is_diag, context);
  CalculateLstmOutputFloat(n_batch, n_cell, n_output, cell_state_ptr,
                           output_gate_scratch, params->activation,
                           projection_weights_ptr, projection_bias_ptr,
                           params->proj_clip, output_state_ptr, scratch2,
                           accumulation_scratch_buffer, context);
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}
inline void LstmStepHybrid(
    const float* input_ptr, const int8_t* input_to_input_weights_ptr,
    const uint8_t* input_to_input_weights_ledger_ptr,
    float input_to_input_weights_scale,
    const int8_t* input_to_forget_weights_ptr,
    const uint8_t* input_to_forget_weights_ledger_ptr,
    float input_to_forget_weights_scale,
    const int8_t* input_to_cell_weights_ptr,
    const uint8_t* input_to_cell_weights_ledger_ptr,
    float input_to_cell_weights_scale,
    const int8_t* input_to_output_weights_ptr,
    const uint8_t* input_to_output_weights_ledger_ptr,
    float input_to_output_weights_scale, const float* aux_input_ptr,
    const int8_t* aux_input_to_input_weights_ptr,
    float aux_input_to_input_weights_scale,
    const int8_t* aux_input_to_forget_weights_ptr,
    float aux_input_to_forget_weights_scale,
    const int8_t* aux_input_to_cell_weights_ptr,
    float aux_input_to_cell_weights_scale,
    const int8_t* aux_input_to_output_weights_ptr,
    float aux_input_to_output_weights_scale,
    const int8_t* recurrent_to_input_weights_ptr,
    const float* recurrent_to_input_diag,
    const uint8_t* recurrent_to_input_weights_ledger_ptr,
    float recurrent_to_input_weights_scale,
    const int8_t* recurrent_to_forget_weights_ptr,
    const float* recurrent_to_forget_diag,
    const uint8_t* recurrent_to_forget_weights_ledger_ptr,
    float recurrent_to_forget_weights_scale,
    const int8_t* recurrent_to_cell_weights_ptr,
    const float* recurrent_to_cell_diag,
    const uint8_t* recurrent_to_cell_weights_ledger_ptr,
    float recurrent_to_cell_weights_scale,
    const int8_t* recurrent_to_output_weights_ptr,
    const float* recurrent_to_output_diag,
    const uint8_t* recurrent_to_output_weights_ledger_ptr,
    float recurrent_to_output_weights_scale,
    const int8_t* cell_to_input_weights_ptr, float cell_to_input_weights_scale,
    const int8_t* cell_to_forget_weights_ptr,
    float cell_to_forget_weights_scale,
    const int8_t* cell_to_output_weights_ptr,
    float cell_to_output_weights_scale,
    const float* input_layer_norm_coefficients_ptr,
    const float* forget_layer_norm_coefficients_ptr,
    const float* cell_layer_norm_coefficients_ptr,
    const float* output_layer_norm_coefficients_ptr,
    const float* input_gate_bias_ptr, const float* forget_gate_bias_ptr,
    const float* cell_gate_bias_ptr, const float* output_gate_bias_ptr,
    const int8_t* projection_weights_ptr,
    const uint8_t* projection_weights_ledger_ptr,
    float projection_weights_scale, const float* projection_bias_ptr,
    const TfLiteLSTMParams* params, int n_batch, int n_cell, int n_input,
    int n_aux_input, int n_output, int output_batch_leading_dim,
    float* scratch0, float* scratch1, float* scratch2, float* scratch3,
    float* input_sf, float* aux_input_sf, float* output_state_sf,
    float* scaling_factors_scratch, float* recovered_cell_weights,
    int8_t* quantized_input_ptr, int8_t* quantized_aux_input_ptr,
    int8_t* quantized_output_state_ptr, int8_t* quantized_output_scratch,
    float* output_state_ptr, float* cell_state_ptr, int32_t* accum_scratch_ptr,
    float* output_ptr, int32_t* input_zp, int32_t* aux_input_zp,
    int32_t* output_state_zp, int32_t* row_sums, int row_sums_size,
    bool* compute_row_sums, bool asymmetric_quantize_inputs,
    bool recurrent_to_input_is_diag, bool recurrent_to_forget_is_diag,
    bool recurrent_to_cell_is_diag, bool recurrent_to_output_is_diag,
    CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("LstmStepHybrid");
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  float* input_gate_scratch = scratch0;
  float* forget_gate_scratch = scratch1;
  float* cell_gate_scratch = scratch2;
  float* output_gate_scratch = scratch3;
  int32_t* input_to_input_row_sums = nullptr;
  int32_t* input_to_forget_row_sums = nullptr;
  int32_t* input_to_cell_row_sums = nullptr;
  int32_t* input_to_output_row_sums = nullptr;
  int32_t* aux_input_to_input_row_sums = nullptr;
  int32_t* aux_input_to_forget_row_sums = nullptr;
  int32_t* aux_input_to_cell_row_sums = nullptr;
  int32_t* aux_input_to_output_row_sums = nullptr;
  int32_t* recurrent_to_input_row_sums = nullptr;
  int32_t* recurrent_to_forget_row_sums = nullptr;
  int32_t* recurrent_to_cell_row_sums = nullptr;
  int32_t* recurrent_to_output_row_sums = nullptr;
  int32_t* projection_weights_row_sums = nullptr;
  if (asymmetric_quantize_inputs) {
    int num_row_sums = use_cifg ? 6 : 8;
    if (aux_input_ptr != nullptr) {
      num_row_sums += use_cifg ? 3 : 4;
    }
    if (projection_weights_ptr != nullptr) {
      num_row_sums += ceil(static_cast<float>(n_output) / n_cell);
    }
    TF_LITE_ASSERT(row_sums_size == num_row_sums);
    input_to_input_row_sums = row_sums;
    input_to_forget_row_sums =
        use_cifg ? input_to_input_row_sums : input_to_input_row_sums + n_cell;
    input_to_cell_row_sums = input_to_forget_row_sums + n_cell;
    input_to_output_row_sums = input_to_cell_row_sums + n_cell;
    if (aux_input_ptr != nullptr) {
      aux_input_to_input_row_sums = input_to_output_row_sums + n_cell;
      aux_input_to_forget_row_sums = use_cifg
                                         ? aux_input_to_input_row_sums
                                         : aux_input_to_input_row_sums + n_cell;
      aux_input_to_cell_row_sums = aux_input_to_forget_row_sums + n_cell;
      aux_input_to_output_row_sums = aux_input_to_cell_row_sums + n_cell;
    }
    recurrent_to_input_row_sums = aux_input_ptr
                                      ? aux_input_to_output_row_sums + n_cell
                                      : input_to_output_row_sums + n_cell;
    recurrent_to_forget_row_sums = use_cifg
                                       ? recurrent_to_input_row_sums
                                       : recurrent_to_input_row_sums + n_cell;
    recurrent_to_cell_row_sums = recurrent_to_forget_row_sums + n_cell;
    recurrent_to_output_row_sums = recurrent_to_cell_row_sums + n_cell;
    if (projection_weights_ptr != nullptr) {
      projection_weights_row_sums = recurrent_to_output_row_sums + n_cell;
    }
    if (*compute_row_sums) {
      ComputeRowSums(
          input_to_input_row_sums, input_to_forget_row_sums,
          input_to_cell_row_sums, input_to_output_row_sums,
          aux_input_to_input_row_sums, aux_input_to_forget_row_sums,
          aux_input_to_cell_row_sums, aux_input_to_output_row_sums,
          recurrent_to_input_row_sums, recurrent_to_forget_row_sums,
          recurrent_to_cell_row_sums, recurrent_to_output_row_sums,
          projection_weights_row_sums, row_sums, n_cell, n_input, n_aux_input,
          n_output, input_to_input_weights_ptr, input_to_forget_weights_ptr,
          input_to_cell_weights_ptr, input_to_output_weights_ptr,
          aux_input_to_input_weights_ptr, aux_input_to_forget_weights_ptr,
          aux_input_to_cell_weights_ptr, aux_input_to_output_weights_ptr,
          recurrent_to_input_weights_ptr, recurrent_to_forget_weights_ptr,
          recurrent_to_cell_weights_ptr, recurrent_to_output_weights_ptr,
          projection_weights_ptr, use_cifg, aux_input_ptr,
          recurrent_to_input_is_diag, recurrent_to_forget_is_diag,
          recurrent_to_cell_is_diag, recurrent_to_output_is_diag);
      *compute_row_sums = false;
    }
  }
  const bool is_input_all_zeros =
      tensor_utils::IsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
      (aux_input_ptr == nullptr ||
       tensor_utils::IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  const bool is_output_state_all_zeros =
      tensor_utils::IsZeroVector(output_state_ptr, n_batch * n_output);
  if (!is_input_all_zeros) {
    tensor_utils::BatchQuantizeFloats(input_ptr, n_batch, n_input,
                                      quantized_input_ptr, input_sf, input_zp,
                                      asymmetric_quantize_inputs);
  }
  if (!is_aux_input_all_zeros) {
    tensor_utils::BatchQuantizeFloats(aux_input_ptr, n_batch, n_aux_input,
                                      quantized_aux_input_ptr, aux_input_sf,
                                      aux_input_zp, asymmetric_quantize_inputs);
  }
  if (!is_output_state_all_zeros) {
    tensor_utils::BatchQuantizeFloats(
        output_state_ptr, n_batch, n_output, quantized_output_state_ptr,
        output_state_sf, output_state_zp, asymmetric_quantize_inputs);
  }
  if (!use_cifg) {
    CalculateLstmGateHybrid(
        quantized_input_ptr, input_sf, input_zp, input_to_input_weights_ptr,
        input_to_input_weights_ledger_ptr, input_to_input_weights_scale,
        input_to_input_row_sums, quantized_aux_input_ptr, aux_input_sf,
        aux_input_zp, aux_input_to_input_weights_ptr,
        aux_input_to_input_weights_scale, aux_input_to_input_row_sums,
        quantized_output_state_ptr, output_state_ptr, output_state_sf,
        output_state_zp, recurrent_to_input_weights_ptr,
        recurrent_to_input_diag, recurrent_to_input_weights_ledger_ptr,
        recurrent_to_input_weights_scale, recurrent_to_input_row_sums,
        cell_state_ptr, cell_to_input_weights_ptr, cell_to_input_weights_scale,
        input_layer_norm_coefficients_ptr, input_gate_bias_ptr, n_batch,
        n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid,
        input_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros,
        is_output_state_all_zeros, compute_row_sums, context,
        scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr,
        recurrent_to_input_is_diag);
  }
  CalculateLstmGateHybrid(
      quantized_input_ptr, input_sf, input_zp, input_to_forget_weights_ptr,
      input_to_forget_weights_ledger_ptr, input_to_forget_weights_scale,
      input_to_forget_row_sums, quantized_aux_input_ptr, aux_input_sf,
      aux_input_zp, aux_input_to_forget_weights_ptr,
      aux_input_to_forget_weights_scale, aux_input_to_forget_row_sums,
      quantized_output_state_ptr, output_state_ptr, output_state_sf,
      output_state_zp, recurrent_to_forget_weights_ptr,
      recurrent_to_forget_diag, recurrent_to_forget_weights_ledger_ptr,
      recurrent_to_forget_weights_scale, recurrent_to_forget_row_sums,
      cell_state_ptr, cell_to_forget_weights_ptr, cell_to_forget_weights_scale,
      forget_layer_norm_coefficients_ptr, forget_gate_bias_ptr, n_batch,
      n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid,
      forget_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros,
      is_output_state_all_zeros, compute_row_sums, context,
      scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr,
      recurrent_to_forget_is_diag);
  CalculateLstmGateHybrid(
      quantized_input_ptr, input_sf, input_zp, input_to_cell_weights_ptr,
      input_to_cell_weights_ledger_ptr, input_to_cell_weights_scale,
      input_to_cell_row_sums, quantized_aux_input_ptr, aux_input_sf,
      aux_input_zp, aux_input_to_cell_weights_ptr,
      aux_input_to_cell_weights_scale, aux_input_to_cell_row_sums,
      quantized_output_state_ptr, output_state_ptr, output_state_sf,
      output_state_zp, recurrent_to_cell_weights_ptr, recurrent_to_cell_diag,
      recurrent_to_cell_weights_ledger_ptr, recurrent_to_cell_weights_scale,
      recurrent_to_cell_row_sums,
      nullptr, nullptr,
      0.0f, cell_layer_norm_coefficients_ptr,
      cell_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      params->activation, cell_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, is_output_state_all_zeros, compute_row_sums,
      context, scaling_factors_scratch, recovered_cell_weights,
      accum_scratch_ptr, recurrent_to_cell_is_diag);
  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch,
                      forget_gate_scratch, cell_gate_scratch, use_cifg,
                      params->cell_clip);
  CalculateLstmGateHybrid(
      quantized_input_ptr, input_sf, input_zp, input_to_output_weights_ptr,
      input_to_output_weights_ledger_ptr, input_to_output_weights_scale,
      input_to_output_row_sums, quantized_aux_input_ptr, aux_input_sf,
      aux_input_zp, aux_input_to_output_weights_ptr,
      aux_input_to_output_weights_scale, aux_input_to_output_row_sums,
      quantized_output_state_ptr, output_state_ptr, output_state_sf,
      output_state_zp, recurrent_to_output_weights_ptr,
      recurrent_to_output_diag, recurrent_to_output_weights_ledger_ptr,
      recurrent_to_output_weights_scale, recurrent_to_output_row_sums,
      cell_state_ptr, cell_to_output_weights_ptr, cell_to_output_weights_scale,
      output_layer_norm_coefficients_ptr, output_gate_bias_ptr, n_batch,
      n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid,
      output_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros,
      is_output_state_all_zeros, compute_row_sums, context,
      scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr,
      recurrent_to_output_is_diag);
  CalculateLstmOutputHybrid(
      n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
      params->activation, projection_weights_ptr, projection_weights_ledger_ptr,
      projection_weights_scale, projection_bias_ptr, params->proj_clip,
      output_state_ptr, asymmetric_quantize_inputs, projection_weights_row_sums,
      compute_row_sums, context, scratch2, quantized_output_scratch, input_sf,
      input_zp, accum_scratch_ptr);
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}
inline void LstmStepInteger8x8_16(
    const int8_t* input_ptr, const int8_t* input_to_input_weight_ptr,
    int32_t effective_input_to_input_scale_a,
    int32_t effective_input_to_input_scale_b,
    const int8_t* input_to_forget_weight_ptr,
    int32_t effective_input_to_forget_scale_a,
    int32_t effective_input_to_forget_scale_b,
    const int8_t* input_to_cell_weight_ptr,
    int32_t effective_input_to_cell_scale_a,
    int32_t effective_input_to_cell_scale_b,
    const int8_t* input_to_output_weight_ptr,
    int32_t effective_input_to_output_scale_a,
    int32_t effective_input_to_output_scale_b,
    const int8_t* recurrent_to_input_weight_ptr,
    int32_t effective_recurrent_to_input_scale_a,
    int32_t effective_recurrent_to_input_scale_b,
    const int8_t* recurrent_to_forget_weight_ptr,
    int32_t effective_recurrent_to_forget_scale_a,
    int32_t effective_recurrent_to_forget_scale_b,
    const int8_t* recurrent_to_cell_weight_ptr,
    int32_t effective_recurrent_to_cell_scale_a,
    int32_t effective_recurrent_to_cell_scale_b,
    const int8_t* recurrent_to_output_weight_ptr,
    int32_t effective_recurrent_to_output_scale_a,
    int32_t effective_recurrent_to_output_scale_b,
    const int16_t* cell_to_input_weight_ptr,
    int32_t effective_cell_to_input_scale_a,
    int32_t effective_cell_to_input_scale_b,
    const int16_t* cell_to_forget_weight_ptr,
    int32_t effective_cell_to_forget_scale_a,
    int32_t effective_cell_to_forget_scale_b,
    const int16_t* cell_to_output_weight_ptr,
    int32_t effective_cell_to_output_scale_a,
    int32_t effective_cell_to_output_scale_b,
    const int8_t* projection_weight_ptr, int32_t effective_proj_scale_a,
    int32_t effective_proj_scale_b, int32_t hidden_zp,
    int32_t effective_hidden_scale_a, int32_t effective_hidden_scale_b,
    const int16_t* layer_norm_input_weight_ptr,
    int32_t layer_norm_input_scale_a, int32_t layer_norm_input_scale_b,
    const int16_t* layer_norm_forget_weight_ptr,
    int32_t layer_norm_forget_scale_a, int32_t layer_norm_forget_scale_b,
    const int16_t* layer_norm_cell_weight_ptr, int32_t layer_norm_cell_scale_a,
    int32_t layer_norm_cell_scale_b,
    const int16_t* layer_norm_output_weight_ptr,
    int32_t layer_norm_output_scale_a, int32_t layer_norm_output_scale_b,
    const int32_t* input_gate_bias_ptr, const int32_t* forget_gate_bias_ptr,
    const int32_t* cell_gate_bias_ptr, const int32_t* output_gate_bias_ptr,
    int16_t quantized_cell_clip, int8_t quantized_proj_clip,
    int32_t cell_state_scale, int32_t input_variance_guard,
    int32_t forget_variance_guard, int32_t cell_variance_guard,
    int32_t output_variance_guard,
    const int32_t* input_to_forget_effective_bias,
    const int32_t* recurrent_to_forget_effective_bias,
    const int32_t* input_to_cell_effective_bias,
    const int32_t* recurrent_to_cell_effective_bias,
    const int32_t* input_to_output_effective_bias,
    const int32_t* recurrent_to_output_effective_bias,
    const int32_t* input_to_input_effective_bias,
    const int32_t* recurrent_to_input_effective_bias,
    const int32_t* projection_effective_bias, int n_batch, int n_cell,
    int n_input, int n_output, int8_t* output_state_ptr,
    int32_t output_state_zp, int16_t* cell_state_ptr, int8_t* output_ptr,
    int16_t* scratch0, int16_t* scratch1, int16_t* scratch2, int16_t* scratch3,
    int8_t* scratch4, int32_t* scratch5, CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("LstmStepInteger8x8_16");
  int16_t* input_gate_scratch = scratch0;
  int16_t* forget_gate_scratch = scratch1;
  int16_t* cell_gate_scratch = scratch2;
  int16_t* output_gate_scratch = scratch3;
  const bool use_cifg = (input_to_input_weight_ptr == nullptr);
  TFLITE_DCHECK(input_to_forget_effective_bias);
  TFLITE_DCHECK(recurrent_to_forget_effective_bias);
  TFLITE_DCHECK(input_to_cell_effective_bias);
  TFLITE_DCHECK(recurrent_to_cell_effective_bias);
  TFLITE_DCHECK(input_to_output_effective_bias);
  TFLITE_DCHECK(recurrent_to_output_effective_bias);
  if (!use_cifg) {
    TFLITE_DCHECK(input_to_input_effective_bias);
    TFLITE_DCHECK(recurrent_to_input_effective_bias);
  }
  const bool use_projection = (projection_weight_ptr != nullptr);
  if (use_projection) {
    TFLITE_DCHECK(projection_effective_bias);
  }
  if (!use_cifg) {
    CalculateLstmGateInteger8x8_16(
        input_ptr, input_to_input_weight_ptr, input_to_input_effective_bias,
        effective_input_to_input_scale_a, effective_input_to_input_scale_b,
        output_state_ptr, recurrent_to_input_weight_ptr,
        recurrent_to_input_effective_bias, effective_recurrent_to_input_scale_a,
        effective_recurrent_to_input_scale_b, cell_state_ptr,
        cell_to_input_weight_ptr, effective_cell_to_input_scale_a,
        effective_cell_to_input_scale_b, layer_norm_input_weight_ptr,
        input_gate_bias_ptr, layer_norm_input_scale_a, layer_norm_input_scale_b,
        input_variance_guard, n_batch, n_input, n_output, n_cell,
        kTfLiteActSigmoid, input_gate_scratch, context, scratch5);
  }
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_forget_weight_ptr, input_to_forget_effective_bias,
      effective_input_to_forget_scale_a, effective_input_to_forget_scale_b,
      output_state_ptr, recurrent_to_forget_weight_ptr,
      recurrent_to_forget_effective_bias, effective_recurrent_to_forget_scale_a,
      effective_recurrent_to_forget_scale_b, cell_state_ptr,
      cell_to_forget_weight_ptr, effective_cell_to_forget_scale_a,
      effective_cell_to_forget_scale_b, layer_norm_forget_weight_ptr,
      forget_gate_bias_ptr, layer_norm_forget_scale_a,
      layer_norm_forget_scale_b, forget_variance_guard, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, forget_gate_scratch, context,
      scratch5);
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_cell_weight_ptr, input_to_cell_effective_bias,
      effective_input_to_cell_scale_a, effective_input_to_cell_scale_b,
      output_state_ptr, recurrent_to_cell_weight_ptr,
      recurrent_to_cell_effective_bias, effective_recurrent_to_cell_scale_a,
      effective_recurrent_to_cell_scale_b, cell_state_ptr,
      nullptr, 0,
      0, layer_norm_cell_weight_ptr,
      cell_gate_bias_ptr, layer_norm_cell_scale_a, layer_norm_cell_scale_b,
      cell_variance_guard, n_batch, n_input, n_output, n_cell, kTfLiteActTanh,
      cell_gate_scratch, context, scratch5);
  UpdateLstmCellInteger(n_batch, n_cell, cell_state_ptr, cell_state_scale,
                        input_gate_scratch, forget_gate_scratch,
                        cell_gate_scratch, use_cifg, quantized_cell_clip);
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_output_weight_ptr, input_to_output_effective_bias,
      effective_input_to_output_scale_a, effective_input_to_output_scale_b,
      output_state_ptr, recurrent_to_output_weight_ptr,
      recurrent_to_output_effective_bias, effective_recurrent_to_output_scale_a,
      effective_recurrent_to_output_scale_b, cell_state_ptr,
      cell_to_output_weight_ptr, effective_cell_to_output_scale_a,
      effective_cell_to_output_scale_b, layer_norm_output_weight_ptr,
      output_gate_bias_ptr, layer_norm_output_scale_a,
      layer_norm_output_scale_b, output_variance_guard, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, output_gate_scratch, context,
      scratch5);
  CalculateLstmOutputInteger8x8_16(
      n_batch, n_cell, n_output, cell_state_ptr, cell_state_scale,
      output_gate_scratch, effective_hidden_scale_a, effective_hidden_scale_b,
      hidden_zp, projection_weight_ptr, effective_proj_scale_a,
      effective_proj_scale_b, projection_effective_bias, output_state_zp,
      quantized_proj_clip, output_state_ptr, context, scratch0, scratch4,
      scratch5);
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}
inline void LstmStepInteger8x8_8(
    const int8_t* input_ptr, int32_t input_zp,
    const int8_t* input_to_input_weight_ptr,
    int32_t effective_input_to_input_scale_a,
    int32_t effective_input_to_input_scale_b,
    const int8_t* input_to_forget_weight_ptr,
    int32_t effective_input_to_forget_scale_a,
    int32_t effective_input_to_forget_scale_b,
    const int8_t* input_to_cell_weight_ptr,
    int32_t effective_input_to_cell_scale_a,
    int32_t effective_input_to_cell_scale_b,
    const int8_t* input_to_output_weight_ptr,
    int32_t effective_input_to_output_scale_a,
    int32_t effective_input_to_output_scale_b,
    const int8_t* recurrent_to_input_weight_ptr,
    int32_t effective_recurrent_to_input_scale_a,
    int32_t effective_recurrent_to_input_scale_b,
    const int8_t* recurrent_to_forget_weight_ptr,
    int32_t effective_recurrent_to_forget_scale_a,
    int32_t effective_recurrent_to_forget_scale_b,
    const int8_t* recurrent_to_cell_weight_ptr,
    int32_t effective_recurrent_to_cell_scale_a,
    int32_t effective_recurrent_to_cell_scale_b,
    const int8_t* recurrent_to_output_weight_ptr,
    int32_t effective_recurrent_to_output_scale_a,
    int32_t effective_recurrent_to_output_scale_b,
    const int8_t* cell_to_input_weight_ptr,
    int32_t effective_cell_to_input_scale_a,
    int32_t effective_cell_to_input_scale_b,
    const int8_t* cell_to_forget_weight_ptr,
    int32_t effective_cell_to_forget_scale_a,
    int32_t effective_cell_to_forget_scale_b,
    const int8_t* cell_to_output_weight_ptr,
    int32_t effective_cell_to_output_scale_a,
    int32_t effective_cell_to_output_scale_b,
    const int8_t* projection_weight_ptr, int32_t effective_proj_scale_a,
    int32_t effective_proj_scale_b, const int16_t* layer_norm_input_weight_ptr,
    int32_t layer_norm_input_scale_a, int32_t layer_norm_input_scale_b,
    const int16_t* layer_norm_forget_weight_ptr,
    int32_t layer_norm_forget_scale_a, int32_t layer_norm_forget_scale_b,
    const int16_t* layer_norm_cell_weight_ptr, int32_t layer_norm_cell_scale_a,
    int32_t layer_norm_cell_scale_b,
    const int16_t* layer_norm_output_weight_ptr,
    int32_t layer_norm_output_scale_a, int32_t layer_norm_output_scale_b,
    const int32_t* input_gate_bias_ptr, const int32_t* forget_gate_bias_ptr,
    const int32_t* cell_gate_bias_ptr, const int32_t* output_gate_bias_ptr,
    const int32_t* projection_bias_ptr, const TfLiteLSTMParams* params,
    const int32_t* intermediate_scale_a, const int32_t* intermediate_scale_b,
    const int32_t* intermediate_zp, int16_t quantized_cell_clip,
    int8_t quantized_proj_clip, int n_batch, int n_cell, int n_input,
    int n_output, int output_batch_leading_dim, int8_t* output_state_ptr,
    int32_t output_state_zp, int16_t* cell_state_ptr, int8_t* output_ptr,
    int8_t* scratch0, int8_t* scratch1, int16_t* scratch2, int16_t* scratch3,
    int16_t* scratch4, int16_t* scratch5, int16_t* scratch6,
    int16_t* scratch7) {
  ruy::profiler::ScopeLabel label("LstmStepInteger8x8_8");
  int16_t* forget_gate_scratch = scratch2;
  int16_t* cell_gate_scratch = scratch3;
  int16_t* output_gate_scratch = scratch4;
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_forget_weight_ptr,
      effective_input_to_forget_scale_a, effective_input_to_forget_scale_b,
      intermediate_scale_a[2], intermediate_scale_b[2], intermediate_zp[4],
      output_state_ptr, output_state_zp, recurrent_to_forget_weight_ptr,
      effective_recurrent_to_forget_scale_a,
      effective_recurrent_to_forget_scale_b, intermediate_scale_a[3],
      intermediate_scale_b[3], intermediate_zp[5], layer_norm_forget_weight_ptr,
      layer_norm_forget_scale_a, layer_norm_forget_scale_b,
      forget_gate_bias_ptr, n_batch, n_input, n_output, n_cell,
      kTfLiteActSigmoid, forget_gate_scratch, scratch0, scratch1);
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_cell_weight_ptr,
      effective_input_to_cell_scale_a, effective_input_to_cell_scale_b,
      intermediate_scale_a[4], intermediate_scale_b[4], intermediate_zp[7],
      output_state_ptr, output_state_zp, recurrent_to_cell_weight_ptr,
      effective_recurrent_to_cell_scale_a, effective_recurrent_to_cell_scale_b,
      intermediate_scale_a[5], intermediate_scale_b[5], intermediate_zp[8],
      layer_norm_cell_weight_ptr, layer_norm_cell_scale_a,
      layer_norm_cell_scale_b, cell_gate_bias_ptr, n_batch, n_input, n_output,
      n_cell, kTfLiteActTanh, cell_gate_scratch, scratch0, scratch1);
  UpdateLstmCellInteger(n_batch, n_cell, cell_state_ptr,
                        -15, nullptr,
                        forget_gate_scratch, cell_gate_scratch,
                        true, quantized_cell_clip);
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_output_weight_ptr,
      effective_input_to_output_scale_a, effective_input_to_output_scale_b,
      intermediate_scale_a[6], intermediate_scale_b[6], intermediate_zp[10],
      output_state_ptr, output_state_zp, recurrent_to_output_weight_ptr,
      effective_recurrent_to_output_scale_a,
      effective_recurrent_to_output_scale_b, intermediate_scale_a[11],
      intermediate_scale_b[7], intermediate_zp[7], layer_norm_output_weight_ptr,
      layer_norm_output_scale_a, layer_norm_output_scale_b,
      output_gate_bias_ptr, n_batch, n_input, n_output, n_cell,
      kTfLiteActSigmoid, output_gate_scratch, scratch0, scratch1);
  CalculateLstmOutputInteger8x8_8(
      n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
      projection_weight_ptr, effective_proj_scale_a, effective_proj_scale_b,
      projection_bias_ptr, output_state_zp, quantized_proj_clip,
      output_state_ptr, scratch2);
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}
}  
TfLiteStatus EvalFloat(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    int output_offset, TfLiteTensor* scratch_buffer, TfLiteTensor* output_state,
    TfLiteTensor* cell_state, TfLiteTensor* output,
    bool recurrent_to_input_is_diag, bool recurrent_to_forget_is_diag,
    bool recurrent_to_cell_is_diag, bool recurrent_to_output_is_diag,
    CpuBackendContext* context) {
  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  int max_time, n_batch;
  if (input->dims->size == 3) {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  } else {
    max_time = 1;
    n_batch = input->dims->data[0];
  }
  const int n_input = input->dims->data[input->dims->size - 1];
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_is_diag
                           ? recurrent_to_output_weights->dims->data[0]
                           : recurrent_to_output_weights->dims->data[1];
  const bool use_cifg = (input_to_input_weights == nullptr);
  float* scratch_buffer_ptr = GetTensorData<float>(scratch_buffer);
  float* input_gate_scratch = nullptr;
  float* cell_gate_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  float* accumulation_scratch_buffer = nullptr;
  if (use_cifg) {
    cell_gate_scratch = scratch_buffer_ptr;
    forget_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
    accumulation_scratch_buffer = scratch_buffer_ptr + 3 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer_ptr;
    cell_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 3 * n_cell * n_batch;
    accumulation_scratch_buffer = scratch_buffer_ptr + 4 * n_cell * n_batch;
  }
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  if (time_major) {
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      const int t_rel = forward_sequence ? t : max_time - t - 1;
      const float* input_ptr = GetTensorData<float>(input) + t_rel * input_step;
      const float* aux_input_ptr = nullptr;
      if (aux_input) {
        aux_input_ptr = GetTensorData<float>(aux_input) + t_rel * input_step;
      }
      float* output_ptr =
          GetTensorData<float>(output) + t_rel * output_step + output_offset;
      LstmStepFloat(
          input_ptr, GetTensorData<float>(input_to_input_weights),
          GetTensorData<float>(input_to_forget_weights),
          GetTensorData<float>(input_to_cell_weights),
          GetTensorData<float>(input_to_output_weights), aux_input_ptr,
          GetTensorData<float>(aux_input_to_input_weights),
          GetTensorData<float>(aux_input_to_forget_weights),
          GetTensorData<float>(aux_input_to_cell_weights),
          GetTensorData<float>(aux_input_to_output_weights),
          GetTensorData<float>(recurrent_to_input_weights),
          GetTensorData<float>(recurrent_to_forget_weights),
          GetTensorData<float>(recurrent_to_cell_weights),
          GetTensorData<float>(recurrent_to_output_weights),
          GetTensorData<float>(cell_to_input_weights),
          GetTensorData<float>(cell_to_forget_weights),
          GetTensorData<float>(cell_to_output_weights),
          GetTensorData<float>(input_layer_norm_coefficients),
          GetTensorData<float>(forget_layer_norm_coefficients),
          GetTensorData<float>(cell_layer_norm_coefficients),
          GetTensorData<float>(output_layer_norm_coefficients),
          GetTensorData<float>(input_gate_bias),
          GetTensorData<float>(forget_gate_bias),
          GetTensorData<float>(cell_gate_bias),
          GetTensorData<float>(output_gate_bias),
          GetTensorData<float>(projection_weights),
          GetTensorData<float>(projection_bias), params, n_batch, n_cell,
          n_input, aux_input_size, n_output, output_batch_leading_dim,
          GetTensorData<float>(output_state), GetTensorData<float>(cell_state),
          input_gate_scratch, forget_gate_scratch, cell_gate_scratch,
          output_gate_scratch, accumulation_scratch_buffer, output_ptr,
          recurrent_to_input_is_diag, recurrent_to_forget_is_diag,
          recurrent_to_cell_is_diag, recurrent_to_output_is_diag, context);
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const float* input_ptr =
            GetTensorData<float>(input) + time_offset * input_step;
        const float* aux_input_ptr = nullptr;
        if (aux_input) {
          aux_input_ptr =
              GetTensorData<float>(aux_input) + time_offset * input_step;
        }
        float* output_ptr = GetTensorData<float>(output) +
                            time_offset * output_step + output_offset;
        float* output_state_ptr =
            GetTensorData<float>(output_state) + b * output_batch_leading_dim;
        float* cell_state_ptr = GetTensorData<float>(cell_state) + b * n_cell;
        float* input_gate_scratch_ptr =
            input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        float* forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        float* cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
        float* output_gate_scratch_ptr = output_gate_scratch + b * n_cell;
        LstmStepFloat(
            input_ptr, GetTensorData<float>(input_to_input_weights),
            GetTensorData<float>(input_to_forget_weights),
            GetTensorData<float>(input_to_cell_weights),
            GetTensorData<float>(input_to_output_weights), aux_input_ptr,
            GetTensorData<float>(aux_input_to_input_weights),
            GetTensorData<float>(aux_input_to_forget_weights),
            GetTensorData<float>(aux_input_to_cell_weights),
            GetTensorData<float>(aux_input_to_output_weights),
            GetTensorData<float>(recurrent_to_input_weights),
            GetTensorData<float>(recurrent_to_forget_weights),
            GetTensorData<float>(recurrent_to_cell_weights),
            GetTensorData<float>(recurrent_to_output_weights),
            GetTensorData<float>(cell_to_input_weights),
            GetTensorData<float>(cell_to_forget_weights),
            GetTensorData<float>(cell_to_output_weights),
            GetTensorData<float>(input_layer_norm_coefficients),
            GetTensorData<float>(forget_layer_norm_coefficients),
            GetTensorData<float>(cell_layer_norm_coefficients),
            GetTensorData<float>(output_layer_norm_coefficients),
            GetTensorData<float>(input_gate_bias),
            GetTensorData<float>(forget_gate_bias),
            GetTensorData<float>(cell_gate_bias),
            GetTensorData<float>(output_gate_bias),
            GetTensorData<float>(projection_weights),
            GetTensorData<float>(projection_bias), params, 1,
            n_cell, n_input, aux_input_size, n_output, output_batch_leading_dim,
            output_state_ptr, cell_state_ptr, input_gate_scratch_ptr,
            forget_gate_scratch_ptr, cell_gate_scratch_ptr,
            output_gate_scratch_ptr, accumulation_scratch_buffer, output_ptr,
            recurrent_to_input_is_diag, recurrent_to_forget_is_diag,
            recurrent_to_cell_is_diag, recurrent_to_output_is_diag, context);
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_input_weights_ledger,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_forget_weights_ledger,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_cell_weights_ledger,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* input_to_output_weights_ledger,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_input_weights_ledger,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_forget_weights_ledger,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_cell_weights_ledger,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* recurrent_to_output_weights_ledger,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights,
    const TfLiteTensor* projection_weights_ledger,
    const TfLiteTensor* projection_bias, const TfLiteLSTMParams* params,
    bool forward_sequence, bool time_major, int output_offset,
    TfLiteTensor* scratch_buffer, TfLiteTensor* input_sf,
    TfLiteTensor* aux_input_sf, TfLiteTensor* output_state_sf,
    TfLiteTensor* prod_scaling_factors, TfLiteTensor* recovered_cell_weights,
    TfLiteTensor* input_quantized, TfLiteTensor* aux_input_quantized,
    TfLiteTensor* output_state_quantized, TfLiteTensor* cell_state_quantized,
    TfLiteTensor* output_state, TfLiteTensor* cell_state,
    TfLiteTensor* output_scratch_buffer, TfLiteTensor* output,
    TfLiteTensor* input_zp, TfLiteTensor* aux_input_zp,
    TfLiteTensor* output_state_zp, TfLiteTensor* row_sums, int row_sums_size,
    bool* compute_row_sums, bool recurrent_to_input_is_diag,
    bool recurrent_to_forget_is_diag, bool recurrent_to_cell_is_diag,
    bool recurrent_to_output_is_diag, CpuBackendContext* context) {
  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  }
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_is_diag
                           ? recurrent_to_output_weights->dims->data[0]
                           : recurrent_to_output_weights->dims->data[1];
  const bool use_cifg = (input_to_input_weights == nullptr);
  float* scratch_buffer_ptr = GetTensorData<float>(scratch_buffer);
  float* input_gate_scratch = nullptr;
  float* cell_gate_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_gate_scratch = scratch_buffer_ptr;
    forget_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer_ptr;
    cell_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 3 * n_cell * n_batch;
  }
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  int32_t* input_zp_ptr = nullptr;
  int32_t* aux_input_zp_ptr = nullptr;
  int32_t* output_state_zp_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs) {
    input_zp_ptr = GetTensorData<int32_t>(input_zp);
    aux_input_zp_ptr = GetTensorData<int32_t>(aux_input_zp);
    output_state_zp_ptr = GetTensorData<int32_t>(output_state_zp);
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }
  if (time_major) {
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      const int t_rel = forward_sequence ? t : max_time - t - 1;
      const float* input_ptr = GetTensorData<float>(input) + t_rel * input_step;
      const float* aux_input_ptr = nullptr;
      if (aux_input) {
        aux_input_ptr = GetTensorData<float>(aux_input) + t_rel * input_step;
      }
      float* output_ptr =
          GetTensorData<float>(output) + t_rel * output_step + output_offset;
      LstmStepHybrid(
          input_ptr, GetTensorData<int8_t>(input_to_input_weights),
          GetTensorData<uint8_t>(input_to_input_weights_ledger),
          GetTensorScale(input_to_input_weights),
          GetTensorData<int8_t>(input_to_forget_weights),
          GetTensorData<uint8_t>(input_to_forget_weights_ledger),
          GetTensorScale(input_to_forget_weights),
          GetTensorData<int8_t>(input_to_cell_weights),
          GetTensorData<uint8_t>(input_to_cell_weights_ledger),
          GetTensorScale(input_to_cell_weights),
          GetTensorData<int8_t>(input_to_output_weights),
          GetTensorData<uint8_t>(input_to_output_weights_ledger),
          GetTensorScale(input_to_output_weights), aux_input_ptr,
          GetTensorData<int8_t>(aux_input_to_input_weights),
          GetTensorScale(aux_input_to_input_weights),
          GetTensorData<int8_t>(aux_input_to_forget_weights),
          GetTensorScale(aux_input_to_forget_weights),
          GetTensorData<int8_t>(aux_input_to_cell_weights),
          GetTensorScale(aux_input_to_cell_weights),
          GetTensorData<int8_t>(aux_input_to_output_weights),
          GetTensorScale(aux_input_to_output_weights),
          GetTensorData<int8_t>(recurrent_to_input_weights),
          GetTensorData<float>(recurrent_to_input_weights),
          GetTensorData<uint8_t>(recurrent_to_input_weights_ledger),
          GetTensorScale(recurrent_to_input_weights),
          GetTensorData<int8_t>(recurrent_to_forget_weights),
          GetTensorData<float>(recurrent_to_forget_weights),
          GetTensorData<uint8_t>(recurrent_to_forget_weights_ledger),
          GetTensorScale(recurrent_to_forget_weights),
          GetTensorData<int8_t>(recurrent_to_cell_weights),
          GetTensorData<float>(recurrent_to_cell_weights),
          GetTensorData<uint8_t>(recurrent_to_cell_weights_ledger),
          GetTensorScale(recurrent_to_cell_weights),
          GetTensorData<int8_t>(recurrent_to_output_weights),
          GetTensorData<float>(recurrent_to_output_weights),
          GetTensorData<uint8_t>(recurrent_to_output_weights_ledger),
          GetTensorScale(recurrent_to_output_weights),
          GetTensorData<int8_t>(cell_to_input_weights),
          GetTensorScale(cell_to_input_weights),
          GetTensorData<int8_t>(cell_to_forget_weights),
          GetTensorScale(cell_to_forget_weights),
          GetTensorData<int8_t>(cell_to_output_weights),
          GetTensorScale(cell_to_output_weights),
          GetTensorData<float>(input_layer_norm_coefficients),
          GetTensorData<float>(forget_layer_norm_coefficients),
          GetTensorData<float>(cell_layer_norm_coefficients),
          GetTensorData<float>(output_layer_norm_coefficients),
          GetTensorData<float>(input_gate_bias),
          GetTensorData<float>(forget_gate_bias),
          GetTensorData<float>(cell_gate_bias),
          GetTensorData<float>(output_gate_bias),
          GetTensorData<int8_t>(projection_weights),
          GetTensorData<uint8_t>(projection_weights_ledger),
          GetTensorScale(projection_weights),
          GetTensorData<float>(projection_bias), params, n_batch, n_cell,
          n_input, aux_input_size, n_output, output_batch_leading_dim,
          input_gate_scratch, forget_gate_scratch, cell_gate_scratch,
          output_gate_scratch, GetTensorData<float>(input_sf),
          GetTensorData<float>(aux_input_sf),
          GetTensorData<float>(output_state_sf),
          GetTensorData<float>(prod_scaling_factors),
          GetTensorData<float>(recovered_cell_weights),
          GetTensorData<int8_t>(input_quantized),
          GetTensorData<int8_t>(aux_input_quantized),
          GetTensorData<int8_t>(output_state_quantized),
          GetTensorData<int8_t>(cell_state_quantized),
          GetTensorData<float>(output_state), GetTensorData<float>(cell_state),
          GetTensorData<int32_t>(output_scratch_buffer), output_ptr,
          input_zp_ptr, aux_input_zp_ptr, output_state_zp_ptr, row_sums_ptr,
          row_sums_size, compute_row_sums, params->asymmetric_quantize_inputs,
          recurrent_to_input_is_diag, recurrent_to_forget_is_diag,
          recurrent_to_cell_is_diag, recurrent_to_output_is_diag, context);
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const float* input_ptr =
            GetTensorData<float>(input) + time_offset * input_step;
        const float* aux_input_ptr = nullptr;
        if (aux_input) {
          aux_input_ptr =
              GetTensorData<float>(aux_input) + time_offset * input_step;
        }
        float* output_ptr = GetTensorData<float>(output) +
                            time_offset * output_step + output_offset;
        float* output_state_ptr =
            GetTensorData<float>(output_state) + b * output_batch_leading_dim;
        float* cell_state_ptr = GetTensorData<float>(cell_state) + b * n_cell;
        float* input_gate_scratch_ptr =
            input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        float* forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        float* cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
        float* output_gate_scratch_ptr = output_gate_scratch + b * n_cell;
        LstmStepHybrid(
            input_ptr, GetTensorData<int8_t>(input_to_input_weights),
            GetTensorData<uint8_t>(input_to_input_weights_ledger),
            GetTensorScale(input_to_input_weights),
            GetTensorData<int8_t>(input_to_forget_weights),
            GetTensorData<uint8_t>(input_to_forget_weights_ledger),
            GetTensorScale(input_to_forget_weights),
            GetTensorData<int8_t>(input_to_cell_weights),
            GetTensorData<uint8_t>(input_to_cell_weights_ledger),
            GetTensorScale(input_to_cell_weights),
            GetTensorData<int8_t>(input_to_output_weights),
            GetTensorData<uint8_t>(input_to_output_weights_ledger),
            GetTensorScale(input_to_output_weights), aux_input_ptr,
            GetTensorData<int8_t>(aux_input_to_input_weights),
            GetTensorScale(aux_input_to_input_weights),
            GetTensorData<int8_t>(aux_input_to_forget_weights),
            GetTensorScale(aux_input_to_forget_weights),
            GetTensorData<int8_t>(aux_input_to_cell_weights),
            GetTensorScale(aux_input_to_cell_weights),
            GetTensorData<int8_t>(aux_input_to_output_weights),
            GetTensorScale(aux_input_to_output_weights),
            GetTensorData<int8_t>(recurrent_to_input_weights),
            GetTensorData<float>(recurrent_to_input_weights),
            GetTensorData<uint8_t>(recurrent_to_input_weights_ledger),
            GetTensorScale(recurrent_to_input_weights),
            GetTensorData<int8_t>(recurrent_to_forget_weights),
            GetTensorData<float>(recurrent_to_forget_weights),
            GetTensorData<uint8_t>(recurrent_to_forget_weights_ledger),
            GetTensorScale(recurrent_to_forget_weights),
            GetTensorData<int8_t>(recurrent_to_cell_weights),
            GetTensorData<float>(recurrent_to_cell_weights),
            GetTensorData<uint8_t>(recurrent_to_cell_weights_ledger),
            GetTensorScale(recurrent_to_cell_weights),
            GetTensorData<int8_t>(recurrent_to_output_weights),
            GetTensorData<float>(recurrent_to_output_weights),
            GetTensorData<uint8_t>(recurrent_to_output_weights_ledger),
            GetTensorScale(recurrent_to_output_weights),
            GetTensorData<int8_t>(cell_to_input_weights),
            GetTensorScale(cell_to_input_weights),
            GetTensorData<int8_t>(cell_to_forget_weights),
            GetTensorScale(cell_to_forget_weights),
            GetTensorData<int8_t>(cell_to_output_weights),
            GetTensorScale(cell_to_output_weights),
            GetTensorData<float>(input_layer_norm_coefficients),
            GetTensorData<float>(forget_layer_norm_coefficients),
            GetTensorData<float>(cell_layer_norm_coefficients),
            GetTensorData<float>(output_layer_norm_coefficients),
            GetTensorData<float>(input_gate_bias),
            GetTensorData<float>(forget_gate_bias),
            GetTensorData<float>(cell_gate_bias),
            GetTensorData<float>(output_gate_bias),
            GetTensorData<int8_t>(projection_weights),
            GetTensorData<uint8_t>(projection_weights_ledger),
            GetTensorScale(projection_weights),
            GetTensorData<float>(projection_bias), params,
            1, n_cell, n_input, aux_input_size, n_output,
            output_batch_leading_dim, input_gate_scratch_ptr,
            forget_gate_scratch_ptr, cell_gate_scratch_ptr,
            output_gate_scratch_ptr, GetTensorData<float>(input_sf),
            GetTensorData<float>(aux_input_sf),
            GetTensorData<float>(output_state_sf),
            GetTensorData<float>(prod_scaling_factors),
            GetTensorData<float>(recovered_cell_weights),
            GetTensorData<int8_t>(input_quantized),
            GetTensorData<int8_t>(aux_input_quantized),
            GetTensorData<int8_t>(output_state_quantized),
            GetTensorData<int8_t>(cell_state_quantized), output_state_ptr,
            cell_state_ptr, GetTensorData<int32_t>(output_scratch_buffer),
            output_ptr, input_zp_ptr, aux_input_zp_ptr, output_state_zp_ptr,
            row_sums_ptr, row_sums_size, compute_row_sums,
            params->asymmetric_quantize_inputs, recurrent_to_input_is_diag,
            recurrent_to_forget_is_diag, recurrent_to_cell_is_diag,
            recurrent_to_output_is_diag, context);
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus EvalInteger8x8_16(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteTensor* output_state, TfLiteTensor* cell_state, TfLiteTensor* output,
    TfLiteTensor* scratch0, TfLiteTensor* scratch1, TfLiteTensor* scratch2,
    TfLiteTensor* scratch3, TfLiteTensor* scratch4, TfLiteTensor* scratch5,
    CpuBackendContext* context) {
  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  }
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];
  int output_state_zp = output_state->params.zero_point;
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  if (time_major) {
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      const int t_rel = t;
      int8_t* output_ptr = GetTensorData<int8_t>(output) + t_rel * output_step;
      const int8_t* input_ptr =
          GetTensorData<int8_t>(input) + t_rel * input_step;
      LstmStepInteger8x8_16(
          input_ptr, GetTensorData<int8_t>(input_to_input_weights),
          integer_lstm_param->effective_input_to_input_scale_a,
          integer_lstm_param->effective_input_to_input_scale_b,
          GetTensorData<int8_t>(input_to_forget_weights),
          integer_lstm_param->effective_input_to_forget_scale_a,
          integer_lstm_param->effective_input_to_forget_scale_b,
          GetTensorData<int8_t>(input_to_cell_weights),
          integer_lstm_param->effective_input_to_cell_scale_a,
          integer_lstm_param->effective_input_to_cell_scale_b,
          GetTensorData<int8_t>(input_to_output_weights),
          integer_lstm_param->effective_input_to_output_scale_a,
          integer_lstm_param->effective_input_to_output_scale_b,
          GetTensorData<int8_t>(recurrent_to_input_weights),
          integer_lstm_param->effective_recurrent_to_input_scale_a,
          integer_lstm_param->effective_recurrent_to_input_scale_b,
          GetTensorData<int8_t>(recurrent_to_forget_weights),
          integer_lstm_param->effective_recurrent_to_forget_scale_a,
          integer_lstm_param->effective_recurrent_to_forget_scale_b,
          GetTensorData<int8_t>(recurrent_to_cell_weights),
          integer_lstm_param->effective_recurrent_to_cell_scale_a,
          integer_lstm_param->effective_recurrent_to_cell_scale_b,
          GetTensorData<int8_t>(recurrent_to_output_weights),
          integer_lstm_param->effective_recurrent_to_output_scale_a,
          integer_lstm_param->effective_recurrent_to_output_scale_b,
          GetTensorData<int16_t>(cell_to_input_weights),
          integer_lstm_param->effective_cell_to_input_scale_a,
          integer_lstm_param->effective_cell_to_input_scale_b,
          GetTensorData<int16_t>(cell_to_forget_weights),
          integer_lstm_param->effective_cell_to_forget_scale_a,
          integer_lstm_param->effective_cell_to_forget_scale_b,
          GetTensorData<int16_t>(cell_to_output_weights),
          integer_lstm_param->effective_cell_to_output_scale_a,
          integer_lstm_param->effective_cell_to_output_scale_b,
          GetTensorData<int8_t>(projection_weights),
          integer_lstm_param->effective_proj_scale_a,
          integer_lstm_param->effective_proj_scale_b,
          integer_lstm_param->hidden_zp,
          integer_lstm_param->effective_hidden_scale_a,
          integer_lstm_param->effective_hidden_scale_b,
          GetTensorData<int16_t>(input_layer_norm_coefficients),
          integer_lstm_param->layer_norm_input_scale_a,
          integer_lstm_param->layer_norm_input_scale_b,
          GetTensorData<int16_t>(forget_layer_norm_coefficients),
          integer_lstm_param->layer_norm_forget_scale_a,
          integer_lstm_param->layer_norm_forget_scale_b,
          GetTensorData<int16_t>(cell_layer_norm_coefficients),
          integer_lstm_param->layer_norm_cell_scale_a,
          integer_lstm_param->layer_norm_cell_scale_b,
          GetTensorData<int16_t>(output_layer_norm_coefficients),
          integer_lstm_param->layer_norm_output_scale_a,
          integer_lstm_param->layer_norm_output_scale_b,
          GetTensorData<int32_t>(input_gate_bias),
          GetTensorData<int32_t>(forget_gate_bias),
          GetTensorData<int32_t>(cell_gate_bias),
          GetTensorData<int32_t>(output_gate_bias),
          integer_lstm_param->quantized_cell_clip,
          integer_lstm_param->quantized_proj_clip,
          integer_lstm_param->cell_scale,
          integer_lstm_param->input_variance_guard,
          integer_lstm_param->forget_variance_guard,
          integer_lstm_param->cell_variance_guard,
          integer_lstm_param->output_variance_guard,
          integer_lstm_param->input_to_forget_effective_bias.get(),
          integer_lstm_param->recurrent_to_forget_effective_bias.get(),
          integer_lstm_param->input_to_cell_effective_bias.get(),
          integer_lstm_param->recurrent_to_cell_effective_bias.get(),
          integer_lstm_param->input_to_output_effective_bias.get(),
          integer_lstm_param->recurrent_to_output_effective_bias.get(),
          integer_lstm_param->input_to_input_effective_bias.get(),
          integer_lstm_param->recurrent_to_input_effective_bias.get(),
          integer_lstm_param->projection_effective_bias.get(), n_batch, n_cell,
          n_input, n_output, GetTensorData<int8_t>(output_state),
          output_state_zp, GetTensorData<int16_t>(cell_state), output_ptr,
          GetTensorData<int16_t>(scratch0), GetTensorData<int16_t>(scratch1),
          GetTensorData<int16_t>(scratch2), GetTensorData<int16_t>(scratch3),
          GetTensorData<int8_t>(scratch4), GetTensorData<int32_t>(scratch5),
          context);
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const int8_t* input_ptr =
            GetTensorData<int8_t>(input) + time_offset * input_step;
        int8_t* output_ptr =
            GetTensorData<int8_t>(output) + time_offset * output_step;
        int8_t* output_state_ptr =
            GetTensorData<int8_t>(output_state) + b * output_batch_leading_dim;
        int16_t* cell_state_ptr =
            GetTensorData<int16_t>(cell_state) + b * n_cell;
        LstmStepInteger8x8_16(
            input_ptr, GetTensorData<int8_t>(input_to_input_weights),
            integer_lstm_param->effective_input_to_input_scale_a,
            integer_lstm_param->effective_input_to_input_scale_b,
            GetTensorData<int8_t>(input_to_forget_weights),
            integer_lstm_param->effective_input_to_forget_scale_a,
            integer_lstm_param->effective_input_to_forget_scale_b,
            GetTensorData<int8_t>(input_to_cell_weights),
            integer_lstm_param->effective_input_to_cell_scale_a,
            integer_lstm_param->effective_input_to_cell_scale_b,
            GetTensorData<int8_t>(input_to_output_weights),
            integer_lstm_param->effective_input_to_output_scale_a,
            integer_lstm_param->effective_input_to_output_scale_b,
            GetTensorData<int8_t>(recurrent_to_input_weights),
            integer_lstm_param->effective_recurrent_to_input_scale_a,
            integer_lstm_param->effective_recurrent_to_input_scale_b,
            GetTensorData<int8_t>(recurrent_to_forget_weights),
            integer_lstm_param->effective_recurrent_to_forget_scale_a,
            integer_lstm_param->effective_recurrent_to_forget_scale_b,
            GetTensorData<int8_t>(recurrent_to_cell_weights),
            integer_lstm_param->effective_recurrent_to_cell_scale_a,
            integer_lstm_param->effective_recurrent_to_cell_scale_b,
            GetTensorData<int8_t>(recurrent_to_output_weights),
            integer_lstm_param->effective_recurrent_to_output_scale_a,
            integer_lstm_param->effective_recurrent_to_output_scale_b,
            GetTensorData<int16_t>(cell_to_input_weights),
            integer_lstm_param->effective_cell_to_input_scale_a,
            integer_lstm_param->effective_cell_to_input_scale_b,
            GetTensorData<int16_t>(cell_to_forget_weights),
            integer_lstm_param->effective_cell_to_forget_scale_a,
            integer_lstm_param->effective_cell_to_forget_scale_b,
            GetTensorData<int16_t>(cell_to_output_weights),
            integer_lstm_param->effective_cell_to_output_scale_a,
            integer_lstm_param->effective_cell_to_output_scale_b,
            GetTensorData<int8_t>(projection_weights),
            integer_lstm_param->effective_proj_scale_a,
            integer_lstm_param->effective_proj_scale_b,
            integer_lstm_param->hidden_zp,
            integer_lstm_param->effective_hidden_scale_a,
            integer_lstm_param->effective_hidden_scale_b,
            GetTensorData<int16_t>(input_layer_norm_coefficients),
            integer_lstm_param->layer_norm_input_scale_a,
            integer_lstm_param->layer_norm_input_scale_b,
            GetTensorData<int16_t>(forget_layer_norm_coefficients),
            integer_lstm_param->layer_norm_forget_scale_a,
            integer_lstm_param->layer_norm_forget_scale_b,
            GetTensorData<int16_t>(cell_layer_norm_coefficients),
            integer_lstm_param->layer_norm_cell_scale_a,
            integer_lstm_param->layer_norm_cell_scale_b,
            GetTensorData<int16_t>(output_layer_norm_coefficients),
            integer_lstm_param->layer_norm_output_scale_a,
            integer_lstm_param->layer_norm_output_scale_b,
            GetTensorData<int32_t>(input_gate_bias),
            GetTensorData<int32_t>(forget_gate_bias),
            GetTensorData<int32_t>(cell_gate_bias),
            GetTensorData<int32_t>(output_gate_bias),
            integer_lstm_param->quantized_cell_clip,
            integer_lstm_param->quantized_proj_clip,
            integer_lstm_param->cell_scale,
            integer_lstm_param->input_variance_guard,
            integer_lstm_param->forget_variance_guard,
            integer_lstm_param->cell_variance_guard,
            integer_lstm_param->output_variance_guard,
            integer_lstm_param->input_to_forget_effective_bias.get(),
            integer_lstm_param->recurrent_to_forget_effective_bias.get(),
            integer_lstm_param->input_to_cell_effective_bias.get(),
            integer_lstm_param->recurrent_to_cell_effective_bias.get(),
            integer_lstm_param->input_to_output_effective_bias.get(),
            integer_lstm_param->recurrent_to_output_effective_bias.get(),
            integer_lstm_param->input_to_input_effective_bias.get(),
            integer_lstm_param->recurrent_to_input_effective_bias.get(),
            integer_lstm_param->projection_effective_bias.get(), 1,
            n_cell, n_input, n_output, output_state_ptr, output_state_zp,
            cell_state_ptr, output_ptr, GetTensorData<int16_t>(scratch0),
            GetTensorData<int16_t>(scratch1), GetTensorData<int16_t>(scratch2),
            GetTensorData<int16_t>(scratch3), GetTensorData<int8_t>(scratch4),
            GetTensorData<int32_t>(scratch5), context);
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus EvalInteger8x8_8(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, TfLiteTensor* output_state,
    TfLiteTensor* cell_state, TfLiteTensor* output,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteTensor* scratch0, TfLiteTensor* scratch1, TfLiteTensor* scratch2,
    TfLiteTensor* scratch3, TfLiteTensor* scratch4, TfLiteTensor* scratch5,
    TfLiteTensor* scratch6, TfLiteTensor* scratch7) {
  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = input->dims->data[0];
    n_batch = input->dims->data[1];
  }
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];
  const int32_t input_zp = input->params.zero_point;
  const int32_t output_state_zp = output_state->params.zero_point;
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  const int input_step = n_batch * n_input;
  const int output_step = n_batch * output_batch_leading_dim;
  for (int t = 0; t < max_time; t++) {
    const int t_rel = t;
    int8_t* output_ptr = GetTensorData<int8_t>(output) + t_rel * output_step;
    const int8_t* input_ptr = GetTensorData<int8_t>(input) + t_rel * input_step;
    lstm_eval::LstmStepInteger8x8_8(
        input_ptr, input_zp,
        GetTensorData<int8_t>(input_to_input_weights),
        integer_lstm_param->effective_input_to_input_scale_a,
        integer_lstm_param->effective_input_to_input_scale_b,
        GetTensorData<int8_t>(input_to_forget_weights),
        integer_lstm_param->effective_input_to_forget_scale_a,
        integer_lstm_param->effective_input_to_forget_scale_b,
        GetTensorData<int8_t>(input_to_cell_weights),
        integer_lstm_param->effective_input_to_cell_scale_a,
        integer_lstm_param->effective_input_to_cell_scale_b,
        GetTensorData<int8_t>(input_to_output_weights),
        integer_lstm_param->effective_input_to_output_scale_a,
        integer_lstm_param->effective_input_to_output_scale_b,
        GetTensorData<int8_t>(recurrent_to_input_weights),
        integer_lstm_param->effective_recurrent_to_input_scale_a,
        integer_lstm_param->effective_recurrent_to_input_scale_b,
        GetTensorData<int8_t>(recurrent_to_forget_weights),
        integer_lstm_param->effective_recurrent_to_forget_scale_a,
        integer_lstm_param->effective_recurrent_to_forget_scale_b,
        GetTensorData<int8_t>(recurrent_to_cell_weights),
        integer_lstm_param->effective_recurrent_to_cell_scale_a,
        integer_lstm_param->effective_recurrent_to_cell_scale_b,
        GetTensorData<int8_t>(recurrent_to_output_weights),
        integer_lstm_param->effective_recurrent_to_output_scale_a,
        integer_lstm_param->effective_recurrent_to_output_scale_b,
        GetTensorData<int8_t>(cell_to_input_weights),
        integer_lstm_param->effective_cell_to_input_scale_a,
        integer_lstm_param->effective_cell_to_input_scale_b,
        GetTensorData<int8_t>(cell_to_forget_weights),
        integer_lstm_param->effective_cell_to_forget_scale_a,
        integer_lstm_param->effective_cell_to_forget_scale_b,
        GetTensorData<int8_t>(cell_to_output_weights),
        integer_lstm_param->effective_cell_to_output_scale_a,
        integer_lstm_param->effective_cell_to_output_scale_b,
        GetTensorData<int8_t>(projection_weights),
        integer_lstm_param->effective_proj_scale_a,
        integer_lstm_param->effective_proj_scale_b,
        GetTensorData<int16_t>(input_layer_norm_coefficients),
        integer_lstm_param->layer_norm_input_scale_a,
        integer_lstm_param->layer_norm_input_scale_b,
        GetTensorData<int16_t>(forget_layer_norm_coefficients),
        integer_lstm_param->layer_norm_forget_scale_a,
        integer_lstm_param->layer_norm_forget_scale_b,
        GetTensorData<int16_t>(cell_layer_norm_coefficients),
        integer_lstm_param->layer_norm_cell_scale_a,
        integer_lstm_param->layer_norm_cell_scale_b,
        GetTensorData<int16_t>(output_layer_norm_coefficients),
        integer_lstm_param->layer_norm_output_scale_a,
        integer_lstm_param->layer_norm_output_scale_b,
        GetTensorData<int32_t>(input_gate_bias),
        GetTensorData<int32_t>(forget_gate_bias),
        GetTensorData<int32_t>(cell_gate_bias),
        GetTensorData<int32_t>(output_gate_bias),
        GetTensorData<int32_t>(projection_bias),
        params, integer_lstm_param->intermediate_scale_a,
        integer_lstm_param->intermediate_scale_b,
        integer_lstm_param->intermediate_zp,
        integer_lstm_param->quantized_cell_clip,
        integer_lstm_param->quantized_proj_clip, n_batch, n_cell, n_input,
        n_output, output_batch_leading_dim, GetTensorData<int8_t>(output_state),
        output_state_zp, GetTensorData<int16_t>(cell_state), output_ptr,
        GetTensorData<int8_t>(scratch0), GetTensorData<int8_t>(scratch1),
        GetTensorData<int16_t>(scratch2), GetTensorData<int16_t>(scratch3),
        GetTensorData<int16_t>(scratch4), GetTensorData<int16_t>(scratch5),
        GetTensorData<int16_t>(scratch6), GetTensorData<int16_t>(scratch7));
  }
  return kTfLiteOk;
}
}  
}  
}  
}  