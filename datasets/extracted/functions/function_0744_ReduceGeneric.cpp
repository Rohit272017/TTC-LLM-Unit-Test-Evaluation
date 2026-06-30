#include <string.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
namespace toco {
namespace {
void ReduceGeneric(bool keep_dims, const std::vector<int>& axes,
                   const Shape& input_shape, const std::vector<float>& input,
                   Shape* check_output_shape, std::vector<float>* output,
                   const std::function<float(float, float)>& reducer) {
  if (!IsNonEmpty(input_shape)) {
    return;
  }
  Shape output_shape = input_shape;
  std::vector<int> reduction_mask(input_shape.dimensions_count(), 1);
  for (const auto& axis : axes) {
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_shape.dimensions_count());
    reduction_mask[axis] = 0;
    output_shape.mutable_dims()->at(axis) = 1;
  }
  std::vector<int> output_indices(input_shape.dimensions_count());
  for (size_t input_offset = 0; input_offset < input.size(); ++input_offset) {
    std::vector<int> input_indices = ReverseOffset(input_shape, input_offset);
    for (int i = 0; i < input_shape.dimensions_count(); ++i) {
      output_indices[i] = input_indices[i] * reduction_mask[i];
    }
    int output_offset = Offset(output_shape, output_indices);
    if (input_indices == output_indices) {
      output->at(output_offset) = input.at(input_offset);
    } else {
      output->at(output_offset) =
          reducer(output->at(output_offset), input.at(input_offset));
    }
  }
  if (!keep_dims) {
    std::vector<int> new_dims;
    for (int i = 0; i < output_shape.dimensions_count(); ++i) {
      if (reduction_mask[i]) {
        new_dims.push_back(output_shape.dims(i));
      }
    }
    output_shape.mutable_dims()->swap(new_dims);
  }
  *check_output_shape = output_shape;
}
}  
bool CopyMinMaxFromFirstInput(const Operator& op, Model* model) {
  auto& output_array = model->GetArray(op.outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op.inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax.min;
  output_minmax.max = input_minmax.max;
  return true;
}
::tensorflow::Status ResolveConstantUnaryOperator::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
  *modified = false;
  const auto unary_it = model->operators.begin() + op_index;
  const auto* unary_op = unary_it->get();
  switch (unary_op->type) {
    case OperatorType::kCast:
    case OperatorType::kExp:
    case OperatorType::kLog:
    case OperatorType::kNeg:
    case OperatorType::kRsqrt:
    case OperatorType::kSqrt:
    case OperatorType::kSquare:
    case OperatorType::kSum:
    case OperatorType::kReduceMin:  
    case OperatorType::kReduceMax:  
    case OperatorType::kReshape:
    case OperatorType::kRelu6:
    case OperatorType::kRelu1:
    case OperatorType::kRelu:
      break;
    default:
      return absl::OkStatus();
  }
  if (!IsConstantParameterArray(*model, unary_op->inputs[0])) {
    return absl::OkStatus();
  }
  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (unary_op->inputs[0] == rnn_state.back_edge_source_array()) {
      return absl::OkStatus();
    }
    if (unary_op->inputs[0] == rnn_state.state_array()) {
      return absl::OkStatus();
    }
  }
  auto& output_array = model->GetArray(unary_op->outputs[0]);
  if (!output_array.has_shape()) {
    return absl::OkStatus();
  }
  if (unary_op->fused_activation_function !=
      FusedActivationFunctionType::kNone) {
    AddMessageF(
        "Not resolving constant %s "
        " because it has a fused activation function",
        LogName(*unary_op));
    return absl::OkStatus();
  }
  if (unary_op->type == OperatorType::kReshape) {
    CopyMinMaxFromFirstInput(*unary_op, model);
  }
  const auto& input_array = model->GetArray(unary_op->inputs[0]);
  CHECK(input_array.buffer);
  std::vector<DataType<ArrayDataType::kFloat>> const* input_float_data =
      nullptr;
  if (unary_op->type == OperatorType::kCast) {
    CastOperator const* cast_op = static_cast<CastOperator const*>(unary_op);
    if (cast_op->dst_data_type != ArrayDataType::kFloat) {
      AddMessageF(
          "Not resolving constant %s because we currently only support casting "
          "to float",
          LogName(*unary_op));
      return absl::OkStatus();
    }
    if (cast_op->src_data_type != input_array.buffer->type) {
      AddMessageF(
          "Not resolving constant %s because cast op source type does not "
          "match input type",
          LogName(*unary_op));
    }
  } else {
    if (input_array.buffer->type != ArrayDataType::kFloat) {
      return absl::OkStatus();
    }
    input_float_data = &(input_array.GetBuffer<ArrayDataType::kFloat>().data);
  }
  const Shape& output_shape = output_array.shape();
  const int output_dims_count = output_shape.dimensions_count();
  const int output_buffer_size = RequiredBufferSizeForShape(output_shape);
  auto& output_float_data =
      output_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  output_float_data.resize(output_buffer_size);
  const Shape& input_shape = input_array.shape();
  const int input_buffer_size = RequiredBufferSizeForShape(input_shape);
  if (unary_op->type == OperatorType::kCast) {
    for (int i = 0; i < output_buffer_size; i++) {
      float outval = 0.0f;
      if (input_array.buffer->type == ArrayDataType::kFloat) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kFloat>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kUint8) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kUint8>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kInt32) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kInt32>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kInt64) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kInt64>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kBool) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kBool>().data[i]);
      } else {
        LOG(FATAL) << "Unsupported cast op input type";
      }
      output_float_data[i] = outval;
    }
  } else if (unary_op->type == OperatorType::kReshape) {
    CHECK(input_buffer_size == output_buffer_size);
    output_float_data = *input_float_data;
  } else if (unary_op->type == OperatorType::kSum) {
    CHECK_EQ(unary_op->inputs.size(), 2) << "Sum needs 2 inputs";
    if (!IsConstantParameterArray(*model, unary_op->inputs[1])) {
      AddMessageF("Axis input is non-constant");
      return absl::OkStatus();
    }
    auto& axis_array = model->GetArray(unary_op->inputs[1]);
    CHECK(axis_array.data_type == ArrayDataType::kInt32);
    auto sum_op = static_cast<const TensorFlowSumOperator*>(unary_op);
    Shape check_output_shape;
    ReduceGeneric(
        sum_op->keep_dims, axis_array.GetBuffer<ArrayDataType::kInt32>().data,
        input_shape, *input_float_data, &check_output_shape, &output_float_data,
        [](float existing, float current) -> float {
          return existing + current;
        });
    CHECK(check_output_shape == output_shape)
        << "Shape propagation output shape doesn't match output shape from op";
  } else if (unary_op->type == OperatorType::kReduceMin) {
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), 1);
    }
    float min = (*input_float_data)[0];
    for (int i = 0; i < input_buffer_size; i++) {
      min = std::min(min, (*input_float_data)[i]);
    }
    output_float_data[0] = min;
  } else if (unary_op->type == OperatorType::kReduceMax) {
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), 1);
    }
    float max = (*input_float_data)[0];
    for (int i = 0; i < input_buffer_size; i++) {
      max = std::max(max, (*input_float_data)[i]);
    }
    output_float_data[0] = max;
  } else if (unary_op->type == OperatorType::kExp ||
             unary_op->type == OperatorType::kNeg ||
             unary_op->type == OperatorType::kLog ||
             unary_op->type == OperatorType::kRsqrt ||
             unary_op->type == OperatorType::kSqrt ||
             unary_op->type == OperatorType::kSquare) {
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), input_shape.dims(i));
    }
    for (int i = 0; i < output_buffer_size; i++) {
      const float val = (*input_float_data)[i];
      float outval = 0.f;
      if (unary_op->type == OperatorType::kExp) {
        outval = std::exp(val);
      } else if (unary_op->type == OperatorType::kNeg) {
        outval = -val;
      } else if (unary_op->type == OperatorType::kLog) {
        outval = std::log(val);
      } else if (unary_op->type == OperatorType::kRsqrt) {
        outval = 1.0f / std::sqrt(val);
      } else if (unary_op->type == OperatorType::kSqrt) {
        outval = std::sqrt(val);
      } else if (unary_op->type == OperatorType::kSquare) {
        outval = val * val;
      } else {
        LOG(FATAL) << "should not get here.";
      }
      output_float_data[i] = outval;
    }
  } else if (unary_op->type == OperatorType::kRelu6 ||
             unary_op->type == OperatorType::kRelu1 ||
             unary_op->type == OperatorType::kRelu) {
    for (int i = 0; i < output_buffer_size; ++i) {
      const float value = (*input_float_data)[i];
      float new_value = 0.0f;
      switch (unary_op->type) {
        case OperatorType::kRelu: {
          static constexpr float kLower = 0;
          new_value = value < kLower ? kLower : value;
          break;
        }
        case OperatorType::kRelu1: {
          static constexpr float kUpper = 1;
          static constexpr float kLower = -1;
          new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
          break;
        }
        case OperatorType::kRelu6: {
          static constexpr float kUpper = 6;
          static constexpr float kLower = 0;
          new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
          break;
        }
        default:
          LOG(FATAL) << "Unsupported activation function "
                     << LogName(*unary_op);
          return absl::OkStatus();
      }
      output_float_data[i] = new_value;
    }
  } else {
    LOG(FATAL) << "should not get here.";
  }
  DeleteOpAndArrays(model, unary_op);
  *modified = true;
  return absl::OkStatus();
}
}  