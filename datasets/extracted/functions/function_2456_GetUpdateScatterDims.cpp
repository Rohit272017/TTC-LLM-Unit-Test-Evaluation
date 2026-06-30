#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>
#include "Eigen/Core"  
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_scatter {
namespace {
constexpr int kInputsTensor = 0;
constexpr int kScatterIndicesTensor = 1;
constexpr int kUpdatesTensor = 2;
constexpr int kOutputTensor = 0;
enum class ComputationType {
  kUpdate,
  kAdd,
  kMultiply,
  kMaximum,
  kMinimum,
  kOther
};
struct OpData {
  ComputationType computation_type;
};
using DimVector = std::vector<int64_t>;
static DimVector GetUpdateScatterDims(int64_t updates_rank,
                                      const int64_t* update_window_dims,
                                      int num_update_window_dims) {
  DimVector result;
  for (int64_t dim = 0; dim < updates_rank; ++dim) {
    if (!ArrayContains(update_window_dims, num_update_window_dims, dim)) {
      result.push_back(dim);
    }
  }
  return result;
}
template <typename IndexType>
static Index<IndexType> GatherIndex(const Index<IndexType>& index,
                                    const DimVector& dims) {
  Index<IndexType> result;
  for (auto dim : dims) {
    result.push_back(index[dim]);
  }
  return result;
}
template <typename IndexType>
static bool IsInBounds(Index<IndexType> index, RuntimeShape shape) {
  if (index.size() != shape.DimensionsCount()) {
    return false;
  }
  for (int dim = 0; dim < shape.DimensionsCount(); ++dim) {
    if (index[dim] >= shape.Dims(dim)) {
      return false;
    }
  }
  return true;
}
static ComputationType OpCodeToComputationType(int op_code) {
  switch (op_code) {
    case kTfLiteBuiltinStablehloAdd:
      return ComputationType::kAdd;
    case kTfLiteBuiltinStablehloMultiply:
      return ComputationType::kMultiply;
    case kTfLiteBuiltinStablehloMaximum:
      return ComputationType::kMaximum;
    case kTfLiteBuiltinStablehloMinimum:
      return ComputationType::kMinimum;
    default:
      return ComputationType::kOther;
  }
}
static TfLiteStatus GetComputationType(const Subgraph* computation_subgraph,
                                       ComputationType* computation_type,
                                       TfLiteContext* context) {
  if (computation_subgraph->execution_plan().empty()) {
    *computation_type = ComputationType::kUpdate;
    return kTfLiteOk;
  }
  if (computation_subgraph->execution_plan().size() > 1) {
    TF_LITE_KERNEL_LOG(context,
                       "Only one kernel allowed withing the stablehlo region. "
                       "(%zu) kernels found.\n",
                       computation_subgraph->execution_plan().size());
    return kTfLiteError;
  }
  const TfLiteRegistration* kernel =
      &(computation_subgraph
            ->node_and_registration(computation_subgraph->execution_plan()[0])
            ->second);
  *computation_type = OpCodeToComputationType(kernel->builtin_code);
  if (*computation_type == ComputationType::kOther) {
    TF_LITE_KERNEL_LOG(
        context,
        "Only update, Add, Multiply, Maximum and Minimum operations are "
        "currently supported for stablehlo.scatter.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}
template <typename DataType, typename IndexType>
static TfLiteStatus ApplyComputation(TfLiteTensor* tensor,
                                     Index<IndexType> index,
                                     DataType input_value,
                                     DataType update_value,
                                     ComputationType computation_type,
                                     TfLiteContext* context) {
  DataType* tensor_data = GetTensorData<DataType>(tensor);
  DataType result;
  if (computation_type == ComputationType::kUpdate) {
    result = update_value;
  } else if (computation_type == ComputationType::kAdd) {
    result = input_value + update_value;
  } else if (computation_type == ComputationType::kMultiply) {
    result = input_value * update_value;
  } else if (computation_type == ComputationType::kMaximum) {
    result = std::max(input_value, update_value);
  } else if (computation_type == ComputationType::kMinimum) {
    result = std::min(input_value, update_value);
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Provided kernel in the stablehlo scatter region is not "
                       "yet supported.");
    return kTfLiteError;
  }
  tensor_data[TensorIndexToFlat(index.data(), index.size(),
                                GetTensorShape(tensor))] = result;
  return kTfLiteOk;
}
template <typename IndexType, typename DataType>
TfLiteStatus EvalWithTypes(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteStablehloScatterParams* data =
      reinterpret_cast<TfLiteStablehloScatterParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputsTensor, &input));
  const TfLiteTensor* scatter_indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kScatterIndicesTensor,
                                          &scatter_indices));
  const TfLiteTensor* updates;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kUpdatesTensor, &updates));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  memcpy(output->data.data, input->data.data, input->bytes);
  RuntimeShape input_shape = GetTensorShape(input);
  int input_rank = input_shape.DimensionsCount();
  const DataType* output_data = GetTensorData<DataType>(output);
  RuntimeShape scatter_indices_shape = GetTensorShape(scatter_indices);
  RuntimeShape updates_shape = GetTensorShape(updates);
  int64_t updates_rank = updates_shape.DimensionsCount();
  Index<IndexType> update_index = Index<IndexType>(updates_rank, 0);
  const DataType* updates_data = GetTensorData<DataType>(updates);
  DimVector update_scatter_dims = GetUpdateScatterDims(
      updates_rank, data->update_window_dims, data->num_update_window_dims);
  std::vector<int64_t> update_window_dims_vec(
      data->update_window_dims,
      data->update_window_dims + data->num_update_window_dims);
  do {
    Index<IndexType> update_scatter_index =
        GatherIndex(update_index, update_scatter_dims);
    Index<IndexType> start_index =
        ReadIndexVector(scatter_indices, scatter_indices_shape,
                        update_scatter_index, data->index_vector_dim);
    Index<IndexType> full_start_index;
    TF_LITE_ENSURE_STATUS(ScatterIndex(
        start_index, data->scatter_dims_to_operand_dims,
        data->num_scatter_dims_to_operand_dims, input_rank, &full_start_index));
    Index<IndexType> window_index =
        GatherIndex(update_index, update_window_dims_vec);
    Index<IndexType> full_window_index;
    TF_LITE_ENSURE_STATUS(ExpandDims(window_index, data->inserted_window_dims,
                                     data->num_inserted_window_dims,
                                     &full_window_index));
    Index<IndexType> result_index =
        AddIndices(full_start_index, full_window_index);
    if (!IsInBounds(result_index, input_shape)) {
      continue;
    }
    DataType input_value = output_data[TensorIndexToFlat(
        result_index.data(), input_rank, input_shape)];
    DataType update_value = updates_data[TensorIndexToFlat(
        update_index.data(), updates_rank, updates_shape)];
    TF_LITE_ENSURE_STATUS(ApplyComputation(output, result_index, input_value,
                                           update_value,
                                           op_data->computation_type, context));
  } while (
      NextIndex(updates_rank, updates_shape.DimsData(), update_index.data()));
  return TfLiteStatus::kTfLiteOk;
}
template <typename IndexType>
TfLiteStatus EvalWithIndexType(TfLiteContext* context, TfLiteNode* node,
                               TfLiteType index_type, TfLiteType data_type) {
  switch (data_type) {
    case kTfLiteFloat16:
      return EvalWithTypes<IndexType, Eigen::half>(context, node);
    case kTfLiteFloat32:
      return EvalWithTypes<IndexType, float>(context, node);
    case kTfLiteFloat64:
      return EvalWithTypes<IndexType, double>(context, node);
    case kTfLiteInt8:
      return EvalWithTypes<IndexType, int8_t>(context, node);
    case kTfLiteInt16:
      return EvalWithTypes<IndexType, int16_t>(context, node);
    case kTfLiteInt32:
      return EvalWithTypes<IndexType, int32_t>(context, node);
    case kTfLiteInt64:
      return EvalWithTypes<IndexType, int64_t>(context, node);
    case kTfLiteUInt8:
      return EvalWithTypes<IndexType, uint8_t>(context, node);
    case kTfLiteUInt16:
      return EvalWithTypes<IndexType, uint16_t>(context, node);
    case kTfLiteUInt32:
      return EvalWithTypes<IndexType, uint32_t>(context, node);
    case kTfLiteUInt64:
      return EvalWithTypes<IndexType, uint64_t>(context, node);
    default:
      TF_LITE_KERNEL_LOG(
          context, "(Index Type: %s, Data Type: %s) currently not supported.\n",
          TfLiteTypeGetName(index_type), TfLiteTypeGetName(data_type));
      return TfLiteStatus::kTfLiteError;
  }
}
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new ComputationType{ComputationType::kOther};
}
void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<ComputationType*>(buffer);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputsTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));
  const TfLiteStablehloScatterParams* data =
      reinterpret_cast<TfLiteStablehloScatterParams*>(node->builtin_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  if (data->update_computation_subgraph_index >= subgraphs->size()) {
    TF_LITE_KERNEL_LOG(context,
                       "Computation subgraph not found for stablehlo.scatter.");
    return TfLiteStatus::kTfLiteError;
  }
  Subgraph* computation_subgraph =
      (*subgraphs)[data->update_computation_subgraph_index].get();
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_STATUS(GetComputationType(
      computation_subgraph, &op_data->computation_type, context));
  return TfLiteStatus::kTfLiteOk;
}
}  
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputsTensor, &input));
  const TfLiteTensor* scatter_indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kScatterIndicesTensor,
                                          &scatter_indices));
  TfLiteType index_type = scatter_indices->type;
  TfLiteType data_type = input->type;
  if (index_type == kTfLiteInt32) {
    return EvalWithIndexType<int32_t>(context, node, index_type, data_type);
  } else if (index_type == kTfLiteInt64) {
    return EvalWithIndexType<int64_t>(context, node, index_type, data_type);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(index_type));
    return TfLiteStatus::kTfLiteError;
  }
}
}  
TfLiteRegistration* Register_STABLEHLO_SCATTER() {
  static TfLiteRegistration r = {
      stablehlo_scatter::Init, stablehlo_scatter::Free,
      stablehlo_scatter::Prepare, stablehlo_scatter::Eval};
  return &r;
}
}  
}  
}  