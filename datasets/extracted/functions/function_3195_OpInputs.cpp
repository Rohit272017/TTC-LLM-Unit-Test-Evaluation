#include "tensorflow/lite/delegates/flex/kernel.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "flatbuffers/flexbuffers.h"  
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_type.h"
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeAndType;
using tensorflow::shape_inference::ShapeHandle;
namespace tflite {
namespace flex {
constexpr char kReadVariableOp[] = "ReadVariableOp";
constexpr char kInterOpParallelismAttrName[] = "use_inter_op_parallelism";
struct OpNode;
struct TensorSource {
  OpNode* node;
  int node_output_index;
};
class OpInputs {
 public:
  explicit OpInputs(const TfLiteIntArray* indexes) {
    for (int index : TfLiteIntArrayView(indexes)) {
      inputs_.push_back(index);
    }
    forwardable_.resize(inputs_.size());
  }
  ~OpInputs() = default;
  int Size() const { return inputs_.size(); }
  int TfLiteIndex(int i) const { return inputs_[i]; }
  void InitializeTensorSources(
      const std::map<int, TensorSource>& tflite_tensor_sources) {
    sources_.clear();
    for (int i : inputs_) {
      auto it = tflite_tensor_sources.find(i);
      if (it == tflite_tensor_sources.end()) {
        sources_.push_back({nullptr, 0});
      } else {
        sources_.push_back(it->second);
      }
    }
  }
  void SetForwardable(int i, bool v) { forwardable_[i] = v; }
  bool IsForwardable(int i) const { return forwardable_[i]; }
  TensorSource GetTensorSource(int i) const { return sources_[i]; }
 private:
  std::vector<int> inputs_;
  std::vector<TensorSource> sources_;
  std::vector<int> forwardable_;
};
class OpOutputs {
 public:
  explicit OpOutputs(const TfLiteIntArray* indexes) {
    for (int index : TfLiteIntArrayView(indexes)) {
      outputs_.push_back(index);
    }
    vector_.resize(outputs_.size());
  }
  ~OpOutputs() = default;
  void InitializeGraphOutputs(const std::set<int>& subgraph_outputs) {
    subgraph_outputs_.clear();
    for (int i : outputs_) {
      subgraph_outputs_.push_back(subgraph_outputs.count(i) > 0);
    }
  }
  bool IsSubgraphOutput(int i) const { return subgraph_outputs_[i]; }
  const tensorflow::Tensor& GetTensor(int i) const { return vector_[i]; }
  tensorflow::Tensor ReleaseTensor(int i) { return std::move(vector_[i]); }
  int Size() const { return outputs_.size(); }
  int TfLiteIndex(int i) const { return outputs_[i]; }
  absl::InlinedVector<tensorflow::Tensor, 2UL>* GetTensors() {
    return &vector_;
  }
 private:
  std::vector<int> outputs_;
  std::vector<bool> subgraph_outputs_;
  absl::InlinedVector<tensorflow::Tensor, 2UL> vector_;
};
struct OpDataInfo {
  BufferMap* buffer_map;
  std::map<int, int>* tensor_release_map;
  std::set<int> already_transferred_outputs;
};
class OpNode {
 public:
  OpNode(const TfLiteIntArray* inputs, const TfLiteIntArray* outputs)
      : inputs_(inputs), outputs_(outputs) {}
  ~OpNode() = default;
  const string& name() const { return name_; }
  void set_name(const string& name) { name_ = name; }
  int index() const { return index_; }
  void set_index(int index) { index_ = index; }
  const tensorflow::NodeDef& nodedef() const { return nodedef_; }
  const tensorflow::OpRegistrationData* op_reg_data() const {
    return op_reg_data_;
  }
  const OpInputs& inputs() const { return inputs_; }
  OpInputs* mutable_inputs() { return &inputs_; }
  const OpOutputs& outputs() const { return outputs_; }
  OpOutputs* mutable_outputs() { return &outputs_; }
  int NumInputs() const { return inputs_.Size(); }
  int NumOutputs() const { return outputs_.Size(); }
  const tensorflow::tfrt_stub::OpKernelRunner& op_kernel_runner() const {
    return op_kernel_runner_;
  }
  tensorflow::Status InitializeNodeDef(const void* custom_initial_data,
                                       int custom_initial_data_size) {
    if (!custom_initial_data) {
      return tensorflow::errors::Internal(
          "Cannot convert empty data into a valid NodeDef");
    }
    const flexbuffers::Vector& v =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(custom_initial_data),
            custom_initial_data_size)
            .AsVector();
    name_ = v[0].AsString().str();
    if (!nodedef_.ParseFromString(v[1].AsString().str())) {
      nodedef_.Clear();
      return tensorflow::errors::Internal(
          "Failed to parse data into a valid NodeDef");
    }
    TF_RETURN_IF_ERROR(
        tensorflow::OpRegistry::Global()->LookUp(nodedef_.op(), &op_reg_data_));
    AddDefaultsToNodeDef(op_reg_data_->op_def, &nodedef_);
    const auto& op_def = op_reg_data_->op_def;
    for (const auto& attr : op_def.attr()) {
      if (attr.name() == kInterOpParallelismAttrName) {
        (*nodedef_.mutable_attr())[kInterOpParallelismAttrName].set_b(false);
        break;
      }
    }
    return absl::OkStatus();
  }
  tensorflow::Status BuildOpKernelRunner(
      tensorflow::EagerContext* eager_context) {
    TF_ASSIGN_OR_RETURN(op_kernel_runner_,
                        tensorflow::tfrt_stub::OpKernelRunner::Create(
                            name_, inputs_.Size(), 
                            [this](tensorflow::AttrValueMap* attr_value_map) {
                              *attr_value_map = nodedef_.attr();
                              return absl::OkStatus();
                            },
                            *eager_context->pflr(),
                            eager_context->local_device_mgr()->HostCPU()));
    return absl::OkStatus();
  }
  tensorflow::Status BuildOpKernelInputs(
      const BufferMap* buffer_map,
      tensorflow::tfrt_stub::OpKernelRunState* run_state) {
    run_state->input_tf_tensors.resize(inputs_.Size());
    run_state->input_tf_tensor_values.resize(inputs_.Size());
    for (int i = 0; i < inputs_.Size(); ++i) {
      int input_index = inputs_.TfLiteIndex(i);
      TensorSource s = inputs_.GetTensorSource(i);
      if (!s.node) {
        if (!buffer_map->HasTensor(input_index)) {
          return tensorflow::errors::Internal(
              "Cannot read from invalid tensor index ", input_index);
        }
        run_state->input_tf_tensors[i] = buffer_map->GetTensor(input_index);
      } else {
        if (inputs_.IsForwardable(i)) {
          run_state->input_tf_tensors[i] =
              s.node->outputs_.ReleaseTensor(s.node_output_index);
        } else {
          run_state->input_tf_tensors[i] =
              s.node->outputs_.GetTensor(s.node_output_index);
        }
      }
      run_state->input_tf_tensor_values[i].tensor =
          &run_state->input_tf_tensors[i];
    }
    return absl::OkStatus();
  }
  bool ShouldPersistTensorflowTensor(TfLiteContext* context,
                                     const OpDataInfo* shared_info,
                                     int tensor_index, int node_index) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (IsResourceOrVariant(tensor) || tensor->type == kTfLiteString) {
      return true;
    }
    auto it = shared_info->tensor_release_map->find(tensor_index);
    return it != shared_info->tensor_release_map->end() &&
           it->second > node_index;
  }
  TfLiteStatus CopyToTfLiteTensor(TfLiteContext* context,
                                  OpDataInfo* shared_info, TfLiteTensor* tensor,
                                  tensorflow::Tensor* tf_tensor,
                                  int tensor_index) const {
    if (tensor->allocation_type == kTfLiteDynamic) {
      CopyShapeAndType(context, *tf_tensor, tensor);
    }
    tensorflow::StringPiece t_data = tf_tensor->tensor_data();
    if (tf_tensor->NumElements() != NumElements(tensor) ||
        tf_tensor->TotalBytes() != tensor->bytes) {
      TF_LITE_KERNEL_LOG(context,
                         "FlexDelegate: Tensor %s(%d) buffer size mismatch "
                         "%zu(%lld) != %ld(%ld)",
                         tensor->name, tensor_index, tf_tensor->TotalBytes(),
                         tf_tensor->NumElements(), tensor->bytes,
                         NumElements(tensor));
      return kTfLiteError;
    }
    memcpy(tensor->data.raw, t_data.data(), t_data.size());
    *tf_tensor = {};
    shared_info->already_transferred_outputs.insert(tensor_index);
    return kTfLiteOk;
  }
  tensorflow::Status MaybePersistTensorflowOutputs(TfLiteContext* context,
                                                   OpDataInfo* shared_info,
                                                   int node_index) {
    auto* tensors = outputs_.GetTensors();
    for (int i = 0; i < outputs_.Size(); ++i) {
      if (outputs_.IsSubgraphOutput(i)) {
        tensorflow::Tensor& tf_tensor = tensors->at(i);
        const int tflite_index = outputs_.TfLiteIndex(i);
        TfLiteTensor* tensor = &context->tensors[tflite_index];
        if (!ShouldPersistTensorflowTensor(context, shared_info, tflite_index,
                                           node_index)) {
          if (CopyToTfLiteTensor(context, shared_info, tensor, &tf_tensor,
                                 tflite_index) != kTfLiteOk) {
            return tensorflow::Status(absl::StatusCode::kInternal,
                                      "failed to copy data from TF tensor");
          }
        } else {
          shared_info->buffer_map->SetFromTensorFlow(outputs_.TfLiteIndex(i),
                                                     tf_tensor);
        }
      }
    }
    return absl::OkStatus();
  }
 private:
  OpNode(const OpNode&) = delete;
  OpNode& operator=(const OpNode&) = delete;
  string name_;
  int index_;
  tensorflow::NodeDef nodedef_;
  const tensorflow::OpRegistrationData* op_reg_data_;
  OpInputs inputs_;
  OpOutputs outputs_;
  tensorflow::tfrt_stub::OpKernelRunner op_kernel_runner_;
};
struct OpData {
  tensorflow::EagerContext* eager_context;
  tensorflow::CancellationManager* cancellation_manager;
  std::vector<std::unique_ptr<OpNode>> nodes;
  std::vector<int> subgraph_inputs;
  std::vector<int> subgraph_outputs;
  std::set<int>
      disable_reusing_buffer_tensors;  
  OpDataInfo shared_info;
};
tensorflow::Status DelegateKernel::ExecuteOpKernelRunner(
    tensorflow::tfrt_stub::OpKernelRunState* run_state, TfLiteContext* context,
    OpNode* node_data) {
  const auto& op_kernel_runner = node_data->op_kernel_runner();
  if (op_kernel_runner.op_kernel()->num_outputs() != node_data->NumOutputs()) {
    return tensorflow::errors::Internal(
        "Unexpected number of outputs from tensorflow::OpKernel");
  }
  TF_RETURN_IF_ERROR(node_data->BuildOpKernelInputs(
      op_data_->shared_info.buffer_map, run_state));
  run_state->params.inputs = run_state->input_tf_tensor_values;
  run_state->params.op_kernel = op_kernel_runner.op_kernel();
  run_state->params.input_alloc_attrs = op_kernel_runner.input_alloc_attrs();
  run_state->params.output_attr_array =
      op_kernel_runner.output_alloc_attrs().data();
  run_state->params.function_library =
      op_kernel_runner.function_library_runtime();
  tensorflow::OpKernelContext tf_context(&run_state->params,
                                         node_data->NumOutputs());
  op_kernel_runner.Run(&tf_context);
  TF_RETURN_IF_ERROR(tf_context.status());
  auto& outputs = *node_data->mutable_outputs()->GetTensors();
  for (int i = 0; i < tf_context.num_outputs(); ++i) {
    outputs[i] = std::move(*tf_context.mutable_output(i));
  }
  return node_data->MaybePersistTensorflowOutputs(
      context, &(op_data_->shared_info), node_data->index());
}
DelegateKernel::DelegateKernel() : op_data_(new OpData) {}
DelegateKernel::~DelegateKernel() = default;
TfLiteStatus DelegateKernel::Init(TfLiteContext* context,
                                  const TfLiteDelegateParams* params) {
  auto* flex_delegate_data =
      reinterpret_cast<FlexDelegate*>(params->delegate->data_)->mutable_data();
  op_data_->eager_context = flex_delegate_data->GetEagerContext();
  op_data_->cancellation_manager = flex_delegate_data->GetCancellationManager();
  op_data_->shared_info.buffer_map = flex_delegate_data->GetBufferMap(context);
  op_data_->shared_info.tensor_release_map =
      flex_delegate_data->GetTensorReleaseMap(context);
  CHECK(params->output_tensors);
  std::set<int> output_set;
  for (auto tensor_index : TfLiteIntArrayView(params->output_tensors)) {
    op_data_->subgraph_outputs.push_back(tensor_index);
    output_set.insert(tensor_index);
  }
  CHECK(params->input_tensors);
  for (auto tensor_index : TfLiteIntArrayView(params->input_tensors)) {
    op_data_->subgraph_inputs.push_back(tensor_index);
  }
  std::set<int> subgraph_inputs(op_data_->subgraph_inputs.begin(),
                                op_data_->subgraph_inputs.end());
  op_data_->nodes.reserve(params->nodes_to_replace->size);
  CHECK(params->nodes_to_replace);
  tensorflow::Status status;
  auto check_if_op_reuses_input = [](const string& op_name) {
    return op_name == "TensorListPushBack" || op_name == "TensorListSetItem" ||
           op_name == "SparseReshape" || op_name == "StridedSlice" ||
           op_name == "RaggedTensorToVariant" || op_name == "TensorMapInsert";
  };
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, node_index, &node, &reg);
    op_data_->nodes.emplace_back(new OpNode(node->inputs, node->outputs));
    OpNode& node_data = *op_data_->nodes.back();
    node_data.set_index(node_index);
    node_data.set_name("");
    status = node_data.InitializeNodeDef(node->custom_initial_data,
                                         node->custom_initial_data_size);
    if (!status.ok()) break;
    status = node_data.BuildOpKernelRunner(op_data_->eager_context);
    if (!status.ok()) break;
    for (auto tensor_index : TfLiteIntArrayView(node->inputs)) {
      int node_id = node_index;
      if (const std::map<int, int>::iterator it =
              op_data_->shared_info.tensor_release_map->find(tensor_index);
          it != op_data_->shared_info.tensor_release_map->end()) {
        node_id = std::max(it->second, node_index);
      }
      (*op_data_->shared_info.tensor_release_map)[tensor_index] = node_id;
      if (subgraph_inputs.count(tensor_index) &&
          check_if_op_reuses_input(node_data.nodedef().op())) {
        op_data_->disable_reusing_buffer_tensors.insert(tensor_index);
      }
    }
  }
  TF_LITE_ENSURE_STATUS(ConvertStatus(context, status));
  std::map<int, TensorSource> tflite_tensor_sources;
  for (auto& node_data : op_data_->nodes) {
    node_data->mutable_outputs()->InitializeGraphOutputs(output_set);
    for (int i = 0; i < node_data->outputs().Size(); ++i) {
      int output_index = node_data->outputs().TfLiteIndex(i);
      tflite_tensor_sources[output_index] = TensorSource{node_data.get(), i};
    }
  }
  for (auto& node_data : op_data_->nodes) {
    node_data->mutable_inputs()->InitializeTensorSources(tflite_tensor_sources);
  }
  return kTfLiteOk;
}
TfLiteStatus DelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_MSG(
      context, op_data_->eager_context != nullptr,
      "Failed to initialize eager context. This often happens when a CPU "
      "device has not been registered, presumably because some symbols from "
      "tensorflow/core:core_cpu_impl were not linked into the binary.");
  std::map<int, int> tensor_ref_count;
  BufferMap* buffer_map = op_data_->shared_info.buffer_map;
  for (auto tensor_index : op_data_->subgraph_inputs) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (IsConstantTensor(tensor)) {
      if (!tensor->data_is_stale || !buffer_map->HasTensor(tensor_index)) {
        buffer_map->SetFromTfLite(tensor_index, tensor);
      }
    }
    tensor_ref_count[tensor_index] += 2;
  }
  if (shapes_are_valid_) {
    shapes_are_valid_ =
        (ValidateOutputTensorShapeConsistency(context) == kTfLiteOk);
    if (shapes_are_valid_) {
      TFLITE_LOG(tflite::TFLITE_LOG_INFO,
                 "FlexDelegate: All tensor shapes are consistent.");
    } else {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "FlexDelegate: Some tensor shapes are inconsistent.");
    }
  }
  for (auto tensor_index : op_data_->subgraph_outputs) {
    if (!shapes_are_valid_) {
      SetTensorToDynamic(&context->tensors[tensor_index]);
    }
    ++tensor_ref_count[tensor_index];
  }
  for (const auto& node_data : op_data_->nodes) {
    if (node_data->nodedef().op().empty()) {
      TF_LITE_KERNEL_LOG(context, "Invalid NodeDef in Flex op '%s'",
                         node_data->name().c_str());
      return kTfLiteError;
    }
    TF_LITE_ENSURE(context, node_data->op_kernel_runner());
    for (int i = 0; i < node_data->inputs().Size(); ++i) {
      ++tensor_ref_count[node_data->inputs().TfLiteIndex(i)];
    }
  }
  for (auto& node_data : op_data_->nodes) {
    for (int i = 0; i < node_data->inputs().Size(); ++i) {
      bool f = (tensor_ref_count[node_data->inputs().TfLiteIndex(i)] == 1);
      node_data->mutable_inputs()->SetForwardable(i, f);
    }
  }
  return kTfLiteOk;
}
TfLiteStatus DelegateKernel::ValidateOutputTensorShapeConsistency(
    TfLiteContext* context) const {
  for (const auto& node_data : op_data_->nodes) {
    auto op_name = node_data->name().c_str();
    auto num_inputs = node_data->inputs().Size();
    std::vector<const tensorflow::Tensor*> input_tensors_vector(num_inputs,
                                                                nullptr);
    InferenceContext c(
        TF_GRAPH_DEF_VERSION, node_data->nodedef(),
        node_data->op_reg_data()->op_def, std::vector<ShapeHandle>(num_inputs),
        input_tensors_vector, {},
        std::vector<std::unique_ptr<std::vector<ShapeAndType>>>());
    for (int i = 0; i < num_inputs; ++i) {
      const auto input_tensor_index = node_data->inputs().TfLiteIndex(i);
      TfLiteTensor* tfl_tensor = &context->tensors[input_tensor_index];
      if (IsConstantTensor(tfl_tensor)) {
        input_tensors_vector[i] =
            op_data_->shared_info.buffer_map->GetTensorPtr(input_tensor_index);
      }
      const auto dims_array = tfl_tensor->dims;
      std::vector<DimensionHandle> dims(dims_array->size);
      for (int j = 0; j < dims_array->size; ++j) {
        dims[j] = c.MakeDim(dims_array->data[j]);
      }
      c.SetInput(i, c.MakeShape(dims));
    }
    c.set_input_tensors(input_tensors_vector);
    tensorflow::Status status = c.construction_status();
    if (!status.ok()) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "Shape construction failed for op '%s'", op_name);
      return kTfLiteError;
    }
    if (node_data->op_reg_data()->shape_inference_fn == nullptr) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "No shape inference function exists for op '%s'", op_name);
      return kTfLiteError;
    }
    status = c.Run(node_data->op_reg_data()->shape_inference_fn);
    auto num_outputs = node_data->outputs().Size();
    if (num_outputs != c.num_outputs()) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "Number of output tensors are mismatched for op '%s' %d != %d",
                 op_name, num_outputs, c.num_outputs());
      return kTfLiteError;
    }
    for (int i = 0; i < num_outputs; ++i) {
      const auto output_tensor_index = node_data->outputs().TfLiteIndex(i);
      TfLiteTensor* tfl_tensor = &context->tensors[output_tensor_index];
      const std::string tfl_shape_string =
          GetShapeDebugString(tfl_tensor->dims);
      const std::string calculated_shape_string = c.DebugString(c.output(i));
      if (tfl_shape_string != calculated_shape_string) {
        if ((strcmp(op_name, kReadVariableOp) == 0) &&
            (tfl_tensor->dims->size > 0)) {
          continue;
        }
        TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                   "op '%s' output%d tensor#%d shape mismatch for  %s != %s",
                   op_name, i, output_tensor_index, tfl_shape_string.c_str(),
                   calculated_shape_string.c_str());
        return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}
static tensorflow::CancellationManager* GetDefaultCancellationManager() {
  static auto* const cancellation_manager = new tensorflow::CancellationManager;
  return cancellation_manager;
}
TfLiteStatus DelegateKernel::Eval(TfLiteContext* context, TfLiteNode* node) {
  BufferMap* buffer_map = op_data_->shared_info.buffer_map;
  for (auto tensor_index : op_data_->subgraph_inputs) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (!IsConstantTensor(tensor)) {
      if (!tensor->data_is_stale || !buffer_map->HasTensor(tensor_index)) {
        buffer_map->SetFromTfLite(
            tensor_index, tensor,
            !op_data_->disable_reusing_buffer_tensors.count(tensor_index));
      }
    }
  }
  auto& eager_context = *op_data_->eager_context;
  {
    tensorflow::tfrt_stub::OpKernelRunState run_state;
    run_state.params.step_container = eager_context.StepContainer();
    auto* device = eager_context.local_device_mgr()->HostCPU();
    run_state.params.device = device;
    run_state.params.resource_manager = device->resource_manager();
    run_state.params.runner = eager_context.runner();
    run_state.params.cancellation_manager =
        op_data_->cancellation_manager ? op_data_->cancellation_manager
                                       : GetDefaultCancellationManager();
    for (auto& node_data : op_data_->nodes) {
      TFLITE_SCOPED_DELEGATE_PROFILED_OPERATOR_PROFILE(
          reinterpret_cast<Profiler*>(context->profiler),
          node_data->name().c_str(), node_data->index());
      if (op_data_->cancellation_manager != nullptr &&
          op_data_->cancellation_manager->IsCancelled()) {
        TF_LITE_KERNEL_LOG(
            context, "Client requested cancel during DelegateKernel::Eval");
        return kTfLiteError;
      }
      auto status = ExecuteOpKernelRunner(&run_state, context, node_data.get());
      TF_LITE_ENSURE_OK(context, ConvertStatus(context, status));
    }
  }
  for (auto tensor_index : op_data_->subgraph_outputs) {
    if (op_data_->shared_info.already_transferred_outputs.count(tensor_index) !=
        0) {
      continue;
    }
    if (!buffer_map->HasTensor(tensor_index)) {
      TF_LITE_KERNEL_LOG(context, "Cannot write to invalid tensor index %d",
                         tensor_index);
      return kTfLiteError;
    }
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    const tensorflow::Tensor& tf_tensor = buffer_map->GetTensor(tensor_index);
    if (tensor->allocation_type == kTfLiteDynamic) {
      TF_LITE_ENSURE_OK(context, CopyShapeAndType(context, tf_tensor, tensor));
      tensor->buffer_handle = tensor_index;
      tensor->data_is_stale = true;
      continue;
    }
    if (tf_tensor.NumElements() != NumElements(tensor) ||
        tf_tensor.TotalBytes() != tensor->bytes) {
      TF_LITE_KERNEL_LOG(context,
                         "FlexDelegate: Tensor %s(%d) buffer size mismatch "
                         "%zu(%lld) != %ld(%ld)",
                         tensor->name, tensor_index, tf_tensor.TotalBytes(),
                         tf_tensor.NumElements(), tensor->bytes,
                         NumElements(tensor));
      return kTfLiteError;
    }
    tensorflow::StringPiece t_data = tf_tensor.tensor_data();
    memcpy(tensor->data.raw, t_data.data(), t_data.size());
  }
  return kTfLiteOk;
}
const std::map<int, int>& DelegateKernel::GetTensorReleaseMap() const {
  return *(op_data_->shared_info.tensor_release_map);
}
}  
}  