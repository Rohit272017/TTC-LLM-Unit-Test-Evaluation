#include "tensorflow/core/common_runtime/shape_refiner.h"
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/eval_const_tensor.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tsl/platform/statusor.h"
namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;
ShapeRefiner::ShapeRefiner(int graph_def_version,
                           const OpRegistryInterface* ops)
    : graph_def_version_(graph_def_version),
      ops_registry_(ops),
      graph_runner_(Env::Default()) {}
ShapeRefiner::ShapeRefiner(const VersionDef& versions,
                           const OpRegistryInterface* ops)
    : ShapeRefiner(versions.producer(), ops) {}
ShapeRefiner::~ShapeRefiner() {
  const_tensor_map_.clear();
}
namespace {
constexpr char kArgOp[] = "_Arg";
constexpr char kRetvalOp[] = "_Retval";
}  
Status ShapeRefiner::InferShapesForFunctionSubNode(
    const Node* node, InferenceContext* outer_context) {
  TF_RETURN_IF_ERROR(AddNodeInternal(node, outer_context));
  InferenceContext* node_context = CHECK_NOTNULL(GetContext(node));
  if (StringPiece(node->type_string()) == kArgOp) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node->def()), "index", &index));
    if (index < 0 || outer_context->num_inputs() <= index) {
      return errors::Internal(
          "Function instantiation included invalid input index: ", index,
          " not in [0, ", outer_context->num_inputs(), ").");
    }
    if (outer_context->input(index).SameHandle(ShapeHandle())) {
      VLOG(1) << "Function instantiation has undefined input shape at "
              << "index: " << index << " in the outer inference context.";
      node_context->set_output(0, node_context->UnknownShape());
    } else {
      node_context->set_output(0, outer_context->input(index));
    }
    auto* resource = outer_context->input_handle_shapes_and_types(index);
    if (resource) {
      node_context->set_output_handle_shapes_and_types(0, *resource);
    }
  } else if (StringPiece(node->type_string()) == kRetvalOp) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node->def()), "index", &index));
    if (index < 0 || outer_context->num_outputs() <= index) {
      return errors::Internal(
          "Function instantiation included invalid output index: ", index,
          " not in [0, ", outer_context->num_outputs(), ").");
    }
    ShapeHandle handle;
    TensorShapeProto proto;
    node_context->ShapeHandleToProto(node_context->input(0), &proto);
    TF_RETURN_IF_ERROR(outer_context->MakeShapeFromShapeProto(proto, &handle));
    outer_context->set_output(index, handle);
    const std::vector<ShapeAndType>* resource =
        node_context->input_handle_shapes_and_types(0);
    if (resource) {
      std::vector<ShapeAndType> copied_shapes_and_types;
      for (auto& shape_and_type : *resource) {
        ShapeHandle handle;
        TensorShapeProto proto;
        node_context->ShapeHandleToProto(shape_and_type.shape, &proto);
        TF_RETURN_IF_ERROR(
            outer_context->MakeShapeFromShapeProto(proto, &handle));
        copied_shapes_and_types.push_back(
            ShapeAndType(handle, shape_and_type.dtype, shape_and_type.type));
      }
      outer_context->set_output_handle_shapes_and_types(
          index, copied_shapes_and_types);
    }
  }
  return absl::OkStatus();
}
Status ShapeRefiner::InferShapesForFunction(const FunctionDef* function_def,
                                            AttrSlice attributes,
                                            InferenceContext* outer_context) {
  const Graph* graph;
  const string& fname = function_def->signature().name();
  auto it = functions_.find(fname);
  if (it != functions_.end()) {
    graph = it->second.get();
  } else {
    InstantiationResult result;
    TF_RETURN_IF_ERROR(InstantiateFunction(
        *function_def, attributes,
        [this](const string& op, const OpDef** sig) {
          return this->function_library_->LookUpOpDef(op, sig);
        },
        &result));
    Graph* new_graph = new Graph(function_library_);
    GraphConstructorOptions options;
    options.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(
        ConvertNodeDefsToGraph(options, result.nodes, new_graph));
    functions_[fname].reset(new_graph);
    graph = new_graph;
  }
  absl::flat_hash_set<const Node*> function_nodes;
  Status inference_status = absl::OkStatus();
  {
    auto node_shape_inference_lambda = [this, &outer_context, &function_nodes,
                                        &inference_status](const Node* node) {
      if (!inference_status.ok()) return;
      inference_status = InferShapesForFunctionSubNode(node, outer_context);
      function_nodes.insert(node);
    };
    ReverseDFS(*graph, {}, node_shape_inference_lambda);
  }
  for (const Node* node : function_nodes) {
    node_to_context_.erase(node);
  }
  return inference_status;
}
Status ShapeRefiner::AddNode(const Node* node) {
  return AddNodeInternal(node, nullptr);
}
Status ShapeRefiner::AddNodeInternal(
    const Node* node, shape_inference::InferenceContext* outer_context) {
  std::unique_ptr<InferenceContext> ic(new InferenceContext(
      graph_def_version_, node->def(), node->op_def(),
      std::vector<ShapeHandle>(node->num_inputs()), {}, {}, {}));
  TF_RETURN_IF_ERROR(ic->construction_status());
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;
    if (e->dst_input() < 0) {
      return tensorflow::errors::Internal(
          "Index ", e->dst_input(), " is negative but not a control edge.");
    }
    const Node* input = e->src();
    auto it = node_to_context_.find(input);
    if (it == node_to_context_.end()) {
      ic->SetInput(e->dst_input(), ic->UnknownShape());
      continue;
    }
    InferenceContext* input_ic = it->second.get();
    ic->SetInput(e->dst_input(), input_ic->output(e->src_output()));
    const auto* in_v =
        input_ic->output_handle_shapes_and_types(e->src_output());
    if (in_v != nullptr) {
      DataType input_type = e->src()->output_type(e->src_output());
      DCHECK(input_type == DT_RESOURCE || input_type == DT_VARIANT);
      ic->set_input_handle_shapes_and_types(e->dst_input(),
                                            std::vector<ShapeAndType>(*in_v));
    }
  }
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr &&
      require_shape_inference_fns_) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }
  TF_RETURN_IF_ERROR(RunShapeFn(node, op_reg_data, ic.get(), outer_context));
  node_to_context_[node].swap(ic);
  return absl::OkStatus();
}
Status ShapeRefiner::SetShape(const Node* node, int output_port,
                              ShapeHandle shape) {
  auto c = GetContext(node);
  if (c == nullptr) {
    return errors::Internal("Could not find context for ", node->name());
  }
  if (output_port < 0 || output_port >= node->num_outputs()) {
    return errors::InvalidArgument(
        "output_port '", output_port, "' is out of range, ", "node '",
        node->name(), "' has ", node->num_outputs(), " outputs");
  }
  if (node->num_outputs() > c->num_outputs()) {
    TF_RETURN_IF_ERROR(c->ExpandOutputs(node->num_outputs()));
  }
  ShapeHandle existing_shape = c->output(output_port);
  TF_RETURN_IF_ERROR(c->Merge(existing_shape, shape, &shape));
  c->set_output(output_port, shape);
  return absl::OkStatus();
}
Status ShapeRefiner::UpdateNode(const Node* node, bool relax, bool* refined) {
  auto it = node_to_context_.find(node);
  if (it == node_to_context_.end()) {
    *refined = true;
    return AddNode(node);
  }
  InferenceContext* node_context = it->second.get();
  TF_RETURN_IF_ERROR(node_context->construction_status());
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;
    int dst_input = e->dst_input();
    int src_output = e->src_output();
    Node* input = e->src();
    auto iter = node_to_context_.find(input);
    if (iter == node_to_context_.end()) {
      return errors::FailedPrecondition(
          "Input ", dst_input, " ('", input->name(), "') for '", node->name(),
          "' was not previously added to ShapeRefiner.");
    }
    InferenceContext* c = iter->second.get();
    DCHECK_GE(dst_input, 0);
    ShapeHandle existing_input = node_context->input(dst_input);
    if (!relax) {
      if (node_context->MergeInput(dst_input, c->output(src_output))) {
        if (!SameDefinedShape(node_context, node_context->input(dst_input),
                              existing_input)) {
          *refined = true;
        }
      }
    } else {
      if (node_context->RelaxInput(dst_input, c->output(src_output))) {
        if (!SameDefinedShape(node_context, node_context->input(dst_input),
                              existing_input)) {
          *refined = true;
        }
      }
    }
    if (node_context->requested_input_tensor_as_partial_shape(dst_input)) {
      *refined = true;
    }
    if (e->src()->output_type(src_output) == DT_RESOURCE) {
      auto* outputs = c->output_handle_shapes_and_types(src_output);
      if (!outputs) continue;
      if (!relax &&
          node_context->MergeInputHandleShapesAndTypes(dst_input, *outputs)) {
        *refined = true;
      } else if (relax) {
        std::vector<ShapeAndType> existing_inputs;
        const std::vector<ShapeAndType>* inputs =
            node_context->input_handle_shapes_and_types(dst_input);
        if (inputs) {
          existing_inputs = *inputs;
        }
        if (node_context->RelaxInputHandleShapesAndMergeTypes(dst_input,
                                                              *outputs)) {
          if (IsUpdatedShapesOrTypes(
                  node_context, existing_inputs,
                  *node_context->input_handle_shapes_and_types(dst_input))) {
            *refined = true;
          }
        }
      }
    }
  }
  if (!*refined) {
    return absl::OkStatus();
  }
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr &&
      require_shape_inference_fns_) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }
  if (!op_reg_data->shape_inference_fn) {
    return absl::OkStatus();
  }
  return RunShapeFn(node, op_reg_data, node_context);
}
Status ShapeRefiner::EvaluateConstantTensorForEdge(
    const Node* node, int dst_idx, bool* evaluated, Tensor* result,
    InferenceContext* outer_context) {
  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));
  const Node& src = *input_edge->src();
  const int src_output = input_edge->src_output();
  auto lookup = [&](const Node& node, int index) -> std::optional<Tensor> {
    if (node.IsArg() && outer_context != nullptr) {
      int index;
      if (GetNodeAttr(node.def(), "index", &index).ok() && 0 <= index &&
          index < outer_context->num_inputs()) {
        const auto* tensor = outer_context->input_tensor(index);
        outer_context->request_input_tensor(index);
        if (tensor != nullptr) {
          return *tensor;
        }
      }
    }
    auto it = const_tensor_map_.find({node.id(), index});
    if (it != const_tensor_map_.end()) {
      return it->second;
    }
    return std::optional<Tensor>();
  };
  std::optional<EvaluateConstantTensorRunner> runner;
  if (!disable_constant_propagation_) {
    runner = EvaluateConstantTensorRunner{
        ops_registry_,
        graph_def_version_,
        &graph_runner_,
    };
  }
  TF_ASSIGN_OR_RETURN(auto tensor, EvaluateConstantTensor(
                                       src, src_output, *this, lookup, runner));
  *evaluated = tensor.has_value();
  if (tensor.has_value()) {
    if (tensor->TotalBytes() <= kMaxTensorSize) {
      const_tensor_map_.emplace(std::make_pair(src.id(), src_output), *tensor);
    }
    *result = *std::move(tensor);
  }
  return absl::OkStatus();
}
Status ShapeRefiner::EvaluateConstantIntScalarEdge(
    const Node* node, int dst_idx, bool* evaluated, int64_t* result,
    shape_inference::InferenceContext* outer_context) {
  Tensor scalar;
  TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(node, dst_idx, evaluated,
                                                   &scalar, outer_context));
  if (*evaluated) {
    if (scalar.NumElements() != 1) {
      return errors::InvalidArgument(
          "EvaluateConstantIntScalarEdge called on non-scalar edge: ",
          scalar.NumElements());
    }
    if (scalar.dtype() == DT_INT32) {
      *result = scalar.scalar<int32>()();
    } else {
      if (scalar.dtype() != DT_INT64) {
        return errors::InvalidArgument(
            "EvaluateConstantIntScalarEdge called on non-integer edge: ",
            scalar.dtype());
      }
      *result = scalar.scalar<int64_t>()();
    }
  }
  return absl::OkStatus();
}
Status ShapeRefiner::ConstantPartialShape(
    InferenceContext* target_context, const Node* node, int dst_idx,
    ShapeHandle* result, shape_inference::InferenceContext* outer_context) {
  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));
  InferenceContext* src_context = GetContext(input_edge->src());
  if (src_context == nullptr) return errors::Internal("Missing src context");
  ShapeHandle src_shape = src_context->output(input_edge->src_output());
  if (src_context->Value(src_context->Rank(src_shape)) == 0) {
    Tensor t;
    bool evaluated = false;
    TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(node, dst_idx, &evaluated,
                                                     &t, outer_context));
    if (!evaluated) {
      return errors::InvalidArgument(
          "Received a shape scalar with unknown static value.  A static value "
          "of '-1' is required to represent an unknown shape.");
    }
    if (t.dims() == 0) {
      if (t.dtype() == DT_INT32 && t.scalar<int32>()() == -1) {
        *result = target_context->UnknownShape();
        return absl::OkStatus();
      } else if (t.dtype() == DT_INT64 && t.scalar<int64_t>()() == -1) {
        *result = target_context->UnknownShape();
        return absl::OkStatus();
      }
    }
    return errors::InvalidArgument(
        "Received an invalid shape scalar with a static value that is not "
        "'-1': ",
        t.DebugString());
  }
  TF_RETURN_IF_ERROR(src_context->WithRank(src_shape, 1, &src_shape));
  const string& src_op = input_edge->src()->type_string();
  if (src_context->Value(src_context->Dim(src_shape, 0)) == 0) {
    *result = target_context->Scalar();
  } else if (src_op == "Cast") {
    Tensor t;
    bool evaluated = false;
    if (EvaluateConstantTensorForEdge(node, dst_idx, &evaluated, &t,
                                      outer_context)
            .ok()) {
      if (evaluated &&
          target_context->MakeShapeFromTensor(&t, src_shape, result).ok()) {
        return absl::OkStatus();
      }
    }
    ShapeHandle pre_cast_shape;
    if (!ConstantPartialShape(target_context, input_edge->src(), 0,
                              &pre_cast_shape, outer_context)
             .ok()) {
      TF_RETURN_IF_ERROR(
          target_context->MakeShapeFromTensor(nullptr, src_shape, result));
    }
    if (!target_context->RankKnown(pre_cast_shape)) {
      *result = target_context->UnknownShape();
      return absl::OkStatus();
    }
    auto* dest_type = input_edge->src()->attrs().Find("DstT");
    if (dest_type == nullptr || dest_type->value_case() != AttrValue::kType ||
        (dest_type->type() != DT_INT32 && dest_type->type() != DT_INT64)) {
      *result = target_context->MakeShape(std::vector<DimensionHandle>(
          target_context->Rank(pre_cast_shape), target_context->UnknownDim()));
      return absl::OkStatus();
    }
    *result = pre_cast_shape;
  } else if (src_op == "Shape") {
    *result = src_context->input(0);
  } else if (src_op == "ShapeN") {
    *result = src_context->input(input_edge->src_output());
  } else if (src_op == "Pack") {
    std::vector<DimensionHandle> dims;
    for (int i = 0; i < src_context->num_inputs(); ++i) {
      int64_t size;
      bool evaluated;
      TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(
          input_edge->src(), i, &evaluated, &size, outer_context));
      if (evaluated) {
        dims.push_back(size < 0 ? target_context->UnknownDim()
                                : target_context->MakeDim(size));
      } else {
        dims.push_back(target_context->UnknownDim());
      }
    }
    *result = target_context->MakeShape(dims);
  } else if (src_op == "Concat" || src_op == "ConcatV2") {
    *result = target_context->Scalar();
    const int concat_dim =
        src_op == "Concat" ? 0 : src_context->num_inputs() - 1;
    for (int i = 0; i < src_context->num_inputs(); ++i) {
      if (i == concat_dim) continue;
      ShapeHandle sub_result;
      TF_RETURN_IF_ERROR(ConstantPartialShape(target_context, input_edge->src(),
                                              i, &sub_result, outer_context));
      if (!target_context->RankKnown(sub_result)) {
        *result = target_context->UnknownShape();
        return absl::OkStatus();
      }
      TF_RETURN_IF_ERROR(
          target_context->Concatenate(*result, sub_result, result));
    }
  } else if (src_op == "StridedSlice") {
    TF_RETURN_IF_ERROR(PartialStridedSliceShape(input_edge->src(), src_context,
                                                result, outer_context));
  } else if (src_op == "VariableShape") {
    auto* handle_data = src_context->input_handle_shapes_and_types(0);
    if (handle_data != nullptr && !handle_data->empty()) {
      *result = handle_data->at(0).shape;
    } else {
      *result = target_context->UnknownShape();
    }
  } else {
    Tensor t;
    bool evaluated = false;
    TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(node, dst_idx, &evaluated,
                                                     &t, outer_context));
    TF_RETURN_IF_ERROR(target_context->MakeShapeFromTensor(
        evaluated ? &t : nullptr, src_shape, result));
  }
  return absl::OkStatus();
}
Status ShapeRefiner::PartialStridedSliceShape(
    Node* slice_node, InferenceContext* ctx, ShapeHandle* result,
    shape_inference::InferenceContext* outer_context) {
  for (int i = 1; i <= 3; ++i) {
    ShapeHandle input_shape = ctx->input(i);
    if (ctx->Value(ctx->Dim(input_shape, 0)) != 1) {
      *result = ctx->UnknownShape();
      return absl::OkStatus();
    }
  }
  int begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(slice_node->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "ellipsis_mask", &ellipsis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(slice_node->attrs(), "shrink_axis_mask", &shrink_axis_mask));
  if (!(begin_mask == 0 || begin_mask == 1) ||
      !(end_mask == 0 || end_mask == 1) || ellipsis_mask != 0 ||
      new_axis_mask != 0 || shrink_axis_mask != 0) {
    *result = ctx->UnknownShape();
    return absl::OkStatus();
  }
  bool evaluated;
  int64_t begin;
  if (begin_mask == 1) {
    begin = 0;
  } else {
    TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(slice_node, 1, &evaluated,
                                                     &begin, outer_context));
    if (!evaluated) {
      *result = ctx->UnknownShape();
      return absl::OkStatus();
    }
  }
  int64_t end;
  if (end_mask == 1) {
    end = std::numeric_limits<int64_t>::max();
  } else {
    TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(slice_node, 2, &evaluated,
                                                     &end, outer_context));
    if (!evaluated) {
      *result = ctx->UnknownShape();
      return absl::OkStatus();
    }
  }
  int64_t stride;
  TF_RETURN_IF_ERROR(EvaluateConstantIntScalarEdge(slice_node, 3, &evaluated,
                                                   &stride, outer_context));
  if (!evaluated) {
    *result = ctx->UnknownShape();
    return absl::OkStatus();
  }
  ShapeHandle input;
  TF_RETURN_IF_ERROR(
      ConstantPartialShape(ctx, slice_node, 0, &input, outer_context));
  TF_RETURN_IF_ERROR(ctx->Subshape(input, begin, end, stride, result));
  return absl::OkStatus();
}
Status ShapeRefiner::RunShapeFn(const Node* node,
                                const OpRegistrationData* op_reg_data,
                                InferenceContext* c,
                                InferenceContext* outer_context) {
  std::vector<const Tensor*> input_tensors(node->num_inputs(), nullptr);
  std::vector<Tensor> real_tensors(node->num_inputs());
  std::vector<bool> attempted_materialization(node->num_inputs());
  std::vector<bool> attempted_tensor_as_shape_conversion(node->num_inputs());
  std::vector<ShapeHandle> input_tensors_as_shapes;
  c->set_input_tensors(input_tensors);
  c->set_input_tensors_as_shapes(input_tensors_as_shapes);
  auto run_inference_lambda = [&]() {
    if (function_library_ && IsFunctionCall(*function_library_, *node)) {
      bool disable_shape_inference;
      if (!GetNodeAttr(AttrSlice(node->def()), "_disable_call_shape_inference",
                       &disable_shape_inference)
               .ok() ||
          !disable_shape_inference) {
        NameAttrList function;
        TF_RETURN_IF_ERROR(
            NameAndAttrsFromFunctionCall(node->def(), &function));
        const FunctionDef* function_def =
            function_library_->Find(function.name());
        if (function_def != nullptr) {
          auto const_tensor_map_copy = const_tensor_map_;
          const_tensor_map_.clear();
          VLOG(4) << "Running shape inference for function \""
                  << function.name() << "\".";
          Status function_inference_status = InferShapesForFunction(
              function_def, AttrSlice(&function.attr()), c);
          const_tensor_map_ = const_tensor_map_copy;
          VLOG(4) << "Shape inference for function \"" << function.name()
                  << "\" returned status " << function_inference_status << ".";
          return function_inference_status;
        }
      }
    }
    if (op_reg_data->shape_inference_fn) {
      VLOG(4) << "Running shape inference function for node \"" << node->name()
              << "\" of type \"" << node->type_string() << "\".";
      TF_RETURN_IF_ERROR(c->Run(op_reg_data->shape_inference_fn));
    } else {
      VLOG(4) << "Unknown shape inference function for node \"" << node->name()
              << "\" of type \"" << node->type_string() << "\".";
      TF_RETURN_IF_ERROR(c->Run(shape_inference::UnknownShape));
    }
    VLOG(4) << "Shape inference passed for node \"" << node->name()
            << "\" of type \"" << node->type_string() << "\".";
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(run_inference_lambda());
  bool rerun_shape_fn;
  do {
    rerun_shape_fn = false;
    for (int i = 0; i < c->num_inputs(); ++i) {
      if (!c->requested_input_tensor(i)) {
        continue;
      }
      if (!attempted_materialization[i]) {
        attempted_materialization[i] = true;
        Tensor result;
        bool evaluated = false;
        TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(
            node, i, &evaluated, &result, outer_context));
        if (evaluated) {
          real_tensors[i] = result;
          input_tensors[i] = &real_tensors[i];
          rerun_shape_fn = true;
        }
      }
      if (c->requested_input_tensor_as_partial_shape(i) &&
          !attempted_tensor_as_shape_conversion[i]) {
        attempted_tensor_as_shape_conversion[i] = true;
        if (i >= input_tensors_as_shapes.size()) {
          input_tensors_as_shapes.resize(i + 1);
        }
        ShapeHandle s;
        TF_RETURN_IF_ERROR(ConstantPartialShape(c, node, i, &s, outer_context));
        input_tensors_as_shapes[i] = s;
        rerun_shape_fn = true;
      }
    }
    if (rerun_shape_fn) {
      c->set_input_tensors(input_tensors);
      c->set_input_tensors_as_shapes(input_tensors_as_shapes);
      TF_RETURN_IF_ERROR(run_inference_lambda());
    }
  } while (rerun_shape_fn);
  return absl::OkStatus();
}
bool ShapeRefiner::SameDefinedShape(InferenceContext* c, ShapeHandle s0,
                                    ShapeHandle s1) {
  if (s0.SameHandle(s1)) {
    return true;
  }
  if (c->Rank(s0) != c->Rank(s1)) {
    return false;
  }
  if (!c->RankKnown(s0) && !c->RankKnown(s1)) {
    return false;
  }
  for (int i = 0; i < c->Rank(s0); ++i) {
    if (!c->Dim(s0, i).SameHandle(c->Dim(s1, i))) {
      int64_t val0 = c->Value(c->Dim(s0, i));
      int64_t val1 = c->Value(c->Dim(s1, i));
      if (val0 < 0 || val1 < 0 || val0 != val1) {
        return false;
      }
    }
  }
  return true;
}
bool ShapeRefiner::IsUpdatedShapesOrTypes(
    InferenceContext* c, const std::vector<ShapeAndType>& existing,
    const std::vector<ShapeAndType>& updated) {
  if (existing.size() != updated.size()) {
    return true;
  }
  for (int i = 0; i < existing.size(); i++) {
    if (!SameDefinedShape(c, existing[i].shape, updated[i].shape) ||
        existing[i].dtype != updated[i].dtype) {
      return true;
    }
  }
  return false;
}
}  