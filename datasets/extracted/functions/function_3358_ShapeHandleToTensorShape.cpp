#include "tensorflow/compiler/jit/shape_inference.h"
#include <cstdint>
#include <map>
#include <vector>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace tensorflow {
namespace {
Status ShapeHandleToTensorShape(shape_inference::InferenceContext* context,
                                const shape_inference::ShapeHandle& handle,
                                PartialTensorShape* shape) {
  if (!context->RankKnown(handle)) return absl::OkStatus();
  std::vector<int64_t> dims(context->Rank(handle));
  for (int32_t i = 0, end = dims.size(); i < end; ++i) {
    dims[i] = context->Value(context->Dim(handle, i));
  }
  return PartialTensorShape::MakePartialShape(dims.data(), dims.size(), shape);
}
Status PropagateShapes(Graph* graph,
                       const std::map<int, InferredShape>& arg_shapes,
                       const std::vector<BackEdgeHelper::BackEdge>& back_edges,
                       ShapeRefiner* shape_refiner) {
  std::map<const Node*, const Node*> merge_to_next_iteration;
  for (const auto& e : back_edges) {
    if (e.src->IsNextIteration() && e.dst->IsMerge()) {
      merge_to_next_iteration[e.dst] = e.src;
    }
  }
  std::vector<Node*> order;
  GetReversePostOrder(*graph, &order);
  for (Node* n : order) {
    VLOG(4) << "Propagating shape for node " << n->name()
            << ", type: " << n->type_string();
    Status status = shape_refiner->AddNode(n);
    if (!status.ok()) {
      VLOG(1) << "Shape inference failed for node " << n->name() << ": "
              << status;
    } else {
      shape_inference::InferenceContext* context = shape_refiner->GetContext(n);
      for (int i = 0; i < n->num_outputs(); i++) {
        shape_inference::ShapeHandle handle = context->output(i);
        VLOG(4) << "Output " << i << " for node " << n->name() << ": "
                << context->DebugString(handle);
      }
    }
    int index = -1;
    if (n->type_string() == "_Arg") {
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    } else if (n->type_string() == "Placeholder") {
      if (const auto s = GetNodeAttr(n->attrs(), "_index", &index); !s.ok()) {
        VLOG(1) << "Failed to get node index for node " << n->name();
      }
    }
    if (index >= 0) {
      if (auto it = arg_shapes.find(index); it != arg_shapes.end()) {
        const InferredShape& arg_shape = it->second;
        shape_inference::InferenceContext* context =
            shape_refiner->GetContext(n);
        if (arg_shape.handle_type != DT_INVALID) {
          shape_inference::ShapeHandle handle;
          TF_RETURN_IF_ERROR(context->MakeShapeFromPartialTensorShape(
              arg_shape.handle_shape, &handle));
          context->set_output_handle_shapes_and_types(
              0, std::vector<shape_inference::ShapeAndType>{
                     {handle, arg_shape.handle_type}});
        }
        shape_inference::ShapeHandle handle;
        TF_RETURN_IF_ERROR(
            context->MakeShapeFromPartialTensorShape(arg_shape.shape, &handle));
        TF_RETURN_IF_ERROR(shape_refiner->SetShape(n, 0, handle));
      }
    }
    if (n->type_string() == "VariableShape") {
      shape_inference::InferenceContext* context = shape_refiner->GetContext(n);
      auto handle_shapes_and_types = context->input_handle_shapes_and_types(0);
      if (handle_shapes_and_types && !handle_shapes_and_types->empty()) {
        shape_inference::ShapeHandle handle =
            handle_shapes_and_types->at(0).shape;
        TensorShapeProto shape_proto;
        context->ShapeHandleToProto(handle, &shape_proto);
        if (!shape_proto.unknown_rank()) {
          NodeDef const_def;
          const_def.set_op("Const");
          Node* var_node;
          TF_RETURN_IF_ERROR(n->input_node(0, &var_node));
          const_def.set_name(
              graph->NewName(absl::StrCat("var_shape_", var_node->name())));
          DataType dtype = n->output_type(0);
          AddNodeAttr("dtype", dtype, &const_def);
          TensorProto value;
          value.set_dtype(dtype);
          value.mutable_tensor_shape()->add_dim()->set_size(
              shape_proto.dim_size());
          for (const auto& dim : shape_proto.dim()) {
            if (dtype == DT_INT32) {
              value.add_int_val(dim.size());
            } else {
              value.add_int64_val(dim.size());
            }
          }
          AddNodeAttr("value", value, &const_def);
          for (auto const& attr : n->attrs()) {
            if (*attr.first.begin() == '_') {
              AddNodeAttr(attr.first, attr.second, &const_def);
            }
          }
          TF_ASSIGN_OR_RETURN(Node * const_node, graph->AddNode(const_def));
          graph->AddControlEdge(var_node, const_node);
          std::vector<const Edge*> out_edges(n->out_edges().begin(),
                                             n->out_edges().end());
          for (const Edge* e : out_edges) {
            if (e->IsControlEdge()) {
              graph->AddControlEdge(const_node, e->dst());
              graph->RemoveEdge(e);
            } else {
              Node* dst = e->dst();
              int dst_input = e->dst_input();
              graph->RemoveEdge(e);
              graph->AddEdge(const_node, 0, dst, dst_input);
            }
          }
        }
      }
    }
    if (n->IsMerge() && n->output_type(0) == DT_RESOURCE) {
      auto iter = merge_to_next_iteration.find(n);
      if (iter != merge_to_next_iteration.end()) {
        const Node *next_iter = iter->second, *node = next_iter;
        do {
          TF_RETURN_IF_ERROR(node->input_node(0, &node));
        } while (node->IsIdentity());
        const Node* switch_input;
        bool is_loop_invariant = node->IsSwitch() &&
                                 node->input_node(0, &switch_input).ok() &&
                                 switch_input == n;
        if (is_loop_invariant) {
          shape_inference::InferenceContext* context =
              shape_refiner->GetContext(n);
          for (int i = 0; i < n->num_inputs(); i++) {
            const Node* input_node;
            if (n->input_node(i, &input_node).ok()) {
              auto shapes_and_types = context->input_handle_shapes_and_types(i);
              if (shapes_and_types) {
                context->set_output_handle_shapes_and_types(0,
                                                            *shapes_and_types);
              }
              break;
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}
Status StoreOutputShapes(const Graph& graph, const ShapeRefiner& shape_refiner,
                         GraphShapeInfo* shape_info) {
  for (const Node* node : graph.nodes()) {
    shape_inference::InferenceContext* context = shape_refiner.GetContext(node);
    if (!context) continue;
    auto& outputs = (*shape_info)[node->name()];
    outputs.resize(context->num_outputs());
    for (int i = 0; i < context->num_outputs(); ++i) {
      auto& output = outputs[i];
      TF_RETURN_IF_ERROR(
          ShapeHandleToTensorShape(context, context->output(i), &output.shape));
      const auto* handle_shapes_and_types =
          context->output_handle_shapes_and_types(i);
      if (handle_shapes_and_types != nullptr) {
        if (handle_shapes_and_types->size() == 1) {
          TF_RETURN_IF_ERROR(ShapeHandleToTensorShape(
              context, (*handle_shapes_and_types)[0].shape,
              &output.handle_shape));
          output.handle_type = (*handle_shapes_and_types)[0].dtype;
        } else {
        }
      }
      VLOG(4) << node->name() << " output " << i << " shape"
              << output.shape.DebugString() << " handle_type "
              << DataTypeString(output.handle_type) << " handle_shape "
              << output.handle_shape.DebugString();
    }
  }
  return absl::OkStatus();
}
}  
Status InferShapes(Graph* graph, const std::map<int, InferredShape>& arg_shapes,
                   const tensorflow::FunctionLibraryDefinition* fnlib_def,
                   GraphShapeInfo* shape_info) {
  ShapeRefiner shape_refiner(graph->versions(), graph->op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  BackEdgeHelper back_edge;
  TF_RETURN_IF_ERROR(back_edge.Remove(graph));
  TF_RETURN_IF_ERROR(PropagateShapes(graph, arg_shapes,
                                     back_edge.RemovedEdges(), &shape_refiner));
  TF_RETURN_IF_ERROR(back_edge.Replace());
  return StoreOutputShapes(*graph, shape_refiner, shape_info);
}
absl::StatusOr<InferredShape> MergeInferredShapes(const InferredShape& a,
                                                  const InferredShape& b) {
  InferredShape result;
  TF_RETURN_IF_ERROR(a.shape.MergeWith(b.shape, &result.shape));
  if (a.handle_type == DT_INVALID) {
    result.handle_type = b.handle_type;
  } else if (b.handle_type == DT_INVALID) {
    result.handle_type = a.handle_type;
  } else if (a.handle_type == b.handle_type) {
    result.handle_type = a.handle_type;
  } else {
    return errors::InvalidArgument(
        "Mismatched resource types: ", DataTypeString(a.handle_type), " vs. ",
        DataTypeString(b.handle_type));
  }
  TF_RETURN_IF_ERROR(
      a.handle_shape.MergeWith(b.handle_shape, &result.handle_shape));
  return result;
}
}  