#include "tensorflow/core/grappler/optimizers/shape_optimizer.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
namespace grappler {
Status ShapeOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
  bool can_optimize = false;
  bool has_div = false;
  bool has_size = false;
  bool has_shape = false;
  bool has_prod = false;
  auto is_int = [](const NodeDef& node) -> bool {
    return node.attr().at("T").type() == DT_INT32 ||
           node.attr().at("T").type() == DT_INT64;
  };
  for (const NodeDef& node : item.graph.node()) {
    if (IsShape(node)) {
      has_shape = true;
    } else if (IsProd(node) && is_int(node)) {
      has_prod = true;
    } else if (IsDiv(node) && is_int(node)) {
      has_div = true;
    } else if (IsSize(node)) {
      has_size = true;
    }
    if ((has_shape && has_prod) || (has_div && has_size)) {
      can_optimize = true;
      break;
    }
  }
  if (!can_optimize) {
    return absl::AbortedError("Nothing to do.");
  }
  *optimized_graph = item.graph;
  GraphProperties properties(item);
  bool inferred_properties = false;
  {
    MutableGraphView graph(optimized_graph);
    for (auto& node : *optimized_graph->mutable_node()) {
      if (!IsShape(node)) {
        continue;
      }
      for (MutableGraphView::InputPort fanout :
           graph.GetFanout(MutableGraphView::OutputPort(&node, 0))) {
        if (fanout.node->op() != "Prod") {
          continue;
        }
        if (fanout.node->attr().count("keep_dims") != 0 &&
            fanout.node->attr().at("keep_dims").b()) {
          continue;
        }
        const MutableGraphView::OutputPort reduce_indices =
            graph.GetRegularFanin(MutableGraphView::InputPort(fanout.node, 1));
        if (!inferred_properties) {
          TF_RETURN_IF_ERROR(
              properties.InferStatically(false,
                                         false,
                                         false));
          inferred_properties = true;
        }
        const auto& prop =
            properties.GetOutputProperties(reduce_indices.node->name());
        const int prop_size = prop.size();
        if (prop_size <= reduce_indices.port_id) {
          continue;
        }
        const TensorShapeProto& reduction_indices_shape =
            prop[reduce_indices.port_id].shape();
        if (NumCoefficients(reduction_indices_shape) == 1) {
          const auto& input_props = properties.GetInputProperties(node.name());
          if (input_props.size() != 1) {
            continue;
          }
          NodeDef size_node(*fanout.node);
          const DataType type = input_props[0].dtype();
          size_node.set_op("Size");
          size_node.set_input(0, node.input(0));
          size_node.set_input(1, AsControlDependency(node));
          size_node.mutable_attr()->erase("Tidx");
          size_node.mutable_attr()->erase("keep_dims");
          (*size_node.mutable_attr())["out_type"] = fanout.node->attr().at("T");
          (*size_node.mutable_attr())["T"].set_type(type);
          size_node.set_device(node.device());
          Status s = IsKernelRegisteredForNode(size_node);
          if (!s.ok()) {
            continue;
          }
          fanout.node->Swap(&size_node);
        }
      }
    }
  }
  {
    MutableGraphView graph(optimized_graph);
    for (auto& node : *optimized_graph->mutable_node()) {
      if (node.op() == "Div") {
        const MutableGraphView::OutputPort input1 =
            graph.GetRegularFanin(MutableGraphView::InputPort(&node, 0));
        const MutableGraphView::OutputPort input2 =
            graph.GetRegularFanin(MutableGraphView::InputPort(&node, 1));
        if (input1.node == nullptr || input2.node == nullptr) continue;
        if (!IsSize(*input1.node) || !IsSize(*input2.node)) {
          continue;
        }
        if (!inferred_properties) {
          TF_RETURN_IF_ERROR(
              properties.InferStatically(false,
                                         false,
                                         false));
          inferred_properties = true;
        }
        const auto& prop1 = properties.GetInputProperties(input1.node->name());
        const auto& prop2 = properties.GetInputProperties(input2.node->name());
        if (prop1.size() != 1 || prop2.size() != 1) {
          continue;
        }
        const TensorShapeProto& shape1 = prop1[0].shape();
        const TensorShapeProto& shape2 = prop2[0].shape();
        int64_t result = ComputeSizeRatio(shape1, shape2);
        if (result >= 0) {
          node.set_op("Const");
          DataType dtype = node.attr().at("T").type();
          node.mutable_attr()->erase("T");
          (*node.mutable_attr())["dtype"].set_type(dtype);
          TensorProto* t = (*node.mutable_attr())["value"].mutable_tensor();
          t->set_dtype(dtype);
          *t->mutable_tensor_shape() = TensorShapeProto();
          if (dtype == DT_INT32) {
            t->add_int_val(result);
          } else {
            t->add_int64_val(result);
          }
          node.set_input(0, AsControlDependency(node.input(0)));
          node.set_input(1, AsControlDependency(node.input(1)));
        }
      }
    }
  }
  return absl::OkStatus();
}
}  
}  