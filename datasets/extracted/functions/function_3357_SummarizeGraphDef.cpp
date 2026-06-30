#include "tensorflow/core/framework/graph_def_util.h"
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
namespace tensorflow {
string SummarizeGraphDef(const GraphDef& graph_def) {
  string ret;
  strings::StrAppend(
      &ret, "versions = ", graph_def.versions().ShortDebugString(), ";\n");
  for (const NodeDef& node : graph_def.node()) {
    strings::StrAppend(&ret, SummarizeNodeDef(node), ";\n");
  }
  return ret;
}
Status ValidateExternalGraphDefSyntax(const GraphDef& graph_def) {
  for (const NodeDef& node : graph_def.node()) {
    TF_RETURN_IF_ERROR(ValidateExternalNodeDefSyntax(node));
  }
  return absl::OkStatus();
}
Status AddDefaultAttrsToGraphDef(GraphDef* graph_def,
                                 const OpRegistryInterface& op_registry,
                                 int node_offset) {
  return AddDefaultAttrsToGraphDef(graph_def, op_registry, node_offset, false);
}
Status AddDefaultAttrsToGraphDef(GraphDef* graph_def,
                                 const OpRegistryInterface& op_registry,
                                 int node_offset, bool skip_unknown_ops) {
  if (node_offset > graph_def->node_size()) {
    return errors::InvalidArgument(
        "Tried to add default attrs to GraphDef "
        "starting at offset ",
        node_offset, " with total nodes in graph: ", graph_def->node_size());
  }
  for (int i = node_offset; i < graph_def->node_size(); ++i) {
    NodeDef* node_def = graph_def->mutable_node(i);
    const OpDef* op_def;
    Status s = op_registry.LookUpOpDef(node_def->op(), &op_def);
    if (s.ok()) {
      AddDefaultsToNodeDef(*op_def, node_def);
    } else if (!skip_unknown_ops) {
      return s;
    }
  }
  return absl::OkStatus();
}
static Status RemoveNewDefaultAttrsFromNodeDef(
    NodeDef* node_def, const OpRegistryInterface& consumer_op_registry,
    const OpRegistryInterface& producer_op_registry,
    std::set<std::pair<string, string>>* op_attr_removed) {
  const OpDef* producer_op_def;
  const OpDef* consumer_op_def;
  TF_RETURN_IF_ERROR(
      producer_op_registry.LookUpOpDef(node_def->op(), &producer_op_def));
  TF_RETURN_IF_ERROR(
      consumer_op_registry.LookUpOpDef(node_def->op(), &consumer_op_def));
  std::vector<string> to_remove;
  for (const auto& attr : node_def->attr()) {
    if (!absl::StartsWith(attr.first, "_") &&
        FindAttr(attr.first, *consumer_op_def) == nullptr) {
      const OpDef::AttrDef* producer_attr_def =
          FindAttr(attr.first, *producer_op_def);
      if (producer_attr_def == nullptr) {
        return errors::InvalidArgument(
            "Attr '", attr.first,
            "' missing in producer's OpDef: ", SummarizeOpDef(*producer_op_def),
            " but found in node: ", FormatNodeDefForError(*node_def));
      }
      if (producer_attr_def->has_default_value() &&
          AreAttrValuesEqual(producer_attr_def->default_value(), attr.second)) {
        to_remove.emplace_back(attr.first);
      }
    }
  }
  for (const string& attr_name : to_remove) {
    node_def->mutable_attr()->erase(attr_name);
    if (op_attr_removed != nullptr) {
      op_attr_removed->insert(std::make_pair(node_def->op(), attr_name));
    }
  }
  return absl::OkStatus();
}
static bool IsFunction(const GraphDef& graph_def, const string& op_name) {
  for (const auto& func_def : graph_def.library().function()) {
    if (op_name == func_def.signature().name()) return true;
  }
  return false;
}
Status RemoveNewDefaultAttrsFromGraphDef(
    GraphDef* graph_def, const OpRegistryInterface& consumer_op_registry,
    const OpRegistryInterface& producer_op_registry,
    std::set<std::pair<string, string>>* op_attr_removed) {
  for (int n = 0; n < graph_def->node_size(); ++n) {
    NodeDef* node_def = graph_def->mutable_node(n);
    if (!IsFunction(*graph_def, node_def->op())) {
      TF_RETURN_IF_ERROR(RemoveNewDefaultAttrsFromNodeDef(
          node_def, consumer_op_registry, producer_op_registry,
          op_attr_removed));
    }
  }
  for (int f = 0; f < graph_def->library().function_size(); ++f) {
    FunctionDef* func_def = graph_def->mutable_library()->mutable_function(f);
    for (int n = 0; n < func_def->node_def_size(); ++n) {
      NodeDef* node_def = func_def->mutable_node_def(n);
      if (!IsFunction(*graph_def, node_def->op())) {
        TF_RETURN_IF_ERROR(RemoveNewDefaultAttrsFromNodeDef(
            node_def, consumer_op_registry, producer_op_registry,
            op_attr_removed));
      }
    }
  }
  return absl::OkStatus();
}
void StripDefaultAttributes(const OpRegistryInterface& op_registry,
                            protobuf::RepeatedPtrField<NodeDef>* nodes) {
  for (int i = 0; i < nodes->size(); ++i) {
    NodeDef* node = nodes->Mutable(i);
    const OpDef* op_def;
    const OpRegistrationData* op_reg_data = nullptr;
    Status s = op_registry.LookUp(node->op(), &op_reg_data);
    if (!s.ok()) {
      VLOG(1) << "Ignoring encountered unknown operation "
              << SummarizeNodeDef(*node)
              << " when stripping default attributes. It is likely a function, "
                 "in which case ignoring it is fine";
      continue;
    }
    op_def = &op_reg_data->op_def;
    for (const OpDef::AttrDef& attr_def : op_def->attr()) {
      if (attr_def.has_default_value()) {
        AttrValueMap* attrs = node->mutable_attr();
        const string& name = attr_def.name();
        auto iter = attrs->find(name);
        if (iter != attrs->end()) {
          const AttrValue& default_value = attr_def.default_value();
          if (AreAttrValuesEqual(iter->second, default_value,
                                 true)) {
            attrs->erase(name);
          }
        }
      }
    }
  }
}
void OpsUsedByGraph(const GraphDef& graph_def,
                    std::set<string>* ops_used_in_graph) {
  std::unordered_map<string, const FunctionDef*> name_to_function;
  for (const auto& function : graph_def.library().function()) {
    name_to_function.insert(
        std::make_pair(function.signature().name(), &function));
  }
  std::set<string> used_ops;  
  std::vector<const FunctionDef*> functions_to_process;  
  const auto mark_op_as_used = [&used_ops, &functions_to_process,
                                &name_to_function](const string& op) {
    if (used_ops.insert(op).second) {
      const auto it = name_to_function.find(op);
      if (it != name_to_function.end()) {
        functions_to_process.push_back(it->second);
      }
    }
  };
  for (const auto& node : graph_def.node()) {
    mark_op_as_used(node.op());
  }
  while (!functions_to_process.empty()) {
    const FunctionDef* fun = functions_to_process.back();
    functions_to_process.pop_back();
    for (const auto& node : fun->node_def()) {
      mark_op_as_used(node.op());
    }
  }
  ops_used_in_graph->clear();
  for (const string& op_name : used_ops) {
    if (name_to_function.find(op_name) == name_to_function.end()) {
      ops_used_in_graph->insert(op_name);
    }
  }
}
Status StrippedOpListForGraph(const GraphDef& graph_def,
                              const OpRegistryInterface& op_registry,
                              OpList* stripped_op_list) {
  std::set<string> used_ops;
  OpsUsedByGraph(graph_def, &used_ops);
  stripped_op_list->clear_op();
  for (const string& op_name : used_ops) {
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(op_registry.LookUpOpDef(op_name, &op_def));
    OpDef* stripped_op = stripped_op_list->add_op();
    stripped_op->CopyFrom(*op_def);
    RemoveDescriptionsFromOpDef(stripped_op);
  }
  return absl::OkStatus();
}
}  