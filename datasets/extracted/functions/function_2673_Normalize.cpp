#include "tensorflow/core/framework/graph_to_functiondef.h"
#include <memory>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
namespace tensorflow {
namespace {
class NodeNameMapping {
 public:
  NodeNameMapping() = default;
  string GetInputName(const string& name);
  string GetOutputName(const string& name);
  string Uniquify(const string& name);
  Status UseOutputName(const string& name);
  string Lookup(const string& name) const;
 private:
  string UniquifyHelper(const string& name);
  static string Normalize(string name);
  absl::flat_hash_map<string, uint64> used_names_;
  absl::flat_hash_map<string, string> name_mapping_;
};
string NodeNameMapping::Normalize(string name) {
  if (name.empty()) return "unknown";
  const int n = name.size();
  for (int i = 0; i < n; ++i) {
    char c = name[i];
    if (isalnum(c)) {
      if (isupper(c)) {
        name[i] = tolower(c);
      }
    } else {
      name[i] = '_';
    }
  }
  int i = 0;
  for (; i < n; ++i) {
    if (isalpha(name[i])) break;
  }
  return i == n ? "unknown" : name.substr(i);
}
string NodeNameMapping::UniquifyHelper(const string& name) {
  auto it = used_names_.emplace(name, 0);
  if (it.second) return name;
  while (true) {
    const string candidate = strings::StrCat(name, "_", it.first->second);
    it.first->second++;
    if (used_names_.emplace(candidate, 0).second) return candidate;
  }
}
string NodeNameMapping::GetInputName(const string& name) {
  const string& input_name = UniquifyHelper(Normalize(name));
  name_mapping_[name] = input_name;
  return input_name;
}
string NodeNameMapping::GetOutputName(const string& name) {
  const string& input_name = UniquifyHelper(Normalize(name));
  return input_name;
}
string NodeNameMapping::Uniquify(const string& name) {
  const string uniqued = UniquifyHelper(name);
  name_mapping_[name] = uniqued;
  return uniqued;
}
Status NodeNameMapping::UseOutputName(const string& name) {
  const auto& iter = used_names_.find(name);
  if (iter != used_names_.end()) {
    return errors::InvalidArgument(
        "Cannot have duplicate output names. Name '", name,
        "' appears more than once in 'output_names' array.");
  }
  used_names_.emplace(name, 0);
  return absl::OkStatus();
}
string NodeNameMapping::Lookup(const string& name) const {
  const auto iter = name_mapping_.find(name);
  if (iter == name_mapping_.end()) return string();
  return iter->second;
}
Status FillFunctionBody(
    const string& fn_name, const NodeNameMapping& node_names,
    const std::vector<const Node*>& body_nodes,
    const absl::flat_hash_map<string, string>& tensor_renaming,
    bool set_stateful_from_nodes, bool copy_placeholder_attrs_from_nodes,
    bool allow_destructive_reads, FunctionDef* fdef) {
  absl::flat_hash_set<string> func_attr_names;
  for (const auto& func_attr : fdef->signature().attr()) {
    func_attr_names.insert(func_attr.name());
  }
  std::vector<const Edge*> in_edges;
  std::vector<const Edge*> control_edges;
  for (const Node* node : body_nodes) {
    NodeDef* node_def = fdef->add_node_def();
    NodeDebugInfo debug_info(node->def());
    if (allow_destructive_reads) {
      Node* mutable_node = const_cast<Node*>(node);
      *node_def->mutable_op() =
          node->def()
              .op();  
      *node_def->mutable_attr() =
          std::move(*mutable_node->mutable_def()->mutable_attr());
      if (node->def().has_experimental_debug_info()) {
        *node_def->mutable_experimental_debug_info() = std::move(
            *mutable_node->mutable_def()->mutable_experimental_debug_info());
      }
      if (node->def().has_experimental_type()) {
        *node_def->mutable_experimental_type() = std::move(
            *mutable_node->mutable_def()->mutable_experimental_type());
      }
    } else {
      *node_def = node->def();
      MergeDebugInfo(NodeDebugInfo(node->def()), node_def);
      node_def->clear_input();
    }
    if (!node->assigned_device_name().empty()) {
      node_def->set_device(node->assigned_device_name());
    }
    node_def->set_name(node_names.Lookup(node->name()));
    in_edges.clear();
    in_edges.resize(node->num_inputs(), nullptr);
    control_edges.clear();
    for (const Edge* edge : node->in_edges()) {
      if (edge->src()->IsSource()) continue;
      if (edge->IsControlEdge()) {
        control_edges.push_back(edge);
      } else {
        in_edges[edge->dst_input()] = edge;
      }
    }
    std::sort(control_edges.begin(), control_edges.end(),
              [](const Edge* a, const Edge* b) {
                return a->src()->name() < b->src()->name();
              });
    for (size_t i = 0; i < in_edges.size(); ++i) {
      const Edge* edge = in_edges[i];
      std::string original_input_name;
      if (edge == nullptr) {
        if (i >= node->requested_inputs().size()) {
          return errors::InvalidArgument(
              "Graph to be converted to function appears to be malformed. ",
              "Node ", node->name(), " is missing input edge ", i);
        }
        original_input_name =
            ParseTensorName(node->requested_inputs()[i]).ToString();
      } else {
        original_input_name =
            strings::StrCat(edge->src()->name(), ":", edge->src_output());
      }
      const auto iter = tensor_renaming.find(original_input_name);
      if (iter == tensor_renaming.end()) {
        return errors::InvalidArgument(
            "Input ", i, ", '", original_input_name, "', of node '",
            node->name(), "' in function '", fn_name,
            "' is not available. You might need to include it in inputs "
            "or include its source node in the body");
      }
      node_def->add_input(iter->second);
    }
    for (const Edge* edge : control_edges) {
      const string normalized = node_names.Lookup(edge->src()->name());
      if (normalized.empty()) {
        return errors::InvalidArgument(
            "The source of control edge ", edge->DebugString(),
            " is not in the body. Encountered while creating function '",
            fn_name, "'");
      }
      node_def->add_input(strings::StrCat("^", normalized));
    }
    if (set_stateful_from_nodes && node->op_def().is_stateful()) {
      fdef->mutable_signature()->set_is_stateful(true);
    }
    if (!copy_placeholder_attrs_from_nodes) {
      continue;
    }
    for (const auto& iter : node_def->attr()) {
      if (iter.second.placeholder().empty()) {
        continue;
      }
      const std::string& func_attr_name = iter.second.placeholder();
      if (func_attr_names.find(func_attr_name) != func_attr_names.end()) {
        continue;
      }
      const std::string& node_attr_name = iter.first;
      const OpDef::AttrDef* node_attr_def = nullptr;
      for (const auto& node_attr : node->op_def().attr()) {
        if (node_attr.name() == node_attr_name) {
          node_attr_def = &node_attr;
        }
      }
      if (!node_attr_def) {
        return errors::Unimplemented(
            "Placeholder value is not supported for attributes not in OpDef. "
            "Attribute: ",
            node_attr_name, ", OpDef: ", node->op_def().DebugString());
      }
      OpDef::AttrDef* attr_def = fdef->mutable_signature()->add_attr();
      attr_def->set_name(func_attr_name);
      attr_def->set_type(node_attr_def->type());
      func_attr_names.insert(func_attr_name);
    }
  }
  return absl::OkStatus();
}
Status GraphToFunctionDefHelper(
    const Graph& fn_body, const string& fn_name, bool append_hash_to_fn_name,
    bool set_stateful_from_nodes, bool copy_placeholder_attrs_from_nodes,
    const std::vector<const Node*>& body_nodes,
    const std::vector<OutputTensor>& inputs,
    const std::vector<OutputTensor>& outputs,
    const std::vector<string>& output_names,
    const std::vector<const Node*>& control_outputs,
    const std::vector<string>& control_output_names, const char* description,
    bool allow_destructive_reads, FunctionDef* fdef) {
  if (!output_names.empty()) {
    DCHECK_EQ(output_names.size(), outputs.size());
  }
  if (description != nullptr) {
    fdef->mutable_signature()->set_description(description);
  }
  NodeNameMapping node_names;
  absl::flat_hash_map<string, string> tensor_renaming;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const Node* node = outputs[i].node;
    int idx = outputs[i].index;
    OpDef::ArgDef* argdef = fdef->mutable_signature()->add_output_arg();
    if (node->IsRetval()) {
      argdef->set_type(node->input_type(idx));
    } else {
      argdef->set_type(node->output_type(idx));
    }
    if (!output_names.empty()) {
      TF_RETURN_IF_ERROR(node_names.UseOutputName(output_names[i]));
      argdef->set_name(output_names[i]);
    } else {
      argdef->set_name(node_names.GetOutputName(node->name()));
    }
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Node* node = inputs[i].node;
    int idx = inputs[i].index;
    OpDef::ArgDef* argdef = fdef->mutable_signature()->add_input_arg();
    argdef->set_type(node->output_type(idx));
    const string& input_name = node_names.GetInputName(node->name());
    argdef->set_name(input_name);
    FunctionDef::ArgAttrs arg_attrs;
    int64_t resource_arg_unique_id = -1;
    for (const auto& attr : node->attrs()) {
      if (absl::StartsWith(attr.first, "_")) {
        arg_attrs.mutable_attr()->insert(attr);
      } else if (attr.first == "shape" && argdef->type() != DT_RESOURCE) {
        AttrValue value;
        *(value.mutable_list()->add_shape()) = attr.second.shape();
        arg_attrs.mutable_attr()->insert({"_output_shapes", value});
      } else if (attr.first == "value" && node->type_string() == "Const") {
        AttrValue value;
        *(value.mutable_list()->add_shape()) =
            attr.second.tensor().tensor_shape();
        arg_attrs.mutable_attr()->insert({"_output_shapes", value});
      }
      if (attr.first == "_resource_arg_unique_id") {
        resource_arg_unique_id = attr.second.i();
      }
    }
    if (arg_attrs.attr_size() > 0) {
      (*fdef->mutable_arg_attr())[i] = std::move(arg_attrs);
    }
    if (resource_arg_unique_id >= 0) {
      (*fdef->mutable_resource_arg_unique_id())[idx] = resource_arg_unique_id;
    }
    tensor_renaming[strings::StrCat(node->name(), ":", idx)] = input_name;
  }
  for (const Node* node : body_nodes) {
    const string& node_name = node_names.Uniquify(node->name());
    NameRangeMap output_ranges;
    TF_RETURN_IF_ERROR(
        NameRangesForNode(*node, node->op_def(), nullptr, &output_ranges));
    for (const auto& output : output_ranges) {
      const StringPiece& output_name = output.first;
      int index_start = output.second.first;
      int index_end = output.second.second;
      for (int i = index_start; i < index_end; ++i) {
        const string& original_name = strings::StrCat(node->name(), ":", i);
        const string& new_name =
            strings::StrCat(node_name, ":", output_name, ":", i - index_start);
        if (tensor_renaming.find(original_name) == tensor_renaming.end()) {
          tensor_renaming[original_name] = new_name;
        }
      }
    }
  }
  TF_RETURN_IF_ERROR(FillFunctionBody(
      fn_name, node_names, body_nodes, tensor_renaming, set_stateful_from_nodes,
      copy_placeholder_attrs_from_nodes, allow_destructive_reads, fdef));
  for (int r = 0; r < fdef->signature().output_arg_size(); ++r) {
    const string& ret_name = fdef->signature().output_arg(r).name();
    string return_value;
    if (outputs[r].node->IsRetval()) {
      Edge const* edge;
      TF_RETURN_IF_ERROR(outputs[r].node->input_edge(0, &edge));
      return_value =
          strings::StrCat(edge->src()->name(), ":", edge->src_output());
    } else {
      return_value =
          strings::StrCat(outputs[r].node->name(), ":", outputs[r].index);
    }
    const auto iter = tensor_renaming.find(return_value);
    if (iter == tensor_renaming.end()) {
      return errors::InvalidArgument(
          "TF_Output ", return_value, " is neither in the function body ",
          "nor among function inputs. Encountered while creating function '",
          fn_name, "'");
    }
    (*fdef->mutable_ret())[ret_name] = iter->second;
  }
  if (append_hash_to_fn_name) {
    const uint64 hash = FunctionDefHash(*fdef);
    string encoded;
    TF_RETURN_IF_ERROR(Base64Encode(
        StringPiece(reinterpret_cast<const char*>(&hash), sizeof(hash)),
        &encoded));
    std::replace(encoded.begin(), encoded.end(), '-', 'a');
    std::replace(encoded.begin(), encoded.end(), '_', 'A');
    fdef->mutable_signature()->set_name(strings::StrCat(fn_name, "_", encoded));
  } else {
    fdef->mutable_signature()->set_name(fn_name);
  }
  if (!control_output_names.empty() &&
      (control_outputs.size() != control_output_names.size())) {
    return errors::InvalidArgument(
        "Expected number of control outputs (", control_outputs.size(),
        ") and the number of control output names (",
        control_output_names.size(), ") to match but they do not.");
  }
  std::set<string> control_output_names_set;
  for (int i = 0; i < control_outputs.size(); ++i) {
    string signature_name;
    if (!control_output_names.empty()) {
      signature_name = control_output_names[i];
    } else {
      signature_name = control_outputs[i]->name();
    }
    if (signature_name.empty()) {
      return errors::InvalidArgument("Control output name must be not empty");
    }
    if (!control_output_names_set.insert(signature_name).second) {
      return errors::InvalidArgument("Repeated control output name: ",
                                     signature_name);
    }
    const string control_output_node =
        node_names.Lookup(control_outputs[i]->name());
    if (control_output_node.empty()) {
      return errors::InvalidArgument(
          "Control output node name must be not empty");
    }
    (*fdef->mutable_control_ret())[signature_name] = control_output_node;
  }
  for (const string& control_output : control_output_names_set) {
    fdef->mutable_signature()->add_control_output(control_output);
  }
  return absl::OkStatus();
}
Status GraphToFunctionDefHelper(
    const Graph& graph, const string& name,
    const std::function<absl::optional<string>(const Node*)>& control_ret,
    const std::vector<string>& output_names, bool allow_destructive_reads,
    FunctionDef* fdef) {
  auto add_arg_or_retval = [](Node* node,
                              std::vector<OutputTensor>* args_or_retvals) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
    if (index >= args_or_retvals->size()) {
      args_or_retvals->resize(index + 1);
    }
    if ((*args_or_retvals)[index].node == nullptr) {
      (*args_or_retvals)[index].node = node;
    } else {
      return errors::InvalidArgument(
          "Multiple '", node->type_string(), "' nodes found with index ", index,
          "; originally we already have:\n",
          (*args_or_retvals)[index].node->DebugString(), "\nNow we have:\n",
          node->DebugString());
    }
    return absl::OkStatus();
  };
  std::vector<const Node*> body_nodes;
  std::vector<OutputTensor> inputs;
  std::vector<OutputTensor> outputs;
  std::vector<const Node*> control_outputs;
  std::vector<string> control_output_names;
  for (Node* node : graph.op_nodes()) {
    if (node->IsArg()) {
      TF_RETURN_IF_ERROR(add_arg_or_retval(node, &inputs));
      continue;
    }
    if (node->IsRetval()) {
      TF_RETURN_IF_ERROR(add_arg_or_retval(node, &outputs));
      continue;
    }
    if (control_ret) {
      auto control_ret_name = control_ret(node);
      if (control_ret_name.has_value()) {
        control_outputs.push_back(node);
        control_output_names.push_back(control_ret_name.value());
      }
    }
    body_nodes.push_back(node);
  }
  auto validate_args_retvals =
      [](const std::vector<OutputTensor>& args_or_retvals,
         const string& op_type) {
        for (int i = 0, e = args_or_retvals.size(); i < e; ++i) {
          if (args_or_retvals[i].node == nullptr) {
            return errors::InvalidArgument("Missing '", op_type,
                                           "' node at index ", i);
          }
        }
        return absl::OkStatus();
      };
  TF_RETURN_IF_ERROR(validate_args_retvals(inputs, "_Arg"));
  TF_RETURN_IF_ERROR(validate_args_retvals(outputs, "_Retval"));
  return GraphToFunctionDefHelper(
      graph, name, false,
      false,
      false, body_nodes, inputs, outputs,
      output_names, control_outputs, control_output_names,
      nullptr, allow_destructive_reads, fdef);
}
}  
Status GraphToFunctionDef(const Graph& fn_body, const string& fn_name,
                          bool append_hash_to_fn_name,
                          bool set_stateful_from_nodes,
                          bool copy_placeholder_attrs_from_nodes,
                          const std::vector<const Node*>& body_nodes,
                          const std::vector<OutputTensor>& inputs,
                          const std::vector<OutputTensor>& outputs,
                          const std::vector<string>& output_names,
                          const std::vector<const Node*>& control_outputs,
                          const std::vector<string>& control_output_names,
                          const char* description, FunctionDef* fdef) {
  return GraphToFunctionDefHelper(
      fn_body, fn_name, append_hash_to_fn_name, set_stateful_from_nodes,
      copy_placeholder_attrs_from_nodes, body_nodes, inputs, outputs,
      output_names, control_outputs, control_output_names, description,
      false, fdef);
  return absl::OkStatus();
}
Status GraphToFunctionDef(
    const Graph& graph, const string& name,
    const std::function<absl::optional<string>(const Node*)>& control_ret,
    FunctionDef* fdef) {
  return GraphToFunctionDefHelper(graph, name, control_ret,
                                  {},
                                  false, fdef);
}
Status GraphToFunctionDef(const Graph& graph, const string& name,
                          FunctionDef* fdef) {
  return GraphToFunctionDef(graph, name, nullptr, fdef);
}
Status GraphToFunctionDef(const Graph& graph, const string& name,
                          const std::vector<std::string>& output_names,
                          FunctionDef* fdef) {
  return GraphToFunctionDefHelper(graph, name, nullptr,
                                  output_names,
                                  false, fdef);
}
Status GraphToFunctionDef(
    std::unique_ptr<Graph> graph, const string& name,
    const std::function<std::optional<string>(const Node*)>& control_ret,
    FunctionDef* fdef) {
  return GraphToFunctionDefHelper(*graph, name, control_ret,
                                  {},
                                  true, fdef);
}
}  