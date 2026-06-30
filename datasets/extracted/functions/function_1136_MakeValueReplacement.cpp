#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.h"
#include <any>
#include <string>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
namespace tflite {
namespace gpu {
namespace gl {
namespace {
std::pair<std::string, std::string> MakeValueReplacement(int n, int k) {
  return {absl::StrCat("value_", n), absl::StrCat("value_", k)};
}
std::pair<std::string, std::string> MakeDataReplacement(int n, int k) {
  return {absl::StrCat("input_data_", n), absl::StrCat("input_data_", k)};
}
}  
TransformResult FuseAutoInput::ApplyToNode(Node* node, GraphFloat32* graph) {
  auto& node_attr =
      std::any_cast<CompiledNodeAttributes&>(node->operation.attributes);
  auto& node_code = node_attr.code;
  if (node_code.input != IOStructure::AUTO) {
    return {TransformStatus::SKIPPED, ""};
  }
  uint3 workgroup = node_code.workgroup;
  auto node_outputs = graph->FindOutputs(node->id);
  std::vector<std::pair<Node*, int>> nodes_to_fuse;
  std::vector<std::pair<ValueId, int>> input_values;
  int input_num = -1;
  for (auto input_value : graph->FindInputs(node->id)) {
    input_num++;
    const ValueId input_id = input_value->id;
    input_values.push_back({input_id, input_num});
    if (graph->FindConsumers(input_id).size() > 1) {
      continue;  
    }
    Node* input_producer = graph->FindProducer(input_id);
    if (input_producer == nullptr) {
      continue;  
    }
    if (graph->FindOutputs(input_producer->id).size() != 1) {
      continue;  
    }
    auto& input_producer_attr = std::any_cast<const CompiledNodeAttributes&>(
        input_producer->operation.attributes);
    if (input_producer_attr.code.output != IOStructure::AUTO) {
      continue;
    }
    if (input_producer_attr.code.workload != node_code.workload &&
        uint3() != input_producer_attr.code.workload) {
      continue;
    }
    if (input_producer_attr.code.workgroup != uint3()) {
      if (workgroup != uint3()) {
        continue;
      }
      workgroup = input_producer_attr.code.workgroup;
    }
    nodes_to_fuse.push_back({input_producer, input_num});
    input_values.pop_back();  
  }
  if (nodes_to_fuse.empty()) {
    return {TransformStatus::SKIPPED, ""};
  }
  {
    absl::flat_hash_set<ValueId> all_inputs;
    for (const auto& node_to_fuse : nodes_to_fuse) {
      for (const auto& input : graph->FindInputs(node_to_fuse.first->id)) {
        if (all_inputs.find(input->id) != all_inputs.end()) {
          return {TransformStatus::SKIPPED, ""};
        }
        all_inputs.insert(input->id);
      }
    }
    for (const auto& input : graph->FindInputs(node->id)) {
      if (all_inputs.find(input->id) != all_inputs.end()) {
        return {TransformStatus::SKIPPED, ""};
      }
      all_inputs.insert(input->id);
    }
  }
  for (auto value : graph->FindInputs(node->id)) {
    if (!graph->RemoveConsumer(node->id, value->id).ok()) {
      return {TransformStatus::INVALID, ""};
    }
  }
  std::string operation_type;
  std::string source_code;
  std::string values;
  std::swap(source_code, node_code.source_code);
  int extra_input_num = input_num;
  input_num = 0;
  for (auto input_and_num : nodes_to_fuse) {
    auto& input = input_and_num.first;
    auto& attr =
        std::any_cast<CompiledNodeAttributes&>(input->operation.attributes);
    auto super_inputs = graph->FindInputs(input->id);
    std::vector<std::pair<std::string, std::string>> replacements;
    for (int i = 0; i < super_inputs.size(); ++i) {
      int value_index = i == 0 ? input_and_num.second : ++extra_input_num;
      replacements.push_back(MakeValueReplacement(i, value_index));
      replacements.push_back(MakeDataReplacement(i, input_num));
      if (attr.code.input == IOStructure::AUTO) {
        absl::StrAppend(&values, "  value_", value_index, " = $input_data_",
                        input_num, "[gid.x, gid.y, gid.z]$;\n");
      }
      if (!graph->AddConsumer(node->id, super_inputs[i]->id).ok()) {
        return {TransformStatus::INVALID, ""};
      }
      input_num++;
    }
    for (auto& param : attr.code.parameters) {
      param.name = absl::StrReplaceAll(param.name, replacements);
    }
    attr.code.source_code =
        absl::StrReplaceAll(attr.code.source_code, replacements);
    if (!MergeCode(&attr, &node_attr).ok()) {
      return {TransformStatus::INVALID, "Unable to merge the code"};
    }
    absl::StrAppend(&node_attr.code.source_code, "{\n", attr.code.source_code,
                    "\n}");
    if (!operation_type.empty()) {
      operation_type += ",";
    }
    operation_type += input->operation.type;
    if (!graph->DeleteNode(input->id).ok()) {
      return {TransformStatus::INVALID, ""};
    }
  }
  for (int i = 0; i < input_values.size(); i++) {
    if (node_code.input == IOStructure::AUTO) {
      absl::StrAppend(&values, "  value_", input_values[i].second,
                      " = $input_data_", input_num,
                      "[gid.x, gid.y, gid.z]$;\n");
    }
    if (!graph->AddConsumer(node->id, input_values[i].first).ok()) {
      return {TransformStatus::INVALID, ""};
    }
    input_num++;
  }
  node_code.input = IOStructure::ONLY_DEFINITIONS;
  absl::StrAppend(&node->operation.type, "(", operation_type, ")");
  node_code.source_code =
      absl::StrCat(values, node_code.source_code, "{
                   node->operation.type, "\n", source_code, "\n}");
  return {TransformStatus::APPLIED, ""};
}
}  
}  
}  