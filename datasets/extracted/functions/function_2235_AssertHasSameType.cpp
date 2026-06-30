#include "tensorflow/lite/tools/tool_params.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "tensorflow/lite/tools/logging.h"
namespace tflite {
namespace tools {
void ToolParam::AssertHasSameType(ToolParam::ParamType a,
                                  ToolParam::ParamType b) {
  TFLITE_TOOLS_CHECK(a == b) << "Type mismatch while accessing parameter.";
}
template <>
ToolParam::ParamType ToolParam::GetValueType<int32_t>() {
  return ToolParam::ParamType::TYPE_INT32;
}
template <>
ToolParam::ParamType ToolParam::GetValueType<bool>() {
  return ToolParam::ParamType::TYPE_BOOL;
}
template <>
ToolParam::ParamType ToolParam::GetValueType<float>() {
  return ToolParam::ParamType::TYPE_FLOAT;
}
template <>
ToolParam::ParamType ToolParam::GetValueType<std::string>() {
  return ToolParam::ParamType::TYPE_STRING;
}
void ToolParams::AssertParamExists(const std::string& name) const {
  TFLITE_TOOLS_CHECK(HasParam(name)) << name << " was not found.";
}
void ToolParams::Set(const ToolParams& other) {
  for (const auto& param : params_) {
    const ToolParam* other_param = other.GetParam(param.first);
    if (other_param == nullptr) continue;
    param.second->Set(*other_param);
  }
}
void ToolParams::Merge(const ToolParams& other, bool overwrite) {
  for (const auto& one : other.params_) {
    auto it = params_.find(one.first);
    if (it == params_.end()) {
      AddParam(one.first, one.second->Clone());
    } else if (overwrite) {
      it->second->Set(*one.second);
    }
  }
}
}  
}  