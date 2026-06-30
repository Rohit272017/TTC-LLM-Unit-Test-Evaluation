#include "tensorflow/lite/tools/gen_op_registration.h"
#include <algorithm>
#include <string>
#include <vector>
#include "re2/re2.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/schema/schema_utils.h"
namespace tflite {
string NormalizeCustomOpName(const string& op) {
  string method(op);
  RE2::GlobalReplace(&method, "([a-z])([A-Z])", "\\1_\\2");
  std::transform(method.begin(), method.end(), method.begin(), ::toupper);
  return method;
}
void ReadOpsFromModel(const ::tflite::Model* model,
                      tflite::RegisteredOpMap* builtin_ops,
                      tflite::RegisteredOpMap* custom_ops) {
  if (!model) return;
  auto opcodes = model->operator_codes();
  if (!opcodes) return;
  for (const auto* opcode : *opcodes) {
    const int version = opcode->version();
    auto builtin_code = GetBuiltinCode(opcode);
    if (builtin_code != ::tflite::BuiltinOperator_CUSTOM) {
      auto iter_and_bool = builtin_ops->insert(
          std::make_pair(tflite::EnumNameBuiltinOperator(builtin_code),
                         std::make_pair(version, version)));
      auto& versions = iter_and_bool.first->second;
      versions.first = std::min(versions.first, version);
      versions.second = std::max(versions.second, version);
    } else {
      auto iter_and_bool = custom_ops->insert(std::make_pair(
          opcode->custom_code()->c_str(), std::make_pair(version, version)));
      auto& versions = iter_and_bool.first->second;
      versions.first = std::min(versions.first, version);
      versions.second = std::max(versions.second, version);
    }
  }
}
}  