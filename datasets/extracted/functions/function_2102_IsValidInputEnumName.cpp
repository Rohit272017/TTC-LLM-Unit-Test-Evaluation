#include "tensorflow/lite/schema/builtin_ops_header/generator.h"
#include <string>
#include "tensorflow/lite/schema/schema_generated.h"
namespace tflite {
namespace builtin_ops_header {
namespace {
const char* kFileHeader =
    R"(
#ifndef TENSORFLOW_LITE_BUILTIN_OPS_H_
#define TENSORFLOW_LITE_BUILTIN_OPS_H_
#ifdef __cplusplus
extern "C" {
#endif  
typedef enum {
)";
const char* kFileFooter =
    R"(} TfLiteBuiltinOperator;
#ifdef __cplusplus
}  
#endif  
#endif  
)";
}  
bool IsValidInputEnumName(const std::string& name) {
  const char* begin = name.c_str();
  const char* ch = begin;
  while (*ch != '\0') {
    if (ch != begin) {
      if (*ch != '_') {
        return false;
      }
      ++ch;
    }
    bool empty = true;
    while (isupper(*ch) || isdigit(*ch)) {
      empty = false;
      ++ch;
    }
    if (empty) {
      return false;
    }
  }
  return true;
}
std::string ConstantizeVariableName(const std::string& name) {
  std::string result = "kTfLiteBuiltin";
  bool uppercase = true;
  for (char input_char : name) {
    if (input_char == '_') {
      uppercase = true;
    } else if (uppercase) {
      result += toupper(input_char);
      uppercase = false;
    } else {
      result += tolower(input_char);
    }
  }
  return result;
}
bool GenerateHeader(std::ostream& os) {
  auto enum_names = tflite::EnumNamesBuiltinOperator();
  for (auto enum_value : EnumValuesBuiltinOperator()) {
    auto enum_name = enum_names[enum_value];
    if (!IsValidInputEnumName(enum_name)) {
      std::cerr << "Invalid input enum name: " << enum_name << std::endl;
      return false;
    }
  }
  os << kFileHeader;
  for (auto enum_value : EnumValuesBuiltinOperator()) {
    auto enum_name = enum_names[enum_value];
    os << "  ";
    os << ConstantizeVariableName(enum_name);
    os << " = ";
    os << enum_value;
    os << ",\n";
  }
  os << kFileFooter;
  return true;
}
}  
}  