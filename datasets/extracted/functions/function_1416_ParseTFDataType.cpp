#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "absl/strings/numbers.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/util/padding.h"
#include "tfrt/core_runtime/op_attr_type.h"  
#include "tfrt/core_runtime/op_attrs.h"  
#include "tfrt/host_context/kernel_utils.h"  
namespace tensorflow {
DataType ParseTFDataType(StringPiece dtype) {
  if (dtype == "DT_INT8") {
    return DataType::DT_INT8;
  } else if (dtype == "DT_INT32") {
    return DataType::DT_INT32;
  } else if (dtype == "DT_INT64") {
    return DataType::DT_INT64;
  } else if (dtype == "DT_FLOAT") {
    return DataType::DT_FLOAT;
  } else if (dtype == "DT_DOUBLE") {
    return DataType::DT_DOUBLE;
  } else {
    assert(false && "Unsupported dtype");
    abort();
  }
}
bool ParseBoolAttrValue(StringPiece attr_value) {
  if (attr_value == "false") {
    return false;
  } else if (attr_value == "true") {
    return true;
  } else {
    assert(false && "Bool attribute value invalid");
    abort();
  }
}
Status ParseValue(StringPiece input, bool* value) {
  *value = ParseBoolAttrValue(input);
  return absl::OkStatus();
}
Status ParseValue(StringPiece input, int32* value) {
  bool parse_result = absl::SimpleAtoi(input, value);
  if (!parse_result) {
    return errors::InvalidArgument("Could not parse int32 from ", input);
  }
  return absl::OkStatus();
}
Status ParseValue(StringPiece input, DataType* value) {
  *value = ParseTFDataType(input);
  return absl::OkStatus();
}
Status ParseValue(StringPiece input, std::string* value) {
  *value = std::string(input);
  return absl::OkStatus();
}
Status ParseValue(StringPiece input, std::vector<int32>* value) {
  std::vector<std::string> parts = str_util::Split(input, ",");
  value->reserve(parts.size());
  for (const auto& value_str : parts) {
    int32_t value_int;
    bool parse_result = absl::SimpleAtoi(value_str, &value_int);
    if (!parse_result) {
      return errors::InvalidArgument("Could not parse list of integers from ",
                                     input);
    }
    value->push_back(value_int);
  }
  return absl::OkStatus();
}
Status ParseValue(StringPiece input, Padding* value) {
  return GetPaddingFromString(input, value);
}
Status AddOpAttr(const std::string& name, const std::string& attr_value,
                 tfrt::OpAttrs* opattrs) {
  Status s;
  std::vector<absl::string_view> value_split = tfd::AttrValueSplit(attr_value);
  auto& type = value_split[0];
  auto& value = value_split[1];
  if (type == "bool") {
    bool val;
    s = ParseValue(value, &val);
    opattrs->Set<bool>(name, val);
  } else if (type == "i32") {
    int32_t val;
    s = ParseValue(value, &val);
    opattrs->Set<int32>(name, val);
  } else if (type == "string" || type == "padding") {
    std::string val;
    s = ParseValue(value, &val);
    opattrs->SetString(name, val);
  } else if (type == "tfdtype") {
    DataType val;
    s = ParseValue(value, &val);
    opattrs->Set<tfrt::OpAttrType>(name, tfd::ConvertFromTfDataType(val));
  } else if (type == "list(i32)") {
    std::vector<int32> val;
    s = ParseValue(value, &val);
    opattrs->SetArray<int32>(name, val);
  }
  return s;
}
Status FillOpAttrs(tfrt::RemainingAttributes attrs, tfrt::OpAttrs* opattrs) {
  int num_tf_attrs = attrs.size() / 2;
  Status status;
  for (int i = 0; i < num_tf_attrs; ++i) {
    std::string name = attrs.GetStringAttribute(i * 2).str();
    std::string attr_value = attrs.GetStringAttribute(i * 2 + 1).str();
    Status s = AddOpAttr(name, attr_value, opattrs);
    status.Update(s);
  }
  return status;
}
}  