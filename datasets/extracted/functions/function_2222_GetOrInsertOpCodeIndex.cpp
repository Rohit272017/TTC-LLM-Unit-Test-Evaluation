#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
namespace mlir {
namespace lite {
namespace toco_legacy {
using std::string;
using tflite::BuiltinOperator;
using tflite::BuiltinOperator_DEQUANTIZE;
using tflite::ModelT;
using tflite::OperatorCodeT;
using tflite::OperatorT;
using tflite::TensorT;
using tflite::TensorType;
int32_t GetOrInsertOpCodeIndex(ModelT* model, const BuiltinOperator& op_code,
                               int32_t version) {
  for (size_t i = 0; i < model->operator_codes.size(); ++i) {
    if (tflite::GetBuiltinCode(model->operator_codes[i].get()) == op_code) {
      return i;
    }
  }
  model->operator_codes.push_back(std::make_unique<OperatorCodeT>());
  int op_code_idx = model->operator_codes.size() - 1;
  model->operator_codes[op_code_idx]->builtin_code = op_code;
  model->operator_codes[op_code_idx]->deprecated_builtin_code =
      tflite::ConvertBuiltinCodeToDeprecatedBuiltinCode(op_code);
  model->operator_codes[op_code_idx]->version = version;
  return op_code_idx;
}
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output) {
  OperatorT* op_raw = new OperatorT;
  op_raw->opcode_index =
      GetOrInsertOpCodeIndex(model, BuiltinOperator_DEQUANTIZE, 2);
  op_raw->inputs = {input};
  op_raw->outputs = {output};
  op->reset(op_raw);
}
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                const std::vector<int32_t>& shape_signature,
                const TensorType& type, std::unique_ptr<TensorT>* tensor) {
  TensorT* tensor_raw = new TensorT;
  tensor_raw->name = name;
  tensor_raw->shape = shape;
  if (!shape_signature.empty()) {
    tensor_raw->shape_signature = shape_signature;
  }
  tensor_raw->type = type;
  tensor->reset(tensor_raw);
}
bool HasMinMax(const TensorT* tensor) {
  return tensor->quantization && !tensor->quantization->min.empty() &&
         !tensor->quantization->max.empty();
}
}  
}  
}  