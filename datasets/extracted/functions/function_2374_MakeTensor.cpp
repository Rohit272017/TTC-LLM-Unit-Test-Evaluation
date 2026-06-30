#include "tensorflow/lite/tools/optimize/quantization_wrapper_utils.h"
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include "tensorflow/compiler/mlir/lite/tools/optimize/operator_property.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace tflite {
namespace impl {
class FlatBufferModel;
}
namespace optimize {
namespace {
#ifdef TFLITE_CUSTOM_LSTM
constexpr bool kUseCustomLSTM = true;
#else
constexpr bool kUseCustomLSTM = false;
#endif
void MakeTensor(const string& name, std::unique_ptr<TensorT>* tensor) {
  TensorT* tensor_raw = new TensorT;
  tensor_raw->name = name;
  tensor_raw->shape = {0};
  tensor_raw->type = TensorType_FLOAT32;
  tensor->reset(tensor_raw);
}
string CreateTensorName(int op_index, int tensor_index) {
  return "intermediate_" + std::to_string(op_index) + "_" +
         std::to_string(tensor_index);
}
bool IntermediateTensorExists(ModelT* model) {
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       ++subgraph_idx) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      if (!op->intermediates.empty()) {
        return true;
      }
    }
  }
  return false;
}
}  
TfLiteStatus LoadModel(const string& path, ModelT* model) {
  auto input_model = impl::FlatBufferModel::BuildFromFile(path.c_str());
  if (!input_model) {
    return kTfLiteError;
  }
  auto readonly_model = input_model->GetModel();
  if (!readonly_model) {
    return kTfLiteError;
  }
  readonly_model->UnPackTo(model);
  return kTfLiteOk;
}
TfLiteStatus AddIntermediateTensorsToFusedOp(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model) {
  if (model->subgraphs.size() == 1 && model->subgraphs[0]->operators.empty()) {
    return kTfLiteOk;
  }
  if (IntermediateTensorExists(model)) {
    return kTfLiteOk;
  }
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       ++subgraph_idx) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      if (property.intermediates.empty()) {
        continue;
      }
      const int next_tensor_index = subgraph->tensors.size();
      int num_intermediates = property.intermediates.size();
      if (kUseCustomLSTM) {
        num_intermediates = 12;
      }
      for (int i = 0; i < num_intermediates; ++i) {
        std::unique_ptr<TensorT> intermediate_tensor;
        auto name = CreateTensorName(op_idx, i);
        MakeTensor(name, &intermediate_tensor);
        subgraph->tensors.push_back(std::move(intermediate_tensor));
        op->intermediates.push_back(next_tensor_index + i);
      }
    }
  }
  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model);
  FinishModelBuffer(*builder, output_model_location);
  return kTfLiteOk;
}
bool WriteFile(const std::string& out_file, const uint8_t* bytes,
               size_t num_bytes) {
  std::fstream stream(out_file, std::ios::binary | std::ios::out);
  for (size_t i = 0; i < num_bytes; i++) {
    stream << bytes[i];
  }
  return (!stream.bad() && !stream.fail());
}
}  
}  