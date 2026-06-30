#include "tensorflow/lite/delegates/gpu/common/transformations/add_quant_adjustments.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
namespace tflite {
namespace gpu {
class AddQuantAdjustments : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type ==
        ToString(OperationType::QUANTIZE_AND_DEQUANTIZE)) {
      return {TransformStatus::SKIPPED, ""};
    }
    bool transform_applied = false;
    auto node_outputs = graph->FindOutputs(node->id);
    for (auto output_value : node_outputs) {
      if (!output_value->quant_params) continue;
      auto consumers = graph->FindConsumers(output_value->id);
      if (consumers.empty()) {
        continue;
      }
      Node* quant_and_dequant_node;
      absl::Status status =
          graph->InsertNodeAfter(node->id, &quant_and_dequant_node);
      if (!status.ok()) {
        return {TransformStatus::INVALID, "Could not insert new node."};
      }
      quant_and_dequant_node->operation.type =
          ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
      QuantizeAndDequantizeAttributes attr;
      attr.min = output_value->quant_params.value().min;
      attr.max = output_value->quant_params.value().max;
      attr.scale = output_value->quant_params.value().scale;
      quant_and_dequant_node->operation.attributes = attr;
      Value* adjusted_value = graph->NewValue();
      adjusted_value->tensor = output_value->tensor;
      status =
          graph->SetProducer(quant_and_dequant_node->id, adjusted_value->id);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                "Could not create QuantizeAndDequantize node."};
      }
      for (auto& consumer : consumers) {
        status = graph->ReplaceInput(consumer->id, output_value->id,
                                     adjusted_value->id);
        if (!status.ok()) {
          return {TransformStatus::INVALID,
                  absl::StrCat(
                      "Failed to associate quant-adjusted value for consumer: ",
                      status.message())};
        }
      }
      status = graph->AddConsumer(quant_and_dequant_node->id, output_value->id);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                absl::StrCat(
                    "Could not associate output to QuantizeAndDequantize: ",
                    status.message())};
      }
      output_value->quant_params.reset();
      transform_applied = true;
    }
    if (transform_applied) {
      return {TransformStatus::APPLIED, ""};
    }
    return {TransformStatus::SKIPPED, ""};
  }
};
std::unique_ptr<NodeTransformation> NewAddQuantAdjustments() {
  return std::make_unique<AddQuantAdjustments>();
}
}  
}  