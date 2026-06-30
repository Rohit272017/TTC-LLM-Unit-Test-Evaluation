#include "tensorflow/lite/delegates/gpu/common/transformations/make_fully_connected.h"
#include <any>
#include <memory>
#include <string>
#include <vector>
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
namespace tflite {
namespace gpu {
namespace {
bool IsConvEquivalentToFullyConnected(const Convolution2DAttributes& attr) {
  return attr.weights.shape.w == 1 &&           
         attr.weights.shape.h == 1 &&           
         attr.strides == HW(1, 1) &&            
         attr.dilations == HW(1, 1) &&          
         attr.padding.prepended == HW(0, 0) &&  
         attr.padding.appended == HW(0, 0);
}
class MakeFullyConnectedFromConvolution : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::CONVOLUTION_2D)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto inputs = graph->FindInputs(node->id);
    if (inputs.size() != 1) {
      return {TransformStatus::SKIPPED, ""};
    }
    const auto& input_shape = inputs[0]->tensor.shape;
    if (input_shape.w != 1 || input_shape.h != 1) {
      return {TransformStatus::SKIPPED, ""};
    }
    const auto& conv_attr = std::any_cast<const Convolution2DAttributes&>(
        node->operation.attributes);
    if (!IsConvEquivalentToFullyConnected(conv_attr)) {
      return {TransformStatus::SKIPPED, ""};
    }
    FullyConnectedAttributes fc_attr;
    fc_attr.weights = conv_attr.weights;
    fc_attr.bias = conv_attr.bias;
    node->operation.attributes = fc_attr;
    node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    return {TransformStatus::APPLIED,
            "Replaced convolution with fully connected."};
  }
};
}  
std::unique_ptr<NodeTransformation> NewMakeFullyConnectedFromConvolution() {
  return std::make_unique<MakeFullyConnectedFromConvolution>();
}
}  
}  