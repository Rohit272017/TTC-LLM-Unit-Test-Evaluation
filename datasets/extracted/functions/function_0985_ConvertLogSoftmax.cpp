#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
namespace tensorflow {
namespace tensorrt {
namespace convert {
class ConvertLogSoftmax : public OpConverterBase<ConvertLogSoftmax> {
 public:
  explicit ConvertLogSoftmax(const OpConverterParams *params)
      : OpConverterBase<ConvertLogSoftmax>(params) {}
  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("logits", TrtInputArg::kTensor)};
  }
  Status Validate() {
    const auto &params = *this->params_;
    const auto &inputs = params.inputs;
    ITensorProxyPtr logits_tensor = inputs.at(0).tensor();
    const int num_trt_dims = logits_tensor->getDimensions().nbDims;
    if (!num_trt_dims && params.use_implicit_batch) {
      return errors::InvalidArgument(
          "TensorRT LogSoftmax cannot apply on the batch dimension");
    }
    return OkStatus();
  }
  Status Convert() {
    const auto &params = *this->params_;
    const auto &inputs = params.inputs;
    const auto &node_def = params.node_def;
    ITensorProxyPtr logits_tensor = inputs.at(0).tensor();
    const int num_trt_dims = logits_tensor->getDimensions().nbDims;
    nvinfer1::IUnaryLayer *exp = params.converter->network()->addUnary(
        *logits_tensor->trt_tensor(), nvinfer1::UnaryOperation::kEXP);
    TFTRT_RETURN_ERROR_IF_NULLPTR(exp, node_def.name());
    params.converter->SetLayerName(exp, node_def, "exp");
    nvinfer1::IReduceLayer *reduced_sum =
        params.converter->network()->addReduce(
            *exp->getOutput(0), nvinfer1::ReduceOperation::kSUM,
            (1 << (num_trt_dims - 1)), 
            true );
    params.converter->SetLayerName(reduced_sum, node_def, "reduced_sum");
    nvinfer1::IUnaryLayer *log_reduced_sum =
        params.converter->network()->addUnary(*reduced_sum->getOutput(0),
                                              nvinfer1::UnaryOperation::kLOG);
    TFTRT_RETURN_ERROR_IF_NULLPTR(log_reduced_sum, node_def.name());
    params.converter->SetLayerName(log_reduced_sum, node_def,
                                   "log_reduced_sum");
    nvinfer1::IElementWiseLayer *sub =
        params.converter->network()->addElementWise(
            *logits_tensor->trt_tensor(), *log_reduced_sum->getOutput(0),
            nvinfer1::ElementWiseOperation::kSUB);
    TFTRT_RETURN_ERROR_IF_NULLPTR(sub, node_def.name());
    params.converter->SetLayerName(sub, node_def, "sub");
    params.outputs->push_back(TRT_TensorOrWeights(sub->getOutput(0)));
    return OkStatus();
  }
};
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLogSoftmax>(),
                                  "LogSoftmax");
}  
}  
}  
#endif  