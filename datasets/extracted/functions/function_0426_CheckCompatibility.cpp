#include "tensorflow/lite/tools/delegates/compatibility/nnapi/nnapi_compatibility_lib.h"
#include <map>
#include <utility>
#include <vector>
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
namespace tflite {
namespace tools {
using ::tflite::delegate::nnapi::NNAPIValidationFailure;
TfLiteStatus CheckCompatibility(
    TfLiteContext* context, int32_t runtime_feature_level,
    std::vector<int>* supported_nodes,
    std::map<int, std::vector<NNAPIValidationFailure>>* failures_by_node) {
  if (!context) {
    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_ERROR, "Context is nullptr.");
    return kTfLiteError;
  }
  TfLiteIntArray* execution_plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
  for (int node_index : TfLiteIntArrayView(execution_plan)) {
    TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Node index: %d", node_index);
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    std::vector<delegate::nnapi::NNAPIValidationFailure> map_failures;
    if (NNAPIDelegateKernel::Validate(
            context, registration, runtime_feature_level, node,
             true,
             nullptr, &map_failures)) {
      TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Built-in Code: %d",
                      registration->builtin_code);
      if (supported_nodes) {
        supported_nodes->push_back(node_index);
      }
    } else {
      if (failures_by_node) {
        (*failures_by_node)[node_index] = std::move(map_failures);
      }
    }
  }
  return kTfLiteOk;
}
}  
}  