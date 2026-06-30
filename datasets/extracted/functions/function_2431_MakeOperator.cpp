#include "tensorflow/lite/c/c_api_opaque_internal.h"
#include <memory>
#include <unordered_map>
#include <utility>
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/c/operator.h"
#include "tensorflow/lite/core/subgraph.h"
namespace tflite {
namespace internal {
namespace {
TfLiteOperator* MakeOperator(const TfLiteRegistration* registration,
                             int node_index) {
  auto* registration_external = TfLiteOperatorCreate(
      static_cast<TfLiteBuiltinOperator>(registration->builtin_code),
      registration->custom_name, registration->version,
      nullptr);
  registration_external->node_index = node_index;
  return registration_external;
}
}  
TfLiteOperator* CommonOpaqueConversionUtil::CachedObtainOperator(
    OperatorsCache* registration_externals_cache,
    const TfLiteRegistration* registration, int node_index) {
  OpResolver::OpId op_id{registration->builtin_code, registration->custom_name,
                         registration->version};
  auto it = registration_externals_cache->find(op_id);
  if (it != registration_externals_cache->end()) {
    return it->second.get();
  }
  auto* registration_external = MakeOperator(registration, node_index);
  registration_externals_cache->insert(
      it, std::make_pair(op_id, registration_external));
  return registration_external;
}
TfLiteOperator* CommonOpaqueConversionUtil::ObtainOperator(
    TfLiteContext* context, const TfLiteRegistration* registration,
    int node_index) {
  auto* subgraph = static_cast<tflite::Subgraph*>(context->impl_);
  if (!subgraph->registration_externals_) {
    subgraph->registration_externals_ = std::make_shared<OperatorsCache>();
  }
  return CachedObtainOperator(subgraph->registration_externals_.get(),
                              registration, node_index);
}
}  
}  