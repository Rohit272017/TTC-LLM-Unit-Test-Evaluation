#include "tensorflow/lite/mutable_op_resolver_utils.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace tflite {
void AddOp(MutableOpResolver* mutable_op_resolver, const TfLiteOperator* op,
           int min_version, int max_version) {
  TfLiteRegistration registration{};
  registration.builtin_code = TfLiteOperatorGetBuiltInCode(op);
  registration.custom_name = TfLiteOperatorGetCustomName(op);
  registration.version = TfLiteOperatorGetVersion(op);
  registration.registration_external = const_cast<TfLiteOperator*>(op);
  if (registration.custom_name != nullptr) {
    mutable_op_resolver->AddCustom(registration.custom_name, &registration,
                                   min_version, max_version);
  } else {
    mutable_op_resolver->AddBuiltin(BuiltinOperator(registration.builtin_code),
                                    &registration, min_version, max_version);
  }
}
void AddOp(MutableOpResolver* mutable_op_resolver, const TfLiteOperator* op) {
  int version = TfLiteOperatorGetVersion(op);
  AddOp(mutable_op_resolver, op, version, version);
}
}  