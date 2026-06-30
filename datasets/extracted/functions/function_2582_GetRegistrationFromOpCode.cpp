#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace tflite {
TfLiteStatus GetRegistrationFromOpCode(
    const OperatorCode* opcode, const OpResolver& op_resolver,
    ErrorReporter* error_reporter, const TfLiteRegistration** registration) {
  TfLiteStatus status = kTfLiteOk;
  *registration = nullptr;
  auto builtin_code = GetBuiltinCode(opcode);
  int version = opcode->version();
  if (builtin_code > BuiltinOperator_MAX) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Op builtin_code out of range: %d. Are you using old TFLite binary "
        "with newer model?",
        builtin_code);
    status = kTfLiteError;
  } else if (builtin_code != BuiltinOperator_CUSTOM) {
    *registration = op_resolver.FindOp(builtin_code, version);
    if (*registration == nullptr) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Didn't find op for builtin opcode '%s' version '%d'. "
          "An older version of this builtin might be supported. "
          "Are you using an old TFLite binary with a newer model?\n",
          EnumNameBuiltinOperator(builtin_code), version);
      status = kTfLiteError;
    }
  } else if (!opcode->custom_code()) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Operator with CUSTOM builtin_code has no custom_code.\n");
    status = kTfLiteError;
  } else {
    const char* name = opcode->custom_code()->c_str();
    *registration = op_resolver.FindOp(name, version);
    if (*registration == nullptr) {
      status = kTfLiteError;
    }
  }
  return status;
}
}  