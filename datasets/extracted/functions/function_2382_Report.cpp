#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include <cstdarg>
namespace tflite {
int ErrorReporter::Report(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}
int ErrorReporter::ReportError(void*, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}
}  