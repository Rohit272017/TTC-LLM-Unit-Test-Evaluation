#include "tensorflow/lite/stderr_reporter.h"
#include <stdarg.h>
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
namespace tflite {
int StderrReporter::Report(const char* format, va_list args) {
  logging_internal::MinimalLogger::LogFormatted(TFLITE_LOG_ERROR, format, args);
  return 0;
}
ErrorReporter* DefaultErrorReporter() {
  static StderrReporter* error_reporter = new StderrReporter;
  return error_reporter;
}
}  