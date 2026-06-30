#include "xla/status_macros.h"
#include <algorithm>
#include <string>
#include "absl/base/attributes.h"
#include "absl/base/log_severity.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/status.h"
namespace xla {
namespace status_macros {
ABSL_CONST_INIT const char kPossibleAutoJitAlternative[] =
    "This error might be occurring with the use of xla.compile. If it is not "
    "necessary that every Op be compiled with XLA, an alternative is to use "
    "auto_jit with OptimizerOptions.global_jit_level = ON_2 or the environment "
    "variable TF_XLA_FLAGS=\"tf_xla_auto_jit=2\" which will attempt to use xla "
    "to compile as much of the graph as the compiler is able to.";
static void LogError(const absl::Status& status, const char* filename, int line,
                     absl::LogSeverity log_severity,
                     bool should_log_stack_trace) {
  std::string stack_trace;
  if (should_log_stack_trace) {
    stack_trace = absl::StrCat("\n", tsl::CurrentStackTrace());
  }
  switch (log_severity) {
    case absl::LogSeverity::kInfo:
      LOG(INFO) << status << stack_trace;
      break;
    case absl::LogSeverity::kWarning:
      LOG(WARNING) << status << stack_trace;
      break;
    case absl::LogSeverity::kError:
      LOG(ERROR) << status << stack_trace;
      break;
    case absl::LogSeverity::kFatal:
      LOG(FATAL) << status << stack_trace;
      break;
    default:
      LOG(FATAL) << "Unknown LOG severity " << log_severity;
  }
}
static absl::Status MakeError(const char* filename, int line,
                              absl::StatusCode code, const std::string& message,
                              bool should_log, absl::LogSeverity log_severity,
                              bool should_log_stack_trace) {
  if (ABSL_PREDICT_FALSE(code == absl::StatusCode::kOk)) {
    LOG(ERROR) << "Cannot create error with status OK";
    code = absl::StatusCode::kUnknown;
  }
  const absl::Status status = absl::Status(code, message);
  if (ABSL_PREDICT_TRUE(should_log)) {
    LogError(status, filename, line, log_severity, should_log_stack_trace);
  }
  return status;
}
MakeErrorStream::MakeErrorStreamWithOutput&
MakeErrorStream::add_ret_check_failure(const char* condition) {
  return *this << "RET_CHECK failure (" << impl_->file_ << ":" << impl_->line_
               << ") " << condition << " ";
}
void MakeErrorStream::CheckNotDone() const { impl_->CheckNotDone(); }
MakeErrorStream::Impl::Impl(const char* file, int line, tsl::error::Code code,
                            MakeErrorStream* error_stream,
                            bool is_logged_by_default)
    : file_(file),
      line_(line),
      code_(static_cast<absl::StatusCode>(code)),
      is_done_(false),
      should_log_(is_logged_by_default),
      log_severity_(absl::LogSeverity::kError),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {}
MakeErrorStream::Impl::Impl(const absl::Status& status,
                            PriorMessageHandling prior_message_handling,
                            const char* file, int line,
                            MakeErrorStream* error_stream)
    : file_(file),
      line_(line),
      code_(!status.ok() ? static_cast<absl::StatusCode>(status.code())
                         : absl::StatusCode::kUnknown),
      prior_message_handling_(prior_message_handling),
      prior_message_(status.message()),
      is_done_(false),
      should_log_(true),
      log_severity_(absl::LogSeverity::kError),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {
  DCHECK(!status.ok()) << "Attempted to append/prepend error text to status OK";
}
MakeErrorStream::Impl::~Impl() {
  if (!is_done_) {
    LOG(ERROR) << "MakeErrorStream destructed without getting absl::Status: "
               << file_ << ":" << line_ << " " << stream_.str();
  }
}
absl::Status MakeErrorStream::Impl::GetStatus() {
  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream got absl::Status more than once: " << file_
               << ":" << line_ << " " << stream_.str();
  }
  is_done_ = true;
  const std::string& stream_str = stream_.str();
  const std::string str = prior_message_handling_ == kAppendToPriorMessage
                              ? absl::StrCat(prior_message_, stream_str)
                              : absl::StrCat(stream_str, prior_message_);
  if (ABSL_PREDICT_FALSE(str.empty())) {
    return MakeError(
        file_, line_, code_,
        absl::StrCat(str, "Error without message at ", file_, ":", line_),
        true, absl::LogSeverity::kError,
        should_log_stack_trace_);
  } else {
    return MakeError(file_, line_, code_, str, should_log_, log_severity_,
                     should_log_stack_trace_);
  }
}
void MakeErrorStream::Impl::CheckNotDone() const {
  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream shift called after getting absl::Status: "
               << file_ << ":" << line_ << " " << stream_.str();
  }
}
}  
}  