#include "tensorflow/core/common_runtime/device/device_utils.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
namespace tensorflow {
namespace device_utils {
Status ValidateDeviceType(StringPiece type) {
  static const LazyRE2 kTfDeviceTypeRegEx = {"[A-Z][A-Z_]*"};
  bool matches = RE2::FullMatch(type, *kTfDeviceTypeRegEx);
  if (!matches) {
    return Status(absl::StatusCode::kFailedPrecondition,
                  strings::StrCat("Device name/type '", type, "' must match ",
                                  kTfDeviceTypeRegEx->pattern(), "."));
  }
  return absl::OkStatus();
}
}  
}  