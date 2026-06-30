#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUSOR_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUSOR_H_
#include "absl/status/statusor.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#define TF_ASSIGN_OR_RETURN(lhs, rexpr) \
  TF_ASSIGN_OR_RETURN_IMPL(             \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)
#define TF_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                             \
  if (TF_PREDICT_FALSE(!statusor.ok())) {              \
    return statusor.status();                          \
  }                                                    \
  lhs = std::move(statusor).value()
#endif  