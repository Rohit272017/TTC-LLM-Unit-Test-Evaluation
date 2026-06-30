#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_REQUIRES_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_REQUIRES_H_
#include <utility>
#include "tensorflow/core/platform/macros.h"
namespace tensorflow {
#define OP_REQUIRES(CTX, EXP, STATUS)                     \
  do {                                                    \
    if (!TF_PREDICT_TRUE(EXP)) {                          \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return;                                             \
    }                                                     \
  } while (0)
#define OP_REQUIRES_OK(CTX, ...)                                        \
  do {                                                                  \
    if (!TF_PREDICT_TRUE(                                               \
            ::tensorflow::op_requires_internal::OkImpl<::absl::Status>( \
                (CTX), __FILE__, __LINE__,                              \
                static_cast<const ::absl::Status&>(__VA_ARGS__)))) {    \
      return;                                                           \
    }                                                                   \
  } while (0)
#define OP_REQUIRES_OK_OR_SET_PAYLOAD(CTX, PAYLOAD_KEY, PAYLOAD_VALUE, STATUS) \
  do {                                                                         \
    if (!TF_PREDICT_TRUE(STATUS.ok())) {                                       \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC");                   \
      if (!PAYLOAD_VALUE.empty()) {                                            \
        STATUS.SetPayload(PAYLOAD_KEY, absl::Cord(PAYLOAD_VALUE));             \
      }                                                                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, STATUS);                \
      return;                                                                  \
    }                                                                          \
  } while (0)
#define OP_REQUIRES_ASYNC(CTX, EXP, STATUS, CALLBACK)  \
  do {                                                 \
    if (!TF_PREDICT_TRUE(EXP)) {                       \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS)); \
      (CALLBACK)();                                    \
      return;                                          \
    }                                                  \
  } while (0)
#define OP_REQUIRES_OK_ASYNC(CTX, STATUS, CALLBACK)                          \
  do {                                                                       \
    if (!TF_PREDICT_TRUE(                                                    \
            ::tensorflow::op_requires_internal::OkAsyncImpl<::absl::Status>( \
                (CTX), __FILE__, __LINE__, (STATUS)))) {                     \
      (CALLBACK)();                                                          \
      return;                                                                \
    }                                                                        \
  } while (0)
#define OP_REQUIRES_VALUE(lhs, ctx, rexpr)                                   \
  OP_REQUIRES_VALUE_IMPL(                                                    \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, ctx, \
      rexpr)
#define OP_REQUIRES_VALUE_IMPL(statusor, lhs, ctx, rexpr) \
  auto statusor = (rexpr);                                \
  OP_REQUIRES_OK(ctx, statusor.status());                 \
  lhs = std::move(statusor.value())
namespace op_requires_internal {
template <typename S, typename Ctx>
bool OkImpl(Ctx&& ctx, const char* file, int line, const S& s) {
  if (!TF_PREDICT_TRUE(s.ok())) {
    CheckNotInComputeAsync(ctx, "OP_REQUIRES_OK_ASYNC");
    ctx->CtxFailureWithWarning(file, line, s);
    return false;
  } else {
    return true;
  }
}
template <typename S, typename Ctx>
bool OkAsyncImpl(Ctx&& ctx, const char* file, int line, const S& s) {
  if (!TF_PREDICT_TRUE(s.ok())) {
    ctx->CtxFailureWithWarning(file, line, s);
    return false;
  } else {
    return true;
  }
}
}  
}  
#endif  